import asyncio
import json
import math
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("chromadb")  # the rag extra: pip install -e ".[rag]"

from evalkit.openai_agent import OpenAICompatibleAgent
from evalkit.rag import (
    MAX_RECORDED_SEARCH,
    MAX_TOP_K,
    DocumentIndex,
    hash_embed,
    load_corpus,
    rag_tools,
    retrieval_config,
)
from evalkit.models import EvaluationCase, Score, Span, Trial
from evalkit.report import retrieval_diagnosis
from evalkit.retry import RetryPolicy
from evalkit.run import _real_agent_config
from evalkit.runner import run_experiment
from evalkit.sample_data import DATASETS_DIR, dataset_by_name, rag_cases
from evalkit.scorers import retrieval_recall, retrieved_doc_ids, score_trials
from evalkit.tools import ToolRegistry
from evalkit.tracer import Tracer
from scripts.build_rag_dataset import cases as build_cases
from scripts.build_rag_dataset import corpus as build_corpus
from scripts.build_rag_dataset import to_jsonl

corpus = load_corpus()
documents = {document.id: document for document in corpus}
cases = rag_cases()

@pytest.fixture(scope="module")
def registry():
    # Word matching, so these tests need no model download and give the same results every time.
    return ToolRegistry(rag_tools(top_k=4, embedding="hashing"))

# --- The dataset itself ---

def test_jsonl_files_match_the_builder():
    # Both files are generated; if this fails, run: python scripts/build_rag_dataset.py
    built_corpus = build_corpus()
    assert (DATASETS_DIR / "rag_corpus.jsonl").read_text(encoding="utf-8") == to_jsonl(built_corpus)
    assert (DATASETS_DIR / "rag.jsonl").read_text(encoding="utf-8") == to_jsonl(build_cases(built_corpus))

def test_every_category_has_several_cases():
    counts = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    assert set(counts) == {"single_doc", "distractor", "outdated", "multi_doc", "unanswerable", "doc_injection"}
    assert min(counts.values()) >= 4, counts

def test_ids_are_unique_and_every_relevant_doc_exists():
    assert len({case.id for case in cases}) == len(cases)
    assert len(documents) == len(corpus)
    for case in cases:
        assert case.relevant_docs is not None, case.input
        assert set(case.relevant_docs) <= set(documents), case.input
        assert case.required_tools == ["search_docs"]

def test_only_unanswerable_cases_have_no_relevant_docs():
    for case in cases:
        assert (case.relevant_docs == []) == (case.category == "unanswerable"), case.input
        assert (case.expected_output == "unknown") == (case.category == "unanswerable"), case.input

def _numbers(text: str) -> set[float]:
    return {float(match.replace(",", "")) for match in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}

@pytest.mark.parametrize("case", [case for case in cases if case.category not in ("multi_doc", "unanswerable")],
                         ids=lambda case: f"case-{case.id}")
def test_stated_answers_appear_in_their_documents(case):
    # Guards against a typo in a question's answer: anything not worked out from several
    # documents must be written in one of the relevant ones.
    texts = [documents[doc_id].text for doc_id in case.relevant_docs]
    try:
        expected = float(case.expected_output)
    except ValueError:
        assert any(case.expected_output.casefold() in text.casefold() for text in texts), case.input
    else:
        assert any(expected in _numbers(text) for text in texts), case.input

def test_injected_text_is_in_the_documents_it_is_tested_with():
    injected = [case for case in cases if case.must_not_contain]
    assert {case.category for case in injected} == {"doc_injection"}
    for case in injected:
        texts = " ".join(documents[doc_id].text for doc_id in case.relevant_docs)
        assert all(forbidden in texts for forbidden in case.must_not_contain), case.input

def test_superseded_documents_disagree_with_the_current_ones():
    # The outdated category only tests something if the old document gives a different answer.
    assert "2 days" in documents["pol-001"].text and "3 days" in documents["pol-002"].text
    assert "180 USD" in documents["pol-004"].text and "210 USD" in documents["pol-005"].text
    assert "520 USD" in documents["prod-011"].text and "560 USD" in documents["prod-012"].text

def test_dataset_by_name_finds_rag():
    assert dataset_by_name("rag") == cases

# --- Search ---

def test_hash_embedding_is_deterministic_and_unit_length():
    vector = hash_embed("Osprey centrifuge noise level")
    assert vector == hash_embed("osprey CENTRIFUGE noise level")
    assert math.isclose(sum(value * value for value in vector), 1.0)
    assert hash_embed("") == [0.0] * len(vector)

def test_search_returns_k_documents_best_match_first():
    index = DocumentIndex(corpus, embedding="hashing")

    found = index.search("Kestrel calibration kit part number", k=3)

    assert len(found) == 3
    assert found[0]["doc_id"] == "prod-005"
    assert set(found[0]) == {"doc_id", "title", "date", "text"}
    assert len(index.search("anything", k=100)) == len(corpus)  # k larger than the corpus

def test_search_results_are_recorded_whole():
    tracer = Tracer(1)
    widest = ToolRegistry(rag_tools(top_k=MAX_TOP_K, embedding="hashing"))

    result = widest.call("search_docs", '{"query": "remote work policy days per week"}', tracer)

    span = tracer.spans[0]
    assert len(span.output) > 1000  # longer than other tools may record, and still not cut
    assert [document["doc_id"] for document in json.loads(span.output)] == [document["doc_id"] for document in result]
    assert {"pol-001", "pol-002"} <= {document["doc_id"] for document in result}

def test_the_largest_possible_result_fits_in_the_record():
    longest = sorted(corpus, key=lambda document: len(document.model_dump_json()), reverse=True)[:MAX_TOP_K]
    result = [{"doc_id": d.id, "title": d.title, "date": d.date, "text": d.text} for d in longest]
    assert len(json.dumps(result, ensure_ascii=False)) < MAX_RECORDED_SEARCH

@pytest.mark.parametrize("top_k", [0, MAX_TOP_K + 1])
def test_top_k_out_of_range_is_refused(top_k):
    with pytest.raises(ValueError):
        rag_tools(top_k=top_k, embedding="hashing")

def test_unknown_embedding_is_refused():
    with pytest.raises(ValueError):
        rag_tools(embedding="word2vec")

# --- Versions: a retrieval change must show up as a config change ---

def test_retrieval_settings_and_the_corpus_are_part_of_the_config():
    base = retrieval_config(4, "minilm")

    assert base != retrieval_config(1, "minilm")
    assert base != retrieval_config(4, "hashing")
    edited = [document.model_copy(update={"text": document.text + " Edited."}) if document.id == "pol-003"
              else document for document in corpus]
    assert base["corpus_version"] != retrieval_config(4, "minilm", edited)["corpus_version"]

def test_only_rag_configs_record_retrieval_settings():
    agent = SimpleNamespace(model="m", system_prompt="p", max_steps=5, temperature=0.0,
                            tools=ToolRegistry([]), retry_policy=RetryPolicy(), requests_per_minute=None,
                            tokens_per_minute=None)

    rag = _real_agent_config(agent, "groq", SimpleNamespace(dataset="rag", top_k=2, embedding="hashing"))
    hard = _real_agent_config(agent, "groq", SimpleNamespace(dataset="hard", top_k=2, embedding="hashing"))

    assert rag.params["retrieval"]["top_k"] == 2
    assert "retrieval" not in hard.params

# --- End to end: search, answer, score ---

def test_a_search_then_answer_is_scored_as_grounded(registry):
    from test_openai_agent import NO_WAIT, fake_client, final_answer, tool_call

    case = next(case for case in cases if case.relevant_docs == ["prod-005"] and case.expected_output == "6")
    client = fake_client(tool_call("search_docs", '{"query": "Kestrel calibration interval"}'), final_answer("6"))
    agent = OpenAICompatibleAgent(client=client, tools=registry, retry_policy=NO_WAIT)

    trials = asyncio.run(run_experiment(agent, [case], attempts=1))
    scores = {score.scorer: score.value for score in score_trials([case], trials)}

    assert scores["exact_match"] == 1.0
    assert scores["tool_selection"] == 1.0
    assert scores["tool_faithfulness"] == 1.0  # "6" is in the retrieved calibration guide
    assert scores["retrieval_recall"] == 1.0
    assert retrieval_diagnosis(score_trials([case], trials)) == {"correct": 1}

def test_an_answer_from_memory_is_not_grounded(registry):
    from test_openai_agent import NO_WAIT, fake_client, final_answer, tool_call

    case = next(case for case in cases if case.expected_output == "Daniel Okafor")
    # The search misses the leadership document, and the model names someone anyway.
    client = fake_client(tool_call("search_docs", '{"query": "pipette volume"}'), final_answer("Mara Lindqvist"))
    agent = OpenAICompatibleAgent(client=client, tools=registry, retry_policy=NO_WAIT)

    trials = asyncio.run(run_experiment(agent, [case], attempts=1))
    scores = {score.scorer: score.value for score in score_trials([case], trials)}

    assert scores["exact_match"] == 0.0
    assert scores["tool_faithfulness"] == 0.0
    assert scores["retrieval_recall"] == 0.0
    assert retrieval_diagnosis(score_trials([case], trials)) == {"retrieval_failure": 1}

# --- Retrieval scoring ---

def _search_span(doc_ids: list[str], status: str = "success", output: str | None = None) -> Span:
    results = [{"doc_id": doc_id, "title": "", "date": "", "text": ""} for doc_id in doc_ids]
    return Span(trial_id=1, kind="tool", name="search_docs", start=0, end=0, status=status,
                output=output if output is not None else json.dumps(results))

def _llm_span() -> Span:
    return Span(trial_id=1, kind="llm", name="model", start=0, end=0, status="success")

def _trial(*spans: Span, output: str = "x") -> Trial:
    return Trial(id=1, case_id=1, attempt=1, output=output, spans=[_llm_span(), *spans])

two_docs = EvaluationCase(id=1, input="q", expected_output="a", relevant_docs=["prod-011", "prod-012"])

def test_recall_is_the_share_of_relevant_documents_any_search_found():
    assert retrieval_recall(two_docs, _trial(_search_span(["prod-011", "org-001"]))).value == 0.5
    # Found across two searches still counts as found.
    assert retrieval_recall(two_docs, _trial(_search_span(["prod-011"]), _search_span(["prod-012"]))).value == 1.0

def test_not_searching_scores_zero_but_a_model_that_never_answered_is_not_judged():
    assert retrieval_recall(two_docs, _trial()).value == 0.0
    never_answered = Trial(id=1, case_id=1, attempt=1, spans=[
        Span(trial_id=1, kind="llm", name="model", start=0, end=0, status="error", error_type="rate_limit"),
    ])
    assert retrieval_recall(two_docs, never_answered) is None

def test_recall_is_not_judged_without_relevant_documents():
    assert retrieval_recall(two_docs.model_copy(update={"relevant_docs": []}), _trial()) is None  # unanswerable
    assert retrieval_recall(two_docs.model_copy(update={"relevant_docs": None}), _trial()) is None  # not rag

def test_failed_or_unreadable_searches_find_nothing():
    trial = _trial(_search_span(["prod-011"], status="error"), _search_span([], output='[{"doc_id": "prod-0'))
    assert retrieved_doc_ids(trial) == set()

def test_diagnosis_separates_retrieval_failures_from_answer_failures():
    def scores(trial_id: int, recall: float, correct: bool) -> list[Score]:
        return [Score(trial_id=trial_id, scorer="retrieval_recall", value=recall),
                Score(trial_id=trial_id, scorer="exact_match", value=1.0 if correct else 0.0)]

    all_scores = (scores(1, 1.0, True) + scores(2, 1.0, False) + scores(3, 0.5, False) + scores(4, 0.0, True)
                  + [Score(trial_id=5, scorer="exact_match", value=1.0)])  # no retrieval score: left out

    assert retrieval_diagnosis(all_scores) == {
        "correct": 1, "answer_failure": 1, "retrieval_failure": 1, "unsupported_correct": 1,
    }
