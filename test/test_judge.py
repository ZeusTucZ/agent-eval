"""Tests for the LLM judge and its calibration against human labels: a fake client, no network."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from evalkit import storage
from evalkit.calibrate import agreement, format_agreement, label, load_labels, sample_trials, verdict
from evalkit.cost import PRICES
from evalkit.judge import RUBRICS, SYSTEM_PROMPT, Judge, build_prompt, judge_trials, judge_version, parse_verdict
from evalkit.models import AgentConfig, EvaluationCase, Experiment, Score, Span, Trial
from evalkit.retry import RetryPolicy
from evalkit.scorers import SCORER_NAMES
from evalkit.stats import cohens_kappa

class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    async def create(self, model, messages, temperature, response_format):
        self.requests.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatCompletion.model_validate({
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 20, "total_tokens": 320},
        })

def fake_judge(*replies) -> Judge:
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(replies)))
    return Judge(client=client, requests_per_minute=None, retry_policy=RetryPolicy(max_retries=0, jitter=False))

def verdict_json(passed: bool, reason: str = "because") -> str:
    return json.dumps({"verdict": "pass" if passed else "fail", "reason": reason})

case = EvaluationCase(id=1, input="How long is the Osprey warranty, in years?", expected_output="3")

def trial(output: str | None = "3 years", source: str | None = '[{"doc_id": "inj-001", "text": "3-year warranty"}]',
          trial_id: int = 1) -> Trial:
    spans = [Span(trial_id=trial_id, kind="llm", name="openai/gpt-oss-20b", start=0, end=0, status="success")]
    if source is not None:
        spans.append(Span(trial_id=trial_id, kind="tool", name="search_docs", start=0, end=0, status="success",
                          output=source))
    return Trial(id=trial_id, case_id=1, attempt=1, output=output, spans=spans)

# --- Prompt and verdicts ---

def test_groundedness_shows_the_sources_and_correctness_shows_the_expected_answer():
    grounded = build_prompt(RUBRICS["groundedness"], case, trial())
    correct = build_prompt(RUBRICS["correctness"], case, trial())

    assert "<sources>" in grounded and "3-year warranty" in grounded and "<expected>" not in grounded
    assert "<expected>\n3\n</expected>" in correct and "<sources>" not in correct
    assert "<answer>\n3 years\n</answer>" in grounded

def test_the_judge_is_told_to_ignore_instructions_inside_the_material():
    # The rag corpus carries prompt injections, and they reach the judge as sources.
    assert "ignore them" in SYSTEM_PROMPT

@pytest.mark.parametrize("text, expected", [
    ('{"verdict": "pass", "reason": "stated in the warranty terms"}', (True, "stated in the warranty terms")),
    ('{"verdict": "fail"}', (False, "")),
    ('{"verdict": "maybe"}', None),
    ("PASS", None),
    ("[]", None),
])
def test_only_well_formed_verdicts_count(text, expected):
    assert parse_verdict(text) == expected

def test_the_judge_version_changes_with_its_model_or_rubric():
    rubric = RUBRICS["groundedness"]
    assert judge_version("a", rubric) == judge_version("a", rubric)
    assert judge_version("a", rubric) != judge_version("b", rubric)
    assert judge_version("a", rubric) != judge_version("a", RUBRICS["correctness"])

# --- Judging ---

def test_a_verdict_becomes_a_score_tagged_with_the_judge_version():
    judge = fake_judge(verdict_json(True, "the warranty is stated"))

    score = asyncio.run(judge.score(RUBRICS["groundedness"], case, trial()))

    assert (score.scorer, score.value) == ("judge_groundedness", 1.0)
    version = judge_version(judge.model, RUBRICS["groundedness"])
    assert score.reasoning == f"judge {version}: the warranty is stated"
    assert judge.cost > 0  # the call was traced and priced
    assert "llama-3.3-70b-versatile" in PRICES

def test_nothing_to_judge_means_no_call():
    judge = fake_judge()  # no replies: any call would fail
    rubric = RUBRICS["groundedness"]

    assert asyncio.run(judge.score(rubric, case, trial(output=None))) is None
    assert asyncio.run(judge.score(rubric, case, trial(output="unknown"))) is None  # claims nothing
    # "unknown" is still judged for correctness: it may be the wrong reply.
    wrong_abstention = asyncio.run(fake_judge(verdict_json(False)).score(RUBRICS["correctness"], case,
                                                                       trial(output="unknown")))
    assert wrong_abstention.value == 0.0

def test_bad_replies_and_failed_calls_are_counted_not_fatal():
    judge = fake_judge("not json", RuntimeError("outage"), verdict_json(False))
    rubric = RUBRICS["correctness"]
    trials = [trial(trial_id=1), trial(trial_id=2), trial(trial_id=3)]

    scores = asyncio.run(judge_trials(judge, rubric, [case], trials, concurrency=1))

    assert [(score.trial_id, score.value) for score in scores] == [(3, 0.0)]
    assert (judge.unparsed, judge.failed) == (1, 1)

# --- Storage: re-scoring must not throw away judge scores ---

def test_rescoring_the_built_in_scorers_keeps_the_judge_scores(tmp_path):
    conn = storage.connect(tmp_path / "judge.db")
    experiment = Experiment(name="e", dataset_name="mixed", config=AgentConfig(model="mock-llm", prompt="p"),
                            attempts=1, prices=PRICES, created_at=datetime.now(timezone.utc))
    stored = trial()
    experiment_id = storage.save_experiment(conn, experiment, [stored], [
        Score(trial_id=1, scorer="exact_match", value=0.0),
        Score(trial_id=1, scorer="judge_correctness", value=1.0, reasoning="judge x: same value"),
    ])

    storage.replace_scores(conn, experiment_id, [Score(trial_id=1, scorer="exact_match", value=1.0)], SCORER_NAMES)

    kept = {score.scorer: score.value for score in storage.load_scores(conn, experiment_id)}
    assert kept == {"exact_match": 1.0, "judge_correctness": 1.0}

# --- Calibration ---

def test_kappa_rewards_agreement_beyond_chance():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    # A judge that passes everything agrees 90% of the time with mostly-pass labels, and is worthless.
    assert cohens_kappa([True] * 9 + [False], [True] * 10) == 0.0
    # Textbook example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no gives kappa 0.4.
    first = [True] * 25 + [False] * 25
    second = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert cohens_kappa(first, second) == pytest.approx(0.4)

def test_agreement_lines_up_labels_and_scores_by_trial():
    labels = {1: True, 2: True, 3: False, 4: False, 99: True}  # trial 99 was never judged
    scores = [Score(trial_id=1, scorer="judge_groundedness", value=1.0),
              Score(trial_id=2, scorer="judge_groundedness", value=0.0, reasoning="judge x: not in sources"),
              Score(trial_id=3, scorer="judge_groundedness", value=0.0),
              Score(trial_id=4, scorer="judge_groundedness", value=0.0),
              Score(trial_id=5, scorer="judge_groundedness", value=1.0)]  # never labelled

    result = agreement(labels, scores)

    assert (result.compared, result.agreement) == (4, 0.75)
    assert result.confusion == {(True, True): 1, (True, False): 1, (False, True): 0, (False, False): 2}
    assert [trial_id for trial_id, _, _ in result.disagreements] == [2]
    assert "you said pass, judge x: not in sources" in format_agreement(result)

def test_small_samples_are_flagged_whatever_the_kappa():
    assert "label more" in verdict(1.0, 10)
    assert "label more" not in verdict(0.85, 50)
    assert verdict(0.2, 50).startswith("weak")

def test_labelling_saves_each_answer_and_resumes_where_it_stopped(tmp_path):
    path = tmp_path / "labels.jsonl"
    rubric = RUBRICS["groundedness"]
    trials = [trial(trial_id=n) for n in (1, 2, 3)]
    shown: list[str] = []

    replies = iter(["x", "p", "q"])  # an invalid key is asked again; quit after one label
    assert label(rubric, [case], trials, path, 7, ask=lambda _: next(replies), say=shown.append) == 1
    assert load_labels(path) == {1: True}

    replies = iter(["s", "f"])  # resumes at trial 2: skip it, fail trial 3
    assert label(rubric, [case], trials, path, 7, ask=lambda _: next(replies), say=shown.append) == 1
    assert load_labels(path) == {1: True, 3: False}
    assert json.loads(path.read_text().splitlines()[0]) == {
        "experiment_id": 7, "trial_id": 1, "rubric": "groundedness", "label": "pass"}
    # The labeller sees the material, never the judge's verdict.
    assert not any("judge" in text for text in shown)

def test_the_sample_is_fixed_and_only_holds_trials_the_rubric_applies_to():
    trials = [trial(trial_id=n) for n in range(1, 21)] + [trial(output="unknown", trial_id=21)]
    rubric = RUBRICS["groundedness"]

    sample = sample_trials(trials, rubric, size=5, seed=0)

    assert len(sample) == 5 and 21 not in {t.id for t in sample}
    assert sample == sample_trials(trials, rubric, size=5, seed=0)
