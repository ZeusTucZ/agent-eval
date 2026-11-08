"""The 1.0 features: the agenteval command and config, saving as trials finish, resuming,
budgets, sampling, redaction, your own scorers, exports, power, prices, the broader judge,
automatic tracing for OpenAI clients and LangChain, trace import/export, test generation
and the Python API. No network: fake clients and the mock agent throughout."""
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from agenteval import Case, evaluate, llm_call, storage, tool, wrap_openai
from agenteval.cli import config_arguments
from agenteval.cli import main as cli
from agenteval.compare import ExperimentData, check_comparable
from agenteval.cost import PRICES
from agenteval.execution import RunPlan, aexecute
from agenteval.export import render
from agenteval.judge import RUBRICS, Judge, applies, build_prompt, judge_version, load_rubric
from agenteval.mock_agent import FaultConfig, MockAgent
from agenteval.models import AgentConfig, EvaluationCase, Experiment, Span, Trial
from agenteval.power import detection_rate
from agenteval.pricing import estimate, parse_litellm
from agenteval.privacy import redact, redact_trial
from agenteval.retry import RetryPolicy
from agenteval.sample_data import arithmetic_cases, dataset_by_name, sample_cases, sample_name
from agenteval.scorers import as_scorer, exact_match
from agenteval.sdk import FunctionAgent

NO_LATENCY = (0, 0)

def completion(content: str, model: str = "m", prompt_tokens: int = 100, completion_tokens: int = 10):
    return ChatCompletion.model_validate({
        "id": "c", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    })

class FakeCompletions:
    """Async or sync, replying with the queued contents and recording each request."""

    def __init__(self, replies, sync=False):
        self.replies, self.requests, self.sync = list(replies), [], sync

    def __getattr__(self, name):
        if name == "create":
            return self._sync_create if self.sync else self._async_create
        raise AttributeError(name)

    def _next(self, kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return completion(reply, kwargs.get("model", "m"))

    def _sync_create(self, **kwargs):
        return self._next(kwargs)

    async def _async_create(self, **kwargs):
        return self._next(kwargs)

def fake_client(*replies, sync=False):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(replies, sync=sync)))

def mock_plan(cases=5, attempts=2, **faults) -> RunPlan:
    config = FaultConfig(tool_error_rate=0.0, hallucination_rate=0.0, rate_limit_rate=0.0, latency_ms=NO_LATENCY,
                         **faults)
    return RunPlan(agent=MockAgent(config), config=AgentConfig(model="mock-llm", prompt="p"),
                   dataset_name=f"arithmetic-{cases}", cases=arithmetic_cases(cases), attempts=attempts,
                   concurrency=4, timeout_s=5, prices=dict(PRICES))

# --- The command and its config ---

def test_config_tables_become_command_line_defaults():
    config = {"db": "results.db", "run": {"agent": "a:b", "price": [0.1, 0.2], "redact": True, "quiet": False,
                                         "scorer": ["x:y", "z:w"], "max_cost": 2}}
    assert config_arguments("run", config) == [
        "--agent", "a:b", "--price", "0.1", "0.2", "--redact", "--scorer", "x:y", "--scorer", "z:w",
        "--max-cost", "2", "--db", "results.db"]
    assert config_arguments("gate", config) == []  # gate takes its databases as arguments

def test_init_then_run_then_report_works_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "my_agent", raising=False)  # another test's my_agent.py may be cached
    cli(["init"])
    assert {p.name for p in tmp_path.iterdir()} >= {"my_agent.py", "cases.jsonl", "agenteval.toml"}
    with pytest.raises(SystemExit, match="Not overwriting"):
        cli(["init"])

    cli(["run", "--quiet"])  # every option comes from agenteval.toml
    cli(["report"])
    out = capsys.readouterr().out
    assert "Pass rate:        100.0%" in out and "first-run" in out
    cli(["list"])
    assert "first-run" in capsys.readouterr().out

def test_unknown_commands_show_the_list(capsys):
    with pytest.raises(SystemExit, match="Unknown command"):
        cli(["nope"])
    cli([])
    assert "agenteval <command>" in capsys.readouterr().out

# --- Saving as trials finish, resuming, budgets ---

def test_each_trial_is_saved_as_it_finishes(tmp_path):
    conn = storage.connect(tmp_path / "a.db")
    seen_in_db = []

    plan = mock_plan()
    original = plan.agent.run

    async def run_and_peek(case, attempt, tracer):
        seen_in_db.append(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
        return await original(case, attempt, tracer)

    plan.agent.run = run_and_peek
    plan.concurrency = 1
    result = asyncio.run(aexecute(plan, conn, progress=False))

    assert seen_in_db == list(range(10))  # each trial saw all the earlier ones already saved
    experiment = storage.load_experiment(conn, result.experiment_id)
    assert experiment.status == "completed" and experiment.planned_trials == 10
    assert "python" in experiment.environment

def test_an_interrupted_run_resumes_with_only_the_missing_trials(tmp_path):
    conn = storage.connect(tmp_path / "a.db")
    plan = mock_plan()
    calls = 0
    original = plan.agent.run

    hang = asyncio.Event()  # never set: the fifth trial waits until the run is cancelled

    async def stall_after_four(case, attempt, tracer):
        nonlocal calls
        calls += 1
        if calls > 4:
            await hang.wait()
        return await original(case, attempt, tracer)

    plan.agent.run = stall_after_four
    plan.concurrency = 1
    plan.timeout_s = 60

    async def interrupt():
        # Ctrl+C under asyncio.run cancels the main task (Python 3.11+); do the same.
        task = asyncio.create_task(aexecute(plan, conn, progress=False))
        while calls <= 4:
            await asyncio.sleep(0.01)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(interrupt())
    experiment_id = storage.latest_experiment_id(conn)
    assert storage.load_experiment(conn, experiment_id).status == "interrupted"
    assert len(storage.load_trials(conn, experiment_id)) == 4

    resumed = mock_plan()
    ran = []
    original_resumed = resumed.agent.run

    async def record(case, attempt, tracer):
        ran.append((case.id, attempt))
        return await original_resumed(case, attempt, tracer)

    resumed.agent.run = record
    result = asyncio.run(aexecute(resumed, conn, resume_id=experiment_id, progress=False))
    assert len(ran) == 6 and (1, 1) not in ran
    assert [t.id for t in result.trials] == list(range(1, 11))
    assert storage.load_experiment(conn, experiment_id).status == "completed"

def test_resuming_with_different_options_is_refused(tmp_path):
    conn = storage.connect(tmp_path / "a.db")
    first = asyncio.run(aexecute(mock_plan(attempts=1), conn, progress=False))
    with pytest.raises(ValueError, match="attempts differ"):
        asyncio.run(aexecute(mock_plan(attempts=2), conn, resume_id=first.experiment_id, progress=False))

def test_max_cost_stops_starting_trials(tmp_path):
    conn = storage.connect(tmp_path / "a.db")
    plan = mock_plan(cases=20)
    plan.concurrency = 1
    result = asyncio.run(aexecute(plan, conn, max_cost=0.002, progress=False))
    assert 0 < len(result.trials) < 40 and result.stopped_by_budget
    assert storage.load_experiment(conn, result.experiment_id).status == "interrupted"

def test_compare_refuses_an_interrupted_experiment():
    def data(status):
        experiment = Experiment(id=1, name="x", dataset_name="d", dataset_version="v", attempts=1,
                                config=AgentConfig(model="m", prompt="p"), prices={},
                                created_at=datetime.now(UTC), status=status, planned_trials=10)
        return ExperimentData(experiment, [], [])

    assert any("interrupted" in e for e in check_comparable(data("completed"), data("interrupted")).errors)

# --- Sampling, redaction, scorers ---

def test_samples_are_stratified_reproducible_and_rebuildable():
    cases = [EvaluationCase(id=n, input=str(n), expected_output="x", category="a" if n <= 30 else "b")
             for n in range(1, 41)]
    sample = sample_cases(cases, 8, seed=1)
    assert [c.category for c in sample].count("a") == 6 and len(sample) == 8
    assert sample == sample_cases(cases, 8, seed=1) != sample_cases(cases, 8, seed=2)
    assert dataset_by_name(sample_name("hard", 10, 3)) == sample_cases(dataset_by_name("hard"), 10, 3)

@pytest.mark.parametrize("text, expected", [
    ("mail ana@example.com now", "mail [EMAIL] now"),
    ("card 4111 1111 1111 1111.", "card [CARD]."),
    ("call +34 612 345 678", "call [PHONE]"),
    ("from 10.0.0.12", "from [IP]"),
    ("key gsk_abcdefghijklmnopqrstuvwx", "key [SECRET]"),
    ("the answer is 42", "the answer is 42"),
])
def test_redaction(text, expected):
    assert redact(text) == expected

def test_redacted_runs_score_the_real_text_but_store_the_redacted_one(tmp_path):
    async def agent(question):
        await tool("lookup", lambda email: f"found {email}", email="ana@example.com")
        return "ana@example.com"

    cases = [EvaluationCase(id=1, input="whose?", expected_output="ana@example.com")]
    results = evaluate(agent, cases, model="m", price=(0, 0), attempts=1, redact=True, db=tmp_path / "r.db")
    assert results.pass_rate == 1.0  # scored before redaction
    stored = results.trials[0]
    assert stored.output == "[EMAIL]" and "[EMAIL]" in stored.spans[0].output
    assert "ana@" not in json.dumps([t.model_dump() for t in results.trials])
    assert redact_trial(Trial(id=1, case_id=1, attempt=1, output=None)).output is None

def test_your_own_scorers_accept_numbers_bools_pairs_and_scores():
    case = EvaluationCase(id=1, input="q", expected_output="a")
    trial = Trial(id=1, case_id=1, attempt=1, output="a short answer")

    def brevity(case, trial):
        return len(trial.output) < 20, "short enough"

    score = as_scorer(brevity)(case, trial)
    assert (score.scorer, score.value, score.reasoning) == ("brevity", 1.0, "short enough")
    assert as_scorer(lambda c, t: 0.25, name="quarter")(case, trial).value == 0.25
    assert as_scorer(lambda c, t: None)(case, trial) is None
    with pytest.raises(ValueError, match="between 0 and 1"):
        as_scorer(lambda c, t: 3)(case, trial)

def test_custom_scorers_run_in_the_cli_and_again_in_rescore(tmp_path, monkeypatch, capsys):
    (tmp_path / "my_scorers.py").write_text("def short(case, trial):\n    return len(trial.output or '') < 3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    db = str(tmp_path / "s.db")
    cli(["run", "--cases", "5", "--attempts", "1", "--scorer", "my_scorers:short", "--db", db, "--quiet"])
    conn = storage.connect(db)
    assert {s.scorer for s in storage.load_scores(conn, 1)} >= {"short", "exact_match"}
    assert storage.load_experiment(conn, 1).custom_scorers == ["my_scorers:short"]
    cli(["rescore", "1", "--db", db])
    assert "short" in {s.scorer for s in storage.load_scores(conn, 1)}
    cli(["report", "1", "--db", db])
    assert "Other scores:" in capsys.readouterr().out

def test_exact_match_skips_cases_without_a_reference():
    assert exact_match(EvaluationCase(id=1, input="q", expected_output=""), Trial(id=1, case_id=1, attempt=1)) is None

# --- Reports and exports ---

def test_every_report_format_renders(tmp_path):
    results = evaluate(MockAgent(FaultConfig(latency_ms=NO_LATENCY, seed=3)), "arithmetic-10",
                       model="mock-llm", attempts=2)
    for fmt in ("json", "markdown", "html", "csv"):
        assert render(fmt, results.experiment, results.trials, results.scores, results.cases)
    data = json.loads(results.report("json"))
    assert data["trials"] == 20 and data["status"] == "completed"
    html = results.report("html")
    assert html.startswith("<!doctype html>") and "<script" not in html
    assert results.report("csv").splitlines()[0].startswith("trial,case,attempt,category,question")
    assert results.save(tmp_path / "r.md").read_text(encoding="utf-8").startswith("# python-run")

# --- Power, prices, estimates ---

def test_detection_rate_grows_with_cases_and_drop_size():
    small, large = (detection_rate(n, 3, 0.85, 0.15, simulations=150) for n in (30, 400))
    assert small < large and large > 0.9
    assert detection_rate(100, 3, 0.85, 0.0, simulations=150) < 0.1  # no drop: rarely "detected"

def test_litellm_prices_are_parsed_per_million_and_under_bare_names():
    prices = parse_litellm({
        "gpt-4o-mini": {"mode": "chat", "input_cost_per_token": 1.5e-07, "output_cost_per_token": 6e-07},
        "groq/openai/gpt-oss-20b": {"mode": "chat", "input_cost_per_token": 7.5e-08, "output_cost_per_token": 3e-07},
        "text-embedding-3-small": {"mode": "embedding", "input_cost_per_token": 2e-08},
        "sample_spec": "not a model",
    })
    assert prices["gpt-4o-mini"].input_per_million == pytest.approx(0.15)
    assert prices["openai/gpt-oss-20b"].output_per_million == pytest.approx(0.3)
    assert "text-embedding-3-small" not in prices

def test_user_prices_override_the_built_in_ones(tmp_path, monkeypatch):
    from agenteval.cost import load_user_prices

    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"prices": {"mock-llm": {"input_per_million": 1, "output_per_million": 2}}}))
    assert load_user_prices(path)["mock-llm"].output_per_million == 2
    assert load_user_prices(tmp_path / "missing.json") == {}

def test_estimates_come_from_earlier_runs(tmp_path):
    conn = storage.connect(tmp_path / "e.db")
    plan = mock_plan(cases=5, attempts=2)
    asyncio.run(aexecute(plan, conn, progress=False))
    result = estimate(conn, "mock-llm", planned_trials=100, dataset="arithmetic-5")
    assert result["same_dataset"] and result["cost"] == pytest.approx(result["cost_per_trial"] * 100)
    assert estimate(conn, "other-model", 10) is None

# --- The judge's new rubrics, providers and cache ---

def judged_trial(output="3", sources='["The warranty is 3 years."]'):
    spans = [Span(trial_id=1, kind="llm", name="m", start=0, end=0, status="success"),
             Span(trial_id=1, kind="tool", name="search", start=0, end=0, status="success", output=sources)]
    return Trial(id=1, case_id=1, attempt=1, output=output, spans=spans)

def test_context_relevance_judges_the_sources_without_the_answer():
    rubric = RUBRICS["context_relevance"]
    case = EvaluationCase(id=1, input="How long is the warranty?", expected_output="3")
    prompt = build_prompt(rubric, case, judged_trial())
    assert "<sources>" in prompt and "<answer>" not in prompt
    assert applies(rubric, judged_trial(output=None))  # no answer needed
    assert not applies(rubric, judged_trial(sources=None).model_copy(update={"spans": []}))

def test_built_in_rubric_versions_are_unchanged_and_custom_rubrics_load(tmp_path):
    assert judge_version("m", RUBRICS["groundedness"]) == judge_version("m", RUBRICS["groundedness"])
    path = tmp_path / "polite.toml"
    path.write_text('name = "politeness"\nquestion = "Is the ANSWER polite?"\n')
    rubric = load_rubric(str(path))
    assert (rubric.name, rubric.uses_expected, rubric.uses_answer) == ("politeness", False, True)
    with pytest.raises(ValueError, match="unknown rubric"):
        load_rubric("nonsense")

def test_verdicts_are_cached_so_rejudging_is_free(tmp_path):
    conn = storage.connect(tmp_path / "c.db")
    case = EvaluationCase(id=1, input="q", expected_output="3")
    verdict = json.dumps({"verdict": "pass", "reason": "same"})
    first = Judge(client=fake_client(verdict), requests_per_minute=None, cache=conn,
                  retry_policy=RetryPolicy(max_retries=0))
    assert asyncio.run(first.score(RUBRICS["correctness"], case, judged_trial())).value == 1.0
    second = Judge(client=fake_client(), requests_per_minute=None, cache=conn)  # any call would fail
    assert asyncio.run(second.score(RUBRICS["correctness"], case, judged_trial())).value == 1.0
    assert second.cached == 1

def test_any_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("MY_GATEWAY_KEY", "k")
    judge = Judge(provider="custom", base_url="https://gateway.example/v1", api_key_env="MY_GATEWAY_KEY",
                  requests_per_minute=None)
    assert str(judge.client.base_url).startswith("https://gateway.example/v1")

# --- Automatic tracing ---

@pytest.mark.parametrize("sync", [False, True])
def test_wrap_openai_records_each_call_with_its_tokens(sync):
    client = wrap_openai(fake_client("4", sync=sync))

    async def async_agent(question):
        response = await client.chat.completions.create(model="gpt-4o-mini", messages=[])
        return response.choices[0].message.content

    def sync_agent(question):
        return client.chat.completions.create(model="gpt-4o-mini", messages=[]).choices[0].message.content

    agent = FunctionAgent(sync_agent if sync else async_agent, model="default")
    from agenteval.runner import run_experiment

    trials = asyncio.run(run_experiment(agent, [EvaluationCase(id=1, input="2+2", expected_output="4")], 1))
    (span,) = trials[0].spans
    assert (span.kind, span.name, span.input_tokens, span.output_tokens) == ("llm", "gpt-4o-mini", 100, 10)
    assert trials[0].output == "4"

def test_wrap_openai_with_the_real_async_sdk():
    # The SDK's async methods hide behind a sync decorator, which an earlier version missed.
    import httpx
    import openai

    def handler(request):
        return httpx.Response(200, json=completion("ok", "gpt-4o-mini").model_dump())

    client = wrap_openai(openai.AsyncOpenAI(api_key="test", base_url="https://example.invalid/v1",
                                            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))

    async def agent(question):
        response = await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": question}])
        return response.choices[0].message.content

    results = evaluate(agent, [{"input": "hi", "expected_output": "ok"}], model="gpt-4o-mini",
                       price=(0.15, 0.60), attempts=1)
    (span,) = results.trials[0].spans
    assert (span.name, span.input_tokens, span.output_tokens) == ("gpt-4o-mini", 100, 10)
    assert results.cost > 0

def test_langchain_calls_and_tools_become_steps():
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool as lc_tool

    from agenteval.integrations.langchain import AgentEvalCallback

    @lc_tool
    def capital(country: str) -> str:
        """The capital of a country."""
        return {"France": "Paris"}[country]

    def agent(question: str) -> str:
        callback = AgentEvalCallback()
        model = GenericFakeChatModel(messages=iter([AIMessage(
            content="Paris", usage_metadata={"input_tokens": 50, "output_tokens": 2, "total_tokens": 52})]))
        found = capital.invoke({"country": "France"}, config={"callbacks": [callback]})
        reply = model.invoke(question, config={"callbacks": [callback]})
        return reply.content if found == "Paris" else "?"

    with pytest.warns(UserWarning, match="no price for GenericFakeChatModel"):
        results = evaluate(agent, [{"input": "Capital of France?", "expected_output": "Paris",
                                    "expected_tools": ["capital"]}], model="m", price=(0, 0), attempts=1)
    kinds = [(s.kind, s.name) for s in results.trials[0].spans]
    assert ("tool", "capital") in kinds and any(kind == "llm" for kind, _ in kinds)
    llm = next(s for s in results.trials[0].spans if s.kind == "llm")
    assert (llm.input_tokens, llm.output_tokens) == (50, 2)
    assert results.score("tool_selection") == 1.0 and results.pass_rate == 1.0

# --- Import and export ---

def test_traces_import_with_steps_and_ragas_samples(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"input": "What is 2+2?", "output": "4", "expected_output": "4",
         "steps": [{"kind": "llm", "name": "gpt-4o-mini", "input_tokens": 80, "output_tokens": 3, "duration_ms": 400},
                   {"kind": "tool", "name": "calculator", "input": {"a": 2}, "output": 4, "duration_ms": 5}]},
        {"user_input": "Warranty?", "response": "3 years", "retrieved_contexts": ["Warranty: 3 years."],
         "reference": "3 years"},
        {"input": "No reference here", "output": "something"},
    ]) + "\n")
    cli(["import", "logs.jsonl", "--db", "i.db"])
    conn = storage.connect("i.db")
    trials = storage.load_trials(conn, 1)
    assert [len(t.spans) for t in trials] == [2, 1, 0]
    assert trials[0].duration_ms == pytest.approx(405)
    assert trials[1].spans[0].name == "retrieve"
    scores = {(s.trial_id, s.scorer): s.value for s in storage.load_scores(conn, 1)}
    assert scores[(1, "exact_match")] == 1.0 and scores[(2, "exact_match")] == 1.0
    assert (3, "exact_match") not in scores  # no reference: skipped, not failed

    cli(["export", "1", "--db", "i.db", "-o", "back.jsonl"])
    cli(["export", "1", "--db", "i.db", "--format", "ragas", "-o", "ragas.jsonl"])
    ragas = [json.loads(line) for line in Path("ragas.jsonl").read_text().splitlines()]
    assert ragas[1] == {"user_input": "Warranty?", "response": "3 years",
                        "retrieved_contexts": ['["Warranty: 3 years."]'], "reference": "3 years"}
    cli(["import", "back.jsonl", "--db", "i.db", "--name", "round-trip"])
    assert [t.output for t in storage.load_trials(conn, 2)] == [t.output for t in trials]

def test_bad_trace_lines_name_the_line(tmp_path):
    from agenteval.traces import import_traces

    path = tmp_path / "bad.jsonl"
    path.write_text('{"input": "q", "output": "a"}\n{"output": "no question"}\n')
    with pytest.raises(ValueError, match="line 2"):
        import_traces(storage.connect(tmp_path / "b.db"), path, "x", "m", tmp_path / "cases.jsonl")

# --- Generating cases ---

def test_generate_writes_cases_linked_to_their_passages(tmp_path):
    from agenteval.generate import generate_cases, read_passages

    (tmp_path / "handbook.md").write_text("# Leave\n\nEmployees get 25 days of leave.\n\n# Travel\n\nHotels up to 200 USD.")
    documents = read_passages([tmp_path])
    assert [d.id for d in documents] == ["handbook-1"] and "25 days" in documents[0].text

    client = fake_client(json.dumps({"question": "How many days of leave?", "answer": "25"}),
                         "not json", json.dumps({"question": "Is there a gym?"}))
    cases = asyncio.run(generate_cases(client, "m", documents, count=3, unanswerable=0.0, concurrency=1))
    assert cases[0] == {"id": 1, "input": "How many days of leave?", "expected_output": "25",
                        "category": "generated", "relevant_docs": ["handbook-1"]}
    assert len(cases) == 1  # the malformed and answerless replies are dropped

# --- The Python API ---

def test_evaluate_runs_in_memory_and_compares_versions():
    async def good(question):
        return str(eval(question.removeprefix("What is ")))  # noqa: S307 - test-only arithmetic

    async def bad(question):
        return "0"

    cases = [Case(id=n, input=f"What is {n} + 1", expected_output=str(n + 1)) for n in range(1, 31)]
    before = evaluate(bad, cases, model="m", price=(0, 0), attempts=1, name="bad")
    after = evaluate(good, cases, model="m", price=(0, 0), attempts=1, name="good")
    assert before.pass_rate == 0.0 and after.pass_rate == 1.0 and before.experiment_id == 1
    assert len(before.failures()) == 30 and before.to_records()[0]["question"] == "What is 1 + 1"
    assert "Verdict: Better" in after.compare(before)
    assert "pass rate 100.0%" in repr(after)

def test_evaluate_works_inside_a_running_event_loop():
    async def notebook_cell():
        return evaluate(lambda q: "x", [{"input": "q", "expected_output": "x"}], model="m", price=(0, 0), attempts=1)

    assert asyncio.run(notebook_cell()).pass_rate == 1.0

def test_evaluate_needs_a_price():
    with pytest.raises(ValueError, match="no price"):
        evaluate(lambda q: "x", [{"input": "q", "expected_output": "x"}], model="unpriced-model")

def test_llm_call_and_tool_are_exported_from_the_package():
    assert callable(llm_call) and callable(tool) and os.path.basename(__file__)

@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="argparse's help layout differs between Pythons")
def test_the_cli_reference_matches_every_commands_help():
    # Generated with Python 3.12; if this fails, run: python scripts/build_cli_reference.py
    from scripts.build_cli_reference import OUTPUT, build

    assert OUTPUT.read_text(encoding="utf-8") == build()

def test_a_run_left_running_is_marked_interrupted(tmp_path):
    conn = storage.connect(tmp_path / "m.db")
    plan = mock_plan()
    experiment = Experiment(name="x", dataset_name="d", config=plan.config, attempts=1, prices={},
                            created_at=datetime.now(UTC), status="running")
    experiment_id = storage.start_experiment(conn, experiment)
    storage.mark_interrupted(conn, experiment_id)
    assert storage.load_experiment(conn, experiment_id).status == "interrupted"
    done = storage.start_experiment(conn, experiment.model_copy(update={"status": "completed"}))
    storage.mark_interrupted(conn, done)
    assert storage.load_experiment(conn, done).status == "completed"  # a finished run is left alone
