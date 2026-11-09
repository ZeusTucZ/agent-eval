"""1.1: trajectories, Anthropic and OpenTelemetry, conversations, pytest checks, templates,
the response cache, the larger dataset, trends and the demo. No network."""
import json

import pytest

from agenteval.models import EvaluationCase, Span, Trial
from agenteval.models import dataset_version
from agenteval.sample_data import hard_cases
from agenteval.scorers import redundant_calls, step_efficiency, trajectory_order

def tool(name, inp='{"q": 1}', status="success", trial_id=1):
    return Span(trial_id=trial_id, kind="tool", name=name, start=0, end=0, status=status, input=inp)

def responded(*spans, output="x"):
    llm = Span(trial_id=1, kind="llm", name="m", start=0, end=0, status="success")
    return Trial(id=1, case_id=1, attempt=1, output=output, spans=[llm, *spans])

case = EvaluationCase(id=1, input="q", expected_output="x", expected_trajectory=["search", "calculator"])

# --- Trajectories ---

def test_trajectory_order_allows_calls_in_between_but_not_out_of_order():
    assert trajectory_order(case, responded(tool("search"), tool("lookup"), tool("calculator"))).value == 1.0
    assert trajectory_order(case, responded(tool("calculator"), tool("search"))).value == 0.0
    assert trajectory_order(case, responded(tool("search"))).value == 0.0
    assert trajectory_order(case.model_copy(update={"expected_trajectory": None}), responded()) is None

def test_step_efficiency_is_needed_over_made():
    assert step_efficiency(case, responded(tool("search"), tool("calculator"))).value == 1.0
    four = responded(tool("search", "1"), tool("search", "2"), tool("calculator", "3"), tool("calculator", "4"))
    assert step_efficiency(case, four).value == 0.5
    assert step_efficiency(case.model_copy(update={"max_tool_calls": 4}), four).value == 1.0
    assert step_efficiency(case, responded()) is None  # nothing called: tool_selection's job

def test_repeating_a_call_that_worked_is_redundant_but_retrying_an_error_is_not():
    assert redundant_calls(case, responded(tool("search"), tool("search"))).value == 0.0
    assert redundant_calls(case, responded(tool("search", status="error"), tool("search"))).value == 1.0
    assert redundant_calls(case, responded(tool("search", '{"q": 1}'), tool("search", '{"q": 2}'))).value == 1.0
    assert redundant_calls(case, responded(tool("search", inp=None))) is None  # arguments not recorded

def test_new_case_fields_leave_dataset_versions_alone():
    assert dataset_version(hard_cases()) == "627d99e96799"


# --- Anthropic and OpenTelemetry ---

class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o
        self.cache_read_input_tokens = 5
        self.cache_creation_input_tokens = 0

class _Message:
    def __init__(self, text):
        self.content = [type("Block", (), {"text": text})()]
        self.usage = _Usage(40, 7)

class _Messages:
    def __init__(self, sync=False):
        self.sync = sync

    def create(self, **kwargs):
        if self.sync:
            return _Message("Paris")

        async def reply():
            return _Message("Paris")
        return reply()

@pytest.mark.parametrize("sync", [False, True])
def test_wrap_anthropic_records_each_message_with_its_tokens(sync):
    from types import SimpleNamespace

    from agenteval import evaluate, wrap_anthropic

    client = wrap_anthropic(SimpleNamespace(messages=_Messages(sync)))

    async def async_agent(question):
        message = await client.messages.create(model="claude-opus-5-5", max_tokens=10, messages=[])
        return message.content[0].text

    def sync_agent(question):
        return client.messages.create(model="claude-opus-5-5", max_tokens=10, messages=[]).content[0].text

    results = evaluate(sync_agent if sync else async_agent,
                       [{"input": "Capital of France?", "expected_output": "Paris"}],
                       model="claude-opus-5-5", price=(4, 20), attempts=1)
    (span,) = results.trials[0].spans
    assert (span.name, span.input_tokens, span.output_tokens) == ("claude-opus-5-5", 45, 7)
    assert results.pass_rate == 1.0

def otlp_span(trace, span_id, parent, name, start, end, attributes, error=False):
    def value(v):
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        return {"stringValue": v}
    return {"traceId": trace, "spanId": span_id, "parentSpanId": parent, "name": name,
            "startTimeUnixNano": str(start), "endTimeUnixNano": str(end),
            "attributes": [{"key": k, "value": value(v)} for k, v in attributes.items()],
            "status": {"code": 2, "message": "timeout"} if error else {"code": 1}}

def test_openinference_traces_import_one_trial_per_trace(tmp_path):
    from agenteval import storage
    from agenteval.traces import import_traces

    spans = [
        otlp_span("t1", "a", "", "agent", 0, 2_000_000_000,
                  {"openinference.span.kind": "AGENT", "input.value": json.dumps({"input": "Weather in Lisbon?"}),
                   "output.value": "Sunny, 24C"}),
        otlp_span("t1", "b", "a", "ChatOpenAI", 100_000_000, 900_000_000,
                  {"openinference.span.kind": "LLM", "llm.model_name": "gpt-4o-mini",
                   "llm.token_count.prompt": 120, "llm.token_count.completion": 15}),
        otlp_span("t1", "c", "a", "get_weather", 1_000_000_000, 1_200_000_000,
                  {"openinference.span.kind": "TOOL", "tool.name": "get_weather",
                   "input.value": json.dumps({"city": "Lisbon"}), "output.value": "Sunny, 24C"}),
        # A second trace using the GenAI conventions, whose tool call failed.
        otlp_span("t2", "x", "", "invoke_agent", 3_000_000_000, 4_000_000_000,
                  {"agenteval.input": "Weather in Oslo?", "agenteval.output": "unknown"}),
        otlp_span("t2", "y", "x", "chat", 3_100_000_000, 3_500_000_000,
                  {"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4o-mini",
                   "gen_ai.usage.input_tokens": 90, "gen_ai.usage.output_tokens": 5}),
        otlp_span("t2", "z", "x", "execute_tool get_weather", 3_600_000_000, 3_700_000_000,
                  {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_weather"}, error=True),
    ]
    path = tmp_path / "traces.json"
    path.write_text(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}))
    conn = storage.connect(tmp_path / "o.db")

    experiment_id = import_traces(conn, path, "otel", "gpt-4o-mini", tmp_path / "cases.jsonl")

    first, second = storage.load_trials(conn, experiment_id)
    assert first.output == "Sunny, 24C"
    assert [(s.kind, s.name) for s in first.spans] == [("llm", "gpt-4o-mini"), ("tool", "get_weather")]
    assert (first.spans[0].input_tokens, first.spans[0].output_tokens) == (120, 15)
    assert first.spans[0].duration_ms == pytest.approx(800)
    assert second.spans[1].status == "error" and second.spans[1].error_message == "timeout"
    scores = {(s.trial_id, s.scorer): s.value for s in storage.load_scores(conn, experiment_id)}
    assert scores[(first.id, "tool_faithfulness")] == 1.0  # the answer is in the tool's result
    cases = [json.loads(line) for line in (tmp_path / "cases.jsonl").read_text().splitlines()]
    assert [c["input"] for c in cases] == ["Weather in Lisbon?", "Weather in Oslo?"]

# --- The response cache ---

def test_the_response_cache_replays_identical_requests_per_attempt(tmp_path):
    import httpx
    import openai

    from agenteval import evaluate, wrap_openai

    calls = []

    def handler(request):
        calls.append(1)
        body = {"id": "c", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
        return httpx.Response(200, json=body)

    def make_agent():
        client = wrap_openai(openai.AsyncOpenAI(api_key="t", base_url="https://example.invalid/v1",
                                                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))),
                             cache=tmp_path / "responses.db")

        async def agent(question):
            response = await client.chat.completions.create(model="gpt-4o-mini",
                                                            messages=[{"role": "user", "content": question}])
            return response.choices[0].message.content
        return agent

    cases = [{"input": "hi", "expected_output": "ok"}]
    first = evaluate(make_agent(), cases, model="gpt-4o-mini", price=(0.15, 0.6), attempts=2)
    assert len(calls) == 2  # two attempts: two different cache keys
    again = evaluate(make_agent(), cases, model="gpt-4o-mini", price=(0.15, 0.6), attempts=2)
    assert len(calls) == 2  # everything replayed
    assert again.pass_rate == 1.0 and first.cost > 0 and again.cost == 0


# --- Conversations ---

conversation = EvaluationCase(
    id=1, expected_output="Your refund is on its way.", must_not_contain=["password"],
    turns=["I want to return my order.", "It's order 123.", "Thanks, when do I get my money?"],
)

def test_a_conversation_case_takes_its_question_from_the_last_turn():
    assert conversation.input == "Thanks, when do I get my money?"
    with pytest.raises(ValueError, match="input"):
        EvaluationCase(id=2, expected_output="x")

def test_the_agent_answers_each_turn_and_the_transcript_is_kept(tmp_path):
    from agenteval import evaluate, storage

    seen = []

    async def support_agent(messages):
        seen.append(len(messages))
        last = messages[-1]["content"]
        if "return" in last:
            return "Sure, what's the order number?"
        if "order 123" in last:
            return "Found it. The refund is approved."
        return "Your refund is on its way."

    results = evaluate(support_agent, [conversation], model="m", price=(0, 0), attempts=1, db=tmp_path / "c.db")
    trial = results.trials[0]
    assert seen == [1, 3, 5]  # the conversation so far, each time
    assert [m["role"] for m in trial.transcript] == ["user", "assistant"] * 3
    assert trial.output == "Your refund is on its way." and results.pass_rate == 1.0
    stored = storage.load_trials(storage.connect(tmp_path / "c.db"), results.experiment_id)[0]
    assert stored.transcript == trial.transcript

def test_forbidden_text_counts_in_every_reply_not_only_the_last():
    from agenteval import evaluate

    async def leaky(messages):
        return "Tell me your password." if len(messages) == 1 else "Your refund is on its way."

    results = evaluate(leaky, [conversation], model="m", price=(0, 0), attempts=1)
    assert results.score("avoids_forbidden_text") == 0.0
    assert results.pass_rate == 1.0  # the last answer was right; the leak is caught separately

def test_conversations_need_an_agent_that_takes_messages():
    from agenteval import evaluate

    results = evaluate(lambda question: "x", [conversation], model="m", price=(0, 0), attempts=1)
    assert results.trials[0].failure_type == "agent_error"

def test_the_judge_sees_the_whole_conversation():
    from agenteval.judge import RUBRICS, build_prompt

    trial = Trial(id=1, case_id=1, attempt=1, output="On its way.", transcript=[
        {"role": "user", "content": "I want to return my order."},
        {"role": "assistant", "content": "What's the order number?"},
        {"role": "user", "content": "Thanks, when do I get my money?"},
        {"role": "assistant", "content": "On its way."},
    ])
    prompt = build_prompt(RUBRICS["answer_relevance"], conversation, trial)
    assert "<conversation>" in prompt and "What's the order number?" in prompt
    assert "<answer>\nOn its way.\n</answer>" in prompt

def test_dataset_files_accept_turns_without_input(tmp_path):
    from agenteval.sample_data import load_dataset_file

    path = tmp_path / "chat.jsonl"
    path.write_text(json.dumps({"turns": ["hi", "what's 2+2?"], "expected_output": "4"}) + "\n")
    (case,) = load_dataset_file(path)
    assert case.input == "what's 2+2?" and case.turns == ["hi", "what's 2+2?"]

# --- Checks for tests, and init templates ---

def test_check_passes_or_says_exactly_what_fell_short():
    from agenteval import evaluate

    cases = [{"input": f"{n}", "expected_output": "even" if n % 2 == 0 else "odd"} for n in range(10)]
    results = evaluate(lambda q: "even", cases, model="m", price=(0, 0), attempts=1)
    assert results.check(min_pass_rate=0.5, max_cost=1) is results
    with pytest.raises(AssertionError) as error:
        results.check(min_pass_rate=0.9, min_scores={"tool_selection": 1.0}, max_failures=-1)
    message = str(error.value)
    assert "pass rate 50.0% is below 90.0%" in message
    assert "tool_selection produced no scores" in message
    assert "case 2: '1' -> 'even' (expected 'odd')" in message

@pytest.mark.parametrize("template", ["basic", "rag"])
def test_templates_that_need_no_key_run_out_of_the_box(template, tmp_path, monkeypatch, capsys):
    from agenteval import storage
    from agenteval.cli import main as cli

    import sys

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    # Each template writes its own my_agent.py: don't reuse another test's cached module.
    monkeypatch.delitem(sys.modules, "my_agent", raising=False)
    cli(["init", "--template", template])
    cli(["run", "--quiet"])
    scores = storage.load_scores(storage.connect(tmp_path / "agenteval.db"), 1)
    assert any(s.scorer == "exact_match" for s in scores)
    if template == "rag":
        assert any(s.scorer == "retrieval_recall" for s in scores)
    sys.modules.pop("my_agent", None)

@pytest.mark.parametrize("template", ["openai", "anthropic", "langgraph"])
def test_api_templates_name_their_model_and_what_they_need(template, tmp_path, monkeypatch, capsys):
    import tomllib

    from agenteval.cli import main as cli

    monkeypatch.chdir(tmp_path)
    cli(["init", "--template", template])
    out = capsys.readouterr().out
    config = tomllib.loads((tmp_path / "agenteval.toml").read_text())
    assert config["run"]["model"] != "example-model" and config["run"]["price"][1] > 0
    assert "API_KEY" in out
    compile((tmp_path / "my_agent.py").read_text(), "my_agent.py", "exec")  # valid Python

# --- The extended dataset ---

def test_extended_matches_its_builder_and_every_answer_is_reachable():
    from scripts.build_extended_dataset import build, to_jsonl

    from agenteval.sample_data import DATASETS_DIR, dataset_by_name, extended_cases, tools_for_dataset
    from agenteval.tools import ToolRegistry

    assert (DATASETS_DIR / "extended.jsonl").read_text(encoding="utf-8") == to_jsonl(build())
    cases = extended_cases()
    assert len(cases) == 200 and len({c.id for c in cases}) == 200 and len({c.input for c in cases}) == 200
    assert dataset_by_name("extended") == cases
    registry = ToolRegistry(tools_for_dataset("extended"))
    for case in cases:
        assert set(case.required_tools or []) <= set(registry.tools), case.input
    assert {t.name for t in tools_for_dataset("extended#sample=20,seed=0")} == set(registry.tools)

def test_extended_catches_a_ten_point_drop_most_of_the_time():
    from agenteval.power import detection_rate

    assert detection_rate(200, 3, 0.85, 0.10, simulations=200) > 0.7
    assert detection_rate(55, 3, 0.85, 0.10, simulations=200) < 0.4


def test_an_inline_dataset_keeps_its_questions_in_reports_and_the_dashboard(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from agenteval import evaluate, storage
    from agenteval.api import app
    from agenteval.compare import load
    from agenteval.sample_data import cases_for

    db = tmp_path / "runs.db"
    cases = [{"input": f"Is {n} even?", "expected_output": "yes" if n % 2 == 0 else "no",
              "category": "parity"} for n in range(4)]
    results = evaluate(lambda q: "yes", cases, model="m", price=(0, 0), attempts=1, db=db)

    conn = storage.connect(str(db))
    experiment = storage.load_experiment(conn, results.experiment_id)
    assert experiment.dataset_name == "inline"  # no file or built-in name could rebuild these
    assert [case.input for case in cases_for(conn, experiment)] == [case["input"] for case in cases]
    assert load(conn, results.experiment_id).case_list()[0].category == "parity"

    monkeypatch.setenv("DATABASE_PATH", str(db))
    with TestClient(app) as client:
        trials = client.get(f"/api/experiments/{results.experiment_id}/trials").json()["trials"]
        summary = client.get(f"/api/experiments/{results.experiment_id}").json()
    assert {trial["case"]["input"] for trial in trials} == {case["input"] for case in cases}
    assert "parity" in json.dumps(summary)
