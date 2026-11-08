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
        message = await client.messages.create(model="claude-sonnet-4-5", max_tokens=10, messages=[])
        return message.content[0].text

    def sync_agent(question):
        return client.messages.create(model="claude-sonnet-4-5", max_tokens=10, messages=[]).content[0].text

    results = evaluate(sync_agent if sync else async_agent,
                       [{"input": "Capital of France?", "expected_output": "Paris"}],
                       model="claude-sonnet-4-5", price=(3, 15), attempts=1)
    (span,) = results.trials[0].spans
    assert (span.name, span.input_tokens, span.output_tokens) == ("claude-sonnet-4-5", 45, 7)
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
