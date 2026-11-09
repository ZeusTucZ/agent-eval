"""Tests for 1.2: Claude as the judge, live OpenTelemetry tracing, simulated users, sharing
runs, production monitoring and custom agents from the dashboard. No network."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agenteval.models import EvaluationCase, Trial
from agenteval.retry import RetryPolicy

# --- Claude as the judge ---

class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    async def create(self, **request):
        from anthropic.types import Message

        self.requests.append(request)
        text, stop = self.replies.pop(0)
        return Message.model_validate({
            "id": "msg_1", "type": "message", "role": "assistant", "model": request["model"],
            "content": [{"type": "text", "text": text}] if text else [],
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 280, "output_tokens": 18, "cache_read_input_tokens": 20,
                      "cache_creation_input_tokens": 0},
        })

def claude_judge(*replies):
    from agenteval.judge import Judge

    messages = FakeMessages(replies)
    judge = Judge(model="claude-sonnet-5-5", provider="anthropic", client=SimpleNamespace(messages=messages),
                  requests_per_minute=None, retry_policy=RetryPolicy(max_retries=0, jitter=False))
    return judge, messages

def test_claude_judges_with_an_enforced_verdict_form_and_no_sampling_settings():
    from agenteval.judge import RUBRICS

    judge, messages = claude_judge((json.dumps({"verdict": "fail", "reason": "says 2 years"}), "end_turn"))
    case = EvaluationCase(id=1, input="Warranty?", expected_output="3 years")
    trial = Trial(id=7, case_id=1, attempt=1, output="2 years", spans=[])

    score = asyncio.run(judge.score(RUBRICS["correctness"], case, trial))

    assert (score.value, score.scorer) == (0.0, "judge_correctness")
    request = messages.requests[0]
    assert request["model"] == "claude-sonnet-5-5" and "temperature" not in request
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "<answer>\n2 years\n</answer>" in request["messages"][0]["content"]
    span = judge.tracer.spans[0]
    assert (span.input_tokens, span.output_tokens) == (300, 18)  # cache reads count as input
    assert judge.cost == pytest.approx((300 * 2.0 + 18 * 10.0) / 1e6)

def test_a_refusal_is_not_taken_for_a_verdict():
    from agenteval.judge import RUBRICS

    judge, _ = claude_judge(("", "refusal"))
    case = EvaluationCase(id=1, input="q", expected_output="a")
    assert asyncio.run(judge.score(RUBRICS["correctness"], case, Trial(id=1, case_id=1, attempt=1, output="a"))) is None
    assert judge.unparsed == 1

def test_the_judge_command_defaults_to_a_claude_model_for_the_anthropic_provider(monkeypatch):
    from agenteval import judge

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        judge.main(["1", "--rubric", "correctness", "--provider", "anthropic"])

# --- Live OpenTelemetry spans ---

def test_openinference_and_genai_spans_become_steps_of_the_running_trial():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Status, StatusCode

    from agenteval import evaluate
    from agenteval.integrations.otel import instrument

    provider = TracerProvider()
    instrument(provider)
    tracer = provider.get_tracer("my-framework")

    def agent(question: str) -> str:
        with tracer.start_as_current_span("agent run"):
            with tracer.start_as_current_span("ChatOpenAI", attributes={
                    "openinference.span.kind": "LLM", "llm.model_name": "gpt-4o-mini",
                    "llm.token_count.prompt": 120, "llm.token_count.completion": 9}):
                pass
            with tracer.start_as_current_span("execute_tool lookup", attributes={
                    "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "lookup",
                    "gen_ai.tool.call.arguments": '{"q": "warranty"}', "gen_ai.tool.call.result": "3 years"}):
                pass
            with tracer.start_as_current_span("execute_tool broken", attributes={
                    "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "broken"}) as span:
                span.set_status(Status(StatusCode.ERROR, "timed out"))
        return "3 years"

    with tracer.start_as_current_span("outside an evaluation", attributes={"openinference.span.kind": "LLM"}):
        pass  # ignored: no trial is running

    results = evaluate(agent, [{"input": "Warranty?", "expected_output": "3 years", "expected_tools": ["lookup"]}],
                       model="gpt-4o-mini", price=(0.15, 0.6), attempts=1)
    spans = results.trials[0].spans
    assert [(s.kind, s.name, s.status) for s in spans] == [
        ("llm", "gpt-4o-mini", "success"), ("tool", "lookup", "success"), ("tool", "broken", "error")]
    assert (spans[0].input_tokens, spans[0].output_tokens) == (120, 9)
    assert (spans[1].input, spans[1].output) == ('{"q": "warranty"}', "3 years")
    assert spans[2].error_message == "timed out"
    assert results.score("tool_selection") == 1.0 and results.cost > 0

# --- Simulated users ---

REFUND_CASE = {"input": "I want my money back.", "expected_output": "refund approved",
               "simulated_user": "Bought headphones 10 days ago (order 1234); they broke. Wants a refund.",
               "max_turns": 4}

def support_agent(messages):
    last = messages[-1]["content"]
    if "1234" in last:
        return "refund approved"
    return "Sure, what's your order number?"

def test_a_simulated_user_answers_the_agent_until_it_is_done():
    from agenteval import evaluate

    seen = []

    def simulator(case, messages):
        seen.append([m["role"] for m in messages])
        return None if messages[-1]["content"] == "refund approved" else "It's order 1234."

    results = evaluate(support_agent, [REFUND_CASE], model="m", price=(0, 0), attempts=1, user_simulator=simulator)
    trial = results.trials[0]
    assert trial.output == "refund approved" and results.pass_rate == 1.0
    assert [m["content"] for m in trial.transcript] == [
        "I want my money back.", "Sure, what's your order number?", "It's order 1234.", "refund approved"]
    assert seen == [["user", "assistant"], ["user", "assistant", "user", "assistant"]]
    assert results.experiment.config.params["user_simulator"] == "simulator"

def test_the_conversation_stops_at_max_turns_and_without_a_simulator_it_fails_clearly():
    from agenteval import evaluate

    stubborn = evaluate(support_agent, [REFUND_CASE], model="m", price=(0, 0), attempts=1,
                        user_simulator=lambda case, messages: "No.")
    assert sum(m["role"] == "user" for m in stubborn.trials[0].transcript) == 4

    with pytest.raises(ValueError, match="user_simulator"):
        evaluate(support_agent, [REFUND_CASE], model="m", price=(0, 0), attempts=1)

def test_the_simulator_model_plays_the_user_and_says_done(monkeypatch):
    from agenteval.models import EvaluationCase
    from agenteval.simulate import UserSimulator, parse_reply, simulator_prompt

    class Messages:
        async def create(self, **request):
            self.request = request
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="DONE.")])

    messages = Messages()
    simulator = UserSimulator("claude-haiku-5-5", provider="anthropic", client=SimpleNamespace(messages=messages))
    case = EvaluationCase(id=1, **REFUND_CASE)
    history = [{"role": "user", "content": "I want my money back."}, {"role": "assistant", "content": "Done!"}]

    assert asyncio.run(simulator(case, history)) is None
    assert "order 1234" in messages.request["messages"][0]["content"] and "temperature" not in messages.request
    assert "Assistant: Done!" in simulator_prompt(case, history)
    assert parse_reply("  Here it is: 1234 ") == "Here it is: 1234"

def test_simulated_cases_keep_older_dataset_versions():
    from agenteval.models import EvaluationCase, dataset_version

    case = EvaluationCase(id=1, input="q", expected_output="a")
    assert dataset_version([case]) == dataset_version([EvaluationCase(id=1, input="q", expected_output="a",
                                                                       simulated_user=None)])

# --- Sharing experiments with a team server ---

def test_push_copies_an_experiment_with_its_cases_to_the_shared_server(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from agenteval import evaluate, share, storage
    from agenteval.api import app

    local = tmp_path / "mine.db"
    cases = [{"input": f"{n} + 1?", "expected_output": str(n + 1), "category": "sums"} for n in range(3)]
    results = evaluate(lambda q: "2", cases, model="m", price=(0, 0), attempts=2, db=local, name="lorenzo-v2")

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "team.db"))
    monkeypatch.setenv("API_TOKEN", "secret")
    monkeypatch.delenv("READ_ONLY", raising=False)
    with TestClient(app) as client:
        def fake_send(url, token, body, timeout=60.0):
            response = client.post("/api/experiments", json=body, headers={"X-API-Token": token})
            if response.status_code >= 400:
                raise RuntimeError(response.json()["detail"])
            return response.json()

        monkeypatch.setattr(share, "send", fake_send)
        summary = share.push(storage.connect(str(local)), results.experiment_id, "http://team", "secret")
        assert summary["name"] == "lorenzo-v2" and summary["pass_rate"] == pytest.approx(1 / 3)
        trials = client.get(f"/api/experiments/{summary['id']}/trials").json()["trials"]
        assert {t["case"]["input"] for t in trials} == {c["input"] for c in cases}
        assert results.push("http://team", "secret") == summary["id"] + 1

        with pytest.raises(RuntimeError, match="X-API-Token"):
            fake_send("http://team", "wrong", share.payload(results.experiment, results.trials, results.scores, None))
        bad = share.payload(results.experiment, results.trials, results.scores, None)
        bad["scores"][0]["trial_id"] = 999
        assert client.post("/api/experiments", json=bad, headers={"X-API-Token": "secret"}).status_code == 422

def test_push_says_what_is_missing(monkeypatch):
    from agenteval import share

    monkeypatch.delenv("AGENTEVAL_SERVER", raising=False)
    with pytest.raises(SystemExit, match="--to"):
        share.main([])

# --- Production monitoring ---

def test_the_monitor_records_a_sample_of_real_calls_into_a_daily_experiment(tmp_path):
    from agenteval import storage, tool
    from agenteval.monitor import Monitor

    monitor = Monitor(tmp_path / "prod.db", model="gpt-4o-mini", price=(0.15, 0.6), sample_rate=1.0, redact=True)

    @monitor.watch
    async def answer(question: str) -> str:
        if "crash" in question:
            await tool("lookup", lambda: 1 / 0)
        await tool("lookup", lambda q: "3 years", question)
        return "3 years, mail me at ana@example.com" if "mail" in question else "3 years"

    assert asyncio.run(answer("Warranty?")) == "3 years"
    assert asyncio.run(answer("Warranty? mail ana@example.com")).endswith("ana@example.com")  # untouched for the caller
    with pytest.raises(ZeroDivisionError):
        asyncio.run(answer("crash"))
    monitor.flush()

    conn = storage.connect(str(tmp_path / "prod.db"))
    [experiment_id] = storage.list_experiment_ids(conn)
    experiment = storage.load_experiment(conn, experiment_id)
    assert experiment.name.startswith("production 20") and experiment.dataset_name == "production"
    trials = storage.load_trials(conn, experiment_id)
    assert [t.failure_type for t in trials] == [None, None, "tool_error"]
    assert "ana@example.com" not in (trials[1].output or "") and "[EMAIL]" in trials[1].output
    cases = storage.load_cases(conn, experiment_id)
    assert [c.input for c in cases][0] == "Warranty?" and "ana@example.com" not in cases[1].input
    scorers = {s.scorer for s in storage.load_scores(conn, experiment_id)}
    assert "tool_execution" in scorers and "exact_match" not in scorers  # no expected answer in production

def test_unsampled_calls_are_not_recorded_and_a_chat_is_kept_as_a_conversation(tmp_path):
    from agenteval import storage
    from agenteval.monitor import Monitor

    never = Monitor(tmp_path / "none.db", model="m", price=(0, 0), sample_rate=0.0)
    assert never.watch(lambda q: "x")("q") == "x"
    never.flush()
    assert not (tmp_path / "none.db").exists() or not storage.list_experiment_ids(storage.connect(str(tmp_path / "none.db")))

    monitor = Monitor(tmp_path / "chat.db", model="m", price=(0, 0), sample_rate=1.0)
    chat = monitor.watch(lambda messages: "Your order shipped.")
    chat([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
          {"role": "user", "content": "Where is my order?"}])
    monitor.flush()
    conn = storage.connect(str(tmp_path / "chat.db"))
    trial = storage.load_trials(conn, storage.list_experiment_ids(conn)[0])[0]
    assert trial.transcript[-1] == {"role": "assistant", "content": "Your order shipped."}
    assert storage.load_cases(conn, storage.list_experiment_ids(conn)[0])[0].input == "Where is my order?"
    with pytest.raises(ValueError):
        Monitor(model="m", sample_rate=2)

def test_redaction_covers_every_message_of_a_conversation():
    from agenteval.privacy import redact_trial

    trial = Trial(id=1, case_id=1, attempt=1, output="ok",
                  transcript=[{"role": "user", "content": "I'm ana@example.com"}, {"role": "assistant", "content": "ok"}])
    assert redact_trial(trial).transcript[0]["content"] == "I'm [EMAIL]"

# --- Your own agent from the dashboard ---

def test_the_dashboard_runs_the_project_agent_named_in_agenteval_toml(tmp_path, monkeypatch):
    import sys

    from fastapi.testclient import TestClient

    from agenteval import storage, ui
    from agenteval.api import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "team_agent", raising=False)
    (tmp_path / "team_agent.py").write_text("def answer(question):\n    return 'Paris'\n", encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(
        '{"input": "Capital of France?", "expected_output": "Paris"}\n'
        '{"input": "Capital of Spain?", "expected_output": "Madrid"}\n', encoding="utf-8")
    (tmp_path / "agenteval.toml").write_text(
        '[run]\nagent = "team_agent:answer"\ndataset = "cases.jsonl"\nmodel = "my-model"\nprice = [0, 0]\n',
        encoding="utf-8")

    project = ui.project_agent()
    assert project == {"agent": "team_agent:answer", "dataset": "cases.jsonl", "model": "my-model", "price": [0, 0]}
    monkeypatch.setenv("AGENTEVAL_PROJECT_AGENT", json.dumps(project))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "agenteval.db"))
    monkeypatch.setenv("API_TOKEN", "t")
    monkeypatch.delenv("READ_ONLY", raising=False)
    try:
        with TestClient(app) as client:
            assert client.get("/api/run-options").json()["project"]["agent"] == "team_agent:answer"
            job = client.post("/api/runs", json={"agent": "project", "attempts": 1, "name": "from-ui"},
                              headers={"X-API-Token": "t"}).json()
            finished = client.get(f"/api/runs/{job['id']}").json()
            assert finished["status"] == "completed", finished["error"]
            summary = client.get(f"/api/experiments/{finished['experiment_id']}").json()
            assert summary["name"] == "from-ui" and summary["pass_rate"] == 0.5
    finally:
        sys.modules.pop("team_agent", None)
    conn = storage.connect(str(tmp_path / "agenteval.db"))
    assert storage.load_experiment(conn, finished["experiment_id"]).config.params["agent"] == "team_agent:answer"

def test_without_a_project_agent_the_api_runs_no_code_from_the_request(monkeypatch):
    from fastapi.testclient import TestClient

    from agenteval.api import app

    monkeypatch.delenv("AGENTEVAL_PROJECT_AGENT", raising=False)
    monkeypatch.setenv("API_TOKEN", "t")
    monkeypatch.delenv("READ_ONLY", raising=False)
    with TestClient(app) as client:
        assert client.get("/api/run-options").json() == {"project": None}
        refused = client.post("/api/runs", json={"agent": "project"}, headers={"X-API-Token": "t"})
        assert refused.status_code == 400
        named = client.post("/api/runs", json={"agent": "os:system"}, headers={"X-API-Token": "t"})
        assert named.status_code == 422
