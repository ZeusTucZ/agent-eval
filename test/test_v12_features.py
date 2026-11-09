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
