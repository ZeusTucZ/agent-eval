"""Tests for the OpenAI-compatible agent (Groq, Ollama) against a fake client: no key, no network.

Responses and errors are built with the real openai SDK types, so the agent reads them
exactly as it would read real ones.
"""
import asyncio
import json
from types import SimpleNamespace

import httpx2
import openai
from openai.types.chat import ChatCompletion

from evalkit.models import EvaluationCase
from evalkit.openai_agent import OpenAICompatibleAgent, classify_openai_error, openai_retry_after
from evalkit.retry import RetryPolicy
from evalkit.runner import run_experiment
from evalkit.scorers import score_trials

class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def create(self, model, messages, tools, temperature):
        self.requests.append(json.loads(json.dumps(messages)))  # a snapshot, as the API would see it
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

def fake_client(*responses):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(responses)))

def completion(message: dict, prompt_tokens: int = 120, completion_tokens: int = 30) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "openai/gpt-oss-20b",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    })

def tool_call(name: str, arguments: str, call_id: str = "call_1") -> ChatCompletion:
    return completion({"content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}},
    ]})

def final_answer(text: str) -> ChatCompletion:
    return completion({"content": text}, prompt_tokens=200, completion_tokens=5)

def rate_limited(message: str, retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx2.Response(429, headers=headers,
                               request=httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return openai.RateLimitError(message, response=response, body={"error": {"message": message}})

PER_MINUTE = "Rate limit reached for model `openai/gpt-oss-20b` on requests per minute (RPM): Limit 30, Used 30"
PER_DAY = "Rate limit reached for model `openai/gpt-oss-20b` on requests per day (RPD): Limit 1000, Used 1000"

NO_WAIT = RetryPolicy(base_delay_s=0.0, jitter=False)

case = EvaluationCase(id=1, input="What is 48213 * 9917?", expected_output="478128321", expected_tool="calculator")

def run(agent, cases=(case,)):
    trials = asyncio.run(run_experiment(agent, list(cases), attempts=1))
    return trials, score_trials(list(cases), trials)

def agent_with(*responses, **options):
    client = fake_client(*responses)
    return OpenAICompatibleAgent(client=client, retry_policy=NO_WAIT, **options), client.chat.completions

def test_tool_loop_calls_the_tool_and_returns_the_final_answer():
    agent, completions = agent_with(
        tool_call("calculator", '{"a": 48213, "operator": "*", "b": 9917}'),
        final_answer("478128321"),
    )
    trials, scores = run(agent)

    trial = trials[0]
    assert trial.output == "478128321"
    assert [(span.kind, span.name) for span in trial.spans] == [
        ("llm", "openai/gpt-oss-20b"), ("tool", "calculator"), ("llm", "openai/gpt-oss-20b"),
    ]
    assert [(span.input_tokens, span.output_tokens) for span in trial.spans if span.kind == "llm"] == [(120, 30), (200, 5)]
    assert {score.scorer: score.value for score in scores} == {
        "exact_match": 1.0, "tool_selection": 1.0, "tool_arguments_valid": 1.0, "tool_execution": 1.0,
    }

    # The second request holds the system prompt, the question, the assistant's tool call
    # and the tool's result, linked by the call id.
    second = completions.requests[1]
    assert [message["role"] for message in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["id"] == second[3]["tool_call_id"] == "call_1"
    assert json.loads(second[3]["content"]) == {"result": 478128321}

def test_malformed_json_arguments_count_as_invalid_and_go_back_to_the_model():
    agent, completions = agent_with(
        tool_call("calculator", '{"a": 48213, "operator": "*", "b": '),  # cut off mid-JSON
        tool_call("calculator", '{"a": 48213, "operator": "*", "b": 9917}', call_id="call_2"),
        final_answer("478128321"),
    )
    trials, scores = run(agent)

    assert trials[0].output == "478128321"
    assert trials[0].spans[1].error_type == "ToolArgumentError"
    assert "not valid JSON" in json.loads(completions.requests[1][3]["content"])["error"]
    assert {score.scorer: score.value for score in scores}["tool_arguments_valid"] == 0.0

def test_per_minute_rate_limits_are_retried_using_retry_after():
    agent, _ = agent_with(rate_limited(PER_MINUTE, retry_after="0"), final_answer("478128321"))
    trials, _ = run(agent)

    assert trials[0].output == "478128321"
    assert [(span.retry, span.error_type) for span in trials[0].spans] == [(0, "rate_limit"), (1, None)]
    assert "requests per minute" in trials[0].spans[0].error_message

def test_daily_limit_stops_the_experiment_early():
    agent, completions = agent_with(rate_limited(PER_DAY), final_answer("never sent"))
    dataset = [case.model_copy(update={"id": index}) for index in range(1, 6)]
    trials, _ = run(agent, dataset)

    assert [trial.failure_type for trial in trials] == ["quota_exhausted"] * 5
    assert len(completions.requests) == 1

def test_error_classification():
    assert classify_openai_error(rate_limited(PER_MINUTE)) == "rate_limit"
    assert classify_openai_error(rate_limited(PER_DAY)) == "quota_exhausted"
    assert classify_openai_error(openai.APIConnectionError(request=httpx2.Request("POST", "https://x"))) == "connection_error"
    assert classify_openai_error(ValueError("other")) is None

    assert openai_retry_after(rate_limited(PER_MINUTE, retry_after="7")) == 7.0
    assert openai_retry_after(rate_limited(PER_MINUTE)) is None

def test_the_sdk_does_not_retry_on_its_own():
    # Hidden SDK retries would make our retry counts and latencies wrong.
    agent = OpenAICompatibleAgent(provider="ollama")
    assert agent.client.max_retries == 0
    assert str(agent.client.base_url).startswith("http://localhost:11434")
