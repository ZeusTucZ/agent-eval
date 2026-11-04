"""Tests for the Gemini agent loop against a fake client: no API key, no network, no cost.

The responses are built with the real SDK types, so the agent reads them exactly as it
would read real ones.
"""
import asyncio
from types import SimpleNamespace

from google.genai import errors, types

from evalkit.gemini_agent import GeminiAgent
from evalkit.models import EvaluationCase
from evalkit.retry import RetryPolicy
from evalkit.runner import run_experiment
from evalkit.scorers import score_trials

class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def generate_content(self, model, contents, config):
        self.requests.append(list(contents))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

def fake_client(*responses):
    return SimpleNamespace(aio=SimpleNamespace(models=FakeModels(responses)))

def usage(prompt: int, output: int, thoughts: int = 0):
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt, candidates_token_count=output, thoughts_token_count=thoughts,
    )

def function_call(name: str, **arguments) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=arguments))],
        ))],
        usage_metadata=usage(100, 20),
    )

def final_answer(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))],
        usage_metadata=usage(150, 5, thoughts=30),
    )

def rate_limited() -> errors.ClientError:
    return errors.ClientError(429, {"error": {"code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}})

NO_WAIT = RetryPolicy(base_delay_s=0.0, jitter=False)

case = EvaluationCase(id=1, input="What is 48213 * 9917?", expected_output="478128321", expected_tool="calculator")

def run(agent: GeminiAgent, cases=(case,)):
    trials = asyncio.run(run_experiment(agent, list(cases), attempts=1))
    return trials, score_trials(list(cases), trials)

def test_tool_loop_calls_the_tool_and_returns_the_final_answer():
    client = fake_client(function_call("calculator", a=48213, operator="*", b=9917), final_answer("478128321"))
    trials, scores = run(GeminiAgent(client=client, retry_policy=NO_WAIT))

    trial = trials[0]
    assert trial.output == "478128321"
    assert [(span.kind, span.name) for span in trial.spans] == [
        ("llm", "gemini-2.5-flash"), ("tool", "calculator"), ("llm", "gemini-2.5-flash"),
    ]
    # Thinking tokens count as output tokens.
    assert [(span.input_tokens, span.output_tokens) for span in trial.spans if span.kind == "llm"] == [(100, 20), (150, 35)]
    assert {score.scorer: score.value for score in scores} == {
        "exact_match": 1.0, "tool_selection": 1.0, "tool_arguments_valid": 1.0, "tool_execution": 1.0,
        "tool_faithfulness": 1.0,
    }

    # The second request sent the model its own call plus the tool's result.
    second_request = client.aio.models.requests[1]
    assert second_request[2].parts[0].function_response.response == {"result": 478128321}

def test_rate_limits_are_retried_and_recorded():
    client = fake_client(rate_limited(), rate_limited(), final_answer("478128321"))
    trials, _ = run(GeminiAgent(client=client, retry_policy=NO_WAIT))

    spans = trials[0].spans
    assert trials[0].output == "478128321"
    assert [(span.retry, span.status, span.error_type) for span in spans] == [
        (0, "error", "rate_limit"), (1, "error", "rate_limit"), (2, "success", None),
    ]

def test_tool_errors_go_back_to_the_model_instead_of_ending_the_trial():
    client = fake_client(
        function_call("calculator", a=48213, operator="x", b=9917),  # invalid operator
        function_call("calculator", a=48213, operator="*", b=9917),
        final_answer("478128321"),
    )
    trials, scores = run(GeminiAgent(client=client, retry_policy=NO_WAIT))

    assert trials[0].failure_type is None
    assert trials[0].output == "478128321"
    assert "ToolArgumentError" in client.aio.models.requests[1][2].parts[0].function_response.response["error"]

    by_scorer = {score.scorer: score.value for score in scores}
    assert by_scorer["tool_arguments_valid"] == 0.0  # the bad first call is still counted
    assert by_scorer["tool_execution"] == 1.0

def test_answering_from_memory_is_caught_by_tool_selection():
    client = fake_client(final_answer("478128321"))
    _, scores = run(GeminiAgent(client=client, retry_policy=NO_WAIT))

    assert {score.scorer: score.value for score in scores} == {
        "exact_match": 1.0, "tool_selection": 0.0, "tool_faithfulness": 0.0,
    }

def test_endless_tool_calls_stop_at_max_steps():
    client = fake_client(*[function_call("calculator", a=1, operator="+", b=1) for _ in range(3)])
    trials, _ = run(GeminiAgent(client=client, max_steps=3, retry_policy=NO_WAIT))

    assert trials[0].failure_type == "max_steps_exceeded"
    assert trials[0].output is None

def test_an_empty_answer_is_a_failure_not_a_silent_blank():
    trials, _ = run(GeminiAgent(client=fake_client(final_answer("   ")), retry_policy=NO_WAIT))

    assert trials[0].failure_type == "empty_output"
