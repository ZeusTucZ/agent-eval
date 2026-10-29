import json
import os
from random import Random
from typing import Any

import openai

from evalkit.models import EvaluationCase
from evalkit.retry import RateLimiter, RetryPolicy, call_llm_with_retries
from evalkit.runner import MaxStepsExceeded, QuotaExhausted
from evalkit.tools import DEFAULT_TOOLS, SYSTEM_PROMPT, ToolRegistry
from evalkit.tracer import Tracer

# Providers that speak the OpenAI chat-completions API: (base URL, environment variable with the key).
PROVIDERS: dict[str, tuple[str, str | None]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),  # runs on your own computer; no key
}

def classify_openai_error(error: BaseException) -> str | None:
    if isinstance(error, openai.RateLimitError):
        # Groq says which limit was hit, e.g. "... on requests per day (RPD): Limit 1000, Used 1000".
        # A daily limit won't clear by retrying; a per-minute one will.
        return "quota_exhausted" if "per day" in str(error).lower() else "rate_limit"
    if isinstance(error, openai.APIConnectionError):  # includes timeouts
        return "connection_error"
    if isinstance(error, openai.APIStatusError) and error.status_code >= 500:
        return "server_error"
    return None

def openai_retry_after(error: BaseException) -> float | None:
    """Rate-limit responses carry a standard retry-after header, in seconds."""
    if not isinstance(error, openai.APIStatusError):
        return None
    try:
        return float(error.response.headers.get("retry-after", ""))
    except ValueError:
        return None

class OpenAICompatibleAgent:
    """The same agent loop as GeminiAgent, for any provider with an OpenAI-style API (Groq, Ollama...)."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        provider: str = "groq",
        tools: ToolRegistry | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = 5,
        temperature: float = 0.0,
        retry_policy: RetryPolicy | None = None,
        requests_per_minute: int | None = None,
        client: Any = None,  # tests pass a fake client
    ):
        self.model = model
        self.provider = provider
        self.tools = tools or ToolRegistry(DEFAULT_TOOLS)
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.temperature = temperature
        self.retry_policy = retry_policy or RetryPolicy(max_retries=5, base_delay_s=2.0, max_delay_s=60.0)
        self.requests_per_minute = requests_per_minute
        self.rate_limiter = RateLimiter(requests_per_minute) if requests_per_minute else None
        self.quota_exhausted = False  # circuit breaker, as in GeminiAgent
        self.client = client or self._build_client()

    def _build_client(self) -> openai.AsyncOpenAI:
        base_url, key_variable = PROVIDERS[self.provider]
        api_key = os.environ[key_variable] if key_variable else "not-needed"
        # max_retries=0: the SDK would otherwise retry 429s itself, invisibly. Our wrapper
        # does the retrying so every attempt is traced and counted.
        return openai.AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=60.0)

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.arguments.model_json_schema(),
                },
            }
            for tool in self.tools.tools.values()
        ]

    async def run(self, case: EvaluationCase, attempt: int, tracer: Tracer) -> str:
        if self.quota_exhausted:
            raise QuotaExhausted("skipped: the daily quota ran out earlier in this experiment")

        random = Random(f"{case.id}-{attempt}")  # only used for retry jitter
        tools = self._tool_definitions()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": case.input},
        ]

        async def generate() -> tuple[Any, int, int]:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, temperature=self.temperature,
            )
            usage = response.usage
            return response, (usage and usage.prompt_tokens) or 0, (usage and usage.completion_tokens) or 0

        for _ in range(self.max_steps):
            try:
                response = await call_llm_with_retries(
                    tracer, self.model, generate, self.retry_policy, classify_openai_error, random,
                    retry_after=openai_retry_after, rate_limiter=self.rate_limiter,
                )
            except openai.RateLimitError as error:
                if classify_openai_error(error) == "quota_exhausted":
                    self.quota_exhausted = True
                    raise QuotaExhausted(str(error)) from error
                raise

            message = response.choices[0].message
            if not message.tool_calls:
                return (message.content or "").strip()

            # Echo the assistant turn back by hand: some providers reject extra fields
            # (such as a reasoning trace) that the SDK's message object carries.
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.function.name, "arguments": call.function.arguments}}
                    for call in message.tool_calls
                ],
            })
            for call in message.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(self._call_tool(call.function.name, call.function.arguments, tracer)),
                })

        raise MaxStepsExceeded(f"no final answer after {self.max_steps} model calls")

    def _call_tool(self, name: str, arguments: str, tracer: Tracer) -> dict[str, Any]:
        # As in GeminiAgent: a failed tool goes back to the model instead of ending the trial.
        try:
            return {"result": self.tools.call(name, arguments, tracer)}
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}
