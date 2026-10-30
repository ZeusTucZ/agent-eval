import os
from random import Random
from typing import Any

from google import genai
from google.genai import errors, types

from evalkit.models import EvaluationCase
from evalkit.retry import RateLimiter, RetryPolicy, call_llm_with_retries
from evalkit.runner import MaxStepsExceeded, QuotaExhausted
from evalkit.tools import DEFAULT_TOOLS, SYSTEM_PROMPT, ToolRegistry
from evalkit.tracer import Tracer

def _error_details(error: errors.APIError) -> list[dict[str, Any]]:
    if not isinstance(error.details, dict):
        return []
    return error.details.get("error", {}).get("details", [])

def _is_daily_quota(error: errors.APIError) -> bool:
    """A 429 whose QuotaFailure names a per-day quota, e.g. "GenerateRequestsPerDayPerProjectPerModel-FreeTier".

    Per-minute limits clear up within a minute; a daily one won't clear until it resets.
    """
    for detail in _error_details(error):
        if detail.get("@type", "").endswith("google.rpc.QuotaFailure"):
            for violation in detail.get("violations", []):
                if "PerDay" in violation.get("quotaId", ""):
                    return True
    return False

def classify_gemini_error(error: BaseException) -> str | None:
    if isinstance(error, errors.APIError):
        if error.code == 429:
            return "quota_exhausted" if _is_daily_quota(error) else "rate_limit"
        if error.code >= 500:
            return "server_error"
    return None

def gemini_retry_after(error: BaseException) -> float | None:
    """Gemini's 429s include a RetryInfo detail such as {"retryDelay": "37s"}."""
    if not isinstance(error, errors.APIError):
        return None

    for detail in _error_details(error):
        if detail.get("@type", "").endswith("google.rpc.RetryInfo"):
            delay = str(detail.get("retryDelay", ""))
            try:
                return float(delay.removesuffix("s"))
            except ValueError:
                return None
    return None

class GeminiAgent:
    """A real agent loop: model -> tool calls -> model -> ... -> final answer."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        tools: ToolRegistry | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = 5,
        temperature: float = 0.0,
        retry_policy: RetryPolicy | None = None,
        requests_per_minute: int | None = None,  # set to your quota to avoid most 429s
        client: Any = None,  # tests pass a fake client; otherwise one is built from GEMINI_API_KEY
    ):
        self.model = model
        self.tools = tools or ToolRegistry(DEFAULT_TOOLS)
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.temperature = temperature
        self.retry_policy = retry_policy or RetryPolicy(max_retries=5, base_delay_s=2.0, max_delay_s=60.0)
        self.requests_per_minute = requests_per_minute
        self.rate_limiter = RateLimiter(requests_per_minute) if requests_per_minute else None
        # Circuit breaker: once the daily quota is gone, every request would fail until it
        # resets, so later trials fail immediately instead of waiting on hopeless retries.
        self.quota_exhausted = False
        self.client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _generation_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.temperature,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.arguments.model_json_schema(),
                )
                for tool in self.tools.tools.values()
            ])],
            # We run the tool loop ourselves so every call is traced and validated.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def run(self, case: EvaluationCase, attempt: int, tracer: Tracer) -> str:
        if self.quota_exhausted:
            raise QuotaExhausted("skipped: the daily quota ran out earlier in this experiment")

        random = Random(f"{case.id}-{attempt}")  # only used for retry jitter
        config = self._generation_config()
        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part.from_text(text=case.input)]),
        ]

        async def generate() -> tuple[types.GenerateContentResponse, int, int]:
            response = await self.client.aio.models.generate_content(
                model=self.model, contents=contents, config=config,
            )
            usage = response.usage_metadata
            input_tokens = (usage and usage.prompt_token_count) or 0
            # Thinking tokens are billed as output tokens.
            output_tokens = ((usage and usage.candidates_token_count) or 0) + ((usage and usage.thoughts_token_count) or 0)
            return response, input_tokens, output_tokens

        for _ in range(self.max_steps):
            try:
                response = await call_llm_with_retries(
                    tracer, self.model, generate, self.retry_policy, classify_gemini_error, random,
                    retry_after=gemini_retry_after, rate_limiter=self.rate_limiter,
                )
            except errors.APIError as error:
                if classify_gemini_error(error) == "quota_exhausted":
                    self.quota_exhausted = True
                    raise QuotaExhausted(str(error)) from error
                raise

            function_calls = response.function_calls or []
            if not function_calls:
                return (response.text or "").strip()

            contents.append(response.candidates[0].content)
            contents.append(types.Content(role="user", parts=[
                types.Part.from_function_response(
                    name=call.name,
                    response=self._call_tool(call.name, dict(call.args or {}), tracer),
                )
                for call in function_calls
            ]))

        raise MaxStepsExceeded(f"no final answer after {self.max_steps} model calls")

    def _call_tool(self, name: str, arguments: dict[str, Any], tracer: Tracer) -> dict[str, Any]:
        # A failed tool doesn't end the trial: the error goes back to the model,
        # which can correct itself. The tool span still records the failure.
        try:
            return {"result": self.tools.call(name, arguments, tracer)}
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}
