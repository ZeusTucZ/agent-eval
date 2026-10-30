import asyncio
import re
from dataclasses import dataclass
from random import Random

from evalkit.models import EvaluationCase
from evalkit.retry import RetryPolicy, call_llm_with_retries
from evalkit.tracer import Tracer

@dataclass
class FaultConfig:
    tool_error_rate: float = 0.1
    rate_limit_rate: float = 0.05  # each LLM call fails with a 429 at this rate and is retried
    hallucination_rate: float = 0.15
    hang_rate: float = 0.0  # the LLM call never returns; the runner's timeout must stop it
    latency_ms: tuple[int, int] = (50, 300)
    seed: int = 42

class ToolError(Exception):
    pass

class RateLimitError(Exception):
    pass

def classify_mock_error(error: BaseException) -> str | None:
    return "rate_limit" if isinstance(error, RateLimitError) else None

def calculator(a: int, operator: str, b: int, random: Random, config: FaultConfig) -> int:
    if random.random() < config.tool_error_rate:
        raise ToolError("calculator unavailable")

    if operator == "+":
        return a + b
    if operator == "*":
        return a * b
    raise ValueError(f"unsupported operator: {operator}")

class MockAgent:
    """Answers "What is A + B" questions. Handles one case per call."""

    def __init__(self, config: FaultConfig, model: str = "mock-llm", retry_policy: RetryPolicy | None = None):
        self.config = config
        self.model = model
        # Millisecond backoff: the mock's rate limits are simulated, so there is nothing to wait for.
        self.retry_policy = retry_policy or RetryPolicy(base_delay_s=0.001, max_delay_s=0.01)

    async def run(self, case: EvaluationCase, attempt: int, tracer: Tracer) -> str:
        # One generator per trial, seeded from what the trial *is* rather than
        # when it runs. A shared generator would give different results each run
        # once trials run concurrently, because the order they draw from it changes.
        random = Random(f"{self.config.seed}-{case.id}-{attempt}")

        async def llm_call() -> tuple[None, int, int]:
            usage = await self._fake_llm_call(random)
            return None, usage["input_tokens"], usage["output_tokens"]

        await call_llm_with_retries(tracer, self.model, llm_call, self.retry_policy, classify_mock_error, random)

        a, operator, b = self._parse(case.input)

        with tracer.span(kind="tool", name="calculator"):
            result = calculator(a, operator, b, random, self.config)

        return str(self._maybe_hallucinate(result, random))

    async def _fake_llm_call(self, random: Random) -> dict[str, int]:
        low, high = self.config.latency_ms
        await asyncio.sleep(random.uniform(low, high) / 1000)

        if random.random() < self.config.hang_rate:
            await asyncio.sleep(3600)

        if random.random() < self.config.rate_limit_rate:
            raise RateLimitError("429: too many requests")

        return {
            "input_tokens": random.randint(50, 150),
            "output_tokens": random.randint(10, 40),
        }

    def _parse(self, question: str) -> tuple[int, str, int]:
        match = re.search(r"(-?\d+)\s*([+*])\s*(-?\d+)", question)
        if match is None:
            raise ValueError(f"cannot parse question: {question!r}")
        return int(match.group(1)), match.group(2), int(match.group(3))

    def _maybe_hallucinate(self, correct: int, random: Random) -> int:
        if random.random() < self.config.hallucination_rate:
            return correct + random.randint(1, 100)
        return correct
