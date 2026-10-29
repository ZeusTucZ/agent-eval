import asyncio
import re
from dataclasses import dataclass
from random import Random

from evalkit.models import EvaluationCase
from evalkit.tracer import Tracer

@dataclass
class FaultConfig:
    tool_error_rate: float = 0.1
    rate_limit_rate: float = 0.05  # not used yet: needs the retry wrapper (Phase 2)
    hallucination_rate: float = 0.15
    hang_rate: float = 0.0  # the LLM call never returns; the runner's timeout must stop it
    latency_ms: tuple[int, int] = (50, 300)
    seed: int = 42

class ToolError(Exception):
    pass

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

    def __init__(self, config: FaultConfig, model: str = "mock-llm"):
        self.config = config
        self.model = model

    async def run(self, case: EvaluationCase, attempt: int, tracer: Tracer) -> str:
        # One generator per trial, seeded from what the trial *is* rather than
        # when it runs. A shared generator would give different results each run
        # once trials run concurrently, because the order they draw from it changes.
        random = Random(f"{self.config.seed}-{case.id}-{attempt}")

        with tracer.span(kind="llm", name=self.model) as span:
            usage = await self._fake_llm_call(random)
            span.input_tokens = usage["input_tokens"]
            span.output_tokens = usage["output_tokens"]

        a, operator, b = self._parse(case.input)

        with tracer.span(kind="tool", name="calculator"):
            result = calculator(a, operator, b, random, self.config)

        return str(self._maybe_hallucinate(result, random))

    async def _fake_llm_call(self, random: Random) -> dict[str, int]:
        low, high = self.config.latency_ms
        await asyncio.sleep(random.uniform(low, high) / 1000)

        if random.random() < self.config.hang_rate:
            await asyncio.sleep(3600)

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
