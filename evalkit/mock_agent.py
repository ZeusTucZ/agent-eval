import re
import time
from dataclasses import dataclass
from random import Random

from evalkit.models import EvaluationCase
from evalkit.tracer import Tracer

@dataclass
class FaultConfig:
    tool_error_rate: float = 0.1
    rate_limit_rate: float = 0.05  # not used yet: needs the retry wrapper (Phase 2)
    hallucination_rate: float = 0.15
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

    def __init__(self, config: FaultConfig):
        self.config = config
        self.random = Random(config.seed)

    def run(self, case: EvaluationCase, tracer: Tracer) -> str:
        with tracer.span(kind="llm", name="mock-llm") as span:
            usage = self._fake_llm_call()
            span.input_tokens = usage["input_tokens"]
            span.output_tokens = usage["output_tokens"]

        a, operator, b = self._parse(case.input)

        with tracer.span(kind="tool", name="calculator"):
            result = calculator(a, operator, b, self.random, self.config)

        return str(self._maybe_hallucinate(result))

    def _fake_llm_call(self) -> dict[str, int]:
        low, high = self.config.latency_ms
        time.sleep(self.random.uniform(low, high) / 1000)

        return {
            "input_tokens": self.random.randint(50, 150),
            "output_tokens": self.random.randint(10, 40),
        }

    def _parse(self, question: str) -> tuple[int, str, int]:
        match = re.search(r"(-?\d+)\s*([+*])\s*(-?\d+)", question)
        if match is None:
            raise ValueError(f"cannot parse question: {question!r}")
        return int(match.group(1)), match.group(2), int(match.group(3))

    def _maybe_hallucinate(self, correct: int) -> int:
        if self.random.random() < self.config.hallucination_rate:
            return correct + self.random.randint(1, 100)
        return correct
