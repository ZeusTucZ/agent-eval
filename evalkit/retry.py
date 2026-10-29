import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from random import Random
from typing import TypeVar

from evalkit.tracer import Tracer

T = TypeVar("T")

# Turns an exception into a retry reason such as "rate_limit", or None if retrying won't help.
Classifier = Callable[[BaseException], str | None]

@dataclass
class RetryPolicy:
    max_retries: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: bool = True

    def delay(self, retry: int, random: Random) -> float:
        """Exponential backoff: base, 2x base, 4x base... capped at max_delay_s.

        Jitter picks a random point between half and all of that delay, so
        many clients that failed together don't all retry at the same instant.
        """
        delay = min(self.max_delay_s, self.base_delay_s * 2 ** retry)
        if self.jitter:
            delay *= random.uniform(0.5, 1.0)
        return delay

async def call_llm_with_retries(
    tracer: Tracer,
    model: str,
    call: Callable[[], Awaitable[tuple[T, int, int]]],
    policy: RetryPolicy,
    classify: Classifier,
    random: Random,
) -> T:
    """Run an LLM call, retrying transient failures.

    `call` returns (result, input_tokens, output_tokens). Every attempt gets its
    own span, with `retry` set to 0 for the first try, so retries can be counted
    from the trace and each failed attempt keeps its reason as `error_type`.
    """
    for retry in range(policy.max_retries + 1):
        try:
            with tracer.span(kind="llm", name=model) as span:
                span.retry = retry
                try:
                    result, span.input_tokens, span.output_tokens = await call()
                except Exception as error:
                    span.error_type = classify(error)
                    raise
            return result
        except Exception as error:
            if classify(error) is None or retry == policy.max_retries:
                raise
            await asyncio.sleep(policy.delay(retry, random))

    raise AssertionError("unreachable")
