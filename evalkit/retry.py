import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from random import Random
from typing import TypeVar

from evalkit.tracer import Tracer

T = TypeVar("T")

# Labels an exception with a reason such as "rate_limit", or None if it isn't one we know.
# Whether that reason is worth retrying is up to RetryPolicy.retry_on.
Classifier = Callable[[BaseException], str | None]

# Reads how long the server asked us to wait, if it said (e.g. Gemini's retryDelay).
RetryAfter = Callable[[BaseException], float | None]

@dataclass
class RetryPolicy:
    max_retries: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: bool = True
    # Reasons that clear up by themselves. Others, e.g. "quota_exhausted", fail straight away.
    retry_on: tuple[str, ...] = ("rate_limit", "server_error", "connection_error")

    def delay(self, retry: int, random: Random) -> float:
        """Exponential backoff: base, 2x base, 4x base... capped at max_delay_s.

        Jitter picks a random point between half and all of that delay, so
        many clients that failed together don't all retry at the same instant.
        """
        delay = min(self.max_delay_s, self.base_delay_s * 2 ** retry)
        if self.jitter:
            delay *= random.uniform(0.5, 1.0)
        return delay

class RateLimiter:
    """Allows at most `requests_per_minute` calls in any 60-second window.

    Retries react to 429s after they happen; this avoids most of them by never
    sending faster than the quota. Shared by every trial using the same agent.
    """

    def __init__(
        self,
        requests_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.requests_per_minute = requests_per_minute
        self.clock = clock
        self.sleep = sleep
        self.sent: deque[float] = deque()  # when each request in the current window was sent
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            while True:
                now = self.clock()
                while self.sent and now - self.sent[0] >= 60:
                    self.sent.popleft()

                if len(self.sent) < self.requests_per_minute:
                    self.sent.append(now)
                    return

                # Wait until the oldest request leaves the window.
                await self.sleep(60 - (now - self.sent[0]))

async def call_llm_with_retries(
    tracer: Tracer,
    model: str,
    call: Callable[[], Awaitable[tuple[T, int, int]]],
    policy: RetryPolicy,
    classify: Classifier,
    random: Random,
    retry_after: RetryAfter | None = None,
    rate_limiter: RateLimiter | None = None,
) -> T:
    """Run an LLM call, retrying transient failures.

    `call` returns (result, input_tokens, output_tokens). Every attempt gets its
    own span, with `retry` set to 0 for the first try, so retries can be counted
    from the trace and each failed attempt keeps its reason as `error_type`.
    Waiting (for the rate limiter or between retries) happens outside the spans,
    so span time is time spent on the request itself.
    """
    for retry in range(policy.max_retries + 1):
        if rate_limiter is not None:
            await rate_limiter.acquire()

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
            if classify(error) not in policy.retry_on or retry == policy.max_retries:
                raise

            delay = policy.delay(retry, random)
            # If the server says how long to wait, waiting less just earns another 429.
            requested = retry_after(error) if retry_after is not None else None
            if requested is not None:
                delay = min(max(delay, requested), policy.max_delay_s)
            await asyncio.sleep(delay)

    raise AssertionError("unreachable")
