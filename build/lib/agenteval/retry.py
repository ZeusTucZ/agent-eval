import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from random import Random
from typing import TypeVar

from agenteval.tracer import Tracer

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
    """Spaces requests evenly to stay under `requests_per_minute`, and optionally under
    `tokens_per_minute` too.

    Retries react to 429s after they happen; this avoids most of them by never
    sending faster than the quota. Shared by every trial using the same agent.

    Requests go out one every 60 / requests_per_minute seconds rather than all at
    once: an earlier version allowed the whole minute's quota in a burst and then
    waited, which put most of a run's p95 latency in the limiter itself.

    Providers also cap tokens per minute, and a request's tokens are only known once
    it returns. So each finished call reports its tokens with `record`, and a request
    waits while the tokens of the last 60 s plus a typical request's would pass the cap.
    """

    def __init__(
        self,
        requests_per_minute: int | None,
        tokens_per_minute: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.interval = 60 / requests_per_minute if requests_per_minute else 0.0  # None: no request cap
        self.clock = clock
        self.sleep = sleep
        self.next_send: float | None = None  # the earliest the next request may go
        self.used: deque[tuple[float, int]] = deque()  # (when, tokens) for calls in the last 60 s
        self.recent: deque[int] = deque(maxlen=20)  # tokens of the latest calls, to estimate the next
        self.lock = asyncio.Lock()

    def expected_tokens(self) -> float:
        """What the next request will likely use: the mean of recent ones, 0 before any."""
        return sum(self.recent) / len(self.recent) if self.recent else 0.0

    def record(self, tokens: int) -> None:
        """Report a finished call's tokens (input and output)."""
        self.used.append((self.clock(), tokens))
        self.recent.append(tokens)

    async def acquire(self) -> None:
        async with self.lock:
            now = self.clock()
            if self.next_send is not None and now < self.next_send:
                await self.sleep(self.next_send - now)
                now = self.clock()

            if self.tokens_per_minute is not None:
                while True:
                    while self.used and now - self.used[0][0] >= 60:
                        self.used.popleft()
                    in_window = sum(tokens for _, tokens in self.used)
                    # An empty window always lets one through, or a request bigger than
                    # the whole budget would wait forever.
                    if not self.used or in_window + self.expected_tokens() <= self.tokens_per_minute:
                        break
                    # Wait until the oldest call leaves the window.
                    await self.sleep(60 - (now - self.used[0][0]))
                    now = self.clock()

            self.next_send = now + self.interval

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
            if rate_limiter is not None:
                rate_limiter.record(span.input_tokens + span.output_tokens)
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
