import asyncio
from random import Random

from google.genai import errors

from evalkit.gemini_agent import gemini_retry_after
from evalkit.models import Span, Trial
from evalkit.report import latency_percentiles
from evalkit.retry import RateLimiter, RetryPolicy, call_llm_with_retries
from evalkit.tracer import Tracer

class FakeTime:
    """A clock that only moves when something sleeps, so tests are instant and exact."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

def test_rate_limiter_waits_once_the_minute_is_used_up():
    time = FakeTime()
    limiter = RateLimiter(requests_per_minute=3, clock=time.clock, sleep=time.sleep)

    async def send(count):
        sent_at = []
        for _ in range(count):
            await limiter.acquire()
            sent_at.append(time.now)
        return sent_at

    # Three go straight away; the 4th waits for the 1st to be 60 s old.
    assert asyncio.run(send(5)) == [0, 0, 0, 60, 60]

def test_rate_limiter_frees_slots_as_time_passes():
    time = FakeTime()
    limiter = RateLimiter(requests_per_minute=2, clock=time.clock, sleep=time.sleep)

    async def scenario():
        await limiter.acquire()            # t=0
        time.now = 45
        await limiter.acquire()            # t=45
        await limiter.acquire()            # window full: wait until t=60
        return time.now

    assert asyncio.run(scenario()) == 60
    assert time.sleeps == [15]

def rate_limited(retry_delay: str | None) -> errors.ClientError:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}] if retry_delay else []
    return errors.ClientError(429, {"error": {"code": 429, "message": "quota", "details": details}})

def test_reads_gemini_retry_delay():
    assert gemini_retry_after(rate_limited("37s")) == 37.0
    assert gemini_retry_after(rate_limited("1.5s")) == 1.5
    assert gemini_retry_after(rate_limited(None)) is None
    assert gemini_retry_after(ValueError("not an API error")) is None

def test_retry_waits_at_least_as_long_as_the_server_asks(monkeypatch):
    waits = []

    async def record_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    responses = [ValueError("429"), ValueError("429"), ("ok", 1, 1)]

    async def call():
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=30.0, jitter=False)
    result = asyncio.run(call_llm_with_retries(
        Tracer(1), "model", call, policy,
        classify=lambda error: "rate_limit", random=Random(0),
        retry_after=lambda error: 10.0,
    ))

    assert result == "ok"
    # Backoff alone would wait 1 s then 2 s; the server asked for 10 s each time.
    assert waits == [10.0, 10.0]

def test_working_time_leaves_out_waits_between_spans():
    ms = 1_000_000
    trial = Trial(id=1, case_id=1, attempt=1, spans=[
        Span(trial_id=1, kind="llm", name="m", start=0, end=100 * ms, status="error", error_type="rate_limit"),
        # 5 s of backoff here
        Span(trial_id=1, kind="llm", name="m", start=5100 * ms, end=5400 * ms, status="success", retry=1),
    ])

    assert trial.duration_ms == 5400
    assert trial.working_ms == 400
    assert latency_percentiles([trial, trial], working_only=True)["p50"] == 400
