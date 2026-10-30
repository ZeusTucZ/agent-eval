import asyncio
from random import Random

from google.genai import errors

from evalkit.gemini_agent import GeminiAgent, classify_gemini_error, gemini_retry_after
from evalkit.mock_agent import FaultConfig, MockAgent
from evalkit.models import EvaluationCase, Span, Trial
from evalkit.runner import run_experiment
from evalkit.sample_data import arithmetic_cases
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

# --- Error messages, daily quotas and timeout labels ---

def quota_error(quota_id: str) -> errors.ClientError:
    return errors.ClientError(429, {"error": {
        "code": 429,
        "message": f"You exceeded your current quota. Quota: {quota_id}",
        "details": [{
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                            "quotaId": quota_id}],
        }],
    }})

DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"

def test_daily_quota_is_told_apart_from_per_minute_limits():
    assert classify_gemini_error(quota_error(DAILY)) == "quota_exhausted"
    assert classify_gemini_error(quota_error(PER_MINUTE)) == "rate_limit"
    assert classify_gemini_error(rate_limited("37s")) == "rate_limit"  # no QuotaFailure detail

def test_spans_keep_the_error_message():
    tracer = Tracer(1)
    try:
        with tracer.span(kind="llm", name="m"):
            raise ValueError("which quota was hit")
    except ValueError:
        pass
    assert tracer.spans[0].error_message == "which quota was hit"

def test_reasons_outside_retry_on_are_not_retried():
    calls = 0

    async def call():
        nonlocal calls
        calls += 1
        raise ValueError("daily quota")

    tracer = Tracer(1)
    try:
        asyncio.run(call_llm_with_retries(
            tracer, "m", call, RetryPolicy(base_delay_s=0.0),
            classify=lambda error: "quota_exhausted", random=Random(0),
        ))
    except ValueError:
        pass

    assert calls == 1
    assert tracer.spans[0].error_type == "quota_exhausted"

def test_daily_quota_stops_the_experiment_early():
    from test_gemini_agent import NO_WAIT, fake_client, final_answer

    client = fake_client(quota_error(DAILY), final_answer("never sent"))
    dataset = [
        EvaluationCase(id=index, input="What is 1 + 1?", expected_output="2", expected_tool="calculator")
        for index in range(1, 6)
    ]
    trials = asyncio.run(run_experiment(GeminiAgent(client=client, retry_policy=NO_WAIT), dataset, attempts=1))

    assert [trial.failure_type for trial in trials] == ["quota_exhausted"] * 5
    assert len(client.aio.models.requests) == 1  # one request, then the circuit breaker opened
    assert DAILY in trials[0].spans[0].error_message

def test_timeout_while_waiting_to_retry_names_the_real_cause():
    # Every call is rate-limited and the backoff is longer than the timeout, so the
    # trial runs out of time between attempts, not inside a call.
    config = FaultConfig(tool_error_rate=0.0, rate_limit_rate=1.0, hallucination_rate=0.0, latency_ms=(0, 0))
    agent = MockAgent(config, retry_policy=RetryPolicy(base_delay_s=10.0, jitter=False))
    trials = asyncio.run(run_experiment(agent, arithmetic_cases(2), attempts=1, timeout_s=0.1))

    assert [trial.failure_type for trial in trials] == ["timeout_after_rate_limit"] * 2

def test_a_step_that_hangs_is_still_a_plain_timeout():
    config = FaultConfig(tool_error_rate=0.0, rate_limit_rate=0.0, hallucination_rate=0.0, hang_rate=1.0, latency_ms=(0, 0))
    trials = asyncio.run(run_experiment(MockAgent(config), arithmetic_cases(2), attempts=1, timeout_s=0.05))

    assert [trial.failure_type for trial in trials] == ["timeout"] * 2
