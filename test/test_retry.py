import asyncio
from random import Random

import pytest

from evalkit.mock_agent import FaultConfig, MockAgent
from evalkit.report import retry_reasons
from evalkit.retry import RetryPolicy, call_llm_with_retries
from evalkit.runner import run_experiment
from evalkit.sample_data import arithmetic_cases
from evalkit.tracer import Tracer

def test_backoff_doubles_and_is_capped():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=5.0, jitter=False)
    assert [policy.delay(retry, Random(0)) for retry in range(5)] == [1.0, 2.0, 4.0, 5.0, 5.0]

def test_jitter_stays_between_half_and_full_delay():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=100.0, jitter=True)
    random = Random(0)
    for retry in range(5):
        full = 2.0 ** retry
        assert full / 2 <= policy.delay(retry, random) <= full

def test_retries_recover_from_rate_limits():
    # Each call fails with probability 0.2, so a trial needs on average 0.2 / 0.8 = 0.25 retries.
    config = FaultConfig(tool_error_rate=0.0, rate_limit_rate=0.2, hallucination_rate=0.0, latency_ms=(0, 0))
    trials = asyncio.run(run_experiment(MockAgent(config), arithmetic_cases(500), attempts=3, concurrency=50))

    retries = retry_reasons(trials)
    assert set(retries) == {"rate_limit"}

    per_trial = sum(retries.values()) / len(trials)
    assert 0.20 < per_trial < 0.30, f"expected ~0.25 retries per trial, got {per_trial:.3f}"

    # All five tries fail with probability 0.2^5 = 0.03%, i.e. about 0.5 of 1,500 trials:
    # rare, but not never. More than a few would mean retries aren't working.
    exhausted = sum(trial.failure_type == "llm_error" for trial in trials)
    assert exhausted <= 3, f"{exhausted} trials ran out of retries; expected about 0.5"

def test_gives_up_after_max_retries():
    config = FaultConfig(tool_error_rate=0.0, rate_limit_rate=1.0, hallucination_rate=0.0, latency_ms=(0, 0))
    trials = asyncio.run(run_experiment(MockAgent(config), arithmetic_cases(3), attempts=1))

    for trial in trials:
        assert trial.failure_type == "llm_error"
        assert [span.retry for span in trial.spans] == [0, 1, 2, 3, 4]
        assert all(span.error_type == "rate_limit" for span in trial.spans)

def test_errors_that_retrying_cannot_fix_are_not_retried():
    calls = 0

    async def bad_request():
        nonlocal calls
        calls += 1
        raise ValueError("400: invalid request")

    tracer = Tracer(trial_id=1)
    with pytest.raises(ValueError):
        asyncio.run(call_llm_with_retries(
            tracer, "model", bad_request, RetryPolicy(), classify=lambda error: None, random=Random(0),
        ))

    assert calls == 1
    assert tracer.spans[0].error_type == "ValueError"
