import asyncio
import time

from evalkit.mock_agent import FaultConfig, MockAgent
from evalkit.report import failure_rate
from evalkit.runner import run_experiment
from evalkit.sample_data import arithmetic_cases

def test_concurrency_runs_trials_in_parallel():
    # 100 trials x 20 ms: about 2 s one at a time, about 0.2 s ten at a time.
    config = FaultConfig(tool_error_rate=0.0, hallucination_rate=0.0, latency_ms=(20, 20))

    started = time.perf_counter()
    trials = asyncio.run(run_experiment(MockAgent(config), arithmetic_cases(100), attempts=1, concurrency=10))
    elapsed = time.perf_counter() - started

    assert len(trials) == 100
    assert elapsed < 1.0, f"took {elapsed:.2f}s; trials are not running in parallel"

def test_results_are_reproducible_regardless_of_concurrency():
    config = FaultConfig(tool_error_rate=0.1, hallucination_rate=0.15, latency_ms=(0, 5))
    dataset = arithmetic_cases(100)

    def outcomes(concurrency: int):
        trials = asyncio.run(run_experiment(MockAgent(config), dataset, attempts=3, concurrency=concurrency))
        return [(trial.case_id, trial.attempt, trial.output, trial.failure_type) for trial in trials]

    assert outcomes(concurrency=1) == outcomes(concurrency=50)

def test_hanging_trials_time_out():
    config = FaultConfig(tool_error_rate=0.0, rate_limit_rate=0.0, hallucination_rate=0.0, hang_rate=0.10, latency_ms=(0, 0))
    trials = asyncio.run(run_experiment(
        MockAgent(config), arithmetic_cases(500), attempts=1, concurrency=100, timeout_s=0.05,
    ))

    measured = failure_rate(trials, "timeout")
    assert 0.06 < measured < 0.14, f"expected ~0.10, got {measured:.3f}"

    for trial in trials:
        if trial.failure_type == "timeout":
            # The cancelled step is still recorded, and marked as an error rather than left "running".
            assert trial.output is None
            assert trial.spans[-1].kind == "llm"
            assert trial.spans[-1].status == "error"
            assert trial.spans[-1].error_type == "CancelledError"
