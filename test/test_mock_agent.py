import asyncio

from evalkit.mock_agent import FaultConfig, MockAgent
from evalkit.models import Trial
from evalkit.report import latency_percentiles, mean_score, tool_error_rate
from evalkit.runner import run_experiment
from evalkit.sample_data import arithmetic_cases
from evalkit.scorers import exact_match

dataset_of_500_cases = arithmetic_cases(500)

NO_LATENCY = (0, 0)

def run(config: FaultConfig, dataset, attempts: int = 3, concurrency: int = 50) -> list[Trial]:
    return asyncio.run(run_experiment(MockAgent(config), dataset, attempts=attempts, concurrency=concurrency))

def test_tool_error_rate_matches_injected():
    config = FaultConfig(tool_error_rate=0.10, hallucination_rate=0.0, latency_ms=NO_LATENCY)
    trials = run(config, dataset_of_500_cases)

    assert len(trials) == 1500

    measured = tool_error_rate(trials)
    assert 0.07 < measured < 0.13, f"expected ~0.10, got {measured:.3f}"

    for trial in trials:
        if trial.failure_type == "tool_error":
            assert trial.output is None
            assert trial.spans[-1].kind == "tool"
            assert trial.spans[-1].status == "error"
            assert trial.spans[-1].error_type == "ToolError"

def test_each_case_gets_three_attempts():
    config = FaultConfig(tool_error_rate=0.0, hallucination_rate=0.0, latency_ms=NO_LATENCY)
    trials = run(config, dataset_of_500_cases[:2])

    assert [(trial.case_id, trial.attempt) for trial in trials] == [
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
    ]
    assert len({trial.id for trial in trials}) == 6

def test_llm_span_records_tokens():
    config = FaultConfig(tool_error_rate=0.0, hallucination_rate=0.0, latency_ms=NO_LATENCY)
    trials = run(config, dataset_of_500_cases[:10], attempts=1)

    for trial in trials:
        llm_span = trial.spans[0]
        assert llm_span.kind == "llm"
        assert llm_span.name == "mock-llm"
        assert 50 <= llm_span.input_tokens <= 150
        assert 10 <= llm_span.output_tokens <= 40

def test_latency_percentiles_match_injected():
    # Uniform(100, 300) ms -> p50 ~200 ms, p95 ~290 ms. The range is wide because
    # asyncio timers on Windows are only accurate to about 15 ms.
    config = FaultConfig(tool_error_rate=0.0, hallucination_rate=0.0, latency_ms=(100, 300))
    trials = run(config, dataset_of_500_cases[:200], attempts=1)

    percentiles = latency_percentiles(trials)
    assert 180 < percentiles["p50"] < 225, percentiles
    assert 275 < percentiles["p95"] < 320, percentiles

def test_hallucination_rate_matches_injected():
    config = FaultConfig(tool_error_rate=0.0, hallucination_rate=0.15, latency_ms=NO_LATENCY)
    trials = run(config, dataset_of_500_cases)

    cases = {case.id: case for case in dataset_of_500_cases}
    scores = [exact_match(cases[trial.case_id], trial) for trial in trials]

    measured = 1 - mean_score(scores)
    assert 0.12 < measured < 0.18, f"expected ~0.15, got {measured:.3f}"

def test_exact_match_scorer():
    case = dataset_of_500_cases[0]  # "What is 1 + 1" -> "2"

    right = exact_match(case, Trial(id=1, case_id=1, attempt=1, output="2"))
    wrong = exact_match(case, Trial(id=2, case_id=1, attempt=1, output="47"))
    failed = exact_match(case, Trial(id=3, case_id=1, attempt=1, output=None, failure_type="tool_error"))

    assert (right.value, wrong.value, failed.value) == (1.0, 0.0, 0.0)
    assert right.scorer == "exact_match"
    assert "tool_error" in failed.reasoning
