from statistics import quantiles

from evalkit.models import Score, Trial

def tool_error_rate(trials: list[Trial]) -> float:
    if not trials:
        return 0.0

    tool_error_count = sum(
        trial.failure_type == "tool_error"
        for trial in trials
    )

    return tool_error_count / len(trials)

def latency_percentiles(trials: list[Trial]) -> dict[str, float]:
    """p50/p95 of trial duration in milliseconds."""
    durations = [trial.duration_ms for trial in trials]
    if len(durations) < 2:
        raise ValueError("need at least 2 trials to compute percentiles")

    cuts = quantiles(durations, n=100, method="inclusive")
    return {"p50": cuts[49], "p95": cuts[94]}

def mean_score(scores: list[Score]) -> float:
    if not scores:
        return 0.0
    return sum(score.value for score in scores) / len(scores)
