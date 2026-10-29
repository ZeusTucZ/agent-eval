from datasets.models import Trial

def tool_error_rate(trials: list[Trial]) -> float:
    if not trials:
        return 0.0

    tool_error_count = sum(
        trial.failure_type == "tool_error"
        for trial in trials
    )

    return tool_error_count / len(trials)