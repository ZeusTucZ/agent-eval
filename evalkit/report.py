import argparse
from collections import Counter
from statistics import quantiles

from evalkit import storage
from evalkit.cost import total_cost
from evalkit.models import EvaluationCase, Experiment, Score, Trial
from evalkit.sample_data import dataset_by_name

def tool_error_rate(trials: list[Trial]) -> float:
    return failure_rate(trials, "tool_error")

def failure_rate(trials: list[Trial], failure_type: str) -> float:
    if not trials:
        return 0.0

    count = sum(trial.failure_type == failure_type for trial in trials)
    return count / len(trials)

def latency_percentiles(trials: list[Trial], working_only: bool = False) -> dict[str, float]:
    """p50/p95 of trial duration in milliseconds; working_only leaves out time spent waiting."""
    durations = [trial.working_ms if working_only else trial.duration_ms for trial in trials]
    if len(durations) < 2:
        raise ValueError("need at least 2 trials to compute percentiles")

    cuts = quantiles(durations, n=100, method="inclusive")
    return {"p50": cuts[49], "p95": cuts[94]}

def mean_score(scores: list[Score]) -> float:
    if not scores:
        return 0.0
    return sum(score.value for score in scores) / len(scores)

def retry_reasons(trials: list[Trial]) -> Counter[str]:
    """How many retries happened, by the reason the previous attempt failed."""
    reasons: Counter[str] = Counter()
    for trial in trials:
        previous = None
        for span in trial.spans:
            if span.kind != "llm":
                continue
            if span.retry > 0 and previous is not None:
                reasons[previous.error_type or "unknown"] += 1
            previous = span
    return reasons

def _scores_named(scores: list[Score], scorer: str) -> list[Score]:
    return [score for score in scores if score.scorer == scorer]

def _rate_line(label: str, scores: list[Score]) -> str:
    if not scores:
        return f"{label:<18}n/a"
    return f"{label:<18}{mean_score(scores):.1%}  ({len(scores)} trials)"

def pass_rate_by_category(
    trials: list[Trial], scores: list[Score], cases: list[EvaluationCase],
) -> dict[str, tuple[float, int]]:
    """category -> (exact-match pass rate, number of trials), in the order categories first appear."""
    category_of_case = {case.id: case.category for case in cases}
    category_of_trial = {trial.id: category_of_case[trial.case_id] for trial in trials}

    by_category: dict[str, list[Score]] = {}
    for case in cases:
        by_category.setdefault(case.category, [])
    for score in _scores_named(scores, "exact_match"):
        by_category[category_of_trial[score.trial_id]].append(score)

    return {category: (mean_score(found), len(found)) for category, found in by_category.items() if found}

def summarize(
    experiment: Experiment,
    trials: list[Trial],
    scores: list[Score],
    cases: list[EvaluationCase] | None = None,
) -> str:
    exact_match_scores = _scores_named(scores, "exact_match")
    successes = sum(score.value == 1.0 for score in exact_match_scores)
    cost = total_cost(trials, experiment.prices)
    latency = latency_percentiles(trials)
    working = latency_percentiles(trials, working_only=True)
    failures = Counter(trial.failure_type for trial in trials if trial.failure_type)
    retries = retry_reasons(trials)

    cost_per_success = f"${cost / successes:.6f} per success" if successes else "no successes"
    failure_text = ", ".join(f"{name} {count}" for name, count in failures.most_common()) or "none"
    retry_text = ", ".join(f"{reason} {count}" for reason, count in retries.most_common())

    lines = [
        f"Experiment {experiment.id}: {experiment.name}   config v={experiment.config.version}   {len(trials)} trials",
        f"Model:            {experiment.config.model}   dataset: {experiment.dataset_name}",
        f"Pass rate:        {mean_score(exact_match_scores):.1%}",
    ]

    # A single pass rate hides which kind of question fails, so break it down when there
    # is more than one category.
    categories = pass_rate_by_category(trials, scores, cases) if cases else {}
    if len(categories) > 1:
        lines.append("By category:")
        width = max(len(category) for category in categories) + 2
        lines += [f"  {category:<{width}}{rate:.1%}  ({count} trials)" for category, (rate, count) in categories.items()]

    forbidden = _scores_named(scores, "avoids_forbidden_text")
    lines += [
        f"Latency p50/p95:  {latency['p50']:.0f}ms / {latency['p95']:.0f}ms total,"
        f" {working['p50']:.0f}ms / {working['p95']:.0f}ms excluding waits",
        "Tool calls:",
        "  " + _rate_line("right tool", _scores_named(scores, "tool_selection")),
        "  " + _rate_line("valid arguments", _scores_named(scores, "tool_arguments_valid")),
        "  " + _rate_line("ran without error", _scores_named(scores, "tool_execution")),
    ]
    if forbidden:
        lines += ["Safety:", "  " + _rate_line("no injected text", forbidden)]
    lines += [
        f"Retries:          {sum(retries.values())}" + (f" ({retry_text})" if retry_text else ""),
        f"Timeouts:         {failure_rate(trials, 'timeout'):.1%}",
        f"Cost:             ${cost:.4f} total, {cost_per_success}",
        f"Failures:         {failure_text}",
    ]
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Print the report for a saved experiment.")
    parser.add_argument("experiment_id", type=int)
    parser.add_argument("--db", default="evalkit.db")
    args = parser.parse_args()

    conn = storage.connect(args.db)
    experiment = storage.load_experiment(conn, args.experiment_id)
    try:
        cases = dataset_by_name(experiment.dataset_name)
    except ValueError:
        cases = None  # an unknown dataset only loses the per-category breakdown
    print(summarize(
        experiment,
        storage.load_trials(conn, args.experiment_id),
        storage.load_scores(conn, args.experiment_id),
        cases,
    ))

if __name__ == "__main__":
    main()
