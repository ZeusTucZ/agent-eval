import argparse
from collections import Counter
from statistics import quantiles

from evalkit import storage
from evalkit.cost import total_cost
from evalkit.models import Experiment, Score, Trial

def tool_error_rate(trials: list[Trial]) -> float:
    return failure_rate(trials, "tool_error")

def failure_rate(trials: list[Trial], failure_type: str) -> float:
    if not trials:
        return 0.0

    count = sum(trial.failure_type == failure_type for trial in trials)
    return count / len(trials)

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

def summarize(experiment: Experiment, trials: list[Trial], scores: list[Score]) -> str:
    exact_match_scores = [score for score in scores if score.scorer == "exact_match"]
    successes = sum(score.value == 1.0 for score in exact_match_scores)
    cost = total_cost(trials, experiment.prices)
    latency = latency_percentiles(trials)
    failures = Counter(trial.failure_type for trial in trials if trial.failure_type)

    cost_per_success = f"${cost / successes:.6f} per success" if successes else "no successes"
    failure_text = ", ".join(f"{name} {count}" for name, count in failures.most_common()) or "none"

    return "\n".join([
        f"Experiment {experiment.id}: {experiment.name}   config v={experiment.config.version}   {len(trials)} trials",
        f"Model:            {experiment.config.model}   dataset: {experiment.dataset_name}",
        f"Pass rate:        {mean_score(exact_match_scores):.1%}",
        f"Latency p50/p95:  {latency['p50']:.0f}ms / {latency['p95']:.0f}ms",
        f"Tool error rate:  {tool_error_rate(trials):.1%}",
        f"Timeouts:         {failure_rate(trials, 'timeout'):.1%}",
        f"Cost:             ${cost:.4f} total, {cost_per_success}",
        f"Failures:         {failure_text}",
    ])

def main() -> None:
    parser = argparse.ArgumentParser(description="Print the report for a saved experiment.")
    parser.add_argument("experiment_id", type=int)
    parser.add_argument("--db", default="evalkit.db")
    args = parser.parse_args()

    conn = storage.connect(args.db)
    experiment = storage.load_experiment(conn, args.experiment_id)
    print(summarize(
        experiment,
        storage.load_trials(conn, args.experiment_id),
        storage.load_scores(conn, args.experiment_id),
    ))

if __name__ == "__main__":
    main()
