import argparse
import sys
from pathlib import Path
from collections import Counter
from statistics import quantiles

from agenteval import storage
from agenteval.environment import describe as describe_environment
from agenteval.cost import total_cost
from agenteval.models import EvaluationCase, Experiment, Score, Trial
from agenteval.sample_data import dataset_by_name

def tool_error_rate(trials: list[Trial]) -> float:
    return failure_rate(trials, "tool_error")

def failure_rate(trials: list[Trial], failure_type: str) -> float:
    if not trials:
        return 0.0

    count = sum(trial.failure_type == failure_type for trial in trials)
    return count / len(trials)

def latency_percentiles(trials: list[Trial], working_only: bool = False) -> dict[str, float]:
    """p50/p95 of trial duration in milliseconds; working_only leaves out time spent waiting.
    With fewer than two trials (a run interrupted at once) the one duration stands in."""
    durations = [trial.working_ms if working_only else trial.duration_ms for trial in trials]
    if len(durations) < 2:
        only = durations[0] if durations else 0.0
        return {"p50": only, "p95": only}

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

# Where a retrieval trial went right or wrong: did the search find every relevant document,
# and was the answer correct? In the order the report prints them.
DIAGNOSES = {
    "correct": "found the documents, answered right",
    "answer_failure": "found the documents, answered wrong",
    "retrieval_failure": "missed a document, answered wrong",
    "unsupported_correct": "missed a document, answered right anyway",
}

def retrieval_diagnosis(scores: list[Score]) -> Counter[str]:
    """Counts of each diagnosis, over trials scored for both retrieval and correctness.

    "unsupported_correct" is worth reading: the answer can't have come from the documents
    the search returned, so it was a guess, memory, or a fact found in some other document.
    """
    recall = {score.trial_id: score.value for score in _scores_named(scores, "retrieval_recall")}
    correct = {score.trial_id: score.value == 1.0 for score in _scores_named(scores, "exact_match")}

    diagnoses: Counter[str] = Counter()
    for trial_id, value in recall.items():
        if trial_id not in correct:
            continue
        found = value == 1.0
        if found:
            diagnoses["correct" if correct[trial_id] else "answer_failure"] += 1
        else:
            diagnoses["unsupported_correct" if correct[trial_id] else "retrieval_failure"] += 1
    return diagnoses

# Scorers the report lays out in their own sections; any other (an LLM judge's, your own)
# is listed under "Other scores".
BUILT_IN_SCORERS = {
    "exact_match", "tool_selection", "tool_arguments_valid", "tool_execution", "tool_faithfulness",
    "retrieval_recall", "avoids_forbidden_text",
}

def other_scores(scores: list[Score]) -> dict[str, tuple[float, int]]:
    """scorer -> (mean, count) for judge and custom scorers, in name order."""
    names = sorted({score.scorer for score in scores} - BUILT_IN_SCORERS)
    return {name: (mean_score(found), len(found)) for name in names for found in [_scores_named(scores, name)]}

def status_line(experiment: Experiment, trials: list[Trial]) -> str | None:
    """Only for a run that didn't finish: how far it got and how to continue it."""
    if experiment.complete:
        return None
    planned = f" of {experiment.planned_trials}" if experiment.planned_trials else ""
    return (f"Status:           {experiment.status.upper()}, {len(trials)}{planned} trials. "
            f"Numbers below cover only these; finish with: agenteval run ... --resume {experiment.id}")

def summary_data(
    experiment: Experiment,
    trials: list[Trial],
    scores: list[Score],
    cases: list[EvaluationCase] | None = None,
) -> dict:
    """Everything the report shows, as data: for --format json, the Python API and exports."""
    exact_match_scores = _scores_named(scores, "exact_match")
    successes = sum(score.value == 1.0 for score in exact_match_scores)
    cost = total_cost(trials, experiment.prices)

    def mean_or_none(name: str) -> float | None:
        found = _scores_named(scores, name)
        return mean_score(found) if found else None

    return {
        "id": experiment.id,
        "name": experiment.name,
        "status": experiment.status,
        "trials": len(trials),
        "planned_trials": experiment.planned_trials,
        "model": experiment.config.model,
        "dataset": experiment.dataset_name,
        "dataset_version": experiment.dataset_version,
        "config_version": experiment.config.version,
        "attempts": experiment.attempts,
        "created_at": experiment.created_at.isoformat(),
        "environment": experiment.environment,
        "pass_rate": mean_score(exact_match_scores),
        "by_category": {
            category: {"pass_rate": rate, "trials": count}
            for category, (rate, count) in (pass_rate_by_category(trials, scores, cases) if cases else {}).items()
        },
        "latency_ms": latency_percentiles(trials),
        "latency_working_ms": latency_percentiles(trials, working_only=True),
        "cost": cost,
        "cost_per_success": cost / successes if successes else None,
        "checks": {
            name: mean_or_none(name)
            for name in ("tool_selection", "tool_arguments_valid", "tool_execution", "tool_faithfulness",
                         "retrieval_recall", "avoids_forbidden_text")
        },
        "retrieval_diagnosis": dict(retrieval_diagnosis(scores)),
        "other_scores": {name: {"mean": mean, "trials": count} for name, (mean, count) in other_scores(scores).items()},
        "retries": dict(retry_reasons(trials)),
        "failures": dict(Counter(trial.failure_type for trial in trials if trial.failure_type)),
    }

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
    ]
    status = status_line(experiment, trials)
    if status:
        lines.append(status)
    where = describe_environment(experiment.environment)
    if where:
        lines.append(f"Ran with:         {where}")
    lines.append(f"Pass rate:        {mean_score(exact_match_scores):.1%}")

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
        "  " + _rate_line("backed by tools", _scores_named(scores, "tool_faithfulness")),
    ]
    recall = _scores_named(scores, "retrieval_recall")
    if recall:
        diagnoses = retrieval_diagnosis(scores)
        total = sum(diagnoses.values())
        lines += ["Retrieval:", "  " + _rate_line("relevant found", recall)]
        lines += [
            f"  {description:<42}{diagnoses[name]:>4}  ({diagnoses[name] / total:.0%})"
            for name, description in DIAGNOSES.items() if total
        ]
    if forbidden:
        lines += ["Safety:", "  " + _rate_line("no injected text", forbidden)]
    others = other_scores(scores)
    if others:
        width = max(18, max(len(name) for name in others) + 2)
        lines.append("Other scores:")
        lines += [f"  {name:<{width}}{mean:.1%}  ({count} trials)" for name, (mean, count) in others.items()]
    lines += [
        f"Retries:          {sum(retries.values())}" + (f" ({retry_text})" if retry_text else ""),
        f"Timeouts:         {failure_rate(trials, 'timeout'):.1%}",
        f"Cost:             ${cost:.4f} total, {cost_per_success}",
        f"Failures:         {failure_text}",
    ]
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> None:
    from agenteval import export

    parser = argparse.ArgumentParser(prog="agenteval report", description="Print the report for a saved experiment.")
    parser.add_argument("experiment_id", type=int, nargs="?", help="default: the latest experiment")
    parser.add_argument("--db", default="agenteval.db")
    parser.add_argument("--format", choices=["text", "json", "markdown", "html", "csv"], default="text",
                        help="csv is one row per trial, with the question, answer and every score")
    parser.add_argument("-o", "--output", help="write to this file instead of printing")
    args = parser.parse_args(argv)

    conn = storage.connect(args.db)
    experiment_id = args.experiment_id or storage.latest_experiment_id(conn)
    if experiment_id is None:
        raise SystemExit(f"No experiments in {args.db} yet: run one with `agenteval run`.")
    try:
        experiment = storage.load_experiment(conn, experiment_id)
    except LookupError as error:
        raise SystemExit(str(error)) from None
    try:
        cases = dataset_by_name(experiment.dataset_name)
    except (ValueError, OSError):
        cases = None  # an unknown or moved dataset only loses the per-category breakdown and questions
    trials, scores = storage.load_trials(conn, experiment_id), storage.load_scores(conn, experiment_id)

    if args.format == "text":
        text = summarize(experiment, trials, scores, cases)
    else:
        text = export.render(args.format, experiment, trials, scores, cases)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote the {args.format} report for experiment {experiment_id} to {args.output}")
    else:
        sys.stdout.reconfigure(errors="replace")
        print(text)

if __name__ == "__main__":
    main()
