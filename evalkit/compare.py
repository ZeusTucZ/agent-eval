import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from evalkit import storage
from evalkit.cost import trial_cost
from evalkit.models import AgentConfig, EvaluationCase, Experiment, Score, Trial
from evalkit.sample_data import dataset_by_name
from evalkit.stats import Interval, bootstrap_mean, mcnemar_exact

SIGNIFICANCE_LEVEL = 0.05
# A cost or latency rise smaller than this is noise as far as a decision goes, even if it's real.
MEANINGFUL_INCREASE = 0.10
SMALLEST_LATENCY_RISE_MS = 100.0
MAX_TIME_APART = timedelta(hours=24)

@dataclass
class ExperimentData:
    experiment: Experiment
    trials: list[Trial]
    scores: list[Score]

def load(conn, experiment_id: int) -> ExperimentData:
    return ExperimentData(
        storage.load_experiment(conn, experiment_id),
        storage.load_trials(conn, experiment_id),
        storage.load_scores(conn, experiment_id),
    )

# --- Is this comparison fair? ---

def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        flat.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
    return flat

def config_changes(a: AgentConfig, b: AgentConfig) -> list[str]:
    """Which settings differ, e.g. ["prompt", "params.temperature"]."""
    flat_a, flat_b = _flatten(a.model_dump()), _flatten(b.model_dump())
    return sorted(key for key in flat_a.keys() | flat_b.keys() if flat_a.get(key) != flat_b.get(key))

def is_infrastructure_failure(failure_type: str | None) -> bool:
    return failure_type is not None and (
        failure_type in {"quota_exhausted", "llm_error"} or failure_type.startswith("timeout")
    )

@dataclass
class Fairness:
    errors: list[str] = field(default_factory=list)  # the comparison would be meaningless
    warnings: list[str] = field(default_factory=list)  # it's possible, but read it with care
    changes: list[str] = field(default_factory=list)

def check_comparable(a: ExperimentData, b: ExperimentData) -> Fairness:
    """A comparison only means something if the thing under test is all that changed."""
    fairness = Fairness(changes=config_changes(a.experiment.config, b.experiment.config))

    if a.experiment.dataset_name != b.experiment.dataset_name:
        fairness.errors.append(
            f"different datasets: {a.experiment.dataset_name} vs {b.experiment.dataset_name}"
        )
    elif a.experiment.dataset_version is None or b.experiment.dataset_version is None:
        fairness.warnings.append("can't verify the dataset is unchanged: an experiment predates dataset versions")
    elif a.experiment.dataset_version != b.experiment.dataset_version:
        fairness.errors.append(
            f"dataset {a.experiment.dataset_name} changed between runs "
            f"({a.experiment.dataset_version} vs {b.experiment.dataset_version})"
        )

    if not fairness.changes:
        fairness.warnings.append("the configs are identical: this measures noise, not a change")
    elif len(fairness.changes) > 1:
        fairness.warnings.append(
            f"{len(fairness.changes)} settings changed, so any difference can't be pinned on one of them"
        )

    if a.experiment.attempts != b.experiment.attempts:
        fairness.warnings.append(f"different attempts per case: {a.experiment.attempts} vs {b.experiment.attempts}")

    if abs(a.experiment.created_at - b.experiment.created_at) > MAX_TIME_APART:
        fairness.warnings.append("the runs are over 24 hours apart: the provider's model or load may have changed")

    for label, data in (("A", a), ("B", b)):
        failed = sum(is_infrastructure_failure(trial.failure_type) for trial in data.trials)
        if failed:
            fairness.warnings.append(
                f"{failed} of {len(data.trials)} trials in {label} failed for infrastructure reasons, not the model's"
            )

    return fairness

# --- The comparison ---

@dataclass
class CaseChange:
    case: EvaluationCase
    a_rate: float
    b_rate: float
    a_output: str | None
    b_output: str | None

@dataclass
class MetricChange:
    a: float
    b: float
    change: Interval

    def meaningfully_worse(self, smallest_that_matters: float = 0.0) -> bool:
        """Clearly higher in B, by enough to matter. For metrics where lower is better.

        The floor stops tiny baselines from misleading: 0.3 ms to 0.4 ms is "33% slower",
        but nobody could notice it.
        """
        return (
            self.change.low > 0
            and self.change.mean > MEANINGFUL_INCREASE * self.a
            and self.change.mean > smallest_that_matters
        )

@dataclass
class Comparison:
    cases_compared: int
    pass_rate: MetricChange
    p_value: float
    improved: list[CaseChange]
    regressed: list[CaseChange]
    partly_changed: int  # cases whose rate moved without a clean flip, e.g. 1 of 2 -> 2 of 2
    working_ms: MetricChange
    cost: MetricChange
    categories: dict[str, tuple[float, float, int]]  # category -> (A rate, B rate, cases)
    safety_regressions: list[CaseChange]
    verdict: str
    reason: str

def _by_case(values: dict[int, float], trials: list[Trial]) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for trial in trials:
        if trial.id in values:
            grouped.setdefault(trial.case_id, []).append(values[trial.id])
    return grouped

def _case_rates(data: ExperimentData, scorer: str) -> dict[int, float]:
    """case id -> mean score across its attempts, one number per case."""
    values = {score.trial_id: score.value for score in data.scores if score.scorer == scorer}
    return {case_id: sum(found) / len(found) for case_id, found in _by_case(values, data.trials).items()}

def _case_means(data: ExperimentData, measure) -> dict[int, float]:
    values = {trial.id: measure(trial) for trial in data.trials}
    return {case_id: sum(found) / len(found) for case_id, found in _by_case(values, data.trials).items()}

def _typical_output(data: ExperimentData, case_id: int) -> str | None:
    outputs = Counter(trial.output for trial in data.trials if trial.case_id == case_id)
    return outputs.most_common(1)[0][0] if outputs else None

def _metric(a: dict[int, float], b: dict[int, float], case_ids: list[int], resamples: int) -> MetricChange:
    return MetricChange(
        a=sum(a[case_id] for case_id in case_ids) / len(case_ids),
        b=sum(b[case_id] for case_id in case_ids) / len(case_ids),
        change=bootstrap_mean([b[case_id] - a[case_id] for case_id in case_ids], resamples),
    )

def _passes(rate: float) -> bool:
    return rate > 0.5  # a case passes if most of its attempts did

def _fails(rate: float) -> bool:
    return rate < 0.5  # exactly half is neither: with an even number of attempts, that's a tie

def _decide(
    pass_rate: MetricChange,
    p_value: float,
    working_ms: MetricChange,
    cost: MetricChange,
    safety_regressions: list[CaseChange],
    disagreements: int,
) -> tuple[str, str]:
    change = pass_rate.change
    points = change.mean * 100

    if p_value < SIGNIFICANCE_LEVEL and change.excludes_zero:
        if change.mean < 0:
            return "Worse", f"pass rate fell by {-points:.1f} points, unlikely to be chance"

        downsides = []
        if safety_regressions:
            downsides.append(f"{len(safety_regressions)} new safety failure(s)")
        if cost.meaningfully_worse():
            downsides.append(f"cost up {cost.change.mean / cost.a:.0%}")
        if working_ms.meaningfully_worse(SMALLEST_LATENCY_RISE_MS):
            downsides.append(f"latency up {working_ms.change.mean / working_ms.a:.0%}")
        if downsides:
            return "Trade-off", f"pass rate rose by {points:.1f} points, but: {', '.join(downsides)}"
        return "Better", f"pass rate rose by {points:.1f} points, unlikely to be chance"

    if disagreements == 0:
        return "Inconclusive", "no case changed result, so this dataset can't separate the two versions"
    note = f"; {len(safety_regressions)} new safety failure(s) need a look" if safety_regressions else ""
    return "Inconclusive", (
        f"this could be chance: the true change is plausibly between "
        f"{change.low * 100:+.0f} and {change.high * 100:+.0f} points{note}"
    )

def compare_experiments(
    a: ExperimentData,
    b: ExperimentData,
    cases: list[EvaluationCase],
    resamples: int = 10_000,
) -> Comparison:
    a_rates, b_rates = _case_rates(a, "exact_match"), _case_rates(b, "exact_match")
    case_ids = sorted(a_rates.keys() & b_rates.keys())
    if not case_ids:
        raise ValueError("the two experiments have no scored cases in common")
    by_id = {case.id: case for case in cases}

    def change_for(case_id: int) -> CaseChange:
        return CaseChange(by_id[case_id], a_rates[case_id], b_rates[case_id],
                          _typical_output(a, case_id), _typical_output(b, case_id))

    # Only clean flips count as evidence. A tied case has no majority either way, so treating
    # 1 of 2 -> 2 of 2 as an improvement adds noise to McNemar's test; the pass rate still
    # reflects it. Counting ties once let a 23.6-point drop pass as "could be chance".
    improved = [change_for(i) for i in case_ids if _fails(a_rates[i]) and _passes(b_rates[i])]
    regressed = [change_for(i) for i in case_ids if _passes(a_rates[i]) and _fails(b_rates[i])]
    partly_changed = sum(a_rates[i] != b_rates[i] for i in case_ids) - len(improved) - len(regressed)

    pass_rate = _metric(a_rates, b_rates, case_ids, resamples)
    p_value = mcnemar_exact(len(improved), len(regressed))
    working_ms = _metric(
        _case_means(a, lambda t: t.working_ms), _case_means(b, lambda t: t.working_ms), case_ids, resamples,
    )
    cost = _metric(
        _case_means(a, lambda t: trial_cost(t, a.experiment.prices)),
        _case_means(b, lambda t: trial_cost(t, b.experiment.prices)),
        case_ids,
        resamples,
    )

    grouped: dict[str, list[int]] = {}
    for case_id in case_ids:
        grouped.setdefault(by_id[case_id].category, []).append(case_id)
    categories = {
        category: (
            sum(a_rates[i] for i in ids) / len(ids),
            sum(b_rates[i] for i in ids) / len(ids),
            len(ids),
        )
        for category, ids in grouped.items()
    }

    a_safe, b_safe = _case_rates(a, "avoids_forbidden_text"), _case_rates(b, "avoids_forbidden_text")
    safety_regressions = [
        change_for(i) for i in case_ids
        if i in a_safe and i in b_safe and _passes(a_safe[i]) and not _passes(b_safe[i])
    ]

    verdict, reason = _decide(
        pass_rate, p_value, working_ms, cost, safety_regressions,
        len(improved) + len(regressed) + partly_changed,
    )
    return Comparison(
        len(case_ids), pass_rate, p_value, improved, regressed, partly_changed, working_ms, cost,
        categories, safety_regressions, verdict, reason,
    )

# --- Output ---

def _case_line(change: CaseChange) -> str:
    return (f"  case {change.case.id} [{change.case.category}] {change.case.input[:60]!r}: "
            f"{change.a_output!r} ({change.a_rate:.0%}) -> {change.b_output!r} ({change.b_rate:.0%})")

def format_comparison(a: ExperimentData, b: ExperimentData, fairness: Fairness, result: Comparison) -> str:
    pass_change, working, cost = result.pass_rate.change, result.working_ms.change, result.cost.change
    lines = [
        f"A: experiment {a.experiment.id} {a.experiment.name} (config {a.experiment.config.version})",
        f"B: experiment {b.experiment.id} {b.experiment.name} (config {b.experiment.config.version})",
        f"Dataset:      {a.experiment.dataset_name}, {result.cases_compared} cases compared",
        f"Differs in:   {', '.join(fairness.changes) or 'nothing'}",
    ]
    lines += [f"WARNING: {warning}" for warning in fairness.warnings]
    lines += [
        "",
        f"Pass rate:    {result.pass_rate.a:.1%} -> {result.pass_rate.b:.1%}   "
        f"{pass_change.mean * 100:+.1f} points   95% CI [{pass_change.low * 100:+.1f}, {pass_change.high * 100:+.1f}]",
        f"McNemar:      {len(result.improved)} improved, {len(result.regressed)} regressed   p = {result.p_value:.3f}"
        + (f"   ({result.partly_changed} more changed only partly)" if result.partly_changed else ""),
        f"Latency:      {result.working_ms.a:.0f}ms -> {result.working_ms.b:.0f}ms per trial, excluding waits   "
        f"{working.mean:+.0f}ms   95% CI [{working.low:+.0f}, {working.high:+.0f}]",
        f"Cost/trial:   ${result.cost.a:.6f} -> ${result.cost.b:.6f}   "
        f"{cost.mean:+.6f}   95% CI [{cost.low:+.6f}, {cost.high:+.6f}]",
    ]

    moved = {name: rates for name, rates in result.categories.items() if rates[0] != rates[1]}
    if moved:
        lines += ["", "Categories that moved (leads to investigate, not proof):"]
        width = max(len(name) for name in moved) + 2
        lines += [f"  {name:<{width}}{a_rate:.1%} -> {b_rate:.1%}  ({count} cases)"
                  for name, (a_rate, b_rate, count) in moved.items()]

    if result.safety_regressions:
        lines += ["", f"NEW SAFETY FAILURES ({len(result.safety_regressions)}):"]
        lines += [_case_line(change) for change in result.safety_regressions]
    if result.regressed:
        lines += ["", f"Regressed ({len(result.regressed)}):"]
        lines += [_case_line(change) for change in result.regressed]
    if result.improved:
        lines += ["", f"Improved ({len(result.improved)}):"]
        lines += [_case_line(change) for change in result.improved]

    lines += ["", f"Verdict: {result.verdict}. {result.reason}."]
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two saved experiments: is B really better than A?")
    parser.add_argument("a", type=int, help="the baseline experiment")
    parser.add_argument("b", type=int, help="the experiment to judge")
    parser.add_argument("--db", default="evalkit.db")
    args = parser.parse_args()

    # Model outputs can hold characters a Windows console can't show.
    sys.stdout.reconfigure(errors="replace")

    conn = storage.connect(args.db)
    a, b = load(conn, args.a), load(conn, args.b)

    fairness = check_comparable(a, b)
    if fairness.errors:
        raise SystemExit("Can't compare these experiments: " + "; ".join(fairness.errors))

    result = compare_experiments(a, b, dataset_by_name(a.experiment.dataset_name))
    print(format_comparison(a, b, fairness, result))

if __name__ == "__main__":
    main()
