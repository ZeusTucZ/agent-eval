"""The regression gate: should this change be allowed through?

Compares the latest experiment in two databases and turns the result into an exit code
CI can act on:

    python -m evalkit.gate baseline.db candidate.db --markdown report.md

    0  pass
    1  block: quality dropped, or there is a new safety failure
    2  can't judge: the comparison would mean nothing, so re-run or review by hand
"""
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from evalkit import storage
from evalkit.compare import (
    Comparison,
    ExperimentData,
    Fairness,
    check_comparable,
    compare_experiments,
    format_comparison,
    is_infrastructure_failure,
    load,
)
from evalkit.sample_data import dataset_by_name

PASS = 0
BLOCK = 1
CANNOT_JUDGE = 2

# Above this, a run says more about the API that day than about the agent.
MAX_INFRASTRUCTURE_FAILURE_RATE = 0.10

# Finds the bot's earlier comment on a PR so it can be updated rather than duplicated.
COMMENT_MARKER = "<!-- agent-eval-gate -->"

HEADINGS = {PASS: "✅ Pass", BLOCK: "❌ Blocked", CANNOT_JUDGE: "⚠️ Can't judge"}

@dataclass
class Decision:
    exit_code: int
    reasons: list[str] = field(default_factory=list)

def decide(a: ExperimentData, b: ExperimentData, fairness: Fairness, result: Comparison | None) -> Decision:
    if fairness.errors:
        return Decision(CANNOT_JUDGE, fairness.errors)

    damaged = []
    for label, data in (("baseline", a), ("candidate", b)):
        failed = sum(is_infrastructure_failure(trial.failure_type) for trial in data.trials)
        if failed > MAX_INFRASTRUCTURE_FAILURE_RATE * len(data.trials):
            damaged.append(f"{failed} of {len(data.trials)} {label} trials failed for infrastructure reasons")
    if damaged:
        return Decision(CANNOT_JUDGE, damaged)

    if result is None:
        return Decision(CANNOT_JUDGE, ["no comparison could be made"])

    # Checked first and on its own: a safety failure blocks even if the overall score rose
    # or the verdict is Inconclusive.
    if result.safety_regressions:
        return Decision(BLOCK, [f"{len(result.safety_regressions)} new safety failure(s)", result.reason])

    if result.verdict == "Worse":
        return Decision(BLOCK, [result.reason])

    if result.verdict == "Inconclusive":
        return Decision(PASS, [f"no significant drop, but this doesn't prove the change is safe: {result.reason}"])
    return Decision(PASS, [result.reason])

def _inline(text: str | None, limit: int = 80) -> str:
    """Show model output safely: it's untrusted text, so it must not act as Markdown in a PR comment."""
    if text is None:
        return "*no answer*"
    flattened = " ".join(text.split()).replace("`", "'").replace("|", "/")
    clipped = flattened[:limit] + ("..." if len(flattened) > limit else "")
    return f"`{clipped}`" if clipped else "*empty*"

def _case_rows(changes) -> list[str]:
    rows = ["| Case | Category | Before | After |", "|---|---|---|---|"]
    for change in changes:
        rows.append(
            f"| {change.case.id} | {change.case.category} "
            f"| {_inline(change.a_output)} ({change.a_rate:.0%}) "
            f"| {_inline(change.b_output)} ({change.b_rate:.0%}) |"
        )
    return rows

def format_markdown(
    a: ExperimentData | None,
    fairness: Fairness | None,
    result: Comparison | None,
    decision: Decision,
) -> str:
    lines = [COMMENT_MARKER, f"## Agent eval: {HEADINGS[decision.exit_code]}", ""]
    lines += [f"- {reason}" for reason in decision.reasons]

    if result is not None and a is not None:
        pass_change, working, cost = result.pass_rate.change, result.working_ms.change, result.cost.change
        lines += [
            "",
            f"**Verdict: {result.verdict}** on `{a.experiment.dataset_name}`, "
            f"{result.cases_compared} cases, model `{a.experiment.config.model}`",
            "",
            "| Metric | Baseline | This PR | Change | 95% CI |",
            "|---|---|---|---|---|",
            f"| Pass rate | {result.pass_rate.a:.1%} | {result.pass_rate.b:.1%} | {pass_change.mean * 100:+.1f} pts "
            f"| {pass_change.low * 100:+.1f} to {pass_change.high * 100:+.1f} |",
            f"| Latency per trial | {result.working_ms.a:.0f} ms | {result.working_ms.b:.0f} ms | {working.mean:+.0f} ms "
            f"| {working.low:+.0f} to {working.high:+.0f} |",
            f"| Cost per trial | ${result.cost.a:.6f} | ${result.cost.b:.6f} | {cost.mean:+.6f} "
            f"| {cost.low:+.6f} to {cost.high:+.6f} |",
            "",
            f"{len(result.improved)} improved, {len(result.regressed)} regressed (McNemar p = {result.p_value:.3f})"
            + (f", {result.partly_changed} more changed only partly" if result.partly_changed else ""),
        ]

        moved = {name: rates for name, rates in result.categories.items() if rates[0] != rates[1]}
        if moved:
            lines += ["", "| Category | Baseline | This PR |", "|---|---|---|"]
            lines += [f"| {name} | {a_rate:.1%} | {b_rate:.1%} |" for name, (a_rate, b_rate, _) in moved.items()]

        for title, changes in (
            ("New safety failures", result.safety_regressions),
            ("Regressed", result.regressed),
            ("Improved", result.improved),
        ):
            if changes:
                lines += ["", f"**{title} ({len(changes)})**", "", *_case_rows(changes)]

    if fairness is not None and fairness.warnings:
        lines += ["", "**Warnings**", ""]
        lines += [f"- {warning}" for warning in fairness.warnings]

    return "\n".join(lines) + "\n"

def _load_latest(path: str) -> ExperimentData | None:
    conn = storage.connect(path)
    experiment_id = storage.latest_experiment_id(conn)
    return load(conn, experiment_id) if experiment_id is not None else None

def run_gate(baseline_path: str, candidate_path: str) -> tuple[Decision, str, str]:
    """Returns the decision, a Markdown report and a plain-text report."""
    a, b = _load_latest(baseline_path), _load_latest(candidate_path)

    missing = [label for label, data in (("baseline", a), ("candidate", b)) if data is None]
    if missing:
        decision = Decision(CANNOT_JUDGE, [f"no saved experiment for: {', '.join(missing)}"])
        return decision, format_markdown(None, None, None, decision), "\n".join(decision.reasons)

    fairness = check_comparable(a, b)
    result = None
    if not fairness.errors:
        try:
            result = compare_experiments(a, b, dataset_by_name(a.experiment.dataset_name))
        except ValueError as error:
            fairness.errors.append(str(error))

    decision = decide(a, b, fairness, result)
    text = format_comparison(a, b, fairness, result) if result is not None else ""
    summary = f"Gate: {HEADINGS[decision.exit_code]}\n" + "\n".join(f"  {reason}" for reason in decision.reasons)
    return decision, format_markdown(a, fairness, result, decision), f"{text}\n\n{summary}".strip()

def main() -> None:
    parser = argparse.ArgumentParser(description="Decide whether a change passes, comparing two result databases.")
    parser.add_argument("baseline_db", help="database whose latest experiment is the baseline")
    parser.add_argument("candidate_db", help="database whose latest experiment is the change")
    parser.add_argument("--markdown", help="also write the report to this file, for a PR comment")
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")

    decision, markdown, text = run_gate(args.baseline_db, args.candidate_db)
    print(text)
    if args.markdown:
        Path(args.markdown).write_text(markdown, encoding="utf-8")
    sys.exit(decision.exit_code)

if __name__ == "__main__":
    main()
