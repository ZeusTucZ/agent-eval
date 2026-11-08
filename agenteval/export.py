"""Reports in other formats: JSON for tools, Markdown for PRs and notes, a standalone HTML
page to share, and CSV with one row per trial for spreadsheets and notebooks."""
import csv
import html
import io
import json

from agenteval.cost import trial_cost
from agenteval.models import EvaluationCase, Experiment, Score, Trial
from agenteval.report import DIAGNOSES, summary_data

CHECK_LABELS = {
    "tool_selection": "Used the right tool",
    "tool_arguments_valid": "Valid tool arguments",
    "tool_execution": "Tool ran without error",
    "trajectory_order": "Tools in the right order",
    "step_efficiency": "Step efficiency",
    "redundant_calls": "No repeated calls",
    "tool_faithfulness": "Answer backed by tool results",
    "retrieval_recall": "Search found the documents",
    "avoids_forbidden_text": "Ignored injected instructions",
}

def render(fmt: str, experiment: Experiment, trials: list[Trial], scores: list[Score],
           cases: list[EvaluationCase] | None) -> str:
    if fmt == "json":
        data = summary_data(experiment, trials, scores, cases)
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if fmt == "markdown":
        return to_markdown(summary_data(experiment, trials, scores, cases))
    if fmt == "html":
        return to_html(summary_data(experiment, trials, scores, cases), trial_rows(experiment, trials, scores, cases))
    if fmt == "csv":
        return to_csv(trial_rows(experiment, trials, scores, cases))
    raise ValueError(f"unknown format {fmt!r}")

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"

def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}" if value < 0.01 else f"${value:.4f}"

def trial_rows(experiment: Experiment, trials: list[Trial], scores: list[Score],
               cases: list[EvaluationCase] | None) -> list[dict]:
    """One flat row per trial: the question and expected answer (when the dataset can be
    found), what the agent said, how it went, and a column per scorer."""
    by_case = {case.id: case for case in cases or []}
    by_trial: dict[int, dict[str, float]] = {}
    for score in scores:
        by_trial.setdefault(score.trial_id, {})[score.scorer] = score.value
    scorer_names = sorted({score.scorer for score in scores})

    rows = []
    for trial in trials:
        case = by_case.get(trial.case_id)
        try:
            cost = trial_cost(trial, experiment.prices)
        except KeyError:
            cost = None
        row = {
            "trial": trial.id,
            "case": trial.case_id,
            "attempt": trial.attempt,
            "category": case.category if case else "",
            "question": case.input if case else "",
            "expected": case.expected_output if case else "",
            "answer": trial.output or "",
            "failure": trial.failure_type or "",
            "duration_ms": round(trial.duration_ms, 1),
            "cost_usd": cost,
            "steps": len(trial.spans),
        }
        row.update({name: by_trial.get(trial.id, {}).get(name, "") for name in scorer_names})
        rows.append(row)
    return rows

def to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()

def to_markdown(data: dict) -> str:
    lines = [
        f"# {data['name']}",
        "",
        f"**{data['model']}** on `{data['dataset']}`, {data['trials']} trials"
        + (f" (**{data['status']}**, {data['planned_trials']} planned)" if data["status"] != "completed" else ""),
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Pass rate | {_pct(data['pass_rate'])} |",
        f"| Latency p50 / p95 | {data['latency_ms']['p50']:.0f} / {data['latency_ms']['p95']:.0f} ms |",
        f"| Cost | {_usd(data['cost'])} total, {_usd(data['cost_per_success'])} per correct answer |",
    ]
    if data["by_category"]:
        lines += ["", "## By category", "", "| Category | Pass rate | Trials |", "|---|---|---|"]
        lines += [f"| {name} | {_pct(row['pass_rate'])} | {row['trials']} |" for name, row in data["by_category"].items()]
    checks = {name: value for name, value in data["checks"].items() if value is not None}
    if checks or data["other_scores"]:
        lines += ["", "## Checks", "", "| Check | Passed |", "|---|---|"]
        lines += [f"| {CHECK_LABELS[name]} | {_pct(value)} |" for name, value in checks.items()]
        lines += [f"| {name} | {_pct(row['mean'])} |" for name, row in data["other_scores"].items()]
    if data["retrieval_diagnosis"]:
        lines += ["", "## Search or answer?", "", "| Outcome | Trials |", "|---|---|"]
        lines += [f"| {DIAGNOSES[name]} | {data['retrieval_diagnosis'].get(name, 0)} |" for name in DIAGNOSES]
    if data["failures"]:
        lines += ["", "## Failures", "", "| Cause | Trials |", "|---|---|"]
        lines += [f"| {name} | {count} |" for name, count in sorted(data["failures"].items(), key=lambda x: -x[1])]
    return "\n".join(lines) + "\n"

_STYLE = """
:root{--bg:#fff;--fg:#18181b;--muted:#71717a;--line:#e4e4e7;--good:#15803d;--bad:#b91c1c;--accent:#4f46e5}
@media (prefers-color-scheme:dark){:root{--bg:#0f0f11;--fg:#f4f4f5;--muted:#a1a1aa;--line:#27272a;
--good:#4ade80;--bad:#f87171;--accent:#818cf8}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 16px}h1{font-size:1.6rem;margin:0}h2{font-size:1rem;margin:36px 0 10px}
.meta{color:var(--muted)}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0;
border:1px solid var(--line);border-radius:8px;margin-top:24px}.stat{padding:14px 18px;border-right:1px solid var(--line)}
.stat:last-child{border-right:0}.label{color:var(--muted);font-size:.8rem}.value{font-size:1.5rem;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:.9rem}th{text-align:left;color:var(--muted);font-weight:500}
th,td{padding:8px 10px 8px 0;border-bottom:1px solid var(--line);vertical-align:top}
.good{color:var(--good)}.bad{color:var(--bad)}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85rem}
.bar{height:6px;background:var(--line);border-radius:3px;min-width:80px}.bar>span{display:block;height:100%;
background:var(--accent);border-radius:3px}
"""

def to_html(data: dict, rows: list[dict]) -> str:
    """A standalone page: no scripts, no external files, light and dark themes."""
    e = html.escape

    def bar(value: float | None) -> str:
        return "" if value is None else f'<div class="bar"><span style="width:{value * 100:.0f}%"></span></div>'

    rate = data["pass_rate"]
    tone = "" if rate is None else "good" if rate >= 0.9 else "bad" if rate < 0.7 else ""
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{e(data['name'])}</title><style>{_STYLE}</style></head><body><main>",
        f"<h1>{e(data['name'])}</h1>",
        f"<p class='meta'>{e(data['model'])} · {e(data['dataset'])} · {data['trials']} trials · "
        f"{e(data['created_at'][:16].replace('T', ' '))}"
        + (f" · <strong>{e(data['status'])}</strong>" if data["status"] != "completed" else "") + "</p>",
        "<div class='stats'>",
        f"<div class='stat'><div class='label'>Pass rate</div><div class='value {tone}'>{_pct(data['pass_rate'])}</div></div>",
        f"<div class='stat'><div class='label'>Failures</div><div class='value'>{sum(data['failures'].values())}</div></div>",
        f"<div class='stat'><div class='label'>Latency p50</div><div class='value'>{data['latency_ms']['p50']:.0f} ms</div></div>",
        f"<div class='stat'><div class='label'>Cost per correct answer</div><div class='value'>{_usd(data['cost_per_success'])}</div></div>",
        "</div>",
    ]
    if data["by_category"]:
        parts.append("<h2>By category</h2><table><tr><th>Category</th><th>Pass rate</th><th></th><th>Trials</th></tr>")
        parts += [f"<tr><td>{e(name)}</td><td>{_pct(row['pass_rate'])}</td><td>{bar(row['pass_rate'])}</td>"
                  f"<td>{row['trials']}</td></tr>" for name, row in data["by_category"].items()]
        parts.append("</table>")
    checks = [(CHECK_LABELS[name], value) for name, value in data["checks"].items() if value is not None]
    checks += [(name, row["mean"]) for name, row in data["other_scores"].items()]
    if checks:
        parts.append("<h2>Checks</h2><table><tr><th>Check</th><th>Passed</th><th></th></tr>")
        parts += [f"<tr><td>{e(label)}</td><td>{_pct(value)}</td><td>{bar(value)}</td></tr>" for label, value in checks]
        parts.append("</table>")
    if data["failures"]:
        parts.append("<h2>Failures</h2><table><tr><th>Cause</th><th>Trials</th></tr>")
        parts += [f"<tr><td>{e(name)}</td><td>{count}</td></tr>"
                  for name, count in sorted(data["failures"].items(), key=lambda x: -x[1])]
        parts.append("</table>")
    not_passed = [row for row in rows if row.get("exact_match") != 1.0]
    if not_passed:
        parts.append(f"<h2>Trials that didn't pass ({len(not_passed)})</h2>")
        parts.append("<table><tr><th>Case</th><th>Question</th><th>Expected</th><th>Answer</th><th>Failure</th></tr>")
        parts += [f"<tr><td>{row['case']}·{row['attempt']}</td><td>{e(row['question'])}</td>"
                  f"<td class='mono'>{e(row['expected'])}</td><td class='mono'>{e(row['answer'][:200])}</td>"
                  f"<td class='bad'>{e(row['failure'])}</td></tr>" for row in not_passed[:200]]
        parts.append("</table>")
    parts.append("<p class='meta' style='margin-top:40px'>Generated by agenteval</p></main></body></html>")
    return "\n".join(parts)
