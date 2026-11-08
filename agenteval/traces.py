"""agenteval import and agenteval export: score traces you already have, and take results out.

Import evaluates an agent without running it: point it at a JSONL file of what the agent
did (production logs, another framework's traces) and every scorer, the judge, reports
and comparisons work on it. One record per line, in either form:

    {"input": "...", "output": "...", "expected_output": "...", "category": "...",
     "steps": [{"kind": "llm", "name": "gpt-4o-mini", "input_tokens": 812, "output_tokens": 40,
                "duration_ms": 950},
               {"kind": "tool", "name": "search", "input": {"q": "..."}, "output": [...],
                "error": null, "duration_ms": 120}]}

    {"user_input": "...", "response": "...", "retrieved_contexts": ["...", "..."], "reference": "..."}

The second is a Ragas sample: retrieved_contexts become a "retrieve" step, so groundedness
and context relevance can be judged on them. Only the question and the answer are required;
without an expected answer exact_match is skipped rather than failed.

Export writes an experiment back out in the first form (so it round-trips), or as Ragas
samples, to use Ragas's own metrics on runs made here.
"""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agenteval import storage
from agenteval.cost import PRICES
from agenteval.models import AgentConfig, EvaluationCase, Experiment, Price, Span, Trial, dataset_version
from agenteval.sample_data import cases_for
from agenteval.scorers import DEFAULT_SCORERS, score_trials
from agenteval.tracer import clip

RETRIEVE_STEP = "retrieve"

def _text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)

def parse_record(record: dict, number: int) -> tuple[EvaluationCase, Trial]:
    """One imported line -> a case and the trial that answered it."""
    question = record.get("input", record.get("user_input"))
    if not isinstance(question, str) or not question:
        raise ValueError('needs "input" (or Ragas\'s "user_input") with the question')
    answer = record.get("output", record.get("response"))
    case = EvaluationCase(
        id=int(record.get("id", number)),
        input=question,
        expected_output=str(record.get("expected_output", record.get("reference")) or ""),
        expected_tools=record.get("expected_tools"),
        category=str(record.get("category", "general")),
        must_not_contain=record.get("must_not_contain", []),
        relevant_docs=record.get("relevant_docs"),
    )

    clock = 0  # imported steps have durations, not timestamps: lay them end to end
    spans = []
    steps = list(record.get("steps", []))
    if record.get("retrieved_contexts") is not None:
        steps.insert(0, {"kind": "tool", "name": RETRIEVE_STEP, "input": {"query": question},
                         "output": record["retrieved_contexts"]})
    for step in steps:
        kind = step.get("kind", "tool")
        if kind not in ("llm", "tool"):
            raise ValueError(f'step kind must be "llm" or "tool", got {kind!r}')
        duration = int(float(step.get("duration_ms", 0)) * 1_000_000)
        error = step.get("error")
        spans.append(Span(
            trial_id=number, kind=kind, name=str(step.get("name", "unknown")),
            start=clock, end=clock + duration, status="error" if error else "success",
            error_type=str(step.get("error_type") or ("Error" if error else "")) or None,
            error_message=clip(str(error)) if error else None,
            input=clip(_text(step.get("input")) or "") if kind == "tool" else None,
            output=clip(_text(step.get("output")) or "", 8000) if kind == "tool" and not error else None,
            input_tokens=int(step.get("input_tokens", 0)), output_tokens=int(step.get("output_tokens", 0)),
        ))
        clock += duration

    trial = Trial(id=number, case_id=case.id, attempt=int(record.get("attempt", 1)),
                  output=None if answer is None else str(answer),
                  failure_type=record.get("failure_type") or (None if answer else "empty_output"), spans=spans)
    return case, trial

def import_traces(conn, path: Path, name: str, model: str, cases_out: Path,
                  prices: dict[str, Price] | None = None, fmt: str = "auto") -> int:
    """fmt: "jsonl" (agenteval or Ragas records), "otlp" (OpenTelemetry JSON), or "auto"."""
    from agenteval.otel import is_otlp, records_from_otlp

    records = []
    if fmt == "otlp" or (fmt == "auto" and is_otlp(path)):
        for number, record in enumerate(records_from_otlp(path), start=1):
            records.append(parse_record(record, number))
    else:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                try:
                    records.append(parse_record(json.loads(line), len(records) + 1))
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}, line {number}: {error}") from None
    if not records:
        raise ValueError(f"{path} has no records (for OpenTelemetry traces: no trace had a root span "
                         "with input.value or agenteval.input)")

    # One case per distinct question, so repeated attempts at the same question pair up.
    cases: dict[int, EvaluationCase] = {}
    by_question: dict[str, int] = {}
    trials = []
    for case, trial in records:
        case_id = by_question.setdefault(case.input, case.id)
        cases.setdefault(case_id, case.model_copy(update={"id": case_id}))
        trials.append(trial.model_copy(update={"case_id": case_id}))
    case_list = list(cases.values())

    # The cases are written next to the experiment, so report, rescore and compare find them.
    cases_out.parent.mkdir(parents=True, exist_ok=True)
    cases_out.write_text("".join(case.model_dump_json(exclude_none=True) + "\n" for case in case_list),
                         encoding="utf-8")

    all_prices = dict(prices or PRICES)
    for trial in trials:
        for span in trial.spans:
            if span.kind == "llm":
                all_prices.setdefault(span.name, Price(input_per_million=0.0, output_per_million=0.0))
    experiment = Experiment(
        name=name, dataset_name=cases_out.as_posix(), dataset_version=dataset_version(case_list),
        config=AgentConfig(model=model, prompt="", params={"imported_from": path.name}),
        attempts=max(trial.attempt for trial in trials), prices=all_prices, created_at=datetime.now(UTC),
        planned_trials=len(trials),
    )
    return storage.save_experiment(conn, experiment, trials, score_trials(case_list, trials, DEFAULT_SCORERS),
                                   case_list)

def export_records(experiment: Experiment, trials: list[Trial], cases: list[EvaluationCase] | None,
                   fmt: str) -> list[dict]:
    by_case = {case.id: case for case in cases or []}
    records: list[dict[str, Any]] = []
    for trial in trials:
        case = by_case.get(trial.case_id)
        tool_outputs = [span.output for span in trial.spans if span.kind == "tool" and span.output]
        if fmt == "ragas":
            records.append({
                "user_input": case.input if case else "",
                "response": trial.output or "",
                "retrieved_contexts": tool_outputs,
                **({"reference": case.expected_output} if case and case.expected_output else {}),
            })
            continue
        records.append({
            "id": trial.id, "attempt": trial.attempt,
            "input": case.input if case else "", "output": trial.output,
            **({"expected_output": case.expected_output, "category": case.category} if case else {}),
            "failure_type": trial.failure_type,
            "steps": [{
                "kind": span.kind, "name": span.name, "duration_ms": span.duration_ms,
                "input_tokens": span.input_tokens, "output_tokens": span.output_tokens,
                **({"input": span.input, "output": span.output} if span.kind == "tool" else {}),
                **({"error": span.error_message or span.error_type} if span.status == "error" else {}),
            } for span in trial.spans],
        })
    return records

def main_import(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval import", description="Import recorded traces as an experiment.")
    parser.add_argument("traces", type=Path, help="JSONL records, or OpenTelemetry traces as OTLP JSON")
    parser.add_argument("--format", choices=["auto", "jsonl", "otlp"], default="auto",
                        help="otlp: OpenTelemetry / OpenInference traces, one trace per trial")
    parser.add_argument("--name", help="default: the file's name")
    parser.add_argument("--model", default="imported", help="the model that answered, for reports")
    parser.add_argument("--cases-out", type=Path, help="where to write the questions; default: next to the traces")
    parser.add_argument("--db", default="agenteval.db")
    args = parser.parse_args(argv)

    cases_out = args.cases_out or args.traces.with_name(args.traces.stem + ".cases.jsonl")
    try:
        experiment_id = import_traces(storage.connect(args.db), args.traces, args.name or args.traces.stem,
                                      args.model, cases_out, fmt=args.format)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    print(f"Imported {args.traces} as experiment {experiment_id} (questions in {cases_out}).\n"
          f"See it with: agenteval report {experiment_id} --db {args.db}\n"
          f"Judge it with: agenteval judge {experiment_id} --rubric answer_relevance --db {args.db}")

def main_export(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval export", description="Export an experiment's trials as JSONL.")
    parser.add_argument("experiment_id", type=int)
    parser.add_argument("--format", choices=["jsonl", "ragas"], default="jsonl",
                        help="jsonl: agenteval's import format; ragas: SingleTurnSample fields")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--db", default="agenteval.db")
    args = parser.parse_args(argv)

    conn = storage.connect(args.db)
    try:
        experiment = storage.load_experiment(conn, args.experiment_id)
    except LookupError as error:
        raise SystemExit(str(error)) from None
    cases = cases_for(conn, experiment)
    records = export_records(experiment, storage.load_trials(conn, args.experiment_id), cases, args.format)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"Wrote {len(records)} {args.format} records to {args.output}")

main = main_import
