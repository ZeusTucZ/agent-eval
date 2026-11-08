"""Check an LLM judge against your own labels before trusting it.

A judge's pass rate means nothing until you know how often it agrees with a person.
First label a random sample of answers yourself, without seeing the judge's verdicts:

    agenteval calibrate label 12 --rubric groundedness --sample 50

Then measure agreement on the trials both of you judged:

    agenteval calibrate agree 12 --rubric groundedness

Labels are saved in the experiment's database as you go, so labelling can stop and resume,
and can also be done in the dashboard (agenteval ui). `agree --export labels.jsonl` writes
them to a file to keep in the repository; `--labels FILE` brings such a file back in.
"""
import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import Random

from agenteval import storage
from agenteval.judge import Rubric, applies, load_rubric, sources_of
from agenteval.models import EvaluationCase, Score, Trial
from agenteval.sample_data import cases_for
from agenteval.stats import cohens_kappa

def read_label_file(path: Path) -> dict[int, bool]:
    """Labels from a JSONL file (one {"trial_id", "label"} per line): trial id -> passed."""
    if not path.exists():
        raise ValueError(f"no label file at {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["trial_id"]: row["label"] == "pass" for row in rows}

def import_label_file(conn, experiment_id: int, rubric: str, path: Path) -> int:
    labels = read_label_file(path)
    for trial_id, passed in labels.items():
        storage.save_label(conn, experiment_id, trial_id, rubric, passed)
    return len(labels)

def export_label_file(conn, experiment_id: int, rubric: str, path: Path) -> int:
    labels = storage.load_labels(conn, experiment_id, rubric)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps({"experiment_id": experiment_id, "trial_id": trial_id, "rubric": rubric,
                    "label": "pass" if passed else "fail"}) + "\n"
        for trial_id, passed in sorted(labels.items())
    ), encoding="utf-8")
    return len(labels)

def sample_trials(trials: list[Trial], rubric: Rubric, size: int, seed: int) -> list[Trial]:
    """A fixed random sample of the trials the rubric applies to, so a resumed session
    asks about the same trials in the same order."""
    eligible = [trial for trial in trials if applies(rubric, trial)]
    Random(seed).shuffle(eligible)
    return eligible[:size]

def show(rubric: Rubric, case: EvaluationCase, trial: Trial) -> str:
    """What the labeller sees: the same material the judge gets, and nothing about its verdict."""
    lines = [f"Question:  {case.input}"]
    if rubric.uses_expected:
        lines.append(f"Expected:  {case.expected_output}")
    if rubric.uses_sources:
        lines.append(f"Sources:\n{sources_of(trial) or '(no tool results)'}")
    lines += [f"Answer:    {trial.output}", "", rubric.question]
    return "\n".join(lines)

def label(
    rubric: Rubric,
    cases: list[EvaluationCase],
    trials: list[Trial],
    conn,
    experiment_id: int,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> int:
    """Ask for a verdict on each sampled trial not yet labelled; returns how many were added."""
    by_id = {case.id: case for case in cases}
    done = storage.load_labels(conn, experiment_id, rubric.name)
    todo = [trial for trial in trials if trial.id not in done]

    added = 0
    for number, trial in enumerate(todo, start=len(done) + 1):
        say(f"\n--- {number} of {len(done) + len(todo)} (trial {trial.id}) ---")
        say(show(rubric, by_id[trial.case_id], trial))
        while True:
            reply = ask("[p]ass, [f]ail, [s]kip, [q]uit: ").strip().lower()[:1]
            if reply in ("p", "f", "s", "q"):
                break
        if reply == "q":
            break
        if reply == "s":
            continue
        # Saved at once, so quitting loses nothing.
        storage.save_label(conn, experiment_id, trial.id, rubric.name, reply == "p")
        added += 1
    return added

@dataclass
class Agreement:
    compared: int
    agreement: float  # share of trials where judge and human agree
    kappa: float
    human_pass_rate: float
    judge_pass_rate: float
    # (human, judge) -> count, e.g. (True, False): the human passed it, the judge failed it
    confusion: dict[tuple[bool, bool], int]
    disagreements: list[tuple[int, bool, Score]]  # (trial id, human verdict, the judge's score)

def agreement(labels: dict[int, bool], judge_scores: list[Score]) -> Agreement:
    by_trial = {score.trial_id: score for score in judge_scores}
    both = sorted(trial_id for trial_id in labels if trial_id in by_trial)
    if not both:
        raise ValueError("no trial has both a human label and a judge score")

    human = [labels[trial_id] for trial_id in both]
    judged = [by_trial[trial_id].value == 1.0 for trial_id in both]
    confusion = {(h, j): sum(1 for a, b in zip(human, judged, strict=True) if (a, b) == (h, j))
                 for h in (True, False) for j in (True, False)}
    return Agreement(
        compared=len(both),
        agreement=sum(a == b for a, b in zip(human, judged, strict=True)) / len(both),
        kappa=cohens_kappa(human, judged),
        human_pass_rate=sum(human) / len(both),
        judge_pass_rate=sum(judged) / len(both),
        confusion=confusion,
        disagreements=[(trial_id, h, by_trial[trial_id])
                       for trial_id, h, j in zip(both, human, judged, strict=True) if h != j],
    )

def verdict(kappa: float, compared: int) -> str:
    """Conventional reading of kappa (Landis and Koch), with a warning for small samples."""
    if kappa >= 0.8:
        reading = "strong agreement: the judge can stand in for a person on this rubric"
    elif kappa >= 0.6:
        reading = "substantial agreement: usable for comparing versions, but read its failures"
    elif kappa >= 0.4:
        reading = "moderate agreement: treat its scores as leads, not measurements"
    else:
        reading = "weak agreement: don't rely on this judge; change its prompt or model"
    if compared < 30:
        reading += f" (only {compared} trials compared: label more before deciding)"
    return reading

def format_agreement(result: Agreement) -> str:
    c = result.confusion
    lines = [
        f"Trials compared:  {result.compared}",
        f"Agreement:        {result.agreement:.1%}",
        f"Cohen's kappa:    {result.kappa:.2f}  {verdict(result.kappa, result.compared)}",
        f"Pass rate:        human {result.human_pass_rate:.1%}, judge {result.judge_pass_rate:.1%}",
        "",
        "                  judge pass   judge fail",
        f"  human pass      {c[(True, True)]:>10}   {c[(True, False)]:>10}",
        f"  human fail      {c[(False, True)]:>10}   {c[(False, False)]:>10}",
    ]
    if result.disagreements:
        lines += ["", "Disagreements (read these to fix the rubric):"]
        lines += [f"  trial {trial_id}: you said {'pass' if human else 'fail'}, {score.reasoning}"
                  for trial_id, human, score in result.disagreements]
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval calibrate", description="Label answers yourself and measure an LLM judge's agreement.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("label", "label a random sample of answers"),
                            ("agree", "compare your labels with the judge's scores")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("experiment_id", type=int)
        command.add_argument("--rubric", required=True, help="a built-in rubric or your rubric .toml file")
        command.add_argument("--db", default="agenteval.db")
        command.add_argument("--labels", type=Path, help="first import labels from this JSONL file")
        if name == "label":
            command.add_argument("--sample", type=int, default=50, help="how many answers to label")
            command.add_argument("--seed", type=int, default=0)
        else:
            command.add_argument("--export", type=Path, help="also write your labels to this JSONL file")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(errors="replace")
    conn = storage.connect(args.db)
    try:
        rubric = load_rubric(args.rubric)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if args.labels:
        try:
            print(f"Imported {import_label_file(conn, args.experiment_id, rubric.name, args.labels)} labels")
        except (ValueError, KeyError) as error:
            raise SystemExit(f"Can't read {args.labels}: {error}") from None

    if args.command == "label":
        experiment = storage.load_experiment(conn, args.experiment_id)
        trials = sample_trials(storage.load_trials(conn, args.experiment_id), rubric, args.sample, args.seed)
        added = label(rubric, cases_for(conn, experiment) or [], trials, conn, args.experiment_id)
        total = len(storage.load_labels(conn, args.experiment_id, rubric.name))
        print(f"\nSaved {added} new labels ({total} in total for {rubric.name})")
        return

    judge_scores = [score for score in storage.load_scores(conn, args.experiment_id)
                    if score.scorer == f"judge_{rubric.name}"]
    if not judge_scores:
        raise SystemExit(f"No judge scores yet: agenteval judge {args.experiment_id} --rubric {rubric.name}")
    labels = storage.load_labels(conn, args.experiment_id, rubric.name)
    if not labels:
        raise SystemExit(f"No labels yet: agenteval calibrate label {args.experiment_id} --rubric {rubric.name}")
    if args.export:
        print(f"Wrote {export_label_file(conn, args.experiment_id, rubric.name, args.export)} labels to {args.export}")
    print(format_agreement(agreement(labels, judge_scores)))

if __name__ == "__main__":
    main()
