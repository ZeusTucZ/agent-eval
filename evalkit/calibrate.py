"""Check an LLM judge against your own labels before trusting it.

A judge's pass rate means nothing until you know how often it agrees with a person.
First label a random sample of answers yourself, without seeing the judge's verdicts:

    python -m evalkit.calibrate label 12 --rubric groundedness --sample 50

Then measure agreement on the trials both of you judged:

    python -m evalkit.calibrate agree 12 --rubric groundedness

Labels are saved one per line as you go (in labels/ by default), so labelling can stop
and resume, and the file can be kept in the repository next to the dataset.
"""
import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import Random

from evalkit import storage
from evalkit.judge import RUBRICS, Rubric, applies, sources_of
from evalkit.models import EvaluationCase, Score, Trial
from evalkit.sample_data import dataset_by_name
from evalkit.stats import cohens_kappa

LABELS_DIR = Path("labels")

def labels_path(experiment_id: int, rubric: str) -> Path:
    return LABELS_DIR / f"experiment-{experiment_id}-{rubric}.jsonl"

def load_labels(path: Path) -> dict[int, bool]:
    """trial id -> the human's verdict (True for pass)."""
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["trial_id"]: row["label"] == "pass" for row in rows}

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
    path: Path,
    experiment_id: int,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> int:
    """Ask for a verdict on each sampled trial not yet labelled; returns how many were added."""
    by_id = {case.id: case for case in cases}
    done = load_labels(path)
    todo = [trial for trial in trials if trial.id not in done]
    path.parent.mkdir(parents=True, exist_ok=True)

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
        row = {"experiment_id": experiment_id, "trial_id": trial.id, "rubric": rubric.name,
               "label": "pass" if reply == "p" else "fail"}
        with path.open("a", encoding="utf-8") as file:  # saved at once, so quitting loses nothing
            file.write(json.dumps(row) + "\n")
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Label answers yourself and measure an LLM judge's agreement.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("label", "label a random sample of answers"),
                            ("agree", "compare your labels with the judge's scores")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("experiment_id", type=int)
        command.add_argument("--rubric", choices=sorted(RUBRICS), required=True)
        command.add_argument("--db", default="evalkit.db")
        command.add_argument("--labels", type=Path, help="label file; default: labels/experiment-<id>-<rubric>.jsonl")
        if name == "label":
            command.add_argument("--sample", type=int, default=50, help="how many answers to label")
            command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")
    conn = storage.connect(args.db)
    rubric = RUBRICS[args.rubric]
    path = args.labels or labels_path(args.experiment_id, rubric.name)

    if args.command == "label":
        experiment = storage.load_experiment(conn, args.experiment_id)
        trials = sample_trials(storage.load_trials(conn, args.experiment_id), rubric, args.sample, args.seed)
        added = label(rubric, dataset_by_name(experiment.dataset_name), trials, path, args.experiment_id)
        print(f"\nSaved {added} new labels to {path} ({len(load_labels(path))} in total)")
        return

    judge_scores = [score for score in storage.load_scores(conn, args.experiment_id)
                    if score.scorer == f"judge_{rubric.name}"]
    if not judge_scores:
        raise SystemExit(f"No judge scores yet: python -m evalkit.judge {args.experiment_id} --rubric {rubric.name}")
    print(format_agreement(agreement(load_labels(path), judge_scores)))

if __name__ == "__main__":
    main()
