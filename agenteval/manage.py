"""agenteval list and agenteval delete: see and tidy the saved experiments."""
import argparse
import sys

from agenteval import storage
from agenteval.report import mean_score

def main_list(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval list", description="List saved experiments, newest first.")
    parser.add_argument("--db", default="agenteval.db")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    conn = storage.connect(args.db)
    ids = list(reversed(storage.list_experiment_ids(conn)))[: args.limit]
    if not ids:
        print(f"No experiments in {args.db} yet: run one with `agenteval run`.")
        return

    rows = []
    for experiment_id in ids:
        experiment = storage.load_experiment(conn, experiment_id)
        scores = [s for s in storage.load_scores(conn, experiment_id) if s.scorer == "exact_match"]
        trials = conn.execute("SELECT COUNT(*) FROM trials WHERE experiment_id = ?", (experiment_id,)).fetchone()[0]
        status = "" if experiment.complete else f" ({experiment.status})"
        rows.append((str(experiment_id), experiment.name + status, experiment.config.model, experiment.dataset_name,
                     str(trials), f"{mean_score(scores):.1%}" if scores else "n/a",
                     experiment.created_at.strftime("%Y-%m-%d %H:%M")))

    header = ("ID", "Name", "Model", "Dataset", "Trials", "Pass", "Created")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip())

def main_delete(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval delete", description="Delete an experiment and everything saved with it.")
    parser.add_argument("experiment_id", type=int)
    parser.add_argument("--db", default="agenteval.db")
    parser.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    args = parser.parse_args(argv)

    conn = storage.connect(args.db)
    try:
        experiment = storage.load_experiment(conn, args.experiment_id)
    except LookupError as error:
        raise SystemExit(str(error)) from None
    if not args.yes:
        reply = input(f"Delete experiment {args.experiment_id} ({experiment.name}) and all its trials? "
                      "This can't be undone. [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Nothing deleted.")
            return
    storage.delete_experiment(conn, args.experiment_id)
    print(f"Deleted experiment {args.experiment_id}.")

main = main_list
