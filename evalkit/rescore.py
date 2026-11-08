import argparse

from evalkit import storage
from evalkit.sample_data import dataset_by_name
from evalkit.scorers import SCORER_NAMES, score_trials

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run the scorers on a saved experiment without calling the agent again.",
    )
    parser.add_argument("experiment_id", type=int)
    parser.add_argument("--db", default="evalkit.db")
    args = parser.parse_args()

    conn = storage.connect(args.db)
    experiment = storage.load_experiment(conn, args.experiment_id)
    trials = storage.load_trials(conn, args.experiment_id)

    scores = score_trials(dataset_by_name(experiment.dataset_name), trials)
    # Only the built-in scorers' scores are replaced; an LLM judge's (evalkit.judge) are kept.
    storage.replace_scores(conn, args.experiment_id, scores, SCORER_NAMES)

    print(f"Re-scored {len(trials)} trials of experiment {args.experiment_id} ({len(scores)} scores)")

if __name__ == "__main__":
    main()
