import argparse

from agenteval import storage
from agenteval.sample_data import cases_for
from agenteval.scorers import DEFAULT_SCORERS, SCORER_NAMES, load_scorer, score_trials

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval rescore",
        description="Re-run the scorers on a saved experiment without calling the agent again.",
    )
    parser.add_argument("experiment_id", type=int)
    parser.add_argument("--db", default="agenteval.db")
    args = parser.parse_args(argv)

    conn = storage.connect(args.db)
    experiment = storage.load_experiment(conn, args.experiment_id)
    trials = storage.load_trials(conn, args.experiment_id)

    try:
        custom = [load_scorer(spec) for spec in experiment.custom_scorers]
    except (ImportError, ValueError) as error:
        raise SystemExit(f"Can't load this experiment's scorers {experiment.custom_scorers}: {error}") from None
    cases = cases_for(conn, experiment)
    if cases is None:
        raise SystemExit(f"Can't find the cases of experiment {args.experiment_id} ({experiment.dataset_name}).")
    scores = score_trials(cases, trials, [*DEFAULT_SCORERS, *custom])
    # Only the built-in scorers' and your own scorers' scores are replaced; an LLM judge's
    # (agenteval judge) are kept, since they cost money to make.
    storage.replace_scores(conn, args.experiment_id, scores, [*SCORER_NAMES, *(s.__name__ for s in custom)])

    print(f"Re-scored {len(trials)} trials of experiment {args.experiment_id} ({len(scores)} scores)")

if __name__ == "__main__":
    main()
