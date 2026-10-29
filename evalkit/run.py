import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone

from evalkit import storage
from evalkit.cost import PRICES
from evalkit.mock_agent import FaultConfig, MockAgent
from evalkit.models import AgentConfig, Experiment
from evalkit.runner import run_experiment
from evalkit.sample_data import arithmetic_cases
from evalkit.scorers import exact_match

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock agent over the arithmetic dataset and save the results.")
    parser.add_argument("--name", default="smoke-test")
    parser.add_argument("--db", default="evalkit.db")
    parser.add_argument("--model", default="mock-llm", choices=sorted(PRICES))
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=2.0, help="seconds per trial")
    parser.add_argument("--tool-error-rate", type=float, default=0.10)
    parser.add_argument("--hallucination-rate", type=float, default=0.15)
    parser.add_argument("--hang-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    faults = FaultConfig(
        tool_error_rate=args.tool_error_rate,
        hallucination_rate=args.hallucination_rate,
        hang_rate=args.hang_rate,
        seed=args.seed,
    )
    # The faults are part of the config, so changing one gives a new version.
    config = AgentConfig(
        model=args.model,
        prompt="Answer arithmetic questions using the calculator tool.",
        params=asdict(faults),
    )
    dataset = arithmetic_cases(args.cases)

    trials = asyncio.run(run_experiment(
        MockAgent(faults, model=args.model),
        dataset,
        attempts=args.attempts,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
    ))
    cases = {case.id: case for case in dataset}
    scores = [exact_match(cases[trial.case_id], trial) for trial in trials]

    experiment = Experiment(
        name=args.name,
        dataset_name=f"arithmetic-{args.cases}",
        config=config,
        attempts=args.attempts,
        prices=PRICES,
        created_at=datetime.now(timezone.utc),
    )
    experiment_id = storage.save_experiment(storage.connect(args.db), experiment, trials, scores)

    print(f"Saved experiment {experiment_id} ({len(trials)} trials) to {args.db}")
    print(f"View it with: python -m evalkit.report {experiment_id} --db {args.db}")

if __name__ == "__main__":
    main()
