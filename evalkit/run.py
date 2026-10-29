import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone

from evalkit import storage
from evalkit.cost import PRICES
from evalkit.models import AgentConfig, EvaluationCase, Experiment
from evalkit.runner import Agent, run_experiment
from evalkit.sample_data import arithmetic_cases, mixed_cases
from evalkit.scorers import score_trials

def build_mock(args) -> tuple[Agent, AgentConfig]:
    from evalkit.mock_agent import FaultConfig, MockAgent

    faults = FaultConfig(
        tool_error_rate=args.tool_error_rate,
        rate_limit_rate=args.rate_limit_rate,
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
    return MockAgent(faults, model=args.model), config

def build_gemini(args) -> tuple[Agent, AgentConfig]:
    import os

    from evalkit.gemini_agent import GeminiAgent

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set the GEMINI_API_KEY environment variable first (get a key at aistudio.google.com).")

    agent = GeminiAgent(
        model=args.model,
        max_steps=args.max_steps,
        temperature=args.temperature,
        requests_per_minute=args.rpm or None,
    )
    config = AgentConfig(
        model=args.model,
        prompt=agent.system_prompt,
        params={
            "max_steps": agent.max_steps,
            "temperature": agent.temperature,
            "tools": sorted(agent.tools.tools),
            "retry_policy": asdict(agent.retry_policy),
            "requests_per_minute": agent.requests_per_minute,
        },
    )
    return agent, config

def load_dataset(args) -> tuple[str, list[EvaluationCase]]:
    if args.dataset == "mixed":
        return "mixed", mixed_cases()
    return f"arithmetic-{args.cases}", arithmetic_cases(args.cases)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run an agent over a dataset and save the results.")
    parser.add_argument("--name", default="smoke-test")
    parser.add_argument("--db", default="evalkit.db")
    parser.add_argument("--agent", choices=["mock", "gemini"], default="mock")
    parser.add_argument("--model", help="default: mock-llm for mock, gemini-2.5-flash for gemini")
    parser.add_argument("--dataset", choices=["arithmetic", "mixed"],
                        help="default: arithmetic for mock, mixed for gemini")
    parser.add_argument("--cases", type=int, default=100, help="size of the arithmetic dataset")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, help="default: 20 for mock, 2 for gemini")
    parser.add_argument("--timeout", type=float, help="seconds per trial; default: 2 for mock, 360 for gemini")

    mock = parser.add_argument_group("mock agent faults")
    mock.add_argument("--tool-error-rate", type=float, default=0.10)
    mock.add_argument("--rate-limit-rate", type=float, default=0.05)
    mock.add_argument("--hallucination-rate", type=float, default=0.15)
    mock.add_argument("--hang-rate", type=float, default=0.0)
    mock.add_argument("--seed", type=int, default=42)

    gemini = parser.add_argument_group("gemini agent")
    gemini.add_argument("--max-steps", type=int, default=5)
    gemini.add_argument("--temperature", type=float, default=0.0)
    gemini.add_argument("--rpm", type=int, default=8,
                        help="requests per minute to stay under (default 8; check your quota in AI Studio; 0 = no limit)")
    args = parser.parse_args()

    is_gemini = args.agent == "gemini"
    args.model = args.model or ("gemini-2.5-flash" if is_gemini else "mock-llm")
    args.dataset = args.dataset or ("mixed" if is_gemini else "arithmetic")
    args.concurrency = args.concurrency or (2 if is_gemini else 20)
    # Gemini's timeout must outlast its retry schedule (up to 5 waits of 60 s), or a trial is
    # cut off mid-retry; that is what happened to experiment 3 with the old 120 s default.
    args.timeout = args.timeout or (360.0 if is_gemini else 2.0)

    # Check the price before spending anything: an unknown model would only fail at report time.
    if args.model not in PRICES:
        raise SystemExit(f"No price for {args.model!r}. Add it to PRICES in evalkit/cost.py.")

    agent, config = build_gemini(args) if is_gemini else build_mock(args)
    dataset_name, dataset = load_dataset(args)

    trials = asyncio.run(run_experiment(
        agent,
        dataset,
        attempts=args.attempts,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
    ))
    scores = score_trials(dataset, trials)

    experiment = Experiment(
        name=args.name,
        dataset_name=dataset_name,
        config=config,
        attempts=args.attempts,
        prices=PRICES,
        created_at=datetime.now(timezone.utc),
    )
    experiment_id = storage.save_experiment(storage.connect(args.db), experiment, trials, scores)

    print(f"Saved experiment {experiment_id} ({len(trials)} trials) to {args.db}")

    quota_failures = sum(trial.failure_type == "quota_exhausted" for trial in trials)
    if quota_failures:
        print(f"WARNING: the daily quota ran out; {quota_failures} trials were not run. "
              "These results measure the quota, not the model. Re-run after it resets.")
    print(f"View it with: python -m evalkit.report {experiment_id} --db {args.db}")

if __name__ == "__main__":
    main()
