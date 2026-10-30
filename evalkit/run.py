import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone

from evalkit import storage
from evalkit.cost import PRICES
from evalkit.models import AgentConfig, EvaluationCase, Experiment
from evalkit.runner import Agent, run_experiment
from evalkit.sample_data import arithmetic_cases, dataset_by_name, tools_for_dataset
from evalkit.scorers import score_trials
from evalkit.tools import ToolRegistry

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

def _real_agent_config(agent, provider: str) -> AgentConfig:
    return AgentConfig(
        model=agent.model,
        prompt=agent.system_prompt,
        params={
            "provider": provider,
            "max_steps": agent.max_steps,
            "temperature": agent.temperature,
            "tools": sorted(agent.tools.tools),
            "retry_policy": asdict(agent.retry_policy),
            "requests_per_minute": agent.requests_per_minute,
        },
    )

def _require_key(variable: str, where: str) -> None:
    import os

    if not os.environ.get(variable):
        raise SystemExit(f"Set the {variable} environment variable first (get a key at {where}).")

def build_gemini(args) -> tuple[Agent, AgentConfig]:
    from evalkit.gemini_agent import GeminiAgent

    _require_key("GEMINI_API_KEY", "aistudio.google.com")
    agent = GeminiAgent(
        model=args.model,
        tools=ToolRegistry(tools_for_dataset(args.dataset)),
        max_steps=args.max_steps,
        temperature=args.temperature,
        requests_per_minute=args.rpm or None,
    )
    return agent, _real_agent_config(agent, "gemini")

def build_groq(args) -> tuple[Agent, AgentConfig]:
    from evalkit.openai_agent import OpenAICompatibleAgent

    _require_key("GROQ_API_KEY", "console.groq.com")
    agent = OpenAICompatibleAgent(
        model=args.model,
        tools=ToolRegistry(tools_for_dataset(args.dataset)),
        provider="groq",
        max_steps=args.max_steps,
        temperature=args.temperature,
        requests_per_minute=args.rpm or None,
    )
    return agent, _real_agent_config(agent, "groq")

BUILDERS = {"mock": build_mock, "gemini": build_gemini, "groq": build_groq}

# Per-agent defaults for options left unset. Real agents' timeouts must outlast their retry
# schedule (up to 5 waits of 60 s), or a trial is cut off mid-retry; that is what happened
# to experiment 3 with an older 120 s default.
AGENT_DEFAULTS = {
    "mock": {"model": "mock-llm", "dataset": "arithmetic", "concurrency": 20, "timeout": 2.0, "rpm": 0},
    "gemini": {"model": "gemini-2.5-flash", "dataset": "mixed", "concurrency": 2, "timeout": 360.0, "rpm": 8},
    "groq": {"model": "openai/gpt-oss-20b", "dataset": "mixed", "concurrency": 4, "timeout": 360.0, "rpm": 25},
}

def load_dataset(args) -> tuple[str, list[EvaluationCase]]:
    if args.dataset == "arithmetic":
        return f"arithmetic-{args.cases}", arithmetic_cases(args.cases)
    return args.dataset, dataset_by_name(args.dataset)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run an agent over a dataset and save the results.")
    parser.add_argument("--name", default="smoke-test")
    parser.add_argument("--db", default="evalkit.db")
    parser.add_argument("--agent", choices=sorted(BUILDERS), default="mock")
    parser.add_argument("--model", help="default: mock-llm, gemini-2.5-flash or openai/gpt-oss-20b")
    parser.add_argument("--dataset", choices=["arithmetic", "mixed", "hard"],
                        help="default: arithmetic for mock, mixed for real agents")
    parser.add_argument("--cases", type=int, default=100, help="size of the arithmetic dataset")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, help="default: 20 mock, 2 gemini, 4 groq")
    parser.add_argument("--timeout", type=float, help="seconds per trial; default: 2 for mock, 360 for real agents")

    mock = parser.add_argument_group("mock agent faults")
    mock.add_argument("--tool-error-rate", type=float, default=0.10)
    mock.add_argument("--rate-limit-rate", type=float, default=0.05)
    mock.add_argument("--hallucination-rate", type=float, default=0.15)
    mock.add_argument("--hang-rate", type=float, default=0.0)
    mock.add_argument("--seed", type=int, default=42)

    real = parser.add_argument_group("real agents (gemini, groq)")
    real.add_argument("--max-steps", type=int, default=5)
    real.add_argument("--temperature", type=float, default=0.0)
    real.add_argument("--rpm", type=int,
                      help="requests per minute to stay under; default 8 gemini, 25 groq; check your quota; 0 = no limit")
    args = parser.parse_args()

    for option, default in AGENT_DEFAULTS[args.agent].items():
        if getattr(args, option) is None:
            setattr(args, option, default)

    # Check the price before spending anything: an unknown model would only fail at report time.
    if args.model not in PRICES:
        raise SystemExit(f"No price for {args.model!r}. Add it to PRICES in evalkit/cost.py.")

    if args.agent == "mock" and args.dataset != "arithmetic":
        raise SystemExit("The mock agent only understands the arithmetic dataset; use --agent groq or gemini.")

    agent, config = BUILDERS[args.agent](args)
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
