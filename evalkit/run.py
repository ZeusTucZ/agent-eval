import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from evalkit import storage
from evalkit.cost import PRICES
from evalkit.models import AgentConfig, EvaluationCase, Experiment, Price, Trial, dataset_version
from evalkit.rag import DEFAULT_EMBEDDING, DEFAULT_TOP_K, EMBEDDINGS, MAX_TOP_K, retrieval_config
from evalkit.runner import Agent, run_experiment
from evalkit.sample_data import (
    arithmetic_cases,
    dataset_by_name,
    is_dataset_file,
    load_dataset_file,
    tools_for_dataset,
)
from evalkit.scorers import score_trials
from evalkit.tools import SYSTEM_PROMPT, ToolRegistry

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

def _real_agent_config(agent, provider: str, args) -> AgentConfig:
    params = {
        "provider": provider,
        "max_steps": agent.max_steps,
        "temperature": agent.temperature,
        "tools": sorted(agent.tools.tools),
        "retry_policy": asdict(agent.retry_policy),
        "requests_per_minute": agent.requests_per_minute,
    }
    # Only when set, so configs from before the token limit keep their versions.
    if agent.tokens_per_minute is not None:
        params["tokens_per_minute"] = agent.tokens_per_minute
    # Only for rag, so the other datasets' configs keep the versions their saved runs have.
    if args.dataset == "rag":
        params["retrieval"] = retrieval_config(args.top_k, args.embedding)
    return AgentConfig(model=agent.model, prompt=agent.system_prompt, params=params)

def _tools(args) -> ToolRegistry:
    return ToolRegistry(tools_for_dataset(args.dataset, top_k=args.top_k, embedding=args.embedding))

def _require_key(variable: str, where: str) -> None:
    import os

    if not os.environ.get(variable):
        raise SystemExit(f"Set the {variable} environment variable first (get a key at {where}).")

def _system_prompt(args) -> str:
    if args.prompt_file is None:
        return SYSTEM_PROMPT
    return Path(args.prompt_file).read_text(encoding="utf-8").strip()

def build_gemini(args) -> tuple[Agent, AgentConfig]:
    from evalkit.gemini_agent import GeminiAgent

    _require_key("GEMINI_API_KEY", "aistudio.google.com")
    agent = GeminiAgent(
        model=args.model,
        tools=_tools(args),
        system_prompt=_system_prompt(args),
        max_steps=args.max_steps,
        temperature=args.temperature,
        requests_per_minute=args.rpm or None,
        tokens_per_minute=args.tpm,
    )
    return agent, _real_agent_config(agent, "gemini", args)

def build_groq(args) -> tuple[Agent, AgentConfig]:
    from evalkit.openai_agent import OpenAICompatibleAgent

    _require_key("GROQ_API_KEY", "console.groq.com")
    agent = OpenAICompatibleAgent(
        model=args.model,
        tools=_tools(args),
        provider="groq",
        system_prompt=_system_prompt(args),
        max_steps=args.max_steps,
        temperature=args.temperature,
        requests_per_minute=args.rpm or None,
        tokens_per_minute=args.tpm,
    )
    return agent, _real_agent_config(agent, "groq", args)

def build_custom(args) -> tuple[Agent, AgentConfig]:
    """Your own agent, imported from --agent module.path:name (see evalkit.sdk)."""
    from evalkit.sdk import load_agent

    try:
        agent, source_hash = load_agent(args.agent, args.model)
    except (ImportError, ValueError) as error:
        raise SystemExit(f"Can't load the agent {args.agent!r}: {error}") from None
    # The source hash makes any edit to the agent's module a new version, so evalkit.compare
    # can tell two versions of your agent apart even when the model and prompt are the same.
    config = AgentConfig(model=args.model, prompt="", params={"agent": args.agent, "source": source_hash})
    return agent, config

def is_custom_agent(name: str) -> bool:
    return ":" in name

BUILDERS = {"mock": build_mock, "gemini": build_gemini, "groq": build_groq}

# Per-agent defaults for options left unset. Real agents' timeouts must outlast their retry
# schedule (up to 5 waits of 60 s), or a trial is cut off mid-retry; that is what happened
# to experiment 3 with an older 120 s default.
AGENT_DEFAULTS = {
    "mock": {"model": "mock-llm", "dataset": "arithmetic", "concurrency": 20, "timeout": 2.0, "rpm": 0},
    "gemini": {"model": "gemini-2.5-flash", "dataset": "mixed", "concurrency": 2, "timeout": 360.0, "rpm": 8},
    "groq": {"model": "openai/gpt-oss-20b", "dataset": "mixed", "concurrency": 4, "timeout": 360.0, "rpm": 25},
    # Your own agent manages its own rate limits, and needs a dataset written for it.
    "custom": {"model": None, "dataset": None, "concurrency": 4, "timeout": 360.0, "rpm": 0},
}

BUILT_IN_DATASETS = ("arithmetic", "mixed", "hard", "rag")

def load_dataset(args) -> tuple[str, list[EvaluationCase]]:
    if args.dataset == "arithmetic":
        return f"arithmetic-{args.cases}", arithmetic_cases(args.cases)
    if is_dataset_file(args.dataset):
        # The path as given is stored with the experiment, so report and compare find the
        # file again when they're run from the same place.
        return args.dataset, load_dataset_file(Path(args.dataset))
    return args.dataset, dataset_by_name(args.dataset)

def prices_for(args, trials: list[Trial] | None = None) -> dict[str, Price]:
    """The price table for this run: the built-in one plus --price for --model. With trials,
    any other model they called is added at $0 with a warning, rather than failing the
    report later on a run that has already been paid for."""
    prices = dict(PRICES)
    if args.price is not None:
        prices[args.model] = Price(input_per_million=args.price[0], output_per_million=args.price[1])
    if trials is not None:
        unpriced = sorted({span.name for trial in trials for span in trial.spans
                           if span.kind == "llm" and span.name not in prices})
        for model in unpriced:
            prices[model] = Price(input_per_million=0.0, output_per_million=0.0)
        if unpriced:
            print(f"WARNING: no price for {', '.join(unpriced)}; their cost is recorded as $0. "
                  "Name the model in llm_call() as --model, or add it to PRICES in evalkit/cost.py.")
    return prices

def main() -> None:
    parser = argparse.ArgumentParser(description="Run an agent over a dataset and save the results.")
    parser.add_argument("--name", default="smoke-test")
    parser.add_argument("--db", default="evalkit.db")
    parser.add_argument("--agent", default="mock",
                        help="mock, gemini, groq, or your own as module.path:function (see evalkit.sdk)")
    parser.add_argument("--model", help="default: mock-llm, gemini-2.5-flash or openai/gpt-oss-20b; "
                                        "required for your own agent")
    parser.add_argument("--price", type=float, nargs=2, metavar=("INPUT", "OUTPUT"),
                        help="dollars per million input and output tokens for --model, if it isn't in "
                             "evalkit/cost.py; 0 0 if you don't track cost")
    parser.add_argument("--dataset",
                        help="arithmetic, mixed, hard, rag, or the path of your own .jsonl file; "
                             "default: arithmetic for mock, mixed for gemini and groq")
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
    real.add_argument("--prompt-file", help="file holding the system prompt; default: the built-in prompt")
    real.add_argument("--rpm", type=int,
                      help="requests per minute to stay under; default 8 gemini, 25 groq; check your quota; 0 = no limit")

    real.add_argument("--tpm", type=int,
                      help="tokens per minute to stay under, e.g. your Groq model's TPM limit; default: no limit")

    rag = parser.add_argument_group("rag dataset (needs: pip install -e \".[rag]\")")
    rag.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, choices=range(1, MAX_TOP_K + 1), metavar="K",
                     help=f"documents each search returns, 1 to {MAX_TOP_K}; default {DEFAULT_TOP_K}")
    rag.add_argument("--embedding", choices=EMBEDDINGS, default=DEFAULT_EMBEDDING,
                     help="minilm: semantic, downloaded on first use; hashing: word matching, no download")
    args = parser.parse_args()

    custom = is_custom_agent(args.agent)
    if not custom and args.agent not in BUILDERS:
        raise SystemExit(f"Unknown agent {args.agent!r}: use {', '.join(sorted(BUILDERS))}, "
                         "or your own as module.path:function.")
    for option, default in AGENT_DEFAULTS["custom" if custom else args.agent].items():
        if getattr(args, option) is None:
            setattr(args, option, default)

    if custom and args.model is None:
        raise SystemExit("Name the model your agent calls with --model, so its tokens can be priced.")
    if custom and args.dataset is None:
        raise SystemExit("Pass --dataset: your agent needs questions written for it, e.g. --dataset my_cases.jsonl.")
    if args.dataset not in BUILT_IN_DATASETS and not is_dataset_file(args.dataset):
        raise SystemExit(f"Unknown dataset {args.dataset!r}: use {', '.join(BUILT_IN_DATASETS)}, "
                         "or the path of a .jsonl file.")

    # Check the price before spending anything: an unknown model would only fail at report time.
    if args.model not in prices_for(args):
        raise SystemExit(f"No price for {args.model!r}. Pass --price INPUT OUTPUT (dollars per million "
                         "tokens), or add it to PRICES in evalkit/cost.py.")

    if args.agent == "mock" and args.dataset != "arithmetic":
        raise SystemExit("The mock agent only understands the arithmetic dataset; use --agent groq or gemini.")

    try:
        dataset_name, dataset = load_dataset(args)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    agent, config = build_custom(args) if custom else BUILDERS[args.agent](args)

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
        dataset_version=dataset_version(dataset),
        config=config,
        attempts=args.attempts,
        prices=prices_for(args, trials),
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
