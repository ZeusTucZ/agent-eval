"""agenteval prices and agenteval estimate: what models cost, and what a run will cost.

`prices --update` fetches the price list LiteLLM maintains for thousands of models and saves
it to ~/.agenteval/prices.json, which takes precedence over the built-in table. `estimate`
predicts a run's cost and time from earlier runs of the same model, before you spend anything.
"""
import argparse
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from agenteval import storage
from agenteval.cost import BUILT_IN_PRICES, PRICES, PRICES_CHECKED, load_user_prices, trial_cost, user_prices_file
from agenteval.models import Price

LITELLM_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# LiteLLM names many models "provider/model"; agents call them by the bare name the
# provider's API uses, so both are kept for these providers.
STRIP_PROVIDERS = ("groq/", "gemini/", "openrouter/", "together_ai/", "fireworks_ai/", "deepinfra/", "mistral/")

def parse_litellm(data: dict) -> dict[str, Price]:
    """Chat models with both prices, per million tokens, under their own and bare names."""
    prices: dict[str, Price] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict) or entry.get("mode") != "chat":
            continue
        input_cost, output_cost = entry.get("input_cost_per_token"), entry.get("output_cost_per_token")
        if not isinstance(input_cost, int | float) or not isinstance(output_cost, int | float):
            continue
        price = Price(input_per_million=round(input_cost * 1e6, 6), output_per_million=round(output_cost * 1e6, 6))
        prices[name] = price
        for prefix in STRIP_PROVIDERS:
            if name.startswith(prefix):
                prices.setdefault(name.removeprefix(prefix), price)
    return prices

def update_prices(url: str = LITELLM_PRICES_URL, path: Path | None = None) -> int:
    path = path or user_prices_file()
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - a fixed https URL
        prices = parse_litellm(json.load(response))
    if not prices:
        raise RuntimeError("the price list had no chat models; it may have changed format")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "source": url,
        "updated": datetime.now(UTC).date().isoformat(),
        "prices": {model: price.model_dump() for model, price in sorted(prices.items())},
    }, indent=1), encoding="utf-8")
    return len(prices)

def _age_in_months(month: str, today: datetime) -> int:
    year, number = (int(part) for part in month[:7].split("-"))
    return (today.year - year) * 12 + today.month - number

def main_prices(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval prices", description="Show or update model prices.")
    parser.add_argument("search", nargs="?", help="show models whose name contains this")
    parser.add_argument("--update", action="store_true",
                        help="fetch current prices from LiteLLM's list into ~/.agenteval/prices.json "
                             "(or AGENTEVAL_PRICES_FILE)")
    args = parser.parse_args(argv)

    if args.update:
        try:
            count = update_prices()
        except (OSError, RuntimeError, ValueError) as error:
            raise SystemExit(f"Couldn't update prices: {error}") from None
        print(f"Saved prices for {count} models to {user_prices_file()}; they now take precedence.")
        return

    updated = None
    try:
        updated = json.loads(user_prices_file().read_text(encoding="utf-8")).get("updated")
    except (OSError, ValueError):
        pass
    today = datetime.now(UTC)
    checked = updated or PRICES_CHECKED
    age = _age_in_months(checked, today)
    source = f"updated {updated}" if updated else f"built-in, checked {PRICES_CHECKED}"
    print(f"Prices per million tokens ({source}; {len(PRICES)} models)")
    if age >= 3:
        print(f"  These are {age} months old: run `agenteval prices --update`.")

    shown = {name: price for name, price in PRICES.items() if not args.search or args.search.lower() in name.lower()}
    if not args.search:
        shown = {name: price for name, price in shown.items() if name in BUILT_IN_PRICES or len(PRICES) < 40}
    for name, price in sorted(shown.items()):
        print(f"  {name:<40} ${price.input_per_million:>8.3f} in   ${price.output_per_million:>8.3f} out")
    if not args.search and len(PRICES) >= 40:
        print(f"  ... and {len(PRICES) - len(shown)} more: `agenteval prices <name>` to search.")

def estimate(conn, model: str, planned_trials: int, dataset: str | None = None) -> dict | None:
    """Mean cost and duration per trial from earlier runs of `model` (same dataset preferred)."""
    samples = []
    for experiment_id in storage.list_experiment_ids(conn):
        experiment = storage.load_experiment(conn, experiment_id)
        if experiment.config.model != model:
            continue
        trials = storage.load_trials(conn, experiment_id)
        same_dataset = dataset is not None and experiment.dataset_name.split("#")[0] == dataset
        for trial in trials:
            try:
                samples.append((same_dataset, trial_cost(trial, experiment.prices), trial.duration_ms))
            except KeyError:
                continue
    if not samples:
        return None
    matching = [sample for sample in samples if sample[0]] or samples
    cost = sum(sample[1] for sample in matching) / len(matching)
    duration = sum(sample[2] for sample in matching) / len(matching)
    return {"trials_seen": len(matching), "same_dataset": bool(matching[0][0]), "cost_per_trial": cost,
            "duration_ms_per_trial": duration, "cost": cost * planned_trials,
            "trial_seconds": duration * planned_trials / 1000}

def main_estimate(argv: list[str] | None = None) -> None:
    from agenteval.sample_data import dataset_by_name

    parser = argparse.ArgumentParser(prog="agenteval estimate",
                                     description="Estimate a run's cost and time from earlier runs of the same model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="a built-in dataset or your .jsonl file")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--sample", type=int, help="as in agenteval run")
    parser.add_argument("--cases", type=int, default=100, help="size of the arithmetic dataset")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--db", default="agenteval.db")
    args = parser.parse_args(argv)

    name = f"arithmetic-{args.cases}" if args.dataset == "arithmetic" else args.dataset
    try:
        cases = len(dataset_by_name(name))
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    if args.sample:
        cases = min(cases, args.sample)
    planned = cases * args.attempts

    result = estimate(storage.connect(args.db), args.model, planned, args.dataset)
    print(f"Planned: {cases} cases x {args.attempts} attempts = {planned} trials of {args.model}")
    if result is None:
        price = PRICES.get(args.model)
        print("No earlier runs of this model to learn from. A cheap way to get one:\n"
              f"  agenteval run --model {args.model} --dataset {args.dataset} --sample 5 --attempts 1")
        if price:
            # Rough: two model calls of ~800 input and ~100 output tokens per trial.
            rough = planned * 2 * (800 * price.input_per_million + 100 * price.output_per_million) / 1e6
            print(f"Rough guess from typical tool-calling trials: ${rough:.4f}")
        return
    basis = "this dataset" if result["same_dataset"] else "other datasets"
    minutes = result["trial_seconds"] / max(1, args.concurrency) / 60
    print(f"From {result['trials_seen']} earlier trials on {basis}:")
    print(f"  Cost:  about ${result['cost']:.4f} (${result['cost_per_trial']:.6f} per trial)")
    print(f"  Time:  about {minutes:.1f} min at concurrency {args.concurrency}, before rate-limit waits")
    print("Cap it with: agenteval run ... --max-cost <USD>")

main = main_prices

__all__ = ["main_estimate", "main_prices", "estimate", "parse_litellm", "update_prices", "load_user_prices"]
