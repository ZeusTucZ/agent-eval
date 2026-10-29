from evalkit.models import Price, Span, Trial

# Made-up prices for the mock models. Real ones come from each provider's pricing page.
PRICES: dict[str, Price] = {
    "mock-llm": Price(input_per_million=3.00, output_per_million=15.00),
    "mock-llm-small": Price(input_per_million=0.25, output_per_million=1.25),
}

def span_cost(span: Span, prices: dict[str, Price]) -> float:
    if span.kind != "llm":
        return 0.0

    price = prices.get(span.name)
    if price is None:
        raise KeyError(f"no price for model {span.name!r}")

    return (
        span.input_tokens * price.input_per_million
        + span.output_tokens * price.output_per_million
    ) / 1_000_000

def trial_cost(trial: Trial, prices: dict[str, Price]) -> float:
    return sum(span_cost(span, prices) for span in trial.spans)

def total_cost(trials: list[Trial], prices: dict[str, Price]) -> float:
    return sum(trial_cost(trial, prices) for trial in trials)
