from evalkit.models import Price, Span, Trial

PRICES: dict[str, Price] = {
    # Made-up prices for the mock models.
    "mock-llm": Price(input_per_million=3.00, output_per_million=15.00),
    "mock-llm-small": Price(input_per_million=0.25, output_per_million=1.25),
    # Paid-tier Gemini prices as listed in October 2026; on the free tier you pay nothing,
    # so these show what the run *would* cost. Check ai.google.dev/gemini-api/docs/pricing
    # before relying on them, and add an entry here to use another model.
    "gemini-2.5-flash": Price(input_per_million=0.30, output_per_million=2.50),
    "gemini-2.5-flash-lite": Price(input_per_million=0.10, output_per_million=0.40),
    # Groq paid-tier prices as listed by third-party trackers in October 2026; the free plan
    # costs nothing. Check groq.com/pricing, and the console's model list for current names.
    "openai/gpt-oss-20b": Price(input_per_million=0.075, output_per_million=0.30),
    "openai/gpt-oss-120b": Price(input_per_million=0.15, output_per_million=0.60),
    "llama-3.3-70b-versatile": Price(input_per_million=0.59, output_per_million=0.79),
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
