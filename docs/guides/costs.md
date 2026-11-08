# Control cost

## Know the price

Cost is tokens × a price table, in dollars per million tokens.

```bash
agenteval prices              # the table in use, and how old it is
agenteval prices gpt-4o       # search it
agenteval prices --update     # fetch current prices for thousands of models
```

`--update` downloads the list LiteLLM maintains and saves it to
`~/.agenteval/prices.json` (or `AGENTEVAL_PRICES_FILE`), which takes precedence over the
built-in table. For a model that isn't listed, pass `--price INPUT OUTPUT` to `run` or
`judge`, or `price=(input, output)` to `evaluate`; `0 0` if you don't track cost. A run
refuses to start without a price for its `--model` rather than report a misleading $0. A
*different* model your agent calls that has no price is recorded at $0 with a warning.

Each experiment keeps a snapshot of the prices it ran with, so later updates don't rewrite
history.

## Before a run

```bash
agenteval estimate --model gpt-4o-mini --dataset cases.jsonl --attempts 3
```

Estimates cost and time from earlier runs of the same model (same dataset preferred). With
no history, it suggests a 5-case pilot: `agenteval run ... --sample 5 --attempts 1`.

## During a run

```bash
agenteval run ... --max-cost 2.50
```

Stops starting new trials once $2.50 has been spent; trials already running finish and are
saved. Continue later with the same options, `--resume <id>` and a higher `--max-cost`.

## Spend less

- **`--sample N`** runs N cases, spread across categories: the same N every time, so sampled
  runs compare with each other.
- **Fewer attempts** while iterating, more (and odd) when it matters.
- **`agenteval rescore`** re-runs the scorers on saved trials without calling the model.
- **The judge's cache** makes re-judging the same answers free; `--sample` judges a subset.
- **Rate limits**: `--rpm` spaces requests evenly and `--tpm` respects a tokens-per-minute
  cap, so you aren't paying in retries for 429s.
