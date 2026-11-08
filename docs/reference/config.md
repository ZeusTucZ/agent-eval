# agenteval.toml

Options for every command can live in `agenteval.toml` in the directory you run from, so a
project's runs need no long command lines. `agenteval init` writes one.

```toml
# Applies to every command that has a --db option.
db = "results/agenteval.db"

[run]
agent = "my_project.agents:answer"
model = "gpt-4o-mini"
price = [0.15, 0.60]          # a list for options that take several values
dataset = "cases.jsonl"
attempts = 3
max_cost = 2.0
redact = true                 # true for flags; false or leaving it out for off
scorer = ["my_scorers:concise", "my_scorers:polite"]   # repeated options: a list

[judge]
model = "llama-3.3-70b-versatile"
provider = "groq"

[report]
format = "markdown"
```

- Each table is named after a command, and each key is one of its options (see the
  [command line reference](cli.md)) with `-` or `_`.
- Anything on the command line wins over the file.
- For `calibrate`, the `[calibrate]` options apply to both `label` and `agree`.

## Environment variables

| Variable | Used by |
|---|---|
| `GROQ_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` | built-in agents, the judge, `generate` |
| `AGENTEVAL_PRICES_FILE` | where `prices --update` saves prices (default `~/.agenteval/prices.json`) |
| `DATABASE_PATH` | the API server's database |
| `API_TOKEN` | the API server's token for changes (`agenteval ui` makes one up) |
| `READ_ONLY` | `1` refuses every change on the API server |
| `DEMO_SEED` | `1` fills an empty database with example runs when the API server starts |
| `CORS_ORIGINS` | sites allowed to call the API server from a browser |
| `REQUESTS_PER_CLIENT_PER_MINUTE` | the API server's per-client limit (120; 0 for none) |
| `AGENTEVAL_STATIC_DIR` | a built dashboard for the API server to serve at `/` |
