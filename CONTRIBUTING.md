# Contributing

Thanks for helping. Bug reports, questions and ideas are as welcome as code: open an issue
or a discussion.

## Set up

```bash
git clone https://github.com/ZeusTucZ/agent-eval && cd agent-eval
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]" langchain-core   # .venv/bin on macOS/Linux
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check .
```

The tests need no network and no API keys, and run in under a minute. The dashboard:

```bash
cd dashboard && npm ci && npm run dev      # with the API: uvicorn agenteval.api:app --port 8000
npm run build                              # type-checks too
```

## What a change needs

- **A test** for new behaviour, and for a bug, one that fails without the fix.
- **Docs** when behaviour changes: the command's `--help`, the page in `docs/`, and
  `python scripts/build_cli_reference.py` if a command's options changed (a test checks it).
- **A line in CHANGELOG.md** under an "Unreleased" heading.
- Generated files are rebuilt, not edited: `scripts/build_hard_dataset.py`,
  `scripts/build_rag_dataset.py`, `scripts/build_cli_reference.py`.

## How the code is laid out

| Module | Does |
|---|---|
| `runner.py`, `execution.py` | run cases × attempts, save each trial, resume, budgets |
| `tracer.py`, `sdk.py`, `integrations/` | record steps: by hand, OpenAI clients, LangChain |
| `scorers.py`, `judge.py`, `calibrate.py` | deterministic scores, the LLM judge, checking it |
| `report.py`, `export.py`, `compare.py`, `stats.py`, `gate.py`, `power.py` | read and decide |
| `storage.py` | SQLite; `_migrate` upgrades older databases in place |
| `cli.py` | the `agenteval` command; each command's `main(argv)` lives in its module |
| `api.py`, `ui.py`, `dashboard/` | the HTTP API and the dashboard |

Principles worth keeping: never show 0% for "not measured" (return `None`); label failures by
cause; keep infrastructure failures apart from the agent's; don't change the version hash of
existing configs or datasets (saved experiments must stay comparable).

## This repository's own regression gate

`.github/workflows/agent-eval.yml` runs the built-in agent on Groq for pull requests labelled
`run-agent-eval` that change the agent, and compares with a baseline stored from `main` by
`agent-eval-baseline.yml`. To use it in a fork:

1. Add a `GROQ_API_KEY` repository secret.
2. Run the `agent-eval-baseline` workflow once.
3. Create the label: `gh label create run-agent-eval`, and add it to a PR when it's ready.

A run uses about 140,000 tokens and Groq's free tier allows 200,000 a day: label one pull
request a day.

## Releasing

Bump `version` in `pyproject.toml` and `action.yml`'s default, move "Unreleased" in
CHANGELOG.md under the new version, merge, then `git tag vX.Y.Z && git push --tags`.
`release.yml` builds the dashboard into the package and publishes to PyPI through trusted
publishing.
