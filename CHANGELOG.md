# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org): within 1.x,
command-line options, the Python API and the database format only change in ways that keep
working for existing users; anything else waits for 2.0 and is announced here first.

## 1.0.0

The first release on PyPI, as `agenteval-kit`.

### Use it on your own agent
- `pip install agenteval-kit`; the package and command are `agenteval`.
- `agenteval init` sets up a project: an example agent, cases and `agenteval.toml`.
- Your agent as a function, its steps recorded with `llm_call` and `tool`, or automatically
  with `wrap_openai` (any OpenAI-compatible client) and `AgentEvalCallback` (LangChain, LangGraph).
- Your own datasets as JSONL; `agenteval generate` writes cases from your documents.
- `evaluate()` from Python and notebooks, returning `Results` (report, failures, records,
  pandas, compare).
- Your own scorers (`--scorer`, `scorers=`) and judge rubrics (a TOML file).

### Measure
- Retrieval recall, and a diagnosis that separates search failures from answer failures.
- LLM judge: correctness, groundedness, answer relevance and context relevance rubrics;
  OpenAI, OpenRouter, Groq, Ollama or any OpenAI-compatible endpoint; cached verdicts.
- Calibration against your labels, in the terminal or the dashboard, with Cohen's kappa.
- `agenteval power`: how many cases you need to catch a drop of a given size.

### Run safely
- Every trial is saved as it finishes; `--resume` continues an interrupted run.
- `--max-cost` caps spending; `agenteval estimate` predicts cost from earlier runs;
  `agenteval prices --update` fetches current prices.
- `--sample` for quick, comparable runs; `--redact` keeps personal data out of the database.
- Requests spaced evenly and an optional tokens-per-minute limit.
- Each experiment records its status and environment (git commit, Python, package versions).

### Read and decide
- Reports as text, JSON, Markdown, standalone HTML and CSV.
- `agenteval ui`: a local dashboard to browse experiments, compare two, label answers,
  rename and delete.
- The gate can block Inconclusive results or a pass rate below a minimum, and ships as a
  GitHub Action.
- `agenteval import` scores recorded traces or Ragas samples; `agenteval export` writes
  them back or as Ragas samples.

### Changed
- Renamed from `evalkit` to `agenteval`. Built-in datasets ship inside the package.
- `fastapi`/`uvicorn` and `google-genai` are extras (`[server]`, `[gemini]`); `[all]` for everything.
- Python 3.11 or newer.

### Fixed
- Random HTTP 500s from the API when FastAPI opened and used a database connection on
  different threads.
- `rescore` no longer deletes LLM judge scores.
