# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org): within 1.x,
command-line options, the Python API and the database format only change in ways that keep
working for existing users; anything else waits for 2.0 and is announced here first.

## 1.2.0

### Claude, everywhere
- `agenteval judge --provider anthropic` uses Claude as the judge (`claude-sonnet-5-5` by
  default), with the verdict's form enforced by structured outputs.
- Prices for current Claude models; the `anthropic` template uses `claude-opus-5-5`.
  New extras: `[anthropic]`, `[otel]`.

### Any framework
- `agenteval.integrations.otel.instrument()` records steps live from OpenTelemetry spans:
  LlamaIndex, the OpenAI Agents SDK, CrewAI, DSPy and others through OpenInference;
  PydanticAI and Semantic Kernel through the GenAI conventions.

### Conversations that react
- Simulated users: a case with `simulated_user` (and `max_turns`) has another model play the
  user, replying to what the agent says. `evaluate(user_simulator=...)`,
  `agenteval run --simulate-users MODEL`.

### Teams and production
- `agenteval push` and `results.push()` send experiments to a shared agenteval server
  (`POST /api/experiments`), so a team's runs share one dashboard.
- `agenteval.monitor.Monitor` samples real requests in production, records them as trials of
  a daily experiment and scores them; judge them later with the reference-free rubrics.
- The dashboard's New run offers your own agent, as configured in `agenteval.toml`, when
  started with `agenteval ui`.

### Docs
- A tutorial from first run to production, and the home page, quickstart and tutorial in
  Spanish.

### Fixes
- `--redact` now also redacts a conversation's transcript.
- CI actions moved to versions that run on Node 24.

## 1.1.0

### How the agent got there
- Three trajectory scorers, on by default: `trajectory_order` (tools called in the order of
  a case's new `expected_trajectory`), `step_efficiency` (calls needed over calls made;
  `max_tool_calls` sets what's needed) and `redundant_calls` (the same call repeated after
  it worked).
- `tool_faithfulness` now also checks imported traces, where the expected tools are unknown.

### More agents, less code
- `wrap_anthropic(client)` records Anthropic `messages.create` calls, like `wrap_openai`.
- Both wrappers take `cache="file.db"`: identical requests are replayed from the file,
  so re-running an unchanged agent costs nothing.
- `agenteval import` reads OpenTelemetry traces (OTLP JSON) using the OpenInference or GenAI
  conventions; `--format` picks the format when detection isn't enough.
- `agenteval init --template openai|anthropic|langgraph|rag` starts from a working agent of
  that kind.

### Conversations
- A case with `turns` is a multi-turn conversation. Agents with a `messages` parameter get
  the conversation so far at each turn; the transcript is saved and shown in reports, the
  dashboard and to the judge. `must_not_contain` is checked in every reply.

### Tests, data and the dashboard
- `results.check(min_pass_rate=..., min_scores=..., max_cost=..., max_failures=...)` fails a
  pytest test with what fell short and the first failing questions.
- The built-in `extended` dataset: 200 cases in 11 categories, enough to catch a drop of
  about 10 points.
- The dashboard shows each run's pass rate over time next to the runs before it, with a
  link to compare with the previous one. `DEMO_SEED=1` starts an empty server with example
  runs.
- Experiments keep a copy of their cases, so reports, comparisons, re-scoring and the
  dashboard work for inline datasets and moved files. Older experiments are unaffected.
- The package is type-checked with mypy in CI.

## 1.0.1

The same code as 1.0.0, published under a clearer name: `pip install pyagenteval`.
The import and the command are still `agenteval`. 1.0.0 was published as
`zeustucz-agenteval`, which is no longer updated: switch with
`pip uninstall zeustucz-agenteval && pip install pyagenteval`.

## 1.0.0

The first release on PyPI, as `zeustucz-agenteval`.

### Use it on your own agent
- `pip install zeustucz-agenteval`; the package and command are `agenteval`.
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
