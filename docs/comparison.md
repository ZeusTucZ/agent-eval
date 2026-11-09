# agenteval vs other tools

Tools for evaluating LLM applications overlap, but each starts from a different question.
This is how agenteval relates to the better-known ones; check each project's own docs for
current details.

| | agenteval | Ragas | DeepEval | promptfoo |
|---|---|---|---|---|
| Starts from | "is my agent reliable, and is this version better?" | "is this RAG answer good?" | "does my LLM app pass these tests?" | "which prompt or model is best?" |
| Runs your agent | yes, many times per question, with timeouts and concurrency | no: scores records you give it | partly, through its test runner | yes, prompts against providers |
| Traces every step | yes: model and tool calls, tokens, errors | no | through its tracing | no |
| Failures labelled by cause | yes: tool error, timeout, rate limit, quota... | no | no | no |
| Metrics | deterministic first (exact match, tool use, faithfulness, retrieval recall), LLM judge on top | mostly LLM-based | mostly LLM-based | assertions, some LLM-based |
| Judge calibrated against your labels | yes, with Cohen's kappa | no | no | no |
| Says whether a change is real | yes: paired McNemar test and confidence intervals | no | no | side-by-side scores |
| Blocks a PR on a regression | yes: gate and GitHub Action | no | through test failures | through assertions |
| Interface | CLI, Python, dashboard | Python | Python, cloud platform | CLI, web viewer |

## When to use which

- **agenteval** when an agent calls tools, runs in production, and you need to know why it
  fails and whether a change helps, not only how good one answer is.
- **Ragas** when you want its catalogue of LLM-judged RAG metrics. They work together:
  `agenteval export --format ragas` turns a run into Ragas samples, and `agenteval import`
  reads Ragas samples back.
- **DeepEval** when you want evaluations written as pytest-style unit tests with a hosted
  platform around them.
- **promptfoo** when the job is comparing prompts and models across providers in a matrix.

## What agenteval doesn't do

- Hosting: it runs on your machine or your server; there is no managed service.
- Hosted storage: results live in SQLite files; a team shares them by pushing to one
  agenteval server, not through a database server.
- Frameworks with neither OpenTelemetry instrumentation nor an integration here: those are
  traced by wrapping their calls with `llm_call` and `tool`.
