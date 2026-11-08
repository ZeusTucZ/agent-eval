# agent-eval

[![tests](https://github.com/ZeusTucZ/agent-eval/actions/workflows/tests.yml/badge.svg)](https://github.com/ZeusTucZ/agent-eval/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-zeustucz.github.io%2Fagent--eval-4f46e5)](https://zeustucz.github.io/agent-eval/)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Run your LLM agent many times, trace every step, and find out why it fails.**

An agent that answers well in a demo can still be slow, expensive, flaky, or right for the
wrong reason: it answered from memory instead of using its tools, or a rate limit made half
its runs fail. A single accuracy number hides all of that. agenteval runs your agent over a
set of questions, several times each, records every model call and tool call, and tells you
**how often it's right, why it fails, how it uses its tools, what it costs**, and **whether a
change really made it better**.

```bash
pip install "agenteval-kit[server]"
agenteval init      # an example agent, a few questions, a config
agenteval run       # no API key needed for the example
agenteval ui        # see it in your browser
```

```
Experiment 5: groq-first   config v=c98ba11d65cb   48 trials
Model:            openai/gpt-oss-20b   dataset: mixed
Pass rate:        100.0%
Latency p50/p95:  1042ms / 57768ms total, 987ms / 1572ms excluding waits
Tool calls:
  right tool        100.0%  (48 trials)
  valid arguments   100.0%  (48 trials)
  ran without error 93.8%  (48 trials)
Retries:          1 (rate_limit 1)
Cost:             $0.0027 total, $0.000057 per success
Failures:         none
```

*A real run against Groq. More in [Results from real runs](https://zeustucz.github.io/agent-eval/results/).*

## Your agent, unchanged

Write it as a function from question to answer, and wrap its steps, or let an integration
do it:

```python
from agenteval import evaluate, llm_call, tool, wrap_openai

client = wrap_openai(AsyncOpenAI())              # every chat completion becomes a step

async def answer(question: str) -> str:
    weather = await tool("get_weather", get_weather, city="Lisbon")   # a tool step
    response = await client.chat.completions.create(model="gpt-4o-mini", messages=[...])
    return response.choices[0].message.content

results = evaluate(answer, "cases.jsonl", model="gpt-4o-mini", price=(0.15, 0.60))
print(results)               # the report
results.failures()           # the trials that didn't pass, with question and answer
results.compare(baseline)    # is this version really better?
```

LangChain and LangGraph agents need one callback. The same works from the command line:
`agenteval run --agent my_agents:answer --model gpt-4o-mini --dataset cases.jsonl`.

## What makes it different

- **Failures labelled by cause.** A tool error, a timeout, a rate limit, a daily quota, a
  truncated answer: each gets its own label, so "68% correct" and "100% correct, 32% blocked
  by rate limits" don't look the same.
- **Tool use checked step by step**: the right tool, valid arguments, ran without error, and
  whether the answer is backed by what the tools returned.
- **RAG diagnosed**: did the search miss the document, or did the model have it and still
  answer wrong?
- **Changes judged with statistics.** A paired McNemar test and bootstrap intervals say
  whether B is really better than A, and `agenteval power` says how many questions you need
  to tell. A gate and a GitHub Action block pull requests that make the agent worse.
- **An LLM judge you can check.** Label 50 answers yourself and see the judge's agreement
  with you (Cohen's kappa) before trusting it.
- **Built for real APIs**: retries with backoff, request and token rate limits, a circuit
  breaker for daily quotas, cost caps, and runs that resume after an interruption.

## What's in the box

| | |
|---|---|
| **Run** | your agent, or built-in Groq, Gemini and fault-injecting mock agents; `--sample`, `--max-cost`, `--resume`, `--redact` |
| **Score** | exact match, tool selection / arguments / execution, tool faithfulness, retrieval recall, forbidden text, your own scorers, and an LLM judge (correctness, groundedness, answer and context relevance, your own rubrics) |
| **Datasets** | your JSONL file, cases generated from your documents (`agenteval generate`), or built-in `mixed`, `hard` (55 cases, 10 categories) and `rag` (45 cases over a document corpus) |
| **Read** | reports as text, JSON, Markdown, HTML and CSV; a local dashboard to browse, compare and label |
| **Decide** | `compare`, `power`, `gate`, and a GitHub Action |
| **Bring data** | import recorded traces or Ragas samples; export to Ragas |

## Documentation

**[zeustucz.github.io/agent-eval](https://zeustucz.github.io/agent-eval/)**: quickstart,
guides to connecting your agent, writing datasets, the metrics, the judge, comparing versions,
cost and privacy, and the full command-line and Python reference. The same pages are in
[`docs/`](docs/).

## Live demo

The API runs at https://agent-eval-co10.onrender.com (Render's free tier: the first request
after a while can take about a minute to wake it up), with the dashboard deployed alongside it
as a static site. Its database lives on the service's ephemeral disk, so it empties on each
redeploy.

## Developing

```bash
git clone https://github.com/ZeusTucZ/agent-eval && cd agent-eval
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"   # .venv/bin on macOS/Linux
.venv/Scripts/python -m pytest
```

The harness is tested against itself: a mock agent injects tool crashes, hallucinations, rate
limits and hangs at known rates, and the tests check the measured rates match. Real SDKs are
tested through fake clients built from their own response and error types, and timing through
a fake clock. The dashboard is in [`dashboard/`](dashboard/) (`npm ci && npm run dev`, with
the API on port 8000). See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
