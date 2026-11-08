# Quickstart

## 1. Install

```bash
pip install "agenteval-kit[server]"
```

Python 3.11 or newer. `[server]` adds the dashboard; leave it out for the command line and
Python API only. Other extras: `[gemini]` for Google's SDK, `[rag]` for the RAG dataset's
search index, `[all]` for everything.

## 2. Start a project

```bash
mkdir my-eval && cd my-eval
agenteval init
```

This writes three files:

| File | What it is |
|---|---|
| `my_agent.py` | an example agent: a function that takes a question and returns an answer |
| `cases.jsonl` | four questions with their expected answers |
| `agenteval.toml` | which agent, model, price and dataset `agenteval run` uses |

## 3. Run it

```bash
agenteval run
agenteval report
```

The example needs no API key: its "model" is a few lines of rules. The report shows the pass
rate, every check, latency, cost and failures.

## 4. Put in your own agent

Replace the body of `answer` in `my_agent.py` with your agent. Wrap model calls in
`llm_call` and tool calls in `tool`, so each becomes a step agenteval can time, price and score:

```python
from openai import AsyncOpenAI
from agenteval import llm_call, tool

client = AsyncOpenAI()

async def answer(question: str) -> str:
    with llm_call("gpt-4o-mini") as call:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": question}])
        call.tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
    return response.choices[0].message.content
```

Using an OpenAI client? `wrap_openai(client)` records every call for you; LangChain and
LangGraph have a callback. See [Connect your agent](guides/your-agent.md).

Then set the model and its price in `agenteval.toml`:

```toml
[run]
agent = "my_agent:answer"
model = "gpt-4o-mini"
price = [0.15, 0.60]   # dollars per million input and output tokens
```

`agenteval prices --update` fetches current prices for thousands of models, after which
`price` can usually be left out.

## 5. Write your questions

One JSON object per line in `cases.jsonl`. Only `input` and `expected_output` are required:

```json
{"input": "What is our refund window?", "expected_output": "30 days", "category": "policy"}
```

No questions yet? `agenteval generate docs/` writes them from your documents. See
[Write a dataset](guides/datasets.md).

## 6. Look, change, compare

```bash
agenteval ui                 # browse runs, open failures, compare two runs
agenteval compare 1 2        # is run 2 really better than run 1?
```

From Python or a notebook instead:

```python
from agenteval import evaluate
results = evaluate(answer, "cases.jsonl", model="gpt-4o-mini", price=(0.15, 0.60))
results            # the report (an HTML table in Jupyter)
results.failures() # the trials that didn't pass, with question and answer
```
