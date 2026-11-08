# Python API

```python
from agenteval import (
    evaluate, aevaluate, Results,          # run and read results
    llm_call, tool, ToolArgumentError,     # trace your agent's steps
    wrap_openai,                           # trace an OpenAI client automatically
    as_scorer, Case, EvaluationCase,       # your own scorers and cases
)
from agenteval.integrations.langchain import AgentEvalCallback
```

## evaluate

```python
results = evaluate(agent, cases, *, model, price=None, attempts=3, scorers=(), name="python-run",
                   db=None, concurrency=4, timeout=120.0, max_cost=None, sample=None,
                   redact=False, progress=False, config=None)
```

| Argument | Meaning |
|---|---|
| `agent` | a function taking the question (or a parameter named `case`) and returning the answer, sync or async; or an object with `run(case, attempt, tracer)` |
| `cases` | a `.jsonl` path, a built-in dataset name (`"hard"`, `"arithmetic-50"`), or a list of dicts or `Case`s |
| `model` | the model `llm_call` uses by default, for pricing |
| `price` | `(input, output)` dollars per million tokens, if `model` isn't in the price table |
| `scorers` | functions of `(case, trial)`, see [`as_scorer`](#as_scorer) |
| `db` | save to this SQLite file like a CLI run; in memory when `None` |
| `max_cost` | stop starting trials once this much has been spent |
| `sample` | run this many cases, spread across categories |
| `redact` | replace personal data before saving |
| `config` | extra settings recorded in the version, e.g. `{"prompt": PROMPT, "temperature": 0}` |

Works inside a running event loop (Jupyter) by running in its own thread. `aevaluate` is the
`async` form, with the same arguments.

## Results

| Member | What it gives |
|---|---|
| `pass_rate` | exact-match pass rate, or `None` without expected answers |
| `cost` | total cost in dollars |
| `score("tool_selection")` | the mean of one scorer |
| `summary` | everything the report shows, as a dict |
| `report(fmt="text")` | the report: `"text"`, `"json"`, `"markdown"`, `"html"` or `"csv"` |
| `save("report.html")` | write the report; the format follows the extension |
| `failures()` | the trials that didn't pass, with question, expected answer and answer |
| `to_records()`, `to_pandas()` | one row per trial with every score (pandas is optional) |
| `compare(baseline)` | is this run better than `baseline`, as `agenteval compare` |
| `experiment`, `trials`, `scores`, `cases` | the raw objects |

In Jupyter, a `Results` displays as the HTML report.

## Tracing

```python
with llm_call(model=None) as call:     # a model step; model defaults to evaluate's model
    ...
    call.tokens(input_tokens, output_tokens)
    call.failed_because("rate_limit")  # optional, before raising: label the error by cause

result = await tool(name, function, *args, **kwargs)  # a tool step; function sync or async
```

Raise `ToolArgumentError` inside a tool for arguments it can't accept. Outside an evaluation
both helpers simply run what they wrap.

`wrap_openai(client)` returns the client with `chat.completions.create` recorded (sync or
async). `AgentEvalCallback()` goes in a LangChain or LangGraph `config={"callbacks": [...]}`.

## as_scorer

```python
def concise(case, trial):                 # return 0..1, a bool, (value, reason), a Score or None
    return len(trial.output or "") < 200, "under 200 characters"

evaluate(agent, cases, model="m", scorers=[concise])
as_scorer(lambda case, trial: 0.5, name="half")   # name a lambda
```

A trial has `output`, `failure_type`, `attempt`, `case_id` and `spans`; each span has `kind`
(`"llm"` or `"tool"`), `name`, `status`, `input`, `output`, `error_type`, tokens and times.
