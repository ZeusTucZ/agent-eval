# Connect your agent

Your agent stays your code. agenteval needs two things from it: a way to call it with a
question, and to know which model calls and tool calls it made. Pick whichever fits.

## A function, with steps wrapped by hand

```python
from agenteval import llm_call, tool, ToolArgumentError

async def answer(question: str) -> str:
    with llm_call("gpt-4o-mini") as call:
        response = await client.chat.completions.create(...)
        call.tokens(response.usage.prompt_tokens, response.usage.completion_tokens)

    weather = await tool("get_weather", get_weather, city="Lisbon")
    ...
    return final_answer
```

- The function may be **sync or async**. Sync agents run in a thread, so they don't block
  each other; a timeout can't stop a hung thread, though, so prefer async if steps can hang.
- A parameter named `case` receives the whole case (with its category and expected tools)
  instead of the question.
- `llm_call(model)` records one model request. Report its tokens with `call.tokens(...)` so
  cost can be counted. Before re-raising an error, `call.failed_because("rate_limit")` labels
  it by cause rather than exception class.
- `tool(name, function, *args, **kwargs)` calls a sync or async tool and records its
  arguments and result, or its error. Raise `ToolArgumentError` for arguments the tool can't
  accept: that's scored as the agent's bad call, not the tool's failure.
- Outside an evaluation both helpers just run what they wrap: the agent works normally.

## An OpenAI client, recorded automatically

```python
from openai import AsyncOpenAI
from agenteval import wrap_openai

client = wrap_openai(AsyncOpenAI())  # or OpenAI(); any compatible base_url works
```

Every `client.chat.completions.create(...)` becomes a model step named after its model, with
the tokens the response reports. Works with Groq, OpenRouter, Ollama, vLLM, Azure and
anything else behind the OpenAI SDK. Tools still go through `tool(...)`.

## LangChain and LangGraph

```python
from agenteval.integrations.langchain import AgentEvalCallback

async def answer(question: str) -> str:
    result = await graph.ainvoke({"messages": [("user", question)]},
                                 config={"callbacks": [AgentEvalCallback()]})
    return result["messages"][-1].content
```

Needs `langchain-core`. Every chat model call becomes a model step (with the tokens
LangChain reports) and every tool call a tool step.

## An object with a run method

The built-in agents implement `async def run(self, case, attempt, tracer) -> str`. An
object with that method can be passed directly, for full control over the trace.

## Running it

=== "Command line"

    ```bash
    agenteval run --agent my_project.agents:answer --model gpt-4o-mini --price 0.15 0.60 \
        --dataset cases.jsonl
    ```

=== "Python"

    ```python
    from agenteval import evaluate
    results = evaluate(answer, "cases.jsonl", model="gpt-4o-mini", price=(0.15, 0.60))
    ```

`--model` names the model `llm_call` uses when it isn't given one, and prices it. A hash of
the agent's source file becomes part of its version, so two versions of your code can be
compared even when the model and prompt are the same.

## The built-in agents

`--agent groq`, `--agent gemini` and `--agent mock` (fault-injecting, for testing the
harness) are tool-calling agents you can run on the built-in datasets with no code at all.
`examples/groq_agent.py` in the repository is a real tool-calling agent to copy from.
