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

## An Anthropic client, recorded automatically

```python
from anthropic import AsyncAnthropic
from agenteval import wrap_anthropic

client = wrap_anthropic(AsyncAnthropic())  # or Anthropic()
```

Every `client.messages.create(...)` becomes a model step, its input tokens including any read
from or written to the prompt cache.

## Any framework that emits OpenTelemetry

LlamaIndex, the OpenAI Agents SDK, CrewAI, DSPy, Haystack and smolagents are instrumented by
[OpenInference](https://github.com/Arize-ai/openinference); PydanticAI and Semantic Kernel
emit OpenTelemetry's GenAI conventions. Either way, one line records their steps:

```python
from agenteval.integrations.otel import instrument
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

instrument()                           # agenteval reads the spans while it evaluates
OpenAIAgentsInstrumentor().instrument()  # your framework's instrumentation, as usual
```

Needs `pip install "pyagenteval[otel]"`. Every model span becomes a model step (model,
tokens, duration) and every tool span a tool step (arguments, result, error). Outside an
evaluation nothing changes, and your spans still reach your own exporters. Don't also wrap
the same client with `wrap_openai`, or each call counts twice.

## Replay responses instead of paying again

Both wrappers take a cache:

```python
client = wrap_openai(AsyncOpenAI(), cache=".agenteval-cache.db")
```

A request identical to one already made in the same attempt is answered from the file,
instantly and with no tokens. Re-running an experiment while you tune scorers, a judge or
the report then costs nothing; change the prompt or the model and the requests differ, so
they go to the API. Each attempt has its own entries, so attempts stay independent.

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

## Conversations

A case with `turns` is a conversation (see [datasets](datasets.md#conversations)). Give
your function a parameter named `messages` and it gets the conversation so far, one call per
user turn:

```python
async def answer(messages: list[dict]) -> str:
    response = await client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content
```

`messages` is `[{"role": "user" | "assistant", "content": ...}]`, ending with the newest user
turn. Your replies are added to it, the last reply is the answer that's scored, and the whole
transcript is saved and shown in reports, the dashboard and to the LLM judge.

With a [simulated user](datasets.md#simulated-users), another model writes the user's turns
in reply to yours:

```python
from agenteval.simulate import UserSimulator

evaluate(answer, "conversations.jsonl", model="gpt-4o-mini",
         user_simulator=UserSimulator("claude-haiku-5-5", provider="anthropic"))
```

From the command line: `agenteval run --simulate-users claude-haiku-5-5 --simulator-provider
anthropic`. Any function `(case, messages) -> str | None` works as a simulator too.

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
