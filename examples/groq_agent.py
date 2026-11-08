"""A real LLM agent wired into agent-eval: a template to copy for your own.

A tool-calling loop on any OpenAI-compatible API (Groq here). The only agent-eval parts
are `llm_call` around each model request and `tool` around each tool: the rest is
ordinary agent code you'd write anyway.

    set GROQ_API_KEY=...   (or export on macOS/Linux)
    python -m agenteval.run --agent examples.groq_agent:answer --model openai/gpt-oss-20b \
        --dataset examples/quickstart_cases.jsonl --attempts 3 --name groq-template

Retries, rate limits and timeouts are your agent's own business here; agent-eval records
what happened, labels each failure by the step that raised, and times every step.
"""
import json
import os

import openai

from agenteval import llm_call, tool
from examples.quickstart_agent import calculator, population

MODEL = "openai/gpt-oss-20b"
MAX_STEPS = 5

TOOLS = {"calculator": calculator, "population": population}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Exact arithmetic on two numbers.",
        "parameters": {"type": "object", "required": ["a", "operator", "b"], "properties": {
            "a": {"type": "number"}, "operator": {"type": "string", "enum": ["+", "-", "*", "/"]},
            "b": {"type": "number"},
        }},
    }},
    {"type": "function", "function": {
        "name": "population",
        "description": "Look up a city's population in the reference database.",
        "parameters": {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}},
    }},
]
SYSTEM_PROMPT = (
    "Answer using the tools. Reply with only the final answer, no units or separators. "
    "If the tools can't answer, reply exactly: unknown"
)

_client = None

def client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
    return _client

async def answer(question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    for _ in range(MAX_STEPS):
        with llm_call(MODEL) as call:
            response = await client().chat.completions.create(
                model=MODEL, messages=messages, tools=TOOL_SCHEMAS, temperature=0.0,
            )
            call.tokens(response.usage.prompt_tokens, response.usage.completion_tokens)

        message = response.choices[0].message
        if not message.tool_calls:
            return (message.content or "").strip()

        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in message.tool_calls
        ]})
        for call_request in message.tool_calls:
            try:
                result = await tool(call_request.function.name, TOOLS[call_request.function.name],
                                    **json.loads(call_request.function.arguments))
                content = json.dumps({"result": result})
            except Exception as error:  # noqa: BLE001 - the model sees the error and can recover
                content = json.dumps({"error": f"{type(error).__name__}: {error}"})
            messages.append({"role": "tool", "tool_call_id": call_request.id, "content": content})

    return ""  # out of steps: agent-eval records it as an empty answer
