"""Record every call an Anthropic client makes, as wrap_openai does for OpenAI.

    from anthropic import AsyncAnthropic
    from agenteval import wrap_anthropic

    client = wrap_anthropic(AsyncAnthropic())

    async def my_agent(question: str) -> str:
        message = await client.messages.create(model="claude-opus-5-5", max_tokens=1024,
                                               messages=[{"role": "user", "content": question}])
        return message.content[0].text

Each `messages.create` (sync or async client) becomes a model step named after its model,
with the input and output tokens the response reports. Tools your agent runs on the model's
tool_use blocks still go through agenteval's `tool(...)`. `cache=` works as for wrap_openai.
"""
from pathlib import Path
from typing import Any

from agenteval.integrations._recording import Namespace, RecordedCreate, ResponseCache

def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    # Cached prompt tokens are billed too; count them as input.
    cached = (getattr(usage, "cache_read_input_tokens", 0) or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    return (getattr(usage, "input_tokens", 0) or 0) + cached, getattr(usage, "output_tokens", 0) or 0

def wrap_anthropic(client: Any, cache: str | Path | ResponseCache | None = None) -> Any:
    """Return `client` with its messages.create calls recorded as steps of the current trial."""
    if cache is not None and not isinstance(cache, ResponseCache):
        cache = ResponseCache(cache)
    return Namespace(client, messages=RecordedCreate(client.messages, _usage, cache))
