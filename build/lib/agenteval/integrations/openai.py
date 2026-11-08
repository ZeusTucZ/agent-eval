"""Record every call an OpenAI client makes, without wrapping each one in llm_call.

    from openai import AsyncOpenAI
    from agenteval import wrap_openai

    client = wrap_openai(AsyncOpenAI())

    async def my_agent(question: str) -> str:
        response = await client.chat.completions.create(model="gpt-4o-mini", messages=[...])
        return response.choices[0].message.content

Each `chat.completions.create` (sync or async client, and any OpenAI-compatible provider:
Groq, OpenRouter, Ollama, vLLM...) becomes a model step named after its model, with the
tokens the response reports, and its error if it fails. Everything else on the client
passes through untouched.

`wrap_openai(client, cache="responses.db")` also replays responses to identical requests
from earlier runs, so re-running an experiment while tuning scorers costs nothing; see
agenteval.integrations._recording.ResponseCache.
"""
from pathlib import Path
from typing import Any

from agenteval.integrations._recording import Namespace, RecordedCreate, ResponseCache

def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0

def wrap_openai(client: Any, cache: str | Path | ResponseCache | None = None) -> Any:
    """Return `client` with its chat completions recorded as steps of the current trial."""
    if cache is not None and not isinstance(cache, ResponseCache):
        cache = ResponseCache(cache)
    completions = RecordedCreate(client.chat.completions, _usage, cache)
    return Namespace(client, chat=Namespace(client.chat, completions=completions))
