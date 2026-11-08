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
passes through untouched. Streaming calls are recorded too, with their tokens when the
stream reports usage (stream_options={"include_usage": True}).
"""
import inspect
from typing import Any

from agenteval.sdk import llm_call

class _Completions:
    def __init__(self, completions: Any):
        self._completions = completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        # The step opens before the request starts. Whether the client is async can't be told
        # from the method (the SDK's async methods are wrapped by a sync decorator), only from
        # what it returns: an awaitable keeps the step open until it completes.
        step = llm_call(kwargs.get("model"))
        call = step.__enter__()
        try:
            result = self._completions.create(*args, **kwargs)
        except BaseException as error:
            step.__exit__(type(error), error, error.__traceback__)
            raise
        if inspect.isawaitable(result):
            return self._finish(step, call, result)
        _record_usage(call, result)
        step.__exit__(None, None, None)
        return result

    @staticmethod
    async def _finish(step: Any, call: Any, pending: Any) -> Any:
        try:
            response = await pending
        except BaseException as error:
            step.__exit__(type(error), error, error.__traceback__)
            raise
        _record_usage(call, response)
        step.__exit__(None, None, None)
        return response

def _record_usage(call: Any, response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is not None:
        call.tokens(getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))

class _Chat:
    def __init__(self, chat: Any):
        self._chat = chat
        self.completions = _Completions(chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)

class _Client:
    def __init__(self, client: Any):
        self._client = client
        self.chat = _Chat(client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

def wrap_openai(client: Any) -> Any:
    """Return `client` with its chat completions recorded as steps of the current trial."""
    return _Client(client)
