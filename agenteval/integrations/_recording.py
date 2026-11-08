"""Shared by the SDK wrappers: record a create() call as a model step, sync or async, and
optionally answer it from a cache of earlier responses."""
import hashlib
import importlib
import inspect
import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agenteval.sdk import _active, llm_call

Usage = Callable[[Any], tuple[int, int]]  # response -> (input tokens, output tokens)

class ResponseCache:
    """Responses by a hash of the request and the attempt, in a SQLite file.

    Re-running an experiment with the same agent then replays its answers instead of paying
    for them again, which is what you want while tuning scorers or a judge. The attempt is
    part of the key, so the attempts of one run still get independent answers. A replayed
    call is recorded with no tokens (it cost nothing) and is near-instant.
    """

    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, kind TEXT, body TEXT)")
        self.lock = threading.Lock()
        self.hits = 0

    @staticmethod
    def key(kwargs: dict[str, Any]) -> str:
        trial = _active.get()
        attempt = trial.attempt if trial is not None else 0
        request = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(f"{attempt}|{request}".encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        with self.lock:
            row = self.conn.execute("SELECT kind, body FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        module, _, name = row[0].rpartition(".")
        try:
            cls = getattr(importlib.import_module(module), name)
            response = cls.model_validate_json(row[1])
        except (ImportError, AttributeError, ValueError):
            return None  # an SDK upgrade changed the type: ask again
        self.hits += 1
        return response

    def put(self, key: str, response: Any) -> None:
        dump = getattr(response, "model_dump_json", None)
        if dump is None:
            return  # not a pydantic response (e.g. a stream): can't be replayed
        kind = f"{type(response).__module__}.{type(response).__qualname__}"
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (key, kind, dump()))

class RecordedCreate:
    """Wraps an SDK resource (chat.completions, messages) so create() becomes a model step."""

    def __init__(self, resource: Any, usage: Usage, cache: ResponseCache | None = None):
        self._resource = resource
        self._usage = usage
        self._cache = cache

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resource, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        key = None
        if self._cache is not None and not args and not kwargs.get("stream"):
            key = self._cache.key(kwargs)
            cached = self._cache.get(key)
            if cached is not None:
                with llm_call(kwargs.get("model")):
                    pass  # recorded as a step, with no tokens: it cost nothing
                return self._replay(cached)

        # The step opens before the request starts. Whether the client is async can't be told
        # from the method (SDKs wrap async methods in sync decorators), only from what it
        # returns: an awaitable keeps the step open until it completes.
        step = llm_call(kwargs.get("model"))
        call = step.__enter__()
        try:
            result = self._resource.create(*args, **kwargs)
        except BaseException as error:
            step.__exit__(type(error), error, error.__traceback__)
            raise
        if inspect.isawaitable(result):
            return self._finish(step, call, result, key)
        self._record(call, result, key)
        step.__exit__(None, None, None)
        return result

    def _replay(self, response: Any) -> Any:
        # Answer the way the client would: an async client's caller will await the result.
        if inspect.iscoroutinefunction(getattr(self._resource, "create", None)) or self._is_async_client():
            async def ready() -> Any:
                return response
            return ready()
        return response

    def _is_async_client(self) -> bool:
        client = getattr(self._resource, "_client", None)
        return type(client).__name__.startswith("Async")

    def _record(self, call: Any, response: Any, key: str | None) -> None:
        input_tokens, output_tokens = self._usage(response)
        call.tokens(input_tokens, output_tokens)
        if key is not None and self._cache is not None:
            self._cache.put(key, response)

    async def _finish(self, step: Any, call: Any, pending: Any, key: str | None) -> Any:
        try:
            response = await pending
        except BaseException as error:
            step.__exit__(type(error), error, error.__traceback__)
            raise
        self._record(call, response, key)
        step.__exit__(None, None, None)
        return response

class Namespace:
    """A pass-through stand-in for an SDK object, with some attributes replaced."""

    def __init__(self, wrapped: Any, **replacements: Any):
        self._wrapped = wrapped
        self.__dict__.update(replacements)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)
