"""Record a LangChain or LangGraph agent's model and tool calls automatically.

    from agenteval.integrations.langchain import AgentEvalCallback

    async def my_agent(question: str) -> str:
        result = await graph.ainvoke({"messages": [("user", question)]},
                                     config={"callbacks": [AgentEvalCallback()]})
        return result["messages"][-1].content

Every chat model or LLM call becomes a model step (named after the model, with the tokens
LangChain reports) and every tool call a tool step (with its input, output or error), so
all of agenteval's scorers, reports and comparisons work on the agent unchanged.
Needs langchain-core; outside an evaluation the callback does nothing.
"""
import json
from time import perf_counter_ns
from typing import Any
from uuid import UUID

from agenteval.models import Span
from agenteval.sdk import _active
from agenteval.sdk import MAX_RECORDED_TOOL_OUTPUT
from agenteval.tracer import clip

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as error:  # pragma: no cover - only without the optional dependency
    raise ImportError("AgentEvalCallback needs langchain-core: pip install langchain-core") from error

def _model_name(serialized: dict | None, kwargs: dict) -> str:
    params = kwargs.get("invocation_params") or {}
    metadata = kwargs.get("metadata") or {}
    for value in (params.get("model"), params.get("model_name"), metadata.get("ls_model_name")):
        if value:
            return str(value)
    serialized = serialized or {}
    return str((serialized.get("kwargs") or {}).get("model") or serialized.get("name") or "unknown-model")

def _tokens(response: Any) -> tuple[int, int]:
    """(input, output) tokens, from llm_output (OpenAI-style) or the message's usage_metadata."""
    usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if usage:
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    total_in = total_out = 0
    for generations in getattr(response, "generations", []) or []:
        for generation in generations:
            metadata = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
            total_in += int(metadata.get("input_tokens", 0))
            total_out += int(metadata.get("output_tokens", 0))
    return total_in, total_out

class AgentEvalCallback(BaseCallbackHandler):
    """Turns LangChain callbacks into agenteval steps for the trial being run."""

    raise_error = False  # a tracing problem must never break the agent

    def __init__(self) -> None:
        super().__init__()
        self._open: dict[UUID, tuple[Any, Span]] = {}  # run id -> (tracer, span in progress)

    def _start(self, run_id: UUID, kind: str, name: str, input_text: str | None = None) -> None:
        trial = _active.get()
        if trial is None:
            return
        now = perf_counter_ns()
        span = Span(trial_id=trial.tracer.trial_id, kind=kind, name=name, start=now, end=now, status="running",
                    input=clip(input_text) if input_text else None)
        self._open[run_id] = (trial.tracer, span)

    def _finish(self, run_id: UUID, error: BaseException | None = None, **updates: Any) -> None:
        opened = self._open.pop(run_id, None)
        if opened is None:
            return
        tracer, span = opened
        span.end = perf_counter_ns()
        span.status = "error" if error else "success"
        if error is not None:
            span.error_type = type(error).__name__
            span.error_message = clip(str(error))
        for field, value in updates.items():
            setattr(span, field, value)
        tracer.spans.append(span)

    # --- models ---

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: UUID, **kwargs: Any) -> None:
        self._start(run_id, "llm", _model_name(serialized, kwargs))

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: UUID, **kwargs: Any) -> None:
        self._start(run_id, "llm", _model_name(serialized, kwargs))

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        input_tokens, output_tokens = _tokens(response)
        self._finish(run_id, input_tokens=input_tokens, output_tokens=output_tokens)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, error)

    # --- tools ---

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: UUID, inputs: dict | None = None,
                      **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        text = json.dumps(inputs, ensure_ascii=False, default=str) if inputs else input_str
        self._start(run_id, "tool", str(name), text)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        content = getattr(output, "content", output)
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        self._finish(run_id, output=clip(text, MAX_RECORDED_TOOL_OUTPUT))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, error)
