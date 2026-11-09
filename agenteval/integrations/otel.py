"""Record an agent's steps from the OpenTelemetry spans its framework already emits.

    from agenteval.integrations.otel import instrument

    instrument()  # once, before your agent runs

Frameworks instrumented with OpenInference (LlamaIndex, the OpenAI Agents SDK, CrewAI, DSPy,
Haystack, smolagents, Bedrock, ...) or emitting OpenTelemetry's GenAI conventions (PydanticAI,
Semantic Kernel, ...) describe each model call and tool call as a span. While agenteval runs
your agent, this span processor turns those spans into the trial's steps, as `llm_call` and
`tool` would: model, tokens, duration, tool arguments and results, errors. Outside an
evaluation it does nothing, and your spans still go wherever your exporters send them.

Don't combine it with wrap_openai or wrap_anthropic on the same calls, or each call counts twice.
Needs opentelemetry-sdk.
"""
import json
from time import perf_counter_ns, time_ns
from typing import Any

from agenteval.models import Span
from agenteval.otel import step_from_span
from agenteval.sdk import MAX_RECORDED_TOOL_OUTPUT, _active, _Trial
from agenteval.tracer import clip

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
    from opentelemetry.trace import StatusCode
except ImportError as error:  # pragma: no cover - only without the optional dependency
    raise ImportError("agenteval.integrations.otel needs opentelemetry-sdk: pip install opentelemetry-sdk") from error

def _text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    return clip(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str), limit)

class AgentEvalSpanProcessor(SpanProcessor):
    """Adds every model and tool span that starts inside an evaluated trial to that trial."""

    def __init__(self) -> None:
        # Which trial each open span belongs to: decided where the span starts, which is in the
        # trial's own context, even if it ends somewhere else (a callback, another thread).
        self._open: dict[tuple[int, int], _Trial] = {}

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        trial = _active.get()
        if trial is not None:
            context = span.get_span_context()
            self._open[(context.trace_id, context.span_id)] = trial

    def on_end(self, span: ReadableSpan) -> None:
        context = span.get_span_context()
        if context is None:
            return
        trial = self._open.pop((context.trace_id, context.span_id), None)
        if trial is None:
            return
        failed = span.status.status_code == StatusCode.ERROR
        error = (span.status.description or "error") if failed else None
        duration_ns = max(0, (span.end_time or 0) - (span.start_time or 0))
        step = step_from_span(dict(span.attributes or {}), span.name, duration_ns / 1e6, error)
        if step is None:
            return
        # agenteval's own steps are timed with a monotonic clock; place this one on it, so
        # steps keep the order they ran in.
        end = perf_counter_ns() - max(0, time_ns() - (span.end_time or time_ns()))
        recorded = Span(trial_id=trial.tracer.trial_id, kind=step["kind"], name=str(step["name"]),
                        start=end - duration_ns, end=end, status="error" if failed else "success")
        if failed:
            recorded.error_type = "otel_error"
            recorded.error_message = clip(str(error))
        if step["kind"] == "llm":
            recorded.input_tokens, recorded.output_tokens = step["input_tokens"], step["output_tokens"]
        else:
            recorded.input = _text(step["input"], MAX_RECORDED_TOOL_OUTPUT)
            recorded.output = _text(step["output"], MAX_RECORDED_TOOL_OUTPUT)
        trial.tracer.spans.append(recorded)

    def shutdown(self) -> None:
        self._open.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

def instrument(tracer_provider: Any = None) -> AgentEvalSpanProcessor:
    """Add the processor to `tracer_provider`, or to the global one; if no SDK provider is set
    yet, one is created and made global so instrumentations have somewhere to send spans."""
    provider = tracer_provider or trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    processor = AgentEvalSpanProcessor()
    provider.add_span_processor(processor)
    return processor
