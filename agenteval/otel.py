"""Read OpenTelemetry traces (OTLP JSON) into agenteval's import records.

Most agent frameworks can export OpenTelemetry traces, through OpenInference
instrumentations (LlamaIndex, OpenAI Agents SDK, CrewAI, DSPy, Haystack, smolagents...) or
the GenAI semantic conventions. Export them as OTLP JSON (the OpenTelemetry Collector's file
exporter, or any exporter writing the OTLP JSON format) and `agenteval import` reads them:
one trace becomes one trial.

- A model span is recognised by `openinference.span.kind = LLM` or a `gen_ai.*` operation of
  chat or completion; its model and tokens come from `llm.model_name` / `gen_ai.request.model`
  and `llm.token_count.*` / `gen_ai.usage.*`.
- A tool span by `openinference.span.kind = TOOL` or `gen_ai.operation.name = execute_tool`;
  its name, arguments and result from `tool.name`, `input.value` and `output.value` (or the
  `gen_ai.tool.*` equivalents).
- The question and answer come from the root span's `input.value` and `output.value`, or
  `agenteval.input` / `agenteval.output` if you set them yourself.

Spans of other kinds (chains, agents, retrievers) are walked through: their children count.
"""
import json
from pathlib import Path
from typing import Any

LLM_OPERATIONS = {"chat", "text_completion", "generate_content", "completion"}

def _value(attribute_value: dict) -> Any:
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in attribute_value:
            value = attribute_value[key]
            return int(value) if key == "intValue" else value
    if "arrayValue" in attribute_value:
        return [_value(v) for v in attribute_value["arrayValue"].get("values", [])]
    return None

def _attributes(span: dict) -> dict[str, Any]:
    return {a["key"]: _value(a.get("value", {})) for a in span.get("attributes", [])}

def _first(attributes: dict, *keys: str) -> Any:
    for key in keys:
        if attributes.get(key) not in (None, ""):
            return attributes[key]
    return None

def _text(value: Any) -> str | None:
    """OpenInference stores inputs as text or JSON; find the human-readable part."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
        value = parsed
    if isinstance(value, dict):
        for key in ("input", "question", "query", "output", "answer", "content", "text", "result"):
            if isinstance(value.get(key), str):
                return value[key]
        messages = value.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict) and isinstance(last.get("content"), str):
                return last["content"]
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

def _kind(attributes: dict) -> str | None:
    kind = str(attributes.get("openinference.span.kind", "")).upper()
    operation = str(attributes.get("gen_ai.operation.name", "")).lower()
    if kind == "LLM" or operation in LLM_OPERATIONS:
        return "llm"
    if kind == "TOOL" or operation == "execute_tool":
        return "tool"
    return None

def read_spans(path: Path) -> list[dict]:
    """Every span in an OTLP JSON file: one export object, or one per line."""
    text = path.read_text(encoding="utf-8").strip()
    documents = []
    try:
        documents.append(json.loads(text))
    except json.JSONDecodeError:
        documents = [json.loads(line) for line in text.splitlines() if line.strip()]
    spans = []
    for document in documents:
        for resource in document.get("resourceSpans", []):
            for scope in resource.get("scopeSpans", resource.get("instrumentationLibrarySpans", [])):
                spans.extend(scope.get("spans", []))
    return spans

def is_otlp(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as file:
            head = file.read(4096)
    except OSError:
        return False
    return '"resourceSpans"' in head

def records_from_otlp(path: Path) -> list[dict]:
    """One import record (see agenteval.traces) per trace, in start-time order."""
    traces: dict[str, list[dict]] = {}
    for span in read_spans(path):
        traces.setdefault(span.get("traceId", ""), []).append(span)

    records = []
    for spans in sorted(traces.values(), key=lambda group: min(int(s.get("startTimeUnixNano", 0)) for s in group)):
        ids = {span.get("spanId") for span in spans}
        roots = [span for span in spans if not span.get("parentSpanId") or span.get("parentSpanId") not in ids]
        root = min(roots or spans, key=lambda s: int(s.get("startTimeUnixNano", 0)))
        root_attributes = _attributes(root)
        question = _text(_first(root_attributes, "agenteval.input", "input.value", "gen_ai.prompt"))
        if not question:
            continue  # nothing to evaluate: not an agent run (e.g. a health check)
        answer = _text(_first(root_attributes, "agenteval.output", "output.value", "gen_ai.completion"))

        steps = []
        for span in sorted(spans, key=lambda s: int(s.get("startTimeUnixNano", 0))):
            attributes = _attributes(span)
            kind = _kind(attributes)
            if kind is None:
                continue
            duration_ms = (int(span.get("endTimeUnixNano", 0)) - int(span.get("startTimeUnixNano", 0))) / 1e6
            failed = (span.get("status") or {}).get("code") in (2, "STATUS_CODE_ERROR")
            error = (span.get("status") or {}).get("message") or "error" if failed else None
            if kind == "llm":
                steps.append({
                    "kind": "llm",
                    "name": _first(attributes, "llm.model_name", "gen_ai.request.model", "gen_ai.response.model")
                    or span.get("name", "unknown-model"),
                    "input_tokens": int(_first(attributes, "llm.token_count.prompt", "gen_ai.usage.input_tokens") or 0),
                    "output_tokens": int(_first(attributes, "llm.token_count.completion",
                                                "gen_ai.usage.output_tokens") or 0),
                    "duration_ms": duration_ms,
                    "error": error,
                })
            else:
                steps.append({
                    "kind": "tool",
                    "name": _first(attributes, "tool.name", "gen_ai.tool.name") or span.get("name", "tool"),
                    "input": _first(attributes, "input.value", "tool.parameters", "gen_ai.tool.call.arguments"),
                    "output": _first(attributes, "output.value", "gen_ai.tool.call.result"),
                    "duration_ms": duration_ms,
                    "error": error,
                })
        records.append({"input": question, "output": answer, "steps": steps})
    return records
