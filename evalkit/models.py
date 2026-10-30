import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

class EvaluationCase(BaseModel):
    id: int
    input: str
    expected_output: str
    expected_tool: str | None = None
    # Every tool the case needs, repeated once per call it needs: ["calculator", "calculator"]
    # means at least two calculator calls. [] means the agent must call no tool at all.
    # None falls back to expected_tool.
    expected_tools: list[str] | None = None
    category: str = "general"  # for the per-category breakdown in reports
    must_not_contain: list[str] = []  # text the output must never include, e.g. an injected "HACKED"
    # The tool is meant to fail here (division by zero, a missing record): the test is how the
    # agent handles the error, so a failed tool call isn't held against it.
    expects_tool_error: bool = False

    @property
    def required_tools(self) -> list[str] | None:
        if self.expected_tools is not None:
            return self.expected_tools
        return [self.expected_tool] if self.expected_tool is not None else None

class Span(BaseModel):
    trial_id: int
    kind: Literal["llm", "tool"]
    name: str  # for llm spans: the model name, used to look up its price
    start: int
    end: int
    status: Literal["running", "success", "error"]
    error_type: str | None = None
    error_message: str | None = None  # the error's text, e.g. which quota a 429 hit
    input: str | None = None  # tool spans: the arguments, as the model sent them
    output: str | None = None  # tool spans: the result, as JSON
    input_tokens: int = 0
    output_tokens: int = 0
    retry: int = 0  # 0 for a first try; 1, 2... for retries of the same call

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) / 1_000_000

class Trial(BaseModel):
    id: int
    case_id: int
    attempt: int
    output: str | None = None
    failure_type: str | None = None
    spans: list[Span] = []

    @property
    def duration_ms(self) -> float:
        """First span start to last span end, so time spent waiting between retries counts."""
        if not self.spans:
            return 0.0
        return (max(span.end for span in self.spans) - min(span.start for span in self.spans)) / 1_000_000

    @property
    def working_ms(self) -> float:
        """Time inside spans only: duration minus waiting for retries or the rate limiter."""
        return sum(span.duration_ms for span in self.spans)

class Score(BaseModel):
    trial_id: int
    scorer: str
    value: float
    reasoning: str | None = None

class Price(BaseModel):
    """Dollars per million tokens."""
    input_per_million: float
    output_per_million: float

class AgentConfig(BaseModel):
    model: str
    prompt: str
    params: dict[str, Any] = {}  # anything JSON-serializable

    @property
    def version(self) -> str:
        """Content hash: any change to the config gives a new version."""
        content = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

class Experiment(BaseModel):
    id: int | None = None  # assigned by storage
    name: str
    dataset_name: str
    config: AgentConfig
    attempts: int
    prices: dict[str, Price]  # snapshot at run time, so later price changes don't rewrite history
    created_at: datetime
    dataset_version: str | None = None  # None for experiments saved before Phase 4

def dataset_version(cases: list[EvaluationCase]) -> str:
    """Content hash of a dataset: if any case changes, runs are no longer comparable."""
    content = json.dumps([case.model_dump() for case in cases], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:12]
