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

class Span(BaseModel):
    trial_id: int
    kind: Literal["llm", "tool"]
    name: str  # for llm spans: the model name, used to look up its price
    start: int
    end: int
    status: Literal["running", "success", "error"]
    error_type: str | None = None
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
