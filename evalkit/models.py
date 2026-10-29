from typing import Literal

from pydantic import BaseModel

class EvaluationCase(BaseModel):
    id: int
    input: str
    expected_output: str
    expected_tool: str | None = None

class Span(BaseModel):
    trial_id: int
    kind: Literal["llm", "tool"]
    name: str
    start: int
    end: int
    status: Literal["running", "success", "error"]
    error_type: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

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
        return sum(span.duration_ms for span in self.spans)

class Score(BaseModel):
    trial_id: int
    scorer: str
    value: float
    reasoning: str | None = None
