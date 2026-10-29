from pydantic import BaseModel

class EvaluationCase(BaseModel):
    id: int
    input: str
    expected_output: str
    expected_tool: str

class Span(BaseModel):
    trial_id: int
    kind: str
    name: str
    start: int
    end: int
    status: str
    error_type: str | None = None
    input_tokens: int
    output_tokens: int

class Trial(BaseModel):
    id: int
    case_id: int
    attempt: int
    output: str
    failure_type: str | None = None
    spans: list[Span] | None = None

class Score(BaseModel):
    trial_id: int
    scorer: float
    value: float
    reasoning: str | None = None
