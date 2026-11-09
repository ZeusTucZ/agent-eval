import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, model_validator

class EvaluationCase(BaseModel):
    id: int
    input: str = ""  # for a conversation, defaults to its last user turn
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
    # Retrieval cases: the ids of the documents the answer is in, all of which a search should
    # find. [] means no document has the answer; None means the case involves no retrieval.
    relevant_docs: list[str] | None = None
    # The tools the agent should use, in this order (other calls may come in between):
    # ["search", "calculator"] means search first, then calculate. For trajectory_order.
    expected_trajectory: list[str] | None = None
    # The most tool calls a good run needs; more counts against step_efficiency.
    max_tool_calls: int | None = None
    # A conversation: the user's messages in order. The agent answers each in turn and
    # expected_output is checked against its last reply.
    turns: list[str] | None = None
    # A conversation with a simulated user: who they are and what they want. `input` is their
    # first message; a user simulator writes each next one in reply to the agent.
    simulated_user: str | None = None
    max_turns: int | None = None  # user messages in a simulated conversation (default 5)

    @model_validator(mode="after")
    def _question(self) -> "EvaluationCase":
        if not self.input:
            if not self.turns:
                raise ValueError('a case needs "input" (the question) or "turns" (a conversation)')
            self.input = self.turns[-1]
        return self

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
    # For a conversation: every message, [{"role": "user" | "assistant", "content": ...}].
    transcript: list[dict[str, str]] | None = None

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
    # "running" while trials are still being saved, "completed", or "interrupted" when a run
    # stopped early (Ctrl+C, a crash, --max-cost); an interrupted run can be resumed.
    status: Literal["running", "completed", "interrupted"] = "completed"
    planned_trials: int | None = None  # cases x attempts, so a partial run shows what's missing
    # Where it ran: Python, agenteval and provider SDK versions, and the git commit of the
    # code, so a result can be traced back to exactly what produced it.
    environment: dict[str, Any] = {}
    # Your own scorers, as "module:function", so rescoring later can run them again.
    custom_scorers: list[str] = []

    @property
    def complete(self) -> bool:
        return self.status == "completed"

# Fields added to EvaluationCase after dataset versions were first stored. Left out of the
# hash while unset, so adding a field doesn't make every saved experiment look like it ran
# on a different dataset (compare and the gate refuse to compare those).
FIELDS_ADDED_LATER = {"relevant_docs", "expected_trajectory", "max_tool_calls", "turns", "simulated_user",
                      "max_turns"}

def dataset_version(cases: list[EvaluationCase]) -> str:
    """Content hash of a dataset: if any case changes, runs are no longer comparable."""
    dumped = [
        {name: value for name, value in case.model_dump().items()
         if not (name in FIELDS_ADDED_LATER and value is None)}
        for case in cases
    ]
    content = json.dumps(dumped, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:12]
