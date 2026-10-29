from typing import Protocol

from evalkit.models import EvaluationCase, Trial
from evalkit.tracer import Tracer

class Agent(Protocol):
    def run(self, case: EvaluationCase, tracer: Tracer) -> str: ...

def run_experiment(agent: Agent, dataset: list[EvaluationCase], attempts: int = 3) -> list[Trial]:
    trials = []
    trial_id = 0

    for case in dataset:
        for attempt in range(1, attempts + 1):
            trial_id += 1
            tracer = Tracer(trial_id)
            output = None
            failure_type = None

            try:
                output = agent.run(case, tracer)
            except Exception:
                failure_type = _failure_type(tracer)

            trials.append(Trial(
                id=trial_id,
                case_id=case.id,
                attempt=attempt,
                output=output,
                failure_type=failure_type,
                spans=tracer.spans,
            ))

    return trials

def _failure_type(tracer: Tracer) -> str:
    """Blame the span that raised, e.g. "tool_error". Errors outside any span are "agent_error"."""
    for span in reversed(tracer.spans):
        if span.status == "error":
            return f"{span.kind}_error"
    return "agent_error"
