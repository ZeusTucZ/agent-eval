import asyncio
from collections.abc import Callable
from typing import Protocol

from agenteval.models import EvaluationCase, Trial
from agenteval.tracer import Tracer

class AgentFailure(Exception):
    """Raised by an agent for failures that don't belong to any one step, e.g. too many steps."""
    failure_type = "agent_error"

class MaxStepsExceeded(AgentFailure):
    failure_type = "max_steps_exceeded"

class QuotaExhausted(AgentFailure):
    failure_type = "quota_exhausted"

# Model failures, not infrastructure ones, so they get their own labels rather than "llm_error".

class OutputTruncated(AgentFailure):
    """The model hit its output-token limit (often mid-reasoning) before giving an answer."""
    failure_type = "output_truncated"

class EmptyOutput(AgentFailure):
    """The model finished normally but said nothing."""
    failure_type = "empty_output"

class InvalidModelOutput(AgentFailure):
    """The model produced output the provider couldn't parse, e.g. a malformed tool call."""
    failure_type = "invalid_model_output"

class Agent(Protocol):
    async def run(self, case: EvaluationCase, attempt: int, tracer: Tracer) -> str: ...

async def run_experiment(
    agent: Agent,
    dataset: list[EvaluationCase],
    attempts: int = 3,
    concurrency: int = 10,
    timeout_s: float = 30.0,
    on_trial: Callable[[Trial], None] | None = None,
    skip: set[int] = frozenset(),
    should_stop: Callable[[], bool] | None = None,
) -> list[Trial]:
    """Run every case `attempts` times. Trial ids are fixed by position (case order, then
    attempt), so a resumed run gives the missing trials the ids they would have had.

    on_trial is called as each trial finishes (to save it, or show progress). Trials whose
    ids are in `skip` aren't run: they were saved by an earlier, interrupted run. Once
    should_stop() returns True, trials that haven't started are left out of the result.
    """
    # The semaphore caps how many trials run at once, like an API rate limit would.
    semaphore = asyncio.Semaphore(concurrency)

    async def run_trial(trial_id: int, case: EvaluationCase, attempt: int) -> Trial | None:
        async with semaphore:
            if should_stop is not None and should_stop():
                return None
            tracer = Tracer(trial_id)
            output = None
            failure_type = None

            try:
                output = await asyncio.wait_for(agent.run(case, attempt, tracer), timeout=timeout_s)
            except TimeoutError:
                failure_type = _timeout_type(tracer)
            except AgentFailure as error:
                failure_type = error.failure_type
            except Exception:
                failure_type = _failure_type(tracer)

            trial = Trial(
                id=trial_id,
                case_id=case.id,
                attempt=attempt,
                output=output,
                failure_type=failure_type,
                spans=tracer.spans,
                transcript=tracer.transcript,
            )
            if on_trial is not None:
                on_trial(trial)
            return trial

    jobs = [(case, attempt) for case in dataset for attempt in range(1, attempts + 1)]

    # gather returns results in the order the jobs were given, not the order they finished.
    results = await asyncio.gather(*(
        run_trial(trial_id, case, attempt)
        for trial_id, (case, attempt) in enumerate(jobs, start=1)
        if trial_id not in skip
    ))
    return [trial for trial in results if trial is not None]

def _timeout_type(tracer: Tracer) -> str:
    """Tell a step that hung apart from a trial that ran out of time waiting to retry.

    A step cut off mid-call ends with CancelledError. If the last span failed some other
    way, the trial was between attempts (e.g. backing off after a 429) when time ran out,
    and the label says so: "timeout_after_rate_limit" points at the real cause.
    """
    last = tracer.spans[-1] if tracer.spans else None
    if last is None or last.status != "error" or last.error_type == "CancelledError":
        return "timeout"
    return f"timeout_after_{last.error_type}"

def _failure_type(tracer: Tracer) -> str:
    """Blame the span that raised, e.g. "tool_error". Errors outside any span are "agent_error"."""
    for span in reversed(tracer.spans):
        if span.status == "error":
            return f"{span.kind}_error"
    return "agent_error"
