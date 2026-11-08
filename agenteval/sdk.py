"""Connect your own agent to agent-eval.

Write the agent as a function that takes the question and returns the answer, and wrap
its model calls and tool calls so each one is recorded as a step of the trial:

    from agenteval import llm_call, tool

    async def my_agent(question: str) -> str:
        with llm_call("gpt-4o-mini") as call:
            response = await client.chat.completions.create(...)
            call.tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
        weather = await tool("get_weather", get_weather, city="Lisbon")
        ...
        return answer

Then run it like a built-in agent:

    python -m agenteval.run --agent my_project.agents:my_agent --dataset my_cases.jsonl

Steps recorded this way get the same treatment as the built-in agents' steps: latency,
cost from the tokens, failures labelled by the step that raised, every scorer, the
report, version comparisons and the gate. Outside an evaluation the helpers just run
what they wrap, so the agent works on its own too.
"""
import asyncio
import hashlib
import importlib
import inspect
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenteval.models import EvaluationCase, Span
from agenteval.runner import Agent, EmptyOutput
from agenteval.tools import ToolArgumentError  # noqa: F401 - re-exported: raise it for invalid arguments
from agenteval.tracer import Tracer, clip

# Tool results are recorded up to this length: scorers such as tool_faithfulness read them.
MAX_RECORDED_TOOL_OUTPUT = 4000

@dataclass
class _Trial:
    tracer: Tracer
    model: str  # the run's --model, used when llm_call isn't given one
    attempt: int = 1  # so a response cache keeps attempts at the same question independent

_active: ContextVar[_Trial | None] = ContextVar("agenteval_trial", default=None)

class LLMCall:
    """What `llm_call` yields: report the call's tokens so its cost can be counted."""

    def __init__(self, span: Span):
        self.span = span

    def tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.span.input_tokens = int(input_tokens or 0)
        self.span.output_tokens = int(output_tokens or 0)

    def failed_because(self, reason: str) -> None:
        """Label the error the call is about to raise, e.g. "rate_limit", so the report counts
        it by cause rather than by exception class."""
        self.span.error_type = reason

@contextmanager
def llm_call(model: str | None = None) -> Iterator[LLMCall]:
    """Record one request to a model. `model` names its price in the price table; it defaults
    to the run's --model."""
    trial = _active.get()
    if trial is None:  # not being evaluated: nothing to record
        yield LLMCall(Span(trial_id=0, kind="llm", name=model or "", start=0, end=0, status="running"))
        return
    with trial.tracer.span(kind="llm", name=model or trial.model) as span:
        yield LLMCall(span)

async def tool(name: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call `function(*args, **kwargs)` as the tool `name`, recording its arguments and result.

    `function` may be sync or async. An exception is recorded as the tool's error and raised
    again, so the agent can handle it (and a trial that ends on it is labelled "tool_error").
    Raise agenteval.ToolArgumentError for arguments the tool can't accept: it is scored as a
    bad call by the agent rather than a failure of the tool.
    """
    trial = _active.get()
    if trial is None:
        result = function(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    with trial.tracer.span(kind="tool", name=name) as span:
        arguments = {"args": list(args), **kwargs} if args else kwargs
        span.input = clip(json.dumps(arguments, ensure_ascii=False, default=str))
        result = function(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        span.output = clip(json.dumps(result, ensure_ascii=False, default=str), MAX_RECORDED_TOOL_OUTPUT)
        return result

class FunctionAgent:
    """Runs a plain function as an agent: it gets the question (or, if it has a parameter
    named `case`, the whole EvaluationCase) and returns the answer."""

    def __init__(self, function: Callable[..., Any], model: str):
        self.function = function
        self.model = model
        self.wants_case = "case" in inspect.signature(function).parameters

    async def run(self, case: EvaluationCase, attempt: int, tracer: Tracer) -> str:
        argument = case if self.wants_case else case.input
        token = _active.set(_Trial(tracer, self.model, attempt))
        try:
            if inspect.iscoroutinefunction(self.function):
                answer = await self.function(argument)
            else:
                # In a thread, so a blocking agent doesn't stop the other trials. The thread gets
                # a copy of this context, so llm_call and tool still find the trial. A timeout
                # can't stop a thread, though: prefer an async agent if steps can hang.
                answer = await asyncio.to_thread(self.function, argument)
        finally:
            _active.reset(token)

        if answer is None or not str(answer).strip():
            raise EmptyOutput("the agent returned no answer")
        return str(answer)

def load_agent(spec: str, model: str) -> tuple[Agent, str]:
    """Import "package.module:name". Returns the agent and a hash of its module's source.

    `name` can be a function (wrapped in FunctionAgent) or an object with the built-in
    agents' `run(case, attempt, tracer)` method. The source hash goes in the agent's config,
    so editing the agent's code makes a new version that agenteval.compare can tell apart.
    """
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"expected module.path:name, got {spec!r}")

    # `python -m` from the project root already has it on the path; a script entry point may not.
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    module = importlib.import_module(module_name)
    try:
        target = getattr(module, attribute)
    except AttributeError:
        raise ValueError(f"{module_name} has no {attribute!r}") from None

    # Read from disk, not with inspect.getsource: that is cached per process, so a notebook
    # that edits and reloads the agent would keep the old version.
    module_file = getattr(module, "__file__", None)
    source = Path(module_file).read_bytes() if module_file else spec.encode()  # e.g. a built-in module
    source_hash = hashlib.sha256(source).hexdigest()[:12]

    if callable(getattr(target, "run", None)) and not inspect.isfunction(target):
        return target, source_hash
    if callable(target):
        return FunctionAgent(target, model), source_hash
    raise ValueError(f"{spec} is neither a function nor an object with a run(case, attempt, tracer) method")
