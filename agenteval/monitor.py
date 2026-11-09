"""Watch an agent in production: record a sample of real requests and score them.

    from agenteval.monitor import Monitor

    monitor = Monitor("production.db", model="gpt-4o-mini", sample_rate=0.05, redact=True)

    @monitor.watch
    async def answer(question: str) -> str:
        ...  # your agent, with llm_call/tool, wrap_openai/wrap_anthropic or instrument()

Each sampled call is recorded the way an evaluation records a trial (its steps, tokens,
latency, errors and answer) into one experiment per day, "production YYYY-MM-DD". The
checks that need no expected answer run on it: tool use, trajectory, faithfulness to tool
results, failures, cost and latency. Then judge with the reference-free rubrics and follow it
in the dashboard like any other experiment:

    agenteval judge 42 --rubric answer_relevance --db production.db --sample 100

Recording never gets in the way of the request: unsampled calls run untouched, sampled ones
are written by a background thread, and a failure to record is dropped, not raised. Your
function's own errors are recorded and raised as before. A call whose argument is a list of
messages is recorded as a conversation.
"""
import atexit
import functools
import inspect
import queue
import random
import threading
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from agenteval import environment, storage
from agenteval.cost import PRICES
from agenteval.models import AgentConfig, EvaluationCase, Experiment, Price, Trial
from agenteval.privacy import redact, redact_trial
from agenteval.runner import _failure_type
from agenteval.scorers import DEFAULT_SCORERS, score_trials
from agenteval.sdk import _active, _Trial
from agenteval.tracer import Tracer

DATASET = "production"

def _question(args: tuple, kwargs: dict) -> tuple[str, list[dict[str, str]] | None]:
    """What was asked, and the conversation if the call was given one."""
    value = args[0] if args else next(iter(kwargs.values()), "")
    if isinstance(value, list) and value and all(isinstance(m, dict) and "content" in m for m in value):
        messages = [{"role": str(m.get("role", "user")), "content": str(m["content"])} for m in value]
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), messages[-1]["content"])
        return last_user, messages
    return str(value), None

class Monitor:
    def __init__(self, db: str | Path = "production.db", *, model: str, sample_rate: float = 0.1,
                 price: tuple[float, float] | None = None, redact: bool = False, name: str = "production",
                 seed: int | None = None):
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate is a fraction of requests, between 0 and 1")
        self.db = str(db)
        self.model = model
        self.sample_rate = sample_rate
        self.redact = redact
        self.name = name
        self.prices = dict(PRICES)
        if price is not None:
            self.prices[model] = Price(input_per_million=price[0], output_per_million=price[1])
        self.dropped = 0  # sampled calls that couldn't be recorded
        self._random = random.Random(seed)
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._experiments: dict[str, int] = {}

    # --- wrapping the agent ---

    def watch(self, function: Callable) -> Callable:
        """Decorate your production function (sync or async); it behaves exactly as before."""
        if inspect.iscoroutinefunction(function):
            @functools.wraps(function)
            async def watched_async(*args: Any, **kwargs: Any) -> Any:
                if not self._sampled():
                    return await function(*args, **kwargs)
                tracer, token = self._start()
                try:
                    answer = await function(*args, **kwargs)
                except BaseException as error:
                    self._finish(tracer, token, args, kwargs, None, error, function)
                    raise
                self._finish(tracer, token, args, kwargs, answer, None, function)
                return answer
            return watched_async

        @functools.wraps(function)
        def watched(*args: Any, **kwargs: Any) -> Any:
            if not self._sampled():
                return function(*args, **kwargs)
            tracer, token = self._start()
            try:
                answer = function(*args, **kwargs)
            except BaseException as error:
                self._finish(tracer, token, args, kwargs, None, error, function)
                raise
            self._finish(tracer, token, args, kwargs, answer, None, function)
            return answer
        return watched

    def _sampled(self) -> bool:
        with self._lock:
            return self._random.random() < self.sample_rate

    def _start(self) -> tuple[Tracer, Any]:
        tracer = Tracer(0)
        return tracer, _active.set(_Trial(tracer, self.model, attempt=1))

    def _finish(self, tracer: Tracer, token: Any, args: tuple, kwargs: dict, answer: Any,
                error: BaseException | None, function: Callable) -> None:
        _active.reset(token)
        try:
            question, messages = _question(args, kwargs)
            failure = None
            if error is not None:
                failure = "timeout" if isinstance(error, TimeoutError) else _failure_type(tracer)
            elif answer is None or not str(answer).strip():
                failure = "empty_output"
            transcript = None
            if messages is not None:
                transcript = [*messages, {"role": "assistant", "content": "" if answer is None else str(answer)}]
            self._queue.put((datetime.now(UTC), question, None if answer is None else str(answer), failure,
                             list(tracer.spans), transcript, f"{function.__module__}:{function.__qualname__}"))
            self._ensure_writer()
        except Exception:  # recording must never break the request
            self.dropped += 1

    # --- writing, off the request path ---

    def _ensure_writer(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._write_forever, name="agenteval-monitor", daemon=True)
                self._thread.start()
                atexit.register(self.flush)

    def _write_forever(self) -> None:
        conn = storage.connect(self.db)
        while True:
            item = self._queue.get()
            try:
                self._write(conn, *item)
            except Exception as exc:  # a full disk or a locked database: lose the sample, not the app
                self.dropped += 1
                warnings.warn(f"agenteval monitor couldn't record a request: {exc}", stacklevel=1)
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Wait until every sampled call so far is saved (tests, and before the process exits)."""
        if self._thread is not None and self._thread.is_alive():
            self._queue.join()

    def _experiment_id(self, conn, day: str, agent: str) -> int:
        name = f"{self.name} {day}"
        if name in self._experiments:
            return self._experiments[name]
        row = conn.execute("SELECT id FROM experiments WHERE name = ? AND dataset_name = ? ORDER BY id DESC",
                           (name, DATASET)).fetchone()
        if row is not None:
            experiment_id = row["id"]
        else:
            experiment_id = storage.start_experiment(conn, Experiment(
                name=name, dataset_name=DATASET, config=AgentConfig(model=self.model, prompt="",
                                                                    params={"agent": agent, "monitor": True}),
                attempts=1, prices=self.prices, created_at=datetime.now(UTC), status="completed",
                environment=environment.capture(),
            ), [])
        self._experiments[name] = experiment_id
        return experiment_id

    def _write(self, conn, when: datetime, question: str, answer: str | None, failure: str | None, spans: list,
               transcript: list | None, agent: str) -> None:
        experiment_id = self._experiment_id(conn, when.date().isoformat(), agent)
        row = conn.execute("SELECT COALESCE(MAX(trial_id), 0) AS last FROM trials WHERE experiment_id = ?",
                           (experiment_id,)).fetchone()
        number = row["last"] + 1
        for span in spans:
            span.trial_id = number
        case = EvaluationCase(id=number, input=question, expected_output="", category="production")
        trial = Trial(id=number, case_id=number, attempt=1, output=answer, failure_type=failure, spans=spans,
                      transcript=transcript)
        if self.redact:
            trial = redact_trial(trial)
            case = case.model_copy(update={"input": redact(question) or ""})
        cases = [*(storage.load_cases(conn, experiment_id) or []), case]
        storage.save_trials(conn, experiment_id, [trial], score_trials(cases, [trial], DEFAULT_SCORERS))
        storage.save_cases(conn, experiment_id, cases)
