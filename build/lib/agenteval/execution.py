"""Running an experiment and saving it as it goes: shared by the CLI and the Python API.

Each trial is saved the moment it finishes, so a run that is interrupted (Ctrl+C, a crash,
a budget reached) keeps everything already paid for, and can be resumed: the missing
trials run with the ids they would have had, and nothing done twice.
"""
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agenteval import environment, storage
from agenteval.cost import trial_cost
from agenteval.models import AgentConfig, EvaluationCase, Experiment, Price, Score, Trial, dataset_version
from agenteval.privacy import redact_trial
from agenteval.runner import Agent, run_experiment
from agenteval.scorers import DEFAULT_SCORERS, Scorer

@dataclass
class RunPlan:
    agent: Agent
    config: AgentConfig
    dataset_name: str
    cases: list[EvaluationCase]
    attempts: int = 3
    concurrency: int = 4
    timeout_s: float = 360.0
    prices: dict[str, Price] = field(default_factory=dict)
    scorers: list[Scorer] = field(default_factory=lambda: list(DEFAULT_SCORERS))
    custom_scorer_specs: list[str] = field(default_factory=list)  # saved, so rescore can reload them
    redact: bool = False

    @property
    def planned_trials(self) -> int:
        return len(self.cases) * self.attempts

class BudgetGuard:
    """Stops starting new trials once the money spent passes `max_cost`. Trials already
    running finish (and are saved): stopping them would waste what they've cost so far."""

    def __init__(self, max_cost: float | None, prices: dict[str, Price]):
        self.max_cost = max_cost
        self.prices = prices
        self.spent = 0.0

    def add(self, trial: Trial) -> None:
        try:
            self.spent += trial_cost(trial, self.prices)
        except KeyError:
            pass  # a model without a price can't be counted; prices_for warns about it

    def exceeded(self) -> bool:
        return self.max_cost is not None and self.spent >= self.max_cost

class Progress:
    """One line on stderr: trials done, failures so far, money spent and time left."""

    def __init__(self, total: int, done: int = 0, enabled: bool = True, stream=sys.stderr):
        self.total, self.done, self.failed = total, done, 0
        self.started = time.monotonic()
        self.run_done = 0  # trials finished in this run, for the time estimate
        self.enabled = enabled
        self.stream = stream
        self.interactive = enabled and stream.isatty()
        self.last_printed = -1

    def update(self, trial: Trial, spent: float) -> None:
        self.done += 1
        self.run_done += 1
        self.failed += trial.failure_type is not None
        if not self.enabled:
            return
        # On a terminal, redraw one line; in CI logs, print every 10% instead of flooding them.
        step = max(1, self.total // 10)
        if not self.interactive and self.done != self.total and self.done // step == self.last_printed:
            return
        self.last_printed = self.done // step

        elapsed = time.monotonic() - self.started
        remaining = (self.total - self.done) * elapsed / self.run_done
        # Plain ASCII: some Windows consoles can't print anything fancier.
        line = (f"  {self.done}/{self.total} trials, {self.failed} failed, ${spent:.4f}, "
                f"{_duration(remaining)} left")
        if self.interactive:
            print(f"\r{line:<70}", end="" if self.done < self.total else "\n", file=self.stream, flush=True)
        else:
            print(line, file=self.stream, flush=True)

def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds // 60:.0f}m{seconds % 60:02.0f}s"
    return f"{seconds // 3600:.0f}h{seconds % 3600 // 60:02.0f}m"

@dataclass
class RunResult:
    experiment_id: int
    trials: list[Trial]  # every trial of the experiment, including any saved by an earlier run
    scores: list[Score]
    stopped_by_budget: bool = False

def new_experiment(plan: RunPlan, name: str) -> Experiment:
    return Experiment(
        name=name,
        dataset_name=plan.dataset_name,
        dataset_version=dataset_version(plan.cases),
        config=plan.config,
        attempts=plan.attempts,
        prices=plan.prices,
        created_at=datetime.now(UTC),
        status="running",
        planned_trials=plan.planned_trials,
        environment=environment.capture(),
        custom_scorers=plan.custom_scorer_specs,
    )

def check_resumable(stored: Experiment, plan: RunPlan) -> None:
    """Resuming only makes sense with the same agent, data and attempts as the first run."""
    problems = []
    if stored.config.version != plan.config.version:
        problems.append(f"the agent's config changed (version {stored.config.version} -> {plan.config.version})")
    if stored.dataset_version != dataset_version(plan.cases):
        problems.append("the dataset changed")
    if stored.attempts != plan.attempts:
        problems.append(f"attempts differ ({stored.attempts} vs {plan.attempts})")
    if problems:
        raise ValueError("can't resume: " + "; ".join(problems) + ". Start a new experiment instead.")

async def aexecute(
    plan: RunPlan,
    conn,
    name: str = "experiment",
    resume_id: int | None = None,
    max_cost: float | None = None,
    progress: bool = True,
    finalize_prices: Callable[[list[Trial]], dict[str, Price]] | None = None,
) -> RunResult:
    """Run the plan, saving each trial as it finishes. With resume_id, continue that
    experiment: only the trials it doesn't have yet are run."""
    if resume_id is not None:
        experiment = storage.load_experiment(conn, resume_id)
        check_resumable(experiment, plan)
        experiment_id = resume_id
        done = {trial.id for trial in storage.load_trials(conn, resume_id)}
        experiment.status = "running"
        storage.finish_experiment(conn, experiment_id, experiment)
    else:
        experiment = new_experiment(plan, name)
        experiment_id = storage.start_experiment(conn, experiment, plan.cases)
        done = set()

    cases = {case.id: case for case in plan.cases}
    budget = BudgetGuard(max_cost, plan.prices)
    for trial in storage.load_trials(conn, experiment_id) if done else []:
        budget.add(trial)  # an earlier run's spending counts toward the budget too
    bar = Progress(plan.planned_trials, done=len(done), enabled=progress)

    def save(trial: Trial) -> None:
        scores = [score for scorer in plan.scorers if (score := scorer(cases[trial.case_id], trial)) is not None]
        stored = redact_trial(trial) if plan.redact else trial
        storage.save_trials(conn, experiment_id, [stored], scores)
        budget.add(trial)
        bar.update(trial, budget.spent)

    try:
        await run_experiment(
            plan.agent, plan.cases, attempts=plan.attempts, concurrency=plan.concurrency,
            timeout_s=plan.timeout_s, on_trial=save, skip=done, should_stop=budget.exceeded,
        )
    finally:
        # Whatever stopped the run (Ctrl+C, a crash, the budget), record how far it got, so
        # the experiment isn't left "running" and can be resumed.
        trials = storage.load_trials(conn, experiment_id)
        complete = len(trials) == plan.planned_trials
        experiment.status = "completed" if complete else "interrupted"
        if finalize_prices is not None:
            experiment.prices = finalize_prices(trials)
        storage.finish_experiment(conn, experiment_id, experiment)

    return RunResult(
        experiment_id=experiment_id,
        trials=trials,
        scores=storage.load_scores(conn, experiment_id),
        stopped_by_budget=budget.exceeded() and not complete,
    )
