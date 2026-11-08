"""The Python API: evaluate an agent from a notebook, a script or a test.

    from agenteval import evaluate

    results = evaluate(my_agent, "cases.jsonl", model="gpt-4o-mini", price=(0.15, 0.60))
    print(results)                 # the report
    results.pass_rate              # 0.875
    results.failures()             # the trials that didn't pass, with question and answer
    results.to_pandas()            # one row per trial
    results.compare(other)         # is this version really better than `other`?

Results are kept in memory unless you pass db="agenteval.db", in which case they're saved
like a CLI run, so `agenteval report` and the dashboard can show them.
"""
import asyncio
import hashlib
import inspect
import threading
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from agenteval import storage
from agenteval.compare import ExperimentData, check_comparable, compare_experiments, format_comparison
from agenteval.cost import PRICES, with_unpriced_models
from agenteval.execution import RunPlan, aexecute
from agenteval.export import render, trial_rows
from agenteval.models import AgentConfig, EvaluationCase, Experiment, Price, Score, Trial
from agenteval.report import summarize, summary_data
from agenteval.runner import Agent
from agenteval.sample_data import dataset_by_name, load_dataset_file, sample_cases, sample_name
from agenteval.scorers import DEFAULT_SCORERS, Scorer, as_scorer
from agenteval.sdk import FunctionAgent

CasesLike = str | Path | Iterable[EvaluationCase | dict[str, Any]]

class Results:
    """One evaluated experiment: its trials, scores and the cases they answered."""

    def __init__(self, experiment: Experiment, trials: list[Trial], scores: list[Score],
                 cases: list[EvaluationCase]):
        self.experiment, self.trials, self.scores, self.cases = experiment, trials, scores, cases

    @property
    def experiment_id(self) -> int | None:
        return self.experiment.id

    @property
    def summary(self) -> dict:
        """Everything the report shows, as a dict."""
        return summary_data(self.experiment, self.trials, self.scores, self.cases)

    @property
    def pass_rate(self) -> float | None:
        return self.summary["pass_rate"]

    @property
    def cost(self) -> float:
        return self.summary["cost"]

    def score(self, scorer: str) -> float | None:
        """Mean of one scorer's scores, e.g. results.score("tool_selection")."""
        found = [s.value for s in self.scores if s.scorer == scorer]
        return sum(found) / len(found) if found else None

    def report(self, fmt: str = "text") -> str:
        """The report as text, or "json", "markdown", "html" or "csv"."""
        if fmt == "text":
            return summarize(self.experiment, self.trials, self.scores, self.cases)
        return render(fmt, self.experiment, self.trials, self.scores, self.cases)

    def save(self, path: str | Path) -> Path:
        """Write the report; the format follows the extension (.html, .md, .json, .csv, .txt)."""
        path = Path(path)
        fmt = {".html": "html", ".md": "markdown", ".json": "json", ".csv": "csv"}.get(path.suffix, "text")
        path.write_text(self.report(fmt), encoding="utf-8")
        return path

    def to_records(self) -> list[dict]:
        """One dict per trial: question, expected answer, answer, failure, cost and every score."""
        return trial_rows(self.experiment, self.trials, self.scores, self.cases)

    def to_pandas(self):
        import pandas as pd  # optional: only needed here

        return pd.DataFrame(self.to_records())

    def failures(self) -> list[dict]:
        """The trials that didn't pass exact_match, or stopped with a failure."""
        return [row for row in self.to_records() if row["failure"] or row.get("exact_match") == 0.0]

    def check(
        self,
        min_pass_rate: float | None = None,
        min_scores: dict[str, float] | None = None,
        max_cost: float | None = None,
        max_failures: int | None = None,
    ) -> "Results":
        """Raise AssertionError, listing what fell short, unless every limit is met. For tests:

            def test_support_agent():
                evaluate(answer, "cases.jsonl", model="gpt-4o-mini").check(
                    min_pass_rate=0.9, min_scores={"tool_selection": 1.0}, max_cost=0.50)

        Returns the results, so the call can be chained or kept.
        """
        problems = []
        rate = self.pass_rate
        if min_pass_rate is not None and (rate is None or rate < min_pass_rate):
            shown = "n/a" if rate is None else f"{rate:.1%}"
            problems.append(f"pass rate {shown} is below {min_pass_rate:.1%}")
        for scorer, minimum in (min_scores or {}).items():
            value = self.score(scorer)
            if value is None:
                problems.append(f"{scorer} produced no scores (does it apply to these cases?)")
            elif value < minimum:
                problems.append(f"{scorer} is {value:.1%}, below {minimum:.1%}")
        if max_cost is not None and self.cost > max_cost:
            problems.append(f"cost ${self.cost:.4f} is over ${max_cost:.4f}")
        failures = [t for t in self.trials if t.failure_type]
        if max_failures is not None and len(failures) > max_failures:
            problems.append(f"{len(failures)} trials failed (more than {max_failures})")
        if problems:
            examples = [
                f"  case {row['case']}: {row['question'][:70]!r} -> {(row['answer'] or row['failure'])[:70]!r}"
                + (f" (expected {row['expected'][:40]!r})" if row["expected"] else "")
                for row in self.failures()[:5]
            ]
            message = "agenteval check failed:\n  " + "\n  ".join(problems)
            if examples:
                message += "\nTrials that didn't pass:\n" + "\n".join(examples)
            raise AssertionError(message)
        return self

    def compare(self, baseline: "Results") -> str:
        """Is this run better than `baseline`? The same paired test as `agenteval compare`."""
        a = ExperimentData(baseline.experiment, baseline.trials, baseline.scores)
        b = ExperimentData(self.experiment, self.trials, self.scores)
        fairness = check_comparable(a, b)
        if fairness.errors:
            return "Can't compare: " + "; ".join(fairness.errors)
        return format_comparison(a, b, fairness, compare_experiments(a, b, self.cases))

    def __repr__(self) -> str:
        rate = self.pass_rate
        return (f"<Results {self.experiment.name!r}: {len(self.trials)} trials, "
                f"pass rate {'n/a' if rate is None else f'{rate:.1%}'}>")

    def __str__(self) -> str:
        return self.report()

    def _repr_html_(self) -> str:  # Jupyter shows the HTML report
        return self.report("html")

def _to_cases(cases: CasesLike) -> tuple[str, list[EvaluationCase]]:
    if isinstance(cases, str | Path):
        name = str(cases)
        if name.endswith(".jsonl"):
            return name, load_dataset_file(Path(name))
        return name, dataset_by_name(name)
    built = []
    for number, case in enumerate(cases, start=1):
        if isinstance(case, EvaluationCase):
            built.append(case)
        else:
            built.append(EvaluationCase.model_validate({"id": number, **case}))
    if not built:
        raise ValueError("no cases to evaluate")
    return "inline", built

def _agent_and_config(agent: Callable | Agent, model: str, config: dict | None) -> tuple[Agent, AgentConfig]:
    if callable(getattr(agent, "run", None)) and not inspect.isfunction(agent):
        wrapped, target = agent, type(agent)
    else:
        wrapped, target = FunctionAgent(agent, model), agent
    # The agent's source is part of its version, as in the CLI, so compare can tell edits apart.
    try:
        source = Path(inspect.getsourcefile(target)).read_bytes()
    except (TypeError, OSError):
        source = repr(target).encode()
    name = f"{getattr(target, '__module__', '?')}:{getattr(target, '__qualname__', type(target).__name__)}"
    params = {"agent": name, "source": hashlib.sha256(source).hexdigest()[:12], **(config or {})}
    return wrapped, AgentConfig(model=model, prompt=str((config or {}).get("prompt", "")), params=params)

async def aevaluate(
    agent: Callable | Agent,
    cases: CasesLike,
    *,
    model: str,
    price: tuple[float, float] | None = None,
    attempts: int = 3,
    scorers: Iterable[Callable] = (),
    name: str = "python-run",
    db: str | Path | None = None,
    concurrency: int = 4,
    timeout: float = 120.0,
    max_cost: float | None = None,
    sample: int | None = None,
    redact: bool = False,
    progress: bool = False,
    config: dict | None = None,
) -> Results:
    """The async form of evaluate(), for code already running an event loop."""
    dataset_name, case_list = _to_cases(cases)
    if sample:
        case_list = sample_cases(case_list, sample)
        dataset_name = sample_name(dataset_name, sample, 0)
    prices = dict(PRICES)
    if price is not None:
        prices[model] = Price(input_per_million=price[0], output_per_million=price[1])
    if model not in prices:
        raise ValueError(f"no price for {model!r}: pass price=(input, output) in dollars per million tokens, "
                         "or price=(0, 0) if you don't track cost")
    wrapped, agent_config = _agent_and_config(agent, model, config)
    custom: list[Scorer] = [as_scorer(fn) for fn in scorers]

    plan = RunPlan(agent=wrapped, config=agent_config, dataset_name=dataset_name, cases=case_list,
                   attempts=attempts, concurrency=concurrency, timeout_s=timeout, prices=prices,
                   scorers=[*DEFAULT_SCORERS, *custom], redact=redact)
    conn = storage.connect(str(db) if db is not None else ":memory:")
    def finalize_prices(trials: list[Trial]) -> dict[str, Price]:
        complete, unpriced = with_unpriced_models(prices, trials)
        if unpriced:
            warnings.warn(f"no price for {', '.join(unpriced)}; their cost is counted as $0. Pass price=(...) "
                          "for the model your agent calls, or name it in llm_call().", stacklevel=3)
        return complete

    result = await aexecute(plan, conn, name=name, max_cost=max_cost, progress=progress,
                            finalize_prices=finalize_prices)
    experiment = storage.load_experiment(conn, result.experiment_id)
    return Results(experiment, result.trials, result.scores, case_list)

def evaluate(agent: Callable | Agent, cases: CasesLike, **options: Any) -> Results:
    """Run `agent` over `cases` and score every trial. See the module docstring, and
    aevaluate() for the options (model is required; price unless the model is in the table).

    `cases` is a .jsonl path, a built-in dataset name, or a list of dicts or EvaluationCases
    (only "input" and "expected_output" are required). Works in notebooks too: when an event
    loop is already running, the evaluation runs in its own thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aevaluate(agent, cases, **options))

    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["results"] = asyncio.run(aevaluate(agent, cases, **options))
        except BaseException as error:  # noqa: BLE001 - re-raised in the caller's thread
            outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["results"]
