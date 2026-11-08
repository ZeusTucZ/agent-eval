"""HTTP layer over the existing agenteval runner and storage.

Thin on purpose: no evaluation logic lives here. It only exposes what
storage.py, report.py, rescore.py and run.py already do, over HTTP.

Rules the plan fixes and this module keeps:
- All GET endpoints are public and read-only.
- POST /api/runs and POST /api/experiments/{id}/rescore require the
  header ``X-API-Token`` matching the server's API_TOKEN (HTTP 401
  otherwise). The token is not a user account; it only stops a visitor
  from spending the Groq quota by launching runs.
- A run never executes inside the HTTP request: POST /api/runs returns
  a job id at once and the runner executes in a FastAPI BackgroundTask.
  The client polls GET /api/runs/{id} until the job is completed or
  failed. Runs are serialized with a lock: SQLite has a single writer,
  so a second run waits for the first instead of hitting
  "database is locked".
"""
import argparse
import asyncio
import importlib.util
import math
import sqlite3
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agenteval import config, storage
from agenteval.cost import PRICES, total_cost
from agenteval.models import Experiment, Score, Trial
from agenteval.rag import DEFAULT_EMBEDDING, DEFAULT_TOP_K, EMBEDDINGS, MAX_TOP_K
from agenteval.calibrate import agreement, sample_trials
from agenteval.calibrate import verdict as verdict_reading
from agenteval.compare import check_comparable, compare_experiments
from agenteval.compare import load as load_experiment_data
from agenteval.environment import describe as describe_environment
from agenteval.judge import RUBRICS, sources_of
from agenteval.models import EvaluationCase
from agenteval.report import (
    latency_percentiles,
    mean_score,
    other_scores,
    pass_rate_by_category,
    retrieval_diagnosis,
)
from agenteval.sample_data import dataset_by_name
from agenteval.scorers import SCORER_NAMES, score_trials

app = FastAPI(title="agent-eval API", version="0.1.0")


# ---------------------------------------------------------------------------
# Per-client request limit. The read endpoints are public, so one client
# looping on them could slow the service for everyone else.
# ---------------------------------------------------------------------------

class ClientLimiter:
    """At most `limit` requests per client in any 60 seconds."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.hits: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def retry_after(self, client: str, limit: int) -> float | None:
        """Record a request; None if it's allowed, else seconds until it would be."""
        now = self.clock()
        with self.lock:
            if len(self.hits) > 10_000:  # forget clients that went quiet, so memory stays bounded
                self.hits = {key: times for key, times in self.hits.items() if times and now - times[-1] < 60}
            times = self.hits.setdefault(client, deque())
            while times and now - times[0] >= 60:
                times.popleft()
            if len(times) >= limit:
                return 60 - (now - times[0])
            times.append(now)
            return None


_client_limiter = ClientLimiter()


def client_address(request: Request) -> str:
    """The caller's IP. Behind Render's proxy every request comes from the proxy, and the
    real client is the last X-Forwarded-For entry, the one the proxy added; earlier entries
    are whatever the client sent, so trusting them would let anyone dodge the limit."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def limit_requests(request: Request, call_next):
    limit = config.requests_per_client_per_minute()
    if limit and request.url.path != "/health":  # Render's health check must always get through
        wait = _client_limiter.retry_after(client_address(request), limit)
        if wait is not None:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests; slow down and try again shortly."},
                headers={"Retry-After": str(math.ceil(wait))},
            )
    return await call_next(request)


# Added after the limiter so it wraps it: a 429 still carries CORS headers, and the
# browser shows its message instead of a bare network error.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins(),
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["X-API-Token", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Database connection (one per request, never shared between requests).
# ---------------------------------------------------------------------------

def get_conn():
    # FastAPI opens this dependency, runs the endpoint and closes it from worker threads that
    # can differ, so SQLite's same-thread check fails at random with a 500. Each connection
    # belongs to one request and is used by one thread at a time, so the check can go.
    conn = storage.connect(config.database_path(), check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


Conn = Annotated[sqlite3.Connection, Depends(get_conn)]


# ---------------------------------------------------------------------------
# Auth: simple shared token for the two POST endpoints.
# ---------------------------------------------------------------------------

def require_token(x_api_token: str | None = Header(default=None)) -> None:
    """Fail closed: with no API_TOKEN configured, every change is refused. In read-only mode
    (a public demo) changes are refused whatever the token."""
    if config.read_only():
        raise HTTPException(status_code=403, detail="this server is read-only")
    expected = config.api_token()
    if not expected or x_api_token != expected:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Token")


# ---------------------------------------------------------------------------
# Summaries: the fields the dashboard needs, computed from report.py
# helpers so the numbers match the CLI report exactly.
# ---------------------------------------------------------------------------

def _latency(trials: list[Trial], working_only: bool = False) -> dict[str, float]:
    """p50/p95 of trial duration in ms. latency_percentiles needs at least
    two trials; with fewer, the single duration (or 0.0) stands in."""
    if not trials:
        return {"p50": 0.0, "p95": 0.0}
    if len(trials) == 1:
        duration = trials[0].working_ms if working_only else trials[0].duration_ms
        return {"p50": duration, "p95": duration}
    return latency_percentiles(trials, working_only=working_only)


def _mean(scores: list[Score], scorer: str) -> float | None:
    found = [score for score in scores if score.scorer == scorer]
    return mean_score(found) if found else None


# Datasets rebuilt from an experiment's stored name, for questions and categories. Cached: a
# built-in dataset or a sample of one is the same every time it is rebuilt.
_datasets: dict[str, list[EvaluationCase] | None] = {}


def _cases_for(experiment: Experiment) -> list[EvaluationCase] | None:
    name = experiment.dataset_name
    if name not in _datasets:
        try:
            _datasets[name] = dataset_by_name(name)
        except (ValueError, OSError):
            _datasets[name] = None  # e.g. a .jsonl file that isn't on this server
    return _datasets[name]


def experiment_summary(experiment: Experiment, trials: list[Trial], scores: list[Score],
                       cases: list[EvaluationCase] | None = None) -> dict:
    exact_match = [score for score in scores if score.scorer == "exact_match"]
    successes = sum(score.value == 1.0 for score in exact_match)
    cost = total_cost(trials, experiment.prices)
    latency = _latency(trials)
    working = _latency(trials, working_only=True)
    return {
        "id": experiment.id,
        "name": experiment.name,
        "dataset_name": experiment.dataset_name,
        "dataset_version": experiment.dataset_version,
        "model": experiment.config.model,
        "config_version": experiment.config.version,
        "attempts": experiment.attempts,
        "created_at": experiment.created_at.isoformat(),
        "trial_count": len(trials),
        "pass_rate": mean_score(exact_match) if exact_match else None,
        "status": experiment.status,
        "planned_trials": experiment.planned_trials,
        "environment": describe_environment(experiment.environment),
        "by_category": {
            name: {"pass_rate": rate, "trials": count}
            for name, (rate, count) in (pass_rate_by_category(trials, scores, cases) if cases else {}).items()
        },
        "other_scores": {name: {"mean": mean, "trials": count} for name, (mean, count) in other_scores(scores).items()},
        "latency_ms": {"p50": latency["p50"], "p95": latency["p95"]},
        "latency_working_ms": {"p50": working["p50"], "p95": working["p95"]},
        "cost": cost,
        "cost_per_success": (cost / successes) if successes else None,
        "tool_rates": {
            "right_tool": _mean(scores, "tool_selection"),
            "valid_arguments": _mean(scores, "tool_arguments_valid"),
            "ran_without_error": _mean(scores, "tool_execution"),
            "backed_by_tools": _mean(scores, "tool_faithfulness"),
            "no_injected_text": _mean(scores, "avoids_forbidden_text"),
            "relevant_found": _mean(scores, "retrieval_recall"),
            "right_order": _mean(scores, "trajectory_order"),
            "step_efficiency": _mean(scores, "step_efficiency"),
            "no_repeated_calls": _mean(scores, "redundant_calls"),
        },
        # Empty for datasets without retrieval. Keys are report.DIAGNOSES.
        "retrieval_diagnosis": dict(retrieval_diagnosis(scores)),
        "failures": dict(Counter(
            trial.failure_type for trial in trials if trial.failure_type
        )),
    }


def _load_or_404(conn: sqlite3.Connection, experiment_id: int) -> Experiment:
    try:
        return storage.load_experiment(conn, experiment_id)
    except LookupError:
        raise HTTPException(status_code=404, detail=f"no experiment with id {experiment_id}") from None


# ---------------------------------------------------------------------------
# Public read endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# A summary is computed from every trial and score of an experiment: slow for real runs,
# and the list needs one per experiment on every request. Saved trials never change and
# scores only change when re-scored, so each summary is cached under a fingerprint of its
# scores and rebuilt only when they differ, whichever process changed them.
_summaries: dict[tuple[str, int], tuple[tuple, dict]] = {}  # (database, id) -> (fingerprint, summary)
_summaries_lock = threading.Lock()


def _scores_fingerprint(conn: sqlite3.Connection, experiment_id: int) -> tuple:
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(value), 0), COALESCE(SUM(LENGTH(reasoning)), 0)"
        " FROM scores WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    return tuple(row)


def _summary(conn: sqlite3.Connection, experiment_id: int) -> dict:
    key = (config.database_path(), experiment_id)  # ids restart in every database
    fingerprint = _scores_fingerprint(conn, experiment_id)
    with _summaries_lock:
        cached = _summaries.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    experiment = _load_or_404(conn, experiment_id)
    summary = experiment_summary(
        experiment,
        storage.load_trials(conn, experiment_id),
        storage.load_scores(conn, experiment_id),
        _cases_for(experiment),
    )
    with _summaries_lock:
        _summaries[key] = (fingerprint, summary)
    return summary


@app.get("/api/experiments")
def list_experiments(
    conn: Conn,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[dict]:
    """Newest first, so the dashboard shows the latest run on top; a page at a time."""
    rows = conn.execute(
        "SELECT id FROM experiments ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset),
    ).fetchall()
    return [_summary(conn, row["id"]) for row in rows]


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: int, conn: Conn) -> dict:
    return _summary(conn, experiment_id)


@app.get("/api/experiments/{experiment_id}/trials")
def list_trials(experiment_id: int, conn: Conn) -> dict:
    experiment = _load_or_404(conn, experiment_id)
    cases = {case.id: case for case in _cases_for(experiment) or []}
    trials = storage.load_trials(conn, experiment_id)
    scores = storage.load_scores(conn, experiment_id)
    scores_by_trial: dict[int, list[dict]] = {}
    for score in scores:
        scores_by_trial.setdefault(score.trial_id, []).append({
            "scorer": score.scorer,
            "value": score.value,
            "reasoning": score.reasoning,
        })
    return {
        "experiment_id": experiment_id,
        "trials": [
            {**trial.model_dump(), "scores": scores_by_trial.get(trial.id, []),
             # The question it answered, so a failure can be read without the dataset file.
             "case": (cases[trial.case_id].model_dump(include={"input", "expected_output", "category"})
                      if trial.case_id in cases else None)}
            for trial in trials
        ],
    }


# ---------------------------------------------------------------------------
# Background jobs: POST /api/runs returns at once; the run happens after.
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    agent: Literal["mock", "groq", "gemini"] = "mock"
    dataset: Literal["arithmetic", "mixed", "hard", "extended", "rag"] = "mixed"
    # Upper bounds keep one request from occupying the server (and the quota) for hours.
    attempts: int = Field(default=3, ge=1, le=10)
    model: str | None = None
    name: str = Field(default="api-run", max_length=80)
    # Only for the arithmetic dataset: how many "What is N + 1" cases.
    cases: int = Field(default=100, ge=1, le=1000)
    seed: int = 42
    # Only for the rag dataset.
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    embedding: Literal[EMBEDDINGS] = DEFAULT_EMBEDDING


_jobs: dict[str, dict] = {}  # in memory: a restart forgets them, but the experiments are saved
_jobs_lock = threading.Lock()
# Runs execute one at a time, so a longer queue only means waiting; refuse beyond this.
MAX_WAITING_JOBS = 3
# Finished jobs kept for polling; older ones are dropped so memory doesn't grow forever.
MAX_KEPT_JOBS = 100
# SQLite has a single writer: runs queue behind each other instead of
# colliding on "database is locked".
_run_lock = threading.Lock()


def _new_job(request: RunRequest) -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "status": "queued",
        "agent": request.agent,
        "dataset": request.dataset,
        "attempts": request.attempts,
        "model": request.model,
        "name": request.name,
        "experiment_id": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _jobs_lock:
        _jobs[job["id"]] = job
        finished = [job_id for job_id, kept in _jobs.items() if kept["status"] in ("completed", "failed")]
        for job_id in finished[: max(0, len(_jobs) - MAX_KEPT_JOBS)]:  # oldest first: dicts keep order
            del _jobs[job_id]
    return job


def _waiting_jobs() -> int:
    with _jobs_lock:
        return sum(job["status"] in ("queued", "running") for job in _jobs.values())


def _set_job(job_id: str, **updates) -> None:
    with _jobs_lock:
        _jobs[job_id].update(updates)


def _run_args(request: RunRequest) -> argparse.Namespace:
    """The namespace run.py's agent builders expect, with the same
    per-agent defaults the CLI uses (run.py AGENT_DEFAULTS)."""
    from agenteval.run import AGENT_DEFAULTS

    defaults = AGENT_DEFAULTS[request.agent]
    return argparse.Namespace(
        agent=request.agent,
        dataset=request.dataset,
        model=request.model or defaults["model"],
        cases=request.cases,
        attempts=request.attempts,
        concurrency=defaults["concurrency"],
        timeout=defaults["timeout"],
        rpm=defaults["rpm"],
        tpm=None,
        max_steps=5,
        temperature=0.0,
        prompt_file=None,
        tool_error_rate=0.10,
        rate_limit_rate=0.05,
        hallucination_rate=0.15,
        hang_rate=0.0,
        seed=request.seed,
        top_k=request.top_k,
        embedding=request.embedding,
    )


def _execute_run(job_id: str, request: RunRequest) -> None:
    """Runs in a background thread after the HTTP response is sent."""
    from agenteval.models import dataset_version
    from agenteval.run import BUILDERS, load_dataset
    from agenteval.runner import run_experiment

    with _run_lock:
        _set_job(job_id, status="running")
        try:
            args = _run_args(request)
            if args.model not in PRICES:
                raise ValueError(f"No price for {args.model!r}. Add it to PRICES in agenteval/cost.py.")
            agent, agent_config = BUILDERS[request.agent](args)
            dataset_name, dataset = load_dataset(args)

            trials = asyncio.run(run_experiment(
                agent,
                dataset,
                attempts=args.attempts,
                concurrency=args.concurrency,
                timeout_s=args.timeout,
            ))
            scores = score_trials(dataset, trials)
            experiment = Experiment(
                name=request.name,
                dataset_name=dataset_name,
                dataset_version=dataset_version(dataset),
                config=agent_config,
                attempts=args.attempts,
                prices=PRICES,
                created_at=datetime.now(timezone.utc),
            )
            conn = storage.connect(config.database_path())
            try:
                experiment_id = storage.save_experiment(conn, experiment, trials, scores)
            finally:
                conn.close()
            _set_job(job_id, status="completed", experiment_id=experiment_id)
        except BaseException as error:  # noqa: BLE001 - a failed job must report, not crash the worker
            _set_job(job_id, status="failed", error=str(error))


@app.post("/api/runs", status_code=202)
def create_run(
    request: RunRequest,
    background_tasks: BackgroundTasks,
    _token: None = Depends(require_token),
) -> dict:
    """Validate, queue the run, and answer immediately with the job id."""
    from agenteval.run import AGENT_DEFAULTS

    if request.agent == "mock" and request.dataset != "arithmetic":
        # Same rule as the CLI: the mock only answers arithmetic questions.
        raise HTTPException(
            status_code=400,
            detail="The mock agent only understands the arithmetic dataset; use groq or gemini for mixed/hard.",
        )
    model = request.model or AGENT_DEFAULTS[request.agent]["model"]
    if model not in PRICES:
        raise HTTPException(
            status_code=400,
            detail=f"No price for {model!r}. Add it to PRICES in agenteval/cost.py.",
        )
    if request.agent == "groq" and not config.groq_api_key():
        raise HTTPException(status_code=400, detail="Set the GROQ_API_KEY environment variable first.")
    if request.agent == "gemini" and not config.gemini_api_key():
        raise HTTPException(status_code=400, detail="Set the GEMINI_API_KEY environment variable first.")
    # Checked now rather than failing in the background: the API image doesn't install chromadb.
    if request.dataset == "rag" and importlib.util.find_spec("chromadb") is None:
        raise HTTPException(status_code=400, detail='The rag dataset needs chromadb: pip install -e ".[rag]".')

    if _waiting_jobs() >= MAX_WAITING_JOBS:
        raise HTTPException(status_code=429, detail="Too many runs waiting; try again when one finishes.")

    job = _new_job(request)
    background_tasks.add_task(_execute_run, job["id"], request)
    return job


@app.get("/api/runs/{job_id}")
def get_run(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job with id {job_id}")
    return dict(job)


# ---------------------------------------------------------------------------
# Re-score a saved experiment without calling the model again.
# ---------------------------------------------------------------------------

@app.post("/api/experiments/{experiment_id}/rescore")
def rescore_experiment(
    experiment_id: int,
    conn: Conn,
    _token: None = Depends(require_token),
) -> dict:
    from agenteval.sample_data import dataset_by_name

    experiment = _load_or_404(conn, experiment_id)
    trials = storage.load_trials(conn, experiment_id)
    try:
        dataset = dataset_by_name(experiment.dataset_name)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"unknown dataset {experiment.dataset_name!r}; cannot re-score",
        ) from None
    # Only the built-in scorers' scores are replaced; an LLM judge's are kept.
    storage.replace_scores(conn, experiment_id, score_trials(dataset, trials), SCORER_NAMES)
    return experiment_summary(experiment, trials, storage.load_scores(conn, experiment_id), dataset)


# ---------------------------------------------------------------------------
# Managing experiments: rename and delete need the token.
# ---------------------------------------------------------------------------

class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@app.patch("/api/experiments/{experiment_id}")
def rename_experiment(experiment_id: int, body: RenameRequest, conn: Conn,
                      _token: None = Depends(require_token)) -> dict:
    try:
        storage.rename_experiment(conn, experiment_id, body.name)
    except LookupError:
        raise HTTPException(status_code=404, detail=f"no experiment with id {experiment_id}") from None
    _forget_summary(experiment_id)
    return _summary(conn, experiment_id)


@app.delete("/api/experiments/{experiment_id}", status_code=204)
def delete_experiment(experiment_id: int, conn: Conn, _token: None = Depends(require_token)) -> None:
    try:
        storage.delete_experiment(conn, experiment_id)
    except LookupError:
        raise HTTPException(status_code=404, detail=f"no experiment with id {experiment_id}") from None
    _forget_summary(experiment_id)


def _forget_summary(experiment_id: int) -> None:
    with _summaries_lock:
        _summaries.pop((config.database_path(), experiment_id), None)


# ---------------------------------------------------------------------------
# Comparing two experiments: the same paired test as `agenteval compare`.
# ---------------------------------------------------------------------------

def _change(change) -> dict:
    return {"case_id": change.case.id, "input": change.case.input, "category": change.case.category,
            "expected": change.case.expected_output, "before": change.a_output, "after": change.b_output,
            "before_rate": change.a_rate, "after_rate": change.b_rate}


def _interval(metric) -> dict:
    return {"before": metric.a, "after": metric.b, "change": metric.change.mean,
            "low": metric.change.low, "high": metric.change.high}


@app.get("/api/compare")
def compare(a: int, b: int, conn: Conn) -> dict:
    """Is experiment b better than experiment a?"""
    try:
        first, second = load_experiment_data(conn, a), load_experiment_data(conn, b)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    fairness = check_comparable(first, second)
    base = {"a": a, "b": b, "errors": fairness.errors, "warnings": fairness.warnings, "changes": fairness.changes}
    if fairness.errors:
        return {**base, "verdict": None}
    cases = _cases_for(first.experiment)
    if cases is None:
        return {**base, "errors": [f"the dataset {first.experiment.dataset_name} isn't available here"], "verdict": None}
    result = compare_experiments(first, second, cases)
    return {
        **base,
        "verdict": result.verdict,
        "reason": result.reason,
        "cases_compared": result.cases_compared,
        "p_value": result.p_value,
        "pass_rate": _interval(result.pass_rate),
        "latency_ms": _interval(result.working_ms),
        "cost": _interval(result.cost),
        "categories": {name: {"before": before, "after": after, "cases": count}
                       for name, (before, after, count) in result.categories.items()},
        "improved": [_change(change) for change in result.improved],
        "regressed": [_change(change) for change in result.regressed],
        "safety_regressions": [_change(change) for change in result.safety_regressions],
        "partly_changed": result.partly_changed,
    }


# ---------------------------------------------------------------------------
# Labelling answers to check an LLM judge (agenteval calibrate, in the browser).
# ---------------------------------------------------------------------------

@app.get("/api/rubrics")
def list_rubrics() -> list[dict]:
    return [{"name": r.name, "question": r.question, "uses_expected": r.uses_expected,
             "uses_sources": r.uses_sources, "uses_answer": r.uses_answer} for r in RUBRICS.values()]


@app.get("/api/experiments/{experiment_id}/labels")
def get_labels(experiment_id: int, rubric: str, conn: Conn) -> dict:
    """Your labels, the judge's scores, the next answers to label and, once there are both,
    how far the judge agrees with you."""
    experiment = _load_or_404(conn, experiment_id)
    if rubric not in RUBRICS:
        raise HTTPException(status_code=400, detail=f"unknown rubric {rubric!r}")
    trials = storage.load_trials(conn, experiment_id)
    labels = storage.load_labels(conn, experiment_id, rubric)
    judge_scores = [s for s in storage.load_scores(conn, experiment_id) if s.scorer == f"judge_{rubric}"]
    cases = {case.id: case for case in _cases_for(experiment) or []}

    queue = [t for t in sample_trials(trials, RUBRICS[rubric], 50, seed=0) if t.id not in labels]
    agreement_data = None
    if labels and judge_scores:
        try:
            result = agreement(labels, judge_scores)
        except ValueError:
            result = None
        if result is not None:
            agreement_data = {"compared": result.compared, "agreement": result.agreement, "kappa": result.kappa,
                              "reading": verdict_reading(result.kappa, result.compared),
                              "human_pass_rate": result.human_pass_rate, "judge_pass_rate": result.judge_pass_rate}
    return {
        "rubric": rubric,
        "labelled": len(labels),
        "judged": len(judge_scores),
        "agreement": agreement_data,
        "next": [{
            "trial_id": t.id,
            "question": cases[t.case_id].input if t.case_id in cases else None,
            "expected": cases[t.case_id].expected_output if t.case_id in cases else None,
            "sources": sources_of(t),
            "answer": t.output,
        } for t in queue[:5]],
    }


class LabelRequest(BaseModel):
    trial_id: int
    rubric: str
    passed: bool


@app.post("/api/experiments/{experiment_id}/labels")
def add_label(experiment_id: int, body: LabelRequest, conn: Conn, _token: None = Depends(require_token)) -> dict:
    _load_or_404(conn, experiment_id)
    if body.rubric not in RUBRICS:
        raise HTTPException(status_code=400, detail=f"unknown rubric {body.rubric!r}")
    if not any(t.id == body.trial_id for t in storage.load_trials(conn, experiment_id)):
        raise HTTPException(status_code=404, detail=f"no trial {body.trial_id} in experiment {experiment_id}")
    storage.save_label(conn, experiment_id, body.trial_id, body.rubric, body.passed)
    return {"saved": True}


# ---------------------------------------------------------------------------
# The dashboard itself, when built into the package (agenteval ui) or pointed at with
# AGENTEVAL_STATIC_DIR. Mounted last, so it never shadows an API route.
# ---------------------------------------------------------------------------

_static = config.static_dir()
if _static is not None:
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_static, html=True), name="dashboard")
