"""HTTP layer over the existing evalkit runner and storage.

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
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from evalkit import config, storage
from evalkit.cost import PRICES, total_cost
from evalkit.models import Experiment, Score, Trial
from evalkit.rag import DEFAULT_EMBEDDING, DEFAULT_TOP_K, EMBEDDINGS, MAX_TOP_K
from evalkit.report import latency_percentiles, mean_score
from evalkit.scorers import SCORER_NAMES, score_trials

app = FastAPI(title="agent-eval API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Token", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Database connection (one per request; SQLite connections are not shared
# across the threads FastAPI runs sync endpoints in).
# ---------------------------------------------------------------------------

def get_conn():
    conn = storage.connect(config.database_path())
    try:
        yield conn
    finally:
        conn.close()


Conn = Annotated[sqlite3.Connection, Depends(get_conn)]


# ---------------------------------------------------------------------------
# Auth: simple shared token for the two POST endpoints.
# ---------------------------------------------------------------------------

def require_token(x_api_token: str | None = Header(default=None)) -> None:
    """Fail closed: with no API_TOKEN configured, every POST is refused."""
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


def experiment_summary(experiment: Experiment, trials: list[Trial], scores: list[Score]) -> dict:
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
        "pass_rate": mean_score(exact_match),
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
        },
        "failures": dict(Counter(
            trial.failure_type for trial in trials if trial.failure_type
        )),
    }


def _load_or_404(conn: sqlite3.Connection, experiment_id: int) -> Experiment:
    try:
        return storage.load_experiment(conn, experiment_id)
    except LookupError:
        raise HTTPException(status_code=404, detail=f"no experiment with id {experiment_id}")


# ---------------------------------------------------------------------------
# Public read endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/experiments")
def list_experiments(conn: Conn) -> list[dict]:
    """Newest first, so the dashboard shows the latest run on top."""
    rows = conn.execute("SELECT id FROM experiments ORDER BY id DESC").fetchall()
    return [
        experiment_summary(
            storage.load_experiment(conn, row["id"]),
            storage.load_trials(conn, row["id"]),
            storage.load_scores(conn, row["id"]),
        )
        for row in rows
    ]


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: int, conn: Conn) -> dict:
    experiment = _load_or_404(conn, experiment_id)
    return experiment_summary(
        experiment,
        storage.load_trials(conn, experiment_id),
        storage.load_scores(conn, experiment_id),
    )


@app.get("/api/experiments/{experiment_id}/trials")
def list_trials(experiment_id: int, conn: Conn) -> dict:
    _load_or_404(conn, experiment_id)
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
            {**trial.model_dump(), "scores": scores_by_trial.get(trial.id, [])}
            for trial in trials
        ],
    }


# ---------------------------------------------------------------------------
# Background jobs: POST /api/runs returns at once; the run happens after.
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    agent: Literal["mock", "groq", "gemini"] = "mock"
    dataset: Literal["arithmetic", "mixed", "hard", "rag"] = "mixed"
    attempts: int = Field(default=3, ge=1)
    model: str | None = None
    name: str = "api-run"
    # Only for the arithmetic dataset: how many "What is N + 1" cases.
    cases: int = Field(default=100, ge=1)
    seed: int = 42
    # Only for the rag dataset.
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    embedding: Literal[EMBEDDINGS] = DEFAULT_EMBEDDING


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
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
    return job


def _set_job(job_id: str, **updates) -> None:
    with _jobs_lock:
        _jobs[job_id].update(updates)


def _run_args(request: RunRequest) -> argparse.Namespace:
    """The namespace run.py's agent builders expect, with the same
    per-agent defaults the CLI uses (run.py AGENT_DEFAULTS)."""
    from evalkit.run import AGENT_DEFAULTS

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
    from evalkit.models import dataset_version
    from evalkit.run import BUILDERS, load_dataset
    from evalkit.runner import run_experiment

    with _run_lock:
        _set_job(job_id, status="running")
        try:
            args = _run_args(request)
            if args.model not in PRICES:
                raise ValueError(f"No price for {args.model!r}. Add it to PRICES in evalkit/cost.py.")
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
            experiment_id = storage.save_experiment(
                storage.connect(config.database_path()), experiment, trials, scores,
            )
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
    from evalkit.run import AGENT_DEFAULTS

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
            detail=f"No price for {model!r}. Add it to PRICES in evalkit/cost.py.",
        )
    if request.agent == "groq" and not config.groq_api_key():
        raise HTTPException(status_code=400, detail="Set the GROQ_API_KEY environment variable first.")
    if request.agent == "gemini" and not config.gemini_api_key():
        raise HTTPException(status_code=400, detail="Set the GEMINI_API_KEY environment variable first.")
    # Checked now rather than failing in the background: the API image doesn't install chromadb.
    if request.dataset == "rag" and importlib.util.find_spec("chromadb") is None:
        raise HTTPException(status_code=400, detail='The rag dataset needs chromadb: pip install -e ".[rag]".')

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
    from evalkit.sample_data import dataset_by_name

    experiment = _load_or_404(conn, experiment_id)
    trials = storage.load_trials(conn, experiment_id)
    try:
        dataset = dataset_by_name(experiment.dataset_name)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"unknown dataset {experiment.dataset_name!r}; cannot re-score",
        )
    # Only the built-in scorers' scores are replaced; an LLM judge's are kept.
    storage.replace_scores(conn, experiment_id, score_trials(dataset, trials), SCORER_NAMES)
    return experiment_summary(experiment, trials, storage.load_scores(conn, experiment_id))
