"""Tests for the Phase 1 HTTP layer (agenteval/api.py).

The API reads its database path and token from the environment on every
request (see agenteval/config.py), so each test points it at a fresh
temporary SQLite file with monkeypatch.setenv. Runs use the mock agent
over a 5-case arithmetic dataset: deterministic, no API keys, fast.
"""
import asyncio
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from agenteval import storage
from agenteval.api import app
from agenteval.cost import PRICES
from agenteval.mock_agent import FaultConfig, MockAgent
from agenteval.models import AgentConfig, Experiment
from agenteval.runner import run_experiment
from agenteval.sample_data import arithmetic_cases
from agenteval.scorers import score_trials

TOKEN = "test-token"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("API_TOKEN", TOKEN)
    # Off by default: tests that poll a job would otherwise trip it. One test turns it on.
    monkeypatch.setenv("REQUESTS_PER_CLIENT_PER_MINUTE", "0")
    with TestClient(app) as test_client:
        yield test_client


def seed_experiment(db_path, name="seeded"):
    """A small mock run saved straight to storage, like the CLI would."""
    dataset = arithmetic_cases(5)
    config = FaultConfig(tool_error_rate=0.0, hallucination_rate=0.0, latency_ms=(0, 0))
    trials = asyncio.run(run_experiment(MockAgent(config), dataset, attempts=1))
    scores = score_trials(dataset, trials)
    experiment = Experiment(
        name=name,
        dataset_name="arithmetic-5",
        config=AgentConfig(model="mock-llm", prompt="Answer arithmetic questions."),
        attempts=1,
        prices=PRICES,
        created_at=datetime(2026, 11, 7, tzinfo=timezone.utc),
    )
    conn = storage.connect(db_path)
    experiment_id = storage.save_experiment(conn, experiment, trials, scores)
    conn.close()
    return experiment_id


def auth_headers():
    return {"X-API-Token": TOKEN}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_experiments_empty_on_fresh_db(client):
    assert client.get("/api/experiments").json() == []


def test_experiments_lists_seeded_experiment(client, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    experiment_id = seed_experiment(tmp_path / "api.db")

    experiments = client.get("/api/experiments").json()
    assert len(experiments) == 1
    listed = experiments[0]
    assert listed["id"] == experiment_id
    assert listed["name"] == "seeded"
    assert listed["dataset_name"] == "arithmetic-5"
    assert listed["model"] == "mock-llm"
    assert listed["trial_count"] == 5
    assert listed["pass_rate"] == 1.0
    assert listed["latency_ms"]["p50"] >= 0
    assert listed["latency_ms"]["p95"] >= listed["latency_ms"]["p50"]
    assert listed["cost"] >= 0
    assert listed["tool_rates"]["right_tool"] == 1.0


def test_experiment_detail_and_missing(client, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    experiment_id = seed_experiment(tmp_path / "api.db")

    detail = client.get(f"/api/experiments/{experiment_id}").json()
    assert detail["name"] == "seeded"
    assert detail["pass_rate"] == 1.0
    assert detail["failures"] == {}

    missing = client.get("/api/experiments/999")
    assert missing.status_code == 404


def test_trials_include_spans_and_scores(client, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    experiment_id = seed_experiment(tmp_path / "api.db")

    body = client.get(f"/api/experiments/{experiment_id}/trials").json()
    assert body["experiment_id"] == experiment_id
    assert len(body["trials"]) == 5
    trial = body["trials"][0]
    assert trial["output"] == "2"
    assert trial["spans"], "trials must carry their spans"
    assert trial["spans"][0]["kind"] == "llm"
    scorers = {score["scorer"] for score in trial["scores"]}
    assert "exact_match" in scorers
    assert "tool_selection" in scorers

    assert client.get("/api/experiments/999/trials").status_code == 404


def test_run_requires_token(client):
    body = {"agent": "mock", "dataset": "arithmetic", "attempts": 1, "model": "mock-llm", "cases": 5}
    assert client.post("/api/runs", json=body).status_code == 401
    assert client.post("/api/runs", json=body, headers={"X-API-Token": "wrong"}).status_code == 401


def _wait_for_job(client, job_id, timeout_s=15.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = client.get(f"/api/runs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s (last: {job})")


def test_run_mock_completes_in_background(client):
    response = client.post(
        "/api/runs",
        json={"agent": "mock", "dataset": "arithmetic", "attempts": 1,
              "model": "mock-llm", "cases": 5, "name": "api-mock-run"},
        headers=auth_headers(),
    )
    # 202 at once: the request returns before the run finishes.
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert job["id"]

    finished = _wait_for_job(client, job["id"])
    assert finished["status"] == "completed", finished
    assert finished["experiment_id"] is not None

    experiments = client.get("/api/experiments").json()
    assert any(exp["id"] == finished["experiment_id"] and exp["name"] == "api-mock-run"
               for exp in experiments)


def test_run_rejects_mock_with_non_arithmetic_dataset(client):
    # Same rule as the CLI: the mock only answers arithmetic questions.
    response = client.post(
        "/api/runs",
        json={"agent": "mock", "dataset": "mixed", "attempts": 1},
        headers=auth_headers(),
    )
    assert response.status_code == 400


def test_run_rejects_unknown_model(client):
    response = client.post(
        "/api/runs",
        json={"agent": "mock", "dataset": "arithmetic", "attempts": 1, "model": "not-a-model"},
        headers=auth_headers(),
    )
    assert response.status_code == 400


def test_run_rejects_real_agent_without_key(client, monkeypatch):
    # Without GROQ_API_KEY, groq must refuse before spending anything. Removed
    # explicitly, since a developer's shell may have a real key set.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.post(
        "/api/runs",
        json={"agent": "groq", "dataset": "mixed", "attempts": 1},
        headers=auth_headers(),
    )
    assert response.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_rescore_requires_token_and_recomputes(client, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    experiment_id = seed_experiment(tmp_path / "api.db")

    assert client.post(f"/api/experiments/{experiment_id}/rescore").status_code == 401
    response = client.post(
        f"/api/experiments/{experiment_id}/rescore", headers=auth_headers(),
    )
    assert response.status_code == 200
    assert response.json()["pass_rate"] == 1.0
    assert response.json()["id"] == experiment_id

    assert client.post("/api/experiments/999/rescore", headers=auth_headers()).status_code == 404


# --- Limits that keep the public API usable ---


@pytest.mark.parametrize("field, value", [("attempts", 11), ("cases", 1001), ("name", "x" * 81)])
def test_runs_are_bounded(client, field, value):
    body = {"agent": "mock", "dataset": "arithmetic", "attempts": 1, field: value}
    assert client.post("/api/runs", json=body, headers=auth_headers()).status_code == 422


def test_too_many_waiting_runs_are_refused(client, monkeypatch):
    from agenteval import api

    monkeypatch.setattr(api, "_jobs", {str(n): {"status": "queued"} for n in range(api.MAX_WAITING_JOBS)})
    response = client.post("/api/runs", json={"agent": "mock", "dataset": "arithmetic", "attempts": 1},
                           headers=auth_headers())
    assert response.status_code == 429


def test_finished_jobs_are_pruned_oldest_first(monkeypatch):
    from agenteval import api

    finished = {f"old-{n}": {"status": "completed"} for n in range(api.MAX_KEPT_JOBS)}
    monkeypatch.setattr(api, "_jobs", {**finished, "running": {"status": "running"}})

    api._new_job(api.RunRequest(agent="mock", dataset="arithmetic"))

    assert len(api._jobs) == api.MAX_KEPT_JOBS
    assert "old-0" not in api._jobs and "running" in api._jobs  # unfinished jobs are never dropped


def test_experiments_are_listed_a_page_at_a_time(client, tmp_path):
    for name in ("first", "second", "third"):
        seed_experiment(tmp_path / "api.db", name=name)

    assert [e["name"] for e in client.get("/api/experiments?limit=2").json()] == ["third", "second"]
    assert [e["name"] for e in client.get("/api/experiments?limit=2&offset=2").json()] == ["first"]
    assert client.get("/api/experiments?limit=101").status_code == 422


def test_a_cached_summary_is_rebuilt_when_scores_change(client, tmp_path):
    experiment_id = seed_experiment(tmp_path / "api.db")
    before = client.get(f"/api/experiments/{experiment_id}").json()

    conn = storage.connect(tmp_path / "api.db")
    with conn:  # what agenteval.rescore would do from another process
        conn.execute("UPDATE scores SET value = 0 WHERE experiment_id = ? AND scorer = 'exact_match'",
                     (experiment_id,))

    after = client.get(f"/api/experiments/{experiment_id}").json()
    assert before["pass_rate"] == 1.0 and after["pass_rate"] == 0.0


def test_clients_over_the_limit_get_429_with_cors_headers(client, monkeypatch):
    from agenteval import api

    monkeypatch.setenv("REQUESTS_PER_CLIENT_PER_MINUTE", "2")
    monkeypatch.setattr(api, "_client_limiter", api.ClientLimiter())
    origin = {"Origin": "http://localhost:5173"}

    assert client.get("/api/experiments", headers=origin).status_code == 200
    assert client.get("/api/experiments", headers=origin).status_code == 200
    limited = client.get("/api/experiments", headers=origin)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) > 0
    assert limited.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert client.get("/health").status_code == 200  # the health check is never limited


def test_the_limit_is_per_client_and_uses_the_proxys_address():
    from types import SimpleNamespace

    from agenteval.api import ClientLimiter, client_address

    clock = SimpleNamespace(now=0.0)
    limiter = ClientLimiter(clock=lambda: clock.now)
    assert limiter.retry_after("a", 1) is None
    assert limiter.retry_after("a", 1) == 60
    assert limiter.retry_after("b", 1) is None  # another client has its own budget
    clock.now = 60
    assert limiter.retry_after("a", 1) is None  # the minute has passed

    # The client can put anything first; the last entry is the one Render's proxy added.
    spoofed = SimpleNamespace(headers={"x-forwarded-for": "1.2.3.4, 203.0.113.9"}, client=None)
    assert client_address(spoofed) == "203.0.113.9"


def test_a_request_connection_works_from_another_thread(tmp_path, monkeypatch):
    # FastAPI may open get_conn, run the endpoint and close the connection on different
    # worker threads. With SQLite's default same-thread check that failed at random with 500s.
    import threading

    from agenteval.api import get_conn

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "threads.db"))
    dependency = get_conn()
    conn = next(dependency)  # opened on this thread
    errors = []

    def use_and_close():
        try:
            conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
            dependency.close()  # runs the dependency's cleanup: conn.close()
        except Exception as error:  # noqa: BLE001 - reported below
            errors.append(error)

    worker = threading.Thread(target=use_and_close)
    worker.start()
    worker.join()
    assert errors == []


# --- 1.0: questions with trials, compare, manage, labels, read-only, the built dashboard ---


def test_trials_carry_the_question_they_answered(client, tmp_path):
    experiment_id = seed_experiment(tmp_path / "api.db")
    trial = client.get(f"/api/experiments/{experiment_id}/trials").json()["trials"][0]
    assert trial["case"] == {"input": "What is 1 + 1", "expected_output": "2", "category": "general"}


def test_summaries_have_status_categories_and_other_scores(client, tmp_path):
    experiment_id = seed_experiment(tmp_path / "api.db")
    summary = client.get(f"/api/experiments/{experiment_id}").json()
    assert summary["status"] == "completed"
    assert summary["by_category"] == {"general": {"pass_rate": 1.0, "trials": 5}}
    assert summary["other_scores"] == {}


def test_compare_two_experiments(client, tmp_path):
    a, b = seed_experiment(tmp_path / "api.db", "a"), seed_experiment(tmp_path / "api.db", "b")
    result = client.get(f"/api/compare?a={a}&b={b}").json()
    assert result["verdict"] == "Inconclusive" and result["cases_compared"] == 5
    assert "the configs are identical: this measures noise, not a change" in result["warnings"]
    assert client.get(f"/api/compare?a={a}&b=999").status_code == 404


def test_rename_and_delete_need_the_token(client, tmp_path):
    experiment_id = seed_experiment(tmp_path / "api.db")
    assert client.patch(f"/api/experiments/{experiment_id}", json={"name": "x"}).status_code == 401
    renamed = client.patch(f"/api/experiments/{experiment_id}", json={"name": "renamed"}, headers=auth_headers())
    assert renamed.json()["name"] == "renamed"
    assert client.get("/api/experiments").json()[0]["name"] == "renamed"  # the cached summary was dropped
    assert client.delete(f"/api/experiments/{experiment_id}", headers=auth_headers()).status_code == 204
    assert client.get(f"/api/experiments/{experiment_id}").status_code == 404
    assert client.delete(f"/api/experiments/{experiment_id}", headers=auth_headers()).status_code == 404


def test_labelling_in_the_browser_and_agreement(client, tmp_path):
    experiment_id = seed_experiment(tmp_path / "api.db")
    conn = storage.connect(tmp_path / "api.db")
    from agenteval.models import Score
    storage.replace_scores(conn, experiment_id, [Score(trial_id=t, scorer="judge_correctness", value=1.0,
                                                       reasoning="judge x: same") for t in range(1, 6)],
                           ["judge_correctness"])

    queue = client.get(f"/api/experiments/{experiment_id}/labels?rubric=correctness").json()
    assert queue["labelled"] == 0 and queue["judged"] == 5 and queue["agreement"] is None
    first = queue["next"][0]
    assert first["question"].startswith("What is") and first["expected"]

    for item in queue["next"]:
        response = client.post(f"/api/experiments/{experiment_id}/labels", headers=auth_headers(),
                               json={"trial_id": item["trial_id"], "rubric": "correctness", "passed": True})
        assert response.status_code == 200
    done = client.get(f"/api/experiments/{experiment_id}/labels?rubric=correctness").json()
    assert done["labelled"] == 5 and done["next"] == []
    assert done["agreement"]["agreement"] == 1.0 and "label more" in done["agreement"]["reading"]
    assert client.post(f"/api/experiments/{experiment_id}/labels", headers=auth_headers(),
                       json={"trial_id": 99, "rubric": "correctness", "passed": True}).status_code == 404
    assert client.get(f"/api/experiments/{experiment_id}/labels?rubric=nope").status_code == 400
    assert {r["name"] for r in client.get("/api/rubrics").json()} >= {"groundedness", "correctness"}


def test_read_only_mode_refuses_every_change(client, tmp_path, monkeypatch):
    experiment_id = seed_experiment(tmp_path / "api.db")
    monkeypatch.setenv("READ_ONLY", "1")
    assert client.delete(f"/api/experiments/{experiment_id}", headers=auth_headers()).status_code == 403
    assert client.post("/api/runs", json={"agent": "mock", "dataset": "arithmetic"},
                       headers=auth_headers()).status_code == 403
    assert client.get(f"/api/experiments/{experiment_id}").status_code == 200


def test_the_built_dashboard_is_served_at_the_root(tmp_path, monkeypatch):
    import importlib

    import agenteval.api

    (tmp_path / "index.html").write_text("<!doctype html><title>agenteval</title>")
    monkeypatch.setenv("AGENTEVAL_STATIC_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "s.db"))
    reloaded = importlib.reload(agenteval.api)
    try:
        with TestClient(reloaded.app) as static_client:
            assert "agenteval" in static_client.get("/").text
            assert static_client.get("/health").json() == {"status": "ok"}  # API routes still win
            assert static_client.get("/api/experiments").json() == []
    finally:
        monkeypatch.delenv("AGENTEVAL_STATIC_DIR")
        importlib.reload(agenteval.api)


def test_trends_list_completed_runs_of_a_dataset_and_model_oldest_first(client, tmp_path):
    for name in ("first", "second"):
        seed_experiment(tmp_path / "api.db", name=name)
    points = client.get("/api/trends?dataset=arithmetic-5&model=mock-llm").json()
    assert [p["name"] for p in points] == ["first", "second"]
    assert points[0]["pass_rate"] == 1.0 and "latency_ms" in points[0]
    assert client.get("/api/trends?dataset=hard").json() == []


def test_demo_seed_fills_an_empty_database_only(tmp_path, monkeypatch):
    from agenteval.demo import seed_if_empty

    path = str(tmp_path / "demo.db")
    ids = seed_if_empty(path)
    assert len(ids) == 3
    assert seed_if_empty(path) == []  # not twice
    conn = storage.connect(path)
    names = [storage.load_experiment(conn, i).name for i in ids]
    assert names == ["baseline", "fewer-hallucinations", "flaky-tools"]
    assert all(storage.load_experiment(conn, i).complete for i in ids)


def test_the_server_seeds_the_demo_on_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "startup.db"))
    monkeypatch.setenv("DEMO_SEED", "1")
    with TestClient(app) as demo_client:
        assert len(demo_client.get("/api/experiments").json()) == 3
