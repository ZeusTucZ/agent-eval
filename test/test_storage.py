import asyncio
import sqlite3
from datetime import datetime, timezone

import pytest

from evalkit import storage
from evalkit.cost import PRICES
from evalkit.mock_agent import FaultConfig, MockAgent
from evalkit.models import AgentConfig, Experiment, Score
from evalkit.report import summarize
from evalkit.runner import run_experiment
from evalkit.sample_data import arithmetic_cases
from evalkit.scorers import exact_match

def make_experiment(name: str = "test") -> Experiment:
    return Experiment(
        name=name,
        dataset_name="arithmetic-20",
        config=AgentConfig(model="mock-llm", prompt="Answer arithmetic questions."),
        attempts=3,
        prices=PRICES,
        created_at=datetime(2026, 10, 29, tzinfo=timezone.utc),
    )

def run_and_score():
    dataset = arithmetic_cases(20)
    config = FaultConfig(tool_error_rate=0.2, hallucination_rate=0.2, latency_ms=(0, 0))
    trials = asyncio.run(run_experiment(MockAgent(config), dataset, attempts=3))
    cases = {case.id: case for case in dataset}
    return trials, [exact_match(cases[trial.case_id], trial) for trial in trials]

def test_saved_experiment_loads_back_identical(tmp_path):
    trials, scores = run_and_score()
    experiment = make_experiment()

    conn = storage.connect(tmp_path / "eval.db")
    experiment_id = storage.save_experiment(conn, experiment, trials, scores)

    assert storage.load_experiment(conn, experiment_id) == experiment.model_copy(update={"id": experiment_id})
    assert storage.load_trials(conn, experiment_id) == trials
    assert storage.load_scores(conn, experiment_id) == scores

def test_results_survive_reconnecting(tmp_path):
    # The Phase 1 milestone: run, close, and read the results later.
    trials, scores = run_and_score()
    conn = storage.connect(tmp_path / "eval.db")
    experiment_id = storage.save_experiment(conn, make_experiment(), trials, scores)
    conn.close()

    reopened = storage.connect(tmp_path / "eval.db")
    report = summarize(
        storage.load_experiment(reopened, experiment_id),
        storage.load_trials(reopened, experiment_id),
        storage.load_scores(reopened, experiment_id),
    )

    assert f"Experiment {experiment_id}: test" in report
    assert "60 trials" in report

def test_experiments_are_kept_apart(tmp_path):
    trials, scores = run_and_score()
    conn = storage.connect(tmp_path / "eval.db")

    first = storage.save_experiment(conn, make_experiment("first"), trials, scores)
    second = storage.save_experiment(conn, make_experiment("second"), trials[:6], scores[:6])

    assert len(storage.load_trials(conn, first)) == 60
    assert len(storage.load_trials(conn, second)) == 6

def test_failed_save_leaves_nothing_behind(tmp_path):
    trials, scores = run_and_score()
    duplicate_score = Score(trial_id=scores[0].trial_id, scorer=scores[0].scorer, value=0.0)

    conn = storage.connect(tmp_path / "eval.db")
    with pytest.raises(sqlite3.IntegrityError):
        storage.save_experiment(conn, make_experiment(), trials, [*scores, duplicate_score])

    assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 0

def test_databases_from_phase_1_are_migrated(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    phase_1_schema = (
        storage.SCHEMA
        .replace("    retry         INTEGER NOT NULL DEFAULT 0,\n", "")
        .replace("    error_message TEXT,\n", "")
    )
    assert "retry " not in phase_1_schema and "error_message" not in phase_1_schema
    old.executescript(phase_1_schema)
    old.close()

    conn = storage.connect(path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(spans)")}
    assert {"retry", "error_message"} <= columns

    trials, scores = run_and_score()
    experiment_id = storage.save_experiment(conn, make_experiment(), trials, scores)
    assert storage.load_trials(conn, experiment_id) == trials

def test_missing_experiment_raises(tmp_path):
    conn = storage.connect(tmp_path / "eval.db")
    with pytest.raises(LookupError):
        storage.load_experiment(conn, 999)
