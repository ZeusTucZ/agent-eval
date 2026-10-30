import json
import sqlite3
from datetime import datetime
from pathlib import Path

from evalkit.models import AgentConfig, Experiment, Price, Score, Span, Trial

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    dataset_name   TEXT NOT NULL,
    config_json    TEXT NOT NULL,
    config_version TEXT NOT NULL,
    attempts       INTEGER NOT NULL,
    prices_json    TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

-- Trial ids restart at 1 in every experiment, so the key is (experiment_id, trial_id).
CREATE TABLE IF NOT EXISTS trials (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    trial_id      INTEGER NOT NULL,
    case_id       INTEGER NOT NULL,
    attempt       INTEGER NOT NULL,
    output        TEXT,
    failure_type  TEXT,
    PRIMARY KEY (experiment_id, trial_id)
);

CREATE TABLE IF NOT EXISTS spans (
    experiment_id INTEGER NOT NULL,
    trial_id      INTEGER NOT NULL,
    position      INTEGER NOT NULL,  -- keeps spans in the order they happened
    kind          TEXT NOT NULL,
    name          TEXT NOT NULL,
    start_ns      INTEGER NOT NULL,
    end_ns        INTEGER NOT NULL,
    status        TEXT NOT NULL,
    error_type    TEXT,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    retry         INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    PRIMARY KEY (experiment_id, trial_id, position),
    FOREIGN KEY (experiment_id, trial_id) REFERENCES trials(experiment_id, trial_id)
);

CREATE TABLE IF NOT EXISTS scores (
    experiment_id INTEGER NOT NULL,
    trial_id      INTEGER NOT NULL,
    scorer        TEXT NOT NULL,
    value         REAL NOT NULL,
    reasoning     TEXT,
    PRIMARY KEY (experiment_id, trial_id, scorer),
    FOREIGN KEY (experiment_id, trial_id) REFERENCES trials(experiment_id, trial_id)
);
"""

def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn

def _migrate(conn: sqlite3.Connection) -> None:
    """Bring databases created by older versions up to date. CREATE TABLE IF NOT EXISTS skips existing tables."""
    span_columns = {row["name"] for row in conn.execute("PRAGMA table_info(spans)")}
    added_columns = {  # column -> definition, in the order they were added
        "retry": "INTEGER NOT NULL DEFAULT 0",
        "error_message": "TEXT",
    }
    with conn:
        for column, definition in added_columns.items():
            if column not in span_columns:
                conn.execute(f"ALTER TABLE spans ADD COLUMN {column} {definition}")

def save_experiment(
    conn: sqlite3.Connection,
    experiment: Experiment,
    trials: list[Trial],
    scores: list[Score],
) -> int:
    """Save everything in one transaction: a crash halfway leaves nothing behind, not half an experiment."""
    with conn:
        cursor = conn.execute(
            "INSERT INTO experiments (name, dataset_name, config_json, config_version, attempts, prices_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                experiment.name,
                experiment.dataset_name,
                experiment.config.model_dump_json(),
                experiment.config.version,
                experiment.attempts,
                json.dumps({model: price.model_dump() for model, price in experiment.prices.items()}),
                experiment.created_at.isoformat(),
            ),
        )
        experiment_id = cursor.lastrowid

        conn.executemany(
            "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?)",
            [(experiment_id, t.id, t.case_id, t.attempt, t.output, t.failure_type) for t in trials],
        )
        conn.executemany(
            "INSERT INTO spans (experiment_id, trial_id, position, kind, name, start_ns, end_ns,"
            " status, error_type, input_tokens, output_tokens, retry, error_message)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (experiment_id, t.id, position, s.kind, s.name, s.start, s.end,
                 s.status, s.error_type, s.input_tokens, s.output_tokens, s.retry, s.error_message)
                for t in trials
                for position, s in enumerate(t.spans)
            ],
        )
        conn.executemany(
            "INSERT INTO scores VALUES (?, ?, ?, ?, ?)",
            [(experiment_id, s.trial_id, s.scorer, s.value, s.reasoning) for s in scores],
        )

    return experiment_id

def replace_scores(conn: sqlite3.Connection, experiment_id: int, scores: list[Score]) -> None:
    """Swap in new scores for stored trials, e.g. after fixing a scorer. The trials stay as they were."""
    with conn:
        conn.execute("DELETE FROM scores WHERE experiment_id = ?", (experiment_id,))
        conn.executemany(
            "INSERT INTO scores VALUES (?, ?, ?, ?, ?)",
            [(experiment_id, s.trial_id, s.scorer, s.value, s.reasoning) for s in scores],
        )

def load_experiment(conn: sqlite3.Connection, experiment_id: int) -> Experiment:
    row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None:
        raise LookupError(f"no experiment with id {experiment_id}")

    return Experiment(
        id=row["id"],
        name=row["name"],
        dataset_name=row["dataset_name"],
        config=AgentConfig.model_validate_json(row["config_json"]),
        attempts=row["attempts"],
        prices={model: Price(**price) for model, price in json.loads(row["prices_json"]).items()},
        created_at=datetime.fromisoformat(row["created_at"]),
    )

def load_trials(conn: sqlite3.Connection, experiment_id: int) -> list[Trial]:
    spans_by_trial: dict[int, list[Span]] = {}
    for row in conn.execute(
        "SELECT * FROM spans WHERE experiment_id = ? ORDER BY trial_id, position", (experiment_id,)
    ):
        spans_by_trial.setdefault(row["trial_id"], []).append(Span(
            trial_id=row["trial_id"],
            kind=row["kind"],
            name=row["name"],
            start=row["start_ns"],
            end=row["end_ns"],
            status=row["status"],
            error_type=row["error_type"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            retry=row["retry"],
            error_message=row["error_message"],
        ))

    return [
        Trial(
            id=row["trial_id"],
            case_id=row["case_id"],
            attempt=row["attempt"],
            output=row["output"],
            failure_type=row["failure_type"],
            spans=spans_by_trial.get(row["trial_id"], []),
        )
        for row in conn.execute(
            "SELECT * FROM trials WHERE experiment_id = ? ORDER BY trial_id", (experiment_id,)
        )
    ]

def load_scores(conn: sqlite3.Connection, experiment_id: int) -> list[Score]:
    return [
        Score(trial_id=row["trial_id"], scorer=row["scorer"], value=row["value"], reasoning=row["reasoning"])
        for row in conn.execute(
            "SELECT * FROM scores WHERE experiment_id = ? ORDER BY trial_id, scorer", (experiment_id,)
        )
    ]
