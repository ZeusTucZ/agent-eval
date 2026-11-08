import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from agenteval.models import AgentConfig, Experiment, Price, Score, Span, Trial

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    dataset_name   TEXT NOT NULL,
    dataset_version TEXT,
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
    input_text    TEXT,
    output_text   TEXT,
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

-- A person's verdict on an answer, for checking an LLM judge (agenteval calibrate).
CREATE TABLE IF NOT EXISTS labels (
    experiment_id INTEGER NOT NULL,
    trial_id      INTEGER NOT NULL,
    rubric        TEXT NOT NULL,
    label         TEXT NOT NULL CHECK (label IN ('pass', 'fail')),
    created_at    TEXT NOT NULL,
    PRIMARY KEY (experiment_id, trial_id, rubric),
    FOREIGN KEY (experiment_id, trial_id) REFERENCES trials(experiment_id, trial_id)
);
"""

def connect(path: str | Path, check_same_thread: bool = True) -> sqlite3.Connection:
    """check_same_thread=False lets a connection opened in one thread be used in another, for a
    caller that makes sure only one thread uses it at a time (see api.get_conn)."""
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn

def _migrate(conn: sqlite3.Connection) -> None:
    """Bring databases created by older versions up to date. CREATE TABLE IF NOT EXISTS skips existing tables."""
    added_columns = {  # table -> {column: definition}, in the order they were added
        "spans": {
            "retry": "INTEGER NOT NULL DEFAULT 0",
            "error_message": "TEXT",
            "input_text": "TEXT",
            "output_text": "TEXT",
        },
        "experiments": {
            "dataset_version": "TEXT",
            # Experiments from before incremental saving were only ever saved complete.
            "status": "TEXT NOT NULL DEFAULT 'completed'",
            "planned_trials": "INTEGER",
            "environment_json": "TEXT",
            "custom_scorers_json": "TEXT",
        },
    }
    with conn:
        for table, columns in added_columns.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column, definition in columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

def start_experiment(conn: sqlite3.Connection, experiment: Experiment) -> int:
    """Record an experiment before its trials run, so each trial can be saved as it finishes:
    a run cut short keeps everything already paid for, and can be resumed."""
    with conn:
        return _insert_experiment(conn, experiment)

def _insert_experiment(conn: sqlite3.Connection, experiment: Experiment) -> int:
    cursor = conn.execute(
        "INSERT INTO experiments (name, dataset_name, dataset_version, config_json, config_version,"
        " attempts, prices_json, created_at, status, planned_trials, environment_json, custom_scorers_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            experiment.name,
            experiment.dataset_name,
            experiment.dataset_version,
            experiment.config.model_dump_json(),
            experiment.config.version,
            experiment.attempts,
            _prices_json(experiment),
            experiment.created_at.isoformat(),
            experiment.status,
            experiment.planned_trials,
            json.dumps(experiment.environment),
            json.dumps(experiment.custom_scorers),
        ),
    )
    return cursor.lastrowid

def _prices_json(experiment: Experiment) -> str:
    return json.dumps({model: price.model_dump() for model, price in experiment.prices.items()})

def save_trials(conn: sqlite3.Connection, experiment_id: int, trials: list[Trial], scores: list[Score]) -> None:
    """Add finished trials and their scores, in one transaction."""
    with conn:
        _insert_trials(conn, experiment_id, trials, scores)

def _insert_trials(conn: sqlite3.Connection, experiment_id: int, trials: list[Trial], scores: list[Score]) -> None:
    conn.executemany(
        "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?)",
        [(experiment_id, t.id, t.case_id, t.attempt, t.output, t.failure_type) for t in trials],
    )
    conn.executemany(
        "INSERT INTO spans (experiment_id, trial_id, position, kind, name, start_ns, end_ns,"
        " status, error_type, input_tokens, output_tokens, retry, error_message, input_text, output_text)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (experiment_id, t.id, position, s.kind, s.name, s.start, s.end,
             s.status, s.error_type, s.input_tokens, s.output_tokens, s.retry, s.error_message, s.input, s.output)
            for t in trials
            for position, s in enumerate(t.spans)
        ],
    )
    conn.executemany(
        "INSERT INTO scores VALUES (?, ?, ?, ?, ?)",
        [(experiment_id, s.trial_id, s.scorer, s.value, s.reasoning) for s in scores],
    )

def finish_experiment(conn: sqlite3.Connection, experiment_id: int, experiment: Experiment) -> None:
    """Set the final status, and the prices, which may have gained models the run discovered."""
    with conn:
        conn.execute("UPDATE experiments SET status = ?, prices_json = ? WHERE id = ?",
                     (experiment.status, _prices_json(experiment), experiment_id))

def save_experiment(
    conn: sqlite3.Connection,
    experiment: Experiment,
    trials: list[Trial],
    scores: list[Score],
) -> int:
    """Save a finished experiment whole, in one transaction: a crash halfway leaves nothing behind."""
    with conn:  # sqlite3 transactions don't nest, so this uses the inserts without their own
        experiment_id = _insert_experiment(conn, experiment)
        _insert_trials(conn, experiment_id, trials, scores)
    return experiment_id

def delete_experiment(conn: sqlite3.Connection, experiment_id: int) -> None:
    with conn:
        for table in ("labels", "scores", "spans", "trials"):
            conn.execute(f"DELETE FROM {table} WHERE experiment_id = ?", (experiment_id,))
        if conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,)).rowcount == 0:
            raise LookupError(f"no experiment with id {experiment_id}")

def rename_experiment(conn: sqlite3.Connection, experiment_id: int, name: str) -> None:
    with conn:
        if conn.execute("UPDATE experiments SET name = ? WHERE id = ?", (name, experiment_id)).rowcount == 0:
            raise LookupError(f"no experiment with id {experiment_id}")

def list_experiment_ids(conn: sqlite3.Connection) -> list[int]:
    return [row[0] for row in conn.execute("SELECT id FROM experiments ORDER BY id")]

def save_label(conn: sqlite3.Connection, experiment_id: int, trial_id: int, rubric: str, passed: bool) -> None:
    """A person's verdict; labelling the same answer again replaces it."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?, ?)",
            (experiment_id, trial_id, rubric, "pass" if passed else "fail", datetime.now(UTC).isoformat()),
        )

def load_labels(conn: sqlite3.Connection, experiment_id: int, rubric: str) -> dict[int, bool]:
    """trial id -> True for pass."""
    return {
        row["trial_id"]: row["label"] == "pass"
        for row in conn.execute("SELECT trial_id, label FROM labels WHERE experiment_id = ? AND rubric = ?",
                                (experiment_id, rubric))
    }

def replace_scores(
    conn: sqlite3.Connection,
    experiment_id: int,
    scores: list[Score],
    scorers: Iterable[str] | None = None,
) -> None:
    """Swap in new scores for stored trials, e.g. after fixing a scorer. The trials stay as they were.

    With `scorers`, only scores by those names are replaced and the rest are kept: re-running
    the cheap scorers must not throw away an LLM judge's scores, which cost money to make.
    """
    with conn:
        if scorers is None:
            conn.execute("DELETE FROM scores WHERE experiment_id = ?", (experiment_id,))
        else:
            names = list(scorers)
            assert {score.scorer for score in scores} <= set(names), "a score outside `scorers` would be duplicated"
            conn.executemany("DELETE FROM scores WHERE experiment_id = ? AND scorer = ?",
                             [(experiment_id, name) for name in names])
        conn.executemany(
            "INSERT INTO scores VALUES (?, ?, ?, ?, ?)",
            [(experiment_id, s.trial_id, s.scorer, s.value, s.reasoning) for s in scores],
        )

def latest_experiment_id(conn: sqlite3.Connection) -> int | None:
    return conn.execute("SELECT MAX(id) FROM experiments").fetchone()[0]

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
        dataset_version=row["dataset_version"],
        status=row["status"],
        planned_trials=row["planned_trials"],
        environment=json.loads(row["environment_json"] or "{}"),
        custom_scorers=json.loads(row["custom_scorers_json"] or "[]"),
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
            input=row["input_text"],
            output=row["output_text"],
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
