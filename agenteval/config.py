"""Environment configuration, read in one place.

Every value comes from the environment and is read lazily (per call),
so a process that starts before its variables are set never freezes a
stale value, and tests can change them with ``monkeypatch.setenv``.
Nothing here holds a real secret: the defaults are empty strings, and
``.env.example`` only lists placeholder names.
"""
import os


def database_path() -> str:
    """Where the SQLite file lives. Render mounts its persistent disk at /data."""
    return os.environ.get("DATABASE_PATH", "agenteval.db")


def api_token() -> str:
    """Shared secret for POST /api/runs and POST .../rescore (header X-API-Token)."""
    return os.environ.get("API_TOKEN", "")


def groq_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")


def gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "")


def cors_origins() -> list[str]:
    """Comma-separated list of origins allowed to call the API from a browser."""
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def requests_per_client_per_minute() -> int:
    """How many requests one client may make per minute; 0 turns the limit off."""
    return int(os.environ.get("REQUESTS_PER_CLIENT_PER_MINUTE", "120"))


def project_agent() -> dict | None:
    """The agent the dashboard may run besides the built-in ones: only the one configured in
    the project's agenteval.toml, which `agenteval ui` passes here. The API never runs code
    named in a request."""
    import json

    raw = os.environ.get("AGENTEVAL_PROJECT_AGENT")
    if not raw:
        return None
    try:
        project = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return project if isinstance(project, dict) and ":" in str(project.get("agent", "")) else None


def demo_seed() -> bool:
    """DEMO_SEED=1 fills an empty database with example experiments when the server starts."""
    return os.environ.get("DEMO_SEED", "").lower() in ("1", "true", "yes")


def read_only() -> bool:
    """READ_ONLY=1 refuses every change (runs, re-scoring, labels, deletes): for a public demo."""
    return os.environ.get("READ_ONLY", "").lower() in ("1", "true", "yes")


def static_dir() -> str | None:
    """The built dashboard to serve at /: AGENTEVAL_STATIC_DIR, or the copy built into the
    package (agenteval/static), if there is one."""
    from pathlib import Path

    configured = os.environ.get("AGENTEVAL_STATIC_DIR")
    candidate = Path(configured) if configured else Path(__file__).resolve().parent / "static"
    return str(candidate) if (candidate / "index.html").is_file() else None


def port() -> int:
    """Local default 8000; Render injects its own PORT."""
    return int(os.environ.get("PORT", "8000"))
