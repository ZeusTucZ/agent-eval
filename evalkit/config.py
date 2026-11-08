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
    return os.environ.get("DATABASE_PATH", "evalkit.db")


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


def port() -> int:
    """Local default 8000; Render injects its own PORT."""
    return int(os.environ.get("PORT", "8000"))
