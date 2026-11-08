"""Where a run happened, saved with it so a result can be traced back to what produced it."""
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

# Packages whose version can change an agent's behaviour or how it's measured.
TRACKED_PACKAGES = ("pyagenteval", "openai", "google-genai", "chromadb", "pydantic")

def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(["git", *arguments], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()

def capture() -> dict[str, Any]:
    """Python, OS, package versions, and the git commit of the current directory if it is a
    repository, with whether it had uncommitted changes (then the commit alone doesn't say
    what ran)."""
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue

    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "packages": packages,
        "command": " ".join(sys.argv),
    }
    commit = _git("rev-parse", "--short", "HEAD")
    if commit:
        environment["git_commit"] = commit
        environment["git_dirty"] = bool(_git("status", "--porcelain"))
    return environment

def describe(environment: dict[str, Any]) -> str:
    """One line for reports: "commit 2be961b (uncommitted changes), Python 3.12.4"."""
    parts = []
    if "git_commit" in environment:
        parts.append(f"commit {environment['git_commit']}" + (" (uncommitted changes)" if environment.get("git_dirty") else ""))
    if "python" in environment:
        parts.append(f"Python {environment['python']}")
    version = environment.get("packages", {}).get("pyagenteval")
    if version:
        parts.append(f"agenteval {version}")
    return ", ".join(parts)
