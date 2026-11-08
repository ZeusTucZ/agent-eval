"""agenteval ui: the dashboard for your local results, in your browser.

Serves the API and the built dashboard together on 127.0.0.1, so nothing is reachable from
other machines. A token for this session is made up and handed to the browser in the URL,
so starting runs, re-scoring and labelling work without typing one.
"""
import argparse
import os
import secrets
import threading
import webbrowser
from pathlib import Path

def find_dashboard() -> Path | None:
    """The dashboard built into the package, or a development build in the repository."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "static", here.parent / "dashboard" / "dist"):
        if (candidate / "index.html").is_file():
            return candidate
    return None

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval ui", description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", default="agenteval.db")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps it on this machine")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--read-only", action="store_true", help="refuse runs, re-scoring, labels and deletes")
    args = parser.parse_args(argv)

    try:
        import fastapi  # noqa: F401 - only checking it's installed
        import uvicorn
    except ImportError:
        raise SystemExit('The dashboard needs the server extra: pip install "pyagenteval[server]"') from None
    dashboard = find_dashboard()
    if dashboard is None:
        raise SystemExit("The dashboard isn't built. In a clone of the repository:\n"
                         "  cd dashboard && npm ci && npm run build")

    os.environ["DATABASE_PATH"] = str(Path(args.db).resolve())
    os.environ["AGENTEVAL_STATIC_DIR"] = str(dashboard)
    os.environ.setdefault("REQUESTS_PER_CLIENT_PER_MINUTE", "0")  # it's only you
    if args.read_only:
        os.environ["READ_ONLY"] = "1"
    token = os.environ.get("API_TOKEN") or secrets.token_urlsafe(16)
    os.environ["API_TOKEN"] = token

    url = f"http://{args.host}:{args.port}/"
    print(f"agenteval dashboard for {args.db} at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        # The token travels in the fragment, which browsers never send to a server.
        threading.Timer(1.0, webbrowser.open, [f"{url}#token={token}"]).start()
    uvicorn.run("agenteval.api:app", host=args.host, port=args.port, log_level="warning")
