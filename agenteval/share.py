"""Share experiments with your team: send them to an agenteval server everyone can open.

Each person runs evaluations on their own machine, with their own keys, into their own
agenteval.db. `agenteval push` copies an experiment, with its trials, scores and cases, to a
shared server (`uvicorn agenteval.api:app` on a machine with a persistent disk, or the
repository's Dockerfile), whose dashboard then shows everyone's runs side by side, ready to
compare.

    agenteval push               # the latest experiment
    agenteval push 12 --to https://evals.example.com

The server's address and token come from --to and --token, or AGENTEVAL_SERVER and
AGENTEVAL_TOKEN (the server's API_TOKEN), or a [push] section in agenteval.toml (keep the
token out of files you commit).
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from agenteval import storage
from agenteval.models import EvaluationCase, Experiment, Score, Trial
from agenteval.sample_data import cases_for

def payload(experiment: Experiment, trials: list[Trial], scores: list[Score],
            cases: list[EvaluationCase] | None) -> dict:
    """An experiment in the form POST /api/experiments accepts."""
    return {
        "experiment": experiment.model_dump(mode="json", exclude={"id"}),
        "trials": [trial.model_dump(mode="json") for trial in trials],
        "scores": [score.model_dump(mode="json") for score in scores],
        "cases": [case.model_dump(mode="json", exclude_none=True) for case in cases] if cases is not None else None,
    }

def push(conn, experiment_id: int, url: str, token: str, timeout: float = 60.0) -> dict:
    """Send one experiment; returns the server's summary of its copy (with the server's id)."""
    experiment = storage.load_experiment(conn, experiment_id)
    return send(url, token, payload(experiment, storage.load_trials(conn, experiment_id),
                                    storage.load_scores(conn, experiment_id), cases_for(conn, experiment)), timeout)

def send(url: str, token: str, body: dict, timeout: float = 60.0) -> dict:
    """POST an experiment payload to a server; raises RuntimeError saying why it was refused."""
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/experiments", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-API-Token": token, "User-Agent": "agenteval"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - the user's own server
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read()).get("detail", error.reason)
        except (json.JSONDecodeError, AttributeError):
            detail = error.reason
        raise RuntimeError(f"the server refused the experiment ({error.code}): {detail}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"can't reach {url}: {error.reason}") from None

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval push", description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment_id", type=int, nargs="?", help="default: the latest experiment")
    parser.add_argument("--db", default="agenteval.db")
    parser.add_argument("--to", dest="url", default=os.environ.get("AGENTEVAL_SERVER"),
                        help="the server's address (or AGENTEVAL_SERVER)")
    parser.add_argument("--token", default=os.environ.get("AGENTEVAL_TOKEN"),
                        help="the server's API_TOKEN (or AGENTEVAL_TOKEN)")
    args = parser.parse_args(argv)
    if not args.url:
        raise SystemExit("Say where to push: --to https://your-server, or set AGENTEVAL_SERVER.")
    if not args.token:
        raise SystemExit("The server needs its token: --token, or set AGENTEVAL_TOKEN.")

    conn = storage.connect(args.db)
    experiment_id = args.experiment_id or storage.latest_experiment_id(conn)
    if experiment_id is None:
        raise SystemExit(f"No experiments in {args.db} yet.")
    try:
        summary = push(conn, experiment_id, args.url, args.token)
    except (RuntimeError, LookupError) as error:
        raise SystemExit(str(error)) from None
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    print(f"Pushed experiment {experiment_id} as #{summary['id']} on {args.url.rstrip('/')}: "
          f"{args.url.rstrip('/')}/#/e/{summary['id']}")

if __name__ == "__main__":
    main()
