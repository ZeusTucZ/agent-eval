"""The `agenteval` command: one entry point for every tool.

    agenteval init                 set up a project: an agent, a few cases, a config
    agenteval run                  run an agent over a dataset
    agenteval report [ID]          what happened, and why trials failed
    agenteval ui                   the dashboard, in your browser

Options can live in agenteval.toml in the current directory, so a project's runs don't
need long command lines. Each table is a command's options, and `db` at the top applies to
every command; anything on the command line wins:

    db = "results/agenteval.db"

    [run]
    agent = "my_agents:answer"
    model = "gpt-4o-mini"
    price = [0.15, 0.60]
    dataset = "cases.jsonl"
"""
import importlib
import sys
import tomllib
from pathlib import Path
from typing import Any

CONFIG_FILE = "agenteval.toml"

# command -> (module with a main(argv), one-line description). Imported only when used,
# so `agenteval report` doesn't load what `agenteval run` needs.
COMMANDS: dict[str, tuple[str, str]] = {
    "init": ("agenteval.scaffold", "set up a project: an example agent, cases and agenteval.toml"),
    "run": ("agenteval.run", "run an agent over a dataset and save every step"),
    "report": ("agenteval.report", "show an experiment: pass rate, failures, cost (text, json, markdown, html, csv)"),
    "list": ("agenteval.manage", "list saved experiments"),
    "compare": ("agenteval.compare", "is version B really better than A?"),
    "gate": ("agenteval.gate", "pass, block or can't judge a change, for CI"),
    "judge": ("agenteval.judge", "score answers with an LLM judge"),
    "calibrate": ("agenteval.calibrate", "label answers yourself and measure the judge's agreement"),
    "rescore": ("agenteval.rescore", "re-run the scorers without calling the agent again"),
    "estimate": ("agenteval.pricing", "estimate what a run will cost before starting it"),
    "prices": ("agenteval.pricing", "show or update the model price table"),
    "power": ("agenteval.power", "how many cases you need to catch a drop of a given size"),
    "generate": ("agenteval.generate", "write test cases from your documents with an LLM"),
    "import": ("agenteval.traces", "import traces your agent already recorded, to score them"),
    "export": ("agenteval.traces", "export an experiment's trials (jsonl, or Ragas samples)"),
    "push": ("agenteval.share", "send an experiment to your team's agenteval server"),
    "delete": ("agenteval.manage", "delete an experiment"),
    "ui": ("agenteval.ui", "open the dashboard in your browser (needs pyagenteval[server])"),
}

# Options that may be given several times (action="append"): a list in the config means
# the flag repeated, not one flag with several values.
REPEATABLE = {"scorer"}

# Commands without a --db option, or whose databases are positional.
NO_DB = {"init", "gate", "power", "prices", "generate"}

def config_arguments(command: str, config: dict[str, Any]) -> list[str]:
    """The config's options for `command`, as command-line arguments."""
    options = dict(config.get(command, {}))
    if "db" in config and command not in NO_DB:
        options.setdefault("db", config["db"])

    arguments: list[str] = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is False or value is None:
            continue
        if value is True:
            arguments.append(flag)
        elif isinstance(value, list) and key in REPEATABLE:
            for item in value:
                arguments += [flag, str(item)]
        elif isinstance(value, list):
            arguments += [flag, *map(str, value)]
        else:
            arguments += [flag, str(value)]
    return arguments

def load_config(path: Path = Path(CONFIG_FILE)) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise SystemExit(f"{path} isn't valid TOML: {error}") from None

def usage() -> str:
    width = max(len(name) for name in COMMANDS) + 2
    lines = ["usage: agenteval <command> [options]", "", "Commands:"]
    lines += [f"  {name:<{width}}{description}" for name, (_, description) in COMMANDS.items()]
    lines += ["", "Run `agenteval <command> --help` for a command's options.",
              f"Options can also be set in {CONFIG_FILE}; see `agenteval init`."]
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return
    if argv[0] in ("-V", "--version", "version"):
        from importlib import metadata

        try:
            print(f"agenteval {metadata.version('pyagenteval')}")
        except metadata.PackageNotFoundError:
            print("agenteval (not installed as a package)")
        return

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}.\n\n{usage()}")

    # Config options go first so the same option on the command line overrides them. For
    # calibrate, which has its own sub-commands, they go after the sub-command.
    defaults = config_arguments(command, load_config())
    if command == "calibrate" and rest and not rest[0].startswith("-"):
        rest = [rest[0], *defaults, *rest[1:]]
    else:
        rest = [*defaults, *rest]

    module = importlib.import_module(COMMANDS[command][0])
    entry = getattr(module, f"main_{command}", None) or module.main
    entry(rest)

if __name__ == "__main__":
    main()
