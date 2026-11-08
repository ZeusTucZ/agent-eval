"""Builds docs/reference/cli.md from each command's own --help, so the reference can't drift
from the code. A test checks the file matches; to update it, run:

    python scripts/build_cli_reference.py
"""
import contextlib
import io
import os
from pathlib import Path

from agenteval.cli import COMMANDS, main

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "reference" / "cli.md"

HEADER = """# Command line

Every command is `agenteval <command>`. Options can also be set in
[agenteval.toml](config.md); the command line wins. This page is generated from each
command's `--help` by `scripts/build_cli_reference.py`.
"""

def help_text(arguments: list[str]) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
        main([*arguments, "--help"])
    return buffer.getvalue().rstrip()

def build() -> str:
    # A fixed width, so the output doesn't depend on the terminal it was built in.
    os.environ["COLUMNS"] = "100"
    parts = [HEADER]
    for command, (_, description) in COMMANDS.items():
        parts.append(f"## {command}\n\n{description[0].upper()}{description[1:]}.\n\n```\n{help_text([command])}\n```\n")
        if command == "calibrate":
            for sub in ("label", "agree"):
                parts.append(f"### calibrate {sub}\n\n```\n{help_text([command, sub])}\n```\n")
    return "\n".join(parts)

if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
