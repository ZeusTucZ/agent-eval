"""agenteval init: a working project in one command.

Writes an example agent, a few cases and an agenteval.toml that ties them together, so
`agenteval run` works straight away (no API key needed) and each file shows where your own
agent, questions and settings go.
"""
import argparse
from pathlib import Path

AGENT = '''"""Your agent. Replace the body with your own: keep llm_call around model calls and tool
around tool calls, and agenteval records each one as a step.

    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    async def answer(question: str) -> str:
        with llm_call("gpt-4o-mini") as call:
            response = await client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": question}])
            call.tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content
"""
import re

from agenteval import llm_call, tool

def add(a: float, b: float) -> float:
    return a + b

async def answer(question: str) -> str:
    # A stand-in for a model: finds two numbers to add. Swap in your real model call.
    with llm_call() as call:
        numbers = [float(n) for n in re.findall(r"-?\\d+(?:\\.\\d+)?", question)]
        call.tokens(len(question.split()), 3)
    if len(numbers) != 2:
        return "unknown"
    total = await tool("add", add, a=numbers[0], b=numbers[1])
    return str(int(total) if total == int(total) else total)
'''

CASES = """\
{"input": "What is 2 + 2?", "expected_output": "4", "expected_tools": ["add"], "category": "arithmetic"}
{"input": "Add 17 and 25.", "expected_output": "42", "expected_tools": ["add"], "category": "arithmetic"}
{"input": "What is 1.5 plus 2.25?", "expected_output": "3.75", "expected_tools": ["add"], "category": "arithmetic"}
{"input": "What is the capital of France?", "expected_output": "unknown", "expected_tools": [], "category": "out_of_scope"}
"""

CONFIG = """\
# agenteval settings for this project. Every key is a command-line option of the command
# its table is named after (`agenteval run --help` lists them); the command line wins.
db = "agenteval.db"

[run]
agent = "{module}:answer"     # your agent: module.path:function
model = "example-model"       # the model llm_call() records, for pricing
price = [0, 0]                # dollars per million input and output tokens
dataset = "{cases}"
attempts = 3                  # LLMs vary between runs, so ask each question more than once
name = "first-run"

# [judge]
# model = "llama-3.3-70b-versatile"
"""

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval init", description="Set up a project in this directory.")
    parser.add_argument("--agent-file", default="my_agent.py")
    parser.add_argument("--cases-file", default="cases.jsonl")
    parser.add_argument("--force", action="store_true", help="overwrite files that exist")
    args = parser.parse_args(argv)

    module = Path(args.agent_file).with_suffix("").as_posix().replace("/", ".")
    files = {
        Path(args.agent_file): AGENT,
        Path(args.cases_file): CASES,
        Path("agenteval.toml"): CONFIG.format(module=module, cases=Path(args.cases_file).as_posix()),
    }
    existing = [str(path) for path in files if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"Not overwriting {', '.join(existing)}; pass --force to replace them.")

    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  wrote {path}")
    print("\nNext:\n  agenteval run        run the example agent (no API key needed)\n"
          "  agenteval report     see what happened\n"
          "  agenteval ui         browse it in the dashboard (pip install zeustucz-agenteval[server])\n\n"
          f"Then put your own agent in {args.agent_file} and your questions in {args.cases_file}.")
