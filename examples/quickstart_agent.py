"""The smallest agent you can evaluate: no API key, no network, runs in a second.

Its "model" is a few rules instead of an LLM, so you can see how agent-eval traces and
scores your own agent before wiring in a real one (see groq_agent.py for that):

    python -m evalkit.run --agent examples.quickstart_agent:answer --model rules --price 0 0 \
        --dataset examples/quickstart_cases.jsonl --name quickstart
    python -m evalkit.report 1

It gets some cases wrong on purpose: a city missing from its "database" and a question it
can't parse, so the report has failures to explain.
"""
import re

from evalkit import ToolArgumentError, llm_call, tool

POPULATIONS = {"lisbon": 545_000, "tallinn": 457_000, "montevideo": 1_320_000}

def calculator(a: float, operator: str, b: float) -> float:
    if operator not in "+-*/":
        raise ToolArgumentError(f"unknown operator {operator!r}")
    result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else float("nan")}[operator]
    return int(result) if result == int(result) else result

def population(city: str) -> int:
    try:
        return POPULATIONS[city.lower()]
    except KeyError:
        raise LookupError(f"no population on record for {city!r}") from None

async def answer(question: str) -> str:
    # Where a real agent asks its model what to do, this one matches patterns. The call is
    # still recorded, so latency and cost would be counted the same way.
    with llm_call() as call:
        sum_match = re.search(r"(-?\d+(?:\.\d+)?)\s*([-+*/])\s*(-?\d+(?:\.\d+)?)", question)
        city_match = re.search(r"population of (\w+)", question, re.IGNORECASE)
        call.tokens(len(question.split()), 5)

    if sum_match:
        a, operator, b = sum_match.groups()
        return str(await tool("calculator", calculator, a=float(a), operator=operator, b=float(b)))
    if city_match:
        try:
            return str(await tool("population", population, city=city_match.group(1)))
        except LookupError:
            return "unknown"  # the honest answer when the tool has no record
    return "I don't know how to answer that."
