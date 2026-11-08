"""agenteval: run an LLM agent many times, trace every step, and find out why it fails.

    from agenteval import evaluate, llm_call, tool

    async def my_agent(question: str) -> str:
        with llm_call("gpt-4o-mini") as call:
            ...
        return answer

    results = evaluate(my_agent, "cases.jsonl", model="gpt-4o-mini", price=(0.15, 0.60))
    print(results)

See https://zeustucz.github.io/agent-eval/ for the guide.
"""
from agenteval.evaluation import Results, aevaluate, evaluate
from agenteval.integrations.openai import wrap_openai
from agenteval.models import EvaluationCase, Score, Span, Trial
from agenteval.scorers import as_scorer
from agenteval.sdk import ToolArgumentError, llm_call, tool

Case = EvaluationCase  # shorter, for building cases in code

__all__ = [
    "Case", "EvaluationCase", "Results", "Score", "Span", "ToolArgumentError", "Trial",
    "aevaluate", "as_scorer", "evaluate", "llm_call", "tool", "wrap_openai",
]
