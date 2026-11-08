"""Starting points for `agenteval init --template ...`: an agent file and cases for each.

Each template is (agent source, cases, model, price, what to install, what it needs set).
"""

OPENAI_AGENT = '''"""An agent on the OpenAI API. wrap_openai records every model call, with its tokens,
so agenteval can time, price and score it. Any OpenAI-compatible provider works the same:
pass base_url (and api_key) to AsyncOpenAI for Groq, OpenRouter, Ollama or vLLM.

Needs OPENAI_API_KEY. Tools your agent runs go through agenteval's tool(...):
    weather = await tool("get_weather", get_weather, city="Lisbon")
"""
from openai import AsyncOpenAI

from agenteval import wrap_openai

client = wrap_openai(AsyncOpenAI())
MODEL = "gpt-4o-mini"

async def answer(question: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "Answer with only the answer, as briefly as possible."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""
'''

ANTHROPIC_AGENT = '''"""An agent on the Anthropic API. wrap_anthropic records every model call, with its tokens,
so agenteval can time, price and score it.

Needs ANTHROPIC_API_KEY. Tools your agent runs on tool_use blocks go through agenteval's tool(...):
    weather = await tool("get_weather", get_weather, city="Lisbon")
"""
from anthropic import AsyncAnthropic

from agenteval import wrap_anthropic

client = wrap_anthropic(AsyncAnthropic())
MODEL = "claude-sonnet-4-5"

async def answer(question: str) -> str:
    message = await client.messages.create(
        model=MODEL,
        max_tokens=256,
        system="Answer with only the answer, as briefly as possible.",
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in message.content if block.type == "text")
'''

LANGGRAPH_AGENT = '''"""A LangGraph ReAct agent. AgentEvalCallback records every model and tool call it makes.

Needs: pip install langgraph langchain-openai, and OPENAI_API_KEY.
"""
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agenteval.integrations.langchain import AgentEvalCallback

@lc_tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers exactly."""
    return a * b

graph = create_react_agent(
    ChatOpenAI(model="gpt-4o-mini", temperature=0),
    tools=[multiply],
    prompt="Use the tools for any arithmetic. Answer with only the number.",
)

async def answer(question: str) -> str:
    result = await graph.ainvoke({"messages": [("user", question)]},
                                 config={"callbacks": [AgentEvalCallback()]})
    return result["messages"][-1].content
'''

LANGGRAPH_CASES = """\
{"input": "What is 1234 times 5678?", "expected_output": "7006652", "expected_tools": ["multiply"], "category": "arithmetic"}
{"input": "What is 0.25 times 640?", "expected_output": "160", "expected_tools": ["multiply"], "category": "arithmetic"}
{"input": "What is the capital of France?", "expected_output": "Paris", "expected_tools": [], "category": "no_tool"}
"""

SIMPLE_CASES = """\
{"input": "What is the capital of France?", "expected_output": "Paris", "category": "geography"}
{"input": "How many legs does a spider have?", "expected_output": "8", "category": "science"}
{"input": "What is 17 squared?", "expected_output": "289", "category": "arithmetic"}
"""

RAG_AGENT = '''"""A small retrieval agent: it searches a handful of documents, then answers from the best
match. Runs with no API key (the "model" is a rule), so you can see retrieval recall and the
search-or-answer diagnosis straight away. Replace search() with your vector store and the
rule with a model call.

The warranty question fails on purpose: the naive word match ranks the shipping document
first (both contain "the"), and the report shows it as a retrieval failure, not a wrong
answer. Try improving search() and run again.
"""
import re

from agenteval import llm_call, tool

DOCUMENTS = {
    "refunds": "Refunds are accepted within 30 days of purchase. ANSWER: 30 days",
    "shipping": "Standard shipping in the EU takes 5 business days. ANSWER: 5 business days",
    "support": "Support is open Monday to Friday, 9:00 to 18:00. ANSWER: 9:00 to 18:00",
    "warranty": "Every product has a 2-year warranty. ANSWER: 2 years",
}

def search(query: str, k: int = 2) -> list[dict]:
    words = set(re.findall(r"\\w+", query.lower()))
    ranked = sorted(DOCUMENTS.items(), key=lambda item: -len(words & set(re.findall(r"\\w+", item[1].lower()))))
    return [{"doc_id": doc_id, "text": text} for doc_id, text in ranked[:k]]

async def answer(question: str) -> str:
    results = await tool("search_docs", search, query=question)
    with llm_call() as call:  # stands in for asking a model to answer from the results
        call.tokens(len(question.split()) + sum(len(r["text"].split()) for r in results), 4)
        best = results[0]["text"] if results else ""
    found = re.search(r"ANSWER: (.+)$", best)
    return found.group(1) if found else "unknown"
'''

RAG_CASES = """\
{"input": "How many days do I have to ask for a refund?", "expected_output": "30 days", "relevant_docs": ["refunds"], "expected_tools": ["search_docs"], "category": "policy"}
{"input": "How long does standard shipping take?", "expected_output": "5 business days", "relevant_docs": ["shipping"], "expected_tools": ["search_docs"], "category": "policy"}
{"input": "When is support open?", "expected_output": "9:00 to 18:00", "relevant_docs": ["support"], "expected_tools": ["search_docs"], "category": "support"}
{"input": "How long is the warranty?", "expected_output": "2 years", "relevant_docs": ["warranty"], "expected_tools": ["search_docs"], "category": "product"}
"""

# name -> (agent source or None for the basic one, cases or None, model, price, install, environment)
TEMPLATES: dict[str, tuple[str | None, str | None, str, list[float], str | None, str | None]] = {
    "basic": (None, None, "example-model", [0, 0], None, None),
    "openai": (OPENAI_AGENT, SIMPLE_CASES, "gpt-4o-mini", [0.15, 0.60], None, "OPENAI_API_KEY"),
    "anthropic": (ANTHROPIC_AGENT, SIMPLE_CASES, "claude-sonnet-4-5", [3.0, 15.0], "anthropic", "ANTHROPIC_API_KEY"),
    "langgraph": (LANGGRAPH_AGENT, LANGGRAPH_CASES, "gpt-4o-mini", [0.15, 0.60], "langgraph langchain-openai",
                  "OPENAI_API_KEY"),
    "rag": (RAG_AGENT, RAG_CASES, "example-model", [0, 0], None, None),
}
