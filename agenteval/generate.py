"""agenteval generate: write test cases from your own documents with an LLM.

Give it a folder of .md or .txt files. It splits them into passages, asks a model for one
question per passage whose short answer is stated in that passage, and writes:

- the cases (input, expected_output, relevant_docs = the passage's id), ready for --dataset;
- the passages as a corpus file, the form agenteval.rag.load_corpus reads, so the same
  documents can back your agent's search tool in tests.

A share of the questions (--unanswerable) asks about something the documents don't cover,
with "unknown" expected, so you also measure whether the agent admits it can't answer.

Generated questions are a starting point: read them before relying on them, and drop any
whose answer is wrong or ambiguous.
"""
import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from random import Random
from typing import Any

from agenteval.openai_agent import PROVIDERS
from agenteval.rag import Document

MAX_PASSAGE = 1800  # characters: long enough for context, short enough for one clear fact

QUESTION_PROMPT = (
    "Write one question that can be answered from the passage below, and its answer. The "
    "answer must be stated in the passage, be short (a number, a name, a date or a few words) "
    "and be the only correct answer. Don't refer to 'the passage' in the question. Reply with "
    'JSON only: {"question": "...", "answer": "..."}'
)
UNANSWERABLE_PROMPT = (
    "Write one question on the same topic as the passage below that the passage does NOT "
    "answer, and that someone might plausibly ask. Reply with JSON only: "
    '{"question": "..."}'
)

def read_passages(paths: list[Path]) -> list[Document]:
    """Files split at headings and blank lines into passages of at most MAX_PASSAGE characters."""
    files = sorted(p for path in paths for p in ([path] if path.is_file() else path.rglob("*"))
                   if p.is_file() and p.suffix.lower() in (".md", ".txt", ".markdown", ".rst"))
    documents = []
    for file in files:
        text = file.read_text(encoding="utf-8", errors="replace")
        blocks = [b.strip() for b in re.split(r"\n(?=#{1,6} )|\n\s*\n", text) if b.strip()]
        passage = ""
        number = 0
        for block in [*blocks, ""]:
            if passage and (not block or len(passage) + len(block) + 2 > MAX_PASSAGE):
                number += 1
                title = next((line.lstrip("# ").strip() for line in passage.splitlines() if line.strip()), file.stem)
                documents.append(Document(id=f"{file.stem}-{number}", title=title[:80], date="", text=passage))
                passage = ""
            if block:
                passage = f"{passage}\n\n{block}".strip() if passage else block[:MAX_PASSAGE]
    return documents

def _parse(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", text or "", re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) and data.get("question") else None

async def generate_cases(client: Any, model: str, documents: list[Document], count: int,
                         unanswerable: float = 0.1, seed: int = 0, concurrency: int = 4) -> list[dict]:
    random = Random(seed)
    chosen = [random.choice(documents) for _ in range(count)] if len(documents) < count \
        else random.sample(documents, count)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, document: Document) -> dict | None:
        no_answer = random.random() < unanswerable
        prompt = UNANSWERABLE_PROMPT if no_answer else QUESTION_PROMPT
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model, temperature=0.7, response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": f"{prompt}\n\n<passage>\n{document.text}\n</passage>"}],
                )
            except Exception:  # noqa: BLE001 - one failed request shouldn't lose the rest
                return None
        data = _parse(response.choices[0].message.content)
        if data is None or (not no_answer and not data.get("answer")):
            return None
        return {
            "id": index,
            "input": str(data["question"]).strip(),
            "expected_output": "unknown" if no_answer else str(data["answer"]).strip(),
            "category": "unanswerable" if no_answer else "generated",
            "relevant_docs": [] if no_answer else [document.id],
        }

    results = await asyncio.gather(*(one(i, doc) for i, doc in enumerate(chosen, start=1)))
    return [case for case in results if case is not None]

def main(argv: list[str] | None = None) -> None:
    import openai

    parser = argparse.ArgumentParser(prog="agenteval generate", description="Write test cases from your documents.")
    parser.add_argument("docs", type=Path, nargs="+", help="files or folders of .md / .txt")
    parser.add_argument("-n", "--count", type=int, default=30)
    parser.add_argument("-o", "--output", type=Path, default=Path("generated_cases.jsonl"))
    parser.add_argument("--corpus-out", type=Path, default=Path("corpus.jsonl"))
    parser.add_argument("--unanswerable", type=float, default=0.1, help="share of questions the docs don't answer")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="groq")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    documents = read_passages(args.docs)
    if not documents:
        raise SystemExit("No .md or .txt files found in " + ", ".join(map(str, args.docs)))
    base_url, key_variable = PROVIDERS[args.provider]
    key_variable = args.api_key_env or key_variable
    if key_variable and not os.environ.get(key_variable):
        raise SystemExit(f"Set the {key_variable} environment variable first.")
    client = openai.AsyncOpenAI(base_url=args.base_url or base_url,
                                api_key=os.environ.get(key_variable, "not-needed") if key_variable else "not-needed")

    print(f"Writing {args.count} questions from {len(documents)} passages with {args.model}...")
    cases = asyncio.run(generate_cases(client, args.model, documents, args.count, args.unanswerable, args.seed))
    args.output.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    args.corpus_out.write_text("".join(doc.model_dump_json() + "\n" for doc in documents), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {args.output} and {len(documents)} passages to {args.corpus_out}.\n"
          "Read the questions before relying on them, and delete any whose answer is wrong or ambiguous.")
