import json
from random import Random
from pathlib import Path

from agenteval.models import EvaluationCase
from agenteval.rag import DEFAULT_EMBEDDING, DEFAULT_TOP_K, rag_tools
from agenteval.tools import DEFAULT_TOOLS, HARD_TOOLS, Tool

DATASETS_DIR = Path(__file__).resolve().parent / "data"  # shipped inside the package

def load_jsonl(path: Path) -> list[EvaluationCase]:
    return [
        EvaluationCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def load_dataset_file(path: Path) -> list[EvaluationCase]:
    """Your own dataset: one JSON object per line, in the same form as the built-in hard dataset.

    Only "input" and "expected_output" are required. "id" defaults to the line number. Errors
    name the line, since a dataset is usually written by hand.
    """
    if not path.is_file():
        raise ValueError(f"no dataset file at {path}")

    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError("each line must be a JSON object")
            data.setdefault("id", number)
            cases.append(EvaluationCase.model_validate(data))
        except ValueError as error:  # json.JSONDecodeError and pydantic's ValidationError are ValueErrors
            raise ValueError(f"{path}, line {number}: {error}") from None

    if not cases:
        raise ValueError(f"{path} has no cases")
    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"{path}: case ids must be unique; repeated: {duplicates}")
    return cases

def is_dataset_file(name: str) -> bool:
    return split_sample(name)[0].endswith(".jsonl")

SAMPLE_MARK = "#sample="

def sample_name(name: str, size: int, seed: int) -> str:
    """The name a sampled dataset is stored under, e.g. "hard#sample=20,seed=0": enough to
    rebuild exactly the same cases later."""
    return f"{name}{SAMPLE_MARK}{size},seed={seed}"

def split_sample(name: str) -> tuple[str, tuple[int, int] | None]:
    """("hard", (20, 0)) for "hard#sample=20,seed=0"; (name, None) when it isn't a sample."""
    base, mark, rest = name.partition(SAMPLE_MARK)
    if not mark:
        return name, None
    size, _, seed = rest.partition(",seed=")
    return base, (int(size), int(seed or 0))

def sample_cases(cases: list[EvaluationCase], size: int, seed: int = 0) -> list[EvaluationCase]:
    """A fixed random sample, spread across categories in proportion to their size, so a
    quick run still covers every kind of question. In the dataset's original order."""
    if size >= len(cases):
        return list(cases)
    by_category: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)

    random = Random(seed)
    # Largest remainder: each category gets its share, rounded so the total is exactly `size`.
    shares = {name: size * len(group) / len(cases) for name, group in by_category.items()}
    counts = {name: int(share) for name, share in shares.items()}
    leftovers = sorted(shares, key=lambda name: (shares[name] - counts[name], random.random()), reverse=True)
    for name in leftovers[: size - sum(counts.values())]:
        counts[name] += 1

    chosen = {case.id for name, group in by_category.items() for case in random.sample(group, counts[name])}
    return [case for case in cases if case.id in chosen]

def hard_cases() -> list[EvaluationCase]:
    """About 55 cases in 10 categories, built by scripts/build_hard_dataset.py."""
    return load_jsonl(DATASETS_DIR / "hard.jsonl")

def rag_cases() -> list[EvaluationCase]:
    """About 45 questions over a fictional company's documents, built by scripts/build_rag_dataset.py."""
    return load_jsonl(DATASETS_DIR / "rag.jsonl")

def tools_for_dataset(name: str, top_k: int = DEFAULT_TOP_K, embedding: str = DEFAULT_EMBEDDING) -> list[Tool]:
    """The hard dataset needs extra tools (and reference data with planted errors and injections),
    and the rag dataset a search index. top_k and embedding only apply to rag."""
    if name == "rag":
        return rag_tools(top_k, embedding)
    return HARD_TOOLS if name == "hard" else DEFAULT_TOOLS

def arithmetic_cases(count: int) -> list[EvaluationCase]:
    return [
        EvaluationCase(
            id=index,
            input=f"What is {index} + 1",
            expected_output=str(index + 1),
            expected_tool="calculator",
        )
        for index in range(1, count + 1)
    ]

def dataset_by_name(name: str) -> list[EvaluationCase]:
    """Rebuild a dataset from the name stored with an experiment, e.g. "mixed", "arithmetic-100",
    the path of your own .jsonl file (relative to where the run was started), or a sample of
    any of those, e.g. "hard#sample=20,seed=0"."""
    base, sample = split_sample(name)
    if sample is not None:
        return sample_cases(dataset_by_name(base), *sample)
    if is_dataset_file(name):
        return load_dataset_file(Path(name))
    if name == "mixed":
        return mixed_cases()
    if name == "hard":
        return hard_cases()
    if name == "rag":
        return rag_cases()
    if name.startswith("arithmetic-"):
        return arithmetic_cases(int(name.removeprefix("arithmetic-")))
    raise ValueError(f"unknown dataset {name!r}")

def mixed_cases() -> list[EvaluationCase]:
    """A small dataset for real models: calculations too big to do reliably in your head,
    capital-city questions a model could answer from memory (does it use the tool anyway?),
    and one country missing from the tool's database (does it admit that, or guess?)."""
    calculations = [
        ("48213 * 9917", "478128321"),
        ("731942 + 268177", "1000119"),
        ("9876543 - 1234567", "8641976"),
        ("123456 * 789", "97406784"),
        ("88888 * 77777", "6913441976"),
        ("1000000 - 999999", "1"),
        ("45678 * 32109", "1466674902"),
        ("98765 / 5", "19753"),
    ]
    capitals = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Brazil", "Brasília"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
        ("Nigeria", "Abuja"),
        ("Kazakhstan", "Astana"),
    ]

    cases = [
        EvaluationCase(id=index, input=f"What is {expression}?", expected_output=answer, expected_tool="calculator")
        for index, (expression, answer) in enumerate(calculations, start=1)
    ]
    cases += [
        EvaluationCase(id=index, input=f"What is the capital of {country}?", expected_output=capital,
                       expected_tool="lookup_capital")
        for index, (country, capital) in enumerate(capitals, start=len(cases) + 1)
    ]
    # Not in the tool's database. The honest answer is to say the tool has no record.
    cases.append(EvaluationCase(
        id=len(cases) + 1,
        input="What is the capital of Bhutan?",
        expected_output="unknown",
        expected_tool="lookup_capital",
        expects_tool_error=True,
    ))
    return cases
