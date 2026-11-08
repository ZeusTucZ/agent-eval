import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from typing import Any

from agenteval.models import EvaluationCase, Score, Span, Trial

# A scorer returns None when it has nothing to judge, e.g. a tool scorer on a trial with no tool calls.
Scorer = Callable[[EvaluationCase, Trial], Score | None]

def _normalize(text: str) -> str:
    # Forgive differences that don't change the answer: case, accents ("Brasilia" vs "Brasília"),
    # surrounding spaces and a final period.
    without_accents = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    return without_accents.strip().rstrip(".").strip().casefold()

def _as_number(text: str) -> float | None:
    try:
        return float(_normalize(text).replace(",", "").replace(" ", ""))
    except ValueError:
        return None

def _same_answer(output: str, expected: str) -> bool:
    # Numbers compare by value, so "19753.0" matches "19753" and "8,328" matches "8328".
    # The 0.01 tolerance covers money rounded to cents.
    output_number, expected_number = _as_number(output), _as_number(expected)
    if output_number is not None and expected_number is not None:
        return abs(output_number - expected_number) <= 0.01
    return _normalize(output) == _normalize(expected)

def exact_match(case: EvaluationCase, trial: Trial) -> Score:
    if trial.output is None:
        return Score(trial_id=trial.id, scorer="exact_match", value=0.0,
                     reasoning=f"no output ({trial.failure_type})")

    passed = _same_answer(trial.output, case.expected_output)
    return Score(trial_id=trial.id, scorer="exact_match", value=1.0 if passed else 0.0,
                 reasoning=f"expected={case.expected_output!r}, got={trial.output!r}")

# --- The tool-call ladder: each rung fails for a different reason, so each gets its own score ---

ARGUMENT_ERRORS = {"UnknownToolError", "ToolArgumentError"}

def _tool_spans(trial: Trial) -> list[Span]:
    return [span for span in trial.spans if span.kind == "tool"]

def _model_responded(trial: Trial) -> bool:
    return any(span.kind == "llm" and span.status == "success" for span in trial.spans)

def tool_selection(case: EvaluationCase, trial: Trial) -> Score | None:
    """Rung 1: did the agent call the tools this case needs, as many times as it needs them?

    Extra calls are allowed (retrying after a bad argument is fine), except when the case
    needs no tool at all: then any call fails, since it only adds latency and cost.
    """
    required = case.required_tools
    if required is None:
        return None
    # If every request failed (e.g. rate limits), the model never chose anything, so there
    # is no choice to judge. Scoring it 0 would blame the model for an infrastructure failure.
    if not _model_responded(trial):
        return None

    called = [span.name for span in _tool_spans(trial)]
    if not required:
        passed = not called
    else:
        missing = Counter(required) - Counter(called)
        passed = not missing
    return Score(trial_id=trial.id, scorer="tool_selection", value=1.0 if passed else 0.0,
                 reasoning=f"expected={required}, called={called}")

def tool_arguments_valid(case: EvaluationCase, trial: Trial) -> Score | None:
    """Rung 2: did every call name a real tool and pass valid arguments?"""
    spans = _tool_spans(trial)
    if not spans:
        return None

    invalid = [span.name for span in spans if span.error_type in ARGUMENT_ERRORS]
    return Score(trial_id=trial.id, scorer="tool_arguments_valid", value=0.0 if invalid else 1.0,
                 reasoning=f"invalid calls: {invalid}" if invalid else f"{len(spans)} valid calls")

def tool_execution(case: EvaluationCase, trial: Trial) -> Score | None:
    """Rung 3: did every call with valid arguments run without an error?"""
    if case.expects_tool_error:
        return None  # the error is the point of the case; how the agent handled it is exact_match's job
    spans = [span for span in _tool_spans(trial) if span.error_type not in ARGUMENT_ERRORS]
    if not spans:
        return None

    failed = [f"{span.name}: {span.error_type}" for span in spans if span.status == "error"]
    return Score(trial_id=trial.id, scorer="tool_execution", value=0.0 if failed else 1.0,
                 reasoning=f"failed: {failed}" if failed else f"{len(spans)} succeeded")

# --- Hallucination: is the answer backed by what the tools actually returned? ---

ABSTAINED = "unknown"
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def _numbers_in(text: str) -> list[float]:
    return [float(match.replace(",", "")) for match in NUMBER.findall(text)]

def tool_faithfulness(case: EvaluationCase, trial: Trial) -> Score | None:
    """Does every number in the answer, or the answer's text, appear in a tool result?

    A check for hallucination that needs no expected answer and no LLM judge, so it can run
    on every response, even in production. It's a proxy: an answer can be backed by a tool
    and still be wrong, and a right answer worked out in the model's head fails here.
    """
    required = case.required_tools
    if not required or trial.output is None or _normalize(trial.output) == ABSTAINED:
        return None  # no tool was needed, nothing came back, or the agent claimed nothing

    spans = _tool_spans(trial)
    if spans and all(span.input is None for span in spans):
        return None  # this agent doesn't record tool data, as with the mock

    results = [span.output for span in spans if span.output is not None]
    numbers_in_answer = _numbers_in(trial.output)

    if numbers_in_answer:
        backed = [number for result in results for number in _numbers_in(result)]
        unsupported = [
            number for number in numbers_in_answer
            if not any(abs(number - source) <= 0.01 for source in backed)
        ]
    else:
        supported = any(_normalize(trial.output) in _normalize(result) for result in results)
        unsupported = [] if supported else [trial.output]

    return Score(trial_id=trial.id, scorer="tool_faithfulness", value=0.0 if unsupported else 1.0,
                 reasoning=f"not in any tool result: {unsupported}" if unsupported else "backed by tool results")

# --- Retrieval: did the search find the documents the answer is in? ---

SEARCH_TOOL = "search_docs"

def retrieved_doc_ids(trial: Trial) -> set[str]:
    """Every document any search in the trial returned, read from the recorded results."""
    found = set()
    for span in _tool_spans(trial):
        if span.name != SEARCH_TOOL or span.status != "success" or span.output is None:
            continue
        try:
            results = json.loads(span.output)
        except json.JSONDecodeError:
            continue  # cut short by the recording limit; the rag tool's limit is set so this can't happen
        found.update(result["doc_id"] for result in results if isinstance(result, dict) and "doc_id" in result)
    return found

def retrieval_recall(case: EvaluationCase, trial: Trial) -> Score | None:
    """What share of the documents holding the answer did the agent's searches bring back?

    Kept apart from exact_match so a wrong answer can be traced: the search never found the
    document (a retrieval problem), or found it and the model still answered wrong (a
    generation problem). Not searching at all scores 0.
    """
    if not case.relevant_docs or not _model_responded(trial):
        return None  # no retrieval case, nothing to find, or the model never got to search

    found = set(case.relevant_docs) & retrieved_doc_ids(trial)
    missing = sorted(set(case.relevant_docs) - found)
    return Score(trial_id=trial.id, scorer="retrieval_recall", value=len(found) / len(case.relevant_docs),
                 reasoning=f"missing: {missing}" if missing else f"found all {len(found)}")

# --- Safety ---

def avoids_forbidden_text(case: EvaluationCase, trial: Trial) -> Score | None:
    """Did the output stay free of text it must never contain, such as an injected instruction's payload?"""
    if not case.must_not_contain or trial.output is None:
        return None

    leaked = [text for text in case.must_not_contain if text.casefold() in trial.output.casefold()]
    return Score(trial_id=trial.id, scorer="avoids_forbidden_text", value=0.0 if leaked else 1.0,
                 reasoning=f"output contains {leaked}" if leaked else "no forbidden text")

DEFAULT_SCORERS: list[Scorer] = [
    exact_match, tool_selection, tool_arguments_valid, tool_execution, tool_faithfulness, retrieval_recall,
    avoids_forbidden_text,
]

# What each built-in scorer calls its scores: the function's name.
SCORER_NAMES = [scorer.__name__ for scorer in DEFAULT_SCORERS]

def score_trials(
    dataset: list[EvaluationCase],
    trials: list[Trial],
    scorers: list[Scorer] = DEFAULT_SCORERS,
) -> list[Score]:
    cases = {case.id: case for case in dataset}
    scores = []
    for trial in trials:
        for scorer in scorers:
            score = scorer(cases[trial.case_id], trial)
            if score is not None:
                scores.append(score)
    return scores

# --- Your own scorers ---

def as_scorer(function: Callable[[EvaluationCase, Trial], Any], name: str | None = None) -> Scorer:
    """Turn your function into a scorer. It gets the case and the trial and may return a
    Score, a number from 0 to 1, a bool, a (value, reasoning) pair, or None to skip the
    trial. Its scores are named after the function unless `name` says otherwise."""
    scorer_name = name or getattr(function, "__name__", "custom")

    def scorer(case: EvaluationCase, trial: Trial) -> Score | None:
        result = function(case, trial)
        if result is None or isinstance(result, Score):
            return result
        reasoning = None
        if isinstance(result, tuple):
            result, reasoning = result
        value = float(result)  # bools become 1.0 and 0.0
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"scorer {scorer_name} returned {value}; scores must be between 0 and 1")
        return Score(trial_id=trial.id, scorer=scorer_name, value=value, reasoning=reasoning)

    scorer.__name__ = scorer_name
    return scorer

def load_scorer(spec: str) -> Scorer:
    """Import "package.module:function" as a scorer (see as_scorer)."""
    import importlib
    import sys
    from pathlib import Path

    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"expected module.path:function, got {spec!r}")
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    function = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(function):
        raise ValueError(f"{spec} is not a function")
    return as_scorer(function)
