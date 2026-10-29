import unicodedata
from collections.abc import Callable

from evalkit.models import EvaluationCase, Score, Span, Trial

# A scorer returns None when it has nothing to judge, e.g. a tool scorer on a trial with no tool calls.
Scorer = Callable[[EvaluationCase, Trial], Score | None]

def _normalize(text: str) -> str:
    # Forgive differences that don't change the answer: case, accents ("Brasilia" vs "Brasília"),
    # surrounding spaces and a final period.
    without_accents = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    return without_accents.strip().rstrip(".").strip().casefold()

def exact_match(case: EvaluationCase, trial: Trial) -> Score:
    if trial.output is None:
        return Score(trial_id=trial.id, scorer="exact_match", value=0.0,
                     reasoning=f"no output ({trial.failure_type})")

    passed = _normalize(trial.output) == _normalize(case.expected_output)
    return Score(trial_id=trial.id, scorer="exact_match", value=1.0 if passed else 0.0,
                 reasoning=f"expected={case.expected_output!r}, got={trial.output!r}")

# --- The tool-call ladder: each rung fails for a different reason, so each gets its own score ---

ARGUMENT_ERRORS = {"UnknownToolError", "ToolArgumentError"}

def _tool_spans(trial: Trial) -> list[Span]:
    return [span for span in trial.spans if span.kind == "tool"]

def _model_responded(trial: Trial) -> bool:
    return any(span.kind == "llm" and span.status == "success" for span in trial.spans)

def tool_selection(case: EvaluationCase, trial: Trial) -> Score | None:
    """Rung 1: did the agent call the tool this case needs?"""
    if case.expected_tool is None:
        return None
    # If every request failed (e.g. rate limits), the model never chose anything, so there
    # is no choice to judge. Scoring it 0 would blame the model for an infrastructure failure.
    if not _model_responded(trial):
        return None

    called = [span.name for span in _tool_spans(trial)]
    passed = case.expected_tool in called
    return Score(trial_id=trial.id, scorer="tool_selection", value=1.0 if passed else 0.0,
                 reasoning=f"expected={case.expected_tool!r}, called={called}")

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
    spans = [span for span in _tool_spans(trial) if span.error_type not in ARGUMENT_ERRORS]
    if not spans:
        return None

    failed = [f"{span.name}: {span.error_type}" for span in spans if span.status == "error"]
    return Score(trial_id=trial.id, scorer="tool_execution", value=0.0 if failed else 1.0,
                 reasoning=f"failed: {failed}" if failed else f"{len(spans)} succeeded")

DEFAULT_SCORERS: list[Scorer] = [exact_match, tool_selection, tool_arguments_valid, tool_execution]

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
