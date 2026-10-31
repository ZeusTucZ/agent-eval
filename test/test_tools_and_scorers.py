import pytest

from evalkit.models import EvaluationCase, Span, Trial
from evalkit.scorers import (
    exact_match,
    score_trials,
    tool_arguments_valid,
    tool_execution,
    tool_faithfulness,
    tool_selection,
)
from evalkit.tools import DEFAULT_TOOLS, ToolArgumentError, ToolRegistry, UnknownToolError
from evalkit.tracer import Tracer

registry = ToolRegistry(DEFAULT_TOOLS)

# --- Tool registry ---

def test_valid_call_runs_the_tool_and_is_traced():
    tracer = Tracer(trial_id=1)
    assert registry.call("calculator", {"a": 48213, "operator": "*", "b": 9917}, tracer) == 478128321
    assert (tracer.spans[0].kind, tracer.spans[0].name, tracer.spans[0].status) == ("tool", "calculator", "success")

@pytest.mark.parametrize("name, arguments, error, error_type", [
    ("search_web", {"query": "x"}, UnknownToolError, "UnknownToolError"),
    ("calculator", {"a": 1, "operator": "^", "b": 2}, ToolArgumentError, "ToolArgumentError"),
    ("calculator", {"a": 1}, ToolArgumentError, "ToolArgumentError"),
    ("lookup_capital", {"country": "Bhutan"}, LookupError, "LookupError"),
])
def test_failed_calls_are_traced_with_their_reason(name, arguments, error, error_type):
    tracer = Tracer(trial_id=1)
    with pytest.raises(error):
        registry.call(name, arguments, tracer)
    assert (tracer.spans[0].status, tracer.spans[0].error_type) == ("error", error_type)

# --- Scorers ---

case = EvaluationCase(id=1, input="What is the capital of Brazil?", expected_output="Brasília",
                      expected_tool="lookup_capital")

def tool_span(name: str, error_type: str | None = None) -> Span:
    return Span(trial_id=1, kind="tool", name=name, start=0, end=0,
                status="error" if error_type else "success", error_type=error_type)

def trial_with(*spans: Span, output: str | None = "Brasília") -> Trial:
    return Trial(id=1, case_id=1, attempt=1, output=output, spans=list(spans))

@pytest.mark.parametrize("output", ["Brasília", "brasilia", " Brasília. ", "BRASILIA"])
def test_exact_match_forgives_case_accents_and_punctuation(output):
    assert exact_match(case, trial_with(output=output)).value == 1.0

def test_exact_match_still_rejects_different_answers():
    assert exact_match(case, trial_with(output="Rio de Janeiro")).value == 0.0

def llm_span(error_type: str | None = None) -> Span:
    return Span(trial_id=1, kind="llm", name="model", start=0, end=0,
                status="error" if error_type else "success", error_type=error_type)

def test_tool_selection_is_not_judged_when_the_model_never_responded():
    # Every request was rate-limited: the model made no choice, so it didn't choose wrong.
    trial = trial_with(llm_span("rate_limit"), llm_span("rate_limit"), output=None)
    assert tool_selection(case, trial) is None

def test_answering_from_memory_fails_tool_selection_but_not_exact_match():
    trial = trial_with(llm_span())  # right answer, model responded, no tool called
    assert exact_match(case, trial).value == 1.0
    assert tool_selection(case, trial).value == 0.0
    assert tool_arguments_valid(case, trial) is None  # nothing to judge
    assert tool_execution(case, trial) is None

def test_each_ladder_rung_catches_its_own_failure():
    bad_arguments = trial_with(llm_span(), tool_span("lookup_capital", "ToolArgumentError"))
    assert tool_selection(case, bad_arguments).value == 1.0
    assert tool_arguments_valid(case, bad_arguments).value == 0.0
    assert tool_execution(case, bad_arguments) is None  # never got to run

    crashed = trial_with(tool_span("lookup_capital", "LookupError"))
    assert tool_arguments_valid(case, crashed).value == 1.0
    assert tool_execution(case, crashed).value == 0.0

    # The agent corrected itself: one bad call, then a good one.
    recovered = trial_with(tool_span("lookup_capital", "ToolArgumentError"), tool_span("lookup_capital"))
    assert tool_arguments_valid(case, recovered).value == 0.0
    assert tool_execution(case, recovered).value == 1.0

def test_score_trials_skips_scorers_with_nothing_to_judge():
    no_expected_tool = case.model_copy(update={"expected_tool": None})
    scores = score_trials([no_expected_tool], [trial_with()])
    assert [score.scorer for score in scores] == ["exact_match"]

# --- Hallucination: tool_faithfulness ---

def recorded_tool(result: str | None, error_type: str | None = None) -> Span:
    return Span(trial_id=1, kind="tool", name="calculator", start=0, end=0, input='{"a": 1}', output=result,
                status="error" if error_type else "success", error_type=error_type)

calculation = EvaluationCase(id=1, input="What is 48213 * 9917?", expected_output="478128321", expected_tool="calculator")

@pytest.mark.parametrize("output, results, expected", [
    ("478128321", ["478128321"], 1.0),
    ("478,128,321", ["478128321"], 1.0),         # formatting doesn't matter
    ("478128322", ["478128321"], 0.0),           # one digit away is an invented number
    ("623 534.97", ["623", "534.97"], 1.0),      # every number has a source
    ("623 999", ["623"], 0.0),                   # one number doesn't
    ("391", [], 0.0),                            # answered without any tool result
])
def test_numbers_must_come_from_tool_results(output, results, expected):
    trial = trial_with(llm_span(), *[recorded_tool(result) for result in results], output=output)
    assert tool_faithfulness(calculation, trial).value == expected

@pytest.mark.parametrize("output, result, expected", [
    ("Brasília", '"Brasília"', 1.0),
    ("Brasilia", '"Brasília"', 1.0),
    ("Rio de Janeiro", '"Brasília"', 0.0),
])
def test_text_must_come_from_tool_results(output, result, expected):
    assert tool_faithfulness(case, trial_with(llm_span(), recorded_tool(result), output=output)).value == expected

def test_the_tool_being_wrong_is_not_a_hallucination():
    # It measures faithfulness to the source, not truth: this answer is wrong but faithful.
    turkey = EvaluationCase(id=1, input="Capital of Turkey?", expected_output="Ankara", expected_tool="lookup_capital")
    assert tool_faithfulness(turkey, trial_with(llm_span(), recorded_tool('"Istanbul"'), output="Istanbul")).value == 1.0

def test_a_failed_tool_call_backs_nothing():
    trial = trial_with(llm_span(), recorded_tool(None, "LookupError"), output="Thimphu")
    assert tool_faithfulness(case, trial).value == 0.0

@pytest.mark.parametrize("trial, the_case", [
    (trial_with(llm_span(), output="unknown"), case),                                         # claimed nothing
    (trial_with(llm_span(), output=None), case),                                              # no answer
    (trial_with(llm_span(), output="god"), case.model_copy(update={"expected_tool": None})),  # no tool required
    (trial_with(llm_span(), tool_span("calculator"), output="391"), case),                    # tool data not recorded
])
def test_nothing_to_judge(trial, the_case):
    assert tool_faithfulness(the_case, trial) is None
