"""Bringing your own agent and dataset: agenteval.sdk, --agent module:name and --dataset file.jsonl."""
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from agenteval import ToolArgumentError, llm_call, tool
from agenteval import storage
from agenteval.models import EvaluationCase
from agenteval.report import summarize
from agenteval.runner import run_experiment
from agenteval.sample_data import dataset_by_name, load_dataset_file
from agenteval.scorers import score_trials
from agenteval.sdk import FunctionAgent, load_agent
from agenteval.tracer import Tracer

case = EvaluationCase(id=1, input="What is 2 + 3?", expected_output="5", expected_tools=["add"])

def add(a: int, b: int) -> int:
    return a + b

def run_one(agent, cases=(case,)):
    trials = asyncio.run(run_experiment(agent, list(cases), attempts=1, timeout_s=5))
    return trials, {s.scorer: s.value for s in score_trials(list(cases), trials)}

# --- Tracing your agent's steps ---

def test_model_and_tool_calls_become_spans_and_scores():
    async def agent(question: str) -> str:
        with llm_call("my-model") as call:
            call.tokens(120, 8)
        return str(await tool("add", add, a=2, b=3))

    trials, scores = run_one(FunctionAgent(agent, model="default-model"))

    llm, add_span = trials[0].spans
    assert (llm.kind, llm.name, llm.input_tokens, llm.output_tokens) == ("llm", "my-model", 120, 8)
    assert (add_span.kind, add_span.name, add_span.status) == ("tool", "add", "success")
    assert json.loads(add_span.input) == {"a": 2, "b": 3} and add_span.output == "5"
    assert scores == {"exact_match": 1.0, "tool_selection": 1.0, "tool_arguments_valid": 1.0,
                      "tool_execution": 1.0, "tool_faithfulness": 1.0}

def test_llm_call_defaults_to_the_runs_model_and_async_tools_work():
    async def async_add(a: int, b: int) -> int:
        return a + b

    async def agent(question: str) -> str:
        with llm_call():
            pass
        return str(await tool("add", async_add, 2, 3))  # positional arguments are recorded too

    trials, _ = run_one(FunctionAgent(agent, model="run-model"))
    assert trials[0].spans[0].name == "run-model"
    assert json.loads(trials[0].spans[1].input) == {"args": [2, 3]}

def test_a_failing_tool_is_recorded_and_labels_the_trial():
    def broken(**_):
        raise RuntimeError("database down")

    async def agent(question: str) -> str:
        return await tool("add", broken, a=2, b=3)

    trials, scores = run_one(FunctionAgent(agent, model="m"))
    assert trials[0].failure_type == "tool_error"
    assert trials[0].spans[0].error_type == "RuntimeError"
    assert scores["exact_match"] == 0.0

def test_tool_argument_errors_count_against_the_agent_not_the_tool():
    def strict(a, b):
        raise ToolArgumentError("b must be a number")

    async def agent(question: str) -> str:
        try:
            await tool("add", strict, a=2, b="three")
        except ToolArgumentError:
            pass
        return str(await tool("add", add, a=2, b=3))  # the agent corrects itself

    _, scores = run_one(FunctionAgent(agent, model="m"))
    assert scores["tool_arguments_valid"] == 0.0
    assert scores["tool_execution"] == 1.0  # the valid call ran fine
    assert scores["exact_match"] == 1.0

def test_a_labelled_model_failure_is_counted_by_its_cause():
    async def agent(question: str) -> str:
        with llm_call() as call:
            call.failed_because("rate_limit")
            raise RuntimeError("429")

    trials, _ = run_one(FunctionAgent(agent, model="m"))
    assert trials[0].spans[0].error_type == "rate_limit"
    assert trials[0].failure_type == "llm_error"

def test_sync_agents_and_agents_that_want_the_whole_case():
    def agent(case: EvaluationCase) -> int:  # sync, gets the case, returns a non-string
        with llm_call():
            pass
        return int(case.expected_output)

    trials, scores = run_one(FunctionAgent(agent, model="m"))
    assert trials[0].output == "5" and len(trials[0].spans) == 1  # the thread still found the trial
    assert scores["exact_match"] == 1.0

def test_an_empty_answer_is_its_own_failure():
    async def agent(question: str) -> str:
        return "   "

    trials, _ = run_one(FunctionAgent(agent, model="m"))
    assert trials[0].failure_type == "empty_output"

def test_the_helpers_just_run_outside_an_evaluation():
    async def standalone():
        with llm_call("m") as call:
            call.tokens(1, 1)
        return await tool("add", add, a=2, b=3)

    assert asyncio.run(standalone()) == 5

def test_concurrent_trials_keep_their_own_spans():
    async def agent(question: str) -> str:
        n = int(question)
        await asyncio.sleep(0.01 * (5 - n))  # finish out of order
        return str(await tool("echo", lambda value: value, value=n))

    cases = [EvaluationCase(id=n, input=str(n), expected_output=str(n)) for n in range(1, 6)]
    trials = asyncio.run(run_experiment(FunctionAgent(agent, model="m"), cases, attempts=1, concurrency=5))
    for trial in trials:
        assert [span.output for span in trial.spans] == [trial.output]

# --- Loading an agent by name ---

def test_load_agent_wraps_functions_and_keeps_agent_objects(tmp_path, monkeypatch):
    module = tmp_path / "my_agents.py"
    module.write_text(
        "class Echo:\n"
        "    async def run(self, case, attempt, tracer):\n"
        "        return case.input\n"
        "echo = Echo()\n"
        "async def answer(question):\n"
        "    return question\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("my_agents", None)

    function_agent, source_hash = load_agent("my_agents:answer", "m")
    object_agent, _ = load_agent("my_agents:echo", "m")

    assert isinstance(function_agent, FunctionAgent)
    assert asyncio.run(object_agent.run(case, 1, Tracer(1))) == case.input
    assert len(source_hash) == 12

@pytest.mark.parametrize("spec", ["no_colon", "my_missing_module:x", "json:not_there", "json:__name__"])
def test_bad_agent_specs_say_what_is_wrong(spec):
    with pytest.raises((ValueError, ImportError)):
        load_agent(spec, "m")

def test_editing_the_agent_changes_its_version(tmp_path, monkeypatch):
    module = tmp_path / "versioned_agent.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    hashes = []
    for body in ("return 'a'", "return 'b'"):
        module.write_text(f"def answer(question):\n    {body}\n", encoding="utf-8")
        sys.modules.pop("versioned_agent", None)
        hashes.append(load_agent("versioned_agent:answer", "m")[1])
    assert hashes[0] != hashes[1]

# --- Your own dataset ---

def test_dataset_files_need_only_input_and_expected_output(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"input": "a?", "expected_output": "1"}\n\n{"id": 7, "input": "b?", "expected_output": "2"}\n',
                    encoding="utf-8")

    cases = load_dataset_file(path)

    assert [(c.id, c.input) for c in cases] == [(1, "a?"), (7, "b?")]  # id defaults to the line number
    assert dataset_by_name(str(path)) == cases  # how report, rescore and compare find it again

@pytest.mark.parametrize("content, message", [
    ('{"input": "a?"}\n', "line 1"),
    ('{"input": "a?", "expected_output": "1"}\nnot json\n', "line 2"),
    ('["a list"]\n', "line 1"),
    ('{"id": 1, "input": "a", "expected_output": "1"}\n{"id": 1, "input": "b", "expected_output": "2"}\n',
     "repeated: [1]"),
    ("\n", "no cases"),
])
def test_dataset_errors_point_at_the_line(tmp_path, content, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_dataset_file(path)

# --- End to end, through the command line ---

def run_cli(monkeypatch, *arguments):
    from agenteval import run

    monkeypatch.setattr(sys, "argv", ["agenteval.run", *arguments])
    run.main()

def test_the_quickstart_example_runs_and_reports(tmp_path, monkeypatch, capsys):
    db = tmp_path / "quickstart.db"
    run_cli(monkeypatch, "--agent", "examples.quickstart_agent:answer", "--model", "rules", "--price", "0", "0",
            "--dataset", "examples/quickstart_cases.jsonl", "--attempts", "1", "--db", str(db))

    conn = storage.connect(db)
    experiment = storage.load_experiment(conn, 1)
    trials, scores = storage.load_trials(conn, 1), storage.load_scores(conn, 1)
    assert experiment.config.params["agent"] == "examples.quickstart_agent:answer"
    assert experiment.prices["rules"].input_per_million == 0.0
    report = summarize(experiment, trials, scores, dataset_by_name(experiment.dataset_name))
    assert "Pass rate:        75.0%" in report  # two of eight are wrong on purpose

@pytest.mark.parametrize("arguments, message", [
    (["--agent", "examples.quickstart_agent:answer", "--dataset", "examples/quickstart_cases.jsonl"], "--model"),
    (["--agent", "examples.quickstart_agent:answer", "--model", "rules"], "--dataset"),
    (["--agent", "examples.quickstart_agent:answer", "--model", "rules", "--dataset",
      "examples/quickstart_cases.jsonl"], "--price"),
    (["--agent", "nope"], "Unknown agent"),
    (["--agent", "groq", "--dataset", "my_cases.txt"], "Unknown dataset"),
])
def test_the_cli_says_what_is_missing(monkeypatch, arguments, message):
    with pytest.raises(SystemExit, match=message):
        run_cli(monkeypatch, *arguments)

def test_models_without_a_price_are_recorded_at_zero_with_a_warning(capsys):
    from agenteval.models import Span, Trial
    from agenteval.run import prices_for

    args = SimpleNamespace(price=[1.0, 2.0], model="mine")
    trial = Trial(id=1, case_id=1, attempt=1, spans=[
        Span(trial_id=1, kind="llm", name="other-model", start=0, end=0, status="success"),
    ])

    prices = prices_for(args, [trial])

    assert prices["mine"].output_per_million == 2.0
    assert prices["other-model"].input_per_million == 0.0
    assert "no price for other-model" in capsys.readouterr().out
