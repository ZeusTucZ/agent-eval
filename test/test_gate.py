import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evalkit import gate, storage
from evalkit.compare import ExperimentData, check_comparable, compare_experiments
from evalkit.cost import PRICES
from evalkit.gate import BLOCK, CANNOT_JUDGE, COMMENT_MARKER, PASS, decide, format_markdown, run_gate
from evalkit.mock_agent import FaultConfig, MockAgent
from evalkit.models import AgentConfig, Experiment, dataset_version
from evalkit.runner import run_experiment
from evalkit.sample_data import arithmetic_cases
from evalkit.scorers import score_trials
from evalkit.tracer import Tracer
from test_compare import make_cases, make_data, outcomes

def decision_for(a: ExperimentData, b: ExperimentData, count: int = 20):
    fairness = check_comparable(a, b)
    result = None if fairness.errors else compare_experiments(a, b, make_cases(count), resamples=500)
    return decide(a, b, fairness, result)

# --- The decision table ---

def test_a_significant_improvement_passes():
    assert decision_for(make_data(outcomes(5)), make_data(outcomes(18), prompt="B")).exit_code == PASS

def test_a_significant_drop_blocks():
    assert decision_for(make_data(outcomes(18)), make_data(outcomes(5), prompt="B")).exit_code == BLOCK

def test_inconclusive_passes_but_says_it_proves_nothing():
    decision = decision_for(make_data(outcomes(10)), make_data(outcomes(11), prompt="B"))
    assert decision.exit_code == PASS
    assert "doesn't prove the change is safe" in decision.reasons[0]

def test_costlier_but_better_passes_with_the_trade_off_noted():
    decision = decision_for(make_data(outcomes(5), tokens=100), make_data(outcomes(18), tokens=300, prompt="B"))
    assert decision.exit_code == PASS
    assert "cost up" in decision.reasons[0]

def test_a_new_safety_failure_blocks_even_when_the_score_rises():
    a = make_data(outcomes(5), safety_checked=(20,))
    b = make_data(outcomes(18), prompt="B", safety_checked=(20,), unsafe_cases=(20,))
    decision = decision_for(a, b)
    assert decision.exit_code == BLOCK
    assert "1 new safety failure" in decision.reasons[0]

def test_a_new_safety_failure_blocks_even_when_inconclusive():
    a = make_data(outcomes(10), safety_checked=(1,))
    b = make_data(outcomes(10), prompt="B", safety_checked=(1,), unsafe_cases=(1,))
    assert decision_for(a, b).exit_code == BLOCK

def test_different_datasets_cannot_be_judged():
    decision = decision_for(make_data(outcomes(10)), make_data(outcomes(10), dataset_version="other"))
    assert decision.exit_code == CANNOT_JUDGE
    assert "changed between runs" in decision.reasons[0]

def test_a_run_damaged_by_infrastructure_cannot_be_judged():
    # Even a run that looks much worse says nothing if rate limits caused it.
    b = make_data(outcomes(5), prompt="B")
    for trial in b.trials[:3]:
        trial.failure_type = "quota_exhausted"

    decision = decision_for(make_data(outcomes(18)), b)
    assert decision.exit_code == CANNOT_JUDGE
    assert "3 of 20 candidate trials" in decision.reasons[0]

def test_a_few_infrastructure_failures_are_tolerated():
    b = make_data(outcomes(18), prompt="B")
    b.trials[0].failure_type = "timeout"
    assert decision_for(make_data(outcomes(5)), b).exit_code == PASS

# --- The report ---

def report(a: ExperimentData, b: ExperimentData) -> str:
    fairness = check_comparable(a, b)
    result = compare_experiments(a, b, make_cases(len(a.trials)), resamples=500)
    return format_markdown(a, fairness, result, decide(a, b, fairness, result))

def test_markdown_report_has_the_marker_decision_and_cases():
    markdown = report(make_data(outcomes(18)), make_data(outcomes(5), prompt="B"))

    assert markdown.startswith(COMMENT_MARKER)
    assert "Blocked" in markdown
    assert "**Verdict: Worse**" in markdown
    assert "| Pass rate | 90.0% | 25.0% | -65.0 pts |" in markdown
    assert "**Regressed (13)**" in markdown

@pytest.mark.parametrize("output, expected", [
    ("Paris", "`Paris`"),
    ("a | b", "`a / b`"),                 # a pipe would break the table
    ("use `rm -rf`", "`use 'rm -rf'`"),   # a backtick would escape the code span
    ("line one\n\n## Heading", "`line one ## Heading`"),
    (None, "*no answer*"),
    ("", "*empty*"),
    ("x" * 100, f"`{'x' * 80}...`"),
])
def test_model_output_cannot_act_as_markdown_in_a_comment(output, expected):
    assert gate._inline(output) == expected

# --- End to end, with real databases ---

def save_mock_run(path, hallucination_rate: float, seed: int, count: int = 200) -> None:
    cases = arithmetic_cases(count)
    faults = FaultConfig(tool_error_rate=0.0, rate_limit_rate=0.0, hallucination_rate=hallucination_rate,
                         latency_ms=(0, 0), seed=seed)
    trials = asyncio.run(run_experiment(MockAgent(faults), cases, attempts=1, concurrency=50))
    experiment = Experiment(
        name="mock", dataset_name=f"arithmetic-{count}", dataset_version=dataset_version(cases), attempts=1,
        prices=PRICES, created_at=datetime.now(timezone.utc),
        config=AgentConfig(model="mock-llm", prompt="p", params={"hallucination_rate": hallucination_rate}),
    )
    storage.save_experiment(storage.connect(path), experiment, trials, score_trials(cases, trials))

def test_a_known_bad_version_is_blocked_end_to_end(tmp_path, monkeypatch, capsys):
    baseline, candidate, report_path = tmp_path / "a.db", tmp_path / "b.db", tmp_path / "report.md"
    save_mock_run(baseline, 0.05, seed=1)
    save_mock_run(candidate, 0.40, seed=2)

    monkeypatch.setattr("sys.argv", ["gate", str(baseline), str(candidate), "--markdown", str(report_path)])
    with pytest.raises(SystemExit) as exit_info:
        gate.main()

    assert exit_info.value.code == BLOCK
    assert "Blocked" in capsys.readouterr().out
    assert report_path.read_text(encoding="utf-8").startswith(COMMENT_MARKER)

def test_a_good_version_passes_end_to_end(tmp_path):
    baseline, candidate = tmp_path / "a.db", tmp_path / "b.db"
    save_mock_run(baseline, 0.40, seed=1)
    save_mock_run(candidate, 0.05, seed=2)

    decision, markdown, _ = run_gate(str(baseline), str(candidate))
    assert decision.exit_code == PASS
    assert "Pass" in markdown

def test_the_latest_experiment_in_each_database_is_used(tmp_path):
    baseline, candidate = tmp_path / "a.db", tmp_path / "b.db"
    save_mock_run(baseline, 0.05, seed=1)
    save_mock_run(candidate, 0.05, seed=2)
    save_mock_run(candidate, 0.40, seed=3)  # the newest run is the bad one

    assert run_gate(str(baseline), str(candidate))[0].exit_code == BLOCK

def test_a_run_that_saved_nothing_cannot_be_judged(tmp_path):
    baseline, candidate = tmp_path / "a.db", tmp_path / "b.db"
    save_mock_run(baseline, 0.05, seed=1)

    decision, markdown, _ = run_gate(str(baseline), str(candidate))
    assert decision.exit_code == CANNOT_JUDGE
    assert "no saved experiment for: candidate" in decision.reasons[0]
    assert "Can't judge" in markdown

# --- The workflows ---

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"

def run_settings(name: str) -> dict[str, str]:
    env = (WORKFLOWS / name).read_text(encoding="utf-8").split("\nenv:\n")[1].split("\n\n")[0]
    return {line.split(":")[0].strip(): line.split(":")[1].split("#")[0].strip() for line in env.splitlines()}

def test_the_baseline_and_the_check_run_with_the_same_settings():
    # If these drift apart, every comparison quietly stops being fair.
    assert run_settings("agent-eval.yml") == run_settings("agent-eval-baseline.yml")
    assert int(run_settings("agent-eval.yml")["ATTEMPTS"]) % 2 == 1  # odd, so cases can't tie

def test_a_damaged_baseline_fails_when_compared_with_itself(tmp_path):
    # The baseline workflow relies on this to avoid publishing a run spoiled by rate limits.
    path = tmp_path / "baseline.db"
    save_mock_run(path, 0.05, seed=1)
    with storage.connect(path) as conn:
        conn.execute("UPDATE trials SET failure_type = 'quota_exhausted'")

    assert run_gate(str(path), str(path))[0].exit_code == CANNOT_JUDGE

def test_a_healthy_baseline_passes_when_compared_with_itself(tmp_path):
    path = tmp_path / "baseline.db"
    save_mock_run(path, 0.05, seed=1)
    assert run_gate(str(path), str(path))[0].exit_code == PASS

def test_account_ids_are_redacted_from_recorded_text():
    tracer = Tracer(1)
    with pytest.raises(ValueError):
        with tracer.span(kind="llm", name="m"):
            raise ValueError("Rate limit reached in organization `org_fake123example` on tokens per day")

    message = tracer.spans[0].error_message
    assert "org_fake123" not in message
    assert "org_REDACTED" in message
    assert "tokens per day" in message
