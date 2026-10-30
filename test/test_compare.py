import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from evalkit import compare, storage
from evalkit.compare import ExperimentData, check_comparable, compare_experiments, config_changes
from evalkit.cost import PRICES
from evalkit.mock_agent import FaultConfig, MockAgent
from evalkit.models import AgentConfig, EvaluationCase, Experiment, Score, Span, Trial, dataset_version
from evalkit.runner import run_experiment
from evalkit.sample_data import arithmetic_cases
from evalkit.scorers import score_trials

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
MS = 1_000_000

def make_cases(count: int) -> list[EvaluationCase]:
    return [EvaluationCase(id=i, input=f"q{i}", expected_output="x", category="odd" if i % 2 else "even")
            for i in range(1, count + 1)]

def make_data(
    results: dict[int, list[bool]],
    prompt: str = "A",
    tokens: int = 100,
    duration_ms: int = 1000,
    unsafe_cases: tuple[int, ...] = (),
    safety_checked: tuple[int, ...] = (),
    **experiment_fields,
) -> ExperimentData:
    """Build an experiment by hand: case id -> whether each attempt passed."""
    trials, scores = [], []
    for case_id, attempts in results.items():
        for attempt, passed in enumerate(attempts, start=1):
            trial_id = len(trials) + 1
            trials.append(Trial(id=trial_id, case_id=case_id, attempt=attempt, output="x" if passed else "wrong", spans=[
                Span(trial_id=trial_id, kind="llm", name="mock-llm", start=0, end=duration_ms * MS,
                     status="success", input_tokens=tokens, output_tokens=tokens),
            ]))
            scores.append(Score(trial_id=trial_id, scorer="exact_match", value=1.0 if passed else 0.0))
            if case_id in safety_checked:
                scores.append(Score(trial_id=trial_id, scorer="avoids_forbidden_text",
                                    value=0.0 if case_id in unsafe_cases else 1.0))

    fields = {"name": "run", "dataset_name": "test", "dataset_version": "abc", "attempts": 1,
              "config": AgentConfig(model="mock-llm", prompt=prompt), "prices": PRICES, "created_at": NOW}
    return ExperimentData(Experiment(**(fields | experiment_fields)), trials, scores)

def outcomes(passing: int, total: int = 20) -> dict[int, list[bool]]:
    """Cases 1..passing pass; the rest fail."""
    return {i: [i <= passing] for i in range(1, total + 1)}

# --- Fairness ---

def test_different_datasets_are_refused():
    fairness = check_comparable(make_data({}), make_data({}, dataset_name="other"))
    assert "different datasets" in fairness.errors[0]

def test_a_changed_dataset_is_refused():
    fairness = check_comparable(make_data({}), make_data({}, dataset_version="def"))
    assert "changed between runs" in fairness.errors[0]

def test_old_experiments_without_a_dataset_version_only_warn():
    fairness = check_comparable(make_data({}, dataset_version=None), make_data({}))
    assert not fairness.errors
    assert any("can't verify" in warning for warning in fairness.warnings)

def test_config_changes_are_listed_including_nested_settings():
    a = AgentConfig(model="m", prompt="p", params={"temperature": 0.0, "retry": {"max": 3}})
    b = AgentConfig(model="m", prompt="q", params={"temperature": 0.5, "retry": {"max": 3}})
    assert config_changes(a, b) == ["params.temperature", "prompt"]

@pytest.mark.parametrize("b_fields, expected", [
    ({"prompt": "B", "model": "other"}, "2 settings changed"),
    ({"attempts": 3}, "different attempts"),
    ({"created_at": NOW + timedelta(hours=25)}, "over 24 hours apart"),
])
def test_unfair_conditions_warn(b_fields, expected):
    prompt = b_fields.pop("prompt", "B")
    model = b_fields.pop("model", "mock-llm")
    b = make_data({}, prompt=prompt, config=AgentConfig(model=model, prompt=prompt), **b_fields)
    assert any(expected in warning for warning in check_comparable(make_data({}), b).warnings)

def test_identical_configs_warn_that_this_only_measures_noise():
    assert any("identical" in warning for warning in check_comparable(make_data({}), make_data({})).warnings)

def test_infrastructure_failures_warn_but_model_failures_do_not():
    a = make_data({1: [True]})
    a.trials[0].failure_type = "quota_exhausted"
    b = make_data({1: [True]}, prompt="B")
    b.trials[0].failure_type = "output_truncated"

    warnings = check_comparable(a, b).warnings
    assert any("1 of 1 trials in A" in warning for warning in warnings)
    assert not any("trials in B" in warning for warning in warnings)

# --- Comparing ---

def test_counts_improved_and_regressed_cases():
    a = make_data({1: [False], 2: [False], 3: [True], 4: [True]})
    b = make_data({1: [True], 2: [False], 3: [False], 4: [True]}, prompt="B")
    result = compare_experiments(a, b, make_cases(4))

    assert [change.case.id for change in result.improved] == [1]
    assert [change.case.id for change in result.regressed] == [3]
    assert result.pass_rate.a == result.pass_rate.b == 0.5
    assert result.p_value == 1.0

def test_six_improved_and_one_regressed_is_the_hand_calculated_p_value():
    a = make_data({i: [i == 7] for i in range(1, 8)})
    b = make_data({i: [i != 7] for i in range(1, 8)}, prompt="B")
    result = compare_experiments(a, b, make_cases(7))
    assert (len(result.improved), len(result.regressed), result.p_value) == (6, 1, 0.125)

def test_attempts_are_combined_into_one_result_per_case():
    # 2 of 3 attempts passing counts as a pass, and one case is one unit however many attempts it has.
    a = make_data({1: [True, False, False]}, attempts=3)
    b = make_data({1: [True, True, False]}, attempts=3, prompt="B")
    result = compare_experiments(a, b, make_cases(1))

    assert len(result.improved) == 1
    assert result.pass_rate.a == pytest.approx(1 / 3)
    assert result.pass_rate.b == pytest.approx(2 / 3)

def test_a_tied_case_is_not_a_clean_flip():
    # 1 of 2 -> 2 of 2 moved, but the baseline has no majority, so it isn't evidence of a flip.
    a = make_data({1: [True, False], 2: [False, False]}, attempts=2)
    b = make_data({1: [True, True], 2: [True, True]}, attempts=2, prompt="B")
    result = compare_experiments(a, b, make_cases(2))

    assert [change.case.id for change in result.improved] == [2]
    assert result.regressed == []
    assert result.partly_changed == 1

def test_ties_cannot_mask_a_clear_drop():
    # The shape of a real false pass: 16 clean regressions, 3 clean improvements and 3 ties.
    a_results = {i: [True, True] for i in range(1, 17)}          # regress
    a_results |= {i: [False, False] for i in range(17, 20)}      # clean improvements
    a_results |= {i: [True, False] for i in range(20, 23)}       # ties that improve
    a_results |= {i: [True, True] for i in range(23, 56)}        # unchanged
    b_results = {i: [False, False] for i in range(1, 17)}
    b_results |= {i: [True, True] for i in range(17, 23)}
    b_results |= {i: [True, True] for i in range(23, 56)}

    result = compare_experiments(
        make_data(a_results, attempts=2), make_data(b_results, attempts=2, prompt="B"), make_cases(55),
    )

    assert (len(result.improved), len(result.regressed), result.partly_changed) == (3, 16, 3)
    assert result.p_value == pytest.approx(2 * (1 + 19 + 171 + 969) / 2 ** 19)  # hand-calculated
    assert result.verdict == "Worse"

def test_only_ties_moving_is_still_reported_as_a_change():
    a = make_data({1: [True, False]}, attempts=2)
    b = make_data({1: [True, True]}, attempts=2, prompt="B")
    result = compare_experiments(a, b, make_cases(1))
    assert result.partly_changed == 1
    assert "no case changed" not in result.reason

def test_more_attempts_do_not_shrink_the_interval_when_they_agree():
    # Repeating identical results three times adds no information, so the interval must not narrow.
    once = compare_experiments(make_data(outcomes(10)), make_data(outcomes(15), prompt="B"), make_cases(20))
    thrice = compare_experiments(
        make_data({i: passed * 3 for i, passed in outcomes(10).items()}, attempts=3),
        make_data({i: passed * 3 for i, passed in outcomes(15).items()}, attempts=3, prompt="B"),
        make_cases(20),
    )
    assert thrice.pass_rate.change == once.pass_rate.change

def test_only_cases_in_both_experiments_are_compared():
    a = make_data({1: [True], 2: [True], 3: [True]})
    b = make_data({2: [True], 3: [True], 4: [True]}, prompt="B")
    assert compare_experiments(a, b, make_cases(4)).cases_compared == 2

def test_experiments_with_nothing_in_common_are_an_error():
    with pytest.raises(ValueError, match="no scored cases"):
        compare_experiments(make_data({1: [True]}), make_data({2: [True]}, prompt="B"), make_cases(2))

def test_category_rates():
    a = make_data({1: [False], 2: [True], 3: [False], 4: [True]})
    b = make_data({1: [True], 2: [True], 3: [True], 4: [True]}, prompt="B")
    result = compare_experiments(a, b, make_cases(4))
    assert result.categories == {"odd": (0.0, 1.0, 2), "even": (1.0, 1.0, 2)}

# --- Verdicts ---

def test_verdict_better():
    result = compare_experiments(make_data(outcomes(5)), make_data(outcomes(18), prompt="B"), make_cases(20))
    assert result.verdict == "Better"

def test_verdict_worse():
    result = compare_experiments(make_data(outcomes(18)), make_data(outcomes(5), prompt="B"), make_cases(20))
    assert result.verdict == "Worse"

def test_a_big_looking_gain_can_still_be_inconclusive():
    a = make_data({i: [i >= 7] for i in range(1, 56)})  # 49 of 55 pass
    b_results = {i: [True] for i in range(1, 56)}
    b_results[7] = [False]  # 54 of 55 pass: 6 improved, 1 regressed
    result = compare_experiments(a, make_data(b_results, prompt="B"), make_cases(55))

    assert result.pass_rate.change.mean == pytest.approx(5 / 55)
    assert result.verdict == "Inconclusive"
    assert "plausibly between" in result.reason

def test_no_changes_at_all_says_the_dataset_cannot_separate_them():
    result = compare_experiments(make_data(outcomes(20)), make_data(outcomes(20), prompt="B"), make_cases(20))
    assert result.verdict == "Inconclusive"
    assert "no case changed" in result.reason

def test_better_but_much_costlier_is_a_trade_off():
    a = make_data(outcomes(5), tokens=100)
    b = make_data(outcomes(18), tokens=300, prompt="B")
    result = compare_experiments(a, b, make_cases(20))
    assert result.verdict == "Trade-off"
    assert "cost up 200%" in result.reason

def test_better_but_slower_is_a_trade_off():
    a = make_data(outcomes(5), duration_ms=1000)
    b = make_data(outcomes(18), duration_ms=2000, prompt="B")
    assert compare_experiments(a, b, make_cases(20)).verdict == "Trade-off"

def test_a_large_percentage_of_almost_nothing_does_not_spoil_a_win():
    a = make_data(outcomes(5), duration_ms=1)
    b = make_data(outcomes(18), duration_ms=2, prompt="B")  # "100% slower", but only 1 ms
    assert compare_experiments(a, b, make_cases(20)).verdict == "Better"

def test_a_tiny_cost_rise_does_not_spoil_a_win():
    a = make_data(outcomes(5), tokens=100)
    b = make_data(outcomes(18), tokens=105, prompt="B")
    assert compare_experiments(a, b, make_cases(20)).verdict == "Better"

def test_better_score_with_a_new_safety_failure_is_a_trade_off():
    a = make_data(outcomes(5), safety_checked=(20,))
    b = make_data(outcomes(18), prompt="B", safety_checked=(20,), unsafe_cases=(20,))
    result = compare_experiments(a, b, make_cases(20))

    assert [change.case.id for change in result.safety_regressions] == [20]
    assert result.verdict == "Trade-off"
    assert "1 new safety failure" in result.reason

# --- With the mock agent, where the true difference is known ---

def mock_run(cases, seed: int, hallucination_rate: float) -> ExperimentData:
    faults = FaultConfig(tool_error_rate=0.0, rate_limit_rate=0.0, hallucination_rate=hallucination_rate,
                         latency_ms=(0, 0), seed=seed)
    trials = asyncio.run(run_experiment(MockAgent(faults), cases, attempts=1, concurrency=50))
    experiment = Experiment(
        name="mock", dataset_name="test", dataset_version=dataset_version(cases), attempts=1, prices=PRICES,
        config=AgentConfig(model="mock-llm", prompt="p", params={"hallucination_rate": hallucination_rate}),
        created_at=NOW,
    )
    return ExperimentData(experiment, trials, score_trials(cases, trials))

def test_a_real_improvement_is_detected():
    cases = arithmetic_cases(200)
    result = compare_experiments(mock_run(cases, 1, 0.30), mock_run(cases, 2, 0.05), cases)
    assert result.verdict == "Better"

def test_a_real_regression_is_detected():
    cases = arithmetic_cases(200)
    result = compare_experiments(mock_run(cases, 1, 0.05), mock_run(cases, 2, 0.30), cases)
    assert result.verdict == "Worse"

def test_identical_versions_rarely_look_different():
    # An A/A test: the same config with different seeds. At a 5% significance level, about
    # 1 run in 20 may look significant by chance, so a few false alarms are expected, not many.
    cases = arithmetic_cases(100)
    false_alarms = sum(
        compare_experiments(mock_run(cases, seed, 0.2), mock_run(cases, seed + 1000, 0.2), cases, resamples=500).verdict
        != "Inconclusive"
        for seed in range(40)
    )
    assert false_alarms <= 5, f"{false_alarms} of 40 identical comparisons looked significant"

# --- The command ---

def saved(conn, data: ExperimentData) -> int:
    return storage.save_experiment(conn, data.experiment, data.trials, data.scores)

def run_main(monkeypatch, capsys, path, a: int, b: int) -> str:
    monkeypatch.setattr("sys.argv", ["compare", str(a), str(b), "--db", str(path)])
    compare.main()
    return capsys.readouterr().out

def test_command_prints_a_comparison(tmp_path, monkeypatch, capsys):
    cases = arithmetic_cases(50)
    path = tmp_path / "eval.db"
    conn = storage.connect(path)

    def real(seed, rate):
        data = mock_run(cases, seed, rate)
        data.experiment.dataset_name = "arithmetic-50"
        return saved(conn, data)

    output = run_main(monkeypatch, capsys, path, real(1, 0.5), real(2, 0.05))

    assert "Differs in:   params.hallucination_rate" in output
    assert "McNemar:" in output
    assert "Improved (" in output
    assert "Verdict: Better." in output

def test_command_refuses_unfair_comparisons(tmp_path, monkeypatch, capsys):
    path = tmp_path / "eval.db"
    conn = storage.connect(path)
    a = saved(conn, make_data({1: [True]}))
    b = saved(conn, make_data({1: [True]}, dataset_name="other"))

    with pytest.raises(SystemExit, match="different datasets"):
        run_main(monkeypatch, capsys, path, a, b)
