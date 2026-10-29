from evalkit.models import EvaluationCase, Score, Trial

def exact_match(case: EvaluationCase, trial: Trial) -> Score:
    if trial.output is None:
        return Score(trial_id=trial.id, scorer="exact_match", value=0.0,
                     reasoning=f"no output ({trial.failure_type})")

    passed = trial.output.strip() == case.expected_output.strip()
    return Score(trial_id=trial.id, scorer="exact_match", value=1.0 if passed else 0.0,
                 reasoning=f"expected={case.expected_output!r}, got={trial.output!r}")
