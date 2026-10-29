from random import Random

from dataclasses import dataclass

from datasets.models import EvaluationCase, Span, Trial

@dataclass
class FaultConfig:
    tool_error_rate: float = 0.1
    rate_limit_rate: float = 0.05
    hallucination_rate: float = 0.15
    latency_ms: tuple[int, int] = (50, 300)
    seed: int = 42

class MockAgent:
    def __init__(self, config: FaultConfig):
        self.config = config
        self.random = Random(config.seed)

    def run(self, dataset: list[EvaluationCase]) -> list[Trial]:
        trials = []

        for trial_id, case in enumerate(dataset, start=1):
            tool_failed = self.random.random() < self.config.tool_error_rate

            span = Span(
                trial_id=trial_id,
                kind="tool",
                name=case.expected_tool,
                start=0,
                end=0,
                status="error" if tool_failed else "success",
                error_type="tool_error" if tool_failed else None,
                input_tokens=0,
                output_tokens=0,
            )

            trial = Trial(
                id=trial_id,
                case_id=case.id,
                attempt=1,
                output="" if tool_failed else case.expected_output,
                failure_type="tool_error" if tool_failed else None,
                spans=[span],
            )

            trials.append(trial)

        return trials
    
