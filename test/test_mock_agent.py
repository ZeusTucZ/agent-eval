from datasets.mock_agent import FaultConfig, MockAgent
from datasets.models import EvaluationCase
from datasets.report import tool_error_rate

dataset_of_500_cases = [
    EvaluationCase(
        id=index,
        input=f"What is {index} + 1",
        expected_output=str(index + 1),
        expected_tool="calculator",
    )
    for index in range(1, 501)
]

def test_tool_error_rate_matches_injected():
    config = FaultConfig(tool_error_rate=0.10)
    agent = MockAgent(config)

    trials = agent.run(dataset_of_500_cases)

