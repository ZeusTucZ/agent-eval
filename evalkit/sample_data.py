from evalkit.models import EvaluationCase

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
