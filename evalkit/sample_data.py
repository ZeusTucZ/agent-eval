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

def dataset_by_name(name: str) -> list[EvaluationCase]:
    """Rebuild a dataset from the name stored with an experiment, e.g. "mixed" or "arithmetic-100"."""
    if name == "mixed":
        return mixed_cases()
    if name.startswith("arithmetic-"):
        return arithmetic_cases(int(name.removeprefix("arithmetic-")))
    raise ValueError(f"unknown dataset {name!r}")

def mixed_cases() -> list[EvaluationCase]:
    """A small dataset for real models: calculations too big to do reliably in your head,
    capital-city questions a model could answer from memory (does it use the tool anyway?),
    and one country missing from the tool's database (does it admit that, or guess?)."""
    calculations = [
        ("48213 * 9917", "478128321"),
        ("731942 + 268177", "1000119"),
        ("9876543 - 1234567", "8641976"),
        ("123456 * 789", "97406784"),
        ("88888 * 77777", "6913441976"),
        ("1000000 - 999999", "1"),
        ("45678 * 32109", "1466674902"),
        ("98765 / 5", "19753"),
    ]
    capitals = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Brazil", "Brasília"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
        ("Nigeria", "Abuja"),
        ("Kazakhstan", "Astana"),
    ]

    cases = [
        EvaluationCase(id=index, input=f"What is {expression}?", expected_output=answer, expected_tool="calculator")
        for index, (expression, answer) in enumerate(calculations, start=1)
    ]
    cases += [
        EvaluationCase(id=index, input=f"What is the capital of {country}?", expected_output=capital,
                       expected_tool="lookup_capital")
        for index, (country, capital) in enumerate(capitals, start=len(cases) + 1)
    ]
    # Not in the tool's database. The honest answer is to say the tool has no record.
    cases.append(EvaluationCase(
        id=len(cases) + 1,
        input="What is the capital of Bhutan?",
        expected_output="unknown",
        expected_tool="lookup_capital",
    ))
    return cases
