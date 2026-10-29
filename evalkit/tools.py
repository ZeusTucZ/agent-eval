from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from evalkit.tracer import Tracer

class UnknownToolError(Exception):
    pass

class ToolArgumentError(Exception):
    pass

@dataclass
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]  # validates the model's arguments and describes them to the model
    function: Callable[..., Any]

class ToolRegistry:
    """The only way an agent calls tools, so every call is traced and validated."""

    def __init__(self, tools: list[Tool]):
        self.tools = {tool.name: tool for tool in tools}

    def call(self, name: str, arguments: dict[str, Any], tracer: Tracer) -> Any:
        with tracer.span(kind="tool", name=name):
            tool = self.tools.get(name)
            if tool is None:
                raise UnknownToolError(f"no tool named {name!r}")

            try:
                parsed = tool.arguments.model_validate(arguments)
            except ValidationError as error:
                raise ToolArgumentError(str(error)) from error

            return tool.function(**parsed.model_dump())

# --- The tools the real agent can use ---

class CalculatorArguments(BaseModel):
    a: float
    operator: Literal["+", "-", "*", "/"]
    b: float

def calculate(a: float, operator: str, b: float) -> float:
    if operator == "+":
        result = a + b
    elif operator == "-":
        result = a - b
    elif operator == "*":
        result = a * b
    else:
        result = a / b
    return int(result) if result == int(result) else result

class CapitalArguments(BaseModel):
    country: str

# A deliberately small "database": the point is to see whether the model looks
# answers up or answers from memory.
CAPITALS = {
    "france": "Paris",
    "japan": "Tokyo",
    "brazil": "Brasília",
    "canada": "Ottawa",
    "australia": "Canberra",
    "nigeria": "Abuja",
    "mexico": "Mexico City",
    "kazakhstan": "Astana",
}

def lookup_capital(country: str) -> str:
    capital = CAPITALS.get(country.strip().lower())
    if capital is None:
        raise LookupError(f"no capital on record for {country!r}")
    return capital

DEFAULT_TOOLS = [
    Tool(
        name="calculator",
        description="Exact arithmetic on two numbers. Use for any calculation.",
        arguments=CalculatorArguments,
        function=calculate,
    ),
    Tool(
        name="lookup_capital",
        description="Look up the capital city of a country in the reference database.",
        arguments=CapitalArguments,
        function=lookup_capital,
    ),
]
