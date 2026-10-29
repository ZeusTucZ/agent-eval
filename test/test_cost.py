import pytest

from evalkit.cost import span_cost, total_cost, trial_cost
from evalkit.models import Price, Span, Trial

PRICES = {"big-model": Price(input_per_million=3.00, output_per_million=15.00)}

def llm_span(input_tokens: int, output_tokens: int, model: str = "big-model") -> Span:
    return Span(trial_id=1, kind="llm", name=model, start=0, end=0, status="success",
                input_tokens=input_tokens, output_tokens=output_tokens)

def test_span_cost_uses_input_and_output_prices():
    # 1,000 x $3/1M + 500 x $15/1M = $0.003 + $0.0075
    assert span_cost(llm_span(1000, 500), PRICES) == pytest.approx(0.0105)

def test_tool_spans_cost_nothing():
    tool = Span(trial_id=1, kind="tool", name="calculator", start=0, end=0, status="success")
    assert span_cost(tool, PRICES) == 0.0

def test_trial_and_total_cost_add_up_spans():
    trial = Trial(id=1, case_id=1, attempt=1, spans=[llm_span(1000, 500), llm_span(1000, 500)])

    assert trial_cost(trial, PRICES) == pytest.approx(0.021)
    assert total_cost([trial, trial], PRICES) == pytest.approx(0.042)

def test_unknown_model_is_an_error_not_free():
    with pytest.raises(KeyError, match="unknown-model"):
        span_cost(llm_span(1000, 500, model="unknown-model"), PRICES)
