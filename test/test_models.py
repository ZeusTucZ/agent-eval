from evalkit.models import AgentConfig

def test_identical_configs_have_the_same_version():
    first = AgentConfig(model="mock-llm", prompt="Answer.", params={"temperature": 0.0, "seed": 1})
    second = AgentConfig(model="mock-llm", prompt="Answer.", params={"seed": 1, "temperature": 0.0})

    assert first.version == second.version

def test_any_change_gives_a_new_version():
    base = AgentConfig(model="mock-llm", prompt="Answer.")

    assert base.version != base.model_copy(update={"prompt": "Answer!"}).version
    assert base.version != base.model_copy(update={"model": "mock-llm-small"}).version
    assert base.version != base.model_copy(update={"params": {"temperature": 0.5}}).version
