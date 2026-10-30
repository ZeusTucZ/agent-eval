from pathlib import Path
from types import SimpleNamespace

from evalkit.models import AgentConfig, dataset_version
from evalkit.run import _system_prompt
from evalkit.sample_data import arithmetic_cases
from evalkit.tools import SYSTEM_PROMPT

def test_identical_configs_have_the_same_version():
    first = AgentConfig(model="mock-llm", prompt="Answer.", params={"temperature": 0.0, "seed": 1})
    second = AgentConfig(model="mock-llm", prompt="Answer.", params={"seed": 1, "temperature": 0.0})

    assert first.version == second.version

def test_any_change_gives_a_new_version():
    base = AgentConfig(model="mock-llm", prompt="Answer.")

    assert base.version != base.model_copy(update={"prompt": "Answer!"}).version
    assert base.version != base.model_copy(update={"model": "mock-llm-small"}).version
    assert base.version != base.model_copy(update={"params": {"temperature": 0.5}}).version

def test_dataset_version_changes_when_any_case_changes():
    cases = arithmetic_cases(3)
    changed = [cases[0], cases[1], cases[2].model_copy(update={"expected_output": "wrong"})]

    assert dataset_version(cases) == dataset_version(arithmetic_cases(3))
    assert dataset_version(cases) != dataset_version(changed)

def test_the_prompt_files_are_the_built_in_prompt_and_one_fix():
    prompts = Path(__file__).resolve().parent.parent / "prompts"

    assert (prompts / "v1.txt").read_text(encoding="utf-8").strip() == SYSTEM_PROMPT
    assert (prompts / "v2.txt").read_text(encoding="utf-8").strip() != SYSTEM_PROMPT

def test_prompt_file_overrides_the_built_in_prompt(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("Be brief.\n", encoding="utf-8")

    assert _system_prompt(SimpleNamespace(prompt_file=str(path))) == "Be brief."
    assert _system_prompt(SimpleNamespace(prompt_file=None)) == SYSTEM_PROMPT
