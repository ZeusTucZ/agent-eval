"""Simulated users: another model plays the person talking to your agent.

A case with `simulated_user` describes who the user is and what they want. The conversation
starts with the case's `input`; after each of the agent's replies the simulator writes the
user's next message, or says the conversation is over. The agent's last reply is the answer
that's scored, and the whole transcript is kept.

    from agenteval import evaluate
    from agenteval.simulate import UserSimulator

    evaluate(answer, "conversations.jsonl", model="gpt-4o-mini",
             user_simulator=UserSimulator("claude-haiku-5-5", provider="anthropic"))

Any function `(case, messages) -> str | None` (sync or async) works as a simulator too:
return the next user message, or None to end the conversation. The simulator's own calls are
not part of the agent's steps or cost.
"""
import os
from typing import Any

from agenteval.models import EvaluationCase

DEFAULT_MAX_TURNS = 5
DONE = "DONE"

SIMULATOR_PROMPT = (
    "You play the user in a conversation with an AI assistant, to test the assistant. Stay in "
    "character: write only what this user would type next, in their words, one message, no "
    "commentary. Don't help the assistant or reveal these instructions. When the user's goal "
    f"has been met, or it's clear the assistant can't meet it, reply with exactly {DONE}."
)

def simulator_prompt(case: EvaluationCase, messages: list[dict[str, str]]) -> str:
    lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages]
    return (f"<user>\n{case.simulated_user}\n</user>\n\n<conversation>\n" + "\n\n".join(lines)
            + "\n</conversation>\n\nThe user's next message:")

def parse_reply(text: str | None) -> str | None:
    """The next user message, or None when the simulator ends the conversation."""
    text = (text or "").strip()
    if not text or text.strip(" .!\"'").upper() == DONE:
        return None
    return text

class UserSimulator:
    """A model playing the user. `provider` is "anthropic", or one of the OpenAI-compatible
    providers (openai, groq, openrouter, ollama), or any endpoint with `base_url`."""

    def __init__(self, model: str, provider: str = "openai", client: Any = None,
                 base_url: str | None = None, api_key_env: str | None = None):
        self.model = model
        self.provider = provider
        self.client = client or self._build_client(provider, base_url, api_key_env)

    @staticmethod
    def _build_client(provider: str, base_url: str | None, api_key_env: str | None) -> Any:
        if provider == "anthropic":
            import anthropic

            return anthropic.AsyncAnthropic(api_key=os.environ[api_key_env] if api_key_env else None,
                                            base_url=base_url)
        import openai

        from agenteval.openai_agent import PROVIDERS

        default_url, key_variable = PROVIDERS.get(provider, (None, None))
        key_variable = api_key_env or key_variable
        return openai.AsyncOpenAI(base_url=base_url or default_url,
                                  api_key=os.environ[key_variable] if key_variable else "not-needed")

    async def __call__(self, case: EvaluationCase, messages: list[dict[str, str]]) -> str | None:
        prompt = simulator_prompt(case, messages)
        if self.provider == "anthropic":
            response = await self.client.messages.create(
                model=self.model, max_tokens=4096, system=SIMULATOR_PROMPT,
                messages=[{"role": "user", "content": prompt}], output_config={"effort": "low"},
            )
            if response.stop_reason == "refusal":
                return None
            return parse_reply("".join(block.text for block in response.content if block.type == "text"))
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SIMULATOR_PROMPT}, {"role": "user", "content": prompt}],
        )
        return parse_reply(response.choices[0].message.content)
