"""agent-eval: run an agent many times, trace every step, and measure what matters.

To evaluate your own agent, wrap its steps with these and see evalkit.sdk:

    from evalkit import llm_call, tool
"""
from evalkit.sdk import ToolArgumentError, llm_call, tool

__all__ = ["ToolArgumentError", "llm_call", "tool"]
