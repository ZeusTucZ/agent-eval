# Concepts

| Concept | What it is |
|---|---|
| **Case** | One question, its expected answer, and optionally the tools it needs, its category, text the answer must never contain, and the documents it is answered from. |
| **Trial** | One attempt at one case. Each case runs several times (`--attempts`, 3 by default), because LLMs don't answer the same way every time. |
| **Span** (step) | One model call or tool call inside a trial: start and end time, tokens, status, error type and message, and for tools the arguments and result. |
| **Score** | A judgement about a trial by one scorer, from 0 to 1, with its reasoning. Scores are stored apart from trials, so scorers can be fixed or added and re-run without calling the model again (`agenteval rescore`). |
| **Experiment** | One run of one agent configuration over one dataset: its trials, scores, prices at the time, status and the environment it ran in. |
| **Version** | A hash of the agent's configuration (model, prompt, parameters, and for your own agent the hash of its source file). Any change gives a new version, so comparisons know what changed. |

## What happens in a run

1. The runner starts each trial (case × attempt) with a timeout, a few at a time
   (`--concurrency`).
2. Your agent answers. Every `llm_call` and `tool` becomes a span.
3. When a trial finishes, the scorers judge it, and the trial and its scores are **saved at
   once**. A run cut short keeps everything already done and can be resumed with `--resume`.
4. If the trial ended on an exception, it gets a **failure type** from the step that raised:
   `tool_error`, `llm_error`, `timeout`, `timeout_after_rate_limit`, `quota_exhausted`,
   `max_steps_exceeded`, `output_truncated`, `empty_output`, `invalid_model_output`.

## Failures vs wrong answers

A **failure** stopped the trial: it never produced an answer. A **wrong answer** finished
but didn't match the expected one. They need different fixes (infrastructure and error
handling vs the model, prompt or tools), so reports and the dashboard always show them apart.

## Infrastructure vs model

A rate limit or an exhausted daily quota says something about the API that day, not about
the agent. agenteval labels them separately, and the gate refuses to judge a change when
more than 10% of trials failed for infrastructure reasons.
