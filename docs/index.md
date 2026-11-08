# agenteval

**Run your LLM agent many times, trace every step, and find out why it fails.**

An agent that answers well in a demo can still be slow, expensive, flaky, or right for the
wrong reason: it answered from memory instead of using its tools, or a rate limit made half
its runs fail. A single accuracy number hides all of that.

agenteval runs your agent over a set of questions, several times each, records every model
call and tool call, and tells you:

- **how often it's right**, overall and by kind of question;
- **why it fails**: a wrong answer, a tool error, a timeout, a rate limit, a quota, an
  injected instruction it obeyed. Each failure is labelled with its cause;
- **how it uses its tools**: the right tool, valid arguments, results it actually used;
- **what it costs and how long it takes**, per correct answer and at p50/p95;
- **whether a change made it better**, with a paired statistical test, so you don't ship
  a prompt tweak that only looked better by chance.

```bash
pip install "agenteval-kit[server]"
agenteval init      # an example agent, a few questions, a config
agenteval run       # no API key needed for the example
agenteval ui        # see it in your browser
```

```
Experiment 5: groq-first   config v=c98ba11d65cb   48 trials
Model:            openai/gpt-oss-20b   dataset: mixed
Pass rate:        100.0%
Latency p50/p95:  1042ms / 57768ms total, 987ms / 1572ms excluding waits
Tool calls:
  right tool        100.0%  (48 trials)
  valid arguments   100.0%  (48 trials)
  ran without error 93.8%  (48 trials)
Retries:          1 (rate_limit 1)
Cost:             $0.0027 total, $0.000057 per success
Failures:         none
```

## Where to go next

- **[Quickstart](quickstart.md)**: your own agent evaluated in five minutes.
- **[Connect your agent](guides/your-agent.md)**: plain functions, OpenAI clients,
  LangChain and LangGraph.
- **[Compare versions and gate CI](guides/compare.md)**: block a pull request that makes
  the agent worse.
- **[agenteval vs other tools](comparison.md)**: how it relates to Ragas, DeepEval and promptfoo.

!!! note "Names"
    The project is agent-eval, the Python package and command are `agenteval`, and you
    install it from PyPI as `agenteval-kit` (the shorter names were taken).
