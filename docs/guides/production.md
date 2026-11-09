# Watch it in production

An evaluation tells you how the agent does on the questions you wrote. Production tells you
how it does on the ones people actually ask. `Monitor` records a sample of real requests the
way an evaluation records trials:

```python
from agenteval.monitor import Monitor

monitor = Monitor("production.db", model="gpt-4o-mini", sample_rate=0.05, redact=True)

@monitor.watch
async def answer(question: str) -> str:
    ...  # your agent, traced as in an evaluation (wrap_openai, llm_call/tool, instrument())
```

- **What's kept**: each sampled call's question, answer, steps, tokens, latency and failure,
  as one trial in an experiment per day, `production YYYY-MM-DD`. A call given a list of
  messages is kept as a conversation.
- **What's scored right away**: everything that needs no expected answer: tool selection,
  arguments and execution, trajectory, repeated calls, faithfulness to tool results,
  forbidden text, failures, cost and latency.
- **What to judge later**: answer quality, with the reference-free rubrics, on a sample:

  ```bash
  agenteval judge 42 --db production.db --rubric answer_relevance --sample 100
  agenteval judge 42 --db production.db --rubric groundedness --sample 100
  ```

- **Your users' data**: `redact=True` replaces emails, phone and card numbers, IPs and API
  keys in everything kept. Mind your privacy policy before recording real conversations.

It stays out of the request's way: unsampled calls run untouched, sampled ones are written
by a background thread, a failure to record is dropped rather than raised, and your
function's own errors reach the caller as before. Call `monitor.flush()` before a short-lived
process exits.

Open `agenteval ui --db production.db` to follow the days in the dashboard's trend line, or
`agenteval push` a day to your team's server. A failure you find there makes a good new
case for your dataset: copy its question into `cases.jsonl` with the answer it should have
given.
