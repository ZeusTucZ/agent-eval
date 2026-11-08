# What gets measured

Every trial is scored by these, when they apply. A scorer that doesn't apply (no expected
tools, no documents) gives no score rather than a misleading 0.

## Correctness

**`exact_match`**: the answer equals the expected one, forgiving case, accents, surrounding
space and a final period. Numbers compare by value, within 0.01 (`19753.0` = `19,753`).
Skipped when the case has no expected answer.

For open-ended answers, where wording varies, use the LLM judge's `correctness` rubric.

## Tool use, step by step

Each rung fails for a different reason, so each has its own score:

1. **`tool_selection`**: the agent called every tool the case needs, as often as needed.
   Answering from memory fails this even when the answer is right. Not judged when the model
   never responded (that's infrastructure, not a choice).
2. **`tool_arguments_valid`**: every call named a real tool with valid arguments.
3. **`tool_execution`**: every call with valid arguments ran without an error.

## Faithfulness and retrieval

- **`tool_faithfulness`**: every number in the answer, or its text, appears in a tool
  result. A deterministic hallucination check: it needs no reference and no LLM. It measures
  faithfulness to sources, not truth: a right answer worked out in the model's head fails it.
- **`retrieval_recall`**: the share of the case's `relevant_docs` that any search returned.
  Crossed with `exact_match`, reports say whether a wrong answer came from the search
  (a document was missed) or from the model (it had them all).

## Safety

**`avoids_forbidden_text`**: the answer contains none of the case's `must_not_contain`, such
as an injected instruction's payload. The gate blocks any new failure here whatever else improved.

## Latency, cost and failures

- **Latency** p50/p95, both total (what a user waits) and excluding waits (time inside
  requests, without retries and rate limiting).
- **Cost**: tokens × the price table, total and per correct answer.
- **Retries** by reason, and **failures** by cause (see [Concepts](../concepts.md)).

## LLM judge rubrics

Scored by a second model with `agenteval judge`; see [LLM judge](judge.md).

| Rubric | Question | Needs |
|---|---|---|
| `correctness` | same answer as expected, whatever the wording? | expected answer |
| `groundedness` | every claim supported by the tool results? | tool results |
| `answer_relevance` | does it address the question asked? | nothing |
| `context_relevance` | do the retrieved sources contain what's needed? | tool results |

## Your own scorers

A function of `(case, trial)` returning a number from 0 to 1, a bool, a `(value, reason)`
pair, a `Score`, or `None` to skip:

```python
def concise(case, trial):
    return len(trial.output or "") < 200, "under 200 characters"
```

```bash
agenteval run ... --scorer my_scorers:concise
```

or `evaluate(..., scorers=[concise])`. Its scores appear under "Other scores", and
`agenteval rescore` runs it again.
