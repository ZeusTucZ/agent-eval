# Write a dataset

A dataset is a JSONL file: one case per line.

```json
{"input": "What is the refund window?", "expected_output": "30 days", "category": "policy"}
{"input": "Convert 100 EUR to USD.", "expected_output": "108.70", "expected_tools": ["convert_currency"]}
{"input": "Who is our CFO?", "expected_output": "unknown", "category": "unanswerable"}
```

| Field | Required | Meaning |
|---|---|---|
| `input` | yes, unless `turns` | the question |
| `expected_output` | yes | the right answer; `""` for none (exact match is then skipped) |
| `id` | no | defaults to the line number; must be unique |
| `category` | no | kind of question; reports break the pass rate down by it |
| `expected_tools` | no | tools the agent must call, repeated once per call needed; `[]` means it must call none |
| `must_not_contain` | no | text the answer must never include, e.g. an injected instruction's payload |
| `expects_tool_error` | no | the tool is meant to fail here; how the agent handles it is what's tested |
| `relevant_docs` | no | ids of the documents holding the answer, for retrieval recall; `[]` for none |
| `expected_trajectory` | no | tool names in the order they should be called, for `trajectory_order` |
| `max_tool_calls` | no | how many tool calls the task needs, for `step_efficiency` |
| `turns` | no | a conversation: the user's messages in order (see below) |

Errors in the file name the line they're on.

## Conversations

```json
{"turns": ["I want to return my order.", "It's order 1234, bought 40 days ago."],
 "expected_output": "not eligible", "category": "refunds"}
```

The agent answers each turn in turn ([how it gets them](your-agent.md#conversations)); the
answer to the last one is checked against `expected_output`, and `must_not_contain` is
checked against every reply. `input` defaults to the last turn.

## How many cases?

More than you'd think. With 55 cases and 3 attempts, a 10-point drop is caught only about a
fifth of the time. Check before you trust a comparison:

```bash
agenteval power --pass-rate 0.85 --attempts 3
```

Use an odd number of attempts (ties can't count as a change), and cover each category with
several cases.

## Generate cases from your documents

```bash
agenteval generate docs/ -n 50 -o cases.jsonl --corpus-out corpus.jsonl
```

Splits `.md` and `.txt` files into passages, and asks a model for one question per passage
whose short answer is stated in it, linked by `relevant_docs`. About 10% are questions the
documents don't answer, expecting `unknown` (`--unanswerable`). Read the questions before
relying on them: drop any whose answer is wrong or ambiguous.

## Quick runs on a sample

```bash
agenteval run --dataset cases.jsonl --sample 20
```

20 cases spread across categories, the same 20 every time for the same `--sample-seed`, so
two sampled runs can be compared.

## The built-in datasets

| Name | Cases | Use |
|---|---|---|
| `arithmetic` | `--cases N` | the mock agent; validating the harness |
| `mixed` | 16 | real models: calculations, capitals, one missing country |
| `hard` | 55 in 10 categories | planted errors, prompt injection, ambiguity, tool errors |
| `extended` | 200 in 11 categories | the same tools as `hard`, enough cases to catch drops of about 10 points |
| `rag` | 45 in 6 categories | search over a fictional company's documents (`[rag]` extra) |
