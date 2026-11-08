# Import traces and Ragas

## Score traces you already have

When you can't or don't want agenteval to run your agent (production logs, another
framework's traces), import what it did:

```bash
agenteval import traces.jsonl --model gpt-4o-mini
agenteval report
agenteval judge 1 --rubric answer_relevance
```

One record per line, in agenteval's form:

```json
{"input": "What is our refund window?", "output": "30 days", "expected_output": "30 days",
 "category": "policy",
 "steps": [{"kind": "llm", "name": "gpt-4o-mini", "input_tokens": 812, "output_tokens": 40, "duration_ms": 950},
           {"kind": "tool", "name": "search", "input": {"q": "refunds"}, "output": ["..."], "duration_ms": 120}]}
```

or as a Ragas sample:

```json
{"user_input": "What is our refund window?", "response": "30 days",
 "retrieved_contexts": ["Refunds are accepted within 30 days."], "reference": "30 days"}
```

Only the question and the answer are required. `retrieved_contexts` become a `retrieve`
step, so groundedness and context relevance can be judged on them. Without an expected
answer, exact match is skipped rather than failed: the reference-free judge rubrics
(`answer_relevance`, `context_relevance`, `groundedness`) still apply. Records with the same
question are grouped as attempts at one case. The questions are written next to the traces
(`traces.cases.jsonl`) so reports can show them.

### OpenTelemetry

Traces exported as OTLP JSON (a collector's file exporter, or Phoenix and Langfuse exports)
import the same way; the format is detected, or force it with `--format otlp`:

```bash
agenteval import spans.json --model gpt-4o-mini
```

Each trace becomes one trial. Model and tool spans are recognized by the OpenInference
conventions (`openinference.span.kind`, `llm.token_count.*`, `tool.name`, as written by
Arize's instrumentors) or OpenTelemetry's GenAI conventions (`gen_ai.*`). The question and
answer come from the root span's `input.value` and `output.value`, or `agenteval.input` and
`agenteval.output` if you set them yourself. There are no expected answers in traces, so
exact match is skipped; tool checks, faithfulness and the judge still apply.

## Export

```bash
agenteval export 3 -o trials.jsonl                 # agenteval's form: imports back as-is
agenteval export 3 --format ragas -o samples.jsonl # Ragas SingleTurnSample fields
```

The Ragas export lets you run Ragas's own metrics on runs made here, alongside agenteval's.
