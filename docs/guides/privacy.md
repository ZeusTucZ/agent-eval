# Privacy

## What is stored

Every experiment is a SQLite file on your machine (`agenteval.db` by default). It holds each
question, answer, tool argument, tool result and error message, up to a length per field.
Nothing is sent anywhere by agenteval itself: there is no telemetry.

## What leaves your machine

- **Your agent's calls** go wherever your agent sends them, as they would without agenteval.
- **The LLM judge** sends the question, answer and, for some rubrics, the tool results and
  expected answer to the judge's provider. Use a provider you're allowed to share that data
  with, or a local model (`--provider ollama`).
- **`agenteval generate`** sends passages of your documents to the model you choose.
- **`agenteval prices --update`** downloads a public price list; it sends nothing.

## Keeping personal data out of the database

```bash
agenteval run ... --redact
```

replaces emails, phone and card numbers, IBANs, IP addresses and API keys with markers like
`[EMAIL]` before anything is saved. Scores are computed first, from the real text, so a
correct answer that contains an email still counts as correct. `evaluate(..., redact=True)`
does the same from Python. Account ids in provider error messages (`org_...`) are always
removed, since databases end up in CI artifacts.

Redaction is pattern-based: it catches common formats, not names or free-text secrets. For
sensitive data, run on synthetic or anonymised cases, and keep databases out of public places.

## Reporting a security problem

See [SECURITY.md](https://github.com/ZeusTucZ/agent-eval/blob/main/SECURITY.md).
