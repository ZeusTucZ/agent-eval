# LLM judge and calibration

Some answers can't be scored by matching strings. `agenteval judge` asks a second model for a
pass or fail verdict on each answer of a saved experiment.

```bash
agenteval judge 12 --rubric groundedness
```

- **Model and provider**: `llama-3.3-70b-versatile` on Groq by default. `--provider openai`,
  `--provider openrouter` (which reaches Anthropic, Google and many others), `--provider ollama`,
  or any OpenAI-compatible endpoint with `--base-url` and `--api-key-env`. A different, larger
  model than the agent's is the usual choice.
- **Cost**: `--sample 100` judges a fixed random sample. Verdicts are cached by the exact
  request, so judging the same answers again costs nothing (`--no-cache` to ask anyway).
- **Prompt injection**: the judge is told that everything it evaluates is data, not
  instructions. That matters when sources contain injected text.
- **Your own rubric**, from a TOML file:

```toml
name = "politeness"
question = "Is the ANSWER polite to the user? Fail if it is curt or condescending."
uses_expected = false
uses_sources = false
```

```bash
agenteval judge 12 --rubric politeness.toml
```

Scores are stored as `judge_<rubric>`, tagged with a hash of the judge's model and prompt.
`agenteval rescore` never deletes them.

## Check the judge before trusting it

A judge is a model too. Its pass rate means nothing until you know how often it agrees with
a person. Label a fixed random sample yourself, without seeing its verdicts:

```bash
agenteval calibrate label 12 --rubric groundedness --sample 50
```

or in the dashboard: an experiment's **Label answers** page, with `p` and `f` as shortcuts.
Then:

```bash
agenteval calibrate agree 12 --rubric groundedness
```

```
Trials compared:  50
Agreement:        88.0%
Cohen's kappa:    0.74  substantial agreement: usable for comparing versions, but read its failures
Pass rate:        human 64.0%, judge 70.0%
```

**Cohen's kappa** is agreement beyond chance: a judge that passes everything agrees 90% of
the time with labels that are 90% passes, and gets a kappa of 0. Read the listed
disagreements: they show what to fix in the rubric. Labels are saved in the experiment's
database; `--export labels.jsonl` keeps a copy next to your dataset.
