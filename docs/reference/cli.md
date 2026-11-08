# Command line

Every command is `agenteval <command>`. Options can also be set in
[agenteval.toml](config.md); the command line wins. This page is generated from each
command's `--help` by `scripts/build_cli_reference.py`.

## init

Set up a project: an example agent, cases and agenteval.toml.

```
usage: agenteval init [-h] [--agent-file AGENT_FILE] [--cases-file CASES_FILE] [--force]

Set up a project in this directory.

options:
  -h, --help            show this help message and exit
  --agent-file AGENT_FILE
  --cases-file CASES_FILE
  --force               overwrite files that exist
```

## run

Run an agent over a dataset and save every step.

```
usage: agenteval run [-h] [--name NAME] [--db DB] [--agent AGENT] [--model MODEL]
                     [--price INPUT OUTPUT] [--dataset DATASET] [--cases CASES]
                     [--attempts ATTEMPTS] [--concurrency CONCURRENCY] [--timeout TIMEOUT]
                     [--tool-error-rate TOOL_ERROR_RATE] [--rate-limit-rate RATE_LIMIT_RATE]
                     [--hallucination-rate HALLUCINATION_RATE] [--hang-rate HANG_RATE]
                     [--seed SEED] [--max-steps MAX_STEPS] [--temperature TEMPERATURE]
                     [--prompt-file PROMPT_FILE] [--rpm RPM] [--tpm TPM] [--top-k K]
                     [--embedding {minilm,hashing}] [--max-cost USD] [--sample N]
                     [--sample-seed SAMPLE_SEED] [--resume ID] [--scorer MODULE:FUNCTION]
                     [--redact] [--quiet]

Run an agent over a dataset and save the results.

options:
  -h, --help            show this help message and exit
  --name NAME
  --db DB
  --agent AGENT         mock, gemini, groq, or your own as module.path:function (see
                        agenteval.sdk)
  --model MODEL         default: mock-llm, gemini-2.5-flash or openai/gpt-oss-20b; required for
                        your own agent
  --price INPUT OUTPUT  dollars per million input and output tokens for --model, if it isn't in
                        agenteval/cost.py; 0 0 if you don't track cost
  --dataset DATASET     arithmetic, mixed, hard, rag, or the path of your own .jsonl file;
                        default: arithmetic for mock, mixed for gemini and groq
  --cases CASES         size of the arithmetic dataset
  --attempts ATTEMPTS
  --concurrency CONCURRENCY
                        default: 20 mock, 2 gemini, 4 groq
  --timeout TIMEOUT     seconds per trial; default: 2 for mock, 360 for real agents

mock agent faults:
  --tool-error-rate TOOL_ERROR_RATE
  --rate-limit-rate RATE_LIMIT_RATE
  --hallucination-rate HALLUCINATION_RATE
  --hang-rate HANG_RATE
  --seed SEED

real agents (gemini, groq):
  --max-steps MAX_STEPS
  --temperature TEMPERATURE
  --prompt-file PROMPT_FILE
                        file holding the system prompt; default: the built-in prompt
  --rpm RPM             requests per minute to stay under; default 8 gemini, 25 groq; check your
                        quota; 0 = no limit
  --tpm TPM             tokens per minute to stay under, e.g. your Groq model's TPM limit;
                        default: no limit

rag dataset (needs: pip install -e ".[rag]"):
  --top-k K             documents each search returns, 1 to 8; default 4
  --embedding {minilm,hashing}
                        minilm: semantic, downloaded on first use; hashing: word matching, no
                        download

cost, scope and storage:
  --max-cost USD        stop starting trials once this much has been spent; resume later with
                        --resume
  --sample N            run N cases instead of all, spread across categories; same N and --sample-
                        seed, same cases, so samples can be compared
  --sample-seed SAMPLE_SEED
  --resume ID           continue an interrupted experiment: only its missing trials run (same
                        options)
  --scorer MODULE:FUNCTION
                        add your own scorer (repeatable); see agenteval.scorers.as_scorer
  --redact              replace emails, phone and card numbers, IPs and API keys in what is saved
  --quiet               no progress line
```

## report

Show an experiment: pass rate, failures, cost (text, json, markdown, html, csv).

```
usage: agenteval report [-h] [--db DB] [--format {text,json,markdown,html,csv}] [-o OUTPUT]
                        [experiment_id]

Print the report for a saved experiment.

positional arguments:
  experiment_id         default: the latest experiment

options:
  -h, --help            show this help message and exit
  --db DB
  --format {text,json,markdown,html,csv}
                        csv is one row per trial, with the question, answer and every score
  -o OUTPUT, --output OUTPUT
                        write to this file instead of printing
```

## list

List saved experiments.

```
usage: agenteval list [-h] [--db DB] [--limit LIMIT]

List saved experiments, newest first.

options:
  -h, --help     show this help message and exit
  --db DB
  --limit LIMIT
```

## compare

Is version B really better than A?.

```
usage: agenteval compare [-h] [--db DB] a b

Compare two saved experiments: is B really better than A?

positional arguments:
  a           the baseline experiment
  b           the experiment to judge

options:
  -h, --help  show this help message and exit
  --db DB
```

## gate

Pass, block or can't judge a change, for CI.

```
usage: agenteval gate [-h] [--markdown MARKDOWN] [--block-inconclusive] [--min-pass-rate RATE]
                      baseline_db candidate_db

Decide whether a change passes, comparing two result databases.

positional arguments:
  baseline_db           database whose latest experiment is the baseline
  candidate_db          database whose latest experiment is the change

options:
  -h, --help            show this help message and exit
  --markdown MARKDOWN   also write the report to this file, for a PR comment
  --block-inconclusive  block when a drop can't be ruled out, not only when one is shown
  --min-pass-rate RATE  block when the change's pass rate is below this, e.g. 0.85
```

## judge

Score answers with an LLM judge.

```
usage: agenteval judge [-h] --rubric RUBRIC [--db DB] [--model MODEL]
                       [--provider {groq,ollama,openai,openrouter}] [--base-url BASE_URL]
                       [--api-key-env API_KEY_ENV] [--price INPUT OUTPUT] [--sample N]
                       [--no-cache] [--rpm RPM] [--concurrency CONCURRENCY]
                       experiment_id

Score a saved experiment's answers with an LLM judge.

positional arguments:
  experiment_id

options:
  -h, --help            show this help message and exit
  --rubric RUBRIC       answer_relevance, context_relevance, correctness, groundedness, or the
                        path of your own rubric .toml file
  --db DB
  --model MODEL         the judge; a different, larger model than the agent's is the usual choice
  --provider {groq,ollama,openai,openrouter}
  --base-url BASE_URL   any other OpenAI-compatible endpoint (Azure, vLLM, a gateway)
  --api-key-env API_KEY_ENV
                        the environment variable holding that endpoint's key
  --price INPUT OUTPUT  dollars per million tokens, for a model not in the price table
  --sample N            judge a fixed random sample of N trials
  --no-cache            ask again even for answers already judged
  --rpm RPM             requests per minute to stay under; 0 = no limit
  --concurrency CONCURRENCY
```

## calibrate

Label answers yourself and measure the judge's agreement.

```
usage: agenteval calibrate [-h] {label,agree} ...

Label answers yourself and measure an LLM judge's agreement.

positional arguments:
  {label,agree}
    label        label a random sample of answers
    agree        compare your labels with the judge's scores

options:
  -h, --help     show this help message and exit
```

### calibrate label

```
usage: agenteval calibrate label [-h] --rubric RUBRIC [--db DB] [--labels LABELS]
                                 [--sample SAMPLE] [--seed SEED]
                                 experiment_id

positional arguments:
  experiment_id

options:
  -h, --help       show this help message and exit
  --rubric RUBRIC  a built-in rubric or your rubric .toml file
  --db DB
  --labels LABELS  first import labels from this JSONL file
  --sample SAMPLE  how many answers to label
  --seed SEED
```

### calibrate agree

```
usage: agenteval calibrate agree [-h] --rubric RUBRIC [--db DB] [--labels LABELS]
                                 [--export EXPORT]
                                 experiment_id

positional arguments:
  experiment_id

options:
  -h, --help       show this help message and exit
  --rubric RUBRIC  a built-in rubric or your rubric .toml file
  --db DB
  --labels LABELS  first import labels from this JSONL file
  --export EXPORT  also write your labels to this JSONL file
```

## rescore

Re-run the scorers without calling the agent again.

```
usage: agenteval rescore [-h] [--db DB] experiment_id

Re-run the scorers on a saved experiment without calling the agent again.

positional arguments:
  experiment_id

options:
  -h, --help     show this help message and exit
  --db DB
```

## estimate

Estimate what a run will cost before starting it.

```
usage: agenteval estimate [-h] --model MODEL --dataset DATASET [--attempts ATTEMPTS]
                          [--sample SAMPLE] [--cases CASES] [--concurrency CONCURRENCY] [--db DB]

Estimate a run's cost and time from earlier runs of the same model.

options:
  -h, --help            show this help message and exit
  --model MODEL
  --dataset DATASET     a built-in dataset or your .jsonl file
  --attempts ATTEMPTS
  --sample SAMPLE       as in agenteval run
  --cases CASES         size of the arithmetic dataset
  --concurrency CONCURRENCY
  --db DB
```

## prices

Show or update the model price table.

```
usage: agenteval prices [-h] [--update] [search]

Show or update model prices.

positional arguments:
  search      show models whose name contains this

options:
  -h, --help  show this help message and exit
  --update    fetch current prices from LiteLLM's list into ~/.agenteval/prices.json (or
              AGENTEVAL_PRICES_FILE)
```

## power

How many cases you need to catch a drop of a given size.

```
usage: agenteval power [-h] [--cases CASES [CASES ...]] [--attempts ATTEMPTS]
                       [--pass-rate PASS_RATE] [--drop DROP [DROP ...]]
                       [--simulations SIMULATIONS] [--target TARGET]

agenteval power: how many cases you need before a comparison can catch a given drop.

options:
  -h, --help            show this help message and exit
  --cases CASES [CASES ...]
  --attempts ATTEMPTS
  --pass-rate PASS_RATE
                        your agent's current pass rate, 0 to 1
  --drop DROP [DROP ...]
                        drops to test, as fractions: 0.10 is 10 points
  --simulations SIMULATIONS
  --target TARGET       detection rate to aim for
```

## generate

Write test cases from your documents with an LLM.

```
usage: agenteval generate [-h] [-n COUNT] [-o OUTPUT] [--corpus-out CORPUS_OUT]
                          [--unanswerable UNANSWERABLE] [--model MODEL]
                          [--provider {groq,ollama,openai,openrouter}] [--base-url BASE_URL]
                          [--api-key-env API_KEY_ENV] [--seed SEED]
                          docs [docs ...]

Write test cases from your documents.

positional arguments:
  docs                  files or folders of .md / .txt

options:
  -h, --help            show this help message and exit
  -n COUNT, --count COUNT
  -o OUTPUT, --output OUTPUT
  --corpus-out CORPUS_OUT
  --unanswerable UNANSWERABLE
                        share of questions the docs don't answer
  --model MODEL
  --provider {groq,ollama,openai,openrouter}
  --base-url BASE_URL
  --api-key-env API_KEY_ENV
  --seed SEED
```

## import

Import traces your agent already recorded, to score them.

```
usage: agenteval import [-h] [--name NAME] [--model MODEL] [--cases-out CASES_OUT] [--db DB]
                        traces

Import recorded traces as an experiment.

positional arguments:
  traces                JSONL file, one record per answer

options:
  -h, --help            show this help message and exit
  --name NAME           default: the file's name
  --model MODEL         the model that answered, for reports
  --cases-out CASES_OUT
                        where to write the questions; default: next to the traces
  --db DB
```

## export

Export an experiment's trials (jsonl, or Ragas samples).

```
usage: agenteval export [-h] [--format {jsonl,ragas}] -o OUTPUT [--db DB] experiment_id

Export an experiment's trials as JSONL.

positional arguments:
  experiment_id

options:
  -h, --help            show this help message and exit
  --format {jsonl,ragas}
                        jsonl: agenteval's import format; ragas: SingleTurnSample fields
  -o OUTPUT, --output OUTPUT
  --db DB
```

## delete

Delete an experiment.

```
usage: agenteval delete [-h] [--db DB] [--yes] experiment_id

Delete an experiment and everything saved with it.

positional arguments:
  experiment_id

options:
  -h, --help     show this help message and exit
  --db DB
  --yes          don't ask for confirmation
```

## ui

Open the dashboard in your browser (needs agenteval-kit[server]).

```
usage: agenteval ui [-h] [--db DB] [--port PORT] [--host HOST] [--no-browser] [--read-only]

agenteval ui: the dashboard for your local results, in your browser.

options:
  -h, --help    show this help message and exit
  --db DB
  --port PORT
  --host HOST   127.0.0.1 keeps it on this machine
  --no-browser
  --read-only   refuse runs, re-scoring, labels and deletes
```
