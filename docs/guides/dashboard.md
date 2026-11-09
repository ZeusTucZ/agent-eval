# The dashboard

```bash
pip install "pyagenteval[server]"
agenteval ui
```

Opens your browser on `http://127.0.0.1:8765` with the experiments in `agenteval.db`
(`--db` for another). It listens on this machine only, and a token for the session is made up
and handed to the browser, so actions work without typing one.

## What it shows

- **Experiments**, newest first, with pass rate, latency, cost and age. Interrupted runs are
  marked. Tick two to compare them.
- **An experiment**, ordered by what you want to know: four headline numbers; why trials
  didn't pass (failures and wrong answers, largest first); search or answer, for RAG runs;
  pass rate by category; every check, the judge's and your own scorers; then the trials,
  opened on the ones that didn't pass, each with its question, expected answer, answer,
  steps and scores with their reasoning; for conversations, the whole exchange. Below the
  headline, the pass rate of every run of the same model on the same dataset, oldest
  first, with a link to compare this run with the one before.
- **A comparison**: the verdict in plain words, the change with its confidence interval,
  categories that moved, and the questions that flipped, with both answers.
- **Label answers**: one answer at a time, `p` or `f`, to check the LLM judge; the
  agreement appears once both exist.
- **New run**: your own agent, when `agenteval ui` starts in a project whose `agenteval.toml`
  names one (`[run] agent = "module:function"`), with its dataset, model and price; or the
  built-in agents and datasets. The server only ever runs that configured agent.

Every view has its own link (`#/e/3`, `#/compare/1/2`), so you can share them.

## Share runs with your team

Everyone evaluates on their own machine, with their own keys. To see all the runs in one
place, deploy a server (below) with a persistent disk and `API_TOKEN` set, then push:

```bash
export AGENTEVAL_SERVER=https://evals.example.com AGENTEVAL_TOKEN=...
agenteval push          # the latest experiment; or agenteval push 12
```

From Python: `results.push(url, token)`. The experiment arrives with its trials, scores and
questions, under a new id, ready to compare with everyone else's.

## Deploying it

`agenteval ui` is for your own machine. To publish results:

- **One service**: build the dashboard into the package (`cd dashboard && npm ci && npm run
  build`, then copy `dashboard/dist` to `agenteval/static`), and run `uvicorn
  agenteval.api:app`. It serves the API and the dashboard together.
- **Separately**: deploy `dashboard/` as a static site with `VITE_API_URL` set to the API's
  address, and add the site's address to the API's `CORS_ORIGINS`.

For a public demo, set `READ_ONLY=1`: every change (runs, re-scoring, labels, deletes) is
refused whatever the token. With `DEMO_SEED=1` an empty database starts with three example
runs of the mock agent, so a demo on a disk that's wiped at each deploy isn't empty. Otherwise set `API_TOKEN` and keep it secret; the API also limits
each client to `REQUESTS_PER_CLIENT_PER_MINUTE` (120 by default). The repository's
`Dockerfile` and `render.yaml` deploy the API on Render.
