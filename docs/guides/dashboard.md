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
  steps and scores with their reasoning.
- **A comparison**: the verdict in plain words, the change with its confidence interval,
  categories that moved, and the questions that flipped, with both answers.
- **Label answers**: one answer at a time, `p` or `f`, to check the LLM judge; the
  agreement appears once both exist.
- **New run**: the built-in agents and datasets. Your own agents run from the command line.

Every view has its own link (`#/e/3`, `#/compare/1/2`), so you can share them.

## Deploying it

`agenteval ui` is for your own machine. To publish results:

- **One service**: build the dashboard into the package (`cd dashboard && npm ci && npm run
  build`, then copy `dashboard/dist` to `agenteval/static`), and run `uvicorn
  agenteval.api:app`. It serves the API and the dashboard together.
- **Separately**: deploy `dashboard/` as a static site with `VITE_API_URL` set to the API's
  address, and add the site's address to the API's `CORS_ORIGINS`.

For a public demo, set `READ_ONLY=1`: every change (runs, re-scoring, labels, deletes) is
refused whatever the token. Otherwise set `API_TOKEN` and keep it secret; the API also limits
each client to `REQUESTS_PER_CLIENT_PER_MINUTE` (120 by default). The repository's
`Dockerfile` and `render.yaml` deploy the API on Render.
