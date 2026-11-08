# agent-eval dashboard

React + Vite + TypeScript dashboard for the agent-eval FastAPI backend
(Phase 4 of the deployment plan). It only displays fields the API actually
returns — see `src/types.ts` for the exact contract (from
`agenteval/api.py` `experiment_summary()` and `agenteval/models.py`).

## Run locally

```bash
npm install
npm run dev        # Vite on http://localhost:5173
```

The API base URL comes from `VITE_API_URL` (only read in `src/api.ts`). It
defaults to `http://localhost:8000` — run the backend first:

```bash
uvicorn agenteval.api:app --reload   # from the repo root
```

## Build

```bash
npm run build      # tsc + vite build -> dashboard/dist
```

## Deploy (Render Static Site, done by hand — not by this repo)

- Root Directory: `dashboard`
- Build Command: `npm run build`
- Publish Directory: `dashboard/dist`
- Environment variable: `VITE_API_URL=https://agent-eval-co10.onrender.com`
  (no trailing slash). `VITE_API_URL` is baked in at build time: changing it
  in Render requires a rebuild (Manual Deploy > Deploy latest commit).

After the Static Site exists, add its URL to the backend's `CORS_ORIGINS`
or the browser will block the API calls.

## Design

Minimal and organised around what someone opening a run wants to know, in order:

1. **How good was it?** Four numbers: pass rate, trials that didn't pass, typical latency and cost per correct answer.
2. **Why did trials fail?** Failures (the trial stopped) and wrong answers (it finished with the wrong result) in one list, largest first, since they need different fixes.
3. **Search or answer?** For RAG runs only: whether a wrong answer came from the search or from the model.
4. **Which checks failed?** Only the checks that apply to the dataset; one that doesn't apply is left out rather than shown as n/a.
5. **The trials themselves**, filtered to the ones that didn't pass. Each one opens to show its answer, every step and every score with its reasoning.

Identifiers, the latency breakdown and re-scoring are under "Details" at the end. Colour is used only for good or bad news, light and dark follow the system setting, and the layout works on a phone.

## Notes

- All GET endpoints are public; POST `/api/runs` and POST
  `/api/experiments/{id}/rescore` need the API token, entered in the form
  and sent as the `X-API-Token` header. The token lives only in the browser
  tab: it is never written to the repo or local storage.
- The run form only offers combinations the API accepts: the mock agent
  with the `arithmetic` dataset, real agents with `mixed`, `hard` or `rag`,
  and only models that have a price.
- POST `/api/runs` does not accept `--rpm` / `--timeout`: the server applies
  the per-agent defaults from `agenteval/run.py` `AGENT_DEFAULTS`.
