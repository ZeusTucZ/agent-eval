# agent-eval dashboard

React + Vite + TypeScript dashboard for the agent-eval FastAPI backend
(Phase 4 of the deployment plan). It only displays fields the API actually
returns — see `src/types.ts` for the exact contract (from
`evalkit/api.py` `experiment_summary()` and `evalkit/models.py`).

## Run locally

```bash
npm install
npm run dev        # Vite on http://localhost:5173
```

The API base URL comes from `VITE_API_URL` (only read in `src/api.ts`). It
defaults to `http://localhost:8000` — run the backend first:

```bash
uvicorn evalkit.api:app --reload   # from the repo root
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

## Notes

- All GET endpoints are public; POST `/api/runs` and POST
  `/api/experiments/{id}/rescore` need the API token, entered in the form
  and sent as the `X-API-Token` header. The token lives only in the browser
  tab — it is never written to the repo or local storage.
- The mock agent only understands the `arithmetic` dataset (the API refuses
  mock + mixed/hard with 400).
- POST `/api/runs` does not accept `--rpm` / `--timeout`: the server applies
  the per-agent defaults from `evalkit/run.py` `AGENT_DEFAULTS`. The form
  shows those defaults read-only.
- Rates with no scores (e.g. tool faithfulness on arithmetic runs) render
  as “n/a”, never as 0%.
