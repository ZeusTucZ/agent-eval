// Launch a run: POST /api/runs, then poll GET /api/runs/{id} until the job
// is completed or failed. The API accepts agent, dataset, attempts, model,
// name, cases and seed (evalkit/api.py RunRequest). It does NOT accept
// --rpm or --timeout overrides: the server applies the per-agent defaults
// from run.py AGENT_DEFAULTS, shown below read-only so nobody thinks the
// form controls them. The mock agent only understands the arithmetic
// dataset (the API refuses mock + mixed/hard with 400).
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Job, RunRequest } from "../types";

type Agent = RunRequest["agent"];

// Per-agent defaults from evalkit/run.py AGENT_DEFAULTS. Display only.
const AGENT_DEFAULTS: Record<Agent, { model: string; dataset: string; timeout: number; rpm: number }> = {
  mock: { model: "mock-llm", dataset: "arithmetic", timeout: 2.0, rpm: 0 },
  gemini: { model: "gemini-2.5-flash", dataset: "mixed", timeout: 360.0, rpm: 8 },
  groq: { model: "openai/gpt-oss-20b", dataset: "mixed", timeout: 360.0, rpm: 25 },
};

// Models with a price in evalkit/cost.py PRICES. Any other model is refused
// by the API with 400 ("No price for ...").
const PRICED_MODELS = [
  "mock-llm",
  "mock-llm-small",
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
  "llama-3.3-70b-versatile",
];

interface Props {
  token: string;
  onTokenChange: (token: string) => void;
  onExperimentReady: (id: number) => void;
}

export default function RunForm({ token, onTokenChange, onExperimentReady }: Props) {
  const [agent, setAgent] = useState<Agent>("mock");
  const [dataset, setDataset] = useState<RunRequest["dataset"]>("arithmetic");
  const [attempts, setAttempts] = useState(1);
  const [model, setModel] = useState("");
  const [name, setName] = useState("dashboard-run");
  const [cases, setCases] = useState(100);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const timer = useRef<number | null>(null);

  const defaults = AGENT_DEFAULTS[agent];

  useEffect(() => {
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current);
    };
  }, []);

  function pickAgent(next: Agent) {
    setAgent(next);
    setDataset(AGENT_DEFAULTS[next].dataset as RunRequest["dataset"]);
    setModel("");
  }

  async function poll(jobId: string) {
    try {
      const current = await api.getRun(jobId);
      setJob(current);
      if (current.status === "completed" || current.status === "failed") {
        if (timer.current !== null) {
          window.clearInterval(timer.current);
          timer.current = null;
        }
      }
    } catch (err) {
      if (timer.current !== null) {
        window.clearInterval(timer.current);
        timer.current = null;
      }
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setJob(null);
    if (!token) {
      setError("Enter your API token — POST /api/runs requires the X-API-Token header.");
      return;
    }
    if (agent === "mock" && dataset !== "arithmetic") {
      setError("The mock agent only understands the arithmetic dataset; pick groq or gemini for mixed/hard.");
      return;
    }
    const body: RunRequest = {
      agent,
      dataset,
      attempts,
      name: name.trim() || "dashboard-run",
      seed: 42,
    };
    if (model) body.model = model;
    if (dataset === "arithmetic") body.cases = cases;
    setSubmitting(true);
    try {
      const created = await api.createRun(body, token);
      setJob(created);
      timer.current = window.setInterval(() => void poll(created.id), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <p className="section-label">Evaluate an agent</p>
          <h2>New run</h2>
          <p>
            Runs execute on the server in the background; this page polls the job until it
            finishes. Groq and Gemini runs spend the server&apos;s API quota — the mock
            agent costs nothing and needs no provider key.
          </p>
        </div>
      </div>
      <form onSubmit={handleSubmit} className="run-form">
        <label>
          API token (sent as X-API-Token header; never stored in the repo)
          <input
            type="password"
            value={token}
            onChange={(e) => onTokenChange(e.target.value)}
            placeholder="Paste the API token here"
            autoComplete="off"
            required
          />
        </label>
        <label>
          Run name
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label>
          Agent
          <select value={agent} onChange={(e) => pickAgent(e.target.value as Agent)}>
            <option value="mock">mock</option>
            <option value="groq">groq</option>
            <option value="gemini">gemini</option>
          </select>
        </label>
        <label>
          Dataset
          <select value={dataset} onChange={(e) => setDataset(e.target.value as RunRequest["dataset"])}>
            <option value="arithmetic">arithmetic</option>
            <option value="mixed">mixed</option>
            <option value="hard">hard</option>
            <option value="rag">rag</option>
          </select>
        </label>
        {dataset === "arithmetic" ? (
          <label>
            Cases (arithmetic dataset size)
            <input
              type="number"
              min={1}
              value={cases}
              onChange={(e) => setCases(Number(e.target.value))}
            />
          </label>
        ) : null}
        <label>
          Attempts
          <input
            type="number"
            min={1}
            value={attempts}
            onChange={(e) => setAttempts(Number(e.target.value))}
          />
        </label>
        <label>
          Model (blank = server default: {defaults.model})
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="">Server default ({defaults.model})</option>
            {PRICED_MODELS.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>
        <fieldset>
          <legend>Server defaults for this agent (not sent — the API applies them)</legend>
          <p>
            Timeout: {defaults.timeout}s per trial · Rate limit:{" "}
            {defaults.rpm === 0 ? "unlimited" : `${defaults.rpm} rpm`} · Dataset default:{" "}
            {defaults.dataset}
          </p>
          <p className="footnote">
            POST /api/runs does not accept --rpm or --timeout; the server uses these
            per-agent defaults from run.py.
          </p>
        </fieldset>
        <button type="submit" disabled={submitting}>
          {submitting ? "Starting…" : "Launch run"}
        </button>
      </form>

      {error ? <p className="error">{error}</p> : null}

      {job ? (
        <div className="job-card">
          <h3>Job {job.id}</h3>
          <p>
            Status:{" "}
            <span className={`job-status job-${job.status}`}>{job.status}</span>
            {job.error ? ` — ${job.error}` : ""}
          </p>
          {job.status === "completed" && job.experiment_id !== null ? (
            <p>
              <button type="button" onClick={() => onExperimentReady(job.experiment_id as number)}>
                View experiment {job.experiment_id}
              </button>
            </p>
          ) : null}
          {job.status === "running" || job.status === "queued" ? (
            <p>Runs take a while — this page keeps polling. Don&apos;t launch a second run until this one finishes (one SQLite writer).</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
