// Start a run: POST /api/runs, then poll GET /api/runs/{id} until the job finishes.
// Only the choices that matter are up front; the rest sit under "More options" with
// the server's defaults. Choices that the API would refuse aren't offered: the mock
// agent only answers the arithmetic dataset, and real agents need a real dataset.
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { go } from "../App";
import type { Job, RunRequest } from "../types";

type Agent = RunRequest["agent"];
type Dataset = RunRequest["dataset"];

const AGENTS: { value: Agent; label: string; note: string; model: string }[] = [
  { value: "mock", label: "Mock", note: "Free, instant, no key", model: "mock-llm" },
  { value: "groq", label: "Groq", note: "Uses the server's quota", model: "openai/gpt-oss-20b" },
  { value: "gemini", label: "Gemini", note: "20 requests a day free", model: "gemini-2.5-flash" },
];

const DATASETS: Record<Dataset, string> = {
  arithmetic: "Arithmetic: checks the harness itself",
  mixed: "Mixed: 16 easy questions",
  hard: "Hard: 55 questions in 10 categories",
  rag: "RAG: 45 questions over company documents",
};

// Models with a price in agenteval/cost.py; the API refuses any other.
const MODELS: Record<Agent, string[]> = {
  mock: ["mock-llm", "mock-llm-small"],
  groq: ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
  gemini: ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
};

function datasetsFor(agent: Agent): Dataset[] {
  return agent === "mock" ? ["arithmetic"] : ["mixed", "hard", "rag"];
}

interface Props {
  token: string;
  onTokenChange: (token: string) => void;
}

export default function RunForm({ token, onTokenChange }: Props) {
  const [agent, setAgent] = useState<Agent>("mock");
  const [dataset, setDataset] = useState<Dataset>("arithmetic");
  const [attempts, setAttempts] = useState(3);
  const [model, setModel] = useState("");
  const [name, setName] = useState("");
  const [cases, setCases] = useState(100);
  const [topK, setTopK] = useState(4);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const timer = useRef<number | null>(null);

  const stopPolling = () => {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = null;
  };
  useEffect(() => stopPolling, []);

  function pickAgent(next: Agent) {
    setAgent(next);
    setDataset(datasetsFor(next)[0]);
    setModel("");
  }

  async function poll(jobId: string) {
    try {
      const current = await api.getRun(jobId);
      setJob(current);
      if (current.status === "completed" || current.status === "failed") stopPolling();
    } catch (err) {
      stopPolling();
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setJob(null);
    const body: RunRequest = {
      agent,
      dataset,
      attempts,
      name: name.trim() || `${agent}-${dataset}`,
      seed: 42,
    };
    if (model) body.model = model;
    if (dataset === "arithmetic") body.cases = cases;
    if (dataset === "rag") body.top_k = topK;
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

  const running = job !== null && (job.status === "queued" || job.status === "running");
  const defaultModel = AGENTS.find((a) => a.value === agent)?.model ?? "";

  return (
    <>
      <div className="page-head">
        <div>
          <h1>New run</h1>
          <p className="meta">Runs on the server; you can watch it finish here.</p>
        </div>
      </div>

      <form className="form" onSubmit={handleSubmit}>
        <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
          <legend style={{ fontWeight: 500, fontSize: "0.9rem", marginBottom: 6 }}>Agent</legend>
          <div className="choice-group">
            {AGENTS.map((option) => (
              <label key={option.value} className="choice">
                <input
                  type="radio"
                  name="agent"
                  value={option.value}
                  checked={agent === option.value}
                  onChange={() => pickAgent(option.value)}
                />
                <strong>{option.label}</strong>
                <span>{option.note}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <label>
          Dataset
          <select value={dataset} onChange={(e) => setDataset(e.target.value as Dataset)}>
            {datasetsFor(agent).map((value) => (
              <option key={value} value={value}>
                {DATASETS[value]}
              </option>
            ))}
          </select>
          {agent === "mock" ? <span className="hint">The mock agent only answers arithmetic.</span> : null}
        </label>

        <label>
          API token
          <input
            type="password"
            value={token}
            onChange={(e) => onTokenChange(e.target.value)}
            autoComplete="off"
            required
          />
          <span className="hint">Needed to start runs. Kept in this tab only.</span>
        </label>

        <details>
          <summary>More options</summary>
          <div className="form">
            <div className="row">
              <label>
                Attempts per question
                <input type="number" min={1} max={10} value={attempts} onChange={(e) => setAttempts(Number(e.target.value))} />
              </label>
              {dataset === "arithmetic" ? (
                <label>
                  Questions
                  <input type="number" min={1} max={1000} value={cases} onChange={(e) => setCases(Number(e.target.value))} />
                </label>
              ) : dataset === "rag" ? (
                <label>
                  Documents per search
                  <input type="number" min={1} max={8} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
                </label>
              ) : null}
            </div>
            <label>
              Model
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                <option value="">Default ({defaultModel})</option>
                {MODELS[agent]
                  .filter((m) => m !== defaultModel)
                  .map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
              </select>
            </label>
            <label>
              Name
              <input type="text" value={name} placeholder={`${agent}-${dataset}`} onChange={(e) => setName(e.target.value)} />
            </label>
          </div>
        </details>

        <div>
          <button type="submit" className="primary" disabled={submitting || running || !token}>
            {submitting ? "Starting…" : running ? "Running…" : "Start run"}
          </button>
        </div>
      </form>

      <div aria-live="polite" style={{ maxWidth: 520, marginTop: 24 }}>
        {error ? <p className="notice error">{error}</p> : null}
        {job ? (
          job.status === "completed" && job.experiment_id !== null ? (
            <div className="notice success" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
              <span>Finished.</span>
              <button type="button" onClick={() => go(`e/${job.experiment_id}`)}>
                See results
              </button>
            </div>
          ) : job.status === "failed" ? (
            <p className="notice error">The run failed: {job.error}</p>
          ) : (
            <p className="notice info">
              {job.status === "queued" ? "Waiting for another run to finish…" : "Running…"} Real models can take
              several minutes; you can leave this page open.
            </p>
          )
        ) : null}
      </div>
    </>
  );
}
