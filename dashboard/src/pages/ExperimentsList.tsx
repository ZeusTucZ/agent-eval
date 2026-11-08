import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExperimentSummary } from "../types";
import { pct, usd } from "../format";

interface Props {
  onOpen: (id: number) => void;
  onNewRun?: () => void;
}

function passRateClass(rate: number): string {
  if (rate >= 0.9) return "pill-good";
  if (rate >= 0.7) return "pill-warn";
  return "pill-bad";
}

export default function ExperimentsList({ onOpen, onNewRun }: Props) {
  const [experiments, setExperiments] = useState<ExperimentSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listExperiments()
      .then(setExperiments)
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error) return <p className="error">Could not load experiments: {error}</p>;
  if (experiments === null) return <p>Loading experiments…</p>;
  if (experiments.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 3h6M10 3v6.3L4.7 18a2 2 0 0 0 1.8 3h11a2 2 0 0 0 1.8-3L14 9.3V3" />
            <path d="M7.5 14h9" />
          </svg>
        </div>
        <p className="section-label">Getting started</p>
        <h2>No experiments yet</h2>
        <p>
          Launch your first run — the mock agent on the arithmetic dataset needs no API
          key and costs nothing.
        </p>
        {onNewRun ? (
          <button type="button" className="btn-primary" onClick={onNewRun}>
            Launch your first run →
          </button>
        ) : null}
      </div>
    );
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <p className="section-label">Results</p>
          <h2>Experiments</h2>
          <p>
            {experiments.length} experiment{experiments.length === 1 ? "" : "s"} · click a
            row for trials, latency and cost.
          </p>
        </div>
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Dataset</th>
              <th>Model</th>
              <th>Trials</th>
              <th>Pass rate</th>
              <th>p50 / p95</th>
              <th>Cost</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp) => (
              <tr key={exp.id} onClick={() => onOpen(exp.id)} className="clickable">
                <td>{exp.id}</td>
                <td><strong>{exp.name}</strong></td>
                <td>{exp.dataset_name}</td>
                <td>{exp.model}</td>
                <td>{exp.trial_count}</td>
                <td>
                  <span className={`pill ${passRateClass(exp.pass_rate)}`}>
                    {pct(exp.pass_rate)}
                  </span>
                </td>
                <td>
                  {exp.latency_ms.p50.toFixed(0)} / {exp.latency_ms.p95.toFixed(0)} ms
                </td>
                <td>{usd(exp.cost)}</td>
                <td>{new Date(exp.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
