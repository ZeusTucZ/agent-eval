import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExperimentSummary } from "../types";
import { pct, usd } from "../format";

interface Props {
  onOpen: (id: number) => void;
}

export default function ExperimentsList({ onOpen }: Props) {
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
      <p>
        No experiments yet. Launch a run from the “New run” tab — the mock agent on the
        arithmetic dataset needs no API key.
      </p>
    );
  }

  return (
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
            <td>{exp.name}</td>
            <td>{exp.dataset_name}</td>
            <td>{exp.model}</td>
            <td>{exp.trial_count}</td>
            <td>{pct(exp.pass_rate)}</td>
            <td>
              {exp.latency_ms.p50.toFixed(0)} / {exp.latency_ms.p95.toFixed(0)} ms
            </td>
            <td>{usd(exp.cost)}</td>
            <td>{new Date(exp.created_at).toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
