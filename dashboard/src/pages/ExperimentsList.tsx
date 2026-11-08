import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExperimentSummary } from "../types";
import { ms, pct, rateTone, relativeTime, usd } from "../format";

interface Props {
  onOpen: (id: number) => void;
  onNewRun: () => void;
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

  if (error) return <p className="notice error">Couldn&apos;t load experiments: {error}</p>;
  if (experiments === null) return <p className="skeleton">Loading experiments…</p>;

  if (experiments.length === 0) {
    return (
      <div className="empty">
        <h1>No experiments yet</h1>
        <p>Run the mock agent first: it needs no API key and costs nothing.</p>
        <button type="button" className="primary" onClick={onNewRun}>
          Start a run
        </button>
      </div>
    );
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Experiments</h1>
          <p className="meta">
            {experiments.length} run{experiments.length === 1 ? "" : "s"}, newest first
          </p>
        </div>
        <button type="button" className="primary" onClick={onNewRun}>
          New run
        </button>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Experiment</th>
              <th className="num">Pass rate</th>
              <th className="num hide-sm">Latency p50</th>
              <th className="num hide-sm">Cost</th>
              <th className="num hide-sm">When</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp) => (
              <tr key={exp.id} className="row-link" onClick={() => onOpen(exp.id)}>
                <td>
                  {/* A real button, so the row can be opened from the keyboard too. */}
                  <button
                    type="button"
                    className="row-title"
                    onClick={(event) => {
                      event.stopPropagation();
                      onOpen(exp.id);
                    }}
                  >
                    {exp.name}
                  </button>
                  <div className="meta">
                    {exp.model} · {exp.dataset_name} · {exp.trial_count} trials
                  </div>
                </td>
                <td className={`num ${rateTone(exp.pass_rate)}`}>
                  <strong>{pct(exp.pass_rate)}</strong>
                </td>
                <td className="num hide-sm">{ms(exp.latency_ms.p50)}</td>
                <td className="num hide-sm">{usd(exp.cost)}</td>
                <td className="num hide-sm muted" title={new Date(exp.created_at).toLocaleString()}>
                  {relativeTime(exp.created_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
