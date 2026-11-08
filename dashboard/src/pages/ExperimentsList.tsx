import { useEffect, useState } from "react";
import { api, PAGE_SIZE } from "../api";
import { go } from "../App";
import type { ExperimentSummary } from "../types";
import { ms, pct, rateTone, relativeTime, usd } from "../format";

export default function ExperimentsList() {
  const [experiments, setExperiments] = useState<ExperimentSummary[] | null>(null);
  const [more, setMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Two ticked experiments can be compared: the first ticked is the baseline.
  const [picked, setPicked] = useState<number[]>([]);

  useEffect(() => {
    api
      .listExperiments()
      .then((page) => {
        setExperiments(page);
        setMore(page.length === PAGE_SIZE);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  async function loadMore() {
    if (!experiments) return;
    try {
      const page = await api.listExperiments(experiments.length);
      setExperiments([...experiments, ...page]);
      setMore(page.length === PAGE_SIZE);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function toggle(id: number) {
    setPicked((current) =>
      current.includes(id) ? current.filter((x) => x !== id) : [...current, id].slice(-2),
    );
  }

  if (error) return <p className="notice error">Couldn&apos;t load experiments: {error}</p>;
  if (experiments === null) return <p className="skeleton">Loading experiments…</p>;

  if (experiments.length === 0) {
    return (
      <div className="empty">
        <h1>No experiments yet</h1>
        <p>
          Try the mock agent here, or evaluate your own from the terminal with{" "}
          <code>agenteval init</code> and <code>agenteval run</code>.
        </p>
        <button type="button" className="primary" onClick={() => go("run")}>
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
          <p className="meta">Newest first. Tick two to compare them.</p>
        </div>
        {picked.length === 2 ? (
          <button type="button" className="primary" onClick={() => go(`compare/${picked[0]}/${picked[1]}`)}>
            Compare #{picked[0]} → #{picked[1]}
          </button>
        ) : (
          <button type="button" className="primary" onClick={() => go("run")}>
            New run
          </button>
        )}
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th style={{ width: 28 }}><span className="sr-only">Compare</span></th>
              <th>Experiment</th>
              <th className="num">Pass rate</th>
              <th className="num hide-sm">Latency p50</th>
              <th className="num hide-sm">Cost</th>
              <th className="num hide-sm">When</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp) => (
              <tr key={exp.id} className="row-link" onClick={() => go(`e/${exp.id}`)}>
                <td onClick={(event) => event.stopPropagation()}>
                  <input
                    type="checkbox"
                    className="pick"
                    aria-label={`Compare ${exp.name}`}
                    checked={picked.includes(exp.id)}
                    onChange={() => toggle(exp.id)}
                  />
                </td>
                <td>
                  <a className="row-title" href={`#/e/${exp.id}`} onClick={(event) => event.stopPropagation()}>
                    {exp.name}
                  </a>
                  {exp.status && exp.status !== "completed" ? (
                    <span className="badge warn" style={{ marginLeft: 8 }}>{exp.status}</span>
                  ) : null}
                  <div className="meta">
                    #{exp.id} · {exp.model} · {exp.dataset_name} · {exp.trial_count} trials
                  </div>
                </td>
                <td className={`num ${exp.pass_rate === null ? "muted" : rateTone(exp.pass_rate)}`}>
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
      {more ? (
        <button type="button" className="link" style={{ marginTop: 16 }} onClick={() => void loadMore()}>
          Show older experiments
        </button>
      ) : null}
    </>
  );
}
