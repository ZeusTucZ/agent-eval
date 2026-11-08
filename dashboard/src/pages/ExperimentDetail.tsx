import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExperimentSummary, Trial } from "../types";
import { ms, pct, usd } from "../format";
import MetricCard from "../components/MetricCard";
import LatencyChart from "../components/LatencyChart";
import CostTable from "../components/CostTable";
import TrialsTable from "../components/TrialsTable";
import FailureBreakdown from "../components/FailureBreakdown";

interface Props {
  id: number;
  token: string;
  onTokenChange: (token: string) => void;
  onBack: () => void;
}

export default function ExperimentDetail({ id, token, onTokenChange, onBack }: Props) {
  const [experiment, setExperiment] = useState<ExperimentSummary | null>(null);
  const [trials, setTrials] = useState<Trial[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rescoreState, setRescoreState] = useState<string | null>(null);

  const load = () => {
    setError(null);
    api
      .getExperiment(id)
      .then(setExperiment)
      .catch((err: Error) => setError(err.message));
    api
      .listTrials(id)
      .then((res) => setTrials(res.trials))
      .catch((err: Error) => setError(err.message));
  };

  useEffect(load, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleRescore() {
    if (!token) {
      setRescoreState("Enter your API token below first — re-scoring requires it.");
      return;
    }
    setRescoreState("Re-scoring…");
    try {
      const updated = await api.rescoreExperiment(id, token);
      setExperiment(updated);
      setRescoreState("Re-scored. Scores were recomputed without calling the model again.");
    } catch (err) {
      setRescoreState(`Re-score failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  if (error) {
    return (
      <div>
        <p className="error">Could not load experiment {id}: {error}</p>
        <button type="button" onClick={onBack}>← Back to experiments</button>
      </div>
    );
  }
  if (!experiment || trials === null) return <p>Loading experiment…</p>;

  const rates = experiment.tool_rates;

  return (
    <div>
      <button type="button" onClick={onBack}>← Back to experiments</button>
      <h2>
        Experiment {experiment.id}: {experiment.name}
      </h2>
      <p>
        {experiment.model} · dataset {experiment.dataset_name}
        {experiment.dataset_version ? ` (v ${experiment.dataset_version})` : ""} ·{" "}
        {experiment.attempts} attempt{experiment.attempts === 1 ? "" : "s"} · config{" "}
        {experiment.config_version} · {new Date(experiment.created_at).toLocaleString()}
      </p>

      <div className="metric-grid">
        <MetricCard label="Pass rate" value={pct(experiment.pass_rate)} sub={`${experiment.trial_count} trials`} />
        <MetricCard label="Total cost" value={usd(experiment.cost)} sub={`per success: ${usd(experiment.cost_per_success, 6)}`} />
        <MetricCard label="Latency p50" value={ms(experiment.latency_ms.p50)} sub={`excl. waits: ${ms(experiment.latency_working_ms.p50)}`} />
        <MetricCard label="Latency p95" value={ms(experiment.latency_ms.p95)} sub={`excl. waits: ${ms(experiment.latency_working_ms.p95)}`} />
      </div>

      <h3>Tool quality</h3>
      <table className="data-table">
        <tbody>
          <tr><th scope="row">Right tool</th><td>{pct(rates.right_tool)}</td></tr>
          <tr><th scope="row">Valid arguments</th><td>{pct(rates.valid_arguments)}</td></tr>
          <tr><th scope="row">Ran without error</th><td>{pct(rates.ran_without_error)}</td></tr>
          <tr><th scope="row">Backed by tools (tool faithfulness)</th><td>{pct(rates.backed_by_tools)}</td></tr>
          <tr><th scope="row">No injected text</th><td>{pct(rates.no_injected_text)}</td></tr>
        </tbody>
      </table>

      <h3>Latency</h3>
      <LatencyChart
        p50={experiment.latency_ms.p50}
        p95={experiment.latency_ms.p95}
        p50Working={experiment.latency_working_ms.p50}
        p95Working={experiment.latency_working_ms.p95}
      />
      <p className="footnote">
        “Excl. waits” leaves out time spent waiting on retries or the rate limiter
        (latency_working_ms from the API).
      </p>

      <h3>Cost</h3>
      <CostTable cost={experiment.cost} costPerSuccess={experiment.cost_per_success} trialCount={experiment.trial_count} />

      <h3>Failures</h3>
      <FailureBreakdown failures={experiment.failures} trialCount={experiment.trial_count} trials={trials} />

      <h3>Trials</h3>
      <TrialsTable trials={trials} />

      <h3>Re-score</h3>
      <p>
        Re-scores the saved trials without calling the model again (no API cost). Requires
        the API token.
      </p>
      <label>
        API token{" "}
        <input
          type="password"
          value={token}
          onChange={(e) => onTokenChange(e.target.value)}
          placeholder="X-API-Token value"
          autoComplete="off"
        />
      </label>{" "}
      <button type="button" onClick={handleRescore}>Re-score experiment</button>
      {rescoreState ? <p>{rescoreState}</p> : null}
    </div>
  );
}
