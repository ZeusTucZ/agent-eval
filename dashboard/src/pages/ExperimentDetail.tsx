// One experiment, ordered by what a reader wants to know: how good was it, why did
// trials fail, which checks failed, and then the trials themselves. Identifiers, the
// latency breakdown and re-scoring sit in "Details" at the end.
import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExperimentSummary, Trial } from "../types";
import { humanize, ms, pct, rateTone, relativeTime, usd } from "../format";
import Stat from "../components/Stat";
import BarRow from "../components/BarRow";
import FailureBreakdown, { wrongAnswers } from "../components/FailureBreakdown";
import TrialsTable from "../components/TrialsTable";

interface Props {
  id: number;
  token: string;
  onTokenChange: (token: string) => void;
  onBack: () => void;
}

// Plain-language names for the checks, in the order they happen in a trial.
const CHECKS: { key: keyof ExperimentSummary["tool_rates"]; label: string }[] = [
  { key: "right_tool", label: "Used the right tool" },
  { key: "valid_arguments", label: "Valid tool arguments" },
  { key: "ran_without_error", label: "Tool ran without error" },
  { key: "relevant_found", label: "Search found the documents" },
  { key: "backed_by_tools", label: "Answer backed by tool results" },
  { key: "no_injected_text", label: "Ignored injected instructions" },
];

const DIAGNOSES: { key: string; label: string; tone?: "bad" }[] = [
  { key: "correct", label: "Found the documents, answered right" },
  { key: "answer_failure", label: "Found the documents, answered wrong", tone: "bad" },
  { key: "retrieval_failure", label: "Missed a document, answered wrong", tone: "bad" },
  { key: "unsupported_correct", label: "Missed a document, right anyway" },
];

export default function ExperimentDetail({ id, token, onTokenChange, onBack }: Props) {
  const [experiment, setExperiment] = useState<ExperimentSummary | null>(null);
  const [trials, setTrials] = useState<Trial[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rescore, setRescore] = useState<{ kind: "info" | "success" | "error"; text: string } | null>(null);

  useEffect(() => {
    setError(null);
    Promise.all([api.getExperiment(id), api.listTrials(id)])
      .then(([summary, list]) => {
        setExperiment(summary);
        setTrials(list.trials);
      })
      .catch((err: Error) => setError(err.message));
  }, [id]);

  async function handleRescore() {
    setRescore({ kind: "info", text: "Re-scoring…" });
    try {
      setExperiment(await api.rescoreExperiment(id, token));
      setTrials((await api.listTrials(id)).trials);
      setRescore({ kind: "success", text: "Scores recomputed from the saved trials." });
    } catch (err) {
      setRescore({ kind: "error", text: `Re-scoring failed: ${err instanceof Error ? err.message : String(err)}` });
    }
  }

  const back = (
    <button type="button" className="link" onClick={onBack} style={{ marginBottom: 20 }}>
      ← All experiments
    </button>
  );

  if (error) {
    return (
      <>
        {back}
        <p className="notice error">Couldn&apos;t load experiment {id}: {error}</p>
      </>
    );
  }
  if (!experiment || trials === null) return <p className="skeleton">Loading experiment…</p>;

  const total = experiment.trial_count;
  const passedCount = Math.round(experiment.pass_rate * total);
  const failureCount = Object.values(experiment.failures).reduce((n, c) => n + c, 0);
  const notPassed = total - passedCount;
  const topCause = Object.entries(experiment.failures).sort((a, b) => b[1] - a[1])[0];
  const wrong = wrongAnswers(trials);
  // A check that produced no scores doesn't apply to this dataset: leave it out rather than show n/a.
  const checks = CHECKS.filter((check) => (experiment.tool_rates[check.key] ?? null) !== null);
  const diagnosis = experiment.retrieval_diagnosis ?? {};
  const diagnosed = Object.values(diagnosis).reduce((n, c) => n + c, 0);

  return (
    <>
      {back}
      <div className="page-head">
        <div>
          <h1>{experiment.name}</h1>
          <p className="meta">
            {experiment.model} · {experiment.dataset_name} · {experiment.attempts} attempt
            {experiment.attempts === 1 ? "" : "s"} per question ·{" "}
            <span title={new Date(experiment.created_at).toLocaleString()}>{relativeTime(experiment.created_at)}</span>
          </p>
        </div>
      </div>

      <div className="stats">
        <Stat
          label="Pass rate"
          value={pct(experiment.pass_rate)}
          sub={`${passedCount} of ${total} trials`}
          tone={rateTone(experiment.pass_rate)}
        />
        <Stat
          label="Didn't pass"
          value={String(notPassed)}
          sub={
            notPassed === 0
              ? "none"
              : wrong >= (topCause?.[1] ?? 0)
                ? `mostly wrong answers`
                : `mostly ${humanize(topCause[0]).toLowerCase()}`
          }
        />
        <Stat
          label="Typical latency"
          value={ms(experiment.latency_ms.p50)}
          sub={`slowest 5%: ${ms(experiment.latency_ms.p95)}`}
        />
        <Stat
          label="Cost per correct answer"
          value={usd(experiment.cost_per_success)}
          sub={`${usd(experiment.cost)} in total`}
        />
      </div>

      <section>
        <h2>Why trials didn&apos;t pass</h2>
        <p className="section-note">
          Share of all {total} trials. A failure stopped the trial; a wrong answer finished with the wrong result.
        </p>
        <FailureBreakdown failures={experiment.failures} trials={trials} />
      </section>

      {diagnosed > 0 ? (
        <section>
          <h2>Search or answer?</h2>
          <p className="section-note">
            For questions answered from documents: was the problem finding them, or using them?
          </p>
          <div className="bars">
            {DIAGNOSES.map((row) => (
              <BarRow
                key={row.key}
                label={row.label}
                share={(diagnosis[row.key] ?? 0) / diagnosed}
                value={String(diagnosis[row.key] ?? 0)}
                tone={row.tone ?? "accent"}
              />
            ))}
          </div>
        </section>
      ) : null}

      {checks.length > 0 ? (
        <section>
          <h2>Checks</h2>
          <p className="section-note">Share of trials that passed each check that applies to this dataset.</p>
          <div className="bars">
            {checks.map((check) => {
              const rate = experiment.tool_rates[check.key] ?? null;
              return <BarRow key={check.key} label={check.label} share={rate} value={pct(rate, 0)} tone="accent" />;
            })}
          </div>
        </section>
      ) : null}

      <section>
        <h2>Trials</h2>
        <TrialsTable trials={trials} />
      </section>

      <section>
        <details>
          <summary>Details and re-scoring</summary>
          <div style={{ display: "grid", gap: 28 }}>
            <table>
              <tbody>
                <tr><td className="muted">Experiment</td><td className="mono">#{experiment.id}</td></tr>
                <tr><td className="muted">Config version</td><td className="mono">{experiment.config_version}</td></tr>
                <tr><td className="muted">Dataset version</td><td className="mono">{experiment.dataset_version ?? "n/a"}</td></tr>
                <tr>
                  <td className="muted">Latency p50 / p95</td>
                  <td>{ms(experiment.latency_ms.p50)} / {ms(experiment.latency_ms.p95)}</td>
                </tr>
                <tr>
                  <td className="muted">Excluding waits</td>
                  <td>
                    {ms(experiment.latency_working_ms.p50)} / {ms(experiment.latency_working_ms.p95)}{" "}
                    <span className="muted">(time not spent on retries or the rate limiter)</span>
                  </td>
                </tr>
                <tr><td className="muted">Failed trials</td><td>{failureCount}</td></tr>
                <tr><td className="muted">Cost per trial</td><td>{usd(total ? experiment.cost / total : null)}</td></tr>
              </tbody>
            </table>

            <form
              className="form"
              onSubmit={(event) => {
                event.preventDefault();
                void handleRescore();
              }}
            >
              <div>
                <h2>Re-score</h2>
                <p className="meta">
                  Runs the scorers again on the saved trials, for when a scorer has changed. The model isn&apos;t
                  called, so it costs nothing.
                </p>
              </div>
              <label>
                API token
                <input
                  type="password"
                  value={token}
                  onChange={(e) => onTokenChange(e.target.value)}
                  autoComplete="off"
                  required
                />
              </label>
              <div>
                <button type="submit" disabled={!token || rescore?.text === "Re-scoring…"}>
                  Re-score
                </button>
              </div>
              {rescore ? (
                <p className={`notice ${rescore.kind}`} role="status">
                  {rescore.text}
                </p>
              ) : null}
            </form>
          </div>
        </details>
      </section>
    </>
  );
}
