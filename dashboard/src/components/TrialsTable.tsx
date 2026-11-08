// Trials as GET /api/experiments/{id}/trials returns them: id, case_id,
// attempt, output, failure_type, spans (kind, name, start, end, status,
// error_type, input, output, tokens, retry) and scores (scorer, value,
// reasoning). Trial duration is derived from its spans (see format.ts).
import { useState } from "react";
import type { Trial } from "../types";
import { ms, trialDurationMs } from "../format";

function scoreValue(trial: Trial, scorer: string): string {
  const found = trial.scores.find((s) => s.scorer === scorer);
  return found ? `${(found.value * 100).toFixed(0)}%` : "—";
}

export default function TrialsTable({ trials }: { trials: Trial[] }) {
  const [openId, setOpenId] = useState<number | null>(null);

  if (trials.length === 0) {
    return <p>No trials recorded for this experiment.</p>;
  }

  return (
    <>
      <table className="data-table">
        <thead>
          <tr>
            <th>Trial</th>
            <th>Case</th>
            <th>Attempt</th>
            <th>Status</th>
            <th>Duration*</th>
            <th>Spans</th>
            <th>Exact match</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {trials.flatMap((trial) => {
            const open = openId === trial.id;
            const retries = trial.spans.reduce((n, s) => n + s.retry, 0);
            const rows = [
              <tr key={trial.id}>
                <td>#{trial.id}</td>
                <td>{trial.case_id}</td>
                <td>{trial.attempt}</td>
                <td>
                  <span className={`pill ${trial.failure_type ? "pill-bad" : "pill-good"}`}>
                    {trial.failure_type ?? "ok"}
                  </span>
                </td>
                <td>{ms(trialDurationMs(trial.spans))}</td>
                <td>
                  {trial.spans.length}
                  {retries > 0 ? ` (${retries} retr${retries === 1 ? "y" : "ies"})` : ""}
                </td>
                <td>{scoreValue(trial, "exact_match")}</td>
                <td>
                  <button
                    type="button"
                    onClick={() => setOpenId(open ? null : trial.id)}
                    aria-expanded={open}
                  >
                    {open ? "Hide" : "Spans & scores"}
                  </button>
                </td>
              </tr>,
            ];
            if (open) {
              rows.push(
                <tr key={`${trial.id}-detail`}>
                  <td colSpan={8}>
                    {trial.output !== null && trial.output !== "" ? (
                      <p>
                        <strong>Output:</strong> {trial.output}
                      </p>
                    ) : null}
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Span</th>
                          <th>Kind</th>
                          <th>Status</th>
                          <th>Duration</th>
                          <th>Retry</th>
                          <th>Error</th>
                          <th>Tokens in/out</th>
                        </tr>
                      </thead>
                      <tbody>
                        {trial.spans.map((span, i) => (
                          <tr key={i}>
                            <td>{span.name}</td>
                            <td>{span.kind}</td>
                            <td>{span.status}</td>
                            <td>{ms((span.end - span.start) / 1_000_000)}</td>
                            <td>{span.retry}</td>
                            <td>{span.error_type ?? "—"}</td>
                            <td>
                              {span.kind === "llm"
                                ? `${span.input_tokens} / ${span.output_tokens}`
                                : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {trial.scores.length > 0 ? (
                      <ul>
                        {trial.scores.map((score, i) => (
                          <li key={i}>
                            <strong>{score.scorer}:</strong> {(score.value * 100).toFixed(0)}%
                            {score.reasoning ? ` — ${score.reasoning}` : ""}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p>No scores for this trial.</p>
                    )}
                  </td>
                </tr>,
              );
            }
            return rows;
          })}
        </tbody>
      </table>
      <p className="footnote">
        * Duration is derived from the trial&apos;s spans (last end − first start); the API
        returns spans, not a per-trial duration.
      </p>
    </>
  );
}
