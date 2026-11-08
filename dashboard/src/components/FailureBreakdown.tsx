// Why trials didn't pass. Two kinds, told apart because they need different fixes:
// failures (the trial stopped: a tool error, a timeout, a rate limit) as the experiment
// summary returns them, and wrong answers (it finished, exact_match says the answer is
// wrong), counted from the returned trials.
import type { Trial } from "../types";
import { humanize, pct } from "../format";
import BarRow from "./BarRow";

/** Retries by the reason the previous attempt failed, the same way report.py counts them. */
export function retryReasons(trials: Trial[]): Record<string, number> {
  const reasons: Record<string, number> = {};
  for (const trial of trials) {
    let previous: Trial["spans"][number] | null = null;
    for (const span of trial.spans) {
      if (span.kind !== "llm") continue;
      if (span.retry > 0 && previous !== null) {
        const reason = previous.error_type ?? "unknown";
        reasons[reason] = (reasons[reason] ?? 0) + 1;
      }
      previous = span;
    }
  }
  return reasons;
}

export function wrongAnswers(trials: Trial[]): number {
  return trials.filter(
    (trial) => !trial.failure_type && trial.scores.some((s) => s.scorer === "exact_match" && s.value < 1),
  ).length;
}

interface Props {
  failures: Record<string, number>;
  trials: Trial[];
}

export default function FailureBreakdown({ failures, trials }: Props) {
  const total = Math.max(trials.length, 1);
  const rows = Object.entries(failures).map(([type, count]) => ({ label: humanize(type), count }));
  const wrong = wrongAnswers(trials);
  if (wrong > 0) rows.push({ label: "Wrong answer", count: wrong });
  rows.sort((a, b) => b.count - a.count);

  const retries = Object.entries(retryReasons(trials)).sort((a, b) => b[1] - a[1]);
  const retryCount = retries.reduce((n, [, c]) => n + c, 0);

  if (rows.length === 0) {
    return <p className="notice success">Every trial finished and answered correctly.</p>;
  }

  return (
    <>
      <div className="bars">
        {rows.map((row) => (
          <BarRow
            key={row.label}
            label={row.label}
            share={row.count / total}
            value={`${row.count} · ${pct(row.count / total, 0)}`}
            tone="bad"
          />
        ))}
      </div>
      {retryCount > 0 ? (
        <p className="meta" style={{ marginTop: 14 }}>
          {retryCount} request{retryCount === 1 ? " was" : "s were"} retried (
          {retries.map(([reason, count]) => `${humanize(reason).toLowerCase()} ${count}`).join(", ")}).
          Retried requests that then succeeded don&apos;t count as failures.
        </p>
      ) : null}
    </>
  );
}
