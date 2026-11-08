// Failures grouped by failure_type, exactly as the experiment summary
// returns them (failures: {failure_type: count}). Retry reasons are
// computed from the returned trial spans the same way report.py does:
// a retried llm span is blamed on the previous span's error_type
// ("unknown" when the previous span has none).
import type { Trial } from "../types";

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

interface Props {
  failures: Record<string, number>;
  trialCount: number;
  trials: Trial[];
}

export default function FailureBreakdown({ failures, trialCount, trials }: Props) {
  const entries = Object.entries(failures).sort((a, b) => b[1] - a[1]);
  const totalFailed = entries.reduce((n, [, c]) => n + c, 0);
  const retries = retryReasons(trials);
  const retryEntries = Object.entries(retries).sort((a, b) => b[1] - a[1]);

  return (
    <div>
      <p>
        {totalFailed} of {trialCount} trials failed
        {totalFailed > 0 ? ` (${((totalFailed / Math.max(trialCount, 1)) * 100).toFixed(1)}%)` : ""}.
      </p>
      {entries.length === 0 ? (
        <p>No failures recorded.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Failure type</th>
              <th>Trials</th>
              <th>Share</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([type, count]) => {
              const share = (count / Math.max(trialCount, 1)) * 100;
              return (
                <tr key={type}>
                  <td>{type}</td>
                  <td>{count}</td>
                  <td>
                    <span className="share-row">
                      <span className="share-bar">
                        <span style={{ width: `${share}%` }} />
                      </span>
                      {share.toFixed(1)}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <h3>Retries by reason</h3>
      {retryEntries.length === 0 ? (
        <p>No retries recorded.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Reason (previous span error)</th>
              <th>Retries</th>
            </tr>
          </thead>
          <tbody>
            {retryEntries.map(([reason, count]) => (
              <tr key={reason}>
                <td>{reason}</td>
                <td>{count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
