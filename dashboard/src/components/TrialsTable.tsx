// Trials as GET /api/experiments/{id}/trials returns them, with a filter so the ones
// worth reading (the failures) are one click away. Opening a trial shows its answer,
// every step (span) and every score with its reasoning.
import { Fragment, useState } from "react";
import type { Trial } from "../types";
import { humanize, ms, trialDurationMs } from "../format";

type Filter = "all" | "failed" | "passed";

const PAGE = 25; // trials shown at first, and added by "Show more"

function passed(trial: Trial): boolean {
  return !trial.failure_type && trial.scores.some((s) => s.scorer === "exact_match" && s.value === 1);
}

function Status({ trial }: { trial: Trial }) {
  if (trial.failure_type) return <span className="badge bad">{humanize(trial.failure_type)}</span>;
  if (passed(trial)) return <span className="badge good">Correct</span>;
  return <span className="badge warn">Wrong answer</span>;
}

function TrialDetail({ trial }: { trial: Trial }) {
  return (
    <div className="detail">
      {trial.case ? (
        <div className="row">
          <div>
            <div className="stat-label">Question</div>
            <div>{trial.case.input}</div>
          </div>
          <div>
            <div className="stat-label">Expected</div>
            <div className="answer">{trial.case.expected_output || <span className="muted">no reference</span>}</div>
          </div>
        </div>
      ) : null}
      <div>
        <div className="stat-label">Answer</div>
        <div className="answer">{trial.output ?? <span className="muted">No answer</span>}</div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>Status</th>
              <th className="num">Time</th>
              <th className="num">Tokens in / out</th>
            </tr>
          </thead>
          <tbody>
            {trial.spans.map((span, i) => (
              <tr key={i}>
                <td>
                  <span className="mono">{span.name}</span>{" "}
                  <span className="muted">
                    {span.kind === "llm" ? "model" : "tool"}
                    {span.retry > 0 ? ` · retry ${span.retry}` : ""}
                  </span>
                  {span.kind === "tool" && span.input ? (
                    <div className="answer muted truncate" title={span.input}>
                      {span.input}
                    </div>
                  ) : null}
                </td>
                <td className={span.status === "error" ? "bad" : "muted"}>
                  {span.status === "error" ? humanize(span.error_type ?? "error") : "OK"}
                </td>
                <td className="num">{ms((span.end - span.start) / 1_000_000)}</td>
                <td className="num muted">
                  {span.kind === "llm" ? `${span.input_tokens} / ${span.output_tokens}` : "–"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {trial.scores.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Check</th>
                <th>Result</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {trial.scores.map((score) => (
                <tr key={score.scorer}>
                  <td>{humanize(score.scorer)}</td>
                  <td className={score.value === 1 ? "good" : score.value === 0 ? "bad" : "warn"}>
                    {score.value === 1 ? "Pass" : score.value === 0 ? "Fail" : `${Math.round(score.value * 100)}%`}
                  </td>
                  <td className="muted">{score.reasoning ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

export default function TrialsTable({ trials }: { trials: Trial[] }) {
  const failedCount = trials.filter((t) => !passed(t)).length;
  // Start on the failures when there are any: they are what people open this page to read.
  const [filter, setFilter] = useState<Filter>(failedCount > 0 ? "failed" : "all");
  const [openId, setOpenId] = useState<number | null>(null);
  const [limit, setLimit] = useState(PAGE);

  const shown = trials.filter((t) => filter === "all" || (filter === "failed" ? !passed(t) : passed(t)));
  const counts: Record<Filter, number> = {
    all: trials.length,
    failed: failedCount,
    passed: trials.length - failedCount,
  };

  return (
    <>
      <div className="segmented" role="group" aria-label="Show trials" style={{ marginBottom: 16 }}>
        {(["failed", "passed", "all"] as Filter[]).map((option) => (
          <button
            key={option}
            type="button"
            aria-pressed={filter === option}
            onClick={() => {
              setFilter(option);
              setLimit(PAGE);
            }}
          >
            {option === "all" ? "All" : option === "failed" ? "Didn't pass" : "Passed"} {counts[option]}
          </button>
        ))}
      </div>

      {shown.length === 0 ? (
        <p className="muted">No trials here.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Trial</th>
                <th>Result</th>
                <th className="hide-sm">Answer</th>
                <th className="num">Time</th>
              </tr>
            </thead>
            <tbody>
              {shown.slice(0, limit).map((trial) => {
                const open = openId === trial.id;
                return (
                  <Fragment key={trial.id}>
                    <tr className="row-link" onClick={() => setOpenId(open ? null : trial.id)}>
                      <td>
                        <button
                          type="button"
                          className="row-title"
                          aria-expanded={open}
                          onClick={(event) => {
                            event.stopPropagation();
                            setOpenId(open ? null : trial.id);
                          }}
                        >
                          {open ? "▾" : "▸"} {trial.case ? trial.case.input : `Case ${trial.case_id}`}
                        </button>
                        <span className="muted"> · attempt {trial.attempt}</span>
                      </td>
                      <td>
                        <Status trial={trial} />
                      </td>
                      <td className="hide-sm">
                        <div className="answer truncate">{trial.output ?? "–"}</div>
                      </td>
                      <td className="num muted">{ms(trialDurationMs(trial.spans))}</td>
                    </tr>
                    {open ? (
                      <tr>
                        <td colSpan={4} style={{ padding: 0, borderBottom: 0 }}>
                          <TrialDetail trial={trial} />
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
          {shown.length > limit ? (
            <button type="button" className="link" style={{ marginTop: 16 }} onClick={() => setLimit(limit + PAGE)}>
              Show {Math.min(PAGE, shown.length - limit)} more of {shown.length - limit}
            </button>
          ) : null}
        </div>
      )}
    </>
  );
}
