// Is B really better than A? The same paired test as `agenteval compare`, read top to
// bottom: the verdict in plain words, the size of the change with its uncertainty, where it
// moved, and the questions that flipped.
import { useEffect, useState } from "react";
import { api } from "../api";
import type { CaseChange, Comparison, Interval } from "../types";
import { humanize, pct } from "../format";
import Stat from "../components/Stat";

const VERDICT_TEXT: Record<string, { tone: "good" | "bad" | "warn" | undefined; text: string }> = {
  Better: { tone: "good", text: "B is better, and it's unlikely to be chance." },
  Worse: { tone: "bad", text: "B is worse, and it's unlikely to be chance." },
  "Trade-off": { tone: "warn", text: "B answers better, but with new problems: read what got worse." },
  Inconclusive: { tone: undefined, text: "Can't tell: the difference could be chance. This doesn't mean they're the same." },
};

function points(interval: Interval, scale = 100, unit = " pts", digits = 1): string {
  const f = (v: number) => `${v >= 0 ? "+" : ""}${(v * scale).toFixed(digits)}`;
  return `${f(interval.change)}${unit} (95% CI ${f(interval.low)} to ${f(interval.high)})`;
}

function Changes({ title, changes }: { title: string; changes: CaseChange[] }) {
  if (changes.length === 0) return null;
  return (
    <section>
      <h2>{title} ({changes.length})</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Question</th>
              <th className="hide-sm">Expected</th>
              <th>A</th>
              <th>B</th>
            </tr>
          </thead>
          <tbody>
            {changes.map((change) => (
              <tr key={change.case_id}>
                <td>
                  {change.input}
                  <div className="meta">{humanize(change.category)}</div>
                </td>
                <td className="hide-sm answer">{change.expected}</td>
                <td className="answer">
                  {change.before ?? <span className="muted">no answer</span>}
                  <div className="meta">{pct(change.before_rate, 0)} right</div>
                </td>
                <td className="answer">
                  {change.after ?? <span className="muted">no answer</span>}
                  <div className="meta">{pct(change.after_rate, 0)} right</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function ComparePage({ a, b }: { a: number; b: number }) {
  const [result, setResult] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.compare(a, b).then(setResult).catch((err: Error) => setError(err.message));
  }, [a, b]);

  const back = <a className="link back" href="#/">← All experiments</a>;
  if (error) return <>{back}<p className="notice error">Couldn&apos;t compare: {error}</p></>;
  if (!result) return <p className="skeleton">Comparing…</p>;

  const heading = (
    <div className="page-head">
      <div>
        <h1>
          <a href={`#/e/${a}`}>#{a}</a> → <a href={`#/e/${b}`}>#{b}</a>
        </h1>
        <p className="meta">
          A is the baseline, B the change{result.changes.length ? `. Differs in: ${result.changes.join(", ")}` : ""}.
        </p>
      </div>
      <a className="button" href={`#/compare/${b}/${a}`}>Swap</a>
    </div>
  );

  if (result.verdict === null) {
    return (
      <>
        {back}
        {heading}
        <p className="notice error">These two can&apos;t be compared fairly: {result.errors.join("; ")}.</p>
      </>
    );
  }

  const verdict = VERDICT_TEXT[result.verdict];
  const allCategories = Object.entries(result.categories ?? {});
  // With a single category the overall numbers already say it all.
  const categories = allCategories.length > 1 ? allCategories.filter(([, row]) => row.before !== row.after) : [];

  return (
    <>
      {back}
      {heading}

      <p className={`verdict ${verdict.tone ?? ""}`}>
        <strong>{result.verdict}.</strong> {verdict.text}
      </p>
      {result.reason ? <p className="meta" style={{ marginBottom: 24 }}>{result.reason}</p> : null}

      <div className="stats">
        <Stat label="Pass rate, A → B" value={`${pct(result.pass_rate!.before)} → ${pct(result.pass_rate!.after)}`}
              sub={points(result.pass_rate!)} />
        <Stat label="Questions that flipped"
              value={`${result.improved!.length} ↑  ${result.regressed!.length} ↓`}
              sub={`McNemar p = ${result.p_value!.toFixed(3)}${result.partly_changed ? `, ${result.partly_changed} partly` : ""}`} />
        <Stat label="Latency per trial" value={`${result.latency_ms!.change >= 0 ? "+" : ""}${result.latency_ms!.change.toFixed(0)} ms`}
              sub={`CI ${result.latency_ms!.low.toFixed(0)} to ${result.latency_ms!.high.toFixed(0)} ms`} />
        <Stat label="Cases compared" value={String(result.cases_compared)} sub="paired, question by question" />
      </div>

      {result.warnings.length ? (
        <ul className="notice warn warnings">
          {result.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}

      {categories.length ? (
        <section>
          <h2>Categories that moved</h2>
          <p className="section-note">Leads to investigate, not proof: each category has few cases.</p>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Category</th><th className="num">A</th><th className="num">B</th><th className="num">Cases</th></tr></thead>
              <tbody>
                {categories.map(([name, row]) => (
                  <tr key={name}>
                    <td>{humanize(name)}</td>
                    <td className="num">{pct(row.before, 0)}</td>
                    <td className={`num ${row.after > row.before ? "good" : "bad"}`}>{pct(row.after, 0)}</td>
                    <td className="num muted">{row.cases}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <Changes title="New safety failures" changes={result.safety_regressions ?? []} />
      <Changes title="Got worse" changes={result.regressed ?? []} />
      <Changes title="Got better" changes={result.improved ?? []} />
    </>
  );
}
