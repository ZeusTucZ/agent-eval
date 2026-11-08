// Label answers yourself to check an LLM judge, one at a time, as `agenteval calibrate label`
// does in the terminal. You see what the judge saw and never its verdict, so your labels
// stay independent; once both exist, the page shows how far the judge agrees with you.
import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { LabelQueue, Rubric } from "../types";
import { humanize, pct } from "../format";

interface Props {
  id: number;
  token: string;
  onTokenChange: (token: string) => void;
}

export default function LabelPage({ id, token, onTokenChange }: Props) {
  const [rubrics, setRubrics] = useState<Rubric[]>([]);
  const [rubric, setRubric] = useState("correctness");
  const [queue, setQueue] = useState<LabelQueue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = useCallback(() => {
    api.labels(id, rubric).then(setQueue).catch((err: Error) => setError(err.message));
  }, [id, rubric]);

  useEffect(() => {
    api.rubrics().then(setRubrics).catch(() => setRubrics([]));
  }, []);
  useEffect(load, [load]);

  const current = queue?.next[0];
  const chosen = rubrics.find((r) => r.name === rubric);

  async function answer(passed: boolean) {
    if (!current) return;
    setSaving(true);
    setError(null);
    try {
      await api.addLabel(id, current.trial_id, rubric, passed, token);
      load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  // Keyboard: p to pass, f to fail, for labelling quickly.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!token || saving || !current || event.target instanceof HTMLInputElement) return;
      if (event.key === "p") void answer(true);
      if (event.key === "f") void answer(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <>
      <a className="link back" href={`#/e/${id}`}>← Experiment #{id}</a>
      <div className="page-head">
        <div>
          <h1>Label answers</h1>
          <p className="meta">
            Your verdicts check the LLM judge. You see what it saw, never what it decided.
          </p>
        </div>
        <label style={{ minWidth: 200 }}>
          Rubric
          <select value={rubric} onChange={(e) => { setQueue(null); setRubric(e.target.value); }}>
            {(rubrics.length ? rubrics.map((r) => r.name) : ["correctness", "groundedness"]).map((name) => (
              <option key={name} value={name}>{humanize(name)}</option>
            ))}
          </select>
        </label>
      </div>

      {!token ? (
        <label className="form" style={{ marginBottom: 24 }}>
          API token
          <input type="password" value={token} onChange={(e) => onTokenChange(e.target.value)} autoComplete="off" />
          <span className="hint">Needed to save labels. With <code>agenteval ui</code> it&apos;s filled in for you.</span>
        </label>
      ) : null}

      {error ? <p className="notice error">{error}</p> : null}
      {!queue ? <p className="skeleton">Loading…</p> : (
        <>
          <div className="stats" style={{ marginBottom: 32 }}>
            <div className="stat"><div className="stat-label">Labelled</div><div className="stat-value">{queue.labelled}</div>
              <div className="stat-sub">aim for 50</div></div>
            <div className="stat"><div className="stat-label">Judged</div><div className="stat-value">{queue.judged}</div>
              <div className="stat-sub">{queue.judged ? "by the LLM judge" : `run: agenteval judge ${id} --rubric ${rubric}`}</div></div>
            <div className="stat"><div className="stat-label">Agreement</div>
              <div className="stat-value">{queue.agreement ? pct(queue.agreement.agreement, 0) : "–"}</div>
              <div className="stat-sub">{queue.agreement ? `kappa ${queue.agreement.kappa.toFixed(2)}` : "needs both"}</div></div>
          </div>

          {queue.agreement ? <p className="notice info" style={{ marginBottom: 24 }}>{queue.agreement.reading}.</p> : null}

          {current ? (
            <div className="detail label-card">
              {chosen ? <p className="meta">{chosen.question}</p> : null}
              <div>
                <div className="stat-label">Question</div>
                <div>{current.question ?? <span className="muted">(not available)</span>}</div>
              </div>
              {chosen?.uses_expected && current.expected ? (
                <div><div className="stat-label">Expected</div><div className="answer">{current.expected}</div></div>
              ) : null}
              {chosen?.uses_sources ? (
                <div><div className="stat-label">Sources</div>
                  <div className="answer sources">{current.sources || <span className="muted">no tool results</span>}</div></div>
              ) : null}
              {chosen?.uses_answer !== false ? (
                <div><div className="stat-label">Answer</div><div className="answer">{current.answer}</div></div>
              ) : null}
              <div style={{ display: "flex", gap: 8 }}>
                <button type="button" className="primary" disabled={!token || saving} onClick={() => void answer(true)}>
                  Pass <kbd>p</kbd>
                </button>
                <button type="button" disabled={!token || saving} onClick={() => void answer(false)}>
                  Fail <kbd>f</kbd>
                </button>
              </div>
            </div>
          ) : (
            <p className="notice success">
              Nothing left to label for this rubric{queue.labelled ? `: ${queue.labelled} labels saved` : ""}.
            </p>
          )}
        </>
      )}
    </>
  );
}
