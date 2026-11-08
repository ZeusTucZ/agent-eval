interface Props {
  label: string;
  share: number | null; // 0 to 1; null when there is nothing to measure
  value: string;
  tone?: "accent" | "bad";
}

/** A label, a bar for its share, and the number itself, so the bar is never the only cue. */
export default function BarRow({ label, share, value, tone }: Props) {
  return (
    <div className="bar-row">
      <span>{label}</span>
      <span className={`bar ${tone ?? ""}`} aria-hidden="true">
        <span style={{ width: `${Math.max(0, Math.min(1, share ?? 0)) * 100}%` }} />
      </span>
      <span className={`value ${share === null ? "muted" : ""}`}>{value}</span>
    </div>
  );
}
