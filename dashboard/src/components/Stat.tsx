interface Props {
  label: string;
  value: string;
  sub?: string;
  tone?: "good" | "warn" | "bad";
}

/** One headline number. Colour only when the value is good or bad news. */
export default function Stat({ label, value, sub, tone }: Props) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${tone ?? ""}`}>{value}</div>
      {sub ? <div className="stat-sub">{sub}</div> : null}
    </div>
  );
}
