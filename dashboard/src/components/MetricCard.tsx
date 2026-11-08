interface Props {
  label: string;
  value: string;
  sub?: string;
}

export default function MetricCard({ label, value, sub }: Props) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub ? <div className="metric-sub">{sub}</div> : null}
    </div>
  );
}
