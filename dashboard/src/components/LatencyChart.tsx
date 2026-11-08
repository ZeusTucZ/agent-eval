// Simple SVG bar chart of the four latency numbers the API returns:
// p50/p95 total and p50/p95 excluding waits (latency_working_ms).
// No chart library, no invented data — the bars are exactly these values.
import { ms } from "../format";

interface Props {
  p50: number;
  p95: number;
  p50Working: number;
  p95Working: number;
}

export default function LatencyChart({ p50, p95, p50Working, p95Working }: Props) {
  const bars = [
    { label: "p50 total", value: p50 },
    { label: "p95 total", value: p95 },
    { label: "p50 excl. waits", value: p50Working },
    { label: "p95 excl. waits", value: p95Working },
  ];
  const max = Math.max(...bars.map((b) => b.value), 1);
  const width = 560;
  const rowH = 34;
  const height = bars.length * rowH + 10;

  return (
    <svg
      className="latency-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Latency p50 and p95, total and excluding waits"
    >
      {bars.map((bar, i) => {
        const barWidth = Math.max((bar.value / max) * 380, bar.value > 0 ? 2 : 0);
        const y = 10 + i * rowH;
        return (
          <g key={bar.label}>
            <text x={0} y={y + 15} className="chart-label">
              {bar.label}
            </text>
            <rect x={130} y={y} width={barWidth} height={20} rx={3} className="chart-bar" />
            <text x={140 + barWidth} y={y + 15} className="chart-value">
              {ms(bar.value)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
