// How this experiment's pass rate compares with the other runs of the same model on the same
// dataset, in the order they ran. A line, the current run marked, and a link to compare it
// with the one before. Shown only when there are at least two runs to draw.
import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExperimentSummary, TrendPoint } from "../types";
import { pct } from "../format";

const WIDTH = 600;
const HEIGHT = 48;

export default function Trend({ experiment }: { experiment: ExperimentSummary }) {
  const [points, setPoints] = useState<TrendPoint[]>([]);

  useEffect(() => {
    api.trends(experiment.dataset_name, experiment.model).then(setPoints).catch(() => setPoints([]));
  }, [experiment.dataset_name, experiment.model]);

  const rated = points.filter((p) => p.pass_rate !== null);
  if (rated.length < 2) return null;

  const index = rated.findIndex((p) => p.id === experiment.id);
  const previous = index > 0 ? rated[index - 1] : null;
  const x = (i: number) => 4 + (i * (WIDTH - 8)) / (rated.length - 1);
  const y = (rate: number) => HEIGHT - 4 - rate * (HEIGHT - 8);
  const line = rated.map((p, i) => `${x(i)},${y(p.pass_rate as number)}`).join(" ");

  return (
    <section>
      <h2>Over time</h2>
      <p className="section-note">
        Pass rate of {rated.length} runs of {experiment.model} on {experiment.dataset_name.split("#")[0]}, oldest
        first. Hover a point for its run.
      </p>
      <svg className="sparkline" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" role="img"
           aria-label={`Pass rate across ${rated.length} runs`}>
        <polyline points={line} fill="none" stroke="currentColor" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        {rated.map((p, i) => (
          <a key={p.id} href={`#/e/${p.id}`}>
            <circle cx={x(i)} cy={y(p.pass_rate as number)} r={p.id === experiment.id ? 5 : 3}
                    fill={p.id === experiment.id ? "currentColor" : "var(--bg)"} stroke="currentColor" strokeWidth="2">
              <title>{`#${p.id} ${p.name}: ${pct(p.pass_rate)}`}</title>
            </circle>
          </a>
        ))}
      </svg>
      {previous ? (
        <p className="meta" style={{ marginTop: 8 }}>
          The run before this one, <a href={`#/e/${previous.id}`}>#{previous.id} {previous.name}</a>, passed{" "}
          {pct(previous.pass_rate)}.{" "}
          <a href={`#/compare/${previous.id}/${experiment.id}`}>Is this run really better or worse?</a>
        </p>
      ) : null}
    </section>
  );
}
