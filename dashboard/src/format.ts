// Formatting helpers for values the API returns. A null rate means the
// scorer produced no scores for that experiment (e.g. tool_faithfulness is
// null for arithmetic runs); show "n/a", never 0%.

export function pct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "n/a";
  return `${(value * 100).toFixed(1)}%`;
}

export function usd(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined) return "n/a";
  return `$${value.toFixed(digits)}`;
}

export function ms(value: number): string {
  return `${value.toFixed(0)} ms`;
}

/** Trial duration in ms, derived from the returned spans (first start to
 * last end). Trial itself carries no duration field in the API response. */
export function trialDurationMs(spans: { start: number; end: number }[]): number {
  if (spans.length === 0) return 0;
  const start = Math.min(...spans.map((s) => s.start));
  const end = Math.max(...spans.map((s) => s.end));
  return (end - start) / 1_000_000;
}
