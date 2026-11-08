// Formatting helpers for values the API returns. A null rate means the
// scorer produced no scores for that experiment (e.g. tool_faithfulness is
// null for arithmetic runs); show "n/a", never 0%.

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "n/a";
  return `${(value * 100).toFixed(digits)}%`;
}

/** Dollars with enough digits to be readable at any size: $0.0774, $0.000018. */
export function usd(value: number | null | undefined): string {
  if (value === null || value === undefined) return "n/a";
  if (value === 0) return "$0";
  if (value >= 1) return `$${value.toFixed(2)}`;
  return `$${value.toPrecision(2)}`;
}

export function ms(value: number): string {
  if (value >= 10_000) return `${(value / 1000).toFixed(1)} s`;
  return `${value.toFixed(0)} ms`;
}

/** "3 min ago", "yesterday", or a date for anything older than a week. */
export function relativeTime(iso: string, now: Date = new Date()): string {
  const then = new Date(iso);
  const seconds = (now.getTime() - then.getTime()) / 1000;
  const format = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  if (seconds < 60) return "just now";
  if (seconds < 3600) return format.format(-Math.round(seconds / 60), "minute");
  if (seconds < 86_400) return format.format(-Math.round(seconds / 3600), "hour");
  if (seconds < 7 * 86_400) return format.format(-Math.round(seconds / 86_400), "day");
  return then.toLocaleDateString("en", { day: "numeric", month: "short", year: "numeric" });
}

/** Green from 90%, amber from 70%, red below: the same cut-offs everywhere. */
export function rateTone(rate: number): "good" | "warn" | "bad" {
  if (rate >= 0.9) return "good";
  if (rate >= 0.7) return "warn";
  return "bad";
}

/** Failure labels as the API returns them ("timeout_after_rate_limit"), made readable. */
export function humanize(label: string): string {
  const text = label.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Trial duration in ms, derived from the returned spans (first start to
 * last end). Trial itself carries no duration field in the API response. */
export function trialDurationMs(spans: { start: number; end: number }[]): number {
  if (spans.length === 0) return 0;
  const start = Math.min(...spans.map((s) => s.start));
  const end = Math.max(...spans.map((s) => s.end));
  return (end - start) / 1_000_000;
}
