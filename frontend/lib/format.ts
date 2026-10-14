import type { ScoreKind } from "./types";

export function formatScore(score: number, kind: ScoreKind): string {
  if (kind === "anomaly_score") return score.toFixed(3); // a score, not a percentage
  if (score > 0 && score < 0.0001) return "<0.01%";
  if (score > 0.999 && score < 1) return ">99.9%"; // never round a probability up to certainty
  return `${(score * 100).toFixed(score < 0.1 ? 2 : 1)}%`;
}

export function scoreLabel(kind: ScoreKind): string {
  return kind === "anomaly_score" ? "Anomaly score" : kind === "constant_prior" ? "Prior rate" : "Risk (calibrated)";
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "-";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

export function pct(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined ? "-" : `${(x * 100).toFixed(digits)}%`;
}

export function num(x: number | null | undefined, digits = 3): string {
  return x === null || x === undefined || Number.isNaN(x) ? "-" : x.toFixed(digits);
}

export function shortVersion(v: string): string {
  // synthetic-xgboost-20261014T114016Z -> xgboost · 10-14 11:40
  const m = v.match(/^([a-z]+)-([a-z_]+)-(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/);
  return m ? `${m[2]} · ${m[4]}-${m[5]} ${m[6]}:${m[7]}` : v;
}

export function displayValue(v: unknown): string {
  if (v === null || v === undefined) return "missing";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return String(v);
}
