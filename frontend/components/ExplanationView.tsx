"use client";

import { displayValue } from "@/lib/format";
import type { Explanation } from "@/lib/types";

/** SHAP contributions as a diverging bar list. Bars are drawn to one shared scale so lengths are
 *  comparable; numbers are on the stated output scale (e.g. calibrated log-odds). */
export function ExplanationView({ explanation, status }: { explanation: Explanation | null; status: string }) {
  if (status === "pending") return <p className="text-muted">Explanation is being generated in the background. Refresh in a moment.</p>;
  if (status === "not_requested") return <p className="text-muted">No explanation was requested for this prediction.</p>;
  if (!explanation || explanation.status !== "ok") {
    return (
      <div className="rounded-md border border-line bg-panel-2 px-3 py-2 text-sm">
        <p className="font-medium">Explanation {explanation?.status ?? status}</p>
        <p className="text-muted">{explanation?.reason ?? "The explainer did not return a result."} No explanation is shown rather than a guessed one.</p>
      </div>
    );
  }
  const rows = explanation.contributions;
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-9);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted">
        <span>
          Scale: <span className="text-ink">{explanation.output_scale}</span>
        </span>
        <span>
          Base value <span className="num text-ink">{explanation.base_value?.toFixed(3)}</span>
        </span>
        <span>
          Output <span className="num text-ink">{explanation.output_value?.toFixed(3)}</span>
        </span>
      </div>
      <ul className="space-y-1.5" aria-label="Feature contributions">
        {rows.map((r) => {
          const w = `${(Math.abs(r.contribution) / max) * 50}%`;
          const up = r.contribution > 0;
          return (
            <li key={r.feature} className="grid grid-cols-[minmax(9rem,14rem)_1fr_4.5rem] items-center gap-3 text-sm" title={r.description}>
              <div className="min-w-0">
                <div className="truncate font-medium">{r.feature}</div>
                <div className="num truncate text-xs text-muted">= {displayValue(r.value)}</div>
              </div>
              <div className="relative h-4 rounded-sm bg-panel-2">
                <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                <div
                  className={`absolute inset-y-0 rounded-sm ${up ? "left-1/2 bg-increase" : "right-1/2 bg-decrease"}`}
                  style={{ width: w }}
                />
              </div>
              <div className={`num text-right ${up ? "text-increase" : "text-decrease"}`}>
                {r.contribution > 0 ? "+" : ""}
                {r.contribution.toFixed(3)}
              </div>
            </li>
          );
        })}
      </ul>
      <div className="flex justify-between text-xs text-muted">
        <span>◀ lowers the score</span>
        <span>raises the score ▶</span>
      </div>
      {explanation.summary && <p className="text-sm">{explanation.summary}</p>}
      {rows.some((r) => r.components.length > 1) && (
        <p className="text-xs text-muted">
          Categorical features combine the contributions of their one-hot columns (SHAP values are additive), so one
          bar shows the whole feature.
        </p>
      )}
      <p className="text-xs text-muted">{explanation.disclaimer}</p>
    </div>
  );
}
