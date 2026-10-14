"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { useMe } from "@/components/AppShell";
import { Button, EmptyState, ErrorState, inputClass, Label, Loading, PageHeader, Panel, Pill, SyntheticBadge } from "@/components/ui";
import { api, errorText } from "@/lib/api";
import { formatDate, num, pct, shortVersion } from "@/lib/format";
import type { Evaluation, EvaluationListItem, EvaluationResult, ModelSummary } from "@/lib/types";

const SERIES_COLORS: Record<string, string> = {
  xgboost: "var(--accent)",
  logreg: "var(--ok)",
  isolation_forest: "var(--risk)",
  dummy: "var(--muted)",
};

export default function ModelsPage() {
  const evals = useQuery({ queryKey: ["evaluations"], queryFn: () => api<EvaluationListItem[]>("/evaluations") });
  const [chosen, setChosen] = useState<string | null>(null);
  // Default to the newest benchmark (non-synthetic) session.
  const current = chosen ?? evals.data?.find((e) => !e.synthetic)?.id ?? evals.data?.[0]?.id ?? null;
  const evaluation = useQuery({
    queryKey: ["evaluation", current],
    queryFn: () => api<Evaluation>(`/evaluations/${current}`),
    enabled: !!current,
  });

  return (
    <>
      <PageHeader
        title="Model comparison"
        subtitle="Offline evaluation from training. Decisions (tuning, calibration, threshold, selection) used validation data only; the test set was scored once for these numbers."
        actions={
          evals.data && evals.data.length > 0 ? (
            <div className="w-80">
              <Label htmlFor="evaluation">Training session</Label>
              <select id="evaluation" className={inputClass} value={current ?? ""} onChange={(e) => setChosen(e.target.value)}>
                {evals.data.map((e) => (
                  <option key={e.id} value={e.id}>
                    {e.dataset} {e.synthetic ? "(synthetic)" : "(benchmark)"} · {formatDate(e.created_at)}
                  </option>
                ))}
              </select>
            </div>
          ) : undefined
        }
      />
      {evals.isPending ? (
        <Loading />
      ) : evals.isError ? (
        <ErrorState error={evals.error} />
      ) : evals.data.length === 0 ? (
        <EmptyState title="No training sessions yet">Run <code className="num">tg train creditcard</code> to produce a comparison.</EmptyState>
      ) : evaluation.isPending ? (
        <Loading />
      ) : evaluation.isError ? (
        <ErrorState error={evaluation.error} />
      ) : (
        <EvaluationView ev={evaluation.data} />
      )}
      <Registry />
    </>
  );
}

function EvaluationView({ ev }: { ev: Evaluation }) {
  const [focus, setFocus] = useState(ev.selection.selected);
  const focused = ev.results.find((r) => r.model === focus) ?? ev.results[0]!;
  const prData = mergeCurves(ev.results.filter((r) => r.model !== "dummy"));
  return (
    <div className="space-y-4">
      {ev.synthetic && (
        <p className="rounded-md border border-synthetic/40 bg-synthetic-soft px-3 py-2 text-sm text-synthetic">
          Synthetic session. The fraud patterns were designed by the generator, so these scores are high by construction and say
          nothing about real-world performance.
        </p>
      )}
      <Panel title={<span className="flex items-center gap-2">Data split <SyntheticBadge synthetic={ev.synthetic} /></span>}>
        <div className="grid gap-3 sm:grid-cols-3">
          {(["train", "valid", "test"] as const).map((k) => (
            <div key={k} className="rounded-md bg-panel-2 px-3 py-2">
              <div className="text-xs uppercase tracking-wide text-muted">{k === "valid" ? "validation" : k}</div>
              <div className="num">{ev.split[k].rows.toLocaleString()} rows</div>
              <div className="num text-sm text-muted">
                {ev.split[k].positives} positives · {pct(ev.split[k].prevalence, 3)}
              </div>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted">
          Split: {String(ev.split.config.strategy)} · Primary metric: {ev.primary_metric}. A model that ignores its inputs scores the prevalence.
        </p>
      </Panel>

      <Panel title="Results on the untouched test set">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[60rem] text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
                <th className="py-2 pr-3 font-medium">Model</th>
                <th className="py-2 pr-3 font-medium">Score type</th>
                <th className="py-2 pr-3 text-right font-medium">Val AP</th>
                <th className="py-2 pr-3 text-right font-medium">Test AP (95% CI)</th>
                <th className="py-2 pr-3 text-right font-medium">Precision</th>
                <th className="py-2 pr-3 text-right font-medium">Recall</th>
                <th className="py-2 pr-3 text-right font-medium">F1</th>
                <th className="py-2 pr-3 text-right font-medium">FPR</th>
                <th className="py-2 pr-3 text-right font-medium">TP / FP / FN</th>
                <th className="py-2 pr-3 text-right font-medium">ROC-AUC</th>
                <th className="py-2 text-right font-medium">p50 / p95 ms</th>
              </tr>
            </thead>
            <tbody>
              {ev.results.map((r) => (
                <tr
                  key={r.model}
                  onClick={() => setFocus(r.model)}
                  className={`cursor-pointer border-b border-line last:border-0 ${focus === r.model ? "bg-panel-2" : "hover:bg-panel-2"}`}
                >
                  <td className="py-2 pr-3 font-medium">
                    {r.model} {r.selected && <Pill tone="accent">selected</Pill>}
                  </td>
                  <td className="py-2 pr-3 text-xs text-muted">{r.score_kind.replace("_", " ")}</td>
                  <td className="num py-2 pr-3 text-right">{num(r.validation.average_precision)}</td>
                  <td className="num py-2 pr-3 text-right">
                    {num(r.test.average_precision)}
                    {r.test.average_precision_ci95 && (
                      <span className="text-xs text-muted">
                        {" "}
                        [{num(r.test.average_precision_ci95[0], 2)}–{num(r.test.average_precision_ci95[1], 2)}]
                      </span>
                    )}
                  </td>
                  <td className="num py-2 pr-3 text-right">{num(r.test.precision)}</td>
                  <td className="num py-2 pr-3 text-right">{num(r.test.recall)}</td>
                  <td className="num py-2 pr-3 text-right">{num(r.test.f1)}</td>
                  <td className="num py-2 pr-3 text-right">{num(r.test.false_positive_rate, 5)}</td>
                  <td className="num py-2 pr-3 text-right">
                    {r.test.confusion_matrix.tp} / {r.test.confusion_matrix.fp} / {r.test.confusion_matrix.fn}
                  </td>
                  <td className="num py-2 pr-3 text-right">{num(r.test.roc_auc)}</td>
                  <td className="num py-2 text-right">{r.latency ? `${r.latency.single_row_ms_p50.toFixed(1)} / ${r.latency.single_row_ms_p95.toFixed(1)}` : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 space-y-1 text-xs text-muted">
          <p>Selection rule: {ev.selection.rule}.</p>
          <p>
            Thresholds were chosen on validation data ({String(focused.threshold_policy.name)}
            {focused.threshold_policy.name === "min_precision" && ` ≥ ${String(focused.threshold_policy.min_precision)}`}). Isolation Forest outputs an anomaly
            score; compare it only through ranking metrics (AP, ROC-AUC), not by score values.
          </p>
          {ev.test_uncertainty &&
            Object.entries(ev.test_uncertainty.differences)
              .filter(([k]) => k.includes(ev.selection.selected))
              .map(([k, d]) => (
                <p key={k}>
                  Paired bootstrap, test AP difference {k}: {num(d.mean)} (95% CI {num(d.ci95[0])} to {num(d.ci95[1])})
                  {d.ci95[0] <= 0 && d.ci95[1] >= 0 ? " — the interval includes zero, so the ranking is not settled." : "."}
                </p>
              ))}
          <p>Latency is in-process scoring only (features + preprocessing + model), measured during training.</p>
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Precision–recall curves (test)">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={prData} margin={{ top: 8, right: 16, bottom: 20, left: 0 }}>
                <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
                <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={{ fill: "var(--muted)", fontSize: 12 }} label={{ value: "Recall", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 12 }} />
                <YAxis domain={[0, 1]} tick={{ fill: "var(--muted)", fontSize: 12 }} label={{ value: "Precision", angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 12 }} />
                <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", color: "var(--ink)" }} formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)} />
                <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12 }} />
                {ev.results
                  .filter((r) => r.model !== "dummy")
                  .map((r) => (
                    <Line key={r.model} type="stepAfter" dataKey={r.model} stroke={SERIES_COLORS[r.model] ?? "var(--ink)"} dot={false} strokeWidth={2} connectNulls isAnimationActive={false} />
                  ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <FocusedModel r={focused} />
      </div>
    </div>
  );
}

/** Recharts needs one row per x value; merge each model's (recall, precision) points. */
function mergeCurves(results: EvaluationResult[]) {
  const rows: Record<string, number>[] = [];
  for (const r of results) for (const p of r.test_pr_curve) rows.push({ recall: p.recall, [r.model]: p.precision });
  return rows.sort((a, b) => (a.recall ?? 0) - (b.recall ?? 0));
}

function FocusedModel({ r }: { r: EvaluationResult }) {
  const max = Math.max(...r.global_importance.map((g) => g.mean_abs_contribution), 1e-9);
  return (
    <Panel title={`${r.model}: what drives its scores`}>
      {r.global_importance.length === 0 ? (
        <p className="text-muted">No global explanation for this model (the baseline ignores its inputs).</p>
      ) : (
        <ul className="space-y-1.5">
          {r.global_importance.slice(0, 10).map((g) => (
            <li key={g.feature} className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-2 text-sm">
              <span className="truncate">{g.feature}</span>
              <span className="h-3 rounded-sm bg-panel-2">
                <span className="block h-3 rounded-sm bg-accent" style={{ width: `${(g.mean_abs_contribution / max) * 100}%` }} />
              </span>
              <span className="num text-right text-xs">{g.mean_abs_contribution.toFixed(3)}</span>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-xs text-muted">Mean absolute SHAP contribution on a validation sample. This ranks influence on the model, not causes of fraud.</p>
      {r.test_calibration && (
        <div className="mt-4">
          <h3 className="text-sm font-semibold">Calibration on test</h3>
          <p className="num text-xs text-muted">
            Brier {r.test_calibration.brier.toFixed(5)} · ECE {r.test_calibration.ece.toFixed(5)}
          </p>
          <table className="mt-2 w-full text-xs">
            <thead>
              <tr className="text-left text-muted">
                <th className="font-medium">Bin</th>
                <th className="text-right font-medium">Rows</th>
                <th className="text-right font-medium">Mean predicted</th>
                <th className="text-right font-medium">Observed rate</th>
              </tr>
            </thead>
            <tbody>
              {r.test_calibration.bins.slice(-5).map((b) => (
                <tr key={b.bin}>
                  <td className="num">{b.bin}</td>
                  <td className="num text-right">{b.count.toLocaleString()}</td>
                  <td className="num text-right">{pct(b.mean_predicted, 3)}</td>
                  <td className="num text-right">{pct(b.observed_rate, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-xs text-muted">Highest-score bins shown (equal-count bins). Close values mean the displayed probabilities can be read at face value.</p>
        </div>
      )}
    </Panel>
  );
}

function Registry() {
  const me = useMe();
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: () => api<ModelSummary[]>("/models") });
  const activate = useMutation({
    mutationFn: (v: string) => api(`/models/${v}/activate`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries(),
  });
  return (
    <Panel title="Model registry" className="mt-4">
      {models.isPending ? (
        <Loading />
      ) : models.isError ? (
        <ErrorState error={models.error} />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[44rem] text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
                <th className="py-2 pr-3 font-medium">Version</th>
                <th className="py-2 pr-3 font-medium">Dataset</th>
                <th className="py-2 pr-3 text-right font-medium">Threshold</th>
                <th className="py-2 pr-3 font-medium">MLflow run</th>
                <th className="py-2 font-medium" />
              </tr>
            </thead>
            <tbody>
              {models.data.map((m) => (
                <tr key={m.model_version} className="border-b border-line last:border-0">
                  <td className="num py-2 pr-3">{shortVersion(m.model_version)}</td>
                  <td className="py-2 pr-3">
                    <SyntheticBadge synthetic={m.contract === "synthetic"} /> <span className="ml-1">{m.contract}</span>
                  </td>
                  <td className="num py-2 pr-3 text-right">{m.threshold.toFixed(4)}</td>
                  <td className="num py-2 pr-3 text-xs text-muted">{m.mlflow_run_id?.slice(0, 12) ?? "-"}</td>
                  <td className="py-2 text-right">
                    {m.is_active ? (
                      <Pill tone="ok">serving</Pill>
                    ) : me.data?.role === "admin" ? (
                      <Button variant="secondary" onClick={() => activate.mutate(m.model_version)} disabled={activate.isPending}>
                        Activate
                      </Button>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {activate.isError && <p className="mt-2 text-sm text-danger">{errorText(activate.error)}</p>}
        </div>
      )}
    </Panel>
  );
}
