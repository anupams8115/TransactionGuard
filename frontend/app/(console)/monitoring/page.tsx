"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";

import { useMe } from "@/components/AppShell";
import { Button, EmptyState, ErrorState, inputClass, JobPill, Label, Loading, PageHeader, Panel, Pill, Stat, SyntheticBadge } from "@/components/ui";
import { api, errorText } from "@/lib/api";
import { formatDate, num, pct, shortVersion } from "@/lib/format";
import type { ContractName, MonitoringReport, MonitoringSummary } from "@/lib/types";

interface Operational {
  since: string;
  requests: number;
  server_error_rate: number;
  routes: { route: string; method: string; requests: number; p50_ms: number; p95_ms: number; client_error_rate: number; server_error_rate: number }[];
  batch_jobs: Record<string, number>;
  note: string;
}

export default function MonitoringPage() {
  const qc = useQueryClient();
  const me = useMe();
  const [selected, setSelected] = useState<string | null>(null);
  const reports = useQuery({
    queryKey: ["monitoring-reports"],
    queryFn: () => api<MonitoringReport[]>("/monitoring/reports"),
    refetchInterval: (q) => (q.state.data?.some((r) => r.status === "queued" || r.status === "running") ? 2000 : false),
  });
  const create = useMutation({
    mutationFn: (body: { contract: ContractName; window_days: number }) =>
      api<MonitoringReport>("/monitoring/reports", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: (r) => {
      setSelected(r.id);
      qc.invalidateQueries({ queryKey: ["monitoring-reports"] });
    },
  });
  const current = reports.data?.find((r) => r.id === selected) ?? reports.data?.find((r) => r.status === "succeeded");

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    create.mutate({ contract: f.get("contract") as ContractName, window_days: Number(f.get("window_days")) });
  }

  return (
    <>
      <PageHeader
        title="Monitoring"
        subtitle="Compares recently scored transactions with the model's validation period. Drift says the data changed; only confirmed labels can say whether accuracy changed."
      />
      <div className="grid gap-4 lg:grid-cols-[20rem_1fr]">
        <div className="space-y-4">
          <Panel title="New report">
            <form onSubmit={onSubmit} className="space-y-3">
              <div>
                <Label htmlFor="mon-contract">Dataset</Label>
                <select id="mon-contract" name="contract" className={inputClass} defaultValue="synthetic">
                  <option value="synthetic">synthetic (demo)</option>
                  <option value="creditcard">creditcard (benchmark)</option>
                </select>
              </div>
              <div>
                <Label htmlFor="mon-window" hint="days of scored traffic">
                  Window
                </Label>
                <input id="mon-window" name="window_days" type="number" min={1} max={365} defaultValue={7} className={`${inputClass} num`} />
              </div>
              <Button type="submit" disabled={create.isPending || me.data?.role === "viewer"}>
                {create.isPending ? "Requesting…" : "Generate report"}
              </Button>
              {create.isError && <p className="text-sm text-danger">{errorText(create.error)}</p>}
            </form>
          </Panel>
          <Panel title="Reports">
            {reports.isPending ? (
              <Loading />
            ) : reports.isError ? (
              <ErrorState error={reports.error} />
            ) : reports.data.length === 0 ? (
              <EmptyState title="No reports yet">Generate one to compare recent traffic with the reference.</EmptyState>
            ) : (
              <ul className="space-y-1">
                {reports.data.map((r) => (
                  <li key={r.id}>
                    <button
                      type="button"
                      onClick={() => setSelected(r.id)}
                      className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-sm ${current?.id === r.id ? "bg-panel-2" : "hover:bg-panel-2"}`}
                    >
                      <span>
                        <span className="capitalize">{r.contract}</span>
                        <span className="block text-xs text-muted">{formatDate(r.created_at)}</span>
                      </span>
                      <JobPill status={r.status} />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
        <div className="space-y-4">
          {current ? <ReportView report={current} /> : <EmptyState title="Select or generate a report" />}
          {me.data?.role === "admin" && <OperationalPanel />}
        </div>
      </div>
    </>
  );
}

function statusPill(s: MonitoringSummary["status"]) {
  if (s === "drift_detected") return <Pill tone="risk">drift detected</Pill>;
  if (s === "no_drift_detected") return <Pill tone="ok">no drift detected</Pill>;
  return <Pill>insufficient data</Pill>;
}

function ReportView({ report }: { report: MonitoringReport }) {
  if (report.status === "failed") return <ErrorState error={new Error(report.error ?? "Report failed")} />;
  if (report.status !== "succeeded" || !report.summary) return <Loading label="Generating report" />;
  const s = report.summary;
  const drift = Object.entries(s.drift?.columns ?? {}).sort((a, b) => Number(b[1].drift_detected) - Number(a[1].drift_detected) || a[0].localeCompare(b[0]));
  return (
    <>
      <Panel
        title={
          <span className="flex flex-wrap items-center gap-2">
            {report.contract} · {shortVersion(report.model_version)} <SyntheticBadge synthetic={!!s.synthetic} /> {statusPill(s.status)}
          </span>
        }
        action={
          s.status !== "insufficient_data" ? (
            <a href={`/api/v1/monitoring/reports/${report.id}/html`} target="_blank" rel="noreferrer" className="text-sm text-accent hover:underline">
              Full Evidently report
            </a>
          ) : undefined
        }
      >
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Stat label="Reference rows" value={s.reference?.rows.toLocaleString() ?? "-"} hint={s.reference?.source ?? ""} />
          <Stat label="Window rows" value={s.window?.rows.toLocaleString() ?? "-"} hint={`${formatDate(s.window?.start)} → ${formatDate(s.window?.end)}`} />
          <Stat label="Alert rate" value={pct(s.scores?.current_alert_rate ?? null, 2)} hint={`reference ${pct(s.scores?.reference_alert_rate, 2)}`} tone="risk" />
          <Stat label="Drifted columns" value={s.drift?.drifted_columns ?? "-"} hint={s.drift?.drifted_share !== undefined ? `${pct(s.drift.drifted_share)} of monitored` : s.note} />
        </div>
        <p className="mt-3 text-xs text-muted">{s.interpretation}</p>
        {s.note && <p className="mt-1 text-sm">{s.note}</p>}
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Distribution drift by column">
          {drift.length === 0 ? (
            <p className="text-muted">Not computed (not enough rows in the window).</p>
          ) : (
            <div className="max-h-96 overflow-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-muted">
                    <th className="py-1 font-medium">Column</th>
                    <th className="py-1 font-medium">Test</th>
                    <th className="py-1 text-right font-medium">Statistic (threshold)</th>
                    <th className="py-1 text-right font-medium">Missing now</th>
                    <th className="py-1" />
                  </tr>
                </thead>
                <tbody>
                  {drift.map(([col, d]) => (
                    <tr key={col} className="border-t border-line">
                      <td className="py-1">{col}</td>
                      <td className="py-1 text-xs text-muted">{d.method}</td>
                      <td className="num py-1 text-right">
                        {d.statistic < 0.0001 ? "<0.0001" : d.statistic.toFixed(4)} <span className="text-xs text-muted">({d.statistic_kind === "p_value" ? "p <" : "≥"} {d.threshold})</span>
                      </td>
                      <td className="num py-1 text-right">{pct(s.missing_share_current?.[col] ?? null)}</td>
                      <td className="py-1 text-right">{d.drift_detected ? <Pill tone="risk">drift</Pill> : null}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="mt-2 text-xs text-muted">Evidently picks the test by sample size: p-values (drift when small) for small samples, distances (drift when large) for large ones. A flagged column means the distribution moved, not that predictions are wrong.</p>
        </Panel>
        <div className="space-y-4">
          <Panel title="Scores">
            {s.scores && (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-muted">
                    <th className="font-medium">Quantile</th>
                    <th className="text-right font-medium">Reference</th>
                    <th className="text-right font-medium">Window</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(s.scores.reference_quantiles).map((q) => (
                    <tr key={q} className="border-t border-line">
                      <td className="num py-1">p{Number(q) * 100}</td>
                      <td className="num py-1 text-right">{num(s.scores!.reference_quantiles[q], 5)}</td>
                      <td className="num py-1 text-right">{num(s.scores!.current_quantiles?.[q], 5)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {s.alerts_by_day && s.alerts_by_day.length > 0 && (
              <p className="mt-2 text-xs text-muted">
                Alerts by scoring day: {s.alerts_by_day.map((d) => `${d.day}: ${d.alerts}/${d.scored}`).join(" · ")}
              </p>
            )}
          </Panel>
          <Panel title="Schema violations">
            {s.schema_violations && (
              <p className="text-sm">
                {s.schema_violations.batch_rows_invalid.toLocaleString()} of {s.schema_violations.batch_rows_total.toLocaleString()} uploaded rows were
                invalid; {s.schema_violations.rejected_single_requests_422} single requests were rejected.
                {Object.keys(s.schema_violations.issues_by_code).length > 0 && (
                  <span className="block text-xs text-muted">
                    {Object.entries(s.schema_violations.issues_by_code)
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(" · ")}
                  </span>
                )}
              </p>
            )}
          </Panel>
          <Panel title="Model performance">
            {s.performance?.available && s.performance.metrics ? (
              <>
                <p className="num text-sm">
                  {s.performance.labelled_rows.toLocaleString()} labelled rows ({pct(s.performance.label_coverage)} coverage) · AP {num(s.performance.metrics.average_precision)} ·
                  precision {num(s.performance.metrics.precision)} · recall {num(s.performance.metrics.recall)}
                </p>
                <p className="mt-1 text-xs text-muted">{s.performance.caveat}</p>
              </>
            ) : (
              <p className="text-sm text-muted">{s.performance?.reason}</p>
            )}
          </Panel>
        </div>
      </div>
    </>
  );
}

function OperationalPanel() {
  const ops = useQuery({ queryKey: ["operational"], queryFn: () => api<Operational>("/monitoring/operational?hours=24") });
  return (
    <Panel title="Service health, last 24 hours" action={<Button variant="ghost" onClick={() => ops.refetch()}>Refresh</Button>}>
      {ops.isPending ? (
        <Loading />
      ) : ops.isError ? (
        <ErrorState error={ops.error} />
      ) : (
        <>
          <p className="num mb-2 text-sm">
            {ops.data.requests.toLocaleString()} requests · server error rate {pct(ops.data.server_error_rate, 2)} · batch jobs{" "}
            {Object.entries(ops.data.batch_jobs).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[36rem] text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-muted">
                  <th className="font-medium">Route</th>
                  <th className="text-right font-medium">Requests</th>
                  <th className="text-right font-medium">p50 ms</th>
                  <th className="text-right font-medium">p95 ms</th>
                  <th className="text-right font-medium">4xx</th>
                  <th className="text-right font-medium">5xx</th>
                </tr>
              </thead>
              <tbody>
                {ops.data.routes.map((r) => (
                  <tr key={`${r.method} ${r.route}`} className="border-t border-line">
                    <td className="num py-1 text-xs">
                      {r.method} {r.route}
                    </td>
                    <td className="num py-1 text-right">{r.requests}</td>
                    <td className="num py-1 text-right">{r.p50_ms.toFixed(1)}</td>
                    <td className="num py-1 text-right">{r.p95_ms.toFixed(1)}</td>
                    <td className="num py-1 text-right">{pct(r.client_error_rate)}</td>
                    <td className="num py-1 text-right">{pct(r.server_error_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">{ops.data.note}</p>
        </>
      )}
    </Panel>
  );
}
