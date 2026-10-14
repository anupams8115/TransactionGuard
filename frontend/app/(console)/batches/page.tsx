"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { type FormEvent, useRef, useState } from "react";

import { Button, EmptyState, ErrorState, inputClass, JobPill, Label, Loading, PageHeader, Panel, SyntheticBadge } from "@/components/ui";
import { api, errorText } from "@/lib/api";
import { formatDate, shortVersion } from "@/lib/format";
import type { BatchJob, ContractName } from "@/lib/types";

export default function BatchesPage() {
  const qc = useQueryClient();
  const [contract, setContract] = useState<ContractName>("synthetic");
  const [file, setFile] = useState<File | null>(null);
  // One idempotency key per chosen file: a double-click or network retry returns the same job.
  const keyRef = useRef<string>("");
  const [expanded, setExpanded] = useState<string | null>(null);

  const jobs = useQuery({
    queryKey: ["batches"],
    queryFn: () => api<BatchJob[]>("/batches"),
    refetchInterval: (q) => (q.state.data?.some((j) => j.status === "queued" || j.status === "running") ? 1500 : false),
  });

  const upload = useMutation({
    mutationFn: async () => {
      const body = new FormData();
      body.set("contract", contract);
      body.set("file", file!);
      return api<BatchJob>("/batches", { method: "POST", body, headers: { "Idempotency-Key": keyRef.current } });
    },
    onSuccess: (job) => {
      setExpanded(job.id);
      qc.invalidateQueries({ queryKey: ["batches"] });
    },
  });

  const retry = useMutation({
    mutationFn: (id: string) => api<BatchJob>(`/batches/${id}/retry`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["batches"] }),
  });

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (file) upload.mutate();
  }

  return (
    <>
      <PageHeader
        title="Batch upload"
        subtitle="Upload a CSV with the dataset's input columns. Rows are validated first; invalid rows are reported, not guessed. Scoring runs in the background."
      />
      <div className="grid gap-4 lg:grid-cols-[22rem_1fr]">
        <Panel title="New batch">
          <form onSubmit={onSubmit} className="space-y-3">
            <div>
              <Label htmlFor="batch-contract">Dataset contract</Label>
              <select id="batch-contract" className={inputClass} value={contract} onChange={(e) => setContract(e.target.value as ContractName)}>
                <option value="synthetic">synthetic (demo)</option>
                <option value="creditcard">creditcard (benchmark)</option>
              </select>
            </div>
            <div>
              <Label htmlFor="batch-file" hint="CSV, max 20 MB">
                File
              </Label>
              <input
                id="batch-file"
                type="file"
                accept=".csv,text/csv"
                className="block w-full text-sm file:mr-3 file:rounded-md file:border file:border-line file:bg-panel-2 file:px-3 file:py-1.5 file:text-sm"
                onChange={(e) => {
                  setFile(e.target.files?.[0] ?? null);
                  keyRef.current = crypto.randomUUID();
                  upload.reset();
                }}
              />
            </div>
            <Button type="submit" disabled={!file || upload.isPending}>
              {upload.isPending ? "Uploading…" : "Validate and queue"}
            </Button>
            {upload.isError && (
              <p role="alert" className="text-sm text-danger">
                {errorText(upload.error)}
              </p>
            )}
            {upload.data && (
              <p role="status" className="text-sm text-ok">
                {upload.data.idempotent_replay ? "This file was already submitted; showing the existing job." : "Queued. Progress updates below."}
              </p>
            )}
            <p className="text-xs text-muted">
              Labels in the file (for example <code className="num">is_fraud</code>) are ignored. Batch scoring never accepts training labels.
            </p>
          </form>
        </Panel>

        <Panel title="Your batch jobs">
          {jobs.isPending ? (
            <Loading />
          ) : jobs.isError ? (
            <ErrorState error={jobs.error} retry={() => jobs.refetch()} />
          ) : jobs.data.length === 0 ? (
            <EmptyState title="No batch jobs yet">Upload a CSV to score many transactions at once.</EmptyState>
          ) : (
            <ul className="divide-y divide-line">
              {jobs.data.map((j) => {
                const pct = j.valid_rows ? Math.round((j.processed_rows / j.valid_rows) * 100) : 0;
                const issues = j.validation_report?.issues ?? [];
                return (
                  <li key={j.id} className="py-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{j.file_name}</span>
                          <JobPill status={j.status} />
                          <SyntheticBadge synthetic={j.contract === "synthetic"} />
                        </div>
                        <div className="num text-xs text-muted">
                          {formatDate(j.created_at)} · {shortVersion(j.model_version)} · attempt {j.attempts}
                        </div>
                      </div>
                      <div className="flex items-center gap-3 text-sm">
                        <span className="num">
                          {j.processed_rows.toLocaleString()}/{j.valid_rows.toLocaleString()} scored · {j.flagged_rows} flagged
                        </span>
                        {j.status === "succeeded" && (
                          <Link href={`/transactions?batch_job_id=${j.id}`} className="text-accent hover:underline">
                            Results
                          </Link>
                        )}
                        {j.status === "failed" && (
                          <Button variant="secondary" onClick={() => retry.mutate(j.id)} disabled={retry.isPending}>
                            Retry
                          </Button>
                        )}
                        <button type="button" className="text-accent hover:underline" onClick={() => setExpanded(expanded === j.id ? null : j.id)} aria-expanded={expanded === j.id}>
                          Details
                        </button>
                      </div>
                    </div>
                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-panel-2" aria-hidden>
                      <div className={`h-full ${j.status === "failed" ? "bg-danger" : "bg-accent"}`} style={{ width: `${j.status === "succeeded" ? 100 : pct}%` }} />
                    </div>
                    {j.error && <p className="mt-2 text-sm text-danger">{j.error}</p>}
                    {expanded === j.id && (
                      <div className="mt-3 rounded-md bg-panel-2 p-3 text-sm">
                        <p className="num">
                          {j.total_rows} rows uploaded · {j.valid_rows} valid · {j.invalid_rows} invalid
                          {j.validation_report?.scoring && ` · ${j.validation_report.scoring.replayed_existing_predictions} already scored earlier (reused)`}
                        </p>
                        {issues.length > 0 ? (
                          <ul className="mt-2 space-y-1">
                            {issues.map((i) => (
                              <li key={`${i.code}-${i.column}`}>
                                <span className={i.severity === "error" ? "text-danger" : "text-risk"}>{i.code}</span> {i.column && <span className="num">[{i.column}]</span>} {i.message} ({i.count}){" "}
                                {i.example_ids.length > 0 && <span className="num text-xs text-muted">e.g. {i.example_ids.slice(0, 3).join(", ")}</span>}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="mt-1 text-muted">No validation issues.</p>
                        )}
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </Panel>
      </div>
    </>
  );
}
