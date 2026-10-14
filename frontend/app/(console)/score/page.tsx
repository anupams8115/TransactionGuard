"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { type FormEvent, useMemo, useState } from "react";
import { z } from "zod";

import { ExplanationView } from "@/components/ExplanationView";
import { Button, ErrorState, FlagPill, inputClass, Label, Loading, PageHeader, Panel, SyntheticBadge } from "@/components/ui";
import { api, ApiError, errorText } from "@/lib/api";
import { formatScore, scoreLabel, shortVersion } from "@/lib/format";
import type { Contract, ContractName, FieldSpec, Page, Prediction, PredictionDetail } from "@/lib/types";

type Values = Record<string, string>;

/** Build a zod schema from the data contract, so the browser checks exactly what the API checks. */
function schemaFor(fields: FieldSpec[]) {
  const shape: Record<string, z.ZodType> = {};
  for (const f of fields) {
    if (f.dtype === "float" || f.dtype === "int") {
      let n = z.coerce.number({ message: "Enter a number" }).refine(Number.isFinite, "Must be finite");
      if (f.min_value !== null) n = n.refine((v) => v >= f.min_value!, `Must be ≥ ${f.min_value}`);
      if (f.max_value !== null) n = n.refine((v) => v <= f.max_value!, `Must be ≤ ${f.max_value}`);
      if (f.dtype === "int") n = n.refine(Number.isInteger, "Must be a whole number");
      shape[f.name] = n;
    } else if (f.dtype === "category") {
      shape[f.name] = z.enum(f.categories as [string, ...string[]], { message: "Choose a value" });
    } else if (f.dtype === "datetime") {
      shape[f.name] = z.string().min(1, "Required").transform((s, ctx) => {
        const d = new Date(s.endsWith("Z") || s.includes("+") ? s : `${s}Z`); // form value is UTC
        if (Number.isNaN(d.getTime())) ctx.addIssue({ code: "custom", message: "Invalid date" });
        return d.toISOString();
      });
    } else {
      shape[f.name] = z.string().trim().min(1, "Required").max(64);
    }
  }
  return z.object(shape);
}

function toFormValue(f: FieldSpec, v: unknown): string {
  if (v === null || v === undefined) return "";
  if (f.dtype === "datetime") return String(v).replace(/(\+00:00|Z)$/, "").slice(0, 19);
  return String(v);
}

export default function ScorePage() {
  const [contractName, setContractName] = useState<ContractName>("synthetic");
  const contract = useQuery({ queryKey: ["contract", contractName], queryFn: () => api<Contract>(`/contracts/${contractName}`) });
  const [values, setValues] = useState<Values>({});
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [exampleNote, setExampleNote] = useState<string | null>(null);
  const fields = useMemo(() => contract.data?.input_fields ?? [], [contract.data]);

  const submit = useMutation({
    mutationFn: (transaction: Record<string, unknown>) =>
      api<Prediction>("/predictions", { method: "POST", body: JSON.stringify({ contract: contractName, transaction, explain: true }) }),
  });
  const detail = useQuery({
    queryKey: ["prediction", submit.data?.id],
    queryFn: () => api<PredictionDetail>(`/predictions/${submit.data!.id}`),
    enabled: !!submit.data,
  });

  async function fillExample() {
    // Start from a stored transaction so the example is realistic, and give it a fresh id. It keeps
    // the original timestamp, so its account-history features use the same earlier transactions.
    setExampleNote(null);
    const recent = await api<Page<Prediction>>(`/predictions?contract=${contractName}&page_size=50&sort=score_desc`);
    const pick = recent.items[Math.floor(Math.random() * recent.items.length)];
    if (!pick) {
      setExampleNote("No stored transactions to copy yet. Fill the form manually or upload a batch first.");
      return;
    }
    const d = await api<PredictionDetail>(`/predictions/${pick.id}`);
    const next: Values = {};
    for (const f of fields) next[f.name] = toFormValue(f, d.transaction[f.name]);
    next[fields.find((f) => f.role === "id")!.name] = `${pick.transaction_id}-copy-${Math.random().toString(36).slice(2, 7)}`;
    setValues(next);
    setFieldErrors({});
    setExampleNote(`Copied from ${pick.transaction_id} (score ${formatScore(pick.score, pick.score_kind)}) with a new transaction id.`);
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    const parsed = schemaFor(fields).safeParse(values);
    if (!parsed.success) {
      const errs: Record<string, string> = {};
      for (const issue of parsed.error.issues) errs[String(issue.path[0])] = issue.message;
      setFieldErrors(errs);
      return;
    }
    setFieldErrors({});
    submit.mutate(parsed.data);
  }

  const result = submit.data;
  return (
    <>
      <PageHeader
        title="Score a transaction"
        subtitle="The form shows exactly the fields the active model's pipeline consumes. Nothing is filled in silently."
        actions={
          <div role="tablist" aria-label="Dataset" className="flex rounded-md border border-line bg-panel p-0.5">
            {(["synthetic", "creditcard"] as const).map((c) => (
              <button
                key={c}
                role="tab"
                aria-selected={contractName === c}
                type="button"
                onClick={() => {
                  setContractName(c);
                  setValues({});
                  setFieldErrors({});
                  submit.reset();
                }}
                className={`rounded px-3 py-1 text-sm ${contractName === c ? "bg-accent text-accent-ink" : "text-ink"}`}
              >
                {c === "synthetic" ? "Synthetic demo" : "Credit-card benchmark"}
              </button>
            ))}
          </div>
        }
      />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <Panel
          title={
            <span className="flex items-center gap-2">
              Transaction <SyntheticBadge synthetic={contractName === "synthetic"} />
            </span>
          }
          action={
            <Button variant="secondary" onClick={() => fillExample().catch((e) => setExampleNote(errorText(e)))} disabled={!fields.length}>
              Fill with an example
            </Button>
          }
        >
          {contract.isPending ? (
            <Loading />
          ) : contract.isError ? (
            <ErrorState error={contract.error} />
          ) : (
            <form onSubmit={onSubmit} noValidate>
              {exampleNote && <p className="mb-3 text-xs text-muted">{exampleNote}</p>}
              <div className={`grid gap-3 ${fields.length > 12 ? "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4" : "sm:grid-cols-2"}`}>
                {fields.map((f) => (
                  <div key={f.name}>
                    <Label htmlFor={`f-${f.name}`} hint={f.dtype === "datetime" ? "UTC" : f.unit ?? undefined}>
                      {f.name}
                    </Label>
                    {f.dtype === "category" ? (
                      <select
                        id={`f-${f.name}`}
                        value={values[f.name] ?? ""}
                        onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                        className={inputClass}
                        aria-invalid={!!fieldErrors[f.name]}
                      >
                        <option value="">Choose…</option>
                        {f.categories!.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        id={`f-${f.name}`}
                        type={f.dtype === "datetime" ? "datetime-local" : f.dtype === "float" || f.dtype === "int" ? "number" : "text"}
                        step={f.dtype === "float" ? "any" : f.dtype === "datetime" ? 1 : undefined}
                        value={values[f.name] ?? ""}
                        onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                        className={`${inputClass} num`}
                        aria-invalid={!!fieldErrors[f.name]}
                        title={f.description}
                      />
                    )}
                    {fieldErrors[f.name] && <p className="mt-0.5 text-xs text-danger">{fieldErrors[f.name]}</p>}
                  </div>
                ))}
              </div>
              {contract.data.notes.length > 0 && (
                <ul className="mt-4 list-disc space-y-0.5 pl-5 text-xs text-muted">
                  {contract.data.notes.map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                </ul>
              )}
              <div className="mt-4 flex items-center gap-3">
                <Button type="submit" disabled={submit.isPending}>
                  {submit.isPending ? "Scoring…" : "Score transaction"}
                </Button>
                {submit.isError && (
                  <p role="alert" className="text-sm text-danger">
                    {submit.error instanceof ApiError && submit.error.status === 409
                      ? "This transaction id was already scored with different values. Use a new id."
                      : errorText(submit.error)}
                  </p>
                )}
              </div>
            </form>
          )}
        </Panel>

        <Panel title="Result">
          {!result ? (
            <p className="text-muted">Submit a transaction to see its risk score, the threshold it was compared with, and why.</p>
          ) : (
            <div className="space-y-4">
              {result.idempotent_replay && (
                <p className="rounded-md bg-panel-2 px-3 py-2 text-sm">This transaction was already scored. Showing the stored result; no duplicate was created.</p>
              )}
              <div className="flex flex-wrap items-end gap-6">
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted">{scoreLabel(result.score_kind)}</div>
                  <div className={`num text-4xl ${result.flagged ? "text-risk" : ""}`}>{formatScore(result.score, result.score_kind)}</div>
                </div>
                <div className="text-sm">
                  <div className="text-muted">Threshold</div>
                  <div className="num">{formatScore(result.threshold, result.score_kind)}</div>
                </div>
                <FlagPill flagged={result.flagged} />
              </div>
              <p className="text-xs text-muted">{result.score_meaning}</p>
              {result.warnings.length > 0 && (
                <ul className="rounded-md border border-risk/40 bg-risk-soft px-3 py-2 text-xs text-risk">
                  {result.warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              )}
              {detail.data ? <ExplanationView explanation={detail.data.explanation} status={detail.data.explanation_status} /> : <Loading label="Loading explanation" />}
              <p className="text-xs text-muted">
                Model <span className="num">{shortVersion(result.model_version)}</span> · {result.latency_ms.toFixed(1)} ms ·{" "}
                <Link href={`/transactions/${result.id}`} className="text-accent hover:underline">
                  Open full trace and review
                </Link>
              </p>
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}
