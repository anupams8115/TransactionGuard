"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { use, useState } from "react";

import { useMe } from "@/components/AppShell";
import { ExplanationView } from "@/components/ExplanationView";
import { Button, ErrorState, FlagPill, inputClass, KeyValues, Label, Loading, PageHeader, Panel, ReviewPill, SyntheticBadge } from "@/components/ui";
import { api, errorText } from "@/lib/api";
import { formatDate, formatScore, scoreLabel } from "@/lib/format";
import type { Assessment, PredictionDetail, ReviewStatus } from "@/lib/types";

interface ReviewEvent {
  actor_id: string;
  from_status: string | null;
  to_status: string;
  assessment: string | null;
  note: string | null;
  created_at: string;
}

export default function TransactionDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const qc = useQueryClient();
  const me = useMe();
  const detail = useQuery({
    queryKey: ["prediction", id],
    queryFn: () => api<PredictionDetail>(`/predictions/${id}`),
    refetchInterval: (q) => (q.state.data?.explanation_status === "pending" ? 2000 : false),
  });
  const history = useQuery({ queryKey: ["review-history", id], queryFn: () => api<ReviewEvent[]>(`/predictions/${id}/review/history`) });

  const explain = useMutation({
    mutationFn: () => api(`/predictions/${id}/explanation`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["prediction", id] }),
  });

  if (detail.isPending) return <Loading />;
  if (detail.isError) return <ErrorState error={detail.error} retry={() => detail.refetch()} />;
  const p = detail.data;
  const canWrite = me.data?.role !== "viewer";

  return (
    <>
      <PageHeader
        title={p.transaction_id}
        subtitle={
          <span className="flex flex-wrap items-center gap-2">
            <SyntheticBadge synthetic={p.synthetic} />
            <FlagPill flagged={p.flagged} />
            <ReviewPill status={p.review?.status} />
            <span>Scored {formatDate(p.created_at)}</span>
          </span>
        }
      />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <Panel title="Score">
            <div className="flex flex-wrap items-end gap-8">
              <div>
                <div className="text-xs uppercase tracking-wide text-muted">{scoreLabel(p.score_kind)}</div>
                <div className={`num text-4xl ${p.flagged ? "text-risk" : ""}`}>{formatScore(p.score, p.score_kind)}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-muted">Threshold applied</div>
                <div className="num text-xl">{formatScore(p.threshold, p.score_kind)}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-muted">Confirmed label</div>
                <div className="text-sm">{p.confirmed_label === null ? "None recorded" : p.confirmed_label ? "Fraud" : "Genuine"}
                  {p.confirmed_label_source && (
                    <span className="block text-xs text-muted">
                      source: {p.confirmed_label_source === "dataset_label" ? "replayed dataset label (demo)" : p.confirmed_label_source.replace("_", " ")}
                    </span>
                  )}</div>
              </div>
            </div>
            <p className="mt-3 text-xs text-muted">{p.score_meaning}</p>
            {p.warnings.length > 0 && <p className="mt-2 text-xs text-risk">{p.warnings.join("; ")}</p>}
          </Panel>

          <Panel
            title="Why this score"
            action={
              canWrite && p.explanation_status !== "ok" && p.explanation_status !== "pending" ? (
                <Button variant="secondary" onClick={() => explain.mutate()} disabled={explain.isPending}>
                  {explain.isPending ? "Explaining…" : "Compute explanation"}
                </Button>
              ) : undefined
            }
          >
            <ExplanationView explanation={p.explanation} status={p.explanation_status} />
            {explain.isError && <p className="mt-2 text-sm text-danger">{errorText(explain.error)}</p>}
          </Panel>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Submitted transaction">
              <KeyValues data={p.transaction} />
            </Panel>
            <Panel title="Model inputs (at scoring time)">
              <KeyValues data={p.features} />
            </Panel>
          </div>
        </div>

        <div className="space-y-4">
          <ReviewPanel id={id} detail={p} canWrite={canWrite} onSaved={() => {
            qc.invalidateQueries({ queryKey: ["prediction", id] });
            qc.invalidateQueries({ queryKey: ["review-history", id] });
          }} />
          <Panel title="Review history">
            {history.data && history.data.length > 0 ? (
              <ol className="space-y-2 text-sm">
                {history.data.map((h) => (
                  <li key={h.created_at}>
                    <span className="num text-xs text-muted">{formatDate(h.created_at)}</span>{" "}
                    {h.from_status ?? "new"} → <strong>{h.to_status.replace("_", " ")}</strong>
                    {h.assessment && <> · {h.assessment.replace("_", " ")}</>}
                    {h.note && <p className="text-muted">“{h.note}”</p>}
                  </li>
                ))}
              </ol>
            ) : (
              <p className="text-sm text-muted">No review activity yet.</p>
            )}
          </Panel>
          {me.data?.role === "admin" && <LabelPanel id={id} current={p.confirmed_label} onSaved={() => qc.invalidateQueries({ queryKey: ["prediction", id] })} />}
          <Panel title="Trace">
            <p className="mb-2 text-xs text-muted">Everything needed to reproduce this score: model, preprocessing, threshold and training run.</p>
            <KeyValues data={p.trace} />
          </Panel>
        </div>
      </div>
    </>
  );
}

function ReviewPanel({ id, detail, canWrite, onSaved }: { id: string; detail: PredictionDetail; canWrite: boolean; onSaved: () => void }) {
  const [status, setStatus] = useState<ReviewStatus>(detail.review?.status ?? "unreviewed");
  const [assessment, setAssessment] = useState<Assessment | "">(detail.review?.assessment ?? "");
  const [note, setNote] = useState(detail.review?.note ?? "");
  const save = useMutation({
    mutationFn: () =>
      api(`/predictions/${id}/review`, {
        method: "PUT",
        body: JSON.stringify({ status, assessment: assessment || null, note: note || null }),
      }),
    onSuccess: onSaved,
  });
  return (
    <Panel title="Analyst review">
      <p className="mb-3 text-xs text-muted">
        Your assessment is an opinion recorded for the workflow. It is stored separately from confirmed outcomes and never becomes a training label.
      </p>
      <div className="space-y-3">
        <div>
          <Label htmlFor="review-status">Status</Label>
          <select id="review-status" className={inputClass} value={status} onChange={(e) => setStatus(e.target.value as ReviewStatus)} disabled={!canWrite}>
            <option value="unreviewed">Unreviewed</option>
            <option value="under_review">Under review</option>
            <option value="resolved">Resolved</option>
          </select>
        </div>
        <div>
          <Label htmlFor="review-assessment" hint="required to resolve">
            Assessment
          </Label>
          <select id="review-assessment" className={inputClass} value={assessment} onChange={(e) => setAssessment(e.target.value as Assessment | "")} disabled={!canWrite}>
            <option value="">Not yet assessed</option>
            <option value="suspicious">Suspicious</option>
            <option value="not_suspicious">Not suspicious</option>
            <option value="inconclusive">Inconclusive</option>
          </select>
        </div>
        <div>
          <Label htmlFor="review-note">Note</Label>
          <textarea id="review-note" rows={3} maxLength={2000} className={inputClass} value={note} onChange={(e) => setNote(e.target.value)} disabled={!canWrite} />
        </div>
        {canWrite ? (
          <div className="flex items-center gap-3">
            <Button onClick={() => save.mutate()} disabled={save.isPending}>
              {save.isPending ? "Saving…" : "Save review"}
            </Button>
            {save.isSuccess && <span className="text-sm text-ok">Saved</span>}
            {save.isError && <span className="text-sm text-danger">{errorText(save.error)}</span>}
          </div>
        ) : (
          <p className="text-sm text-muted">Viewers can read reviews but not change them.</p>
        )}
      </div>
    </Panel>
  );
}

function LabelPanel({ id, current, onSaved }: { id: string; current: boolean | null; onSaved: () => void }) {
  const [isFraud, setIsFraud] = useState(current === null ? "" : String(current));
  const [source, setSource] = useState("chargeback");
  const save = useMutation({
    mutationFn: () => api(`/predictions/${id}/label`, { method: "PUT", body: JSON.stringify({ is_fraud: isFraud === "true", source }) }),
    onSuccess: onSaved,
  });
  return (
    <Panel title="Confirmed outcome (admin)">
      <p className="mb-3 text-xs text-muted">Record only outcomes confirmed independently of the model, such as a chargeback. Used for monitoring, not retraining.</p>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <Label htmlFor="label-value">Outcome</Label>
          <select id="label-value" className={inputClass} value={isFraud} onChange={(e) => setIsFraud(e.target.value)}>
            <option value="">Choose…</option>
            <option value="true">Fraud</option>
            <option value="false">Genuine</option>
          </select>
        </div>
        <div>
          <Label htmlFor="label-source">Source</Label>
          <select id="label-source" className={inputClass} value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="chargeback">Chargeback</option>
            <option value="customer_confirmed">Customer confirmed</option>
            <option value="investigation_closed">Investigation closed</option>
          </select>
        </div>
      </div>
      <Button className="mt-3" variant="secondary" onClick={() => save.mutate()} disabled={!isFraud || save.isPending}>
        Record outcome
      </Button>
      {save.isError && <p className="mt-2 text-sm text-danger">{errorText(save.error)}</p>}
    </Panel>
  );
}
