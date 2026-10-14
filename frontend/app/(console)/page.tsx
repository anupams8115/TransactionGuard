"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";

import { ErrorState, FlagPill, Loading, PageHeader, Panel, Stat, SyntheticBadge, EmptyState } from "@/components/ui";
import { api } from "@/lib/api";
import { formatDate, formatScore, shortVersion } from "@/lib/format";
import type { Page, Prediction } from "@/lib/types";

interface Overview {
  predictions_total: number;
  predictions_last_7d: number;
  flagged_total: number;
  review_backlog: Record<string, number>;
  batch_jobs: Record<string, number>;
  synthetic_transactions: number;
  active_models: Record<string, string>;
}

export default function OverviewPage() {
  const overview = useQuery({ queryKey: ["overview"], queryFn: () => api<Overview>("/overview") });
  const queue = useQuery({ queryKey: ["queue", "top"], queryFn: () => api<Page<Prediction>>("/reviews/queue?page_size=6") });

  if (overview.isPending) return <Loading />;
  if (overview.isError) return <ErrorState error={overview.error} retry={() => overview.refetch()} />;
  const o = overview.data;
  const backlog = (o.review_backlog.unreviewed ?? 0) + (o.review_backlog.under_review ?? 0);

  return (
    <>
      <PageHeader title="Overview" subtitle="What the models have flagged and what still needs an analyst." />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Scored, last 7 days" value={o.predictions_last_7d.toLocaleString()} hint={`${o.predictions_total.toLocaleString()} in total`} />
        <Stat label="Flagged for review" value={o.flagged_total.toLocaleString()} tone="risk" hint="score ≥ model threshold" />
        <Stat label="Open reviews" value={backlog.toLocaleString()} hint={`${o.review_backlog.under_review ?? 0} under review`} />
        <Stat label="Failed batch jobs" value={(o.batch_jobs.failed ?? 0).toLocaleString()} tone={o.batch_jobs.failed ? "danger" : undefined} hint={`${o.batch_jobs.succeeded ?? 0} succeeded`} />
      </div>

      {o.synthetic_transactions > 0 && (
        <p className="mt-4 rounded-md border border-synthetic/40 bg-synthetic-soft px-3 py-2 text-sm text-synthetic">
          This workspace contains {o.synthetic_transactions.toLocaleString()} synthetic transactions. Results on them demonstrate
          the workflow only and are not evidence of real-world performance.
        </p>
      )}

      <div className="mt-6 grid gap-4 lg:grid-cols-[2fr_1fr]">
        <Panel title="Highest-risk open reviews" action={<Link href="/reviews" className="text-sm text-accent hover:underline">Open queue</Link>}>
          {queue.isPending ? (
            <Loading />
          ) : queue.isError ? (
            <ErrorState error={queue.error} />
          ) : queue.data.items.length === 0 ? (
            <EmptyState title="Nothing waiting for review">Flagged transactions appear here.</EmptyState>
          ) : (
            <ul className="divide-y divide-line">
              {queue.data.items.map((p) => (
                <li key={p.id} className="flex flex-wrap items-center justify-between gap-2 py-2">
                  <div className="min-w-0">
                    <Link href={`/transactions/${p.id}`} className="num font-medium text-accent hover:underline">
                      {p.transaction_id}
                    </Link>
                    <div className="text-xs text-muted">
                      {p.account_id ?? "no account id"} · scored {formatDate(p.created_at)}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <SyntheticBadge synthetic={p.synthetic} />
                    <span className="num">{formatScore(p.score, p.score_kind)}</span>
                    <FlagPill flagged={p.flagged} />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="Serving models">
          {Object.keys(o.active_models).length === 0 ? (
            <EmptyState title="No active model">Train one with <code className="num">tg train synthetic</code>.</EmptyState>
          ) : (
            <ul className="space-y-3">
              {Object.entries(o.active_models).map(([contract, version]) => (
                <li key={contract}>
                  <div className="flex items-center gap-2">
                    <span className="font-medium capitalize">{contract}</span>
                    <SyntheticBadge synthetic={contract === "synthetic"} />
                  </div>
                  <div className="num text-xs text-muted">{shortVersion(version)}</div>
                </li>
              ))}
            </ul>
          )}
          <Link href="/models" className="mt-4 inline-block text-sm text-accent hover:underline">
            Compare models
          </Link>
        </Panel>
      </div>
    </>
  );
}
