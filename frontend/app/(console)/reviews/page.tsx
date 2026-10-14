"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { useMe } from "@/components/AppShell";
import { Button, EmptyState, ErrorState, Loading, PageHeader, Pagination, Panel, ReviewPill, SyntheticBadge } from "@/components/ui";
import { api, query } from "@/lib/api";
import { formatDate, formatScore, shortVersion } from "@/lib/format";
import type { Page, Prediction } from "@/lib/types";

export default function ReviewsPage() {
  const qc = useQueryClient();
  const me = useMe();
  const [page, setPage] = useState(1);
  const [includeResolved, setIncludeResolved] = useState(false);
  const queue = useQuery({
    queryKey: ["queue", page, includeResolved],
    queryFn: () => api<Page<Prediction>>(`/reviews/queue${query({ page, page_size: 25, include_resolved: includeResolved })}`),
    placeholderData: (prev) => prev,
  });
  const start = useMutation({
    mutationFn: (id: string) => api(`/predictions/${id}/review`, { method: "PUT", body: JSON.stringify({ status: "under_review" }) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["queue"] }),
  });

  return (
    <>
      <PageHeader
        title="Review queue"
        subtitle="Transactions whose score reached the model's threshold, highest risk first. Each one needs an analyst decision."
        actions={
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={includeResolved} onChange={(e) => { setIncludeResolved(e.target.checked); setPage(1); }} />
            Include resolved
          </label>
        }
      />
      <Panel>
        {queue.isPending ? (
          <Loading />
        ) : queue.isError ? (
          <ErrorState error={queue.error} retry={() => queue.refetch()} />
        ) : queue.data.items.length === 0 ? (
          <EmptyState title="The queue is empty">Flagged transactions will appear here after scoring.</EmptyState>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[48rem] text-sm">
                <thead>
                  <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
                    <th className="py-2 pr-3 font-medium">Transaction</th>
                    <th className="py-2 pr-3 font-medium">Data</th>
                    <th className="py-2 pr-3 text-right font-medium">Score</th>
                    <th className="py-2 pr-3 font-medium">Status</th>
                    <th className="py-2 pr-3 font-medium">Model</th>
                    <th className="py-2 pr-3 font-medium">Scored</th>
                    <th className="py-2 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {queue.data.items.map((p) => (
                    <tr key={p.id} className="border-b border-line last:border-0">
                      <td className="py-2 pr-3">
                        <Link href={`/transactions/${p.id}`} className="num text-accent hover:underline">
                          {p.transaction_id}
                        </Link>
                        <div className="num text-xs text-muted">{p.account_id ?? ""}</div>
                      </td>
                      <td className="py-2 pr-3">
                        <SyntheticBadge synthetic={p.synthetic} />
                      </td>
                      <td className="num py-2 pr-3 text-right text-risk">{formatScore(p.score, p.score_kind)}</td>
                      <td className="py-2 pr-3">
                        <ReviewPill status={p.review?.status} />
                      </td>
                      <td className="num py-2 pr-3 text-xs text-muted">{shortVersion(p.model_version)}</td>
                      <td className="py-2 pr-3 text-xs text-muted">{formatDate(p.created_at)}</td>
                      <td className="py-2 text-right">
                        {p.review?.status === "unreviewed" && me.data?.role !== "viewer" ? (
                          <Button variant="secondary" onClick={() => start.mutate(p.id)} disabled={start.isPending}>
                            Start review
                          </Button>
                        ) : (
                          <Link href={`/transactions/${p.id}`} className="text-accent hover:underline">
                            Open
                          </Link>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Pagination page={page} pageSize={25} total={queue.data.total} onPage={setPage} />
          </>
        )}
      </Panel>
    </>
  );
}
