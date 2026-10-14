"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { type FormEvent, Suspense } from "react";

import { Button, EmptyState, ErrorState, FlagPill, inputClass, Label, Loading, PageHeader, Pagination, Panel, ReviewPill, SyntheticBadge } from "@/components/ui";
import { api, query } from "@/lib/api";
import { formatDate, formatScore, shortVersion } from "@/lib/format";
import type { ModelSummary, Page, Prediction } from "@/lib/types";

const FILTER_KEYS = ["q", "contract", "date_from", "date_to", "score_min", "score_max", "model_version", "review_status", "flagged", "batch_job_id"] as const;

function TransactionsInner() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const page = Number(params.get("page") ?? 1);
  const sort = params.get("sort") ?? "created_desc";
  const filters = Object.fromEntries(FILTER_KEYS.map((k) => [k, params.get(k) ?? ""]));

  const models = useQuery({ queryKey: ["models"], queryFn: () => api<ModelSummary[]>("/models") });
  const list = useQuery({
    queryKey: ["predictions", params.toString()],
    queryFn: () =>
      api<Page<Prediction>>(
        `/predictions${query({
          ...filters,
          // Date inputs are calendar days in UTC; make the end date inclusive.
          date_from: filters.date_from ? `${filters.date_from}T00:00:00Z` : "",
          date_to: filters.date_to ? `${filters.date_to}T23:59:59Z` : "",
          sort,
          page,
          page_size: 25,
        })}`,
      ),
    placeholderData: (prev) => prev,
  });

  function apply(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    const next = new URLSearchParams();
    for (const k of FILTER_KEYS) {
      const v = String(form.get(k) ?? "");
      if (v) next.set(k, v);
    }
    next.set("sort", String(form.get("sort") ?? sort));
    router.push(`${pathname}?${next.toString()}`);
  }

  function setPage(p: number) {
    const next = new URLSearchParams(params.toString());
    next.set("page", String(p));
    router.push(`${pathname}?${next.toString()}`);
  }

  return (
    <>
      <PageHeader title="Transactions" subtitle="Every scored transaction, with the model version and threshold that produced its score." />
      <Panel className="mb-4">
        <form key={params.toString()} onSubmit={apply} className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          <div className="lg:col-span-2">
            <Label htmlFor="q">Search</Label>
            <input id="q" name="q" defaultValue={filters.q} placeholder="Transaction or account id prefix" className={inputClass} />
          </div>
          <div>
            <Label htmlFor="contract">Dataset</Label>
            <select id="contract" name="contract" defaultValue={filters.contract} className={inputClass}>
              <option value="">All</option>
              <option value="synthetic">synthetic</option>
              <option value="creditcard">creditcard</option>
            </select>
          </div>
          <div>
            <Label htmlFor="date_from" hint="scored, UTC">From</Label>
            <input id="date_from" name="date_from" type="date" defaultValue={filters.date_from} className={inputClass} />
          </div>
          <div>
            <Label htmlFor="date_to" hint="UTC">To</Label>
            <input id="date_to" name="date_to" type="date" defaultValue={filters.date_to} className={inputClass} />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Label htmlFor="score_min">Score ≥</Label>
              <input id="score_min" name="score_min" type="number" step="any" min={0} defaultValue={filters.score_min} className={`${inputClass} num`} />
            </div>
            <div>
              <Label htmlFor="score_max">Score ≤</Label>
              <input id="score_max" name="score_max" type="number" step="any" defaultValue={filters.score_max} className={`${inputClass} num`} />
            </div>
          </div>
          <div className="lg:col-span-2">
            <Label htmlFor="model_version">Model version</Label>
            <select id="model_version" name="model_version" defaultValue={filters.model_version} className={inputClass}>
              <option value="">All</option>
              {models.data?.map((m) => (
                <option key={m.model_version} value={m.model_version}>
                  {m.contract} · {shortVersion(m.model_version)}
                  {m.is_active ? " (active)" : ""}
                </option>
              ))}
            </select>
          </div>
          <div>
            <Label htmlFor="review_status">Review status</Label>
            <select id="review_status" name="review_status" defaultValue={filters.review_status} className={inputClass}>
              <option value="">Any</option>
              <option value="unreviewed">Unreviewed</option>
              <option value="under_review">Under review</option>
              <option value="resolved">Resolved</option>
              <option value="none">Not flagged (no review)</option>
            </select>
          </div>
          <div>
            <Label htmlFor="flagged">Recommendation</Label>
            <select id="flagged" name="flagged" defaultValue={filters.flagged} className={inputClass}>
              <option value="">Any</option>
              <option value="true">Review</option>
              <option value="false">No action</option>
            </select>
          </div>
          <div>
            <Label htmlFor="sort">Sort</Label>
            <select id="sort" name="sort" defaultValue={sort} className={inputClass}>
              <option value="created_desc">Newest first</option>
              <option value="score_desc">Highest score</option>
              <option value="score_asc">Lowest score</option>
              <option value="created_asc">Oldest first</option>
            </select>
          </div>
          <input type="hidden" name="batch_job_id" defaultValue={filters.batch_job_id} />
          <div className="flex items-end gap-2">
            <Button type="submit">Apply</Button>
            <Button variant="ghost" onClick={() => router.push(pathname)}>
              Clear
            </Button>
          </div>
        </form>
        {filters.batch_job_id && <p className="mt-3 text-xs text-muted">Showing results of batch job <span className="num">{filters.batch_job_id}</span>.</p>}
        <p className="mt-2 text-xs text-muted">
          Score ranges compare raw scores: probabilities (0–1) for classifiers, unbounded anomaly scores for Isolation Forest.
        </p>
      </Panel>

      <Panel>
        {list.isPending ? (
          <Loading />
        ) : list.isError ? (
          <ErrorState error={list.error} retry={() => list.refetch()} />
        ) : list.data.items.length === 0 ? (
          <EmptyState title="No transactions match these filters">Try widening the date or score range.</EmptyState>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[56rem] text-sm">
                <thead>
                  <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
                    <th className="py-2 pr-3 font-medium">Transaction</th>
                    <th className="py-2 pr-3 font-medium">Account</th>
                    <th className="py-2 pr-3 font-medium">Data</th>
                    <th className="py-2 pr-3 text-right font-medium">Score</th>
                    <th className="py-2 pr-3 text-right font-medium">Threshold</th>
                    <th className="py-2 pr-3 font-medium">Recommendation</th>
                    <th className="py-2 pr-3 font-medium">Review</th>
                    <th className="py-2 pr-3 font-medium">Model</th>
                    <th className="py-2 font-medium">Scored</th>
                  </tr>
                </thead>
                <tbody className={list.isFetching ? "opacity-60" : ""}>
                  {list.data.items.map((p) => (
                    <tr key={p.id} className="border-b border-line last:border-0 hover:bg-panel-2">
                      <td className="py-2 pr-3">
                        <Link href={`/transactions/${p.id}`} className="num text-accent hover:underline">
                          {p.transaction_id}
                        </Link>
                      </td>
                      <td className="num py-2 pr-3 text-muted">{p.account_id ?? "-"}</td>
                      <td className="py-2 pr-3">
                        <SyntheticBadge synthetic={p.synthetic} />
                      </td>
                      <td className={`num py-2 pr-3 text-right ${p.flagged ? "text-risk" : ""}`}>{formatScore(p.score, p.score_kind)}</td>
                      <td className="num py-2 pr-3 text-right text-muted">{formatScore(p.threshold, p.score_kind)}</td>
                      <td className="py-2 pr-3">
                        <FlagPill flagged={p.flagged} />
                      </td>
                      <td className="py-2 pr-3">
                        <ReviewPill status={p.review?.status} />
                      </td>
                      <td className="num py-2 pr-3 text-xs text-muted">{shortVersion(p.model_version)}</td>
                      <td className="py-2 text-xs text-muted">{formatDate(p.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Pagination page={page} pageSize={25} total={list.data.total} onPage={setPage} />
          </>
        )}
      </Panel>
    </>
  );
}

export default function TransactionsPage() {
  return (
    <Suspense fallback={<Loading />}>
      <TransactionsInner />
    </Suspense>
  );
}
