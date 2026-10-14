"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type ReactNode, useEffect } from "react";

import { api, ApiError } from "@/lib/api";
import type { User } from "@/lib/types";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/score", label: "Score transaction" },
  { href: "/batches", label: "Batch upload" },
  { href: "/transactions", label: "Transactions" },
  { href: "/reviews", label: "Review queue" },
  { href: "/models", label: "Models" },
  { href: "/monitoring", label: "Monitoring" },
];

export function useMe() {
  return useQuery({ queryKey: ["me"], queryFn: () => api<User>("/auth/me"), retry: false, staleTime: 60_000 });
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const qc = useQueryClient();
  const me = useMe();

  useEffect(() => {
    if (me.error instanceof ApiError && me.error.status === 401) router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [me.error, pathname, router]);

  async function logout() {
    await api("/auth/logout", { method: "POST" }).catch(() => undefined);
    qc.clear();
    router.replace("/login");
  }

  if (!me.data) {
    return <div className="p-8 text-muted">{me.isError && !(me.error instanceof ApiError && me.error.status === 401) ? "The API is not reachable. Is the backend running on port 8000?" : "Checking your session…"}</div>;
  }

  return (
    <div className="min-h-screen md:grid md:grid-cols-[13.5rem_1fr]">
      <aside className="border-b border-line bg-panel md:sticky md:top-0 md:h-screen md:border-b-0 md:border-r">
        <div className="px-4 pb-3 pt-4">
          <Link href="/" className="font-display text-lg font-semibold tracking-tight">
            TransactionGuard
          </Link>
          <p className="text-xs text-muted">Risk signals for review</p>
        </div>
        <nav aria-label="Main" className="flex gap-1 overflow-x-auto px-2 pb-2 md:flex-col md:overflow-visible">
          {NAV.map((n) => {
            const active = n.href === "/" ? pathname === "/" : pathname.startsWith(n.href);
            return (
              <Link
                key={n.href}
                href={n.href}
                aria-current={active ? "page" : undefined}
                className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm ${active ? "bg-panel-2 font-semibold text-accent" : "text-ink hover:bg-panel-2"}`}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>
      </aside>
      <div className="min-w-0">
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-panel px-4 py-2 sm:px-6">
          <p className="text-xs text-muted">Scores are risk signals for analyst review, not proof of fraud.</p>
          <div className="flex items-center gap-3 text-sm">
            <span className="text-muted">
              {me.data.email} · <span className="capitalize">{me.data.role}</span> · {me.data.workspace_name}
            </span>
            <button type="button" onClick={logout} className="text-accent hover:underline">
              Sign out
            </button>
          </div>
        </header>
        <main className="px-4 py-6 sm:px-6">{children}</main>
      </div>
    </div>
  );
}
