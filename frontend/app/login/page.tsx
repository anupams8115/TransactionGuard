"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useRouter, useSearchParams } from "next/navigation";
import { type FormEvent, Suspense, useState } from "react";

import { Button, inputClass, Label } from "@/components/ui";
import { ApiError } from "@/lib/api";

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const qc = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const form = new FormData(e.currentTarget);
    const body = new URLSearchParams({ username: String(form.get("email")), password: String(form.get("password")) });
    try {
      const res = await fetch("/api/v1/auth/login", { method: "POST", body, credentials: "same-origin" });
      if (!res.ok) throw new ApiError(res.status, res.status === 401 ? "Incorrect email or password." : "Sign-in failed.");
      qc.clear();
      const next = params.get("next");
      router.replace(next && next.startsWith("/") && !next.startsWith("//") ? next : "/");
    } catch (err) {
      setError(err instanceof ApiError ? String(err.detail) : "The API is not reachable. Is the backend running?");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-4" aria-describedby={error ? "login-error" : undefined}>
      <div>
        <Label htmlFor="email">Email</Label>
        <input id="email" name="email" type="email" autoComplete="username" required className={inputClass} />
      </div>
      <div>
        <Label htmlFor="password">Password</Label>
        <input id="password" name="password" type="password" autoComplete="current-password" required className={inputClass} />
      </div>
      {error && (
        <p id="login-error" role="alert" className="text-sm text-danger">
          {error}
        </p>
      )}
      <Button type="submit" disabled={busy} className="w-full">
        {busy ? "Signing in…" : "Sign in"}
      </Button>
    </form>
  );
}

export default function LoginPage() {
  return (
    <main className="grid min-h-screen place-items-center px-4">
      <div className="w-full max-w-sm rounded-md border border-line bg-panel p-6">
        <h1 className="text-2xl font-semibold">TransactionGuard</h1>
        <p className="mb-5 mt-1 text-sm text-muted">Sign in to review transaction-risk signals.</p>
        <Suspense>
          <LoginForm />
        </Suspense>
        <p className="mt-5 text-xs text-muted">
          Local demo accounts are created by <code className="num">tg demo seed</code>.
        </p>
      </div>
    </main>
  );
}
