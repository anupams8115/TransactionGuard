// Thin fetch wrapper. All calls go to the same origin (/api/v1 is proxied to FastAPI by Next.js),
// so the HttpOnly session cookie is sent automatically. The X-Requested-With header is required by
// the API for cookie-authenticated writes (CSRF defence).

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : `Request failed (${status})`);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("X-Requested-With", "fetch");
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`/api/v1${path}`, { ...init, headers, credentials: "same-origin" });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  const body = text ? safeJson(text) : null;
  if (!res.ok) {
    const detail = body && typeof body === "object" && "detail" in body ? (body as { detail: unknown }).detail : body;
    throw new ApiError(res.status, detail);
  }
  return body as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export function query(params: Record<string, string | number | boolean | null | undefined>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

/** Human-readable error text for any thrown value, including FastAPI 422 validation lists. */
export function errorText(err: unknown): string {
  if (err instanceof ApiError) {
    if (Array.isArray(err.detail)) {
      return err.detail
        .map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(-1).join(".")}: ${d.msg ?? "invalid"}`)
        .join("; ");
    }
    if (err.status === 401) return "Your session has expired. Sign in again.";
    if (err.status === 403) return "Your role does not allow this action.";
    if (typeof err.detail === "string") return err.detail;
    return `Request failed (${err.status}).`;
  }
  return err instanceof Error ? err.message : "Something went wrong.";
}
