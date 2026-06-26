import type { LatencyExpectations, StoreGridMatrix } from "./types";

// Thin client for the quantzero dashboard API. The SPA is served same-origin by the dashboard FastAPI app,
// so absolute /api/... paths hit it directly (the Vite dev server proxies them in `npm run dev`).

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" }, ...init });
  if (!res.ok) {
    throw new Error(`${url} -> ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export function fetchMatrix(): Promise<StoreGridMatrix> {
  return getJson<StoreGridMatrix>("/api/store/matrix");
}

export function fetchLatency(): Promise<LatencyExpectations> {
  return getJson<LatencyExpectations>("/api/latency");
}

export function refreshLatency(): Promise<LatencyExpectations> {
  return getJson<LatencyExpectations>("/api/latency/refresh", { method: "POST" });
}
