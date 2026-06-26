// Shapes returned by the quantzero dashboard API (quantzero/dashboard/app.py).

import type { ColumnKind } from "./theme";

export interface GridColumn {
  key: string;
  label: string;
  kind: ColumnKind; // "raw" (tape layer) | "group" (feature group)
  trusted: boolean;
  features: string[]; // groups only: the feature inventory, for the horizontal expand
}

// `coverage[i][j]` is a 0..255 byte = (tickers with column j on date i) / universe_size. Columns are raw
// layers first (the substrate), then feature groups.
export interface StoreGridMatrix {
  generated_at: string;
  store_root: string;
  universe_size: number;
  dates: string[];
  columns: GridColumn[];
  coverage: number[][];
}

// Shapes returned by /api/latency — one per-group latency profile (slowest-first) plus the bar->vector context.
export interface LatencyGroup {
  group: string;
  feat_count: number;
  kind: string;
  mechanism: string;
  incremental_ready: string;
  p50_ns: number;
  p95_ns: number;
  p99_ns: number;
  own_ns: number; // marginal own cost above the state-only baseline
}

export interface LatencyE2EContext {
  vector_p50_ns: number; // full feature-vector compute p50
  total_own_ns: number; // sum of per-group own costs
  budget_ns: number; // per-group budget (1 ms)
}

export interface LatencyScenario {
  measure_minutes: number;
  quotes_per_minute: number;
  trades_per_minute: number;
}

export interface LatencyExpectations {
  units: string;
  sorted_by: string;
  generated_at: string;
  group_count: number;
  feature_count: number;
  budget_ns: number;
  all_under_budget: boolean;
  e2e_context: LatencyE2EContext;
  groups: LatencyGroup[];
  scenario: LatencyScenario;
}
