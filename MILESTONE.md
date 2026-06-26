# Milestone 1 — live feature vectors, fast, at the open

Goal: stream real Alpaca data and emit a good feature vector, in a few ms, on each new
minute bar for a ticker — with tests and a script runnable at market open.

## Status

| Piece | Status | Where |
|-------|--------|-------|
| Event core (events / state / engine) | ✅ done | `quantzero/{events,state,engine,driver}.py` |
| Feature-specific cache primitives | ✅ done | `quantzero/caches/` |
| 20 feature groups (53 columns) | ✅ done | `quantzero/features/` |
| Simulation source (fake quotes/trades/minutes) | ✅ done | `quantzero/sources/simulation.py` |
| Tests (caches, features, engine, no-lookahead, store, universe) | ✅ 21 passing | `tests/` |
| Per-feature latency benchmark (outside simulation) | ✅ done | `quantzero/bench.py` |
| Prometheus metrics + Grafana dashboard | ✅ done | `quantzero/metrics.py`, `grafana/` |
| Alpaca live streaming → vectors | ✅ done | `quantzero/run_live.py` |
| Backfill as replay (same core) | ✅ done | `quantzero/sources/alpaca.py` (`ReplaySource`) |
| Versioned, source-transparent feature store | ✅ done | `quantzero/store.py` |
| Universe construction (~7k → ranked), documented | ✅ done | `quantzero/universe.py`, `docs/UNIVERSE.md` |

## Measured
- Full 53-feature vector: ~34µs mean / ~46µs p99 per bar (simulation).
- Per-feature steady-state: sub-2µs each, ~17µs total (`make bench`).

## How it ships (PRs, none merged — review in order)
- **PR1 — engine core + features + simulation + tests.** The reviewable heart, no network.
- **PR2 — Alpaca live/replay + universe + feature store + benchmark + observability + docs.**

## To run at the open
```bash
make live ARGS="--tickers AAPL,MSFT,NVDA --warmup"
```
Prints, per minute bar: compute time, bar→vector latency, valid count, and sample features.
Metrics on `:9300` (set `QZ_METRICS_PORT`); Grafana dashboard in `grafana/`.

## Known gaps / next
- `--warmup` and `ReplaySource` are wired but only exercised against live Alpaca at the
  open — validate the first morning. (Unit-tested logic is offline; the network calls are not.)
- Feature store writes one small parquet per minute (fine for a day; batch later).
- Multi-process sharding (deterministic ticker hashing across 32 procs) is **not yet** built —
  the engine is process-ready (per-ticker, no shared mutable state) but a single process drives
  all tickers today. This is the next PR.
- No corporate-actions / split handling yet.
