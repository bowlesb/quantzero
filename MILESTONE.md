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
| Per-feature latency harness (reusable abstraction) | ✅ done | `quantzero/latency.py` |
| Prometheus metrics + Grafana dashboard | ✅ done | `quantzero/metrics.py`, `grafana/` |
| Alpaca live streaming → vectors | ✅ done | `quantzero/run_live.py` |
| Backfill as replay (same core) | ✅ done | `quantzero/sources/alpaca.py` (`ReplaySource`) |
| Versioned, source-transparent feature store | ✅ done | `quantzero/store.py` |
| Universe construction (~7k → ranked), documented | ✅ done | `quantzero/universe.py`, `docs/UNIVERSE.md` |
| 32-process sharding (router + workers, parity-tested) | ✅ done | `quantzero/sharding.py`, `quantzero/run_sharded.py` |
| Dashboard: feature-latency + feature-store views | ✅ done | `quantzero/dashboard/` (`make dashboard`) |

## Measured
- Full 53-feature vector: ~34µs mean / ~46µs p99 per bar (simulation).
- Per-feature **own marginal cost** 0.5–2.3µs each (state-only baseline ~0.9µs), ~21µs total (`make latency`).
- **Real Alpaca data validated**: replayed AAPL 2026-06-24 (858 minute bars, SIP feed) end
  to end through the engine at ~29µs/vector. Credentials + feed entitlement confirmed.
  Bars-only replay leaves the 7 microstructure features (trade/quote) NaN — those populate
  from live ticks at the open.

## How it ships (PRs, none merged — review in order)
- **PR1 — engine core + features + simulation + tests.** The reviewable heart, no network.
- **PR2 — Alpaca live/replay + universe + feature store + latency harness + observability + docs.**
- **PR3 — 32-process sharding (router + workers) + sharded sim/live runner.**

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
- Sharding distributes events by pickling them onto per-worker queues; fine for minute bars
  and moderate tick rates. Very high tick volume across many tickers may want a leaner IPC
  (struct-packing / shared memory) later. crc32 hashing is uneven for a handful of tickers but
  balances at universe scale. Worker results (vector summaries) return to the router; full
  vectors are computed in-worker (wire each worker to the feature store next).
- No corporate-actions / split handling yet.
