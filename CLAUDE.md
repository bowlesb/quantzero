# quantzero — Claude Code instructions

A low-latency trading feature platform. A minute bar flows in for a ticker; we emit a
full feature vector in a few milliseconds by maintaining feature-specific incremental
caches. Real-time (Alpaca websocket) and backfill (historical replay) feed the SAME
event core, so they cannot structurally diverge.

## The one idea that matters
Each feature owns a cache shaped to its needs, updated incrementally on each event
(`on_minute` / `on_trade` / `on_quote`). Computing the feature's value is then a few
lookups/ops — never a from-scratch recompute. The reusable cache primitives live in
`quantzero/caches/`. Compose them; do not recompute rolling stats from slices on the hot
path unless the window is tiny.

## Architecture (read before changing)
- `events.py` — `Quote` / `Trade` / `MinuteBar` immutable events.
- `state.py` — `TickerState`: per-ticker day buffers (numpy minute ring + latest quote/trade).
- `caches/` — reusable O(1) incremental caches (rolling sum/moments, EWMA, rolling min/max, Welford, session accumulators).
- `feature.py` — `Feature` base: `setup()`, `on_*` hooks, `values()`.
- `engine.py` — `FeatureEngine` (one per ticker): drives events, assembles the vector on each minute bar.
- `sources/` — `simulation`, `replay` (Alpaca historical), `live` (Alpaca websocket). Same core.
- `store.py` — versioned, source-transparent parquet feature store.
- `universe.py` — daily tradable-ticker universe construction (see docs/UNIVERSE.md).
- `bench.py` — standalone per-feature latency benchmark (no simulation needed).
- `metrics.py` — Prometheus latency metrics.

## Code standards (enforced by `make qa`)
- ALL imports at top of file. Absolute imports only (`from quantzero.x import Y`). No imports inside functions/try.
- No nested function definitions. Constants and lookup tables at module level.
- Full type annotations on every function (params + return). Avoid `Any`.
- Prefer `dict["key"]` over `.get()` — fail loud on missing required data.
- No bare `except Exception`. Catch specific types or let it raise.
- No silent `return np.nan` for MISSING input data (that hides a broken pipeline). Return
  `np.nan` only for mathematically-undefined results (e.g. insufficient warmup, div-by-zero).
- Feature code must use `bar.ts_ns` / event timestamps, NEVER wall-clock `datetime.now()`,
  so features compute identically for historical replay.
- Descriptive names. No decorative `# ====` dividers. Comments explain *why*.

## Before reporting done
Run `make qa` (autoflake, isort, ruff, black, mypy) and `make test`. Fix ALL errors,
regardless of origin. No "done" with failing checks or uncommitted changes.

## Running
- `make test` — full test suite.
- `make bench` — per-feature latency report.
- `make universe` — build today's ticker universe to the store.
- `make live ARGS="--tickers AAPL,MSFT,NVDA"` — stream live and print feature vectors + latency.

Secrets live in `.env` (gitignored): `ALPACA_KEY_ID`, `ALPACA_SECRET_KEY`, `ALPACA_DATA_FEED`.
