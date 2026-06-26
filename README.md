# quantzero

A low-latency trading feature platform. A minute bar flows in for a ticker; we emit a
full feature vector in **a few microseconds** by maintaining a feature-specific
incremental cache for every feature. Real-time (Alpaca websocket) and backfill
(historical replay) feed the **same** event core, so they cannot structurally diverge.

## The one idea

Each feature owns a cache shaped to its needs, updated incrementally on each event
(`on_minute` / `on_trade` / `on_quote`). Computing the feature's value is then a handful
of lookups — never a from-scratch recompute. The reusable cache primitives
(rolling sum/moments, EWMA, rolling min/max, Welford, session accumulators) live in
`quantzero/caches/`.

```
              ┌──────────────────────┐
 LiveSource ──┤   identical engine   ├─► FeatureEngine (per ticker)
 (Alpaca ws)  │   core: events in,   │     ├─ TickerState: day buffers (minutes + last quote/trade)
 ReplaySource─┤   vector out at each  │     └─ Feature[] each with its own cache
 (past day)   │   minute boundary    │          on_trade/on_quote/on_minute → O(1) cache update
              └──────────────────────┘          values() → a few ops → 53-wide vector → store
```

Because replay feeds the same hooks in time order, a feature computes the **same** value
live or backfilled — verified by a no-lookahead point-in-time test.

## Measured speed (simulation + benchmark)

- Full 53-feature vector: **~34µs mean** per minute bar (p99 ~46µs).
- Per-feature steady-state cost: **sub-2µs each**, ~17µs summed across all 20 groups.

(Run `make bench` and `make ARGS=... live` to reproduce.)

## Quickstart

```bash
cp .env.example .env          # fill in ALPACA_KEY_ID / ALPACA_SECRET_KEY (or it's pre-seeded)
make test                     # 21 tests
make bench                    # per-feature latency table
python -m quantzero.run_sim --tickers AAA,BBB --minutes 120 --store   # offline end-to-end

# At the market open:
make live ARGS="--tickers AAPL,MSFT,NVDA --warmup"
```

`run_live` connects to the Alpaca websocket, computes a feature vector on every minute
bar, and prints compute time + bar→vector latency. `--warmup` replays today's earlier
bars through the engine first (via REST) so caches are warm on the first live bar.

## Layout

| Path | What |
|------|------|
| `quantzero/events.py` | `Quote` / `Trade` / `MinuteBar` immutable events |
| `quantzero/caches/` | reusable O(1) incremental cache primitives |
| `quantzero/state.py` | `TickerState`: per-ticker day buffers |
| `quantzero/feature.py` | `Feature` base + registry |
| `quantzero/features/` | 20 feature groups (53 columns) |
| `quantzero/engine.py` | `FeatureEngine`: events → vector |
| `quantzero/driver.py` | routes events to per-ticker engines |
| `quantzero/sources/` | `simulation`, `alpaca` (replay), live (in `run_live`) |
| `quantzero/store.py` | versioned, source-transparent parquet feature store |
| `quantzero/universe.py` | daily tradable universe (see `docs/UNIVERSE.md`) |
| `quantzero/bench.py` | standalone per-feature latency benchmark |
| `quantzero/metrics.py` | Prometheus metrics (Grafana in `grafana/`) |

See `MILESTONE.md` for status and `CLAUDE.md` for code standards.
