"""Run the feature pipeline across N worker processes.

Simulation (offline, no network):

    python -m quantzero.run_sharded sim --tickers AAA,BBB,CCC,DDD --workers 4 --minutes 120

Live (router process holds the one Alpaca stream, fans out to workers):

    python -m quantzero.run_sharded live --tickers AAPL,MSFT,NVDA,AMD --workers 4
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from alpaca.data.live import StockDataStream

from quantzero.clock import ns_to_et
from quantzero.config import alpaca_config
from quantzero.run_live import live_quote_to_event, live_trade_to_event
from quantzero.sharding import ShardedRunner, VectorSummary, assign_tickers
from quantzero.sources.alpaca import bar_to_event, data_feed
from quantzero.sources.simulation import SimulationConfig, SimulationSource

_NS_PER_MINUTE = 60_000_000_000


def _print_assignment(tickers: list[str], n_workers: int) -> None:
    assignment = assign_tickers(tickers, n_workers)
    for worker_id in range(n_workers):
        print(f"  worker {worker_id}: {', '.join(assignment[worker_id]) or '(none)'}")


def run_sim(tickers: list[str], n_workers: int, minutes: int, seed: int) -> None:
    print(f"sharded simulation: {len(tickers)} tickers across {n_workers} workers")
    _print_assignment(tickers, n_workers)
    config = SimulationConfig(tickers=tickers, n_minutes=minutes, seed=seed)
    runner = ShardedRunner(tickers, n_workers)
    started = time.perf_counter()
    summaries = runner.run_source(SimulationSource(config))
    elapsed = time.perf_counter() - started

    per_worker: dict[int, int] = {}
    compute = np.array([s.compute_ns for s in summaries], dtype=np.float64) / 1000.0
    for summary in summaries:
        per_worker[summary.worker_id] = per_worker.get(summary.worker_id, 0) + 1
    print(f"computed {len(summaries)} vectors in {elapsed:.2f}s")
    print(f"compute per vector: mean={compute.mean():.1f}us p99={np.percentile(compute, 99):.1f}us")
    print(f"per-worker vector counts: {dict(sorted(per_worker.items()))}")


class LiveSink:
    """Prints each worker's vector summary as it arrives at the router."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, summary: VectorSummary) -> None:
        recv_ns = time.time_ns()
        latency_s = max((recv_ns - (summary.ts_ns + _NS_PER_MINUTE)) / 1e9, 0.0)
        when = ns_to_et(summary.ts_ns).strftime("%H:%M")
        self.count += 1
        print(
            f"{when} ET w{summary.worker_id} {summary.ticker:<6} "
            f"compute={summary.compute_ns / 1e6:6.2f}ms latency={latency_s:5.2f}s "
            f"valid={summary.n_valid}/{summary.n_features}"
        )


def run_live(tickers: list[str], n_workers: int) -> None:
    config = alpaca_config()
    print(f"sharded live: {len(tickers)} tickers across {n_workers} workers ({config.data_feed})")
    _print_assignment(tickers, n_workers)

    runner = ShardedRunner(tickers, n_workers)
    runner.start()
    runner.drain_forever(LiveSink())

    stream = StockDataStream(config.key_id, config.secret_key, feed=data_feed(config))

    async def on_bar(bar: object) -> None:
        runner.dispatch(bar_to_event(bar.symbol, bar))  # type: ignore[attr-defined]

    async def on_trade(trade: object) -> None:
        runner.dispatch(live_trade_to_event(trade))

    async def on_quote(quote: object) -> None:
        runner.dispatch(live_quote_to_event(quote))

    stream.subscribe_bars(on_bar, *tickers)
    stream.subscribe_trades(on_trade, *tickers)
    stream.subscribe_quotes(on_quote, *tickers)
    stream.run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline across worker processes.")
    parser.add_argument("mode", choices=["sim", "live"])
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minutes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.mode == "sim":
        run_sim(tickers, args.workers, args.minutes, args.seed)
    else:
        run_live(tickers, args.workers)


if __name__ == "__main__":
    main()
