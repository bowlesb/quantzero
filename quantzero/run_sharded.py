"""Run the feature pipeline across N worker processes — the production ingestion path.

A single ROUTER process holds the one Alpaca connection and fans each event to the worker
that owns its ticker (crc32 hash). Each of N workers runs its own engine over its subset
and writes vectors to the feature store via an async StoreWriter (off the compute path).

Backfill uses the SAME workers and the SAME engine: a ReplaySource feeds a past day's
events through the router exactly as the live stream would, writing source=backfill. The
only thing that changes between live and backfill is where the events come from.

    # Simulation (offline):
    python -m quantzero.run_sharded sim --tickers AAA,BBB --workers 4 --minutes 120

    # Live production (32 workers over the stored universe, async store writes):
    python -m quantzero.run_sharded live --universe --workers 32

    # Backfill a past day through the same 32 workers (source=backfill):
    python -m quantzero.run_sharded backfill --universe --workers 32 --dates 2026-06-24
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import signal
import time
from types import FrameType

import numpy as np
import polars as pl
from alpaca.data.live import StockDataStream

from quantzero.clock import ns_to_et
from quantzero.config import alpaca_config, store_root
from quantzero.run_live import live_quote_to_event, live_trade_to_event
from quantzero.sharding import ShardedRunner, VectorSummary, assign_tickers
from quantzero.sources.alpaca import ReplaySource, bar_to_event, data_feed
from quantzero.sources.simulation import SimulationConfig, SimulationSource
from quantzero.store import StoreConfig

_NS_PER_MINUTE = 60_000_000_000
SET_VERSION = "0.1.0"
SUBSCRIBE_CHUNK = 1000  # symbols per Alpaca subscribe call


def load_universe() -> list[str]:
    """The latest daily universe written by ``quantzero.universe`` (newest date partition)."""
    base = pathlib.Path(store_root()) / "universe"
    partitions = sorted(base.glob("date=*"), reverse=True)
    if not partitions:
        raise FileNotFoundError(
            f"no universe under {base}; run `python -m quantzero.universe build` first"
        )
    frame = pl.read_parquet(str(partitions[0] / "universe.parquet"), columns=["symbol"])
    return frame["symbol"].to_list()


def resolve_tickers(args: argparse.Namespace) -> list[str]:
    if args.universe:
        tickers = load_universe()
        print(f"universe: {len(tickers)} tickers from the latest daily build")
        return tickers
    return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]


def _print_assignment(tickers: list[str], n_workers: int) -> None:
    assignment = assign_tickers(tickers, n_workers)
    sizes = [len(assignment[w]) for w in range(n_workers)]
    print(f"sharded across {n_workers} workers: {min(sizes)}–{max(sizes)} tickers each")


class LiveSink:
    """Prints a sampled vector summary as workers report them to the router."""

    def __init__(self, sample_every: int = 1) -> None:
        self.count = 0
        self.sample_every = sample_every

    def __call__(self, summary: VectorSummary) -> None:
        self.count += 1
        if self.count % self.sample_every != 0:
            return
        recv_ns = time.time_ns()
        latency_s = max((recv_ns - (summary.ts_ns + _NS_PER_MINUTE)) / 1e9, 0.0)
        when = ns_to_et(summary.ts_ns).strftime("%H:%M")
        print(
            f"{when} ET w{summary.worker_id:<2} {summary.ticker:<6} "
            f"compute={summary.compute_ns / 1e6:6.2f}ms latency={latency_s:5.2f}s "
            f"valid={summary.n_valid}/{summary.n_features}  [{self.count} vectors]"
        )


def run_sim(tickers: list[str], n_workers: int, minutes: int, seed: int, store: bool) -> None:
    print(f"sharded simulation: {len(tickers)} tickers across {n_workers} workers")
    _print_assignment(tickers, n_workers)
    config = SimulationConfig(tickers=tickers, n_minutes=minutes, seed=seed)
    store_config = StoreConfig(store_root(), SET_VERSION, "sim") if store else None
    runner = ShardedRunner(tickers, n_workers, store_config)
    started = time.perf_counter()
    summaries = runner.run_source(SimulationSource(config))
    elapsed = time.perf_counter() - started
    compute = np.array([s.compute_ns for s in summaries], dtype=np.float64) / 1000.0
    print(f"computed {len(summaries)} vectors in {elapsed:.2f}s")
    print(f"compute per vector: mean={compute.mean():.1f}us p99={np.percentile(compute, 99):.1f}us")
    if store:
        print(f"persisted to {store_root()} (source=sim, async)")


def run_backfill(
    tickers: list[str], n_workers: int, dates: list[dt.date], with_ticks: bool
) -> None:
    store_config = StoreConfig(store_root(), SET_VERSION, "backfill")
    print(f"backfill: {len(tickers)} tickers across {n_workers} workers, {len(dates)} day(s)")
    _print_assignment(tickers, n_workers)
    for day in dates:
        runner = ShardedRunner(tickers, n_workers, store_config)
        started = time.perf_counter()
        summaries = runner.run_source(ReplaySource(tickers, day, with_ticks=with_ticks))
        print(
            f"  {day.isoformat()}: {len(summaries)} vectors in {time.perf_counter() - started:.1f}s "
            f"-> source=backfill"
        )


def run_live(tickers: list[str], n_workers: int, with_ticks: bool, sample_every: int) -> None:
    config = alpaca_config()
    store_config = StoreConfig(store_root(), SET_VERSION, "stream")
    print(f"sharded live: {len(tickers)} tickers across {n_workers} workers ({config.data_feed})")
    _print_assignment(tickers, n_workers)

    runner = ShardedRunner(tickers, n_workers, store_config)
    runner.start()
    runner.drain_forever(LiveSink(sample_every))

    # Graceful shutdown: SIGINT/SIGTERM must stop the workers (sentinels -> drain writers ->
    # join), otherwise a hard kill orphans the spawned worker processes.
    def _shutdown(signum: int, frame: FrameType | None) -> None:
        print("\nshutting down workers…")
        runner.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    stream = StockDataStream(config.key_id, config.secret_key, feed=data_feed(config))

    async def on_bar(bar: object) -> None:
        runner.dispatch(bar_to_event(bar.symbol, bar))  # type: ignore[attr-defined]

    async def on_trade(trade: object) -> None:
        runner.dispatch(live_trade_to_event(trade))

    async def on_quote(quote: object) -> None:
        runner.dispatch(live_quote_to_event(quote))

    for start in range(0, len(tickers), SUBSCRIBE_CHUNK):
        chunk = tickers[start : start + SUBSCRIBE_CHUNK]
        stream.subscribe_bars(on_bar, *chunk)
        if with_ticks:
            stream.subscribe_trades(on_trade, *chunk)
            stream.subscribe_quotes(on_quote, *chunk)
    print(f"subscribed bars for {len(tickers)} tickers; streaming (source=stream, async writes)")
    stream.run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline across worker processes.")
    parser.add_argument("mode", choices=["sim", "live", "backfill"])
    parser.add_argument("--tickers", default="AAA,BBB", help="Comma-separated symbols")
    parser.add_argument("--universe", action="store_true", help="Use the latest stored universe")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minutes", type=int, default=120, help="sim only")
    parser.add_argument("--seed", type=int, default=7, help="sim only")
    parser.add_argument("--store", action="store_true", help="sim only: persist as source=sim")
    parser.add_argument("--with-ticks", action="store_true", help="subscribe/replay trades+quotes")
    parser.add_argument("--dates", default="", help="backfill: comma-separated YYYY-MM-DD")
    parser.add_argument("--sample-every", type=int, default=1, help="live: print every Nth vector")
    args = parser.parse_args(argv)

    tickers = resolve_tickers(args)
    if args.mode == "sim":
        run_sim(tickers, args.workers, args.minutes, args.seed, args.store)
    elif args.mode == "backfill":
        dates = [dt.date.fromisoformat(d.strip()) for d in args.dates.split(",") if d.strip()]
        if not dates:
            parser.error("backfill requires --dates YYYY-MM-DD[,YYYY-MM-DD...]")
        run_backfill(tickers, args.workers, dates, args.with_ticks)
    else:
        run_live(tickers, args.workers, args.with_ticks, args.sample_every)


if __name__ == "__main__":
    main()
