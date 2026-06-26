"""Two-stage backfill, with a clean separation of concerns.

stage 1  raw      : Alpaca REST  -> raw store.   The ONLY Alpaca consumer for backfill.
stage 2  features : raw store    -> engine -> feature store (source=backfill, batched).
                    Never touches Alpaca; replays the raw bars landed by stage 1 through
                    the SAME engine the live stream uses, so backfill == live by design.

  python -m quantzero.backfill raw      --tickers AAPL,MSFT --start 2026-05-26 --end 2026-06-26
  python -m quantzero.backfill features --tickers AAPL,MSFT --start 2026-05-26 --end 2026-06-26
  python -m quantzero.backfill both     --universe --start 2026-05-26 --end 2026-06-26 --workers 8
"""

from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from quantzero.config import AlpacaConfig, alpaca_config, raw_root, store_root
from quantzero.driver import EngineDriver
from quantzero.features import default_features
from quantzero.raw_store import RawReplaySource, RawStore
from quantzero.sources.alpaca import (
    data_feed,
    fetch_bars_multi,
    fetch_quotes_day,
    fetch_trades_day,
    historical_client,
)
from quantzero.store import FeatureStore

SET_VERSION = "0.1.0"
RAW_FETCH_CHUNK = 200  # bars are tiny -> many symbols per request
# Two-level download fan-out: PROCESSES x THREADS. Threads overlap the network waits (the
# work is mostly I/O); processes parallelize the response parsing + parquet writing (the CPU
# part the GIL would otherwise serialize). Default to all cores x 128 threads.
DEFAULT_THREADS = 128

# Each fetch thread reuses its own historical client (the SDK's session isn't shared-safe).
_thread_local = threading.local()


def _thread_client(config: AlpacaConfig) -> object:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = historical_client(config)
        _thread_local.client = client
    return client


def trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """Weekdays in [start, end] (Alpaca returns empty for holidays, which we skip on write)."""
    days: list[dt.date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _fetch_bars_chunk(args: tuple) -> int:
    """Fetch one (day, symbol-chunk) of bars and write them. Returns ticker-days written."""
    cfg, raw_root_path, day, day_str, feed, chunk = args
    store = RawStore(raw_root_path)
    client = _thread_client(cfg)
    data = fetch_bars_multi(client, chunk, day, feed)  # type: ignore[arg-type]
    written = 0
    for ticker, bars in data.items():
        if store.write_bars(day_str, ticker, bars) is not None:
            written += 1
    return written


def _fetch_ticks_one(args: tuple) -> None:
    """Fetch one ticker-day's trades and/or quotes (per-ticker -> memory-safe). Idempotent."""
    cfg, raw_root_path, day, day_str, feed, ticker, do_trades, do_quotes = args
    store = RawStore(raw_root_path)
    client = _thread_client(cfg)
    if do_trades and not store.has_trades(day_str, ticker):
        store.write_trades(day_str, ticker, fetch_trades_day(client, ticker, day, feed))  # type: ignore[arg-type]
    if do_quotes and not store.has_quotes(day_str, ticker):
        store.write_quotes(day_str, ticker, fetch_quotes_day(client, ticker, day, feed))  # type: ignore[arg-type]


def _thread_run(fn: Callable[[tuple], object], jobs: list, threads: int) -> int:
    """Run jobs across a thread pool inside one process (overlaps network waits)."""
    total = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for result in pool.map(fn, jobs):
            total += result if isinstance(result, int) else 0
    return total


def _bar_batch(payload: tuple) -> int:
    batch, threads = payload
    return _thread_run(_fetch_bars_chunk, batch, threads)


def _tick_batch(payload: tuple) -> int:
    batch, threads = payload
    _thread_run(_fetch_ticks_one, batch, threads)
    return 0


def _split(items: list, n: int) -> list[list]:
    """Round-robin split into <= n roughly-equal batches (one per process)."""
    buckets: list[list] = [[] for _ in range(max(n, 1))]
    for i, item in enumerate(items):
        buckets[i % len(buckets)].append(item)
    return [b for b in buckets if b]


def _run_batches(
    jobs: list, batch_fn: Callable[[tuple], int], processes: int, threads: int, label: str
) -> int:
    if not jobs:
        return 0
    batches = _split(jobs, processes)
    total = 0
    with mp.get_context("spawn").Pool(len(batches)) as pool:
        for done, count in enumerate(
            pool.imap_unordered(batch_fn, [(batch, threads) for batch in batches]), start=1
        ):
            total += count
            print(f"  {label}: {done}/{len(batches)} process-batches done", flush=True)
    return total


def backfill_raw(
    tickers: list[str],
    days: list[dt.date],
    raw_root_path: str,
    config: AlpacaConfig | None = None,
    fetch_trades: bool = False,
    fetch_quotes: bool = False,
    processes: int | None = None,
    threads: int = DEFAULT_THREADS,
) -> int:
    """Stage 1: land raw bars (and optionally trades/quotes) from Alpaca. Idempotent/resumable.

    Fan-out is PROCESSES x THREADS. Bars go many-symbols-per-request across (day, chunk);
    trades/quotes go PER TICKER so memory is bounded to one name at a time. Trades and quotes
    are separate flags so they can keep different lookbacks (e.g. trades 6mo, quotes 5wk).
    """
    cfg = config or alpaca_config()
    feed = data_feed(cfg)
    store = RawStore(raw_root_path)
    procs = processes or (os.cpu_count() or 1)

    bar_jobs = [
        (cfg, raw_root_path, day, day.isoformat(), feed, chunk)
        for day in days
        for chunk in _chunks(
            [t for t in tickers if not store.has_bars(day.isoformat(), t)], RAW_FETCH_CHUNK
        )
    ]
    written = _run_batches(bar_jobs, _bar_batch, procs, threads, "bars")

    if fetch_trades or fetch_quotes:
        tick_jobs = [
            (cfg, raw_root_path, day, day.isoformat(), feed, ticker, fetch_trades, fetch_quotes)
            for day in days
            for ticker in tickers
        ]
        _run_batches(tick_jobs, _tick_batch, procs, threads, "ticks")
    return written


def _feature_job(job: tuple[str, str, str, str]) -> int:
    ticker, day_str, raw_root_path, feature_root = job
    source = RawReplaySource([ticker], day_str, raw_root_path)
    driver = EngineDriver([ticker], default_features())
    vectors = list(driver.run_source(source))
    if vectors:
        FeatureStore(feature_root, SET_VERSION, "backfill").write_day(vectors)
    return len(vectors)


def backfill_features(
    tickers: list[str],
    days: list[dt.date],
    raw_root_path: str,
    feature_root: str,
    workers: int = 8,
) -> int:
    """Stage 2: replay the raw store through the engine into the feature store (batched)."""
    store = RawStore(raw_root_path)
    jobs = [
        (ticker, day.isoformat(), raw_root_path, feature_root)
        for day in days
        for ticker in tickers
        if store.has_bars(day.isoformat(), ticker)
    ]
    if not jobs:
        return 0
    total = 0
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            for n_vectors in pool.imap_unordered(_feature_job, jobs):
                total += n_vectors
    else:
        for job in jobs:
            total += _feature_job(job)
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Two-stage backfill (raw -> features).")
    parser.add_argument("stage", choices=["raw", "features", "both"])
    parser.add_argument("--tickers", default="", help="comma-separated symbols")
    parser.add_argument("--universe", action="store_true", help="use the latest stored universe")
    parser.add_argument("--start", required=True, help="ET start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", required=True, help="ET end date YYYY-MM-DD (inclusive)")
    parser.add_argument("--workers", type=int, default=8, help="feature-stage processes")
    parser.add_argument("--trades", action="store_true", help="fetch trades")
    parser.add_argument("--quotes", action="store_true", help="fetch quotes")
    parser.add_argument("--with-ticks", action="store_true", help="fetch both trades and quotes")
    parser.add_argument(
        "--processes", type=int, default=None, help="fetch processes (default: all cores)"
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="threads per process")
    args = parser.parse_args(argv)
    fetch_trades = args.trades or args.with_ticks
    fetch_quotes = args.quotes or args.with_ticks

    if args.universe:
        from quantzero.run_sharded import load_universe

        tickers = load_universe()
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    days = trading_days(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))
    print(
        f"{args.stage}: {len(tickers)} tickers x {len(days)} weekdays "
        f"({args.start}..{args.end})"
    )

    if args.stage in ("raw", "both"):
        started = time.perf_counter()
        n = backfill_raw(
            tickers,
            days,
            raw_root(),
            fetch_trades=fetch_trades,
            fetch_quotes=fetch_quotes,
            processes=args.processes,
            threads=args.threads,
        )
        print(
            f"stage raw: wrote {n} bar-days to {raw_root()} in {time.perf_counter()-started:.1f}s"
        )
    if args.stage in ("features", "both"):
        started = time.perf_counter()
        n = backfill_features(tickers, days, raw_root(), store_root(), args.workers)
        print(
            f"stage features: wrote {n} vectors to {store_root()} (source=backfill) "
            f"in {time.perf_counter()-started:.1f}s"
        )


if __name__ == "__main__":
    main()
