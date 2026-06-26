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
import threading
import time
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
TICK_CONCURRENCY = 8  # parallel PER-TICKER tick fetches (bounds memory to one name at a time)

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


def _fetch_ticks_one(args: tuple) -> None:
    """Fetch one ticker-day's trades and/or quotes (per-ticker -> memory-safe). Idempotent."""
    cfg, raw_root_path, day, day_str, feed, ticker, do_trades, do_quotes = args
    store = RawStore(raw_root_path)
    client = _thread_client(cfg)
    if do_trades and not store.has_trades(day_str, ticker):
        store.write_trades(day_str, ticker, fetch_trades_day(client, ticker, day, feed))  # type: ignore[arg-type]
    if do_quotes and not store.has_quotes(day_str, ticker):
        store.write_quotes(day_str, ticker, fetch_quotes_day(client, ticker, day, feed))  # type: ignore[arg-type]


def backfill_raw(
    tickers: list[str],
    days: list[dt.date],
    raw_root_path: str,
    config: AlpacaConfig | None = None,
    fetch_trades: bool = False,
    fetch_quotes: bool = False,
    concurrency: int = TICK_CONCURRENCY,
) -> int:
    """Stage 1: land raw bars (and optionally trades/quotes) from Alpaca. Idempotent/resumable.

    Bars are fetched many-symbols-per-request (tiny). Trades and quotes are fetched PER TICKER
    in parallel — so memory is bounded to one name at a time (most of the 7000+ universe is
    thin and fast) and a heavy name never co-buffers with 50 others. Trades and quotes are
    separate flags so you can keep different lookbacks (e.g. trades 6mo, quotes 5wk).
    """
    cfg = config or alpaca_config()
    client = historical_client(cfg)
    feed = data_feed(cfg)
    store = RawStore(raw_root_path)
    written = 0
    for day in days:
        day_str = day.isoformat()
        missing = [t for t in tickers if not store.has_bars(day_str, t)]
        for chunk in _chunks(missing, RAW_FETCH_CHUNK):
            data = fetch_bars_multi(client, chunk, day, feed)
            for ticker, bars in data.items():
                if store.write_bars(day_str, ticker, bars) is not None:
                    written += 1
        if fetch_trades or fetch_quotes:
            jobs = [
                (cfg, raw_root_path, day, day_str, feed, ticker, fetch_trades, fetch_quotes)
                for ticker in tickers
            ]
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(_fetch_ticks_one, jobs))
        print(
            f"  raw {day_str}: {len(missing)} bar-tickers "
            f"(trades={fetch_trades} quotes={fetch_quotes}); {written} bar-days total"
        )
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
        "--concurrency", type=int, default=TICK_CONCURRENCY, help="parallel tick fetch"
    )
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
            concurrency=args.concurrency,
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
