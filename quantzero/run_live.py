"""Stream live Alpaca data and print a feature vector + latency on each minute bar.

Run at the market open:

    make live ARGS="--tickers AAPL,MSFT,NVDA"
    # or
    python -m quantzero.run_live --tickers AAPL,MSFT,NVDA --warmup

``--warmup`` replays today's earlier bars through the engine via REST before streaming, so
caches are warm and the first live vector is already meaningful.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
from alpaca.data.live import StockDataStream

from quantzero.clock import ET, ns_to_et, to_ns
from quantzero.config import alpaca_config, metrics_port
from quantzero.driver import EngineDriver
from quantzero.engine import FeatureEngine, FeatureVector
from quantzero.events import Quote, Trade
from quantzero.features import default_features
from quantzero.metrics import (
    BAR_TO_VECTOR_SECONDS,
    FEATURE_COMPUTE_SECONDS,
    VECTORS_TOTAL,
    start_metrics_server,
)
from quantzero.sources.alpaca import ReplaySource, bar_to_event, data_feed

_NS_PER_MINUTE = 60_000_000_000
DISPLAY_FEATURES = ("returns_r_1", "vwap_dist", "rsi_rsi_14", "macd_hist", "tradeflow_ofi_1m")


def live_trade_to_event(trade: object) -> Trade:
    return Trade(
        ticker=trade.symbol,  # type: ignore[attr-defined]
        ts_ns=to_ns(trade.timestamp),  # type: ignore[attr-defined]
        price=float(trade.price),  # type: ignore[attr-defined]
        size=float(trade.size),  # type: ignore[attr-defined]
    )


def live_quote_to_event(quote: object) -> Quote:
    return Quote(
        ticker=quote.symbol,  # type: ignore[attr-defined]
        ts_ns=to_ns(quote.timestamp),  # type: ignore[attr-defined]
        bid=float(quote.bid_price),  # type: ignore[attr-defined]
        ask=float(quote.ask_price),  # type: ignore[attr-defined]
        bid_size=float(quote.bid_size),  # type: ignore[attr-defined]
        ask_size=float(quote.ask_size),  # type: ignore[attr-defined]
    )


def display_indices(columns: list[str]) -> list[tuple[str, int]]:
    lookup = {name: idx for idx, name in enumerate(columns)}
    return [(name, lookup[name]) for name in DISPLAY_FEATURES if name in lookup]


class LiveHandlers:
    """Async websocket handlers that route events into the engine driver."""

    def __init__(self, driver: EngineDriver, display: list[tuple[str, int]]) -> None:
        self.driver = driver
        self.display = display
        self.vectors = 0

    async def on_bar(self, bar: object) -> None:
        recv_ns = time.time_ns()
        vector = self.driver.process(bar_to_event(bar.symbol, bar))  # type: ignore[attr-defined]
        if vector is None:
            return
        close_ns = vector.ts_ns + _NS_PER_MINUTE
        latency_s = max((recv_ns - close_ns) / 1e9, 0.0)
        compute_s = vector.compute_ns / 1e9
        BAR_TO_VECTOR_SECONDS.observe(latency_s)
        FEATURE_COMPUTE_SECONDS.observe(compute_s)
        VECTORS_TOTAL.labels(ticker=vector.ticker).inc()
        self.vectors += 1
        print(self._format(vector, latency_s, compute_s))

    async def on_trade(self, trade: object) -> None:
        self.driver.process(live_trade_to_event(trade))

    async def on_quote(self, quote: object) -> None:
        self.driver.process(live_quote_to_event(quote))

    def _format(self, vector: FeatureVector, latency_s: float, compute_s: float) -> str:
        when = ns_to_et(vector.ts_ns).strftime("%H:%M")
        n_valid = int(np.count_nonzero(~np.isnan(vector.values)))
        head = (
            f"{when} ET {vector.ticker:<6} "
            f"compute={compute_s * 1e3:6.2f}ms latency={latency_s:5.2f}s "
            f"valid={n_valid}/{len(vector.columns)}"
        )
        samples = "  ".join(f"{name}={vector.values[idx]:+.4f}" for name, idx in self.display)
        return f"{head} | {samples}"


def warmup(driver: EngineDriver, tickers: list[str], day: dt.date) -> int:
    """Replay today's bars-so-far through the engine so caches are warm. Returns bar count."""
    source = ReplaySource(tickers, day, with_ticks=False)
    count = 0
    for _ in driver.run_source(source):
        count += 1
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Alpaca data and emit feature vectors.")
    parser.add_argument("--tickers", required=True, help="Comma-separated symbols, e.g. AAPL,MSFT")
    parser.add_argument(
        "--warmup", action="store_true", help="Replay today's bars before streaming"
    )
    parser.add_argument(
        "--no-ticks", action="store_true", help="Subscribe bars only (no trades/quotes)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tickers = [symbol.strip().upper() for symbol in args.tickers.split(",") if symbol.strip()]
    config = alpaca_config()

    driver = EngineDriver(tickers, default_features())
    columns = FeatureEngine(tickers[0], default_features()).columns
    handlers = LiveHandlers(driver, display_indices(columns))

    start_metrics_server(metrics_port())

    if args.warmup:
        today = dt.datetime.now(ET).date()
        n_bars = warmup(driver, tickers, today)
        print(f"warmup: replayed {n_bars} bars for {len(tickers)} tickers")

    stream = StockDataStream(config.key_id, config.secret_key, feed=data_feed(config))
    stream.subscribe_bars(handlers.on_bar, *tickers)
    if not args.no_ticks:
        stream.subscribe_trades(handlers.on_trade, *tickers)
        stream.subscribe_quotes(handlers.on_quote, *tickers)

    print(f"streaming {len(tickers)} tickers ({config.data_feed} feed): {', '.join(tickers)}")
    stream.run()


if __name__ == "__main__":
    main()
