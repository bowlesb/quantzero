"""Alpaca adapters: convert SDK objects to quantzero events; historical fetch + replay.

The live websocket path lives in :mod:`quantzero.run_live`. This module covers the
historical REST path used for warmup and for the replay (backfill) source, which feeds
the exact same events through the engine in the exact same per-minute order as live.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame

from quantzero.clock import date_to_ns_range, to_ns
from quantzero.config import AlpacaConfig, alpaca_config
from quantzero.events import Event, MinuteBar, Quote, Trade
from quantzero.sources.base import order_events_per_minute


def data_feed(config: AlpacaConfig) -> DataFeed:
    return DataFeed.SIP if config.data_feed == "sip" else DataFeed.IEX


def historical_client(config: AlpacaConfig | None = None) -> StockHistoricalDataClient:
    cfg = config or alpaca_config()
    return StockHistoricalDataClient(cfg.key_id, cfg.secret_key)


def bar_to_event(ticker: str, bar: object) -> MinuteBar:
    """Convert an alpaca-py bar object to a :class:`MinuteBar`."""
    vwap = getattr(bar, "vwap", None)
    trade_count = getattr(bar, "trade_count", None)
    return MinuteBar(
        ticker=ticker,
        ts_ns=to_ns(bar.timestamp),  # type: ignore[attr-defined]
        open=float(bar.open),  # type: ignore[attr-defined]
        high=float(bar.high),  # type: ignore[attr-defined]
        low=float(bar.low),  # type: ignore[attr-defined]
        close=float(bar.close),  # type: ignore[attr-defined]
        volume=float(bar.volume),  # type: ignore[attr-defined]
        trade_count=int(trade_count) if trade_count is not None else 0,
        vwap=float(vwap) if vwap is not None else float(bar.close),  # type: ignore[attr-defined]
    )


def fetch_bars_day(
    client: StockHistoricalDataClient,
    ticker: str,
    day: dt.date,
    feed: DataFeed,
) -> list[MinuteBar]:
    """One ET day of 1-minute bars for a ticker, as MinuteBar events."""
    start_ns, end_ns = date_to_ns_range(day)
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=dt.datetime.fromtimestamp(start_ns / 1e9, tz=dt.UTC),
        end=dt.datetime.fromtimestamp(end_ns / 1e9, tz=dt.UTC),
        adjustment=Adjustment.RAW,
        feed=feed,
    )
    barset = client.get_stock_bars(request)
    bars = barset.data.get(ticker, [])  # type: ignore[union-attr]
    return [bar_to_event(ticker, bar) for bar in bars]


def fetch_bars_multi(
    client: StockHistoricalDataClient,
    tickers: list[str],
    day: dt.date,
    feed: DataFeed,
) -> dict[str, list[MinuteBar]]:
    """One ET day of 1-minute bars for many tickers in one request (chunked by the SDK)."""
    start_ns, end_ns = date_to_ns_range(day)
    request = StockBarsRequest(
        symbol_or_symbols=list(tickers),
        timeframe=TimeFrame.Minute,
        start=dt.datetime.fromtimestamp(start_ns / 1e9, tz=dt.UTC),
        end=dt.datetime.fromtimestamp(end_ns / 1e9, tz=dt.UTC),
        adjustment=Adjustment.RAW,
        feed=feed,
    )
    barset = client.get_stock_bars(request)
    data = barset.data  # type: ignore[union-attr]
    return {t: [bar_to_event(t, bar) for bar in data.get(t, [])] for t in tickers}


def fetch_trades_day(
    client: StockHistoricalDataClient,
    ticker: str,
    day: dt.date,
    feed: DataFeed,
) -> list[Trade]:
    start_ns, end_ns = date_to_ns_range(day)
    request = StockTradesRequest(
        symbol_or_symbols=ticker,
        start=dt.datetime.fromtimestamp(start_ns / 1e9, tz=dt.UTC),
        end=dt.datetime.fromtimestamp(end_ns / 1e9, tz=dt.UTC),
        feed=feed,
    )
    tradeset = client.get_stock_trades(request)
    trades = tradeset.data.get(ticker, [])  # type: ignore[union-attr]
    return [
        Trade(ticker=ticker, ts_ns=to_ns(t.timestamp), price=float(t.price), size=float(t.size))
        for t in trades
    ]


def fetch_quotes_day(
    client: StockHistoricalDataClient,
    ticker: str,
    day: dt.date,
    feed: DataFeed,
) -> list[Quote]:
    start_ns, end_ns = date_to_ns_range(day)
    request = StockQuotesRequest(
        symbol_or_symbols=ticker,
        start=dt.datetime.fromtimestamp(start_ns / 1e9, tz=dt.UTC),
        end=dt.datetime.fromtimestamp(end_ns / 1e9, tz=dt.UTC),
        feed=feed,
    )
    quoteset = client.get_stock_quotes(request)
    quotes = quoteset.data.get(ticker, [])  # type: ignore[union-attr]
    return [
        Quote(
            ticker=ticker,
            ts_ns=to_ns(q.timestamp),
            bid=float(q.bid_price),
            ask=float(q.ask_price),
            bid_size=float(q.bid_size),
            ask_size=float(q.ask_size),
        )
        for q in quotes
    ]


class ReplaySource:
    """Backfill as replay: fetch one historical day and feed it like the live stream."""

    def __init__(
        self,
        tickers: list[str],
        day: dt.date,
        with_ticks: bool = False,
        config: AlpacaConfig | None = None,
    ) -> None:
        self.tickers = tickers
        self.day = day
        self.with_ticks = with_ticks
        self.config = config or alpaca_config()
        self._client = historical_client(self.config)
        self._feed = data_feed(self.config)

    def iter_events(self) -> Iterator[Event]:
        collected: list[Event] = []
        for ticker in self.tickers:
            collected.extend(fetch_bars_day(self._client, ticker, self.day, self._feed))
            if self.with_ticks:
                collected.extend(fetch_quotes_day(self._client, ticker, self.day, self._feed))
                collected.extend(fetch_trades_day(self._client, ticker, self.day, self._feed))
        yield from order_events_per_minute(collected)
