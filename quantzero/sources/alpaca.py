"""Alpaca adapters: convert SDK objects to quantzero events; historical fetch + replay.

The live websocket path lives in :mod:`quantzero.run_live`. This module covers the
historical REST path used for warmup and for the replay (backfill) source, which feeds
the exact same events through the engine in the exact same per-minute order as live.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable, Iterator
from typing import TypeVar

from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame

from quantzero.clock import date_to_ns_range, to_ns
from quantzero.config import AlpacaConfig, alpaca_config
from quantzero.events import Event, MinuteBar, Quote, Trade
from quantzero.sources.base import order_events_per_minute

_MAX_RETRIES = 5
_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 30.0
_T = TypeVar("_T")


def _with_retry(call: Callable[[], _T]) -> _T:
    """Run an Alpaca request with bounded exponential backoff on transient APIErrors.

    This is what keeps a long, universe-scale backfill from dying on a single 429 / 5xx /
    network blip — the request is retried rather than aborting the whole job.
    """
    attempt = 0
    while True:
        try:
            return call()
        except APIError:
            attempt += 1
            if attempt > _MAX_RETRIES:
                raise
            time.sleep(min(_BACKOFF_BASE_S * 2 ** (attempt - 1), _BACKOFF_CAP_S))


def _utc(start_ns: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(start_ns / 1e9, tz=dt.UTC)


def trade_to_event(ticker: str, trade: object) -> Trade:
    return Trade(
        ticker=ticker,
        ts_ns=to_ns(trade.timestamp),  # type: ignore[attr-defined]
        price=float(trade.price),  # type: ignore[attr-defined]
        size=float(trade.size),  # type: ignore[attr-defined]
    )


def quote_to_event(ticker: str, quote: object) -> Quote:
    return Quote(
        ticker=ticker,
        ts_ns=to_ns(quote.timestamp),  # type: ignore[attr-defined]
        bid=float(quote.bid_price),  # type: ignore[attr-defined]
        ask=float(quote.ask_price),  # type: ignore[attr-defined]
        bid_size=float(quote.bid_size),  # type: ignore[attr-defined]
        ask_size=float(quote.ask_size),  # type: ignore[attr-defined]
    )


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
    client: StockHistoricalDataClient, ticker: str, day: dt.date, feed: DataFeed
) -> list[MinuteBar]:
    """One ET day of 1-minute bars for a ticker, as MinuteBar events."""
    return fetch_bars_multi(client, [ticker], day, feed).get(ticker, [])


def fetch_bars_multi(
    client: StockHistoricalDataClient, tickers: list[str], day: dt.date, feed: DataFeed
) -> dict[str, list[MinuteBar]]:
    """One ET day of 1-minute bars for many tickers in one (retried) request."""
    start_ns, end_ns = date_to_ns_range(day)
    request = StockBarsRequest(
        symbol_or_symbols=list(tickers),
        timeframe=TimeFrame.Minute,
        start=_utc(start_ns),
        end=_utc(end_ns),
        adjustment=Adjustment.RAW,
        feed=feed,
    )
    data = _with_retry(lambda: client.get_stock_bars(request)).data  # type: ignore[union-attr]
    return {t: [bar_to_event(t, bar) for bar in data.get(t, [])] for t in tickers}


def fetch_trades_multi(
    client: StockHistoricalDataClient, tickers: list[str], day: dt.date, feed: DataFeed
) -> dict[str, list[Trade]]:
    """One ET day of trades for many tickers in one (retried, paginated) request."""
    start_ns, end_ns = date_to_ns_range(day)
    request = StockTradesRequest(
        symbol_or_symbols=list(tickers), start=_utc(start_ns), end=_utc(end_ns), feed=feed
    )
    data = _with_retry(lambda: client.get_stock_trades(request)).data  # type: ignore[union-attr]
    return {t: [trade_to_event(t, tr) for tr in data.get(t, [])] for t in tickers}


def fetch_quotes_multi(
    client: StockHistoricalDataClient, tickers: list[str], day: dt.date, feed: DataFeed
) -> dict[str, list[Quote]]:
    """One ET day of NBBO quotes for many tickers in one (retried, paginated) request."""
    start_ns, end_ns = date_to_ns_range(day)
    request = StockQuotesRequest(
        symbol_or_symbols=list(tickers), start=_utc(start_ns), end=_utc(end_ns), feed=feed
    )
    data = _with_retry(lambda: client.get_stock_quotes(request)).data  # type: ignore[union-attr]
    return {t: [quote_to_event(t, q) for q in data.get(t, [])] for t in tickers}


def fetch_trades_day(
    client: StockHistoricalDataClient, ticker: str, day: dt.date, feed: DataFeed
) -> list[Trade]:
    return fetch_trades_multi(client, [ticker], day, feed).get(ticker, [])


def fetch_quotes_day(
    client: StockHistoricalDataClient, ticker: str, day: dt.date, feed: DataFeed
) -> list[Quote]:
    return fetch_quotes_multi(client, [ticker], day, feed).get(ticker, [])


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
