"""Immutable market-data events.

Every event carries ``ts_ns`` — an integer UTC epoch in nanoseconds. Using an int
keeps the hot path allocation-free and makes ordering across event types trivial. The
event core consumes these one at a time, in non-decreasing ``ts_ns`` order, regardless
of whether they came from the live websocket or a historical replay.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Quote:
    """A National Best Bid/Offer snapshot."""

    ticker: str
    ts_ns: int
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True, slots=True)
class Trade:
    """A single executed trade (print)."""

    ticker: str
    ts_ns: int
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class MinuteBar:
    """A completed one-minute OHLCV bar. ``ts_ns`` is the bar's opening minute."""

    ticker: str
    ts_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    vwap: float


Event = Quote | Trade | MinuteBar
