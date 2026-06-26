"""Shared test helpers."""

from __future__ import annotations

import datetime as dt

from quantzero.clock import ET, to_ns
from quantzero.engine import FeatureEngine, FeatureVector
from quantzero.events import MinuteBar
from quantzero.feature import Feature

_NS_PER_MINUTE = 60_000_000_000
_SESSION_OPEN = to_ns(dt.datetime(2026, 6, 24, 9, 30, tzinfo=ET))


def make_bar(
    ticker: str,
    minute: int,
    close: float,
    *,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1000.0,
    trade_count: int = 50,
) -> MinuteBar:
    open_price = open_ if open_ is not None else close
    return MinuteBar(
        ticker=ticker,
        ts_ns=_SESSION_OPEN + minute * _NS_PER_MINUTE,
        open=open_price,
        high=high if high is not None else max(open_price, close),
        low=low if low is not None else min(open_price, close),
        close=close,
        volume=volume,
        trade_count=trade_count,
        vwap=close,
    )


def run_closes(
    feature_classes: list[type[Feature]], closes: list[float], ticker: str = "T"
) -> FeatureVector:
    """Feed a sequence of close-only bars and return the final feature vector."""
    engine = FeatureEngine(ticker, feature_classes)
    vector: FeatureVector | None = None
    for minute, close in enumerate(closes):
        vector = engine.on_minute(make_bar(ticker, minute, close))
    assert vector is not None
    return vector


def column_value(vector: FeatureVector, name: str) -> float:
    return float(vector.values[vector.columns.index(name)])
