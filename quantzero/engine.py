"""The per-ticker feature engine: events in, a feature vector out on each minute bar."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from quantzero.clock import et_session_date
from quantzero.events import Event, MinuteBar, Quote, Trade
from quantzero.feature import Feature
from quantzero.state import TickerState


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """The computed features for one ticker at one minute boundary."""

    ticker: str
    ts_ns: int
    columns: list[str]
    values: np.ndarray
    compute_ns: int


class FeatureEngine:
    """Owns one ticker's state and feature instances; rebuilds them each session."""

    def __init__(self, ticker: str, feature_classes: list[type[Feature]]) -> None:
        self.ticker = ticker
        self.state = TickerState(ticker)
        self._feature_classes = feature_classes
        self._features: list[Feature] = []
        self._columns: list[str] = []
        self._width = 0

    @property
    def columns(self) -> list[str]:
        if not self._columns:
            self._build_for_session(0)
        return self._columns

    def _build_for_session(self, session_date: int) -> None:
        self.state.reset(session_date)
        self._features = [cls(self.ticker, self.state) for cls in self._feature_classes]
        self._columns = [name for feature in self._features for name in feature.output_names]
        self._width = len(self._columns)

    def on_event(self, event: Event) -> FeatureVector | None:
        """Dispatch any event; returns a FeatureVector only on a minute bar."""
        if isinstance(event, MinuteBar):
            return self.on_minute(event)
        if isinstance(event, Trade):
            self.on_trade(event)
            return None
        self.on_quote(event)
        return None

    def on_quote(self, quote: Quote) -> None:
        self.state.on_quote(quote)
        for feature in self._features:
            feature.on_quote(quote)

    def on_trade(self, trade: Trade) -> None:
        self.state.on_trade(trade)
        for feature in self._features:
            feature.on_trade(trade)

    def on_minute(self, bar: MinuteBar) -> FeatureVector:
        session_date = et_session_date(bar.ts_ns)
        if not self._features or session_date != self.state.session_date:
            self._build_for_session(session_date)
        self.state.on_minute(bar)

        start = time.perf_counter_ns()
        out = np.empty(self._width, dtype=np.float64)
        offset = 0
        for feature in self._features:
            feature.on_minute(bar)
            block = feature.values()
            width = len(feature.columns)
            out[offset : offset + width] = block
            offset += width
        compute_ns = time.perf_counter_ns() - start

        return FeatureVector(
            ticker=self.ticker,
            ts_ns=bar.ts_ns,
            columns=self._columns,
            values=out,
            compute_ns=compute_ns,
        )
