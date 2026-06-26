"""Microstructure features driven by trades and quotes.

These rely on the engine contract that a minute's trades and quotes are delivered before
that minute's bar (the live stream emits the bar at minute close; the replay source emits
each minute's ticks before its bar). So at ``on_minute`` the per-minute accumulators hold
exactly the just-completed minute's flow.
"""

from __future__ import annotations

import math

import numpy as np

from quantzero.caches import Ewma
from quantzero.events import MinuteBar, Quote, Trade
from quantzero.feature import Feature, register

SPREAD_EWMA_SPAN = 50


@register
class TradeFlow(Feature):
    """Order-flow imbalance: trades classified buy/sell against the prevailing mid."""

    name = "tradeflow"
    columns = ("ofi_session", "ofi_1m")

    def setup(self) -> None:
        self._session_signed = 0.0
        self._session_volume = 0.0
        self._minute_signed = 0.0
        self._minute_volume = 0.0
        self._ofi_1m = math.nan

    def on_trade(self, trade: Trade) -> None:
        quote = self.state.last_quote
        reference = quote.mid if quote is not None else trade.price
        sign = 1.0 if trade.price >= reference else -1.0
        signed = sign * trade.size
        self._session_signed += signed
        self._session_volume += trade.size
        self._minute_signed += signed
        self._minute_volume += trade.size

    def on_minute(self, bar: MinuteBar) -> None:
        self._ofi_1m = (
            self._minute_signed / self._minute_volume if self._minute_volume > 0 else math.nan
        )
        self._minute_signed = 0.0
        self._minute_volume = 0.0

    def values(self) -> np.ndarray:
        ofi_session = (
            self._session_signed / self._session_volume if self._session_volume > 0 else math.nan
        )
        return np.array([ofi_session, self._ofi_1m])


@register
class QuoteSpread(Feature):
    """Spread (bps), size imbalance, and microprice distance from the latest quote."""

    name = "quotespread"
    columns = ("spread_bps", "imbalance", "microprice_dist")

    def values(self) -> np.ndarray:
        quote = self.state.last_quote
        if quote is None:
            return np.array([math.nan, math.nan, math.nan])
        mid = quote.mid
        if mid <= 0:
            return np.array([math.nan, math.nan, math.nan])
        spread_bps = (quote.ask - quote.bid) / mid * 1e4
        total_size = quote.bid_size + quote.ask_size
        if total_size <= 0:
            return np.array([spread_bps, math.nan, math.nan])
        imbalance = (quote.bid_size - quote.ask_size) / total_size
        microprice = (quote.ask * quote.bid_size + quote.bid * quote.ask_size) / total_size
        return np.array([spread_bps, imbalance, microprice / mid - 1.0])


@register
class QuoteDynamics(Feature):
    """EWMA of the spread and the current spread relative to it."""

    name = "quotedyn"
    columns = ("spread_ewma_bps", "spread_ratio")

    def setup(self) -> None:
        self._spread_ewma = Ewma(SPREAD_EWMA_SPAN)
        self._last_spread_bps = math.nan

    def on_quote(self, quote: Quote) -> None:
        mid = quote.mid
        if mid > 0:
            spread_bps = (quote.ask - quote.bid) / mid * 1e4
            self._spread_ewma.push(spread_bps)
            self._last_spread_bps = spread_bps

    def values(self) -> np.ndarray:
        ewma = self._spread_ewma.value
        if not self._spread_ewma.initialized or ewma <= 0:
            return np.array([math.nan, math.nan])
        return np.array([ewma, self._last_spread_bps / ewma])
