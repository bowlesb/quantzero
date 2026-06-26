"""Complex microstructure features driven by individual trades and quotes.

These exercise the full sequential multi-stream path: trades and quotes update per-minute
accumulators on every ``on_trade`` / ``on_quote``, then the minute's bar (``on_minute``)
finalizes the statistics and resets. They are NaN in a bars-only backfill and populate only
when the raw store (or live stream) carries trades and quotes.
"""

from __future__ import annotations

import math

import numpy as np

from quantzero.caches import RollingMoments
from quantzero.feature import Feature, register

FLOW_WINDOW = 20


@register
class TradeMicro(Feature):
    """Within-minute trade microstructure, computed from the actual trade ticks."""

    name = "trademicro"
    columns = ("n_trades", "avg_size", "vwap_dist", "signed_frac", "top_size_frac")

    def setup(self) -> None:
        self._out = (math.nan,) * len(self.columns)
        self._reset()

    def _reset(self) -> None:
        self._n = 0
        self._vol = 0.0
        self._pv = 0.0  # sum(price * size) -> trade VWAP
        self._signed = 0.0  # signed volume vs the prevailing mid
        self._max_size = 0.0

    def on_trade(self) -> None:
        trade = self.state.last_trade
        if trade is None:
            return
        quote = self.state.last_quote
        reference = quote.mid if quote is not None else trade.price
        self._n += 1
        self._vol += trade.size
        self._pv += trade.price * trade.size
        self._signed += trade.size if trade.price >= reference else -trade.size
        if trade.size > self._max_size:
            self._max_size = trade.size

    def on_minute(self) -> None:
        if self._vol > 0 and self._n > 0:
            trade_vwap = self._pv / self._vol
            close = self.state.minutes.last_close
            self._out = (
                float(self._n),
                self._vol / self._n,
                close / trade_vwap - 1.0 if trade_vwap > 0 else math.nan,
                self._signed / self._vol,
                self._max_size / self._vol,
            )
        else:
            self._out = (0.0, math.nan, math.nan, math.nan, math.nan)
        self._reset()

    def values(self) -> np.ndarray:
        return np.array(self._out)


@register
class QuoteMicro(Feature):
    """Within-minute, TIME-WEIGHTED quote microstructure from the NBBO tick stream."""

    name = "quotemicro"
    columns = ("n_quotes", "twa_spread_bps", "twa_imbalance", "spread_range_bps")

    def setup(self) -> None:
        self._out = (math.nan,) * len(self.columns)
        self._reset()

    def _reset(self) -> None:
        self._n = 0
        self._prev_ts: int | None = None
        self._prev_spread = math.nan
        self._prev_imbalance = math.nan
        self._weight = 0.0  # total dt
        self._w_spread = 0.0  # sum(prev_spread * dt)
        self._w_imbalance = 0.0
        self._max_spread = -math.inf
        self._min_spread = math.inf

    def on_quote(self) -> None:
        quote = self.state.last_quote
        if quote is None:
            return
        mid = quote.mid
        if mid <= 0:
            return
        spread_bps = (quote.ask - quote.bid) / mid * 1e4
        total_size = quote.bid_size + quote.ask_size
        imbalance = (quote.bid_size - quote.ask_size) / total_size if total_size > 0 else 0.0
        self._n += 1
        # The previous quote prevailed over [prev_ts, now]; weight its values by that dt.
        if self._prev_ts is not None and self._prev_spread == self._prev_spread:
            dt_ns = max(quote.ts_ns - self._prev_ts, 0)
            self._weight += dt_ns
            self._w_spread += self._prev_spread * dt_ns
            self._w_imbalance += self._prev_imbalance * dt_ns
        self._prev_ts = quote.ts_ns
        self._prev_spread = spread_bps
        self._prev_imbalance = imbalance
        self._max_spread = max(self._max_spread, spread_bps)
        self._min_spread = min(self._min_spread, spread_bps)

    def on_minute(self) -> None:
        spread_range = (
            self._max_spread - self._min_spread
            if self._max_spread >= self._min_spread
            else math.nan
        )
        if self._weight > 0:
            self._out = (
                float(self._n),
                self._w_spread / self._weight,
                self._w_imbalance / self._weight,
                spread_range,
            )
        elif self._n > 0:  # one quote, no elapsed time to weight: fall back to its level
            self._out = (float(self._n), self._prev_spread, self._prev_imbalance, 0.0)
        else:
            self._out = (0.0, math.nan, math.nan, math.nan)
        self._reset()

    def values(self) -> np.ndarray:
        return np.array(self._out)


@register
class FlowFreq(Feature):
    """Inter-minute frequency dynamics: how this minute's trade/quote rates compare to recent."""

    name = "flowfreq"
    columns = ("trade_rate_z", "quote_rate_z", "trade_quote_ratio")

    def setup(self) -> None:
        self._trades = RollingMoments(FLOW_WINDOW)
        self._quotes = RollingMoments(FLOW_WINDOW)
        self._minute_trades = 0
        self._minute_quotes = 0
        self._out = (math.nan, math.nan, math.nan)

    def on_trade(self) -> None:
        self._minute_trades += 1

    def on_quote(self) -> None:
        self._minute_quotes += 1

    def on_minute(self) -> None:
        trade_count = float(self._minute_trades)
        quote_count = float(self._minute_quotes)
        # z-score this minute against the PRIOR window (compute before pushing -> no lookahead).
        trade_z = self._z(trade_count, self._trades)
        quote_z = self._z(quote_count, self._quotes)
        ratio = trade_count / quote_count if quote_count > 0 else math.nan
        self._trades.push(trade_count)
        self._quotes.push(quote_count)
        self._out = (trade_z, quote_z, ratio)
        self._minute_trades = 0
        self._minute_quotes = 0

    @staticmethod
    def _z(value: float, moments: RollingMoments) -> float:
        std = moments.std
        if std == std and std > 0:
            return (value - moments.mean) / std
        return math.nan

    def values(self) -> np.ndarray:
        return np.array(self._out)
