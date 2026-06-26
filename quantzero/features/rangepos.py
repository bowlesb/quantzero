"""Range-position features: where price sits within session and rolling ranges, plus RSI."""

from __future__ import annotations

import math

import numpy as np

from quantzero.caches import Ewma, RollingMax, RollingMin, RunningExtrema
from quantzero.events import MinuteBar
from quantzero.feature import Feature, register

ROLLING_RANGE_WINDOWS = (15, 60)
RSI_PERIOD = 14
# Wilder smoothing (alpha = 1/N) corresponds to an EWMA span of 2N-1.
RSI_SPAN = 2 * RSI_PERIOD - 1


@register
class SessionRange(Feature):
    """Close position within the session high/low and the session range as a percent."""

    name = "sessionrange"
    columns = ("pos", "range_pct")

    def setup(self) -> None:
        self._high = RunningExtrema()
        self._low = RunningExtrema()

    def on_minute(self, bar: MinuteBar) -> None:
        self._high.push(bar.high)
        self._low.push(bar.low)

    def values(self) -> np.ndarray:
        if not self._high.initialized:
            return np.array([math.nan, math.nan])
        high = self._high.max
        low = self._low.min
        close = self.state.minutes.last_close
        span = high - low
        if span <= 0:
            return np.array([0.5, 0.0])
        return np.array([(close - low) / span, span / low if low > 0 else math.nan])


@register
class RollingRangePosition(Feature):
    """Close position within rolling high/low windows (monotonic-deque extrema)."""

    name = "rollrange"
    columns = tuple(f"pos_{w}" for w in ROLLING_RANGE_WINDOWS)

    def setup(self) -> None:
        self._max = {w: RollingMax(w) for w in ROLLING_RANGE_WINDOWS}
        self._min = {w: RollingMin(w) for w in ROLLING_RANGE_WINDOWS}

    def on_minute(self, bar: MinuteBar) -> None:
        for window in ROLLING_RANGE_WINDOWS:
            self._max[window].push(bar.high)
            self._min[window].push(bar.low)

    def values(self) -> np.ndarray:
        close = self.state.minutes.last_close
        out = np.full(len(ROLLING_RANGE_WINDOWS), math.nan)
        for i, window in enumerate(ROLLING_RANGE_WINDOWS):
            high = self._max[window].value
            low = self._min[window].value
            span = high - low
            out[i] = (close - low) / span if span > 0 else 0.5
        return out


@register
class Rsi(Feature):
    """Wilder RSI(14) via EWMA of gains and losses."""

    name = "rsi"
    columns = ("rsi_14",)

    def setup(self) -> None:
        self._gain = Ewma(RSI_SPAN)
        self._loss = Ewma(RSI_SPAN)
        self._prev_close = math.nan

    def on_minute(self, bar: MinuteBar) -> None:
        if self._prev_close == self._prev_close:
            delta = bar.close - self._prev_close
            self._gain.push(max(delta, 0.0))
            self._loss.push(max(-delta, 0.0))
        self._prev_close = bar.close

    def values(self) -> np.ndarray:
        if not self._gain.initialized:
            return np.array([math.nan])
        avg_gain = self._gain.value
        avg_loss = self._loss.value
        if avg_loss <= 0:
            return np.array([100.0 if avg_gain > 0 else 50.0])
        rs = avg_gain / avg_loss
        return np.array([100.0 - 100.0 / (1.0 + rs)])
