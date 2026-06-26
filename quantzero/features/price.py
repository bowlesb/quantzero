"""Price / trend features: returns, EMAs, MACD, momentum, intraday gap, candle shape."""

from __future__ import annotations

import math

import numpy as np

from quantzero.caches import Ewma
from quantzero.events import MinuteBar
from quantzero.feature import Feature, register

RETURN_HORIZONS = (1, 2, 5, 15, 30)
EMA_SPANS = (9, 12, 26, 50)


@register
class Returns(Feature):
    """Simple close-to-close returns over several horizons. Pure lookups."""

    name = "returns"
    columns = tuple(f"r_{h}" for h in RETURN_HORIZONS)

    def values(self) -> np.ndarray:
        close = self.state.minutes.close
        n = self.state.minutes.n
        out = np.full(len(RETURN_HORIZONS), math.nan)
        for i, horizon in enumerate(RETURN_HORIZONS):
            if n > horizon:
                past = close[n - 1 - horizon]
                if past > 0.0:
                    out[i] = close[n - 1] / past - 1.0
        return out


@register
class Ema(Feature):
    """EMAs of close and the close's distance from each (in return units)."""

    name = "ema"
    columns = (*(f"ema_{span}" for span in EMA_SPANS), "dist_9", "dist_50")

    def setup(self) -> None:
        self._emas = {span: Ewma(span) for span in EMA_SPANS}

    def on_minute(self, bar: MinuteBar) -> None:
        for ema in self._emas.values():
            ema.push(bar.close)

    def values(self) -> np.ndarray:
        close = self.state.minutes.last_close
        levels = [self._emas[span].value for span in EMA_SPANS]
        dist_9 = close / levels[0] - 1.0 if levels[0] > 0 else math.nan
        dist_50 = close / levels[3] - 1.0 if levels[3] > 0 else math.nan
        return np.array([*levels, dist_9, dist_50])


@register
class Macd(Feature):
    """MACD line, signal line, and histogram from 12/26/9 EMAs."""

    name = "macd"
    columns = ("macd", "signal", "hist")

    def setup(self) -> None:
        self._fast = Ewma(12)
        self._slow = Ewma(26)
        self._signal = Ewma(9)
        self._macd = math.nan
        self._sig = math.nan

    def on_minute(self, bar: MinuteBar) -> None:
        self._fast.push(bar.close)
        self._slow.push(bar.close)
        self._macd = self._fast.value - self._slow.value
        self._signal.push(self._macd)
        self._sig = self._signal.value

    def values(self) -> np.ndarray:
        return np.array([self._macd, self._sig, self._macd - self._sig])


@register
class Momentum(Feature):
    """Rate of change over 10 bars and one-bar return acceleration."""

    name = "momentum"
    columns = ("roc_10", "accel")

    def setup(self) -> None:
        self._prev_r1 = math.nan

    def values(self) -> np.ndarray:
        close = self.state.minutes.close
        n = self.state.minutes.n
        roc = math.nan
        if n > 10 and close[n - 11] > 0:
            roc = close[n - 1] / close[n - 11] - 1.0
        r1 = math.nan
        if n > 1 and close[n - 2] > 0:
            r1 = close[n - 1] / close[n - 2] - 1.0
        accel = r1 - self._prev_r1 if (r1 == r1 and self._prev_r1 == self._prev_r1) else math.nan
        self._prev_r1 = r1
        return np.array([roc, accel])


@register
class IntradayGap(Feature):
    """Move from the session's opening price: current close and session high vs open."""

    name = "gap"
    columns = ("from_open", "high_from_open")

    def setup(self) -> None:
        self._open = math.nan
        self._session_high = -math.inf

    def on_minute(self, bar: MinuteBar) -> None:
        if self._open != self._open:
            self._open = bar.open
        if bar.high > self._session_high:
            self._session_high = bar.high

    def values(self) -> np.ndarray:
        if not self._open > 0:
            return np.array([math.nan, math.nan])
        close = self.state.minutes.last_close
        return np.array([close / self._open - 1.0, self._session_high / self._open - 1.0])


@register
class Candle(Feature):
    """Shape of the most recent bar: body and wick fractions of its range."""

    name = "candle"
    columns = ("body", "upper_wick", "lower_wick")

    def values(self) -> np.ndarray:
        i = self.state.minutes.n - 1
        if i < 0:
            return np.array([math.nan, math.nan, math.nan])
        ring = self.state.minutes
        high = ring.high[i]
        low = ring.low[i]
        open_ = ring.open[i]
        close = ring.close[i]
        rng = high - low
        if rng <= 0:
            return np.array([0.0, 0.0, 0.0])
        body = abs(close - open_) / rng
        upper = (high - max(open_, close)) / rng
        lower = (min(open_, close) - low) / rng
        return np.array([body, upper, lower])
