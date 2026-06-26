"""Volatility features: rolling return std, realized vol, ATR, Bollinger bands."""

from __future__ import annotations

import math

import numpy as np

from quantzero.caches import RollingMoments, RollingSum
from quantzero.events import MinuteBar
from quantzero.feature import Feature, register

VOL_WINDOWS = (5, 15, 30)
ATR_WINDOWS = (5, 14)
BOLLINGER_WINDOW = 20
BOLLINGER_K = 2.0


@register
class RollingVolatility(Feature):
    """Std of one-minute returns over rolling windows, kept O(1) via running moments."""

    name = "vol"
    columns = tuple(f"std_{w}" for w in VOL_WINDOWS)

    def setup(self) -> None:
        self._moments = {w: RollingMoments(w) for w in VOL_WINDOWS}
        self._prev_close = math.nan

    def on_minute(self, bar: MinuteBar) -> None:
        if self._prev_close > 0:
            ret = bar.close / self._prev_close - 1.0
            for moments in self._moments.values():
                moments.push(ret)
        self._prev_close = bar.close

    def values(self) -> np.ndarray:
        return np.array([self._moments[w].std for w in VOL_WINDOWS])


@register
class RealizedVolatility(Feature):
    """Realized vol = sqrt(sum of squared returns) over rolling windows."""

    name = "rvol"
    columns = tuple(f"rv_{w}" for w in VOL_WINDOWS)

    def setup(self) -> None:
        self._sumsq = {w: RollingSum(w) for w in VOL_WINDOWS}
        self._prev_close = math.nan

    def on_minute(self, bar: MinuteBar) -> None:
        if self._prev_close > 0:
            ret = bar.close / self._prev_close - 1.0
            squared = ret * ret
            for cache in self._sumsq.values():
                cache.push(squared)
        self._prev_close = bar.close

    def values(self) -> np.ndarray:
        return np.array(
            [
                math.sqrt(self._sumsq[w].sum) if self._sumsq[w].count else math.nan
                for w in VOL_WINDOWS
            ]
        )


@register
class Atr(Feature):
    """Average True Range (SMA of true range) over short windows."""

    name = "atr"
    columns = tuple(f"atr_{w}" for w in ATR_WINDOWS)

    def setup(self) -> None:
        self._tr = {w: RollingSum(w) for w in ATR_WINDOWS}
        self._prev_close = math.nan

    def on_minute(self, bar: MinuteBar) -> None:
        if self._prev_close == self._prev_close:
            true_range = max(
                bar.high - bar.low,
                abs(bar.high - self._prev_close),
                abs(bar.low - self._prev_close),
            )
        else:
            true_range = bar.high - bar.low
        for cache in self._tr.values():
            cache.push(true_range)
        self._prev_close = bar.close

    def values(self) -> np.ndarray:
        return np.array([self._tr[w].mean for w in ATR_WINDOWS])


@register
class Bollinger(Feature):
    """%B (position within the bands) and bandwidth, from a 20-bar mean/std."""

    name = "boll"
    columns = ("pct_b", "bandwidth")

    def setup(self) -> None:
        self._moments = RollingMoments(BOLLINGER_WINDOW)

    def on_minute(self, bar: MinuteBar) -> None:
        self._moments.push(bar.close)

    def values(self) -> np.ndarray:
        if not self._moments.full:
            return np.array([math.nan, math.nan])
        mean = self._moments.mean
        std = self._moments.std
        if std <= 0 or mean <= 0:
            return np.array([0.5, 0.0])
        close = self.state.minutes.last_close
        lower = mean - BOLLINGER_K * std
        width = 2.0 * BOLLINGER_K * std
        return np.array([(close - lower) / width, width / mean])
