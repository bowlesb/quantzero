"""Volume / VWAP features."""

from __future__ import annotations

import math

import numpy as np

from quantzero.caches import RollingMoments, SessionSum
from quantzero.feature import Feature, register

REL_VOLUME_WINDOW = 20
TRADE_COUNT_WINDOW = 15


@register
class VwapDistance(Feature):
    """Session VWAP (typical-price weighted) and the close's distance from it."""

    name = "vwap"
    columns = ("level", "dist")

    def setup(self) -> None:
        self._pv = SessionSum()
        self._vol = SessionSum()

    def on_minute(self) -> None:
        minutes = self.state.minutes
        typical = (minutes.last_high + minutes.last_low + minutes.last_close) / 3.0
        volume = minutes.last_volume
        self._pv.push(typical * volume)
        self._vol.push(volume)

    def values(self) -> np.ndarray:
        if self._vol.sum <= 0:
            return np.array([math.nan, math.nan])
        vwap = self._pv.sum / self._vol.sum
        close = self.state.minutes.last_close
        return np.array([vwap, close / vwap - 1.0 if vwap > 0 else math.nan])


@register
class VolumeProfile(Feature):
    """Relative volume vs a rolling mean, a volume z-score, and cumulative session volume."""

    name = "volprofile"
    columns = ("rel_vol", "vol_z", "cum_vol")

    def setup(self) -> None:
        self._moments = RollingMoments(REL_VOLUME_WINDOW)
        self._cum = SessionSum()

    def on_minute(self) -> None:
        volume = self.state.minutes.last_volume
        self._moments.push(volume)
        self._cum.push(volume)

    def values(self) -> np.ndarray:
        volume = self.state.minutes.last_volume
        mean = self._moments.mean
        rel = volume / mean if mean and mean > 0 else math.nan
        std = self._moments.std
        z = (volume - mean) / std if std and std > 0 else math.nan
        return np.array([rel, z, self._cum.sum])


@register
class TradeIntensity(Feature):
    """Rolling mean and z-score of per-bar trade count."""

    name = "tradeintensity"
    columns = ("tc_mean", "tc_z")

    def setup(self) -> None:
        self._moments = RollingMoments(TRADE_COUNT_WINDOW)

    def on_minute(self) -> None:
        self._moments.push(float(self.state.minutes.last_trade_count))

    def values(self) -> np.ndarray:
        mean = self._moments.mean
        std = self._moments.std
        tc = float(self.state.minutes.last_trade_count)
        z = (tc - mean) / std if std and std > 0 else math.nan
        return np.array([mean, z])
