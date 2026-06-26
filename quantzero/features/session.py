"""Calendar / session-clock features derived purely from the bar timestamp."""

from __future__ import annotations

import numpy as np

from quantzero.clock import RTH_CLOSE_MINUTE, RTH_OPEN_MINUTE, minutes_since_open, minutes_to_close
from quantzero.events import MinuteBar
from quantzero.feature import Feature, register

RTH_LENGTH_MINUTES = RTH_CLOSE_MINUTE - RTH_OPEN_MINUTE


@register
class SessionClock(Feature):
    """Where the current bar sits in the trading day (normalized to the regular session)."""

    name = "sessionclock"
    columns = ("since_open", "to_close", "progress")

    def setup(self) -> None:
        self._ts_ns = 0

    def on_minute(self, bar: MinuteBar) -> None:
        self._ts_ns = bar.ts_ns

    def values(self) -> np.ndarray:
        since_open = minutes_since_open(self._ts_ns)
        to_close = minutes_to_close(self._ts_ns)
        progress = since_open / RTH_LENGTH_MINUTES
        return np.array([float(since_open), float(to_close), progress])
