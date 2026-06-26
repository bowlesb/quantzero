"""Calendar / session-clock features derived purely from the bar timestamp."""

from __future__ import annotations

import numpy as np

from quantzero.clock import RTH_CLOSE_MINUTE, RTH_OPEN_MINUTE, minutes_since_open, minutes_to_close
from quantzero.feature import Feature, register

RTH_LENGTH_MINUTES = RTH_CLOSE_MINUTE - RTH_OPEN_MINUTE


@register
class SessionClock(Feature):
    """Where the current bar sits in the trading day (normalized to the regular session)."""

    name = "sessionclock"
    columns = ("since_open", "to_close", "progress")

    def values(self) -> np.ndarray:
        ts_ns = self.state.minutes.last_ts_ns
        since_open = minutes_since_open(ts_ns)
        to_close = minutes_to_close(ts_ns)
        progress = since_open / RTH_LENGTH_MINUTES
        return np.array([float(since_open), float(to_close), progress])
