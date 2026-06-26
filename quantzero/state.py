"""Per-ticker, per-session state shared by every feature in a process.

The minute ring is a day-bounded preallocated set of numpy arrays. A trading session
(including extended hours, 04:00-20:00 ET) is at most 960 minutes, so a fixed capacity
gives contiguous, allocation-free slices and O(1) appends without a circular wrap. It is
reset at each session rollover.
"""

from __future__ import annotations

import numpy as np

from quantzero.events import MinuteBar, Quote, Trade

DEFAULT_MINUTE_CAPACITY = 1600


class MinuteRing:
    """Contiguous day buffer of minute bars. ``n`` is the count appended this session."""

    def __init__(self, capacity: int = DEFAULT_MINUTE_CAPACITY) -> None:
        self.capacity = capacity
        self.ts_ns = np.zeros(capacity, dtype=np.int64)
        self.open = np.zeros(capacity, dtype=np.float64)
        self.high = np.zeros(capacity, dtype=np.float64)
        self.low = np.zeros(capacity, dtype=np.float64)
        self.close = np.zeros(capacity, dtype=np.float64)
        self.volume = np.zeros(capacity, dtype=np.float64)
        self.trade_count = np.zeros(capacity, dtype=np.int64)
        self.vwap = np.zeros(capacity, dtype=np.float64)
        self.n = 0

    def append(self, bar: MinuteBar) -> None:
        if self.n >= self.capacity:
            raise RuntimeError(f"MinuteRing overflow at capacity {self.capacity}")
        i = self.n
        self.ts_ns[i] = bar.ts_ns
        self.open[i] = bar.open
        self.high[i] = bar.high
        self.low[i] = bar.low
        self.close[i] = bar.close
        self.volume[i] = bar.volume
        self.trade_count[i] = bar.trade_count
        self.vwap[i] = bar.vwap
        self.n += 1

    def reset(self) -> None:
        self.n = 0

    def closes(self) -> np.ndarray:
        return self.close[: self.n]

    def highs(self) -> np.ndarray:
        return self.high[: self.n]

    def lows(self) -> np.ndarray:
        return self.low[: self.n]

    def volumes(self) -> np.ndarray:
        return self.volume[: self.n]

    @property
    def last_close(self) -> float:
        return float(self.close[self.n - 1]) if self.n else float("nan")


class TickerState:
    """Everything a process knows about one ticker for the current session."""

    def __init__(self, ticker: str, minute_capacity: int = DEFAULT_MINUTE_CAPACITY) -> None:
        self.ticker = ticker
        self.minutes = MinuteRing(minute_capacity)
        self.last_quote: Quote | None = None
        self.last_trade: Trade | None = None
        self.session_date: int | None = None

    def reset(self, session_date: int) -> None:
        self.minutes.reset()
        self.last_quote = None
        self.last_trade = None
        self.session_date = session_date

    def on_quote(self, quote: Quote) -> None:
        self.last_quote = quote

    def on_trade(self, trade: Trade) -> None:
        self.last_trade = trade

    def on_minute(self, bar: MinuteBar) -> None:
        self.minutes.append(bar)
