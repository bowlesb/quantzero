"""Per-ticker, per-session state shared by every feature in a process.

We only ever hold the *current session* in memory. New minute bars are appended to a
day buffer that grows as the session fills and is cleared at the next session rollover —
nothing is ever overwritten (this is an accumulator, not a circular ring buffer). The
backing numpy arrays grow by doubling, so memory tracks the actual bar count rather than a
fixed worst case (a session is at most ~960 minutes including extended hours).

Trades and quotes are not buffered tick-by-tick: features digest them incrementally into
their own caches as they arrive, and ``TickerState`` keeps only the latest of each for
features that need the current quote/trade.
"""

from __future__ import annotations

import numpy as np

from quantzero.events import MinuteBar, Quote, Trade

DEFAULT_INITIAL_CAPACITY = 1024


def _grown(array: np.ndarray, capacity: int) -> np.ndarray:
    """Return a larger copy of ``array`` with the existing values preserved at the front."""
    out = np.zeros(capacity, dtype=array.dtype)
    out[: array.shape[0]] = array
    return out


class MinuteBuffer:
    """Day-scoped, append-only buffer of minute bars. ``n`` is this session's bar count.

    Fields are contiguous numpy arrays; ``arr[:n]`` is the session so far, so features can
    slice recent history directly.
    """

    def __init__(self, initial_capacity: int = DEFAULT_INITIAL_CAPACITY) -> None:
        self._capacity = max(initial_capacity, 1)
        self.ts_ns = np.zeros(self._capacity, dtype=np.int64)
        self.open = np.zeros(self._capacity, dtype=np.float64)
        self.high = np.zeros(self._capacity, dtype=np.float64)
        self.low = np.zeros(self._capacity, dtype=np.float64)
        self.close = np.zeros(self._capacity, dtype=np.float64)
        self.volume = np.zeros(self._capacity, dtype=np.float64)
        self.trade_count = np.zeros(self._capacity, dtype=np.int64)
        self.vwap = np.zeros(self._capacity, dtype=np.float64)
        self.n = 0

    def _grow(self) -> None:
        self._capacity *= 2
        self.ts_ns = _grown(self.ts_ns, self._capacity)
        self.open = _grown(self.open, self._capacity)
        self.high = _grown(self.high, self._capacity)
        self.low = _grown(self.low, self._capacity)
        self.close = _grown(self.close, self._capacity)
        self.volume = _grown(self.volume, self._capacity)
        self.trade_count = _grown(self.trade_count, self._capacity)
        self.vwap = _grown(self.vwap, self._capacity)

    def append(self, bar: MinuteBar) -> None:
        if self.n == self._capacity:
            self._grow()
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
        # Keep the allocation; reuse it next session to avoid per-day reallocation churn.
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

    def __init__(self, ticker: str, minute_capacity: int = DEFAULT_INITIAL_CAPACITY) -> None:
        self.ticker = ticker
        self.minutes = MinuteBuffer(minute_capacity)
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
