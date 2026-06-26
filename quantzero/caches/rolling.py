"""Fixed-window incremental caches: O(1) rolling sum, moments, and extrema.

These are the workhorses that let a feature read a rolling statistic without scanning a
window every bar. Each is reset daily (the engine rebuilds features per session), so the
subtractive running sums never accumulate meaningful float drift within a day.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class RollingSum:
    """Sum/mean over the last ``window`` pushed values. O(1) push."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._buf = np.zeros(window, dtype=np.float64)
        self._idx = 0
        self._count = 0
        self.sum = 0.0

    def push(self, value: float) -> None:
        evicted = self._buf[self._idx]
        self._buf[self._idx] = value
        self.sum += value - evicted
        self._idx = (self._idx + 1) % self.window
        if self._count < self.window:
            self._count += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def full(self) -> bool:
        return self._count == self.window

    @property
    def mean(self) -> float:
        return self.sum / self._count if self._count else math.nan


class RollingMoments:
    """Mean / variance / std over the last ``window`` values via running sum & sumsq."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._buf = np.zeros(window, dtype=np.float64)
        self._idx = 0
        self._count = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def push(self, value: float) -> None:
        evicted = self._buf[self._idx]
        self._buf[self._idx] = value
        self.sum += value - evicted
        self.sumsq += value * value - evicted * evicted
        self._idx = (self._idx + 1) % self.window
        if self._count < self.window:
            self._count += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def full(self) -> bool:
        return self._count == self.window

    @property
    def mean(self) -> float:
        return self.sum / self._count if self._count else math.nan

    @property
    def var(self) -> float:
        """Population variance (clamped at 0 to absorb tiny negative float error)."""
        if self._count < 2:
            return math.nan
        value = self.sumsq / self._count - (self.sum / self._count) ** 2
        return value if value > 0.0 else 0.0

    @property
    def std(self) -> float:
        variance = self.var
        return math.sqrt(variance) if variance == variance else math.nan


class _MonotonicExtremum:
    """Shared monotonic-deque machinery for rolling min/max. Amortized O(1) push."""

    def __init__(self, window: int, keep_larger: bool) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._keep_larger = keep_larger
        self._deque: deque[tuple[int, float]] = deque()
        self._pushes = 0

    def push(self, value: float) -> None:
        index = self._pushes
        self._pushes += 1
        work = self._deque
        if self._keep_larger:
            while work and work[-1][1] <= value:
                work.pop()
        else:
            while work and work[-1][1] >= value:
                work.pop()
        work.append((index, value))
        oldest_allowed = index - self.window + 1
        while work and work[0][0] < oldest_allowed:
            work.popleft()

    @property
    def value(self) -> float:
        return self._deque[0][1] if self._deque else math.nan


class RollingMax(_MonotonicExtremum):
    """Maximum over the last ``window`` values."""

    def __init__(self, window: int) -> None:
        super().__init__(window, keep_larger=True)


class RollingMin(_MonotonicExtremum):
    """Minimum over the last ``window`` values."""

    def __init__(self, window: int) -> None:
        super().__init__(window, keep_larger=False)


class RollingArray:
    """Keeps the last ``window`` values as an ordered numpy view.

    Use only for features that genuinely need the array (e.g. a regression slope). Read
    is O(window); prefer the running-stat caches above when a scalar statistic suffices.
    """

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._buf = np.zeros(window, dtype=np.float64)
        self._idx = 0
        self._count = 0

    def push(self, value: float) -> None:
        self._buf[self._idx] = value
        self._idx = (self._idx + 1) % self.window
        if self._count < self.window:
            self._count += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def full(self) -> bool:
        return self._count == self.window

    def values(self) -> np.ndarray:
        """Oldest-to-newest ordered copy of the stored values."""
        if self._count < self.window:
            return self._buf[: self._count].copy()
        return np.concatenate((self._buf[self._idx :], self._buf[: self._idx]))
