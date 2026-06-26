"""Unbounded (session-scoped) incremental caches: EWMA, Welford, accumulators, extrema."""

from __future__ import annotations

import math


class Ewma:
    """Exponentially-weighted moving average. ``span`` maps to alpha = 2/(span+1)."""

    def __init__(self, span: float) -> None:
        if span <= 0:
            raise ValueError("span must be > 0")
        self.alpha = 2.0 / (span + 1.0)
        self._value = math.nan
        self._initialized = False

    def push(self, value: float) -> None:
        if not self._initialized:
            self._value = value
            self._initialized = True
        else:
            self._value += self.alpha * (value - self._value)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def value(self) -> float:
        return self._value


class Welford:
    """Online mean/variance over an unbounded stream (Welford's algorithm)."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def push(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)

    @property
    def var(self) -> float:
        return self._m2 / self.count if self.count >= 2 else math.nan

    @property
    def std(self) -> float:
        variance = self.var
        return (
            math.sqrt(variance)
            if variance == variance and variance > 0
            else (0.0 if self.count >= 2 else math.nan)
        )


class SessionSum:
    """Running sum and count over the whole session."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def push(self, value: float) -> None:
        self.sum += value
        self.count += 1

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else math.nan


class RunningExtrema:
    """Session-wide running min and max."""

    def __init__(self) -> None:
        self.min = math.inf
        self.max = -math.inf

    def push(self, value: float) -> None:
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value

    @property
    def initialized(self) -> bool:
        return self.max >= self.min
