"""Cache primitives validated against numpy ground truth."""

from __future__ import annotations

import math

import numpy as np

from quantzero.caches import (
    Ewma,
    RollingArray,
    RollingMax,
    RollingMin,
    RollingMoments,
    RollingSum,
    RunningExtrema,
    SessionSum,
    Welford,
)


def test_rolling_sum_matches_window_mean() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=200)
    window = 17
    cache = RollingSum(window)
    for i, value in enumerate(data):
        cache.push(value)
        expected = data[max(0, i - window + 1) : i + 1]
        assert math.isclose(cache.sum, expected.sum(), rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(cache.mean, expected.mean(), rel_tol=1e-9, abs_tol=1e-9)


def test_rolling_moments_match_numpy_std() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=200)
    window = 20
    cache = RollingMoments(window)
    for i, value in enumerate(data):
        cache.push(value)
        if i + 1 >= window:
            expected = data[i - window + 1 : i + 1]
            assert math.isclose(cache.mean, expected.mean(), rel_tol=1e-9, abs_tol=1e-9)
            assert math.isclose(cache.std, expected.std(), rel_tol=1e-7, abs_tol=1e-7)


def test_rolling_extrema_match_numpy() -> None:
    rng = np.random.default_rng(2)
    data = rng.normal(size=150)
    window = 11
    high = RollingMax(window)
    low = RollingMin(window)
    for i, value in enumerate(data):
        high.push(value)
        low.push(value)
        expected = data[max(0, i - window + 1) : i + 1]
        assert high.value == expected.max()
        assert low.value == expected.min()


def test_ewma_recurrence() -> None:
    cache = Ewma(span=9)
    alpha = 2.0 / (9 + 1)
    value = math.nan
    for x in [1.0, 2.0, 3.0, 2.5, 4.0]:
        cache.push(x)
        value = x if value != value else value + alpha * (x - value)
        assert math.isclose(cache.value, value, rel_tol=1e-12)


def test_welford_matches_numpy() -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(size=100)
    cache = Welford()
    for value in data:
        cache.push(value)
    assert math.isclose(cache.mean, data.mean(), rel_tol=1e-9)
    assert math.isclose(cache.var, data.var(), rel_tol=1e-7)


def test_session_sum_and_extrema() -> None:
    accumulator = SessionSum()
    extrema = RunningExtrema()
    for value in [3.0, 1.0, 4.0, 1.5]:
        accumulator.push(value)
        extrema.push(value)
    assert accumulator.sum == 9.5
    assert accumulator.count == 4
    assert extrema.min == 1.0
    assert extrema.max == 4.0


def test_rolling_array_order() -> None:
    cache = RollingArray(3)
    for value in [1.0, 2.0, 3.0, 4.0]:
        cache.push(value)
    assert list(cache.values()) == [2.0, 3.0, 4.0]
