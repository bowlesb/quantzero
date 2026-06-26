"""Correctness checks for individual feature groups against hand-computed values."""

from __future__ import annotations

import math

import numpy as np

from quantzero.engine import FeatureEngine
from quantzero.features.price import Candle, Returns
from quantzero.features.volatility import RollingVolatility
from tests.helpers import column_value, make_bar, run_closes


def test_returns_horizons() -> None:
    closes = [100.0, 101.0, 102.0, 100.0]
    vector = run_closes([Returns], closes)
    assert math.isclose(column_value(vector, "returns_r_1"), 100.0 / 102.0 - 1.0, rel_tol=1e-12)
    assert math.isclose(column_value(vector, "returns_r_2"), 100.0 / 101.0 - 1.0, rel_tol=1e-12)


def test_returns_nan_before_warmup() -> None:
    vector = run_closes([Returns], [100.0])
    assert math.isnan(column_value(vector, "returns_r_5"))


def test_candle_shape() -> None:
    engine = FeatureEngine("T", [Candle])
    bar = make_bar("T", 0, close=104.0, open_=100.0, high=106.0, low=99.0)
    vector = engine.on_minute(bar)
    rng = 106.0 - 99.0
    assert math.isclose(column_value(vector, "candle_body"), 4.0 / rng, rel_tol=1e-12)
    assert math.isclose(
        column_value(vector, "candle_upper_wick"), (106.0 - 104.0) / rng, rel_tol=1e-12
    )
    assert math.isclose(
        column_value(vector, "candle_lower_wick"), (100.0 - 99.0) / rng, rel_tol=1e-12
    )


def test_rolling_volatility_matches_numpy() -> None:
    rng = np.random.default_rng(5)
    closes = list(100.0 * np.cumprod(1.0 + rng.normal(0, 0.001, 40)))
    vector = run_closes([RollingVolatility], closes)
    returns = np.diff(closes) / np.asarray(closes)[:-1]
    expected = returns[-5:].std()
    assert math.isclose(column_value(vector, "vol_std_5"), expected, rel_tol=1e-6, abs_tol=1e-9)
