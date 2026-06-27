"""End-to-end engine/driver tests over the simulation source, including no-lookahead."""

from __future__ import annotations

import datetime as dt

import numpy as np

from quantzero.clock import ET, to_ns
from quantzero.driver import EngineDriver
from quantzero.engine import FeatureEngine
from quantzero.events import MinuteBar
from quantzero.features import default_features
from quantzero.sources.simulation import SimulationConfig, SimulationSource

_NS_PER_MINUTE = 60_000_000_000


def _day_bars(date: dt.date, closes: list[float]) -> list[MinuteBar]:
    open_ns = to_ns(dt.datetime(date.year, date.month, date.day, 9, 30, tzinfo=ET))
    return [
        MinuteBar("T", open_ns + m * _NS_PER_MINUTE, c, c, c, c, 1000.0, 50, c)
        for m, c in enumerate(closes)
    ]


def _config() -> SimulationConfig:
    return SimulationConfig(tickers=["AAA", "BBB"], n_minutes=60, seed=11)


def test_driver_emits_one_vector_per_bar() -> None:
    config = _config()
    driver = EngineDriver(config.tickers, default_features())
    vectors = list(driver.run_source(SimulationSource(config)))
    assert len(vectors) == config.n_minutes * len(config.tickers)
    width = len(FeatureEngine("AAA", default_features()).columns)
    assert all(v.values.shape == (width,) for v in vectors)
    assert all(v.compute_ns > 0 for v in vectors)


def test_features_warm_up() -> None:
    config = _config()
    driver = EngineDriver(config.tickers, default_features())
    vectors = [v for v in driver.run_source(SimulationSource(config)) if v.ticker == "AAA"]
    last = vectors[-1]
    n_valid = int(np.count_nonzero(~np.isnan(last.values)))
    # After a full session almost everything should be defined. The simulation feeds a
    # CONSTANT per-minute tick rate, so flowfreq's rate z-scores stay NaN (std 0); they
    # populate on real data where the rate varies. Allow a few such NaNs.
    assert n_valid >= last.values.shape[0] - 4


def test_determinism() -> None:
    config = _config()
    run_a = [
        v.values
        for v in EngineDriver(config.tickers, default_features()).run_source(
            SimulationSource(config)
        )
    ]
    run_b = [
        v.values
        for v in EngineDriver(config.tickers, default_features()).run_source(
            SimulationSource(config)
        )
    ]
    for a, b in zip(run_a, run_b, strict=True):
        np.testing.assert_array_equal(a, b)


def test_no_lookahead_point_in_time() -> None:
    """A vector at bar k must depend only on events up to bar k.

    Truncating the stream right after the k-th bar of a ticker and recomputing must yield
    the identical vector. This is the property that makes historical replay == live.
    """
    config = _config()
    events = list(SimulationSource(config).iter_events())
    full = list(
        EngineDriver(config.tickers, default_features()).run_source(SimulationSource(config))
    )
    aaa_vectors = [v for v in full if v.ticker == "AAA"]

    bar_positions = [
        i for i, e in enumerate(events) if isinstance(e, MinuteBar) and e.ticker == "AAA"
    ]
    for k in (10, 25, 59):
        prefix = events[: bar_positions[k] + 1]
        driver = EngineDriver(config.tickers, default_features())
        truncated_last = None
        for event in prefix:
            vector = driver.process(event)
            if vector is not None and vector.ticker == "AAA":
                truncated_last = vector
        assert truncated_last is not None
        np.testing.assert_array_equal(truncated_last.values, aaa_vectors[k].values)


def test_each_day_starts_fresh_like_live() -> None:
    """A day's vectors must be identical whether or not the prior day was processed first.

    Continuous live resets in place at the ET-day boundary (_ensure_session rebuilds fresh
    Feature instances); backfill uses a fresh engine per (ticker, day). This proves those are
    the same thing — the daily reset leaks nothing from the prior day.
    """
    feats = default_features()
    rng = np.random.default_rng(9)
    day1 = _day_bars(dt.date(2026, 6, 24), list(100.0 * np.cumprod(1.0 + rng.normal(0, 0.001, 70))))
    day2 = _day_bars(dt.date(2026, 6, 25), list(120.0 * np.cumprod(1.0 + rng.normal(0, 0.001, 70))))

    # Engine that saw day1 then day2 (in-place rollover) — i.e. continuous live.
    rolled = FeatureEngine("T", feats)
    for bar in day1:
        rolled.on_minute(bar)
    rolled_day2 = [rolled.on_minute(bar).values.copy() for bar in day2]

    # Fresh engine that saw only day2 — i.e. a per-day backfill job.
    fresh = FeatureEngine("T", feats)
    fresh_day2 = [fresh.on_minute(bar).values.copy() for bar in day2]

    for a, b in zip(rolled_day2, fresh_day2, strict=True):
        np.testing.assert_array_equal(a, b)
