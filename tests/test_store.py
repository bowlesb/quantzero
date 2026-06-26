"""Feature store: round-trip and source-transparent merge (backfill overrides stream)."""

from __future__ import annotations

import numpy as np

from quantzero.driver import EngineDriver
from quantzero.features import default_features
from quantzero.sources.simulation import SimulationConfig, SimulationSource
from quantzero.store import FeatureStore, read_features, settled_dates

SET_VERSION = "test"


def _vectors() -> list:
    config = SimulationConfig(tickers=["AAA"], n_minutes=20, seed=1)
    driver = EngineDriver(config.tickers, default_features())
    return list(driver.run_source(SimulationSource(config)))


def test_multi_ticker_same_minute_no_collision(tmp_path) -> None:
    """Many tickers share a minute's ts_ns; their files must not overwrite each other."""
    tickers = ["AAA", "BBB", "CCC"]
    config = SimulationConfig(tickers=tickers, n_minutes=10, seed=2)
    driver = EngineDriver(tickers, default_features())
    vectors = list(driver.run_source(SimulationSource(config)))
    FeatureStore(tmp_path, SET_VERSION, "sim").write_many(vectors)

    start = min(v.ts_ns for v in vectors)
    end = max(v.ts_ns for v in vectors) + 1
    frame = read_features(tmp_path, SET_VERSION, start, end, source="auto", provisional="sim")
    assert frame.height == len(tickers) * config.n_minutes
    assert set(frame["ticker"].unique().to_list()) == set(tickers)


def test_write_and_read_roundtrip(tmp_path) -> None:
    vectors = _vectors()
    FeatureStore(tmp_path, SET_VERSION, "sim").write_many(vectors)
    start = min(v.ts_ns for v in vectors)
    end = max(v.ts_ns for v in vectors) + 1
    frame = read_features(tmp_path, SET_VERSION, start, end, source="auto", provisional="sim")
    assert frame.height == len(vectors)
    assert "returns_r_1" in frame.columns
    assert frame["ticker"].unique().to_list() == ["AAA"]


def test_backfill_overrides_stream(tmp_path) -> None:
    vectors = _vectors()
    # Stream/live writes first (provisional).
    FeatureStore(tmp_path, SET_VERSION, "stream").write_many(vectors)
    # Backfill later writes a differing value for the same (ticker, minute) keys.
    store = FeatureStore(tmp_path, SET_VERSION, "backfill")
    for vector in vectors:
        vector.values[vector.columns.index("returns_r_1")] = 42.0
        store.write(vector)

    start = min(v.ts_ns for v in vectors)
    end = max(v.ts_ns for v in vectors) + 1
    merged = read_features(tmp_path, SET_VERSION, start, end, source="auto", provisional="stream")
    # Backfill is truth: every row should show the backfilled value.
    assert np.allclose(merged["returns_r_1"].to_numpy(), 42.0)
    assert settled_dates(tmp_path, SET_VERSION)


def test_names_projection(tmp_path) -> None:
    vectors = _vectors()
    FeatureStore(tmp_path, SET_VERSION, "sim").write_many(vectors)
    start = min(v.ts_ns for v in vectors)
    end = max(v.ts_ns for v in vectors) + 1
    frame = read_features(
        tmp_path, SET_VERSION, start, end, names=["rsi_rsi_14"], source="auto", provisional="sim"
    )
    assert set(frame.columns) == {"ticker", "ts_ns", "rsi_rsi_14"}
