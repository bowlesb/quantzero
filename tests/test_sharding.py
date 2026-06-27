"""Sharding: stable hashing, full coverage, parity with a single driver, and a spawn smoke test."""

from __future__ import annotations

import numpy as np

from quantzero.driver import EngineDriver
from quantzero.features import default_features
from quantzero.sharding import ShardedRouter, ShardedRunner, assign_tickers, worker_for
from quantzero.sources.simulation import SimulationConfig, SimulationSource
from quantzero.store import StoreConfig, read_features

TICKERS = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "META", "GOOGL"]


def test_worker_for_is_stable_and_in_range() -> None:
    for ticker in TICKERS:
        worker = worker_for(ticker, 4)
        assert 0 <= worker < 4
        assert worker == worker_for(ticker, 4)  # deterministic


def test_assign_covers_all_tickers_disjointly() -> None:
    assignment = assign_tickers(TICKERS, 4)
    flattened = [t for group in assignment.values() for t in group]
    assert sorted(flattened) == sorted(TICKERS)
    assert len(flattened) == len(set(flattened))  # no ticker in two shards


def test_router_parity_with_single_driver() -> None:
    """Each ticker computes the same vector whether sharded or not (pure partitioning)."""
    config = SimulationConfig(tickers=TICKERS, n_minutes=40, seed=5)

    single = {t: EngineDriver([t], default_features()) for t in TICKERS}
    router = ShardedRouter(TICKERS, n_workers=4, feature_classes=default_features())

    for event in SimulationSource(config).iter_events():
        sharded_vector = router.process(event)
        baseline_vector = single[event.ticker].process(event)
        assert (sharded_vector is None) == (baseline_vector is None)
        if sharded_vector is not None and baseline_vector is not None:
            np.testing.assert_array_equal(sharded_vector.values, baseline_vector.values)


def test_multiprocess_runner_smoke() -> None:
    config = SimulationConfig(tickers=TICKERS, n_minutes=30, seed=3)
    runner = ShardedRunner(TICKERS, n_workers=3)
    summaries = runner.run_source(SimulationSource(config))

    assert len(summaries) == config.n_minutes * len(TICKERS)
    assert {s.ticker for s in summaries} == set(TICKERS)
    # each ticker landed on exactly the worker its hash assigns
    for summary in summaries:
        assert summary.worker_id == worker_for(summary.ticker, 3)


def test_workers_write_to_store_async(tmp_path) -> None:
    """Workers persist every computed vector via their async StoreWriter (across processes)."""
    config = SimulationConfig(tickers=TICKERS, n_minutes=20, seed=4)
    store_config = StoreConfig(str(tmp_path), "test", "stream")
    runner = ShardedRunner(TICKERS, n_workers=4, store_config=store_config)
    summaries = runner.run_source(SimulationSource(config))

    events = list(SimulationSource(config).iter_events())
    start = min(e.ts_ns for e in events)
    end = max(e.ts_ns for e in events) + 1
    frame = read_features(tmp_path, "test", start, end, source="auto", provisional="stream")
    assert frame.height == len(summaries) == config.n_minutes * len(TICKERS)
    assert set(frame["ticker"].unique().to_list()) == set(TICKERS)
