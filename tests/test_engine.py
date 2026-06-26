"""End-to-end engine/driver tests over the simulation source, including no-lookahead."""

from __future__ import annotations

import numpy as np

from quantzero.driver import EngineDriver
from quantzero.engine import FeatureEngine
from quantzero.events import MinuteBar
from quantzero.features import default_features
from quantzero.sources.simulation import SimulationConfig, SimulationSource


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
    # After a full session almost everything should be defined.
    assert n_valid >= last.values.shape[0] - 2


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
