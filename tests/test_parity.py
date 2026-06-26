"""Live-vs-backfill parity comparison logic."""

from __future__ import annotations

import datetime as dt

from quantzero.driver import EngineDriver
from quantzero.features import default_features
from quantzero.parity import compare_sources
from quantzero.sources.simulation import SimulationConfig, SimulationSource
from quantzero.store import FeatureStore

DAY = dt.date(2026, 6, 24)  # SimulationConfig default day


def _write_both(tmp_path, perturb: bool) -> int:
    config = SimulationConfig(tickers=["AAA", "BBB"], n_minutes=15, seed=3)
    vectors = list(
        EngineDriver(config.tickers, default_features()).run_source(SimulationSource(config))
    )
    FeatureStore(tmp_path, "test", "stream").write_many(vectors)
    if perturb:
        idx = vectors[0].columns.index("rsi_rsi_14")
        for vector in vectors:
            vector.values[idx] += 5.0
    FeatureStore(tmp_path, "test", "backfill").write_many(vectors)
    return len(vectors)


def test_identical_sources_fully_agree(tmp_path) -> None:
    n = _write_both(tmp_path, perturb=False)
    report = compare_sources(DAY, set_version="test", root=str(tmp_path))
    assert report.matched_keys == n
    assert report.only_a == 0 and report.only_b == 0
    assert report.cells_mismatch == 0
    assert report.cell_match_pct == 100.0


def test_perturbation_is_detected(tmp_path) -> None:
    n = _write_both(tmp_path, perturb=True)
    report = compare_sources(DAY, set_version="test", root=str(tmp_path))
    assert report.matched_keys == n
    # exactly the perturbed feature should mismatch, on every row
    assert report.worst
    feature, mismatches, _max_diff = report.worst[0]
    assert feature == "rsi_rsi_14"  # only the perturbed feature mismatches
    assert mismatches > 0  # not every row (RSI is NaN during warmup; NaN==NaN holds)
