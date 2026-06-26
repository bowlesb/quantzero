"""Latency regression guard: every feature must stay extremely fast.

Uses a small scenario so it runs in well under a second, but still exercises the warmed
steady state. The budget is generous (catches a regression of ~10x), not a hard SLA.
"""

from __future__ import annotations

import numpy as np

from quantzero.features import default_features
from quantzero.features.rangepos import Rsi
from quantzero.latency import (
    LatencyScenario,
    LatencyStats,
    measure_all,
    measure_baseline,
    measure_feature,
)

# Per-feature marginal cost budget. Features measure ~0.3-2us here; 12us is a wide guard.
MARGINAL_BUDGET_NS = 12_000.0

_FAST_SCENARIO = LatencyScenario(warmup_minutes=40, measure_minutes=400, seed=1)


def test_latency_stats_from_samples() -> None:
    stats = LatencyStats.from_samples(np.array([10.0, 20.0, 30.0, 40.0]))
    assert stats.n == 4
    assert stats.mean_ns == 25.0
    assert stats.max_ns == 40.0


def test_single_feature_measure_shape() -> None:
    baseline = measure_baseline(_FAST_SCENARIO)
    result = measure_feature(Rsi, _FAST_SCENARIO, baseline)
    assert result.name == "rsi"
    assert result.columns == 1
    assert result.per_minute.n == _FAST_SCENARIO.measure_minutes
    assert result.values_only.mean_ns > 0


def test_all_features_within_budget() -> None:
    results = measure_all(default_features(), _FAST_SCENARIO)
    assert len(results) == len(default_features())
    slow = [r.name for r in results if r.marginal_mean_ns > MARGINAL_BUDGET_NS]
    assert not slow, f"features over {MARGINAL_BUDGET_NS / 1000:.0f}us marginal budget: {slow}"
