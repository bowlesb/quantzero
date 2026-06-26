"""Trade/quote-driven microstructure features over the sequential multi-stream path."""

from __future__ import annotations

import numpy as np

from quantzero.driver import EngineDriver
from quantzero.features import default_features
from quantzero.sources.simulation import SimulationConfig, SimulationSource


def _last_vector(with_ticks: bool):
    config = SimulationConfig(tickers=["AAA"], n_minutes=40, seed=2, with_ticks=with_ticks)
    vectors = list(EngineDriver(["AAA"], default_features()).run_source(SimulationSource(config)))
    last = vectors[-1]
    return last, {c: i for i, c in enumerate(last.columns)}, config


def test_microstructure_populates_with_ticks() -> None:
    last, idx, config = _last_vector(with_ticks=True)
    # per-minute counts match the simulator's tick rates
    assert last.values[idx["trademicro_n_trades"]] == config.trades_per_minute
    assert last.values[idx["quotemicro_n_quotes"]] == config.quotes_per_minute
    # the genuinely tick-derived stats are defined
    for col in (
        "trademicro_signed_frac",
        "trademicro_vwap_dist",
        "quotemicro_twa_spread_bps",
        "quotemicro_twa_imbalance",
        "flowfreq_trade_quote_ratio",
    ):
        assert not np.isnan(last.values[idx[col]]), col
    # trade/quote ratio == trades-per-min / quotes-per-min
    ratio = config.trades_per_minute / config.quotes_per_minute
    assert abs(last.values[idx["flowfreq_trade_quote_ratio"]] - ratio) < 1e-9


def test_microstructure_nan_without_ticks() -> None:
    last, idx, _ = _last_vector(with_ticks=False)
    # no trades/quotes -> tick-driven stats are NaN; counts are zero
    assert last.values[idx["trademicro_n_trades"]] == 0.0
    assert np.isnan(last.values[idx["trademicro_signed_frac"]])
    assert np.isnan(last.values[idx["quotemicro_twa_spread_bps"]])
    assert np.isnan(last.values[idx["flowfreq_trade_quote_ratio"]])
