"""Pure-function universe logic (no network): ETF detection and selection/ranking."""

from __future__ import annotations

from quantzero.universe import SymbolStat, is_etf_like, select_universe


def test_is_etf_like() -> None:
    assert is_etf_like("ProShares UltraPro QQQ")
    assert is_etf_like("Direxion Daily 3X Bull")
    assert is_etf_like("SPDR S&P 500 ETF Trust")
    assert not is_etf_like("Apple Inc. Common Stock")
    assert not is_etf_like("NVIDIA Corporation")


def test_select_universe_thresholds_and_rank() -> None:
    stats = [
        SymbolStat("AAA", price=10.0, adv_dollar=50_000_000.0),
        SymbolStat("BBB", price=4.0, adv_dollar=99_000_000.0),  # below price floor
        SymbolStat("CCC", price=20.0, adv_dollar=5_000_000.0),  # below ADV floor
        SymbolStat("DDD", price=30.0, adv_dollar=80_000_000.0),
    ]
    chosen = select_universe(stats, max_symbols=10)
    assert [s.symbol for s in chosen] == ["DDD", "AAA"]  # ranked by ADV$ desc


def test_select_universe_caps_and_breaks_ties_by_symbol() -> None:
    stats = [
        SymbolStat("ZZZ", price=10.0, adv_dollar=20_000_000.0),
        SymbolStat("AAA", price=10.0, adv_dollar=20_000_000.0),
        SymbolStat("MMM", price=10.0, adv_dollar=20_000_000.0),
    ]
    chosen = select_universe(stats, max_symbols=2)
    assert [s.symbol for s in chosen] == ["AAA", "MMM"]
