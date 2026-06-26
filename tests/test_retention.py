"""Raw-store retention policy: quotes pruned sooner than bars/trades."""

from __future__ import annotations

import datetime as dt

from quantzero.events import Quote, Trade
from quantzero.raw_store import RawStore
from quantzero.retention import prune_raw
from tests.helpers import make_bar

TODAY = dt.date(2026, 6, 26)
OLD = "2025-06-01"  # > 180 days ago
MID = "2026-05-20"  # > 35 days but < 180 days ago
RECENT = "2026-06-20"  # within all windows


def _seed(root) -> None:
    store = RawStore(root)
    for day in (OLD, MID, RECENT):
        store.write_bars(day, "AAA", [make_bar("AAA", 0, 100.0)])
        store.write_trades(day, "AAA", [Trade("AAA", 1, 100.0, 1.0)])
        store.write_quotes(day, "AAA", [Quote("AAA", 1, 99.0, 101.0, 1.0, 1.0)])


def test_prune_policy(tmp_path) -> None:
    _seed(tmp_path)
    result = prune_raw(tmp_path, TODAY, quotes_days=35, trades_days=180, bars_days=180)
    assert result == {
        "quotes": 2,
        "trades": 1,
        "bars": 1,
    }  # quotes drop OLD+MID; bars/trades drop OLD

    store = RawStore(tmp_path)
    assert not store.has_quotes(OLD, "AAA") and not store.has_quotes(MID, "AAA")
    assert store.has_quotes(RECENT, "AAA")
    assert store.has_trades(MID, "AAA") and store.has_bars(MID, "AAA")  # kept (< 180d)
    assert not store.has_trades(OLD, "AAA") and not store.has_bars(OLD, "AAA")


def test_dry_run_deletes_nothing(tmp_path) -> None:
    _seed(tmp_path)
    result = prune_raw(tmp_path, TODAY, dry_run=True)
    assert result["quotes"] == 2
    assert RawStore(tmp_path).has_quotes(OLD, "AAA")  # still there
