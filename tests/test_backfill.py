"""Raw store + two-stage backfill (raw -> features), no network."""

from __future__ import annotations

import datetime as dt

from quantzero.backfill import backfill_features
from quantzero.events import Quote, Trade
from quantzero.raw_store import RawReplaySource, RawStore
from quantzero.store import day_ns_bounds, read_features
from tests.helpers import make_bar

_NS_PER_MINUTE = 60_000_000_000

DAY = "2026-06-24"  # tests.helpers anchors bars to this ET date


def test_raw_store_roundtrip(tmp_path) -> None:
    bars = [make_bar("AAA", minute, 100.0 + minute) for minute in range(10)]
    store = RawStore(tmp_path)
    store.write_bars(DAY, "AAA", bars)
    assert store.has_bars(DAY, "AAA")
    assert store.days() == [DAY]
    assert store.tickers(DAY) == ["AAA"]

    back = store.read_bars(DAY, "AAA")
    assert len(back) == 10
    assert back[0].close == 100.0
    assert back[-1].ts_ns == bars[-1].ts_ns

    events = list(RawReplaySource(["AAA"], DAY, tmp_path).iter_events())
    assert [e.ts_ns for e in events] == [b.ts_ns for b in bars]


def test_raw_trades_quotes_roundtrip_and_merge_order(tmp_path) -> None:
    bars = [make_bar("AAA", minute, 100.0 + minute) for minute in range(2)]
    base = bars[0].ts_ns
    store = RawStore(tmp_path)
    store.write_bars(DAY, "AAA", bars)
    store.write_trades(
        DAY,
        "AAA",
        [
            Trade("AAA", base + 1000, 100.0, 5.0),
            Trade("AAA", base + _NS_PER_MINUTE + 1000, 101.0, 3.0),
        ],
    )
    store.write_quotes(
        DAY,
        "AAA",
        [
            Quote("AAA", base + 500, 99.9, 100.1, 10, 10),
            Quote("AAA", base + 600, 99.8, 100.2, 8, 12),
        ],
    )
    assert store.has_trades(DAY, "AAA") and store.has_quotes(DAY, "AAA")
    assert len(store.read_trades(DAY, "AAA")) == 2
    assert len(store.read_quotes(DAY, "AAA")) == 2

    events = list(RawReplaySource(["AAA"], DAY, tmp_path, with_ticks=True).iter_events())
    kinds = [type(e).__name__ for e in events]
    # minute 0: its two quotes and one trade, THEN the bar (bar ranks after ticks in its minute)
    assert kinds[:4] == ["Quote", "Quote", "Trade", "MinuteBar"]


def test_backfill_features_reads_raw_and_batches(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    feature_root = tmp_path / "feat"
    bars = [make_bar("AAA", minute, 100.0 + 0.1 * minute) for minute in range(30)]
    RawStore(raw_root).write_bars(DAY, "AAA", bars)

    n = backfill_features(
        ["AAA"], [dt.date(2026, 6, 24)], str(raw_root), str(feature_root), workers=1
    )
    assert n == 30

    start, end = day_ns_bounds(dt.date(2026, 6, 24))
    frame = read_features(feature_root, "0.1.0", start, end, source="backfill")
    assert frame.height == 30
    assert frame["ticker"].unique().to_list() == ["AAA"]

    # batched: exactly one parquet file for the ticker-day (not one per minute)
    files = list(feature_root.glob("v=*/source=backfill/date=*/*.parquet"))
    assert len(files) == 1
