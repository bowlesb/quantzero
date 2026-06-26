"""Raw store + two-stage backfill (raw -> features), no network."""

from __future__ import annotations

import datetime as dt

from quantzero.backfill import backfill_features
from quantzero.raw_store import RawReplaySource, RawStore
from quantzero.store import day_ns_bounds, read_features
from tests.helpers import make_bar

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
