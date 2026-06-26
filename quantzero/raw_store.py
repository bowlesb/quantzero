"""Raw market-data store — the substrate backfill is built on.

Clear separation of concerns:

  * **Raw ingestion is the ONLY thing that talks to Alpaca for backfill.** It lands raw
    minute bars on disk, partitioned by date, one parquet per ticker-day.
  * **Feature backfill reads the raw store, never Alpaca.** It replays the raw bars through
    the same engine the live stream uses, so backfilled features match live by construction.

Layout::

    <raw_root>/bars/date=<YYYY-MM-DD>/<ticker>.parquet   # raw OHLCV minute bars, no features
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl

from quantzero.events import Event, MinuteBar

_BAR_COLUMNS = ("ts_ns", "open", "high", "low", "close", "volume", "trade_count", "vwap")


class RawStore:
    """Reads/writes raw minute bars, one parquet per (date, ticker)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _bars_dir(self, day: str) -> Path:
        return self.root / "bars" / f"date={day}"

    def bars_path(self, day: str, ticker: str) -> Path:
        return self._bars_dir(day) / f"{ticker.replace('/', '_')}.parquet"

    def has_bars(self, day: str, ticker: str) -> bool:
        return self.bars_path(day, ticker).exists()

    def write_bars(self, day: str, ticker: str, bars: list[MinuteBar]) -> Path | None:
        if not bars:
            return None
        directory = self._bars_dir(day)
        directory.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(
            {
                "ts_ns": [b.ts_ns for b in bars],
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
                "trade_count": [b.trade_count for b in bars],
                "vwap": [b.vwap for b in bars],
            }
        )
        path = self.bars_path(day, ticker)
        tmp = path.with_suffix(".parquet.tmp")
        frame.write_parquet(tmp)
        tmp.replace(path)
        return path

    def read_bars(self, day: str, ticker: str) -> list[MinuteBar]:
        path = self.bars_path(day, ticker)
        if not path.exists():
            return []
        frame = pl.read_parquet(path, columns=list(_BAR_COLUMNS)).sort("ts_ns")
        return [
            MinuteBar(
                ticker=ticker,
                ts_ns=int(row["ts_ns"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                trade_count=int(row["trade_count"]),
                vwap=float(row["vwap"]),
            )
            for row in frame.iter_rows(named=True)
        ]

    def days(self) -> list[str]:
        base = self.root / "bars"
        if not base.exists():
            return []
        return sorted(p.name.removeprefix("date=") for p in base.glob("date=*"))

    def tickers(self, day: str) -> list[str]:
        directory = self._bars_dir(day)
        if not directory.exists():
            return []
        return sorted(p.stem for p in directory.glob("*.parquet"))


class RawReplaySource:
    """An :class:`EventSource` that replays raw bars from the raw store (no network)."""

    def __init__(self, tickers: list[str], day: str, root: str | Path) -> None:
        self.tickers = tickers
        self.day = day
        self.store = RawStore(root)

    def iter_events(self) -> Iterator[Event]:
        bars: list[MinuteBar] = []
        for ticker in self.tickers:
            bars.extend(self.store.read_bars(self.day, ticker))
        bars.sort(key=lambda b: b.ts_ns)
        yield from bars
