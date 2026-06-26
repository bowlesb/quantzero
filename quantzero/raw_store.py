"""Raw market-data store — the substrate backfill is built on.

Clear separation of concerns:

  * **Raw ingestion is the ONLY thing that talks to Alpaca for backfill.** It lands raw
    minute bars, trades, and quotes on disk, partitioned by date, one parquet per
    (kind, date, ticker).
  * **Feature backfill reads the raw store, never Alpaca.** :class:`RawReplaySource` merges
    the three streams into one time-ordered sequence (ticks before each minute's bar) and
    replays them through the same engine the live stream uses — so backfill exercises the
    exact same caches (including the trade/quote-driven ones) as real time.

Layout::

    <raw_root>/bars/date=<d>/<ticker>.parquet      # raw OHLCV minute bars
    <raw_root>/trades/date=<d>/<ticker>.parquet    # raw trades
    <raw_root>/quotes/date=<d>/<ticker>.parquet    # raw NBBO quotes
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl

from quantzero.events import Event, MinuteBar, Quote, Trade
from quantzero.sources.base import order_events_per_minute

_BAR_COLUMNS = ("ts_ns", "open", "high", "low", "close", "volume", "trade_count", "vwap")
_TRADE_COLUMNS = ("ts_ns", "price", "size")
_QUOTE_COLUMNS = ("ts_ns", "bid", "ask", "bid_size", "ask_size")


def _atomic_write(frame: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)
    return path


class RawStore:
    """Reads/writes raw bars, trades, and quotes — one parquet per (kind, date, ticker)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, kind: str, day: str, ticker: str) -> Path:
        return self.root / kind / f"date={day}" / f"{ticker.replace('/', '_')}.parquet"

    # ---- bars ----
    def bars_path(self, day: str, ticker: str) -> Path:
        return self._path("bars", day, ticker)

    def has_bars(self, day: str, ticker: str) -> bool:
        return self.bars_path(day, ticker).exists()

    def write_bars(self, day: str, ticker: str, bars: list[MinuteBar]) -> Path | None:
        if not bars:
            return None
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
        return _atomic_write(frame, self.bars_path(day, ticker))

    def read_bars(self, day: str, ticker: str) -> list[MinuteBar]:
        path = self.bars_path(day, ticker)
        if not path.exists():
            return []
        frame = pl.read_parquet(path, columns=list(_BAR_COLUMNS)).sort("ts_ns")
        return [
            MinuteBar(
                ticker=ticker,
                ts_ns=int(r["ts_ns"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
                trade_count=int(r["trade_count"]),
                vwap=float(r["vwap"]),
            )
            for r in frame.iter_rows(named=True)
        ]

    # ---- trades ----
    def trades_path(self, day: str, ticker: str) -> Path:
        return self._path("trades", day, ticker)

    def has_trades(self, day: str, ticker: str) -> bool:
        return self.trades_path(day, ticker).exists()

    def write_trades(self, day: str, ticker: str, trades: list[Trade]) -> Path | None:
        if not trades:
            return None
        frame = pl.DataFrame(
            {
                "ts_ns": [t.ts_ns for t in trades],
                "price": [t.price for t in trades],
                "size": [t.size for t in trades],
            }
        )
        return _atomic_write(frame, self.trades_path(day, ticker))

    def read_trades(self, day: str, ticker: str) -> list[Trade]:
        path = self.trades_path(day, ticker)
        if not path.exists():
            return []
        frame = pl.read_parquet(path, columns=list(_TRADE_COLUMNS)).sort("ts_ns")
        return [
            Trade(
                ticker=ticker, ts_ns=int(r["ts_ns"]), price=float(r["price"]), size=float(r["size"])
            )
            for r in frame.iter_rows(named=True)
        ]

    # ---- quotes ----
    def quotes_path(self, day: str, ticker: str) -> Path:
        return self._path("quotes", day, ticker)

    def has_quotes(self, day: str, ticker: str) -> bool:
        return self.quotes_path(day, ticker).exists()

    def write_quotes(self, day: str, ticker: str, quotes: list[Quote]) -> Path | None:
        if not quotes:
            return None
        frame = pl.DataFrame(
            {
                "ts_ns": [q.ts_ns for q in quotes],
                "bid": [q.bid for q in quotes],
                "ask": [q.ask for q in quotes],
                "bid_size": [q.bid_size for q in quotes],
                "ask_size": [q.ask_size for q in quotes],
            }
        )
        return _atomic_write(frame, self.quotes_path(day, ticker))

    def read_quotes(self, day: str, ticker: str) -> list[Quote]:
        path = self.quotes_path(day, ticker)
        if not path.exists():
            return []
        frame = pl.read_parquet(path, columns=list(_QUOTE_COLUMNS)).sort("ts_ns")
        return [
            Quote(
                ticker=ticker,
                ts_ns=int(r["ts_ns"]),
                bid=float(r["bid"]),
                ask=float(r["ask"]),
                bid_size=float(r["bid_size"]),
                ask_size=float(r["ask_size"]),
            )
            for r in frame.iter_rows(named=True)
        ]

    def days(self) -> list[str]:
        base = self.root / "bars"
        if not base.exists():
            return []
        return sorted(p.name.removeprefix("date=") for p in base.glob("date=*"))

    def tickers(self, day: str) -> list[str]:
        directory = self.root / "bars" / f"date={day}"
        if not directory.exists():
            return []
        return sorted(p.stem for p in directory.glob("*.parquet"))


class RawReplaySource:
    """Replays raw bars (+ optional trades/quotes) from the raw store, merged in live order."""

    def __init__(
        self, tickers: list[str], day: str, root: str | Path, with_ticks: bool = True
    ) -> None:
        self.tickers = tickers
        self.day = day
        self.with_ticks = with_ticks
        self.store = RawStore(root)

    def iter_events(self) -> Iterator[Event]:
        events: list[Event] = []
        for ticker in self.tickers:
            events.extend(self.store.read_bars(self.day, ticker))
            if self.with_ticks:
                events.extend(self.store.read_quotes(self.day, ticker))
                events.extend(self.store.read_trades(self.day, ticker))
        yield from order_events_per_minute(events)
