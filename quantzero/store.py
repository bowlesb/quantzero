"""Versioned, source-transparent parquet feature store.

Ported (simplified) from the quant-fp design. Two properties we keep:

  * **Versioning** lives in the path (``v=<set_version>``); a new version writes alongside
    the old, and readers select a version explicitly.
  * **Source transparency**: rows are written under ``source=backfill|stream|sim`` but the
    reader merges them so callers never branch on provenance. Backfill is truth; live
    (stream/sim) fills only the ``(ticker, minute)`` keys not yet backfilled. Ties break
    latest-write-wins by file mtime.

Layout::

    <root>/v=<set_version>/source=<source>/date=<YYYY-MM-DD>/data-<ts_ns>.parquet

Each file holds rows keyed by ``(ticker, ts_ns)`` with one column per feature.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from quantzero.clock import date_to_ns_range, ns_to_et
from quantzero.engine import FeatureVector

KEY_COLUMNS = ("ticker", "ts_ns")
PROVISIONAL_SOURCES = ("stream", "sim")


def _partition_dir(root: str | Path, set_version: str, source: str, day: str) -> Path:
    return Path(root) / f"v={set_version}" / f"source={source}" / f"date={day}"


def vector_to_row(vector: FeatureVector) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"ticker": vector.ticker, "ts_ns": vector.ts_ns}
    for name, value in zip(vector.columns, vector.values, strict=True):
        row[name] = float(value)
    return row


class FeatureStore:
    """Writes feature vectors to one ``(set_version, source)`` partition tree."""

    def __init__(self, root: str | Path, set_version: str, source: str) -> None:
        if source not in ("backfill", *PROVISIONAL_SOURCES):
            raise ValueError(f"unknown source {source!r}")
        self.root = Path(root)
        self.set_version = set_version
        self.source = source

    def write(self, vector: FeatureVector) -> Path:
        day = ns_to_et(vector.ts_ns).date().isoformat()
        directory = _partition_dir(self.root, self.set_version, self.source, day)
        directory.mkdir(parents=True, exist_ok=True)
        # Filename must be unique per (ticker, minute): many tickers share a minute's ts_ns,
        # so keying on ts_ns alone would let them overwrite each other.
        safe_ticker = vector.ticker.replace("/", "_")
        path = directory / f"data-{safe_ticker}-{vector.ts_ns}.parquet"
        frame = pl.DataFrame([vector_to_row(vector)])
        tmp = path.with_suffix(".parquet.tmp")
        frame.write_parquet(tmp)
        tmp.replace(path)
        return path

    def write_many(self, vectors: list[FeatureVector]) -> int:
        for vector in vectors:
            self.write(vector)
        return len(vectors)


def _scan_source(
    root: str | Path,
    set_version: str,
    source: str,
    start_ns: int,
    end_ns: int,
) -> pl.DataFrame:
    base = Path(root) / f"v={set_version}" / f"source={source}"
    if not base.exists():
        return pl.DataFrame()
    files = sorted(base.glob("date=*/data-*.parquet"), key=lambda p: p.stat().st_mtime)
    frames = [pl.read_parquet(path) for path in files]
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="vertical_relaxed")
    combined = combined.filter((pl.col("ts_ns") >= start_ns) & (pl.col("ts_ns") < end_ns))
    # Latest-write-wins within a source (files already sorted oldest->newest).
    return combined.unique(subset=list(KEY_COLUMNS), keep="last", maintain_order=True)


def settled_dates(root: str | Path, set_version: str) -> set[str]:
    """Dates that have a backfill partition (i.e. settled truth)."""
    base = Path(root) / f"v={set_version}" / "source=backfill"
    if not base.exists():
        return set()
    return {p.name.removeprefix("date=") for p in base.glob("date=*")}


def read_features(
    root: str | Path,
    set_version: str,
    start_ns: int,
    end_ns: int,
    names: list[str] | None = None,
    source: str = "auto",
    provisional: str = "stream",
) -> pl.DataFrame:
    """Read features for a time range, merging sources transparently.

    ``source="auto"`` returns backfill where present and fills remaining keys from the
    provisional source (``stream`` or ``sim``). Returns rows keyed by (ticker, ts_ns).
    """
    if source == "auto":
        backfill = _scan_source(root, set_version, "backfill", start_ns, end_ns)
        live = _scan_source(root, set_version, provisional, start_ns, end_ns)
        if backfill.height == 0:
            merged = live
        elif live.height == 0:
            merged = backfill
        else:
            extra = live.join(backfill.select(KEY_COLUMNS), on=list(KEY_COLUMNS), how="anti")
            merged = pl.concat([backfill, extra], how="vertical_relaxed")
    else:
        merged = _scan_source(root, set_version, source, start_ns, end_ns)

    if merged.height == 0:
        return merged
    if names is not None:
        keep = [*KEY_COLUMNS, *[n for n in names if n in merged.columns]]
        merged = merged.select(keep)
    return merged.sort(list(KEY_COLUMNS))


def day_ns_bounds(day: dt.date) -> tuple[int, int]:
    return date_to_ns_range(day)
