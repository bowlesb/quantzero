"""Live-vs-backfill parity check.

Proves the thesis: features computed from the real-time stream (``source=stream``) match
features computed by replaying the same day through the same engine (``source=backfill``).
For every ``(ticker, minute)`` present in both sources, we compare each feature value and
report the agreement rate and the worst-disagreeing features.

    python -m quantzero.parity --date 2026-06-26 [--tickers AAPL,MSFT] [--rtol 1e-3]
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from quantzero.config import store_root
from quantzero.store import KEY_COLUMNS, day_ns_bounds, read_features

SET_VERSION = "0.1.0"


@dataclass(frozen=True)
class ParityReport:
    date: str
    source_a: str
    source_b: str
    matched_keys: int
    only_a: int
    only_b: int
    n_features: int
    cells_compared: int
    cells_mismatch: int
    worst: list[tuple[str, int, float]]  # (feature, mismatches, max_abs_diff)

    @property
    def cell_match_pct(self) -> float:
        if self.cells_compared == 0:
            return float("nan")
        return 100.0 * (1.0 - self.cells_mismatch / self.cells_compared)


def compare_sources(
    day: dt.date,
    tickers: list[str] | None = None,
    set_version: str = SET_VERSION,
    source_a: str = "stream",
    source_b: str = "backfill",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    root: str | None = None,
) -> ParityReport:
    root = root or store_root()
    start_ns, end_ns = day_ns_bounds(day)
    frame_a = read_features(root, set_version, start_ns, end_ns, source=source_a)
    frame_b = read_features(root, set_version, start_ns, end_ns, source=source_b)
    if tickers is not None:
        keep = set(tickers)
        frame_a = frame_a.filter(pl.col("ticker").is_in(keep)) if frame_a.height else frame_a
        frame_b = frame_b.filter(pl.col("ticker").is_in(keep)) if frame_b.height else frame_b

    keys = list(KEY_COLUMNS)
    keys_a = set(map(tuple, frame_a.select(keys).iter_rows())) if frame_a.height else set()
    keys_b = set(map(tuple, frame_b.select(keys).iter_rows())) if frame_b.height else set()
    only_a = len(keys_a - keys_b)
    only_b = len(keys_b - keys_a)

    if frame_a.height == 0 or frame_b.height == 0:
        return ParityReport(
            day.isoformat(), source_a, source_b, len(keys_a & keys_b), only_a, only_b, 0, 0, 0, []
        )

    feature_cols = [c for c in frame_a.columns if c not in KEY_COLUMNS]
    joined = frame_a.join(frame_b, on=keys, how="inner", suffix="__b").sort(keys)
    matched = joined.height

    cells_mismatch = 0
    worst: list[tuple[str, int, float]] = []
    for col in feature_cols:
        a = joined[col].to_numpy().astype(np.float64)
        b = joined[f"{col}__b"].to_numpy().astype(np.float64)
        close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        n_bad = int((~close).sum())
        if n_bad:
            max_diff = float(np.nanmax(np.abs(a - b)[~close])) if n_bad else 0.0
            worst.append((col, n_bad, max_diff))
            cells_mismatch += n_bad

    worst.sort(key=lambda item: item[1], reverse=True)
    return ParityReport(
        date=day.isoformat(),
        source_a=source_a,
        source_b=source_b,
        matched_keys=matched,
        only_a=only_a,
        only_b=only_b,
        n_features=len(feature_cols),
        cells_compared=matched * len(feature_cols),
        cells_mismatch=cells_mismatch,
        worst=worst[:10],
    )


def format_report(report: ParityReport) -> str:
    lines = [
        f"parity {report.source_a} vs {report.source_b}  ({report.date})",
        f"  matched (ticker,minute) keys : {report.matched_keys}",
        f"  only in {report.source_a:<8}            : {report.only_a}",
        f"  only in {report.source_b:<8}            : {report.only_b}",
        f"  feature cells compared        : {report.cells_compared} "
        f"({report.n_features} features)",
        f"  cell agreement                : {report.cell_match_pct:.4f}% "
        f"({report.cells_mismatch} mismatches)",
    ]
    if report.worst:
        lines.append("  worst features (mismatches, max|Δ|):")
        for feature, n_bad, max_diff in report.worst:
            lines.append(f"    {feature:28s} {n_bad:5d}  {max_diff:.3e}")
    else:
        lines.append("  all compared cells agree within tolerance ✓")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Live-vs-backfill feature parity check.")
    parser.add_argument("--date", required=True, help="ET trade date YYYY-MM-DD")
    parser.add_argument("--tickers", default="", help="comma-separated subset (default: all)")
    parser.add_argument("--source-a", default="stream")
    parser.add_argument("--source-b", default="backfill")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None
    report = compare_sources(
        dt.date.fromisoformat(args.date),
        tickers=tickers,
        source_a=args.source_a,
        source_b=args.source_b,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
