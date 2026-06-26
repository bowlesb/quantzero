"""Raw-store retention: keep quotes a short window, bars+trades much longer.

Quotes are by far the largest layer, so they are kept only for a recent window; bars and
trades are tiny by comparison and kept for far longer. Features stay recomputable from the
bars+trades that remain (the feature store itself is never pruned here).

    python -m quantzero.retention                      # prune with the default policy
    python -m quantzero.retention --dry-run            # show what would be deleted
    python -m quantzero.retention --quotes-days 35 --trades-days 180 --bars-days 180
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

from quantzero.clock import ET
from quantzero.config import raw_root

DEFAULT_QUOTES_DAYS = 35  # ~5 weeks
DEFAULT_TRADES_DAYS = 180  # ~6 months
DEFAULT_BARS_DAYS = 180  # ~6 months


def _prune_kind(root: str | Path, kind: str, cutoff: dt.date, dry_run: bool) -> int:
    base = Path(root) / kind
    if not base.exists():
        return 0
    pruned = 0
    for partition in base.glob("date=*"):
        day = dt.date.fromisoformat(partition.name.removeprefix("date="))
        if day < cutoff:
            if not dry_run:
                shutil.rmtree(partition)
            pruned += 1
    return pruned


def prune_raw(
    root: str | Path,
    today: dt.date,
    quotes_days: int = DEFAULT_QUOTES_DAYS,
    trades_days: int = DEFAULT_TRADES_DAYS,
    bars_days: int = DEFAULT_BARS_DAYS,
    dry_run: bool = False,
) -> dict[str, int]:
    """Delete raw date-partitions older than each layer's retention. Returns deleted counts."""
    return {
        "quotes": _prune_kind(root, "quotes", today - dt.timedelta(days=quotes_days), dry_run),
        "trades": _prune_kind(root, "trades", today - dt.timedelta(days=trades_days), dry_run),
        "bars": _prune_kind(root, "bars", today - dt.timedelta(days=bars_days), dry_run),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prune the raw store by retention policy.")
    parser.add_argument("--quotes-days", type=int, default=DEFAULT_QUOTES_DAYS)
    parser.add_argument("--trades-days", type=int, default=DEFAULT_TRADES_DAYS)
    parser.add_argument("--bars-days", type=int, default=DEFAULT_BARS_DAYS)
    parser.add_argument("--today", help="override 'today' as YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    today = dt.date.fromisoformat(args.today) if args.today else dt.datetime.now(ET).date()
    result = prune_raw(
        raw_root(),
        today,
        quotes_days=args.quotes_days,
        trades_days=args.trades_days,
        bars_days=args.bars_days,
        dry_run=args.dry_run,
    )
    verb = "would delete" if args.dry_run else "deleted"
    print(
        f"retention ({raw_root()}, today={today.isoformat()}): {verb} date-partitions — "
        f"quotes>{args.quotes_days}d: {result['quotes']}, "
        f"trades>{args.trades_days}d: {result['trades']}, "
        f"bars>{args.bars_days}d: {result['bars']}"
    )


if __name__ == "__main__":
    main()
