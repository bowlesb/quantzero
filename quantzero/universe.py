"""Daily tradable-ticker universe construction from Alpaca.

The full process is documented in ``docs/UNIVERSE.md``. In short:

  1. Pull all ACTIVE US-equity assets from Alpaca (~7000-7500).
  2. Filter to real, liquid common stock: tradable, on a major exchange, no
     fractional-only identifiers, and not ETF/ETN/leveraged-fund-like by name.
  3. Fetch a lookback of daily bars and compute average daily dollar volume (ADV$).
  4. Keep names above price and ADV$ thresholds, rank by ADV$ desc, take the top N.
  5. Persist the per-day membership to parquet (point-in-time, no survivorship bias).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from quantzero.config import AlpacaConfig, alpaca_config, store_root
from quantzero.sources.alpaca import data_feed, historical_client

KEEP_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"}
MIN_PRICE = 5.0
MIN_ADV_DOLLAR = 10_000_000.0
DEFAULT_MAX_SYMBOLS = 1000
LOOKBACK_DAYS = 20
FETCH_CHUNK = 200

_ETF_LIKE = re.compile(
    r"\b(ETF|ETN|ProShares|Direxion|iShares|SPDR|Invesco|VanEck|Global X|WisdomTree|"
    r"UltraPro|Ultra[- ]?Short|Leveraged|Inverse|Index Fund|Income Fund|"
    r"Exchange[- ]Traded|Bull|Bear|[1-3]X)\b",
    re.IGNORECASE,
)


def is_etf_like(name: str) -> bool:
    """Heuristic: does this asset name look like an ETF/ETN/leveraged fund?"""
    return bool(_ETF_LIKE.search(name or ""))


@dataclass(frozen=True)
class SymbolStat:
    symbol: str
    price: float
    adv_dollar: float


def trading_client(config: AlpacaConfig) -> TradingClient:
    paper = config.key_id.startswith("PK")
    return TradingClient(config.key_id, config.secret_key, paper=paper)


def candidate_symbols(client: TradingClient) -> list[tuple[str, str]]:
    """ACTIVE US equities that pass the structural (non-price) filters."""
    assets = client.get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    )
    kept: list[tuple[str, str]] = []
    for asset in assets:
        if isinstance(asset, str):
            continue
        exchange = str(getattr(asset.exchange, "value", asset.exchange))
        if not asset.tradable:
            continue
        if exchange not in KEEP_EXCHANGES:
            continue
        if "/" in asset.symbol:
            continue
        if is_etf_like(asset.name or ""):
            continue
        kept.append((asset.symbol, asset.name or ""))
    return kept


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def compute_adv(
    data_client: StockHistoricalDataClient,
    symbols: list[str],
    day: dt.date,
    feed_value: str,
) -> list[SymbolStat]:
    """Average daily dollar volume over the lookback window, per symbol."""
    start = dt.datetime.combine(day - dt.timedelta(days=2 * LOOKBACK_DAYS), dt.time(), dt.UTC)
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(), dt.UTC)
    feed = data_feed(AlpacaConfig("", "", feed_value))
    stats: list[SymbolStat] = []
    for chunk in _chunks(symbols, FETCH_CHUNK):
        request = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=feed,
        )
        barset = data_client.get_stock_bars(request)
        data = barset.data  # type: ignore[union-attr]
        for symbol in chunk:
            bars = data.get(symbol, [])
            if not bars:
                continue
            recent = bars[-LOOKBACK_DAYS:]
            adv = sum(bar.close * bar.volume for bar in recent) / len(recent)
            stats.append(
                SymbolStat(symbol=symbol, price=float(recent[-1].close), adv_dollar=float(adv))
            )
    return stats


def select_universe(
    stats: list[SymbolStat],
    min_price: float = MIN_PRICE,
    min_adv_dollar: float = MIN_ADV_DOLLAR,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
) -> list[SymbolStat]:
    """Threshold by price/ADV$, rank by ADV$ desc (symbol tie-break), take top N."""
    eligible = [s for s in stats if s.price >= min_price and s.adv_dollar >= min_adv_dollar]
    eligible.sort(key=lambda s: (-s.adv_dollar, s.symbol))
    return eligible[:max_symbols]


def write_universe(root: str | Path, day: dt.date, chosen: list[SymbolStat]) -> Path:
    directory = Path(root) / "universe" / f"date={day.isoformat()}"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "trade_date": [day.isoformat()] * len(chosen),
            "rank": list(range(1, len(chosen) + 1)),
            "symbol": [s.symbol for s in chosen],
            "price": [s.price for s in chosen],
            "adv_dollar": [s.adv_dollar for s in chosen],
        }
    )
    path = directory / "universe.parquet"
    frame.write_parquet(path)
    return path


def build_universe(
    day: dt.date,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    config: AlpacaConfig | None = None,
    root: str | Path | None = None,
) -> Path:
    cfg = config or alpaca_config()
    candidates = candidate_symbols(trading_client(cfg))
    symbols = [symbol for symbol, _ in candidates]
    stats = compute_adv(historical_client(cfg), symbols, day, cfg.data_feed)
    chosen = select_universe(stats, max_symbols=max_symbols)
    path = write_universe(root or store_root(), day, chosen)
    print(
        f"universe {day.isoformat()}: {len(candidates)} candidates -> "
        f"{len(stats)} priced -> {len(chosen)} selected -> {path}"
    )
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the daily tradable universe.")
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--date", help="ET trade date YYYY-MM-DD (default: today)")
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_MAX_SYMBOLS)
    args = parser.parse_args(argv)
    day = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(dt.UTC).date()
    build_universe(day, max_symbols=args.max_symbols)


if __name__ == "__main__":
    main()
