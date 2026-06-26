"""Deterministic synthetic event source for offline tests and demos.

Generates a session of quotes, trades, and minute bars for a set of tickers. For each
minute it emits that minute's ticks first, then the bar, honoring the engine's ordering
contract. Seeded, so a given config always produces identical events.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from quantzero.clock import ET, to_ns
from quantzero.events import Event, MinuteBar, Quote, Trade

_NS_PER_MINUTE = 60_000_000_000


@dataclass
class SimulationConfig:
    tickers: list[str] = field(default_factory=lambda: ["AAA", "BBB"])
    day: dt.date = dt.date(2026, 6, 24)
    n_minutes: int = 90
    start_price: float = 100.0
    minute_vol: float = 0.0008
    base_volume: float = 5000.0
    seed: int = 7
    with_ticks: bool = True
    trades_per_minute: int = 6
    quotes_per_minute: int = 4


class SimulationSource:
    """An :class:`EventSource` producing a deterministic synthetic session."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self._open_ns = to_ns(
            dt.datetime(
                self.config.day.year,
                self.config.day.month,
                self.config.day.day,
                9,
                30,
                tzinfo=ET,
            )
        )

    def iter_events(self) -> Iterator[Event]:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        prices = {
            ticker: cfg.start_price * (1.0 + 0.01 * idx) for idx, ticker in enumerate(cfg.tickers)
        }

        for minute in range(cfg.n_minutes):
            minute_start = self._open_ns + minute * _NS_PER_MINUTE
            for ticker in cfg.tickers:
                yield from self._minute_events(ticker, prices, minute_start, rng)

    def _minute_events(
        self,
        ticker: str,
        prices: dict[str, float],
        minute_start: int,
        rng: np.random.Generator,
    ) -> Iterator[Event]:
        cfg = self.config
        open_price = prices[ticker]
        steps = max(cfg.trades_per_minute, 2)
        increments = rng.normal(0.0, cfg.minute_vol, steps)
        path = open_price * np.cumprod(1.0 + increments)
        high = float(max(open_price, path.max()))
        low = float(min(open_price, path.min()))
        close = float(path[-1])
        volume = float(cfg.base_volume * (0.5 + rng.random()))
        spread = max(open_price * 0.0001, 0.01)

        if cfg.with_ticks:
            n_quotes = cfg.quotes_per_minute
            n_trades = cfg.trades_per_minute
            for i in range(n_quotes):
                ts = minute_start + int((i + 0.5) / n_quotes * _NS_PER_MINUTE * 0.9)
                mid = float(path[min(i, steps - 1)])
                yield Quote(
                    ticker=ticker,
                    ts_ns=ts,
                    bid=mid - spread / 2,
                    ask=mid + spread / 2,
                    bid_size=float(rng.integers(1, 20)),
                    ask_size=float(rng.integers(1, 20)),
                )
            for i in range(n_trades):
                ts = minute_start + int((i + 0.5) / n_trades * _NS_PER_MINUTE * 0.95)
                yield Trade(
                    ticker=ticker,
                    ts_ns=ts,
                    price=float(path[min(i, steps - 1)]),
                    size=float(rng.integers(1, 100)),
                )

        yield MinuteBar(
            ticker=ticker,
            ts_ns=minute_start,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            trade_count=int(cfg.trades_per_minute),
            vwap=float((high + low + close) / 3.0),
        )
        prices[ticker] = close
