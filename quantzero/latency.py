"""Per-feature latency measurement — the standard way we time any feature.

The harness drives one feature through warmed-up, realistic per-minute cycles (a minute's
quotes, then its trades, then the bar, then ``values()``) and reports the distribution of
per-minute cost. Because the per-minute total also includes the trivial shared state
updates, we measure a state-only **baseline** (a null feature) once and report each
feature's **marginal** mean — its own cost above the baseline pipeline.

Usage:

    from quantzero.latency import measure_feature, measure_all
    result = measure_feature(Rsi)            # one feature
    results = measure_all(default_features())  # every feature

    python -m quantzero.latency               # CLI table
    python -m quantzero.latency --json out.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np

from quantzero.clock import ET, to_ns
from quantzero.events import MinuteBar, Quote, Trade
from quantzero.feature import Feature
from quantzero.state import TickerState

_NS_PER_MINUTE = 60_000_000_000
_OPEN_NS = to_ns(dt.datetime(2026, 6, 24, 9, 30, tzinfo=ET))
_TICKER = "LAT"


@dataclass(frozen=True, slots=True)
class MinuteSlice:
    """One minute of input: the quotes and trades that arrive, then the closing bar."""

    quotes: list[Quote]
    trades: list[Trade]
    bar: MinuteBar


@dataclass(frozen=True)
class LatencyScenario:
    warmup_minutes: int = 200
    measure_minutes: int = 4000
    trades_per_minute: int = 8
    quotes_per_minute: int = 6
    seed: int = 0


@dataclass(frozen=True)
class LatencyStats:
    n: int
    mean_ns: float
    p50_ns: float
    p90_ns: float
    p99_ns: float
    max_ns: float

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> LatencyStats:
        return cls(
            n=int(samples.shape[0]),
            mean_ns=float(samples.mean()),
            p50_ns=float(np.percentile(samples, 50)),
            p90_ns=float(np.percentile(samples, 90)),
            p99_ns=float(np.percentile(samples, 99)),
            max_ns=float(samples.max()),
        )


@dataclass(frozen=True)
class FeatureLatency:
    name: str
    columns: int
    per_minute: LatencyStats
    values_only: LatencyStats
    marginal_mean_ns: float


def generate_slices(n_minutes: int, scenario: LatencyScenario) -> list[MinuteSlice]:
    """Deterministic, realistic per-minute input for one ticker."""
    rng = np.random.default_rng(scenario.seed)
    price = 100.0
    n_quotes = scenario.quotes_per_minute
    n_trades = scenario.trades_per_minute
    steps = max(n_trades, 2)
    slices: list[MinuteSlice] = []
    for minute in range(n_minutes):
        minute_start = _OPEN_NS + minute * _NS_PER_MINUTE
        path = price * np.cumprod(1.0 + rng.normal(0.0, 0.0008, steps))
        high = float(max(price, path.max()))
        low = float(min(price, path.min()))
        close = float(path[-1])
        spread = max(price * 0.0001, 0.01)
        quotes = [
            Quote(
                ticker=_TICKER,
                ts_ns=minute_start + int((i + 0.5) / n_quotes * _NS_PER_MINUTE * 0.9),
                bid=float(path[min(i, steps - 1)]) - spread / 2,
                ask=float(path[min(i, steps - 1)]) + spread / 2,
                bid_size=float(rng.integers(1, 20)),
                ask_size=float(rng.integers(1, 20)),
            )
            for i in range(n_quotes)
        ]
        trades = [
            Trade(
                ticker=_TICKER,
                ts_ns=minute_start + int((i + 0.5) / n_trades * _NS_PER_MINUTE * 0.95),
                price=float(path[min(i, steps - 1)]),
                size=float(rng.integers(1, 100)),
            )
            for i in range(n_trades)
        ]
        bar = MinuteBar(
            ticker=_TICKER,
            ts_ns=minute_start,
            open=price,
            high=high,
            low=low,
            close=close,
            volume=float(rng.integers(1000, 10000)),
            trade_count=n_trades,
            vwap=(high + low + close) / 3.0,
        )
        slices.append(MinuteSlice(quotes=quotes, trades=trades, bar=bar))
        price = close
    return slices


class _NullFeature(Feature):
    """A do-nothing feature; driving it measures the shared per-minute pipeline baseline."""

    name = "_null"
    columns: ClassVar[tuple[str, ...]] = ()

    def values(self) -> np.ndarray:
        return _EMPTY


_EMPTY = np.empty(0, dtype=np.float64)


def measure_feature(
    feature_cls: type[Feature],
    scenario: LatencyScenario | None = None,
    baseline_ns: float | None = None,
) -> FeatureLatency:
    """Time one feature over warmed-up, realistic per-minute cycles.

    ``baseline_ns`` is the state-only per-minute mean (from :func:`measure_baseline`); when
    provided, ``marginal_mean_ns`` is this feature's own cost above that baseline.
    """
    scenario = scenario or LatencyScenario()
    warmup = scenario.warmup_minutes
    measure = scenario.measure_minutes
    slices = generate_slices(warmup + measure, scenario)

    state = TickerState(_TICKER, minute_capacity=warmup + measure + 2)
    feature = feature_cls(_TICKER, state)
    for sliced in slices[:warmup]:
        _drive(state, feature, sliced)

    totals = np.empty(measure, dtype=np.float64)
    values = np.empty(measure, dtype=np.float64)
    for i in range(measure):
        sliced = slices[warmup + i]
        start = time.perf_counter_ns()
        for quote in sliced.quotes:
            state.on_quote(quote)
            feature.on_quote()
        for trade in sliced.trades:
            state.on_trade(trade)
            feature.on_trade()
        state.on_minute(sliced.bar)
        feature.on_minute()
        before_values = time.perf_counter_ns()
        feature.values()
        end = time.perf_counter_ns()
        totals[i] = end - start
        values[i] = end - before_values

    per_minute = LatencyStats.from_samples(totals)
    marginal = per_minute.mean_ns - baseline_ns if baseline_ns is not None else per_minute.mean_ns
    return FeatureLatency(
        name=feature_cls.name,
        columns=len(feature_cls.columns),
        per_minute=per_minute,
        values_only=LatencyStats.from_samples(values),
        marginal_mean_ns=marginal,
    )


def _drive(state: TickerState, feature: Feature, sliced: MinuteSlice) -> None:
    for quote in sliced.quotes:
        state.on_quote(quote)
        feature.on_quote()
    for trade in sliced.trades:
        state.on_trade(trade)
        feature.on_trade()
    state.on_minute(sliced.bar)
    feature.on_minute()
    feature.values()


def measure_baseline(scenario: LatencyScenario | None = None) -> float:
    """State-only per-minute mean (ns): the pipeline cost with no real feature work."""
    return measure_feature(_NullFeature, scenario).per_minute.mean_ns


def measure_all(
    feature_classes: list[type[Feature]],
    scenario: LatencyScenario | None = None,
) -> list[FeatureLatency]:
    """Measure every feature against one shared baseline."""
    scenario = scenario or LatencyScenario()
    baseline = measure_baseline(scenario)
    return [measure_feature(cls, scenario, baseline) for cls in feature_classes]


def format_table(results: list[FeatureLatency]) -> str:
    header = f"{'group':16s} {'cols':>4s} {'own(mean)':>11s} {'values':>9s} {'p50':>9s} {'p99':>9s}"
    lines = [header]
    total_marginal = 0.0
    for result in results:
        total_marginal += result.marginal_mean_ns
        lines.append(
            f"{result.name:16s} {result.columns:4d} "
            f"{result.marginal_mean_ns / 1000:8.3f}us "
            f"{result.values_only.mean_ns / 1000:6.3f}us "
            f"{result.per_minute.p50_ns / 1000:6.3f}us "
            f"{result.per_minute.p99_ns / 1000:6.3f}us"
        )
    lines.append(f"{'TOTAL own':16s} {'':>4s} {total_marginal / 1000:8.3f}us")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Per-feature latency harness.")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measure", type=int, default=4000)
    parser.add_argument("--json", type=str, default=None, help="Write results to this JSON path")
    args = parser.parse_args(argv)

    from quantzero.features import default_features

    scenario = LatencyScenario(warmup_minutes=args.warmup, measure_minutes=args.measure)
    baseline = measure_baseline(scenario)
    results = [measure_feature(cls, scenario, baseline) for cls in default_features()]
    print(f"baseline (state-only) per minute: {baseline / 1000:.3f}us")
    print(format_table(results))

    if args.json:
        payload = {
            "scenario": _scenario_dict(scenario),
            "baseline_ns": baseline,
            "features": [_result_dict(r) for r in results],
        }
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json}")


def _scenario_dict(scenario: LatencyScenario) -> dict[str, int]:  # noqa: D401
    return {
        "warmup_minutes": scenario.warmup_minutes,
        "measure_minutes": scenario.measure_minutes,
        "trades_per_minute": scenario.trades_per_minute,
        "quotes_per_minute": scenario.quotes_per_minute,
        "seed": scenario.seed,
    }


def _result_dict(result: FeatureLatency) -> dict[str, object]:
    return {
        "name": result.name,
        "columns": result.columns,
        "marginal_mean_ns": result.marginal_mean_ns,
        "per_minute": asdict(result.per_minute),
        "values_only": asdict(result.values_only),
    }


if __name__ == "__main__":
    main()
