"""Standalone per-feature latency benchmark — measured outside any simulation.

For each feature group we warm its cache with a realistic history, then time a single
event cycle (``on_minute`` + ``values``) repeated many times. This isolates the *steady
state* cost of each feature — the cost that actually runs on every live bar once caches
are populated — independent of the rest of the pipeline.

    python -m quantzero.bench [--iters 20000] [--warmup 400]
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np

from quantzero.clock import ET, to_ns
from quantzero.events import MinuteBar
from quantzero.feature import Feature
from quantzero.features import default_features
from quantzero.metrics import FEATURE_GROUP_SECONDS
from quantzero.state import TickerState

_NS_PER_MINUTE = 60_000_000_000
_OPEN_NS = to_ns(dt.datetime(2026, 6, 24, 9, 30, tzinfo=ET))


def synthetic_bars(n: int, seed: int = 0) -> list[MinuteBar]:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.001, n))
    bars: list[MinuteBar] = []
    prev = 100.0
    for i in range(n):
        close = float(closes[i])
        high = max(prev, close) * (1.0 + abs(rng.normal(0, 0.0005)))
        low = min(prev, close) * (1.0 - abs(rng.normal(0, 0.0005)))
        bars.append(
            MinuteBar(
                ticker="BENCH",
                ts_ns=_OPEN_NS + i * _NS_PER_MINUTE,
                open=prev,
                high=high,
                low=low,
                close=close,
                volume=float(rng.integers(1000, 10000)),
                trade_count=int(rng.integers(10, 500)),
                vwap=(high + low + close) / 3.0,
            )
        )
        prev = close
    return bars


def bench_feature(feature_cls: type[Feature], warmup: int, iters: int) -> tuple[float, float]:
    """Return (mean_ns, p99_ns) for one steady-state event cycle of a feature group."""
    bars = synthetic_bars(warmup + iters + 1)
    state = TickerState("BENCH", minute_capacity=warmup + iters + 2)
    feature = feature_cls("BENCH", state)
    for bar in bars[:warmup]:
        state.on_minute(bar)
        feature.on_minute()
        feature.values()

    samples = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        bar = bars[warmup + i]
        state.on_minute(bar)
        start = time.perf_counter_ns()
        feature.on_minute()
        feature.values()
        samples[i] = time.perf_counter_ns() - start
    return float(samples.mean()), float(np.percentile(samples, 99))


def run(warmup: int, iters: int) -> None:
    print(f"per-feature steady-state latency  (warmup={warmup}, iters={iters})")
    print(f"{'group':16s} {'mean':>10s} {'p99':>10s}")
    total_mean = 0.0
    for feature_cls in default_features():
        mean_ns, p99_ns = bench_feature(feature_cls, warmup, iters)
        total_mean += mean_ns
        FEATURE_GROUP_SECONDS.labels(group=feature_cls.name).observe(mean_ns / 1e9)
        print(f"{feature_cls.name:16s} {mean_ns / 1000:8.2f}us {p99_ns / 1000:8.2f}us")
    print(f"{'TOTAL':16s} {total_mean / 1000:8.2f}us")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Per-feature latency benchmark.")
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--iters", type=int, default=20000)
    args = parser.parse_args(argv)
    run(args.warmup, args.iters)


if __name__ == "__main__":
    main()
