"""Drive the simulation source through the engine — a no-network, no-persistence test.

Simulation exists to exercise the SAME caching + feature code as live and backfill, and to
measure latency off-hours. It deliberately writes nothing to the feature store (that would
just waste space); it computes in memory and reports compute-time stats.

    python -m quantzero.run_sim --tickers AAA,BBB --minutes 120
"""

from __future__ import annotations

import argparse

import numpy as np

from quantzero.driver import EngineDriver
from quantzero.engine import FeatureVector
from quantzero.features import default_features
from quantzero.sources.simulation import SimulationConfig, SimulationSource


def summarize(vectors: list[FeatureVector]) -> None:
    times_us = np.array([v.compute_ns for v in vectors], dtype=np.float64) / 1000.0
    print(
        f"vectors={len(vectors)} columns={len(vectors[0].columns)}  "
        f"compute mean={times_us.mean():.1f}us p50={np.percentile(times_us, 50):.1f}us "
        f"p99={np.percentile(times_us, 99):.1f}us max={times_us.max():.1f}us"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the simulation through the engine.")
    parser.add_argument("--tickers", default="AAA,BBB")
    parser.add_argument("--minutes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    config = SimulationConfig(tickers=tickers, n_minutes=args.minutes, seed=args.seed)
    driver = EngineDriver(tickers, default_features())
    vectors = list(driver.run_source(SimulationSource(config)))
    summarize(vectors)


if __name__ == "__main__":
    main()
