"""Drive the simulation source through the engine — a no-network end-to-end demo.

    python -m quantzero.run_sim --tickers AAA,BBB --minutes 120 --store

With ``--store`` the resulting vectors are written to the feature store under
``source=sim``, demonstrating the same write path the live runner uses.
"""

from __future__ import annotations

import argparse

import numpy as np

from quantzero.config import store_root
from quantzero.driver import EngineDriver
from quantzero.engine import FeatureVector
from quantzero.features import default_features
from quantzero.sources.simulation import SimulationConfig, SimulationSource
from quantzero.store import FeatureStore

SET_VERSION = "0.1.0"


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
    parser.add_argument("--store", action="store_true", help="Persist vectors as source=sim")
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    config = SimulationConfig(tickers=tickers, n_minutes=args.minutes, seed=args.seed)
    driver = EngineDriver(tickers, default_features())

    store = FeatureStore(store_root(), SET_VERSION, source="sim") if args.store else None
    vectors: list[FeatureVector] = []
    for vector in driver.run_source(SimulationSource(config)):
        vectors.append(vector)
        if store is not None:
            store.write(vector)

    summarize(vectors)
    if store is not None:
        print(f"wrote {len(vectors)} vectors to {store_root()} (source=sim, v={SET_VERSION})")


if __name__ == "__main__":
    main()
