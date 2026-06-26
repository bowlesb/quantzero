"""FastAPI dashboard, styled exactly like the quant-fp dashboard.

Two views (the quant-fp chrome and CSS, reproduced verbatim):
  * Coverage grid  — the feature-store coverage heatmap (dates x raw-layers + feature groups).
  * Latency        — the per-feature latency bar chart (slowest-first), from the latency harness.

Run it:

    make dashboard            # uvicorn on :8099 (QZ_DASHBOARD_PORT overrides)
    python -m quantzero.dashboard.app
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib

import numpy as np
import polars as pl
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from quantzero.config import store_root
from quantzero.driver import EngineDriver
from quantzero.features import default_features
from quantzero.latency import FeatureLatency, LatencyScenario, measure_all
from quantzero.sources.simulation import SimulationConfig, SimulationSource

BUDGET_NS = 1_000_000  # 1 ms per feature group
_SCENARIO = LatencyScenario(warmup_minutes=150, measure_minutes=2000)
_HERE = pathlib.Path(__file__).parent

# Raw tape layers that feed the features (shown as the leftmost grid columns, slate-coloured).
RAW_LAYERS = (("raw::minutes", "minutes"), ("raw::trades", "trades"), ("raw::quotes", "quotes"))

# group -> (category, mechanism) for the latency tooltip.
GROUP_META: dict[str, tuple[str, str]] = {
    "returns": ("price / trend", "ring lookups"),
    "ema": ("price / trend", "running EMAs"),
    "macd": ("price / trend", "running EMAs"),
    "momentum": ("price / trend", "ring lookups"),
    "gap": ("price / trend", "session accumulators"),
    "candle": ("price / trend", "current-bar arithmetic"),
    "vol": ("volatility", "rolling moments (sum/sumsq)"),
    "rvol": ("volatility", "rolling sum of squares"),
    "atr": ("volatility", "rolling sum of true range"),
    "boll": ("volatility", "rolling moments (20)"),
    "vwap": ("volume", "session price*volume accumulators"),
    "volprofile": ("volume", "rolling moments + session sum"),
    "tradeintensity": ("volume", "rolling moments of trade count"),
    "sessionrange": ("range", "running session extrema"),
    "rollrange": ("range", "monotonic-deque rolling min/max"),
    "rsi": ("oscillator", "Wilder EWMA of gains/losses"),
    "sessionclock": ("calendar", "timestamp arithmetic"),
    "tradeflow": ("microstructure", "signed-volume accumulators (on_trade)"),
    "quotespread": ("microstructure", "latest-quote read"),
    "quotedyn": ("microstructure", "EWMA of spread (on_quote)"),
}

app = FastAPI(title="quantzero dashboard")
_latency_cache: list[FeatureLatency] = []
_vector_p50_cache: dict[str, float] = {}


def _vector_compute_p50_ns() -> float:
    if "value" not in _vector_p50_cache:
        config = SimulationConfig(tickers=["LAT"], n_minutes=600, seed=4)
        driver = EngineDriver(config.tickers, default_features())
        samples = np.array(
            [v.compute_ns for v in driver.run_source(SimulationSource(config))], dtype=np.float64
        )
        _vector_p50_cache["value"] = float(np.percentile(samples[150:], 50))
    return _vector_p50_cache["value"]


def _latency_groups() -> list[FeatureLatency]:
    if not _latency_cache:
        _latency_cache.extend(measure_all(default_features(), _SCENARIO))
    return _latency_cache


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _latency_payload() -> dict[str, object]:
    results = _latency_groups()
    groups = []
    for result in results:
        category, mechanism = GROUP_META.get(result.name, ("feature group", "incremental cache"))
        groups.append(
            {
                "group": result.name,
                "p50_ns": result.per_minute.p50_ns,
                "p95_ns": result.per_minute.p95_ns,
                "p99_ns": result.per_minute.p99_ns,
                "own_ns": result.marginal_mean_ns,
                "kind": category,
                "mechanism": mechanism,
                "incremental_ready": "ready",
                "feat_count": result.columns,
            }
        )
    total_own = sum(result.marginal_mean_ns for result in results)
    feature_count = sum(result.columns for result in results)
    return {
        "units": "µs",
        "sorted_by": "slowest first (p99)",
        "generated_at": _now_iso(),
        "group_count": len(groups),
        "feature_count": feature_count,
        "budget_ns": BUDGET_NS,
        "all_under_budget": all(result.per_minute.p99_ns < BUDGET_NS for result in results),
        "e2e_context": {
            "vector_p50_ns": _vector_compute_p50_ns(),
            "total_own_ns": total_own,
            "budget_ns": BUDGET_NS,
        },
        "groups": groups,
        "scenario": {
            "measure_minutes": _SCENARIO.measure_minutes,
            "quotes_per_minute": _SCENARIO.quotes_per_minute,
            "trades_per_minute": _SCENARIO.trades_per_minute,
        },
    }


def _columns() -> list[dict[str, object]]:
    columns: list[dict[str, object]] = [
        {"kind": "raw", "key": key, "label": label, "features": []} for key, label in RAW_LAYERS
    ]
    for cls in default_features():
        columns.append(
            {
                "kind": "group",
                "key": cls.name,
                "label": cls.name,
                "features": [f"{cls.name}_{column}" for column in cls.columns],
            }
        )
    return columns


def _coverage(columns: list[dict[str, object]]) -> tuple[list[str], list[list[int]], int]:
    root = pathlib.Path(store_root())
    date_to_tickers: dict[str, set[str]] = {}
    for date_dir in root.glob("v=*/source=*/date=*"):
        date = date_dir.name.removeprefix("date=")
        try:
            frame = pl.read_parquet(str(date_dir / "*.parquet"), columns=["ticker"])
        except (OSError, pl.exceptions.PolarsError):
            continue
        date_to_tickers.setdefault(date, set()).update(frame["ticker"].unique().to_list())

    if not date_to_tickers:
        return [], [], 1
    universe = max(len(set().union(*date_to_tickers.values())), 1)
    dates = sorted(date_to_tickers, reverse=True)  # newest first
    coverage: list[list[int]] = []
    for date in dates:
        byte = round(255 * len(date_to_tickers[date]) / universe)
        coverage.append([byte] * len(columns))
    return dates, coverage, universe


_DIST = _HERE / "frontend" / "dist"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_DIST / "index.html")


if (_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")


@app.get("/api/latency")
def latency() -> dict[str, object]:
    return _latency_payload()


@app.post("/api/latency/refresh")
def refresh_latency() -> dict[str, object]:
    _latency_cache.clear()
    _vector_p50_cache.clear()
    return _latency_payload()


@app.get("/api/store/matrix")
def store_matrix() -> dict[str, object]:
    columns = _columns()
    dates, coverage, universe = _coverage(columns)
    return {
        "generated_at": _now_iso(),
        "universe_size": universe,
        "store_root": store_root(),
        "columns": columns,
        "dates": dates,
        "coverage": coverage,
    }


def main() -> None:
    port = int(os.environ.get("QZ_DASHBOARD_PORT", "8099"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
