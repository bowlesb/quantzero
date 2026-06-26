"""FastAPI dashboard with two views: feature latency and feature store.

Run it:

    make dashboard            # uvicorn on :8099
    # or
    python -m quantzero.dashboard.app

The latency view runs the per-feature harness (quantzero.latency) and shows each group's
own cost; the budget line is 1ms, which every group sits far below. The feature-store view
shows the feature catalog and any persisted parquet partitions.
"""

from __future__ import annotations

import os
import pathlib

import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from quantzero.config import store_root
from quantzero.features import default_features
from quantzero.latency import FeatureLatency, LatencyScenario, measure_all

BUDGET_NS = 1_000_000  # 1 ms per feature group
_SCENARIO = LatencyScenario(warmup_minutes=150, measure_minutes=2000)
_INDEX_HTML = (pathlib.Path(__file__).parent / "index.html").read_text()

app = FastAPI(title="quantzero dashboard")
_latency_cache: list[dict[str, float | int | str]] = []


def _compute_latency() -> list[dict[str, float | int | str]]:
    results = measure_all(default_features(), _SCENARIO)
    return [_latency_row(result) for result in results]


def _latency_row(result: FeatureLatency) -> dict[str, float | int | str]:
    return {
        "group": result.name,
        "columns": result.columns,
        "own_mean_ns": result.marginal_mean_ns,
        "values_mean_ns": result.values_only.mean_ns,
        "p50_ns": result.per_minute.p50_ns,
        "p99_ns": result.per_minute.p99_ns,
        "under_budget": result.per_minute.p99_ns < BUDGET_NS,
    }


def _catalog() -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    for cls in default_features():
        catalog.append(
            {
                "group": cls.name,
                "version": cls.version,
                "n_columns": len(cls.columns),
                "columns": [f"{cls.name}_{column}" for column in cls.columns],
            }
        )
    return catalog


def _partitions() -> list[dict[str, object]]:
    root = pathlib.Path(store_root())
    partitions: list[dict[str, object]] = []
    if not root.exists():
        return partitions
    for date_dir in sorted(root.glob("v=*/source=*/date=*")):
        version = date_dir.parts[-3].removeprefix("v=")
        source = date_dir.parts[-2].removeprefix("source=")
        date = date_dir.name.removeprefix("date=")
        files = sorted(date_dir.glob("*.parquet"))
        rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
        partitions.append(
            {
                "version": version,
                "source": source,
                "date": date,
                "files": len(files),
                "rows": rows,
            }
        )
    return partitions


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


@app.get("/api/latency")
def latency() -> dict[str, object]:
    if not _latency_cache:
        _latency_cache.extend(_compute_latency())
    total_own = sum(float(row["own_mean_ns"]) for row in _latency_cache)
    return {
        "budget_ns": BUDGET_NS,
        "total_own_ns": total_own,
        "all_under_budget": all(bool(row["under_budget"]) for row in _latency_cache),
        "scenario": {
            "warmup_minutes": _SCENARIO.warmup_minutes,
            "measure_minutes": _SCENARIO.measure_minutes,
            "trades_per_minute": _SCENARIO.trades_per_minute,
            "quotes_per_minute": _SCENARIO.quotes_per_minute,
        },
        "groups": _latency_cache,
    }


@app.post("/api/latency/refresh")
def refresh_latency() -> dict[str, object]:
    _latency_cache.clear()
    _latency_cache.extend(_compute_latency())
    return {"refreshed": len(_latency_cache)}


@app.get("/api/store")
def store() -> dict[str, object]:
    catalog = _catalog()
    n_columns = sum(len(cls.columns) for cls in default_features())
    return {
        "n_groups": len(catalog),
        "n_columns": n_columns,
        "store_root": store_root(),
        "catalog": catalog,
        "partitions": _partitions(),
    }


def main() -> None:
    port = int(os.environ.get("QZ_DASHBOARD_PORT", "8099"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
