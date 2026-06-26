"""Prometheus metrics for feature-computation speed (scraped into Grafana)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, start_http_server

# Sub-millisecond-aware buckets: we care about the few-ms target.
_COMPUTE_BUCKETS = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
)
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

FEATURE_COMPUTE_SECONDS = Histogram(
    "qz_feature_compute_seconds",
    "Time to compute a full feature vector for one minute bar.",
    buckets=_COMPUTE_BUCKETS,
)
BAR_TO_VECTOR_SECONDS = Histogram(
    "qz_bar_to_vector_seconds",
    "Wall-clock latency from the bar's timestamp to the computed feature vector.",
    buckets=_LATENCY_BUCKETS,
)
VECTORS_TOTAL = Counter(
    "qz_vectors_total",
    "Feature vectors computed.",
    labelnames=("ticker",),
)


def start_metrics_server(port: int) -> None:
    """Expose /metrics on ``port`` if ``port`` is non-zero."""
    if port:
        start_http_server(port)
