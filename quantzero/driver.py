"""Routes events to per-ticker engines. Shared by simulation, replay, and live."""

from __future__ import annotations

from collections.abc import Iterator

from quantzero.engine import FeatureEngine, FeatureVector
from quantzero.events import Event
from quantzero.feature import Feature
from quantzero.sources.base import EventSource


class EngineDriver:
    """Holds one :class:`FeatureEngine` per ticker and dispatches events to them."""

    def __init__(self, tickers: list[str], feature_classes: list[type[Feature]]) -> None:
        self.engines = {ticker: FeatureEngine(ticker, feature_classes) for ticker in tickers}

    def process(self, event: Event) -> FeatureVector | None:
        """Route one event; return a FeatureVector when it completes a minute bar."""
        if event.ticker not in self.engines:
            return None
        return self.engines[event.ticker].on_event(event)

    def run_source(self, source: EventSource) -> Iterator[FeatureVector]:
        """Drive a synchronous source to exhaustion, yielding each feature vector."""
        for event in source.iter_events():
            vector = self.process(event)
            if vector is not None:
                yield vector
