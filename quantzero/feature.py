"""The ``Feature`` base class and the global registry.

A feature is bound at construction to the shared :class:`~quantzero.state.TickerState` —
the per-ticker raw buffers the process appends to. The ``on_*`` hooks are *signals*, not
data: they carry no payload. When ``on_minute`` fires, the newest bar is already in
``state.minutes`` (read it via ``state.minutes.last_close`` etc., or index back N bars);
when ``on_trade`` / ``on_quote`` fires, the newest tick is in ``state.last_trade`` /
``state.last_quote``. A feature reacts by updating its own cache. ``values()`` then reads
that cache in a few operations. A feature may be:

  * *stateless* — leave the ``on_*`` hooks empty and read ``state.minutes`` slices in
    ``values()`` (fine when the window is tiny);
  * *stateful* — maintain running caches in the hooks so ``values()`` is a pure read (the
    preferred pattern for anything non-trivial).

Either way the engine drives it identically, and historical replay fires the same signals
in time order — so a feature computes the same value live or backfilled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from quantzero.state import TickerState


class Feature(ABC):
    """One feature group instance, bound to a single ticker for one session."""

    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    columns: ClassVar[tuple[str, ...]]

    def __init__(self, ticker: str, state: TickerState) -> None:
        self.ticker = ticker
        self.state = state
        self.setup()

    def setup(self) -> None:
        """Build the feature-specific cache. Override; default is no state."""

    def on_quote(self) -> None:
        """Signal: a new quote is in ``state.last_quote``. Override if the feature uses quotes."""

    def on_trade(self) -> None:
        """Signal: a new trade is in ``state.last_trade``. Override if the feature uses trades."""

    def on_minute(self) -> None:
        """Signal: a new bar is in ``state.minutes``. Override to update the cache."""

    @abstractmethod
    def values(self) -> np.ndarray:
        """Return one float per entry in ``columns`` (use ``np.nan`` before warmup)."""

    @property
    def output_names(self) -> list[str]:
        return [f"{self.name}_{column}" for column in self.columns]


_REGISTRY: list[type[Feature]] = []


def register(feature_cls: type[Feature]) -> type[Feature]:
    """Class decorator that adds a feature to the default registry."""
    if not hasattr(feature_cls, "name"):
        raise ValueError(f"{feature_cls.__name__} must define a class-level `name`")
    _REGISTRY.append(feature_cls)
    return feature_cls


def all_features() -> list[type[Feature]]:
    """Registered feature classes, in registration order."""
    return list(_REGISTRY)
