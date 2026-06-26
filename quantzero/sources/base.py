"""The event-source contract shared by simulation, replay, and live."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from quantzero.events import Event


class EventSource(Protocol):
    """Yields market events in non-decreasing time order.

    Per-minute invariant: a minute's trades and quotes are yielded *before* that minute's
    bar, so feature caches see the ticks first (matching the live websocket's ordering).
    """

    def iter_events(self) -> Iterator[Event]: ...
