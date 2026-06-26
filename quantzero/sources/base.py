"""The event-source contract + the per-minute ordering rule shared by replay sources."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from quantzero.events import Event, MinuteBar

_NS_PER_MINUTE = 60_000_000_000


class EventSource(Protocol):
    """Yields market events in non-decreasing time order.

    Per-minute invariant: a minute's trades and quotes are yielded *before* that minute's
    bar, so feature caches see the ticks first (matching the live websocket's ordering).
    """

    def iter_events(self) -> Iterator[Event]: ...


def order_events_per_minute(events: list[Event]) -> list[Event]:
    """Stable-sort events so each minute's ticks precede that minute's bar.

    Sort key is (minute_bucket, kind_rank, ts). Bars rank after ticks within their minute,
    matching how the live stream delivers a bar only at minute close. This is the single
    ordering rule that makes a merged bars+trades+quotes replay indistinguishable from live.
    """

    def sort_key(event: Event) -> tuple[int, int, int]:
        minute_bucket = event.ts_ns // _NS_PER_MINUTE
        kind_rank = 1 if isinstance(event, MinuteBar) else 0
        return (minute_bucket, kind_rank, event.ts_ns)

    return sorted(events, key=sort_key)
