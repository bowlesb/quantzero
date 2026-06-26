"""Deterministic ticker sharding across worker processes.

A single router holds the one Alpaca connection (Alpaca allows one stream per account) and
fans each event out to the worker that owns its ticker, chosen by a stable hash. Each worker
runs its own :class:`~quantzero.driver.EngineDriver` over its ticker subset, with no shared
mutable state — so sharding is pure partitioning: a ticker computes the same vector no matter
which worker owns it (verified by the parity test).

Events cross the process boundary as pickled dataclasses on per-worker queues; workers return
compact :class:`VectorSummary` objects on a shared results queue, which the router drains.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing.queues import Queue

import numpy as np

from quantzero.driver import EngineDriver
from quantzero.engine import FeatureVector
from quantzero.events import Event
from quantzero.feature import Feature
from quantzero.features import default_features
from quantzero.sources.base import EventSource


def worker_for(ticker: str, n_workers: int) -> int:
    """Stable worker id for a ticker (crc32 is deterministic across processes; ``hash`` is not)."""
    return zlib.crc32(ticker.encode()) % n_workers


def assign_tickers(tickers: Iterable[str], n_workers: int) -> dict[int, list[str]]:
    """Partition tickers across workers by :func:`worker_for`."""
    assignment: dict[int, list[str]] = {worker_id: [] for worker_id in range(n_workers)}
    for ticker in tickers:
        assignment[worker_for(ticker, n_workers)].append(ticker)
    return assignment


@dataclass(frozen=True, slots=True)
class VectorSummary:
    """Compact result a worker returns to the router for each computed vector."""

    worker_id: int
    ticker: str
    ts_ns: int
    compute_ns: int
    n_features: int
    n_valid: int


@dataclass(frozen=True, slots=True)
class _WorkerDone:
    worker_id: int


class ShardedRouter:
    """In-process model of the sharded system: one :class:`EngineDriver` per shard.

    Useful for tests and for reasoning — it routes exactly like the multiprocess runner but
    stays in one process, so results are directly comparable to a single driver.
    """

    def __init__(
        self, tickers: list[str], n_workers: int, feature_classes: list[type[Feature]]
    ) -> None:
        self.n_workers = n_workers
        assignment = assign_tickers(tickers, n_workers)
        self.drivers = {
            worker_id: EngineDriver(assignment[worker_id], feature_classes)
            for worker_id in range(n_workers)
        }

    def process(self, event: Event) -> FeatureVector | None:
        return self.drivers[worker_for(event.ticker, self.n_workers)].process(event)


def _worker_main(
    worker_id: int,
    tickers: list[str],
    in_queue: Queue[Event | None],
    out_queue: Queue[VectorSummary | _WorkerDone],
) -> None:
    """Worker process: drive an engine over assigned tickers until a sentinel arrives."""
    driver = EngineDriver(tickers, default_features())
    while True:
        event = in_queue.get()
        if event is None:
            break
        vector = driver.process(event)
        if vector is not None:
            n_valid = int(np.count_nonzero(~np.isnan(vector.values)))
            out_queue.put(
                VectorSummary(
                    worker_id=worker_id,
                    ticker=vector.ticker,
                    ts_ns=vector.ts_ns,
                    compute_ns=vector.compute_ns,
                    n_features=len(vector.columns),
                    n_valid=n_valid,
                )
            )
    out_queue.put(_WorkerDone(worker_id))


class ShardedRunner:
    """Spawns worker processes and routes events to them; drains their result summaries."""

    def __init__(self, tickers: list[str], n_workers: int) -> None:
        self.tickers = tickers
        self.n_workers = n_workers
        self.assignment = assign_tickers(tickers, n_workers)
        # 'spawn' avoids the fork-in-a-multi-threaded-process deadlock the router risks
        # (it runs a drain thread); workers are long-lived so startup cost is irrelevant.
        self._ctx = mp.get_context("spawn")
        self._in: list[Queue[Event | None]] = [self._ctx.Queue() for _ in range(n_workers)]
        self._out: Queue[VectorSummary | _WorkerDone] = self._ctx.Queue()
        self._procs: list[mp.process.BaseProcess] = []

    def start(self) -> None:
        for worker_id in range(self.n_workers):
            proc = self._ctx.Process(
                target=_worker_main,
                args=(worker_id, self.assignment[worker_id], self._in[worker_id], self._out),
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)

    def dispatch(self, event: Event) -> None:
        self._in[worker_for(event.ticker, self.n_workers)].put(event)

    def _send_sentinels(self) -> None:
        for in_queue in self._in:
            in_queue.put(None)

    def drain_forever(self, callback: Callable[[VectorSummary], None]) -> threading.Thread:
        """Background thread that calls ``callback(VectorSummary)`` until all workers finish."""

        def _loop() -> None:
            done = 0
            while done < self.n_workers:
                item = self._out.get()
                if isinstance(item, _WorkerDone):
                    done += 1
                else:
                    callback(item)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        return thread

    def run_source(self, source: EventSource) -> list[VectorSummary]:
        """Drive a bounded source to completion across the workers; return all summaries."""
        self.start()
        results: list[VectorSummary] = []
        collector = self.drain_forever(results.append)
        for event in source.iter_events():
            self.dispatch(event)
        self._send_sentinels()
        collector.join()
        for proc in self._procs:
            proc.join(timeout=5)
        return results

    def shutdown(self) -> None:
        self._send_sentinels()
        for proc in self._procs:
            proc.join(timeout=5)


def drain_now(out_queue: Queue[VectorSummary | _WorkerDone]) -> list[VectorSummary]:
    """Non-blocking drain of whatever summaries are currently available."""
    drained: list[VectorSummary] = []
    while True:
        try:
            item = out_queue.get_nowait()
        except queue.Empty:
            break
        if isinstance(item, VectorSummary):
            drained.append(item)
    return drained
