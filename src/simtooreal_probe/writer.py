"""
probe.writer — the TraceWriter: one async queue that fans events out to sinks.

Harvests the proven `uploads.metrics.MetricsQueue` worker pattern (background
daemon thread, batch-on-size-or-interval, drain-on-close) but generalises it to
the PolicyTraceEvent envelope and an ordered list of sinks.

Capture levels (the FPS escape hatch — hot-readable from PROBE_CAPTURE):
    off       emit nothing
    metrics   ITERATION grain only (cheapest; ~one event/iteration)
    episodes  ITERATION + EPISODE rollups (default)
    full      everything incl. per-STEP policy trace (the moat data; opt-in)

`emit()` never blocks the caller — events are queued and flushed by the worker.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import List, Optional, Sequence

from .trace import DimRegistry, Grain, PolicyTraceEvent
from .sinks import Sink

_FLUSH_INTERVAL = 2.0     # seconds between forced flushes
_BATCH_SIZE = 256         # flush when this many events are queued

# Ordered capture levels; an event passes if its grain is allowed at the level.
_LEVELS = ("off", "metrics", "episodes", "full")
_LEVEL_GRAINS = {
    "off": set(),
    "metrics": {Grain.ITERATION},
    "episodes": {Grain.ITERATION, Grain.EPISODE},
    "full": {Grain.ITERATION, Grain.EPISODE, Grain.STEP},
}


def resolve_capture(level: Optional[str] = None) -> str:
    lvl = (level or os.environ.get("PROBE_CAPTURE") or "episodes").strip().lower()
    return lvl if lvl in _LEVELS else "episodes"


class TraceWriter:
    """Async, non-blocking trace emitter.

    Parameters
    ----------
    run_id   : the run these events belong to.
    sinks    : ordered output destinations (LocalSink first, HttpSink after).
    capture  : one of off|metrics|episodes|full (env PROBE_CAPTURE overrides None).
    """

    def __init__(
        self,
        run_id: str,
        sinks: Sequence[Sink],
        *,
        capture: Optional[str] = None,
        flush_interval: float = _FLUSH_INTERVAL,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        self.run_id = run_id
        self._sinks: List[Sink] = list(sinks)
        self.capture = resolve_capture(capture)
        self._allowed = _LEVEL_GRAINS[self.capture]
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._q: "queue.Queue[PolicyTraceEvent]" = queue.Queue()
        self._stop = threading.Event()
        self._closed = False
        self._counts = {g: 0 for g in Grain}
        self._dropped = 0
        self._worker_error: Optional[Exception] = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="probe_trace")
        self._thread.start()

    # ── emit API ──────────────────────────────────────────────────────────────

    def set_capture(self, level: str) -> None:
        """Hot-change capture level mid-run (the FPS escape hatch)."""
        self.capture = resolve_capture(level)
        self._allowed = _LEVEL_GRAINS[self.capture]

    def emit(self, event: PolicyTraceEvent) -> None:
        if event.grain not in self._allowed:
            if event.grain.value == "step" and not getattr(self, "_warned_step_drop", False):
                self._warned_step_drop = True
                import warnings
                warnings.warn(
                    "probe is dropping STEP events (capture=%r). "
                    "Set PROBE_CAPTURE=full or probe.init(capture='full') / probe.watch() "
                    "to keep per-step traces." % (self.capture,),
                    stacklevel=2,
                )
            return
        self._q.put(event)

    def emit_many(self, events: Sequence[PolicyTraceEvent]) -> None:
        for ev in events:
            self.emit(ev)

    def register_dims(self, registry: DimRegistry) -> None:
        for s in self._sinks:
            try:
                s.write_dims(registry)
            except Exception:
                pass

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self, timeout: float = 30.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=timeout)
        # Deterministic safety net: if the worker didn't finish draining within
        # `timeout` (large `full`-capture backlog flushing to parquet/HTTP), drain
        # the rest synchronously here so queued STEP events are never lost when the
        # daemon thread is killed at interpreter exit.
        leftover: List[PolicyTraceEvent] = []
        while True:
            try:
                leftover.append(self._q.get_nowait())
            except queue.Empty:
                break
        if leftover:
            self._flush(leftover)
        for s in self._sinks:
            try:
                s.close()
            except Exception:
                pass

    @property
    def stats(self) -> dict:
        return {
            "run_id": self.run_id,
            "capture": self.capture,
            "emitted": {g.value: n for g, n in self._counts.items()},
            "dropped": self._dropped,
            "queue_depth": self._q.qsize(),
            "worker_alive": self._thread.is_alive(),
            "worker_error": str(self._worker_error) if self._worker_error else None,
        }

    def health_check(self) -> dict:
        """Return a human-readable worker-health summary. Safe to call from any thread."""
        s = self.stats
        ok = s["worker_alive"] and s["worker_error"] is None
        issues = []
        if not s["worker_alive"] and not self._closed:
            issues.append("worker thread died unexpectedly")
        if s["worker_error"]:
            issues.append(f"worker error: {s['worker_error']}")
        if s["queue_depth"] > 1000:
            issues.append(f"queue depth={s['queue_depth']} — sinks may be slow")
        if s["dropped"] > 0:
            issues.append(f"{s['dropped']} events dropped due to sink errors")
        return {"ok": ok, "issues": issues, **s}

    # ── worker ────────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        last_flush = time.time()
        batch: List[PolicyTraceEvent] = []
        try:
            while not self._stop.is_set():
                try:
                    batch.append(self._q.get(timeout=0.2))
                except queue.Empty:
                    pass
                now = time.time()
                if batch and (len(batch) >= self._batch_size or now - last_flush >= self._flush_interval):
                    self._flush(batch)
                    batch = []
                    last_flush = now
            # Drain on shutdown.
            while True:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if batch:
                self._flush(batch)
        except Exception as exc:  # noqa: BLE001
            # Record the crash so health_check() can surface it. The daemon
            # thread dying here means all subsequent emit() calls queue into
            # an undrainable queue — close() will drain synchronously when called.
            self._worker_error = exc

    def _flush(self, batch: List[PolicyTraceEvent]) -> None:
        for ev in batch:
            self._counts[ev.grain] = self._counts.get(ev.grain, 0) + 1
        for s in self._sinks:
            try:
                s.write(batch)
            except Exception:
                # Count once per sink failure, not once per event per sink —
                # events are NOT lost from other sinks just because one failed.
                self._dropped += len(batch)
