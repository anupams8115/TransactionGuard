"""Non-blocking request metrics.

Writing a metrics row synchronously inside the request path is a classic mistake: on SQLite it
can deadlock against the request's own open write transaction, and on any database it adds
latency to every call. Instead, requests drop a tiny record into an in-memory queue and a
background thread writes batches. Records are lost if the process crashes before a flush; that is
acceptable for operational metrics, but it would not be acceptable for predictions.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from transactionguard.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RequestRecord:
    route: str
    method: str
    status_code: int
    latency_ms: float
    created_at: datetime


class MetricsBuffer:
    def __init__(self, flush_every_s: float = 2.0, max_queue: int = 50_000):
        self._q: queue.Queue[RequestRecord] = queue.Queue(maxsize=max_queue)
        self._flush_every = flush_every_s
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.dropped = 0

    def record(self, route: str, method: str, status_code: int, latency_ms: float) -> None:
        try:
            self._q.put_nowait(RequestRecord(route, method, status_code, latency_ms, datetime.now(UTC)))
        except queue.Full:
            self.dropped += 1  # never block a request for metrics
        self._ensure_thread()

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="metrics-writer", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        stop = threading.Event()
        while not stop.wait(self._flush_every):
            self.flush()

    def flush(self) -> int:
        """Write everything queued so far (also called before metrics are read)."""
        from transactionguard.db.models import RequestLog
        from transactionguard.db.session import session_scope

        with self._lock:
            batch: list[RequestRecord] = []
            while True:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if not batch:
                return 0
            try:
                with session_scope() as db:
                    db.add_all(
                        RequestLog(
                            route=r.route,
                            method=r.method,
                            status_code=r.status_code,
                            latency_ms=r.latency_ms,
                            created_at=r.created_at,
                        )
                        for r in batch
                    )
            except Exception as exc:
                log.warning("metrics_flush_failed", n=len(batch), error=type(exc).__name__)
                return 0
            return len(batch)


METRICS = MetricsBuffer()
