"""Interactive Search's ``recall`` rows, written after the response in batches.

A bench Search writes its ``recall`` delivery row inside the Writer critical
section that also rechecks and enqueues the response (``Writer.record_search_delivery``).
That section waits for any write in progress, so a background Add can
hold up a Search, and the row write itself is about a
fifth of a Search's CPU. An interactive Search (plugin profile) delivers without
the Writer mutex (``SnapshotReadGuard.deliver_search_published``) and hands its
row to ``DeferredRecallRows``: a flusher thread writes the queued rows in one
Writer scope per batch, shortly after the responses.

A forget or privacy erasure removes the ``recall`` rows that named what it erased
(``store/forget_run.py``). A queued row could name what an erasure that began
after its Search removed, so each row is written only when, inside the batch's
Writer scope, its scope's erase and consent epochs still equal the ones its
Search delivered under and no tomb, erasure or maintenance fences the scope. A
row that fails this is dropped and counted, never written: its Search delivered
before the erasure, and the erasure would have removed it. A row written earlier
is durable when the erasure admits, so the erasure's own scan finds and removes it.

What this costs, and what the plugin profile documents: a crash or a drain that
misses its deadline loses the rows still queued (at most a short flush delay of
Searches), and a Search that overlapped an erasure of its scope leaves no row.
The responses themselves are unchanged.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quotemem.core.errors import QuoteMemError
from quotemem.service.search_priority import mark_background
from quotemem.store.domain import ReplayableRecallRowV1

if TYPE_CHECKING:
    from quotemem.store.read_model import StoreReadModel
    from quotemem.store.writer import Writer

FLUSH_DELAY_SECONDS = 0.5
"""How long the flusher waits after a row arrives, so close Searches share one write.

A write of one row costs about as much as a write of a few (25-30 ms at 1,000
facts with frozen providers, mostly the Writer scope itself), so a user's
back-to-back Searches should share one. The delay is also the window of rows a
crash can lose.
"""

MAX_BATCH_ROWS = 256
"""Rows written in one Writer scope."""

MAX_QUEUED_ROWS = 4096
"""Rows that may wait; a full queue makes the enqueuing Search write its batch itself."""


@dataclass(frozen=True, slots=True)
class _Queued:
    row: ReplayableRecallRowV1
    erase_epoch: int
    consent_epoch: int


class DeferredRecallRows:
    """Queue interactive Search delivery rows and write them off the request path.

    Args:
        writer: The sole Writer the rows belong to.
        publisher: That Writer's registered read model.
        delay: Seconds between a row's arrival and its batch write.

    Construction starts no thread; the first ``enqueue`` starts the flusher.
    """

    def __init__(
        self, writer: Writer, publisher: StoreReadModel, *, delay: float = FLUSH_DELAY_SECONDS,
    ) -> None:
        self._writer = writer
        self._publisher = publisher
        self._delay = delay
        self._condition = threading.Condition(threading.Lock())
        self._queue: deque[_Queued] = deque()
        self._flush_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self.written = 0
        """Rows written."""
        self.dropped = 0
        """Rows dropped because an erasure of their scope began before their write."""
        self.failed: QuoteMemError | None = None
        """The first write failure; later rows keep being attempted."""

    def enqueue(self, row: ReplayableRecallRowV1, *, erase_epoch: int, consent_epoch: int) -> None:
        """Queue one delivered Search's row without taking the Writer mutex.

        Args:
            row: The complete strict delivery row, with a fresh identity.
            erase_epoch: The scope's erase epoch its delivery was checked under.
            consent_epoch: The scope's consent epoch its delivery was checked under.

        Raises:
            QuoteMemError: ``binding_mismatch`` for another tenant or a missing latency;
                ``service_unhealthy`` once closed.
        """
        if row.tenant_id != self._writer.tenant_id or row.latency_ms is None:
            raise QuoteMemError("binding_mismatch", "Search delivery row lacks its exact binding")
        with self._condition:
            if self._closed:
                raise QuoteMemError("service_unhealthy", "Search delivery rows are closed")
            self._queue.append(_Queued(row, erase_epoch, consent_epoch))
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="quotemem-recall-rows", daemon=True,
                )
                self._thread.start()
            self._condition.notify_all()

    @property
    def pending(self) -> int:
        """Rows queued and not yet written or dropped."""
        with self._condition:
            return len(self._queue)

    def flush(self) -> int:
        """Write every row queued now, in batches; return how many were written.

        Takes the Writer mutex once per batch. Call it outside any Writer scope.
        """
        written = 0
        with self._flush_lock:
            while True:
                with self._condition:
                    batch = [self._queue.popleft()
                             for _ in range(min(MAX_BATCH_ROWS, len(self._queue)))]
                if not batch:
                    return written
                written += self._write(batch)

    def close(self, deadline: float) -> bool:
        """Refuse new rows, write the queued ones and stop the flusher by ``deadline``.

        Returns:
            bool: True when every queued row was written or dropped and the flusher stopped.
        """
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                return False
        self.flush()
        return self.pending == 0

    def _run(self) -> None:
        mark_background()
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if not self._queue and self._closed:
                    return
                # Close Searches share one write; only ``close`` ends this wait early.
                until = time.monotonic() + self._delay
                while not self._closed and (remaining := until - time.monotonic()) > 0:
                    self._condition.wait(remaining)
            try:
                self.flush()
            except Exception:  # noqa: BLE001 - recorded in ``failed``; the next batch retries.
                time.sleep(max(self._delay, 0.01))

    def _write(self, batch: list[_Queued]) -> int:
        from quotemem.store.writer import RowWrite

        writer = self._writer
        try:
            with writer.write_scope():
                kept: list[RowWrite] = []
                for queued in batch:
                    if self._erased(queued):
                        self.dropped += 1
                    else:
                        kept.append(RowWrite("recall", queued.row))
                if kept:
                    _, plan = writer.plan_rows(kept, None)
                    writer.write_plan(plan)
                self.written += len(kept)
                return len(kept)
        except QuoteMemError as error:
            if self.failed is None:
                self.failed = error
            raise

    def _erased(self, queued: _Queued) -> bool:
        """Whether an erasure of the row's scope began after its Search delivered.

        Runs inside the batch's Writer scope, where the published root is the
        committed one (``StoreReadModel.publish_reception``).
        """
        row = queued.row
        fence = self._publisher.committed_scope_fence(row.scope_id)
        if fence is None or fence.tomb or fence.active_erasure or fence.maintenance:
            return True
        epochs = (queued.erase_epoch, queued.consent_epoch)
        if (fence.erase_epoch, fence.consent_epoch) != epochs:
            return True
        return (
            self._writer.get_counter(f"erase_epoch:{row.scope_id}"),
            self._writer.get_counter(f"consent_epoch:{row.scope_id}"),
        ) != epochs
