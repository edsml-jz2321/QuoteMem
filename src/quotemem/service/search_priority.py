"""Search-first scheduling of one interactive service process (plugin profile).

Python runs one thread at a time. A Search that runs while an Add owner writes
shares that core with it, and a full garbage collection started by any thread
stops every thread. In a plugin, the user waits for the Search, while the Add
of earlier turns runs unseen in the background. ``SearchActivity`` gives the
Search the core for its short run:

- **Write scopes yield.** A background Add worker that is about to enter an
  outermost Writer scope (``Writer.write_scope``) first waits until every
  Search that was already running when it arrived has finished. Searches that
  start later do not extend the wait, and the wait is bounded by
  ``SEARCH_YIELD_SECONDS``, so a stream of Searches never starves an Add. A
  write scope that has started always finishes: its fences, lease checks and
  writes are unchanged. Only threads marked as background work wait
  (``mark_background``, or a bench worker named ``quotemem-bench-<n>``); the lease
  heartbeat, the discovery pass, the Search threads and every other thread
  never do.
- **Collection waits.** While a Search runs, automatic garbage collection is
  paused, and it resumes when the last running Search ends: a collection that
  became due during the Search runs then. A pause never lasts longer than
  ``MAX_GC_PAUSE_SECONDS``, so continuous Search load still collects.

Nothing here decides a storage effect. Waiting only delays a write scope; it
never skips one, and ``bench`` serving never constructs this object.
"""

from __future__ import annotations

import gc
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

SEARCH_YIELD_SECONDS = 0.5
"""Longest wait of one background write scope for the Searches already running."""

MAX_GC_PAUSE_SECONDS = 5.0
"""Longest time automatic collection stays paused while Searches keep running."""

_WORKER_NAME = re.compile(r"quotemem-bench-\d+")
"""The bench Add work pool's worker threads (``service/bench_workers.py``)."""

_local = threading.local()


def mark_background() -> None:
    """Mark the calling thread as background work that yields to running Searches."""
    _local.background = True


def _background() -> bool:
    if getattr(_local, "background", False):
        return True
    return _WORKER_NAME.fullmatch(threading.current_thread().name) is not None


def in_search() -> bool:
    """Whether the calling thread is running a Search inside ``SearchActivity.searching``."""
    return getattr(_local, "searching", 0) > 0


class SearchActivity:
    """Track the running Searches of one process and give them the core first.

    Args:
        pause_collection: Pause automatic garbage collection while any Search runs.
        yield_seconds: Bound on one background write scope's wait.
        max_pause_seconds: Bound on one continuous collection pause.

    Construction starts no thread and changes no collector state.
    """

    def __init__(
        self, *, pause_collection: bool = True, yield_seconds: float = SEARCH_YIELD_SECONDS,
        max_pause_seconds: float = MAX_GC_PAUSE_SECONDS,
    ) -> None:
        if not (yield_seconds >= 0 and max_pause_seconds >= 0):
            raise ValueError("Search priority bounds must be nonnegative")
        self._condition = threading.Condition(threading.Lock())
        self._running: set[int] = set()
        self._started = 0
        self._pause_collection = pause_collection
        self._yield_seconds = yield_seconds
        self._max_pause = max_pause_seconds
        self._paused_at: float | None = None
        self.yields = 0
        """Background write scopes that waited for a running Search."""
        self.yield_seconds_total = 0.0
        """Total time those write scopes waited."""

    @property
    def running(self) -> int:
        """The number of Searches running now."""
        with self._condition:
            return len(self._running)

    @contextmanager
    def searching(self) -> Iterator[None]:
        """Run one Search on the calling thread with priority over background writes.

        Yields:
            None while the Search runs; nested entry on the same thread is one Search.
        """
        depth = getattr(_local, "searching", 0)
        if depth:
            _local.searching = depth + 1
            try:
                yield
            finally:
                _local.searching = depth
            return
        with self._condition:
            self._started += 1
            number = self._started
            self._running.add(number)
            self._pause(time.monotonic())
        _local.searching = 1
        try:
            yield
        finally:
            _local.searching = 0
            with self._condition:
                self._running.discard(number)
                if not self._running:
                    self._resume()
                self._condition.notify_all()

    def _pause(self, now: float) -> None:
        """Pause automatic collection for the running Searches, within its bound."""
        if not self._pause_collection:
            return
        if self._paused_at is None:
            if gc.isenabled():
                gc.disable()
                self._paused_at = now
        elif now - self._paused_at >= self._max_pause:
            # Continuous Search load: let the collector run again until the
            # next time no Search runs.
            gc.enable()
            self._paused_at = None

    def _resume(self) -> None:
        if self._paused_at is not None:
            self._paused_at = None
            gc.enable()

    def yield_to_searches(self) -> None:
        """Wait, on a background thread, for the Searches already running to finish.

        Returns at once on a thread that is not background work or that runs a
        Search itself, and when no Search runs. Otherwise waits until every
        Search that started before this call has finished, or for at most the
        configured bound. Call it before taking the Writer mutex, never while
        owning it.
        """
        if in_search() or not _background():
            return
        with self._condition:
            if not self._running:
                return
            horizon = self._started
            began = time.monotonic()
            deadline = began + self._yield_seconds
            while self._running and min(self._running) <= horizon:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            self.yields += 1
            self.yield_seconds_total += time.monotonic() - began
