"""The Writer's reentrant exclusion, delivery priority, and content-free progress."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from operator import attrgetter
from typing import Any, Protocol, cast

from quotemem.core.gc_safe import CheckedLock
from quotemem.core.timing import measure

DELIVERY_HANDOFF_SECONDS = 0.25
"""Longest deferral of other contenders while a reserved FIFO head finishes one attempt."""


class ProgressCancellation(Protocol):
    """A nonblocking cancellation state and a scoped notification subscription.

    Cancellation is monotonic for the operation and cannot reset after notice.
    Producers publish cancellation before invoking wake callbacks and invoke
    them outside producer locks. The cancelled read must not block: the waiter
    checks it while holding its own bookkeeping condition. Subscription entry
    and cleanup run outside that condition and without owning the Writer mutex.
    """

    @property
    def cancelled(self) -> bool:
        """Read cancellation without blocking or invoking callbacks."""
        ...

    def on_cancel(self, wake: Callable[[], None]) -> AbstractContextManager[None]:
        """Subscribe a wake notification until the returned context exits."""
        ...


class WriterMutex:
    """Wrap one RLock with release notifications and conservative Writer priority.

    Every blocking first acquisition takes priority over new ordinary nonblocking
    delivery attempts. This includes read-only holders; no caller-supplied role
    decides priority. Recursive entry never waits behind another thread.
    Ordering between blocking contenders remains unspecified, as with RLock.
    The application's separate queue owns FIFO delivery admission.

    The application's ready FIFO head may hold one thread-bound delivery
    reservation for exactly one guarded attempt (ticket priority: an
    ordinary delivery never overtakes the ready head, and lock contention only
    consumes time). While reserved, the head's own nonblocking attempt succeeds
    whenever the RLock is free, regardless of registered blocking contenders,
    and a bounded handoff window defers new acquisitions by other threads until
    the reservation ends or the window expires. The window never preempts a
    current holder, never blocks the reserved thread, and is capped by
    DELIVERY_HANDOFF_SECONDS, so a destructive writer or barrier waits at most
    that long behind one delivery. Ordinary nonblocking attempts stay refused
    while any blocking contender waits or a window is open.

    Availability progress advances on outermost release or withdrawal of the
    last blocking waiter. It is neither a write counter nor a storage authority.
    Waiting executes no storage work and must happen after releasing borrowed
    content. Cancellation registration runs outside the bookkeeping condition.

    A garbage collection may run any weak-reference callback or finalizer
    inside a bookkeeping section. Such a callback must never take this mutex
    (``quotemem.core.gc_safe``); one that does raises ``RuntimeError`` from the
    bookkeeping ``CheckedLock`` instead of waiting for its own thread forever.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Bound once: asking ownership allocates nothing inside a bookkeeping section.
        self._is_owned = cast(Callable[[], bool], attrgetter("_is_owned")(self._lock))
        # Condition takes any lock with acquire, release and _is_owned; its stub
        # names only the stdlib lock types.
        self._condition = threading.Condition(cast(Any, CheckedLock()))
        self._waiting = 0
        self._generation = 0
        self._sequence = 0
        self._reserved: int | None = None
        self._handoff_until: float | None = None
        # Recursion depth of the owning thread; changed only under the condition.
        self._depth = 0
        self._outer_release: Callable[[bool], None] | None = None
        self._before_contend: Callable[[], None] | None = None

    @property
    def progress_generation(self) -> int:
        """Capture progress before a nonblocking attempt to avoid missed wakeups."""
        with self._condition:
            return self._generation

    @property
    def sequence(self) -> int:
        """Return the critical-section sequence: odd while any thread owns the mutex.

        The value advances on every outermost acquisition and every outermost
        release, so two equal even readings prove that no critical section
        began or ended between them (a seqlock reader's consistency check).
        """
        with self._condition:
            return self._sequence

    def _take(self) -> bool:
        """Try the raw lock under the condition; count an outermost acquisition."""
        owned = self._is_owned()
        taken = self._lock.acquire(False)
        if taken:
            self._depth += 1
            if not owned:
                self._sequence += 1
        return taken

    @property
    def reserved(self) -> bool:
        """Report whether a delivery head currently holds the reservation."""
        with self._condition:
            return self._reserved is not None

    def _deferred(self, ident: int, now: float) -> bool:
        """Report whether another thread's acquisition must yield to the reserved head."""
        return (
            self._reserved is not None and self._reserved != ident
            and self._handoff_until is not None and now < self._handoff_until
        )

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Acquire exclusion, prioritizing blocking contenders over new delivery.

        Args:
            blocking: Whether to wait for ownership; recursive entry is immediate.
            timeout: RLock-compatible seconds, with -1 meaning no timeout.

        Returns:
            True when acquired, False on contention or timeout. A nonblocking
            attempt by the reserved delivery head is refused only while the
            RLock is actually owned by another thread.

        Raises:
            ValueError: For a negative nondefault timeout, NaN, or a timeout on
                a nonblocking call.
            OverflowError: If timeout exceeds threading.TIMEOUT_MAX.
        """
        if math.isnan(timeout) or timeout < 0 and timeout != -1:
            raise ValueError("Invalid lock timeout")
        if not blocking and timeout != -1:
            raise ValueError("Cannot specify a timeout for a nonblocking acquire")
        if timeout > threading.TIMEOUT_MAX:
            raise OverflowError("Lock timeout exceeds threading.TIMEOUT_MAX")
        hook = self._before_contend
        if blocking and hook is not None and not self._is_owned():
            # Outside every bookkeeping section: the hook may wait (``before_contend``).
            hook()
        deadline = None if timeout == -1 else time.monotonic() + timeout
        ident = threading.get_ident()
        with self._condition:
            if self._is_owned():
                return self._take()
            if not blocking:
                if self._reserved == ident:
                    return self._take()
                if self._waiting or self._deferred(ident, time.monotonic()):
                    return False
                return self._take()
            self._waiting += 1
            acquired = False
            try:
                while True:
                    now = time.monotonic()
                    deferred = self._deferred(ident, now)
                    if not deferred and self._take():
                        acquired = True
                        return True
                    remaining = None if deadline is None else deadline - now
                    if remaining is not None and remaining <= 0:
                        return False
                    wait = remaining
                    if deferred and self._handoff_until is not None:
                        window = self._handoff_until - now
                        wait = window if wait is None else min(wait, window)
                    with measure("queue"):
                        self._condition.wait(None if wait is None else max(0.0, wait))
            finally:
                self._waiting -= 1
                if not acquired and not self._waiting:
                    self._generation += 1
                    self._condition.notify_all()

    def release(self) -> None:
        """Release one level and notify progress only after outermost release.

        An outermost release by a thread other than the reserved delivery head
        opens or refreshes the bounded handoff window for that head. Before an
        outermost release, while the section still owns the mutex, the
        ``on_outermost_release`` callback runs; the release happens even if it
        raises.
        """
        hook = self._outer_release
        try:
            if hook is not None and self._depth == 1 and self._is_owned():
                hook(False)
        finally:
            with self._condition:
                self._lock.release()
                self._depth -= 1
                if not self._is_owned():
                    self._generation += 1
                    self._sequence += 1
                    if self._reserved is not None and self._reserved != threading.get_ident():
                        self._handoff_until = time.monotonic() + DELIVERY_HANDOFF_SECONDS
                    self._condition.notify_all()

    def locked(self) -> bool:
        """Report whether any thread owns the mutex, without waiting for it.

        ``threading.Condition`` exports its lock's ``locked`` from Python 3.14
        on. The answer is the parity of the critical-section sequence, which
        is odd exactly while a thread owns the mutex, so asking takes neither
        the mutex nor its bookkeeping condition. A ``Condition.wait`` leaves
        the mutex unowned until it reacquires it.

        Returns:
            bool: Whether the mutex is owned, by this thread or another.
        """
        return self._sequence % 2 == 1

    def on_outermost_release(self, hook: Callable[[bool], None] | None) -> None:
        """Run ``hook`` at the end of every critical section, before its release.

        The hook runs on the releasing thread while it still owns the mutex, so
        hooks of successive sections never overlap. Its argument is True when a
        ``Condition.wait`` suspends the section rather than ending it. It must
        not raise or block, and may take only a leaf lock of its own; None
        removes it.
        """
        self._outer_release = hook

    def before_contend(self, hook: Callable[[], None] | None) -> None:
        """Run ``hook`` before every blocking first acquisition, outside the mutex.

        The interactive (plugin) service installs ``SearchActivity.yield_to_searches``
        here, so background Add work lets the running Searches finish before it
        takes the Writer (``service/search_priority.py``). The hook may wait for a
        bounded time but must not take this mutex or raise. Nonblocking and
        recursive acquisitions never run it. None removes it.
        """
        self._before_contend = hook

    @contextmanager
    def delivery_reservation(self) -> Iterator[None]:
        """Reserve ticket priority for the FIFO head's one guarded delivery attempt.

        The caller must be the application's current FIFO head and must not own
        the mutex. Entry opens the bounded handoff window at once, so a mutex
        that is free or is released during the attempt is not taken by a newly
        arriving ordinary contender before the head's nonblocking attempt. Exit
        always clears the reservation and wakes deferred contenders. The head
        must still release its reader guard and content before waiting for
        progress or for its turn after the attempt.

        Yields:
            None while this thread holds the reservation.

        Raises:
            RuntimeError: If this thread owns the mutex or another reservation
                is already held; the FIFO queue admits exactly one head.
        """
        ident = threading.get_ident()
        with self._condition:
            if self._is_owned():
                raise RuntimeError("Delivery reservation requires releasing the Writer mutex")
            if self._reserved is not None:
                raise RuntimeError("Another delivery head already holds the reservation")
            self._reserved = ident
            self._handoff_until = time.monotonic() + DELIVERY_HANDOFF_SECONDS
        try:
            yield
        finally:
            with self._condition:
                self._reserved = None
                self._handoff_until = None
                self._condition.notify_all()

    def __enter__(self) -> bool:
        """Acquire with the same blocking semantics as RLock context entry."""
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        """Release even when the protected operation raises."""
        self.release()

    def _release_save(self) -> object:
        # Condition.wait must release every recursion level as one outer release.
        hook = self._outer_release
        if hook is not None and self._is_owned():
            with suppress(Exception):
                hook(True)
        with self._condition:
            state = cast(Callable[[], object], attrgetter("_release_save")(self._lock))()
            depth, self._depth = self._depth, 0
            self._generation += 1
            self._sequence += 1
            self._condition.notify_all()
            return state, depth

    def _acquire_restore(self, saved: object) -> None:
        # Join ordinary blocking priority, then restore recursion while bookkeeping
        # excludes other wrapper acquisitions. The temporary raw release is private.
        state, depth = cast(tuple[object, int], saved)
        self.acquire()
        with self._condition:
            self._lock.release()
            cast(Callable[[object], None], attrgetter("_acquire_restore")(self._lock))(state)
            self._depth = depth

    def _wake_progress_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def wait_for_progress(
        self, after: int, *, deadline: float, cancellation: ProgressCancellation | None = None,
    ) -> bool:
        """Wait for availability progress without retaining the Writer mutex.

        Args:
            after: Generation captured before the failed nonblocking attempt.
            deadline: Absolute finite time.monotonic deadline.
            cancellation: Optional operation state with a scoped wake subscription.

        Returns:
            True once a later generation exists, including release or priority
            withdrawal preceding this call; False on cancellation or deadline.
            Cancellation takes precedence over already observed progress.

        Raises:
            RuntimeError: If this thread still owns the Writer mutex.
            ValueError: If the deadline or generation is invalid.
        """
        if self._is_owned():
            raise RuntimeError("Progress waits require releasing the Writer mutex")
        if not math.isfinite(deadline) or after < 0:
            raise ValueError("Invalid progress generation or deadline")
        subscription = (
            nullcontext() if cancellation is None
            else cancellation.on_cancel(self._wake_progress_waiters)
        )
        # Subscription hooks can notify synchronously or acquire their own
        # locks. Neither entry nor cleanup may run under this condition.
        with subscription:
            with self._condition:
                if after > self._generation:
                    raise ValueError("Progress generation is from the future")
                while True:
                    if cancellation is not None and cancellation.cancelled:
                        return False
                    if self._generation != after:
                        return True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    with measure("queue"):
                        self._condition.wait(min(remaining, threading.TIMEOUT_MAX))
