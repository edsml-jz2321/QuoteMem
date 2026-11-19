"""Coalesce successful platform traffic outside the request's persistence path."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import TypeAdapter

from aml.core.semantic import SystemTimestamp

_TIMESTAMP: TypeAdapter[int] = TypeAdapter(SystemTimestamp)
_FLUSH_INTERVAL_SECONDS = 60.0


@dataclass(frozen=True)
class TrafficState:
    """One detached scheduling observation; it grants no persistence authority.

    ``acknowledged_at`` is the fully acknowledged watermark. ``pending_at``
    includes any active attempt until it succeeds, and may already contain newer
    traffic. ``next_attempt_at`` is the end of the interval that still paces the
    next attempt, or None when none does. Scheduling times use the coalescer's
    monotonic clock, not epoch time.
    """

    acknowledged_at: int
    pending_at: int | None
    in_flight: bool
    next_attempt_at: float | None
    sampled_at: float

    @property
    def settled(self) -> bool:
        """Whether this observation contains neither pending nor active work."""
        return self.pending_at is None and not self.in_flight

    @property
    def due(self) -> bool:
        """Whether the sole background caller should try ``flush_due`` now."""
        return self.wait_seconds == 0.0

    @property
    def wait_seconds(self) -> float | None:
        """Return the throttle delay, or None while empty or already active.

        An active flush needs completion notification, not another persistence
        caller. The managed worker is the sole caller and owns that completion.
        """
        if self.pending_at is None or self.in_flight:
            return None
        if self.next_attempt_at is None:
            return 0.0
        return max(0.0, self.next_attempt_at - self.sampled_at)


class JobTraffic:
    """Keep one content-free high watermark for one already bound bench job.

    The composition owner supplies the sole Writer's monotonic traffic CAS and
    calls ``flush_due`` from its background lifecycle. Construction starts no
    thread. Request success only calls ``observe``; it performs no persistence
    or user callback. This timestamp is activity, never a job-completion signal.

    R20 paces the CAS at most once per 60-second interval while traffic can
    still arrive. Once the owner has sealed observations at shutdown it calls
    ``finish``: the last watermark is then attempted at once (the terminal
    flush) instead of after an interval that a successful attempt began.

    Args:
        persist: Persist at least the supplied timestamp for the bound job, then
            return. An exception must leave acknowledgement to a future attempt.
        acknowledged_at: Last durable platform timestamp read during job binding.
        monotonic: Process monotonic clock used exclusively for flush throttling.

    Raises:
        ValidationError: If acknowledged_at violates the shared timestamp type.
    """

    def __init__(
        self,
        persist: Callable[[int], None],
        *,
        acknowledged_at: int,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._persist = persist
        self._monotonic = monotonic
        self._acknowledged = _TIMESTAMP.validate_python(acknowledged_at, strict=True)
        self._pending: int | None = None
        self._lock = threading.Lock()
        self._flushing = False
        self._next_attempt: float | None = None
        self._failed = False
        self._finished = False

    def observe(self, at: int) -> None:
        """Atomic-max one successful Add or successfully enqueued Search timestamp.

        Args:
            at: Shared SystemTimestamp in epoch milliseconds, captured at success.

        Returns:
            None. Older, repeated and already acknowledged observations are no-ops.

        Raises:
            ValidationError: If at violates the shared strict timestamp type.

        The caller must not use this for request arrival, failures or health
        checks. The lock only protects integer state and never surrounds I/O.
        """
        at = _TIMESTAMP.validate_python(at, strict=True)
        with self._lock:
            if at > self._acknowledged:
                self._pending = max(at, self._pending if self._pending is not None else at)

    def state(self) -> TrafficState:
        """Observe pending, acknowledged, active and due state without I/O.

        Returns:
            TrafficState: A frozen observation, never a drain or future-work guarantee.

        Raises:
            ValueError: If the monotonic clock returns a non-finite value.
            Exception: If the supplied trusted clock fails.
        """
        now = self._monotonic()
        if not math.isfinite(now):
            raise ValueError("Traffic scheduling requires a finite monotonic clock")
        with self._lock:
            return TrafficState(
                self._acknowledged, self._pending, self._flushing, self._pacing(), now,
            )

    def finish(self) -> None:
        """End observations: attempt the last watermark without waiting out the interval.

        The sole owner calls this after sealing observations at shutdown, once
        HTTP and controller drain have completed, so the pending watermark is the
        last one. It is then due at once instead of at the end of an interval that
        a successful attempt began. An interval that a failed attempt began still
        paces the retry, and attempts never overlap. Idempotent; performs no I/O.
        """
        with self._lock:
            self._finished = True

    def _pacing(self) -> float | None:
        """Return the end of the interval that paces the next attempt; caller holds the lock."""
        if self._finished and not self._failed:
            return None
        return self._next_attempt

    def flush_due(self) -> bool:
        """Attempt one background flush without blocking traffic observations.

        Until ``finish``, at most one attempt starts in each 60-second monotonic
        interval, including failures; afterwards only a failed attempt's interval
        paces the next. Concurrent callers never overlap persistence. A success
        clears only acknowledged observations; newer traffic and failed writes
        remain pending. After shutdown admission drains, the lifecycle must retain
        this object and its Writer until the final background flush settles.

        Returns:
            True when a due pending watermark was persisted; False for no work,
            a throttle interval, or another active flush.

        Raises:
            ValueError: If the monotonic clock returns a non-finite value.
            Exception: The persistence failure, with pending observations retained.
        """
        now = self._monotonic()
        if not math.isfinite(now):
            raise ValueError("Traffic flush requires a finite monotonic clock")
        with self._lock:
            pacing = self._pacing()
            if self._flushing or self._pending is None or (pacing is not None and now < pacing):
                return False
            observed = self._pending
            self._flushing = True
            self._next_attempt = now + _FLUSH_INTERVAL_SECONDS
        persisted = False
        try:
            self._persist(observed)
            persisted = True
            with self._lock:
                self._acknowledged = max(self._acknowledged, observed)
                if self._pending <= self._acknowledged:
                    self._pending = None
            return True
        finally:
            with self._lock:
                self._failed = not persisted
                self._flushing = False
