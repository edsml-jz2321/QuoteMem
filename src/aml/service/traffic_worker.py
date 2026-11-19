"""Own the one background traffic flusher without owning Writer or service I/O."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from aml.service.traffic import JobTraffic, TrafficState


class TrafficFailure(StrEnum):
    """Closed content-free failures; exception messages and tracebacks are not retained."""

    PERSISTENCE = "persistence_failed"
    OBSERVATION = "observation_failed"
    LATE_OBSERVATION = "observation_after_seal"
    CALLBACK = "failure_callback_failed"
    WORKER = "worker_failed"


@dataclass(frozen=True)
class TrafficDrainResult:
    """Report one drain attempt without authorizing resource closure on timeout.

    ``settled`` requires a joined worker and no pending or active traffic. A prior
    failure remains visible even after successful retry; it cannot restore service
    readiness. ``traffic`` is None only when its trusted scheduling clock failed.
    """

    outcome: Literal["settled", "pending_timeout", "active_timeout", "join_timeout", "failed"]
    joined: bool
    traffic: TrafficState | None
    failure: TrafficFailure | None


class JobTrafficWorker:
    """Run and drain one already-bound job's existing coalescer.

    Args:
        traffic: Sole watermark owner, bound to the actual Writer's complete
            ``flush_job_traffic`` acknowledgement. No other caller may flush it
            while this worker owns it.
        clock_ms: Trusted, nonblocking SystemTimestamp clock for HTTP success.
        on_failure: Required nonblocking, content-free failure notification. The
            composition binds service admission withdrawal and also includes
            ``require_ready`` in its readiness verifier. It must not perform I/O.

    Construction starts no thread. Start after actual job binding and before
    activating service admission. The worker never enters ServiceLifetime
    admission: pending flushes must still be possible after that admission stops.
    Stop listeners and drain complete HTTP workers, including their success
    callbacks, before calling ``drain``. This object closes no Writer, HB or owner.
    """

    def __init__(
        self,
        traffic: JobTraffic,
        *,
        clock_ms: Callable[[], int],
        on_failure: Callable[[TrafficFailure], None],
    ) -> None:
        if not callable(clock_ms) or not callable(on_failure):
            raise TypeError("Traffic clocks and failure notification must be callable")
        self._traffic = traffic
        self._clock_ms = clock_ms
        self._on_failure = on_failure
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._sealed = False
        self._failure: TrafficFailure | None = None
        self._fatal = False

    def _fail(self, reason: TrafficFailure, *, fatal: bool = False) -> None:
        with self._lock:
            notify = self._failure is None
            self._failure = reason
            self._fatal |= fatal
        self._wake.set()
        if notify:
            try:
                self._on_failure(reason)
            except BaseException:
                with self._lock:
                    self._failure = TrafficFailure.CALLBACK
                    self._fatal = True
                self._wake.set()

    def start(self) -> None:
        """Start the sole owned, non-daemon thread exactly once.

        Raises:
            RuntimeError: If already started, sealed, failed or unable to start.
                Start failure withdraws readiness through the failure callback.
        """
        failed = False
        with self._lock:
            if self._thread is not None or self._sealed or self._failure is not None:
                raise RuntimeError("Traffic worker cannot be started again")
            thread = threading.Thread(target=self._run, name="aml-job-traffic", daemon=False)
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._sealed = True
                self._fatal = True
                failed = True
        if failed:
            self._fail(TrafficFailure.WORKER, fatal=True)
            raise RuntimeError("Traffic worker could not start") from None

    def observe_success(self) -> None:
        """Record and wake for an accepted HTTP success without persistence or I/O.

        The callback receives no request, credentials or response. In-flight HTTP
        work may still observe after readiness failure until the composition
        drains it. Invalid clocks or calls before start/after seal latch a safe
        failure instead of changing already accepted HTTP outcomes or reopening
        observations. Failure notification is outside both state locks.

        Returns:
            None. This callback never propagates clock or notification failures.
        """
        reason: TrafficFailure | None = None
        with self._lock:
            if self._thread is None or self._sealed:
                reason = TrafficFailure.LATE_OBSERVATION
            else:
                try:
                    self._traffic.observe(self._clock_ms())
                except BaseException:
                    reason = TrafficFailure.OBSERVATION
        self._wake.set()
        if reason is not None:
            self._fail(reason, fatal=True)

    def require_ready(self) -> None:
        """Reject unavailable or failed worker state without reading storage.

        Raises:
            RuntimeError: If not started, draining, failed or unexpectedly stopped.
                A later successful flush never erases an earlier readiness failure.
        """
        with self._lock:
            if self._thread is None or self._sealed or self._failure is not None \
                    or not self._thread.is_alive():
                raise RuntimeError("Job traffic worker is not ready")

    def _run(self) -> None:
        crashed = False
        try:
            while True:
                # Clear before observing state: traffic arriving after an empty
                # snapshot must still wake the subsequent wait.
                self._wake.clear()
                with self._lock:
                    sealed, fatal = self._sealed, self._fatal
                if fatal:
                    break
                state = self._traffic.state()
                if sealed and state.settled:
                    break
                if state.due:
                    failed = False
                    try:
                        self._traffic.flush_due()
                    except Exception:
                        failed = True
                    if failed:
                        # Notify after leaving the exception handler so the
                        # callback cannot inherit a raw sys.exception traceback.
                        self._fail(TrafficFailure.PERSISTENCE)
                    continue
                delay = state.wait_seconds
                self._wake.wait(None if delay is None else min(delay, threading.TIMEOUT_MAX))
        except BaseException:
            crashed = True
        if crashed:
            self._fail(TrafficFailure.WORKER, fatal=True)
        self._wake.set()

    def drain(self, deadline: float) -> TrafficDrainResult:
        """Seal observations, attempt the last watermark at once and join (R20).

        The seal ends observations, so the pending watermark is the last one: it
        is attempted without waiting out an interval that a successful attempt
        began (``JobTraffic.finish``). A failed attempt's interval still paces
        its retry.

        Args:
            deadline: Finite absolute deadline in ``time.monotonic``'s domain.
                All HTTP success callbacks must already have drained.

        Returns:
            TrafficDrainResult: Settled only after joining with no pending/active
                watermark. A timeout retains the worker and all pending state;
                retain Writer/HB ownership and call drain again later. A failure
                is explicit and supplies no permission to discard pending state.

        Raises:
            ValueError: If the absolute deadline is not finite numeric time.
            RuntimeError: If not started or called from the owned worker itself.
        """
        if type(deadline) not in {int, float}:
            raise ValueError("Traffic drain deadline must be a finite monotonic time")
        try:
            finite = math.isfinite(deadline)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Traffic drain deadline must be a finite monotonic time")
        with self._lock:
            thread = self._thread
            if thread is None or threading.current_thread() is thread:
                raise RuntimeError("Drain requires the caller that owns the started worker")
            self._sealed = True
        self._traffic.finish()
        self._wake.set()
        while thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(min(remaining, threading.TIMEOUT_MAX))
        joined = not thread.is_alive()
        try:
            state = self._traffic.state()
        except BaseException:
            state = None
        if state is None:
            self._fail(TrafficFailure.WORKER, fatal=True)
        with self._lock:
            failure, fatal = self._failure, self._fatal
        if state is None or fatal or joined and not state.settled:
            outcome = "failed"
        elif joined:
            outcome = "settled"
        elif state.in_flight:
            outcome = "active_timeout"
        elif state.pending_at is not None:
            outcome = "pending_timeout"
        else:
            outcome = "join_timeout"
        return TrafficDrainResult(outcome, joined, state, failure)
