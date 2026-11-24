"""Explicit service admission and draining; resource ownership stays with bootstrap."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal


class ServiceUnavailable(RuntimeError):
    """The service has not activated or has permanently withdrawn admission."""


class ServiceLifetime:
    """Protect verification and admitted work until the caller drains shutdown.

    Args:
        verifier: Required current owner/publisher/runtime readiness check. It
            must raise on unsafe state and must not close service resources.

    Construction performs no verification. The caller completes actual boot,
    recovery and projection before activate. Shutdown withdraws admission, stops
    listeners/workers, waits for admitted work, and only then closes Writer, HB
    and the workspace owner outside this object. No threads or I/O are owned here.
    """

    def __init__(self, verifier: Callable[[], None]) -> None:
        if not callable(verifier):
            raise TypeError("A service readiness verifier is required")
        self._verifier = verifier
        self._condition = threading.Condition()
        self._state: Literal[
            "unavailable", "activating", "running", "stopping", "failed", "closed"
        ] = "unavailable"
        self._active = 0

    def _verify(self) -> None:
        try:
            self._verifier()
        except BaseException:
            with self._condition:
                self._state = "failed"
                self._condition.notify_all()
            raise

    def _release(self) -> None:
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def activate(self) -> None:
        """Verify and publish readiness once, after the caller's actual startup.

        Activation itself is counted before verification. Concurrent withdrawal
        prevents it from publishing readiness; failure or closure cannot reopen.

        Raises:
            ServiceUnavailable: If activation is repeated or shutdown intervenes.
            Exception: The verifier's original failure, after permanent withdrawal.
        """
        with self._condition:
            if self._state != "unavailable":
                raise ServiceUnavailable("Service activation is no longer available")
            self._state = "activating"
            self._active += 1
            self._condition.notify_all()
        try:
            self._verify()
            with self._condition:
                if self._state != "activating":
                    raise ServiceUnavailable("Service was withdrawn during activation")
                self._state = "running"
                self._condition.notify_all()
        finally:
            self._release()

    @contextmanager
    def admission(self) -> Iterator[None]:
        """Reserve in-flight ownership before checking readiness and doing work.

        Work admitted before withdrawal may finish. A verifier failure withdraws
        all future admission; ordinary user-work exceptions only release this
        reservation. Every exit, including cancellation, decrements the count.

        Raises:
            ServiceUnavailable: If service is not currently accepting work.
            Exception: The original verifier or user-work exception.
        """
        with self._condition:
            if self._state != "running":
                raise ServiceUnavailable("Service is not accepting work")
            self._active += 1
        try:
            self._verify()
            yield
        finally:
            self._release()

    def healthy(self) -> bool:
        """Verify health under admission protection and observe any withdrawal.

        Ordinary verifier exceptions return false after permanently withdrawing
        service. Process-control exceptions still propagate after releasing the
        reservation. A health observation creates no future admission guarantee.
        """
        try:
            with self.admission():
                with self._condition:
                    return self._state == "running"
        except Exception:
            return False

    def stopping(self) -> bool:
        """Whether readiness has been withdrawn for good, so work in flight is being cut."""
        with self._condition:
            return self._state in {"stopping", "failed", "closed"}

    def stop_accepting(self) -> None:
        """Permanently withdraw readiness and new admission without closing work."""
        with self._condition:
            if self._state not in {"failed", "closed"}:
                self._state = "stopping"
            self._condition.notify_all()

    def wait_drained(self, deadline: float) -> bool:
        """Wait until active work reaches zero or an absolute monotonic deadline.

        Args:
            deadline: Finite absolute time in time.monotonic's clock domain.

        Returns:
            Whether all reservations drained after permanent withdrawal.

        Raises:
            ValueError: If the deadline is not a finite real numeric value.
            RuntimeError: If admission has not been permanently withdrawn.

        Notifications wake the wait on state/count changes. Distant deadlines
        are capped at threading.TIMEOUT_MAX to prevent platform timeout overflow.
        """
        if type(deadline) not in {int, float}:
            raise ValueError("Drain deadline must be a finite monotonic time")
        try:
            finite = math.isfinite(deadline)
        except OverflowError as error:
            raise ValueError("Drain deadline must be a finite monotonic time") from error
        if not finite:
            raise ValueError("Drain deadline must be a finite monotonic time")
        with self._condition:
            if self._state not in {"stopping", "failed", "closed"}:
                raise RuntimeError("Withdraw service admission before waiting for drain")
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=min(remaining, threading.TIMEOUT_MAX))
            return True

    def close(self) -> None:
        """Mark this lifetime terminal only when no reservation remains.

        This closes no external resource. The caller owns listener/worker drain
        and subsequent Writer, HB and owner shutdown.

        Raises:
            RuntimeError: If verification or user work is still active.
        """
        with self._condition:
            if self._active:
                raise RuntimeError("Cannot close service lifetime while work is active")
            self._state = "closed"
            self._condition.notify_all()
