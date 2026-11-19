"""One monotonic budget for request-owned work, separate from shared durable work."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from aml.core.errors import AMLError

PLATFORM_REQUEST_TIMEOUT_SECONDS = 1800.0
"""AMLB client I/O timeout; external callers retain this larger window."""
PLATFORM_RESPONSE_MARGIN_SECONDS = 60.0
"""Reserve under AMLB's 1,800 s I/O timeout for commit, serialization and network."""
DEFAULT_REQUEST_TIMEOUT_SECONDS = (
    PLATFORM_REQUEST_TIMEOUT_SECONDS - PLATFORM_RESPONSE_MARGIN_SECONDS
)
"""Receipt-to-response Add/Search allowance, independent of provider call limits."""
DEFAULT_COMPLETION_WORK_BUDGET_MS = int(DEFAULT_REQUEST_TIMEOUT_SECONDS * 1000)
"""Current owner allowance, separate from each request waiter."""

MAX_AUTOMATIC_RESUMES = 32
"""Counted automatic owner resumptions, matching AMLB's 32 Add attempts.

The original allowance is uncounted. Scheduler discovery and HTTP retries use
the same durable recovery_count; neither retries nor restarts reset this bound.
"""
MAX_OWNER_RESUMES = MAX_AUTOMATIC_RESUMES + 1
"""Shared automatic/explicit resume cap, leaving one explicit repair after 32.

Every counted resume grants a fresh owner allowance through the same join,
privacy and integrity proofs, and renews eligible provider send allowances.
"""

type RequestContractVersion = Literal[2]
DEFAULT_REQUEST_CONTRACT_VERSION: RequestContractVersion = 2
"""Ordinary service acknowledgement requires publication of the accepted Add prefix."""
type RequestBoundary = Literal[
    "accepted", "body_complete", "worker_started", "wait_finished", "response_ready",
    "send_started", "send_finished", "disconnected", "exhausted", "closed",
]


@dataclass(frozen=True, slots=True)
class RequestBudgetObservation:
    """Content-free boundary times; send completion is not client receipt."""

    started_at: float
    deadline: float
    boundaries: tuple[tuple[RequestBoundary, float], ...]
    cancelled: bool
    operation: Literal["add", "search"] | None = None
    request_sha256: str | None = None


class _Wake:
    def __init__(self, callback: Callable[[], None]) -> None:
        self._callback: Callable[[], None] | None = callback
        self._lock = threading.RLock()

    def invoke(self) -> None:
        with self._lock:
            if self._callback is not None:
                self._callback()

    def retire(self) -> None:
        with self._lock:
            self._callback = None


class RequestBudget:
    """Bound one request without resetting its clock at any phase transition.

    Args:
        seconds: Positive complete request allowance.
        clock: Monotonic clock shared by every phase of this request.

    The budget is inherited by this request's existing worker. Shared finalize
    workers must explicitly run outside this context and retain their own owner.
    Expiry stops further work at cooperative boundaries, never reports an
    unfinished provider thread or durable commit as drained.
    """

    def __init__(self, seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Request budget must be finite and positive")
        self._clock = clock
        self.started_at = clock()
        self.deadline = self.started_at + seconds
        self._work_deadline = self.deadline
        self._response_reserved = False
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: set[_Wake] = set()
        self._boundaries: dict[RequestBoundary, float] = {"accepted": self.started_at}
        self._operation: Literal["add", "search"] | None = None
        self._request_sha256: str | None = None

    def received_at(self, now_ms: int) -> int:
        """Project receipt onto the trusted wall clock without resetting at admission.

        Args:
            now_ms: Current trusted wall time in milliseconds, sampled by reception.

        Returns:
            The logical request start for its durable Run. Round elapsed time up
            so conversion to milliseconds cannot extend the receipt deadline.
        """
        return now_ms - math.ceil((self._clock() - self.started_at) * 1000)

    def identify(self, operation: Literal["add", "search"], request_sha256: str) -> None:
        """Bind the measurement identity after complete bytes, without retaining them."""
        if len(request_sha256) != 64 or any(c not in "0123456789abcdef" for c in request_sha256):
            raise ValueError("Request observation needs one SHA-256 identity")
        with self._lock:
            if self._operation is not None and (self._operation, self._request_sha256) != (
                operation, request_sha256,
            ):
                raise ValueError("Request observation identity cannot change")
            self._operation, self._request_sha256 = operation, request_sha256

    @property
    def cancelled(self) -> bool:
        """Read cancellation or elapsed time without acquiring an application lock."""
        return self._cancelled.is_set() or self._clock() >= self._work_deadline

    def elapsed(self) -> float:
        """Return the seconds this request has spent since its acceptance."""
        return max(0.0, self._clock() - self.started_at)

    def remaining(self) -> float:
        """Return remaining seconds; cancellation leaves no budget for new work."""
        return 0.0 if self._cancelled.is_set() else max(0.0, self._work_deadline - self._clock())

    def reserve_response(self, seconds: float) -> None:
        """Stop HTTP application work early while retaining its observed response ceiling.

        Called once by the transport before body receipt. Independent durable
        owners do not reserve response time and retain their whole work budget.
        """
        if self._response_reserved or not math.isfinite(seconds) or not (
            0 < seconds < self.deadline - self.started_at
        ):
            raise ValueError("Response reserve must be positive and installed once")
        self._work_deadline = self.deadline - seconds
        self._response_reserved = True

    def check(self) -> None:
        """Reject exhausted request work using the existing non-success contract."""
        if self.cancelled:
            self.mark("exhausted")
            raise AMLError("delivery_busy", "The complete request budget is exhausted")

    def mark(self, boundary: RequestBoundary) -> None:
        """Retain each fixed boundary once, without content or unbounded event growth."""
        with self._lock:
            self._boundaries.setdefault(boundary, self._clock())

    def cancel(self) -> None:
        """Wake only this waiter; the shared finalize owner is never cancelled here."""
        with self._lock:
            if self._cancelled.is_set():
                return
            self._cancelled.set()
            callbacks = tuple(self._callbacks)
        failed = False
        for callback in callbacks:
            try:
                callback.invoke()
            except BaseException:
                failed = True
        if failed:
            raise AMLError("service_unhealthy", "Request cancellation notification failed")

    @contextmanager
    def on_cancel(self, wake: Callable[[], None]) -> Iterator[None]:
        """Subscribe a nonblocking wake for the lifetime of one owned wait."""
        subscription = _Wake(wake)
        with self._lock:
            self._callbacks.add(subscription)
        try:
            if self.cancelled:
                subscription.invoke()
            yield
        finally:
            with self._lock:
                self._callbacks.discard(subscription)
            subscription.retire()

    def observed(self) -> RequestBudgetObservation:
        """Copy actual boundaries; missing send completion stays absent."""
        with self._lock:
            return RequestBudgetObservation(
                self.started_at, self.deadline, tuple(self._boundaries.items()), self.cancelled,
                self._operation, self._request_sha256,
            )


_CURRENT: ContextVar[RequestBudget | None] = ContextVar("aml_request_budget", default=None)


def current_budget() -> RequestBudget | None:
    """Read the current request's budget; independent durable workers have None."""
    return _CURRENT.get()


def check_request_budget() -> None:
    """Check request-owned work; shared durable workers have no request budget."""
    if (budget := current_budget()) is not None:
        budget.check()


@contextmanager
def request_budget(budget: RequestBudget | None) -> Iterator[None]:
    """Bind one owner, or explicitly detach shared work from a request waiter."""
    token = _CURRENT.set(budget)
    try:
        yield
    finally:
        _CURRENT.reset(token)
