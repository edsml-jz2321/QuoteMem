"""Request-owned duration accounting without storage or application authority.

The adapter installs one clock-domain ledger after receiving complete bytes.
Its existing worker inherits that context. These counters observe real guarded
operations; constructing a ledger or exercising it in a test proves no operation
ran. No source, query, credential, provider output or response is retained here.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from quotemem.core.identity import ULID, ScopeId

type TimingKind = Literal["queue", "embedding", "locked_segment"]
"""The observed intervals: the outer queue wait and two service components."""


@dataclass(frozen=True, slots=True)
class FinalizeCompletion:
    """Time after the real executor verified its actual released marker."""

    scope_id: ScopeId
    owner_run_id: ULID
    finalize_cycle: int
    completed_at_ns: int


@dataclass(frozen=True, slots=True)
class RequestDurations:
    """Integer nanoseconds observed before one immutable enqueue point.

    A missing component means that its actual producer was not observed. Search
    never waits for an Add owner, so the queue wait is the only outer wait.
    """

    end_to_end_ns: int
    queue_wait_ns: int
    embedding_ns: int | None
    locked_segment_ns: int | None

    @property
    def service_ns(self) -> int:
        """Return elapsed time excluding the outer queue wait."""
        return self.end_to_end_ns - self.queue_wait_ns


class RequestTiming:
    """Accumulate only bounded scalars along an existing request's owned call.

    The route and its joined worker phases use this ledger sequentially.
    Reentrant or nested waits of the same kind are counted once.
    An invalid observation latches failure; it cannot change business execution
    and cannot produce a valid timing result.

    Args:
        received_at_ns: Monotonic clock value at complete request-byte receipt.
        clock: Same monotonic nanosecond clock; tests may supply a controlled one.
    """

    def __init__(
        self, received_at_ns: int, *, clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.received_at_ns = received_at_ns
        self._clock = clock
        self._totals: dict[TimingKind, int] = {}
        self._depths: dict[TimingKind, int] = {}
        self._last_at = received_at_ns
        self._finished = False
        self._last_submission_started_ns: int | None = None
        self._invalid = type(received_at_ns) is not int or received_at_ns < 0

    def _sample(self) -> int | None:
        try:
            now = self._clock()
            if type(now) is not int or now < self._last_at:
                self._invalid = True
                return None
            self._last_at = now
            return now
        except Exception:
            self._invalid = True
            return None

    def elapsed_queue(self, started_at_ns: int) -> None:
        """Record a joined phase's thread wait at its worker's first instruction.

        The adapter samples the start immediately before its existing submit.
        This method belongs to the worker, before it enters any application
        span. It does not claim that a pending/cancelled callable ever started.
        """
        if self._finished:
            return
        previous_sample = self._last_at
        now = self._sample()
        if (
            self._depths or now is None
            or type(started_at_ns) is not int
            or not previous_sample <= started_at_ns <= now
            or self._last_submission_started_ns is not None and (
                started_at_ns <= self._last_submission_started_ns
            )
        ):
            self._invalid = True
            return
        self._last_submission_started_ns = started_at_ns
        self._totals["queue"] = self._totals.get("queue", 0) + now - started_at_ns

    @contextmanager
    def span(self, kind: TimingKind) -> Iterator[None]:
        """Observe one real interval without suppressing its exception or cancellation."""
        if self._finished:
            # The owned worker can still drain guards/resources after enqueue.
            # Such work is outside the already frozen response-latency interval.
            yield
            return
        started = self._sample()
        depth = self._depths.get(kind, 0)
        included = depth == 0
        # Components describe service time, so placing one inside the queue wait
        # is an instrumentation error, not a reason to subtract it twice.
        if kind != "queue" and self._depths.get("queue", 0):
            self._invalid = True
        self._depths[kind] = depth + 1
        try:
            yield
        finally:
            ended = self._sample()
            if depth:
                self._depths[kind] = depth
            else:
                self._depths.pop(kind, None)
            if included and started is not None and ended is not None:
                self._totals[kind] = self._totals.get(kind, 0) + ended - started

    def finish(self, enqueued_at_ns: int) -> RequestDurations | None:
        """Freeze actual enqueue timing, or return None for an incomplete observation.

        Call only after every measured operation has left its scope. No new
        clock is sampled: worker/guard cleanup after enqueue is not request
        latency. Repeated completion, active intervals or clock regression
        invalidate the measurement without granting any success fact.
        """
        if (
            self._finished or self._depths or self._invalid
            or type(enqueued_at_ns) is not int or enqueued_at_ns < self._last_at
        ):
            self._invalid = True
            return None
        self._finished = True
        elapsed = enqueued_at_ns - self.received_at_ns
        queue = self._totals.get("queue", 0)
        components = sum(self._totals.get(kind, 0) for kind in ("embedding", "locked_segment"))
        if queue > elapsed or components > elapsed - queue:
            self._invalid = True
            return None
        return RequestDurations(
            elapsed, queue, self._totals.get("embedding"), self._totals.get("locked_segment"),
        )


_CURRENT: ContextVar[RequestTiming | None] = ContextVar("aml_request_timing", default=None)


@contextmanager
def request_timing(timing: RequestTiming | None) -> Iterator[None]:
    """Install a request ledger, or suspend it while the server starts another request."""
    token = _CURRENT.set(timing)
    try:
        yield
    finally:
        _CURRENT.reset(token)


@contextmanager
def measure(kind: TimingKind) -> Iterator[None]:
    """Observe the current request when enabled, with an inert default path."""
    timing = _CURRENT.get()
    if timing is None:
        yield
    else:
        with timing.span(kind):
            yield
