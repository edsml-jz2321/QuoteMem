"""Transport-neutral application interface behind the AMLB HTTP endpoints.

The protocol describes delivery obligations; it provides no storage, scheduler,
Add owner, authentication provider, or recovery implementation. Production
composition supplies the real application.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from quotemem.core.add_outcome import AddOutcomeV1
from quotemem.core.query import AsOfV1
from quotemem.core.request_budget import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS as DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from quotemem.core.time_order import TimeOrderV1
from quotemem.ingestion.boundary import AddRequestV1


class AddDispatch(Protocol):
    """Cooperative Add phases owned by one admitted request, outside worker waits."""

    def step(self) -> bool:
        """Prepare or order once; return whether this request has its durable receipt."""
        ...

    def completion_ready(self) -> bool:
        """Recheck whether completion can proceed without waiting for another Add phase."""
        ...

    async def wait_for_change(self) -> None:
        """Wait for scheduling progress without occupying an application worker."""
        ...


@runtime_checkable
class AcceptingAddDispatch(AddDispatch, Protocol):
    """An Add dispatch that can acknowledge before completion (``Prefer: respond-async``)."""

    @property
    def accepted(self) -> bool:
        """Whether this request's source is durably saved, ordered or not."""
        ...

    def completed(self) -> bool:
        """Whether this request's result is already visible, without waiting for it."""
        ...


@runtime_checkable
class BackgroundAddDispatch(AcceptingAddDispatch, Protocol):
    """An accepting dispatch whose acknowledged Add completes in the background.

    While its background completion runs or after it ended, an exact resend reads that
    state (``completed``) without preparing or ordering again: a forget running for the
    Add holds the scope's erasure fence, which ordering would answer as busy.
    """

    def in_background(self) -> bool:
        """Whether this Add's background completion runs or ended."""
        ...


@dataclass(frozen=True)
class SearchRequest:
    """Validated search input, preserving the exact user and query strings.

    Args:
        query: Retrieval text, never ingestion input or an instruction to answer.
        user_id: Exact external user identity; session is deliberately absent.
        top_k: Positive result bound within the frozen recall policy's maximum.
        options: Optional multiple-choice strings; null and missing share None.
            Like ``query`` they serve retrieval only: no memory, Evidence or
            model input ever contains them.
        time_order: Explicit fact occurrence/effective-time order. Reorder only
            dated top-k results; undated results keep their relevance positions.
            This does not filter by time.
        as_of: Explicit world or knowledge instant. When omitted, bench Search
            reads ``plain``: current memories plus past states that have a
            witnessed end. The product profile reads ``current`` memories only.
            ``as_of`` is never inferred from the query text.
    """

    query: str = field(repr=False)
    user_id: str = field(repr=False)
    top_k: int
    options: tuple[str, ...] | None = field(default=None, repr=False)
    time_order: TimeOrderV1 | None = field(default=None, repr=False)
    as_of: AsOfV1 | None = field(default=None, repr=False)


@dataclass(frozen=True)
class SearchHit:
    """One authorized, ordered hit at the application's final delivery boundary.

    Args:
        id: Nonempty stable public result identity.
        content: Nonempty content frozen by the application's current read guard.
        scope_id: Internal exact scope tag for the independent transport exit check.
        score: Optional finite relevance score.
        created_at: Optional RFC 3339 timestamp selected by the application.
    """

    id: str
    content: str = field(repr=False)
    scope_id: str = field(repr=False)
    score: float | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """Complete immutable result, already ranked and bounded by requested top_k.

    Args:
        hits: Authorized final hits; an empty tuple means a real zero-hit result.
    """

    hits: tuple[SearchHit, ...]


class SearchDelivery(Protocol):
    """Request-local transport control with a fixed application-configured budget.

    Cancellation wakes are content-free, nonblocking notifications. They run
    outside transport locks; consumers unregister them after their waits drain
    and without holding queue/Writer locks. Unregistration joins an in-flight
    notification. A wake must not reenter its own cancellation subscription.
    They do not forcibly stop an already running synchronous provider call.
    """

    @property
    def received_at(self) -> float:
        """Return monotonic seconds when the adapter obtained the complete request bytes."""
        ...

    @property
    def deadline(self) -> float:
        """Return the immutable absolute monotonic enqueue deadline, including queue time."""
        ...

    @property
    def cancelled(self) -> bool:
        """Report permanent receiver cancellation for cooperative operation checks."""
        ...

    def on_cancel(self, wake: Callable[[], None]) -> AbstractContextManager[None]:
        """Register a wake or immediately invoke it if cancellation already happened.

        Args:
            wake: Nonblocking, content-free notification, invoked outside transport locks.

        Returns:
            A context unregistering/joining the notification after the consumer's
            wait drains, without holding a queue or Writer lock.
        """
        ...

    def deliver(self, result: SearchResult) -> None:
        """Validate and atomically enqueue exactly one guarded response before the deadline.

        Args:
            result: Complete ranked result borrowed under the final delivery guard.

        Returns:
            None after immutable bytes are accepted. Later application failures
            cannot replace them; receiver disconnection may discard unsent bytes.

        Raises:
            QuoteMemError: For cancelled/expired admission, duplicate enqueue or invalid output.
        """
        ...


class BenchmarkApplication(Protocol):
    """Application boundary consumed by HTTP and isolated transport tests."""

    def add(self, request: AddRequestV1) -> AddOutcomeV1:
        """Return the exact request result after its actual completion.

        Args:
            request: Validated SDK input after strict HTTP wire validation.

        Returns:
            Immutable request counts after extraction, semantic review and publication settle.

        Raises:
            QuoteMemError: Application failure; transport maps it to the public closure.
        """
        ...

    def search(self, request: SearchRequest, delivery: SearchDelivery) -> None:
        """Deliver exactly once at the final guarded transport-enqueue point.

        Args:
            request: Exact user-scoped retrieval input, without transport credentials.
            delivery: Fixed deadline and cancellation signal from complete-byte
                receipt. Register queue/wait wakeups and check it cooperatively.
                Invoke deliver under the final guard and commit mutex; it validates
                scopes, serializes and checks deadline/cancellation at atomic enqueue.
                Failure or a closed receiver enqueues nothing.

        Returns:
            None: The response has been enqueued successfully. The transport sends
            those exact bytes afterward without reading or rebuilding content.
            Search reads committed state only and never waits for an Add owner.

        Raises:
            QuoteMemError: No response is delivered when authorization, visibility,
                recovery, receipt, or final enqueue fails.
        """
        ...

    def health(self) -> bool:
        """Report whether the initialized owner can safely accept traffic.

        Args:
            None.

        Returns:
            bool: True only after required owner, durability, epoch, recovery, and
            snapshot checks. Readiness is withdrawn before shutdown admission stops.
        """
        ...
