"""Bounded, content-free FIFO admission for guarded nonblocking delivery."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import TIMEOUT_MAX, Condition, Lock
from time import monotonic
from typing import Literal

from aml.core.errors import AMLError

type DeliveryAttempt = Literal["delivered", "writer_busy", "queued"]


@dataclass(frozen=True, eq=False, slots=True)
class DeliveryTicket:
    """Opaque queue identity carrying only an absolute monotonic deadline."""

    deadline: float


class DeliveryQueue:
    """Admit one FIFO head and at most two following prefetch operations.

    Tickets contain no requests, guards, callbacks or hydrated bytes. Callers
    must release guards and scrub content before either waiting method. Each
    ticket belongs to one caller with at most one concurrent guarded attempt.
    The ticket context must enclose every guard lifetime and its scrubbing.
    Terminal tickets retain their FIFO/prefetch slot until that context exits.
    The attempt callback must be nonblocking and must not reenter this queue.
    Writer priority remains the Writer mutex's policy.
    """

    def __init__(self, *, max_pending: int = 256) -> None:
        """Create an application-owned queue without starting a worker.

        Args:
            max_pending: Positive maximum number of outstanding tickets.

        Returns:
            None.

        Raises:
            ValueError: If the capacity is not an exact positive integer.
        """
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("Delivery capacity must be a positive integer")
        self._max_pending = max_pending
        self._condition = Condition(Lock())
        self._pending: deque[DeliveryTicket] = deque()
        self._terminal: set[DeliveryTicket] = set()

    @contextmanager
    def ticket(self, deadline: float) -> Iterator[DeliveryTicket]:
        """Reserve one FIFO position until the outer context confirms guard cleanup.

        Args:
            deadline: Finite absolute time.monotonic deadline in seconds.

        Yields:
            DeliveryTicket: Opaque identity for one caller. Nest guard contexts
                inside this context, including every failure/cancellation path.

        Raises:
            AMLError: delivery_busy if the deadline is invalid/expired or capacity is full.
        """
        if type(deadline) not in (int, float):
            raise AMLError("delivery_busy", "Delivery requires a finite deadline")
        try:
            deadline = float(deadline)
        except OverflowError:
            raise AMLError("delivery_busy", "Delivery requires a finite deadline") from None
        if not math.isfinite(deadline):
            raise AMLError("delivery_busy", "Delivery requires a finite deadline")
        with self._condition:
            now = monotonic()
            self._expire(now)
            if deadline <= now or len(self._pending) >= self._max_pending:
                raise AMLError("delivery_busy", "Delivery admission is unavailable")
            ticket = DeliveryTicket(float(deadline))
            self._pending.append(ticket)
            self._condition.notify_all()
        try:
            yield ticket
        finally:
            with self._condition:
                self._remove(ticket)

    def wait_for_prefetch(self, ticket: DeliveryTicket) -> None:
        """Wait without a guard until this ticket is among the first three.

        Args:
            ticket: Active ticket whose caller has not retained hydrated content.

        Returns:
            None when one pin/hydrate attempt may begin.

        Raises:
            AMLError: delivery_busy if the ticket is cancelled, expired or no longer active.
        """
        self._wait(ticket, positions=3)

    def wait_for_turn(self, ticket: DeliveryTicket) -> None:
        """Wait without a guard or hydrated content until this ticket is head.

        Args:
            ticket: Active ticket retained after a non-head attempt unwound.

        Returns:
            None when a fresh guarded attempt may begin.

        Raises:
            AMLError: delivery_busy if the ticket is cancelled, expired or no longer active.
        """
        self._wait(ticket, positions=1)

    def is_head(self, ticket: DeliveryTicket) -> bool:
        """Report whether this active ticket is the FIFO head, invoking no callback.

        Args:
            ticket: Active ticket whose caller has not pinned for the attempt yet.

        Returns:
            True only for the current head, which carries ticket priority for
            the Writer during its attempt; False for a queued prefetch position.

        Raises:
            AMLError: delivery_busy if the ticket is cancelled, expired or inactive.
        """
        with self._condition:
            self._expire(monotonic())
            return self._position(ticket) == 0

    def attempt(self, ticket: DeliveryTicket, deliver: Callable[[], bool]) -> DeliveryAttempt:
        """Invoke the nonblocking delivery callback only for the current FIFO head.

        Args:
            ticket: Active ticket for a caller currently holding its read guard.
            deliver: Nonblocking callback returning True exactly after guarded
                enqueue, or False for Writer contention. It must not reenter this queue.

        Returns:
            delivered after one successful enqueue, retaining a terminal head
            until context cleanup; writer_busy retaining the same active head;
            queued without invoking a non-head callback. Callers must unwind/scrub
            before waiting for Writer progress or their turn.

        Raises:
            AMLError: delivery_busy for an expired/inactive ticket; service_unhealthy
                if the callback violates its boolean contract.
            Exception: Callback failure, after marking the ticket terminal.
                The outer ticket context releases its slot after guard cleanup.
        """
        with self._condition:
            self._expire(monotonic())
            position = self._position(ticket)
            if position:
                return "queued"
            try:
                delivered = deliver()
                if type(delivered) is not bool:
                    raise AMLError(
                        "service_unhealthy", "Delivery callback returned an invalid result"
                    )
            except BaseException:
                self._finish(ticket)
                raise
            if delivered:
                self._finish(ticket)
                return "delivered"
            return "writer_busy"

    def cancel(self, ticket: DeliveryTicket) -> None:
        """Mark a ticket terminal and wake its owner for guard cleanup.

        Args:
            ticket: Ticket to cancel. Already retired or foreign tickets are harmless.

        Returns:
            None. An in-progress bounded callback settles before cancellation obtains
            the queue lock; cancellation cannot retract bytes already enqueued.
            The slot remains occupied until the outer ticket context exits.
        """
        with self._condition:
            self._finish(ticket)

    def _finish(self, ticket: DeliveryTicket) -> None:
        if ticket in self._pending:
            self._terminal.add(ticket)
            self._condition.notify_all()

    def _remove(self, ticket: DeliveryTicket) -> None:
        try:
            self._pending.remove(ticket)
        except ValueError:
            return
        self._terminal.discard(ticket)
        self._condition.notify_all()

    def _expire(self, now: float) -> None:
        expired = {
            ticket
            for ticket in self._pending
            if ticket.deadline <= now and ticket not in self._terminal
        }
        self._terminal.update(expired)
        if expired:
            self._condition.notify_all()

    def _position(self, ticket: DeliveryTicket) -> int:
        if ticket in self._terminal:
            raise AMLError("delivery_busy", "Delivery ticket is no longer active")
        try:
            return self._pending.index(ticket)
        except ValueError:
            raise AMLError("delivery_busy", "Delivery ticket is no longer active") from None

    def _wait(self, ticket: DeliveryTicket, *, positions: int) -> None:
        with self._condition:
            while True:
                now = monotonic()
                self._expire(now)
                if self._position(ticket) < positions:
                    return
                # Terminal tickets retain slots until their owners scrub and
                # exit. Only live deadlines need a wakeup; never spin on an
                # expired head. Platform wait limits do not alter the deadline.
                wake_at = min(
                    pending.deadline for pending in self._pending if pending not in self._terminal
                )
                self._condition.wait(min(TIMEOUT_MAX, max(0.0, wake_at - now)))
