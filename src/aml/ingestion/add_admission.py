"""One source-ordered Add per scope, independent of receiver lifetime.

The Writer proves every saved input and ordering turn. This bounded directory
only schedules those inputs; client withdrawal never deletes accepted work.
An Add never waits for a predecessor that has not been received: it queues
only behind lower-ordered input of its scope that is already present.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from aml.core.errors import AMLError
from aml.ingestion.segment import message_chunk_index

MAX_PENDING_REQUESTS = 128
"""Bound on preparing and saved descriptors in each scope's scheduling directory."""


@dataclass(frozen=True, slots=True)
class AddReceptionTicket:
    """One content-free attempt registered before application-worker dispatch."""

    scope_id: str
    identity: tuple[str, str]
    attempt: object
    received_at: int


@dataclass(slots=True)
class _Reception:
    run_id: str
    received_at: int
    sequence: int
    source_time: int | None
    work_deadline_at: int | None
    identity: tuple[str, str] | None = None
    attempts: set[object] = field(default_factory=set[object])
    saved: bool = True
    active: bool = True
    retained: bool = False
    completion_required: bool = False
    bound: bool = True
    received_monotonic: float = 0.0

    @property
    def chunk_index(self) -> int | None:
        return None if self.identity is None else message_chunk_index(self.identity[1])


@dataclass(slots=True)
class _Scope:
    receptions: dict[str, _Reception] = field(default_factory=dict[str, _Reception])
    identities: dict[tuple[str, str], str] = field(default_factory=dict[tuple[str, str], str])
    selected: str | None = None
    selected_serving: bool = False
    committed: _Reception | None = None
    running: _Reception | None = None


class AddAdmissionQueue:
    """Schedule one immutable input per owner while other scopes proceed.

    Args:
        clock: Monotonic scheduling clock; it never supplies durable timestamps.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._condition = threading.Condition()
        self._scopes: dict[str, _Scope] = {}
        self._sequence = 0
        self._generation = 0
        self._notifications: set[Callable[[], None]] = set()
        self._on_change: Callable[[], None] | None = None

    def set_change_callback(self, callback: Callable[[], None] | None) -> None:
        """Bind a nonblocking scheduling wake, normally threading.Event.set.

        Called under the queue lock: it must not acquire a Writer/controller
        lock, perform application work, or raise an exception.
        """
        with self._condition:
            self._on_change = callback

    def _wake(self) -> None:
        self._generation += 1
        self._condition.notify_all()
        for notify in self._notifications:
            notify()
        if self._on_change is not None:
            self._on_change()

    def wake(self) -> None:
        """Wake receivers after cancellation without changing accepted work."""
        with self._condition:
            self._wake()

    @property
    def generation(self) -> int:
        """Read a token before an unlocked Writer lease/admission check."""
        with self._condition:
            return self._generation

    async def wait_for_notification(
        self, generation: int, *, deadline: float, check: Callable[[], None],
    ) -> None:
        """Wait without borrowing a worker, retaining races and cancellation."""
        check()
        loop = asyncio.get_running_loop()
        changed = asyncio.Event()

        def notify() -> None:
            loop.call_soon_threadsafe(changed.set)

        with self._condition:
            if generation != self._generation:
                return
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise AMLError("delivery_busy", "The scope's admission queue is busy; retry")
            self._notifications.add(notify)
        try:
            try:
                await asyncio.wait_for(changed.wait(), min(remaining, 0.25))
            except TimeoutError:
                pass
        finally:
            with self._condition:
                self._notifications.remove(notify)
        check()

    def wait_for_change(
        self, generation: int, *, deadline: float, check: Callable[[], None],
    ) -> None:
        """Wait for progress outside the Writer."""
        check()
        with self._condition:
            if generation != self._generation:
                return
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise AMLError("delivery_busy", "The scope's admission queue is busy; retry")
            self._condition.wait(min(remaining, 0.25))
        check()

    @staticmethod
    def _order(reception: _Reception) -> tuple[int, int, int, int, int]:
        chunk = reception.chunk_index
        if chunk is not None:
            return 0, chunk, 0, reception.received_at, reception.sequence
        return (1, int(reception.source_time is None), reception.source_time or 0,
                reception.received_at, reception.sequence)

    @classmethod
    def _head(cls, scope: _Scope) -> _Reception | None:
        return min(scope.receptions.values(), key=cls._order, default=None)

    def _ready_head(self, scope: _Scope) -> _Reception | None:
        """Return the lowest-ordered received input once nothing of its scope runs.

        A missing predecessor is never awaited: only input already present can
        precede the head, and a lower-ordered one still preparing holds it.
        """
        if scope.selected is not None or scope.committed is not None or scope.running is not None:
            return None
        head = self._head(scope)
        if head is None or not head.saved:
            return None
        return head

    def next_run(self, scope_id: str) -> str | None:
        """Hint at the saved head; the service must still prove its turn and open fence."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            head = None if scope is None else self._ready_head(scope)
            return None if head is None else head.run_id

    def pending_scopes(self) -> tuple[str, ...]:
        """Snapshot scopes with queued/committed input, including absent clients."""
        with self._condition:
            return tuple(key for key, value in self._scopes.items()
                         if value.receptions or value.committed is not None)

    def reserve(
        self, scope_id: str, identity: tuple[str, str], received_at: int, *,
        source_time: int | None = None,
    ) -> AddReceptionTicket:
        """Reserve one position; concurrent exact attempts own separate tokens."""
        with self._condition:
            scope = self._scopes.setdefault(scope_id, _Scope())
            key = scope.identities.get(identity)
            reception = None if key is None else scope.receptions.get(key)
            if reception is None:
                if len(scope.receptions) >= MAX_PENDING_REQUESTS:
                    raise AMLError("delivery_busy", "The scope's reception queue is full; retry")
                self._sequence += 1
                key = f"preparing:{self._sequence}"
                reception = _Reception(key, received_at, self._sequence, source_time, None,
                                       identity=identity, saved=False, bound=False,
                                       received_monotonic=self._clock())
                scope.receptions[key] = reception
                scope.identities[identity] = key
            attempt = object()
            reception.attempts.add(attempt)
            reception.active = True
            self._wake()
            return AddReceptionTicket(scope_id, identity, attempt, reception.received_at)

    def bind(self, ticket: AddReceptionTicket, run_id: str, received_at: int) -> None:
        """Bind preparation to its actual Run without issuing write authority."""
        with self._condition:
            scope = self._scopes.get(ticket.scope_id)
            key = None if scope is None else scope.identities.get(ticket.identity)
            reception = None if scope is None or key is None else scope.receptions.get(key)
            if scope is None or reception is None or ticket.attempt not in reception.attempts:
                raise AMLError("delivery_busy", "Reception preparation was deferred; retry")
            if key != run_id:
                if run_id in scope.receptions:
                    raise AMLError("binding_mismatch", "Reception has two scheduling identities")
                del scope.receptions[reception.run_id]
                scope.receptions[run_id] = reception
                scope.identities[ticket.identity] = run_id
                if scope.selected == key:
                    scope.selected = run_id
                reception.run_id = run_id
            if reception.saved and reception.received_at != received_at:
                raise AMLError("binding_mismatch", "Saved reception changed its original time")
            reception.received_at, reception.bound = received_at, True
            self._wake()

    def _prune(self, scope_id: str, scope: _Scope) -> None:
        if not scope.receptions and scope.selected is None and scope.committed is None and (
            scope.running is None
        ):
            self._scopes.pop(scope_id, None)

    def finish(self, ticket: AddReceptionTicket) -> None:
        """Release this receiver; accepted or uncertain saved work keeps its place."""
        with self._condition:
            scope = self._scopes.get(ticket.scope_id)
            key = None if scope is None else scope.identities.get(ticket.identity)
            reception = None if scope is None or key is None else scope.receptions.get(key)
            if scope is None or reception is None:
                return
            reception.attempts.discard(ticket.attempt)
            reception.active = bool(reception.attempts)
            if not reception.active and scope.selected == reception.run_id:
                scope.selected, scope.selected_serving = None, False
            if not reception.saved and not reception.active and not reception.retained:
                self._discard(scope, reception.run_id)
            self._prune(ticket.scope_id, scope)
            self._wake()

    def retain(self, ticket: AddReceptionTicket) -> None:
        """Keep an uncertain save until the Writer proves its saved state on replay."""
        with self._condition:
            scope = self._scopes.get(ticket.scope_id)
            key = None if scope is None else scope.identities.get(ticket.identity)
            reception = None if scope is None or key is None else scope.receptions.get(key)
            if reception is not None:
                reception.retained = True
            self._wake()

    def replayed(self, ticket: AddReceptionTicket, run_id: str) -> None:
        """Retire a placeholder after the service proves an existing receipt."""
        with self._condition:
            scope = self._scopes.get(ticket.scope_id)
            key = None if scope is None else scope.identities.get(ticket.identity)
            reception = None if scope is None or key is None else scope.receptions.get(key)
            if scope is not None and reception is not None:
                if reception.saved and reception.run_id != run_id:
                    raise AMLError("binding_mismatch", "Replay names another saved reception")
                self._discard(scope, reception.run_id)
                self._prune(ticket.scope_id, scope)
            self._wake()

    def register(
        self, scope_id: str, run_id: str, received_at: int, *, source_time: int | None = None,
        work_deadline: float | None = None, work_deadline_at: int | None = None,
        now_at: int | None = None, identity: tuple[str, str] | None = None,
        reception_sequence: int | None = None,
    ) -> None:
        """Register Writer-proved source; receiver age cannot revoke queued work.

        Deadline coordinates are informational exact-witness fields. Called
        under the Writer's short scope; this lock never acquires the Writer.
        """
        _ = work_deadline, now_at
        with self._condition:
            scope = self._scopes.setdefault(scope_id, _Scope())
            reception = scope.receptions.get(run_id)
            already_saved = reception is not None and reception.saved
            if reception is None:
                if len(scope.receptions) >= MAX_PENDING_REQUESTS:
                    raise AMLError("delivery_busy", "The scope's reception queue is full; retry")
                self._sequence += 1
                reception = _Reception(run_id, received_at, self._sequence, source_time,
                                       work_deadline_at, received_monotonic=self._clock())
                scope.receptions[run_id] = reception
            elif not reception.saved:
                if reception.received_at != received_at:
                    raise AMLError("binding_mismatch", "Preparation changed its reception time")
                reception.source_time, reception.work_deadline_at = source_time, work_deadline_at
                reception.saved, reception.retained = True, False
            elif (reception.received_at, reception.source_time, reception.work_deadline_at) != (
                received_at, source_time, work_deadline_at,
            ):
                raise AMLError("binding_mismatch", "Reception queue changed its saved input")
            if reception_sequence is not None:
                if reception_sequence <= 0 or (
                    already_saved and reception.sequence != reception_sequence
                ):
                    raise AMLError("binding_mismatch", "Reception changed its original sequence")
                reception.sequence = reception_sequence
                self._sequence = max(self._sequence, reception_sequence)
            if identity is not None:
                if reception.identity not in (None, identity) or (
                    identity in scope.identities and scope.identities[identity] != run_id
                ):
                    raise AMLError("binding_mismatch", "Reception changed its logical identity")
                reception.identity = identity
                scope.identities[identity] = run_id
            reception.active = True
            self._wake()

    def reception_sequence(self, scope_id: str, run_id: str) -> int | None:
        """Read the reserved order for the Writer's immutable first reception record."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            entry = None if scope is None else scope.receptions.get(run_id)
            return None if entry is None else entry.sequence

    def has_saved_input(self, scope_id: str, run_id: str) -> bool:
        """Hint whether registration observed a successful save, rather than an uncertain one."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            entry = None if scope is None else scope.receptions.get(run_id)
            return entry is not None and entry.saved

    def require_reception_capacity(self, scope_id: str, run_id: str, *, now_at: int) -> None:
        """Refuse unsaved overflow before persistence; exact saved retries keep capacity."""
        _ = now_at
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is not None and run_id not in scope.receptions and (
                len(scope.receptions) >= MAX_PENDING_REQUESTS
            ):
                raise AMLError("delivery_busy", "The scope's reception queue is full; retry")

    def wait_turn(
        self, scope_id: str, run_id: str, *, fence_open: Callable[[], bool],
        check: Callable[[], None], deadline: float, serving: bool = False, wait: bool = True,
    ) -> bool:
        """Select only the source-ordered head outside application locks.

        Args:
            scope_id: Canonical user/conversation scope.
            run_id: Immutable saved input Run.
            fence_open: Borrowed Writer lease check, called outside this lock.
            check: Receiver cancellation and scoped refusal check, also unlocked.
            deadline: This attempt's wait limit, never the work's lifetime.
            serving: Reserve a semantic owner after this single input orders.
            wait: False for a cooperative step or autonomous ready-head probe.

        Returns:
            Whether this input owns the sole ordering turn.

        Raises:
            AMLError: On changed source or an exhausted receiver allowance.
        """
        while True:
            check()
            with self._condition:
                generation = self._generation
            opened = fence_open()
            with self._condition:
                if self._clock() > deadline:
                    raise AMLError("delivery_busy", "The scope's admission queue is busy; retry")
                scope = self._scopes.get(scope_id)
                reception = None if scope is None else scope.receptions.get(run_id)
                if scope is None or reception is None:
                    raise AMLError("reception_stale", "Reception no longer has its saved turn")
                if not reception.saved:
                    raise AMLError("reception_stale", "This attempt must receive its source again")
                if opened and scope.selected == run_id:
                    return True
                head = self._ready_head(scope) if opened else None
                if head is not None and head.run_id == run_id:
                    scope.selected = run_id
                    scope.selected_serving = serving or reception.completion_required
                    self._wake()
                    return True
                if not wait:
                    return False
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise AMLError("delivery_busy", "The scope's admission queue is busy; retry")
                if generation == self._generation:
                    self._condition.wait(remaining)

    def require_ordering_turn(self, scope_id: str, run_id: str) -> None:
        """Re-prove the selected input under the Writer immediately before commit."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is None or run_id not in scope.receptions:
                return
            if scope.selected != run_id or not scope.receptions[run_id].saved:
                raise AMLError("delivery_busy", "Reception lost its ordering turn; retry")

    def ordered(self, scope_id: str, run_id: str) -> None:
        """Reserve exactly one successor owner after its receipt becomes durable."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is None:
                return
            reception = scope.receptions.get(run_id)
            if scope.selected == run_id:
                assert reception is not None and scope.committed is None
                reception.completion_required |= scope.selected_serving
                scope.selected, scope.selected_serving = None, False
                scope.committed = reception
            scope.receptions.pop(run_id, None)
            if reception is not None and reception.identity is not None:
                scope.identities.pop(reception.identity, None)
            self._wake()

    def request_completion(self, scope_id: str, run_id: str) -> None:
        """Ask discovery to drive an input through ordinary Writer owner admission."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is None:
                return
            reception = scope.receptions.get(run_id)
            if reception is None and scope.committed is not None and (
                scope.committed.run_id == run_id
            ):
                reception = scope.committed
            if reception is not None:
                reception.completion_required = True
            self._wake()

    def restore_ordered(
        self, scope_id: str, run_id: str, received_at: int, identity: tuple[str, str],
    ) -> None:
        """Restore one Writer-proved receipt whose semantic owner has not started.

        The caller must prove the durable receipt, current epochs and incomplete
        completion prefix. This scheduling hint grants no ordering authority.
        An incompatible selected input or owner is an integrity refusal rather
        than permission to let another queued input pass the restored receipt.
        """
        with self._condition:
            scope = self._scopes.setdefault(scope_id, _Scope())
            reserved = scope.committed if scope.committed is not None else scope.running
            if reserved is not None:
                if (reserved.run_id, reserved.received_at, reserved.identity) != (
                    run_id, received_at, identity,
                ):
                    raise AMLError("binding_mismatch", "Restored receipt conflicts with its owner")
                reserved.completion_required = True
                self._wake()
                return
            if scope.selected is not None or scope.identities.get(identity, run_id) != run_id:
                raise AMLError(
                    "binding_mismatch", "Restored receipt conflicts with an ordering turn",
                )
            reception = scope.receptions.get(run_id)
            if reception is not None:
                if (reception.received_at, reception.identity) != (received_at, identity):
                    raise AMLError(
                        "binding_mismatch", "Restored receipt changed its original identity",
                    )
                self._discard(scope, run_id)
            else:
                self._sequence += 1
                reception = _Reception(
                    run_id, received_at, self._sequence, None, None, identity=identity,
                    active=False, received_monotonic=self._clock(),
                )
            reception.completion_required = True
            scope.committed = reception
            self._wake()

    def owner_ready(self, scope_id: str) -> bool:
        """Hint that an ordered serving receipt still needs its actual owner."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            return scope is not None and scope.committed is not None and (
                scope.committed.completion_required
            )

    def reconciliation_runs(self, scope_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Snapshot bounded pending/committed anchors for negative Writer epoch proof."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is None:
                return (), ()
            pending = tuple(value.run_id for value in scope.receptions.values() if value.bound)
            owner = scope.committed if scope.committed is not None else scope.running
            committed = () if owner is None else (owner.run_id,)
            return pending, committed

    def retire_owner_reservation(self, scope_id: str, committed: tuple[str, ...]) -> None:
        """Retire only the exact unowned input proved obsolete by the Writer."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is None:
                return
            owner = scope.committed if scope.committed is not None else scope.running
            if owner is None or committed != (owner.run_id,):
                return
            scope.committed = None
            scope.running = None
            self._prune(scope_id, scope)
            self._wake()

    def retire_rejected(self, scope_id: str, run_id: str) -> None:
        """Retire permanently epoch-rejected source after Writer-proved deletion."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is not None:
                self._discard(scope, run_id)
                self._prune(scope_id, scope)
            self._wake()

    def activate(self, scope_id: str, run_id: str, attempt: object) -> None:
        """Retain a receiver token; its presence is not an ordering condition."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            reception = None if scope is None else scope.receptions.get(run_id)
            if reception is None:
                raise AMLError("reception_stale", "Add attempt lost its saved reception")
            reception.attempts.add(attempt)
            reception.active = True
            self._wake()

    def withdraw(self, scope_id: str, run_id: str, *, attempt: object | None = None) -> None:
        """Withdraw only the receiver; saved inputs remain eligible for autonomous work."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is None:
                return
            reception = scope.receptions.get(run_id)
            if reception is not None:
                if attempt is not None:
                    reception.attempts.discard(attempt)
                reception.active = bool(reception.attempts)
                if not reception.active and scope.selected == run_id:
                    scope.selected, scope.selected_serving = None, False
                if not reception.saved and not reception.active and not reception.retained:
                    self._discard(scope, run_id)
            self._prune(scope_id, scope)
            self._wake()

    def discard(self, scope_id: str, run_id: str) -> None:
        """Reflect Writer-proved source deletion, retaining live receivers' tokens."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is None:
                return
            reception = scope.receptions.get(run_id)
            if reception is not None and reception.identity is not None and reception.attempts:
                reception.saved, reception.retained = False, False
                if scope.selected == run_id:
                    scope.selected, scope.selected_serving = None, False
            else:
                self._discard(scope, run_id)
            self._prune(scope_id, scope)
            self._wake()

    @staticmethod
    def _discard(scope: _Scope, run_id: str) -> None:
        reception = scope.receptions.pop(run_id, None)
        if reception is not None and reception.identity is not None:
            scope.identities.pop(reception.identity, None)
        if scope.selected == run_id:
            scope.selected, scope.selected_serving = None, False

    def require_finalize_allowed(self, scope_id: str) -> None:
        """Prevent a new owner from passing queued input or a running owner."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is not None and (
                scope.selected is not None or scope.running is not None
                or scope.committed is None and scope.receptions
            ):
                raise AMLError("finalize_pending", "The next Add must order before its owner")

    def owner_started(self, scope_id: str) -> None:
        """Consume one input only after its actual preparing lease was written."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is not None and scope.committed is not None:
                scope.running, scope.committed = scope.committed, None
                self._wake()

    def reception_only(self, scope_id: str) -> None:
        """Keep receipt-only SDK staging immediate and free of a semantic-owner turn."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is not None and (
                scope.committed is None or not scope.committed.completion_required
            ):
                scope.committed = None
                self._prune(scope_id, scope)
                self._wake()

    def owner_released(self, scope_id: str) -> None:
        """Advance after actual release; the next received input may order at once."""
        with self._condition:
            scope = self._scopes.get(scope_id)
            if scope is not None:
                scope.running = None
                self._prune(scope_id, scope)
            self._wake()
