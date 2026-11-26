"""The order of erasures in one store, and what reads withhold while a forget waits its turn.

Rules (the service holds them; no client order is relied on):

1. Only the service erases. The plugin and every other client only send requests; the
   service alone decides their order. A client sending its forget before or after an Add
   changes nothing about correctness.
2. One erasure per scope at a time. A forget that arrives while another erasure of its
   scope runs, whatever started that one (a confirmation, an in-chat request, the Add's
   check of restated forgotten facts, recovery at boot), waits in its request for its
   turn, up to ``FORGET_WAIT_SECONDS`` and never past the request's own deadline, with the
   Writer mutex released; then every check of its admission runs again. Turns are given
   in arrival order. A request a user made (a prepare, a confirmation, an in-chat
   sentence) is served before background enforcement that has not started yet; a run
   that has started always finishes first. When the bound passes the forget answers
   ``erasure_in_progress`` (Retry-After stays as it is). Two refusals come at once: past
   the gate's bounds on waiting requests (`MAX_SCOPE_WAITERS`, `MAX_WAITERS`), and a
   prepare whose own request's confirmation (or a request crossed with it) waits to join
   the operation that prepare runs (`ErasureGate.wait_unheld`). Nothing overtakes a
   waiting request silently.
3. One operation per request. The same request (the same prepared plan, or the same
   scope and normalised request words within the in-chat record's lifetime) is one
   operation: the later one waits for the earlier one and then runs on its own, reading
   from the stored record what the earlier one did, so nothing is erased twice. A
   different request never joins another, except under the plugin reading of an in-chat
   request ending in a symbol (`ErasureGate.operation`'s ``aliases``).
4. Reads are consistent. From the moment a forget's targets are accepted, reads of the
   scope do not return those memories or their quotes (`ErasureGate.withhold`); while the
   erasure runs, the fence answers reads as it always has. A failed or refused forget shows
   its targets again; nothing stored changes before the erasure itself.
5. Every entry point to an erasure goes through `ErasureGate.turn` in
   ``quotemem.store.forget_run.ForgetOwner``: ``prepare``, ``prepare_spans``, ``forget``,
   ``forget_spans`` and ``resume`` (recovery at boot, a repeated confirmation). The HTTP
   routes, forget by statement, the in-chat forget and the Add's restatement check reach
   an erasure only through those methods.

The gate is held in memory only. A restart loses every waiting turn and every withheld
set with the requests that held them; the durable fences and records are the only truth
about what was erased, and recovery resumes a fence before any request is served.
"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import count
from typing import Final, Protocol

from quotemem.core.entities import MemorySnapshotV2, QuoteSpanV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.request_budget import RequestBudget, bind_budget, current_budget

FORGET_WAIT_SECONDS: Final = 5.0
"""How long a forget waits in its request for another erasure of its scope."""
READ_WAIT_SECONDS: Final = 2.0
"""How long Search, History, Why and the memory list wait for a running erasure."""
FORGET_HIDE_SECONDS: Final = 120.0
"""The longest a forget with no deadline of its own keeps its targets withheld while it waits
for its turn."""
PATIENT_WAIT_SECONDS: Final = 60.0
"""The longest an in-chat forget waits for its scope's turn, within its Add's budget."""
PATIENT_MARGIN_SECONDS: Final = 2.0
"""What a patient wait leaves of its request's budget for the forget it waited for."""
JOIN_WAIT_SECONDS: Final = 30.0
"""How long a request waits in its request for the running operation of the same request
(`ErasureGate.operation`); the request's own budget bounds it further."""
PLAN_FAILURES: Final = frozenset({
    "operation_conflict", "intent_expired", "scope_denied", "authentication_failed",
    "invalid_request",
})
"""Refusals that concern one request's own input or grant, not the forget it asked for: a
request that joined one failing so runs on its own instead of failing the same way."""
OWN_FAILURES: Final = frozenset({"erasure_in_progress", "delivery_busy"})
"""Refusals that end one request on its own budget: its wait for a turn or for the same
request passed its bound or its deadline, it was cancelled, or its request budget ran out.
A request that joined it keeps its own budget, so it runs on its own instead of failing the
same way."""
MAX_SCOPE_WAITERS: Final = 8
"""The most requests that wait in the gate for one scope at once; past it a request is
answered at once, as before the gate existed (`ErasureGate`)."""
MAX_WAITERS: Final = 16
"""The most forgets and prepares that wait in the gate at once over every scope: each holds
an HTTP worker thread while it waits, so the others, /health among them, always find one."""
RESERVED_WAITERS: Final = 8
"""Waits past `MAX_WAITERS` kept for scopes with no request waiting yet: the first contended
forget of another user still waits when a few scopes fill the shared bound."""
MAX_READ_WAITERS: Final = 8
"""The most reads (Search, History, Why, the memory list) that wait in the gate at once over
every scope; they never take a forget's place."""
MAX_SCOPE_READ_WAITERS: Final = 4
"""The most reads of one scope that wait in the gate at once."""
RESERVED_READ_WAITERS: Final = 4
"""Read waits past `MAX_READ_WAITERS` kept for scopes with no read waiting yet."""
OPERATION_FINISH_SECONDS: Final = 120.0
"""The least budget an operation runs on once it admitted its first forget
(`one_operation`); one whose request still had more keeps that."""
POLL_SECONDS: Final = 0.05
"""How often a waiter looks again at a fence no turn of this process holds."""

_WAIT_UNTIL: ContextVar[float | None] = ContextVar("aml_erasure_wait_until", default=None)
_BACKGROUND: ContextVar[bool] = ContextVar("aml_erasure_background", default=False)
_ADMITTED: ContextVar[_Admission | None] = ContextVar("aml_erasure_admitted", default=None)
_HELD: ContextVar[list[bool] | None] = ContextVar("aml_erasure_held", default=None)


@dataclass
class _Admission:
    """Whether an operation admitted a forget, and how to give its request back its own
    budget and deadline when the operation ends (`one_operation`)."""

    started: bool = False
    restore: list[Callable[[], None]] = field(default_factory=lambda: list[Callable[[], None]]())


@contextmanager
def one_operation() -> Iterator[None]:
    """Run the forgets of this context as one operation (a confirmation of a statement plan,
    an in-chat forget): the request's end refuses its first admission only
    (`check_request_end`); once one forget of it was admitted, the others are admitted too.
    From that admission on, the operation runs on a budget of its own instead of its
    request's: as long as the request's own budget or deadline still had left, and at
    least `OPERATION_FINISH_SECONDS`, never cut short when its client leaves. So neither
    the request's end nor its client stops it with some of its targets erased; only that
    budget bounds the calls it still makes, never below what the request itself allowed.
    The request's budget and deadline come back when it ends. A nested call joins the
    enclosing operation."""
    if _ADMITTED.get() is not None:
        yield
        return
    admission = _Admission()
    token = _ADMITTED.set(admission)
    try:
        yield
    finally:
        for restore in reversed(admission.restore):
            restore()
        _ADMITTED.reset(token)


def note_admitted() -> None:
    """Record that a forget of the current operation was admitted (`one_operation`): the
    rest of the operation is detached from its request's budget and deadline, on a budget
    never shorter than what they still had."""
    admission = _ADMITTED.get()
    if admission is None or admission.started:
        return
    admission.started = True
    now = time.monotonic()
    budget = current_budget()
    until = _WAIT_UNTIL.get()
    seconds = max(OPERATION_FINISH_SECONDS,
                  0.0 if budget is None else budget.until_deadline(),
                  0.0 if until is None else until - now)
    wait = _WAIT_UNTIL.set(None)
    admission.restore.append(lambda: _WAIT_UNTIL.reset(wait))
    admission.restore.append(bind_budget(RequestBudget(seconds)))


@contextmanager
def noting_holds() -> Iterator[Callable[[], bool]]:
    """Yield whether a forget run in this context has withheld anything so far: a run that
    held its targets and then failed has shown them again, so its request must not erase
    them later in the same request (`quotemem.service.chat_forget.ChatForget`)."""
    seen = [False]
    token = _HELD.set(seen)
    try:
        yield lambda: seen[0]
    finally:
        _HELD.reset(token)


@contextmanager
def background(value: bool = True) -> Iterator[None]:
    """Mark the forgets started in this context as background enforcement (rule 2): the
    Add's check of restated forgotten facts, recovery. Their turn comes after every user
    request already waiting for the scope."""
    token = _BACKGROUND.set(value)
    try:
        yield
    finally:
        _BACKGROUND.reset(token)


def user_request() -> bool:
    """Whether the forgets of the current context are a user's request."""
    return not _BACKGROUND.get()


@contextmanager
def waiting_until(deadline: float) -> Iterator[None]:
    """Bound every wait for a turn in this context by an absolute monotonic ``deadline``.

    A request whose deadline is not the current request budget's (an Add carrying out
    its in-chat forget after its own owner released) sets it here, so no wait outlives it.
    """
    token = _WAIT_UNTIL.set(deadline)
    try:
        yield
    finally:
        _WAIT_UNTIL.reset(token)


def wait_deadline(seconds: float, *, now: float | None = None) -> float:
    """The absolute monotonic end of a wait of at most ``seconds`` in the current request.

    The current request budget's remaining time and an enclosing `waiting_until` both cap
    it, so a wait never outlives the request that waits.
    """
    start = time.monotonic() if now is None else now
    end = start + max(0.0, seconds)
    bound = _WAIT_UNTIL.get()
    if bound is not None:
        end = min(end, bound)
    budget = current_budget()
    if budget is not None:
        end = min(end, start + budget.remaining())
    return end


def request_end() -> float | None:
    """The absolute monotonic end of the current request: its budget's end or an enclosing
    `waiting_until`, the earlier; None for a request that has neither."""
    now = time.monotonic()
    bound = _WAIT_UNTIL.get()
    budget = current_budget()
    ends = [end for end in (bound, None if budget is None else now + budget.remaining())
            if end is not None]
    return min(ends) if ends else None


def patient_deadline() -> float:
    """The end of a patient wait for a turn (an in-chat forget): `PATIENT_WAIT_SECONDS`, and
    `PATIENT_MARGIN_SECONDS` before the request's end, so the forget it waited for runs
    within the budget."""
    now = time.monotonic()
    bound = request_end()
    if bound is None:
        # No request end: the hold ends `FORGET_HIDE_SECONDS` after it began, and the wait
        # ends before it, so a forget never runs after its targets showed again.
        bound = now + FORGET_HIDE_SECONDS
    return min(now + PATIENT_WAIT_SECONDS, bound - PATIENT_MARGIN_SECONDS)


def check_request_end() -> None:
    """Refuse an operation's first admission once the current request's end has passed: the
    operation fails as a whole, nothing is erased, and what it withholds shows again. Once
    one forget of the operation was admitted (`one_operation`), the rest is never refused
    by its request: the operation finishes, its hold kept while it runs.

    Raises:
        QuoteMemError: ``delivery_busy`` past the request's end or once it is cancelled,
            before the operation's first admission.
    """
    admission = _ADMITTED.get()
    if admission is not None and admission.started:
        return
    bound = request_end()
    if _cancelled() or (bound is not None and time.monotonic() >= bound):
        raise QuoteMemError("delivery_busy", "The request ended before the forget was admitted")


def on_event_loop() -> bool:
    """Whether the calling thread runs an asyncio event loop (which must never block)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _cancelled() -> bool:
    budget = current_budget()
    return budget is not None and budget.cancelled


@dataclass
class _Ticket:
    seq: int
    user: bool
    thread: int


@dataclass
class _Scope:
    holder: int | None = None
    depth: int = 0
    waiting: list[_Ticket] = field(default_factory=lambda: list[_Ticket]())

    def first(self) -> _Ticket | None:
        """The ticket whose turn is next: the oldest user request, else the oldest one."""
        users = [ticket for ticket in self.waiting if ticket.user]
        pool = users or self.waiting
        return min(pool, key=lambda ticket: ticket.seq) if pool else None


@dataclass(frozen=True)
class Withheld:
    """What reads of one scope withhold now: memory ids and Evidence spans."""

    memory_ids: frozenset[str] = frozenset()
    spans: tuple[QuoteSpanV1, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.memory_ids or self.spans)

    def covers(self, span: QuoteSpanV1) -> bool:
        """Whether ``span`` overlaps a withheld span of the same Evidence row."""
        return any(
            held.evidence_id == span.evidence_id
            and held.byte_start < span.byte_end and span.byte_start < held.byte_end
            for held in self.spans)

    def touches(self, memory_id: str, quotes: Sequence[QuoteSpanV1] = ()) -> bool:
        """Whether a memory, or any of its quotes, is withheld."""
        return memory_id in self.memory_ids or any(self.covers(span) for span in quotes)

    def evidence(self, evidence_id: str) -> bool:
        """Whether any span of one Evidence row is withheld."""
        return any(held.evidence_id == evidence_id for held in self.spans)


@dataclass
class _Operation:
    """One running request of a scope that the same request arriving again joins."""

    thread: int
    done: bool = False
    failure: QuoteMemError | None = None
    aliases: dict[tuple[str, str], str] = field(
        default_factory=lambda: dict[tuple[str, str], str]())
    """Names others may join it by that it does not answer to itself, each with the key of
    the request it stands for (`ErasureGate.operation`'s ``aliases``)."""
    joiners: set[int] = field(default_factory=lambda: set[int]())
    """The threads waiting to join it: a prepare running it is refused at once rather than
    wait for their holds (`ErasureGate.wait_unheld`)."""


@dataclass(frozen=True)
class _Hold:
    scope_id: str
    memory_ids: frozenset[str]
    spans: tuple[QuoteSpanV1, ...]
    until: float
    thread: int


class ErasureGate:
    """One store's erasure turns per scope and the withheld sets of accepted forgets.

    A turn is reentrant in its thread: a confirmation that holds the turn of its scope
    runs each target's forget inside it. A turn is never waited for while the Writer
    mutex is held (the caller passes ``mutex_held``): that wait could block the holder,
    so such a caller gets no wait and its own fence check refuses as before.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._cond = threading.Condition()
        self._scopes: dict[str, _Scope] = {}
        self._holds: dict[int, _Hold] = {}
        self._seq = count(1)
        self._closed = False
        self._generations: dict[str, int] = {}
        self._operations: dict[tuple[str, str], _Operation] = {}
        self._waiters: dict[str, int] = {}
        self._read_waiters: dict[str, int] = {}
        self._unheld: dict[int, list[str]] = {}

    # Turns -------------------------------------------------------------------------------
    def holds(self, scope_id: str) -> bool:
        """Whether the calling thread holds the turn of ``scope_id``."""
        with self._cond:
            state = self._scopes.get(scope_id)
            return state is not None and state.holder == threading.get_ident()

    def busy(self, scope_id: str) -> bool:
        """Whether some thread holds, or waits for, the turn of ``scope_id``."""
        with self._cond:
            state = self._scopes.get(scope_id)
            return state is not None and (state.holder is not None or bool(state.waiting))

    @contextmanager
    def turn(
        self, scope_id: str, *, user: bool, deadline: float | None = None,
        mutex_held: bool = False,
    ) -> Iterator[None]:
        """Hold the erasure turn of one scope for the body.

        Args:
            scope_id: The scope.
            user: Whether a user asked for this forget (served before background work
                that has not started).
            deadline: The absolute monotonic end of the wait; by default
                `FORGET_WAIT_SECONDS` in the current request (`wait_deadline`).
            mutex_held: Whether the caller holds the Writer mutex; then it does not wait.

        Raises:
            QuoteMemError: ``erasure_in_progress`` when the turn is not given before the
                deadline, the request is cancelled or the service is stopping.
        """
        me = threading.get_ident()
        end = wait_deadline(FORGET_WAIT_SECONDS) if deadline is None else deadline
        # A coroutine never waits here: that would block its event loop.
        mutex_held = mutex_held or on_event_loop()
        with self._cond:
            state = self._scopes.setdefault(scope_id, _Scope())
            if state.holder == me:
                state.depth += 1
                reentered = True
            else:
                reentered = False
                ticket = _Ticket(next(self._seq), user, me)
                state.waiting.append(ticket)
                try:
                    if not (state.holder is None and state.first() is ticket):
                        with self._waiter(scope_id) as admitted:
                            if not admitted:
                                raise QuoteMemError("erasure_in_progress",
                                                    "Too many requests wait for this scope")
                            while not (state.holder is None and state.first() is ticket):
                                left = end - self._clock()
                                if self._closed or mutex_held or left <= 0 or _cancelled():
                                    raise QuoteMemError("erasure_in_progress",
                                                        "Another forget of this scope is running")
                                self._cond.wait(min(left, POLL_SECONDS))
                finally:
                    state.waiting.remove(ticket)
                    self._cond.notify_all()
                state.holder, state.depth = me, 1
        try:
            yield
        finally:
            with self._cond:
                if reentered:
                    state.depth -= 1
                else:
                    state.holder, state.depth = None, 0
                    if not state.waiting:
                        self._scopes.pop(scope_id, None)
                self._cond.notify_all()

    def retry_fenced[T](self, operation: Callable[[], T], *, deadline: float,
                        fenced: Callable[[], bool]) -> T:
        """Run ``operation`` again while it refuses ``erasure_in_progress`` behind a fence
        no turn of this process holds (a fence left by a failed run, or one a recovery
        resumes), until ``deadline``; the operation's own write scope is left between
        tries, so the Writer mutex is never held while waiting."""
        while True:
            try:
                return operation()
            except QuoteMemError as error:
                if error.code != "erasure_in_progress" or not fenced():
                    raise
                with self._cond:
                    left = deadline - self._clock()
                    if self._closed or left <= 0 or _cancelled():
                        raise
                    self._cond.wait(min(left, POLL_SECONDS))

    def wait_quiet(self, scope_id: str, readable: Callable[[], bool], *,
                   deadline: float) -> None:
        """Wait, up to ``deadline``, until ``readable`` holds (a read's scope is no longer
        fenced); return either way, the read then answering as it always has. Past the
        gate's bounds on waiting requests it returns at once."""
        if readable():
            return
        with self._waiter(scope_id, read=True) as admitted:
            while admitted and not readable():
                with self._cond:
                    left = deadline - self._clock()
                    if self._closed or left <= 0 or _cancelled():
                        return
                    self._cond.wait(min(left, POLL_SECONDS))

    @contextmanager
    def _waiter(self, scope_id: str, *, read: bool = False) -> Iterator[bool]:
        """Count one request waiting in the gate for ``scope_id``; yield whether it may wait.
        Forgets and prepares: at most `MAX_SCOPE_WAITERS` for a scope and `MAX_WAITERS` in all,
        and `RESERVED_WAITERS` more for scopes none of them waits for yet. Reads, counted
        apart: at most `MAX_SCOPE_READ_WAITERS` for a scope and `MAX_READ_WAITERS` in all, and
        `RESERVED_READ_WAITERS` more for scopes no read waits for yet. The others are answered
        at once (Retry-After as before)."""
        waiters, per_scope, total, reserved = (
            self._read_waiters, MAX_SCOPE_READ_WAITERS, MAX_READ_WAITERS, RESERVED_READ_WAITERS,
        ) if read else (self._waiters, MAX_SCOPE_WAITERS, MAX_WAITERS, RESERVED_WAITERS)
        with self._cond:
            here = waiters.get(scope_id, 0)
            waiting = sum(waiters.values())
            admitted = here < per_scope and (waiting < total or (
                here == 0 and waiting < total + reserved))
            if admitted:
                waiters[scope_id] = here + 1
        try:
            yield admitted
        finally:
            if admitted:
                with self._cond:
                    left = waiters[scope_id] - 1
                    if left:
                        waiters[scope_id] = left
                    else:
                        del waiters[scope_id]

    @contextmanager
    def operation(
        self, scope_id: str, keys: Sequence[str], *, deadline: float | None = None,
        mutex_held: bool = False, aliases: Mapping[str, str] | None = None,
    ) -> Iterator[str | None]:
        """Run the body as the one operation of each request named in ``keys``; yield the
        key of the request this one joined only by an alias, else None.

        A key names one request of one scope (the keyed digest of its session and its
        normalised words, which the in-chat forget and the forget by statement compute
        alike). When another thread runs an operation of the same key, this request joins
        it: it waits until that one ends, then fails as it failed or runs its own body,
        which reads from the stored record what the operation did and erases only what
        changed after it. It runs its own body, within its own budget, when the failure was
        about that request's own plan or grant (`PLAN_FAILURES`), its own budget, deadline
        or cancellation (`OWN_FAILURES`), or was no refusal at all (an interruption or a
        fault of that request's thread); so it never fails for want of a budget it still
        has, and it reports only what it verifies itself. Two operations of one request
        never run at once; a request of another scope or other words never joins, except
        through ``aliases``: other names the operation is also joined by, each with the key
        it stands for (an in-chat request ending in a symbol, as the released plugins send
        it). The operation itself never
        waits behind another request by an alias, and a request that joined it only by an
        alias always runs its own body afterwards, never failing as the aliased one failed.
        Nothing here is stored: the stored
        record of what a request did (``forget_handled:{s}``) and the forget's own durable
        intent and fence carry it across a restart.

        Raises:
            QuoteMemError: ``erasure_in_progress`` when the joined operation does not end before
                the deadline (`JOIN_WAIT_SECONDS` in the current request by default), the
                request is cancelled or the service is stopping; the joined operation's
                own failure.
        """
        me = threading.get_ident()
        named = [(scope_id, key) for key in dict.fromkeys(keys)]
        aliased = {(scope_id, alias): key for alias, key in (aliases or {}).items()
                   if (scope_id, alias) not in named}
        if not named and not aliased:
            yield None
            return
        end = wait_deadline(JOIN_WAIT_SECONDS) if deadline is None else deadline
        mutex_held = mutex_held or on_event_loop()
        mine = _Operation(me, aliases=aliased)
        taken: list[tuple[str, str]] = []
        stands_for: str | None = None
        with self._cond:
            while True:
                found = next(((name, op) for name in named
                              if (op := self._operations.get(name)) is not None
                              and op.thread != me), None)
                if found is None:
                    break
                name, running = found
                with self._waiter(scope_id) as admitted:
                    if not admitted:
                        raise QuoteMemError("erasure_in_progress",
                                            "Too many requests wait for this scope")
                    running.joiners.add(me)
                    self._cond.notify_all()
                    try:
                        while not running.done:
                            left = end - self._clock()
                            if self._closed or mutex_held or left <= 0 or _cancelled():
                                raise QuoteMemError("erasure_in_progress",
                                                    "The same forget request is still running")
                            self._cond.wait(min(left, POLL_SECONDS))
                    finally:
                        running.joiners.discard(me)
                failure = running.failure
                stands_for = running.aliases.get(name)
                if failure is not None and stands_for is None and (
                        failure.code not in PLAN_FAILURES | OWN_FAILURES):
                    raise QuoteMemError(failure.code, "The same forget request failed")
            for name in [*named, *aliased]:
                if name not in self._operations:
                    self._operations[name] = mine
                    taken.append(name)
        try:
            yield stands_for
        except QuoteMemError as error:
            mine.failure = error
            raise
        except BaseException:
            # Not a refusal of the forget: a request that joined this one runs on its own.
            raise
        finally:
            with self._cond:
                mine.done = True
                for name in taken:
                    if self._operations.get(name) is mine:
                        del self._operations[name]
                self._cond.notify_all()

    def running(self, scope_id: str, key: str) -> bool:
        """Whether an operation of the request ``key`` of ``scope_id`` is running."""
        with self._cond:
            return (scope_id, key) in self._operations

    def close(self) -> None:
        """Stop giving turns: every waiter answers ``erasure_in_progress`` at once."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # Withheld sets ------------------------------------------------------------------------
    @contextmanager
    def withhold(
        self, scope_id: str, memory_ids: Sequence[str], spans: Sequence[QuoteSpanV1] = (),
        *, seconds: float | None = None,
    ) -> Iterator[None]:
        """Withhold memories and Evidence spans from reads of one scope for the body.

        The set is released when the body ends, whatever the outcome (a failed or
        refused forget shows its targets again). Expiry only bounds a hold whose thread is
        still waiting for the scope's turn: it ends at the request's own end (`request_end`;
        ``seconds`` from now for a request with none), so a request that never gets its turn
        cannot keep a value hidden for ever. A hold whose thread holds the scope's turn
        never expires: a forget never erases what it has shown again. Its run checks the
        request's end before each admission instead (`check_request_end`).
        """
        if not memory_ids and not spans:
            yield
            return
        bound = request_end()
        until = time.monotonic() + (FORGET_HIDE_SECONDS if seconds is None else seconds) if (
            bound is None) else bound
        hold = _Hold(scope_id, frozenset(memory_ids), tuple(spans), until,
                     threading.get_ident())
        if (seen := _HELD.get()) is not None:
            seen[0] = True
        key = id(hold)
        with self._cond:
            self._holds[key] = hold
            self._generations[scope_id] = self._generations.get(scope_id, 0) + 1
        try:
            yield
        finally:
            with self._cond:
                self._holds.pop(key, None)
                self._cond.notify_all()

    def wait_unheld(self, scope_id: str, *, deadline: float, mutex_held: bool = False) -> int:
        """Wait, up to ``deadline``, until no other request withholds anything of
        ``scope_id``: a prepare made while an accepted forget waits for its turn or runs reads
        the scope as that forget leaves it, never what it withholds. Return the scope's hold
        generation (`generation`) read at the instant nothing was found withheld, so a
        forget accepted after that instant always changes it.

        A thread that holds the scope's turn or a withheld set of its own does not wait (a
        confirmation preparing each of its targets in its turn); neither does a coroutine or
        a caller holding the Writer mutex, which only waits when nothing is withheld. No
        hold is ever skipped: a hold whose request waits to join an operation that a request
        waiting here runs (the confirmation of the request this prepare runs, or two
        requests crossed) could only end after that wait, so the prepare is refused at once
        instead, and the request it held back runs.

        Raises:
            QuoteMemError: ``erasure_in_progress`` when something is still withheld at the deadline,
                at once when a hold waits for an operation a waiting prepare runs, when the
                request is cancelled or the service is stopping.
        """
        me = threading.get_ident()
        mutex_held = mutex_held or on_event_loop()
        with self._cond, ExitStack() as slot:
            admitted = False
            self._unheld.setdefault(me, []).append(scope_id)
            try:
                while True:
                    self._expire()
                    holds = [hold for hold in self._holds.values() if hold.scope_id == scope_id]
                    state = self._scopes.get(scope_id)
                    if not holds or any(hold.thread == me for hold in holds) or (
                            state is not None and state.holder == me):
                        return self._generations.get(scope_id, 0)
                    if any(self._held_back(hold.thread, me) for hold in holds):
                        raise QuoteMemError("erasure_in_progress",
                                            "The same forget request is being confirmed")
                    left = deadline - self._clock()
                    if self._closed or mutex_held or left <= 0 or _cancelled():
                        raise QuoteMemError("erasure_in_progress",
                                            "Another forget of this scope is waiting or running")
                    if not admitted:
                        admitted = slot.enter_context(self._waiter(scope_id))
                        if not admitted:
                            raise QuoteMemError("erasure_in_progress",
                                                "Too many requests wait for this scope")
                    self._cond.wait(min(left, POLL_SECONDS))
            finally:
                waits = self._unheld[me]
                waits.pop()
                if not waits:
                    del self._unheld[me]

    def _held_back(self, thread: int, me: int, seen: frozenset[int] = frozenset()) -> bool:
        """Whether the hold of ``thread`` cannot end before the wait of ``me`` does:
        ``thread`` waits to join an operation that ``me`` runs, or one whose own thread waits
        in `wait_unheld` for a hold held back so in turn (two requests crossed). A hold of any
        other request is waited for. Under the lock."""
        for operation in self._operations.values():
            if thread not in operation.joiners:
                continue
            owner = operation.thread
            if owner == me:
                return True
            if owner in self._unheld and owner not in seen:
                scope_id = self._unheld[owner][-1]
                if any(self._held_back(hold.thread, me, seen | {owner})
                       for hold in self._holds.values()
                       if hold.scope_id == scope_id and hold.thread != owner):
                    return True
        return False

    def _expire(self) -> None:
        """Drop the holds past their end whose thread is still waiting: one whose thread
        holds its scope's turn is running and never expires (`withhold`). Under the lock."""
        now = self._clock()
        for key, hold in list(self._holds.items()):
            state = self._scopes.get(hold.scope_id)
            if hold.until <= now and not (state is not None and state.holder == hold.thread):
                del self._holds[key]

    def generation(self, scope_id: str) -> int:
        """How many withheld sets of ``scope_id`` were registered so far. A read notes it
        before it reads the withheld set and compares it at the instant it hands out its
        result: a change means a forget of its scope was accepted meanwhile, and the read
        builds its result again. Another scope's forget never makes it build again."""
        with self._cond:
            return self._generations.get(scope_id, 0)

    def withheld(self, scope_id: str) -> Withheld:
        """What reads of ``scope_id`` withhold now (expired holds dropped)."""
        with self._cond:
            if not self._holds:
                return Withheld()
            self._expire()
            holds = [hold for hold in self._holds.values() if hold.scope_id == scope_id]
        if not holds:
            return Withheld()
        return Withheld(frozenset[str]().union(*(hold.memory_ids for hold in holds)),
                        tuple(span for hold in holds for span in hold.spans))


_GATES: weakref.WeakKeyDictionary[object, ErasureGate] = weakref.WeakKeyDictionary()
_GATES_LOCK = threading.Lock()


def gate_for(writer: object) -> ErasureGate:
    """The one gate of a store's Writer; every owner and reader of it shares it."""
    with _GATES_LOCK:
        gate = _GATES.get(writer)
        if gate is None:
            gate = _GATES[writer] = ErasureGate()
        return gate


def known_gate(writer: object) -> ErasureGate | None:
    """The gate of ``writer`` if one was made; reads never make one. An object that cannot
    be weakly referenced never had one."""
    with _GATES_LOCK:
        try:
            return _GATES.get(writer)
        except TypeError:
            return None


def withheld_in(writer: object, scope_id: str) -> Withheld:
    """What reads of ``scope_id`` in ``writer``'s store withhold now; nothing for a store
    whose gate was never made (no forget ever ran or waited in this process)."""
    gate = known_gate(writer)
    return Withheld() if gate is None else gate.withheld(scope_id)


class HeldVersion(Protocol):
    """A version of a memory, as an Add's read view gives it."""

    @property
    def memory_id(self) -> str: ...

    @property
    def snapshot(self) -> MemorySnapshotV2: ...


def withheld_versions(writer: object, scope_id: str) -> Callable[[HeldVersion], bool] | None:
    """Whether an accepted forget of ``scope_id`` withholds a memory version now: its memory
    is taken or it quotes a withheld sentence. None when nothing is withheld, so an Add
    that meets no forget reads exactly as before."""
    held = withheld_in(writer, scope_id)
    if not held:
        return None
    return lambda version: held.touches(version.memory_id, version.snapshot.quote_spans)


def hold_generation(writer: object, scope_id: str) -> int:
    """`ErasureGate.generation` of ``scope_id`` in ``writer``'s store, 0 for a store
    without a gate."""
    gate = known_gate(writer)
    return 0 if gate is None else gate.generation(scope_id)
