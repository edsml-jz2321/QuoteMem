"""Context and workspace locks that nest across workspaces without deadlock.

Every Context tree has one reentrant Context lock, and every live workspace
has one reader-writer lock, shared by all facades over that workspace id in
one Context tree. A thread is *outer* when it holds no Context or workspace
lock, and *nested* otherwise. The protocol is:

- The thread holding a workspace's exclusive side always holds the Context
  lock, so one writer per Context tree proceeds at a time.
- An outer write first queues, in arrival order, behind the other outer
  writers of the same workspace. At the head of that queue it *drains*:
  holding no HeavenBase lock, it waits until the outer readers that were
  inside when it became the head have left, and until every held-back reader
  that a completed write entitled to enter has entered and left. Then it
  takes the Context lock. If outer readers that entered meanwhile are still
  running there, it gives the Context lock back and drains them once more.
  Otherwise, or after that one extra drain, it waits under the Context lock
  for readers that are still running and not parked behind it, and for
  entitled readers to enter, exactly like a nested write.
- A nested write (from a hook, or inside another read) skips the queue and
  the drains: it takes the Context lock and then waits only for running
  readers that are not parked behind it. A write by a thread that reads the
  same workspace is such a nested write: it does not wait for its own read.
  Two readers of one workspace that both write it cannot deadlock: the one
  that gets the Context lock first writes while the other is parked behind
  it. The other writes after that, once the first one's write and the rest
  of its read have ended.
- A read takes only the shared side. A nested read waits only for an active
  writer. An outer read also waits while a waiting writer holds it back
  (phase-fair). A waiting nested writer holds new outer readers back, and so
  does the queue head while it owns the Context lock. While no thread owns
  the Context lock, only one draining queue head per Context tree, the
  *drainer*, holds them back. A held-back reader enters once another thread
  takes the Context lock, or after the next completed write of the
  workspace, before the next queued outer writer takes the exclusive side,
  so neither readers nor outer writers starve.
- A thread that waits for the Context lock, or for a workspace whose
  exclusive side the Context-lock owner holds, is *parked*: it cannot continue
  until that owner releases the lock it waits for. The Context-lock owner does
  not wait for parked readers, because they would wait for it in turn.

A draining head holds no HeavenBase lock. It holds new outer readers back only
while no other thread owns the Context lock, so the Context-lock owner, which
may wait (through a hook's worker thread or a lock outside HeavenBase) for an
outer reader, never waits for a reader the head holds back. A reader held
back while the lock was free is woken as soon as another thread takes it.
While the lock is free, only the drainer holds readers back, so a reader that
one draining head waits for, and that waits in turn (through a worker thread)
for a reader of another workspace, is never held back by a second draining
head. A writer that owns the Context lock waits only for threads that are
running, as before the drains existed. So, within one Context tree, a set of
threads that only call HeavenBase cannot deadlock through these locks,
whatever the nesting of reads and writes across the workspaces of that tree,
and the drains add no cycle through the Context lock.

A close or reset that waits for the Context lock reserves it
(``_ContextLock.acquire_until``): another thread that finds the lock free, or
gets it as its owner lets it go, gives it back and waits until the close or
reset has taken it. That thread waits only while the lock is free, and the
close or reset takes it without waiting for anything else, so the
reservation adds no cycle. A forked child does not inherit it: the close or
reset runs in the parent only.

A cycle can also pass through a database: a drop's statement can wait for a
connection another thread has in use, such as an open MySQL transaction
that ``DROP DATABASE`` waits for. The storage layer installs a guard
(``_WAIT_GUARD``), so a thread with a connection in use on a database that
the Context-lock owner is dropping, or is waiting to see dropped, raises
``RuntimeError`` instead of waiting for the lock.

A signal handler, such as the one for Ctrl-C, can raise between any two steps
of a call here. The interrupted call gives back what it has taken, or
finishes giving back what it held, before the exception goes on, so other
threads go on using the locks. For that, the exit of a block runs no Python
code outside a ``try`` statement (`_at_exit`), a step that gives something
back changes nothing when it runs again, and the mutexes are C locks that
``with`` statements hold for short sections, never while a thread waits. A
recovery that a further exception cuts short runs again, up to three times;
exceptions that keep arriving can still leave a lock held. On Python 3.10, a
second exception that arrives while the first one propagates can skip any
``with`` or ``finally`` cleanup, and so leave a lock held, as it can with any
lock. All this holds for blocks; a release that is called directly can be
skipped as the call is entered.

A block (``with``) of a Context lock, or of a side of a workspace lock, that its
thread holds already changes nothing, nor does a block of the shared side
inside one of the exclusive side: the outermost block takes the lock and
gives it back.
"""

from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager, nullcontext
from functools import partial
import os
import threading
import time
import weakref
from typing import Any, Callable, Iterator

__all__: list[str] = []

_STATE = threading.local()
_PARKED: dict[int, object] = {}
# Every Context lock not yet collected, for the reset of a forked child.
_CONTEXT_LOCKS: "weakref.WeakSet[_ContextLock]" = weakref.WeakSet()
# Times a queue head that finds outer readers running once it owns the Context
# lock gives the lock back and drains them, before it waits for them under it.
_EXTRA_DRAINS = 1
# Refuses an untimed wait for a Context lock that cannot end, set by the storage layer
# (``heavenbase.workspace.hidden_store``): ``(polls, check)``. ``polls()`` tells, cheaply, whether the
# calling thread may need refusing, such as one with a database connection in use; ``check(owner)`` raises
# while the lock's owner thread waits for something that waits for the calling thread.
_WAIT_GUARD: "tuple[Any, Any] | None" = None
# Seconds between the checks of a guarded wait.
_GUARD_POLL = 0.05
# Seconds between the checks of a thread that lets a waiting close or reset take the free Context lock first.
_RESERVED_POLL = 0.001
# Seconds after which a thread that waits for a workspace lock looks again by itself, should exceptions have cut a wake-up short.
_RECHECK = 1.0
# How often a step that gives something back runs, at most, to recover from an exception.
_ATTEMPTS = (1, 2, 3)
# Nested calls that a block may need to take its lock, to recover, or to give it back.
_ROOM = 8


def _complete(give_back: Callable[..., None], *args: Any) -> None:
    """Call ``give_back`` again when an exception interrupts it, and raise that exception once it is done.

    Args:
        give_back (Callable[..., None]): Step that changes nothing when it runs again.
        *args (Any): Its arguments.
    """
    try:
        give_back(*args)
    except BaseException:
        for _ in _ATTEMPTS:
            try:
                give_back(*args)
                break
            except BaseException:
                pass
        raise


def _exit_steps(give_back: Callable[..., None], args: tuple[Any, ...]) -> Iterator[int]:
    """Generator of `_at_exit`: its first resumption gives back, inside a ``try`` statement that it stands in already."""
    try:
        yield 0
        give_back(*args)
    except GeneratorExit:
        return  # the block was not entered
    except BaseException:
        for _ in _ATTEMPTS:
            try:
                give_back(*args)
                break
            except BaseException:
                pass
        raise
    while True:
        yield 0


def _at_exit(give_back: Callable[..., None], *args: Any) -> Callable[..., Any]:
    """Return the ``__exit__`` of one block, which calls ``give_back`` whatever exception interrupts it.

    A signal handler can raise as the interpreter enters a Python function,
    before the first statement. A Python ``__exit__`` can be skipped that way,
    which leaves its lock held. The callable returned here enters no
    function: ``max`` is C code that hands 0 and the three exit arguments to
    its key, the ``send`` of a generator that stands in a ``try`` statement.
    The first of these resumptions gives back, the others do nothing, and each
    yields 0, so ``max`` returns its first argument, 0, and an exception of the
    block goes on.

    Args:
        give_back (Callable[..., None]): Step that changes nothing when it
            runs again.
        *args (Any): Its arguments.

    Returns:
        Callable[..., Any]: The ``__exit__`` of a block that is being entered.
    """
    _ensure_room()
    steps = _exit_steps(give_back, args)
    next(steps)
    return partial(max, 0, key=steps.send)  # type: ignore[attr-defined]


def _ensure_room(calls: int = _ROOM) -> None:
    """Raise ``RecursionError`` before a block takes its lock, when giving it back would exceed the recursion limit."""
    if calls:
        _ensure_room(calls - 1)


def _nothing(*_exc: Any) -> None:
    """Enter or leave a block of a lock that the thread holds already."""


# Block of a lock side that the thread holds already.
_NESTED = nullcontext()


class _HeldLocks:
    """Locks the current thread holds, for nesting and parking decisions."""

    __slots__ = ("contexts", "exclusive", "shared")

    def __init__(self) -> None:
        self.contexts: set[_ContextLock] = set()
        self.exclusive: set[_WorkspaceLock] = set()
        self.shared: set[_WorkspaceLock] = set()

    def any(self) -> bool:
        """Return whether the thread holds any Context or workspace lock."""
        return bool(self.contexts or self.exclusive or self.shared)


def _held() -> _HeldLocks:
    held = getattr(_STATE, "held", None)
    if held is None:
        held = _STATE.held = _HeldLocks()
    return held


def _park(target: object) -> None:
    """Record that the current thread waits for ``target`` and wake writers.

    Writers waiting for a workspace this thread reads re-check their wait,
    because this reader may now be parked behind them. The caller removes the
    record when the wait ends.
    """
    _PARKED[threading.get_ident()] = target
    for lock in tuple(_held().shared):
        if lock is not target:
            with lock._mutex:
                lock._wake()


def _parked_behind_me(thread_id: int) -> bool:
    """Return whether ``thread_id`` waits for a lock the current thread holds."""
    target = _PARKED.get(thread_id)
    if isinstance(target, _ContextLock):
        return target.owned()
    return isinstance(target, _WorkspaceLock) and target in _held().exclusive


class _ContextLock:
    """Reentrant lock of one Context tree that records the threads waiting for it.

    A ``with`` statement holds it for a block, and ``acquire`` and ``release``
    come in pairs; it also reports whether the current thread owns it.
    Workspace locks whose queue
    head drains register here, so a new owner wakes the outer readers they
    held back while the lock was free; at most one of them, the *drainer*,
    holds readers back while the lock is free. A close or reset that waits
    for the lock reserves it (see `acquire_until`).
    """

    def __init__(self) -> None:
        """Create an unheld lock.

        Returns:
            None: This initializer does not return a value.
        """
        self._inner = threading.RLock()  # taken once by the owner, whatever its depth
        self._owner: int | None = None
        self._depth = 0
        self._watchers: set[_WorkspaceLock] = set()  # workspace locks with a queue head
        self._drainer: _WorkspaceLock | None = None  # the one whose draining head holds readers back
        self._watchers_lock = threading.RLock()
        self._reserved: set[int] = set()  # threads of a close or reset waiting for the lock (see `acquire_until`)
        _CONTEXT_LOCKS.add(self)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} owner={self._owner} depth={self._depth}>"

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Take the lock, parking the thread while it waits.

        Args:
            blocking (bool): Whether to wait for another owner.
            timeout (float): Seconds to wait; ``-1`` waits without limit.

        Returns:
            bool: Whether the lock is now held by this thread.
        """
        me = threading.get_ident()
        if self._owner == me:  # only this thread sets or clears its own ownership
            self._depth += 1
            return True
        untimed = timeout is None or timeout < 0
        try:
            while True:
                if self._yields_to_reserved(me):
                    # A close or reset of another thread waits for the lock: it goes first (see `acquire_until`).
                    if not blocking:
                        return False
                    if untimed:
                        while self._reserved and self._owner is None:
                            time.sleep(_RESERVED_POLL)
                if not self._acquire_inner(me, blocking, timeout):
                    return False
                if untimed and self._yields_to_reserved(me):
                    # Taken while a close or reset of another thread reserved it: give it back so that one goes first.
                    self._inner.release()
                    if not blocking:
                        return False
                    continue
                self._took(me)
                return True
        except BaseException:
            for _ in _ATTEMPTS:
                try:
                    self._leave(0)
                    break
                except BaseException:
                    pass
            raise

    @property
    def __enter__(self) -> Callable[..., Any]:
        return _nothing if self._owner == threading.get_ident() else self.acquire

    @property
    def __exit__(self) -> Callable[..., Any]:
        return _nothing if self._owner == threading.get_ident() else _at_exit(self._leave, 0)

    def _yields_to_reserved(self, me: int) -> bool:
        """Return whether a close or reset of another thread has reserved the lock, which ``me`` does not own."""
        return bool(self._reserved) and me not in self._reserved and self._owner != me

    def _acquire_inner(self, me: int, blocking: bool, timeout: float) -> bool:
        """Take the inner lock as `acquire` does, parking an untimed waiter, without recording the new owner."""
        if self._inner.acquire(False):
            return True
        if not blocking:
            return False
        if timeout is not None and timeout >= 0:
            # A timed waiter may give up and continue its read, so it is not parked.
            return self._inner.acquire(True, timeout)
        try:
            _park(self)
            guard = _WAIT_GUARD
            if guard is not None and guard[0]():
                # The owner may be running a drop that waits for this thread's connection: check while waiting.
                while not (acquired := self._inner.acquire(True, _GUARD_POLL)):
                    guard[1](self._owner)
            else:
                acquired = self._inner.acquire(True, -1)
        finally:
            _PARKED.pop(me, None)
        return acquired

    def release(self) -> None:
        """Release one hold of the calling thread.

        Raises:
            RuntimeError: If the calling thread does not own the lock.
        """
        _complete(self._leave, *self._below())

    def owned(self) -> bool:
        """Return whether the calling thread owns the lock."""
        return self._owner == threading.get_ident()

    def _held_depth(self) -> int:
        """Return how often the calling thread holds the lock."""
        return self._depth if self._owner == threading.get_ident() else 0

    def _below(self) -> tuple[int]:
        """Return the calling thread's depth after one release, for `_leave`."""
        depth = self._held_depth()
        if not depth:
            raise RuntimeError("cannot release un-acquired lock")
        return (depth - 1,)

    def _leave(self, depth: int) -> None:
        """Bring the calling thread's depth down to ``depth``, as far as it is not yet; it may hold the lock unrecorded."""
        me = threading.get_ident()
        if depth:
            if self._owner == me:
                self._depth = depth
            return
        if self._owner == me:
            self._depth = 0
            self._owner = None
        _held().contexts.discard(self)
        if self._inner._is_owned():  # type: ignore[attr-defined]
            self._inner.release()

    def acquire_until(self, deadline: float) -> bool:
        """Take the lock for a close or reset, waiting for other owners at most until ``deadline``.

        While it waits, the lock is reserved for it: another thread that
        finds the lock free, or gets it as the previous owner lets it go, does
        not keep it but waits until the reserving thread has taken it. So the
        close or reset waits only for the current owner, not for every thread
        that asks for the lock meanwhile. Such a thread waits only while the
        lock is free, and the reserving thread takes it without waiting for
        anything else, so the reservation adds no wait cycle. A timed acquire
        does not give way. The wait is checked by
        ``_WAIT_GUARD`` as an untimed wait is, so a thread whose connection
        the owner's drop waits for raises instead of waiting until the
        deadline. The thread is not parked: it holds no HeavenBase lock that
        the owner could wait for.

        Args:
            deadline (float): ``time.monotonic()`` value at which to give up.

        Returns:
            bool: Whether the lock is now held by this thread.
        """
        me = threading.get_ident()
        depth = self._held_depth()
        try:
            if self.acquire(False):
                return True
            try:
                self._reserved.add(me)
                guard = _WAIT_GUARD
                check = guard[1] if guard is not None and guard[0]() else None
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    if self._inner.acquire(True, remaining if check is None else min(remaining, _GUARD_POLL)):
                        self._took(me)
                        return True
                    if check is not None:
                        check(self._owner)
            finally:
                self._reserved.discard(me)
        except BaseException:
            for _ in _ATTEMPTS:
                try:
                    self._leave(depth)
                    break
                except BaseException:
                    pass
            raise

    def _took(self, me: int) -> None:
        """Record ``me`` as the new owner and wake the outer readers held back while the lock was free."""
        self._depth = 1
        self._owner = me
        _held().contexts.add(self)
        if self._watchers:
            self._wake_held_back()

    def watch(self, lock: "_WorkspaceLock") -> None:
        """Register a workspace lock whose queue head may hold readers back.

        Args:
            lock (_WorkspaceLock): Lock that has a queue head.
        """
        with self._watchers_lock:
            self._watchers.add(lock)

    def unwatch(self, lock: "_WorkspaceLock") -> None:
        """Drop a workspace lock whose queue is now empty.

        Args:
            lock (_WorkspaceLock): Lock that no longer has a queue head.
        """
        with self._watchers_lock:
            self._watchers.discard(lock)
            if self._drainer is lock:
                self._drainer = None

    def claim_drain(self, lock: "_WorkspaceLock") -> bool:
        """Let ``lock``'s draining head hold readers back, unless another head does.

        Args:
            lock (_WorkspaceLock): Lock whose queue head drains.

        Returns:
            bool: Whether ``lock`` is now the drainer.
        """
        with self._watchers_lock:
            if self._drainer is None:
                self._drainer = lock
            return self._drainer is lock

    def end_drain(self, lock: "_WorkspaceLock") -> None:
        """Stop ``lock``'s head from holding readers back while this lock is free.

        Args:
            lock (_WorkspaceLock): Lock whose head no longer drains.
        """
        with self._watchers_lock:
            if self._drainer is lock:
                self._drainer = None

    def _wake_held_back(self) -> None:
        """Wake outer readers held back while this lock was free; it now has an owner."""
        me = self._owner
        with self._watchers_lock:
            watchers = tuple(self._watchers)
        for lock in watchers:
            if lock._waiting:
                with lock._mutex:
                    if lock._held_back and lock._head != me:
                        lock._wake()


class _Until:
    """Block that holds a Context lock which it takes at once, or by a deadline (see `_ContextLock.acquire_until`).

    Args:
        lock (_ContextLock): Lock to hold.
        deadline (float | None): ``time.monotonic()`` value until which to wait
            for another owner, or ``None`` to require the lock at once.
        refusal (str): Message of the ``RuntimeError`` raised when another
            thread still owns the lock.
    """

    __slots__ = ("_lock", "_deadline", "_refusal")

    def __init__(self, lock: _ContextLock, deadline: float | None, refusal: str) -> None:
        self._lock = lock
        self._deadline = deadline
        self._refusal = refusal

    def __enter__(self) -> None:
        lock, deadline = self._lock, self._deadline
        if lock.owned():
            return
        try:
            if not (lock.acquire(False) or (deadline is not None and lock.acquire_until(deadline))):
                raise RuntimeError(self._refusal)
        except BaseException:
            for _ in _ATTEMPTS:
                try:
                    lock._leave(0)
                    break
                except BaseException:
                    pass
            raise

    @property
    def __exit__(self) -> Callable[..., Any]:
        return self._lock.__exit__


class _WriteAbandoned(Exception):
    """Internal signal: a non-blocking write could not be taken at once."""


class _Turn:
    """One call that takes the exclusive side, from its announcement until it holds it, or until its write ends when it queued."""

    __slots__ = ("thread", "gate", "head")

    def __init__(self, thread: int) -> None:
        self.thread = thread
        self.gate = threading.Lock()  # closed while the call waits
        self.head = False  # whether it has become the head of the queue


class _SharedSide:
    """Block that takes the shared side of a workspace lock, of which its thread holds no side."""

    __slots__ = ("_lock",)

    def __init__(self, lock: "_WorkspaceLock") -> None:
        self._lock = lock

    @property
    def __enter__(self) -> Callable[..., Any]:
        return self._lock.acquire_read

    @property
    def __exit__(self) -> Callable[..., Any]:
        return _at_exit(self._lock._leave_read, 0)


class _WorkspaceLock:
    """Reentrant reader-writer lock of one workspace in one Context tree.

    Any number of threads may hold the shared side together. The exclusive
    side is taken after the Context lock and released before it. A thread
    holding the exclusive side may nest both sides. A thread holding the
    shared side may also take the exclusive side, as a nested writer: it keeps
    its shared hold and waits only for the other readers that are running and
    not parked behind it. See the module docstring for the full protocol.

    Args:
        context_lock (_ContextLock): Lock of the owning Context tree.
    """

    def __init__(self, context_lock: _ContextLock) -> None:
        """Create an unheld lock bound to one Context lock.

        Returns:
            None: This initializer does not return a value.
        """
        self._context_lock = context_lock
        self._mutex = threading.RLock()  # guards the fields below; not held while waiting
        self._waiters: set[Any] = set()  # closed gates of the threads that wait for a change (see `_wake`)
        self._readers: dict[int, int] = {}
        self._outer_readers: set[int] = set()  # readers that entered holding no HeavenBase lock
        self._writer: int | None = None
        self._writer_depth = 0
        self._writers_waiting: set[_Turn] = set()  # announced or queued writers that do not hold the exclusive side yet
        self._nested_waiting: set[_Turn] = set()  # nested writers among them; they wait holding the Context lock
        self._turns = 0  # completed outermost writes, for phase-fair reader admission
        self._waiting: dict[Any, int] = {}  # outer readers held back: their gate, and the turn they arrived in
        self._pending: set[int] = set()  # outer readers the draining head waits for
        self._queue: deque[_Turn] = deque()  # outer writers in arrival order; the first is the head

    @property
    def __enter__(self) -> Callable[..., Any]:
        return _nothing if self._writer == threading.get_ident() else self.acquire_write

    @property
    def __exit__(self) -> Callable[..., Any]:
        return _nothing if self._writer == threading.get_ident() else _at_exit(self._leave_write, 0, self._context_lock._held_depth())

    def read(self) -> AbstractContextManager[Any]:
        """Return a block that holds the shared side."""
        return _NESTED if self.held() else _SharedSide(self)

    def held(self) -> bool:
        """Return whether the calling thread holds either side."""
        me = threading.get_ident()  # only this thread changes its own holds
        return me in self._readers or self._writer == me

    @property
    def _held_back(self) -> int:
        """Outer readers waiting that arrived during the current turn."""
        return sum(arrival == self._turns for arrival in tuple(self._waiting.values())) if self._waiting else 0

    @property
    def _entitled(self) -> int:
        """Outer readers waiting that a completed turn entitles to enter."""
        return sum(arrival < self._turns for arrival in tuple(self._waiting.values())) if self._waiting else 0

    @property
    def _head(self) -> int | None:
        """Thread of the outer writer at the head of the queue."""
        for turn in tuple(self._queue):
            return turn.thread
        return None

    # ------------------------------------------------------------------ waiting

    def _arm(self, gate: Any) -> None:
        """Close ``gate`` until the next `_wake` (under the mutex); the caller then waits at it."""
        gate.acquire(False)
        self._waiters.add(gate)

    def _wake(self) -> None:
        """Let every waiting thread look again (under the mutex)."""
        for gate in tuple(self._waiters):
            if gate.locked():  # an interrupted run has opened it already otherwise
                gate.release()
        self._waiters.clear()

    # ------------------------------------------------------------------ readers

    def _holds_back_outer_readers(self) -> bool:
        """Return whether a waiting writer holds back new outer readers.

        A nested writer always does; it owns the Context lock. The queue head
        does while it owns the Context lock, and while no thread owns it only
        if this lock is the Context lock's drainer. While another thread owns
        the lock, that owner may wait for a new outer reader (a hook's worker
        thread, or a lock outside HeavenBase that the reader holds), so the
        head does not hold such readers back. While the lock is free, a
        reader that another draining head waits for may itself wait for a new
        reader of this workspace, so only one head holds readers back then.
        """
        if self._nested_waiting:
            return True
        if not self._queue:
            return False
        context_lock = self._context_lock
        owner = context_lock._owner
        if owner is None:
            return context_lock._drainer is self
        return owner == self._queue[0].thread

    def _admits_reader(self, nested: bool, arrival: int | None) -> bool:
        if self._writer is not None:
            return False
        if nested or not self._holds_back_outer_readers():
            return True
        # Phase-fair: an outer reader that found a writer waiting enters after one completed write.
        return arrival is not None and self._turns > arrival

    def _admit(self, me: int, depth: int, nested: bool, arrival: int | None = None) -> None:
        """Count one more hold of the calling thread, which has ``depth`` of them (under the mutex)."""
        if not depth and self._writer != me:
            _held().shared.add(self)
            if not nested:
                self._outer_readers.add(me)
                if arrival is not None and arrival < self._turns and self._queue:
                    self._pending.add(me)  # an entitled reader: the head lets it finish before writing
        self._readers[me] = depth + 1

    def acquire_read(self) -> None:
        """Take the shared side.

        A nested thread waits only for an active writer. An outer thread also
        waits while a waiting writer holds it back, until another thread takes
        the Context lock or the next write completes, and in the latter case
        it enters before the next queued outer writer.
        """
        me = threading.get_ident()
        depth = self._readers.get(me, 0)  # only this thread changes its own holds
        gate = None
        try:
            nested = _held().any()
            with self._mutex:
                if depth or self._writer == me or self._admits_reader(nested, None):
                    self._admit(me, depth, nested)
                    return
                gate = threading.Lock()
                arrival = self._turns
                if not nested:  # counted, so the next queue head waits for it once it is entitled
                    self._waiting[gate] = arrival
            try:
                _park(self)
                while True:
                    with self._mutex:
                        if self._admits_reader(nested, arrival):
                            self._waiting.pop(gate, None)
                            self._waiters.discard(gate)
                            self._admit(me, depth, nested, arrival)
                            return
                        self._arm(gate)
                    gate.acquire(True, _RECHECK)
            finally:
                _PARKED.pop(me, None)
        except BaseException:
            for _ in _ATTEMPTS:
                try:
                    self._leave_read(depth, gate)  # also wakes a draining writer, which must re-check
                    break
                except BaseException:
                    pass
            raise

    def release_read(self) -> None:
        """Release one shared hold of the calling thread.

        Raises:
            RuntimeError: If the calling thread does not hold the shared side.
        """
        _complete(self._leave_read, *self._below_read())

    def _below_read(self) -> tuple[int]:
        """Return the calling thread's shared depth after one release, for `_leave_read`."""
        depth = self._readers.get(threading.get_ident(), 0)
        if not depth:
            raise RuntimeError("Workspace read lock released by a thread that does not hold it")
        return (depth - 1,)

    def _leave_read(self, depth: int, gate: Any = None) -> None:
        """Bring the calling thread's shared depth down to ``depth``, as far as it is not yet, and end its wait at ``gate``."""
        me = threading.get_ident()
        with self._mutex:
            if gate is not None:
                self._waiting.pop(gate, None)
                self._waiters.discard(gate)
            if depth:
                if me in self._readers:
                    self._readers[me] = depth
                return
            self._readers.pop(me, None)
            self._outer_readers.discard(me)
            self._pending.discard(me)
            if self._writers_waiting:  # only writers wait for readers to leave
                self._wake()
        _held().shared.discard(self)

    # ------------------------------------------------------------------ writers

    def _admits_writer(self) -> bool:
        """Return whether a writer that owns the Context lock may take the exclusive side.

        It waits for the other readers that are still running and not parked
        behind it. Its own read of this workspace, when it has one, is not
        waited for: that read continues once the write ends.
        """
        if self._writer is not None:
            return False
        me = threading.get_ident()
        return all(reader == me or _parked_behind_me(reader) for reader in self._readers)

    def _drained(self) -> bool:
        """Return whether the draining queue head may take the Context lock.

        The outer readers it waits for have left, and no held-back reader that
        a completed write entitled to enter is still waiting.
        """
        return not self._pending and not self._entitled

    def _runs_outer_readers(self) -> bool:
        """Return whether the queue head, owning the Context lock, would wait for outer readers."""
        if self._entitled:
            return True
        return any(reader in self._outer_readers and not _parked_behind_me(reader) for reader in self._readers)

    def _lead(self) -> None:
        """Make the first queued writer the head, draining the outer readers inside now, unless it is already (under the mutex)."""
        if not self._queue:
            self._pending.clear()
            self._context_lock.unwatch(self)
            return
        turn = self._queue[0]
        if not turn.head:
            self._context_lock.watch(self)
            self._pending = set(self._outer_readers)
            if turn.gate.locked():  # it waits for the previous head to end its write
                turn.gate.release()
            turn.head = True

    def acquire_write(self, blocking: bool = True) -> bool:
        """Take the exclusive side and the Context lock.

        An outer thread queues behind the other outer writers of this
        workspace and drains outer readers before it takes the Context lock
        (see the module docstring); a nested thread takes the Context lock at
        once. A thread that holds the shared side is nested: it keeps its read
        and waits for the other readers only. Readers parked behind this
        thread are never waited for.

        Args:
            blocking (bool): Whether to wait. ``False`` returns immediately
                when the lock cannot be taken at once.

        Returns:
            bool: Whether the exclusive side and the Context lock are now held
            by this thread.
        """
        me = threading.get_ident()
        depth, context_depth = self._held_depths()
        turn = None
        try:
            if depth:  # only this thread sets or clears its own ownership
                self._context_lock.acquire()
                self._writer_depth = depth + 1
                return True
            turn = _Turn(me)
            if _held().any():
                self._write_nested(me, turn, blocking)
            else:
                self._write_outer(me, turn, blocking)
            return True
        except BaseException as error:
            for _ in _ATTEMPTS:
                try:
                    self._leave_write(depth, context_depth, turn)
                    break
                except BaseException:
                    pass
            if isinstance(error, _WriteAbandoned):
                return False
            raise

    def _write_nested(self, me: int, turn: _Turn, blocking: bool) -> None:
        if not self._context_lock.acquire(blocking):
            raise _WriteAbandoned
        with self._mutex:
            if not blocking and not self._admits_writer():
                raise _WriteAbandoned
            self._writers_waiting.add(turn)
            self._nested_waiting.add(turn)
        while True:
            with self._mutex:
                if self._admits_writer():
                    self._become_writer(me, turn)
                    return
                self._arm(turn.gate)
            turn.gate.acquire(True, _RECHECK)

    def _write_outer(self, me: int, turn: _Turn, blocking: bool) -> None:
        context_lock = self._context_lock
        with self._mutex:
            if self._queue and not blocking:
                raise _WriteAbandoned
            self._writers_waiting.add(turn)
            self._queue.append(turn)
            self._lead()
        while not turn.head:
            with self._mutex:
                self._lead()  # in case an exception cut the hand-over short
                if turn.head:
                    break
                turn.gate.acquire(False)  # closed until the previous head hands the queue over
            turn.gate.acquire(True, _RECHECK)
        extra_drains = _EXTRA_DRAINS
        while True:
            while True:
                with self._mutex:
                    if not context_lock.claim_drain(self) and self._held_back:
                        self._wake()  # another head is the drainer: admit them
                    if self._drained():
                        break
                    if not blocking:
                        raise _WriteAbandoned
                    self._arm(turn.gate)
                turn.gate.acquire(True, _RECHECK)
            if not context_lock.acquire(False):
                if not blocking:
                    raise _WriteAbandoned
                # Another thread owns the lock, so new outer readers enter while this head waits.
                context_lock.acquire()
            context_lock.end_drain(self)  # owning the lock, the head holds readers back itself
            redrain = extra_drains > 0
            while True:
                with self._mutex:
                    redrain = redrain and self._runs_outer_readers()
                    if redrain:
                        extra_drains -= 1
                        self._pending = set(self._outer_readers)  # let them finish without the Context lock
                        break
                    if not self._entitled and self._admits_writer():
                        self._become_writer(me, turn)
                        self._pending.clear()
                        return
                    if not blocking:
                        raise _WriteAbandoned
                    self._arm(turn.gate)
                turn.gate.acquire(True, _RECHECK)
            context_lock.release()

    def _become_writer(self, me: int, turn: _Turn) -> None:
        """Give ``me`` the exclusive side, which it has been admitted to (under the mutex)."""
        self._writers_waiting.discard(turn)
        self._nested_waiting.discard(turn)
        self._waiters.discard(turn.gate)
        self._writer_depth = 1
        self._writer = me
        _held().exclusive.add(self)

    def release_write(self) -> None:
        """Release one exclusive hold and its Context lock hold.

        Raises:
            RuntimeError: If the calling thread does not hold the exclusive side.
        """
        _complete(self._leave_write, *self._below_write())

    def _held_depths(self) -> tuple[int, int]:
        """Return how often the calling thread holds the exclusive side, and the Context lock."""
        return (self._writer_depth if self._writer == threading.get_ident() else 0, self._context_lock._held_depth())

    def _below_write(self) -> tuple[int, int]:
        """Return the calling thread's depths after one release, for `_leave_write`."""
        depth, context_depth = self._held_depths()
        if not depth:
            raise RuntimeError("Workspace write lock released by a thread that does not hold it")
        return (depth - 1, context_depth - 1)

    def _leave_write(self, depth: int, context_depth: int, turn: _Turn | None = None) -> None:
        """Bring the calling thread's depths down to these, as far as they are not yet, and end its ``turn``, or the turn it has.

        The end of a turn is the end of an outermost write, or of a call that
        gives up the wait for it.
        """
        me = threading.get_ident()
        with self._mutex:
            if depth:
                if self._writer == me:
                    self._writer_depth = depth
            else:
                if turn is None and self._queue and self._queue[0].thread == me:
                    turn = self._queue[0]  # of the outer write that ends
                if turn is not None:
                    if turn.head:  # it may still drain
                        self._context_lock.end_drain(self)
                    self._writers_waiting.discard(turn)
                    self._nested_waiting.discard(turn)
                    self._waiters.discard(turn.gate)
                    if turn in self._queue:
                        self._queue.remove(turn)
                if self._writer == me:
                    self._turns += 1  # the readers that waited for this write may enter
                    self._writer_depth = 0
                    self._writer = None
                self._lead()
                self._wake()
        if not depth:
            _held().exclusive.discard(self)
        self._context_lock._leave(context_depth)


class _WorkspaceLockTable:
    """One workspace lock per workspace id within one Context tree."""

    def __init__(self, context_lock: _ContextLock) -> None:
        """Create an empty table for one Context lock.

        Returns:
            None: This initializer does not return a value.
        """
        self._context_lock = context_lock
        self._locks: "weakref.WeakValueDictionary[str, _WorkspaceLock]" = weakref.WeakValueDictionary()
        self._mutex = threading.Lock()

    def get(self, ws_id: str | None) -> _WorkspaceLock:
        """Return the shared lock of ``ws_id``, or a private lock for ``None``.

        Args:
            ws_id (str | None): Workspace id, or ``None`` for a facade that owns
                its lock alone (a detached workspace).

        Returns:
            _WorkspaceLock: Lock that every live facade of ``ws_id`` shares.
        """
        if ws_id is None:
            return _WorkspaceLock(self._context_lock)
        with self._mutex:
            lock = self._locks.get(ws_id)
            if lock is None:
                lock = _WorkspaceLock(self._context_lock)
                self._locks[ws_id] = lock
            return lock


def holds_any_lock() -> bool:
    """Return whether the calling thread holds any Context or workspace lock."""
    return _held().any()


def _release_reservations_after_fork() -> None:
    """Free a forked child's Context locks from the reservations of closes and resets that run only in the parent."""
    for lock in tuple(_CONTEXT_LOCKS):
        lock._reserved = set()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_release_reservations_after_fork)
