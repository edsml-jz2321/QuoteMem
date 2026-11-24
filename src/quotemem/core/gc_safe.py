"""Locks and identity memos that a garbage collection cannot deadlock.

Python runs a weak reference's callback, and an object's finalizer, wherever
the object dies: where its last reference drops, or inside a garbage
collection, which may start at almost any point of whichever thread is
running, in the middle of any of that thread's critical sections. A callback
that takes a lock there can wait forever: the section it interrupted may hold
that very lock, or another thread holds it while waiting for a lock the
interrupted thread holds. A callback that edits shared state can also break
the section it interrupted. No such callback may take a lock.

``IdentityMemo`` keeps values tied to a live object's lifetime with a callback
that takes no lock. ``CheckedLock`` turns any remaining re-entry of a lock by
its own holder into a loud ``RuntimeError`` instead of a silent deadlock.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from operator import attrgetter
from typing import cast
from weakref import ReferenceType, ref


class CheckedLock:
    """A non-reentrant lock that refuses its own holder instead of deadlocking.

    This is an error-checking mutex. A blocking attempt by the thread that
    holds the lock raises ``RuntimeError``; a nonblocking one returns False,
    as a plain ``threading.Lock`` would. A callback that a collection runs
    inside the holder's section therefore fails instead of waiting for its own
    thread. The holder is the C-level owner of an ``RLock``, recorded
    atomically with the acquisition, so no interruption can fall between
    taking the lock and recording who holds it. The lock may serve as the lock
    of a ``threading.Condition``.
    """

    __slots__ = ("_is_owned", "_lock")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # threading.Condition asks the holder through this same RLock internal.
        self._is_owned = cast(Callable[[], bool], attrgetter("_is_owned")(self._lock))

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Take the lock.

        Args:
            blocking: Whether to wait while another thread holds the lock.
            timeout: Longest wait in seconds for a blocking attempt; -1 waits
                without a limit.

        Returns:
            bool: Whether this call took the lock. A nonblocking attempt by
            the holder returns False.

        Raises:
            RuntimeError: If the holder makes a blocking attempt.
        """
        if self._is_owned():
            if blocking:
                raise RuntimeError("A thread re-entered a CheckedLock it holds")
            return False
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        """Release the lock.

        Raises:
            RuntimeError: If the calling thread does not hold the lock.
        """
        self._lock.release()

    def __enter__(self) -> bool:
        """Take the lock, waiting while another thread holds it."""
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        """Release the lock."""
        self.release()


class IdentityMemo[K, V]:
    """Remember one value per live object, by identity, until that object dies.

    ``recall`` answers only for the very object remembered: a later object
    that reuses a dead one's ``id`` never inherits its value. When an object
    dies, the callback of the memo's weak reference to it releases its value
    at once and queues its entry for removal without any lock: it empties the
    entry's own one-item slot and appends to a ``deque``, and both operations
    are atomic. The next ``remember``, membership test or count removes the
    queued entries under the memo's ``CheckedLock``, and so does a ``recall``
    that finds that lock free; a ``recall`` never waits. That lock guards
    nothing else, and every entry it removes or replaces is released after it,
    so no finalizer runs while it is held. Keys must support weak references.
    """

    __slots__ = ("_dead", "_entries", "_lock")

    def __init__(self) -> None:
        self._entries: dict[int, tuple[ReferenceType[K], list[V]]] = {}
        self._dead: deque[tuple[int, ReferenceType[K]]] = deque()
        self._lock = CheckedLock()

    def __contains__(self, identity: object) -> bool:
        """Report whether a live object with this ``id`` has an entry."""
        self._purge()
        return identity in self._entries

    def __len__(self) -> int:
        """Count the entries of live objects."""
        self._purge()
        return len(self._entries)

    def remember(self, key: K, value: V) -> None:
        """Remember ``value`` for this live object, replacing any value it had.

        Args:
            key: A live object that supports weak references.
            value: The value to recall for exactly this object while it lives.
        """
        identity, slot, dead = id(key), [value], self._dead

        def collected(reference: ReferenceType[K]) -> None:
            # Runs wherever the key dies. It takes no lock and touches only its
            # own slot and the atomic queue.
            slot.clear()
            dead.append((identity, reference))

        entry = (ref(key, collected), slot)
        with self._lock:
            released = self._take_dead()
            replaced = self._entries.get(identity)
            if replaced is not None:
                released.append(replaced)
            self._entries[identity] = entry
        released.clear()

    def recall(self, key: K) -> V | None:
        """Return the value remembered for this same live object.

        Args:
            key: The object to look up.

        Returns:
            V | None: Its value, or None when this very object has no entry.
        """
        if self._dead:
            self._purge(blocking=False)
        entry = self._entries.get(id(key))
        return entry[1][0] if entry is not None and entry[0]() is key else None

    def clear(self) -> None:
        """Forget every entry; the values are released after the memo's lock."""
        with self._lock:
            released = [*self._entries.values()]
            self._entries.clear()
            self._dead.clear()
        released.clear()

    def _purge(self, *, blocking: bool = True) -> None:
        """Remove the queued entries; a nonblocking call leaves them while the lock is busy."""
        if not self._dead or not self._lock.acquire(blocking):
            return
        try:
            released = self._take_dead()
        finally:
            self._lock.release()
        released.clear()

    def _take_dead(self) -> list[tuple[ReferenceType[K], list[V]]]:
        """Pop the entries of the objects queued as dead; the caller holds the lock."""
        taken: list[tuple[ReferenceType[K], list[V]]] = []
        dead = self._dead
        while dead:
            identity, reference = dead.popleft()
            entry = self._entries.get(identity)
            if entry is not None and entry[0] is reference:
                taken.append(self._entries.pop(identity))
        return taken


class HeldIdentityMemo[K, V]:
    """Remember one value per object, by identity, holding the object, for a bounded count.

    For objects that cannot be weakly referenced, such as slotted frozen rows.
    Each entry holds its object strongly, so no other object can take its
    ``id`` while the entry lives and ``recall`` answers only for the very
    object remembered. At most ``capacity`` entries are kept. As in
    ``quotemem.core.snapshot_memo``, a full memo refuses new objects until
    ``capacity // 8`` refusals have passed, then evicts the entries not
    recalled during the last ``16 * capacity`` recalls; an object read on
    every pass of a cyclic scan therefore stays, and one no longer read
    leaves at a sweep. ``recall`` takes no lock; ``remember`` and ``clear``
    take a ``CheckedLock`` and release every evicted entry after it, so no
    finalizer runs while it is held.

    Args:
        capacity (int): The most objects kept.

    Raises:
        ValueError: If ``capacity`` is not positive.
    """

    __slots__ = ("_capacity", "_entries", "_lock", "_reads", "_refused")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("A held identity memo keeps at least one object")
        self._capacity = capacity
        self._entries: dict[int, list[object]] = {}
        self._lock = CheckedLock()
        self._reads = 0
        self._refused = 0

    @property
    def capacity(self) -> int:
        """The most objects kept."""
        return self._capacity

    def __len__(self) -> int:
        """Count the kept objects."""
        return len(self._entries)

    def recall(self, key: K) -> V | None:
        """Return the value remembered for this very object, or None.

        Args:
            key: The object to look up.

        Returns:
            V | None: Its value, or None when this object has no entry.
        """
        # The count is advisory: a lost increment under concurrent readers only
        # delays an eviction.
        self._reads += 1
        entry = self._entries.get(id(key))
        if entry is None or entry[0] is not key:
            return None
        entry[2] = self._reads
        return cast(V, entry[1])

    def remember(self, key: K, value: V) -> None:
        """Keep ``value`` for this object, when the memo has room.

        Args:
            key: The object; the entry holds it.
            value: Its value, which must be immutable or never mutated.
        """
        evicted: list[list[object]] = []
        with self._lock:
            entries = self._entries
            if id(key) not in entries and len(entries) >= self._capacity:
                if self._refused < max(1, self._capacity // 8):
                    self._refused += 1
                    return
                self._refused = 0
                oldest = self._reads - 16 * self._capacity
                for identity, entry in list(entries.items()):
                    if cast(int, entry[2]) < oldest:
                        evicted.append(entries.pop(identity))
                if len(entries) >= self._capacity:
                    return
            replaced = entries.get(id(key))
            if replaced is not None:
                evicted.append(replaced)
            entries[id(key)] = [key, value, self._reads]
        evicted.clear()

    def discard(self, doomed: Callable[[K], bool]) -> None:
        """Forget the entries of every object ``doomed`` names, released after the memo's lock.

        Args:
            doomed: A pure, nonblocking test of one kept object; it runs under the lock.
        """
        with self._lock:
            released = [
                self._entries.pop(identity)
                for identity, entry in list(self._entries.items())
                if doomed(cast(K, entry[0]))
            ]
        released.clear()

    def clear(self) -> None:
        """Forget every entry; the entries are released after the memo's lock."""
        with self._lock:
            released = [*self._entries.values()]
            self._entries.clear()
            self._refused = self._reads = 0
        released.clear()
