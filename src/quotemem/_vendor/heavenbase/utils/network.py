"""Network environment helpers."""

from __future__ import annotations

__all__ = ["NetworkProxy"]

import os
import sys
import threading
from contextvars import ContextVar

from .typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

# One open hold: the instance, the value it applied to each variable it sets, what it found,
# the depth of the locked section that entered it, and its entry number.
_Hold = Tuple["NetworkProxy", Dict[str, Optional[str]], Dict[str, Optional[str]], int, int]
# A claim on one hold: empty until a hold is chosen, then that hold, then one of the states below.
_Exit = List[Any]
# States of a claim after its hold. An exit marks its hold before it queues or releases it, so a retry after an exception
# finishes that step instead of choosing another hold. An enter marks its hold open once it is applied and no longer
# claimed, so that another exit may have chosen it since.
_QUEUED, _RELEASED, _OPEN = "queued", "released", "open"
# The attempts an interrupted ``__enter__`` or ``__exit__`` makes at its recovery while further exceptions cut it short;
# a tuple, so that starting the loop calls no function, after which a signal handler could run.
_ATTEMPTS = (1, 2, 3)
# The instances that the running thread or task, or the task that started it, entered, each with the value it applied to
# each variable it sets, oldest first. An exit on another thread or task leaves its entry, so a reader skips an instance
# that has no open hold.
_Entered = Tuple[Tuple["NetworkProxy", Dict[str, Optional[str]]], ...]
_ENTERED: ContextVar[_Entered] = ContextVar("heavenbase_network_proxy_entered", default=())


class NetworkProxy:
    """Temporarily apply proxy environment variables and restore them on exit.

    ``os.environ`` is process-wide, so the holds of every instance are tracked
    together, in entry order, under one lock: concurrent ``LLM.achat`` calls
    each enter their own instance, and a ``Database`` enters its one instance
    from several threads at once. Each hold applies its settings when it is
    entered, so the newest hold wins while holds overlap, and it records the
    environment it found. When the newest hold exits, it restores what it
    found, as nested use always did. When an older hold exits first, the newer
    holds keep their values: what the older hold found is handed to the next
    newer hold that took it over, so it returns when that hold exits, and a
    variable that no newer hold sets returns to it at once. The original
    environment therefore returns when the last hold exits, whatever the exit
    order. An instance cannot tell which of its open holds is exiting, so it
    gives up its newest one that no other exit in progress has chosen; this is
    exact for nested use. An instance whose settings are all ``None`` changes
    nothing and takes no lock.

    The lock is held only while the environment and the hold list change, never
    while the code inside a ``with`` block runs. A garbage-collector finalizer
    or a signal handler can still run on a thread that is inside that short
    section, for example when the collector closes an abandoned proxied stream
    whose generator holds a proxy across its yields. Such a call never waits
    for the lock its own thread holds. When it exits a hold that was entered
    before the interrupted section began, the exit is queued and applied as
    soon as the interrupted section finishes, so the section never sees the
    environment change under it. A hold entered and exited entirely inside the
    interrupting call applies and restores at once. When the interrupting call
    exits an instance that is entering or exiting at the same time, it never
    gives up the hold that the interrupted call is entering or has chosen to
    give up.

    An exception that a signal handler raises during ``__enter__`` or
    ``__exit__``, ``KeyboardInterrupt`` included, never leaves the lock held.
    When it ends ``__enter__``, the call first gives up the hold it recorded,
    because the caller will not exit it; when an interrupting exit of the same
    instance has already given that hold up, the call gives up the instance's
    newest hold that no other exit has chosen instead. When it arrives inside
    ``__exit__``, the exit is still completed; as with any context manager
    written in Python, only an exception raised before the first statement of
    ``__exit__`` runs leaves the hold open. A recovery that further exceptions
    cut short runs again, up to three times in all, and the first exception is
    the one raised; exceptions that arrive more often than that, or in the few
    instructions between two attempts, can still leave the hold open. On
    Python 3.10 only, a second exception that arrives while the first one
    propagates can skip a ``with`` or ``finally`` cleanup in any Python code,
    and so leave the lock held. In a child process created by ``os.fork``, the lock starts free even
    when another thread of the parent held it. During interpreter shutdown, a
    daemon thread that the interpreter stopped while it held the lock never
    releases it; an enter or exit that runs then, for example when shutdown
    collects an abandoned proxied stream, changes nothing instead of waiting
    for the lock, so the process still exits.
    """

    PROXY_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy")
    # Reentrant so that a finalizer or signal handler running inside a locked section on the same thread does not wait on itself.
    _lock: ClassVar[Any] = threading.RLock()
    # One entry per locked section open on the thread that holds ``_lock``, outermost first: the entry number the next
    # hold would get when that section began. There is one normally, more while a finalizer or signal handler interrupts one.
    _sections: ClassVar[List[int]] = []
    _entries: ClassVar[int] = 0
    # Every open hold, oldest first.
    _held: ClassVar[List["_Hold"]] = []
    # Holds whose exit interrupted a locked section they did not open in; the outermost section applies them before it ends.
    _deferred: ClassVar[List["_Hold"]] = []
    # Every claim in progress. An exit keeps its hold from other exits until it has released or queued it, even while an
    # exception that interrupted it is handled, and an enter keeps the hold it is entering until that hold is applied,
    # so two exits of one instance, or an exit and an interrupted enter giving up its hold, never give up the same hold.
    _exits: ClassVar[List["_Exit"]] = []

    def __init__(self, http_proxy: Optional[str] = None, https_proxy: Optional[str] = None, no_proxy: Optional[str] = None) -> None:
        """
        Initialize the utility object with the provided runtime settings.

        Use this helper when one operation needs temporary proxy environment variables without leaking them to later code.

        Args:
            http_proxy (Optional[str]): Temporary ``HTTP_PROXY`` value. ``None`` leaves it unchanged and ``""`` clears it inside the context. Defaults to ``None``.
            https_proxy (Optional[str]): Temporary ``HTTPS_PROXY`` value. ``None`` leaves it unchanged and ``""`` clears it inside the context. Defaults to ``None``.
            no_proxy (Optional[str]): Temporary ``NO_PROXY`` value. ``None`` leaves it unchanged and ``""`` clears it inside the context. Defaults to ``None``.

        Returns:
            None: This function returns ``None`` after initializing the object.
        """
        self.settings = {
            "HTTP_PROXY": http_proxy,
            "http_proxy": http_proxy,
            "HTTPS_PROXY": https_proxy,
            "https_proxy": https_proxy,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }
        self.backup: Dict[str, Optional[str]] = {}

    @property
    def active(self) -> bool:
        """Return whether entering this context changes any environment variable.

        Returns:
            bool: ``True`` when at least one setting is not ``None``.
        """
        return any(value is not None for value in self.settings.values())

    @staticmethod
    def _section(work: Optional[Callable[..., None]], *args: Any) -> None:
        """Run ``work(*args)`` in a new locked section; ``len(NetworkProxy._sections)`` is then the section's depth.

        The depth is 1 normally, and more inside a section that a finalizer or signal handler interrupted on this thread.

        The outermost section applies the exits queued while it was interrupted before it ends. The lock is taken by a
        ``with`` statement and the section's bookkeeping is undone in ``finally`` clauses, so an exception that a signal
        handler raises anywhere in the section, ``KeyboardInterrupt`` included, never leaves the lock or the section held
        (on Python 3.10, as long as a second exception does not arrive while the first one propagates).
        """
        while True:
            lock = NetworkProxy._lock
            # During interpreter shutdown a daemon thread that the interpreter stopped inside a section holds the lock
            # forever, so waiting for it would hang the exit. The process is ending: the section is skipped instead.
            finalizing = sys.is_finalizing()
            if finalizing and not lock.acquire(blocking=False):
                return
            try:
                with lock:  # reentrant, so immediate when the shutdown check above already holds it
                    sections = NetworkProxy._sections
                    mark = len(sections)
                    try:
                        sections.append(NetworkProxy._entries)
                        if work is not None:
                            work(*args)
                    finally:
                        try:
                            if not mark:
                                NetworkProxy._drain()
                        finally:
                            del sections[mark:]
            finally:
                if finalizing:
                    lock.release()
            # An exit queued after the drain above but before the section ended would otherwise wait for the next section.
            if mark or not NetworkProxy._deferred:
                return
            work = None

    @staticmethod
    def _drain() -> None:
        """Apply the queued exits; the caller holds the lock.

        An exception that interrupts one exit finishes it and the remaining ones before it propagates.
        """
        queued = NetworkProxy._deferred
        interrupted: Optional[BaseException] = None
        while queued:
            hold = queued[0]
            try:
                NetworkProxy._discard(hold)
            except BaseException as error:
                NetworkProxy._discard(hold)  # a no-op once the hold is released
                interrupted = interrupted or error
            index = NetworkProxy._find(queued, hold)
            if index is not None:
                del queued[index]
        if interrupted is not None:
            raise interrupted

    @staticmethod
    def _find(entries: List[Any], item: Any) -> Optional[int]:
        """Return the position of ``item`` itself in ``entries``, or ``None``; searches newest first."""
        for index in range(len(entries) - 1, -1, -1):
            if entries[index] is item:
                return index
        return None

    @staticmethod
    def _unregister(claim: "_Exit") -> None:
        """Remove ``claim`` itself from ``_exits`` if it is there; the caller holds the lock."""
        index = NetworkProxy._find(NetworkProxy._exits, claim)
        if index is not None:
            del NetworkProxy._exits[index]

    @staticmethod
    def _taken(hold: "_Hold", own: "_Exit") -> bool:
        """Return whether a claim other than ``own`` has chosen ``hold``, or an exit queued it; the caller holds the lock."""
        if NetworkProxy._find(NetworkProxy._deferred, hold) is not None:
            return True
        for other in NetworkProxy._exits:
            if other is not own and other and other[0] is hold:
                return True
        return False

    @staticmethod
    def _discard(hold: "_Hold") -> None:
        """Release ``hold`` if it is still open; the caller holds the lock."""
        index = NetworkProxy._find(NetworkProxy._held, hold)
        if index is not None:
            NetworkProxy._release(index)

    @staticmethod
    def _reset_after_fork() -> None:
        """Free the lock in a forked child, where the thread that held it no longer exists."""
        NetworkProxy._lock = threading.RLock()
        NetworkProxy._sections = []

    @staticmethod
    def _own_settings() -> Dict[str, Optional[str]]:
        """Return what the open blocks of the running thread or task set, the newest winning; ``None`` is a cleared variable.

        Unlike ``os.environ``, this leaves out the blocks of other threads and tasks, except of the task that started this one.
        """
        entered = _ENTERED.get()
        if not entered:
            return {}
        held = {id(hold[0]) for hold in tuple(NetworkProxy._held)}
        settings: Dict[str, Optional[str]] = {}
        for proxy, applied in entered:
            if id(proxy) in held:
                settings.update(applied)
        return settings

    def _forget_entry(self) -> None:
        """Remove this instance's newest entry of the running thread or task, if it has one."""
        entered = _ENTERED.get()
        for index in range(len(entered) - 1, -1, -1):
            if entered[index][0] is self:
                _ENTERED.set(entered[:index] + entered[index + 1 :])
                return

    @classmethod
    def _write(cls, values: Dict[str, Optional[str]]) -> None:
        """Set each named variable to its value, removing it for ``None``."""
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def __enter__(self) -> "NetworkProxy":
        """
        Enter the context manager and apply its temporary state.

        Use this helper when one operation needs temporary proxy environment variables without leaking them to later code.
        ``backup`` holds the environment that this instance's oldest open hold found. When a setting cannot be applied,
        for example a value that is not a string, the environment is left as it was and no hold is recorded. When an
        exception, such as ``KeyboardInterrupt``, interrupts the call, one hold of this instance is given up before it
        propagates: the one the call recorded, unless an exit that interrupted the call already gave that one up.

        Returns:
            'NetworkProxy': The active proxy context.
        """
        if not self.active:
            return self
        entered: "_Exit" = []
        try:
            NetworkProxy._section(self._apply, entered)
        except BaseException:
            # The caller never reaches ``__exit__`` when ``__enter__`` raises, so a hold recorded here must not stay open.
            for _ in _ATTEMPTS:
                try:
                    NetworkProxy._section(self._give_up, entered)
                    break
                except BaseException:  # another exception cut the recovery short: run it again
                    pass
            raise  # the first exception: a handled inner one no longer counts
        return self

    def _apply(self, entered: "_Exit") -> None:
        """Apply this instance's settings and record the hold in ``entered``; runs in a locked section.

        ``entered`` claims the hold until it is applied, so an exit that interrupts this call never chooses it.
        """
        held = NetworkProxy._held
        depth = len(NetworkProxy._sections)
        found = {name: os.environ.get(name) for name in self.PROXY_VARS}
        applied = {name: None if value == "" else value for name, value in self.settings.items() if value is not None}
        first = not any(entry[0] is self for entry in held)
        number = NetworkProxy._entries
        NetworkProxy._entries = number + 1
        hold: "_Hold" = (self, applied, found, depth, number)
        NetworkProxy._exits.append(entered)
        entered.append(hold)
        try:
            self._write(applied)
            held.append(hold)
        except BaseException:
            if NetworkProxy._find(held, hold) is None:
                self._write(found)
            raise
        if first:
            self.backup = dict(found)
        _ENTERED.set((*(entry for entry in _ENTERED.get() if any(other[0] is entry[0] for other in held)), (self, applied)))
        entered.append(_OPEN)
        NetworkProxy._unregister(entered)

    def _give_up(self, entered: "_Exit") -> None:
        """Give up the hold that an interrupted ``__enter__`` recorded in ``entered``, if any; runs in a locked section."""
        if entered:
            self._leave(entered)
        else:
            NetworkProxy._unregister(entered)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """
        Exit the context manager and restore the previous state.

        Use this helper when one operation needs temporary proxy environment variables without leaking them to later code.
        The newest hold restores the environment it found; an older hold hands what it found to the newer holds, so the
        original environment returns after the last hold of any instance exits. When an exception, such as
        ``KeyboardInterrupt``, interrupts the call after it has begun, the exit is still completed before it propagates.

        Args:
            exc_type (Any): Exception type passed by the context-manager protocol, or ``None`` on a clean exit.
            exc (Any): Exception instance passed by the context-manager protocol, or ``None`` on a clean exit.
            tb (Any): Traceback passed by the context-manager protocol, or ``None`` on a clean exit.

        Returns:
            bool: ``False`` so context-manager exceptions are never swallowed.
        """
        chosen: "_Exit" = []
        try:
            if not self.active:
                return False
            self._forget_entry()
            NetworkProxy._section(self._leave, chosen)
        except BaseException:
            for _ in _ATTEMPTS:
                try:
                    NetworkProxy._section(self._leave, chosen)  # finish an exit that the exception cut short
                    break
                except BaseException:  # another exception cut the recovery short: run it again
                    pass
            raise  # the first exception: a handled inner one no longer counts
        return False

    def _leave(self, chosen: "_Exit") -> None:
        """Give up one open hold of this instance, once, through the claim ``chosen``; runs in a locked section.

        The hold is the newest one that no other claim has chosen. ``chosen`` stays registered in ``_exits`` until its
        hold is released or queued, so a retry after an exception finishes the same exit. A claim that an enter left
        open gives up its own hold when no other exit has chosen it, and otherwise chooses as an exit does.
        """
        held = NetworkProxy._held
        if NetworkProxy._find(NetworkProxy._exits, chosen) is None:
            NetworkProxy._exits.append(chosen)
        while True:
            if not chosen:
                index = next((i for i in range(len(held) - 1, -1, -1) if held[i][0] is self and not NetworkProxy._taken(held[i], chosen)), None)
                if index is None:
                    break
                chosen.append(held[index])
            hold, state = chosen[0], chosen[1] if len(chosen) > 1 else None
            if state is None or state == _OPEN:
                # An exit that interrupted this one between its choice and its claim, or that ran while an enter's claim
                # was open, may have taken the same hold; it went first, so this exit chooses again.
                if NetworkProxy._taken(hold, chosen) or (state == _OPEN and NetworkProxy._find(held, hold) is None):
                    del chosen[:]
                    continue
                del chosen[1:]
            self._settle(chosen)
            break
        NetworkProxy._unregister(chosen)

    @staticmethod
    def _settle(chosen: "_Exit") -> None:
        """Release or queue the hold that ``chosen`` claims, or finish doing so; runs in a locked section."""
        hold = chosen[0]
        index = NetworkProxy._find(NetworkProxy._held, hold)
        if index is None:
            return
        queued = NetworkProxy._deferred
        state = chosen[1] if len(chosen) > 1 else None
        if state == _QUEUED:
            if NetworkProxy._find(queued, hold) is None:
                queued.append(hold)
            return
        depth = len(NetworkProxy._sections)
        # Inside a finalizer or signal handler that interrupted a locked section (depth above 1), release at once only a
        # hold that the interruption itself entered; any other exit waits until the interrupted section finishes.
        if state == _RELEASED or depth == 1 or (hold[3] >= depth and hold[4] >= NetworkProxy._sections[depth - 2]):
            if state is None:
                chosen.append(_RELEASED)
            NetworkProxy._release(index)
        else:
            chosen.append(_QUEUED)
            queued.append(hold)

    @staticmethod
    def _release(index: int) -> None:
        """Remove the hold at ``index`` and restore or hand over what it found; the caller holds the lock.

        An exception that interrupts the restore runs it again before it propagates; running it twice changes nothing.
        """
        held = NetworkProxy._held
        hold = held[index]
        newer = held[index + 1 :]
        try:
            NetworkProxy._restore(hold, newer)
        except BaseException:
            NetworkProxy._restore(hold, newer)
            raise
        finally:
            del held[index]
            if not any(entry[0] is hold[0] for entry in held):
                hold[0].backup = {}

    @staticmethod
    def _restore(hold: "_Hold", newer: List["_Hold"]) -> None:
        """Restore what ``hold`` found, or hand it to the ``newer`` holds that took its variables over."""
        _, applied, found, _, _ = hold
        if not newer:
            NetworkProxy._write(found)
            return
        for name, value in applied.items():
            # Each newer hold up to the first that sets this variable found this hold's value; give it what this hold found.
            taken = False
            for _, newer_applied, newer_found, _, _ in newer:
                if newer_found.get(name) == value:
                    newer_found[name] = found[name]
                if name in newer_applied:
                    taken = True
                    break
            if not taken and os.environ.get(name) == value:
                NetworkProxy._write({name: found[name]})


if hasattr(os, "register_at_fork"):  # POSIX only; Windows has no fork
    os.register_at_fork(after_in_child=NetworkProxy._reset_after_fork)
