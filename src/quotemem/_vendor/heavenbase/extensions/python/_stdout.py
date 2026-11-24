"""Process-wide stdout capture shared by installed and Registry-loaded runners.

A run installs a capture view as `sys.stdout` that routes each write by the
writing thread. The view is a text stream (an `io.TextIOBase`), as 0.1.2.2's
run buffer was. A run's threads see their run's capture through it. Any other
thread sees the stream the view wraps cut down to a text stream: it can write
to it, flush it and ask about it, but it cannot close, detach, read, seek or
truncate it or reach it through `buffer`, `fileno()` or any other public
attribute beyond a text stream's. During a run, 0.1.2.2 gave every thread the
run's buffer, which refused all of these too, except that a close closed that
buffer. Routing takes no lock: a view reads its thread's own capture and
shared state that only the bookkeeping changes. A run's capture keeps its text
under a lock of its own, which only that run's threads take, and only while
the capture updates that text in memory. No lock of this module is held while
any other stream is written, flushed or closed. So a write, flush or attribute
lookup through a view never waits on a thread that cannot run again.
Interpreter exit stops daemon threads wherever they are, but it flushes
`sys.stdout` from the main thread, which belongs to no run by then. A forked
child inherits the parent's memory but none of its other threads, and it gets
fresh locks for the bookkeeping and for every capture of an active run.

A run's end closes its capture to writes before routing stops sending writes
to it. A write that reaches a capture closed that way goes to the stream the
view wraps instead, so every write reaches either a run's capture or the
stream the view wraps, never neither. A view that wraps None, because
`sys.stdout` was None when it was installed, answers as a null text stream
that drops what it is given, and so does a view whose stream is gone.
Once closed to writes, a capture's text and count no longer change. Closing a
run's `sys.stdout` from one of its threads, as `with sys.stdout:` does, closes
the capture only as a text stream: it still takes the run's writes, as
0.1.2.2's run buffer did. Leaving a `with sys.stdout:` block, or an ExitStack
that entered it, calls the close a program set on the view, if it is still
set, or else closes the capture that was entered, wherever and whenever it is
left. A close() call closes the capture of the calling thread's run, and only
through a live view, one in use by an active run. So a close that one run sets
up and another run's thread performs, such as `contextlib.closing(sys.stdout)`
in a suspended generator, closes nothing once every run that used the view it
holds has ended; while a run that overlapped the one that set it up still uses
that view, it closes the capture of the run whose thread performs it. On a
thread outside runs, a run's own thread after the run ended included, close()
closes nothing, where 0.1.2.2 closed the run's buffer, which every thread
shared.

Each thread keeps the capture it writes to in its own thread-local slot, so a
finished thread leaves nothing behind. Every view a run installs or finds in
`sys.stdout` is kept for the life of the process, because `print()` on Python
3.10 and 3.11 holds `sys.stdout` by a borrowed reference across its writes, so
a view that left `sys.stdout` may still be in use. Such a view holds its
stream strongly only while it is live, that is, until a run ends and no active
run is left that started when the view was installed or later. A retired view
holds the stream weakly, so the host's stream is released as soon as the host
drops it, and the view is reused for a later run once nothing else refers to
it. What was set on a view itself is cleared when it retires and again when a
run takes it up, so it ends with the runs that used the view.
"""

from __future__ import annotations

import _thread
import collections
import contextlib
import functools
import io
import itertools
import os
import sys
import threading
import types
from typing import Any, Callable, Iterable, Iterator, TextIO
import weakref


class _CaptureClosed(ValueError):
    """Report a write that reached a capture after its run ended."""


class _BoundedText(io.TextIOBase):
    """Capture at most one character budget while counting everything written.

    The lock only covers updates of this capture's text in memory; the run's
    end closes the capture to writes, and then it refuses them. Closing it as
    a text stream, as a program's `with sys.stdout:` does, leaves it taking
    writes, as 0.1.2.2's run buffer did.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.chars = 0
        self._parts: list[str] = []
        self._kept = 0
        self._open = True
        self._lock = threading.RLock()

    def write(self, text: str) -> int:
        """Capture the bounded prefix and report the complete write length.

        Raises:
            TypeError: If `text` is not a string.
            _CaptureClosed: If the run that owned this capture has ended.
        """
        if not isinstance(text, str):
            raise TypeError("captured stdout accepts text only")
        text = str.__str__(text)  # a plain copy of a str subclass, so no code of it runs under the lock
        length = len(text)
        with self._lock:
            if not self._open:
                raise _CaptureClosed("the run that owned this stdout capture has ended")
            self.chars += length
            remaining = self.limit - self._kept
            if remaining > 0:
                part = text[:remaining]
                self._parts.append(part)
                self._kept += len(part)
        return length

    def close_capture(self) -> None:
        """Refuse later writes, so the kept text and the count stay as they are now."""
        with self._lock:
            self._open = False

    def snapshot(self) -> tuple[str, int]:
        """Return the retained output prefix and the number of characters written, read together."""
        with self._lock:
            return "".join(self._parts)[: self.limit], self.chars

    def getvalue(self) -> str:
        """Return the retained output prefix."""
        return self.snapshot()[0]


class _NullText(io.TextIOBase):
    """Take text and drop it: the stream that a view of None stands for.

    It is shared by every such view, so closing it does nothing.
    """

    def write(self, text: str) -> int:
        """Drop `text` and report its length.

        Raises:
            TypeError: If `text` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("write() argument must be str")
        return len(text)

    def close(self) -> None:
        """Do nothing, so the shared null stream stays open."""


_NULL = _NullText()


def _gone() -> None:
    """Answer as a dead weak reference, in place of one to a stream that takes none."""


def _weak_ref(stream: TextIO | None) -> Callable[[], TextIO | None] | None:
    """Return a weak reference to `stream`, a dead one if it takes no weak references, or None for None."""
    if stream is None:
        return None
    try:
        return weakref.ref(stream)
    except TypeError:
        return _gone


_TEXT_STREAM_NAMES = frozenset(name for name in dir(io.TextIOBase) if not name.startswith("_"))


class _StreamMethod:
    """A method of a text stream on a view, there exactly when the stream the calling thread writes to has it.

    Looking it up on a view gives the view's own method, which routes the call
    when it is made, so no method of the wrapped stream, and so not the stream
    itself, is handed out. It is a non-data descriptor, so what a program sets
    on the view takes its place.
    """

    def __init__(self, own: Callable[..., Any]) -> None:
        self._own = own
        self._name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, view: Any, owner: type | None = None) -> Any:
        if view is None:
            return self
        capture = getattr(_LOCAL, "capture", None)
        if capture is None or _RUNS.get(capture, -1) < view._since:
            # Outside runs the view has the name exactly when the stream it
            # wraps has it; a run's capture, a text stream, has every one. The
            # stream is read here as `_original` reads it, so a flush runs no
            # more of this module than it did when `flush` was a plain method.
            stream = view._stream
            if stream is None and view._weak is not None:
                stream = view._weak()
            if not hasattr(_NULL if stream is None else stream, self._name):
                raise AttributeError(f"{type(view).__name__!r} object has no attribute {self._name!r}")
        return types.MethodType(self._own, view)

    def __call__(self, view: Any, *args: Any, **kwargs: Any) -> Any:
        return self.__get__(view)(*args, **kwargs)


_UNSET = object()


def _leave(view: _ThreadStdout, entered: _BoundedText | None, *exc_info: Any) -> Any:
    """Leave a `with` block on `view`: call the close a program set on it, else close the capture the block entered."""
    own_close = view.__dict__.get("close", _UNSET)
    if own_close is not _UNSET:
        # As io.IOBase.__exit__ returns `self.close()`, which finds it first.
        return own_close()
    if entered is not None:
        entered.close()
    return None


def _leave_through(capture: _BoundedText | None, order: int, view: _ThreadStdout, *exc_info: Any) -> Any:
    """Leave what `contextlib.ExitStack` entered through `view`, as `_leave` does."""
    return _leave(view, capture if capture is not None and order >= view._since else None)


class _EnteredExit:
    """`__exit__` of a view: leaving a `with` block calls the program's close, or closes only the capture that was entered.

    A `with` statement and `contextlib.ExitStack` look `__exit__` up on the
    entering thread before they call `__enter__`, so the capture that thread
    writes to through the view is known then. Leaving the block calls the
    close a program set on the view, if it is still set, as io.IOBase's
    `__exit__` would; otherwise it closes that capture and nothing else, on
    whichever thread and whenever it is left, and a block entered outside
    runs closes nothing.
    """

    def __get__(self, view: Any, owner: type | None = None) -> Any:
        if view is not None:
            return functools.partial(_leave, view, view._capture())
        capture = getattr(_LOCAL, "capture", None)  # ExitStack looks it up on the class and passes the view later
        return functools.partial(_leave_through, capture, -1 if capture is None else _RUNS.get(capture, -1))


class _ThreadStdout(io.TextIOBase):
    """Stand in for one stream as `sys.stdout` and route each write by thread.

    A thread writes to its run's capture while that run is active, if the run
    started no earlier than this view was installed; every other write goes to
    the stream this view wraps, or is dropped by a null text stream when the
    view wraps None or its stream is gone. The view is a text stream, as
    0.1.2.2's run buffer was, so `isinstance`, `with`, iteration and duck
    typing see a file. On a run's thread it answers as the run's capture,
    whose public attributes it has and `dir()` lists. On any other thread, a
    run's own thread after its run ended included, it has a public name of a
    text stream exactly when the stream it wraps has that name, `write`
    always, and no other public name of that stream, so `buffer`,
    `reconfigure()` and the wrapped stream's other attributes are not there,
    as they were not on 0.1.2.2's run buffer. There, writes and flushes go to
    the wrapped stream, and isatty(), writable(), tell(), closed, encoding,
    errors and newlines answer as it does. detach(), fileno(), read(),
    readline(), readlines(), seek() and truncate() raise
    io.UnsupportedOperation and readable() and seekable() are False, as on
    0.1.2.2's run buffer; close() does nothing, where 0.1.2.2 closed the run's
    buffer. No method the view hands out is the wrapped stream's. close()
    closes the calling thread's run's capture through a live view only, and
    leaving a `with` block calls the close a program set on the view, if it
    is still set, or else closes the capture that was entered. The special
    methods and io.IOBase's private helpers are the view's own. On every
    thread, what is set on the view itself, such as a program's patched
    `write`, is found first and `dir()` lists it; it is cleared when the view
    retires and when a run takes it up again. The view holds its stream
    strongly only while it is live (see `_capture_stdout`) and weakly
    otherwise, except that a copy holds a stream that takes no weak references
    strongly.
    """

    __slots__ = ("_stream", "_weak", "_since", "_kept", "_live", "_queued")

    def __init__(self, stream: TextIO | None, since: int) -> None:
        self._stream = stream
        self._weak = _weak_ref(stream)
        self._since = since
        self._kept = False
        self._live = False
        self._queued = False

    @property
    def _original(self) -> TextIO | None:
        """Return the stream this view wraps, or None when it wraps None or that stream is gone."""
        stream = self._stream
        if stream is None and self._weak is not None:
            stream = self._weak()
        return stream

    def _capture(self) -> _BoundedText | None:
        capture = getattr(_LOCAL, "capture", None)
        if capture is not None and _RUNS.get(capture, -1) >= self._since:
            return capture
        return None

    def _stream_for_thread(self) -> TextIO | _BoundedText:
        capture = self._capture()
        if capture is not None:
            return capture
        stream = self._original
        return _NULL if stream is None else stream

    def _capture_or_refuse(self, name: str) -> _BoundedText:
        """Return the calling thread's capture; outside runs, refuse `name` as 0.1.2.2's run buffer did."""
        capture = self._capture()
        if capture is None:
            raise io.UnsupportedOperation(name)
        return capture

    def write(self, text: str) -> int:
        # A plain method, so a print pays no lookup beyond a method's: every
        # stream that stands as stdout has `write`.
        capture = self._capture()
        if capture is not None:
            try:
                return capture.write(text)
            except _CaptureClosed:
                pass  # the run ended while this write was on its way, so the wrapped stream takes it
        stream = self._original
        return (_NULL if stream is None else stream).write(text)

    def _writelines(self, lines: Iterable[str]) -> None:
        # As io.IOBase.writelines does, refuse a closed stream before any line.
        if getattr(self._stream_for_thread(), "closed", False):
            raise ValueError("I/O operation on closed file.")
        for line in lines:
            self.write(line)

    def _flush(self) -> None:
        # Routed as `write` routes, so a flush looks the thread's capture up once.
        capture = self._capture()
        if capture is not None:
            capture.flush()
        else:
            stream = self._original
            (_NULL if stream is None else stream).flush()

    def _close(self) -> None:
        # Only a run's capture closes through a view, and only through a live
        # one: a view that is not live is a reference kept from runs that
        # have all ended, or a copy that no run took up, so its close, such
        # as a `contextlib.closing` or a `finally` in a generator that a later
        # run's thread ends, closes nothing. The stream the view wraps belongs
        # to the host, so on a thread outside runs, a run's own thread after
        # the run ended included, closing does nothing either.
        capture = self._capture() if self._live else None
        if capture is not None:
            capture.close()

    def _detach(self) -> Any:
        return self._capture_or_refuse("detach").detach()

    def _fileno(self) -> int:
        return self._capture_or_refuse("fileno").fileno()

    def _read(self, *args: Any, **kwargs: Any) -> str:
        return self._capture_or_refuse("read").read(*args, **kwargs)

    def _readable(self) -> bool:
        capture = self._capture()
        return False if capture is None else capture.readable()

    def _readline(self, *args: Any, **kwargs: Any) -> str:
        return self._capture_or_refuse("readline").readline(*args, **kwargs)

    def _readlines(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._capture_or_refuse("readlines").readlines(*args, **kwargs)

    def _isatty(self) -> bool:
        return self._stream_for_thread().isatty()

    def _tell(self) -> int:
        return self._stream_for_thread().tell()

    def _writable(self) -> bool:
        return self._stream_for_thread().writable()

    def _seek(self, *args: Any, **kwargs: Any) -> int:
        return self._capture_or_refuse("seek").seek(*args, **kwargs)

    def _seekable(self) -> bool:
        capture = self._capture()
        return False if capture is None else capture.seekable()

    def _truncate(self, *args: Any, **kwargs: Any) -> int:
        return self._capture_or_refuse("truncate").truncate(*args, **kwargs)

    # Every other method of a text stream is there exactly when the calling
    # thread's stream has it. The view routes writes and flushes itself, so a
    # write reaches a run's capture or the stream the view wraps, and on a
    # thread outside runs it keeps every method that could close, detach,
    # move, read or reach the file under the wrapped stream away from it.
    writelines = _StreamMethod(_writelines)
    flush = _StreamMethod(_flush)
    close = _StreamMethod(_close)
    detach = _StreamMethod(_detach)
    fileno = _StreamMethod(_fileno)
    read = _StreamMethod(_read)
    readable = _StreamMethod(_readable)
    readline = _StreamMethod(_readline)
    readlines = _StreamMethod(_readlines)
    seek = _StreamMethod(_seek)
    seekable = _StreamMethod(_seekable)
    truncate = _StreamMethod(_truncate)
    isatty = _StreamMethod(_isatty)
    tell = _StreamMethod(_tell)
    writable = _StreamMethod(_writable)
    __exit__ = _EnteredExit()

    # The properties of a text stream answer as the calling thread's stream;
    # where it lacks one, the AttributeError falls through to __getattr__,
    # which finds nothing either.
    @property
    def closed(self) -> bool:
        return self._stream_for_thread().closed

    @property
    def encoding(self) -> Any:
        return self._stream_for_thread().encoding

    @property
    def errors(self) -> Any:
        return self._stream_for_thread().errors

    @property
    def newlines(self) -> Any:
        return self._stream_for_thread().newlines

    def __del__(self) -> None:
        # A view never owns the stream it routes to, so dropping one, such as
        # a copy, closes nothing; io.IOBase would close that stream here.
        pass

    def __copy__(self) -> _ThreadStdout:
        copied = _ThreadStdout.__new__(_ThreadStdout)
        # A copy is not live, so it holds a stream that takes weak references
        # only weakly, as a retired view does.
        copied._stream = self._stream if self._weak is _gone else None
        copied._weak, copied._since = self._weak, self._since
        copied._kept = copied._live = copied._queued = False
        return copied

    def __deepcopy__(self, memo: dict[int, Any]) -> _ThreadStdout:
        # A deep copy shares the wrapped stream, as a copy does: copying a
        # file or a terminal is impossible, and a copied stream is not stdout.
        return self.__copy__()

    def __getattr__(self, name: str) -> Any:
        # The view's own fields and special names are never the stream's, so a
        # copy or an object made without __init__ looks them up like any object.
        if name in ("_stream", "_weak", "_since", "_kept", "_live", "_queued", "_original") or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        capture = self._capture()
        if capture is None:
            # A thread outside runs gets nothing of the wrapped stream beyond
            # a text stream, as 0.1.2.2's run buffer gave nothing more: no
            # `buffer`, `reconfigure` or other name that could reach it.
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(capture, name)

    def __dir__(self) -> list[str]:
        # The public names are those the calling thread can look up: its run's
        # capture's, or on a thread outside runs the text-stream names the
        # wrapped stream has, and what is set on the view itself; names that
        # start with an underscore are the view's own.
        own = [name for name in super().__dir__() if name.startswith("_") or name in self.__dict__]
        capture = self._capture()
        if capture is not None:
            names = {name for name in dir(capture) if not name.startswith("_")}
        else:
            names = {"write", *(name for name in dir(self._stream_for_thread()) if name in _TEXT_STREAM_NAMES)}
        return sorted({*own, *names})


# Routing reads `_RUNS` without a lock; only `_LOCK` holders change it, each
# dict read or write is atomic, and the bookkeeping iterates copies. Each
# thread sets its own `_LOCAL.capture`, so a thread's capture goes with it.
_LOCAL = threading.local()  # capture of a run's own thread, or of a thread a run started
_RUNS: dict[_BoundedText, int] = {}  # capture of each active run -> the order the run started in
_ORDER = itertools.count()
_LOCK = threading.RLock()
_restore_thread_hooks: Callable[[], None] | None = None

# Every view this module put in `sys.stdout` or found there, kept for the life
# of the process: `print()` on Python 3.10 and 3.11 holds `sys.stdout` by a
# borrowed reference across its writes, so a view that left `sys.stdout` may
# still be in use by a printing thread and must never be freed. Views are
# small, and a retired one that nothing else refers to is reused. A live view
# holds its stream strongly; a retired view holds it weakly and waits in `_IDLE`
# to stand in for the next stream a run finds in `sys.stdout`. Whether anything
# else still refers to a retired view is read from `sys.getrefcount`, which
# every CPython this package supports has; the count is calibrated on an item
# read exactly as `_attach_view` reads `_IDLE[-1]`, a global deque's item read
# in a function.
_VIEWS: list[_ThreadStdout] = []
_LIVE: list[_ThreadStdout] = []
_IDLE: collections.deque[_ThreadStdout] = collections.deque()
_probe: collections.deque[object] = collections.deque([object()])


def _sole_references() -> int:
    """Return what `sys.getrefcount` reports for a deque item that nothing else holds, read as `_IDLE[-1]` is."""
    return sys.getrefcount(_probe[-1])


_SOLE = _sole_references()
del _probe, _sole_references


def _keep(view: _ThreadStdout) -> None:
    """Keep `view` for the life of the process (`_LOCK` held)."""
    if not view._kept:
        view._kept = True
        _VIEWS.append(view)


def _attach_view(stream: TextIO | None, order: int, released: list[Any]) -> _ThreadStdout:
    """Return a live view of `stream` installed at `order`, reusing a retired one (`_LOCK` held).

    What was set on a reused view is cleared first, its values moved to
    `released`, which the caller drops after it releases `_LOCK`.
    """
    checked = 0
    while _IDLE and checked < 2:
        if _IDLE[-1]._live:
            _IDLE.pop()._queued = False  # it went live again through `sys.stdout`; this entry is stale
            continue
        # A retired view that something else refers to, such as a handler
        # that kept `sys.stdout`, is not reused while it is referenced; it
        # waits at the other end, so each run looks at no more than two.
        checked += 1
        if sys.getrefcount(_IDLE[-1]) <= _SOLE + 1:
            view = _IDLE.pop()
            view._queued = False
            _drop_attributes(view, released)
            view._stream, view._weak, view._since = stream, _weak_ref(stream), order
            view._live = True
            _LIVE.append(view)
            return view
        _IDLE.rotate(1)
    view = _ThreadStdout(stream, order)
    view._live = True
    _keep(view)
    _LIVE.append(view)
    return view


def _hold_installed(view: _ThreadStdout, released: list[Any]) -> None:
    """Make a view found in `sys.stdout` live again, for example one host code put back after it retired (`_LOCK` held).

    What was set on a retired view is cleared first, its values moved to
    `released`, which the caller drops after it releases `_LOCK`.
    """
    _keep(view)
    if not view._live:
        _drop_attributes(view, released)
        view._stream = view._original
        view._live = True
        _LIVE.append(view)


def _drop_attributes(view: _ThreadStdout, released: list[Any]) -> None:
    """Clear what was set on `view` itself, such as a program's patched `write` (`_LOCK` held).

    The values move to `released`, which the caller drops after it releases
    `_LOCK`, so what they keep alive, such as the namespace of the run that set
    them, is freed outside it.
    """
    attributes = view.__dict__
    if attributes:
        released.append(dict(attributes))
        attributes.clear()


def _retire(view: _ThreadStdout, released: list[Any]) -> None:
    """Let `view` hold its stream only weakly and wait in `_IDLE` for reuse (`_LOCK` held).

    The strong reference moves to `released`, which the caller drops after it
    releases `_LOCK`, so a stream that nothing else holds is closed outside it.
    What was set on the view is cleared the same way, so it ends with the run.
    """
    released.append(view._stream)
    view._stream = None
    view._live = False
    _drop_attributes(view, released)
    if not view._queued:
        view._queued = True
        _IDLE.append(view)


def _stream_in_place_of(view: _ThreadStdout) -> TextIO | _ThreadStdout | None:
    """Return what `sys.stdout` becomes when it is `view` and no active run writes through it (`_LOCK` held).

    That is the stream the view wraps, or None when it wraps None. When that
    stream is gone it is `sys.__stdout__`, or the view itself, which drops what
    it is given, where `sys.__stdout__` takes no output: where it is None,
    closed or detached. So host code never finds None, or a stream that
    refuses its output, where it had a stream of its own.
    """
    stream = view._original
    if stream is not None or view._weak is None:
        return stream
    stream = getattr(sys, "__stdout__", None)
    try:
        usable = stream is not None and not getattr(stream, "closed", False)
    except Exception:  # such as the ValueError of a stream detached from its buffer
        usable = False
    return stream if usable else view


def _install_thread_hooks() -> Callable[[], None]:
    """Make threads started by a run's threads inherit their capture; return the undo."""
    original_start = threading.Thread.start
    original_start_new_thread = _thread.start_new_thread

    def start(thread: threading.Thread) -> None:
        capture = getattr(_LOCAL, "capture", None)
        if capture is None or capture not in _RUNS:
            return original_start(thread)
        bootstrap = thread._bootstrap

        def bootstrap_with_capture() -> None:
            _LOCAL.capture = capture
            bootstrap()

        # `Thread.start` hands `self._bootstrap` to the new thread, so this
        # instance attribute runs first there and gives that thread the
        # starting thread's capture; the thread keeps it in its own `_LOCAL`.
        # That is how CPython 3.10 to 3.13 start a thread. On a Python whose
        # `Thread.start` did not, threads started by a run's threads would
        # write to the stream the view wraps, as threads outside runs do.
        thread._bootstrap = bootstrap_with_capture
        try:
            original_start(thread)
        finally:
            thread.__dict__.pop("_bootstrap", None)

    def start_new_thread(function: Any, /, *args: Any) -> int:
        capture = getattr(_LOCAL, "capture", None)
        if capture is None or capture not in _RUNS or not callable(function):
            return original_start_new_thread(function, *args)

        def run(*args: Any, **kwargs: Any) -> Any:
            previous = getattr(_LOCAL, "capture", None)
            _LOCAL.capture = capture
            try:
                return function(*args, **kwargs)
            finally:
                _LOCAL.capture = previous

        return original_start_new_thread(run, *args)

    threading.Thread.start = start
    _thread.start_new_thread = start_new_thread

    def restore() -> None:
        if threading.Thread.start is start:
            threading.Thread.start = original_start
        if _thread.start_new_thread is start_new_thread:
            _thread.start_new_thread = original_start_new_thread

    return restore


def _renew_locks_after_fork() -> None:
    """Give a forked child unheld locks: a thread that may have held one does not exist there."""
    global _LOCK
    _LOCK = threading.RLock()
    for capture in list(_RUNS):
        capture._lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_renew_locks_after_fork)


@contextlib.contextmanager
def _capture_stdout(capture: _BoundedText) -> Iterator[None]:
    """Route the current run and its descendants until this run exits.

    A run installs a capture view as `sys.stdout` unless a view is already
    there. A thread of an active run writes to its run's capture through any
    view installed no later than that run started; every other write through a
    view, including any write from a thread outside runs, goes to the stream
    the view wraps. So a stream that becomes `sys.stdout` after a run started,
    such as a `contextlib.redirect_stdout` entered by its program or by
    another thread, receives that run's output while it is current, as plain
    Python stdout would. While `sys.stdout` is a stream that is not a view,
    every thread writes to it. A view installed while `sys.stdout` was None
    answers threads outside runs as a null text stream: their writes are
    dropped, `isatty()` is False, and closing it does nothing.

    A view is a text stream (an `io.TextIOBase`), as 0.1.2.2's run buffer was,
    so `isinstance`, `with`, iteration and duck typing see a file. On a run's
    thread it answers as the run's capture. On any other thread, which
    includes a run's thread after the run ended, it has a public name of a
    text stream exactly when the stream it wraps has that name, `write`
    always, and no other public name beyond what is set on the view itself.
    Writes and flushes reach that stream, and isatty(), writable(), tell(),
    closed, encoding, errors and newlines answer as it does. detach(),
    fileno(), read(), readline(), readlines(), seek() and truncate() raise
    io.UnsupportedOperation, readable() and seekable() are False, and
    `buffer`, `reconfigure()` and the stream's other names are not there, as
    on 0.1.2.2's run buffer, so no public name reaches that stream beyond
    writes, flushes and those answers; close() does nothing there, where
    0.1.2.2 closed the run's buffer, which every thread shared. Leaving a
    `with sys.stdout:` block, or an ExitStack that entered it, calls the close
    a program set on the view, if it is still set, or else closes the capture
    that was entered, as it closed 0.1.2.2's run buffer, wherever and whenever
    the block is left, and the run's later writes are still captured, as they
    were then. A direct close() closes the calling thread's capture, through a
    live view only, so a close that one run sets up and another run's thread
    performs, such as `contextlib.closing(sys.stdout)` in a suspended
    generator, closes nothing once every run that used the view it holds has
    ended; while a run that overlapped the one that set it up still uses that
    view, it closes the capture of the run whose thread performs it. The
    special methods and io.IOBase's private helpers are the view's own. What
    is set on a view itself, such as a program's patched `write`, is found
    before the stream's names, `dir()` lists it, and it is cleared when the
    view retires and when a run takes it up again, so it ends with the runs
    that used the view, as it ended with 0.1.2.2's run buffer.

    When a run ends and `sys.stdout` is not a live view, or does not exist,
    the run puts back what it found, as 0.1.2.2's `contextlib.redirect_stdout`
    did, also while other runs are active: the view it used stands in again.
    Then, if no active run writes through the view in `sys.stdout`, that is,
    if it was installed after every active run started, the stream it wraps is
    put back. So a program that assigns `sys.stdout` itself (anything but a
    live view), or deletes it, gives the host its stdout back when its run
    ends, and a nested run, on whatever thread, gives its outer run back the
    stdout it found, unless a run that started after the assignment is still
    active: such a run found the assigned stream, so its view stays over that
    stream, which comes back when the last such run ends, as it did when such
    a run ended in 0.1.2.2. A run's end undoes in the same way what host code,
    or the program of a run on another thread, assigned during the run, as
    0.1.2.2's redirect did. With no other run active, for example, a
    `contextlib.redirect_stdout` that host code entered during the run, and
    whose target is `sys.stdout` when the run ends, stops receiving what is
    written there, and one whose target the run found, left during the run,
    has that target put back. A run that starts while `sys.stdout` does not
    exist raises AttributeError, as 0.1.2.2's redirect did, and changes
    nothing.

    A view that host code puts back while no run is active, such as one that a
    redirect entered during a run saved, passes writes from outside runs to
    the stream it wraps while that stream is alive; later runs capture through
    it, and the stream it wraps comes back when a later run ends with no other
    run active. A retired view holds its stream weakly, so a stream that only
    `sys.stdout` refers to is released when such a view takes its place there,
    and a stream that takes no weak references is lost to the view when it
    retires. The view then drops those writes, and instead of the stream
    `sys.__stdout__` comes back, or, where that is None, closed or detached,
    the view stays.

    Every view a run installs, or finds in `sys.stdout` at its start or end,
    stays alive for the life of the process, so a thread still printing
    through one that left `sys.stdout` never uses a freed object. A view that
    a run installs, or finds there at its start, is live, and holds its stream
    strongly, until a run ends and no active run is left that started when the
    view was installed or later. Then it retires and holds the stream weakly,
    so the stream is released, and closed, as soon as the host drops it, on
    the host's thread or after this module's lock is released. So while runs
    overlap without a gap, an older view keeps its stream, where 0.1.2.2's
    redirect let it go when its own run ended. A retired view that nothing
    else refers to is reused for a later run, and one that host code puts back
    is live again once a run starts. A copy or deep copy of a view shares the
    stream it wraps, which it holds weakly if the stream takes weak
    references; it is live only once a run starts while it is in `sys.stdout`,
    so close() through it closes nothing before that.

    A run's end closes its capture to writes before it stops routing writes
    there. A write from one of its threads that reaches the capture after that
    goes to the stream the view wraps, so every write reaches the run's
    capture or that stream, never neither. A thread a run's thread starts
    takes that capture with it and drops it when it ends. A child forked while
    runs are active keeps the routing its forking thread had: forked from a
    thread outside runs it writes to the stream the view wraps, and forked
    from a thread of a run it writes to its own copy of that run's capture, so
    neither the parent's run nor its stream receives that output. The type of
    `sys` and of every module is never changed.
    """
    global _restore_thread_hooks
    dropped: list[Any] = []
    with _LOCK:
        # Read first, so a missing `sys.stdout` raises before anything changes.
        stream = sys.stdout
        order = next(_ORDER)
        if _restore_thread_hooks is None:
            _restore_thread_hooks = _install_thread_hooks()
        if type(stream) is _ThreadStdout:
            _hold_installed(stream, dropped)
            entered = stream
        else:
            entered = sys.stdout = _attach_view(stream, order, dropped)
        del stream
        _RUNS[capture] = order
    del dropped
    previous = getattr(_LOCAL, "capture", None)
    _LOCAL.capture = capture
    try:
        yield
    finally:
        _LOCAL.capture = previous
        capture.close_capture()
        released: list[Any] = []
        with _LOCK:
            del _RUNS[capture]
            active = list(_RUNS.values())
            current = getattr(sys, "stdout", None)
            if type(current) is _ThreadStdout and current._live:
                view = current
            else:
                # `sys.stdout` is missing or is anything but a view that a run
                # uses, such as a stream the run's program assigned: the run
                # puts back what it found, as 0.1.2.2's redirect did, also
                # while other runs are active. What it replaces is dropped
                # after `_LOCK` is released, which closes a stream that nothing
                # else holds; a view, such as a copy of the one the run used,
                # is kept, because a thread may still print through it.
                view = entered
                if type(current) is _ThreadStdout:
                    _keep(current)
                    _retire(current, released)
                else:
                    released.append(current)
            # The view gives way to the stream it stood in for unless an active
            # run still writes through it. A view always wraps a stream that is
            # not a view, so one step is enough.
            if all(started < view._since for started in active):
                sys.stdout = _stream_in_place_of(view)
            elif view is not current:
                sys.stdout = view
            del current, view
            # A view that no active run uses any more retires: it keeps its
            # stream only weakly from now on.
            for live in list(_LIVE):
                if all(started < live._since for started in active):
                    _LIVE.remove(live)
                    _retire(live, released)
            if not _RUNS and _restore_thread_hooks is not None:
                _restore_thread_hooks()
                _restore_thread_hooks = None
        del released, entered
