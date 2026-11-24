"""Reusable SQL execution response."""

from __future__ import annotations

__all__ = ["SQLResponse", "table_display"]

import atexit
import gc
import os
import queue
import sys
import threading
import traceback as tb
from collections import deque

from ..utils.typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union, deepcopy
from ..utils.values import raise_mismatch
from .display import table_display
from .response_exports import _to_numpy, _to_numpy_dict, _to_pandas, _to_pyarrow

ColumnRef = Union[str, int]
RowFormat = Literal["dict", "list", "tuple"]

_DONE = object()

_collecting: Optional[int] = None
"""Identity of the thread running a garbage collection; collections never overlap."""


def _track_collection(phase: str, _info: Dict[str, int]) -> None:
    global _collecting
    _collecting = threading.get_ident() if phase == "start" else None


gc.callbacks.append(_track_collection)

_END_WAIT = 30.0
"""Seconds a call on another thread waits for the transaction of a finished cursor to end (see ``SQLResponse``)."""

_abandoned: "deque[SQLResponse]" = deque()
"""Responses of abandoned iterators that could not be drained where they were dropped, oldest first.

Finalizers only append to it and put a wake-up on ``_wake``: both are safe inside a garbage collection.
"""
_wake: "queue.SimpleQueue[None]" = queue.SimpleQueue()
_handoff = threading.Lock()
"""Guards taking responses from ``_abandoned`` together with ``_current``; never held while settling one."""
_current: "Optional[SQLResponse]" = None
"""The response the settler thread is settling."""
_settler: Optional[threading.Thread] = None
_starting = threading.Lock()
_settling = threading.Lock()
"""Held by the settler thread or the exit hook while it settles a response."""

_pins: Dict[int, Tuple[Any, ...]] = {}
"""Strong references, by response id, to the cursor and callbacks of a lazy response that has an iterator.

A garbage collection that frees such a response together with its abandoned iterator would otherwise reset the
statement's connection, and close its row generator, before the iterator's finalizer could queue the drain.
"""


def _abandon(response: "SQLResponse") -> None:
    """Queue ``response`` for the settler; called by finalizers, so it only appends."""
    _abandoned.append(response)
    _wake.put(None)


def _settle(response: "SQLResponse") -> None:
    """End an abandoned iterator's statement, as dropping the iterator does where that is safe.

    A response that something still references is drained, so its rows stay readable. One that only the queue keeps
    alive, freed by a garbage collection together with its iterator, has its remaining rows read without keeping
    them; when ``readonly`` accepted its statement, whose transaction ends with a rollback anyway, it is only rolled
    back and released.
    """
    try:
        if not response._orphaned:
            response._drain()
        elif response._rollback_only:
            response._release_connection()
        else:
            response._exhaust()
    except Exception:
        print("Exception ignored while closing an abandoned SQLResponse iterator:", file=sys.stderr)
        tb.print_exc()
        response._release_connection()


def _take() -> "Optional[SQLResponse]":
    global _current
    with _handoff:
        _current = _abandoned.popleft() if _abandoned else None
        return _current


def _settle_queued() -> None:
    while True:
        _wake.get()
        while True:
            with _settling:
                response = _take()
                if response is None:
                    break
                _settle(response)
                del response


def _owned(response: "SQLResponse", owner: Tuple[int, str]) -> bool:
    # A response that this thread is reading cannot be settled until that read ends.
    return response._owner == owner and response._reader != threading.get_ident() and response._connected()


def abandoned_connections(token: int) -> Dict[str, int]:
    """Return, by database, how many connections of abandoned responses executed by the thread with ``token`` are still open.

    The settler thread releases them without help from that thread (see ``_settle``), so a thread whose only
    connections on a database are these may wait for that database's drop. ``token`` is the registry's
    ``_thread_token``; a response records it, with its database, when ``Database.execute`` streams it.
    """
    with _handoff:
        pending = list(_abandoned) + ([_current] if _current is not None else [])
    counts: Dict[str, int] = {}
    seen = set()
    for response in pending:
        owner = response._owner
        if owner is None or owner[0] != token or id(response) in seen or not _owned(response, owner):
            continue
        seen.add(id(response))
        counts[owner[1]] = counts.get(owner[1], 0) + 1
    return counts


def settle_abandoned(token: int, database: str) -> None:
    """Settle, on the calling thread, the abandoned responses it executed on ``database``, as the settler would.

    A drop calls this before it checks the thread's connections, so a response that a garbage collection freed
    together with its iterator does not keep the database in use. A response the settler is settling is waited
    for, as long as a call on another thread waits for its transaction to end (see ``SQLResponse``).
    """
    owner = (token, database)
    with _handoff:
        pending = list(_abandoned)
        _abandoned.clear()
        current = _current
    mine = []
    for response in pending:
        if _owned(response, owner):
            if all(response is not other for other in mine):
                mine.append(response)
        else:
            _abandoned.append(response)
    if len(mine) < len(pending):
        _wake.put(None)
    for response in mine:
        _settle(response)
    if current is not None and current._owner == owner and current._reader != threading.get_ident():
        current._drain()


def _start_settler() -> None:
    """Start the thread that drains abandoned responses, outside a garbage collection (starting a thread takes locks)."""
    global _settler
    if (_settler is not None and _settler.is_alive()) or _collecting == threading.get_ident():
        return
    with _starting:
        if _settler is not None and _settler.is_alive():
            return
        thread = threading.Thread(target=_settle_queued, name="heavenbase-sqlresponse-settler", daemon=True)
        try:
            thread.start()
        except RuntimeError:  # interpreter shutdown; the exit hook settles what is queued
            return
        _settler = thread


@atexit.register
def _settle_at_exit() -> None:
    """Settle abandoned iterators before the interpreter exits.

    A collection first frees the reference cycles that still hold a pinned response, so their iterators are
    queued; the interpreter's own final collection runs after the exit hooks, when no thread drains anymore.
    """
    if _pins:
        gc.collect()
    while True:
        with _settling:
            response = _take()
            if response is None:
                break
            _settle(response)
            del response


def _forget_after_fork() -> None:
    # The child has no settler thread, and queued responses use the parent's connections.
    global _abandoned, _wake, _handoff, _current, _settler, _settling, _starting
    _abandoned = deque()
    _wake = queue.SimpleQueue()
    _handoff = threading.Lock()
    _current = None
    _settler = None
    _settling = threading.Lock()
    _starting = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_after_fork)


def _as_row_dict(row: Any) -> Dict[str, Any]:
    return dict(row._mapping) if hasattr(row, "_mapping") else dict(row)


class _ResponseIter:
    def __init__(self, response: "SQLResponse") -> None:
        self.response = response
        self.idx = 0
        self.closed = False
        self._advance_lock = threading.Lock()

    def __iter__(self) -> "_ResponseIter":
        return self

    def __next__(self) -> Dict[str, Any]:
        if not self._advance_lock.acquire(blocking=False):
            raise ValueError("SQLResponse iterator already executing")
        try:
            if self.closed:
                raise StopIteration
            rows = self.response._rows
            row = rows[self.idx] if self.idx < len(rows) else self.response._row(self.idx)
            if row is _DONE:
                self.closed = True
                raise StopIteration
            self.idx += 1
            return dict(row)
        finally:
            self._advance_lock.release()

    def close(self) -> None:
        if not self.closed:
            self.response.cache()
            self.closed = True

    def __del__(self, _get_ident: Callable[[], int] = threading.get_ident) -> None:
        # Closing an abandoned iterator drains its cursor and ends its
        # transaction, which commits what a streamed read wrote through a
        # function it calls unless ``readonly`` accepted it (Database.execute
        # streams only reads and commits every other statement before it
        # returns). Any thread may do that where it drops the iterator: the
        # response serializes its cursor reads. A garbage
        # collection can run this finalizer inside any critical section of any
        # thread, and a thread reading this response is inside one, so both
        # queue the drain for the settler thread instead. The queue keeps the
        # response alive until then, and ``_pins`` keeps its connection and
        # cursor out of a collection that frees the response too.
        response = self.response
        if self.closed or response._row_iter is None:
            return
        thread = _get_ident()
        if _collecting != thread and response._reader != thread:
            # A finalizer does not wait for another thread to end the transaction (see SQLResponse).
            response._drain(wait=False)
            self.closed = True
        else:
            _abandon(response)


class SQLResponse:
    """Unified, reusable result of a SQL execution.

    Rows are cached lazily as callers iterate or export them. Column projection helpers validate requested columns by default;
    pass ``check=False`` when missing column names should be rendered as ``None``.

    A response may represent either success or failure. Failed responses are produced only by ``safe=True`` execution paths;
    call ``raise_on_error()`` to re-raise the original exception, or inspect ``error_type`` and ``error_message`` for display.
    Export helpers drain the underlying cursor once and then reuse the cached rows, so repeated conversions are deterministic.
    Cursor reads are serialized, so separate iterators over one response may be advanced and dropped on different threads.
    Overlapping calls to ``next()`` on the same iterator raise ``ValueError``. Dropping an
    unfinished iterator drains the cursor and ends the statement's own transaction: a rollback when ``readonly`` accepted
    the statement, otherwise a commit, which keeps what a read wrote through a function it calls (``Database.execute``
    streams only reads and commits every other statement, such as ``INSERT ... RETURNING``, before it returns). The thread
    that drops the iterator does this, except inside a garbage collection or inside a read of the same response: the
    ``heavenbase-sqlresponse-settler`` thread then does it soon after, also when the same collection frees the response.
    That thread reads the remaining rows of a response freed this way without keeping them, which costs as much as
    reading them, and the statement keeps its connection until then; when ``readonly`` accepted the statement, whose
    transaction ends with a rollback anyway, it only rolls the statement back and releases the connection. A drop on the
    thread that executed the response settles it first, on that thread (see ``Database.drop_database``). An exit hook
    runs a garbage collection while an iterator over an unfinished response exists, then settles what is queued; an
    iterator that the interpreter's own final collection frees, after the exit hooks, is not drained, and its statement
    rolls back, as in 0.1.2.2. A response dropped before its rows were drained
    or it was closed rolls its statement back and returns the statement's connection at once, where its last reference
    goes, as the pool's reset did once a garbage collection freed the connection; one that a garbage collection frees
    leaves that to the collection, unless an iterator was created over it (see above). The transaction ends on the thread
    that finishes or closes the cursor, after the read lock is released, so code that runs when it ends (such as an engine
    ``commit`` listener) may use the same response. A call on another thread that finds the cursor already finished waits
    for that transaction to end, for at most 30 seconds, so that such code waiting on that thread delays the call instead
    of deadlocking; the finalizer of a dropped iterator does not wait.
    """

    def __init__(self) -> None:
        """Create an empty SQL execution response envelope.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        self.ok = True
        self._columns: List[str] = []
        self._rows: List[Dict[str, Any]] = []
        self._row_count = -1
        self._lastrowid: Any = None
        self._error_type = ""
        self._error_message = ""
        self._exception: BaseException | None = None
        self._query: str | None = None
        self._statement: Any = None
        self._params: Any = None
        self._elapsed: float | None = None
        self._row_iter: Iterator[Any] | None = None
        self._close: Callable[[], None] | None = None
        # Releases the statement's own connection when the response is dropped before ``_close`` ran (see __del__).
        self._discard: Callable[[], None] | None = None
        self._lock = threading.RLock()
        self._reader: Optional[int] = None
        self._ending: Callable[[], None] | None = None
        # Set once the transaction of a finished cursor has ended on thread ``_ender`` (see _release).
        self._ended: threading.Event | None = None
        self._ender: Optional[int] = None
        # Set by ``Database.execute`` on a streamed response: the registry token of the executing thread (which
        # checked out the connection) with the database, and whether its transaction ends with a rollback.
        self._owner: Optional[Tuple[int, str]] = None
        self._rollback_only = False
        # Set when a garbage collection frees this response with an iterator over it (see __del__).
        self._orphaned = False

    @classmethod
    def from_cursor(
        cls,
        result: Any,
        *,
        query: Optional[str] = None,
        params: Any = None,
        elapsed: Optional[float] = None,
        close: Callable[[], None] | None = None,
        statement: Any = None,
        discard: Callable[[], None] | None = None,
    ) -> "SQLResponse":
        """Build a success response from a SQLAlchemy result.

        Args:
            result: SQLAlchemy ``CursorResult`` or compatible object.
            query: Optional SQL text recorded for diagnostics.
            params: Optional execution parameters recorded for diagnostics.
            elapsed: Optional execution duration in seconds.
            close: Optional callback invoked when lazy row consumption finishes.
            statement: Optional SQLAlchemy construct rendered on first ``query`` access when ``query`` is omitted, so
                a successful execution never compiles its statement for diagnostics alone.
            discard: Optional callback that releases the statement's connection when the response is dropped before
                its rows were drained or it was closed.

        Returns:
            SQLResponse: Success response with lazy row iteration when the result returns rows.
        """
        resp = cls()
        resp._query = query
        resp._statement = statement
        resp._params = params
        resp._elapsed = elapsed
        resp._close = close
        resp._discard = discard
        try:
            returns_rows = bool(result.returns_rows)
        except Exception:
            returns_rows = False
        try:
            resp._columns = list(result.keys()) if returns_rows else []
        except Exception:
            resp._columns = []
        try:
            resp._row_count = int(getattr(result, "rowcount", -1))
        except Exception:
            resp._row_count = -1
        if not returns_rows:
            try:
                resp._lastrowid = getattr(result, "lastrowid", None)
            except Exception:
                pass
        if returns_rows:
            resp._row_iter = iter(result.mappings())
        else:
            resp.close()
        return resp

    @classmethod
    def empty(cls, *, query: Optional[str] = None, params: Any = None, elapsed: Optional[float] = None) -> "SQLResponse":
        """Build an empty successful response.

        Args:
            query: Optional SQL text recorded for diagnostics.
            params: Optional execution parameters recorded for diagnostics.
            elapsed: Optional execution duration in seconds.

        Returns:
            A cached response with zero rows.
        """
        resp = cls()
        resp._query = query
        resp._params = params
        resp._elapsed = elapsed
        resp._row_count = 0
        return resp

    @classmethod
    def from_rows(
        cls, rows: Iterable[Dict[str, Any]], *, columns: Optional[List[str]] = None, query: Optional[str] = None, params: Any = None
    ) -> "SQLResponse":
        """Build a cached response from Python row dictionaries.

        Args:
            rows: Iterable of row mappings.
            columns: Optional explicit column order. Defaults to the first row's keys.
            query: Optional query label recorded for diagnostics.
            params: Optional parameter payload recorded for diagnostics.

        Returns:
            SQLResponse: Success response with rows already cached.
        """
        resp = cls()
        resp._rows = [dict(row) for row in rows]
        resp._columns = list(columns or (resp._rows[0].keys() if resp._rows else []))
        resp._row_count = len(resp._rows)
        resp._query = query
        resp._params = params
        return resp

    @classmethod
    def from_error(
        cls,
        exc: BaseException,
        *,
        query: Optional[str] = None,
        params: Any = None,
        elapsed: Optional[float] = None,
        statement: Any = None,
    ) -> "SQLResponse":
        """Build an error response without raising immediately.

        Args:
            exc: Captured exception.
            query: Optional SQL text or statement string.
            params: Optional execution parameters.
            elapsed: Optional elapsed seconds before failure.
            statement: Optional SQLAlchemy construct rendered on first ``query`` access when ``query`` is omitted.

        Returns:
            SQLResponse: ``ok=False`` response preserving exception details.
        """
        resp = cls()
        resp.ok = False
        resp._exception = exc
        resp._error_type = type(exc).__name__
        resp._error_message = str(exc)
        resp._query = query
        resp._statement = statement
        resp._params = params
        resp._elapsed = elapsed
        return resp

    @property
    def columns(self) -> List[str]:
        """Return result column names.

        Returns:
            A detached list in result order.
        """
        return list(self._columns)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the cached table dimensions.

        Returns:
            The ``(row_count, column_count)`` pair.
        """
        self.cache()
        return len(self._rows), len(self._columns)

    @property
    def row_count(self) -> int:
        """Return the driver row count or current cached-row count.

        Returns:
            The affected or cached row count.
        """
        return self._row_count if self._row_count >= 0 else len(self._rows)

    @property
    def lastrowid(self) -> Any:
        """Return the last inserted row identifier.

        Returns:
            The driver-provided identifier, or ``None``.
        """
        return self._lastrowid

    @property
    def error_type(self) -> str:
        """Return the captured error class name.

        Returns:
            The class name, or an empty string for successful responses.
        """
        return self._error_type

    @property
    def error_message(self) -> str:
        """Return the captured error message.

        Returns:
            The message, or an empty string for successful responses.
        """
        return self._error_message

    @property
    def exception(self) -> BaseException | None:
        """Return the captured exception.

        Returns:
            The original exception, or ``None``.
        """
        return self._exception

    @property
    def query(self) -> str | None:
        """Return the recorded SQL text.

        Returns:
            The SQL text, the executed construct rendered on first access, or ``None``.
        """
        if self._query is None and self._statement is not None:
            self._query = str(self._statement)
        return self._query

    @property
    def params(self) -> Any:
        """Return the recorded execution parameters.

        Returns:
            The original parameter payload, or ``None``.
        """
        return self._params

    @property
    def elapsed(self) -> Optional[float]:
        """Return SQL execution duration.

        Returns:
            Elapsed seconds, or ``None``.
        """
        return self._elapsed

    def _ensure_ok(self) -> None:
        if not self.ok and self._exception is not None:
            raise self._exception

    def __del__(self, _get_ident: Callable[[], int] = threading.get_ident) -> None:
        # A response dropped before its rows were drained or it was closed releases its statement's connection here,
        # where its last reference went, and rolls the statement back, as the pool's reset did when a garbage
        # collection freed the connection later (SQLAlchemy's Connection and its transaction refer to each other).
        # So the connection no longer counts as in use by the thread. A garbage collection can run this inside any
        # critical section, so a response it frees leaves the connection to the collection, as before, unless an
        # iterator was created over it: then ``_pins`` keeps the connection from the collection, and the settler
        # thread drains the response, as the finalizer of its abandoned iterator also asks.
        if _collecting == _get_ident():
            if id(self) in _pins and self.__dict__.get("_row_iter") is not None:
                self._orphaned = True
                _abandon(self)
            return
        if "_lock" in self.__dict__ and (self.__dict__.get("_discard") is not None or id(self) in _pins):
            self._release_connection(wait=False)

    def _release_connection(self, *, wait: bool = True) -> None:
        """Roll back and release an unfinished statement's connection, and drop its pin.

        A response built without a ``discard`` callback leaves its connection to whoever holds it, as before.
        """
        outer = self._acquire()
        try:
            _pins.pop(id(self), None)
            discard = self.__dict__.get("_discard")
            if discard is not None and self.__dict__.get("_close") is not None:
                self._ending, self._row_iter, self._close, self._discard = discard, None, None, None
        finally:
            self._release(outer, wait=wait)

    def _connected(self) -> bool:
        """Return whether the statement's connection may still be open: its cursor or its transaction has not ended."""
        ended = self._ended
        return self._close is not None or self._ending is not None or (ended is not None and not ended.is_set())

    def _exhaust(self) -> None:
        """Read the remaining rows without keeping them and end the transaction, for a response nothing else reads."""
        outer = self._acquire()
        try:
            rows = self._row_iter
            if rows is not None:
                for _ in rows:
                    pass
                self._finish()
        finally:
            self._release(outer)

    def _acquire(self) -> Optional[int]:
        # The lock is reentrant: SQLAlchemy result processing runs inside a fetch and may call back into this response.
        self._lock.acquire()
        outer, self._reader = self._reader, threading.get_ident()
        return outer

    def _release(self, outer: Optional[int], *, wait: bool = True) -> None:
        # The outermost release runs the close callback claimed under the lock, after releasing it: ending the
        # transaction fires engine events, whose code may use this response or wait on a thread that does. Another
        # thread's outermost release then waits for that end, at most ``_END_WAIT`` seconds, so that a listener
        # waiting on such a thread delays it instead of deadlocking.
        end = ended = None
        if outer is None:
            end, self._ending = self._ending, None
            if end is not None:
                ended = self._ended = threading.Event()
                self._ender = threading.get_ident()
            elif wait and self._ended is not None and self._ender != threading.get_ident():
                ended = self._ended
        self._reader = outer
        self._lock.release()
        if end is not None:
            try:
                end()
            finally:
                ended.set()  # type: ignore[union-attr]
        elif ended is not None:
            ended.wait(_END_WAIT)

    def _row(self, idx: int) -> Any:
        outer = self._acquire()
        try:
            return self._rows[idx] if idx < len(self._rows) else self._pull()
        finally:
            self._release(outer)

    def _pull(self) -> Any:
        # The caller holds ``_lock``.
        if self._row_iter is None:
            return _DONE
        try:
            row = _as_row_dict(next(self._row_iter))
            self._rows.append(row)
            return row
        except StopIteration:
            self._finish()
            if self._row_count < 0:
                self._row_count = len(self._rows)
            return _DONE

    def _finish(self) -> None:
        # The caller holds ``_lock``; its outermost ``_release`` runs the callback.
        _pins.pop(id(self), None)
        self._row_iter = None
        self._discard = None
        if self._close is not None:
            self._ending, self._close = self._close, None

    def cache(self) -> "SQLResponse":
        """Drain remaining cursor rows into this response.

        Args:
            None.

        Returns:
            This reusable response.

        Raises:
            BaseException: The original exception for a failed response.
        """
        self._ensure_ok()
        self._drain()
        return self

    def _drain(self, *, wait: bool = True) -> None:
        outer = self._acquire()
        try:
            while self._pull() is not _DONE:
                pass
        finally:
            self._release(outer, wait=wait)

    def close(self) -> None:
        """Close the underlying cursor or connection callback.

        Args:
            None.

        Returns:
            None.
        """
        outer = self._acquire()
        try:
            self._finish()
        finally:
            self._release(outer)

    def raise_on_error(self) -> None:
        """Raise the original exception when this response failed.

        Args:
            None.

        Returns:
            None.

        Raises:
            BaseException: The original execution exception.
        """
        self._ensure_ok()

    def traceback(self) -> Optional[str]:
        """Render the captured traceback.

        Args:
            None.

        Returns:
            The formatted traceback, or ``None``.
        """
        return "".join(tb.format_exception(type(self._exception), self._exception, self._exception.__traceback__)) if self._exception else None

    def _column_name(self, col: ColumnRef, *, check: bool = True) -> str:
        if isinstance(col, int):
            try:
                return self._columns[col]
            except IndexError as exc:
                raise ValueError(f"Column index {col} out of range for {len(self._columns)} columns") from exc
        name = str(col)
        if check and name not in self._columns:
            raise ValueError(f"Unknown column {name!r}. Available columns: {self._columns}")
        return name

    def _column_names(self, columns: Optional[List[ColumnRef]] = None, *, check: bool = True) -> List[str]:
        return list(self._columns) if columns is None else [self._column_name(col, check=check) for col in columns]

    def to_list(self, row_fmt: RowFormat = "dict", columns: Optional[List[Union[str, int]]] = None, check: bool = True) -> Any:
        """Export rows as dictionaries, lists, or tuples.

        Args:
            row_fmt: Row shape: ``"dict"``, ``"list"``, or ``"tuple"``.
            columns: Optional column names or indexes to include.
            check: Validate requested column names. Set ``False`` to emit ``None`` for missing column names.

        Returns:
            Any: Rows in the requested shape.

        Raises:
            ValueError: If a requested column is invalid or ``row_fmt`` is unsupported.
        """
        self.cache()
        names = self._column_names(columns, check=check)
        rows = [{name: row.get(name) for name in names} for row in self._rows]
        if row_fmt == "dict":
            return deepcopy(rows)
        if row_fmt == "list":
            return [[row.get(name) for name in names] for row in rows]
        if row_fmt == "tuple":
            return [tuple(row.get(name) for name in names) for row in rows]
        return raise_mismatch(("dict", "list", "tuple"), got=row_fmt, name="SQL response row format")

    def to_dicts(self, columns: Optional[List[Union[str, int]]] = None, check: bool = True) -> List[Dict[str, Any]]:
        """Return rows as list-of-dicts.

        Args:
            columns: Optional names or indexes to project.
            check: Validate requested column names.

        Returns:
            List[Dict[str, Any]]: Deep-copied row dictionaries in result order.
        """
        return self.to_list("dict", columns=columns, check=check)

    def to_lists(self, columns: Optional[List[Union[str, int]]] = None, check: bool = True) -> List[List[Any]]:
        """Return projected rows as lists.

        Args:
            columns: Optional names or indexes to project.
            check: Validate requested columns.

        Returns:
            Projected values in row order.
        """
        return self.to_list("list", columns=columns, check=check)

    def to_pandas(self, *, columns: Optional[List[Union[str, int]]] = None, check: bool = True) -> Any:
        """Return rows as a pandas DataFrame.

        Args:
            columns: Optional names or indexes to project.
            check: Validate requested column names.

        Returns:
            pandas.DataFrame: DataFrame with the requested column order.
        """
        return _to_pandas(self, columns=columns, check=check)

    def to_pyarrow(self, *, columns: Optional[List[Union[str, int]]] = None, schema: Any = None, check: bool = True) -> Any:
        """Return rows as a pyarrow Table.

        Args:
            columns: Optional names or indexes to project.
            schema: Optional pyarrow schema passed to the export adapter.
            check: Validate requested column names.

        Returns:
            pyarrow.Table: Arrow table containing cached rows.
        """
        return _to_pyarrow(self, columns=columns, schema=schema, check=check)

    def to_numpy(self, *columns: str, dtype: Any = None, check: bool = True) -> Any:
        """Return rows as a NumPy array.

        Args:
            *columns: Optional column names to project. Defaults to all columns.
            dtype: Optional NumPy dtype.
            check: Validate requested column names.

        Returns:
            numpy.ndarray: Two-dimensional array shaped ``(rows, columns)``.
        """
        return _to_numpy(self, *columns, dtype=dtype, check=check)

    def to_numpy_dict(self, *, columns: Optional[List[str]] = None, check: bool = True) -> dict[str, Any]:
        """Return projected columns as NumPy arrays.

        Args:
            columns: Optional column names to project.
            check: Validate requested columns.

        Returns:
            A mapping of requested column names to NumPy arrays.
        """
        return _to_numpy_dict(self, columns=columns, check=check)

    def scalar(self) -> Any:
        """Return the first value in the first row.

        Args:
            None.

        Returns:
            The first value, or ``None`` for an empty result.
        """
        first = self.first()
        return next(iter(first.values()), None) if first else None

    def first(self) -> Optional[Dict[str, Any]]:
        """Return the first row as a dictionary.

        Args:
            None.

        Returns:
            A detached row dictionary, or ``None`` for an empty result.
        """
        self.cache()
        return dict(self._rows[0]) if self._rows else None

    def column(self, col: str | int, check: bool = True) -> List[Any]:
        """Return one column as a list.

        Args:
            col: Column name or zero-based index.
            check: Validate the column name or index.

        Returns:
            List[Any]: Values in row order.
        """
        self.cache()
        name = self._column_name(col, check=check)
        return [row.get(name) for row in self._rows]

    def table_display(
        self,
        columns: Optional[List[Union[str, int]]] = None,
        *,
        max_rows: int = 64,
        max_width: int = 64,
        check: bool = True,
        err: bool = False,
        footer: Union[bool, str] = True,
    ) -> str:
        """Render this response as a compact ASCII table.

        Args:
            columns: Optional column names or indexes to display.
            max_rows: Maximum displayed row count.
            max_width: Maximum rendered cell width.
            check: Validate requested columns. Set ``False`` to render missing columns as empty cells.
            err: Include the full traceback for failed responses.
            footer: ``True`` to show the row-count line; ``False`` to suppress all footer text; a ``str`` to append custom text after the row count.

        Returns:
            str: Human-readable table display.
        """
        if not self.ok:
            if footer is False:
                return ""
            trace = self.traceback() if err else None
            parts = [f"{self._error_type}: {self._error_message}"]
            if trace:
                parts.append(trace)
            hint = footer if isinstance(footer, str) else ""
            if hint:
                parts.append(hint)
            return "\n".join(parts)
        self.cache()
        if not self._columns:
            if footer is False:
                return ""
            msg = "The SQL executes successfully but the result table is empty"
            if self._row_count >= 0:
                msg += f" ({self._row_count} rows affected)"
            hint = footer if isinstance(footer, str) else ""
            if hint:
                msg += f"\n{hint}"
            return msg
        if not self._rows:
            if footer is False:
                return table_display(self, columns=columns, max_rows=max_rows, max_width=max_width, check=check, footer=False)
            hint = footer if isinstance(footer, str) else ""
            combined = "The SQL executes successfully but the result table is empty." + (f"\n{hint}" if hint else "")
            return table_display(self, columns=columns, max_rows=max_rows, max_width=max_width, check=check, footer=combined)
        return table_display(self, columns=columns, max_rows=max_rows, max_width=max_width, check=check, footer=footer)

    def clone(self) -> "SQLResponse":
        """Build an independent cached response.

        Args:
            None.

        Returns:
            A deep copy with no live cursor.
        """
        self.cache()
        return deepcopy(self)

    def __getstate__(self) -> Dict[str, Any]:
        """Return copy and pickle state that keeps the rendered SQL text, not the executed construct.

        A SQLAlchemy construct references its tables and their whole ``MetaData``, which can be large or hold
        unpicklable column types and defaults. ``clone()``, ``copy.deepcopy`` and pickling therefore record the
        ``query`` text instead, as they did before the construct was rendered lazily.

        Returns:
            Dict[str, Any]: Instance attributes with ``_query`` rendered, ``_statement`` cleared and no read lock.
        """
        state = self.__dict__.copy()
        state["_query"] = self.query
        state["_statement"] = None
        state.pop("_lock", None)
        state.pop("_reader", None)
        state.pop("_ending", None)
        state.pop("_ended", None)
        state.pop("_ender", None)
        state["_owner"] = None
        state["_orphaned"] = False
        # Releases the original's connection; a copy owns none.
        state.pop("_discard", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore copied or unpickled state with a fresh read lock.

        Args:
            state: Attributes produced by ``__getstate__``.

        Returns:
            None: This function does not return a value.
        """
        self.__dict__.update(state)
        self._statement = None
        self._lock = threading.RLock()
        self._reader = None
        self._ending = None
        self._ended = None
        self._ender = None
        self._owner = None
        self._rollback_only = False
        self._orphaned = False

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        if self._row_iter is not None:
            with self._lock:
                if self._row_iter is not None and id(self) not in _pins:
                    _pins[id(self)] = (self._row_iter, self._close, self._discard)
            _start_settler()
        return _ResponseIter(self)

    def __len__(self) -> int:
        self.cache()
        return len(self._rows)

    def __getitem__(self, idx: Any) -> Any:
        self.cache()
        if isinstance(idx, str):
            return self.column(idx)
        if isinstance(idx, tuple) and len(idx) == 2:
            row, col = idx
            name = self._column_name(col)
            return self._rows[row].get(name)
        return self._rows[idx]

    def __bool__(self) -> bool:
        return self.ok

    def to_str(self, err: bool = False) -> str:
        """Return a compact human-readable summary.

        Args:
            err: Include the traceback for failed responses.

        Returns:
            The rendered response text.
        """
        return self.table_display(err=err)

    def __repr__(self) -> str:
        return f"SQLResponse(ok={self.ok!r}, rows={len(self._rows)}, columns={self._columns!r})"

    def __str__(self) -> str:
        return self.table_display()
