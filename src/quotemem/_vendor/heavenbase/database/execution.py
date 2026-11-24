"""SQL execution and transaction helpers for the Database facade."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy.sql.dml import UpdateBase
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.selectable import SelectBase, TextualSelect

from ..utils.typing import Any, List, Optional, Sequence
from .readonly import _construct_text, _is_sql_read, _sql_readonly, _token
from .registry import _thread_token
from .response import SQLResponse


class DatabaseError(RuntimeError):
    """Database execution error."""


def _executed_sql(result: Any) -> Optional[str]:
    """Return the SQL text the driver ran for ``result``, so that classifying it needs no second render."""
    statement = getattr(getattr(result, "context", None), "statement", None)
    return statement if isinstance(statement, str) else None


def _rollback_connection(conn: Any) -> None:
    if conn.dialect.name == "sqlite" and not conn.invalidated:
        driver = conn.connection.driver_connection
        if getattr(driver, "autocommit", None) is False and not driver.in_transaction:
            # RAISE(ROLLBACK) can end SQLite's transaction before SQLAlchemy
            # sees the error. sqlite3 in this mode needs a transaction to roll
            # back, so restore an empty one and let SQLAlchemy reset its state.
            conn.exec_driver_sql("BEGIN")
    conn.rollback()


class _DatabaseExecutionMixin:
    """Provide transaction state and SQL execution helpers for ``Database``."""

    def _init_execution_state(self) -> None:
        self._state = threading.local()
        self._conn = None
        self._tx_readonly: Optional[bool] = None
        self._tx_readonly_set = False
        self._tx_dirty = False
        self._tx_failed = False

    def _state_get(self, name: str, default: Any) -> Any:
        if not hasattr(self._state, name):
            setattr(self._state, name, default)
        return getattr(self._state, name)

    def _state_set(self, name: str, value: Any) -> None:
        setattr(self._state, name, value)

    def _lock_retry_config(self) -> tuple[int, float, float]:
        cfg = dict(self.spec.args.get("lock_retry") or {})
        attempts = max(1, int(cfg.get("attempts", 1) or 1))
        base_delay = max(0.0, float(cfg.get("base_delay", 0.0) or 0.0))
        max_delay = max(base_delay, float(cfg.get("max_delay", base_delay) or base_delay))
        return attempts, base_delay, max_delay

    def _retryable_lock_error(self, exc: Exception) -> bool:
        return self.dialect_spec.is_retryable_error(exc)

    def _execute_with_lock_retry(self, conn: Any, stmt: Any, params: Any = None) -> Any:
        attempts, base_delay, max_delay = self._lock_retry_config()
        for attempt in range(attempts):
            try:
                return conn.execute(stmt, params) if params is not None else conn.execute(stmt)
            except Exception as exc:
                if attempt + 1 >= attempts or not self._retryable_lock_error(exc):
                    raise
                delay = min(max_delay, base_delay * (2**attempt))
                if delay > 0:
                    time.sleep(delay)

    @property
    def _conn(self) -> Any:
        return self._state_get("conn", None)

    @_conn.setter
    def _conn(self, value: Any) -> None:
        self._state_set("conn", value)

    @property
    def _tx_readonly(self) -> Optional[bool]:
        return self._state_get("tx_readonly", None)

    @_tx_readonly.setter
    def _tx_readonly(self, value: Optional[bool]) -> None:
        self._state_set("tx_readonly", value)

    @property
    def _tx_readonly_set(self) -> bool:
        return bool(self._state_get("tx_readonly_set", False))

    @_tx_readonly_set.setter
    def _tx_readonly_set(self, value: bool) -> None:
        self._state_set("tx_readonly_set", value)

    @property
    def _tx_dirty(self) -> bool:
        return bool(self._state_get("tx_dirty", False))

    @_tx_dirty.setter
    def _tx_dirty(self, value: bool) -> None:
        self._state_set("tx_dirty", value)

    @property
    def _tx_schema_dirty(self) -> bool:
        return bool(self._state_get("tx_schema_dirty", False))

    @_tx_schema_dirty.setter
    def _tx_schema_dirty(self, value: bool) -> None:
        self._state_set("tx_schema_dirty", value)

    @property
    def _tx_failed(self) -> bool:
        return bool(self._state_get("tx_failed", False))

    @_tx_failed.setter
    def _tx_failed(self, value: bool) -> None:
        self._state_set("tx_failed", value)

    def __call__(self, readonly: Optional[bool] = False) -> "_DatabaseExecutionMixin":
        """Configure this facade as a transaction context manager.

        Args:
            readonly: Optional transaction-level readonly mode. ``False`` is mutating by default, ``True`` rejects mutating
                statements, and ``None`` enables conservative auto-detection.

        Returns:
            _DatabaseExecutionMixin: This facade, ready for context-manager entry.
        """
        self._entered_conn()  # what an interrupt left set is cleared before the new mode is recorded
        self._tx_readonly = readonly
        self._tx_readonly_set = True
        return self

    def tx(self, readonly: Optional[bool] = False) -> "_DatabaseExecutionMixin":
        """Return this database configured as a transaction context manager.

        Args:
            readonly: Optional transaction-level readonly mode. ``False`` is mutating by default, ``True`` rejects mutating
                statements, and ``None`` enables conservative auto-detection.

        Returns:
            _DatabaseExecutionMixin: This facade, ready for ``with db.tx():``.

        Example:
            ``with db.tx(readonly=None) as tx: tx.execute("SELECT 1")`` auto-detects reads, while leaving default calls
            in the same transaction on the mutating path.
        """
        return self(readonly=readonly)

    def in_transaction(self) -> bool:
        """Return whether this facade currently owns an active transaction.

        Args:
            None.

        Returns:
            bool: ``True`` when this facade owns an open SQLAlchemy transaction.
        """
        return self._entered_conn() is not None and self._conn.in_transaction()

    def _entered_conn(self) -> Any:
        """Return the connection of the transaction this thread entered and has not left, or ``None``.

        A connection that an interrupt left set on the way out of a transaction is closed here, so its uncommitted
        work is rolled back and no later statement of the thread runs in it.
        """
        conn = self._conn
        if conn is None or self._state_get("entered", None) is conn:
            return conn
        try:
            if not self._state_get("joined", False):
                conn.close()
        finally:
            if self._tx_schema_dirty:
                self.invalidate_schema_index()
            self._clear_transaction()
        return None

    @contextmanager
    def _join_transaction(self, owner: "_DatabaseExecutionMixin") -> Iterator[None]:
        """Use the transaction of another facade over the same physical database on this thread."""
        if self._entered_conn() is not None or not owner.in_transaction():
            raise DatabaseError("Joining a transaction requires an idle facade and an active owner")
        try:
            # The owner closes its connection; this facade only forgets it.
            self._state.joined = True
            self._state.entered = self._state.conn = owner._conn
            yield
        finally:
            self._state.entered = None
            owner._tx_dirty = owner._tx_dirty or self._tx_dirty
            owner._tx_schema_dirty = owner._tx_schema_dirty or self._tx_schema_dirty
            owner._tx_failed = owner._tx_failed or self._tx_failed
            self._conn = None
            self._state.joined = False
            self._tx_dirty = False
            self._tx_schema_dirty = False
            self._tx_failed = False

    def commit(self) -> None:
        """Commit the active transaction context, if any.

        Args:
            None.

        Returns:
            None: This method commits the current transaction and starts a fresh transaction inside the same context.

        Raises:
            DatabaseError: If a previous safe execution failed and the transaction must be rolled back first.
        """
        if self._tx_failed:
            raise DatabaseError("Cannot commit after a safe execution failure; call rollback() before continuing")
        if self._conn is not None and self._conn.in_transaction():
            self._finish_transaction(commit=True)
            self._conn.begin()
        self._tx_dirty = False

    def rollback(self) -> None:
        """Rollback the active transaction context, if any.

        Args:
            None.

        Returns:
            None: This method rolls back the current transaction and starts a fresh transaction inside the same context.
        """
        if self._conn is not None and self._conn.in_transaction():
            self._finish_transaction(commit=False)
            self._conn.begin()
        self._tx_dirty = False
        self._tx_failed = False

    def __enter__(self) -> "_DatabaseExecutionMixin":
        """Open a transaction context for this facade.

        Args:
            None.

        Returns:
            _DatabaseExecutionMixin: This facade with an active connection and transaction.

        Raises:
            DatabaseError: If this facade already owns an active transaction.
        """
        if self._entered_conn() is not None:
            raise DatabaseError("Nested Database transactions are not supported; use clone() for parallel work")
        conn = self.engine.connect()
        try:
            conn.begin()
            self._tx_dirty = False
            self._tx_schema_dirty = False
            self._tx_failed = False
        except BaseException:
            # The transaction may be open in the driver: closing the connection rolls it back.
            conn.close()
            raise
        # The connection is set last, in stores that no signal handler runs between, so ``__exit__`` runs for it.
        self._state.entered = conn
        self._state.conn = conn
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Close a transaction context, committing only successful dirty work.

        Args:
            exc_type: Exception type raised inside the context, if any.
            exc_val: Exception value raised inside the context, if any.
            exc_tb: Exception traceback raised inside the context, if any.

        Returns:
            bool: Always ``False`` so caller exceptions propagate normally.
        """
        try:
            self._finish_transaction(commit=exc_type is None and not self._tx_failed and self._tx_dirty)
        finally:
            try:
                if self._conn is not None:
                    self._conn.close()
            finally:
                # First and in one store, so a connection that an interrupt leaves set here is never taken for an open transaction.
                self._state.entered = None
                self._clear_transaction()
        return False

    def _finish_transaction(self, *, commit: bool) -> None:
        """End this transaction and discard any schema reflected while it was open."""
        conn = self._conn
        try:
            if conn is not None and conn.in_transaction():
                conn.commit() if commit else _rollback_connection(conn)
        finally:
            if self._tx_schema_dirty:
                self.invalidate_schema_index()
                self._tx_schema_dirty = False

    def _clear_transaction(self) -> None:
        self._state.entered = None
        self._state.joined = False
        self._conn = None
        self._tx_readonly = None
        self._tx_readonly_set = False
        self._tx_dirty = False
        self._tx_schema_dirty = False
        self._tx_failed = False

    def _readonly(self, stmt: Any, query_text: Optional[str], readonly: Optional[bool], bound: Any = None, *, rendered: Optional[str] = None) -> bool:
        readonly = self._tx_readonly if readonly is False and self._conn is not None and self._tx_readonly_set else readonly
        inferred = _sql_readonly(stmt, query_text, self.dialect, self.dialect_spec, bound, rendered=rendered) if readonly is not False else False
        if readonly is True and not inferred:
            raise DatabaseError("readonly=True rejects mutating SQL")
        return bool(inferred)

    def _is_schema_mutation(self, stmt: Any, query_text: Optional[str], executed: Optional[str] = None) -> bool:
        if executed is not None:
            query_text = executed
        if query_text is None:
            # When the driver does not expose SQL, avoid recompiling ordinary
            # selects and DML. TextualSelect carries literal SQL and may be DDL.
            if isinstance(stmt, (SelectBase, UpdateBase)) and not isinstance(stmt, TextualSelect):
                return False
            query_text = _construct_text(stmt)
        return _token(query_text or "") in {"alter", "create", "drop", "rename", "truncate"}

    def _response(
        self, conn: Any, stmt: Any, params: Any, query_text: Optional[str], readonly: Optional[bool], autocommit: Optional[bool], safe: bool
    ) -> SQLResponse:
        if readonly is False and self._conn is not None and self._tx_readonly_set:
            readonly = self._tx_readonly
        query_readonly = readonly is True and self._readonly(stmt, query_text, readonly, conn.dialect)
        t0 = time.perf_counter()
        try:
            result = self._execute_with_lock_retry(conn, stmt, params)
            elapsed = time.perf_counter() - t0
            resp = SQLResponse.from_cursor(result, query=query_text, params=params, elapsed=elapsed, statement=stmt)
            if self._conn is not None:
                executed = _executed_sql(result)
                if self._is_schema_mutation(stmt, query_text, executed):
                    self._tx_schema_dirty = True
                    self.invalidate_schema_index()
                if readonly is None:
                    query_readonly = self._readonly(stmt, query_text, readonly, conn.dialect, rendered=executed)
                resp.cache()
                if not query_readonly:
                    self._tx_dirty = True
                if autocommit is True and self._conn.in_transaction():
                    self._finish_transaction(commit=True)
                    self._conn.begin()
                    self._tx_dirty = False
            return resp
        except Exception as exc:
            if self._conn is not None:
                self._tx_failed = True
                if autocommit is True and self._conn.in_transaction():
                    self._finish_transaction(commit=False)
                    self._conn.begin()
                    self._tx_dirty = False
                    self._tx_failed = False
            if safe:
                return SQLResponse.from_error(exc, query=query_text, params=params, elapsed=time.perf_counter() - t0, statement=stmt)
            raise DatabaseError(f"Database execution failed: {exc}\nQuery: {query_text or stmt}\nParams: {params}") from exc

    def _exec(
        self,
        stmt: Any,
        params: Any = None,
        *,
        query_text: Optional[str] = None,
        autocommit: Optional[bool] = None,
        readonly: Optional[bool] = False,
        safe: bool = False,
    ) -> SQLResponse:
        if autocommit is False and not self.in_transaction():
            error = DatabaseError("autocommit=False requires a transaction context: use with db.tx():")
            if safe:
                return SQLResponse.from_error(error, query=query_text, params=params, statement=stmt)
            raise error
        if self._entered_conn() is not None:
            return self._response(self._conn, stmt, params, query_text, readonly, autocommit, safe)
        conn = None
        tx = None
        close_response = None
        schema_changed = False

        try:
            conn = self.engine.connect()
            tx = conn.begin()
            query_readonly = readonly is True and self._readonly(stmt, query_text, readonly, conn.dialect)

            def _close_response() -> None:
                try:
                    if tx.is_active:
                        _rollback_connection(conn) if query_readonly else tx.commit()
                finally:
                    conn.close()

            def _discard_response() -> None:
                try:
                    if tx.is_active:
                        _rollback_connection(conn)
                finally:
                    conn.close()

            close_response = _close_response
            t0 = time.perf_counter()
            result = self._execute_with_lock_retry(conn, stmt, params)
            elapsed = time.perf_counter() - t0
            executed = _executed_sql(result)
            if readonly is None:
                query_readonly = self._readonly(stmt, query_text, readonly, conn.dialect, rendered=executed)
            # Only a statement known to be a read streams its rows and ends its transaction
            # when they are consumed or closed. Every other statement commits before it
            # returns, even when its cursor reports rows (RETURNING, or a driver such as
            # DuckDB's that reports rows for every statement): a caller that drops the
            # response would otherwise roll it back.
            if result.returns_rows and (query_readonly or _is_sql_read(stmt, query_text, rendered=executed, dialect=conn.dialect)):
                streamed = SQLResponse.from_cursor(
                    result, query=query_text, params=params, elapsed=elapsed, close=close_response, statement=stmt, discard=_discard_response
                )
                # The connection counts as this thread's in the engine registry until the response ends it; a drop
                # by this thread first settles the response if its iterator was abandoned (see settle_abandoned).
                streamed._owner = (_thread_token(), self.spec._database_key)
                streamed._rollback_only = bool(query_readonly)
                return streamed
            schema_changed = self._is_schema_mutation(stmt, query_text, executed)
            resp = SQLResponse.from_cursor(result, query=query_text, params=params, elapsed=elapsed, statement=stmt).cache()
            _rollback_connection(conn) if query_readonly else tx.commit()
            conn.close()
            return resp
        except Exception as exc:
            # A failed statement never commits: an executemany whose later parameter set failed may have applied the
            # earlier ones on drivers such as sqlite3, and they are rolled back with it.
            try:
                if tx is not None and tx.is_active:
                    _rollback_connection(conn)
            finally:
                conn.close() if conn is not None else None
            if safe:
                return SQLResponse.from_error(exc, query=query_text, params=params, statement=stmt)
            raise DatabaseError(f"Database execution failed: {exc}\nQuery: {query_text or stmt}\nParams: {params}") from exc
        finally:
            if schema_changed:
                self.invalidate_schema_index()

    def execute(
        self,
        query: Any,
        params: Any = None,
        *,
        transpile: Optional[str] = None,
        autocommit: Optional[bool] = None,
        safe: bool = False,
        readonly: Optional[bool] = False,
        **kwargs: Any,
    ) -> SQLResponse:
        """Execute raw SQL or a SQLAlchemy statement.

        Args:
            query: Raw SQL text, ``sqlalchemy.text(...)``, or another SQLAlchemy clause element.
            params: Optional execution parameters. Raw SQL accepts ``:name``, ``?``, ``%s``, ``%(name)s``, ``$name``, and
                ``$1`` style placeholders; they are normalized to SQLAlchemy named binds before execution. Batch execution
                accepts ``List[Dict]`` or ``List[Tuple]`` where supported by the driver.
            transpile: Optional source dialect for SQLGlot transpilation.
            autocommit: ``False`` requires an existing ``with db.tx():`` block. ``None`` and ``True`` both allow standalone execution.
            safe: Return ``SQLResponse(ok=False)`` instead of raising on execution errors.
            readonly: ``False`` executes as mutating by default. ``True`` checks the statement before executing it, using the
                live dialect when it needs to compile a construct. ``None`` classifies the statement for commit or rollback
                after execution, using the SQL the driver ran when classification needs the statement text.
            **kwargs: Reserved for future execution options.

        Returns:
            SQLResponse: Reusable execution response with cached/lazy rows and export helpers. Outside ``db.tx()`` a
            statement known to be a read streams its rows lazily and ends its connection's transaction once they are
            consumed or the response is closed: one that ``readonly`` accepts as read-only, a SELECT construct that
            ``is_sql_readonly`` classifies structurally as one, or text holding one statement led by ``SELECT``,
            ``VALUES``, ``SHOW``, ``DESCRIBE``, ``EXPLAIN`` (not ``EXPLAIN ANALYZE``) or ``WITH`` whose code outside
            comments and quotes names no write verb or ``INTO``. A SELECT construct that ``is_sql_readonly`` classifies
            by its text is judged by the SQL the driver ran for it. Every other
            statement commits before this returns, with any rows it reports already cached. A statement that fails is
            rolled back whole, every parameter set of an executemany included, before the error is raised or, with
            ``safe=True``, returned; on an engine that commits every statement, what ran before the failure stays
            committed. A read that writes through
            a function it calls (a PostgreSQL ``SELECT f()`` whose ``f`` inserts) is streamed as a read: it commits when
            its rows are consumed or the response is closed, and a response dropped unconsumed rolls it back.

        Warnings:
            ``readonly=True`` is a guard, not a SQL firewall. It rejects obvious mutating statements but should not be used as a
            complete security sandbox for untrusted SQL.
        """
        if isinstance(query, ClauseElement):
            return self.orm_execute(query, params=params, autocommit=autocommit, readonly=readonly, safe=safe)
        sql, out_params = self.sql_processor.process(str(query), params, transpile_from=transpile)
        return self._exec(sa.text(sql), out_params, query_text=sql, autocommit=autocommit, readonly=readonly, safe=safe)

    def execute_many(
        self,
        queries: Sequence[Any],
        params: Optional[Sequence[Any]] = None,
        *,
        transpile: Optional[str] = None,
        autocommit: Optional[bool] = None,
        safe: bool = False,
        readonly: Optional[bool] = False,
    ) -> List[SQLResponse]:
        """Execute multiple commands in one transaction context.

        Args:
            queries: SQL strings or SQLAlchemy statements to execute in order.
            params: Optional per-query parameter sequence. Length must match ``queries``.
            transpile: Optional source dialect for raw SQL strings.
            autocommit: Optional inline commit behavior for each statement.
            safe: Return failed responses instead of raising where supported.
            readonly: ``False`` treats all statements as mutating, ``True`` rejects mutating statements, and ``None``
                enables conservative per-statement auto-detection.

        Returns:
            List[SQLResponse]: Responses in input order.

        Warnings:
            The method opens one transaction context around all statements. With ``safe=True`` and ``autocommit`` not set to
            ``True``, the first failed response stops the batch and the context rolls back. Use ``autocommit=True`` only when
            explicit best-effort, per-statement commits are desired.
        """
        args = list(params) if params is not None else [None] * len(queries)
        if len(args) != len(queries):
            raise ValueError("execute_many params length must match queries length")
        with self.tx(readonly=readonly) as tx:
            res = []
            for query, param in zip(queries, args):
                resp = tx.execute(query, param, transpile=transpile, autocommit=autocommit, safe=safe, readonly=readonly)
                res.append(resp)
                if safe and not resp.ok and autocommit is not True:
                    break
            return res

    def orm_execute(
        self, stmt: Any, params: Any = None, *, autocommit: Optional[bool] = None, readonly: Optional[bool] = False, safe: bool = False
    ) -> SQLResponse:
        """Execute a SQLAlchemy statement.

        Args:
            stmt: SQLAlchemy ``ClauseElement`` such as ``select(...)``, ``insert(...)``, or DDL.
            params: Optional execution parameters.
            autocommit: ``False`` requires an existing transaction context.
            readonly: ``False`` executes as mutating by default. ``True`` rejects statements that are not confidently
                read-only. ``None`` enables conservative auto-detection for commit/rollback behavior.
            safe: Return ``SQLResponse(ok=False)`` instead of raising on execution errors.

        Returns:
            SQLResponse: Reusable execution response.
        """
        if not isinstance(stmt, ClauseElement):
            raise TypeError("orm_execute expects a SQLAlchemy ClauseElement")
        return self._exec(stmt, params, autocommit=autocommit, readonly=readonly, safe=safe)

    def orm_execute_many(
        self,
        stmts: Sequence[Any],
        params: Optional[Sequence[Any]] = None,
        *,
        autocommit: Optional[bool] = None,
        readonly: Optional[bool] = False,
        safe: bool = False,
    ) -> List[SQLResponse]:
        """Execute multiple SQLAlchemy statements in one transaction context.

        Args:
            stmts: SQLAlchemy clause elements to execute in order.
            params: Optional per-statement parameter sequence. Length must match ``stmts`` when provided.
            autocommit: Optional inline commit behavior for each statement.
            readonly: ``False`` treats all statements as mutating, ``True`` rejects mutating statements, and ``None``
                enables conservative per-statement auto-detection.
            safe: Return failed responses instead of raising where supported.

        Returns:
            List[SQLResponse]: Responses in input order.

        Warnings:
            With ``safe=True`` and ``autocommit`` not set to ``True``, the first failed response stops the batch and rolls
            back the surrounding transaction. ``autocommit=True`` opts into best-effort per-statement commits.
        """
        args = list(params) if params is not None else [None] * len(stmts)
        if len(args) != len(stmts):
            raise ValueError("orm_execute_many params length must match stmts length")
        with self.tx(readonly=readonly) as tx:
            res = []
            for stmt, param in zip(stmts, args):
                resp = tx.orm_execute(stmt, params=param, autocommit=autocommit, readonly=readonly, safe=safe)
                res.append(resp)
                if safe and not resp.ok and autocommit is not True:
                    break
            return res
