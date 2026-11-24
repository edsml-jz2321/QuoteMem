"""SQL-hosted sparse-GRAM strategy adapter."""

from __future__ import annotations

__all__ = [
    "SQLSparseGramAdapter",
    "SparseGramIndexConflictError",
]

import os
from contextlib import ExitStack, contextmanager, nullcontext
from hashlib import sha256
from typing import NamedTuple
from uuid import uuid4

from heavenbase.utils import Any, Callable, Iterable, Iterator, Sequence
from heavenbase.utils.hash import md5hash
from heavenbase.utils.ids import sql_index_name

from ....execution.frame.result import ResultFrame
from ....schema.model import EntitySchema
from ....strategies import SparseGramIndex
from ....types import ShortText
from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp
from ...base import _RegistrationPublication
from ...strategy import StrategyAdapter
from ..base import (
    _duckdb_covered_columns,
    _duckdb_release_without_row_replacement,
    _mysql_error_code,
    _object_ids_match,
    _refuse_duckdb_row_replacement,
)
from .gram.config import GramConfig
from .gram.runtime import (
    GramKeywordRow,
    GramRowWrite,
    StoredKeywordMatcher,
    contained_in_matches_by_id,
    contained_in_probe_data,
    index_keyword_rows,
    scan_contained_in_matcher,
)

# Fingerprint left in a retired per-workspace namespace. It matches no GramConfig
# fingerprint, so HeavenBase 0.1.2.2, which rebuilds an index whose fingerprint
# changed, rebuilds that namespace from stored rows when it opens the database.
_RETIRED_FINGERPRINT = "retired:shared-namespace"


class SparseGramIndexConflictError(RuntimeError):
    """The shared sparse-GRAM index of a field no longer matches this backend's registration of it."""


class _TablesAbsent:
    """Write-check state of a database that holds no sparse-GRAM index tables."""

    def __repr__(self) -> str:
        return "<sparse-GRAM index tables absent>"


_TABLES_ABSENT = _TablesAbsent()


class _WriteGuard(NamedTuple):
    """The stored index state one write expects, and the bound values of the SQL condition that checks it.

    ``state`` is ``_TABLES_ABSENT``, ``None`` while the entity has no epoch
    row, or the epoch value. ``kind`` names the condition's shape.
    """

    state: Any
    kind: str
    params: dict[str, Any]


def _batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


# Backend types whose row statements can carry the write check, each with the database dialect its SQL was written for
# (see ``SQLSparseGramAdapter.write_guards``). The generic ``sql`` type may reach any server, so it is not one of them;
# nor is ``mssql``, whose lock and write conditions have not been run against a SQL Server.
_GUARDED_BACKEND_TYPES = {
    "sqlite": "sqlite",
    "duckdb": "duckdb",
    "postgres": "postgresql",
    "pgvector": "postgresql",
    "supabase": "postgresql",
    "mysql": "mysql",
    "dolt": "mysql",
    "seekdb": "mysql",
    "oceanbase": "mysql",
    "oracle": "oracle",
}
# Types on the MySQL dialect whose server has InnoDB's shared row and gap locks, and whose transaction reads every row
# as of its first plain read (see ``_SQLGramIndex.lock_entity_rows``). dolt takes no locks at all, and seekdb and
# OceanBase have no shared, gap or table locks and read the latest committed rows in each statement.
_INNODB_BACKEND_TYPES = frozenset({"mysql"})


class _DuckDBWriteGates:
    """Process-wide gates that hold off an entity's DuckDB writes while a registration changes its keyword index.

    DuckDB has no lock statement: ``LOCK TABLE`` and ``SELECT ... FOR
    UPDATE`` or ``FOR SHARE`` do not parse, and concurrent transactions see
    each other's rows only through their snapshots. A database file is open
    for writing in one process at a time (DuckDB locks the file), so a gate
    in this process sees every writer of the file. A write holds its
    entity's gate shared and a fingerprint change holds it exclusively, each
    from before its transaction first reads the file until that transaction
    has committed or rolled back. A connection that read the file before it
    took the gate may read from an older snapshot; ``hold`` refuses it when
    the other side changed the entity since, and the caller writes again.

    Gates are kept per connection in ``Connection.info`` (the pooled DBAPI
    connection's record), and released by engine events after the commit,
    on rollback, or when the connection returns to the pool.
    """

    _HELD = "heavenbase.duckdb_gates.held"
    _COMMITTED = "heavenbase.duckdb_gates.committed"
    _FIRST_READ = "heavenbase.duckdb_gates.first_read"
    _WAIT_STEP = 1.0

    def __init__(self) -> None:
        self._reset()
        self._sequence = 0

    def _reset(self) -> None:
        """Start without a holder, as a forked child must: the parent's other threads do not run in it."""
        import threading

        # Only the plain lock is entered: an interrupted call always leaves it free, while ``Condition.__enter__`` can be
        # interrupted once it has the lock. The condition is only waited on, for at most _WAIT_STEP at a time because an
        # interrupt can come before a notification, and notified.
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        # key -> [shared holders, exclusive holder, last index change, last write commit, exclusive waiters]; see ``hold``
        self._gates: dict[Any, list[Any]] = {}

    def install(self, engine: Any) -> None:
        """Release gates at the end of each transaction of ``engine``; idempotent."""
        from sqlalchemy import event

        if event.contains(engine, "commit", self._on_commit):
            return
        event.listen(engine, "before_cursor_execute", self._on_execute)
        event.listen(engine, "commit", self._on_commit)
        event.listen(engine, "rollback", self._on_rollback)
        event.listen(engine, "begin", self._on_begin)
        for name in ("checkin", "close", "invalidate", "reset"):
            event.listen(engine.pool, name, self._on_pool_event)

    def hold(self, connection: Any, key: Any, *, exclusive: bool, what: str) -> None:
        """Take ``key`` for the transaction of ``connection``, shared or exclusive, waiting for the other side.

        A waiting exclusive request holds off new shared ones, except from a
        thread that already holds the gate shared through another transaction.

        Raises:
            SparseGramIndexConflictError: If the connection read the file before
                taking the gate and the other side changed the entity since, or
                if this thread holds the gate shared through another open transaction.
        """
        import threading

        info = connection.info
        token, thread = id(info), threading.get_ident()
        with self._lock:
            # shared holders (info id -> thread id), exclusive holder, last index change, last write commit, exclusive waiters
            gate = self._gates.setdefault(key, [{}, None, 0, 0, set()])
            if gate[1] == token or (not exclusive and token in gate[0]):
                return
            # named before the gate changes, so that ``_end`` finds whatever an interrupt leaves there
            info.setdefault(self._HELD, set()).add(key)
            try:
                if exclusive:
                    gate[4].add(token)
                while True:
                    readers, writer = gate[0], gate[1]
                    others = {holder: owner for holder, owner in readers.items() if holder != token}
                    if exclusive and thread in others.values():
                        raise SparseGramIndexConflictError(
                            f"Cannot change the keyword index of {what} while this thread writes it in an open transaction; end that transaction first."
                        )
                    if writer in (None, token) and (not others if exclusive else (not gate[4] or thread in readers.values())):
                        break
                    self._changed.wait(self._WAIT_STEP)
            finally:
                if exclusive:
                    gate[4].discard(token)
            first_read = info.get(self._FIRST_READ)
            changed = gate[3] if exclusive else gate[2]
            if first_read is not None and changed > first_read:
                self._changed.notify_all()
                raise SparseGramIndexConflictError(
                    f"{'A write' if exclusive else 'A change of the keyword index'} of {what} committed after this transaction "
                    "first read the database, so it may read from a snapshot without it; roll the transaction back and try again."
                )
            if exclusive:
                gate[1] = token
                gate[0].pop(token, None)
            else:
                gate[0][token] = thread

    def release(self, connection: Any, key: Any) -> None:
        """Release one gate that ``connection`` holds exclusively and has not used to change the index."""
        with self._lock:
            held = connection.info.get(self._HELD) or set()
            if key in held:
                self._free(id(connection.info), {key}, committed=False)
                held.discard(key)

    def _free(self, token: int, held: set[Any], *, committed: bool) -> None:
        self._sequence += 1
        for key in held:
            gate = self._gates.get(key)
            if gate is None:
                continue
            if gate[1] == token:
                if committed:
                    gate[2] = self._sequence
                gate[1] = None
            if token in gate[0]:
                if committed:
                    gate[3] = self._sequence
                gate[0].pop(token)
            gate[4].discard(token)
        self._changed.notify_all()

    def _end(self, info: Any, *, committed: bool) -> None:
        with self._lock:
            pending, held = info.get(self._COMMITTED), info.get(self._HELD)
            info.pop(self._FIRST_READ, None)
            if pending:
                self._free(id(info), pending, committed=True)
            if held:
                self._free(id(info), held, committed=committed)
            # forgotten once free, so that after an interrupt the next event still finds them
            info.pop(self._COMMITTED, None)
            info.pop(self._HELD, None)

    def _on_execute(self, connection: Any, *args: Any) -> None:
        info = connection.info
        if self._FIRST_READ not in info and connection.in_transaction():
            with self._lock:
                info[self._FIRST_READ] = self._sequence

    def _on_commit(self, connection: Any) -> None:
        # this runs before the DBAPI commit: the gates are released once it has finished
        info = connection.info
        with self._lock:
            held = info.get(self._HELD)
            if held:
                info.setdefault(self._COMMITTED, set()).update(held)
            info.pop(self._HELD, None)
            info.pop(self._FIRST_READ, None)

    def _on_rollback(self, connection: Any) -> None:
        self._end(connection.info, committed=False)

    def _on_begin(self, connection: Any) -> None:
        # the previous transaction's commit has finished; gates still held without one were never committed
        self._end(connection.info, committed=False)

    def _on_pool_event(self, dbapi_connection: Any, record: Any, *args: Any) -> None:
        # the pool rolls back a connection it takes back, after any commit has finished
        if record is not None:
            self._end(record.info, committed=False)


_DUCKDB_GATES = _DuckDBWriteGates()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_DUCKDB_GATES._reset)


# MySQL's error for an index name that the table already has
_MYSQL_DUPLICATE_KEY_NAME = 1061
# MySQL's errors for a read of a table created after the transaction's snapshot, and of a missing table
_MYSQL_TABLE_DEFINITION_CHANGED = 1412
_MYSQL_NO_SUCH_TABLE = 1146

# The most bound values besides an IN list's own that one sparse-GRAM statement binds: namespace, entity id
# and field name.
_IN_LIST_OTHER_BINDS = 3


class _SQLGramIndex:
    """SQL table implementation for the SQL sparse-GRAM adapter."""

    def __init__(self, backend: Any, *, namespace: str, page_size: int = 500) -> None:
        """Create a SQL sparse-GRAM table manager.

        Args:
            backend: Owning SQL backend.
            namespace: Namespace that isolates this index's rows from other rows in the same tables.
            page_size: Maximum rows per bulk insert and values per ``IN`` list; an ``IN``
                list is further capped by the dialect's bind limit.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the owning SQL backend has been dropped and has no database.
        """
        from sqlalchemy import MetaData

        if backend.db is None:
            raise RuntimeError(f"SQL backend {backend.name!r} is unavailable after drop")
        self.backend = backend
        self.db = backend.db
        self._mysql_ids = backend._uses_mysql_id_keys()
        self.namespace = namespace
        self.page_size = max(1, int(page_size))
        # how many fingerprint changes this index has made (``lock_entity_rows``)
        self.entity_changes = 0
        # whether a PostgreSQL login may lock metadata rows, once asked (``may_lock_meta``)
        self._meta_lockable: bool | None = None
        self.metadata = MetaData()
        self._define_tables()

    @property
    def in_list_size(self) -> int:
        """Reserve space for the other values bound by each statement."""
        return max(1, min(self.page_size, self.backend._id_list_limit(self.db) - _IN_LIST_OTHER_BINDS))

    def _object_id_batches(self, object_ids: Sequence[str]) -> Iterable[Sequence[str]]:
        limit = self.in_list_size
        if self._mysql_ids:
            # The native and exact comparisons bind the same IDs separately.
            limit = max(1, min(limit, (self.backend._id_list_limit(self.db) - _IN_LIST_OTHER_BINDS) // 2))
        return _batches(object_ids, limit)

    def _object_filter(self, entity_id: str, object_ids: Sequence[str]) -> Any:
        columns = self.keywords.c
        return (columns.ns == self.namespace) & (columns.entity_id == entity_id) & _object_ids_match(columns.object_id, self._mysql_ids, object_ids)

    def _define_tables(self) -> None:
        from heavenbase.database.types import DatabaseIntegerType, DatabaseTextType
        from sqlalchemy import Column, Index, Table

        dialect_spec = getattr(self.db, "dialect_spec", None)

        def text_id() -> Any:
            """Return the SQL type used for sparse-GRAM identifiers.

            Args:
                None.

            Returns:
                Any: Database text type sized for identifiers and hashes.
            """
            return DatabaseTextType(length=191, dialect_spec=dialect_spec)

        def text_long() -> Any:
            """Return the SQL type used for sparse-GRAM long text payloads.

            Args:
                None.

            Returns:
                Any: Database text type sized for keywords and masks.
            """
            return DatabaseTextType(length=65535, dialect_spec=dialect_spec)

        self.meta = Table(
            "hb_gram_meta",
            self.metadata,
            Column("ns", text_id(), primary_key=True),
            Column("key", text_id(), primary_key=True),
            Column("value", text_long(), nullable=False),
            extend_existing=True,
        )
        self.keywords = Table(
            "hb_gram_keywords",
            self.metadata,
            Column("ns", text_id(), nullable=False),
            Column("keyword_id", text_id(), primary_key=True),
            Column("entity_id", text_id(), nullable=False),
            Column("field_name", text_id(), nullable=False),
            Column("object_id", text_id(), nullable=False),
            Column("ordinal", DatabaseIntegerType(), nullable=False),
            Column("keyword", text_long(), nullable=False),
            Column("keyword_norm", text_long(), nullable=False),
            Column("keyword_len", DatabaseIntegerType(), nullable=True),
            Column("normalizer", text_id(), nullable=False),
            Column("full_mask", text_long(), nullable=False),
            Column("part_count", DatabaseIntegerType(), nullable=True),
            extend_existing=True,
        )
        self.postings = Table(
            "hb_gram_postings",
            self.metadata,
            Column("_id", text_id(), primary_key=True),
            Column("ns", text_id(), nullable=False),
            Column("entity_id", text_id(), nullable=False),
            Column("field_name", text_id(), nullable=False),
            Column("gram_hash", text_id(), nullable=False),
            Column("keyword_id", text_id(), nullable=False),
            Column("mask_bit", text_long(), nullable=False),
            extend_existing=True,
        )
        Index(sql_index_name(self.keywords.name, "ns", "entity_id", "field_name"), self.keywords.c.ns, self.keywords.c.entity_id, self.keywords.c.field_name)
        Index(sql_index_name(self.keywords.name, "object"), self.keywords.c.ns, self.keywords.c.entity_id, self.keywords.c.object_id)
        # HeavenBase 0.1.2.2 has no such index; it only speeds up the read of the indexed keyword lengths (see ``ensure``)
        self.width_index = Index(
            sql_index_name(self.keywords.name, "width"),
            self.keywords.c.ns,
            self.keywords.c.entity_id,
            self.keywords.c.field_name,
            self.keywords.c.keyword_len,
        )
        Index(
            sql_index_name(self.postings.name, "lookup"),
            self.postings.c.ns,
            self.postings.c.entity_id,
            self.postings.c.field_name,
            self.postings.c.gram_hash,
        )
        Index(sql_index_name(self.postings.name, "keyword"), self.postings.c.ns, self.postings.c.keyword_id)

    def ensure(self) -> None:
        """Prepare sparse-GRAM metadata, keyword, and posting tables.

        A keyword table that HeavenBase 0.1.2.2 made lacks the index on keyword
        length. Where the server refuses to add it, such as for a login without
        that right, lookups read without it. A table this call creates gets its
        indexes with it, and a refusal there raises.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        tables = [self.meta, self.keywords, self.postings]
        self.db.create_tables(tables)
        self._sync_columns(self.keywords)
        for table in tables:
            try:
                self._create_missing_indexes(table)
            except Exception as exc:
                # SQLAlchemy reads no index from the ``SHOW CREATE TABLE`` of seekdb and OceanBase, so an index that
                # another registration created is created again and refused as a duplicate name
                if self.db.dialect != "mysql" or _mysql_error_code(exc) != _MYSQL_DUPLICATE_KEY_NAME:
                    raise
                self._create_indexes_absent_by_name(table)
        note = getattr(self.backend, "_note_storage_engines", None)
        if note is not None:
            note(tables)

    def _create_missing_indexes(self, table: Any) -> None:
        """Create the indexes of ``table`` whose columns no index on the server covers, as far as DuckDB lets a table with rows take them."""
        existing = self.db.index_column_sets(table)
        missing = [index for index in table.indexes if tuple(str(column.name).lower() for column in index.columns) not in existing]
        covered = _duckdb_covered_columns(self.db, table) if missing else None
        for index in missing:
            if covered is None or covered.issuperset(str(column.name).lower() for column in index.columns):
                self._create_index(index, checkfirst=True)

    def _create_index(self, index: Any, *, checkfirst: bool) -> None:
        """Create one index on a table that already exists, going on without the length index where the server refuses it."""
        from sqlalchemy.exc import DBAPIError

        try:
            self.db.create_index(index, checkfirst=checkfirst)
        except DBAPIError as exc:
            if index is not self.width_index or exc.connection_invalidated:
                raise

    def _create_indexes_absent_by_name(self, table: Any) -> None:
        """Create each declared index of ``table`` that the server does not report by name (MySQL dialect)."""
        from sqlalchemy import text

        rows = self.db.orm_execute(
            text("SELECT DISTINCT index_name FROM information_schema.statistics WHERE table_schema = DATABASE() AND table_name = :name"),
            {"name": table.name},
            readonly=True,
        ).to_lists()
        present = {str(row[0]).lower() for row in rows}
        for index in table.indexes:
            if str(index.name).lower() not in present:
                self._create_index(index, checkfirst=False)

    def _sync_columns(self, table: Any) -> None:
        existing = set(self.db.columns(table.name, schema=table.schema))
        for column in table.c:
            if column.name not in existing:
                self.db.add_table_col(table, column)

    def _tx(self) -> Any:
        return nullcontext(self.db) if self.db.in_transaction() else self.db.tx()

    def set_meta(self, key: str, value: str) -> None:
        """Persist one namespace metadata value.

        Args:
            key: Metadata key within this sparse-GRAM namespace.
            value: Metadata value to store.

        Returns:
            None: This function does not return a value.
        """
        from sqlalchemy import delete, insert

        old_duckdb = _duckdb_release_without_row_replacement(self.db)
        if old_duckdb is not None and self.get_meta(key) is not None:
            _refuse_duckdb_row_replacement(old_duckdb, f"the sparse-GRAM metadata {key!r}")
        with self._tx():
            self.db.orm_execute(delete(self.meta).where((self.meta.c.ns == self.namespace) & (self.meta.c.key == key)), readonly=False)
            self.db.orm_execute(insert(self.meta).values(ns=self.namespace, key=key, value=value), readonly=False)

    def get_meta(self, key: str, *, lock: bool = False) -> str | None:
        """Return one namespace metadata value.

        Args:
            key: Metadata key within this sparse-GRAM namespace.
            lock: Read the row to change the index by it (see ``_read``).

        Returns:
            str | None: Stored metadata value, or ``None`` when absent.
        """
        from sqlalchemy import select

        value = self._read(select(self.meta.c.value).where((self.meta.c.ns == self.namespace) & (self.meta.c.key == key)), lock=lock).scalar()
        return None if value is None else str(value)

    def meta_entries(self, prefix: str) -> dict[str, str]:
        """Return the namespace metadata entries whose keys start with one prefix.

        Args:
            prefix: Literal key prefix; SQL wildcard characters in it match only themselves.

        Returns:
            dict[str, str]: Matching keys and their values.
        """
        from sqlalchemy import select

        scope = (self.meta.c.ns == self.namespace) & self.meta.c.key.startswith(prefix, autoescape=True)
        rows = self._read(select(self.meta.c.key, self.meta.c.value).where(scope)).to_lists()
        # a case-insensitive collation (MySQL) can widen LIKE, so the prefix is checked exactly here
        return {str(key): str(value) for key, value in rows if str(key).startswith(prefix)}

    def meta_entries_any(self, prefixes: Sequence[str], keys: Sequence[str] = (), *, lock: bool = False) -> dict[str, str]:
        """Return the namespace metadata entries whose keys start with any of the prefixes or equal one of the keys, in one query.

        Args:
            prefixes: Literal key prefixes; SQL wildcard characters in them match only themselves.
            keys: Exact keys to read as well.
            lock: Read for the check of a write (see ``_read``).

        Returns:
            dict[str, str]: Matching keys and their values.
        """
        from sqlalchemy import or_, select

        if not prefixes and not keys:
            return {}
        matches = [self.meta.c.key.startswith(prefix, autoescape=True) for prefix in prefixes] + [self.meta.c.key == key for key in keys]
        stmt = select(self.meta.c.key, self.meta.c.value).where((self.meta.c.ns == self.namespace) & or_(*matches))
        rows = self._read(stmt, lock=lock).to_lists()
        # a case-insensitive collation (MySQL) can widen LIKE and equality, so the keys are checked exactly here
        exact = set(keys)
        return {str(key): str(value) for key, value in rows if str(key) in exact or any(str(key).startswith(prefix) for prefix in prefixes)}

    def _read(self, stmt: Any, *, lock: bool = False) -> Any:
        """Run one read of the index tables, as a locking read inside a transaction on the ``mysql`` backend type.

        A MySQL transaction reads every row as of its first plain read. Index
        changes read the metadata this way, so the first plain read of a
        registration that changes an index is the row count or rebuild scan
        after ``lock_entity_rows``, which then sees every committed row.
        PostgreSQL takes a locking read only where ``lock`` asks for it, in the
        check of a write and in a change of an index, and only for a login that
        may (see ``may_lock_meta``): a lookup or an unchanged registration reads
        plainly, as a read-only transaction has to. Other types read plainly:
        dolt has no ``FOR SHARE``, and seekdb and OceanBase take exclusive locks
        for ``LOCK IN SHARE MODE`` and read the latest committed rows anyway.
        """
        if self.innodb and (lock or self.db.in_transaction()):
            return self.db.orm_execute(stmt.with_for_update(read=True), readonly=False)
        if lock and self.db.dialect == "postgresql" and self.db.in_transaction() and self.may_lock_meta():
            # FOR SHARE reads the latest committed rows at READ COMMITTED and, at REPEATABLE READ and SERIALIZABLE, fails
            # with a serialization failure for rows changed after the transaction's snapshot
            return self.db.orm_execute(stmt.with_for_update(read=True), readonly=False)
        return self.db.orm_execute(stmt, readonly=True)

    def may_lock_meta(self) -> bool:
        """Return whether this PostgreSQL login may update the metadata's values, as a ``FOR SHARE`` read of them also requires.

        Asked once per backend connection, when the first entity is registered.
        """
        from sqlalchemy import func, select

        if self._meta_lockable is None:
            answer = self.db.orm_execute(select(func.has_column_privilege(func.to_regclass(self.meta.name), "value", "UPDATE")), readonly=True).scalar()
            if answer is None:  # the table does not exist yet
                return True
            self._meta_lockable = bool(answer)
        return self._meta_lockable

    def tables_created_after_snapshot(self) -> bool:
        """Return whether the index tables were created after this MySQL-protocol transaction's snapshot.

        A REPEATABLE READ transaction reads ``information_schema`` from its
        snapshot too, so it reports the tables absent; a read of a table
        created after the snapshot fails with error 1412 instead, and a read
        of a missing table with 1146. The failed statement leaves the
        transaction open.
        """
        from sqlalchemy.exc import DBAPIError

        name = self.db.engine.dialect.identifier_preparer.format_table(self.meta)
        try:
            # a read of no row is optimized away without opening the table, so one row is read
            self.db._conn.exec_driver_sql(f"SELECT 1 FROM {name} LIMIT 1").all()
        except DBAPIError as exc:
            code = _mysql_error_code(exc)
            if code == _MYSQL_TABLE_DEFINITION_CHANGED:
                return True
            if code == _MYSQL_NO_SUCH_TABLE:
                return False
            raise
        return False

    def refuse_tables_after_snapshot(self, what: str) -> None:
        """Raise ``SparseGramIndexConflictError`` if the index tables were created after this transaction's snapshot (MySQL protocol)."""
        if self.tables_created_after_snapshot():
            raise SparseGramIndexConflictError(
                f"The keyword index tables were created after this transaction's snapshot was taken, so {what} cannot see them; "
                "roll the transaction back and write again."
            )

    def tables_beyond_snapshot(self) -> bool:
        """Return whether PostgreSQL reports the index tables while this transaction's snapshot does not hold them.

        The write condition's ``to_regclass`` reads the latest catalog, while a
        query of ``pg_class`` reads the transaction's snapshot, which at
        ``REPEATABLE READ`` and ``SERIALIZABLE`` was taken by its first
        statement; the rows of tables created after it cannot be read in it.
        """
        from sqlalchemy import text

        if self.db.dialect != "postgresql" or not self.db.in_transaction():
            return False
        query = text("SELECT to_regclass(:name) IS NOT NULL, EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE oid = to_regclass(:name))")
        latest, in_snapshot = self.db.orm_execute(query, {"name": self.meta.name}, readonly=True).to_lists()[0]
        return bool(latest) and not bool(in_snapshot)

    def create_epoch(self, entity_id: str) -> None:
        """Give an entity with no epoch row one, on PostgreSQL, unless another registration does first.

        A write at ``REPEATABLE READ`` or ``SERIALIZABLE`` can only notice a
        change of a row its snapshot holds (see ``write_guard_clause``), so
        every entity registered on a database with the index tables has one.
        """
        from sqlalchemy.dialects.postgresql import insert

        stmt = insert(self.meta).values(ns=self.namespace, key=self.epoch_key(entity_id), value=uuid4().hex).on_conflict_do_nothing()
        self.db.orm_execute(stmt, readonly=False)

    @staticmethod
    def epoch_key(entity_id: str) -> str:
        """Return the metadata key of one entity's epoch, which every change of its stored fingerprints replaces."""
        return f"epoch:{entity_id}"

    def bump_epoch(self, entity_id: str) -> None:
        """Give the entity a new epoch, in the caller's transaction, so every writer's cached index state stops matching."""
        self.set_meta(self.epoch_key(entity_id), uuid4().hex)

    def lock_entity_rows(self, entity_id: str, table: Any) -> None:
        """Wait for the entity's writes in progress and hold off new ones until the caller's transaction ends.

        A change of the entity's stored fingerprints calls this first, inside
        its transaction, then gives the entity a new epoch. A write that began
        earlier therefore commits before the change reads the stored rows, and
        a write that begins later sees the new epoch. On the ``postgres``,
        ``pgvector`` and ``oracle`` backend types this takes a ``SHARE`` lock on
        the entity table, which conflicts with the lock every ``INSERT`` and
        ``DELETE`` takes. On the ``mysql`` type it takes shared locks on
        every row of the table and the gaps between them. SQLite runs one
        write transaction at a time, so the epoch write alone waits for the
        write in progress and holds off the next one. On ``duckdb`` a gate in
        this process does it (see ``_DuckDBWriteGates``). ``dolt``, ``seekdb``,
        ``oceanbase``, ``mssql`` and the generic ``sql`` type take no lock:
        dolt has no locks, and seekdb and OceanBase have only exclusive row
        locks and no gap or table locks, so writers would have to hold off each
        other too.

        Args:
            entity_id: Entity whose epoch the change replaces.
            table: The entity's table on this database, or ``None`` if it has none.
        """
        self.hold_entity_rows(table)
        self.bump_epoch(entity_id)
        self.entity_changes += 1

    @property
    def guarded(self) -> bool:
        """Return whether this backend type is one whose SQL the write conditions were written for."""
        return _GUARDED_BACKEND_TYPES.get(getattr(self.backend, "type", None)) == self.db.dialect

    @property
    def innodb(self) -> bool:
        """Return whether this backend type's server has InnoDB's shared row and gap locks (``mysql``)."""
        return self.db.dialect == "mysql" and getattr(self.backend, "type", None) in _INNODB_BACKEND_TYPES

    def duckdb_gate(self, table: Any) -> Any:
        """Return the key of the process-wide gate of one entity table on a DuckDB file, or ``None`` elsewhere."""
        if table is None or self.db.dialect != "duckdb" or not self.guarded:
            return None
        file_key = self.__dict__.get("_duckdb_file")
        if file_key is None:
            import os

            from heavenbase.utils.files import file_uri_to_path

            spec = self.db.spec
            if spec.in_memory or not spec.database:
                return None  # an in-memory database exists only inside its one connection
            try:
                stat = os.stat(file_uri_to_path(spec.database))
            except (OSError, ValueError):
                return None
            file_key = self._duckdb_file = (stat.st_dev, stat.st_ino)
        _DUCKDB_GATES.install(self.db.engine)
        return (file_key, table.schema, table.name)

    def hold_entity_writes(self, table: Any, *, what: str) -> None:
        """Hold one entity table's DuckDB gate shared for the caller's write transaction (see ``_DuckDBWriteGates``)."""
        key = self.duckdb_gate(table)
        if key is not None and self.db.in_transaction():
            _DUCKDB_GATES.hold(self.db._conn, key, exclusive=False, what=what)

    def hold_entity_rows(self, table: Any) -> None:
        """Take the lock of ``lock_entity_rows`` without changing the epoch."""
        from sqlalchemy import text

        if table is None or not self.guarded:
            return
        name = self.db.engine.dialect.identifier_preparer.format_table(table)
        if self.db.dialect in ("postgresql", "oracle"):
            self.db.orm_execute(text(f"LOCK TABLE {name} IN SHARE MODE"), readonly=False)
        elif self.innodb:
            self.db.orm_execute(text(f"SELECT COUNT(*) FROM (SELECT object_id FROM {name} LOCK IN SHARE MODE) AS hb_locked_rows"), readonly=False)
        elif (key := self.duckdb_gate(table)) is not None and self.db.in_transaction():
            _DUCKDB_GATES.hold(self.db._conn, key, exclusive=True, what=f"table {table.name}")

    def release_entity_rows(self, table: Any) -> None:
        """Release a DuckDB gate taken by ``hold_entity_rows`` before the registration read anything it guards."""
        key = self.duckdb_gate(table)
        if key is not None and self.db.in_transaction():
            _DUCKDB_GATES.release(self.db._conn, key)

    def catalog_name(self, table: Any) -> str:
        """Return a table's name as this database's catalog stores it (upper case on Oracle for a case-insensitive name)."""
        if self.db.dialect == "oracle":
            return str(self.db.engine.dialect.denormalize_name(table.name))
        return str(table.name)

    def write_guard_clause(self, kind: str) -> Any:
        """Return the SQL condition a write statement carries to check that the entity's stored index state is unchanged.

        Args:
            kind: ``"tables"`` (the index tables are absent), ``"no-epoch"``
                (the entity has no epoch row) or ``"epoch"`` (its epoch is ``:hbg_token``).

        Returns:
            Any: A boolean SQLAlchemy clause with bound parameters named ``hbg_*``.
        """
        from sqlalchemy import bindparam, exists, select, text

        dialect = self.db.dialect
        if kind == "tables":
            if dialect == "sqlite":
                return text("NOT EXISTS (SELECT 1 FROM main.sqlite_master WHERE type = 'table' AND name = :hbg_table)")
            if dialect == "postgresql":
                return text("to_regclass(:hbg_table) IS NULL")
            if dialect == "oracle":
                return text("NOT EXISTS (SELECT 1 FROM all_tables WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') AND table_name = :hbg_table)")
            if dialect == "duckdb":
                return text(
                    "NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_catalog = current_database() "
                    "AND table_schema = current_schema() AND table_name = :hbg_table)"
                )
            return text("NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = :hbg_table)")
        scope = (self.meta.c.ns == bindparam("hbg_ns")) & (self.meta.c.key == bindparam("hbg_key"))
        lookup = select(self.meta.c.key).where(scope if kind == "no-epoch" else scope & self._value_is(bindparam("hbg_token")))
        if self.innodb or (dialect == "postgresql" and self.may_lock_meta()):
            # A locking read sees the latest committed epoch whatever the transaction read before, and holds it. At
            # PostgreSQL's REPEATABLE READ and SERIALIZABLE it fails instead when the epoch changed after the snapshot.
            lookup = lookup.with_for_update(read=True)
        return ~exists(lookup) if kind == "no-epoch" else exists(lookup)

    def _value_is(self, value: Any) -> Any:
        """Return the condition that a metadata row's ``value`` equals ``value``.

        The column is a ``CLOB`` on Oracle, which ``=`` cannot compare (ORA-22848).
        """
        from sqlalchemy import func, literal_column

        if self.db.dialect == "oracle":
            return func.dbms_lob.compare(self.meta.c.value, value) == literal_column("0")
        return self.meta.c.value == value

    def namespaces_with_meta(self, key: str) -> dict[str, str]:
        """Return the value every namespace in these tables stores under one metadata key.

        Args:
            key: Metadata key to look up across namespaces.

        Returns:
            dict[str, str]: Namespace to stored value, this index's own namespace included.
        """
        from sqlalchemy import select

        rows = self._read(select(self.meta.c.ns, self.meta.c.value).where(self.meta.c.key == key)).to_dicts()
        return {str(row["ns"]): str(row["value"]) for row in rows}

    def lock_fingerprint(self, key: str, fingerprint: str) -> None:
        """Lock the matching fingerprint row in the caller's write transaction, also for a login without the ``UPDATE`` right.

        PostgreSQL registrations of the entity already wait for each other
        (``lock_registration``), so such a login takes no row lock there. On
        MySQL servers the row is deleted and inserted again, which needs only
        the rights every registration uses.
        """
        from sqlalchemy import delete, insert, update

        scope = (self.meta.c.ns == self.namespace) & (self.meta.c.key == key) & self._value_is(fingerprint)
        if self.db.dialect == "mysql":
            if self.db.orm_execute(delete(self.meta).where(scope), readonly=False)._row_count > 0:
                self.db.orm_execute(insert(self.meta).values(ns=self.namespace, key=key, value=fingerprint), readonly=False)
        elif self.db.dialect != "postgresql" or self.may_lock_meta():
            self.db.orm_execute(update(self.meta).where(scope).values(value=fingerprint), readonly=False)

    def lock_registration(self, entity_id: str) -> None:
        """Serialize PostgreSQL registrations of this entity until publication ends."""
        if self.db.dialect != "postgresql":
            return
        from sqlalchemy import BigInteger, func, literal, select
        from sqlalchemy.dialects.postgresql import OID

        # A metadata row can be absent or replaced by the rebuild. Use a key
        # that stays fixed until all posting and metadata changes commit, and
        # separate indexes in distinct PostgreSQL schemas.
        digest = sha256(f"heavenbase:gram-registration:{self.namespace}:{entity_id}".encode("utf-8")).digest()
        lock_id = int.from_bytes(digest[:8], "big", signed=True)
        relation = func.to_regclass(self.meta.name).cast(OID).cast(BigInteger)
        key = literal(lock_id, type_=BigInteger).op("#")(relation)
        self.db.orm_execute(select(func.pg_advisory_xact_lock(key)), readonly=False)

    def delete_meta(self, key: str) -> None:
        """Remove one namespace metadata value.

        Args:
            key: Metadata key within this sparse-GRAM namespace.

        Returns:
            None: This function does not return a value.
        """
        from sqlalchemy import delete

        with self._tx():
            self.db.orm_execute(delete(self.meta).where((self.meta.c.ns == self.namespace) & (self.meta.c.key == key)), readonly=False)

    def upsert_rows(self, rows: Sequence[GramRowWrite]) -> None:
        """Replace sparse-GRAM keyword and posting rows in batches.

        Args:
            rows: Row write payloads containing keyword and posting records.

        Returns:
            None: This function does not return a value.
        """
        from sqlalchemy import delete, insert, select

        writes_by_key: dict[tuple[str, str], GramRowWrite] = {}
        for item in rows:
            writes_by_key[(str(item.entity_id), str(item.object_id))] = GramRowWrite(
                str(item.entity_id),
                str(item.object_id),
                item.row,
                item.keywords,
                item.postings,
            )
        writes = list(writes_by_key.values())
        if not writes:
            return

        object_ids_by_entity: dict[str, list[str]] = {}
        keyword_records: list[dict[str, Any]] = []
        posting_records: list[dict[str, Any]] = []
        for item in writes:
            object_ids_by_entity.setdefault(item.entity_id, []).append(item.object_id)
            keyword_records.extend(self._keyword_record(keyword) for keyword in item.keywords)
            posting_records.extend(self._posting_record(posting) for posting in item.postings)

        with self._tx():
            old_ids: list[str] = []
            for entity_id, object_ids in object_ids_by_entity.items():
                for batch in self._object_id_batches(object_ids):
                    old_ids.extend(
                        str(keyword_id)
                        for keyword_id in self.db.orm_execute(
                            select(self.keywords.c.keyword_id).where(self._object_filter(entity_id, batch)),
                            readonly=True,
                        ).column(0)
                    )
            self._delete_postings(old_ids)
            for entity_id, object_ids in object_ids_by_entity.items():
                for batch in self._object_id_batches(object_ids):
                    self.db.orm_execute(
                        delete(self.keywords).where(self._object_filter(entity_id, batch)),
                        readonly=False,
                    )
            for batch in _batches(keyword_records, self.page_size):
                self.db.orm_execute(insert(self.keywords), list(batch), readonly=False)
            for batch in _batches(posting_records, self.page_size):
                self.db.orm_execute(insert(self.postings), list(batch), readonly=False)

    def insert_field_rows(self, entity_id: str, field_name: str, rows: Sequence[GramRowWrite]) -> None:
        """Insert the keyword and posting rows of one entity field, after its old rows were deleted.

        Args:
            entity_id: Entity identifier whose field index is written.
            field_name: Field name whose sparse-GRAM rows are written.
            rows: Row write payloads; keywords and postings of other fields are skipped.

        Returns:
            None: This function does not return a value.
        """
        from sqlalchemy import insert

        entity_id = str(entity_id)
        field_name = str(field_name)
        keyword_records: list[dict[str, Any]] = []
        posting_records: list[dict[str, Any]] = []
        for item in rows:
            keyword_records.extend(
                self._keyword_record(keyword) for keyword in item.keywords if str(keyword.entity_id) == entity_id and str(keyword.field_name) == field_name
            )
            posting_records.extend(
                self._posting_record(posting)
                for posting in item.postings
                if str(posting["entity_id"]) == entity_id and str(posting["field_name"]) == field_name
            )
        with self._tx():
            for batch in _batches(keyword_records, self.page_size):
                self.db.orm_execute(insert(self.keywords), list(batch), readonly=False)
            for batch in _batches(posting_records, self.page_size):
                self.db.orm_execute(insert(self.postings), list(batch), readonly=False)

    def delete_objects(self, entity_id: str, object_ids: Iterable[str]) -> None:
        """Delete objects from sparse-GRAM keyword and posting tables.

        Args:
            entity_id: Entity identifier whose object rows should be removed.
            object_ids: Object identifiers to delete.

        Returns:
            None: This function does not return a value.
        """
        from sqlalchemy import delete, select

        ordered_ids = list(dict.fromkeys(str(object_id) for object_id in object_ids))
        if not ordered_ids:
            return
        with self._tx():
            old_ids: list[str] = []
            for batch in self._object_id_batches(ordered_ids):
                old_ids.extend(
                    str(keyword_id)
                    for keyword_id in self.db.orm_execute(
                        select(self.keywords.c.keyword_id).where(self._object_filter(entity_id, batch)),
                        readonly=True,
                    ).column(0)
                )
            self._delete_postings(old_ids)
            for batch in self._object_id_batches(ordered_ids):
                self.db.orm_execute(
                    delete(self.keywords).where(self._object_filter(entity_id, batch)),
                    readonly=False,
                )

    def sweep_objects(self, entity_id: str, object_ids: Iterable[str]) -> None:
        """Delete whatever keyword and posting rows these objects have, reading the latest committed keyword rows.

        A row written without keyword maintenance (``RowOp.strategies=False``)
        has no indexed values, so it should have no keyword rows; another
        writer may still have committed some for the same object before this
        write replaced its row. The keyword IDs are read with an exclusive lock
        on the ``mysql`` backend type, whose plain reads inside a transaction
        see only the rows committed before its first read; every other type
        reads the latest committed rows with a plain read here, or runs one
        writer at a time. A read that finds no keyword row deletes nothing.

        Args:
            entity_id: Entity identifier whose object rows are swept.
            object_ids: Object identifiers whose keyword rows are removed.
        """
        from sqlalchemy import delete, select

        ordered_ids = list(dict.fromkeys(str(object_id) for object_id in object_ids))
        if not ordered_ids:
            return
        with self._tx():
            for batch in self._object_id_batches(ordered_ids):
                stmt = select(self.keywords.c.keyword_id).where(self._object_filter(entity_id, batch))
                if self.innodb:
                    keyword_ids = list(self.db.orm_execute(stmt.with_for_update(), readonly=False).column(0))
                else:
                    keyword_ids = list(self.db.orm_execute(stmt, readonly=True).column(0))
                if not keyword_ids:
                    continue
                self._delete_postings(keyword_ids)
                self.db.orm_execute(delete(self.keywords).where(self._object_filter(entity_id, batch)), readonly=False)

    def object_index_ids(self, entity_id: str, field_names: Sequence[str], object_ids: Sequence[str]) -> dict[str, tuple[set[str], set[str]]]:
        """Return the stored keyword IDs and posting IDs of these objects' indexed fields.

        Args:
            entity_id: Entity identifier whose rows are read.
            field_names: Indexed fields whose keyword rows are read; other fields' rows are left out.
            object_ids: Object identifiers to read.

        Returns:
            dict[str, tuple[set[str], set[str]]]: Keyword IDs and posting IDs by object ID, for objects that have any.
        """
        from sqlalchemy import select

        wanted = set(field_names)
        found: dict[str, tuple[set[str], set[str]]] = {}
        owners: dict[str, str] = {}
        for batch in self._object_id_batches(list(dict.fromkeys(str(object_id) for object_id in object_ids))):
            stmt = select(self.keywords.c.object_id, self.keywords.c.field_name, self.keywords.c.keyword_id).where(self._object_filter(entity_id, batch))
            exact = set(batch)
            for object_id, field_name, keyword_id in self.db.orm_execute(stmt, readonly=True).to_lists():
                # a case-insensitive collation can widen the lookup, so the object is matched exactly here
                if str(object_id) in exact and str(field_name) in wanted:
                    found.setdefault(str(object_id), (set(), set()))[0].add(str(keyword_id))
                    owners[str(keyword_id)] = str(object_id)
        for batch in _batches(list(owners), self.in_list_size):
            stmt = select(self.postings.c.keyword_id, self.postings.c._id).where(
                (self.postings.c.ns == self.namespace) & self.postings.c.keyword_id.in_(list(batch))
            )
            for keyword_id, posting_id in self.db.orm_execute(stmt, readonly=True).to_lists():
                owner = owners.get(str(keyword_id))
                if owner is not None:
                    found[owner][1].add(str(posting_id))
        return found

    def delete_field(self, entity_id: str, field_name: str) -> None:
        """Delete all sparse-GRAM index rows for one entity field.

        Args:
            entity_id: Entity identifier whose field index should be removed.
            field_name: Field name to remove from the sparse-GRAM index.

        Returns:
            None: This function does not return a value.
        """
        from sqlalchemy import delete, select

        with self._tx():
            keyword_ids = [
                str(keyword_id)
                for keyword_id in self.db.orm_execute(
                    select(self.keywords.c.keyword_id).where(
                        (self.keywords.c.ns == self.namespace) & (self.keywords.c.entity_id == entity_id) & (self.keywords.c.field_name == field_name)
                    ),
                    readonly=True,
                ).column(0)
            ]
            self._delete_postings(keyword_ids)
            self.db.orm_execute(
                delete(self.keywords).where(
                    (self.keywords.c.ns == self.namespace) & (self.keywords.c.entity_id == entity_id) & (self.keywords.c.field_name == field_name)
                ),
                readonly=False,
            )

    def retire_field(self, entity_id: str, field_name: str, *meta_keys: str) -> None:
        """Delete one entity field's metadata and then its index rows, in one transaction.

        The metadata goes first, so the fingerprint never vouches for a partly
        deleted index, even on an engine that commits every statement on its
        own: a failure leaves either the whole index with its fingerprint or
        index rows without one, and a registration that binds the field again
        then trusts the whole index or rebuilds it from stored rows.

        Args:
            entity_id: Entity identifier whose field index should be removed.
            field_name: Field name to remove from the sparse-GRAM index.
            *meta_keys: Metadata keys of the field to delete with it, its fingerprint among them.

        Returns:
            None: This function does not return a value.
        """
        with self._tx():
            for key in meta_keys:
                self.delete_meta(key)
            self.delete_field(entity_id, field_name)

    def lookup_candidates(self, entity_id: str, field_name: str, gram_hashes: Iterable[str]) -> list[GramKeywordRow]:
        """Return keyword candidates whose gram mask is covered by query grams.

        Args:
            entity_id: Entity identifier to search.
            field_name: Sparse-GRAM indexed field name.
            gram_hashes: Query gram hashes.

        Returns:
            list[GramKeywordRow]: Candidate keyword rows whose full mask is covered by
            the query grams.
        """
        from sqlalchemy import func, select

        hashes = list(dict.fromkeys(str(gram_hash) for gram_hash in gram_hashes))
        if not hashes:
            return []
        if len(hashes) > self.in_list_size or self.db.dialect == "oracle":
            # Oracle cannot group by the keyword table's CLOB columns or count distinct CLOB masks (ORA-22849)
            return self._lookup_candidates_mask(entity_id, field_name, hashes)
        cols = self._keyword_cols()
        join_expr = self.postings.join(self.keywords, (self.postings.c.ns == self.keywords.c.ns) & (self.postings.c.keyword_id == self.keywords.c.keyword_id))
        rows = self.db.orm_execute(
            select(*cols)
            .select_from(join_expr)
            .where(
                (self.postings.c.ns == self.namespace)
                & (self.postings.c.entity_id == entity_id)
                & (self.postings.c.field_name == field_name)
                & (self.postings.c.gram_hash.in_(hashes))
            )
            .group_by(*cols)
            .having(func.count(func.distinct(self.postings.c.mask_bit)) == self.keywords.c.part_count),
            readonly=True,
        ).to_dicts()
        out = [self._keyword_row(row) for row in rows]
        out.sort(key=lambda item: (item.object_id, item.field_name, item.ordinal, item.keyword_id))
        return out

    def posting_count(self, entity_id: str | None = None, field_name: str | None = None) -> int:
        """Return posting row count, optionally scoped to one entity field.

        Args:
            entity_id: Optional entity identifier to filter by.
            field_name: Optional field name to filter by.

        Returns:
            int: Posting row count for this namespace and optional scope.
        """
        from sqlalchemy import func, select

        stmt = select(func.count()).select_from(self.postings).where(self.postings.c.ns == self.namespace)
        if entity_id is not None:
            stmt = stmt.where(self.postings.c.entity_id == entity_id)
        if field_name is not None:
            stmt = stmt.where(self.postings.c.field_name == field_name)
        return int(self.db.orm_execute(stmt, readonly=True).scalar() or 0)

    def posting_hits(self, entity_id: str, field_name: str, gram_hashes: Iterable[str]) -> int:
        """Return posting rows hit by query gram hashes.

        Args:
            entity_id: Entity identifier to search.
            field_name: Sparse-GRAM indexed field name.
            gram_hashes: Query gram hashes.

        Returns:
            int: Number of posting rows hit by the query grams.
        """
        from sqlalchemy import func, select

        hashes = list(dict.fromkeys(str(gram_hash) for gram_hash in gram_hashes))
        if not hashes:
            return 0
        total = 0
        for batch in _batches(hashes, self.in_list_size):
            total += int(
                self.db.orm_execute(
                    select(func.count())
                    .select_from(self.postings)
                    .where(
                        (self.postings.c.ns == self.namespace)
                        & (self.postings.c.entity_id == entity_id)
                        & (self.postings.c.field_name == field_name)
                        & (self.postings.c.gram_hash.in_(batch))
                    ),
                    readonly=True,
                ).scalar()
                or 0
            )
        return total

    def posting_stats(self, entity_id: str | None = None, field_name: str | None = None, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most common posting hashes for inspection.

        Args:
            entity_id: Optional entity identifier to filter by.
            field_name: Optional field name to filter by.
            limit: Maximum number of posting hash rows to return.

        Returns:
            list[dict[str, Any]]: Posting hash rows with ``gram_hash`` and ``posting_count``.
        """
        from sqlalchemy import func, select

        count_expr = func.count().label("posting_count")
        stmt = (
            select(self.postings.c.gram_hash, count_expr)
            .where(self.postings.c.ns == self.namespace)
            .group_by(self.postings.c.gram_hash)
            .order_by(count_expr.desc(), self.postings.c.gram_hash)
            .limit(max(1, int(limit)))
        )
        if entity_id is not None:
            stmt = stmt.where(self.postings.c.entity_id == entity_id)
        if field_name is not None:
            stmt = stmt.where(self.postings.c.field_name == field_name)
        rows = self.db.orm_execute(stmt, readonly=True).to_dicts()
        return [{"gram_hash": str(row["gram_hash"]), "posting_count": int(row["posting_count"])} for row in rows]

    def short_widths(self, entity_id: str, field_name: str, max_width: int) -> set[int]:
        """Return indexed keyword lengths up to ``max_width`` for one field.

        Every query reads them afresh: another backend instance or process
        that shares these rows may have indexed a short keyword since, and a
        missing width would drop that keyword's matches. One range query on
        the width index returns all distinct widths for the field.

        Args:
            entity_id: Entity identifier to inspect.
            field_name: Sparse-GRAM indexed field name.
            max_width: Inclusive maximum keyword width.

        Returns:
            set[int]: Indexed keyword lengths up to ``max_width``.
        """
        from sqlalchemy import select

        scope = (self.keywords.c.ns == self.namespace) & (self.keywords.c.entity_id == entity_id) & (self.keywords.c.field_name == field_name)
        width = self.keywords.c.keyword_len
        stmt = select(width).distinct().where(scope, width.between(1, max_width))
        return {int(value) for value in self.db.orm_execute(stmt, readonly=True).column(0)}

    def has_field(self, entity_id: str, field_name: str) -> bool:
        """Return whether this namespace holds any keyword row for one field.

        Args:
            entity_id: Entity identifier to inspect.
            field_name: Sparse-GRAM indexed field name.

        Returns:
            bool: ``True`` when at least one keyword row is indexed for the field.
        """
        from sqlalchemy import select

        scope = (self.keywords.c.ns == self.namespace) & (self.keywords.c.entity_id == entity_id) & (self.keywords.c.field_name == field_name)
        return self._read(select(self.keywords.c.keyword_id).where(scope).limit(1)).scalar() is not None

    def drop(self) -> None:
        """Delete all sparse-GRAM rows in this backend namespace.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        from sqlalchemy import delete

        with self._tx():
            # the metadata first, so no fingerprint is left vouching for partly deleted rows
            for table in (self.meta, self.postings, self.keywords):
                self.db.orm_execute(delete(table).where(table.c.ns == self.namespace), readonly=False)

    def _delete_postings(self, keyword_ids: Iterable[str]) -> None:
        from sqlalchemy import delete

        for batch in _batches(list(dict.fromkeys(str(keyword_id) for keyword_id in keyword_ids)), self.in_list_size):
            self.db.orm_execute(delete(self.postings).where((self.postings.c.ns == self.namespace) & (self.postings.c.keyword_id.in_(batch))), readonly=False)

    def _lookup_candidates_mask(self, entity_id: str, field_name: str, hashes: Sequence[str]) -> list[GramKeywordRow]:
        """Return candidates with a Python mask fallback for very large query hash sets."""
        from sqlalchemy import select

        mask_by_keyword: dict[str, int] = {}
        candidate_by_keyword: dict[str, GramKeywordRow] = {}
        join_expr = self.postings.join(self.keywords, (self.postings.c.ns == self.keywords.c.ns) & (self.postings.c.keyword_id == self.keywords.c.keyword_id))
        for batch in _batches(hashes, self.in_list_size):
            rows = self.db.orm_execute(
                select(self.postings.c.mask_bit, *self._keyword_cols())
                .select_from(join_expr)
                .where(
                    (self.postings.c.ns == self.namespace)
                    & (self.postings.c.entity_id == entity_id)
                    & (self.postings.c.field_name == field_name)
                    & (self.postings.c.gram_hash.in_(batch))
                ),
                readonly=True,
            ).to_dicts()
            for row in rows:
                keyword_id = str(row["keyword_id"])
                mask_by_keyword[keyword_id] = mask_by_keyword.get(keyword_id, 0) | int(row["mask_bit"])
                if keyword_id not in candidate_by_keyword:
                    candidate_by_keyword[keyword_id] = self._keyword_row(row)
        out = [
            candidate
            for keyword_id, candidate in candidate_by_keyword.items()
            if mask_by_keyword.get(keyword_id, 0) & candidate.full_mask == candidate.full_mask
        ]
        out.sort(key=lambda item: (item.object_id, item.field_name, item.ordinal, item.keyword_id))
        return out

    def _keyword_cols(self) -> list[Any]:
        return [
            self.keywords.c.entity_id,
            self.keywords.c.field_name,
            self.keywords.c.keyword_id,
            self.keywords.c.object_id,
            self.keywords.c.ordinal,
            self.keywords.c.keyword,
            self.keywords.c.keyword_norm,
            self.keywords.c.normalizer,
            self.keywords.c.full_mask,
            self.keywords.c.part_count,
        ]

    def _stored_keyword_id(self, keyword_id: str) -> str:
        """Return the table-wide keyword id of one keyword in this namespace.

        ``hb_gram_keywords.keyword_id`` is the table's whole primary key, while
        one database can hold the same keyword under two namespaces: HeavenBase
        0.1.2.2 and earlier indexed rows per workspace, and a database that
        several workspaces shared keeps those rows until each workspace opens
        it again. The namespace is therefore part of the stored id; every read
        and delete selects ids by namespace.
        """
        return md5hash([self.namespace, keyword_id])

    def _keyword_record(self, row: GramKeywordRow) -> dict[str, Any]:
        return {
            "ns": self.namespace,
            "entity_id": row.entity_id,
            "field_name": row.field_name,
            "keyword_id": self._stored_keyword_id(row.keyword_id),
            "object_id": row.object_id,
            "ordinal": row.ordinal,
            "keyword": row.keyword,
            "keyword_norm": row.keyword_norm,
            "keyword_len": row.unit_count or len(row.keyword_norm),
            "normalizer": row.normalizer,
            "full_mask": str(row.full_mask),
            "part_count": row.part_count,
        }

    def _posting_record(self, row: dict[str, Any]) -> dict[str, Any]:
        keyword_id = self._stored_keyword_id(str(row["keyword_id"]))
        gram_hash = str(row["gram_hash"])
        mask_bit = str(row["mask_bit"])
        return {
            "_id": md5hash([self.namespace, keyword_id, gram_hash, mask_bit]),
            "ns": self.namespace,
            "entity_id": str(row["entity_id"]),
            "field_name": str(row["field_name"]),
            "gram_hash": gram_hash,
            "keyword_id": keyword_id,
            "mask_bit": mask_bit,
        }

    def _keyword_row(self, row: dict[str, Any]) -> GramKeywordRow:
        return GramKeywordRow(
            entity_id=str(row["entity_id"]),
            field_name=str(row["field_name"]),
            keyword_id=str(row["keyword_id"]),
            object_id=str(row["object_id"]),
            ordinal=int(row["ordinal"]),
            keyword=str(row["keyword"]),
            keyword_norm=str(row["keyword_norm"]),
            normalizer=str(row["normalizer"]),
            full_mask=int(row["full_mask"]),
            part_count=int(row.get("part_count") or int(row["full_mask"]).bit_count()),
            unit_count=int(row.get("keyword_len") or len(str(row["keyword_norm"]))),
        )


class SQLSparseGramAdapter(StrategyAdapter):
    """Maintain and query sparse-GRAM postings inside a SQL backend."""

    identifier = "sql_sparse_gram"
    strategy = SparseGramIndex

    def __init__(self, backend: Any) -> None:
        """Create a sparse-GRAM strategy adapter for a SQL backend.

        Args:
            backend (Any): Owning SQL backend instance.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(backend)
        self._config: GramConfig | None = None
        self._index: _SQLGramIndex | None = None
        self._ensured_index: _SQLGramIndex | None = None
        self._fields: dict[str, dict[str, Any]] = {}
        self._configs: dict[tuple[str, str], GramConfig] = {}
        # fingerprint key -> the fingerprint this adapter found or wrote for the field at its registration
        self._fingerprints: dict[str, str] = {}
        self._deferred: list[tuple[EntitySchema, Any]] | None = None
        # entity id -> indexable fields its registration on this backend left without the index
        self._unbound: dict[str, set[str]] = {}
        self._seen_index: _SQLGramIndex | None = None
        # entity id -> (epoch state, stored fingerprints or None while the index tables are absent) as last read
        self._snapshots: dict[str, tuple[Any, dict[str, str] | None]] = {}
        # (transaction, entity id -> stored fingerprints or None when the index tables are absent) of the last check
        self._checked_in_transaction: tuple[Any, dict[str, dict[str, str] | None]] | None = None

    @property
    def config(self) -> GramConfig:
        """Return runtime sparse-GRAM defaults after Backend bootstrap.

        Returns:
            GramConfig: Defaults loaded lazily from `CM_HVNB`.
        """
        if self._config is None:
            self._config = GramConfig.load(manager=self.backend.config)
        return self._config

    def reset(self) -> None:
        """Clear cached state after the owning backend reconnects.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        self._config = None
        self._index = None
        self._ensured_index = None
        self._fields.clear()
        self._configs.clear()
        self._fingerprints.clear()
        self._unbound.clear()
        self._seen_index = None
        self._snapshots.clear()
        self._checked_in_transaction = None

    @property
    def maintains_index(self) -> bool:
        """Return whether this adapter keeps a keyword index on its backend's engine.

        The index stays consistent with the stored rows only because every
        write of those rows changes both in one transaction. An engine that may
        commit every statement on its own (one that withdraws the backend's
        ``atomic-upsert`` claim at construction) cannot give that. Nor can a
        backend type whose HeavenBase 0.1.2.2 kept no index (``pgvector`` and
        ``starrocks``): 0.1.2.2 writes such a database without touching any
        index and leaves no mark that it did, so an index this release kept
        would miss its rows. On such a backend the adapter indexes no field:
        ``contained_in`` scans the stored rows, and every field is handled as
        one registered without the index.

        Returns:
            bool: ``False`` for a backend type that keeps no keyword index or an
            engine that may commit every statement.
        """
        return getattr(self.backend, "_keeps_keyword_index", True) and not getattr(self.backend, "_autocommit_engine", False)

    def supports(self, schema: EntitySchema, binding: Any) -> bool:
        """Return whether this adapter supports one SQL storage binding.

        Args:
            schema: Entity schema being registered.
            binding: Storage binding routed to the SQL backend.

        Returns:
            bool: True when the binding is an Array[ShortText] SparseGramIndex binding
                owned by this backend, and the backend's engine runs each write in one
                transaction (see ``maintains_index``).
        """
        if binding.backend != self.backend.name or binding.strategy != SparseGramIndex.identifier or binding.field_name == schema.pk_name:
            return False
        return self.maintains_index and self._indexable(schema.get(binding.field_name))

    @staticmethod
    def _indexable(field_schema: Any) -> bool:
        """Return whether a field is an ``Array[ShortText]`` this adapter could index."""
        item_type = getattr(field_schema.dtype, "item_type", None)
        return item_type is not None and type(item_type).identifier == ShortText.identifier

    def prune(self, schema: EntitySchema, bindings: Any, *, protected: frozenset[str] | set[str] = frozenset(), retire_stored: bool = True) -> None:
        """Stop indexing fields that are no longer bound to this adapter.

        A field that was indexed under an earlier registration, in this process
        or a previous one, leaves a fingerprint in the index metadata. Every
        such field that the entity no longer binds is retired: one switched to
        ``gram=False`` or another placement, one whose type is no longer
        ``Array[ShortText]``, one removed from the entity altogether, and every
        field on a backend that keeps no keyword index (see
        ``maintains_index``). Its
        keyword and posting rows are deleted together with that fingerprint, so
        a later registration that binds the field again rebuilds the index from
        stored rows instead of trusting a stale one. The stored index is
        deleted before the field stops being maintained, so a failed deletion
        leaves the field maintained, and the next registration that binds it
        checks its fingerprint again. Shared and eligible legacy index retirement
        use the same transaction. A database that has no
        index tables cannot hold such a fingerprint, so pruning never creates
        them: an entity registered with ``gram=False`` on a fresh backend
        leaves no trace of the index. The fields stay recorded as unbound, so a
        write of the entity through this backend raises once another
        registration indexes them on the same database, also on a backend that
        keeps no keyword index. The stored state read here is kept for the
        write check (see ``write_guards``).

        Args:
            schema: Entity schema of the registration that keeps ``bindings``.
            bindings: Sparse-GRAM bindings the entity keeps on this backend.
            protected: Fields that another alias to the same database still binds.
            retire_stored: Whether to delete unused stored indexes. Reattaching
                a previous registration updates only the adapter's local state.

        Returns:
            None: This function does not return a value.
        """
        kept = {binding.field_name for binding in bindings}
        entity_id = schema.entity_id
        self._checked_in_transaction = None
        candidates = {name for name, field_schema in schema.fields.items() if name != schema.pk_name and self._indexable(field_schema)}
        candidates.update(self._fields.get(entity_id, {}))
        self._snapshots.pop(entity_id, None)
        index = self._need_index() if retire_stored and self.backend.db is not None and self._index_tables_exist() else None
        if index is not None:
            stored = self._read_stored_state([entity_id], registering=True)[entity_id]
            candidates.update(stored or {})
        elif retire_stored and self.backend.db is not None:
            self._snapshots[entity_id] = (_TABLES_ABSENT, None)
        retired = sorted(candidates - kept)
        if not retired:
            self._unbound.pop(entity_id, None)
            return
        # A field already recorded as unbound stays recorded while it stays retired. A field this adapter maintains is
        # recorded once its stored index is gone, so one whose deletion failed is still maintained and never refused as
        # unbound. A field it does not maintain is recorded before the deletion: if that fails, a write here must still be
        # refused while the stored index keeps its fingerprint, since nothing here would keep that index up to date.
        unbound = self._unbound.get(entity_id, set()) & (set(retired) - protected)
        self._unbound[entity_id] = unbound
        for name in retired:
            if name in protected:
                self._retire(entity_id, name, None)
                continue
            if name not in self._fields.get(entity_id, {}):
                unbound.add(name)
            self._retire(entity_id, name, index)
            unbound.add(name)

    def abandon(self, schema: EntitySchema, bindings: Any) -> None:
        """Leave the index and adapter state restored by the registration context.

        Args:
            schema: Entity schema of the registration that stays live, or of the failed one when none does.
            bindings: Sparse-GRAM bindings the entity keeps on this backend under the registration that stays live.

        Returns:
            None: This function does not return a value.
        """
        del schema, bindings

    def _retire(self, entity_id: str, name: str, index: _SQLGramIndex | None) -> None:
        """Stop maintaining one field after deleting its stored index."""
        key = self._fingerprint_key(entity_id, name)
        if index is not None:
            self._ensure_index()
            legacy_key = self._legacy_key(entity_id, name)
            with index._tx():
                if index.get_meta(key) is not None or index.get_meta(legacy_key) is not None:
                    if index.get_meta(key, lock=True) is not None:
                        index.lock_entity_rows(entity_id, self.backend.tables.get(entity_id))
                        # the new epoch does not match the state kept for the write check
                        self._snapshots.pop(entity_id, None)
                    index.retire_field(entity_id, name, key, legacy_key)
                legacy = self._legacy_namespaces(entity_id, name)
                if len(legacy) <= 1:
                    self._retire_workspace_indexes(entity_id, name, self._workspace_indexes_to_retire(legacy, entity_id, name))
        self._fingerprints.pop(key, None)
        known = self._fields.get(entity_id, {})
        known.pop(name, None)
        if not known:
            self._fields.pop(entity_id, None)
        self._configs.pop((entity_id, name), None)

    def _index_tables_exist(self) -> bool:
        """Return whether the index tables are present, without creating them.

        Inside a write transaction the check runs on that transaction's
        connection, so it sees the same database state as the writes it guards.
        Tables once seen are not looked up again on this connection.
        """
        index = self._need_index()
        if self._ensured_index is index or self._seen_index is index:
            return True
        from sqlalchemy import inspect

        db = self.backend.db
        bind = db._conn if db.in_transaction() else db.engine
        if db.dialect == "sqlite" and index.meta.schema is None:
            # one query: SQLAlchemy's lookup reads ``main`` and then ``temp``, and these tables are never temporary
            from sqlalchemy import text

            lookup = text("SELECT name FROM main.sqlite_master WHERE type = 'table' AND name = :name")
            if bind is db.engine:
                with bind.connect() as connection:
                    present = connection.execute(lookup, {"name": index.meta.name}).first() is not None
            else:
                present = bind.execute(lookup, {"name": index.meta.name}).first() is not None
        else:
            present = bool(inspect(bind).has_table(index.meta.name))
        if present:
            self._seen_index = index
        return present

    def ensure(self, schema: EntitySchema, binding: Any) -> None:
        """Ensure sparse-GRAM tables and field-specific strategy config.

        Args:
            schema: Entity schema being registered.
            binding: Sparse-GRAM storage binding.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the owning SQL backend has been dropped and not reconnected.
        """
        if not self.supports(schema, binding):
            return
        self._ensure_index()
        if self._deferred is not None:
            self._deferred.append((schema, binding))
        else:
            index = self._need_index()
            with index._tx() if index.db.dialect == "postgresql" else nullcontext():
                self._ensure_field(schema, binding)
                self._read_stored_state([schema.entity_id], registering=True)

    @contextmanager
    def _defer_ensures(self) -> Iterator[list[tuple[EntitySchema, Any]]]:
        """Defer binding and index changes until storage DDL has finished."""
        pending: list[tuple[EntitySchema, Any]] = []
        self._deferred = pending
        try:
            yield pending
        finally:
            self._deferred = None

    @contextmanager
    def _publish_registration(
        self,
        entity_id: str,
        pending: Sequence[tuple[EntitySchema, Any]],
        aliases: Sequence[Any],
        *,
        publication: _RegistrationPublication,
        reattaching: bool,
    ) -> Iterator[None]:
        """Roll back pending keyword rows, fingerprints, and local bindings if publication fails."""
        with self.backend._lock:
            fields = dict(self._fields.get(entity_id, {}))
            unbound = set(self._unbound.get(entity_id, ()))
            config_keys = [(entity_id, binding.field_name) for _, binding in pending]
            fingerprint_keys = [self._fingerprint_key(*key) for key in config_keys]
            configs = {key: self._configs[key] for key in config_keys if key in self._configs}
            fingerprints = {key: self._fingerprints[key] for key in fingerprint_keys if key in self._fingerprints}
            try:
                index = self._need_index()
                table = self.backend.tables.get(entity_id)
                with ExitStack() as transaction:
                    early = False
                    if not index.db.in_transaction():
                        transaction.enter_context(index._tx())
                        publication.stage(index.db._conn, transaction)
                        # DuckDB reads from the snapshot of the transaction's first read, so a change of the index has
                        # to hold the entity's writes off before it, which the gate does before any statement runs
                        early = index.duckdb_gate(table) is not None
                        if early:
                            index.hold_entity_rows(table)
                    for alias in aliases:
                        if not alias.in_transaction():
                            transaction.enter_context(alias._join_transaction(index.db))
                    changes = index.entity_changes
                    for schema, binding in pending:
                        self._ensure_field(schema, binding, publication=publication, reattaching=reattaching)
                    if early and index.entity_changes == changes:
                        index.release_entity_rows(table)  # no fingerprint changed, so the writes need not wait
                    # Keep the state this publication leaves for the write check. If the publication does not commit,
                    # its epoch never matches and the next write reads the state again.
                    self._read_stored_state([entity_id], registering=True)
                    yield
            except BaseException:
                self._fields.pop(entity_id, None)
                self._unbound.pop(entity_id, None)
                if fields:
                    self._fields[entity_id] = fields
                if unbound:
                    self._unbound[entity_id] = unbound
                for key in config_keys:
                    self._configs.pop(key, None)
                for key in fingerprint_keys:
                    self._fingerprints.pop(key, None)
                self._configs.update(configs)
                self._fingerprints.update(fingerprints)
                raise

    def _ensure_field(self, schema: EntitySchema, binding: Any, *, publication: _RegistrationPublication | None = None, reattaching: bool = False) -> None:
        self._checked_in_transaction = None
        self._snapshots.pop(schema.entity_id, None)
        field_schema = schema.get(binding.field_name)
        self._fields.setdefault(schema.entity_id, {})[field_schema.name] = field_schema
        config = self.config.with_options(**dict(getattr(binding, "strategy_options", {}) or {}))
        self._configs[(schema.entity_id, field_schema.name)] = config
        self._ensure_fingerprint(schema.entity_id, field_schema.name, config, publication=publication, reattaching=reattaching)
        unbound = self._unbound.get(schema.entity_id)
        if unbound is not None:
            unbound.discard(field_schema.name)
            if not unbound:
                self._unbound.pop(schema.entity_id)

    def _namespace(self) -> str:
        # A SQL backend names an entity's tables by entity id alone, so every
        # workspace and backend instance that opens one database reads and
        # writes the same rows, and the index derived from them is shared the
        # same way. Backends that scope table names inside a shared database
        # (Supabase, ClickHouse, shared-database pgvector) install no adapter.
        return self.identifier

    def _workspace_namespace(self) -> str:
        """Return the per-workspace namespace HeavenBase 0.1.2.2 and earlier indexed this backend's rows under."""
        return f"{self.backend.ws_id or 'unbound'}:{self.backend.name}:{self.identifier}"

    def _connect_index(self) -> _SQLGramIndex:
        return _SQLGramIndex(
            self.backend,
            namespace=self._namespace(),
            page_size=self.config.posting_page_size,
        )

    def _need_index(self) -> _SQLGramIndex:
        if self.backend.db is None:
            raise RuntimeError(f"SQL backend {self.backend.name!r} is unavailable after drop")
        if self._index is None or self._index.db is not self.backend.db:
            self._index = self._connect_index()
        return self._index

    def _ensure_index(self) -> None:
        # Ensure fixed index tables once per backend connection. DuckDB and similar single-connection
        # engines reject nested DDL transactions when this runs inside the outer row-upsert transaction.
        index = self._need_index()
        if self._ensured_index is not index:
            index.ensure()
            self._ensured_index = index

    @staticmethod
    def _fingerprint_key(entity_id: str, field_name: str) -> str:
        return f"fingerprint:{entity_id}:{field_name}"

    @staticmethod
    def _legacy_key(entity_id: str, field_name: str) -> str:
        return f"legacy:{entity_id}:{field_name}"

    def _validate_reattachment(self, entity_id: str, binding: Any) -> None:
        """Compare a proposed binding with the stored fingerprint without changing it."""
        field_name = binding.field_name
        existing = self._need_index().get_meta(self._fingerprint_key(entity_id, field_name))
        config = self.config.with_options(**binding.strategy_options)
        self._check_reattachment(entity_id, field_name, existing, config.fingerprint())

    @staticmethod
    def _check_reattachment(entity_id: str, field_name: str, existing: str | None, fingerprint: str) -> None:
        if existing is not None and existing != fingerprint:
            raise SparseGramIndexConflictError(
                f"Cannot reattach the stored registration for {entity_id}.{field_name} with this sparse-GRAM configuration. "
                "Register the previous entity definition with its stored catalog and gram options before changing them."
            )

    def _ensure_fingerprint(
        self, entity_id: str, field_name: str, config: GramConfig, *, publication: _RegistrationPublication | None = None, reattaching: bool = False
    ) -> None:
        # A field with no recorded fingerprint but stored rows was written
        # without this index (an earlier registration that did not bind it, or
        # a field added later), so its index is built from storage before it is
        # trusted; a changed fingerprint rebuilds it the same way. The stored
        # fingerprint, and the per-workspace indexes of HeavenBase 0.1.2.2, are
        # read on every registration: another workspace or process on this
        # database may have rebuilt the shared index with other options, or
        # 0.1.2.2 may have indexed rows since, and registering again rebuilds it.
        key = self._fingerprint_key(entity_id, field_name)
        index = self._need_index()
        fingerprint = config.fingerprint()
        index.lock_registration(entity_id)
        existing = index.get_meta(key)
        if reattaching and existing is not None:
            self._check_reattachment(entity_id, field_name, existing, fingerprint)
            self._fingerprints[key] = fingerprint
            return
        legacy = self._legacy_namespaces(entity_id, field_name)
        meta = {key: fingerprint}
        several_legacy = len(legacy) > 1
        retire = [] if several_legacy else self._workspace_indexes_to_retire(legacy, entity_id, field_name)
        # A legacy rewrite can change canonical rows even if it restores the
        # exact keyword rows already present in one preserved namespace.
        reconcile_legacy = several_legacy or bool(retire)
        if existing == fingerprint and self._fingerprints.get(key) == fingerprint and not reconcile_legacy:
            return
        # A rebuild under the same fingerprint repairs the index for the
        # existing registrations too. Protect changes that depend on publication.
        if publication is not None and (
            (existing is not None and existing != fingerprint) or retire or (several_legacy and _RETIRED_FINGERPRINT in legacy.values())
        ):
            publication.protect(index.db._conn)
        if reconcile_legacy or existing != fingerprint:
            with index._tx():
                if index.db.dialect == "postgresql":
                    index.get_meta(key, lock=True)  # fails for a fingerprint changed after the transaction's snapshot
                # Writes of the entity in progress commit first, and later ones see the new epoch, so the row
                # count and the rebuild below see every row that this index has to cover.
                index.lock_entity_rows(entity_id, self.backend.tables.get(entity_id))
                if reconcile_legacy or existing is not None or self.backend.count(entity_id) > 0:
                    self._rebuild_field_index(entity_id, field_name, meta=meta)
                else:
                    for name, value in meta.items():
                        index.set_meta(name, value)
        self._fingerprints[key] = fingerprint
        if not legacy:
            # Give 0.1.2.2 one namespace from which to rebuild a database only
            # this release has indexed. Check again after taking the write lock
            # so two current workspaces cannot seed different markers.
            with index._tx():
                index.lock_fingerprint(key, fingerprint)
                if index.get_meta(key) == fingerprint and not self._legacy_namespaces(entity_id, field_name):
                    self._legacy_index(self._workspace_namespace()).set_meta(key, _RETIRED_FINGERPRINT)
        elif several_legacy:
            self._keep_workspace_indexes(entity_id, field_name, legacy, fingerprint)
        else:
            # Retire only after the rebuild has written the index and its fingerprint. An interrupted rebuild leaves
            # the per-workspace index in place, so the next registration rebuilds again.
            self._retire_workspace_indexes(entity_id, field_name, retire)

    def _require_fingerprint(self, entity_id: str, field_name: str, index: _SQLGramIndex) -> None:
        """Raise when the shared index of a field this adapter registered is no longer built with its options.

        Every workspace and backend instance on one database shares one index
        per entity field, built with one set of options. Another one that
        registers the field with other options rebuilds it with those, and one
        that registers it without the index deletes it; both replace the stored
        fingerprint. A query or write with this registration's options would
        then silently miss rows, so it raises instead.

        Raises:
            SparseGramIndexConflictError: If the stored fingerprint is not the one this adapter registered.
        """
        key = self._fingerprint_key(entity_id, field_name)
        if key in self._fingerprints:
            self._check_fingerprint(entity_id, field_name, index.get_meta(key))

    def _check_fingerprint(self, entity_id: str, field_name: str, stored: str | None) -> None:
        expected = self._fingerprints.get(self._fingerprint_key(entity_id, field_name))
        if expected is None or stored == expected:
            return
        state = "deleted it" if stored is None else "rebuilt it with other SparseGramIndex options"
        raise SparseGramIndexConflictError(
            f"The sparse-GRAM index of {entity_id}.{field_name} on SQL backend {self.backend.name!r} is shared by every workspace "
            f"and process using this database, and another one {state} after this backend registered the field. Use the same entity "
            "definition, SparseGramIndex options, heavenbase.strategy_defaults.sparse_gram defaults and gram option for the field "
            "everywhere this database is used, on engines that run each write in one transaction (an engine that may commit every "
            "statement keeps no keyword index). Registering the entity again here with allow_replace=True, or releasing this workspace "
            "and opening it again, rebuilds the index with this registration's options."
        )

    def _raise_unmaintained_field(self, entity_id: str, field_name: str) -> None:
        registration = "registered it without the index" if field_name in self._unbound.get(entity_id, ()) else "has no binding for its current index"
        raise SparseGramIndexConflictError(
            f"The sparse-GRAM index of {entity_id}.{field_name} on SQL backend {self.backend.name!r} is shared by every "
            f"workspace and process using this database, and another one indexes the field, while this backend {registration} "
            "(gram=False, a placement without SparseGramIndex, a definition without the field, a backend type that keeps "
            "no keyword index such as pgvector or StarRocks, or an engine that may commit every statement). "
            "A write here would leave that index stale. Use the same definition, SparseGramIndex placement and gram "
            "option for the field everywhere this database is used. Registering the entity again here with allow_replace=True deletes the index if "
            "the field stays unbound, or rebuilds it if the registration binds it."
        )

    def _legacy_namespaces(self, entity_id: str, field_name: str) -> dict[str, str]:
        """Return the per-workspace namespaces that hold a fingerprint for one field, with the value each one stores.

        HeavenBase 0.1.2.2 and earlier indexed a backend's rows under a
        per-workspace namespace (``<workspace id>:<backend name>:sql_sparse_gram``),
        so its index missed every write made through another workspace or
        through this version. Every such namespace on the database counts,
        whichever workspace wrote it: rows that 0.1.2.2 wrote through any
        workspace are rows of the shared tables. The lookup reads the small
        metadata table once.
        """
        key = self._fingerprint_key(entity_id, field_name)
        own = self._namespace()
        suffix = f":{self.identifier}"
        return {
            namespace: value
            for namespace, value in sorted(self._need_index().namespaces_with_meta(key).items())
            if namespace != own and namespace.endswith(suffix)
        }

    def _workspace_indexes_to_retire(self, legacy: dict[str, str], entity_id: str, field_name: str) -> list[str]:
        """Return the one-workspace legacy namespaces whose index 0.1.2.2 wrote since its last retirement.

        A namespace whose fingerprint for the field is not the retirement
        marker, or that holds a keyword row of it, has indexed rows since.

        Returns:
            list[str]: Namespaces to retire, in sorted order.
        """
        return [
            namespace for namespace, value in legacy.items() if value != _RETIRED_FINGERPRINT or self._legacy_index(namespace).has_field(entity_id, field_name)
        ]

    def _retire_workspace_indexes(self, entity_id: str, field_name: str, namespaces: Sequence[str]) -> None:
        """Delete one field's per-workspace index in each namespace and leave the retirement marker as its fingerprint.

        Used only when a single workspace indexed the field under 0.1.2.2. The
        marker matches no configuration, so 0.1.2.2 opening the database
        again rebuilds that workspace's index from every stored row rather
        than trusting the deleted one. This step can go once no supported
        database holds rows that 0.1.2.2 wrote.
        """
        key = self._fingerprint_key(entity_id, field_name)
        for namespace in namespaces:
            legacy = self._legacy_index(namespace)
            with legacy._tx():
                # the marker first, so 0.1.2.2 never trusts a partly deleted index
                legacy.set_meta(key, _RETIRED_FINGERPRINT)
                legacy.delete_field(entity_id, field_name)

    def _keep_workspace_indexes(self, entity_id: str, field_name: str, legacy: dict[str, str], fingerprint: str) -> None:
        """Leave the per-workspace indexes of a field that several 0.1.2.2 workspaces indexed as they are.

        0.1.2.2 computed keyword ids without the namespace, so two of its
        workspaces could share a database only while each indexed rows the
        other did not. Making each of them rebuild its index from every stored
        row would insert the same keyword ids twice, and the second workspace
        would fail to open. Their indexes are therefore kept, and 0.1.2.2 opens
        each workspace with the index it had. A namespace that still carries
        the retirement marker from a time when it was the only one gets this
        registration's fingerprint instead, which 0.1.2.2 accepts without a
        rebuild when its options are the same.
        """
        key = self._fingerprint_key(entity_id, field_name)
        for namespace, value in legacy.items():
            if value == _RETIRED_FINGERPRINT:
                self._legacy_index(namespace).set_meta(key, fingerprint)

    def _legacy_index(self, namespace: str) -> _SQLGramIndex:
        return _SQLGramIndex(self.backend, namespace=namespace, page_size=self.config.posting_page_size)

    def _rebuild_field_index(self, entity_id: str, field_name: str, *, meta: dict[str, str] | None = None) -> None:
        """Rebuild one field index from canonical SQL rows after strategy config changes.

        The caller holds the entity's rows (``_SQLGramIndex.lock_entity_rows``)
        in the same transaction, so no write of the entity commits while the
        stored rows are read. The old rows are deleted first; the new rows and
        ``meta`` (the fingerprint) are written in the same transaction, so a
        reader that checks the fingerprint never trusts rows built with other
        options.
        """
        index = self._need_index()
        field_schema = self._fields.get(entity_id, {}).get(field_name)
        old_duckdb = _duckdb_release_without_row_replacement(index.db)
        if old_duckdb is not None and index.has_field(entity_id, field_name):
            _refuse_duckdb_row_replacement(old_duckdb, f"the sparse-GRAM rows of {entity_id}.{field_name}")
        with index._tx():
            index.delete_field(entity_id, field_name)
            if field_schema is not None:
                writes = []
                for row in self.backend.scan(entity_id):
                    object_id = row.get("object_id")
                    if object_id is None:
                        continue
                    keywords, postings = index_keyword_rows(
                        entity_id,
                        str(object_id),
                        row,
                        {field_name: field_schema},
                        self._config_for,
                        resolver=self.backend.module_resolver,
                    )
                    writes.append(GramRowWrite(entity_id, str(object_id), {}, keywords, postings))
                index.insert_field_rows(entity_id, field_name, writes)
            for name, value in (meta or {}).items():
                index.set_meta(name, value)

    def validate_upsert(
        self, rows: Sequence[RowOp], *, aliases: dict[str, tuple[SQLSparseGramAdapter, ...]] | None = None, confirmed: Iterable[str] = ()
    ) -> None:
        """Refuse row mutations that cannot maintain a shared sparse-GRAM index.

        Every strategy-maintained write is checked, through every SQL route,
        whether or not its backend keeps a keyword index. A write is refused
        when another workspace indexes a field of the entity that none of the
        routes writing the row maintains, or rebuilt or deleted the index of a
        field that one of them maintains. ``write_guards`` lets the row
        statements themselves carry the check; the entities in ``confirmed``
        passed it there and are not checked again. For any other entity this
        sends one query: the index-table lookup while those tables are absent,
        otherwise one read of the stored fingerprints and epochs of every such
        entity in the call. The first query after another workspace creates
        the tables sends both: the lookup that finds them, then the read. Inside
        one database transaction a later call reuses what an earlier one read
        for the same entities. On a route that declares atomic upserts the
        owning backend calls this inside the write transaction after the row
        statements, and a refusal rolls them back; on other routes it runs
        before the row statements, and ``confirmed`` names the entity whose
        first row statement carries the check instead.

        Args:
            rows: Backend row mutations from the owning SQL backend.
            aliases: Other adapters maintaining fields of the same physical row, keyed by entity.
            confirmed: Entities whose write statements already found the stored index state they expected.

        Raises:
            SparseGramIndexConflictError: If the registered fields cannot maintain the current shared index.
        """
        done = set(confirmed)
        entity_ids = sorted({row.entity_id for row in rows} - done)
        if not entity_ids:
            return
        stored = self._stored_fingerprints(entity_ids)
        upserts = {row.entity_id for row in rows if not row.is_delete()}
        for entity_id in entity_ids:
            self._check_stored(entity_id, stored[entity_id], aliases, upsert=entity_id in upserts)

    def _check_stored(
        self, entity_id: str, current: dict[str, str] | None, aliases: dict[str, tuple[SQLSparseGramAdapter, ...]] | None, *, upsert: bool
    ) -> None:
        """Raise if a write of one entity through these adapters would leave the stored index ``current`` stale."""
        if current is None:
            return
        adapters = (self, *(aliases or {}).get(entity_id, ()))
        maintained = {field for adapter in adapters for field in adapter._fields.get(entity_id, {})}
        unmaintained = current.keys() - maintained
        if unmaintained:
            self._raise_unmaintained_field(entity_id, min(unmaintained))
        if upsert:
            for adapter in adapters:
                for field_name in sorted(adapter._fields.get(entity_id, {})):
                    adapter._check_fingerprint(entity_id, field_name, current.get(field_name))

    @property
    def guards_writes(self) -> bool:
        """Return whether the row statements of this backend's dialect can carry the write check (see ``write_guards``)."""
        db = self.backend.db
        # DuckDB reports a statement's row count as its one result row instead (see ``SQLBackend._affected_rows``)
        return db is not None and self._need_index().guarded and (bool(db.engine.dialect.supports_sane_rowcount) or db.dialect == "duckdb")

    def write_guards(self, rows: Sequence[RowOp], *, aliases: dict[str, tuple[SQLSparseGramAdapter, ...]] | None = None) -> dict[str, _WriteGuard]:
        """Return, per entity, the stored index state its write can expect without a query of its own.

        The adapter keeps the state it last read for each entity: whether the
        index tables exist, the entity's stored fingerprints and its epoch,
        which every change of those fingerprints replaces in the same
        transaction (see ``_SQLGramIndex.lock_entity_rows``). If the write
        passes the check against that state, the first row statement of the
        entity carries a condition that holds only while the state is
        unchanged: the guarded ``INSERT`` adds no row and the guarded
        ``DELETE`` removes none otherwise. The backend then calls
        ``confirm_write``, which reads the state again and checks the write
        against it. A write the kept state would refuse, and an entity with no
        kept state, get no guard and are checked by ``validate_upsert``. On a
        route whose statements may commit on their own, the backend lets only
        the call's first row statement carry its guard and confirms a miss at
        once, before any other row statement.

        Args:
            rows: Strategy-maintained row mutations of one write call.
            aliases: Other adapters maintaining fields of the same physical row, keyed by entity.

        Returns:
            dict[str, _WriteGuard]: Guards of the entities whose kept state passes the check.
        """
        upserts = {row.entity_id for row in rows if not row.is_delete()}
        guards = {}
        for entity_id in {row.entity_id for row in rows}:
            kept = self._snapshots.get(entity_id)
            if kept is None:
                continue
            state, current = kept
            try:
                self._check_stored(entity_id, current, aliases, upsert=entity_id in upserts)
            except SparseGramIndexConflictError:
                continue
            guards[entity_id] = self._guard(entity_id, state)
        return guards

    def _guard(self, entity_id: str, state: Any) -> _WriteGuard:
        index = self._need_index()
        if state is _TABLES_ABSENT:
            return _WriteGuard(state, "tables", {"hbg_table": index.catalog_name(index.meta)})
        params = {"hbg_ns": index.namespace, "hbg_key": index.epoch_key(entity_id)}
        if state is None:
            return _WriteGuard(state, "no-epoch", params)
        return _WriteGuard(state, "epoch", {**params, "hbg_token": state})

    def confirm_write(
        self, entity_id: str, rows: Sequence[RowOp], *, aliases: dict[str, tuple[SQLSparseGramAdapter, ...]] | None = None, expected: Any = None
    ) -> Any:
        """Read one entity's stored index state again after its guarded statement did not match, and check the write against it.

        Args:
            expected: The state the guard expected. When it was ``_TABLES_ABSENT``
                and PostgreSQL reports index tables that the transaction's
                snapshot does not hold, their rows cannot be read in it.

        Returns:
            Any: The state now stored, which the caller compares with its guard's.

        Raises:
            SparseGramIndexConflictError: If the registered fields cannot maintain the current shared index,
                or if the index tables appeared after the transaction's snapshot.
        """
        if self._checked_in_transaction is not None:
            self._checked_in_transaction[1].pop(entity_id, None)
        if expected is _TABLES_ABSENT and self._need_index().tables_beyond_snapshot():
            raise SparseGramIndexConflictError(
                f"The keyword index tables appeared after this transaction's snapshot was taken, so a write of {entity_id!r} cannot "
                "see their rows; roll the transaction back and write again."
            )
        current = self._stored_fingerprints([entity_id])[entity_id]
        self._check_stored(entity_id, current, aliases, upsert=any(row.entity_id == entity_id and not row.is_delete() for row in rows))
        return self._snapshots[entity_id][0]

    def _read_stored_state(self, entity_ids: Sequence[str], *, lock: bool = False, registering: bool = False) -> dict[str, dict[str, str] | None]:
        """Read the stored fingerprints and epochs of the entities in one query, and keep them for ``write_guards``.

        A registration (``registering``) on PostgreSQL first gives each entity
        that has none an epoch row (see ``_SQLGramIndex.create_epoch``).

        Returns:
            dict[str, dict[str, str] | None]: Stored fingerprints by field name, or ``None`` while the index tables are absent.
        """
        if not self._index_tables_exist():
            index = self._need_index()
            if index.db.dialect == "mysql" and index.db.in_transaction():
                index.refuse_tables_after_snapshot("this check")
            for entity_id in entity_ids:
                self._snapshots[entity_id] = (_TABLES_ABSENT, None)
            return dict.fromkeys(entity_ids)
        index = self._need_index()
        prefixes = [self._fingerprint_key(entity_id, "") for entity_id in entity_ids]
        epochs = [index.epoch_key(entity_id) for entity_id in entity_ids]
        entries = index.meta_entries_any(prefixes, epochs, lock=lock)
        if registering and index.db.dialect == "postgresql":
            index.may_lock_meta()  # known before the first write, so a write sends no query for it
            if any(epoch not in entries for epoch in epochs):
                for entity_id, epoch in zip(entity_ids, epochs):
                    if epoch not in entries:
                        index.create_epoch(entity_id)
                entries = index.meta_entries_any(prefixes, epochs, lock=lock)
        result: dict[str, dict[str, str] | None] = {}
        for entity_id, prefix, epoch in zip(entity_ids, prefixes, epochs):
            # field names are Python names, so a remainder with a colon belongs to another entity's key
            fingerprints = {
                key[len(prefix) :]: value for key, value in entries.items() if key.startswith(prefix) and key[len(prefix) :] and ":" not in key[len(prefix) :]
            }
            result[entity_id] = fingerprints
            self._snapshots[entity_id] = (entries.get(epoch), fingerprints)
        return result

    def _stored_fingerprints(self, entity_ids: Sequence[str]) -> dict[str, dict[str, str] | None]:
        """Return each entity's stored fingerprints (``None`` while the index tables are absent) in at most one query.

        Inside one database transaction the answers are kept and reused by later write calls.
        """
        db = self.backend.db
        transaction = db._conn.get_transaction() if db.in_transaction() else None
        cached: dict[str, dict[str, str] | None] = {}
        if transaction is not None and self._checked_in_transaction is not None and self._checked_in_transaction[0] is transaction:
            cached = self._checked_in_transaction[1]
        missing = [entity_id for entity_id in entity_ids if entity_id not in cached]
        if missing:
            # inside a write transaction the ``mysql`` and PostgreSQL types read the latest committed rows with a locking read
            cached.update(self._read_stored_state(missing, lock=transaction is not None))
        self._checked_in_transaction = (transaction, cached) if transaction is not None else None
        return {entity_id: cached[entity_id] for entity_id in entity_ids}

    def upsert(self, rows: Sequence[RowOp], *, aliases: dict[str, tuple[SQLSparseGramAdapter, ...]] | None = None) -> None:
        """Apply validated SQL row mutations to sparse-GRAM index rows.

        Args:
            rows: Backend row mutations from the owning SQL backend.
            aliases: Other adapters maintaining fields of the same physical row, keyed by entity.

        Raises:
            RuntimeError: If the owning SQL backend has been dropped and not reconnected.
        """
        bindings = {}
        for entity_id in {row.entity_id for row in rows}:
            adapters = (self, *(aliases or {}).get(entity_id, ()))
            owners = {field: adapter for adapter in adapters for field in adapter._fields.get(entity_id, {})}
            if owners:
                bindings[entity_id] = (
                    {field: owner._fields[entity_id][field] for field, owner in owners.items()},
                    {field: owner._config_for(entity_id, field) for field, owner in owners.items()},
                )
        rows = [row for row in rows if row.entity_id in bindings]
        if not rows:
            return
        self._ensure_index()
        writes_by_key: dict[tuple[str, str], GramRowWrite] = {}
        delete_keys: set[tuple[str, str]] = set()
        for row in rows:
            key = (row.entity_id, str(row.object_id))
            if row.is_delete():
                writes_by_key.pop(key, None)
                delete_keys.add(key)
                continue
            delete_keys.discard(key)
            fields, configs = bindings[row.entity_id]
            keywords, postings = index_keyword_rows(
                row.entity_id,
                str(row.object_id),
                row.values,
                fields,
                lambda _entity_id, field: configs[field],
                resolver=self.backend.module_resolver,
            )
            writes_by_key[key] = GramRowWrite(row.entity_id, str(row.object_id), {}, keywords, postings)

        writes = list(writes_by_key.values())
        if writes:
            self._need_index().upsert_rows(writes)

        deletes_by_entity: dict[str, list[str]] = {}
        for entity_id, object_id in delete_keys:
            deletes_by_entity.setdefault(entity_id, []).append(object_id)
        for entity_id, object_ids in deletes_by_entity.items():
            self._need_index().delete_objects(entity_id, object_ids)

    def sweep_unmaintained(self, rows: Sequence[RowOp], *, aliases: dict[str, tuple[SQLSparseGramAdapter, ...]] | None = None) -> None:
        """Remove keyword rows left for rows that were written without keyword maintenance.

        A ``RowOp`` with ``strategies=False`` (an identity Catalog row that the
        merge lookup did not find, or one written in fresh mode) holds no value
        any index derives keywords from, so after its row statements the object
        must have no keyword rows. Another writer that created the same row with
        keywords after the lookup may have committed some; this deletes them in
        the same transaction, after the row statements, so whichever write
        commits its row last also leaves the keyword rows that match it.

        Args:
            rows: Backend row mutations of one write call; only unmaintained upserts are swept.
            aliases: Other adapters maintaining fields of the same physical row, keyed by entity.
        """
        by_entity: dict[str, list[str]] = {}
        for row in rows:
            if row.strategies or not row.is_upsert():
                continue
            adapters = (self, *(aliases or {}).get(row.entity_id, ()))
            if any(adapter._fields.get(row.entity_id) for adapter in adapters):
                by_entity.setdefault(row.entity_id, []).append(str(row.object_id))
        if not by_entity or not self._index_tables_exist():
            return
        index = self._need_index()
        for entity_id, object_ids in by_entity.items():
            index.sweep_objects(entity_id, object_ids)

    def stale_keyword_objects(self, entity_id: str, object_ids: Iterable[Any]) -> list[str]:
        """Return the objects whose keyword or posting rows differ from what their stored rows index.

        Each object's row is read from this backend and indexed again with the
        fields this adapter indexes and their registered options; the keyword
        and posting rows stored for those fields must be exactly the result. An
        object without a stored row must have none. Rows of fields this adapter
        does not index (another workspace's, or another route's), and of fields
        whose shared index no longer has this registration's options (a lookup
        then raises), are not read. An object that differs is read once more, so
        a write between the two reads of the first pass is not reported.

        Args:
            entity_id: Entity identifier whose index rows are checked.
            object_ids: Object identifiers to check.

        Returns:
            list[str]: Object IDs whose index rows differ, in input order.
        """
        fields = self._fields.get(entity_id)
        ids = list(dict.fromkeys(str(object_id) for object_id in object_ids))
        if not fields or not ids or not self._index_tables_exist():
            return []
        index = self._need_index()
        fields = {
            name: field
            for name, field in fields.items()
            if index.get_meta(self._fingerprint_key(entity_id, name)) == self._fingerprints.get(self._fingerprint_key(entity_id, name))
        }
        if not fields:
            return []
        stale = self._objects_with_other_rows(entity_id, fields, ids)
        return self._objects_with_other_rows(entity_id, fields, stale) if stale else []

    def _objects_with_other_rows(self, entity_id: str, fields: dict[str, Any], ids: list[str]) -> list[str]:
        """Return the objects whose stored index rows of ``fields`` differ from what their stored rows index, in one pass."""
        index = self._need_index()
        rows = self.backend.get_many(entity_id, ids)
        stored = index.object_index_ids(entity_id, list(fields), ids)
        stale = []
        for object_id, row in zip(ids, rows):
            expected: tuple[set[str], set[str]] = (set(), set())
            if row is not None:
                keywords, postings = index_keyword_rows(
                    entity_id,
                    object_id,
                    row,
                    fields,
                    lambda _entity_id, field: self._config_for(entity_id, field),
                    resolver=self.backend.module_resolver,
                )
                expected = (
                    {index._stored_keyword_id(keyword.keyword_id) for keyword in keywords},
                    {index._posting_record(posting)["_id"] for posting in postings},
                )
            if stored.get(object_id, (set(), set())) != expected:
                stale.append(object_id)
        return stale

    def can_execute(self, fragment: QueryFragment) -> bool:
        """Return whether this adapter owns one compiled query fragment.

        Args:
            fragment: Compiled query fragment.

        Returns:
            bool: True when this adapter can execute the fragment without SQLBackend
                fallback.
        """
        if not self.maintains_index:
            return False
        if fragment.kind() == "sql_sparse_gram_contained_in":
            return True
        if fragment.op in ("and", "or"):
            return bool(fragment.children()) and all(self.can_execute(child) for child in fragment.children())
        if fragment.op == "not":
            return bool(fragment.children()) and self.can_execute(fragment.children()[0])
        return False

    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute sparse-GRAM fragments and sparse-GRAM-only logical trees.

        The index lookup and the row read are separate statements, so a write
        through another backend instance on the same database (another
        workspace, Context or process) can land between them. The candidate
        ids of the whole fragment tree are found in the index first, and each
        candidate row is then checked against the tree as it was read: the
        row a ``not`` read to find the ids it subtracts from, or one
        ``get_many`` read for the rest. A row that no longer satisfies the
        tree, or no longer exists, is left out, and every entry in a returned
        row's ``match`` metadata names a keyword that row holds.

        Args:
            fragment: Compiled sparse-GRAM query fragment.

        Returns:
            ResultFrame: Rows matched by sparse-GRAM candidate lookup plus exact verification.

        Raises:
            TypeError: If the fragment is not owned by this adapter.
            RuntimeError: If the owning SQL backend has been dropped and not reconnected.
        """
        if fragment.kind() != "sql_sparse_gram_contained_in" and not (fragment.op in LOGICAL_OPS and self.can_execute(fragment)):
            raise TypeError(f"{self.identifier} cannot execute fragment kind={fragment.kind()!r} op={fragment.op!r}")
        found: dict[int, tuple[dict[str, list[dict[str, Any]]], set[int]]] = {}
        scanned: dict[str, dict[str, Any]] = {}

        def universe(entity_id: str | None) -> ResultFrame:
            # every stored id, the universe a ``not`` subtracts from; the rows are kept for the check below
            if not entity_id:
                return ResultFrame.empty()
            ids = []
            for row in self.backend.scan(entity_id):
                object_id = row.get("object_id")
                scanned[str(object_id)] = row
                ids.append({"object_id": object_id})
            return ResultFrame.from_rows(ids)

        def candidates_of(node: QueryFragment) -> ResultFrame:
            if node.kind() != "sql_sparse_gram_contained_in":
                return node.execute(candidates_of, universe)
            found[id(node)] = self._contained_in_matches(node)
            matches_by_id = found[id(node)][0]
            # ids only: the rows and their match metadata are read and checked below
            return ResultFrame.from_rows([{"object_id": object_id} for object_id in sorted(matches_by_id)], metadata_columns={"match"})

        candidates = candidates_of(fragment)
        entity_id = fragment.entity_id()
        if not entity_id or len(candidates) == 0:
            return candidates
        verify = self._verifier(fragment, found)
        unread = [object_id for object_id in candidates.ids if str(object_id) not in scanned]
        read = {str(object_id): row for object_id, row in zip(unread, self.backend.get_many(entity_id, unread) if unread else [])}
        out = []
        for object_id in candidates.ids:
            row = scanned.get(str(object_id)) or read.get(str(object_id))
            if row is None:
                continue
            ok, matches = verify(str(object_id), row)
            if not ok:
                continue
            payload = dict(row)
            if matches:
                payload["match"] = matches
            out.append(payload)
        return ResultFrame.from_rows(out, metadata_columns=candidates.metadata_columns)

    def _contained_in_matches(self, fragment: QueryFragment) -> tuple[dict[str, list[dict[str, Any]]], set[int]]:
        """Return the index's exact-verified match metadata for one ``contained_in`` leaf, keyed by object id, and the short widths it looked up."""
        entity_id = fragment.entity_id()
        if not entity_id:
            return {}, set()
        field_name = fragment.field
        index = self._need_index()
        self._require_fingerprint(entity_id, field_name, index)
        short_widths = self._indexed_short_widths(entity_id, field_name, index)
        matches = contained_in_matches_by_id(
            entity_id,
            field_name,
            fragment.payload.get("query", ""),
            op=fragment.op,
            config=self._config_for(entity_id, field_name),
            short_widths=short_widths,
            candidates_for=lambda hashes: index.lookup_candidates(entity_id, field_name, hashes),
            resolver=self.backend.module_resolver,
        )
        # Checked again after the lookup: a rebuild with other options that committed while it ran replaced the fingerprint
        # in the same transaction as the rows it read.
        self._require_fingerprint(entity_id, field_name, index)
        return matches, short_widths

    def _verifier(
        self,
        fragment: QueryFragment,
        found: dict[int, tuple[dict[str, list[dict[str, Any]]], set[int]]],
        *,
        negated: bool = False,
    ) -> Callable[[str, dict[str, Any]], tuple[bool, list[dict[str, Any]]]]:
        """Return a check of one stored row against a fragment tree, using the keywords the row stores.

        The check returns whether the row satisfies the fragment and the match
        metadata it carries: the matches of every ``contained_in`` leaf it
        satisfies, in fragment order, as composing the leaves' frames gives;
        a negated leaf contributes none, as the ``not`` above it drops them.
        Every leaf decides the keywords the row holds with the test the index
        route applies (:class:`StoredKeywordMatcher`), and a leaf outside any
        ``not`` builds its match metadata from them the way the lookup does.
        On a row the index holds this agrees with the index, and a row whose
        keywords a write changed after the lookup is judged by the keywords
        it holds when read, whatever their order or position. A row that came
        to match only after the lookups found the candidate ids is missed, as
        by any read that is not a snapshot.
        """
        if fragment.kind() != "sql_sparse_gram_contained_in":
            op = fragment.op
            children = [self._verifier(child, found, negated=negated != (op == "not")) for child in fragment.children()]

            def check_tree(object_id: str, row: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
                results = [child(object_id, row) for child in children]
                if op == "not":
                    return not results[0][0], []
                ok = all(hit for hit, _ in results) if op == "and" else any(hit for hit, _ in results)
                return ok, [match for hit, matches in results if hit for match in matches] if ok else []

            return check_tree

        field_name = fragment.field
        _, short_widths = found.get(id(fragment), ({}, set()))
        matcher = StoredKeywordMatcher(
            field_name,
            fragment.payload.get("query", ""),
            self._config_for(str(fragment.entity_id()), field_name),
            op=fragment.op,
            short_widths=short_widths,
            resolver=self.backend.module_resolver,
        )
        if negated:

            def check_row(object_id: str, row: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
                return matcher.holds(row.get(field_name)), []

            return check_row

        def check_candidate(object_id: str, row: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
            matches = matcher.matches(row.get(field_name))
            return bool(matches), matches

        return check_candidate

    def contained_in_probe(self, entity_id: str, field_name: str, query: Any) -> dict[str, Any]:
        """Return exact-recall lookup metrics for one SQL-hosted sparse-GRAM field.

        Args:
            entity_id: Logical entity identifier.
            field_name: Sparse-GRAM field name.
            query: Query text to probe.

        Returns:
            dict[str, Any]: Candidate lookup and analyzer metrics.
        """
        config = self._config_for(entity_id, field_name)
        index = self._need_index()
        self._require_fingerprint(entity_id, field_name, index)
        return contained_in_probe_data(
            entity_id,
            field_name,
            query,
            config=config,
            short_widths=self._indexed_short_widths(entity_id, field_name, index),
            candidates_for=lambda hashes: index.lookup_candidates(entity_id, field_name, hashes),
            posting_hits=lambda hashes: index.posting_hits(entity_id, field_name, hashes),
            resolver=self.backend.module_resolver,
        )

    def posting_count(self, entity_id: str | None = None, field_name: str | None = None) -> int:
        """Return adapter posting row count.

        Args:
            entity_id: Optional entity filter.
            field_name: Optional field filter.

        Returns:
            int: Number of sparse-GRAM posting rows.
        """
        return self._need_index().posting_count(entity_id, field_name)

    def posting_stats(self, entity_id: str | None = None, field_name: str | None = None, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return hot posting hashes for adapter diagnostics.

        Args:
            entity_id: Optional entity filter.
            field_name: Optional field filter.
            limit: Maximum number of posting hash rows to return.

        Returns:
            list[dict[str, Any]]: Posting hash counts ordered by descending frequency.
        """
        return self._need_index().posting_stats(entity_id, field_name, limit=limit)

    def to_dict(self) -> dict[str, Any]:
        """Return the SQL sparse-GRAM ``strategy-adapters`` Tag envelope.

        Args:
            None.

        Returns:
            dict[str, Any]: Stable identity fields plus adapter-specific metadata.
        """
        return {
            "identifier": self.identifier,
            "strategy": self.strategy,
            "metadata": {
                "operations": ["contained_in"],
                "provider": self.backend.provider,
            },
        }

    def contained_in_scan_matcher(
        self, entity_id: str | None, field_name: str, query: Any, *, strategy_options: dict[str, Any] | None = None
    ) -> Callable[[Any], list[dict[str, Any]]]:
        """Compile a row matcher with the field's registered sparse-GRAM options.

        Args:
            entity_id: Entity identifier bound at compilation, if available.
            field_name: Logical keyword field name.
            query: Text searched for the stored keywords.
            strategy_options: Field binding options when the engine keeps no index.

        Returns:
            A row value matcher that returns keyword match metadata.
        """
        if self.maintains_index:
            config = self._config_for(entity_id, field_name) if entity_id is not None else self.config
        else:
            config = self.config.with_options(**(strategy_options or {}))
        # A peer can rebuild or retire its shared index while the canonical rows stay readable.
        return scan_contained_in_matcher(field_name, query, config, short_widths=None, resolver=self.backend.module_resolver)

    def _config_for(self, entity_id: str, field_name: str) -> GramConfig:
        return self._configs.get((entity_id, field_name), self.config)

    def _indexed_short_widths(self, entity_id: str, field_name: str, index: _SQLGramIndex) -> set[int]:
        return index.short_widths(entity_id, field_name, self._config_for(entity_id, field_name).gram_width)
