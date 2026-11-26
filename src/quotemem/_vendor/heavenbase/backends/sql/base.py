"""Shared SQLAlchemy backend over :class:`heavenbase.database.Database`."""

from __future__ import annotations

__all__ = [
    "SQLBackend",
]

import re
import threading
import time
import weakref
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from itertools import groupby
from os.path import samefile
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.mysql import VARCHAR as MySQLVarchar

from heavenbase.utils import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence
from heavenbase.utils import dataclass
from heavenbase.utils.files import file_uri_to_path
from heavenbase.utils.hash import md5hash
from heavenbase.utils.serialize import loads_json

from ...execution.frame.result import ResultFrame
from ...database.config import DB_ENGINE, DBEngine
from ...modules import ModuleResolver
from ...strategies import SideTable, SparseGramIndex
from ...schema.model import EntitySchema
from ...types import LongText
from ...types.storage import stores_as_integer
from ...database.dialect.spec import DatabaseDialectSpec
from ...utils.ids import sql_index_name
from ..base import Backend, _RegistrationPublication, backend_locked, native_layer
from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp
from .capabilities import register_sql_dialect, sql_dialect_spec
from .dialect import SQLDialectSpec
from ._utils import default_database

_MYSQL_DIALECTS = frozenset({"mysql", "mariadb"})
# MySQL error ER_SP_DOES_NOT_EXIST: the server does not provide the function (such as GET_LOCK).
_MYSQL_UNKNOWN_FUNCTION = 1305
# Dialects whose database is one local file that several engines can open.
_FILE_DIALECTS = frozenset({"sqlite", "duckdb"})
# DuckDB before 1.2 keeps a deleted key in its indexes until the deleting transaction commits, so in that transaction
# it rejects inserting the key again and updating an indexed column of the row. Every row replacement does one of them.
_DUCKDB_ROW_REPLACEMENT_VERSION = (1, 2)


def _duckdb_release_without_row_replacement(db: Any) -> str | None:
    """Return the installed duckdb release when ``db`` is DuckDB and the release cannot replace a stored row.

    A release string that does not start with ``<major>.<minor>`` is taken to replace rows.
    """
    if db.dialect != "duckdb":
        return None
    import duckdb

    match = re.match(r"(\d+)\.(\d+)", duckdb.__version__)
    if match is None or (int(match[1]), int(match[2])) >= _DUCKDB_ROW_REPLACEMENT_VERSION:
        return None
    return duckdb.__version__


def _refuse_duckdb_row_replacement(release: str, what: str) -> None:
    """Raise the error for a row replacement that a DuckDB release before 1.2 cannot make."""
    raise RuntimeError(
        f"Cannot replace {what}: duckdb {release} keeps a deleted key in its indexes until the transaction commits, so it "
        "rejects inserting the key again in that transaction. Replacing stored rows on DuckDB needs duckdb 1.2 or later; "
        "connecting, inserting new rows and reading work on older releases"
    )


class _MySQLObjectId(sa.TypeDecorator):
    """Store Unicode IDs in a case-sensitive, preferably no-pad VARCHAR."""

    impl = sa.String
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name not in _MYSQL_DIALECTS:
            return self.impl
        version = dialect.server_version_info or ()
        collation = "utf8mb4_bin"
        if dialect.is_mariadb:
            if version >= (10, 2, 2):
                collation = "utf8mb4_nopad_bin"
        elif version >= (8, 0, 17):
            collation = "utf8mb4_0900_bin"
        elif version >= (8, 0, 11):
            collation = "utf8mb4_0900_as_cs"
        return dialect.type_descriptor(MySQLVarchar(self.impl.length, charset="utf8mb4", collation=collation))


def _mysql_id_key(value: Any) -> Any:
    from sqlalchemy import cast, func, literal_column
    from sqlalchemy.dialects.mysql import BINARY

    # Normalize legacy columns and connection parameters to the same encoding.
    return cast(func.convert(value.op("USING")(literal_column("utf8mb4"))), BINARY)


class _MySQLIdList(sa.TypeDecorator):
    """Apply the binary comparison to each value of an expanding ID bind."""

    impl = sa.String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str:
        return str(value)

    def bind_expression(self, bindvalue: Any) -> Any:
        return _mysql_id_key(bindvalue)


def _mysql_error_code(exc: Any) -> Any:
    args = getattr(getattr(exc, "orig", None), "args", ())
    return args[0] if args else None


def _object_id_matches(column: Any, value: Any, mysql_ids: bool) -> Any:
    matches = column == value
    if mysql_ids:
        # Keep the ordinary predicate available to indexes and check exact
        # identity even on columns created with an older default collation.
        matches = matches & (_mysql_id_key(column) == _mysql_id_key(value))
    return matches


def _object_ids_match(column: Any, mysql_ids: bool, object_ids: Sequence[Any] | None = None) -> Any:
    from sqlalchemy import bindparam

    ids = bindparam("object_ids", object_ids, expanding=True, required=object_ids is None)
    matches = column.in_(ids)
    if mysql_ids:
        exact_ids = bindparam("exact_ids", object_ids, expanding=True, required=object_ids is None, type_=_MySQLIdList())
        matches = matches & _mysql_id_key(column).in_(exact_ids)
    return matches


def _sa_col(dtype: Any, dialect_spec: DatabaseDialectSpec | None = None, *, max_chars: int = 65535):
    from heavenbase.database.types import (
        DatabaseBigIntegerType,
        DatabaseBlobType,
        DatabaseBooleanType,
        DatabaseIntegerType,
        DatabaseJsonType,
        DatabaseTextType,
        DatabaseVectorType,
    )
    from sqlalchemy import Float

    identifier = type(dtype).identifier
    if identifier == "categorical":
        return DatabaseIntegerType() if dtype.base == "integer" else DatabaseTextType(length=dtype.max_chars, dialect_spec=dialect_spec, max_chars=max_chars)
    if identifier in ("timestamp", "duration"):
        return DatabaseBigIntegerType()
    if identifier == "date":
        return DatabaseTextType(length=dtype.max_chars, dialect_spec=dialect_spec, max_chars=max_chars)
    if identifier in ("identifier", "short-text", "medium-text", "long-text"):
        return DatabaseTextType(length=dtype.max_chars, dialect_spec=dialect_spec, max_chars=max_chars)
    if identifier == "integer":
        return DatabaseIntegerType()
    if identifier == "float":
        return Float()
    if identifier == "boolean":
        return DatabaseBooleanType()
    if identifier == "artifact":
        return DatabaseBlobType(dialect_spec=dialect_spec, max_chars=max_chars)
    if identifier == "vector":
        return DatabaseVectorType(dialect_spec=dialect_spec, max_chars=max_chars)
    return DatabaseJsonType(dialect_spec=dialect_spec, max_chars=max_chars)


def _table_name(entity_id: str) -> str:
    return Backend._physical_name(entity_id)


def _side_table_name(entity_id: str, field_name: str) -> str:
    return f"{_table_name(entity_id)}_{field_name}"


def _side_present_column(field_name: str) -> str:
    """Return the parent-table column recording whether a side field was set.

    A side table holds one row per item, so an absent value and an empty list
    both store nothing and read back the same. This column keeps the difference.
    The leading underscore cannot collide with a declared field: `check_py_name`
    rejects those names.

    Args:
        field_name (str): Side-routed field name.

    Returns:
        str: Physical marker column name.
    """
    return f"_present_{field_name}"


def _norm_cols(cols: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(getattr(col, "name", col)).lower() for col in cols)


def _duckdb_covered_columns(db: Any, table: Any) -> set[str] | None:
    """Return the lower-cased columns a new index of ``table`` is limited to, or ``None`` where any column may be indexed.

    DuckDB 1.5.3 cannot replay a write-ahead log that writes rows of a table,
    adds an index to it with a column that none of its indexes covers, and
    writes rows again: the next process that opens the file fails on its first
    write of the table. A table of a DuckDB file that DuckDB counts rows for,
    deleted ones included until a checkpoint, can have such a log, so only the
    columns its primary key and its indexes cover are left.
    """
    if getattr(db, "dialect", None) != "duckdb" or db.spec.in_memory:
        return None
    stmt = sa.text(
        "SELECT estimated_size FROM duckdb_tables() WHERE table_name = :name "
        "AND schema_name = coalesce(:schema, current_schema()) AND database_name = current_database()"
    )
    rows = db.orm_execute(stmt, {"name": table.name, "schema": table.schema}, readonly=True).to_lists()
    if not rows or int(rows[0][0] or 0) == 0:
        return None
    return {column for columns in db.index_column_sets(table) for column in columns}


def _index_already_exists_error(exc: Exception) -> bool:
    """Return whether a provider reported the desired existing-index state."""
    message = str(exc).lower()
    original = getattr(exc, "orig", None)
    args = getattr(original, "args", ())
    return "already exists" in message or "duplicate key name" in message or (bool(args) and args[0] == 1061)


#: MySQL-family storage engines that keep no transactions, for a server whose ``information_schema.ENGINES``
#: does not say (an engine it lists with ``TRANSACTIONS`` ``NO`` counts as well).
_NONTRANSACTIONAL_MYSQL_ENGINES = frozenset(
    {"myisam", "mrg_myisam", "merge", "memory", "heap", "csv", "archive", "blackhole", "federated", "aria", "connect", "s3"}
)


#: Dialects on which the generic ``sql`` type needs no check for IDs the server only matches under a collation: SQLite's
#: BINARY and DuckDB's default collation compare bytes, PostgreSQL's deterministic collations break ties by bytes, Oracle
#: compares with NLS_COMP BINARY by default, and ClickHouse compares bytes.
_EXACT_ID_DIALECTS = frozenset({"sqlite", "duckdb", "postgresql", "oracle", "clickhousedb"})

#: Backend types whose routes on separate engines recognize a shared table through a table made for the check.
_PROBED_TYPES = frozenset({"sql"})


_HYPERG_INDEX_TYPES = {"identifier", "short-text", "integer", "float", "categorical", "timestamp", "duration", "date"}


#: Fragment kinds the SQL backend compiles to one WHERE clause.
_SQL_FRAGMENT_KINDS = ("sql_where", "sql_array_contains", "sql_array_exact", "sql_hyperg_exists", "sql_hyperg_all")

#: A connection-initialization statement naming either word can switch the
#: session's transaction mode after the driver set it (``SET autocommit=1`` on
#: MySQL, ``SET IMPLICIT_TRANSACTIONS OFF`` under SQL Server ODBC manual commit).
_TRANSACTION_MODE_STATEMENT = re.compile(r"\b(?:autocommit|implicit_transactions)\b", re.IGNORECASE)


def _dbapi_connect_params(url: Any, connect_args: Dict[str, Any]) -> Dict[str, Any]:
    """Return the keyword arguments ``create_engine`` passes to the DBAPI ``connect()``.

    The engine's dialect turns the URL, whose query carries the backend's
    ``params``, into connect arguments, and ``connect_args`` then overrides
    them, as ``create_engine`` does. The dialect is built without its driver,
    so no connection opens; a dialect that cannot be built that way, or that
    needs its driver to read the URL (``oracledb``), yields the raw URL query
    instead, which keeps every ``autocommit`` setting the URL carries.

    Args:
        url: SQLAlchemy URL of the engine.
        connect_args: ``connect_args`` given to ``create_engine``.

    Returns:
        Dict[str, Any]: DBAPI connect keyword arguments.
    """
    try:
        params = dict(url.get_dialect()().create_connect_args(url)[1])
    except Exception:
        params = dict(url.query)
    params.update(connect_args)
    return params


def _engine_autocommits(engine_kwargs: Dict[str, Any], pragmas: Sequence[str] = ()) -> bool:
    """Return whether an engine built from these arguments may commit every statement.

    Args:
        engine_kwargs: Keyword arguments the database passes to ``create_engine``.
        pragmas: Connection-initialization statements the engine runs on each
            new DBAPI connection.

    Returns:
        bool: ``True`` for SQLAlchemy ``AUTOCOMMIT`` isolation, given directly or
        through ``execution_options``; for a DBAPI connection that may open in
        autocommit mode; and for a pragma that names ``autocommit`` or
        ``implicit_transactions`` (case-insensitive), since it runs after the
        driver set the mode and may switch it. A DBAPI connect argument
        ``autocommit``, from the backend's ``params`` (the URL query) or
        ``connect_args``, opens one when the value the driver receives is true
        in Python, which is how pymysql and pymssql read it, or is ``None``,
        which leaves the mode to the driver or server default (pymysql and
        mysqlclient then send nothing, and MySQL's default is autocommit). The
        URL query reaches pymysql and pymssql as a string, so any non-empty
        value, even ``"false"``, counts there, while a dialect that converts the
        string first, such as SQLAlchemy's pyodbc dialect, passes ``"false"`` on
        as ``False``. ``sqlite3`` reads ``autocommit=True`` as autocommit and
        ``False`` as transactional. A DBAPI connect argument
        ``isolation_level=None`` counts as autocommit for every driver, as it is
        in ``sqlite3``'s default legacy mode, except on ``sqlite3`` when a
        ``bool`` ``autocommit`` is given too, which then decides.
    """
    from sqlalchemy.engine import make_url

    if any(_TRANSACTION_MODE_STATEMENT.search(str(pragma)) for pragma in pragmas):
        return True
    execution_options = engine_kwargs.get("execution_options") or {}
    for level in (engine_kwargs.get("isolation_level"), execution_options.get("isolation_level")):
        if isinstance(level, str) and level.upper() == "AUTOCOMMIT":
            return True
    url = make_url(engine_kwargs["url"])
    params = _dbapi_connect_params(url, engine_kwargs.get("connect_args") or {})
    autocommit = params.get("autocommit")
    legacy_autocommit = "isolation_level" in params and params["isolation_level"] is None
    if url.get_backend_name() == "sqlite":
        if isinstance(autocommit, bool):
            return autocommit
        return legacy_autocommit
    if "autocommit" in params and autocommit is None:
        return True
    return bool(autocommit) or legacy_autocommit


def _engine_isolation_level(engine_kwargs: Dict[str, Any]) -> str | None:
    """Return the isolation level an engine built from these ``create_engine`` arguments sets, upper-cased.

    Args:
        engine_kwargs: Keyword arguments the database passes to ``create_engine``.

    Returns:
        str | None: SQLAlchemy's ``isolation_level`` given directly or through ``execution_options``
        (the latter wins, as in SQLAlchemy), or ``None`` when the engine keeps the server's default.
    """
    execution_options = engine_kwargs.get("execution_options") or {}
    for level in (execution_options.get("isolation_level"), engine_kwargs.get("isolation_level")):
        if isinstance(level, str):
            return _normalized_isolation_level(level)
    return None


def _normalized_isolation_level(level: str) -> str:
    """Return ``level`` upper-cased with words separated by single spaces (``read-uncommitted`` reads ``READ UNCOMMITTED``)."""
    return " ".join(level.replace("_", " ").replace("-", " ").split()).upper()


# Dialects whose sessions may read at a level the engine's ``create_engine`` arguments do not show (a server
# default or a level set when each connection opens) and that honour ``READ UNCOMMITTED`` or need the level for
# the SQL Server rule; the backend asks such a database for the level its connections read at.
_SESSION_LEVEL_DIALECTS = frozenset({"mysql", "mariadb", "mssql", "sqlite"})


def _held_by_current_thread(lock: Any) -> bool:
    """Return whether the calling thread holds ``lock``, a ``threading.RLock``; ``False`` for a lock that cannot say."""
    is_owned = getattr(lock, "_is_owned", None)
    return bool(is_owned()) if callable(is_owned) else False


def _batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


# The most values one ``IN`` list and the other bound values of its statement bind on each dialect. Oracle
# refuses an ``IN`` list of more than 1,000 expressions (ORA-01795) and SQL Server a statement with more than 2,100
# parameters. SQLite binds at most 999 variables per statement before 3.32 and 32,766 since, its default
# compile-time defaults. Active transactions can also use the connection's actual limit.
# Any other dialect binds at most 32,766, half of the 65,535 parameters PostgreSQL's protocol allows one statement.
# A statement that binds other values besides its id list leaves room for them: a ``get_many`` projecting side-table
# fields binds the JSON key names of their items, and the sparse-GRAM adapter binds a namespace, entity and field.
_ID_LIST_LIMITS = {"oracle": 1000, "mssql": 2000}
_DEFAULT_ID_LIST_LIMIT = 32766
_SQLITE_LEGACY_VARIABLE_LIMIT = 999
_SQLITE_VARIABLE_LIMIT = 32766


def _sqlite_variable_limit(db: Any) -> int:
    """Return the SQLite bind budget, capped by the active connection when available."""
    dbapi = getattr(getattr(getattr(db, "engine", None), "dialect", None), "dbapi", None)
    version = getattr(dbapi, "sqlite_version_info", None)
    budget = _SQLITE_VARIABLE_LIMIT if version is not None and tuple(version) >= (3, 32, 0) else _SQLITE_LEGACY_VARIABLE_LIMIT
    category = getattr(dbapi, "SQLITE_LIMIT_VARIABLE_NUMBER", None)
    if category is not None and db.in_transaction():
        driver = db._conn.connection.driver_connection
        getlimit = getattr(driver, "getlimit", None)
        if getlimit is not None:
            return min(budget, getlimit(category))
    return budget


def _bound_values_besides(stmt: Any, key: str) -> int:
    """Return how many bound values ``stmt`` renders besides the bind parameter named ``key``.

    Every occurrence counts, so a parameter rendered twice counts twice; the count is never below what the
    statement binds.
    """
    from sqlalchemy.sql import visitors
    from sqlalchemy.sql.elements import BindParameter

    return sum(1 for element in visitors.iterate(stmt) if isinstance(element, BindParameter) and element.key != key)


class _PoolSlots:
    """Share one engine's bounded queue pool between its backends' locks and the reads running outside them.

    A queue pool hands out at most ``pool_size + max_overflow`` connections, and a checkout that waits longer
    than ``pool_timeout`` fails. Each SQL backend on the engine keeps one connection for the thread that holds
    its lock (a mutation, DDL, or a read that keeps the lock), and the reads running outside their backends'
    locks share the rest, less every connection checked out by anything else on the engine (a ``Database``
    facade's own statements, or a backend that has not yet read or upserted), which the pool's checkout and
    checkin events count. Ordinary reads wait here, without a time limit, when other reads use their share.
    When the backends and other users leave no connection over, ordinary reads keep their backend lock.
    A read nested on the same thread and engine as a slot read can borrow a reserved connection. It remains
    subject to the pool's checkout timeout.
    """

    _by_engine: "weakref.WeakKeyDictionary[Any, _PoolSlots]" = weakref.WeakKeyDictionary()
    _registry_lock = threading.Lock()
    _POLL_SECONDS = 0.05  # how soon a waiting read notices a connection another user checked in
    _WAIT_STEP = 1.0  # how soon a waiting read notices a slot freed by a read that was interrupted before it could notify

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._owners: "weakref.WeakValueDictionary[int, Any]" = weakref.WeakValueDictionary()
        # The lock itself is entered, which an interrupted call always leaves free; the condition is only waited on and notified.
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._others: Dict[Any, None] = {}  # pool records checked out by neither a slot read nor a lock holder
        self.reads: Dict[Any, Any] = {}  # reads holding a slot -> the thread of each (see _me)
        self._thread = threading.local()

    @classmethod
    def for_engine(cls, engine: Any) -> "_PoolSlots | None":
        """Return the slots of ``engine``'s pool, or ``None`` when the pool hands out connections without a bound.

        Args:
            engine: SQLAlchemy engine whose pool the reads share.

        Returns:
            _PoolSlots | None: The engine's shared slots for a queue pool with a finite ``pool_size`` and
            ``max_overflow``; ``None`` for a ``null`` pool, a queue pool with ``pool_size: 0`` or a negative
            ``max_overflow``, or any other pool class.
        """
        from sqlalchemy import event
        from sqlalchemy.pool import QueuePool

        pool = engine.pool
        if not isinstance(pool, QueuePool):
            return None
        size, overflow = pool.size(), getattr(pool, "_max_overflow", -1)
        if size <= 0 or overflow < 0:
            return None
        with cls._registry_lock:
            slots = cls._by_engine.get(engine)
            if slots is None:
                slots = cls._by_engine[engine] = cls(size + overflow)
                # A recreated pool (``engine.dispose()``) keeps these listeners.
                event.listen(pool, "checkout", slots._checked_out)
                event.listen(pool, "checkin", slots._checked_in)
                event.listen(pool, "detach", slots._checked_in)
            return slots

    def _checked_out(self, dbapi_connection: Any, record: Any, proxy: Any) -> None:
        """Count a checkout made neither by a read holding a slot nor by a thread holding a joined backend's lock."""
        if self._me() in self.reads.values():
            return
        with self._lock:
            owners = list(self._owners.values())
        if not any(_held_by_current_thread(getattr(owner, "_lock", None)) for owner in owners):
            self._others[record] = None

    def _checked_in(self, dbapi_connection: Any, record: Any) -> None:
        """Forget a counted checkout once its connection is back in the pool or detached from it.

        The pool may call this from a garbage-collection finalizer on any thread, so it takes no lock.
        """
        self._others.pop(record, None)

    def join(self, owner: Any) -> None:
        """Keep one connection for ``owner``'s lock holder until ``owner`` leaves or is collected."""
        with self._lock:
            self._owners[id(owner)] = owner

    def leave(self, owner: Any) -> None:
        """Release the connection kept for ``owner``'s lock holder."""
        with self._lock:
            self._owners.pop(id(owner), None)
            self._changed.notify_all()

    def _me(self) -> Any:
        """Return an object that stands for the calling thread and, unlike its identifier, for no later one."""
        me = getattr(self._thread, "me", None)
        if me is None:
            me = self._thread.me = object()
        return me

    def acquire(self, read: Any = None) -> bool:
        """Wait for a connection free for a read outside its backend lock.

        Args:
            read: Identity of the read, which frees its slot by removing it from ``reads`` and calling ``freed``.
                ``None`` lets :meth:`release` free the slot.

        Returns:
            bool: ``True`` once an outer read holds a slot, or immediately for a nested read. ``False`` when
            kept connections and other users leave no slot for an outer read, so it takes its backend lock.
        """
        me = self._me()
        with self._lock:
            # Nested reads count against capacity but go straight to pool checkout. A callback
            # cannot wait for its outer read to release the slot or rely on acquiring a backend lock.
            while me not in self.reads.values():
                # A connection whose checkin event an interrupt cut off is not in use any more.
                others = sum(1 for record in tuple(self._others) if record.in_use)
                free = self._capacity - len(self._owners) - others
                if free <= 0:
                    self._changed.notify_all()  # every other waiter must take its lock too
                    return False
                if len(self.reads) < free:
                    break
                # A slot read's release notifies; a checkin by another user cannot, so poll for it. An interrupted
                # read may end without notifying, so check again in either case.
                self._changed.wait(self._POLL_SECONDS if others else self._WAIT_STEP)
            self.reads[object() if read is None else read] = me
            return True

    def release(self) -> None:
        """Return a slot taken by :meth:`acquire` without a ``read`` on the same thread."""
        me = self._me()
        self.reads.pop(next((read for read in reversed(tuple(self.reads)) if self.reads.get(read) == me), None), None)
        self.freed()

    def freed(self) -> None:
        """Wake a read waiting for a slot after one was freed."""
        with self._lock:
            self._changed.notify()


# How long ``drop`` waits for the reads of the database it drops that are running outside the backend lock.
_DROP_READ_WAIT_SECONDS = 30.0


class _ReadGate:
    """Track one SQL backend's reads that run their statements outside its lock, for ``drop``.

    A read running its statements outside the backend lock counts in ``running`` until it ends, so ``drop`` can wait
    for it before it removes the database; a database that ``drop`` has begun to remove is in ``dropped`` and admits
    no further read.
    """

    def __init__(self) -> None:
        # The lock itself is entered, which an interrupted call always leaves free; the condition is only waited on and notified.
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.running: set[Any] = set()
        self.dropped: "weakref.WeakSet[Any]" = weakref.WeakSet()


_MYSQL_CHARSET_NAME = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class _SideTableSpec:
    field_schema: Any
    table: Any
    # SQL dialect that projects and decodes this side table's aggregate.
    dialect_spec: SQLDialectSpec
    # Character set and collation of the stored ``object_id`` column on MySQL, when known.
    id_collation: tuple[str, str] | None = None

    @property
    def name(self) -> str:
        """Return the source field name for this side table."""
        return self.field_schema.name

    @property
    def dtype_identifier(self) -> str:
        """Return the logical type identifier stored in this side table."""
        return type(self.field_schema.dtype).identifier

    def rows(self, object_id: Any, value: Any) -> list[dict[str, Any]]:
        """Lower an Array or HyperG value into side-table row dictionaries.

        Args:
            object_id: Owner row object id.
            value: Logical field value to decompose into side-table rows.

        Returns:
            list[dict[str, Any]]: Backend-ready side-table row dictionaries.

        Raises:
            TypeError: If the side-table field logical type is unsupported.
            ValueError: If the value is not list-shaped.
        """
        items = value or []
        if not isinstance(items, list):
            raise ValueError(f"{self.name} SideTable expects a list")
        if self.dtype_identifier == "array":
            return [self._array_row(object_id, idx, item) for idx, item in enumerate(items)]
        if self.dtype_identifier == "hyper-g":
            return [self._hyperg_row(object_id, idx, item) for idx, item in enumerate(items)]
        raise TypeError(f"SideTable does not support {self.dtype_identifier}")

    def group(self, records: Any) -> Dict[Any, list]:
        """Group side-table records back by owning ``object_id``.

        Args:
            records: SQLAlchemy result records from the side table.

        Returns:
            Dict[Any, list]: Side-table values grouped by owner object id.
        """
        grouped: Dict[Any, list] = {}
        for record in records:
            data = record._mapping
            grouped.setdefault(data["object_id"], []).append(self._value(data))
        return grouped

    def select_cols(self) -> list[Any]:
        """Return SQL columns needed to rebuild this side-table field.

        Args:
            None.

        Returns:
            list[Any]: SQLAlchemy columns required for reconstruction.
        """
        if self.dtype_identifier == "array":
            return [self.table.c.object_id, self.table.c.idx, self.table.c.item]
        return [self.table.c.object_id, self.table.c.idx, *[self.table.c[field.name] for field in self.field_schema.dtype.fields]]

    def order_cols(self) -> list[Any]:
        """Return stable ordering columns for side-table scans.

        Args:
            None.

        Returns:
            list[Any]: SQLAlchemy columns used for deterministic side-table ordering.
        """
        return [self.table.c.object_id, self.table.c.idx]

    def owner_matches(self, main_table: Any, mysql_ids: bool) -> Any:
        """Return the predicate that joins side rows to the main row that owns them.

        On MySQL the side table and the main table may declare ``object_id`` with
        different collations, for example a side table created by this release
        beside a main table created by an earlier one. The main ID is converted
        to the side column's character set and collation, so the side table's
        ``object_id`` index finds the candidate rows; the binary comparison then
        keeps only the exact ID. Without a known side collation the binary
        comparison alone stays exact, but it cannot use the index.

        Args:
            main_table: SQLAlchemy table for the owning entity.
            mysql_ids: Compare IDs exactly for the MySQL backend.

        Returns:
            Any: SQLAlchemy predicate correlating the side table with ``main_table``.
        """
        side, main = self.table.c.object_id, main_table.c.object_id
        if not mysql_ids:
            return side == main
        exact = _mysql_id_key(side) == _mysql_id_key(main)
        if self.id_collation is None:
            return exact
        charset, collation = self.id_collation
        lookup = sa.collate(sa.func.convert(main.op("USING")(sa.literal_column(charset))), collation)
        return (side == lookup) & exact

    def json_expr(self, main_table: Any, *, mysql_ids: bool) -> Any:
        """Return a correlated JSON aggregation expression for SELECT projection.

        Args:
            main_table: SQLAlchemy table for the owning entity.
            mysql_ids: Compare IDs exactly for the MySQL backend.

        Returns:
            Any: SQLAlchemy scalar subquery labeled with the logical field name.
        """
        from sqlalchemy import select

        dialect_spec = self.dialect_spec
        payload = (
            self.table.c.item
            if self.dtype_identifier == "array"
            else dialect_spec.side_table_json_object_expression(tuple((field.name, self.table.c[field.name]) for field in self.field_schema.dtype.fields))
        )
        aggregate = dialect_spec.side_table_json_agg_expression(payload, self.table.c.idx)
        stmt = select(aggregate).where(self.owner_matches(main_table, mysql_ids))
        return stmt.scalar_subquery().label(self.name)

    def decode(self, value: Any) -> list:
        """Decode a projected JSON value back to a Python list.

        Args:
            value: Projected aggregate value from a SQL SELECT.

        Returns:
            list: Decoded logical field value.
        """
        if value in (None, ""):
            return []
        data = self.dialect_spec.side_table_json_agg_items(value if isinstance(value, list) else loads_json(value))
        return data if self.dtype_identifier == "array" else [self._value(item) for item in data]

    def _side_row_id(self, object_id: Any, idx: int) -> str:
        return str(md5hash([self.table.name, object_id, idx]))

    def _array_row(self, object_id: Any, idx: int, item: Any) -> dict[str, Any]:
        return {"_id": self._side_row_id(object_id, idx), "object_id": object_id, "idx": idx, "item": item}

    def _hyperg_row(self, object_id: Any, idx: int, item: Any) -> dict[str, Any]:
        return {"_id": self._side_row_id(object_id, idx), "object_id": object_id, "idx": idx, **self._hyperg_item(item)}

    def _hyperg_item(self, item: Any) -> dict[str, Any]:
        data = self.field_schema.dtype.validate([item])[0]
        return {field.name: data.get(field.name) for field in self.field_schema.dtype.fields}

    def _value(self, data: Any) -> Any:
        if self.dtype_identifier == "array":
            return data["item"]
        out = {}
        for field in self.field_schema.dtype.fields:
            value = data.get(field.name)
            if value is None and not field.required:
                continue
            out[field.name] = loads_json(value) if type(field.dtype).identifier == "json" and isinstance(value, str) else value
        return out


@dataclass(frozen=True)
class _SharedRowStorage:
    routes: tuple[str, ...]
    table: Any
    side_specs: dict[str, _SideTableSpec]
    gram_aliases: tuple[Any, ...]


class SQLBackend(Backend):
    """SQLAlchemy backend using HeavenBase ``Database`` for connection lifecycle.

    Reads (``scan``, ``scan_page``, ``get_many``, ``exists_many``, ``count``,
    and ``execute``) take the backend lock only to plan: they resolve the live
    ``Database``, the entity's table and side-table specs, and the cached
    statement under it, then run the statement and decode its rows outside it,
    so concurrent readers share the engine's connection pool instead of one
    lock. Mutations and lifecycle methods (``upsert``, ``ensure``,
    ``ensure_indexes``, ``drop``, and workspace rebinding) hold the lock for
    their whole call, as does the construct cache, which is touched only under
    it; a read therefore still waits for a mutation in progress before it can
    plan. A read keeps the lock while its statement runs as well in two cases. When the engine cannot serve two threads
    at once (``pool_class`` ``static``, ``assertion`` or ``singleton``, or any
    in-memory database, whose second connection would be a separate empty
    database): on the one shared DBAPI connection a reader's rollback would
    discard a writer's uncommitted rows. ``Database`` keeps such an engine private
    to this backend's facade, so the backend lock covers every statement on it,
    even when another backend opens the same database. And when the backend type
    does not claim the ``atomic-upsert`` tag, so its ``upsert`` may not be one
    transaction (the generic ``sql`` type, ClickHouse, StarRocks, Trino, a type
    the Registry does not know, or an engine that may commit every statement): a
    read on the same backend instance then cannot
    see a row between the upsert's delete and its insert. A type that claims the
    tag commits each ``upsert`` as one transaction. Reads keep the lock as well
    when one statement may see part of another transaction: when the
    connections read at ``READ UNCOMMITTED``, and on SQL Server unless they
    read at ``SNAPSHOT``, or at ``READ COMMITTED`` (the default) on a database
    that reads it from row versions (``READ_COMMITTED_SNAPSHOT ON``). The
    level is the engine's ``isolation_level``; on MySQL, MariaDB, SQL Server
    and SQLite it is the level the database reports for a pooled connection,
    so a server default or a level set as each connection opens counts too.
    Otherwise each statement a read runs outside the lock sees an upsert
    either whole or not at all. ``get_many`` and ``exists_many`` over more
    ids than one batch run their batches under the lock, as in 0.1.2.2, so
    they see a write of this backend whole or not at all too; ``iter_scan``
    runs one statement per page and may see a write between two. ``drop``
    waits for the reads running outside the lock before it removes the
    database, and a read that would run after that raises ``RuntimeError``
    (see ``_read_scope``).

    On a queue pool with a finite ``pool_size`` and ``max_overflow`` the
    connections are shared out per engine: each backend that has read or
    upserted keeps one for the thread holding its lock, and the reads outside
    their locks share the rest, less the connections other users of the engine
    (a ``Database`` facade on the same database, say) hold, waiting without a
    time limit for one to be free instead of failing after ``pool_timeout``.
    A read made by a thread that holds the backend's lock runs under it on the
    kept connection. When no connection is left for ordinary reads, they keep
    the lock and run one at a time, as in 0.1.2.2. A read nested on the same
    thread and engine as a slot read can use a kept connection and follows
    ``pool_timeout`` if the pool is exhausted. Nested reads and direct engine
    users can therefore make a lock holder wait in the pool.

    Args:
        name: Workspace backend instance name.
        provider: SQLAlchemy provider or dialect identifier.
        database: Optional database name, URL, or local database path.
        identifier: Optional HeavenBase backend type identifier. Defaults to ``provider``.
        ws_id: Optional workspace namespace.
        **kwargs: Provider-specific database connection options.
    """

    identifier = "sql"
    sql_name_warning_provider: str | None = None
    sql_reserved_words: Sequence[str] = ()
    sql_max_identifier_length: int | None = None
    #: Upper bound on reusable statement constructs kept per backend; the oldest is evicted first.
    _construct_cache_size = 512
    #: Whether this backend type speaks MySQL's SQL grammar and compares ``object_id`` exactly on a MySQL connection.
    _mysql_exact_ids = False
    #: This backend type's ``atomic-upsert`` claim, cached once the Registry answers.
    _atomic_upsert_claim: bool | None = None
    #: Whether this backend type keeps a sparse-GRAM keyword index for its ``SparseGramIndex`` fields.
    #: A type whose 0.1.2.2 release kept none (pgvector, StarRocks) keeps none either: 0.1.2.2 still
    #: writes such a database without touching any index, so an index kept here could not be trusted.
    _keeps_keyword_index = True
    #: True while a resolved storage plan, not ``ensure``, decides the entity's scalar indexes.
    _plan_owns_scalar_indexes = False

    @classmethod
    def from_config(
        cls,
        name: str,
        data: dict[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "SQLBackend":
        """Build a SQL backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``provider``, ``database``, optional
                ``ws_id``, and provider-specific database connection options.
            resolver: Context-owned module authority.
            config: Config authority used for backend defaults and database resolution.

        Returns:
            SQLBackend: Configured SQL backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        provider = str(cfg.pop("provider", "sqlite") or "sqlite").lower()
        database = default_database(ws_id, cfg)
        return cls(
            name,
            provider=provider,
            database=database,
            identifier=provider,
            ws_id=ws_id,
            resolver=resolver,
            config=config,
            **cfg,
        )

    def __init__(
        self,
        name: str = "sql",
        provider: str = "sqlite",
        database: str | None = None,
        identifier: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        config_engine: DBEngine | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a SQL backend instance.

        Args:
            name: Workspace backend instance name.
            provider: SQLAlchemy provider or dialect identifier.
            database: Optional database name, URL, or local database path.
            identifier: Optional HeavenBase backend type identifier. Defaults to ``provider``.
            ws_id: Optional workspace namespace.
            resolver: Context-owned module authority.
            config: Config authority used for backend defaults and database resolution.
            config_engine: Explicit database policy engine used by control-plane bootstrap.
            **kwargs: Provider-specific database connection options.

        Returns:
            None: This function does not return a value.
        """
        if config is not None and resolver is None and config_engine is None:
            from heavenbase.utils.config import CM_HVNB

            if config is CM_HVNB:
                from heavenbase.context.authority import resolve_runtime_authority
                from heavenbase.context.runtime import _BOOTSTRAPPING_CONFIG

                if _BOOTSTRAPPING_CONFIG.get() is not config:
                    authority = resolve_runtime_authority(config=config, label="SQLBackend")
                    resolver, config = authority.resolver, authority.config
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        from sqlalchemy import MetaData

        self._sql_id_lookup_batch_size = max(
            1,
            int(self._construction_config_get("heavenbase.backend_defaults.sql.id_lookup.batch_size", default=500)),
        )
        self._sql_long_text_max_chars = LongText.default_max_chars
        kwargs = self._database_config(dict(kwargs))
        self.provider = provider
        self.type = identifier or provider
        self._database_override = database
        self._database_config_engine = config_engine
        self._db_kwargs = dict(kwargs)
        self.database = self._database_name()
        self.db = None
        self.metadata = MetaData()
        self.tables: Dict[str, Any] = {}
        self._side_specs: Dict[str, Dict[str, _SideTableSpec]] = {}
        self._shared_row_storage: dict[str, _SharedRowStorage] = {}
        # Routes decided after staging to share this route's database, keyed by entity: (any session, compatible sessions),
        # and the databases that join this route's sparse-GRAM publication transaction.
        self._staged_row_aliases: dict[str, tuple[list[SQLBackend], list[SQLBackend]]] = {}
        self._staged_publication_aliases: dict[str, list[Any]] = {}
        self._constructs: Dict[tuple[str, str], Any] = {}
        #: Tables whose storage engine keeps no transactions (see ``_note_storage_engines``).
        self._nontransactional_tables: set[str] = set()
        self._connect_database()
        # Standalone SQL backends resolve through the documented process
        # default and can install their adapters immediately.  The sole
        # deferred case is a Context control backend carrying config before
        # its ModuleService has been constructed.
        if self._module_resolver is not None or self._config is None:
            self._install_strategy_adapters()

    def _finalize_bootstrap_resolver(self, resolver: ModuleResolver) -> None:
        """Finalize control-plane resolver authority without rebuilding SQL state."""
        super()._finalize_bootstrap_resolver(resolver)
        if self.db is not None:
            self.db._finalize_bootstrap_resolver(resolver)
        if not self._strategy_adapters:
            self._install_strategy_adapters()
        with self._lock:
            self._forget_constructs()

    def _database_config(self, values: dict[str, Any]) -> dict[str, Any]:
        """Return provider database values after backend-bound default resolution."""
        return values

    def _uses_workspace_database(self) -> bool:
        if not self.ws_id:
            return False
        if self._module_resolver is None and self._config is not None:
            if self._database_config_engine is None:
                raise RuntimeError("Direct SQL control bootstrap requires an explicit database config engine")
            value = self._database_config_engine.bootstrap_backend_tag("workspace-database-namespace", False)
        else:
            result = self.tag("workspace-database-namespace")
            value = False if result is None else result.value
        if not isinstance(value, bool):
            raise TypeError("Backend tag 'workspace-database-namespace' must be Boolean")
        return value

    def _construction_tag_claims(self) -> dict[str, Any]:
        """Withdraw the ``atomic-upsert`` claim when the configured engine may commit every statement.

        A ``db.tx()`` block alone does not stop the driver from committing each
        statement. The SQLite transaction for strategy-maintained writes does
        not cover writes with strategies disabled. A backend makes no
        ``atomic-upsert`` claim, whatever its
        type declares, when its engine uses SQLAlchemy ``isolation_level="AUTOCOMMIT"``
        (directly or through ``execution_options``); when an ``autocommit``
        connect argument from ``params`` or ``connect_args`` is true in Python or
        ``None`` (the driver or server default); when an ``isolation_level``
        connect argument is ``None`` (``sqlite3``'s legacy autocommit mode),
        unless ``sqlite3`` also receives a ``bool`` ``autocommit``; or when one of its ``pragmas`` names
        ``autocommit`` or ``implicit_transactions`` and so may switch the mode
        after the driver set it. The SQL sparse-GRAM adapter keeps no keyword
        index on such an engine either (see ``SQLSparseGramAdapter.maintains_index``).

        Returns:
            dict[str, Any]: Construction overlays, holding ``atomic-upsert: False``
            for such an engine.
        """
        claims = super()._construction_tag_claims()
        if getattr(self, "_autocommit_engine", False) or getattr(self, "_nontransactional_tables", None):
            claims["atomic-upsert"] = False
        return claims

    def _note_storage_engines(self, tables: Sequence[Any]) -> None:
        """Withdraw the ``atomic-upsert`` claim once a table of this backend has a storage engine without transactions.

        On a MySQL-dialect server each table names its own storage engine. A
        table kept by MyISAM, MEMORY, Aria or another engine without
        transactions keeps every statement the moment it runs, so a rollback
        undoes none of its rows and a write call to it cannot apply all of its
        rows or none. The engine of each table is read from
        ``information_schema`` when the table is ensured; an engine that
        ``information_schema.ENGINES`` lists with ``TRANSACTIONS`` ``NO``, or a
        known engine without transactions that it does not list, withdraws the
        claim of the whole backend (see ``_construction_tag_claims``), so
        ``atomic=True`` is refused on it from then on.
        """
        if self.db is None or getattr(self.db, "dialect", None) != "mysql" or not tables:
            return
        if not self._nontransactional_tables and not self._upserts_are_atomic():
            # a type that claims no atomic upserts (``starrocks``, the generic ``sql`` type) has nothing to withdraw
            return
        from sqlalchemy import bindparam, func, literal, or_, select, table as sql_table, column

        info = sql_table("TABLES", column("TABLE_SCHEMA"), column("TABLE_NAME"), column("ENGINE"), schema="information_schema")
        engines = sql_table("ENGINES", column("ENGINE"), column("TRANSACTIONS"), schema="information_schema")
        names: dict[str | None, list[str]] = {}
        for table in tables:
            names.setdefault(table.schema, []).append(table.name)
        stmt = (
            select(info.c.TABLE_NAME, info.c.ENGINE, engines.c.TRANSACTIONS)
            .select_from(info.outerjoin(engines, func.upper(engines.c.ENGINE) == func.upper(info.c.ENGINE)))
            .where(
                or_(
                    *(
                        (info.c.TABLE_SCHEMA == (literal(schema) if schema is not None else func.database()))
                        & info.c.TABLE_NAME.in_(bindparam(f"names_{position}", tuple(group), expanding=True))
                        for position, (schema, group) in enumerate(names.items())
                    )
                )
            )
        )
        found = set()
        for name, engine, transactions in self.db.orm_execute(stmt, readonly=True).to_lists():
            answer = str(transactions or "").strip().upper()
            if answer == "NO" or (answer != "YES" and str(engine or "").strip().lower() in _NONTRANSACTIONAL_MYSQL_ENGINES):
                found.add(str(name))
        noted = (self._nontransactional_tables - {table.name for table in tables}) | found
        if noted != self._nontransactional_tables:
            self._nontransactional_tables = noted
            self._atomic_upsert_claim = None
            self._whole_commit_reads = None

    def _database_name(self) -> str | None:
        if self._uses_workspace_database():
            return self._physical_name(self.ws_id)
        if self._database_override:
            return self._database_override
        return self._physical_name(self.ws_id) if self.ws_id else None

    def _connect_database(self) -> None:
        from heavenbase.database import Database

        self.database = self._database_name()
        kwargs = dict(self._db_kwargs)
        self.db = Database(
            database=self.database,
            provider=self.provider,
            resolver=self._module_resolver,
            config=self.config,
            config_engine=self._database_config_engine,
            **kwargs,
        )
        engine_kwargs = DB_ENGINE.apply(self.db.spec, profile="engine")
        self._autocommit_engine = _engine_autocommits(engine_kwargs, DB_ENGINE.apply(self.db.spec, profile="pragmas"))
        self._isolation_level = _engine_isolation_level(engine_kwargs)
        self._atomic_upsert_claim = None  # the claim depends on the engine just built
        self._whole_commit_reads = None  # and so does the read claim
        self._leave_pool()

    def _read_scope(self, db: Any) -> AbstractContextManager[Any]:
        """Return the scope one read holds while its statement runs and its rows are decoded.

        A read plans under the backend lock and normally runs outside it on its
        own pooled connection. An engine that cannot serve two threads at once
        defeats that: a ``static`` pool hands every checkout the same DBAPI
        connection, on which a reader's rollback would discard a writer's
        uncommitted rows; an ``assertion`` pool refuses a second checkout; a
        ``singleton`` pool closes other threads' connections, even one running
        a statement, once more threads than its ``pool_size`` (five by default)
        have checked out; and an in-memory database exists only inside its first
        connection, so a second pooled connection (``queue`` or ``null`` on
        ``:memory:``) is a separate empty database. The engine is private to this
        backend's ``Database`` facade, so no other backend's threads use it. A
        backend type that does not claim ``atomic-upsert`` defeats it too: its
        ``upsert`` may run as separate statements, and a read outside the lock
        could run between two of them. So does an isolation level at which one
        statement may see part of another transaction (see
        ``_probe_whole_commit_reads``). In these cases the read takes the backend
        lock again for that phase and reads run one at a time.

        A read made by a thread that already holds this backend's lock (the
        sparse-GRAM adapter reads rows that way) stays under it and uses the
        connection kept for the lock holder. Otherwise the read takes a slot of
        its engine's bounded pool (see ``_PoolSlots``) for that phase, waiting
        without a time limit while the reads outside their locks use every
        connection that neither the backends on the engine keep for their lock
        holders nor other users of the engine hold. It takes the backend lock
        instead when no connection is left for ordinary reads. A synchronous
        nested read on the same engine goes directly to the pool when an outer
        read holds a slot on this thread. It can use a kept connection and
        follows ``pool_timeout`` if the pool is exhausted. Nested reads and
        direct engine users can also exhaust the pool for a lock holder.

        A read outside the lock counts as running until its scope ends, and
        ``drop`` waits for the running reads before it removes the database (see
        ``_drain_reads``). A read of a database that ``drop`` has begun to
        remove raises ``RuntimeError`` instead, under the lock or outside it, and
        so does a read whose statement fails because a ``drop`` that stopped
        waiting removed its database meanwhile.

        Args:
            db: The ``Database`` the read captured under the backend lock. The
                rule reads its engine from this snapshot rather than from the
                backend, which ``drop`` may have cleared since.

        Returns:
            AbstractContextManager[Any]: The backend lock when the engine or
            backend type serializes reads, otherwise a running read that holds
            a pool slot on a bounded queue pool.

        Raises:
            RuntimeError: If ``drop`` has begun to remove ``db``.
        """
        if self._serialized_reads(db) or _held_by_current_thread(self._lock):
            return self._locked_read(db)
        return self._running_read(db, self._pool_slots(db))

    def _read_gate(self) -> _ReadGate:
        """Return this backend's read gate, creating it for an instance restored without ``__init__``."""
        gate = self.__dict__.get("_reads")
        if gate is None:
            gate = self.__dict__.setdefault("_reads", _ReadGate())
        return gate

    def _unavailable_after_drop(self) -> RuntimeError:
        return RuntimeError(f"SQL backend {self.name!r} is unavailable after drop")

    @contextmanager
    def _locked_read(self, db: Any) -> Iterator[None]:
        """Hold the backend lock for one read of ``db``, refusing it once ``drop`` has begun to remove ``db``."""
        with self._lock:
            if db in self._read_gate().dropped:
                raise self._unavailable_after_drop()
            yield

    @contextmanager
    def _running_read(self, db: Any, slots: _PoolSlots | None) -> Iterator[None]:
        """Run one read of ``db`` outside the backend lock, counted for ``drop``, in a slot of ``slots`` when bounded.

        When the backends on the engine keep every connection, the read stops
        counting and takes the backend lock instead (``_locked_read``).
        """
        gate = self._read_gate()
        read = object()
        try:
            with gate.lock:
                if db in gate.dropped:
                    raise self._unavailable_after_drop()
                gate.running.add(read)
            if slots is None or slots.acquire(read):
                try:
                    yield
                except Exception as exc:
                    if db in gate.dropped:
                        raise self._unavailable_after_drop() from exc
                    raise
                return
        finally:
            # Each removal comes before anything an interrupt can cut off, so the counts end right in any case.
            try:
                if slots is not None:
                    slots.reads.pop(read, None)
            finally:
                gate.running.discard(read)
            if slots is not None:
                slots.freed()
            with gate.lock:
                gate.changed.notify_all()
        with self._locked_read(db):
            yield

    def _drain_reads(self, db: Any) -> None:
        """Admit no further read of ``db`` and wait for the reads running outside the lock to end.

        ``drop`` calls this under the backend lock before it removes the
        database. A running read never waits for that lock, so the wait ends
        once the running reads do; it gives up after ``_DROP_READ_WAIT_SECONDS``
        (30 seconds), and a read still running then fails with ``RuntimeError``
        if its database is gone.
        """
        gate = self._read_gate()
        deadline = time.monotonic() + _DROP_READ_WAIT_SECONDS
        try:
            with gate.lock:
                gate.dropped.add(db)
                while gate.running:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    # An interrupted read may end without notifying, so check again.
                    gate.changed.wait(min(remaining, 1.0))
        except BaseException:
            # Interrupted before anything was dropped.
            self._admit_reads(db)
            raise

    def _admit_reads(self, db: Any) -> None:
        """Admit reads of ``db`` again after a drop that was refused and removed nothing."""
        gate = self._read_gate()
        with gate.lock:
            gate.dropped.discard(db)

    def _serialized_reads(self, db: Any) -> bool:
        """Return whether a read on ``db`` must keep the backend lock while its statement runs.

        One rule: the engine cannot serve two threads at once, the backend type
        does not claim ``atomic-upsert``, or one read statement may see part of
        another transaction at the engine's isolation level.
        """
        return db is not self.db or db.spec.serialized or not self._upserts_are_atomic() or not self._reads_see_whole_commits(db)

    def _reads_see_whole_commits(self, db: Any) -> bool:
        """Return whether one read statement on ``db`` sees each other transaction's writes either all or none, cached per connection.

        The first read after the backend connects answers it under the backend
        lock, having joined the engine's pool slots, so the database is asked
        once, on the connection kept for the lock holder.
        """
        claim = self.__dict__.get("_whole_commit_reads")
        if claim is None:
            with self._lock:
                claim = self.__dict__.get("_whole_commit_reads")
                if db is not self.db:
                    return False
                if claim is None:
                    self._pool_slots(db)
                    claim = self._whole_commit_reads = self._probe_whole_commit_reads(db)
        return claim

    def _probe_whole_commit_reads(self, db: Any) -> bool:
        """Return whether one read statement sees each other transaction's writes either all or none.

        PostgreSQL, MySQL/InnoDB and the MySQL-compatible types, Oracle, SQLite
        and DuckDB read each statement from a snapshot of committed data at
        their default isolation level, and their stricter levels keep that
        guarantee. ``READ UNCOMMITTED`` reads rows a transaction has written
        but not yet committed, so a connection reading at it does not. The
        level is the engine's ``isolation_level`` (given directly or through
        ``execution_options``), except on MySQL, MariaDB, SQL Server and
        SQLite, whose connections may read at a level the engine does not set
        (a server default such as MySQL's ``transaction_isolation``, or a level
        set as each connection opens): there the database is asked for the
        level of a pooled connection, and a database that cannot answer keeps
        reads under the lock.

        SQL Server's default ``READ COMMITTED`` releases each row's shared lock
        once the row is read unless the database turns on
        ``READ_COMMITTED_SNAPSHOT``. One scan can then read some rows of an
        upsert before it starts, wait for its locks, and read the rest after it
        commits, or pair a row's old columns with its new side-table values. So
        on SQL Server only ``SNAPSHOT`` isolation, or ``READ COMMITTED`` (the
        default) on a database whose ``is_read_committed_snapshot_on`` is set,
        qualifies; ``REPEATABLE READ`` and ``SERIALIZABLE`` keep reads under the
        lock whatever the database option. The database is asked once each time
        the backend connects, and a failed query counts as no.

        Returns:
            bool: Whether reads of this engine may run outside the backend lock
            as far as transaction visibility goes.
        """
        level = self.__dict__.get("_isolation_level")
        dialect = db.spec.dialect
        if dialect in _SESSION_LEVEL_DIALECTS:
            level = self._session_isolation_level(db) or level
            if level is None:
                return False  # the database could not say what its connections read
        if level == "READ UNCOMMITTED":
            return False
        if dialect != "mssql":
            return True
        if level != "READ COMMITTED":
            return level == "SNAPSHOT"
        response = db.execute("SELECT is_read_committed_snapshot_on FROM sys.databases WHERE database_id = DB_ID()", readonly=True, safe=True)
        return bool(response.ok and response.to_lists() == [[1]])

    def _session_isolation_level(self, db: Any) -> str | None:
        """Ask the database for the isolation level a pooled connection of ``db`` reads at.

        Returns:
            str | None: The level SQLAlchemy reports for the connection, upper-cased with single spaces, or
            ``None`` when the database or its dialect cannot answer.
        """
        try:
            with nullcontext(db._conn) if db.in_transaction() else db.engine.connect() as connection:
                level = connection.get_isolation_level()
        except Exception:
            return None
        return _normalized_isolation_level(level) if isinstance(level, str) else None

    def _upserts_are_atomic(self) -> bool:
        """Return whether this backend claims ``atomic-upsert``, so each ``upsert`` is one transaction.

        The claim is the type's static claim with this instance's construction
        overlay (an engine that may commit every statement withdraws it), so it is
        read once per connected engine and cached. A type the Registry does not
        know counts as not claiming it. A Context
        bootstrap backend not yet bound to its modules has no answer; its reads
        keep the lock and ask again later.
        """
        claim = self._atomic_upsert_claim
        if claim is None:
            try:
                value = self.tag("atomic-upsert")
            except RuntimeError:
                return False
            except KeyError:
                value = None
            claim = self._atomic_upsert_claim = value is not None and value.value is True
        return claim

    @staticmethod
    def _id_list_limit(db: Any) -> int:
        """Return the most values one ``IN`` list and its statement may bind on ``db``'s dialect.

        It is 1,000 on Oracle, 2,000 on SQL Server, 999 on SQLite before 3.32
        and 32,766 on SQLite 3.32 or later and elsewhere. The active SQLite
        connection can further lower this budget.
        """
        dialect = db.spec.dialect
        if dialect == "sqlite":
            return _sqlite_variable_limit(db)
        return _ID_LIST_LIMITS.get(dialect, _DEFAULT_ID_LIST_LIMIT)

    def _id_batch_size(self, db: Any, other_binds: int = 0) -> int:
        """Return how many object ids one ``IN`` list binds on ``db``.

        It is the configured ``heavenbase.backend_defaults.sql.id_lookup.batch_size``
        (500 by default), capped by the dialect's limit less the ``other_binds``
        values the statement binds besides its id list, and at least 1.
        """
        return max(1, min(self._sql_id_lookup_batch_size, self._id_list_limit(db) - other_binds))

    def _install_strategy_adapters(self) -> None:
        from .strategies import SQLSparseGramAdapter

        self._strategy_adapters.clear()
        SQLSparseGramAdapter(self).register()

    def _need_db(self) -> None:
        if self.db is None:
            raise RuntimeError(f"SQL backend {self.name!r} is unavailable after drop")

    def _uses_mysql_id_keys(self) -> bool:
        # The MySQL-family backend types opt in: a MySQL-wire connection can serve a different SQL grammar (StarRocks).
        return self._mysql_exact_ids and self.db is not None and self.db.dialect in _MYSQL_DIALECTS

    def _mysql_id_collation(self, table: Any) -> tuple[str, str] | None:
        """Return the character set and collation that the stored ``object_id`` column of ``table`` declares on MySQL.

        An existing table keeps the collation it was created with, which can
        differ from the one this release declares. ``None`` means the server did
        not report a usable name.
        """
        from sqlalchemy import column, func, select, table as sql_table

        columns = sql_table(
            "COLUMNS",
            column("TABLE_SCHEMA"),
            column("TABLE_NAME"),
            column("COLUMN_NAME"),
            column("CHARACTER_SET_NAME"),
            column("COLLATION_NAME"),
            schema="information_schema",
        )
        stmt = select(columns.c.CHARACTER_SET_NAME, columns.c.COLLATION_NAME).where(
            columns.c.TABLE_SCHEMA == (table.schema if table.schema is not None else func.database()),
            columns.c.TABLE_NAME == table.name,
            columns.c.COLUMN_NAME == "object_id",
        )
        rows = self.db.orm_execute(stmt, readonly=True).to_dicts()
        if len(rows) != 1:
            return None
        charset, collation = (str(rows[0][name] or "") for name in ("CHARACTER_SET_NAME", "COLLATION_NAME"))
        if not (_MYSQL_CHARSET_NAME.fullmatch(charset) and _MYSQL_CHARSET_NAME.fullmatch(collation)):
            return None
        return charset, collation

    def _pool_slots(self, db: Any) -> _PoolSlots | None:
        """Return the slots this backend's reads on ``db`` share on its engine's pool, keeping a connection for its lock.

        The first call for a ``Database`` joins its engine's slots. Ordinary reads outside backend locks leave
        one connection for this backend's lock holder. Every read outside the lock and every ``upsert`` calls
        this before it uses the engine. ``None`` means the reads need no slot: the engine serializes its reads
        under the backend lock, or its pool is unbounded.
        """
        held = self.__dict__.get("_pool_share")
        if held is not None and held[0] is db:
            return held[1]
        with self._lock:
            if db is not self.db:
                return None
            held = self.__dict__.get("_pool_share")
            if held is not None and held[0] is db:
                return held[1]
            self._leave_pool()
            slots = None if db is None or db.spec.serialized else _PoolSlots.for_engine(db.engine)
            if slots is not None:
                slots.join(self)
            self._pool_share = (db, slots)
            return slots

    def _leave_pool(self) -> None:
        """Release the connection this backend kept on its previous engine's pool."""
        held = self.__dict__.pop("_pool_share", None)
        if held is not None and held[1] is not None:
            held[1].leave(self)

    @backend_locked
    def _refresh_workspace(self) -> None:
        next_database = self._database_name()
        if next_database == self.database:
            return
        self._reset_storage_state()
        self._connect_database()
        self._install_strategy_adapters()

    def _reset_storage_state(self) -> None:
        from sqlalchemy import MetaData

        self.metadata = MetaData()
        self.tables.clear()
        self._side_specs.clear()
        if getattr(self, "_nontransactional_tables", None):
            self._nontransactional_tables = set()
            self._atomic_upsert_claim = None
            self._whole_commit_reads = None
        self._shared_row_storage = {}
        self._staged_row_aliases = {}
        self._staged_publication_aliases = {}
        self._forget_constructs()
        for adapter in self._strategy_adapters.values():
            adapter.reset()
        self._strategy_adapters.clear()

    @backend_locked
    def drop(self) -> None:
        """Drop the current SQL workspace namespace.

        Args:
            None.

        Returns:
            None: This function does not return a value.

        Notes:
            Dialects without database-drop support fall back to dropping reflected tables.
            ``drop`` holds the backend lock, so no read plans meanwhile, and before it
            removes anything it waits for the reads already running their statements
            outside that lock to finish (for 30 seconds at most). A read that would start
            a statement on the database after that, including one planned before the
            drop, raises ``RuntimeError``, as does a read still running when a wait that
            gave up removed its database, if it fails.
            A local database file is removed with its ``-wal`` and ``-shm`` files by
            ``Database.drop_database()``, inside the drop; nothing is deleted after it
            returns, so a client that creates the database again meanwhile keeps its writes.
            A drop that ``Database.drop_database()`` refuses with ``DropRefusedError``,
            because a connection still uses the database (such as another Context's
            transaction on a DuckDB file), raises it, drops no table and leaves the
            backend usable.
        """
        if self.db is None:
            return
        from ...database.registry import DropRefusedError

        spec = self.db.spec
        refused = False
        self._drain_reads(self.db)
        try:
            if spec.in_memory or not spec.database or not spec.dialect_spec.supports("database-drop"):
                self.db.drop()
                self.db.dispose()
            else:
                try:
                    self.db.drop_database()
                except DropRefusedError:
                    # The drop removed nothing because a connection still uses the database: dropping its tables
                    # instead would remove what that connection's transaction works on.
                    raise
                except Exception:
                    # A failed namespace drop reopens the existing database.
                    # Fall back to dropping its reflected tables.
                    self.db.drop()
                    self.db.dispose()
        except DropRefusedError:
            # Nothing was dropped, so the backend keeps its storage state and stays usable.
            refused = True
            self._admit_reads(self.db)
            raise
        finally:
            if not refused:
                self._reset_storage_state()
                self._leave_pool()
                self.db = None

    def _my_bindings(self, bindings: Sequence[Any]) -> Sequence[Any]:
        return [b for b in bindings if b.backend == self.name]

    def _column_type(self, dtype: Any) -> Any:
        """Return this backend's physical SQLAlchemy type for a logical type."""
        return _sa_col(
            dtype,
            self.db.dialect_spec if self.db is not None else None,
            max_chars=self._sql_long_text_max_chars,
        )

    def _column_constraints(self, name: str, dtype: Any) -> tuple[Any, ...]:
        """Return provider-owned constraints for one physical column."""
        return ()

    def _column(self, name: str, dtype: Any, *, primary_key: bool = False, nullable: bool = True) -> Any:
        """Build one SQLAlchemy column through provider-owned physical hooks."""
        from sqlalchemy import Column

        physical_type = self._column_type(dtype)
        if name == "object_id" and type(dtype).identifier == "identifier" and self._uses_mysql_id_keys():
            physical_type = _MySQLObjectId(dtype.max_chars)
        return Column(name, physical_type, *self._column_constraints(name, dtype), primary_key=primary_key, nullable=nullable)

    def _table_name(self, entity_id: str) -> str:
        """Return the provider-owned physical main-table name."""
        return _table_name(entity_id)

    def _side_table_name(self, entity_id: str, field_name: str) -> str:
        """Return the provider-owned physical side-table name."""
        return _side_table_name(entity_id, field_name)

    def _table_args(self, *, order_by: Sequence[str]) -> tuple[Any, ...]:
        """Return provider-owned positional SQLAlchemy table arguments."""
        return ()

    def _create_main_table(self, table: Any) -> None:
        """Create the entity's main table unless the database already holds it."""
        self.db.create_table(table)

    def _add_stored_column(self, table: Any, column: Any) -> None:
        """Add ``column`` to the stored main ``table`` unless the database already holds it."""
        self.db.add_table_col(table, column)

    def _side_table_fields(self, bindings: Sequence[Any]) -> Sequence[Any]:
        return [b for b in self._my_bindings(bindings) if b.strategy == SideTable.identifier]

    def _main_table_fields(self, bindings: Sequence[Any]) -> Sequence[Any]:
        return [b for b in self._my_bindings(bindings) if b.strategy != SideTable.identifier]

    def _adapter_fields(self, schema: EntitySchema, bindings: Sequence[Any]) -> Sequence[Any]:
        return [
            binding
            for binding in self._my_bindings(bindings)
            if (adapter := self._strategy_adapters.get(binding.strategy)) is not None and adapter.supports(schema, binding)
        ]

    def _sql_name_warning_rules(self) -> tuple[int | None, set[str], bool]:
        """Return SQL identifier warning rules: max length, reserved words, and case sensitivity."""
        dialect = getattr(getattr(self.db, "engine", None), "dialect", None) if self.db is not None else None
        max_length, reserved = self.db.dialect_spec.name_warning_rules(
            dialect=dialect,
            extra_reserved=self.sql_reserved_words,
        )
        if self.sql_max_identifier_length is not None:
            max_length = self.sql_max_identifier_length
        return max_length, reserved, False

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> List[str]:
        """Return optional SQL physical-name warnings for this schema.

        Args:
            schema: Entity schema being registered.
            bindings: Storage bindings routed to this backend.

        Returns:
            List[str]: User-facing warnings about SQL table or column names.
        """
        warnings = super().schema_warnings(schema, bindings)
        max_length, reserved, case_sensitive = self._sql_name_warning_rules()
        table_names = [
            (entity_id, table.name) for entity_id, table in self.tables.items() if entity_id != schema.entity_id and not self._is_internal_entity_id(entity_id)
        ]
        table_names.append((schema.entity_id, self._table_name(schema.entity_id)))
        table_names.extend(
            (f"{schema.entity_id}.{binding.field_name}", self._side_table_name(schema.entity_id, binding.field_name))
            for binding in self._side_table_fields(bindings)
        )
        warnings.extend(self._physical_name_warnings(table_names, kind="SQL table", max_length=max_length, reserved=reserved, case_sensitive=case_sensitive))
        main_field_names = {"object_id", *{binding.field_name for binding in self._main_table_fields(bindings)}}
        column_names = [(field_name, field_name) for field_name in sorted(main_field_names)]
        warnings.extend(
            self._physical_name_warnings(
                column_names,
                kind=f"SQL column for {schema.entity_id!r}",
                max_length=max_length,
                reserved=reserved,
                case_sensitive=case_sensitive,
            )
        )
        return warnings

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure main and side-table SQL storage for an entity.

        Args:
            schema: Entity schema being registered.
            bindings: Storage bindings routed to this backend.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.
            TypeError: If a SideTable binding uses an unsupported logical type.
        """
        try:
            self._ensure_storage(schema, bindings)
        finally:
            # columns or side tables may have changed shape; reusable constructs are rebuilt on demand
            self._forget_constructs(schema.entity_id)

    def _ensure_storage(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure the entity's storage, leaving this backend's cached storage as the database holds it if that fails.

        A cached table gains a column only once the database holds it, so a column
        the database rejects (a required field added to a table that already holds
        rows, for example) never reaches this backend's reads and writes. A failed
        call forgets a main table it cached first and the side-table specs it added,
        so the next registration, such as the workspace re-staging the previous
        one, builds them again over the stored tables.
        """
        self._need_db()
        entity_id = schema.entity_id
        cached = entity_id in self.tables
        side_specs = dict(self._side_specs.get(entity_id, {}))
        try:
            self._ensure_entity_storage(schema, bindings)
        except BaseException:
            if not cached and (table := self.tables.pop(entity_id, None)) is not None:
                self.metadata.remove(table)
            if side_specs:
                self._side_specs[entity_id] = side_specs
            else:
                self._side_specs.pop(entity_id, None)
            raise

    def _ensure_entity_storage(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        main_bindings = self._main_table_fields(bindings)
        side_bindings = self._side_table_fields(bindings)
        main_field_names = {b.field_name for b in main_bindings}

        if schema.entity_id not in self.tables:
            from ...types import Identifier
            from sqlalchemy import Table

            object_id = schema.fields.get("object_id")
            object_id_type = object_id.dtype if object_id is not None else Identifier()
            cols = [self._column("object_id", object_id_type, primary_key=True, nullable=False)]
            for field_schema in schema.fields.values():
                if field_schema.name != "object_id" and field_schema.name in main_field_names:
                    cols.append(self._column(field_schema.name, field_schema.dtype, nullable=not field_schema.required))
            table = Table(
                self._table_name(schema.entity_id),
                self.metadata,
                *cols,
                *self._table_args(order_by=("object_id",)),
                extend_existing=True,
            )
            self.tables[schema.entity_id] = table
            self._create_main_table(table)
            # A table stored by an earlier registration may lack columns this one declares; if adding one fails,
            # the table is forgotten with the rest of this call's cached storage.
            self._sync_columns(table)
            self._add_columns(table, self._side_present_columns(table, side_bindings))
        else:
            table = self.tables[schema.entity_id]
            self._add_columns(table, [*self._missing_columns(table, schema, main_field_names), *self._side_present_columns(table, side_bindings)])
        self._ensure_main_indexes(table, schema, main_field_names)

        for binding in side_bindings:
            field_schema = schema.get(binding.field_name)
            self._side_specs.setdefault(schema.entity_id, {})[field_schema.name] = self._side_spec(schema, field_schema)
        if side_bindings:
            # mark before creating the side tables: a side table that exists records that its field's mark is done,
            # so an interrupted register marks again on the next one
            self._mark_side_fields_absent(table, self._side_fields_new_to_stored_rows(table, self._side_specs.get(schema.entity_id, {})))
            self.db.create_tables([spec.table for spec in self._side_specs.get(schema.entity_id, {}).values()])
            for spec in self._side_specs.get(schema.entity_id, {}).values():
                if self._uses_mysql_id_keys():
                    spec.id_collation = self._mysql_id_collation(spec.table)
                self._ensure_side_indexes(spec)
        self._note_storage_engines([table, *(spec.table for spec in self._side_specs.get(schema.entity_id, {}).values())])

        # Fields this registration no longer binds keep their strategy state until it is published; see ``Backend.prune``.
        for binding in self._adapter_fields(schema, bindings):
            adapter = self._strategy_adapters.get(binding.strategy)
            if adapter is not None:
                adapter.ensure(schema, binding)

    @backend_locked
    def _stage_registration(
        self,
        schema: EntitySchema,
        storage_plan: Any,
        *,
        peers: Sequence[Backend] = (),
        publication: _RegistrationPublication | None = None,
        reattaching: bool = False,
    ) -> AbstractContextManager[None]:
        """Finish storage DDL before opening a sparse-GRAM transaction for publication.

        Which routes share this route's database is decided afterwards, in
        ``_decide_registration_aliases``, once every route has been staged.
        """
        from .strategies import SQLSparseGramAdapter

        # An earlier registration that failed before publication may have left its answers.
        self._staged_row_aliases.pop(schema.entity_id, None)
        self._staged_publication_aliases.pop(schema.entity_id, None)
        adapter = self._strategy_adapters.get(SparseGramIndex.identifier)
        if not isinstance(adapter, SQLSparseGramAdapter):
            return super()._stage_registration(schema, storage_plan)
        with adapter._defer_ensures() as pending:
            self.ensure_plan(schema, storage_plan)
        if not pending or publication is None:
            return nullcontext()
        return self._publish_staged(adapter, schema, storage_plan, peers, pending, publication, reattaching)

    @backend_locked
    def _decide_registration_aliases(self, schema: EntitySchema, storage_plan: Any, peers: Sequence[Backend]) -> None:
        """Decide which routes of the entity share this route's database.

        The workspace calls this for every staged route after all of them have
        run their storage DDL and before it opens any publication transaction.
        Every route then sees the same database files and tables, whichever
        route created them, so the routes of one database agree; and no check
        needs a connection that a publication transaction holds. A failed
        check fails the registration.
        """
        from .strategies import SQLSparseGramAdapter

        routes = {binding.backend for binding in storage_plan.bindings}
        staged = self._staged_row_aliases[schema.entity_id] = self._shared_row_aliases(
            [peer for peer in peers if peer.name in routes], registration=(schema, storage_plan.bindings)
        )
        adapter = self._strategy_adapters.get(SparseGramIndex.identifier)
        registration = (schema, storage_plan.bindings)
        if isinstance(adapter, SQLSparseGramAdapter) and any(adapter.supports(schema, binding) for binding in self._adapter_fields(*registration)):
            found = self._checked_registration_aliases(peers, registration=registration)
            # the routes found through the table made for the row check share the index tables as well; it is not made twice
            found.extend(peer for peer in staged[0] if peer not in found and self._probes_aliases(peer))
            self._staged_publication_aliases[schema.entity_id] = [peer.db for peer in found]

    @contextmanager
    def _publish_staged(
        self,
        adapter: Any,
        schema: EntitySchema,
        storage_plan: Any,
        peers: Sequence[Backend],
        pending: Sequence[tuple[EntitySchema, Any]],
        publication: _RegistrationPublication,
        reattaching: bool,
    ) -> Iterator[None]:
        """Publish the staged sparse-GRAM changes in one transaction with the routes that share this database."""
        with self._lock:
            aliases = self._staged_publication_aliases.pop(schema.entity_id, None)
            if aliases is None:
                # A caller that skipped the decide step; the workspace always runs it.
                aliases = [peer.db for peer in self._checked_registration_aliases(peers, registration=(schema, storage_plan.bindings))]
        with adapter._publish_registration(schema.entity_id, pending, aliases, publication=publication, reattaching=reattaching):
            yield

    def _shared_row_aliases(
        self, peers: Sequence[Backend], *, registration: tuple[EntitySchema, Sequence[Any]] | None = None
    ) -> tuple[list[SQLBackend], list[SQLBackend]]:
        """Return the ``peers`` sharing this route's database, and those among them that write the same main table.

        On PostgreSQL a peer writes the same main table when the entity's table
        name resolves to the same table through its own roles and search path;
        without a registration (a caller that skipped the decide step) its
        roles and search path must match this route's instead.

        Raises:
            RuntimeError: If a server cannot be asked whether the routes share a database.
            ValueError: If PostgreSQL routes write the same main table but resolve different keyword index tables
                while one of them binds a ``SparseGramIndex`` field.
        """
        table = self.tables.get(registration[0].entity_id) if registration is not None else None
        aliases = self._checked_registration_aliases(peers, table=table)
        if self.db.dialect != "postgresql" or not aliases:
            return aliases, aliases
        return aliases, self._checked_registration_aliases(aliases, registration=registration if table is not None else None, compatible=True, table=table)

    def _checked_registration_aliases(
        self,
        peers: Sequence[Backend],
        *,
        registration: tuple[EntitySchema, Sequence[Any]] | None = None,
        compatible: bool = False,
        table: Any | None = None,
    ) -> list[SQLBackend]:
        """Run ``_registration_aliases``, failing the registration with a ``RuntimeError`` when a server cannot be asked.

        Raises:
            RuntimeError: If a server cannot be asked whether the routes share a database.
        """
        from sqlalchemy.exc import SQLAlchemyError

        try:
            return self._registration_aliases(peers, registration=registration, compatible=compatible, table=table)
        except SQLAlchemyError as exc:
            others = [peer for peer in peers if peer is not self and isinstance(peer, SQLBackend)]
            raise self._alias_check_error(others, exc) from exc

    def _registration_aliases(
        self,
        peers: Sequence[Backend],
        *,
        registration: tuple[EntitySchema, Sequence[Any]] | None = None,
        compatible: bool = False,
        table: Any | None = None,
    ) -> list[SQLBackend]:
        """Find routes that open the database holding this route's tables, including its sparse-GRAM index.

        Routes sharing this route's engine always qualify. PostgreSQL and
        MySQL-family routes on other engines are recognized through the server,
        and SQLite and DuckDB routes through the database file. Routes on any
        server through the generic ``sql`` type, and MySQL-family
        routes of a server without named locks, are recognized by a table this
        route creates for the check, given the entity's main ``table`` (see
        ``_probe_registration_aliases``). Routes of the remaining types share
        only an engine.
        """
        spec = self.db.spec
        candidates = [
            peer
            for peer in peers
            if peer is not self and isinstance(peer, SQLBackend) and peer.db is not None and peer.db is not self.db and peer.db.dialect == self.db.dialect
        ]
        if self.db.dialect == "postgresql":
            return self._postgres_registration_aliases(candidates, registration=registration, compatible=compatible, table=table)
        engine = self.db.engine
        aliases = [peer for peer in candidates if peer.db.engine is engine]
        if self._uses_mysql_id_keys():
            return self._mysql_registration_aliases([peer for peer in candidates if peer._uses_mysql_id_keys()], table=table)
        if spec.dialect not in _FILE_DIALECTS:
            others = [peer for peer in candidates if peer.db.engine is not engine and self._probes_aliases(peer)]
            return [*aliases, *self._probe_registration_aliases(others, table)]
        if spec.in_memory or not spec.database:
            return aliases
        path = file_uri_to_path(spec.database)
        for peer in candidates:
            other = peer.db.spec
            if peer.db.engine is self.db.engine or other.in_memory or not other.database:
                continue
            try:
                if samefile(path, file_uri_to_path(other.database)):
                    aliases.append(peer)
            except OSError:
                # The peer may point to a database file that has not been opened yet.
                continue
        return aliases

    def _postgres_registration_aliases(
        self, peers: Sequence[SQLBackend], *, registration: tuple[EntitySchema, Sequence[Any]] | None, compatible: bool, table: Any | None = None
    ) -> list[SQLBackend]:
        """Find PostgreSQL aliases by observing one live transaction's lock.

        A peer that sees the lock is on the same database. It shares this
        route's tables when both sessions use the same roles and search path,
        which does not depend on which HeavenBase tables exist yet. A peer with
        another role or search path is an alias only while both resolve the
        same sparse-GRAM metadata table or, given the entity's main ``table``,
        the same main table. With ``table``, a ``compatible`` peer is one that
        resolves that main table to the same table, whatever its roles and
        search path, so the routes write each row together; such a peer that
        resolves other keyword index tables while either route binds a
        sparse-GRAM field of the entity raises, since one route's writes would
        leave the other's index behind. Without ``table`` a ``compatible`` peer
        uses the same roles and search path, and binding a sparse-GRAM field
        through a peer that does not raises.
        """
        from sqlalchemy import text

        engine = self.db.engine
        aliases = [peer for peer in peers if peer.db.engine is engine]
        others = [peer for peer in peers if peer.db.engine is not engine]
        if not others:
            return aliases

        # PostgreSQL advisory locks are visible across sessions in the database.
        # A fresh two-integer key identifies this transaction even when routes
        # use different addresses, pools, or connection options.
        nonce = uuid4().int
        key = {"hi": (nonce >> 96) & 0x7FFFFFFF, "lo": nonce & 0x7FFFFFFF}
        quoted = engine.dialect.identifier_preparer.format_table(table) if table is not None else None
        key["main"] = quoted
        session = "to_regclass('hb_gram_meta')::oid, current_user, session_user, current_schemas(true), to_regclass(CAST(:main AS text))::oid"
        visible = text(
            f"SELECT {session}, EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND granted "
            "AND database = (SELECT oid FROM pg_database WHERE datname = current_database()) "
            "AND classid = :hi AND objid = :lo AND objsubid = 2)"
        )
        with engine.connect().execution_options(isolation_level="READ COMMITTED") as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(:hi, :lo)"), key)
            owner = connection.execute(text(f"SELECT {session}"), key).one()
            for peer in others:
                with peer.db.engine.connect() as other:
                    *settings, same_database = other.execute(visible, key).one()
                # Matching roles and search path resolve every unqualified table alike, whether or not it exists yet.
                same_session = tuple(settings[1:4]) == tuple(owner[1:4])
                same_table = owner[4] is not None and settings[4] == owner[4]
                if not same_database or not (same_session or same_table or (owner[0] is not None and settings[0] == owner[0])):
                    continue
                if compatible and table is not None:
                    if not same_table:
                        continue
                    gram = registration is not None and any(
                        binding.strategy == SparseGramIndex.identifier for route in (self, peer) for binding in route._adapter_fields(*registration)
                    )
                    # index tables that do not exist yet are created where each search path puts them
                    if gram and (settings[0] != owner[0] or (owner[0] is None and settings[3] != owner[3])):
                        raise ValueError(
                            f"PostgreSQL routes {self.name!r} and {peer.name!r} write the same table of entity {registration[0].entity_id!r} but resolve "
                            "different sparse-GRAM index tables through their roles or search paths; give them the same search path"
                        )
                    aliases.append(peer)
                elif (registration is None and not compatible) or same_session:
                    aliases.append(peer)
                elif registration is not None and any(binding.strategy == SparseGramIndex.identifier for binding in peer._adapter_fields(*registration)):
                    raise ValueError(f"PostgreSQL routes {self.name!r} and {peer.name!r} share the sparse-GRAM index but use different roles or search paths")
        return aliases

    def _probes_aliases(self, peer: SQLBackend) -> bool:
        """Return whether this route and ``peer`` recognize a shared table through a table made for the check."""
        return {self.type, peer.type} <= _PROBED_TYPES and self.db.dialect == peer.db.dialect and self.db.dialect not in _FILE_DIALECTS | {"postgresql"}

    def _probe_registration_aliases(self, peers: Sequence[SQLBackend], table: Any | None) -> list[SQLBackend]:
        """Find the ``peers`` whose main table of the entity is this route's ``table``, through a table made for the check.

        Every route has run its storage DDL by now, so a shared table holds the
        columns of both routes; a peer whose columns this route's table lacks
        stores its fields elsewhere, and nothing is created for it. For the
        others this route creates a table under a fresh name, as it creates its
        entity tables, and each peer looks that name up in its own default
        schema, where it creates and reads its own tables. The table is dropped
        again before this returns.

        A server that refuses to create the table, such as for a login without
        that right, gives no answer, and no peer is recognized. A server that
        refuses to drop it keeps the table, which a ``RuntimeWarning`` names.

        Raises:
            RuntimeError: If a server cannot be reached for the check, so the registration fails instead of guessing.
        """
        if table is None:
            return []
        import warnings

        from sqlalchemy import Column, Integer, MetaData, Table, inspect
        from sqlalchemy.exc import DBAPIError

        stored = {str(name) for name in self.db.columns(table.name, schema=table.schema)}
        peers = [peer for peer in peers for other in (peer.tables.values()) if other.key == table.key and set(other.c.keys()) <= stored]
        if not peers:
            return []
        probe = Table(f"hb_probe_{uuid4().hex[:12]}", MetaData(), Column("object_id", Integer(), primary_key=True))
        try:
            with self.db.engine.connect() as connection:
                try:
                    probe.create(connection, checkfirst=False)
                    connection.commit()
                except DBAPIError as exc:
                    if exc.connection_invalidated:
                        raise
                    connection.rollback()
                    return []
                try:
                    return [peer for peer in peers if inspect(peer.db.engine).has_table(probe.name)]
                finally:
                    try:
                        probe.drop(connection, checkfirst=False)
                        connection.commit()
                    except DBAPIError as exc:
                        if exc.connection_invalidated:
                            raise
                        connection.rollback()
                        warnings.warn(
                            f"SQL route {self.name!r} made table {probe.name!r} to see which routes share its tables, and the server "
                            f"refused to drop it again; drop it by hand: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
        except Exception as exc:
            raise self._alias_check_error(peers, exc) from exc

    def _mysql_registration_aliases(self, peers: Sequence[SQLBackend], *, table: Any | None = None) -> list[SQLBackend]:
        """Find MySQL routes on the same server and current database by observing one session's named lock.

        A server without the named-lock functions answers through the table
        check of ``_probe_registration_aliases`` instead.

        Raises:
            RuntimeError: If a server cannot be asked, so the registration fails instead of guessing.
        """
        from sqlalchemy import text
        from sqlalchemy.exc import DBAPIError, SQLAlchemyError

        engine = self.db.engine
        aliases = [peer for peer in peers if peer.db.engine is engine]
        others = [peer for peer in peers if peer.db.engine is not engine]
        if not others:
            return aliases
        # Named locks are visible to every session on one server; a fresh name identifies this session.
        lock = {"name": f"heavenbase-route-{uuid4().hex}"}
        try:
            with engine.connect() as connection:
                if connection.execute(text("SELECT GET_LOCK(:name, 0)"), lock).scalar() != 1:
                    raise RuntimeError("the server did not grant a fresh named lock")
                try:
                    database = connection.execute(text("SELECT DATABASE()")).scalar()
                    for peer in others:
                        with peer.db.engine.connect() as other:
                            held, current = other.execute(text("SELECT IS_USED_LOCK(:name) IS NOT NULL, DATABASE()"), lock).one()
                        if held and database is not None and current == database:
                            aliases.append(peer)
                finally:
                    connection.execute(text("SELECT RELEASE_LOCK(:name)"), lock)
        except DBAPIError as exc:
            if _mysql_error_code(exc) == _MYSQL_UNKNOWN_FUNCTION:
                return [*(peer for peer in peers if peer.db.engine is engine), *self._probe_registration_aliases(others, table)]
            raise self._alias_check_error(others, exc) from exc
        except (SQLAlchemyError, RuntimeError) as exc:
            raise self._alias_check_error(others, exc) from exc
        return aliases

    def _alias_check_error(self, peers: Sequence[SQLBackend], exc: BaseException) -> RuntimeError:
        names = ", ".join(repr(peer.name) for peer in peers)
        return RuntimeError(
            f"Cannot tell whether SQL route {self.name!r} shares its database with {names}: {type(exc).__name__}: {exc}. "
            "Routes that share a database must write each row together, so the registration stops; register the entity again "
            "once the servers answer."
        )

    @backend_locked
    def _prepare_registration_scan(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Open stored SQL routes without reconciling scalar or sparse-GRAM indexes."""
        from .strategies import SQLSparseGramAdapter

        adapter = self._strategy_adapters.get(SparseGramIndex.identifier)
        with adapter._defer_ensures() if isinstance(adapter, SQLSparseGramAdapter) else nullcontext(), self._scalar_indexes_from_plan():
            super()._prepare_registration_scan(schema, storage_plan)

    @backend_locked
    def _validate_registration_reattachment(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Validate SQL sparse-GRAM reattachment without preparing the proposed schema."""
        from .strategies import SQLSparseGramAdapter

        adapter = self._strategy_adapters.get(SparseGramIndex.identifier)
        if isinstance(adapter, SQLSparseGramAdapter):
            for binding in self._adapter_fields(schema, storage_plan.bindings):
                if binding.strategy == SparseGramIndex.identifier:
                    adapter._validate_reattachment(schema.entity_id, binding)

    @backend_locked
    def _stored_scalar_index_exists(self, entity_id: str, fields: Sequence[str], *, unique: bool) -> bool | None:
        """Reflect live index state when a legacy MetaSchema projection is ambiguous."""
        from sqlalchemy.exc import SQLAlchemyError

        from ...database.execution import DatabaseError

        self._need_db()
        table = self.tables.get(entity_id)
        try:
            indexes = self.db._reflected_indexes(self._table_name(entity_id), table.schema if table is not None else self.metadata.schema)
        except (SQLAlchemyError, NotImplementedError, DatabaseError):
            return None
        return any(tuple(index.get("column_names") or ()) == tuple(fields) and bool(index.get("unique")) == unique for index in indexes)

    @backend_locked
    def _prune_registration(self, schema: EntitySchema, bindings: Sequence[Any], peers: Sequence[Backend], *, reattaching: bool) -> None:
        from .strategies import SQLSparseGramAdapter

        adapter = self._strategy_adapters.get(SparseGramIndex.identifier)
        staged = self._staged_row_aliases.pop(schema.entity_id, None)
        detected, compatible = staged if staged is not None else self._shared_row_aliases(peers)
        aliases = {peer.name: peer for peer in detected}
        self._set_shared_row_storage(schema, bindings, compatible)
        if not isinstance(adapter, SQLSparseGramAdapter) or (not aliases and not reattaching):
            super()._prune_registration(schema, bindings, peers, reattaching=reattaching)
            return
        kept = [binding for binding in bindings if adapter.supports(schema, binding)]
        protected = set()
        for binding in bindings:
            peer = aliases.get(binding.backend)
            if peer is not None:
                other = peer._strategy_adapters.get(SparseGramIndex.identifier)
                if isinstance(other, SQLSparseGramAdapter) and other.supports(schema, binding):
                    protected.add(binding.field_name)
        adapter.prune(schema, kept, protected=protected, retire_stored=not reattaching)
        if not reattaching:
            for other in self._strategy_adapters.values():
                if other is not adapter:
                    other.prune(schema, [binding for binding in bindings if other.supports(schema, binding)])

    def _set_shared_row_storage(self, schema: EntitySchema, bindings: Sequence[Any], aliases: Sequence[SQLBackend]) -> None:
        """Prepare one complete physical row for the routes that store fields of this entity in the same table.

        ``aliases`` are the routes found during staging to share this route's
        database with compatible sessions. Each route alone would replace the
        whole row with only its own columns, so the first route by name writes
        every route's fragment of the row. Each field is written the way the
        route bound to it stores it: into that route's column, with its column
        type, and into that route's side table. A binding's strategy alone does
        not decide this, because a route may keep a ``SideTable`` field inline
        (pgvector does).
        """
        from sqlalchemy import Column, Integer, MetaData, Table
        from .strategies import SQLSparseGramAdapter

        entity_id = schema.entity_id
        if self._shared_row_storage.pop(entity_id, None) is not None:
            self._forget_constructs(entity_id)
        if not aliases:
            return
        source = self.tables[entity_id]
        peers = [
            peer
            for peer in aliases
            if (table := peer.tables.get(entity_id)) is not None
            and table.key == source.key
            and isinstance(peer._strategy_adapters.get(SparseGramIndex.identifier), SQLSparseGramAdapter)
        ]
        if not peers or not isinstance(self._strategy_adapters.get(SparseGramIndex.identifier), SQLSparseGramAdapter):
            return
        routes = tuple(sorted([self.name, *(peer.name for peer in peers)]))
        if self.name != routes[0]:
            self._shared_row_storage[entity_id] = _SharedRowStorage(routes, None, {}, ())
            return
        owners = {route.name: route for route in (self, *peers)}
        columns = {"object_id": source.c.object_id._copy()}
        side_specs: dict[str, _SideTableSpec] = {}
        for binding in bindings:
            route = owners.get(binding.backend)
            if route is None:
                continue
            stored = route.tables[entity_id].c
            spec = route._side_specs.get(entity_id, {}).get(binding.field_name)
            if spec is not None:
                side_specs.setdefault(binding.field_name, spec)
                marker = _side_present_column(binding.field_name)
                if marker not in columns:
                    columns[marker] = stored[marker]._copy() if marker in stored else Column(marker, Integer(), nullable=True)
            if binding.field_name in stored and binding.field_name not in columns:
                columns[binding.field_name] = stored[binding.field_name]._copy()
        table = Table(source.name, MetaData(), *columns.values(), schema=source.schema)
        gram_aliases = tuple(peer._strategy_adapters[SparseGramIndex.identifier] for peer in peers)
        self._shared_row_storage[entity_id] = _SharedRowStorage(routes, table, side_specs, gram_aliases)
        if any(adapter._fields.get(entity_id) for adapter in gram_aliases):
            # This route writes the peers' keyword rows inside its row-write transaction, where creating or checking
            # the index tables would need a second connection, which a one-connection pool cannot give; ensure them now.
            # The entry is recorded first, so the routes still agree on who writes the row if this fails once; the
            # next write ensures the tables again before it opens its transaction (see ``_apply_row_ops``).
            self._strategy_adapters[SparseGramIndex.identifier]._ensure_index()

    def _row_write_routes(self, entity_id: str) -> tuple[str, ...]:
        storage = self._shared_row_storage.get(entity_id)
        return storage.routes if storage is not None else super()._row_write_routes(entity_id)

    def _side_spec(self, schema: EntitySchema, field_schema: Any) -> _SideTableSpec:
        from ...types import Identifier
        from sqlalchemy import Column, Integer, Table

        st_name = self._side_table_name(schema.entity_id, field_schema.name)
        projection = sql_dialect_spec(self.db.dialect, resolver=self.module_resolver)
        table = self.metadata.tables.get(st_name)
        if table is not None:
            return _SideTableSpec(field_schema, table, projection)
        # A side row references its main row, so its object_id column takes the entity's own object_id domain.
        object_id = schema.fields.get("object_id")
        cols = [
            self._column("_id", Identifier(), primary_key=True, nullable=False),
            self._column("object_id", object_id.dtype if object_id is not None else Identifier(), nullable=False),
        ]
        if type(field_schema.dtype).identifier == "array":
            item_dtype = getattr(field_schema.dtype, "item_type", None)
            cols.extend(
                [
                    Column("idx", Integer(), nullable=False),
                    self._column("item", item_dtype if item_dtype is not None else field_schema.dtype, nullable=False),
                ]
            )
        elif type(field_schema.dtype).identifier == "hyper-g":
            cols.append(Column("idx", Integer(), nullable=False))
            cols.extend([self._column(field.name, field.dtype, nullable=not field.required) for field in field_schema.dtype.fields])
        else:
            raise TypeError(f"SideTable does not support {type(field_schema.dtype).identifier}")
        table = Table(
            st_name,
            self.metadata,
            *cols,
            *self._table_args(order_by=("object_id", "idx")),
            extend_existing=True,
        )
        return _SideTableSpec(field_schema, table, projection)

    def _add_index(self, table: Any, *cols: Any, unique: bool = False) -> str:
        from sqlalchemy import Index

        col_names = _norm_cols(cols)
        role = ("unique",) if unique else ()
        name = sql_index_name(table.name, *role, *[col.name for col in cols])
        if not col_names or all(getattr(col, "primary_key", False) for col in cols):
            return name
        declared_names = {str(index.name) for index in table.indexes}
        reflected = self.db.table_indexes(table)
        reflected_names = {str(item.get("name") or "") for item in reflected}
        if name in declared_names or name in reflected_names:
            return name
        external_covering = [
            item for item in reflected if _norm_cols(item.get("column_names") or []) == col_names and not str(item.get("name") or "").startswith("ix_")
        ]
        if unique:
            if any(item.get("unique") for item in external_covering):
                return name
        elif external_covering:
            return name
        options = sql_dialect_spec(self.type, resolver=self.module_resolver).unique_index_options(tuple(cols)) if unique else {}
        Index(name, *cols, unique=unique, **options)
        return name

    def _server_index_column_sets(self, table: Any) -> dict[tuple[str, ...], bool]:
        return self.db.index_column_sets(table)

    def _create_indexes(self, table: Any, *, ignoring: set[str] | frozenset[str] = frozenset()) -> None:
        """Create declared indexes despite incomplete provider reflection.

        An index named in ``ignoring`` is about to be dropped, so it does not count as covering a declared one.
        """
        from sqlalchemy.exc import SQLAlchemyError

        if not table.indexes:
            return
        existing = self.db.index_column_sets(table, ignoring=ignoring) if ignoring else self.db.index_column_sets(table)

        def satisfied(index: Any) -> bool:
            columns = _norm_cols(index.columns)
            return existing.get(columns, False) if index.unique else columns in existing

        pending = sorted((index for index in table.indexes if str(index.name) not in ignoring), key=lambda item: str(item.name))
        if not all(satisfied(index) for index in pending) and (covered := _duckdb_covered_columns(self.db, table)) is not None:
            # reads scan for an index that is left out; a unique one was refused before any DDL (see ``ensure_plan``)
            for index in pending:
                if not satisfied(index) and not covered.issuperset(_norm_cols(index.columns)):
                    table.indexes.discard(index)
            pending = [index for index in pending if index in table.indexes]
        for position, index in enumerate(pending):
            if satisfied(index):
                continue
            try:
                self.db.create_index(index, checkfirst=True)
            except BaseException as exc:
                if not (isinstance(exc, SQLAlchemyError) and _index_already_exists_error(exc)):
                    # An index stays declared on the cached table only once the database holds it, as a column
                    # does: a unique index that stored duplicates reject, for example, is forgotten with the ones
                    # not built yet, so the previous registration can be established again.
                    for left in pending[position:]:
                        if not satisfied(left):
                            table.indexes.discard(left)
                    raise
            existing[_norm_cols(index.columns)] = bool(index.unique)

    def _ensure_main_indexes(self, table: Any, schema: EntitySchema, field_names: set[str]) -> None:
        """Create the default scalar indexes of a plain ``ensure``; under a resolved plan ``ensure_indexes`` creates them instead."""
        if "control-plane" in schema.tags or self._plan_owns_scalar_indexes:
            return
        if not self.scalar_index().value:
            return
        for field_schema in schema.fields.values():
            if (
                field_schema.name != "object_id"
                and field_schema.name in field_names
                and type(field_schema.dtype).identifier in ("identifier", "short-text", "integer", "float")
            ):
                self._add_index(table, table.c[field_schema.name])
        self._create_indexes(table)

    @backend_locked
    def ensure_plan(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Ensure an entity's storage and the scalar indexes its resolved plan asks for.

        The plan alone decides the main table's scalar indexes, so ``ensure``
        creates no default index here: a field that opts out of its default
        index (``.index(enabled=False)``) is never indexed and then dropped
        again on a repeat registration.

        Args:
            schema: Entity schema to create or refresh in physical storage.
            storage_plan: Resolved placement, index, and constraint plan.

        Returns:
            None: This function does not return a value.
        """
        self._refuse_new_duckdb_constraints(schema, storage_plan)
        with self._scalar_indexes_from_plan():
            self.ensure(schema, storage_plan.bindings)
        self.ensure_indexes(schema, storage_plan)

    def _refuse_new_duckdb_constraints(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Refuse, before any DDL, a unique constraint with a column that no index of a DuckDB table with rows covers.

        Its unique index cannot be created (see ``_duckdb_covered_columns``), and
        without it the constraint would not be enforced.

        Raises:
            ValueError: If the plan asks for such a constraint on this route.
        """
        constraints = storage_plan.backend_constraints(self.name)
        if not constraints or getattr(self.db, "dialect", None) != "duckdb":
            return
        table = sa.Table(self._table_name(schema.entity_id), sa.MetaData())
        if (covered := _duckdb_covered_columns(self.db, table)) is None:
            return
        try:
            self.db._reflected_indexes(table.name, table.schema)
        except Exception:
            return  # the catalog cannot say which constraints the table has, so none is refused and none is created
        existing = self.db.index_column_sets(table)
        for constraint in constraints:
            columns = _norm_cols(constraint.fields)
            if not existing.get(columns, False) and not covered.issuperset(columns):
                fields = ", ".join(constraint.fields)
                raise ValueError(
                    f"Cannot add a unique constraint on {schema.entity_id}({fields}) on DuckDB backend {self.name!r}: the stored table has "
                    "held rows and no index on one of these columns, and DuckDB cannot write a table again in a later process after such "
                    "a column was indexed. Declare the constraint before the first row is stored, or store the rows in a new database "
                    "that was registered with it"
                )

    @contextmanager
    def _scalar_indexes_from_plan(self) -> Iterator[None]:
        """Leave the main table's scalar indexes to the caller's plan while the ensures inside run."""
        previous = self._plan_owns_scalar_indexes
        self._plan_owns_scalar_indexes = True
        try:
            yield
        finally:
            self._plan_owns_scalar_indexes = previous

    @backend_locked
    def ensure_indexes(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Create and reconcile resolved main-table scalar indexes.

        Args:
            schema: Entity schema owning the SQL table.
            storage_plan: Resolved storage, index, and constraint plan.

        Returns:
            None: The server matches the resolved Heaven-owned index set.

        Note:
            The table can also hold the indexes of other routes and workspaces on the same database.
            An index the plan asks for on any route stays. So does an index on a column that this route's
            table does not declare, such as one of a field that only another definition of the entity has.
            On DuckDB no index is dropped: DuckDB 1.5.3 cannot replay a write-ahead log that drops indexes
            and then writes rows, so a process that ends without a checkpoint would leave a file that no
            process can open. A DuckDB table that has held rows gets no index with a column that none of
            its indexes covers, for a like reason (see ``_duckdb_covered_columns``).
        """
        if not self.scalar_index().value or schema.entity_id not in self.tables:
            return
        table = self.tables[schema.entity_id]
        desired: set[str] = set()
        for binding in storage_plan.backend_indexes(self.name):
            desired.add(self._add_index(table, *[table.c[name] for name in binding.fields]))
        for binding in storage_plan.backend_constraints(self.name):
            desired.add(self._add_index(table, *[table.c[name] for name in binding.fields], unique=True))
        kept = desired.union(
            (sql_index_name(table.name, *binding.fields) for binding in storage_plan.indexes),
            (sql_index_name(table.name, "unique", *binding.fields) for binding in storage_plan.constraints),
        )
        declared = {str(name).lower() for name in table.c.keys()}
        retiring = [
            reflected
            for reflected in (() if getattr(self.db, "dialect", None) == "duckdb" else self.db.table_indexes(table))
            if self._owns_index(table, reflected)
            and str(reflected["name"]) not in kept
            and all(str(column).lower() in declared for column in reflected["column_names"])
        ]
        # An index about to be dropped covers nothing the plan asks for: a field whose unique constraint is removed
        # gets its plain index now, before the unique one goes.
        self._create_indexes(table, ignoring={str(reflected["name"]) for reflected in retiring})
        for reflected in retiring:
            name = str(reflected["name"])
            self.db.drop_index(table, name, reflected.get("column_names") or [])
            for index in list(table.indexes):
                if index.name == name:
                    table.indexes.discard(index)

    @staticmethod
    def _owns_index(table: Any, reflected: dict[str, Any]) -> bool:
        """Return whether a reflected index carries the name HeavenBase gives an index on its columns.

        Only such an index is HeavenBase's to drop; an index someone else created, even one named like
        ``ix_<hash>``, is left alone. Column names are matched to the table's ignoring case, as some
        dialects reflect them in another case.
        """
        name, columns = str(reflected.get("name") or ""), [str(column) for column in reflected.get("column_names") or []]
        if not name.startswith("ix_") or not columns:
            return False
        stored = {column.name.lower(): column.name for column in table.c}
        role = ("unique",) if reflected.get("unique") else ()
        return name == sql_index_name(table.name, *role, *(stored.get(column.lower(), column) for column in columns))

    def _ensure_side_indexes(self, spec: _SideTableSpec) -> None:
        self._add_index(spec.table, spec.table.c.object_id)
        if spec.dtype_identifier == "array":
            self._add_index(spec.table, spec.table.c.object_id, spec.table.c.idx)
            self._add_index(spec.table, spec.table.c.item, spec.table.c.object_id)
        if spec.dtype_identifier == "hyper-g":
            self._add_index(spec.table, spec.table.c.object_id, spec.table.c.idx)
            fields = [field for field in spec.field_schema.dtype.filterable_fields if type(field.dtype).identifier in _HYPERG_INDEX_TYPES]
            for field in fields:
                self._add_index(spec.table, spec.table.c[field.name], spec.table.c.object_id)
            if fields:
                self._add_index(spec.table, *[spec.table.c[field.name] for field in fields[:3]], spec.table.c.object_id)
        self._create_indexes(spec.table)

    def _missing_columns(self, table: Any, schema: EntitySchema, allowed_field_names: set[str]) -> list[Callable[[], Any]]:
        """Return a builder for each main-table field column that the cached ``table`` lacks."""
        return [
            partial(self._column, field_schema.name, field_schema.dtype, nullable=not field_schema.required)
            for field_schema in schema.fields.values()
            if field_schema.name != "object_id" and field_schema.name in allowed_field_names and field_schema.name not in table.c
        ]

    def _side_present_columns(self, table: Any, side_bindings: Sequence[Any]) -> list[Callable[[], Any]]:
        """Return a builder for each side-routed field's presence marker that the cached main ``table`` lacks.

        Args:
            table (Any): Main SQLAlchemy table for the entity.
            side_bindings (Sequence[Any]): Bindings routed to a side table.

        Returns:
            list[Callable[[], Any]]: One builder per missing marker column.
        """
        from sqlalchemy import Column, Integer

        names = dict.fromkeys(_side_present_column(binding.field_name) for binding in side_bindings)
        return [partial(Column, name, Integer(), nullable=True) for name in names if name not in table.c]

    def _add_columns(self, table: Any, columns: Sequence[Callable[[], Any]]) -> None:
        """Add columns to the stored main table, each to the cached ``table`` only once the database holds it.

        Args:
            table (Any): Cached main SQLAlchemy table for the entity.
            columns (Sequence[Callable[[], Any]]): Builders of the columns to add, in order.

        Returns:
            None.
        """
        from sqlalchemy import MetaData

        existing = set(self.db.columns(table.name, schema=table.schema)) if columns else set()
        for build in columns:
            column = build()
            if column.name not in existing:
                # The DDL compiles a column bound to a copy of the table, as some dialects (MSSQL) require.
                table.to_metadata(MetaData()).append_column(column)
                self._add_stored_column(table, column)
                column = build()
            table.append_column(column)

    def _sync_columns(self, table: Any) -> None:
        existing = set(self.db.columns(table.name, schema=table.schema))
        missing = [column for column in table.c if column.name not in existing]
        for column in missing:
            self._add_stored_column(table, column)

    def _side_fields_new_to_stored_rows(self, table: Any, side_specs: Dict[str, _SideTableSpec]) -> list[str]:
        """Return side fields whose side table does not exist yet while the main table already holds rows.

        Those rows were stored before the field existed: they hold no value for it,
        and a row without a presence marker would read an empty list. A side table
        that already exists holds the items of the rows written beside it, including
        rows written before markers existed, which keep reading those items. The
        decision depends only on what is stored, so a register interrupted after
        adding a marker column decides the same way the next time.

        Args:
            table (Any): Main SQLAlchemy table for the entity.
            side_specs (Dict[str, _SideTableSpec]): Side-table specs keyed by field name.

        Returns:
            list[str]: Field names whose stored rows must be marked absent.
        """
        from sqlalchemy import select

        if not side_specs:
            return []
        existing = {name.casefold() for name in self.db.tables(schema=table.schema)}
        fields = [name for name, spec in side_specs.items() if spec.table.name.casefold() not in existing]
        if not fields or not self.db.orm_execute(select(table.c.object_id).limit(1), readonly=True).to_dicts():
            return []
        return fields

    def _mark_side_fields_absent(self, table: Any, field_names: Sequence[str]) -> None:
        """Mark stored rows as holding no value for side fields that did not exist when they were written.

        Without the mark such a row would read an empty list, while a row written
        later without the field reads ``None``, as a main-table field added later does.

        Args:
            table (Any): Main SQLAlchemy table for the entity.
            field_names (Sequence[str]): Fields returned by `_side_fields_new_to_stored_rows`.

        Returns:
            None.
        """
        if not field_names:
            return
        from sqlalchemy import update

        with self.db.tx():
            for name in field_names:
                marker = table.c[_side_present_column(name)]
                self.db.orm_execute(update(table).where(marker.is_(None)).values({marker.name: 0}), readonly=False)

    @contextmanager
    def _row_write_tx(self, *, maintain_strategies: bool) -> Iterator[None]:
        """Keep a row batch and its strategy changes in one write scope."""
        db = self.db
        if db.in_transaction():
            if db.dialect == "sqlite" and not self._autocommit_engine:
                with self._sqlite_row_savepoint():
                    yield
                return
            try:
                yield
            except BaseException:
                # Other transactional providers may lack savepoints. Discard
                # the enclosing work before control returns to its caller.
                if self._upserts_are_atomic():
                    db.rollback()
                raise
            return
        with db.tx():
            if not (maintain_strategies and db.dialect == "sqlite" and self._autocommit_engine):
                yield
                return
            connection = db._conn
            driver = connection.connection.driver_connection
            if driver.in_transaction:
                yield
                return
            # SQLAlchemy's transaction does not issue BEGIN in SQLite autocommit
            # mode, and sqlite3 with autocommit=True also needs explicit COMMIT.
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield
                connection.exec_driver_sql("COMMIT")
            except BaseException:
                if driver.in_transaction:
                    connection.exec_driver_sql("ROLLBACK")
                raise

    @contextmanager
    def _sqlite_row_savepoint(self) -> Iterator[None]:
        """Isolate a row batch while SQLite keeps the caller's transaction open."""
        db = self.db
        connection = db._conn
        driver = connection.connection.driver_connection
        if not driver.in_transaction:
            # Legacy sqlite3 defers BEGIN until DML. Otherwise releasing the
            # first savepoint would commit instead of joining the outer scope.
            connection.exec_driver_sql("BEGIN")
        dirty, failed = db._tx_dirty, db._tx_failed
        savepoint = connection.begin_nested()
        closed = False
        try:
            try:
                yield
            except BaseException:
                if driver.in_transaction:
                    savepoint.rollback()
                    db._tx_dirty, db._tx_failed = dirty, failed
                else:
                    # SQLite can roll back the outer transaction and remove
                    # every savepoint. Reset SQLAlchemy's transaction state.
                    db.rollback()
                    db._tx_failed = True
                closed = True
                raise
            savepoint.commit()
            closed = True
        finally:
            if not closed:
                db.rollback()

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply row mutations to SQL main tables and side tables.

        Args:
            rows: Backend row mutations. Upserts replace main-table row fragments and
                side-table rows; deletes remove both. Strategy adapters see only the
                mutations whose ``strategies`` flag is true.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.
        """
        self._apply_row_ops(rows)

    @backend_locked
    def _write_workspace_rows(self, rows: Sequence[RowOp]) -> None:
        shared = {
            entity_id: storage
            for entity_id in {row.entity_id for row in rows}
            if (storage := self._shared_row_storage.get(entity_id)) is not None and storage.table is not None
        }
        if shared:
            self._apply_row_ops(rows, shared=shared)
        else:
            self.upsert(rows)

    @backend_locked
    def upsert_owned(self, rows: Sequence[RowOp], *, owner_field: str, owner: str) -> list[RowOp]:
        """Apply ownership-guarded mutations in the same transaction as strategy maintenance."""
        atomic = self.tag("atomic-upsert")
        # a table without transactions withdraws the claim but leaves each guarded statement as it was
        if (atomic is None or atomic.value is not True) and not self._nontransactional_tables:
            return super().upsert_owned(rows, owner_field=owner_field, owner=owner)
        return self._apply_row_ops(rows, ownership=(owner_field, owner))

    def _apply_row_ops(
        self, rows: Sequence[RowOp], *, ownership: tuple[str, str] | None = None, shared: dict[str, _SharedRowStorage] | None = None
    ) -> list[RowOp]:
        """Apply row operations with the backend lock held."""
        self._need_db()
        self._pool_slots(self.db)  # this lock holder's connection stays free of the reads running outside the lock
        from sqlalchemy import bindparam, delete, insert

        mysql_ids = self._uses_mysql_id_keys()
        shared = shared or {}
        if self.type == "sql" and self.db.dialect not in _EXACT_ID_DIALECTS:
            rows = self._exact_id_ops(rows, shared)
        maintained = [row for row in rows if row.strategies]
        adapters = tuple(self._strategy_adapters.values()) if maintained else ()
        gram = self._strategy_adapters.get(SparseGramIndex.identifier)
        gram_aliases = {entity_id: storage.gram_aliases for entity_id, storage in shared.items()}
        # entities whose guarded row statement found the stored keyword-index state the write expected
        confirmed: set[str] = set()
        actions = [
            (
                (partial(adapter.validate_upsert, aliases=gram_aliases), partial(adapter.upsert, aliases=gram_aliases))
                if gram_aliases and adapter is gram
                else (adapter.validate_upsert, adapter.upsert)
            )
            for adapter in adapters
        ]
        if gram in adapters and getattr(gram, "write_guards", None) is not None:
            actions = [
                (partial(validate, confirmed=confirmed), upsert) if adapter is gram else (validate, upsert)
                for adapter, (validate, upsert) in zip(adapters, actions)
            ]
        validate_first = bool(actions) and not self._upserts_are_atomic()
        guards: dict[str, Any] = {}
        if maintained and gram in adapters and ownership is None and getattr(gram, "guards_writes", False):
            # The first row statement of each entity checks the keyword-index state the write expects, so a write
            # whose state is unchanged sends no query of its own for the check (see ``SQLSparseGramAdapter.write_guards``).
            guards = gram.write_guards(maintained, aliases=gram_aliases)
            if validate_first:
                # Where each statement may commit on its own, only the call's first statement carries the check, so a
                # refusal still comes before any row is written; a miss is checked at once.
                guards = {entity_id: guard for entity_id, guard in guards.items() if entity_id == rows[0].entity_id}

        def confirm(entity_id: str, guard: Any) -> bool:
            """Check the write again after its guarded statement did not match; return whether the state changed."""
            confirmed.add(entity_id)
            return gram.confirm_write(entity_id, maintained, aliases=gram_aliases, expected=guard.state) != guard.state

        if actions and not self.db.in_transaction():
            if any(alias._fields.get(entity_id) for entity_id, aliases in gram_aliases.items() for alias in aliases):
                # This route writes its peers' keyword rows inside the row-write transaction, where creating or
                # checking the index tables would need a second connection, which a one-connection pool cannot give.
                # Registration ensured them; if that failed once, ensure them now, before the transaction opens.
                gram._ensure_index()
        applied = []
        old_duckdb = _duckdb_release_without_row_replacement(self.db)
        written: set[tuple[str, str]] = set()
        removed: set[tuple[str, str]] = set()
        # a transaction the caller opened may have taken its snapshot before this call
        outer = self.db.in_transaction()
        with self._row_write_tx(maintain_strategies=bool(actions)):
            if gram is not None and self.db.dialect == "duckdb":
                # DuckDB has no lock statement, so a process-wide gate holds off a change of each entity's keyword
                # index while this transaction writes it (see ``_SQLGramIndex.lock_entity_rows``)
                index = gram._need_index()
                for entity_id in sorted({row.entity_id for row in rows}):
                    storage = shared.get(entity_id)
                    index.hold_entity_writes(storage.table if storage is not None else self.tables.get(entity_id), what=f"entity {entity_id!r}")
            if validate_first:
                for adapter, (validate, _) in zip(adapters, actions):
                    if adapter is gram and guards:
                        # a guarded entity is checked by its first row statement below
                        gram.validate_upsert(maintained, aliases=gram_aliases, confirmed=set(guards))
                    else:
                        validate(maintained)

            for entity_id, is_delete, batch in self.ordered_row_op_batches(rows):
                storage = shared.get(entity_id)
                table = storage.table if storage is not None else self.tables[entity_id]
                shape = "shared_" if storage is not None else ""
                if old_duckdb is not None:
                    self._refuse_old_duckdb_replacement(old_duckdb, entity_id, table, batch, is_delete, ownership, written, removed)
                    if not is_delete:
                        written.update((entity_id, str(row.object_id)) for row in batch)
                if ownership is None:
                    delete_ids = self._construct(entity_id, shape + "delete_ids", lambda: delete(table).where(_object_ids_match(table.c.object_id, mysql_ids)))
                    # MySQL binds each ID twice for the index lookup and exact comparison.
                    batch_size = self._id_batch_size(self.db, self._id_list_limit(self.db) // 2 if mysql_ids else 0)
                    guard = guards.get(entity_id)
                    if guard is not None and guard.kind == "tables" and outer and self.db.dialect == "mysql" and entity_id not in confirmed:
                        # information_schema shows a MySQL REPEATABLE READ transaction its snapshot, so the condition
                        # cannot see index tables created after it; a read of them can
                        gram._need_index().refuse_tables_after_snapshot(f"a write of {entity_id!r}")
                    # IDs whose guarded DELETE may have been held back by a changed keyword-index state
                    held: list[Any] = []
                    for ids in _batches([row.object_id for row in batch], batch_size):
                        params = {"object_ids": list(ids)}
                        if mysql_ids:
                            params["exact_ids"] = params["object_ids"]
                        if guard is None or entity_id in confirmed:
                            self.db.orm_execute(delete_ids, params, readonly=False)
                            continue
                        guarded = self._construct(
                            entity_id,
                            f"{shape}guarded_delete_ids:{guard.kind}",
                            lambda: delete(table).where(_object_ids_match(table.c.object_id, mysql_ids), gram._need_index().write_guard_clause(guard.kind)),
                        )
                        # the condition holds for every row or for none, so a removed row confirms the state; no
                        # removed row means that no row was stored or that the state changed
                        if self._affected_rows(self.db.orm_execute(guarded, {**params, **guard.params}, readonly=False)) > 0:
                            confirmed.add(entity_id)
                        elif not is_delete and not validate_first:
                            held.extend(ids)  # the guarded INSERT below tells which, in the same transaction
                        elif confirm(entity_id, guard):
                            self.db.orm_execute(delete_ids, params, readonly=False)
                else:
                    batch = self._admit_owned_rows(entity_id, table, batch, ownership)
                    applied.extend(batch)
                if batch and not is_delete:
                    side_names = storage.side_specs if storage is not None else self._side_specs.get(entity_id, {})
                    # One executemany INSERT takes its column list from its first
                    # parameter set. Batch only consecutive rows with the same
                    # columns so the database observes the caller's insert order.
                    payloads = []
                    for row in batch:
                        payload = row.to_row(fields=table.c.keys())
                        # the side table cannot tell an absent value from an empty
                        # list, so the main row carries the distinction
                        for field_name in side_names:
                            payload[_side_present_column(field_name)] = 0 if row.values.get(field_name) is None else 1
                        payloads.append(payload)
                    insert_rows = self._construct(entity_id, shape + "insert", lambda: insert(table))
                    guard = guards.get(entity_id) if ownership is None and entity_id not in confirmed else None
                    if guard is not None and validate_first:
                        # a statement that commits on its own never carries the check after another one has committed
                        confirm(entity_id, guard)
                        guard = None
                    if guard is not None:
                        first = payloads.pop(0)
                        columns = list(first)
                        guarded_insert = self._construct(
                            entity_id,
                            f"{shape}guarded_insert:{guard.kind}:" + "\x1f".join(columns),
                            lambda: self._guarded_insert(table, columns, gram._need_index().write_guard_clause(guard.kind)),
                        )
                        values = {f"hbv_{position}": first[column] for position, column in enumerate(columns)}
                        if self._affected_rows(self.db.orm_execute(guarded_insert, {**values, **guard.params}, readonly=False)) == 1:
                            confirmed.add(entity_id)
                        else:
                            confirm(entity_id, guard)
                            # the state changed, so the guarded DELETEs above removed nothing
                            for ids in _batches(held, batch_size):
                                params = {"object_ids": list(ids)}
                                if mysql_ids:
                                    params["exact_ids"] = params["object_ids"]
                                self.db.orm_execute(delete_ids, params, readonly=False)
                            self.db.orm_execute(insert_rows, [first], readonly=False)
                    for _, consecutive in groupby(payloads, key=frozenset):
                        self.db.orm_execute(insert_rows, list(consecutive), readonly=False)

            maintained_rows = rows if ownership is None else applied
            for row in maintained_rows:
                storage = shared.get(row.entity_id)
                side_specs = storage.side_specs if storage is not None else self._side_specs.get(row.entity_id, {})
                shape = "shared_" if storage is not None else ""
                for field_name, spec in side_specs.items():
                    delete_side = self._construct(
                        row.entity_id,
                        f"{shape}side_delete:{field_name}",
                        lambda: delete(spec.table).where(_object_id_matches(spec.table.c.object_id, bindparam("row_id"), mysql_ids)),
                    )
                    self.db.orm_execute(delete_side, {"row_id": row.object_id}, readonly=False)
                    if row.is_upsert():
                        side_rows = spec.rows(row.object_id, row.values.get(field_name) or [])
                        if side_rows:
                            insert_side = self._construct(row.entity_id, f"{shape}side_insert:{field_name}", lambda: insert(spec.table))
                            self.db.orm_execute(insert_side, side_rows, readonly=False)
            maintained = [row for row in maintained_rows if row.strategies]
            if maintained:
                for validate, upsert in actions:
                    if not validate_first:
                        validate(maintained)
                    upsert(maintained)
            if gram is not None and len(maintained) < len(maintained_rows) and hasattr(gram, "sweep_unmaintained"):
                # a row written without maintenance must leave no keyword rows, even ones another writer committed
                gram.sweep_unmaintained(maintained_rows, aliases=gram_aliases)
        return applied

    def _refuse_old_duckdb_replacement(
        self,
        release: str,
        entity_id: str,
        table: Any,
        batch: list[RowOp],
        is_delete: bool,
        ownership: tuple[str, str] | None,
        written: set[tuple[str, str]],
        removed: set[tuple[str, str]],
    ) -> None:
        """Refuse, before the batch runs, a batch that would replace a stored row on DuckDB before 1.2.

        A row this write inserted earlier lives in the transaction's own storage, which such a release replaces
        correctly, so only rows stored before the write count. ``removed`` collects the stored rows that delete
        batches of this write remove: inserting one of them again is a replacement too.
        """
        from sqlalchemy import select

        fresh = [str(row.object_id) for row in batch if (entity_id, str(row.object_id)) not in written]
        column = table.c.object_id
        owned = table.c[ownership[0]] == ownership[1] if ownership is not None and ownership[0] in table.c else None
        stored: set[str] = set()
        for ids in _batches(fresh, self._id_batch_size(self.db)):
            query = select(column).where(column.in_(list(ids)))
            stored.update(str(value) for value in self.db.orm_execute(query if owned is None else query.where(owned), readonly=True).column(0))
        if is_delete:
            removed.update((entity_id, object_id) for object_id in stored)
            return
        replaced = next((str(row.object_id) for row in batch if str(row.object_id) in stored or (entity_id, str(row.object_id)) in removed), None)
        if replaced is not None:
            _refuse_duckdb_row_replacement(release, f"the stored {entity_id!r} row {replaced!r} on backend {self.name!r}")

    def _exact_id_ops(self, rows: Sequence[RowOp], shared: dict[str, _SharedRowStorage]) -> list[RowOp]:
        """Keep a write from changing a stored row whose ``object_id`` the server only matches under its collation.

        The generic ``sql`` type cannot know how its server compares text, so a
        ``DELETE ... WHERE object_id IN (...)`` can match every stored ID the
        table's collation counts as equal, and an upsert of ``ALPHA`` would
        replace a stored ``alpha``. The stored IDs the server matches are read
        first, and compared with the requested ones in Python: an upsert whose
        ID matches a stored ID that differs from it is refused before the call
        writes anything, as a duplicate primary key is elsewhere, unless the same
        call deletes that stored ID before its first upsert of the entity. A
        delete of an ID that is not stored exactly is left out, so it removes
        nothing, side-table and keyword rows included.

        Raises:
            ValueError: If an upsert's ``object_id`` matches a stored ID that differs from it.
        """
        from sqlalchemy import bindparam, select

        kept: set[tuple[str, str]] = set()
        for entity_id in dict.fromkeys(row.entity_id for row in rows):
            storage = shared.get(entity_id)
            table = storage.table if storage is not None else self.tables.get(entity_id)
            if table is None:
                kept.update((entity_id, str(row.object_id)) for row in rows if row.entity_id == entity_id)
                continue
            shape = "shared_" if storage is not None else ""
            lookup = self._construct(
                entity_id, f"{shape}exact_id_lookup", lambda: select(table.c.object_id).where(table.c.object_id.in_(bindparam("object_ids", expanding=True)))
            )

            def matched(ids: list[str]) -> set[str]:
                found: set[str] = set()
                for batch in _batches(ids, self._id_batch_size(self.db)):
                    found.update(str(value) for value in self.db.orm_execute(lookup, {"object_ids": list(batch)}, readonly=True).column(0))
                return found

            upserts = list(dict.fromkeys(str(row.object_id) for row in rows if row.entity_id == entity_id and row.is_upsert()))
            deletes = list(dict.fromkeys(str(row.object_id) for row in rows if row.entity_id == entity_id and not row.is_upsert()))
            first_upsert = next((position for position, row in enumerate(rows) if row.entity_id == entity_id and row.is_upsert()), 0)
            removed = {str(row.object_id) for row in rows[:first_upsert] if row.entity_id == entity_id}
            if upserts and (others := matched(upserts) - set(upserts) - removed):
                raise ValueError(
                    f"{entity_id} object_id among {upserts!r} matches stored object_id {sorted(others)!r} under the table's collation, "
                    f"which does not compare IDs exactly on backend {self.name!r} of type {self.type!r}; this backend call wrote "
                    "nothing. Use IDs that differ under that collation, or a table whose object_id column has a binary collation"
                )
            kept.update((entity_id, object_id) for object_id in upserts)
            if deletes:
                kept.update((entity_id, object_id) for object_id in matched(deletes) if object_id in set(deletes))
        return [row for row in rows if (row.entity_id, str(row.object_id)) in kept]

    def _guarded_insert(self, table: Any, columns: Sequence[str], guard: Any) -> Any:
        """Return an ``INSERT`` of one row that adds the row only while ``guard`` holds.

        The values are bound as ``hbv_<position>`` in the order of ``columns``,
        each with its column's type. Oracle cannot bind a value longer than
        a ``VARCHAR2`` in a ``SELECT`` list (ORA-01461), so there the row goes
        through the ``VALUES`` clause of a conditional ``INSERT ALL``.
        """
        from sqlalchemy import bindparam, insert, select, text

        values = [bindparam(f"hbv_{position}", type_=table.c[column].type) for position, column in enumerate(columns)]
        if self.db.dialect == "duckdb":
            # SQLAlchemy drops the count row DuckDB returns for an INSERT, so the inserted row is returned instead
            return insert(table).from_select(list(columns), select(*values).where(guard)).returning(table.c.object_id)
        if self.db.dialect != "oracle":
            return insert(table).from_select(list(columns), select(*values).where(guard))
        dialect = self.db.engine.dialect
        quote = dialect.identifier_preparer
        names = ", ".join(quote.quote(column) for column in columns)
        binds = ", ".join(f":hbv_{position}" for position in range(len(columns)))
        condition = guard.compile(dialect=dialect)
        statement = (
            f"INSERT ALL WHEN hb_ok = 1 THEN INTO {quote.format_table(table)} ({names}) VALUES ({binds}) "
            f"SELECT CASE WHEN {condition} THEN 1 ELSE 0 END AS hb_ok FROM DUAL"
        )
        return text(statement).bindparams(*values)

    def _affected_rows(self, response: Any) -> int:
        """Return how many rows one ``INSERT`` or ``DELETE`` changed.

        DuckDB's driver reports ``-1`` and returns the count of a ``DELETE`` as
        the statement's one ``Count`` row, which the response then counts as
        its one row; the guarded ``INSERT`` returns the row it inserted.
        """
        if self.db.dialect == "duckdb":
            rows = response.to_lists()
            if response._columns == ["Count"]:
                return int(rows[0][0]) if rows else 0
            return len(rows)
        return int(response._row_count)

    def _admit_owned_rows(self, entity_id: str, table: Any, batch: list[RowOp], ownership: tuple[str, str]) -> list[RowOp]:
        """Delete only matching owners and admit missing upserts to the plain INSERT path."""
        from sqlalchemy import bindparam, delete, select

        field, owner = ownership
        if field not in table.c:
            raise NotImplementedError(f"Cannot check ownership of {entity_id!r} on {self.name!r}: this route does not store {field!r}")
        dialect = self.db.engine.dialect
        returning = not dialect.supports_sane_rowcount and dialect.delete_returning
        key = _object_id_matches(table.c.object_id, bindparam("row_id"), self._uses_mysql_id_keys())

        def delete_owned() -> Any:
            stmt = delete(table).where(key, table.c[field] == bindparam("owner"))
            return stmt.returning(table.c.object_id) if returning else stmt

        remove = self._construct(entity_id, f"delete_owned:{field}", delete_owned)
        occupied = self._construct(entity_id, "owned_key_exists", lambda: select(table.c.object_id).where(key))
        admitted = []
        for row in batch:
            params = {"row_id": row.object_id}
            # The public row_count falls back to zero when a driver reports no count.
            result = self.db.orm_execute(remove, {**params, "owner": owner}, readonly=False)
            removed = len(result.to_dicts()) if returning else result._row_count
            if removed == 1:
                admitted.append(row)
            elif removed != 0:
                raise RuntimeError(f"Backend {self.name!r} did not report whether the ownership-guarded delete matched")
            elif row.is_upsert() and not self.db.orm_execute(occupied, params, readonly=True).to_dicts():
                # A concurrent insert here can cause a primary-key conflict. A plain
                # INSERT leaves the winner intact and rolls back this transaction.
                admitted.append(row)
        return admitted

    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return all rows for one SQL-backed entity.

        The statement is planned under the backend lock and runs outside it.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            list[dict[str, Any]]: Decoded row fragments ordered by ``object_id``.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.
        """
        from sqlalchemy import select

        with self._lock:
            storage = self._read_storage(entity_id)
            if storage is None:
                return []
            db, table, side_specs = storage
            stmt = self._construct(entity_id, "scan", lambda: select(*self._select_cols(entity_id, table)).order_by(table.c.object_id))
        with self._read_scope(db):
            rows = self._decode_side_rows(side_specs, db.orm_execute(stmt, readonly=True).to_dicts())
        rows.sort(key=lambda row: str(row.get("object_id")))
        return rows

    def iter_scan(self, entity_id: str, *, batch_size: int, columns: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
        """Iterate SQL rows by bounded primary-key pages.

        Each page is one ``scan_page`` call, so the backend lock is held only
        while a page is planned and never across a ``yield``. The pages are
        native even on a subclass that pages through the portable default
        because it overrides ``scan`` or ``iter_scan``: such a subclass reaches
        this method only through ``super()``, from its override.

        Args:
            entity_id (str): Logical entity identifier.
            batch_size (int): Positive page size.
            columns (Sequence[str] | None): Optional field names to select. ``None`` selects
                every column and side table; a selection selects ``object_id`` plus the named
                columns and side tables only, so unrequested values are never fetched. A name
                this backend does not store is left out.

        Yields:
            dict[str, Any]: Decoded row fragment in database primary-key order.

        Raises:
            ValueError: If the page size is not positive.
            TypeError: If ``columns`` contains a non-string name.
            RuntimeError: If storage is unavailable.

        Notes:
            Keyset pagination avoids OFFSET growth. Concurrent external writes
            are not a snapshot; use an immutable source version when caching.
        """
        after = None
        page = native_layer(self, "scan_page")
        while rows := page(entity_id, after=after, batch_size=batch_size, columns=columns):
            after = rows[-1]["object_id"]
            yield from rows

    def scan_page(self, entity_id: str, *, after: Any, batch_size: int, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return one keyset page of decoded rows ordered by ``object_id``, strictly after ``after``.

        The cursor and page size travel as bind parameters, so every page of
        every scan with the same column selection reuses one cached construct.
        The page is planned under the backend lock and fetched outside it.

        Args:
            entity_id (str): Logical entity identifier.
            after (Any): ``object_id`` the page must follow, or ``None`` for the first page.
            batch_size (int): Positive maximum number of rows in the page.
            columns (Sequence[str] | None): Optional field names to select, as for ``iter_scan``.

        Returns:
            list[dict[str, Any]]: Decoded row fragments in ``object_id`` order; empty when
            the entity is unknown here or nothing follows ``after``.

        Raises:
            ValueError: If the page size is not positive.
            TypeError: If ``columns`` contains a non-string name.
            RuntimeError: If the backend has been dropped and not reconnected.
        """
        self._check_batch_size(batch_size)
        from sqlalchemy import Integer, bindparam, select

        with self._lock:
            storage = self._read_storage(entity_id)
            if storage is None:
                return []
            db, table, side_specs = storage
            shape, selection = self._projection(entity_id, table, columns)

            def page_select():
                return select(*self._select_cols(entity_id, table, selection)).order_by(table.c.object_id).limit(bindparam("page_size", type_=Integer))

            if after is None:
                page = self._construct(entity_id, f"scan_page{shape}", page_select)
                params: dict[str, Any] = {"page_size": batch_size}
            else:
                page = self._construct(entity_id, f"scan_page_after{shape}", lambda: page_select().where(table.c.object_id > bindparam("after_id")))
                params = {"after_id": after, "page_size": batch_size}
        with self._read_scope(db):
            return self._decode_side_rows(side_specs, db.orm_execute(page, params, readonly=True).to_dicts(), selection)

    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> List[Optional[Dict[str, Any]]]:
        """Return rows with batched ``WHERE object_id IN (...)`` lookups.

        The statement is planned under the backend lock. One batch runs
        outside it; several batches (more distinct ids than
        ``heavenbase.backend_defaults.sql.id_lookup.batch_size``, or the
        dialect's bind limit) run under it, as in 0.1.2.2, so no write of this
        backend lands between two of them.

        Args:
            entity_id: Logical entity identifier.
            object_ids: Object ids to read.
            columns: Optional field names to select. ``None`` selects every column and side
                table; a selection selects ``object_id`` plus the named columns and side
                tables only, so unrequested values are never fetched or decoded. A name
                this backend does not store is left out.

        Returns:
            List[Optional[Dict[str, Any]]]: Decoded row fragments aligned to ``object_ids``.
                Missing rows are ``None``.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.
            TypeError: If ``columns`` contains a non-string name.
        """
        ordered_ids = list(object_ids)
        from sqlalchemy import bindparam, select

        with self._lock:
            storage = self._read_storage(entity_id)
            if storage is None or not ordered_ids:
                return [None for _ in ordered_ids]
            db, table, side_specs = storage
            shape, selection = self._projection(entity_id, table, columns)
            stmt = self._construct(
                entity_id,
                f"get_many{shape}",
                lambda: select(*self._select_cols(entity_id, table, selection)).where(table.c.object_id.in_(bindparam("object_ids", expanding=True))),
            )
            # a side-table projection binds the JSON key names of its items besides the id list
            other_binds = self._construct(entity_id, f"get_many{shape}:other-binds", lambda: _bound_values_besides(stmt, "object_ids"))
        batches = list(_batches(list(dict.fromkeys(ordered_ids)), self._id_batch_size(db, other_binds)))
        rows_by_id: Dict[Any, Dict[str, Any]] = {}
        with self._read_scope(db) if len(batches) == 1 else self._locked_read(db):
            for batch in batches:
                rows = db.orm_execute(stmt, {"object_ids": list(batch)}, readonly=True).to_dicts()
                for row in self._decode_side_rows(side_specs, rows, selection):
                    rows_by_id[row.get("object_id")] = row
        return [rows_by_id.get(row_id) for row_id in ordered_ids]

    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> List[bool]:
        """Return existence flags using only the SQL object_id column.

        The statement is planned and its batches run as for ``get_many``, so no
        write of this backend lands between two batches.

        Args:
            entity_id: Logical entity identifier.
            object_ids: Object ids to test.

        Returns:
            List[bool]: Boolean flags aligned to ``object_ids``.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.
        """
        ordered_ids = list(object_ids)
        from sqlalchemy import bindparam, select

        with self._lock:
            storage = self._read_storage(entity_id)
            if storage is None or not ordered_ids:
                return [False for _ in ordered_ids]
            db, table, _side_specs = storage
            stmt = self._construct(
                entity_id, "exists_many", lambda: select(table.c.object_id).where(table.c.object_id.in_(bindparam("object_ids", expanding=True)))
            )
        batches = list(_batches(list(dict.fromkeys(ordered_ids)), self._id_batch_size(db)))
        found = set()
        with self._read_scope(db) if len(batches) == 1 else self._locked_read(db):
            for batch in batches:
                found.update(db.orm_execute(stmt, {"object_ids": list(batch)}, readonly=True).column(0))
        return [row_id in found for row_id in ordered_ids]

    def count(self, entity_id: str) -> int:
        """Return row count without scanning SQL rows into Python.

        The statement is planned under the backend lock and runs outside it.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            int: Count of rows in the SQL table, or ``0`` when the entity is unknown.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.
        """
        from sqlalchemy import func, select

        with self._lock:
            storage = self._read_storage(entity_id)
            if storage is None:
                return 0
            db, table, _side_specs = storage
            stmt = self._construct(entity_id, "count", lambda: select(func.count()).select_from(table))
        with self._read_scope(db):
            return int(db.orm_execute(stmt, readonly=True).scalar() or 0)

    def _read_storage(self, entity_id: str) -> tuple[Any, Any, dict[str, _SideTableSpec]] | None:
        """Return the live database, main table, and side-table specs one read needs.

        Call under the backend lock. The returned references are the read's
        snapshot: the statement built from them and the rows decoded with them
        stay consistent even when ``ensure``, ``drop``, or a workspace rebind
        replaces the backend's storage state once the lock is released.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            tuple[Any, Any, dict[str, _SideTableSpec]] | None: ``(db, table, side_specs)``,
            or ``None`` when the entity has no table on this backend.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.
        """
        self._need_db()
        table = self.tables.get(entity_id)
        if table is None:
            return None
        return self.db, table, dict(self._side_specs.get(entity_id, {}))

    def _construct_cache(self) -> Dict[tuple[str, str], Any]:
        """Return the reusable-construct cache, creating it for an instance restored without ``__init__``."""
        cache = self.__dict__.get("_constructs")
        if cache is None:
            cache = self._constructs = {}
        return cache

    def _construct(self, entity_id: str, shape: str, build: Callable[[], Any]) -> Any:
        """Return one reusable SQLAlchemy construct per entity and statement shape.

        The cache is bounded by ``_construct_cache_size`` and evicts the least
        recently used construct: a hit moves the construct to the end of the
        order, so the shapes a workload keeps using are never evicted by a
        stream of one-off selections. A shape may name a column selection but
        never a caller-supplied value. Call under the backend lock: every hit
        reorders the cache and a miss may evict, so it is not safe to share
        between unlocked threads.

        Args:
            entity_id: Logical entity identifier the construct reads or writes.
            shape: Statement shape label; parameter values travel as bind parameters, never in the shape.
            build: Zero-argument builder invoked once per shape until ``_forget_constructs`` runs or the shape is evicted.

        Returns:
            Any: The cached construct. Reusing one object lets SQLAlchemy memoize its cache key and serve every later
            execution from the engine's compiled cache instead of compiling again.
        """
        cache = self._construct_cache()
        key = (entity_id, shape)
        construct = cache.pop(key, None)
        if construct is None:
            construct = build()
            while len(cache) >= self._construct_cache_size:
                del cache[next(iter(cache))]
        cache[key] = construct
        return construct

    def _forget_constructs(self, entity_id: str | None = None) -> None:
        """Drop reusable constructs for one entity, or for all entities, after their shape may have changed."""
        cache = self._construct_cache()
        if entity_id is None:
            cache.clear()
            return
        for key in [key for key in cache if key[0] == entity_id]:
            del cache[key]

    def _projection(self, entity_id: str, table: Any, columns: Sequence[str] | None) -> tuple[str, tuple[str, ...] | None]:
        """Narrow a column selection to the fields one entity's SQL storage holds.

        As the ``Backend.get_many`` contract states, a selection returns
        ``object_id`` plus the named fields this backend stores; a name it does
        not store (a field routed to another backend, or an internal
        presence-marker column) is left out rather than rejected.

        Args:
            entity_id: Logical entity identifier.
            table: Main SQLAlchemy table for the entity.
            columns: Requested field names, or ``None`` for every column.

        Returns:
            tuple[str, tuple[str, ...] | None]: Statement-shape suffix (empty for the complete
            row) and the normalized selection of stored names.

        Raises:
            TypeError: If a requested name is not a string.
        """
        selection = self._row_columns(columns)
        if selection is None:
            return "", None
        stored = {column.name for column in table.c if not column.name.startswith("_present_")}
        stored.update(self._side_specs.get(entity_id, {}))
        selection = tuple(name for name in selection if name in stored)
        return ":" + ",".join(selection), selection

    def _select_cols(self, entity_id: str, table: Any, selection: tuple[str, ...] | None = None) -> list[Any]:
        side_specs = self._side_specs.get(entity_id, {})
        if selection is None:
            wanted_columns = None
            wanted_side = side_specs
        else:
            # a selected side field needs its presence marker to tell an absent value from an empty list
            wanted_side = {name: spec for name, spec in side_specs.items() if name in selection}
            wanted_columns = {*selection, *(_side_present_column(name) for name in wanted_side)}
        return self._construct(
            entity_id,
            "columns" if selection is None else "columns:" + ",".join(selection),
            lambda: [
                *[column for column in table.c if wanted_columns is None or column.name in wanted_columns],
                *[spec.json_expr(table, mysql_ids=self._uses_mysql_id_keys()) for spec in wanted_side.values()],
            ],
        )

    @staticmethod
    def _decode_side_rows(side_specs: Dict[str, _SideTableSpec], rows: list[dict[str, Any]], selection: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Decode side-table values in place using the specs snapshotted under the backend lock."""
        for row in rows:
            for field_name, spec in side_specs.items():
                if selection is not None and field_name not in selection:
                    continue
                # a row written before markers existed, beside an existing side table,
                # has no marker; treat it as present so its stored items keep reading
                # as they did (rows stored before their side table existed are marked 0)
                present = row.pop(_side_present_column(field_name), None)
                row[field_name] = None if present == 0 else spec.decode(row.get(field_name))
        return rows

    def _is_sql_fragment(self, fragment: QueryFragment) -> bool:
        if fragment.kind() in _SQL_FRAGMENT_KINDS:
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_sql_fragment(child) for child in fragment.children())
        if fragment.op == "not":
            return self._is_sql_fragment(fragment.children()[0])
        return False

    def _can_scan_fragment(self, fragment: QueryFragment) -> bool:
        """Return whether every leaf exposes a portable row predicate."""
        if fragment.op in LOGICAL_OPS:
            return all(self._can_scan_fragment(child) for child in fragment.children())
        if fragment.notes.get("strategy_adapter"):
            # Adapter-owned execution may apply strategy-specific normalization
            # that a generic row predicate cannot reproduce.
            return False
        if callable(fragment.notes.get("row_payload")):
            return True
        return fragment.mode() == "py_scan" and callable(fragment.payload)

    def _where(self, fragment: QueryFragment, table: Any, entity_id: str) -> Any:
        from sqlalchemy import and_, not_, or_

        if fragment.kind() == "sql_where":
            return fragment.payload(table)
        if fragment.kind() == "sql_array_contains":
            from sqlalchemy import exists, select

            field_name = fragment.field
            value = fragment.payload["value"]
            spec = self._side_specs.get(entity_id, {}).get(field_name)
            if spec is None or spec.dtype_identifier != "array":
                raise RuntimeError(f"No side table for {field_name}")
            owner_matches = spec.owner_matches(table, self._uses_mysql_id_keys())
            return exists(select(1).where(owner_matches, spec.table.c.item == value))
        if fragment.kind() == "sql_array_exact":
            from sqlalchemy import exists, false, func, select

            field_name = fragment.field
            spec = self._side_specs.get(entity_id, {}).get(field_name)
            if spec is None or spec.dtype_identifier != "array":
                raise RuntimeError(f"No side table for {field_name}")
            owner_matches = spec.owner_matches(table, self._uses_mysql_id_keys())

            def exact(candidate: Any) -> Any:
                """Return a SQL predicate for exact array equality.

                Args:
                    candidate: Candidate array value to compare with side-table rows.

                Returns:
                    Any: SQLAlchemy predicate matching rows whose side-table values exactly
                    equal ``candidate`` by length and ordinal item values.
                """
                if not isinstance(candidate, list):
                    return false()
                predicates = [
                    select(func.count()).where(owner_matches).scalar_subquery() == len(candidate),
                ]
                for idx, item in enumerate(candidate):
                    predicates.append(
                        exists(
                            select(1).where(
                                owner_matches,
                                spec.table.c.idx == idx,
                                spec.table.c.item.is_(None) if item is None else spec.table.c.item == item,
                            )
                        )
                    )
                return and_(*predicates)

            if fragment.op == "in":
                clauses = [exact(candidate) for candidate in fragment.payload.get("values", [])]
                return or_(*clauses) if clauses else false()
            condition = exact(fragment.payload.get("value"))
            return not_(condition) if fragment.op == "ne" else condition
        if fragment.kind() == "sql_hyperg_exists":
            from sqlalchemy import exists, select

            field_name = fragment.field
            spec = self._side_specs.get(entity_id, {}).get(field_name)
            if spec is None or spec.dtype_identifier != "hyper-g":
                raise RuntimeError(f"No side table for {field_name}")
            predicates = [spec.owner_matches(table, self._uses_mysql_id_keys())]
            for hyperg_field, ops in fragment.payload["predicates"].items():
                col = spec.table.c[hyperg_field]
                predicates.extend(self._hyperg_sql_predicate(col, op, value) for op, value in ops.items())
            return exists(select(1).where(and_(*predicates)))
        if fragment.kind() == "sql_hyperg_all":
            from sqlalchemy import exists, select

            field_name = fragment.field
            spec = self._side_specs.get(entity_id, {}).get(field_name)
            if spec is None or spec.dtype_identifier != "hyper-g":
                raise RuntimeError(f"No side table for {field_name}")
            predicates = [spec.owner_matches(table, self._uses_mysql_id_keys())]
            for hyperg_field, ops in fragment.payload["where"].items():
                col = spec.table.c[hyperg_field]
                predicates.extend(self._hyperg_sql_predicate(col, op, value) for op, value in ops.items())
            failed = []
            for hyperg_field, ops in fragment.payload["then"].items():
                col = spec.table.c[hyperg_field]
                failed.extend(self._hyperg_sql_predicate(col, op, value) for op, value in ops.items())
            return not_(exists(select(1).where(and_(*predicates, not_(and_(*failed))))))
        if fragment.op == "and":
            return and_(*[self._where(child, table, entity_id) for child in fragment.children()])
        if fragment.op == "or":
            return or_(*[self._where(child, table, entity_id) for child in fragment.children()])
        if fragment.op == "not":
            return not_(self._where(fragment.children()[0], table, entity_id))
        raise TypeError(f"Cannot compile {fragment.op!r} to SQL")

    def _hyperg_sql_predicate(self, col: Any, op: str, value: Any) -> Any:
        from sqlalchemy import and_, false, or_

        if op == "eq":
            return col.is_(None) if value is None else and_(col.is_not(None), col == value)
        if op == "ne":
            return col.is_not(None) if value is None else or_(col.is_(None), col != value)
        if op == "lt":
            return and_(col.is_not(None), col < value)
        if op == "lte":
            return and_(col.is_not(None), col <= value)
        if op == "gt":
            return and_(col.is_not(None), col > value)
        if op == "gte":
            return and_(col.is_not(None), col >= value)
        if op == "in":
            values = list(value)
            non_null = [item for item in values if item is not None]
            clauses = []
            if non_null:
                clauses.append(col.in_(non_null))
            if any(item is None for item in values):
                clauses.append(col.is_(None))
            return or_(*clauses) if clauses else false()
        raise ValueError(f"Unsupported SQL HyperG predicate op {op}")

    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute SQL-native fragments or fall back to scanning.

        The statement is planned under the backend lock and runs outside it.
        A fragment owned by a strategy adapter runs entirely under the lock,
        because the adapter's index handles and fingerprints are guarded by
        that lock alone; its row reads re-enter it.

        Args:
            fragment: Compiled query fragment.

        Returns:
            ResultFrame: Rows matched by SQL pushdown, strategy adapter execution, or
                portable scanning fallback.

        Raises:
            RuntimeError: If the backend has been dropped and not reconnected.
        """
        with self._lock:
            self._need_db()
            db = self.db
            if fragment.kind() == "sql_aggregate":
                read = self._aggregate_read(fragment)
            elif (adapter := self.strategy_adapter_for_fragment(fragment)) is not None:
                return adapter.execute(fragment)
            else:
                read = self._fragment_read(fragment)
        if self._serialized_reads(db):
            with self._lock:
                return read()
        return read()  # each statement it runs takes its own read scope

    def _aggregate_read(self, fragment: QueryFragment) -> Callable[[], ResultFrame]:
        """Plan one aggregate statement under the backend lock; the returned callable runs it outside."""
        payload = fragment.payload
        entity_id = fragment.entity_id()
        table = self.tables.get(entity_id) if entity_id else None
        if table is None:
            return lambda: ResultFrame.from_table(payload.get("empty_rows", ()), cols=payload.get("columns", ()))
        where_fragment = payload.get("where")
        where_clause = self._where(where_fragment, table, entity_id) if where_fragment is not None else None
        statement = payload["build"](table, where_clause)
        db = self.db

        def read() -> ResultFrame:
            with self._read_scope(db):
                rows = db.orm_execute(statement, readonly=True).to_dicts()
            for row in rows:
                for name, cast in payload.get("casts", {}).items():
                    if row.get(name) is not None:
                        row[name] = int(row[name]) if cast == "int" else float(row[name])
            return ResultFrame.from_table(rows, cols=payload["columns"])

        return read

    def _fragment_read(self, fragment: QueryFragment) -> Callable[[], ResultFrame]:
        """Plan one filter fragment under the backend lock; the returned callable runs it outside.

        A fragment SQL cannot express falls back to scanning through the public
        ``scan`` and ``execute`` methods, each of which locks its own planning.
        """
        portable = super().execute
        if fragment.kind() not in _SQL_FRAGMENT_KINDS and fragment.op not in LOGICAL_OPS:
            return lambda: portable(fragment)
        entity_id = fragment.entity_id()
        table = self.tables.get(entity_id) if entity_id else None
        if table is None:
            return lambda: ResultFrame.empty()
        if fragment.op in LOGICAL_OPS and fragment.uses_scan() and self._can_scan_fragment(fragment):
            # Evaluate the complete boolean tree against one exhaustive row
            # snapshot when every leaf exposes a portable predicate, so
            # built-in mixed trees do not need one scan per child.
            return lambda: portable(fragment)
        if not self._is_sql_fragment(fragment):
            if fragment.op in LOGICAL_OPS:
                return lambda: fragment.execute(
                    self.execute, lambda item_entity_id: ResultFrame.from_rows(self.scan(item_entity_id)) if item_entity_id else ResultFrame.empty()
                )
            return lambda: portable(fragment)
        from sqlalchemy import select

        stmt = select(*self._select_cols(entity_id, table)).where(self._where(fragment, table, entity_id)).order_by(table.c.object_id)
        db, side_specs = self.db, dict(self._side_specs.get(entity_id, {}))

        def read() -> ResultFrame:
            with self._read_scope(db):
                rows = self._decode_side_rows(side_specs, db.orm_execute(stmt, readonly=True).to_dicts())
            rows.sort(key=lambda row: str(row.get("object_id")))
            return ResultFrame.from_rows(rows)

        return read

    def to_dict(self) -> dict[str, Any]:
        """Return SQL backend metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend name, backend type, provider, and active database.
        """
        return {"name": self.name, "type": self.type, "provider": self.provider, "database": self.database}
