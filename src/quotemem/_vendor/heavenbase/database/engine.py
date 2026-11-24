"""SQLAlchemy engine construction for resolved database specs."""

from __future__ import annotations

__all__ = ["create_database_engine"]

import random
import re
import time
from os.path import isabs, realpath

from sqlalchemy import create_engine, event

from ..utils.log import get_logger
from ..utils.typing import Any
from .ddl import create_database
from .config import DB_ENGINE, DBSpec
from .sql_text import strip_sql_comments

logger = get_logger(__name__)


# SQLite answers a journal-mode change with "database is locked" at once,
# without its busy handler, while another connection creates or converts the
# same file; such a pragma is retried. A SQLite connection's initialization
# retries within the connection's own busy timeout (see _busy_seconds), so a
# database another connection holds locked fails after one busy timeout, as
# in 0.1.2.2; other dialects retry within this many seconds.
_PRAGMA_LOCK_DEADLINE = 30.0

_SQLITE_SPACE = r"(?:\s|\ufeff|--[^\n]*(?:\n|\Z)|/\*(?:[^*]|\*(?!/))*\*/)"
_SQLITE_IDENTIFIER = r"""(?:[a-z_\x80-\U0010ffff][a-z0-9_$\x80-\U0010ffff]*|"(?:[^"]|"")*"|'(?:[^']|'')*'|`(?:[^`]|``)*`|\[[^\]]*\])"""
_SQLITE_BUSY_TIMEOUT = re.compile(
    rf"\A(?:{_SQLITE_SPACE}|;)*PRAGMA\b{_SQLITE_SPACE}*(?:(?P<schema>{_SQLITE_IDENTIFIER}){_SQLITE_SPACE}*\.{_SQLITE_SPACE}*)?"
    rf"""(?:busy_timeout|"busy_timeout"|'busy_timeout'|`busy_timeout`|\[busy_timeout\]){_SQLITE_SPACE}*[=(]""",
    re.IGNORECASE | re.ASCII,
)
_POSTGRES_SET_LOCAL = re.compile(r"\A(?:\s|;)*SET\s+LOCAL\s+", re.IGNORECASE | re.ASCII)


def _locked(error: Exception) -> bool:
    text = str(error).lower()
    return "database is locked" in text or "database table is locked" in text


def _ordered_pragmas(pragmas: list[str]) -> list[str]:
    """Set SQLite's busy handler early, preserving the order of other setup SQL."""
    timeouts, other = [], []
    for pragma in pragmas:
        match = _SQLITE_BUSY_TIMEOUT.match(pragma)
        if match is None:
            other.append(pragma)
            continue
        schema = match["schema"]
        if schema:
            if schema[0] in "\"'`[":
                schema = schema[1:-1]
            if schema.lower() not in {"main", "temp"}:
                # The attached schema must already exist. All assignments set
                # the same handler, so keep their configured order.
                return pragmas
        timeouts.append(pragma)
    return timeouts + other


def _busy_seconds(dbapi_connection: Any) -> float:
    """Return a SQLite connection's busy timeout in seconds, or ``_PRAGMA_LOCK_DEADLINE`` when it cannot be read."""
    try:
        value = dbapi_connection.execute("PRAGMA busy_timeout").fetchone()[0]
        return max(int(value), 0) / 1000
    except Exception:
        return _PRAGMA_LOCK_DEADLINE


def _execute_pragma(cursor: Any, pragma: str, deadline: float | None = None) -> None:
    """Run one initialization statement, retrying it while the database answers "locked" until ``deadline``.

    ``deadline`` is a ``time.monotonic()`` value; ``None`` allows ``_PRAGMA_LOCK_DEADLINE`` seconds from now.
    """
    deadline = time.monotonic() + _PRAGMA_LOCK_DEADLINE if deadline is None else deadline
    pause = 0.005
    while True:
        try:
            cursor.execute(pragma)
            return
        except Exception as error:
            if not _locked(error) or time.monotonic() + pause > deadline:
                raise
        time.sleep(random.uniform(pause / 2, pause))
        pause = min(pause * 2, 0.25)


def _register_pragmas(engine: Any, pragmas: list[str]) -> None:
    if not pragmas:
        return
    ordered = _ordered_pragmas(pragmas) if engine.dialect.name == "sqlite" else pragmas

    sqlite = engine.dialect.name == "sqlite"

    @event.listens_for(engine, "connect")
    def _run_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
        started = time.monotonic()
        cursor = dbapi_connection.cursor()
        try:
            for pragma in ordered:
                # The busy timeout read before each statement is the one the statements before it set.
                deadline = started + _busy_seconds(dbapi_connection) if sqlite else None
                _execute_pragma(cursor, pragma, deadline)
        finally:
            cursor.close()
        # Finish initialization before application SQL can roll it back.
        dbapi_connection.commit()

    if engine.dialect.name == "postgresql":
        local = [statement for statement in ordered if _POSTGRES_SET_LOCAL.match(strip_sql_comments(statement))]
        if local:

            @event.listens_for(engine, "begin")
            def _run_transaction_pragmas(conn: Any) -> None:
                # SET LOCAL expires at commit or rollback, including the
                # setup commit above. Apply it in each application transaction.
                for statement in local:
                    conn.exec_driver_sql(statement)


def _open_resolved_file(engine: Any, spec: DBSpec) -> None:
    """Make each connection of a file-backed SQLite engine open the file by the path its symbolic links resolve to.

    SQLite names a database's ``-wal`` and ``-shm`` files after the path it opens, and its Windows file layer does
    not resolve symbolic links, so a database opened through a file link in WAL mode kept its own ``<link>-wal``
    beside the link. Each connection resolves the link as it opens, so a link that changes target is followed as
    before. Only the file name given to the driver changes: the engine's URL and error messages keep the
    configured name. SQLite's Unix file layer resolves links itself, so this opens the same file there as before.
    DuckDB is left as in 0.1.2.2: it names its catalog after the file name it opens.
    """
    configured = engine.url.database
    if spec.in_memory or engine.dialect.name != "sqlite" or not configured or not isabs(configured):
        return
    url = engine.url
    original_args, original_params = engine.dialect.create_connect_args(url)
    original_args = list(original_args)

    @event.listens_for(engine, "do_connect")
    def _connect_resolved(dialect: Any, connection_record: Any, cargs: list[Any], cparams: dict[str, Any]) -> None:
        resolved = realpath(configured)
        if resolved == configured:
            return
        args, params = dialect.create_connect_args(url.set(database=resolved))
        for index, (before, after) in enumerate(zip(original_args, args)):
            if before != after and index < len(cargs) and cargs[index] == before:
                cargs[index] = after
        for key, after in params.items():
            before = original_params.get(key)
            if before != after and cparams.get(key) == before:
                cparams[key] = after


def _register_mysql_transactions(engine: Any) -> None:
    """Turn the server's autocommit off on each new PyMySQL connection that asked for transactions.

    PyMySQL sends ``SET autocommit=0`` only when the server's handshake reports
    autocommit on. Dolt and OceanBase report a status without that flag while
    their sessions start in autocommit mode, so without this every statement
    would commit on its own and a rollback would undo nothing. A connection
    that asked for autocommit (``isolation_level="AUTOCOMMIT"`` or a true
    ``autocommit`` argument) or left it to the server (``autocommit=None``)
    is left alone, and so are other MySQL drivers, which send the setting
    themselves. It runs before the configured pragmas, so a pragma still wins.
    """
    if engine.dialect.name != "mysql":
        return

    @event.listens_for(engine, "connect")
    def _start_transactions(dbapi_connection: Any, connection_record: Any) -> None:
        if getattr(dbapi_connection, "autocommit_mode", None) is not False:
            return
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SET autocommit=0")
        finally:
            cursor.close()


def create_database_engine(spec: DBSpec, autocreate: bool = True) -> Any:
    """Create a SQLAlchemy engine from a resolved database spec.

    Args:
        spec (DBSpec): Resolved database configuration.
        autocreate (bool): Whether to attempt best-effort database namespace creation before opening the engine.

    Returns:
        Any: SQLAlchemy engine configured from `spec`.

    Warnings:
        Autocreate failures are logged and the engine connection still proceeds so existing databases can be opened.
    """
    if autocreate:
        try:
            create_database(spec)
        except Exception as exc:
            logger.warning("Database autocreate failed; attempting direct engine connection for existing database: %s", exc)
    kw = DB_ENGINE.apply(spec, profile="engine")
    url = kw.pop("url")
    engine = create_engine(url, **kw)
    _open_resolved_file(engine, spec)
    _register_mysql_transactions(engine)
    _register_pragmas(engine, DB_ENGINE.apply(spec, profile="pragmas"))
    return engine
