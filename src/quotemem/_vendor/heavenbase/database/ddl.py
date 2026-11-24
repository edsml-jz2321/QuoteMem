"""Database create/drop helpers."""

from __future__ import annotations

__all__ = ["create_database", "drop_database"]

from pathlib import Path

from sqlalchemy import create_engine, text

from ..utils.files import get_file_dir, pj, touch_dir
from .config import DB_ENGINE, DBSpec

_MAINTENANCE_DBS = {"postgres", "mysql", "master", "default", "freepdb1", "system", "sys"}


def _server_ddl(spec: DBSpec, drop: bool = False, force: bool = False) -> None:
    dialect = spec.dialect_spec
    namespace = dialect.lifecycle_namespace(spec)
    if not namespace:
        return
    if drop and not force and namespace.lower() in _MAINTENANCE_DBS:
        raise ValueError(f"Refusing to drop maintenance database {namespace!r}")
    exists_sql = dialect.database_exists_sql()
    ddl_sql = dialect.drop_database_statement(namespace) if drop else dialect.create_database_statement(spec, namespace)
    if not ddl_sql:
        return
    kw = DB_ENGINE.apply(spec, profile="superuser")
    url = kw.pop("url")
    engine = create_engine(url, **kw)
    try:
        with engine.connect() as conn:
            exists = conn.execute(text(exists_sql), {"name": namespace}).scalar() if exists_sql else None
            if drop and exists_sql and not exists:
                return
            if not drop and exists_sql and exists:
                return
            for sql, params in dialect.before_drop_force_statements(namespace, force=force):
                conn.execute(text(sql), params or {})
            conn.execute(text(ddl_sql))
            for sql in dialect.after_create_statements(spec, namespace) if not drop else []:
                conn.execute(text(sql))
            conn.commit()
    finally:
        engine.dispose()


def create_database(spec: DBSpec) -> None:
    """Create the configured database namespace when supported.

    SQLite and DuckDB create parent folders only. Server backends use the superuser materialization from ``DBSpec`` and
    skip creation when the namespace already exists. Oracle treats the target namespace as ``schema``/``username`` because
    ``database`` is the PDB service name.

    Args:
        spec (DBSpec): Resolved database configuration.

    Returns:
        None: This function does not return a value.
    """
    dialect = spec.dialect_spec
    namespace = dialect.lifecycle_namespace(spec)
    if not namespace:
        return
    if dialect.supports("database-local-path"):
        if not spec.in_memory and spec.database:
            from ..utils.files import file_uri_to_path

            folder = get_file_dir(file_uri_to_path(spec.database))
            if folder:
                touch_dir(folder)
        return
    if dialect.supports("database-create"):
        _server_ddl(spec)


def _drops_namespace(spec: DBSpec) -> bool:
    """Return whether ``drop_database()`` removes anything for this spec.

    Args:
        spec (DBSpec): Resolved database configuration.

    Returns:
        bool: ``True`` when the spec names a local file or a server namespace that ``drop_database()`` removes.
    """
    dialect = spec.dialect_spec
    namespace = dialect.lifecycle_namespace(spec)
    if not namespace:
        return False
    if dialect.supports("database-local-path"):
        return not spec.in_memory and bool(spec.database)
    return dialect.supports("database-drop") and bool(dialect.drop_database_statement(namespace))


def drop_database(spec: DBSpec, *, force: bool = False) -> None:
    """Drop the configured database namespace when supported.

    Args:
        spec: Resolved database spec.
        force: Permit destructive drops for non-maintenance namespaces and request active-session cleanup where supported.

    Warnings:
        Maintenance namespaces such as ``postgres``, ``mysql``, ``master``, ``default``, ``FREEPDB1``, ``system``, and
        ``sys`` are refused unless ``force=True``. Trino catalogs are connector-managed and are intentionally left alone.
        This does nothing for an in-memory database, for a spec without a lifecycle namespace (a server URL without a
        database, or Trino), and for a dialect without a registered database drop (such as ClickHouse).

    Notes:
        A local file is removed with its ``-wal`` and ``-shm`` files, and a DuckDB file also with its ``.wal`` file.
        A thread that has a connection of HeavenBase's engines to the database in use, such as an open transaction,
        cannot drop it: this raises ``DropRefusedError`` (a ``RuntimeError``) at once. A spec that names nothing to
        remove (an in-memory database, for example) returns before this check, as it removes nothing. A DuckDB file is
        removed only once no connection of HeavenBase's engines to it, or to another name of it, is open: connections
        opened meanwhile wait for the drop before DuckDB opens the file, idle and returned connections are closed, and
        the drop raises ``DropRefusedError`` without removing anything when connections stay in use for 30 seconds.

    Returns:
        None: This function does not return a value.
    """
    if not _drops_namespace(spec):
        return
    from .registry import DatabaseEngineRegistry

    # The drop's statement or file removal would wait for a connection of this thread to the database.
    DatabaseEngineRegistry._need_no_connection(spec)
    if spec.dialect_spec.supports("database-local-path"):
        from ..utils.files import file_uri_to_path

        database = file_uri_to_path(spec.database)
        names = [database, f"{database}-wal", f"{database}-shm"]
        if spec.dialect == "duckdb":
            names.append(f"{database}.wal")
        # DuckDB keeps a database open while any connection to its file is open: close them all first.
        with DatabaseEngineRegistry._closing_duckdb_file(spec):
            for path in names:
                try:
                    # Resolved as delete_file does, so a drop through a symbolic link removes its target.
                    Path(pj(path, abs=True)).unlink()
                except FileNotFoundError:
                    # Absent, or removed meanwhile by the close of a database's last connection (SQLite removes its
                    # -wal and -shm files then, DuckDB its .wal file).
                    pass
        return
    _server_ddl(spec, drop=True, force=force)
