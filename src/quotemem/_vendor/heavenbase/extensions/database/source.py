"""Resolve external database ingest sources into live database handles."""

from __future__ import annotations

__all__ = ["ResolvedDatabase", "redact_dsn_summary", "redacted_source", "resolve_database"]

from heavenbase.utils import Any, CM_HVNB, Dict, Mapping, dataclass
from heavenbase.database import DB_ENGINE, DBSpec, Database
from heavenbase.utils.files import exists_file, get_file_ext
from heavenbase.utils.files import path_to_file_uri


@dataclass(frozen=True)
class ResolvedDatabase:
    """Live database selected for ingest plus its optional workspace backend reference."""

    db: Database
    backend_ref: str = ""


def _database_authority(workspace: Any) -> Dict[str, Any]:
    """Return paired resolver/config bindings for a real workspace.

    Lightweight standalone callers may supply an object with neither
    authority. A partial binding is rejected because it would mix one
    Context's module catalog with process-default configuration, or vice versa.
    """
    resolver = getattr(workspace, "module_resolver", None)
    config = getattr(getattr(workspace, "context", None), "config", None)
    if resolver is None and config is None:
        return {}
    if resolver is None or config is None:
        raise TypeError("Database extension workspace must expose both module_resolver and context.config")
    return {"resolver": resolver, "config": config}


def redact_dsn_summary(spec: DBSpec) -> str:
    """Return a secret-redacted connection summary for one database spec.

    Args:
        spec: Database configuration spec.

    Returns:
        URL-like summary with passwords redacted, or a redacted config summary when URL materialization is empty.
    """
    return str(DB_ENGINE.apply(spec, profile="print")["uri"])


def redacted_source(spec: DBSpec) -> Dict[str, Any]:
    """Return a diagnostic database config dictionary with secrets removed.

    Args:
        spec: Database configuration spec.

    Returns:
        Dictionary safe to store in ``db-database.source``.
    """
    return dict(DB_ENGINE.apply(spec, profile="print")["data"])


def resolve_database(workspace: Any, db: Any, *, backend_ref: str = "") -> ResolvedDatabase:
    """Resolve one ingest input into a live database handle.

    Args:
        workspace: Workspace that may own named SQL backends.
        db: ``Database`` instance, config mapping, workspace backend name, DSN URL, or local SQLite path.
        backend_ref: Optional backend reference carried through recursive or caller-owned resolution.

    Returns:
        Resolved live database and the workspace backend name when one was selected.

    Raises:
        ValueError: If a named workspace backend exists but does not expose a SQL database.
    """
    authority = _database_authority(workspace)
    if isinstance(db, Database):
        return ResolvedDatabase(db=db, backend_ref=backend_ref)
    if isinstance(db, Mapping):
        backend = workspace.backends.get(str(db.get("name") or "")) if db.get("type") and hasattr(workspace, "backends") else None
        if backend is not None and getattr(backend, "db", None) is not None:
            return ResolvedDatabase(db=backend.db, backend_ref=str(db.get("name") or backend_ref))
        return ResolvedDatabase(db=Database(**{**dict(db), **authority}), backend_ref=backend_ref)

    key = str(db).strip()
    if hasattr(workspace, "backends"):
        try:
            backend = workspace.backends.get(key)
        except Exception:
            backend = None
        if backend is not None:
            live_db = getattr(backend, "db", None)
            if live_db is None:
                raise ValueError(f"Workspace backend {key!r} does not expose a SQL database")
            return ResolvedDatabase(db=live_db, backend_ref=key)

    if key.startswith("file://"):
        return ResolvedDatabase(db=Database(provider="sqlite", database=key, **authority), backend_ref=backend_ref)
    if "://" in key:
        return ResolvedDatabase(db=Database(url=key, **authority), backend_ref=backend_ref)

    manager = authority.get("config", CM_HVNB)
    database = manager.pj("%", key, abs=True)
    if not database.startswith("file:") and (get_file_ext(key) == "db" or exists_file(key)):
        database = path_to_file_uri(database)
    return ResolvedDatabase(db=Database(provider="sqlite", database=database, **authority), backend_ref=backend_ref)
