"""Database source resolution for snippet execution."""

from __future__ import annotations

__all__ = ["database_from_row"]

from heavenbase.utils import Any, Mapping
from heavenbase.database import Database

from ..source import _database_authority


def database_from_row(workspace: Any, row: Mapping[str, Any]) -> Database:
    """Rebuild a live Database instance from a persisted db-database row.

    Args:
        workspace (Any): Live workspace owning database extension backends.
        row (Mapping[str, Any]): Persisted ``db-database`` row.

    Returns:
        Database: Reconnected database facade.

    Raises:
        ValueError: If the row has no reconnectable backend or source.
    """
    database_id = str(row.get("object_id") or "")
    extension_api = getattr(workspace, "database", None)
    live_source = getattr(extension_api, "_source", lambda object_id: None)(database_id)
    if live_source is not None:
        return live_source
    backend_ref = str(row.get("backend_ref") or "").strip()
    if backend_ref:
        backend = workspace.backends.get(backend_ref)
        live_db = getattr(backend, "db", None)
        if live_db is not None:
            return live_db
    source = dict(row.get("source") or {})
    if source:
        if any(str(value).strip() in {"***", "<redacted>"} for value in source.values()):
            raise ValueError(f"Database row {database_id!r} contains redacted credentials; re-ingest it in this process or use a named workspace backend")
        return Database(**{**source, **_database_authority(workspace)})
    raise ValueError(f"Database row {row.get('object_id')!r} has no reconnectable source")
