"""MCP tool for executing ad hoc read-only SQL."""

from __future__ import annotations

__all__ = ["build_tool", "run_sql"]

from heavenbase.database.readonly import is_sql_readonly
from heavenbase.utils import Any, Dict, Mapping, Optional

from ..enable import ensure_database_extension
from ..errors import mcp_raise
from ..policy import database_result_limit
from ..result import frame_from_response
from ..source import database_from_row


def run_sql(
    workspace: Any,
    database_id: str,
    sql: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute arbitrary read-only SQL against one ingested database.

    Args:
        workspace: Target workspace with ingested database rows.
        database_id: Target ``db-database`` object_id.
        sql: Read-only SQL statement.
        params: Optional bind parameters.
        limit: Maximum rows returned to the caller.

    Returns:
        Bounded query result frame dictionary.

    Raises:
        PermissionError: When SQL is not read-only.
        RuntimeError: When execution fails; error text is redacted.
    """
    ensure_database_extension(workspace)
    try:
        database_row = workspace.get(database_id, entity="db-database")
        if database_row is None:
            raise KeyError(f"Database {database_id!r} not found")
        statement = str(sql or "").strip()
        if not statement:
            raise ValueError("SQL is empty")
        live_db = database_from_row(workspace, database_row)
        if not is_sql_readonly(statement, dialect=live_db.dialect):
            raise PermissionError("run_sql accepts read-only SQL only")
        cap = database_result_limit(config=workspace.context.config, limit=limit)
        response = live_db.execute(statement, dict(params or {}) or None, readonly=True)
        return {"database_id": database_id, "readonly": True, **frame_from_response(response, limit=cap)}
    except Exception as exc:
        mcp_raise(exc)


def build_tool(workspace: Any) -> Any:
    """Build the ``run_sql`` tool for one workspace."""

    def run_sql_tool(
        database_id: str,
        sql: str,
        params: Optional[Mapping[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        return run_sql(workspace, database_id, sql, params=params, limit=limit)

    return run_sql_tool
