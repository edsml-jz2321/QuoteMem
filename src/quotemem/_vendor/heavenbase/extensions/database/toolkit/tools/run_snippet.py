"""MCP tool for executing registered SQL snippets."""

from __future__ import annotations

__all__ = ["build_tool", "run_snippet"]

from heavenbase.utils import Any, Dict, Mapping, Optional, deepcopy
from heavenbase.database.readonly import is_sql_readonly

from ..errors import mcp_raise
from ..policy import database_allow_write, database_result_limit
from ..enable import ensure_database_extension
from ..result import frame_from_response
from ..source import database_from_row


def run_snippet(
    workspace: Any,
    snippet_id: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute one registered db-snippet against its source database.

    Args:
        workspace: Target workspace with ingested database rows.
        snippet_id: ``db-snippet`` object_id.
        params: Optional bind parameters merged over snippet defaults.
        limit: Maximum rows returned to the caller.

    Returns:
        Query result frame dictionary.

    Raises:
        PermissionError: When a write snippet is refused by policy.
        RuntimeError: When execution fails; error text is redacted.
    """
    ensure_database_extension(workspace)
    try:
        snippet = workspace.get(snippet_id, entity="db-snippet")
        if snippet is None:
            raise KeyError(f"Snippet {snippet_id!r} not found")
        database_row = workspace.get(snippet["database_id"], entity="db-database")
        if database_row is None:
            raise KeyError(f"Database {snippet['database_id']!r} not found")
        readonly = bool(snippet.get("readonly", True))
        config = workspace.context.config
        allow_write = database_allow_write(config=config)
        cap = database_result_limit(config=config, limit=limit)
        if not readonly and not allow_write:
            raise PermissionError("Write snippets require heavenbase.database.allow_write=true")
        sql = str(snippet.get("sql") or "").strip()
        if not sql:
            raise ValueError("Snippet SQL is empty")
        bind = deepcopy(snippet.get("params") or {})
        bind.update(dict(params or {}))
        live_db = database_from_row(workspace, database_row)
        if readonly or not allow_write:
            if not is_sql_readonly(sql, dialect=live_db.dialect):
                raise PermissionError("Only read-only SQL is allowed unless heavenbase.database.allow_write=true")
            response = live_db.execute(sql, bind or None, readonly=True)
        else:
            response = live_db.execute(sql, bind or None, readonly=False)
        return {"snippet_id": snippet_id, "readonly": readonly, **frame_from_response(response, limit=cap)}
    except Exception as exc:
        mcp_raise(exc)


def build_tool(workspace: Any) -> Any:
    """Build the ``run_snippet`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to database MCP tools.

    Returns:
        Any: Callable MCP tool function for ``run_snippet``.
    """

    def run_snippet_tool(
        snippet_id: str,
        params: Optional[Mapping[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        return run_snippet(workspace, snippet_id, params=params, limit=limit)

    return run_snippet_tool
