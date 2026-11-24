"""MCP tool for verifying and saving successful read-only SQL."""

from __future__ import annotations

__all__ = ["build_tool", "save_snippet"]

from collections.abc import Iterable
from time import time

from heavenbase.utils import Any, Dict, Mapping, Optional

from ...rows import snippet_id
from ..errors import mcp_raise
from .run_sql import run_sql


def save_snippet(
    workspace: Any,
    database_id: str,
    title: str,
    sql: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    card_text: str = "",
    tables: Optional[Iterable[str]] = None,
    synonyms: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify read-only SQL and persist it only after successful execution.

    Args:
        workspace: Target workspace with ingested database rows.
        database_id: Target ``db-database`` object_id.
        title: Stable human-readable snippet title.
        sql: Read-only SQL statement.
        params: Optional default bind parameters.
        card_text: Searchable natural-language description.
        tables: Optional qualified relation names used by the SQL.
        synonyms: Optional alternate question phrases.
        limit: Maximum verification rows returned to the caller.

    Returns:
        Persisted snippet id and successful verification frame.
    """
    try:
        name = str(title or "").strip()
        if not name:
            raise ValueError("Snippet title is empty")
        statement = str(sql or "").strip()
        result = run_sql(workspace, database_id, statement, params=params, limit=limit)
        database_row = workspace.get(database_id, entity="db-database")
        if database_row is None:
            raise KeyError(f"Database {database_id!r} not found")
        object_id = snippet_id(database_id, name)
        workspace.upsert(
            "db-snippet",
            {
                "object_id": object_id,
                "database_id": database_id,
                "database_name": str(database_row.get("name") or ""),
                "dialect": str(database_row.get("dialect") or ""),
                "title": name,
                "sql": statement,
                "params": dict(params or {}),
                "readonly": True,
                "card_text": str(card_text or "").strip(),
                "tables": list(dict.fromkeys(str(item).strip() for item in (tables or []) if str(item).strip())),
                "synonyms": list(dict.fromkeys(str(item).strip() for item in (synonyms or []) if str(item).strip())),
                "verified": True,
                "last_succeeded_at": int(time() * 1000),
            },
        )
        return {"snippet_id": object_id, "verified": True, "result": result}
    except Exception as exc:
        mcp_raise(exc)


def build_tool(workspace: Any) -> Any:
    """Build the ``save_snippet`` tool for one workspace."""

    def save_snippet_tool(
        database_id: str,
        title: str,
        sql: str,
        params: Optional[Mapping[str, Any]] = None,
        card_text: str = "",
        tables: Optional[list[str]] = None,
        synonyms: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        return save_snippet(
            workspace,
            database_id,
            title,
            sql,
            params=params,
            card_text=card_text,
            tables=tables,
            synonyms=synonyms,
            limit=limit,
        )

    return save_snippet_tool
