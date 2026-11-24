"""Result formatting helpers for database MCP tools."""

from __future__ import annotations

__all__ = ["frame_from_response"]

from heavenbase.utils import Any, Dict


def frame_from_response(response: Any, *, limit: int) -> Dict[str, Any]:
    """Return the stable MCP payload for a database response.

    Args:
        response (Any): Database query response exposing ``to_dicts`` and
            optional ``columns``.
        limit: Maximum rows returned in the payload.

    Returns:
        Dict[str, Any]: MCP payload with columns, bounded rows, and exact total.
    """
    rows = response.to_dicts()
    columns = list(rows[0].keys()) if rows else list(getattr(response, "columns", []) or [])
    page = rows[:limit]
    return {"columns": columns, "rows": page, "n": len(page), "total": len(rows), "truncated": len(rows) > limit, "limit": limit}
