"""Result formatting helpers for memory MCP tools."""

from __future__ import annotations

__all__ = ["frame_dict"]

from heavenbase.utils import Any, Dict


def frame_dict(frame: Any) -> Dict[str, Any]:
    """Return the stable MCP payload for a memory query frame.

    Args:
        frame (Any): Query result frame exposing ``rows``, ``col_names``,
            ``total``/``truncated``, and optional identity metadata.

    Returns:
        Dict[str, Any]: MCP payload with ``columns``, ``rows``, ``n``, and
        optional ``total``/``truncated`` when the query limit withheld rows,
        plus optional identity fields. ``n`` counts returned rows and ``total``
        counts every row that matched, so a caller never mistakes a page for a
        whole.
    """
    rows = frame.rows()
    payload = {"columns": frame.col_names(), "rows": rows, "n": len(rows)}
    if getattr(frame, "truncated", False):
        payload["total"] = frame.total
        payload["truncated"] = True
    if getattr(frame, "has_identity", True):
        payload["ids"] = list(frame.ids)
    identity_column = getattr(frame, "identity_column", "object_id")
    if identity_column != "object_id":
        payload["identity_column"] = identity_column
    return payload
