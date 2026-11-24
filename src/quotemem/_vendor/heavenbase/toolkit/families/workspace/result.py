"""Result formatting helpers for workspace MCP tools."""

from __future__ import annotations

__all__ = ["frame_dict"]

from heavenbase.utils import Any, Dict


def frame_dict(frame: Any, *, near_bounded: bool = False) -> Dict[str, Any]:
    """Return the stable MCP payload for a result frame.

    Args:
        frame (Any): Query or workspace result frame with ``to_dict``, ``rows``, and optional identity metadata.
        near_bounded (bool): Whether a ``near`` clause bounded the rows this
            frame was built from. Defaults to ``False``.

    Returns:
        Dict[str, Any]: JSON-safe payload with ``columns``, ``rows``, ``n``, and
        optional ``total``/``truncated`` when pagination withheld rows,
        ``near_bounded``, ``ids``, or ``identity_column``. ``n`` counts returned
        rows and ``total`` counts every row the query produced, so a caller
        never mistakes a page for a whole — except under ``near_bounded``, where
        the query itself only ever saw its top-k window.
    """
    payload = {"columns": frame.to_dict()["columns"], "rows": frame.rows(), "n": len(frame)}
    if getattr(frame, "truncated", False):
        payload["total"] = frame.total
        payload["truncated"] = True
    if near_bounded:
        # A near clause feeds the pipeline only its top-k window, so any total
        # taken downstream counts that window rather than the matching rows.
        payload["near_bounded"] = True
    if getattr(frame, "has_identity", True):
        payload["ids"] = list(frame.ids)
    identity_column = getattr(frame, "identity_column", "object_id")
    if identity_column != "object_id":
        payload["identity_column"] = identity_column
    return payload
