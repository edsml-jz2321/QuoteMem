"""Versioned registry row selection helpers."""

from __future__ import annotations

__all__ = ["select_versioned_row"]

from heavenbase.utils import Any, Dict, Mapping, Optional


def _negative_version_index(version: Optional[str]) -> Optional[int]:
    if version is None:
        return None
    try:
        value = int(str(version).strip())
    except (TypeError, ValueError):
        return None
    return abs(value) - 1 if value < 0 else None


def _version_rank(value: Any) -> tuple[int, int, str]:
    text = str(value or "")
    try:
        return (1, int(text), text)
    except ValueError:
        return (0, 0, text)


def _registry_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row.get("updated_at") or 0, *_version_rank(row.get("version")), str(row.get("object_id") or ""))


def select_versioned_row(rows: list[Dict[str, Any]], version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the selected row from a versioned registry row list.

    Args:
        rows (list[Dict[str, Any]]): Registry rows with optional ``version``, ``updated_at``, and ``object_id`` fields.
        version (Optional[str]): Explicit version string, negative version index such as ``"-1"``, or ``None`` for latest.

    Returns:
        Dict[str, Any] | None: Selected row, or ``None`` when no row matches.
    """
    sorted_rows = sorted(rows, key=_registry_row_key, reverse=True)
    index = _negative_version_index(version)
    if index is not None:
        return sorted_rows[index] if index < len(sorted_rows) else None
    if version is not None:
        sorted_rows = [row for row in sorted_rows if str(row.get("version")) == str(version)]
    return sorted_rows[0] if sorted_rows else None
