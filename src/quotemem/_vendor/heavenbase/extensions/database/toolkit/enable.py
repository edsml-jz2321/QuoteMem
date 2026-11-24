"""Database extension enablement helper for MCP tools."""

from __future__ import annotations

__all__ = ["ensure_database_extension"]

from heavenbase.utils import Any

from ..extension import DATABASE_EXTENSION_ID


def ensure_database_extension(workspace: Any) -> None:
    """Enable the database extension when needed.

    Args:
        workspace (Any): Live workspace that should expose database entities.

    Returns:
        None: This function does not return a value.
    """
    if DATABASE_EXTENSION_ID not in workspace.extensions():
        workspace.enable_extension(DATABASE_EXTENSION_ID)
