"""FastMCP transform builders registered by the Toolkit extension."""

from __future__ import annotations

__all__ = ["namespace_transform"]

from heavenbase.utils import Any


def namespace_transform(*, prefix: str) -> Any:
    """Build a FastMCP namespace transform.

    Args:
        prefix (str): Prefix added to exposed component names.

    Returns:
        Any: FastMCP `Namespace` transform.

    Raises:
        ValueError: If `prefix` is empty.
    """
    selected = str(prefix).strip()
    if not selected:
        raise ValueError("MCP namespace transform prefix must be non-empty")
    import fastmcp
    from fastmcp.server.transforms import Namespace

    return Namespace(selected)
