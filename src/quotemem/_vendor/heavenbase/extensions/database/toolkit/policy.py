"""Database MCP policy helpers."""

from __future__ import annotations

__all__ = ["database_allow_write", "database_result_limit"]

from heavenbase.types import Boolean
from heavenbase.utils import Any, coerce_posint


def database_allow_write(*, config: Any | None = None) -> bool:
    """Return whether write snippets are allowed by one config authority.

    Args:
        config: Context-owned config. Omission selects the process default for
            standalone policy checks.

    Returns:
        Whether database extension snippets may execute writes.
    """
    from heavenbase.context.authority import omitted_config

    manager = omitted_config() if config is None else config
    return bool(Boolean().validate(manager.get("heavenbase.database.allow_write", default=False)))


def database_result_limit(*, config: Any | None = None, limit: Any | None = None) -> int:
    """Return the validated database tool result-row limit.

    Args:
        config: Context-owned config. Omission selects the process default.
        limit: Optional per-call override.

    Returns:
        Positive maximum row count returned by a SQL tool.
    """
    from heavenbase.context.authority import omitted_config

    manager = omitted_config() if config is None else config
    maximum = coerce_posint(manager.get("heavenbase.database.max_query_rows", default=200), "database max_query_rows")
    cap = maximum if limit is None else coerce_posint(limit, "database result limit")
    if cap > maximum:
        raise ValueError(f"database result limit exceeds configured maximum {maximum}")
    return cap
