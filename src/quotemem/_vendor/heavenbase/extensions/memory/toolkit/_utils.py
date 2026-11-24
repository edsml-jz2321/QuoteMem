"""Private defaults shared by flat memory MCP tools."""

from __future__ import annotations

from time import time

from heavenbase.utils import Any

MEMORY_SELECT = [
    "text",
    "title",
    "tags",
    "metadata",
    "source",
    "project_id",
    "keypath",
    "version",
    "previous_id",
    "deleted",
    "created_at",
    "updated_at",
]


def default_source(*, config: Any | None = None) -> str:
    """Return the memory producer default from the selected config authority."""
    from heavenbase.context.authority import omitted_config

    manager = omitted_config() if config is None else config
    return str(manager.get("heavenbase.mcp.memory.default_source", default="agent") or "agent")


def list_limit(*, config: Any | None = None) -> int:
    """Return the memory listing limit from the selected config authority."""
    from heavenbase.context.authority import omitted_config

    manager = omitted_config() if config is None else config
    return int(manager.get("heavenbase.mcp.memory.list_limit", default=20) or 20)


def search_limit(*, config: Any | None = None) -> int:
    """Return the memory search limit from the selected config authority."""
    from heavenbase.context.authority import omitted_config

    manager = omitted_config() if config is None else config
    return int(manager.get("heavenbase.mcp.memory.search_limit", default=10) or 10)


def now() -> int:
    return int(time())
