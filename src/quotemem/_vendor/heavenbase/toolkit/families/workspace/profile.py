"""Core workspace MCP profile metadata."""

from __future__ import annotations

__all__ = [
    "WORKSPACE_CORE_TOOL_NAMES",
    "WORKSPACE_QUERY_TOOL_NAMES",
    "WORKSPACE_READ_TOOL_NAMES",
    "WORKSPACE_SCHEMA_TOOL_NAMES",
    "WORKSPACE_WRITE_TOOL_NAMES",
]

WORKSPACE_SCHEMA_TOOL_NAMES = ("define_entity", "list_entities", "describe_entity")
WORKSPACE_WRITE_TOOL_NAMES = ("upsert", "upsert_many", "set", "set_many", "delete", "delete_many")
WORKSPACE_READ_TOOL_NAMES = ("get", "get_many", "exists", "exists_many", "count")
WORKSPACE_QUERY_TOOL_NAMES = ("query", "explain")
WORKSPACE_CORE_TOOL_NAMES = (
    *WORKSPACE_SCHEMA_TOOL_NAMES,
    "upsert",
    "upsert_many",
    "get",
    "get_many",
    "set",
    "set_many",
    "delete",
    "delete_many",
    "exists",
    "exists_many",
    "count",
    *WORKSPACE_QUERY_TOOL_NAMES,
)
