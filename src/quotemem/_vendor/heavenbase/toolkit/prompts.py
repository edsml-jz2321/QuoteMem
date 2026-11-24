"""Prompt descriptions owned by the workspace Toolkit domain."""

from __future__ import annotations

__all__ = ["PromptSpec", "WORKSPACE_TOOL_DESCS", "WORKSPACE_TOOLKIT_DESC"]

from heavenbase.utils.typing import Dict, dataclass


@dataclass(frozen=True)
class PromptSpec:
    """Static prompt text bundled as a Python resource.

    Args:
        name (str): Stable dotted resource name.
        text (str): Prompt or tool-description text.
        description (str): Human-facing description of where the prompt is used.

    Returns:
        PromptSpec: Immutable prompt resource descriptor.
    """

    name: str
    text: str
    description: str = ""


WORKSPACE_TOOLKIT_DESC = PromptSpec(
    name="heavenbase.mcp.workspace_toolkit",
    description="System description for a HeavenBase workspace MCP Toolkit.",
    text=(
        "HeavenBase workspace tools over one schema/query surface. Inspect types with list_entities, describe_entity, "
        "or query sys-metaschema; discover objects with sys-catalog (set available=true unless stale/hidden rows are needed); "
        "hydrate catalog hits with get. Use query for reads, upsert/set for writes, define_entity for new schemas, and explain "
        "for routing. Select only fields needed for the answer."
    ),
)

WORKSPACE_TOOL_DESCS: Dict[str, PromptSpec] = {
    "define_entity": PromptSpec(
        "heavenbase.mcp.tool.define_entity",
        (
            "Register an entity schema from JSON, for example {'name': 'Product', 'fields': {...}}. "
            "Adds object_id by default. Accepts JSON-Schema aliases string->short-text, number->float, object->json. "
            "Later calls accept entity id or class-style name."
        ),
    ),
    "list_entities": PromptSpec("heavenbase.mcp.tool.list_entities", "List entity ids registered in this workspace."),
    "describe_entity": PromptSpec("heavenbase.mcp.tool.describe_entity", "Return one entity schema and storage plan."),
    "upsert": PromptSpec("heavenbase.mcp.tool.upsert", "Insert or replace one row for an entity type."),
    "upsert_many": PromptSpec("heavenbase.mcp.tool.upsert_many", "Insert or replace rows for an entity type."),
    "get": PromptSpec("heavenbase.mcp.tool.get", "Fetch one row by object_id."),
    "get_many": PromptSpec("heavenbase.mcp.tool.get_many", "Fetch rows by object_id values."),
    "set": PromptSpec("heavenbase.mcp.tool.set", "Set fields from a row patch."),
    "set_many": PromptSpec("heavenbase.mcp.tool.set_many", "Set fields from row patches."),
    "delete": PromptSpec("heavenbase.mcp.tool.delete", "Delete one target."),
    "delete_many": PromptSpec("heavenbase.mcp.tool.delete_many", "Delete targets in input order."),
    "exists": PromptSpec("heavenbase.mcp.tool.exists", "Return whether one object_id exists."),
    "exists_many": PromptSpec("heavenbase.mcp.tool.exists_many", "Return existence flags for object_id values."),
    "count": PromptSpec("heavenbase.mcp.tool.count", "Return the row count for one entity type."),
    "query": PromptSpec(
        "heavenbase.mcp.tool.query",
        "Run a Mongo-style JSON query. For sys-catalog, set available=true unless stale or hidden rows are explicitly needed.",
    ),
    "explain": PromptSpec("heavenbase.mcp.tool.explain", "Explain storage placement and handlers for a JSON query."),
    "remember": PromptSpec("heavenbase.mcp.tool.remember", "Store one memory note with optional key, title, tags, metadata, and source."),
    "recall": PromptSpec("heavenbase.mcp.tool.recall", "Fetch one memory note by key."),
    "search_memory": PromptSpec(
        "heavenbase.mcp.tool.search_memory",
        "Search memory notes by text match. Returns at most `limit` notes and is not ranked by relevance. When truncated is true, total is the exact number that matched — report total, not the rows you received.",
    ),
    "list_memory": PromptSpec(
        "heavenbase.mcp.tool.list_memory",
        "List memory notes from the configured memory entity. Returns at most `limit` notes, not the newest ones. When truncated is true, total is the exact number that matched — report total, not the rows you received.",
    ),
    "set_memory": PromptSpec("heavenbase.mcp.tool.set_memory", "Patch one memory note by key."),
    "memstate_remember": PromptSpec(
        "heavenbase.mcp.tool.memstate_remember",
        "Store markdown memory for a project as a versioned keypath note.",
    ),
    "memstate_set": PromptSpec("heavenbase.mcp.tool.memstate_set", "Set one short project keypath value and preserve prior versions."),
    "memstate_get": PromptSpec(
        "heavenbase.mcp.tool.memstate_get",
        "Browse current project memories or one keypath subtree. Returns at most `limit` entries. When truncated is true, total is the exact number that matched — report total, not the rows you received.",
    ),
    "memstate_search": PromptSpec(
        "heavenbase.mcp.tool.memstate_search",
        "Search current project memories by text, title, keypath, or tags. Returns at most `limit` matches ordered by keypath, not relevance. When truncated is true, total is the exact number that matched — report total, not the rows you received.",
    ),
    "memstate_history": PromptSpec("heavenbase.mcp.tool.memstate_history", "Return the full version history for one project keypath."),
    "memstate_delete": PromptSpec("heavenbase.mcp.tool.memstate_delete", "Soft-delete one project keypath with a tombstone while preserving history."),
    "run_python": PromptSpec(
        "heavenbase.mcp.tool.run_python",
        "Run Python against the open workspace; hb, ws, and entity classes are bound and a trailing expression is the result. Call python_guide first if unsure of the API. Prefer the query tool when a JSON spec can express the question — the engine validates a spec, a wrong program does not.",
    ),
    "python_guide": PromptSpec("heavenbase.mcp.tool.python_guide", "Return the Python query surface to write code against for run_python."),
    "list_skills": PromptSpec("heavenbase.mcp.tool.list_skills", "List skills available under the active MCP profile, including workspace-guide."),
    "read_skill": PromptSpec(
        "heavenbase.mcp.tool.read_skill",
        "Read a skill file. Use skill_name workspace-guide for live workspace guidance before touching user data.",
    ),
    "run_snippet": PromptSpec(
        "heavenbase.mcp.tool.run_snippet",
        "Execute one registered db-snippet against its source database. Write snippets are refused unless heavenbase.database.allow_write=true.",
    ),
    "suggest_sql": PromptSpec(
        "heavenbase.mcp.tool.suggest_sql",
        "Search ingested db-snippet and db-predicate rows by question text without executing SQL.",
    ),
}
