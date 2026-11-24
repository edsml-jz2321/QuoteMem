"""Generated workspace-guide skill text for MCP clients."""

from __future__ import annotations

__all__ = ["WORKSPACE_GUIDE_SKILL", "entity_summaries", "render_workspace_guide"]

from heavenbase.utils import Any, List

from .context import WorkspaceToolContext

WORKSPACE_GUIDE_SKILL = "workspace-guide"


def entity_summaries(workspace: Any, entity_scope: Any = None) -> List[str]:
    """Return readable entity summary lines for toolkit descriptions.

    Args:
        workspace (Any): HeavenBase workspace whose registered entities are summarized.
        entity_scope (Any): Closed entity allowlist, ``"*"``, or ``None`` for all entities. Defaults to ``None``.

    Returns:
        List[str]: ``entity_id`` or ``entity_id: description`` lines in sorted entity-id order.
    """
    entity_ids = sorted(workspace.entities)
    if entity_scope not in (None, "*"):
        allowed = frozenset(entity_scope)
        entity_ids = [entity_id for entity_id in entity_ids if entity_id in allowed]
    summaries: List[str] = []
    for entity_id in entity_ids:
        schema = workspace.entities[entity_id].schema()
        desc = str(schema.desc or "").strip()
        summaries.append(f"{entity_id}: {desc}" if desc else entity_id)
    return summaries


def render_workspace_guide(context: WorkspaceToolContext) -> str:
    """Render the generated workspace-guide skill for one MCP context.

    Args:
        context (WorkspaceToolContext): Shared workspace, profile, and selected tool names.

    Returns:
        str: Markdown workspace-guide skill document for MCP clients.
    """
    workspace = context.workspace
    profile = context.profile
    tool_names = list(context.tool_names)
    entities = entity_summaries(workspace, profile.entities)
    skills_scope = profile.skills if profile.skills else "none"
    entity_lines = "\n".join(f"- {line}" for line in entities) or "- (no registered entities yet)"
    return "\n".join(
        [
            "---",
            f"name: {WORKSPACE_GUIDE_SKILL}",
            'description: "Live workspace guide for MCP agents."',
            "---",
            "",
            "# Workspace Guide",
            "",
            f"Workspace id: `{workspace.id}`",
            "",
            "## Active MCP profile",
            "",
            f"- Profile: `{profile.name}`",
            f"- Serializer: `{profile.serializer}`",
            f"- Entity scope: `{profile.entities}`",
            f"- Skills scope: `{skills_scope}`",
            f"- Tools: {', '.join(tool_names)}",
            "",
            "## Registered entities",
            "",
            entity_lines,
            "",
            "## Query JSON shape",
            "",
            "Use the `query` tool with Mongo-style specs:",
            "",
            "```json",
            "{",
            '  "filter": {"$and": [{"status": "open"}, {"summary": {"$match": "refund"}}]},',
            '  "select": ["summary", "status"],',
            '  "limit": 10',
            "}",
            "```",
            "",
            "Common operators: `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`, `$in`, `$match`, `$like`, `$contains`.",
            "",
            "For engine-side aggregation add `group_by` plus `aggregate` (one `{op: {field, ...options}}` per metric; registered ops include `count`/`min`/`max`/`sum`/`avg`; not combinable with `select`):",
            "",
            "```json",
            "{",
            '  "filter": {"status": "open"},',
            '  "group_by": ["region"],',
            '  "aggregate": {"tickets": {"count": {"field": "object_id"}}, "worst": {"max": {"field": "severity"}}},',
            '  "order": ["-tickets"]',
            "}",
            "```",
            "",
            'For `sys-catalog` discovery, set top-level `"available": true` unless stale or hidden rows are required.',
            "",
            "## Json fields",
            "",
            "Json fields hold open payloads. Filter nested values with dotted paths or the explicit `$get` form:",
            "",
            "```json",
            '{"filter": {"meta.venue": "NeurIPS", "meta.score": {"$gte": 8}}}',
            '{"filter": {"meta": {"$get": {"path": "review.verdict", "eq": "accept"}}}}',
            "```",
            "",
            "Path comparisons support `eq`, `ne`, `in`, `lt`, `lte`, `gt`, `gte`, and string-only `contains`; a",
            "missing or null path matches nothing. Path operands must have a compatible JSON type, so booleans do not",
            "equal numbers and heterogeneous `ne` comparisons also match nothing. Whole-value `$eq`/`$ne`/`$in`",
            "compare the entire payload with the same strict JSON equality. Other text operators do not apply to Json",
            "fields. When the interesting names live in the data rather than the query, fold the whole column with a",
            "registered reducer instead of naming them. Pick by what you are counting:",
            "",
            "| aggregate | payload | answers |",
            "|---|---|---|",
            "| `key_sums` | object | total of the values under each key |",
            "| `key_counts` | object | how many rows carry each key |",
            "| `value_counts` | list or scalar | how often each element occurs |",
            "",
            "```json",
            '{"aggregate": {"totals": {"key_sums": {"field": "tool_calls"}}}}',
            '{"aggregate": {"tags": {"value_counts": {"field": "keywords"}}}}',
            '{"group_by": ["region"], "aggregate": {"totals": {"key_sums": {"field": "tool_calls"}}}}',
            "```",
            "",
            "All three cover every row exactly, and the wrong one for a payload raises rather than returning an empty",
            "mapping, because empty and none-found are indistinguishable to a reader. The result is one mapping per",
            "group, so `group_by` on declared columns works normally while `order` and `having` cannot reach inside",
            "it — rank in the caller. Register your own fold with `hb.ext.register_aggregate_reducer` when no shape",
            "fits; it takes an importable module-level function, not a lambda. For a one-off question not worth",
            "registering, `code_reduce` carries the expression itself — it executes in the query process, which is",
            "allowed by default and refused where `heavenbase.query.allow_code_reduce` is set to false:",
            "",
            "```json",
            '{"aggregate": {"spread": {"code_reduce": {"field": "amount", "code": "lambda v: max(v) - min(v)"}}}}',
            '{"aggregate": {"mean": {"code_reduce": {"field": "amount", "code": "lambda v: sum(v)/len(v)", "dtype": "float"}}}}',
            "```",
            "",
            "`dtype` declares the output column and the result normalizes into it, the way a declared field normalizes",
            "a write; omit it and the output is untyped, which `having` then compares without normalization.",
            "",
            "Dotted paths also work in `group_by`, `order`, and `select`:",
            "",
            "```json",
            '{"group_by": ["meta.venue"], "aggregate": {"n": {"count": {"field": "object_id"}}}, "order": ["-n"]}',
            "```",
            "",
            "The output column is named by the path itself, rows with a missing or null path keep a `None` value",
            "(grouping under `None`, sorting last in both directions), and dotted clauses always run on the portable",
            "Python route instead of native pushdown — fine for moderate data. Lift a hot nested value into a",
            "declared computed field",
            '(`hb.field(hb.ShortText).compute(lambda meta: (meta or {}).get("venue", ""), inputs=["meta"])`)',
            "when native performance matters. `describe_entity` lists each field's supported `filter_ops`.",
            "",
            "## Object id semantics",
            "",
            "- Every entity row has exactly one primary key field: `object_id` (logical type `Identifier`, max 63 chars).",
            "- `entity_id` / entity type names identify schemas; `object_id` identifies rows.",
            "- Catalog rows use `target_entity` + `target_id` to point at concrete rows; hydrate with `get(object_id=target_id, entity=target_entity)`.",
            "- Delete/get/set/upsert tools accept raw object ids or `{entity, object_id}` targets when disambiguation is needed.",
        ]
    )
