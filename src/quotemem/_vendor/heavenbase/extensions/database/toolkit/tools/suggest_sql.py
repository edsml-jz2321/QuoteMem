"""MCP tool for finding registered SQL snippets and predicates."""

from __future__ import annotations

__all__ = ["build_tool", "suggest_sql"]

from heavenbase.utils import Any, Dict, List, Optional

from ..errors import mcp_raise
from ..enable import ensure_database_extension

_SEARCH_LIMIT = 10


def _rule_trigger_match(question: str, row: Dict[str, Any]) -> bool:
    """Return whether one rule explicitly names a phrase in the question."""
    needle = " ".join(str(question or "").lower().split())
    terms = [
        *list(row.get("triggers") or []),
        *list(row.get("synonyms") or []),
    ]
    return bool(needle) and any((term := " ".join(str(value or "").lower().split())) and (term in needle or needle in term) for value in terms)


def _suggest_rules(workspace: Any, text: str, cap: int) -> tuple[List[Dict[str, Any]], int, bool]:
    """Return card-ranked rules plus exact literal trigger matches."""
    max_search = int(workspace.context.config.get("heavenbase.search.max_limit", default=200))
    frame = workspace.search(text, target_entities=["db-rule"], limit=max(cap, max_search))
    ranked = [row for card in frame.rows() if (row := workspace.get(card["target_id"], entity="db-rule")) is not None and row.get("enabled", True)]
    seen = {str(row["object_id"]) for row in ranked}
    # Rules are a deliberately small durable memory lane. Literal triggers are
    # scanned so an ordinary long or unsegmented question (notably Chinese)
    # can still retrieve an explicitly authored phrase even when whole-card
    # search treats the question as one unmatched token.
    if not frame.truncated:
        for row in workspace.query("db-rule").where({"enabled": True}).execute().rows():
            object_id = str(row["object_id"])
            if object_id not in seen and _rule_trigger_match(text, row):
                ranked.append(row)
                seen.add(object_id)
    total = frame.total if frame.truncated and frame.total is not None else len(ranked)
    return ranked[:cap], int(total), bool(frame.truncated or total > cap)


def suggest_sql(workspace: Any, question: str, *, limit: Optional[int] = None) -> Dict[str, Any]:
    """Return matching snippets, predicates, and enabled rules for one question.

    Args:
        workspace: Target workspace with ingested database rows.
        question: Free-text question used for card-text search.
        limit: Maximum matches per entity kind.

    Returns:
        Dict[str, Any]: ``{"question", "snippets", "predicates", "rules", "n", "total",
        "truncated", "limit"}``. The cap applies per entity kind, so ``n`` counts
        the rows returned while ``total`` counts every card that matched; a
        caller reporting a match count must report ``total``, not ``n``.
    """
    ensure_database_extension(workspace)
    try:
        cap = int(limit if limit is not None else _SEARCH_LIMIT)
        text = str(question or "").strip()
        snippets: List[Dict[str, Any]] = []
        predicates: List[Dict[str, Any]] = []
        rules: List[Dict[str, Any]] = []
        total = 0
        truncated = False
        if text:
            found: Dict[str, List[Dict[str, Any]]] = {}
            for entity_id in ("db-snippet", "db-predicate"):
                frame = workspace.search(text, target_entities=[entity_id], limit=cap)
                rows = [row for card in frame.rows() if (row := workspace.get(card["target_id"], entity=entity_id)) is not None]
                found[entity_id] = rows
                total += frame.total if frame.total is not None else len(rows)
                truncated = truncated or frame.truncated
            rules, rule_total, rule_truncated = _suggest_rules(workspace, text, cap)
            snippets = found["db-snippet"]
            predicates = found["db-predicate"]
            total += rule_total
            truncated = truncated or rule_truncated
        return {
            "question": text,
            "snippets": snippets,
            "predicates": predicates,
            "rules": rules,
            "n": len(snippets) + len(predicates) + len(rules),
            "total": total,
            "truncated": truncated,
            "limit": cap,
        }
    except Exception as exc:
        mcp_raise(exc)


def build_tool(workspace: Any) -> Any:
    """Build the ``suggest_sql`` tool for one workspace.

    Args:
        workspace (Any): Live workspace bound to database MCP tools.

    Returns:
        Any: Callable MCP tool function for ``suggest_sql``.
    """

    def suggest_sql_tool(question: str, limit: Optional[int] = None) -> Dict[str, Any]:
        return suggest_sql(workspace, question, limit=limit)

    return suggest_sql_tool
