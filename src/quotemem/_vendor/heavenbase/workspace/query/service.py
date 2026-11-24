"""Workspace query, search, execute, and explain command service."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, Iterable, Mapping, Optional

from ...execution.frame.result import ResultFrame
from ...query.json import load as load_json_query
from ...query.spec import QuerySpec, _bind_query_policy
from ...query.value import Query


class _QueryService:
    """Own user-facing query command assembly for one workspace."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def query(self, entity: Any) -> Query:
        """Return an empty immutable Python query for a registered entity."""
        ws = self.workspace
        entity_cls = ws._schema.require(entity)
        return Query(ws, entity_cls)

    def query_json(self, entity: Any, spec: Mapping[str, Any]) -> Query:
        """Return an immutable query loaded from a JSON query spec."""
        ws = self.workspace
        entity_cls = ws._schema.require(entity)
        return load_json_query(ws, entity_cls, spec)

    def search(
        self,
        card_text: str,
        *,
        limit: Optional[int] = None,
        target_entities: Optional[Iterable[str]] = None,
    ) -> ResultFrame:
        """Search workspace cards by free text."""
        return self.workspace._search.search(card_text, limit=limit, target_entities=target_entities)

    def execute(self, query: QuerySpec) -> ResultFrame:
        """Execute a normalized query request."""
        self.workspace._schema.require(query.entity_id)
        return self.workspace._router.execute(_bind_query_policy(query, self.workspace.context.config))

    def explain(self, query: QuerySpec) -> Dict[str, Any]:
        """Return storage, handler, and fallback diagnostics for a query."""
        self.workspace._schema.require(query.entity_id)
        return self.workspace._query_explain.explain(_bind_query_policy(query, self.workspace.context.config))
