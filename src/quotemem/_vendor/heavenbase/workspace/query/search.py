"""Workspace card-search service."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, Iterable, List, Optional
from heavenbase.utils.values import coerce_posint

from ...execution.frame.result import ResultFrame
from ..catalog.entity import Catalog
from ..metaschema.entity import MetaSchema

_SEARCH_COLUMNS = [
    "object_id",
    "source_entity",
    "kind",
    "target_entity",
    "target_id",
    "subject_id",
    "field",
    "name",
    "desc",
    "type",
    "score",
]


class _SearchService:
    """Search Catalog and MetaSchema card text for one workspace."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def search(
        self,
        card_text: str,
        *,
        limit: Optional[int] = None,
        target_entities: Optional[Iterable[str]] = None,
    ) -> ResultFrame:
        """Return Catalog and MetaSchema cards matching ``card_text``.

        Every scored card is ranked before the configured cap applies, so the
        returned frame carries the exact match count as
        :attr:`~heavenbase.execution.frame.result.ResultFrame.total` and reports
        `truncated` when the cap withheld cards.
        """
        text = str(card_text or "").strip()
        if not text:
            raise ValueError("card_text must not be empty")
        cap = _search_limit(self.workspace.context.config, limit)
        raw_targets = [target_entities] if isinstance(target_entities, str) else target_entities
        targets = None if raw_targets is None else {str(entity).strip() for entity in raw_targets if str(entity).strip()}
        if target_entities is not None and not targets:
            raise ValueError("target_entities must select at least one entity")
        rows = self._catalog_cards(text, targets=targets) + ([] if targets is not None else self._metaschema_cards(text))
        rows = [row for row in rows if row["score"] > 0]
        rows.sort(key=lambda row: (-row["score"], str(row["kind"]), str(row["name"]), str(row["object_id"])))
        # slice through the frame contract so the cut records what it withheld
        return ResultFrame.from_rows(rows, _SEARCH_COLUMNS).limit(cap)

    def _catalog_cards(self, text: str, *, targets: Optional[set[str]] = None) -> List[Dict[str, Any]]:
        rows = []
        for row in self.workspace._query.query(Catalog).available().execute().rows():
            if targets is not None and row.get("target_entity") not in targets:
                continue
            score = _score(
                text,
                [
                    (100, row.get("name")),
                    (80, row.get("desc")),
                    (50, row.get("summary")),
                    (45, row.get("keywords")),
                    (45, row.get("tags")),
                    (30, row.get("type")),
                    (25, row.get("target_entity")),
                    (20, row.get("target_id")),
                ],
            )
            rows.append(
                {
                    "object_id": row.get("object_id"),
                    "source_entity": Catalog.schema().entity_id,
                    "kind": "object",
                    "target_entity": row.get("target_entity", ""),
                    "target_id": row.get("target_id", ""),
                    "subject_id": "",
                    "field": "",
                    "name": row.get("name", ""),
                    "desc": row.get("desc", ""),
                    "type": row.get("type", ""),
                    "score": score,
                }
            )
        return rows

    def _metaschema_cards(self, text: str) -> List[Dict[str, Any]]:
        rows = []
        for row in self.workspace._query.query(MetaSchema).where({"active": True}).execute().rows():
            score = _score(
                text,
                [
                    (90, row.get("name")),
                    (80, row.get("desc")),
                    (45, row.get("tags")),
                    (35, row.get("subject_id")),
                    (35, row.get("field")),
                    (30, row.get("kind")),
                    (25, row.get("dtype")),
                    (25, row.get("type")),
                    (25, row.get("backend")),
                    (25, row.get("strategy")),
                ],
            )
            rows.append(
                {
                    "object_id": row.get("object_id"),
                    "source_entity": MetaSchema.schema().entity_id,
                    "kind": row.get("kind", ""),
                    "target_entity": "",
                    "target_id": "",
                    "subject_id": row.get("subject_id", ""),
                    "field": row.get("field", ""),
                    "name": row.get("name", ""),
                    "desc": row.get("desc", ""),
                    "type": row.get("type", ""),
                    "score": score,
                }
            )
        return rows


def _search_limit(config: Any, limit: Optional[int]) -> int:
    value = int(config.get("heavenbase.search.default_limit", default=20)) if limit is None else limit
    cap = coerce_posint(value, "limit")
    max_limit = int(config.get("heavenbase.search.max_limit", default=200))
    if cap > max_limit:
        raise ValueError(f"limit exceeds configured maximum {max_limit}")
    return cap


def _score(query: str, weighted_values: Iterable[tuple[int, Any]]) -> int:
    needle = _normalize(query)
    terms = [term for term in needle.split(" ") if term]
    best = 0
    for weight, value in weighted_values:
        haystack = _normalize(_card_text(value))
        if not haystack:
            continue
        if haystack == needle:
            best = max(best, weight + 30)
        elif needle in haystack:
            best = max(best, weight)
        elif terms and all(term in haystack for term in terms):
            best = max(best, max(1, weight - 10))
        elif terms and any(term in haystack for term in terms):
            best = max(best, max(1, weight // 2))
    return best


def _card_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_card_text(item) for pair in value.items() for item in pair)
    if isinstance(value, (list, tuple, set)):
        return " ".join(_card_text(item) for item in value)
    return str(value)


def _normalize(value: str) -> str:
    return " ".join(str(value or "").lower().split())
