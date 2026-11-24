"""Explicit Search route selection; no query-intent inference."""

from decimal import Decimal
from typing import Final

from quotemem.core.delivery import MemoryMode
from quotemem.core.recall_policy import VECTOR_ROUTES as VECTOR_ROUTES
from quotemem.core.recall_policy import RecallPolicySnapshotV4, Route

MEMORY_ROUTES: Final[frozenset[Route]] = frozenset({"dense", "lexical", "trigger", "entity"})
CONTENT_ROUTES: Final[frozenset[Route]] = frozenset({"dense", "lexical"})
EVIDENCE_ROUTES: Final[frozenset[Route]] = frozenset({"evidence_dense", "evidence_lexical"})


def select_query_routes(policy: RecallPolicySnapshotV4, mode: MemoryMode) -> tuple[Route, ...]:
    """Return positive-weight routes permitted by the explicit policy and read mode.

    Knowledge has no Evidence channel. The producer must freeze these routes
    and a query vector when any vector route is active; an embedding failure
    is an error, never permission to remove a route.
    """
    return tuple(
        route
        for route in policy.parameters.enabled_routes
        if Decimal(getattr(policy.parameters.route_weights, route)) > 0
        and (mode != "knowledge" or route not in EVIDENCE_ROUTES)
    )
