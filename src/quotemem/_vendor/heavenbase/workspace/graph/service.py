"""Workspace GraphEdge command service."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, List, Optional

from .maintenance import check_graph as graph_check
from .maintenance import scrub_dangling as graph_scrub_dangling
from .traversal import shortest_path as graph_shortest_path
from .traversal import traverse as graph_traverse


class _GraphService:
    """Own user-facing GraphEdge traversal and maintenance commands."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def traverse(self, field_ref: Any, start_id: Any, *, min_hops: int = 1, max_hops: int = 1, direction: str = "out") -> List[Dict[str, Any]]:
        """Return records reachable over a GraphEdge field."""
        return graph_traverse(self.workspace, field_ref, start_id, min_hops=min_hops, max_hops=max_hops, direction=direction)

    def shortest_path(self, field_ref: Any, start_id: Any, end_id: Any, *, max_hops: int = 10, direction: str = "out") -> Optional[List[tuple[str, str]]]:
        """Return the shortest typed path over a GraphEdge field, or ``None``."""
        return graph_shortest_path(self.workspace, field_ref, start_id, end_id, max_hops=max_hops, direction=direction)

    def check_graph(self, field_ref: Any = None) -> List[Dict[str, str]]:
        """Report dangling GraphEdge edges."""
        return graph_check(self.workspace, field_ref)

    def scrub_dangling(self, field_ref: Any = None, *, confirm: bool = False) -> Dict[str, Any]:
        """Remove dangling GraphEdge edges when confirmed."""
        return graph_scrub_dangling(self.workspace, field_ref, confirm=confirm)
