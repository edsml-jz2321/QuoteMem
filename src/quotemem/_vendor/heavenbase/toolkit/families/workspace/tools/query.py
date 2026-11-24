"""MCP tool for running workspace queries."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict, Optional, loads_json

from ..context import WorkspaceToolContext
from ..errors import mcp_raise
from ..result import frame_dict


def build_tool(context: WorkspaceToolContext) -> Any:
    """Build the ``query`` tool for one workspace context.

    Args:
        context (WorkspaceToolContext): Shared workspace and profile state for the tool factory.

    Returns:
        Any: MCP-callable ``query`` function bound to ``context``.
    """

    def query(entity: str, spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a Mongo-style JSON query for one entity.

        Supports plain reads and engine-side aggregation. Spec keys: ``filter``
        (Mongo-style operators such as ``$eq``/``$gte``/``$in``/``$match``;
        nested Json payload values filter with dotted paths such as
        ``{"meta.venue": "NeurIPS"}`` or ``{"meta.review.score": {"$gte": 8}}``),
        ``select``, ``order`` (list of field-name strings,
        ``-`` prefix for descending, for example ``{"order": ["-year"]}``),
        ``offset``, ``limit``, ``available``, ``near`` (which feeds the rest
        of the pipeline only its top-k nearest rows, so the result carries
        ``near_bounded`` and any count describes that window),
        ``traverse``, plus ``group_by`` (list of fields) with ``aggregate`` mapping each
        output name to exactly one ``{op: {"field": field, ...options}}`` where op is
        a registered reducer such as ``count``/``min``/``max``/``sum``/``avg``, for
        example ``{"group_by": ["region"], "aggregate": {"revenue": {"sum":
        {"field": "amount"}}}}``.
        When the names of interest live in the data rather than the query, fold a
        whole Json column instead of naming them: ``key_sums`` totals the values
        under each key of an object payload, ``key_counts`` counts the rows
        carrying each key, and ``value_counts`` counts occurrences of each
        element in a list payload. For example
        ``{"aggregate": {"totals": {"key_sums": {"field": "tool_calls"}}}}``
        returns one mapping covering every row exactly. Using the wrong one for
        a payload raises rather than returning an empty mapping. ``group_by`` on
        declared columns works normally; ``order`` and ``having`` cannot reach
        inside the returned mapping.
        For a one-off reduction with no registered name, ``code_reduce`` carries
        the expression: ``{"spread": {"code_reduce": {"field": "amount", "code":
        "lambda v: max(v) - min(v)", "dtype": "float"}}}``. It executes code in
        the query process, which is allowed by default and refused where
        ``heavenbase.query.allow_code_reduce`` is false. ``dtype`` is optional
        and declares the output column, normalizing the result into it.
        ``select`` cannot be combined with ``aggregate``; without ``group_by``
        an aggregate returns one scalar row.
        Dotted Json paths also work in ``group_by``, ``order``, and ``select``
        (``{"group_by": ["meta.venue"], "aggregate": {"n": {"count": {"field":
        "object_id"}}}}``): the output column is named by the path, a missing
        or null path yields ``None`` (grouping under ``None``, sorting last),
        and dotted clauses run on the portable Python route rather than native
        pushdown — fine for moderate data, slower on very large tables.

        Args:
            entity: Entity id, class name, or display name to resolve.
            spec: Optional Mongo-style query specification.

        Returns:
            Result frame as columns, rows, ids, and row count.

        Raises:
            RuntimeError: If routing or execution fails; the message is redacted for MCP use.
        """
        try:
            if isinstance(spec, str):  # some models double-encode the nested spec argument
                spec = loads_json(spec)
            resolved = context.require_entity(entity)
            spec = spec or {}
            # A near clause feeds the pipeline only its top-k window, so a total
            # taken downstream counts that window, not the matching rows.
            return frame_dict(context.workspace.query_json(resolved, spec).execute(), near_bounded=bool(spec.get("near")))
        except Exception as exc:
            mcp_raise(exc)

    return query
