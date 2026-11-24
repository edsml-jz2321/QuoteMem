"""Shared ``eq``/``in`` handlers for row stores that keep a value index.

Plain row stores have no query engine, so by default every operator runs through
the Python scan fallback. A backend *can* answer equality/membership natively if
it maintains a ``value -> object_ids`` secondary index. ``redis`` does this with
server-side ``SET`` indexes; this module compiles ``eq``/``in`` on such backends
into an ``index_filter`` fragment the backend resolves through its index, while
still carrying the portable ``notes['row_payload']`` predicate so non-indexable
fields and mixed boolean trees degrade to scan.

The in-memory backend opts into the same contract with a process-local value
map. File backends remain scan-only because rebuilding an ephemeral map would
not improve their reference-backend role.

``array_contains`` compiles to an ``index_filter`` only on a backend that keeps
an array *item* index (``redis`` adds one SET member per item of every
indexable array field: string-like, date or string-categorical items). The
in-memory value map holds only whole scalar values of declared indexes and
single-field unique constraints, so it registers ``eq``/``in`` alone and leaves
``array_contains`` to the portable scan handler; a native fragment there would
resolve to no candidates.
"""

from __future__ import annotations

__all__ = [
    "indexed_filter_handler",
    "indexed_array_contains_handler",
    "register_indexed_handlers",
]

from heavenbase.utils import Any

from ...filters.ops import OP_ARRAY_CONTAINS, OP_EQ, OP_IN
from ..contracts import HandlerContext, QueryFragment
from .filters import filter_handler, normalize_filter_value


def indexed_filter_handler(op: str):
    """Return an ``eq``/``in`` handler that compiles to a value-index lookup.

    Args:
        op (str): Filter operation identifier, either ``"eq"`` or ``"in"``.

    Returns:
        Callable[[Any, Any, HandlerContext], QueryFragment]: Handler compiler that
        emits native index fragments when safe and otherwise marks scan fallback.
    """

    def _compile(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
        value = normalize_filter_value(field_schema.dtype, op, value, resolver=ctx.resolver)
        py_fragment = filter_handler(op)(field_schema, value, ctx)  # portable scan predicate
        if not ctx.scalar_indexed:
            return py_fragment.mark_scan()
        if op == OP_IN and not isinstance(value, (list, tuple, set)):
            raise TypeError("in filter expects a list, tuple, or set")
        members = [value] if op == OP_EQ else list(value)
        if any(member is None for member in members):
            # A None value is never written to the SET index (NULL has no index member), so a
            # native eq None / in [None] would return nothing while the scan predicate matches
            # NULL/missing rows -> route None operands to scan to keep both channels identical.
            return py_fragment.mark_scan()
        payload = {"value": value} if op == OP_EQ else {"values": members}
        return QueryFragment(
            ctx.backend,
            op,
            field_schema.name,
            payload,
            notes={"kind": "index_filter", "op": op, "mode": "native", "row_payload": py_fragment.payload},
        )

    return _compile


def _array_contains_indexable(field_schema: Any) -> bool:
    """Return whether Redis SET membership can safely mirror Python array_contains."""
    if type(field_schema.dtype).identifier != "array":
        return False
    item_type = getattr(field_schema.dtype, "item_type", None)
    if item_type is None:
        return False
    item_identifier = type(item_type).identifier
    if item_identifier == "categorical":
        return getattr(item_type, "base", "string") == "string"
    return item_identifier in ("date", "identifier", "long-text", "medium-text", "short-text")


def indexed_array_contains_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Return an ``array_contains`` handler backed by a Redis SET item index when safe.

    Register it only for a backend whose index holds one member per array item
    for every indexable array field (see ``register_indexed_handlers``): the
    fragment carries no index declaration check, and the backend resolves its
    candidates from that item index alone.

    Args:
        field_schema (Any): Routed array field schema.
        value (Any): Normalized array membership operand.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native index fragment when safe, otherwise a scan fallback fragment.
    """
    py_fragment = filter_handler(OP_ARRAY_CONTAINS)(field_schema, value, ctx)
    if value is None or not _array_contains_indexable(field_schema):
        return py_fragment.mark_scan()
    return QueryFragment(
        ctx.backend,
        OP_ARRAY_CONTAINS,
        field_schema.name,
        {"value": value},
        notes={"kind": "index_filter", "op": OP_ARRAY_CONTAINS, "mode": "native", "row_payload": py_fragment.payload},
    )


def register_indexed_handlers(registry: Any, ctx: Any, backend_identifier: str, *, array_items: bool = False) -> None:
    """Register native equality/index compilers for one row Backend.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.
        backend_identifier: Canonical Backend type receiving the compilers.
        array_items: Whether the backend indexes every item of each indexable
            array field, so ``array_contains`` can resolve through that index.
            When false, ``array_contains`` keeps the portable scan handler.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    for dtype in ctx.filter_types:
        ops = ctx.filter_ops(dtype)
        for strategy in ctx.filter_strategies:
            for op in (OP_EQ, OP_IN):
                if op in ops:
                    ctx.handler(dtype, op, backend_identifier, strategy, indexed_filter_handler(op)).register(registry, replace=True)
            if array_items and OP_ARRAY_CONTAINS in ops:
                ctx.handler(dtype, OP_ARRAY_CONTAINS, backend_identifier, strategy, indexed_array_contains_handler).register(registry, replace=True)
