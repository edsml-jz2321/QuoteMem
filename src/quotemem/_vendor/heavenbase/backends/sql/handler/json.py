"""SQL-native Json path (`get`) filter compiler.

Dialects opt in by declaring a ``json_path_style`` (currently SQLite json1);
everything else keeps the portable Python scan with an honest explain reason.
Native compilation is restricted to filter shapes whose SQL evaluation is
provably row-set-identical to the portable scan semantics.
"""

from __future__ import annotations

__all__ = [
    "sql_json_get_handler",
]

import math
import re

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.execution.contracts import QueryFragment
from heavenbase.backends.sql.capabilities import sql_dialect_spec
from heavenbase.filters.ops import OP_CONTAINS, OP_GET, OP_IN
from heavenbase.execution.handlers.filters import filter_handler, normalize_filter_value
from .text import TEXT_SEMANTICS_NOT_PROVEN

# Keys embedded in a quoted SQL JSON path: anything without quotes, backslashes, or control chars.
_SQL_JSON_KEY_RE = re.compile(r'[^"\\\x00-\x1f]+')
_SQL_INT_MIN, _SQL_INT_MAX = -(2**63), 2**63 - 1


def _json_get_scan_reason(path: list[str], op: str, value: Any) -> str | None:
    """Return why one `get` filter shape must stay on Python scan, or `None`.

    The rules mirror the portable scan predicates exactly: shapes whose Python
    semantics cannot be reproduced by guarded SQL JSON expressions fall back.

    Args:
        path (list[str]): Normalized nested key path.
        op (str): Inner comparison operation.
        value (Any): Inner comparison operand.

    Returns:
        str | None: Human-readable fallback reason, or `None` when native
        compilation preserves scan parity.
    """
    if any(_SQL_JSON_KEY_RE.fullmatch(part) is None for part in path):
        return "path keys with quotes, backslashes, or control characters are not compiled into SQL JSON paths"
    if op == OP_CONTAINS:
        return "contains matches Python stringifications of non-text payloads, which SQL cannot reproduce"
    for item in list(value) if op == OP_IN else [value]:
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, int):
            if not _SQL_INT_MIN <= item <= _SQL_INT_MAX:
                return "integer operands beyond the 64-bit SQL range compare exactly only in Python"
        elif isinstance(item, float):
            if math.isnan(item) or math.isinf(item):
                return "non-finite float operands have no SQL comparison parity"
        elif isinstance(item, str):
            return TEXT_SEMANTICS_NOT_PROVEN
        else:
            return "non-scalar operands compare structurally only in Python"
    return None


def sql_json_get_handler(field_schema: Any, value: Any, ctx: HandlerContext) -> QueryFragment:
    """Compile one Json path filter natively when the SQL dialect declares support.

    Missing or null paths match nothing, heterogeneous comparisons match
    nothing instead of coercing, and double-encoded JSON string storage is
    unwrapped per row. Any shape outside the proven-parity envelope returns
    the portable scan fragment marked with an explicit fallback reason.

    Args:
        field_schema (Any): Routed Json field schema.
        value (Any): `get` payload with `path`, inner operation, and operand.
        ctx (HandlerContext): Selected backend and strategy context.

    Returns:
        QueryFragment: Native `sql_where` fragment carrying the portable row
        payload, or the Python scan fragment when pushdown is unsafe.

    Raises:
        TypeError: If an inner `in` operand is not a list, tuple, or set.
        ValueError: If the `get` payload is malformed.
    """
    value = normalize_filter_value(field_schema.dtype, OP_GET, value, resolver=ctx.resolver)
    py_fragment = filter_handler(OP_GET)(field_schema, value, ctx)
    dialect = sql_dialect_spec(ctx.backend_identifier, resolver=ctx.resolver)
    if not dialect.supports_native_filter(OP_GET):
        return py_fragment
    path, inner_op, inner_value = value["path"], value["op"], value["value"]
    reason = _json_get_scan_reason(path, inner_op, inner_value)
    if reason is not None:
        if reason == TEXT_SEMANTICS_NOT_PROVEN:
            return py_fragment.mark_scan(reason)
        return py_fragment.mark_scan(f"SQL JSON path pushdown skipped: {reason}.")
    operand = list(inner_value) if inner_op == OP_IN else inner_value

    def _where(table: Any) -> Any:
        return dialect.json_get_expression(table.c[field_schema.name], tuple(path), inner_op, operand)

    return QueryFragment(ctx.backend, OP_GET, field_schema.name, _where, notes={"kind": "sql_where", "mode": "native", "row_payload": py_fragment.payload})
