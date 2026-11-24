"""Logical operation vocabulary for filters, near search, and handlers."""

from __future__ import annotations

__all__ = [
    "OP_ALL",
    "OP_ARRAY_CONTAINS",
    "OP_CONTAINED_IN",
    "OP_CONTAINS",
    "OP_ENDS_WITH",
    "OP_EQ",
    "OP_EXISTS",
    "OP_GET",
    "OP_GT",
    "OP_GTE",
    "OP_ILIKE",
    "OP_IN",
    "OP_LIKE",
    "OP_LT",
    "OP_LTE",
    "OP_MATCH",
    "OP_NE",
    "OP_NEAR",
    "OP_REGEX",
    "OP_STARTS_WITH",
    "OP_WILDCARD",
    "normalize_op",
    "op_aliases",
    "register_op",
    "registered_ops",
]

from heavenbase.utils.ids import check_id
from heavenbase.utils.typing import Any, Mapping, Sequence

from ..modules import ModuleResolver, ModuleService, module_resolver

OP_ALL = "all"
OP_ARRAY_CONTAINS = "array_contains"
OP_CONTAINED_IN = "contained_in"
OP_CONTAINS = "contains"
OP_EQ = "eq"
OP_EXISTS = "exists"
OP_GET = "get"
OP_GT = "gt"
OP_GTE = "gte"
OP_ILIKE = "ilike"
OP_IN = "in"
OP_LIKE = "like"
OP_LT = "lt"
OP_LTE = "lte"
OP_MATCH = "match"
OP_NE = "ne"
OP_NEAR = "near"
OP_REGEX = "regex"
OP_STARTS_WITH = "starts_with"
OP_ENDS_WITH = "ends_with"
OP_WILDCARD = "wildcard"


def _priority(row: dict) -> int:
    """Return one operation record's explicit ordering priority."""
    meta = row["record"].get("meta")
    value = meta.get("priority", 1000) if isinstance(meta, dict) else 1000
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Operation {row['identifier']!r} priority must be an integer")
    return value


def _ordered_ops(rows: Sequence[dict]) -> tuple[str, ...]:
    """Return inspection rows in record-defined operation order."""
    return tuple(str(row["identifier"]) for row in sorted(rows, key=lambda row: (_priority(row), str(row["identifier"]))))


def _module_service(resolver: ModuleResolver) -> ModuleService:
    """Require lifecycle authority for one operation mutation."""
    if not isinstance(resolver, ModuleService):
        raise TypeError("Operation registration requires a mutable ModuleService resolver")
    return resolver


def register_op(
    op: str,
    aliases: Sequence[str] = (),
    tags: Mapping[str, Any] | None = None,
    *,
    resolver: ModuleResolver | None = None,
) -> str:
    """Register one canonical logical operation.

    Operation names are the shared vocabulary between JSON filters, Python
    field refs, handler capability metadata, and backend fragment compilers.

    Args:
        op (str): Canonical operation token such as `eq`, `near`, or `match`.
        aliases (Sequence[str]): Accepted aliases for the operation. JSON
            aliases commonly include a `$` prefix.
        tags (Mapping[str, Any] | None): Raw JSON-compatible Tag claims copied
            onto the published operation record. Definitions are optional and
            registration never creates them.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted. Registration requires a
            mutable `ModuleService` implementation.

    Returns:
        str: Registered canonical operation token.
    """
    name = check_id(str(op), "operation identifier")
    service = _module_service(module_resolver(resolver))
    found = service.inspect("operation", name)
    existing_aliases = tuple(found[0]["aliases"]) if found else ()
    if tags is not None and not isinstance(tags, Mapping):
        raise TypeError("Operation tags must be a string-keyed mapping")
    authored_tags = dict(tags or {})
    if not all(isinstance(identifier, str) for identifier in authored_tags):
        raise TypeError("Operation tags must be a string-keyed mapping")
    existing_meta = found[0]["record"].get("meta") if found else {}
    existing_tags = existing_meta.get("tags", {}) if isinstance(existing_meta, Mapping) else {}
    if not isinstance(existing_tags, Mapping):
        raise TypeError(f"Operation {name!r} record meta.tags must be a mapping")
    alias_values = tuple(dict.fromkeys((*existing_aliases, *(str(alias).strip() for alias in aliases if str(alias).strip() != name))))
    tag_values = {**existing_tags, **authored_tags}
    coordinate = f"programmatic/operation/{name}"
    if found:
        installation = found[0]["record"]["meta"]["installation"]
        unchanged = alias_values == existing_aliases and tag_values == existing_tags
        if unchanged:
            return name
        if installation["coordinate"] != coordinate:
            raise ValueError(f"Operation {name!r} belongs to {installation['coordinate']!r} and cannot be changed through register_op")
        record = dict(found[0]["record"])
        record["meta"] = {key: value for key, value in record["meta"].items() if key != "installation"}
    else:
        next_priority = max((_priority(row) for row in service.inspect("operation")), default=-10) + 10
        record = {
            "name": name.replace("_", " ").title(),
            "desc": "Registered logical operation.",
            "source": "inline",
            "target": "definition",
            "active": True,
            "meta": {
                "schema_version": 1,
                **({"tags": tag_values} if tag_values else {}),
                "priority": next_priority,
                "definition": {"kind": "operation"},
            },
        }
    if found:
        record["meta"] = {**record["meta"], **({"tags": tag_values} if tag_values else {})}
        if not tag_values:
            record["meta"].pop("tags", None)
    service.register("operation", name, record, aliases=alias_values, coordinate=coordinate)
    return name


def normalize_op(op: str, *, resolver: ModuleResolver | None = None) -> str:
    """Return the canonical operation name for a user token or alias.

    Args:
        op (str): Operation token or alias.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        str: Canonical registered operation name.

    Raises:
        LookupError: If the operation or alias has no active registration.
    """
    value = str(op)
    selected = module_resolver(resolver)
    rows = selected.inspect("operation", value.strip().lower())
    if not rows:
        raise LookupError(f"Unknown operation {value!r}; register it before use")
    return str(rows[0]["identifier"])


def op_aliases(
    op: str,
    *,
    include_canonical: bool = False,
    resolver: ModuleResolver | None = None,
) -> tuple[str, ...]:
    """Return accepted aliases for one operation.

    Args:
        op (str): Operation token or alias.
        include_canonical (bool): Whether to include the canonical operation in
            the returned tuple.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        tuple[str, ...]: Aliases known to the operation registry.
    """
    selected = module_resolver(resolver)
    canonical = normalize_op(op, resolver=selected)
    row = selected.inspect("operation", canonical)[0]
    aliases = tuple(str(alias) for alias in row["aliases"])
    return (canonical, *aliases) if include_canonical else aliases


def registered_ops(*, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
    """Return all registered canonical operation names.

    Args:
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        tuple[str, ...]: Canonical operation names in registration order.
    """
    return _ordered_ops(module_resolver(resolver).inspect("operation"))
