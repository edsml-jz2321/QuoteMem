"""Registry-backed SQL dialect metadata and generic SQL filter policy."""

from __future__ import annotations

__all__ = [
    "SQL_DEFAULT_NATIVE_AGGREGATE_OPS",
    "SQL_DEFAULT_NATIVE_OPS",
    "SQL_JSON_AGG_STYLE_ARRAYAGG",
    "SQL_JSON_AGG_STYLE_GROUP_ARRAY",
    "SQL_JSON_AGG_STYLE_POSTGRES_ORDERED",
    "SQL_JSON_OBJECT_STYLE_BUILD_OBJECT",
    "SQL_JSON_OBJECT_STYLE_OBJECT",
    "SQL_JSON_PATH_STYLE_JSON1",
    "SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH",
    "SQL_REGEX_STYLE_POSTGRES_TILDE",
    "SQL_REGEX_STYLE_REGEXP_LIKE",
    "SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE",
    "SQL_WILDCARD_STYLE_GLOB",
    "SQL_WILDCARD_STYLE_LIKE",
    "SQLDialectSpec",
    "register_sql_dialect",
    "registered_sql_dialects",
    "sql_dialect_spec",
]

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from heavenbase.utils import Any, Dict, deepcopy

from ...modules import ModuleResolver, module_resolver, python_executable_ref
from ...modules.tags import tags as inspect_tags
from ...modules.base import ResolverCache
from ...types import LogicalType
from ...utils.ids import check_id
from .dialect import (
    SQL_DEFAULT_NATIVE_AGGREGATE_OPS,
    SQL_DEFAULT_NATIVE_OPS,
    SQL_JSON_AGG_STYLE_ARRAYAGG,
    SQL_JSON_AGG_STYLE_GROUP_ARRAY,
    SQL_JSON_AGG_STYLE_POSTGRES_ORDERED,
    SQL_JSON_OBJECT_STYLE_BUILD_OBJECT,
    SQL_JSON_OBJECT_STYLE_OBJECT,
    SQL_JSON_PATH_STYLE_JSON1,
    SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH,
    SQL_REGEX_STYLE_POSTGRES_TILDE,
    SQL_REGEX_STYLE_REGEXP_LIKE,
    SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE,
    SQL_WILDCARD_STYLE_GLOB,
    SQL_WILDCARD_STYLE_LIKE,
    SQLDialectSpec,
)

if TYPE_CHECKING:
    from ...modules import ModuleService

SQL_SCAN_FILTER_DOMAINS = frozenset(("array", "json", "struct"))
_SQL_FILTER_SCAN_CACHE = ResolverCache[bool]()


def sql_filter_requires_scan(dtype: Any, *, resolver: ModuleResolver | None = None) -> bool:
    """Return whether generic SQL scalar filters must use row-payload scan.

    Args:
        dtype: Logical type class or instance.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        True for logical types in structured filter domains whose canonical
        values are payloads rather than scalar SQL column values.
    """
    selected = module_resolver(resolver)
    identifier = LogicalType.identifier_of(dtype, resolver=selected)
    rows = selected.inspect("logical_type", identifier)
    if not rows:
        raise KeyError(f"Unknown logical type {identifier!r}")

    def build(row: Mapping[str, Any]) -> bool:
        domains = inspect_tags(selected, "logical_type", row["identifier"]).raw.get("filter-domains")
        return isinstance(domains, (list, tuple)) and all(isinstance(domain, str) for domain in domains) and bool(SQL_SCAN_FILTER_DOMAINS.intersection(domains))

    return _SQL_FILTER_SCAN_CACHE.get(selected, rows[0], build)


_CALLBACK_FIELDS = ("text_evidence", "text_filter_factory", "text_key_evidence", "text_key_factory")
_SQL_DIALECT_CACHE = ResolverCache[SQLDialectSpec]()


def _module_service(resolver: ModuleResolver) -> "ModuleService":
    """Require lifecycle authority for a SQL dialect mutation."""
    from ...modules import ModuleService

    if not isinstance(resolver, ModuleService):
        raise TypeError("SQL dialect mutation requires a ModuleService resolver")
    return resolver


def _sql_dialect_definition(spec: SQLDialectSpec) -> dict[str, Any]:
    """Return one durable inline SQL dialect definition."""
    definition = spec.to_dict()
    definition.pop("identifier")
    for field in _CALLBACK_FIELDS:
        value = getattr(spec, field)
        definition[field] = None if value is None else python_executable_ref(value, label=f"SQL dialect {spec.identifier!r} {field}")
    return definition


def _sql_dialect_record(spec: SQLDialectSpec) -> dict[str, Any]:
    """Return one plain central registration record for `spec`."""
    return {
        "name": f"{spec.identifier.replace('_', ' ').title()} SQL Dialect",
        "desc": f"SQL handler and projection metadata for {spec.identifier}.",
        "source": "inline",
        "target": "definition",
        "active": True,
        "meta": {"schema_version": 1, "tags": {"sql": True}, "definition": _sql_dialect_definition(spec)},
    }


def _sql_dialect_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> SQLDialectSpec:
    """Materialize and cache one validated SQL dialect record."""

    def build(current_row: Mapping[str, Any]) -> SQLDialectSpec:
        raw = resolver.resolve("sql_dialect", str(current_row["identifier"]))
        if not isinstance(raw, Mapping):
            raise TypeError(f"SQL dialect {current_row['identifier']!r} definition must be a mapping")
        allowed = set(SQLDialectSpec(str(current_row["identifier"])).to_dict()) - {"identifier"}
        allowed.update(_CALLBACK_FIELDS)
        unsupported = set(raw) - allowed
        if unsupported:
            raise ValueError(f"SQL dialect {current_row['identifier']!r} definition has unsupported fields {sorted(unsupported)!r}")
        values = deepcopy(dict(raw))
        for field in _CALLBACK_FIELDS:
            reference = values.get(field)
            if reference is not None and not isinstance(reference, Mapping):
                raise TypeError(f"SQL dialect {current_row['identifier']!r} {field} must be an executable reference or null")
            values[field] = None if reference is None else resolver.resolve_executable(reference, owner=current_row)
        return SQLDialectSpec(str(current_row["identifier"]), **values)

    return _SQL_DIALECT_CACHE.get(resolver, row, build)


def register_sql_dialect(
    spec: SQLDialectSpec,
    *,
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> SQLDialectSpec:
    """Publish SQL handler metadata through the central module Registry.

    Args:
        spec: SQL dialect metadata.
        replace: Whether an existing spec may be overwritten.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted. Registration requires a `ModuleService`.

    Returns:
        SQLDialectSpec: The registered spec.

    Raises:
        TypeError: If `spec` is not a `SQLDialectSpec`.
        ValueError: If the dialect exists or belongs to another owner.
    """
    if not isinstance(spec, SQLDialectSpec):
        raise TypeError("register_sql_dialect expects SQLDialectSpec")
    selected = module_resolver(resolver)
    service = _module_service(selected)
    found = service.inspect("sql_dialect", spec.identifier)
    coordinate = f"programmatic/sql_dialect/{spec.identifier}"
    if found:
        existing = _sql_dialect_from_row(found[0], selected)
        if _sql_dialect_definition(existing) == _sql_dialect_definition(spec):
            return existing
        if not replace:
            raise ValueError(f"SQL dialect {spec.identifier!r} is already registered")
        existing_coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if existing_coordinate != coordinate:
            raise ValueError(f"SQL dialect {spec.identifier!r} belongs to {existing_coordinate!r} and cannot be replaced")
    service.register(
        "sql_dialect",
        spec.identifier,
        _sql_dialect_record(spec),
        coordinate=coordinate,
    )
    _SQL_DIALECT_CACHE.clear(selected)
    return cast(SQLDialectSpec, _sql_dialect_from_row(service.inspect("sql_dialect", spec.identifier)[0], selected))


def sql_dialect_spec(identifier: str, *, resolver: ModuleResolver | None = None) -> SQLDialectSpec:
    """Return registered SQL handler metadata for one backend type.

    Args:
        identifier: Backend type identifier used by ``HandlerContext.backend_identifier``.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        SQLDialectSpec: Registry-materialized SQL dialect metadata.

    Raises:
        LookupError: If the identifier has no active registration.
    """
    selected = module_resolver(resolver)
    key = check_id(str(identifier).strip(), "SQL dialect identifier")
    found = selected.inspect("sql_dialect", key)
    if not found:
        raise LookupError(f"Unknown SQL dialect {key!r}; install its registration before use")
    return _sql_dialect_from_row(found[0], selected)


def registered_sql_dialects(*, resolver: ModuleResolver | None = None) -> Dict[str, SQLDialectSpec]:
    """Return registered SQL dialect specs keyed by backend identifier.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        Dict[str, SQLDialectSpec]: Snapshot of SQL dialect metadata.
    """
    selected = module_resolver(resolver)
    return {str(row["identifier"]): _sql_dialect_from_row(row, selected) for row in selected.inspect("sql_dialect")}
