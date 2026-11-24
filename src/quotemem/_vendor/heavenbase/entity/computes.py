"""Registry-backed entity compute hooks and their registration API."""

from __future__ import annotations

__all__ = [
    "ComputeCallable",
    "register_compute",
    "register_query_compute",
    "registered_computes",
    "registered_query_computes",
]

from typing import TYPE_CHECKING

from heavenbase.utils import Any, Callable, Mapping, Optional, Sequence, Tuple
from heavenbase.utils.registry import named_key, python_target

from ..modules import ModuleResolver, module_resolver
from ..modules.base import ResolverCache

if TYPE_CHECKING:
    from ..modules import ModuleService

ComputeCallable = Callable[..., Any]

COMPUTE_KIND = "entity_compute"
QUERY_COMPUTE_KIND = "entity_query_compute"

_LABELS = {COMPUTE_KIND: "compute function", QUERY_COMPUTE_KIND: "query compute function"}
_COMPUTE_CACHE = ResolverCache[ComputeCallable]()


def register_compute(
    name: str,
    fn: ComputeCallable,
    *,
    aliases: Sequence[str] = (),
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> ComputeCallable:
    """Register one named compute hook so stored entity schemas can restore it.

    Entity schemas persist a compute hook by reference, so a workspace reopened from
    its stored manifest can only rebuild a computed field when that reference resolves
    again. Registration records the callable's import target, which lets both stored
    shapes resolve in any process that uses the same module registry and can import
    the owning module: the identifier registered here, and the module-qualified
    reference `f"{fn.__module__}.{fn.__qualname__}"` that Python-authored fields
    serialize to.

    Args:
        name (str): Stable compute identifier, lower-case with `-` or `_` separators.
        fn (ComputeCallable): Importable top-level callable computing the field value.
        aliases (Sequence[str]): Optional alternate identifiers resolving to ``name``.
        replace (bool): Whether a programmatically owned record may be replaced.
        resolver (ModuleResolver | None): Context-owned module resolver. The default
            Context is selected when omitted. Registration requires a `ModuleService`.

    Returns:
        ComputeCallable: Materialized registered callable.

    Raises:
        TypeError: If `fn` is not callable, or the resolver cannot register records.
        ValueError: If the identifier is owned elsewhere or replacement is not allowed.

    Examples:
        >>> import heavenbase as hb
        >>> from myapp.vectors import embed_text  # doctest: +SKIP
        >>> hb.ext.register_compute("myapp_embed_text", embed_text)  # doctest: +SKIP
    """
    return _register(name, fn, kind=COMPUTE_KIND, aliases=aliases, replace=replace, resolver=resolver)


def register_query_compute(
    name: str,
    fn: ComputeCallable,
    *,
    aliases: Sequence[str] = (),
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> ComputeCallable:
    """Register one named query-compute hook so stored entity schemas can restore it.

    Query-compute hooks turn a query-time input into a comparable field value; they
    are persisted and restored exactly like compute hooks. See `register_compute`
    for the reference-resolution contract.

    Args:
        name (str): Stable query-compute identifier, lower-case with `-` or `_` separators.
        fn (ComputeCallable): Importable top-level callable accepting one query input.
        aliases (Sequence[str]): Optional alternate identifiers resolving to ``name``.
        replace (bool): Whether a programmatically owned record may be replaced.
        resolver (ModuleResolver | None): Context-owned module resolver. The default
            Context is selected when omitted. Registration requires a `ModuleService`.

    Returns:
        ComputeCallable: Materialized registered callable.

    Raises:
        TypeError: If `fn` is not callable, or the resolver cannot register records.
        ValueError: If the identifier is owned elsewhere or replacement is not allowed.
    """
    return _register(name, fn, kind=QUERY_COMPUTE_KIND, aliases=aliases, replace=replace, resolver=resolver)


def registered_computes(*, resolver: ModuleResolver | None = None) -> Tuple[str, ...]:
    """Return the sorted identifiers of every registered compute hook.

    Args:
        resolver (ModuleResolver | None): Context-owned module resolver. The default
            Context is selected when omitted.

    Returns:
        Tuple[str, ...]: Sorted registered compute identifiers.
    """
    return _registered(COMPUTE_KIND, resolver=resolver)


def registered_query_computes(*, resolver: ModuleResolver | None = None) -> Tuple[str, ...]:
    """Return the sorted identifiers of every registered query-compute hook.

    Args:
        resolver (ModuleResolver | None): Context-owned module resolver. The default
            Context is selected when omitted.

    Returns:
        Tuple[str, ...]: Sorted registered query-compute identifiers.
    """
    return _registered(QUERY_COMPUTE_KIND, resolver=resolver)


def _register(
    name: str,
    fn: ComputeCallable,
    *,
    kind: str,
    aliases: Sequence[str],
    replace: bool,
    resolver: ModuleResolver | None,
) -> ComputeCallable:
    """Publish one compute hook as an ordinary module record."""
    label = _LABELS[kind]
    if not callable(fn):
        raise TypeError(f"{label} must be callable")
    key = named_key(name, f"{label} name")
    selected = module_resolver(resolver)
    service = _module_service(selected, label)
    found = service.inspect(kind, key)
    if found:
        if not replace:
            raise ValueError(f"{label.capitalize()} {key!r} is already registered")
        coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if coordinate != f"programmatic/{kind}/{key}":
            raise ValueError(f"{label.capitalize()} {key!r} belongs to {coordinate!r} and cannot be replaced")
    service.register(
        kind,
        key,
        {
            "name": key,
            "desc": _doc_summary(fn),
            "source": "python-import",
            "target": python_target(fn, label=f"{label} {key!r}"),
            "active": True,
            "meta": {"schema_version": 1},
        },
        aliases=aliases,
    )
    _COMPUTE_CACHE.clear(selected)
    resolved = _resolve(kind, key, resolver=selected)
    if resolved is None:  # pragma: no cover - a record written above is always readable back
        raise ValueError(f"{label.capitalize()} {key!r} did not resolve after registration")
    return resolved


def _registered(kind: str, *, resolver: ModuleResolver | None) -> Tuple[str, ...]:
    """Return sorted registered identifiers for one compute-hook kind."""
    return tuple(sorted(str(row["identifier"]) for row in module_resolver(resolver).inspect(kind)))


def _resolve(kind: str, name: str, *, resolver: ModuleResolver | None = None) -> Optional[ComputeCallable]:
    """Return one registered compute hook, or ``None`` when the reference is unknown.

    Entity schemas persist a compute hook as whichever reference the schema was
    authored with: a registered identifier, or the module-qualified reference
    `f"{fn.__module__}.{fn.__qualname__}"` that a Python-authored field
    serializes to. Both shapes resolve here, the second by matching the
    registered import target.
    """
    selected = module_resolver(resolver)
    rows = selected.inspect(kind, name) or tuple(row for row in selected.inspect(kind) if _import_ref(row) == name)
    return _from_row(rows[0], selected) if rows else None


def _import_ref(row: Mapping[str, Any]) -> str:
    """Return one record's import target as a module-qualified reference."""
    return str(row["record"].get("target") or "").replace(":", ".", 1)


def _module_service(resolver: ModuleResolver, label: str) -> "ModuleService":
    """Require lifecycle authority for a compute-hook mutation."""
    from ..modules import ModuleService

    if not isinstance(resolver, ModuleService):
        raise TypeError(f"{label.capitalize()} registration requires a ModuleService resolver")
    return resolver


def _from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> ComputeCallable:
    """Materialize and cache one compute-hook inspection row."""

    def build(current_row: Mapping[str, Any]) -> ComputeCallable:
        fn = resolver.resolve(str(current_row["kind"]), str(current_row["identifier"]))
        if not callable(fn):
            raise TypeError(f"Compute hook {current_row['identifier']!r} did not resolve to a callable")
        return fn

    return _COMPUTE_CACHE.get(resolver, row, build)


def _doc_summary(fn: Any) -> str:
    """Return the first documentation line for one compute hook."""
    lines = str(getattr(fn, "__doc__", "") or "").strip().splitlines()
    return lines[0].strip() if lines else ""
