"""Registry-backed sparse-GRAM callable helpers."""

from __future__ import annotations

__all__ = ["GramCallable", "callable_ref", "register_gram_callable", "resolve_gram_callable"]

from typing import TYPE_CHECKING

from heavenbase.utils import Any, Callable, Mapping, Sequence
from heavenbase.utils.registry import named_key, python_target

from .....modules import ModuleResolver, module_resolver
from .....modules.base import ResolverCache

if TYPE_CHECKING:
    from .....modules import ModuleService

GramCallable = Callable[..., Any]
_GRAM_CACHE = ResolverCache[GramCallable]()


def _module_service(resolver: ModuleResolver) -> "ModuleService":
    """Require lifecycle authority for a sparse-GRAM mutation."""
    from .....modules import ModuleService

    if not isinstance(resolver, ModuleService):
        raise TypeError("GRAM callable mutation requires a ModuleService resolver")
    return resolver


def callable_ref(fn: GramCallable) -> str:
    """Return a stable best-effort module-qualified callable reference.

    Args:
        fn (GramCallable): Callable whose reference should be serialized.

    Returns:
        str: Best-effort module-qualified callable reference string.
    """
    module = str(getattr(fn, "__module__", "") or "")
    name = str(getattr(fn, "__qualname__", getattr(fn, "__name__", "callable")) or "callable")
    return f"{module}.{name}" if module else name


def register_gram_callable(
    name: str,
    fn: GramCallable,
    *,
    kind: str,
    aliases: Sequence[str] = (),
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> GramCallable:
    """Publish one sparse-GRAM callable as an ordinary module record.

    Args:
        name: Stable callable identifier.
        fn: Importable callable implementation.
        kind: Record kind, `gram_normalizer` or `gram_tokenizer`.
        aliases: Optional alternate identifiers.
        replace: Whether a programmatically owned record may be replaced.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted. Registration requires a `ModuleService`.

    Returns:
        GramCallable: Materialized registered callable.

    Raises:
        TypeError: If `fn` is not callable.
        ValueError: If the identifier is owned elsewhere or replacement is
            not allowed.
    """
    if kind not in {"gram_normalizer", "gram_tokenizer"}:
        raise ValueError(f"Unsupported GRAM callable kind {kind!r}")
    if not callable(fn):
        raise TypeError(f"{kind.replace('_', ' ')} must be callable")
    key = named_key(name, f"{kind.replace('_', ' ')} name")
    selected = module_resolver(resolver)
    service = _module_service(selected)
    found = service.inspect(kind, key)
    if found:
        existing = _gram_from_row(found[0], selected)
        if existing is fn and replace:
            return existing
        if not replace:
            raise ValueError(f"{kind.replace('_', ' ').title()} {key!r} is already registered")
        coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if coordinate != f"programmatic/{kind}/{key}":
            raise ValueError(f"{kind.replace('_', ' ').title()} {key!r} belongs to {coordinate!r} and cannot be replaced")
    service.register(
        kind,
        key,
        {
            "name": key,
            "desc": _doc_summary(fn),
            "source": "python-import",
            "target": python_target(fn, label=f"{kind.replace('_', ' ')} {key!r}"),
            "active": True,
            "meta": {"schema_version": 1, "tags": {"gram": True}},
        },
        aliases=aliases,
    )
    _GRAM_CACHE.clear(selected)
    return resolve_gram_callable(kind, key, resolver=selected)


def resolve_gram_callable(
    kind: str,
    raw: Any,
    *,
    resolver: ModuleResolver | None = None,
) -> GramCallable:
    """Resolve a direct callable, registered identifier, or alias.

    Args:
        kind: Record kind, `gram_normalizer` or `gram_tokenizer`.
        raw: Direct callable, registered identifier, or alias.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        GramCallable: Validated callable implementation.
    """
    if callable(raw):
        return raw
    selected = module_resolver(resolver)
    name = named_key(raw, f"{kind.replace('_', ' ')} name")
    rows = selected.inspect(kind, name)
    if not rows:
        known = sorted(str(row["identifier"]) for row in selected.inspect(kind))
        raise ValueError(f"Unknown {kind.replace('_', ' ')} {name!r}; known={known}")
    return _gram_from_row(rows[0], selected)


def _gram_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> GramCallable:
    """Materialize and cache one sparse-GRAM callable inspection row."""

    def build(current_row: Mapping[str, Any]) -> GramCallable:
        fn = resolver.resolve(str(current_row["kind"]), str(current_row["identifier"]))
        if not callable(fn):
            raise TypeError(f"GRAM callable {current_row['identifier']!r} did not resolve to a callable")
        return fn

    return _GRAM_CACHE.get(resolver, row, build)


def _doc_summary(fn: Any) -> str:
    """Return the first documentation line for one GRAM callable."""
    lines = str(getattr(fn, "__doc__", "") or "").strip().splitlines()
    return lines[0].strip() if lines else ""
