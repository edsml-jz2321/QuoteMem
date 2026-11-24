"""Resolver-driven shared handler registration."""

from __future__ import annotations

import copy
from types import MappingProxyType

__all__ = [
    "filter_ops",
    "seed_handlers",
]

from heavenbase.utils import Any, Mapping, dataclass

from ...filters.ops import OP_EQ, OP_NEAR
from ...modules import ModuleResolver, module_resolver
from ...types import LogicalType, get_logical_type
from .filters import filter_handler
from .near import near_handler
from .plugins import HandlerPlugin, HandlerSeedContext, handler_plugins
from .registry import HandlerRegistry


def _freeze_claim(value: Any) -> Any:
    """Return one recursively immutable detached claim value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_claim(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_claim(item) for item in value)
    return value


@dataclass(frozen=True)
class _SeedClaims:
    """Serve all handler-seeding claims from one coherent Registry snapshot."""

    resolver: ModuleResolver
    publication: int
    revision: int
    snapshot: str
    _identifiers: Mapping[str, tuple[str, ...]]
    _canonical: Mapping[tuple[str, str], str]
    _raw: Mapping[tuple[str, str], Mapping[str, Any]]

    def identifiers(self, kind: str) -> tuple[str, ...]:
        """Return active canonical identifiers for one exact record kind."""
        return self._identifiers.get(kind, ())

    def raw_tags(self, kind: str, identifier: str) -> Mapping[str, Any]:
        """Return one subject's exact static claims, or an empty mapping."""
        canonical = self._canonical.get((kind, identifier), identifier)
        return self._raw.get((kind, canonical), MappingProxyType({}))

    def boolean(self, kind: str, identifier: str, tag: str) -> bool:
        """Return whether one raw claim is exactly Boolean true."""
        return self.raw_tags(kind, identifier).get(tag) is True

    def identifier_set(self, kind: str, identifier: str, tag: str) -> frozenset[str]:
        """Return one locally validated identifier-set claim."""
        value = self.raw_tags(kind, identifier).get(tag)
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
            return frozenset()
        return frozenset(value)

    def identifiers_with(self, kind: str, tag: str) -> tuple[str, ...]:
        """Return exact-kind identifiers whose raw Boolean claim is true."""
        return tuple(identifier for identifier in self.identifiers(kind) if self.boolean(kind, identifier, tag))

    def canonical_identifier(self, kind: str, identifier: str) -> str | None:
        """Return one captured canonical identifier or alias target."""
        return self._canonical.get((kind, identifier))

    def filter_ops_for(self, dtype_identifier: str) -> tuple[str, ...]:
        """Return filter operations sharing a domain with one logical type."""
        domains = self.identifier_set("logical_type", dtype_identifier, "filter-domains")
        if not domains:
            return ()
        return tuple(
            operation
            for operation in self.identifiers("operation")
            if self.boolean("operation", operation, "filter") and domains.intersection(self.identifier_set("operation", operation, "filter-domains"))
        )

    def __call__(self, dtype: Any) -> tuple[str, ...]:
        """Return filter operations for a logical type selector."""
        if isinstance(dtype, type) and issubclass(dtype, LogicalType):
            identifier = dtype.identifier
        elif isinstance(dtype, LogicalType):
            identifier = type(dtype).identifier
        else:
            identifier = LogicalType.identifier_of(dtype, resolver=self.resolver)
        return self.filter_ops_for(str(identifier))


def _capture_seed_claims(resolver: ModuleResolver) -> _SeedClaims:
    """Capture active records and raw claims once for one seeding attempt."""
    rows = resolver.inspect()
    publications = {int(row["publication"]) for row in rows}
    revisions = {int(row["revision"]) for row in rows}
    if len(publications) > 1 or len(revisions) > 1:
        raise RuntimeError("Handler seeding received records from multiple Registry snapshots")
    publication = next(iter(publications), int(resolver.catalog.publication))
    revision = next(iter(revisions), int(getattr(resolver.catalog, "revision", 0)))
    identifiers: dict[str, list[str]] = {}
    canonical: dict[tuple[str, str], str] = {}
    raw_claims: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        kind = str(row["kind"])
        identifier = str(row["identifier"])
        identifiers.setdefault(kind, []).append(identifier)
        canonical[(kind, identifier)] = identifier
        for alias in row.get("aliases") or ():
            canonical[(kind, str(alias))] = identifier
        record = row.get("record")
        meta = record.get("meta") if isinstance(record, Mapping) else None
        raw = meta.get("tags", {}) if isinstance(meta, Mapping) else {}
        if not isinstance(raw, Mapping) or not all(isinstance(tag, str) for tag in raw):
            raw = {}
        raw_claims[(kind, identifier)] = _freeze_claim(raw)
    return _SeedClaims(
        resolver,
        publication,
        revision,
        f"{publication}:{revision}",
        MappingProxyType({kind: tuple(values) for kind, values in identifiers.items()}),
        MappingProxyType(canonical),
        MappingProxyType(raw_claims),
    )


def _logical_types_with(claims: _SeedClaims, resolver: ModuleResolver, tag: str) -> tuple[type[LogicalType], ...]:
    """Materialize logical types whose captured Boolean claim is true."""
    return tuple(get_logical_type(identifier, resolver=resolver) for identifier in claims.identifiers_with("logical_type", tag))


def _context_identifiers_with(ctx: HandlerSeedContext, kind: str, tag: str) -> tuple[str, ...]:
    """Return context-snapshot identifiers carrying one Boolean claim."""
    return tuple(identifier for identifier in ctx.identifiers(kind) if ctx.raw_tags(kind, identifier).get(tag) is True)


def _context_logical_types_with(ctx: HandlerSeedContext, tag: str) -> tuple[type[LogicalType], ...]:
    """Materialize logical types selected from the context claim snapshot."""
    return tuple(get_logical_type(identifier, resolver=ctx.resolver) for identifier in _context_identifiers_with(ctx, "logical_type", tag))


def filter_ops(dtype: Any, *, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
    """Return supported filter operations for one logical type class.

    Args:
        dtype (Any): Logical type class, instance, identifier, or alias.
        resolver (ModuleResolver | None): Context-owned module resolver. The
            default Context is selected when omitted.

    Returns:
        tuple[str, ...]: Operation identifiers supported by the logical type's
        `filter-domains` intersection with each operation.
    """
    selected = module_resolver(resolver)
    return _capture_seed_claims(selected)(dtype)


def register_generic_handlers(registry: HandlerRegistry, ctx: HandlerSeedContext) -> None:
    """Register portable fallback compilers for every eligible Backend.

    Args:
        registry: Workspace handler registry receiving compiler descriptors.
        ctx: Registry-derived logical-type, strategy, and operation vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    backend_identifiers = _context_identifiers_with(ctx, "backend_type", "portable")
    for dtype in ctx.filter_types:
        for strategy in ctx.filter_strategies:
            for op in ctx.filter_ops(dtype):
                for backend_identifier in backend_identifiers:
                    ctx.handler(dtype, op, backend_identifier, strategy, filter_handler(op)).register(registry, replace=True)
    if ctx.raw_tags("operation", OP_NEAR).get("near") is True:
        for dtype in ctx.vector_types:
            for strategy in _context_identifiers_with(ctx, "strategy", "near"):
                for backend_identifier in backend_identifiers:
                    ctx.handler(dtype, OP_NEAR, backend_identifier, strategy, near_handler).register(registry, replace=True)
    if ctx.raw_tags("operation", OP_EQ).get("vector") is True:
        for dtype in _context_logical_types_with(ctx, "vector"):
            for strategy in _context_identifiers_with(ctx, "strategy", "vector"):
                for backend_identifier in backend_identifiers:
                    ctx.handler(dtype, OP_EQ, backend_identifier, strategy, filter_handler(OP_EQ)).register(
                        registry,
                        replace=True,
                    )
    nested_types = _context_logical_types_with(ctx, "nested-filter")
    nested_operations = _context_identifiers_with(ctx, "operation", "nested-filter")
    nested_strategies = _context_identifiers_with(ctx, "strategy", "nested-filter")
    for dtype in nested_types:
        for operation in nested_operations:
            for strategy in nested_strategies:
                for backend_identifier in backend_identifiers:
                    ctx.handler(dtype, operation, backend_identifier, strategy, filter_handler(operation)).register(
                        registry,
                        replace=True,
                    )


def _handler_seed_context(resolver: ModuleResolver) -> HandlerSeedContext:
    """Build the stable provider-plugin vocabulary from raw canonical claims."""
    while True:
        claims = _capture_seed_claims(resolver)
        filter_types = tuple(dtype for dtype in _logical_types_with(claims, resolver, "filter") if claims(dtype))
        filter_strategies = tuple(
            identifier for identifier in claims.identifiers_with("strategy", "filter") if claims.boolean("strategy", identifier, "portable")
        )
        vector_types = _logical_types_with(claims, resolver, "near")
        context = HandlerSeedContext(
            filter_types,
            filter_strategies,
            vector_types,
            claims,
            resolver,
        )
        if resolver.catalog.publication == claims.publication:
            return context


_SEEDED_KEY = "handler-seed"
# The package's own handler contributions: their seeds read only the seeding snapshot.
_PACKAGE_PUBLISHER = "heavenbase"


@dataclass(frozen=True)
class _Seeding:
    """One resolver's seeding of the package's own leading contributions at one catalog publication."""

    resolver: Any
    publication: int
    template: HandlerRegistry
    context: HandlerSeedContext
    owned: frozenset[str]
    leading: tuple[tuple[Any, ...], ...]


def _apply(registry: HandlerRegistry, plugins: tuple[HandlerPlugin, ...], ctx: HandlerSeedContext) -> None:
    """Apply plugins' reasons and seeds, in order, into one registry."""
    for plugin in plugins:
        plugin.apply_reasons(registry)

    def seed(draft: HandlerRegistry) -> None:
        for plugin in plugins:
            plugin.apply(draft, ctx)

    registry._mutate_handlers(seed)


def _package_owned(selected: ModuleResolver) -> frozenset[str]:
    """Return the identifiers of the handler contributions the package itself installed."""
    from ...modules.manifest import _record_publisher

    return frozenset(str(row["identifier"]) for row in selected.inspect("handler") if _record_publisher(row["record"]) == _PACKAGE_PUBLISHER)


def _leading(plugins: tuple[HandlerPlugin, ...], owned: frozenset[str]) -> tuple[tuple[Any, ...], ...]:
    """Return what the leading run of package-owned contributions applies, in order.

    These are values rather than the plugin objects, which callers can hold
    and change, so a seeding is reused only for exactly the same callbacks.
    """
    leading: list[tuple[Any, ...]] = []
    for plugin in plugins:
        if plugin.identifier not in owned:
            break
        leading.append((plugin.identifier, plugin.priority, plugin.seed, plugin.reasons))
    return tuple(leading)


def _seeding(selected: ModuleResolver, plugins: tuple[HandlerPlugin, ...]) -> _Seeding:
    """Seed the leading package-owned contributions once; later ones run for each registry.

    Contributions apply in priority order, so only the leading run of the
    package's own contributions can be seeded ahead: their seeds read only
    the seeding snapshot and keep nothing of the registry they fill. Every
    contribution from the first one installed by anyone else on, including
    later package ones, runs again for each new registry, as before.
    """
    ctx = _handler_seed_context(selected)
    owned = _package_owned(selected)
    leading = _leading(plugins, owned)
    template = HandlerRegistry(resolver=selected)
    _apply(template, plugins[: len(leading)], ctx)
    return _Seeding(selected, int(ctx.filter_ops.publication), template, ctx, owned, leading)


def seed_handlers(*, resolver: ModuleResolver | None = None) -> HandlerRegistry:
    """Build the default workspace handler registry.

    The seed process reads raw canonical claims from Backend, logical-type,
    strategy, operation, and provider-plugin records. Extension packages can
    register new vocabulary before workspace construction and the seeded
    registry will include matching fallback or provider-native handlers.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        HandlerRegistry: Registry populated with generic scan handlers,
        vector-near defaults, HyperG mini-row predicates, and provider-native
        plugin overrides, bound to ``resolver``. Each call returns a new
        registry; registrations made in it stay its own. For a
        ``ModuleService`` whose catalog has not published since, the
        package's own leading contributions come from a copy of their last
        seeding, with handler descriptors of its own; every other
        contribution's seed runs again into the new registry, under the
        configuration in effect at the call. Any other resolver is seeded
        afresh.
    """
    from ...modules import ModuleService

    selected = module_resolver(resolver)
    derived_of = getattr(selected.catalog, "_derived", None)
    if not isinstance(selected, ModuleService) or derived_of is None:
        plugins = handler_plugins(resolver=selected)
        seeding = _seeding(selected, plugins)
        registry = seeding.template
    else:
        # The seeding is kept with the catalog publication it reflects and dies with it.
        publication, derived = derived_of()
        plugins = handler_plugins(resolver=selected)
        seeding = derived.get(_SEEDED_KEY)
        if seeding is None or seeding.resolver is not selected or seeding.leading != _leading(plugins, seeding.owned):
            seeding = _seeding(selected, plugins)
            if seeding.publication == publication:
                derived[_SEEDED_KEY] = seeding
        registry = seeding.template._copy()
    remaining = plugins[len(seeding.leading) :]
    if remaining:
        _apply(registry, remaining, copy.copy(seeding.context))
    return registry
