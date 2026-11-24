"""Handler plugin records for provider-native compilers and reasons."""

from __future__ import annotations

__all__ = [
    "HandlerPlugin",
    "HandlerSeedContext",
    "handler_plugins",
]

from collections.abc import Mapping
from dataclasses import InitVar
from typing import TYPE_CHECKING

from heavenbase.utils import Any, Callable, Optional, dataclass
from ...modules import ModuleResolver, module_resolver, python_executable_ref
from ...modules.base import ResolverCache
from ...utils.ids import check_id
from .base import Handler, HandlerCompiler, backend_identifier

if TYPE_CHECKING:
    from ...modules import ModuleService


@dataclass
class HandlerSeedContext:
    """Shared vocabulary passed to provider handler plugins.

    Args:
        filter_types (tuple[Any, ...]): Logical type classes with a non-empty
            canonical `filter-domains` candidate set.
        filter_strategies (tuple[str, ...]): Strategy identifiers claiming both
            `filter` and `portable`.
        vector_types (tuple[Any, ...]): Logical type classes claiming `near`;
            the historical field name is retained for provider plugin parity.
        filter_ops (Callable[[Any], tuple[str, ...]]): Function returning the
            operations whose canonical filter domains intersect one logical
            type's domains.
        resolver (ModuleResolver | None): Context-owned module resolver used by
            every typed facade during seeding. The default Context is selected
            for standalone construction.
    """

    filter_types: tuple[Any, ...]
    filter_strategies: tuple[str, ...]
    vector_types: tuple[Any, ...]
    filter_ops: Callable[[Any], tuple[str, ...]]
    resolver: ModuleResolver | None = None

    def __post_init__(self) -> None:
        """Normalize the optional standalone resolver to a concrete authority."""
        self.resolver = module_resolver(self.resolver)

    @property
    def claim_snapshot(self) -> str | None:
        """Return the snapshot token backing canonical claim reads, if present."""
        value = getattr(self.filter_ops, "snapshot", None)
        return value if isinstance(value, str) else None

    def identifiers(self, kind: str) -> tuple[str, ...]:
        """Return exact-kind identifiers from the shared seeding snapshot.

        Args:
            kind (str): Module kind whose identifiers should be listed.

        Returns:
            tuple[str, ...]: Canonical identifiers visible in the seeding snapshot.
        """
        reader = getattr(self.filter_ops, "identifiers", None)
        if callable(reader):
            return tuple(str(identifier) for identifier in reader(kind))
        return tuple(str(row["identifier"]) for row in self.resolver.inspect(kind))

    def raw_tags(self, kind: str, identifier: str) -> Mapping[str, Any]:
        """Return one record's exact static claims from the seeding snapshot.

        Args:
            kind (str): Module kind owning the record.
            identifier (str): Canonical record identifier.

        Returns:
            Mapping[str, Any]: Exact static tag claims, or an empty mapping when absent.
        """
        reader = getattr(self.filter_ops, "raw_tags", None)
        if callable(reader):
            value = reader(kind, identifier)
        else:
            rows = self.resolver.inspect(kind, identifier)
            meta = rows[0]["record"].get("meta") if rows else None
            value = meta.get("tags", {}) if isinstance(meta, Mapping) else {}
        if not isinstance(value, Mapping) or not all(isinstance(tag, str) for tag in value):
            return {}
        return value

    def handler(
        self,
        dtype: Any,
        op: str,
        backend_identifier: Any,
        strategy: Any,
        compile: HandlerCompiler,
    ) -> Handler:
        """Build a handler from this context's coherent canonical snapshot.

        Args:
            dtype (Any): Logical type class, instance, identifier, or alias.
            op (str): Operation identifier or alias.
            backend_identifier (Any): Backend type identifier or alias.
            strategy (Any): Storage strategy class, instance, identifier, or alias.
            compile (HandlerCompiler): Pure compiler returning a ``QueryFragment``.

        Returns:
            Handler: Handler descriptor normalized against the seeding snapshot.
        """
        canonical = getattr(self.filter_ops, "canonical_identifier", None)
        values = (dtype, op, backend_identifier, strategy)
        kinds = ("logical_type", "operation", "backend_type", "strategy")
        identifiers = tuple(str(getattr(value, "identifier", value)) for value in values)
        if callable(canonical):
            normalized = tuple(canonical(kind, identifier) for kind, identifier in zip(kinds, identifiers))
            if all(identifier is not None for identifier in normalized):
                return Handler._from_canonical(*normalized, compile, resolver=self.resolver)
        return Handler(dtype, op, backend_identifier, strategy, compile, resolver=self.resolver)


@dataclass
class HandlerPlugin:
    """Seed provider-owned handlers and optional routing reasons.

    Args:
        identifier (str): Stable plugin identifier, usually the backend type id.
        seed (Callable[[Any, HandlerSeedContext], None]): Callback
            that registers provider-native compilers on a `HandlerRegistry`.
        reasons (Optional[Callable[..., None]]): Optional callback
            that registers fallback and unsupported-operation reason text.
        priority (int): Seed order. Lower values run first so later,
            provider-specific contributions can replace generic handlers.
        resolver (ModuleResolver | None): Resolver used to normalize the plugin
            identifier during construction. It is not retained by the descriptor.
    """

    identifier: str
    seed: Callable[[Any, HandlerSeedContext], None]
    reasons: Optional[Callable[..., None]] = None
    priority: int = 100
    resolver: InitVar[ModuleResolver | None] = None

    def __post_init__(self, resolver: ModuleResolver | None) -> None:
        """Normalize the identifier and validate plugin callbacks.

        Args:
            resolver: Context-owned module resolver used for normalization.

        Returns:
            None: This initializer stores a canonical plugin identifier.

        Raises:
            TypeError: If a plugin callback is not callable.
        """
        selected = module_resolver(resolver)
        try:
            self.identifier = backend_identifier(self.identifier, resolver=selected)
        except KeyError:
            self.identifier = check_id(self.identifier, "handler plugin identifier")
        if not callable(self.seed):
            raise TypeError("HandlerPlugin seed must be callable")
        if self.reasons is not None and not callable(self.reasons):
            raise TypeError("HandlerPlugin reasons must be callable")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("HandlerPlugin priority must be an integer")

    def register(
        self,
        *,
        replace: bool = False,
        resolver: ModuleResolver | None = None,
    ) -> "HandlerPlugin":
        """Publish this handler contribution as one central record.

        Args:
            replace (bool): Whether to replace an existing plugin. Defaults to
                ``False``.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted. Registration
                requires a `ModuleService`.

        Returns:
            HandlerPlugin: This registered plugin descriptor.
        """
        selected = module_resolver(resolver)
        service = _module_service(selected)
        found = service.inspect("handler", self.identifier)
        coordinate = f"programmatic/handler/{self.identifier}"
        record = _plugin_record(self)
        if found:
            existing = _plugin_from_row(found[0], selected)
            if existing == self:
                return existing
            if not replace:
                raise ValueError(f"Handler contribution {self.identifier!r} is already registered")
            existing_coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
            if existing_coordinate != coordinate:
                raise ValueError(f"Handler contribution {self.identifier!r} belongs to {existing_coordinate!r} and cannot be replaced")
        service.register(
            "handler",
            self.identifier,
            record,
            coordinate=coordinate,
        )
        _HANDLER_PLUGIN_CACHE.clear(selected)
        return _plugin_from_row(service.inspect("handler", self.identifier)[0], selected)

    def apply(self, registry: Any, ctx: HandlerSeedContext) -> None:
        """Seed this plugin's handlers into one handler registry.

        `seed_handlers` calls this for each new workspace registry, with
        the configuration in effect at the call. The exception is the
        leading run of contributions that the HeavenBase package itself
        installs: those are seeded once per catalog publication into a
        template, and each workspace of the Context gets a copy of it.

        Args:
            registry (Any): Handler registry receiving descriptors.
            ctx (HandlerSeedContext): Shared seeding vocabulary.

        Returns:
            None: This method delegates registration to the provider callback.
        """
        self.seed(registry, ctx)
        return None

    def apply_reasons(self, registry: Any) -> None:
        """Register this contribution's reasons on one owning registry.

        Args:
            registry (Any): Workspace handler registry receiving reason catalogs.

        Returns:
            None: This function does not return a value.
        """
        if self.reasons is not None:
            registry.register_reasons(self.reasons)


def _module_service(resolver: ModuleResolver) -> "ModuleService":
    """Require lifecycle authority for a handler-plugin mutation."""
    from ...modules import ModuleService

    if not isinstance(resolver, ModuleService):
        raise TypeError("Handler plugin mutation requires a ModuleService resolver")
    return resolver


def _plugin_record(plugin: HandlerPlugin) -> dict[str, Any]:
    """Return one inert handler-contribution registration record."""
    return {
        "name": plugin.identifier.replace("-", " ").title(),
        "desc": (plugin.seed.__doc__ or "Registered handler compiler contribution.").strip().splitlines()[0],
        "source": "inline",
        "target": "definition",
        "active": True,
        "meta": {
            "schema_version": 1,
            "priority": plugin.priority,
            "definition": {
                "seed": python_executable_ref(plugin.seed, label=f"Handler contribution {plugin.identifier!r} seed"),
                "reasons": (
                    None if plugin.reasons is None else python_executable_ref(plugin.reasons, label=f"Handler contribution {plugin.identifier!r} reasons")
                ),
            },
        },
    }


_HANDLER_PLUGIN_CACHE = ResolverCache[HandlerPlugin]()


def _plugin_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> HandlerPlugin:
    """Materialize and cache one validated handler contribution."""

    def build(current_row: Mapping[str, Any]) -> HandlerPlugin:
        identifier = str(current_row["identifier"])
        raw_meta = current_row["record"].get("meta")
        definition = resolver.resolve("handler", identifier)
        if callable(definition):
            seed = definition
            reasons = None
        elif isinstance(definition, Mapping) and set(definition) == {"seed", "reasons"}:
            seed_ref = definition.get("seed")
            reasons_ref = definition.get("reasons")
            if not isinstance(seed_ref, Mapping):
                raise TypeError(f"Handler contribution {identifier!r} seed must be an executable reference")
            seed = resolver.resolve_executable(seed_ref, owner=current_row)
            if reasons_ref is None:
                reasons = None
            elif isinstance(reasons_ref, Mapping):
                reasons = resolver.resolve_executable(reasons_ref, owner=current_row)
            else:
                raise TypeError(f"Handler contribution {identifier!r} reasons must be an executable reference or null")
        else:
            raise ValueError(f"Handler contribution {identifier!r} must resolve to a seed callback or seed/reasons mapping")
        priority = raw_meta.get("priority", 100) if isinstance(raw_meta, Mapping) else 100
        return HandlerPlugin(identifier, seed, reasons, priority, resolver=resolver)

    return _HANDLER_PLUGIN_CACHE.get(resolver, row, build)


def handler_plugins(*, resolver: ModuleResolver | None = None) -> tuple[HandlerPlugin, ...]:
    """Return registered handler plugins in deterministic order.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        tuple[HandlerPlugin, ...]: Registered plugin descriptors sorted by id.
    """
    selected = module_resolver(resolver)
    plugins = tuple(_plugin_from_row(row, selected) for row in selected.inspect("handler"))
    return tuple(sorted(plugins, key=lambda plugin: (plugin.priority, plugin.identifier)))
