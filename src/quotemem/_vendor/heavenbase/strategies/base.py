"""Storage strategy value contract."""

from __future__ import annotations

__all__ = ["StorageStrategy"]

from heavenbase.utils import Any, Mapping, Sequence, raise_mismatch
from heavenbase.utils.registry import class_identifier

from ..modules import ModuleResolver, module_resolver


class StorageStrategy:
    """Describe immutable field placement or index semantics.

    Strategy values contain only a stable identifier and strategy-owned options.
    They never hold provider clients, perform IO, compile queries, or choose a
    backend. Storage bindings persist their serialized values; handlers use the
    identifier in capability keys; backend-hosted strategy adapters implement
    provider behavior.

    Attributes:
        identifier (str): Canonical lower-hyphen strategy identifier.
        option_fields (tuple[str, ...]): Accepted strategy-owned option names.
    """

    identifier = "storage-strategy"
    option_fields: tuple[str, ...] = ()

    def __init__(self, **options: Any) -> None:
        """Create a strategy value with normalized options.

        Args:
            **options (Any): Strategy-owned values. Names must appear in
                `option_fields`; values set to `None` are discarded.

        Returns:
            None: This initializer stores normalized strategy options.

        Raises:
            ValueError: If an option is not declared in `option_fields`.
        """
        self._options = self.normalize_options(options)

    @property
    def params(self) -> dict[str, Any]:
        """Return per-field placement parameters.

        Strategy options describe reusable strategy semantics. Placement params
        describe one field binding, such as a `GraphEdge` endpoint, and remain a
        separate payload.

        Returns:
            dict[str, Any]: Per-field placement parameters. Base strategies
            return an empty mapping.
        """
        return {}

    @classmethod
    def register(
        cls,
        tags: Mapping[str, Any] | None = None,
        aliases: Sequence[str] = (),
        *,
        resolver: ModuleResolver | None = None,
    ) -> type["StorageStrategy"]:
        """Register this storage strategy class.

        Args:
            tags (Mapping[str, Any] | None): Raw JSON-compatible Tag claims
                copied onto the published strategy record.
            aliases (Sequence[str]): Additional accepted lookup names.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            type[StorageStrategy]: This registered strategy class.

        Raises:
            ValueError: If its identifier is invalid.
        """
        from .registry import register_strategy

        return register_strategy(cls, tags=tags, aliases=aliases, resolver=resolver)

    @classmethod
    def load(cls, key: Any, *, resolver: ModuleResolver | None = None) -> type["StorageStrategy"]:
        """Load a registered strategy class.

        Args:
            key (Any): Identifier, alias, strategy class, strategy value, or
                serialized mapping with `type`, `strategy`, or `identifier`.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            type[StorageStrategy]: Registered strategy class.

        Raises:
            KeyError: If the identifier or alias is unknown.
            TypeError: If `key` cannot identify a strategy.
            ValueError: If a mapping lacks an identifier field.
        """
        selected = module_resolver(resolver)
        if isinstance(key, Mapping):
            key = cls._mapping_identifier(key)
        if isinstance(key, StorageStrategy):
            key = type(key)
        from .registry import _load_strategy

        return _load_strategy(key, selected)

    @classmethod
    def identifier_of(cls, strategy: Any, *, resolver: ModuleResolver | None = None) -> str:
        """Return a strategy's canonical identifier.

        Args:
            strategy (Any): Strategy class, value, identifier, alias, or
                serialized mapping.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            str: Canonical lower-hyphen identifier.

        Raises:
            KeyError: If an identifier or alias is unknown.
            TypeError: If `strategy` cannot identify a strategy.
            ValueError: If a mapping lacks an identifier field.
        """
        if isinstance(strategy, Mapping):
            return cls.identifier_of(cls._mapping_identifier(strategy), resolver=resolver)
        if isinstance(strategy, StorageStrategy):
            return class_identifier(type(strategy), label="strategy identifier")
        if isinstance(strategy, type) and issubclass(strategy, StorageStrategy):
            return class_identifier(strategy, label="strategy identifier")
        return cls.load(strategy, resolver=resolver).identifier

    @classmethod
    def value_of(cls, strategy: Any, *, resolver: ModuleResolver | None = None) -> "StorageStrategy":
        """Return a concrete strategy value.

        Args:
            strategy (Any): Strategy class, value, identifier, alias, or
                serialized mapping with strategy-owned options.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            StorageStrategy: Existing or newly constructed strategy value.

        Raises:
            KeyError: If an identifier or alias is unknown.
            TypeError: If the input cannot identify a strategy.
            ValueError: If serialized options are invalid.
        """
        if isinstance(strategy, StorageStrategy):
            return strategy
        strategy_cls = cls.load(strategy, resolver=resolver)
        return strategy_cls(**cls.options_of(strategy, resolver=resolver))

    @classmethod
    def options_of(cls, strategy: Any, *, resolver: ModuleResolver | None = None) -> dict[str, Any]:
        """Return normalized options from a strategy value or spec.

        Args:
            strategy (Any): Strategy value, serialized mapping, strategy class,
                identifier, or alias.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            dict[str, Any]: Normalized strategy-owned options.

        Raises:
            KeyError: If a serialized mapping names an unknown strategy.
            ValueError: If a serialized mapping contains unsupported options.
        """
        if isinstance(strategy, StorageStrategy):
            return strategy.options()
        if isinstance(strategy, Mapping):
            strategy_cls = cls.load(strategy, resolver=resolver)
            options = {str(key): value for key, value in strategy.items() if key not in {"type", "strategy", "identifier"}}
            return strategy_cls.normalize_options(options)
        return {}

    @classmethod
    def spec_of(
        cls,
        strategy: Any,
        options: Mapping[str, Any] | None = None,
        *,
        resolver: ModuleResolver | None = None,
    ) -> str | dict[str, Any]:
        """Return a strategy's public JSON shape.

        Args:
            strategy (Any): Strategy class, value, identifier, alias, or
                serialized mapping.
            options (Mapping[str, Any] | None): Strategy-owned options merged
                over options already present on `strategy`.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            str | dict[str, Any]: Canonical identifier when no options exist,
            otherwise a mapping with `type` and normalized options.

        Raises:
            KeyError: If an identifier or alias is unknown.
            ValueError: If any option is unsupported.
        """
        identifier = cls.identifier_of(strategy, resolver=resolver)
        strategy_options = cls.options_of(strategy, resolver=resolver)
        if options:
            strategy_cls = (
                type(strategy) if isinstance(strategy, StorageStrategy) else strategy if isinstance(strategy, type) else cls.load(identifier, resolver=resolver)
            )
            strategy_options.update(strategy_cls.normalize_options(options))
        return {"type": identifier, **strategy_options} if strategy_options else identifier

    @classmethod
    def normalize_options(cls, options: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and normalize strategy-owned options.

        Args:
            options (Mapping[str, Any]): Raw option mapping.

        Returns:
            dict[str, Any]: Declared options in `option_fields` order with
            `None` values omitted.

        Raises:
            ValueError: If an option is not declared in `option_fields`.
        """
        data = {str(key): value for key, value in dict(options).items() if value is not None}
        unknown = sorted(set(data) - set(cls.option_fields))
        if unknown:
            raise_mismatch(
                cls.option_fields,
                unknown[0],
                name=f"{cls.identifier} strategy option",
                comment=f"Allowed option_fields for {cls.__name__}: {cls.option_fields!r}.",
            )
        return {key: data[key] for key in cls.option_fields if key in data}

    @staticmethod
    def _mapping_identifier(data: Mapping[str, Any]) -> Any:
        for key in ("type", "strategy", "identifier"):
            if data.get(key):
                return data[key]
        raise ValueError("Serialized strategy specs need a type field")

    def options(self) -> dict[str, Any]:
        """Return a copy of this value's strategy-owned options.

        Args:
            None.

        Returns:
            dict[str, Any]: Normalized option payload.
        """
        return dict(self._options)

    def to_dict(self) -> dict[str, Any]:
        """Return this strategy as a JSON-compatible spec.

        Args:
            None.

        Returns:
            dict[str, Any]: Mapping with `type` and strategy-owned options.
        """
        return {"type": self.identifier_of(self), **self.options()}

    @classmethod
    def aliases(cls, strategy: Any, *, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
        """Return accepted aliases for a registered strategy.

        Args:
            strategy (Any): Strategy class, value, identifier, alias, or spec.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            tuple[str, ...]: Registered aliases excluding the canonical id.
        """
        from .registry import strategy_aliases

        return strategy_aliases(strategy, resolver=resolver)
