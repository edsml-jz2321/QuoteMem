"""Authored field placement rules."""

from __future__ import annotations

__all__ = ["StorageRule"]

from dataclasses import InitVar, KW_ONLY

from heavenbase.utils import Any, Dict, Mapping, dataclass, field

from ..modules import ModuleResolver
from ..strategies import InlineColumn, StorageStrategy


@dataclass
class StorageRule:
    """Declare storage intent for one field before workspace resolution.

    Args:
        backend (str | None): Target workspace backend name. ``None`` lets the
            resolver choose from the active storage profile.
        strategy (Any): Storage strategy class, instance, identifier, or alias.
        strategy_options (Dict[str, Any]): Explicit strategy-owned options.
        optional (bool): Whether an absent target backend skips this rule.
        reason (str): Human-readable placement source.
        params (Dict[str, Any]): Durable placement parameters such as graph
            endpoint metadata.
        resolver (ModuleResolver | None): Context-owned module resolver used to
            normalize the strategy. The default Context is selected when
            omitted.
    """

    backend: str | None = None
    strategy: Any = InlineColumn
    strategy_options: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False
    reason: str = "default"
    params: Dict[str, Any] = field(default_factory=dict)
    _: KW_ONLY
    resolver: InitVar[ModuleResolver | None] = None

    def __post_init__(self, resolver: ModuleResolver | None) -> None:
        """Normalize strategy identity, options, and placement parameters.

        Args:
            None.

        Returns:
            None: This initializer stores canonical strategy metadata in place.
        """
        if self.backend is not None:
            self.backend = str(self.backend).strip()
            if not self.backend:
                raise ValueError("StorageRule backend cannot be empty")
        self._module_resolver = resolver
        authored = self.strategy
        if isinstance(authored, StorageStrategy):
            self.params = {**authored.params, **self.params}
            options = authored.options()
            options.update(self.strategy_options)
            strategy_cls = type(authored)
        elif isinstance(authored, type):
            if not issubclass(authored, StorageStrategy):
                raise TypeError("StorageRule strategy identifier must be a StorageStrategy class, value, mapping, or string")
            options = dict(self.strategy_options)
            strategy_cls = authored
        elif self._module_resolver is None:
            if isinstance(authored, Mapping):
                identifier = StorageStrategy._mapping_identifier(authored)
                if not isinstance(identifier, str):
                    raise TypeError("StorageRule serialized strategy identifier must be a string")
                options = {str(key): value for key, value in authored.items() if key not in {"type", "strategy", "identifier"}}
                options.update(self.strategy_options)
            elif isinstance(authored, str):
                identifier = authored
                options = dict(self.strategy_options)
            else:
                raise TypeError("StorageRule strategy identifier must be a StorageStrategy class, value, mapping, or string")
            self.strategy = identifier.strip()
            if not self.strategy:
                raise ValueError("StorageRule strategy cannot be empty")
            self.strategy_options = {str(key): value for key, value in options.items() if value is not None}
            self.optional = bool(self.optional)
            self.reason = str(self.reason or "default")
            self.params = dict(self.params)
            return
        else:
            options = StorageStrategy.options_of(authored, resolver=self._module_resolver)
            options.update(self.strategy_options)
            strategy_cls = StorageStrategy.load(authored, resolver=self._module_resolver)
        self.strategy = strategy_cls.identifier
        self.strategy_options = strategy_cls.normalize_options(options)
        self.optional = bool(self.optional)
        self.reason = str(self.reason or "default")
        self.params = dict(self.params)

    def strategy_spec(self) -> str | dict[str, Any]:
        """Return the public serialized strategy shape.

        Args:
            None.

        Returns:
            str | dict[str, Any]: Identifier for an optionless strategy, or a
            strategy specification mapping.
        """
        if self._module_resolver is None:
            return {"type": self.strategy, **self.strategy_options} if self.strategy_options else self.strategy
        return StorageStrategy.spec_of(self.strategy, self.strategy_options, resolver=self._module_resolver)
