"""Logical-type default placement policy."""

from __future__ import annotations

__all__ = ["StorageProfile"]

from dataclasses import InitVar, KW_ONLY

from heavenbase.utils import Any, Dict, Mapping, Sequence, dataclass, field

from ..modules import ModuleResolver, module_resolver
from ..strategies import StorageStrategy
from ..types import Array
from ..types.descriptor import logical_type_from_any
from ..types.registry import get_logical_type
from ..utils.ids import check_id

_MISSING = object()


def _profile_keys(dtype: Any, resolver: ModuleResolver | None) -> tuple[str, ...]:
    if dtype == "*":
        return ("*",)
    if isinstance(dtype, type):
        return (dtype.identifier,)
    if isinstance(dtype, str):
        try:
            dtype = logical_type_from_any(dtype, resolver=resolver)
        except (KeyError, TypeError, ValueError):
            return (get_logical_type(dtype, resolver=resolver).identifier,)
    base = type(dtype).identifier
    if base == Array.identifier and getattr(dtype, "item_type", None) is not None:
        return (f"{base}[{type(dtype.item_type).identifier}]", base)
    return (base,)


@dataclass(frozen=True)
class _TagCondition:
    """One exact placement-owned test over a raw static tag claim."""

    tag: str
    operator: str
    expected: Any

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "_TagCondition":
        """Validate one direct tag condition.

        Args:
            raw (Mapping[str, Any]): Raw tag-condition mapping.

        Returns:
            _TagCondition: Validated immutable tag condition.
        """
        if not isinstance(raw, Mapping):
            raise TypeError("storage profile tag conditions must be mappings")
        tag = check_id(str(raw.get("tag") or ""), "storage profile tag")
        operators = [name for name in ("equals", "contains") if name in raw]
        if len(operators) != 1 or set(raw) != {"tag", operators[0]}:
            raise ValueError("storage profile tag conditions require exactly one of equals or contains")
        return cls(tag, operators[0], raw[operators[0]])

    def matches(self, claims: Mapping[str, Any]) -> bool:
        """Return whether raw claims satisfy this local condition.

        Args:
            claims (Mapping[str, Any]): Raw static tag claims for one backend.

        Returns:
            bool: Whether the condition matches ``claims``.
        """
        value = claims.get(self.tag, _MISSING)
        if value is _MISSING:
            return False
        if self.operator == "equals":
            return type(value) is type(self.expected) and value == self.expected
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return False
        return any(type(item) is type(self.expected) and item == self.expected for item in value)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible condition.

        Args:
            None.

        Returns:
            dict[str, Any]: Tag, operator, and expected value payload.
        """
        return {"tag": self.tag, self.operator: self.expected}


@dataclass(frozen=True)
class _TagPredicate:
    """A bounded conjunction of placement tag conditions."""

    conditions: tuple[_TagCondition, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "_TagPredicate":
        """Validate one direct condition or one bounded conjunction.

        Args:
            raw (Mapping[str, Any]): Raw predicate mapping or ``{"all": [...]}`` form.

        Returns:
            _TagPredicate: Validated immutable tag predicate.
        """
        if not isinstance(raw, Mapping):
            raise TypeError("storage profile backend predicates must be mappings")
        if "all" not in raw:
            return cls((_TagCondition.from_mapping(raw),))
        if set(raw) != {"all"}:
            raise ValueError("storage profile conjunctions may contain only the 'all' field")
        conditions = raw["all"]
        if isinstance(conditions, (str, bytes)) or not isinstance(conditions, Sequence) or not conditions:
            raise TypeError("storage profile conjunction 'all' must be a non-empty list")
        return cls(tuple(_TagCondition.from_mapping(condition) for condition in conditions))

    def matches(self, claims: Mapping[str, Any]) -> bool:
        """Return whether every condition matches raw static claims.

        Args:
            claims (Mapping[str, Any]): Raw static tag claims for one backend.

        Returns:
            bool: Whether every nested condition matches.
        """
        return all(condition.matches(claims) for condition in self.conditions)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible predicate.

        Args:
            None.

        Returns:
            dict[str, Any]: Single-condition or ``{"all": [...]}`` predicate payload.
        """
        if len(self.conditions) == 1:
            return self.conditions[0].to_dict()
        return {"all": [condition.to_dict() for condition in self.conditions]}


@dataclass(frozen=True)
class _StrategyRule:
    """One ordered placement predicate and resulting strategy."""

    when: _TagPredicate
    strategy: str

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None,
    ) -> "_StrategyRule":
        """Validate and normalize one strategy rule.

        Args:
            raw (Mapping[str, Any]): Raw ``when``/``strategy`` rule mapping.
            resolver (ModuleResolver | None): Module resolver used to normalize
                strategy identifiers. Defaults to the process Context resolver.

        Returns:
            _StrategyRule: Validated immutable strategy rule.
        """
        if not isinstance(raw, Mapping) or set(raw) != {"when", "strategy"}:
            raise ValueError("storage profile strategy rules require exactly when and strategy")
        return cls(
            _TagPredicate.from_mapping(raw["when"]),
            StorageStrategy.identifier_of(raw["strategy"], resolver=resolver),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible strategy rule.

        Args:
            None.

        Returns:
            dict[str, Any]: ``when`` predicate and resolved ``strategy`` identifier.
        """
        return {"when": self.when.to_dict(), "strategy": self.strategy}


@dataclass
class StorageProfile:
    """Choose an ordered Backend tag policy and strategy for one logical type.

    Args:
        dtype (Any): Logical type class, instance, identifier, or descriptor.
        backend_preferences (Sequence[Mapping[str, Any]]): Ordered static
            Backend tag predicates.
        default_strategy (Any): Strategy used when no strategy rule matches.
        reason (str): Source surfaced on resolved bindings and metadata rows.
        strategy_rules (Sequence[Mapping[str, Any]]): Ordered conditional
            strategy selections.
        resolver (ModuleResolver | None): Context-owned module resolver used to
            normalize registered logical types and strategies. The default
            Context is selected when omitted.
    """

    dtype: Any
    backend_preferences: Sequence[Mapping[str, Any]]
    default_strategy: Any
    reason: str
    strategy_rules: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    _: KW_ONLY
    resolver: InitVar[ModuleResolver | None] = None

    def __post_init__(self, resolver: ModuleResolver | None) -> None:
        """Normalize profile identity and policy values.

        Args:
            None.

        Returns:
            None: This initializer stores canonical immutable metadata.

        Raises:
            ValueError: If a predicate or strategy rule is malformed.
        """
        self.dtype = self.keys_for(self.dtype, resolver=resolver)[0]
        if isinstance(self.backend_preferences, (str, bytes)) or not isinstance(self.backend_preferences, Sequence):
            raise TypeError("storage profile backend_preferences must be a sequence of predicates")
        if not self.backend_preferences:
            raise ValueError("storage profile backend_preferences must not be empty")
        self.backend_preferences = tuple(_TagPredicate.from_mapping(raw) for raw in self.backend_preferences)
        self.default_strategy = StorageStrategy.identifier_of(self.default_strategy, resolver=resolver)
        self.reason = str(self.reason or "")
        if isinstance(self.strategy_rules, (str, bytes)) or not isinstance(self.strategy_rules, Sequence):
            raise TypeError("storage profile strategy_rules must be a sequence of mappings")
        self.strategy_rules = tuple(_StrategyRule.from_mapping(raw, resolver=resolver) for raw in self.strategy_rules)

    @property
    def identifier(self) -> str:
        """Return the logical-type lookup key owned by this profile.

        Args:
            None.

        Returns:
            str: Canonical logical-type or parametrized-type key.
        """
        return self.dtype

    @staticmethod
    def keys_for(
        dtype: Any,
        *,
        resolver: ModuleResolver | None = None,
    ) -> tuple[str, ...]:
        """Return profile lookup keys from most specific to least specific.

        Args:
            dtype (Any): Logical type class, instance, identifier, or descriptor.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            tuple[str, ...]: Canonical lookup keys. Parametrized arrays include
            an item-specific key before the generic array key.
        """
        return _profile_keys(dtype, resolver)

    @classmethod
    def from_mapping(
        cls,
        dtype: Any,
        raw: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
    ) -> "StorageProfile":
        """Build a storage profile from registry or MetaSchema metadata.

        Args:
            dtype (Any): Logical type that owns the profile.
            raw (Mapping[str, Any]): Direct profile payload or row containing
                profile metadata under ``meta``.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            StorageProfile: Normalized profile value.
        """
        key = cls.keys_for(dtype, resolver=resolver)[0]
        meta = raw.get("meta") if isinstance(raw, Mapping) else None
        data = meta if isinstance(meta, Mapping) else raw
        backend_preferences = data.get("backend_preferences") or raw.get("backend_preferences")
        default_strategy = data.get("default_strategy") or raw.get("default_strategy")
        reason = data.get("reason") or raw.get("source") or f"workspace-{key}"
        strategy_rules = data.get("strategy_rules") or raw.get("strategy_rules") or ()
        return cls(dtype, backend_preferences, default_strategy, reason, strategy_rules, resolver=resolver)

    def register(
        self,
        *,
        replace: bool = False,
        resolver: ModuleResolver | None = None,
    ) -> "StorageProfile":
        """Register this default storage profile in `system-modules`.

        Args:
            replace (bool): Whether to replace an existing profile. Defaults to
                ``False``.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            StorageProfile: This registered profile.
        """
        from .profiles import _register_profile

        return _register_profile(self, replace=replace, resolver=module_resolver(resolver))

    def strategy_for(self, claims: Mapping[str, Any]) -> str:
        """Return the strategy selected for one concrete backend.

        Args:
            claims (Mapping[str, Any]): Raw static claims from the selected
                Backend type's authoritative record.

        Returns:
            str: First matching strategy rule or the default strategy.
        """
        for rule in self.strategy_rules:
            if rule.when.matches(claims):
                return rule.strategy
        return self.default_strategy

    def to_dict(self) -> Dict[str, object]:
        """Serialize this storage profile.

        Args:
            None.

        Returns:
            Dict[str, object]: JSON-compatible profile metadata.
        """
        return {
            "dtype": self.dtype,
            "backend_preferences": [predicate.to_dict() for predicate in self.backend_preferences],
            "default_strategy": self.default_strategy,
            "reason": self.reason,
            "strategy_rules": [rule.to_dict() for rule in self.strategy_rules],
        }
