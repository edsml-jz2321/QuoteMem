"""Identifier logical type."""

from __future__ import annotations

__all__ = [
    "Identifier",
]

from heavenbase.utils import Any

from ..modules import ModuleResolver
from .limits import _BoundedText


def _target_entity_id(entity: Any) -> Any:
    """Resolve a target-entity argument to its ``entity_id`` string.

    Accepts ``None``, an ``entity_id`` string, or an Entity class/instance (read
    from ``entity_id``/``identifier``). Kept lenient so it does not import the
    entity layer.
    """
    if entity is None or isinstance(entity, str):
        return entity
    for attr in ("entity_id", "identifier"):
        value = getattr(entity, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(entity)


class Identifier(_BoundedText):
    """Stable non-empty identifier logical type.

    ``Identifier[Entity]`` annotates the id with the entity it points at. The
    annotation is schema-level metadata used only when resolving the target
    (for example a GraphEdge endpoint's Neo4j label); the stored value stays the
    bare ``object_id`` string, so a plain ``Identifier`` and an
    ``Identifier[Entity]`` persist identically.

    Args:
        target_entity (Any): Optional target entity identifier or Entity-like
            value exposing `entity_id` or `identifier`.
        max_chars (int | None): Positive maximum character count. `None` uses
            the stable built-in default of 63.
    """

    identifier = "identifier"
    default_max_chars = 63

    @classmethod
    def from_dict(cls, spec: dict, *, resolver: ModuleResolver | None = None) -> "Identifier":
        """Build an Identifier from a durable descriptor.

        Args:
            spec (dict): Descriptor with optional `target_entity` and positive
                `max_chars` values.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            Identifier: Configured identifier descriptor.

        Raises:
            TypeError: If `spec` or `max_chars` has an invalid type.
            ValueError: If the selector, descriptor keys, or limit is invalid.
        """
        cls._check_descriptor(spec, resolver=resolver, extra=("target_entity",))
        return cls(spec.get("target_entity"), max_chars=spec.get("max_chars"))

    def __init__(self, target_entity: Any = None, *, max_chars: int | None = None) -> None:
        """Initialize identifier bounds and optional target annotation."""
        super().__init__(max_chars=max_chars)
        self._target_entity = _target_entity_id(target_entity)

    def __class_getitem__(cls, entity: Any) -> "Identifier":
        return cls(entity)

    @property
    def target_entity(self) -> Any:
        """Return the immutable target-entity annotation, when present."""
        return self._target_entity

    def validate(self, value: Any) -> str:
        """Validate a non-empty identifier string within this descriptor.

        Args:
            value (Any): Candidate identifier value.

        Returns:
            str: Non-empty identifier within `max_chars`.

        Raises:
            ValueError: If the identifier is null, empty, or too long.
        """
        if value is None:
            raise ValueError("Identifier cannot be None")
        text = str(value)
        if not text:
            raise ValueError("Identifier cannot be empty")
        if len(text) > self.max_chars:
            raise ValueError(f"Identifier exceeds {self.max_chars} characters")
        return text

    def to_dict(self) -> dict[str, Any]:
        """Return the complete durable Identifier descriptor.

        Args:
            None.

        Returns:
            dict[str, Any]: Type, concrete length, and optional target entity.
        """
        out = super().to_dict()
        if self.target_entity is not None:
            out["target_entity"] = self.target_entity
        return out

    def clone(self) -> "Identifier":
        """Return a new Identifier carrying identical schema semantics.

        Args:
            None.

        Returns:
            Identifier: Independent descriptor with the same target and limit.
        """
        return type(self)(self.target_entity, max_chars=self.max_chars)
