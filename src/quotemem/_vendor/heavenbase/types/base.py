"""Logical type value contract."""

from __future__ import annotations

__all__ = ["LogicalType"]

from heavenbase.utils import Any, Mapping, Sequence, raise_mismatch

from ..modules import ModuleResolver


class LogicalType:
    """Define field value semantics independently of physical storage.

    A logical type validates user input, exposes one canonical Python value,
    converts that value at backend codec boundaries, and normalizes query
    operands. It does not choose storage placement, inspect workspaces, compile
    provider queries, or hold backend clients.

    Attributes:
        identifier (str): Canonical lower-hyphen type identifier.
    """

    identifier = "logical-type"

    @classmethod
    def register(
        cls,
        tags: Mapping[str, Any] | None = None,
        aliases: Sequence[str] = (),
        *,
        resolver: ModuleResolver | None = None,
    ) -> type["LogicalType"]:
        """Register this logical type class.

        Args:
            tags (Mapping[str, Any] | None): Raw JSON-compatible Tag claims
                copied onto the published logical-type record.
            aliases (Sequence[str]): Additional accepted descriptor names.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            type[LogicalType]: This registered logical type class.

        Raises:
            ValueError: If the class identifier or an alias is invalid.
        """
        from .registry import register_logical_type

        return register_logical_type(cls, tags=tags, aliases=aliases, resolver=resolver)

    @classmethod
    def from_any(cls, raw: Any, *, resolver: ModuleResolver | None = None) -> "LogicalType":
        """Build a logical type from any supported descriptor.

        Args:
            raw (Any): Logical type class, instance, identifier, alias,
                parametric shorthand, or JSON-compatible descriptor mapping.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            LogicalType: Configured logical type instance. A concrete subclass
            accepts only descriptors that resolve to that subclass.

        Raises:
            TypeError: If `raw` has an unsupported shape or resolves to a
                different concrete subclass.
            ValueError: If the descriptor is unknown or under-specified.
        """
        from .descriptor import logical_type_from_any

        dtype = logical_type_from_any(raw, resolver=resolver)
        if cls is LogicalType or isinstance(dtype, cls):
            return dtype
        raise TypeError(f"{cls.__name__}.from_any expected {cls.identifier!r}, got {dtype.to_str()!r}")

    @classmethod
    def from_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
    ) -> "LogicalType":
        """Build this logical type from a JSON-compatible descriptor.

        The base implementation accepts only selector keys. Parameterized types
        override this method and own every additional descriptor field.

        Args:
            spec (Mapping[str, Any]): Descriptor with `type`, `identifier`, or
                `dtype` plus type-specific parameters.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            LogicalType: Configured logical type instance.

        Raises:
            TypeError: If `spec` is not a mapping.
            ValueError: If a concrete subclass receives an unexpected selector
                or parameter.
        """
        if not isinstance(spec, Mapping):
            raise TypeError(f"{cls.__name__}.from_dict expects a mapping")
        if cls is LogicalType:
            from .descriptor import logical_type_from_mapping

            return logical_type_from_mapping(spec, resolver=resolver)
        key = spec.get("type") or spec.get("identifier") or spec.get("dtype")
        if key is not None:
            if isinstance(key, LogicalType):
                resolved = type(key)
            elif isinstance(key, type) and issubclass(key, LogicalType):
                resolved = key
            else:
                resolved = cls.load(key, resolver=resolver)
            if resolved is not cls:
                raise ValueError(f"{cls.__name__}.from_dict expected {cls.identifier!r}, got {resolved.identifier!r}")
        allowed = ("type", "identifier", "dtype")
        extras = sorted(set(spec) - set(allowed))
        if extras:
            raise_mismatch(
                allowed,
                extras[0],
                name=f"{cls.identifier} descriptor key",
                comment=f"Override {cls.__name__}.from_dict(...) to accept type-specific parameters.",
            )
        return cls()

    def validate(self, value: Any) -> Any:
        """Normalize user input to canonical Python.

        Args:
            value (Any): Raw user or interop input.

        Returns:
            Any: Canonical Python value for this logical type.

        Raises:
            ValueError: If `value` cannot be normalized.
            TypeError: If `value` has an unsupported input type.
        """
        return value

    def encode(self, value: Any) -> Any:
        """Convert canonical Python to a backend-ready value.

        The default re-validates and returns the canonical value unchanged.

        Args:
            value (Any): Canonical Python value, usually already validated.

        Returns:
            Any: Value ready for backend persistence.

        Raises:
            ValueError: If `value` cannot be encoded.
            TypeError: If `value` has an unsupported input type.
        """
        return self.validate(value)

    def decode(self, value: Any) -> Any:
        """Restore a backend value to canonical Python.

        Args:
            value (Any): Raw value read from a backend.

        Returns:
            Any: Canonical Python value exposed to callers.
        """
        return value

    def filter_value(self, op: str, value: Any) -> Any:
        """Normalize a query operand for one operation.

        Args:
            op (str): Operation identifier such as `eq`, `in`, or `contains`.
            value (Any): User-provided query operand.

        Returns:
            Any: Canonical operand consumed by a handler compiler.
        """
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible logical type descriptor.

        Args:
            None.

        Returns:
            dict[str, Any]: Descriptor containing `type` and any parameters.
        """
        return {"type": type(self).identifier}

    def to_str(self) -> str:
        """Return the canonical logical type identifier.

        Args:
            None.

        Returns:
            str: Canonical lower-hyphen identifier.
        """
        return type(self).identifier

    @classmethod
    def spec_of(cls, dtype: Any, *, resolver: ModuleResolver | None = None) -> dict[str, Any]:
        """Return a normalized public descriptor for a logical type.

        Args:
            dtype (Any): Logical type class, instance, identifier, alias,
                shorthand, or descriptor mapping.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            dict[str, Any]: Normalized descriptor containing a `type` key.

        Raises:
            TypeError: If `dtype` has an unsupported shape.
            ValueError: If the descriptor is unknown or under-specified.
        """
        return cls.from_any(dtype, resolver=resolver).to_dict()

    @classmethod
    def load(cls, key: Any, *, resolver: ModuleResolver | None = None) -> type["LogicalType"]:
        """Load a registered logical type class.

        Args:
            key (Any): Logical type class, instance, identifier, or alias.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            type[LogicalType]: Registered logical type class.

        Raises:
            KeyError: If `key` is not registered.
            TypeError: If `key` cannot identify a logical type.
        """
        from .registry import get_logical_type

        return get_logical_type(key, resolver=resolver)

    @classmethod
    def identifier_of(cls, dtype: Any, *, resolver: ModuleResolver | None = None) -> str:
        """Return a logical type's canonical identifier.

        Args:
            dtype (Any): Logical type class, instance, identifier, alias,
                shorthand, or descriptor mapping.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            str: Canonical lower-hyphen identifier.

        Raises:
            TypeError: If `dtype` has an unsupported shape.
            ValueError: If the descriptor is unknown or under-specified.
        """
        return cls.from_any(dtype, resolver=resolver).to_str() if not isinstance(dtype, type) else cls.load(dtype, resolver=resolver).identifier

    def clone(self) -> "LogicalType":
        """Return an independent value with the same parameters.

        Args:
            None.

        Returns:
            LogicalType: Independent logical type value.
        """
        return type(self)()
