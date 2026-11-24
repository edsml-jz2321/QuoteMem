"""Serializable option rows for HeavenBase capability inspection."""

from __future__ import annotations

from dataclasses import InitVar
from types import MappingProxyType

from heavenbase.utils import Any, Dict, Mapping, Optional, dataclass

from ..execution.handlers.base import HandlerKey


@dataclass(frozen=True)
class LogicalTypeOption:
    """One selectable logical type exposed by capability discovery.

    Args:
        identifier (str): Canonical logical type identifier.
        dtype (Any): Registered logical type class.
        aliases (tuple[str, ...]): Accepted lookup aliases for the type.
    """

    identifier: str
    dtype: Any
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly option row.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serializable option metadata containing identifier,
            aliases, and class name.
        """
        return {
            "identifier": self.identifier,
            "aliases": list(self.aliases),
            "class": getattr(self.dtype, "__name__", str(self.dtype)),
        }


@dataclass(frozen=True)
class StrategyOption:
    """One selectable storage strategy exposed by capability discovery.

    Args:
        identifier (str): Canonical storage strategy identifier.
        strategy (Any): Registered storage strategy class.
        aliases (tuple[str, ...]): Accepted lookup aliases for the strategy.
    """

    identifier: str
    strategy: Any
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly option row.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serializable option metadata containing identifier,
            aliases, and class name.
        """
        return {
            "identifier": self.identifier,
            "aliases": list(self.aliases),
            "class": getattr(self.strategy, "__name__", str(self.strategy)),
        }


@dataclass(frozen=True)
class BackendOption:
    """One selectable backend type or workspace backend instance.

    Args:
        type (str): Canonical backend type identifier.
        aliases (tuple[str, ...]): Accepted lookup aliases for the backend type.
        name (str | None): Workspace backend instance name when the option comes from `ws.capabilities`.
        backend (Any): Live backend instance when the option comes from `ws.capabilities`.
    """

    type: str
    aliases: tuple[str, ...] = ()
    name: Optional[str] = None
    backend: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly option row.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serializable backend metadata with type, aliases, optional workspace name,
            and optional live backend class name.
        """
        row = {"type": self.type, "aliases": list(self.aliases)}
        if self.name is not None:
            row["name"] = self.name
        if self.backend is not None:
            row["class"] = type(self.backend).__name__
        return row


@dataclass(frozen=True)
class CapabilityOption:
    """One logical type, operation, backend, and strategy support row.

    Args:
        dtype (str): Logical type identifier supported by the capability.
        op (str): Normalized operation token supported by the handler.
        backend_identifier (str): Backend type identifier supported by the handler.
        strategy (str): Storage strategy identifier supported by the handler.
        backend_name (str | None): Workspace backend instance name when the row comes from `ws.capabilities`.
        aliases (Mapping[str, tuple[str, ...]] | None): Optional precomputed alias metadata keyed by `dtype`, `op`,
            `backend`, and `strategy`.
        resolver: Context-owned module resolver used to capture omitted alias
            metadata. Callers may instead provide an immutable alias snapshot.
    """

    dtype: str
    op: str
    backend_identifier: str
    strategy: str
    backend_name: Optional[str] = None
    aliases: Optional[Mapping[str, tuple[str, ...]]] = None
    resolver: InitVar[Any] = None

    def __post_init__(self, resolver: Any) -> None:
        """Copy nested metadata into read-only mappings.

        Args:
            None.

        Returns:
            None: This initializer stores immutable mapping snapshots.
        """
        aliases = self.aliases
        if aliases is None:
            if resolver is None:
                raise TypeError("CapabilityOption requires an alias snapshot or an explicit module resolver")
            key = HandlerKey(self.dtype, self.op, self.backend_identifier, self.strategy, resolver=resolver)
            aliases = key.aliases(resolver=resolver)
        object.__setattr__(self, "aliases", MappingProxyType({name: tuple(values) for name, values in aliases.items()}))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly capability row.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serializable capability metadata with normalized keys
            plus alias metadata.
        """
        row = {"dtype": self.dtype, "op": self.op, "backend_identifier": self.backend_identifier, "strategy": self.strategy}
        if self.backend_name is not None:
            row["backend_name"] = self.backend_name
        row["aliases"] = {name: list(values) for name, values in self.aliases.items()}
        return row
