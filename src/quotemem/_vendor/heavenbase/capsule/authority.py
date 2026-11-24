"""Resolved configuration and module authority for Capsule operations."""

from __future__ import annotations

from dataclasses import dataclass

from heavenbase.context.authority import resolve_runtime_authority
from heavenbase.modules import ModuleResolver
from heavenbase.utils.config import ConfigManager

__all__: list[str] = []


@dataclass(frozen=True, eq=False, slots=True)
class CapsuleAuthority:
    """Bind one Capsule operation to exact module and config authorities."""

    resolver: ModuleResolver
    config: ConfigManager

    @classmethod
    def resolve(
        cls,
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> "CapsuleAuthority":
        """Resolve omitted standalone authorities once at the Capsule boundary.

        Args:
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.
            config (ConfigManager | None): Config authority paired with
                `resolver`. The standalone process config is selected when omitted.

        Returns:
            CapsuleAuthority: Authority bound to one coherent resolver/config pair.
        """
        authority = resolve_runtime_authority(resolver=resolver, config=config, label="Capsule")
        return cls(authority.resolver, authority.config)

    def replace(
        self,
        *,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> "CapsuleAuthority":
        """Return an authority with explicit overrides and paired omissions.

        Args:
            resolver (ModuleResolver | None): Replacement module resolver.
                The current resolver is retained when omitted.
            config (ConfigManager | None): Replacement config authority.
                The current config is retained when omitted.

        Returns:
            CapsuleAuthority: Updated authority, or ``self`` when both arguments
            are omitted.
        """
        if resolver is None and config is None:
            return self
        return type(self).resolve(
            resolver=self.resolver if resolver is None else resolver,
            config=self.config if config is None else config,
        )

    @property
    def identity(self) -> tuple[int, int]:
        """Return the process-local resolver and config identity pair."""
        return id(self.resolver), id(self.config)
