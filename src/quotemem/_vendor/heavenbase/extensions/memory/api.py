"""Workspace API for the optional memory extension."""

from __future__ import annotations

__all__ = ["MemoryAPI"]

from heavenbase.utils import Any, List, Optional, raise_mismatch

from heavenbase.modules import ModuleResolver, module_resolver
from .versions import MemoryVersions


def _memory_profiles(resolver: ModuleResolver) -> tuple[str, ...]:
    """Return MCP profiles owned by the memory extension records."""
    identifiers: list[str] = []
    for row in resolver.inspect("mcp_profile"):
        record = row.get("record")
        meta = record.get("meta") if isinstance(record, dict) else None
        raw = meta.get("tags", {}) if isinstance(meta, dict) else {}
        if isinstance(raw, dict) and raw.get("memory") is True:
            identifiers.append(str(row["identifier"]))
    return tuple(identifiers)


class MemoryAPI:
    """Memory extension operations bound to one workspace."""

    def __init__(
        self,
        workspace: Any,
        *,
        resolver: ModuleResolver | None = None,
    ) -> None:
        """Bind memory operations to one workspace.

        Args:
            workspace (Any): Workspace with the memory extension enabled.
            resolver (ModuleResolver | None): Context-owned module resolver.
                When omitted, the workspace's resolver is selected.

        Returns:
            None: This initializer stores the workspace.
        """
        self.workspace = workspace
        self._resolver = module_resolver(resolver if resolver is not None else workspace.module_resolver)

    def versions(self, project_id: str = "default") -> MemoryVersions:
        """Return the version store for one project namespace.

        Args:
            project_id (str): Project namespace. Empty values resolve to `default`.

        Returns:
            MemoryVersions: Workspace-bound append-only memory version store.
        """
        from .extension import ensure_memory_entity

        return MemoryVersions(self.workspace, ensure_memory_entity(self.workspace), project_id)

    def to_mcp(
        self,
        *,
        name: Optional[str] = None,
        profile: str = "memory",
        description: Optional[str] = None,
        tools: Optional[List[str]] = None,
    ) -> Any:
        """Return a Toolkit for a memory-owned MCP profile.

        Args:
            name (Optional[str]): Optional Toolkit name. Defaults to a workspace-derived name.
            profile (str): Memory profile id. Use `memory` for flat notes or
                `memstate` for project keypath state. Defaults to `memory`.
            description (Optional[str]): Optional Toolkit description. Defaults to a profile summary.
            tools (Optional[List[str]]): Optional explicit tool allowlist. Leave unset for the profile defaults.

        Returns:
            Any: Toolkit bound to the owning workspace and selected memory profile.

        Raises:
            ValueError: If ``profile`` is not ``"memory"`` or ``"memstate"``.
        """
        profile_id = str(profile or "memory")
        profiles = _memory_profiles(self._resolver)
        if profile_id not in profiles:
            raise_mismatch(profiles, profile_id, name="memory MCP profile")
        return self.workspace.to_mcp(name=name, description=description, profile=profile_id, tools=tools)

    def to_mcp_json(
        self,
        *,
        name: Optional[str] = None,
        profile: str = "memory",
        description: Optional[str] = None,
        tools: Optional[List[str]] = None,
        transport: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> str:
        """Return MCP client config JSON for a memory-owned profile.

        Args:
            name (Optional[str]): Optional Toolkit/server name. Defaults to a workspace-derived name.
            profile (str): Memory profile id. Use `memory` or `memstate`.
            description (Optional[str]): Optional Toolkit description. Defaults to a profile summary.
            tools (Optional[List[str]]): Optional explicit tool allowlist. Leave unset for profile defaults.
            transport (Optional[str]): MCP transport. Defaults to configured MCP transport.
            host (Optional[str]): HTTP host for server transports.
            port (Optional[int]): HTTP port for server transports.

        Returns:
            str: JSON string consumable by an MCP client.

        Raises:
            ValueError: If ``profile`` is not ``"memory"`` or ``"memstate"``.
        """
        profile_id = str(profile or "memory")
        profiles = _memory_profiles(self._resolver)
        if profile_id not in profiles:
            raise_mismatch(profiles, profile_id, name="memory MCP profile")
        return self.workspace.to_mcp_json(
            name=name,
            description=description,
            profile=profile_id,
            tools=tools,
            transport=transport,
            host=host,
            port=port,
        )
