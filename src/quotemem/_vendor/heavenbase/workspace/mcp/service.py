"""Workspace MCP exposure service."""

from __future__ import annotations

from heavenbase.utils import Any, List, Optional

from ...backends.base import Backend
from .support import backend_toolkit, mcp_host, mcp_port, mcp_transport, sync_mcp_registry_metadata, workspace_toolkit


class _MCPService:
    """Build and serve MCP Toolkits for one workspace facade."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def sync_registry_metadata(self, *, load_workspace_rows: bool = True) -> Any:
        """Publish MCP registry metadata rows for this workspace."""
        return sync_mcp_registry_metadata(
            self.workspace,
            load_workspace_rows=load_workspace_rows,
            resolver=self.workspace.module_resolver,
        )

    def to_mcp(
        self,
        *,
        name: Optional[str] = None,
        backend: Optional[Any] = None,
        description: Optional[str] = None,
        profile: Optional[str] = None,
        tools: Optional[List[str]] = None,
    ) -> Any:
        """Return a HeavenBase-native Toolkit for this workspace or one of its backends."""
        if backend is not None:
            selected = self.workspace.backends.get(backend) if isinstance(backend, str) else backend
            if not isinstance(selected, Backend):
                raise TypeError("backend must be a backend name or Backend instance")
            from ..setup.lifecycle import _workspace_config_scope

            return backend_toolkit(
                selected,
                name=name,
                description=description,
                profile=profile,
                tools=tools,
                resolver=self.workspace.module_resolver,
                scope=_workspace_config_scope(self.workspace.context, self.workspace.id),
            )
        return workspace_toolkit(
            self.workspace,
            name=name,
            description=description,
            profile=profile,
            tools=tools,
            resolver=self.workspace.module_resolver,
        )

    def to_mcp_json(
        self,
        *,
        name: Optional[str] = None,
        backend: Optional[Any] = None,
        description: Optional[str] = None,
        profile: Optional[str] = None,
        tools: Optional[List[str]] = None,
        transport: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> str:
        """Return MCP client config JSON for this workspace or one of its backends."""
        config = self.workspace.context.config
        return self.to_mcp(name=name, backend=backend, description=description, profile=profile, tools=tools).to_mcp_json(
            transport=mcp_transport(transport, config=config, resolver=self.workspace.module_resolver),
            host=mcp_host(host, config=config, resolver=self.workspace.module_resolver),
            port=mcp_port(port, config=config, resolver=self.workspace.module_resolver),
        )

    def to_skill(
        self,
        *,
        profile: str,
        name: Optional[str] = None,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> Any:
        """Return an installable agent skill for one workspace MCP profile."""
        from .skill import WorkspaceSkill

        return WorkspaceSkill(
            self.workspace,
            profile=profile,
            name=name,
            url=url,
            host=host,
            port=port,
        )
