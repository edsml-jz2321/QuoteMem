"""Registry adapters from workspace MCP services to the Toolkit extension."""

from __future__ import annotations

from heavenbase.utils import Any

from ...extensions.contributions import resolve_contribution
from ...modules import ModuleResolver, module_resolver

__all__ = [
    "backend_toolkit",
    "mcp_host",
    "mcp_port",
    "mcp_transport",
    "sync_mcp_registry_metadata",
    "workspace_mcp_profile",
    "workspace_toolkit",
]


def _toolkit_function(
    identifier: str,
    *,
    resolver: ModuleResolver | None = None,
    workspace: Any | None = None,
) -> Any:
    """Resolve one installed Toolkit contribution under one authority."""
    return resolve_contribution(
        "toolkit_function",
        identifier,
        resolver=resolver,
        workspace=workspace,
        callable_only=True,
    )


def workspace_toolkit(workspace: Any, *args: Any, **kwargs: Any) -> Any:
    """Return a workspace-bound Toolkit from the installed artifact.

    Args:
        workspace: Live workspace bound to the Toolkit.
        *args (Any): Positional arguments forwarded to the Toolkit factory.
        **kwargs (Any): Keyword arguments forwarded to the Toolkit factory.

    Returns:
        Any: Workspace-bound Toolkit instance.
    """
    resolver = kwargs.get("resolver")
    factory = _toolkit_function("workspace_toolkit", resolver=resolver, workspace=workspace)
    return factory(workspace, *args, **kwargs)


def backend_toolkit(backend: Any, *args: Any, resolver: ModuleResolver | None = None, **kwargs: Any) -> Any:
    """Return a backend-bound Toolkit from the installed artifact.

    Args:
        backend (Any): Backend instance or backend name to expose through MCP.
        *args (Any): Positional arguments forwarded to the Toolkit factory.
        resolver (ModuleResolver | None): Context-owned module resolver.
            Defaults to ``None``.
        **kwargs (Any): Keyword arguments forwarded to the Toolkit factory.

    Returns:
        Any: Backend-bound Toolkit instance.
    """
    factory = _toolkit_function("backend_toolkit", resolver=resolver)
    return factory(backend, *args, resolver=resolver, **kwargs)


def sync_mcp_registry_metadata(
    workspace: Any,
    *,
    load_workspace_rows: bool = True,
    resolver: ModuleResolver | None = None,
) -> Any:
    """Publish MCP registry metadata rows for one workspace.

    Args:
        workspace (Any): Live workspace receiving MCP registry metadata rows.
        load_workspace_rows (bool): Whether to include workspace-specific profile,
            serializer, and toolkit-family rows before publishing.
        resolver: Context-owned module resolver. A resolver bound to
            ``workspace`` is preferred before the standalone default.

    Returns:
        Any: Result returned by the lower registry metadata sync helper.
    """
    bound = getattr(workspace, "module_resolver", None)
    chosen = module_resolver(resolver if resolver is not None else bound)
    sync = _toolkit_function(
        "sync_workspace_mcp_registry_metadata",
        resolver=chosen,
        workspace=workspace,
    )
    return sync(
        workspace,
        load_workspace_rows=load_workspace_rows,
        resolver=chosen,
    )


def workspace_mcp_profile(workspace: Any, profile: str, *, resolver: ModuleResolver | None = None) -> Any:
    """Return one MCP profile resolved against the workspace's own registry rows.

    Args:
        workspace (Any): Live workspace whose profile and serializer rows apply.
        profile (str): MCP profile name to resolve.
        resolver (ModuleResolver | None): Context-owned module resolver. A
            resolver bound to ``workspace`` is preferred before the standalone default.

    Returns:
        Any: Flattened profile spec from the installed Toolkit artifact.
    """
    bound = getattr(workspace, "module_resolver", None)
    chosen = module_resolver(resolver if resolver is not None else bound)
    resolve = _toolkit_function("resolve_workspace_mcp_profile", resolver=chosen, workspace=workspace)
    return resolve(workspace, profile, resolver=chosen)


def mcp_transport(value: Any = None, *, config: Any | None = None, resolver: ModuleResolver | None = None) -> str:
    """Return the resolved MCP transport.

    Args:
        value (Any): Optional explicit transport value. When omitted, the lower server
            helper resolves the configured default.
        config: Config authority used when ``value`` is omitted.

    Returns:
        str: Resolved MCP transport identifier.
    """
    return _toolkit_function("mcp_transport", resolver=resolver)(value, config=config)


def mcp_host(value: Any = None, *, config: Any | None = None, resolver: ModuleResolver | None = None) -> str:
    """Return the resolved MCP host.

    Args:
        value (Any): Optional explicit host value. When omitted, the lower server helper
            resolves the configured default.
        config: Config authority used when ``value`` is omitted.

    Returns:
        str: Resolved MCP host.
    """
    return _toolkit_function("mcp_host", resolver=resolver)(value, config=config)


def mcp_port(value: Any = None, *, config: Any | None = None, resolver: ModuleResolver | None = None) -> int:
    """Return the resolved MCP port.

    Args:
        value (Any): Optional explicit port value. When omitted, the lower server helper
            resolves the configured default.
        config: Config authority used when ``value`` is omitted.

    Returns:
        int: Resolved MCP port.
    """
    return _toolkit_function("mcp_port", resolver=resolver)(value, config=config)
