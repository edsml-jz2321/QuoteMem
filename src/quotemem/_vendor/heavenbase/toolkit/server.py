"""Toolkit MCP serving helpers."""

from __future__ import annotations

__all__ = ["mcp_host", "mcp_port", "mcp_transport", "serve_toolkit"]

from heavenbase.utils import Any, Optional


def _config(config: Any | None) -> Any:
    from heavenbase.context.authority import omitted_config

    return omitted_config() if config is None else config


def mcp_transport(transport: Optional[str] = None, *, config: Any | None = None) -> str:
    """Return effective MCP transport.

    Args:
        transport (Optional[str]): Explicit transport override. Defaults to config ``heavenbase.mcp.transport`` or ``"http"``.
        config (Any | None): Config authority. When omitted, the process default is selected.

    Returns:
        str: Selected MCP transport name.
    """
    return transport if transport is not None else str(_config(config).get("heavenbase.mcp.transport", default="http"))


def mcp_host(host: Optional[str] = None, *, config: Any | None = None) -> str:
    """Return effective MCP host.

    Args:
        host (Optional[str]): Explicit host override. Defaults to config ``heavenbase.mcp.host`` or ``"127.0.0.1"``.
        config (Any | None): Config authority. When omitted, the process default is selected.

    Returns:
        str: Selected MCP host.
    """
    return host if host is not None else str(_config(config).get("heavenbase.mcp.host", default="127.0.0.1"))


def mcp_port(port: Optional[int] = None, *, config: Any | None = None) -> int:
    """Return effective MCP port.

    Args:
        port (Optional[int]): Explicit port override. Defaults to config ``heavenbase.mcp.port`` or ``7001``.
        config (Any | None): Config authority. When omitted, the process default is selected.

    Returns:
        int: Selected MCP port.
    """
    return int(port if port is not None else _config(config).get("heavenbase.mcp.port", default=7001))


def serve_toolkit(
    toolkit: Any,
    *,
    transport: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    config: Any | None = None,
    **server_kwargs: Any,
) -> Any:
    """Serve a Toolkit through FastMCP.

    Args:
        toolkit (Any): Toolkit whose ``to_fastmcp`` server should run.
        transport (Optional[str]): MCP transport override. Defaults to config ``heavenbase.mcp.transport``.
        host (Optional[str]): HTTP host override for network transports.
        port (Optional[int]): HTTP port override for network transports.
        config (Any | None): Config authority. When omitted, the process default is selected.
        **server_kwargs (Any): Additional FastMCP ``run`` options.

    Returns:
        Any: FastMCP run result after the blocking server lifecycle completes.
    """
    selected = mcp_transport(transport, config=config)
    server = toolkit.to_fastmcp()
    kwargs = dict(server_kwargs)
    if selected == "stdio":
        kwargs.setdefault("show_banner", False)
    if selected in {"http", "streamable-http", "sse"}:
        kwargs.setdefault("host", mcp_host(host, config=config))
        kwargs.setdefault("port", mcp_port(port, config=config))
    return server.run(transport=selected, **kwargs)
