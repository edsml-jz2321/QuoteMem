"""FastMCP adapters for runtime callables and Capsule-backed Toolkits."""

from __future__ import annotations

__all__ = ["callables_from_fastmcp", "toolkit_from_fastmcp", "toolkit_to_fastmcp", "toolkit_to_mcp_json"]

import asyncio
import inspect
from time import perf_counter
from typing import get_type_hints

from heavenbase.utils import Any, Dict, Mapping, deepcopy, dumps_json

from .cache import MCPToolCache, MCPToolCacheConfig
from .errors import ToolExportError


def _mcp_output_schema(schema: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    native = deepcopy(dict(schema or {}))
    if not native:
        return None
    if native.get("type") == "object":
        return native
    return {
        "type": "object",
        "properties": {"result": native},
        "required": ["result"],
        "x-fastmcp-wrap-result": True,
    }


def _mcp_annotations(tool: Any) -> Any:
    """Return validated protocol annotations for one HeavenBase Tool."""
    import mcp
    from mcp.types import ToolAnnotations

    annotations = dict(getattr(tool, "annotations", None) or {})
    if "readOnlyHint" not in annotations and isinstance(tool.metadata.get("read_only"), bool):
        annotations["readOnlyHint"] = tool.metadata["read_only"]
    return ToolAnnotations.model_validate(annotations) if annotations else None


def _wrapper_for(toolkit: Any, tool: Any, cache: MCPToolCache, output_schema: Mapping[str, Any] | None) -> Any:
    fn = tool.capsule.to_func()
    signature = inspect.signature(fn)
    wrapped = bool(output_schema and output_schema.get("x-fastmcp-wrap-result"))

    async def wrapper(**kwargs: Any) -> Any:
        import fastmcp
        from fastmcp.tools import ToolResult
        from pydantic_core import PydanticSerializationError

        start = perf_counter()
        error = False
        cached = False
        native: Any = None
        try:
            cached, native = cache.get(tool.name, kwargs)
            if not cached:
                native = await tool.arun(**kwargs)
                cache.set(tool.name, kwargs, native)
            content = await tool.ato_str(native)
            if output_schema is not None:
                return ToolResult(
                    content=content,
                    structured_content={"result": native} if wrapped else native,
                    meta={"fastmcp": {"wrap_result": True}} if wrapped else None,
                )
            try:
                runtime_wrapped = not isinstance(native, dict)
                return ToolResult(
                    content=content,
                    structured_content={"result": native} if runtime_wrapped else native,
                    meta={"fastmcp": {"wrap_result": True}} if runtime_wrapped else None,
                )
            except PydanticSerializationError:
                return ToolResult(content=content)
        except Exception:
            error = True
            raise
        finally:
            toolkit._emit_call_hooks(
                tool.name,
                kwargs,
                native,
                perf_counter() - start,
                error,
                cached=cached,
            )

    wrapper.__name__ = tool.name
    wrapper.__doc__ = tool.description
    wrapper.__signature__ = signature.replace(return_annotation=Any)
    try:
        annotations = dict(get_type_hints(fn))
    except Exception:
        annotations = dict(getattr(fn, "__annotations__", {}) or {})
    for name in signature.parameters:
        annotations.setdefault(name, Any)
    annotations["return"] = Any
    wrapper.__annotations__ = annotations
    return wrapper


def _mcp_transforms(toolkit: Any, transforms: Any = None) -> list[Any]:
    configured = transforms is None
    selected = toolkit._config.get("heavenbase.mcp.transforms", default=[]) if configured else transforms
    if selected is None:
        return []
    if isinstance(selected, str):
        selected = [selected]
    elif isinstance(selected, Mapping):
        raise TypeError("heavenbase.mcp.transforms must be an iterable of transform objects or Registry identifiers")
    try:
        items = list(selected)
    except TypeError as exc:
        raise TypeError("heavenbase.mcp.transforms must be an iterable of transform objects or Registry identifiers") from exc
    if not configured:
        return items

    out: list[Any] = []
    for item in items:
        if isinstance(item, str):
            name = item
            args: Mapping[str, Any] = {}
        elif isinstance(item, Mapping):
            name = str(item.get("name") or item.get("identifier") or "").strip()
            raw_args = item.get("args", {})
            if not name:
                raise ValueError("configured MCP transforms require a non-empty name")
            if not isinstance(raw_args, Mapping):
                raise TypeError(f"configured MCP transform {name!r} args must be a mapping")
            args = raw_args
        else:
            raise TypeError("configured MCP transforms must be Registry identifiers or mappings with name and args")
        builder = toolkit._resolver.resolve("mcp_transform", name)
        if not callable(builder):
            raise TypeError(f"MCP transform builder {name!r} must resolve to a callable")
        out.append(builder(**dict(args)))
    return out


def _mcp_lifespan(toolkit: Any, cache: MCPToolCache) -> Any:
    import fastmcp
    from fastmcp.server.lifespan import lifespan

    @lifespan
    async def toolkit_lifespan(server: Any) -> Any:
        cache.open()
        try:
            yield {"heavenbase": {"toolkit": toolkit, "cache": cache}}
        finally:
            cache.close()

    return toolkit_lifespan


def toolkit_to_fastmcp(toolkit: Any, name: str | None = None, *, transforms: Any = None) -> Any:
    """Create a FastMCP server for a Toolkit.

    Args:
        toolkit (Any): Toolkit whose tools should be exported.
        name (str | None): Optional MCP server name. Defaults to the Toolkit name.
        transforms (Any): Optional live FastMCP transforms. When omitted,
            Registry `mcp_transform` identifiers and argument mappings are read
            from `heavenbase.mcp.transforms`.

    Returns:
        Any: FastMCP server with one MCP tool per Toolkit tool.

    Raises:
        TypeError: If configured transforms do not form an iterable.
        ToolExportError: If a tool manifest marks MCP export as unsupported.
    """
    from fastmcp import FastMCP
    from fastmcp.tools import FunctionTool

    cache = MCPToolCache(MCPToolCacheConfig.load(toolkit._config))
    server = FastMCP(
        name or toolkit.name,
        transforms=_mcp_transforms(toolkit, transforms),
        lifespan=_mcp_lifespan(toolkit, cache),
    )
    for tool in toolkit.tools.values():
        cap = tool.capsule.manifest.capabilities.get("mcp_tool", {})
        if not cap.get("ok", True):
            raise ToolExportError(f"Tool {tool.name!r} cannot be exported to MCP: {cap.get('reasons')}")
        output_schema = _mcp_output_schema(tool.output_schema)
        exported = FunctionTool.from_function(
            _wrapper_for(toolkit, tool, cache, output_schema),
            name=tool.name,
            description=tool.description,
            annotations=_mcp_annotations(tool),
            output_schema=output_schema,
        )
        server.add_tool(
            exported.model_copy(
                update={"parameters": deepcopy(tool.input_schema)},
            )
        )
    return server


def _run_client_call(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return coro if loop.is_running() else loop.run_until_complete(coro)


def _tool_schema(tool: Any) -> Dict[str, Any]:
    return dict(getattr(tool, "input_schema", None) or {})


def _tool_output_schema(tool: Any) -> Dict[str, Any]:
    schema = dict(getattr(tool, "output_schema", None) or {})
    if schema.get("x-fastmcp-wrap-result"):
        return dict(schema.get("properties", {}).get("result", {}) or {})
    return schema


def _tool_desc(tool: Any) -> str:
    return str(getattr(tool, "description", None) or "")


def _tool_annotations(tool: Any) -> Dict[str, Any]:
    annotations = getattr(tool, "annotations", None)
    if annotations is None:
        return {}
    dump = getattr(annotations, "model_dump", None)
    if callable(dump):
        return dict(dump(by_alias=True, exclude_none=True))
    return dict(annotations)


async def _list_remote_tools(server: Any) -> Any:
    from fastmcp import Client

    async with Client(server) as client:
        return await client.list_tools()


async def _call_remote_tool(server: Any, tool_name: str, kwargs: Dict[str, Any]) -> Any:
    from fastmcp import Client

    async with Client(server) as client:
        result = await client.call_tool(tool_name, kwargs)
    data = getattr(result, "data", None)
    if data is not None:
        return data
    structured = getattr(result, "structured_content", None)
    fastmcp_meta = (getattr(result, "meta", None) or {}).get("fastmcp", {})
    if isinstance(structured, Mapping) and isinstance(fastmcp_meta, Mapping) and fastmcp_meta.get("wrap_result"):
        return structured.get("result")
    if structured is not None:
        return structured
    content = list(getattr(result, "content", None) or [])
    if len(content) == 1 and hasattr(content[0], "text"):
        return content[0].text
    if content:
        return [getattr(item, "text", item) for item in content]
    return result


def callables_from_fastmcp(server: Any) -> list[Any]:
    """Import an MCP server as runtime-only schema-bearing callables.

    This is the lightweight counterpart to :func:`toolkit_from_fastmcp` for
    consumers that need ordinary callables but no Capsule persistence. FastMCP
    owns transport and protocol negotiation; each returned callable carries
    the remote description, input schema, and official ToolAnnotations.

    Args:
        server (Any): FastMCP server, URL, transport, or other value accepted
            by ``fastmcp.Client``.

    Returns:
        list[Any]: Runtime callables in the remote server's tool order.
    """
    callables: list[Any] = []
    for remote_tool in _run_client_call(_list_remote_tools(server)):
        tool_name = str(remote_tool.name)

        def make_call(selected_name: str) -> Any:
            def call_remote(**kwargs: Any) -> Any:
                return _run_client_call(_call_remote_tool(server, selected_name, kwargs))

            return call_remote

        call_remote = make_call(tool_name)
        call_remote.__name__ = tool_name
        call_remote.__doc__ = _tool_desc(remote_tool)
        call_remote.__tool_schema__ = _tool_schema(remote_tool)
        call_remote.annotations = _tool_annotations(remote_tool)
        callables.append(call_remote)
    return callables


def toolkit_from_fastmcp(
    server: Any,
    name: str | None = None,
    namespace: str = "mcp-client",
    *,
    resolver: Any = None,
    config: Any = None,
) -> Any:
    """Import a FastMCP server as a local Toolkit client.

    Args:
        server (Any): FastMCP server, URL, transport, or other value accepted by ``fastmcp.Client``.
        name (str | None): Local Toolkit name. Defaults to the remote server name or ``"mcp-client"``.
        namespace (str): Namespace for runtime proxy Capsules. Defaults to ``"mcp-client"``.
        resolver (Any): Context-owned module resolver.
        config (Any): Config authority paired with ``resolver``.

    Returns:
        Any: Runtime-only Toolkit whose tools proxy calls to the remote MCP server.

    Warnings:
        The returned Toolkit closes over the live remote server or transport and is intentionally marked as not registry-persistent.
        Its raw ``run`` value is the FastMCP client payload, preferring ``data`` when present, then ``structured_content``,
        then text content. ``run_to_str`` serializes that proxy payload through the local Tool serializer.
    """
    from heavenbase.capsule import Capsule
    from .tool import Tool
    from .toolkit import Toolkit

    remote_tools = _run_client_call(_list_remote_tools(server))
    toolkit = Toolkit(
        name or getattr(server, "name", None) or "mcp-client",
        namespace=namespace,
        resolver=resolver,
        config=config,
    )
    for remote_tool in remote_tools:
        tool_name = str(remote_tool.name)

        def _make_call_remote(selected_tool_name: str) -> Any:
            def call_remote(**kwargs: Any) -> Any:
                return _run_client_call(_call_remote_tool(server, selected_tool_name, kwargs))

            return call_remote

        call_remote = _make_call_remote(tool_name)
        call_remote.__name__ = tool_name
        call_remote.__doc__ = _tool_desc(remote_tool)
        capsule = Capsule(
            call_remote,
            name=tool_name,
            namespace=namespace,
            version="1",
            description=_tool_desc(remote_tool),
            layers=("source",),
            resolver=resolver,
            config=config,
        )
        capsule.manifest.capabilities["registry_persistence"] = {
            "ok": False,
            "reasons": ["FastMCP client proxy closes over a live remote server"],
        }
        toolkit.add(
            Tool(
                capsule,
                name=tool_name,
                description=_tool_desc(remote_tool),
                input_schema=_tool_schema(remote_tool),
                output_schema=_tool_output_schema(remote_tool),
                annotations=_tool_annotations(remote_tool),
            )
        )
    return toolkit


def toolkit_to_mcp_json(
    toolkit: Any,
    *,
    transport: str = "http",
    host: str | None = None,
    port: int | None = None,
    workspace: str | None = None,
) -> str:
    """Return a small MCP client config for this Toolkit.

    Args:
        toolkit (Any): Toolkit to expose.
        transport (str): MCP transport. ``"http"`` and ``"streamable-http"`` use ``/mcp``; ``"sse"`` uses ``/sse``; ``"stdio"``
            emits an ``hb mcp stdio`` command. Defaults to ``"http"``.
        host (str | None): HTTP host. Defaults to ``None``.
        port (int | None): HTTP port. Defaults to ``None``.
        workspace (str | None): Workspace id added to stdio serving args.
            Defaults to the workspace bound by `Toolkit.load`.

    Returns:
        str: JSON client config.
    """
    host = host or "127.0.0.1"
    port = int(port or 7001)
    if transport == "stdio":
        ref = f"{toolkit.namespace}/{toolkit.name}:{toolkit.version}"
        args = ["mcp", "stdio", ref]
        selected_workspace = workspace or getattr(getattr(toolkit, "_workspace", None), "id", None)
        if selected_workspace:
            args.extend(["--workspace", str(selected_workspace)])
        server = {"transport": "stdio", "command": "hb", "args": args}
    elif transport == "sse":
        server = {"transport": "sse", "url": f"http://{host}:{port}/sse"}
    else:
        server = {"transport": transport, "url": f"http://{host}:{port}/mcp"}
    return dumps_json({"mcpServers": {toolkit.name: server}}, indent=4, sort_keys=True)
