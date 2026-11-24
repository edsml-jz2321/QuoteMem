"""Capsule-backed Toolkit collection and registry runtime."""

from __future__ import annotations

__all__ = ["Toolkit"]

from collections import OrderedDict
from copy import deepcopy
from time import perf_counter, time

from heavenbase.entity import Entity, field
from heavenbase.types import Float, Json, LongText, ShortText
from heavenbase.utils import Any, Dict, Iterator, List, Mapping, Optional, hash_id
from heavenbase.utils.config import ConfigManager

from heavenbase.capsule import Capsule
from heavenbase.capsule.authority import CapsuleAuthority
from heavenbase.modules import ModuleResolver
from .tool import Tool


def _default_namespace(config: ConfigManager) -> str:
    fallback = config.get("heavenbase.capsule.namespace", default="default")
    return str(config.get("heavenbase.toolkit.namespace", default=fallback) or "default")


class Toolkit(Entity):
    """Ordered collection of Capsule-backed tools.

    Toolkits provide local execution, registry persistence, and MCP/agent schema exports for a named set of tools.
    The class is also the system entity registered as ``sys-toolkit``; registry storage uses :meth:`to_row` so runtime
    methods and manifest serialization stay user-facing.

    Warnings:
        Toolkit loading restores each referenced Capsule. Treat persisted Toolkits as executable-code bundles, not as
        inert metadata.
    """

    identifier = "sys-toolkit"

    name = field(ShortText).desc("Toolkit name")
    namespace = field(ShortText).default("default").desc("Toolkit namespace")
    version = field(ShortText).default("1").desc("Explicit toolkit version")
    status = field(ShortText).default("active").desc("Toolkit status")
    description = field(LongText).default("").desc("Toolkit description")
    capsule_refs = field(Json).default([]).desc("Ordered Capsule references")
    serving = field(Json).default({}).desc("Serving metadata")
    capabilities = field(Json).default({}).desc("Toolkit capability summary")
    manifest = field(Json).default({}).desc("Full toolkit manifest")
    last_reason = field(LongText).default("").desc("Last registry write reason")
    created_at = field(Float).default(0.0).desc("Creation timestamp")
    updated_at = field(Float).default(0.0).desc("Last update timestamp")

    def __init__(
        self,
        name: str,
        tools: Optional[Any] = None,
        *,
        description: str = "",
        namespace: Optional[str] = None,
        version: str = "1",
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> None:
        """Create a runtime Toolkit under explicit module and config authorities.

        Args:
            name: Toolkit identifier.
            tools: Optional tools added in declaration order.
            description: Human-readable Toolkit description.
            namespace: Optional Toolkit namespace. The standalone configured
                default is selected when omitted.
            version: Explicit Toolkit version.
            resolver: Context-owned module resolver. The process default is
                selected when omitted.
            config: Config authority forwarded to registered serializers.

        Returns:
            None: This initializer stores the Toolkit definition.
        """
        self.name = name
        self.description = description
        self._authority = CapsuleAuthority.resolve(resolver=resolver, config=config)
        self.namespace = namespace or _default_namespace(self._authority.config)
        self.version = version
        self._resolver = self._authority.resolver
        self._config = self._authority.config
        self._workspace: Any = None
        self.tools: "OrderedDict[str, Tool]" = OrderedDict()
        self._call_hooks: List[Any] = []
        self.add_many(tools or [])

    def _bind_authority(self, resolver: Any, config: Any) -> None:
        """Bind this Toolkit and its Capsules to one workspace authority."""
        self._authority = self._authority.replace(resolver=resolver, config=config)
        self._resolver = self._authority.resolver
        self._config = self._authority.config
        for tool in self.tools.values():
            tool._bind_authority(self._resolver, self._config)

    @property
    def id(self) -> str:
        """Return the stable Toolkit id."""
        return hash_id("toolkit", self.namespace, self.name, self.version)

    def __len__(self) -> int:
        """Return the number of tools in this Toolkit."""
        return len(self.tools)

    def __iter__(self) -> Iterator[str]:
        """Iterate over tool names in execution order."""
        return iter(self.tools)

    def __contains__(self, name: str) -> bool:
        """Return whether a tool name exists in this Toolkit."""
        return name in self.tools

    def __getitem__(self, name: str) -> Tool:
        """Return one tool by name."""
        return self.tools[name]

    def get(self, name: str, default: Optional[Tool] = None) -> Optional[Tool]:
        """Return one tool by name, or a default value.

        Args:
            name (str): Tool name.
            default (Tool | None): Value returned when the tool is missing. Defaults to ``None``.

        Returns:
            Tool | None: Matching tool or ``default``.
        """
        return self.tools.get(name, default)

    def add(
        self,
        item: Any,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
        version: str = "1",
        layers: Optional[List[str]] = None,
        layer_options: Optional[Mapping[str, Any]] = None,
        include_cloudpickle: bool = False,
        prefer_import: bool = True,
        serializer: Any = None,
        annotations: Optional[Mapping[str, Any]] = None,
    ) -> "Toolkit":
        """Add a callable, Capsule, or Tool to this Toolkit.

        Args:
            item (Any): Callable, Capsule, or Tool to add.
            name (Optional[str]): Tool alias. Defaults to the callable/Capsule name.
            description (Optional[str]): Tool description. Defaults to the Capsule description.
            namespace (Optional[str]): Capsule namespace for callable capture. Defaults to the Toolkit namespace.
            version (str): Capsule version for callable capture. Defaults to ``"1"``.
            layers (Optional[List[str]]): Optional Capsule layer order for callable capture. Defaults to ``None``.
            layer_options (Optional[Mapping[str, Any]]): Optional per-layer capture options. Defaults to ``None``.
            include_cloudpickle (bool): Whether to store a trusted-local binary fallback. Defaults to ``False``.
            prefer_import (bool): Whether to include import-path restore when available. Defaults to ``True``.
            serializer (Any): Optional callable or Capsule used to serialize raw outputs for agent/MCP boundaries. ``None``
                keeps the default :func:`dumps_jstr` behavior at execution time.
            annotations (Optional[Mapping[str, Any]]): Optional MCP
                ``ToolAnnotations`` mapping retained on the Tool. Defaults to
                ``None``.

        Returns:
            Toolkit: This Toolkit for chaining.
        """
        if isinstance(item, Tool):
            tool = item
            tool._bind_authority(self._resolver, self._config)
            if annotations is not None:
                tool.annotations = dict(annotations)
            if serializer is not None:
                tool.set_serializer(
                    serializer,
                    layers=layers,
                    layer_options=layer_options,
                    include_cloudpickle=include_cloudpickle,
                    prefer_import=prefer_import,
                    config=self._config,
                )
        else:
            capsule = (
                item
                if isinstance(item, Capsule)
                else Capsule(
                    item,
                    name=name,
                    namespace=namespace or self.namespace,
                    version=version,
                    description=description,
                    layers=layers,
                    layer_options=dict(layer_options or {}),
                    include_cloudpickle=include_cloudpickle,
                    prefer_import=prefer_import,
                    resolver=self._resolver,
                    config=self._config,
                )
            )
            capsule._bind_authority(resolver=self._resolver, config=self._config)
            tool_name = name or capsule.manifest.name
            tool = Tool(
                capsule,
                name=tool_name,
                description=description or capsule.manifest.description,
                annotations=annotations,
            )
            tool._bind_authority(self._resolver, self._config)
            if serializer is not None:
                tool.set_serializer(
                    serializer,
                    layers=layers,
                    layer_options=layer_options,
                    include_cloudpickle=include_cloudpickle,
                    prefer_import=prefer_import,
                    config=self._config,
                )
        if tool.name in self.tools:
            raise ValueError(f"Toolkit {self.name!r} already has tool {tool.name!r}")
        self.tools[tool.name] = tool
        return self

    def add_many(self, tools: Any) -> "Toolkit":
        """Add tools from a list, tuple pairs, or a name-to-tool mapping.

        Args:
            tools (Any): Iterable of callables/tools, ``(name, item)`` pairs, or a mapping from names to items.

        Returns:
            Toolkit: This Toolkit for chaining.
        """
        if isinstance(tools, Mapping):
            for name, item in tools.items():
                self.add(item, name=str(name))
            return self
        for item in tools or []:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                self.add(item[1], name=item[0])
            else:
                self.add(item)
        return self

    def tool(
        self,
        func: Optional[Any] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
        version: str = "1",
        layers: Optional[List[str]] = None,
        layer_options: Optional[Mapping[str, Any]] = None,
        include_cloudpickle: bool = False,
        prefer_import: bool = True,
        serializer: Any = None,
        annotations: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Return a decorator that adds a function to this Toolkit.

        Args:
            func (Optional[Any]): Function when used as ``@toolkit.tool`` without parentheses. Defaults to ``None``.
            name (Optional[str]): Tool alias. Defaults to the function name.
            description (Optional[str]): Tool description. Defaults to the function docstring summary.
            namespace (Optional[str]): Capsule namespace. Defaults to the Toolkit namespace.
            version (str): Capsule version. Defaults to ``"1"``.
            layers (Optional[List[str]]): Optional Capsule layer order for callable capture. Defaults to ``None``.
            layer_options (Optional[Mapping[str, Any]]): Optional per-layer capture options. Defaults to ``None``.
            include_cloudpickle (bool): Whether to store a trusted-local binary fallback. Defaults to ``False``.
            prefer_import (bool): Whether to include import-path restore when available. Defaults to ``True``.
            serializer (Any): Optional callable or Capsule used to serialize raw outputs for agent/MCP boundaries. ``None``
                keeps the default :func:`dumps_jstr` behavior at execution time.
            annotations (Optional[Mapping[str, Any]]): Optional MCP
                ``ToolAnnotations`` mapping retained on the Tool. Defaults to
                ``None``.

        Returns:
            Any: Decorated function or decorator.
        """

        def _decorator(item: Any) -> Any:
            self.add(
                item,
                name=name,
                description=description,
                namespace=namespace,
                version=version,
                layers=layers,
                layer_options=layer_options,
                include_cloudpickle=include_cloudpickle,
                prefer_import=prefer_import,
                serializer=serializer,
                annotations=annotations,
            )
            return item

        return _decorator(func) if func is not None else _decorator

    def list_tools(self) -> List[str]:
        """Return tool names in insertion order.

        Args:
            None.

        Returns:
            List[str]: Tool names in the order they were added.
        """
        return list(self.tools)

    def _add_call_hook(self, hook: Any) -> "Toolkit":
        """Add an internal callback invoked after toolkit execution."""
        if not any(existing is hook for existing in self._call_hooks):
            self._call_hooks.append(hook)
        return self

    def _emit_call_hooks(
        self,
        tool_name: str,
        kwargs: Mapping[str, Any],
        result: Any,
        duration_s: float,
        error: bool,
        *,
        cached: bool = False,
    ) -> None:
        for hook in tuple(self._call_hooks):
            hook(
                tool_name=tool_name,
                kwargs=dict(kwargs),
                result=result,
                duration_s=duration_s,
                error=error,
                cached=cached,
            )

    def _run_with_hooks(self, tool_name: str, kwargs: Mapping[str, Any], call: Any) -> Any:
        if not self._call_hooks:
            return call()
        start = perf_counter()
        error = False
        result: Any = None
        try:
            result = call()
            return result
        except Exception:
            error = True
            raise
        finally:
            self._emit_call_hooks(tool_name, kwargs, result, perf_counter() - start, error)

    async def _arun_with_hooks(self, tool_name: str, kwargs: Mapping[str, Any], call: Any) -> Any:
        if not self._call_hooks:
            return await call()
        start = perf_counter()
        error = False
        result: Any = None
        try:
            result = await call()
            return result
        except Exception:
            error = True
            raise
        finally:
            self._emit_call_hooks(tool_name, kwargs, result, perf_counter() - start, error)

    def run(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        """Run one named tool.

        Args:
            tool_name (str): Name of the tool to call.
            *args (Any): Positional arguments are not supported and raise ``TypeError``.
            **kwargs (Any): Keyword arguments for the selected tool.

        Returns:
            Any: Native Python return value from the tool.
        """
        if args:
            raise TypeError("Toolkit.run expects keyword arguments after the tool name")
        return self._run_with_hooks(tool_name, kwargs, lambda: self.tools[tool_name].run(**kwargs))

    async def arun(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        """Run one named tool and await awaitable return values.

        Args:
            tool_name (str): Name of the tool to call.
            *args (Any): Positional arguments are not supported and raise ``TypeError``.
            **kwargs (Any): Keyword arguments for the selected tool.

        Returns:
            Any: Native Python return value from the tool.
        """
        if args:
            raise TypeError("Toolkit.arun expects keyword arguments after the tool name")
        return await self._arun_with_hooks(tool_name, kwargs, lambda: self.tools[tool_name].arun(**kwargs))

    def to_str(self, tool_name: str, output: Any) -> str:
        """Convert raw output from one named tool through its serializer.

        Args:
            tool_name (str): Name of the tool whose serializer should run.
            output (Any): Native Python value returned by ``run``.

        Returns:
            str: Serialized tool output for agent and MCP boundaries.

        Raises:
            TypeError: If the serializer returns an awaitable value. Use ``ato_str`` instead.
        """
        return self.tools[tool_name].to_str(output)

    async def ato_str(self, tool_name: str, output: Any) -> str:
        """Convert raw output from one named tool through its serializer and await async serializers.

        Args:
            tool_name (str): Name of the tool whose serializer should run.
            output (Any): Native Python value returned by ``run`` or ``arun``.

        Returns:
            str: Serialized tool output for agent and MCP boundaries.
        """
        return await self.tools[tool_name].ato_str(output)

    def run_to_str(self, tool_name: str, *args: Any, **kwargs: Any) -> str:
        """Run one named tool and return its serialized string output.

        Args:
            tool_name (str): Name of the tool to call.
            *args (Any): Positional arguments are not supported and raise ``TypeError``.
            **kwargs (Any): Keyword arguments for the selected tool.

        Returns:
            str: Serialized tool output for agent, MCP, and LLM boundaries.

        Raises:
            TypeError: If the Capsule returns an awaitable value. Use ``arun_to_str`` instead.
        """
        if args:
            raise TypeError("Toolkit.run_to_str expects keyword arguments after the tool name")
        return self._run_with_hooks(tool_name, kwargs, lambda: self.tools[tool_name].run_to_str(**kwargs))

    async def arun_to_str(self, tool_name: str, *args: Any, **kwargs: Any) -> str:
        """Run one named tool asynchronously and return its serialized string output.

        Args:
            tool_name (str): Name of the tool to call.
            *args (Any): Positional arguments are not supported and raise ``TypeError``.
            **kwargs (Any): Keyword arguments for the selected tool.

        Returns:
            str: Serialized tool output for agent, MCP, and LLM boundaries.
        """
        if args:
            raise TypeError("Toolkit.arun_to_str expects keyword arguments after the tool name")
        return await self._arun_with_hooks(tool_name, kwargs, lambda: self.tools[tool_name].arun_to_str(**kwargs))

    def set_serializer(self, tool_name: str, serializer: Any = None, **kwargs: Any) -> "Toolkit":
        """Set the serializer for one named tool.

        Args:
            tool_name (str): Tool name.
            serializer (Any): Callable or Capsule that accepts the raw output and returns text. ``None`` clears any custom
                serializer and restores the default :func:`dumps_jstr` behavior at execution time.
            **kwargs (Any): Serializer Capsule capture options accepted by ``Tool.set_serializer``.

        Returns:
            Toolkit: This Toolkit for chaining.
        """
        kwargs.setdefault("config", self._config)
        self.tools[tool_name].set_serializer(serializer, **kwargs)
        return self

    def to_manifest(self) -> Dict[str, Any]:
        """Return a persistent Toolkit manifest referencing Capsules by id.

        Each ``capsule_refs`` entry stores both the main Capsule and its serializer Capsule.
        ``capabilities`` is keyed by tool alias and nests ``main`` and ``serializer`` capability
        summaries for the two Capsules.

        Args:
            None.

        Returns:
            Dict[str, Any]: JSON-safe manifest suitable for registry persistence.
        """
        refs = [
            {
                "name": tool.capsule.manifest.name,
                "alias": name,
                "capsule_id": tool.capsule.manifest.id,
                "serializer_capsule_id": tool._serializer_capsule().manifest.id,
                "serializer_name": tool._serializer_capsule().manifest.name,
                "version": tool.capsule.manifest.version,
                "serializer_version": tool._serializer_capsule().manifest.version,
                "description": tool.description,
                "input_schema": deepcopy(tool.input_schema),
                "output_schema": deepcopy(tool.output_schema),
                "annotations": deepcopy(tool.annotations),
                "metadata": deepcopy(tool.metadata),
                "required": True,
            }
            for name, tool in self.tools.items()
        ]
        capabilities = {
            name: {"main": tool.capsule.manifest.capabilities, "serializer": tool._serializer_capsule().manifest.capabilities}
            for name, tool in self.tools.items()
        }
        manifest = {
            "id": self.id,
            "name": self.name,
            "namespace": self.namespace,
            "version": self.version,
            "status": "active",
            "description": self.description,
            "capsule_refs": refs,
            "serving": {},
            "capabilities": capabilities,
            "created_at": time(),
            "updated_at": time(),
        }
        manifest["manifest"] = dict(manifest)
        return manifest

    def to_row(self, existing: Mapping[str, Any] | None = None, *, reason: str = "") -> Dict[str, Any]:
        """Return the persistent ``sys-toolkit`` row for this Toolkit.

        Args:
            existing (Mapping[str, Any] | None): Optional existing row used to preserve ``created_at``. Defaults to ``None``.
            reason (str): Reason stored on the row. Defaults to ``""``.

        Returns:
            Dict[str, Any]: Row payload including ``object_id`` and manifest fields.
        """
        now = time()
        manifest = self.to_manifest()
        created_at = (existing or {}).get("created_at") or now
        manifest["created_at"] = created_at
        manifest["updated_at"] = now
        manifest["manifest"] = dict(manifest)
        row = dict(manifest)
        row["object_id"] = manifest["id"]
        row["last_reason"] = reason
        return row

    def register(
        self,
        *,
        ws: Any = None,
        overwrite: bool = True,
        on_conflict: Optional[str] = None,
        reason: str = "",
    ) -> str:
        """Register this Toolkit and its Capsules in one workspace.

        Args:
            ws (Any): Workspace that owns the Toolkit and referenced Capsule
                rows. When omitted, `HeavenBase.load()` resolves the active or
                configured default workspace without creating one.
            overwrite (bool): Whether same-id Capsule and Toolkit rows should
                be replaced. Defaults to `True`.
            on_conflict (Optional[str]): Conflict mode. Supported values:
                - `"overwrite"`: Replace different content at the same id.
                - `"raise"`: Reject different content at the same id.
                - `"skip"`: Keep the existing row.
                When omitted, follows `overwrite`.
            reason (str): Reason stored on the Toolkit row.

        Returns:
            str: Stable Toolkit id.

        Raises:
            KeyError: If `ws` is omitted and no selected workspace exists.

        Warnings:
            Runtime-only workspace/backend Toolkits cannot be registered because they close over live objects instead of durable functions.
        """
        from .registry import toolkit_registry

        store = toolkit_registry(ws if ws is not None else self._workspace)
        toolkit_id = store.register(self, overwrite=overwrite, on_conflict=on_conflict, reason=reason)
        self._workspace = store.workspace
        return toolkit_id

    @classmethod
    def load(
        cls,
        toolkit_id: Optional[str] = None,
        *,
        ws: Any = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "Toolkit":
        """Load a Toolkit and its Capsules from one workspace.

        Args:
            toolkit_id (Optional[str]): Stable Toolkit id. Defaults to ``None``.
            ws (Any): Workspace that owns the Toolkit row. When omitted,
                `HeavenBase.load()` resolves the active or configured default
                workspace without creating one.
            name (Optional[str]): Toolkit name when loading by identity. Defaults to ``None``.
            namespace (Optional[str]): Toolkit namespace. Defaults to configured Toolkit namespace.
            version (Optional[str]): Toolkit version. Defaults to latest matching row.

        Returns:
            Toolkit: Runtime Toolkit whose tools restore referenced Capsules lazily.

        Raises:
            KeyError: If the workspace or Toolkit row does not exist.

        Warnings:
            Loading a Toolkit may execute Capsule restore code when a tool is first called or exported.
        """
        from .registry import toolkit_registry

        store = toolkit_registry(ws)
        return store.load(
            toolkit_id,
            name=name,
            namespace=namespace or _default_namespace(store.authority.config),
            version=version,
        )

    @classmethod
    def list(
        cls,
        *,
        ws: Any = None,
        namespace: Optional[str] = None,
        status: Optional[str] = "active",
    ) -> List[Dict[str, Any]]:
        """List Toolkit rows from one workspace.

        Args:
            ws (Any): Workspace that owns the Toolkit rows. When omitted,
                `HeavenBase.load()` resolves the selected workspace.
            namespace (Optional[str]): Optional namespace filter. Defaults to ``None``.
            status (Optional[str]): Optional status filter. Defaults to ``"active"``.

        Returns:
            List[Dict[str, Any]]: Matching Toolkit rows.
        """
        from .registry import toolkit_registry

        return toolkit_registry(ws).list(namespace=namespace, status=status)

    @classmethod
    def delete(
        cls,
        toolkit_id: str,
        *,
        ws: Any = None,
        tombstone: bool = True,
    ) -> bool:
        """Delete or tombstone a Toolkit row in one workspace.

        Args:
            toolkit_id (str): Stable Toolkit id.
            ws (Any): Workspace that owns the Toolkit row. When omitted,
                `HeavenBase.load()` resolves the selected workspace.
            tombstone (bool): Mark deleted instead of physically deleting. Defaults to ``True``.

        Returns:
            bool: Whether an existing row was removed or tombstoned.
        """
        from .registry import toolkit_registry

        return toolkit_registry(ws).delete(toolkit_id, tombstone=tombstone)

    @classmethod
    def describe(
        cls,
        toolkit_id: Optional[str] = None,
        *,
        ws: Any = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return one Toolkit row from a workspace.

        Args:
            toolkit_id (Optional[str]): Stable Toolkit id. Defaults to ``None``.
            ws (Any): Workspace that owns the Toolkit row. When omitted,
                `HeavenBase.load()` resolves the selected workspace.
            name (Optional[str]): Toolkit name when loading by identity. Defaults to ``None``.
            namespace (Optional[str]): Toolkit namespace. Defaults to configured Toolkit namespace.
            version (Optional[str]): Explicit version. Defaults to the latest matching row.

        Returns:
            Dict[str, Any]: Raw Toolkit row.
        """
        from .registry import toolkit_registry

        store = toolkit_registry(ws)
        return store.describe(
            toolkit_id,
            name=name,
            namespace=namespace or _default_namespace(store.authority.config),
            version=version,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the persistent Toolkit manifest.

        Args:
            None.

        Returns:
            Dict[str, Any]: JSON-safe manifest from :meth:`to_manifest`.
        """
        return self.to_manifest()

    def to_fastmcp(self, name: Optional[str] = None, *, transforms: Any = None) -> Any:
        """Return a FastMCP server for this Toolkit.

        Args:
            name (Optional[str]): Optional server name. Defaults to the Toolkit name.
            transforms (Any): Optional FastMCP transforms. When omitted,
                `heavenbase.mcp.transforms` is used.

        Returns:
            Any: FastMCP server instance.
        """
        from .mcp import toolkit_to_fastmcp

        return toolkit_to_fastmcp(self, name=name, transforms=transforms)

    @classmethod
    def from_fastmcp(
        cls,
        server: Any,
        *,
        name: Optional[str] = None,
        namespace: str = "mcp-client",
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> "Toolkit":
        """Import a FastMCP server as a runtime Toolkit client.

        Args:
            server (Any): FastMCP server or client transport accepted by ``fastmcp.Client``.
            name (Optional[str]): Toolkit name. Defaults to the remote server name or ``"mcp-client"``.
            namespace (str): Toolkit namespace. Defaults to ``"mcp-client"``.
            resolver: Context-owned module resolver.
            config: Config authority paired with ``resolver``.

        Returns:
            Toolkit: Runtime-only Toolkit that proxies calls through FastMCP.
        """
        from .mcp import toolkit_from_fastmcp

        return toolkit_from_fastmcp(server, name=name, namespace=namespace, resolver=resolver, config=config)

    def to_mcp_json(
        self,
        *,
        transport: str = "http",
        host: Optional[str] = None,
        port: Optional[int] = None,
        workspace: Optional[str] = None,
    ) -> str:
        """Return an MCP client config JSON string.

        Args:
            transport (str): MCP transport, usually ``"http"``, ``"streamable-http"``, or ``"stdio"``. Defaults to ``"http"``.
            host (Optional[str]): HTTP host. Defaults to ``None``.
            port (Optional[int]): HTTP port. Defaults to ``None``.
            workspace (Optional[str]): Workspace id added to stdio args.
                Defaults to the workspace bound by `Toolkit.load`.

        Returns:
            str: JSON client config.
        """
        from .mcp import toolkit_to_mcp_json

        return toolkit_to_mcp_json(self, transport=transport, host=host, port=port, workspace=workspace)

    def serve(self, **kwargs: Any) -> Any:
        """Serve this Toolkit as an MCP server.

        Args:
            **kwargs (Any): Transport, host, port, and FastMCP server options.

        Returns:
            Any: Result from the underlying FastMCP server runner.
        """
        from .server import serve_toolkit

        kwargs.setdefault("config", self._config)
        return serve_toolkit(self, **kwargs)

    def to_anthropic_tools(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Return Anthropic Programmatic Tool Calling definitions.

        Args:
            **kwargs (Any): Export options forwarded to the Anthropic helper.

        Returns:
            List[Dict[str, Any]]: Tool definition dictionaries.
        """
        from .anthropic import to_anthropic_tools

        return to_anthropic_tools(self, **kwargs)

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible function tool definitions.

        Args:
            None.

        Returns:
            List[Dict[str, Any]]: OpenAI function-tool dictionaries for each registered tool.
        """
        return [tool.to_openai_tool() for tool in self.tools.values()]
