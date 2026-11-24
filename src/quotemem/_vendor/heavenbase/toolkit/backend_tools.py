"""Backend-specific MCP toolkit builders."""

from __future__ import annotations

__all__ = [
    "backend_toolkit",
]

from heavenbase.utils import Any, Iterable
from heavenbase.modules import ModuleResolver, module_resolver
from .serializers import ToolSerializer
from .toolkit import Toolkit


def _scoped_callable(item: Any, scope: Any | None) -> Any:
    """Apply an explicitly supplied workspace scope to one runtime callable."""
    return item if scope is None else scope(item)


def backend_toolkit(
    backend: Any,
    *,
    name: str | None = None,
    description: str | None = None,
    profile: Any = None,
    tools: Iterable[str] | None = None,
    resolver: ModuleResolver | None = None,
    scope: Any | None = None,
) -> Toolkit:
    """Build a Toolkit directly from one backend instance.

    Args:
        backend: Backend instance whose ``mcp()`` spec should be exposed.
        name: Optional Toolkit name override. Defaults to ``backend-<backend.name>`` or the configured
            ``heavenbase.mcp.backend_toolkit`` value.
        description: Optional Toolkit description override. Defaults to the backend MCP spec description
            plus the selected profile/tool list.
        profile: Optional backend MCP profile selection accepted by ``BackendMCP.select``.
        tools: Optional explicit allowlist of backend tool names. Overrides ``profile`` when supplied.
        resolver: Context-owned module resolver. When omitted, the backend's
            resolver is selected.
        scope: Explicit workspace scope decorator supplied by the workspace
            MCP boundary. Direct Backend toolkits leave this as ``None``.

    Returns:
        Toolkit: Runtime-only Toolkit bound to the live backend instance.

    Examples:
        Build a tag-inspection-only backend Toolkit::

            toolkit = backend_toolkit(backend, tools=["backend_tags"])
    """
    chosen = module_resolver(resolver if resolver is not None else backend.module_resolver)
    spec = backend.mcp()
    selected_profile, selected_tools = spec.select(profile=profile, tools=tools)
    configured_name = backend.config.get("heavenbase.mcp.backend_toolkit", default="")
    toolkit_name = name or str(configured_name or f"backend-{backend.name}")
    toolkit = Toolkit(
        toolkit_name,
        description=description or _toolkit_description(backend, spec.desc, selected_profile, selected_tools),
        namespace=f"backend-{backend.name}",
        resolver=chosen,
        config=backend.config,
    )
    serializer = _scoped_callable(
        ToolSerializer.load(spec.serializer, resolver=chosen).serialize,
        scope,
    )
    for tool in selected_tools:
        toolkit.add(
            _scoped_callable(tool.fn, scope),
            name=tool.name,
            description=tool.desc,
            namespace=toolkit.namespace,
            version="1",
            serializer=serializer,
            annotations={"readOnlyHint": tool.read_only},
        )
        toolkit.tools[tool.name].metadata["runtime_only"] = True
        toolkit.tools[tool.name].metadata["read_only"] = tool.read_only
        toolkit.tools[tool.name].capsule.manifest.capabilities["registry_persistence"] = {
            "ok": False,
            "reasons": ["backend MCP tools close over a live backend instance"],
        }
    return toolkit


def _toolkit_description(backend: Any, spec_desc: str, profile: Any, tools: list[Any]) -> str:
    """Return the backend Toolkit description for one selected profile.

    Args:
        backend: Backend instance being exposed.
        spec_desc: Backend MCP specification prompt/description.
        profile: Resolved backend MCP profile.
        tools: Ordered tool contracts included in the Toolkit.

    Returns:
        str: Description used for the runtime Toolkit and MCP server metadata.
    """
    tool_names = [tool.name for tool in tools]
    suffix = f" Profile: {profile.name}." if profile.name else ""
    profile_desc = f" {profile.desc}" if profile.desc else ""
    return f"HeavenBase backend {backend.name!r} ({backend.type}).{suffix} {spec_desc}{profile_desc} Available tools: {', '.join(tool_names)}."
