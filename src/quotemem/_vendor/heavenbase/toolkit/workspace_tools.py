"""HeavenBase workspace and backend MCP toolkit builders."""

from __future__ import annotations

__all__ = [
    "TOOL_DESCS",
    "TOOLKIT_DESC",
    "registered_workspace_tool_names",
    "workspace_toolkit",
]

from heavenbase.utils import Any, Iterable, List, Mapping, Optional

from heavenbase.extensions.ids import TOOLKIT_EXTENSION_ID
from heavenbase.modules import ModuleResolver, module_resolver
from heavenbase.workspace.setup.lifecycle import _workspace_config_scope

from .families.registry import workspace_toolkit_tool_names
from .families.assembly import (
    available_workspace_tool_names,
    skill_tools_enabled,
    workspace_tool_functions,
    workspace_toolkit_description,
)
from .families.workspace import WORKSPACE_CORE_TOOL_NAMES
from .logging import attach_mcp_call_logging, mcp_call_logging_enabled, open_mcp_call_log
from .prompts import WORKSPACE_TOOL_DESCS, WORKSPACE_TOOLKIT_DESC
from .profiles import resolve_profile_tools
from .registry_metadata import sync_mcp_registry_metadata, workspace_mcp_profile_specs, workspace_mcp_serializers, workspace_mcp_toolkit_families
from .serializers import ToolSerializer
from .toolkit import Toolkit

TOOLKIT_DESC = WORKSPACE_TOOLKIT_DESC.text

TOOL_DESCS = {name: spec.text for name, spec in WORKSPACE_TOOL_DESCS.items()}

_WORKSPACE_TOOL_NAMES = WORKSPACE_CORE_TOOL_NAMES


def _callable_doc_summary(func: Any) -> str:
    """Return the first non-empty docstring line for one tool callable."""
    doc = getattr(func, "__doc__", None)
    if not isinstance(doc, str):
        return ""
    for line in doc.strip().splitlines():
        text = line.strip()
        if text:
            return text
    return ""


def _family_tool_descriptions(
    toolkit_families: Mapping[str, Any] | None,
    *,
    resolver: ModuleResolver | None = None,
    workspace: Any = None,
) -> dict[str, str]:
    """Collect optional per-tool description overlays from family metadata."""
    from .families.registry import workspace_toolkit_family_specs

    out: dict[str, str] = {}
    specs = list(workspace_toolkit_family_specs(resolver=resolver, workspace=workspace))
    specs.extend((toolkit_families or {}).values())
    for spec in specs:
        meta = getattr(spec, "meta", None) or {}
        if not isinstance(meta, Mapping):
            continue
        descs = meta.get("tool_descriptions") or meta.get("tool_descs") or {}
        if not isinstance(descs, Mapping):
            continue
        for tool_name, text in descs.items():
            cleaned = str(text or "").strip()
            if cleaned:
                out[str(tool_name)] = cleaned
    return out


def _tool_description(tool_name: str, func: Any, *, overlays: Mapping[str, str] | None = None) -> str:
    """Resolve one MCP tool description from host map, family overlay, or docstring."""
    mapped = TOOL_DESCS.get(tool_name)
    if mapped:
        return mapped
    if overlays:
        overlay = str(overlays.get(tool_name) or "").strip()
        if overlay:
            return overlay
    return _callable_doc_summary(func) or tool_name


def _heavenbase_cls() -> Any:
    from heavenbase.workspace import HeavenBase

    return HeavenBase


def registered_workspace_tool_names(
    *,
    resolver: ModuleResolver | None = None,
    workspace: Any | None = None,
) -> tuple[str, ...]:
    """Return installed or workspace-eligible MCP tool names.

    Args:
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the default Context is selected.
        workspace (Any | None): Optional workspace whose active extensions constrain family contributions.

    Returns:
        tuple[str, ...]: Core workspace tool names followed by registered family tools without duplicates.
    """
    return workspace_toolkit_tool_names(_WORKSPACE_TOOL_NAMES, resolver=resolver, workspace=workspace)


def workspace_toolkit(
    workspace: Any,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    profile: Optional[str] = None,
    tools: Optional[Iterable[str]] = None,
    resolver: ModuleResolver | None = None,
) -> Toolkit:
    """Build a toolkit bound to a HeavenBase workspace.

    Args:
        workspace: Workspace whose entities and rows the tools should manage.
        name: Optional toolkit name override.
        description: Optional toolkit description override.
        profile: Optional configured MCP profile name.
        tools: Optional explicit allowlist of tool names.
        resolver: Context-owned module resolver. When omitted, the workspace's
            resolver is selected.

    Returns:
        Runtime-only toolkit containing workspace MCP tools.

    Raises:
        TypeError: If ``workspace`` is not a ``HeavenBase`` instance.
        ValueError: If the selected profile or tool allowlist is invalid.
    """
    HeavenBase = _heavenbase_cls()
    if not isinstance(workspace, HeavenBase):
        raise TypeError("to_toolkit expects a HeavenBase instance")
    if TOOLKIT_EXTENSION_ID not in workspace.extensions():
        raise RuntimeError("Workspace Toolkit support is inactive; call enable_extension('toolkit') before building a Toolkit")
    chosen = module_resolver(resolver if resolver is not None else workspace.module_resolver)
    config = workspace.context.config
    profile_specs = workspace_mcp_profile_specs(workspace, resolver=chosen)
    serializers = workspace_mcp_serializers(workspace, resolver=chosen)
    toolkit_families = workspace_mcp_toolkit_families(workspace, resolver=chosen)
    sync_mcp_registry_metadata(
        workspace,
        profile_specs=profile_specs,
        serializers=serializers,
        toolkit_family_specs=toolkit_families,
        resolver=chosen,
    )
    configured_name = config.get("heavenbase.mcp.workspace_toolkit", default="")
    toolkit_name = name or str(configured_name or f"heavenbase-{workspace.id}")
    available = available_workspace_tool_names(
        workspace,
        registered_workspace_tool_names(resolver=chosen, workspace=workspace),
        toolkit_families,
        resolver=chosen,
    )
    profile_spec, tool_names = resolve_profile_tools(
        available,
        profile=profile,
        tools=tools,
        profile_specs=profile_specs,
        serializer_names=serializers,
        resolver=chosen,
        config=config,
        workspace=workspace,
    )
    if tools is None and profile_spec.skills and skill_tools_enabled(workspace, resolver=chosen):
        for skill_tool in ("list_skills", "read_skill"):
            if skill_tool not in tool_names:
                tool_names.append(skill_tool)
    funcs = workspace_tool_functions(workspace, profile_spec, tool_names, toolkit_families, resolver=chosen)
    tool_overlays = _family_tool_descriptions(toolkit_families, resolver=chosen, workspace=workspace)
    serializer = _workspace_config_scope(workspace.context, workspace.id)(
        ToolSerializer.load(profile_spec.serializer, aliases=serializers, resolver=chosen, workspace=workspace).serialize
    )
    namespace = f"workspace-{workspace.id}"
    toolkit = Toolkit(
        toolkit_name,
        description=description
        or workspace_toolkit_description(
            workspace,
            tool_names,
            profile_spec,
            toolkit_families,
            resolver=chosen,
        ),
        namespace=namespace,
        resolver=chosen,
        config=config,
    )
    for tool_name in tool_names:
        func = funcs[tool_name]
        description = _tool_description(tool_name, func, overlays=tool_overlays)
        toolkit.add(
            func,
            name=tool_name,
            description=description,
            namespace=namespace,
            version="1",
            serializer=serializer,
        )
        toolkit.tools[tool_name].metadata["runtime_only"] = True
        toolkit.tools[tool_name].capsule.manifest.capabilities["registry_persistence"] = {
            "ok": False,
            "reasons": ["workspace-bound MCP tools close over a live HeavenBase workspace"],
        }
    if mcp_call_logging_enabled(config=config):
        log = open_mcp_call_log(
            workspace,
            toolkit_name=toolkit_name,
            profile=profile_spec.name,
            resolver=chosen,
        )
        attach_mcp_call_logging(toolkit, workspace, log)
    return toolkit
