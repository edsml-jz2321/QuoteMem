"""Workspace MCP toolkit family assembly helpers."""

from __future__ import annotations

__all__ = [
    "SKILL_TOOL_NAMES",
    "available_workspace_tool_names",
    "skill_tools_enabled",
    "workspace_tool_functions",
    "workspace_toolkit_description",
]

from heavenbase.utils import Any, Dict, List, Mapping, Optional

from heavenbase.modules import ModuleResolver, module_resolver
from heavenbase.workspace.setup.lifecycle import _workspace_config_scope
from ..prompts import WORKSPACE_TOOLKIT_DESC
from .family import WorkspaceToolkitFamilySpec
from .registry import workspace_toolkit_family_specs
from .workspace import WorkspaceToolContext, entity_summaries, workspace_core_tool_functions
from ..profile import ProfileSpec

SKILL_TOOL_NAMES = frozenset({"list_skills", "read_skill"})


def skill_tools_enabled(workspace: Any, *, resolver: ModuleResolver | None = None) -> bool:
    """Return whether an eligible installed family provides Agent skill tools.

    Args:
        workspace (Any): HeavenBase workspace whose active extensions constrain family eligibility.
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the workspace resolver is selected.

    Returns:
        bool: ``True`` when an eligible toolkit family declares ``list_skills`` or ``read_skill``.
    """
    chosen = module_resolver(resolver if resolver is not None else workspace.module_resolver)
    return any(SKILL_TOOL_NAMES.intersection(spec.tool_names) for spec in workspace_toolkit_family_specs(resolver=chosen, workspace=workspace))


def available_workspace_tool_names(
    workspace: Any,
    core_tool_names: tuple[str, ...],
    toolkit_families: Optional[Mapping[str, WorkspaceToolkitFamilySpec]] = None,
    *,
    resolver: ModuleResolver | None = None,
) -> tuple[str, ...]:
    """Return core plus extension-contributed workspace MCP tool names.

    Args:
        workspace (Any): HeavenBase workspace used to resolve extension families.
        core_tool_names (tuple[str, ...]): Protocol-level core tool names to place first.
        toolkit_families (Optional[Mapping[str, WorkspaceToolkitFamilySpec]]): Optional family overlay keyed by identifier.
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the workspace resolver is selected.

    Returns:
        tuple[str, ...]: Core names followed by family tool names without duplicates.
    """
    names = list(core_tool_names)
    for spec in (toolkit_families or {}).values():
        for tool_name in spec.tool_names:
            if tool_name not in names:
                names.append(tool_name)
    return tuple(names)


def workspace_tool_functions(
    workspace: Any,
    profile: ProfileSpec,
    tool_names: List[str],
    toolkit_families: Optional[Mapping[str, WorkspaceToolkitFamilySpec]] = None,
    *,
    resolver: ModuleResolver | None = None,
) -> Dict[str, Any]:
    """Build selected workspace and extension MCP tool functions.

    Args:
        workspace (Any): HeavenBase workspace whose entities and rows the tools manage.
        profile (ProfileSpec): Resolved MCP profile controlling tool, entity, and serializer scope.
        tool_names (List[str]): Concrete tool names to materialize in declaration order.
        toolkit_families (Optional[Mapping[str, WorkspaceToolkitFamilySpec]]): Optional family overlay keyed by identifier.
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the workspace resolver is selected.

    Returns:
        Dict[str, Any]: Tool name to MCP-callable function, each wrapped in the workspace config scope.
    """
    chosen = module_resolver(resolver if resolver is not None else workspace.module_resolver)
    context = WorkspaceToolContext(workspace=workspace, profile=profile, tool_names=tuple(tool_names))
    funcs = workspace_core_tool_functions(context)
    extension_funcs = _extension_tool_functions(workspace, context, chosen, toolkit_families)
    for tool_name in extension_funcs:
        if tool_name in funcs:
            raise ValueError(f"Duplicate MCP tool name {tool_name!r} conflicts with a core workspace tool")
    funcs.update(extension_funcs)
    return {name: _workspace_config_scope(workspace.context, workspace.id)(func) for name, func in funcs.items()}


def _extension_tool_functions(
    workspace: Any,
    context: WorkspaceToolContext,
    resolver: ModuleResolver,
    toolkit_families: Optional[Mapping[str, WorkspaceToolkitFamilySpec]] = None,
) -> Dict[str, Any]:
    funcs: Dict[str, Any] = {}
    owners: Dict[str, str] = {}
    selected = set(context.tool_names)
    for spec in _toolkit_family_specs(resolver, toolkit_families, workspace=workspace):
        metadata_alias = spec.meta.get("source") == "workspace" and (bool(spec.meta.get("metadata_only")) or spec.meta.get("alias_of") is not None)
        if spec.name == "workspace" or metadata_alias:
            continue
        declared = [tool_name for tool_name in spec.tool_names if tool_name in selected]
        if not declared:
            continue
        built = dict(_build_family_tool_functions(spec, workspace, context))
        missing = [tool_name for tool_name in declared if tool_name not in built]
        if missing:
            raise ValueError(f"Workspace MCP toolkit family {spec.name!r} builder is missing declared tools {missing!r}")
        for tool_name in declared:
            previous = owners.get(tool_name)
            if previous is not None and previous != spec.name:
                raise ValueError(f"Duplicate MCP tool name {tool_name!r} contributed by families {previous!r} and {spec.name!r}")
            funcs[tool_name] = built[tool_name]
            owners[tool_name] = spec.name
    return funcs


def _build_family_tool_functions(spec: WorkspaceToolkitFamilySpec, workspace: Any, context: WorkspaceToolContext) -> Mapping[str, Any]:
    builder_context = spec.meta.get("builder_context", "workspace")
    if builder_context == "workspace_tool_context":
        return spec.builder(context)
    if builder_context in ("workspace", "", None):
        return spec.builder(workspace)
    raise ValueError(f"Unknown workspace toolkit family builder_context {builder_context!r} for family {spec.name!r}")


def _toolkit_family_specs(
    resolver: ModuleResolver,
    toolkit_families: Optional[Mapping[str, WorkspaceToolkitFamilySpec]] = None,
    *,
    workspace: Any | None = None,
) -> tuple[WorkspaceToolkitFamilySpec, ...]:
    specs = list((toolkit_families or {}).values())
    specs.extend(workspace_toolkit_family_specs(resolver=resolver, workspace=workspace))
    return tuple(specs)


def _provider_description(
    tool_names: List[str],
    resolver: ModuleResolver,
    toolkit_families: Optional[Mapping[str, WorkspaceToolkitFamilySpec]] = None,
    *,
    workspace: Any | None = None,
) -> str:
    selected = set(tool_names)
    if not selected:
        return ""
    for spec in _toolkit_family_specs(resolver, toolkit_families, workspace=workspace):
        provider_set = set(spec.tool_names)
        if spec.desc and (selected <= provider_set or provider_set.intersection(selected)):
            return spec.desc
    return ""


def workspace_toolkit_description(
    workspace: Any,
    tool_names: Optional[List[str]] = None,
    profile: Optional[ProfileSpec] = None,
    toolkit_families: Optional[Mapping[str, WorkspaceToolkitFamilySpec]] = None,
    *,
    resolver: ModuleResolver | None = None,
) -> str:
    """Return the provider/profile/entity-aware workspace toolkit description.

    Args:
        workspace (Any): HeavenBase workspace whose entities appear in the description suffix.
        tool_names (Optional[List[str]]): Selected tool names appended to the description. Defaults to ``None``.
        profile (Optional[ProfileSpec]): Active MCP profile appended when present. Defaults to ``None``.
        toolkit_families (Optional[Mapping[str, WorkspaceToolkitFamilySpec]]): Optional family overlay used for provider copy.
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the workspace resolver is selected.

    Returns:
        str: Human-readable toolkit description including workspace id, profile, entities, and tools.
    """
    chosen = module_resolver(resolver if resolver is not None else workspace.module_resolver)
    names = list(tool_names or [])
    provider_desc = _provider_description(names, chosen, toolkit_families, workspace=workspace)
    if provider_desc:
        text = provider_desc
    else:
        base = WORKSPACE_TOOLKIT_DESC.text
        text = base.replace("upsert/set/delete for mutations", "upsert/set for single-row mutations") if names and "delete" not in names else base
    entity_lines = entity_summaries(workspace, profile.entities if profile is not None else "*")
    entity_suffix = f" Entities: {'; '.join(entity_lines)}." if entity_lines else ""
    suffix = f" Available tools: {', '.join(names)}." if names else ""
    profile_suffix = f" Profile: {profile.name}." if profile is not None and profile.name else ""
    return f"HeavenBase workspace {workspace.id!r}.{profile_suffix} {text}{entity_suffix}{suffix}"
