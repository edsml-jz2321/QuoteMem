"""Workspace MCP toolkit family publication and materialization."""

from __future__ import annotations

__all__ = [
    "WorkspaceToolkitFamilySpec",
    "register_workspace_toolkit_family",
    "registered_workspace_toolkit_families",
    "workspace_toolkit_family_specs",
    "workspace_toolkit_tool_names",
]

from heavenbase.utils import Any, Dict, Iterable, Mapping
from heavenbase.utils.registry import named_key

from heavenbase.extensions.contributions import eligible_contributions
from heavenbase.modules import ModuleResolver, ModuleService, module_resolver, python_executable_ref
from heavenbase.modules.base import ResolverCache
from .family import WorkspaceToolkitFamilySpec

_FAMILY_CACHE = ResolverCache[WorkspaceToolkitFamilySpec]()


def register_workspace_toolkit_family(
    spec: WorkspaceToolkitFamilySpec,
    *,
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> WorkspaceToolkitFamilySpec:
    """Publish one workspace MCP toolkit family as an ordinary record.

    Args:
        spec: Family specification to publish.
        replace: Whether a programmatically owned record may be replaced.
        resolver: Writable Context-owned module resolver. When omitted, the
            default Context is selected.

    Returns:
        WorkspaceToolkitFamilySpec: The materialized registered family.

    Raises:
        ValueError: If another owner controls the identifier or replacement is
            not allowed.
    """
    name = named_key(spec.name, "workspace toolkit family name")
    chosen = module_resolver(resolver)
    if not isinstance(chosen, ModuleService):
        raise TypeError("Workspace toolkit family registration requires a writable ModuleService")
    service = chosen
    found = service.inspect("toolkit_family", name)
    if found:
        existing = _family_from_row(found[0], service)
        if existing == spec and replace:
            return existing
        if not replace:
            raise ValueError(f"Workspace toolkit family {name!r} is already registered")
        coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if coordinate != f"programmatic/toolkit_family/{name}":
            raise ValueError(f"Workspace toolkit family {name!r} belongs to {coordinate!r} and cannot be replaced")
    service.register(
        "toolkit_family",
        name,
        {
            "name": name,
            "desc": spec.desc,
            "source": "inline",
            "target": "definition",
            "active": True,
            "meta": {
                "schema_version": 1,
                "tags": {"toolkit": True, "mcp": True},
                "definition": {
                    "tool_names": list(spec.tool_names),
                    "builder": python_executable_ref(spec.builder, label=f"workspace toolkit family {name!r} builder"),
                    "meta": dict(spec.meta),
                },
            },
        },
    )
    return _load_family(name, service)


def workspace_toolkit_family_specs(
    *,
    resolver: ModuleResolver | None = None,
    workspace: Any | None = None,
) -> tuple[WorkspaceToolkitFamilySpec, ...]:
    """Return installed or workspace-eligible toolkit families in Registry order.

    Args:
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the default Context is selected.
        workspace (Any | None): Optional workspace whose active extensions constrain eligible contributions.

    Returns:
        tuple[WorkspaceToolkitFamilySpec, ...]: Materialized family specifications in Registry order.
    """
    chosen = module_resolver(resolver)
    rows = eligible_contributions(chosen.inspect("toolkit_family"), workspace)
    return tuple(_family_from_row(row, chosen) for row in rows)


def registered_workspace_toolkit_families(
    *,
    resolver: ModuleResolver | None = None,
    workspace: Any | None = None,
) -> Dict[str, WorkspaceToolkitFamilySpec]:
    """Return installed or workspace-eligible toolkit families by identifier.

    Args:
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the default Context is selected.
        workspace (Any | None): Optional workspace whose active extensions constrain eligible contributions.

    Returns:
        Dict[str, WorkspaceToolkitFamilySpec]: Family specifications keyed by identifier.
    """
    return {spec.name: spec for spec in workspace_toolkit_family_specs(resolver=resolver, workspace=workspace)}


def workspace_toolkit_tool_names(
    core_tool_names: Iterable[str] = (),
    *,
    resolver: ModuleResolver | None = None,
    workspace: Any | None = None,
) -> tuple[str, ...]:
    """Return workspace MCP tool names in core/family order.

    Args:
        core_tool_names: Protocol-level core names to place first.
        resolver: Context-owned module resolver. When omitted, the default
            Context is selected.

    Returns:
        tuple[str, ...]: Core and registered family names without duplicates.
    """
    chosen = module_resolver(resolver)
    names = [named_key(name, "workspace MCP tool name") for name in core_tool_names]
    for row in eligible_contributions(chosen.inspect("toolkit_family"), workspace):
        for tool_name in _family_tool_names(row):
            if tool_name not in names:
                names.append(tool_name)
    return tuple(names)


def registered_family_tool_names(
    *,
    resolver: ModuleResolver | None = None,
    workspace: Any | None = None,
) -> tuple[str, ...]:
    """Return concrete tool names declared by installed or eligible records.

    Args:
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the default Context is selected.
        workspace (Any | None): Optional workspace whose active extensions constrain eligible contributions.

    Returns:
        tuple[str, ...]: Declared tool names in first-seen Registry order without duplicates.
    """
    chosen = module_resolver(resolver)
    names: list[str] = []
    for row in eligible_contributions(chosen.inspect("toolkit_family"), workspace):
        for name in _family_tool_names(row):
            if name not in names:
                names.append(name)
    return tuple(names)


def _family_tool_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Read tool names from one inline family record without importing code."""
    record = row["record"]
    meta = record.get("meta")
    if record.get("source") != "inline" or not isinstance(meta, Mapping):
        raise TypeError(f"Workspace toolkit family {row['identifier']!r} must use an inline definition")
    definition = meta.get(str(record.get("target")))
    if not isinstance(definition, Mapping):
        raise TypeError(f"Workspace toolkit family {row['identifier']!r} definition must be a mapping")
    raw = definition.get("tool_names")
    if not isinstance(raw, list) or not raw:
        raise TypeError(f"Workspace toolkit family {row['identifier']!r} tool_names must be a non-empty list")
    return tuple(named_key(name, "workspace MCP tool name") for name in raw)


def _load_family(name: str, resolver: ModuleResolver) -> WorkspaceToolkitFamilySpec:
    """Materialize one family from its authoritative record."""
    rows = resolver.inspect("toolkit_family", named_key(name, "workspace toolkit family name"))
    if not rows:
        raise KeyError(name)
    return _family_from_row(rows[0], resolver)


def _family_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> WorkspaceToolkitFamilySpec:
    """Materialize and cache one toolkit family inspection row."""

    def build(current_row: Mapping[str, Any]) -> WorkspaceToolkitFamilySpec:
        identifier = str(current_row["identifier"])
        raw = resolver.resolve("toolkit_family", identifier)
        if not isinstance(raw, Mapping):
            raise TypeError(f"Workspace toolkit family {identifier!r} definition must be a mapping")
        builder = raw.get("builder")
        if not isinstance(builder, Mapping):
            raise TypeError(f"Workspace toolkit family {identifier!r} builder must be an executable reference")
        meta = raw.get("meta") or {}
        if not isinstance(meta, Mapping):
            raise TypeError(f"Workspace toolkit family {identifier!r} meta must be a mapping")
        return WorkspaceToolkitFamilySpec(
            identifier,
            _family_tool_names(current_row),
            resolver.resolve_executable(builder, owner=current_row),
            desc=str(current_row["record"].get("desc") or ""),
            meta=dict(meta),
        )

    return _FAMILY_CACHE.get(resolver, row, build)
