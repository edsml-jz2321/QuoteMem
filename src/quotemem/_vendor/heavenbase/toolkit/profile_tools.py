"""Accepted workspace MCP tool names and groups."""

from __future__ import annotations

__all__ = [
    "PROFILE_SHAPE_HINT",
    "TOOL_GROUPS",
    "expand_profile_tool_refs",
    "registered_profile_tool_names",
    "validate_profile_tool_refs",
]

from heavenbase.utils import Dict, Iterable, List, Optional, Tuple, raise_mismatch

from heavenbase.modules import ModuleResolver, module_resolver

TOOL_GROUPS: Dict[str, Tuple[str, ...]] = {
    "schema": ("define_entity", "list_entities", "describe_entity"),
    "read": ("get", "get_many", "exists", "exists_many", "count"),
    "write": ("upsert", "upsert_many", "set", "set_many"),
    "delete": ("delete", "delete_many"),
    "query": ("query", "explain"),
}

PROFILE_SHAPE_HINT = (
    "Use a ProfileSpec object with 'extends', 'tools' (tool names or groups: "
    "schema, read, write, delete, query), 'entities', 'serializer', and optional 'desc'."
)


def registered_profile_tool_names(*, resolver: ModuleResolver | None = None) -> tuple[str, ...]:
    """Return concrete tool names accepted by registered toolkit families.

    Args:
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the default Context is selected.

    Returns:
        tuple[str, ...]: Sorted union of protocol tool groups and registered family tool names.
    """
    from .families.registry import registered_family_tool_names

    protocol_names = {name for group in TOOL_GROUPS.values() for name in group}
    return tuple(sorted(protocol_names | set(registered_family_tool_names(resolver=resolver))))


def validate_profile_tool_refs(
    tool_refs: Iterable[str],
    *,
    resolver: ModuleResolver | None = None,
) -> None:
    """Validate MCP profile tool names and group references.

    Args:
        tool_refs: Tool names or group references from a profile.
        resolver: Context-owned module resolver. When omitted, the default
            Context is selected.

    Returns:
        None: This function does not return a value.

    Raises:
        ValueError: If a reference names no group or registered family tool.
    """
    chosen = module_resolver(resolver)
    known_tools = set(registered_profile_tool_names(resolver=chosen))
    for ref in tool_refs:
        if ref in TOOL_GROUPS or ref in known_tools:
            continue
        raise_mismatch(
            set(TOOL_GROUPS) | known_tools,
            ref,
            name="MCP profile tool or tool group",
            comment=PROFILE_SHAPE_HINT,
        )


def expand_profile_tool_refs(
    tool_refs: Iterable[str],
    *,
    available: Optional[Iterable[str]] = None,
    resolver: ModuleResolver | None = None,
) -> List[str]:
    """Expand MCP profile groups into ordered concrete tool names.

    Args:
        tool_refs: Tool names and group references.
        available: Optional current workspace tool surface.
        resolver: Context-owned module resolver. When omitted, the default
            Context is selected.

    Returns:
        list[str]: Expanded unique tool names in declaration order.

    Raises:
        ValueError: If a reference is unknown or unavailable.
    """
    chosen = module_resolver(resolver)
    validate_profile_tool_refs(tool_refs, resolver=chosen)
    expanded: List[str] = []
    for ref in tool_refs:
        if ref in TOOL_GROUPS:
            for tool_name in TOOL_GROUPS[ref]:
                if tool_name not in expanded:
                    expanded.append(tool_name)
            continue
        if ref not in expanded:
            expanded.append(ref)
    if available is not None:
        known = set(available)
        for ref in tool_refs:
            if ref not in TOOL_GROUPS and ref not in known:
                raise_mismatch(set(TOOL_GROUPS) | known, ref, name="MCP workspace tool or group", comment=PROFILE_SHAPE_HINT)
        missing = [tool_name for tool_name in expanded if tool_name not in known]
        if missing:
            raise_mismatch(known, missing[0], name="MCP workspace tool", comment=PROFILE_SHAPE_HINT)
    return expanded
