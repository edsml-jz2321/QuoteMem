"""MCP profile registry, parsing, and resolution."""

from __future__ import annotations

__all__ = [
    "ProfileSpec",
    "TOOL_GROUPS",
    "available_profiles",
    "entity_allowed",
    "profile_spec_from_mapping",
    "register_profile",
    "registered_profiles",
    "registered_profile_tool_names",
    "resolve_profile",
    "resolve_profile_tools",
]

from heavenbase.utils import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple, raise_mismatch
from heavenbase.utils.registry import named_key

from heavenbase.extensions.contributions import eligible_contributions
from heavenbase.modules import ModuleResolver, ModuleService, module_resolver
from heavenbase.modules.base import ResolverCache
from .profile import EntitiesScope, ProfileSpec, SkillsScope
from .profile_tools import (
    PROFILE_SHAPE_HINT,
    TOOL_GROUPS,
    expand_profile_tool_refs,
    registered_profile_tool_names,
    validate_profile_tool_refs,
)
from .serializers import _serializer_names

_PROFILE_CACHE = ResolverCache[ProfileSpec]()


def _normalize_scope(values: Any, *, name: str) -> Tuple[str, ...] | Literal["*"]:
    if values in ("*", None):
        return "*"
    if isinstance(values, str):
        return (values,)
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, Mapping)):
        return tuple(str(item) for item in values)
    raise ValueError(f"MCP profile {name!r} field must be '*' or a list of names")


def profile_spec_from_mapping(name: str, raw: Any) -> ProfileSpec:
    """Parse one config or registry profile mapping into ``ProfileSpec``.

    Args:
        name: Profile name.
        raw: Mapping with ProfileSpec fields.

    Returns:
        Parsed profile specification.

    Raises:
        ValueError: If ``raw`` uses the removed bare tool-list shape or is invalid.
    """
    profile_name = str(name).strip()
    if isinstance(raw, (list, tuple)):
        raise ValueError(f"MCP profile {profile_name!r} uses the removed bare tool-list shape; {PROFILE_SHAPE_HINT}")
    if not isinstance(raw, Mapping):
        raise ValueError(f"MCP profile {profile_name!r} must be a mapping; {PROFILE_SHAPE_HINT}")
    if "extensions" in raw:
        raise ValueError(f"MCP profile {profile_name!r} uses removed field 'extensions'; use 'extends' for parent profiles. {PROFILE_SHAPE_HINT}")
    extends = tuple(str(item) for item in (raw.get("extends") or ()))
    tools = tuple(str(item) for item in (raw.get("tools") or ()))
    entities = _normalize_scope(raw.get("entities", "*"), name=profile_name)
    skills_raw = raw.get("skills")
    skills: SkillsScope = () if skills_raw in (None, ()) else _normalize_scope(skills_raw, name=profile_name)
    return ProfileSpec(
        name=profile_name,
        extends=extends,
        tools=tools,
        entities=entities,
        skills=skills,
        serializer=str(raw.get("serializer") or "jstr"),
        desc=str(raw.get("desc") or ""),
        meta=dict(raw.get("meta") or {}),
    )


def register_profile(
    spec: ProfileSpec,
    *,
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> None:
    """Register one MCP workspace profile.

    Args:
        spec: Profile specification to register.
        replace: Whether an existing profile with the same name may be replaced.
        resolver: Writable Context-owned module resolver. When omitted, the
            default Context is selected.

    Returns:
        None: This function does not return a value.

    Raises:
        ValueError: If the profile name is already registered and ``replace`` is false.
    """
    chosen = module_resolver(resolver)
    if not isinstance(chosen, ModuleService):
        raise TypeError("MCP profile registration requires a writable ModuleService")
    name = named_key(spec.name, "MCP profile name")
    validate_profile_tool_refs(spec.tools, resolver=chosen)
    normalized = ProfileSpec(
        name=name,
        extends=tuple(spec.extends),
        tools=tuple(spec.tools),
        entities=spec.entities,
        skills=spec.skills,
        serializer=spec.serializer,
        desc=spec.desc,
        meta=dict(spec.meta),
    )
    service = chosen
    found = service.inspect("mcp_profile", name)
    if found:
        existing = _profile_from_row(found[0], service)
        if existing == normalized and replace:
            return
        if not replace:
            raise ValueError(f"MCP profile {name!r} is already registered")
        coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if coordinate != f"programmatic/mcp_profile/{name}":
            raise ValueError(f"MCP profile {name!r} belongs to {coordinate!r} and cannot be replaced")
    service.register(
        "mcp_profile",
        name,
        {
            "name": name,
            "desc": normalized.desc,
            "source": "inline",
            "target": "definition",
            "active": True,
            "meta": {
                "schema_version": 1,
                "tags": {"toolkit": True, "mcp": True},
                "definition": _profile_mapping(normalized),
            },
        },
    )


def registered_profiles(
    *,
    resolver: ModuleResolver | None = None,
    workspace: Any | None = None,
) -> Dict[str, ProfileSpec]:
    """Return installed or workspace-eligible profiles keyed by name.

    Args:
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the default Context is selected.
        workspace (Any | None): Optional workspace whose active extensions constrain Registry contributions.

    Returns:
        Dict[str, ProfileSpec]: Profile specifications keyed by identifier.
    """
    chosen = module_resolver(resolver)
    rows = eligible_contributions(chosen.inspect("mcp_profile"), workspace)
    return {str(row["identifier"]): _profile_from_row(row, chosen) for row in rows}


def available_profiles(
    extra: Optional[Mapping[str, ProfileSpec]] = None,
    *,
    resolver: ModuleResolver | None = None,
    config: Any | None = None,
    workspace: Any | None = None,
) -> Dict[str, ProfileSpec]:
    """Return registered, config-backed, and optional profiles keyed by name.

    Args:
        extra: Optional workspace-local profile overlay.
        resolver: Context-owned module resolver. When omitted, the process
            default is selected.
        config: Context-owned config authority. When omitted, the config of
            the Context that owns `resolver` is used, or the process default
            when both are omitted.
        workspace: Optional workspace whose active extensions constrain
            Registry contributions.

    Returns:
        Dict[str, ProfileSpec]: Available profile specs keyed by identifier.
    """
    if config is None:
        from heavenbase.context.authority import omitted_config

        config = omitted_config(None if resolver is None else module_resolver(resolver))
    profiles = registered_profiles(resolver=resolver, workspace=workspace)
    skipped = set(profiles)
    if extra:
        skipped.update(str(name).strip() for name in extra)
    for name, spec in _configured_profiles(skip=skipped, strict=False, config=config).items():
        profiles.setdefault(name, spec)
    if extra:
        profiles.update(dict(extra))
    return profiles


def _configured_profile_mappings(config: Any) -> Dict[str, Any]:
    manager = config
    default_mcp = manager.load_default().get("heavenbase", {}).get("mcp", {}) or {}
    raw_profiles = dict(default_mcp.get("profiles", {}) or {})
    raw_profiles.update(manager.get("heavenbase.mcp.profiles", default={}) or {})
    return {str(name).strip(): raw for name, raw in raw_profiles.items() if str(name).strip()}


def _configured_profiles(
    *,
    skip: Optional[Iterable[str]] = None,
    strict: bool = True,
    config: Any,
) -> Dict[str, ProfileSpec]:
    skipped = {str(name).strip() for name in (skip or ())}
    profiles: Dict[str, ProfileSpec] = {}
    for name, raw in _configured_profile_mappings(config).items():
        if name in skipped:
            continue
        try:
            profiles[name] = profile_spec_from_mapping(name, raw)
        except (TypeError, ValueError):
            if strict:
                raise
    return profiles


def _lookup_profile(
    name: str,
    resolver: ModuleResolver,
    *,
    profile_specs: Optional[Mapping[str, ProfileSpec]] = None,
    config: Any,
    workspace: Any | None = None,
) -> ProfileSpec:
    key = str(name).strip()
    if profile_specs and key in profile_specs:
        return profile_specs[key]
    profiles = registered_profiles(resolver=resolver, workspace=workspace)
    if key in profiles:
        return profiles[key]
    configured = _configured_profile_mappings(config)
    if key in configured:
        return profile_spec_from_mapping(key, configured[key])
    profiles.update(available_profiles(profile_specs, resolver=resolver, config=config, workspace=workspace))
    if key in profiles:
        return profiles[key]
    raise_mismatch(
        set(profiles),
        key,
        name="MCP workspace profile",
        comment=PROFILE_SHAPE_HINT,
    )


def _flatten_extends(
    name: str,
    resolver: ModuleResolver,
    *,
    profile_specs: Optional[Mapping[str, ProfileSpec]] = None,
    visiting: Optional[set[str]] = None,
    config: Any,
    workspace: Any | None = None,
) -> List[ProfileSpec]:
    path = set(visiting or ())
    if name in path:
        raise ValueError(f"MCP profile extends cycle detected at {name!r}")
    spec = _lookup_profile(name, resolver, profile_specs=profile_specs, config=config, workspace=workspace)
    chain: List[ProfileSpec] = []
    path.add(name)
    for parent in spec.extends:
        chain.extend(
            _flatten_extends(
                parent,
                resolver,
                profile_specs=profile_specs,
                visiting=path,
                config=config,
                workspace=workspace,
            )
        )
    chain.append(spec)
    return chain


def _merge_profiles(chain: Iterable[ProfileSpec]) -> ProfileSpec:
    merged_tools: List[str] = []
    entities: EntitiesScope = "*"
    skills: SkillsScope = ()
    serializer = "jstr"
    desc = ""
    meta: Dict[str, Any] = {}
    name = ""
    for spec in chain:
        name = spec.name
        for tool in spec.tools:
            if tool not in merged_tools:
                merged_tools.append(tool)
        if spec.entities != "*":
            entities = spec.entities
        if spec.skills:
            skills = spec.skills
        if spec.serializer:
            serializer = spec.serializer
        if spec.desc:
            desc = spec.desc
        meta.update(dict(spec.meta))
    return ProfileSpec(name=name, tools=tuple(merged_tools), entities=entities, skills=skills, serializer=serializer, desc=desc, meta=meta)


def entity_allowed(entity_id: Optional[str], allowed: EntitiesScope) -> bool:
    """Return whether one entity id is visible under a profile entity scope.

    Args:
        entity_id (Optional[str]): Entity identifier to test.
        allowed (EntitiesScope): Closed entity allowlist or ``"*"`` for unrestricted visibility.

    Returns:
        bool: ``True`` when ``entity_id`` is permitted under ``allowed``.
    """
    if allowed == "*":
        return True
    if entity_id is None:
        return False
    return entity_id in allowed


def resolve_profile(
    name: str,
    *,
    profile_specs: Optional[Mapping[str, ProfileSpec]] = None,
    serializer_names: Optional[Iterable[str]] = None,
    resolver: ModuleResolver | None = None,
    config: Any | None = None,
    workspace: Any | None = None,
) -> ProfileSpec:
    """Resolve and flatten one profile by name.

    Args:
        name: Registered or configured profile name.
        profile_specs: Optional workspace-local profile specs that override process/config profiles.
        serializer_names: Optional workspace-local serializer names accepted by this profile build.
        resolver: Context-owned module resolver. When omitted, the default
            Context is selected.
        config: Context-owned config authority used for configured profile
            overlays. When omitted, the config of the Context that owns
            `resolver` is used, or the default Context's config when both are
            omitted.
        workspace: Optional workspace whose active extensions constrain
            Registry contributions.

    Returns:
        Flattened profile with ``extends`` cleared and tool groups expanded.

    Raises:
        ValueError: If profile inheritance contains a cycle or references are unknown.
    """
    from heavenbase.context.authority import omitted_config

    chosen = module_resolver(resolver)
    manager = omitted_config(chosen) if config is None else config
    chain = _flatten_extends(name, chosen, profile_specs=profile_specs, config=manager, workspace=workspace)
    merged = _merge_profiles(chain)
    expanded_tools = tuple(expand_profile_tool_refs(merged.tools, resolver=chosen))
    # Membership by name: loading every serializer here would let one unrelated serializer whose source is gone,
    # such as a Capsule a drop removed, fail every profile resolution.
    known_serializers = set(_serializer_names(chosen, workspace=workspace))
    if serializer_names is not None:
        known_serializers.update(str(name) for name in serializer_names)
    if merged.serializer not in known_serializers:
        raise_mismatch(
            known_serializers,
            merged.serializer,
            name="MCP serializer",
            comment="Built-in serializers: jstr, markdown.",
        )
    return ProfileSpec(
        name=merged.name,
        tools=expanded_tools,
        entities=merged.entities,
        skills=merged.skills,
        serializer=merged.serializer,
        desc=merged.desc,
        meta=merged.meta,
    )


def resolve_profile_tools(
    available: Iterable[str],
    *,
    profile: Optional[str] = None,
    tools: Optional[Iterable[str]] = None,
    profile_specs: Optional[Mapping[str, ProfileSpec]] = None,
    serializer_names: Optional[Iterable[str]] = None,
    resolver: ModuleResolver | None = None,
    config: Any | None = None,
    workspace: Any | None = None,
) -> tuple[ProfileSpec, List[str]]:
    """Resolve profile tools against an available tool surface.

    Args:
        available: Tool names supported by the current workspace toolkit builder.
        profile: Optional configured MCP profile name.
        tools: Optional explicit allowlist of tool names that overrides the profile.
        profile_specs: Optional workspace-local profile specs that override process/config profiles.
        serializer_names: Optional workspace-local serializer names accepted by this toolkit build.
        resolver: Context-owned module resolver. When omitted, the default
            Context is selected.
        config: Context-owned config authority used for configured profile
            selection and overlays. When omitted, the config of the Context
            that owns `resolver` is used, or the default Context's config when
            both are omitted.
        workspace: Optional workspace whose active extensions constrain
            Registry contributions.

    Returns:
        Tuple of resolved profile spec and selected tool names in declaration order.

    Raises:
        ValueError: If the profile, tool group, serializer, or requested tool name is unknown.
    """
    from heavenbase.context.authority import omitted_config

    chosen = module_resolver(resolver)
    manager = omitted_config(chosen) if config is None else config
    names = list(available)
    known = set(names)
    if tools is not None:
        selected = list(tools)
        unknown = [tool_name for tool_name in selected if tool_name not in known]
        if unknown:
            raise_mismatch(names, unknown[0], name="MCP workspace tool")
        default_mcp = manager.load_default().get("heavenbase", {}).get("mcp", {}) or {}
        profile_name = str(profile or manager.get("heavenbase.mcp.workspace_profile", default=default_mcp.get("workspace_profile", "full")) or "full")
        spec = resolve_profile(
            profile_name,
            profile_specs=profile_specs,
            serializer_names=serializer_names,
            resolver=chosen,
            config=manager,
            workspace=workspace,
        )
        return spec, [tool_name for tool_name in names if tool_name in set(selected)]
    default_mcp = manager.load_default().get("heavenbase", {}).get("mcp", {}) or {}
    profile_name = str(profile or manager.get("heavenbase.mcp.workspace_profile", default=default_mcp.get("workspace_profile", "full")) or "full")
    spec = resolve_profile(
        profile_name,
        profile_specs=profile_specs,
        serializer_names=serializer_names,
        resolver=chosen,
        config=manager,
        workspace=workspace,
    )
    expanded = expand_profile_tool_refs(spec.tools, available=known, resolver=chosen)
    return spec, [tool_name for tool_name in names if tool_name in set(expanded)]


def _profile_mapping(spec: ProfileSpec) -> dict[str, Any]:
    """Return the JSON-safe definition for one ProfileSpec."""
    return {
        "extends": list(spec.extends),
        "tools": list(spec.tools),
        "entities": spec.entities if spec.entities == "*" else list(spec.entities),
        "skills": spec.skills if spec.skills == "*" else list(spec.skills),
        "serializer": spec.serializer,
        "desc": spec.desc,
        "meta": dict(spec.meta),
    }


def _profile_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> ProfileSpec:
    """Materialize and cache one MCP profile inspection row."""

    def build(current_row: Mapping[str, Any]) -> ProfileSpec:
        identifier = str(current_row["identifier"])
        restored = resolver.resolve("mcp_profile", identifier)
        dynamic = callable(restored) and not isinstance(restored, ProfileSpec)
        if dynamic:
            restored = restored()
        if isinstance(restored, ProfileSpec):
            spec = restored
        elif isinstance(restored, Mapping):
            spec = profile_spec_from_mapping(identifier, restored)
        else:
            raise TypeError(f"MCP profile {identifier!r} did not resolve to ProfileSpec or a mapping")
        if named_key(spec.name, "MCP profile name") != identifier:
            raise ValueError(f"MCP profile record {identifier!r} resolved profile {spec.name!r}")
        validate_profile_tool_refs(spec.tools, resolver=resolver)
        return spec

    return _PROFILE_CACHE.get(
        resolver,
        row,
        build,
        should_cache=lambda current_row: current_row["record"].get("source") == "inline",
    )
