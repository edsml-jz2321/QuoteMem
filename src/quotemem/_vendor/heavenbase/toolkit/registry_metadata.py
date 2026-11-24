"""Mirror runtime MCP registry state into workspace system metadata."""

from __future__ import annotations

__all__ = [
    "resolve_workspace_mcp_profile",
    "sync_mcp_registry_metadata",
    "sync_workspace_mcp_registry_metadata",
    "workspace_mcp_toolkit_families",
    "workspace_mcp_profile_specs",
    "workspace_mcp_serializers",
]

from heavenbase.utils import Any, Dict, List, Mapping, Optional, deepcopy

from heavenbase.modules import ModuleResolver, module_resolver
from .families.family import WorkspaceToolkitFamilySpec
from .families.registry import registered_workspace_toolkit_families
from .profile import ProfileSpec
from .profiles import available_profiles, profile_spec_from_mapping, registered_profiles, resolve_profile
from .serializers import SerializerFn, ToolSerializer, _available_serializers, _serializer_names

_METASCHEMA_ROWS_CACHE_ATTR = "_mcp_registry_metaschema_rows_cache"


def sync_workspace_mcp_registry_metadata(
    workspace: Any,
    *,
    load_workspace_rows: bool = True,
    resolver: ModuleResolver | None = None,
) -> Dict[str, int]:
    """Publish eligible Registry definitions and optional workspace overlays.

    Args:
        workspace (Any): HeavenBase workspace whose MCP metadata rows should be synchronized.
        load_workspace_rows (bool): Whether workspace-owned profile, serializer, and family overlays are loaded first. Defaults to ``True``.
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the workspace resolver is selected.

    Returns:
        Dict[str, int]: Counts of published profile, toolkit-family, and serializer rows.
    """
    chosen = module_resolver(resolver if resolver is not None else workspace.module_resolver)
    if load_workspace_rows:
        profile_specs = workspace_mcp_profile_specs(workspace, resolver=chosen)
        serializers = workspace_mcp_serializers(workspace, resolver=chosen)
        toolkit_families = workspace_mcp_toolkit_families(workspace, resolver=chosen)
    else:
        profile_specs = {}
        serializers = {}
        toolkit_families = {}
    return sync_mcp_registry_metadata(
        workspace,
        profile_specs=profile_specs,
        serializers=serializers,
        toolkit_family_specs=toolkit_families,
        resolver=chosen,
    )


def workspace_mcp_profile_specs(
    workspace: Any,
    *,
    resolver: ModuleResolver | None = None,
) -> Dict[str, ProfileSpec]:
    """Load workspace-owned MCP profile specs from active ``sys-metaschema`` rows.

    Args:
        workspace (Any): HeavenBase workspace whose metaschema rows are read.
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the workspace resolver is selected.

    Returns:
        Dict[str, ProfileSpec]: Workspace-owned profile specs keyed by name.
    """
    chosen = module_resolver(resolver if resolver is not None else workspace.module_resolver)
    lock = getattr(workspace, "_lock", None)
    if lock is None:
        return _workspace_mcp_profile_specs_unlocked(workspace, chosen)
    with lock:
        return _workspace_mcp_profile_specs_unlocked(workspace, chosen)


def workspace_mcp_serializers(
    workspace: Any,
    *,
    resolver: ModuleResolver | None = None,
) -> Dict[str, SerializerFn]:
    """Load workspace-owned MCP serializer aliases from active ``sys-metaschema`` rows.

    Workspace serializer rows may alias already registered serializer names or callable
    references. They do not import arbitrary callables from row data.

    Args:
        workspace (Any): HeavenBase workspace whose metaschema rows are read.
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the workspace resolver is selected.

    Returns:
        Dict[str, SerializerFn]: Workspace serializer aliases keyed by name.

    Raises:
        ValueError: If a row aliases an unknown serializer or replaces a registered name.
    """
    chosen = module_resolver(resolver if resolver is not None else workspace.module_resolver)
    lock = getattr(workspace, "_lock", None)
    if lock is None:
        return _workspace_mcp_serializers_unlocked(workspace, chosen)
    with lock:
        return _workspace_mcp_serializers_unlocked(workspace, chosen)


def resolve_workspace_mcp_profile(
    workspace: Any,
    profile: str,
    *,
    resolver: ModuleResolver | None = None,
) -> ProfileSpec:
    """Resolve one MCP profile with the workspace's own profile and serializer rows.

    Args:
        workspace (Any): HeavenBase workspace whose metaschema rows and config
            constrain the profile.
        profile (str): Registered, configured, or workspace-owned profile name.
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the workspace resolver is selected.

    Returns:
        ProfileSpec: Flattened profile with tool groups expanded.

    Raises:
        ValueError: If the profile, its inheritance, or its serializer is unknown.
    """
    chosen = module_resolver(resolver if resolver is not None else workspace.module_resolver)
    return resolve_profile(
        profile,
        profile_specs=workspace_mcp_profile_specs(workspace, resolver=chosen),
        serializer_names=workspace_mcp_serializers(workspace, resolver=chosen),
        resolver=chosen,
        config=workspace.context.config,
        workspace=workspace,
    )


def workspace_mcp_toolkit_families(
    workspace: Any,
    *,
    resolver: ModuleResolver | None = None,
) -> Dict[str, WorkspaceToolkitFamilySpec]:
    """Load workspace-owned toolkit family aliases from active ``sys-metaschema`` rows.

    Workspace toolkit-family rows may alias already registered families by family
    name or registered builder callable reference. They do not import arbitrary
    callables from row data.

    Args:
        workspace (Any): HeavenBase workspace whose metaschema rows are read.
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the workspace resolver is selected.

    Returns:
        Dict[str, WorkspaceToolkitFamilySpec]: Workspace family aliases keyed by name.

    Raises:
        ValueError: If a row references unknown tools or registered families incorrectly.
    """
    chosen = module_resolver(resolver if resolver is not None else workspace.module_resolver)
    lock = getattr(workspace, "_lock", None)
    if lock is None:
        return _workspace_mcp_toolkit_families_unlocked(workspace, chosen)
    with lock:
        return _workspace_mcp_toolkit_families_unlocked(workspace, chosen)


def sync_mcp_registry_metadata(
    workspace: Any,
    *,
    profile_specs: Optional[Mapping[str, ProfileSpec]] = None,
    serializers: Optional[Mapping[str, SerializerFn]] = None,
    toolkit_family_specs: Optional[Mapping[str, WorkspaceToolkitFamilySpec]] = None,
    resolver: ModuleResolver | None = None,
) -> Dict[str, int]:
    """Publish MCP profile, toolkit-family, and serializer registry rows.

    Ordinary registered definitions are projected directly by MetaSchema.
    This helper publishes only workspace-owned aliases and configured overlays.

    Args:
        workspace (Any): HeavenBase workspace whose system metadata rows are written.
        profile_specs (Optional[Mapping[str, ProfileSpec]]): Optional workspace profile overlay. Defaults to ``None``.
        serializers (Optional[Mapping[str, SerializerFn]]): Optional workspace serializer overlay. Defaults to ``None``.
        toolkit_family_specs (Optional[Mapping[str, WorkspaceToolkitFamilySpec]]): Optional workspace family overlay. Defaults to ``None``.
        resolver (ModuleResolver | None): Context-owned module resolver. When omitted, the workspace resolver is selected.

    Returns:
        Dict[str, int]: Counts of published profile, toolkit-family, and serializer rows.
    """
    chosen = module_resolver(resolver if resolver is not None else workspace.module_resolver)
    lock = getattr(workspace, "_lock", None)
    if lock is None:
        return _sync_mcp_registry_metadata_unlocked(
            workspace,
            chosen,
            profile_specs=profile_specs,
            serializers=serializers,
            toolkit_family_specs=toolkit_family_specs,
        )
    with lock:
        return _sync_mcp_registry_metadata_unlocked(
            workspace,
            chosen,
            profile_specs=profile_specs,
            serializers=serializers,
            toolkit_family_specs=toolkit_family_specs,
        )


def _sync_mcp_registry_metadata_unlocked(
    workspace: Any,
    resolver: ModuleResolver,
    *,
    profile_specs: Optional[Mapping[str, ProfileSpec]] = None,
    serializers: Optional[Mapping[str, SerializerFn]] = None,
    toolkit_family_specs: Optional[Mapping[str, WorkspaceToolkitFamilySpec]] = None,
) -> Dict[str, int]:
    config = workspace.context.config
    publisher = getattr(workspace, "_metaschema", None) or getattr(workspace, "_catalog", None)
    build = getattr(publisher, "build_system_row", None)
    write = getattr(publisher, "write_meta_rows", None)
    if not callable(build) or not callable(write):
        return {"profiles": 0, "toolkit_families": 0, "serializers": 0}

    rows: List[dict] = []
    rows.extend(
        _profile_rows(
            publisher,
            resolver,
            workspace=workspace,
            profile_specs=profile_specs,
            serializers=serializers,
            config=config,
        )
    )
    rows.extend(_toolkit_family_rows(publisher, toolkit_family_specs=toolkit_family_specs))
    rows.extend(_serializer_rows(publisher, resolver, workspace=workspace, serializers=serializers))
    write(rows, invalidate_mcp_registry_cache=False)
    return {
        "profiles": len(_profile_specs(resolver, profile_specs, config=config, workspace=workspace)),
        "toolkit_families": len(_toolkit_family_specs(toolkit_family_specs)),
        "serializers": len(serializers or {}),
    }


def _workspace_mcp_profile_specs_unlocked(
    workspace: Any,
    resolver: ModuleResolver,
) -> Dict[str, ProfileSpec]:
    specs: Dict[str, ProfileSpec] = {}
    for row in _metaschema_rows(workspace):
        if row.get("kind") != "mcp_profile" or row.get("active") is False:
            continue
        raw = row.get("meta")
        if not isinstance(raw, Mapping):
            continue
        if raw.get("record_kind") or raw.get("source") in {"registry", "config"}:
            continue
        name = str(raw.get("name") or row.get("name") or row.get("subject_id") or "").strip()
        if not name:
            continue
        try:
            specs[name] = profile_spec_from_mapping(name, raw)
        except Exception as exc:
            row_id = row.get("object_id", name)
            raise ValueError(f"Invalid MCP profile metadata row {row_id!r}: {exc}") from exc
    return specs


def _workspace_mcp_serializers_unlocked(
    workspace: Any,
    resolver: ModuleResolver,
) -> Dict[str, SerializerFn]:
    serializers: Dict[str, SerializerFn] = {}
    # Registered serializers are checked by name and loaded only when a row needs one, so one unrelated
    # serializer whose source is gone, such as a Capsule a drop removed, does not fail every workspace.
    names = set(_serializer_names(resolver, workspace=workspace))
    registered_refs: Optional[Dict[str, str]] = None
    for row in _metaschema_rows(workspace):
        if row.get("kind") != "mcp_serializer" or row.get("active") is False:
            continue
        raw = row.get("meta") if isinstance(row.get("meta"), Mapping) else {}
        if raw.get("record_kind") or raw.get("source") in {"registry", "config"}:
            continue
        name = str(raw.get("name") or row.get("name") or row.get("subject_id") or "").strip()
        if not name:
            continue
        if name in names:
            row_id = row.get("object_id", name)
            raise ValueError(f"Invalid MCP serializer metadata row {row_id!r}: workspace serializer aliases cannot replace registered serializer {name!r}")
        target = str(raw.get("serializer") or raw.get("alias_of") or raw.get("callable") or row.get("target") or "").strip()
        if not target:
            row_id = row.get("object_id", name)
            raise ValueError(f"Invalid MCP serializer metadata row {row_id!r}: missing serializer target")
        if target in names:
            base_name: Optional[str] = target
        else:
            if registered_refs is None:
                registered = _available_serializers(resolver, workspace=workspace)
                registered_refs = {ToolSerializer(key, fn).to_ref(): key for key, fn in registered.items()}
            base_name = registered_refs.get(target)
        if base_name is None:
            row_id = row.get("object_id", name)
            raise ValueError(f"Invalid MCP serializer metadata row {row_id!r}: unknown registered serializer target {target!r}")
        serializers[name] = ToolSerializer.load(base_name, resolver=resolver, workspace=workspace).serialize
    return serializers


def _workspace_mcp_toolkit_families_unlocked(
    workspace: Any,
    resolver: ModuleResolver,
) -> Dict[str, WorkspaceToolkitFamilySpec]:
    families: Dict[str, WorkspaceToolkitFamilySpec] = {}
    registered = {
        name: {
            "name": spec.name,
            "tool_names": spec.tool_names,
            "builder": spec.builder,
            "desc": spec.desc,
            "meta": dict(spec.meta),
        }
        for name, spec in registered_workspace_toolkit_families(resolver=resolver, workspace=workspace).items()
    }
    registered_refs = {_callable_ref(spec["builder"]): name for name, spec in registered.items()}
    registered_tools = _tool_names_for(registered.values())
    for row in _metaschema_rows(workspace):
        if row.get("kind") != "toolkit_family" or row.get("active") is False:
            continue
        raw = row.get("meta") if isinstance(row.get("meta"), Mapping) else {}
        if raw.get("record_kind") or raw.get("source") in {"registry", "config"}:
            continue
        name = str(raw.get("name") or row.get("name") or row.get("subject_id") or "").strip()
        if not name:
            continue
        row_id = row.get("object_id", name)
        if name in registered:
            raise ValueError(f"Invalid MCP toolkit_family metadata row {row_id!r}: workspace family aliases cannot replace {name!r}")
        row_target = str(row.get("target") or "").strip()
        metadata_only = bool(raw.get("metadata_only")) or row_target.endswith("_metadata_only_tool_functions")
        builder_target = "" if metadata_only else raw.get("builder")
        target = str(raw.get("alias_of") or raw.get("family") or builder_target or ("" if metadata_only else row_target) or "").strip()
        if target:
            base_name = target if target in registered else registered_refs.get(target)
            if base_name is None:
                raise ValueError(f"Invalid MCP toolkit_family metadata row {row_id!r}: unknown registered toolkit family target {target!r}")
            base = registered[base_name]
            base_tools = tuple(base["tool_names"])
            tool_names = _string_tuple(raw.get("tool_names") or base_tools)
            missing = [tool_name for tool_name in tool_names if tool_name not in base_tools]
            if missing:
                raise ValueError(f"Invalid MCP toolkit_family metadata row {row_id!r}: tool {missing[0]!r} is not provided by registered family {base_name!r}")
            builder = base["builder"]
            alias_of = base_name
            meta = dict(base.get("meta") or {})
        else:
            tool_names = _string_tuple(raw.get("tool_names"))
            missing = [tool_name for tool_name in tool_names if tool_name not in registered_tools]
            if missing:
                raise ValueError(f"Invalid MCP toolkit_family metadata row {row_id!r}: unknown workspace tool {missing[0]!r}")
            builder = _metadata_only_tool_functions
            alias_of = None
            meta = {}
        if not tool_names or any(not tool_name for tool_name in tool_names):
            raise ValueError(f"Invalid MCP toolkit_family metadata row {row_id!r}: tool_names must be non-empty")
        meta.update(dict(raw.get("meta") or {}))
        meta.update({"source": "workspace", "alias_of": alias_of, "runtime_only": False, "metadata_only": alias_of is None})
        desc = str(raw.get("desc") or row.get("desc") or (registered.get(alias_of, {}).get("desc") if alias_of else "") or "")
        families[name] = WorkspaceToolkitFamilySpec(name, tool_names, builder, desc=desc, meta=meta)
    return families


def _metadata_only_tool_functions(_workspace: Any) -> Mapping[str, Any]:
    """Return no tool functions for workspace-owned metadata-only families."""
    return {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item).strip() for item in value)


def _tool_names_for(specs: Any) -> set[str]:
    names: set[str] = set()
    for spec in specs:
        names.update(str(tool_name) for tool_name in spec["tool_names"])
    return names


def _metaschema_rows(workspace: Any) -> List[dict]:
    version = int(getattr(getattr(workspace, "_metaschema", None), "revision", 0) or 0)
    cached = getattr(workspace, _METASCHEMA_ROWS_CACHE_ATTR, None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == version:
        return deepcopy(cached[1])

    rows = _read_metaschema_rows(workspace)
    setattr(workspace, _METASCHEMA_ROWS_CACHE_ATTR, (version, deepcopy(rows)))
    return rows


def _read_metaschema_rows(workspace: Any) -> List[dict]:
    publisher = getattr(workspace, "_metaschema", None) or getattr(workspace, "_catalog", None)
    system_backend_name = getattr(publisher, "system_backend_name", None)
    row_codec = getattr(workspace, "_row_codec", None)
    backends = getattr(workspace, "backends", None)
    if not callable(system_backend_name) or backends is None:
        return []
    try:
        from heavenbase.workspace.metaschema import MetaSchema

        entity = MetaSchema
        schema = entity.schema()
        entity_id = entity.schema().entity_id
        backend_name = system_backend_name(entity)
        backend = backends.get(backend_name)
        decode = getattr(row_codec, "decode_row_items", None)
        if not callable(decode):
            return [dict(row) for row in backend.scan(entity_id)]
        return [decode(schema, backend_name, dict(row)) for row in backend.scan(entity_id)]
    except (KeyError, NotImplementedError):
        return []


def _profile_specs(
    resolver: ModuleResolver,
    profile_specs: Optional[Mapping[str, ProfileSpec]] = None,
    *,
    config: Any | None = None,
    workspace: Any | None = None,
) -> Dict[str, ProfileSpec]:
    profiles = available_profiles(profile_specs, resolver=resolver, config=config, workspace=workspace)
    registered = set(registered_profiles(resolver=resolver, workspace=workspace))
    workspace_owned = set(profile_specs or {})
    return {name: spec for name, spec in profiles.items() if name not in registered or name in workspace_owned}


def _profile_rows(
    catalog: Any,
    resolver: ModuleResolver,
    *,
    workspace: Any | None = None,
    profile_specs: Optional[Mapping[str, ProfileSpec]] = None,
    serializers: Optional[Mapping[str, SerializerFn]] = None,
    config: Any | None = None,
) -> List[dict]:
    workspace_profiles = dict(profile_specs or {})
    rows = []
    for name, spec in sorted(_profile_specs(resolver, profile_specs, config=config, workspace=workspace).items()):
        if name in workspace_profiles:
            source = "workspace"
        else:
            source = "config"
        payload = _profile_payload(
            spec,
            resolver,
            source=source,
            profile_specs=profile_specs,
            serializers=serializers,
            config=config,
            workspace=workspace,
        )
        rows.append(
            catalog.build_system_row(
                "mcp_profile",
                name,
                name=name,
                desc=spec.desc,
                tags=["mcp", "profile"],
                type=spec.serializer,
                source=payload["source"],
                target=spec.serializer,
                meta=payload,
            )
        )
    return rows


def _profile_payload(
    spec: ProfileSpec,
    resolver: ModuleResolver,
    *,
    source: str,
    profile_specs: Optional[Mapping[str, ProfileSpec]] = None,
    serializers: Optional[Mapping[str, SerializerFn]] = None,
    config: Any | None = None,
    workspace: Any | None = None,
) -> dict:
    payload = {
        "name": spec.name,
        "source": source,
        "extends": list(spec.extends),
        "tools": list(spec.tools),
        "entities": _scope_payload(spec.entities),
        "skills": _scope_payload(spec.skills),
        "serializer": spec.serializer,
        "desc": spec.desc,
        "meta": dict(spec.meta),
    }
    try:
        resolved = resolve_profile(
            spec.name,
            profile_specs=profile_specs,
            serializer_names=(serializers or {}).keys(),
            resolver=resolver,
            config=config,
            workspace=workspace,
        )
    except Exception as exc:  # pragma: no cover - defensive for malformed runtime config
        payload["resolve_error"] = str(exc)
    else:
        payload["resolved_tools"] = list(resolved.tools)
        payload["resolved_entities"] = _scope_payload(resolved.entities)
        payload["resolved_skills"] = _scope_payload(resolved.skills)
    return payload


def _scope_payload(value: Any) -> Any:
    if value == "*":
        return "*"
    if value in (None, ""):
        return []
    return list(value)


def _toolkit_family_specs(toolkit_family_specs: Optional[Mapping[str, WorkspaceToolkitFamilySpec]] = None) -> Dict[str, Mapping[str, Any]]:
    specs: Dict[str, Mapping[str, Any]] = {}
    for name, spec in (toolkit_family_specs or {}).items():
        specs[name] = {
            "name": spec.name,
            "tool_names": spec.tool_names,
            "builder": spec.builder,
            "desc": spec.desc,
            "meta": dict(spec.meta),
        }
    return specs


def _toolkit_family_rows(catalog: Any, *, toolkit_family_specs: Optional[Mapping[str, WorkspaceToolkitFamilySpec]] = None) -> List[dict]:
    rows = []
    workspace_families = set(toolkit_family_specs or {})
    for name, spec in sorted(_toolkit_family_specs(toolkit_family_specs).items()):
        builder_ref = _callable_ref(spec["builder"])
        workspace_owned = name in workspace_families
        meta = dict(spec["meta"])
        payload = {
            "name": spec["name"],
            "source": "workspace" if workspace_owned else "registry",
            "tool_names": list(spec["tool_names"]),
            "builder": builder_ref,
            "runtime_only": not workspace_owned,
            "metadata_only": workspace_owned and spec["builder"] is _metadata_only_tool_functions,
            "desc": spec["desc"],
            "meta": meta,
        }
        if workspace_owned:
            payload["alias_of"] = meta.get("alias_of")
        rows.append(
            catalog.build_system_row(
                "toolkit_family",
                name,
                name=name,
                desc=str(spec["desc"] or ""),
                tags=["mcp", "toolkit", "family"],
                type="workspace",
                source="workspace" if workspace_owned else "registry",
                target=builder_ref,
                meta=payload,
            )
        )
    return rows


def _serializer_rows(
    catalog: Any,
    resolver: ModuleResolver,
    *,
    workspace: Any | None = None,
    serializers: Optional[Mapping[str, SerializerFn]] = None,
) -> List[dict]:
    rows = []
    # The registered callables are needed only to name what a workspace alias points at.
    registered = _available_serializers(resolver, workspace=workspace) if serializers else {}
    for name, fn in sorted((serializers or {}).items()):
        callable_ref = ToolSerializer(name, fn).to_ref()
        desc = _doc_summary(fn)
        alias_of = _registered_serializer_name(fn, registered)
        rows.append(
            catalog.build_system_row(
                "mcp_serializer",
                name,
                name=name,
                desc=desc,
                tags=["mcp", "serializer"],
                type="serializer",
                source="workspace",
                target=callable_ref,
                meta={
                    "name": name,
                    "source": "workspace",
                    "alias_of": alias_of,
                    "callable": callable_ref,
                    "runtime_only": False,
                    "desc": desc,
                },
            )
        )
    return rows


def _callable_ref(fn: Any) -> str:
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", ""))
    return f"{module}.{qualname}" if module and qualname else repr(fn)


def _registered_serializer_name(fn: SerializerFn, registered: Mapping[str, SerializerFn]) -> str:
    for name, registered_fn in registered.items():
        if registered_fn is fn:
            return name
    ref = _callable_ref(fn)
    for name, registered_fn in registered.items():
        if _callable_ref(registered_fn) == ref:
            return name
    return ref


def _doc_summary(fn: Any) -> str:
    doc = str(getattr(fn, "__doc__", "") or "").strip().splitlines()
    return doc[0].strip() if doc else ""
