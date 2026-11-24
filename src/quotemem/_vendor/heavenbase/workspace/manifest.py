"""Workspace construction and schema manifests."""

from __future__ import annotations

__all__ = [
    "WORKSPACE_MANIFEST_KIND",
    "WORKSPACE_MANIFEST_VERSION",
    "WorkspaceManifest",
]

import warnings

from heavenbase.utils import Any, Dict, Iterable, List, Mapping, Optional, dataclass, field
from heavenbase.utils import deepcopy, dump_json, dump_yaml, dumps_json, get_file_ext, load_json, load_yaml

from ..context import Context
from ..entity.compiler import _encoded_field
from ..utils.ids import check_workspace_id
from .setup.backends import (
    WorkspaceConstruction,
    backend_identifiers_from_config,
)

WORKSPACE_MANIFEST_KIND = "heavenbase.workspace.manifest"
WORKSPACE_MANIFEST_VERSION = 2


def _same_manifest_data(left: Any, right: Any) -> bool:
    """Compare definition data without equating JSON Booleans and numbers."""
    if left != right:
        return False
    try:
        return dumps_json(left, compact=True, sort_keys=True) == dumps_json(right, compact=True, sort_keys=True)
    except (TypeError, ValueError):
        # The live definition can contain values that the manifest cannot replay.
        return True


@dataclass(frozen=True)
class WorkspaceManifest:
    """Serializable workspace construction and schema manifest.

    The manifest captures one canonical workspace construction contract,
    enabled optional extensions, and user entity definitions. It intentionally
    does not export Backend row data.

    Args:
        id (str): Workspace identifier to construct.
        construction: Canonical ambient, preset, configured-Backend, or
            runtime-Backend reconstruction contract.
        entities (List[Dict[str, Any]]): JSON-compatible user entity definitions.
        extensions (List[str]): Optional extension identifiers to enable after
            workspace creation.
        backend_summary (List[Dict[str, Any]]): Non-authoritative backend
            metadata for inspection.
        version (int): Manifest schema version. Defaults to `2`.
        kind (str): Manifest kind marker. Defaults to
            `"heavenbase.workspace.manifest"`.
    """

    id: str
    construction: WorkspaceConstruction = field(default_factory=WorkspaceConstruction)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    extensions: List[str] = field(default_factory=list)
    backend_summary: List[Dict[str, Any]] = field(default_factory=list)
    version: int = WORKSPACE_MANIFEST_VERSION
    kind: str = WORKSPACE_MANIFEST_KIND

    def __post_init__(self) -> None:
        """Validate the manifest contract and isolate nested values.

        Args:
            None.

        Returns:
            None: This initializer replaces mutable fields with isolated copies.

        Raises:
            ValueError: If the workspace identifier, manifest kind, or manifest
                version is unsupported.
        """
        kind = str(self.kind)
        version = int(self.version)
        if kind != WORKSPACE_MANIFEST_KIND:
            raise ValueError(f"Unsupported workspace manifest kind {kind!r}")
        if version != WORKSPACE_MANIFEST_VERSION:
            raise ValueError(f"Unsupported workspace manifest version {version!r}")
        construction = self.construction if isinstance(self.construction, WorkspaceConstruction) else WorkspaceConstruction.from_dict(self.construction)
        object.__setattr__(self, "id", check_workspace_id(str(self.id)))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "construction", WorkspaceConstruction.from_dict(construction.to_dict()))
        object.__setattr__(self, "entities", [deepcopy(dict(item)) for item in self.entities])
        object.__setattr__(self, "extensions", _extension_ids(self.extensions))
        object.__setattr__(self, "backend_summary", [deepcopy(dict(item)) for item in self.backend_summary])

    @classmethod
    def from_workspace(cls, workspace: Any, *, include_system: bool = False) -> "WorkspaceManifest":
        """Build a manifest from a live workspace.

        Args:
            workspace (Any): Live `HeavenBase` workspace.
            include_system (bool): Whether to include extension-owned Entity
                schemas. Requested extension roots are unchanged. Defaults to
                `False`.

        Returns:
            WorkspaceManifest: Construction/schema manifest for ``workspace``.
        """
        return manifest_from_workspace(workspace, include_system=include_system)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkspaceManifest":
        """Build a manifest from JSON-compatible data.

        Args:
            data (Mapping[str, Any]): Manifest mapping produced by `to_dict()`.

        Returns:
            WorkspaceManifest: Parsed manifest.

        Raises:
            TypeError: If ``data`` is not a mapping.
            ValueError: If the manifest kind or version is unsupported.
        """
        if not isinstance(data, Mapping):
            raise TypeError("WorkspaceManifest.from_dict expects a mapping")
        kind = str(data["kind"]) if "kind" in data else WORKSPACE_MANIFEST_KIND
        legacy = "config" in data and "construction" not in data
        version = int(data["version"]) if "version" in data else (1 if legacy else WORKSPACE_MANIFEST_VERSION)
        if version not in {1, WORKSPACE_MANIFEST_VERSION}:
            raise ValueError(f"Unsupported workspace manifest version {version!r}")
        if version == 1:
            if "construction" in data:
                raise ValueError("Workspace manifest version 1 must use 'config', not 'construction'")
            construction = WorkspaceConstruction.from_config(data.get("config") or {})
        else:
            if "config" in data:
                raise ValueError("Workspace manifest version 2 must use 'construction', not 'config'")
            if "construction" not in data:
                raise ValueError("Workspace manifest version 2 requires 'construction'")
            construction = WorkspaceConstruction.from_dict(data["construction"])
        return cls(
            id=str(data.get("id") or ""),
            construction=construction,
            entities=[dict(item) for item in data.get("entities") or []],
            extensions=data.get("extensions") or [],
            backend_summary=[dict(item) for item in data.get("backend_summary") or []],
            version=WORKSPACE_MANIFEST_VERSION,
            kind=kind,
        )

    @classmethod
    def load(cls, path: str) -> "WorkspaceManifest":
        """Load a workspace manifest from a JSON or YAML file.

        Args:
            path (str): Path to a `.json`, `.yaml`, or `.yml` manifest file.
                JSON paths use JSON; all other extensions use YAML.

        Returns:
            WorkspaceManifest: Parsed manifest.
        """
        data = load_json(path, strict=True) if get_file_ext(path).lower() == ".json" else load_yaml(path, strict=True)
        return cls.from_dict(data or {})

    def save(self, path: str) -> None:
        """Save this manifest to a JSON or YAML file.

        Args:
            path (str): Destination path. `.json` writes JSON; all other
                extensions write YAML.

        Returns:
            None: This method writes the manifest file in place.
        """
        data = self.to_dict()
        if get_file_ext(path).lower() == ".json":
            dump_json(data, path)
        else:
            dump_yaml(data, path)

    def to_dict(self) -> Dict[str, Any]:
        """Return this manifest as isolated JSON-compatible data.

        Args:
            None.

        Returns:
            Dict[str, Any]: Deep-copied manifest payload suitable for JSON or
            YAML serialization.
        """
        return {
            "kind": self.kind,
            "version": self.version,
            "id": self.id,
            "construction": self.construction.to_dict(),
            "extensions": list(self.extensions),
            "entities": deepcopy(self.entities),
            "backend_summary": deepcopy(self.backend_summary),
        }

    def with_construction(self, construction: WorkspaceConstruction | Mapping[str, Any]) -> "WorkspaceManifest":
        """Return this manifest with one replacement construction contract.

        Backend replacement is whole-contract substitution; configurations are
        never partially merged into the exported construction authority.

        Args:
            construction (WorkspaceConstruction | Mapping[str, Any]):
                Replacement construction contract.

        Returns:
            WorkspaceManifest: Copy of this manifest with the new construction
            contract.
        """
        selected = construction if isinstance(construction, WorkspaceConstruction) else WorkspaceConstruction.from_dict(construction)
        return type(self)(
            id=self.id,
            construction=selected,
            entities=self.entities,
            extensions=self.extensions,
            backend_summary=self.backend_summary,
        )

    def with_extensions(self, extensions: Iterable[str]) -> "WorkspaceManifest":
        """Return this manifest with one replacement requested-root sequence.

        Args:
            extensions (Iterable[str]): Extension identifiers to enable after
                workspace creation.

        Returns:
            WorkspaceManifest: Copy of this manifest with the replacement
            extension list.
        """
        return type(self)(
            id=self.id,
            construction=self.construction,
            entities=self.entities,
            extensions=list(extensions),
            backend_summary=self.backend_summary,
        )

    def open(
        self,
        *,
        context: Any = None,
        backends: Optional[Mapping[str, Any]] = None,
        preset: Optional[str] = None,
        computes: Optional[Mapping[str, Any]] = None,
        query_computes: Optional[Mapping[str, Any]] = None,
        allow_replace: bool = False,
        detached: bool = False,
    ) -> Any:
        """Construct a live workspace from this manifest.

        Args:
            context: Context that owns construction authority. Defaults to the
                process Context.
            backends: Optional complete Backend config or live-instance
                replacement. Runtime replacements must match the recorded
                Backend name-to-identifier signature.
            preset: Optional complete preset replacement. Explicit Backends
                take precedence.
            computes (Mapping[str, Any] | None): Optional compute function
                registry used by entity JSON definitions. Defaults to `None`.
            query_computes (Mapping[str, Any] | None): Optional query-compute
                function registry. Defaults to `None`.
            allow_replace (bool): Whether entity registration may replace an
                existing schema with the same identifier. Defaults to `False`.
            detached: Whether the returned workspace is caller-owned and
                absent from durable and live Context lookup. Defaults to
                `False`.

        Returns:
            Any: Live ``HeavenBase`` workspace.
        """
        return open_manifest(
            self,
            context=context,
            backends=backends,
            preset=preset,
            computes=computes,
            query_computes=query_computes,
            allow_replace=allow_replace,
            detached=detached,
        )


def manifest_from_workspace(workspace: Any, *, include_system: bool = False) -> WorkspaceManifest:
    """Build a construction and schema manifest from a live workspace.

    Args:
        workspace (Any): Live `HeavenBase` workspace.
        include_system (bool): Whether to include extension-owned Entity
            schemas. Requested extension roots are always exported. Defaults
            to `False`.

    Returns:
        WorkspaceManifest: Serializable workspace shell without row data.
    """
    extension_service = workspace._extension_service
    from .foundation import FIXED_ENTITY_IDS

    # Extension activation registers each extension entity's only definition (changes to it are refused), so a
    # default manifest leaves those schemas to activation on import.
    excluded = FIXED_ENTITY_IDS if include_system else FIXED_ENTITY_IDS | extension_service.entity_ids()
    entities = [_entity_manifest(workspace, workspace.entities[entity_id]) for entity_id in sorted(workspace.entities.keys() - excluded)]
    # A manifest owns the requested activation roots, never the resolved
    # dependency closure. ``include_system`` controls schema detail only;
    # promoting dependencies here would change future activation semantics.
    extensions = list(extension_service.roots())
    return WorkspaceManifest(
        id=workspace.id,
        construction=workspace._construction,
        entities=entities,
        extensions=extensions,
        backend_summary=[backend.to_dict() for _, backend in workspace.backends.items()],
    )


def workspace_manifest(workspace: Any, *, include_system: bool = False) -> Dict[str, Any]:
    """Return a JSON-compatible workspace construction manifest.

    Args:
        workspace (Any): Live `HeavenBase` workspace.
        include_system (bool): Whether to include extension-owned Entity
            schemas. Requested extension roots are always exported. Defaults
            to `False`.

    Returns:
        Dict[str, Any]: JSON-compatible workspace shell without row data.
    """
    return manifest_from_workspace(workspace, include_system=include_system).to_dict()


def _entity_manifest(workspace: Any, entity: type) -> Dict[str, Any]:
    schema = entity.schema()
    spec = {
        "entity_id": schema.entity_id,
        "name": schema.name,
        "desc": schema.desc,
        "tags": list(schema.tags),
        "fields": {name: _field_manifest(workspace, entity, name) for name in schema.fields},
    }
    derives = workspace._schema_metadata.entity_derives(entity)
    if derives:
        spec["derives"] = derives
    indexes = [rule.to_dict() for rule in getattr(entity, "_hb_indexes", []) if rule.source == "entity"]
    if indexes:
        spec["indexes"] = indexes
    constraints = [rule.to_dict() for rule in getattr(entity, "_hb_constraints", []) if rule.source == "entity"]
    if constraints:
        spec["constraints"] = constraints
    options = workspace.entity_options.get(schema.entity_id)
    if options is not None:
        spec.update(options.to_dict())
    return spec


def _field_manifest(workspace: Any, entity: type, name: str) -> Dict[str, Any]:
    field_schema = entity.schema().get(name)
    spec = {
        "type": field_schema.dtype.to_dict(),
        "pk": field_schema.pk,
        "required": field_schema.required,
        "desc": field_schema.desc,
        "tags": list(field_schema.tags),
        "examples": list(field_schema.examples),
    }
    if field_schema.has_default:
        spec["has_default"] = True
        spec["default"] = field_schema.default
    field_spec = workspace._schema_metadata.field_spec(entity, name)
    if field_spec is not None:
        if field_spec.stores:
            spec["stores"] = [workspace._schema_metadata.storage_rule_dict(rule) for rule in field_spec.stores]
        if field_spec.index_rules:
            rules = [{"enabled": rule.enabled, **({"to": rule.backend} if rule.backend is not None else {})} for rule in field_spec.index_rules]
            if len(rules) == 1 and set(rules[0]) == {"enabled"}:
                spec["index"] = rules[0]["enabled"]
            else:
                spec["indexes"] = rules
        if field_spec.unique_rules:
            rules = [({"to": rule.backend} if rule.backend is not None else {}) for rule in field_spec.unique_rules]
            spec["unique"] = True if len(rules) == 1 and not rules[0] else rules[0]
        _add_hook_specs(spec, "computes", (workspace._schema_metadata.compute_dict(hook) for hook in field_spec.computes))
        _add_hook_specs(spec, "query_computes", (workspace._schema_metadata.query_compute_dict(hook) for hook in field_spec.query_computes))
    return _encoded_field(spec)


def _encoded_definition(definition: Mapping[str, Any]) -> Dict[str, Any]:
    """Return an entity definition with its date and datetime defaults and examples as a JSON definition holds them."""
    out = dict(definition)
    fields = definition.get("fields")
    if isinstance(fields, Mapping):
        out["fields"] = {name: _encoded_field(spec) for name, spec in fields.items()}
    elif isinstance(fields, list):
        out["fields"] = [_encoded_field(spec) for spec in fields]
    return out


def _add_hook_specs(spec: Dict[str, Any], key: str, items: Iterable[dict]) -> None:
    values = list(items)
    if values:
        spec[key] = values


def open_manifest(
    data: Mapping[str, Any] | WorkspaceManifest,
    *,
    context: Context | None = None,
    backends: Optional[Mapping[str, Any]] = None,
    preset: Optional[str] = None,
    computes: Optional[Mapping[str, Any]] = None,
    query_computes: Optional[Mapping[str, Any]] = None,
    allow_replace: bool = False,
    detached: bool = False,
) -> Any:
    """Construct a live workspace from a manifest.

    Args:
        data (Mapping[str, Any] | WorkspaceManifest): Manifest mapping or parsed
            manifest object.
        context: Context that owns the reconstructed workspace. Defaults to the
            process Context.
        backends: Optional complete Backend config or live-instance replacement.
        preset: Optional complete preset replacement. Explicit Backends take
            precedence.
        computes (Mapping[str, Any] | None): Optional compute function registry
            for entity definitions. Defaults to `None`.
        query_computes (Mapping[str, Any] | None): Optional query-compute
            function registry. Defaults to `None`.
        allow_replace (bool): Whether entity registration may replace an
            existing schema. Defaults to `False`.
        detached: Whether the returned workspace is caller-owned and absent
            from durable and live Context lookup. Defaults to `False`.

    Returns:
        Any: Live workspace reconstructed from the manifest.
    """
    from . import HeavenBase

    return HeavenBase.from_manifest(
        data,
        context=context,
        backends=backends,
        preset=preset,
        computes=computes,
        query_computes=query_computes,
        allow_replace=allow_replace,
        detached=detached,
    )


def restore_workspace_manifest(
    workspace_cls: type,
    data: Mapping[str, Any] | WorkspaceManifest,
    *,
    context: Context,
    backends: Optional[Mapping[str, Any]] = None,
    preset: Optional[str] = None,
    computes: Optional[Mapping[str, Any]] = None,
    query_computes: Optional[Mapping[str, Any]] = None,
    allow_replace: bool = False,
    detached: bool = False,
) -> Any:
    """Resolve manifest construction dependencies and replay one workspace.

    Args:
        workspace_cls (type): Workspace facade class to construct, typically
            ``HeavenBase``.
        data (Mapping[str, Any] | WorkspaceManifest): Manifest mapping or
            parsed ``WorkspaceManifest``.
        context (Context): Machine composition root that owns the
            reconstructed workspace.
        backends (Mapping[str, Any] | None): Optional complete Backend config
            or live-instance replacement. Defaults to ``None``.
        preset (str | None): Optional complete preset replacement. Defaults to
            ``None``.
        computes (Mapping[str, Any] | None): Optional compute-function registry.
            Defaults to ``None``.
        query_computes (Mapping[str, Any] | None): Optional query-compute
            registry. Defaults to ``None``.
        allow_replace (bool): Whether entity registration may replace an
            existing schema. Defaults to ``False``.
        detached (bool): Whether the returned workspace is caller-owned and
            absent from durable and live Context lookup. Defaults to
            ``False``.

    Returns:
        Any: Live workspace reconstructed from the manifest.

    Raises:
        RuntimeError: If runtime-only construction requires live Backends that
            are unavailable.
        TypeError: If explicit Backend replacement mixes config mappings and
            live instances.
        ValueError: If fixed workspace entities appear in the manifest.
    """
    from .setup.lifecycle import (
        _registered_workspace,
        _workspace_config_scope,
        prepare_workspace,
    )

    manifest = data if isinstance(data, WorkspaceManifest) else WorkspaceManifest.from_dict(data)
    _reject_fixed_entity_definitions(manifest)
    explicit_replacement = backends is not None or preset is not None
    required_identifiers = manifest.construction.backend_identifiers
    effective_backends = backends
    effective_preset = preset

    if not explicit_replacement:
        if manifest.construction.runtime_only:
            live = _registered_workspace(context, manifest.id)
            if live is None:
                raise RuntimeError(
                    f"Workspace manifest {manifest.id!r} requires live Backends "
                    f"{required_identifiers!r}; provide backends=... or open a matching "
                    "same-id workspace in this Context"
                )
            live._need_live()
            effective_backends = {name: backend for name, backend in live.backends.items()}
            actual = WorkspaceConstruction.from_runtime_backends(
                effective_backends,
                resolver=live.module_resolver,
            )
            _require_runtime_identifiers(
                manifest.id,
                required_identifiers,
                actual.backend_identifiers,
            )
        else:
            config = manifest.construction.kwargs
            effective_backends = config.get("backends")
            effective_preset = config.get("preset")
    elif manifest.construction.runtime_only:
        resolver = _prepared_module_resolver(context)
        if backends is not None:
            values = tuple(backends.values())
            if any(_is_live_backend(value) for value in values):
                if not values or not all(_is_live_backend(value) for value in values):
                    raise TypeError("backends must contain either config mappings or live Backend instances, not both")
                actual_identifiers = WorkspaceConstruction.from_runtime_backends(
                    backends,
                    resolver=resolver,
                ).backend_identifiers
            else:
                actual_identifiers = backend_identifiers_from_config(
                    backends,
                    resolver=resolver,
                )
        else:
            from .presets import resolve_workspace_backends, resolve_workspace_preset

            with _workspace_config_scope(context, manifest.id):
                preset_spec = resolve_workspace_preset(
                    preset,
                    manager=context.config,
                )
                configured = resolve_workspace_backends(
                    preset_spec,
                    ws_id=manifest.id,
                    manager=context.config,
                )
            actual_identifiers = backend_identifiers_from_config(
                configured,
                resolver=resolver,
            )
        _require_runtime_identifiers(
            manifest.id,
            required_identifiers,
            actual_identifiers,
        )

    if not detached:
        asserted = manifest.construction if not explicit_replacement or manifest.construction.runtime_only else None
        _, effective_backends, effective_preset, existing_spec = prepare_workspace(
            context,
            manifest.id,
            backends=effective_backends,
            preset=effective_preset,
            asserted_construction=asserted,
        )
        if existing_spec is not None and existing_spec.extension_roots is not None:
            roots = tuple(dict.fromkeys((*existing_spec.extension_roots, *manifest.extensions)))
            manifest = manifest.with_extensions(roots)

    construction_config = (
        {"backends": effective_backends} if effective_backends is not None else ({"preset": effective_preset} if effective_preset is not None else {})
    )
    return open_workspace_manifest(
        workspace_cls,
        manifest,
        context=context,
        computes=computes,
        query_computes=query_computes,
        allow_replace=allow_replace,
        tracked=not detached,
        register_workspace=not detached,
        track_live=not detached,
        detached=detached,
        construction_config=construction_config,
        required_backend_identifiers=required_identifiers or None,
    )


def open_workspace_manifest(
    workspace_cls: type,
    data: Mapping[str, Any] | WorkspaceManifest,
    *,
    context: Context,
    computes: Optional[Mapping[str, Any]] = None,
    query_computes: Optional[Mapping[str, Any]] = None,
    allow_replace: bool = False,
    tracked: bool = True,
    register_workspace: bool = True,
    track_live: bool = True,
    detached: bool = False,
    workspace: Any = None,
    construction_config: Optional[Mapping[str, Any]] = None,
    required_backend_identifiers: Optional[Mapping[str, str]] = None,
) -> Any:
    """Construct a workspace and replay manifest extensions and schemas.

    Args:
        workspace_cls (type): Workspace facade class to construct.
        data (Mapping[str, Any] | WorkspaceManifest): Manifest mapping or parsed
            manifest object.
        context: Context that owns the reconstructed workspace.
        computes (Mapping[str, Any] | None): Optional compute function registry
            for entity definitions. Defaults to `None`.
        query_computes (Mapping[str, Any] | None): Optional query-compute
            function registry. Defaults to `None`.
        allow_replace (bool): Whether entity registration may replace an
            existing schema. Defaults to `False`.
        tracked (bool): Whether to register the workspace for process lookup.
            Defaults to `True`.
        register_workspace (bool): Whether a successful tracked reconstruction
            should persist the manifest through `WorkspaceSpec`. Detached
            reconstructions never register. Defaults to `True`.
        track_live (bool): Whether class-level explicit-drop lookup owns the
            facade. Defaults to `True`.
        detached: Whether this facade is caller-owned and cannot participate
            in Context selection. Defaults to `False`.
        workspace: Optional already-allocated facade to initialize. Internal
            constructor paths use this to replay an existing registration
            without a second public construction API. Defaults to `None`.
        construction_config: Optional constructor-input override. Runtime-only
            workspaces use this to supply fresh live Backend instances while
            preserving the manifest's schema payload. Defaults to `None`.
        required_backend_identifiers: Optional complete runtime Backend
            name-to-identifier dependency signature to validate before schema
            replay.

    Returns:
        Any: Workspace reconstructed from construction config, extensions, and
        entity schemas.
    """
    from .setup.bootstrap import initialize_workspace
    from .setup.lifecycle import _publish_workspace, _workspace_config_scope

    manifest = data if isinstance(data, WorkspaceManifest) else WorkspaceManifest.from_dict(data)
    _reject_fixed_entity_definitions(manifest)
    target = workspace if workspace is not None else workspace_cls.__new__(workspace_cls)
    config = manifest.construction.kwargs if construction_config is None else dict(construction_config)
    initialize_workspace(
        target,
        manifest.id,
        context=context,
        backends=config.get("backends"),
        preset=config.get("preset"),
        detached=detached,
        tracked=False,
        register_workspace=False,
        register_pending=False,
        track_live=False,
        publish=False,
        extension_roots=manifest.extensions,
    )
    if required_backend_identifiers:
        actual = WorkspaceConstruction.from_runtime_backends(
            target.backends,
            resolver=target.module_resolver,
        )
        _require_runtime_identifiers(
            target.id,
            required_backend_identifiers,
            actual.backend_identifiers,
        )
        target._construction = manifest.construction
    save_replacements = register_workspace and tracked and allow_replace
    with _workspace_config_scope(context, target.id):
        with target._lock:
            # Settled before anything is registered, so a refused definition leaves no manifest entity in the store.
            replayed = _replayed_definitions(target, manifest.entities, computes=computes, query_computes=query_computes)
            target._schema._replaying = True
            try:
                for entity in replayed:
                    target._schema.register(
                        entity,
                        computes=computes,
                        query_computes=query_computes,
                        allow_replace=allow_replace,
                        replay_definition=entity if save_replacements else None,
                    )
            finally:
                target._schema._replaying = False
    replay_manifest = WorkspaceManifest(
        id=manifest.id,
        construction=target._construction,
        entities=[_encoded_definition(definition) for definition in replayed],
        extensions=list(target._extension_service.roots()),
        backend_summary=manifest.backend_summary,
    )
    finalize = None
    if register_workspace and tracked:

        def register_manifest() -> None:
            from .registry import WorkspaceSpec

            WorkspaceSpec.register(
                target.id,
                construction=target._construction,
                manifest=replay_manifest.to_dict(),
                context=context,
            )

        finalize = register_manifest

    with context.config.scoped(context.config.base_scope):
        _publish_workspace(
            target,
            tracked=tracked,
            track_live=track_live,
            finalize=finalize,
        )
    return target


def _replayed_definitions(
    workspace: Any,
    definitions: Iterable[Mapping[str, Any]],
    *,
    computes: Optional[Mapping[str, Any]] = None,
    query_computes: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return the manifest definitions replay registers, leaving out those of extension-registered entities.

    Extension activation has already registered each entity of an enabled extension, so its definition in the manifest
    is left out (and so is not saved for reopening by id) when it has the extension's structure and options. Its name,
    descriptions, tags and examples are not applied: the extension owns them, and a release may have changed them since
    the manifest was written. A ``UserWarning`` names those the definition states differently.

    Raises:
        ValueError: If a definition gives an extension-registered entity another structure or option, before any
            definition is registered.
    """
    from .schema.service import _definition_entity_id

    replayed = []
    for definition in definitions:
        entity_id = _definition_entity_id(definition)
        owner = workspace._extension_service.owner(entity_id) if entity_id else None
        if owner is None or entity_id not in workspace.entities:
            replayed.append(dict(definition))
            continue
        if not _matches_extension_definition(workspace, entity_id, definition, computes=computes, query_computes=query_computes):
            raise ValueError(workspace._schema._extension_entity_message(entity_id, owner, "imported with a different definition"))
        skipped = _unapplied_metadata(_entity_manifest(workspace, workspace.entities[entity_id]), _encoded_definition(definition))
        if skipped:
            warnings.warn(
                f"Entity {entity_id!r} is defined by extension {owner!r}, so the manifest's {', '.join(skipped)} for it were not "
                "applied; the extension's own are kept.",
                UserWarning,
                stacklevel=2,
            )
    return replayed


def _unapplied_metadata(live: Mapping[str, Any], definition: Mapping[str, Any]) -> List[str]:
    """Name the metadata a skipped extension entity definition states differently from the registered ``live`` definition."""

    def differs(stated: Mapping[str, Any], current: Mapping[str, Any], key: str) -> bool:
        if key not in stated:
            return False
        if key in ("tags", "examples"):
            return not _same_manifest_data(list(stated[key] or []), list(current.get(key) or []))
        return str(stated[key] or "") != str(current.get(key) or "")

    labels = (("name", "name"), ("desc", "description"), ("tags", "tags"))
    skipped = [label for key, label in labels if differs(definition, live, key)]
    fields = definition.get("fields") or {}
    for name, spec in fields.items() if isinstance(fields, Mapping) else ((item.get("name"), item) for item in fields if isinstance(item, Mapping)):
        current = live["fields"].get(str(name))
        if isinstance(spec, Mapping) and current is not None:
            for key, label in (("desc", "description"), ("tags", "tags"), ("examples", "examples")):
                if differs(spec, current, key):
                    skipped.append(f"{label} of field {name!r}")
    return skipped


def _matches_extension_definition(
    workspace: Any,
    entity_id: str,
    definition: Mapping[str, Any],
    *,
    computes: Optional[Mapping[str, Any]],
    query_computes: Optional[Mapping[str, Any]],
) -> bool:
    """Return whether ``definition`` has the structure and options of the registered ``entity_id``."""
    from .schema.options import EntityOptions

    live = workspace.entities[entity_id]
    options = workspace.entity_options[entity_id]
    if EntityOptions.resolve(options, definition, catalog=None, gram=None) != options:
        return False
    if _same_manifest_data(_definition_data(_entity_manifest(workspace, live), metadata=False), _definition_data(definition, metadata=False)):
        return True
    # Another spelling of the same definition; one with a lambda derive cannot compile and so does not match.
    try:
        compiled = workspace._schema.entity_cls(dict(definition), computes=computes, query_computes=query_computes, allow_dict=True)
    except (KeyError, TypeError, ValueError):
        return False
    metadata = workspace._schema_metadata
    # Type-aware, so a JSON default such as 0 does not match False.
    return _same_manifest_data(metadata.entity_structure(compiled), metadata.entity_structure(live))


def _definition_data(definition: Mapping[str, Any], *, metadata: bool) -> Dict[str, Any]:
    """Return a definition as JSON data to compare with another: no options, Context-local refs made relative.

    The one comparison of an extension entity's definitions: manifest replay compares them without metadata (the name,
    descriptions, tags and examples, which the extension supplies), and ``register`` and ``alter_entity`` compare them
    with it, since a change to the metadata is refused. Code an extension loads from the Registry runs in a
    content-addressed package whose name differs per Context root (see ``callable_structure_ref``), so a compute or
    derive reference is compared without that package name. A definition that names its entity only by ``name`` is
    compared under the entity id the compiler derives from it.
    """
    from .schema.service import _definition_entity_id

    dropped = ("catalog", "gram") if metadata else ("catalog", "gram", "name", "desc", "tags")
    shape = {key: _relative_refs(value) for key, value in _encoded_definition(definition).items() if key not in dropped}
    shape["entity_id"] = _definition_entity_id(definition)
    fields = shape.get("fields")
    if not metadata and isinstance(fields, Mapping):
        shape["fields"] = {
            name: (
                {key: value for key, value in spec.items() if key not in ("desc", "tags", "examples", "example_types")} if isinstance(spec, Mapping) else spec
            )
            for name, spec in fields.items()
        }
    return shape


def _relative_refs(value: Any) -> Any:
    """Return ``value`` with every ``fn`` reference into an artifact package written relative to that package.

    ``_definition_data`` compares definitions through it, since the artifact package name differs per Context root.
    """
    from .schema.metadata import _ARTIFACT_NAMESPACE

    if isinstance(value, Mapping):
        out = {key: _relative_refs(item) for key, item in value.items()}
        ref = out.get("fn")
        if isinstance(ref, str):
            namespace, separator, relative = ref.partition(".")
            if separator and _ARTIFACT_NAMESPACE.fullmatch(namespace):
                out["fn"] = f"artifact:{relative}"
        return out
    if isinstance(value, (list, tuple)):
        return [_relative_refs(item) for item in value]
    return deepcopy(value)


def _reject_fixed_entity_definitions(manifest: WorkspaceManifest) -> None:
    """Reject replayable definitions for code-owned workspace kernel Entities."""
    from .foundation import FIXED_ENTITY_IDS

    claimed = sorted(
        {
            str(entity.get("entity_id") or entity.get("identifier") or "")
            for entity in manifest.entities
            if str(entity.get("entity_id") or entity.get("identifier") or "") in FIXED_ENTITY_IDS
        }
    )
    if claimed:
        raise ValueError(f"Workspace manifests cannot define fixed workspace Entities {claimed!r}")


def _extension_ids(values: Iterable[Any]) -> List[str]:
    """Return canonical requested extension ids in first-seen order."""
    if isinstance(values, (str, bytes)):
        raise TypeError("Workspace manifest extensions must be an iterable of identifiers")
    return list(dict.fromkeys(text for value in values if (text := str(value).strip())))


def _is_live_backend(value: Any) -> bool:
    """Return whether a manifest replacement value is a live Backend."""
    from ..backends.base import Backend

    return isinstance(value, Backend)


def _prepared_module_resolver(context: Context) -> Any:
    """Return the Context resolver after its explicit module setup boundary."""
    with context.config.scoped(context.config.base_scope):
        context._setup_modules()
        return context.modules()


def _require_runtime_identifiers(
    ws_id: str,
    expected: Mapping[str, str],
    actual: Mapping[str, str],
) -> None:
    """Require an exact runtime Backend dependency signature."""
    expected_data = dict(expected)
    actual_data = dict(actual)
    if actual_data != expected_data:
        raise RuntimeError(f"Workspace manifest {ws_id!r} requires runtime Backends " f"{expected_data!r}, got {actual_data!r}")
