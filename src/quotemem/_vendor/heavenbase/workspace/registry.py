"""Persistent workspace registry row contract."""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

__all__ = [
    "WorkspaceSpec",
]

from heavenbase.utils import check_workspace_id, deepcopy, to_mutable_data
from heavenbase.utils.typing import Any, Dict, Iterable, List, Optional, Tuple, dataclass, field

from ..context import Context, context_or_default
from ..context.reset import reset_default_local_state
from .setup.backends import WorkspaceConstruction


@dataclass(frozen=True)
class WorkspaceSpec:
    """Persistent metadata for one Context-registered workspace.

    `WorkspaceSpec` is the typed value stored by the Context-owned workspace
    Registry. It points from workspace names to one canonical construction
    contract and does not own live workspace objects.

    Args:
        id (str): Registered workspace id.
        construction: Canonical ambient, preset, configured-backend, or
            runtime-backend construction contract.
        manifest (Dict[str, Any]): Optional workspace manifest replayed when
            reopening this workspace. A tracked workspace replaces the saved
            definition of an entity it defines when it changes that entity.
        extension_roots: Explicit/default extension roots snapshotted for this
            workspace. ``None`` marks a legacy row whose defaults have not yet
            been snapshotted.
        created_at (str): ISO timestamp for first registration.
        updated_at (str): ISO timestamp for the latest registration update.
        active (bool): Whether the owning Context currently selects this workspace.
    """

    id: str
    construction: WorkspaceConstruction = field(default_factory=WorkspaceConstruction)
    manifest: Dict[str, Any] = field(default_factory=dict)
    extension_roots: Optional[Tuple[str, ...]] = None
    created_at: str = ""
    updated_at: str = ""
    active: bool = False

    @property
    def config(self) -> Dict[str, Any]:
        """Return replayable ``HeavenBase`` constructor arguments.

        Returns:
            Dict[str, Any]: Detached preset/backend configuration.

        Raises:
            RuntimeError: If the workspace requires caller-supplied live
                Backend instances.
        """
        return self.construction.kwargs

    @property
    def runtime_only(self) -> bool:
        """Return whether reopening requires caller-supplied Backend instances."""
        return self.construction.runtime_only

    @classmethod
    def register(
        cls,
        ws_id: str,
        config: Optional[Dict[str, Any]] = None,
        manifest: Optional[Dict[str, Any]] = None,
        extension_roots: Optional[Iterable[str]] = None,
        *,
        construction: WorkspaceConstruction | Dict[str, Any] | None = None,
        context: Context | None = None,
    ) -> "WorkspaceSpec":
        """Register or update one globally browsable workspace.

        Args:
            ws_id (str): Workspace id to register.
            config (Dict[str, Any] | None): Workspace construction config. Defaults to ``None``.
            manifest (Dict[str, Any] | None): Optional workspace manifest replayed when reopening this workspace. Defaults to ``None``.
            extension_roots: Requested workspace extension identifiers. When
                omitted, existing state is preserved and a supplied manifest
                supplies the roots.
            construction: Canonical construction contract. This is the runtime
                registration path; manual callers may continue to supply
                replayable ``config`` instead. Defaults to ``None``.
            context: Context that owns the workspace directory.

        Returns:
            WorkspaceSpec: Registered workspace metadata.

        Raises:
            ValueError: If multiple construction authorities are supplied or a
                manifest disagrees with the canonical construction contract.

        Notes:
            When another Context or process published the workspace registry
            after this Context last read it, the registry is read again and the
            update is computed afresh from that state, so registering a
            workspace never fails because another one was registered or
            changed meanwhile.
        """
        from ..utils.registry import RegistryConflictError
        from .manifest import _same_manifest_data

        selected = context_or_default(context)
        clean_id = check_workspace_id(ws_id)
        with selected._lock:
            registry = _registry(selected)
            required_ids: Tuple[str, ...] = ()
            if extension_roots is not None or manifest is not None:
                from .setup.extensions import trusted_required_extension_ids

                selected._setup_modules()
                required_ids = trusted_required_extension_ids(selected.modules())
            explicit_roots = _normalize_extension_roots(extension_roots, excluded=required_ids)
            while True:
                existing = registry.get(clean_id)
                canonical = _next_construction(existing, config=config, construction=construction, manifest=manifest)
                next_manifest = _next_manifest(clean_id, existing, construction=canonical, manifest=manifest)
                next_roots = _next_extension_roots(
                    existing,
                    explicit=explicit_roots,
                    manifest=manifest,
                    normalized_manifest=next_manifest,
                    excluded=required_ids,
                )
                next_manifest = _manifest_with_extension_roots(next_manifest, next_roots)
                if (
                    existing is not None
                    and canonical == existing.construction
                    and _same_manifest_data(next_manifest, existing.manifest)
                    and next_roots == existing.extension_roots
                ):
                    # The no-op skips the revision check in update(), so confirm the snapshot here.
                    if registry.refresh():
                        continue
                    return cls.load(clean_id, context=selected)
                now = _now()
                spec = cls(
                    id=clean_id,
                    construction=canonical,
                    manifest=next_manifest,
                    extension_roots=next_roots,
                    created_at=str(existing.created_at if existing is not None else now),
                    updated_at=now,
                )
                try:
                    registry.update(clean_id, spec)
                except RegistryConflictError:
                    if not registry.refresh():
                        raise
                    continue
                return cls.load(clean_id, context=selected)

    @classmethod
    def _save_manifest(cls, spec: "WorkspaceSpec", manifest: Dict[str, Any], *, context: Context) -> None:
        """Replace the saved manifest of a registered workspace, keeping its construction and extension roots.

        The update is published against the registry state ``spec`` was read
        from. Unlike ``register``, it sets up no modules: the extension roots,
        and so the trusted modules, stay as they are.

        Args:
            spec: The workspace's registration, as read from this Context's registry.
            manifest: Replacement manifest with the same construction.
            context: Context that owns the workspace directory.

        Returns:
            None: The registry is updated in place.

        Raises:
            KeyError: If the workspace is no longer registered.
            ValueError: If the manifest disagrees with the registered construction.
            RegistryConflictError: If another Context or process published the registry after this Context read it.
        """
        from .manifest import _same_manifest_data

        with context._lock:
            registry = _registry(context)
            existing = registry.get(spec.id)
            if existing is None:
                raise KeyError(f"No registered workspace {spec.id!r}")
            next_manifest = _checked_manifest(spec.id, manifest, construction=existing.construction)
            if _same_manifest_data(next_manifest, existing.manifest):
                return
            registry.update(spec.id, replace(existing, manifest=next_manifest, updated_at=_now()))

    @classmethod
    def setup(
        cls,
        reset: bool = False,
        ws_id: Optional[str] = None,
        *,
        force: bool = False,
        context: Context | None = None,
    ) -> "WorkspaceSpec":
        """Initialize global HeavenBase config and register the default workspace.

        Args:
            reset (bool): Whether to reset package-owned local state first. Defaults to ``False``.
            ws_id (str | None): Workspace id to register. Defaults to configured default workspace.
            force (bool): Whether to re-index the packaged module set even when
                its durable composition marker is current, without resetting
                other local state. Implied by ``reset``. Defaults to ``False``.
            context: Context that owns setup and workspace registration.

        Returns:
            WorkspaceSpec: Registered workspace metadata.
        """
        selected = context_or_default(context)
        if reset:
            reset_default_local_state(selected)
        selected.config.setup(reset=reset)
        selected._setup_modules(force=reset or force)
        target = check_workspace_id(ws_id or selected.default_workspace_id)
        return cls.register(target, context=selected)

    @classmethod
    def list(cls, *, context: Context | None = None) -> List["WorkspaceSpec"]:
        """Return globally registered workspace specs.

        Args:
            context: Context that owns the workspace directory.

        Returns:
            List[WorkspaceSpec]: Registered workspace metadata rows.
        """
        selected = context_or_default(context)
        active_id = selected.active_workspace_id
        return [cls.from_dict(item.to_dict() | {"active": item.id == active_id}) for item in sorted(_registry(selected).values(), key=lambda value: value.id)]

    @classmethod
    def load(cls, ws_id: Optional[str] = None, *, context: Context | None = None) -> "WorkspaceSpec":
        """Return one workspace spec, defaulting to the active or configured default workspace.

        Args:
            ws_id (str | None): Workspace id to load. Defaults to active or configured default workspace.
            context: Context that owns the workspace directory.

        Returns:
            WorkspaceSpec: Registered workspace metadata.

        Raises:
            KeyError: If no registered workspace matches the target id.
        """
        selected = context_or_default(context)
        target = ws_id or selected.effective_workspace_id
        item = _registry(selected).get(target)
        if item is None:
            raise KeyError(f"No registered workspace {target!r}")
        active_id = selected.active_workspace_id
        return cls.from_dict(item.to_dict() | {"active": target == active_id})

    @classmethod
    def unregister(
        cls,
        ws_id: str,
        *,
        context: Context | None = None,
    ) -> bool:
        """Remove a globally registered workspace.

        Args:
            ws_id (str): Workspace id to remove.
            context: Context that owns the workspace directory.

        Returns:
            bool: Whether a workspace registration existed.
        """
        selected = context_or_default(context)
        clean_id = check_workspace_id(ws_id)
        registry = _registry(selected)
        from ..utils.registry import RegistryConflictError

        with selected._lock:
            registry.refresh()
            if registry.get(clean_id) is None:
                return False
            was_active = selected.active_workspace_id == clean_id
            if was_active:
                selected.deactivate_workspace()
            try:
                while True:
                    try:
                        return registry.delete(clean_id, missing_ok=True) is not None
                    except RegistryConflictError:
                        if not registry.refresh():
                            raise
            except BaseException:
                if was_active:
                    selected.activate_workspace(clean_id)
                raise

    def open(self, *, context: Context | None = None) -> Any:
        """Open this registered workspace as a live ``HeavenBase`` facade.

        Args:
            context: Context that owns the live workspace.

        Returns:
            Any: Live ``HeavenBase`` workspace.
        """
        from heavenbase.workspace import HeavenBase

        return HeavenBase.load(self.id, context=context)

    def open_detached(self, *, context: Context | None = None) -> Any:
        """Open a new caller-owned facade outside process lifecycle registries.

        The result never replaces a same-id process workspace and cannot be
        found by ``HeavenBase.load`` or class-level ``HeavenBase.drop(id)``.
        This is intended for bounded staging work whose caller retains the
        facade for as long as it is needed.

        Args:
            context (Context | None): Machine composition root that owns the
                detached workspace. Defaults to the process Context.

        Returns:
            Any: New detached ``HeavenBase`` workspace.
        """
        from heavenbase.workspace import HeavenBase

        selected = context_or_default(context)
        if self.manifest:
            return HeavenBase.from_manifest(
                self.manifest,
                context=selected,
                detached=True,
            )
        if self.runtime_only:
            live = selected._workspace_registry.get(self.id)
            if live is None:
                raise RuntimeError(
                    f"Workspace {self.id!r} uses runtime-only backends; reconstruct it with " "HeavenBase(id, backends=..., context=..., detached=True)"
                )
            actual = WorkspaceConstruction.from_runtime_backends(
                live.backends,
                resolver=live.module_resolver,
            )
            if actual != self.construction:
                raise RuntimeError(f"Live backend identifiers disagree with the registered construction for workspace {self.id!r}")
            return HeavenBase._construct(
                self.id,
                context=selected,
                backends={name: backend for name, backend in live.backends.items()},
                detached=True,
                tracked=False,
                register_workspace=False,
                register_pending=False,
                track_live=False,
                publish=False,
                extension_roots=self.extension_roots,
            )
        return HeavenBase._construct(
            self.id,
            context=selected,
            detached=True,
            tracked=False,
            register_workspace=False,
            register_pending=False,
            track_live=False,
            **deepcopy(self.config),
            extension_roots=self.extension_roots,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceSpec":
        """Build a workspace spec from JSON-compatible data.

        Args:
            data (Dict[str, Any]): Persisted workspace registry row.

        Returns:
            WorkspaceSpec: Parsed workspace registry row.

        Raises:
            ValueError: If the embedded manifest id does not match the
            workspace id.
        """
        clean_id = check_workspace_id(data["id"])
        construction = (
            WorkspaceConstruction.from_dict(data["construction"]) if "construction" in data else WorkspaceConstruction.from_config(data.get("config") or {})
        )
        manifest = _checked_manifest(clean_id, data.get("manifest") or {}, construction=construction)
        roots = _normalize_extension_roots(data.get("extension_roots")) if "extension_roots" in data else None
        if manifest:
            from .manifest import WorkspaceManifest

        if roots is None and manifest:
            roots = tuple(WorkspaceManifest.from_dict(manifest).extensions)
        if roots is not None and manifest:
            manifest_roots = tuple(WorkspaceManifest.from_dict(manifest).extensions)
            if manifest_roots != roots:
                raise ValueError(f"Workspace manifest extensions disagree with registered extension roots for {clean_id!r}")
        return cls(
            id=clean_id,
            construction=construction,
            manifest=manifest,
            extension_roots=roots,
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            active=bool(data.get("active", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return this registry row as JSON-compatible data.

        Args:
            None.

        Returns:
            Dict[str, Any]: Deep-copied persistence payload.
        """
        data = {
            "id": self.id,
            "construction": to_mutable_data(self.construction.to_dict()),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.manifest:
            data["manifest"] = to_mutable_data(self.manifest)
        if self.extension_roots is not None:
            data["extension_roots"] = list(self.extension_roots)
        return data


def _checked_manifest(
    ws_id: str,
    manifest: Any,
    *,
    construction: WorkspaceConstruction,
) -> Dict[str, Any]:
    """Return a detached manifest validated against one construction authority."""
    from .manifest import WorkspaceManifest

    clean_id = check_workspace_id(ws_id)
    data = deepcopy(dict(manifest or {}))
    if not data:
        return {}
    if check_workspace_id(str(data.get("id") or "")) != clean_id:
        raise ValueError(f"Workspace manifest id {data.get('id')!r} does not match registered workspace {clean_id!r}")
    parsed = WorkspaceManifest.from_dict(data)
    if int(data.get("version") or 1) == 1 and "construction" not in data and not dict(data.get("config") or {}) and construction.runtime_only:
        parsed = parsed.with_construction(construction)
    if parsed.construction != construction:
        raise ValueError(f"Workspace manifest construction disagrees with registered construction for {clean_id!r}")
    return parsed.to_dict()


def _next_construction(
    existing: WorkspaceSpec | None,
    *,
    config: Optional[Dict[str, Any]],
    construction: WorkspaceConstruction | Dict[str, Any] | None,
    manifest: Optional[Dict[str, Any]],
) -> WorkspaceConstruction:
    """Resolve one canonical construction authority for a registry update."""
    if config is not None and construction is not None:
        raise ValueError("Workspace registration accepts either config or construction, not both")
    if construction is not None:
        return construction if isinstance(construction, WorkspaceConstruction) else WorkspaceConstruction.from_dict(construction)
    if config is not None:
        return WorkspaceConstruction.from_config(config)
    if manifest:
        from .manifest import WorkspaceManifest

        return WorkspaceManifest.from_dict(manifest).construction
    return existing.construction if existing is not None else WorkspaceConstruction()


def _next_manifest(
    ws_id: str,
    existing: WorkspaceSpec | None,
    *,
    construction: WorkspaceConstruction,
    manifest: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the replacement or preserved manifest for one registry update."""
    if manifest is not None:
        return _checked_manifest(ws_id, manifest, construction=construction)
    current = deepcopy(existing.manifest) if existing is not None else {}
    if not current:
        return current
    from .manifest import WorkspaceManifest

    updated = WorkspaceManifest.from_dict(current).with_construction(construction)
    return _checked_manifest(ws_id, updated.to_dict(), construction=construction)


def _normalize_extension_roots(values: Any, *, excluded: Iterable[str] = ()) -> Optional[Tuple[str, ...]]:
    """Return stable requested extension roots or the legacy ``None`` marker."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError("Workspace extension roots must be an iterable of identifiers")
    ignored = frozenset(str(value) for value in excluded)
    return tuple(dict.fromkeys(text for value in values if (text := str(value).strip()) and text not in ignored))


def _next_extension_roots(
    existing: WorkspaceSpec | None,
    *,
    explicit: Optional[Tuple[str, ...]],
    manifest: Optional[Dict[str, Any]],
    normalized_manifest: Dict[str, Any],
    excluded: Iterable[str] = (),
) -> Optional[Tuple[str, ...]]:
    """Resolve one monotonic durable requested-extension authority."""
    manifest_roots: Optional[Tuple[str, ...]] = None
    if manifest is not None and normalized_manifest:
        from .manifest import WorkspaceManifest

        manifest_roots = _normalize_extension_roots(WorkspaceManifest.from_dict(normalized_manifest).extensions, excluded=excluded)
    if explicit is not None and manifest_roots is not None and explicit != manifest_roots:
        raise ValueError("Workspace manifest extensions disagree with explicit extension roots")
    proposed = explicit if explicit is not None else manifest_roots
    if existing is None or existing.extension_roots is None:
        return proposed
    current = _normalize_extension_roots(existing.extension_roots, excluded=excluded) or ()
    if proposed is None:
        return current
    return tuple(dict.fromkeys((*current, *proposed)))


def _manifest_with_extension_roots(manifest: Dict[str, Any], roots: Optional[Tuple[str, ...]]) -> Dict[str, Any]:
    """Keep an embedded reconstructive manifest synchronized with root state."""
    if not manifest or roots is None:
        return manifest
    from .manifest import WorkspaceManifest

    parsed = WorkspaceManifest.from_dict(manifest)
    return WorkspaceManifest(
        id=parsed.id,
        construction=parsed.construction,
        entities=parsed.entities,
        extensions=list(roots),
        backend_summary=parsed.backend_summary,
    ).to_dict()


def _registry(context: Context) -> Any:
    return context.registry(
        "system-workspaces",
        encode_value=_encode_workspace,
        decode_value=_decode_workspace,
    )


def _encode_workspace(value: WorkspaceSpec) -> Dict[str, Any]:
    if not isinstance(value, WorkspaceSpec):
        raise TypeError(f"Workspace Registry values must be WorkspaceSpec, got {type(value).__name__}")
    return value.to_dict()


def _decode_workspace(value: Any) -> WorkspaceSpec:
    if not isinstance(value, dict):
        raise TypeError(f"Workspace Registry payloads must be mappings, got {type(value).__name__}")
    return WorkspaceSpec.from_dict(value)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
