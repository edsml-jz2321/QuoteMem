"""Workspace runtime bootstrap assembly."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, Iterable, Mapping, Optional

from ...context import Context
from ...entity import pending_entities
from ...execution.handlers.registry import HandlerRegistry
from ...placement import StoragePlan, StorageResolver
from ...utils.ids import check_workspace_id
from ..foundation import FIXED_ENTITY_TYPES, bootstrap_fixed_entities
from ..schema.options import EntityOptions
from .backends import resolve_workspace_backend_setup
from .lifecycle import _new_workspace_lock, _publish_workspace, _workspace_config_scope
from .services import attach_workspace_services


def initialize_workspace(
    workspace: Any,
    ws_id: str,
    *,
    context: Context,
    backends: Optional[Mapping[str, Any]],
    preset: Optional[str],
    detached: bool,
    tracked: bool,
    register_workspace: bool,
    register_pending: bool,
    track_live: bool,
    publish: bool,
    extension_roots: Optional[Iterable[str]] = None,
) -> None:
    """Initialize a workspace facade with runtime registries, services, and system rows.

    Args:
        workspace (Any): Workspace facade being initialized.
        ws_id (str): Stable workspace identifier.
        context (Context): Owning machine composition root.
        backends (Mapping[str, Any] | None): Backend config or live instances.
        preset (str | None): Optional named backend preset.
        detached (bool): Whether the workspace is caller-owned outside Context
            lookup.
        tracked (bool): Whether durable workspace registration should occur.
        register_workspace (bool): Whether to register a ``WorkspaceSpec``.
        register_pending (bool): Whether pending entity declarations should bind
            to this workspace.
        track_live (bool): Whether the live workspace registry should retain
            the facade.
        publish (bool): Whether system rows should be published during
            initialization.
        extension_roots (Iterable[str] | None): Optional extension identifiers
            to enable. Defaults to ``None``.

    Returns:
        None: This function does not return a value.

    Raises:
        ValueError: If detached construction requests durable or live Context
            ownership.
    """
    if detached and (tracked or register_workspace or track_live):
        raise ValueError("Detached workspace construction cannot publish durable or live Context ownership")
    workspace.id = check_workspace_id(ws_id)
    workspace.context = context
    workspace._authority_token = context.authority_token
    workspace._detached = bool(detached)
    workspace._register_pending_entities = bool(register_pending)
    requested_tracked = bool(tracked)
    workspace._tracked = False
    workspace._lock = _new_workspace_lock(context, None if detached else workspace.id)
    # Construction writes system rows into the store that every facade of this id
    # shares, so it holds that workspace's exclusive side (after the Context lock).
    with workspace._lock:
        with context.config.scoped(context.config.base_scope):
            workspace.module_resolver = context.modules()
            context._setup_modules()
        with _workspace_config_scope(context, workspace.id):
            backend_setup = resolve_workspace_backend_setup(
                workspace.id,
                backends,
                preset,
                manager=context.config,
                resolver=workspace.module_resolver,
            )
            workspace.preset = backend_setup.preset
            workspace.preset_spec = backend_setup.preset_spec
            workspace._construction = backend_setup.construction
            workspace.backends = backend_setup.backends

            from ...capabilities.index import CapabilityIndex
            from ...execution.handlers.seed import seed_handlers

            workspace.handlers: HandlerRegistry = seed_handlers(resolver=workspace.module_resolver)
            workspace.capabilities = CapabilityIndex(workspace.handlers, workspace.backends, resolver=workspace.module_resolver)
            backend_names = workspace.backends.names()
            default_backend = "main" if "main" in backend_names else (backend_names[0] if backend_names else "main")
            workspace.storage_resolver = StorageResolver(default_backend=default_backend, resolver=workspace.module_resolver)
            workspace.entities: Dict[str, type] = {}
            workspace.storage_plans: Dict[str, StoragePlan] = {}
            workspace.entity_options: Dict[str, EntityOptions] = {}

            attach_workspace_services(workspace)

            bootstrap_fixed_entities(workspace)
            workspace._extension_service.enable_required_and_defaults(extension_roots)
            workspace._metaschema.publish_workspace_rows()
            _register_pending_entities(workspace)
        if publish:
            finalize = None
            if register_workspace:

                def register_workspace_spec() -> None:
                    from ..registry import WorkspaceSpec

                    WorkspaceSpec.register(
                        workspace.id,
                        construction=workspace._construction,
                        extension_roots=workspace._extension_service.roots(),
                        context=context,
                    )

                finalize = register_workspace_spec

            with context.config.scoped(context.config.base_scope):
                _publish_workspace(
                    workspace,
                    tracked=requested_tracked,
                    track_live=track_live,
                    finalize=finalize,
                )


def _register_pending_entities(workspace: Any) -> None:
    """Register entity classes declared for this workspace before it existed."""
    if not workspace._register_pending_entities:
        return
    for entity in pending_entities():
        binding = (workspace.context.identity_token, workspace.id)
        if entity not in FIXED_ENTITY_TYPES and binding in getattr(entity, "_hb_workspace_bindings", set()):
            with workspace._lock:
                workspace._schema.register(entity)
