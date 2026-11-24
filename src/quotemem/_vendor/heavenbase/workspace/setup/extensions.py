"""Workspace extension activation and durable root policy."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, Iterable, List, Mapping, Optional

from ...extensions.base import Extension
from ..foundation import FIXED_ENTITY_IDS


def trusted_required_extension_ids(resolver: Any) -> tuple[str, ...]:
    """Return installer-trusted built-ins declaring universal activation.

    Args:
        resolver (Any): Module resolver used to inspect installed extension
            records.

    Returns:
        tuple[str, ...]: Required extension identifiers in Registry order.
    """
    required: List[str] = []
    for row in resolver.inspect("extension"):
        groups = row.get("groups") or {}
        trusted = "system" in groups.get("provenance", ()) and "builtin" in groups.get("tag", ())
        if not trusted or "system" not in groups.get("scope", ()):
            continue
        item = Extension.load(str(row["identifier"]), resolver=resolver)
        if item.required:
            required.append(item.identifier)
    return tuple(required)


class _ExtensionService:
    """Resolve, activate, and report Registry-defined workspace extensions."""

    def __init__(self, workspace: Any):
        self.workspace = workspace
        # extension id -> entities its activation registered; recorded once activation returns
        self._entity_ids: Dict[str, tuple[str, ...]] = {}
        workspace._enabled_extensions: Dict[str, Extension] = {}
        workspace._extension_apis: Dict[str, Any] = {}
        workspace._extension_api_owners: Dict[str, str] = {}
        workspace._extension_roots: List[str] = []

    def enable_required_and_defaults(self, extension_roots: Optional[Iterable[str]] = None) -> None:
        """Enable trusted universal extensions and one explicit/default root set.

        ``None`` means a newly created or legacy workspace whose configured
        defaults must be snapshotted. Any iterable, including an empty one, is
        authoritative replay state.
        """
        for extension_id in self._trusted_required_ids():
            self._activate(extension_id)
        roots = self.default_ids() if extension_roots is None else self._clean_ids(extension_roots)
        for extension_id in roots:
            self.enable(extension_id, persist=False)

    def enable(self, extension: str, *, persist: bool = True) -> Extension:
        """Enable one requested extension and its exact dependency closure.

        Args:
            extension: Registered Extension identifier.
            persist: Whether a tracked workspace should durably snapshot this
                requested root. Bootstrap and manifest replay pass ``False``.

        Returns:
            Enabled descriptor for the requested root.
        """
        if not isinstance(extension, str):
            raise TypeError("Workspace extension enablement requires a registered identifier")
        if persist:
            self._sync_persisted_roots()
        requested = Extension.load(extension, resolver=self.workspace.module_resolver)
        self._activate(requested.identifier)
        added = requested.identifier not in self.workspace._extension_roots and not requested.required
        if added:
            self.workspace._extension_roots.append(requested.identifier)
        try:
            if persist and added:
                self._persist_roots()
        except BaseException:
            if added:
                self.workspace._extension_roots.remove(requested.identifier)
            raise
        return self.enabled()[requested.identifier]

    def _sync_persisted_roots(self) -> None:
        """Bring a stale same-id facade up to the durable monotonic root set."""
        if not self.workspace._tracked or self.workspace.detached:
            return
        from ..registry import WorkspaceSpec

        persisted = WorkspaceSpec.load(self.workspace.id, context=self.workspace.context).extension_roots or ()
        for extension_id in persisted:
            if extension_id in self.workspace._extension_roots:
                continue
            self._activate(extension_id)
            self.workspace._extension_roots.append(extension_id)

    def _activate(self, extension_id: str) -> Extension:
        """Preflight and activate one deterministic dependency closure."""
        plan = self._activation_plan(extension_id)
        self._preflight(plan)
        enabled = self.enabled()
        for item in plan:
            if item.identifier in enabled:
                continue
            entities = item._apply(self.workspace)
            self.workspace._metaschema.publish_extension_row(item, entities)
            self._entity_ids[item.identifier] = tuple(entity.schema().entity_id for entity in entities)
            enabled[item.identifier] = item
        return enabled[extension_id]

    def _activation_plan(self, extension_id: str) -> tuple[Extension, ...]:
        """Return dependency-first activation order or report the full cycle."""
        resolver = self.workspace.module_resolver
        state: Dict[str, str] = {}
        stack: List[str] = []
        ordered: List[Extension] = []

        def visit(identifier: str) -> None:
            status = state.get(identifier)
            if status == "done":
                return
            if status == "visiting":
                start = stack.index(identifier)
                cycle = stack[start:] + [identifier]
                raise ValueError(f"Extension activation dependency cycle: {' -> '.join(cycle)}")
            item = Extension.load(identifier, resolver=resolver)
            state[identifier] = "visiting"
            stack.append(identifier)
            for dependency in item.requires:
                visit(dependency)
            stack.pop()
            state[identifier] = "done"
            ordered.append(item)

        visit(extension_id)
        return tuple(ordered)

    def _preflight(self, plan: Iterable[Extension]) -> None:
        """Reject predictable activation failures before schema mutation."""
        ws = self.workspace
        api_owners = dict(ws._extension_api_owners)
        for item in plan:
            fixed = sorted(FIXED_ENTITY_IDS.intersection(item.entity_ids()))
            if fixed:
                raise ValueError(f"Extension {item.identifier!r} cannot own fixed workspace Entities {fixed!r}")
            item._check_api_slot(ws, api_owners)
            if item.api_name is not None:
                api_owners[item.api_name] = item.identifier
            for declaration in item.entities:
                entity = ws._schema.entity_cls(declaration, allow_dict=True)
                entity_id = entity.schema().entity_id
                existing = ws.entities.get(entity_id)
                if existing is not None and ws._schema_metadata.entity_structure(existing) != ws._schema_metadata.entity_structure(entity):
                    raise ValueError(f"Extension {item.identifier!r} Entity {entity_id!r} conflicts with the workspace schema")

    def _trusted_required_ids(self) -> tuple[str, ...]:
        """Return installer-trusted built-ins that declare universal activation."""
        return trusted_required_extension_ids(self.workspace.module_resolver)

    def _persist_roots(self) -> None:
        """Write requested roots through the Context-owned WorkspaceSpec."""
        if not self.workspace._tracked or self.workspace.detached:
            return
        from ..registry import WorkspaceSpec

        persisted = WorkspaceSpec.register(
            self.workspace.id,
            construction=self.workspace._construction,
            extension_roots=self.roots(),
            context=self.workspace.context,
        )
        self.workspace._extension_roots[:] = list(persisted.extension_roots or ())

    def enabled(self) -> Dict[str, Extension]:
        """Return enabled runtime extensions keyed by extension id."""
        return self.workspace._enabled_extensions

    def roots(self) -> tuple[str, ...]:
        """Return requested/default activation roots in durable order."""
        return tuple(self.workspace._extension_roots)

    def entity_ids(self) -> set[str]:
        """Return entity identifiers registered by enabled extensions, setup-hook entities included."""
        return {entity_id for entities in self._entity_ids.values() for entity_id in entities}

    def owner(self, entity_id: str) -> Optional[str]:
        """Return the enabled extension that defines ``entity_id``.

        An extension registers its own definition of its entities each time the
        workspace opens, so a definition changed after activation would be
        replaced on the next reopen and left out of the default manifest.
        Registration paths therefore refuse such a change. An extension's
        entities are recorded only once its activation returns, so activation
        itself registers them freely.

        Args:
            entity_id (str): Entity identifier to look up.

        Returns:
            str | None: Owning extension identifier, or ``None``.
        """
        return next((extension_id for extension_id, entities in self._entity_ids.items() if entity_id in entities), None)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return enabled extension metadata keyed by extension id."""
        return {
            extension_id: {**extension.to_dict(), "entities": list(self._entity_ids[extension_id])}
            for extension_id, extension in sorted(self.enabled().items())
        }

    def default_ids(self) -> List[str]:
        """Return configured extension roots for a newly created workspace."""
        raw = self.workspace.context.config.get("heavenbase.extensions.default", default=[])
        if raw in (None, "", "__HB_REMOVE__"):
            return []
        if isinstance(raw, str):
            values = raw.split(",")
        elif isinstance(raw, Mapping):
            if raw.get("__HB_REMOVE__"):
                return []
            raise TypeError("heavenbase.extensions.default must be a string or iterable of extension ids")
        elif isinstance(raw, Iterable):
            values = raw
        else:
            raise TypeError("heavenbase.extensions.default must be a string or iterable of extension ids")
        return self._clean_ids(values)

    @staticmethod
    def _clean_ids(values: Iterable[Any]) -> List[str]:
        """Return non-empty, de-duplicated extension identifiers."""
        cleaned = [str(item).strip() for item in values]
        return list(dict.fromkeys(item for item in cleaned if item and item != "__HB_REMOVE__"))

    def list_entities(self) -> List[str]:
        """Return user-facing entity identifiers registered in the workspace."""
        hidden = set(FIXED_ENTITY_IDS)
        for extension_id, extension in self.enabled().items():
            if extension.meta.get("hide_from_list_entities"):
                hidden.update(self._entity_ids[extension_id])
        return sorted(entity_id for entity_id in self.workspace.entities if entity_id not in hidden)
