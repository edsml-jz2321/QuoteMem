"""Toolkit persistence composed over Toolkit rows and CapsuleRegistry."""

from __future__ import annotations

__all__ = ["ToolkitRegistry", "toolkit_registry"]

from time import time

from heavenbase.utils import Any, Dict, Optional

from heavenbase.capsule.registry import CapsuleRegistry
from heavenbase.capsule.versions import select_versioned_row
from .store import ToolkitRowStore


class ToolkitRegistry:
    """Persist Toolkit manifests while delegating executable refs to Capsule."""

    def __init__(
        self,
        store: ToolkitRowStore,
        capsules: CapsuleRegistry,
        *,
        toolkit_type: type | None = None,
        workspace: Any = None,
    ) -> None:
        """Create a Toolkit registry over row and Capsule persistence services.

        Args:
            store (ToolkitRowStore): Row store for ``sys-toolkit`` manifests.
            capsules (CapsuleRegistry): Capsule registry for tool executable refs.
            toolkit_type (type | None): Entity class used to validate Toolkit values. Defaults to ``Toolkit``.
            workspace (Any): Optional workspace bound to loaded Toolkits.

        Returns:
            None: This initializer stores registry services on the instance.

        Raises:
            TypeError: If ``store``, ``capsules``, or ``toolkit_type`` are invalid.
        """
        if not isinstance(store, ToolkitRowStore):
            raise TypeError("ToolkitRegistry store must implement the ToolkitRowStore protocol")
        if not isinstance(capsules, CapsuleRegistry):
            raise TypeError("ToolkitRegistry capsules must be a CapsuleRegistry")
        if toolkit_type is None:
            from .toolkit import Toolkit

            toolkit_type = Toolkit
        if not isinstance(toolkit_type, type) or toolkit_type.schema().entity_id != "sys-toolkit":
            raise TypeError("ToolkitRegistry toolkit_type must define the sys-toolkit Entity")
        self.store = store
        self.capsules = capsules
        self.toolkit_type = toolkit_type
        self.authority = capsules.authority
        self.config = capsules.config
        self.workspace = workspace

    @classmethod
    def from_workspace(cls, workspace: Any) -> "ToolkitRegistry":
        """Build Toolkit and Capsule row services over one workspace.

        Args:
            workspace (Any): Explicit `HeavenBase` workspace that owns the
                concrete Toolkit and referenced Capsule rows.

        Returns:
            ToolkitRegistry: Workspace-bound Toolkit persistence service.
        """
        from heavenbase.entity.registry import load_entity_type
        from heavenbase.workspace.rows.adapter import WorkspaceEntityRowStore

        resolver = workspace.module_resolver
        config = workspace.context.config
        capsule_type = load_entity_type("sys-capsule", resolver=resolver)
        toolkit_type = load_entity_type("sys-toolkit", resolver=resolver)
        capsules = CapsuleRegistry(
            WorkspaceEntityRowStore(workspace, capsule_type, extension="capsule"),
            registry_config={"workspace": workspace.id},
            resolver=resolver,
            config=config,
        )
        return cls(
            WorkspaceEntityRowStore(workspace, toolkit_type, extension="toolkit"),
            capsules,
            toolkit_type=toolkit_type,
            workspace=workspace,
        )

    def list(self, namespace: Optional[str] = None, status: Optional[str] = "active") -> list[Dict[str, Any]]:
        """List matching Toolkit rows.

        Args:
            namespace (Optional[str]): Optional namespace filter. Defaults to no namespace filtering.
            status (Optional[str]): Optional status filter. Defaults to ``"active"``; pass ``None`` to disable status filtering.

        Returns:
            list[Dict[str, Any]]: Matching Toolkit rows from the row store.
        """
        rows = self.store.rows()
        if namespace is not None:
            rows = [row for row in rows if row.get("namespace") == namespace]
        if status is not None:
            rows = [row for row in rows if row.get("status") == status]
        return rows

    def delete(self, toolkit_id: str, *, tombstone: bool = True) -> bool:
        """Delete or tombstone one Toolkit row.

        Args:
            toolkit_id (str): Stable Toolkit object id.
            tombstone (bool): Mark deleted instead of physically removing the row. Defaults to ``True``.

        Returns:
            bool: ``True`` when an existing row was deleted or tombstoned.
        """
        if not tombstone:
            return self.store.delete(toolkit_id)
        row = self.store.get(toolkit_id)
        if row is None:
            return False
        row["status"] = "deleted"
        row["updated_at"] = time()
        self.store.set(row)
        return True

    def _register_capsule_ref(
        self,
        tool: Any,
        capsule: Any,
        *,
        role: str,
        mode: str,
        reason: str,
        registered: Dict[str, Any],
    ) -> str:
        existing = registered.get(capsule.manifest.id)
        if existing is not None:
            if existing.manifest.fingerprints != capsule.manifest.fingerprints:
                raise ValueError(f"Toolkit includes different {role} Capsule content for shared id {capsule.manifest.id!r}")
            return capsule.manifest.id
        capability = capsule.manifest.capabilities.get("registry_persistence", {})
        if capability and not capability.get("ok", True):
            raise ValueError(f"Tool {tool.name!r} {role} cannot be registered as a Capsule: {capability.get('reasons')}")
        try:
            capsule_id = self.capsules.register(capsule, overwrite=mode == "overwrite", reason=reason)
        except ValueError:
            if mode == "skip":
                capsule_id = capsule.manifest.id
            else:
                raise
        registered[capsule_id] = capsule
        return capsule_id

    def _register_tool_capsules(self, tool: Any, *, mode: str, reason: str, registered: Dict[str, Any]) -> None:
        self._register_capsule_ref(tool, tool.capsule, role="main", mode=mode, reason=reason, registered=registered)
        self._register_capsule_ref(
            tool,
            tool._serializer_capsule(),
            role="serializer",
            mode=mode,
            reason=reason,
            registered=registered,
        )

    def register(
        self,
        toolkit: Any,
        *,
        overwrite: bool = True,
        on_conflict: Optional[str] = None,
        reason: str = "",
    ) -> str:
        """Register a Toolkit manifest and every referenced Capsule.

        Args:
            toolkit (Any): Runtime Toolkit value to persist.
            overwrite (bool): Whether same-id rows may be replaced. Defaults to ``True``.
            on_conflict (Optional[str]): Conflict mode: ``"overwrite"``, ``"raise"``, or ``"skip"``. Defaults to overwrite behavior from ``overwrite``.
            reason (str): Reason stored on the Toolkit row. Defaults to ``""``.

        Returns:
            str: Stable Toolkit id from the persisted manifest.

        Raises:
            TypeError: If ``toolkit`` is not a ``sys-toolkit`` value.
            ValueError: If conflict handling rejects the registration.
        """
        toolkit_schema = getattr(type(toolkit), "schema", None)
        if (
            not callable(toolkit_schema)
            or toolkit_schema().to_dict() != self.toolkit_type.schema().to_dict()
            or not callable(getattr(toolkit, "to_manifest", None))
            or not callable(getattr(toolkit, "to_row", None))
        ):
            raise TypeError(f"ToolkitRegistry can register sys-toolkit values, got {type(toolkit).__name__}")
        mode = on_conflict or ("overwrite" if overwrite else "raise")
        if mode not in {"overwrite", "raise", "skip"}:
            raise ValueError("on_conflict must be one of: overwrite, raise, skip")
        bind = getattr(toolkit, "_bind_authority", None)
        if callable(bind):
            bind(self.authority.resolver, self.authority.config)
        registered: Dict[str, Any] = {}
        for tool in toolkit.tools.values():
            self._register_tool_capsules(tool, mode=mode, reason=f"toolkit:{toolkit.name}", registered=registered)
        manifest = toolkit.to_manifest()
        existing = self.store.get(manifest["id"])
        if existing and mode == "raise" and existing.get("manifest") != manifest:
            raise ValueError(f"Toolkit {manifest['id']!r} already exists with different content")
        if existing and mode == "skip" and existing.get("manifest") != manifest:
            return manifest["id"]
        self.store.upsert(toolkit.to_row(existing, reason=reason))
        return manifest["id"]

    def _find(
        self,
        toolkit_id: Optional[str],
        *,
        name: Optional[str],
        namespace: str,
        version: Optional[str],
    ) -> Dict[str, Any] | None:
        if toolkit_id:
            return self.store.get(toolkit_id)
        matches = [row for row in self.store.rows() if row.get("name") == name and row.get("namespace") == namespace]
        return select_versioned_row(matches, version=version)

    def load(
        self,
        toolkit_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        namespace: str = "default",
        version: Optional[str] = None,
    ) -> Any:
        """Load a Toolkit and restore its referenced Capsules lazily.

        Args:
            toolkit_id (Optional[str]): Stable Toolkit object id. Defaults to ``None``.
            name (Optional[str]): Toolkit name when loading by identity. Defaults to ``None``.
            namespace (str): Toolkit namespace. Defaults to ``"default"``.
            version (Optional[str]): Explicit version or latest match when omitted. Defaults to ``None``.

        Returns:
            Any: Runtime Toolkit with tools restored from referenced Capsules.

        Raises:
            KeyError: If no matching Toolkit row exists.
            ValueError: If a stored manifest lacks required serializer Capsule refs.
        """
        row = self._find(toolkit_id, name=name, namespace=namespace, version=version)
        if row is None:
            raise KeyError(toolkit_id or f"{namespace}/{name}:{version or '*'}")
        toolkit = self.toolkit_type(
            row["name"],
            description=row.get("description") or "",
            namespace=row.get("namespace") or "default",
            version=row.get("version") or "1",
            resolver=self.authority.resolver,
            config=self.authority.config,
        )
        toolkit._workspace = self.workspace
        from .tool import Tool

        for ref in row.get("capsule_refs") or []:
            tool_label = ref.get("alias") or ref.get("name") or ref.get("capsule_id") or "<unknown>"
            if not ref.get("serializer_capsule_id"):
                raise ValueError(f"Toolkit manifest missing serializer_capsule_id for tool {tool_label!r}; re-register toolkit")
            capsule = self.capsules.load(ref["capsule_id"])
            serializer = self.capsules.load(ref["serializer_capsule_id"])
            tool = Tool(
                capsule,
                name=ref.get("alias") or capsule.manifest.name,
                description=ref.get("description") or capsule.manifest.description,
                input_schema=ref.get("input_schema"),
                output_schema=ref.get("output_schema"),
                annotations=ref.get("annotations"),
                metadata=ref.get("metadata"),
                serializer=serializer,
                config=self.authority.config,
            )
            toolkit.add(tool)
        return toolkit

    def describe(
        self,
        toolkit_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        namespace: str = "default",
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return one raw Toolkit registry row.

        Args:
            toolkit_id (Optional[str]): Stable Toolkit object id. Defaults to ``None``.
            name (Optional[str]): Toolkit name when loading by identity. Defaults to ``None``.
            namespace (str): Toolkit namespace. Defaults to ``"default"``.
            version (Optional[str]): Explicit version or latest match when omitted. Defaults to ``None``.

        Returns:
            Dict[str, Any]: Raw persisted Toolkit row.

        Raises:
            KeyError: If no matching Toolkit row exists.
        """
        row = self._find(toolkit_id, name=name, namespace=namespace, version=version)
        if row is None:
            raise KeyError(toolkit_id or f"{namespace}/{name}:{version or '*'}")
        return row


def toolkit_registry(workspace: Any = None) -> ToolkitRegistry:
    """Return a Toolkit registry bound to one existing workspace.

    Args:
        workspace (Any): Explicit workspace. When omitted,
            `HeavenBase.load()` resolves the active or configured default
            workspace without creating one.

    Returns:
        ToolkitRegistry: Workspace-bound Toolkit persistence service.
    """
    from ._workspace import resolve_toolkit_workspace

    return ToolkitRegistry.from_workspace(resolve_toolkit_workspace(workspace))
