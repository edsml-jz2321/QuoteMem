"""Workspace adapter for domain-owned Entity row-store ports."""

from __future__ import annotations

__all__ = ["WorkspaceEntityRowStore"]

from collections.abc import Callable

from heavenbase.utils import Any, Dict, List, Optional


class WorkspaceEntityRowStore:
    """Bind one domain row-store port to an explicit workspace and Entity.

    Args:
        workspace (Any | Callable[[], Any]): Workspace facade or deferred
            workspace resolver.
        entity (type): Exact Entity class stored through the adapter.
        extension (str): Extension that owns the Entity schema.
        owner (Any | None): Optional lifecycle owner exposed to infrastructure
            caches. Defaults to the workspace value.
    """

    def __init__(
        self,
        workspace: Any | Callable[[], Any],
        entity: type,
        *,
        extension: str,
        owner: Any | None = None,
    ) -> None:
        """Bind one domain row-store port to an explicit workspace and Entity.

        Returns:
            None: This initializer does not return a value.
        """
        self._workspace_source = workspace
        self._entity = entity
        self._extension = extension
        self._owner = workspace if owner is None else owner

    @property
    def owner(self) -> Any:
        """Return the adapter lifecycle owner.

        Args:
            None.

        Returns:
            Any: Explicit owner supplied during construction.
        """
        return self._owner

    def _workspace(self) -> Any:
        """Return the workspace and ensure the owning Extension is active."""
        source = self._workspace_source
        workspace = source() if callable(source) else source
        entity_id = self._entity.schema().entity_id
        if entity_id not in workspace.entities:
            workspace.enable_extension(self._extension)
        return workspace

    def get(self, object_id: str) -> Optional[Dict[str, object]]:
        """Return one logical Entity row.

        Args:
            object_id (str): Stable row identifier.

        Returns:
            Optional[Dict[str, object]]: Matching row, or `None`.
        """
        return self._workspace().get(object_id, entity=self._entity)

    def rows(self) -> List[Dict[str, object]]:
        """Return every logical Entity row.

        Args:
            None.

        Returns:
            List[Dict[str, object]]: Rows in workspace order.
        """
        return self._workspace().rows(self._entity)

    def upsert(self, row: Dict[str, object]) -> None:
        """Insert or replace one logical Entity row.

        Args:
            row (Dict[str, object]): Entity row containing `object_id`.

        Returns:
            None: This method writes the row to the workspace.
        """
        self._workspace().upsert(self._entity, row)

    def set(self, row: Dict[str, object]) -> None:
        """Replace one existing logical Entity row.

        Args:
            row (Dict[str, object]): Complete replacement row.

        Returns:
            None: This method writes the replacement.
        """
        self._workspace().set(row, entity=self._entity)

    def delete(self, object_id: str) -> bool:
        """Delete one logical Entity row.

        Args:
            object_id (str): Stable row identifier.

        Returns:
            bool: Whether the row existed.
        """
        return self._workspace().delete((self._entity, object_id))
