"""Persistence port consumed by the Toolkit registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from heavenbase.utils import Any, Dict, List, Optional

__all__ = ["ToolkitRegistryPort", "ToolkitRowStore"]


@runtime_checkable
class ToolkitRowStore(Protocol):
    """Store rows for the optional ``sys-toolkit`` Entity."""

    def get(self, object_id: str) -> Optional[Dict[str, object]]:
        """Return one Toolkit row by object id.

        Args:
            object_id (str): Stable Toolkit object id.

        Returns:
            Optional[Dict[str, object]]: Matching row, or ``None`` when absent.
        """
        ...

    def rows(self) -> List[Dict[str, object]]:
        """Return all Toolkit rows.

        Args:
            None.

        Returns:
            List[Dict[str, object]]: All persisted Toolkit rows.
        """
        ...

    def upsert(self, row: Dict[str, object]) -> None:
        """Insert or replace one Toolkit row.

        Args:
            row (Dict[str, object]): Toolkit row payload to persist.

        Returns:
            None: This function does not return a value.
        """
        ...

    def set(self, row: Dict[str, object]) -> None:
        """Replace one existing Toolkit row.

        Args:
            row (Dict[str, object]): Toolkit row payload that must already exist.

        Returns:
            None: This function does not return a value.
        """
        ...

    def delete(self, object_id: str) -> bool:
        """Delete one Toolkit row by object id.

        Args:
            object_id (str): Stable Toolkit object id.

        Returns:
            bool: ``True`` when an existing row was deleted.
        """
        ...


@runtime_checkable
class ToolkitRegistryPort(Protocol):
    """Structural Toolkit persistence authority.

    Module artifacts intentionally load extension classes in an isolated
    package namespace. The protocol keeps source-tree callers and canonical
    artifact-loaded Toolkit values interoperable without treating Python class
    identity as Registry identity.
    """

    store: ToolkitRowStore
    capsules: Any
    authority: Any

    def register(
        self,
        toolkit: Any,
        *,
        overwrite: bool = True,
        on_conflict: Optional[str] = None,
        reason: str = "",
    ) -> str:
        """Register one Toolkit value.

        Args:
            toolkit (Any): Runtime Toolkit value to persist.
            overwrite (bool): Whether same-id rows may be replaced. Defaults to ``True``.
            on_conflict (Optional[str]): Conflict mode: ``"overwrite"``, ``"raise"``, or ``"skip"``.
            reason (str): Reason stored on the Toolkit row. Defaults to ``""``.

        Returns:
            str: Stable Toolkit id from the persisted manifest.
        """
        ...

    def load(
        self,
        toolkit_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        namespace: str = "default",
        version: Optional[str] = None,
    ) -> Any:
        """Load one Toolkit value.

        Args:
            toolkit_id (Optional[str]): Stable Toolkit object id. Defaults to ``None``.
            name (Optional[str]): Toolkit name when loading by identity. Defaults to ``None``.
            namespace (str): Toolkit namespace. Defaults to ``"default"``.
            version (Optional[str]): Explicit version or latest match when omitted. Defaults to ``None``.

        Returns:
            Any: Runtime Toolkit with tools restored from referenced Capsules.
        """
        ...

    def list(self, namespace: Optional[str] = None, status: Optional[str] = "active") -> List[Dict[str, object]]:
        """List Toolkit rows.

        Args:
            namespace (Optional[str]): Optional namespace filter. Defaults to no namespace filtering.
            status (Optional[str]): Optional status filter. Defaults to ``"active"``; pass ``None`` to disable status filtering.

        Returns:
            List[Dict[str, object]]: Matching Toolkit rows.
        """
        ...

    def delete(self, toolkit_id: str, *, tombstone: bool = True) -> bool:
        """Delete or tombstone one Toolkit row.

        Args:
            toolkit_id (str): Stable Toolkit object id.
            tombstone (bool): Mark deleted instead of physically removing the row. Defaults to ``True``.

        Returns:
            bool: ``True`` when an existing row was deleted or tombstoned.
        """
        ...

    def describe(
        self,
        toolkit_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        namespace: str = "default",
        version: Optional[str] = None,
    ) -> Dict[str, object]:
        """Describe one Toolkit row.

        Args:
            toolkit_id (Optional[str]): Stable Toolkit object id. Defaults to ``None``.
            name (Optional[str]): Toolkit name when loading by identity. Defaults to ``None``.
            namespace (str): Toolkit namespace. Defaults to ``"default"``.
            version (Optional[str]): Explicit version or latest match when omitted. Defaults to ``None``.

        Returns:
            Dict[str, object]: Raw persisted Toolkit row.
        """
        ...
