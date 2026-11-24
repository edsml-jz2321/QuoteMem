"""Persistence port consumed by the Capsule registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from heavenbase.utils import Dict, List, Optional

__all__ = ["CapsuleRowStore"]


@runtime_checkable
class CapsuleRowStore(Protocol):
    """Store rows for the universal ``sys-capsule`` Entity.

    The Capsule domain owns this narrow port. Workspace construction and
    Backend selection belong to the adapter supplied by the Context
    composition root.
    """

    def get(self, object_id: str) -> Optional[Dict[str, object]]:
        """Return one Capsule row by object id.

        Args:
            object_id (str): Stable Capsule row identifier.

        Returns:
            Dict[str, object] | None: Matching row, or ``None`` when absent.
        """
        ...

    def rows(self) -> List[Dict[str, object]]:
        """Return all Capsule rows.

        Args:
            None.

        Returns:
            List[Dict[str, object]]: All persisted Capsule rows.
        """
        ...

    def upsert(self, row: Dict[str, object]) -> None:
        """Insert or replace one Capsule row.

        Args:
            row (Dict[str, object]): Row payload to insert or replace.

        Returns:
            None: This method does not return a value.
        """
        ...

    def set(self, row: Dict[str, object]) -> None:
        """Replace one existing Capsule row.

        Args:
            row (Dict[str, object]): Existing row payload to replace.

        Returns:
            None: This method does not return a value.
        """
        ...

    def delete(self, object_id: str) -> bool:
        """Delete one Capsule row by object id.

        Args:
            object_id (str): Stable Capsule row identifier.

        Returns:
            bool: Whether a row was deleted.
        """
        ...
