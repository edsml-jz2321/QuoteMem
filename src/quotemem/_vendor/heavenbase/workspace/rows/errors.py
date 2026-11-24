"""Logical-row materialization errors."""

from __future__ import annotations

__all__ = ["SchemaVersionMismatchError"]


class SchemaVersionMismatchError(ValueError):
    """Stored row is missing fields required by the current entity schema.

    Args:
        entity_id: Entity type identifier such as ``product``.
        object_id: Row identity value.
        fields: Schema field names still absent from the stored row.
    """

    def __init__(self, entity_id: str, object_id: str, fields: list[str]):
        """Initialize one schema-version mismatch error.

        Returns:
            None: This initializer does not return a value.
        """
        self.entity_id = entity_id
        self.object_id = object_id
        self.fields = list(fields)
        super().__init__(
            f"{entity_id} object_id {object_id!r} predates current schema fields {fields!r}; "
            "re-upsert affected rows or run ws.repair() for system catalog rows"
        )
