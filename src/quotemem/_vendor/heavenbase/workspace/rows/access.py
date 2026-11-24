"""Workspace row listing and Python collection-style access service."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, Iterator, List, Mapping

from ...entity import Entity


class _RowAccessService:
    """Own row listing helpers and Python collection protocol adapters."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def ids(self, entity: Any = None) -> List[Any]:
        """Return object ids for one entity, or the sole user entity."""
        ws = self.workspace
        entity_cls = self.single_user_entity() if entity is None else ws._schema.require(entity)
        return list(ws._row_store.universe(entity_cls.schema().entity_id).ids)

    def rows(self, entity: Any = None) -> List[Dict[str, Any]]:
        """Return rows for one entity, or the sole user entity."""
        ws = self.workspace
        entity_cls = self.single_user_entity() if entity is None else ws._schema.require(entity)
        return ws._row_store.universe(entity_cls.schema().entity_id).rows()

    def items(self, entity: Any = None) -> List[tuple[Any, Dict[str, Any]]]:
        """Return ``(object_id, row)`` pairs for one entity."""
        return [(row.get("object_id"), row) for row in self.rows(entity)]

    def row_pages(self, entity: Any, *, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Return the bounded decoded-row pages of one present entity.

        The entity and page size are validated before the page iterator is
        returned, so an unknown class or bad size fails at the call rather than
        on the first page.
        """
        ws = self.workspace
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        entity_cls = ws._schema.require(entity)
        return ws._row_store.iter_row_pages(entity_cls.schema().entity_id, batch_size=batch_size)

    def get_item(self, key: Any) -> Dict[str, Any]:
        """Return one row for Python item access."""
        entity, object_id = self.operator_target(key)
        row = self.workspace._crud.get(object_id, entity=entity)
        if row is None:
            raise KeyError(object_id)
        return row

    def set_item(self, key: Any, value: Any) -> None:
        """Set one row for Python item assignment."""
        entity, object_id = self.operator_target(key)
        if isinstance(value, Entity):
            entity = entity or type(value)
            data = value.to_dict()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError("Workspace item assignment needs an Entity or row mapping")
        self.apply_operator_id(data, object_id)
        self.workspace._crud.set(data, entity=entity)

    def delete_item(self, key: Any) -> None:
        """Delete one row for Python item deletion."""
        entity, object_id = self.operator_target(key)
        target = (entity, object_id) if entity is not None else object_id
        if not self.workspace._crud.delete(target):
            raise KeyError(object_id)

    def contains_item(self, key: Any) -> bool:
        """Return whether a Python item-access key resolves to an existing row."""
        try:
            entity, object_id = self.operator_target(key)
            return self.workspace._crud.exists(object_id, entity=entity, unique=False)
        except Exception:
            return False

    def iter_ids(self):
        """Return an iterator over the sole user entity's object ids."""
        return iter(self.ids())

    def len_user_entity(self) -> int:
        """Return row count for the sole non-system entity."""
        return self.workspace._crud.count(self.single_user_entity())

    def operator_target(self, key: Any) -> tuple[Any, Any]:
        """Normalize Python item-access keys into optional entity and object_id."""
        if isinstance(key, (tuple, list)) and len(key) == 2:
            return key[0], key[1]
        return None, key

    def apply_operator_id(self, data: Dict[str, Any], object_id: Any) -> None:
        """Inject item-assignment object_id and reject conflicting aliases."""
        if data.get("object_id") not in (None, "", object_id):
            raise ValueError(f"Workspace item assignment object_id {object_id!r} conflicts with row object_id={data['object_id']!r}")
        data.setdefault("object_id", object_id)

    def single_user_entity(self) -> type:
        """Return the sole non-system entity for entity-omitted collection calls."""
        ws = self.workspace
        entity_ids = ws._extension_service.list_entities()
        if len(entity_ids) != 1:
            raise TypeError("Workspace iteration/len/rows without entity needs exactly one non-system entity; pass entity=... explicitly")
        return ws.entities[entity_ids[0]]
