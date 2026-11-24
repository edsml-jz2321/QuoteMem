"""Workspace row command input normalization."""

from __future__ import annotations

__all__ = [
    "DeleteTarget",
    "RowPatch",
    "UpsertRow",
]

from heavenbase.utils import Any, Dict, Iterable, List, Mapping, Optional, Sequence, dataclass

from ...entity import Entity
from .lookup import RowMatch


@dataclass
class UpsertRow:
    """Carry one normalized logical row into upsert planning.

    Args:
        entity_cls (type): Entity class that owns the row schema.
        data (Dict[str, Any]): Mutable logical row data to materialize and route.
    """

    entity_cls: type
    data: Dict[str, Any]

    @classmethod
    def from_data(cls, entity_cls: type, data: Mapping[str, Any]) -> "UpsertRow":
        """Build an upsert row from normalized logical data.

        Args:
            entity_cls (type): Entity class that owns the row schema.
            data (Mapping[str, Any]): Logical row data before materialization.

        Returns:
            UpsertRow: Command value with a mutable top-level data copy.
        """
        return cls(entity_cls, dict(data))


@dataclass
class RowPatch:
    """Carry one normalized target and patch into set merge planning.

    Args:
        entity_cls (type): Entity class that owns the target row.
        patch (Dict[str, Any]): Mutable logical patch with normalized identity.
        object_id (str): Normalized target object identifier.
    """

    entity_cls: type
    patch: Dict[str, Any]
    object_id: str

    @classmethod
    def from_data(cls, entity_cls: type, patch: Mapping[str, Any], object_id: str) -> "RowPatch":
        """Build a row patch from normalized target data.

        Args:
            entity_cls (type): Entity class that owns the target row.
            patch (Mapping[str, Any]): Logical patch after identity normalization.
            object_id (str): Normalized target object identifier.

        Returns:
            RowPatch: Command value with a mutable top-level patch copy.
        """
        return cls(entity_cls, dict(patch), object_id)


@dataclass(frozen=True)
class DeleteTarget:
    """Identify one normalized workspace row to delete.

    Args:
        entity_cls (type): Registered entity class that owns the row.
        object_id (str): Normalized object identifier to delete.
    """

    entity_cls: type
    object_id: str


class _RowInputService:
    """Normalize flexible public row inputs into workspace command values."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def upsert_row(self, entity: Any, row: Optional[Dict[str, Any]], values: Dict[str, Any]) -> UpsertRow:
        """Normalize one flexible upsert input into a logical row command."""
        ws = self.workspace
        if isinstance(entity, Entity):
            entity_cls = type(entity)
            data = entity.to_dict()
            if row:
                data.update(row)
        else:
            entity_cls = type(row) if isinstance(row, Entity) else ws._schema.entity_cls(entity)
            data = row.to_dict() if isinstance(row, Entity) else self.row_mapping(row)
        data.update(values)
        return UpsertRow.from_data(entity_cls, data)

    def upsert_rows(self, entity: Any, rows: Iterable[Any]) -> List[UpsertRow]:
        """Normalize a public upsert batch into logical row commands."""
        default_entity_cls = self.workspace._schema.require(entity)
        return [self.upsert_row(row, None, {}) if isinstance(row, Entity) else self.upsert_row(default_entity_cls, row, {}) for row in rows]

    def row_mapping(self, row: Any) -> Dict[str, Any]:
        """Return mutable row data from a mapping or record-shaped object."""
        if row is None:
            return {}
        if isinstance(row, Mapping):
            return dict(row)
        from ...interop import row_from_any

        return row_from_any(row)

    def set_patch(self, row: Any, values: Optional[Dict[str, Any]], *, entity: Any = None) -> RowPatch:
        """Normalize one flexible set input into a targeted row patch."""
        if isinstance(row, Entity):
            entity_cls = type(row)
            patch = row.to_dict()
            patch.update(values or {})
        elif isinstance(row, Mapping):
            patch = dict(row)
            patch.update(values or {})
            entity_cls = self.entity_for_patch(patch, entity=entity)
        else:
            if values is None:
                raise TypeError("set(object_id, values, entity=...) needs a values dict")
            patch = dict(values)
            patch.setdefault("object_id", row)
            entity_cls = self.entity_for_patch(patch, entity=entity)
        return self._row_patch(entity_cls, patch, command="set")

    def set_patches(self, rows: Iterable[Any], *, entity: Any = None) -> List[RowPatch]:
        """Normalize a public set batch into targeted row patches."""
        row_list = list(rows)
        entity_classes = self.entities_for_patches(row_list, entity=entity)
        return [
            self._row_patch(entity_cls, row.to_dict() if isinstance(row, Entity) else dict(row), command="set_many")
            for entity_cls, row in zip(entity_classes, row_list)
        ]

    def _row_patch(self, entity_cls: type, patch: Dict[str, Any], *, command: str) -> RowPatch:
        """Resolve the owning Entity class and normalize one required patch target."""
        ws = self.workspace
        entity_cls = ws._schema.require(entity_cls)
        schema = entity_cls.schema()
        ws._materializer.normalize_input_id(schema, patch)
        if schema.pk_name not in patch or patch.get(schema.pk_name) in (None, ""):
            raise ValueError(f"{command} needs object_id in each row patch" if command == "set_many" else "set needs object_id in the object or patch")
        object_id = ws._row_lookup.normalize_object_id(schema, patch[schema.pk_name])
        patch[schema.pk_name] = object_id
        return RowPatch.from_data(entity_cls, patch, object_id)

    def patched_row(self, entity_cls: type, source: RowMatch | None, patch: Dict[str, Any]) -> UpsertRow:
        """Merge a patch over its source and prepare any derives for recomputation."""
        data = {key: value for key, value in (source.row if source is not None else {}).items() if key != "object_id"}
        patch_data = dict(patch)
        patch_fields = set(patch_data)
        for name, derive in self.workspace._materializer.native_derives(entity_cls).items():
            dependencies = {item for hook in derive.computes for item in hook.inputs}
            full_row_compute = any(not hook.inputs for hook in derive.computes)
            dependencies_changed = bool(dependencies & patch_fields) or (full_row_compute and bool(patch_fields - {name}))
            if name not in patch_fields or dependencies_changed:
                data.pop(name, None)
            if dependencies_changed:
                patch_data.pop(name, None)
        data.update(patch_data)
        return UpsertRow(entity_cls, data)

    def entity_for_patch(self, patch: Dict[str, Any], *, entity: Any = None) -> type:
        """Resolve a patch entity explicitly or from an existing object id."""
        ws = self.workspace
        if entity is not None:
            return ws._schema.entity_cls(entity)
        raw_id = patch.get("object_id")
        if raw_id not in (None, ""):
            match = ws._row_lookup.id_match(raw_id, for_mutation=True)
            if match is not None:
                return match.entity_cls
        raise TypeError("set with a plain dict for a new object_id needs entity=...")

    def entities_for_patches(self, rows: Sequence[Any], *, entity: Any = None) -> List[type]:
        """Resolve one registered entity class per batch patch."""
        ws = self.workspace
        if entity is not None:
            return [ws._schema.require(entity)] * len(rows)
        result: List[Optional[type]] = [None] * len(rows)
        unresolved: List[tuple[int, str]] = []
        for pos, row in enumerate(rows):
            if isinstance(row, Entity):
                result[pos] = type(row)
            elif isinstance(row, Mapping):
                if row.get("entity") is not None:
                    result[pos] = ws._schema.entity_cls(row["entity"])
                else:
                    raw_id = row.get("object_id")
                    if raw_id not in (None, ""):
                        unresolved.append((pos, raw_id))
                    else:
                        raise TypeError("set with a plain dict for a new object_id needs entity=...")
            else:
                raise TypeError("set_many expects Entity instances or row mappings")
        if unresolved:
            object_ids = [raw_id for _, raw_id in unresolved]
            matches = ws._row_lookup.id_matches_many(object_ids, unique=False, for_mutation=True)
            for (pos, raw_id), match in zip(unresolved, matches):
                if match is None:
                    raise TypeError(f"set with a plain dict for object_id {raw_id!r} needs entity=...")
                result[pos] = match.entity_cls
        if any(entity_cls is None for entity_cls in result):
            raise RuntimeError("set_many entity resolution left an unresolved patch")
        return [ws._schema.require(entity_cls) for entity_cls in result if entity_cls is not None]

    def delete_target(self, target: Any) -> Optional[DeleteTarget]:
        """Resolve a flexible public delete input into a normalized target."""
        ws = self.workspace
        entity_cls = None
        raw_id = None
        if isinstance(target, Entity):
            entity_cls = type(target)
            raw_id = target.to_dict().get("object_id")
        elif isinstance(target, Mapping):
            raw_id = target.get("object_id")
            if target.get("entity") is not None:
                entity_cls = ws._schema.entity_cls(target["entity"])
        elif isinstance(target, (tuple, list)) and len(target) == 2:
            entity_cls = ws._schema.entity_cls(target[0])
            raw_id = target[1]
        else:
            match = ws._row_lookup.id_match(target, for_mutation=True)
            return DeleteTarget(match.entity_cls, match.object_id) if match is not None else None
        if raw_id in (None, ""):
            raise ValueError("delete target needs object_id")
        if entity_cls is None:
            match = ws._row_lookup.id_match(raw_id, for_mutation=True)
            return DeleteTarget(match.entity_cls, match.object_id) if match is not None else None
        entity_cls = ws._schema.require(entity_cls)
        object_id = ws._row_lookup.normalize_object_id(entity_cls.schema(), raw_id)
        return DeleteTarget(entity_cls, object_id)
