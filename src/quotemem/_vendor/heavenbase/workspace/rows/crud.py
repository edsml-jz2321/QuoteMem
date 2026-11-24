"""Private CRUD orchestration for HeavenBase workspaces."""

from __future__ import annotations

from heavenbase.utils import Any, Dict, Iterable, List, Literal, Optional, Sequence
from heavenbase.utils.values import _type_label

from ..metaschema.entity import MetaSchema
from .inputs import RowPatch, UpsertRow
from .lookup import RowMatch

_UPSERT_FLAG_KEYWORDS = {
    "overwrite": "replace flag",
    "atomic": "one-transaction flag",
}


def _require_bool_flag(name: str, value: Any, reason: str = "") -> None:
    """Refuse a non-bool write flag.

    Args:
        name: Flag name used in the error message, ``overwrite`` or ``atomic``.
        value: Value received for the flag.
        reason: Optional explanation appended to the error message.

    Returns:
        None: A bool value returns without effect.

    Raises:
        TypeError: If ``value`` is not a bool.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {_type_label(value)}{reason}")


def _split_flag_keywords(entity_cls: type, values: Dict[str, Any], *, overwrite: bool) -> tuple[Dict[str, Any], bool, bool]:
    """Split the ``overwrite`` and ``atomic`` keywords of ``upsert`` into field values or write flags.

    A keyword whose name the Entity class declares as a field is that field's
    value, so it is never dropped; ``atomic`` keeps its 0.1.2.2 meaning there.
    Otherwise the keyword is the write flag of that name, which must be a bool.

    Args:
        entity_cls: Entity class the row resolved to.
        values: Keyword field values of one ``upsert`` call, possibly holding ``overwrite`` or ``atomic``.
        overwrite: Replace flag to use when no ``overwrite`` keyword is read as the flag.

    Returns:
        tuple[Dict[str, Any], bool, bool]: Field values to merge into the row, the
        replace flag, and the transaction flag.

    Raises:
        TypeError: If a keyword is read as a flag and is not a bool.
    """
    declared = entity_cls.schema().fields
    fields = dict(values)
    flags: Dict[str, bool] = {"overwrite": overwrite, "atomic": False}
    for name, meaning in _UPSERT_FLAG_KEYWORDS.items():
        if name not in fields or name in declared:
            continue
        value = fields.pop(name)
        _require_bool_flag(name, value, f"; {entity_cls.__name__} declares no field named {name!r}, so the upsert keyword is the {meaning}")
        flags[name] = value
    return fields, flags["overwrite"], flags["atomic"]


class _CrudService:
    """Own row-level CRUD flows while ``HeavenBase`` stays the public facade."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def upsert(
        self,
        entity: Any,
        row: Optional[Dict[str, Any]] = None,
        values: Optional[Dict[str, Any]] = None,
        *,
        overwrite: bool = True,
        allowed_entities: frozenset[str] | None = None,
    ) -> str:
        """Upsert one row; ``values`` are keyword field values, ``overwrite`` and ``atomic`` among them read by ``_split_flag_keywords``."""
        ws = self.workspace
        item = ws._row_inputs.upsert_row(entity, row, {})
        fields, overwrite, atomic = _split_flag_keywords(item.entity_cls, values or {}, overwrite=overwrite)
        item.data.update(fields)
        ws._schema.require(item.entity_cls)
        writer = ws._writer
        if atomic:
            writer.require_atomic_upsert([item.entity_cls])
        self._establish_metaschema_entities([item], allowed_entities=allowed_entities)
        plan = writer.prepare_upsert_rows([item], overwrite=overwrite)
        ws._graph_mutation.validate_upserts([item], allowed_entities=allowed_entities)
        writer.write_upsert(plan, atomic=atomic)
        self._invalidate_mcp_registry_metadata_if_needed(plan.entity_classes)
        return plan.object_ids[0]

    def upsert_many(
        self,
        entity: Any,
        rows: Iterable[Any],
        *,
        overwrite: bool = True,
        catalog_mode: Literal["merge", "fresh"] = "merge",
        atomic: bool = False,
        allowed_entities: frozenset[str] | None = None,
    ) -> List[str]:
        _require_bool_flag("atomic", atomic)
        ws = self.workspace
        writer = ws._writer
        items = ws._row_inputs.upsert_rows(entity, rows)
        for item in items:
            ws._schema.require(item.entity_cls)
        if atomic:
            writer.require_atomic_upsert([item.entity_cls for item in items])
        self._establish_metaschema_entities(items, allowed_entities=allowed_entities)
        plan = writer.prepare_upsert_rows(items, overwrite=overwrite, catalog_mode=catalog_mode)
        ws._graph_mutation.validate_upserts(items, allowed_entities=allowed_entities)
        writer.write_upsert(plan, atomic=atomic)
        self._invalidate_mcp_registry_metadata_if_needed(plan.entity_classes)
        return plan.object_ids

    def get(self, object_id: Any, entity: Any = None, *, unique: bool = True, columns: Sequence[str] | None = None) -> Optional[Dict[str, Any]]:
        match = self.workspace._row_lookup.id_match(object_id, entity=entity, unique=unique, columns=columns)
        return match.row if match is not None else None

    def get_many(
        self,
        object_ids: Iterable[Any],
        entity: Any = None,
        *,
        unique: bool = True,
        columns: Sequence[str] | None = None,
    ) -> List[Optional[Dict[str, Any]]]:
        matches = self.workspace._row_lookup.id_matches_many(object_ids, entity=entity, unique=unique, columns=columns)
        return [match.row if match is not None else None for match in matches]

    def read_rows(self, entity: Any, object_ids: Iterable[Any]) -> List[Optional[Dict[str, Any]]]:
        ws = self.workspace
        entity_cls = ws._schema.require(entity)
        return [match.row if match is not None else None for match in ws._row_lookup.entity_matches_many(entity_cls, object_ids)]

    def set(
        self,
        row: Any,
        values: Optional[Dict[str, Any]] = None,
        *,
        entity: Any = None,
        overwrite: bool = True,
        allowed_entities: frozenset[str] | None = None,
    ) -> str:
        patch = self.workspace._row_inputs.set_patch(row, values, entity=entity)
        return self._set_patches([patch], overwrite=overwrite, command="set", allowed_entities=allowed_entities)[0]

    def set_many(
        self,
        rows: Iterable[Any],
        *,
        entity: Any = None,
        overwrite: bool = True,
        allowed_entities: frozenset[str] | None = None,
    ) -> List[str]:
        patches = self.workspace._row_inputs.set_patches(rows, entity=entity)
        return self._set_patches(patches, overwrite=overwrite, command="set_many", allowed_entities=allowed_entities)

    def _set_patches(
        self,
        patches: Iterable[RowPatch],
        *,
        overwrite: bool,
        command: str,
        allowed_entities: frozenset[str] | None,
    ) -> List[str]:
        """Merge, materialize, and write normalized patches through one path."""
        ws = self.workspace
        patch_list = list(patches)
        object_ids_by_entity: Dict[type, List[str]] = {}
        for item in patch_list:
            object_ids_by_entity.setdefault(item.entity_cls, []).append(item.object_id)

        current_by_key: Dict[tuple[str, str], RowMatch | None] = {}
        for entity_cls, object_ids in object_ids_by_entity.items():
            schema = entity_cls.schema()
            for object_id, current in zip(object_ids, ws._row_lookup.entity_matches_many(entity_cls, object_ids, for_patch=True)):
                current_by_key[(schema.entity_id, object_id)] = current

        rows: List[UpsertRow] = []
        for item in patch_list:
            schema = item.entity_cls.schema()
            current = current_by_key[(schema.entity_id, item.object_id)]
            if current is not None and not overwrite:
                raise ValueError(f"{schema.entity_id} object_id {item.object_id!r} already exists")
            row = ws._row_inputs.patched_row(item.entity_cls, current, item.patch)
            data = row.data
            ws._materializer.materialize(item.entity_cls, data)
            written_id = ws._row_lookup.normalize_object_id(schema, data[schema.pk_name])
            if written_id != item.object_id:
                raise ValueError(f"{command} cannot move object_id {item.object_id!r} to {written_id!r}; use upsert and delete explicitly")
            rows.append(row)
        writer = ws._writer
        self._establish_metaschema_entities(rows, allowed_entities=allowed_entities)
        plan = writer.prepare_upsert_rows(rows, overwrite=True)
        ws._graph_mutation.validate_upserts(rows, allowed_entities=allowed_entities)
        writer.write_upsert(plan)
        self._invalidate_mcp_registry_metadata_if_needed(plan.entity_classes)
        return plan.object_ids

    def delete(
        self,
        target: Any,
        *,
        edges: str = "schema",
        allowed_entities: frozenset[str] | None = None,
    ) -> bool:
        deleted = self.delete_many([target], edges=edges, allowed_entities=allowed_entities)
        return bool(deleted and deleted[0])

    def delete_many(
        self,
        targets: Iterable[Any],
        *,
        edges: str = "schema",
        allowed_entities: frozenset[str] | None = None,
    ) -> List[bool]:
        ws = self.workspace
        resolved_list = [ws._row_inputs.delete_target(target) for target in targets]
        plan = ws._graph_mutation.delete_rows(resolved_list, edges=edges, allowed_entities=allowed_entities)
        if any(self._is_metaschema_entity(entity_cls) for entity_cls in plan.entity_classes):
            ws._metaschema.invalidate_readers()
        return plan.deleted

    def exists(self, object_id: Any, entity: Any = None, *, unique: bool = True) -> bool:
        return self.exists_many([object_id], entity=entity, unique=unique)[0]

    def exists_many(self, object_ids: Iterable[Any], entity: Any = None, *, unique: bool = True) -> List[bool]:
        return self.workspace._row_lookup.id_exists_many(object_ids, entity=entity, unique=unique)

    def count(self, entity: Any) -> int:
        ws = self.workspace
        entity_cls = ws._schema.require(entity)
        return ws._row_store.count(entity_cls.schema().entity_id)

    def _invalidate_mcp_registry_metadata_if_needed(self, entity_classes: Iterable[type]) -> None:
        if any(self._is_metaschema_entity(entity_cls) for entity_cls in entity_classes):
            self.workspace._metaschema.invalidate_readers()

    def _establish_metaschema_entities(
        self,
        rows: Iterable[UpsertRow],
        *,
        allowed_entities: frozenset[str] | None,
    ) -> None:
        """Route definition-shaped MetaSchema rows through schema authority."""
        meta_rows = [item.data for item in rows if self._is_metaschema_entity(item.entity_cls)]
        if meta_rows:
            self.workspace._metaschema.establish_entity_classes(meta_rows, allowed_entities=allowed_entities)

    def _is_metaschema_entity(self, entity_cls: type) -> bool:
        return entity_cls.schema().entity_id == MetaSchema.schema().entity_id
