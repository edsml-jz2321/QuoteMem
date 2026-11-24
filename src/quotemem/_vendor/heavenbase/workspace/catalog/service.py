"""Private Catalog object-row lifecycle service for HeavenBase workspaces."""

from __future__ import annotations

from collections.abc import Mapping
from time import time

from heavenbase.utils import Any, Dict, copy_data, deepcopy

from ..foundation import FIXED_ENTITY_IDS
from .entity import Catalog
from .rows import (
    CATALOG_AUTO_FIELDS,
    CATALOG_DEFAULTS,
    build_catalog_row,
    catalog_audit_timestamps,
    catalog_materialize_timing,
    catalog_row_id,
    catalog_write_timestamps,
)


class _CatalogService:
    """Own Catalog object rows, signatures, derives, and row identity."""

    def __init__(self, workspace: Any):
        self.workspace = workspace
        self.signature: Dict[str, Any] = {}

    def sign(
        self,
        *,
        owner: str | None = None,
        creator: str | None = None,
        auth: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Set default Catalog ownership fields for future entity writes."""
        schema = Catalog.schema()
        updates = {}
        if owner is not None:
            updates["owner"] = schema.get("owner").dtype.validate(owner)
        if creator is not None:
            updates["creator"] = schema.get("creator").dtype.validate(creator)
        if auth is not None:
            updates["auth"] = schema.get("auth").dtype.validate(deepcopy(dict(auth)))
        if metadata is not None:
            updates["metadata"] = schema.get("metadata").dtype.validate(deepcopy(dict(metadata)))
        self.signature.update(updates)
        return self.workspace

    def signature_dict(self) -> Dict[str, Any]:
        """Return the default Catalog signature applied to future writes."""
        return deepcopy(self.signature)

    def validate_derives(self, entity: type) -> None:
        """Validate Catalog-specific derive targets for an entity."""
        catalog_fields = set(Catalog.schema().fields)
        for target in getattr(entity, "_hb_derives", {}):
            if target.startswith("catalog."):
                field_name = target.split(".", 1)[1]
                if field_name in CATALOG_AUTO_FIELDS:
                    raise ValueError(f"Catalog field {field_name!r} is auto-generated and cannot be derived")
                if field_name not in catalog_fields:
                    raise ValueError(f"Catalog has no field {field_name!r}")

    def derives(self, entity: type) -> Dict[str, Any]:
        """Return Catalog derives keyed by concrete Catalog field name."""
        return {target.split(".", 1)[1]: derive for target, derive in getattr(entity, "_hb_derives", {}).items() if target.startswith("catalog.")}

    def now(self) -> int:
        """Return the current Catalog timestamp in milliseconds."""
        return int(time() * 1000)

    def hash_length(self) -> int:
        """Return the configured deterministic identifier hash length."""
        return int(self.workspace.context.config.get("heavenbase.entity.id.hash_length", default=42))

    def row_id(self, schema: Any, object_id: str, *, hash_length: int | None = None) -> str:
        """Return the Catalog primary key for a concrete entity instance."""
        max_chars = Catalog.schema().get("object_id").dtype.max_chars
        resolved_hash_length = self.hash_length() if hash_length is None else int(hash_length)
        return catalog_row_id(self.workspace.id, schema.entity_id, object_id, max_chars, hash_length=resolved_hash_length)

    def mode(self, entity_id: str) -> str | None:
        """Return the Catalog mode of one entity, or ``None`` for a system entity.

        Args:
            entity_id (str): Registered entity identifier.

        Returns:
            str | None: ``"full"``, ``"identity"``, or ``"none"`` for a user
            entity; ``None`` for Catalog and MetaSchema themselves.
        """
        if entity_id in self.system_entity_ids():
            return None
        options = self.workspace.entity_options.get(entity_id)
        return options.catalog if options is not None else "full"

    def row(
        self,
        entity: type,
        schema: Any,
        object_id: str,
        data: Dict[str, Any],
        *,
        existing: Dict[str, Any] | None = None,
        hash_length: int | None = None,
        mode: str = "full",
    ) -> dict:
        """Build the Catalog row for a concrete entity instance.

        Args:
            entity (type): Registered entity class.
            schema (Any): Entity schema.
            object_id (str): Normalized entity ``object_id``.
            data (Dict[str, Any]): Materialized logical row.
            existing (Dict[str, Any] | None): Stored Catalog row whose lifecycle
                fields (``created_at``, ``active``, ``visible``) the new row keeps
                in either mode; ``updated_at`` is always the write time.
            hash_length (int | None): Identifier hash length override.
            mode (str): ``"full"`` projects, signs, and derives; ``"identity"``
                keeps only identity, the preserved lifecycle fields, and field
                defaults, merging nothing else from ``existing``.

        Returns:
            dict: Materialized Catalog row.
        """
        row = self._base_row(schema, object_id, hash_length)
        row.update(catalog_write_timestamps(existing, self.now()))
        if mode == "identity":
            row.update(catalog_materialize_timing(row, resolver=self.workspace.module_resolver))
        else:
            self.apply_derives(entity, data, row)
        self.workspace._materializer.materialize(Catalog, row)
        return row

    def audit_row(
        self,
        entity: type,
        schema: Any,
        object_id: str,
        data: Dict[str, Any],
        existing: Dict[str, Any] | None,
        *,
        hash_length: int | None = None,
        mode: str = "full",
    ) -> dict:
        """Build an expected Catalog row without changing existing timestamps."""
        row = self._base_row(schema, object_id, hash_length)
        row.update(catalog_audit_timestamps(existing, self.now()))
        if mode == "identity":
            row.update(catalog_materialize_timing(row, resolver=self.workspace.module_resolver))
        else:
            self.apply_derives(entity, data, row)
        self.workspace._materializer.materialize(Catalog, row)
        return row

    @staticmethod
    def target_id_limit() -> int:
        """Return the longest ``object_id`` a Catalog row's ``target_id`` holds."""
        return Catalog.schema().get("target_id").dtype.max_chars

    def _base_row(self, schema: Any, object_id: str, hash_length: int | None) -> dict:
        """Return the identity and default fields shared by every Catalog row shape.

        Raises:
            ValueError: If ``object_id`` is longer than ``Catalog.target_id`` can hold.
        """
        target_limit = self.target_id_limit()
        if len(object_id) > target_limit:
            raise ValueError(
                f"Entity {schema.entity_id!r} object_id has {len(object_id)} characters, but its Catalog row's target_id holds at most "
                f"{target_limit}; register the entity with catalog='none' to store longer ids"
            )
        max_chars = Catalog.schema().get("object_id").dtype.max_chars
        resolved_hash_length = self.hash_length() if hash_length is None else int(hash_length)
        return build_catalog_row(self.workspace.id, schema, object_id, max_chars, hash_length=resolved_hash_length)

    def apply_derives(self, entity: type, data: Dict[str, Any], row: Dict[str, Any]) -> None:
        """Apply Catalog defaults and user-defined derives to a Catalog row."""
        self.apply_signature(row)
        self.apply_same_name_derives(entity, data, row)
        for field_name, derive in self.derives(entity).items():
            row[field_name] = self.workspace._materializer.derive_value(f"catalog.{field_name}", derive, data)
        row.update(catalog_materialize_timing(row, resolver=self.workspace.module_resolver))

    def apply_signature(self, row: Dict[str, Any]) -> None:
        """Apply workspace-level Catalog ownership defaults."""
        for field_name, value in self.signature.items():
            if field_name in CATALOG_DEFAULTS:
                row[field_name] = self.copy_value(value)

    def apply_same_name_derives(self, entity: type, data: Dict[str, Any], row: Dict[str, Any]) -> None:
        """Project structurally matching native fields when no explicit derive exists."""
        explicit = set(self.derives(entity))
        catalog_schema = Catalog.schema()
        native_schema = entity.schema()
        native_fields = set(native_schema.fields)
        for field_name in sorted(CATALOG_DEFAULTS):
            if field_name in explicit or field_name not in native_fields or field_name not in data:
                continue
            if native_schema.get(field_name).dtype.to_dict() != catalog_schema.get(field_name).dtype.to_dict():
                continue
            value = data[field_name]
            if not self.empty_value(value):
                row[field_name] = self.copy_value(value)

    def copy_value(self, value: Any) -> Any:
        """Return a Catalog-safe copy for projected mutable values."""
        if isinstance(value, tuple):
            return [self.copy_value(item) for item in value]
        return copy_data(value)

    def empty_value(self, value: Any) -> bool:
        """Return whether a projected Catalog value should leave the default intact."""
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        if isinstance(value, (list, tuple, dict, set)):
            return not value
        return False

    def name_value(self, entity: type, data: Dict[str, Any]) -> Any:
        """Return the Catalog-derived name used as a fallback object_id."""
        derive = self.derives(entity).get("name")
        value = (
            self.workspace._materializer.derive_value("catalog.name", derive, data)
            if derive is not None
            else data.get("name") if "name" in entity.schema().fields else None
        )
        if value in (None, ""):
            return None
        value = Catalog.schema().get("name").dtype.validate(value)
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def system_entity_ids(self) -> set[str]:
        """Return internal entity ids reserved from user catalog behavior."""
        return set(FIXED_ENTITY_IDS)
