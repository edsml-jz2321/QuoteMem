"""Workspace schema storage placement service."""

from __future__ import annotations

from collections.abc import Mapping

from heavenbase.utils import Any, Dict, Iterable, List, loads_json

from ...entity import Entity
from ...entity.base import EntityMeta
from ...strategies import GraphEdge, InlineColumn, SparseGramIndex, StorageStrategy
from ...strategies.graph_edge import resolve_graph_endpoint
from ...placement import StorageBinding, StoragePlan, StorageProfile, StorageRule
from ...placement.profiles import _registered_profiles
from ...placement.index import IndexResolver
from ..metaschema.entity import MetaSchema


def _backend_tag(backend: Any, identifier: str) -> Any:
    """Return one effective Backend tag value or `None`."""
    result = backend.tag(identifier)
    return result.value if result is not None else None


class _StorageProfileService:
    """Own effective storage profiles, including workspace-owned MetaSchema overrides."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def effective(self, dtypes: Iterable[str] | None = None) -> Dict[str, StorageProfile]:
        """Return effective storage profiles for this workspace."""
        # Shared registered profiles, read only during planning, then this workspace's overrides.
        profiles = dict(_registered_profiles(self.workspace.module_resolver))
        profiles.update(self.workspace_overrides(dtypes))
        return profiles

    def workspace_overrides(self, dtypes: Iterable[str] | None = None) -> Dict[str, StorageProfile]:
        """Return workspace-owned storage profile overrides from MetaSchema rows."""
        rows = self.rows(dtypes)
        if not rows:
            return {}
        profiles: Dict[str, StorageProfile] = {}
        for row in rows:
            if row.get("kind") != "storage_profile" or row.get("active") is False:
                continue
            meta = row.get("meta") if isinstance(row.get("meta"), Mapping) else {}
            if row.get("source") != "workspace" and meta.get("source") != "workspace":
                continue
            dtype = row.get("dtype") or row.get("subject_id") or row.get("name")
            row_id = row.get("object_id", dtype)
            try:
                profile = StorageProfile.from_mapping(dtype, row, resolver=self.workspace.module_resolver)
                profiles[profile.identifier] = profile
            except Exception as exc:
                raise ValueError(f"Invalid workspace storage_profile row {row_id!r}: {exc}") from exc
        return profiles

    def rows(self, dtypes: Iterable[str] | None = None) -> List[Dict[str, Any]]:
        """Return raw candidate MetaSchema storage-profile rows, read by id from every readable MetaSchema route."""
        ws = self.workspace
        entity_id = MetaSchema.schema().entity_id
        if entity_id not in ws.entities or entity_id not in ws.storage_plans:
            return []
        dtype_keys = self.candidate_dtypes(dtypes)
        if not dtype_keys:
            return []
        row_ids = [ws._metaschema.system_row_id("storage_profile", dtype) for dtype in dtype_keys]
        return [self.normalized_row(row) for row in ws._metaschema.stored_rows(row_ids).values()]

    def candidate_dtypes(self, dtypes: Iterable[str] | None = None) -> List[str]:
        """Return logical type identifiers worth checking for workspace overrides."""
        if dtypes is not None:
            return sorted({str(dtype) for dtype in dtypes if dtype})
        resolver = self.workspace.module_resolver
        known = set(_registered_profiles(resolver))
        known.update(str(row["identifier"]) for row in resolver.inspect("logical_type"))
        return sorted(dtype for dtype in known if dtype)

    def normalized_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of a raw MetaSchema row with JSON string metadata decoded."""
        out = dict(row)
        meta = out.get("meta")
        if isinstance(meta, str):
            try:
                decoded = loads_json(meta)
            except (TypeError, ValueError):
                decoded = None
            if isinstance(decoded, Mapping):
                out["meta"] = decoded
        return out


class _SchemaPlacementService:
    """Own storage-plan resolution and backend placement compatibility."""

    def __init__(self, workspace: Any):
        self.workspace = workspace
        self.index_resolver = IndexResolver()

    def with_inline_keywords(self, entity: type) -> type:
        """Reconstruct the inline declaration for a stored gram-disabled registration."""
        resolver = self.workspace.module_resolver
        fields = {}
        for name, spec in entity._hb_specs.items():
            gram_rules = [StorageStrategy.identifier_of(rule.strategy, resolver=resolver) == SparseGramIndex.identifier for rule in spec.stores]
            if any(gram_rules):
                inline = spec.clone()
                inline.stores = [
                    StorageRule(rule.backend, InlineColumn, optional=rule.optional, reason=rule.reason) if is_gram else rule
                    for rule, is_gram in zip(inline.stores, gram_rules)
                ]
                fields[name] = inline
        if not fields:
            return entity
        return EntityMeta(
            entity.__name__,
            (Entity,),
            {
                "__module__": entity.__module__,
                "__doc__": entity.__doc__,
                "__tags__": entity.schema().tags,
                "__indexes__": [rule for rule in entity._hb_indexes if rule.source == "entity"],
                "__constraints__": [rule for rule in entity._hb_constraints if rule.source == "entity"],
                "__derive__": tuple(entity._hb_derives.values()),
                "identifier": entity.schema().entity_id,
                **entity._hb_specs,
                **fields,
            },
            _pending=False,
        )

    def resolve(self, entity: type, *, gram: bool = True) -> StoragePlan:
        """Resolve and validate storage placement for one entity class.

        Args:
            entity (type): Entity class whose fields are placed.
            gram (bool): Whether profile-selected sparse-GRAM routes are kept.
                ``False`` stores those fields inline instead. Defaults to ``True``.

        Returns:
            StoragePlan: Resolved routes, indexes, and constraints.
        """
        schema = entity.schema()
        overrides = {name: spec.stores for name, spec in entity._hb_specs.items() if spec.stores}
        return self.resolve_schema(
            schema,
            overrides,
            indexes=getattr(entity, "_hb_indexes", ()),
            constraints=getattr(entity, "_hb_constraints", ()),
            gram=gram,
        )

    def resolve_schema(self, schema: Any, overrides: Mapping[str, Any], *, indexes: Any = (), constraints: Any = (), gram: bool = True) -> StoragePlan:
        """Resolve effective storage bindings and reject unsupported backend placements."""
        ws = self.workspace
        storage_plan = ws.storage_resolver.resolve(
            schema,
            ws.backends,
            dict(overrides),
            storage_profiles=ws._storage_profiles.effective(self.storage_profile_dtypes(schema)),
        )
        if not gram:
            storage_plan = self.without_gram(schema, storage_plan, overrides)
        self.validate_storage_support(schema, storage_plan)
        self.validate_graph_edges(schema, storage_plan)
        return self.index_resolver.resolve(schema, storage_plan, indexes, constraints, ws.backends)

    def without_gram(self, schema: Any, storage_plan: StoragePlan, overrides: Mapping[str, Any]) -> StoragePlan:
        """Replace profile-selected sparse-GRAM routes with inline storage.

        Args:
            schema (Any): Entity schema being placed.
            storage_plan (StoragePlan): Plan resolved from profiles and rules.
            overrides (Mapping[str, Any]): Field-authored storage rules.

        Returns:
            StoragePlan: Plan whose sparse-GRAM routes now store the field inline
            on the same backend.

        Raises:
            ValueError: If a field explicitly asks for ``SparseGramIndex`` storage,
                which ``gram=False`` cannot honor.
        """
        resolver = self.workspace.module_resolver
        gram_id = SparseGramIndex.identifier
        explicit = sorted(
            name for name, rules in overrides.items() if any(StorageStrategy.identifier_of(rule.strategy, resolver=resolver) == gram_id for rule in rules)
        )
        if explicit:
            raise ValueError(f"gram=False conflicts with the explicit SparseGramIndex placement of {schema.entity_id} field(s) {explicit}")
        bindings = [
            (
                StorageBinding(binding.entity_id, binding.field_name, binding.backend, InlineColumn.identifier, binding.source, resolver=resolver)
                if binding.strategy == gram_id
                else binding
            )
            for binding in storage_plan.bindings
        ]
        return StoragePlan(storage_plan.entity_id, bindings, storage_plan.indexes, storage_plan.constraints)

    def validate_storage_support(self, schema: Any, storage_plan: StoragePlan) -> None:
        """Reject field placement on backends that cannot store/query the logical type."""
        for binding in storage_plan.bindings:
            field_schema = schema.get(binding.field_name)
            backend = self.workspace.backends.get(binding.backend)
            unsupported = _backend_tag(backend, "unsupported-storage-types")
            if not isinstance(unsupported, (list, tuple, set, frozenset)):
                unsupported = ()
            if field_schema is None or type(field_schema.dtype).identifier not in unsupported:
                continue
            dtype_identifier = type(field_schema.dtype).identifier
            dtype_name = type(field_schema.dtype).__name__
            raise NotImplementedError(
                f"{backend.type} backend {binding.backend!r} does not support {dtype_name} storage for field "
                f"{schema.entity_id}.{field_schema.name}; choose a backend that supports logical type {dtype_identifier!r}."
            )

    def validate_graph_edges(self, schema: Any, storage_plan: StoragePlan) -> None:
        """Validate logical GraphEdge layouts before provider-specific ensure calls."""
        by_field: dict[str, list[Any]] = {}
        for binding in storage_plan.bindings:
            if binding.strategy == GraphEdge.identifier:
                by_field.setdefault(binding.field_name, []).append(binding)
        for field_name, bindings in by_field.items():
            field_schema = schema.get(field_name)
            resolved = [
                (
                    *resolve_graph_endpoint(field_schema, (binding.params or {}).get("target"), schema.entity_id),
                    (binding.strategy_options or {}).get("on_delete", "restrict"),
                )
                for binding in bindings
            ]
            configs = {(endpoint_col, target_entity, on_delete) for endpoint_col, target_entity, _prop_cols, on_delete in resolved}
            if len(configs) != 1:
                raise ValueError(f"GraphEdge field {schema.entity_id}.{field_name} has conflicting target or on_delete metadata across storage bindings")
            endpoint_col, _target_entity, prop_cols, _on_delete = resolved[0]
            endpoint = next(column for column in field_schema.dtype.fields if column.name == endpoint_col)
            if not endpoint.required:
                raise ValueError(f"GraphEdge field {schema.entity_id}.{field_name}: endpoint column {endpoint_col!r} must be required")
            compound = {"artifact", "array", "hyper-g", "json", "vector"}
            invalid = [column.name for column in field_schema.dtype.fields if column.name in prop_cols and type(column.dtype).identifier in compound]
            if invalid:
                raise ValueError(
                    f"GraphEdge field {schema.entity_id}.{field_name} has non-scalar edge-property column(s) {invalid}; " "edge properties must be scalar"
                )

    def validate_unchanged(self, existing_storage_plan: StoragePlan, incoming_storage_plan: StoragePlan, field_names: Iterable[str]) -> None:
        """Reject storage placement changes for fields that already have data."""
        metadata = self.workspace._schema_metadata
        old_storage = metadata.storage_signature(existing_storage_plan, field_names)
        new_storage = metadata.storage_signature(incoming_storage_plan, field_names)
        if old_storage != new_storage:
            raise ValueError("alter_entity cannot change storage for existing fields; add new fields or update descriptions/tags instead")

    def storage_profile_dtypes(self, schema: Any) -> List[str]:
        """Return logical type identifiers that need default storage profiles."""
        keys = set()
        for field_schema in schema.fields.values():
            if field_schema.name == schema.pk_name:
                continue
            keys.update(StorageProfile.keys_for(field_schema.dtype, resolver=self.workspace.module_resolver))
        return sorted(keys)
