"""Private schema-aware row materialization for workspaces."""

from __future__ import annotations

import inspect

from heavenbase.utils import Any, Dict, Iterable, List, Mapping

from ...entity import Entity
from ...execution.frame.result import ResultFrame
from ...utils.ids import hash_id
from .errors import SchemaVersionMismatchError


class _RowMaterializer:
    """Own defaults, derives, object-id fallback, dtype validation, and query row checks."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def fill_row_defaults(self, entity: type, row: Dict[str, Any]) -> None:
        """Fill absent row keys from field defaults declared on the entity schema."""
        for name, spec in entity._hb_specs.items():
            if name not in row and spec.has_default:
                row[name] = spec.default_value

    def native_derives(self, entity: type) -> Dict[str, Any]:
        """Return field and non-Catalog derives applied to entity rows."""
        derives: Dict[str, Any] = {}
        for name, spec in entity._hb_specs.items():
            if spec.computes:
                derive = Entity.derive(name)
                derive.computes = list(spec.computes)
                derives[name] = derive
        for target, derive in getattr(entity, "_hb_derives", {}).items():
            if not target.startswith("catalog."):
                derives[target] = derive
        return derives

    def apply_derive(self, data: Dict[str, Any], target: str, derive: Any) -> None:
        """Apply one derive to a mutable row."""
        if not derive.computes:
            raise ValueError(f"Derive target {target!r} needs compute")
        for hook in derive.computes:
            data[target] = self.compute_hook_value(target, hook, data)

    def derive_value(self, target: str, derive: Any, data: Dict[str, Any]) -> Any:
        """Return the final value from one derive."""
        if not derive.computes:
            raise ValueError(f"Derive target {target!r} needs compute")
        value = None
        for hook in derive.computes:
            value = self.compute_hook_value(target, hook, data)
        return value

    def compute_hook_value(self, target: str, hook: Any, data: Dict[str, Any]) -> Any:
        """Call one compute hook with contextual missing-input errors."""
        try:
            if hook.inputs:
                return hook.fn(*[data[name] for name in hook.inputs])
            return hook.fn() if self.zero_arg_callable(hook.fn) else hook.fn(data)
        except KeyError as exc:
            raise KeyError(f"Derive target {target!r} is missing input field {exc.args[0]!r}") from None

    def zero_arg_callable(self, fn: Any) -> bool:
        """Return whether a callable can be invoked with no positional data row."""
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        required_kwonly = [
            param for param in signature.parameters.values() if param.kind == inspect.Parameter.KEYWORD_ONLY and param.default is inspect.Parameter.empty
        ]
        positional = [
            param
            for param in signature.parameters.values()
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        ]
        return not positional and not required_kwonly

    def missing_schema_fields(self, entity: type, row: Mapping[str, Any]) -> List[str]:
        """Return schema fields still absent from one stored logical row."""
        return [
            field_schema.name
            for field_schema in entity.schema().fields.values()
            if field_schema.name not in row and (field_schema.required or field_schema.has_default)
        ]

    def assert_query_row_schema(self, entity: type, row: Mapping[str, Any], *, fields: Iterable[str] | None = None) -> None:
        """Raise when a logical row is missing required or defaulted schema fields.

        Args:
            entity (type): Entity class whose schema the row must satisfy.
            row (Mapping[str, Any]): Logical row to check.
            fields (Iterable[str] | None): Field names the caller asked for; when given, only
                those fields are checked, so a projected row is judged on what it requested.

        Raises:
            SchemaVersionMismatchError: If a required or defaulted field is absent.
        """
        missing = self.missing_schema_fields(entity, row)
        if fields is not None:
            requested = set(fields)
            missing = [name for name in missing if name in requested]
        if not missing:
            return
        schema = entity.schema()
        object_id = row.get(schema.pk_name) or row.get("object_id") or ""
        raise SchemaVersionMismatchError(schema.entity_id, str(object_id), missing)

    def complete_query_frame(self, entity: type, frame: ResultFrame) -> ResultFrame:
        """Apply schema-version checks to query rows."""
        for row in frame._iter_rows():
            self.assert_query_row_schema(entity, row)
        return frame

    def materialize(self, entity: type, data: Dict[str, Any]) -> None:
        """Apply defaults, derives, object_id fallback, and dtype validation to a row."""
        schema = entity.schema()
        self.normalize_input_id(schema, data)
        self.fill_row_defaults(entity, data)
        for name, derive in self.native_derives(entity).items():
            if name not in data or data.get(name) is None or (name == schema.pk_name and data.get(name) == ""):
                self.apply_derive(data, name, derive)
        if schema.pk_name not in data or data.get(schema.pk_name) in (None, ""):
            hash_length = int(self.workspace.context.config.get("heavenbase.entity.id.hash_length", default=42))
            data[schema.pk_name] = hash_id(schema.entity_id, self.auto_id_value(entity, data), length=hash_length)
        for field_schema in schema.fields.values():
            if field_schema.name == schema.pk_name:
                continue
            if field_schema.name in data:
                if data[field_schema.name] is None and not field_schema.required:
                    continue
                data[field_schema.name] = field_schema.dtype.validate(data[field_schema.name])
            elif field_schema.required:
                raise ValueError(f"Missing required field {schema.entity_id}.{field_schema.name}")
        field_schema = schema.get(schema.pk_name)
        if field_schema.name in data:
            data[field_schema.name] = field_schema.dtype.validate(data[field_schema.name])

    def normalize_input_id(self, schema: Any, data: Dict[str, Any]) -> None:
        """Normalize fixed object_id input for the current schema."""
        if schema.pk_name != "object_id" and "object_id" in data:
            object_id = data.pop("object_id")
            if schema.pk_name not in data or data.get(schema.pk_name) in (None, ""):
                data[schema.pk_name] = object_id
            elif object_id not in (None, "", data[schema.pk_name]):
                raise ValueError("object_id and primary key must match when both are provided")

    def auto_id_value(self, entity: type, data: Dict[str, Any]) -> Any:
        """Return the source value used for deterministic object_id fallback."""
        schema = entity.schema()
        value = self.workspace._catalog.name_value(entity, data)
        if value not in (None, ""):
            return value
        raise ValueError(
            f"Missing object_id for {schema.entity_id}; object_id is required and cannot be null. Provide object_id, "
            "define object_id = hb.field(hb.Identifier).compute(...), provide a non-empty native name field, "
            'or derive catalog.name with Entity.derive("catalog.name").compute(...)'
        )
