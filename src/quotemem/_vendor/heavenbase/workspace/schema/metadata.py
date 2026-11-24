"""Private schema metadata and introspection service for workspaces."""

from __future__ import annotations

import inspect
import re
import textwrap

from heavenbase.utils import Any, Dict, List, Optional, sha256hash

from ...placement import StoragePlan
from ...utils.serialize import dumps_json

_ARTIFACT_NAMESPACE = re.compile(r"_heavenbase_artifact_[0-9a-f]{64}\Z")


class _SchemaMetadataService:
    """Own stable schema metadata shapes used by manifests, Catalog rows, and MCP descriptions."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def storage_signature(self, storage_plan: StoragePlan, fields: set[str]) -> List[tuple[str, str, str, str, str]]:
        """Return comparable storage bindings for a subset of fields."""
        return sorted(
            (
                binding.field_name,
                binding.backend,
                binding.strategy,
                dumps_json(binding.strategy_options, compact=True, sort_keys=True),
                dumps_json(binding.params, compact=True, sort_keys=True),
            )
            for binding in storage_plan.bindings
            if binding.field_name in fields
        )

    def entity_structure(self, entity: type) -> dict:
        """Return schema structure excluding human-facing metadata."""
        schema = entity.schema()
        return {
            "entity_id": schema.entity_id,
            "derives": self.entity_structure_derives(entity),
            "indexes": [rule.to_dict() for rule in getattr(entity, "_hb_indexes", [])],
            "constraints": [rule.to_dict() for rule in getattr(entity, "_hb_constraints", [])],
            "fields": {
                name: {
                    "dtype": field_schema.dtype.to_dict(),
                    "pk": field_schema.pk,
                    "required": field_schema.required,
                    "has_default": field_schema.has_default,
                    "default": field_schema.default if field_schema.has_default else None,
                    "stores": [self.storage_rule_dict(rule) for rule in self.field_spec(entity, name).stores],
                    "computes": [self.compute_structure_dict(hook) for hook in self.field_spec(entity, name).computes],
                    "query_computes": [self.query_compute_structure_dict(hook) for hook in self.field_spec(entity, name).query_computes],
                }
                for name, field_schema in schema.fields.items()
            },
        }

    def storage_rule_dict(self, rule: Any) -> dict:
        """Return the stable structure used when comparing/serializing storage rules.

        ``params`` (e.g. a ``GraphEdge(target=...)`` endpoint) is included only when non-empty, so the
        common shape stays stable while a parametrised strategy round-trips through the manifest and is
        distinguished by structural-equality comparison.
        """
        out = {"backend": rule.backend, "strategy": rule.strategy_spec(), "optional": rule.optional, "reason": rule.reason}
        if getattr(rule, "params", None):
            out["params"] = dict(rule.params)
        return out

    def field_spec(self, entity: type, name: str) -> Any:
        """Return the FieldSpec metadata for a declared entity field."""
        return entity._hb_specs.get(name)

    def entity_derives(self, entity: type) -> List[dict]:
        """Return resolvable metadata for class-level entity and Catalog derives."""
        return [self.derive_dict(target, derive) for target, derive in sorted(getattr(entity, "_hb_derives", {}).items())]

    def entity_structure_derives(self, entity: type) -> List[dict]:
        """Return comparison metadata for class-level entity and Catalog derives."""
        return [self.derive_structure_dict(target, derive) for target, derive in sorted(getattr(entity, "_hb_derives", {}).items())]

    def derive_by_target(self, entity: type, target: str) -> Optional[dict]:
        """Return one derive metadata record for comparison or ``None``."""
        for item in self.entity_structure_derives(entity):
            if item["target"] == target:
                return item
        return None

    def compute_dict(self, hook: Any) -> dict:
        """Return serializable compute-hook metadata."""
        return {"fn": self.callable_ref(hook.fn), "inputs": list(hook.inputs)}

    def query_compute_dict(self, hook: Any) -> dict:
        """Return serializable query-compute-hook metadata."""
        return {"fn": self.callable_ref(hook.fn), "ops": list(hook.ops)}

    def derive_dict(self, target: str, derive: Any) -> dict:
        """Return resolvable derive metadata."""
        return {"target": target, "computes": [self.compute_dict(hook) for hook in derive.computes]}

    def compute_structure_dict(self, hook: Any) -> dict:
        """Return compute-hook metadata for structural comparison."""
        return {"fn": self.callable_structure_ref(hook.fn), "inputs": list(hook.inputs)}

    def query_compute_structure_dict(self, hook: Any) -> dict:
        """Return query-compute metadata for structural comparison."""
        return {"fn": self.callable_structure_ref(hook.fn), "ops": list(hook.ops)}

    def derive_structure_dict(self, target: str, derive: Any) -> dict:
        """Return derive metadata for structural comparison."""
        return {"target": target, "computes": [self.compute_structure_dict(hook) for hook in derive.computes]}

    def callable_ref(self, fn: Any) -> str:
        """Return a resolvable module-qualified callable reference string."""
        return f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', 'callable'))}"

    def callable_structure_ref(self, fn: Any) -> str:
        """Return a stable callable reference for structural comparisons.

        Artifact generations load below private content-addressed package
        namespaces. Those package identities are intentionally Context-local,
        so equivalent source-backed callables compare by their relative module,
        qualified name, and source hash instead. Ambient callables retain the
        established module-qualified reference.
        """
        module = str(getattr(fn, "__module__", ""))
        qualname = str(getattr(fn, "__qualname__", getattr(fn, "__name__", "callable")))
        namespace, separator, relative_module = module.partition(".")
        if _ARTIFACT_NAMESPACE.fullmatch(namespace) is None:
            return f"{module}.{qualname}"
        try:
            source = textwrap.dedent(inspect.getsource(fn)).strip()
        except (OSError, TypeError):
            return f"{module}.{qualname}"
        relative = f"{relative_module}." if separator else ""
        return f"artifact:{relative}{qualname}:{sha256hash(source)}"

    def query_capabilities(self, entity: type) -> Dict[str, Any]:
        """Return query-compute capabilities keyed by field name."""
        return {field_name: capability for field_name in entity.schema().fields if (capability := self.field_query_capability(entity, field_name))}

    def field_query_capability(self, entity: type, field_name: str) -> Dict[str, Any]:
        """Return query-compute capability metadata for one field."""
        spec = self.field_spec(entity, field_name)
        if spec is None or not spec.query_computes:
            return {}
        dtype_id = type(entity.schema().get(field_name).dtype).identifier
        out: Dict[str, Any] = {}
        for hook in spec.query_computes:
            for op in hook.ops:
                out[op] = {
                    "accepts": ["vector", "text"] if op == "near" and dtype_id == "vector" else ["field_value", "query_compute_input"],
                    "compute": self.callable_ref(hook.fn),
                    "status": "supported" if op == "near" else "custom",
                }
        return out

    def refresh_metadata(self, target: type, source: type) -> None:
        """Copy entity and field metadata from source class to target class."""
        from ...interop.base import apply_entity_meta

        schema = source.schema()
        fields = {
            name: {"desc": field_schema.desc, "tags": field_schema.tags, "examples": field_schema.examples} for name, field_schema in schema.fields.items()
        }
        apply_entity_meta(target, desc=schema.desc, tags=schema.tags, fields=fields)
