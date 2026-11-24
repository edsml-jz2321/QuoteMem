"""Entity declaration assembly into schema IR and query references."""

from __future__ import annotations

__all__ = ["EntityAssembly", "assemble_entity"]

from heavenbase.utils import Any, Dict, List, Mapping, dataclass

from ..schema.model import EntitySchema
from ..types import Identifier
from ..utils.ids import check_entity_id, check_field_name
from ..utils.values import clean_doc, kebab
from .fields import FieldSpec, field as make_field
from .ref import FieldRef
from ..placement import IndexRule, UniqueRule


@dataclass
class EntityAssembly:
    """Collect compiled metadata for one entity class declaration.

    Args:
        schema (EntitySchema): Passive compiled schema IR.
        specs (Dict[str, FieldSpec]): Isolated authored field declarations.
        refs (Dict[str, FieldRef]): Query field references.
        indexes (List[IndexRule]): Authored performance index declarations.
        constraints (List[UniqueRule]): Authored uniqueness constraints.
    """

    schema: EntitySchema
    specs: Dict[str, FieldSpec]
    refs: Dict[str, FieldRef]
    indexes: List[IndexRule]
    constraints: List[UniqueRule]


def assemble_entity(name: str, ns: Mapping[str, Any]) -> EntityAssembly:
    """Compile one class namespace into schema, declarations, and references.

    Args:
        name (str): Python entity class name.
        ns (Mapping[str, Any]): Raw namespace passed to ``EntityMeta``.

    Returns:
        EntityAssembly: Validated isolated entity metadata.

    Raises:
        TypeError: If ``object_id`` does not use ``Identifier``.
        ValueError: If a field or primary-key declaration violates the entity
            contract.
    """
    entity_id = check_entity_id(ns.get("identifier") or kebab(name))
    schema = EntitySchema(entity_id, name, clean_doc(ns.get("__doc__", "")), list(ns.get("__tags__", [])))
    specs = _collect_field_specs(ns)
    _ensure_object_id(specs)
    refs: Dict[str, FieldRef] = {}
    for attr, spec in specs.items():
        schema.set(spec.to_schema(attr))
        refs[attr] = FieldRef(entity_id, attr)
    indexes, constraints = _collect_scalar_rules(ns, specs)
    return EntityAssembly(schema, specs, refs, indexes, constraints)


def _collect_field_specs(ns: Mapping[str, Any]) -> Dict[str, FieldSpec]:
    specs = {check_field_name(key): value.clone() for key, value in ns.items() if isinstance(value, FieldSpec)}
    for attr, annotation in dict(ns.get("__annotations__") or {}).items():
        if not attr.startswith("_") and attr not in specs:
            specs[check_field_name(attr)] = make_field(annotation)
    return specs


def _ensure_object_id(specs: Dict[str, FieldSpec]) -> None:
    if "object_id" not in specs:
        ordered = {"object_id": make_field(Identifier).pk().describe("Built-in stable object_id"), **specs}
        specs.clear()
        specs.update(ordered)
    object_id = specs["object_id"]
    if type(object_id.dtype).identifier != Identifier.identifier:
        raise TypeError("Entity object_id field must use logical type Identifier")
    if object_id.stores:
        raise ValueError("Entity object_id field storage is fixed and cannot be customized")
    for attr, spec in specs.items():
        if attr != "object_id" and spec.is_pk:
            raise ValueError(f"Only fixed object_id field may be primary key; got {attr!r}")
    object_id.pk()


def _collect_scalar_rules(ns: Mapping[str, Any], specs: Dict[str, FieldSpec]) -> tuple[List[IndexRule], List[UniqueRule]]:
    """Collect field- and entity-level scalar index intent."""
    indexes: List[IndexRule] = []
    constraints: List[UniqueRule] = []
    for field_name, spec in specs.items():
        if field_name == "object_id" and (spec.index_rules or spec.unique_rules):
            raise ValueError("object_id indexing is fixed by the backend and cannot be customized")
        indexes.extend(IndexRule((field_name,), rule.backend, rule.enabled, "field") for rule in spec.index_rules)
        constraints.extend(UniqueRule((field_name,), rule.backend, "field") for rule in spec.unique_rules)
    for rule in list(ns.get("__indexes__", []) or []):
        if not isinstance(rule, IndexRule):
            raise TypeError("__indexes__ entries must be created with hb.index(...)")
        _check_rule_fields(rule.fields, specs)
        indexes.append(rule)
    for rule in list(ns.get("__constraints__", []) or []):
        if not isinstance(rule, UniqueRule):
            raise TypeError("__constraints__ entries must be created with hb.unique(...)")
        _check_rule_fields(rule.fields, specs)
        constraints.append(rule)
    seen_constraints: set[tuple[str, ...]] = set()
    duplicate_constraints: set[tuple[str, ...]] = set()
    for rule in constraints:
        if rule.fields in seen_constraints:
            duplicate_constraints.add(rule.fields)
        seen_constraints.add(rule.fields)
    if duplicate_constraints:
        raise ValueError(f"Each logical unique constraint must have one enforcement owner; duplicate fields: {sorted(duplicate_constraints)}")
    return indexes, constraints


def _check_rule_fields(fields: tuple[str, ...], specs: Dict[str, FieldSpec]) -> None:
    """Reject scalar declarations that reference unknown fields."""
    missing = [field_name for field_name in fields if field_name not in specs]
    if missing:
        raise ValueError(f"Scalar index declaration references unknown fields: {missing}")
