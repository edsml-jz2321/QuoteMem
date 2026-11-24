"""GraphEdge field resolution helpers."""

from __future__ import annotations

from heavenbase.utils import Any, dataclass

from ...strategies import GraphEdge
from ...strategies.graph_edge import resolve_graph_endpoint

NodeRef = tuple[str, str]


@dataclass(frozen=True)
class GraphField:
    """Describe one logical GraphEdge field independently of physical copies."""

    entity_id: str
    field_name: str
    field_ref: Any
    binding: Any
    owner_backends: tuple[str, ...]
    graph_backends: tuple[str, ...]
    endpoint_col: str
    target_entity: str
    on_delete: str


def resolve_edge(ws: Any, field_ref: Any) -> tuple[str, str, Any, str, str]:
    """Resolve a GraphEdge field reference to entity, field, binding, endpoint, and target entity.

    Args:
        ws (Any): Workspace owning the GraphEdge field.
        field_ref (Any): GraphEdge field reference such as ``Doc.cites``.

    Returns:
        tuple[str, str, Any, str, str]: ``(entity_id, field_name, binding,
        endpoint_col, target_entity)``.

    Raises:
        ValueError: If the field is not GraphEdge-backed or bindings disagree
            on target metadata.
    """
    entity_cls = ws._schema.require(field_ref.entity_id)
    entity_id, field_name = entity_cls.schema().entity_id, field_ref.name
    bindings = [binding for binding in ws.storage_plans[entity_id].field_bindings(field_name) if getattr(binding, "strategy", None) == GraphEdge.identifier]
    if not bindings:
        raise ValueError(f"{entity_id}.{field_name} is not a GraphEdge field; declare it with .store(strategy=GraphEdge)")
    field_schema = entity_cls.schema().fields.get(field_name)
    resolved = [
        (
            binding,
            *resolve_graph_endpoint(field_schema, (getattr(binding, "params", None) or {}).get("target"), entity_id),
            (getattr(binding, "strategy_options", None) or {}).get("on_delete", "restrict"),
        )
        for binding in bindings
    ]
    configs = {(endpoint_col, target_entity, on_delete) for _binding, endpoint_col, target_entity, _prop_cols, on_delete in resolved}
    if len(configs) != 1:
        raise ValueError(f"GraphEdge field {entity_id}.{field_name} has conflicting target or on_delete metadata across storage bindings")
    binding, endpoint_col, target_entity, _prop_cols, _on_delete = resolved[0]
    return entity_id, field_name, binding, endpoint_col, target_entity


def graph_fields(ws: Any) -> list[Any]:
    """Return every GraphEdge field reference registered in the workspace.

    Args:
        ws (Any): Workspace whose storage plans should be scanned.

    Returns:
        list[Any]: GraphEdge field references in stable entity/field order.
    """
    refs: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for entity_id, plan in ws.storage_plans.items():
        entity_cls = ws.entities.get(entity_id)
        if entity_cls is None:
            continue
        for binding in plan.bindings:
            key = (entity_id, binding.field_name)
            if getattr(binding, "strategy", None) == GraphEdge.identifier and key not in seen:
                refs.append(getattr(entity_cls, binding.field_name))
                seen.add(key)
    return refs


def graph_field_specs(ws: Any) -> list[GraphField]:
    """Return resolved logical GraphEdge fields in stable entity/field order.

    Args:
        ws (Any): Workspace whose GraphEdge fields should be resolved.

    Returns:
        list[GraphField]: Resolved GraphEdge metadata for every registered
        field.
    """
    specs = []
    for ref in graph_fields(ws):
        entity_id, field_name, binding, endpoint_col, target_entity = resolve_edge(ws, ref)
        field_bindings = ws.storage_plans[entity_id].field_bindings(field_name)
        specs.append(
            GraphField(
                entity_id,
                field_name,
                ref,
                binding,
                tuple(sorted({item.backend for item in field_bindings})),
                tuple(sorted({item.backend for item in field_bindings if getattr(item, "strategy", None) == GraphEdge.identifier})),
                endpoint_col,
                target_entity,
                (getattr(binding, "strategy_options", None) or {}).get("on_delete", "restrict"),
            )
        )
    return sorted(specs, key=lambda item: (item.entity_id, item.field_name))
