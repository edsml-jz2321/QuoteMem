"""GraphEdge audit and cleanup helpers."""

from __future__ import annotations

from heavenbase.utils import Any, Mapping

from ..rows.inputs import UpsertRow
from ..rows.lookup import RowMatch
from .edges import GraphField, graph_field_specs, resolve_edge


def _selected_specs(ws: Any, field_ref: Any) -> list[GraphField]:
    """Return deterministic GraphEdge specs for one field or the workspace."""
    specs = graph_field_specs(ws)
    if field_ref is None:
        return specs
    entity_id, field_name, _binding, _endpoint_col, _target_entity = resolve_edge(ws, field_ref)
    return [spec for spec in specs if (spec.entity_id, spec.field_name) == (entity_id, field_name)]


def _reference_key(reference: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    """Return deterministic typed identity for one GraphEdge reference."""
    return (
        reference["entity"],
        reference["field"],
        reference["source"],
        reference["target_entity"],
        reference["target"],
    )


def _audit(
    ws: Any,
    specs: list[GraphField],
) -> tuple[list[dict[str, str]], dict[tuple[str, str], GraphField]]:
    """Exhaustively scan every physical source owner and batch endpoint checks."""
    specs_by_source: dict[str, list[GraphField]] = {}
    specs_by_field: dict[tuple[str, str], GraphField] = {}
    for spec in specs:
        specs_by_source.setdefault(spec.entity_id, []).append(spec)
        specs_by_field[(spec.entity_id, spec.field_name)] = spec

    refs_by_target: dict[str, dict[str, list[dict[str, str]]]] = {}
    for source_entity, source_specs in sorted(specs_by_source.items()):
        ordered_specs = sorted(source_specs, key=lambda item: item.field_name)
        required_backends = {backend for spec in ordered_specs for backend in spec.owner_backends}
        rows = sorted(
            ws._row_store.universe(
                source_entity,
                required_backends=required_backends,
                backend_names=required_backends,
                validate_schema=False,
            ).rows(),
            key=lambda item: str(item.get("object_id")),
        )
        for stored in rows:
            source = str(stored.get("object_id"))
            for spec in ordered_specs:
                for edge in stored.get(spec.field_name) or []:
                    if not isinstance(edge, Mapping):
                        continue
                    raw_target = edge.get(spec.endpoint_col)
                    if raw_target is None:
                        continue
                    target = str(raw_target)
                    refs_by_target.setdefault(spec.target_entity, {}).setdefault(target, []).append(
                        {
                            "entity": source_entity,
                            "field": spec.field_name,
                            "source": source,
                            "target_entity": spec.target_entity,
                            "target": target,
                        }
                    )

    dangling: list[dict[str, str]] = []
    for target_entity, refs_by_id in sorted(refs_by_target.items()):
        target_ids = sorted(refs_by_id)
        target_cls = ws.entities.get(target_entity)
        exists = ws._row_lookup.id_exists_many(target_ids, entity=target_cls) if target_cls is not None else [False] * len(target_ids)
        for target, present in zip(target_ids, exists):
            if not present:
                dangling.extend(refs_by_id[target])
    dangling.sort(key=_reference_key)
    return dangling, specs_by_field


def _public_references(references: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the stable public audit record shape."""
    return [
        {
            "entity": reference["entity"],
            "field": reference["field"],
            "source": reference["source"],
            "target": reference["target"],
        }
        for reference in references
    ]


def check_graph(ws: Any, field_ref: Any = None) -> list[dict[str, str]]:
    """Report dangling GraphEdge edges for one field or every GraphEdge field.

    Args:
        ws (Any): Workspace whose stored GraphEdge values should be audited.
        field_ref (Any | None): Optional GraphEdge field reference limiting the
            audit. Defaults to ``None``.

    Returns:
        list[dict[str, str]]: Dangling-edge records with ``entity``,
        ``field``, ``source``, and ``target`` keys.
    """
    dangling, _specs = _audit(ws, _selected_specs(ws, field_ref))
    return _public_references(dangling)


def scrub_dangling(ws: Any, field_ref: Any = None, *, confirm: bool = False) -> dict[str, Any]:
    """Remove dangling GraphEdges row-coherently from selected source rows.

    Args:
        ws (Any): Workspace whose dangling GraphEdge values should be repaired.
        field_ref (Any | None): Optional GraphEdge field reference selecting
            source rows. Defaults to ``None``.
        confirm (bool): Whether to apply the repair. ``False`` returns a dry-run
            report. Defaults to ``False``.

    Returns:
        dict[str, Any]: ``{dangling, removed}`` repair report. ``removed`` is
        zero on a dry run.
    """
    specs = graph_field_specs(ws)
    selected_field: tuple[str, str] | None = None
    if field_ref is not None:
        entity_id, field_name, _binding, _endpoint_col, _target_entity = resolve_edge(ws, field_ref)
        selected_field = (entity_id, field_name)
        specs = [spec for spec in specs if spec.entity_id == entity_id]

    dangling, specs_by_field = _audit(ws, specs)
    if selected_field is not None:
        affected_sources = {(reference["entity"], reference["source"]) for reference in dangling if (reference["entity"], reference["field"]) == selected_field}
        dangling = [reference for reference in dangling if (reference["entity"], reference["source"]) in affected_sources]
    report = _public_references(dangling)
    if not confirm:
        return {"dangling": report, "removed": 0, "note": "dry run; pass confirm=True to remove"}

    missing_targets = {(reference["target_entity"], reference["target"]) for reference in dangling}
    for entity_id, object_id in sorted(missing_targets):
        if entity_id in ws.entities:
            ws._row_lookup.validate_absent_row(entity_id, object_id)

    bad_by_source: dict[tuple[str, str], dict[str, set[str]]] = {}
    for reference in dangling:
        source_key = (reference["entity"], reference["source"])
        bad_by_source.setdefault(source_key, {}).setdefault(reference["field"], set()).add(reference["target"])

    rows_by_source: dict[tuple[str, str], RowMatch] = {}
    source_ids_by_entity: dict[str, list[str]] = {}
    for entity_id, source in bad_by_source:
        source_ids_by_entity.setdefault(entity_id, []).append(source)
    for entity_id, source_ids in sorted(source_ids_by_entity.items()):
        entity_cls = ws.entities[entity_id]
        ordered_ids = sorted(source_ids)
        matches = ws._row_lookup.entity_matches_many(entity_cls, ordered_ids, for_patch=True)
        for source, match in zip(ordered_ids, matches):
            if match is None:
                raise RuntimeError(f"GraphEdge repair could not hydrate discovered source {entity_id}:{source}")
            rows_by_source[(entity_id, source)] = match

    removed = 0
    rewrite_rows: list[UpsertRow] = []
    for source_key, bad_by_field in sorted(bad_by_source.items()):
        entity_id, source = source_key
        entity_cls = ws.entities[entity_id]
        match = rows_by_source[source_key]
        patch: dict[str, Any] = {"object_id": source}
        row_removed = 0
        for field_name, bad_targets in sorted(bad_by_field.items()):
            spec = specs_by_field[(entity_id, field_name)]
            current = list(match.row.get(field_name) or [])
            kept = [edge for edge in current if not isinstance(edge, Mapping) or str(edge.get(spec.endpoint_col)) not in bad_targets]
            row_removed += len(current) - len(kept)
            patch[field_name] = kept
        if not row_removed:
            continue
        removed += row_removed
        rewrite_rows.append(ws._row_inputs.patched_row(entity_cls, match, patch))

    if rewrite_rows:
        writer = ws._writer
        plan = writer.prepare_upsert_rows(rewrite_rows, overwrite=True)
        ws._graph_mutation.validate_upserts(rewrite_rows)
        writer.write_upsert(plan)
        ws._crud._invalidate_mcp_registry_metadata_if_needed(plan.entity_classes)
    return {"dangling": report, "removed": removed}
