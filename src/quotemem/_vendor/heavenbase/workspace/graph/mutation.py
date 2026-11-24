"""Workspace GraphEdge referential mutation policy."""

from __future__ import annotations

from heavenbase.utils import Any, Iterable, Mapping, Sequence, raise_mismatch

from ..rows.inputs import DeleteTarget, UpsertRow
from ..rows.lookup import RowMatch
from ..rows.writer import DeletePlan, DeleteRoute
from .edges import GraphField, graph_field_specs
from .errors import GraphDeleteRestrictedError


def _tag_value(backend: Any, identifier: str) -> Any:
    """Return one effective Backend tag value or `None`."""
    result = backend.tag(identifier)
    return result.value if result is not None else None


class _GraphMutationService:
    """Own strong GraphEdge writes and target-deletion policy."""

    def __init__(self, workspace: Any) -> None:
        self.workspace = workspace

    def validate_upserts(
        self,
        rows: Iterable[UpsertRow],
        *,
        allowed_entities: frozenset[str] | None = None,
    ) -> None:
        """Reject invalid or out-of-scope GraphEdge endpoints before row mutation."""
        ws = self.workspace
        row_list = list(rows)
        if not row_list:
            return
        if allowed_entities is not None and any(item.entity_cls.schema().entity_id not in allowed_entities for item in row_list):
            raise PermissionError("GraphEdge write exceeds the active entity scope")
        specs_by_entity: dict[str, list[GraphField]] = {}
        for spec in graph_field_specs(ws):
            specs_by_entity.setdefault(spec.entity_id, []).append(spec)
        batch = {(item.entity_cls.schema().entity_id, str(item.data[item.entity_cls.schema().pk_name])) for item in row_list}
        refs_by_target: dict[str, dict[str, list[dict[str, str]]]] = {}
        for item in row_list:
            schema = item.entity_cls.schema()
            source = str(item.data[schema.pk_name])
            for spec in specs_by_entity.get(schema.entity_id, []):
                if spec.field_name not in item.data:
                    continue
                if item.data.get(spec.field_name) and allowed_entities is not None and spec.target_entity not in allowed_entities:
                    raise PermissionError("GraphEdge write exceeds the active entity scope")
                seen: set[str] = set()
                for edge in item.data.get(spec.field_name) or []:
                    target = edge.get(spec.endpoint_col) if isinstance(edge, Mapping) else None
                    if target is None:
                        raise ValueError(f"GraphEdge {spec.entity_id}.{spec.field_name} on source {source!r} " f"is missing endpoint {spec.endpoint_col!r}")
                    target = str(target)
                    if target in seen:
                        raise ValueError(
                            f"GraphEdge {spec.entity_id}.{spec.field_name} on source {source!r} has parallel edges "
                            f"to target {target!r}; one edge per source, field, and target is supported"
                        )
                    seen.add(target)
                    if (spec.target_entity, target) in batch:
                        continue
                    refs_by_target.setdefault(spec.target_entity, {}).setdefault(target, []).append(
                        {
                            "entity": spec.entity_id,
                            "field": spec.field_name,
                            "source": source,
                            "target_entity": spec.target_entity,
                            "target": target,
                        }
                    )
        missing: list[dict[str, str]] = []
        for target_entity, refs_by_id in sorted(refs_by_target.items()):
            target_cls = ws.entities.get(target_entity)
            target_ids = sorted(refs_by_id)
            flags = ws._row_lookup.id_exists_many(target_ids, entity=target_cls) if target_cls is not None else [False] * len(target_ids)
            for target, exists in zip(target_ids, flags):
                if not exists:
                    missing.extend(refs_by_id[target])
        if missing:
            missing.sort(key=self._reference_key)
            preview = ", ".join(f"{item['entity']}.{item['field']}:{item['source']}->{item['target_entity']}:{item['target']}" for item in missing[:8])
            suffix = "" if len(missing) <= 8 else f", +{len(missing) - 8} more"
            raise ValueError(f"GraphEdge endpoint does not exist for {len(missing)} reference(s): {preview}{suffix}")

    @staticmethod
    def normalize_delete_mode(edges: str) -> str:
        """Return one validated GraphEdge delete mode."""
        return raise_mismatch(("schema", "restrict", "detach"), str(edges).lower(), name="GraphEdge delete mode")

    def delete_rows(
        self,
        targets: Iterable[DeleteTarget | None],
        *,
        edges: str = "schema",
        allowed_entities: frozenset[str] | None = None,
    ) -> DeletePlan:
        """Preflight GraphEdge references and commit one logical delete batch.

        A target that no route returned and whose absence no route shows, as
        after a Pinecone fetch miss on an entity stored only on Pinecone, may
        still exist. Its references are checked and rewritten as for a present
        target, so the delete neither bypasses a restriction nor leaves edges
        to a deleted row; it still reports ``False`` for that target.
        """
        ws = self.workspace
        mode = self.normalize_delete_mode(edges)
        target_list = list(targets)
        writer = ws._writer
        delete_plan = writer.prepare_delete_rows(target_list)
        target_keys = {
            (target.entity_cls.schema().entity_id, target.object_id)
            for target, existed in zip(target_list, delete_plan.deleted)
            if target is not None and existed
        }
        missing = [target for target, existed in zip(target_list, delete_plan.deleted) if target is not None and not existed]
        if missing:
            graph_targets = {spec.target_entity for spec in graph_field_specs(ws)}
            for target in missing:
                entity_id = target.entity_cls.schema().entity_id
                if entity_id in graph_targets and ws._row_lookup.absent_row_error(entity_id, target.object_id) is not None:
                    target_keys.add((entity_id, target.object_id))
        if not target_keys:
            writer.write_delete(delete_plan)
            return delete_plan

        if allowed_entities is not None and any(entity_id not in allowed_entities for entity_id, _object_id in target_keys):
            raise PermissionError("GraphEdge delete exceeds the active entity scope")

        blockers, rewrite_rows = self._delete_rewrites(target_keys, mode, allowed_entities=allowed_entities)
        if blockers:
            raise GraphDeleteRestrictedError(blockers)
        self._add_anchor_delete_routes(delete_plan, target_keys)

        rewrite_plan = writer.prepare_upsert_rows(rewrite_rows, overwrite=True) if rewrite_rows else None
        if rewrite_plan is not None:
            try:
                self.validate_upserts(rewrite_rows, allowed_entities=allowed_entities)
            except PermissionError as exc:
                raise PermissionError("GraphEdge delete exceeds the active entity scope") from exc
        ops_by_backend: dict[str, list[Any]] = {}
        if rewrite_plan is not None:
            self._extend_ops(ops_by_backend, rewrite_plan.entity_ops)
            self._extend_ops(ops_by_backend, rewrite_plan.catalog_ops)
        self._extend_ops(ops_by_backend, writer.delete_ops(delete_plan))
        writer.write_dependency_ordered_ops(ops_by_backend)
        return delete_plan

    def _add_anchor_delete_routes(self, plan: DeletePlan, target_keys: set[tuple[str, str]]) -> None:
        """Route logical target deletion to non-authoritative native endpoint anchors."""
        targets_by_entity: dict[str, list[str]] = {}
        for entity_id, object_id in sorted(target_keys):
            targets_by_entity.setdefault(entity_id, []).append(object_id)
        for spec in graph_field_specs(self.workspace):
            object_ids = targets_by_entity.get(spec.target_entity)
            if not object_ids:
                continue
            for backend_name in spec.graph_backends:
                backend = self.workspace.backends.get(backend_name)
                if _tag_value(backend, "graph-endpoint-materialization") == "identity_anchor":
                    route = DeleteRoute(backend_name, spec.target_entity)
                    plan.object_ids_by_route.setdefault(route, []).extend(object_ids)

    def _delete_rewrites(
        self,
        target_keys: set[tuple[str, str]],
        mode: str,
        *,
        allowed_entities: frozenset[str] | None,
    ) -> tuple[list[dict[str, str]], list[UpsertRow]]:
        """Return restriction references and batched surviving-source rewrites."""
        ws = self.workspace
        targets_by_entity: dict[str, set[str]] = {}
        for entity_id, object_id in target_keys:
            targets_by_entity.setdefault(entity_id, set()).add(object_id)
        specs_by_source: dict[str, list[GraphField]] = {}
        for spec in graph_field_specs(ws):
            if spec.target_entity in targets_by_entity:
                specs_by_source.setdefault(spec.entity_id, []).append(spec)

        blockers: list[dict[str, str]] = []
        patches: list[tuple[RowMatch, dict[str, Any]]] = []
        for source_entity, specs in sorted(specs_by_source.items()):
            source_cls = ws.entities[source_entity]
            source_allowed = allowed_entities is None or source_entity in allowed_entities
            try:
                candidate_ids = self._discover_source_ids(source_entity, specs, targets_by_entity)
            except Exception as exc:
                if not source_allowed:
                    raise PermissionError("GraphEdge delete exceeds the active entity scope") from exc
                raise
            candidate_ids = [source for source in candidate_ids if (source_entity, source) not in target_keys]
            if candidate_ids and not source_allowed:
                raise PermissionError("GraphEdge delete exceeds the active entity scope")
            for match in self._hydrate_rows(source_cls, candidate_ids):
                source = match.object_id
                row = match.row
                patch: dict[str, Any] = {"object_id": source}
                for spec in specs:
                    targets = targets_by_entity[spec.target_entity]
                    current = list(row.get(spec.field_name) or [])
                    matched = [edge for edge in current if isinstance(edge, Mapping) and str(edge.get(spec.endpoint_col)) in targets]
                    if not matched:
                        continue
                    policy = spec.on_delete if mode == "schema" else mode
                    if policy == "restrict":
                        blockers.extend(
                            {
                                "entity": source_entity,
                                "field": spec.field_name,
                                "source": source,
                                "target_entity": spec.target_entity,
                                "target": str(edge.get(spec.endpoint_col)),
                            }
                            for edge in matched
                        )
                        continue
                    patch[spec.field_name] = [edge for edge in current if not isinstance(edge, Mapping) or str(edge.get(spec.endpoint_col)) not in targets]
                if len(patch) > 1:
                    patches.append((match, patch))

        if blockers:
            unique_blockers = {self._reference_key(item): item for item in blockers}
            return [unique_blockers[key] for key in sorted(unique_blockers)], []

        rewrites: list[UpsertRow] = []
        for match, patch in patches:
            entity_id = match.entity_cls.schema().entity_id
            ws._row_lookup.validate_missing_fragments(entity_id, match.object_id, match.missing_routes)
            ws._materializer.assert_query_row_schema(match.entity_cls, match.row)
            rewrites.append(ws._row_inputs.patched_row(match.entity_cls, match, patch))
        return [], rewrites

    def _discover_source_ids(
        self,
        source_entity: str,
        specs: Sequence[GraphField],
        targets_by_entity: Mapping[str, set[str]],
    ) -> list[str]:
        """Discover a complete source-id superset through native lookups and focused scans."""
        ws = self.workspace
        candidates: set[str] = set()
        fallback_backends: set[str] = set()
        for spec in specs:
            target_ids = sorted(targets_by_entity[spec.target_entity])
            for backend_name in spec.owner_backends:
                backend = ws.backends.get(backend_name)
                if backend_name in spec.graph_backends and _tag_value(backend, "graph-incoming-lookup-native") is True:
                    candidates.update(
                        str(source)
                        for source in backend.graph_incoming_sources(
                            source_entity,
                            spec.field_name,
                            spec.target_entity,
                            target_ids,
                        )
                    )
                else:
                    fallback_backends.add(backend_name)

        if fallback_backends:
            scan_specs = [spec for spec in specs if any(backend in fallback_backends for backend in spec.owner_backends)]
            discovered = ws._row_store.universe(
                source_entity,
                required_backends=fallback_backends,
                backend_names=fallback_backends,
                validate_schema=False,
            ).rows()
            candidates.update(
                str(stored["object_id"])
                for stored in discovered
                if stored.get("object_id") is not None
                and any(
                    isinstance(edge, Mapping) and str(edge.get(spec.endpoint_col)) in targets_by_entity[spec.target_entity]
                    for spec in scan_specs
                    for edge in stored.get(spec.field_name) or []
                )
            )
        return sorted(candidates)

    def _hydrate_rows(self, entity_cls: type, object_ids: Sequence[str]) -> Iterable[RowMatch]:
        """Read incoming edges and retain missing routes for any required rewrite."""
        matches = self.workspace._row_lookup.entity_matches_many(entity_cls, object_ids, validate_schema=False)
        entity_id = entity_cls.schema().entity_id
        for object_id, match in zip(object_ids, matches):
            if match is None:
                raise RuntimeError(f"GraphEdge integrity could not hydrate discovered source {entity_id}:{object_id}")
            yield match

    @staticmethod
    def _reference_key(reference: Mapping[str, str]) -> tuple[str, str, str, str, str]:
        """Return the deterministic identity of one GraphEdge reference."""
        return (
            reference["entity"],
            reference["field"],
            reference["source"],
            reference["target_entity"],
            reference["target"],
        )

    @staticmethod
    def _extend_ops(target: dict[str, list[Any]], source: Mapping[str, Sequence[Any]]) -> None:
        """Append backend operation groups without changing provider order."""
        for backend_name, ops in source.items():
            target.setdefault(backend_name, []).extend(ops)
