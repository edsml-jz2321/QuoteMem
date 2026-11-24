"""Private query routing and execution orchestration for workspaces."""

from __future__ import annotations

from heavenbase.utils import Any

from ...execution.contracts import QueryFragment
from ...execution.frame.merge import intersect
from ...execution.frame.result import ResultFrame
from ...execution.frame.json_path import derive_json_path_columns
from ...query.json_path import json_path_columns
from ...query.spec import QuerySpec
from .aggregate import _AggregateRouteService
from .filter import _FilterRouteService
from .near import _NearRouteService, _NearValueService
from .traverse import _TraverseRouteService


class _WorkspaceRouter:
    """Route, compile, and execute workspace queries."""

    def __init__(self, workspace: Any):
        self.workspace = workspace
        self._filters = _FilterRouteService(self)
        self._aggregate = _AggregateRouteService(self)
        self._near = _NearRouteService(self)
        self._near_values = _NearValueService(self)
        self._traverse = _TraverseRouteService(workspace)

    def execute(self, query: QuerySpec) -> ResultFrame:
        ws = self.workspace
        query = self._near_values.normalize_query(query)
        entity = ws.entities[query.entity_id]
        aggregate_route = self._aggregate.plan(query, entity) if query.aggregate is not None else None
        if query.traverse is not None:
            query, _mode = self._traverse.apply(query, entity)
        if aggregate_route is not None and aggregate_route.mode == "native":
            frame = self._aggregate.execute_native(aggregate_route, query.entity_id)
            frame = self._aggregate.execute_having(frame, query, entity)
            for order in reversed(query.order):
                frame = frame.sort(order.field, order.desc)
            return frame.limit(query.limit, query.offset)

        compiled_near = self._near.compile_near(query.near, entity) if query.near is not None else None
        near_fragment = compiled_near[1] if compiled_near is not None else None
        exact = bool(query.near and query.near.exact)
        near_filter_mode = (
            "exact_scan"
            if self._near.uses_exact_scan(query.near, compiled_near)
            else self._near.near_filter_route(query.near, query.filter, entity, unsafe=query.unsafe, compiled=compiled_near)
        )
        complete_near = near_filter_mode == "exact_scan"

        if query.aggregate is not None:
            schema = entity.schema()
            filter_applied = False
            if query.near is None:
                backend_names = self._aggregate.fold_backend_names(query, entity)
                frame = ws._row_store.universe(query.entity_id, backend_names=backend_names, exhaustive=True)
            else:
                frame = (
                    self._near.exact_scan(query.near, query.filter, entity)
                    if complete_near
                    else self._near.eval_near_filter(compiled_near, query.filter, entity, mode=near_filter_mode, exact=exact, unsafe=query.unsafe)
                )
                filter_applied = frame is not None
                if frame is None:
                    frame = self._near.eval_near(compiled_near)
                if not complete_near and ws._row_store.needs_query_hydration(schema, frame):
                    frame = ws._row_store.hydrate_query_frame(schema, frame, near_fragment=near_fragment)
            frame = ws._materializer.complete_query_frame(entity, frame)
            if query.filter is not None and not filter_applied:
                frame = self._filters.eval_frame(query.filter, entity, frame)
            frame = derive_json_path_columns(schema, frame, json_path_columns(schema, query.aggregate.group_by))
            frame = self._aggregate.execute_fold(frame, query.aggregate, entity)
            frame = self._aggregate.execute_having(frame, query, entity)
            for order in reversed(query.order):
                frame = frame.sort(order.field, order.desc)
            return frame.limit(query.limit, query.offset)

        fields = (
            {"object_id", "score", *(name.split(".")[0] for name in query.select), *(order.field.split(".")[0] for order in query.order)}
            if query.select
            else None
        )
        frame = (
            self._near.exact_scan(query.near, query.filter, entity, fields=fields)
            if complete_near
            else self._near.eval_near_filter(compiled_near, query.filter, entity, mode=near_filter_mode, exact=exact, unsafe=query.unsafe)
        )
        if frame is None:
            filt = self._filters.eval_expr(query.filter, entity, unsafe=query.unsafe) if query.filter else None
            near = self._near.eval_near(compiled_near) if compiled_near is not None else None
            frame = intersect([near, filt]) if near is not None and filt is not None else near if near is not None else filt
            if frame is None:
                frame = ws._row_store.universe(query.entity_id)
        schema = entity.schema()
        if not complete_near and (query.filter is not None or query.near is not None) and ws._row_store.needs_query_hydration(schema, frame):
            frame = ws._row_store.hydrate_query_frame(schema, frame, near_fragment=near_fragment)
        if not complete_near:
            frame = ws._materializer.complete_query_frame(entity, frame)
        # every declared column is present, in one order, whichever rows matched: a field that no matched row stored reads as null
        frame = frame._with_leading_columns(_declared_columns(schema, frame, query))
        entity_cols = frame.col_names()
        names = [*(order.field for order in query.order), *query.select]
        frame = derive_json_path_columns(schema, frame, json_path_columns(schema, names))
        for order in reversed(query.order):
            frame = frame.sort(order.field, order.desc)
        frame = frame.limit(query.limit, query.offset)
        if query.select:
            return frame.project(query.select)
        return frame if frame.col_names() == entity_cols else frame.project(entity_cols)  # columns added only to order rows do not widen the row shape

    def execute_fragment(self, fragment: QueryFragment, backend_name: str, entity_id: str) -> ResultFrame:
        """Execute and decode one backend query fragment."""
        ws = self.workspace
        fragment.bind(entity_id, backend=backend_name)
        schema = ws.entities[entity_id].schema()
        frame = ws.backends.get(backend_name).execute(fragment)
        return ws._row_codec.decode_schema_frame(schema, frame)


def _declared_columns(schema: Any, frame: ResultFrame, query: QuerySpec) -> list[str]:
    """Return the result columns a row query declares, in result order, whether or not rows matched.

    They are the identity, the other entity fields sorted by name, the result
    metadata that the executed filters declare on `frame` (such as the `match`
    of `contained_in`) sorted by name, and last the `score` of a near clause.
    A frame built from stored rows holds only the keys those rows carry, so on
    a backend that stores only present keys (in-memory, file, document stores)
    an empty result holds none of the fields and a populated one lacks an
    optional field that no matched row stored; the router adds each such
    column as null and puts every column in this order, so the result's
    columns follow from the query alone, as they do on a SQL backend. `sort`
    and `project` reject any other order or select name, rows or not, unless
    it is a dotted path into a declared Json field.
    """
    identity = [frame.identity_column] if frame.identity_column is not None else []
    fields = sorted(name for name in schema.fields if name not in identity)
    has_score = query.near is not None or "score" in frame.metadata_columns
    metadata = sorted(name for name in frame.metadata_columns if name not in schema.fields and name not in identity and name != "score")
    return [*identity, *fields, *metadata, *(["score"] if has_score else [])]
