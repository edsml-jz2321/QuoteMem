"""Workspace-level graph traversal over ``GraphEdge`` fields.

``ws.traverse`` / ``ws.shortest_path`` resolve the exact GraphEdge field binding rather than selecting
an arbitrary graph-family backend. A readiness-trusted single physical owner such as Neo4j may run
native traversal; replicated fields use the portable merge so every physical owner participates and
conflicts fail closed. Other backends use the same portable frontier expansion over canonical edge lists.
Both honour ``direction`` (out / in / any) and the same hop bounds and use "reachable by a path of
length in [min_hops, max_hops]" semantics — so a query reads identically on either for acyclic graphs
(see ``_py_traverse`` for the tight-cycle caveat at hop depths > 1).

``ws.check_graph`` / ``ws.scrub_dangling`` are implemented in ``_graph_maintenance.py``.
"""

from __future__ import annotations

from heavenbase.utils import Any, Dict, List, Optional, Tuple

from .edges import NodeRef, resolve_edge


def _cap_max_hops(ws: Any, max_hops: int) -> int:
    """Enforce the hard hop ceiling so a deep/cyclic traversal cannot run away.

    Raises rather than silently clamping, so the caller knows their bound was rejected; the ceiling is
    configurable via ``heavenbase.query.graph.max_hops`` (default 64). Note the native ``RETURN DISTINCT``
    only dedups the *output*; this ``max_hops`` cap — not DISTINCT — is what bounds traversal *compute* on
    a cyclic graph.
    """
    cap = int(ws.context.config.get("heavenbase.query.graph.max_hops", default=64))
    if int(max_hops) > cap:
        raise ValueError(f"GraphEdge traversal max_hops {max_hops} exceeds the cap {cap} (raise heavenbase.query.graph.max_hops to allow deeper traversal)")
    return max_hops


def _check_direction(direction: str) -> str:
    """Return the normalized graph direction or raise for an unsupported value."""
    normalized = str(direction).lower()
    if normalized not in ("out", "in", "any"):
        raise ValueError(f"GraphEdge traversal direction must be 'out', 'in', or 'any', got {direction!r}")
    return normalized


def _native_graph_backend(ws: Any, entity_id: str, field_name: str, binding: Any) -> Any:
    """Return a readiness-trusted native backend only for a single-owner edge field."""
    owner_backends = {item.backend for item in ws.storage_plans[entity_id].field_bindings(field_name)}
    if owner_backends != {binding.backend}:
        return None
    backend = ws.backends.get(binding.backend)
    claim = backend.tag("graph-traversal-native")
    return backend if claim is not None and claim.value is True else None


def _edge_pairs(ws: Any, entity_id: str, field_name: str, endpoint_col: str, target_entity: str) -> List[Tuple[NodeRef, NodeRef]]:
    """Return existing typed directed edge pairs: ``(source_entity, id) -> (target_entity, id)``."""
    owner_backends = {binding.backend for binding in ws.storage_plans[entity_id].field_bindings(field_name)}
    source_rows = ws._row_store.universe(
        entity_id,
        required_backends=owner_backends,
        backend_names=owner_backends,
        validate_schema=False,
    ).rows()
    candidate_target_ids = sorted({str(target) for row in source_rows for edge in row.get(field_name) or [] if (target := edge.get(endpoint_col)) is not None})
    target_cls = ws.entities.get(target_entity)
    flags = ws._row_lookup.id_exists_many(candidate_target_ids, entity=target_cls) if target_cls is not None else [False] * len(candidate_target_ids)
    target_ids = {target for target, exists in zip(candidate_target_ids, flags) if exists}
    pairs: List[Tuple[NodeRef, NodeRef]] = []
    for row in source_rows:
        source_id = row.get("object_id")
        if source_id is None:
            continue
        source = (entity_id, str(source_id))
        for edge in row.get(field_name) or []:
            target = edge.get(endpoint_col)
            if target is None:
                continue
            target_id = str(target)
            if target_id in target_ids:
                pairs.append((source, (target_entity, target_id)))
    return pairs


def _adjacency(ws: Any, entity_id: str, field_name: str, endpoint_col: str, target_entity: str, direction: str) -> Dict[NodeRef, set[NodeRef]]:
    """Build the out / in / any typed adjacency map from existing edge endpoints."""
    direction = _check_direction(direction)
    pairs = _edge_pairs(ws, entity_id, field_name, endpoint_col, target_entity)
    out: Dict[NodeRef, set[NodeRef]] = {}

    def add(src: NodeRef, dst: NodeRef) -> None:
        out.setdefault(src, set()).add(dst)

    for src, dst in pairs:
        if direction == "out":
            add(src, dst)
        elif direction == "in":
            add(dst, src)
        else:
            add(src, dst)
            add(dst, src)
    return out


def _start_refs(entity_id: str, target_entity: str, start_id: Any, direction: str) -> set[NodeRef]:
    """Return possible typed start nodes for one traversal request."""
    direction = _check_direction(direction)
    start = str(start_id)
    if direction == "out":
        return {(entity_id, start)}
    if direction == "in":
        return {(target_entity, start)}
    return {(entity_id, start), (target_entity, start)}


def _end_refs(entity_id: str, target_entity: str, end_id: Any, direction: str) -> set[NodeRef]:
    """Return possible typed end nodes for one shortest-path request."""
    direction = _check_direction(direction)
    end = str(end_id)
    if direction == "out":
        return {(target_entity, end)}
    if direction == "in":
        return {(entity_id, end)}
    return {(entity_id, end), (target_entity, end)}


def _py_traverse_refs(
    ws: Any, entity_id: str, field_name: str, endpoint_col: str, target_entity: str, start_id: Any, min_hops: int, max_hops: int, direction: str
) -> List[NodeRef]:
    """Portable traversal returning typed nodes reachable from start by a walk of length in [min_hops, max_hops].

    Typed refs are needed for cross-entity GraphEdge fields: ``out`` reaches target-entity rows,
    ``in`` reaches source-entity rows, and ``any`` may contain both.
    """
    lo, hi = max(1, int(min_hops)), max(1, int(max_hops))
    if lo > hi:
        raise ValueError(f"GraphEdge traversal min_hops {lo} cannot exceed max_hops {hi}")
    adjacency = _adjacency(ws, entity_id, field_name, endpoint_col, target_entity, direction)
    starts = _start_refs(entity_id, target_entity, start_id, direction)
    reached: set[NodeRef] = set()
    frontier = set(starts)
    for hop in range(1, hi + 1):
        frontier = {neighbour for node in frontier for neighbour in adjacency.get(node, set())}
        if not frontier:
            break
        if hop >= lo:
            reached |= frontier
    reached -= starts
    # Sort by object_id (matching the native Cypher ORDER BY oid), with entity as a deterministic tiebreak,
    # so the delegate traverse returns the same order on the fallback as on a graph backend.
    return sorted(reached, key=lambda ref: (ref[1], ref[0]))


def _py_traverse(
    ws: Any, entity_id: str, field_name: str, endpoint_col: str, target_entity: str, start_id: Any, min_hops: int, max_hops: int, direction: str
) -> List[str]:
    """Portable traversal ID helper used by the routed self-referential query op."""
    return [
        object_id for _entity, object_id in _py_traverse_refs(ws, entity_id, field_name, endpoint_col, target_entity, start_id, min_hops, max_hops, direction)
    ]


def _py_shortest_path_refs(
    ws: Any, entity_id: str, field_name: str, endpoint_col: str, target_entity: str, start_id: Any, end_id: Any, max_hops: int, direction: str
) -> Optional[List[NodeRef]]:
    """Portable BFS shortest path as typed refs, or None if unreachable within max_hops."""
    starts = _start_refs(entity_id, target_entity, start_id, direction)
    ends = _end_refs(entity_id, target_entity, end_id, direction)
    if starts & ends:
        return [sorted(starts & ends)[0]]
    adjacency = _adjacency(ws, entity_id, field_name, endpoint_col, target_entity, direction)
    parent: Dict[NodeRef, Optional[NodeRef]] = {start: None for start in starts}
    frontier = sorted(starts)
    for _ in range(max(1, int(max_hops))):
        nxt: List[NodeRef] = []
        for node in frontier:
            for neighbour in sorted(adjacency.get(node, set())):
                if neighbour in parent:
                    continue
                parent[neighbour] = node
                if neighbour in ends:
                    path, cursor = [neighbour], node
                    while cursor is not None:
                        path.append(cursor)
                        cursor = parent[cursor]
                    return list(reversed(path))
                nxt.append(neighbour)
        frontier = nxt
        if not frontier:
            break
    return None


def reached_ids(
    ws: Any, field_ref: Any, start_id: Any, *, min_hops: int = 1, max_hops: int = 1, direction: str = "out", endpoint_preds: Any = None
) -> Tuple[List[str], str, str]:
    """Return reachable object ids, traversal mode, and target entity.

    ``endpoint_preds`` are pushable scalar predicates the routed op fuses into
    the native traverse's ``WHERE b.*``; the portable fallback ignores them
    since the routed op re-applies the full filter afterwards on both paths.

    Args:
        ws (Any): Workspace executing the traversal.
        field_ref (Any): GraphEdge field reference such as ``Doc.cites``.
        start_id (Any): Source ``object_id`` to traverse from.
        min_hops (int): Smallest hop depth to include. Defaults to ``1``.
        max_hops (int): Largest hop depth to include. Defaults to ``1``.
        direction (str): ``"out"``, ``"in"``, or ``"any"``. Defaults to
            ``"out"``.
        endpoint_preds (Any | None): Optional scalar predicates fused into a
            native traverse endpoint filter. Defaults to ``None``.

    Returns:
        tuple[list[str], str, str]: Reachable ``object_id`` values, mode
        ``native_traverse`` or ``py_traverse``, and ``target_entity``.
    """
    _cap_max_hops(ws, max_hops)
    entity_id, field_name, binding, endpoint_col, target_entity = resolve_edge(ws, field_ref)
    backend = _native_graph_backend(ws, entity_id, field_name, binding)
    if backend is not None:
        ids = backend.graph_traverse(entity_id, start_id, field_name, min_hops=min_hops, max_hops=max_hops, direction=direction, endpoint_preds=endpoint_preds)
        return ids, "native_traverse", target_entity
    ids = _py_traverse(ws, entity_id, field_name, endpoint_col, target_entity, start_id, min_hops, max_hops, direction)
    return ids, "py_traverse", target_entity


def traverse(ws: Any, field_ref: Any, start_id: Any, *, min_hops: int = 1, max_hops: int = 1, direction: str = "out") -> List[Dict[str, Any]]:
    """Return the records reachable from ``start_id`` over a GraphEdge field.

    Args:
        ws (Any): Workspace executing the traversal.
        field_ref (Any): GraphEdge field reference such as ``Doc.cites``.
        start_id (Any): Source ``object_id`` to traverse from.
        min_hops (int): Smallest hop depth to include. Defaults to ``1``.
        max_hops (int): Largest hop depth to include. Defaults to ``1``.
        direction (str): ``"out"``, ``"in"``, or ``"any"``. Defaults to
            ``"out"``.

    Returns:
        List[Dict[str, Any]]: Reached target records, deduplicated and ordered
        by ``object_id``.
    """
    _cap_max_hops(ws, max_hops)
    entity_id, field_name, binding, endpoint_col, target_entity = resolve_edge(ws, field_ref)
    backend = _native_graph_backend(ws, entity_id, field_name, binding)
    if backend is not None:
        refs = backend.graph_traverse_refs(entity_id, start_id, field_name, min_hops=min_hops, max_hops=max_hops, direction=direction)
    else:
        refs = _py_traverse_refs(ws, entity_id, field_name, endpoint_col, target_entity, start_id, min_hops, max_hops, direction)
    # ``refs`` already come ordered by object_id on both paths — native via Cypher ORDER BY oid, the
    # fallback via _py_traverse_refs — so records land in the docstring's stated object_id order with no re-sort.
    records = [ws._crud.get(oid, entity=ws.entities[ref_entity]) for ref_entity, oid in refs]
    return [record for record in records if record is not None]


def shortest_path(ws: Any, field_ref: Any, start_id: Any, end_id: Any, *, max_hops: int = 10, direction: str = "out") -> Optional[List[Tuple[str, str]]]:
    """Return the shortest path over a GraphEdge field, or ``None`` when unreachable.

    Args:
        ws (Any): Workspace executing the path search.
        field_ref (Any): GraphEdge field reference such as ``Doc.cites``.
        start_id (Any): Path start ``object_id``.
        end_id (Any): Path end ``object_id``.
        max_hops (int): Maximum path length to search. Defaults to ``10``.
        direction (str): ``"out"``, ``"in"``, or ``"any"``. Defaults to
            ``"out"``.

    Returns:
        Optional[List[tuple[str, str]]]: ``(entity_id, object_id)`` sequence
        from start to end inclusive, or ``None`` when no path exists within
        ``max_hops``.
    """
    _cap_max_hops(ws, max_hops)
    entity_id, field_name, binding, endpoint_col, target_entity = resolve_edge(ws, field_ref)
    # start == end -> the shortest path is the zero-length path [start]. Resolve it here so every backend agrees:
    # native Neo4j ``shortestPath`` has an implicit min length of 1 and would return None for this degenerate
    # case, diverging from the portable fallback. (Same ``starts & ends`` short-circuit as _py_shortest_path_refs.)
    trivial = _start_refs(entity_id, target_entity, start_id, direction) & _end_refs(entity_id, target_entity, end_id, direction)
    if trivial:
        ref_entity, oid = sorted(trivial)[0]
        return [(ref_entity, str(oid))]
    backend = _native_graph_backend(ws, entity_id, field_name, binding)
    if backend is not None:
        refs = backend.graph_shortest_path_refs(entity_id, start_id, end_id, field_name, max_hops=max_hops, direction=direction)
    else:
        refs = _py_shortest_path_refs(ws, entity_id, field_name, endpoint_col, target_entity, start_id, end_id, max_hops, direction)
    if refs is None:
        return None
    return [(ref_entity, str(oid)) for ref_entity, oid in refs]
