"""Neo4j backend: one graph node per object, used as an entity/row store.

Each entity record is stored as a single node, labeled and key-scoped per
``(workspace, instance name, entity)`` with ``object_id`` as the node key (a
uniqueness constraint). Scalar ``eq``/``in`` and ``lt``/``lte``/``gt``/``gte``
filters plus non-empty text substring/affix filters push down to an index-backed
Cypher ``WHERE``, and a pure cosine ``near`` runs native KNN through a Neo4j vector
index (see ``handler/runtime.py`` for the parity-exact subset). Mixed logical trees are composed from child fragments so native
leaves can stay native while unsupported leaves use the inherited Python scan fallback
(``Backend.execute``).

Compound logical types that a Neo4j property cannot hold natively are reduced at
the codec boundary: ``json``/``array``/``hyper-g`` become a JSON-string property
(Neo4j properties cannot store nested maps/heterogeneous lists), and ``vector``
becomes a native ``list[float]`` property indexed by a Neo4j vector index for
native cosine ``near``. The Backend's ``binary-storage: base64`` tag makes Artifact bytes arrive as
base64 text (``Artifact.decode`` reverses on read).

A HyperG field declared ``.store(strategy=GraphEdge)`` is materialised as real Neo4j
relationships instead of a JSON property: the field's lone ``Identifier[Entity]`` column is
the edge endpoint (rel-type = the field name, target label = the endpoint's entity), the
remaining columns are edge properties, and reads rebuild the edge list with a Cypher pattern
comprehension. This unlocks native variable-length traversal and ``shortestPath``
(``graph_traverse`` / ``graph_shortest_path``). Plain ``Identifier``-valued *scalar* properties
are still stored as-is (the ``Identifier`` + ``Catalog.target_entity/target_id`` convention).
"""

from __future__ import annotations

__all__ = [
    "Neo4jBackend",
]

from heavenbase.utils import Any, Dict, Iterable, List, Optional, Sequence, dumps_json, loads_json, safe_error

from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...strategies import GraphEdge
from ...strategies.graph_edge import resolve_graph_endpoint
from ...types.storage import stores_as_integer
from ...types.vector import _scale_cosine_query
from ..base import Backend, backend_locked
from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp

CYPHER_RESERVED_WORDS = frozenset(
    "all and any as asc ascending by call case contains count create csv delete desc descending detach distinct else "
    "end ends exists false fieldterminator foreach for in index is limit load match merge node null on optional or "
    "order remove require return scalar scan set skip starts then true union unique unwind using when where with xor "
    "yield".split()
)


class _GraphEdgeSpec:
    """How one ``GraphEdge`` HyperG field maps to Neo4j relationships.

    ``rel_type`` is the encoded field name (the relationship type); ``endpoint_col`` is the
    HyperG column whose ``Identifier[Entity]`` dtype names the target node, and
    ``target_entity`` is that entity's id (used for the target node label); ``prop_cols`` are
    the remaining columns (edge properties).
    """

    __slots__ = ("field_name", "rel_type", "endpoint_col", "target_entity", "prop_cols")

    def __init__(self, field_name: str, rel_type: str, endpoint_col: str, target_entity: str, prop_cols: List[str]) -> None:
        """Create a Neo4j GraphEdge field mapping.

        Args:
            field_name: Logical field name declared on the entity.
            rel_type: Neo4j relationship type for the field.
            endpoint_col: HyperG column containing target object identifiers.
            target_entity: Entity identifier for target nodes.
            prop_cols: HyperG columns stored as relationship properties.

        Returns:
            None: This function does not return a value.
        """
        self.field_name = field_name
        self.rel_type = rel_type
        self.endpoint_col = endpoint_col
        self.target_entity = target_entity
        self.prop_cols = prop_cols


class Neo4jBackend(Backend):
    """Neo4j provider adapter storing one node per object (entity/row store)."""

    identifier = "neo4j"
    provider_reserved_words = CYPHER_RESERVED_WORDS

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "Neo4jBackend":
        """Build the Neo4j entity and row backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``url``, optional ``ws_id``, and
                Neo4j driver options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            Neo4jBackend: Configured Neo4j backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("url", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "neo4j",
        url: str | None = None,
        *,
        ws_id: str | None = None,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Neo4j backend.

        Args:
            name: Workspace backend instance name.
            url: Neo4j Bolt URL.
            ws_id: Optional workspace namespace for labels and indexes.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Extra Neo4j driver options. `user` and `password` are
                consumed by the backend.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        if "graph_on_missing" in kwargs:
            raise TypeError("Neo4j graph_on_missing is not supported; GraphEdge endpoints must be validated workspace rows")
        self.url = url or self._construction_config_get("heavenbase.backend_defaults.neo4j.url", default="bolt://localhost:7687")
        user = kwargs.pop("user", None) or self._construction_config_get("heavenbase.backend_defaults.neo4j.user", default="neo4j")
        password = kwargs.pop("password", None) or self._construction_config_get("heavenbase.backend_defaults.neo4j.password", default="neo4j")
        self._auth = (user, password)
        self._driver_kwargs = dict(kwargs)
        self._schemas: Dict[str, EntitySchema] = {}
        self._graph_specs: Dict[str, Dict[str, "_GraphEdgeSpec"]] = {}  # entity_id -> field -> edge layout
        self._connect()

    def _namespace(self) -> str:
        return self.ws_id or "default"

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase  # top-level neo4j driver (lazy import; optional dependency)

            self.driver = GraphDatabase.driver(self.url, auth=self._auth, **self._driver_kwargs)
            self.driver.verify_connectivity()
            self.client_error = ""
        except Exception as exc:  # provider-unavailable: degrade, raise lazily in _need_client
            self.driver = None
            self.client_error = safe_error(exc)

    def _need_client(self) -> None:
        if self.driver is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"Neo4j backend {self.name!r} is unavailable{detail}")

    def _refresh_workspace(self) -> None:
        self._schemas.clear()
        if self.driver is not None:
            self.driver.close()  # a neo4j driver owns a real connection pool; release it before rebinding to a new namespace
        self._connect()

    # ------------------------------------------------------------------ naming
    def _label(self, entity_id: str) -> str:
        # One node label per (workspace, instance name, entity) so a neo4j + another backend
        # (or two neo4j instances) sharing one database and workspace stay isolated when users choose
        # distinct provider-safe names. Separator-normalized conflicts are intentionally not hidden.
        return f"E_{self._physical_name(self._namespace(), self.name)}_{self._physical_name(entity_id)}"

    def _label_prefix(self) -> str:
        # Trailing "_" keeps this a true segment boundary: one (ws, name) is never a prefix of another's.
        return f"E_{self._physical_name(self._namespace(), self.name)}_"

    @staticmethod
    def _ident(name: Any) -> str:
        """Return a backtick-quoted Cypher identifier for a validated label or property name."""
        text = str(name)
        if "`" in text:
            raise ValueError(f"Unsafe Neo4j identifier: {name!r}")
        return f"`{text}`"

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> List[str]:
        """Return Neo4j label, property, and relationship-type naming warnings.

        Args:
            schema: Entity schema being registered.
            bindings: Storage bindings for the entity.

        Returns:
            List[str]: Provider naming warnings for labels, node properties, and
            relationship types.
        """
        warnings = super().schema_warnings(schema, bindings)
        warnings.extend(
            self._physical_name_warnings(
                [(schema.entity_id, self._label(schema.entity_id))],
                kind="Neo4j label",
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        fields = [(field_name, field_name) for field_name in sorted(schema.fields)]
        warnings.extend(
            self._physical_name_warnings(
                fields,
                kind=f"Neo4j property for {schema.entity_id!r}",
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        rel_fields = [
            (binding.field_name, self._physical_name(binding.field_name))
            for binding in bindings
            if getattr(binding, "backend", None) == self.name and getattr(binding, "strategy", None) == GraphEdge.identifier
        ]
        warnings.extend(
            self._physical_name_warnings(
                rel_fields,
                kind=f"Neo4j relationship type for {schema.entity_id!r}",
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        return warnings

    # ------------------------------------------------------------- row codec
    def _encode_props(self, schema: EntitySchema, object_id: Any, values: Dict[str, Any]) -> Dict[str, Any]:
        """Lower row values to Neo4j node properties, switching on the logical-type identifier."""
        props: Dict[str, Any] = {"object_id": str(object_id)}
        graph_fields = self._graph_specs.get(schema.entity_id, {})
        for name, value in values.items():
            field_schema = schema.fields.get(name)
            if field_schema is None or value is None or name in graph_fields:  # GraphEdge fields become relationships, not node props
                continue
            ident = type(field_schema.dtype).identifier
            if ident == "vector":
                props[name] = [float(v) for v in value]  # native homogeneous float array
            elif ident == "json":
                # The row store already serialized Json via dtype.encode (-> a JSON string); store it
                # as-is. A second dumps_json here would double-encode, and _decode_node's single
                # loads_json would then leave a JSON string that scan-fallback filters compare against
                # the canonical operand -> silent misses. A non-str (direct codec call) is serialized.
                props[name] = value if isinstance(value, str) else dumps_json(value)
            elif ident in ("array", "hyper-g"):
                props[name] = dumps_json(value)  # list / list-of-maps -> JSON string (Neo4j can't hold nested/heterogeneous values)
            elif ident == "boolean":
                props[name] = bool(value)
            elif ident == "float":
                props[name] = float(value)
            elif stores_as_integer(field_schema.dtype):
                props[name] = int(value)
            else:
                props[name] = str(value)  # identifier / text / categorical / date / base64 artifact
        return props

    def _decode_node(self, schema: EntitySchema, node: Dict[str, Any]) -> Dict[str, Any]:
        """Decode a Neo4j node's properties back into a row dict (inverse of ``_encode_props``)."""
        row: Dict[str, Any] = {"object_id": node.get("object_id")}
        for key, raw in node.items():
            if key == "object_id":
                continue
            field_schema = schema.fields.get(key)
            if field_schema is None:
                row[key] = raw
                continue
            ident = type(field_schema.dtype).identifier
            if ident == "vector":
                row[key] = [float(v) for v in raw] if raw is not None else None
            elif ident in ("json", "array", "hyper-g"):
                row[key] = loads_json(raw) if raw else None
            elif ident == "boolean":
                row[key] = bool(raw)
            elif ident == "float":
                row[key] = float(raw) if raw is not None else None
            elif stores_as_integer(field_schema.dtype):
                row[key] = int(raw) if raw is not None else None
            else:
                row[key] = raw
        return row

    # ---------------------------------------------------------------- graph edges
    def _build_graph_specs(self, schema: EntitySchema, bindings: Sequence[Any]) -> Dict[str, "_GraphEdgeSpec"]:
        """Detect this backend's GraphEdge fields from the storage bindings and resolve each edge layout."""
        specs: Dict[str, _GraphEdgeSpec] = {}
        for binding in bindings:
            if getattr(binding, "backend", None) != self.name or getattr(binding, "strategy", None) != GraphEdge.identifier:
                continue
            field_schema = schema.fields.get(binding.field_name)
            if field_schema is not None:
                target = (getattr(binding, "params", None) or {}).get("target")
                specs[binding.field_name] = self._build_graph_spec(field_schema, target, schema.entity_id)
        return specs

    def _build_graph_spec(self, field_schema: Any, target: Optional[str], owning_entity: str) -> "_GraphEdgeSpec":
        """Resolve one GraphEdge HyperG field into its layout: the endpoint column (named or auto-detected) + edge properties."""
        endpoint_col, target_entity, prop_cols = resolve_graph_endpoint(field_schema, target, owning_entity)
        columns = list(getattr(field_schema.dtype, "fields", []))
        # Edge properties must be scalar (json/array/vector/hyper-g and artifact bytes are forbidden):
        # a Neo4j relationship property cannot hold a nested map / heterogeneous list / raw bytes, and even a
        # vector list is disallowed by the spec. Reject at ensure() rather than failing deep in the upsert tx.
        unstorable = [c.name for c in columns if c.name in prop_cols and type(c.dtype).identifier in ("json", "array", "vector", "hyper-g", "artifact")]
        if unstorable:
            raise ValueError(
                f"GraphEdge field {field_schema.name!r} has non-scalar edge-property column(s) {unstorable}; edge properties "
                f"must be scalar (ShortText/Integer/Float/Boolean/timestamp…) — store nested/vector data on the node instead"
            )
        return _GraphEdgeSpec(field_schema.name, self._physical_name(field_schema.name), endpoint_col, target_entity, prop_cols)

    def _graph_read_parts(self, entity_id: str) -> List[tuple]:
        """Cypher pattern-comprehension RETURN parts that rebuild each GraphEdge field's edge list on read."""
        parts: List[tuple] = []
        for field_name, spec in self._graph_specs.get(entity_id, {}).items():
            keys = f"{self._ident(spec.endpoint_col)}: b.object_id" + "".join(f", {self._ident(c)}: r.{self._ident(c)}" for c in spec.prop_cols)
            alias = f"g_{spec.rel_type}"
            parts.append((field_name, alias, f"[(n)-[r:{self._ident(spec.rel_type)}]->(b) | {{{keys}}}] AS {self._ident(alias)}"))
        return parts

    def _sync_edges(self, tx: Any, spec: "_GraphEdgeSpec", src_label: str, src_id: str, edge_rows: Sequence[Any]) -> None:
        """Replace one node's outgoing edges for one GraphEdge field (full-list upsert)."""
        rel = spec.rel_type
        tgt_label = self._label(spec.target_entity)
        src_ident = self._ident(src_label)
        tgt_ident = self._ident(tgt_label)
        rel_ident = self._ident(rel)
        # A relationship is unique per (source, rel-type, target), so two edge rows to the same target would
        # MERGE into one (silent last-wins loss). Reject the parallel edges loudly instead (one edge per target).
        targets = [edge.get(spec.endpoint_col) for edge in edge_rows]
        duplicates = sorted({str(t) for t in targets if t is not None and targets.count(t) > 1})
        if duplicates:
            raise ValueError(
                f"GraphEdge {spec.field_name!r} has parallel edges to the same target {duplicates}; a GraphEdge models "
                f"one edge per (source, target) — parallel edges are not supported"
            )
        tx.run(f"MATCH (a:{src_ident} {{object_id: $src}})-[r:{rel_ident}]->() DELETE r", src=src_id)
        for edge in edge_rows:
            tgt = edge.get(spec.endpoint_col)
            if tgt is None:
                raise ValueError(f"GraphEdge {spec.field_name!r} row is missing its endpoint {spec.endpoint_col!r}")
            props = {c: edge[c] for c in spec.prop_cols if edge.get(c) is not None}
            record = tx.run(
                f"MATCH (a:{src_ident} {{object_id: $src}}), (b:{tgt_ident} {{object_id: $tid}}) "
                f"MERGE (a)-[r:{rel_ident}]->(b) SET r = $props RETURN count(b) AS n",
                src=src_id,
                tid=str(tgt),
                props=props,
            ).single()
            if record is None or record["n"] == 0:
                raise ValueError(
                    f"GraphEdge {spec.field_name!r}: target {tgt!r} (entity {spec.target_entity!r}) does not exist; "
                    "create the endpoint first or include it in the same workspace batch"
                )

    @staticmethod
    def _arrows(direction: str) -> tuple:
        """Cypher relationship arrow pair for a traversal direction (out -> ()-[]->(), in <-[]-, any -[]-)."""
        d = str(direction).lower()
        if d == "out":
            return ("-", "->")
        if d == "in":
            return ("<-", "-")
        if d == "any":
            return ("-", "-")
        raise ValueError(f"GraphEdge traversal direction must be 'out', 'in', or 'any', got {direction!r}")

    def _render_endpoint_preds(self, endpoint_preds: Optional[Sequence[tuple]], params: Dict[str, Any]) -> str:
        """Render pushable endpoint predicates (field, op, value) as a ``WHERE b.*`` Cypher fragment.

        Each binds a fresh ``$w<i>`` parameter; ``in`` renders a list membership, the scalar ops use the
        shared ``_CYPHER_OPS`` map. Returns ``""`` when there is nothing to fuse.
        """
        conditions = []
        for index, (field, op, value) in enumerate(endpoint_preds or []):
            key = f"w{index}"
            params[key] = list(value) if op == "in" else value
            field_ident = self._ident(field)
            conditions.append(f"b.{field_ident} IN ${key}" if op == "in" else f"b.{field_ident} {self._CYPHER_OPS[op]} ${key}")
        return (" AND " + " AND ".join(conditions)) if conditions else ""

    @staticmethod
    def _entity_for_labels(labels: Sequence[str], entity_id: str, target_entity: str, src_label: str, tgt_label: str) -> str:
        """Map one returned node's labels back to the logical entity it belongs to."""
        label_set = set(labels)
        if src_label in label_set:
            return entity_id
        if tgt_label in label_set:
            return target_entity
        raise ValueError(f"Neo4j returned a graph node without either expected label {src_label!r} or {tgt_label!r}: {sorted(label_set)}")

    @backend_locked
    def graph_incoming_sources(
        self,
        entity_id: str,
        field_name: str,
        target_entity: str,
        target_ids: Sequence[Any],
    ) -> List[str]:
        """Return a complete source-id candidate superset for requested typed targets.

        Neo4j relationships are the canonical value for GraphEdge fields placed here,
        so this indexed lookup has no false negatives; the workspace still hydrates and
        rechecks candidates before mutation.

        Args:
            entity_id (str): Logical entity that owns the GraphEdge field.
            field_name (str): GraphEdge field whose incoming relationships are queried.
            target_entity (str): Logical entity type of every requested target.
            target_ids (Sequence[Any]): Target object ids whose sources are needed.

        Returns:
            List[str]: Distinct source object ids in deterministic order.

        Raises:
            RuntimeError: If the Neo4j driver is unavailable.
            ValueError: If the field is not a registered GraphEdge to ``target_entity``.
        """
        self._need_client()
        spec = self._graph_specs.get(entity_id, {}).get(field_name)
        if spec is None or spec.target_entity != target_entity:
            raise ValueError(f"Neo4j backend {self.name!r}: {entity_id}.{field_name} is not a GraphEdge to {target_entity!r}")
        ids = sorted({str(object_id) for object_id in target_ids})
        if not ids:
            return []
        cypher = (
            f"MATCH (a:{self._ident(self._label(entity_id))})-[:{self._ident(spec.rel_type)}]->"
            f"(b:{self._ident(self._label(target_entity))}) WHERE b.object_id IN $ids "
            "RETURN DISTINCT a.object_id AS oid ORDER BY oid"
        )
        with self.driver.session() as session:
            return [str(record["oid"]) for record in session.run(cypher, ids=ids)]

    def graph_traverse_refs(
        self,
        entity_id: str,
        start_id: Any,
        field_name: str,
        *,
        min_hops: int = 1,
        max_hops: int = 1,
        direction: str = "out",
        endpoint_preds: Optional[Sequence[tuple]] = None,
    ) -> List[tuple[str, str]]:
        """Native variable-length traversal as ``(entity_id, object_id)`` refs reachable at hop depth [min_hops, max_hops].

        ``endpoint_preds`` (pushable scalar predicates on the reached node) fuse into the traverse's
        ``WHERE b.*`` so Neo4j prunes during traversal; the routed op still re-applies the
        full filter afterwards, so this is a pruning optimisation, never the sole filter.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object identifier.
            field_name: GraphEdge field name.
            min_hops: Inclusive minimum traversal depth.
            max_hops: Inclusive maximum traversal depth.
            direction: Traversal direction. Supported values are ``out``, ``in``, and ``any``.
            endpoint_preds: Optional pushable predicates over reached endpoint fields.

        Returns:
            List[tuple[str, str]]: Typed object references reached by the traversal.

        Raises:
            RuntimeError: If the Neo4j driver is unavailable.
            ValueError: If the field is not a realized GraphEdge field or the hop/direction
                values are invalid.
        """
        self._need_client()
        spec = self._graph_specs.get(entity_id, {}).get(field_name)
        if spec is None:
            raise ValueError(f"Neo4j backend {self.name!r}: {entity_id}.{field_name} is not a realized GraphEdge field")
        lo, hi = max(1, int(min_hops)), max(1, int(max_hops))
        if lo > hi:  # an inverted range silently matches nothing; reject it as the caller's mistake
            raise ValueError(f"GraphEdge traversal min_hops {lo} cannot exceed max_hops {hi}")
        src_label, tgt_label = self._label(entity_id), self._label(spec.target_entity)
        direction = str(direction).lower()
        params: Dict[str, Any] = {"id": str(start_id)}
        # Exclude the start node only on a SELF-referential edge, where b can cycle back to it. On a cross-entity
        # edge the out/in match binds b to the OPPOSITE label from the start a, so the start can never reappear as
        # b; excluding by object_id alone would then wrongly drop a same-id target (object_ids are per-entity
        # namespaced, so a target can share the start's id). Mirrors the typed Python fallback (_py_traverse_refs
        # removes only the typed start ref, not every same-id node) — see workspace/graph/traversal.py.
        start_guard = "b.object_id <> $id" if entity_id == spec.target_entity else "TRUE"
        if direction == "out":
            left, right = self._arrows(direction)
            result_entity = spec.target_entity
            cypher = (
                f"MATCH (a:{self._ident(src_label)} {{object_id: $id}}){left}[:{self._ident(spec.rel_type)}*{lo}..{hi}]{right}(b:{self._ident(tgt_label)}) "
                f"WHERE {start_guard}{self._render_endpoint_preds(endpoint_preds, params)} RETURN DISTINCT b.object_id AS oid ORDER BY oid"
            )
            with self.driver.session() as session:
                return [(result_entity, str(record["oid"])) for record in session.run(cypher, **params)]
        if direction == "in":
            left, right = self._arrows(direction)
            result_entity = entity_id
            cypher = (
                f"MATCH (a:{self._ident(tgt_label)} {{object_id: $id}}){left}[:{self._ident(spec.rel_type)}*{lo}..{hi}]{right}(b:{self._ident(src_label)}) "
                f"WHERE {start_guard}{self._render_endpoint_preds(endpoint_preds, params)} RETURN DISTINCT b.object_id AS oid ORDER BY oid"
            )
            with self.driver.session() as session:
                return [(result_entity, str(record["oid"])) for record in session.run(cypher, **params)]
        if direction != "any":
            self._arrows(direction)
        params.update({"src_label": src_label, "tgt_label": tgt_label})
        cypher = (
            "MATCH (a) WHERE ($src_label IN labels(a) OR $tgt_label IN labels(a)) AND a.object_id = $id "
            f"MATCH (a)-[:{self._ident(spec.rel_type)}*{lo}..{hi}]-(b) "
            f"WHERE b.object_id <> $id AND ($src_label IN labels(b) OR $tgt_label IN labels(b)){self._render_endpoint_preds(endpoint_preds, params)} "
            "RETURN DISTINCT labels(b) AS labels, b.object_id AS oid ORDER BY oid"
        )
        with self.driver.session() as session:
            return [
                (self._entity_for_labels(record["labels"], entity_id, spec.target_entity, src_label, tgt_label), str(record["oid"]))
                for record in session.run(cypher, **params)
            ]

    def graph_traverse(
        self,
        entity_id: str,
        start_id: Any,
        field_name: str,
        *,
        min_hops: int = 1,
        max_hops: int = 1,
        direction: str = "out",
        endpoint_preds: Optional[Sequence[tuple]] = None,
    ) -> List[str]:
        """Return native variable-length traversal object ids.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object identifier.
            field_name: GraphEdge field name.
            min_hops: Inclusive minimum traversal depth.
            max_hops: Inclusive maximum traversal depth.
            direction: Traversal direction. Supported values are ``out``, ``in``, and ``any``.
            endpoint_preds: Optional pushable predicates over reached endpoint fields.

        Returns:
            List[str]: Object identifiers reached by the traversal.
        """
        return [
            object_id
            for _entity, object_id in self.graph_traverse_refs(
                entity_id, start_id, field_name, min_hops=min_hops, max_hops=max_hops, direction=direction, endpoint_preds=endpoint_preds
            )
        ]

    def graph_shortest_path_refs(
        self, entity_id: str, start_id: Any, end_id: Any, field_name: str, *, max_hops: int = 10, direction: str = "out"
    ) -> Optional[List[tuple[str, str]]]:
        """Return native shortest path typed refs.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object identifier.
            end_id: Target object identifier.
            field_name: GraphEdge field name.
            max_hops: Inclusive maximum path length.
            direction: Traversal direction. Supported values are ``out``, ``in``, and ``any``.

        Returns:
            Optional[List[tuple[str, str]]]: Typed path refs, or ``None`` when unreachable.

        Raises:
            RuntimeError: If the Neo4j driver is unavailable.
            ValueError: If the field is not a realized GraphEdge field or direction is invalid.
        """
        self._need_client()
        spec = self._graph_specs.get(entity_id, {}).get(field_name)
        if spec is None:
            raise ValueError(f"Neo4j backend {self.name!r}: {entity_id}.{field_name} is not a realized GraphEdge field")
        hi = max(1, int(max_hops))
        src_label, tgt_label = self._label(entity_id), self._label(spec.target_entity)
        direction = str(direction).lower()
        if direction == "out":
            left, right = self._arrows(direction)
            cypher = (
                f"MATCH (a:{self._ident(src_label)} {{object_id: $s}}), (b:{self._ident(tgt_label)} {{object_id: $e}}), "
                f"p = shortestPath((a){left}[:{self._ident(spec.rel_type)}*..{hi}]{right}(b)) "
                "RETURN [node IN nodes(p) | {oid: node.object_id, labels: labels(node)}] AS nodes"
            )
        elif direction == "in":
            left, right = self._arrows(direction)
            cypher = (
                f"MATCH (a:{self._ident(tgt_label)} {{object_id: $s}}), (b:{self._ident(src_label)} {{object_id: $e}}), "
                f"p = shortestPath((a){left}[:{self._ident(spec.rel_type)}*..{hi}]{right}(b)) "
                "RETURN [node IN nodes(p) | {oid: node.object_id, labels: labels(node)}] AS nodes"
            )
        elif direction == "any":
            cypher = (
                "MATCH (a), (b) "
                "WHERE ($src_label IN labels(a) OR $tgt_label IN labels(a)) AND a.object_id = $s "
                "AND ($src_label IN labels(b) OR $tgt_label IN labels(b)) AND b.object_id = $e "
                f"MATCH p = shortestPath((a)-[:{self._ident(spec.rel_type)}*..{hi}]-(b)) "
                "RETURN [node IN nodes(p) | {oid: node.object_id, labels: labels(node)}] AS nodes LIMIT 1"
            )
        else:
            self._arrows(direction)
        with self.driver.session() as session:
            record = session.run(cypher, s=str(start_id), e=str(end_id), src_label=src_label, tgt_label=tgt_label).single()
            if record is None:
                return None
            return [
                (self._entity_for_labels(item["labels"], entity_id, spec.target_entity, src_label, tgt_label), str(item["oid"])) for item in record["nodes"]
            ]

    def graph_shortest_path(
        self, entity_id: str, start_id: Any, end_id: Any, field_name: str, *, max_hops: int = 10, direction: str = "out"
    ) -> Optional[List[str]]:
        """Return native shortest path object ids.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object identifier.
            end_id: Target object identifier.
            field_name: GraphEdge field name.
            max_hops: Inclusive maximum path length.
            direction: Traversal direction. Supported values are ``out``, ``in``, and ``any``.

        Returns:
            Optional[List[str]]: Path object ids, or ``None`` when unreachable.
        """
        refs = self.graph_shortest_path_refs(entity_id, start_id, end_id, field_name, max_hops=max_hops, direction=direction)
        return [object_id for _entity, object_id in refs] if refs is not None else None

    # --------------------------------------------------------------- lifecycle
    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Record the schema, the ``object_id`` node-key constraint, and per-field indexes.

        The constraint mirrors the SQL primary key (and enables native id lookup via ``MATCH`` by
        object_id). Each indexable scalar field gets a property index so the Cypher ``WHERE``
        pushdown is index-backed; each ``Vector`` field gets a Neo4j vector index (cosine) so ``near``
        runs native KNN. json/array/hyper-g/artifact get none. Idempotent; a no-op when the driver
        is down. Vector indexes are awaited so a following ``near`` sees an online index.

        Args:
            schema: Entity schema to store in Neo4j.
            bindings: Storage bindings supplied by the workspace.

        Returns:
            None: This function does not return a value.
        """
        self._schemas[schema.entity_id] = schema
        self._graph_specs[schema.entity_id] = self._build_graph_specs(schema, bindings)
        if self.driver is None:
            return
        label = self._label(schema.entity_id)
        graph_fields = self._graph_specs[schema.entity_id]
        vector_indexes: list[str] = []
        with self.driver.session() as session:
            label_ident = self._ident(label)
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label_ident}) REQUIRE n.object_id IS UNIQUE")
            for target_entity in sorted({spec.target_entity for spec in graph_fields.values()}):
                target_label = self._ident(self._label(target_entity))
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{target_label}) REQUIRE n.object_id IS UNIQUE")
            for field_schema in schema.fields.values():
                name = field_schema.name
                ident = type(field_schema.dtype).identifier
                if name == "object_id" or name in graph_fields:  # GraphEdge fields are relationships, not indexed node properties
                    continue
                if ident == "vector":
                    dim = getattr(field_schema.dtype, "dim", None)
                    if dim:
                        vidx = f"vec_{label}_{name}"  # label and name come from validated ids
                        session.run(
                            f"CREATE VECTOR INDEX {self._ident(vidx)} IF NOT EXISTS FOR (n:{label_ident}) ON (n.{self._ident(name)}) "
                            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dim)}, `vector.similarity_function`: 'cosine'}}}}"
                        )
                        vector_indexes.append(vidx)
                elif self.indexable(
                    field_schema.dtype,
                    resolver=self.module_resolver,
                ):  # list/JSON-string types have no comparable scalar -> no property index, no WHERE pushdown
                    index = f"ix_{label}_{name}"
                    session.run(f"CREATE INDEX {self._ident(index)} IF NOT EXISTS FOR (n:{label_ident}) ON (n.{self._ident(name)})")
            # Await only this entity's vector indexes. The global awaitIndexes
            # procedure is vulnerable to unrelated concurrent schema creation.
            timeout = int(self.config.get("heavenbase.backend_defaults.neo4j.index_timeout", default=60))
            for index_name in vector_indexes:
                session.run("CALL db.awaitIndex($name, $timeout)", name=index_name, timeout=timeout)

    @backend_locked
    def ensure_indexes(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Reconcile resolved scalar indexes and uniqueness constraints for one label.

        Args:
            schema: Entity schema owning the Neo4j label.
            storage_plan: Resolved storage, index, and constraint plan.

        Returns:
            None: Heaven-owned indexes and constraints match the active plan.
        """
        if self.driver is None:
            return
        label = self._label(schema.entity_id)
        label_ident = self._ident(label)
        graph_fields = set(self._graph_specs.get(schema.entity_id, {}))
        desired_indexes: set[str] = set()
        desired_constraints: set[str] = set()
        with self.driver.session() as session:
            for binding in storage_plan.backend_indexes(self.name):
                if any(field_name in graph_fields for field_name in binding.fields):
                    continue
                name = f"ix_{label}_{'-'.join(binding.fields)}"
                properties = ", ".join(f"n.{self._ident(field_name)}" for field_name in binding.fields)
                session.run(f"CREATE INDEX {self._ident(name)} IF NOT EXISTS FOR (n:{label_ident}) ON ({properties})")
                desired_indexes.add(name)
            for binding in storage_plan.backend_constraints(self.name):
                name = f"uq_{label}_{'-'.join(binding.fields)}"
                properties = ", ".join(f"n.{self._ident(field_name)}" for field_name in binding.fields)
                target = f"({properties})" if len(binding.fields) > 1 else properties
                session.run(f"CREATE CONSTRAINT {self._ident(name)} IF NOT EXISTS FOR (n:{label_ident}) REQUIRE {target} IS UNIQUE")
                desired_constraints.add(name)
            self._reconcile_scalar_indexes(session, label, desired_indexes, desired_constraints)

    def _reconcile_scalar_indexes(self, session: Any, label: str, desired_indexes: set[str], desired_constraints: set[str]) -> None:
        """Drop de-declared Heaven-owned scalar indexes and constraints."""
        index_prefix = f"ix_{label}_"
        for record in session.run("SHOW INDEXES YIELD name, labelsOrTypes RETURN name, labelsOrTypes"):
            name = str(record["name"])
            if record["labelsOrTypes"] == [label] and name.startswith(index_prefix) and name not in desired_indexes:
                session.run(f"DROP INDEX {self._ident(name)} IF EXISTS")
        constraint_prefix = f"uq_{label}_"
        for record in session.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes RETURN name, labelsOrTypes"):
            name = str(record["name"])
            if record["labelsOrTypes"] == [label] and name.startswith(constraint_prefix) and name not in desired_constraints:
                session.run(f"DROP CONSTRAINT {self._ident(name)} IF EXISTS")

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Write one node per object in a single transaction; ``MERGE`` keys it, ``SET n = $props`` replaces it.

        ``SET n = $props`` overwrites all properties (dropping stale fields) for full-row replace
        semantics, matching the redis/redisearch backends. Delete operations remove the node and
        its relationships via ``DETACH DELETE``. The transaction rolls back on any error.

        GraphEdge fields are realized in ordered batches. Within each batch, nodes are written first,
        validated target ids are merge-only projected as non-authoritative anchors, and each field's
        edges are then synced by full replacement. Completing all three steps before the next batch
        preserves caller order, so a later endpoint delete cannot be undone by an earlier row's anchor.
        Under the coherent StoragePlan that created it, an anchor never substitutes for a logical row;
        route migrations must remove or rewrite obsolete anchors before the new plan is coherent.

        Args:
            rows: Row mutations to apply.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the driver is unavailable or the entity was not ensured.
            ValueError: If a GraphEdge value is malformed or has parallel edges.
        """
        self._need_client()
        rows = list(rows)
        if not rows:
            return
        with self.driver.session() as session, session.begin_transaction() as tx:
            for row in rows:
                if row.is_upsert() and row.entity_id not in self._schemas:
                    raise RuntimeError(f"Neo4j backend {self.name!r}: ensure() not called for {row.entity_id!r}")
            for _entity_id, _is_delete, batch in self.ordered_row_op_batches(rows):
                for row in batch:
                    label = self._label(row.entity_id)
                    label_ident = self._ident(label)
                    if row.is_delete():
                        tx.run(f"MATCH (n:{label_ident} {{object_id: $oid}}) DETACH DELETE n", oid=str(row.object_id))
                        continue
                    schema = self._schemas[row.entity_id]
                    props = self._encode_props(schema, row.object_id, row.values)
                    tx.run(f"MERGE (n:{label_ident} {{object_id: $oid}}) SET n = $props", oid=props["object_id"], props=props)
                anchors: Dict[str, set[str]] = {}
                for row in batch:
                    if row.is_delete():
                        continue
                    for field_name, spec in self._graph_specs.get(row.entity_id, {}).items():
                        for edge in row.values.get(field_name) or []:
                            target = edge.get(spec.endpoint_col)
                            if target is not None:
                                anchors.setdefault(spec.target_entity, set()).add(str(target))
                for target_entity, object_ids in sorted(anchors.items()):
                    label_ident = self._ident(self._label(target_entity))
                    tx.run(f"UNWIND $ids AS oid MERGE (n:{label_ident} {{object_id: oid}})", ids=sorted(object_ids))
                for row in batch:
                    if row.is_delete():
                        continue
                    specs = self._graph_specs.get(row.entity_id, {})
                    src_label = self._label(row.entity_id)
                    for field_name, spec in specs.items():
                        self._sync_edges(tx, spec, src_label, str(row.object_id), row.values.get(field_name) or [])
            tx.commit()

    @staticmethod
    def _matches_prefix(labels: Any, prefix: str) -> bool:
        """Whether a schema artifact's label list belongs to this backend namespace."""
        return any(str(label).startswith(prefix) for label in (labels or []))

    @backend_locked
    def drop(self) -> None:
        """Delete every node and the backing schema artifacts under this workspace+instance namespace.

        Labels are discovered from the server (``CALL db.labels()``) by the ``E_{ns}_{name}_`` prefix,
        not just in-memory state, so a label created in an earlier process is not orphaned. The
        ``object_id`` constraints and the property/vector indexes on those labels are dropped too:
        ``ensure`` only ever issues ``CREATE ... IF NOT EXISTS``, so a leftover vector index would
        otherwise pin a stale dimension/config and break a later recreate of the same field.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        if self.driver is None:
            return
        prefix = self._label_prefix()
        with self.driver.session() as session:
            labels = [record["label"] for record in session.run("CALL db.labels() YIELD label RETURN label")]
            for label in labels:
                if label.startswith(prefix):
                    session.run(f"MATCH (n:{self._ident(label)}) DETACH DELETE n")
            # Drop constraints first (this removes their backing index), then any standalone
            # property/vector indexes left on our labels, so the namespace's schema is fully reset.
            for record in session.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes RETURN name, labelsOrTypes"):
                if self._matches_prefix(record["labelsOrTypes"], prefix):
                    session.run(f"DROP CONSTRAINT {self._ident(record['name'])} IF EXISTS")
            for record in session.run("SHOW INDEXES YIELD name, labelsOrTypes RETURN name, labelsOrTypes"):
                if self._matches_prefix(record["labelsOrTypes"], prefix):
                    session.run(f"DROP INDEX {self._ident(record['name'])} IF EXISTS")
        self._schemas.clear()

    # -------------------------------------------------------------------- read
    def _row_with_edges(self, schema: EntitySchema, parts: List[tuple], record: Any) -> Dict[str, Any]:
        """Decode a node row and inject each GraphEdge field's rebuilt edge list (pattern-comprehension result).

        Drop properties that come back null (an optional edge column omitted on write), so an edge reads
        back with the same key set as the JSON-degrade backends (which omit absent optional columns).
        """
        row = self._decode_node(schema, dict(record["n"]))
        for field_name, alias, _ in parts:
            row[field_name] = [{key: value for key, value in dict(edge).items() if value is not None} for edge in record[alias]]
        return row

    @backend_locked
    def scan(self, entity_id: str) -> List[Dict[str, Any]]:
        """Return every row for one entity by label.

        Args:
            entity_id: Entity identifier whose label should be scanned.

        Returns:
            List[Dict[str, Any]]: Decoded rows with GraphEdge fields rebuilt from
            relationships.

        Raises:
            RuntimeError: If the Neo4j driver is unavailable.
        """
        self._need_client()
        schema = self._schemas.get(entity_id)
        if schema is None:
            return []
        label = self._label(entity_id)
        parts = self._graph_read_parts(entity_id)
        ret = "n" + "".join(f", {expr}" for _, _, expr in parts)
        with self.driver.session() as session:
            return [self._row_with_edges(schema, parts, record) for record in session.run(f"MATCH (n:{self._ident(label)}) RETURN {ret}")]

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> List[Optional[Dict[str, Any]]]:
        """Return rows via a native ``object_id IN`` lookup.

        Args:
            entity_id: Entity identifier whose rows should be loaded.
            object_ids: Object identifiers to fetch in caller order.
            columns: Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            List[Optional[Dict[str, Any]]]: Rows aligned with ``object_ids``. Missing rows
            are returned as ``None``.

        Raises:
            RuntimeError: If the Neo4j driver is unavailable.
        """
        self._need_client()
        schema = self._schemas.get(entity_id)
        ordered = [str(object_id) for object_id in object_ids]
        if schema is None or not ordered:
            return [None for _ in ordered]
        label = self._label(entity_id)
        parts = self._graph_read_parts(entity_id)
        ret = "n" + "".join(f", {expr}" for _, _, expr in parts)
        found: Dict[str, Dict[str, Any]] = {}
        with self.driver.session() as session:
            for record in session.run(f"MATCH (n:{self._ident(label)}) WHERE n.object_id IN $ids RETURN {ret}", ids=ordered):
                row = self._row_with_edges(schema, parts, record)
                found[str(row.get("object_id"))] = row
        return self._project_rows([found.get(object_id) for object_id in ordered], columns)

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return row count via a native label count.

        Args:
            entity_id: Entity identifier whose label should be counted.

        Returns:
            int: Number of rows stored for the entity.

        Raises:
            RuntimeError: If the Neo4j driver is unavailable.
        """
        self._need_client()
        if self._schemas.get(entity_id) is None:
            return 0
        label = self._label(entity_id)
        with self.driver.session() as session:
            record = session.run(f"MATCH (n:{self._ident(label)}) RETURN count(n) AS c").single()
            return int(record["c"]) if record is not None else 0

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live tag claims derived from Neo4j availability.

        Args:
            None.

        Returns:
            dict[str, Any]: Live Backend tag claims.
        """
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.driver is not None else "unavailable",
        }

    # ----------------------------------------------------------- native filter
    _CYPHER_OPS = {"eq": "=", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
    _CYPHER_TEXT_OPS = {"contains": "CONTAINS", "starts_with": "STARTS WITH", "ends_with": "ENDS WITH"}

    def _is_cypher_fragment(self, fragment: QueryFragment) -> bool:
        """Whether the whole fragment tree pushes down to Cypher (every leaf native; no ``not``)."""
        if fragment.kind() == "cypher_where":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_cypher_fragment(child) for child in fragment.children())
        return False  # a "not", or any non-native leaf, cannot become one Cypher WHERE

    def _cypher_where(self, fragment: QueryFragment, params: Dict[str, Any], entity_id: Optional[str] = None) -> str:
        """Render a native fragment tree to a parameterized Cypher WHERE expression."""
        if fragment.notes.get("edge_exists"):  # GraphEdge 1-hop exists -> native relationship EXISTS subquery
            return self._cypher_edge_exists(fragment, params, entity_id)
        if fragment.kind() == "cypher_where":
            op, field, value = fragment.payload["op"], fragment.payload["field"], fragment.payload["value"]
            key = f"p{len(params)}"
            params[key] = value
            if op == "in":
                return f"n.{self._ident(field)} IN ${key}"
            if op in self._CYPHER_TEXT_OPS:
                field_ident = self._ident(field)
                return f"(n.{field_ident} IS NOT NULL AND n.{field_ident} {self._CYPHER_TEXT_OPS[op]} ${key})"
            return f"n.{self._ident(field)} {self._CYPHER_OPS[op]} ${key}"
        joiner = " AND " if fragment.op == "and" else " OR "
        return "(" + joiner.join(self._cypher_where(child, params, entity_id) for child in fragment.children()) + ")"

    def _cypher_edge_exists(self, fragment: QueryFragment, params: Dict[str, Any], entity_id: Optional[str]) -> str:
        """Render a GraphEdge 1-hop ``exists`` filter as a native Cypher ``EXISTS`` over the relationship.

        Each predicate column maps to the endpoint (``b.object_id``) or an edge property (``r.<col>``);
        values bind as parameters, label/rel come from the resolved spec.
        """
        spec = self._graph_specs.get(entity_id, {}).get(fragment.payload["field"]) if entity_id else None
        if spec is None:
            raise RuntimeError(f"Neo4j backend {self.name!r}: no GraphEdge spec for {entity_id}.{fragment.payload['field']}")
        conditions = []
        for col, value in fragment.payload["predicates"].items():
            key = f"p{len(params)}"
            params[key] = value
            conditions.append(f"b.object_id = ${key}" if col == spec.endpoint_col else f"r.{self._ident(col)} = ${key}")
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        return f"EXISTS {{ (n)-[r:{self._ident(spec.rel_type)}]->(b:{self._ident(self._label(spec.target_entity))}){where} }}"

    def _has_vector_index(self, session: Any, index: str) -> bool:
        """Whether a vector index by this name exists and is ONLINE (else the caller scans)."""
        for record in session.run("SHOW INDEXES YIELD name, state RETURN name, state"):
            if record["name"] == index:
                return record["state"] == "ONLINE"
        return False

    def _run_near(self, fragment: QueryFragment) -> ResultFrame:
        """Answer a pure cosine ``near`` via the Neo4j vector index; degrade everything else to scan.

        Native KNN needs an ONLINE cosine vector index, which ``ensure`` builds only for a
        fixed-dimension field. A ``near`` co-located with a filter, a non-cosine metric, a
        dimensionless ``Vector`` field, or a missing/not-yet-online index degrades to
        ``super().execute`` (the portable scan), whose scorer matches the requested metric — so the
        native and scan paths return the same top-k. Neo4j's cosine index score is ``(1+cos)/2``;
        ``2*score - 1`` recovers the raw cosine that ``cosine_score`` (the scan scorer) returns.
        """
        params = fragment.payload
        entity_id = fragment.entity_id()
        schema = self._schemas.get(entity_id) if entity_id else None
        field_schema = schema.fields.get(fragment.field) if schema is not None else None
        dim = getattr(field_schema.dtype, "dim", None) if field_schema is not None else None
        metric = str(params.get("metric", "cosine")).lower()
        if schema is None or dim is None or metric != "cosine" or fragment.filter_fragment() is not None:
            return super().execute(fragment)
        # Neo4j vector indexes keep float32 coordinates; cosine ignores the exact power-of-two scale.
        vector = _scale_cosine_query([float(value) for value in params["vector"]])
        top_k = int(params.get("top_k") or 50)
        index = f"vec_{self._label(entity_id)}_{fragment.field}"
        with self.driver.session() as session:
            if not self._has_vector_index(session, index):
                records = None  # index missing / still building -> portable cosine scan below
            else:
                records = list(
                    session.run(
                        "CALL db.index.vector.queryNodes($index, $k, $vector) YIELD node, score RETURN node, score", index=index, k=top_k, vector=vector
                    )
                )
        if records is None:
            return super().execute(fragment)
        rows = []
        for record in records:
            row = self._decode_node(schema, dict(record["node"]))
            row["score"] = 2.0 * float(record["score"]) - 1.0  # Neo4j cosine (1+cos)/2 -> raw cosine == cosine_score
            rows.append(row)
        return ResultFrame.from_rows(rows)

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Push native filters and vector ``near`` down to Cypher; compose mixed logical trees.

        A scalar/logical filter whose every leaf is a ``cypher_where`` compiles to an index-backed
        ``WHERE``; a pure cosine ``near`` runs native KNN via the vector index. Mixed logical filters
        execute their children through this same method and compose frames, so unsupported leaves
        stay on the portable Python scan without forcing native siblings to scan. Values bind only
        as parameters; labels/fields come from the schema.

        Args:
            fragment: Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.

        Raises:
            RuntimeError: If the Neo4j driver is unavailable.
        """
        self._need_client()
        if fragment.kind() == "neo4j_vector_search" and fragment.op == "near":
            return self._run_near(fragment)
        if fragment.kind() == "cypher_where" or fragment.op in LOGICAL_OPS:
            entity_id = fragment.entity_id()
            schema = self._schemas.get(entity_id) if entity_id else None
            if schema is None:
                return ResultFrame.empty()
            if not self._is_cypher_fragment(fragment):
                if fragment.op in LOGICAL_OPS:
                    return fragment.execute(
                        self.execute, lambda item_entity_id: ResultFrame.from_rows(self.scan(item_entity_id)) if item_entity_id else ResultFrame.empty()
                    )
                return super().execute(fragment)
            params: Dict[str, Any] = {}
            where = self._cypher_where(fragment, params, entity_id)
            label = self._label(entity_id)
            parts = self._graph_read_parts(entity_id)
            ret = "n" + "".join(f", {expr}" for _, _, expr in parts)
            with self.driver.session() as session:
                rows = [
                    self._row_with_edges(schema, parts, record)
                    for record in session.run(f"MATCH (n:{self._ident(label)}) WHERE {where} RETURN {ret}", **params)
                ]
            return ResultFrame.from_rows(rows)
        return super().execute(fragment)  # unsupported ops -> scan

    def to_dict(self) -> dict[str, Any]:
        """Return Neo4j backend metadata and connection state.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata, Neo4j URL, namespace, and provider connection
            status.
        """
        return {
            "name": self.name,
            "type": self.type,
            "url": self.url,
            "namespace": self._namespace(),
            "connected": self.driver is not None,
            "error": self.client_error,
        }
