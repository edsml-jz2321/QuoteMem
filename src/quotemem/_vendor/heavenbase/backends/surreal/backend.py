"""SurrealDB multi-model backend: one record per object in a per-(instance, workspace) database.

SurrealDB is a multi-model store, so one ``SurrealBackend`` serves row, text, vector, and graph
domains at once. Each object becomes a
``SCHEMALESS`` record addressed by a bound ``RecordID(table, object_id)``, with the logical
``object_id`` also stored as a field so reads never depend on SurrealDB's record-id coercion:

- **row** — ``UPSERT … CONTENT`` full-row replace; native ``WHERE`` pushdown for the parity-safe
  scalar subset (``handler/runtime.py``), index-backed by the per-field ``DEFINE INDEX``.
- **text** — substring/affix ops push to SurrealDB ``string::*`` functions (NONE-guarded).
- **vector** — a fixed-dim ``Vector`` field gets an HNSW cosine index queried natively via the
  ``<|K,EF|>`` operator; a dimensionless one uses the brute-force ``<|K,COSINE|>`` form. A
  non-cosine metric or a near co-located with a filter degrades to the metric-matched scan.
- **graph** — a ``GraphEdge`` field is stored inline as its canonical value **and** materialized as
  ``RELATE`` edge records. The relation records are an untrusted shadow projection until a future
  versioned readiness/rebuild contract exists, so workspace traversal and delete discovery use the
  canonical inline value; provider-level graph methods remain available for projection verification.

Workspace isolation: every ``(backend instance name, workspace)`` pair maps to its own
SurrealDB ``DATABASE`` (under a shared ``NAMESPACE``), so two SurrealDB backends sharing one server
stay isolated when users choose distinct provider-safe names. Physical names (table, database) use
transparent separator normalization; HeavenBase does not hash names or hide provider conflicts.
The driver is imported lazily inside ``_connect``
and an unreachable server degrades to ``client=None`` (``client_error`` captured); operations raise
lazily via ``_need_client``.
"""

from __future__ import annotations

__all__ = [
    "SurrealBackend",
]

from heavenbase.utils import Any, Dict, Iterable, List, Optional, Sequence, loads_json, safe_error

from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...strategies import GraphEdge, VectorIndex
from ...strategies.graph_edge import resolve_graph_endpoint
from ...types.vector import _scale_cosine_query
from ..base import Backend, backend_locked
from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp

SURREAL_RESERVED_WORDS = frozenset(
    "all and any array as asc begin break cancel continue create database define delete desc else end false for from "
    "function if in info insert into is let limit namespace none not null on option or order output relate remove "
    "return select set sleep table then throw true type update use value where with".split()
)

# Embedded URL schemes run the storage engine in-process (no server, no auth); remote schemes
# (ws/wss/http/https) require signin before use().
_EMBEDDED_SCHEMES = ("mem", "memory", "surrealkv", "rocksdb", "file")
# Leaf filter op -> SurrealQL comparison operator. Only the parity-safe subset is pushed down
# (see handler/runtime.py); ne/like/wildcard/regex ops stay on the portable scan.
_WHERE_OPS = {"eq": "=", "in": "IN", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
# Substring/affix text op -> SurrealQL string function. These match HeavenBase's substring predicates
# (NOT BM25 token search), so they stay parity-exact; the leaf guards NONE so a missing field is False.
# `match` is case-insensitive and compiled separately in _compile_where, so it is not in this map.
_TEXT_FUNCS = {"contains": "string::contains", "starts_with": "string::starts_with", "ends_with": "string::ends_with"}


class SurrealBackend(Backend):
    """SurrealDB provider adapter storing one record per object via SurrealQL."""

    identifier = "surreal"
    aliases = ("surrealdb",)
    provider_reserved_words = SURREAL_RESERVED_WORDS

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any], *, resolver: ModuleResolver | None = None, config: Any | None = None) -> "SurrealBackend":
        """Build the SurrealDB backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. Supports ``url``, optional ``ws_id``, namespace,
                database, auth, and driver options.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            SurrealBackend: Configured SurrealDB backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("url", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "surreal",
        url: str | None = None,
        *,
        ws_id: str | None = None,
        namespace: str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a SurrealDB backend.

        Args:
            name: Workspace backend instance name.
            url: SurrealDB connection URL.
            ws_id: Optional workspace namespace for database selection.
            namespace: Optional SurrealDB namespace override.
            database: Optional SurrealDB database override.
            user: Optional remote SurrealDB username.
            password: Optional remote SurrealDB password.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Extra keyword arguments passed to the SurrealDB driver.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        self.url = url or self._construction_config_get("heavenbase.backend_defaults.surreal.url", default="ws://localhost:8000/rpc")
        self._namespace_cfg = namespace or self._construction_config_get("heavenbase.backend_defaults.surreal.namespace", default="heavenbase")
        self._database_cfg = database
        self._user = user or self._construction_config_get("heavenbase.backend_defaults.surreal.user", default="root")
        self._password = password or self._construction_config_get("heavenbase.backend_defaults.surreal.password", default="root")
        self._client_kwargs = dict(kwargs)
        self._schemas: Dict[str, EntitySchema] = {}
        self._graph_specs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._connect()

    # ------------------------------------------------------------- connection
    def _is_embedded(self) -> bool:
        scheme = str(self.url).split("://", 1)[0].split(":", 1)[0].lower()
        return scheme in _EMBEDDED_SCHEMES

    def _namespace(self) -> str:
        return self._namespace_cfg

    def _database(self) -> str:
        # One SurrealDB DATABASE per (instance name, workspace) when the normalized provider name is distinct.
        base = self._database_cfg or self.name
        return self._physical_name(base, self.ws_id) if self.ws_id else base

    def _connect(self) -> None:
        try:
            from surrealdb import Surreal  # top-level driver (lazy import; optional dependency)

            client = Surreal(self.url, **self._client_kwargs)
            connect = getattr(client, "connect", None)
            if connect is not None:
                connect()
            if not self._is_embedded():
                client.signin({"username": self._user, "password": self._password})
            client.use(self._namespace(), self._database())
            self.client = client
            self.client_error = ""
        except Exception as exc:  # provider-unavailable: degrade, raise lazily in _need_client
            self.client = None
            self.client_error = safe_error(exc)

    def _need_client(self) -> None:
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"SurrealDB backend {self.name!r} is unavailable{detail}")

    def _refresh_workspace(self) -> None:
        self._schemas.clear()
        self._graph_specs.clear()
        if self.client is None:
            self._connect()
            return
        try:
            self.client.use(self._namespace(), self._database())
        except Exception as exc:
            self.client = None
            self.client_error = safe_error(exc)

    def _query(self, surql: str, variables: Optional[Dict[str, Any]] = None) -> Any:
        self._need_client()
        return self.client.query(surql, variables or {})

    @staticmethod
    def _ident(name: Any) -> str:
        """Return a backtick-quoted SurrealDB identifier for a validated table/field name."""
        text = str(name)
        if "`" in text:
            raise ValueError(f"Unsafe SurrealDB identifier: {name!r}")
        return f"`{text}`"

    def _table(self, entity_id: str) -> str:
        return self._physical_name(entity_id)

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> List[str]:
        """Return SurrealDB table and field-name warnings for this schema.

        Args:
            schema: Entity schema being registered.
            bindings: Storage bindings for the entity.

        Returns:
            List[str]: Provider naming warnings for tables and fields.
        """
        warnings = super().schema_warnings(schema, bindings)
        warnings.extend(
            self._physical_name_warnings(
                [(schema.entity_id, self._table(schema.entity_id))],
                kind="SurrealDB table",
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        warnings.extend(
            self._physical_name_warnings(
                [(field_name, field_name) for field_name in sorted(schema.fields)],
                kind=f"SurrealDB field for {schema.entity_id!r}",
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        return warnings

    _COMPOUND_TYPES = ("json", "array", "hyper-g")

    def _decode_row(self, entity_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the record id and decode JSON-string compound fields back to native dicts/lists.

        The workspace row store encodes Json/Array/HyperG via ``dtype.encode`` (a JSON string) before the
        backend sees them, so decoding here hands the scan predicates and the GraphEdge portable traversal
        native values, not strings (mirrors the neo4j ``_decode_node`` contract).
        """
        out = {key: value for key, value in row.items() if key != "id"}
        schema = self._schemas.get(entity_id)
        if schema is not None:
            for name, field_schema in schema.fields.items():
                if isinstance(out.get(name), str) and type(field_schema.dtype).identifier in self._COMPOUND_TYPES:
                    out[name] = loads_json(out[name])
        return out

    # ------------------------------------------------------------- lifecycle
    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Record the schema, then (when connected) define the entity table and per-field indexes.

        Recording the schema first means a backend that is down still satisfies the
        ensure-before-upsert ordering and degrades like the other document backends instead of
        raising at ``ensure`` time; the SurrealQL DDL runs only once a client is available.

        Args:
            schema: Entity schema to store in SurrealDB.
            bindings: Storage bindings supplied by the workspace.

        Returns:
            None: This function does not return a value.
        """
        self._schemas[schema.entity_id] = schema
        self._graph_specs.pop(schema.entity_id, None)  # re-ensure rebuilds specs from scratch (no stale fields)
        if self.client is None:
            return
        table = self._table(schema.entity_id)
        self._query(f"DEFINE TABLE IF NOT EXISTS {self._ident(table)} SCHEMALESS;")
        for binding in bindings:
            if binding.backend != self.name or binding.field_name == "object_id":
                continue
            self._define_field_index(schema.entity_id, table, schema.get(binding.field_name), binding)

    def _define_field_index(self, entity_id: str, table: str, field_schema: Any, binding: Any) -> None:
        """Define the per-field index (or record the graph-edge spec) for the binding's storage strategy."""
        name = field_schema.name
        if binding.strategy == GraphEdge.identifier:
            self._record_graph_spec(entity_id, table, field_schema, binding)
            return
        if binding.strategy == VectorIndex.identifier:
            dim = getattr(field_schema.dtype, "dim", None)
            if dim:  # a fixed-dim vector gets an HNSW index; a dimensionless one falls back to brute-force KNN
                index = self._ident(f"ix_{table}_{name}_hnsw")
                self._query(
                    f"DEFINE INDEX IF NOT EXISTS {index} ON {self._ident(table)} FIELDS {self._ident(name)} HNSW DIMENSION {int(dim)} DIST COSINE TYPE F32;"
                )
            return
        if self.indexable(field_schema.dtype, resolver=self.module_resolver):
            index = self._ident(f"ix_{table}_{name}")
            self._query(f"DEFINE INDEX IF NOT EXISTS {index} ON {self._ident(table)} FIELDS {self._ident(name)};")

    @backend_locked
    def ensure_indexes(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Reconcile resolved scalar indexes and uniqueness constraints.

        Args:
            schema: Entity schema owning the SurrealDB table.
            storage_plan: Resolved storage, index, and constraint plan.

        Returns:
            None: Heaven-owned scalar index DDL matches the active plan.
        """
        if self.client is None:
            return
        table = self._table(schema.entity_id)
        desired: set[str] = set()
        for binding in storage_plan.backend_indexes(self.name):
            name = f"ix_{table}_{'-'.join(binding.fields)}"
            fields_sql = ", ".join(self._ident(field_name) for field_name in binding.fields)
            self._query(f"DEFINE INDEX IF NOT EXISTS {self._ident(name)} ON {self._ident(table)} FIELDS {fields_sql};")
            desired.add(name)
        for binding in storage_plan.backend_constraints(self.name):
            name = f"ix_{table}_{'-'.join(binding.fields)}--unique"
            fields_sql = ", ".join(self._ident(field_name) for field_name in binding.fields)
            self._query(f"DEFINE INDEX IF NOT EXISTS {self._ident(name)} ON {self._ident(table)} FIELDS {fields_sql} UNIQUE;")
            desired.add(name)
        self._reconcile_scalar_indexes(table, desired)

    def _reconcile_scalar_indexes(self, table: str, desired: set[str]) -> None:
        """Drop de-declared scalar indexes while preserving vector HNSW indexes."""
        prefix = f"ix_{table}_"
        for name in self._table_index_names(table):
            if name.startswith(prefix) and not name.endswith("_hnsw") and name not in desired:
                self._query(f"REMOVE INDEX IF EXISTS {self._ident(name)} ON TABLE {self._ident(table)};")

    def _table_index_names(self, table: str) -> List[str]:
        """Return live index names from SurrealDB table metadata."""
        info = self._query(f"INFO FOR TABLE {self._ident(table)};")

        def find_indexes(value: Any) -> Dict[str, Any]:
            if isinstance(value, dict):
                indexes = value.get("indexes")
                if isinstance(indexes, dict):
                    return indexes
                for child in value.values():
                    found = find_indexes(child)
                    if found:
                        return found
            if isinstance(value, list):
                for child in value:
                    found = find_indexes(child)
                    if found:
                        return found
            return {}

        return [str(name) for name in find_indexes(info)]

    def _record_graph_spec(self, entity_id: str, table: str, field_schema: Any, binding: Any) -> None:
        """Resolve a GraphEdge field's endpoint/target/props and record its RELATE edge-table spec.

        The field is still stored inline (its HyperG array round-trips through scan/get like any other
        document field); the recorded spec drives the parallel ``RELATE`` shadow projection and its
        provider-level verification methods. Edge properties must be scalar (a relation property cannot
        hold nested/blob data).
        """
        target = (getattr(binding, "params", None) or {}).get("target")
        endpoint_col, target_entity, prop_cols = resolve_graph_endpoint(field_schema, target, entity_id)
        for column in field_schema.dtype.fields:
            if column.name in prop_cols and type(column.dtype).identifier in self._COMPOUND_TYPES + ("vector", "artifact"):
                raise ValueError(f"GraphEdge field {field_schema.name!r}: edge property {column.name!r} must be a scalar, not {type(column.dtype).identifier}")
        spec = {
            "entity_id": entity_id,
            "endpoint_col": endpoint_col,
            "target_entity": target_entity,
            "prop_cols": prop_cols,
            "source_table": table,
            "target_table": self._table(target_entity),
            "edge_table": f"e_{table}_{field_schema.name}",
        }
        self._graph_specs.setdefault(entity_id, {})[field_schema.name] = spec
        self._query(f"DEFINE TABLE IF NOT EXISTS {self._ident(spec['target_table'])} SCHEMALESS;")
        self._query(
            f"DEFINE TABLE IF NOT EXISTS {self._ident(spec['edge_table'])} TYPE RELATION "
            f"IN {self._ident(spec['source_table'])} OUT {self._ident(spec['target_table'])} SCHEMALESS;"
        )

    @backend_locked
    def drop(self) -> None:
        """Remove this backend's workspace database and re-select an empty one.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        if self.client is None:
            return
        database = self._database()
        self.client.query(f"REMOVE DATABASE IF EXISTS {self._ident(database)};")
        self.client.use(self._namespace(), database)
        self._schemas.clear()
        self._graph_specs.clear()

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply row mutations as ``UPSERT … CONTENT`` / ``DELETE`` plus GraphEdge ``RELATE`` edges, atomically.

        Validation runs first across the whole batch (every row's entity must be ``ensure``-d and no
        GraphEdge field may carry parallel edges); a rejected batch raises before any write is issued. The
        accepted writes are emitted as one ``BEGIN TRANSACTION … COMMIT TRANSACTION`` multi-statement
        query. Within each ordered row batch, record ``UPSERT``/``DELETE`` statements run before validated
        id-anchor ``UPSERT ... MERGE`` and per-field edge ``DELETE``/``RELATE`` statements. Completing those
        steps before the next batch preserves caller order, so a later endpoint delete cannot be undone by
        an earlier row's anchor. The apply phase remains atomic: an arbitrary runtime error rolls the whole
        transaction back and leaves no partial record or dangling edge (verified against the embedded
        engine). Records are written before edges so each validated GraphEdge target has a physical record
        before its edge relates to it.
        Under the coherent StoragePlan that created it, an id-only target record is a non-authoritative
        native projection and never participates in logical workspace existence. Route migrations must
        remove or rewrite obsolete anchors before the new plan is coherent.

        A GraphEdge field is also stored inline (so scan/get round-trip it like any document field), and
        the parallel ``RELATE`` edges it maintains here back provider-level projection checks. Because
        ``UPSERT … CONTENT`` is a full-row replace, a graph field *omitted* from a row is cleared inline —
        so its native edges are cleared too (an absent field syncs as an empty list), keeping the inline
        copy and the native edges in agreement (matching the inline backends' full-row-replace semantics).

        Args:
            rows: Row mutations to apply.

        Returns:
            None: This function does not return a value.

        Raises:
            RuntimeError: If the client is unavailable or an entity was not ensured.
            ValueError: If a GraphEdge field has parallel edges.
        """
        self._need_client()
        from surrealdb import RecordID  # lazy import; optional dependency

        plans: List[tuple] = []  # (row, specs, {field_name: [(target_oid, props), ...]}) validated up front
        for row in rows:
            if row.entity_id not in self._schemas and not row.is_delete():
                raise RuntimeError(f"SurrealDB backend {self.name!r}: ensure() not called for {row.entity_id!r}")
            specs = self._graph_specs.get(row.entity_id, {})
            edge_targets: Dict[str, List[tuple]] = {}
            if row.is_upsert():
                # Sync EVERY graph spec, not just fields present in row.values: a full-row replace that
                # omits a graph field clears it inline, so its native edges must be cleared too (missing
                # field -> empty list) or the canonical value and shadow projection would disagree.
                for field_name, spec in specs.items():
                    edge_targets[field_name] = self._edge_targets(spec, row.values.get(field_name))
            plans.append((row, specs, edge_targets))
        statements: List[str] = []
        variables: Dict[str, Any] = {}
        plan_offset = 0
        for _entity_id, _is_delete, batch in self.ordered_row_op_batches(rows):
            batch_plans = plans[plan_offset : plan_offset + len(batch)]
            plan_offset += len(batch)
            for row, specs, _edge_targets in batch_plans:
                object_id = str(row.object_id)
                record = self._bind(variables, RecordID(self._table(row.entity_id), object_id))
                if row.is_delete():
                    statements.append(f"DELETE ${record};")
                    for spec in specs.values():
                        self._emit_delete_out_edges(spec, object_id, statements, variables)
                    continue
                data = row.to_row()
                data["object_id"] = object_id
                statements.append(f"UPSERT ${record} CONTENT ${self._bind(variables, data)};")
            anchors = {
                (specs[field_name]["target_entity"], str(target_oid))
                for _row, specs, edge_targets in batch_plans
                for field_name, targets in edge_targets.items()
                for target_oid, _props in targets
            }
            for target_entity, target_oid in sorted(anchors):
                record = self._bind(variables, RecordID(self._table(target_entity), target_oid))
                values = self._bind(variables, {"object_id": target_oid})
                statements.append(f"UPSERT ${record} MERGE ${values};")
            for row, specs, edge_targets in batch_plans:
                for field_name, targets in edge_targets.items():
                    self._emit_write_edges(specs[field_name], str(row.object_id), targets, statements, variables)
        if not statements:
            return
        self._query("BEGIN TRANSACTION;\n" + "\n".join(statements) + "\nCOMMIT TRANSACTION;", variables)

    @staticmethod
    def _bind(variables: Dict[str, Any], value: Any) -> str:
        """Bind ``value`` to a fresh ``$pN`` parameter and return that parameter name (no value interpolation)."""
        name = f"p{len(variables)}"
        variables[name] = value
        return name

    def _emit_delete_out_edges(self, spec: Dict[str, Any], source_oid: str, statements: List[str], variables: Dict[str, Any]) -> None:
        """Append the ``DELETE … WHERE in = …`` statement clearing one source record's out-edges (full-list replace step)."""
        from surrealdb import RecordID  # lazy import; optional dependency

        source = self._bind(variables, RecordID(spec["source_table"], str(source_oid)))
        statements.append(f"DELETE {self._ident(spec['edge_table'])} WHERE in = ${source};")

    def _edge_targets(self, spec: Dict[str, Any], value: Any) -> List[tuple]:
        """Parse one GraphEdge field value into ordered ``(target_oid, props)``, rejecting parallel edges.

        Pure parse/validation with no writes: a field carrying two edges to the same target raises here,
        before ``upsert`` writes anything, so a rejected batch never leaves a partial record/edge behind.
        """
        edges = loads_json(value) if isinstance(value, str) else (value or [])
        endpoint_col, prop_cols = spec["endpoint_col"], spec["prop_cols"]
        targets: List[tuple] = []
        seen: set = set()
        for edge in edges:
            target = edge.get(endpoint_col)
            if target is None:
                continue
            target = str(target)
            if target in seen:
                raise ValueError(f"GraphEdge field has parallel edges to {target!r}; a GraphEdge models one edge per (source, target)")
            seen.add(target)
            props = {column: edge.get(column) for column in prop_cols if edge.get(column) is not None}
            targets.append((target, props))
        return targets

    def _emit_write_edges(self, spec: Dict[str, Any], source_oid: str, targets: Sequence[tuple], statements: List[str], variables: Dict[str, Any]) -> None:
        """Append the full-list-replace edge statements (clear then ``RELATE`` each target) for one GraphEdge field."""
        self._emit_delete_out_edges(spec, source_oid, statements, variables)
        for target_oid, props in targets:
            self._emit_relation(spec, source_oid, target_oid, props, statements, variables)

    def _emit_relation(
        self, spec: Dict[str, Any], source_oid: str, target_oid: str, props: Dict[str, Any], statements: List[str], variables: Dict[str, Any]
    ) -> None:
        """Append one ``RELATE`` statement (``in``/``out`` record links + scalar properties) as a parameterized query.

        ``RELATE`` requires record values in the arrow endpoints, so the source/target are bound
        as ``RecordID`` parameters and the edge table is the only interpolated token (backtick-quoted via
        ``_ident``); every value still binds as a parameter.
        """
        from surrealdb import RecordID  # lazy import; optional dependency

        src = self._bind(variables, RecordID(spec["source_table"], str(source_oid)))
        dst = self._bind(variables, RecordID(spec["target_table"], str(target_oid)))
        content = self._bind(variables, dict(props))
        statements.append(f"RELATE ${src}->{self._ident(spec['edge_table'])}->${dst} CONTENT ${content};")

    @backend_locked
    def scan(self, entity_id: str) -> List[Dict[str, Any]]:
        """Return every row for one entity.

        Args:
            entity_id: Entity identifier whose table should be scanned.

        Returns:
            List[Dict[str, Any]]: Decoded rows.

        Raises:
            RuntimeError: If the SurrealDB client is unavailable.
        """
        self._need_client()
        return [self._decode_row(entity_id, row) for row in self._query("SELECT * FROM type::table($tb);", {"tb": self._table(entity_id)})]

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> List[Optional[Dict[str, Any]]]:
        """Return rows with a native ``WHERE object_id IN`` lookup.

        Args:
            entity_id: Entity identifier whose rows should be loaded.
            object_ids: Object identifiers to fetch in caller order.
            columns: Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            List[Optional[Dict[str, Any]]]: Rows aligned with ``object_ids``. Missing rows
            are returned as ``None``.

        Raises:
            RuntimeError: If the SurrealDB client is unavailable.
        """
        self._need_client()
        ordered = [str(object_id) for object_id in object_ids]
        if not ordered:
            return []
        found: Dict[Any, Dict[str, Any]] = {}
        for row in self._query("SELECT * FROM type::table($tb) WHERE object_id IN $ids;", {"tb": self._table(entity_id), "ids": ordered}):
            decoded = self._decode_row(entity_id, row)
            found[decoded.get("object_id")] = decoded
        return self._project_rows([found.get(object_id) for object_id in ordered], columns)

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return row count without scanning rows into Python.

        Args:
            entity_id: Entity identifier whose table should be counted.

        Returns:
            int: Number of rows stored for the entity.

        Raises:
            RuntimeError: If the SurrealDB client is unavailable.
        """
        self._need_client()
        result = self._query("SELECT count() FROM type::table($tb) GROUP ALL;", {"tb": self._table(entity_id)})
        return int(result[0]["count"]) if result else 0

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live tag claims derived from SurrealDB availability.

        Args:
            None.

        Returns:
            dict[str, Any]: Live Backend tag claims.
        """
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.client is not None else "unavailable",
        }

    # --------------------------------------------------- native WHERE execute
    def _is_surreal_fragment(self, fragment: QueryFragment) -> bool:
        # `not` is kept on the portable scan (its NULL/absent-field semantics can diverge from the
        # Python predicate), so a tree containing a `not` composes through QueryFragment.execute.
        if fragment.kind() == "surreal_where":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_surreal_fragment(child) for child in fragment.children())
        return False

    def _compile_where(self, fragment: QueryFragment, variables: Dict[str, Any]) -> str:
        if fragment.op == "and":
            return "(" + " AND ".join(self._compile_where(child, variables) for child in fragment.children()) + ")"
        if fragment.op == "or":
            return "(" + " OR ".join(self._compile_where(child, variables) for child in fragment.children()) + ")"
        payload = fragment.payload
        op = payload["op"]
        field = self._ident(payload["field"])
        name = f"p{len(variables)}"
        variables[name] = payload["value"]
        if op in _WHERE_OPS:
            return f"{field} {_WHERE_OPS[op]} ${name}"
        if op == "match":  # case-insensitive substring; the operand is already lowercased by the handler
            return f"({field} IS NOT NONE AND string::contains(string::lowercase({field}), ${name}))"
        return f"({field} IS NOT NONE AND {_TEXT_FUNCS[op]}({field}, ${name}))"

    def _run_near(self, fragment: QueryFragment) -> ResultFrame:
        """Run native cosine KNN (HNSW or brute-force); degrade to the carried scorer for filters/other metrics."""
        entity_id = fragment.entity_id()
        if not entity_id:
            return ResultFrame.empty()
        params = fragment.payload
        metric = str(params.get("metric") or "cosine").lower()
        if metric != "cosine" or fragment.filter_fragment() is not None:
            return super().execute(fragment)  # non-cosine or near+filter -> portable scan with the carried scorer
        top_k = int(params["top_k"])
        # A fixed-dim Vector has an HNSW cosine index (see _define_field_index): query it with the
        # numeric-EF operator `<|K,EF|>` (EF >= K) so SurrealDB iterates the index. The metric-keyword
        # form `<|K,COSINE|>` is a brute-force table scan even when the index exists (verified via
        # EXPLAIN), so it is used only for a dimensionless Vector, which has no index to back it.
        schema = self._schemas.get(entity_id)
        field_schema = schema.fields.get(fragment.field) if schema is not None else None
        dim = getattr(field_schema.dtype, "dim", None) if field_schema is not None else None
        if dim:
            ef = max(top_k, int(self.config.get("heavenbase.backend_defaults.surreal.ef_search", default=64)))
            knn = f"<|{top_k},{ef}|>"
        else:
            knn = f"<|{top_k},COSINE|>"
        surql = f"SELECT *, vector::distance::knn() AS __dist FROM type::table($tb) WHERE {self._ident(fragment.field)} {knn} $vec ORDER BY __dist;"
        scored = []
        # The HNSW index is F32; cosine ignores the exact power-of-two scale, so any finite query fits it.
        vector = _scale_cosine_query([float(value) for value in params["vector"]])
        for row in self._query(surql, {"tb": self._table(entity_id), "vec": vector}):
            decoded = self._decode_row(entity_id, row)
            distance = decoded.pop("__dist", None)
            if distance is None:
                continue
            decoded["score"] = 1.0 - float(distance)  # cosine distance -> similarity (matches cosine_score)
            scored.append(decoded)
        return ResultFrame.from_rows(scored[:top_k])

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Run native SurrealQL for filters and vector ``near`` fragments.

        Args:
            fragment: Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.

        Raises:
            RuntimeError: If the SurrealDB client is unavailable.
        """
        self._need_client()
        if fragment.op == "near" and fragment.kind() == "surreal_vector":
            return self._run_near(fragment)
        if fragment.kind() == "surreal_where" or fragment.op in LOGICAL_OPS:
            entity_id = fragment.entity_id()
            if not entity_id:
                return ResultFrame.empty()
            if not self._is_surreal_fragment(fragment):
                if fragment.op in LOGICAL_OPS:
                    return fragment.execute(
                        self.execute, lambda item_entity_id: ResultFrame.from_rows(self.scan(item_entity_id)) if item_entity_id else ResultFrame.empty()
                    )
                return super().execute(fragment)
            variables: Dict[str, Any] = {"tb": self._table(entity_id)}
            where = self._compile_where(fragment, variables)
            rows = [self._decode_row(entity_id, row) for row in self._query(f"SELECT * FROM type::table($tb) WHERE {where};", variables)]
            return ResultFrame.from_rows(rows)
        return super().execute(fragment)

    # --------------------------------------------------- native graph traversal
    @staticmethod
    def _check_direction(direction: str) -> str:
        normalized = str(direction).lower()
        if normalized not in ("out", "in", "any"):
            raise ValueError(f"GraphEdge traversal direction must be 'out', 'in', or 'any', got {direction!r}")
        return normalized

    @staticmethod
    def _start_refs(entity_id: str, target_entity: str, start_id: Any, direction: str) -> set:
        start = str(start_id)
        if direction == "out":
            return {(entity_id, start)}
        if direction == "in":
            return {(target_entity, start)}
        return {(entity_id, start), (target_entity, start)}

    @staticmethod
    def _end_refs(entity_id: str, target_entity: str, end_id: Any, direction: str) -> set:
        end = str(end_id)
        if direction == "out":
            return {(target_entity, end)}
        if direction == "in":
            return {(entity_id, end)}
        return {(entity_id, end), (target_entity, end)}

    def _graph_spec(self, entity_id: str, field_name: str) -> Dict[str, Any]:
        spec = self._graph_specs.get(entity_id, {}).get(field_name)
        if spec is None:
            raise RuntimeError(f"SurrealDB backend {self.name!r}: no GraphEdge spec for {entity_id}.{field_name} (ensure not called?)")
        return spec

    def _compile_endpoint_where(self, endpoint_preds: Any) -> tuple:
        """Compile pushable endpoint predicates into a ``(where_body, params)`` pair, binding every value.

        ``endpoint_preds`` is the routed op's hint: a list of ``(field, op, value)`` with ops in the
        ``_WHERE_OPS`` subset (eq/lt/lte/gt/gte/in). Values bind to ``$pN`` parameters (never interpolated);
        field names are backtick-quoted via ``_ident``. Returns ``("", {})`` when there is nothing to push, so
        the arrow keeps its plain ``->edge->to`` shape. This prunes the *reached* node only (a strict subset of
        the unpruned result), mirroring neo4j's ``WHERE b.*`` — the routed op still re-applies the full filter.
        """
        params: Dict[str, Any] = {}
        conditions = []
        for field, op, value in endpoint_preds or []:
            name = f"p{len(params)}"
            params[name] = list(value) if op == "in" else value
            conditions.append(f"{self._ident(field)} {_WHERE_OPS[op]} ${name}")
        return (f"({' AND '.join(conditions)})", params) if conditions else ("", {})

    def _edge_expand(self, edge_table: str, from_table: str, to_table: str, ids: Sequence[str], outgoing: bool, where: tuple = ("", {})) -> List[str]:
        """Return the distinct neighbour object_ids of a frontier in one native arrow query.

        ``->edge->to`` (or ``<-edge<-to``) only resolves to records that still exist in ``to_table``, so a
        dangling edge (its endpoint deleted) is skipped — matching the portable fallback's existence filter.
        ``where`` is the ``(body, params)`` pair from ``_compile_endpoint_where``: when non-empty it fuses into
        the reached side as ``->edge->(to WHERE …)`` so SurrealDB prunes during traversal (values stay bound).
        """
        body, params = where
        reached = self._ident(to_table) if not body else f"({self._ident(to_table)} WHERE {body})"
        arrow = f"->{self._ident(edge_table)}->{reached}" if outgoing else f"<-{self._ident(edge_table)}<-{reached}"
        surql = f"RETURN array::distinct(array::flatten(SELECT VALUE {arrow}.object_id FROM type::table($ft) WHERE object_id IN $ids));"
        result = self._query(surql, {"ft": from_table, "ids": [str(item) for item in ids], **params})
        return [str(item) for item in (result or []) if item is not None]

    def _graph_neighbors(self, spec: Dict[str, Any], frontier: set, direction: str, where: tuple = ("", {})) -> set:
        """Return the typed neighbour refs of a frontier (one native query per active direction).

        With a non-empty ``where`` the arrow prunes the reached node natively, so the returned set is already
        the pushed-down subset; callers that need the *unpruned* frontier for deeper hops pass the default.
        """
        entity_id, target_entity = spec["entity_id"], spec["target_entity"]
        neighbours: set = set()
        source_ids = [object_id for entity, object_id in frontier if entity == entity_id]
        target_ids = [object_id for entity, object_id in frontier if entity == target_entity]
        if direction in ("out", "any") and source_ids:
            for nid in self._edge_expand(spec["edge_table"], spec["source_table"], spec["target_table"], source_ids, True, where):
                neighbours.add((target_entity, nid))
        if direction in ("in", "any") and target_ids:
            for nid in self._edge_expand(spec["edge_table"], spec["target_table"], spec["source_table"], target_ids, False, where):
                neighbours.add((entity_id, nid))
        return neighbours

    @backend_locked
    def graph_traverse_refs(
        self, entity_id: str, start_id: Any, field_name: str, *, min_hops: int = 1, max_hops: int = 1, direction: str = "out", endpoint_preds: Any = None
    ) -> List[tuple]:
        """Inspect typed nodes reachable through the derived ``RELATE`` shadow projection.

        This provider-level method verifies or benchmarks the projection; public workspace traversal
        uses canonical inline GraphEdge values until a versioned readiness/rebuild contract exists.
        Projection results may therefore differ from canonical rows after legacy or out-of-band writes.

        ``endpoint_preds`` (a list of ``(field, op, value)`` from a future routed op) fuse natively into the
        reached side of the arrow (``->edge->(to WHERE …)``) so SurrealDB prunes the result during traversal.
        Only the *collected* nodes are pruned — the continuation frontier stays unfiltered — so an
        intermediate node that fails the predicate still extends the walk to a deeper node that passes it
        (matching Neo4j's ``WHERE b.*`` on the reached node, never the path interior). Native pruning is thus a
        strict subset of the unpruned shadow reach. A future readiness-gated route must still re-apply the full
        filter after this candidate phase.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object identifier.
            field_name: GraphEdge field name.
            min_hops: Inclusive minimum traversal depth.
            max_hops: Inclusive maximum traversal depth.
            direction: Traversal direction. Supported values are ``out``, ``in``, and ``any``.
            endpoint_preds: Optional pushable predicates over reached endpoint fields.

        Returns:
            List[tuple]: Typed ``(entity_id, object_id)`` refs reached by traversal.

        Raises:
            RuntimeError: If the client is unavailable or no GraphEdge spec exists.
            ValueError: If hop bounds or direction are invalid.
        """
        self._need_client()
        spec = self._graph_spec(entity_id, field_name)
        direction = self._check_direction(direction)
        lo, hi = max(1, int(min_hops)), max(1, int(max_hops))
        if lo > hi:
            raise ValueError(f"GraphEdge traversal min_hops {lo} cannot exceed max_hops {hi}")
        where = self._compile_endpoint_where(endpoint_preds)
        pruned = bool(where[0])
        starts = self._start_refs(entity_id, spec["target_entity"], start_id, direction)
        reached: set = set()
        frontier = set(starts)
        for hop in range(1, hi + 1):
            more_hops = hop < hi
            # The continuation frontier advances UNPRUNED — a pruned interior node must still reach deeper
            # passing nodes — so the endpoint WHERE applies only when collecting results (mirroring neo4j's
            # WHERE on the reached node, never the path interior). The unpruned advance is computed only when a
            # deeper hop will consume it, so a single-hop / final-hop prune costs one native query, not two.
            advance = self._graph_neighbors(spec, frontier, direction) if more_hops else set()
            if hop >= lo:
                reached |= (
                    self._graph_neighbors(spec, frontier, direction, where)
                    if pruned
                    else (advance if more_hops else self._graph_neighbors(spec, frontier, direction))
                )
            if more_hops:
                if not advance:
                    break
                frontier = advance
        reached -= starts
        return sorted(reached, key=lambda ref: (ref[1], ref[0]))

    def graph_traverse(
        self, entity_id: str, start_id: Any, field_name: str, *, min_hops: int = 1, max_hops: int = 1, direction: str = "out", endpoint_preds: Any = None
    ) -> List[str]:
        """Inspect reachable object ids in the self-referential ``RELATE`` shadow.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object identifier.
            field_name: GraphEdge field name.
            min_hops: Inclusive minimum traversal depth.
            max_hops: Inclusive maximum traversal depth.
            direction: Traversal direction. Supported values are ``out``, ``in``, and ``any``.
            endpoint_preds: Optional pushable predicates over reached endpoint fields.

        Returns:
            List[str]: Object identifiers reached by traversal.
        """
        return [
            object_id
            for _entity, object_id in self.graph_traverse_refs(
                entity_id, start_id, field_name, min_hops=min_hops, max_hops=max_hops, direction=direction, endpoint_preds=endpoint_preds
            )
        ]

    @backend_locked
    def graph_shortest_path_refs(
        self, entity_id: str, start_id: Any, end_id: Any, field_name: str, *, max_hops: int = 10, direction: str = "out"
    ) -> Optional[List[tuple]]:
        """Inspect one BFS shortest path in the derived ``RELATE`` shadow projection.

        Public workspace shortest-path queries use canonical inline GraphEdge values until
        projection readiness can be proven.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object identifier.
            end_id: Target object identifier.
            field_name: GraphEdge field name.
            max_hops: Inclusive maximum path length.
            direction: Traversal direction. Supported values are ``out``, ``in``, and ``any``.

        Returns:
            Optional[List[tuple]]: Typed path refs, or ``None`` when unreachable.

        Raises:
            RuntimeError: If the client is unavailable or no GraphEdge spec exists.
            ValueError: If direction is invalid.
        """
        self._need_client()
        spec = self._graph_spec(entity_id, field_name)
        direction = self._check_direction(direction)
        starts = self._start_refs(entity_id, spec["target_entity"], start_id, direction)
        ends = self._end_refs(entity_id, spec["target_entity"], end_id, direction)
        if starts & ends:
            return [sorted(starts & ends)[0]]
        parent: Dict[tuple, Optional[tuple]] = {start: None for start in starts}
        frontier = sorted(starts)
        for _ in range(max(1, int(max_hops))):
            nxt: List[tuple] = []
            for node in frontier:
                for neighbour in sorted(self._graph_neighbors(spec, {node}, direction)):
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

    def graph_shortest_path(
        self, entity_id: str, start_id: Any, end_id: Any, field_name: str, *, max_hops: int = 10, direction: str = "out"
    ) -> Optional[List[str]]:
        """Inspect one shortest path in the derived ``RELATE`` shadow as object ids.

        Public workspace shortest-path queries use canonical inline GraphEdge values.

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
        return None if refs is None else [object_id for _entity, object_id in refs]

    def to_dict(self) -> dict[str, Any]:
        """Return SurrealDB backend metadata and connection state.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata, SurrealDB namespace/database, and provider
            connection status.
        """
        return {
            "name": self.name,
            "type": self.type,
            "url": self.url,
            "namespace": self._namespace(),
            "database": self._database(),
            "connected": self.client is not None,
            "error": self.client_error,
        }
