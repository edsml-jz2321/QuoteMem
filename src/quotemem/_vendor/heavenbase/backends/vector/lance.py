"""LanceDB vector backend with native storage and vector search."""

from __future__ import annotations

__all__ = [
    "LanceBackend",
]

from heavenbase.utils import Any, Iterable, Mapping, Sequence, dumps_json, loads_json, safe_error

from heavenbase.utils.files import delete_dir, file_uri_to_path, touch_dir
from heavenbase.utils.files import pj
from ...execution.frame.result import ResultFrame
from ...modules import ModuleResolver
from ...schema.model import EntitySchema
from ...types.storage import stores_as_integer
from ...types.vector import _scale_cosine_query
from ..base import Backend, backend_locked
from heavenbase.execution.contracts import LOGICAL_OPS, QueryFragment, RowOp

_SQL_COMMON_RESERVED_WORDS = frozenset(
    "all alter and any as asc between by case cast check column constraint create cross current current_date "
    "current_time current_timestamp delete desc distinct drop else end except exists false for foreign from full "
    "grant group having in inner insert intersect into is join left like limit natural not null on or order outer "
    "primary references right select set table then true union unique update using values when where with".split()
)
LANCE_RESERVED_WORDS = _SQL_COMMON_RESERVED_WORDS | frozenset("anti arrow boolean date decimal explain interval lateral regexp struct timestamp".split())


def _to_arrow_schema(schema: EntitySchema) -> Any:
    import pyarrow as pa

    fields = [pa.field("object_id", pa.string())]
    for field_schema in schema.fields.values():
        if field_schema.name == "object_id":
            continue
        identifier = type(field_schema.dtype).identifier
        if identifier == "vector":
            fields.append(pa.field(field_schema.name, pa.list_(pa.float32(), field_schema.dtype.dim) if field_schema.dtype.dim else pa.list_(pa.float32())))
        elif stores_as_integer(field_schema.dtype):
            fields.append(pa.field(field_schema.name, pa.int64()))
        elif identifier == "float":
            fields.append(pa.field(field_schema.name, pa.float64()))
        elif identifier == "boolean":
            fields.append(pa.field(field_schema.name, pa.bool_()))
        else:
            # text, Json (already JSON text) and Array (written as JSON text by upsert)
            fields.append(pa.field(field_schema.name, pa.string()))
    return pa.schema(fields)


def _lit(value: Any) -> str:
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _id_lookup_batch_size(config: Any) -> int:
    return max(1, int(config.get("heavenbase.backend_defaults.lance.id_lookup.batch_size", default=500)))


def _batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _id_filter(object_ids: Sequence[Any]) -> str:
    return f"object_id IN ({', '.join(_lit(object_id) for object_id in object_ids)})"


def _column_values(column: Any) -> list[Any]:
    """Decode an Arrow column in row order without concatenating its chunks."""
    import pyarrow as pa

    arrow_type = column.type
    if not (pa.types.is_fixed_size_list(arrow_type) or pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)):
        return column.to_pylist()
    if not (pa.types.is_floating(arrow_type.value_type) or pa.types.is_integer(arrow_type.value_type)):
        return column.to_pylist()
    cells = []
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else (column,)
    for chunk in chunks:
        values = _numeric_list_cells(chunk)
        if values is None:
            return column.to_pylist()
        cells.extend(values)
    return cells


def _numeric_list_cells(array: Any) -> list[Any] | None:
    """Copy one Arrow list chunk into writable NumPy slices, with ``None`` for null cells.

    A null item inside a present cell requires decoding the column as Python
    lists so the null item survives. Null items inside a missing cell can be
    ignored for floats, where Arrow can represent them as NaN in the copy.
    """
    import numpy as np
    import pyarrow as pa

    arrow_type = array.type
    count = len(array)
    if pa.types.is_fixed_size_list(arrow_type):
        size = arrow_type.list_size
        bounds = np.arange(count + 1, dtype=np.int64) * size
        child = array.values.slice(array.offset * size, count * size)
    else:
        offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        bounds = offsets - offsets[0]
        child = array.values.slice(int(offsets[0]), int(bounds[-1]))

    present = array.is_valid().to_numpy(zero_copy_only=False) if array.null_count else None
    if child.null_count:
        if present is None or not pa.types.is_floating(child.type):
            return None
        missing = np.flatnonzero(~child.is_valid().to_numpy(zero_copy_only=False))
        if present[np.searchsorted(bounds, missing, side="right") - 1].any():
            return None

    flat = child.to_numpy(zero_copy_only=False, writable=True)
    if pa.types.is_fixed_size_list(arrow_type):
        cells = list(flat.reshape(count, arrow_type.list_size))
    else:
        cells = [flat[bounds[index] : bounds[index + 1]] for index in range(count)]
    return cells if present is None else [cell if valid else None for cell, valid in zip(cells, present)]


def _records(result: Any, json_fields: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Return LanceDB rows as Python values.

    Arrow keeps a null as ``None`` and an ``int64`` as ``int``. Numeric list
    columns use NumPy slices when all present vectors have non-null items.
    Reading columns without concatenation also preserves chunked string
    results whose aggregate payload would overflow 32-bit Arrow offsets. JSON text columns decode to their stored values.

    Args:
        result (Any): LanceDB table or query builder exposing ``to_arrow()``.
        json_fields (Sequence[str]): Columns holding JSON text written by ``upsert``.

    Returns:
        list[dict[str, Any]]: One mapping per row, with ``object_id`` present.
    """
    table = result.to_arrow()
    columns = {name: _column_values(table.column(name)) for name in table.column_names}
    for name in json_fields:
        if name in columns:
            columns[name] = [None if value is None else loads_json(value) for value in columns[name]]
    rows = [dict(zip(columns, values)) for values in zip(*columns.values())]
    for row in rows:
        row["object_id"] = row.get("object_id", "")
    return rows


def _json_text(row: dict[str, Any], json_fields: Sequence[str]) -> dict[str, Any]:
    """Return ``row`` with each JSON-stored field written as JSON text."""
    for name in json_fields:
        value = row.get(name)
        if value is not None and not isinstance(value, str):
            row[name] = dumps_json(value)
    return row


class LanceBackend(Backend):
    """LanceDB provider adapter with native vector search and scalar where."""

    identifier = "lance"
    provider_reserved_words = LANCE_RESERVED_WORDS

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "LanceBackend":
        """Build the LanceDB backend from a config mapping.

        Args:
            name (str): Workspace backend instance name.
            data (Mapping[str, Any]): Backend config mapping. Supported keys are LanceDB
                constructor parameters, including ``uri`` and ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            LanceBackend: Configured LanceDB backend instance.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        return cls(name, cfg.pop("uri", None), ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "lance",
        uri: str | None = None,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a LanceDB backend.

        Args:
            name (str): Workspace backend instance name.
            uri (str | None): LanceDB directory or URI. When omitted, uses the configured
                local LanceDB default.
            ws_id (str | None): Optional workspace namespace appended to local LanceDB paths.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs (Any): Extra keyword arguments passed to ``lancedb.connect``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        default_uri = self._construction_config_get("heavenbase.backend_defaults.lance.uri", default="file://%/lance")
        selected = str(uri or default_uri)
        self._remote = "://" in selected and not selected.startswith("file://")
        self._base_uri = selected if self._remote else file_uri_to_path(selected, home=self.config.root)
        self._connect_kwargs = dict(kwargs)
        self._schemas: dict[str, EntitySchema] = {}
        self._connect()

    def _connect(self) -> None:
        if self._remote and self.ws_id:
            raise ValueError("Remote Lance URI cannot be combined with ws_id until a provider-native namespace strategy is registered")
        self.uri = pj(self._base_uri, self.ws_id) if self.ws_id and not self._remote else self._base_uri
        import lancedb

        if not self._remote:
            touch_dir(self.uri)
        self.client = lancedb.connect(self.uri, **self._connect_kwargs)

    def _refresh_workspace(self) -> None:
        self._schemas.clear()
        self._connect()

    @backend_locked
    def drop(self) -> None:
        """Delete the LanceDB directory for the current workspace.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        if not self._remote:
            delete_dir(self.uri)
        self._schemas.clear()

    def _table_name(self, entity_id: str) -> str:
        return self._physical_name(entity_id)

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> list[str]:
        """Return LanceDB table and field-name warnings for this schema.

        Args:
            schema (EntitySchema): Entity schema being registered.
            bindings (Sequence[Any]): Storage bindings for the entity.

        Returns:
            list[str]: Provider naming warnings for physical table and field names.
        """
        warnings = super().schema_warnings(schema, bindings)
        warnings.extend(
            self._physical_name_warnings(
                [(schema.entity_id, self._table_name(schema.entity_id))],
                kind="LanceDB table",
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        warnings.extend(
            self._physical_name_warnings(
                [(field_name, field_name) for field_name in sorted(schema.fields)],
                kind=f"LanceDB field for {schema.entity_id!r}",
                reserved=self.provider_reserved_words,
                case_sensitive=False,
            )
        )
        return warnings

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure a LanceDB table for the entity schema.

        Args:
            schema (EntitySchema): Entity schema to store in LanceDB.
            bindings (Sequence[Any]): Storage bindings supplied by the workspace.

        Returns:
            None: This function does not return a value.
        """
        if self._schemas.get(schema.entity_id) == schema:
            return
        table_name = self._table_name(schema.entity_id)
        arrow_schema = _to_arrow_schema(schema)
        try:
            table = self.client.open_table(table_name)
        except Exception:
            self.client.create_table(table_name, schema=arrow_schema)
        else:
            # a field added since the table was created (alter_entity, or a wider definition on reopen)
            # becomes a null column, so its values are stored instead of dropped
            stored = set(table.schema.names)
            missing = [arrow_field for arrow_field in arrow_schema if arrow_field.name not in stored]
            if missing:
                table.add_columns(missing)
        self._schemas[schema.entity_id] = schema

    def _table(self, entity_id: str) -> Any:
        return self.client.open_table(self._table_name(entity_id))

    def _json_fields(self, entity_id: str) -> tuple[str, ...]:
        """Return the fields this backend stores as JSON text: an Array has no fixed Arrow item type."""
        schema = self._schemas.get(entity_id)
        if schema is None:
            return ()
        return tuple(name for name, field_schema in schema.fields.items() if type(field_schema.dtype).identifier == "array")

    def _optional_table(self, entity_id: str) -> Any | None:
        """Open an existing table without hiding provider failures."""
        table_name = self._table_name(entity_id)
        try:
            return self.client.open_table(table_name)
        except ValueError as exc:
            if str(exc) == f"Table {table_name!r} was not found":
                return None
            raise RuntimeError(f"LanceDB could not open table {table_name!r}: {safe_error(exc)}") from exc
        except Exception as exc:
            raise RuntimeError(f"LanceDB could not open table {table_name!r}: {safe_error(exc)}") from exc

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply row mutations to LanceDB tables.

        Args:
            rows (Sequence[RowOp]): Row mutations to apply. ``delete`` operations remove
                rows; other operations merge or insert row values.

        Returns:
            None: This function does not return a value.
        """
        import pyarrow as pa

        for entity_id, is_delete, ops in self.ordered_row_op_batches(rows):
            table = self._table(entity_id)
            if is_delete:
                for batch in _batches([row.object_id for row in ops], _id_lookup_batch_size(self.config)):
                    table.delete(f"object_id IN ({', '.join(_lit(row_id) for row_id in batch)})")
                continue
            json_fields = self._json_fields(entity_id)
            items = [_json_text(row.to_row(), json_fields) for row in ops]
            arrow_schema = _to_arrow_schema(self._schemas[entity_id]) if entity_id in self._schemas else None
            arrow_tbl = pa.Table.from_pylist(items, schema=arrow_schema)
            # LanceDB commits the merge as one transaction. A delete/add fallback
            # would expose missing rows, and rewinding the table can erase peer writes.
            table.merge_insert("object_id").when_matched_update_all().when_not_matched_insert_all().execute(arrow_tbl)

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return all rows from a LanceDB table.

        Args:
            entity_id (str): Entity identifier whose table should be scanned.

        Returns:
            list[dict[str, Any]]: Stored rows, or an empty list when the table is absent.
        """
        table = self._optional_table(entity_id)
        if table is None:
            return []
        return _records(table, self._json_fields(entity_id))

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return rows with batched LanceDB object-id filters.

        Args:
            entity_id (str): Entity identifier whose rows should be loaded.
            object_ids (Iterable[Any]): Object identifiers to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any] | None]: Rows aligned with ``object_ids``. Missing rows are
            returned as ``None``.

        Raises:
            RuntimeError: If LanceDB rejects the lookup query.
        """
        ordered_ids = list(object_ids)
        if not ordered_ids:
            return []
        table = self._optional_table(entity_id)
        if table is None:
            return [None for _ in ordered_ids]
        unique_ids = list(dict.fromkeys(ordered_ids))
        rows_by_id: dict[Any, dict[str, Any]] = {}
        json_fields = self._json_fields(entity_id)
        try:
            for batch in _batches(unique_ids, _id_lookup_batch_size(self.config)):
                for row in _records(table.search().where(_id_filter(batch)).limit(len(batch)), json_fields):
                    rows_by_id[row["object_id"]] = row
        except Exception as exc:
            raise RuntimeError(f"LanceDB get_many failed for {entity_id!r}: {safe_error(exc)}") from exc
        return self._project_rows([rows_by_id.get(row_id) for row_id in ordered_ids], columns)

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags through LanceDB object-id filters.

        Args:
            entity_id (str): Entity identifier whose rows should be checked.
            object_ids (Iterable[Any]): Object identifiers to check in caller order.

        Returns:
            list[bool]: Existence flags aligned with ``object_ids``.
        """
        return [row is not None for row in self.get_many(entity_id, object_ids)]

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return LanceDB table row count without materializing rows.

        Args:
            entity_id (str): Entity identifier whose table should be counted.

        Returns:
            int: Number of rows in the table, or ``0`` when the table is absent.

        Raises:
            RuntimeError: If LanceDB rejects the count query.
        """
        table = self._optional_table(entity_id)
        if table is None:
            return 0
        try:
            return int(table.count_rows())
        except Exception as exc:
            raise RuntimeError(f"LanceDB count failed for {entity_id!r}: {safe_error(exc)}") from exc

    def candidate_id_fragment(self, object_ids: Sequence[Any]) -> QueryFragment:
        """Return a LanceDB object-id predicate for candidate prefiltering.

        Args:
            object_ids (Sequence[Any]): Candidate object identifiers.

        Returns:
            QueryFragment: Native ``object_id IN (...)`` fragment for LanceDB.
        """
        py_fragment = super().candidate_id_fragment(object_ids)
        return QueryFragment(self.name, "in", "object_id", _id_filter(object_ids), notes={**py_fragment.notes, "kind": "lance_where", "mode": "native"})

    @backend_locked
    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute LanceDB-native vector or scalar fragments when possible.

        Args:
            fragment (QueryFragment): Backend-ready query fragment compiled by the handler layer.

        Returns:
            ResultFrame: Matching rows and object identifiers.

        Raises:
            NotImplementedError: If the fragment cannot be executed natively or by scan fallback.
            RuntimeError: If LanceDB rejects a vector or filter query.
        """
        entity_id = fragment.entity_id()
        if fragment.kind() == "lance_search" and fragment.op == "near":
            params = fragment.payload
            filt = fragment.filter_fragment()
            if filt is not None and not self._is_lance_fragment(filt):
                return super().execute(fragment)
            table = self._optional_table(entity_id)
            if table is None:
                return ResultFrame.empty()
            try:
                query = params["vector"]
                if params.get("metric") == "cosine":
                    # LanceDB stores and searches float32; cosine ignores the exact power-of-two scale.
                    query = _scale_cosine_query(query)
                search = table.search(query, vector_column_name=fragment.field)
                if filt is not None:
                    search = search.where(self._where(filt))
                if params.get("metric") and hasattr(search, "metric"):
                    search = search.metric(params["metric"])
                rows = []
                for data in _records(search.limit(params["top_k"]), self._json_fields(entity_id)):
                    distance = data.pop("_distance", 0.0)
                    # LanceDB's cosine distance is 1 - cos and its dot distance is 1 - x.y, so both
                    # turn back into the similarity the scan scorers return; l2 is a squared distance.
                    data["score"] = 1.0 - distance if params.get("metric") in ("cosine", "dot") else -distance
                    rows.append(data)
                return ResultFrame.from_rows(rows)
            except Exception as exc:
                raise RuntimeError(f"LanceDB vector query failed for {entity_id!r}: {safe_error(exc)}") from exc
        if fragment.kind() == "lance_where" or fragment.op in LOGICAL_OPS:
            if not self._is_lance_fragment(fragment):
                return super().execute(fragment)
            try:
                table = self._table(entity_id)
                search = table.search().where(self._where(fragment))
                return ResultFrame.from_rows(_records(search, self._json_fields(entity_id)))
            except Exception as exc:
                raise RuntimeError(f"LanceDB filter query failed for {entity_id!r}: {safe_error(exc)}") from exc

        if fragment.mode() == "py_scan":
            return super().execute(fragment)
        raise NotImplementedError(f"LanceDB cannot execute fragment {fragment.op!r}")

    def _is_lance_fragment(self, fragment: QueryFragment) -> bool:
        if fragment.kind() == "lance_where":
            return True
        if fragment.op in ("and", "or"):
            return all(self._is_lance_fragment(child) for child in fragment.children())
        if fragment.op == "not":
            return self._is_lance_fragment(fragment.children()[0])
        return False

    def _where(self, fragment: QueryFragment) -> str:
        if fragment.kind() == "lance_where":
            return fragment.payload
        if fragment.op == "and":
            return " AND ".join(f"({self._where(child)})" for child in fragment.children())
        if fragment.op == "or":
            return " OR ".join(f"({self._where(child)})" for child in fragment.children())
        if fragment.op == "not":
            return f"NOT ({self._where(fragment.children()[0])})"
        raise TypeError(f"Cannot compile {fragment.op!r} to LanceDB where")

    def to_dict(self) -> dict[str, Any]:
        """Return LanceDB backend metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata and active LanceDB URI.
        """
        return {"name": self.name, "type": self.type, "uri": self.uri}
