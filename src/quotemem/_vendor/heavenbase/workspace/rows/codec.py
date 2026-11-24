"""Workspace routed row encoding and decoding helpers."""

from __future__ import annotations

from fractions import Fraction
from math import isfinite, isnan

from heavenbase.utils import Any, Dict, Mapping, Optional

from ...execution.frame.result import ResultFrame


class _RowCodec:
    """Own backend-specific row field selection, encoding, decoding, and value comparison."""

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def field_names_for_backend(self, schema: Any, backend_name: str) -> set[str]:
        """Return row fields physically owned by one backend route."""
        ws = self.workspace
        names = {schema.pk_name}
        with ws._lock.read():
            bindings = list(ws.storage_plans[schema.entity_id].bindings)
        names.update(binding.field_name for binding in bindings if binding.backend == backend_name)
        return names

    def artifact_uses_text_storage(self, backend_name: str) -> bool:
        """Return whether a backend stores Artifact payloads as base64 text."""
        backend = self.workspace.backends.get(backend_name)
        claim = backend.tag("binary-storage")
        return claim is not None and claim.value == "base64"

    def encode_for_backend(self, schema: Any, backend_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encode canonical values into backend-ready format for a specific backend."""
        from heavenbase.utils.serialize import dumps_b64

        result: Dict[str, Any] = {}
        for name in self.field_names_for_backend(schema, backend_name):
            if name in data:
                field_schema = schema.get(name)
                result[name] = self.encode_field_value(field_schema, data[name], backend_name, dumps_b64) if field_schema else data[name]
        return result

    def encode_field_value(self, field_schema: Any, value: Any, backend_name: str, dumps_b64: Any) -> Any:
        """Encode one canonical field value, preserving null optional fields."""
        if value is None and not field_schema.required:
            return None
        encoded = field_schema.dtype.encode(value)
        if type(field_schema.dtype).identifier == "artifact" and isinstance(encoded, bytes) and self.artifact_uses_text_storage(backend_name):
            encoded = dumps_b64(encoded)
        return encoded

    def decode_field_value(self, field_schema: Any, value: Any) -> Any:
        """Decode one backend field value, preserving null optional fields."""
        if value is None and not field_schema.required:
            return None
        return field_schema.dtype.decode(value)

    def decode_row_items(self, schema: Any, backend_name: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """Decode backend values back to canonical form, keeping only owned fields."""
        names = {"object_id", *self.field_names_for_backend(schema, backend_name)}
        result: Dict[str, Any] = {}
        for key, value in row.items():
            if key in names:
                field_schema = schema.get(key)
                result[key] = self.decode_field_value(field_schema, value) if field_schema and key != "object_id" else value
        return result

    def decode_schema_row(self, schema: Any, row: Dict[str, Any]) -> Dict[str, Any]:
        """Decode backend storage values to canonical Python for one logical row."""
        out = dict(row)
        for name, field_schema in schema.fields.items():
            if name in out:
                out[name] = self.decode_field_value(field_schema, out[name])
        return out

    def decode_schema_frame(self, schema: Any, frame: ResultFrame) -> ResultFrame:
        """Decode backend storage values to canonical Python for one query frame."""
        if len(frame) == 0:
            return frame
        return ResultFrame.from_rows([self.decode_schema_row(schema, row) for row in frame.rows()], frame.col_names(), metadata_columns=frame.metadata_columns)

    def normalize_row_value(self, value: Any) -> Any:
        """Normalize provider-returned arrays before row-fragment comparison."""
        if hasattr(value, "tolist"):
            return self.normalize_row_value(value.tolist())
        if isinstance(value, tuple):
            return [self.normalize_row_value(item) for item in value]
        if isinstance(value, list):
            return [self.normalize_row_value(item) for item in value]
        if isinstance(value, Mapping):
            return {key: self.normalize_row_value(item) for key, item in value.items()}
        return value

    def same_row_value(self, left: Any, right: Any) -> bool:
        """Return value equality without NumPy/Pandas ambiguous truth errors."""
        left = self.normalize_row_value(left)
        right = self.normalize_row_value(right)
        numeric = self.same_numeric_value(left, right)
        if numeric is not None:
            return numeric
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(self.same_row_value(left_item, right_item) for left_item, right_item in zip(left, right))
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return set(left) == set(right) and all(self.same_row_value(left[key], right[key]) for key in left)
        try:
            same = left == right
        except ValueError:
            return False
        if hasattr(same, "all"):
            return bool(same.all())
        return bool(same)

    def same_field_value(self, schema: Any, field_name: str, left: Any, right: Any) -> bool:
        """Compare one logical field, treating GraphEdge rows as an endpoint-keyed set."""
        from ...strategies import GraphEdge
        from ...strategies.graph_edge import resolve_graph_endpoint

        bindings = self.workspace.storage_plans[schema.entity_id].field_bindings(field_name)
        graph_bindings = [binding for binding in bindings if binding.strategy == GraphEdge.identifier]
        if not graph_bindings:
            return self.same_row_value(left, right)
        target = (graph_bindings[0].params or {}).get("target")
        endpoint_col, _target_entity, _prop_cols = resolve_graph_endpoint(schema.get(field_name), target, schema.entity_id)
        left_valid, left_edges = self._graph_edges_by_target(left, endpoint_col)
        right_valid, right_edges = self._graph_edges_by_target(right, endpoint_col)
        if not left_valid or not right_valid:
            return self.same_row_value(left, right)
        return self.same_row_value(left_edges, right_edges)

    def _graph_edges_by_target(self, value: Any, endpoint_col: str) -> tuple[bool, dict[str, Any]]:
        """Return a GraphEdge value keyed by typed target, or report malformed input."""
        if value is None:
            value = []
        if not isinstance(value, (list, tuple)):
            return False, {}
        edges: dict[str, Any] = {}
        for edge in value:
            if not isinstance(edge, Mapping) or edge.get(endpoint_col) is None:
                return False, {}
            target = str(edge[endpoint_col])
            if target in edges:
                return False, {}
            edges[target] = self.normalize_row_value(edge)
        return True, edges

    def same_numeric_value(self, left: Any, right: Any) -> Optional[bool]:
        """Distinguish Booleans and compare numbers with floating-point tolerance; equal infinities and two NaNs agree.

        Only values whose type converts itself to a float count as numbers.
        ``float()`` also parses text, so ``str``, ``bytes``, ``bytearray`` and
        ``memoryview`` values that look like numbers are compared exactly by the
        caller, and ``None`` is returned for them.
        """
        if isinstance(left, bool) or isinstance(right, bool):
            return left is right
        if not (self._is_number(left) and self._is_number(right)):
            return None
        if isinstance(left, int) and isinstance(right, int):
            return left == right
        if (isinstance(left, int) and isinstance(right, float)) or (isinstance(left, float) and isinstance(right, int)):
            floating = left if isinstance(left, float) else right
            return isfinite(floating) and (left == right or abs(Fraction(left) - Fraction(right)) <= 1e-6)
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError, OverflowError):
            return None
        if left_value == right_value or (isnan(left_value) and isnan(right_value)):
            return True
        return abs(left_value - right_value) <= 1e-6

    @staticmethod
    def _is_number(value: Any) -> bool:
        """Return whether ``value``'s type converts to a float without parsing text."""
        value_type = type(value)
        return hasattr(value_type, "__float__") or hasattr(value_type, "__index__")
