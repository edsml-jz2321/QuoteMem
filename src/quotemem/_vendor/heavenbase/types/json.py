"""Json logical type."""

from __future__ import annotations

import math

__all__ = [
    "Json",
]

from heavenbase.utils import Any, Mapping, Sequence

from ..modules import ModuleResolver
from ..utils.serialize import _json_restore, dumps_json, loads_json
from .base import LogicalType


class Json(LogicalType):
    """HeavenBase JSON payload logical type.

    Storage accepts the extended values supported by :func:`dumps_json`, such
    as tuples, sets, dates, large integers, and callable references. Backends
    store tagged compact JSON strings and restore those values with
    :meth:`decode`. Query operands use a separate strict JSON normalization in
    :meth:`filter_value` so portable comparisons keep JSON type semantics.

    Warnings:
        :meth:`validate` parses JSON strings but does not enforce a fixed
        schema. Use entity-level constraints or application validation for
        structured payloads.

    Examples:
        ``hb.Json().validate({"a": 1})`` returns ``{"a": 1}``.
        ``hb.Json().validate('{"a": 1}')`` parses the string first.
    """

    identifier = "json"

    @classmethod
    def from_dict(cls, spec: dict, *, resolver: ModuleResolver | None = None) -> "Json":
        """Build a Json logical type from a serialized descriptor.

        Args:
            spec (dict): Serialized descriptor. Json accepts only the type
                selector and ignores extra keys.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            Json: Configured JSON logical type.
        """
        return cls()

    def validate(self, value: Any) -> Any:
        """Normalize a storage value to its extended Python representation.

        Args:
            value: Extended JSON value, encoded JSON string, or array-like
                object with ``tolist()``.

        Returns:
            Parsed or normalized storage value.

        Raises:
            ValueError: If a string value is not valid JSON.
        """
        if isinstance(value, str):
            try:
                return loads_json(value)
            except Exception as exc:
                raise ValueError(f"Json cannot parse {value!r}") from exc
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def encode(self, value: Any) -> str:
        """Encode an extended value to tagged compact backend storage.

        Args:
            value: Extended Python payload accepted by :func:`dumps_json`.

        Returns:
            Tagged compact JSON string with sorted object keys.

        Raises:
            TypeError: If ``value`` cannot be adapted or serialized.
            ValueError: If the JSON encoder rejects ``value``.
        """
        return dumps_json(self.validate(value), sort_keys=True, compact=True)

    def decode(self, value: Any) -> Any:
        """Restore tagged backend JSON to its extended Python value.

        Args:
            value: Backend storage value, usually a JSON string. A backend that
                parses JSON itself (Milvus does) returns the parsed object or
                array, whose tagged values are restored the same way.

        Returns:
            Restored extended JSON value, or ``None`` for an optional null
            storage value.

        Raises:
            TypeError: If the decoded tagged payload has an invalid shape.
            ValueError: If a string is corrupt JSON or has an invalid tagged
                value.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return loads_json(value)
        if isinstance(value, (dict, list)):
            return _json_restore(value)
        return value

    def path_value(self, value: Any, path: Sequence[str]) -> Any:
        """Return the nested payload value at one key path.

        This is the single traversal shared by ``get`` filters and derived
        dotted-path columns: the raw value decodes through :meth:`decode`
        first, then each key walks one nested object level.

        Args:
            value: Raw backend storage value.
            path: Nested object keys, such as ``["review", "score"]``.

        Returns:
            Nested value, or ``None`` when the path is missing or ``null``,
            or when it crosses a non-object value.

        Raises:
            TypeError: If the decoded tagged payload has an invalid shape.
            ValueError: If a string is corrupt JSON or has an invalid tagged
                value.
        """
        found = self.decode(value)
        for part in path:
            if not isinstance(found, dict):
                return None
            found = found.get(part)
        return found

    def filter_value(self, op: str, value: Any) -> Any:
        """Normalize whole-value and ``get`` query operands.

        Args:
            op: Operation identifier.
            value: User-authored operand. For ``get`` this is a mapping with a
                ``path`` (dotted string or list of keys) plus exactly one
                comparison entry, such as ``{"path": "venue", "eq": "NeurIPS"}``.

        ``None`` stays ``None`` for whole-value ``eq``, ``ne`` and ``in``, so
        the filter tests for a missing value.

        Returns:
            Strict JSON query operand. Tuples become JSON arrays, tagged
            storage markers remain plain objects, and other whole-value
            operands must be objects or arrays. For ``get``: ``{"path": [...],
            "op": ..., "value": ...}``.

        Raises:
            TypeError: If an ``in`` operand is not a container, a ``get``
                payload is not a mapping, a path segment is not a string, or
                an operand is outside the strict JSON domain.
            ValueError: If the ``get`` payload is missing a path, carries no
                comparison, uses an unsupported comparison operation, or
                contains a non-finite number.
        """
        if op in ("eq", "ne"):
            return None if value is None else _strict_json_container(value)
        if op == "in":
            if not isinstance(value, (list, tuple, set)):
                raise TypeError("Json in filters expect a list, tuple, or set")
            return [None if item is None else _strict_json_container(item) for item in value]
        if op != "get":
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f'Json get filters expect a mapping like {{"path": "venue", "eq": ...}}, got {type(value).__name__}')
        payload = dict(value)
        raw_path = payload.pop("path", None)
        if "op" in payload and "value" in payload and not (set(payload) - {"op", "value"}):
            inner_op, inner_value = payload["op"], payload["value"]
        else:
            if len(payload) != 1:
                raise ValueError(f'Json get filters need exactly one comparison beside "path", got {sorted(payload) or "none"}')
            inner_op, inner_value = next(iter(payload.items()))
        inner_op = str(inner_op).strip()
        if not inner_op:
            raise ValueError("Json get filters require a non-empty comparison operation")
        if raw_path is None:
            raise ValueError('Json get filters need a "path" of nested keys, such as "venue" or ["author", "name"]')
        if isinstance(raw_path, str):
            path = raw_path.split(".")
        elif isinstance(raw_path, (list, tuple)):
            path = list(raw_path)
        else:
            raise TypeError(f"Json get filter paths expect a dotted string, list, or tuple; got {type(raw_path).__name__}")
        if any(not isinstance(part, str) for part in path):
            raise TypeError(f"Json get filter path segments must be strings, got {raw_path!r}")
        if not path or any(not part for part in path):
            raise ValueError(f"Json get filters need non-empty path segments, got {raw_path!r}")
        if inner_op == "in":
            if not isinstance(inner_value, (list, tuple, set)):
                raise TypeError("Json get in filters expect a list, tuple, or set")
            inner_value = [_canonical_json(item) for item in inner_value]
        else:
            inner_value = _canonical_json(inner_value)
        return {"path": path, "op": inner_op, "value": inner_value}


_JSON_GET_OPS = ("eq", "ne", "in", "lt", "lte", "gt", "gte", "contains")


def _canonical_json(value: Any) -> Any:
    """Normalize one recursively strict JSON value without tagged restoration."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Json numbers must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Json object keys must be strings")
        return {key: _canonical_json(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if converted is not value:
            return _canonical_json(converted)
    raise TypeError(f"Json values must use null, bool, number, string, list, tuple, or string-keyed mapping; got {type(value).__name__}")


def _canonical_json_container(value: Any) -> Any:
    """Normalize one top-level JSON object or array."""
    canonical = _canonical_json(value)
    if not isinstance(canonical, (dict, list)):
        raise TypeError(f"Json values must be objects or arrays, got {type(canonical).__name__}")
    return canonical


def _strict_json_container(value: Any) -> Any:
    """Parse and normalize one strict object or array query operand."""
    if isinstance(value, str):
        try:
            value = loads_json(value, restore=False)
        except Exception as exc:
            raise ValueError(f"Json filter cannot parse {value!r}") from exc
    return _canonical_json_container(value)
