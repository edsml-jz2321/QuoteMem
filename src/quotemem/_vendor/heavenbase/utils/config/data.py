"""Nested data operations for HeavenBase configuration."""

from __future__ import annotations

__all__ = ["copy_data", "dflat", "dget", "dmerge", "dset", "dsetdef", "dunflat", "dunset", "to_mutable_data"]

import copy
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

_REMOVE = "__HB_REMOVE__"
_OVERWRITE = "__HB_OVERWRITE__"
_DOT = "#HB_DOT#"
_ATOMIC_TYPES = frozenset((str, int, float, bool, bytes, type(None)))
_COPY_MISSING = object()


def copy_data(value: Any) -> Any:
    """
    Return an isolated copy of JSON-like data.

    ``dict``, ``list``, and ``tuple`` containers are rebuilt recursively and
    their ``str``, ``int``, ``float``, ``bool``, ``bytes``, and ``None`` leaves
    are shared, so a row holding a long numeric vector is copied by one list
    construction instead of one ``copy.deepcopy`` dispatch per element. Any
    other object, including container subclasses, is deep-copied, so the result
    is as detached from the input as ``copy.deepcopy`` for row, record, and
    configuration data. Mapping keys are shared. Repeated references and
    cycles are preserved within one copy.

    Args:
        value (Any): Row, record, or configuration value to copy.

    Returns:
        Any: A copy that shares no mutable container with ``value``.

    Raises:
        TypeError: If a leaf cannot be deep-copied.

    Examples:
        >>> from heavenbase.utils.config import copy_data
        >>> source = {"embedding": [1.0, 2.0], "tags": ("a",)}
        >>> copied = copy_data(source)
        >>> copied == source and copied["embedding"] is not source["embedding"]
        True
    """
    return _copy_data(value, {})


def _copy_data(value: Any, memo: dict[int, Any]) -> Any:
    kind = type(value)
    if kind in _ATOMIC_TYPES:
        return value
    identity = id(value)
    cached = memo.get(identity, _COPY_MISSING)
    if cached is not _COPY_MISSING:
        return cached
    if kind is dict:
        result: dict[Any, Any] = {}
        memo[identity] = result
        for key, item in value.items():
            result[key] = item if type(item) in _ATOMIC_TYPES else _copy_data(item, memo)
        return result
    if kind is list:
        items: list[Any] = []
        memo[identity] = items
        items.extend(item if type(item) in _ATOMIC_TYPES else _copy_data(item, memo) for item in value)
        return items
    if kind is tuple:
        copied = tuple(item if type(item) in _ATOMIC_TYPES else _copy_data(item, memo) for item in value)
        # A mutable child can reach this tuple again and finish copying it first.
        return memo.setdefault(identity, copied)
    return copy.deepcopy(value, memo)


def _split_path(path: str) -> List[str]:
    return [part.replace(_DOT, ".") for part in path.replace("\\.", _DOT).split(".") if part]


def _list_tail(path: str) -> Optional[Tuple[str, str]]:
    parts = _split_path(path)
    if not parts:
        return None
    last = parts[-1]
    if not last.endswith("]") or "[" not in last:
        return None
    key, op = last[:-1].rsplit("[", 1)
    if not key or op == "*":
        raise ValueError(f"Unsupported list path {path!r}")
    return path[: path.rindex("[")], op


def dget(data: Any, path: Optional[str] = None, default: Any = None) -> Any:
    """
    Read a nested dictionary or list value with a dotted path.

    Use this helper for nested config dictionaries and scoped runtime configuration layers.

    Args:
        data (Any): Dictionary layer or payload to read, write, merge, or render.
        path (Optional[str]): Local path, config path, or dotted dictionary path used by the helper. Defaults to ``None``.
        default (Any): Fallback value used when a path or config key is missing. Defaults to ``None``.

    Returns:
        Any: The one nested value from a mapping/list structure by dot path.

    Examples:
        >>> from heavenbase.utils.config import dget
        >>> dget({"a": {"b": 2}}, "a.b")
        2
    """
    if path is None:
        return data
    cur = data
    for part in _split_path(path):
        if part.endswith("]") and "[" in part:
            key, idx = part[:-1].rsplit("[", 1)
            if key:
                if not isinstance(cur, Mapping) or key not in cur or not isinstance(cur[key], (list, tuple)):
                    return default
                values = cur[key]
            elif isinstance(cur, (list, tuple)):
                values = cur
            else:
                return default
            try:
                cur = values[int(idx)]
            except Exception:
                return default
        elif isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _parse_list_part(part: str) -> Optional[Tuple[str, str]]:
    if not part.endswith("]") or "[" not in part:
        return None
    key, op = part[:-1].rsplit("[", 1)
    return key, op


def _list_set(values: List[Any], op: str, value: Any) -> bool:
    if op in ("", "-"):
        values.append(value)
        return True
    if op.endswith("+"):
        values.insert(int(op[:-1]), value)
        return True
    idx = int(op)
    if idx < -len(values):
        return False
    if idx >= len(values):
        values.extend([None] * (idx - len(values) + 1))
    values[idx] = value
    return True


def _list_get_or_create(values: List[Any], op: str) -> Optional[Dict[str, Any]]:
    if op in ("", "-"):
        values.append({})
        return values[-1]
    if op.endswith("+"):
        item: Dict[str, Any] = {}
        values.insert(int(op[:-1]), item)
        return item
    idx = int(op)
    if idx < -len(values):
        return None
    if idx >= len(values):
        values.extend({} for _ in range(idx - len(values) + 1))
    if not isinstance(values[idx], dict):
        values[idx] = {}
    return values[idx]


def dset(data: Dict[str, Any], path: Optional[str], value: Any = None) -> bool:
    """
    Set a nested dictionary or list value with a dotted path.

    Use this helper for nested config dictionaries and scoped runtime configuration layers.

    Args:
        data (Dict[str, Any]): Dictionary layer or payload to read, write, merge, or render.
        path (Optional[str]): Local path, config path, or dotted dictionary path used by the helper.
        value (Any): Value to read, set, validate, hash, or format. Defaults to ``None``.

    Returns:
        bool: The result of ``dset``.

    Examples:
        >>> from heavenbase.utils.config import dset
        >>> data = {}
        >>> dset(data, "a.b", 2)
        >>> data
        {'a': {'b': 2}}
    """
    if path is None:
        if not isinstance(value, Mapping):
            return False
        data.clear()
        data.update(copy.deepcopy(value or {}))
        return True
    if not isinstance(data, dict):
        return False
    cur = data
    parts = _split_path(path)
    if not parts:
        return False
    for part in parts[:-1]:
        parsed = _parse_list_part(part)
        if parsed:
            key, op = parsed
            if key:
                if not isinstance(cur, dict):
                    return False
                values = cur.get(key)
                if not isinstance(values, list):
                    values = []
                    cur[key] = values
            elif isinstance(cur, list):
                values = cur
            else:
                return False
            nxt = _list_get_or_create(values, op)
            if nxt is None:
                return False
            cur = nxt
            continue
        if not isinstance(cur, dict):
            return False
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    last = parts[-1]
    parsed = _parse_list_part(last)
    if parsed:
        key, op = parsed
        if key:
            if not isinstance(cur, dict):
                return False
            values = cur.get(key)
            if not isinstance(values, list):
                values = []
                cur[key] = values
        elif isinstance(cur, list):
            values = cur
        else:
            return False
        return _list_set(values, op, value)
    if not isinstance(cur, dict):
        return False
    cur[last] = value
    return True


def dunset(data: Dict[str, Any], path: Optional[str]) -> bool:
    """
    Remove one nested value from a mapping/list structure by dot path.

    Use this helper for nested config dictionaries and scoped runtime configuration layers.

    Args:
        data (Dict[str, Any]): Dictionary layer or payload to read, write, merge, or render.
        path (Optional[str]): Local path, config path, or dotted dictionary path used by the helper.

    Returns:
        bool: The result of ``dunset``.
    """
    if path is None:
        data.clear()
        return True
    cur: Any = data
    parts = _split_path(path)
    if not parts:
        return False
    for part in parts[:-1]:
        parsed = _parse_list_part(part)
        if parsed:
            key, op = parsed
            if key:
                if not isinstance(cur, Mapping) or key not in cur or not isinstance(cur[key], list):
                    return False
                values = cur[key]
            elif isinstance(cur, list):
                values = cur
            else:
                return False
            idx = int(op)
            if idx >= len(values) or idx < -len(values):
                return False
            cur = values[idx]
            continue
        if not isinstance(cur, Mapping) or part not in cur:
            return False
        cur = cur[part]
    last = parts[-1]
    parsed = _parse_list_part(last)
    if parsed:
        key, op = parsed
        if key:
            if not isinstance(cur, Mapping) or key not in cur or not isinstance(cur[key], list):
                return False
            values = cur[key]
        elif isinstance(cur, list):
            values = cur
        else:
            return False
        idx = int(op)
        if idx >= len(values) or idx < -len(values):
            return False
        del values[idx]
        return True
    if isinstance(cur, dict) and last in cur:
        del cur[last]
        return True
    return False


def dsetdef(data: Dict[str, Any], path: str, default: Any = None) -> bool:
    """
    Set a nested default only when the path is missing or currently None.

    Use this helper for nested config dictionaries and scoped runtime configuration layers.

    Args:
        data (Dict[str, Any]): Dictionary layer or payload to read, write, merge, or render.
        path (str): Local path, config path, or dotted dictionary path used by the helper.
        default (Any): Fallback value used when a path or config key is missing. Defaults to ``None``.

    Returns:
        bool: The result of ``dsetdef``.
    """
    if dget(data, path, default=None) is not None:
        return False
    return dset(data, path, default)


def dflat(data: Dict[str, Any], prefix: str = "", enum: bool = False) -> Iterator[Tuple[str, Any]]:
    """Yield dot-path/value pairs from a nested mapping/list structure.

    Use this helper for nested config dictionaries and scoped runtime configuration layers.

    Args:
        data (Dict[str, Any]): Dictionary layer or payload to read, write, merge, or render.
        prefix (str): Text prefix inserted before rendered lines. Defaults to ``""``.
        enum (bool): Whether flat index entries for containers should be included. Defaults to ``False``.

    Yields:
        Iterator[Tuple[str, Any]]: The result of ``dflat``.
    """
    for key, value in data.items():
        clean_key = str(key).replace(".", "\\.")
        full_key = f"{prefix}.{clean_key}" if prefix else clean_key
        if enum:
            yield full_key, value
        if isinstance(value, dict):
            yield from dflat(value, prefix=full_key, enum=enum)
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                list_key = f"{full_key}[{idx}]"
                if enum:
                    yield list_key, item
                if isinstance(item, dict):
                    yield from dflat(item, prefix=list_key, enum=enum)
                elif isinstance(item, list):
                    yield from dflat({list_key: item}, enum=enum)
                elif not enum:
                    yield list_key, item
        elif not enum:
            yield full_key, value


def dunflat(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a nested dictionary from dot-path/value pairs.

    Use this helper for nested config dictionaries and scoped runtime configuration layers.

    Args:
        data (Dict[str, Any]): Dictionary layer or payload to read, write, merge, or render.

    Returns:
        Dict[str, Any]: The a nested dictionary from dot-path/value pairs.
    """
    out: Dict[str, Any] = {}
    for path, value in data.items():
        dset(out, path, value)
    return out


def _tombstone() -> Dict[str, bool]:
    return {_REMOVE: True}


def _tombstone_unset(data: Dict[str, Any], path: str) -> None:
    dset(data, path, _tombstone())


def _clear_tombstone_ancestors(data: Dict[str, Any], path: str) -> None:
    """Replace tombstone-marked ancestors along ``path`` with empty mappings."""
    parts = _split_path(path)
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        child = node[part]
        if isinstance(child, Mapping) and child.get(_REMOVE) is True:
            node[part] = {}
        node = node[part]


def _merge_value(base: Any, incoming: Any) -> Any:
    if isinstance(incoming, Mapping) and incoming.get(_REMOVE) is True:
        return _tombstone()
    if isinstance(incoming, Mapping) and incoming.get(_OVERWRITE) is True:
        out = {}
        for key, value in incoming.items():
            if key == _OVERWRITE:
                continue
            merged = _merge_value(None, value)
            if not (isinstance(merged, Mapping) and merged.get(_REMOVE) is True):
                out[key] = merged
        return out
    if isinstance(incoming, Mapping):
        out = {k: copy_data(v) for k, v in base.items()} if isinstance(base, Mapping) else {}
        for key, value in incoming.items():
            merged = _merge_value(out.get(key), value)
            if isinstance(merged, Mapping) and merged.get(_REMOVE) is True:
                out.pop(key, None)
            else:
                out[key] = merged
        return out
    return copy_data(_unfreeze(incoming) if isinstance(incoming, MappingProxyType) else incoming)


def _merge_layers(layers: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for layer in layers:
        out = _merge_value(out, layer or {})
    return out


def dmerge(iterable: Iterable[Dict[str, Any]], start: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Deep-merge dictionaries while preserving nested structure.

    Use this helper for nested config dictionaries and scoped runtime configuration layers.

    Args:
        iterable (Iterable[Dict[str, Any]]): Iterable of dictionaries merged from left to right.
        start (Optional[Dict[str, Any]]): Initial dictionary merged before values from ``iterable``. Defaults to ``None``.

    Returns:
        Dict[str, Any]: The result of ``dmerge``.
    """
    layers = [] if start is None else [start]
    layers.extend(iterable)
    return _merge_layers(layers)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_unfreeze(v) for v in value]
    return copy_data(value)


def to_mutable_data(value: Any) -> Any:
    """
    Return a mutable copy of config-style data.

    Config snapshots intentionally freeze mappings and lists so callers cannot
    mutate cached runtime state. Use this helper when a caller needs to persist,
    edit, or return a plain JSON-like copy of a config value.

    Args:
        value (Any): Config value, mapping, list, tuple, or scalar to copy.

    Returns:
        Any: A mutable copy where mappings are ``dict`` objects and sequences are lists.
    """
    return _unfreeze(value)
