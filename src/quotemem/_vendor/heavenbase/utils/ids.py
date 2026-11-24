"""Identifier validation helpers for public schema and backend names."""

from __future__ import annotations

__all__ = [
    "check_id",
    "check_entity_id",
    "check_field_name",
    "check_py_name",
    "check_workspace_id",
    "hash_id",
    "sql_index_name",
]

from .typing import Any, Optional
from .hash import md5hash
from .config import CM_HVNB
from .config.data import dget
from heavenbase._lazy import resolve_late

import re
from keyword import iskeyword

_ENTITY_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")


def check_id(value: str, label: str = "id") -> str:
    """
    Return a normalized lower-case id or raise for unsafe public ids.

    Use this helper before an identifier crosses a public API, file path, SQL table, or backend boundary.

    Args:
        value (str): Value to read, set, validate, hash, or format.
        label (str): Human-readable field name used in validation error messages. Defaults to ``'id'``.

    Returns:
        str: The normalized lower-case id or raise for unsafe public ids.
    """
    value = str(value or "").strip()
    if not _ENTITY_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid {label} {value!r}; use lower-case letters, digits, '-' or '_', starting with a letter")
    return value


def check_entity_id(entity_id: str) -> str:
    """
    Return a normalized entity identifier or raise for unsafe catalog/backend names.

    Use this helper before an identifier crosses a public API, file path, SQL table, or backend boundary.

    Args:
        entity_id (str): Public entity identifier before backend-safe normalization.

    Returns:
        str: The normalized entity identifier or raise for unsafe catalog/backend names.
    """
    return check_id(entity_id, "entity_id")


def check_workspace_id(ws_id: str) -> str:
    """
    Return a normalized workspace id or raise for unsafe backend namespaces.

    Use this helper before an identifier crosses a public API, file path, SQL table, or backend boundary.

    Args:
        ws_id (str): Workspace identifier before backend-safe normalization.

    Returns:
        str: The normalized workspace identifier or raise for unsafe backend namespaces.
    """
    return check_id(ws_id, "workspace_id")


def check_py_name(name: str, label: str = "name") -> str:
    """
    Return a public Python identifier or raise with context.

    Use this helper before an identifier crosses a public API, file path, SQL table, or backend boundary.

    Args:
        name (str): Name, label, scope, package, or logger identifier used by the helper.
        label (str): Human-readable field name used in validation error messages. Defaults to ``'name'``.

    Returns:
        str: A public Python identifier or raise with context.
    """
    value = str(name or "").strip()
    if not value.isidentifier() or iskeyword(value) or value.startswith("_"):
        raise ValueError(f"Invalid {label} {value!r}; use a public Python identifier")
    return value


def check_field_name(name: str) -> str:
    """
    Return a public Entity field name or raise with context.

    Use this helper before an identifier crosses a public API, file path, SQL table, or backend boundary.

    Args:
        name (str): Name, label, scope, package, or logger identifier used by the helper.

    Returns:
        str: A public Entity field name or raise with context.
    """
    return check_py_name(name, "field name")


def sql_index_name(table_name: str, *parts: str) -> str:
    """
    Return the physical SQL index name for a table plus column/key parts.

    Use this helper before declaring SQLAlchemy indexes so SQL-backed stores share the same deterministic naming rule.

    Args:
        table_name (str): Physical table name.
        *parts (str): Column names or stable index role names.

    Returns:
        str: The physical SQL index name.
    """
    return f"ix_{md5hash([table_name, *parts])}"


def hash_id(*values: Any, length: Optional[int] = None) -> str:
    """
    Return a zero-padded decimal md5 id for one value or a value tuple.

    Use this helper before an identifier crosses a public API, file path, SQL table, or backend boundary.

    An omitted ``length`` reads ``heavenbase.entity.id.hash_length`` from ``CM_HVNB``. In explicit Context mode the
    process has no default Context to own ``CM_HVNB``, so the packaged length is used and hashing never selects the
    process default. When the saved config cannot be opened for a lasting cause (an unwritable home or an invalid
    bootstrap file), the packaged length is used too, as in 0.1.2.2. Any other config error is raised, so a passing
    failure never changes the id length.

    Args:
        length (Optional[int]): Required string length or hash width, depending on the helper. Defaults to ``None``.
        *values (Any): Additional ``values`` values.

    Returns:
        str: A zero-padded decimal md5 id for one value or a value tuple.
    """
    if not values:
        raise ValueError("hash_id needs at least one value")
    n_digits = int(length if length is not None else _process_hash_length())
    if n_digits < 1:
        raise ValueError("hash_id length must be positive")
    value = values[0] if len(values) == 1 else tuple(values)
    return md5hash(value, length=n_digits)


_HASH_LENGTH_KEY = "heavenbase.entity.id.hash_length"


def _process_hash_length() -> int:
    """Return the process id hash length without selecting a forbidden default Context."""
    try:
        explicit = bool(resolve_late("heavenbase.context.explicit_mode"))
    except KeyError:
        # The explicit-mode owner binds this hook when it is imported; unbound means the mode was never selected.
        explicit = False
    if explicit:
        return int(dget(CM_HVNB.default, _HASH_LENGTH_KEY, 42))
    return int(CM_HVNB._get_or_packaged(_HASH_LENGTH_KEY, default=42))
