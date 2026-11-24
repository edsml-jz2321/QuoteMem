"""SQLite physical text contract, evidence, and exact expressions."""

from __future__ import annotations

import re

from heavenbase.utils import Any
from heavenbase.utils.hash import md5hash
from heavenbase.database.types import DatabaseTextType
from sqlalchemy import String

SQLITE_TEXT_OPS = ("eq", "ne", "lt", "lte", "gt", "gte", "in", "contains", "starts_with", "ends_with")
SQLITE_TEXT_KEY_OPS = ("group", "distinct", "order")


class SQLiteTextType(DatabaseTextType):
    """SQLite text type with explicit binary collation."""

    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        """Return a binary-collated SQLite string implementation.

        Args:
            dialect: SQLAlchemy dialect requesting the concrete type.

        Returns:
            Any: Dialect-specific binary-collated string type descriptor.
        """
        return dialect.type_descriptor(String(self.length, collation="BINARY"))


def column_type(dtype: Any) -> Any:
    """Return SQLite text storage with an explicit binary collation.

    Args:
        dtype: Logical type whose physical storage is being resolved.

    Returns:
        Any | None: Binary-collated SQLite text type, or ``None`` when the dtype
        is not text-backed.
    """
    from heavenbase.types.storage import stores_as_text

    if not stores_as_text(dtype):
        return None
    return SQLiteTextType(length=dtype.max_chars)


def column_constraints(name: str, dtype: Any) -> tuple[Any, ...]:
    """Return an enforced SQLite text-storage-class constraint.

    Args:
        name (str): Physical column name.
        dtype: Logical type whose storage class is being constrained.

    Returns:
        tuple[Any, ...]: SQLite check constraints enforcing text storage, or an
        empty tuple when the dtype is not text-backed.
    """
    from heavenbase.types.storage import stores_as_text

    if not stores_as_text(dtype):
        return ()
    from sqlalchemy import CheckConstraint

    quoted = f'"{str(name).replace(chr(34), chr(34) * 2)}"'
    return (CheckConstraint(f"typeof({quoted}) IN ('text', 'null')", name=_constraint_name(name)),)


def text_evidence(backend: Any, entity_id: str | None, field_schema: Any, strategy: str) -> tuple[str, ...]:
    """Return exact SQLite scalar operations proved by enforced table DDL.

    Args:
        backend: Live SQL backend instance, if available.
        entity_id (str | None): Logical entity identifier for the field.
        field_schema: Logical field schema being compiled.
        strategy (str): Storage strategy identifier.

    Returns:
        tuple[str, ...]: Filter operations with concrete SQLite parity proof.
    """
    return SQLITE_TEXT_OPS if _has_contract(backend, entity_id, field_schema, strategy) else ()


def text_filter(col: Any, op: str, value: Any) -> Any:
    """Compile an exact SQLite predicate through explicit `BINARY` collation.

    Args:
        col: SQLAlchemy column expression.
        op (str): Portable filter operation identifier.
        value (Any): Normalized filter operand.

    Returns:
        Any | None: SQLAlchemy boolean expression with exact SQLite text semantics,
        or ``None`` when the operation is unsupported.
    """
    from sqlalchemy import LargeBinary, and_, cast, collate, func, true

    key = collate(col, "BINARY")
    if op == "eq":
        return key == value
    if op == "ne":
        return key != value
    if op == "lt":
        return key < value
    if op == "lte":
        return key <= value
    if op == "gt":
        return key > value
    if op == "gte":
        return key >= value
    if op == "in":
        return key.in_(value)
    if op == "contains":
        return func.instr(col, value) > 0
    if op in ("starts_with", "ends_with"):
        encoded = str(value).encode("utf-8")
        if not encoded:
            return true()
        blob = cast(col, LargeBinary)
        offset = 1 if op == "starts_with" else -len(encoded)
        exact = func.substr(blob, offset, len(encoded)) == encoded
        return and_(key >= value, exact) if op == "starts_with" else exact
    return None


def text_key_evidence(backend: Any, entity_id: str | None, field_schema: Any, strategy: str) -> tuple[str, ...]:
    """Return SQLite aggregate key purposes proved by enforced table DDL.

    Args:
        backend: Live SQL backend instance, if available.
        entity_id (str | None): Logical entity identifier for the field.
        field_schema: Logical field schema being compiled.
        strategy (str): Storage strategy identifier.

    Returns:
        tuple[str, ...]: Aggregate key purposes with concrete SQLite parity proof.
    """
    return SQLITE_TEXT_KEY_OPS if _has_contract(backend, entity_id, field_schema, strategy) else ()


def text_key(col: Any, _purpose: str) -> Any:
    """Return a value-preserving SQLite binary comparison key.

    Args:
        col: SQLAlchemy column expression.
        _purpose (str): Aggregate key purpose (`group`, `distinct`, or `order`).

    Returns:
        Any: Binary-collated SQLAlchemy expression suitable for exact ordering.
    """
    from sqlalchemy import collate

    return collate(col, "BINARY")


def _constraint_name(name: str) -> str:
    return f"ck_hb_text_{md5hash(str(name))[:12]}"


def _has_contract(backend: Any, entity_id: str | None, field_schema: Any, strategy: str) -> bool:
    if backend is None or entity_id is None or strategy != "inline" or backend.db is None:
        return False
    table = backend.tables.get(entity_id)
    if table is None or field_schema.name not in table.c:
        return False
    pragma = backend.db.execute("PRAGMA ignore_check_constraints", readonly=True).first() or {}
    if int(pragma.get("ignore_check_constraints", 1)) != 0:
        return False

    from sqlalchemy import inspect

    constraints = inspect(backend.db.engine).get_check_constraints(table.name, schema=table.schema)
    expected_name = _constraint_name(field_schema.name)
    return any(str(item.get("name") or "") == expected_name and _text_check_column(item.get("sqltext")) == field_schema.name for item in constraints)


_TEXT_CHECK = re.compile(
    r'^\s*typeof\s*\(\s*(?P<column>"(?:""|[^"])*"|`(?:``|[^`])*`|\[(?:\]\]|[^\]])*\]|[A-Za-z_][A-Za-z0-9_$]*)\s*\)'
    r"\s+in\s*\(\s*'text'\s*,\s*'null'\s*\)\s*$",
    flags=re.IGNORECASE,
)


def _text_check_column(sqltext: Any) -> str | None:
    """Return the sole column constrained by one exact reflected text check."""
    match = _TEXT_CHECK.fullmatch(str(sqltext or ""))
    if match is None:
        return None
    column = match.group("column")
    if column.startswith('"'):
        return column[1:-1].replace('""', '"')
    if column.startswith("`"):
        return column[1:-1].replace("``", "`")
    if column.startswith("["):
        return column[1:-1].replace("]]", "]")
    return column
