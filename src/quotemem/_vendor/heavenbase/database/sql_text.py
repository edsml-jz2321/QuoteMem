"""Standalone SQL text helpers with optional SQLGlot support."""

from __future__ import annotations

__all__ = [
    "SA_TO_SQLGLOT",
    "SQLGLOT_TO_SA",
    "compare_sqls",
    "escape_sql_binds",
    "prettify_sql",
    "sa_dialect_to_sqlglot",
    "split_sqls",
    "sqlglot_dialect_to_sa",
    "strip_sql_comments",
    "transpile_sql",
    "validate_sql",
]

import re
from collections.abc import Callable, Iterator, Mapping

from ..utils.typing import Any
from .dialect.registry import database_dialects


class _LazyDialectMap(Mapping[str, str]):
    """Expose complete dialect discovery without doing it during module import."""

    def __init__(self, factory: Callable[[], dict[str, str]]) -> None:
        self._factory = factory

    def __getitem__(self, key: str) -> str:
        return self._factory()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._factory())

    def __len__(self) -> int:
        return len(self._factory())

    def __repr__(self) -> str:
        return repr(self._factory())


SA_TO_SQLGLOT: Mapping[str, str] = _LazyDialectMap(database_dialects.sqlalchemy_map)
SQLGLOT_TO_SA: Mapping[str, str] = _LazyDialectMap(database_dialects.sqlglot_map)


def sa_dialect_to_sqlglot(dialect: str | None) -> str:
    """Return the SQLGlot dialect name for a SQLAlchemy dialect.

    Args:
        dialect (str | None): SQLAlchemy dialect identifier.

    Returns:
        str: SQLGlot dialect identifier resolved through `DatabaseDialectSpec` metadata.
    """
    return database_dialects.to_sqlglot(dialect)


def sqlglot_dialect_to_sa(dialect: str | None) -> str:
    """Return the SQLAlchemy dialect name for a SQLGlot dialect.

    Args:
        dialect (str | None): SQLGlot dialect identifier.

    Returns:
        str: SQLAlchemy dialect identifier resolved through `DatabaseDialectSpec` metadata.
    """
    return database_dialects.from_sqlglot(dialect)


def strip_sql_comments(sql: str) -> str:
    """Remove simple SQL line and block comments.

    Args:
        sql (str): SQL text.

    Returns:
        str: SQL text with simple `--` line comments and `/* ... */` block comments removed.
    """
    text = re.sub(r"/\*.*?\*/", " ", str(sql), flags=re.S)
    text = re.sub(r"--[^\n\r]*", " ", text)
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


def split_sqls(sql: str, dialect: str = "sqlite") -> list[str]:
    """Split a SQL script into statements.

    Args:
        sql (str): SQL script text.
        dialect (str): SQLAlchemy dialect used when SQLGlot parsing is available.

    Returns:
        list[str]: SQL statements, empty statements removed.
    """
    try:
        import sqlglot

        return [stmt.sql(dialect=sa_dialect_to_sqlglot(dialect)) for stmt in sqlglot.parse(sql, read=sa_dialect_to_sqlglot(dialect))]
    except Exception:
        return [part.strip() for part in str(sql).split(";") if part.strip()]


def transpile_sql(sql: str, src: str | None, tgt: str, **kwargs: Any) -> str:
    """Transpile SQL with SQLGlot when needed.

    Args:
        sql (str): SQL text.
        src (str | None): Source SQLGlot dialect. When empty or equal to `tgt`, the SQL is returned unchanged.
        tgt (str): Target SQLGlot dialect.
        **kwargs (Any): Additional options forwarded to `sqlglot.transpile`.

    Returns:
        str: Transpiled SQL text, or the original SQL when transpilation produces no output.

    Raises:
        ImportError: If SQLGlot is not installed and transpilation is required.
    """
    if not src or src == tgt:
        return sql
    try:
        import sqlglot
    except ImportError as exc:
        raise ImportError("SQL transpilation requires sqlglot") from exc
    out = sqlglot.transpile(sql, read=src, write=tgt, **kwargs)
    return out[0] if out else sql


def prettify_sql(sql: str, dialect: str = "sqlite") -> str:
    """Return formatted SQL when SQLGlot is available.

    Args:
        sql (str): SQL text.
        dialect (str): SQLAlchemy dialect used for parsing and formatting.

    Returns:
        str: Pretty SQL text, or a conservative semicolon-split cleanup when SQLGlot fails.
    """
    try:
        import sqlglot

        return ";\n".join(stmt.sql(dialect=sa_dialect_to_sqlglot(dialect), pretty=True) for stmt in sqlglot.parse(sql, read=sa_dialect_to_sqlglot(dialect)))
    except Exception:
        return "\n".join(part.strip() for part in strip_sql_comments(sql).split(";") if part.strip())


def compare_sqls(left: str, right: str, dialect: str = "sqlite") -> bool:
    """Return whether two SQL strings normalize to the same SQLGlot form.

    Args:
        left (str): First SQL text.
        right (str): Second SQL text.
        dialect (str): SQLAlchemy dialect used for parsing.

    Returns:
        bool: `True` when SQLGlot or fallback whitespace/comment normalization treats both strings as equivalent.
    """
    try:
        import sqlglot

        read = sa_dialect_to_sqlglot(dialect)
        return [stmt.sql(dialect=read) for stmt in sqlglot.parse(left, read=read)] == [stmt.sql(dialect=read) for stmt in sqlglot.parse(right, read=read)]
    except Exception:

        def norm(value: str) -> str:
            return re.sub(r"\s+", " ", strip_sql_comments(value)).strip().rstrip(";").lower()

        return norm(left) == norm(right)


def validate_sql(sql: str, dialect: str = "sqlite") -> bool:
    """Return whether SQL parses or at least contains one non-empty statement.

    Args:
        sql (str): SQL text.
        dialect (str): SQLAlchemy dialect used for parsing.

    Returns:
        bool: `True` when SQLGlot parses the text or fallback splitting finds one statement.
    """
    try:
        import sqlglot

        return bool(sqlglot.parse(sql, read=sa_dialect_to_sqlglot(dialect)))
    except Exception:
        return bool(split_sqls(strip_sql_comments(sql), dialect=dialect))


def escape_sql_binds(sql: str) -> str:
    """Return SQL with SQLAlchemy bind markers escaped for literal display.

    Args:
        sql (str): SQL text.

    Returns:
        str: SQL with `:name` bind markers escaped while leaving `::` casts intact.
    """
    return re.sub(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)", r"\\:\1", str(sql))
