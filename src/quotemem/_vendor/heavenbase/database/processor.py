"""SQLAlchemy bind normalization for raw SQL execution."""

from __future__ import annotations

__all__ = [
    "SQLGlotProcessor",
    "SQLProcessor",
    "create_sql_processor",
]

from ..utils.typing import Any
from .dialect.spec import DatabaseDialectSpec
from .sql_text import transpile_sql


def _read_name(sql: str, idx: int) -> tuple[str, int]:
    if idx >= len(sql) or not (sql[idx].isalpha() or sql[idx] == "_"):
        return "", idx
    end = idx + 1
    while end < len(sql) and (sql[end].isalnum() or sql[end] == "_"):
        end += 1
    return sql[idx:end], end


def _copy_quoted(sql: str, idx: int, quote: str) -> tuple[str, int]:
    end = idx + 1
    while end < len(sql):
        if sql[end] == quote:
            if end + 1 < len(sql) and sql[end + 1] == quote:
                end += 2
                continue
            end += 1
            break
        end += 1
    return sql[idx:end], end


def _copy_until(sql: str, idx: int, marker: str) -> tuple[str, int]:
    end = sql.find(marker, idx + len(marker))
    return (sql[idx:], len(sql)) if end < 0 else (sql[idx : end + len(marker)], end + len(marker))


_QMARK_PREFIX_KEYWORDS = {
    "and",
    "as",
    "between",
    "by",
    "case",
    "else",
    "end",
    "from",
    "group",
    "having",
    "in",
    "join",
    "limit",
    "not",
    "offset",
    "on",
    "or",
    "order",
    "returning",
    "select",
    "set",
    "then",
    "using",
    "values",
    "when",
    "where",
}


def _prev_nonspace(sql: str, idx: int) -> int:
    pos = idx - 1
    while pos >= 0 and sql[pos].isspace():
        pos -= 1
    return pos


def _next_nonspace(sql: str, idx: int) -> int:
    pos = idx + 1
    while pos < len(sql) and sql[pos].isspace():
        pos += 1
    return pos


def _prev_word(sql: str, idx: int) -> str:
    end = _prev_nonspace(sql, idx) + 1
    start = end - 1
    while start >= 0 and (sql[start].isalnum() or sql[start] == "_"):
        start -= 1
    return sql[start + 1 : end].lower()


def _qmark_is_operator(sql: str, idx: int) -> bool:
    if idx + 1 < len(sql) and sql[idx + 1] in {"|", "&"}:
        return True
    prev_idx = _prev_nonspace(sql, idx)
    next_idx = _next_nonspace(sql, idx)
    if prev_idx < 0 or next_idx >= len(sql):
        return False
    next_ch = sql[next_idx]
    if not (next_ch in {"'", '"', "`", ":", "$", "["} or next_ch.isalpha() or next_ch == "_"):
        return False
    prev_ch = sql[prev_idx]
    if prev_ch in {
        ")",
        "]",
        "}",
        "'",
        '"',
        "`",
    }:
        return True
    return (prev_ch.isalnum() or prev_ch == "_") and _prev_word(sql, idx) not in _QMARK_PREFIX_KEYWORDS


def _rewrite_binds(sql: str, qmark: bool = True) -> tuple[str, list[str]]:
    out, names, idx, n_pos = [], [], 0, 0
    while idx < len(sql):
        ch = sql[idx]
        if ch in {"'", '"', "`"}:
            text, idx = _copy_quoted(sql, idx, ch)
            out.append(text)
            continue
        if ch == "-" and idx + 1 < len(sql) and sql[idx + 1] == "-":
            end = sql.find("\n", idx + 2)
            out.append(sql[idx:] if end < 0 else sql[idx:end])
            idx = len(sql) if end < 0 else end
            continue
        if ch == "/" and idx + 1 < len(sql) and sql[idx + 1] == "*":
            text, idx = _copy_until(sql, idx, "*/")
            out.append(text)
            continue
        if ch == "$" and idx + 1 < len(sql) and sql[idx + 1] == "$":
            text, idx = _copy_until(sql, idx, "$$")
            out.append(text)
            continue
        if ch == "$":
            name, end = _read_name(sql, idx + 1)
            if name and end < len(sql) and sql[end] == "$":
                marker = f"${name}$"
                text, idx = _copy_until(sql, idx, marker)
                out.append(text)
                continue
            if idx + 1 < len(sql) and sql[idx + 1].isdigit():
                end = idx + 2
                while end < len(sql) and sql[end].isdigit():
                    end += 1
                name = f"param_{max(0, int(sql[idx + 1:end]) - 1)}"
                names.append(name)
                out.append(f":{name}")
                idx = end
                continue
            if name:
                names.append(name)
                out.append(f":{name}")
                idx = end
                continue
        if ch == "?":
            if not qmark or _qmark_is_operator(sql, idx):
                out.append(ch)
                idx += 1
                continue
            name = f"param_{n_pos}"
            n_pos += 1
            names.append(name)
            out.append(f":{name}")
            idx += 1
            continue
        if ch == "%" and idx + 1 < len(sql) and sql[idx + 1] == "s":
            name = f"param_{n_pos}"
            n_pos += 1
            names.append(name)
            out.append(f":{name}")
            idx += 2
            continue
        if ch == "%" and idx + 1 < len(sql) and sql[idx + 1] == "(":
            end = sql.find(")s", idx + 2)
            if end > 0:
                name = sql[idx + 2 : end]
                names.append(name)
                out.append(f":{name}")
                idx = end + 2
                continue
        if ch == ":" and not (idx + 1 < len(sql) and sql[idx + 1] == ":") and not (idx > 0 and sql[idx - 1] == ":"):
            name, end = _read_name(sql, idx + 1)
            if name:
                names.append(name)
                out.append(sql[idx:end])
                idx = end
                continue
        out.append(ch)
        idx += 1
    return "".join(out), names


def _bind_order(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _positional_params(names: list[str], values: Any) -> dict[str, Any]:
    vals = list(values if isinstance(values, (list, tuple)) else [values])
    order = _bind_order(names)
    if len(vals) < len(order):
        raise ValueError(f"Expected at least {len(order)} SQL parameters, got {len(vals)}")
    return {name: vals[idx] for idx, name in enumerate(order)}


class SQLProcessor:
    """Normalize raw SQL for SQLAlchemy execution.

    Args:
        dialect (str): Target SQLAlchemy dialect name.
        dialect_spec (DatabaseDialectSpec | None): Optional bound dialect metadata
            used for SQLGlot transpilation and dialect-aware normalization.

    Warnings:
        Placeholder normalization is lexical and intentionally conservative: it skips quoted strings and comments, then
        rewrites common driver styles to SQLAlchemy named binds. PostgreSQL JSON ``?``/``?|``/``?&`` operators and
        ``::`` casts are preserved. Complex vendor-specific SQL should prefer explicit ``:name`` parameters or SQLAlchemy
        Core statements.
    """

    def __init__(self, dialect: str, *, dialect_spec: DatabaseDialectSpec | None = None) -> None:
        """Create a SQL processor for one dialect.

        Returns:
            None: This function does not return a value.
        """
        self.dialect = dialect
        self.dialect_spec = dialect_spec

    def _qmark_binds(self, params: Any = None) -> bool:
        if params is None or isinstance(params, dict):
            return False
        if isinstance(params, list) and params and isinstance(params[0], dict):
            return False
        return True

    def normalize_params(self, sql: str, params: Any = None) -> tuple[str, Any]:
        """Return SQL using ``:name`` binds and matching SQLAlchemy parameters.

        Args:
            sql: Raw SQL containing ``:name``, ``?``, ``%s``, ``%(name)s``, ``$name``, or ``$1`` placeholders.
            params: ``dict``, scalar/sequence positional params, ``List[Dict]`` batch params, or ``List[Tuple]`` batch params.

        Returns:
            tuple[str, Any]: Rewritten SQL and parameters suitable for ``Connection.execute(sa.text(sql), params)``.

        Warnings:
            Bare ``?`` tokens are rewritten only for positional parameter payloads. PostgreSQL JSON operators such as
            ``payload ? 'key'`` are left intact; prefer named binds when an expression mixes JSON operators and parameters.
        """
        out_sql, names = _rewrite_binds(str(sql), qmark=self._qmark_binds(params))
        if params is None:
            return out_sql, None
        if isinstance(params, dict):
            return out_sql, dict(params)
        if isinstance(params, list) and params and isinstance(params[0], dict):
            return out_sql, [dict(item) for item in params]
        if isinstance(params, list) and params and isinstance(params[0], (list, tuple)):
            return out_sql, [_positional_params(names, item) for item in params]
        return out_sql, _positional_params(names, params)

    def process(self, sql: str, params: Any = None, transpile_from: str | None = None) -> tuple[str, Any]:
        """Return processed SQL text and parameters.

        Args:
            sql: Raw SQL string.
            params: SQL execution parameters accepted by ``normalize_params``.
            transpile_from: Optional source SQLGlot dialect. When set, SQL is transpiled before the final bind rewrite.

        Returns:
            tuple[str, Any]: SQLAlchemy-ready SQL and params.
        """
        target = self.dialect_spec.sqlglot_name() if self.dialect_spec is not None else self.dialect
        sql = transpile_sql(sql, transpile_from, target) if transpile_from else sql
        return self.normalize_params(sql, params)


SQLGlotProcessor = SQLProcessor


def create_sql_processor(dialect: str = "sqlite") -> SQLProcessor:
    """Create a SQL processor for one dialect.

    Args:
        dialect (str): Target SQLAlchemy dialect name.

    Returns:
        SQLProcessor: Processor that rewrites raw SQL into SQLAlchemy-ready text and parameters.
    """
    return SQLProcessor(dialect)
