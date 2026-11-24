"""Conservative SQL read-only checks."""

from __future__ import annotations

__all__ = ["is_sql_readonly"]

import functools
import re
from collections.abc import Iterator
from typing import Any

import sqlalchemy
from sqlalchemy.dialects import registry
from sqlalchemy.engine.default import StrCompileDialect
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.sql import visitors
from sqlalchemy.sql.ddl import ExecutableDDLElement
from sqlalchemy.sql.dml import UpdateBase
from sqlalchemy.sql.elements import Cast, ClauseElement, CollationClause, ColumnClause, Extract, TextClause, TypeClause, quoted_name
from sqlalchemy.sql.operators import custom_op
from sqlalchemy.sql.selectable import HasHints, HasPrefixes, HasSuffixes, SelectBase, TextualSelect
from sqlalchemy.sql.type_api import TypeEngine

from .dialect.registry import database_dialects
from .dialect.spec import DatabaseDialectSpec
from .sql_text import strip_sql_comments

_MUTATING = {"insert", "update", "delete", "create", "drop", "alter", "truncate", "merge", "replace", "call", "exec", "execute"}
_MUTATING_CONSTRUCTS = (UpdateBase, ExecutableDDLElement)
# Modules whose classes may make up a construct that renders only from its structure.
_STANDARD_MODULES = {"sqlalchemy", "builtins", "typing", "typing_extensions", "abc", "collections", "_collections_abc"}
# The dialect ``str()`` renders a construct with.
_STR_DIALECT = StrCompileDialect()


def _raw_sql(element: object) -> Iterator[tuple[str, str, bool]]:
    """Yield ``(text, dialect, word)`` for each piece of SQL that ``element`` renders verbatim.

    That is a ``text()`` clause, a literal column, an ``extract()`` field, a
    collation, a statement prefix, suffix or hint, a custom operator, or a name
    (of a table, schema, column, label, alias, function package or sequence)
    declared ``quote=False``. ``dialect`` is the dialect a prefix, suffix or hint is
    scoped to, or ``"*"`` when the text renders on every dialect. ``word`` marks
    text rendered where SQL expects one word, such as a name, which a dialect
    may also render inside a string literal. A schema ``Table``'s ``prefixes``
    belong to its ``CREATE`` statement and are not read.
    """
    if isinstance(element, TextClause):
        yield element.text, "*", False
    elif isinstance(element, TextualSelect):
        yield element.element.text, "*", False
    elif isinstance(element, ColumnClause) and element.is_literal:
        yield element.name, "*", False
    elif isinstance(element, Extract):
        yield element.field, "*", True
    elif isinstance(element, CollationClause):
        yield element.collation, "*", True
    prefixes = getattr(element, "_prefixes", ()) if isinstance(element, HasPrefixes) else ()
    clauses = [*prefixes, *(getattr(element, "_suffixes", ()) if isinstance(element, HasSuffixes) else ())]
    yield from ((clause.text if isinstance(clause, TextClause) else str(clause), dialect, False) for clause, dialect in clauses)
    if isinstance(element, HasHints):
        yield from ((str(hint), dialect, False) for (_table, dialect), hint in dict(getattr(element, "_hints", None) or {}).items())
        yield from ((str(hint), dialect, False) for dialect, hint in getattr(element, "_statement_hints", ()))
    for operator in (getattr(element, "operator", None), getattr(element, "modifier", None)):
        if isinstance(operator, custom_op):
            yield operator.opstring, "*", False
    sequence = getattr(element, "sequence", None)
    names = [getattr(element, "name", None), getattr(element, "schema", None), *(getattr(element, "packagenames", None) or ())]
    names += [getattr(sequence, "name", None), getattr(sequence, "schema", None)]
    yield from ((str(name), "*", True) for name in names if isinstance(name, quoted_name) and name.quote is False)


@functools.lru_cache(maxsize=1024)
def _sqlalchemy_class(cls: type) -> bool:
    """Return whether every class ``cls`` derives from is SQLAlchemy's own (or Python's)."""
    return all(klass.__module__.partition(".")[0] in _STANDARD_MODULES for klass in cls.__mro__)


def _built_in(cls: type) -> bool:
    """Return whether SQLAlchemy alone decides how instances of ``cls`` render: no ``@compiles`` hook overrides it."""
    return _sqlalchemy_class(cls) and not hasattr(cls, "_compiler_dispatcher")


def _renders_as_built(element: object, dialect: Dialect) -> bool:
    """Return whether ``element`` renders for ``dialect`` only from what its structure shows.

    It does not when compiling expands it: an ORM statement expands entities,
    ``column_property()`` expressions and loader options such as
    ``with_expression()``. Nor does it when code outside SQLAlchemy renders it:
    an element class defined elsewhere or overridden with ``@compiles``, a
    ``CAST`` to such a type, or such a type whose dialect implementation wraps
    binds or columns in its own SQL expression.
    """
    if not _built_in(type(element)) or getattr(element, "_with_options", None):
        return False
    if (getattr(element, "_propagate_attrs", None) or {}).get("compile_state_plugin") == "orm":
        return False
    type_ = getattr(element, "type", None)
    if not isinstance(type_, TypeEngine) or _built_in(type(type_)):
        return True
    if isinstance(element, (Cast, TypeClause)):
        return False
    try:
        impl = type_.dialect_impl(dialect)
        return not (impl._has_bind_expression or impl._has_column_expression)
    except Exception:
        return False


def _raw_sql_may_write(text: str, word: bool = False) -> bool:
    """Return whether raw SQL text might hold a write, a second statement, or quoting that dialects read differently.

    Text rendered where SQL expects one word is safe only when it is a plain word.
    """
    if word:
        return _PLAIN_WORD.fullmatch(text) is None
    code = _sql_code(text)
    return code is None or ";" in code or _WRITE_WORDS.search(_WRITE_FUNCTIONS.sub(" (", code.lower())) is not None


def _construct_readonly(query: ClauseElement, dialect: Dialect | None) -> bool | None:
    """Classify a SQLAlchemy construct structurally; ``None`` defers to its text rendered for ``dialect``.

    ``dialect`` is the dialect the deferred text is rendered with, or ``None``
    when ``str()`` renders it. ``str()`` omits every prefix, suffix and hint
    scoped to a dialect and renders a word as no particular dialect does, so
    without a dialect a SELECT holding such text that might write is not a read.
    """
    if isinstance(query, _MUTATING_CONSTRUCTS):
        return False
    if not isinstance(query, SelectBase) or isinstance(query, TextualSelect):
        return None
    # A SELECT stays read-only unless a data-modifying CTE or subquery is nested in it.
    # SQL nested as text is opaque to the structure: when it might write, as a writable
    # CTE written with text() does, the rendered statement is classified. So is a SELECT
    # whose rendering the structure does not show, such as an ORM statement.
    name = dialect.name if dialect is not None else None
    deferred = dialect is not None and not _built_in(dialect.statement_compiler)
    for element in visitors.iterate(query):
        if isinstance(element, _MUTATING_CONSTRUCTS):
            return False
        deferred = deferred or not _renders_as_built(element, dialect or _STR_DIALECT)
        for text, scope, word in _raw_sql(element):
            if scope != "*" and scope != name:
                if name is None and _raw_sql_may_write(text, word):
                    return False  # it may run on a dialect whose rendering str() does not show
                continue  # it does not render on this dialect
            if _raw_sql_may_write(text, word):
                if word and name is None:
                    return False  # dialects render it differently, some inside a string literal
                deferred = True
    return None if deferred else True


@functools.lru_cache(maxsize=64)
def _named_dialect(dialect_cls: type[Dialect]) -> Dialect | None:
    """Return a ``dialect_cls`` that renders named bind parameters, as ``str()`` does, or ``None`` when it cannot be built offline."""
    try:
        return dialect_cls(paramstyle="named")
    except Exception:
        return None


def _render_dialect(dialect: str | None, dialect_spec: DatabaseDialectSpec | None, bound: Dialect | None = None) -> Dialect | None:
    """Return the SQLAlchemy dialect a construct's text is rendered with for classification, or ``None`` for ``str()``."""
    if bound is not None:
        return bound
    name = dialect_spec.identifier if dialect_spec is not None else dialect
    if not name:
        return None
    try:
        return _named_dialect(registry.load(name))
    except Exception:
        return None


def _construct_text(query: ClauseElement, dialect: Dialect | None = None) -> str:
    """Return statement text, rendered for ``dialect``, without compiling when the construct already carries it."""
    if isinstance(query, TextClause):
        return query.text
    if isinstance(query, TextualSelect):
        return query.element.text
    return str(query) if dialect is None else str(query.compile(dialect=dialect))


def _token(text: str) -> str:
    # The first statement can follow empty statements or a UTF-8 BOM.
    match = re.match(r"^[\s;\ufeff]*([a-z_]+)\b", strip_sql_comments(text).lower())
    return match.group(1) if match else ""


def _sqlglot_readonly(text: str, dialect: str | None, dialect_spec: DatabaseDialectSpec | None = None) -> bool | None:
    """Classify statements SQLGlot parses as reads; ``None`` defers to the lexical rule.

    The lexical rule decides text whose quoting dialects read differently, which
    SQLGlot can tokenize unlike the server (it drops a MySQL ``/*! ... */``
    comment that the server runs), and text holding any statement SQLGlot does
    not parse as a read, such as ``SHOW``, ``DESCRIBE``, ``EXPLAIN`` or ``PRAGMA``.
    """
    if _sql_code(text) is None:
        return None
    try:
        import sqlglot
    except Exception:
        return None
    try:
        read = dialect_spec.sqlglot_name() if dialect_spec is not None else database_dialects.to_sqlglot(dialect, default=None)
        # SQLGlot parses blank text, a stray ``;`` or a comment alone as ``None``, and
        # a comment after the last ``;`` as a ``Semicolon`` statement.
        stmts = [stmt for stmt in sqlglot.parse(text, read=read) if stmt is not None and type(stmt).__name__ != "Semicolon"]
    except Exception:
        return None
    if not stmts:
        return False
    if any(type(stmt).__name__ not in {"Select", "Union", "Except", "Intersect", "Values", "With"} for stmt in stmts):
        return None
    # ``Into`` is ``SELECT ... INTO``, which creates a table (or, in MySQL, writes variables or a file).
    mutating = {"Insert", "Update", "Delete", "Create", "Drop", "Alter", "Truncate", "Merge", "Into"}
    for stmt in stmts:
        for node in stmt.walk():
            if type(node).__name__ in mutating:
                return False
            # SQLGlot reads the second statement of a SQL Server batch such as
            # ``SELECT 1 DELETE FROM t`` as an alias, so an unquoted alias spelled
            # like a write word is not read-only.
            if _write_word_alias(node):
                return False
    return True


def _write_word_alias(node: Any) -> bool:
    """Return whether a SQLGlot node is an unquoted column or table alias spelled like a write word."""
    return (
        type(node).__name__ == "Identifier"
        and type(node.parent).__name__ in {"Alias", "TableAlias"}
        and not node.args.get("quoted")
        and node.name.lower() in _MUTATING | _BATCH_WORDS
    )


_READ_LEADS = {"select", "values", "show", "describe", "desc", "explain", "with"}
# Words that can make one statement led by a read keyword write: a data-modifying CTE,
# SELECT ... INTO a new table, or a second statement in a SQL Server batch, which needs
# no semicolon between statements (including WRITETEXT, SEND, ENABLE TRIGGER or SHUTDOWN).
_BATCH_WORDS = set(
    "into copy attach detach grant revoke deny set begin commit rollback save bulk dbcc backup restore"
    " writetext updatetext send receive enable disable kill shutdown reconfigure".split()
)
_WRITE_WORDS = re.compile(r"\b(" + "|".join(sorted(_MUTATING | _BATCH_WORDS)) + r")\b")
# EXPLAIN ANALYZE runs the statement it explains, which may write.
_EXPLAIN_RUNS = re.compile(r"\banaly[sz]e\b")
# String and numeric functions that share a name with a write verb.
_WRITE_FUNCTIONS = re.compile(r"\b(insert|replace|truncate)\s*\(")
# Text rendered where SQL expects one word that cannot change the statement around it.
_PLAIN_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
# ``PRAGMA [schema.]name`` or ``PRAGMA [schema.]name(argument)``; ``name = value`` sets a pragma.
_PRAGMA_WORD = r'(?:"[a-z_][a-z0-9_]*"|[a-z_][a-z0-9_]*)'
_PRAGMA = re.compile(rf'[\s\ufeff]*pragma(?:\s+|(?="))(?:(?:{_PRAGMA_WORD}|"")\s*\.\s*)?({_PRAGMA_WORD})\s*(\([^()]*\))?\s*$')
# SQLite and DuckDB pragmas that only read when queried without an argument. Others,
# such as SQLite ``optimize``, ``wal_checkpoint`` or ``incremental_vacuum`` and DuckDB
# ``force_checkpoint`` or ``enable_profiling``, write or change settings.
_PRAGMA_READS = set(
    "analysis_limit application_id auto_vacuum automatic_index busy_timeout cache_size cache_spill cell_size_check"
    " checkpoint_fullfsync collation_list collations compile_options count_changes data_version database_list"
    " database_size default_cache_size defer_foreign_keys empty_result_callbacks encoding foreign_key_check"
    " foreign_keys freelist_count full_column_names fullfsync function_list functions hard_heap_limit"
    " ignore_check_constraints integrity_check journal_mode journal_size_limit legacy_alter_table legacy_file_format"
    " locking_mode max_page_count metadata_info mmap_size module_list page_count page_size platform pragma_list"
    " query_only quick_check read_uncommitted recursive_triggers reverse_unordered_selects schema_version"
    " secure_delete short_column_names show_databases show_tables show_tables_expanded soft_heap_limit synchronous"
    " table_list temp_store threads trusted_schema user_agent user_version version wal_autocheckpoint writable_schema".split()
)
# Pragmas whose argument names what they read; for the others ``name(value)`` sets ``name``.
_PRAGMA_ARGUMENT_READS = set(
    "foreign_key_check foreign_key_list index_info index_list index_xinfo integrity_check metadata_info quick_check"
    " show storage_info table_info table_list table_xinfo".split()
)


def _sql_code(text: str, *, mark_quotes: bool = False, sqlite: bool = False) -> str | None:
    """Return SQL code with comments and quoted content masked, or ``None`` when unsure.

    With ``mark_quotes=True``, each quoted literal or identifier becomes ``""``.
    For SQLite PRAGMA classification, keep quoted plain words inside a common
    quote delimiter so the grammar can recognize them next to other tokens.

    Text that dialects could split differently returns ``None``: an unterminated
    quote or block comment, a ``--`` not followed by whitespace (MySQL reads it
    as two minus signs), a backslash inside a quote (an escape in MySQL, a plain
    character elsewhere), a nested or MySQL executable ``/*!`` block comment, a
    dollar quote, a quote inside brackets (a SQL Server or SQLite identifier), or
    a ``#`` outside quotes (a comment in MySQL, an operator elsewhere). With
    ``sqlite=True``, backslashes inside quotes and quotes inside brackets are
    literal characters.
    """
    quoted_token = '""' if mark_quotes else " "
    code: list[str] = []
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if text.startswith("--", index):
            if index + 2 < length and not text[index + 2].isspace():
                return None
            end = text.find("\n", index)
            index = length if end < 0 else end
            code.append(" ")
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0 or text.startswith("/*!", index) or "/*" in text[index + 2 : end]:
                return None
            index = end + 2
            code.append(" ")
        elif char in "'\"`":
            end = index + 1
            while (end := text.find(char, end)) >= 0 and text.startswith(char * 2, end):
                end += 2
            if end < 0 or (not sqlite and "\\" in text[index:end]):
                return None
            name = text[index + 1 : end]
            code.append(f'"{name}"' if sqlite and mark_quotes and _PLAIN_WORD.fullmatch(name) else quoted_token)
            index = end + 1
        elif char == "[":
            end = text.find("]", index)
            if end < 0 or (not sqlite and any(quote in text[index:end] for quote in "'\"`")):
                return None
            name = text[index + 1 : end]
            if mark_quotes:
                code.append(f'"{name}"' if sqlite and _PLAIN_WORD.fullmatch(name) else quoted_token)
            else:
                code.append(text[index : end + 1])
            index = end + 1
        elif char == "#" or (char == "$" and _DOLLAR_QUOTE.match(text, index)):
            return None
        else:
            code.append(char)
            index += 1
    return "".join(code)


def _is_sql_read(query: object, query_text: str | None = None, *, rendered: str | None = None, dialect: Dialect | None = None) -> bool:
    """Return whether a statement is certainly a read, classified without parsing SQL.

    A SQLAlchemy construct is classified structurally: a SELECT that nests no
    data-modifying CTE or subquery is a read; DML and DDL are not. Some
    constructs are classified as their text instead: a ``text().columns()``
    construct by its own text, and, rendered for ``dialect``, a SELECT that
    nests SQL rendered verbatim (``text()``, a literal column, an ``extract()``
    field, a collation, a prefix, suffix or hint, a custom operator, or a name
    declared ``quote=False``) which might write, an ORM statement, or a SELECT
    that holds an element class, or a ``CAST`` to a type, defined outside
    SQLAlchemy or overridden with ``@compiles``, or a type whose dialect
    implementation wraps binds or columns in its own SQL expression. Text is a read
    when it is one statement led by ``SELECT``, ``VALUES``, ``SHOW``, ``DESCRIBE``,
    ``EXPLAIN`` or ``WITH`` whose code outside comments and quoted literals or
    identifiers names no write verb (``INSERT``, ``UPDATE``, ``DELETE``, ``SET``
    and the like) and no ``INTO``, and is not ``EXPLAIN ANALYZE``; ``insert(``,
    ``replace(`` and ``truncate(`` count as functions. Anything else, such as ``COPY``,
    ``COMMENT ON``, a ``WITH ... INSERT``, text holding a second statement, or
    text whose quoting dialects read differently, is not reported as a read; a
    false negative only costs caching the statement's rows. A read that writes
    through a function it calls cannot be told apart.

    Args:
        query (object): SQLAlchemy statement or object whose string form should be inspected.
        query_text (str | None): Optional explicit SQL text, which takes precedence over ``query``.
        rendered (str | None): The text ``query`` already rendered for ``dialect``, such as the statement a
            driver ran, used instead of rendering it again when it is classified as text.
        dialect (Dialect | None): The SQLAlchemy dialect ``query`` runs on. ``None`` renders it with ``str()``,
            and a SELECT holding SQL that might write and whose rendering depends on the dialect (a prefix, suffix
            or hint scoped to a dialect, or a name, ``extract()`` field or collation that is not one plain word) is
            then not a read.

    Returns:
        bool: ``True`` when the statement is certainly a read.
    """
    if query_text is None and isinstance(query, ClauseElement):
        structural = _construct_readonly(query, dialect)
        if structural is not None:
            return structural
        query_text = rendered if rendered is not None else _construct_text(query, dialect)
    text = query_text if query_text is not None else str(query) if query is not None else ""
    code = _sql_code(text)
    if code is None:
        return False
    statements = [part for part in code.lower().split(";") if part.strip()]
    if len(statements) != 1:
        return False
    sql = _WRITE_FUNCTIONS.sub(" (", statements[0])
    match = re.match(r"^\s*([a-z_]+)\b", sql)
    if match is None or match.group(1) not in _READ_LEADS:
        return False
    if match.group(1) == "explain" and _EXPLAIN_RUNS.search(sql):
        return False
    return _WRITE_WORDS.search(sql) is None


def _fallback_readonly(text: str, *, sqlite: bool = False) -> bool:
    """Classify SQL text without a parser.

    Text that ``_is_sql_read`` reports as a read is read-only. Otherwise only one
    ``SHOW``, ``DESCRIBE``, ``DESC``, ``EXPLAIN`` or ``PRAGMA`` statement can be.
    An ``EXPLAIN`` (``DESCRIBE`` in MySQL) that names ``ANALYZE`` runs the
    statement it explains, so it is not read-only when it also names a write
    word. A ``PRAGMA`` is read-only only when it queries, without setting it, a
    pragma known to read (``_PRAGMA_READS``, or ``_PRAGMA_ARGUMENT_READS`` with
    an argument). For a SQLite PRAGMA, apply SQLite's quoting rules.
    """
    if _is_sql_read(text):
        return True
    code = _sql_code(text, mark_quotes=True, sqlite=sqlite and _token(text) == "pragma")
    statements = [part for part in code.lower().split(";") if part.strip()] if code is not None else []
    if len(statements) != 1:
        return False
    sql = statements[0]
    token = _token(sql)
    if token == "pragma":
        match = _PRAGMA.match(sql)
        return match is not None and match.group(1).strip('"') in (_PRAGMA_ARGUMENT_READS if match.group(2) else _PRAGMA_READS)
    if token in {"explain", "describe", "desc"}:
        return _EXPLAIN_RUNS.search(sql) is None or _WRITE_WORDS.search(_WRITE_FUNCTIONS.sub(" (", sql)) is None
    return token == "show"


def is_sql_readonly(
    query: object,
    query_text: str | None = None,
    dialect: str | None = None,
    *,
    dialect_spec: DatabaseDialectSpec | None = None,
) -> bool:
    """Return whether a SQL string or SQLAlchemy statement appears read-only.

    Args:
        query (object): SQLAlchemy statement or object whose string form should be
            inspected.
        query_text (str | None): Optional explicit SQL text. When omitted, SQLAlchemy
            SELECT, DML, and DDL constructs are classified structurally without
            compiling, ``text()`` and ``text().columns()`` constructs use their own
            text, and any other object uses ``str(query)``. A SELECT construct is
            classified by its text rendered for the dialect instead when it nests
            SQL rendered verbatim (``text()``, a literal column, an ``extract()``
            field, a collation, a prefix, suffix or hint, a custom operator, or a
            name declared ``quote=False``) which might write, such as a
            data-modifying CTE written with ``text()``; when it is an ORM
            statement, whose entities, ``column_property()`` expressions and loader
            options such as ``with_expression()`` render only when compiled; and
            when it holds an element class, or a ``CAST`` to a type, defined
            outside SQLAlchemy or overridden with ``@compiles``, or a type whose
            dialect implementation wraps binds or columns in its own SQL expression.
        dialect (str | None): Optional SQLAlchemy dialect name for SQLGlot parsing
            and for rendering a construct classified by its text. Without one, or
            when SQLAlchemy cannot load it, ``str(query)`` renders the construct,
            and a SELECT is not read-only when it holds SQL that might write and
            whose rendering depends on the dialect: a prefix, suffix or hint scoped
            to a dialect, or a name, ``extract()`` field or collation that is not
            one plain word. A construct classified by its text that cannot be
            rendered is not read-only either.
        dialect_spec (DatabaseDialectSpec | None): Optional bound dialect metadata
            overriding SQLGlot dialect selection and naming the rendering dialect.

    Returns:
        bool: ``True`` when the SQL appears read-only. Text whose quoting
        dialects read differently is never read-only: a ``--`` not followed by
        whitespace, a ``#`` outside quotes, a backslash inside quotes, a dollar
        quote, a nested or MySQL executable ``/*!`` block comment, an
        unterminated quote or block comment, or a quote inside brackets.
        Otherwise, with SQLGlot installed, text whose statements SQLGlot all
        parses as reads (``SELECT``, set operations, ``VALUES``, ``WITH``) is
        read-only unless a statement nests a write (DML, DDL or ``SELECT ...
        INTO``) or an unquoted column, table or CTE alias spelled like a write
        word listed below, which is how SQLGlot reads a SQL Server batch such as
        ``SELECT 1 DELETE FROM t``. So with SQLGlot, several reads separated by
        ``;``, ``SELECT ... FOR UPDATE`` and a column named like a write word
        stay read-only when SQLGlot parses them as reads. A conservative lexical
        rule decides the rest: text SQLGlot cannot parse or whose dialect it
        cannot map, text holding any statement SQLGlot does not parse as a read
        (such as ``SHOW``, ``DESCRIBE``, ``EXPLAIN`` or ``PRAGMA``), and all
        text when SQLGlot is not installed. The lexical rule reports at most one
        statement as read-only, so text holding a second statement after a
        ``;`` is not read-only. Under it, a ``SELECT``, ``VALUES`` or ``WITH``
        is not read-only when its code outside comments and quoted literals or
        identifiers names a write word, even as a bracketed identifier such as
        ``[update]`` or an unquoted column or CTE name: ``INSERT``, ``UPDATE``,
        ``DELETE``, ``CREATE``, ``DROP``, ``ALTER``, ``TRUNCATE``, ``MERGE``,
        ``REPLACE``, ``CALL``, ``EXEC`` or ``EXECUTE`` (except the functions
        ``insert(``, ``replace(`` and ``truncate(``), or ``INTO``, ``COPY``,
        ``ATTACH``, ``DETACH``, ``GRANT``, ``REVOKE``, ``DENY``, ``SET``,
        ``BEGIN``, ``COMMIT``, ``ROLLBACK``, ``SAVE``, ``BULK``, ``DBCC``,
        ``BACKUP``, ``RESTORE``, ``WRITETEXT``, ``UPDATETEXT``, ``SEND``,
        ``RECEIVE``, ``ENABLE``, ``DISABLE``, ``KILL``, ``SHUTDOWN`` or
        ``RECONFIGURE``, which can start a statement of a SQL Server batch.
        A ``SHOW`` statement is read-only. An ``EXPLAIN``, ``DESCRIBE`` or
        ``DESC`` statement is too, unless it names ``ANALYZE`` (which runs the
        statement it explains) and a write word. A ``PRAGMA`` is read-only only
        when it queries a SQLite or DuckDB pragma known to read without setting
        it, so ``PRAGMA name = value`` and pragmas such as ``optimize`` or
        ``wal_checkpoint`` are not. When the dialect is SQLite, the lexical rule
        uses SQLite quoting rules for PRAGMA names, schema qualifiers and arguments.

    Warnings:
        This is a conservative helper for explicit ``readonly=True`` guards. It is not used for default execution and
        should not be treated as a security sandbox for untrusted SQL.
    """
    return _sql_readonly(query, query_text, dialect, dialect_spec)


def _sql_readonly(
    query: object,
    query_text: str | None,
    dialect: str | None,
    dialect_spec: DatabaseDialectSpec | None,
    bound: Dialect | None = None,
    *,
    rendered: str | None = None,
) -> bool:
    """Classify a query using the connection's dialect and the executed SQL when available."""
    if query_text is None and isinstance(query, ClauseElement):
        render = _render_dialect(dialect, dialect_spec, bound)
        structural = _construct_readonly(query, render)
        if structural is not None:
            return structural
        if rendered is None:
            try:
                query_text = _construct_text(query, render)
            except Exception:
                return False
    text = rendered if rendered is not None else query_text
    text = (text if text is not None else str(query) if query is not None else "").strip()
    parsed = _sqlglot_readonly(text, dialect, dialect_spec)
    if parsed is not None:
        return parsed
    dialect_name = dialect_spec.identifier if dialect_spec is not None else dialect
    if bound is not None:
        dialect_name = bound.name
    return _fallback_readonly(text, sqlite=(dialect_name or "").partition("+")[0] == "sqlite")
