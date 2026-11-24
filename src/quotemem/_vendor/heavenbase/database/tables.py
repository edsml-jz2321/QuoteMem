"""SQLAlchemy table DDL and comment helpers for the Database facade."""

from __future__ import annotations

import re

import sqlalchemy as sa

from ..utils.typing import Any, Callable, Dict, Iterable, List, Optional, Union


def _norm_col_names(cols: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(getattr(col, "name", col)).lower() for col in cols)


_BARE_IDENTIFIER = re.compile(r"[^\W\d][\w$]*")
_QUOTED_IDENTIFIER = re.compile(r'"((?:[^"]|"")*)"')


def _catalog_elements(text: str) -> list[str]:
    """Split a catalog's ``[a, 'b', ...]`` list text into its elements, unquoting the single-quoted ones."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    elements, position = [], 0
    while position < len(text):
        while position < len(text) and text[position] in " ,":
            position += 1
        if position >= len(text):
            break
        if text[position] == "'":
            # a quoted element; DuckDB escapes a quote or backslash inside it with a backslash
            position += 1
            chars = []
            while position < len(text) and text[position] != "'":
                if text[position] == "\\" and position + 1 < len(text):
                    position += 1
                chars.append(text[position])
                position += 1
            elements.append("".join(chars))
            position += 1
            continue
        start, depth, quoted = position, 0, False
        while position < len(text) and (quoted or depth or text[position] != ","):
            char = text[position]
            if char == '"':
                quoted = not quoted
            elif not quoted and char == "(":
                depth += 1
            elif not quoted and char == ")":
                depth -= 1
            position += 1
        elements.append(text[start:position].strip())
    return elements


def _definition_elements(definition: str) -> list[str] | None:
    """Return the indexed elements of a ``CREATE [UNIQUE] INDEX ... ON table(elements)`` statement.

    The elements are split at the commas of the statement's first parenthesized list, outside quotes and
    nested parentheses. ``None`` means the statement has no such list.
    """
    text, quote, depth, start, elements = definition.strip(), "", 0, None, []
    for position, char in enumerate(text):
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
            if depth == 1:
                start = position + 1
        elif char == ")" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                elements.append(text[start:position].strip())
                return [element for element in elements if element]
        elif char == "," and depth == 1 and start is not None:
            elements.append(text[start:position].strip())
            start = position + 1
    return None


def _catalog_column_name(element: Any) -> str | None:
    """Return the column an index element names, or ``None`` for an expression."""
    element = str(element).strip()
    quoted = _QUOTED_IDENTIFIER.fullmatch(element)
    if quoted is not None:
        return quoted.group(1).replace('""', '"')
    return element if _BARE_IDENTIFIER.fullmatch(element) else None


def _catalog_column_names(value: Any, definition: Any = None) -> list[str]:
    """Return the columns of a catalog index row's ``columns`` value (``[a, 'b']`` text or a sequence).

    Where the catalog gives no ``columns`` (DuckDB before 1.1 leaves ``expressions`` empty), they are
    read from ``definition``, the index's ``CREATE INDEX`` statement. An index over an expression names
    no column, so it returns an empty list.
    """
    if value is None:
        elements = _definition_elements(str(definition)) if definition else None
        if not elements:
            return []
    else:
        elements = _catalog_elements(value) if isinstance(value, str) else list(value)
    names = [_catalog_column_name(element) for element in elements]
    return [] if not names or None in names else [name for name in names if name is not None]


def _index_exists_error(exc: Exception) -> bool:
    """Return whether creating an index failed because an index of that name exists (MySQL error 1061 included)."""
    args = getattr(getattr(exc, "orig", None), "args", ())
    return "already exists" in str(exc).lower() or (bool(args) and args[0] == 1061)


class _DatabaseTableMixin:
    """Provide table creation, mutation, index, and comment helpers for ``Database``."""

    def _normalize_table(self, table: Any) -> Any:
        if isinstance(table, sa.Table):
            return table
        tbl = getattr(table, "__table__", None)
        if isinstance(tbl, sa.Table):
            return tbl
        raise TypeError(f"Expected SQLAlchemy Table or ORM class, got {type(table).__name__}")

    def create_table(self, table: Any, checkfirst: bool = True) -> None:
        """Create one SQLAlchemy table.

        Args:
            table: SQLAlchemy ``Table`` or ORM class exposing ``__table__``.
            checkfirst: Skip creation when the table already exists.

        Returns:
            None: This method creates the table and invalidates the schema-index cache.
        """
        self._normalize_table(table).create(self.engine, checkfirst=checkfirst)
        self.invalidate_schema_index()

    def create_tables(self, tables: Iterable[Any], checkfirst: bool = True) -> None:
        """Create multiple SQLAlchemy tables, preserving metadata dependency ordering.

        Args:
            tables: SQLAlchemy ``Table`` objects or ORM classes exposing ``__table__``.
            checkfirst: Skip tables that already exist.

        Returns:
            None: This method creates all requested tables and invalidates the schema-index cache.
        """
        tabs = [self._normalize_table(table) for table in tables]
        metas = list(dict.fromkeys(tab.metadata for tab in tabs))
        for meta in metas:
            meta.create_all(self.engine, tables=[tab for tab in tabs if tab.metadata is meta], checkfirst=checkfirst)
        self.invalidate_schema_index()

    def _reflected_indexes(self, table: str, schema: Optional[str] = None) -> list[dict[str, Any]]:
        """Return a table's named indexes as ``name``/``unique``/``column_names`` descriptors.

        A dialect whose SQLAlchemy inspector reports no indexes (DuckDB) reads
        them from the database catalog with its ``index_reflection_sql``.

        Args:
            table: Table name.
            schema: Optional schema name.

        Returns:
            list[dict[str, Any]]: Named index descriptors.

        Raises:
            Exception: Whatever the inspector or the catalog query raises.
        """
        prep = self.engine.dialect.identifier_preparer
        sql = self.dialect_spec.index_reflection_statement(table, schema, prep.quote, prep.quote_schema)
        if sql is None:
            indexes = [dict(item) for item in sa.inspect(self.engine).get_indexes(table, schema=schema) or [] if item.get("name")]
            if not indexes and self.engine.dialect.name == "mysql":
                # seekdb and OceanBase print their indexes in a ``SHOW CREATE TABLE`` form SQLAlchemy reads none from
                return self._mysql_statistics_indexes(table, schema)
            return indexes
        return [
            {"name": str(row["name"]), "unique": bool(row["unique"]), "column_names": _catalog_column_names(row["columns"], row.get("definition"))}
            for row in self.execute(sql, readonly=True).to_dicts()
            if row.get("name")
        ]

    def _mysql_statistics_indexes(self, table: str, schema: Optional[str]) -> list[dict[str, Any]]:
        """Return a MySQL-dialect table's secondary indexes as ``information_schema.STATISTICS`` lists them.

        An index part over an expression has no column name, so such an index
        names no column, as the catalog readers of other dialects report it.
        """
        stats = sa.table(
            "STATISTICS",
            sa.column("TABLE_SCHEMA"),
            sa.column("TABLE_NAME"),
            sa.column("INDEX_NAME"),
            sa.column("NON_UNIQUE"),
            sa.column("SEQ_IN_INDEX"),
            sa.column("COLUMN_NAME"),
            schema="information_schema",
        )
        stmt = (
            sa.select(stats.c.INDEX_NAME, stats.c.NON_UNIQUE, stats.c.COLUMN_NAME)
            .where(stats.c.TABLE_SCHEMA == (sa.literal(schema) if schema is not None else sa.func.database()), stats.c.TABLE_NAME == table)
            .order_by(stats.c.INDEX_NAME, stats.c.SEQ_IN_INDEX)
        )
        indexes: dict[str, dict[str, Any]] = {}
        for name, non_unique, column_name in self.orm_execute(stmt, readonly=True).to_lists():
            name = str(name or "")
            if not name or name.upper() == "PRIMARY":
                continue
            index = indexes.setdefault(name, {"name": name, "unique": str(non_unique).strip() in ("0", "False"), "column_names": [], "expression": False})
            if column_name is None:
                index["expression"] = True
            else:
                index["column_names"].append(str(column_name))
        return [
            {"name": index["name"], "unique": index["unique"], "column_names": [] if index["expression"] else index["column_names"]}
            for index in indexes.values()
        ]

    def _primary_key_columns(self, inspector: Any, table: str, schema: Optional[str]) -> list[str]:
        """Return a table's primary-key columns, read from the catalog where the inspector reports none.

        The DuckDB driver reports no primary key from DuckDB 0.9 on; its dialect
        reads it with ``PRAGMA table_info`` instead.
        """
        try:
            columns = list((inspector.get_pk_constraint(table, schema=schema) or {}).get("constrained_columns") or [])
        except Exception:
            columns = []
        if columns:
            return columns
        prep = self.engine.dialect.identifier_preparer
        sql = self.dialect_spec.primary_key_reflection_statement(table, schema, prep.quote, prep.quote_schema)
        if not sql:
            return []
        return [str(row["name"]) for row in self.execute(sql, readonly=True).to_dicts() if row.get("pk")]

    def index_column_sets(self, table: Any, *, ignoring: Iterable[str] = ()) -> dict[tuple[str, ...], bool]:
        """Return server-side indexed column sets and uniqueness.

        Args:
            table: SQLAlchemy ``Table`` object or ORM class.
            ignoring: Names of indexes to leave out, such as ones about to be dropped.

        Returns:
            dict[tuple[str, ...], bool]: Lower-cased column tuples mapped to whether
            any covering index enforces uniqueness.
        """
        tab = self._normalize_table(table)
        indexed: dict[tuple[str, ...], bool] = {}
        try:
            inspector = sa.inspect(self.engine)
        except Exception:
            return indexed
        try:
            pk = self._primary_key_columns(inspector, tab.name, tab.schema)
            if pk:
                indexed[_norm_col_names(pk)] = True
        except Exception:
            pass
        skipped = {str(name) for name in ignoring}
        try:
            for unique in inspector.get_unique_constraints(tab.name, schema=tab.schema) or []:
                # MySQL reports a unique index as a unique constraint of the same name
                if unique.get("column_names") and str(unique.get("name")) not in skipped:
                    indexed[_norm_col_names(unique["column_names"])] = True
        except Exception:
            pass
        try:
            for index in self._reflected_indexes(tab.name, tab.schema):
                if index.get("column_names") and str(index.get("name")) not in skipped:
                    columns = _norm_col_names(index["column_names"])
                    indexed[columns] = indexed.get(columns, False) or bool(index.get("unique"))
        except Exception:
            pass
        return indexed

    def table_indexes(self, table: Any) -> list[dict[str, Any]]:
        """Return reflected ordinary index descriptors for one table.

        Args:
            table: SQLAlchemy ``Table`` object or ORM class.

        Returns:
            list[dict[str, Any]]: Named reflected index descriptors.
        """
        tab = self._normalize_table(table)
        try:
            return self._reflected_indexes(tab.name, tab.schema)
        except Exception:
            return []

    def drop_index(self, table: Any, name: str, column_names: Iterable[str]) -> None:
        """Drop one reflected ordinary index when it exists.

        Args:
            table: SQLAlchemy ``Table`` object or ORM class owning the index.
            name: Reflected physical index name.
            column_names: Indexed columns used to bind the SQLAlchemy index.

        Returns:
            None: An index the table does not hold, as ``table_indexes`` reflects it, is ignored.
        """
        tab = self._normalize_table(table)
        columns = [str(column_name) for column_name in column_names]
        if not columns or str(name) not in {str(item["name"]) for item in self.table_indexes(tab)}:
            return
        # bound to a stand-in table, so an index on a column the cached table no longer declares drops too
        stand_in = sa.Table(tab.name, sa.MetaData(), *(sa.Column(column, sa.String()) for column in columns), schema=tab.schema)
        sa.Index(str(name), *stand_in.c).drop(self.engine, checkfirst=False)

    def create_index(self, index: Any, checkfirst: bool = True) -> None:
        """Create one SQLAlchemy index through the shared Database engine.

        Args:
            index: SQLAlchemy ``Index`` object.
            checkfirst: Skip creation when the index already exists.

        Returns:
            None: This method creates the index when needed.

        Note:
            ``checkfirst`` looks the index up by name through ``table_indexes``, which reads the
            catalog on a dialect whose inspector reports no indexes (DuckDB). Another creator can
            still build it between that lookup and the ``CREATE INDEX``; that is the desired end
            state for an idempotent ensure, so an already-exists error is treated as success; any
            other error propagates.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: For any creation failure other than the index already
                existing (and for every failure when ``checkfirst`` is ``False``).
        """
        if checkfirst and str(index.name) in {str(item["name"]) for item in self.table_indexes(index.table)}:
            return
        try:
            index.create(self.engine, checkfirst=False)
        except sa.exc.SQLAlchemyError as exc:
            if not checkfirst or not _index_exists_error(exc):
                raise

    def create_missing_indexes(self, table: Any) -> None:
        """Create locally declared indexes that do not already exist server-side.

        Args:
            table: SQLAlchemy ``Table`` object or ORM class whose ``table.indexes`` should be reconciled.

        Returns:
            None: This method creates any missing indexes declared on the table metadata.
        """
        tab = self._normalize_table(table)
        existing = self.index_column_sets(tab)
        for index in tab.indexes:
            columns = _norm_col_names(index.columns)
            satisfied = existing.get(columns, False) if index.unique else columns in existing
            if not satisfied:
                self.create_index(index, checkfirst=True)

    def clear_table(self, table: str, schema: Optional[str] = None) -> None:
        """Delete all rows from one table without dropping the table.

        Args:
            table: Table name.
            schema: Optional schema name.

        Returns:
            None: This method deletes rows from the selected table.

        Warnings:
            This is destructive and commits like any other mutating statement outside an explicit transaction.
        """
        self.orm_execute(sa.delete(self._table(table, schema=schema)), readonly=False)

    def add_table_col(self, table: Any, column: Any) -> None:
        """Add one column to an existing table.

        Args:
            table: SQLAlchemy ``Table`` object or ORM class.
            column: SQLAlchemy ``Column`` to add.

        Returns:
            None: This method mutates the table when the column is missing and invalidates the schema-index cache.
        """
        tab = self._normalize_table(table)
        if column.name in self.columns(tab.name, schema=tab.schema):
            return
        preparer = self.engine.dialect.identifier_preparer
        table_name = preparer.format_table(tab)
        column_def = str(sa.schema.CreateColumn(column).compile(dialect=self.engine.dialect))
        sql = self.dialect_spec.add_column_statement(table_name, column_def)
        self.execute(sql, readonly=False)
        self.invalidate_schema_index()

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename a table using dialect-appropriate DDL.

        Args:
            old_name: Existing table name.
            new_name: New table name.

        Returns:
            None: This method renames the table and invalidates the schema-index cache.
        """
        prep = self.engine.dialect.identifier_preparer
        sql, params = self.dialect_spec.rename_table_statement(old_name, new_name, prep.quote)
        self.execute(sql, params, readonly=False)
        self.invalidate_schema_index()

    def sort_table_cols(self, table: Any, order: Union[List[str], Callable[[str], Any]]) -> None:
        """Rebuild a table with columns in a requested order.

        Args:
            table: SQLAlchemy ``Table`` object or ORM class.
            order: Explicit column-name order, or key function used to sort current column names.

        Returns:
            None: This method rebuilds the table when its current column order differs from the requested order.

        Warnings:
            This copies the table and does not preserve external triggers, views, or complex table-level constraints.
        """
        tab = self._normalize_table(table)
        col_names = [col.name for col in tab.c]
        sorted_names = (
            sorted(col_names, key=order)
            if callable(order)
            else [name for name in order if name in col_names] + [name for name in col_names if name not in set(order)]
        )
        if sorted_names == col_names:
            return
        prep = self.engine.dialect.identifier_preparer
        tmp_name = f"_hb_reorder_{tab.name}"
        if tmp_name in self.tables(schema=tab.schema):
            self.drop_table(tmp_name, schema=tab.schema)
        tmp_meta = sa.MetaData()
        tmp_cols = [
            sa.Column(col.name, col.type, primary_key=col.primary_key, nullable=col.nullable, server_default=col.server_default)
            for col in (tab.c[name] for name in sorted_names)
        ]
        tmp_tab = sa.Table(tmp_name, tmp_meta, *tmp_cols, schema=tab.schema)
        tmp_tab.create(self.engine)
        try:
            cols = ", ".join(prep.quote(name) for name in sorted_names)
            src = prep.format_table(tab)
            dst = prep.format_table(tmp_tab)
            self.execute(f"INSERT INTO {dst} ({cols}) SELECT {cols} FROM {src}", readonly=False)
            tab.drop(self.engine)
            self.rename_table(tmp_name, tab.name)
        except Exception:
            tmp_tab.drop(self.engine, checkfirst=True)
            raise
        finally:
            self.invalidate_schema_index()

    def drop_table(self, table: str, schema: Optional[str] = None) -> None:
        """Drop one table by name.

        Args:
            table: Table name.
            schema: Optional schema name.

        Returns:
            None: This method drops the table and invalidates the schema-index cache.

        Warnings:
            This removes the table definition and data. Use ``clear_table()`` to keep the table.
        """
        self._table(table, schema=schema).drop(self.engine, checkfirst=True)
        self.invalidate_schema_index()

    def drop_view(self, view: str, schema: Optional[str] = None) -> None:
        """Drop one view by name when the dialect supports ``DROP VIEW``.

        Args:
            view: View name.
            schema: Optional schema name.

        Returns:
            None: This method drops the view and invalidates the schema-index cache.
        """
        prep = self.engine.dialect.identifier_preparer
        name = prep.quote_schema(schema) + "." + prep.quote(view) if schema else prep.quote(view)
        self.execute(f"DROP VIEW IF EXISTS {name}", readonly=False)
        self.invalidate_schema_index()

    def clear(self) -> None:
        """Delete all rows from all reflected tables.

        Args:
            None.

        Returns:
            None: This method deletes rows from every reflected table.

        Warnings:
            This is a destructive data operation across the current database. It does not drop table definitions.
        """
        for table in self.tables():
            self.clear_table(table)

    def table_comment(self, table: str, schema: Optional[str] = None) -> Optional[str]:
        """Return a table comment when supported by the dialect.

        Args:
            table: Table name.
            schema: Optional schema name.

        Returns:
            Optional[str]: Comment text, ``None`` when absent or unsupported.
        """
        try:
            info = sa.inspect(self.engine).get_table_comment(table, schema=schema)
            return info.get("text") if info else None
        except NotImplementedError:
            return None

    def col_comment(self, table: str, column: str, schema: Optional[str] = None) -> Optional[str]:
        """Return a column comment when supported by the dialect.

        Args:
            table: Table name.
            column: Column name.
            schema: Optional schema name.

        Returns:
            Optional[str]: Comment text, ``None`` when absent or unsupported.
        """
        return next((col.get("comment") for col in self.columns(table, schema=schema, full=True) if col.get("name") == column), None)

    def col_comments(self, table: str, schema: Optional[str] = None) -> Dict[str, Optional[str]]:
        """Return comments for all columns in a table.

        Args:
            table: Table name.
            schema: Optional schema name.

        Returns:
            Dict[str, Optional[str]]: Column names mapped to comment text or ``None``.
        """
        return {col["name"]: col.get("comment") for col in self.columns(table, schema=schema, full=True)}

    def set_table_comment(self, table: str, comment: str, schema: Optional[str] = None) -> None:
        """Set or replace a table comment using SQLAlchemy DDL.

        Args:
            table: Table name.
            comment: New comment text.
            schema: Optional schema name.

        Returns:
            None: This method emits SQLAlchemy table-comment DDL.

        Warnings:
            Unsupported dialects may raise from SQLAlchemy DDL compilation or execution.
        """
        tab = self._table(table, schema=schema)
        tab.comment = comment
        self.orm_execute(sa.schema.SetTableComment(tab), readonly=False)

    def set_col_comment(self, table: str, column: str, comment: str, schema: Optional[str] = None) -> None:
        """Set or replace a column comment using SQLAlchemy DDL.

        Args:
            table: Table name.
            column: Column name.
            comment: New comment text.
            schema: Optional schema name.

        Returns:
            None: This method emits SQLAlchemy column-comment DDL.
        """
        tab = self._table(table, schema=schema)
        tab.c[column].comment = comment
        self.orm_execute(sa.schema.SetColumnComment(tab.c[column]), readonly=False)

    def drop_table_comment(self, table: str, schema: Optional[str] = None) -> None:
        """Remove a table comment using SQLAlchemy DDL.

        Args:
            table: Table name.
            schema: Optional schema name.

        Returns:
            None: This method emits SQLAlchemy table-comment removal DDL.
        """
        self.orm_execute(sa.schema.DropTableComment(self._table(table, schema=schema)), readonly=False)

    def drop_col_comment(self, table: str, column: str, schema: Optional[str] = None) -> None:
        """Remove a column comment using SQLAlchemy DDL.

        Args:
            table: Table name.
            column: Column name.
            schema: Optional schema name.

        Returns:
            None: This method emits SQLAlchemy column-comment removal DDL.
        """
        tab = self._table(table, schema=schema)
        self.orm_execute(sa.schema.DropColumnComment(tab.c[column]), readonly=False)
