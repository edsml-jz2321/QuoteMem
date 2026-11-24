"""Database reflection and read-only table inspection helpers."""

from __future__ import annotations

import sqlalchemy as sa

from ..utils.rng import StableRNG
from ..utils.typing import Any, Dict, List, Optional, Sequence
from .registry import DatabaseEngineRegistry
from .response import SQLResponse


class _DatabaseReflectionMixin:
    """Provide reflection, table browsing, and lightweight statistics for ``Database``."""

    def current_database(self) -> Optional[str]:
        """Return the active database name when the dialect exposes it.

        Args:
            None.

        Returns:
            Optional[str]: Database name reported by the server, SQLite's ``main`` database name, or ``None`` when unavailable.
        """
        sql = self.dialect_spec.current_database_sql
        if sql:
            try:
                row = self.execute(sql, readonly=True).first()
                value = row.get("name") if row else None
                return str(value) if value is not None else None
            except Exception:
                pass
        return None

    def schema_index(self, refresh: bool = False) -> Dict[str, List[str]]:
        """Return cached ``{table: [columns...]}`` schema index.

        Args:
            refresh: Rebuild the index from the live database.

        Returns:
            Dict[str, List[str]]: Table and view names mapped to visible column names.
        """

        def build() -> Dict[str, List[str]]:
            names = sorted(set(self.tables()) | set(self.views()))
            return {name: [str(col) for col in self.columns(name)] for name in names}

        return DatabaseEngineRegistry.get_schema_index(
            self.spec,
            builder=build,
            refresh=refresh,
            authority_id=self._authority_id,
            resource_id=self._resource_id,
        )

    def set_schema_index(self, schema_index: Dict[str, Sequence[str]] | None) -> None:
        """Override cached schema-index data for this database.

        Args:
            schema_index: Mapping of table names to column-name sequences. ``None`` clears the override.

        Returns:
            None: This method updates the shared engine-registry schema-index cache.
        """
        DatabaseEngineRegistry.set_schema_index(
            self.spec,
            schema_index,
            authority_id=self._authority_id,
            resource_id=self._resource_id,
        )

    def invalidate_schema_index(self) -> None:
        """Invalidate cached schema-index data for this database.

        Args:
            None.

        Returns:
            None: This method clears cached schema-index data for the resolved database spec.
        """
        DatabaseEngineRegistry.invalidate_schema_index(
            self.spec,
            authority_id=self._authority_id,
            resource_id=self._resource_id,
        )

    def dbs(self) -> List[str]:
        """List databases visible to the current connection.

        Args:
            None.

        Returns:
            List[str]: Database names for supported dialects. Unsupported dialects return an empty list.
        """
        spec = self.dialect_spec
        if spec.list_databases_sql:
            return [str(row["name"]) for row in self.execute(spec.list_databases_sql, readonly=True).to_dicts() if row.get("name") is not None]
        return list(spec.static_database_names)

    def schemas(self) -> List[str]:
        """List schema names.

        Args:
            None.

        Returns:
            List[str]: Distinct schema names returned by SQLAlchemy inspection, in its order, or an empty list if
            inspection is unsupported. (duckdb-engine lists a DuckDB 0.10 schema once per attached database.)
        """
        try:
            return list(dict.fromkeys(sa.inspect(self.engine).get_schema_names()))
        except Exception:
            return []

    def tables(self, schema: Optional[str] = None) -> List[str]:
        """List table names.

        Args:
            schema: Optional schema name.

        Returns:
            List[str]: Table names visible in the requested schema.
        """
        return list(sa.inspect(self.engine).get_table_names(schema=schema))

    def views(self, schema: Optional[str] = None) -> List[str]:
        """List view names.

        Args:
            schema: Optional schema name.

        Returns:
            List[str]: View names visible in the requested schema.
        """
        return list(sa.inspect(self.engine).get_view_names(schema=schema))

    def columns(self, table: str, schema: Optional[str] = None, full: bool = False) -> List[Any]:
        """List columns for a table.

        Args:
            table: Table name.
            schema: Optional schema name.
            full: Return SQLAlchemy inspector dictionaries when ``True``; otherwise return names only.

        Returns:
            List[Any]: Column names or full inspector dictionaries.
        """
        try:
            cols = sa.inspect(self.engine).get_columns(table, schema=schema)
        except Exception:
            spec = self.dialect_spec
            prep = self.engine.dialect.identifier_preparer
            sql = spec.column_reflection_statement(table, schema, prep.quote, prep.quote_schema)
            if not sql:
                raise
            rows = self.execute(sql, readonly=True).to_dicts()
            cols = [
                {
                    "name": row.get("column_name") or row.get("name"),
                    "type": row.get("column_type") or row.get("type"),
                    "nullable": str(row.get("null") or "YES").upper() != "NO",
                    "default": row.get("default"),
                    "comment": None,
                }
                for row in rows
            ]
        return cols if full else [col["name"] for col in cols]

    def pks(self, table: str, schema: Optional[str] = None) -> List[str]:
        """List primary-key columns for a table.

        Args:
            table: Table name.
            schema: Optional schema name.

        Returns:
            List[str]: Primary-key column names. A dialect's ``primary_key_reflection_sql`` answers when the
            SQLAlchemy inspector fails or reports none (DuckDB's inspector reports none from DuckDB 0.9 on).
        """
        error: Optional[Exception] = None
        try:
            found = list((sa.inspect(self.engine).get_pk_constraint(table, schema=schema) or {}).get("constrained_columns", []) or [])
        except Exception as exc:
            found, error = [], exc
        if found:
            return found
        prep = self.engine.dialect.identifier_preparer
        sql = self.dialect_spec.primary_key_reflection_statement(table, schema, prep.quote, prep.quote_schema)
        if not sql:
            if error is not None:
                raise error
            return found
        rows = self.execute(sql, readonly=True).to_dicts()
        return [row["name"] for row in rows if row.get("pk")]

    def fks(self, table: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """List foreign keys for a table.

        Args:
            table: Table name.
            schema: Optional schema name.

        Returns:
            List[Dict[str, Any]]: SQLAlchemy inspector foreign-key dictionaries.
        """
        try:
            return list(sa.inspect(self.engine).get_foreign_keys(table, schema=schema))
        except Exception:
            if not self.dialect_spec.empty_foreign_keys_on_reflection_error:
                raise
            return []

    def n_rows(self, table: str, schema: Optional[str] = None) -> int:
        """Return row count for a table.

        Args:
            table: Table name.
            schema: Optional schema name.

        Returns:
            int: Exact row count.
        """
        tab = self._table(table, schema=schema)
        return int(self.orm_execute(sa.select(sa.func.count()).select_from(tab), readonly=True).scalar() or 0)

    def table_size(self, table: Optional[str] = None, schema: Optional[str] = None) -> int:
        """Return a portable table size approximation.

        Args:
            table: Optional table name. When omitted, sums all tables in ``schema``.
            schema: Optional schema name.

        Returns:
            int: Exact row count approximation of table size.
        """
        return self.n_rows(table, schema=schema) if table else sum(self.n_rows(name, schema=schema) for name in self.tables(schema=schema))

    def col_type(self, table: str, column: str, schema: Optional[str] = None) -> str:
        """Return a reflected column type string.

        Args:
            table: Table name.
            column: Column name.
            schema: Optional schema name.

        Returns:
            str: SQLAlchemy-rendered type string, or ``""`` when the column is not found.
        """
        return next((str(col["type"]) for col in self.columns(table, schema=schema, full=True) if col["name"] == column), "")

    def col_agg(self, table: str, column: str, agg: str = "count", distinct: bool = False, schema: Optional[str] = None) -> Any:
        """Return an aggregate value for one column.

        Args:
            table: Table name.
            column: Column name.
            agg: SQL aggregate function name, for example ``"count"``, ``"sum"``, ``"avg"``, ``"min"``, or ``"max"``.
            distinct: Aggregate distinct column values when ``True``.
            schema: Optional schema name.

        Returns:
            Any: Scalar aggregate value returned by the database.

        Warnings:
            ``agg`` is resolved through ``sqlalchemy.func``. Keep it to trusted function names, not untrusted user input.
        """
        tab = self._table(table, schema=schema)
        col = tab.c[column]
        fn = getattr(sa.func, agg.lower())
        expr = fn(sa.distinct(col)) if distinct else fn(col)
        return self.orm_execute(sa.select(expr.label("value")), readonly=True).scalar()

    def col_distincts(self, table: str, column: str, schema: Optional[str] = None) -> List[Any]:
        """Return distinct values for one column.

        Args:
            table: Table name.
            column: Column name.
            schema: Optional schema name.

        Returns:
            List[Any]: Distinct values in database result order.
        """
        tab = self._table(table, schema=schema)
        return self.orm_execute(sa.select(sa.distinct(tab.c[column]).label(column)), readonly=True).column(column)

    def col_freqs(self, table: str, column: str, topk: Optional[int] = None, asc: bool = False, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return grouped value frequencies for one column.

        Args:
            table: Table name.
            column: Column name.
            topk: Optional maximum number of groups to return.
            asc: Sort by ascending frequency when ``True``; descending frequency by default.
            schema: Optional schema name.

        Returns:
            List[Dict[str, Any]]: Rows shaped as ``{"value": value, "n": count}``.
        """
        tab = self._table(table, schema=schema)
        stmt = sa.select(tab.c[column].label("value"), sa.func.count().label("n")).group_by(tab.c[column]).order_by(sa.asc("n") if asc else sa.desc("n"))
        stmt = stmt.limit(topk) if topk is not None else stmt
        return self.orm_execute(stmt, readonly=True).to_dicts()

    def col_enums(self, table: str, column: str, schema: Optional[str] = None) -> List[Any]:
        """Return all values from one column, including duplicates.

        Args:
            table: Table name.
            column: Column name.
            schema: Optional schema name.

        Returns:
            List[Any]: Column values in database result order.
        """
        tab = self._table(table, schema=schema)
        return self.orm_execute(sa.select(tab.c[column].label(column)), readonly=True).column(column)

    def col_nonnulls(self, table: str, column: str, schema: Optional[str] = None) -> List[Any]:
        """Return all non-null values from one column.

        Args:
            table: Table name.
            column: Column name.
            schema: Optional schema name.

        Returns:
            List[Any]: Non-null values in database result order.
        """
        tab = self._table(table, schema=schema)
        return self.orm_execute(sa.select(tab.c[column].label(column)).where(tab.c[column].isnot(None)), readonly=True).column(column)

    def col_percentile(self, table: str, column: str, percentiles: Optional[List[int]] = None, schema: Optional[str] = None) -> Dict[str, Any]:
        """Return exact percentiles for a numeric column.

        Args:
            table: Table name.
            column: Numeric column name.
            percentiles: Percentiles between 0 and 100. Defaults to ``[0, 25, 50, 75, 100]``.
            schema: Optional schema name.

        Returns:
            Dict[str, Any]: Mapping like ``{"p50": 12.3}``. Empty when no non-null values exist.
        """
        ps = list(dict.fromkeys(percentiles or [0, 25, 50, 75, 100]))
        for p in ps:
            if p < 0 or p > 100:
                raise ValueError("percentiles must be between 0 and 100")
        vals = sorted(float(value) for value in self.col_nonnulls(table, column, schema=schema))
        if not vals:
            return {}
        out = {}
        last = len(vals) - 1
        for p in ps:
            pos = last * (p / 100.0)
            lo = int(pos)
            hi = min(lo + 1, last)
            frac = pos - lo
            out[f"p{p}"] = vals[lo] if lo == hi else vals[lo] * (1 - frac) + vals[hi] * frac
        return out

    def col_lengths(self, table: str, column: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Return min/max/average string lengths for one column.

        Args:
            table: Table name.
            column: Text-like column name.
            schema: Optional schema name.

        Returns:
            Dict[str, Any]: ``min_len``, ``max_len``, and ``avg_len`` values. Empty tables return ``None`` values.
        """
        tab = self._table(table, schema=schema)
        row = self.orm_execute(
            sa.select(
                sa.func.min(sa.func.length(tab.c[column])).label("min_len"),
                sa.func.max(sa.func.length(tab.c[column])).label("max_len"),
                sa.func.avg(sa.func.length(tab.c[column])).label("avg_len"),
            ),
            readonly=True,
        ).first()
        return row or {"min_len": None, "max_len": None, "avg_len": None}

    def browse(
        self, table: str, limit: Optional[int] = None, offset: int = 0, orderby: Optional[str] = None, schema: Optional[str] = None, **filters: Any
    ) -> SQLResponse:
        """Browse table rows with simple equality and ``IN`` filters.

        Args:
            table: Table name.
            limit: Optional maximum number of rows.
            offset: Number of rows to skip.
            orderby: Optional column name. Prefix with ``"-"`` for descending order.
            schema: Optional schema name.
            **filters: Equality filters. List values are lowered to ``IN`` predicates.

        Returns:
            SQLResponse: Read-only response for the selected rows.

        Example:
            ``db.browse("items", status=["new", "done"], orderby="-created_at", limit=20)``.
        """
        tab = self._table(table, schema=schema)
        stmt = sa.select(tab)
        for name, value in filters.items():
            stmt = stmt.where(tab.c[name].in_(value) if isinstance(value, list) else tab.c[name] == value)
        if orderby:
            desc = orderby.startswith("-")
            col = tab.c[orderby[1:] if desc else orderby]
            stmt = stmt.order_by(sa.desc(col) if desc else sa.asc(col))
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.orm_execute(stmt.offset(offset), readonly=True)

    def sample(self, table: str, n: int = 100, seed: int = 42, schema: Optional[str] = None) -> SQLResponse:
        """Return a deterministic table sample without assuming an id column.

        Args:
            table: Table name.
            n: Maximum sampled row count.
            seed: Stable sampling seed.
            schema: Optional schema name.

        Returns:
            SQLResponse: Cached response containing at most ``n`` sampled rows.

        Warnings:
            This provider-neutral implementation scans row dictionaries before sampling. Use backend-native SQL for very large
            tables when approximate or streaming sampling is required.
        """
        rows = self.browse(table, schema=schema).to_dicts()
        picked = rows if len(rows) <= n else StableRNG(seed=seed).hash_sample(rows, k=max(0, int(n)))
        return SQLResponse.from_rows(picked, columns=self.columns(table, schema=schema), query=f"sample({table})", params={"n": n, "seed": seed})

    def describe(self, table: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Return portable table metadata.

        Args:
            table: Table name.
            schema: Optional schema name.

        Returns:
            Dict[str, Any]: Table name, schema, reflected columns, primary keys, foreign keys, and row count.
        """
        return {
            "table": table,
            "schema": schema,
            "columns": self.columns(table, schema=schema, full=True),
            "pks": self.pks(table, schema=schema),
            "fks": self.fks(table, schema=schema),
            "n_rows": self.n_rows(table, schema=schema),
        }
