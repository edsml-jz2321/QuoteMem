"""Workspace API for the optional database extension."""

from __future__ import annotations

__all__ = ["DatabaseAPI"]

from collections.abc import Iterable

from heavenbase.utils import Any, Dict, List, Mapping, Optional


class DatabaseAPI:
    """Database extension operations bound to one workspace."""

    def __init__(self, workspace: Any) -> None:
        """Bind database extension operations to one workspace.

        Args:
            workspace (Any): Live workspace receiving database extension rows
                and tool calls.

        Returns:
            None: This initializer does not return a value.
        """
        self.workspace = workspace
        self._sources: Dict[str, Any] = {}

    def _bind_source(self, database_id: str, db: Any) -> None:
        """Bind a live source without persisting its credentials."""
        self._sources[database_id] = db

    def _source(self, database_id: str) -> Any | None:
        """Return a process-local live source when the database was ingested here."""
        return self._sources.get(database_id)

    def sources(self) -> Dict[str, Any]:
        """List credential-safe named SQL backends available for ingest.

        Args:
            None.

        Returns:
            Dict[str, Any]: Source cards with backend references, database and
            dialect labels, connection state, and existing ingested ids.
        """
        from .toolkit import list_sources

        return list_sources(self.workspace)

    def ingest(
        self,
        db: Any,
        *,
        name: Optional[str] = None,
        tables: str | Iterable[str] | None = None,
        columns: Mapping[str, str | Iterable[str]] | None = None,
        profile: bool = True,
    ) -> Dict[str, List[str]]:
        """Introspect one database into this workspace's ``db-*`` rows.

        Args:
            db: ``Database`` instance, workspace SQL backend name, DSN URL, or local SQLite path.
            name: Optional display label. Defaults to the current database name or source path.
            tables: Optional selected relations, qualified or unqualified.
            columns: Optional relation-to-column selection mapping.
            profile: Whether to collect column statistics and value context.

        Returns:
            Created or updated object ids grouped by entity kind.
        """
        from .ingest import ingest_database

        return ingest_database(self.workspace, db, name=name, tables=tables, columns=columns, profile=profile)

    def run_sql(
        self,
        database_id: str,
        sql: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute arbitrary read-only SQL against one ingested live source.

        Args:
            database_id: Target ``db-database`` object_id.
            sql: Read-only SQL statement.
            params: Optional bind parameters.
            limit: Maximum rows returned to the caller.

        Returns:
            Bounded query result frame dictionary.
        """
        from .toolkit import run_sql

        return run_sql(self.workspace, database_id, sql, params=params, limit=limit)

    def run_snippet(
        self,
        snippet_id: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute one registered ``db-snippet`` against its source database.

        Args:
            snippet_id: ``db-snippet`` object_id.
            params: Optional bind parameters merged over snippet defaults.
            limit: Maximum rows returned to the caller.

        Returns:
            Query result frame dictionary.
        """
        from .toolkit import run_snippet

        return run_snippet(self.workspace, snippet_id, params=params, limit=limit)

    def save_snippet(
        self,
        database_id: str,
        title: str,
        sql: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        card_text: str = "",
        tables: Optional[Iterable[str]] = None,
        synonyms: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Verify read-only SQL and persist it as a reusable ``db-snippet``.

        Args:
            database_id: Target ``db-database`` object_id.
            title: Stable human-readable snippet title.
            sql: Read-only SQL statement to verify.
            params: Optional default bind parameters.
            card_text: Searchable natural-language description.
            tables: Optional qualified relations referenced by the SQL.
            synonyms: Optional alternate user question phrases.
            limit: Maximum verification rows returned to the caller.

        Returns:
            Persisted snippet id plus the successful verification result.
        """
        from .toolkit import save_snippet

        return save_snippet(
            self.workspace,
            database_id,
            title,
            sql,
            params=params,
            card_text=card_text,
            tables=tables,
            synonyms=synonyms,
            limit=limit,
        )

    def suggest_sql(self, question: str, *, limit: Optional[int] = None) -> Dict[str, Any]:
        """Return matching snippets and predicates for one natural-language question.

        Args:
            question: Free-text question used for card-text search.
            limit: Maximum matches per entity kind.

        Returns:
            Matching snippet and predicate rows without executing SQL.
        """
        from .toolkit import suggest_sql

        return suggest_sql(self.workspace, question, limit=limit)
