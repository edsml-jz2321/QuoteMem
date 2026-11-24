"""TinyDB local-document backend runtime."""

from __future__ import annotations

__all__ = ["TinyDBBackend"]

from heavenbase.utils import Any, Iterable, Mapping, Sequence, safe_error
from heavenbase.utils.files import delete_file, file_uri_to_path, get_file_dir, get_file_ext, get_file_name, path_to_file_uri, touch_dir
from heavenbase.utils.redact import redact_secrets
from heavenbase.utils.serialize import validate_json

from heavenbase.backends._scope import provider_scope
from heavenbase.backends.base import Backend, backend_locked
from heavenbase.execution.contracts import RowOp
from heavenbase.modules import ModuleResolver
from heavenbase.schema.model import EntitySchema


class TinyDBBackend(Backend):
    """TinyDB provider adapter backed by one workspace-specific JSON file."""

    identifier = "tinydb"
    aliases = ("tiny",)

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "TinyDBBackend":
        """Build a TinyDB backend from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend config mapping. ``database`` is the canonical local
                file URI; ``path`` is accepted as its compatibility alias.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.

        Returns:
            TinyDBBackend: Configured TinyDB backend instance.

        Raises:
            ValueError: If both ``database`` and ``path`` are provided.
        """
        cfg = dict(data or {})
        ws_id = cfg.pop("ws_id", None)
        database = cfg.pop("database", None)
        path = cfg.pop("path", None)
        if database is not None and path is not None:
            raise ValueError("TinyDB config accepts either 'database' or 'path', not both")
        return cls(name, database or path, ws_id=ws_id, resolver=resolver, config=config, **cfg)

    def __init__(
        self,
        name: str = "tinydb",
        database: str | None = None,
        *,
        ws_id: str | None = None,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a persistent TinyDB backend.

        Args:
            name: Workspace backend instance name.
            database: Strict local ``file://`` URI for the base JSON file.
                A bound workspace receives a sibling file suffixed with the
                collision-resistant workspace/backend owner token.
            ws_id: Optional workspace namespace.
            resolver: Context-owned module authority.
            config: Config authority retained by the backend.
            **kwargs: Extra keyword arguments passed to ``TinyDB``.

        Returns:
            None: This function does not return a value.
        """
        super().__init__(name, ws_id=ws_id, resolver=resolver, config=config)
        configured = database or self._construction_config_get(
            "heavenbase.backend_defaults.tinydb.database",
            default="file://%/tinydb.json",
        )
        self._base_path = file_uri_to_path(str(configured), home=self.config.root)
        self._client_kwargs = dict(kwargs)
        self.path = self._database_path()
        self.database = path_to_file_uri(self.path)
        self.client = None
        self.client_error = ""
        self._schemas: dict[str, EntitySchema] = {}
        self._tables: dict[str, str] = {}
        self._connect()

    def _database_path(self) -> str:
        """Return the current workspace-safe JSON file path."""
        if not self.ws_id:
            return self._base_path
        stem = get_file_name(self._base_path, ext=False)
        extension = get_file_ext(self._base_path)
        suffix = f".{extension}" if extension else ".json"
        owner = provider_scope(self.ws_id, self.name, self.identifier, max_length=96)
        return f"{stem}_{owner}{suffix}"

    def _close(self) -> None:
        """Close the current provider handle when one exists."""
        if self.client is None:
            return
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        self.client = None

    def _connect(self) -> None:
        """Open the optional TinyDB provider at the current physical path."""
        try:
            from tinydb import TinyDB

            self.path = self._database_path()
            self.database = path_to_file_uri(self.path)
            touch_dir(get_file_dir(self.path))
            self.client = TinyDB(self.path, **self._client_kwargs)
            self.client_error = ""
        except Exception as exc:
            self.client = None
            self.client_error = redact_secrets(safe_error(exc))

    def _refresh_workspace(self) -> None:
        """Close and reopen the database under the new workspace path."""
        try:
            self._close()
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            self.client = None
        self._schemas.clear()
        self._tables.clear()
        self.path = self._database_path()
        self.database = path_to_file_uri(self.path)
        self._connect()

    def _need_client(self) -> None:
        """Raise a contextual error when TinyDB cannot be used."""
        if self.client is None:
            detail = f": {self.client_error}" if self.client_error else ""
            raise RuntimeError(f"TinyDB backend {self.name!r} is unavailable{detail}")

    def _table_name(self, entity_id: str) -> str:
        """Return an entity table name scoped to this backend instance."""
        return self._tables.setdefault(entity_id, self._physical_name(self.name, entity_id))

    def _table(self, entity_id: str) -> Any:
        """Return the TinyDB table for one entity."""
        self._need_client()
        return self.client.table(self._table_name(entity_id))

    @backend_locked
    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Ensure a TinyDB table and record its logical schema.

        Args:
            schema: Entity schema to store.
            bindings: Storage bindings routed to this backend.

        Returns:
            None: Provider table creation is idempotent.
        """
        del bindings
        self._schemas[schema.entity_id] = schema
        if self.client is not None:
            self._table(schema.entity_id)

    @backend_locked
    def drop(self) -> None:
        """Close and delete the current workspace database file.

        Args:
            None.

        Returns:
            None: Deletion is idempotent and does not affect sibling workspaces.
        """
        try:
            self._close()
            delete_file(self.path)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"TinyDB drop failed for {self.database!r}: {self.client_error}") from exc
        self._schemas.clear()
        self._tables.clear()

    @backend_locked
    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply ordered replacement upserts and idempotent deletes.

        TinyDB's built-in ``upsert`` merges documents. HeavenBase row mutations
        require replacement, so each adjacent unique-id phase removes old
        documents before inserting complete replacements.

        Args:
            rows: Backend-ready row mutations.

        Returns:
            None: Caller-visible mutation order is preserved.
        """
        self._need_client()
        try:
            for row in rows:
                table = self._table(row.entity_id)
                object_id = str(row.object_id)
                if row.is_delete():
                    table.remove(lambda document, wanted=object_id: str(document.get("object_id")) == wanted)
                    continue
                document = row.to_row()
                validate_json(document)
                previous = [item for item in table.all() if str(item.get("object_id")) == object_id]
                table.remove(lambda item, wanted=object_id: str(item.get("object_id")) == wanted)
                try:
                    table.insert_multiple([document])
                except Exception:
                    table.remove(lambda item, wanted=object_id: str(item.get("object_id")) == wanted)
                    if previous:
                        table.insert_multiple(previous)
                    raise
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"TinyDB mutation failed: {self.client_error}") from exc

    @backend_locked
    def upsert_owned(self, rows: Sequence[RowOp], *, owner_field: str, owner: str) -> list[RowOp]:
        """Apply owner-checked Catalog mutations in this workspace's own TinyDB file."""
        if not self.ws_id or self.ws_id != owner or "storage" in self._client_kwargs:
            raise NotImplementedError("TinyDB ownership-guarded mutations require a workspace-scoped JSON file")
        return self._upsert_owned_under_lock(rows, owner_field=owner_field, owner=owner)

    @backend_locked
    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return every document for one entity table.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            list[dict[str, Any]]: Exhaustive copies of persisted rows.
        """
        try:
            return [dict(document) for document in self._table(entity_id).all()]
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"TinyDB scan failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return documents aligned to caller-supplied object ids.

        Args:
            entity_id (str): Logical entity identifier.
            object_ids (Iterable[Any]): Object ids to fetch in caller order.
            columns (Sequence[str] | None): Optional field names to return; ``None`` returns the complete fragment.

        Returns:
            list[dict[str, Any] | None]: Rows aligned with ``object_ids``. Missing
            documents are returned as ``None``.

        Raises:
            TypeError: If ``columns`` contains a non-string name.
        """
        selection = self._row_columns(columns)
        ordered_ids = [str(object_id) for object_id in object_ids]
        if not ordered_ids:
            return []
        wanted = set(ordered_ids)
        try:
            rows = {str(document.get("object_id")): dict(document) for document in self._table(entity_id).all() if str(document.get("object_id")) in wanted}
            return self._project_rows([rows.get(object_id) for object_id in ordered_ids], selection)
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"TinyDB get_many failed for {entity_id!r}: {self.client_error}") from exc

    @backend_locked
    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags aligned to caller-supplied object ids.

        Args:
            entity_id (str): Logical entity identifier.
            object_ids (Iterable[Any]): Object ids to test in caller order.

        Returns:
            list[bool]: Existence flags aligned with ``object_ids``.
        """
        return [row is not None for row in self.get_many(entity_id, object_ids)]

    @backend_locked
    def count(self, entity_id: str) -> int:
        """Return the exact number of TinyDB documents for one entity.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            int: Exact number of TinyDB documents for the entity.

        Raises:
            RuntimeError: If the TinyDB count query fails.
        """
        try:
            return len(self._table(entity_id))
        except Exception as exc:
            self.client_error = redact_secrets(safe_error(exc))
            raise RuntimeError(f"TinyDB count failed for {entity_id!r}: {self.client_error}") from exc

    def exact_count(self, entity_id: str) -> int:
        """Return the exact TinyDB document count.

        Args:
            entity_id (str): Logical entity identifier.

        Returns:
            int: Exact TinyDB document count suitable for integrity proofs.
        """
        return self.count(entity_id)

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return live availability derived from the optional TinyDB handle."""
        return {
            **super()._live_tag_claims(),
            "backend-availability": "ready" if self.client is not None else "unavailable",
        }

    def to_dict(self) -> dict[str, Any]:
        """Return TinyDB backend metadata and its canonical local URI.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend metadata and connection state.
        """
        return {
            "name": self.name,
            "type": self.type,
            "connected": self.client is not None,
        }
