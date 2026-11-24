"""User-facing SQLAlchemy database facade."""

from __future__ import annotations

__all__ = ["Database", "DatabaseError"]

import uuid
import weakref

import sqlalchemy as sa

from ..context.authority import resolve_runtime_authority
from ..modules import ModuleResolver
from ..utils.config import CM_HVNB
from ..utils.config.data import to_mutable_data
from ..utils.network import NetworkProxy
from ..utils.typing import Any, Dict, Optional
from .ddl import _drops_namespace, drop_database
from .execution import DatabaseError, _DatabaseExecutionMixin
from .processor import SQLProcessor
from .reflection import _DatabaseReflectionMixin
from .registry import DatabaseEngineRegistry
from .config import DBEngine, DBSpec
from .dialect.registry import BootstrapDialectCatalog, ResolverDialectCatalog
from .tables import _DatabaseTableMixin


class Database(_DatabaseExecutionMixin, _DatabaseReflectionMixin, _DatabaseTableMixin):
    """Config-driven SQLAlchemy database facade.

    ``Database`` is the low-level SQL utility used by HeavenBase SQL backends and is also suitable for direct scripts.
    It resolves provider config through ``DB_ENGINE``, shares SQLAlchemy engines through ``DatabaseEngineRegistry``, and
    returns reusable ``SQLResponse`` objects for raw SQL and SQLAlchemy Core statements.

    Args:
        database: Database name or canonical ``file://`` URI. ``":memory:"`` creates an isolated SQLite in-memory engine for this facade;
            the registry forgets that engine once the facade is garbage collected.
        provider: Provider id such as ``"sqlite"``, ``"duckdb"``, ``"postgres"``, or ``"mysql"``.
        preset: Provider preset selected from ``heavenbase.db.presets``. The
            default config selects ``"default"``, which currently uses SQLite.
        pool: Optional SQLAlchemy pool configuration merged onto provider defaults.
        autocreate: Whether first engine access should attempt database namespace creation.
        spec: Optional resolved database spec. This bypasses config resolution and
            is mutually exclusive with ``database``, ``provider``, ``pool``,
            ``config_engine``, and provider-specific keyword fields.
        config_engine: Optional explicit database config engine. Context injects
            a bootstrap-catalog engine while constructing its control Backend.
        resolver: Optional module authority used to capture physical dialect metadata.
        config: Optional config authority used to resolve database policy.
        **kwargs: Provider-specific connection fields, params, pragmas, and engine args.

    Warnings:
        ``readonly`` is explicit. Calls default to ``readonly=False`` because SQL auto-detection is not safe enough for a
        security boundary. Pass ``readonly=True`` to reject statements that are not confidently read-only, or
        ``readonly=None`` when you explicitly want conservative auto-detection for commit/rollback handling.
    """

    def __init__(
        self,
        database: Optional[str] = None,
        provider: Optional[str] = None,
        pool: Optional[Dict[str, Any]] = None,
        autocreate: bool = True,
        *,
        preset: Optional[str] = None,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
        config_engine: DBEngine | None = None,
        spec: DBSpec | None = None,
        **kwargs: Any,
    ) -> None:
        """Open a config-driven SQLAlchemy database facade.

        Returns:
            None: This function does not return a value.
        """
        if spec is not None:
            if not isinstance(spec, DBSpec):
                raise TypeError("Database spec must be a DBSpec")
            conflicting = [
                name
                for name, supplied in (
                    ("database", database is not None),
                    ("provider", provider is not None),
                    ("pool", pool is not None),
                    ("preset", preset is not None),
                    ("config_engine", config_engine is not None),
                    ("provider fields", bool(kwargs)),
                )
                if supplied
            ]
            if conflicting:
                raise ValueError(f"Database spec cannot be combined with: {', '.join(conflicting)}")
        raw = {
            key: value
            for key, value in {
                "preset": preset,
                "provider": provider,
                "database": database,
                **kwargs,
            }.items()
            if value is not None
        }
        if pool:
            raw["pool"] = pool
        if config_engine is not None and not isinstance(config_engine, DBEngine):
            raise TypeError("Database config_engine must be a DBEngine")
        if config_engine is not None and isinstance(config_engine.dialects, BootstrapDialectCatalog):
            if config_engine.manager is not None:
                raise ValueError("Bootstrap database dialect catalogs require manager=None")
            if resolver is not None:
                raise ValueError("Bootstrap Database resolver is finalized only after the module catalog opens")
            self._module_resolver = None
            self._config = CM_HVNB if config is None else config
        elif config_engine is not None:
            catalog_resolver = getattr(config_engine.dialects, "resolver", None)
            engine_manager = config_engine.manager
            authority = resolve_runtime_authority(
                resolver=resolver if resolver is not None else catalog_resolver,
                config=config if config is not None else engine_manager,
                label="Database config_engine",
            )
            if engine_manager is not None and engine_manager is not authority.config:
                raise ValueError("Database config_engine manager belongs to a different Context")
            if catalog_resolver is not None and catalog_resolver is not authority.resolver:
                raise ValueError("Database config_engine dialect catalog belongs to a different Context")
            self._module_resolver = authority.resolver
            self._config = authority.config
        else:
            authority = resolve_runtime_authority(resolver=resolver, config=config, label="Database")
            self._module_resolver = authority.resolver
            self._config = authority.config
        if spec is not None:
            self.spec = spec
        else:
            chosen_engine = config_engine
            if chosen_engine is None:
                chosen_engine = DBEngine(
                    manager=self._config,
                    dialects=ResolverDialectCatalog(self._module_resolver),
                )
            self.spec = chosen_engine.resolve(raw)
        owner = getattr(self._config, "_context", None)
        resolver_authority = self._module_resolver if getattr(self._module_resolver, "authority_token", None) is not None else None
        authority = resolver_authority or owner or self._config
        self._authority = authority
        self._authority_id = getattr(authority, "authority_token", None) or f"object-{id(authority):x}"
        self.dialect_spec = self.spec.dialect_spec
        # An engine that serves one thread at a time (an in-memory database, or a static, assertion or singleton pool)
        # is private to this facade. Its owner serializes only its own work, so a second facade sharing the engine
        # would race it: on a singleton pool the other facade's threads close connections still running a statement.
        # An address is reused once its facade is collected, so it cannot name a private engine: a later facade at
        # the same address would inherit the dead facade's engine, and an in-memory one its tables and rows.
        self._resource_id = f"private-{uuid.uuid4().hex}" if self.spec.serialized else None
        if self._resource_id is not None:
            weakref.finalize(self, DatabaseEngineRegistry.release_resource, self._resource_id)
        self.config = to_mutable_data(self.spec.data)
        self.dialect = self.spec.dialect
        self.autocreate = autocreate
        self.sql_processor = SQLProcessor(self.dialect, dialect_spec=self.dialect_spec)
        self.proxy = NetworkProxy(
            http_proxy=self.spec.args.get("http_proxy"), https_proxy=self.spec.args.get("https_proxy"), no_proxy=self.spec.args.get("no_proxy")
        )
        self._init_execution_state()
        DatabaseEngineRegistry.bind_authority(self._authority_id, authority)
        DatabaseEngineRegistry.clear_disposed(
            self.spec,
            authority_id=self._authority_id,
            resource_id=self._resource_id,
        )

    @property
    def engine(self):
        """Return the underlying SQLAlchemy Engine.

        Returns:
            Any: Shared SQLAlchemy engine for this resolved database spec.

        Warnings:
            Accessing this property may create the engine and run best-effort autocreate when ``autocreate=True``.
        """
        with self.proxy:
            return DatabaseEngineRegistry.get(
                self.spec,
                autocreate=self.autocreate,
                authority_id=self._authority_id,
                resource_id=self._resource_id,
            )

    @property
    def disposed(self) -> bool:
        """Return whether this facade points at a disposed engine key.

        Returns:
            bool: ``True`` when the shared engine key was disposed and this facade should no longer be used.
        """
        return DatabaseEngineRegistry.is_disposed(
            self.spec,
            authority_id=self._authority_id,
            resource_id=self._resource_id,
        )

    def available(self, check: bool = False) -> bool:
        """Return whether this database facade can be used.

        Args:
            check: When ``True``, open a connection and run ``SELECT 1``. When ``False``, only checks disposed state.

        Returns:
            bool: ``True`` when the facade is not disposed and, if requested, a live connection succeeds.
        """
        if self.disposed:
            return False
        if not check:
            return True
        response = self.execute("SELECT 1", readonly=True, safe=True)
        try:
            return response.ok
        finally:
            # Ends the probe's transaction and returns its connection at once.
            response.close()

    def clone(self) -> "Database":
        """Return an independent facade over the same captured DBSpec.

        Args:
            None.

        Returns:
            Database: New facade sharing this instance's resolved ``DBSpec`` and authorities.
        """
        return Database(
            autocreate=self.autocreate,
            resolver=self._module_resolver,
            config=self._config,
            spec=self.spec,
        )

    def _finalize_bootstrap_resolver(self, resolver: ModuleResolver) -> None:
        """Complete the direct-bootstrap Database facade's deferred resolver."""
        if resolver is None:
            raise TypeError("Bootstrap Database resolver finalization requires a concrete module authority")
        if self._module_resolver is resolver:
            return
        if self._module_resolver is not None:
            raise ValueError("The bootstrap Database is already bound to a different module resolver")
        bound_config = getattr(resolver, "authority_config", None)
        if bound_config is not None and bound_config is not self._config:
            raise ValueError("The bootstrap Database resolver belongs to a different Context")
        self._module_resolver = resolver
        self._authority = resolver
        self._authority_id = getattr(resolver, "authority_token", None) or f"object-{id(resolver):x}"
        DatabaseEngineRegistry.bind_authority(self._authority_id, resolver)

    def close(self) -> None:
        """No-op; shared engine lifecycle belongs to DatabaseEngineRegistry.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """

    def dispose(self) -> None:
        """Dispose the shared engine for this database spec.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        DatabaseEngineRegistry.dispose(
            self.spec,
            authority_id=self._authority_id,
            resource_id=self._resource_id,
        )

    def drop_database(self, *, force: bool = False) -> None:
        """Drop the physical database namespace after disposing its cached engines.

        Args:
            force: Allow dropping non-maintenance server databases and ask supported backends to clear active sessions.

        Warnings:
            This removes the configured database namespace. For Oracle the namespace is the target user/schema, while
            ``database`` remains the PDB/service name. Use ``drop()`` when you only want to delete reflected tables.

        Notes:
            The drop removes the database for the whole process: the engines that every Context cached on it are
            removed, and the hidden stores (LLM cache and Capsule registry stores) of every Context on it are
            detached, so each recreates its tables on fresh storage at its next use; one that is opening it meanwhile
            waits for the drop to end and then builds on fresh storage.
            The drop waits for the LLM cache hit bookkeeping already started on them. Drops of one database through
            ``drop_database()`` and ``drop()`` run one after another when they overlap, from any Context and through
            any name of a local file that this process opened or dropped while the file existed. Once the stores are
            detached, the drop closes this Context's scope on the database until it has removed it. Meanwhile this
            Context's facade, engine and schema-index calls on the database wait for the removal, also through an
            engine or index cached before the drop, unless the calling thread holds a HeavenBase Context or workspace
            lock, runs a drop itself, or has a connection to the database in use (such as an open transaction): the
            drop may wait for such a thread, so its call raises "Database is being dropped by this Context" at once.
            A workspace creates its facade under the Context lock, so a workspace opened on the database meanwhile
            raises it. Afterwards the facades this Context had on the database are retired and fail with "Engine has
            been disposed", while newer ones open the database as the drop left it. A thread that has a connection to
            a stored database in use cannot drop it: ``drop_database()`` and ``drop()`` raise ``DropRefusedError`` (a
            ``RuntimeError``) at once, because the drop's statement or file removal would wait for that connection
            (an in-memory database belongs to one facade, so no check is made for it). A
            connection is in use from its checkout until its checkin: an open transaction, or a response that is
            neither consumed nor closed; a response dropped unconsumed releases its connection at once. A response
            that a garbage collection frees together with an iterator over it keeps its connection until the
            ``heavenbase-sqlresponse-settler`` thread has read its remaining rows and ended its transaction. That
            connection does not count as in use by the thread that executed the response: that thread's calls may wait
            for a drop, and its own drop first settles such responses itself. PostgreSQL still refuses the drop while
            another session is connected to the database, such as a connection another Context has in use, unless
            ``force`` is true.

            A local file is removed together with its ``-wal`` and ``-shm`` files, and a DuckDB file also with its
            ``.wal`` file, inside the drop. DuckDB keeps a database open in the process while any connection to its
            file is open, so a DuckDB file is removed only once no connection of HeavenBase's engines to it, or to
            another name of it, is open or being opened, in any Context: the drop closes the idle connections, in
            whichever pool they sit, and each connection returned meanwhile; opening a connection to the file waits
            for the drop before DuckDB opens the file (or raises at once, as above) and then reaches the file as the
            drop left it. When connections stay in use for 30 seconds, such as one in another thread's long
            transaction, the drop raises ``DropRefusedError`` and removes nothing. A workspace drop checks every route
            for a connection of the dropping thread before it drops any, so that refusal drops no route. It then
            waits, without holding a lock and without closing a file, for each DuckDB file in turn until every
            connection that was in use when the wait for that file began has been returned, and raises
            ``DropRefusedError`` with no route dropped when one stays in use for 30 seconds; a connection through a
            hard link of the file is waited for only by the drop itself. It drops its routes in order and stops at a
            refusal, which a connection lent after the wait began can still cause: the routes after it are kept, and
            the error says which routes were dropped and which routes' drops failed before, or that no route was
            dropped when neither happened. Connections of engines that the caller created itself are not seen.

            Nothing can be removed for an in-memory database, for a facade without a lifecycle namespace (a server
            URL without a database, or Trino), or for a dialect without a registered database drop (such as
            ClickHouse). Such a call disposes only this facade's engine; other facades on the same database stay
            usable.

        Returns:
            None: This function does not return a value.
        """
        if not _drops_namespace(self.spec):
            self.dispose()
            return
        with DatabaseEngineRegistry._dropping_database(self.spec, authority_id=self._authority_id, resource_id=self._resource_id, db=self):
            drop_database(self.spec, force=force)

    def drop(self) -> None:
        """Drop all reflected tables in the current database.

        Args:
            None.

        Warnings:
            This drops tables only. Use ``drop_database()`` for explicit namespace/database deletion.

        Notes:
            Like ``drop_database()``, this detaches the hidden stores (LLM cache and Capsule registry stores) of every
            Context on the database, so each recreates its tables at its next use (one that is opening it meanwhile
            waits for the drop to end), waits for the LLM cache hit bookkeeping already started on them, and runs
            after any other drop of the database. A facade that a ``drop_database()`` of this Context retired
            meanwhile fails with "Engine has been disposed". A thread that has a connection to the database in use,
            such as an open transaction, cannot drop its tables: this raises ``DropRefusedError`` (a
            ``RuntimeError``) at once, because the ``DROP TABLE`` statements would wait for that connection.

        Returns:
            None: This function does not return a value.
        """
        with DatabaseEngineRegistry._drop_scope(self.spec, db=self):
            meta = sa.MetaData()
            meta.reflect(bind=self.engine)
            meta.drop_all(bind=self.engine)
            self.invalidate_schema_index()

    def _table(self, table: str, schema: Optional[str] = None) -> Any:
        return sa.Table(table, sa.MetaData(), schema=schema, autoload_with=self.engine)
