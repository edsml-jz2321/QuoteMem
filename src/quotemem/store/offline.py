"""Existing-only HB Database reads without workspace registration or recovery.

The ordinary HB workspace constructor publishes MetaSchema even on reopen. This
owner therefore uses HB's Database facade and existing physical schemas directly;
it never constructs a Context, workspace, backend writer, or mutable Registry.
The caller retains the one exclusive StorageOwner through disposal and deletion.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import yaml
from sqlalchemy import CheckConstraint, Column, MetaData, Table, event, func, inspect, select, true

from quotemem.core.controls import ControlRowV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json, control_id
from quotemem.core.platform_files import open_private_file, validate_private_file
from quotemem.store.artifact_read import read_logical_artifact
from quotemem.store.entities import (
    PhysicalSchemaSpec,
    build_entity_classes,
    physical_schema_description,
)
from quotemem.store.hb_runtime import StorageOwner, force_privacy, packaged_bootstrap
from quotemem.store.layout import StorageLayout, layout_for_binding
from quotemem.store.metadata import StrictModel
from quotemem.store.normalization import NORMALIZED_SCHEMA_SPECS


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_uid,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class _ClosedFiles:
    """Pin complete main files; immutable reads never ignore a surviving WAL.

    SQLite mode=ro alone creates WAL/SHM for a normal closed WAL database. The
    immutable flag is safe here only because the exclusive outer owner is held,
    every WAL/SHM/journal is absent, and the complete file identities, sizes and
    modification times remain unchanged. Interrupted workspaces are unsupported.
    """

    def __init__(self, root: Path, stack: ExitStack) -> None:
        self.root = root
        self.files: dict[str, tuple[int, tuple[int, ...]]] = {}
        self.named_identities: dict[str, tuple[int, ...]] = {}
        self._companions()
        for name in ("memory.db", "hb-system.db"):
            try:
                descriptor = open_private_file(root / name, os.O_RDONLY)
            except QuoteMemError as error:
                raise QuoteMemError(
                    "workspace_binding_mismatch", "Offline database file is unsafe",
                ) from error
            stack.callback(os.close, descriptor)
            value = os.fstat(descriptor)
            if value.st_size < 100:
                raise QuoteMemError("workspace_binding_mismatch", "Offline database file is unsafe")
            self.files[name] = (descriptor, _file_identity(value))
            # CPython/Windows fstat reports change time while lstat reports
            # creation time in st_ctime. Preserve and recheck both observations.
            self.named_identities[name] = (
                _file_identity((root / name).lstat()) if os.name == "nt"
                else _file_identity(value)
            )
        self.check()

    def _companions(self) -> None:
        for name in ("memory.db", "hb-system.db"):
            for suffix in ("-wal", "-shm", "-journal"):
                path = self.root / (name + suffix)
                if path.exists() or path.is_symlink():
                    raise QuoteMemError(
                        "service_unhealthy", "Offline workspace requires SQLite recovery",
                    )

    def check(self) -> None:
        self._companions()
        for name, (descriptor, identity) in self.files.items():
            try:
                validate_private_file(descriptor, self.root / name)
            except QuoteMemError as error:
                raise QuoteMemError(
                    "workspace_binding_mismatch", "Offline database file is unsafe",
                ) from error
            if _file_identity(os.fstat(descriptor)) != identity or (
                _file_identity((self.root / name).lstat()) != self.named_identities[name]
            ):
                raise QuoteMemError("workspace_binding_mismatch",
                                    "Offline database identity changed")


def _readonly_connection(path: Path, _dialect: Any, _record: Any,
                         arguments: list[Any], keywords: dict[str, Any], *,
                         immutable: bool = True) -> None:
    """Constrain HB's actual DBAPI connection to the existing canonical file."""
    if not arguments or not isinstance(arguments[0], str) or Path(arguments[0]) != path:
        raise QuoteMemError("workspace_binding_mismatch", "Offline database path changed")
    arguments[0] = path.as_uri() + ("?mode=ro&immutable=1" if immutable else "?mode=ro")
    keywords["uri"] = True


def _database(path: Path, manager: Any, validate: Callable[[], None], *,
              immutable: bool = True) -> Any:
    from heavenbase.database import Database, DBEngine
    from heavenbase.database.dialect.registry import BootstrapDialectCatalog

    # The same pinned provider configuration the runtime writes with, so the
    # expected column types (packed vectors included) match the stored ones.
    config = copy.deepcopy(packaged_bootstrap()["backend"]["_override"])
    config["local_root"] = path.parent.as_uri()
    config["providers"]["sqlite"]["pragmas"] = []
    engine = DBEngine(
        config=config, manager=None,
        dialects=BootstrapDialectCatalog(config["dialects"]),
    )
    database = Database(
        database=path.as_uri(), provider="sqlite", config=manager,
        config_engine=engine, autocreate=False, pragmas=[],
        pool={"pool_class": "null"},
    )
    # Install before the first connection; HB's engine URL otherwise converts
    # file:// into a native path, losing SQLite's URI mode interpretation.
    def connect(dialect: Any, record: Any, args: list[Any], kwargs: dict[str, Any]) -> None:
        validate()
        _readonly_connection(path, dialect, record, args, kwargs, immutable=immutable)

    try:
        event.listen(database.engine, "do_connect", connect)
    except BaseException:
        database.dispose()
        raise
    return database


def _verified_table(database: Any, entity_id: str, schema: Any) -> Table:
    from heavenbase.backends.sql.base import (
        _sa_col,  # pyright: ignore[reportPrivateUsage]
        _table_name,  # pyright: ignore[reportPrivateUsage]
    )

    name = _table_name(entity_id)
    if not inspect(database.engine).has_table(name):
        raise QuoteMemError("schema_mismatch", "Offline workspace is missing a physical table")
    table = Table(name, MetaData(), autoload_with=database.engine)
    if set(table.c.keys()) != set(schema.fields):
        raise QuoteMemError("schema_mismatch", "Existing physical columns differ")
    dialect = database.engine.dialect
    for field_name, field in schema.fields.items():
        column = cast(Column[Any], table.c[field_name])
        expected = _sa_col(field.dtype, database.dialect_spec)
        if bool(column.primary_key) != field.pk or column.nullable == field.required or (
            str(column.type.compile(dialect=dialect)) != str(expected.compile(dialect=dialect))
        ):
            raise QuoteMemError("schema_mismatch", "Existing physical field definition differs")
        # Reflection retains SQL types, not HB's TypeDecorator result
        # processors. Rebind the exact HB type only after physical parity;
        # this changes query decoding, never the database schema or files.
        column.type = expected
    return table


def _decode(schema: Any, row: dict[str, Any]) -> dict[str, Any]:
    # This is the same HB dtype decode operation as _RowCodec, without
    # constructing a fake workspace or reimplementing JSON/vector codecs.
    return {
        name: (None if value is None and not schema.get(name).required
               else schema.get(name).dtype.decode(value))
        for name, value in row.items()
    }


def _layout_specs(layout: StorageLayout) -> Mapping[str, PhysicalSchemaSpec]:
    if layout is not StorageLayout.NORMALIZED:
        raise QuoteMemError("schema_mismatch", "Only the current storage layout is supported")
    return NORMALIZED_SCHEMA_SPECS


def _layout_entities(dimension: int, layout: StorageLayout) -> dict[str, type[Any]]:
    return build_entity_classes(dimension, specs=_layout_specs(layout))


def _verified_auxiliary_table(database: Any, expected: Table, *, unfinished: bool = False,
                              ) -> Table:
    """Reflect exact HB-owned auxiliary DDL without invoking an ensure method.

    ``unfinished`` admits a table whose declared indexes are not all created yet, as a
    first boot killed between them leaves it (`_verify_auxiliary_tables`); each index it
    has is still exactly a declared one.
    """
    inspector = inspect(database.engine)
    if not inspector.has_table(expected.name):
        raise QuoteMemError("schema_mismatch", "HB auxiliary table group is incomplete")
    actual = Table(expected.name, MetaData(), autoload_with=database.engine)
    dialect = database.engine.dialect
    if list(actual.c.keys()) != list(expected.c.keys()) or (
        list(actual.primary_key.columns.keys()) != list(expected.primary_key.columns.keys())
    ):
        raise QuoteMemError("schema_mismatch", "HB auxiliary columns or primary key differ")
    for name, column in expected.c.items():
        found = actual.c[name]
        if (found.nullable != column.nullable
            or str(found.type.compile(dialect=dialect)) != str(column.type.compile(dialect=dialect))
            or found.server_default is not None or found.computed is not None):
            raise QuoteMemError("schema_mismatch", "HB auxiliary field definition differs")
    indexes = {
        value["name"]: (tuple(value["column_names"]), bool(value["unique"]))
        for value in inspector.get_indexes(expected.name)
    }
    desired = {index.name: (tuple(index.columns.keys()), bool(index.unique))
               for index in expected.indexes}
    expected_checks = {
        constraint.name: str(constraint.sqltext.compile(dialect=dialect))
        for owner in (expected, *expected.c) for constraint in owner.constraints
        if isinstance(constraint, CheckConstraint)
    }
    actual_checks = {value["name"]: value["sqltext"]
                     for value in inspector.get_check_constraints(expected.name)}
    if (not indexes.items() <= desired.items() if unfinished else indexes != desired) or (
        actual_checks != expected_checks
    ) or any(
        value.get("dialect_options") for value in inspector.get_indexes(expected.name)
    ) or inspector.get_unique_constraints(expected.name) or (
        inspector.get_foreign_keys(expected.name)
    ):
        raise QuoteMemError("schema_mismatch",
                            "HB auxiliary index or constraint definition differs")
    return actual


def _verify_capsule_declaration(
    database: Any, workspace_id: str, metadata: Table,
    read: Callable[[Any, Any], list[dict[str, Any]]],
) -> None:
    """Verify the required packaged extension as inert metadata, never Capsule data."""
    from heavenbase.backends.sql.base import _sa_col  # pyright: ignore[reportPrivateUsage]
    from heavenbase.backends.sql.parity.sqlite import column_constraints
    from heavenbase.capsule.runtime import Capsule
    from heavenbase.utils.ids import sql_index_name
    from heavenbase.workspace.metaschema.entity import MetaSchema
    from sqlalchemy import Index

    schema: Any = Capsule.schema()
    table = Table("sys_capsule", MetaData(), *[
        Column(name, _sa_col(field.dtype, database.dialect_spec),
               *column_constraints(name, field.dtype), primary_key=field.pk,
               nullable=not field.required)
        for name, field in schema.fields.items()
    ])
    # The installed SQL inline scalar default (IndexResolver._defaults and
    # SQLBackend._ensure_main_indexes) owns these six Capsule index fields.
    for name in ("name", "namespace", "version", "status", "created_at", "updated_at"):
        Index(sql_index_name(table.name, name), table.c[name])
    _verified_auxiliary_table(database, table)
    selected = read(database, select(metadata).where(
        ((metadata.c.kind == "entity") & (metadata.c.subject_id == schema.entity_id))
        | ((metadata.c.kind == "extension") & (metadata.c.subject_id == "capsule")),
    ).limit(5))
    decoded = [_decode(MetaSchema.schema(), row) for row in selected]
    rows = {row["object_id"]: row for row in decoded}
    expected_ids = {"entity:sys-capsule", "extension:capsule",
                    "registration:entity:sys-capsule", "registration:extension:capsule"}
    if set(rows) != expected_ids or len(selected) != 4 or any(
        row["ws"] != workspace_id or row["active"] is not True for row in rows.values()
    ):
        raise QuoteMemError("schema_mismatch", "Capsule declaration identity or owner differs")
    manifest = yaml.safe_load(resources.files("heavenbase").joinpath(
        "capsule/meta.yaml",
    ).read_text())
    installed = {item["kind"]: item for item in manifest["items"]
                 if item["kind"] in {"entity", "extension"}}
    for kind, identifier in (("entity", "sys-capsule"), ("extension", "capsule")):
        row = rows[f"registration:{kind}:{identifier}"]
        registered = row["meta"]
        expected = installed[kind]
        groups = {"provenance": ["system"], "tag": ["builtin"],
                  "bundle": [manifest["bundle"]], **expected.get("groups", {})}
        if (row["source"] != expected["source"] or row["target"] != expected["target"]
            or registered.get("record_kind") != kind
            or registered.get("identifier") != identifier
            or registered.get("groups") != groups
            or type(registered.get("schema_version")) is not int
            or registered.get("schema_version") != 1
            or type(registered.get("registry_revision")) is not int
            or registered.get("registry_revision", 0) < 1
            or not isinstance(registered.get("registry_fingerprint"), str)
            or re.fullmatch(r"[0-9a-f]{64}", registered["registry_fingerprint"]) is None
            or registered.get("aliases") != expected.get("aliases", [])
            or registered.get("version") != expected["meta"]["version"]
            or rows[f"{kind}:{identifier}"]["meta"].get("registration") != registered):
            raise QuoteMemError("schema_mismatch", "Capsule packaged registration differs")
    entity = rows["entity:sys-capsule"]["meta"]
    if (canonical_json(entity.get("fields")) != canonical_json(schema.to_dict()["fields"])
        or entity.get("entity_id") != schema.entity_id
        or canonical_json(entity.get("options"))
        != canonical_json({"catalog": "full", "gram": True})
        or rows["registration:entity:sys-capsule"]["meta"].get("class")
        != installed["entity"]["target"]):
        raise QuoteMemError("schema_mismatch", "Capsule schema declaration differs")
    extension = rows["extension:capsule"]["meta"]
    registration = rows["registration:extension:capsule"]["meta"]
    definition = installed["extension"]["meta"]["definition"]
    for key in ("identifier", "version", "required", "requires", "entities", "meta"):
        if (canonical_json(extension.get(key)) != canonical_json(definition[key])
            or canonical_json(registration.get(key)) != canonical_json(definition[key])):
            raise QuoteMemError("schema_mismatch", "Capsule required extension closure differs")
    if any(registration.get(key) is not None for key in ("setup", "api", "api_name")):
        raise QuoteMemError("schema_mismatch", "Capsule extension adds executable setup")
    bindings = [_decode(MetaSchema.schema(), row) for row in read(
        database, select(metadata).where(
            metadata.c.kind == "storage", metadata.c.subject_id == schema.entity_id,
        ).limit(len(schema.fields) + 1),
    )]
    expected_routes = {
        (name, "main", "json" if cast(Any, type(field.dtype)).identifier == "json" else "inline")
        for name, field in schema.fields.items()
    }
    if (len(bindings) != len(expected_routes)
        or {(row["field"], row["backend"], row["strategy"]) for row in bindings} != expected_routes
        or any(row["ws"] != workspace_id or row["active"] is not True for row in bindings)):
        raise QuoteMemError("schema_mismatch", "Capsule storage routing differs")


# HeavenBase stores a SHA-256 hex digest of the index options as a fingerprint,
# and a fresh random id in hex as an entity's epoch.
_HEX_DIGITS = frozenset("0123456789abcdef")
_GRAM_FINGERPRINT_LENGTH = 64
_GRAM_EPOCH_LENGTH = 32


def _is_hex(value: object, length: int) -> bool:
    """Return whether a stored value is lowercase hex of exactly `length` digits."""
    return isinstance(value, str) and len(value) == length and set(value) <= _HEX_DIGITS


def _gram_declaration(database: Any, namespace: str) -> Any:
    """Return the installed HeavenBase's inert SQL GRAM table declaration.

    The stand-in backend carries only what the declaration reads: the database,
    the backend name, and whether object ids use MySQL keys, which they never
    do for QuoteMem's SQLite files.
    """
    from heavenbase.backends.sql.strategies.sparse_gram import (
        _SQLGramIndex,  # pyright: ignore[reportPrivateUsage]
    )

    backend = SimpleNamespace(db=database, name="main", _uses_mysql_id_keys=lambda: False)
    return _SQLGramIndex(backend, namespace=namespace)


def _verify_auxiliary_tables(
    database: Any, workspace_id: str, names: set[str],
    read: Callable[[Any, Any], list[dict[str, Any]]],
) -> None:
    """Check fixed auxiliary roles and same-workspace GRAM registration provenance.

    Every GRAM entry must be one the installed HeavenBase writes for this
    workspace's Catalog keywords. It keeps one keyword index per database under
    a shared namespace, the adapter's identifier, with the field's fingerprint
    and the Catalog's epoch, and it seeds a retired marker as the fingerprint
    of the per-workspace namespace that earlier releases indexed under. So the
    metadata may hold only those three entries, and keyword rows may sit only
    under the shared namespace once its fingerprint is written. A store of an
    earlier release, with a real fingerprint or rows under the per-workspace
    namespace, is not migrated: it fails here like any other unknown entry.
    Another workspace's namespace, another field, another entry or a value of
    another shape fails too.

    HeavenBase creates the GRAM tables of a new store before its metadata table, one
    statement at a time, each committed alone, and writes the three entries before that
    table too. A first boot killed in between leaves the store unfinished: no metadata
    table (and no capsule table, made after it), some of the GRAM tables, one perhaps
    without all its indexes, and perhaps the entries. That state alone is admitted: each
    table there is exactly as declared but for the indexes it lacks, the entries pass the
    same checks, and no keyword or Catalog row exists, as none is written before the
    field's registration. HeavenBase's own ensure then creates what is missing, as on a new
    store. A first boot killed later, once the metadata table exists, is not admitted.
    """
    from heavenbase.backends.sql.base import _table_name  # pyright: ignore[reportPrivateUsage]
    from heavenbase.backends.sql.strategies.sparse_gram import (
        _RETIRED_FINGERPRINT,  # pyright: ignore[reportPrivateUsage]
        SQLSparseGramAdapter,
        _SQLGramIndex,  # pyright: ignore[reportPrivateUsage]
    )
    from heavenbase.strategies import SparseGramIndex
    from heavenbase.workspace.catalog.entity import Catalog
    from heavenbase.workspace.metaschema.entity import MetaSchema

    gram_names = {"hb_gram_meta", "hb_gram_keywords", "hb_gram_postings"}
    if not names & (gram_names | {"sys_capsule"}):
        return
    unfinished = "sys_capsule" not in names and (
        _table_name(MetaSchema.schema().entity_id) not in names)
    metadata: Table | None = None
    if not unfinished:
        metadata = _verified_table(database, MetaSchema.schema().entity_id,
                                   MetaSchema.schema())
        if "sys_capsule" in names:
            _verify_capsule_declaration(database, workspace_id, metadata, read)
        if not names & gram_names:
            return
        if names & gram_names != gram_names:
            raise QuoteMemError("schema_mismatch", "HB auxiliary table group is incomplete")
    shared = SQLSparseGramAdapter.identifier
    previous = f"{workspace_id}:main:{shared}"
    # Construct only the installed owner's inert SQLAlchemy declaration. Never
    # ensure, rebuild, mutate, or read keyword/manifest payloads during admission.
    declaration = _gram_declaration(database, shared)
    tables = {table.name: _verified_auxiliary_table(database, table, unfinished=unfinished)
              for table in (declaration.meta, declaration.keywords, declaration.postings)
              if table.name in names}
    # QuoteMem's physical entities store keyword/reference arrays as Json/HyperG.
    # The fixed Catalog Array[ShortText] field is the sole SQL GRAM source here.
    schema: Any = Catalog.schema()
    entity_id, field_name = schema.entity_id, "keywords"
    if not SQLSparseGramAdapter._indexable(schema.get(field_name)):  # pyright: ignore[reportPrivateUsage]
        raise QuoteMemError("schema_mismatch", "Installed Catalog GRAM declaration differs")
    indexed = False
    meta = tables.get("hb_gram_meta")
    if meta is not None:
        if read(database, select(1).select_from(meta).where(
            meta.c.ns.not_in((shared, previous)),
        ).limit(1)):
            raise QuoteMemError("schema_mismatch", "HB GRAM namespace has another owner")
        fingerprint = SQLSparseGramAdapter._fingerprint_key(entity_id, field_name)  # pyright: ignore[reportPrivateUsage]
        epoch = _SQLGramIndex.epoch_key(entity_id)
        allowed = (((meta.c.ns == shared) & meta.c.key.in_((fingerprint, epoch)))
                   | ((meta.c.ns == previous) & (meta.c.key == fingerprint)))
        if read(database, select(1).select_from(meta).where(~allowed).limit(1)):
            raise QuoteMemError("schema_mismatch", "HB GRAM metadata has an unknown entry")
        # The primary index on namespace and entry name leaves at most the three entries
        # allowed above.
        for row in read(database, select(meta.c.ns, meta.c.key, meta.c.value)):
            value = row["value"]
            if row["ns"] == previous:
                valid = value == _RETIRED_FINGERPRINT
            elif row["key"] == epoch:
                valid = _is_hex(value, _GRAM_EPOCH_LENGTH)
            else:
                valid = indexed = _is_hex(value, _GRAM_FINGERPRINT_LENGTH)
            if not valid:
                raise QuoteMemError("schema_mismatch",
                                    "HB GRAM metadata value has an unknown shape")
    if unfinished:
        # No keyword row, and no Catalog row, is written before the field's registration.
        if any(read(database, select(1).select_from(table).limit(1))
               for name, table in tables.items() if name != "hb_gram_meta") or (
                   _table_name(Catalog.schema().entity_id) in names and read(
                       database, select(1).select_from(_verified_table(
                           database, Catalog.schema().entity_id, Catalog.schema())).limit(1))):
            raise QuoteMemError("schema_mismatch", "HB auxiliary table group is incomplete")
        return
    if metadata is None:
        raise QuoteMemError("schema_mismatch", "Offline workspace is missing a physical table")
    # HeavenBase writes keyword rows only under the shared namespace, in the
    # transaction that writes its fingerprint or after it.
    for table in (tables["hb_gram_keywords"], tables["hb_gram_postings"]):
        stray = table.c.ns != shared if indexed else true()
        if read(database, select(1).select_from(table).where(stray).limit(1)):
            raise QuoteMemError("schema_mismatch", "HB GRAM rows lie outside the shared index")
    for table in (tables["hb_gram_keywords"], tables["hb_gram_postings"]):
        if read(database, select(1).select_from(table).where(
            (table.c.entity_id != entity_id) | (table.c.field_name != field_name),
        ).limit(1)):
            raise QuoteMemError("schema_mismatch", "HB GRAM rows have an unknown field")
    registered = [_decode(MetaSchema.schema(), row) for row in read(
        database, select(metadata).where(
            (metadata.c.object_id == "entity:sys-catalog")
            | ((metadata.c.kind == "storage") & (metadata.c.subject_id == entity_id)
               & (metadata.c.field == field_name)),
        ).limit(3),
    )]
    entities = [row for row in registered if row["object_id"] == "entity:sys-catalog"]
    routes = [row for row in registered if row["kind"] == "storage"]
    if (len(registered) != 2 or len(entities) != 1 or len(routes) != 1
        or any(row["ws"] != workspace_id or row["active"] is not True for row in registered)
        or canonical_json(entities[0]["meta"].get("fields"))
        != canonical_json(schema.to_dict()["fields"])
        or (routes[0]["backend"], routes[0]["strategy"]) != ("main", SparseGramIndex.identifier)):
        raise QuoteMemError("schema_mismatch", "HB GRAM field registration differs")


def _read_layout(
    database: Any, *, workspace_id: str, db_uuid: str | None, tenant_id: str,
    dimension: int, read: Callable[[Any, Any], list[dict[str, Any]]],
    allow_unbound: bool,
) -> StorageLayout:
    from heavenbase.backends.sql.base import _table_name  # pyright: ignore[reportPrivateUsage]
    from heavenbase.workspace.catalog.entity import Catalog
    from heavenbase.workspace.metaschema.entity import MetaSchema

    names = set(inspect(database.engine).get_table_names())
    expected_names = {
        _table_name(spec.identifier)
        for layout in StorageLayout for spec in _layout_specs(layout).values()
    }
    system_names = {_table_name(entity.schema().entity_id) for entity in (Catalog, MetaSchema)}
    auxiliary_names = {"sys_capsule", "hb_gram_meta", "hb_gram_keywords", "hb_gram_postings"}
    if names - expected_names - system_names - auxiliary_names:
        raise QuoteMemError("schema_mismatch", "Workspace has an unknown physical table")
    present = names & expected_names
    entities = _layout_entities(dimension, StorageLayout.NORMALIZED)
    # A workspace of another schema line (the previous claim-model columns)
    # fails on its physical columns alone, before any QuoteMem row is read.
    for entity in entities.values():
        schema = cast(Any, entity).schema()
        if _table_name(schema.entity_id) in present:
            _verified_table(database, schema.entity_id, schema)
    _verify_auxiliary_tables(database, workspace_id, names, read)
    control_schema = cast(Any, entities["control"]).schema()
    if _table_name(control_schema.entity_id) in present:
        table = _verified_table(database, control_schema.entity_id, control_schema)
        rows = read(database, select(table).where(
            (table.c.object_id == control_id("binding:workspace"))
            | (table.c.key == "binding:workspace"),
        ).limit(2))
        if rows:
            if len(rows) != 1:
                raise QuoteMemError("workspace_binding_mismatch", "Workspace binding is ambiguous")
            binding = ControlRowV1.model_validate(_decode(control_schema, rows[0]))
            return layout_for_binding(
                binding, workspace_id=workspace_id, db_uuid=db_uuid,
                tenant_id=tenant_id, dimension=dimension,
            )
    if not allow_unbound:
        raise QuoteMemError("workspace_binding_mismatch", "Existing workspace binding is absent")
    # Writer.initialize_base permits a missing binding only before any QuoteMem row
    # exists. Never infer a populated layout from coincidentally matching columns.
    for name in present:
        table = Table(name, MetaData(), autoload_with=database.engine)
        if read(database, select(1).select_from(table).limit(1)):
            raise QuoteMemError("workspace_binding_mismatch",
                                "Populated workspace lost its binding")
    if not present:
        if _table_name(MetaSchema.schema().entity_id) in names:
            metadata = _verified_table(database, MetaSchema.schema().entity_id,
                                            MetaSchema.schema())
            if read(database, select(metadata.c.object_id).where(
                metadata.c.subject_id.startswith("agent-memory-"),
            ).limit(1)):
                raise QuoteMemError("schema_mismatch",
                                    "Declared QuoteMem schema lost its physical tables")
        return StorageLayout.NORMALIZED
    _verify_layout(
        database, workspace_id, dimension, StorageLayout.NORMALIZED, read, partial=True,
    )
    return StorageLayout.NORMALIZED


def probe_workspace_layout(
    root: Path, *, workspace_id: str, db_uuid: str, tenant_id: str,
    dimension: int, validate_owner: Callable[[], None],
) -> StorageLayout:
    """Read the closed binding before runtime construction under a held owner.

    Use HB's Database owner in ordinary read-only SQLite mode, including an
    existing WAL. Unlike the offline deletion path, this startup read does not
    assert the main files are complete or enable SQLite's immutable shortcut.
    Fresh workspaces select normalized storage; empty interrupted initialization
    retains its existing supported declaration.
    """
    from heavenbase.utils.config.manager import ConfigManager

    validate_owner()
    path = root / "memory.db"
    with ExitStack() as stack:
        if not path.exists() and not path.is_symlink():
            if any((root / ("memory.db" + suffix)).exists()
                   or (root / ("memory.db" + suffix)).is_symlink()
                   for suffix in ("-wal", "-shm", "-journal")):
                raise QuoteMemError("workspace_binding_mismatch",
                                    "Missing main database has companions")
            return StorageLayout.NORMALIZED
        descriptor = open_private_file(path, os.O_RDONLY)
        stack.callback(os.close, descriptor)

        def validate() -> None:
            validate_owner()
            validate_private_file(descriptor, path)

        validate()
        if os.fstat(descriptor).st_size == 0:
            if any((root / ("memory.db" + suffix)).exists()
                   or (root / ("memory.db" + suffix)).is_symlink()
                   for suffix in ("-wal", "-shm", "-journal")):
                raise QuoteMemError("workspace_binding_mismatch",
                                    "Empty main database has companions")
            return StorageLayout.NORMALIZED
        if os.fstat(descriptor).st_size < 100:
            raise QuoteMemError("workspace_binding_mismatch", "Main database is incomplete")
        manager = ConfigManager(root=str(root), setup=False)
        force_privacy(manager, workspace_id)
        database = _database(path, manager, validate, immutable=False)
        stack.callback(database.dispose)

        def read(database: Any, statement: Any) -> list[dict[str, Any]]:
            validate()
            response = database.orm_execute(statement, readonly=True)
            try:
                return cast(list[dict[str, Any]], response.to_dicts())
            finally:
                response.close()

        layout = _read_layout(
            database, workspace_id=workspace_id, db_uuid=db_uuid, tenant_id=tenant_id,
            dimension=dimension, read=read, allow_unbound=True,
        )
        from heavenbase.backends.sql.base import _table_name  # pyright: ignore[reportPrivateUsage]

        control_table = _table_name(NORMALIZED_SCHEMA_SPECS["control"].identifier)
        if inspect(database.engine).has_table(control_table):
            schema = cast(Any, build_entity_classes(dimension)["control"]).schema()
            table = _verified_table(database, schema.entity_id, schema)
            bound = bool(read(database, select(table.c.object_id).where(
                table.c.object_id == control_id("binding:workspace"),
            ).limit(1)))
            if bound:
                _verify_layout(database, workspace_id, dimension, layout, read)
        present = any(inspect(database.engine).has_table(_table_name(spec.identifier))
                      for spec in NORMALIZED_SCHEMA_SPECS.values())
        if present:
            system_path = root / "hb-system.db"
            system_descriptor = open_private_file(system_path, os.O_RDONLY)
            stack.callback(os.close, system_descriptor)

            def validate_system() -> None:
                validate()
                validate_private_file(system_descriptor, system_path)

            system = _database(system_path, manager, validate_system, immutable=False)
            stack.callback(system.dispose)
            _verify_registry(system, root, workspace_id, read)
            validate_system()
        validate()
        return layout


def _verify_layout(
    database: Any, workspace_id: str, dimension: int, layout: StorageLayout,
    read: Callable[[Any, Any], list[dict[str, Any]]], *, partial: bool = False,
) -> tuple[Table, dict[str, Table]]:
    """Check stored declarations and routes before any HB schema publication."""
    from heavenbase.backends.sql.base import _table_name  # pyright: ignore[reportPrivateUsage]
    from heavenbase.workspace.catalog.entity import Catalog
    from heavenbase.workspace.metaschema.entity import MetaSchema

    catalog = _verified_table(database, Catalog.schema().entity_id, Catalog.schema())
    metadata = _verified_table(database, MetaSchema.schema().entity_id, MetaSchema.schema())
    workspace = read(database, select(metadata).where(
        metadata.c.object_id == f"workspace:{workspace_id}",
    ).limit(2))
    if len(workspace) != 1:
        raise QuoteMemError("schema_mismatch", "Existing workspace metadata is absent")
    workspace_row = _decode(MetaSchema.schema(), workspace[0])
    if (workspace_row["kind"], workspace_row["subject_id"], workspace_row["ws"],
        workspace_row["active"]) != ("workspace", workspace_id, workspace_id, True):
        raise QuoteMemError("schema_mismatch", "Existing workspace metadata has another owner")
    descriptions = physical_schema_description(dimension, specs=_layout_specs(layout))
    tables: dict[str, Table] = {}
    inspector = inspect(database.engine)
    declared = read(database, select(metadata.c.subject_id).where(
        metadata.c.subject_id.startswith("agent-memory-"),
        metadata.c.kind.in_(("entity", "storage")),
    ).distinct())
    identifiers = {spec.identifier for spec in _layout_specs(layout).values()}
    if any(row["subject_id"] not in identifiers or not inspector.has_table(
        _table_name(row["subject_id"]),
    ) for row in declared):
        raise QuoteMemError("schema_mismatch",
                            "Stored QuoteMem declaration has an unknown or missing table")
    for kind, entity in _layout_entities(dimension, layout).items():
        schema: Any = cast(Any, entity).schema()
        if partial and not inspector.has_table(_table_name(schema.entity_id)):
            continue
        table = _verified_table(database, schema.entity_id, schema)
        tables[kind] = table
        rows = read(database, select(metadata).where(
            metadata.c.subject_id == schema.entity_id,
            metadata.c.kind.in_(("entity", "storage")),
        ).limit(len(schema.fields) + 2))
        decoded = [_decode(MetaSchema.schema(), row) for row in rows]
        if any(row["ws"] != workspace_id or not row["active"] for row in decoded):
            raise QuoteMemError("schema_mismatch", "Stored schema metadata changed its owner")
        entities = [row for row in decoded if row["kind"] == "entity"]
        stored_fields: dict[str, Any] = (
            {} if len(entities) != 1 else entities[0]["meta"].get("fields", {})
        )
        expected_fields = descriptions[kind]["fields"]
        if set(stored_fields) != set(expected_fields) or any(
            {key: stored_fields[name].get(key) for key in expected} != expected
            for name, expected in expected_fields.items()
        ):
            raise QuoteMemError("schema_mismatch", "Existing QuoteMem schema declaration differs")
        routes = {(row["field"], row["backend"], row["strategy"])
                  for row in decoded if row["kind"] == "storage"}
        expected_routes = {(name, value["backend"], value["strategy"])
                           for name, value in descriptions[kind]["routes"].items()}
        if routes != expected_routes or len(decoded) != len(expected_routes) + 1:
            raise QuoteMemError("schema_mismatch", "Existing QuoteMem storage routing differs")
        unique = {tuple(value["column_names"]) for value in
                  inspector.get_unique_constraints(table.name)}
        unique.update(tuple(value["column_names"]) for value in
                      inspector.get_indexes(table.name) if value.get("unique"))
        unique.discard(("object_id",))
        if unique != set(_layout_specs(layout)[kind].unique):
            raise QuoteMemError("schema_mismatch", "Existing QuoteMem unique owners differ")
    return catalog, tables


def _verify_registry(
    database: Any, root: Path, workspace_id: str,
    read: Callable[[Any, Any], list[dict[str, Any]]],
) -> None:
    """Verify HB's retained registry without constructing a workspace."""
    from heavenbase.context.registry import _schema  # pyright: ignore[reportPrivateUsage]
    from heavenbase.utils.registry.codec import (
        _RegistryCodec,  # pyright: ignore[reportPrivateUsage]
    )
    from heavenbase.workspace.registry import WorkspaceSpec

    table = _verified_table(database, "sys-registry-state", _schema())
    rows = read(database, select(table).where(
        table.c.object_id == "system-workspaces",
    ).limit(2))
    if len(rows) != 1:
        raise QuoteMemError("workspace_binding_mismatch", "Existing workspace registry is absent")
    row = _decode(_schema(), rows[0])
    codec = _RegistryCodec[Any, Any](
        "system-workspaces", encode_key=None, decode_key=None,
        encode_value=None, decode_value=None, hash_fn="sha256", max_bytes=4_194_304,
    )
    if type(row["revision"]) is not int or row["revision"] < 1:
        raise QuoteMemError("parity_mismatch", "Workspace registry revision is invalid")
    codec.verify(row["revision"], row["payload"], row["digest"])
    state = codec.decode(row["payload"])
    if set(state.items) != {workspace_id}:
        raise QuoteMemError("workspace_binding_mismatch", "Existing workspace registry differs")
    spec = WorkspaceSpec.from_dict(state.items[workspace_id])
    if spec.id != workspace_id or spec.runtime_only or spec.manifest or spec.extension_roots:
        raise QuoteMemError(
            "workspace_binding_mismatch", "Offline workspace construction is not QuoteMem",
        )
    config = spec.config
    backends = config.get("backends", {})
    if config.get("preset") is not None or set(backends) != {"main"} or (
        backends["main"].get("type") != "sqlite"
        or backends["main"].get("database") != (root / "memory.db").as_uri()
    ):
        raise QuoteMemError("workspace_binding_mismatch", "Workspace database route differs")


class ExistingStore:
    """Own only existing HB read resources beneath an already-held owner.

    No mutation facade escapes. Schema, workspace registry and selected Catalog
    rows are checked through the same read-only connections before metadata can
    authorize an offline status/purge action. Close failure is permanent.
    """

    def __init__(self, root: Path, workspace_id: str, tenant_id: str,
                 dimension: int, owner: StorageOwner) -> None:
        from heavenbase.utils.config.manager import ConfigManager

        self.root = root
        self.workspace_id = workspace_id
        self.tenant_id = tenant_id
        self.owner = owner
        self._closed = False
        self._failed = False
        self._resources = ExitStack()
        self._tables: dict[str, Table] = {}
        manager = ConfigManager(root=str(root), setup=False)
        try:
            if workspace_id.strip().lower() in {"llm-cache", "assistant-log"}:
                raise QuoteMemError("workspace_binding_mismatch",
                                    "Forbidden offline workspace identity")
            force_privacy(manager, workspace_id)
            owner.validate()
            self._files = _ClosedFiles(root, self._resources)
            self._system = _database(root / "hb-system.db", manager, self._check)
            self._resources.callback(self._system.dispose)
            self._main = _database(root / "memory.db", manager, self._check)
            self._resources.callback(self._main.dispose)
            self._registry(workspace_id)
            self._storage_layout = _read_layout(
                self._main, workspace_id=workspace_id, db_uuid=None,
                tenant_id=tenant_id, dimension=dimension, read=self._select,
                allow_unbound=False,
            )
            self._entities = _layout_entities(dimension, self._storage_layout)
            self._layout(dimension)
            self._check()
        except BaseException:
            self._failed = True
            self._resources.close()
            raise

    @property
    def normalized_storage(self) -> bool:
        """Report the verified immutable physical binding without enabling writes."""
        return self._storage_layout.normalized_storage

    def _live(self) -> None:
        if self._closed or self._failed:
            raise QuoteMemError("service_unhealthy", "Offline storage is closed or failed")
        self._check()

    def validate(self) -> None:
        """Recheck retained closed-file evidence without any database read."""
        self._live()

    def _check(self) -> None:
        try:
            self.owner.validate()
            self._files.check()
        except BaseException:
            self._failed = True
            raise

    def close(self) -> None:
        """Dispose every HB pool before the caller releases its filesystem owner."""
        if self._closed:
            if self._failed:
                raise QuoteMemError("service_unhealthy", "Offline storage previously failed")
            return
        try:
            try:
                if self._failed:
                    raise QuoteMemError("service_unhealthy", "Offline storage previously failed")
                self._check()
            finally:
                self._resources.close()
            self.owner.validate()
        except BaseException:
            self._failed = True
            raise
        finally:
            self._closed = True

    def _select(self, database: Any, statement: Any) -> list[dict[str, Any]]:
        self._check()
        response = database.orm_execute(statement, readonly=True)
        try:
            return cast(list[dict[str, Any]], response.to_dicts())
        finally:
            response.close()

    def _registry(self, workspace_id: str) -> None:
        _verify_registry(self._system, self.root, workspace_id, self._select)

    def _layout(self, dimension: int) -> None:
        self._catalog, self._tables = _verify_layout(
            self._main, self.workspace_id, dimension, self._storage_layout, self._select,
        )

    def _catalog_row(self, kind: str, object_id: str) -> None:
        from heavenbase.workspace.catalog.entity import Catalog
        from heavenbase.workspace.catalog.rows import catalog_row_id

        schema: Any = cast(Any, self._entities[kind]).schema()
        rows = self._select(self._main, select(
            self._catalog.c.object_id, self._catalog.c.target_id,
            self._catalog.c.target_entity, self._catalog.c.ws,
        ).where(
            self._catalog.c.target_id == object_id,
            self._catalog.c.target_entity == schema.entity_id,
        ).limit(2))
        expected_id = catalog_row_id(
            self.workspace_id, schema.entity_id, object_id,
            cast(Any, Catalog.schema().get("object_id").dtype).max_chars, hash_length=42,
        )
        if len(rows) != 1 or rows[0]["ws"] != self.workspace_id or (
            rows[0]["object_id"] != expected_id
        ):
            raise QuoteMemError("parity_mismatch", "Existing selected row lost its Catalog entry")

    def get_row(self, kind: str, object_id: str) -> dict[str, Any] | None:
        """Read one existing global metadata row, never source or domain content."""
        self._live()
        if kind not in {"control", "artifact", "run"}:
            raise QuoteMemError("unsupported_variant",
                                "Offline metadata cannot hydrate domain rows")
        table = self._tables[kind]
        statement = select(table).where(table.c.object_id == object_id)
        if kind in {"artifact", "run"}:
            # Restrict before body decoding, including substituted policy refs.
            statement = statement.where(
                table.c.scope_kind.is_(None), table.c.scope_id.is_(None),
            )
        rows = self._select(self._main, statement.limit(2))
        if not rows:
            return None
        if len(rows) != 1:
            raise QuoteMemError("parity_mismatch", "Existing metadata identity is not unique")
        self._catalog_row(kind, object_id)
        row = _decode(cast(Any, self._entities[kind]).schema(), rows[0])
        if kind == "artifact":
            from quotemem.store.normalized_row_codec import restore_row_image

            return restore_row_image(kind, row, tenant_id=self.tenant_id)
        return row

    def get_control(self, key: str) -> dict[str, Any] | None:
        """Read the exact deterministic Control owner."""
        value = self.get_row("control", control_id(key))
        if value is not None and value["key"] != key:
            raise QuoteMemError("binding_mismatch", "Control key differs from its address")
        return value

    def job_rows(self) -> Iterator[dict[str, Any]]:
        """Read the closed job namespace; two rows already prove a conflict."""
        self._live()
        table = self._tables["control"]
        key = table.c.key.collate("BINARY")
        for row in self._select(self._main, select(table).where(
            key >= "job:", key < "job;",
        ).order_by(key).limit(2)):
            self._catalog_row("control", row["object_id"])
            yield _decode(cast(Any, self._entities["control"]).schema(), row)

    def domain_rows(self, kind: str) -> Iterator[dict[str, Any]]:
        """Read every row of one memory-read entity as its restored logical image.

        For an offline listing of memories (``quotemem memories`` with no running
        service): the memories, revisions, links, entities, names, candidates and Evidence a
        cold read root projects. Rows are decoded with HB's own dtype codecs and restored
        exactly as the Writer restores them (``restore_row_image``); nothing is written.
        """
        from quotemem.store.normalized_row_codec import restore_row_image

        if kind not in {"item", "revision", "link", "entity", "entity_name", "candidate",
                        "evidence"}:
            raise QuoteMemError("unsupported_variant", "Offline listing reads no such entity")
        self._live()
        table = self._tables[kind]
        schema: Any = cast(Any, self._entities[kind]).schema()
        for row in self._select(self._main, select(table).order_by(table.c.object_id)):
            yield restore_row_image(kind, _decode(schema, row), tenant_id=self.tenant_id)

    def controls(self, prefix: str) -> list[dict[str, Any]]:
        """Read every Control whose key starts with ``prefix`` (one namespace, by key)."""
        self._live()
        table = self._tables["control"]
        key = table.c.key.collate("BINARY")
        schema: Any = cast(Any, self._entities["control"]).schema()
        return [_decode(schema, row) for row in self._select(self._main, select(table).where(
            key >= prefix, key < prefix + "\U0010ffff").order_by(key))]

    def row_counts(self) -> tuple[tuple[str, int], ...]:
        """Count the admitted physical entities without reading their content."""
        self._live()
        counts: list[tuple[str, int]] = []
        for kind, table in sorted(self._tables.items()):
            rows = self._select(self._main, select(func.count().label("total")).select_from(table))
            if len(rows) != 1 or type(rows[0]["total"]) is not int or rows[0]["total"] < 0:
                raise QuoteMemError("parity_mismatch", "Offline entity count is invalid")
            counts.append((kind, rows[0]["total"]))
        return tuple(counts)

    def read_logical(self, reference: str, expected_kind: str) -> StrictModel:
        """Use the same complete Artifact validator as the sole Writer."""
        if expected_kind not in {"policy_release", "policy_config", "run_input"}:
            raise QuoteMemError("unsupported_variant", "Offline metadata has no content authority")
        return read_logical_artifact(self.get_row, reference, expected_kind)
