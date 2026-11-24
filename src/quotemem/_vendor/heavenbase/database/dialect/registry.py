"""Registry-backed physical database dialect metadata."""

from __future__ import annotations

__all__ = [
    "BootstrapDialectCatalog",
    "DatabaseDialectSpec",
    "DatabaseDialectCatalog",
    "MYSQL_TEXT_MIN_LEN",
    "ORACLE_VARCHAR_MAX",
    "ResolverDialectCatalog",
    "SQL_COMMON_RESERVED_WORDS",
    "SQL_TYPE_CLOB",
    "SQL_TYPE_FLOAT64_BLOB",
    "SQL_TYPE_FLOAT_ARRAY",
    "SQL_TYPE_JSON",
    "SQL_TYPE_JSON_TEXT",
    "SQL_TYPE_MYSQL_TEXT",
    "SQL_TYPE_POSTGRESQL_JSONB",
    "SQL_TYPE_STRING",
    "SQL_TYPE_TEXT",
    "database_dialects",
]

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Optional, Protocol, runtime_checkable

from ...modules import ModuleResolver, ModuleService, module_resolver
from ...modules.base import ResolverCache

from .spec import (
    DatabaseDialectSpec,
    MYSQL_TEXT_MIN_LEN,
    ORACLE_VARCHAR_MAX,
    SQL_COMMON_RESERVED_WORDS,
    SQL_TYPE_CLOB,
    SQL_TYPE_FLOAT64_BLOB,
    SQL_TYPE_FLOAT_ARRAY,
    SQL_TYPE_JSON,
    SQL_TYPE_JSON_TEXT,
    SQL_TYPE_MYSQL_TEXT,
    SQL_TYPE_POSTGRESQL_JSONB,
    SQL_TYPE_STRING,
    SQL_TYPE_TEXT,
    _live_name_warning_rules,
)
from ...utils.ids import check_id


def _dialect_key(value: str) -> str:
    """Return one canonical physical database dialect identifier."""
    return check_id(str(value).strip().lower(), "database dialect identifier")


def _sqlglot_key(value: str) -> str:
    """Return one canonical SQLGlot dialect identifier."""
    return check_id(str(value).strip().lower(), "SQLGlot dialect identifier")


def _database_dialect_definition(spec: DatabaseDialectSpec) -> dict[str, Any]:
    """Return canonical inline metadata for one physical dialect spec."""
    definition = spec.to_dict()
    definition.pop("identifier")
    definition.pop("tags")
    return definition


def _database_dialect_record(spec: DatabaseDialectSpec) -> dict[str, Any]:
    """Return one plain central registration record for `spec`."""
    return {
        "name": f"{spec.identifier.replace('_', ' ').title()} Database Dialect",
        "desc": f"Physical database metadata for {spec.identifier}.",
        "source": "inline",
        "target": "definition",
        "active": True,
        "meta": {
            "schema_version": 1,
            "tags": dict(spec.tags),
            "definition": _database_dialect_definition(spec),
        },
    }


_DATABASE_DIALECT_CACHE = ResolverCache[DatabaseDialectSpec]()


@runtime_checkable
class DatabaseDialectCatalog(Protocol):
    """Provide one explicit physical-dialect lookup authority."""

    def get(self, name: str) -> DatabaseDialectSpec:
        """Return physical metadata for one canonical identifier or alias.

        Args:
            name (str): Canonical SQLAlchemy dialect identifier or alias.

        Returns:
            DatabaseDialectSpec: Physical dialect metadata for ``name``.
        """

    def snapshot(self) -> Mapping[str, DatabaseDialectSpec]:
        """Return immutable-by-contract lookup metadata for config capture.

        Args:
            None.

        Returns:
            Mapping[str, DatabaseDialectSpec]: Immutable dialect lookup snapshot.
        """


class BootstrapDialectCatalog:
    """Adapt validated bootstrap data to the dialect-catalog contract."""

    def __init__(self, definitions: Mapping[str, Any]) -> None:
        """Adapt validated bootstrap dialect definitions to the catalog contract.

        Args:
            definitions (Mapping[str, Any]): Bootstrap dialect identifier to
                metadata mapping used before the module catalog opens.

        Returns:
            None: This function does not return a value.
        """
        if not isinstance(definitions, Mapping):
            raise TypeError("Bootstrap database dialects must be a mapping")
        catalog: dict[str, DatabaseDialectSpec] = {}
        for identifier, raw in definitions.items():
            if not isinstance(raw, Mapping):
                raise TypeError(f"Bootstrap database dialect {identifier!r} must be a mapping")
            spec = DatabaseDialectSpec(str(identifier), **dict(raw))
            for key in (spec.identifier, *spec.sqlalchemy_aliases):
                if key in catalog:
                    raise ValueError(f"Bootstrap database dialect key {key!r} conflicts with another definition")
                catalog[key] = spec
        if not catalog:
            raise ValueError("Bootstrap database dialects must not be empty")
        self._specs = MappingProxyType(catalog)

    def get(self, name: str) -> DatabaseDialectSpec:
        """Return bootstrap metadata for one identifier or alias.

        Args:
            name (str): Canonical SQLAlchemy dialect identifier or alias.

        Returns:
            DatabaseDialectSpec: Bootstrap physical dialect metadata.

        Raises:
            LookupError: If ``name`` is absent from the bootstrap catalog.
        """
        key = _dialect_key(name)
        try:
            return self._specs[key]
        except KeyError as error:
            raise LookupError(f"Database dialect {key!r} is absent from the bootstrap catalog") from error

    def snapshot(self) -> Mapping[str, DatabaseDialectSpec]:
        """Return the immutable bootstrap lookup snapshot.

        Args:
            None.

        Returns:
            Mapping[str, DatabaseDialectSpec]: Identifier and alias keys mapped to
            bootstrap dialect metadata.
        """
        return self._specs


class ResolverDialectCatalog:
    """Adapt one ModuleResolver to the dialect-catalog contract."""

    def __init__(self, resolver: ModuleResolver | None = None) -> None:
        """Bind one module resolver as the dialect lookup authority.

        Args:
            resolver (ModuleResolver | None): Context-owned module resolver. When
                omitted, lookups use the default Context authority.

        Returns:
            None: This function does not return a value.
        """
        self.resolver = resolver

    def get(self, name: str) -> DatabaseDialectSpec:
        """Resolve one dialect through this catalog's module authority.

        Args:
            name (str): Canonical SQLAlchemy dialect identifier or alias.

        Returns:
            DatabaseDialectSpec: Registry-materialized physical dialect metadata.
        """
        return database_dialects.get(name, resolver=self.resolver)

    def snapshot(self) -> Mapping[str, DatabaseDialectSpec]:
        """Capture registered dialects and their SQLAlchemy aliases.

        Args:
            None.

        Returns:
            Mapping[str, DatabaseDialectSpec]: Immutable identifier and alias keys
            mapped to registered dialect metadata.
        """
        catalog: dict[str, DatabaseDialectSpec] = {}
        for spec in database_dialects.captured(resolver=self.resolver).values():
            for key in (spec.identifier, *spec.sqlalchemy_aliases):
                existing = catalog.get(key)
                if existing is not None and existing.identifier != spec.identifier:
                    raise ValueError(f"Database dialect key {key!r} has conflicting registrations")
                catalog[key] = spec
        return MappingProxyType(catalog)


def _database_dialect_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> DatabaseDialectSpec:
    """Materialize and cache one validated physical dialect record."""

    def build(current_row: Mapping[str, Any]) -> DatabaseDialectSpec:
        identifier = str(current_row["identifier"])
        raw = resolver.resolve("database_dialect", identifier)
        if not isinstance(raw, Mapping):
            raise TypeError(f"Database dialect {identifier!r} definition must be a mapping")
        allowed = set(DatabaseDialectSpec(identifier).to_dict()) - {"identifier", "tags"}
        unsupported = set(raw) - allowed
        if unsupported:
            raise ValueError(f"Database dialect {identifier!r} definition has unsupported fields {sorted(unsupported)!r}")
        values = dict(raw)
        raw_meta = current_row["record"].get("meta")
        tags = raw_meta.get("tags", {}) if isinstance(raw_meta, Mapping) else {}
        if not isinstance(tags, Mapping):
            raise TypeError(f"Database dialect {identifier!r} record tags must be a mapping")
        values["tags"] = dict(tags)
        aliases = tuple(str(alias) for alias in current_row["aliases"])
        declared_aliases = tuple(str(alias) for alias in values.get("sqlalchemy_aliases", ()))
        if declared_aliases and declared_aliases != aliases:
            raise ValueError(f"Database dialect {identifier!r} aliases disagree with its Registry indexes")
        values["sqlalchemy_aliases"] = aliases
        spec = DatabaseDialectSpec(identifier, **values)
        indexed_sqlglot = tuple(str(value) for value in current_row["groups"].get("sqlglot", ()))
        expected_sqlglot = (spec.sqlglot_name(),) if spec.sqlglot_preferred else ()
        if indexed_sqlglot != expected_sqlglot:
            raise ValueError(f"Database dialect {identifier!r} SQLGlot groups disagree with its definition")
        return spec

    return _DATABASE_DIALECT_CACHE.get(resolver, row, build)


class _DatabaseDialects:
    """Typed authoring and lookup facade over central database dialect records."""

    def register(
        self,
        spec: DatabaseDialectSpec,
        *,
        replace: bool = False,
        resolver: ModuleResolver | None = None,
    ) -> DatabaseDialectSpec:
        """Publish one durable physical SQLAlchemy dialect specification.

        Args:
            spec: Physical database metadata to publish.
            replace: Whether to reconcile a changed same-owner definition.
            resolver: Writable Context-owned module resolver. When omitted,
                the default Context is selected.

        Returns:
            DatabaseDialectSpec: Registry-materialized specification.

        Raises:
            TypeError: If `spec` is not a `DatabaseDialectSpec`.
            ValueError: If identity, preferred SQLGlot ownership, or record
                ownership conflicts with an existing registration.
        """
        if not isinstance(spec, DatabaseDialectSpec):
            raise TypeError("database_dialects.register expects DatabaseDialectSpec")
        chosen = module_resolver(resolver)
        if not isinstance(chosen, ModuleService):
            raise TypeError("Database dialect registration requires a writable ModuleService")
        service = chosen
        found = service.inspect("database_dialect", spec.identifier)
        coordinate = f"programmatic/database_dialect/{spec.identifier}"
        if found:
            existing = _database_dialect_from_row(found[0], service)
            if existing.to_dict() == spec.to_dict():
                return existing
            if not replace:
                raise ValueError(f"Database dialect {spec.identifier!r} is already registered")
            existing_coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
            if existing_coordinate != coordinate:
                raise ValueError(f"Database dialect {spec.identifier!r} belongs to {existing_coordinate!r} and cannot be replaced")
        if spec.sqlglot_preferred:
            preferred = self.sqlglot_map(resolver=service).get(spec.sqlglot_name())
            if preferred is not None and preferred != spec.identifier:
                raise ValueError(f"SQLGlot dialect {spec.sqlglot_name()!r} already resolves to {preferred!r}")
        groups = []
        if spec.sqlglot_preferred:
            groups.append(("sqlglot", spec.sqlglot_name()))
        service.register(
            "database_dialect",
            spec.identifier,
            _database_dialect_record(spec),
            aliases=spec.sqlalchemy_aliases,
            groups=groups,
            coordinate=coordinate,
        )
        return _database_dialect_from_row(service.inspect("database_dialect", spec.identifier)[0], service)

    def get(
        self,
        name: str,
        *,
        resolver: ModuleResolver | None = None,
    ) -> DatabaseDialectSpec:
        """Resolve registered physical metadata for a SQLAlchemy dialect.

        Args:
            name: Canonical SQLAlchemy dialect identifier or alias.
            resolver: Context-owned module resolver. When omitted, the default
                Context is selected.
        Returns:
            DatabaseDialectSpec: Registry-materialized physical metadata.

        Raises:
            LookupError: If no active registration owns `name`.
        """
        key = _dialect_key(name)
        chosen = module_resolver(resolver)
        found = chosen.inspect("database_dialect", key)
        if not found:
            raise LookupError(f"Unknown database dialect {key!r}; install its registration before use")
        return _database_dialect_from_row(found[0], chosen)

    def captured(self, *, resolver: ModuleResolver | None = None) -> dict[str, DatabaseDialectSpec]:
        """Return an import-free snapshot of registered physical dialects."""
        return self.snapshot(resolver=resolver)

    def snapshot(self, *, resolver: ModuleResolver | None = None) -> dict[str, DatabaseDialectSpec]:
        """Return registered physical dialects by canonical identifier."""
        chosen = module_resolver(resolver)
        return {str(row["identifier"]): _database_dialect_from_row(row, chosen) for row in chosen.inspect("database_dialect")}

    def names(self, *, resolver: ModuleResolver | None = None) -> frozenset[str]:
        """Return registered canonical SQLAlchemy dialect identifiers."""
        return frozenset(self.snapshot(resolver=resolver))

    def to_sqlglot(
        self,
        name: Optional[str],
        default: Optional[str] = "sqlite",
        *,
        resolver: ModuleResolver | None = None,
    ) -> Optional[str]:
        """Resolve a SQLAlchemy dialect to its SQLGlot identifier."""
        key = str(name or "").strip().lower()
        return self.get(key, resolver=resolver).sqlglot_name() if key else default

    def from_sqlglot(
        self,
        name: Optional[str],
        default: Optional[str] = "sqlite",
        *,
        resolver: ModuleResolver | None = None,
    ) -> Optional[str]:
        """Resolve a SQLGlot dialect to its preferred SQLAlchemy identifier.

        Raises:
            LookupError: If a non-empty dialect has no preferred registration.
        """
        key = str(name or "").strip().lower()
        if not key:
            return default
        identifier = self.sqlglot_map(resolver=resolver).get(_sqlglot_key(key))
        if identifier is None:
            raise LookupError(f"Unknown SQLGlot dialect {key!r}; install its preferred database dialect registration before use")
        return identifier

    def superuser_database(
        self,
        name: str,
        fallback: Optional[str] = None,
        *,
        resolver: ModuleResolver | None = None,
    ) -> Optional[str]:
        """Resolve the database token for a superuser connection."""
        database = self.get(name, resolver=resolver).superuser_database
        return fallback if database == "{database}" else database

    def sqlalchemy_map(self, *, resolver: ModuleResolver | None = None) -> dict[str, str]:
        """Return SQLAlchemy-to-SQLGlot mappings, including aliases."""
        chosen = module_resolver(resolver)
        rows = chosen.inspect("database_dialect")
        out: dict[str, str] = {}
        for row in rows:
            spec = _database_dialect_from_row(row, chosen)
            out[spec.identifier] = spec.sqlglot_name()
            out.update({str(alias): spec.sqlglot_name() for alias in row["aliases"]})
        return out

    def sqlglot_map(self, *, resolver: ModuleResolver | None = None) -> dict[str, str]:
        """Return preferred SQLGlot-to-SQLAlchemy mappings."""
        chosen = module_resolver(resolver)
        out: dict[str, str] = {}
        for row in chosen.inspect("database_dialect"):
            spec = _database_dialect_from_row(row, chosen)
            if spec.sqlglot_preferred:
                existing = out.get(spec.sqlglot_name())
                if existing is not None and existing != spec.identifier:
                    raise ValueError(f"SQLGlot dialect {spec.sqlglot_name()!r} has conflicting preferred registrations")
                out[spec.sqlglot_name()] = spec.identifier
        return out

    def warning_rules(
        self,
        *names: str,
        dialect: Any = None,
        extra_reserved: Iterable[str] = (),
        resolver: ModuleResolver | None = None,
    ) -> tuple[int | None, set[str]]:
        """Combine identifier warning rules for registered physical dialects."""
        max_length, reserved = _live_name_warning_rules(dialect)
        for name in names:
            if not str(name or "").strip():
                continue
            provider_max, provider_reserved = self.get(name, resolver=resolver).name_warning_rules(extra_reserved=())
            if max_length is None:
                max_length = provider_max
            reserved.update(provider_reserved)
        reserved.update(str(word).strip().lower() for word in extra_reserved if str(word).strip())
        return max_length, reserved


database_dialects = _DatabaseDialects()
