"""Canonical SQL database configuration and registered Apply profiles."""

from __future__ import annotations

__all__ = ["DBEngine", "DBSpec"]

import hashlib
import re
from collections.abc import Mapping
from dataclasses import replace
from os.path import isabs, normcase, realpath
from types import MappingProxyType
from typing import Any
from urllib.parse import quote

import sqlalchemy
from sqlalchemy.engine import URL, make_url
from sqlalchemy.pool import AssertionPool, NullPool, QueuePool, SingletonThreadPool, StaticPool

from heavenbase.utils.config import CM_HVNB, ConfigEngine, ConfigSpec
from heavenbase.utils.config.data import dmerge, to_mutable_data
from heavenbase.utils.config.engine import redact_paths, redact_url
from heavenbase.utils.config.spec import _Memoized
from heavenbase.database.dialect.registry import DatabaseDialectCatalog, ResolverDialectCatalog
from heavenbase.database.dialect.spec import DatabaseDialectSpec
from heavenbase.utils.files import file_uri_to_path, normalize_file_uri, path_to_file_uri, pj

POOL_CLASS_MAP = {
    "assertion": AssertionPool,
    "null": NullPool,
    "queue": QueuePool,
    "singleton": SingletonThreadPool,
    "static": StaticPool,
}

#: Pool classes whose engine cannot serve two threads at once: ``static`` hands every checkout the same DBAPI
#: connection, ``assertion`` refuses a second checkout, and ``singleton`` closes other threads' connections, even one
#: still running a statement, once more than five threads have checked out.
SERIALIZED_POOL_CLASSES = frozenset({"assertion", "singleton", "static"})

_FIELDS = (
    "preset",
    "provider",
    "dialect",
    "driver",
    "vector_packed_type",
    "username",
    "password",
    "host",
    "port",
    "database",
    "catalog",
    "in_memory",
    "schema",
    "service",
    "sid",
    "dsn",
    "params",
    "pool",
    "pragmas",
    "args",
    "superuser",
)
_STRUCTURAL = frozenset((*_FIELDS, "url", "superuser", "aliases", "desc", "local_root", "default_args", "infer_from_url"))
_SECRET_QUERY_NAMES = ("access_token", "api_key", "auth", "key", "password", "token")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_LOGICAL_NAME = re.compile(r"^[^/\\]+$")
_MISSING = object()
_POOL_KEYS = frozenset(
    {
        "connect_args",
        "max_overflow",
        "pool_class",
        "pool_pre_ping",
        "pool_recycle",
        "pool_size",
        "pool_timeout",
    }
)


class DBSpec(ConfigSpec):
    """Store immutable canonical SQL fields and selected lifecycle bindings."""

    @property
    def preset(self) -> str | None:
        """Return the selected database preset."""
        return self.get("preset")

    @property
    def provider(self) -> str:
        """Return the canonical provider identifier."""
        return str(self.get("provider") or "")

    @property
    def dialect(self) -> str:
        """Return the canonical SQLAlchemy dialect identifier."""
        return str(self.get("dialect") or "")

    @property
    def driver(self) -> str | None:
        """Return the optional SQLAlchemy driver identifier."""
        return self.get("driver")

    @property
    def vector_packed_type(self) -> str | None:
        """Return the Vector storage form chosen for this database, or ``None`` for the dialect's own default.

        ``float64_blob`` packs vectors as little-endian float64 BLOBs; ``json_text`` keeps the JSON
        text form. The bound ``dialect_spec`` already carries the effective value.
        """
        return self.get("vector_packed_type")

    @property
    def username(self) -> str | None:
        """Return the authentication principal."""
        return self.get("username")

    @property
    def password(self) -> str | None:
        """Return the selected credential."""
        return self.get("password")

    @property
    def host(self) -> str | None:
        """Return the selected server host."""
        return self.get("host")

    @property
    def port(self) -> int | None:
        """Return the selected server port."""
        value = self.get("port")
        return int(value) if value is not None else None

    @property
    def database(self) -> str | None:
        """Return the canonical database scope or local-file URI."""
        return self.get("database")

    @_Memoized
    def _database_key(self) -> str:
        """Identify the database this spec opens, captured for the lifetime of this spec.

        Pool options, engine arguments, the driver, credentials and connection setup do not change which database a
        facade opens, so the engine registry groups schema invalidation and drops by this key. A local file is
        identified by its resolved path, so symbolic links share a key. A server database is identified by its
        dialect, host, port, URL database and query, and the namespace that ``drop_database()`` removes (the target
        user/schema on Oracle).
        """
        if self.in_memory:
            target: tuple[Any, ...] = ("memory", self.dialect)
        elif self.database is not None and self.dialect_spec.supports("database-local-path"):
            path = file_uri_to_path(self.database)
            target = ("file", normcase(realpath(path)) if isabs(path) else path)
        else:
            url = _url(self)
            namespace = self.dialect_spec.lifecycle_namespace(self)
            target = ("server", self.dialect, url.host, url.port, url.database, sorted(url.query.items()), namespace)
        return hashlib.sha256(repr(target).encode("utf-8")).hexdigest()

    @property
    def catalog(self) -> str | None:
        """Return the selected catalog scope."""
        return self.get("catalog")

    @property
    def in_memory(self) -> bool:
        """Return whether this spec selects an isolated memory database."""
        return bool(self.get("in_memory"))

    @property
    def serialized(self) -> bool:
        """Return whether every use of this spec's engine must run one thread at a time.

        An in-memory database exists only inside its first connection, and a ``static``, ``assertion`` or
        ``singleton`` pool cannot hand connections to two threads at once. ``Database`` keeps such an engine private
        to its facade, so the facade's owner can serialize every use of it with one lock.
        """
        return self.in_memory or str(self.pool.get("pool_class") or "queue").lower() in SERIALIZED_POOL_CLASSES

    @property
    def schema(self) -> str | None:
        """Return the selected SQL/object schema."""
        return self.get("schema")

    @property
    def service(self) -> str | None:
        """Return the selected Oracle service name."""
        return self.get("service")

    @property
    def sid(self) -> str | None:
        """Return the selected Oracle SID."""
        return self.get("sid")

    @property
    def dsn(self) -> str | None:
        """Return the selected Oracle DSN."""
        return self.get("dsn")

    @property
    def params(self) -> Mapping[str, Any]:
        """Return immutable lossless URL query parameters."""
        return self.get("params", {})

    @property
    def pool(self) -> Mapping[str, Any]:
        """Return immutable SQLAlchemy pool configuration."""
        return self.get("pool", {})

    @property
    def pragmas(self) -> tuple[str, ...]:
        """Return connection initialization statements."""
        return tuple(self.get("pragmas", ()))

    @property
    def args(self) -> Mapping[str, Any]:
        """Return immutable engine and HeavenBase runtime arguments."""
        return self.get("args", {})

    @property
    def superuser(self) -> Mapping[str, Any]:
        """Return immutable lifecycle-only superuser configuration."""
        return self.bindings.get("superuser", {})

    @property
    def dialect_spec(self) -> DatabaseDialectSpec:
        """Return the immutable physical dialect metadata captured at resolution.

        Returns:
            DatabaseDialectSpec: Bound dialect metadata used by execution, DDL,
                reflection, and SQLAlchemy type compilation.

        Raises:
            ValueError: If the resolved spec has no valid dialect binding.
        """
        raw = to_mutable_data(self.bindings.get("dialect", {}))
        identifier = raw.pop("identifier", None)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("DBSpec has no bound database dialect metadata")
        return DatabaseDialectSpec(identifier, **raw)


class _DBCachePolicy:
    def fingerprint(self, value: Any) -> Any:
        return _db_fingerprint(value)

    def freeze(self, value: Any) -> Any:
        return _db_freeze(value)

    def thaw(self, value: Any) -> Any:
        return _db_thaw(value)


class DBEngine(ConfigEngine):
    """Resolve SQL config to canonical fields and deterministic Apply values."""

    domain = "db"
    config_path = "heavenbase.db"
    spec_type = DBSpec
    binding_contracts = ("db-route/v1",)

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        manager: Any = CM_HVNB,
        *,
        dialects: DatabaseDialectCatalog | None = None,
        bootstrap_backend_tags: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a database config engine with optional bootstrap dialect metadata.

        Args:
            config (Mapping[str, Any] | None): Optional ``heavenbase.db`` root override.
            manager (Any): Config manager owning profile resolution and cache keys.
            dialects (DatabaseDialectCatalog | None): Dialect lookup authority. Defaults
                to a resolver-backed catalog.
            bootstrap_backend_tags (Mapping[str, Any] | None): Construction-only Backend
                tag facts available before the module catalog opens.

        Returns:
            None: This function does not return a value.
        """
        if bootstrap_backend_tags is not None and not isinstance(bootstrap_backend_tags, Mapping):
            raise TypeError("DBEngine bootstrap Backend tags must be a mapping")
        tag_values = dict(bootstrap_backend_tags or {})
        if not all(isinstance(identifier, str) for identifier in tag_values):
            raise TypeError("DBEngine bootstrap Backend tags must be a string-keyed mapping")
        workspace_namespace = tag_values.get("workspace-database-namespace")
        if "workspace-database-namespace" in tag_values and not isinstance(workspace_namespace, bool):
            raise TypeError("DBEngine bootstrap Backend tag 'workspace-database-namespace' must be Boolean")
        self._bootstrap_backend_tags = _db_freeze(tag_values)
        root = _db_root(config)
        self.dialects = ResolverDialectCatalog() if dialects is None else dialects
        if not isinstance(self.dialects, DatabaseDialectCatalog):
            raise TypeError("DBEngine dialects must implement DatabaseDialectCatalog")
        super().__init__(manager=manager, root=root)
        policy = _DBCachePolicy()
        self.register_profile("url", self._apply_url, cache_policy=policy)
        self.register_profile("uri", self._apply_uri, cache_policy=policy)
        self.register_profile("engine", self._apply_engine, cache_policy=policy)
        self.register_profile("superuser", self._apply_superuser, cache_policy=policy)
        self.register_profile("cache-key", self._apply_cache_key)
        self.register_profile("pragmas", self._apply_pragmas)
        self.register_profile("print", self._apply_print)

    def bootstrap_backend_tag(self, identifier: str, default: Any = None) -> Any:
        """Return one construction-only control-Backend tag fact.

        This narrow adapter is populated only from :class:`BootstrapSpec` while
        the control Backend is creating the Registry. Ordinary Backend Tag
        inspection remains Registry-backed.

        Args:
            identifier (str): Bootstrap Backend tag identifier.
            default (Any): Value returned when the tag is absent.

        Returns:
            Any: Tag value from the bootstrap snapshot, or ``default``.
        """
        return self._bootstrap_backend_tags.get(str(identifier), default)

    def _prepare_catalog(self, root: Mapping[str, Any], raw: Mapping[str, Any]) -> None:
        direct = to_mutable_data(raw)
        explicit = set(direct)
        expanded = self._expand_preset(direct, root)
        preset = expanded.get("preset")
        preset_cfg = to_mutable_data(root.get("presets", {}).get(preset, {}) if preset else {})
        parsed = _parse_url(direct.get("url"), self.dialects) if "url" in explicit and direct.get("url") is not None else {}
        provider = self._select_provider(expanded, parsed, explicit, root)
        provider_cfg = to_mutable_data(root.get("providers", {}).get(provider, {}))
        dialect = str(_last_value([provider_cfg, preset_cfg, parsed, direct], "dialect") or provider_cfg.get("dialect") or "").strip().lower()
        if dialect:
            self.dialects.get(dialect)

    def _capture_catalog(self, root: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        catalog = {name: spec.to_dict() for name, spec in self.dialects.snapshot().items()}
        revision = hashlib.sha256(repr(catalog).encode("utf-8")).hexdigest()
        return revision, {"dialects": catalog}

    def _normalize(self, data: Mapping[str, Any], *, context: Any) -> Mapping[str, Any]:
        root = context.root
        expanded = to_mutable_data(data)
        explicit = {path[0] for path in context.explicit_paths if path}
        preset = expanded.get("preset")
        preset_cfg = to_mutable_data(root.get("presets", {}).get(preset, {}) if preset else {})
        parsed = _parse_url(expanded.get("url"), self.dialects) if "url" in explicit and expanded.get("url") is not None else {}
        provider = self._select_provider(expanded, parsed, explicit, root)
        provider_cfg = to_mutable_data(root.get("providers", {}).get(provider, {}))
        direct = {key: to_mutable_data(value) for key, value in expanded.items() if key in explicit}

        scalar_layers = [provider_cfg, preset_cfg, parsed, direct]
        values = {name: _last_value(scalar_layers, name) for name in _FIELDS if name not in {"params", "pool", "pragmas", "args"}}
        values["provider"] = provider
        values["preset"] = preset
        values["dialect"] = str(values.get("dialect") or provider_cfg.get("dialect") or "").strip().lower()
        values["driver"] = _optional_text(values.get("driver"))
        values["vector_packed_type"] = _optional_text(values.get("vector_packed_type"))
        values["port"] = int(values["port"]) if values.get("port") is not None else None

        values["params"] = dmerge(
            [
                to_mutable_data(provider_cfg.get("params", {}) or {}),
                to_mutable_data(preset_cfg.get("params", {}) or {}),
                to_mutable_data(parsed.get("params", {}) or {}),
                to_mutable_data(direct.get("params", {}) or {}),
            ]
        )
        values["pool"] = dmerge(
            [
                to_mutable_data(provider_cfg.get("pool", {}) or {}),
                to_mutable_data(preset_cfg.get("pool", {}) or {}),
                to_mutable_data(direct.get("pool", {}) or {}),
            ]
        )
        explicit_pool = {path[1] for path in context.explicit_paths if len(path) > 1 and path[0] == "pool"}
        values["pragmas"] = list(_last_value([provider_cfg, preset_cfg, direct], "pragmas") or [])
        values["superuser"] = dmerge(
            [
                to_mutable_data(provider_cfg.get("superuser", {}) or {}),
                to_mutable_data(preset_cfg.get("superuser", {}) or {}),
                to_mutable_data(direct.get("superuser", {}) or {}),
            ]
        )
        values["args"] = dmerge(
            [
                to_mutable_data(root.get("default_args", {}) or {}),
                _extra_args(provider_cfg),
                to_mutable_data(provider_cfg.get("args", {}) or {}),
                _extra_args(preset_cfg),
                to_mutable_data(preset_cfg.get("args", {}) or {}),
                _extra_args(direct),
                to_mutable_data(direct.get("args", {}) or {}),
            ]
        )
        values.pop("url", None)
        dialect_spec = _dialect_spec(context, values["dialect"])
        self._normalize_scopes(
            values,
            parsed=parsed,
            explicit=explicit,
            explicit_pool=explicit_pool,
            root=root,
            dialect_spec=dialect_spec,
        )
        if str(values["pool"].get("pool_class", "queue") or "queue").lower() != "queue":
            for key in ("pool_size", "max_overflow", "pool_timeout"):
                if key not in explicit_pool:
                    values["pool"].pop(key, None)
        variables = {name: values.get(name) for name in ("database", "catalog", "schema", "service", "sid", "username")}
        for name in ("host", "schema"):
            values[name] = _template(values.get(name), variables)
        values["params"] = _template(values["params"], variables)
        return {name: values.get(name) if name not in {"params", "pool", "args", "superuser"} else values[name] for name in _FIELDS}

    def _bind(self, data: Mapping[str, Any], *, context: Any) -> Mapping[str, Any]:
        variables = {name: data.get(name) for name in ("database", "catalog", "schema", "service", "sid", "username")}
        dialect = _dialect_spec(context, str(data["dialect"]))
        if data.get("vector_packed_type") is not None:
            # the per-database choice overrides the dialect record and is validated like it
            dialect = replace(dialect, vector_packed_type=data["vector_packed_type"])
        return {
            "contract": "db-route/v1",
            "provider_dialect": data["dialect"],
            "dialect": dialect.to_dict(),
            "superuser": _template(to_mutable_data(data.get("superuser", {}) or {}), variables),
        }

    def _validate(self, spec: DBSpec, *, context: Any) -> None:
        if not spec.provider or not spec.dialect:
            raise ValueError("DBSpec requires provider and dialect")
        if spec.bindings.get("provider_dialect") != spec.dialect:
            raise ValueError(f"Database provider/dialect pair is unsupported: {spec.provider!r}/{spec.dialect!r}")
        dialect = spec.dialect_spec
        if dialect.identifier != spec.dialect:
            raise ValueError(f"Unknown bound database dialect contract for {spec.dialect!r}")
        if dialect.requires_host_for_url() and not spec.host and not spec.dsn:
            raise ValueError(f"Database provider {spec.provider!r} requires host")
        if spec.port is not None and not 1 <= spec.port <= 65535:
            raise ValueError("Database port must be between 1 and 65535")
        local = dialect.supports("database-local-path")
        if spec.in_memory and (spec.database or spec.host or not local):
            raise ValueError("Database memory mode conflicts with database, host, or server dialect")
        if local and not spec.in_memory:
            if not spec.database or not str(spec.database).startswith("file:///"):
                raise ValueError("Local database config must normalize to a canonical file:/// URI")
        dialect.validate_scope(spec)
        _pool_kwargs(spec.pool)

    def _default_profile(self, spec: DBSpec) -> str:
        return "engine"

    def _select_provider(
        self,
        expanded: Mapping[str, Any],
        parsed: Mapping[str, Any],
        explicit: set[str],
        root: Mapping[str, Any],
    ) -> str:
        providers = root.get("providers", {})
        if "provider" in explicit and expanded.get("provider") is not None:
            return _provider_name(expanded["provider"], providers)
        if parsed.get("dialect"):
            dialect = str(parsed["dialect"])
            driver = parsed.get("driver")
            # A provider that differs from another only in defaults (a hosted variant) opts out with
            # ``infer_from_url: false``; a URL already carries every setting, so it names the plain provider.
            candidates = [
                name
                for name, config in providers.items()
                if config.get("dialect") == dialect
                and (driver is None or config.get("driver") in {None, driver})
                and config.get("infer_from_url", True) is not False
            ]
            if len(candidates) != 1:
                choices = ", ".join(sorted(candidates)) or "<none>"
                raise ValueError(f"Database URL provider inference for {dialect!r} is ambiguous; pass provider=... from: {choices}")
            return str(candidates[0])
        return _provider_name(expanded.get("provider") or root.get("default_provider") or "sqlite", providers)

    def _normalize_scopes(
        self,
        values: dict[str, Any],
        *,
        parsed: Mapping[str, Any],
        explicit: set[str],
        explicit_pool: set[str],
        root: Mapping[str, Any],
        dialect_spec: DatabaseDialectSpec,
    ) -> None:
        local = dialect_spec.supports("database-local-path")
        database = values.get("database")
        in_memory = bool(values.get("in_memory")) or (dialect_spec.supports_memory and database == ":memory:") or bool(parsed.get("in_memory"))
        if local:
            values["in_memory"] = in_memory
            values["host"] = None
            if in_memory:
                values["database"] = None
                values["pool"] = dialect_spec.normalize_memory_pool(values["pool"], explicit=explicit_pool)
            else:
                values["database"] = self._local_database(database, from_url="url" in explicit, root=root)
        else:
            values["in_memory"] = False
        dialect_spec.normalize_scope(values, parsed=parsed, explicit=explicit)

    def _local_database(self, value: Any, *, from_url: bool, root: Mapping[str, Any]) -> str:
        if value is None:
            raise ValueError("Local database requires a database token or memory mode")
        home, resource = self._file_roots()
        local_root = file_uri_to_path(str(root.get("local_root")), home=home, resource=resource)
        token = str(value)
        if token.startswith("file://"):
            if token.startswith("file://&/"):
                raise ValueError("Writable local databases cannot target packaged resources through file://&/")
            return normalize_file_uri(token, home=home, resource=resource)
        if from_url:
            native = token
            if not (native.startswith("/") or _WINDOWS_ABSOLUTE.match(native)):
                native = pj(local_root, native)
            return path_to_file_uri(native)
        if not _LOGICAL_NAME.fullmatch(token) or token in {".", ".."}:
            raise ValueError("Local database fields accept a logical name or explicit file:// URI, not a native path")
        return path_to_file_uri(pj(local_root, token))

    def _file_roots(self) -> tuple[str, str | None]:
        manager = self.manager or CM_HVNB
        home = str(getattr(manager, "root", CM_HVNB.root))
        resource = manager.pj("&", abs=True) if hasattr(manager, "pj") else None
        return home, resource

    def _apply_url(self, spec: DBSpec) -> URL:
        return _url(spec)

    def _apply_uri(self, spec: DBSpec) -> str:
        url = _url(spec)
        if spec.dialect_spec.file_uri_query and spec.database and not spec.in_memory:
            native = file_uri_to_path(spec.database)
            database = f"file:{quote(native, safe='/')}"
            query = {**_url_query(spec.params), "uri": "true"}
            url = URL.create(url.drivername, database=database, query=query)
        return url.render_as_string(hide_password=False)

    def _apply_engine(self, spec: DBSpec) -> dict[str, Any]:
        args = to_mutable_data(spec.args)
        for key in ("sql_healing", "lock_retry", "http_proxy", "https_proxy", "no_proxy", "aliases", "args", "type", "name", "backend_type", "desc"):
            args.pop(key, None)
        return {"url": _url(spec), **_pool_kwargs(spec.pool), **args}

    def _apply_superuser(self, spec: DBSpec) -> dict[str, Any]:
        superuser = to_mutable_data(spec.bindings.get("superuser", {}) or {})
        if not superuser:
            return self._apply_engine(spec)
        data = spec.to_dict()["data"]
        data.update({key: value for key, value in superuser.items() if key != "params"})
        data["params"] = dmerge([to_mutable_data(spec.params), superuser.get("params", {}) or {}])
        spec.dialect_spec.normalize_superuser(data, superuser, database=spec.database)
        target = DBSpec(spec.domain, data, bindings=spec.bindings)
        return {"url": _url(target), "isolation_level": "AUTOCOMMIT"}

    def _apply_cache_key(self, spec: DBSpec) -> str:
        identity = _db_fingerprint((self._apply_engine(spec), spec.bindings.get("dialect", {})))
        return hashlib.sha256(repr((identity, spec.pragmas)).encode("utf-8")).hexdigest()

    @staticmethod
    def _apply_pragmas(spec: DBSpec) -> list[str]:
        return list(spec.pragmas)

    def _apply_print(self, spec: DBSpec) -> dict[str, Any]:
        data = redact_paths(
            {"domain": spec.domain, "data": to_mutable_data(spec.data)},
            (
                "data.password",
                "data.args.password",
                "data.args.api_key",
                "data.args.auth_token",
                "data.params.password",
                "data.params.api_key",
                "data.params.auth_token",
                "data.params.token",
                "data.superuser.password",
                "data.superuser.params.password",
                "data.superuser.params.api_key",
                "data.superuser.params.auth_token",
                "data.superuser.params.token",
            ),
        )
        uri = redact_url(self._apply_uri(spec), query_names=_SECRET_QUERY_NAMES)
        data["uri"] = uri
        return data


def _db_root(data: Mapping[str, Any] | None) -> dict[str, Any]:
    value = to_mutable_data(data or {})
    return to_mutable_data((value.get("heavenbase", {}) or {}).get("db", {}) or value)


def _dialect_spec(context: Any, dialect: str) -> DatabaseDialectSpec:
    value = context.catalog.get("dialects", {}).get(dialect)
    if not isinstance(value, Mapping):
        raise ValueError(f"Database dialect {dialect!r} is not registered in the captured catalog")
    raw = to_mutable_data(value)
    identifier = raw.pop("identifier", dialect)
    return DatabaseDialectSpec(str(identifier), **raw)


def _provider_name(value: Any, providers: Mapping[str, Any]) -> str:
    token = str(value or "").strip().lower()
    if token in providers:
        return token
    aliases = {str(alias).strip().lower(): name for name, config in providers.items() for alias in config.get("aliases", ())}
    if token in aliases:
        return str(aliases[token])
    choices = ", ".join(sorted(providers)) or "<none>"
    raise ValueError(f"Unknown database provider {token!r}; expected one of: {choices}")


def _parse_url(value: Any, dialects: DatabaseDialectCatalog) -> dict[str, Any]:
    try:
        url = make_url(str(value))
    except Exception as exc:
        raise ValueError(f"Invalid SQLAlchemy database URL: {exc}") from exc
    dialect, _, driver = url.drivername.partition("+")
    params = {key: list(item) if isinstance(item, tuple) else item for key, item in url.query.items()}
    database, params, in_memory = dialects.get(dialect).parse_url_database(url.database, params)
    out = {
        "dialect": dialect.lower(),
        "in_memory": in_memory,
        "params": params,
    }
    for key, item in {
        "driver": driver or None,
        "username": url.username,
        "password": url.password,
        "host": url.host,
        "port": url.port,
        "database": database,
    }.items():
        if item is not None:
            out[key] = item
    return out


def _last_value(layers: list[Mapping[str, Any]], key: str) -> Any:
    value = None
    for layer in layers:
        if key in layer:
            value = to_mutable_data(layer[key])
    return value


def _extra_args(layer: Mapping[str, Any]) -> dict[str, Any]:
    return {key: to_mutable_data(value) for key, value in layer.items() if key not in _STRUCTURAL}


def _optional_text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _template(value: Any, variables: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        out = value
        for key, item in variables.items():
            if item is not None:
                out = out.replace(f"{{{key}}}", str(item))
        return out
    if isinstance(value, Mapping):
        return {key: _template(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_template(item, variables) for item in value]
    return value


def _url(spec: DBSpec) -> URL:
    drivername = f"{spec.dialect}+{spec.driver}" if spec.driver else spec.dialect
    database, host, params = spec.dialect_spec.url_scope(spec, to_mutable_data(spec.params))
    if spec.in_memory:
        database = ":memory:"
    elif database and database.startswith("file:///"):
        database = file_uri_to_path(database)
    return URL.create(
        drivername,
        username=spec.username,
        password=spec.password,
        host=host,
        port=spec.port,
        database=database,
        query=_url_query(params),
    )


def _url_query(params: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in params.items():
        if isinstance(value, list):
            out[str(key)] = tuple(str(item) for item in value)
        elif isinstance(value, tuple):
            out[str(key)] = tuple(str(item) for item in value)
        else:
            out[str(key)] = str(value)
    return out


def _pool_kwargs(pool: Mapping[str, Any]) -> dict[str, Any]:
    config = to_mutable_data(pool or {})
    unknown = sorted(set(config).difference(_POOL_KEYS))
    if unknown:
        raise ValueError(f"Unknown database pool keys: {', '.join(unknown)}")
    pool_class = str(config.pop("pool_class", "queue") or "queue").lower()
    if pool_class not in POOL_CLASS_MAP:
        choices = ", ".join(sorted(POOL_CLASS_MAP))
        raise ValueError(f"Unknown database pool class {pool_class!r}; expected one of: {choices}")
    out: dict[str, Any] = {"poolclass": POOL_CLASS_MAP[pool_class]}
    connect_args = config.pop("connect_args", None)
    if connect_args is not None and not isinstance(connect_args, Mapping):
        raise TypeError("Database pool.connect_args must be a mapping")
    if connect_args:
        out["connect_args"] = connect_args
    if config.pop("pool_pre_ping", False):
        out["pool_pre_ping"] = True
    queue_keys = ("pool_size", "max_overflow", "pool_timeout")
    if pool_class == "queue":
        for key in queue_keys:
            if key in config:
                out[key] = config[key]
    elif any(key in config for key in queue_keys):
        raise ValueError(f"Database pool class {pool_class!r} does not accept queue sizing options")
    if int(config.get("pool_recycle", -1) or -1) > 0:
        out["pool_recycle"] = config["pool_recycle"]
    return out


def _db_fingerprint(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return type(value).__name__, value
    if isinstance(value, URL):
        return "url", value.render_as_string(hide_password=False)
    if isinstance(value, type):
        return "type", value.__module__, value.__qualname__
    if isinstance(value, Mapping):
        return "mapping", tuple(sorted((str(key), _db_fingerprint(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return "sequence", tuple(_db_fingerprint(item) for item in value)
    raise TypeError(f"Uncacheable database Apply value {type(value).__name__}")


def _db_thaw(value: Any) -> Any:
    if isinstance(value, (URL, type)):
        return value
    if isinstance(value, Mapping):
        return {key: _db_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_db_thaw(item) for item in value]
    return value


def _db_freeze(value: Any) -> Any:
    if isinstance(value, (URL, type)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({key: _db_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_db_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_db_freeze(item) for item in value)
    return value


DB_ENGINE = DBEngine()
