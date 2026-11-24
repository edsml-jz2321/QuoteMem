"""Universal staged configuration resolution and Apply profiles."""

from __future__ import annotations

__all__ = ["ConfigEngine"]

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .data import _freeze, dget, dmerge, to_mutable_data
from .interpolation import InterpolationPolicy, _capture_env, _Environment, _resolve_interpolation
from .manager import CM_HVNB, _env_names
from .spec import ConfigSpec, resolve_interpolations

_MISSING = object()
_DEFAULT_NAME_DESC = "Human-readable name; used for deterministic object_id when object_id is omitted"
_FINAL_METHODS = frozenset({"normalize", "validate", "resolve", "apply", "assert_idempotent"})
_JSON_SCALARS = (str, int, float, bool, type(None))


class _Uncacheable(TypeError):
    pass


@dataclass(frozen=True)
class _ResolutionContext:
    root: Mapping[str, Any]
    catalog: Mapping[str, Any]
    catalog_revision: Any
    source_token: Any
    explicit_paths: frozenset[tuple[str, ...]]


@dataclass(frozen=True)
class _CapturedSource:
    root: Mapping[str, Any]
    catalog: Mapping[str, Any]
    catalog_revision: Any
    source_token: Any
    environment: _Environment | None

    def context(self, raw: Mapping[str, Any]) -> _ResolutionContext:
        """Build one resolution context from captured source metadata.

        Args:
            raw (Mapping[str, Any]): Raw configuration mapping being resolved.

        Returns:
            _ResolutionContext: Frozen resolution context for one lookup.
        """
        return _ResolutionContext(
            root=self.root,
            catalog=self.catalog,
            catalog_revision=self.catalog_revision,
            source_token=self.source_token,
            explicit_paths=_explicit_paths(raw),
        )


@dataclass(frozen=True)
class _CachePolicy:
    fingerprint: Callable[[Any], Any]
    freeze: Callable[[Any], Any]
    thaw: Callable[[Any], Any]


@dataclass(frozen=True)
class _Profile:
    name: str
    fn: Callable[..., Any]
    validate: Callable[..., Any] | None
    aliases: tuple[str, ...]
    cache_policy: _CachePolicy | None


class ConfigEngine:
    """Resolve one configuration domain through four fixed stages.

    Subclasses declare ``domain``, optional ``config_path`` and optional
    ``schema`` class metadata, then implement protected normalization and
    validation hooks. Public orchestration is final by contract and always runs
    ``Preset -> Normalize -> Validate -> Apply``.

    Args:
        manager (Any): Config source owner. ``None`` skips manager lookup.
        root (Mapping[str, Any] | None): Frozen domain-root overlay used by
            bootstrap paths and tests. It merges above manager data.
        policy (InterpolationPolicy | None): Interpolation permissions.
    """

    domain = ""
    config_path: str | None = None
    schema: Any = None
    spec_type: type[ConfigSpec] = ConfigSpec
    binding_contracts: tuple[str, ...] = ()
    contract_revision: int | str = 1
    binding_revision: int | str = 1
    validator_revision: int | str = 1
    extra = "allow"
    extras_field: str | None = None
    schema_skip = ("object_id",)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        overridden = sorted(_FINAL_METHODS.intersection(cls.__dict__))
        if overridden:
            raise TypeError(f"{cls.__name__} cannot override final ConfigEngine method(s): {', '.join(overridden)}")

    def __init__(
        self,
        manager: Any = CM_HVNB,
        root: Mapping[str, Any] | None = None,
        policy: InterpolationPolicy | None = None,
    ) -> None:
        """Initialize one configuration engine bound to a manager.

        Returns:
            None: This function does not return a value.
        """
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError(f"{type(self).__name__}.domain must be a non-empty string")
        if root is not None and not isinstance(root, Mapping):
            raise TypeError(f"ConfigEngine root must be a mapping, got {type(root).__name__}")
        if self.extra not in {"allow", "forbid", "collect"}:
            raise ValueError(f"Unknown config extra mode {self.extra!r}")
        if self.extra == "collect" and not self.extras_field:
            raise ValueError("ConfigEngine extra='collect' requires extras_field")
        self.domain = self.domain.strip()
        self.manager = manager
        self.policy = policy or InterpolationPolicy()
        self._root_overlay = _freeze(root or {})
        self._schema = self.schema.schema() if hasattr(self.schema, "schema") else self.schema
        from heavenbase.utils.registry import Registry

        self._profiles = Registry[str, _Profile](
            "config-profiles",
            normalize=lambda value: _identifier(value, "profile"),
            normalize_alias=lambda value: _identifier(value, "profile alias"),
            value_policy="process_identity",
        )
        self._apply_revision = 0
        self._lock = threading.RLock()
        self._resolved_cache: OrderedDict[Any, ConfigSpec] = OrderedDict()
        self._applied_cache: OrderedDict[Any, tuple[_CachePolicy, Any]] = OrderedDict()
        self.max_resolved, self.max_applied = self._cache_bounds()

    def root(self) -> Mapping[str, Any]:
        """Return a read-only snapshot of the current domain root.

        Args:
            None.

        Returns:
            Mapping[str, Any]: Frozen manager/root-overlay snapshot.
        """
        return self._capture_source().root

    def presets(self) -> Mapping[str, Any]:
        """Return the current read-only preset catalog.

        Args:
            None.

        Returns:
            Mapping[str, Any]: Frozen preset-name to template mapping.
        """
        presets = self.root().get("presets", {})
        if not isinstance(presets, Mapping):
            raise TypeError(f"{self.domain}.presets must be a mapping")
        return _freeze(presets)

    def profiles(self) -> Mapping[str, Any]:
        """Return read-only Apply profile discovery metadata.

        Args:
            None.

        Returns:
            Mapping[str, Any]: Profile metadata keyed by canonical name.
        """
        with self._lock:
            return _freeze(
                {name: {"aliases": list(profile.aliases), "cacheable": profile.cache_policy is not None} for name, profile in sorted(self._profiles.items())}
            )

    def register_profile(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        validate: Callable[..., Any] | None = None,
        aliases: Sequence[str] = (),
        cache_policy: Any = None,
        replace: bool = False,
    ) -> None:
        """Register one deterministic final-stage Apply projection.

        Args:
            name (str): Canonical profile identifier.
            fn (Callable[..., Any]): Function called as ``fn(spec, **kwargs)``.
            validate (Callable[..., Any] | None): Optional pre-Apply validator
                called with the same arguments.
            aliases (Sequence[str]): Alternate identifiers accepted at Apply.
            cache_policy (Any): ``None`` uses JSON-like structural caching;
                ``False`` disables caching; otherwise provide an object with
                ``fingerprint``, ``freeze``, and ``thaw`` callables.
            replace (bool): Whether to replace an existing canonical profile.

        Returns:
            None: This function registers the profile in-place.

        Raises:
            TypeError: If a callback or cache policy is invalid.
            ValueError: If identifiers collide without ``replace=True``.
        """
        canonical = _identifier(name, "profile")
        if not callable(fn):
            raise TypeError("Config Apply profile must be callable")
        if validate is not None and not callable(validate):
            raise TypeError("Config Apply profile validator must be callable")
        alias_values = tuple(dict.fromkeys(_identifier(alias, "profile alias") for alias in aliases))
        if canonical in alias_values:
            raise ValueError(f"Profile {canonical!r} cannot alias itself")
        policy = _profile_policy(cache_policy)
        registered = _Profile(canonical, fn, validate, alias_values, policy)
        with self._lock:
            existing = canonical in self._profiles.keys()
            if existing and not replace:
                raise ValueError(f"Config profile {canonical!r} is already registered for {self.domain}")
            try:
                if existing:
                    self._profiles.update(canonical, registered, aliases=alias_values)
                else:
                    self._profiles.create(canonical, registered, aliases=alias_values)
            except ValueError as error:
                raise ValueError(f"Config profile identifier collision for {self.domain}: {error}") from error
            self._apply_revision += 1
            self._applied_cache.clear()

    def normalize(self, config: Any = None) -> ConfigSpec:
        """Run Preset and Normalize without validation or caching.

        Args:
            config (Any): Raw mapping, object with ``to_dict``, compatible
                ``ConfigSpec``, or ``None`` for an empty direct mapping.

        Returns:
            ConfigSpec: Immutable canonical specification.
        """
        compatible = self._compatible_spec(config)
        if compatible is not None:
            return compatible
        raw = _as_mapping(config)
        source = self._capture_source(raw)
        spec, _ = self._normalize_from(raw, source)
        return spec

    def validate(self, config: Any) -> ConfigSpec:
        """Strictly validate a canonical or raw configuration.

        Args:
            config (Any): Compatible ``ConfigSpec`` or raw mapping.

        Returns:
            ConfigSpec: The same immutable specification after validation.

        Raises:
            ValueError: If schema or domain invariants are invalid.
        """
        compatible = self._compatible_spec(config)
        if compatible is not None:
            return self._validate_spec(compatible, _source_free_context())
        raw = _as_mapping(config)
        source = self._capture_source(raw)
        spec, context = self._normalize_from(raw, source)
        return self._validate_spec(spec, context)

    def resolve(self, config: Any = None) -> ConfigSpec:
        """Resolve and cache Preset, Normalize, and Validate.

        Args:
            config (Any): Raw mapping, compatible executable spec, or ``None``.

        Returns:
            ConfigSpec: Immutable validated specification.
        """
        compatible = self._compatible_spec(config)
        if compatible is not None:
            key = self._spec_cache_key(compatible)
            if key is _MISSING or not self.max_resolved:
                return self._validate_spec(compatible, _source_free_context())
            with self._lock:
                cached = self._cache_get(self._resolved_cache, key)
                if cached is not None:
                    return cached
                spec = self._validate_spec(compatible, _source_free_context())
                self._cache_put(self._resolved_cache, key, spec, self.max_resolved)
                return spec

        raw = _as_mapping(config)
        source = self._capture_source(raw)
        raw_key = _try_fingerprint(raw)
        key = None if raw_key is _MISSING or source.environment is None else ("raw", self.domain, self._resolution_revision(), source.source_token, raw_key)
        if key is None or not self.max_resolved:
            spec, context = self._normalize_from(raw, source)
            return self._validate_spec(spec, context)
        with self._lock:
            cached = self._cache_get(self._resolved_cache, key)
            if cached is not None:
                return cached
            spec, context = self._normalize_from(raw, source)
            spec = self._validate_spec(spec, context)
            self._cache_put(self._resolved_cache, key, spec, self.max_resolved)
            return spec

    def apply(self, config: Any = None, *, profile: str | None = None, **kwargs: Any) -> Any:
        """Apply a registered final-stage profile to validated config.

        Args:
            config (Any): Raw mapping or compatible executable spec.
            profile (str | None): Explicit profile identifier. When omitted,
                the domain's pure ``_default_profile`` selector is used.
            **kwargs (Any): Profile-specific immutable Apply context.

        Returns:
            Any: Defensive profile result containing values, never live clients
            or open connections.

        Raises:
            ValueError: If no profile is selected or the profile is unknown.
        """
        spec = self.resolve(config)
        selected = profile if profile is not None else self._default_profile(spec)
        if selected is None:
            raise ValueError(f"Config Apply for {self.domain!r} requires an explicit profile")
        selected_name = _identifier(selected, "profile")
        with self._lock:
            canonical = self._profiles.resolve(selected_name)
            registered = self._profiles.get(canonical)
            if registered is None:
                choices = ", ".join(sorted(self._profiles.keys())) or "<none>"
                raise ValueError(f"Unknown config profile {selected!r} for {self.domain}; expected one of: {choices}")
            apply_revision = self._apply_revision
        kwargs_key = _try_fingerprint(kwargs)
        key = None
        if registered.cache_policy is not None and kwargs_key is not _MISSING:
            spec_key = spec._cache_identity
            if spec_key is not None:
                key = ("apply", spec_key, canonical, apply_revision, kwargs_key)
            if key is not None and self.max_applied:
                with self._lock:
                    cached = self._cache_get(self._applied_cache, key)
                    if cached is not None:
                        policy, value = cached
                        return policy.thaw(value)
                    return self._apply_uncached(registered, spec, kwargs, key)
        return self._apply_uncached(registered, spec, kwargs, key)

    def _apply_uncached(self, registered: _Profile, spec: ConfigSpec, kwargs: Mapping[str, Any], key: Any) -> Any:
        if registered.validate is not None:
            registered.validate(spec, **kwargs)
        value = registered.fn(spec, **kwargs)
        if key is None or registered.cache_policy is None:
            return value
        try:
            registered.cache_policy.fingerprint(value)
            frozen = registered.cache_policy.freeze(value)
        except (TypeError, _Uncacheable):
            return value
        self._cache_put(self._applied_cache, key, (registered.cache_policy, frozen), self.max_applied)
        return registered.cache_policy.thaw(frozen)

    def assert_idempotent(self, config: Any = None) -> ConfigSpec:
        """Verify two uncached Normalize/Bind passes against one source.

        Args:
            config (Any): Raw, normalized, or serialized configuration input.

        Returns:
            ConfigSpec: The first canonical specification when both passes are
            behaviorally identical.

        Raises:
            RuntimeError: If canonical data or executable bindings change.
        """
        raw = to_mutable_data(config.data) if isinstance(config, ConfigSpec) else _as_mapping(config)
        source = self._capture_source(raw)
        first, _ = self._normalize_from(raw, source)
        second, _ = self._normalize_from(to_mutable_data(first.data), source)
        if first.data != second.data:
            path = _first_diff(first.data, second.data)
            raise RuntimeError(f"Config normalization for {self.domain!r} is not idempotent at data{_format_path(path)}")
        if first.bindings != second.bindings:
            path = _first_diff(first.bindings, second.bindings)
            left = _secret_digest(dget(first.bindings, _dotted(path)))
            right = _secret_digest(dget(second.bindings, _dotted(path)))
            raise RuntimeError(f"Config binding for {self.domain!r} is not idempotent at bindings{_format_path(path)} ({left} != {right})")
        return first

    def clear_cache(self) -> None:
        """Clear this engine's resolution and Apply caches.

        Args:
            None.

        Returns:
            None: This function clears cached values in-place.
        """
        with self._lock:
            self._resolved_cache.clear()
            self._applied_cache.clear()

    def _normalize(self, data: Mapping[str, Any], *, context: _ResolutionContext) -> Mapping[str, Any]:
        return data

    def _bind(self, data: Mapping[str, Any], *, context: _ResolutionContext) -> Mapping[str, Any]:
        return {}

    def _validate(self, spec: ConfigSpec, *, context: _ResolutionContext) -> None:
        return None

    def _default_profile(self, spec: ConfigSpec) -> str | None:
        return None

    def _capture_catalog(self, root: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        return 0, {}

    def _prepare_catalog(self, root: Mapping[str, Any], raw: Mapping[str, Any]) -> None:
        """Load only catalog metadata required by one raw input."""
        return None

    def _capture_source(self, raw: Mapping[str, Any] | None = None) -> _CapturedSource:
        manager_root: Mapping[str, Any] = {}
        manager_token: Any = None
        path = self.config_path
        if self.manager is not None:
            if path is None:
                path = f"{getattr(self.manager, 'package', 'heavenbase')}.{self.domain.replace('-', '_')}"
            if hasattr(self.manager, "snapshot"):
                snapshot = self.manager.snapshot(policy=self.policy)
                manager_root = snapshot.get(path, default={}) or {}
                manager_token = (
                    tuple(getattr(self.manager, "scope_chain", ())),
                    tuple(getattr(snapshot, "gen", ())),
                    getattr(self.manager, "_cache_epoch", 0),
                    getattr(snapshot, "policy", self.policy).digest(),
                )
            else:
                bundled = dget(self.manager.load_default(), path, default={}) if hasattr(self.manager, "load_default") else {}
                runtime = self.manager.get(path, default={}) if hasattr(self.manager, "get") else {}
                manager_root = dmerge([bundled or {}, runtime or {}])
                manager_token = _try_fingerprint(manager_root)
        if not isinstance(manager_root, Mapping):
            raise TypeError(f"Config root {path!r} for {self.domain} must be a mapping")
        root = resolve_interpolations(dmerge([to_mutable_data(manager_root), to_mutable_data(self._root_overlay)]), self.policy)
        frozen_root = _freeze(root)
        self._prepare_catalog(frozen_root, _freeze(raw or {}))
        revision, catalog = self._capture_catalog(frozen_root)
        if not isinstance(catalog, Mapping):
            raise TypeError(f"{self.domain} catalog capture must return a mapping")
        frozen_catalog = _freeze(catalog)
        catalog_key = _try_fingerprint(frozen_catalog)
        root_key = _try_fingerprint(frozen_root)
        overlay_key = _try_fingerprint(self._root_overlay)
        if catalog_key is _MISSING or root_key is _MISSING or overlay_key is _MISSING:
            raise TypeError(f"{self.domain} root/catalog snapshot contains an unfingerprintable value")
        environment = _capture_env(_env_names(raw) | _env_names(frozen_root))
        environment_digest = environment.digest if environment is not None else None
        source_token = (manager_token, root_key, overlay_key, self.policy.digest(), environment_digest, revision, catalog_key)
        return _CapturedSource(frozen_root, frozen_catalog, revision, source_token, environment)

    def _normalize_from(self, raw: Mapping[str, Any], source: _CapturedSource) -> tuple[ConfigSpec, _ResolutionContext]:
        direct = to_mutable_data(raw)
        context = source.context(direct)
        expanded = _resolve_interpolation(self._expand_preset(direct, source.root), self.policy, source.environment)
        normalized = self._normalize(_freeze(expanded), context=context)
        if not isinstance(normalized, Mapping):
            raise TypeError(f"{self.domain}._normalize must return a mapping, got {type(normalized).__name__}")
        data = self._schema_normalize(to_mutable_data(normalized))
        frozen_data = _freeze(data)
        bindings = self._bind(frozen_data, context=context)
        if not isinstance(bindings, Mapping):
            raise TypeError(f"{self.domain}._bind must return a mapping, got {type(bindings).__name__}")
        spec = self.spec_type(
            self.domain,
            data,
            bindings=to_mutable_data(bindings),
            raw=direct,
            meta={"catalog_revision": source.catalog_revision},
        )
        self._check_binding_contract(spec)
        return spec, context

    def _expand_preset(self, raw: dict[str, Any], root: Mapping[str, Any]) -> dict[str, Any]:
        selected = raw.get("preset", root.get("default_preset", _MISSING)) if "preset" not in raw else raw["preset"]
        if selected is _MISSING:
            return to_mutable_data(raw)
        if selected is None:
            return to_mutable_data(raw)
        if not isinstance(selected, str) or not selected.strip():
            raise TypeError(f"{self.domain}.preset must be a non-empty string or null")
        name = selected.strip()
        presets = root.get("presets", {})
        if not isinstance(presets, Mapping):
            raise TypeError(f"{self.domain}.presets must be a mapping")
        if name not in presets:
            choices = ", ".join(sorted(str(key) for key in presets)) or "<none>"
            raise ValueError(f"Unknown preset {name!r} for {self.domain}; expected one of: {choices}")
        template = presets[name]
        if not isinstance(template, Mapping):
            raise TypeError(f"{self.domain}.presets.{name} must be a mapping")
        return dmerge([to_mutable_data(template), raw, {"preset": name}])

    def _validate_spec(self, spec: ConfigSpec, context: _ResolutionContext) -> ConfigSpec:
        self._schema_validate(spec.data)
        before = _try_fingerprint((spec.data, spec.bindings))
        self._validate(spec, context=context)
        after = _try_fingerprint((spec.data, spec.bindings))
        if before is not _MISSING and after is not _MISSING and before != after:
            raise RuntimeError(f"{self.domain}._validate must not mutate its ConfigSpec")
        return spec

    def _compatible_spec(self, config: Any) -> ConfigSpec | None:
        if not isinstance(config, ConfigSpec):
            return None
        if config.domain != self.domain:
            raise ValueError(f"ConfigSpec domain {config.domain!r} does not match engine domain {self.domain!r}")
        self._check_binding_contract(config)
        if isinstance(config, self.spec_type):
            return config
        return self.spec_type(
            config.domain,
            config.data,
            bindings=config.bindings,
            raw=config.raw,
            meta=config.meta,
        )

    def _check_binding_contract(self, spec: ConfigSpec) -> None:
        if not spec.bindings:
            return
        contract = spec.bindings.get("contract")
        if not isinstance(contract, str) or contract not in self.binding_contracts:
            choices = ", ".join(self.binding_contracts) or "<none>"
            raise ValueError(f"Unknown binding contract {contract!r} for {self.domain}; installed contracts: {choices}")

    def _schema_fields(self) -> dict[str, Any]:
        if self._schema is None:
            return {}
        return {name: item for name, item in self._schema.fields.items() if name not in self.schema_skip and not _is_builtin_identity_name(name, item)}

    def _schema_normalize(self, data: dict[str, Any]) -> dict[str, Any]:
        if self._schema is None:
            return data
        out = to_mutable_data(data)
        fields = self._schema_fields()
        for name, item in fields.items():
            if name not in out and item.has_default:
                out[name] = to_mutable_data(item.default)
        extras = {key: out.pop(key) for key in list(out) if key not in fields}
        if extras and self.extra == "forbid":
            raise ValueError(f"Unknown config keys for {self.domain}: {', '.join(sorted(extras))}")
        if extras and self.extra == "collect":
            bucket = out.get(self.extras_field) or {}
            if not isinstance(bucket, Mapping):
                raise TypeError(f"{self.domain}.{self.extras_field} must be a mapping when collecting extra keys")
            out[self.extras_field] = dmerge([to_mutable_data(bucket), extras])
        elif extras:
            out.update(extras)
        for name, item in fields.items():
            value = out.get(name, _MISSING)
            if value is _MISSING or value is None:
                if item.required:
                    raise ValueError(f"{self.domain}.{name} is required")
                continue
            try:
                out[name] = item.dtype.validate(value)
            except Exception as exc:
                raise ValueError(f"{self.domain}.{name}: {exc}") from exc
        return out

    def _schema_validate(self, data: Mapping[str, Any]) -> None:
        if self._schema is None:
            return
        fields = self._schema_fields()
        extras = sorted(key for key in data if key not in fields)
        if extras and self.extra == "forbid":
            raise ValueError(f"Unknown config keys for {self.domain}: {', '.join(extras)}")
        for name, item in fields.items():
            value = data.get(name, _MISSING)
            if value is _MISSING or value is None:
                if item.required:
                    raise ValueError(f"{self.domain}.{name} is required")
                continue
            try:
                item.dtype.validate(to_mutable_data(value))
            except Exception as exc:
                raise ValueError(f"{self.domain}.{name}: {exc}") from exc

    def _cache_bounds(self) -> tuple[int, int]:
        cache: Mapping[str, Any] = {}
        if self.manager is not None and hasattr(self.manager, "load_default"):
            cache = dget(self.manager.load_default(), "heavenbase.config.cache", default={}) or {}
        overlay_cache = self._root_overlay.get("cache", {})
        if isinstance(overlay_cache, Mapping) and any(key in overlay_cache for key in ("max_resolved", "max_applied")):
            cache = dmerge([to_mutable_data(cache), to_mutable_data(overlay_cache)])
        resolved = int(cache.get("max_resolved", 1024))
        applied = int(cache.get("max_applied", 1024))
        if resolved < 0:
            raise ValueError("heavenbase.config.cache.max_resolved must be >= 0")
        if applied < 0:
            raise ValueError("heavenbase.config.cache.max_applied must be >= 0")
        return resolved, applied

    def _resolution_revision(self) -> tuple[Any, ...]:
        return self.contract_revision, self.binding_revision, self.validator_revision

    def _spec_cache_key(self, spec: ConfigSpec) -> tuple[Any, ...] | object:
        fingerprint = spec._cache_identity
        if fingerprint is None:
            return _MISSING
        return "spec", fingerprint, self.validator_revision, self.binding_revision

    def _cache_get(self, cache: OrderedDict[Any, Any], key: Any) -> Any:
        if key is None:
            return None
        with self._lock:
            value = cache.get(key, _MISSING)
            if value is _MISSING:
                return None
            cache.move_to_end(key)
            return value

    def _cache_put(self, cache: OrderedDict[Any, Any], key: Any, value: Any, limit: int) -> None:
        if not limit:
            return
        with self._lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > limit:
                cache.popitem(last=False)


def _as_mapping(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return to_mutable_data(config)
    if hasattr(config, "to_dict"):
        value = config.to_dict()
        if isinstance(value, Mapping):
            return to_mutable_data(value)
    raise TypeError(f"Expected config mapping or object with to_dict(), got {type(config).__name__}")


def _identifier(value: Any, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Config {kind} must be a non-empty string")
    return value.strip().lower()


def _explicit_paths(data: Mapping[str, Any]) -> frozenset[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping) and value:
            for key, item in value.items():
                visit(item, (*path, str(key)))
            return
        out.add(path)

    visit(data, ())
    return frozenset(out)


def _fingerprint(value: Any) -> Any:
    if type(value) is float and value == 0.0:
        return "float", value.hex()
    if isinstance(value, _JSON_SCALARS):
        return type(value).__name__, value
    if isinstance(value, bytes):
        return "bytes", value
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise _Uncacheable(f"Config mapping key must be str, got {type(key).__name__}")
            items.append((key, _fingerprint(item)))
        return "mapping", tuple(sorted(items))
    if isinstance(value, (list, tuple)):
        return "sequence", tuple(_fingerprint(item) for item in value)
    raise _Uncacheable(f"Uncacheable config value {type(value).__name__}")


def _try_fingerprint(value: Any) -> Any:
    try:
        return _fingerprint(value)
    except _Uncacheable:
        return _MISSING


def _json_freeze(value: Any) -> Any:
    _fingerprint(value)
    return _freeze(value)


def _json_thaw(value: Any) -> Any:
    return to_mutable_data(value)


_JSON_CACHE_POLICY = _CachePolicy(_fingerprint, _json_freeze, _json_thaw)


def _profile_policy(value: Any) -> _CachePolicy | None:
    if value is False:
        return None
    if value is None:
        return _JSON_CACHE_POLICY
    fields = [getattr(value, name, None) for name in ("fingerprint", "freeze", "thaw")]
    if not all(callable(field) for field in fields):
        raise TypeError("Config profile cache policy requires fingerprint, freeze, and thaw callables")
    return _CachePolicy(*fields)


def _source_free_context() -> _ResolutionContext:
    return _ResolutionContext(MappingProxyType({}), MappingProxyType({}), None, ("spec",), frozenset())


def _is_builtin_identity_name(name: str, item: Any) -> bool:
    return name == "name" and not item.required and item.has_default and item.default == "" and item.desc == _DEFAULT_NAME_DESC


def _first_diff(left: Any, right: Any, path: tuple[str, ...] = ()) -> tuple[str, ...]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return (*path, str(key))
            found = _first_diff(left[key], right[key], (*path, str(key)))
            if found:
                return found
        return ()
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        for index in range(max(len(left), len(right))):
            if index >= len(left) or index >= len(right):
                return (*path, str(index))
            found = _first_diff(left[index], right[index], (*path, str(index)))
            if found:
                return found
        return ()
    return path if left != right else ()


def _format_path(path: tuple[str, ...]) -> str:
    return "" if not path else "." + ".".join(path)


def _dotted(path: tuple[str, ...]) -> str | None:
    return ".".join(path) if path else None


def _secret_digest(value: Any) -> str:
    fingerprint = _try_fingerprint(value)
    raw = repr(type(value).__name__ if fingerprint is _MISSING else fingerprint).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def redact_paths(data: Mapping[str, Any], paths: Sequence[str], replacement: str = "***") -> dict[str, Any]:
    """Return a copy with explicit exact or wildcard paths redacted.

    Args:
        data (Mapping[str, Any]): Nested mapping to copy and redact.
        paths (Sequence[str]): Dot paths; ``*`` matches one mapping or list level.
        replacement (str): Redaction placeholder. Defaults to ``"***"``.

    Returns:
        dict[str, Any]: Detached copy with selected paths redacted.
    """
    out = to_mutable_data(data)

    def redact(value: Any, parts: tuple[str, ...]) -> Any:
        if not parts:
            return replacement
        if isinstance(value, dict):
            head, tail = parts[0], parts[1:]
            for key in list(value):
                if head in {"*", key}:
                    value[key] = redact(value[key], tail)
        elif isinstance(value, list) and parts[0] == "*":
            value = [redact(item, parts[1:]) for item in value]
        return value

    for path in paths:
        out = redact(out, tuple(part for part in path.split(".") if part))
    return out


def redact_url(value: str, replacement: str = "***", query_names: Sequence[str] = ()) -> str:
    """Redact explicit URL secret locations while preserving route shape.

    Args:
        value (str): URL string to sanitize.
        replacement (str): Redaction placeholder. Defaults to ``"***"``.
        query_names (Sequence[str]): Query parameter names to redact.

    Returns:
        str: URL with selected credentials and query values redacted.
    """
    parsed = urlsplit(value)
    host = parsed.netloc.rpartition("@")[2]  # host and port exactly as written: IPv6 brackets and case survive
    netloc = f"{replacement}:{replacement}@{host}" if parsed.username is not None or parsed.password is not None else parsed.netloc
    selected = {name.lower() for name in query_names}
    query = urlencode(
        [(key, replacement if key.lower() in selected else item) for key, item in parse_qsl(parsed.query, keep_blank_values=True)],
        doseq=True,
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def redact_headers(headers: Mapping[str, Any], names: Sequence[str], replacement: str = "***") -> dict[str, Any]:
    """Redact explicitly named HTTP headers case-insensitively.

    Args:
        headers (Mapping[str, Any]): Header mapping to copy.
        names (Sequence[str]): Header names to redact.
        replacement (str): Redaction placeholder. Defaults to ``"***"``.

    Returns:
        dict[str, Any]: Detached header mapping with selected names redacted.
    """
    selected = {name.lower() for name in names}
    return {key: replacement if key.lower() in selected else to_mutable_data(value) for key, value in headers.items()}
