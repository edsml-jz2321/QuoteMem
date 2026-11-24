"""Context-owned lifecycle over the authoritative system module Registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
from functools import wraps
from hashlib import sha256
from importlib import import_module
import os
from pathlib import Path
import tempfile
import threading
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from heavenbase.utils.config.data import copy_data
from heavenbase.utils.registry import Registry, RegistryConflictError, RegistryView, identifier_key, named_key
from heavenbase.version import __version__

from .artifacts import ModuleArtifactStore, SourceUnavailable, forget_readings
from .discovery import discover_package_modules, forget_discoveries
from .executable import ExecutableRef
from .manifest import (
    InstallReceipt,
    ModuleGroup,
    ModuleKey,
    _check_compatibility,
    _installed_items,
    _manifest_uses_path_source,
    _PreparedInstallation,
    _prepare_manifest,
    _record_compatibility,
    _record_coordinate,
    _record_dependencies,
    _record_fingerprint,
    _record_generation,
    _record_publisher,
    _text,
)

if TYPE_CHECKING:
    from .tags import ResolvedTagValue, TagDefinition, TagView

__all__ = ["ModuleCatalog", "ModuleService"]

_PACKAGE = "heavenbase"
_PACKAGE_PUBLISHER = "heavenbase"
_PACKAGE_COORDINATE = "heavenbase/module-set"
_PACKAGE_SET_KEY: ModuleKey = ("module_set", "heavenbase")
_PACKAGE_GROUPS: tuple[ModuleGroup, ...] = (("provenance", "system"), ("tag", "builtin"))
# Prepared installations of captured package folders: (artifact digest, runtime version) -> installation.
# A capture's digest fixes its descriptor bytes, so the preparation is the same for every store.
_PREPARED_LIMIT = 512
_PREPARED_CAPTURES: dict[tuple[str, str], _PreparedInstallation] = {}
_PREPARED_LOCK = threading.Lock()


def _new_prepared_lock_in_child() -> None:
    """Give a forked child an unheld lock: the thread that held the parent's does not exist there."""
    global _PREPARED_LOCK
    _PREPARED_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_new_prepared_lock_in_child)


def _prepare_package_manifest(source: Mapping[str, Any] | Path, artifact_digest: str | None) -> _PreparedInstallation:
    """Prepare one packaged descriptor, reusing the preparation of a capture seen before."""
    if artifact_digest is None:
        return _prepare_manifest(source, runtime_version=__version__, trusted_groups=_PACKAGE_GROUPS)
    key = (artifact_digest, __version__)
    with _PREPARED_LOCK:
        prepared = _PREPARED_CAPTURES.get(key)
    if prepared is None:
        prepared = _prepare_manifest(source, artifact_digest=artifact_digest, runtime_version=__version__, trusted_groups=_PACKAGE_GROUPS)
        with _PREPARED_LOCK:
            _PREPARED_CAPTURES[key] = prepared
            while len(_PREPARED_CAPTURES) > _PREPARED_LIMIT:
                _PREPARED_CAPTURES.pop(next(iter(_PREPARED_CAPTURES)))
    return prepared


def _forget_package_readings() -> None:
    """Make the next package setup read, verify and prepare every module folder again."""
    forget_discoveries()
    forget_readings()
    with _PREPARED_LOCK:
        _PREPARED_CAPTURES.clear()


@dataclass(slots=True)
class _CatalogRows:
    """Hold lazily fingerprinted inspection rows for one Registry view.

    ``derived`` holds reader-owned values computed from this exact publication,
    such as the Tag interpretation capture; it dies with the publication, so a
    reader never has to invalidate it.
    """

    view: RegistryView[ModuleKey, dict[str, Any]]
    by_key: Mapping[ModuleKey, dict[str, Any]]
    fingerprinted: dict[ModuleKey, dict[str, Any]] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)

    def row(self, key: ModuleKey) -> dict[str, Any]:
        """Return one internally owned row, fingerprinting it on first use."""
        found = self.fingerprinted.get(key)
        if found is not None:
            return found
        base = self.by_key[key]
        found = {**base, "fingerprint": _record_fingerprint(base["record"])}
        self.fingerprinted[key] = found
        return found

    def all(self) -> tuple[dict[str, Any], ...]:
        """Return all rows in canonical Registry order."""
        return tuple(self.row(key) for key in self.view.items)


def _legacy_package_owned(record: Mapping[str, Any]) -> bool:
    """Return whether a pre-v2 installer envelope is package-owned rebuild state."""
    meta = record.get("meta")
    installation = meta.get("installation") if isinstance(meta, Mapping) else None
    return isinstance(installation, Mapping) and installation.get("owner") == "package:heavenbase"


def _authority_bound(method: Callable[..., Any]) -> Callable[..., Any]:
    """Keep one resolver operation inside its Context generation lease."""

    @wraps(method)
    def guarded(self: "ModuleService", *args: Any, **kwargs: Any) -> Any:
        with self._authority_guard():
            return method(self, *args, **kwargs)

    return guarded


def _catalog_authority_bound(method: Callable[..., Any]) -> Callable[..., Any]:
    """Keep one read-only catalog operation inside its authority lease."""

    @wraps(method)
    def guarded(self: "ModuleCatalog", *args: Any, **kwargs: Any) -> Any:
        with self._authority_guard():
            return method(self, *args, **kwargs)

    return guarded


class ModuleCatalog:
    """Expose a read-only view over one authoritative module Registry.

    Every inspection captures records, aliases, groups, revision, and payload
    fingerprint from one Registry state. The catalog intentionally exposes no
    mutation methods.
    """

    def __init__(
        self,
        registry: Registry[ModuleKey, dict[str, Any]],
        need_active: Callable[[], None],
        authority_guard: Callable[..., Any],
    ) -> None:
        """Capture one authoritative Registry for read-only catalog access.

        Args:
            registry (Registry): Authoritative ``system-modules`` Registry.
            need_active (Callable[[], None]): Guard that rejects retired service use.
            authority_guard (Callable[..., Any]): Context lifecycle guard context manager.

        Returns:
            None: This constructor does not return a value.
        """
        self._registry = registry
        self._need_active = need_active
        self._authority_guard = authority_guard
        self._cached_view: RegistryView[ModuleKey, dict[str, Any]] | None = None
        self._cached_rows: _CatalogRows | None = None

    @property
    @_catalog_authority_bound
    def revision(self) -> int:
        """Return the current Registry revision."""
        self._need_active()
        return self._registry.revision

    @property
    @_catalog_authority_bound
    def publication(self) -> int:
        """Return the process-local Registry publication sequence."""
        self._need_active()
        return self._registry.publication

    @property
    @_catalog_authority_bound
    def fingerprint(self) -> str | None:
        """Return the current durable payload fingerprint, when available."""
        return self._capture().fingerprint

    @property
    @_catalog_authority_bound
    def persistent(self) -> bool:
        """Return whether the underlying catalog has durable storage."""
        self._need_active()
        return self._registry.persistent

    def _capture(self) -> RegistryView[ModuleKey, dict[str, Any]]:
        """Capture and memoize one coherent view until service publication."""
        self._need_active()
        view = self._cached_view
        if view is None or view.publication != self._registry.publication:
            view = self._registry.view()
            self._cached_view = view
        return view

    def _capture_rows(self) -> _CatalogRows:
        """Derive inspection rows and record fingerprints once per publication."""
        view = self._capture()
        cached = self._cached_rows
        if cached is not None and cached.view.publication == view.publication:
            return cached
        groups_by_key: dict[ModuleKey, dict[str, list[str]]] = {}
        for normalized, members in view.groups.items():
            group, value = view.group_names[normalized]
            for key in members:
                groups_by_key.setdefault(key, {}).setdefault(str(group), []).append(str(value))
        by_key: dict[ModuleKey, dict[str, Any]] = {}
        for key, record in view.items.items():
            _record_coordinate(record)
            row = {
                "kind": key[0],
                "identifier": key[1],
                "aliases": [alias for _, alias in view.alias_names.get(key, ())],
                "groups": groups_by_key.get(key, {}),
                "record": record,
                "publication": view.publication,
                "revision": view.revision,
            }
            by_key[key] = row
        captured = _CatalogRows(view, MappingProxyType(by_key))
        self._cached_rows = captured
        return captured

    def _derived(self) -> tuple[int, dict[str, Any]]:
        """Return the current publication and the reader-owned values derived from exactly it.

        Values kept there die with the publication, so readers that derive
        them, such as handler seeding, never need to invalidate them.
        """
        captured = self._capture_rows()
        return captured.view.publication, captured.derived

    def _invalidate(self) -> None:
        self._cached_view = None
        self._cached_rows = None

    @staticmethod
    def _resolve(view: RegistryView[ModuleKey, dict[str, Any]], key: ModuleKey) -> ModuleKey:
        return view.aliases.get(key, key)

    @_catalog_authority_bound
    def resolve(self, key: ModuleKey) -> ModuleKey:
        """Resolve a canonical key or alias against one captured state.

        Args:
            key (ModuleKey): Canonical ``(kind, identifier)`` key or alias.

        Returns:
            ModuleKey: Canonical key for the captured Registry state.
        """
        view = self._capture()
        return self._resolve(view, key)

    @_catalog_authority_bound
    def get(self, key: ModuleKey, default: Any = None) -> Any:
        """Return one detached record from a captured state.

        Args:
            key (ModuleKey): Canonical key or alias.
            default (Any): Value returned when the key is absent. Defaults to
                ``None``.

        Returns:
            Any: Detached registration record, or ``default`` when absent.
        """
        view = self._capture()
        canonical = self._resolve(view, key)
        return copy_data(view.items[canonical]) if canonical in view.items else default

    @_catalog_authority_bound
    def __getitem__(self, key: ModuleKey) -> dict[str, Any]:
        """Return one detached record or raise `KeyError`."""
        view = self._capture()
        canonical = self._resolve(view, key)
        if canonical not in view.items:
            raise KeyError(key)
        return copy_data(view.items[canonical])

    @_catalog_authority_bound
    def list(self, group: ModuleGroup | None = None) -> tuple[ModuleKey, ...]:
        """List canonical keys, optionally filtered by one typed group.

        Args:
            group (ModuleGroup | None): Optional ``(group_kind, value)`` filter.
                Defaults to ``None`` for all canonical keys.

        Returns:
            tuple[ModuleKey, ...]: Matching canonical keys from the captured state.
        """
        view = self._capture()
        return tuple(view.items) if group is None else tuple(view.groups.get(group, ()))

    @_catalog_authority_bound
    def aliases(self, key: ModuleKey) -> tuple[ModuleKey, ...]:
        """Return declared display aliases for one key.

        Args:
            key (ModuleKey): Canonical key or alias.

        Returns:
            tuple[ModuleKey, ...]: Declared aliases for the resolved canonical key.
        """
        view = self._capture()
        canonical = self._resolve(view, key)
        return tuple(view.alias_names.get(canonical, ())) if canonical in view.items else ()

    @_catalog_authority_bound
    def groups(self, key: ModuleKey) -> tuple[ModuleGroup, ...]:
        """Return typed display groups containing one key.

        Args:
            key (ModuleKey): Canonical key or alias.

        Returns:
            tuple[ModuleGroup, ...]: Typed groups containing the resolved key.
        """
        view = self._capture()
        canonical = self._resolve(view, key)
        return tuple(view.group_names[group] for group, keys in view.groups.items() if canonical in keys)

    @_catalog_authority_bound
    def in_group(self, key: ModuleKey, group: ModuleGroup) -> bool:
        """Return whether one key belongs to a typed group.

        Args:
            key (ModuleKey): Canonical key or alias.
            group (ModuleGroup): Typed group to test membership against.

        Returns:
            bool: Whether the resolved key belongs to ``group``.
        """
        view = self._capture()
        canonical = self._resolve(view, key)
        return canonical in view.items and canonical in view.groups.get(group, ())

    @_catalog_authority_bound
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        view = self._capture()
        return self._resolve(view, key) in view.items

    @_catalog_authority_bound
    def __iter__(self):
        return iter(tuple(self._capture().items))

    @_catalog_authority_bound
    def inspect(
        self,
        kind: str | None = None,
        identifier: str | None = None,
        *,
        include_inactive: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Inspect safe module records from exactly one Registry snapshot.

        Args:
            kind (str | None): Optional registration kind filter.
            identifier (str | None): Optional canonical identifier or alias.
                Requires ``kind`` when supplied.
            include_inactive (bool): Whether to include retired records.
                Defaults to ``False``.

        Returns:
            tuple[dict[str, Any], ...]: Detached inspection rows from one
            captured Registry snapshot.

        Raises:
            ValueError: If ``identifier`` is supplied without ``kind``.
        """
        if identifier is not None and kind is None:
            raise ValueError("ModuleCatalog.inspect identifier requires kind")
        captured = self._capture_rows()
        view = captured.view
        if kind is None:
            keys = tuple(view.items)
        elif identifier is None:
            clean_kind = identifier_key(kind, "registration kind")
            keys = tuple(key for key in view.items if key[0] == clean_kind)
        else:
            lookup = (identifier_key(kind, "registration kind"), named_key(identifier, "registration identifier or alias"))
            canonical = self._resolve(view, lookup)
            keys = (canonical,) if canonical in view.items else ()
        rows: list[dict[str, Any]] = []
        for key in keys:
            row = captured.row(key)
            record = row["record"]
            if not include_inactive and not record.get("active", True):
                continue
            rows.append(copy_data(row))
        return tuple(rows)


class ModuleService:
    """Manage registration records and source artifacts for one module catalog.

    `ModuleService` is not a second catalog. It validates lifecycle requests,
    publishes records through the supplied Registry, captures local source in
    the supplied artifact store, and materializes implementations only in
    `resolve`.

    Args:
        registry: Authoritative Registry whose keys are `(kind, identifier)`.
        artifacts: Content-addressed local module store. Persistent Registries
            require an explicit durable store before installing path sources.
        authority_token: Stable owning Context identity used by resource caches.
        authority_config: Optional Context-owned configuration authority paired
            with this service for cross-domain runtime construction.
        source_restorers: Optional adapters for source kinds owned outside the
            module package. Each adapter receives the persisted target and this
            resolver. Adapters report only absent/retryable sources with
            `SourceUnavailable`; other failures remain fatal.

    Raises:
        ValueError: If `registry` is not the `system-modules` namespace.
    """

    def __init__(
        self,
        registry: Registry[ModuleKey, dict[str, Any]],
        *,
        artifacts: ModuleArtifactStore | None = None,
        authority_token: str | None = None,
        authority_config: Any = None,
        source_restorers: Mapping[str, Callable[[Any, "ModuleService"], Any]] | None = None,
    ) -> None:
        """Initialize module lifecycle over one authoritative Registry."""
        if registry.identifier != "system-modules":
            raise ValueError(f"ModuleService requires Registry 'system-modules', got {registry.identifier!r}")
        if registry.value_policy != "snapshot":
            raise ValueError("ModuleService requires Registry value_policy='snapshot' for failure-atomic lifecycle changes")
        self._registry = registry
        if authority_token is not None and (not isinstance(authority_token, str) or not authority_token):
            raise TypeError("ModuleService authority_token must be a non-empty string")
        self.authority_config = authority_config
        self.authority_token = authority_token or uuid4().hex
        self._artifacts_explicit = artifacts is not None
        self._artifacts = artifacts or ModuleArtifactStore(Path(tempfile.gettempdir()) / "heavenbase-module-artifacts")
        restorers = dict(source_restorers or {})
        invalid = [source for source, restorer in restorers.items() if not isinstance(source, str) or not source or not callable(restorer)]
        if invalid:
            raise TypeError("Module source restorers require non-empty string keys and callable values")
        reserved = set(restorers).intersection({"inline", "python-import", "artifact"})
        if reserved:
            raise ValueError(f"Module source restorers cannot replace built-in sources {sorted(reserved)!r}")
        self._source_restorers = restorers
        self._authority_lock = threading.RLock()
        self._authority_condition = threading.Condition(self._authority_lock)
        self._authority_users = 0
        self._authority_threads: dict[int, int] = {}
        self._active = True
        self.catalog = ModuleCatalog(registry, self._need_active, self._authority_guard)
        self._cache: dict[tuple[Any, ...], Any] = {}
        self._cache_publication = registry.publication
        self._lock = threading.RLock()

    def _need_active(self) -> None:
        """Reject use after the owning Context generation is retired."""
        with self._authority_lock:
            self._need_active_unlocked()

    def _need_active_unlocked(self) -> None:
        """Validate this service and its optional Context lease under lock."""
        if self._authority_threads.get(threading.get_ident(), 0):
            return
        context = getattr(self.authority_config, "_context", None)
        context_token = getattr(context, "authority_token", None)
        if (
            not self._active
            or bool(getattr(context, "_closed", False))
            or bool(getattr(context, "_closing", False))
            or bool(getattr(context, "_resetting", False))
            or (context_token is not None and str(context_token) != self.authority_token)
        ):
            raise RuntimeError("ModuleService belongs to a retired Context authority generation")

    @contextmanager
    def _authority_guard(self):
        """Prevent Context retirement while one resolver operation publishes."""
        with self._authority_condition:
            self._need_active_unlocked()
            thread_id = threading.get_ident()
            self._authority_users += 1
            self._authority_threads[thread_id] = self._authority_threads.get(thread_id, 0) + 1
        try:
            yield
        finally:
            with self._authority_condition:
                self._authority_users -= 1
                remaining = self._authority_threads[thread_id] - 1
                if remaining:
                    self._authority_threads[thread_id] = remaining
                else:
                    self._authority_threads.pop(thread_id)
                if self._authority_users == 0:
                    self._authority_condition.notify_all()

    def _retire_authority(self) -> None:
        """Permanently retire this Context generation without deleting data."""
        with self._authority_condition:
            if not self._active:
                return
            self._need_retirable()
            self._active = False
            while self._authority_users:
                self._authority_condition.wait()
        self._invalidate_cache()
        self._artifacts.release_all(lifecycle_owner=self)

    def _need_retirable(self) -> None:
        """Reject self-retirement that would wait for the current call."""
        with self._authority_condition:
            if self._authority_threads.get(threading.get_ident(), 0):
                raise RuntimeError("Cannot retire ModuleService from one of its active resolver operations")

    @_authority_bound
    def register(
        self,
        kind: str,
        identifier: str,
        record: Mapping[str, Any],
        aliases: Iterable[str] = (),
        groups: Iterable[ModuleGroup] = (),
        *,
        coordinate: str | None = None,
        publisher: str | None = None,
        version: str | None = None,
        compatibility: Mapping[str, Any] | None = None,
    ) -> InstallReceipt:
        """Validate and atomically reconcile one programmatic registration.

        Args:
            kind: Registration kind such as `operation` or `backend_type`.
            identifier: Canonical registration identifier.
            record: Plain MetaSchema-shaped registration value.
            aliases: Alternate identifiers within the same kind.
            groups: `(group_kind, value)` Registry groups.
            coordinate: Stable install-set coordinate. A key-specific
                programmatic coordinate is used when omitted.
            publisher: Installer-assigned publisher claim.
            version: Module release version. Defaults to record `meta.version`.
            compatibility: Supported HeavenBase version interval.

        Returns:
            InstallReceipt: Coordinate, generation, keys, revision, and
                canonical fingerprint.

        Raises:
            TypeError: If the record or index metadata has an invalid shape.
            ValueError: If validation, ownership, dependencies, or indexes fail.
        """
        clean_kind = identifier_key(kind, "registration kind")
        clean_identifier = identifier_key(identifier, f"{clean_kind} identifier")
        body = copy.deepcopy(dict(record))
        forbidden = set(body).intersection({"kind", "identifier", "aliases", "groups", "coordinate", "publisher", "generation"})
        if forbidden:
            raise ValueError(f"Programmatic registration record contains installer envelope fields {sorted(forbidden)!r}")
        meta = body.get("meta")
        inferred_version = meta.get("version") if isinstance(meta, Mapping) else None
        release = version if version is not None else inferred_version
        if release is None:
            release = __version__
        group_map: dict[str, list[str]] = {}
        for group_kind, value in groups:
            group_map.setdefault(str(group_kind), []).append(str(value))
        source = {
            "manifest_version": 2,
            "coordinate": coordinate or f"programmatic/{clean_kind}/{clean_identifier}",
            "version": str(release),
            "compatibility": copy.deepcopy(dict(compatibility or {})),
            "items": [
                {
                    **body,
                    "kind": clean_kind,
                    "identifier": clean_identifier,
                    "aliases": list(aliases),
                    "groups": group_map,
                }
            ],
        }
        return self._publish(_prepare_manifest(source, runtime_version=__version__), publisher=publisher, reconcile=True)

    @_authority_bound
    def install(
        self,
        source: Mapping[str, Any] | str | Path,
        *,
        publisher: str | None = None,
        expected_coordinate: str | None = None,
    ) -> InstallReceipt:
        """Parse and atomically install one manifest mapping, file, or folder.

        Parsing and validation do not import implementation modules. A path
        source must be a folder containing `meta.yaml` or that exact file.

        Args:
            source: Manifest mapping, manifest path, or co-located folder.
            publisher: Installer-assigned publisher claim.
            expected_coordinate: Optional acquisition-level coordinate check.

        Returns:
            InstallReceipt: Atomic installation identity and published revision.

        Raises:
            TypeError: If the manifest is not plain JSON-compatible data.
            ValueError: If validation, ownership, dependencies, or indexes fail.
            RegistryConflictError: If another writer publishes first; callers
                must refresh and retry explicitly.
        """
        artifact = None
        prepared_source: Mapping[str, Any] | str | Path = source
        if not isinstance(source, Mapping):
            if self._registry.persistent and not self._artifacts_explicit:
                raise ValueError("Persistent ModuleService path installation requires an explicit durable ModuleArtifactStore")
            path = Path(source).expanduser()
            if path.is_file() and path.name != "meta.yaml":
                raise ValueError(f"Registration descriptor must be named 'meta.yaml', got: {path}")
            module_root = path if path.is_dir() else path.parent
            artifact = self._artifacts.capture(module_root)
            prepared_source = artifact.root
        prepared = _prepare_manifest(
            prepared_source,
            artifact_digest=artifact.digest if artifact is not None else None,
            expected_coordinate=expected_coordinate,
            runtime_version=__version__,
        )
        return self._publish(prepared, publisher=publisher, reconcile=True)

    def _publish(
        self,
        installation: _PreparedInstallation,
        *,
        publisher: str | None,
        reconcile: bool = False,
        package: bool = False,
    ) -> InstallReceipt:
        clean_publisher = None if publisher is None else _text(publisher, "module publisher")
        if not package:
            if clean_publisher == _PACKAGE_PUBLISHER or installation.coordinate.startswith(f"{_PACKAGE}/"):
                raise ValueError("HeavenBase module coordinates and publisher are reserved for built-in package setup")
            if _PACKAGE_SET_KEY in installation.keys or any(_PACKAGE_SET_KEY in item.aliases for item in installation.items):
                raise ValueError(f"Registration key {_PACKAGE_SET_KEY!r} is reserved for built-in package setup")
            claimed = set(_PACKAGE_GROUPS).intersection(group for item in installation.items for group in item.groups)
            if claimed:
                raise ValueError(f"Registration groups {sorted(claimed)!r} are reserved for built-in package setup")

        def change(draft: Registry[ModuleKey, dict[str, Any]]) -> str:
            generation = uuid4().hex
            self._reconcile_installation(
                draft,
                installation,
                generation=generation,
                publisher=clean_publisher,
                reconcile=reconcile,
            )
            self._check_dependencies(draft)
            return generation

        generation = self._registry.mutate(change)
        self._invalidate_cache()
        return InstallReceipt(
            installation.coordinate,
            generation,
            installation.keys,
            installation.bundles,
            self._registry.revision,
            installation.fingerprint,
        )

    @staticmethod
    def _reconcile_installation(
        draft: Registry[ModuleKey, dict[str, Any]],
        installation: _PreparedInstallation,
        *,
        generation: str,
        publisher: str | None,
        reconcile: bool,
    ) -> None:
        current = tuple((key, record) for key, record in draft.items() if _record_coordinate(record) == installation.coordinate)
        current_publishers = {_record_publisher(record) for _, record in current}
        current_generations = {_record_generation(record) for _, record in current}
        if len(current_publishers) > 1 or len(current_generations) > 1:
            raise ValueError(f"Module coordinate {installation.coordinate!r} has inconsistent installation identity")
        if current_publishers and current_publishers != {publisher}:
            existing = next(iter(current_publishers))
            raise ValueError(f"Module coordinate {installation.coordinate!r} is published by {existing!r}, not {publisher!r}")
        wanted = set(installation.keys)
        if reconcile:
            for key, _ in current:
                if key not in wanted:
                    draft.delete(key)
        for item in _installed_items(installation, generation=generation, publisher=publisher):
            existing = draft.get(item.key)
            if existing is not None:
                existing_coordinate = _record_coordinate(existing)
                if existing_coordinate != installation.coordinate:
                    raise ValueError(f"Registration {item.key!r} belongs to module {existing_coordinate!r}, not {installation.coordinate!r}")
                draft.update(item.key, item.record, aliases=item.aliases, groups=item.groups)
            else:
                draft.create(item.key, item.record, aliases=item.aliases, groups=item.groups)

    @staticmethod
    def _check_dependencies(registry: Registry[ModuleKey, dict[str, Any]]) -> None:
        for key, record in registry.items():
            if not record.get("active", True):
                continue
            for dependency, optional in _record_dependencies(record):
                found = registry.get(dependency)
                if found is None:
                    if optional:
                        continue
                    raise ValueError(f"Registration {key!r} requires missing dependency {dependency!r}")
                if not found.get("active", True) and not optional:
                    raise ValueError(f"Registration {key!r} requires inactive dependency {dependency!r}")

    @_authority_bound
    def resolve(self, kind: str, identifier: str) -> Any:
        """Restore one active registration through its persisted source layers.

        Args:
            kind: Registration kind.
            identifier: Canonical identifier or alias within `kind`.

        Returns:
            Any: Imported object or detached inline definition.

        Raises:
            KeyError: If the registration is absent or retired.
            ImportError: If no registered source layer can be restored.
        """
        key = (identifier_key(kind, "registration kind"), named_key(identifier, "registration identifier or alias"))
        while True:
            view = self.catalog._capture()
            canonical = ModuleCatalog._resolve(view, key)
            record = view.items.get(canonical)
            if record is None:
                raise KeyError(f"Unknown system module {key!r}")
            if not record.get("active", True):
                raise KeyError(f"System module {canonical!r} is retired")
            _check_compatibility(_record_compatibility(record), __version__, f"Module {_record_coordinate(record)!r} compatibility")
            fingerprint = _record_fingerprint(record)
            generation = _record_generation(record)
            cache_key = (id(self._registry), view.publication, view.revision, canonical, generation, fingerprint)
            publication_changed = False
            with self._lock:
                if self._cache_publication != view.publication:
                    self._cache.clear()
                    self._cache_publication = view.publication
                    publication_changed = True
                cached = cache_key in self._cache
                cached_value = self._cache.get(cache_key)
            if publication_changed:
                self._release_unreferenced_artifacts(view.items.values())
            if cached:
                return cached_value

            layers = ({"source": record["source"], "target": record["target"]}, *record["meta"].get("restore_layers", ()))
            errors: list[str] = []
            for layer in layers:
                try:
                    value = self._restore(record, layer)
                except SourceUnavailable as error:
                    errors.append(f"{layer.get('source')}: {error}")
                    continue
                break
            else:
                raise ImportError(f"Could not restore system module {canonical!r}: {'; '.join(errors)}")

            with self._lock:
                if self._registry.publication != view.publication:
                    continue
                self._cache[cache_key] = value
                return value

    def _restore(self, record: Mapping[str, Any], layer: Mapping[str, Any]) -> Any:
        source = layer["source"]
        target = layer["target"]
        if source == "inline":
            return copy.deepcopy(record["meta"][target])
        if source in {"python-import", "artifact", "capsule"}:
            return self._restore_executable_ref(
                ExecutableRef(source, target),
                owner=record,
                require_callable=False,
            )
        restorer = self._source_restorers.get(str(source))
        if restorer is not None:
            return restorer(copy.deepcopy(target), self)
        raise ImportError(f"unsupported source {source!r}")

    @_authority_bound
    def resolve_executable(
        self,
        reference: Mapping[str, Any],
        *,
        owner: Mapping[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Restore one canonical executable reference through this authority.

        Args:
            reference: Versioned executable-reference envelope.
            owner: Owning module record or inspection row. Artifact references
                require its installation coordinate and generation.

        Returns:
            Callable[..., Any]: Restored callable, class, or callable Capsule.

        Raises:
            ImportError: If the selected source cannot be restored.
            TypeError: If the restored value is not callable.
            ValueError: If an artifact reference lacks valid owner identity.
        """
        value = self._restore_executable_ref(
            ExecutableRef.from_dict(reference),
            owner=owner,
            require_callable=True,
        )
        return value

    def _restore_executable_ref(
        self,
        reference: ExecutableRef,
        *,
        owner: Mapping[str, Any] | None,
        require_callable: bool,
    ) -> Any:
        """Restore one parsed reference without entering another authority lease."""
        if reference.source == "python-import":
            module_name, qualname = str(reference.target).split(":", 1)
            try:
                module = import_module(module_name)
            except ModuleNotFoundError as error:
                if error.name == module_name or (isinstance(error.name, str) and module_name.startswith(f"{error.name}.")):
                    raise SourceUnavailable(f"Python module {module_name!r} is unavailable") from error
                raise
            value = ModuleService._attribute(module, qualname)
        elif reference.source == "artifact":
            record = owner.get("record") if isinstance(owner, Mapping) and isinstance(owner.get("record"), Mapping) else owner
            if not isinstance(record, Mapping):
                raise ValueError("Artifact executable references require an owning module record")
            coordinate = _record_coordinate(record)
            generation = _record_generation(record)
            if coordinate is None or generation is None:
                raise ValueError("Artifact executable reference owner lacks installation identity")
            target = reference.target
            artifact = self._artifacts.get(str(target["artifact"]))
            value = self._artifacts.resolve(
                artifact,
                target.get("module"),
                str(target["qualname"]),
                coordinate=coordinate,
                generation=generation,
                lifecycle_owner=self,
            )
        else:
            restorer = self._source_restorers.get("capsule")
            if restorer is None:
                raise ImportError("unsupported executable reference source 'capsule'")
            value = restorer(copy.deepcopy(reference.target), self)
        if require_callable and not callable(value):
            raise TypeError(f"Executable reference restored non-callable {type(value).__name__}")
        return value

    @staticmethod
    def _attribute(root: Any, qualname: str) -> Any:
        value = root
        try:
            for part in qualname.split("."):
                value = getattr(value, part)
        except AttributeError as error:
            raise ImportError(f"import target {qualname!r} does not exist") from error
        return value

    @_authority_bound
    def inspect(
        self,
        kind: str | None = None,
        identifier: str | None = None,
        *,
        include_inactive: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Return safe records through the read-only catalog.

        Args:
            kind: Optional registration kind filter.
            identifier: Optional canonical identifier or alias. Requires `kind`.
            include_inactive: Whether to include retired records.

        Returns:
            tuple[dict[str, Any], ...]: Detached inspection rows containing the
            key, aliases, typed groups, record, revision, and fingerprint.

        Raises:
            ValueError: If `identifier` is supplied without `kind`.
        """
        return self.catalog.inspect(kind, identifier, include_inactive=include_inactive)

    @_authority_bound
    def tag_definition(self, identifier: str) -> "TagDefinition | None":
        """Inspect one optional Tag definition without importing its target.

        Args:
            identifier (str): Canonical Tag identifier or Registry alias.

        Returns:
            TagDefinition | None: Snapshot-bound valid or locally invalid
            definition, or `None` when no active definition exists.

        Raises:
            TypeError: If a stored definition record is malformed.
            ValueError: If the identifier or Registry indexes are invalid.
        """
        from .tags import tag_definition

        return tag_definition(self, identifier)

    @_authority_bound
    def tags(self, kind: str, identifier: str) -> "TagView":
        """Inspect raw claims and typed defaults for one module record.

        Args:
            kind (str): Subject module-record kind.
            identifier (str): Canonical subject identifier or Registry alias.

        Returns:
            TagView: Immutable raw and typed view from one Registry snapshot.

        Raises:
            KeyError: If the active subject is absent.
            TypeError: If the subject's raw Tag mapping is malformed.
            ValueError: If identifiers, definitions, or indexes are invalid.
        """
        from .tags import tags

        return tags(self, kind, identifier)

    @_authority_bound
    def tag(
        self,
        kind: str,
        identifier: str,
        tag_id: str,
    ) -> "ResolvedTagValue | None":
        """Inspect one static claim or typed default for a module record.

        Args:
            kind (str): Subject module-record kind.
            identifier (str): Canonical subject identifier or Registry alias.
            tag_id (str): Canonical Tag identifier, definition alias, or raw
                unknown claim identifier.

        Returns:
            ResolvedTagValue | None: Snapshot-bound result, or `None` when the
            subject has neither an explicit claim nor an applicable default.

        Raises:
            KeyError: If the active subject is absent.
            TypeError: If the subject's raw Tag mapping is malformed.
            ValueError: If identifiers, definitions, or indexes are invalid.
        """
        from .tags import tag

        return tag(self, kind, identifier, tag_id)

    @_authority_bound
    def refresh(self) -> bool:
        """Pull a newer durable Registry revision and clear materialized values.

        Args:
            None.

        Returns:
            bool: Whether a newer snapshot was published locally.
        """
        changed = self._registry.refresh()
        if changed:
            self._invalidate_cache()
        return changed

    @_authority_bound
    def retire(self, kind: str, identifier: str) -> dict[str, Any]:
        """Retain one registration while marking it inactive atomically.

        Args:
            kind: Registration kind.
            identifier: Canonical identifier or alias.

        Returns:
            dict[str, Any]: Detached retired registration record.

        Raises:
            KeyError: If the registration is absent.
            ValueError: If another active registration depends on it.
        """
        retired: dict[str, Any] = {}

        def change(draft: Registry[ModuleKey, dict[str, Any]]) -> None:
            key = self._canonical_in(draft, kind, identifier)
            record = draft[key]
            self._guard_external_lifecycle(record, "retire")
            record["active"] = False
            record["meta"]["status"] = "retired"
            draft.update(key, record)
            self._check_dependencies(draft)
            retired.update(record)

        self._registry.mutate(change)
        self._invalidate_cache()
        return copy.deepcopy(retired)

    @_authority_bound
    def unregister(self, kind: str, identifier: str) -> dict[str, Any]:
        """Remove one registration after reverse-dependency checks.

        Args:
            kind: Registration kind.
            identifier: Canonical identifier or alias.

        Returns:
            dict[str, Any]: Detached removed registration record.

        Raises:
            KeyError: If the registration is absent.
            ValueError: If another registration requires it.
        """
        removed: dict[str, Any] = {}

        def change(draft: Registry[ModuleKey, dict[str, Any]]) -> None:
            key = self._canonical_in(draft, kind, identifier)
            self._guard_external_lifecycle(draft[key], "unregister")
            removed.update(copy.deepcopy(draft.delete(key)))
            self._check_dependencies(draft)

        self._registry.mutate(change)
        self._invalidate_cache()
        return removed

    @_authority_bound
    def uninstall(self, receipt: InstallReceipt) -> tuple[dict[str, Any], ...]:
        """Atomically remove exactly one receipt generation.

        Args:
            receipt: Prior installation receipt.

        Returns:
            tuple[dict[str, Any], ...]: Detached removed records in Registry order.

        Raises:
            KeyError: If a requested key or bundle is absent.
            ValueError: If ownership changed or remaining records depend on the
                requested installation.
        """
        if not isinstance(receipt, InstallReceipt):
            raise TypeError("ModuleService.uninstall expects an InstallReceipt; use uninstall_bundle for bundle removal")
        removed: list[dict[str, Any]] = []

        def change(draft: Registry[ModuleKey, dict[str, Any]]) -> None:
            current = tuple(
                key for key, record in draft.items() if _record_coordinate(record) == receipt.coordinate and _record_generation(record) == receipt.generation
            )
            if set(current) != set(receipt.keys) or len(current) != len(receipt.keys):
                raise ValueError(f"Installation receipt for {receipt.coordinate!r} is stale")
            for key in receipt.keys:
                record = draft.get(key)
                if record is None or _record_coordinate(record) != receipt.coordinate or _record_generation(record) != receipt.generation:
                    raise ValueError(f"Installation receipt for {receipt.coordinate!r} is stale")
                self._guard_external_lifecycle(record, "uninstall")
            removed.extend(copy.deepcopy(draft.delete(key)) for key in receipt.keys)
            self._check_dependencies(draft)

        self._registry.mutate(change)
        self._invalidate_cache()
        return tuple(removed)

    @_authority_bound
    def uninstall_bundle(self, bundle: str) -> tuple[dict[str, Any], ...]:
        """Atomically select and remove one complete bundled installation.

        Args:
            bundle (str): Registration bundle identifier.

        Returns:
            tuple[dict[str, Any], ...]: Detached removed records in Registry order.

        Raises:
            KeyError: If the bundle is unknown.
            ValueError: If the bundle spans multiple installations or is incomplete.
        """
        clean_bundle = identifier_key(bundle, "registration bundle")
        removed: list[dict[str, Any]] = []

        def change(draft: Registry[ModuleKey, dict[str, Any]]) -> None:
            keys = draft.list(("bundle", clean_bundle))
            if not keys:
                raise KeyError(f"Unknown registration bundle {clean_bundle!r}")
            identities = {(_record_coordinate(draft[key]), _record_generation(draft[key])) for key in keys}
            if len(identities) != 1:
                raise ValueError(f"Registration bundle {clean_bundle!r} spans multiple installations")
            coordinate, generation = identities.pop()
            for key in keys:
                self._guard_external_lifecycle(draft[key], "uninstall bundle")
            all_keys = tuple(key for key, record in draft.items() if _record_coordinate(record) == coordinate and _record_generation(record) == generation)
            if set(keys) != set(all_keys):
                raise ValueError(f"Registration bundle {clean_bundle!r} does not identify a complete installation")
            removed.extend(copy.deepcopy(draft.delete(key)) for key in all_keys)
            self._check_dependencies(draft)

        self._registry.mutate(change)
        self._invalidate_cache()
        return tuple(removed)

    @staticmethod
    def _canonical_in(registry: Registry[ModuleKey, dict[str, Any]], kind: str, identifier: str) -> ModuleKey:
        lookup = (identifier_key(kind, "registration kind"), named_key(identifier, "registration identifier or alias"))
        canonical = registry.resolve(lookup)
        if canonical not in registry:
            raise KeyError(f"Unknown system module {lookup!r}")
        return canonical

    @staticmethod
    def _guard_external_lifecycle(record: Mapping[str, Any], action: str) -> None:
        """Reject public lifecycle mutation of package-owned records."""
        if _record_publisher(record) == _PACKAGE_PUBLISHER or str(_record_coordinate(record) or "").startswith(f"{_PACKAGE}/"):
            raise ValueError(f"Cannot {action} a built-in package record; use ModuleService.setup(force=True) for trusted repair")

    def _invalidate_cache(self) -> None:
        view = self._registry.view()
        self.catalog._invalidate()
        with self._lock:
            self._cache.clear()
            self._cache_publication = view.publication
        self._release_unreferenced_artifacts(view.items.values())

    def _release_unreferenced_artifacts(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Release loaded generations absent from one authoritative snapshot."""
        identities: set[tuple[str, str, str]] = set()
        for record in records:
            if not record.get("active", True):
                continue
            coordinate = _record_coordinate(record)
            generation = _record_generation(record)
            if coordinate is None or generation is None:
                continue
            pending: list[Any] = [record]
            while pending:
                value = pending.pop()
                if isinstance(value, Mapping):
                    target = value.get("target")
                    if value.get("source") == "artifact" and isinstance(target, Mapping):
                        digest = target.get("artifact")
                        if isinstance(digest, str):
                            identities.add((coordinate, generation, digest))
                    pending.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    pending.extend(value)
        self._artifacts.release_unreferenced(identities, lifecycle_owner=self)

    @_authority_bound
    def setup(self, force: bool = False) -> bool:
        """Discover and atomically reconcile the packaged module-folder set.

        A durable composition marker avoids publication after the exact
        installed resource set has been indexed. `force=True` performs
        explicit reconciliation even when that marker is current. Discovery
        reads only `resources/modules.yaml` and exact-name `meta.yaml` files
        and does not import registered implementation modules. The process
        keeps its readings of the package folders and reuses them while the
        folders' stat fingerprints are unchanged; the marker is checked
        against this Registry on every call.

        Args:
            force: Whether to rescan and reconcile the current packaged module
                set even when its composition marker is current. A forced
                setup reads every package folder again.

        Returns:
            bool: Whether discovery and validation ran.

        Raises:
            TypeError: If `force` or discovered metadata has an invalid shape.
            ValueError: If discovery, metadata, ownership, or dependencies fail.
            RuntimeError: If the package has no discoverable module folders.
        """
        if not isinstance(force, bool):
            raise TypeError("ModuleService.setup force must be boolean")
        if force:
            _forget_package_readings()
        else:
            self.refresh()

        installations: list[_PreparedInstallation] = []
        for discovered in discover_package_modules(_PACKAGE):
            source: Mapping[str, Any] | str | Path = discovered.manifest
            artifact_digest = None
            try:
                if _manifest_uses_path_source(discovered.manifest):
                    if self._registry.persistent and not self._artifacts_explicit:
                        raise ValueError("Persistent ModuleService package setup with path sources requires an explicit durable ModuleArtifactStore")
                    artifact = self._artifacts.capture_resource(discovered.root)
                    source = artifact.root
                    artifact_digest = artifact.digest
                installation = _prepare_package_manifest(source, artifact_digest)
            except (TypeError, ValueError) as error:
                raise type(error)(f"Invalid packaged module metadata {discovered.name}: {error}") from error
            if installation.coordinate == _PACKAGE_COORDINATE:
                raise ValueError(f"Packaged module metadata {discovered.name} uses reserved coordinate {_PACKAGE_COORDINATE!r}")
            installations.append(installation)
        coordinates = tuple(installation.coordinate for installation in installations)
        if len(set(coordinates)) != len(coordinates):
            raise ValueError(f"Packaged module descriptors repeat coordinates {coordinates!r}")
        composition = self._package_composition(installations, __version__)
        if not force and self._package_is_current(composition):
            return False
        marker = _prepare_manifest(
            self._package_set_source(coordinates, composition),
            runtime_version=__version__,
            trusted_groups=_PACKAGE_GROUPS,
        )
        while True:
            try:
                self._publish_package_batch((*installations, marker), coordinates)
                return True
            except RegistryConflictError:
                if not self._registry.refresh():
                    raise
                if not force and self._package_is_current(composition):
                    return False

    def _publish_package_batch(
        self,
        installations: tuple[_PreparedInstallation, ...],
        coordinates: tuple[str, ...],
    ) -> None:
        """Publish every built-in coordinate and its inventory marker once."""

        def change(draft: Registry[ModuleKey, dict[str, Any]]) -> None:
            previous: tuple[str, ...] = ()
            marker = draft.get(_PACKAGE_SET_KEY)
            if marker is not None:
                if _legacy_package_owned(marker):
                    for key, record in tuple(draft.items()):
                        if _legacy_package_owned(record):
                            draft.delete(key)
                    marker = None
                elif _record_coordinate(marker) != _PACKAGE_COORDINATE or _record_publisher(marker) != _PACKAGE_PUBLISHER:
                    raise ValueError("The built-in module-set marker is not owned by the HeavenBase package installation")
            if marker is not None:
                meta = marker.get("meta")
                definition = meta.get("definition") if isinstance(meta, Mapping) else None
                raw_coordinates = definition.get("coordinates", ()) if isinstance(definition, Mapping) else ()
                if not isinstance(raw_coordinates, list) or not all(isinstance(value, str) for value in raw_coordinates):
                    raise ValueError("The built-in module-set coordinate inventory is invalid")
                previous = tuple(raw_coordinates)

            incoming = set(coordinates)
            removed = set(previous) - incoming
            for key, record in tuple(draft.items()):
                if _record_coordinate(record) in removed:
                    if _record_publisher(record) != _PACKAGE_PUBLISHER:
                        raise ValueError(f"Removed built-in module coordinate {_record_coordinate(record)!r} changed publisher")
                    draft.delete(key)

            for installation in installations:
                self._reconcile_installation(
                    draft,
                    installation,
                    generation=uuid4().hex,
                    publisher=_PACKAGE_PUBLISHER,
                    reconcile=True,
                )
            self._check_dependencies(draft)

        self._registry.mutate(change)
        self._invalidate_cache()

    def _package_is_current(self, composition: str) -> bool:
        """Return whether the exact packaged module composition is indexed."""
        record = self.catalog.get(_PACKAGE_SET_KEY)
        meta = record.get("meta") if isinstance(record, Mapping) else None
        definition = meta.get("definition") if isinstance(meta, Mapping) else None
        return isinstance(definition, Mapping) and definition.get("composition") == composition

    @staticmethod
    def _package_composition(installations: Iterable[_PreparedInstallation], package_version: str) -> str:
        """Hash the opaque package identity and exact built-in descriptors."""
        digest = sha256()
        digest.update(b"heavenbase-module-set-v2\0")
        version = package_version.encode("utf-8")
        digest.update(len(version).to_bytes(8, "big"))
        digest.update(version)
        for installation in installations:
            for value in (installation.coordinate, installation.fingerprint):
                payload = value.encode("utf-8")
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
        return digest.hexdigest()

    @staticmethod
    def _package_set_source(coordinates: tuple[str, ...], composition: str) -> dict[str, Any]:
        return {
            "manifest_version": 2,
            "coordinate": _PACKAGE_COORDINATE,
            "version": __version__,
            "compatibility": {},
            "items": [
                {
                    "kind": _PACKAGE_SET_KEY[0],
                    "identifier": _PACKAGE_SET_KEY[1],
                    "name": "HeavenBase built-in modules",
                    "desc": "Durable receipt for the indexed built-in module-folder set.",
                    "source": "inline",
                    "target": "definition",
                    "active": True,
                    "meta": {
                        "schema_version": 1,
                        "definition": {
                            "package": _PACKAGE,
                            "version": __version__,
                            "descriptor": "meta.yaml",
                            "coordinates": list(coordinates),
                            "composition": composition,
                        },
                    },
                }
            ],
        }
