"""Implement the lock-free copy-on-write Registry."""

from __future__ import annotations

import copy
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
import threading
from typing import Any, Generic, Literal, TypeVar, cast, overload

from .codec import _RegistryCodec
from .errors import RegistryConflictError, RegistryIntegrityError
from .identity import identifier_key
from .store import RegistryStore

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
D = TypeVar("D")
R = TypeVar("R")
RegistryValuePolicy = Literal["snapshot", "process_identity"]


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


@dataclass(frozen=True)
class _State(Generic[K, V]):
    items: dict[K, V]
    aliases: dict[K, K]
    alias_names: dict[K, tuple[K, ...]]
    groups: dict[Hashable, tuple[K, ...]]
    group_names: dict[Hashable, Hashable]
    revision: int = 0
    fingerprint: str | None = None
    publication: int = 0


@dataclass(frozen=True)
class RegistryView(Generic[K, V]):
    """Expose one coherent, read-only Registry state.

    The mappings cannot be changed structurally. Under the default `snapshot`
    value policy, item values are also detached copies, so callers may mutate
    their local values without changing the Registry. A `process_identity`
    Registry deliberately exposes exact process objects instead.

    Args:
        items (Mapping[K, V]): Canonical keys mapped to captured values.
        aliases (Mapping[K, K]): Normalized aliases mapped to canonical keys.
        alias_names (Mapping[K, tuple[K, ...]]): Canonical keys mapped to
            display aliases in declaration order.
        groups (Mapping[Hashable, tuple[K, ...]]): Normalized groups mapped to
            canonical keys in registration order.
        group_names (Mapping[Hashable, Hashable]): Normalized groups mapped to
            their display spelling.
        revision (int): Persistent store revision, or zero for a process-only
            Registry.
        fingerprint (str | None): Integrity fingerprint of the persistent
            canonical payload, or `None` for process-only Registries and stores
            opened without hashing.
        publication (int): Process-local publication sequence for this Registry
            instance. Unlike durable revision, this also advances for
            process-only Registries.
    """

    items: Mapping[K, V]
    aliases: Mapping[K, K]
    alias_names: Mapping[K, tuple[K, ...]]
    groups: Mapping[Hashable, tuple[K, ...]]
    group_names: Mapping[Hashable, Hashable]
    revision: int
    fingerprint: str | None
    publication: int


class Registry(Generic[K, V]):
    """Store normalized keys, aliases, and groups in immutable snapshots.

    Reads capture one state reference and acquire no lock. Writes serialize
    through one local writer lock, build a complete copied state, optionally
    persist it with compare-and-set, and publish it with one assignment.

    Args:
        identifier (str): Stable registry namespace identifier.
        normalize (Callable[[K], K] | None): Canonical key normalizer. Identity
            normalization is used when omitted.
        normalize_alias (Callable[[K], K] | None): Alias lookup normalizer.
            `normalize` is used when omitted.
        normalize_group (Callable[[Hashable], Hashable] | None): Group lookup
            normalizer. Identity normalization is used when omitted.
        max_items (int | None): Optional canonical item limit. `None` is
            unbounded; a configured value must be positive.
        value_policy (Literal["snapshot", "process_identity"]): Value ownership
            policy. `snapshot` copies values on every write and read boundary.
            `process_identity` retains exact objects and is valid only for
            process-only Registries.

    Raises:
        ValueError: If `identifier`, `max_items`, or `value_policy` is invalid.

    Warnings:
        `process_identity` deliberately disables value isolation. Direct value
        mutation, including mutation inside a callback that later fails, is
        outside Registry rollback. Use it only for process-local object
        catalogs whose identity is their contract.
    """

    def __init__(
        self,
        identifier: str,
        *,
        normalize: Callable[[K], K] | None = None,
        normalize_alias: Callable[[K], K] | None = None,
        normalize_group: Callable[[Hashable], Hashable] | None = None,
        max_items: int | None = 65536,
        value_policy: RegistryValuePolicy = "snapshot",
    ) -> None:
        """Initialize one in-memory Registry namespace.

        Returns:
            None: This function does not return a value.
        """
        self.identifier = identifier_key(identifier, "registry identifier")
        if max_items is not None and (not isinstance(max_items, int) or isinstance(max_items, bool) or max_items < 1):
            raise ValueError(f"max_items must be a positive integer or None, got {max_items!r}")
        if value_policy not in ("snapshot", "process_identity"):
            raise ValueError(f"Unsupported Registry value policy {value_policy!r}; expected 'snapshot' or 'process_identity'")
        self.max_items = max_items
        self.value_policy: RegistryValuePolicy = value_policy
        self._normalize = normalize
        self._normalize_alias = normalize_alias or normalize
        self._normalize_group = normalize_group
        self._state = _State[K, V]({}, {}, {}, {}, {})
        self._lock = threading.Lock()
        self._store: RegistryStore | None = None
        self._codec: _RegistryCodec[K, V] | None = None
        self._draft = False
        self._draft_changed = False

    @classmethod
    def load(
        cls,
        identifier: str,
        *,
        store: RegistryStore,
        encode_key: Callable[[K], Any] | None = None,
        decode_key: Callable[[Any], K] | None = None,
        encode_value: Callable[[V], Any] | None = None,
        decode_value: Callable[[Any], V] | None = None,
        hash_fn: str | Callable[[bytes], str] | None = "sha256",
        normalize: Callable[[K], K] | None = None,
        normalize_alias: Callable[[K], K] | None = None,
        normalize_group: Callable[[Hashable], Hashable] | None = None,
        max_items: int | None = 65536,
        max_bytes: int | None = 4 * 1024 * 1024,
        value_policy: RegistryValuePolicy = "snapshot",
    ) -> Registry[K, V]:
        """Load a persistent Registry through a caller-owned store.

        Loading prepares and reads the store but performs no snapshot write for
        an absent namespace. The caller retains store lifecycle ownership.

        Args:
            identifier (str): Persistent registry namespace identifier.
            store (RegistryStore): Snapshot store shared by Registry instances.
            encode_key (Callable[[K], Any] | None): Optional JSON-safe key encoder.
            decode_key (Callable[[Any], K] | None): Optional key decoder paired
                with `encode_key`.
            encode_value (Callable[[V], Any] | None): Optional JSON-safe value encoder.
            decode_value (Callable[[Any], V] | None): Optional value decoder paired
                with `encode_value`.
            hash_fn (str | Callable[[bytes], str] | None): Integrity hashing.
                Supported values are `"sha256"`, a callable that returns a
                non-empty digest string, or `None` to disable digest checks.
            normalize (Callable[[K], K] | None): Canonical key normalizer.
            normalize_alias (Callable[[K], K] | None): Alias lookup normalizer.
            normalize_group (Callable[[Hashable], Hashable] | None): Group lookup normalizer.
            max_items (int | None): Optional canonical item limit.
            max_bytes (int | None): Optional encoded snapshot byte limit.
            value_policy (Literal["snapshot", "process_identity"]): Value
                ownership policy. Persistent Registries require `snapshot`.

        Returns:
            Registry[K, V]: Hydrated Registry using store CAS for writes.

        Raises:
            RegistryIntegrityError: If stored state fails identity, codec,
                revision, or digest validation.
            TypeError: If only one function in an encoder/decoder pair is supplied.
            ValueError: If a configured limit, identifier, or value policy is
                invalid.
        """
        registry = cls(
            identifier,
            normalize=normalize,
            normalize_alias=normalize_alias,
            normalize_group=normalize_group,
            max_items=max_items,
            value_policy=value_policy,
        )
        if registry.value_policy == "process_identity":
            raise ValueError("Persistent Registries require value_policy='snapshot'; process identity cannot survive reload")
        codec = _RegistryCodec[K, V](
            registry.identifier,
            encode_key=encode_key,
            decode_key=decode_key,
            encode_value=encode_value,
            decode_value=decode_value,
            hash_fn=hash_fn,
            max_bytes=max_bytes,
        )
        store.prepare()
        registry._store = store
        registry._codec = codec
        loaded = registry._read_state()
        if loaded is not None:
            registry._state = loaded
        return registry

    @classmethod
    async def aload(cls, identifier: str, *, store: RegistryStore, **kwargs: Any) -> Registry[K, V]:
        """Load a persistent Registry without blocking the event loop.

        Args:
            identifier (str): Persistent registry namespace identifier.
            store (RegistryStore): Snapshot store shared by Registry instances.
            **kwargs (Any): Additional options accepted by `Registry.load`.

        Returns:
            Registry[K, V]: Hydrated persistent Registry.
        """
        import asyncio

        return await asyncio.to_thread(cls.load, identifier, store=store, **kwargs)

    @property
    def revision(self) -> int:
        """Return the published persistent revision.

        Args:
            None.

        Returns:
            int: Current revision, or zero for a process-only Registry.
        """
        return self._state.revision

    @property
    def persistent(self) -> bool:
        """Return whether writes are bound to a Registry store.

        Args:
            None.

        Returns:
            bool: `True` when this Registry was opened with `Registry.load`.
        """
        return self._store is not None

    @property
    def publication(self) -> int:
        """Return the process-local observable-state publication sequence.

        Args:
            None.

        Returns:
            int: Monotonic sequence advanced once per successful top-level
                publication or refresh in this Registry instance.
        """
        return self._state.publication

    def _copy_value(self, value: V, boundary: str) -> V:
        if self.value_policy == "process_identity":
            return value
        try:
            return copy.deepcopy(value)
        except Exception as error:
            raise TypeError(
                f"Registry {self.identifier!r} could not snapshot {boundary} value of type {type(value).__name__}; "
                "use snapshot-compatible values or value_policy='process_identity' for a process-only Registry"
            ) from error

    def _copy_items(self, items: Mapping[K, V], boundary: str) -> dict[K, V]:
        return {key: self._copy_value(value, boundary) for key, value in items.items()}

    def _decoded_state(self, payload: bytes, *, revision: int, fingerprint: str | None) -> _State[K, V]:
        assert self._codec is not None
        decoded = self._codec.decode(payload)
        state = _State(
            self._copy_items(decoded.items, "decoded"),
            decoded.aliases,
            decoded.alias_names,
            decoded.groups,
            decoded.group_names,
            revision,
            fingerprint,
            revision,
        )
        if self.max_items is not None and len(state.items) > self.max_items:
            raise RegistryIntegrityError(f"Registry {self.identifier!r} contains {len(state.items)} items; capacity is {self.max_items}")
        self._validate_loaded_state(state)
        return state

    def _read_state(self) -> _State[K, V] | None:
        assert self._store is not None
        return self._stored_state(self._store.read(self.identifier))

    def _stored_state(self, stored: object) -> _State[K, V] | None:
        assert self._codec is not None
        if stored is None:
            return None
        if not isinstance(stored, tuple) or len(stored) != 3:
            raise RegistryIntegrityError(f"Registry {self.identifier!r} store returned an invalid snapshot")
        revision, payload, digest = stored
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise RegistryIntegrityError(f"Registry {self.identifier!r} has invalid revision {revision!r}")
        payload = self._payload_bytes(payload)
        self._codec.verify(revision, payload, digest)
        fingerprint = digest if isinstance(digest, str) and digest else None
        return self._decoded_state(payload, revision=revision, fingerprint=fingerprint)

    def _payload_bytes(self, payload: object) -> bytes:
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes):
            raise RegistryIntegrityError(f"Registry {self.identifier!r} payload must be bytes")
        return payload

    def _validate_loaded_state(self, state: _State[K, V]) -> None:
        try:
            if any(self._key(key) != key for key in state.items):
                raise ValueError("canonical keys are not normalized")
            expected_aliases = {
                self._alias_key(display): key for key, displays in state.alias_names.items() for display in displays if self._alias_key(display) != key
            }
            if expected_aliases != state.aliases:
                raise ValueError("display aliases do not match the alias index")
            if any(self._alias_key(alias) != alias for alias in state.aliases):
                raise ValueError("alias index keys are not normalized")
            if any(self._group(group) != group for group in state.groups):
                raise ValueError("group index keys are not normalized")
            if any(self._group(display) != group for group, display in state.group_names.items()):
                raise ValueError("display groups do not match the group index")
        except (TypeError, ValueError) as error:
            raise RegistryIntegrityError(f"Registry {self.identifier!r} persisted identities violate its normalizers: {error}") from error

    def _key(self, raw: K) -> K:
        key = self._normalize(raw) if self._normalize is not None else raw
        self._check_hashable(key, "key")
        return key

    def _alias_key(self, raw: K) -> K:
        key = self._normalize_alias(raw) if self._normalize_alias is not None else raw
        self._check_hashable(key, "alias")
        return key

    def _group(self, raw: Hashable) -> Hashable:
        group = self._normalize_group(raw) if self._normalize_group is not None else raw
        self._check_hashable(group, "group")
        return group

    def _check_hashable(self, value: object, kind: str) -> None:
        try:
            hash(value)
        except TypeError as error:
            raise TypeError(f"Registry {self.identifier!r} {kind} must normalize to a hashable value, got {value!r}") from error

    def _resolve(self, state: _State[K, V], raw: K) -> K:
        try:
            canonical = self._key(raw)
        except (TypeError, ValueError):
            alias = self._alias_key(raw)
            return state.aliases.get(alias, alias)
        if canonical in state.items:
            return canonical
        alias = self._alias_key(raw)
        return state.aliases.get(alias, canonical)

    def _aliases(self, raw: Iterable[K]) -> tuple[tuple[K, K], ...]:
        found: dict[K, K] = {}
        for display in raw:
            found.setdefault(self._alias_key(display), display)
        return tuple(found.items())

    def _groups(self, raw: Iterable[Hashable]) -> tuple[tuple[Hashable, Hashable], ...]:
        found: dict[Hashable, Hashable] = {}
        for display in raw:
            found.setdefault(self._group(display), display)
        return tuple(found.items())

    def _keys(self, state: _State[K, V], group: Hashable | None) -> tuple[K, ...]:
        if group is None:
            return tuple(state.items)
        members = set(state.groups.get(self._group(group), ()))
        return tuple(key for key in state.items if key in members)

    def _check_aliases(self, state: _State[K, V], key: K, aliases: tuple[tuple[K, K], ...]) -> None:
        for lookup, display in aliases:
            try:
                display_key = self._key(display)
            except (TypeError, ValueError):
                display_key = None
            if display_key is not None and display_key in state.items and display_key != key:
                raise ValueError(f"Registry {self.identifier!r} alias {display!r} conflicts with key {display_key!r}")
            if lookup == key:
                continue
            if lookup in state.items:
                raise ValueError(f"Registry {self.identifier!r} alias {display!r} conflicts with key {lookup!r}")
            target = state.aliases.get(lookup)
            if target is not None and target != key:
                raise ValueError(f"Registry {self.identifier!r} alias {display!r} is already registered for {target!r}")

    def _check_canonical(self, state: _State[K, V], raw: K, canonical: K) -> None:
        alias_token = self._alias_key(raw)
        if alias_token in state.items and alias_token != canonical:
            raise ValueError(f"Registry {self.identifier!r} key {canonical!r} conflicts with key {alias_token!r}")
        for token in dict.fromkeys((canonical, alias_token)):
            target = state.aliases.get(token)
            if target is not None and target != canonical:
                raise ValueError(f"Registry {self.identifier!r} key {canonical!r} conflicts with alias for {target!r}")

    @staticmethod
    def _replace_aliases(state: _State[K, V], key: K, aliases: tuple[tuple[K, K], ...]) -> tuple[dict[K, K], dict[K, tuple[K, ...]]]:
        index = {alias: target for alias, target in state.aliases.items() if target != key}
        names = dict(state.alias_names)
        displays: list[K] = []
        for lookup, display in aliases:
            if lookup != key:
                index[lookup] = key
                displays.append(display)
        names[key] = tuple(displays)
        return index, names

    @staticmethod
    def _replace_groups(
        state: _State[K, V], key: K, groups: tuple[tuple[Hashable, Hashable], ...]
    ) -> tuple[dict[Hashable, tuple[K, ...]], dict[Hashable, Hashable]]:
        wanted = dict(groups)
        members: dict[Hashable, tuple[K, ...]] = {}
        names = dict(state.group_names)
        for group, keys in state.groups.items():
            kept = tuple(found for found in keys if found != key)
            if group in wanted:
                members[group] = (*kept, key)
                names[group] = wanted.pop(group)
            elif kept:
                members[group] = kept
            else:
                names.pop(group, None)
        for group, display in groups:
            if group in wanted:
                members[group] = (key,)
                names[group] = display
                wanted.pop(group)
        return members, names

    def _next(
        self,
        state: _State[K, V],
        key: K,
        item: V,
        aliases: tuple[tuple[K, K], ...] | None,
        groups: tuple[tuple[Hashable, Hashable], ...] | None,
    ) -> _State[K, V]:
        if self._draft:
            if aliases is not None:
                self._check_aliases(state, key, aliases)
            state.items[key] = self._copy_value(item, "write")
            if aliases is not None:
                for alias, target in tuple(state.aliases.items()):
                    if target == key:
                        del state.aliases[alias]
                displays = []
                for lookup, display in aliases:
                    if lookup != key:
                        state.aliases[lookup] = key
                        displays.append(display)
                state.alias_names[key] = tuple(displays)
            if groups is not None:
                group_index, group_names = self._replace_groups(state, key, groups)
                state.groups.clear()
                state.groups.update(group_index)
                state.group_names.clear()
                state.group_names.update(group_names)
            return state
        items = dict(state.items)
        items[key] = self._copy_value(item, "write")
        if aliases is None:
            alias_index, alias_names = state.aliases, state.alias_names
        else:
            self._check_aliases(state, key, aliases)
            alias_index, alias_names = self._replace_aliases(state, key, aliases)
        if groups is None:
            group_index, group_names = state.groups, state.group_names
        else:
            group_index, group_names = self._replace_groups(state, key, groups)
        return _State(items, alias_index, alias_names, group_index, group_names, state.revision, state.fingerprint, state.publication)

    def _publish(self, state: _State[K, V], next_state: _State[K, V]) -> None:
        if self._draft:
            self._state = next_state
            self._draft_changed = True
            return
        if self._store is not None:
            assert self._codec is not None
            revision = state.revision + 1
            payload = self._codec.encode(
                self._copy_items(next_state.items, "persistent"),
                next_state.aliases,
                next_state.alias_names,
                next_state.groups,
                next_state.group_names,
            )
            digest = self._codec.digest(revision, payload)
            published = self._decoded_state(payload, revision=revision, fingerprint=digest or None)
            won = self._store.compare_and_set(
                self.identifier,
                expected_revision=state.revision,
                revision=revision,
                payload=payload,
                digest=digest,
            )
            if not won:
                raise RegistryConflictError(f"Registry {self.identifier!r} compare-and-set conflict at revision {state.revision}; refresh before retrying")
            next_state = published
        self._state = _State(
            next_state.items,
            next_state.aliases,
            next_state.alias_names,
            next_state.groups,
            next_state.group_names,
            next_state.revision,
            next_state.fingerprint,
            state.publication + 1,
        )

    def resolve(self, key: K) -> K:
        """Resolve a key or alias to its canonical normalized form.

        Args:
            key (K): Canonical key or alias to resolve.

        Returns:
            K: Registered canonical key, or normalized unknown key.
        """
        state = self._state
        return self._resolve(state, key)

    @overload
    def get(self, key: K) -> V | None:
        """Return a registered value when no default is supplied.

        Args:
            key (K): Canonical key or alias to read.

        Returns:
            V | None: Detached registered value, or ``None`` when absent.
        """
        ...

    @overload
    def get(self, key: K, default: D) -> V | D:
        """Return a registered value or a caller-provided default.

        Args:
            key (K): Canonical key or alias to read.
            default (D): Value returned when the key is absent.

        Returns:
            V | D: Detached registered value or ``default``.
        """
        ...

    def get(self, key: K, default: D | None = None) -> V | D | None:
        """Return a registered value or caller-provided default.

        Args:
            key (K): Canonical key or alias to read.
            default (D | None): Value returned when the key is absent.

        Returns:
            V | D | None: Detached registered value under the default
                `snapshot` policy, or the supplied default.
        """
        state = self._state
        canonical = self._resolve(state, key)
        if canonical not in state.items:
            return default
        return self._copy_value(state.items[canonical], "read")

    def list(self, group: Hashable | None = None) -> tuple[K, ...]:
        """List canonical keys in registration order.

        Args:
            group (Hashable | None): Optional group used to filter keys.

        Returns:
            tuple[K, ...]: Canonical keys in stable registration order.
        """
        state = self._state
        return self._keys(state, group)

    def keys(self, group: Hashable | None = None) -> tuple[K, ...]:
        """Return canonical keys using the standard Mapping vocabulary.

        Args:
            group (Hashable | None): Optional group used to filter keys.

        Returns:
            tuple[K, ...]: Canonical keys in stable registration order.
        """
        return self.list(group)

    def values(self, group: Hashable | None = None) -> tuple[V, ...]:
        """List registered values in canonical registration order.

        Args:
            group (Hashable | None): Optional group used to filter values.

        Returns:
            tuple[V, ...]: Registered values detached under the default
                `snapshot` policy.
        """
        state = self._state
        return tuple(self._copy_value(state.items[key], "read") for key in self._keys(state, group))

    def items(self, group: Hashable | None = None) -> tuple[tuple[K, V], ...]:
        """List canonical key-value pairs in registration order.

        Args:
            group (Hashable | None): Optional group used to filter pairs.

        Returns:
            tuple[tuple[K, V], ...]: Canonical keys paired with values detached
                under the default `snapshot` policy.
        """
        state = self._state
        return tuple((key, self._copy_value(state.items[key], "read")) for key in self._keys(state, group))

    def _start_from(self, other: Registry[K, V], *, copy_value: Callable[[V], V] | None = None) -> None:
        """Start this empty process-only Registry from another one's current snapshot.

        Published snapshots are never changed in place, so the two Registries
        share it and stay independent: a later write to either publishes a
        new snapshot for that Registry only. Under the `snapshot` policy the
        shared values are already detached copies.

        Args:
            other: Process-only Registry with the same identifier and value policy.
            copy_value: Optional copier of each value, for the
                `process_identity` policy, whose values are shared otherwise.

        Raises:
            ValueError: If either Registry is persistent, a draft, or not empty,
                or if their identifiers or value policies differ.
        """
        if self._store is not None or other._store is not None or self._draft or other._draft:
            raise ValueError("Only process-only, published Registries can share a snapshot")
        if self.identifier != other.identifier or self.value_policy != other.value_policy:
            raise ValueError("Registries sharing a snapshot need the same identifier and value policy")
        with self._lock:
            if self._state.items:
                raise ValueError(f"Registry {self.identifier!r} must be empty to start from another snapshot")
            state = other._state
            if copy_value is not None:
                state = replace(state, items={key: copy_value(value) for key, value in state.items.items()})
            self._state = state

    def snapshot(self) -> Mapping[K, V]:
        """Return the current read-only canonical item snapshot.

        Under the default `snapshot` value policy, each value is detached from
        Registry state. The returned mapping is stable across later writes.

        Args:
            None.

        Returns:
            Mapping[K, V]: Read-only mapping of captured canonical items.
        """
        state = self._state
        return MappingProxyType(self._copy_items(state.items, "read"))

    def view(self) -> RegistryView[K, V]:
        """Return one coherent read view of items, aliases, groups, and identity.

        Args:
            None.

        Returns:
            RegistryView[K, V]: Read-only structural mappings captured from one
                state, with detached values under the default `snapshot`
                policy, plus its revision and persistent fingerprint.
        """
        state = self._state
        return RegistryView(
            items=MappingProxyType(self._copy_items(state.items, "read")),
            aliases=MappingProxyType(dict(state.aliases)),
            alias_names=MappingProxyType(dict(state.alias_names)),
            groups=MappingProxyType(dict(state.groups)),
            group_names=MappingProxyType(dict(state.group_names)),
            revision=state.revision,
            fingerprint=state.fingerprint,
            publication=state.publication,
        )

    def aliases(self, key: K, *, include_canonical: bool = False) -> tuple[K, ...]:
        """List declared aliases while preserving display spelling.

        Args:
            key (K): Canonical key or alias to inspect.
            include_canonical (bool): Whether to prefix the canonical key.

        Returns:
            tuple[K, ...]: Display aliases, optionally prefixed by the key.
        """
        state = self._state
        canonical = self._resolve(state, key)
        if canonical not in state.items:
            return ()
        aliases = state.alias_names.get(canonical, ())
        return (canonical, *aliases) if include_canonical else aliases

    def groups(self, key: K) -> tuple[Hashable, ...]:
        """List display groups containing one registered key.

        Args:
            key (K): Canonical key or alias to inspect.

        Returns:
            tuple[Hashable, ...]: Groups in group-registration order.
        """
        state = self._state
        canonical = self._resolve(state, key)
        return tuple(state.group_names[group] for group, keys in state.groups.items() if canonical in keys)

    def in_group(self, key: K, group: Hashable) -> bool:
        """Return whether a registered key belongs to a group.

        Args:
            key (K): Canonical key or alias to inspect.
            group (Hashable): Group name to test.

        Returns:
            bool: Whether the registered key belongs to the group.
        """
        state = self._state
        canonical = self._resolve(state, key)
        return canonical in state.items and canonical in state.groups.get(self._group(group), ())

    def create(
        self,
        key: K,
        item: V,
        *,
        aliases: Iterable[K] = (),
        groups: Iterable[Hashable] = (),
    ) -> V:
        """Create one canonical registration without replacement.

        Args:
            key (K): Canonical key to create.
            item (V): Value copied into Registry ownership under the default
                `snapshot` policy. `None` is valid.
            aliases (Iterable[K]): Display aliases for lookup.
            groups (Iterable[Hashable]): Flat semantic groups containing the key.

        Returns:
            V: Published value detached under the default `snapshot` policy.

        Raises:
            OverflowError: If `max_items` would be exceeded.
            ValueError: If the key exists or an alias conflicts.
        """
        canonical = self._key(key)
        alias_pairs = self._aliases(aliases)
        group_pairs = self._groups(groups)
        with self._lock:
            state = self._state
            if canonical in state.items:
                raise ValueError(f"Registry {self.identifier!r} key {canonical!r} is already registered")
            self._check_canonical(state, key, canonical)
            if self.max_items is not None and len(state.items) >= self.max_items:
                raise OverflowError(f"Registry {self.identifier!r} capacity {self.max_items} reached")
            self._publish(state, self._next(state, canonical, item, alias_pairs, group_pairs))
            return self._copy_value(self._state.items[canonical], "read")

    @overload
    def update(
        self,
        key: K,
        item: V,
        *,
        derive: None = None,
        aliases: Iterable[K] | None = None,
        groups: Iterable[Hashable] | None = None,
    ) -> V:
        """Replace one registration with an explicit item value.

        Args:
            key (K): Canonical key or alias to update.
            item (V): Replacement value.
            derive (None): Must remain ``None`` for this overload.
            aliases (Iterable[K] | None): Replacement aliases, if any.
            groups (Iterable[Hashable] | None): Replacement groups, if any.

        Returns:
            V: Published value detached under the default snapshot policy.
        """
        ...

    @overload
    def update(
        self,
        key: K,
        *,
        derive: Callable[[V | None], V],
        aliases: Iterable[K] | None = None,
        groups: Iterable[Hashable] | None = None,
    ) -> V:
        """Create, replace, or derive one registration through a callback.

        Args:
            key (K): Canonical key or alias to update.
            derive (Callable[[V | None], V]): Derivation callback.
            aliases (Iterable[K] | None): Replacement aliases, if any.
            groups (Iterable[Hashable] | None): Replacement groups, if any.

        Returns:
            V: Published value detached under the default snapshot policy.
        """
        ...

    def update(
        self,
        key: K,
        item: object = _MISSING,
        *,
        derive: Callable[[V | None], V] | None = None,
        aliases: Iterable[K] | None = None,
        groups: Iterable[Hashable] | None = None,
    ) -> V:
        """Create, replace, or derive one registration atomically.

        Callable values passed as `item` are stored normally. A `derive`
        callback receives a detached current value under the default
        `snapshot` policy, runs once under the local writer lock, and is never
        retried after a persistent CAS conflict.

        Args:
            key (K): Canonical key or alias to update.
            item (V): Replacement value. Omit when using `derive`.
            derive (Callable[[V | None], V] | None): Callback receiving the
                current value, or `None` when the key is absent.
            aliases (Iterable[K] | None): Replacement aliases. `None` preserves
                metadata for an existing key and means no aliases on creation.
            groups (Iterable[Hashable] | None): Replacement groups. `None`
                preserves metadata for an existing key and means no groups on creation.

        Returns:
            V: Published value detached under the default `snapshot` policy.

        Raises:
            OverflowError: If creating would exceed `max_items`.
            TypeError: If both or neither `item` and `derive` are supplied.
            ValueError: If the key or aliases conflict.

        Notes:
            `derive` receives `None` for both an absent key and a registered
            `None` value. Use a direct replacement when that distinction matters.
        """
        has_item = item is not _MISSING
        if has_item == (derive is not None):
            raise TypeError("update requires exactly one replacement item or derive callback")
        alias_pairs = None if aliases is None else self._aliases(aliases)
        group_pairs = None if groups is None else self._groups(groups)
        with self._lock:
            state = self._state
            resolved = self._resolve(state, key)
            exists = resolved in state.items
            canonical = resolved if exists else self._key(key)
            if not exists:
                self._check_canonical(state, key, canonical)
                if self.max_items is not None and len(state.items) >= self.max_items:
                    raise OverflowError(f"Registry {self.identifier!r} capacity {self.max_items} reached")
                alias_pairs = alias_pairs or ()
                group_pairs = group_pairs or ()
            current = self._copy_value(state.items[canonical], "derive") if canonical in state.items else None
            value = derive(current) if derive is not None else cast(V, item)
            self._publish(state, self._next(state, canonical, value, alias_pairs, group_pairs))
            return self._copy_value(self._state.items[canonical], "read")

    def delete(self, key: K, *, missing_ok: bool = False) -> V | None:
        """Delete one registration and all of its aliases and groups.

        Args:
            key (K): Canonical key or alias to delete.
            missing_ok (bool): Whether an absent key returns `None`.

        Returns:
            V | None: Removed value, or `None` for an allowed missing key.

        Raises:
            KeyError: If the key is absent and `missing_ok` is false.
        """
        with self._lock:
            state = self._state
            canonical = self._resolve(state, key)
            if canonical not in state.items:
                if missing_ok:
                    return None
                raise KeyError(f"Unknown Registry {self.identifier!r} key {key!r}; known={list(state.items)!r}") from None
            value = self._copy_value(state.items[canonical], "read")
            items = dict(state.items)
            del items[canonical]
            aliases = {alias: target for alias, target in state.aliases.items() if target != canonical}
            alias_names = dict(state.alias_names)
            alias_names.pop(canonical, None)
            groups, group_names = self._replace_groups(state, canonical, ())
            self._publish(state, _State(items, aliases, alias_names, groups, group_names, state.revision, state.fingerprint, state.publication))
            return value

    @overload
    def pop(self, key: K) -> V:
        """Delete and return one value without a default.

        Args:
            key (K): Canonical key or alias to delete.

        Returns:
            V: Removed value.

        Raises:
            KeyError: If the key is absent.
        """
        ...

    @overload
    def pop(self, key: K, default: D) -> V | D:
        """Delete and return one value, or return a default when absent.

        Args:
            key (K): Canonical key or alias to delete.
            default (D): Value returned when the key is absent.

        Returns:
            V | D: Removed value or ``default``.
        """
        ...

    def pop(self, key: K, default: object = _MISSING) -> V | D:
        """Delete and return one value using standard mutable-mapping semantics.

        Args:
            key (K): Canonical key or alias to delete.
            default (D): Optional value returned when the key is absent.

        Returns:
            V | D: Removed value or the supplied default.

        Raises:
            KeyError: If the key is absent and no default was supplied.
        """
        if default is _MISSING:
            return cast(V, self.delete(key))
        try:
            return cast(V, self.delete(key))
        except KeyError:
            return cast(D, default)

    def refresh(self) -> bool:
        """Publish a newer persistent store snapshot when available.

        With hashing enabled, a stored snapshot whose revision and digest
        equal the published ones has its digest verified against its payload
        and is not decoded again. Any other stored snapshot has its digest
        verified when hashing is enabled, and is decoded and validated.

        Args:
            None.

        Returns:
            bool: Whether a newer snapshot was published. Process-only
            Registries always return `False`.

        Raises:
            RegistryIntegrityError: If persisted state is corrupt or older
                than the current revision.
        """
        if self._store is None:
            return False
        with self._lock:
            state = self._state
            stored = self._store.read(self.identifier)
            if self._is_published(stored, state):
                return False
            loaded = self._stored_state(stored)
            if loaded is None or loaded.revision == state.revision:
                return False
            if loaded.revision < state.revision:
                raise RegistryIntegrityError(f"Registry {self.identifier!r} persisted revision {loaded.revision} is older than local revision {state.revision}")
            self._state = _State(
                loaded.items,
                loaded.aliases,
                loaded.alias_names,
                loaded.groups,
                loaded.group_names,
                loaded.revision,
                loaded.fingerprint,
                state.publication + 1,
            )
            return True

    def _is_published(self, stored: object, state: _State[K, V]) -> bool:
        """Return whether a stored snapshot is the published one.

        A matching revision and integrity digest identify the payload this
        Registry already decoded. The digest is still verified against the
        stored payload, so a payload changed under the published digest raises;
        only the decode, copy and validation are skipped. Without a digest
        (hashing disabled) the snapshot is decoded and validated as before.

        Raises:
            RegistryIntegrityError: If the stored payload does not match the
                published digest.
        """
        assert self._codec is not None
        if state.fingerprint is None or not isinstance(stored, tuple) or len(stored) != 3:
            return False
        revision, payload, digest = stored
        if revision != state.revision or digest != state.fingerprint:
            return False
        self._codec.verify(revision, self._payload_bytes(payload), digest)
        return True

    def mutate(self, change: Callable[[Registry[K, V]], R]) -> R:
        """Apply several Registry mutations as one atomic publication.

        The callback receives an isolated process-only draft initialized from
        the current snapshot. Its `create`, `update`, and `delete` calls are
        published together in one local assignment and, for persistent
        Registries, one compare-and-set revision. An exception or conflict
        leaves the original Registry unchanged under the default `snapshot`
        policy. The `process_identity` policy can roll back structural changes,
        but cannot undo mutation of the exact objects it deliberately shares.

        Args:
            change (Callable[[Registry[K, V]], R]): Synchronous callback that
                mutates the isolated draft and returns a caller-defined result.

        Returns:
            R: Exact result returned by `change`.

        Raises:
            RegistryConflictError: If the durable revision changed before the
                completed draft could be published.
        """
        with self._lock:
            state = self._state
            draft = type(self)(
                self.identifier,
                normalize=self._normalize,
                normalize_alias=self._normalize_alias,
                normalize_group=self._normalize_group,
                max_items=self.max_items,
                value_policy=self.value_policy,
            )
            draft._draft = True
            draft._state = _State(
                dict(state.items),
                dict(state.aliases),
                dict(state.alias_names),
                dict(state.groups),
                dict(state.group_names),
                state.revision,
                state.fingerprint,
                state.publication,
            )
            result = change(draft)
            if draft._draft_changed:
                self._publish(state, draft._state)
            return result

    async def amutate(self, change: Callable[[Registry[K, V]], R]) -> R:
        """Apply several Registry mutations atomically off the event loop.

        Args:
            change (Callable[[Registry[K, V]], R]): Synchronous callback that
                mutates an isolated draft and returns a caller-defined result.

        Returns:
            R: Exact result returned by `change`.

        Raises:
            RegistryConflictError: If the durable revision changed before the
                completed draft could be published.
        """
        import asyncio

        return await asyncio.to_thread(self.mutate, change)

    async def acreate(self, key: K, item: V, *, aliases: Iterable[K] = (), groups: Iterable[Hashable] = ()) -> V:
        """Create a registration without blocking the event loop.

        Args:
            key (K): Canonical key to create.
            item (V): Value to store.
            aliases (Iterable[K]): Display aliases for lookup.
            groups (Iterable[Hashable]): Semantic groups containing the key.

        Returns:
            V: Published value detached under the default `snapshot` policy.
        """
        import asyncio

        return await asyncio.to_thread(self.create, key, item, aliases=aliases, groups=groups)

    async def aupdate(
        self,
        key: K,
        item: object = _MISSING,
        *,
        derive: Callable[[V | None], V] | None = None,
        aliases: Iterable[K] | None = None,
        groups: Iterable[Hashable] | None = None,
    ) -> V:
        """Create, replace, or derive a registration without blocking the event loop.

        Args:
            key (K): Canonical key or alias.
            item (V): Replacement value. Omit when using `derive`.
            derive (Callable[[V | None], V] | None): Synchronous derivation callback.
            aliases (Iterable[K] | None): Replacement aliases or `None` to preserve.
            groups (Iterable[Hashable] | None): Replacement groups or `None` to preserve.

        Returns:
            V: Published value detached under the default `snapshot` policy.
        """
        import asyncio

        return await asyncio.to_thread(self.update, key, item, derive=derive, aliases=aliases, groups=groups)

    async def adelete(self, key: K, *, missing_ok: bool = False) -> V | None:
        """Delete a registration without blocking the event loop.

        Args:
            key (K): Canonical key or alias to delete.
            missing_ok (bool): Whether an absent key returns `None`.

        Returns:
            V | None: Removed value or `None` for an allowed miss.
        """
        import asyncio

        return await asyncio.to_thread(self.delete, key, missing_ok=missing_ok)

    async def arefresh(self) -> bool:
        """Refresh persistent state without blocking the event loop.

        Args:
            None.

        Returns:
            bool: Whether a newer snapshot was published.
        """
        import asyncio

        return await asyncio.to_thread(self.refresh)

    def __len__(self) -> int:
        return len(self._state.items)

    def __iter__(self) -> Iterator[K]:
        return iter(self._state.items)

    def __contains__(self, key: object) -> bool:
        state = self._state
        try:
            return self._resolve(state, cast(K, key)) in state.items
        except (TypeError, ValueError):
            return False

    def __getitem__(self, key: K) -> V:
        state = self._state
        canonical = self._resolve(state, key)
        if canonical not in state.items:
            raise KeyError(key)
        return self._copy_value(state.items[canonical], "read")
