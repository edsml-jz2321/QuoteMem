"""Module resolution contract and resolver-local materialization cache."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
import threading
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable
import weakref

__all__ = ["ModuleResolver", "module_resolver"]

T = TypeVar("T")
_CacheKey = tuple[tuple[str, str], int, int, str]
_MISSING = object()


class _ModuleCatalog(Protocol):
    """Read contract exposed by a module resolver's catalog."""

    @property
    def publication(self) -> int:
        """Return the process-local publication sequence."""
        ...


@runtime_checkable
class ModuleResolver(Protocol):
    """Read registered module definitions from one Context-owned authority.

    Implementations expose one coherent catalog plus inert inspection and lazy
    resolution. ``ModuleService`` satisfies this protocol structurally; domain
    consumers can depend on this smaller read contract without gaining module
    installation or lifecycle mutation methods.
    """

    @property
    def catalog(self) -> _ModuleCatalog:
        """Return the resolver's read-only module catalog."""
        ...

    def inspect(
        self,
        kind: str | None = None,
        identifier: str | None = None,
        *,
        include_inactive: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Return detached records from one coherent catalog snapshot.

        Args:
            kind (str | None): Optional registration kind filter.
            identifier (str | None): Optional canonical identifier or alias.
                Requires ``kind`` when supplied.
            include_inactive (bool): Whether to include retired records.
                Defaults to ``False``.

        Returns:
            tuple[dict[str, Any], ...]: Detached inspection rows from one
            Registry snapshot.
        """
        ...

    def resolve(self, kind: str, identifier: str) -> Any:
        """Restore one active registration by kind and identifier or alias.

        Args:
            kind (str): Registration kind.
            identifier (str): Canonical identifier or alias within ``kind``.

        Returns:
            Any: Restored registration value.
        """
        ...

    def resolve_executable(
        self,
        reference: Mapping[str, Any],
        *,
        owner: Mapping[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Restore one versioned executable reference.

        Args:
            reference: Canonical executable-reference envelope.
            owner: Optional owning module record required by artifact-relative
                references.

        Returns:
            Callable[..., Any]: Restored executable object.
        """
        ...


def module_resolver(resolver: ModuleResolver | None = None) -> ModuleResolver:
    """Return an explicit resolver or lazily select the default Context.

    Args:
        resolver: Context-owned module resolver. When omitted, the process
            default Context is imported and selected lazily.

    Returns:
        ModuleResolver: The supplied resolver or the default Context's
        ``modules()``.

    Raises:
        TypeError: If ``resolver`` does not implement the read contract.
    """
    if resolver is None:
        from ..context import default_context

        resolver = default_context().modules()
    if not isinstance(resolver, ModuleResolver):
        raise TypeError("Expected a ModuleResolver with catalog, inspect, and resolve")
    return resolver


class _ResolverState(Generic[T]):
    """Hold one resolver's current-publication cache without owning it."""

    def __init__(self, reference: weakref.ReferenceType[ModuleResolver], publication: int) -> None:
        self.reference = reference
        self.publication = publication
        self.values: dict[_CacheKey, T | _WeakValue[T]] = {}


class _WeakValue(Generic[T]):
    """Store one cache value without completing a value-to-resolver cycle."""

    __slots__ = ("reference",)

    def __init__(self, value: T) -> None:
        self.reference = weakref.ref(value)  # type: ignore[arg-type]


class ResolverCache(Generic[T]):
    """Cache typed materializations by weak resolver identity and record state.

    Cache identity includes the canonical ``(kind, identifier)``, process-local
    publication, durable revision, and record fingerprint. Resolver ownership
    is identity-based rather than equality-based. Materializations that support
    weak references are also held weakly, so a value that binds its resolver
    cannot complete a cache-to-value-to-resolver retention cycle.

    Builders run outside the cache lock. Concurrent misses may therefore build
    the same value more than once, but only one value is published and returned
    for a still-current key. A builder receives the authoritative inspection
    row for its attempt; when publication advances during materialization, the
    cache re-inspects the record and retries the whole build.
    """

    def __init__(self) -> None:
        """Create an empty resolver-local materialization cache.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        self._states: dict[int, _ResolverState[T]] = {}
        self._lock = threading.RLock()
        # States of collected resolvers, queued by their weak-reference callbacks.
        self._collected: deque[tuple[int, weakref.ReferenceType[ModuleResolver]]] = deque()

    def get(
        self,
        resolver: ModuleResolver,
        row: Mapping[str, Any],
        builder: Callable[[Mapping[str, Any]], T],
        *,
        should_cache: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> T:
        """Return or build one resolver-local typed materialization.

        Args:
            resolver: Authority that supplied ``row`` and resolves its target.
            row: Canonical module inspection row containing ``kind``,
                ``identifier``, ``publication``, ``revision``, and
                ``fingerprint``.
            builder: Row-aware materializer invoked outside cache locks on a
                miss. The supplied row always belongs to the publication being
                validated for that build attempt.
            should_cache: Optional row-aware policy. A false result preserves
                atomic materialization without publishing that attempt to the
                cache, for registrations whose targets are intentionally
                dynamic.

        Returns:
            T: Cached or newly materialized value.

        Raises:
            TypeError: If the resolver is invalid, cannot be weakly referenced,
                the row state fields have invalid types, or ``builder`` is not
                callable, or ``should_cache`` is neither callable nor null.
            KeyError: If a required row state field is absent.
            ValueError: If the canonical key or fingerprint is empty.
        """
        resolver = module_resolver(resolver)
        if not callable(builder):
            raise TypeError("ResolverCache builder must be callable")
        if should_cache is not None and not callable(should_cache):
            raise TypeError("ResolverCache should_cache policy must be callable or null")
        self._reap()
        current_row = row
        resolver_id = id(resolver)
        while True:
            key = self._key(current_row)
            canonical_key, publication, _, _ = key
            if resolver.catalog.publication != publication:
                current_row = self._refresh(resolver, canonical_key)
                continue

            cache_attempt = True if should_cache is None else should_cache(current_row)
            if not isinstance(cache_attempt, bool):
                raise TypeError("ResolverCache should_cache policy must return bool")
            if cache_attempt:
                with self._lock:
                    state = self._state(resolver, resolver_id, publication)
                    if publication < state.publication:
                        stale = True
                        cached = _MISSING
                    else:
                        stale = False
                        if publication > state.publication:
                            state.publication = publication
                            state.values.clear()
                        cached = self._cached_value(state, key)
            else:
                stale = False
                cached = _MISSING
            if stale:
                current_row = self._refresh(resolver, canonical_key)
                continue
            if cached is not _MISSING:
                if resolver.catalog.publication == publication:
                    return cast(T, cached)
                current_row = self._refresh(resolver, canonical_key)
                continue

            try:
                value = builder(current_row)
            except Exception:
                if resolver.catalog.publication == publication:
                    raise
                current_row = self._refresh(resolver, canonical_key)
                continue

            if not cache_attempt:
                if resolver.catalog.publication == publication:
                    return value
                current_row = self._refresh(resolver, canonical_key)
                continue

            with self._lock:
                state = self._state(resolver, resolver_id, publication)
                if publication < state.publication or resolver.catalog.publication != publication:
                    stale = True
                    cached = _MISSING
                else:
                    stale = False
                    if publication > state.publication:
                        state.publication = publication
                        state.values.clear()
                    cached = self._cached_value(state, key)
                    if cached is _MISSING:
                        state.values[key] = self._stored_value(value)
                        cached = value
            if stale:
                current_row = self._refresh(resolver, canonical_key)
                continue
            return cast(T, cached)

    def clear(self, resolver: ModuleResolver | None = None) -> None:
        """Clear all values or only values owned by one resolver.

        Args:
            resolver: Optional resolver whose values should be removed. When
                omitted, every resolver state is discarded.

        Returns:
            None: Cache state is mutated in place.
        """
        self._reap()
        with self._lock:
            if resolver is None:
                self._states.clear()
                return
            state = self._states.get(id(resolver))
            if state is not None and state.reference() is resolver:
                self._states.pop(id(resolver), None)

    def __len__(self) -> int:
        """Return the number of live cached materializations."""
        self._reap()
        with self._lock:
            dead = [resolver_id for resolver_id, state in self._states.items() if state.reference() is None]
            for resolver_id in dead:
                self._states.pop(resolver_id, None)
            for state in self._states.values():
                expired = [key for key, value in state.values.items() if isinstance(value, _WeakValue) and value.reference() is None]
                for key in expired:
                    state.values.pop(key, None)
            return sum(len(state.values) for state in self._states.values())

    @staticmethod
    def _stored_value(value: T) -> T | _WeakValue[T]:
        """Prefer a weak value while retaining non-weak-referenceable scalars."""
        try:
            return _WeakValue(value)
        except TypeError:
            return value

    @staticmethod
    def _cached_value(state: _ResolverState[T], key: _CacheKey) -> T | object:
        """Return one live value and discard an expired weak entry."""
        stored = state.values.get(key, _MISSING)
        if not isinstance(stored, _WeakValue):
            return stored
        value = stored.reference()
        if value is None:
            state.values.pop(key, None)
            return _MISSING
        return cast(T, value)

    def _state(self, resolver: ModuleResolver, resolver_id: int, publication: int) -> _ResolverState[T]:
        """Return identity-matched state while the caller holds ``_lock``."""
        state = self._states.get(resolver_id)
        if state is not None and state.reference() is resolver:
            return state
        if state is not None:
            self._states.pop(resolver_id, None)
        collected = self._collected
        try:
            reference = weakref.ref(resolver, lambda ref: collected.append((resolver_id, ref)))
        except TypeError as error:
            raise TypeError("ModuleResolver implementations used with ResolverCache must support weak references") from error
        state = _ResolverState(reference, publication)
        self._states[resolver_id] = state
        return state

    def _reap(self) -> None:
        """Discard the states of collected resolvers without disturbing reused ids.

        A resolver can be collected inside any critical section of any thread,
        this cache's own included, where its weak-reference callback must
        neither wait for a lock nor change the states under the interrupted
        section. The callback therefore only queues the state with an atomic
        append; it is discarded here, at an ordinary call, and released after
        the lock so no finalizer runs under it.
        """
        if not self._collected:
            return
        discarded: list[_ResolverState[T]] = []
        with self._lock:
            while self._collected:
                resolver_id, reference = self._collected.popleft()
                state = self._states.get(resolver_id)
                if state is not None and state.reference is reference:
                    discarded.append(self._states.pop(resolver_id))
        discarded.clear()

    @classmethod
    def _refresh(cls, resolver: ModuleResolver, canonical_key: tuple[str, str]) -> Mapping[str, Any]:
        """Re-inspect one canonical record after its publication becomes stale."""
        kind, identifier = canonical_key
        rows = resolver.inspect(kind, identifier)
        if not rows:
            raise KeyError(f"Module record {kind!r}/{identifier!r} disappeared during materialization")
        row = rows[0]
        if cls._key(row)[0] != canonical_key:
            raise ValueError(f"Module resolver returned a different canonical record for {kind!r}/{identifier!r}")
        return row

    @staticmethod
    def _key(row: Mapping[str, Any]) -> _CacheKey:
        """Return the complete cache identity for one canonical row."""
        kind = str(row["kind"]).strip()
        identifier = str(row["identifier"]).strip()
        if not kind or not identifier:
            raise ValueError("ResolverCache rows require a non-empty canonical kind and identifier")
        publication = row["publication"]
        revision = row["revision"]
        if isinstance(publication, bool) or not isinstance(publication, int) or publication < 0:
            raise TypeError("ResolverCache row publication must be a non-negative integer")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise TypeError("ResolverCache row revision must be a non-negative integer")
        fingerprint = row["fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("ResolverCache row fingerprint must be non-empty text")
        return (kind, identifier), publication, revision, fingerprint
