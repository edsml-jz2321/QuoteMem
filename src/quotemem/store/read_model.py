"""Owner-controlled publication of the Evidence read model.

The pure snapshot owns immutable values. This adapter reads HeavenBase at cold
initialization and applies the exact committed delta supplied by the Writer.
"""

from __future__ import annotations

import math
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from operator import attrgetter
from pathlib import Path
from traceback import extract_tb
from typing import Any, Literal, Self, cast

from pydantic import ValidationError

from quotemem.core.add_read_view import AddReadView, TombstoneSource
from quotemem.core.admission import admission_identity, finalize_admission
from quotemem.core.controls import (
    CancelledFinalizeLeaseV1,
    ControlRowV1,
    CounterV1,
    EvidenceHoldV1,
    FinalizedValueV1,
    HeldFinalizeLeaseV1,
    PreparingFinalizeLeaseV1,
    ReleasedFinalizeLeaseV1,
    WriterLeaseV1,
)
from quotemem.core.delivery import DeliveryProjection, MemoryMode, QuoteKey, recheck_delivery
from quotemem.core.embedding import embedding_state_token
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import (
    ArtifactId,
    ContentBindingDigest,
    IntegrityDigest,
    canonical_json,
    integrity_digest,
    vector_encodings,
)
from quotemem.core.identity import (
    scope_id as canonical_scope_id,
)
from quotemem.core.knowledge import KnowledgeReadView
from quotemem.core.query import (
    FrozenScopeGrantSetV1,
    QueryBundleV1,
    ScopeGrantV1,
    ScopeV1,
)
from quotemem.core.recall_corpora import SnapshotRecallCache
from quotemem.core.recall_policy import normalize_recall_policy
from quotemem.core.recall_snapshot import (
    AddViewCarry,
    CarriedRows,
    FinalizeBarrier,
    FinalizedMarker,
    FinalizePhase,
    RecallPolicySelection,
    RecallSnapshot,
    ScopeFence,
    SnapshotPublisher,
    SourceRowV2,
    carried_rows,
)
from quotemem.core.retained_policy import read_released_recall_config
from quotemem.core.run_inputs import FinalizeOwnerRunInputV1, SessionDurableRunInputV1
from quotemem.core.semantic import ScopeKind
from quotemem.providers.auth import PrincipalGrant
from quotemem.store.domain import (
    DOMAIN_ADAPTERS,
    EvidenceRowV1,
)
from quotemem.store.failures import is_runtime_failure
from quotemem.store.finalize_ticket import FinalizeLeaseTicket
from quotemem.store.indexed import IndexedReader
from quotemem.store.snapshot_rows import cold_projections, commit_delta
from quotemem.store.writer import CommittedRows, Writer


@dataclass(frozen=True, slots=True)
class SnapshotMetadata:
    """Content-free publication identity; content requires a scoped guard."""

    generation: int
    root_identity: IntegrityDigest


@dataclass(frozen=True, slots=True, weakref_slot=True)
class SearchContext:
    """Content-free Search authority selected from one registered snapshot pin.

    The guard issues this value from an actual provider grant and its own root.
    Policy bytes and scope fences are immutable; ``allowed_scopes`` returns a
    detached strict model. This value owns no root, rows or hydrated content and
    cannot authorize delivery through another guard.

    Attributes:
        metadata: Exact computation snapshot identity and data generation.
        query_time: Frozen request time at which the original grant is valid.
        fence: The requested exact scope's same-root delivery fence.
        embedding_space: Effective space from that root.
        embedding_token: Complete active embedding-state token from that root.
        recall_policy: Released policy selection linearized at the original pin.
        finalize_phase: Current cycle selected for this computation. During held,
            ``fence`` and ``metadata`` describe its retained baseline, while
            this field describes the actual held phase that selected that view.
    """

    metadata: SnapshotMetadata
    query_time: int
    fence: ScopeFence
    embedding_space: str
    embedding_token: IntegrityDigest
    recall_policy: RecallPolicySelection
    _grants_json: str = field(repr=False)
    finalize_phase: FinalizePhase | None = None

    @property
    def allowed_scopes(self) -> FrozenScopeGrantSetV1:
        """Return a strict detached copy of the frozen singleton authorization."""
        return FrozenScopeGrantSetV1.model_validate_json(self._grants_json)

    @property
    def embedding_state(self) -> Literal["active"]:
        """Return the only state admitted by the current snapshot constructor.

        ``embedding_state_token`` rejects all non-active Control variants before
        a StoreReadModel root can be published. Migration remains unsupported.
        """
        return "active"


@dataclass(frozen=True, slots=True)
class SearchExecution:
    """Content-free attestation of one actual guarded recall invocation.

    Construction grants nothing. The issuing guard retains exact object identity
    and a keyed binding of the complete query, grants and ordered returned tuple.
    No row, query text or response is retained by this value.
    """

    metadata: SnapshotMetadata
    finalize_phase: FinalizePhase | None
    binding: ContentBindingDigest


def _search_time_grant(raw: PrincipalGrant, query_time: int) -> None:
    if type(query_time) is not int or not 0 <= query_time <= 2**63 - 1:
        raise QuoteMemError("schema_invalid", "Search time must be a timestamp")
    scope = raw.requested_scope
    if (
        not raw.issued_at <= query_time < raw.expires_at
        or len(raw.scope_grants) != 1 or raw.scope_grants[0].relation != "exact"
        or raw.scope_grants[0].scope_kind != scope.scope_kind
        or raw.scope_grants[0].scope_id != scope.scope_id
    ):
        raise QuoteMemError("scope_denied", "Search requires a valid exact singleton grant")


def _initial_search(root: RecallSnapshot, raw: PrincipalGrant, query_time: int) -> SearchContext:
    fence = root.scope_fence(raw.requested_scope.scope_id)
    frozen = FrozenScopeGrantSetV1(
        principal_id=raw.principal_id, requested_scope=raw.requested_scope,
        grants=[ScopeGrantV1(
            **raw.requested_scope.model_dump(), relation="exact",
            erase_epoch=fence.erase_epoch, consent_epoch=fence.consent_epoch,
        )], issued_at=raw.issued_at, expires_at=raw.expires_at,
    )
    policy = root.recall_policy
    checked = RecallPolicySelection.from_snapshot(
        policy.release_ref, policy.config_digest, policy.policy_snapshot(),
        policy.policy_snapshot_digest,
    )
    return SearchContext(
        SnapshotMetadata(root.generation, root.root_identity), query_time, fence,
        root.embedding_space, root.embedding_token, checked, canonical_json(frozen),
        fence.finalize_phase,
    )


def _search_bundle(context: SearchContext, bundle: QueryBundleV1) -> QueryBundleV1:
    """Freeze and bind query provenance outside the final Writer critical section."""
    if not isinstance(cast(object, bundle), QueryBundleV1):
        raise QuoteMemError("schema_invalid", "Search query requires its strict bundle model")
    try:
        frozen = QueryBundleV1.model_validate(bundle.model_dump(mode="json"))
    except ValidationError:
        raise QuoteMemError("schema_invalid", "Search query bundle is malformed") from None
    if (
        frozen.query_time != context.query_time
        or frozen.policy_release_ref != context.recall_policy.release_ref
        or canonical_json(frozen.policy_snapshot) != context.recall_policy.snapshot_json
        or frozen.embedding_space != context.embedding_space
        or frozen.embedding_epoch != context.embedding_token
    ):
        raise QuoteMemError("binding_mismatch", "Search query differs from its pinned authority")
    return frozen


def _search_grant(grant: PrincipalGrant) -> PrincipalGrant:
    """Detach strict epoch-free provider authority before entering QuoteMem locks."""
    if not isinstance(cast(object, grant), PrincipalGrant):
        raise QuoteMemError("schema_invalid",
                            "Search authorization requires its strict grant model")
    try:
        # Do not let a declared-field serializer discard injected epoch or
        # authority fields on a mutable provider DTO before strict validation.
        payload = dict(grant.__dict__)
        payload.update(grant.__pydantic_extra__ or {})

        def scope_payload(value: object) -> object:
            if not isinstance(value, ScopeV1):
                return value
            fields = dict(value.__dict__)
            fields.update(value.__pydantic_extra__ or {})
            return fields

        payload["requested_scope"] = scope_payload(payload.get("requested_scope"))
        scopes = payload.get("scope_grants")
        if isinstance(scopes, list):
            payload["scope_grants"] = [scope_payload(scope) for scope in cast(list[object], scopes)]
        return PrincipalGrant.model_validate(payload)
    except ValidationError:
        raise QuoteMemError("schema_invalid", "Search authorization is malformed") from None


def _require_outside_writer(writer: Writer) -> None:
    # CPython's RLock exposes the ownership hook used by threading.Condition.
    is_owned = cast(Callable[[], bool], attrgetter("_is_owned")(writer.mutex))
    if is_owned():
        raise QuoteMemError("reader_busy", "Reader drain cannot wait under the Writer mutex")


def _source_projection(evidence: EvidenceRowV1) -> SourceRowV2:
    """Project one committed Evidence row as the root holds it (``SourceRowV2``)."""
    return SourceRowV2.from_evidence(evidence.model_dump(mode="json"))


def _check_scope(
    root: RecallSnapshot, scope: ScopeV1, *, finalize: FinalizeBarrier | None = None,
) -> None:
    fence = root.scope_fence(scope.scope_id)
    if fence.scope_kind != scope.scope_kind or fence.tomb:
        raise QuoteMemError("scope_denied", "Snapshot lacks an available exact scope")
    if fence.quarantined:
        raise QuoteMemError("scope_quarantined", "The scope requires explicit recovery repair")
    if (
        fence.visibility_barrier or fence.active_erasure or fence.maintenance
        or (fence.finalize_barrier is not None and fence.finalize_barrier != finalize)
    ):
        raise QuoteMemError("index_rebuilding", "Snapshot scope has an active visibility fence")


def _check_quarantined_scope(root: RecallSnapshot, scope: ScopeV1) -> None:
    """Refuse a quarantined scope's Search only on its privacy fences.

    The quarantine and the visibility barrier it raises fence Add work alone; a
    tomb, an erasure or a pending privacy operation still refuses the read, as
    ``_check_scope`` refuses it for any scope.
    """
    fence = root.scope_fence(scope.scope_id)
    if fence.scope_kind != scope.scope_kind or fence.tomb:
        raise QuoteMemError("scope_denied", "Snapshot lacks an available exact scope")
    if fence.active_erasure or fence.maintenance:
        raise QuoteMemError("index_rebuilding", "Snapshot scope has an active visibility fence")


@dataclass(slots=True, weakref_slot=True)
class _ReaderState:
    root: RecallSnapshot | None
    metadata: SnapshotMetadata
    scopes: frozenset[tuple[ScopeKind, str]]
    publication: int
    finalize_input: tuple[str, FinalizeBarrier] | None = None
    finalize_publication: tuple[str, FinalizeBarrier] | None = None
    completions: bool = False
    finalize_search: tuple[str, FinalizeBarrier] | None = None
    quarantine_search: bool = False
    """A Search of a quarantined scope: it reads the scope's frozen committed view
    (``StoreReadModel._quarantine_search_root``), never the live root."""
    cancellation: threading.Event = field(default_factory=threading.Event)
    buffers: list[bytearray] = field(default_factory=list[bytearray])
    executing: dict[int, int] = field(default_factory=dict[int, int])
    operations: dict[tuple[int, ScopeKind, str], int] = field(
        default_factory=dict[tuple[int, ScopeKind, str], int]
    )


@dataclass(slots=True)
class _FinalizeBaseline:
    """One held cycle's retained input view: its own scope only (``RecallSnapshot.scope_view``).

    The owner's input borrow and the scope's held Searches read ``root``;
    ``corpora`` is the held Searches' shared corpus cache of that root.
    """

    barrier: FinalizeBarrier
    root: RecallSnapshot
    phase: Literal["captured", "held", "failed", "marked"] = "captured"
    corpora: SnapshotRecallCache | None = None


@dataclass
class _SearchView:
    """One scope's Search view of the published root (``RecallSnapshot.search_view``).

    ``root`` holds that scope only, so another scope's privacy drain never
    revokes a Search reading it. ``source`` is the published root it
    equals for the scope: a publication that leaves the scope untouched
    carries it (``StoreReadModel._carry_search_views``), any other drops it.
    ``corpora`` is its Searches' shared corpus cache.
    """

    source: RecallSnapshot
    root: RecallSnapshot
    corpora: SnapshotRecallCache | None = None
    adopted: _FinalizeBaseline | None = None
    """The committed view a Search reads while boot's adopted cycle holds the
    scope without a baseline (``StoreReadModel._adopted_search_baseline``)."""


@dataclass
class _QuarantineView:
    """The committed view a quarantined scope's Searches read.

    A quarantine fences the scope's Add work until an explicit repair, never its
    reads: its Searches keep reading what they read just before the fault. That
    is the held cycle's retained baseline when one held the scope, so no commit
    of the refused cycle is ever shown, and otherwise the scope's committed
    state in the published root. ``root`` holds that one scope with its fence
    open (no quarantine, barrier or finalize phase), so the pure recall and
    delivery checks read it as an ordinary committed scope; ``corpora`` is its
    Searches' shared corpus cache. Every privacy change, an embedding change
    and the repair drop it (``StoreReadModel._quarantine_view_holds``).
    """

    root: RecallSnapshot
    corpora: SnapshotRecallCache | None = None


def _opened_scope_view(root: RecallSnapshot, scope_id: str) -> RecallSnapshot:
    """``root``'s view of one scope with every Add fence lifted, for a quarantine's Searches.

    The privacy fields (epochs, tomb, erasure, maintenance and held Evidence)
    stay as they are; the caller refused a scope any of them closes.
    """
    return root.apply(generation=root.generation, scope_changes=(replace(
        root.scope_fence(scope_id), quarantined=False, visibility_barrier=False,
        finalized=None, finalize_barrier=None, finalize_phase=None,
    ),)).search_view(scope_id)


def _quarantine_view_holds(view: _QuarantineView, scope_id: str, fence: ScopeFence) -> bool:
    """Whether a quarantine's committed view still has its scope's privacy authority.

    It holds while the scope keeps its kind, epochs and outliving holds and no
    tomb, erasure or pending privacy operation fences it, as a retained finalize
    baseline holds (``StoreReadModel._discard_fenced_baselines``).
    """
    old = view.root.scope_fence(scope_id)
    return not (fence.tomb or fence.active_erasure or fence.maintenance) and (
        old.scope_kind, old.erase_epoch, old.consent_epoch, old.held_evidence,
    ) == (fence.scope_kind, fence.erase_epoch, fence.consent_epoch, fence.held_evidence)


_MAX_SEARCH_VIEWS = 8
"""The scopes whose Search views the read model retains, least recently used first out."""

_CARRIED_SCOPES = 4
"""The scopes whose baseline view row encodings are carried between finalizes.

One carrier holds its scope's encoded rows, about 24 KB per Evidence row of a
1,024-value vector: a BEAM-1M user's scope at its last Adds holds about 8,000.
A few scopes finalizing in turn each keep theirs; a forgotten carrier only
makes that scope's next view encode its rows again.
"""


class _ReaderCondition(threading.Condition):
    """The reader registry's condition, counting the sections that may mutate it.

    Every mutation of registered reader state (a root dropped, a cancellation,
    a delivery, a finalize borrow pinned, a retained baseline phased or
    discarded, a callback entered or left, a publication) runs inside a
    counted section of this condition: a ``with`` section, an explicit
    ``acquire``/``release`` pair, or the reacquisition after ``wait``.
    ``epoch`` advances once when such a section takes the lock and once more
    while it still holds the lock just before it gives it up, and a
    registry-level ``drop`` advances it too. A thread that recorded ``epoch``
    under the lock, outside any section of its own, and later reads the same
    value without the lock therefore knows that no counted section has
    started or ended since, hence that nothing the sections protect has
    changed: a concurrent section moved the value when it began, and the
    thread's own enclosing section, had there been one, moved it when it
    began and moves it again when it ends. ``observing`` sections hold the
    same lock without advancing it; ``SnapshotReadGuard.check_operation``,
    their only user, records nothing while this thread already owns the lock.
    The value is only ever written by the thread holding the lock, so it
    never moves backwards to a recorded value.
    """

    def __init__(self) -> None:
        lock = threading.RLock()
        super().__init__(lock)
        self._held = lock
        # Ownership is asked of the lock itself, whatever counts acquisitions.
        self._owner = lock
        self.epoch = 0
        # Condition exports the lock's own bound acquire and release as
        # instance attributes; drop them so every acquisition and release
        # runs the counted methods. A Condition that stops exporting them
        # already resolves to the methods below.
        self.__dict__.pop("acquire", None)
        self.__dict__.pop("release", None)

    def owned(self) -> bool:
        """Report whether the calling thread holds the lock, counted or not."""
        return cast(Callable[[], bool], attrgetter("_is_owned")(self._owner))()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        acquired = self._held.acquire(blocking, timeout)
        if acquired:
            self.epoch += 1
        return acquired

    def release(self) -> None:
        # Advance while still holding the lock; an unowned release raises
        # from the lock and must not touch the value without it.
        if self.owned():
            self.epoch += 1
        self._held.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *args: object) -> None:
        self.release()

    def wait(self, timeout: float | None = None) -> bool:
        # A waiter releases the lock and may mutate after reacquiring it,
        # which Condition.wait does on every exit once it released, including
        # an exception escaping the wait. An unowned wait raises before
        # releasing and stays uncounted.
        try:
            return super().wait(timeout)
        finally:
            if self.owned():
                self.epoch += 1

    @contextmanager
    def observing(self) -> Iterator[None]:
        """Hold the lock for a complete guard check without counting the section.

        The check may set a reader's cancellation event and, through
        ``_ReaderRegistry.drop``, retire that reader: the event is set first
        and is read directly by every lock-free repeat, and ``drop`` itself
        advances ``epoch`` so the root, buffers and registry membership it
        retires are never reachable through a record taken before it. Nothing
        else moves inside this section.
        """
        with self._held:
            yield


@dataclass(slots=True, weakref_slot=True)
class _ReaderRegistry:
    writer: Writer
    condition: _ReaderCondition = field(default_factory=_ReaderCondition)
    states: dict[int, _ReaderState] = field(default_factory=dict[int, _ReaderState])
    states_by_scope: dict[str, set[int]] = field(default_factory=dict[str, set[int]])
    publication: int = 0
    searches: dict[int, tuple[weakref.ReferenceType[SearchContext], bool]] = field(
        default_factory=dict[int, tuple[weakref.ReferenceType[SearchContext], bool]],
    )
    finalize_baselines: dict[str, _FinalizeBaseline] = field(
        default_factory=dict[str, _FinalizeBaseline],
    )
    search_views: dict[str, _SearchView] = field(default_factory=dict[str, _SearchView])
    quarantine_views: dict[str, _QuarantineView] = field(
        default_factory=dict[str, _QuarantineView],
    )

    def drop(self, state: _ReaderState) -> None:
        """Retire one registered reader; the caller holds ``condition``.

        The cancellation event is set before anything else so that a
        lock-free ``check_operation`` repeat, which reads only that event,
        can never see the root gone with the event clear; the epoch then
        advances so a record taken before this call, under a counted or an
        observing section, no longer passes anywhere.
        """
        if not self.condition.owned():
            raise QuoteMemError("reader_busy", "Dropping a reader requires its registry condition")
        state.cancellation.set()
        for buffer in state.buffers:
            buffer[:] = b"\x00" * len(buffer)
        state.buffers.clear()
        state.root = None
        identity = id(state)
        self.states.pop(identity, None)
        for _, scope_id in state.scopes:
            members = self.states_by_scope.get(scope_id)
            if members is not None:
                members.discard(identity)
                if not members:
                    del self.states_by_scope[scope_id]
        self.condition.epoch += 1

    def finalize_input_live(self, state: _ReaderState) -> bool:
        if state.finalize_publication is not None:
            scope_id, barrier = state.finalize_publication
            baseline = self.finalize_baselines.get(scope_id)
            return baseline is not None and baseline.phase == "held" and (
                baseline.barrier is barrier and state.root is not None
                and state.root.scope_fence(scope_id).finalize_barrier == barrier
            )
        if state.finalize_input is None:
            return True
        scope_id, barrier = state.finalize_input
        baseline = self.finalize_baselines.get(scope_id)
        return baseline is not None and baseline.phase == "held" and (
            baseline.barrier is barrier and baseline.root is state.root
        )


_EXACT_SCOPE_CAPACITY = 4096
_EXACT_SCOPES: dict[tuple[str, str], ScopeV1] = {}
_CHECKED_THREAD_CAPACITY = 64


def _privacy_row_scope(writer: Writer, kind: str, row: Mapping[str, Any]) -> tuple[Any, Any]:
    """The exact scope of a privacy batch's row; a revision has its memory's."""
    if kind == "revision":
        item = writer.get_row("item", str(row.get("memory_id")))
        return (None, None) if item is None else (item["scope_kind"], item["scope_id"])
    return row.get("scope_kind"), row.get("scope_id")


def _exact_scope(scope: ScopeV1) -> ScopeV1:
    """Return the validated exact ``ScopeV1`` for a scope's two field values.

    Every guarded operation re-validates its scope before using it as a key,
    which is a pure function of ``scope_kind`` and ``scope_id`` for an exact
    ``ScopeV1``. The finalize consumers call this between every borrowed row
    (millions of times per load run), so the validated instance is
    memoized per field pair; a subclass instance keeps the original
    re-validation, which its extra fields reject. The memo never mutates an
    entry and is bounded by a clear when full.
    """
    if type(scope) is not ScopeV1:
        return ScopeV1.model_validate(scope.model_dump())
    key = scope.scope_kind, scope.scope_id
    exact = _EXACT_SCOPES.get(key)
    if exact is None:
        exact = ScopeV1.model_validate(scope.model_dump())
        if len(_EXACT_SCOPES) >= _EXACT_SCOPE_CAPACITY:
            _EXACT_SCOPES.clear()
        _EXACT_SCOPES[key] = exact
    return exact


_PUBLISHERS_LOCK = threading.Lock()
_PUBLISHERS: weakref.WeakKeyDictionary[
    Writer,
    tuple[weakref.ReferenceType[StoreReadModel], weakref.ReferenceType[_ReaderRegistry]],
] = weakref.WeakKeyDictionary()

_BOOT_ROOTS_LOCK = threading.Lock()
_BOOT_ROOTS: weakref.WeakKeyDictionary[Writer, tuple[RecallSnapshot, int]] = (
    weakref.WeakKeyDictionary()
)
"""The step-17 cold root of each Writer, with the ``storage_epoch`` it was built at.

The first ordinary publisher adopts it instead of projecting every committed
row a second time (``StoreReadModel.build_cold_root(..., retain=True)``). Guarded by
``_BOOT_ROOTS_LOCK``, a leaf lock taken under the Writer mutex."""


class SnapshotReadGuard:
    """Own a snapshot and callback lifetime without exporting its bare pointer.

    Callbacks stop when cancelled; read-route callbacks never acquire the Writer.
    Registered workflow callbacks may acquire it only for bounded admission or writes. The
    runtime clears its own roots and zeroes registered mutable buffers after
    callbacks drain. Callbacks must not retain borrowed rows in unregistered
    internal caches. Previously delivered caller-owned responses are outside
    this retrospective ownership boundary.

    Args:
        readers: The publisher-owned reader registry.
        root: The immutable root atomically selected during pin.
        scopes: Exact scope identities authorized by the caller.
    """

    def __init__(
        self, readers: _ReaderRegistry, root: RecallSnapshot, scopes: tuple[ScopeV1, ...]
    ) -> None:
        """Register one reader in its publisher-owned lifetime registry.

        Args:
            readers: The registry protected by the publisher's reader condition.
            root: The atomically selected complete immutable root.
            scopes: Exact scope identities checked before construction.

        Returns:
            None.

        Raises:
            None.
        """
        self._readers = readers
        self._state = _ReaderState(
            root,
            SnapshotMetadata(root.generation, root.root_identity),
            frozenset((scope.scope_kind, scope.scope_id) for scope in scopes),
            readers.publication,
        )
        self._search_context: SearchContext | None = None
        self.metadata = self._state.metadata
        # Per thread: the exact scope key, reader-condition epoch and owner
        # registration at which this thread's last complete check_operation
        # passed. Each thread writes only its own entry.
        self._checked: dict[int, tuple[
            tuple[ScopeKind, str], int,
            tuple[weakref.ReferenceType[StoreReadModel], weakref.ReferenceType[_ReaderRegistry]],
        ]] = {}
        readers.states[id(self._state)] = self._state
        for _, scope_id in self._state.scopes:
            readers.states_by_scope.setdefault(scope_id, set()).add(id(self._state))

    @property
    def reads_baseline(self) -> bool:
        """Whether this Search guard reads a frozen view rather than the live root.

        A held cycle's retained baseline and a quarantined scope's committed view
        (``StoreReadModel._quarantine_search_root``) are both frozen: no commit
        reaches them, so each chain starts at the view's own head.
        """
        return self._state.finalize_search is not None or self._state.quarantine_search

    @property
    def is_cancelled(self) -> bool:
        """Report cancellation without acquiring the Writer mutex.

        Args:
            None.

        Returns:
            True after revocation or release, permanently.

        Raises:
            None.
        """
        return self._state.cancellation.is_set()

    def wait_cancelled(self, timeout: float) -> bool:
        """Wait for revocation or release without exposing a mutable event.

        Args:
            timeout: Finite nonnegative maximum wait in seconds. Zero is a
                nonblocking check. A separate application cancellation monitor
                may call this while another thread owns a scoped operation.

        Returns:
            True when cancelled or released; False when the timeout expires.

        Raises:
            QuoteMemError: If the timeout is invalid or the caller owns the Writer
                mutex. This wait acquires no reader lock and invokes no callback.
        """
        if (
            isinstance(timeout, bool) or not isinstance(cast(object, timeout), int | float)
            or not math.isfinite(timeout) or timeout < 0
        ):
            raise QuoteMemError("schema_invalid",
                                "Cancellation timeout must be finite and nonnegative")
        _require_outside_writer(self._readers.writer)
        return self._state.cancellation.wait(timeout)

    def _live_root(self, scope: ScopeV1 | None = None) -> RecallSnapshot:
        if not self._readers.finalize_input_live(self._state):
            self._state.cancellation.set()
            if not self._state.executing:
                self._drop()
        if self._state.root is None or self.is_cancelled:
            raise QuoteMemError("index_rebuilding", "Snapshot guard was revoked or released")
        if scope is not None and (scope.scope_kind, scope.scope_id) not in self._state.scopes:
            raise QuoteMemError("scope_denied", "Snapshot guard does not cover this exact scope")
        return self._state.root

    def track_buffer(self, value: bytes | bytearray | str) -> bytearray:
        """Copy bytes into a guard-owned hydration or provider buffer.

        Args:
            value: Input bytes or UTF-8 text for this operation.

        Returns:
            A mutable buffer that release or cancellation zeroes after callbacks drain.

        Raises:
            QuoteMemError: If the guard has been revoked or released.
        """
        with self._readers.condition:
            self._live_root()
            buffer = bytearray(value.encode("utf-8") if isinstance(value, str) else value)
            self._state.buffers.append(buffer)
            return buffer

    @contextmanager
    def scoped_buffer(self, value: bytes | bytearray | str) -> Iterator[bytearray]:
        """Own one bounded call's mutable bytes within an executing callback.

        Args:
            value: The bounded request or hydration copy for this call.

        Yields:
            A tracked buffer, zeroed and unregistered on every exit. Equal bytes
            in another active buffer do not share ownership or release together.

        Raises:
            QuoteMemError: If the guard is revoked or the current thread has no
                executing callback. Cancellation still drains that callback.
        """
        with self._readers.condition:
            self._live_root()
            if threading.get_ident() not in self._state.executing:
                raise QuoteMemError("reader_busy", "A scoped buffer requires an executing callback")
            buffer = self.track_buffer(value)
        try:
            yield buffer
        finally:
            with self._readers.condition:
                buffer[:] = b"\x00" * len(buffer)
                for index, owned in enumerate(self._state.buffers):
                    if owned is buffer:
                        self._state.buffers.pop(index)
                        break

    def consume_operation(self, scope: ScopeV1, consumer: Callable[[], None]) -> None:
        """Register a scoped workflow's hydrated content until its callback returns.

        Storage and provider operations run outside the reader condition. A
        workflow may acquire the sole Writer for a bounded admission or commit;
        privacy publication only cancels callbacks under that mutex, and joins
        them afterward outside it. The callback must check cancellation before
        external calls and writes, and must not export its borrowed content.
        """
        exact = _exact_scope(scope)
        self._consume(exact, lambda _root: None, lambda _value: consumer())

    def _publication_barrier(self, scope: ScopeV1) -> FinalizeBarrier | None:
        selected = self._state.finalize_publication
        return selected[1] if selected is not None and selected[0] == scope.scope_id else None

    def _read_barrier(self, root: RecallSnapshot, scope: ScopeV1) -> FinalizeBarrier | None:
        """The finalize barrier this borrow reads through (``StoreReadModel.pin``).

        A completion read passes whatever owner holds the scope in its pinned
        root; any other borrow passes only its own publication's barrier.
        """
        if self._state.completions:
            return root.scope_fence(scope.scope_id).finalize_barrier
        return self._publication_barrier(scope)

    def check_operation(self, writer: Writer, scope: ScopeV1) -> None:
        """Require this thread's active exact-scope borrow from the registered owner.

        Args:
            writer: The sole Writer whose publisher registered this guard.
            scope: The exact scope of the current content-bearing workflow.

        Raises:
            QuoteMemError: If the owner differs, the guard is cancelled or released,
                or this thread has no active callback for this exact scope.

        Pinning several scopes or running a callback on another thread does not
        grant this thread a content lifetime. This check never exports the root.

        The finalize consumers run this between every borrowed row, millions
        of times per load. Once a complete check has passed on this
        thread for this exact scope, a repeat passes without acquiring any
        lock while nothing the complete check reads can have changed: the
        reader condition's epoch is the one recorded under the lock, so no
        counted section has started or ended since and no reader was dropped
        (``_ReaderCondition``); the owner registration for ``writer`` is the
        same pair of references and the owner is still alive; the
        cancellation event, the only state an uncounted section may set
        before advancing the epoch, is clear; and this thread's callback
        count for the scope, which only this thread changes, is still
        positive. The root, delivery marks, the retained finalize baseline,
        the borrow's finalize fields and scopes change only inside counted
        sections. Any difference runs the complete check.

        A record is written only by a thread that entered the condition for
        this check alone. A check nested inside a section this thread already
        holds runs the complete path and records nothing: the enclosing
        section may mutate after the nested check, and the epoch it advanced
        on entry, and advances again on exit, is what keeps the record this
        thread took before the section from passing during or after it.

        The lock-free reads are single dictionary lookups, a weak reference
        call and an event read; on the pinned CPython 3.12 GIL build each is
        atomic, ``cancellation`` only ever becomes set, this thread's
        ``operations`` entry is written only by this thread and a
        registration is replaced only under ``_PUBLISHERS_LOCK``, so reads
        taken at different instants compose. A free-threaded build would
        need this derivation redone.
        """
        thread = threading.get_ident()
        exact = _exact_scope(scope)
        key = exact.scope_kind, exact.scope_id
        state, condition = self._state, self._readers.condition
        checked = self._checked.get(thread)
        if checked is not None and (
            checked[0] == key and checked[1] == condition.epoch
            and _PUBLISHERS.get(writer) is checked[2] and checked[2][0]() is not None
            and not state.cancellation.is_set()
            and state.operations.get((thread, *key), 0) >= 1
        ):
            return
        with _PUBLISHERS_LOCK:
            registered = _PUBLISHERS.get(writer)
            matches = registered is not None and registered[0]() is not None and (
                registered[1]() is self._readers
            )
        if registered is None or not matches:
            raise QuoteMemError("binding_mismatch",
                                "Source guard belongs to another registered owner")
        # Ownership is read before the observing section, which this thread
        # holds inside it either way.
        nested = condition.owned()
        # The complete check mutates nothing but a cancellation, which every
        # unlocked repeat reads directly, and the drop that may follow it,
        # which advances the epoch; counting this section would make
        # concurrent owners invalidate each other's records on every row.
        with condition.observing():
            root = self._live_root(exact)
            _check_scope(root, exact, finalize=self._read_barrier(root, exact))
            if state.operations.get((thread, *key), 0) < 1:
                raise QuoteMemError("reader_busy",
                                    "Source access requires an active scoped callback")
            epoch = condition.epoch
        if nested:
            return
        if len(self._checked) >= _CHECKED_THREAD_CAPACITY:
            self._checked.clear()
        self._checked[thread] = key, epoch, registered

    def search_bundle(self, context: SearchContext, query: QueryBundleV1) -> QueryBundleV1:
        """Validate a detached query against this guard's frozen Search authority.

        Args:
            context: Exact context issued to this guard.
            query: Strict bundle with the pin's time, policy and embedding binding.

        Returns:
            A detached strict bundle, granting no independent read authority.

        Raises:
            QuoteMemError: For a foreign context, revoked guard, or changed bindings.
        """
        if context is not self._search_context:
            raise QuoteMemError("binding_mismatch", "Search context belongs to another guard")
        with self._readers.condition:
            self._live_root(context.allowed_scopes.requested_scope)
        return _search_bundle(context, query)

    def check_search_delivery(
        self, context: SearchContext, grant: PrincipalGrant,
        projections: tuple[DeliveryProjection, ...], *, mode: MemoryMode,
        at_time: int, speakers: Mapping[QuoteKey, str | None],
        as_of: int | None = None, knowledge: KnowledgeReadView | None = None,
    ) -> None:
        """Revalidate an active Search under the final sole-Writer critical section.

        A Search pinned on a held cycle's baseline is rechecked against that
        baseline, and only while the same cycle still holds the scope (a
        commit of the owner is never seen in part). This checks authority and
        bytes only. The application must still commit
        its recall row and enqueue under that same mutex and registered callback.

        Args:
            context: This guard's exact issued context.
            grant: Fresh provider grant, obtained outside every QuoteMem lock.
            projections: The presentations actually rendered at the earlier pin.
            mode: The original Search mode.
            at_time: Trusted delivery clock.
            speakers: Same-pin frozen speaker decisions, re-derived by the caller.
            as_of: Original explicit world instant.
            knowledge: Guard-owned reconstructed knowledge selection, when requested.

        Raises:
            QuoteMemError: For revoked, foreign, expired, changed or stale authority/content.
        """
        writer = self._readers.writer
        if not writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise QuoteMemError("reader_busy", "Search delivery check requires the Writer mutex")
        self._recheck_search_delivery(
            context, grant, projections, mode=mode, at_time=at_time, speakers=speakers,
            as_of=as_of, knowledge=knowledge,
        )

    def deliver_search_published(
        self, context: SearchContext, grant: PrincipalGrant,
        projections: tuple[DeliveryProjection, ...], *, mode: MemoryMode,
        at_time: int, speakers: Mapping[QuoteKey, str | None],
        as_of: int | None, enqueue: Callable[[], None],
    ) -> None:
        """Recheck an interactive Search against the published root and enqueue it, without
        the Writer mutex (plugin profile).

        The same checks as ``check_search_delivery`` run under the read model's
        condition, against the root the last commit published, and ``enqueue`` runs
        under that condition too. Publication swaps roots and cancels this guard
        under the same condition, so the response is enqueued only while the
        published root still holds every shown byte. A commit whose publication
        has not yet run is not seen: the Search is ordered before it. A privacy
        commit cancels this guard when it publishes and then joins it, so its
        response always follows this delivery. ``speakers`` are the labels the
        Search rendered with at its pin; the scope fence check proves the scope
        unchanged since. A knowledge read (``as_of`` mode ``knowledge``) checks the
        Writer's storage generation and keeps ``check_search_delivery``.

        Args:
            context: This guard's exact issued context.
            grant: Fresh provider grant, obtained outside every QuoteMem lock.
            projections: The presentations actually rendered at the earlier pin.
            mode: The original Search mode; never ``knowledge``.
            at_time: Trusted delivery clock.
            speakers: The speaker decisions the projections were rendered with.
            as_of: Original explicit world instant.
            enqueue: Nonblocking enqueue of the response; it must not reenter
                this read model or take the Writer mutex.

        Raises:
            QuoteMemError: For revoked, foreign, expired, changed or stale authority/content.
        """
        if mode == "knowledge":
            raise QuoteMemError("binding_mismatch", "A knowledge Search delivers under the Writer")
        self._recheck_search_delivery(
            context, grant, projections, mode=mode, at_time=at_time, speakers=speakers,
            as_of=as_of, knowledge=None, enqueue=enqueue,
        )

    def _recheck_search_delivery(
        self, context: SearchContext, grant: PrincipalGrant,
        projections: tuple[DeliveryProjection, ...], *, mode: MemoryMode,
        at_time: int, speakers: Mapping[QuoteKey, str | None],
        as_of: int | None, knowledge: KnowledgeReadView | None,
        enqueue: Callable[[], None] | None = None,
    ) -> None:
        writer = self._readers.writer
        if context is not self._search_context:
            raise QuoteMemError("binding_mismatch", "Search delivery belongs to another guard")
        raw = _search_grant(grant)
        _search_time_grant(raw, at_time)
        frozen = context.allowed_scopes
        if raw.principal_id != frozen.principal_id or raw.requested_scope != frozen.requested_scope:
            raise QuoteMemError("scope_denied", "Search delivery changed its principal or scope")
        scope = frozen.requested_scope
        with self._readers.condition:
            self._live_root(scope)
            operation = threading.get_ident(), scope.scope_kind, scope.scope_id
            if not self._state.operations.get(operation):
                raise QuoteMemError("reader_busy", "Search delivery requires a registered callback")
            owner = StoreReadModel.get(writer)
            if owner is None or owner._readers is not self._readers:  # pyright: ignore[reportPrivateUsage]
                raise QuoteMemError("service_unhealthy", "Search publisher is unavailable")
            current = owner._live_publisher().current  # pyright: ignore[reportPrivateUsage]
            fence = current.scope_fence(scope.scope_id)
            held = self._state.finalize_search
            quarantine = self._state.quarantine_search
            barrier = None if held is None else held[1]
            if fence.finalize_barrier != barrier or fence.quarantined != quarantine:
                # A held Search delivers its baseline only while that cycle still
                # holds the scope, and a quarantined scope's Search its committed
                # view only while the quarantine lasts; any other change of either
                # is its one recompute.
                raise QuoteMemError("delivery_stale",
                                    "Search finalize cycle changed before delivery")
            if quarantine:
                _check_quarantined_scope(current, scope)
            else:
                _check_scope(current, scope, finalize=barrier)
            shown = (
                self._current_scope_root(scope.scope_id, current)
                if held is None and not quarantine else self._live_root(scope)
            )
            if (
                shown.scope_fence(scope.scope_id).scope_seq, fence.erase_epoch, fence.consent_epoch,
            ) != (
                context.fence.scope_seq, context.fence.erase_epoch, context.fence.consent_epoch,
            ) or (current.embedding_space, current.embedding_token) != (
                context.embedding_space, context.embedding_token,
            ):
                raise QuoteMemError("delivery_stale", "Search delivery authority changed")
            if (mode == "knowledge") != (knowledge is not None):
                raise QuoteMemError("binding_mismatch", "Search knowledge mode lacks its selection")
            if knowledge is not None and (
                # A held Search's knowledge reads its baseline's own chains, which
                # the owner's commits do not change; the cycle check above
                # binds it to that baseline.
                held is None and not quarantine
                and knowledge.storage_generation != writer.ref_generation
                or knowledge.scope_id != scope.scope_id or knowledge.recorded_as_of != as_of
            ):
                raise QuoteMemError("delivery_stale", "Knowledge changed before delivery")
            recheck_delivery(
                projections, shown if knowledge is None else knowledge.root,
                scope=scope, mode=mode, at_time=at_time, speakers=speakers, as_of=as_of,
            )
            if enqueue is not None:
                enqueue()

    def _current_scope_root(self, scope_id: str, current: RecallSnapshot) -> RecallSnapshot:
        """The scope's Search view while it equals ``current`` for that scope, else ``current``.

        The delivery recheck reads this root's corpus. The Search view holds the same
        rows of the scope and already has that corpus, while the published root
        would build it again for every Search: 20 s for a user of 59k memories.
        """
        view = self._readers.search_views.get(scope_id)
        return view.root if view is not None and view.source is current else current

    def consume_search(
        self, context: SearchContext, consumer: Callable[[RecallSnapshot], None],
    ) -> None:
        """Borrow an issued Search root inside a counted, revocable operation.

        Args:
            context: This guard's issued Search authority.
            consumer: Cooperative consumer; no borrowed content may escape.

        Raises:
            QuoteMemError: For forged authority, cancelled reads, or visibility fences.
        """
        if context is not self._search_context:
            raise QuoteMemError("binding_mismatch", "Search context belongs to another guard")
        scope = context.allowed_scopes.requested_scope
        self._consume(scope, lambda root: root, consumer)

    def consume_sources(
        self, scope: ScopeV1, query_time: int, consumer: Callable[[tuple[SourceRowV2, ...]], None]
    ) -> None:
        """Consume the frozen source route during a registered callback.

        Args:
            scope: An exact scope included at pin time.
            query_time: The operation's frozen TTL evaluation time.
            consumer: Cooperative callback that never blocks on the Writer mutex.

        Returns:
            None after the callback's borrowed content references are released.

        Raises:
            QuoteMemError: If scope authorization, cancellation, or a barrier rejects access.
        """
        exact = _exact_scope(scope)
        self._consume(exact, lambda root: root.source_rows(exact.scope_id, query_time), consumer)

    def consume_add_view(
        self, scope: ScopeV1, *, now: int, tombstones: TombstoneSource,
        consumer: Callable[[AddReadView], None], carry: AddViewCarry | None = None,
    ) -> None:
        """Consume the Add read view of one exact scope during a registered callback.

        The view reads this guard's root and the owner-private tombstone index;
        tombstones never enter the root.

        Args:
            scope: An exact scope included at pin time.
            now: The read's trusted clock in epoch milliseconds.
            tombstones: The owner's tombstone index of the same scope.
            consumer: Cooperative callback receiving the view. Neither the view
                nor anything it returns may outlive the callback.
            carry: The owner's records of earlier views of the scope, which the
                view reuses where their inputs are unchanged (`AddViewCarry`).

        Returns:
            None after the callback's borrowed content references are released.

        Raises:
            QuoteMemError: If scope authorization, cancellation, a barrier or the
                tombstone index's scope rejects access.
        """
        exact = _exact_scope(scope)
        self._consume(
            exact,
            lambda root: root.build_add_view(
                exact.scope_kind, exact.scope_id, now=now, tombstones=tombstones, carry=carry,
            ),
            consumer,
        )

    def _consume[T](
        self, scope: ScopeV1, select: Callable[[RecallSnapshot], T], consumer: Callable[[T], None],
    ) -> None:
        thread = threading.get_ident()
        operation = thread, scope.scope_kind, scope.scope_id
        with self._readers.condition:
            root: RecallSnapshot | None = self._live_root(scope)
            _check_scope(root, scope, finalize=self._read_barrier(root, scope))
            self._state.executing[thread] = self._state.executing.get(thread, 0) + 1
            self._state.operations[operation] = self._state.operations.get(operation, 0) + 1
        selected: T | None = None
        failure: QuoteMemError | None = None
        try:
            selected = select(root)
            root = None
            if self.is_cancelled:
                raise QuoteMemError("index_rebuilding", "Revoked reader cannot enter a consumer")
            consumer(selected)
        except Exception as error:
            # Do not export a traceback whose callback arguments retain borrowed content.
            # Keep only the innermost frame's file name, line and function: a
            # content-free coordinate that lets a retained failure reason name
            # the origin inside the callback (quotemem.core.failure_reason).
            # An inner guard already discarded its callback traceback. Preserve
            # that safe coordinate instead of naming the inner guard's re-raise.
            origin = error.origin_frame if isinstance(error, QuoteMemError) else None
            if origin is None:
                frames = extract_tb(error.__traceback__)
                if frames:
                    last = frames[-1]
                    origin = (Path(last.filename).name, last.lineno or 0, last.name)
            # Name that coordinate in the message as well. The frame is already
            # content-free, and a caller that sees only "Snapshot callback did
            # not complete" has to reproduce the run to learn where it came
            # from: two registrations once died here under different
            # codes and the origin each carried was never read.
            failure = QuoteMemError(
                error.code if isinstance(error, QuoteMemError) else (
                    "service_unhealthy" if is_runtime_failure(error) else "internal_error"
                ),
                "Snapshot callback did not complete" + (
                    "" if origin is None else f" at {origin[0]}:{origin[1]}:{origin[2]}"
                ),
            )
            failure.origin_frame = origin
        finally:
            # Release runtime-owned borrowed data before announcing the drain.
            selected = None
            root = None
            with self._readers.condition:
                self._state.executing[thread] -= 1
                if self._state.executing[thread] == 0:
                    del self._state.executing[thread]
                self._state.operations[operation] -= 1
                if self._state.operations[operation] == 0:
                    del self._state.operations[operation]
                if self.is_cancelled and not self._state.executing:
                    self._drop()
                self._readers.condition.notify_all()
        if failure is not None:
            raise failure
        if self.is_cancelled:
            raise QuoteMemError("index_rebuilding", "Cancelled snapshot result cannot be consumed")

    def _drop(self) -> None:
        self._checked.clear()
        self._readers.drop(self._state)

    def release(self) -> None:
        """Revoke access, drain callbacks, and scrub all owned buffers.

        Args:
            None.

        Returns:
            None once this guard no longer owns root or content references.

        Raises:
            QuoteMemError: If called from its own callback or under the Writer mutex.
        """
        _require_outside_writer(self._readers.writer)
        with self._readers.condition:
            self._state.cancellation.set()
            if threading.get_ident() in self._state.executing:
                raise QuoteMemError("reader_busy", "Release must occur outside the active callback")
            while self._state.executing:
                self._readers.condition.wait()
            self._drop()
            self._readers.condition.notify_all()

    def __enter__(self) -> Self:
        """Enter an active guard's bounded content lifetime.

        Args:
            None.

        Returns:
            This guard while it remains active.

        Raises:
            QuoteMemError: If it was already revoked or released.
        """
        with self._readers.condition:
            self._live_root()
        return self

    def __exit__(self, *_: object) -> None:
        """Release owned references and buffers on context exit.

        Args:
            _: Standard exception context supplied by Python.

        Returns:
            None; exceptions are not suppressed.

        Raises:
            QuoteMemError: If exit attempts to drain under the Writer mutex.
        """
        self.release()


class StoreReadModel:
    """Maintain one current root with fail-closed publication after durable writes.

    Args:
        writer: The sole initialized owner; cold reconstruction occurs before consumers start.
    """

    _closed: bool = False
    _withdrawn: bool = False
    _failed_at: tuple[int | None] | None = None
    _publisher: SnapshotPublisher | None
    _corpora: SnapshotRecallCache | None = None
    """The published root's corpus cache."""
    _unadopted_barred: bool = False
    """Boot step 13's publisher: bar every live cycle step 14 has not adopted yet."""

    def _live_publisher(self) -> SnapshotPublisher:
        if self._closed:
            raise QuoteMemError("service_unhealthy", "Snapshot publisher is permanently closed")
        if self._withdrawn:
            raise QuoteMemError(
                "service_unhealthy", "A repeated publication failure withdrew the read view",
            )
        publisher = self._publisher
        if publisher is None:
            raise QuoteMemError("service_unhealthy", "Snapshot publisher is permanently closed")
        return publisher

    def __init__(self, writer: Writer) -> None:
        """Register the sole publisher and reconstruct complete durable state.

        Args:
            writer: The initialized authoritative persistence owner.

        Returns:
            None after a complete root becomes available.

        Raises:
            QuoteMemError: If another publisher or unreleased registry already owns this Writer.
            ValueError: If strict durable row validation rejects reconstruction.
            RuntimeError: If the underlying HeavenBase workspace is unavailable.
        """
        self._initialize(writer)

    def _initialize(self, writer: Writer, *, session_recovery: bool = False) -> None:
        """Build one registered root from complete durable state."""
        self.writer = writer
        self._readers = _ReaderRegistry(writer)
        self._privacy_cutoff: dict[str, int] = {}
        self._quiesced: dict[str, int] = {}
        self._session_recovery = session_recovery
        self._cold_reconstruction = not session_recovery
        self._unhealthy = False
        with _PUBLISHERS_LOCK:
            previous = _PUBLISHERS.get(writer)
            if previous is not None:
                owner, registry = previous[0](), previous[1]()
                if owner is not None or registry is not None and registry.states:
                    raise QuoteMemError(
                        "service_unhealthy", "Writer already has its authoritative publisher"
                    )
            _PUBLISHERS[writer] = weakref.ref(self), weakref.ref(self._readers)
        try:
            with writer.mutex:
                rebuilt = None if session_recovery or self._unadopted_barred else (
                    self._adopt_boot_root()
                )
                if rebuilt is None:
                    rebuilt = self._rebuild()
                self._publisher = SnapshotPublisher(rebuilt)
                self._corpora = SnapshotRecallCache(rebuilt)
                self._cold_reconstruction = False
        except Exception:
            with _PUBLISHERS_LOCK:
                previous = _PUBLISHERS.get(writer)
                if previous is not None and previous[0]() is self:
                    del _PUBLISHERS[writer]
            raise

    def _adopt_boot_root(self) -> RecallSnapshot | None:
        """Take the Writer's retained step-17 root when no row has changed since it was built.

        ``build_cold_root(..., retain=True)`` leaves the root it built, with the
        Writer's ``storage_epoch`` read under the same mutex hold. An unchanged
        token proves that no row mutation reached the workspace since, and the
        boot holds exclusive storage ownership, so ``_rebuild`` would read the
        very rows that root was projected from: a cold root depends on nothing
        else (no previous root, no retained baseline, a fresh reader registry).
        The retained root is taken once and dropped either way.

        Returns:
            The retained root, or None when the caller must rebuild.
        """
        writer = self.writer
        with _BOOT_ROOTS_LOCK:
            retained = _BOOT_ROOTS.pop(writer, None)
        if retained is None:
            return None
        root, token = retained
        if writer.storage_epoch != token or root.generation != writer.get_counter("commit_seq"):
            return None
        return root

    @classmethod
    def for_session_recovery(cls, writer: Writer) -> StoreReadModel:
        """Build the temporary bootstrap borrow without consuming cold baselines.

        The bootstrap context closes this sole publisher before exposing the
        workspace. Active finalize scopes remain barred; unrelated session
        recovery can use ordinary validated borrows. Only the final service
        publisher performs the first held baseline CAS.
        """
        publisher = cls.__new__(cls)
        publisher._initialize(writer, session_recovery=True)
        return publisher

    @classmethod
    def for_forget_recovery(cls, writer: Writer) -> StoreReadModel:
        """Build boot step 13's publisher, which resumes the forgets a crash interrupted.

        Step 13 runs before step 14 adopts the interrupted Add owners, so
        a live (preparing or held) cycle still carries the previous Writer
        epoch and has no authority to project. This publisher bars each such
        scope instead of refusing the root: it serves no reader, and the
        service's own publisher projects the adopted cycles after step 14. A
        forget's own scope holds no live cycle (cancelled it).

        Args:
            writer: The sole Writer, after the policy load (step 12).

        Returns:
            The registered publisher; the caller drops it before step 14.
        """
        publisher = cls.__new__(cls)
        publisher._unadopted_barred = True
        publisher._initialize(writer)
        return publisher

    @classmethod
    def build_cold_root(cls, writer: Writer, *, retain: bool = False) -> RecallSnapshot:
        """Build a complete root from committed rows without registering a publisher.

        This is the root a new publisher builds at construction: no previous
        root, no retained finalize baseline, and the same refusals. The
        step-17 cold build reads it; the publisher's own construction and its
        recovery rebuild use the same code.

        Args:
            writer: The initialized sole owner.
            retain: Leave the root for the Writer's first ordinary publisher,
                which adopts it while no row has changed since;
                any later root, or a changed row, makes that publisher rebuild.

        Returns:
            The cold root at the current ``commit_seq``.

        Raises:
            QuoteMemError: If durable state cannot be projected.
        """
        builder = cls.__new__(cls)
        builder.writer = writer
        builder._readers = _ReaderRegistry(writer)
        builder._session_recovery = False
        builder._cold_reconstruction = True
        builder._publisher = None
        with writer.mutex:
            token = writer.storage_epoch
            root = builder._rebuild()
            with _BOOT_ROOTS_LOCK:
                if retain:
                    _BOOT_ROOTS[writer] = root, token
                else:
                    _BOOT_ROOTS.pop(writer, None)
            return root

    @classmethod
    def get(cls, writer: Writer) -> StoreReadModel | None:
        """Find the already registered authoritative publisher without constructing one.

        Args:
            writer: The exact persistence owner whose runtime publisher is needed.

        Returns:
            Its existing publisher, or None when no publisher or live reader registry exists.

        Raises:
            QuoteMemError: If the registration has lost its owner while retaining readers,
                or the registered publisher and reader registry disagree.
        """
        with _PUBLISHERS_LOCK:
            registration = _PUBLISHERS.get(writer)
            if registration is None:
                return None
            owner, readers = registration[0](), registration[1]()
            if owner is None:
                if readers is not None and readers.states:
                    raise QuoteMemError("service_unhealthy", "Publisher owner lost live readers")
                return None
            if (
                not isinstance(owner, cls)
                or owner.writer is not writer
                or owner._readers is not readers
            ):
                raise QuoteMemError("service_unhealthy",
                                    "Publisher registration lost owner identity")
            if owner._closed:
                raise QuoteMemError("service_unhealthy", "Snapshot publisher is permanently closed")
            return owner

    def _finalize_state(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int, *, allow_manual: bool = False,
        holds_released: bool = False,
    ) -> tuple[RecallSnapshot, ScopeFence, ControlRowV1, dict[str, Any]]:
        if not writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise QuoteMemError("reader_busy", "Finalize projection requires the Writer mutex")
        if writer is not self.writer or self.get(writer) is not self:
            raise QuoteMemError("binding_mismatch",
                                "Finalize projection has another registered owner")
        writer.validate_storage_owner()
        if self._unhealthy:
            raise QuoteMemError("service_unhealthy", "Finalize projection requires a healthy root")
        root = self._live_publisher().current
        # Also strictly validate the caller's content-free identity before any lookup.
        FinalizeBarrier(owner_run_id, lease_generation, writer_epoch,
                        root.generation, root.root_identity)
        fence = self._scope(scope.scope_kind, scope.scope_id)
        published = root.scope_fence(scope.scope_id)
        if holds_released:
            # The release's marker transaction removes the outliving holds of the
            # requests it completed; its marker publication carries the change.
            published = replace(published, held_evidence=fence.held_evidence)
        if root.generation != writer.get_counter("commit_seq") or published != fence:
            raise QuoteMemError("parity_mismatch",
                                "Finalize projection differs from current storage")
        if fence.tomb or fence.active_erasure or fence.maintenance or fence.visibility_barrier:
            raise QuoteMemError("scope_denied", "Finalize cannot retain a privacy-blocked scope")
        lease, owner = self._finalize_cycle(
            scope, owner_run_id, lease_generation, writer_epoch, fence,
            allow_manual=allow_manual,
        )
        return root, fence, lease, owner

    def _finalize_cycle(
        self, scope: ScopeV1, owner_run_id: str, lease_generation: int,
        writer_epoch: int, fence: ScopeFence, *, allow_manual: bool = False,
        allow_stopped: bool = False, rebase_pending: bool = False,
    ) -> tuple[ControlRowV1, dict[str, Any]]:
        writer = self.writer
        lease = self._finalize_control(f"lease:finalize:{scope.scope_id}")
        value = lease.value
        current_writer = self._finalize_control("lease:writer").value
        owner = DOMAIN_ADAPTERS["run"].validate_python(
            writer.get_row("run", owner_run_id),
        ).model_dump(mode="json")
        stopped = False
        if allow_stopped:
            from quotemem.store.owner_lease import owner_stopped

            stopped = owner_stopped(owner)
        admission = finalize_admission(
            principal_id=owner["principal_id"], scope=scope, cycle=lease_generation,
        )
        family, digest, _ = admission_identity(admission, writer.bind_content)
        if (
            not isinstance(
                value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1 | ReleasedFinalizeLeaseV1,
            )
            or not isinstance(current_writer, WriterLeaseV1)
            or current_writer.writer_epoch != writer_epoch
            or lease.updated_by_run_id != owner_run_id
            or (value.owner_run_id, value.lease_generation, value.writer_epoch)
            != (owner_run_id, lease_generation, writer_epoch)
            or (owner["object_id"], owner["scope_kind"], owner["scope_id"])
            != (owner_run_id, scope.scope_kind, scope.scope_id)
            or owner["run_kind"] != "finalize" or owner["target_kind"] != "scope"
            or owner["target_id"] != scope.scope_id or owner["commit_mode"] != "orchestrator"
            or owner["parent_run_id"] is not None or owner["run_generation"] != 0
            or isinstance(value, PreparingFinalizeLeaseV1) and owner["attempt"] != 0
            # An attempt after the first is an explicit resume of the frozen owner:
            # its selected input exists.
            or owner["attempt"] > 0 and owner["input_ref"] is None
            or owner["writer_epoch"] != writer_epoch
            or owner["status"] not in {"running", "succeeded"} and not (
                allow_manual and owner["status"] == "manual_attention"
                and owner["failure_code"] == "parity_mismatch"
            ) and not stopped
            or (owner["idempotency_key"], owner["admission_digest"]) != (family, digest)
            or not rebase_pending and (
                owner["admitted_erase_epoch"], owner["admitted_consent_epoch"],
            ) != (fence.erase_epoch, fence.consent_epoch)
        ):
            raise QuoteMemError("lease_lost", "Finalize projection lost its exact cycle authority")
        return lease, owner

    def _finalize_control(self, key: str) -> ControlRowV1:
        raw = self.writer.get_control(key)
        if raw is None:
            raise QuoteMemError("lease_lost", "Finalize projection lost its selected Control")
        control = ControlRowV1.model_validate(raw)
        if control.key != key:
            raise QuoteMemError("binding_mismatch", "Finalize Control locator differs")
        return control

    def _finalize_baseline(
        self, scope: ScopeV1, owner_run_id: str, lease_generation: int, writer_epoch: int,
        value: HeldFinalizeLeaseV1 | ReleasedFinalizeLeaseV1,
    ) -> _FinalizeBaseline:
        baseline = self._readers.finalize_baselines.get(scope.scope_id)
        if baseline is None or baseline.barrier != FinalizeBarrier(
            owner_run_id, lease_generation, writer_epoch,
            value.baseline_snapshot_generation, value.baseline_root_identity,
        ):
            raise QuoteMemError("lease_lost", "Finalize baseline is not retained by this process")
        current = self._live_publisher().current
        old, fresh = baseline.root.scope_fence(scope.scope_id), current.scope_fence(scope.scope_id)
        if (
            (old.scope_kind, old.erase_epoch, old.consent_epoch)
            != (fresh.scope_kind, fresh.erase_epoch, fresh.consent_epoch)
            or baseline.root.embedding_token != current.embedding_token
            or baseline.root.embedding_space != current.embedding_space
        ):
            raise QuoteMemError("lease_lost", "Finalize baseline lost its original scope authority")
        return baseline

    def publish_finalize_preparing(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> None:
        """Project the actual preparing CAS before the Writer releases its mutex.

        No baseline is selected here. Search may issue current-root metadata or
        read the complete current view, subject to normal relevant fences.
        """
        try:
            with self._readers.condition:
                root, fence, lease, owner = self._finalize_state(
                    writer, scope, owner_run_id, lease_generation, writer_epoch,
                )
                if (
                    not isinstance(lease.value, PreparingFinalizeLeaseV1)
                    or owner["status"] != "running" or fence.finalize_barrier is not None
                ):
                    raise QuoteMemError("lease_lost",
                                        "Preparing projection requires its actual lease")
                phase = FinalizePhase("preparing", owner_run_id, lease_generation, writer_epoch)
                if fence.finalize_phase != phase:
                    self._publish(root.apply(generation=root.generation, scope_changes=(
                        replace(fence, finalize_phase=phase),
                    )), (scope.scope_id,))
        except Exception:
            self._fail()
            raise

    def publish_finalize_failure(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> None:
        """Revoke pending Search after corrupt takeover or held release proof.

        The owner bookkeeping and this same-generation projection share one
        Writer critical section. Existing pure Search delivery then observes
        the barrier without querying the failed Run or exposing its details.
        """
        try:
            if not writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
                raise QuoteMemError("reader_busy", "Finalize failure requires Writer exclusion")
            if writer is not self.writer or self.get(writer) is not self or self._unhealthy:
                raise QuoteMemError("service_unhealthy",
                                    "Finalize failure lost its registered root")
            writer.validate_storage_owner()
            with self._readers.condition:
                root = self._live_publisher().current
                fence = self._scope(scope.scope_kind, scope.scope_id)
                lease, owner = self._finalize_cycle(
                    scope, owner_run_id, lease_generation, writer_epoch, fence, allow_manual=True,
                )
                value = lease.value
                if isinstance(value, PreparingFinalizeLeaseV1):
                    phase_matches = (
                        value.drain_state == "closed" and owner["stage"] == "lease"
                        and fence.finalize_barrier is None
                    )
                elif isinstance(value, HeldFinalizeLeaseV1):
                    baseline = self._finalize_baseline(
                        scope, owner_run_id, lease_generation, writer_epoch, value,
                    )
                    phase_matches = (
                        owner["stage"] == "release"
                        and owner["input_ref"] is not None and owner["result_ref"] is not None
                        and baseline.phase in {"held", "failed"}
                        and fence.finalize_barrier == baseline.barrier
                    )
                else:
                    phase_matches = False
                if (
                    not phase_matches or owner["status"] != "manual_attention"
                    or owner["failure_code"] != "parity_mismatch"
                    or fence.tomb or fence.active_erasure or fence.maintenance
                    or root.generation != writer.get_counter("commit_seq")
                    or replace(root.scope_fence(scope.scope_id), visibility_barrier=True) != fence
                ):
                    raise QuoteMemError("parity_mismatch", "Finalize failure projection differs")
                if isinstance(value, HeldFinalizeLeaseV1):
                    self._readers.finalize_baselines[scope.scope_id].phase = "failed"
                self._publish(root.apply(generation=root.generation, scope_changes=(fence,)),
                              (scope.scope_id,))
        except Exception:
            self._fail()
            raise

    def publish_finalize_cancelled(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> None:
        """Lift the barrier of a cycle a privacy rebase cancelled.

        The owner's lease CAS and its cancellation rows share the caller's Writer
        critical section. The erase already discarded the owner's baseline;
        any left is dropped, and the scope then has no finalize barrier or phase,
        as a cold root projects a cancelled cycle. The next admission proves the
        cancellation settled (``owner_lease.cancelled_settled``).
        """
        try:
            if not writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
                raise QuoteMemError("reader_busy", "A cancelled cycle requires Writer exclusion")
            if writer is not self.writer or self.get(writer) is not self or self._unhealthy:
                raise QuoteMemError("service_unhealthy",
                                    "A cancelled cycle lost its registered root")
            writer.validate_storage_owner()
            with self._readers.condition:
                root = self._live_publisher().current
                value = self._finalize_control(f"lease:finalize:{scope.scope_id}").value
                if not isinstance(value, CancelledFinalizeLeaseV1) or (
                    value.owner_run_id, value.lease_generation, value.writer_epoch,
                ) != (owner_run_id, lease_generation, writer_epoch) or (
                    root.generation != writer.get_counter("commit_seq")
                ):
                    raise QuoteMemError("parity_mismatch", "A cancelled cycle differs from storage")
                self._readers.finalize_baselines.pop(scope.scope_id, None)
                fence = replace(self._scope(scope.scope_kind, scope.scope_id),
                                finalize_barrier=None, finalize_phase=None)
                self._publish(root.apply(generation=root.generation, scope_changes=(fence,)),
                              (scope.scope_id,))
        except Exception:
            self._fail()
            raise

    def prepare_finalize_baseline(self, scope_id: str) -> None:
        """Restrict the current root to one scope before the Writer mutex is taken.

        ``capture_finalize_baseline`` retains the root's ``scope_view``, a pure
        function of the immutable root that digests every row of the scope. Built
        here, outside the Writer mutex, it is reused by the capture when the root the
        capture reads is this same object; any other root is restricted under the
        mutex as before. A failure here builds nothing and changes nothing.

        Args:
            scope_id: The scope an owner is about to hold.
        """
        try:
            root = self._live_publisher().current
            view = self._scope_view(root, scope_id)
        except Exception:  # noqa: BLE001 - the capture restricts the root itself
            return
        self._prepared().update({scope_id: (root, view)})

    def _scope_view(self, root: RecallSnapshot, scope_id: str) -> RecallSnapshot:
        """Restrict ``root`` to one scope, encoding only rows new since the last view.

        The view's identity digests every row of the scope with its vector, and
        every Add's finalize restricts the root again: a BEAM-1M user's 8,000
        Evidence rows, each dumped and its vector formatted, every Add. The row
        encodings the last view of the scope used are carried to the next
        (``CarriedRows``), under the scope's erasure and consent epochs: a
        change of either drops them, and a row that left the scope or changed is
        dropped by the next view. At most ``_CARRIED_SCOPES`` scopes are carried;
        a carrier is taken while in use.
        """
        fence = root.scope_fence(scope_id)
        epochs = fence.erase_epoch, fence.consent_epoch
        carriers = self._carriers()
        entry = carriers.pop(scope_id, None)
        carried = entry[1] if entry is not None and entry[0] == epochs else CarriedRows()
        with vector_encodings(), carried_rows(carried):
            view = root.scope_view(scope_id)
        while len(carriers) >= _CARRIED_SCOPES:
            carriers.pop(next(iter(carriers)), None)
        carriers[scope_id] = epochs, carried
        return view

    def _carriers(self) -> dict[str, tuple[tuple[int, int], CarriedRows]]:
        """The row encodings carried between scope views, by scope, least recent first."""
        carriers = self.__dict__.get("_carried_vectors")
        if carriers is None:
            carriers = self.__dict__.setdefault("_carried_vectors", {})
        return cast(dict[str, tuple[tuple[int, int], CarriedRows]], carriers)

    @contextmanager
    def hold_finalize_baseline(self, scope_id: str, *, prepare: bool) -> Iterator[None]:
        """Prepare one scope's baseline view for one Writer hold, never beyond it.

        The capture inside the hold consumes the view; a hold that ends without
        its capture discards it on exit, so no prepared view retains a root the
        publisher has since replaced.

        Args:
            scope_id: The scope an owner is about to hold.
            prepare: Whether the hold captures a baseline (a preparing owner).
        """
        if prepare:
            self.prepare_finalize_baseline(scope_id)
        try:
            yield
        finally:
            self._prepared().pop(scope_id, None)

    def _prepared(self) -> dict[str, tuple[RecallSnapshot, RecallSnapshot]]:
        """The scope views prepared outside the Writer mutex, by scope, with their root."""
        prepared = self.__dict__.get("_prepared_views")
        if prepared is None:
            prepared = self.__dict__.setdefault("_prepared_views", {})
        return cast(dict[str, tuple[RecallSnapshot, RecallSnapshot]], prepared)

    def capture_finalize_baseline(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> SnapshotMetadata:
        """Retain the scope's view of the root after the actual preparing drain closes.

        The baseline is the current root restricted to the owner's scope
        (``RecallSnapshot.scope_view``), and the barrier and lease name that
        view. The Writer must already hold its mutex through capture, held CAS
        and barrier publication. A repeated held identity returns the original
        metadata; later child publications can never replace that baseline.
        No content or baseline delivery capability leaves this method.
        """
        try:
            with self._readers.condition:
                root, _, lease, owner = self._finalize_state(
                    writer, scope, owner_run_id, lease_generation, writer_epoch,
                )
                value = lease.value
                existing = self._readers.finalize_baselines.get(scope.scope_id)
                if isinstance(value, HeldFinalizeLeaseV1) and owner["status"] == "running":
                    baseline = self._finalize_baseline(
                        scope, owner_run_id, lease_generation, writer_epoch, value,
                    )
                    return SnapshotMetadata(baseline.root.generation, baseline.root.root_identity)
                if (
                    not isinstance(value, PreparingFinalizeLeaseV1) or value.drain_state != "closed"
                    or owner["status"] != "running" or owner["stage"] != "lease"
                ):
                    raise QuoteMemError(
                        "lease_lost", "Baseline capture requires the closed preparing drain",
                    )
                prepared = self._prepared().pop(scope.scope_id, None)
                view = (prepared[1] if prepared is not None and prepared[0] is root
                        else self._scope_view(root, scope.scope_id))
                barrier = FinalizeBarrier(owner_run_id, lease_generation, writer_epoch,
                                          view.generation, view.root_identity)
                if existing is not None:
                    if existing.barrier != barrier or existing.phase != "captured":
                        raise QuoteMemError("lease_lost", "Finalize baseline cannot be replaced")
                else:
                    self._readers.finalize_baselines[scope.scope_id] = _FinalizeBaseline(
                        barrier, view,
                    )
                return SnapshotMetadata(view.generation, view.root_identity)
        except Exception:
            self._fail()
            raise

    def capture_cold_finalize_baseline(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> SnapshotMetadata:
        """Retain a fresh baseline for a held cycle that boot step 14 adopted.

        The cycle's original baseline died with its process, and its cold root
        keeps the lease's barrier closed. At the owner's first hold in this
        process the scope's view of the current root, with its barrier lifted,
        becomes its input baseline, and the scope's barrier moves to it in the
        same publication, so the scope never opens. The Writer then binds the lease
        to it (``owner_lease.hold_finalize``); it holds its mutex throughout.
        """
        try:
            with self._readers.condition:
                root, fence, lease, owner = self._finalize_state(
                    writer, scope, owner_run_id, lease_generation, writer_epoch,
                )
                value = lease.value
                if not isinstance(value, HeldFinalizeLeaseV1) or owner["status"] != "running" or (
                    fence.finalize_barrier != FinalizeBarrier(
                        owner_run_id, lease_generation, writer_epoch,
                        value.baseline_snapshot_generation, value.baseline_root_identity,
                    )
                ) or scope.scope_id in self._readers.finalize_baselines:
                    raise QuoteMemError("lease_lost",
                                        "A cold baseline requires its adopted held cycle")
                opened = root.apply(generation=root.generation, scope_changes=(
                    replace(fence, finalize_barrier=None),
                )).scope_view(scope.scope_id)
                baseline = _FinalizeBaseline(FinalizeBarrier(
                    owner_run_id, lease_generation, writer_epoch,
                    opened.generation, opened.root_identity,
                ), opened)
                baseline.phase = "held"
                self._readers.finalize_baselines[scope.scope_id] = baseline
                self._publish(root.apply(generation=root.generation, scope_changes=(
                    replace(fence, finalize_barrier=baseline.barrier),
                )), (scope.scope_id,))
                return SnapshotMetadata(opened.generation, opened.root_identity)
        except Exception:
            self._fail()
            raise

    def publish_finalize_held(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> None:
        """Publish the same-generation barrier only after the actual held CAS."""
        try:
            with self._readers.condition:
                root, fence, lease, owner = self._finalize_state(
                    writer, scope, owner_run_id, lease_generation, writer_epoch,
                )
                if not isinstance(lease.value, HeldFinalizeLeaseV1) or owner["status"] != "running":
                    raise QuoteMemError("lease_lost",
                                        "Finalize barrier requires an actual held lease")
                baseline = self._finalize_baseline(
                    scope, owner_run_id, lease_generation, writer_epoch, lease.value,
                )
                if baseline.phase == "marked":
                    raise QuoteMemError("lease_lost",
                                        "A finalized cycle cannot reinstall its barrier")
                baseline.phase = "held"
                phase = FinalizePhase("held", owner_run_id, lease_generation, writer_epoch)
                if fence.finalize_barrier != baseline.barrier or fence.finalize_phase != phase:
                    self._publish(root.apply(generation=root.generation, scope_changes=(
                        replace(fence, finalize_barrier=baseline.barrier, finalize_phase=phase),
                    )), (scope.scope_id,))
        except Exception:
            self._fail()
            raise

    def rebase_finalize_baseline(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> None:
        """Retain the current root's scope view as a held owner's baseline after its rebase.

        Only ``finalize_privacy.rebase_finalize_privacy`` calls this, under the
        Writer mutex, after its lease CAS named that view (``baseline_view``)
        and its Run write moved the owner to the scope's current epochs.
        A privacy publication discarded the old baseline because
        it retained erased content; the replacement is the post-erase
        view, never an older one. The caller then republishes the held barrier
        (``publish_finalize_held``). A retained baseline that still matches
        its scope authority is never replaced.
        """
        try:
            with self._readers.condition:
                root, _, lease, owner = self._finalize_state(
                    writer, scope, owner_run_id, lease_generation, writer_epoch,
                )
                value = lease.value
                view = root.scope_view(scope.scope_id)
                barrier = FinalizeBarrier(owner_run_id, lease_generation, writer_epoch,
                                          view.generation, view.root_identity)
                if (
                    not isinstance(value, HeldFinalizeLeaseV1) or owner["status"] != "running"
                    or (value.baseline_snapshot_generation, value.baseline_root_identity)
                    != (view.generation, view.root_identity)
                ):
                    raise QuoteMemError("lease_lost", "A rebased baseline requires its held lease")
                existing = self._readers.finalize_baselines.get(scope.scope_id)
                if existing is not None and existing.barrier != barrier:
                    old = existing.root.scope_fence(scope.scope_id)
                    fresh = root.scope_fence(scope.scope_id)
                    if (old.erase_epoch, old.consent_epoch) == (
                        fresh.erase_epoch, fresh.consent_epoch,
                    ):
                        raise QuoteMemError("lease_lost", "Finalize baseline cannot be replaced")
                    existing = None
                if existing is None:
                    self._readers.finalize_baselines[scope.scope_id] = _FinalizeBaseline(
                        barrier, view,
                    )
        except Exception:
            self._fail()
            raise

    def baseline_view(self, writer: Writer, scope: ScopeV1) -> SnapshotMetadata:
        """The identity of the scope's view of the current root, a rebased baseline's.

        ``finalize_privacy.rebase_finalize_privacy`` binds it into the held
        lease before ``rebase_finalize_baseline`` retains the same view; both
        run in one Writer critical section, so the root cannot move between.

        Raises:
            QuoteMemError: ``reader_busy`` outside the Writer mutex; ``scope_denied``
                for a scope the root does not project.
        """
        if not writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise QuoteMemError("reader_busy", "A baseline view requires the Writer mutex")
        with self._readers.condition:
            view = self._live_publisher().current.scope_view(scope.scope_id)
        return SnapshotMetadata(view.generation, view.root_identity)

    def validate_finalize_held(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> None:
        """Verify the current held barrier and its retained baseline before freeze.

        The caller holds the Writer mutex for this bounded fixed-state check.
        A durable held lease alone cannot authorize discovery after a failed
        publication, privacy-driven baseline disposal or lost publisher.
        """
        try:
            with self._readers.condition:
                _, fence, lease, owner = self._finalize_state(
                    writer, scope, owner_run_id, lease_generation, writer_epoch,
                )
                if not isinstance(lease.value, HeldFinalizeLeaseV1) or owner["status"] != "running":
                    raise QuoteMemError("lease_lost",
                                        "Finalize freeze requires its actual held cycle")
                baseline = self._finalize_baseline(
                    scope, owner_run_id, lease_generation, writer_epoch, lease.value,
                )
                if baseline.phase != "held" or fence.finalize_barrier != baseline.barrier:
                    raise QuoteMemError("lease_lost", "Finalize freeze lost its published baseline")
        except Exception:
            self._fail()
            raise

    def validate_finalize_publication_guard(
        self, writer: Writer, guard: SnapshotReadGuard, scope: ScopeV1,
    ) -> None:
        """Validate an active held publication borrow against its exact scope.

        Another scope may advance the global root while the original callback
        evaluates its frozen input. The original root, scope fences, policy,
        embedding identity and retained baseline remain authoritative; this
        check neither repins content nor authorizes a different owner.
        """
        with writer.mutex:
            with self._readers.condition:
                guard.check_operation(writer, scope)
                state = guard._state  # pyright: ignore[reportPrivateUsage]
                barrier = guard._publication_barrier(scope)  # pyright: ignore[reportPrivateUsage]
                if (
                    writer is not self.writer
                    or guard._readers is not self._readers  # pyright: ignore[reportPrivateUsage]
                    or self._readers.states.get(id(state)) is not state
                    or barrier is None
                ):
                    raise QuoteMemError("binding_mismatch",
                                        "Publication requires its issued held guard")
                current, fence, lease, owner = self._finalize_state(
                    writer, scope, barrier.run_id,
                    barrier.lease_generation, barrier.writer_epoch,
                )
                if not isinstance(lease.value, HeldFinalizeLeaseV1) or owner["status"] != "running":
                    raise QuoteMemError("lease_lost", "Publication lost its running held owner")
                baseline = self._finalize_baseline(
                    scope, barrier.run_id, barrier.lease_generation,
                    barrier.writer_epoch, lease.value,
                )
                original = guard._live_root(scope)  # pyright: ignore[reportPrivateUsage]
                if (
                    baseline.phase != "held" or baseline.barrier != barrier
                    or fence.finalize_barrier != barrier
                    or original.generation > current.generation
                    or original.scope_fence(scope.scope_id) != fence
                    or original.recall_policy != current.recall_policy
                    or (original.embedding_space, original.embedding_token)
                    != (current.embedding_space, current.embedding_token)
                ):
                    raise QuoteMemError("stale_publication",
                                        "Held publication dependencies changed")

    def pin_finalize_publication(self, ticket: FinalizeLeaseTicket) -> SnapshotReadGuard:
        """Pin the actual current root for the issued held owner's reviewed publication.

        This uses the same retained baseline and lease validation as input
        processing, while selecting current published memory for comparison.
        It never changes the Search baseline. Each group must obtain a new pin
        after its predecessor publishes. The private guard cannot issue Search
        authority, and all privacy fences remain effective.
        """
        return self._pin_finalize(ticket, publication=True)

    def pin_finalize_input(self, ticket: FinalizeLeaseTicket) -> SnapshotReadGuard:
        """Borrow the fixed held baseline for internal finalize input processing.

        Args:
            ticket: The unchanged held ticket issued by this publisher's Writer.
                Call outside the Writer mutex; issuance briefly acquires it.

        Returns:
            An exact-scope guard using the existing callback and buffer lifetime.
            It exposes no bare root and cannot issue or continue Search authority.
            Release it outside callbacks and the Writer mutex after processing.

        Raises:
            QuoteMemError: If ticket issuance, current owner/epochs, healthy publication
                or the exact retained held baseline/barrier cannot be proved.
                Privacy, failure, marker publication and release revoke access.

        This borrow does not prove complete input membership, current source TTL,
        provider permission or delivery. Those remain the consuming workflow's
        obligations. Later child publication never replaces the chosen baseline.
        """
        return self._pin_finalize(ticket, publication=False)

    def _pin_finalize(
        self, ticket: FinalizeLeaseTicket, *, publication: bool,
    ) -> SnapshotReadGuard:
        writer = self.writer
        _require_outside_writer(writer)
        if not isinstance(cast(object, ticket), FinalizeLeaseTicket):
            raise QuoteMemError("lease_lost", "Finalize input requires an issued held ticket")
        try:
            with writer.mutex:
                self._live_publisher()
                writer.validate_storage_owner()
                run, lease, _ = writer.read_finalize_fences(
                    ticket,
                )
                value = lease.value
                if not isinstance(value, HeldFinalizeLeaseV1):
                    raise QuoteMemError("lease_lost", "Finalize input requires the held cycle")
                scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])
                with self._readers.condition:
                    # Invalid caller tickets fail before the existing publisher
                    # validator, whose failures deliberately revoke the root.
                    self.validate_finalize_held(
                        writer, scope, value.owner_run_id,
                        value.lease_generation, value.writer_epoch,
                    )
                    baseline = self._readers.finalize_baselines[scope.scope_id]
                    root = self._live_publisher().current if publication else baseline.root
                    guard = SnapshotReadGuard(self._readers, root, (scope,))
                    state = guard._state  # pyright: ignore[reportPrivateUsage]
                    if publication:
                        state.finalize_publication = (scope.scope_id, baseline.barrier)
                    else:
                        state.finalize_input = (scope.scope_id, baseline.barrier)
                    return guard
        except ValidationError:
            raise QuoteMemError("schema_invalid", "Finalize input authority is malformed") from None

    def _finalize_marker(
        self, scope: ScopeV1, owner: dict[str, Any], lease_generation: int,
    ) -> FinalizedMarker:
        control = self._finalize_control(f"finalized:{scope.scope_id}")
        value = control.value
        # The durable selected input and canonical admission, not a caller's
        # digest/boolean, bind the marker to this exact completed owner cycle.
        if owner["input_ref"] is None:
            raise QuoteMemError("input_required",
                                "Finalize marker requires its selected owner input")
        envelope = self.writer.get_row("artifact", owner["input_ref"])
        if envelope is None or (
            envelope["object_id"], envelope["created_by_run_id"],
            envelope["scope_kind"], envelope["scope_id"],
        ) != (owner["input_ref"], owner["object_id"], scope.scope_kind, scope.scope_id):
            raise QuoteMemError("binding_mismatch", "Finalize input has another creator or scope")
        selected = self.writer.read_logical(owner["input_ref"], "run_input")
        _, digest, payload = admission_identity(finalize_admission(
            principal_id=owner["principal_id"], scope=scope, cycle=lease_generation,
        ), self.writer.bind_content)
        if (
            not isinstance(value, FinalizedValueV1)
            or not isinstance(selected, FinalizeOwnerRunInputV1)
            or (selected.business_key.scope_id, selected.business_key.finalize_cycle)
            != (scope.scope_id, lease_generation)
            or (selected.admission_digest, selected.payload_digest) != (digest, payload)
            or value.run_id != owner["object_id"] or value.lease_generation != lease_generation
            or control.updated_by_run_id != owner["object_id"]
            or value.input_heads_digest != selected.input_heads_digest
            or value.finalized_scope_seq != self.writer.get_counter(f"scope_seq:{scope.scope_id}")
        ):
            raise QuoteMemError("binding_mismatch",
                                "Finalize marker differs from selected owner input")
        return FinalizedMarker(**value.model_dump(mode="python"))

    def publish_finalize_marker(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> None:
        """Read the actual marker and remove the same-cycle barrier atomically.

        The sole Writer proves complete heads and effects before its marker
        write. This adapter verifies the selected durable poststate; it neither
        accepts a caller-issued marker proof nor creates finalization authority.
        """
        try:
            with self._readers.condition:
                root, fence, lease, owner = self._finalize_state(
                    writer, scope, owner_run_id, lease_generation, writer_epoch,
                    holds_released=True,
                )
                if not isinstance(lease.value, HeldFinalizeLeaseV1) or owner["status"] != "running":
                    raise QuoteMemError("lease_lost", "Finalize marker requires its held cycle")
                baseline = self._finalize_baseline(
                    scope, owner_run_id, lease_generation, writer_epoch, lease.value,
                )
                marker = self._finalize_marker(scope, owner, lease_generation)
                if baseline.phase not in {"held", "marked"}:
                    raise QuoteMemError("lease_lost", "Finalize marker lacks its published barrier")
                baseline.phase = "marked"
                if fence.finalize_barrier is not None or fence.finalized != marker or (
                    root.scope_fence(scope.scope_id).held_evidence != fence.held_evidence
                ):
                    self._publish(root.apply(generation=root.generation, scope_changes=(
                        replace(
                            fence, finalize_barrier=None, finalize_phase=None, finalized=marker,
                        ),
                    )), (scope.scope_id,))
        except Exception:
            self._fail()
            raise

    def release_finalize_baseline(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> None:
        """Drop the process pin after the matching durable lease release CAS."""
        try:
            with self._readers.condition:
                _, fence, lease, owner = self._finalize_state(
                    writer, scope, owner_run_id, lease_generation, writer_epoch,
                )
                if not isinstance(lease.value, ReleasedFinalizeLeaseV1):
                    raise QuoteMemError(
                        "lease_lost", "Baseline release requires the actual released lease",
                    )
                marker = self._finalize_marker(scope, owner, lease_generation)
                if fence.finalize_barrier is not None or fence.effective_finalized != marker:
                    raise QuoteMemError(
                        "parity_mismatch", "Released cycle lacks its published exact marker",
                    )
                if scope.scope_id in self._readers.finalize_baselines:
                    baseline = self._finalize_baseline(
                        scope, owner_run_id, lease_generation, writer_epoch, lease.value,
                    )
                    if baseline.phase != "marked":
                        raise QuoteMemError(
                            "parity_mismatch", "Released baseline did not publish its marker",
                        )
                    del self._readers.finalize_baselines[scope.scope_id]
        except Exception:
            self._fail()
            raise

    def begin_search(
        self, grant: PrincipalGrant, *, query_time: int,
        raw_user_id: str | None = None, clock: Callable[[], int] | None = None,
    ) -> SearchContext:
        """Issue Search metadata from the scope's readable view without exporting content.

        The original policy, grant epochs and embedding token all come from one
        atomic view: the current root, or, while an Add owner holds the scope,
        the cycle's retained baseline, so a Search sees the scope before
        the owner and never one of its intermediate commits. A quarantined scope
        is read from its frozen committed view (``_quarantine_search_root``):
        the quarantine fences its Add work, never its reads. Current visibility
        fences are enforced here and again by ``pin_search``. An already
        registered scope uses no Writer mutex, HB or provider. Only an absent
        scope, with its actual raw user ID and trusted clock supplied, enters
        the authorized registration producer and publishes its complete initial
        four-Control projection.
        """
        raw = _search_grant(grant)
        _search_time_grant(raw, query_time)
        if raw_user_id is not None and (
            type(raw_user_id) is not str or not raw_user_id
            or raw.requested_scope.scope_kind != "user"
            or raw.requested_scope.scope_id != canonical_scope_id("user", raw_user_id)
        ):
            raise QuoteMemError("scope_denied", "Search grant differs from its exact external user")
        with self._readers.condition:
            if self._search_scope_present(raw.requested_scope):
                current = self._live_publisher().current
                scope = raw.requested_scope
                committed = self._quarantine_search_root(current, scope)
                baseline = None if committed is not None else self._held_search_baseline(
                    current, scope,
                )
                if committed is not None:
                    context = _initial_search(committed, raw, query_time)
                elif baseline is None:
                    _check_scope(current, scope)
                    context = _initial_search(
                        self._search_view(current, scope.scope_id), raw, query_time,
                    )
                else:
                    _check_scope(current, scope, finalize=baseline.barrier)
                    context = replace(
                        _initial_search(baseline.root, raw, query_time),
                        finalize_phase=current.scope_fence(scope.scope_id).finalize_phase,
                    )
                self._issue_search(context)
                return context
        if raw_user_id is None or clock is None:
            raise QuoteMemError("scope_denied",
                                "Unregistered Search requires its actual user identity")
        self._register_search_scope(raw, raw_user_id, clock)
        return self.begin_search(raw, query_time=clock())

    def _issue_search(self, context: SearchContext) -> None:
        identity = id(context)
        def discard(_: weakref.ReferenceType[SearchContext]) -> None:
            with self._readers.condition:
                self._readers.searches.pop(identity, None)
        self._readers.searches[identity] = (weakref.ref(context, discard), False)

    def _search_view(self, current: RecallSnapshot, scope_id: str) -> RecallSnapshot:
        """The scope's Search view of the published ``current`` root.

        A Search reads its own scope only, so another scope's privacy drain never
        revokes it. The view is built once per change of the scope and
        shared by its Searches with its corpora. The caller holds the readers'
        condition.
        """
        views = self._readers.search_views
        view = views.pop(scope_id, None)
        if view is None or view.source is not current:
            view = _SearchView(current, current.search_view(scope_id))
        views[scope_id] = view
        while len(views) > _MAX_SEARCH_VIEWS:
            del views[next(iter(views))]
        return view.root

    def _carry_search_views(self, previous: RecallSnapshot, following: RecallSnapshot) -> None:
        """Carry each Search view whose scope ``following`` leaves untouched; drop the rest.

        A view stays only while the next root is ``previous``'s direct delta that
        changed none of the scope's projections or its fence; the caller holds
        the readers' condition.
        """
        views = self._readers.search_views
        for scope_id, view in tuple(views.items()):
            if view.source is previous and following.holds_scope(scope_id) and all(
                following.shares_corpus(previous, scope_id, kind=kind)
                for kind in ("memory", "entity", "evidence")
            ) and following.scope_fence(scope_id) == previous.scope_fence(scope_id):
                view.source = following
            else:
                del views[scope_id]

    def _held_search_baseline(
        self, current: RecallSnapshot, scope: ScopeV1,
    ) -> _FinalizeBaseline | None:
        """The retained baseline a Search of a held scope reads, or None when not held.

        The caller holds the readers' condition. A cycle that boot adopted has
        no baseline until its owner holds again in this process, which a stopped
        owner (``repair_required``) may never do: its Searches read the scope's
        committed state (``_adopted_search_baseline``). A failed cycle, or a held
        barrier this process neither retained nor adopted, stays
        ``index_rebuilding``.
        """
        fence = current.scope_fence(scope.scope_id)
        barrier, phase = fence.finalize_barrier, fence.finalize_phase
        if barrier is None:
            return None
        identity = (barrier.run_id, barrier.lease_generation, barrier.writer_epoch)
        if phase is None or phase.state != "held" or (
            phase.run_id, phase.lease_generation, phase.writer_epoch,
        ) != identity:
            raise QuoteMemError("index_rebuilding", "Held Search has no retained baseline")
        baseline = self._readers.finalize_baselines.get(scope.scope_id)
        if baseline is None:
            from quotemem.store.owner_lease import adopted_cycle

            if adopted_cycle(self.writer, scope.scope_id) == identity:
                return self._adopted_search_baseline(current, scope.scope_id, barrier)
        if baseline is None or baseline.phase != "held" or baseline.barrier != barrier:
            raise QuoteMemError("index_rebuilding", "Held Search has no retained baseline")
        return baseline

    def _adopted_search_baseline(
        self, current: RecallSnapshot, scope_id: str, barrier: FinalizeBarrier,
    ) -> _FinalizeBaseline:
        """The committed view a Search reads while an adopted cycle holds the scope.

        It is the view ``capture_cold_finalize_baseline`` binds at the owner's
        next hold: the scope of ``current`` with the barrier lifted. Until that
        hold the owner commits nothing, so this view is exactly the committed
        state, and its Searches deliver against the cycle's barrier like any
        held Search. The view is built once per change of the scope and leaves
        with the scope's Search view, so privacy and publication changes drop it.
        The caller holds the readers' condition.
        """
        self._search_view(current, scope_id)
        view = self._readers.search_views[scope_id]
        if view.adopted is None or view.adopted.barrier != barrier:
            opened = current.apply(generation=current.generation, scope_changes=(
                replace(current.scope_fence(scope_id), finalize_barrier=None),
            )).scope_view(scope_id)
            view.adopted = _FinalizeBaseline(barrier, opened, phase="held")
        return view.adopted

    def pin_search(self, context: SearchContext) -> SnapshotReadGuard:
        """Pin an issued request once; forged, reused or stale contexts fail closed.

        A context issued while an owner held the scope pins that cycle's
        retained baseline (``begin_search``); any finalize phase change since
        issuance is ``delivery_stale``, the caller's one recompute.

        Args:
            context: Exact content-free value issued by begin_search.

        Returns:
            A registered guard for the original scope and frozen authority.

        Raises:
            QuoteMemError: If issuance, scope, privacy or embedding authority changed.
        """
        with self._readers.condition:
            issued = self._readers.searches.get(id(context))
            if issued is None or issued[0]() is not context or issued[1]:
                raise QuoteMemError("binding_mismatch",
                                    "Search context is unissued or already pinned")
            scope = context.allowed_scopes.requested_scope
            current = self._live_publisher().current
            now = current.scope_fence(scope.scope_id)
            committed = self._quarantine_search_root(current, scope)
            if committed is not None:
                fence = committed.scope_fence(scope.scope_id)
                if context.finalize_phase is not None or (
                    committed.generation, committed.root_identity,
                    fence.scope_seq, now.erase_epoch, now.consent_epoch,
                ) != (
                    context.metadata.generation, context.metadata.root_identity,
                    context.fence.scope_seq, context.fence.erase_epoch,
                    context.fence.consent_epoch,
                ) or (current.embedding_space, current.embedding_token) != (
                    context.embedding_space, context.embedding_token,
                ):
                    # Issued before the quarantine, or before its view changed:
                    # one recompute reads the view it leaves (``begin_search``).
                    raise QuoteMemError("delivery_stale", "Search authority changed before pinning")
                self._readers.searches[id(context)] = (issued[0], True)
                guard = SnapshotReadGuard(self._readers, committed, (scope,))
                guard._search_context = context  # pyright: ignore[reportPrivateUsage]
                guard._state.quarantine_search = True  # pyright: ignore[reportPrivateUsage]
                return guard
            baseline = self._held_search_baseline(current, scope)
            phase = context.finalize_phase
            held = phase is not None and phase.state == "held"
            if (baseline is not None) != held or held and now.finalize_phase != phase:
                # An owner held or released the scope since issuance: one recompute
                # reads the view it leaves (``begin_search``).
                raise QuoteMemError("delivery_stale",
                                    "Search finalize phase changed before pinning")
            _check_scope(current, scope, finalize=None if baseline is None else baseline.barrier)
            root = (self._search_view(current, scope.scope_id) if baseline is None
                    else baseline.root)
            if (root.scope_fence(scope.scope_id).scope_seq, now.erase_epoch, now.consent_epoch) != (
                context.fence.scope_seq, context.fence.erase_epoch, context.fence.consent_epoch,
            ) or baseline is not None and (root.generation, root.root_identity) != (
                context.metadata.generation, context.metadata.root_identity,
            ) or (current.embedding_space, current.embedding_token) != (
                context.embedding_space, context.embedding_token,
            ):
                raise QuoteMemError("delivery_stale", "Search authority changed before pinning")
            self._readers.searches[id(context)] = (issued[0], True)
            guard = SnapshotReadGuard(self._readers, root, (scope,))
            guard._search_context = context  # pyright: ignore[reportPrivateUsage]
            if baseline is not None:
                guard._state.finalize_search = (  # pyright: ignore[reportPrivateUsage]
                    scope.scope_id, baseline.barrier,
                )
            return guard

    def _quarantine_search_root(
        self, current: RecallSnapshot, scope: ScopeV1,
    ) -> RecallSnapshot | None:
        """The committed view a quarantined scope's Searches read, or None when not quarantined.

        A quarantine (``owner_lease.quarantine_scope``) fences the scope's Add
        work until an explicit repair; its Searches go on reading what they
        read just before the fault (``_QuarantineView``): the refused held
        cycle's retained baseline, kept by ``publish_scope_quarantine``, or the
        scope's committed state in ``current``. A process that never retained
        the baseline (a restart, an adopted cycle) reads the committed state,
        as an adopted cycle's Searches do (``_adopted_search_baseline``). The
        quarantined owner's pending work is never in either: what it did not
        commit is not in any root. Privacy fences still refuse, and a view
        that lost its privacy authority is built again from ``current``. The
        caller holds the readers' condition.

        Raises:
            QuoteMemError: ``scope_denied`` for a tombstoned scope; ``index_rebuilding``
                while an erasure or a pending privacy operation fences it.
        """
        fence = current.scope_fence(scope.scope_id)
        if not fence.quarantined:
            return None
        _check_quarantined_scope(current, scope)
        views = self._readers.quarantine_views
        view = views.get(scope.scope_id)
        if view is None or view.root.embedding_token != current.embedding_token or (
            not _quarantine_view_holds(view, scope.scope_id, fence)
        ):
            view = views[scope.scope_id] = _QuarantineView(
                _opened_scope_view(current, scope.scope_id),
            )
        return view.root

    def _discard_stale_quarantine_views(
        self, fences: Mapping[str, ScopeFence], embedding_token: str,
    ) -> None:
        """Drop each quarantine view whose scope was repaired or lost its privacy authority.

        ``fences`` are the changed scopes' new fences; a view of an unchanged
        scope stays unless the embedding identity changed. The caller holds the
        readers' condition.
        """
        views = self._readers.quarantine_views
        for scope_id, view in tuple(views.items()):
            fresh = fences.get(scope_id)
            if view.root.embedding_token != embedding_token or fresh is not None and not (
                fresh.quarantined and _quarantine_view_holds(view, scope_id, fresh)
            ):
                del views[scope_id]

    def _search_scope_present(self, scope: ScopeV1) -> bool:
        """Distinguish a missing fence from an existing denied or held scope."""
        if self._unhealthy or self.get(self.writer) is not self:
            raise QuoteMemError("service_unhealthy",
                                "Search requires its registered healthy publisher")
        try:
            self._live_publisher().current.scope_fence(scope.scope_id)
        except QuoteMemError as error:
            if error.code != "scope_denied":
                raise
            return False
        return True

    def _register_search_scope(
        self, grant: PrincipalGrant, raw_user_id: str, clock: Callable[[], int],
    ) -> None:
        """Publish real initial Controls while retaining the sole Writer's admission."""
        writer, scope = self.writer, grant.requested_scope
        with writer.mutex:
            writer.validate_storage_owner()
            _search_time_grant(grant, clock())
            with self._readers.condition:
                if self._search_scope_present(scope):
                    _check_scope(self._live_publisher().current, scope)
                    return
            # A registration delta cannot cover a pre-existing data publication gap.
            _ = self.current
            generation = writer.ref_generation
            try:
                writer.register_scope(
                    scope.scope_kind, scope.scope_id, raw_user_id, clock(),
                    grant=grant, clock=clock,
                )
                _search_time_grant(grant, clock())
                self.publish_scope_registration(writer, scope, raw_user_id)
            except BaseException:
                if writer.ref_generation != generation:
                    self._fail()
                raise

    def publish_scope_registration(
        self, writer: Writer, scope: ScopeV1, raw_external_id: str,
    ) -> None:
        """Publish a newly registered scope before its first Run is admitted.

        The sole Writer supplies the actual initial five Controls. Existing
        scope roots remain unchanged; this cannot fill missing data, reset
        privacy fences or recapture a finalize baseline. Call immediately after
        registration within the same outer Writer critical section.
        """
        if writer is not self.writer:
            raise QuoteMemError("binding_mismatch", "Registration projection has another Writer")
        if not writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise QuoteMemError("reader_busy", "Registration projection requires the Writer mutex")
        writer.validate_storage_owner()
        with self._readers.condition:
            if self._search_scope_present(scope):
                if self._live_publisher().current.scope_fence(scope.scope_id).scope_kind != (
                    scope.scope_kind
                ):
                    raise QuoteMemError("binding_mismatch", "Registered scope kind differs")
                return
        _ = self.current
        rows = writer.read_initial_scope_registration(scope, raw_external_id)
        counters = tuple(row.value for row in rows[1:])
        if len(counters) != 4 or any(not isinstance(value, CounterV1) for value in counters):
            raise QuoteMemError("parity_mismatch", "Registration has another Control projection")
        sequence, erase, consent, _ = cast(
            tuple[CounterV1, CounterV1, CounterV1, CounterV1], counters,
        )
        fence = ScopeFence(
            scope_kind=scope.scope_kind, scope_id=scope.scope_id,
            scope_seq=sequence.seq, erase_epoch=erase.seq, consent_epoch=consent.seq,
        )
        try:
            with self._readers.condition:
                root = self._live_publisher().current
                self._publish(root.apply(
                    generation=root.generation, scope_changes=(fence,),
                ), (scope.scope_id,))
        except BaseException:
            self._fail()
            raise

    def is_scope_published(self, scope: ScopeV1) -> bool:
        """Report whether the current root already publishes a registered scope's fence.

        A registered user's later Adds ask this instead of entering the
        registering critical section: it takes the readers' condition but
        never the Writer mutex. ``True`` authorizes nothing;
        the caller's later admission re-proves the durable registration.

        Args:
            scope: Exact registered scope of the request.

        Returns:
            ``True`` when the current root carries this scope's fence with the
            requested kind; ``False`` when no fence exists yet.

        Raises:
            QuoteMemError: ``service_unhealthy`` if this publisher is unhealthy or is
                not the Writer's registered owner; ``binding_mismatch`` if the
                published scope kind differs from the request.
        """
        with self._readers.condition:
            if not self._search_scope_present(scope):
                return False
            if self._live_publisher().current.scope_fence(scope.scope_id).scope_kind != (
                scope.scope_kind
            ):
                raise QuoteMemError("binding_mismatch", "Registered scope kind differs")
            return True

    def pin(
        self, scopes: Sequence[ScopeV1], *, completions: bool = False,
    ) -> SnapshotReadGuard:
        """Atomically register a reader without waiting for the Writer mutex.

        A completion read (``completions``, the add-items) reads only the
        completion records and the rows they name, never the root's content, so
        an Add owner holding the scope does not fence it and that owner's
        publications do not revoke it: a completed Add stays readable while a
        later owner runs, pauses or is refused. Privacy fences,
        quarantine and health still refuse it.

        Args:
            scopes: Nonempty distinct exact identities already authorized by the caller.
            completions: Pin a completion read.

        Returns:
            A revocable guard owning one complete root and its scoped callbacks.

        Raises:
            QuoteMemError: If health, exact scope, or a visibility fence prevents pinning.
        """
        frozen = tuple(_exact_scope(scope) for scope in scopes)
        if not frozen or len({scope.scope_id for scope in frozen}) != len(frozen):
            raise QuoteMemError("scope_denied", "Snapshot pin requires distinct exact scopes")
        with self._readers.condition:
            if self._unhealthy:
                raise QuoteMemError("service_unhealthy", "Snapshot publication requires recovery")
            root = self._live_publisher().current
            for scope in frozen:
                _check_scope(root, scope, finalize=root.scope_fence(
                    scope.scope_id,
                ).finalize_barrier if completions else None)
            guard = SnapshotReadGuard(self._readers, root, frozen)
            guard._state.completions = completions  # pyright: ignore[reportPrivateUsage]
            return guard

    def recall_corpora(self, root: RecallSnapshot) -> SnapshotRecallCache:
        """Return the corpus cache of a root a reader's guard holds.

        The published root's cache is shared and carried from root to root, and
        a held cycle's Searches share their baseline's; a reader still holding
        an older root gets a cache of its own, so a corpus of one root never
        serves another. A cache grants nothing: every reader applies its mode's
        predicates at its own clock.

        Args:
            root: The root the reader's guard holds.

        Returns:
            The cache of that root.

        Raises:
            QuoteMemError: ``service_unhealthy`` when the publisher is closed or withdrawn.
        """
        with self._readers.condition:
            self._live_publisher()
            cache = self._corpora
            if cache is not None and cache.root is root:
                return cache
            for baseline in self._readers.finalize_baselines.values():
                if baseline.root is root:
                    # The held Searches of one cycle share its baseline's corpora.
                    if baseline.corpora is None:
                        baseline.corpora = SnapshotRecallCache(root)
                    return baseline.corpora
            for view in self._readers.search_views.values():
                if view.root is root:
                    # A scope's Searches share its Search view's corpora.
                    if view.corpora is None:
                        view.corpora = SnapshotRecallCache(root)
                    return view.corpora
                if view.adopted is not None and view.adopted.root is root:
                    if view.adopted.corpora is None:
                        view.adopted.corpora = SnapshotRecallCache(root)
                    return view.adopted.corpora
            for frozen in self._readers.quarantine_views.values():
                if frozen.root is root:
                    # A quarantined scope's Searches share its committed view's corpora.
                    if frozen.corpora is None:
                        frozen.corpora = SnapshotRecallCache(root)
                    return frozen.corpora
        return SnapshotRecallCache(root)

    def _publish(
        self, following: RecallSnapshot, changed_scopes: Sequence[str] = (),
    ) -> None:
        with self._readers.condition:
            previous = self._live_publisher().current
            removed_baselines: set[str] = set()

            def retained_failure(fence: ScopeFence) -> bool:
                baseline = self._readers.finalize_baselines.get(fence.scope_id)
                return (
                    baseline is not None and baseline.phase == "failed"
                    and fence.finalize_barrier == baseline.barrier
                )

            privacy_scopes = tuple(
                scope_id for scope_id in changed_scopes
                if (fence := following.scope_fence(scope_id)).tomb
                or fence.active_erasure or fence.maintenance
                or (
                    fence.visibility_barrier and not fence.quarantined
                    and not retained_failure(fence)
                )
            )
            for private_scope in privacy_scopes:
                if following.scope_fence(private_scope).finalize_barrier is not None:
                    raise QuoteMemError("parity_mismatch",
                                        "Privacy root retains a finalize barrier")
            changed_fences = {
                scope_id: following.scope_fence(scope_id) for scope_id in changed_scopes
            }
            privacy_changed: dict[str, ScopeFence] = {}
            for scope_id, fence in changed_fences.items():
                try:
                    prior = previous.scope_fence(scope_id)
                except QuoteMemError as error:
                    if error.code != "scope_denied":
                        raise
                    continue
                if fence.tomb or fence.active_erasure or fence.maintenance or (
                    (prior.scope_kind, prior.erase_epoch, prior.consent_epoch)
                    != (fence.scope_kind, fence.erase_epoch, fence.consent_epoch)
                ):
                    privacy_changed[scope_id] = fence
            self._discard_stale_quarantine_views(changed_fences, following.embedding_token)
            if privacy_changed or previous.embedding_token != following.embedding_token:
                # A baseline holds its own scope only (``RecallSnapshot.scope_view``):
                # another scope's privacy change never reaches it.
                removed_baselines.update(self._discard_fenced_baselines(
                    privacy_changed, following.embedding_token,
                ))
            for scope_id, fence in changed_fences.items():
                baseline = self._readers.finalize_baselines.get(scope_id)
                if baseline is None:
                    continue
                if (
                    fence.tomb or fence.active_erasure or fence.maintenance
                    or fence.visibility_barrier and not retained_failure(fence)
                ):
                    if fence.finalize_barrier is not None:
                        raise QuoteMemError("parity_mismatch",
                                            "Privacy root retains a finalize barrier")
                    del self._readers.finalize_baselines[scope_id]
                    removed_baselines.add(scope_id)
                elif baseline.phase in {"held", "failed"} and (
                    fence.finalize_barrier != baseline.barrier
                ):
                    raise QuoteMemError(
                        "parity_mismatch", "Published root lost its held finalize barrier",
                    )
            self._carry_search_views(previous, following)
            self._live_publisher().publish(following)
            self._readers.publication += 1
            corpora = self._corpora
            if corpora is not None:
                # The previous root's cache keeps no corpus of a privacy-changed
                # scope; the next carries only corpora whose inputs did not change.
                for scope_id in {*privacy_scopes, *privacy_changed}:
                    corpora.drop(scope_id)
                self._corpora = corpora.advance(following)
            blocked: set[str] = set()
            finalizing: dict[str, FinalizeBarrier] = {}
            for scope_id in changed_scopes:
                scope = following.scope_fence(scope_id)
                if (
                    scope.tomb
                    or scope.visibility_barrier
                    or scope.active_erasure
                    or scope.maintenance
                ):
                    blocked.add(scope_id)
                if scope.finalize_barrier is not None:
                    finalizing[scope_id] = scope.finalize_barrier
            affected = set(changed_scopes) | removed_baselines
            if privacy_scopes or previous.embedding_token != following.embedding_token:
                readers = tuple(self._readers.states.values())
            else:
                identities = {identity for scope_id in affected
                              for identity in self._readers.states_by_scope.get(scope_id, ())}
                readers = tuple(self._readers.states[identity] for identity in identities)
            for guard in readers:
                # A held Search reads its pinned baseline whatever the cycle does
                # meanwhile; its delivery recheck refuses it once the cycle moved on.
                if not self._readers.finalize_input_live(guard) or any(
                    scope_id in blocked or guard.finalize_search is None and (
                        scope_id in finalizing and not guard.completions
                        and guard.finalize_input != (scope_id, finalizing[scope_id])
                        and guard.finalize_publication != (scope_id, finalizing[scope_id])
                    ) or (
                        scope_id in changed_scopes and guard.finalize_input is None
                        and guard.finalize_publication is None and not guard.completions
                        and guard.finalize_search is None
                        and guard.root is not None
                        and following.scope_fence(scope_id).finalize_phase
                        != guard.root.scope_fence(scope_id).finalize_phase
                    )
                    for _, scope_id in guard.scopes
                ):
                    guard.cancellation.set()
                    if not guard.executing:
                        self._readers.drop(guard)
            self._readers.condition.notify_all()

    def _fail(self) -> None:
        with self._readers.condition:
            self._unhealthy = True
            self._readers.search_views.clear()
            if self._corpora is not None:
                self._corpora.clear()
            # A recoverable publication failure revokes access, not the fixed
            # lifetime of a held baseline. Privacy and close discard it below.
            for guard in tuple(self._readers.states.values()):
                guard.cancellation.set()
                if not guard.executing:
                    self._readers.drop(guard)
            self._readers.condition.notify_all()

    def _publication_failed(self) -> None:
        """Fail closed after a failed data publication or rebuild.

        The first failure at a ``commit_seq`` leaves the read view to its
        recovery rebuild. A second failure at the same ``commit_seq``, the
        rebuild's included, withdraws the runtime instead: every later
        publication and recovery refuses ``service_unhealthy``. A
        closed or withdrawn read view has nothing left to recover, so its
        refusal reads no storage.
        """
        if self._closed or self._withdrawn:
            self._fail()
            return
        try:
            generation: int | None = self.writer.get_counter("commit_seq")
        except Exception:
            generation = None
        with self._readers.condition:
            if self._failed_at == (generation,):
                self._withdrawn = True
            self._failed_at = (generation,)
            self._fail()

    def fail_closed(self, *, discard_finalize_baselines: bool = False) -> None:
        """Revoke content access after an owner detects an unpublished durable batch.

        Cancellation is nonblocking: executing callbacks drain later outside the
        Writer mutex through recover_privacy or their guard release.

        Args:
            discard_finalize_baselines: Discard retained content on privacy or
                storage-ownership failure. Ordinary publication recovery must
                retain the original held view rather than recapture newer data.

        Returns:
            None after all active guards are cancelled and future pins are blocked.

        Raises:
            None.
        """
        with self._readers.condition:
            if discard_finalize_baselines:
                self._readers.finalize_baselines.clear()
                self._readers.quarantine_views.clear()
            self._fail()

    def close(self, *, timeout: float = 30.0) -> None:
        """Permanently revoke, drain and release this publisher's content owners.

        Args:
            timeout: Finite nonnegative total wait for callbacks and the Writer
                publication fence. Stop and join application workers before
                calling; this method does not own provider or service workers.

        Raises:
            QuoteMemError: For an invalid timeout, self-drain under the Writer or an
                active callback, or an expired drain deadline. A timeout leaves
                the publisher terminal and revoked but may retain active content;
                callers must block purge and retry only after the actual drain.

        Successful repeated calls are harmless. No durable state, runtime,
        keyring or filesystem ownership changes. Registration remains reserved
        while this publisher exists; recovery cannot reopen it. The drain also
        retires every issued Search: a context its caller still holds can never
        pin again.
        """
        if (
            isinstance(timeout, bool) or not isinstance(cast(object, timeout), int | float)
            or not math.isfinite(timeout) or timeout < 0
        ):
            raise QuoteMemError("schema_invalid", "Close timeout must be finite and nonnegative")
        _require_outside_writer(self.writer)
        deadline = time.monotonic() + timeout
        with self._readers.condition:
            if any(
                threading.get_ident() in state.executing
                for state in self._readers.states.values()
            ):
                raise QuoteMemError("reader_busy", "A callback cannot join its own close drain")
            if self._closed and self._publisher is None:
                return
            self._closed = True
            self._readers.finalize_baselines.clear()
            self._readers.quarantine_views.clear()
            self._fail()
            while any(state.executing for state in self._readers.states.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise QuoteMemError("reader_busy",
                                        "Closed publisher still has active callbacks")
                self._readers.condition.wait(min(remaining, threading.TIMEOUT_MAX))
        if not self.writer.mutex.acquire(timeout=min(
            max(0.0, deadline - time.monotonic()), threading.TIMEOUT_MAX,
        )):
            raise QuoteMemError("reader_busy", "Closed publisher still awaits its Writer fence")
        try:
            with self._readers.condition:
                for state in tuple(self._readers.states.values()):
                    self._readers.drop(state)
                self._readers.searches.clear()
                self._readers.finalize_baselines.clear()
                self._readers.quarantine_views.clear()
                self._privacy_cutoff.clear()
                self._quiesced.clear()
                self._publisher = None
                self._corpora = None
                self.writer.close_pending_recall_references()
                self._readers.condition.notify_all()
        finally:
            self.writer.mutex.release()

    def publish_scope_quarantine(self, scope: ScopeV1) -> None:
        """Publish the Writer's durable scoped refusal and revoke its readers.

        The caller holds the Writer mutex after the receipt write. No content,
        finalize lease or unrelated scope is reconstructed or changed. A held
        cycle's retained baseline stops being an owner's input and becomes the
        scope's committed Search view (``_quarantine_search_root``); the
        scope's current readers are revoked, and their recompute reads that view.
        """
        if not self.writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise QuoteMemError("reader_busy", "Quarantine publication requires Writer exclusion")
        if not self.writer.is_scope_quarantined(scope.scope_id):
            raise QuoteMemError("binding_mismatch",
                                "Quarantine publication lacks its durable receipt")
        with self._readers.condition:
            root = self._live_publisher().current
            fence = root.scope_fence(scope.scope_id)
            if fence.scope_kind != scope.scope_kind:
                raise QuoteMemError("scope_denied", "Quarantine publication changed its scope")
            following = root.apply(generation=root.generation, scope_changes=(replace(
                fence, quarantined=True, visibility_barrier=True,
                finalized=None, finalize_phase=None, finalize_barrier=None,
            ),))
            baseline = self._readers.finalize_baselines.pop(scope.scope_id, None)
            if baseline is not None and baseline.phase == "held" and (
                fence.finalize_barrier == baseline.barrier
            ):
                # The scope's Searches read this baseline until the fault; they go
                # on reading it, never a commit of the refused cycle.
                self._readers.quarantine_views[scope.scope_id] = _QuarantineView(
                    _opened_scope_view(baseline.root, scope.scope_id),
                )
            self._publish(following, (scope.scope_id,))

    def publish_scope_repair(
        self, writer: Writer, scope: ScopeV1, owner_run_id: str,
        lease_generation: int, writer_epoch: int,
    ) -> None:
        """Publish an explicit repair: the scope's quarantine lifted around its resumed cycle.

        The caller holds the Writer mutex after ``owner_lease.repair_quarantine``
        deleted the durable receipt and moved the owner to its next attempt under
        the current Writer epoch. The scope's fence is rebuilt from durable
        state. A preparing cycle publishes its phase. A held cycle keeps its
        lease's barrier closed, as a cold adoption leaves it, until its owner's
        next hold binds a fresh baseline (``capture_cold_finalize_baseline``):
        the quarantine discarded the old one.
        """
        if writer is not self.writer:
            raise QuoteMemError("binding_mismatch", "Repair publication has another Writer")
        if not writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise QuoteMemError("reader_busy", "Repair publication requires Writer exclusion")
        if writer.is_scope_quarantined(scope.scope_id):
            raise QuoteMemError("binding_mismatch", "Repair publication keeps its durable receipt")
        try:
            fence = self._scope(scope.scope_kind, scope.scope_id)
            raw = writer.get_control(f"lease:finalize:{scope.scope_id}")
            value = None if raw is None else ControlRowV1.model_validate(raw).value
            identity = (owner_run_id, lease_generation, writer_epoch)
            barrier: FinalizeBarrier | None = None
            if isinstance(value, HeldFinalizeLeaseV1) and (
                value.owner_run_id, value.lease_generation, value.writer_epoch,
            ) == identity:
                phase = FinalizePhase("held", *identity)
                barrier = FinalizeBarrier(*identity, value.baseline_snapshot_generation,
                                          value.baseline_root_identity)
            elif isinstance(value, PreparingFinalizeLeaseV1) and (
                value.owner_run_id, value.lease_generation, value.writer_epoch,
            ) == identity:
                phase = FinalizePhase("preparing", *identity)
            else:
                raise QuoteMemError("lease_lost", "Repair publication lost its resumed cycle")
            with self._readers.condition:
                root = self._live_publisher().current
                if root.scope_fence(scope.scope_id).scope_kind != scope.scope_kind:
                    raise QuoteMemError("scope_denied", "Repair publication changed its scope")
                self._readers.finalize_baselines.pop(scope.scope_id, None)
                self._publish(root.apply(generation=root.generation, scope_changes=(replace(
                    fence, finalized=None, finalize_phase=phase, finalize_barrier=barrier,
                ),)), (scope.scope_id,))
        except BaseException:
            self._fail()
            raise

    def publish_privacy(
        self,
        writer: Writer,
        scope: ScopeV1,
        committed: CommittedRows | None,
        generation: int,
        *,
        maintenance: bool,
    ) -> None:
        """Publish one privacy transaction while preserving the full-scope barrier.

        Args:
            writer: The sole owner, after its durable batch under its mutex.
            scope: The erase's exact registered scope.
            committed: The rows an erase batch committed (``apply_privacy_erase``):
                Evidence placeholders, candidate expiries and tombstone deletions.
                None for a transaction that changed no projected row: an
                admission, a fence or a completion.
            generation: The exact resulting durable global commit sequence.
            maintenance: Whether a durable pending owner or maintenance Control bars the scope.

        Returns:
            None once one compact complete root is atomically visible.

        Raises:
            QuoteMemError: If durable parity, generation, scope, or reader drain differs.
        """
        self._publish_privacy_delta(
            writer, scope, committed, generation, barrier="maintenance",
            barrier_active=maintenance,
        )

    def retains_memory(self, scope: ScopeV1, memory_id: str) -> bool:
        """Whether the published root still holds a memory of ``scope``.

        Unlike ``pin`` this reads the root while a fence bars readers, so the
        forget final scan can check the view it is about to release.

        Args:
            scope: The exact scope whose fence the caller owns.
            memory_id: The memory that must be gone.

        Returns:
            True when the current root still projects the memory in that scope.
        """
        with self._readers.condition:
            row = self._live_publisher().current.get_memory(memory_id)
        return row is not None and (row.scope_kind, row.scope_id) == (
            scope.scope_kind, scope.scope_id,
        )

    def publish_erasure(
        self,
        writer: Writer,
        scope: ScopeV1,
        committed: CommittedRows | None,
        generation: int,
        *,
        active: bool,
    ) -> None:
        """Publish a forget's erasure barrier as ``publish_privacy`` does maintenance.

        Args:
            writer: The single Writer whose mutex the caller holds.
            scope: The exact scope the ``erasure:{scope}:{controller}`` fence bars.
            committed: The rows the transaction committed, or None for a fence
                install or release that changed no projected row.
            generation: The durable ``commit_seq`` the view now equals.
            active: Whether the durable erasure fence still holds the scope.

        Raises:
            QuoteMemError: If durable parity, generation, scope, or reader drain differs.
        """
        self._publish_privacy_delta(
            writer, scope, committed, generation, barrier="erasure", barrier_active=active,
        )

    def _publish_privacy_delta(
        self,
        writer: Writer,
        scope: ScopeV1,
        committed: CommittedRows | None,
        generation: int,
        *,
        barrier: Literal["maintenance", "erasure"],
        barrier_active: bool,
    ) -> None:
        """Share one exact privacy projection path without replacing either owner.

        The committed rows are proved against storage and projected as a commit
        is (``commit_delta``): sources, memories, entities, names and holds.
        """
        scope = _exact_scope(scope)
        with writer.mutex:
            try:
                current = self._live_publisher().current
                if (
                    writer is not self.writer
                    or self._unhealthy
                    or generation != writer.get_counter("commit_seq")
                    or generation not in {current.generation, current.generation + 1}
                ):
                    raise QuoteMemError("service_unhealthy",
                                        "Privacy root lost its durable generation")
                fence = self._scope(scope.scope_kind, scope.scope_id)
                active = fence.maintenance if barrier == "maintenance" else fence.active_erasure
                if active != barrier_active:
                    raise QuoteMemError(
                        "service_unhealthy", f"Privacy root and durable {barrier} disagree"
                    )
                old_fence = current.scope_fence(scope.scope_id)
                old_active = (old_fence.maintenance if barrier == "maintenance"
                              else old_fence.active_erasure)
                if committed is not None and (not barrier_active or not old_active):
                    raise QuoteMemError(
                        "service_unhealthy", "Privacy effects require an existing barrier"
                    )
                if (
                    not barrier_active
                    and old_active
                    and self._quiesced.get(scope.scope_id)
                    != self._privacy_cutoff.get(scope.scope_id, -1)
                ):
                    raise QuoteMemError(
                        "service_unhealthy", "Privacy barrier cannot clear before reader drain"
                    )
                if committed is not None and (
                    committed.commit_seq != generation
                    or (committed.scope_kind, committed.scope_id)
                    != (scope.scope_kind, scope.scope_id)
                    or any(
                        _privacy_row_scope(writer, kind, row)
                        != (scope.scope_kind, scope.scope_id)
                        for kind, row in (*committed.written, *committed.deleted)
                        if kind not in {"run", "control"}
                    )
                ):
                    raise QuoteMemError(
                        "parity_mismatch", "Privacy rows differ from their exact scope or commit"
                    )
                delta = None if committed is None else commit_delta(writer, current, committed)
                following = current.apply(
                    generation=generation,
                    source_changes=() if delta is None else delta.sources,
                    memory_changes=() if delta is None else delta.memories,
                    entity_changes=() if delta is None else delta.entities,
                    name_changes=() if delta is None else delta.names,
                    held_changes=() if delta is None else delta.held,
                    scope_changes=[fence],
                ).compact()
                self._publish(following, [scope.scope_id])
                self._privacy_cutoff[scope.scope_id] = self._readers.publication
                self._quiesced.pop(scope.scope_id, None)
            except Exception:
                with self._readers.condition:
                    self._readers.finalize_baselines.clear()
                self._publication_failed()
                raise

    def quiesce_privacy(self, scope: ScopeV1) -> None:
        """Drain every reader retaining an older root that holds the scope while it stays fenced.

        Such a reader is revoked and its callbacks are joined, so no root older
        than the barrier keeps the scope's bytes. A root without the
        scope, such as another scope's retained finalize baseline
        (``RecallSnapshot.scope_view``), is left alone: one scope's erasure
        never stops another scope's Add owner.

        Args:
            scope: The exact cleanup scope whose barrier is already published.

        Returns:
            None after callbacks exit and earlier owned roots/buffers are released.

        Raises:
            QuoteMemError: If called under the Writer mutex, from a callback, or without a barrier.
        """
        scope = _exact_scope(scope)
        _require_outside_writer(self.writer)
        with self._readers.condition:
            root = self._live_publisher().current
            fence = root.scope_fence(scope.scope_id)
            if fence.scope_kind != scope.scope_kind or not (
                fence.maintenance or fence.active_erasure or fence.visibility_barrier
            ):
                raise QuoteMemError(
                    "service_unhealthy", "Privacy drain requires a published scope barrier"
                )
            cutoff = self._privacy_cutoff.get(scope.scope_id, self._readers.publication)
            self._privacy_cutoff.setdefault(scope.scope_id, cutoff)
            selected = tuple(
                guard for guard in self._readers.states.values()
                if guard.publication < cutoff and guard.root is not None
                and guard.root.holds_scope(scope.scope_id)
            )
            if any(threading.get_ident() in guard.executing for guard in selected):
                raise QuoteMemError("reader_busy",
                                    "An active callback cannot join its own reader drain")
            for guard in selected:
                guard.cancellation.set()
            while any(guard.executing for guard in selected):
                self._readers.condition.wait()
            for guard in selected:
                self._readers.drop(guard)
            if self._privacy_cutoff.get(scope.scope_id, self._readers.publication) != cutoff:
                raise QuoteMemError("reception_stale", "Privacy root changed while readers drained")
            self._quiesced[scope.scope_id] = cutoff
            self._readers.search_views.pop(scope.scope_id, None)
            self._readers.quarantine_views.pop(scope.scope_id, None)
            if self._corpora is not None:
                # No corpus of the drained scope survives quiescence.
                self._corpora.drop(scope.scope_id)
            self._readers.condition.notify_all()

    def recover_if_needed(self) -> None:
        """Repair a failed owned Catalog/root before starting a new content borrow.

        The application must authorize its requested action first and call this
        outside all read callbacks. The recovery never changes selected domain
        commits, issues grants, or invokes a model provider.
        """
        if self._closed:
            raise QuoteMemError("service_unhealthy", "Closed publisher cannot recover")
        try:
            _ = self.current
        except QuoteMemError as error:
            if error.code != "service_unhealthy":
                raise
            with self.writer.mutex:
                if self._closed:
                    raise QuoteMemError("service_unhealthy",
                                        "Closed publisher cannot recover") from None
                self.writer.recover_catalog()
            self.recover_privacy(self.writer)

    def recover_privacy(self, writer: Writer) -> None:
        """Rebuild durable privacy state before returning the owner to healthy service.

        Args:
            writer: The same sole owner after restart or failed publication.

        Returns:
            None after readers drain and a complete durable root is published.

        Raises:
            QuoteMemError: If called under the Writer mutex or durable reconstruction fails.
        """
        _require_outside_writer(self.writer)
        if self._closed:
            raise QuoteMemError("service_unhealthy", "Closed publisher cannot recover")
        if writer is not self.writer:
            raise QuoteMemError("service_unhealthy",
                                "Root recovery requires the same persistence owner")
        if self._withdrawn:
            raise QuoteMemError("service_unhealthy", "A withdrawn read view cannot recover")
        self._fail()
        with self._readers.condition:
            if any(
                threading.get_ident() in guard.executing for guard in self._readers.states.values()
            ):
                raise QuoteMemError("reader_busy", "A callback cannot join its own recovery drain")
            while any(guard.executing for guard in self._readers.states.values()):
                self._readers.condition.wait()
            for guard in tuple(self._readers.states.values()):
                self._readers.drop(guard)
        with writer.mutex:
            self._live_publisher()
        with writer.mutex:
            try:
                self._live_publisher()
                rebuilt = self._rebuild()
                with self._readers.condition:
                    self._live_publisher().publish(rebuilt)
                    self._readers.search_views.clear()
                    self._corpora = SnapshotRecallCache(rebuilt)
                    self._readers.publication += 1
                    self._privacy_cutoff.clear()
                    self._quiesced.clear()
                    self._unhealthy = False
            except Exception:
                self._publication_failed()
                raise

    def _verification_root(self, scope: ScopeV1) -> RecallSnapshot:
        root = self._live_publisher().current
        fence = root.scope_fence(scope.scope_id)
        if (
            self._unhealthy
            or fence.scope_kind != scope.scope_kind
            or not (fence.maintenance or fence.active_erasure or fence.visibility_barrier)
            or self._quiesced.get(scope.scope_id) != self._privacy_cutoff.get(scope.scope_id, -1)
        ):
            raise QuoteMemError(
                "service_unhealthy", "Privacy verification requires a drained barrier root"
            )
        cutoff = self._privacy_cutoff[scope.scope_id]
        if any(
            state.publication < cutoff and state.root is not None
            and state.root.holds_scope(scope.scope_id)
            for state in self._readers.states.values()
        ):
            raise QuoteMemError(
                "service_unhealthy", "Privacy verification found an earlier owned reader"
            )
        return root

    def verify_privacy_projection(self, scope: ScopeV1, evidence_ids: Sequence[str]) -> None:
        """Verify exclusive tombstone projections after callback and old-root drain.

        Args:
            scope: The exact scope whose maintenance barrier remains published.
            evidence_ids: Every affected Evidence identity from the frozen cleanup input.

        Returns:
            None if the compacted current state retains no affected source pre-image.

        Raises:
            QuoteMemError: If drain, tombstone, scope, or retired-layer checks fail.
        """
        scope = _exact_scope(scope)
        with self._readers.condition:
            root = self._verification_root(scope)
            if not evidence_ids or len(set(evidence_ids)) != len(evidence_ids):
                raise QuoteMemError(
                    "parity_mismatch", "Privacy verification needs distinct Evidence IDs"
                )
            for object_id in evidence_ids:
                row = root.get_source(object_id)
                if (
                    row is None
                    or (row.scope_kind, row.scope_id) != (scope.scope_kind, scope.scope_id)
                    or row.state != "erased"
                    or row.route_state != "pending"
                    or row.text != ""
                    or row.role is not None
                    or row.byte_start != 0
                    or row.byte_end != 0
                    or row.content_hash != "0" * 64
                    or any(row.embedding)
                    or root.retains_source_preimage(object_id)
                ):
                    raise QuoteMemError("service_unhealthy",
                                        "Privacy source projection is not clean")

    def verify_owned_content(
        self, scope: ScopeV1, forbidden: Sequence[bytes] = (), *, retained: bool = False,
    ) -> None:
        """Scan owned source/memory copies and registered buffers without exporting bytes.

        Args:
            scope: The exact fenced cleanup scope.
            forbidden: Optional transient spans; recovery still runs the shared credential detector.
            retained: Whether the erase's storage scan found rows of the scope that still
                hold a recognized credential, which no rule of the erase removes
                (`Writer.scan_privacy_residuals`). The drained roots and buffers
                then copy them, so a detector match counts as theirs; a span match still
                fails. Without it a detector match fails too: a copy storage no longer
                holds is retained content.

        Returns:
            None after all publisher-owned content for the scope has zero matches.

        Raises:
            QuoteMemError: If reader drain is incomplete or any owned content matches.
        """
        from quotemem.ingestion.boundary import contains_recognized_credential

        scope = _exact_scope(scope)
        recognizes = None if retained else contains_recognized_credential
        with self._readers.condition:
            root = self._verification_root(scope)
            spans = tuple(forbidden)
            found = (spans or recognizes is not None) and root.contains_owned_text(
                scope.scope_id, spans, recognizes=recognizes
            )
            for state in self._readers.states.values():
                if state.root is not None and (spans or recognizes is not None):
                    found = found or state.root.contains_owned_text(
                        scope.scope_id, spans, recognizes=recognizes
                    )
                if any(scope_id == scope.scope_id for _, scope_id in state.scopes):
                    found = found or any(
                        span in buffer for span in spans for buffer in state.buffers
                    )
                    found = found or recognizes is not None and any(
                        recognizes(bytes(buffer)) for buffer in state.buffers
                    )
            if found:
                raise QuoteMemError("service_unhealthy",
                                    "Privacy verification found retained content")

    def _scope(self, scope_kind: ScopeKind, scope_id: str) -> ScopeFence:
        registration = self.writer.get_control(f"scope:{scope_id}")
        if registration is None or registration["value"]["scope_kind"] != scope_kind:
            raise QuoteMemError("scope_denied", "Read root requires its exact registered scope")
        reader = IndexedReader(
            self.writer.runtime.workspace,
            self.writer.entities,
            tenant_id=self.writer.tenant_id,
        )
        fences = reader.scope_fences(scope_id)
        # The strict Control schema refuses memory forget's held and replanning
        # maintenance, whose proofs left with forget_takeover, forget_handoff and
        # forget_maintenance_completion; forget recovery rebuilds them.
        for row in fences:
            ControlRowV1.model_validate(row)
        if (
            sum(row["key"].startswith("maintenance:") for row in fences) > 1
            or sum(row["key"].startswith("erasure:") for row in fences) > 1
        ):
            raise QuoteMemError("service_unhealthy",
                                "Read root found conflicting scope fence owners")
        pending = reader.pending_privacy(scope_id, limit=2)
        if len(pending) > 1:
            raise QuoteMemError("service_unhealthy", "Read root found competing privacy owners")
        for raw in pending:
            owner = DOMAIN_ADAPTERS["run"].validate_python(raw).model_dump(mode="json")
            if (
                (owner["scope_kind"], owner["scope_id"]) != (scope_kind, scope_id)
                or owner["run_kind"] != "retention"
                or owner["commit_mode"] != "destructive_scan"
                or owner["target_kind"] != "affected_set"
                or owner["status"] not in {"queued", "running"}
                or any(
                    row["key"].startswith("maintenance:")
                    and row["value"].get("owner_run_id") != owner["object_id"]
                    for row in fences
                )
            ):
                raise QuoteMemError("service_unhealthy", "Read root privacy ownership differs")
        # The admission header is already durable before its target plan exists.
        # It bars this scope without granting authority to attach any orphan input.
        maintenance = bool(pending) or any(
            row["key"].startswith("maintenance:") for row in fences
        )
        erasure = any(row["key"].startswith("erasure:") for row in fences)
        baseline = self._readers.finalize_baselines.get(scope_id)
        publisher = getattr(self, "_publisher", None)
        previous = None
        if publisher is not None:
            try:
                previous = publisher.current.scope_fence(scope_id)
            except QuoteMemError as error:
                if error.code != "scope_denied":
                    raise
        tomb = self.writer.get_control(f"tomb:{scope_id}") is not None
        quarantined = self.writer.is_scope_quarantined(scope_id)
        raw_hold = self.writer.get_control(f"evidence_hold:{scope_id}")
        hold = None if raw_hold is None else ControlRowV1.model_validate(raw_hold).value
        if hold is not None and not isinstance(hold, EvidenceHoldV1):
            raise QuoteMemError("binding_mismatch", "A scope's Evidence hold has another variant")
        finalize_fault = False
        if not quarantined and previous is not None and previous.finalize_phase is not None:
            owner = self.writer.get_row("run", previous.finalize_phase.run_id)
            if owner is None:
                raise QuoteMemError("reference_missing", "Active finalize scope lost its owner")
            checked = DOMAIN_ADAPTERS["run"].validate_python(owner)
            finalize_fault = (
                checked.status == "manual_attention" and checked.failure_code == "parity_mismatch"
            )
        return ScopeFence(
            scope_kind=scope_kind,
            scope_id=scope_id,
            scope_seq=self.writer.get_counter(f"scope_seq:{scope_id}"),
            erase_epoch=self.writer.get_counter(f"erase_epoch:{scope_id}"),
            consent_epoch=self.writer.get_counter(f"consent_epoch:{scope_id}"),
            tomb=tomb,
            maintenance=maintenance,
            active_erasure=erasure,
            quarantined=quarantined,
            visibility_barrier=maintenance or erasure or finalize_fault or quarantined,
            finalized=None if previous is None else previous.finalized,
            finalize_barrier=(
                None if tomb or maintenance or erasure else
                baseline.barrier if baseline is not None and baseline.phase in {"held", "failed"}
                else
                previous.finalize_barrier if baseline is None and previous is not None else None
            ),
            finalize_phase=(
                None if tomb or maintenance or erasure or previous is None
                else previous.finalize_phase
            ),
            held_evidence=() if hold is None else tuple(sorted({
                evidence_id for request in hold.requests for evidence_id in request.evidence_ids
            })),
        )

    def _rebase_pending(
        self, scope: ScopeV1, value: PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1,
        fence: ScopeFence,
    ) -> bool:
        """Whether a privacy change left the scope's live owner behind its epochs.

        Its admitted ``(erase_epoch, consent_epoch)`` trail the scope's, neither is
        ahead, and every other proof of the cycle holds (``_finalize_cycle``).
        """
        owner = self.writer.get_row("run", value.owner_run_id)
        if owner is None:
            return False
        admitted = (owner["admitted_erase_epoch"], owner["admitted_consent_epoch"])
        if admitted == (fence.erase_epoch, fence.consent_epoch) or (
            admitted[0] > fence.erase_epoch or admitted[1] > fence.consent_epoch
        ):
            return False
        self._finalize_cycle(
            scope, value.owner_run_id, value.lease_generation, value.writer_epoch, fence,
            allow_manual=True, allow_stopped=True, rebase_pending=True,
        )
        return True

    def _rebuild_held_fence(
        self, scope: ScopeV1, fence: ScopeFence, value: HeldFinalizeLeaseV1,
    ) -> ScopeFence:
        """Keep a live cycle's original baseline across durable root repair.

        A cold root can hold only a cycle that boot step 14 adopted:
        its baseline died with its process, so the lease's barrier stays closed
        until its owner's next hold binds a fresh one
        (``capture_cold_finalize_baseline``). Any other held cycle fails closed.
        """
        if self._cold_reconstruction or self._session_recovery:
            from quotemem.store.owner_lease import adopted_cycle

            identity = (value.owner_run_id, value.lease_generation, value.writer_epoch)
            if adopted_cycle(self.writer, scope.scope_id) != identity:
                raise QuoteMemError("lease_lost", "Finalize has no pending cold reconstruction")
            self._finalize_cycle(scope, *identity, fence, allow_stopped=True)
            if fence.tomb or fence.maintenance or fence.active_erasure or fence.visibility_barrier:
                raise QuoteMemError("lease_lost", "An adopted cycle lost its scope authority")
            return replace(
                fence, finalized=None, finalize_phase=FinalizePhase("held", *identity),
                finalize_barrier=FinalizeBarrier(
                    *identity, value.baseline_snapshot_generation, value.baseline_root_identity,
                ),
            )
        _, owner = self._finalize_cycle(
            scope, value.owner_run_id, value.lease_generation,
            value.writer_epoch, fence, allow_manual=True,
        )
        if owner["input_ref"] is not None:
            selected = FinalizeOwnerRunInputV1.model_validate(
                self.writer.read_bound_input(owner),
            )
            if selected.business_key.finalize_cycle != value.lease_generation:
                raise QuoteMemError("binding_mismatch", "Held recovery changed its selected input")
        baseline = self._finalize_baseline(
            scope, value.owner_run_id, value.lease_generation, value.writer_epoch, value,
        )
        if baseline.phase not in {"held", "failed"} or (
            owner["status"] != ("manual_attention" if baseline.phase == "failed" else "running")
            or fence.tomb or fence.maintenance or fence.active_erasure
            or fence.visibility_barrier and baseline.phase != "failed"
        ):
            raise QuoteMemError("lease_lost", "Live recovery lost its original held authority")
        return replace(
            fence, finalized=None, finalize_barrier=baseline.barrier,
            visibility_barrier=fence.visibility_barrier,
            finalize_phase=FinalizePhase(
                "held", value.owner_run_id, value.lease_generation, value.writer_epoch,
            ),
        )

    def _discard_fenced_baselines(
        self, fences: Mapping[str, ScopeFence], embedding_token: str,
    ) -> set[str]:
        """Discard a retained baseline whose own scope lost privacy authority.

        A baseline holds its own scope only (``RecallSnapshot.scope_view``). Its
        retention is safe while that scope keeps the same privacy epochs and the
        embedding identity holds; ``fences`` are the changed scopes' new fences.
        Return only removed scope identities so ordinary publications need no
        full inventory copy.
        """
        removed: set[str] = set()
        with self._readers.condition:
            for scope_id, baseline in tuple(self._readers.finalize_baselines.items()):
                old, fresh = baseline.root.scope_fence(scope_id), fences.get(scope_id)
                if baseline.root.embedding_token != embedding_token or fresh is not None and (
                    fresh.tomb or fresh.active_erasure or fresh.maintenance
                    or (old.scope_kind, old.erase_epoch, old.consent_epoch)
                    != (fresh.scope_kind, fresh.erase_epoch, fresh.consent_epoch)
                ):
                    del self._readers.finalize_baselines[scope_id]
                    removed.add(scope_id)
        return removed

    def _rebuild(self) -> RecallSnapshot:
        writer = self.writer
        policy = writer.get_control("policy:recall")
        embedding = writer.get_control("policy:embedding_space")
        if policy is None or embedding is None:
            raise QuoteMemError("service_unhealthy",
                                "Read model lacks initialized policy authority")
        reference = ArtifactId(policy["value"]["current_release_ref"])
        normalized = normalize_recall_policy(read_released_recall_config(writer, reference))
        controls = list(writer.iter_rows("control"))
        scopes: list[ScopeFence] = []
        registered = {
            row["key"][6:] for row in controls if row["key"].startswith("scope:")
        }
        fences = {
            row["key"][6:]: self._scope(row["value"]["scope_kind"], row["key"][6:])
            for row in controls if row["key"].startswith("scope:")
        }
        token = embedding_state_token(ControlRowV1.model_validate(embedding))
        self._discard_fenced_baselines(fences, token)
        with self._readers.condition:
            self._discard_stale_quarantine_views(fences, token)
        for row in controls:
            ControlRowV1.model_validate(row)
            for prefix in ("lease:finalize:", "finalized:"):
                if row["key"].startswith(prefix) and row["key"][len(prefix):] not in registered:
                    raise QuoteMemError("scope_denied",
                                        "Finalize Control lost its registered scope")
            if row["key"].startswith("scope:"):
                scope = ScopeV1(scope_kind=row["value"]["scope_kind"], scope_id=row["key"][6:])
                fence = fences[scope.scope_id]
                if fence.quarantined:
                    scopes.append(replace(
                        fence, finalized=None, finalize_phase=None, finalize_barrier=None,
                    ))
                    continue
                raw_lease = writer.get_control(f"lease:finalize:{scope.scope_id}")
                lease = None if raw_lease is None else ControlRowV1.model_validate(raw_lease)
                if self._unadopted_barred and lease is not None and isinstance(
                    lease.value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1,
                ):
                    # Boot step 13: step 14 adopts this cycle later (``for_forget_recovery``).
                    scopes.append(replace(
                        fence, finalized=None, finalize_barrier=None, finalize_phase=None,
                        visibility_barrier=True,
                    ))
                    continue
                if lease is not None and isinstance(
                    lease.value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1,
                ) and self._rebase_pending(scope, lease.value, fence):
                    # A privacy erase or consent change left the live owner behind:
                    # its next invocation rebases it and republishes its
                    # barrier. Until then the scope has none, as the erase's own
                    # publication left it, so a failed erase can recover its root.
                    scopes.append(replace(
                        fence, finalized=None, finalize_barrier=None, finalize_phase=None,
                    ))
                    continue
                if lease is not None and isinstance(lease.value, PreparingFinalizeLeaseV1):
                    value = lease.value
                    _, owner = self._finalize_cycle(
                        scope, value.owner_run_id, value.lease_generation,
                        value.writer_epoch, fence, allow_manual=True,
                    )
                    failed = owner["status"] == "manual_attention"
                    if owner["stage"] != "lease" or (
                        owner["status"] != "running" and not failed
                    ) or (failed and value.drain_state != "closed"):
                        raise QuoteMemError("lease_lost",
                                            "Preparing cold root lacks its running owner")
                    scopes.append(replace(
                        fence, finalized=None, finalize_barrier=None,
                        visibility_barrier=(
                            fence.visibility_barrier or failed or self._session_recovery
                        ),
                        finalize_phase=FinalizePhase(
                            "preparing", value.owner_run_id, value.lease_generation,
                            value.writer_epoch,
                        ),
                    ))
                    continue
                if lease is not None and isinstance(lease.value, HeldFinalizeLeaseV1):
                    scopes.append(self._rebuild_held_fence(scope, fence, lease.value))
                    continue
                if lease is not None and isinstance(lease.value, ReleasedFinalizeLeaseV1):
                    # A released cycle rebuilds from its own proved records;
                    # its marker may lag the scope sequence, which leaves it dirty.
                    from quotemem.store.released_history import prove_released_history

                    proved = prove_released_history(writer, scope, lease)
                    scopes.append(replace(
                        fence, finalized=FinalizedMarker(**proved.model_dump(mode="python")),
                        finalize_barrier=None, finalize_phase=None,
                    ))
                    continue
                if lease is not None and isinstance(lease.value, CancelledFinalizeLeaseV1):
                    # A forget cancelled the cycle: no owner holds the
                    # scope and no marker of it exists; the next admission proves
                    # the cancellation settled (``owner_lease.cancelled_settled``).
                    scopes.append(replace(
                        fence, finalized=None, finalize_barrier=None, finalize_phase=None,
                    ))
                    continue
                if raw_lease is not None or writer.get_control(f"finalized:{scope.scope_id}"):
                    # Fails closed: a marker without its lease is unproved.
                    raise QuoteMemError(
                        "service_unhealthy", "Released finalize history cannot be verified",
                    )
                scopes.append(replace(
                    fence, finalized=None, finalize_barrier=None, finalize_phase=None,
                ))
        sources: list[SourceRowV2] = []
        for raw in writer.iter_rows("evidence"):
            evidence = EvidenceRowV1.model_validate(raw)
            if (
                evidence.tenant_id != writer.tenant_id
                or len(evidence.embedding) != writer.dimension
            ):
                raise QuoteMemError(
                    "service_unhealthy", "Cold source projection crosses workspace binding"
                )
            sources.append(_source_projection(evidence))
        projected = cold_projections(writer.iter_rows)
        return RecallSnapshot.rebuild(
            generation=writer.get_counter("commit_seq"),
            scopes=scopes,
            embedding_space=embedding["value"]["embedding_space"],
            embedding_token=embedding_state_token(ControlRowV1.model_validate(embedding)),
            recall_policy=RecallPolicySelection.from_snapshot(
                reference, normalized.config_digest, normalized, integrity_digest(normalized)
            ),
            sources=sources,
            memories=projected.memories,
            entities=projected.entities,
            names=projected.names,
            held=projected.held,
        )

    @property
    def current(self) -> SnapshotMetadata:
        """Return publication identity without exposing an untracked content root.

        Args:
            None.

        Returns:
            Content-free generation and root identity.

        Raises:
            QuoteMemError: If durable visibility requires explicit owner recovery.
        """
        with self.writer.mutex:
            if self._unhealthy or self._live_publisher(
                ).current.generation != self.writer.get_counter(
                "commit_seq"
            ):
                self._fail()
                raise QuoteMemError(
                    "service_unhealthy", "Durable visibility awaits complete root recovery"
                )
            root = self._live_publisher().current
            return SnapshotMetadata(root.generation, root.root_identity)

    def committed_scope_fence(self, scope_id: str) -> ScopeFence | None:
        """Return one scope's fence in the committed root, under the caller's Writer mutex.

        Inside a Writer critical section the published root is the committed one
        (``publish_reception`` runs before the commit mutex is released).

        Args:
            scope_id: The exact scope.

        Returns:
            The scope's fence, or None for a scope the root does not know.

        Raises:
            QuoteMemError: ``reader_busy`` outside a Writer critical section.
        """
        if not self.writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise QuoteMemError("reader_busy", "A committed fence read requires the Writer mutex")
        with self._readers.condition:
            root = self._live_publisher().current
        try:
            return root.scope_fence(scope_id)
        except QuoteMemError as error:
            if error.code != "scope_denied":
                raise
            return None

    def publish_reception(
        self,
        writer: Writer,
        receipt: SessionDurableRunInputV1,
        evidence: Sequence[EvidenceRowV1],
    ) -> None:
        """Apply the exact reception delta before the Writer releases its commit mutex.

        Args:
            writer: The same sole persistence owner used at construction.
            receipt: Attached no-content witness in the just-committed transaction.
            evidence: Its complete strict Evidence post-images, supplied by that Writer.

        Returns:
            None after the complete new root is published.

        Raises:
            QuoteMemError: If generations, owner, source binding, or publication fail parity.
        """
        with writer.mutex:
            try:
                self._live_publisher()
                receipt = type(receipt).model_validate(receipt.model_dump(mode="json"))
                evidence = tuple(
                    EvidenceRowV1.model_validate(row.model_dump(mode="json")) for row in evidence
                )
                if (
                    writer is not self.writer
                    or self._unhealthy
                    or (
                        writer.get_counter("commit_seq") != receipt.visibility_commit_seq
                        or self._live_publisher().current.generation + 1
                        != receipt.visibility_commit_seq
                        or writer.bind_content(
                            "finalize-evidence-row",
                            [row.model_dump(mode="json") for row in evidence],
                        )
                        != receipt.evidence_set_digest
                    )
                ):
                    raise QuoteMemError(
                        "service_unhealthy", "Reception root lost its exact durable delta"
                    )
                scopes: set[tuple[ScopeKind, str]] = {
                    (row.scope_kind, row.scope_id) for row in evidence
                }
                following = self._live_publisher().current.apply(
                    generation=receipt.visibility_commit_seq,
                    source_changes=[
                        _source_projection(row) for row in evidence
                    ],
                    scope_changes=[self._scope(kind, scope) for kind, scope in sorted(scopes)],
                )
                self._publish(following, tuple(scope for _, scope in sorted(scopes)))
            except Exception:
                self._publication_failed()
                raise

    def publish_commit(self, writer: Writer, commit: CommittedRows) -> None:
        """Apply a committed natural-language transaction's delta while the Writer holds its mutex.

        The changed rows are proved against storage and their projections
        re-read from committed rows (``store/snapshot_rows.py``), so the new
        root equals a cold root at the same ``commit_seq``.

        Args:
            writer: The same sole owner, right after the commit, under its mutex.
            commit: The committed transaction's rows.

        Raises:
            QuoteMemError: ``service_unhealthy`` when the root is not exactly one
                generation behind the commit, or ``parity_mismatch`` when
                storage differs from the committed rows. Any failure fails the
                read view closed; a second at the same ``commit_seq`` withdraws it.
        """
        with writer.mutex:
            try:
                root = self._live_publisher().current
                if (
                    writer is not self.writer
                    or self._unhealthy
                    or commit.commit_seq != writer.get_counter("commit_seq")
                    or commit.commit_seq != root.generation + 1
                ):
                    raise QuoteMemError("service_unhealthy",
                                        "Commit root lost its exact durable delta")
                delta = commit_delta(writer, root, commit)
                fence = self._scope(cast(ScopeKind, commit.scope_kind), commit.scope_id)
                following = root.apply(
                    generation=commit.commit_seq,
                    source_changes=delta.sources,
                    memory_changes=delta.memories,
                    entity_changes=delta.entities,
                    name_changes=delta.names,
                    held_changes=delta.held,
                    scope_changes=[fence],
                )
                self._publish(following, (commit.scope_id,))
            except Exception:
                self._publication_failed()
                raise

    def check_reception(self, writer: Writer, receipt: SessionDurableRunInputV1) -> None:
        """Verify current publication for replay without reconstructing historical sources.

        Args:
            writer: The same initialized owner.
            receipt: Historical durable witness already checked against its ledger.

        Returns:
            None when a complete current root covers the receipt's generation.

        Raises:
            QuoteMemError: If a durable visibility gap needs explicit owner recovery.
        """
        with writer.mutex:
            if writer is not self.writer or self.current.generation < receipt.visibility_commit_seq:
                raise QuoteMemError(
                    "service_unhealthy", "Durable receipt has no published complete root"
                )
