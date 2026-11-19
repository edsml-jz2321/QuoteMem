"""The single AML persistence authority, including bounded reception and cleanup.

Application services own provider orchestration. Every durable transition uses
this Writer's strict admission, real HeavenBase plans, and publication fences.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import itertools
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast
from weakref import WeakKeyDictionary, WeakValueDictionary
from weakref import ref as weak_ref

from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_json

from aml.core.artifacts import (
    ArtifactRowV1,
    BootstrapPolicyReleaseV1,
    EmbeddingCanaryConfigV1,
    parse_artifact_body,
    placement_refusal,
)
from aml.core.controls import (
    POLICY_POINTERS,
    ActiveEmbeddingSpaceV1,
    ControlRowV1,
    JobProviderBindingsV1,
    JobValueV1,
    PolicyPointerV1,
    RequestCompletionPolicyV2,
    RequestLedgerValueV1,
    control_key_occupies_scope,
    control_scope_id,
    maintenance_fence_key,
    parse_control_value,
)
from aml.core.entities import VERDICT_REF
from aml.core.errors import AMLError
from aml.core.gc_safe import IdentityMemo
from aml.core.identity import (
    ArtifactId,
    ContentBindingDigest,
    IntegrityDigest,
    artifact_id,
    canonical_json,
    control_id,
    integrity_digest,
    new_ulid,
)
from aml.core.identity import (
    scope_id as canonical_scope_id,
)
from aml.core.references import ReferenceProjectionV1
from aml.core.semantic import SystemTimestamp
from aml.providers.canary import CANARY_TEXT_VERSION
from aml.providers.config import EMBEDDING_MODEL, AddTiersV1, ProviderConfig
from aml.providers.keyring import ContentArrayBinding
from aml.service.request_measurement import resource_session_event
from aml.store.catalog_ledger import (
    CLEAN,
    DIRTY,
    UNPROVEN,
    CatalogLedger,
    CatalogLedgerV1,
    start_catalog_ledger,
)
from aml.store.catalog_proof import CatalogProofV1
from aml.store.domain import (
    DOMAIN_ADAPTERS,
    EvidenceRowV1,
    Nonnegative,
    ReplayableRecallRowV1,
)
from aml.store.entities import register_schemas
from aml.store.failures import constraint_violation
from aml.store.finalize_ticket import FinalizeLeaseTicket
from aml.store.hb_runtime import HBRuntime
from aml.store.metadata import StrictModel
from aml.store.pending_recall import (
    PendingRecallReferenceRegistry,
)
from aml.store.reference_context import (
    contextual_references,
)
from aml.store.registry import Reference, extract_references
from aml.store.row_cache import (
    MEMBERSHIP_ANY,
    MISSING,
    ROW_BINDING_BYTES,
    BorrowWatch,
    RowCache,
    ScanExtension,
    ScanMemo,
    copy_row,
    membership_family,
    membership_key,
    note_reads,
    recording_reads,
    sized_copy,
)
from aml.store.row_proofs import (
    MEMORY_ROW_KINDS,
    VectorSpace,
    check_batch,
    check_deletion,
    check_evidence_change,
    check_row,
    row_anchors,
)
from aml.store.writer_mutex import WriterMutex

_T = TypeVar("_T")

_RECEPTION_ROOT: ContextVar[str | None] = ContextVar("aml_reception_root", default=None)
"""The Add root whose pre-ordering bookkeeping the current writes are, if any."""

_OWN_FINALIZE: ContextVar[FinalizeLeaseTicket | None] = ContextVar(
    "aml_own_finalize_writes", default=None,
)
"""The held finalize owner whose own writes the current context makes, if any."""
# Parent links followed from a written Run or Artifact creator to its owner.
_OWN_FINALIZE_DEPTH = 8

OWNER_PROBE_INTERVAL_SECONDS = 0.05
BORROWED_REFRESH_ATTEMPTS = 4
"""Consistent-window read attempts before a borrowed refresh falls back to the mutex."""
BORROWED_REFRESH_WAIT_SECONDS = 0.05
"""Longest wait for an active critical section to end before rereading the window."""
"""Longest age of a completed storage-owner inspection that a borrowed read reuses."""

if TYPE_CHECKING:
    from pathlib import Path

    from heavenbase.workspace.rows.writer import UpsertPlan

    from aml.core.add_read_view import MessageRecord
    from aml.core.artifacts import RequestLedgerV1
    from aml.core.inference import StoredInferenceOutput
    from aml.core.privacy_persistence import PrivacyEraseInputV1, ResidualKind
    from aml.core.query import ScopeV1
    from aml.core.run_inputs import (
        MessagesRequestPayloadV1,
        SessionAdmissionRunInputV1,
        SessionDurableRunInputV1,
    )
    from aml.core.source_labels import SourceLabelsBudgetV1
    from aml.core.work import FinalizeSourceInventoryBlockV1
    from aml.ingestion.boundary import (
        AddAdmissionBinding,
        AddRequestV1,
        EvidencePart,
        Profile,
    )
    from aml.ingestion.finalize_freeze import (
        _SourceInventoryIndex,  # pyright: ignore[reportPrivateUsage]
    )
    from aml.ingestion.privacy_recovery import PendingPrivacy
    from aml.ingestion.receipts import LedgerHead
    from aml.providers.auth import PrincipalGrant
    from aml.providers.contracts import InferenceRequest, InferenceResult, OutputIdentity
    from aml.service.bench_workers import BenchWorkPool
    from aml.service.resource_observer import ResourceObserver
    from aml.store.indexed import IndexedReader
    from aml.store.read_model import (
        SnapshotReadGuard,
        StoreReadModel,
    )
    from aml.store.scope_evidence import ScopeEvidenceIndex
    from aml.store.source_labels import LabelOwner, SourceLabelsPreparation

MAX_BODY_CHARS = 262144
MAX_TX_ROWS = 2000
_REF_ADAPTER: TypeAdapter[ReferenceProjectionV1] = TypeAdapter(ReferenceProjectionV1)
_JOB_TIME: TypeAdapter[int] = TypeAdapter(SystemTimestamp)
_APPEND_ONLY = {"revision", "link", "entity_name", "tombstone"}
_COMMIT_KINDS = frozenset({
    "item", "revision", "link", "candidate", "evidence", "entity", "entity_name", "tombstone",
})
"""The kinds an NL transaction writes (C.12); Evidence only changes there, never starts."""
_DELETE_KINDS = frozenset({"item", "revision", "link", "candidate", "entity_name", "tombstone"})
"""The kinds an NL transaction deletes: forget's retention unit, names and tombstones."""
_TOMBSTONE_BUMPS = frozenset({"tombstone_seq", "erase_epoch"})
"""The counters a tombstone change and a forget admission advance together (D.3, D.7.5)."""
_SCOPE_COUNTERS = ("scope_seq", "erase_epoch", "consent_epoch", "tombstone_seq")
"""The counters every registered scope starts at zero (A.3.9)."""
_SCOPE_SEQ_KINDS = frozenset({
    "item", "revision", "link", "evidence", "entity", "entity_name", "tombstone",
})
"""Writing or deleting a row of these kinds advances ``scope_seq:{s}`` (A.3.9)."""
_ROW_BOOKKEEPING = {"run", "recall", "control"}
_DIRECT_SCOPE_KINDS = frozenset({
    "item", "link", "evidence", "candidate", "entity", "entity_name", "tombstone", "run",
    "recall", "artifact",
})
"""Row kinds that carry their own ``scope_kind``/``scope_id`` columns (A.3)."""
_RUN_BOOKKEEPING = {
    "status",
    "stage",
    "attempt",
    "input_ref",
    "result_ref",
    "auth_context_ref",
    "writer_epoch",
    "failure_code",
    "started_at",
    "updated_at",
    "finished_at",
}

CATALOG_SEAL_SECONDS = 5.0
"""Longest wait for the Writer mutex when a clean close seals its Catalog proof."""


@dataclass(frozen=True)
class WriteTicket:
    """Freeze a caller's execution and scope fences before external work."""

    writer_epoch: int
    run_id: str | None
    scope_kind: str | None
    scope_id: str | None
    erase_epoch: int | None
    consent_epoch: int | None


@dataclass(frozen=True)
class RowWrite:
    """Carry an Entity kind and a complete strict post-image."""

    kind: str
    row: StrictModel


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class PreparedArtifact:
    """Keep one detached logical encoding between address calculation and commit.

    Only ``Writer.prepare_artifact`` issues this value. Its private body and
    physical rows remain in that Writer under a weak identity key. The token
    retains only their address and content-free creator ticket; privacy cache
    release removes all issued rows even while tokens live. Preparation grants
    no write authority: commit rechecks admission, source, placement and references.
    """

    creator: WriteTicket
    reference: ArtifactId


@dataclass(frozen=True, slots=True)
class _PreparedArtifactRows:
    """The Writer's private, detached rows; the producer receives only their token."""

    creator: WriteTicket
    kind: str
    body: StrictModel
    security_level: str
    rows: tuple[ArtifactRowV1, ...]


@dataclass(frozen=True, slots=True)
class _ArtifactPreparationAdmission:
    """One exact ticket's content-free admission within an unchanged Writer hold."""

    section: int
    storage_epoch: int
    owner: FinalizeLeaseTicket
    owner_identity: tuple[str, int, str, int, str]
    owner_digest: ContentBindingDigest


@dataclass(frozen=True)
class RowDelete:
    """Name one stored row an NL transaction deletes (I2)."""

    kind: str
    object_id: str


@dataclass(frozen=True)
class ExpectedHead:
    """An Item's head and lifecycle as its producer read them at its pin (C.11.2 #3).

    The commit compares them with the stored Item under the Writer mutex and
    refuses the whole transaction with ``target_changed`` when either moved.
    """

    memory_id: str
    current_revision_id: str
    lifecycle_state: str


@dataclass(frozen=True)
class CommittedRows:
    """One committed NL transaction as its read-model publisher receives it (I2, I4).

    Attributes:
        commit_seq: The transaction's global ``commit_seq``.
        scope_kind: The kind of the transaction's scope.
        scope_id: The transaction's scope.
        written: Each written row's kind and strict JSON post-image, in batch order.
        deleted: Each deleted row's kind and strict JSON pre-image.
    """

    commit_seq: int
    scope_kind: str
    scope_id: str
    written: tuple[tuple[str, dict[str, Any]], ...]
    deleted: tuple[tuple[str, dict[str, Any]], ...]


@dataclass(frozen=True)
class _ArtifactCapacityKey:
    body_type: type[StrictModel]
    kind: str
    body_digest: IntegrityDigest
    ticket: WriteTicket
    tenant_id: str
    security_level: str
    created_at: int
    layout: IntegrityDigest


@dataclass(frozen=True)
class _CommitJournals:
    """Content-free journal directory and the exact rows proving its membership."""

    token: int
    keys: tuple[tuple[str, str], ...]
    entries: tuple[tuple[str, tuple[ArtifactId, ...]], ...]


class _LogicalMemo(
    dict[tuple[str, str], tuple[int, StrictModel, tuple[tuple[str, str], ...], int]],
):
    """Parsed logical Artifacts by reference and kind, least recently used first.

    Each entry is the token read before its parse, the parsed body, the rows
    the parse read in order and the stored body bytes charged for it; ``size``
    is the sum of those charges.
    """

    def __init__(self) -> None:
        super().__init__()
        self.size = 0

    def clear(self) -> None:
        super().clear()
        self.size = 0


class SessionArtifactCapacity:
    """Retain only measured counts for one ordinary checkpoint handoff.

    The owner references prevent recycled identities from matching another
    workspace. No body, model or prepared operation crosses the handoff.
    Every lookup still belongs to a fresh admitted and strictly parsed batch.
    """

    def __init__(
        self, writer: Writer, stamp: Callable[[], IntegrityDigest | None],
    ) -> None:
        self._writer = writer
        self._stamp = stamp
        self._counts: dict[_ArtifactCapacityKey, int] = {}
        self._closed = False

    def close(self) -> None:
        """Discard the one handoff on success, failure or replacement."""
        self._counts.clear()
        self._closed = True

    def stamp(self, writer: Writer) -> IntegrityDigest | None:
        """Read this owner's encoder dependencies within its existing HB scope."""
        if self._closed or writer is not self._writer:
            return None
        return self._stamp()

    def lookup(self, key: _ArtifactCapacityKey) -> int | None:
        """Return a completed count after the caller's fresh body validation."""
        return self._counts.get(key)

    def store(self, entries: Sequence[tuple[_ArtifactCapacityKey, int]]) -> None:
        """Publish a completed batch, retaining at most one bounded inventory."""
        if self._closed:
            return
        for key, count in entries:
            if len(self._counts) >= 4096:
                self._counts.clear()
            self._counts[key] = count


class _ReceiptStamp(StrictModel):
    """The two Evidence fields that only the Add commit critical section assigns."""

    ingested_at: SystemTimestamp
    ingested_commit_seq: Nonnegative


@dataclass(frozen=True)
class _PreparedAddCommit:
    """One Add commit's preparation, computed without the Writer mutex.

    Every field derives from the commit's arguments or from borrowed reads of
    the rows named by ``read_keys``; ``token`` is the storage epoch read
    before those rows (ADR 0036). The receipt time and visibility generation
    are absent by design: only the commit critical section assigns them.
    """

    canonical: AddAdmissionBinding
    parts: tuple[EvidencePart, ...]
    expected_security: str
    ledger_value: RequestLedgerValueV1
    overflow: RequestLedgerV1 | None
    ledger_json: str
    evidence: tuple[RowWrite, ...]
    read_keys: frozenset[tuple[str, str]]
    token: int


@dataclass(frozen=True)
class BusinessReferenceProof:
    """Creator-local immutable promises checked before the final short write phase."""

    run_id: str
    artifact_generation: int
    references: tuple[Reference, ...]


def _finite_numeric_sequence(items: list[object]) -> bool:
    """Report a sequence of finite Python numbers without a Python-level walk.

    ``sum`` over ints, floats and bools runs in C; an infinite or NaN member
    makes the total non-finite, and a non-numeric member raises ``TypeError``.
    Only ``True`` is conclusive: a mixed or non-numeric sequence still needs the
    element walk, which also rejects nested non-finite values.
    """
    try:
        total = float(sum(cast(list[float], items)))
    except (TypeError, OverflowError):
        return False
    return math.isfinite(total) and all(type(item) in (float, int, bool) for item in items)


def _keys_of_write_ops(
    args: tuple[Any, ...], kwargs: dict[str, Any],
) -> Iterable[tuple[str, Any]] | None:
    """Name the rows a HeavenBase ``write_ops`` call replaces or deletes."""
    ops_by_backend = kwargs.get("ops_by_backend", args[0] if args else None)
    if not isinstance(ops_by_backend, dict):
        return None
    keys: list[tuple[str, Any]] = []
    for ops in cast(dict[str, Any], ops_by_backend).values():
        named = _keys_of_row_ops(ops)
        if named is None:
            return None
        keys.extend(named)
    return keys


def _keys_of_row_ops(ops: object) -> list[tuple[str, Any]] | None:
    if not isinstance(ops, list | tuple):
        return None
    keys: list[tuple[str, Any]] = []
    for op in cast(list[Any] | tuple[Any, ...], ops):
        entity_id = getattr(op, "entity_id", None)
        if not isinstance(entity_id, str) or not hasattr(op, "object_id"):
            return None
        keys.append((entity_id, op.object_id))
    return keys


def _keys_of_upsert(
    args: tuple[Any, ...], kwargs: dict[str, Any],
) -> Iterable[tuple[str, Any]] | None:
    """Name the rows a backend ``upsert(rows)`` call replaces."""
    return _keys_of_row_ops(kwargs.get("rows", args[0] if args else None))


def _keys_of_delete(
    args: tuple[Any, ...], kwargs: dict[str, Any],
) -> Iterable[tuple[str, Any]] | None:
    """Name the rows a backend ``delete(entity_id, object_ids)`` call removes."""
    entity_id = kwargs.get("entity_id", args[0] if args else None)
    object_ids = kwargs.get("object_ids", args[1] if len(args) > 1 else None)
    if not isinstance(entity_id, str) or not isinstance(object_ids, list | tuple | set | frozenset):
        return None  # An iterator cannot be inspected without consuming it.
    return [(entity_id, object_id) for object_id in cast(Any, object_ids)]


def _keys_of_compare_upsert(
    args: tuple[Any, ...], kwargs: dict[str, Any],
) -> Iterable[tuple[str, Any]] | None:
    """Name the row a backend ``compare_upsert(row, ...)`` call may replace."""
    row = kwargs.get("row", args[0] if args else None)
    return _keys_of_row_ops([row]) if row is not None else None


_ADMISSION_MEMO_LIMIT = 4096
# Finalize owners whose latest run of heartbeat-only epoch advances is kept.
_HEARTBEAT_OWNERS = 64


def _row_ops_of(
    keys_of: Callable[[tuple[Any, ...], dict[str, Any]], Iterable[tuple[str, Any]] | None]
    | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[tuple[str, str, Any, Any]] | None:
    """Describe a hooked mutation's rows as ``(entity_id, operation, values, object_id)``.

    ``None`` means the call cannot be inspected (a drop, an iterator, or an
    unknown shape); callers must then assume every row may have changed.
    """
    if keys_of is _keys_of_delete:
        entity_id = kwargs.get("entity_id", args[0] if args else None)
        object_ids = kwargs.get("object_ids", args[1] if len(args) > 1 else None)
        if not isinstance(entity_id, str) or not isinstance(
            object_ids, list | tuple | set | frozenset
        ):
            return None
        return [(entity_id, "delete", None, object_id) for object_id in cast(Any, object_ids)]
    if keys_of is _keys_of_write_ops:
        ops_by_backend = kwargs.get("ops_by_backend", args[0] if args else None)
        if not isinstance(ops_by_backend, dict):
            return None
        ops: list[Any] = []
        for backend_ops in cast(dict[str, Any], ops_by_backend).values():
            if not isinstance(backend_ops, list | tuple):
                return None
            ops.extend(cast(list[Any] | tuple[Any, ...], backend_ops))
    elif keys_of is _keys_of_upsert:
        rows = kwargs.get("rows", args[0] if args else None)
        if not isinstance(rows, list | tuple):
            return None
        ops = list(cast(list[Any] | tuple[Any, ...], rows))
    elif keys_of is _keys_of_compare_upsert:
        row = kwargs.get("row", args[0] if args else None)
        if row is None:
            return None
        ops = [row]
    else:
        return None
    described: list[tuple[str, str, Any, Any]] = []
    for op in ops:
        entity_id = getattr(op, "entity_id", None)
        operation = getattr(op, "op", "upsert")
        if not isinstance(entity_id, str) or not isinstance(operation, str):
            return None
        described.append((
            entity_id, operation, getattr(op, "values", None), getattr(op, "object_id", None),
        ))
    return described


def _membership_of(kind: str, values: dict[str, Any]) -> tuple[tuple[str, str] | None, ...]:
    """Name every set an upserted post-image of ``kind`` joins.

    A ``None`` names a set the post-image cannot identify, and the caller
    retires that family's wildcard for it. A Run without an idempotency key
    joins no family; its post-image names the empty member, which no scan
    proof includes.

    A post-image can join more than one set. A Run joins its idempotency
    family, the children of its parent and its exact scope: a scan enumerating one
    Run's children is keyed by neither the family nor the child's own row, so
    without the second key a child created after that scan could not retire it.
    An Artifact joins the outputs of its creator for the same reason.
    """
    if kind in {"evidence", "candidate", "item"}:
        scope_kind, scope_id = values.get("scope_kind"), values.get("scope_id")
        request: tuple[tuple[str, str] | None, ...] = ()
        if kind == "evidence":
            coordinates = [values.get(name) for name in (
                "scope_kind", "scope_id", "session_id", "request_id",
            )]
            request = (membership_key("evidence", canonical_json(coordinates))
                       if all(isinstance(value, str) for value in coordinates) else None,)
        if isinstance(scope_kind, str) and isinstance(scope_id, str):
            return (membership_key(kind, f"{scope_kind}:{scope_id}"), *request)
        return (None, *request)
    if kind == "revision":
        memory = values.get("memory_id")
        return (membership_key(kind, memory) if isinstance(memory, str) else None,)
    if kind == "control":
        key = values.get("key")
        return (membership_key("control", key.rsplit(":", 1)[0]) if isinstance(key, str) else None,)
    if kind == "artifact":
        creator = values.get("created_by_run_id")
        return (membership_key("artifact", creator) if isinstance(creator, str) else None,)
    family: tuple[str, str] | None
    if "idempotency_key" not in values:
        family = None
    elif values["idempotency_key"] is None:
        family = membership_key("run", "")
    elif isinstance(values["idempotency_key"], str):
        family = membership_key("run", values["idempotency_key"])
    else:
        family = None
    parent = values.get("parent_run_id")
    children = None
    if isinstance(parent, str):
        children = membership_key("run-child", parent)
    elif "parent_run_id" in values and parent is None:
        children = membership_key("run-child", "")
    scope_kind, scope_id = values.get("scope_kind"), values.get("scope_id")
    scope = None
    if isinstance(scope_kind, str) and isinstance(scope_id, str):
        scope = membership_key("run-scope", f"{scope_kind}:{scope_id}")
    elif "scope_kind" in values and "scope_id" in values and (
        scope_kind is None and scope_id is None
    ):
        scope = membership_key("run-scope", "")
    return (family, children, scope)


WRITE_RUN_SECONDS = 1.0
"""Longest run of back-to-back write scopes after which a thread yields the Writer (C.2a).

Neither the Writer mutex nor HeavenBase's writer-preferring workspace lock is
fair: a thread that enters its next write scope at once takes both back before
a woken waiter runs. In the full a7 workload one owner's 976 judge-child
admissions kept them for 51 s in one run, and the scheduler's heartbeat and
discovery passes waited 49 s and 51 s for one borrowed read each.
"""
WRITE_YIELD_SECONDS = 0.01
"""How long a thread stays out of the Writer when its run ends, so waiters go first."""


class _WriteRun:
    """One thread's run of back-to-back outermost write scopes on one Writer (C.2a).

    A run continues while the thread enters its next outermost scope less than
    ``WRITE_YIELD_SECONDS`` after leaving the previous one. A scope entered once
    the run is ``WRITE_RUN_SECONDS`` old first stays out that long and starts a
    new run: every waiter for the Writer mutex or the workspace lock goes first.
    Each scope stays whole: its fences, lease checks and writes are unchanged.
    """

    __slots__ = ("began", "left")

    def __init__(self) -> None:
        self.began = self.left = -math.inf

    def __enter__(self) -> None:
        now = time.monotonic()
        if now - self.left >= WRITE_YIELD_SECONDS:
            self.began = now
        elif now - self.began >= WRITE_RUN_SECONDS:
            time.sleep(WRITE_YIELD_SECONDS)
            self.began = time.monotonic()

    def __exit__(self, *_: object) -> None:
        self.left = time.monotonic()


class _WriteRuns(threading.local):
    """Each thread's own ``_WriteRun`` on one Writer."""

    def __init__(self) -> None:
        self.run = _WriteRun()


class Writer:
    """Own all AML writes under one mutex for an explicitly supplied HB runtime.

    Args:
        runtime: Isolated HeavenBase runtime with privacy configuration enforced.
        tenant_id: The workspace's internal tenant identity.
        dimension: Immutable supported vector dimension.
        bind_content: Tenant-keyed digest function supplied by the composition root.
        provider_config: Exact nonsecret configuration validated by bootstrap;
            required for the bench job infrastructure producer.
        bind_content_array: Tenant-owned streaming canonical-array factory;
            required only by the nonempty finalize input reader.
    """

    # Placeholder artifact addresses of the sizing build in progress (commit_rowops only).
    _sizing_refs: frozenset[str] = frozenset()

    def __init__(
        self,
        runtime: HBRuntime,
        tenant_id: str,
        dimension: int,
        bind_content: Callable[[str, object], ContentBindingDigest],
        *,
        provider_config: ProviderConfig | None = None,
        bind_content_array: Callable[[str], ContentArrayBinding] | None = None,
    ) -> None:
        self.runtime = runtime
        self.tenant_id = tenant_id
        self.dimension = dimension
        self.bind_content = bind_content
        self._bind_content_array = bind_content_array
        self._provider_config_json = (
            None if provider_config is None else canonical_json(
                ProviderConfig.model_validate(provider_config.model_dump(mode="json"))
            )
        )
        self._boot_writer_lease_json: str | None = None
        self._job_cas_needs_restart = False
        self.mutex = WriterMutex()
        self._owner_probe_lock = threading.Lock()
        self._row_cache = RowCache()
        self._scan_memo = ScanMemo(self._row_cache)
        self._binding_memo = ScanMemo(self._row_cache, budget=ROW_BINDING_BYTES)
        # Reserved before every reference-bearing write except a heartbeat-only
        # finalize lease renewal, which changes nothing a fence may read
        # (``_write_finalize_lease_cas``).
        self.ref_generation = 0
        # Plans whose reserved ``ref_generation`` wrote nothing: HeavenBase rolled
        # their transaction back on a constraint (``_write_plan``).
        self._rolled_back_plans = 0
        self._pending_reference_registry = PendingRecallReferenceRegistry()
        self._artifact_generations: dict[str, int] = {}
        self._artifact_change_tokens: dict[tuple[str | None, str | None], int] = {}

        self._child_change_tokens: dict[str, int] = {}
        self._entity_generations: dict[tuple[str, str], int] = {}
        self._run_change_tokens: dict[tuple[str | None, str | None], int] = {}
        self.embedding_write_state = "stable"
        from aml.store.normalization import NORMALIZED_SCHEMA_SPECS

        self.entities = register_schemas(
            runtime, dimension,
            specs=NORMALIZED_SCHEMA_SPECS,
        )
        self._ws: Any = runtime.workspace
        self._backend: Any = self._ws.backends.get("main")
        self._hook_storage_mutators()
        self._session_mutexes: dict[tuple[str, str], threading.RLock] = {}
        self._run_executors: dict[str, threading.Lock] = {}
        from aml.ingestion.add_admission import AddAdmissionQueue

        self.add_admission = AddAdmissionQueue()

        self._finalize_tickets: WeakValueDictionary[str, FinalizeLeaseTicket] = (
            WeakValueDictionary()
        )

        self._finalize_clocks: WeakKeyDictionary[FinalizeLeaseTicket, Callable[[], int]] = (
            WeakKeyDictionary()
        )
        self._finalize_digests: WeakKeyDictionary[FinalizeLeaseTicket, ContentBindingDigest] = (
            WeakKeyDictionary()
        )
        self._finalize_write_tickets: IdentityMemo[WriteTicket, FinalizeLeaseTicket] = (
            IdentityMemo()
        )
        self._finalize_pool: BenchWorkPool | None = None
        self._add_boot_recovery_open = True
        self._privacy_boot_recovery_open = True
        self._session_boot_recovery_open = True
        self._finalize_boot_recovery_open = True
        self._reserved_privacy_parents: frozenset[str] = frozenset()

    @property
    def normalized_storage(self) -> bool:
        """Return the immutable physical layout selected by this Writer's bootstrap."""
        return True

    def content_array_binding(self, purpose: str) -> ContentArrayBinding:
        """Open the composition root's tenant-bound streaming array digest.

        This is a content-binding capability, not storage or execution authority.
        Callers must keep source hydration inside their registered read lifetime.
        Missing initialization fails closed rather than accumulating an unbounded
        fallback array or recovering key material from a bound method.
        """
        if self._bind_content_array is None:
            raise AMLError("unsupported_variant", "Streaming content binding is not initialized")
        return self._bind_content_array(purpose)

    def session_mutex(self, scope_id: str, session_hash: str) -> threading.RLock:
        """Return the process's sole mutex for this canonical ingestion session.

        Args:
            scope_id: Canonical exact scope identity.
            session_hash: Hash of the unchanged external session identity.

        Returns:
            The shared reentrant session lock, always acquired before the commit mutex.
        """
        with self.mutex:
            return self._session_mutexes.setdefault((scope_id, session_hash), threading.RLock())

    def run_executor(self, run_id: str) -> threading.Lock:
        """Serialize one Run's provider boundary without holding storage/session locks.

        The executor lock is acquired first and survives external calls. It is
        process-local scheduling, while durable session leases and Run fences
        remain the only authority after a restart.
        """
        with self.mutex:
            return self._run_executors.setdefault(run_id, threading.Lock())

    def entity_generation(self, scope_id: str, entity_id: str) -> int:
        """Read the process-local generation of one entity's anchored rows (C.14).

        The Writer advances it before every NL transaction attempt that writes
        or deletes a row anchored to the entity (``row_anchors``), acknowledged
        or not, so a nomination cache that reads it before and after its work
        sees every such write (INV:P12, I10). It keys caches and is never
        authority; a new process starts every generation at zero.
        """
        with self.mutex:
            generations: dict[tuple[str, str], int] = self.__dict__.setdefault(
                "_entity_generations", {},
            )
            return generations.get((scope_id, entity_id), 0)

    def child_change_token(self, parent_run_id: str) -> int:
        """Guard indexed child enumeration against insertion, mutation, and deletion.

        This process-local invalidation token is distinct from the persisted
        work_generation protocol. A new process derives coverage from storage.
        """
        with self.mutex:
            return self._child_change_tokens.get(parent_run_id, 0)

    def artifact_change_token(self, scope: ScopeV1) -> tuple[int, int]:
        """Read global and exact-scope Artifact mutation invalidation scalars.

        Every main Artifact attempt reserves these tokens from its actual old
        and new scope before writing. No body or durable authority is retained;
        a new process must inspect its complete actual history again. Control
        heartbeats and other scopes' Artifacts do not change this scope's token,
        nor does an Add's reception bookkeeping (``_reception_bookkeeping``).
        """
        with self.mutex:
            return (
                self._artifact_change_tokens.get((None, None), 0),
                self._artifact_change_tokens.get((scope.scope_kind, scope.scope_id), 0),
            )

    def _invalidate_artifact_rows(self, plan: Any) -> None:
        schema = self.entities["artifact"].schema()
        reception = _RECEPTION_ROOT.get()
        affected: set[tuple[str | None, str | None]] = set()
        for operation in plan.entity_ops.get("main", ()):
            if operation.entity_id != schema.entity_id:
                continue
            before = self.get_row("artifact", operation.object_id)
            after = (self._ws._row_codec.decode_schema_row(schema, operation.to_row())
                     if operation.is_upsert() else None)
            images = [row for row in (before, after) if row is not None]
            if reception is not None and images and all(
                row["created_by_run_id"] == reception for row in images
            ):
                continue
            affected.update((row["scope_kind"], row["scope_id"]) for row in images)
        for scope in affected:
            self._artifact_change_tokens[scope] = self._artifact_change_tokens.get(scope, 0) + 1

    def _invalidate_child_runs(self, plan: Any) -> None:
        schema = self.entities["run"].schema()
        affected: set[str] = set()
        scopes: set[tuple[str | None, str | None]] = set()
        prospective: dict[str, dict[str, Any]] = {}
        preimages: dict[str, dict[str, Any]] = {}
        reception = _RECEPTION_ROOT.get()
        for operation in plan.entity_ops.get("main", ()):
            if operation.entity_id != schema.entity_id:
                continue
            before = self.get_row("run", operation.object_id)
            after = (self._ws._row_codec.decode_schema_row(schema, operation.to_row())
                     if operation.is_upsert() else None)
            if before is not None:
                preimages[operation.object_id] = before
            if after is not None:
                prospective[operation.object_id] = after
            if operation.object_id == reception:
                # An Add root's own reception bookkeeping (run_change_token).
                continue
            for row in (before, after):
                if row is not None:
                    scopes.add((row["scope_kind"], row["scope_id"]))
                    if row["parent_run_id"] is not None:
                        affected.add(row["parent_run_id"])
        for parent in affected:
            before = preimages.get(parent)
            if before is None:
                before = self.get_row("run", parent)
            after = prospective.get(parent)
            for row in (before, after):
                if row is not None:
                    scopes.add((row["scope_kind"], row["scope_id"]))
            self._child_change_tokens[parent] = self._child_change_tokens.get(parent, 0) + 1
        # A missing historical parent is a permitted nonretaining locator. Its
        # earlier deletion already invalidated any proof that inspected it;
        # mutation invalidation must not introduce another GC prerequisite.
        for scope in scopes:
            self._run_change_tokens[scope] = self._run_change_tokens.get(scope, 0) + 1

    def run_change_token(self, scope: ScopeV1) -> tuple[int, int]:
        """Read Run mutation fences for complete direct-leaf absence proofs.

        Reservations include actual changed Run scopes and their actual old/new
        immediate-parent scopes, so a foreign child cannot escape a parent's
        negative-query fence. The maps retain scalars only; Control heartbeats
        do not affect them and new processes reconstruct proof from storage.
        An Add's reception bookkeeping does not affect them either
        (``_reception_bookkeeping``).
        """
        with self.mutex:
            return (
                self._run_change_tokens.get((None, None), 0),
                self._run_change_tokens.get((scope.scope_kind, scope.scope_id), 0),
            )

    @contextmanager
    def _reception_bookkeeping(self, run_id: str) -> Iterator[None]:
        """Keep one Add root's writes before its ordering out of the scope change tokens.

        Admitting the root, saving its request's source, deleting that saved
        source after ordering or a rejection, and ending an unsaved root whose
        reception allowance passed create no Evidence or visible commit,
        and no scope proof enumerates a parentless Add root or its saved
        source; the ordering commit itself changes the scope as usual. A
        finalize owner therefore keeps its frozen discovery proofs while an Add
        that its fence holds back is received or its unsaved root expires.
        """
        token = _RECEPTION_ROOT.set(run_id)
        try:
            yield
        finally:
            _RECEPTION_ROOT.reset(token)

    def _own_finalize_scope(
        self, plan: Any, owner: FinalizeLeaseTicket,
    ) -> tuple[str, str] | None:
        """Return the owner's scope when a plan writes only its own subtree's Runs and Artifacts.

        Every Run image must descend from the owner and every Artifact image be
        created by the owner or a descendant, all in the owner's scope. The
        owner's own row, any other Run or Artifact, or a plan writing neither
        gives None: such a plan's token advances stay uncredited.
        """
        root = self.get_row("run", owner.owner_run_id)
        if root is None or root["scope_id"] != owner.scope_id:
            return None
        scope: tuple[str, str] = (root["scope_kind"], root["scope_id"])
        kinds = {self.entities[kind].schema().entity_id: kind for kind in ("run", "artifact")}
        prospective: dict[str, dict[str, Any]] = {}
        images: list[tuple[str, dict[str, Any]]] = []
        for operation in plan.entity_ops.get("main", ()):
            kind = kinds.get(operation.entity_id)
            if kind is None:
                continue
            before = self.get_row(kind, operation.object_id)
            after = (self._ws._row_codec.decode_schema_row(
                self.entities[kind].schema(), operation.to_row(),
            ) if operation.is_upsert() else None)
            if kind == "run" and after is not None:
                prospective[operation.object_id] = after
            images.extend((kind, row) for row in (before, after) if row is not None)

        def descends(parent: str | None) -> bool:
            for _ in range(_OWN_FINALIZE_DEPTH):
                if parent is None:
                    return False
                if parent == owner.owner_run_id:
                    return True
                row = prospective.get(parent) or self.get_row("run", parent)
                if row is None:
                    return False
                parent = row["parent_run_id"]
            return False

        for kind, row in images:
            if (row["scope_kind"], row["scope_id"]) != scope:
                return None
            if kind == "run":
                if not descends(row["parent_run_id"]):
                    return None
            elif row["created_by_run_id"] != owner.owner_run_id:
                creator = prospective.get(row["created_by_run_id"]) or self.get_row(
                    "run", row["created_by_run_id"],
                )
                if creator is None or not descends(creator["parent_run_id"]):
                    return None
        return scope if images else None

    def _credit_own_source_advance(
        self, owner: FinalizeLeaseTicket, scope: tuple[str, str], before: tuple[int, int],
    ) -> None:
        """Credit what one own plan just advanced, under the same exclusion as the advance."""
        children = self._child_change_tokens.get(owner.owner_run_id, 0) - before[0]
        artifacts = self._artifact_change_tokens.get(scope, 0) - before[1]
        if children or artifacts:
            credits: WeakKeyDictionary[FinalizeLeaseTicket, tuple[int, int]] = (
                self.__dict__.setdefault("_own_source_advances", WeakKeyDictionary())
            )
            previous = credits.get(owner, (0, 0))
            credits[owner] = (previous[0] + children, previous[1] + artifacts)

    def add_capacity(self, rows: Sequence[RowWrite]) -> int:
        """Materialize reception rows without requiring their not-yet-written references.

        This sizes the visibility transaction before the Add is admitted, so it
        is a measurement: the strict rows are parsed outside every lock and
        their RowOps are prepared under the measurement scope, which probes
        process ownership and never takes the commit mutex (plan section 4).
        The count grants no write authority; the later admission validates
        ownership under the mutex before any of the measured rows is written.

        Args:
            rows: Complete strict provisional visibility post-images. No row is written.

        Returns:
            The prepared main-backend RowOp count for the complete visibility plan.

        Raises:
            AMLError: If routing or the materialized transaction exceeds the Add budget.
        """
        if not rows or any(row.kind not in {"evidence", "run", "control"} for row in rows):
            raise AMLError("invalid_request", "Invalid reception capacity projection")
        parsed = [RowWrite(row.kind, self._parse_row(row.kind, row.row.model_dump(mode="json")))
                  for row in rows]
        with self._measure_scope():
            inputs = self._prepare_row_inputs(parsed)
            plan = self._ws._writer.prepare_upsert_rows(
                inputs, overwrite=True, catalog_mode="merge"
            )
        count = sum(len(operations) for operations in plan.entity_ops.values())
        if set(plan.entity_ops) != {"main"} or count > MAX_TX_ROWS:
            raise AMLError("invalid_request", "Add visibility exceeds its physical row budget")
        return count

    def admit_privacy_erase(
        self,
        scope: ScopeV1,
        business_key: PrivacyEraseInputV1,
        grant: PrincipalGrant,
        *,
        now: int,
        parent_run_id: str | None = None,
    ) -> WriteTicket:
        """Admit one privacy erase of already durable Evidence (D.7).

        Split-secret cleanup is the product producer; source expiry and consent
        withdrawal are selected only by test fixtures (INV P13, Open item O10).

        Args:
            scope: The authorized exact source scope.
            business_key: Sorted existing targets and keyed proofs, never incoming bytes.
            grant: Current principal grant resolved outside AML locks.
            now: Trusted admission time in integer milliseconds.
            parent_run_id: Existing ingestion coordinator for the child branch only.

        Returns:
            The sole cleanup identity, without inventing input or effects on replay.

        Raises:
            AMLError: For authorization, stale source, invalid parent, or conflicting ownership.
        """
        from aml.core.admission import (
            admission_identity,
            privacy_erase_admission,
            privacy_erase_target_locator,
            split_secret_parent_binding,
        )
        from aml.core.privacy_persistence import (
            PRIVACY_ERASE_INPUT,
            IngestionSplitSecretInputV1,
            SourceExpiryInputV1,
        )
        from aml.ingestion.authorization import check_add_grant
        from aml.store.domain import SessionTargetRunRowV1
        from aml.store.indexed import IndexedReader

        frozen = PRIVACY_ERASE_INPUT.validate_python(business_key.model_dump(mode="json"))
        target = privacy_erase_target_locator(frozen, self.bind_content)
        if isinstance(frozen, SourceExpiryInputV1) and frozen.observed_at > now:
            raise AMLError(
                "invalid_request", "A source-expiry selection is observed after its admission",
            )
        admission = privacy_erase_admission(scope, frozen, grant.principal_id, self.bind_content)
        family, digest, _ = admission_identity(admission, self.bind_content)
        with self._write_scope():
            check_add_grant(grant, scope, now)
            lease = self.get_control("lease:writer")
            if lease is None:
                raise AMLError("lease_lost", "Privacy admission has no active Writer")
            ticket = WriteTicket(
                lease["value"]["writer_epoch"],
                None,
                scope.scope_kind,
                scope.scope_id,
                frozen.erase_epoch,
                frozen.consent_epoch,
            )
            existing = IndexedReader(
                self._ws,
                self.entities,
                tenant_id=self.tenant_id,
            ).latest_run("retention", family)
            if not isinstance(frozen, IngestionSplitSecretInputV1):
                if parent_run_id is not None:
                    raise AMLError("binding_mismatch", "A privacy-erase root must be independent")
            else:
                parent = self.get_row("run", parent_run_id or "")
                if parent is None or (
                    parent["run_kind"] != "session_ingestion"
                    or parent["commit_mode"] != "orchestrator"
                    or (parent["scope_kind"], parent["scope_id"])
                    != (scope.scope_kind, scope.scope_id)
                    or split_secret_parent_binding(parent, self.bind_content)
                    != frozen.parent_binding_digest
                ):
                    raise AMLError("binding_mismatch", "Cleanup child lost its frozen parent")
                if existing is None:
                    self.ticket(parent["object_id"])
            if existing is not None:
                if (
                    existing["admission_digest"] != digest
                    or existing["parent_run_id"] != parent_run_id
                ):
                    raise AMLError("operation_conflict", "Privacy family binds another input")
                return self.ticket(existing["object_id"])
            self._admit(ticket)
            if self.counter(f"scope_seq:{scope.scope_id}") != frozen.expected_scope_seq:
                raise AMLError("reception_stale", "Privacy selection changed before admission")
            for affected in frozen.affected:
                row = self.get_row("evidence", affected.evidence_id)
                if row is None or (
                    (row["scope_kind"], row["scope_id"]) != (scope.scope_kind, scope.scope_id)
                    or row["state"] != "active"
                    or self.bind_content("row-state", row) != affected.expected_row_digest
                ):
                    raise AMLError("reception_stale", "Privacy Evidence pre-state changed")
                if isinstance(frozen, SourceExpiryInputV1) and (
                    row["expires_at"] is None or row["expires_at"] > frozen.observed_at
                ):
                    raise AMLError("invalid_request", "A source-expiry target has not expired")
            run = SessionTargetRunRowV1.model_validate(
                {
                    "object_id": new_ulid(),
                    "run_kind": "retention",
                    "commit_mode": "destructive_scan",
                    "status": "queued",
                    "stage": "await_input",
                    "attempt": 0,
                    "idempotency_key": family,
                    "run_generation": 0,
                    "admission_digest": digest,
                    "parent_run_id": parent_run_id,
                    "principal_id": grant.principal_id,
                    **scope.model_dump(mode="json"),
                    "target_kind": "affected_set",
                    "target_id": target,
                    "writer_epoch": ticket.writer_epoch,
                    "admitted_erase_epoch": ticket.erase_epoch,
                    "admitted_consent_epoch": ticket.consent_epoch,
                    "created_at": now,
                }
            )
            result = self._admit_run(run, privacy=True)
            from aml.store.read_model import StoreReadModel

            publisher = StoreReadModel.for_writer(self)
            if publisher is not None:
                try:
                    publisher.publish_privacy(
                        self, scope, None, self.counter("commit_seq"), maintenance=True
                    )
                except BaseException:
                    publisher.fail_closed(discard_finalize_baselines=True)
                    raise
            return result

    def start_privacy_erase(
        self,
        ticket: WriteTicket,
        input_ref: ArtifactId,
        plan_digest: IntegrityDigest,
        *,
        now: int,
        publish: Callable[[int], None],
        checkpoint: Callable[[str], None] | None = None,
    ) -> str:
        """Attach the verified plan and publish a durable fence before any cleanup effect.

        Args:
            ticket: The admitted cleanup owner in the active process epoch.
            input_ref: Its complete prewritten strict input selecting the full plan chain.
            plan_digest: Integrity digest of the complete logical effect stream.
            now: Trusted fence-commit time.
            publish: The sole read-model publisher, called under the commit mutex.
            checkpoint: Optional crash observer, without external provider behavior.

        Returns:
            The exact active maintenance key after durable and in-memory publication.

        Raises:
            AMLError: If the chain, any pre-state, or the optimistic generation changed.
        """
        from aml.core.ingestion_cleanup import validate_cleanup_run_input
        from aml.core.privacy_persistence import ZERO_COUNTS, PrivacyEraseProgressV1
        from aml.store.privacy_plan import (
            load_privacy_plan,
            privacy_targets,
            reconstruct_privacy_batch,
        )

        if ticket.run_id is None or ticket.scope_id is None:
            raise AMLError("invalid_write_mode", "Privacy fence needs a scoped owner")
        body = validate_cleanup_run_input(
            self.read_logical(input_ref, "run_input").model_dump(mode="json")
        )
        frozen = body.business_key
        with self.mutex:
            generation = self.ref_generation
        proof = load_privacy_plan(
            self,
            body.plan_ref,
            owner_run_id=ticket.run_id,
            supplied=frozen,
            expected_plan_digest=plan_digest,
            input_ref=input_ref,
            allow_unattached_input=True,
        )
        for batch in proof.batches:
            if reconstruct_privacy_batch(self, proof, batch.batch_no).already_applied:
                raise AMLError("parity_mismatch", "Unfenced cleanup already contains effect rows")
        anchor = privacy_targets(proof)[0].evidence_id
        key = maintenance_fence_key(ticket.scope_id, anchor, self.bind_content)
        with self._write_scope():
            self._admit(ticket)
            if self.ref_generation != generation:
                raise AMLError(
                    "reception_stale", "Privacy source changed during complete preflight"
                )
            run = self.get_row("run", ticket.run_id)
            if run is None or run["input_ref"] is not None or run["status"] != "queued":
                raise AMLError(
                    "invalid_write_mode", "Only an unattached cleanup can create its fence"
                )
            if self.counter(f"scope_seq:{ticket.scope_id}") != frozen.expected_scope_seq:
                raise AMLError("reception_stale", "Privacy scope changed before its fence")
            progress = PrivacyEraseProgressV1.model_validate(
                {
                    "state": "active",
                    "fence_generation": 1,
                    "owner_run_id": ticket.run_id,
                    "evidence_id": anchor,
                    "reason_code": "privacy_erase",
                    "phase": "prepared",
                    "plan_digest": plan_digest,
                    "cursor": 0,
                    "affected_counts": ZERO_COUNTS.model_dump(mode="json"),
                    "erase_epoch_bumps": 0,
                }
            )
            control = self.materialize_control(
                key, progress.model_dump(mode="json"), 0, now, ticket.run_id
            )
            run.update(
                input_ref=input_ref,
                status="running",
                stage="select",
                started_at=now,
                updated_at=now,
            )
            rows = [RowWrite("run", self._parse_row("run", run)), RowWrite("control", control)]
            rows += self._visibility_counters(ticket, scope_changed=True, updated_at=now)
            _, plan = self._prepare_rows(rows, ticket)
            self._write_plan(plan)
            if checkpoint is not None:
                checkpoint("cleanup_fence_committed")
            publish(self.counter("commit_seq"))
            if checkpoint is not None:
                checkpoint("cleanup_fence_published")
        return key

    def apply_privacy_erase(
        self,
        ticket: WriteTicket,
        fence_key: str,
        *,
        now: int,
        publish: Callable[[CommittedRows], None],
        checkpoint: Callable[[str], None] | None = None,
    ) -> bool:
        """Apply exactly one bounded erase transaction: the next batch, or what one left.

        A batch transaction (``_prepare_commit``) writes the batch's Evidence
        placeholders and candidate expiries, deletes its tombstones and advances
        the scope counters it bumps (D.1, D.7.5). What depends on the Evidence
        the erase has erased (`erase_dependents`) joins it while it fits within
        ``MAX_TX_ROWS``: every Item whose head cites that Evidence is
        compensated with the entities it leaves unanchored (D.7.3 steps 1-5,
        `compensate_erased`; no tombstone and no withheld span, UD4/DM-10), then
        the judge records depending on it are deleted (C.13). What does not fit
        follows in transactions of their own, before the next batch; the fence
        makes them privacy-equivalent to one (D.3 P2). After the last batch and
        its dependents, the records of the scope's ended owners that hold a
        credential the reception detector recognizes leave the same way, whole
        groups at a time, counted in the progress (`credential_records`, D.7.5).
        Each transaction advances the fence's progress and publishes its
        committed rows; the phase turns ``verifying`` with the one that leaves
        nothing to apply.

        Args:
            ticket: The current-epoch owner admitted through its exact maintenance fence.
            fence_key: Its unique strict progress key.
            now: Trusted commit time: an expired candidate's ``terminal_at``.
                The frozen transform time remains unchanged.
            publish: The sole read-model publisher of the committed rows, called
                under the commit mutex.
            checkpoint: Optional crash observer, never an external provider.

        Returns:
            True after one atomic transaction, or False when nothing is left to apply.

        Raises:
            AMLError: For a changed authority, malformed plan, third state, or an
                Item or judge-record group that alone exceeds the row budget.
        """
        from aml.core.ingestion_cleanup import validate_cleanup_run_input
        from aml.core.privacy import expire_candidate
        from aml.core.privacy_persistence import (
            IngestionEvidenceTombstoneV1,
            IngestionTombstoneDeleteV1,
            PrivacyEraseProgressV1,
            ingestion_batch_postcondition_digest,
        )
        from aml.store.extraction_erasure import candidate_expiry_records, erased_extraction_record
        from aml.store.privacy_compensation import (
            ErasureDependents,
            compensate_erased,
            credential_records,
            erase_dependents,
            pending_record_candidates,
        )
        from aml.store.privacy_embeddings import prepare_embedding_scrubs
        from aml.store.privacy_plan import (
            PrivacyBatch,
            load_privacy_plan,
            reconstruct_privacy_batch,
        )

        if ticket.run_id is None:
            raise AMLError("invalid_write_mode", "Cleanup effects require their admitted owner")
        with self._write_scope():
            self._admit(ticket)
        run = self.get_row("run", ticket.run_id)
        if run is None or run["input_ref"] is None:
            raise AMLError("input_required", "Cleanup effects require a selected plan")
        body = validate_cleanup_run_input(
            self.read_logical(run["input_ref"], "run_input").model_dump(mode="json")
        )
        raw_control = self.get_control(fence_key)
        if raw_control is None or not self._owns_privacy_fence(run, raw_control):
            raise AMLError("lease_lost", "Cleanup lost its exact active fence")
        progress = PrivacyEraseProgressV1.model_validate(raw_control["value"])
        proof = load_privacy_plan(
            self,
            body.plan_ref,
            owner_run_id=ticket.run_id,
            supplied=body.business_key,
            expected_plan_digest=progress.plan_digest,
        )
        if progress.cursor > len(proof.batches):
            raise AMLError("parity_mismatch", "Cleanup progress extends beyond its plan")
        if (
            progress.cursor
            and progress.last_completed_digest
            != ingestion_batch_postcondition_digest(
                progress.cursor - 1, proof.batches[progress.cursor - 1].entries
            )
        ):
            raise AMLError("parity_mismatch", "Cleanup cursor has a different batch witness")
        if progress.phase == "verifying":
            if progress.cursor != len(proof.batches):
                raise AMLError("parity_mismatch", "Incomplete cleanup cannot enter verification")
            return False

        def effects(batches: Sequence[PrivacyBatch]) -> tuple[frozenset[str], frozenset[str]]:
            """The Evidence the batches erase and the tombstones they delete."""
            entries = [entry for batch in batches for entry in batch.entries]
            return (
                frozenset(entry.object_ref for entry in entries
                          if isinstance(entry, IngestionEvidenceTombstoneV1)),
                frozenset(entry.object_ref for entry in entries
                          if isinstance(entry, IngestionTombstoneDeleteV1)),
            )

        scope = (run["scope_kind"], run["scope_id"])
        batch = proof.batches[progress.cursor] if progress.cursor < len(proof.batches) else None
        reconstructed = None
        if batch is not None:
            reconstructed = reconstruct_privacy_batch(self, proof, batch.batch_no)
            if reconstructed.already_applied:
                raise AMLError(
                    "parity_mismatch", "Effect rows advanced without their atomic cursor",
                )
        with self._write_scope():
            self._admit(ticket)
            if (
                self.get_control(fence_key) != raw_control
                or self.get_row("run", ticket.run_id) != run
            ):
                raise AMLError("reception_stale", "Cleanup ownership changed during reconstruction")
            # What the applied batches left comes first; then the next batch,
            # whose dependents are those of its own effects.
            dependents = erase_dependents(
                self, scope, *effects(proof.batches[:progress.cursor]), now=now,
            )
            if dependents or batch is None or reconstructed is None:
                batch = reconstructed = None
            else:
                for condition in reconstructed.preconditions:
                    current = self.get_row(condition.entity_kind, condition.object_id)
                    digest = None if current is None else self.bind_content("row-state", current)
                    if digest != condition.digest:
                        raise AMLError(
                            "reception_stale", "Cleanup dependency changed before its batch",
                        )
                dependents = erase_dependents(self, scope, *effects([batch]), now=now)
            # After the last batch and its dependents, the records holding a recognized
            # credential leave as erasure dependents do (D.7.5, C.13 erasure manifest).
            residual = batch is None and not dependents
            holders: frozenset[str] = frozenset()
            if residual:
                found = credential_records(self, scope)
                dependents, holders = ErasureDependents(records=found.groups), found.holders
            # Persist only replay identities and admission codes before any selected
            # extraction payload leaves. A crash before deletion merely leaves a
            # redundant content-free witness; replay uses it only for absent bodies.
            extraction: dict[str, set[str]] = {}
            for group in dependents.records:
                for kind, row in group:
                    if kind != "artifact":
                        continue
                    creator = self.get_row("run", row["created_by_run_id"])
                    if creator is not None and creator["run_kind"] == "finalize":
                        extraction.setdefault(creator["object_id"], set()).add(row["object_id"])
            for owner_run_id, references in sorted(extraction.items()):
                record = erased_extraction_record(self, owner_run_id, references)
                if record.erased_extraction or record.proposals:
                    self.put_logical_artifact(
                        ticket, "decision_set", record, created_at=now, security_level="restricted",
                    )
            deleting_refs = frozenset(
                row["object_id"] for group in dependents.records for kind, row in group
                if kind == "artifact"
            )
            for snapshot in prepare_embedding_scrubs(self, scope, deleting_refs):
                self.put_logical_artifact(
                    ticket, "decision_set", snapshot, created_at=now, security_level="restricted",
                )
            dependents = erase_dependents(
                self, scope, *effects(proof.batches[:progress.cursor + (batch is not None)]),
                now=now,
            )
            if residual:
                found = credential_records(self, scope)
                dependents, holders = ErasureDependents(records=found.groups), found.holders
            # The phase stays applying while records holding a credential remain.
            residual_left = (
                not residual and progress.cursor + (batch is not None) == len(proof.batches)
                and bool(credential_records(self, scope).groups)
            )
            entries = () if batch is None else batch.entries
            bumps = frozenset[str]() if batch is None else batch.bumps
            sequence = self.counter("commit_seq") + 1
            planned: list[RowWrite] = [] if reconstructed is None else [
                *reconstructed.rows,
                *(
                    RowWrite("candidate", self._parse_row(
                        "candidate", expire_candidate(row, commit_seq=sequence, now=now),
                    ))
                    for row in reconstructed.expiring
                ),
            ]
            deleting = () if reconstructed is None else reconstructed.deleting
            pending_records = pending_record_candidates(self, scope, dependents.records)
            expiring_ids = {write.row.model_dump()["object_id"] for write in planned
                           if write.kind == "candidate"}
            expiring_ids.update(entry.object_ref for planned_batch in proof.batches
                                for entry in planned_batch.entries
                                if entry.entity_kind == "candidate")
            # Run, progress Control and at most four scope counters.
            room = MAX_TX_ROWS - len(planned) - len(deleting) - 8
            while True:
                compensation = compensate_erased(
                    self,
                    scope=scope,
                    commit_seq=sequence,
                    now=now,
                    run_id=ticket.run_id,
                    evidence=[
                        write.row.model_dump(mode="json")
                        for write in planned if write.kind == "evidence"
                    ],
                    items=dependents.items,
                    room=room,
                    first=batch is None,
                    policy_release_ref=self._extraction_release_ref(),
                )
                records: list[tuple[str, dict[str, Any]]] = []
                record_expiries: list[RowWrite] = []
                groups = 0
                left = room - len(compensation.rows) - len(compensation.deletes)
                if len(compensation.items) == len(dependents.items):
                    for group in dependents.records:
                        expires = [row for kind, record in group if kind == "artifact"
                                   for row in pending_records.get(record["object_id"], ())
                                   if row["object_id"] not in expiring_ids]
                        size = len(group) + len(expires)
                        if size > left and (
                            groups or compensation.items or batch is not None
                        ):
                            break
                        records += group
                        record_expiries += [RowWrite("candidate", self._parse_row(
                            "candidate", expire_candidate(row, commit_seq=sequence, now=now),
                        )) for row in expires]
                        groups += 1
                        left -= size
                taken = len(compensation.items) + groups
                remaining = len(dependents.items) + len(dependents.records) - taken
                deleted = len(holders & {row["object_id"] for kind, row in records
                                         if kind == "artifact"})
                counts = progress.affected_counts.model_dump()
                for entry in entries:
                    counts[entry.entity_kind] += 1
                counts["item"] += compensation.new_items
                counts["revision"] += sum(
                    1 for write in compensation.rows if write.kind == "revision"
                )
                counts["link"] += sum(1 for write in compensation.rows if write.kind == "link")
                counts["candidate"] += len(record_expiries)
                next_cursor = progress.cursor + (batch is not None)
                done = next_cursor == len(proof.batches) and not remaining and not residual_left
                next_progress = PrivacyEraseProgressV1.model_validate(
                    {
                        **progress.model_dump(mode="json"),
                        "cursor": next_cursor,
                        "phase": "verifying" if done else "applying",
                        "affected_counts": counts,
                        "erase_epoch_bumps": progress.erase_epoch_bumps + ("erase_epoch" in bumps),
                        "last_completed_digest": progress.last_completed_digest
                        if batch is None else ingestion_batch_postcondition_digest(
                            batch.batch_no, entries
                        ),
                        "credential_records_deleted": progress.credential_records_deleted
                        + deleted,
                    }
                )
                control = self.materialize_control(
                    fence_key,
                    next_progress.model_dump(mode="json"),
                    raw_control["version"] + 1,
                    now,
                    ticket.run_id,
                )
                owner = dict(run, stage="scan" if done else "compensate", updated_at=now)
                expiry_records = [
                    self.materialize_logical_artifact_rows(
                        ticket, "decision_set", record, created_at=now,
                        security_level="restricted",
                    )
                    for record in candidate_expiry_records(self, record_expiries)
                ]
                expiry_rows = [row for _body, rows in expiry_records for row in rows]
                try:
                    prepared = self._prepare_commit(
                        ticket, [*planned, *compensation.rows, *record_expiries],
                        deletes=[*deleting, *compensation.deletes],
                        bumps=bumps, journal=None, mark_run=False, updated_at=now,
                        owner_rows=(
                            RowWrite("control", control),
                            RowWrite("run", self._parse_row("run", owner)),
                        ),
                    )
                except AMLError as error:
                    if error.code != "budget_exceeded" or taken <= 1 and batch is None:
                        raise
                    prepared = None
                if prepared is not None and expiry_rows:
                    post = {(item.kind, cast(str, item.row.model_dump()["object_id"])): item.row
                            for item in prepared[0]}
                    post.update((("artifact", row.object_id), row) for row in expiry_rows)
                    for row in expiry_rows:
                        if self.get_row("artifact", row.object_id) is not None:
                            raise AMLError("artifact_digest_collision",
                                           "A derived candidate expiry already exists")
                        self._validate_refs("artifact", row, post)
                    for record, rows in expiry_records:
                        self._validate_body_references(record, rows[-1], _post=post)
                    artifact_plan = self._ws._writer.prepare_upsert_rows(
                        self._prepare_row_inputs([
                            RowWrite("artifact", row) for row in expiry_rows
                        ]),
                        overwrite=False, catalog_mode="fresh",
                    )
                    if set(artifact_plan.entity_ops) != {"main"}:
                        raise AMLError("parity_mismatch", "Expiry records left the main backend")
                    prepared[2].entity_ops["main"].extend(artifact_plan.entity_ops["main"])
                    for route, operations in artifact_plan.catalog_ops.items():
                        prepared[2].catalog_ops.setdefault(route, []).extend(operations)
                if prepared is not None and records:
                    # C.13, INV P06: the judge records depending on erased
                    # Evidence leave with it, whole groups at a time.
                    self._plan_deletes(prepared[2], records)
                if prepared is not None and (
                    len(prepared[2].entity_ops["main"]) <= MAX_TX_ROWS
                    or taken <= 1 and batch is None
                ):
                    break
                if taken == 0:
                    raise AMLError(
                        "budget_exceeded", "An erase batch exceeds its transaction budget",
                    )
                room = min(room, len(compensation.rows) + len(compensation.deletes)
                           + len(records) + len(record_expiries)) // 2
            assert prepared is not None
            if expiry_rows:
                self._artifact_generations[ticket.run_id] = (
                    self._artifact_generations.get(ticket.run_id, 0) + 1
                )
            committed = self._commit_prepared(ticket, *prepared)
            if checkpoint is not None:
                checkpoint("cleanup_batch_committed")
            publish(committed)
            if checkpoint is not None:
                checkpoint("cleanup_batch_published")
        return True

    def finish_privacy_erase(
        self,
        ticket: WriteTicket,
        fence_key: str,
        *,
        now: int,
        publisher: StoreReadModel,
        checkpoint: Callable[[str], None] | None = None,
    ) -> ArtifactId:
        """Verify cleanup and atomically pair its terminal receipt with fence removal.

        Args:
            ticket: Exact active cleanup owner, after all bounded batches.
            fence_key: Its unique strict maintenance progress key.
            now: Trusted completion time.
            publisher: The exact registered read-model owner. Its real drained-root
                and residual checks run outside the Writer mutex; it publishes the
                complete clean generation before this method returns.
            checkpoint: Optional crash observer without external side effects.

        Returns:
            The verified, same-owner redaction receipt after clean-root publication.

        Raises:
            AMLError: If any postcondition, residual scan, ownership, or generation fails.
                An uncertain commit requires owner recovery before any further service.
        """
        from aml.core.artifacts import ReceiptResultIndexV1
        from aml.core.ingestion_cleanup import validate_cleanup_run_input
        from aml.core.privacy_persistence import (
            NOT_COVERED,
            CredentialFindingsV1,
            PrivacyEraseProgressV1,
            RedactionReceiptV1,
            ingestion_batch_postcondition_digest,
        )
        from aml.core.query import ScopeV1
        from aml.core.run_inputs import SplitSecretChildRunInputV1
        from aml.store.privacy_plan import (
            load_privacy_plan,
            planned_erase_epoch_bumps,
            privacy_targets,
            verify_privacy_completion,
        )
        from aml.store.read_model import StoreReadModel

        if ticket.run_id is None or ticket.scope_id is None:
            raise AMLError("invalid_write_mode", "Privacy completion requires its scoped owner")
        if StoreReadModel.for_writer(self) is not publisher:
            raise AMLError(
                "binding_mismatch", "Privacy completion requires its registered publisher"
            )
        with self._write_scope():
            self._admit(ticket)
            generation = self.ref_generation
            run = self.get_row("run", ticket.run_id)
            control = self.get_control(fence_key)
        if run is None or control is None or not self._owns_privacy_fence(run, control):
            raise AMLError("lease_lost", "Privacy completion lost its exact fence")
        body = validate_cleanup_run_input(
            self.read_logical(run["input_ref"], "run_input").model_dump(mode="json")
        )
        progress = PrivacyEraseProgressV1.model_validate(control["value"])
        proof = load_privacy_plan(
            self,
            body.plan_ref,
            owner_run_id=ticket.run_id,
            supplied=body.business_key,
            expected_plan_digest=progress.plan_digest,
        )
        if (
            progress.phase != "verifying"
            or progress.cursor != len(proof.batches)
            or progress.last_completed_digest
            != ingestion_batch_postcondition_digest(
                progress.cursor - 1, proof.batches[-1].entries
            )
        ):
            raise AMLError("parity_mismatch", "Privacy completion has no complete cursor proof")
        if progress.erase_epoch_bumps != planned_erase_epoch_bumps(proof, progress.cursor):
            raise AMLError("parity_mismatch", "Privacy completion lost its erase_epoch bumps")
        counts = verify_privacy_completion(self, proof)
        if counts != progress.affected_counts:
            raise AMLError("parity_mismatch", "Privacy receipt counts differ from applied effects")
        parent_before = None
        if isinstance(body, SplitSecretChildRunInputV1):
            parent = self.get_row("run", run["parent_run_id"])
            if parent is None:
                raise AMLError("binding_mismatch", "Privacy completion lost its parent")
            if parent["input_ref"] is None:
                from aml.ingestion.privacy_recovery import privacy_rejected_parent

                parent_before = privacy_rejected_parent(
                    self, parent["object_id"], ticket.run_id, allow_active_child=True
                )
        retained = self.scan_privacy_residuals((run["scope_kind"], run["scope_id"]))
        scope = ScopeV1.model_validate(
            {"scope_kind": ticket.scope_kind, "scope_id": ticket.scope_id}
        )
        targets = privacy_targets(proof)
        publisher.verify_privacy_projection(scope, [target.evidence_id for target in targets])
        publisher.verify_owned_content(scope, [], retained=bool(retained))
        findings = CredentialFindingsV1(
            records_deleted=progress.credential_records_deleted, retained=retained,
        )
        with self._write_scope():
            self._admit(ticket)
            if (
                self.ref_generation != generation
                or self.get_control(fence_key) != control
                or self.get_row("run", ticket.run_id) != run
            ):
                raise AMLError("reception_stale", "Privacy verification changed before completion")
            # Admission excludes ordinary checkpoint and successor selection
            # until this owner completes. Its selected receipt therefore binds
            # erasure ordering to this final transaction, not to wall time or
            # an unselected artifact prewrite.
            completion_sequence = self.counter("commit_seq") + 1
            receipt = RedactionReceiptV1(
                receipt_kind="redaction",
                reason_code=body.business_key.reason_code,
                status="verified",
                affected_digests=sorted(target.expected_row_digest for target in targets),
                counts=counts,
                not_covered=NOT_COVERED,
                completion_commit_seq=completion_sequence,
                credential_findings=findings,
            )
            reference = self._put_physical(
                ticket, "receipt", receipt, "restricted", now, privacy_completion=True
            )
            result = self._put_physical(
                ticket,
                "result_index",
                ReceiptResultIndexV1(receipt_ref=reference),
                "restricted",
                now,
                privacy_completion=True,
            )
            if checkpoint is not None:
                checkpoint("cleanup_receipt_prewritten")
            if self.counter("commit_seq") + 1 != completion_sequence:
                raise AMLError(
                    "reception_stale", "Privacy completion sequence changed before selection"
                )
            run.update(
                result_ref=result, status="succeeded", stage="scan", updated_at=now, finished_at=now
            )
            rows = [RowWrite("run", self._parse_row("run", run))]
            if parent_before is not None:
                parent = parent_before.model_dump(mode="json")
                if self.get_row("run", parent_before.object_id) != parent:
                    raise AMLError(
                        "reception_stale", "Privacy rejection parent changed before completion"
                    )
                parent.update(
                    status="failed",
                    stage="settle",
                    failure_code="policy_reject",
                    updated_at=now,
                    finished_at=now,
                )
                rows.append(RowWrite("run", self._parse_row("run", parent)))
            rows += self._visibility_counters(ticket, scope_changed=True, updated_at=now)
            _, plan = self._prepare_rows(rows, ticket)
            target = self._ws._row_inputs.delete_target(
                (self.entities["control"], control["object_id"])
            )
            deletion = self._ws._writer.prepare_delete_rows([target])
            if deletion.deleted != [True]:
                raise AMLError("parity_mismatch", "Privacy fence disappeared during completion")
            deletes = self._ws._writer.delete_ops(deletion)
            main_id = self.entities["control"].schema().entity_id
            for route, operations in deletes.items():
                for operation in operations:
                    if operation.entity_id == main_id:
                        if route != "main":
                            raise AMLError("parity_mismatch", "Privacy fence has an invalid route")
                        plan.entity_ops["main"].append(operation)
                    else:
                        plan.catalog_ops.setdefault(route, []).append(operation)
            if len(plan.entity_ops["main"]) > MAX_TX_ROWS:
                raise AMLError("budget_exceeded", "Privacy completion exceeds its atomic budget")
            # All AML rows share the pinned HB main uniqueness owner. Ordered
            # upsert/delete RowOps therefore execute in one SQLBackend transaction.
            self._write_plan(plan)
            if checkpoint is not None:
                checkpoint("cleanup_completed")
            publisher.publish_privacy(
                self, scope, None, self.counter("commit_seq"), maintenance=False
            )
            if checkpoint is not None:
                checkpoint("cleanup_clear_published")
            return reference

    def scan_privacy_residuals(
        self, scope: tuple[str, str] | None = None,
    ) -> dict[ResidualKind, int]:
        """Inspect every decoded table without retaining or reporting matched bytes.

        The same supported signatures as Add are checked in row strings and
        decoded Artifact payloads. Each row is parsed by its registered strict
        variant first, so its declared references, local identities and keyed
        digests, which are product-generated, are not inspected. WAL, backups,
        provider copies, and previously delivered caller-owned data are
        explicitly outside this receipt's coverage.

        A privacy erase passes its scope. A credential found there is a finding,
        counted by stored kind: what remains after the erase deleted the records
        holding one (`privacy_compensation.credential_records`) is what no rule of
        that erase removes (`CredentialFindingsV1.retained`). One found in another
        scope is outside that erase's coverage: it holds nothing of the erased
        scope, and only that scope's own erase may change it. Rows of no scope
        count for every scope. Without a scope, any credential fails the scan.

        Args:
            scope: The erase's exact ``(scope_kind, scope_id)``, or None.

        Returns:
            The rows of the scope, or of no scope, that hold a recognized
            credential, by stored kind (D.7.5); empty without a scope.

        Raises:
            AMLError: If a row is outside its strict schema or a logical body is
                unreadable, or, without a scope, if a recognized credential remains.
        """
        from aml.core.privacy_scan import PrivacyResidualScanner
        from aml.ingestion.boundary import contains_recognized_credential

        found: dict[ResidualKind, int] = {}
        counted = [False]
        owners: dict[tuple[str, str], str | None] = {}

        def report(kind: str) -> None:
            if counted[0]:
                key = cast("ResidualKind", kind)
                found[key] = found.get(key, 0) + 1

        def owner_scope(kind: str, object_id: str) -> str | None:
            if (kind, object_id) not in owners:
                owner = self.get_row(kind, object_id)
                owners[kind, object_id] = None if owner is None else scope_of(kind, owner)
            return owners[kind, object_id]

        def scope_of(kind: str, row: Mapping[str, Any]) -> str | None:
            """The scope a row belongs to; None for a row of no scope (Controls)."""
            if "scope_id" in row:
                return row["scope_id"]
            if kind == "revision":
                return owner_scope("item", row["memory_id"])
            if kind == "recall":
                if row.get("parent_recall_id") is not None:
                    return owner_scope("recall", row["parent_recall_id"])
                allowed = cast(dict[str, dict[str, str]] | None, row.get("allowed_scopes"))
                return None if allowed is None else allowed["requested_scope"]["scope_id"]
            return None

        scanner = PrivacyResidualScanner(
            recognizes=contains_recognized_credential,
            report=None if scope is None else report,
        )
        for kind in self.entities:
            for row in self.iter_rows(kind):
                counted[0] = scope is not None and scope_of(kind, row) in (scope[1], None)
                if kind == "artifact":
                    scanner.inspect_artifact(row, load_logical=self.read_logical)
                else:
                    scanner.inspect_value(self._parse_row(kind, row), kind=kind)
        scanner.finish()
        return dict(sorted(found.items()))

    def privacy_rowops(
        self,
        rows: Sequence[RowWrite],
        *,
        deletes: Sequence[RowDelete] = (),
        overhead: Sequence[RowWrite],
    ) -> int:
        """Count the real bounded privacy transaction without creating planned rows.

        Args:
            rows: The closed planner's temporary Evidence placeholder and candidate
                expiry post-images.
            deletes: The stored tombstones the batch deletes (D.7.5).
            overhead: Full intended Run/progress/counter post-images for this batch,
                the counters its bumps advance included.

        Returns:
            The prepared main-database RowOp count. SQL work inside each backend
            operation and the separate Catalog projection are different metrics.

        Raises:
            AMLError: If the projection has an unsupported kind or storage route.
                The caller packs groups using this count and enforces MAX_TX_ROWS.
        """
        if (
            not rows
            or any(row.kind not in {"evidence", "candidate"} for row in rows)
            or any(delete.kind != "tombstone" for delete in deletes)
            or not overhead
            or any(row.kind not in {"run", "control"} for row in overhead)
        ):
            raise AMLError("invalid_write_mode", "Privacy capacity requires domain and owner rows")
        with self._write_scope():
            count = len(self._capacity_plan([*rows, *overhead]).entity_ops["main"])
            if deletes:
                removed: list[tuple[str, dict[str, Any]]] = []
                for delete in deletes:
                    stored = self.get_row(delete.kind, delete.object_id)
                    if stored is None:
                        raise AMLError("reference_missing", "A deleted row does not exist")
                    removed.append((delete.kind, stored))
                count += len(self._delete_operations(removed)[0])
            return count

    def _materialized_rowops(self, rows: Sequence[RowWrite]) -> int:
        """Count one real main plan without admitting or persisting its prospective rows."""
        with self._write_scope():
            return len(self._capacity_plan(rows).entity_ops["main"])

    def _capacity_plan(self, rows: Sequence[RowWrite]) -> Any:
        """Prepare a complete private capacity frame inside the owned write scope."""
        identities = [
            (row.kind, row.row.model_dump(include={"object_id"})["object_id"])
            for row in rows
        ]
        if len(identities) != len(set(identities)):
            raise AMLError("invalid_request", "Capacity contains duplicate row identities")
        parsed: list[RowWrite] = []
        for row in rows:
            # Keep the same JSON value boundary as the cache key, while retaining
            # nonfinite numbers for rejection instead of normalizing them to null.
            encoded = to_json(row.row.model_dump(mode="python"), inf_nan_mode="constants")
            parsed.append(RowWrite(row.kind, self._parse_row(row.kind, json.loads(encoded))))
        from aml.store.normalization import project_normalized_capacity_row

        inputs = [self._ws._row_inputs.upsert_row(
            self.entities[row.kind],
            project_normalized_capacity_row(row.kind, row.row, tenant_id=self.tenant_id), {},
        ) for row in parsed]
        plan = self._ws._writer.prepare_upsert_rows(inputs, overwrite=True, catalog_mode="merge")
        if set(plan.entity_ops) != {"main"}:
            raise AMLError("invalid_write_mode", "Capacity transaction is not routed to main")
        return plan

    def _check_add_evidence(
        self,
        ticket: WriteTicket,
        request: AddRequestV1,
        parts: Sequence[EvidencePart],
        evidence: Sequence[EvidenceRowV1],
        *,
        profile: Profile,
        embedding_space: str,
        expected_security: str,
    ) -> None:
        """Refuse Evidence that differs from its source parts, the policy, or a stored row.

        Args:
            ticket: The admitted external root the rows must belong to.
            request: The strict unchanged source payload.
            parts: The request's physical parts in ordinal and part order.
            evidence: The rows offered for those parts, in the same order.
            profile: The trusted product or bench configuration.
            embedding_space: The active embedding space every row must name.
            expected_security: The classification of the complete source text.

        Raises:
            AMLError: ``binding_mismatch`` when the rows do not cover the parts
                exactly or a row differs from its part, the policy, or an
                existing Evidence identity.
        """
        if len(parts) != len(evidence):
            raise AMLError("binding_mismatch", "Evidence does not cover the complete request")
        # One batched identity read for the whole request instead of one storage
        # read per part: under plan section 9 load the per-part reads were the
        # single largest storage read site (normal-06e: 9,145 reads in six
        # minutes, every one returning no row).
        existing = self.get_rows("evidence", [row.object_id for row in evidence])
        for part, row, present in zip(parts, evidence, existing, strict=True):
            if (
                row.session_id != request.session_id
                or row.request_id != request.request_id
                or row.scope_id != ticket.scope_id
                or row.scope_kind != ticket.scope_kind
                or row.ordinal != part.ordinal
                or row.part_no != part.part_no
                or row.text != part.text
                or row.byte_start != part.byte_start
                or row.byte_end != part.byte_end
                or row.role != part.role
                or row.source_observed_at != part.source_observed_at
                or row.state != "active"
                or row.embedding_space != embedding_space
                or row.route_state != ("indexed" if profile == "bench" else "pending")
                or row.security_level != expected_security
                or row.trust_tier
                != {"user": "reported", "assistant": "inferred", "system": "untrusted"}.get(
                    part.role
                )
                or present is not None
            ):
                raise AMLError("binding_mismatch", "Evidence differs from its source or policy")

    def _prepare_add_commit(
        self,
        ticket: WriteTicket,
        request: AddRequestV1,
        binding: AddAdmissionBinding,
        evidence: Sequence[EvidenceRowV1],
        grant: PrincipalGrant,
        *,
        profile: Profile,
        ledger: ControlRowV1 | None,
        ledger_head: LedgerHead,
        clock: Callable[[], int],
    ) -> _PreparedAddCommit:
        """Prepare everything of one Add commit that needs no Writer mutex.

        The canonical admission, source parts, security classification,
        ledger value and overflow block depend only on the
        arguments. The Evidence rows are compared with their parts, then
        parsed and validated by the same strict batch validation the commit
        batch used under the mutex, as a borrowed read: process ownership is
        probed (ADR 0037) and the storage epoch is taken before the rows they
        are compared against are read, so the critical section proves those
        rows unchanged instead of validating the vectors again (ADR 0036).
        Nothing here writes, admits, or assigns the receipt time and
        generation.

        Args:
            ticket: Empty admitted external root, under its session mutex.
            request: Strict unchanged source payload.
            binding: Principal-bound canonical admission and payload identities.
            evidence: Fully embedded source parts without their receipt stamps.
            grant: Fresh exact authority resolved outside every AML lock.
            profile: Trusted product or bench configuration.
            ledger: Captured reception Control, or None for the first accepted request.
            ledger_head: Verified overflow chain associated with that Control.
            clock: Trusted clock, read once here for the grant precheck.

        Returns:
            The prepared frame the critical section commits or discards.

        Raises:
            AMLError: For an invalid grant, ledger projection, Evidence
                binding, or strict row validation, with the codes the commit
                raised for them under the mutex.
        """
        from aml.core.query import ScopeV1
        from aml.ingestion.authorization import check_add_grant
        from aml.ingestion.boundary import build_add_admission, split_evidence
        from aml.ingestion.receipts import overflow_block
        from aml.ingestion.source_security import classify_source_security

        check_add_grant(
            grant,
            ScopeV1(scope_kind="user", scope_id=canonical_scope_id("user", request.user_id)),
            clock(),
            binding.admission.principal_id,
        )
        canonical = build_add_admission(
            request,
            principal_id=grant.principal_id,
            profile=profile,
            bind_content=self.bind_content,
        )
        parts = split_evidence(request)
        expected_security = classify_source_security(
            "".join(message.content for message in request.messages), profile=profile
        ).security_level
        value = RequestLedgerValueV1.model_validate(
            {"recent": {}} if ledger is None else ledger.value.model_dump(mode="json")
        )
        overflow = overflow_block(value, ledger_head, self.bind_content)
        token = self.storage_epoch
        self.probe_storage_owner()
        embedding = self.get_control("policy:embedding_space")
        if embedding is None:
            raise AMLError("embedding_space_mismatch", "Embedding state changed before reception")
        self._check_add_evidence(
            ticket,
            request,
            parts,
            evidence,
            profile=profile,
            embedding_space=embedding["value"]["embedding_space"],
            expected_security=expected_security,
        )
        parsed = tuple(self._validate_rows([RowWrite("evidence", row) for row in evidence], ticket))
        # Every row the borrowed validation read: the admission rows (Writer
        # lease, Run, scope registration, tombstone, epoch counters), the
        # embedding policy, the evidence rows and their reference targets.
        admission_keys: set[tuple[str, str]] = {
            ("control", control_id("lease:writer")),
            ("control", control_id(f"scope:{ticket.scope_id}")),
            ("control", control_id(f"tomb:{ticket.scope_id}")),
            ("control", control_id(f"erase_epoch:{ticket.scope_id}")),
            ("control", control_id(f"consent_epoch:{ticket.scope_id}")),
        }
        if ticket.run_id is not None:
            admission_keys.add(("run", ticket.run_id))
        read_keys = frozenset(
            admission_keys
            | {("control", control_id("policy:embedding_space"))}
            | {("evidence", row.object_id) for row in evidence}
            | {
                (ref.spec.target_kind, str(ref.target_id))
                for item in parsed
                for ref in contextual_references(item.row)
            }
        )
        return _PreparedAddCommit(
            canonical=canonical,
            parts=parts,
            expected_security=expected_security,
            ledger_value=value,
            overflow=overflow,
            ledger_json=canonical_json(ledger),
            evidence=parsed,
            read_keys=read_keys,
            token=token,
        )

    def receive_add_input(
        self, ticket: WriteTicket, binding: AddAdmissionBinding, grant: PrincipalGrant, *,
        clock: Callable[[], int],
    ) -> SessionAdmissionRunInputV1:
        """Save exact source bytes before indexing, without changing the semantic prefix.

        The caller holds its existing session admission mutex and has completed
        secret preflight. Replays retain the same Run, source bytes and original
        reception metadata. This bookkeeping does not create Evidence or a visible
        commit, so a preceding semantic owner keeps its frozen input.
        """
        from aml.core.query import ScopeV1
        from aml.core.request_budget import DEFAULT_COMPLETION_WORK_BUDGET_MS
        from aml.core.run_inputs import MessagesRequestPayloadV1, SessionAdmissionRunInputV1
        from aml.ingestion.authorization import check_add_grant

        with self._write_scope():
            self._admit(ticket)
            run = self.get_row("run", ticket.run_id or "")
            if run is None or run["run_kind"] != "session_ingestion" or (
                run["status"] != "queued" or run["input_ref"] is not None
                or run["admission_digest"] != binding.admission_digest
            ) or not isinstance(binding.admission.request_payload, MessagesRequestPayloadV1):
                raise AMLError("binding_mismatch", "Reception lost its exact source admission")
            scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])
            check_add_grant(grant, scope, clock(), run["principal_id"])
            received = [body for _, body in self.find_logical_candidates(
                run["object_id"], "run_input",
            ) if isinstance(body, SessionAdmissionRunInputV1)]
            expected = SessionAdmissionRunInputV1.model_validate({
                "run_kind": "session_ingestion", "input_phase": "admission",
                "business_key": binding.admission.raw_business_key,
                "request_payload": binding.admission.request_payload,
                "admission_digest": binding.admission_digest,
                "payload_digest": binding.payload_digest,
                "received_at": run["created_at"],
                "reception_sequence": received[0].reception_sequence if received else (
                    self.add_admission.reception_sequence(run["scope_id"], run["object_id"])
                ),
                "work_deadline_at": run["created_at"] + DEFAULT_COMPLETION_WORK_BUDGET_MS,
            })
            if received:
                if received != [expected]:
                    raise AMLError("binding_mismatch", "Received Add changed its saved input")
                self._register_received_add(run, expected, clock())
                return expected
            # Bound entities are checked once, when the input is first saved (A.3.5:
            # active at reception). A resend of a saved input keeps it even if an entity
            # was erased since, so a crash-resend never loses an admitted input (E02).
            self._require_active_entities(scope, binding.admission.request_payload)
            self.add_admission.require_reception_capacity(
                run["scope_id"], run["object_id"], now_at=clock(),
            )
            with self._reception_bookkeeping(run["object_id"]):
                self.put_logical_artifact(
                    ticket, "run_input", expected, security_level="restricted",
                    created_at=run["created_at"],
                )
            self._register_received_add(run, expected, clock())
            return expected

    def _register_received_add(
        self, run: Mapping[str, Any], received: SessionAdmissionRunInputV1, now: int,
    ) -> None:
        """Restore one accepted input's original queue position, independent of client age."""
        from aml.core.run_inputs import MessagesRequestPayloadV1

        if not isinstance(received.request_payload, MessagesRequestPayloadV1):
            raise AMLError("binding_mismatch", "Add reception requires saved messages")
        if received.received_at is None:
            raise AMLError("binding_mismatch", "Add reception lost its original reception time")
        times = [message.timestamp for message in received.request_payload.messages]
        source_time = min((value for value in times if value is not None), default=None)
        self.add_admission.register(
            run["scope_id"], run["object_id"], received.received_at,
            source_time=source_time if all(value is not None for value in times) else None,
            work_deadline_at=received.work_deadline_at,
            now_at=now,
            identity=(received.business_key.session_id, received.business_key.request_id),
            reception_sequence=received.reception_sequence,
        )

    def _require_active_entities(
        self, scope: ScopeV1, payload: MessagesRequestPayloadV1,
    ) -> None:
        """Refuse an SDK entity binding whose entity is not active in the scope (E.2.5).

        Each distinct bound entity is one bounded single-row read of its
        authoritative row (``IndexedReader.entity_header``, cross-Q15), once,
        just before the input is first saved (ingest-Q10); a resend of a saved
        input is not checked again. An absent, an erased and another scope's
        entity get the same refusal, so a binding never reveals whether an
        entity exists elsewhere.
        """
        entities = sorted({
            binding.entity_id
            for message in payload.messages for binding in message.entity_bindings or ()
        })
        if not entities:
            return
        reader = self.indexed_reader()
        for entity in entities:
            if reader.entity_header(scope, entity, state="active") is None:
                raise AMLError("scope_denied", "An entity binding is not available in this scope")

    def discard_received_add_input(self, run_id: str) -> None:
        """Delete temporary raw transport only after the exact Evidence receipt exists."""
        from aml.core.run_inputs import SessionDurableRunInputV1

        with self._write_scope():
            run = self.get_row("run", run_id)
            if run is None or run["input_ref"] is None:
                raise AMLError("binding_mismatch", "Reception cleanup lacks its durable receipt")
            receipt = self.read_logical(run["input_ref"], "run_input")
            if not isinstance(receipt, SessionDurableRunInputV1):
                raise AMLError("binding_mismatch", "Reception cleanup selected another input")
            self._delete_received_inputs(run, receipt)

    def discard_rejected_add_input(self, run_id: str) -> None:
        """Delete the source bytes that a rejected, never-ordered reception saved.

        Authority, scope-epoch and secret rejections end the request before it is
        ordered, so its saved source has no further use and must not outlive the
        rejection. A deferral (vector failure, held fence, changed embedding
        state) keeps it for the same request's resend until the original
        deadline, when ``abandon_add_run`` voids the request and deletes it. The
        request deletes only its own raw transport, so no admission is required;
        an ordered reception keeps its receipt-bound cleanup instead, and a Run
        that a scope removal already deleted leaves nothing to delete.

        Args:
            run_id: The rejected ordinary Add root.

        Raises:
            AMLError: ``binding_mismatch`` if the Run is not an ordinary Add root or a
                saved input binds another admission.
        """
        with self._write_scope():
            run = self.get_row("run", run_id)
            if run is None:
                return
            if (run["run_kind"], run["commit_mode"]) != ("session_ingestion", "orchestrator"):
                raise AMLError("binding_mismatch", "Rejected reception has another Run kind")
            if run["input_ref"] is None:
                self._delete_received_inputs(run, None)

    def _delete_received_inputs(
        self, run: dict[str, Any], receipt: SessionDurableRunInputV1 | None,
        following: dict[str, Any] | None = None,
    ) -> None:
        """Delete every saved admission input of ``run`` inside the caller's write scope.

        Each input must bind the Run's admission and, once ordered, its durable
        ``receipt`` (admission digest, payload digest and business key). The
        write is the root's reception bookkeeping. ``following`` is the Run's
        terminal header, written with the deletion in one plan (an expired
        reception); without it nothing is written when no input is saved.
        """
        from aml.core.artifacts import ManifestV1
        from aml.core.run_inputs import SessionAdmissionRunInputV1

        refs: list[str] = []
        for reference, body in self.find_logical_candidates(run["object_id"], "run_input"):
            if not isinstance(body, SessionAdmissionRunInputV1):
                continue
            if body.admission_digest != run["admission_digest"] or receipt is not None and (
                body.admission_digest, body.payload_digest, body.business_key,
            ) != (receipt.admission_digest, receipt.payload_digest, receipt.business_key):
                raise AMLError("binding_mismatch", "Reception cleanup changed saved input")
            raw = self.get_row("artifact", reference)
            assert raw is not None
            if raw["artifact_kind"] == "manifest":
                refs.extend(ManifestV1.model_validate_json(raw["body"]).parts)
            refs.append(reference)
        if not refs and following is None:
            return
        _, plan = self._prepare_rows(
            [RowWrite("run", self._parse_row("run", run if following is None else following))],
            None,
        )
        if refs:
            deletion = self._ws._writer.prepare_delete_rows([
                self._ws._row_inputs.delete_target((self.entities["artifact"], reference))
                for reference in refs
            ])
            if deletion.deleted != [True] * len(refs):
                raise AMLError("parity_mismatch", "Saved reception input changed during cleanup")
            for route, operations in self._ws._writer.delete_ops(deletion).items():
                if route == "main":
                    plan.entity_ops.setdefault(route, []).extend(operations)
                else:
                    plan.catalog_ops.setdefault(route, []).extend(operations)
        with self._reception_bookkeeping(run["object_id"]):
            self._write_plan(plan)
        self.add_admission.discard(run["scope_id"], run["object_id"])

    def commit_add(
        self,
        ticket: WriteTicket,
        request: AddRequestV1,
        binding: AddAdmissionBinding,
        evidence: Sequence[EvidenceRowV1],
        grant: PrincipalGrant,
        *,
        profile: Profile,
        ledger: ControlRowV1 | None,
        ledger_head: LedgerHead,
        embedding_token: IntegrityDigest,
        clock: Callable[[], int],
        publish: Callable[[SessionDurableRunInputV1, Sequence[EvidenceRowV1]], None],
        checkpoint: Callable[[str], None] | None = None,
        completion_required: bool = False,
    ) -> str:
        """Commit one ordinary Add with its ledger and historical no-content witness.

        ``_prepare_add_commit`` runs the argument-only work and the strict
        validation of the Evidence rows before the Writer mutex. The critical
        section rereads every live precondition, proves the rows the borrowed
        validation compared against unchanged or validates them once more
        under the mutex, assigns the receipt time and visibility generation,
        prewrites the receipt, plans and writes the one main batch, and
        publishes. Argument and content refusals therefore surface before the
        mutex; live-state refusals keep their order inside it.

        Args:
            ticket: Empty admitted external root, under its session mutex.
            request: Strict unchanged source payload.
            binding: Principal-bound canonical admission and payload identities.
            evidence: Fully embedded source parts; receipt time and sequence are assigned here.
            grant: Fresh exact authority resolved outside every AML lock.
            profile: Trusted product or bench configuration.
            ledger: Captured reception Control, or None for the first accepted request.
            ledger_head: Verified overflow chain associated with that Control.
            embedding_token: Frozen active embedding state binding before the provider call.
            clock: Trusted clock, read inside the final commit mutex for the receipt time.
            publish: Owner callback that swaps the exact complete committed root under this lock.
            checkpoint: Optional fault-injection observer; never an external provider.
            completion_required: Freeze the original completion deadline for a
                completed-prefix serving Add; receipt-only SDK inputs retain
                their independent later owner allowance.

        Returns:
            The attached durable run_input Artifact identity.

        Raises:
            AMLError: For stale authority, source, ledger, embedding, or invalid state.
        """
        from aml.core.embedding import embedding_state_token
        from aml.core.query import ScopeV1
        from aml.core.run_inputs import (
            SessionAdmissionRunInputV1,
            SessionInventoryDurableRunInputV2,
        )
        from aml.core.technical_types import RequestProgressV1
        from aml.core.timing import measure
        from aml.ingestion.authorization import check_add_grant
        from aml.ingestion.inventory import freeze_source_coordinates, source_coordinate
        from aml.store.input_fence import require_input_mutation_allowed

        prepared = self._prepare_add_commit(
            ticket,
            request,
            binding,
            evidence,
            grant,
            profile=profile,
            ledger=ledger,
            ledger_head=ledger_head,
            clock=clock,
        )
        with self._write_scope(), measure("locked_segment"):
            now = clock()
            check_add_grant(
                grant,
                ScopeV1(scope_kind="user", scope_id=canonical_scope_id("user", request.user_id)),
                now,
                binding.admission.principal_id,
            )
            self._admit(ticket)
            run = self.get_row("run", ticket.run_id or "")
            if (
                run is None
                or prepared.canonical != binding
                or (
                    run["input_ref"] is not None
                    or run["status"] != "queued"
                    or run["commit_mode"] != "orchestrator"
                    or run["admission_digest"] != binding.admission_digest
                    or run["idempotency_key"] != binding.idempotency_key
                    or run["principal_id"] != grant.principal_id
                )
            ):
                raise AMLError("binding_mismatch", "Add no longer has its original empty admission")
            self.add_admission.require_ordering_turn(run["scope_id"], run["object_id"])
            finalize = self.get_control(f"lease:finalize:{ticket.scope_id}")
            if finalize is not None and finalize["value"]["state"] == "preparing":
                raise AMLError("finalize_pending", "Accepted input awaits its semantic prefix")
            require_input_mutation_allowed(finalize, run, finalize_owner=(
                self.get_row("run", finalize["value"]["owner_run_id"])
                if finalize is not None and finalize["value"]["state"] == "released" else None
            ))
            current_embedding = self.get_control("policy:embedding_space")
            if (
                current_embedding is None
                or embedding_state_token(ControlRowV1.model_validate(current_embedding))
                != embedding_token
            ):
                raise AMLError(
                    "embedding_space_mismatch", "Embedding state changed before reception"
                )
            ledger_key = f"request_ledger:{ticket.scope_id}:{run['target_id']}"
            if canonical_json(self.get_control(ledger_key)) != prepared.ledger_json:
                raise AMLError("reception_stale", "Reception ledger changed before commit")
            value = prepared.ledger_value
            if binding.request_key in value.recent:
                raise AMLError("operation_conflict", "Replay must use its durable receipt gate")
            generation = self.counter("commit_seq") + 1
            self._check_add_evidence(
                ticket,
                request,
                prepared.parts,
                evidence,
                profile=profile,
                embedding_space=current_embedding["value"]["embedding_space"],
                expected_security=prepared.expected_security,
            )
            validated = prepared.evidence
            if not self.borrowed_rows_unchanged(prepared.read_keys, prepared.token):
                # A mutation named a row the borrowed validation compared
                # against: validate once more here rather than write rows
                # that were checked against stale state.
                validated = tuple(
                    self._validate_rows([RowWrite("evidence", row) for row in evidence], ticket)
                )
            # Checked before the first write below, so a refusal leaves no prewrite.
            receptions = [body for _, body in self.find_logical_candidates(
                run["object_id"], "run_input",
            ) if isinstance(body, SessionAdmissionRunInputV1)]
            if not receptions:
                # A concurrent attempt of this request rejected it and deleted the
                # saved source; the retry receives the same request again.
                raise AMLError("reception_stale", "Received source was discarded before ordering")
            if len(receptions) != 1 or receptions[0].payload_digest != binding.payload_digest:
                raise AMLError("binding_mismatch", "Evidence indexing lost its received source")
            reception = receptions[0]
            stamp = _ReceiptStamp(ingested_at=now, ingested_commit_seq=generation).model_dump()
            source_rows = [
                cast(EvidenceRowV1, item.row).model_copy(update=stamp) for item in validated
            ]
            recent = dict(value.recent)
            if prepared.overflow is not None:
                reference = self.put_artifact(
                    ticket, "request_ledger", prepared.overflow, created_at=now
                )
                value = RequestLedgerValueV1(
                    recent={},
                    ledger_ref=reference,
                    ledger_digest=prepared.overflow.cumulative_digest,
                )
                recent = {}
                if checkpoint is not None:
                    checkpoint("ledger_prewritten")
            recent[binding.request_key] = RequestProgressV1(
                admission_digest=binding.admission_digest
            )
            post_ledger = self.materialize_control(
                ledger_key,
                {
                    **value.model_dump(mode="json"),
                    "recent": {
                        key: progress.model_dump(mode="json") for key, progress in recent.items()
                    },
                },
                0 if ledger is None else ledger.version + 1,
                now,
                ticket.run_id,
            )
            durable = SessionInventoryDurableRunInputV2.model_validate(
                {
                    "receipt_version": 2,
                    "source_inventory": freeze_source_coordinates(
                        (source_coordinate(row) for row in source_rows),
                        message_count=len(request.messages),
                        bind=self.bind_content,
                    ),
                    "run_kind": "session_ingestion",
                    "input_phase": "durable",
                    "business_key": {
                        "session_id": request.session_id,
                        "request_id": request.request_id,
                    },
                    "source_request_locator": {
                        "session_id": request.session_id,
                        "request_id": request.request_id,
                    },
                    "admission_digest": binding.admission_digest,
                    "payload_digest": binding.payload_digest,
                    "request_key": binding.request_key,
                    "visibility_commit_seq": generation,
                    "message_count": len(request.messages),
                    "received_at": reception.received_at,
                    "reception_sequence": reception.reception_sequence,
                    "work_deadline_at": reception.work_deadline_at if completion_required else None,
                    "evidence_set_digest": self.bind_content(
                        "finalize-evidence-row",
                        [row.model_dump(mode="json") for row in source_rows],
                    ),
                }
            )
            reference = self.put_artifact(ticket, "run_input", durable, created_at=now)
            if checkpoint is not None:
                checkpoint("receipt_prewritten")
            # The reception root ends at its ordering commit in both profiles,
            # and the commit writes Evidence, so it advances scope_seq (B.2,
            # A.3.10; store-Q9).
            run.update(
                input_ref=reference,
                status="succeeded",
                stage="settle",
                started_at=now,
                updated_at=now,
                finished_at=now,
            )
            bookkeeping = [
                RowWrite("control", post_ledger),
                RowWrite("run", self._parse_row("run", run)),
            ]
            bookkeeping += self._visibility_counters(ticket, scope_changed=True, updated_at=now)
            plan = self._plan_rows(
                [
                    *(RowWrite("evidence", row) for row in source_rows),
                    *self._validate_rows(bookkeeping, ticket),
                ]
            )
            self._write_plan(plan)
            if profile == "bench":
                resource_session_event(
                    "reception", scope_id=run["scope_id"], run_id=run["object_id"],
                    session_hash=run["target_id"], commit_seq=generation,
                )
            if checkpoint is not None:
                checkpoint("visibility_committed")
            publish(durable, source_rows)
            self.add_admission.ordered(run["scope_id"], run["object_id"])
            if checkpoint is not None:
                checkpoint("root_published")
            return reference

    def validate_storage_owner(self) -> None:
        """Recheck process storage ownership under the already-held Writer mutex.

        This performs no HeavenBase access. Write scopes and final read delivery
        share the same validation and revoke the registered publisher on failure.
        This validation always inspects the filesystem; it never reuses a
        recent borrowed-read probe.

        Raises:
            AMLError: If the calling thread does not own the mutex or the
                supplied process owner rejects the current storage identity.
            OSError: If the process owner cannot inspect the retained paths.
        """
        is_owned = cast(Callable[[], bool], attrgetter("_is_owned")(self.mutex))
        if not is_owned():
            raise AMLError("binding_mismatch", "Storage validation requires the Writer mutex")
        self._probe_storage_owner(force=True)

    def probe_storage_owner(self) -> None:
        """Recheck process storage ownership without Writer exclusion.

        Read-only borrowers call this between storage reads so that an
        unchanged reservation generation still observes a replaced, relinked
        or permission-changed workspace. Probes serialize only with each other
        and with the mutex-held validation on a dedicated leaf lock; they never
        queue behind Writer critical sections. A borrowed-read probe reuses a
        filesystem inspection completed less than ``OWNER_PROBE_INTERVAL_SECONDS``
        earlier by any thread (ADR 0037): the finalize consumers probe between
        every row they borrow, tens of thousands of times per owner, and each
        inspection costs several directory and pinned-file stats. Failure
        revokes the registered publisher exactly as the mutex-held validation
        does and is never reused; the next probe inspects again.

        Raises:
            AMLError: If the process owner rejects the current storage identity.
            OSError: If the process owner cannot inspect the retained paths.
        """
        self._probe_storage_owner(force=False)

    def _probe_storage_owner(self, *, force: bool) -> None:
        # Fixture-constructed Writers may bypass __init__; every thread must
        # still share exactly one leaf lock for owner validation.
        lock = self.__dict__.setdefault("_owner_probe_lock", threading.Lock())
        if not force and self.runtime.owner is not None:
            # Read the stamp before taking the leaf lock: a borrowed probe
            # that a recent inspection already covers must not queue behind
            # the thread performing the next one (the finalize consumers probe
            # between every row they borrow; harness X4 measured 55k probes
            # waiting a mean 0.9 ms on this lock). The stamp is one attribute
            # read; the locked path below re-reads it before inspecting.
            probed_at = self.__dict__.get("_owner_probed_at")
            if probed_at is not None and (
                time.monotonic() - probed_at < OWNER_PROBE_INTERVAL_SECONDS
            ):
                return
        try:
            with lock:
                if self.runtime.owner is not None:
                    probed_at = self.__dict__.get("_owner_probed_at")
                    if not force and probed_at is not None and (
                        time.monotonic() - probed_at < OWNER_PROBE_INTERVAL_SECONDS
                    ):
                        return
                    self.runtime.owner.validate()
                    self.__dict__["_owner_probed_at"] = time.monotonic()
        except BaseException:
            from aml.store.read_model import StoreReadModel

            publisher = StoreReadModel.for_writer(self)
            if publisher is not None:
                publisher.fail_closed(discard_finalize_baselines=True)
            raise

    @property
    def durable_marker(self) -> int:
        """Count the reserved writes that may have reached storage (C.14).

        Every reference-bearing write reserves ``ref_generation`` before its
        attempt, because a committed write can lose its acknowledgement. A plan
        that a constraint rolled back whole (``storage_conflict``) reserved one
        and wrote nothing, so it is subtracted. A caller that fails the read
        model closed on a changed marker therefore keeps it open when the only
        failed attempt wrote nothing; ``ref_generation`` itself never moves back.
        """
        return self.ref_generation - self.__dict__.get("_rolled_back_plans", 0)

    @contextmanager
    def _write_scope(self) -> Iterator[None]:
        """Keep prepared writes inside HB's exclusive lifecycle/cache boundary.

        AML's commit mutex is acquired first. The remaining lock order matches
        the selected HB workspace mutator: workspace, Context, configuration.
        Private RowOp materialization and execution must share this scope.

        An outermost scope that completes renews, before it releases the Writer,
        the due lease of each finalize owner whose execution fence it proved and
        of one other owner holding a live ticket (``owner_lease.renew_due_leases``,
        C.2a): an owner writing back to back starves no live lease. A thread whose
        back-to-back outermost scopes have run ``WRITE_RUN_SECONDS`` first stays out
        ``WRITE_YIELD_SECONDS`` (``_WriteRun``), so no run of them keeps the Writer
        or the workspace lock from a waiter for longer.
        """
        outermost = not self.mutex._is_owned()  # pyright: ignore[reportPrivateUsage]
        with self._write_run(outermost), self.mutex:
            self.validate_storage_owner()
            with self._ws._lock:
                self._ws._need_live()
                context = self._ws.context
                with context._lock:
                    self._ws._need_live()
                    config = context.config
                    with config.scoped(f"{config.base_scope}.{self._ws.id}"):
                        generation = self.durable_marker
                        acting: dict[str, FinalizeLeaseTicket] | None = None
                        if outermost:
                            acting = self.__dict__["_acting_owners"] = {}
                        try:
                            yield
                            if acting is not None:
                                from aml.store.owner_lease import renew_due_leases

                                renew_due_leases(self, acting)
                        except BaseException:
                            if self.durable_marker != generation:
                                from aml.store.read_model import StoreReadModel

                                publisher = StoreReadModel.for_writer(self)
                                if publisher is not None:
                                    publisher.fail_closed()
                            raise
                        finally:
                            if outermost:
                                self.__dict__["_acting_owners"] = None
                                self._clear_artifact_preparation_admissions()

    def _write_run(self, outermost: bool) -> AbstractContextManager[None]:
        """Return the calling thread's run for an outermost scope; a nested scope has none."""
        if not outermost:
            return nullcontext()
        runs = self.__dict__.get("_write_runs")
        if runs is None:
            # Fixture-constructed Writers may bypass __init__.
            runs = self.__dict__.setdefault("_write_runs", _WriteRuns())
        return cast(_WriteRuns, runs).run

    @contextmanager
    def _measure_scope(self) -> Iterator[None]:
        """Materialize private measurement RowOps inside HB's lifecycle boundary.

        A measurement prepares operations that never execute, so it takes the
        HB workspace, Context and configuration scope like a write but not the
        commit mutex: a plan sizing hundreds of frames must not queue behind
        every delivery to learn a row count (plan section 4). Process ownership
        is probed, not validated; the caller re-proves its input under a
        later admission before any of the measured rows is written.
        """
        self.probe_storage_owner()
        # The shared side of HB's workspace lock: a measurement prepares
        # operations against the fixed entity registration and must not stall
        # a committer that already holds the commit mutex, nor block the
        # borrowed readers, as the exclusive side did.
        with self._ws._lock.read():
            self._ws._need_live()
            context = self._ws.context
            with context._lock:
                self._ws._need_live()
                config = context.config
                with config.scoped(f"{config.base_scope}.{self._ws.id}"):
                    yield

    def get_row(self, kind: str, object_id: str) -> dict[str, Any] | None:
        """Read the historical V1 image from its actual physical authorities.

        The row cache retains physical rows. Artifact's pure V1 projection may
        reuse the bounded per-row derived-value memo for exactly equal input
        bytes, and Evidence's while its row is unchanged (``_evidence_image``).
        Items, Candidates and interactions still read their selected
        authorities; removing one cannot leave a cached logical image.
        """
        if kind == "evidence":
            return self._evidence_image(object_id)
        return self._restore_physical_rows(kind, [self._get_physical_row(kind, object_id)])[0]

    def _evidence_image(self, object_id: str) -> dict[str, Any] | None:
        """Memoize the pure V1 projection of one Evidence row while the row is unchanged.

        Every comparison, review and source check reads its Evidence through
        ``get_row``, and each projection re-encodes the row with its embedding:
        each refusal pass of a 976-position reconcile projected the same rows
        thousands of times. The projection reads nothing else
        (``restore_v1_row_image``), so it is reused while no mutation named
        the row since the token read before its physical read (ADR 0036). The
        per-row memo owns byte limits and privacy clearing, a hit takes the
        same liveness check as a memoized physical row, and every returned
        image is detached.
        """
        memo = self.binding_memo()
        key = ("normalized-evidence-image", self.tenant_id, object_id)
        cached = memo.lookup(key)
        if cached is not MISSING:
            self._ws._need_live()
            return None if cached is None else copy_row(cast(dict[str, Any], cached))
        token = self.storage_epoch
        row = self._get_physical_row("evidence", object_id)
        image = None if row is None else self._restore_physical_rows("evidence", [row])[0]
        retained, size = (None, 64) if image is None else sized_copy(image)
        keys = (("evidence", object_id), MEMBERSHIP_ANY)
        if size + 64 * len(keys) <= memo.budget:
            memo.store(key, token, keys, retained, value_bytes=size)
        return image

    def _evidence_images(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Return a restorer of one Evidence statement's rows; it shares ``_evidence_image``'s memo.

        A finalize freeze pages its scope's complete Evidence twice, for its
        head digest and for its source inventory, and each Add's freeze pages
        every earlier row again, restoring each image anew: 1,134 restorations,
        each re-encoding its row and embedding twice, took 1.9 s of Add 38's
        freeze in one long replay. A page row reuses the image memoized under
        its identity while no mutation named the row since the token read here,
        which ``IndexedReader`` does before its statement reads any row (ADR
        0036); a miss restores the decoded row and memoizes it under that token.
        Every returned image is detached.
        """
        from aml.store.normalized_row_codec import restore_v1_row_image

        memo, token = self.binding_memo(), self.storage_epoch

        def restore(row: dict[str, Any]) -> dict[str, Any]:
            key = ("normalized-evidence-image", self.tenant_id, row["object_id"])
            cached = memo.lookup(key)
            if cached is not MISSING and cached is not None:
                return copy_row(cast(dict[str, Any], cached))
            image = restore_v1_row_image("evidence", row, tenant_id=self.tenant_id)
            retained, size = sized_copy(image)
            keys = (("evidence", row["object_id"]), MEMBERSHIP_ANY)
            if size + 64 * len(keys) <= memo.budget:
                memo.store(key, token, keys, retained, value_bytes=size)
            return image

        return restore

    def get_rows(self, kind: str, object_ids: Sequence[str]) -> list[dict[str, Any] | None]:
        """Read aligned historical images, batching their selected dependencies."""
        return self._restore_physical_rows(kind, self._get_physical_rows(kind, object_ids))

    def _restore_physical_rows(
        self, kind: str, rows: Sequence[dict[str, Any] | None], *,
        post: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> list[dict[str, Any] | None]:
        """Restore selected V1 images without caching derived authority values.

        A prepared main plan supplies its exact prospective rows through post.
        Other references are read by kind and identity. Recall parents are read
        physically first, so a corrupt interaction cycle cannot recurse.
        """
        from aml.store.normalization import LOGICAL_IMAGE_KINDS
        from aml.store.normalized_row_codec import restore_v1_row_image

        if kind not in LOGICAL_IMAGE_KINDS:
            return list(rows)
        dependency = {"recall": ("recall", "parent_recall_id")}.get(kind)
        authorities: dict[str, dict[str, Any] | None] = {}
        if dependency is not None:
            target_kind, field = dependency
            identifiers: list[str] = []
            for row in rows:
                if row is None or (kind == "recall" and row["record_class"] != "interaction"):
                    continue
                identifier = row[field]
                if not isinstance(identifier, str):
                    raise AMLError("authority_unavailable", "Physical row lost its authority")
                if identifier not in identifiers:
                    identifiers.append(identifier)
            absent = [value for value in identifiers
                      if post is None or (target_kind, value) not in post]
            authorities.update(zip(
                absent, self._get_physical_rows(target_kind, absent), strict=True,
            ))
            if post is not None:
                authorities.update({value: post[target_kind, value] for value in identifiers
                                    if (target_kind, value) in post})
            for identifier, authority in tuple(authorities.items()):
                if authority is None:
                    raise AMLError("authority_unavailable", "Physical row authority is absent")
                if target_kind == "recall" and authority["record_class"] == "interaction":
                    raise AMLError("authority_unavailable", "Interaction parent is not a delivery")
                authorities[identifier] = self._restore_physical_rows(
                    target_kind, [authority], post=post,
                )[0]
        restored: list[dict[str, Any] | None] = []
        for row in rows:
            if row is None:
                restored.append(None)
                continue
            if kind == "artifact" and post is None:
                restored.append(self._restore_artifact_image(row))
                continue
            authority = None if dependency is None else authorities.get(row[dependency[1]])
            restored.append(restore_v1_row_image(
                kind, row, tenant_id=self.tenant_id,
                parent_recall=(self._parse_row("recall", authority)
                               if kind == "recall" and row["record_class"] == "interaction"
                               else None),
            ))
        return restored

    def _restore_artifact_image(self, row: dict[str, Any]) -> dict[str, Any]:
        """Memoize only the pure projection of an exact normalized Artifact row.

        Body text is already an immutable string; comparing it directly avoids
        encoding its potentially large JSON contents again. The small envelope
        retains canonical type distinctions (including bool versus int). The
        existing per-row memo owns byte limits, mutation retirement and privacy
        clearing. No prospective row, selected reference or live authority is
        memoized, and every returned image is detached.
        """
        from aml.store.normalized_row_codec import restore_v1_row_image

        identity, body = row.get("object_id"), row.get("body")
        if type(identity) is not str or type(body) is not str:
            return restore_v1_row_image("artifact", row, tenant_id=self.tenant_id)
        try:
            header = canonical_json({key: value for key, value in row.items() if key != "body"})
        except (TypeError, ValueError):
            return restore_v1_row_image("artifact", row, tenant_id=self.tenant_id)
        memo = self.binding_memo()
        key = ("normalized-artifact-image", self.tenant_id, identity)
        token = self.storage_epoch
        cached = memo.lookup(key)
        if cached is not MISSING and cached["header"] == header and cached["body"] == body:
            return copy_row(cached["image"])
        restored = restore_v1_row_image("artifact", row, tenant_id=self.tenant_id)
        retained, size = sized_copy({"header": header, "body": body, "image": restored})
        keys = (("artifact", identity), MEMBERSHIP_ANY)
        if size + 64 * len(keys) <= memo.budget:
            memo.store(key, token, keys, retained, value_bytes=size)
        return restored

    def indexed_reader(self) -> IndexedReader:
        """Return a page reader of this workspace whose Artifact and Evidence images reuse memos.

        A page restores each Artifact row it returns, and the owner reservation,
        judge-output and subject-child walks page the same immutable rows on
        every inspection: a 976-position finalize projected them again in each
        refusal pass and in each post-head inspection. This reader's Artifact
        rows pass through ``_restore_artifact_image``, which reuses an image
        only for exactly equal input values, and its Evidence rows through
        ``_evidence_images``, which reuses an image while its row is unchanged;
        every other kind uses the codec.
        """
        from aml.store.indexed import IndexedReader

        return IndexedReader(
            self._ws, self.entities, tenant_id=self.tenant_id,
            artifact_image=self._restore_artifact_image, evidence_image=self._evidence_images,
        )

    def _get_physical_row(self, kind: str, object_id: str) -> dict[str, Any] | None:
        """Read one explicit Entity row, without trusting a Catalog projection.

        Rows are memoized between this owner's raw storage mutations (ADR
        0036). Every return is the caller's private copy, and a retired
        workspace refuses a memoized row exactly as it refuses storage.
        """
        entity = self.entities.get(kind)
        if entity is None:
            raise AMLError("unsupported_variant", "Unknown physical Entity kind")
        cache = self._rows_cache()
        key = (kind, object_id)
        token, hit = cache.lookup(key)
        if hit is not MISSING:
            # A hit touches no backend, so it takes only the liveness check: a
            # retired Context refuses it as storage would, and it never waits
            # behind HB's exclusive writer side the way a storage read does.
            self._ws._need_live()
            return None if hit is None else copy_row(hit)
        row = self._read_row(entity, object_id)
        if row is None:
            cache.store(token, key, None)
            return None
        cache.store(token, key, *sized_copy(row))
        return row

    def _get_physical_rows(
        self, kind: str, object_ids: Sequence[str],
    ) -> list[dict[str, Any] | None]:
        """Read several rows of one kind: memoized hits, then one storage read for the rest.

        The misses are read together through HeavenBase's batched entity-first
        read, one backend-lock acquisition and one IN-list statement for the
        whole batch instead of one per row, and memoized under the admission
        token each lookup took, exactly as ``get_row`` memoizes a single row.
        Under the normal-load registration the artifact collision checks read
        4,800 rows one by one and waited 192 s for the backend lock behind the
        finalize workers (harness X17b, 2026-09-11). Returned rows are private
        copies aligned with ``object_ids``; an absent row is ``None``.
        """
        entity = self.entities.get(kind)
        if entity is None:
            raise AMLError("unsupported_variant", "Unknown physical Entity kind")
        cache = self._rows_cache()
        results: list[dict[str, Any] | None] = [None] * len(object_ids)
        misses: list[tuple[int, int]] = []
        for index, object_id in enumerate(object_ids):
            token, hit = cache.lookup((kind, object_id))
            if hit is MISSING:
                misses.append((index, token))
            elif hit is not None:
                results[index] = copy_row(hit)
        if not misses:
            if object_ids:
                self._ws._need_live()
            return results
        direct = cast(
            Callable[[Any, list[str]], list[dict[str, Any] | None]] | None,
            getattr(self._ws, "read_rows", None),
        )
        missing_ids = [object_ids[index] for index, _ in misses]
        if direct is not None:
            rows = direct(entity, missing_ids)
        else:
            rows = [self._read_row(entity, object_id) for object_id in missing_ids]
        for (index, token), row in zip(misses, rows, strict=True):
            key = (kind, object_ids[index])
            if row is None:
                cache.store(token, key, None)
            else:
                cache.store(token, key, *sized_copy(row))
                results[index] = row
        return results

    def _read_row(self, entity: Any, object_id: str) -> dict[str, Any] | None:
        """Read one row of a known Entity from the sole backend, decoded by HB's codec.

        The workspace facade's ``get`` first resolves the identity through
        HB's Catalog, one extra statement per read whose only purpose is to
        find the Entity; this Writer already names it. The read holds the
        workspace read lock and the liveness check exactly as ``iter_rows``
        does, and decodes with the same storage codec.
        """
        direct = cast(
            Callable[[Any, list[str]], list[dict[str, Any] | None]] | None,
            getattr(self._ws, "read_rows", None),
        )
        if direct is not None:
            # HeavenBase's entity-first read: decoded by the workspace codec under
            # its own read lock and liveness check, without a Catalog lookup.
            return direct(entity, [object_id])[0]
        schema = getattr(entity, "schema", None)
        reader = getattr(getattr(self, "_backend", None), "get_many", None)
        if schema is None or reader is None:
            facade: dict[str, Any] | None = self._ws.get(object_id, entity=entity)
            return facade
        physical = schema()
        with self._ws._lock.read():
            self._ws._need_live()
            raw = reader(physical.entity_id, [object_id])[0]
            if raw is None:
                return None
            decoded: dict[str, Any] = self._ws._row_codec.decode_schema_row(physical, raw)
            return decoded

    def borrowed_refresh(
        self, checked_generation: int | None, refresh: Callable[[], None], *,
        keys: Iterable[tuple[str, str]] | None = None,
    ) -> int:
        """Recheck a borrowed read's fixed rows only when a Writer critical section ran.

        The sole Writer reserves ``ref_generation`` before every reference-bearing
        main write or CAS attempt. There are two exceptions. A bound job's
        traffic watermark has no references and no semantic borrower; its reads
        always use the actual job row. A heartbeat-only finalize lease renewal
        only moves the same owner's expiry later; a borrower that needs the
        lease rereads it, checking its owner and epoch, whenever another write
        runs. Raw row-ledger retirement remains active for both.
        An unchanged generation since ``checked_generation`` therefore proves
        the borrower's provenance rows cannot have changed. That path probes
        process ownership (ADR 0037)
        and takes no Writer exclusion. Any other generation probes ownership
        outside the mutex, then runs ``refresh`` so it rereads and compares
        its rows. Without ``keys`` the reread must fall in a window during
        which no critical section began or ended, proven by the mutex
        sequence and the reservation generation. With ``keys`` naming every
        row the reread depends on, it runs through active critical sections
        as well and is accepted when no in-process mutation named one of the
        keys after the admission token read before it (ADR 0036); a verdict
        ``refresh`` reaches on rows a mutation was replacing is reread, not
        raised. Either proof is retried a bounded number of times before the
        reread runs under the mutex, and a caller that owns the mutex rereads
        under it directly. Writes and deliveries keep their unconditional
        mutex-held validation.

        Args:
            checked_generation: The reservation generation the caller last
                checked, or None before its first refresh.
            refresh: Reread the fixed rows through ``get_row`` and raise when
                they no longer satisfy the borrow.
            keys: The ``(kind, object_id)`` rows ``refresh`` depends on, when
                the borrower can name all of them.

        Returns:
            The reservation generation the caller has now checked: every
            critical section that can still change the fixed rows reserves
            a later one.

        Raises:
            AMLError: If the process owner rejects the storage identity, or
                ``refresh`` rejects rows that no mutation was replacing.
            OSError: If the process owner cannot inspect the retained paths.
        """
        if checked_generation == self.ref_generation:
            self.probe_storage_owner()
            return self.ref_generation
        self.probe_storage_owner()
        mutex = self.mutex
        owned = mutex._is_owned()  # pyright: ignore[reportPrivateUsage]
        if keys is not None and not owned:
            checked = self._refresh_proven_by_keys(tuple(keys), refresh)
            if checked is not None:
                return checked
        elif not owned:
            # A borrower must not queue behind the commit mutex (plan §4): read
            # the fixed rows in a window during which no critical section began
            # or ended, proven by the mutex sequence and the reservation
            # generation, and retry a bounded number of times when one did.
            for _ in range(BORROWED_REFRESH_ATTEMPTS):
                sequence = mutex.sequence
                if sequence % 2:
                    mutex.wait_for_progress(
                        mutex.progress_generation,
                        deadline=time.monotonic() + BORROWED_REFRESH_WAIT_SECONDS,
                    )
                    continue
                generation = self.ref_generation
                refresh()
                if mutex.sequence == sequence and self.ref_generation == generation:
                    return generation
        with mutex:
            refresh()
            return self.ref_generation

    def _refresh_proven_by_keys(
        self, keys: tuple[tuple[str, str], ...], refresh: Callable[[], None],
    ) -> int | None:
        """Reread named rows through active critical sections, proven per key.

        Each attempt reads the reservation generation, the mutex sequence and
        the storage epoch, runs ``refresh`` without waiting for the mutex, and
        accepts the reread when none of ``keys`` was retired after that epoch
        (``borrowed_rows_unchanged``): the memoized rows it read are then the
        rows' current content, whatever unrelated writes ran meanwhile. A
        ``refresh`` verdict is kept only on the same proof; one reached on
        rows a mutation was replacing is no verdict and is reread. A torn
        attempt is retried at once, up to ``BORROWED_REFRESH_ATTEMPTS`` times,
        without waiting for the Writer mutex: the proof is per key and holds
        through active critical sections.

        Args:
            keys: Every ``(kind, object_id)`` row ``refresh`` reads.
            refresh: Reread the fixed rows and raise an ``AMLError`` verdict
                when they no longer satisfy the borrow.

        Returns:
            The reservation generation the caller has now checked, or None
            when every attempt was torn and the caller must reread under the
            mutex. A section active at the read may already have reserved the
            current generation and may still write a fixed row, so the value
            is then the generation before it: an unchanged-generation
            shortcut never skips the reread that section can still require.

        Raises:
            AMLError: If ``refresh`` rejects rows that no mutation was replacing.
        """
        mutex = self.mutex
        for _ in range(BORROWED_REFRESH_ATTEMPTS):
            generation = self.ref_generation
            sequence = mutex.sequence
            token = self.storage_epoch
            try:
                refresh()
            except AMLError:
                if self.borrowed_rows_unchanged(keys, token):
                    raise
            else:
                if self.borrowed_rows_unchanged(keys, token):
                    return generation - 1 if sequence % 2 else generation
            # A torn attempt means a mutation named one of the keys while the
            # rows were being read; the next attempt reads the rows that
            # mutation left. Whether the Writer mutex is busy is irrelevant to
            # a keyed proof (ADR 0036): waiting for a critical section to end
            # exhausted every attempt under load, with the commit mutex busy
            # nine tenths of the time, and sent nearly every refresh to the
            # mutex-held reread (harness runs X4/X5, 2026-09-11: 88% of check()
            # wall time waiting there), so no attempt waits.
        return None

    def borrowed_rows_unchanged(self, keys: Iterable[tuple[str, str]], token: int) -> bool:
        """Prove that none of ``keys`` was mutated in this process after ``token``.

        A borrower that reads a fixed set of rows records ``storage_epoch``
        before the read; while this returns ``True`` for that token those rows
        cannot have changed, whatever unrelated writes ran meanwhile, because
        every in-process row mutation names the rows it replaces or deletes
        (ADR 0036). The check touches no storage and takes no Writer
        exclusion; it does not replace process-ownership probes (ADR 0037).
        Active read records receive ``keys``: whatever this proof decides
        depends on them.
        """
        keys = tuple(keys)
        note_reads(keys)
        return not self._rows_cache().changed_since(keys, token)

    def watch_borrowed_rows(self, keys: Iterable[tuple[str, str]], token: int) -> BorrowWatch:
        """Open a repeatable ``borrowed_rows_unchanged(keys, token)`` proof.

        A borrower that rechecks one large key set between every row it reads
        opens the watch once (O(keys)) and asks ``borrowed_rows_watched``
        afterwards (O(1)); the answers are those ``borrowed_rows_unchanged``
        would give, except that a ledger prune never turns a still-unchanged
        borrow into a change.
        """
        return self._rows_cache().watch_borrow(keys, token)

    def borrowed_rows_watched(self, watch: BorrowWatch) -> bool:
        """Prove that none of ``watch``'s keys was mutated in this process after its token.

        Active read records receive the borrowed keys, as from
        ``borrowed_rows_unchanged``.
        """
        watch._note()  # pyright: ignore[reportPrivateUsage]
        return self._rows_cache().borrow_unchanged(watch)

    @property
    def storage_epoch(self) -> int:
        """Return a token that advances on every in-process HeavenBase row mutation.

        Two equal values prove that no row mutation reached this workspace's
        row writer or backends between the two reads (ADR 0036). Borrowers use
        it to reuse a page they already validated; it is not durable and it is
        not the Writer's reservation generation.
        """
        return self._rows_cache().token

    def _rows_cache(self) -> RowCache:
        # Fixture-constructed Writers may bypass __init__; every thread must
        # still share exactly one cache whose token orders their admissions.
        cache = self.__dict__.get("_row_cache")
        if cache is None:
            cache = self.__dict__.setdefault("_row_cache", RowCache())
        return cast(RowCache, cache)

    def scan_memo(self) -> ScanMemo:
        """Return the process-local memo of complete scans proven by the mutation ledger.

        A consumer that pages every row satisfying a predicate (the finalize
        head digests, a reextract slice family) records the admission token
        before its scan, the rows it read and the membership keys of the sets
        it enumerated, and reuses the result while ``borrowed_rows_unchanged``
        would prove all of them unchanged (ADR 0041). The memo is bounded,
        never durable, never shared across processes, and grants no lease,
        guard or write authority: every consumer keeps its own borrow checks.
        """
        memo = self.__dict__.get("_scan_memo")
        if memo is None:
            memo = self.__dict__.setdefault("_scan_memo", ScanMemo(self._rows_cache()))
        return cast(ScanMemo, memo)

    def binding_memo(self) -> ScanMemo:
        """Return the process-local memo of per-row derived values.

        A row's canonical bytes and its row-state digest are proven by the same
        ledger as any scan, but there is one entry per row rather than per
        scan: a scope's rows are tens of megabytes of them, and sharing one
        table with the scans evicted the ordered members and head digests that
        the per-row values exist to serve. This table has its own bound.
        """
        memo = self.__dict__.get("_binding_memo")
        if memo is None:
            memo = self.__dict__.setdefault(
                "_binding_memo", ScanMemo(self._rows_cache(), budget=ROW_BINDING_BYTES),
            )
        return cast(ScanMemo, memo)

    def scope_evidence(self) -> ScopeEvidenceIndex:
        """Return the process-local per-scope Evidence header index (``aml.store.scope_evidence``).

        It is kept current by this Writer's retirement ledger: a scope is
        scanned once, after which a snapshot re-reads only the Evidence rows
        retired since its token. It grants no lease, guard or write authority.
        """
        from aml.store.scope_evidence import ScopeEvidenceIndex

        index = self.__dict__.get("_scope_evidence")
        if index is None:
            index = self.__dict__.setdefault(
                "_scope_evidence", ScopeEvidenceIndex(self._rows_cache()),
            )
        return cast(ScopeEvidenceIndex, index)

    def _storage_mutated(self) -> None:
        """Retire memoized rows: storage may now differ from every earlier read."""
        self._rows_cache().retire()

    def _hook_storage_mutators(self) -> None:
        """Retire memoized rows around every in-process HeavenBase row mutation.

        The workspace row writer's ``write_ops`` and each registered backend's
        ``upsert``, ``delete``, ``compare_upsert`` and ``drop`` are decorated
        on their instances, so this Writer's own writes, HeavenBase's Catalog
        maintenance and a verification fault injected through HeavenBase's
        facade or a backend all retire the cache before and after they run.
        Only this workspace's objects are touched; the decoration holds a weak
        reference and is inert once the Writer is gone.
        """
        row_writer = self._ws._writer
        row_writer.write_ops = self._hook_retiring(row_writer.write_ops, _keys_of_write_ops)
        self.__dict__["_mutators_hooked"] = True
        for _, backend in self._ws.backends.items():
            for name, keys_of in (
                ("upsert", _keys_of_upsert), ("delete", _keys_of_delete),
                ("compare_upsert", _keys_of_compare_upsert), ("drop", None),
            ):
                original = getattr(backend, name, None)
                if callable(original):
                    setattr(backend, name, self._hook_retiring(
                        original, keys_of, atomic=name == "upsert",
                    ))

    def _hook_retiring(
        self, original: Callable[..., Any],
        keys_of: Callable[[tuple[Any, ...], dict[str, Any]], Iterable[tuple[str, Any]] | None]
        | None = None,
        *, atomic: bool = False,
    ) -> Callable[..., Any]:
        """Decorate one storage mutator so it retires this Writer's memoized rows.

        ``keys_of`` names the ``(entity_id, object_id)`` pairs a call replaces
        or deletes; only those memoized rows are dropped. A mutator without a
        key extractor, or a call whose keys cannot be named, retires every row.
        An ``atomic`` mutator runs as one backend transaction (a backend's
        ``upsert``): a constraint that refused it rolled it back whole, so it
        cannot have left main rows without their Catalog rows.
        """
        ref = weak_ref(self)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            keys = None if keys_of is None else keys_of(args, kwargs)
            ops = _row_ops_of(keys_of, args, kwargs)
            writer = ref()
            if writer is not None:
                if writer.__dict__.get("_catalog_sealed", False):
                    # A mutation after a sealed Catalog proof voids the proof.
                    writer.__dict__["_catalog_unproven"] = True
                writer._ledger_mutation()
                writer._rows_mutated(keys, writer._memberships_of(ops))
                writer._admission_mutation(ops)
            try:
                return original(*args, **kwargs)
            except BaseException as error:
                writer = ref()
                if writer is not None and not (atomic and constraint_violation(error)):
                    # A mutation that did not finish may have left main rows
                    # without their Catalog rows: only a complete audit proves it.
                    writer.__dict__["_catalog_unproven"] = True
                    writer._ledger_unproven()
                raise
            finally:
                writer = ref()
                if writer is not None:
                    writer._rows_mutated(keys, writer._memberships_of(ops))
                    writer._admission_mutation(ops)

        return wrapped

    @contextmanager
    def _catalog_steps(self) -> Iterator[None]:
        """Leave the Catalog unproven when a main write's Catalog step does not follow it.

        A ``storage_conflict`` wrote no main row (``_write_plan``), so its Catalog
        step had nothing to follow and the Catalog stays as proven as it was.
        """
        try:
            yield
        except BaseException as error:
            if not (isinstance(error, AMLError) and error.code == "storage_conflict"):
                self.__dict__["_catalog_unproven"] = True
                self._ledger_unproven()
            raise

    def _ledger_mutation(self) -> None:
        """Record a storage mutation in flight in this lifetime's Catalog ledger.

        A mutation outside the Writer mutex leaves the Catalog unproven: neither a
        sealed close nor a section's end can prove that it finished. After that,
        or after any failed mutation, the ledger stays unproven.
        """
        mutex = cast(WriterMutex | None, self.__dict__.get("mutex"))
        if mutex is not None and not mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            self.__dict__["_catalog_unproven"] = True
        ledger = cast(CatalogLedger | None, self.__dict__.get("_catalog_ledger"))
        if ledger is not None:
            unproven = self.__dict__.get("_catalog_unproven", False)
            ledger.mark(UNPROVEN if unproven else DIRTY)

    def _ledger_unproven(self) -> None:
        """Leave this lifetime's Catalog ledger unproven after a failed mutation."""
        ledger = cast(CatalogLedger | None, self.__dict__.get("_catalog_ledger"))
        if ledger is not None:
            with suppress(AMLError):
                # A record that cannot be voided stays dirty, which proves nothing.
                ledger.mark(UNPROVEN)

    def _ledger_section_end(self, suspended: bool) -> None:
        """Record at a critical section's end that its storage mutations finished.

        Runs before the outermost release of the Writer mutex (``WriterMutex
        .on_outermost_release``): every mutation of the section has returned, each
        with its Catalog step, or the Catalog is unproven. A section suspended by a
        condition wait proves nothing.
        """
        ledger = cast(CatalogLedger | None, self.__dict__.get("_catalog_ledger"))
        if ledger is None or ledger.state != DIRTY:
            return
        unproven = suspended or self.__dict__.get("_catalog_unproven", False)
        with suppress(AMLError):
            ledger.mark(UNPROVEN if unproven else CLEAN)

    def _rows_mutated(
        self, keys: Iterable[tuple[str, Any]] | None,
        memberships: Iterable[tuple[str, str]] = (),
    ) -> None:
        """Retire the memoized rows a storage mutation names, or all of them.

        ``memberships`` are the ledger keys of the sets the mutation's
        post-images may have joined (``_memberships_of``); they are retired in
        the same ledger write as the rows, so a memoized scan (ADR 0041) is
        refused on the same token order as a memoized row.

        While a heartbeat-only lease CAS watches (``_journal_heartbeat``),
        every call is noted with its thread and named rows.
        """
        watch: list[tuple[int, tuple[tuple[str, Any], ...] | None]] | None = (
            self.__dict__.get("_heartbeat_watch")
        )
        if watch is not None:
            keys = None if keys is None else tuple(keys)
            watch.append((threading.get_ident(), keys))
        if keys is None:
            self._storage_mutated()
            return
        kinds = self._kinds_by_entity_id()
        if kinds is None:
            self._storage_mutated()
            return
        cache_keys: list[tuple[str, str]] = []
        for entity_id, object_id in keys:
            kind = kinds.get(entity_id)
            if kind is None:
                continue  # Not an Entity this Writer memoizes (Catalog maintenance).
            if not isinstance(object_id, str):
                self._storage_mutated()
                return
            cache_keys.append((kind, object_id))
        cache_keys.extend(memberships)
        self._rows_cache().invalidate(cache_keys)

    def _memberships_of(
        self, ops: list[tuple[str, str, Any, Any]] | None,
    ) -> tuple[tuple[str, str], ...]:
        """Name the memberships a mutation's upserted post-images may join.

        A memoized scan over "every Evidence row of a scope", "every Control
        under a key namespace" or "every Run of an idempotency family" proves
        the rows it read unchanged per key but cannot name a row that did not
        exist when it ran (ADR 0041). Each upsert therefore retires the
        membership key its post-image names: the Evidence or Candidate row's
        exact scope, the Control key without its last segment
        (``cursor:{scope_id}`` for a session cursor), or the Run's
        ``idempotency_key``. A post-image that cannot name its set retires
        that family's wildcard, which every proof over a set of that family
        includes; a mutation whose rows cannot be inspected at all retires
        ``MEMBERSHIP_ANY``, which every membership proof includes. Deletes
        retire only the deleted rows, which every scan that read them names.
        """
        kinds = None if ops is None else self._kinds_by_entity_id()
        if ops is None or kinds is None:
            return (MEMBERSHIP_ANY,)
        retired: list[tuple[str, str]] = []
        for entity_id, operation, values, object_id in ops:
            kind = kinds.get(entity_id)
            mutex = getattr(self, "mutex", None)
            run_excluded = kind == "run" and mutex is not None and mutex._is_owned()
            if kind not in {
                "evidence", "candidate", "item", "revision", "control", "run", "artifact",
            } or operation != "upsert":
                continue
            if isinstance(values, dict):
                if kind == "run" and any(name not in values for name in (
                    "idempotency_key", "parent_run_id", "scope_kind", "scope_id",
                )):
                    if run_excluded and isinstance(object_id, str):
                        # Existing Run progress patches cannot join an unknown
                        # parent/family/scope while the sole Writer is excluded.
                        # Reuse its actual preimage, retaining explicit changes
                        # in the patch. Missing rows and raw unlocked writes
                        # still retire every unproved membership below.
                        previous = self.get_row("run", object_id)
                        if previous is not None:
                            values = {**previous, **cast(dict[str, Any], values)}
                members = _membership_of(kind, cast(dict[str, Any], values))
            else:
                members = (None, None, None) if kind == "run" else (None,)
            for index, member in enumerate(members):
                if run_excluded and index == 1 and isinstance(values, dict) and (
                    cast(dict[str, Any], values).get("parent_run_id", MISSING) is None
                ):
                    # A proved top-level Run belongs to no child census. The
                    # empty-parent key would collide with the family wildcard.
                    # Its fixed Run, idempotency family and scope still retire;
                    # unknown or unlocked writes keep the wildcard fallback.
                    continue
                if member is not None:
                    retired.append(member)
                    continue
                # Run census readers independently fence family, parent and scope.
                retired.append(
                    membership_family(("run", "run-child", "run-scope")[index])
                    if kind == "run"
                    else membership_family(kind)
                )
        return tuple(retired)

    def _admission_touched(self) -> None:
        """Retire the memoized admission reads of every scope."""
        self.__dict__["_admission_generation"] = (
            self.__dict__.get("_admission_generation", 0) + 1
        )

    def _admission_mutation(self, ops: list[tuple[str, str, Any, Any]] | None) -> None:
        """Retire the memoized admission reads when a mutation could change them.

        The two admission reads select queued or running affected-set retention
        scans among Run rows and the erasure/maintenance fence Controls of a
        scope. Only an upsert of such a Run or Control, any delete of a Run or
        Control, or a mutation whose rows cannot be inspected (``ops`` is
        None, from ``_row_ops_of``) changes their answer; every other write,
        including the finalize heartbeats and Run progress the bench load
        produces continuously, leaves it intact.
        """
        kinds = None if ops is None else self._kinds_by_entity_id()
        if ops is None or kinds is None:
            self._admission_touched()
            return
        for entity_id, operation, values, object_id in ops:
            kind = kinds.get(entity_id)
            if kind not in {"run", "control"}:
                continue
            if operation != "upsert" or not isinstance(values, dict):
                self._admission_touched()
                return
            if kind == "run":
                fields = ("run_kind", "commit_mode", "target_kind")
                described: tuple[Any, Any, Any] | None
                post = cast(dict[str, Any], values)
                if all(name in post for name in fields):
                    described = (post["run_kind"], post["commit_mode"], post["target_kind"])
                else:
                    # A partial post-image cannot create a Run, so its kind,
                    # mode and target are those of the existing row; the bench
                    # load's Run progress fragments were retiring every memo.
                    existing = None if not isinstance(object_id, str) else (
                        self.get_row("run", object_id)
                    )
                    described = None if existing is None else (
                        existing.get("run_kind"), existing.get("commit_mode"),
                        existing.get("target_kind"),
                    )
                if described is None or described == (
                    "retention", "destructive_scan", "affected_set",
                ):
                    self._admission_touched()
                    return
            else:
                key = cast(dict[str, Any], values).get("key")
                if not isinstance(key, str) or key.startswith(("erasure:", "maintenance:")):
                    self._admission_touched()
                    return

    def _admission_reads(
        self, scope: str,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        """Return the pending privacy owners and fence Controls of one scope.

        Under the Writer mutex the rows are read through the indexed reader
        and memoized per scope until a mutation that could change them retires
        the memo (``_admission_mutation``). The sole Writer performs every
        in-process row mutation through the hooked mutators, and process
        ownership is validated before each critical section, so an unretired
        memo is exactly what the two selects would return (ADR 0039). An
        admission prepared without the mutex (the Add commit's off-mutex
        validation) reads the selects directly and touches no memo.
        """
        from aml.store.indexed import IndexedReader

        is_owned = cast(Callable[[], bool], attrgetter("_is_owned")(self.mutex))
        if not is_owned():
            reader = IndexedReader(
                self.runtime.workspace,
                self.entities,
                tenant_id=self.tenant_id,
            )
            return tuple(reader.pending_privacy(scope, limit=2)), tuple(reader.scope_fences(scope))
        return self._memoized_admission_reads(scope)

    def admission_reads_hint(
        self, scope: str,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        """Return the scope's admission reads for a gate that is re-proved later.

        The Add path's scope gate runs twice per Add without the mutex, so its
        four selects were among the largest read sites under plan section 9
        normal load (normal-06g: 1,250 empty selects in seven minutes). An
        unretired memo is what the selects would return; a mutation that
        lands after this hint is caught by the Run admission, which re-reads
        under the mutex before any write. The hint never stores a memo.
        """
        generation = self.__dict__.get("_admission_generation", 0)
        memo = cast(
            dict[str, tuple[int, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]],
            self.__dict__.get("_admission_memo") or {},
        )
        entry = memo.get(scope)
        if entry is not None and entry[0] == generation:
            return entry[1], entry[2]
        from aml.store.indexed import IndexedReader

        reader = IndexedReader(
            self.runtime.workspace,
            self.entities,
            tenant_id=self.tenant_id,
        )
        return tuple(reader.pending_privacy(scope, limit=2)), tuple(reader.scope_fences(scope))

    def _memoized_admission_reads(
        self, scope: str,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        from aml.store.indexed import IndexedReader

        generation = self.__dict__.setdefault("_admission_generation", 0)
        memo = cast(
            dict[str, tuple[int, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]],
            self.__dict__.setdefault("_admission_memo", {}),
        )
        entry = memo.get(scope)
        if entry is not None and entry[0] == generation:
            return entry[1], entry[2]
        reader = IndexedReader(
            self.runtime.workspace,
            self.entities,
            tenant_id=self.tenant_id,
        )
        pending = tuple(reader.pending_privacy(scope, limit=2))
        controls = tuple(reader.scope_fences(scope))
        if len(memo) >= _ADMISSION_MEMO_LIMIT:
            memo.pop(next(iter(memo)))
        memo[scope] = (generation, pending, controls)
        return pending, controls

    def _kinds_by_entity_id(self) -> dict[str, str] | None:
        """Map each memoized Entity's physical identifier back to its kind.

        Returns None when an Entity cannot name its physical identifier; the
        caller then retires every memoized row rather than guessing.
        """
        cached = self.__dict__.get("_kinds_by_entity")
        if cached is None:
            try:
                cached = {
                    entity.schema().entity_id: kind for kind, entity in self.entities.items()
                }
            except AttributeError:
                return None
            self.__dict__["_kinds_by_entity"] = cached
        return cast(dict[str, str], cached)

    def _mutate_storage(self, mutation: Callable[[], _T], *, hooked: bool = True) -> _T:
        """Run one raw storage mutation with memoized rows retired around it.

        A hooked mutation reaches the workspace row writer or a backend
        method decorated by ``_hook_storage_mutators``, which retires exactly
        the rows it names before and after the call; nothing further is
        retired here. A mutation that does not pass through those methods, or
        one issued before the hooks were installed, retires every row before
        the call and again afterwards, even on failure, so a row another
        thread admitted while the mutation was in flight is discarded.
        """
        if hooked and self.__dict__.get("_mutators_hooked", False):
            return mutation()
        self._storage_mutated()
        try:
            return mutation()
        finally:
            self._storage_mutated()

    def iter_rows(self, kind: str) -> Iterator[dict[str, Any]]:
        """Yield restored historical images from bounded physical scan pages."""
        for page in itertools.batched(self._iter_physical_rows(kind), 128):
            restored = self._restore_physical_rows(kind, page)
            for row in restored:
                assert row is not None
                yield row

    def _iter_physical_rows(self, kind: str) -> Iterator[dict[str, Any]]:
        """Decode real bounded Backend.iter_scan rows using HB's storage codec."""
        entity = self.entities[kind]
        decoded_pages = cast(
            Callable[..., Iterator[dict[str, Any]]] | None, getattr(self._ws, "iter_rows", None),
        )
        if decoded_pages is not None:
            # HeavenBase's decoded keyset iteration holds the read lock per page
            # and releases it before yielding, as this method always did.
            yield from decoded_pages(entity, batch_size=128)
            return
        rows = iter(self._backend.iter_scan(entity.schema().entity_id, batch_size=128))
        while True:
            # Release shared ownership before yielding: callers may consume a
            # bounded batch and enter the Writer. This scan is not a snapshot.
            with self._ws._lock.read():
                self._ws._need_live()
                raw = next(rows, None)
                if raw is None:
                    return
                decoded = cast(
                    dict[str, Any], self._ws._row_codec.decode_schema_row(entity.schema(), raw)
                )
            yield decoded

    def get_control(self, key: str) -> dict[str, Any] | None:
        """Read a Control by its canonical namespace identity."""
        return self.get_row("control", control_id(key))

    def counter(self, key: str) -> int:
        """Read a required nonnegative Control counter; absence is not zero."""
        row = self.get_control(key)
        if row is None:
            raise AMLError("service_unhealthy", "Required Control counter is absent")
        value = parse_control_value(key, row["value"])
        sequence = getattr(value, "seq", None)
        if type(sequence) is not int or sequence < 0:
            raise AMLError("parity_mismatch", "Control counter is invalid")
        return sequence

    def initialize_base(
        self,
        workspace_id: str,
        db_uuid: str,
        schema_fingerprint: IntegrityDigest,
        now: int,
    ) -> int:
        """Initialize the plan's pre-Run infrastructure through this same Writer.

        Existing workspace binding is immutable. A reopened owner advances its
        Writer epoch; counter state is never reset. Provider bootstrap is separate.
        """
        with self._write_scope():
            binding = {
                "schema_version": 1,
                "db_uuid": db_uuid,
                "workspace_id": workspace_id,
                "tenant_digest": hashlib.sha256(self.tenant_id.encode()).hexdigest(),
                "embedding_dimension": self.dimension,
                "schema_fingerprint": schema_fingerprint,
                "created_at": now,
            }
            existing = self.get_control("binding:workspace")
            rows: list[RowWrite] = []
            if existing is not None:
                for field, value in binding.items():
                    if field != "created_at" and existing["value"][field] != value:
                        raise AMLError(
                            "workspace_binding_mismatch", "Immutable workspace binding differs"
                        )
            else:
                if any(next(self.iter_rows(kind), None) is not None for kind in self.entities):
                    raise AMLError("service_unhealthy", "Populated workspace lost its binding")
                rows.append(
                    RowWrite(
                        "control", self.materialize_control("binding:workspace", binding, 0, now)
                    )
                )
            if self.get_control("commit_seq") is None:
                if existing is not None:
                    raise AMLError(
                        "service_unhealthy", "Existing workspace lost its commit counter"
                    )
                rows.append(
                    RowWrite("control", self.materialize_control("commit_seq", {"seq": 0}, 0, now))
                )
            lease = self.get_control("lease:writer")
            epoch = 1 if lease is None else lease["value"]["writer_epoch"] + 1
            # The lifetime this boot replaces; its Catalog proof must name it.
            self.__dict__["_replaced_writer_lease"] = None if lease is None else (
                lease["value"]["owner_instance_id"], lease["value"]["writer_epoch"],
            )
            if rows:
                _, plan = self._prepare_rows(rows, None)
                self._write_plan(plan)
            intended = self.materialize_control(
                "lease:writer",
                {"owner_instance_id": new_ulid(), "writer_epoch": epoch, "acquired_at": now},
                0 if lease is None else lease["version"] + 1,
                now,
            )
            _, plan = self._prepare_rows([RowWrite("control", intended)], None)
            self._ws._router._near.cache.invalidate(self.entities["control"].schema().entity_id)
            self._invalidate_pending_recall_references(plan)
            self.ref_generation += 1
            with self._catalog_steps():
                changed = self._mutate_storage(lambda: self._backend.compare_upsert(
                    plan.entity_ops["main"][0],
                    revision_field="version",
                    expected_revision=None if lease is None else lease["version"],
                ))
                if not changed:
                    raise AMLError("lease_lost", "Writer lease changed during boot acquisition")
                self._mutate_storage(lambda: self._ws._writer.write_ops(plan.catalog_ops))
            self._boot_writer_lease_json = canonical_json(intended)
            return epoch

    @property
    def add_tiers(self) -> AddTiersV1 | None:
        """The workspace's per-user Add tiers; None without them or without a configuration."""
        if self._provider_config_json is None:
            return None
        return ProviderConfig.model_validate_json(self._provider_config_json).add_tiers

    def _bench_job_bindings(self) -> JobProviderBindingsV1:
        """Resolve bootstrap-owned configuration and the actual retained policy."""
        from aml.core.retained_policy import read_released_extraction_config
        from aml.providers.bindings import job_provider_bindings

        if self._job_cas_needs_restart:
            raise AMLError("service_unhealthy", "Job CAS requires startup Catalog recovery")
        if self._provider_config_json is None:
            raise AMLError("unsupported_variant", "Job binding requires initialized bench config")
        config = ProviderConfig.model_validate_json(self._provider_config_json)
        if config.profile != "bench":
            raise AMLError("unsupported_variant", "Only bench workspaces can bind jobs")
        if config.embedding_dimensions != self.dimension:
            raise AMLError("workspace_binding_mismatch", "Job embedding dimension differs")
        lease = self.get_control("lease:writer")
        if self._boot_writer_lease_json is None or lease is None or (
            canonical_json(lease) != self._boot_writer_lease_json
        ):
            raise AMLError("lease_lost", "Job infrastructure lost its boot Writer identity")
        pointer = self.get_control("policy:extraction")
        if pointer is None:
            raise AMLError("reference_missing", "Job binding requires the initialized policy")
        parsed_pointer = self._parse_row("control", pointer)
        assert isinstance(parsed_pointer, ControlRowV1)
        value = parsed_pointer.value
        if not isinstance(value, PolicyPointerV1):
            raise AMLError("binding_mismatch", "Job extraction pointer has another value")
        release = self.read_logical(value.current_release_ref, "policy_release")
        owner = self.get_row("run", release.model_dump(mode="json")["release_id"])
        if owner is None:
            raise AMLError("binding_mismatch", "Job extraction release has no actual owner")
        extraction = read_released_extraction_config(
            self, value.current_release_ref,
        )
        return job_provider_bindings(config, extraction)

    def require_bench_profile_invariants(self) -> ControlRowV1 | None:
        """Verify the single job ownership before bench admission.

        The initialized profile supplies authority; no job need exist on first
        bootstrap. Indexed keyset pages contain at most two job Controls and
        never scan source content.

        Returns:
            The single validated existing job, or None before its first binding.

        Raises:
            AMLError: For another profile, malformed Control state or competing job
                ownership. No repair or cleanup is inferred.
        """
        from aml.store.indexed import IndexedReader

        with self.mutex:
            if self._provider_config_json is None or ProviderConfig.model_validate_json(
                self._provider_config_json,
            ).profile != "bench":
                raise AMLError("unsupported_variant", "Bench invariants require initialized config")
            reader = IndexedReader(self._ws, self.entities, tenant_id=self.tenant_id)
            selected: ControlRowV1 | None = None
            after: str | None = None
            for _ in range(50001):
                page = reader.bench_job_controls(after=after)
                for raw in page:
                    row = self._parse_row("control", raw)
                    assert isinstance(row, ControlRowV1)
                    if not row.key.startswith("job:") or selected is not None:
                        raise AMLError(
                            "workspace_binding_mismatch", "Workspace has competing job state",
                        )
                    if not isinstance(row.value, JobValueV1) or (
                        row.updated_by_run_id is not None or row.control_refs
                    ):
                        raise AMLError("binding_mismatch", "Job Control has unexpected provenance")
                    selected = row
                if len(page) < 2:
                    return selected
                after = page[-1]["key"]
            raise AMLError("budget_exceeded", "Bench job history exceeds its indexed read bound")

    def _bench_job_row(self, job_id: str) -> ControlRowV1 | None:
        """Prove the single global job association without interpreting its opaque ID."""
        if type(job_id) is not str or not 1 <= len(job_id) <= 1020:
            raise AMLError("schema_invalid", "Job identity must contain 1 to 1020 characters")
        selected = self.require_bench_profile_invariants()
        if selected is not None and selected.key != f"job:{job_id}":
            raise AMLError("workspace_binding_mismatch", "Workspace belongs to another job")
        return selected

    def _require_unbound_job_workspace(self) -> None:
        """Prevent assigning prior unassociated scoped data to a new external job."""
        for kind in self.entities:
            for row in self.iter_rows(kind):
                if kind == "control":
                    key = row["key"]
                    if not isinstance(key, str) or not key or len(key) > 1024:
                        raise AMLError("schema_invalid", "Control key length is invalid")
                    try:
                        occupied = control_scope_id(key) is not None
                    except AMLError as error:
                        if error.code not in {"unsupported_variant", "schema_invalid"}:
                            raise
                        # An unopened or unrecognized namespace keeps its typed
                        # unsupported value; its row is still prior data this
                        # workspace holds, so binding a new job fails closed.
                        occupied = True
                elif kind in {"artifact", "run"}:
                    occupied = row.get("scope_id") is not None or row.get("scope_kind") is not None
                else:
                    occupied = True
                if occupied:
                    raise AMLError(
                        "workspace_binding_mismatch", "Unbound workspace already has scoped data",
                    )

    @staticmethod
    def _job_timestamp(value: object) -> int:
        """Apply the actual strict system timestamp domain before any job mutation."""
        try:
            return _JOB_TIME.validate_python(value, strict=True)
        except ValidationError:
            raise AMLError("schema_invalid", "Job time must be a valid system timestamp") from None

    def _compare_job_control(
        self, previous: ControlRowV1 | None, intended: ControlRowV1,
    ) -> ControlRowV1:
        """Apply one no-Run job CAS; only exact readback can resolve a lost reply."""
        if not self.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise AMLError("reader_busy", "Job CAS requires the Writer mutex")
        if not intended.key.startswith("job:") or not isinstance(intended.value, JobValueV1) or (
            intended.updated_by_run_id is not None or intended.control_refs
            or intended.version != (0 if previous is None else previous.version + 1)
            or previous is not None and intended.key != previous.key
        ):
            raise AMLError("invalid_write_mode", "Job CAS has another infrastructure shape")
        raw_previous = None if previous is None else previous.model_dump(mode="json")
        if self.get_control(intended.key) != raw_previous:
            raise AMLError("parity_mismatch", "Job CAS lost its exact preimage")
        _, plan = self._prepare_rows([RowWrite("control", intended)], None)
        if len(plan.entity_ops["main"]) != 1:
            raise AMLError("invalid_write_mode", "Job CAS requires exactly one main row")
        self._ws._router._near.cache.invalidate(self.entities["control"].schema().entity_id)
        # A first binding changes workspace authority. Later traffic updates
        # change only a reference-free watermark, not any provenance or issued
        # Why cursor. Raw row retirement still brackets every storage attempt;
        # the restart latch still protects a failed Catalog acknowledgement.
        self._invalidate_pending_recall_references(plan)
        if previous is None:
            self.ref_generation += 1
        elif not isinstance(previous.value, JobValueV1) or (
            intended.value.provider_bindings != previous.value.provider_bindings
        ):
            raise AMLError("binding_mismatch", "Traffic cannot replace its immutable job binding")
        self._job_cas_needs_restart = True
        with self._catalog_steps():
            try:
                changed = self._mutate_storage(lambda: self._backend.compare_upsert(
                    plan.entity_ops["main"][0], revision_field="version",
                    expected_revision=None if previous is None else previous.version,
                ))
            except Exception:
                changed = False
            if not changed:
                actual = self.get_control(intended.key)
                if actual != intended.model_dump(mode="json"):
                    if actual == raw_previous:
                        raise AMLError(
                            "service_unhealthy", "Job CAS did not commit; retain traffic",
                        )
                    raise AMLError(
                        "parity_mismatch", "Job CAS has an unrecognized durable outcome",
                    )
            self._mutate_storage(lambda: self._ws._writer.write_ops(plan.catalog_ops))
        self._job_cas_needs_restart = False
        return intended

    def bind_bench_job(
        self, job_id: str, provider_bindings: JobProviderBindingsV1, *, clock: Callable[[], int],
    ) -> ControlRowV1:
        """Create or verify one immutable job binding before service admission.

        Args:
            job_id: Opaque external identity, never a filesystem or scope locator.
            provider_bindings: Complete tuple matching initialized configuration
                and the actual retained extraction release.
            clock: Trusted creation time; replay preserves all previous row bytes.

        Returns:
            The strict existing or newly confirmed global Control. New jobs use
            traffic time zero until a successful Add or Search is observed.

        Raises:
            AMLError: If identity, profile, bindings, ownership or CAS proof fails.
                Creating a job after an ordinary publisher exists is forbidden.

        This is the narrow no-Run producer from plan.md §5.9. It never advances
        business/scope generations or calls a provider; one workspace holds one job.
        """
        try:
            supplied = JobProviderBindingsV1.model_validate(
                provider_bindings.model_dump(mode="json"),
            )
        except (AttributeError, ValidationError):
            raise AMLError("schema_invalid", "Job provider bindings must be strict") from None
        with self._write_scope():
            expected = self._bench_job_bindings()
            if supplied != expected:
                raise AMLError("binding_mismatch", "Job providers differ from initialized bindings")
            previous = self._bench_job_row(job_id)
            if previous is not None:
                if not isinstance(previous.value, JobValueV1) or (
                    previous.value.provider_bindings != supplied
                ):
                    raise AMLError("binding_mismatch", "Immutable job provider bindings differ")
                return previous
            from aml.store.read_model import StoreReadModel

            if StoreReadModel.for_writer(self) is not None:
                raise AMLError("invalid_write_mode", "Bind the job before opening normal readers")
            self._require_unbound_job_workspace()
            intended = self.materialize_control(f"job:{job_id}", JobValueV1(
                provider_bindings=supplied, last_platform_traffic_at=0,
            ), 0, self._job_timestamp(clock()))
            return self._compare_job_control(None, intended)

    def flush_job_traffic(
        self, job_id: str, observed_at: int, *, clock: Callable[[], int],
    ) -> ControlRowV1:
        """Coalesce successful traffic with the durable monotonic maximum.

        Args:
            job_id: The sole immutable external job association in this workspace.
            observed_at: Trusted successful Add/delivery time from the service's
                in-memory atomic maximum, retained by that caller after failure.
            clock: Trusted bookkeeping time; it does not limit the watermark.

        Returns:
            The exact existing row for older/equal observations, otherwise the
            confirmed CAS postimage. No Run, snapshot or data generation changes.

        Raises:
            AMLError: If the job, initialized binding, timestamp or CAS is invalid.

        The service owns the at-most-once-per-60-second scheduling rule. This
        storage primitive owns only plan.md §5.9's one-row no-Run monotonic CAS.
        """
        observed_at = self._job_timestamp(observed_at)
        with self._write_scope():
            expected = self._bench_job_bindings()
            previous = self._bench_job_row(job_id)
            if previous is None:
                raise AMLError("reference_missing", "Traffic requires the existing job binding")
            if not isinstance(previous.value, JobValueV1) or (
                previous.value.provider_bindings != expected
            ):
                raise AMLError("binding_mismatch", "Traffic cannot change immutable job bindings")
            if observed_at <= previous.value.last_platform_traffic_at:
                return previous
            intended = self.materialize_control(previous.key, JobValueV1(
                provider_bindings=previous.value.provider_bindings,
                last_platform_traffic_at=observed_at,
            ), previous.version + 1, self._job_timestamp(clock()))
            return self._compare_job_control(previous, intended)

    def register_scope(
        self, scope_kind: str, scope_id: str, raw_external_id: str, now: int,
        *, grant: PrincipalGrant | None = None, clock: Callable[[], int] | None = None,
    ) -> None:
        """Create the five pre-Run Controls without repairing orphaned scope state.

        Search supplies its detached actual grant and trusted clock for a final
        authorization check immediately before the registration write. Existing
        reception callers retain their own admission and capacity boundaries.
        """
        from aml.core.query import ScopeV1
        from aml.ingestion.authorization import check_add_grant
        from aml.providers.auth import PrincipalGrant

        scope = ScopeV1.model_validate({"scope_kind": scope_kind, "scope_id": scope_id})
        resolved = None if grant is None else PrincipalGrant.model_validate(
            grant.model_dump(mode="json"),
        )
        if resolved is not None and clock is None:
            raise AMLError("schema_invalid", "Authorized registration requires its action clock")
        with self._write_scope():
            if resolved is not None and clock is not None:
                check_add_grant(resolved, scope, clock())
            if scope_id != canonical_scope_id(scope_kind, raw_external_id):
                raise AMLError(
                    "parity_mismatch", "Scope identity differs from its canonical source"
                )
            existing = self.get_control(f"scope:{scope_id}")
            expected = {"scope_kind": scope_kind, "raw_external_id": raw_external_id}
            if existing is not None:
                checked = ControlRowV1.model_validate(existing)
                if (
                    checked.key != f"scope:{scope_id}"
                    or checked.object_id != control_id(checked.key)
                    or checked.value.model_dump(mode="json") != expected
                    or checked.version != 0 or checked.updated_by_run_id is not None
                    or checked.control_refs
                ):
                    raise AMLError(
                        "operation_conflict", "Scope hash maps to different source identity"
                    )
                for prefix in _SCOPE_COUNTERS:
                    self.counter(f"{prefix}:{scope_id}")
                return
            self._require_unoccupied_scope(scope, frozenset())
            rows = [
                RowWrite("control", self.materialize_control(f"scope:{scope_id}", expected, 0, now))
            ]
            rows.extend(
                RowWrite(
                    "control", self.materialize_control(f"{prefix}:{scope_id}", {"seq": 0}, 0, now)
                )
                for prefix in _SCOPE_COUNTERS
            )
            _, plan = self._prepare_rows(rows, None)
            if resolved is not None and clock is not None:
                check_add_grant(resolved, scope, clock())
            self._write_plan(plan)

    def _require_unoccupied_scope(self, scope: ScopeV1, allowed_controls: frozenset[str]) -> None:
        """Reject existing domain/control authority before projecting an empty scope."""
        from aml.store.indexed import IndexedReader

        reader = IndexedReader(
            self.runtime.workspace,
            self.entities,
            tenant_id=self.tenant_id,
        )
        if self.get_control(f"tomb:{scope.scope_id}") is not None:
            raise AMLError("scope_denied", "The requested scope was removed")
        if reader.pending_privacy(scope.scope_id, limit=1) or reader.scope_fences(scope.scope_id):
            raise AMLError("index_rebuilding", "Scope registration cannot cross privacy fences")
        for row in self.iter_rows("control"):
            if (
                control_key_occupies_scope(row["key"], scope.scope_id)
                and row["key"] not in allowed_controls
            ):
                raise AMLError("service_unhealthy", "Scope state lacks its complete registration")
        if reader.scope_has_rows(scope.scope_kind, scope.scope_id):
            raise AMLError("service_unhealthy", "Existing scope data requires its complete root")

    def read_initial_scope_registration(
        self, scope: ScopeV1, raw_external_id: str,
    ) -> tuple[ControlRowV1, ...]:
        """Read only the exact initial five-Control state under the Writer mutex.

        This excludes all scoped data and additional Controls. It cannot repair
        a missing data projection or turn an existing cycle into an empty scope.
        """
        self.validate_storage_owner()
        if scope.scope_id != canonical_scope_id(scope.scope_kind, raw_external_id):
            raise AMLError("binding_mismatch", "Registration requires its canonical raw identity")
        keys = tuple(
            f"{prefix}:{scope.scope_id}" for prefix in ("scope", *_SCOPE_COUNTERS)
        )
        self._require_unoccupied_scope(scope, frozenset(keys))
        rows = tuple(ControlRowV1.model_validate(self.get_control(key)) for key in keys)
        for position, (key, row) in enumerate(zip(keys, rows, strict=True)):
            value = (
                {"scope_kind": scope.scope_kind, "raw_external_id": raw_external_id}
                if position == 0 else {"seq": 0}
            )
            if row != self.materialize_control(key, value, 0, rows[0].updated_at):
                raise AMLError("parity_mismatch", "Empty projection requires initial Control bytes")
        return rows

    def admit_run(self, row: StrictModel) -> WriteTicket:
        """Create a fully validated Run before any of its immutable artifacts."""
        return self._admit_run(row)

    def admit_add_root(self, row: StrictModel) -> WriteTicket:
        """Admit an ordinary Add's parentless reception root as reception bookkeeping.

        The root has no input until its ordering commit, so its admission leaves
        the scope change tokens of a finalize owner's proofs unchanged
        (``_reception_bookkeeping``). ``_admit_run`` checks the empty state.
        """
        data = row.model_dump()
        if (data.get("run_kind"), data.get("commit_mode"), data.get("target_kind")) != (
            "session_ingestion", "orchestrator", "session",
        ) or data.get("parent_run_id") is not None:
            raise AMLError("invalid_run_admission", "An Add root is a parentless session reception")
        with self._reception_bookkeeping(data["object_id"]):
            return self._admit_run(row)

    def _finalize_identity(
        self, issued: FinalizeLeaseTicket, *, rebased: tuple[int, int] | None = None,
    ) -> tuple[dict[str, Any], ControlRowV1, WriteTicket]:
        """Check the issued cycle before a dedicated coordination effect.

        ``rebased`` is used only by the controlled privacy transition
        (``finalize_privacy.rebase_finalize_privacy``, C.11.1 #1): it names the
        scope's current ``(erase_epoch, consent_epoch)`` that transition is
        about to write to the owner, so every other identity, lease and
        workspace/scope fence below is proved against the epochs the owner
        will hold. Every other caller passes None and is fenced by the
        owner's admitted epochs.
        """
        if self.scope_quarantined(issued.scope_id):
            raise AMLError("scope_quarantined", "The scope requires explicit recovery repair")
        saved = self._finalize_tickets.get(issued.owner_run_id)
        if saved is not issued:
            raise AMLError("lease_lost", "Finalize ticket was not issued by this Writer")
        run = self.get_row("run", issued.owner_run_id)
        raw = self.get_control(f"lease:finalize:{issued.scope_id}")
        if run is None or raw is None:
            raise AMLError("lease_lost", "Finalize lost its durable owner or lease")
        lease = self._validated_finalize_rows(run, raw)
        value = lease.value.model_dump(mode="json")
        identity, digest = self._finalize_admission_identity(issued, run)
        # A self-consistent replacement of principal, family and digest still
        # differs from the canonical digest retained at this ticket's issuance.
        retained = self._finalize_digests.get(issued)
        if (run["idempotency_key"], run["admission_digest"], retained) != (
            identity, digest, digest,
        ):
            raise AMLError("binding_mismatch", "Finalize admission changed after issuance")
        if (
            run["run_kind"] != "finalize" or run["commit_mode"] != "orchestrator"
            or run["target_kind"] != "scope" or run["target_id"] != issued.scope_id
            or run["scope_id"] != issued.scope_id or run["parent_run_id"] is not None
            or lease.updated_by_run_id != issued.owner_run_id
            or run["run_generation"] != 0
            or issued.state == "preparing" and run["attempt"] != 0
            # An attempt after the first is an explicit resume of the frozen owner
            # (C.13 R08, ingest-Q17): its selected input exists.
            or run["attempt"] > 0 and run["input_ref"] is None
            or (run["status"] != "running" and not (
                run["status"] == "manual_attention" and (
                    run["failure_code"] == "parity_mismatch" and (
                        issued.state == "preparing" and value.get("drain_state") == "closed"
                        or issued.state == "held" and run["stage"] == "release"
                        and run["input_ref"] is not None and run["result_ref"] is not None
                    )
                )
            )) or run["writer_epoch"] != issued.writer_epoch
            or (value["owner_run_id"], value["lease_generation"], value["writer_epoch"],
                value["state"])
            != (issued.owner_run_id, issued.lease_generation, issued.writer_epoch, issued.state)
        ):
            raise AMLError("lease_lost", "Finalize durable execution identity changed")
        if issued.state == "preparing" and (
            run["stage"] != "lease" or any(run[name] is not None for name in (
                "input_ref", "result_ref", "auth_context_ref", "commit_marker", "commit_seq",
            ))
        ):
            raise AMLError("binding_mismatch", "Preparing finalize has execution artifacts")
        erase_epoch, consent_epoch = rebased or (
            run["admitted_erase_epoch"], run["admitted_consent_epoch"],
        )
        ticket = WriteTicket(
            issued.writer_epoch, issued.owner_run_id, run["scope_kind"], issued.scope_id,
            erase_epoch, consent_epoch,
        )
        # Coordination precedes the held-only ordinary execution gate. The
        # dedicated identity proof above and ordinary workspace/scope/epoch
        # checks below do not grant generic Artifact or business write access.
        self._admit(replace(ticket, run_id=None))
        self._register_finalize_write_ticket(ticket, issued)
        return run, lease, ticket

    def _validated_finalize_rows(
        self, run: dict[str, Any], raw: dict[str, Any],
    ) -> ControlRowV1:
        """Validate a finalize owner's Run row and parse its lease, reusing equal images.

        Every Writer action under a finalize owner checks its identity first,
        tens of thousands of times in a 976-position reconcile, and each check
        validated the same two rows again; parsing the lease tries the Control
        value models in turn. Both are pure functions of the rows, so the last
        images validated for each owner are kept by their canonical bytes, and
        rows with the same bytes reuse that result. Like ``read_logical``, each
        call takes its own shallow copy of the lease.
        """
        try:
            images = canonical_json(run), canonical_json(raw)
        except (TypeError, ValueError):
            images = None
        known = self.__dict__.get("_finalize_row_memo")
        if known is None:
            known = self.__dict__.setdefault("_finalize_row_memo", OrderedDict())
        memo = cast(OrderedDict[str, tuple[tuple[str, str], ControlRowV1]], known)
        entry = memo.get(run["object_id"])
        if images is not None and entry is not None and entry[0] == images:
            return entry[1].model_copy()
        self._parse_row("run", run)
        lease = ControlRowV1.model_validate(raw)
        if images is not None:
            memo[run["object_id"]] = images, lease.model_copy(deep=True)
            while len(memo) > 256:
                memo.popitem(last=False)
        return lease

    def _finalize_precondition(
        self, issued: FinalizeLeaseTicket,
    ) -> tuple[dict[str, Any], ControlRowV1, WriteTicket]:
        """Fence coordination and descendants using the Writer-bound live clock.

        The lease is judged at the owner's action time (``_finalize_action_time``),
        and the heartbeat it read is kept for ``owner_lease.renew_due_leases``.
        """
        run, lease, ticket = self._finalize_identity(issued)
        clock = self._finalize_clocks.get(issued)
        if clock is None:
            raise AMLError("lease_lost", "Finalize has no current live clock authority")
        heartbeat_at = getattr(lease.value, "heartbeat_at", None)
        if isinstance(heartbeat_at, int):
            self._finalize_heartbeats()[issued] = heartbeat_at
        self._check_finalize_heartbeat(
            lease, self._job_timestamp(self._finalize_action_time(issued, clock)),
        )
        return run, lease, ticket

    def _finalize_action_time(
        self, issued: FinalizeLeaseTicket, clock: Callable[[], int],
    ) -> int:
        """Return the time at which the owner's current action judges its lease (C.2a).

        One Writer critical section is one action: nothing outside it can renew,
        settle or take over a lease while it runs. Every fence of one owner in a
        section therefore judges the lease at the section's first fence of that
        owner, so a scope refuses an expired lease before its first write and
        never between two of its writes, where the refusal would fail the
        publisher closed. A judge child's admission writes the child Run and then
        its snapshot, input and start: in the full a7 workload the lease was alive
        at admission 708's first fence and expired before a later one. A thread
        outside the mutex reads the clock at every check.
        """
        mutex = self.mutex
        if not mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            return clock()
        section = mutex.sequence
        pinned = cast(tuple[int, dict[str, int]] | None, self.__dict__.get("_action_times"))
        if pinned is None or pinned[0] != section:
            pinned = (section, dict[str, int]())
            self.__dict__["_action_times"] = pinned
        times = pinned[1]
        now = times.get(issued.owner_run_id)
        if now is None:
            now = times[issued.owner_run_id] = clock()
        return now

    def _finalize_heartbeats(self) -> WeakKeyDictionary[FinalizeLeaseTicket, int]:
        """Return each live ticket's lease heartbeat as this Writer last read it.

        These only choose which live leases ``owner_lease.renew_due_leases``
        rereads at the end of a write scope; every renewal reproves its lease.
        A fixture Writer built around ``__init__`` starts empty too.
        """
        seen = cast(
            WeakKeyDictionary[FinalizeLeaseTicket, int] | None,
            self.__dict__.get("_seen_heartbeats"),
        )
        if seen is None:
            seen = WeakKeyDictionary[FinalizeLeaseTicket, int]()
            self.__dict__["_seen_heartbeats"] = seen
        return seen

    @staticmethod
    def _check_finalize_heartbeat(lease: ControlRowV1, now: int) -> None:
        value = lease.value.model_dump(mode="json")
        if now - value["heartbeat_at"] > 60_000:
            raise AMLError("lease_lost", "Finalize heartbeat expired before the action")

    def _finalize_admission_identity(
        self, ticket: FinalizeLeaseTicket, run: dict[str, Any],
    ) -> tuple[str, ContentBindingDigest]:
        """Recompute the owner's canonical finalize family and admission digest.

        The pair is a pure function of the principal, scope and cycle under
        this Writer's fixed content key, so recent pairs are remembered.
        """
        from aml.core.admission import admission_identity, finalize_admission
        from aml.core.query import ScopeV1

        key = (run["principal_id"], run["scope_kind"], ticket.scope_id, ticket.lease_generation)
        known = self.__dict__.get("_finalize_admission_memo")
        if known is None:
            known = self.__dict__.setdefault("_finalize_admission_memo", OrderedDict())
        memo = cast(OrderedDict[tuple[Any, ...], tuple[str, ContentBindingDigest]], known)
        remembered = memo.get(key)
        if remembered is not None:
            return remembered
        canonical = finalize_admission(
            principal_id=run["principal_id"],
            scope=ScopeV1(scope_kind=run["scope_kind"], scope_id=ticket.scope_id),
            cycle=ticket.lease_generation,
        )
        identity, digest, _ = admission_identity(canonical, self.bind_content)
        memo[key] = identity, digest
        while len(memo) > 256:
            memo.popitem(last=False)
        return identity, digest

    def _register_finalize_ticket(
        self, ticket: FinalizeLeaseTicket, clock: Callable[[], int],
    ) -> None:
        """Keep the trusted clock and canonical digest outside the mutable public identity.

        The durable owner's admission digest is retained at issuance so every
        later precondition rechecks the same canonical identity, not merely a
        row that is self-consistent with whatever principal it now carries.
        """
        run = self.get_row("run", ticket.owner_run_id)
        if run is None:
            raise AMLError("lease_lost", "Finalize ticket lost its durable owner")
        identity, digest = self._finalize_admission_identity(ticket, run)
        if (run["idempotency_key"], run["admission_digest"]) != (identity, digest):
            raise AMLError("binding_mismatch", "Finalize admission changed after issuance")
        self._finalize_tickets[ticket.owner_run_id] = ticket
        self._finalize_clocks[ticket] = clock
        self._finalize_digests[ticket] = digest

    def _register_finalize_write_ticket(
        self, ticket: WriteTicket, owner: FinalizeLeaseTicket | None,
    ) -> WriteTicket:
        """Bind an actual transient descendant ticket, rejecting equal dataclass copies.

        The binding keeps its owner lease ticket alive exactly as long as this
        ticket object. A transient ticket minted by a borrowed reread dies
        wherever its last reference drops: outside the Writer mutex, or inside
        a garbage collection that interrupted any critical section, the
        mutex's own bookkeeping included. Its collection therefore takes no
        lock (``IdentityMemo``): it releases the owner at once and leaves the
        inert entry to a later registration or lookup.
        """
        if owner is not None:
            self._finalize_write_tickets.remember(ticket, owner)
        return ticket

    def finalize_ticket(
        self, owner_run_id: str, *,
        clock: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    ) -> FinalizeLeaseTicket:
        """Bind an existing lease to a live identity (``owner_lease.finalize_ticket``)."""
        from aml.store.owner_lease import finalize_ticket

        return finalize_ticket(self, owner_run_id, clock=clock)

    def _compare_finalize_lease(
        self, issued: FinalizeLeaseTicket, previous: ControlRowV1, intended: ControlRowV1,
    ) -> None:
        """Perform exactly one proved coordinating CAS through real HeavenBase."""
        _, current, _ = self._finalize_precondition(issued)
        if current != previous or (
            intended.key != current.key or intended.version != current.version + 1
            or intended.updated_by_run_id != issued.owner_run_id
        ):
            raise AMLError("lease_lost", "Finalize CAS preimage changed")
        self._write_finalize_lease_cas(previous, intended)

    def _write_finalize_lease_cas(
        self, previous: ControlRowV1, intended: ControlRowV1,
    ) -> None:
        """Execute the existing lease CAS after live or startup proof under exclusion."""
        from aml.core.controls import HeldFinalizeLeaseV1, PreparingFinalizeLeaseV1

        owns_mutex = self.mutex._is_owned()  # pyright: ignore[reportPrivateUsage]
        if not owns_mutex or intended.key != previous.key or (
            intended.version != previous.version + 1
        ):
            raise AMLError("lease_lost", "Finalize CAS requires its excluded preimage")
        _, plan = self._prepare_rows([RowWrite("control", intended)], None)
        if len(plan.entity_ops["main"]) != 1:
            raise AMLError("service_unhealthy", "Finalize lease did not encode one main row")
        self._ws._router._near.cache.invalidate(self.entities["control"].schema().entity_id)
        self._invalidate_pending_recall_references(plan)
        # A heartbeat-only renewal keeps the same owner, cycle, epoch and state
        # and only moves the expiry later, so it reserves no reference
        # generation: no proof can depend on the lease not being renewed, and a
        # fence that needs the lease checks its owner and epoch itself under
        # the Writer mutex (``_finalize_precondition``). Every other lease
        # transition (drain, hold, takeover, adoption, cancellation, release)
        # reserves one like any other write. Compare the complete row so
        # reference projections and owner identity cannot enter this exception.
        heartbeat_only = (
            isinstance(previous.value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1)
            and isinstance(intended.value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1)
            and type(intended.value) is type(previous.value)
            and intended.value.heartbeat_at >= previous.value.heartbeat_at
            and intended.updated_at == intended.value.heartbeat_at
            and intended.value.model_copy(update={
                "heartbeat_at": previous.value.heartbeat_at,
            }) == previous.value
            and intended.model_copy(update={
                "value": previous.value, "version": previous.version,
                "updated_at": previous.updated_at,
            }) == previous
        )
        if not heartbeat_only:
            self.ref_generation += 1
        watch: list[tuple[int, tuple[tuple[str, Any], ...] | None]] | None = (
            [] if heartbeat_only else None
        )
        before = self.storage_epoch
        self.__dict__["_heartbeat_watch"] = watch
        try:
            with self._catalog_steps():
                try:
                    changed = self._mutate_storage(lambda: self._backend.compare_upsert(
                        plan.entity_ops["main"][0], revision_field="version",
                        expected_revision=previous.version,
                    ))
                except Exception:
                    observed = self.get_control(intended.key)
                    if observed != intended.model_dump(mode="json"):
                        raise
                    changed = True
                if not changed:
                    raise AMLError("lease_lost", "Finalize lease CAS lost its expected version")
                self._mutate_storage(lambda: self._ws._writer.write_ops(plan.catalog_ops))
            if watch is not None:
                # Journaled while the watch is still set: a proof read in
                # between waits for exactly this (``_heartbeat_run``).
                self._journal_heartbeat(intended, before, watch)
        except BaseException:
            if heartbeat_only:
                # A renewal that did not complete is not known to have changed
                # only the heartbeat: reserve a generation, so the enclosing
                # write scope fails its readers closed as for any other write.
                self.ref_generation += 1
            raise
        finally:
            self.__dict__["_heartbeat_watch"] = None

    def _journal_heartbeat(
        self, intended: ControlRowV1, before: int,
        watch: list[tuple[int, tuple[tuple[str, Any], ...] | None]],
    ) -> None:
        """Extend the owner's run of heartbeat-only epoch advances by this CAS.

        Called under the Writer mutex right after the CAS. Its interval counts
        only when every ``storage_epoch`` advance in it was one of this
        thread's own mutations naming no memoized row but the lease row
        (Catalog rows are not memoized): each advance is exactly one
        ``_rows_mutated`` call, so a count that differs reveals any other
        mutation, including one outside the mutex. An interval that starts
        where the owner's run ends extends it; any other starts a new run, so
        a run never spans a mutation that was not this owner's heartbeat.
        """
        owner = intended.updated_by_run_id
        after = self.storage_epoch
        kinds = self._kinds_by_entity_id()
        thread = threading.get_ident()
        lease = ("control", intended.object_id)
        if owner is None or kinds is None or after - before != len(watch) or any(
            ident != thread or keys is None or any(
                kinds.get(entity_id) is not None and (kinds[entity_id], object_id) != lease
                for entity_id, object_id in keys
            )
            for ident, keys in watch
        ):
            return
        journal: OrderedDict[str, tuple[int, int]] = self.__dict__.setdefault(
            "_heartbeat_epochs", OrderedDict(),
        )
        run = journal.pop(owner, None)
        journal[owner] = (run[0], after) if run is not None and run[1] == before else (
            before, after,
        )
        while len(journal) > _HEARTBEAT_OWNERS:
            journal.popitem(last=False)

    def finalize_heartbeats_only_since(self, owner_run_id: str, token: int) -> bool:
        """Prove that every row mutation after ``token`` was this owner's own lease heartbeat.

        A heartbeat-only CAS keeps the complete lease row except its heartbeat
        time and version (``_write_finalize_lease_cas``); when it alone
        advanced ``storage_epoch``, it extends this owner's latest run of such
        advances, kept under the Writer mutex (``_journal_heartbeat``).
        ``True`` means that run covers every advance from ``token`` to now.
        Any other mutation, including another owner's heartbeat, a takeover, a
        fence or a release, ends the run, so a token before it returns
        ``False``. A borrower relying on this must still check its lease,
        which is the one row the run changed.
        """
        if self.storage_epoch == token:
            return True
        run = self._heartbeat_run(owner_run_id)
        return run is not None and run[0] <= token

    def _heartbeat_run(self, owner_run_id: str) -> tuple[int, int] | None:
        """Return the owner's run of heartbeat-only advances if it reaches the current epoch.

        A heartbeat CAS advances ``storage_epoch`` before it journals its run,
        both while it holds the Writer mutex and while ``_heartbeat_watch`` is
        set, so a read in between misses an advance that is only a heartbeat.
        A run that does not reach the current epoch is therefore decided only
        in a window in which no critical section began or ended
        (``WriterMutex.sequence``), after waiting briefly for a heartbeat in
        flight and for no other section.
        """
        def reaching() -> tuple[int, int] | None:
            journal: OrderedDict[str, tuple[int, int]] | None = self.__dict__.get(
                "_heartbeat_epochs",
            )
            run = None if journal is None else journal.get(owner_run_id)
            return run if run is not None and run[1] == self.storage_epoch else None

        run = reaching()
        if run is not None:
            return run
        mutex = self.mutex
        for _ in range(BORROWED_REFRESH_ATTEMPTS):
            progress = mutex.progress_generation
            sequence = mutex.sequence
            run = reaching()
            if run is not None:
                return run
            owned = mutex._is_owned()  # pyright: ignore[reportPrivateUsage]
            if owned or mutex.sequence == sequence and not sequence % 2:
                return None
            if self.__dict__.get("_heartbeat_watch") is not None:
                mutex.wait_for_progress(
                    progress, deadline=time.monotonic() + BORROWED_REFRESH_WAIT_SECONDS,
                )
        return None

    def admit_finalize(
        self, *, scope: ScopeV1, grant: PrincipalGrant, clock: Callable[[], int],
        resources: ResourceObserver | None = None,
        admission_trigger: Literal["first_search", "idle", "add_completion"] | None = None,
        completion_policy: RequestCompletionPolicyV2 | None = None,
    ) -> FinalizeLeaseTicket:
        """Admit or reuse one exact cycle (``owner_lease.admit_finalize``)."""
        from aml.store.owner_lease import admit_finalize

        return admit_finalize(
            self, scope=scope, grant=grant, clock=clock, resources=resources,
            admission_trigger=admission_trigger,
            completion_policy=completion_policy,
        )

    def close_finalize_drain(
        self, ticket: FinalizeLeaseTicket, *, clock: Callable[[], int],
    ) -> None:
        """Close the preparing drain (``owner_lease.close_finalize_drain``)."""
        from aml.store.owner_lease import close_finalize_drain

        close_finalize_drain(self, ticket, clock=clock)

    def hold_finalize(
        self, ticket: FinalizeLeaseTicket, publisher: StoreReadModel,
        *, clock: Callable[[], int],
    ) -> FinalizeLeaseTicket:
        """Install the input fence and its baseline (``owner_lease.hold_finalize``)."""
        from aml.store.owner_lease import hold_finalize

        return hold_finalize(self, ticket, publisher, clock=clock)

    def heartbeat_finalize(
        self, ticket: FinalizeLeaseTicket, *, clock: Callable[[], int],
    ) -> None:
        """Refresh one issued cycle (``owner_lease.heartbeat_finalize``)."""
        from aml.store.owner_lease import heartbeat_finalize

        heartbeat_finalize(self, ticket, clock=clock)

    def put_finalize_source_inventory_block(
        self, lease_ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard,
        body: FinalizeSourceInventoryBlockV1, *, grant: PrincipalGrant,
        clock: Callable[[], int], artifact_generation: int,
        inventory_index: _SourceInventoryIndex | None = None,
    ) -> tuple[ArtifactId, int]:
        """Discover the original first block outside the mutex, then prewrite once.

        The caller supplies the Artifact generation of its complete inventory
        scan. Return this effect's exact generation while still holding the
        Writer; a later write must not be adopted as part of this prewrite.
        Heartbeat renewal keeps the same held baseline and is not a conflict.
        """
        from aml.core.controls import HeldFinalizeLeaseV1
        from aml.core.query import ScopeV1
        from aml.core.work import FinalizeSourceInventoryBlockV1
        from aml.ingestion.authorization import check_add_grant

        if self.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise AMLError("reader_busy", "Source inventory discovery requires an unlocked Writer")
        if type(artifact_generation) is not int or artifact_generation < 0:
            raise AMLError("schema_invalid", "Invalid source inventory Artifact generation")
        block = FinalizeSourceInventoryBlockV1.model_validate(body.model_dump(mode="python"))

        def state() -> tuple[ScopeV1, str, tuple[int, int, int], int, int, tuple[int, str]]:
            run, lease, _ = self._finalize_precondition(lease_ticket)
            scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])
            guard.check_operation(self, scope)
            now = clock()
            check_add_grant(grant, scope, now, run["principal_id"])
            self._check_finalize_heartbeat(lease, now)
            if not isinstance(lease.value, HeldFinalizeLeaseV1):
                raise AMLError("lease_lost", "Source inventory requires held ownership")
            baseline = (
                lease.value.baseline_snapshot_generation, lease.value.baseline_root_identity,
            )
            if baseline != (guard.metadata.generation, guard.metadata.root_identity):
                raise AMLError("binding_mismatch", "Source inventory has another held baseline")
            coordinates = tuple(self.counter(f"{name}:{scope.scope_id}") for name in (
                "scope_seq", "erase_epoch", "consent_epoch",
            ))
            return (
                scope, canonical_json(run), (coordinates[0], coordinates[1], coordinates[2]),
                self._artifact_generations.get(run["object_id"], 0),
                self.child_change_token(run["object_id"]), baseline,
            )

        with self.mutex:
            original = state()
            if original[3] != artifact_generation:
                raise AMLError("stale_evidence", "Inventory prewrite discovery changed")
        first: ArtifactId | None = None
        if block.previous_ref is None:
            first_blocks = [
                (reference, value)
                for reference, value in self.find_logical_candidates(
                    lease_ticket.owner_run_id, "decision_set",
                ) if isinstance(value, FinalizeSourceInventoryBlockV1) and value.block_no == 0
            ]
            if len(first_blocks) > 1 or first_blocks and first_blocks[0][1] != block:
                raise AMLError("binding_mismatch", "Source inventory replaced its first block")
            if first_blocks:
                first = first_blocks[0][0]
            elif block.receipt_high_watermark != original[5][0]:
                raise AMLError("binding_mismatch", "New source inventory changed its cutoff")
        with self._write_scope():
            if state() != original:
                raise AMLError("stale_evidence", "Inventory prewrite discovery changed")
            if first is not None:
                _, _, ticket = self._finalize_precondition(lease_ticket)
                if self._read_owned_business_artifact(ticket, first, "decision_set") != block:
                    raise AMLError("binding_mismatch", "Source inventory changed its first block")
            inventory_token = (None if inventory_index is None else
                               inventory_index.before_write(self, lease_ticket, guard))
            extension: AbstractContextManager[ScanExtension | None] = nullcontext(None)
            if inventory_index is not None and inventory_token is not None:
                try:
                    _, _, effect = self._finalize_precondition(lease_ticket)
                    _, physical = self.materialize_logical_artifact_rows(
                        effect, "decision_set", block, security_level="internal",
                        created_at=block.freeze_time,
                    )
                except AMLError:
                    # Cache preparation never preempts the original write's
                    # authorization/predecessor/capacity refusal ordering.
                    pass
                else:
                    extension = self._rows_cache().extend_artifact_scan(
                        inventory_index.keys, inventory_token,
                        (("artifact", row.object_id) for row in physical),
                        membership_key("artifact", lease_ticket.owner_run_id),
                    )
            # Include callbacks inside the original write. A reentrant raw
            # mutation must deny extension even while HB exclusion is held.
            with extension as receipt:
                reference = self._put_finalize_inventory_block(
                    lease_ticket, guard, block, grant=grant, clock=clock,
                )
                if first is not None and reference != first:
                    raise AMLError(
                        "binding_mismatch", "Source inventory changed its original reference",
                    )
            if inventory_index is not None and inventory_token is not None:
                inventory_index.after_write(
                    inventory_token, None if receipt is None else receipt.token, reference, block,
                )
            return reference, self._artifact_generations.get(lease_ticket.owner_run_id, 0)

    def _put_finalize_inventory_block(
        self, lease_ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard,
        body: FinalizeSourceInventoryBlockV1, *,
        grant: PrincipalGrant, clock: Callable[[], int],
    ) -> ArtifactId:
        """Keep the source inventory chain inside its actual freeze boundary."""
        from aml.core.controls import HeldFinalizeLeaseV1
        from aml.core.query import ScopeV1
        from aml.core.work import FinalizeSourceInventoryBlockV1
        from aml.ingestion.authorization import check_add_grant
        from aml.store.indexed import IndexedReader

        with self._write_scope():
            run, lease, ticket = self._finalize_precondition(lease_ticket)
            scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])

            def check() -> int:
                now = clock()
                guard.check_operation(self, scope)
                check_add_grant(grant, scope, now, run["principal_id"])
                _, current_lease, _ = self._finalize_precondition(lease_ticket)
                self._check_finalize_heartbeat(current_lease, now)
                return now

            now = check()
            if not isinstance(lease.value, HeldFinalizeLeaseV1) or (
                run["result_ref"] is not None
                or run["status"] != "running"
                or IndexedReader(self._ws, self.entities, tenant_id=self.tenant_id).run_children(
                    run["object_id"], limit=1,
                )
            ):
                raise AMLError("invalid_run_transition", "Inventory requires unselected held work")
            block = FinalizeSourceInventoryBlockV1.model_validate(
                body.model_dump(mode="python"),
            )
            if run["input_ref"] is not None or run["stage"] not in {"lease", "freeze"}:
                raise AMLError("invalid_run_transition", "Source inventory is already frozen")
            if any((record.header.tenant_id, record.header.scope_kind, record.header.scope_id)
                   != (self.tenant_id, scope.scope_kind, scope.scope_id)
                   for record in block.records):
                raise AMLError("scope_denied", "Source inventory crossed its exact owner scope")
            if block.receipt_high_watermark > lease.value.baseline_snapshot_generation:
                raise AMLError(
                    "binding_mismatch", "Source inventory cutoff exceeds its baseline",
                )
            if (
                block.owner_run_id != run["object_id"]
                or block.finalize_cycle != lease_ticket.lease_generation
                or block.freeze_time > now
            ):
                raise AMLError("binding_mismatch", "Inventory changed its original held cycle")
            if block.previous_ref is not None:
                previous = self._read_owned_business_artifact(
                    ticket, block.previous_ref, "decision_set",
                )
                if not isinstance(previous, type(block)) or previous.is_head or (
                    previous.block_no + 1 != block.block_no
                    or previous.total_record_count + block.record_count
                    + (block.base_count - len(block.removed_ids) if block.is_head else 0)
                    != block.total_record_count
                    or previous.owner_run_id != block.owner_run_id
                    or previous.finalize_cycle != block.finalize_cycle
                    or previous.freeze_time != block.freeze_time
                    or previous.root_input_heads_digest != block.root_input_heads_digest
                    or previous.base_ref != block.base_ref
                    or previous.base_count != block.base_count
                    or previous.removed_ids != block.removed_ids
                ):
                    raise AMLError("binding_mismatch", "Inventory changed its original predecessor")
                if previous.receipt_high_watermark != block.receipt_high_watermark:
                    raise AMLError(
                        "binding_mismatch", "Source inventory changed its original cutoff",
                    )
            if block.base_ref is not None:
                base = self.read_logical(block.base_ref, "decision_set")
                raw = self.get_row("artifact", block.base_ref)
                if not isinstance(base, type(block)) or not base.is_head or (
                    base.owner_run_id == block.owner_run_id
                    or base.total_record_count != block.base_count
                    or base.receipt_high_watermark > block.receipt_high_watermark
                    or base.freeze_time > block.freeze_time
                    or raw is None
                    or raw["created_by_run_id"] != base.owner_run_id
                    or (raw["tenant_id"], raw["scope_kind"], raw["scope_id"])
                    != (self.tenant_id, scope.scope_kind, scope.scope_id)
                ):
                    raise AMLError("binding_mismatch", "Inventory reused another source snapshot")
            reference = self.put_logical_artifact(
                ticket, "decision_set", block, created_at=block.freeze_time,
            )
            check()
            return reference

    def scope_quarantined(self, scope_id: str) -> bool:
        """Return whether durable quarantine bars all work on this exact scope."""
        return self.get_control(f"scope_quarantine:{scope_id}") is not None

    def quarantine_scope(
        self, scope: ScopeV1, owner_run_id: str, error_code: str, *, clock: Callable[[], int],
    ) -> ControlRowV1:
        """Fence one refused cycle (``owner_lease.quarantine_scope``)."""
        from aml.store.owner_lease import quarantine_scope

        return quarantine_scope(self, scope, owner_run_id, error_code, clock=clock)

    def recover_released_finalizes(self, *, clock: Callable[[], int]) -> None:
        """Quarantine every scope with finalize history (``owner_lease``)."""
        from aml.store.owner_lease import recover_released_finalizes

        recover_released_finalizes(self, clock=clock)

    def _admit_run(
        self, row: StrictModel, *, privacy: bool = False, finalize: bool = False,
    ) -> WriteTicket:
        """Admit one new Run family root as the first write of its identity.

        The strict parse and the shape checks that need no storage run before
        the commit mutex; the mutex-held section admits the ticket, proves the
        family and identity are new, prepares the RowOps and writes them.
        """
        from aml.store.indexed import IndexedReader

        run = self._parse_row("run", row.model_dump(mode="json"))
        data = run.model_dump(mode="json")
        if data["run_kind"] in {"forget", "erasure_controller"} or (
            data["run_kind"] == "retention" and data["target_kind"] == "scope"
        ):
            raise AMLError("invalid_write_mode", "Erasure requires its atomic owner admission")
        if data["run_kind"] == "finalize" and not finalize:
            raise AMLError("invalid_write_mode", "Finalize requires its dedicated admission")
        if (
            data["run_kind"] == "retention"
            and data["target_kind"] == "affected_set"
            and not privacy
        ):
            raise AMLError(
                "invalid_write_mode", "Ingestion cleanup needs its bound dedicated admission"
            )
        if (
            data["status"] != "queued"
            or data["stage"] != "await_input"
            or any(
                data.get(field) is not None
                for field in ("input_ref", "result_ref", "commit_marker", "commit_seq")
            )
        ):
            raise AMLError(
                "invalid_run_admission", "New Run must enter the empty queued admission state"
            )
        ticket = WriteTicket(
            data["writer_epoch"],
            None,
            data.get("scope_kind"),
            data.get("scope_id"),
            data.get("admitted_erase_epoch"),
            data.get("admitted_consent_epoch"),
        )
        with self._write_scope():
            self._admit(ticket)
            finalize_owner = None
            if data.get("parent_run_id") is not None:
                finalize_owner = self._execution_fences(data)
            existing = IndexedReader(
                self._ws,
                self.entities,
                tenant_id=self.tenant_id,
            ).latest_run(
                data["run_kind"], data["idempotency_key"]
            )
            if existing is not None:
                if existing["admission_digest"] != data["admission_digest"]:
                    raise AMLError(
                        "operation_conflict", "Run family has a different admission binding"
                    )
                if existing["run_generation"] >= data["run_generation"]:
                    raise AMLError("operation_conflict", "Run generation already has an identity")
                raise AMLError(
                    "invalid_run_admission", "Successor admission requires recovery proof"
                )
            if data["run_generation"] != 0:
                raise AMLError(
                    "invalid_run_admission", "A new family must begin at generation zero"
                )
            if self.get_row("run", data["object_id"]) is not None:
                raise AMLError("operation_conflict", "Run object identity already exists")
            _, plan = self._prepare_rows([RowWrite("run", run)], ticket)
            self._write_plan(plan)
            return self._register_finalize_write_ticket(
                replace(ticket, run_id=data["object_id"]), finalize_owner,
            )

    def adopt_bootstrap_run(self, run_id: str, now: int) -> WriteTicket:
        """Adopt only an unfinished global bootstrap Run after owner epoch rollover."""
        with self._write_scope():
            run = self.get_row("run", run_id)
            lease = self.get_control("lease:writer")
            if run is None or lease is None:
                raise AMLError("service_unhealthy", "Bootstrap adoption authority is missing")
            epoch = lease["value"]["writer_epoch"]
            if run["run_kind"] not in {"policy_release", "embedding_switch"} or (
                run["commit_mode"] != "cas_only"
                or run["target_kind"] != "none"
                or run.get("scope_id") is not None
                or run["status"] not in {"queued", "running"}
                or run.get("commit_marker") is not None
                or run.get("commit_seq") is not None
                or run["writer_epoch"] > epoch
            ):
                raise AMLError(
                    "lease_lost", "Run is outside the bootstrap epoch adoption whitelist"
                )
            if run.get("input_ref") is not None:
                payload = self.read_logical(run["input_ref"], "run_input").model_dump(mode="json")
                if (
                    payload["run_kind"] != run["run_kind"]
                    or payload["business_key"]["action"] != "bootstrap"
                ):
                    raise AMLError(
                        "parity_mismatch", "Adopted input is not the same bootstrap operation"
                    )
            if run["writer_epoch"] != epoch:
                run["writer_epoch"], run["updated_at"] = epoch, now
                _, plan = self._prepare_rows([RowWrite("run", self._parse_row("run", run))], None)
                self._write_plan(plan)
            return self.ticket(run_id)

    def adopt_add_run(self, run_id: str, now: int) -> WriteTicket:
        """Validate an already adopted empty Add when its client payload returns.

        Args:
            run_id: Exact existing family identity selected under its session mutex.
            now: Trusted current time; validation never changes stored timestamps.

        Returns:
            A current-epoch ticket without mutation or orphan attachment.

        Raises:
            AMLError: If startup adoption is incomplete, the proof is stale, or
                the Run has unsupported input, execution effects, or an old epoch.
        """
        if self._add_boot_recovery_open:
            raise AMLError("service_unhealthy", "Startup Add recovery has not completed")
        run, generation = self._inputless_add_proof(run_id)
        with self._write_scope():
            self._recheck_inputless_add(run, generation)
            if run["status"] != "queued":
                raise AMLError("run_terminal", "Terminal Add requires a new request identity")
            return self.ticket(run_id)

    def read_received_add_input(self, run_id: str) -> SessionAdmissionRunInputV1 | None:
        """Read one proved, saved Add for its existing scheduler to reconstruct.

        The inputless proof checks its original admission family, source binding
        and absence of accepted effects. The final generation check rejects a
        concurrent commit or mutation. This read grants no execution authority;
        the Add service must still resolve current authority and reuse the same
        Run, session lock and ordinary commit path.

        Args:
            run_id: Original ordinary Add Run selected from the durable queue.

        Returns:
            Its exact saved admission, or None if no source is saved or the Run
            already became ordered or terminal.

        Raises:
            AMLError: For an absent or unsupported Run, ambiguous saved input,
                invalid recovery proof, changed generation or stale Writer epoch.
        """
        from aml.core.run_inputs import SessionAdmissionRunInputV1

        raw = self.get_row("run", run_id)
        if raw is None:
            raise AMLError("reference_missing", "Received Add Run does not exist")
        run = self._parse_row("run", raw).model_dump(mode="json")
        if (run["run_kind"], run["commit_mode"], run["scope_kind"], run["target_kind"]) != (
            "session_ingestion", "orchestrator", "user", "session",
        ):
            raise AMLError("unsupported_variant", "Received source requires an ordinary Add")
        with self.session_mutex(run["scope_id"], run["target_id"]):
            current = self.get_row("run", run_id)
            if current is None:
                raise AMLError("reference_missing", "Received Add Run does not exist")
            if current["input_ref"] is not None or current["status"] != "queued":
                return None
            run, generation = self._inputless_add_proof(run_id)
            saved = [body for _, body in self.find_logical_candidates(run_id, "run_input")
                     if isinstance(body, SessionAdmissionRunInputV1)]
            if len(saved) > 1:
                raise AMLError("binding_mismatch", "Received Add has ambiguous source inputs")
            with self.mutex:
                self._recheck_inputless_add(run, generation)
            return saved[0] if saved else None

    def _inputless_add_proof(self, run_id: str) -> tuple[dict[str, Any], int]:
        """Treat concurrent scan changes as retry signals, including apparent corruption."""
        with self.mutex:
            generation = self.ref_generation
        try:
            return self._scan_inputless_add(run_id)
        except Exception as error:
            with self.mutex:
                if generation != self.ref_generation:
                    raise AMLError(
                        "reception_stale", "Inputless proof changed during its scan"
                    ) from None
            if isinstance(error, AMLError):
                raise
            raise AMLError("service_unhealthy", "Inputless Add proof cannot be verified") from None

    def _scan_inputless_add(self, run_id: str) -> tuple[dict[str, Any], int]:
        """Inspect AbandonExemptV1 outside the commit mutex, without adopting inputs."""
        from aml.core.artifacts import ManifestV1, RequestLedgerV1
        from aml.core.identity import base64url
        from aml.core.query import ScopeV1
        from aml.core.run_inputs import SessionAdmissionRunInputV1, SessionDurableRunInputV1
        from aml.core.technical_types import RequestProgressV1
        from aml.ingestion.receipts import inspect_ledger
        from aml.store.indexed import ArtifactRows, EvidenceRows, IndexedReader

        with self.mutex:
            generation = self.ref_generation
            raw = self.get_row("run", run_id)
        if raw is None:
            raise AMLError("reference_missing", "Inputless Add Run does not exist")
        run = self._parse_row("run", raw).model_dump(mode="json")
        if (
            run["run_kind"] != "session_ingestion"
            or run["commit_mode"] != "orchestrator"
            or run["target_kind"] != "session"
            or run["scope_kind"] != "user"
            or run["input_ref"] is not None
        ):
            raise AMLError("unsupported_variant", "Only ordinary inputless Add recovery is open")
        if (
            run["stage"] != "await_input"
            or run["attempt"] != 0
            or any(
                run[key] is not None
                for key in (
                    "result_ref",
                    "auth_context_ref",
                    "commit_marker",
                    "commit_seq",
                    "started_at",
                )
            )
            or run["status"] not in {"queued", "failed"}
            or run["status"] == "failed"
            and run["failure_code"] not in {"input_abandoned", "erase_fenced", "consent_fenced"}
        ):
            raise AMLError("service_unhealthy", "Inputless Add has execution state")
        if run["run_generation"] == 0 and run["parent_run_id"] is not None:
            raise AMLError("unsupported_variant", "Internal dispatch recovery is not open")
        if run["run_generation"] > 0:
            self._execution_fences(run)
        reader = IndexedReader(
            self._ws,
            self.entities,
            tenant_id=self.tenant_id,
        )
        if reader.run_effects(run_id):
            raise AMLError("service_unhealthy", "Inputless Add has child or domain effects")
        after = None
        parts: set[str] = set()
        protected: set[str] = set()
        family_key: tuple[str, str] | None = None
        while page := reader.page(ArtifactRows(created_by_run_id=run_id), after=after):
            after = page[-1]["object_id"]
            for artifact in page:
                if (
                    artifact["tenant_id"] != self.tenant_id
                    or artifact["scope_kind"] != run["scope_kind"]
                    or artifact["scope_id"] != run["scope_id"]
                ):
                    raise AMLError("service_unhealthy", "Inputless artifact owner envelope differs")
                kind = artifact["artifact_kind"]
                if kind == "part":
                    parts.add(artifact["object_id"])
                    continue
                if kind == "manifest":
                    manifest = ManifestV1.model_validate_json(artifact["body"])
                    kind = manifest.logical_kind
                    protected.update(manifest.parts)
                if kind not in {"run_input", "request_ledger"}:
                    raise AMLError("service_unhealthy", "Inputless Add has nonexempt artifacts")
                body = self.read_logical(artifact["object_id"], kind)
                if kind == "run_input":
                    if not isinstance(body, SessionAdmissionRunInputV1 | SessionDurableRunInputV1):
                        raise AMLError(
                            "unsupported_variant", "Intent or dispatcher recovery is not open"
                        )
                    if isinstance(body, SessionDurableRunInputV1) and (
                        body.confirmed_request is not None
                    ):
                        raise AMLError("unsupported_variant", "Confirmed recovery is not open")
                    if isinstance(body, SessionAdmissionRunInputV1) and (
                        body.received_at != run["created_at"]
                        or body.work_deadline_at is None
                        or body.payload_digest != self.bind_content(
                            "run-input-payload", body.request_payload,
                        )
                    ):
                        raise AMLError(
                            "binding_mismatch", "Saved reception input changed its payload",
                        )
                    family = {
                        "scope_kind": run["scope_kind"],
                        "scope_id": run["scope_id"],
                        "business_key": body.business_key.model_dump(mode="json"),
                    }
                    identity = "idem_" + base64url(
                        hashlib.sha256(canonical_json(family).encode()).digest()
                    )
                    session = base64url(
                        hashlib.sha256(body.business_key.session_id.encode()).digest()
                    )[:22]
                    if (
                        body.admission_digest != run["admission_digest"]
                        or identity != run["idempotency_key"]
                        or session != run["target_id"]
                    ):
                        raise AMLError(
                            "service_unhealthy", "Orphan receipt belongs to another admission"
                        )
                    family_key = (body.business_key.session_id, body.business_key.request_id)
        if parts - protected:
            raise AMLError("service_unhealthy", "Inputless Add has ambiguous transport parts")
        # The canonical empty session cursor, the one exempt Control, left
        # with PD:P2 at S0b.
        if reader.control_provenance(run_id, limit=1):
            raise AMLError("service_unhealthy", "Inputless Add has nonexempt Control effects")
        if family_key is not None:
            # The saved input's business key is bound to this Run's idempotency key
            # above, and the family identity is an injective digest of the scope,
            # session_id and request_id: exactly the Evidence of that one request
            # shares it. Its coordinates answer without decoding the scope's other
            # Evidence, whose full scan made every Add linear in its user's history.
            scope = ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"])
            if reader.session_source_coordinates(scope, *family_key, limit=1):
                raise AMLError("service_unhealthy", "Inputless family already has Evidence")
        query = EvidenceRows.model_validate(
            {"scope_kind": run["scope_kind"], "scope_id": run["scope_id"]}
        )
        after = None
        while family_key is None and (page := reader.page(query, after=after)):
            after = page[-1]["object_id"]
            for evidence in page:
                family = {
                    "scope_kind": run["scope_kind"],
                    "scope_id": run["scope_id"],
                    "business_key": {
                        "session_id": evidence["session_id"],
                        "request_id": evidence["request_id"],
                    },
                }
                identity = "idem_" + base64url(
                    hashlib.sha256(canonical_json(family).encode()).digest()
                )
                if identity == run["idempotency_key"]:
                    raise AMLError("service_unhealthy", "Inputless family already has Evidence")
        ledger = self.get_control(f"request_ledger:{run['scope_id']}:{run['target_id']}")
        if ledger is not None:
            value = RequestLedgerValueV1.model_validate(ledger["value"])

            def read(reference: ArtifactId) -> RequestLedgerV1:
                body = self.read_logical(reference, "request_ledger")
                if not isinstance(body, RequestLedgerV1):
                    raise AMLError("service_unhealthy", "Request ledger kind differs")
                return body

            def observe(_key: ContentBindingDigest, progress: RequestProgressV1) -> None:
                if progress.admission_digest == run["admission_digest"]:
                    raise AMLError(
                        "service_unhealthy", "Inputless family already has reception binding"
                    )

            # The lookup result is deliberately unused; the observer proves
            # absence by admission while the shared owner verifies every block.
            inspect_ledger(value, run["admission_digest"], read, self.bind_content, observe=observe)
        return run, generation

    def _recheck_inputless_add(self, run: dict[str, Any], generation: int) -> None:
        """Recheck constant-size proof identities under the existing commit mutex."""
        if self.ref_generation != generation or self.get_row("run", run["object_id"]) != run:
            raise AMLError("reception_stale", "Inputless recovery proof changed during its scan")
        lease = self.get_control("lease:writer")
        if lease is None or run["writer_epoch"] != lease["value"]["writer_epoch"]:
            raise AMLError("lease_lost", "Inputless Run has not been adopted by this startup")

    def recover_privacy_admissions(self) -> tuple[PendingPrivacy, ...]:
        """Prove and adopt unresolved ingestion cleanup before ordinary startup recovery.

        Returns:
            Read-only reservations for attached resume, matching-client replay,
            or explicit denied-source reconstruction. Both semantic ancestors
            of denied cleanup are reserved and adopted as the exact same batch.
            Only writer_epoch is changed; timestamps, inputs, fences and effects
            remain byte-identical. Parents reserved here are never abandoned by
            the ordinary inputless Add path.

        Raises:
            AMLError: If called after startup, a prefix/zero-effect proof fails,
                or an owner has changed scope epochs or invalid process authority.
        """
        from aml.ingestion.privacy_recovery import classify_pending_privacy

        if not self._privacy_boot_recovery_open:
            raise AMLError("invalid_write_mode", "Startup privacy adoption has already closed")
        records = classify_pending_privacy(self)
        with self._write_scope():
            if records and any(record.ref_generation != self.ref_generation for record in records):
                raise AMLError("reception_stale", "Privacy startup proof changed before adoption")
            lease = self.get_control("lease:writer")
            if lease is None:
                raise AMLError("lease_lost", "Privacy startup has no exclusive Writer")
            epoch = lease["value"]["writer_epoch"]
            reserved: set[str] = set()
            for record in records:
                identities = [record.owner_run_id]
                if record.parent_run_id is not None:
                    reserved.add(record.parent_run_id)
                    identities.append(record.parent_run_id)
                rows: list[RowWrite] = []
                for identity in identities:
                    run = self.get_row("run", identity)
                    if run is None or run["writer_epoch"] > epoch or (
                        run["scope_kind"] != record.scope_kind or run["scope_id"] != record.scope_id
                    ):
                        raise AMLError("lease_lost", "Privacy startup has an invalid owner epoch")
                    if (
                        self.get_control(f"tomb:{record.scope_id}") is not None
                        or self.counter(f"erase_epoch:{record.scope_id}")
                        != run["admitted_erase_epoch"] + record.erase_epoch_bumps
                        or self.counter(f"consent_epoch:{record.scope_id}")
                        != run["admitted_consent_epoch"]
                    ):
                        raise AMLError(
                            "service_unhealthy", "Privacy recovery lost its admitted scope"
                        )
                    if run["writer_epoch"] != epoch:
                        run["writer_epoch"] = epoch
                        rows.append(RowWrite("run", self._parse_row("run", run)))
                if rows:
                    _, plan = self._prepare_rows(rows, None)
                    self._write_plan(plan)
            self._reserved_privacy_parents = frozenset(reserved)
            self._privacy_boot_recovery_open = False
        return records

    def recover_inputless_adds(self, now: int) -> tuple[str, ...]:
        """Adopt ordinary inputless Adds and restore accepted source before healthy.

        Args:
            now: Trusted startup time; accepted source does not expire with its client.

        Returns:
            Identities with retained source or awaiting source, without attaching
            orphan receipts or resuming a provider during bootstrap.

        Raises:
            AMLError: If called after startup, effects are not exempt, a proof
                changes, or intent/internal recovery would be required.

        Notes:
            Adoption changes only writer_epoch. Only an unsaved inputless root
            can subsequently expire. Accepted source retains its original
            identity, timestamps and queue position for autonomous processing.
        """
        from aml.store.indexed import IndexedReader

        if not self._add_boot_recovery_open:
            raise AMLError("invalid_write_mode", "Startup adoption has already closed")
        reader = IndexedReader(
            self._ws,
            self.entities,
            tenant_id=self.tenant_id,
        )
        after = None
        waiting: list[str] = []
        while page := reader.inputless_adds(after=after):
            after = page[-1]["object_id"]
            for row in page:
                if self.scope_quarantined(row["scope_id"]):
                    continue
                if row["object_id"] in self._reserved_privacy_parents:
                    continue
                with self.session_mutex(row["scope_id"], row["target_id"]):
                    run, generation = self._inputless_add_proof(row["object_id"])
                    with self._write_scope():
                        if (
                            self.ref_generation != generation
                            or self.get_row("run", row["object_id"]) != run
                        ):
                            raise AMLError("reception_stale", "Startup recovery proof changed")
                        lease = self.get_control("lease:writer")
                        if lease is None or run["writer_epoch"] > lease["value"]["writer_epoch"]:
                            raise AMLError("lease_lost", "Inputless Run has an invalid owner epoch")
                        if run["writer_epoch"] != lease["value"]["writer_epoch"]:
                            run["writer_epoch"] = lease["value"]["writer_epoch"]
                            _, plan = self._prepare_rows(
                                [RowWrite("run", self._parse_row("run", run))], None
                            )
                            self._write_plan(plan)
                    waiting.append(row["object_id"])
        # An inputless root with no saved source can expire after epoch-only
        # adoption. Accepted bytes survive any client or reception deadline.
        remaining = tuple(run_id for run_id in waiting if not self.abandon_add_run(run_id, now))
        from aml.core.run_inputs import MessagesRequestPayloadV1, SessionAdmissionRunInputV1
        from aml.ingestion.segment import message_chunk_index

        # Rebuild the durable backlog before new traffic can admit a younger
        # owner. The controller, rather than a returning client, drives it.
        saved: list[tuple[dict[str, Any], SessionAdmissionRunInputV1]] = []
        for run_id in remaining:
            run = self.get_row("run", run_id)
            assert run is not None
            for _, body in self.find_logical_candidates(run["object_id"], "run_input"):
                if isinstance(body, SessionAdmissionRunInputV1):
                    saved.append((run, body))

        def source_order(
            entry: tuple[dict[str, Any], SessionAdmissionRunInputV1],
        ) -> tuple[str, int, int, int, int, int, str]:
            run, received = entry
            payload = received.request_payload
            if not isinstance(payload, MessagesRequestPayloadV1):
                raise AMLError("binding_mismatch", "Saved Add requires message source")
            chunk = message_chunk_index(received.business_key.request_id)
            if chunk is not None:
                return (run["scope_id"], 0, chunk, 0, run["created_at"],
                        received.reception_sequence or 0, run["object_id"])
            times = [message.timestamp for message in payload.messages]
            source_time = min((value for value in times if value is not None), default=None)
            if any(value is None for value in times):
                source_time = None
            return (run["scope_id"], 1, int(source_time is None), source_time or 0,
                    run["created_at"], received.reception_sequence or 0, run["object_id"])

        for run, body in sorted(saved, key=source_order):
            try:
                self._register_received_add(run, body, now)
            except AMLError as error:
                if error.code != "delivery_busy":
                    raise
                # A bounded scheduling directory cannot refuse boot over a
                # larger backlog. Its prefix follows source order; discovery
                # refills from unchanged source as queue heads complete.
                continue
        # Before healthy: restore the ordered-receipt gap over the now
        # reconstructed saved backlog.
        self._restore_ordered_add_reservations()
        self._add_boot_recovery_open = False
        return remaining

    def _restore_ordered_add_reservations(self) -> None:
        """Restore the one unowned serving receipt in each scope during bootstrap.

        This startup-only scan reads bounded physical pages. Receipt-only SDK
        staging has no serving marker and keeps its independent batch behavior.
        An actual live owner is recovered through its lease, never replaced by
        this scheduling hint. Ordinary admission still proves all owner history.
        """
        from aml.core.controls import (
            CompletedPrefixV2,
            HeldFinalizeLeaseV1,
            PreparingFinalizeLeaseV1,
        )
        from aml.core.run_inputs import SessionDurableRunInputV1

        scopes: dict[str, tuple[tuple[int, int], int] | None] = {}
        restored: set[str] = set()
        for raw in self.iter_rows("run"):
            if (
                raw["run_kind"] != "session_ingestion" or raw["commit_mode"] != "orchestrator"
                or raw["scope_kind"] != "user" or raw["target_kind"] != "session"
                or raw["status"] != "succeeded" or raw["input_ref"] is None
                or raw["object_id"] in self._reserved_privacy_parents
            ):
                continue
            run = self._parse_row("run", raw).model_dump(mode="json")
            scope_id = run["scope_id"]
            if scope_id not in scopes:
                epochs = (self.counter(f"erase_epoch:{scope_id}"),
                          self.counter(f"consent_epoch:{scope_id}"))
                lease_row = self.get_control(f"lease:finalize:{scope_id}")
                lease = None if lease_row is None else ControlRowV1.model_validate(lease_row).value
                if self.scope_quarantined(scope_id) or isinstance(
                    lease, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1,
                ):
                    scopes[scope_id] = None
                else:
                    prefix_row = self.get_control(f"completed_prefix:{scope_id}")
                    cutoff = 0
                    if prefix_row is not None:
                        prefix = ControlRowV1.model_validate(prefix_row).value
                        if not isinstance(prefix, CompletedPrefixV2):
                            raise AMLError(
                                "binding_mismatch", "Completed prefix changed its contract",
                            )
                        if (prefix.erase_epoch, prefix.consent_epoch) == epochs:
                            cutoff = prefix.accepted_prefix_commit_seq
                    scopes[scope_id] = epochs, cutoff
            state = scopes[scope_id]
            if state is None or (
                run["admitted_erase_epoch"], run["admitted_consent_epoch"]
            ) != state[0]:
                continue
            witness = self.read_run_input(run["object_id"])
            if not isinstance(witness, SessionDurableRunInputV1):
                continue
            if witness.visibility_commit_seq <= state[1] or witness.work_deadline_at is None:
                continue
            if witness.received_at is None or run["stage"] != "settle" or (
                witness.visibility_commit_seq > self.counter("commit_seq")
            ):
                raise AMLError(
                    "binding_mismatch", "Ordered serving receipt lost its reception proof",
                )
            if scope_id in restored:
                raise AMLError("binding_mismatch", "Scope has several unowned serving receipts")
            self.add_admission.restore_ordered(
                scope_id, run["object_id"], witness.received_at,
                identity=(witness.business_key.session_id, witness.business_key.request_id),
            )
            restored.add(scope_id)

    def abandon_add_run(self, run_id: str, now: int) -> bool:
        """Void an unsaved ordinary Add root after its unused reception allowance.

        A proved saved admission is accepted durable work. Its client timeout
        never removes its source or spends its request identity. Only a root
        with no saved admission can end as ``input_abandoned`` here; ordinary
        privacy and authority rejection retain their separate deletion rules.

        Args:
            run_id: Exact external Add Run identity.
            now: Trusted current time in milliseconds; tests may supply a fixed clock.

        Returns:
            True once the request is void, ended by this call or an earlier one;
            False for retained saved input, before an unsaved root's allowance,
            or once it was ordered or another owner ended it.

        Raises:
            AMLError: ``scope_quarantined`` in a quarantined scope; others if
                effects, ownership, the saved input or the proof are invalid.
        """
        from aml.core.request_budget import DEFAULT_COMPLETION_WORK_BUDGET_MS
        from aml.core.run_inputs import SessionAdmissionRunInputV1

        raw = self.get_row("run", run_id)
        if raw is None:
            raise AMLError("reference_missing", "Add Run does not exist")
        with self.session_mutex(raw["scope_id"], raw["target_id"]):
            current = self.get_row("run", run_id)
            if current is not None and current["failure_code"] == "input_abandoned":
                terminal = self._parse_row("run", current).model_dump(mode="json")
                if (
                    terminal["run_kind"] != "session_ingestion"
                    or terminal["commit_mode"] != "orchestrator"
                    or terminal["stage"] != "await_input"
                    or terminal["status"] != "failed"
                    or terminal["attempt"] != 0
                    or any(
                        terminal[field] is not None
                        for field in (
                            "input_ref",
                            "result_ref",
                            "auth_context_ref",
                            "commit_marker",
                            "commit_seq",
                            "started_at",
                        )
                    )
                    or terminal["finished_at"]
                    < terminal["created_at"] + DEFAULT_COMPLETION_WORK_BUDGET_MS
                ):
                    raise AMLError("service_unhealthy", "Abandoned Add header is invalid")
                # Later generations can legitimately have durable effects. The
                # first abandonment proved its own zero-effect pre-state.
                return True
            if current is None:
                raise AMLError("reference_missing", "Add Run does not exist")
            if current["status"] != "queued" or (
                now < current["created_at"] + DEFAULT_COMPLETION_WORK_BUDGET_MS
            ):
                return False
            if self.scope_quarantined(current["scope_id"]):
                raise AMLError("scope_quarantined", "The scope requires explicit recovery repair")
            run, generation = self._inputless_add_proof(run_id)
            if run["status"] != "queued":
                raise AMLError("run_terminal", "Only queued inputless Add can expire")
            saved = [body for _, body in self.find_logical_candidates(run_id, "run_input")
                     if isinstance(body, SessionAdmissionRunInputV1)]
            if len(saved) > 1:
                raise AMLError("binding_mismatch", "Received Add has ambiguous source inputs")
            with self._write_scope():
                self._recheck_inputless_add(run, generation)
                if saved:
                    return False
                self._delete_received_inputs(run, None, {
                    **run, "status": "failed", "failure_code": "input_abandoned",
                    "updated_at": now, "finished_at": now,
                })
                return True

    def ticket(self, run_id: str) -> WriteTicket:
        """Capture durable Run fences before a provider or storage operation."""
        with self._write_scope():
            run = self.get_row("run", run_id)
            if run is None:
                raise AMLError("reference_missing", "Owning Run does not exist")
            ticket = WriteTicket(
                run["writer_epoch"],
                run_id,
                run.get("scope_kind"),
                run.get("scope_id"),
                run.get("admitted_erase_epoch"),
                run.get("admitted_consent_epoch"),
            )
            owner = self._admit(ticket, capture_finalize=True)
            self._register_finalize_write_ticket(ticket, owner)
            self._remember_artifact_preparation_admission(ticket, owner)
            return ticket

    def _clear_artifact_preparation_admissions(self) -> None:
        memo = cast(IdentityMemo[WriteTicket, _ArtifactPreparationAdmission] | None,
                    self.__dict__.get("_artifact_preparation_admissions"))
        if memo is not None:
            memo.clear()

    def _remember_artifact_preparation_admission(
        self, ticket: WriteTicket, owner: FinalizeLeaseTicket | None,
    ) -> None:
        """Remember only an admitted and invocation-registered finalize descendant.

        The tuple contains no body, source or authorization grant. It proves
        only the preparation's otherwise repeated admission, never a commit.
        """
        if owner is None or (digest := self._finalize_digests.get(owner)) is None:
            return
        memo = cast(IdentityMemo[WriteTicket, _ArtifactPreparationAdmission],
                    self.__dict__.setdefault("_artifact_preparation_admissions", IdentityMemo()))
        memo.remember(ticket, _ArtifactPreparationAdmission(
            self.mutex.sequence, self.storage_epoch, owner,
            (owner.scope_id, owner.lease_generation, owner.owner_run_id,
             owner.writer_epoch, owner.state), digest,
        ))

    def _artifact_preparation_admitted(self, ticket: WriteTicket) -> bool:
        """Reuse admission only for this ticket in the same unchanged Writer hold."""
        if not self.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            return False
        memo = cast(IdentityMemo[WriteTicket, _ArtifactPreparationAdmission] | None,
                    self.__dict__.get("_artifact_preparation_admissions"))
        proof = None if memo is None else memo.recall(ticket)
        if proof is None or (
            proof.section != self.mutex.sequence or proof.storage_epoch != self.storage_epoch
        ):
            return False
        owner = proof.owner
        return (
            self._finalize_write_tickets.recall(ticket) is owner
            and self._finalize_tickets.get(owner.owner_run_id) is owner
            and (owner.scope_id, owner.lease_generation, owner.owner_run_id,
                 owner.writer_epoch, owner.state) == proof.owner_identity
            and self._finalize_digests.get(owner) == proof.owner_digest
            and self._finalize_clocks.get(owner) is not None
        )

    def _admit(
        self, ticket: WriteTicket, *, capture_finalize: bool = False,
    ) -> FinalizeLeaseTicket | None:
        if ticket.scope_id is not None and self.scope_quarantined(ticket.scope_id):
            raise AMLError("scope_quarantined", "The scope requires explicit recovery repair")
        lease = self.get_control("lease:writer")
        if lease is None or lease["value"]["writer_epoch"] != ticket.writer_epoch:
            raise AMLError("lease_lost", "Writer epoch no longer matches the captured ticket")
        run = self.get_row("run", ticket.run_id) if ticket.run_id is not None else None
        if ticket.run_id is not None:
            if run is None or run["writer_epoch"] != ticket.writer_epoch:
                raise AMLError("lease_lost", "Run does not belong to the active Writer epoch")
            if (run.get("scope_kind"), run.get("scope_id")) != (ticket.scope_kind, ticket.scope_id):
                raise AMLError("scope_denied", "Run scope differs from its captured ticket")
            if run["status"] not in {"queued", "running"}:
                raise AMLError("run_terminal", "Terminal Run cannot acquire new write effects")
        if ticket.scope_id is None:
            if any(
                value is not None
                for value in (ticket.scope_kind, ticket.erase_epoch, ticket.consent_epoch)
            ):
                raise AMLError("scope_denied", "Global ticket contains scoped fences")
            return
        scope = ticket.scope_id
        registration = self.get_control(f"scope:{scope}")
        if registration is None or registration["value"]["scope_kind"] != ticket.scope_kind:
            raise AMLError("scope_denied", "Scoped writes require their exact registered scope")
        if self.get_control(f"tomb:{scope}") is not None:
            raise AMLError("scope_denied", "The target scope is tombstoned")
        pending, controls = self._admission_reads(scope)
        if len(pending) > 1:
            raise AMLError("service_unhealthy", "Scope has competing pending privacy owners")
        if pending and pending[0]["object_id"] != ticket.run_id:
            raise AMLError("index_rebuilding", "Pending privacy admission blocks ordinary writes")
        if len(controls) > 2:
            raise AMLError("service_unhealthy", "Scope has conflicting fence owners")
        maintenance_seen = False
        fenced_epoch: int | None = None
        own_bumps = 0
        for control in controls:
            key = control["key"]
            if key.startswith(f"erasure:{scope}:"):
                from aml.store.forget_run import owns_erasure_fence

                if run is None or not owns_erasure_fence(run, control):
                    raise AMLError(
                        "erasure_in_progress", "An erasure fence blocks ordinary writes",
                    )
                # The fence owner's worker was admitted before its own P1 bump;
                # the fence names the exact epoch it may write at (D.3 P1/P2).
                from aml.store.forget_run import fence_erase_epoch

                fenced_epoch = fence_erase_epoch(control)
                continue
            if key.startswith(f"maintenance:{scope}:"):
                if maintenance_seen or not self._owns_privacy_fence(run, control):
                    raise AMLError("index_rebuilding", "A maintenance fence blocks ordinary writes")
                maintenance_seen = True
                # The erase owner's own batches advanced erase_epoch (D.1, D.7.5).
                own_bumps = control["value"]["erase_epoch_bumps"]
        expected_erase = (
            fenced_epoch if fenced_epoch is not None
            else None if ticket.erase_epoch is None else ticket.erase_epoch + own_bumps
        )
        if self.counter(f"erase_epoch:{scope}") != expected_erase:
            raise AMLError("erase_fenced", "Erase epoch changed during the operation")
        if self.counter(f"consent_epoch:{scope}") != ticket.consent_epoch:
            raise AMLError("consent_fenced", "Consent epoch changed during the operation")
        if run is not None and (
            run.get("admitted_erase_epoch") != ticket.erase_epoch
            or run.get("admitted_consent_epoch") != ticket.consent_epoch
        ):
            raise AMLError("parity_mismatch", "Run admission epochs differ from its ticket")
        if run is not None:
            owner = self._execution_fences(run)
            if not capture_finalize and owner is not None and (
                self._finalize_write_tickets.recall(ticket) is not owner
            ):
                raise AMLError("lease_lost", "Finalize descendant lost its issued invocation")
            return owner

    def _extraction_release_ref(self) -> str:
        """The current extraction policy release: a privacy batch's revisions' release (A.3.2)."""
        pointer = self.get_control("policy:extraction")
        if pointer is None:
            raise AMLError("reference_missing", "Privacy compensation requires the policy pointer")
        parsed = self._parse_row("control", pointer)
        assert isinstance(parsed, ControlRowV1)
        if not isinstance(parsed.value, PolicyPointerV1):
            raise AMLError("binding_mismatch", "The extraction pointer has another value")
        return str(parsed.value.current_release_ref)

    def _owns_privacy_fence(self, run: dict[str, Any] | None, control: dict[str, Any]) -> bool:
        """Admit only the exact privacy-erase owner through its strict maintenance fence."""
        from aml.core.privacy_persistence import PrivacyEraseProgressV1

        if control["value"].get("state") != "active" or run is None or (
            run["run_kind"] != "retention"
            or run["commit_mode"] != "destructive_scan"
            or run["target_kind"] != "affected_set"
            or run["status"] != "running"
            or run["input_ref"] is None
        ):
            return False
        value = PrivacyEraseProgressV1.model_validate(control["value"])
        if value.owner_run_id != run["object_id"]:
            return False
        body = self._bound_input(run)
        if body.get("retention_role") not in {
            "split_secret", "source_expiry", "consent_withdrawal",
        }:
            return False
        affected = body["business_key"]["affected"]
        anchor = affected[0]["evidence_id"] if affected else None
        if value.evidence_id != anchor:
            raise AMLError("parity_mismatch", "Privacy fence lost its admitted Evidence anchor")
        expected = maintenance_fence_key(run["scope_id"], value.evidence_id, self.bind_content)
        if control["key"] != expected:
            raise AMLError("parity_mismatch", "Privacy fence key differs from its Evidence anchor")
        return True

    def read_run_input(self, run_id: str) -> StrictModel:
        """Read a Run's durable input after proving its ownership and admission.

        The Run and its input are read under the Writer mutex. This verifies
        the Artifact creator, Run kind and admission binding in addition to
        the logical content checks; it does not require an active publisher.

        Args:
            run_id (str): Identity of the retained Run whose input is required.

        Returns:
            StrictModel: The validated, typed durable input owned by the Run.

        Raises:
            AMLError: ``reference_missing`` if the Run is absent,
                ``input_required`` if it has no attached input, or a storage
                or binding error if the input cannot be proved.
        """
        from aml.core.run_inputs import parse_run_input

        with self.mutex:
            run = self.get_row("run", run_id)
            if run is None:
                raise AMLError("reference_missing", "Run input requires a retained Run")
            return parse_run_input(self._bound_input(run))

    def _bound_input(self, run: dict[str, Any]) -> dict[str, Any]:
        """Load the durable input and prove its owning admission before consumption.

        Every execution fence of a finalize owner proves its input again. For
        the owner the proof is served from the scan memo (ADR 0041) while the
        exact Run image and every row the proof read are unchanged; a proof
        that enumerated an index set is never memoized.
        """
        if run.get("input_ref") is None:
            raise AMLError("input_required", "Execution requires an attached durable input")
        if run.get("run_kind") != "finalize":
            return self._prove_bound_input(run)
        memo = self.scan_memo()
        memo_key = ("bound-input", canonical_json(run))
        memoized = memo.lookup(memo_key)
        if memoized is not MISSING:
            return copy_row(cast(dict[str, Any], memoized))
        token = self.storage_epoch
        with recording_reads() as record:
            body = self._prove_bound_input(run)
        if record.complete and body.get("worker_role") == "owner":
            # Held in the row cache's own storable form; each hit is a private
            # plain copy, as a row read is.
            stored, size = sized_copy(body)
            memo.store(memo_key, token, (MEMBERSHIP_ANY, *record.keys), stored, value_bytes=size)
        return body

    def _prove_bound_input(self, run: dict[str, Any]) -> dict[str, Any]:
        """Prove the durable input against its admission, reading it afresh."""
        envelope = self.get_row("artifact", run["input_ref"])
        if envelope is None or envelope["created_by_run_id"] != run["object_id"]:
            raise AMLError("binding_mismatch", "Run input has a different creator")
        typed = self.read_logical(run["input_ref"], "run_input")
        body = typed.model_dump(mode="json")
        if (
            body["run_kind"] != run["run_kind"]
            or body["admission_digest"] != run["admission_digest"]
        ):
            raise AMLError("binding_mismatch", "Run input differs from its admitted identity")
        # A label child's input is proved by its rebuilt engine (B.3a, ingest-Q3).
        if body.get("evaluation_kind") == "source_labels":
            from aml.core.run_inputs import SourceLabelsRunInputV1
            from aml.store.source_labels import validate_input as validate_source_labels

            if not isinstance(typed, SourceLabelsRunInputV1):
                raise AMLError("binding_mismatch", "Source-label input has another variant")
            validate_source_labels(self, run, typed)
        # A judge child's input is proved against its owner and snapshot (C.13, S4b).
        if body.get("evaluation_kind") == "judgment":
            from aml.core.judge_records import JudgmentRunInputV1
            from aml.store.judge_records import validate_input as validate_judgment

            if not isinstance(typed, JudgmentRunInputV1):
                raise AMLError("binding_mismatch", "Judgment input has another variant")
            validate_judgment(self, run, typed)
        if body["run_kind"] == "finalize" and body.get("worker_role") == "owner":
            from aml.core.admission import admission_identity, finalize_admission
            from aml.core.query import ScopeV1
            from aml.core.run_inputs import FinalizeOwnerRunInputV1

            if not isinstance(typed, FinalizeOwnerRunInputV1):
                raise AMLError("binding_mismatch", "Finalize input has another strict variant")
            admission = finalize_admission(
                principal_id=run["principal_id"],
                scope=ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"]),
                cycle=typed.business_key.finalize_cycle,
            )
            identity, digest, payload = admission_identity(admission, self.bind_content)
            heads = self.bind_content("finalize-input-heads", {
                "evidence_count": typed.evidence_count,
                "evidence_rows_digest": typed.evidence_rows_digest,
            })
            if (
                (run["idempotency_key"], run["admission_digest"], typed.payload_digest)
                != (identity, digest, payload)
                or typed.business_key.scope_id != run["scope_id"]
                or typed.input_heads_digest != heads
                or run["commit_mode"] != "orchestrator" or run["target_kind"] != "scope"
                or run["target_id"] != run["scope_id"] or run["parent_run_id"] is not None
                or run["run_generation"] != 0
                or run["auth_context_ref"] is not None
                or run["commit_marker"] is not None or run["commit_seq"] is not None
            ):
                raise AMLError("binding_mismatch", "Finalize input changed its exact admission")
            # An explicitly resumed owner (attempt > 0) keeps the input it froze and
            # selected; the checks above re-prove it (C.13 R08, ingest-Q17).
        return body

    def _execution_fences(self, run: dict[str, Any]) -> FinalizeLeaseTicket | None:
        """Enforce existing session ownership and immutable parent-attempt bindings."""
        from aml.store.input_fence import require_input_mutation_allowed

        current = run
        seen: set[str] = set()
        while current.get("parent_run_id") is not None:
            if current["object_id"] in seen:
                raise AMLError("reference_cycle", "Run ancestry contains a cycle")
            seen.add(current["object_id"])
            parent = self.get_row("run", current["parent_run_id"])
            if current["run_kind"] == "session_ingestion" and (
                current["commit_mode"] == "orchestrator" and current["run_generation"] > 0
            ):
                # A generation edge records a terminated predecessor, not an
                # executing coordinator. Its snapshot locator may later be GC'd.
                if parent is not None and (
                    any(
                        parent[key] != current[key]
                        for key in (
                            "run_kind",
                            "commit_mode",
                            "idempotency_key",
                            "admission_digest",
                            "scope_kind",
                            "scope_id",
                            "principal_id",
                            "target_kind",
                            "target_id",
                        )
                    )
                    or parent["run_generation"] != current["run_generation"] - 1
                    or parent["status"] != "failed"
                    or parent["failure_code"]
                    not in {"input_abandoned", "erase_fenced", "consent_fenced"}
                    or parent["input_ref"] is not None
                ):
                    raise AMLError(
                        "unsupported_variant", "Durable or invalid generation lineage is not open"
                    )
                break
            if parent is None or parent["status"] not in {"queued", "running"}:
                raise AMLError("lease_lost", "Executing child lost its nonterminal parent")
            if parent["writer_epoch"] != run["writer_epoch"]:
                raise AMLError("lease_lost", "Child and parent Writer epochs differ")
            if current.get("input_ref") is not None:
                body = self._bound_input(current)
                if "parent_attempt" in body and body["parent_attempt"] != parent["attempt"]:
                    # The owner's one label child continues across owner attempts,
                    # its spent sends unchanged (ingest-Q-S2a-1). The reextract
                    # continuation went with its engine at S0b; nothing else continues.
                    if body.get("evaluation_kind") != "source_labels":
                        raise AMLError(
                            "lease_lost",
                            "Old sibling input has no current continuation authority",
                        )
                    from aml.store.source_labels import validate_continuation

                    validate_continuation(self, current, parent)
            current = parent
        if (
            current["run_kind"] == "session_ingestion"
            and current["commit_mode"] == "orchestrator"
            and current["run_generation"] > 0
            and current.get("parent_run_id") is None
        ):
            raise AMLError("service_unhealthy", "Successor Run lost its predecessor locator")
        scope = run.get("scope_id")
        finalize = self.get_control(f"lease:finalize:{scope}")
        if finalize is not None and finalize["value"]["state"] == "released":
            # Release and owner settlement normally share one critical section.
            # A failed boundary must not admit input in the interval between them.
            require_input_mutation_allowed(
                finalize, run,
                finalize_owner=self.get_row("run", finalize["value"]["owner_run_id"]),
            )
        if current["run_kind"] == "finalize":
            if finalize is None or finalize["value"]["state"] != "held":
                raise AMLError("lease_lost", "Finalize execution requires a held owner lease")
            issued = self._finalize_tickets.get(current["object_id"])
            if issued is None:
                raise AMLError("lease_lost", "Finalize descendant has no live issued owner")
            self._finalize_precondition(issued)
            acting = self.__dict__.get("_acting_owners")
            if acting is not None and self.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
                acting[issued.owner_run_id] = issued
            value = finalize["value"]
            require_input_mutation_allowed(finalize, run, execution_root=current)
            if run.get("input_ref") is not None:
                body = self._bound_input(run)
                if (
                    "lease_generation" in body
                    and body["lease_generation"] != value["lease_generation"]
                ):
                    raise AMLError("lease_lost", "Finalize input belongs to another cycle")
            return issued
        elif current["run_kind"] == "session_ingestion":
            # A reception root is only bookkeeping until its ordering commit, and
            # split-secret cleanup is its only executing child (PD:P2, B.2).
            if run is current and run["stage"] == "await_input":
                return
            if finalize is not None and finalize["value"]["state"] in {"preparing", "held"}:
                raise AMLError("lease_lost", "Finalize ownership blocks ordinary extraction")
            if (
                run["run_kind"] == "retention"
                and run["commit_mode"] == "destructive_scan"
                and run["target_kind"] == "affected_set"
                and run["parent_run_id"] == current["object_id"]
                and current["status"] == "queued"
                and current["stage"] == "await_input"
                and current["input_ref"] is None
                and current["attempt"] == 0
                and (run["principal_id"], run["scope_kind"], run["scope_id"])
                == (current["principal_id"], current["scope_kind"], current["scope_id"])
            ):
                # The only opened affected-set retention producer is split-secret
                # cleanup. Late reception recognition precedes extraction and
                # must not fabricate a session extraction lease.
                if run["input_ref"] is not None:
                    from aml.core.admission import split_secret_parent_binding
                    from aml.core.run_inputs import SplitSecretChildRunInputV1

                    selected = self.read_logical(run["input_ref"], "run_input")
                    if not isinstance(
                        selected, SplitSecretChildRunInputV1
                    ) or selected.business_key.parent_binding_digest != split_secret_parent_binding(
                        current, self.bind_content
                    ):
                        raise AMLError("binding_mismatch", "Reception cleanup lost its parent")
                return
            # The session coordinator lease went with the extraction cursor
            # (PD:P2): nothing else executes under a reception root.
            raise AMLError("lease_lost", "A reception root executes no work but its cleanup")

    def _parse_row(self, kind: str, raw: object) -> StrictModel:
        try:
            if kind == "artifact":
                return ArtifactRowV1.model_validate(raw)
            if kind == "control":
                data = dict(cast(dict[str, Any], raw))
                data["value"] = parse_control_value(data["key"], data["value"])
                return ControlRowV1.model_validate(data)
            adapter = DOMAIN_ADAPTERS.get(kind)
            if adapter is None:
                raise AMLError("unsupported_variant", "Unknown domain schema")
            data = cast(dict[str, Any], raw)
            if kind == "run" and data.get("run_kind") not in {
                "session_ingestion",
                "finalize",
                "publish",
                "evaluate",
                "retention",
                "policy_release",
                "embedding_switch",
                "forget",
                "erasure_controller",
                "entity_update",
            }:
                raise AMLError("unsupported_variant", "Run kind has no open producer")
            if kind == "run" and data.get("target_kind") in {"policy", "run"}:
                raise AMLError(
                    "unsupported_variant", "Run target producer is outside the M0 schema set"
                )
            if kind == "recall" and data.get("record_class") in {"interaction", "telemetry"}:
                raise AMLError("unsupported_variant", "Recall class is outside the M0 schema set")
            return cast(StrictModel, adapter.validate_python(raw))
        except ValidationError as error:
            raise AMLError("schema_invalid", "Row violates its strict schema") from error

    def _lookup(
        self,
        kind: str,
        object_id: str,
        post: dict[tuple[str, str], StrictModel],
        *,
        pre_only: bool = False,
    ) -> dict[str, Any] | None:
        model = None if pre_only else post.get((kind, object_id))
        return model.model_dump(mode="json") if model is not None else self.get_row(kind, object_id)

    def _carried_verdict(
        self, kind: str, row: Mapping[str, Any], verdict_ref: str,
        post: dict[tuple[str, str], StrictModel],
    ) -> bool:
        """Whether a new row carries a verdict reference its Item's stored rows already name.

        A privacy erase deletes the judge records that depend on erased
        Evidence (C.13), while every version and link naming one keeps its
        ``verdict_ref`` (VERDICT_REF ``retain_locator``). A later version that
        copies a stored version's content (a requote, D.7.3) or a link it
        carries over (A.3.3) names the same verdict, so that one reference may
        no longer resolve. A reference naming no stored row of the same Item is
        still required to exist.
        """
        from aml.store.indexed import RevisionLinks, RevisionRows

        if kind == "revision":
            memory_id = row["memory_id"]
        elif kind == "link":
            source = self._reference_lookup("revision", row["source_revision_id"], post)
            if source is None:
                return False
            memory_id = source["memory_id"]
        else:
            return False
        reader = self._reference_reader()
        after: str | None = None
        while True:
            page = reader.page(RevisionRows(memory_id=memory_id), after=after, limit=512)
            for version in page:
                if version["snapshot"]["verdict_ref"] == verdict_ref or any(
                    link["verdict_ref"] == verdict_ref for link in reader.page(
                        RevisionLinks(source_revision_id=version["object_id"]), limit=512,
                    )
                ):
                    return True
            if len(page) < 512:
                return False
            after = page[-1]["object_id"]

    def _reference_lookup(
        self, kind: str, object_id: str, post: dict[tuple[str, str], StrictModel], *,
        pre_only: bool = False, evidence: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> dict[str, Any] | None:
        """Resolve reference identity/scope without reading persisted Evidence content.

        Strict materialized post-images retain the same union-view precedence.
        A persisted Evidence locator needs only its actual entity, identity,
        tenant and scope; eligibility and semantic projection use their own full
        readers. Erased or expired Evidence remains a valid locator. A body's
        validation reads its Evidence locators together (``evidence``).
        """
        model = None if pre_only else post.get((kind, object_id))
        if model is not None:
            return model.model_dump(mode="json")
        if kind == "evidence":
            if evidence is not None:
                return evidence(object_id)
            return self._reference_reader().evidence_reference_header(object_id)
        return self.get_row(kind, object_id)

    def _reference_reader(self) -> IndexedReader:
        from aml.store.indexed import IndexedReader

        return IndexedReader(self._ws, self.entities, tenant_id=self.tenant_id)

    def _scope(
        self,
        kind: str,
        row: dict[str, Any],
        post: dict[tuple[str, str], StrictModel],
        visiting: frozenset[tuple[str, str]] = frozenset(),
    ) -> tuple[str, str] | None:
        key = (kind, row["object_id"])
        if key in visiting:
            raise AMLError("reference_cycle", "Effective scope contains a cycle")
        visiting = visiting | {key}
        if kind in _DIRECT_SCOPE_KINDS:
            scope_kind, scope_id = row.get("scope_kind"), row.get("scope_id")
            if (scope_kind is None) != (scope_id is None):
                raise AMLError("scope_denied", "Scope kind and identity must appear together")
            return (cast(str, scope_kind), cast(str, scope_id)) if scope_kind is not None else None
        if kind == "control":
            scope_id = control_scope_id(row["key"])
            if scope_id is None:
                return None
            registration = self._lookup("control", control_id(f"scope:{scope_id}"), post)
            if registration is None:
                raise AMLError("reference_missing", "Scope registration does not exist")
            return registration["value"]["scope_kind"], scope_id
        target_kind, target_id = {
            "revision": ("item", row.get("memory_id")),
        }.get(kind, (None, None))
        if target_kind is None or target_id is None:
            raise AMLError("schema_invalid", "Effective scope has no declared authority")
        target = self._lookup(target_kind, target_id, post)
        if target is None:
            raise AMLError("reference_missing", "Effective scope authority does not exist")
        return self._scope(target_kind, target, post, visiting)

    @staticmethod
    def _projection_refs(references: Sequence[Reference]) -> list[ReferenceProjectionV1]:
        unique = sorted({(ref.spec.target_kind, ref.target_id) for ref in references})
        return [
            _REF_ADAPTER.validate_python({"kind": kind, "object_id": identifier})
            for kind, identifier in unique
        ]

    def _validate_refs(
        self,
        kind: str,
        model: StrictModel,
        post: dict[tuple[str, str], StrictModel],
        *, _references: tuple[Reference, ...] | None = None,
        _evidence: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> None:
        row = model.model_dump(mode="json")
        owner_scope = self._scope(kind, row, post)
        previous = self.get_row(kind, row["object_id"])
        old_refs: set[tuple[str, str]] = set()
        if previous is not None:
            marker_diagnosis = (
                kind == "run"
                and previous["run_kind"] == "evaluate"
                and previous["target_kind"] == "candidate"
                and previous["commit_mode"] == "single_compensatable"
                and previous["parent_run_id"] is not None
                and previous["input_ref"] is not None
                and previous["status"] == "running"
                and previous["failure_code"] is None
                and previous["started_at"] is not None
                and previous["finished_at"] is None
                and (previous["commit_marker"] is None) != (previous["commit_seq"] is None)
                and row["status"] == "manual_attention"
                and row["failure_code"] == "parity_mismatch"
                and set(previous) == set(row)
                and all(previous[key] == value for key, value in row.items()
                        if key not in {"status", "failure_code", "updated_at"})
            )
            # Only this exact diagnostic preserves every reference and corrupt
            # marker byte while changing three reference-free bookkeeping
            # fields. The strict postrow proves their unchanged field types;
            # the invalid running pre-state cannot itself be parsed (§5.6).
            # All other old rows still require the ordinary strict parser.
            old_model = model if marker_diagnosis else self._parse_row(kind, previous)
            old_refs = {(ref.path, ref.target_id) for ref in extract_references(old_model)}
        references = contextual_references(model) if _references is None else _references
        for ref in references:
            spec = ref.spec
            if ref.target_id in self._sizing_refs:
                continue  # A never-written sizing placeholder (commit_rowops only).
            target = self._reference_lookup(
                spec.target_kind, ref.target_id, post, pre_only=spec.fulfillment == "pre_state",
                evidence=_evidence,
            )
            if target is None:
                if (
                    spec.ref_mode == "historical_locator"
                    and spec.existence_expectation != "must_exist"
                ):
                    continue
                if spec.ref_mode == "snapshot_locator" and (ref.path, ref.target_id) in old_refs:
                    continue
                if spec == VERDICT_REF and self._carried_verdict(kind, row, ref.target_id, post):
                    continue
                if kind == "candidate":
                    from aml.store.extraction_erasure import erased_candidate_reference

                    if erased_candidate_reference(self, row, ref):
                        continue
                raise AMLError("reference_missing", "Required typed reference does not exist")
            if target.get("tenant_id", self.tenant_id) != self.tenant_id:
                raise AMLError("tenant_mismatch", "Reference target belongs to another tenant")
            target_scope = self._scope(spec.target_kind, target, post)
            if spec.scope_rule == "global" and target_scope is not None:
                raise AMLError("scope_denied", "Reference requires a global target")
            if spec.scope_rule == "same_scope" and target_scope != owner_scope:
                raise AMLError("scope_denied", "Reference crosses an exact scope boundary")
            if spec.scope_rule == "same_scope_or_global" and target_scope not in (
                None,
                owner_scope,
            ):
                raise AMLError("scope_denied", "Reference crosses a scope boundary")
            if spec.target_kind == "artifact" and spec.artifact_kinds:
                actual_kind = target["artifact_kind"]
                if actual_kind == "manifest":
                    body = json.loads(target["body"])
                    actual_kind = body["logical_kind"]
                    self._read_logical_post(ref.target_id, actual_kind, post)
                if actual_kind not in spec.artifact_kinds:
                    raise AMLError(
                        "reference_kind_mismatch", "Artifact kind is outside the allowlist"
                    )
            if kind == "run" and ref.path == "parent_run_id":
                if ref.target_id == row["object_id"] or target["created_at"] > row["created_at"]:
                    raise AMLError("reference_cycle", "Parent Run must be a distinct earlier row")

    def _validate_numeric(self, value: object) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise AMLError("schema_invalid", "Stored numeric values must be finite")
        if isinstance(value, dict):
            for child in cast(dict[str, object], value).values():
                self._validate_numeric(child)
        elif isinstance(value, list):
            items = cast(list[object], value)
            if _finite_numeric_sequence(items):
                return
            for child in items:
                self._validate_numeric(child)

    def _validate_projection(
        self,
        kind: str,
        model: StrictModel,
        post: dict[tuple[str, str], StrictModel],
    ) -> None:
        row = model.model_dump(mode="json")
        if row.get("tenant_id", self.tenant_id) != self.tenant_id:
            raise AMLError("tenant_mismatch", "Tenant projection differs from the workspace")
        expected: dict[str, Any] = {}
        if kind == "control":
            value = cast(ControlRowV1, model).value
            expected["object_id"] = control_id(row["key"])
            expected["control_refs"] = [
                ref.model_dump(mode="json")
                for ref in self._projection_refs(extract_references(value))
            ]
            if isinstance(value, PolicyPointerV1):
                policy_kind, tier = POLICY_POINTERS[row["key"]]
                release = self.read_logical(value.current_release_ref, "policy_release")
                if (
                    not isinstance(release, BootstrapPolicyReleaseV1)
                    or release.policy_kind != policy_kind
                ):
                    raise AMLError(
                        "binding_mismatch", "Policy pointer references a different release kind"
                    )
                config = self.read_logical(release.config_ref, "policy_config")
                if config.model_dump()["policy_kind"] != policy_kind:
                    raise AMLError(
                        "binding_mismatch", "Policy release references a different config kind"
                    )
                owner = self.get_row("run", row["updated_by_run_id"] or "")
                envelope = self.get_row("artifact", value.current_release_ref)
                if (
                    owner is None
                    or release.release_id != owner["object_id"]
                    or envelope is None
                    or envelope["created_by_run_id"] != owner["object_id"]
                ):
                    raise AMLError(
                        "binding_mismatch", "Policy release differs from its bootstrap owner"
                    )
                bound = self._bound_input(owner)
                if (
                    bound["run_kind"] != "policy_release"
                    or bound["business_key"]["policy_kind"] != policy_kind
                    or bound["business_key"].get("tier") != tier
                    or bound["business_key"]["builtin_config_digest"]
                    != integrity_digest(config)
                ):
                    raise AMLError(
                        "binding_mismatch",
                        "Policy configuration differs from its frozen digest"
                    )
            elif isinstance(value, ActiveEmbeddingSpaceV1):
                baseline = self.read_logical(value.canary_baseline_ref, "policy_config")
                if not isinstance(baseline, EmbeddingCanaryConfigV1):
                    raise AMLError(
                        "binding_mismatch", "Embedding state requires an embedding canary config"
                    )
                parameters = baseline.parameters
                if parameters.dim != self.dimension:
                    raise AMLError(
                        "embedding_dimension_mismatch", "Canary dimension differs from workspace"
                    )
                if (
                    parameters.model != EMBEDDING_MODEL
                    or parameters.text_version != CANARY_TEXT_VERSION
                ):
                    raise AMLError(
                        "binding_mismatch",
                        "Canary model or text version differs from initialization",
                    )
                digest = integrity_digest(parameters)
                if value.canary_digest != digest or value.embedding_space != (
                    f"{parameters.model}@{digest}@{parameters.dim}"
                ):
                    raise AMLError(
                        "parity_mismatch", "Embedding state differs from its frozen canary baseline"
                    )
        for field, value in expected.items():
            if canonical_json(row[field]) != canonical_json(value):
                raise AMLError("parity_mismatch", "Derived field differs from its sole authority")
        self._validate_numeric(row)
        if "embedding" in row:
            vector = row["embedding"]
            if len(vector) != self.dimension:
                raise AMLError(
                    "embedding_dimension_mismatch", "Vector dimension differs from workspace"
                )
            # An erased source is the only zero-vector owner.
            if not any(vector) and row.get("state") != "erased":
                raise AMLError("schema_invalid", "An active vector cannot have zero norm")
            if self.embedding_write_state == "space_mismatch":
                raise AMLError("embedding_space_mismatch", "Canary mismatch blocks vector writes")
            active = self.get_control("policy:embedding_space")
            if active is None or active["value"].get("state") != "active":
                raise AMLError(
                    "embedding_unavailable", "Active embedding configuration is unavailable"
                )
            if row["embedding_space"] != active["value"]["embedding_space"]:
                raise AMLError(
                    "embedding_space_mismatch", "Vector belongs to a different embedding space"
                )
        old = self.get_row(kind, row["object_id"])
        if old is not None and kind == "evidence":
            if row["state"] != "erased" and old.get("source_participant") != row.get(
                "source_participant",
            ):
                raise AMLError("immutable_field", "Original source participant cannot be replaced")
            for name in (
                "ingested_commit_seq",
                "ingested_at",
                "scope_kind",
                "scope_id",
                "session_id",
                "request_id",
                "ordinal",
                "part_no",
                "kind",
            ):
                if old[name] != row[name]:
                    raise AMLError(
                        "immutable_field", "Evidence reception identity cannot be changed"
                    )
            check_evidence_change(old, row)

    def _vector_space(self) -> VectorSpace:
        """Return the active embedding space every written vector must name (DM-09)."""
        if self.embedding_write_state == "space_mismatch":
            raise AMLError("embedding_space_mismatch", "Canary mismatch blocks vector writes")
        active = self.get_control("policy:embedding_space")
        if active is None or active["value"].get("state") != "active":
            raise AMLError("embedding_unavailable", "Active embedding configuration is unavailable")
        return VectorSpace(self.dimension, active["value"]["embedding_space"])

    def _prepare_rows(
        self,
        rows: Sequence[RowWrite],
        ticket: WriteTicket | None,
    ) -> tuple[list[RowWrite], Any]:
        """Validate one main-row batch and plan its HeavenBase RowOps under the write scope.

        Returns:
            The strictly parsed post-images and the upsert plan whose main
            operations follow their order.
        """
        parsed = self._validate_rows(rows, ticket)
        return parsed, self._plan_rows(parsed)

    def _validate_rows(
        self,
        rows: Sequence[RowWrite],
        ticket: WriteTicket | None,
        *,
        deleted: frozenset[tuple[str, str]] = frozenset(),
        artifacts: Mapping[tuple[str, str], StrictModel] | None = None,
    ) -> list[RowWrite]:
        """Parse and validate one main-row batch against its admission and current rows.

        This is the validation ``_prepare_rows`` runs before planning: strict
        parsing, duplicate identities, Run admission, Control provenance and
        versions, append-only identities, Candidate rules, projections, typed
        references and scope. It writes and admits nothing. A caller that runs
        it without the Writer mutex reads borrowed rows and must prove them
        unchanged, or validate again under the mutex, before any row is
        written (ADR 0036).

        Args:
            rows: Complete strict post-images in write order.
            ticket: The admitted write owner, or None for an owner-less batch.
            deleted: The kind and identity of every row the same transaction
                deletes; the cross-row proofs read them as absent.
            artifacts: The Artifact rows the same transaction creates
                (``commit_artifacts``); the reference proofs read them as present.

        Returns:
            The strictly parsed post-images in batch order.

        Raises:
            AMLError: With the code of the first failed validation.
        """
        from aml.store.indexed import BatchedHeaders

        if not rows:
            raise AMLError("invalid_request", "A write batch cannot be empty")
        parsed = [
            RowWrite(item.kind, self._parse_row(item.kind, item.row.model_dump(mode="json")))
            for item in rows
        ]
        post = {
            (item.kind, cast(str, item.row.model_dump()["object_id"])): item.row for item in parsed
        }
        if len(post) != len(parsed):
            raise AMLError("invalid_request", "A batch contains the same row more than once")
        if ticket is not None:
            self._admit(ticket)
        post_rows = {key: model.model_dump(mode="json") for key, model in post.items()}
        post.update(artifacts or {})
        space = functools.cache(self._vector_space)
        # Share only Evidence ownership headers within this validation. Defer
        # their first read until the first reference needs it, preserving the
        # row/refusal order and the existing pre-state versus post-state lookup.
        # Contextual reference validation still runs at its original row boundary.
        evidence = BatchedHeaders(
            lambda identifiers: self._reference_reader().evidence_reference_headers(identifiers),
            lambda identifier: self._reference_reader().evidence_reference_header(identifier),
            (ref.target_id for item in parsed for ref in extract_references(item.row)
             if ref.spec.target_kind == "evidence" and ref.target_id not in self._sizing_refs
             and (ref.spec.fulfillment == "pre_state"
                  or ("evidence", ref.target_id) not in post)),
        )

        def latest(kind: str, identity: str) -> Mapping[str, Any] | None:
            if (kind, identity) in deleted:
                return None
            found = post_rows.get((kind, identity))
            return self.get_row(kind, identity) if found is None else found

        for item in parsed:
            if item.kind == "artifact":
                raise AMLError(
                    "invalid_write_mode", "Artifacts require independent put-once prewrites"
                )
            row = item.row.model_dump(mode="json")
            if item.kind == "control" and ticket is not None:
                if ticket.run_id is None or row["updated_by_run_id"] != ticket.run_id:
                    raise AMLError(
                        "binding_mismatch", "Control write lacks the exact Run provenance"
                    )
                if row["key"] in {"binding:workspace", "lease:writer"} or row["key"].startswith(
                    "scope:"
                ):
                    raise AMLError(
                        "immutable_field", "Run cannot overwrite initialization authority"
                    )
                previous = self.get_control(row["key"])
                if row["version"] != (0 if previous is None else previous["version"] + 1):
                    raise AMLError(
                        "parity_mismatch", "Control write must advance exactly one version"
                    )
            if item.kind in _APPEND_ONLY and self.get_row(item.kind, row["object_id"]) is not None:
                raise AMLError("immutable_field", "Append-only row identity already exists")
            self._validate_projection(item.kind, item.row, post)
            if item.kind in MEMORY_ROW_KINDS:
                check_row(
                    item.kind, row, latest=latest, stored=self.get_row, batch=post_rows,
                    space=space,
                )
            self._validate_refs(item.kind, item.row, post, _evidence=evidence)
            if ticket is not None and self._scope(item.kind, row, post) not in (
                None,
                (ticket.scope_kind, ticket.scope_id),
            ):
                raise AMLError("scope_denied", "The batch spans a different scope from its Run")
        check_batch(
            (kind, value) for (kind, _), value in post_rows.items() if kind in MEMORY_ROW_KINDS
        )
        return parsed

    def _prepare_row_inputs(self, parsed: Sequence[RowWrite]) -> list[Any]:
        """Encode one complete batch only at the HB physical write boundary.

        All admission, frozen V1 post-images and integrity digests retain their
        existing owner. Removed physical projections are resolved from the exact
        same-batch post-image or a selected existing authority, never a cache of
        independent duplicated values.
        """
        from aml.store.normalized_batch import check_normalized_batch, encode_normalized_batch

        checked = check_normalized_batch(parsed, max_rows=MAX_TX_ROWS)
        keys = checked.authority_keys
        authorities: dict[tuple[str, str], StrictModel | None] = {}
        for kind in sorted({kind for kind, _identifier in keys}):
            identifiers = sorted(identifier for target_kind, identifier in keys
                                 if target_kind == kind)
            for identifier, raw in zip(
                identifiers, self.get_rows(kind, identifiers), strict=True,
            ):
                authorities[kind, identifier] = (
                    None if raw is None else self._parse_row(kind, raw)
                )
        payloads = encode_normalized_batch(
            parsed, tenant_id=self.tenant_id, authorities=authorities, max_rows=MAX_TX_ROWS,
            checked=checked,
        )
        return [self._ws._row_inputs.upsert_row(self.entities[item.kind], payload, {})
                for item, payload in zip(parsed, payloads, strict=True)]

    def _plan_rows(self, parsed: Sequence[RowWrite]) -> Any:
        """Materialize one validated batch's main and Catalog RowOps inside the write scope.

        Args:
            parsed: Strictly parsed post-images from ``_validate_rows``.

        Returns:
            The HeavenBase upsert plan whose main operations follow ``parsed`` order.

        Raises:
            AMLError: If the plan leaves the main backend or exceeds ``MAX_TX_ROWS``.
        """
        inputs = self._prepare_row_inputs(parsed)
        plan = self._ws._writer.prepare_upsert_rows(inputs, overwrite=True, catalog_mode="merge")
        if set(plan.entity_ops) != {"main"} or len(plan.entity_ops["main"]) > MAX_TX_ROWS:
            raise AMLError(
                "budget_exceeded", "The materialized main-row transaction exceeds its budget"
            )
        return plan

    def _invalidate_pending_recall_references(self, plan: UpsertPlan) -> None:
        """Reserve exact main/Catalog changes before their actual storage attempt."""
        self._pending_reference_registry.invalidate(frozenset(
            (operation.entity_id, operation.object_id)
            for grouped in (plan.entity_ops, plan.catalog_ops)
            for operations in grouped.values() for operation in operations
        ))

    def _write_plan(self, plan: Any) -> None:
        """Commit one prepared plan's main rows, then its derived Catalog rows.

        Args:
            plan: HeavenBase upsert plan prepared under this Writer's mutex.

        Raises:
            AMLError: ``storage_conflict`` when a database constraint refused the
                main rows; their transaction was rolled back whole and nothing
                was written. ``catalog_repair_required`` after committed main
                rows whose Catalog step failed. Any other storage failure
                propagates unchanged and stays fatal (``aml.store.failures``).
        """
        # Reserve before the attempt: a committed write can lose its acknowledgement.
        self._invalidate_pending_recall_references(plan)
        owner = _OWN_FINALIZE.get()
        scope = None if owner is None else self._own_finalize_scope(plan, owner)
        before = None if owner is None or scope is None else (
            self._child_change_tokens.get(owner.owner_run_id, 0),
            self._artifact_change_tokens.get(scope, 0),
        )
        self._invalidate_child_runs(plan)
        self._invalidate_artifact_rows(plan)
        if owner is not None and scope is not None and before is not None:
            self._credit_own_source_advance(owner, scope, before)
        self.ref_generation += 1
        row_writer = self._ws._writer
        write_rows = row_writer.write_dependency_ordered_ops
        with self._catalog_steps():
            try:
                self._mutate_storage(lambda: write_rows(plan.entity_ops))
            except Exception as error:
                if not constraint_violation(error):
                    raise
                # The main rows are one HeavenBase transaction on the one main
                # backend (``_plan_rows``): a constraint that refused any of them
                # rolled all of them back, so this plan wrote nothing durable and
                # leaves nothing unpublished. Its scope's caller isolates it.
                self.__dict__["_rolled_back_plans"] = self.__dict__.get(
                    "_rolled_back_plans", 0,
                ) + 1
                raise AMLError(
                    "storage_conflict", "A storage constraint refused the whole transaction",
                ) from error
            try:
                self._mutate_storage(lambda: self._ws._writer.write_ops(plan.catalog_ops))
            except Exception as error:
                raise AMLError(
                    "catalog_repair_required", "Main rows committed; Catalog needs repair"
                ) from error

    def recover_catalog(
        self, *, proof: CatalogProofV1 | None = None, ledger: CatalogLedgerV1 | None = None,
    ) -> None:
        """Reconcile only HB's derived Catalog before returning recovery to service.

        HeavenBase's pinned repair derives Catalog rows from authoritative Entity
        rows. It neither replays AML effects nor advances AML visibility or
        reference generations. Live receipt preparations are invalidated before
        the first audit. A healthy Catalog needs one complete parity proof;
        an inconsistent Catalog still receives the full repair and its final
        verification under this same hold. No audit row content is returned or
        logged.

        At boot, ``proof`` is the previous lifetime's clean-close proof
        (``aml.store.catalog_proof``) and ``ledger`` its running ledger's clean
        last state (``aml.store.catalog_ledger``), each already matched to the
        database files. Either stands for the audit only when it names the Writer
        lease this boot's epoch CAS replaced and no storage mutation of this
        process failed; anything else audits.

        Args:
            proof: The previous lifetime's Catalog proof, or None.
            ledger: The previous lifetime's clean Catalog ledger, or None.

        Raises:
            AMLError: If the derived Catalog remains inconsistent after repair.
        """
        with self._write_scope():
            self._pending_reference_registry.invalidate_all()
            replaced = self.__dict__.get("_replaced_writer_lease")
            if not self.__dict__.get("_catalog_unproven", False) and replaced is not None and any(
                kept is not None and (kept.owner_instance_id, kept.writer_epoch) == replaced
                for kept in (proof, ledger)
            ):
                self.__dict__["_catalog_proven"] = True
                return
            try:
                result = self._mutate_storage(
                    lambda: self._ws.repair(dry_run=True), hooked=False,
                )
                if result["catalog"]["ok"] is not True:
                    result = self._mutate_storage(self._ws.repair, hooked=False)
            except Exception:
                raise AMLError(
                    "service_unhealthy", "Catalog recovery could not be verified"
                ) from None
            if result["catalog"]["ok"] is not True:
                raise AMLError("service_unhealthy", "Catalog recovery did not establish parity")
            # Parity is proved under this hold: earlier failed mutations are repaired.
            self.__dict__["_catalog_unproven"] = False
            self.__dict__["_catalog_proven"] = True

    def start_catalog_ledger(self, root: Path, workspace_id: str) -> bool:
        """Open this lifetime's Catalog ledger once its Catalog is proved (C.13 step 9).

        From here every critical section that mutates storage records its
        mutations in flight (``aml.store.catalog_ledger``), so an unclean close
        leaves the next boot a proof of whether any Catalog step was pending.

        Args:
            root: The workspace directory, owned by this process.
            workspace_id: The workspace binding's identity.

        Returns:
            Whether a ledger was opened; without one, an unclean close audits.
        """
        with self.mutex:
            if self.__dict__.get("_catalog_unproven", False) or not self.__dict__.get(
                "_catalog_proven", False,
            ) or self.__dict__.get("_catalog_ledger") is not None:
                return False
            lease = self._boot_writer_lease_json
            if lease is None:
                return False
            value = json.loads(lease)["value"]
            ledger = start_catalog_ledger(
                root, workspace_id, (value["owner_instance_id"], value["writer_epoch"]),
            )
            if ledger is None:
                return False
            self.__dict__["_catalog_ledger"] = ledger
            self.mutex.on_outermost_release(self._ledger_section_end)
            return True

    def close_catalog_ledger(self) -> None:
        """Stop recording and close this lifetime's Catalog ledger; its state stays."""
        ledger = cast(CatalogLedger | None, self.__dict__.pop("_catalog_ledger", None))
        if ledger is not None:
            self.mutex.on_outermost_release(None)
            ledger.close()

    def seal_catalog_proof(
        self, *, timeout: float = CATALOG_SEAL_SECONDS,
    ) -> tuple[str, int] | None:
        """Seal a clean close's Catalog proof (C.13 boot step 9; ``catalog_proof``).

        Called once the service has drained, before its storage closes. The proof
        holds when this lifetime proved the Catalog (its boot audit, a later repair
        or the previous lifetime's proof) and no storage mutation failed or stopped
        between its main and Catalog rows since. Sealing takes the Writer mutex, so
        no mutation is in flight; any later mutation voids the seal
        (``catalog_proof_intact``).

        Args:
            timeout: Seconds to wait for the Writer mutex; a busy Writer seals nothing.

        Returns:
            This lifetime's Writer lease instance and epoch for the proof, or None.
        """
        if not self.mutex.acquire(timeout=timeout):
            return None
        try:
            if self.__dict__.get("_catalog_unproven", False) or not self.__dict__.get(
                "_catalog_proven", False,
            ):
                return None
            lease = self._boot_writer_lease_json
            if lease is None:
                return None
            value = json.loads(lease)["value"]
            self.__dict__["_catalog_sealed"] = True
            return value["owner_instance_id"], value["writer_epoch"]
        finally:
            self.mutex.release()

    def catalog_proof_intact(self) -> bool:
        """Whether a sealed Catalog proof still holds: no storage mutation since the seal."""
        return bool(self.__dict__.get("_catalog_sealed", False)) and not self.__dict__.get(
            "_catalog_unproven", False,
        )

    def record_search_delivery(self, row: ReplayableRecallRowV1) -> None:
        """Persist one Search delivery's ``recall`` row in the delivery critical section.

        A.3.11 keeps the copied row; E.1.3 makes it the only durable copy of the
        query and options and requires its ``latency_ms``. The Search use case
        calls this after ``SnapshotReadGuard.check_search_delivery`` and before
        its response enqueue, under the same outermost Writer mutex, so the row
        and the delivered bytes describe one linearized delivery. The row has
        no owning Run: it is bookkeeping of the read path, never a domain commit,
        so it advances no ``commit_seq`` or scope counter and nothing publishes.

        Args:
            row: The complete strict delivery row, with a fresh identity.

        Raises:
            AMLError: ``reader_busy`` outside the caller's Writer critical section;
                ``binding_mismatch`` for another tenant or a missing latency;
                ``immutable_field`` for an existing identity; or the first failed
                row validation.
        """
        if not self.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise AMLError("reader_busy", "A Search delivery row requires the delivery mutex")
        if not isinstance(cast(object, row), ReplayableRecallRowV1) or (
            row.tenant_id != self.tenant_id or row.latency_ms is None
        ):
            raise AMLError("binding_mismatch", "Search delivery row lacks its exact binding")
        with self._write_scope():
            if self.get_row("recall", row.object_id) is not None:
                raise AMLError("immutable_field", "Search delivery identity already exists")
            _, plan = self._prepare_rows([RowWrite("recall", row)], None)
            self._write_plan(plan)

    def close_pending_recall_references(self) -> None:
        """Close reference issuance after the publisher's actual callbacks have drained."""
        self._pending_reference_registry.close()

    def _visibility_counters(
        self, ticket: WriteTicket, *, scope_changed: bool, updated_at: int,
        scoped: Sequence[str] = (),
    ) -> list[RowWrite]:
        """Materialize the sole counter updates inside the owning main-row batch.

        ``scoped`` names further scope counters the transaction advances, such
        as ``tombstone_seq`` and ``erase_epoch``.
        """
        keys = ["commit_seq"]
        if scope_changed or scoped:
            if ticket.scope_id is None:
                raise AMLError("scope_denied", "Visible domain changes require a scoped owner")
        if scope_changed:
            keys.append(f"scope_seq:{ticket.scope_id}")
        keys += [f"{prefix}:{ticket.scope_id}" for prefix in scoped]
        counters: list[RowWrite] = []
        for key in keys:
            before = self.get_control(key)
            if before is None:
                raise AMLError("parity_mismatch", "Visibility counter was not initialized")
            counters.append(
                RowWrite(
                    "control",
                    self.materialize_control(
                        key,
                        {"seq": before["value"]["seq"] + 1},
                        before["version"] + 1,
                        updated_at,
                        ticket.run_id,
                    ),
                )
            )
        return counters

    def write_bookkeeping(self, ticket: WriteTicket, rows: Sequence[RowWrite]) -> None:
        """Validate a bounded non-business batch under the captured admission ticket.

        Run updates are restricted to the plan's bookkeeping whitelist. Domain
        business rows use commit_batch and cannot bypass their write-set marker.
        """
        with self._write_scope():
            if any(item.kind not in _ROW_BOOKKEEPING for item in rows):
                raise AMLError("invalid_write_mode", "Business rows require a write-set commit")
            for item in rows:
                data = item.row.model_dump(mode="json")
                old = self.get_row(item.kind, data["object_id"])
                if item.kind == "run" and old is None:
                    raise AMLError(
                        "invalid_write_mode", "New Runs require their admission producer"
                    )
                if item.kind == "run" and old is not None:
                    if old["run_kind"] == "finalize":
                        raise AMLError(
                            "invalid_write_mode", "Finalize requires its transition owner",
                        )
                    # Head-stale revalidation was the only Candidate retention (S0b).
                    if old["run_kind"] == "retention" and old["target_kind"] == "candidate":
                        raise AMLError(
                            "invalid_write_mode", "Head-stale requires its transition owner",
                        )
                    if old["run_kind"] == "retention" and old["target_kind"] == "affected_set":
                        raise AMLError(
                            "invalid_write_mode",
                            "Ingestion cleanup requires its dedicated transition producer",
                        )
                    if data["object_id"] != ticket.run_id:
                        raise AMLError(
                            "invalid_write_mode", "Bookkeeping cannot update another Run"
                        )
                    changed = {key for key, value in data.items() if value != old.get(key)}
                    if not changed <= _RUN_BOOKKEEPING:
                        raise AMLError(
                            "immutable_field", "Run bookkeeping cannot change immutable fields"
                        )
                    if data["writer_epoch"] != old["writer_epoch"]:
                        raise AMLError("lease_lost", "Epoch adoption requires its dedicated proof")
                    if data["status"] != old["status"] and (old["status"], data["status"]) not in {
                        ("queued", "running"),
                        ("queued", "manual_attention"),
                        ("running", "succeeded"),
                        ("running", "failed"),
                        ("running", "manual_attention"),
                    }:
                        raise AMLError(
                            "invalid_run_transition",
                            "Run transition requires an unopened recovery producer",
                        )
                    if data["status"] == "running" or data["stage"] != "await_input":
                        following_input = self._bound_input(data)
                        if following_input.get("input_phase") == "durable" and (
                            old["input_ref"] is None or old["input_ref"] != data["input_ref"]
                        ):
                            raise AMLError(
                                "invalid_write_mode",
                                "Durable reception attachment requires its visibility producer",
                            )
                    if old["input_ref"] is not None and old["input_ref"] != data["input_ref"]:
                        prior = self._bound_input(old)
                        following = self._bound_input(data)
                        if (prior.get("input_phase"), following.get("input_phase")) != (
                            "admission",
                            "durable",
                        ):
                            raise AMLError(
                                "immutable_field",
                                "Attached input cannot be replaced by another execution input",
                            )
                        for name in ("admission_digest", "payload_digest", "business_key"):
                            if prior[name] != following[name]:
                                raise AMLError(
                                    "binding_mismatch",
                                    "Durable input replacement changed admission content",
                                )
                    self._execution_fences(data)
                if item.kind == "control" and (
                    data["key"] == "commit_seq"
                    or data["key"].startswith(
                        (
                            "policy:",
                            "lease:",
                            "erase_epoch:",
                            "consent_epoch:",
                            "scope_seq:",
                            "tombstone_seq:",
                            "request_ledger:",
                            "finalized:",
                            "checked_source:",
                            "request_completion:",
                            "finalize_budget:",
                            "source_labels_budget:",
                            "extraction_budget:",
                            "source_labels_message:",
                            "judge_output:",
                            "completed_prefix:",
                            "scope_quarantine:",
                            "maintenance:",
                        )
                    )
                ):
                    raise AMLError(
                        "invalid_write_mode", "Coordinating Control requires its dedicated producer"
                    )
            _, plan = self._prepare_rows(rows, ticket)
            self._write_plan(plan)

    def commit_rows(
        self,
        ticket: WriteTicket,
        rows: Sequence[RowWrite],
        *,
        deletes: Sequence[RowDelete] = (),
        heads: Sequence[ExpectedHead] = (),
        bumps: frozenset[str] = frozenset(),
        journal: StrictModel | None = None,
        mark_run: bool = False,
        updated_at: int,
        publish: Callable[[CommittedRows], None],
        progress: Sequence[RowWrite] = (),
    ) -> CommittedRows | None:
        """Commit one NL transaction of typed row writes and deletes (I2; C.11.2 #3, C.12, C.13).

        Under the Writer mutex the transaction compares every expected head
        with its stored Item, proves its rows (``_validate_rows``, including the
        cross-row proofs), advances ``commit_seq`` and, when it writes or
        deletes a row of the A.3.9 kinds, ``scope_seq:{s}``, adds its
        create-once journal, advances the generation of every entity its rows
        are anchored to (C.14), then writes its main rows before their Catalog
        rows and publishes the read view before the mutex is released. Which
        transactions a producer may commit is its own admission's decision.

        Args:
            ticket: The admitted, scoped owner of the transaction.
            rows: Complete strict post-images of item, revision, link,
                candidate, Evidence, entity, entity_name and tombstone rows.
                Each row with a ``commit_seq`` carries this transaction's, and
                so does a new candidate's ``created_commit_seq``; a tombstone
                carries the advanced ``tombstone_seq``. Evidence is only
                changed here; reception creates it.
            deletes: Stored item, revision, link, candidate, entity_name and
                tombstone rows to delete.
            heads: Items whose stored head and lifecycle must be unchanged.
            bumps: The scope counters the producer advances besides
                ``commit_seq`` and ``scope_seq``: ``tombstone_seq`` and
                ``erase_epoch`` together, for a forget admission or any
                transaction that writes or deletes a tombstone (A.3.8, D.3,
                D.7.5). An ordinary Add advances neither.
            journal: A ``write_set`` body carrying ``journal_key`` (store-Q5),
                written create-once by the ticket's Run inside the
                transaction. A ``commit_seq`` in it is this transaction's.
            mark_run: Also set the ticket's Run ``commit_marker`` to the
                journal and its ``commit_seq`` (a publish Run's component).
            updated_at: The transaction's time for its counter, Run and
                journal rows.
            publish: The read model's publication of the committed rows,
                called once under the Writer mutex after the write.

        Returns:
            The committed rows that ``publish`` received, or None when the Run
            already stored this journal with the same bytes: the transaction
            committed before, and nothing is written or published again.

        Raises:
            AMLError: ``target_changed`` when an expected head moved;
                ``invalid_write_mode`` for a kind this transaction cannot write
                or delete, a tombstone change without both bumps, or a second
                Run marker; ``artifact_digest_collision`` when the Run stored
                the journal key with other bytes; ``reference_missing`` when a
                written row references a deleted one or a deleted row does not
                exist; ``parity_mismatch`` for a row or journal naming another
                ``commit_seq`` or ``tombstone_seq``; ``budget_exceeded`` beyond
                ``MAX_TX_ROWS``; or the first failed row validation.
        """
        with self._write_scope():
            prepared = self._prepare_commit(
                ticket, rows, deletes=deletes, heads=heads, bumps=bumps, journal=journal,
                mark_run=mark_run, updated_at=updated_at, progress=progress,
            )
            if prepared is None:
                return None
            committed = self._commit_prepared(ticket, *prepared, journal=journal)
            publish(committed)
            return committed

    def commit_rows_within(
        self,
        ticket: WriteTicket,
        rows: Sequence[RowWrite],
        *,
        deletes: Sequence[RowDelete] = (),
        heads: Sequence[ExpectedHead] = (),
        journal: StrictModel | None = None,
        updated_at: int,
        publish: Callable[[CommittedRows], None],
        before_write: Callable[[], None],
        limit: int | None = MAX_TX_ROWS,
        capacity_refusal: bool = False,
    ) -> CommittedRows | int | None:
        """Measure and commit one NL transaction from a single preparation (C.12).

        A producer that measured a chunk or a component with ``commit_rowops`` and then
        committed it with ``commit_rows`` under the same mutex hold proved and planned it
        twice. Here it is proved and planned once, under the write scope: when ``limit``
        is set and its main RowOps exceed it, the count is returned and nothing is
        written; otherwise ``before_write`` runs (the owner's budget check, C.11.2 #3), the
        expected heads are compared and the prepared plan is written and published,
        exactly as ``commit_rows`` writes it.

        Args:
            ticket: The admitted, scoped owner of the transaction.
            rows: As ``commit_rows``.
            deletes: As ``commit_rows``.
            heads: As ``commit_rows``.
            journal: As ``commit_rows``.
            updated_at: As ``commit_rows``.
            publish: As ``commit_rows``.
            before_write: Run under the mutex after the measurement, before the heads
                and the write; it may raise to refuse the write.
            limit: The RowOps a caller can still split below, or None to commit whatever
                the count (``commit_rows`` then refuses one over ``MAX_TX_ROWS``).
            capacity_refusal: Return the preparation's own capacity refusal
                (``budget_exceeded``) as ``MAX_TX_ROWS + 1`` instead of raising it.

        Returns:
            The count when it exceeds ``limit``; None when the Run already stored the
            journal with the same bytes (``before_write`` still runs); else the
            committed rows ``publish`` received.

        Raises:
            AMLError: As ``commit_rowops`` and ``commit_rows``.
        """
        with self._write_scope():
            try:
                prepared = self._prepare_commit(
                    ticket, rows, deletes=deletes, bumps=frozenset[str](), journal=journal,
                    mark_run=False, updated_at=updated_at,
                )
            except AMLError as error:
                if not capacity_refusal or error.code != "budget_exceeded":
                    raise
                return MAX_TX_ROWS + 1
            if prepared is not None and limit is not None and (
                size := len(prepared[2].entity_ops["main"])
            ) > limit:
                return size
            before_write()
            if prepared is None:
                return None
            self._check_heads(heads)
            committed = self._commit_prepared(ticket, *prepared, journal=journal)
            publish(committed)
            return committed

    def _check_heads(self, heads: Sequence[ExpectedHead]) -> None:
        """Refuse a transaction whose expected Item heads moved (``target_changed``)."""
        for head in heads:
            stored = self.get_row("item", head.memory_id)
            if stored is None or (
                stored["current_revision_id"], stored["lifecycle_state"],
            ) != (head.current_revision_id, head.lifecycle_state):
                raise AMLError("target_changed", "An expected head moved before its commit")

    def _commit_prepared(
        self,
        ticket: WriteTicket,
        parsed: Sequence[RowWrite],
        removed: Sequence[tuple[str, dict[str, Any]]],
        plan: Any,
        anchored: frozenset[str],
        *, journal: StrictModel | None = None,
    ) -> CommittedRows:
        """Write one transaction ``_prepare_commit`` proved, under the write scope.

        The generation of every entity its rows are anchored to advances first
        (C.14), because a committed write can lose its acknowledgement.

        Returns:
            The committed rows, for the read model's publication.

        Raises:
            AMLError: ``budget_exceeded`` beyond ``MAX_TX_ROWS``.
        """
        if len(plan.entity_ops["main"]) > MAX_TX_ROWS:
            raise AMLError("budget_exceeded", "The NL transaction exceeds its budget")
        generations: dict[tuple[str, str], int] = self.__dict__.setdefault(
            "_entity_generations", {},
        )
        for entity in anchored:
            key = (cast(str, ticket.scope_id), entity)
            generations[key] = generations.get(key, 0) + 1
        directory = self.scan_memo().lookup(("commit-journals", ticket.run_id)) \
            if journal is not None else MISSING
        extension: AbstractContextManager[ScanExtension | None] = nullcontext(None)
        physical: tuple[tuple[str, str], ...] = ()
        expected: tuple[dict[str, Any], ...] = ()
        journal_key = getattr(journal, "journal_key", None)
        if isinstance(directory, _CommitJournals) and isinstance(journal_key, str):
            schema = self.entities["artifact"].schema()
            # _prepare_commit appends only the journal's physical Artifacts;
            # the final one is its logical root, after any manifest parts.
            operations = [operation for operation in plan.entity_ops["main"]
                          if operation.entity_id == schema.entity_id]
            physical = tuple(("artifact", operation.object_id) for operation in operations)
            expected = tuple(self._ws._row_codec.decode_schema_row(schema, operation.to_row())
                             for operation in operations)
            if physical:
                extension = self._rows_cache().extend_artifact_scan(
                    directory.keys, directory.token, physical,
                    membership_key("artifact", cast(str, ticket.run_id)),
                )
        with extension as receipt:
            self._write_plan(plan)
            verified = self.storage_epoch
            exact = False
            if receipt is not None:
                actual = self._get_physical_rows("artifact", [identity for _, identity in physical])
                exact = all(
                    row is not None and row["body"] == planned["body"]
                    and canonical_json({key: value for key, value in row.items() if key != "body"})
                    == canonical_json({key: value for key, value in planned.items()
                                       if key != "body"})
                    for row, planned in zip(actual, expected, strict=True)
                )
        if isinstance(directory, _CommitJournals) and physical and receipt is not None and (
            receipt.token is not None and isinstance(journal_key, str)
            and exact and not self._rows_cache().changed_since(physical, verified)
        ):
            entries = dict(directory.entries)
            entries[journal_key] = (*entries.get(journal_key, ()), ArtifactId(physical[-1][1]))
            self._store_commit_journals(cast(str, ticket.run_id), _CommitJournals(
                receipt.token, tuple(dict.fromkeys((*directory.keys, *physical))),
                tuple(sorted(entries.items())),
            ))
        return CommittedRows(
            commit_seq=self.counter("commit_seq"),
            scope_kind=cast(str, ticket.scope_kind),
            scope_id=cast(str, ticket.scope_id),
            written=tuple((item.kind, item.row.model_dump(mode="json")) for item in parsed),
            deleted=tuple(removed),
        )

    def commit_rowops(
        self,
        ticket: WriteTicket,
        rows: Sequence[RowWrite],
        *,
        deletes: Sequence[RowDelete] = (),
        bumps: frozenset[str] = frozenset(),
        journal: StrictModel | None = None,
        mark_run: bool = False,
        updated_at: int,
        sizing_refs: frozenset[str] = frozenset(),
    ) -> int:
        """Count the main-backend RowOps ``commit_rows`` would write (C.11.2 #1, C.12; O8).

        The transaction is prepared exactly as ``commit_rows`` prepares it,
        with its counters, journal, Run marker and deletes, and is not
        written; a producer measures a component or a chunk with it against
        ``MAX_TX_ROWS`` before it commits. The count excludes the separate
        Catalog projection.

        Args:
            ticket: The admitted, scoped owner the transaction would have.
            rows: Its row writes, as ``commit_rows`` takes them.
            deletes: Its row deletes.
            bumps: Its further scope counters.
            journal: Its create-once journal.
            mark_run: Whether it marks the ticket's Run.
            updated_at: Its time.
            sizing_refs: Placeholder artifact addresses of a sizing build (W3-RT-14):
                references to them count as present, since they are never written.

        Returns:
            The main RowOp count; 0 when the journal shows the transaction
            already committed.

        Raises:
            AMLError: ``budget_exceeded`` when the rows alone exceed
                ``MAX_TX_ROWS``; otherwise the refusal ``commit_rows`` would
                raise before writing. A count with deletes or a journal may
                exceed ``MAX_TX_ROWS``; ``commit_rows`` refuses it.
        """
        with self._write_scope():
            self._sizing_refs = sizing_refs
            try:
                prepared = self._prepare_commit(
                    ticket, rows, deletes=deletes, bumps=bumps, journal=journal,
                    mark_run=mark_run, updated_at=updated_at,
                )
            finally:
                self._sizing_refs = frozenset[str]()
            return 0 if prepared is None else len(prepared[2].entity_ops["main"])

    def commit_journal(
        self, run_id: str, journal_key: str,
    ) -> tuple[ArtifactId, StrictModel] | None:
        """Find the ``write_set`` journal a Run committed under one key (C.13, store-Q5).

        The zero-effect proof reads it: a transaction whose journal exists
        committed, and one whose journal does not wrote nothing. The lookup is
        bounded by the Run's own ``write_set`` artifacts.

        Args:
            run_id: The creating Run.
            journal_key: The journal's key.

        Returns:
            The journal's reference and body, or None.

        Raises:
            AMLError: ``artifact_digest_collision`` when the Run stored the key
                twice.
        """
        memo_key = ("commit-journals", run_id)
        directory = self.scan_memo().lookup(memo_key)
        if not isinstance(directory, _CommitJournals):
            token = self.storage_epoch
            with recording_reads() as record:
                entries: dict[str, list[ArtifactId]] = {}
                for reference, body in self.find_logical_candidates(run_id, "write_set"):
                    key = getattr(body, "journal_key", None)
                    if isinstance(key, str):
                        entries.setdefault(key, []).append(reference)
            directory = _CommitJournals(
                token, tuple(dict.fromkeys((MEMBERSHIP_ANY,
                    membership_key("artifact", run_id), *record.keys))),
                tuple(sorted((key, tuple(references)) for key, references in entries.items())),
            )
            if record.complete:
                self._store_commit_journals(run_id, directory)
        found = next((references for key, references in directory.entries
                      if key == journal_key), ())
        if len(found) > 1:
            raise AMLError("artifact_digest_collision", "A journal key binds several journals")
        return (found[0], self.read_logical(found[0], "write_set")) if found else None

    def _store_commit_journals(self, run_id: str, directory: _CommitJournals) -> None:
        """Charge the immutable hashes and references to the existing bounded scan memo."""
        size = 128 + sum(192 + len(key) + sum(128 + len(reference) for reference in references)
                         for key, references in directory.entries)
        memo = self.scan_memo()
        if size + 64 * len(directory.keys) > memo.budget:
            return
        memo.store(
            ("commit-journals", run_id), directory.token, directory.keys, directory,
            value_bytes=size,
        )

    def _prepare_commit(
        self,
        ticket: WriteTicket,
        rows: Sequence[RowWrite],
        *,
        deletes: Sequence[RowDelete],
        heads: Sequence[ExpectedHead] = (),
        bumps: frozenset[str],
        journal: StrictModel | None,
        mark_run: bool,
        updated_at: int,
        owner_rows: Sequence[RowWrite] = (),
        progress: Sequence[RowWrite] = (),
    ) -> tuple[list[RowWrite], list[tuple[str, dict[str, Any]]], Any, frozenset[str]] | None:
        """Prove and plan one NL transaction under the write scope, writing nothing.

        ``owner_rows`` are the producer's own Run and Control post-images written
        in the same transaction, such as a privacy erase's progress and Run
        (``apply_privacy_erase``); they are validated with the rows.
        ``progress`` carries the owning erasure worker's fence Control (D.3 P2,
        INV W21): the fence advances in the transaction whose effects it records;
        and the ``evidence_hold:{s}`` Control that keeps a cancelled cohort's holds
        past its deleted rows (D.5).

        Returns:
            The parsed post-images, the deleted rows' pre-images, the main plan
            with the journal and the deletes, and the entities the rows are
            anchored to; or None when the Run already stored the journal with
            the same bytes.
        """
        if ticket.scope_kind is None or ticket.scope_id is None or ticket.run_id is None:
            raise AMLError("scope_denied", "An NL transaction requires a scoped owner Run")
        if any(item.kind not in _COMMIT_KINDS for item in rows) or any(
            item.kind not in _DELETE_KINDS for item in deletes
        ) or any(item.kind not in {"run", "control"} for item in owner_rows):
            raise AMLError("invalid_write_mode", "An NL transaction writes only its row kinds")
        if any(
            item.kind != "control"
            or not (
                cast(str, item.row.model_dump()["key"]).startswith(f"erasure:{ticket.scope_id}:")
                or item.row.model_dump()["key"] == f"evidence_hold:{ticket.scope_id}"
            )
            for item in progress
        ):
            raise AMLError(
                "invalid_write_mode",
                "Only the erasure fence and its outliving holds record transaction progress",
            )
        tombstones = any(item.kind == "tombstone" for item in (*rows, *deletes))
        if not bumps <= _TOMBSTONE_BUMPS or tombstones and bumps != _TOMBSTONE_BUMPS:
            raise AMLError(
                "invalid_write_mode", "A tombstone changes with tombstone_seq and erase_epoch",
            )
        if mark_run and journal is None:
            raise AMLError("invalid_request", "A Run marker names the transaction's journal")
        typed: StrictModel | None = None
        if journal is not None:
            # A replay is recognized by its journal before any of its rows are read.
            typed = parse_artifact_body("write_set", journal.model_dump(mode="json"))
            stored_journal = self.commit_journal(
                ticket.run_id, typed.model_dump(mode="json")["journal_key"],
            )
            if stored_journal is not None:
                if canonical_json(stored_journal[1]) != canonical_json(typed):
                    raise AMLError(
                        "artifact_digest_collision", "The journal key binds other bytes",
                    )
                return None
        self._check_heads(heads)
        removed: list[tuple[str, dict[str, Any]]] = []
        for item in deletes:
            stored = self.get_row(item.kind, item.object_id)
            if stored is None:
                raise AMLError("reference_missing", "A deleted row does not exist")
            if self._scope(item.kind, stored, {}) != (ticket.scope_kind, ticket.scope_id):
                raise AMLError("scope_denied", "The batch spans a different scope from its Run")
            removed.append((item.kind, stored))
        gone = frozenset((kind, row["object_id"]) for kind, row in removed)
        if len(gone) != len(removed):
            raise AMLError("invalid_request", "A batch deletes the same row more than once")
        sequence = self.counter("commit_seq") + 1
        tombstone_seq = self.counter(f"tombstone_seq:{ticket.scope_id}") + (
            "tombstone_seq" in bumps
        )
        for item in rows:
            data = item.row.model_dump(mode="json")
            identity = (item.kind, cast(str, data["object_id"]))
            if identity in gone:
                raise AMLError("invalid_request", "A batch writes and deletes the same row")
            stored = self.get_row(*identity)
            if item.kind == "evidence" and stored is None:
                raise AMLError("invalid_write_mode", "Evidence starts only at reception")
            # The *_commit indexes find every row a later fence must re-check.
            if data.get("commit_seq", sequence) != sequence or (
                item.kind == "candidate" and stored is None
                and data["created_commit_seq"] != sequence
            ):
                raise AMLError(
                    "parity_mismatch", "A written row names another transaction's commit_seq",
                )
            if item.kind == "tombstone" and data["tombstone_seq"] != tombstone_seq:
                raise AMLError("parity_mismatch", "A tombstone names another tombstone_seq")
        journal_rows: tuple[ArtifactRowV1, ...] = ()
        owner: list[RowWrite] = [*owner_rows, *progress]
        if typed is not None:
            if typed.model_dump(mode="json").get("commit_seq", sequence) != sequence:
                raise AMLError("parity_mismatch", "A journal names another commit_seq")
            journal_rows = self._journal_rows(ticket, typed, created_at=updated_at)
            if mark_run:
                run = self.get_row("run", ticket.run_id)
                if run is None or run["commit_marker"] is not None:
                    raise AMLError("invalid_write_mode", "A Run carries one business marker")
                run.update(
                    commit_marker=journal_rows[-1].object_id, commit_seq=sequence,
                    updated_at=updated_at,
                )
                owner.append(RowWrite("run", self._parse_row("run", run)))
        visible = any(
            kind in _SCOPE_SEQ_KINDS
            for kind in (*(item.kind for item in rows), *(kind for kind, _ in removed))
        )
        counters = self._visibility_counters(
            ticket, scope_changed=visible, updated_at=updated_at, scoped=sorted(bumps),
        )
        parsed = self._validate_rows([*rows, *owner, *counters], ticket, deleted=gone)
        for item in parsed:
            for ref in contextual_references(item.row):
                if ref.spec.ref_mode != "historical_locator" and (
                    ref.spec.target_kind, ref.target_id,
                ) in gone:
                    raise AMLError("reference_missing", "A written row references a deleted row")
        written = {
            (item.kind, cast(str, item.row.model_dump()["object_id"])): item.row
            for item in parsed
        }

        def latest(kind: str, identity: str) -> Mapping[str, Any] | None:
            if (kind, identity) in gone:
                return None
            found = written.get((kind, identity))
            return self.get_row(kind, identity) if found is None else found.model_dump(
                mode="json",
            )

        for kind, row in removed:
            check_deletion(kind, row, latest=latest)
        anchored = self._anchored_entities(parsed, removed, latest)
        plan = self._plan_rows(parsed)
        if journal_rows:
            created = self._ws._writer.prepare_upsert_rows(
                self._prepare_row_inputs([RowWrite("artifact", row) for row in journal_rows]),
                overwrite=False, catalog_mode="fresh",
            )
            if set(created.entity_ops) != {"main"}:
                raise AMLError("parity_mismatch", "A journal left the main backend")
            plan.entity_ops["main"].extend(created.entity_ops["main"])
            for route, operations in created.catalog_ops.items():
                plan.catalog_ops.setdefault(route, []).extend(operations)
        if removed:
            self._plan_deletes(plan, removed)
        return parsed, removed, plan, anchored

    def _anchored_entities(
        self, parsed: Sequence[RowWrite], removed: Sequence[tuple[str, dict[str, Any]]],
        latest: Callable[[str, str], Mapping[str, Any] | None],
    ) -> frozenset[str]:
        """Return every entity a transaction's written or deleted rows are anchored to.

        A row counts with its stored image and its post-image; an Item also
        with the anchors of its stored head and of its head after the
        transaction, so ending, erasing or re-heading it reaches every entity
        its head names (C.14).
        """
        anchored: set[str] = set()

        def head(item: Mapping[str, Any] | None, *, stored: bool) -> None:
            if item is not None:
                revision = (
                    self.get_row("revision", item["current_revision_id"]) if stored
                    else latest("revision", item["current_revision_id"])
                )
                if revision is not None:
                    anchored.update(row_anchors("revision", revision))

        for item in parsed:
            if item.kind not in _COMMIT_KINDS:
                continue
            after = item.row.model_dump(mode="json")
            before = self.get_row(item.kind, after["object_id"])
            anchored.update(row_anchors(item.kind, after))
            if before is not None:
                anchored.update(row_anchors(item.kind, before))
            if item.kind == "item":
                head(after, stored=False)
                head(before, stored=True)
        for kind, row in removed:
            anchored.update(row_anchors(kind, row))
            if kind == "item":
                head(row, stored=True)
        return frozenset(anchored)

    def _journal_rows(
        self, ticket: WriteTicket, journal: StrictModel, *, created_at: int,
    ) -> tuple[ArtifactRowV1, ...]:
        """Materialize a new journal's physical rows; the last one is its reference."""
        misplaced = placement_refusal("write_set", ticket.scope_id, "internal", journal)
        if misplaced is not None:
            raise AMLError("scope_denied", misplaced)
        rows = self._parsed_logical_artifact_rows(
            ticket, "write_set", journal, body_type=type(journal), created_at=created_at,
            security_level="internal", sizing=False,
        )
        for row in rows:
            if self.get_row("artifact", row.object_id) is not None:
                raise AMLError("artifact_digest_collision", "A journal row already exists")
            self._validate_refs("artifact", row, {})
        self._validate_body_references(journal, rows[-1])
        return rows

    def _plan_deletes(self, plan: Any, removed: Sequence[tuple[str, dict[str, Any]]]) -> None:
        """Put the deletes of stored rows first in a prepared plan, their Catalog after."""
        first, catalog = self._delete_operations(removed)
        for route, operation in catalog:
            plan.catalog_ops.setdefault(route, []).append(operation)
        # Deletes run first, so a re-derived row may reuse a freed unique key.
        plan.entity_ops["main"][:0] = first

    def _delete_operations(
        self, removed: Sequence[tuple[str, dict[str, Any]]],
    ) -> tuple[list[Any], list[tuple[str, Any]]]:
        """Prepare the main and Catalog RowOps that delete stored rows, writing nothing."""
        deletion = self._ws._writer.prepare_delete_rows([
            self._ws._row_inputs.delete_target((self.entities[kind], row["object_id"]))
            for kind, row in removed
        ])
        if deletion.deleted != [True] * len(removed):
            raise AMLError("parity_mismatch", "A deleted row disappeared before its commit")
        main = {self.entities[kind].schema().entity_id for kind, _ in removed}
        first: list[Any] = []
        catalog: list[tuple[str, Any]] = []
        for route, operations in self._ws._writer.delete_ops(deletion).items():
            for operation in operations:
                if operation.entity_id in main:
                    if route != "main":
                        raise AMLError("parity_mismatch", "A deleted row has another route")
                    first.append(operation)
                else:
                    catalog.append((route, operation))
        return first, catalog

    def _read_owned_business_artifact(
        self, ticket: WriteTicket, reference: str, kind: str, *, restricted: bool = False,
    ) -> StrictModel:
        """Read the exact producer's recovery artifact before accepting its marker.

        Body parity alone cannot establish Artifact ownership. These same
        creator, tenant and scope checks are required by recovery; accepting a
        foreign creator here would make a committed transaction unrecoverable.
        Logical manifests and their parts retain read_logical's full checks.
        """
        row = self.get_row("artifact", reference)
        if row is None or ticket.run_id is None or (
            row["created_by_run_id"] != ticket.run_id
            or row["tenant_id"] != self.tenant_id
            or (row["scope_kind"], row["scope_id"]) != (ticket.scope_kind, ticket.scope_id)
            or restricted and row["security_level"] != "restricted"
        ):
            raise AMLError("binding_mismatch", "Business recovery artifact has another owner")
        return self.read_logical(reference, kind)

    def materialize_control(
        self,
        key: str,
        value: object,
        version: int,
        updated_at: int,
        updated_by_run_id: str | None = None,
    ) -> ControlRowV1:
        """Derive a complete Control post-image once from its typed namespace value."""
        typed = parse_control_value(key, value)
        return ControlRowV1.model_validate(
            {
                "object_id": control_id(key),
                "key": key,
                "value": typed,
                "control_refs": [
                    ref.model_dump(mode="json")
                    for ref in self._projection_refs(extract_references(typed))
                ],
                "version": version,
                "updated_at": updated_at,
                "updated_by_run_id": updated_by_run_id,
            }
        )

    def compare_control(
        self,
        ticket: WriteTicket,
        intended: ControlRowV1,
        expected_version: int | None,
    ) -> bool:
        """Execute real single-row HB CAS after full materialization and admission."""
        return self._compare_control(ticket, intended, expected_version)

    def _compare_control(
        self, ticket: WriteTicket, intended: ControlRowV1, expected_version: int | None,
    ) -> bool:
        """Share actual CAS encoding for frozen cas_only intents.

        The session coordination lease and its close path left with PD:P2 at S0b.
        """
        with self._write_scope():
            self._admit(ticket)
            if ticket.run_id is None:
                raise AMLError("invalid_write_mode", "Public Control CAS requires a Run owner")
            run = self.get_row("run", ticket.run_id)
            if run is None or intended.updated_by_run_id != ticket.run_id:
                raise AMLError("binding_mismatch", "Control provenance differs from its caller")
            body = self._bound_input(run)
            if run["commit_mode"] != "cas_only" or "control_cas_intent" not in body:
                raise AMLError(
                    "invalid_write_mode", "Control CAS requires its frozen cas_only intent"
                )
            intent = body["control_cas_intent"]
            if intended.key == "commit_seq":
                current = self.get_control("commit_seq")
                if (
                    run["run_kind"] != "embedding_switch"
                    or body.get("action") != "bootstrap"
                    or current is None
                    or current["version"] != expected_version
                    or intended.value.model_dump() != {"seq": body["reserved_generation"]}
                    or body["reserved_generation"] != current["value"]["seq"] + 1
                ):
                    raise AMLError(
                        "binding_mismatch",
                        "Embedding reservation differs from its frozen generation",
                    )
            else:
                pre = intent["expected_pre"]
                if (
                    intended.key != intent["control_key"]
                    or canonical_json(intended) != canonical_json(intent["intended_post_image"])
                    or expected_version != (pre["version"] if pre["exists"] else None)
                ):
                    raise AMLError(
                        "binding_mismatch",
                        "CAS arguments differ from the durable full-row intent",
                    )
                current = self.get_control(intended.key)
                if (
                    pre["exists"]
                    and current is not None
                    and current["version"] == expected_version
                ):
                    if integrity_digest(current) != pre["digest"]:
                        raise AMLError(
                            "parity_mismatch", "CAS pre-state changed without a version change"
                        )
            if intended.version != (0 if expected_version is None else expected_version + 1):
                raise AMLError("parity_mismatch", "CAS post-image has the wrong next version")
            _, plan = self._prepare_rows([RowWrite("control", intended)], ticket)
            if len(plan.entity_ops["main"]) != 1:
                raise AMLError(
                    "service_unhealthy", "Control CAS does not encode exactly one main row"
                )
            self._ws._router._near.cache.invalidate(self.entities["control"].schema().entity_id)
            self._invalidate_pending_recall_references(plan)
            self.ref_generation += 1
            with self._catalog_steps():
                changed = self._mutate_storage(lambda: self._backend.compare_upsert(
                    plan.entity_ops["main"][0],
                    revision_field="version",
                    expected_revision=expected_version,
                ))
                if changed:
                    self._mutate_storage(lambda: self._ws._writer.write_ops(plan.catalog_ops))
            return cast(bool, changed)

    LOGICAL_MEMO_CAPACITY = 16384
    """Parsed logical Artifacts memoized between mutations of their own row.

    ``LOGICAL_MEMO_BYTES`` bounds their memory; this bounds their number.
    """

    LOGICAL_MEMO_BYTES = 32 * 1024 * 1024
    """Stored body bytes of the memoized Artifacts before the least recently used go.

    The charge is the stored body text an Artifact's parse read; parsed models
    retain substantially more objects than that text. A 976-position finalize's
    5,174 logical Artifacts charge 17 MB and hold about 140 MB once parsed.
    Limit this graph so later full collections do not repeatedly traverse the
    whole history. Eviction only causes a fresh validated read; it changes no
    durable rows or read authority. An Artifact whose charge exceeds an eighth
    of the budget is never memoized.
    """

    def read_logical(self, reference: str, expected_kind: str) -> StrictModel:
        """Load direct or sharded content with envelope, kind, and digest validation.

        Artifacts are immutable and content-addressed, so a parsed body is
        memoized per reference and kind and reused while no in-process
        mutation named the Artifact row since it was read (ADR 0036; an
        erasure retires the key). Verifying one address recanonicalizes the
        whole envelope, so an overflow forgets the least recently used entry
        instead of restarting the memo. Every Add reads the embedding canary policy
        this way, whose validation walks 32 vectors of every element; parsing
        it once per process instead of once per Add removed a fifth of the
        Add path's interpreter time. Each return is the caller's own shallow
        copy of the immutable model.
        """
        cached = self._logical_memoized(reference, expected_kind)
        if cached is not None:
            note_reads(cached[2])
            return cached[1].model_copy()
        if expected_kind == "semantic_embedding" and self.get_row("artifact", reference) is None:
            from aml.store.privacy_embeddings import retained_embedding

            retained = retained_embedding(self, reference)
            if retained is not None:
                return retained
        return self._logical_parsed(reference, expected_kind, self.get_row)

    def read_logical_observed(
        self, reference: str, expected_kind: str,
        read_row: Callable[[str, str], dict[str, Any] | None],
    ) -> StrictModel:
        """Load one logical Artifact through the caller's row reader, sharing the parse memo.

        ``read_logical`` memoizes a parsed body while no in-process mutation
        named any row its assembly read (ADR 0036). A caller whose reads carry
        their own checks -- a lifetime check, an exact-owner check, an observer
        recording proof keys -- reuses that body here without losing them: a
        served body is returned only after ``read_row`` has been called again
        for every row the original assembly read, in the order it read them, so
        each check and observer sees the reads a fresh parse would make. A
        replayed read that finds no row, or a row other than the stored one,
        abandons the memo and parses afresh through ``read_row``, so the caller
        fails or succeeds exactly as before. A miss parses through ``read_row``
        and memoizes the body only when every row it read is the stored row:
        a reader serving rows it captured earlier never seeds the shared memo.

        Args:
            reference: The logical Artifact's address.
            expected_kind: The logical kind the caller requires.
            read_row: The caller's physical reader; it may raise to refuse a row.

        Returns:
            The caller's own shallow copy of the immutable parsed model.

        Raises:
            AMLError: As the logical reader does, or as ``read_row`` refuses a row.
        """
        cached = self._logical_memoized(reference, expected_kind)
        if cached is not None and all(
            (row := read_row(kind, object_id)) is not None and row == self.get_row(kind, object_id)
            for kind, object_id in cached[2]
        ):
            return cached[1].model_copy()
        return self._logical_parsed(reference, expected_kind, read_row, stored_only=True)

    def release_erased_caches(self) -> None:
        """Release what an erasure left unprovable in this process (D.6 (3)).

        Called under the Writer mutex after the erasure's readers drained. An
        erasure mutates each row whose content it removes, and each mutation
        retires exactly that row's key, so every memoized scan, per-row value
        and parsed logical Artifact that read removed content is refused by its
        own recorded proof; this drops those entries so their payloads leave the
        process. Entries over untouched rows are what a lookup serves now and
        are kept. HeavenBase's near cache records no row dependency, so it is
        invalidated whole for every entity. Prepared Artifact encodings have no
        dependency ledger and are released whole, invalidating their tokens;
        tokens retain only content-free tickets and addresses. No label-scan
        cache exists in v1.

        Raises:
            AMLError: ``reader_busy`` without the Writer mutex.
        """
        if not self.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            raise AMLError("reader_busy", "Privacy cache release requires the Writer mutex")
        prepared = cast(WeakKeyDictionary[PreparedArtifact, _PreparedArtifactRows] | None,
                        self.__dict__.pop("_prepared_artifacts", None))
        if prepared is not None:
            prepared.clear()
        self._clear_artifact_preparation_admissions()
        for memo in (self.scan_memo(), self.binding_memo()):
            memo.release_stale()
        cache = self._rows_cache()
        logical = self._logical_memo_table()
        for key, (token, _body, keys, _size) in list(logical.items()):
            if cache.token != token and cache.changed_since(keys, token):
                released = logical.pop(key, None)
                if released is not None:
                    logical.size -= released[3]
        for entity in self.entities.values():
            self._ws._router._near.cache.invalidate(entity.schema().entity_id)

    def _logical_memo_table(self) -> _LogicalMemo:
        memo = self.__dict__.get("_logical_memo")
        if memo is None:
            memo = self.__dict__.setdefault("_logical_memo", _LogicalMemo())
        return cast(_LogicalMemo, memo)

    def _logical_memoized(
        self, reference: str, expected_kind: str,
    ) -> tuple[int, StrictModel, tuple[tuple[str, str], ...], int] | None:
        """Return a memoized parse whose rows no mutation has named since it was read."""
        memo = self._logical_memo_table()
        key = (reference, expected_kind)
        cached = memo.get(key)
        if cached is None or not self.borrowed_rows_unchanged(cached[2], cached[0]):
            return None
        # Most recently used; the eviction in _logical_parsed is oldest first.
        memo.pop(key, None)
        memo[key] = cached
        return cached

    def _logical_parsed(
        self, reference: str, expected_kind: str,
        read_row: Callable[[str, str], dict[str, Any] | None], *, stored_only: bool = False,
    ) -> StrictModel:
        """Parse through ``read_row``, memoizing the body with its reads in order.

        With ``stored_only`` the body is memoized only when every row
        ``read_row`` returned equals the row this Writer stores.
        """
        from aml.store.artifact_read import read_logical_artifact

        token = self.storage_epoch
        read: list[tuple[str, str]] = []
        rows: list[dict[str, Any] | None] = []

        def recorded(kind: str, object_id: str) -> dict[str, Any] | None:
            # Every row the assembly touched, manifest and parts alike, names
            # the memo's proof: a part rewritten later retires the entry. The
            # order is kept so an observed reader can replay the same reads.
            read.append((kind, object_id))
            row = read_row(kind, object_id)
            rows.append(row)
            return row

        body = read_logical_artifact(recorded, reference, expected_kind, frozenset())
        if stored_only and any(
            row != self.get_row(kind, object_id)
            for (kind, object_id), row in zip(read, rows, strict=True)
        ):
            return body.model_copy()
        size = sum(len(row["body"]) for row in rows
                   if row is not None and isinstance(row.get("body"), str))
        if size > self.LOGICAL_MEMO_BYTES // 8:
            return body.model_copy()
        memo = self._logical_memo_table()
        replaced = memo.pop((reference, expected_kind), None)
        if replaced is not None:
            memo.size -= replaced[3]
        while memo and (
            len(memo) >= self.LOGICAL_MEMO_CAPACITY or memo.size + size > self.LOGICAL_MEMO_BYTES
        ):
            # Forget the least recently used entry rather than every entry: a
            # finalize cycle reads its plan blocks and run inputs repeatedly
            # while other scopes read theirs, and clearing the whole memo made
            # each of them verify and parse again what the other had just read.
            forgotten = memo.pop(next(iter(memo)), None)
            if forgotten is not None:
                memo.size -= forgotten[3]
        memo[(reference, expected_kind)] = (token, body, tuple(read), size)
        memo.size += size
        return body.model_copy()

    @staticmethod
    def _artifact_envelope(row: ArtifactRowV1) -> dict[str, Any]:
        from aml.store.artifact_read import artifact_envelope

        return artifact_envelope(row)

    def _verify_artifact_identity(self, row: ArtifactRowV1) -> None:
        from aml.store.artifact_read import verify_artifact_identity

        verify_artifact_identity(row)

    def _verify_artifact_refs(
        self, row: ArtifactRowV1, body: StrictModel, parts: Sequence[str] = (),
    ) -> None:
        from aml.store.artifact_read import verify_artifact_refs

        verify_artifact_refs(row, body, parts)

    def put_artifact(
        self,
        ticket: WriteTicket,
        kind: str,
        body: object,
        *,
        security_level: str = "internal",
        created_at: int,
    ) -> ArtifactId:
        """Prewrite one immutable strict Artifact; oversize producers must explicitly shard."""
        if kind in {"part", "manifest"}:
            raise AMLError(
                "invalid_write_mode", "Physical fragments require a validated logical producer"
            )
        typed = parse_artifact_body(
            kind, body.model_dump(mode="json") if isinstance(body, StrictModel) else body
        )
        return self._put_physical(ticket, kind, typed, security_level, created_at)

    def put_forget_receipt(
        self, ticket: WriteTicket, kind: str, body: StrictModel, *, created_at: int,
    ) -> ArtifactId:
        """Write one D.3 receipt-pair Artifact under the controller's held erasure fence.

        The write proof (``forget_run.check_controller_artifact``) requires the
        controller Run that names the fence, the ``scanned`` phase, and a final
        receipt whose ``final_scan_digest`` is the fence's bound scan digest.

        Args:
            ticket: The erasure controller's ticket.
            kind: ``receipt:forget_prepare``, ``receipt:forget_final`` or
                ``result_index:erasure_receipt``.
            body: The typed receipt.
            created_at: The trusted write time.

        Returns:
            The stored Artifact id.

        Raises:
            AMLError: ``invalid_write_mode`` without the owner proof.
        """
        physical = kind.split(":", 1)[0]
        typed = parse_artifact_body(physical, body.model_dump(mode="json"))
        if type(typed) is not type(body):
            raise AMLError("invalid_write_mode", "The receipt kind differs from its body")
        return self._put_physical(
            ticket, physical, typed, "restricted", created_at, forget_receipt=True,
        )

    def _materialize_artifact_row(
        self,
        ticket: WriteTicket,
        kind: str,
        typed: StrictModel,
        security_level: str,
        created_at: int,
        *,
        logical: StrictModel | None = None,
        parts: Sequence[str] = (),
        sizing: bool = False,
        _references: Callable[[], tuple[Reference, ...]] | None = None,
        _serialize: Callable[[], str] | None = None,
    ) -> ArtifactRowV1:
        """Share exact envelope and derived-column construction between preflight and put-once.

        A sizing row (``sizing=True``) is never stored: it carries a sizing
        address derived from the creator and a per-Writer sequence instead of
        the canonical envelope address, so a capacity measurement does not
        hash every body it measures. Every other field, the body bytes and
        the byte count are exactly the stored row's, so the StoragePlan routes
        and counts it identically; the put-once path derives the real address.
        """
        if ticket.run_id is None:
            raise AMLError("invalid_write_mode", "Artifact creator must be an admitted Run")
        serialized = canonical_json(typed) if _serialize is None else _serialize()
        if len(serialized) > MAX_BODY_CHARS:
            raise AMLError("artifact_too_large", "Artifact body exceeds the inline character limit")
        source = logical if logical is not None else typed
        from aml.core.artifacts import ReceiptResultIndexV1
        from aml.core.extraction_records import ErasedExtractionV1
        from aml.core.ingestion_cleanup import CleanupRunInput
        from aml.core.privacy_embeddings import PrivacyEmbeddingV1
        from aml.core.privacy_persistence import IngestionErasurePlanV1, RedactionReceiptV1

        owner = self.get_row("run", ticket.run_id)
        # Memory forget and its erasure controller went at S0b (S6a rebuilds them).
        if owner is not None and owner["run_kind"] == "forget":
            raise AMLError("invalid_write_mode", "Memory erasure can prewrite only its plan blocks")
        if owner is not None and owner["run_kind"] == "erasure_controller":
            # S6a: the controller writes exactly its D.3 receipt pair and their selection.
            from aml.store.forget_run import check_controller_artifact

            check_controller_artifact(self, ticket, kind, security_level, source)
        if (
            owner is not None
            and owner["run_kind"] == "retention"
            and owner["target_kind"] == "affected_set"
        ):
            if not isinstance(
                source,
                IngestionErasurePlanV1
                | RedactionReceiptV1
                | CleanupRunInput
                | ReceiptResultIndexV1
                | ErasedExtractionV1
                | PrivacyEmbeddingV1,
                # Erasure accounting and surviving vectors, never a new inference.
            ):
                raise AMLError(
                    "invalid_write_mode", "Ingestion cleanup cannot produce semantic artifacts"
                )
            if security_level != "restricted":
                raise AMLError("scope_denied", "Cleanup artifacts require restricted protection")
            if isinstance(source, ReceiptResultIndexV1):
                receipt = self.get_row("artifact", source.receipt_ref)
                if receipt is None or receipt["created_by_run_id"] != ticket.run_id:
                    raise AMLError("binding_mismatch", "Cleanup result requires its own receipt")

        if isinstance(source, IngestionErasurePlanV1) and (
            ticket.scope_id is None or security_level != "restricted"
        ):
            raise AMLError(
                "scope_denied", "Ingestion erasure plans require a restricted scoped owner"
            )
        references = contextual_references(source) if _references is None else _references()
        body_refs = self._projection_refs(
            [ref for ref in references if ref.spec.target_kind != "artifact"]
        )
        artifacts = (
            sorted(
                {ref.target_id for ref in references if ref.spec.target_kind == "artifact"}
                | set(parts)
            )
            if kind != "part"
            else []
        )
        envelope = {
            "artifact_kind": kind,
            "schema_version": "1",
            "tenant_id": self.tenant_id,
            "scope_kind": ticket.scope_kind,
            "scope_id": ticket.scope_id,
            "security_level": security_level,
            "body_refs": [ref.model_dump(mode="json") for ref in body_refs],
            "created_by_run_id": ticket.run_id,
            "body": serialized,
        }
        if sizing:
            sequence = self.__dict__.setdefault("_sizing_sequence", itertools.count())
            address = artifact_id({"sizing": ticket.run_id, "kind": kind, "seq": next(sequence)})
        else:
            address = artifact_id(envelope)
        # The address is derived from this envelope (or is a sizing address); the
        # model validator need not hash the envelope again (ArtifactRowV1.validate_address).
        return ArtifactRowV1.model_validate(
            {
                **envelope,
                "object_id": address,
                "artifact_refs": artifacts,
                "size_bytes": len(serialized.encode("utf-8")),
                "created_at": created_at,
            },
            context={"artifact_address": "derived"},
        )

    def materialize_logical_artifact_rows(
        self, ticket: WriteTicket, kind: str, body: object, *, created_at: int,
        security_level: str = "restricted", sizing: bool = False,
    ) -> tuple[StrictModel, tuple[ArtifactRowV1, ...]]:
        """Construct exact prospective rows through the existing logical encoder.

        The final row identifies the logical Artifact. This read-only operation
        grants no admission, reference validation or write authority; callers
        must independently authorize an actual creator and all stored effects.
        With ``sizing`` the rows carry sizing addresses instead of canonical
        ones: byte-identical for routing and counting, never stored, and not
        comparable with stored rows (capacity measurement only).
        """
        if kind in {"part", "manifest"}:
            raise AMLError("invalid_write_mode", "Preflight requires a logical producer")
        typed = parse_artifact_body(
            kind, body.model_dump(mode="json") if isinstance(body, StrictModel) else body,
        )
        return typed, self._parsed_logical_artifact_rows(
            ticket, kind, typed, body_type=type(body), created_at=created_at,
            security_level=security_level, sizing=sizing,
        )

    def _parsed_logical_artifact_rows(
        self, ticket: WriteTicket, kind: str, typed: StrictModel, *,
        body_type: type[object], created_at: int, security_level: str, sizing: bool,
        text: str | None = None,
    ) -> tuple[ArtifactRowV1, ...]:
        """Finish this call's freshly parsed body through the sole physical codec."""
        references: tuple[Reference, ...] | None = None

        def logical_text() -> str:
            # The first call stays at the codec's original serialization point.
            # Only an inline row consumes the same detached body's text again.
            nonlocal text
            if text is None:
                text = canonical_json(typed)
            return text

        # Only the exact parsed model's JSON dump proves detachment here. Raw dicts
        # can contain caller-owned models, and subclasses can override model_dump.
        serialize = logical_text if body_type is type(typed) else None

        def logical_references() -> tuple[Reference, ...]:
            # This parsed body is private to one read-only materialization. Resolve
            # it after the first physical row's existing owner/role checks, then
            # reuse the immutable projection for its other parts and manifest.
            nonlocal references
            if references is None:
                references = contextual_references(typed)
            return references

        return tuple(self._materialize_artifact_row(
            ticket, physical_kind, physical, security_level, created_at,
            logical=typed if physical_kind in {"part", "manifest"} else None, parts=parts,
            sizing=sizing, _references=logical_references,
            _serialize=serialize if physical is typed else None,
        ) for physical_kind, physical, parts in self._logical_fragments(
            ticket, kind, typed, security_level, created_at, sizing=sizing,
            _references=logical_references, _serialize=serialize,
        ))

    def _artifact_capacity_rowops_batch(
        self, rows: Sequence[ArtifactRowV1], *, borrowed: bool = False, sizing: bool = False,
    ) -> int:
        """Measure every row of one body through the actual StoragePlan in one scope.

        The rows' stored-bytes collision checks read the memoized rows, with
        one batched storage read for the misses, and need no HeavenBase scope.
        The rows are then materialized and routed together
        by one ``prepare_upsert_rows`` call, so a body of N physical rows takes
        HeavenBase's shared lock once instead of N times while every row keeps
        its own materializer validation and main-only routing proof (plan
        sections 4 and 6). Sizing row by row under that lock held it 11-20 ms
        per row under load and stalled every committer (harness X2, 2026-09-11:
        53% of the lock's read time). A borrowed measurement takes the
        measurement scope itself; a mutex-held one runs inside the caller's
        write scope.
        """
        if borrowed:
            self.probe_storage_owner()
        else:
            self.validate_storage_owner()
        # Sizing rows carry sizing addresses (never stored), so there is nothing
        # to compare; the put-once path derives the real address and checks it.
        existing_rows = (
            [None] * len(rows) if sizing
            else self.get_rows("artifact", [row.object_id for row in rows])
        )
        for row, existing in zip(rows, existing_rows, strict=True):
            if existing is not None:
                old = ArtifactRowV1.model_validate(existing)
                if canonical_json(self._artifact_envelope(old)) != canonical_json(
                    self._artifact_envelope(row),
                ) or old.artifact_refs != row.artifact_refs or old.size_bytes != row.size_bytes:
                    raise AMLError(
                        "artifact_digest_collision",
                        "Artifact sizing found conflicting stored bytes",
                    )
        if not rows:
            return 0
        if borrowed:
            with self._measure_scope():
                return self._size_artifact_rows(rows)
        return self._size_artifact_rows(rows)

    def _size_artifact_rows(self, rows: Sequence[ArtifactRowV1]) -> int:
        """Count actual StoragePlan RowOps, including never-written sizing addresses.

        This private measurement may use the materializer's sizing address. It
        only drops the two omitted physical fields; the resulting plan is never
        returned or committed. Real prewrites use the fully validated encoder.
        """
        from aml.store.normalization import REMOVED_FIELDS

        entity = self.entities["artifact"]
        inputs = [
            self._ws._row_inputs.upsert_row(entity, row.model_dump(
                mode="json",
                exclude=set(REMOVED_FIELDS["artifact"]),
            ), {})
            for row in rows
        ]
        plan = self._ws._writer.prepare_upsert_rows(
            inputs, overwrite=True, catalog_mode="fresh",
        )
        if set(plan.entity_ops) != {"main"}:
            raise AMLError("invalid_write_mode", "Artifact preflight is not routed to main")
        return len(plan.entity_ops["main"])

    def logical_artifact_rowops(
        self,
        ticket: WriteTicket,
        kind: str,
        body: object,
        *,
        created_at: int,
        security_level: str = "restricted",
    ) -> int:
        """Preflight all physical rows of one logical Artifact without persisting any part.

        Args:
            ticket: Its already admitted creator and exact scope.
            kind: Registered logical kind, never a raw part or manifest.
            body: Strict complete logical content before any physical fragmentation.
            created_at: Fixed producer time used by the actual put operation.
            security_level: Exact envelope classification used by the actual put.

        Returns:
            Actual aggregate main RowOps, including every complete part/manifest projection.

        Raises:
            AMLError: If schema, creator, routing, or permitted sharding is invalid.
        """
        _, physical = self.materialize_logical_artifact_rows(
            ticket, kind, body, created_at=created_at, security_level=security_level, sizing=True,
        )
        with self._write_scope():
            self._admit(ticket)
            return self._artifact_capacity_rowops_batch(physical, sizing=True)

    def _put_physical(
        self,
        ticket: WriteTicket,
        kind: str,
        typed: StrictModel,
        security_level: str,
        created_at: int,
        *,
        logical: StrictModel | None = None,
        parts: Sequence[str] = (),
        privacy_completion: bool = False,
        action_check: Callable[[WriteTicket], None] | None = None,
        forget_receipt: bool = False,
        _references: Callable[[], tuple[Reference, ...]] | None = None,
        _serialize: Callable[[], str] | None = None,
    ) -> ArtifactId:
        with self._write_scope():
            source = logical if logical is not None else typed
            # The resume, Subject-failure, reextract-failure, source-label recovery,
            # forget, erasure and maintenance write roles went with their owners at S0b.
            self._admit(ticket)
            self._check_artifact_source(
                ticket, source, privacy_completion=privacy_completion,
                forget_receipt=forget_receipt,
            )
            walked: list[tuple[Reference, ...]] = []

            def references() -> tuple[Reference, ...]:
                # One walk of the body serves its row's projections and its
                # reference proof, at the point the row first needs it.
                if not walked:
                    walked.append(contextual_references(source) if _references is None
                                  else _references())
                return walked[0]

            row = self._materialize_artifact_row(
                ticket,
                kind,
                typed,
                security_level,
                created_at,
                logical=logical,
                parts=parts,
                _references=references,
                _serialize=_serialize,
            )
            envelope = self._artifact_envelope(row)
            address = row.object_id
            self._validate_refs("artifact", row, {})
            existing = self.get_row("artifact", address)
            if existing is not None:
                old = ArtifactRowV1.model_validate(existing)
                if canonical_json(self._artifact_envelope(old)) != canonical_json(envelope):
                    raise AMLError(
                        "artifact_digest_collision",
                        "Existing Artifact has different envelope bytes",
                    )
                if old.artifact_refs != row.artifact_refs or old.size_bytes != row.size_bytes:
                    raise AMLError(
                        "parity_mismatch", "Existing Artifact derived columns are inconsistent"
                    )
                if action_check is not None:
                    action_check(ticket)
                return address
            self._validate_body_references(source, row, _references=references())
            inputs = self._prepare_row_inputs([RowWrite("artifact", row)])
            plan = self._ws._writer.prepare_upsert_rows(
                inputs, overwrite=False, catalog_mode="fresh"
            )
            if set(plan.entity_ops) != {"main"} or len(plan.entity_ops["main"]) > MAX_TX_ROWS:
                raise AMLError(
                    "budget_exceeded", "Artifact materialization exceeds the main-row budget"
                )
            owner = self.get_row("run", row.created_by_run_id)
            if owner is not None and owner.get("input_ref") is not None:
                frozen = self._bound_input(owner)
                if "artifact_rowops" in frozen:
                    from aml.store.indexed import IndexedReader

                    reader = IndexedReader(
                        self._ws,
                        self.entities,
                        tenant_id=self.tenant_id,
                    )
                    count, after = 0, None
                    while identifiers := reader.artifact_ids(row.created_by_run_id, after=after):
                        count += len(identifiers)
                        if count > frozen["artifact_rowops"]:
                            raise AMLError("budget_exceeded", "Stored slice exceeds its row budget")
                        after = identifiers[-1]
                    if len(plan.entity_ops["main"]) != 1:
                        raise AMLError("schema_mismatch", "Artifact budget requires one main owner")
                    if count + len(plan.entity_ops["main"]) > frozen["artifact_rowops"]:
                        raise AMLError(
                            "budget_exceeded", "Artifact writes exceed the frozen slice budget"
                        )
            # Existence was checked under this same exclusive write scope. An
            # exception after main commit is an uncertain acknowledgement, never
            # a create-once hit; propagate it so recovery can repair the Catalog.
            if action_check is not None:
                action_check(ticket)
            self._artifact_generations[row.created_by_run_id] = (
                self._artifact_generations.get(row.created_by_run_id, 0) + 1
            )
            self._write_plan(plan)
            return address

    def _check_artifact_source(
        self, ticket: WriteTicket, source: StrictModel, *, privacy_completion: bool,
        forget_receipt: bool,
    ) -> None:
        """Refuse a body that only its dedicated producer may write."""
        from aml.core.artifacts import ReceiptResultIndexV1
        from aml.core.forget import ForgetPrepareReceiptV3, ForgetReceiptV3
        from aml.core.forget_receipts import ErasureReceiptResultIndexV1
        from aml.core.privacy_persistence import RedactionReceiptV1

        if isinstance(
            source, ErasureReceiptResultIndexV1 | ForgetPrepareReceiptV3 | ForgetReceiptV3,
        ) and not forget_receipt:
            raise AMLError(
                "invalid_write_mode", "Erasure receipts need their actual owner proof",
            )
        # Session Subject-context and reextract failure results had only the
        # settlements S0b removed as producers; head-stale outputs had only the
        # Candidate retention it removed. A label result needs its running child.
        from aml.core.artifacts import SourceLabelsResultIndexV1

        if isinstance(source, SourceLabelsResultIndexV1):
            from aml.store.source_labels import validate_source_labels_result

            validate_source_labels_result(self, ticket, source)
        if hasattr(source, "subject_context_failure"):
            raise AMLError(
                "invalid_write_mode", "Session conflict requires issued source settlement",
            )
        if hasattr(source, "reextract_failure"):
            raise AMLError(
                "invalid_write_mode", "Finalize conflict requires issued source settlement",
            )
        creator = self.get_row("run", ticket.run_id or "")
        if creator is not None and (creator["run_kind"], creator["target_kind"]) == (
            "retention", "candidate",
        ):
            raise AMLError(
                "invalid_write_mode", "Head-stale outputs require their complete atomic graph",
            )
        if (
            isinstance(source, RedactionReceiptV1 | ReceiptResultIndexV1)
            and not privacy_completion
        ):
            raise AMLError(
                "invalid_write_mode", "A verified cleanup receipt requires final verification"
            )

    def _validate_body_references(
        self, body: StrictModel, envelope: ArtifactRowV1, *,
        _post: dict[tuple[str, str], StrictModel] | None = None,
        _references: tuple[Reference, ...] | None = None,
        _parsed: dict[tuple[str, str], StrictModel] | None = None,
    ) -> None:
        from aml.store.indexed import BatchedHeaders

        post = {} if _post is None else _post
        owner_scope = self._scope("artifact", envelope.model_dump(mode="json"), post)
        raw = body.model_dump(mode="json")
        if envelope.artifact_kind == "run_input" or raw.get("run_kind") is not None:
            owner = self._lookup("run", envelope.created_by_run_id, post)
            if (
                owner is None
                or raw["run_kind"] != owner["run_kind"]
                or raw["admission_digest"] != owner["admission_digest"]
            ):
                raise AMLError(
                    "binding_mismatch", "Input artifact does not match its creator admission"
                )
        references = contextual_references(body) if _references is None else _references
        resolved = [ref for ref in references if not (
            ref.spec.ref_mode == "historical_locator"
            and ref.spec.existence_expectation != "must_exist"
        )]
        # One body can name hundreds of Evidence locators: each member of a
        # finalize source inventory block names its source twice, which were
        # 1,128 point queries in one Add's freeze of a long replay.
        evidence = BatchedHeaders(
            lambda identifiers: self._reference_reader().evidence_reference_headers(identifiers),
            lambda identifier: self._reference_reader().evidence_reference_header(identifier),
            (ref.target_id for ref in resolved if ref.spec.target_kind == "evidence"),
        )
        for ref in resolved:
            spec = ref.spec
            target = self._reference_lookup(
                spec.target_kind, ref.target_id, post,
                pre_only=spec.fulfillment == "pre_state", evidence=evidence,
            )
            if target is None:
                raise AMLError("reference_missing", "Artifact body reference does not exist")
            target_scope = self._scope(spec.target_kind, target, post)
            if target.get("tenant_id", self.tenant_id) != self.tenant_id:
                raise AMLError("tenant_mismatch", "Artifact body references a different tenant")
            if (
                spec.scope_rule == "same_scope"
                and target_scope != owner_scope
                or (
                    spec.scope_rule == "same_scope_or_global"
                    and target_scope not in (None, owner_scope)
                )
                or spec.scope_rule == "global"
                and target_scope is not None
            ):
                raise AMLError("scope_denied", "Artifact body reference crosses its allowed scope")
            if spec.target_kind == "artifact" and spec.artifact_kinds:
                kind = target["artifact_kind"]
                direct = kind != "manifest"
                if not direct:
                    kind = json.loads(target["body"])["logical_kind"]
                if kind not in spec.artifact_kinds:
                    raise AMLError(
                        "reference_kind_mismatch", "Artifact body target has the wrong kind"
                    )
                if direct and ("artifact", ref.target_id) not in post:
                    # A stored direct Artifact was read through the restore codec,
                    # which proved its address, its strict body and its reference
                    # projections: reading its logical body again proves nothing more.
                    continue
                self._read_logical_post(ref.target_id, kind, post, _parsed)

    def _read_logical_post(
        self, reference: str, kind: str, post: dict[tuple[str, str], StrictModel],
        parsed: dict[tuple[str, str], StrictModel] | None = None,
    ) -> StrictModel:
        """Use the existing logical reader over an exact prepared union view.

        A batch creates only addresses absent from storage, so a stored target
        is read from exactly the stored rows the union view would serve: it goes
        through ``read_logical`` and its memo. A target the batch creates is
        parsed from the batch's rows, once per batch and kind when the caller
        passes ``parsed``.
        """
        if not post or ("artifact", reference) not in post and (
            self.get_row("artifact", reference) is not None
        ):
            return self.read_logical(reference, kind)
        key = (reference, kind)
        if parsed is not None and key in parsed:
            return parsed[key]
        from aml.store.artifact_read import read_logical_artifact

        body = read_logical_artifact(
            lambda target, identifier: self._lookup(target, identifier, post), reference, kind,
        )
        if parsed is not None:
            parsed[key] = body
        return body

    def _logical_fragments(
        self, ticket: WriteTicket, kind: str, typed: StrictModel,
        security_level: str, created_at: int, *, sizing: bool = False,
        _references: Callable[[], tuple[Reference, ...]] | None = None,
        _serialize: Callable[[], str] | None = None,
    ) -> Iterator[tuple[str, StrictModel, tuple[str, ...]]]:
        """Materialize the one canonical direct/part/manifest encoding without writes.

        ``sizing`` gives the parts sizing addresses (see ``_materialize_artifact_row``);
        the manifest then names those, which keeps its byte count exact.
        """
        from aml.core.artifacts import SHARDABLE_KINDS

        text = canonical_json(typed) if _serialize is None else _serialize()
        if len(text) <= MAX_BODY_CHARS:
            yield kind, typed, ()
            return
        if kind not in SHARDABLE_KINDS:
            raise AMLError("artifact_too_large", "This logical kind is not authorized for sharding")
        payload = text.encode("utf-8")
        parts: list[str] = []
        for offset in range(0, len(payload), 180000):
            part = parse_artifact_body("part", {
                "encoding": "base64",
                "data": base64.b64encode(payload[offset:offset + 180000]).decode("ascii"),
            })
            row = self._materialize_artifact_row(
                ticket, "part", part, security_level, created_at, logical=typed, sizing=sizing,
                _references=_references,
            )
            parts.append(row.object_id)
            yield "part", part, ()
        manifest = parse_artifact_body("manifest", {
            "logical_kind": kind, "schema_version": "1",
            "body_digest": hashlib.sha256(payload).hexdigest(),
            "parts": parts, "total_bytes": len(payload),
        })
        yield "manifest", manifest, tuple(parts)

    def put_logical_artifact(
        self,
        ticket: WriteTicket,
        kind: str,
        body: object,
        *,
        security_level: str = "internal",
        created_at: int,
    ) -> ArtifactId:
        """Validate complete logical content before emitting any bounded physical parts."""
        typed = parse_artifact_body(
            kind, body.model_dump(mode="json") if isinstance(body, StrictModel) else body
        )
        if kind in {"part", "manifest"}:
            raise AMLError("invalid_write_mode", "Physical fragments require a logical producer")
        # Ask where this body may live before deciding how to store it. Once a
        # body is split, the rows that carry it are named `manifest` and `part`,
        # and the envelope validator matches its placement rules against the
        # row's own kind -- so those rules would stop applying to precisely the
        # artifacts large enough to be split.
        misplaced = placement_refusal(kind, ticket.scope_id, security_level, typed)
        if misplaced is not None:
            raise AMLError("scope_denied", misplaced)
        text: str | None = None
        references: tuple[Reference, ...] | None = None

        def logical_text() -> str:
            nonlocal text
            if text is None:
                text = canonical_json(typed)
            return text

        def logical_references() -> tuple[Reference, ...]:
            # Every fragment represents the same detached typed body. Its
            # projection walk stays after the physical row's role checks and
            # is shared only for this one put, never across stored reads.
            nonlocal references
            if references is None:
                references = contextual_references(typed)
            return references

        # A strict body's JSON dump detached its nested values before parsing.
        # Raw dictionaries can instead retain caller-owned model instances.
        detached = type(body) is type(typed)
        reference: ArtifactId | None = None
        for physical_kind, physical, parts in self._logical_fragments(
            ticket, kind, typed, security_level, created_at,
            _references=logical_references if detached else None,
            _serialize=logical_text if detached else None,
        ):
            reference = self._put_physical(
                ticket, physical_kind, physical, security_level, created_at,
                logical=typed if physical_kind in {"part", "manifest"} else None, parts=parts,
                _references=logical_references if detached else None,
                _serialize=logical_text if detached and physical is typed else None,
            )
        assert reference is not None
        return reference

    def prepare_artifact(
        self, ticket: WriteTicket, kind: str, body: StrictModel, *, created_at: int,
        security_level: str = "internal",
    ) -> PreparedArtifact:
        """Detach and encode one reply Artifact under its current ticket, without writes.

        Args:
            ticket: Prospective creator and scope, rechecked at commit.
            kind: Logical Artifact kind; physical fragments are refused.
            body: The complete body, detached before its address is derived.
            created_at: Prospective creation time; commit supplies the stored time.
            security_level: The Artifact's required protection.

        Returns:
            An opaque encoding and its logical reference for ``commit_artifacts``.

        Raises:
            AMLError: The ticket's current fence or the logical encoder's refusal.
        """
        with self._write_scope():
            if not self._artifact_preparation_admitted(ticket):
                owner = self._admit(ticket)
                self._remember_artifact_preparation_admission(ticket, owner)
            typed, physical = self.materialize_logical_artifact_rows(
                ticket, kind, body, created_at=created_at, security_level=security_level,
            )
            prepared = PreparedArtifact(ticket, ArtifactId(physical[-1].object_id))
            pending = cast(WeakKeyDictionary[PreparedArtifact, _PreparedArtifactRows],
                           self.__dict__.setdefault("_prepared_artifacts", WeakKeyDictionary()))
            pending[prepared] = _PreparedArtifactRows(ticket, kind, typed, security_level, physical)
            return prepared

    def commit_artifacts(
        self,
        ticket: WriteTicket,
        artifacts: Sequence[tuple[WriteTicket, str, StrictModel, str] | PreparedArtifact],
        rows: Sequence[RowWrite] = (),
        *,
        created_at: int,
    ) -> tuple[ArtifactId, ...]:
        """Put logical Artifacts and their owner's Run and Control rows in one transaction.

        A judge reply is stored this way: its call manifest, every accepted
        member's output and result index, the children's finishes and their
        output locators are durable together or not at all (C.8 #8, C.13).
        Each Artifact is the put-once logical Artifact ``put_logical_artifact``
        writes for its creator, the ticket's Run or one of its child Runs: an
        equal stored one is kept, and every proof reads the batch's own
        Artifacts and rows. A creator with a frozen Artifact row budget is
        refused; ``_put_physical`` counts that budget row by row.

        Args:
            ticket: The admitted owner of the transaction and of ``rows``.
            artifacts: Per Artifact: its creator's ticket, logical kind, strict
                body and security level, or this Writer's private prepared
                encoding. Prepared rows retain their exact address and body;
                their stored creation time is ``created_at``.
            rows: Run and Control post-images of the same transaction.
            created_at: The Artifacts' creation time.

        Returns:
            Each Artifact's logical reference, in order.

        Raises:
            AMLError: ``invalid_write_mode`` for a physical kind or a creator
                outside the owner's Runs; ``scope_denied`` for a misplaced
                body; ``artifact_digest_collision`` for other stored bytes at
                an Artifact's address; ``budget_exceeded`` beyond
                ``MAX_TX_ROWS``; or the first failed row or reference proof.
        """
        with self._write_scope():
            self._admit(ticket)
            references: list[ArtifactId] = []
            bodies: list[tuple[StrictModel, ArtifactRowV1]] = []
            created: dict[str, ArtifactRowV1] = {}
            for artifact in artifacts:
                body: StrictModel | None = None
                if isinstance(artifact, PreparedArtifact):
                    # A privacy scrub releases the private rows while the token
                    # keeps this content-free ticket, so its current execution
                    # fence remains the first refusal after erasure or takeover.
                    self._admit(artifact.creator)
                    pending = cast(WeakKeyDictionary[PreparedArtifact, _PreparedArtifactRows],
                                   self.__dict__.get("_prepared_artifacts", WeakKeyDictionary()))
                    prepared = pending.get(artifact)
                    if prepared is None:
                        raise AMLError(
                            "binding_mismatch", "Artifact encoding was not issued by this Writer",
                        )
                    creator, kind, security_level = (
                        prepared.creator, prepared.kind, prepared.security_level,
                    )
                    typed = prepared.body
                    physical = tuple(row.model_copy(update={"created_at": created_at})
                                     for row in prepared.rows)
                else:
                    creator, kind, body, security_level = artifact
                    typed = None
                    physical = None
                run = self.get_row("run", creator.run_id or "")
                if kind in {"part", "manifest"} or run is None or ticket.run_id not in (
                    run["object_id"], run["parent_run_id"],
                ) or run["input_ref"] is not None and "artifact_rowops" in self._bound_input(run):
                    raise AMLError(
                        "invalid_write_mode", "Batched Artifacts belong to the owner's Runs",
                    )
                if not isinstance(artifact, PreparedArtifact):
                    self._admit(creator)
                if typed is None:
                    assert body is not None
                    typed = parse_artifact_body(kind, body.model_dump(mode="json"))
                self._check_artifact_source(
                    creator, typed, privacy_completion=False, forget_receipt=False,
                )
                misplaced = placement_refusal(kind, creator.scope_id, security_level, typed)
                if misplaced is not None:
                    raise AMLError("scope_denied", misplaced)
                if physical is None:
                    physical = self._parsed_logical_artifact_rows(
                        creator, kind, typed, body_type=type(typed), created_at=created_at,
                        security_level=security_level, sizing=False,
                    )
                for row in physical:
                    stored = self.get_row("artifact", row.object_id)
                    if stored is None:
                        created[row.object_id] = row
                        continue
                    old = ArtifactRowV1.model_validate(stored)
                    if canonical_json(self._artifact_envelope(old)) != canonical_json(
                        self._artifact_envelope(row),
                    ) or (old.artifact_refs, old.size_bytes) != (row.artifact_refs, row.size_bytes):
                        raise AMLError(
                            "artifact_digest_collision", "Existing Artifact has different bytes",
                        )
                bodies.append((typed, physical[-1]))
                references.append(ArtifactId(physical[-1].object_id))
            post: dict[tuple[str, str], StrictModel] = {
                ("artifact", address): row for address, row in created.items()
            }
            parsed = self._validate_rows(rows, ticket, artifacts=post) if rows else []
            post.update(
                ((item.kind, cast(str, item.row.model_dump()["object_id"])), item.row)
                for item in parsed
            )
            for row in created.values():
                self._validate_refs("artifact", row, post)
            # The batch's own targets (a call manifest its result indexes share) are
            # parsed once per batch; stored targets go through the read memo.
            targets: dict[tuple[str, str], StrictModel] = {}
            for typed, row in bodies:
                self._validate_body_references(typed, row, _post=post, _parsed=targets)
            plan = self._plan_rows(parsed) if parsed else None
            if created:
                # The Artifacts come first: the rows reference them.
                artifact_plan = self._ws._writer.prepare_upsert_rows(
                    self._prepare_row_inputs([RowWrite("artifact", row)
                                              for row in created.values()]),
                    overwrite=False, catalog_mode="fresh",
                )
                if set(artifact_plan.entity_ops) != {"main"}:
                    raise AMLError("parity_mismatch", "Batched Artifacts left the main backend")
                if plan is None:
                    plan = artifact_plan
                else:
                    plan.entity_ops["main"][:0] = artifact_plan.entity_ops["main"]
                    for route, operations in artifact_plan.catalog_ops.items():
                        plan.catalog_ops.setdefault(route, []).extend(operations)
            if plan is None:
                return tuple(references)
            if len(plan.entity_ops["main"]) > MAX_TX_ROWS:
                raise AMLError("budget_exceeded", "The Artifact transaction exceeds its budget")
            for row in created.values():
                self._artifact_generations[row.created_by_run_id] = (
                    self._artifact_generations.get(row.created_by_run_id, 0) + 1
                )
            self._write_plan(plan)
            return tuple(references)

    def find_logical_candidates(
        self, creator: str, kind: str
    ) -> list[tuple[ArtifactId, StrictModel]]:
        """Read complete creator-bound logical artifacts in stable indexed pages.

        Each page releases HB read ownership before logical validation. This is
        not a multi-page snapshot; mutable callers must prove their fixed keys.
        A listing is reused from the scan memo (ADR 0041) while no Artifact
        joined the creator's outputs and no row it read changed; every body is
        still served by ``read_logical``.
        """
        return [(reference, self.read_logical(reference, kind))
                for reference in self.find_logical_candidate_refs(creator, kind)]

    def find_logical_candidate_refs(self, creator: str, kind: str) -> tuple[ArtifactId, ...]:
        """List the addresses ``find_logical_candidates`` would read, parsing none twice.

        A caller that selects among a creator's many artifacts by their stored
        headers parses only the bodies it selects. The listing itself parsed and
        verified every body once; it is reused like ``find_logical_candidates``.
        """
        memo = self.scan_memo()
        memo_key = ("logical-candidates", creator, kind)
        listed = memo.lookup(memo_key)
        if listed is not MISSING:
            return tuple(ArtifactId(reference) for reference in cast(tuple[str, ...], listed))
        token = self.storage_epoch
        with recording_reads() as record:
            found = tuple(reference for reference, _ in self._list_logical_candidates(
                creator, kind,
            ))
        if record.complete:
            memo.store(memo_key, token, (MEMBERSHIP_ANY, *record.keys),
                       tuple(str(reference) for reference in found))
        return found

    def _list_logical_candidates(
        self, creator: str, kind: str
    ) -> list[tuple[ArtifactId, StrictModel]]:
        from aml.core.artifacts import SHARDABLE_KINDS
        from aml.store.indexed import ArtifactRows, IndexedReader

        kinds = (kind, "manifest") if kind in SHARDABLE_KINDS else (kind,)
        try:
            query = ArtifactRows(created_by_run_id=creator, kinds=kinds)
        except ValidationError as error:
            raise AMLError("invalid_request", "Invalid logical Artifact selector") from error
        reader = IndexedReader(
            self._ws,
            self.entities,
            tenant_id=self.tenant_id,
        )
        found: list[tuple[ArtifactId, StrictModel]] = []
        after: str | None = None
        while True:
            page = reader.page(query, after=after, limit=128)
            for row in page:
                if row["artifact_kind"] == "manifest" and kind != "manifest":
                    envelope = ArtifactRowV1.model_validate(row)
                    self._verify_artifact_identity(envelope)
                    manifest = parse_artifact_body("manifest", json.loads(envelope.body))
                    if manifest.model_dump(mode="json")["logical_kind"] != kind:
                        continue
                found.append(
                    (ArtifactId(row["object_id"]), self.read_logical(row["object_id"], kind))
                )
            if len(page) < 128:
                return found
            after = page[-1]["object_id"]

    def _output_lookup(
        self, identity: OutputIdentity, *, request: InferenceRequest | None = None
    ) -> tuple[WriteTicket, str, int, ArtifactId | None, StoredInferenceOutput | None]:
        """Read one frozen output identity outside the final commit critical section.

        The returned Run and creator-generation proof is an ephemeral optimization.
        Callers re-admit and check the exact selected Artifact before consuming it.
        """
        with self._write_scope():
            ticket = self.ticket(identity.run_id)
            run = self.get_row("run", identity.run_id)
            if run is None:
                raise AMLError("reference_missing", "Inference owner no longer exists")
            run_bytes = canonical_json(run)
            generation = self._artifact_generations.get(identity.run_id, 0)
        reference, output = self._frozen_output_lookup(run, identity, request=request)
        return ticket, run_bytes, generation, reference, output

    def _frozen_output_lookup(
        self, run: dict[str, Any], identity: OutputIdentity, *,
        request: InferenceRequest | None = None,
    ) -> tuple[ArtifactId | None, StoredInferenceOutput | None]:
        """Read actual immutable output binding without issuing execution authority.

        Live callers retain the original admitted ticket and creator generation;
        cold callers retain their exclusive startup proof and exact Run preimage.
        Both must recheck that authority before consuming the returned output.
        """
        from aml.core.inference import InferenceBindingV1, StoredInferenceOutput
        from aml.providers.contracts import validate_binding

        if run["object_id"] != identity.run_id:
            raise AMLError("binding_mismatch", "Frozen output lookup changed its original Run")
        frozen = self._bound_input(run)
        raw_binding = frozen.get("inference_binding")
        if raw_binding is None:
            raise AMLError("binding_mismatch", "Inference owner has no frozen binding")
        binding = InferenceBindingV1.model_validate(raw_binding)
        if (
            binding.purpose != identity.purpose
            or integrity_digest(binding) != identity.inference_binding_digest
            or "input_digest" in frozen
            and frozen["input_digest"] != identity.input_digest
        ):
            raise AMLError("binding_mismatch", "Inference lookup differs from the stored Run input")
        if request is not None:
            validate_binding(request, binding)
            if (
                request.input_digest != identity.input_digest
                or "context_digest" in frozen
                and request.context_digest != frozen["context_digest"]
            ):
                raise AMLError("binding_mismatch", "Inference request differs from frozen content")
        found: list[tuple[ArtifactId, StoredInferenceOutput]] = []
        for reference, body in self.find_logical_candidates(identity.run_id, "llm_output"):
            output = StoredInferenceOutput.model_validate(body.model_dump(mode="json"))
            if (
                output.purpose, output.input_digest, output.inference_binding_digest,
                output.call_no,
            ) == (
                identity.purpose,
                identity.input_digest,
                identity.inference_binding_digest,
                identity.call_no,
            ):
                fields = (
                    "purpose", "provider", "requested_model", "schema_id", "prompt_version",
                    "timeout_ms", "retry_policy_digest",
                )
                if any(getattr(output, field) != getattr(binding, field) for field in fields) or (
                    "context_digest" in frozen
                    and output.context_digest != frozen["context_digest"]
                ):
                    raise AMLError(
                        "binding_mismatch", "Stored output differs from its complete frozen binding"
                    )
                found.append((reference, output))
        if len(found) > 1:
            raise AMLError(
                "artifact_digest_collision", "Output identity has multiple immutable values"
            )
        return found[0] if found else (None, None)

    def get_output(
        self, identity: OutputIdentity, *,
        action_check: Callable[[WriteTicket], None] | None = None,
    ) -> StoredInferenceOutput | None:
        """Read an exact output with creator-local mutation and current admission proofs."""
        while True:
            ticket, run_bytes, generation, reference, output = self._output_lookup(identity)
            with self._write_scope():
                self._admit(ticket)
                if (
                    canonical_json(self.get_row("run", identity.run_id)) != run_bytes
                    or self._artifact_generations.get(identity.run_id, 0) != generation
                ):
                    continue
                if action_check is not None:
                    action_check(ticket)
                if reference is not None and output is not None:
                    current = self.get_row("artifact", reference)
                    if current is None:
                        continue
                    envelope = ArtifactRowV1.model_validate(current)
                    if envelope.created_by_run_id != identity.run_id:
                        raise AMLError("binding_mismatch", "Stored output changed its creator")
                    # A direct row or a manifest with its parts (A0.7): the
                    # verified logical body is the one the lookup selected.
                    stored = self.read_logical(reference, "llm_output")
                    if canonical_json(stored) != canonical_json(output):
                        raise AMLError("parity_mismatch", "Stored output differs from its lookup")
                return output

    def put_output_once(
        self,
        identity: OutputIdentity,
        request: InferenceRequest,
        result: InferenceResult,
        summary_context_digest: ContentBindingDigest | None = None,
        *,
        action_check: Callable[[WriteTicket], None] | None = None,
        clock: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    ) -> StoredInferenceOutput:
        """Persist an exact output using the application's trusted action clock.

        Semantic execution supplies the same clock used by its source and
        authorization checks. An existing output keeps its original creation
        time on replay; this parameter does not change its identity or body.
        """
        from aml.providers.reuse import stored_output

        output = stored_output(identity, request, result, summary_context_digest)
        while True:
            ticket, run_bytes, generation, reference, existing = self._output_lookup(
                identity, request=request
            )
            with self._write_scope():
                self._admit(ticket)
                if (
                    canonical_json(self.get_row("run", identity.run_id)) != run_bytes
                    or self._artifact_generations.get(identity.run_id, 0) != generation
                ):
                    continue
                if action_check is not None:
                    action_check(ticket)
                if reference is not None and existing is not None:
                    current = self.get_row("artifact", reference)
                    if current is None:
                        continue
                    envelope = ArtifactRowV1.model_validate(current)
                    if envelope.created_by_run_id != identity.run_id:
                        raise AMLError("binding_mismatch", "Stored output changed its creator")
                    stored = self.read_logical(reference, "llm_output")
                    if canonical_json(stored) != canonical_json(existing):
                        raise AMLError("parity_mismatch", "Stored output differs from its lookup")
                    if canonical_json(existing) != canonical_json(output):
                        raise AMLError(
                            "artifact_digest_collision",
                            "Output identity already binds different bytes",
                        )
                    return existing
                # An output over the row bound is stored as a manifest and its parts.
                self.put_logical_artifact(
                    ticket,
                    "llm_output",
                    output,
                    security_level="restricted",
                    created_at=clock(),
                )
                return output

    def prepare_source_labels(
        self, owner: LabelOwner, messages: Sequence[MessageRecord],
    ) -> SourceLabelsPreparation | None:
        """Freeze the owner's B.3a label request outside the Writer mutex.

        Returns None when no call is needed: every undecided label already has
        a durable decision, or the owner's one label child has ended.
        """
        from aml.store.source_labels import prepare_owner_call

        return prepare_owner_call(self, owner, messages)

    def admit_source_labels(self, prepared: SourceLabelsPreparation) -> str:
        """Admit or reselect the owner's label child and reserve its messages atomically."""
        from aml.store.source_labels import admit

        return admit(self, prepared)

    def authorize_source_labels(
        self, child_run_id: str, prepared: SourceLabelsPreparation,
    ) -> dict[str, Any]:
        """Recheck the owner, the frozen messages and the exact child before an action."""
        from aml.store.source_labels import authorize

        return authorize(self, child_run_id, prepared)

    def reserve_source_labels_attempt(
        self, child_run_id: str, prepared: SourceLabelsPreparation,
    ) -> SourceLabelsBudgetV1:
        """Persist one spent send before HTTP; an unknown outcome stays spent."""
        from aml.store.source_labels import reserve_attempt

        return reserve_attempt(self, child_run_id, prepared)

    def finish_source_labels(
        self, child_run_id: str, prepared: SourceLabelsPreparation,
    ) -> ArtifactId:
        """Close the label child from its unique stored output, with no business effect."""
        from aml.store.source_labels import finish

        return finish(self, child_run_id, prepared)
