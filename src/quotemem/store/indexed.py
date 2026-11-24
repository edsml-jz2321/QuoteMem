"""Bounded scalar reads through the pinned HeavenBase storage owner.

The private table, lock, and codec access is isolated here because HB's public
Query.limit applies after fetching scalar matches. These reads issue SQL LIMIT
and keyset predicates through HB's Database.orm_execute, at most 32 values per
IN and 512 rows per page; each NL (natural-language) query family filters by one declared index.
Each page is coherent with owned writes; multiple pages are not a snapshot.
Authorization and cross-row projection validation remain caller responsibilities.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast

from pydantic import Field, TypeAdapter, ValidationError, model_validator
from sqlalchemy import LargeBinary, Select, Table, and_, case, func, literal, or_, select
from sqlalchemy import cast as sql_cast
from sqlalchemy.sql.elements import ColumnElement

from quotemem.core.entities import (
    CandidateStatus,
    EntityRef,
    EntityState,
    LinkType,
    MemoryLifecycle,
)
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID, ArtifactId, ControlId, ScopeId
from quotemem.core.outcome_table import OUTCOME_TABLE
from quotemem.core.query import ScopeV1
from quotemem.core.semantic import (
    ScopeKind,
    SystemTimestamp,
)
from quotemem.core.statement import MAX_NAME_NORM_CHARS, MAX_REENTRY, HexDigest
from quotemem.core.technical_types import (
    ExternalId,
    HistoricalTimestamp,
    NonNegativeInt,
    SessionHash,
)
from quotemem.core.work import FinalizeSourceHeaderV1
from quotemem.service.cost_attribution import attribute_cost, cost_count, cost_span
from quotemem.store.domain import EvidenceRowV1, RecallObservationId, RunKind, TenantId
from quotemem.store.entities import ENUMS
from quotemem.store.hb_runtime import hb
from quotemem.store.metadata import StrictModel
from quotemem.store.row_cache import (
    MEMBERSHIP_ANY,
    membership_family,
    membership_key,
    note_reads,
    note_unprovable_read,
)

if TYPE_CHECKING:
    from quotemem.core.work import FinalizeSourceHeaderV1

MAX_PAGE_SIZE = 512
MAX_IN_VALUES = 32
IdempotencyKey = Annotated[str, Field(pattern=r"^idem_[A-Za-z0-9_-]{43}$")]


class RunRows(StrictModel):
    """Select one immutable Run family using its composite unique-index prefix.

    run_kind supports session_ingestion, finalize, publish, evaluate, retention,
    policy_release, and embedding_switch, matching the implemented Run model.
    """

    run_kind: RunKind
    idempotency_key: IdempotencyKey


class ScopedRunRows(StrictModel):
    """Page one exact scope's Run kind for content-free recovery discovery.

    This query does not establish authorization or a stable multi-page view.
    The recovery reader must verify complete selected authorities and retain
    its normal generation fence before using any discovered Run.
    """

    scope_kind: ScopeKind
    scope_id: ScopeId
    run_kind: RunKind
    target_id: str | None = None
    """Optional exact target: one session's runs instead of every run of the scope."""


class ArtifactRows(StrictModel):
    """Select one creator's physical artifacts, including direct and manifest rows.

    An empty kinds tuple includes all physical kinds. Nonempty tuples accept only
    the declared artifact_kind values in the physical schema, without duplicates.
    Logical manifest-kind validation remains Writer.read_logical's responsibility.
    """

    created_by_run_id: ULID
    kinds: Annotated[tuple[str, ...], Field(max_length=MAX_IN_VALUES)] = ()

    @model_validator(mode="after")
    def check_kinds(self) -> Self:
        """Reject duplicate or undeclared physical Artifact kinds.

        Args:
            None.

        Returns:
            Self: The validated query.

        Raises:
            ValueError: If a kind is duplicated or absent from the physical schema.
        """
        if len(self.kinds) != len(set(self.kinds)) or any(
            value not in ENUMS["artifact_kind"] for value in self.kinds
        ):
            raise ValueError("Artifact kinds must be distinct declared physical kinds")
        return self


class EvidenceRows(StrictModel):
    """Select exact-scope Evidence, optionally bounded by session and visibility.

    high_watermark includes only ingested_commit_seq at or below the frozen
    value. It does not freeze later expiration, invalidation, or privacy changes.
    scope_kind supports user, project, agent, and organization; each selects only
    that exact kind and scope_id pair, without ancestor expansion.
    """

    scope_kind: ScopeKind
    scope_id: ScopeId
    session_id: ExternalId | None = None
    request_id: ExternalId | None = None
    high_watermark: NonNegativeInt | None = None


_FINALIZE_TRUST_RANK: dict[str, Literal[0, 1, 2, 3]] = {
    "verified": 3, "reported": 2, "inferred": 1, "untrusted": 0,
}


def finalize_source_order(header: FinalizeSourceHeaderV1) -> tuple[int, int, str]:
    """Order frozen metadata exactly like the full Evidence support-order query."""
    observed = (
        header.ingested_at if header.source_observed_at is None else header.source_observed_at
    )
    return -_FINALIZE_TRUST_RANK[header.trust_tier], -observed, header.object_id


class FinalizeEvidenceQuery(StrictModel):
    """Fix the exact scope and eligibility time for one finalize input scan.

    The caller supplies tenant_id from its Writer, never from HB workspace.id.
    SQL deliberately does not filter tenant: an eligible same-scope row with a
    different tenant must be rejected instead of omitted from the input proof.
    This model carries no authorization or held-finalize capability.
    """

    tenant_id: TenantId
    scope: ScopeV1
    freeze_time: SystemTimestamp
    receipt_high_watermark: NonNegativeInt


def _finalize_evidence_predicate(
    table: Table, query: FinalizeEvidenceQuery, tenant: ColumnElement[Any],
) -> ColumnElement[bool]:
    """Select whole original messages at one time and tenant-commit boundary."""
    peer = table.alias("finalize_expired_peer")
    peer_tenant = peer.c.tenant_id if "tenant_id" in peer.c else tenant
    expired_peer = select(literal(1)).where(
        peer_tenant == tenant,
        *(peer.c[name] == table.c[name] for name in (
            "scope_kind", "scope_id", "session_id", "request_id", "ordinal",
        )),
        peer.c.ingested_commit_seq <= query.receipt_high_watermark,
        peer.c.expires_at.is_not(None), peer.c.expires_at <= query.freeze_time,
    ).correlate(table).exists()
    return and_(
        table.c.scope_kind == query.scope.scope_kind, table.c.scope_id == query.scope.scope_id,
        table.c.state == "active",
        table.c.ingested_commit_seq <= query.receipt_high_watermark,
        or_(table.c.expires_at.is_(None), table.c.expires_at > query.freeze_time),
        ~expired_peer,
    )


class FinalizeEvidenceCursor(StrictModel):
    """Exclusive support-order position; it is not a proof of preceding pages.

    rank uses verified=3, reported=2, inferred=1 and untrusted=0. observed_at
    preserves signed historical timestamps, using ingestion time only when
    source_observed_at is None. evidence_id is the binary ascending tie-breaker.
    """

    rank: Literal[0, 1, 2, 3]
    observed_at: HistoricalTimestamp
    evidence_id: ULID

    @classmethod
    def from_evidence(cls, row: EvidenceRowV1) -> Self:
        """Derive the next page boundary from its final fully validated row.

        Args:
            row: Last Evidence returned by finalize_evidence_page.

        Returns:
            Its exact rank, observation time and Evidence identity.

        Raises:
            ValidationError: If a supplied or subsequently mutated row is invalid.
        """
        value = EvidenceRowV1.model_validate(row.model_dump(mode="python"))
        return cls(
            rank=_FINALIZE_TRUST_RANK[value.trust_tier],
            observed_at=(
                value.ingested_at if value.source_observed_at is None
                else value.source_observed_at
            ),
            evidence_id=value.object_id,
        )

    @classmethod
    def at(cls, row: EvidenceRowV1 | FinalizeSourceHeaderV1) -> Self:
        """Return the support-order position of one already validated row or header.

        Args:
            row: A row or header a finalize read returned and validated.

        Returns:
            Its exact rank, observation time and Evidence identity.

        Raises:
            ValidationError: If the position itself is out of range.
        """
        return cls(
            rank=_FINALIZE_TRUST_RANK[row.trust_tier],
            observed_at=row.ingested_at if row.source_observed_at is None
            else row.source_observed_at,
            evidence_id=row.object_id,
        )


def _check_finalize_evidence(
    rows: Sequence[EvidenceRowV1 | FinalizeSourceHeaderV1], selected: FinalizeEvidenceQuery,
    cursor: FinalizeEvidenceCursor | None, limit: int, session_id: str | None,
) -> None:
    """Refuse a finalize page that left its tenant, scope, eligibility or support order."""
    if len(rows) > limit:
        raise ValueError("Finalize Evidence page exceeded its bound")
    preceding = cursor
    for row in rows:
        if (
            row.tenant_id != selected.tenant_id
            or row.scope_kind != selected.scope.scope_kind
            or row.scope_id != selected.scope.scope_id
            or row.state != "active"
            or row.expires_at is not None and row.expires_at <= selected.freeze_time
            or row.ingested_commit_seq > selected.receipt_high_watermark
            or session_id is not None and row.session_id != session_id
        ):
            raise ValueError("Finalize Evidence changed tenant, scope or eligibility")
        current = FinalizeEvidenceCursor.at(row)
        if preceding is not None and not (
            current.rank < preceding.rank
            or (current.rank == preceding.rank
                and current.observed_at < preceding.observed_at)
            or (current.rank == preceding.rank
                and current.observed_at == preceding.observed_at
                and current.evidence_id > preceding.evidence_id)
        ):
            raise ValueError("Finalize Evidence violates its exclusive support order")
        preceding = current


class LabeledRecallRows(StrictModel):
    """Select every retained nonbenchmark used/feedback row in one exact scope.

    The immutable parent, actual target version and label meaning remain the
    caller's responsibility. There is no sampling, outcome filter or guard-count
    truncation; callers must exhaust pages under an unchanged generation proof.
    """

    scope_kind: ScopeKind
    scope_id: ScopeId


class RecallDeliveries(StrictModel):
    """Select one exact scope's Search deliveries through ``ix_recall_scope``.

    The ``replayable_delivery`` and ``minimal_ref_delivery`` rows, each with
    its ``returned`` identities. Forget pages them for the deliveries that
    returned a forgotten memory or an Evidence row it withholds;
    the caller filters ``returned``.
    """

    scope_kind: ScopeKind
    scope_id: ScopeId


class RecallChildren(StrictModel):
    """Select the complete retained interaction family of one delivery.

    Include expanded rows and every environment/outcome: they can extend the
    actual parent's retention even when they are not themselves guard labels.
    Callers prove parent identity, child parity and complete paginated discovery.
    """

    parent_recall_id: ULID


class RevisionLinks(StrictModel):
    """Select the complete stored Link set for one current or historical Revision."""

    source_revision_id: ULID


def _one_index(prefix: bool, *families: bool) -> None:
    """Refuse a filter set that is not one declared index prefix."""
    if not prefix or sum(families) > 1:
        raise ValueError("A page filters by one declared index prefix")


class MemoryTableRows(StrictModel):
    """Page one scope's memories by one declared index.

    Without a filter the page enumerates the scope, as a cold rebuild does.
    ``entity_id``, optionally with ``lifecycle_state``, uses ``ix_item_entity``
    (``""`` selects erased heads); ``statement_digest`` uses ``ix_item_digest``;
    ``after_commit_seq`` selects the rows a later transaction changed
    (``ix_item_commit``).
    """

    scope_kind: ScopeKind
    scope_id: ScopeId
    entity_id: EntityRef | Literal[""] | None = None
    lifecycle_state: MemoryLifecycle | None = None
    statement_digest: HexDigest | None = None
    after_commit_seq: NonNegativeInt | None = None

    @model_validator(mode="after")
    def check_index(self) -> Self:
        """Use one of the three memory indexes."""
        _one_index(
            self.lifecycle_state is None or self.entity_id is not None,
            self.entity_id is not None, self.statement_digest is not None,
            self.after_commit_seq is not None,
        )
        return self


class RevisionRows(StrictModel):
    """Page one memory's versions through ``unique (memory_id, version)``."""

    memory_id: ULID


class LinkTargetRows(StrictModel):
    """Page the links of one type that target one key in one scope (``ix_link_target``).

    The key is a memory for ``supersedes`` and ``contradicts``, a corrected
    revision for ``corrects_history`` and an ``EntityRef`` for ``about_entity``.
    """

    scope_kind: ScopeKind
    scope_id: ScopeId
    link_type: LinkType
    target_key: Annotated[str, Field(min_length=1, max_length=64)]


class CandidateRows(StrictModel):
    """Page one scope's candidates by one declared index.

    ``status`` alone uses ``ix_candidate_status``; ``entity_id``, optionally with
    ``status``, uses ``ix_candidate_entity``; ``after_commit_seq`` selects the
    rows a later transaction changed (``ix_candidate_commit``).
    """

    scope_kind: ScopeKind
    scope_id: ScopeId
    status: CandidateStatus | None = None
    entity_id: EntityRef | None = None
    after_commit_seq: NonNegativeInt | None = None

    @model_validator(mode="after")
    def check_index(self) -> Self:
        """Use one of the three candidate indexes."""
        _one_index(
            True, self.status is not None or self.entity_id is not None,
            self.after_commit_seq is not None,
        )
        return self


class EntityRows(StrictModel):
    """Page one scope's entities in one state (``ix_entity_state``) or changed later
    (``ix_entity_commit``)."""

    scope_kind: ScopeKind
    scope_id: ScopeId
    state: EntityState | None = None
    after_commit_seq: NonNegativeInt | None = None

    @model_validator(mode="after")
    def check_index(self) -> Self:
        """Use one of the two entity indexes."""
        _one_index(True, self.state is not None, self.after_commit_seq is not None)
        return self


class EntityNameRows(StrictModel):
    """Page one scope's derived names by one declared index.

    ``name_norms`` (at most 32, distinct) uses ``ix_name_lookup``;
    ``entity_id`` uses ``ix_name_entity``; ``after_commit_seq`` selects the rows a
    later transaction wrote (``ix_name_commit``). A row counts only under ``p_name``,
    which its reader applies.
    """

    scope_kind: ScopeKind
    scope_id: ScopeId
    name_norms: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=MAX_NAME_NORM_CHARS)], ...],
        Field(max_length=MAX_IN_VALUES),
    ] = ()
    entity_id: EntityRef | None = None
    after_commit_seq: NonNegativeInt | None = None

    @model_validator(mode="after")
    def check_index(self) -> Self:
        """Use one of the three name indexes, with distinct names."""
        if len(set(self.name_norms)) != len(self.name_norms):
            raise ValueError("Name lookups are distinct")
        _one_index(
            True, bool(self.name_norms), self.entity_id is not None,
            self.after_commit_seq is not None,
        )
        return self


class TombstoneRows(StrictModel):
    """Page one scope's tombstones by digest (``ix_tomb_digest``) or written after a
    ``tombstone_seq`` (``ix_tomb_seq``); only the TombstoneIndex and the Writer guard
    read them."""

    scope_kind: ScopeKind
    scope_id: ScopeId
    statement_digest: HexDigest | None = None
    after_tombstone_seq: NonNegativeInt | None = None

    @model_validator(mode="after")
    def check_index(self) -> Self:
        """Use one of the two scoped tombstone indexes."""
        _one_index(True, self.statement_digest is not None, self.after_tombstone_seq is not None)
        return self


class ForgetTombstoneRows(StrictModel):
    """Page the tombstones one forget run wrote (``ix_tomb_run``)."""

    forget_run_id: ULID


IndexedQuery = (
    RunRows | ScopedRunRows | ArtifactRows | EvidenceRows | LabeledRecallRows | RecallDeliveries
    | RecallChildren | RevisionLinks | MemoryTableRows | RevisionRows | LinkTargetRows
    | CandidateRows | EntityRows | EntityNameRows | TombstoneRows | ForgetTombstoneRows
)
_QUERY_KINDS: dict[type[StrictModel], str] = {
    RunRows: "run",
    ScopedRunRows: "run",
    ArtifactRows: "artifact",
    EvidenceRows: "evidence",
    LabeledRecallRows: "recall",
    RecallDeliveries: "recall",
    RecallChildren: "recall",
    RevisionLinks: "link",
    MemoryTableRows: "item",
    RevisionRows: "revision",
    LinkTargetRows: "link",
    CandidateRows: "candidate",
    EntityRows: "entity",
    EntityNameRows: "entity_name",
    TombstoneRows: "tombstone",
    ForgetTombstoneRows: "tombstone",
}
_LOWER_BOUNDS = {"after_commit_seq": "commit_seq", "after_tombstone_seq": "tombstone_seq"}
"""Query fields that select rows strictly after a sequence, and the column they bound."""


def _page_membership(query: IndexedQuery) -> tuple[tuple[str, str], ...] | None:
    """Name the set a page enumerates when a membership key covers every member.

    Every Evidence upsert retires its exact scope's Evidence set, or the
    Evidence wildcard when its post-image omits the scope, and every Run upsert
    retires its idempotency family or the Run wildcard. An Evidence
    or Run-family page is therefore proven by those keys and the rows it
    returned; a receipt inventory that pages a request's parts and receipt
    Runs can then be memoized by its reads. Every other page stays unprovable.
    """
    if isinstance(query, EvidenceRows):
        return (
            membership_family("evidence"),
            membership_key("evidence", f"{query.scope_kind}:{query.scope_id}"),
        )
    if isinstance(query, RunRows):
        return membership_family("run"), membership_key("run", query.idempotency_key)
    return None


_ULID: TypeAdapter[str] = TypeAdapter(ULID)
_RECALL_ID: TypeAdapter[str] = TypeAdapter(RecallObservationId)
_ARTIFACT_ID = TypeAdapter(ArtifactId)
_SCOPE_ID: TypeAdapter[str] = TypeAdapter(ScopeId)
_CONTROL_KEY: TypeAdapter[str] = TypeAdapter(Annotated[str, Field(min_length=1, max_length=1024)])
_CONTROL_ID: TypeAdapter[str] = TypeAdapter(ControlId)
_SESSION_HASH: TypeAdapter[str] = TypeAdapter(SessionHash)
_EXTERNAL_ID: TypeAdapter[str] = TypeAdapter(ExternalId)
_TENANT: TypeAdapter[str] = TypeAdapter(TenantId)


class _EvidenceReferenceHeader(StrictModel):
    """Typed locator ownership only, without Evidence content or eligibility."""

    object_id: ULID
    tenant_id: TenantId
    scope_kind: ScopeKind
    scope_id: ScopeId


class _EntityHeader(StrictModel):
    """The content-free columns of one entity row that reception's check reads."""

    object_id: ULID
    scope_kind: ScopeKind
    scope_id: ScopeId
    entity_type: Annotated[str, Field(max_length=64)]
    state: EntityState
    commit_seq: NonNegativeInt


_EFFECT_PRODUCERS: tuple[tuple[str, str], ...] = (
    ("run", "parent_run_id"),
    ("candidate", "owner_run_id"),
    ("revision", "origin_run_id"),
    ("evidence", "origin_run_id"),
    ("entity", "created_by_run_id"),
    ("tombstone", "forget_run_id"),
)
"""Every row that names the Run that produced it: a child, an owned position and the
revisions, Evidence, entities and tombstones a Run writes (the zero-effect probes)."""
_SCOPE_OWNERS = ("run", "evidence", "item", "candidate", "recall", "artifact")
"""The direct scope owners whose probes decide scope occupancy on the NL tables."""
_ENTITY_STATE: TypeAdapter[EntityState] = TypeAdapter(EntityState)
_REENTERED_CODES: tuple[str, ...] = tuple(
    code for code, rule in OUTCOME_TABLE.items() if rule.rejudge == "yes"
)
"""The reason codes later owners re-enter (rejudge ``yes``), well under 32 per IN."""


class _RequestLedgerHeader(StrictModel):
    """A reception directory entry, without any source or request payload."""

    object_id: ControlId
    key: str
    version: NonNegativeInt


class BenchScopeActivity(StrictModel):
    """Durable notification chronology, without source content or eligibility."""

    scope_id: ScopeId
    last_ingested_at: SystemTimestamp
    last_input_seq: NonNegativeInt


class BenchSessionActivity(StrictModel):
    """One exact session's first and latest durable reception coordinates."""

    session_id: ExternalId
    first_input_seq: NonNegativeInt
    last_input_seq: NonNegativeInt
    last_ingested_at: SystemTimestamp


class _ActivityTimestamp(StrictModel):
    """Nullable scalar aggregate over actual persisted activity timestamps."""

    last_activity: SystemTimestamp | None


class _SourceCoordinate(StrictModel):
    """Exact receipt identity scalars, without Evidence content or liveness authority."""

    object_id: ULID
    tenant_id: TenantId
    scope_kind: ScopeKind
    scope_id: ScopeId
    session_id: ExternalId
    request_id: ExternalId
    ordinal: NonNegativeInt
    part_no: NonNegativeInt
    ingested_commit_seq: NonNegativeInt


class _SourceOrderingHeader(_SourceCoordinate):
    """Original chronology only; separate from the receipt coordinate identity."""

    source_observed_at: HistoricalTimestamp | None


class _EvidenceEligibilityHeader(_EvidenceReferenceHeader):
    """Fixed ownership and liveness scalars, never original source content."""

    state: Literal["active", "invalidated", "expired", "erased"]
    expires_at: SystemTimestamp | None


_SESSION_TAIL_COLUMNS: tuple[str, ...] = (
    "object_id", "tenant_id", "scope_kind", "scope_id", "session_id", "request_id", "ordinal",
    "part_no", "text", "byte_start", "byte_end", "state", "route_state", "expires_at",
    "ingested_commit_seq",
)
"""Evidence columns an active session tail returns; the embedding vector stays in storage."""


def _query_membership(query: object) -> tuple[tuple[str, str], ...] | None:
    """Name the set a page enumerates when every upsert into it retires that key.

    A Run upsert retires its idempotency family's key, an Evidence upsert its
    scope's key and an Artifact upsert its creator's key, so pages over one
    family, one scope or one creator are provable by those keys and the rows
    returned. Other queries stay unprovable.
    """
    if isinstance(query, RunRows):
        return membership_family("run"), membership_key("run", query.idempotency_key)
    if isinstance(query, ArtifactRows):
        return membership_family("artifact"), membership_key(
            "artifact", query.created_by_run_id,
        )
    if isinstance(query, EvidenceRows):
        return membership_family("evidence"), membership_key(
            "evidence", f"{query.scope_kind}:{query.scope_id}",
        )
    return None


class BatchedHeaders:
    """Point headers of a known locator set, read together at the first lookup.

    A caller that resolves many locators one point query each instead names
    them all up front: the first lookup reads every distinct locator through
    ``read_many`` (32 per statement), and each lookup then returns a copy of
    what its own point query returns. Should ``read_many`` refuse (an invalid
    locator or malformed metadata), every lookup reads alone through
    ``read_one``, so each refusal keeps its point query's order and code. The
    caller keeps the storage view fixed across its lookups, as it did across
    its point queries, and owns every check of the returned headers.
    """

    __slots__ = ("_headers", "_identifiers", "_read_many", "_read_one")

    def __init__(
        self,
        read_many: Callable[[Sequence[str]], dict[str, dict[str, Any] | None]],
        read_one: Callable[[str], dict[str, Any] | None],
        identifiers: Iterable[str],
    ) -> None:
        self._read_many, self._read_one = read_many, read_one
        self._identifiers = tuple(dict.fromkeys(identifiers))
        self._headers: dict[str, dict[str, Any] | None] | None = None

    def __call__(self, identifier: str) -> dict[str, Any] | None:
        if self._headers is None:
            try:
                self._headers = self._read_many(self._identifiers)
            except QuoteMemError:
                self._headers = {}
        if identifier not in self._headers:
            return self._read_one(identifier)
        header = self._headers[identifier]
        return None if header is None else dict(header)


class IndexedReader:
    """Read bounded pages from one explicitly owned, already registered workspace.

    The caller retains ownership of the workspace and its lifetime. Construction
    performs no database access. No query can supply SQL, operators, table names,
    JSON paths, or custom sorting. Page size is a local resource bound, not an
    external Add limit. SQL LIMIT bounds returned rows, not SQLite rows examined.
    """

    def __init__(
        self, workspace: hb.HeavenBase, entities: Mapping[str, type[hb.Entity]],
        *, tenant_id: str | None = None,
        artifact_image: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        evidence_image: Callable[[], Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        """Bind this reader to existing HeavenBase authorities.

        Args:
            workspace (hb.HeavenBase): Explicit workspace owned by the service.
            entities (Mapping[str, type[hb.Entity]]): Registered QuoteMem entity classes.
            tenant_id: Verified workspace tenant, required when normalized rows
                omit tenant identity. Never inferred from workspace or Control IDs.
            artifact_image: Restores one exact decoded physical Artifact row in
                place of ``restore_row_image``, returning exactly its image:
                the Writer's projection memo (``Writer.indexed_reader``).
            evidence_image: Called before each Evidence statement executes; the
                restorer it returns replaces ``restore_row_image`` for the
                statement's rows and returns exactly their images: the Writer's
                Evidence projection memo (``Writer.indexed_reader``).

        Returns:
            None: Stores references without opening storage or registering schemas.
        """
        self._workspace: Any = workspace
        self._entities = dict(entities)
        self._artifact_image = artifact_image
        self._evidence_image = evidence_image
        try:
            self._tenant_id = None if tenant_id is None else _TENANT.validate_python(
                tenant_id, strict=True,
            )
        except ValidationError as error:
            raise QuoteMemError("invalid_request", "Indexed workspace tenant is invalid") from error

    def page(
        self, query: IndexedQuery, *, after: str | None = None, limit: int = 128
    ) -> list[dict[str, Any]]:
        """Read one fully drained page in ascending physical object_id order.

        Args:
            query (IndexedQuery): One of the closed query models documented above.
            after (str | None): Exclusive last object_id from the previous page.
                Artifact queries require an ArtifactId; other queries require a ULID.
            limit (int): Inclusive page bound from 1 through MAX_PAGE_SIZE (512).

        Returns:
            list[dict[str, Any]]: At most limit HB-decoded rows. A short or empty
                page exhausts the current view. No connection or lock escapes.

        Raises:
            QuoteMemError: If query, cursor, page bound, or physical routing is invalid.
            RuntimeError: If the workspace or its backend is closed.

        Notes:
            Concurrent mutations may change later pages. Callers must use their
            protocol's immutable watermark or generation/fence checks; this method
            does not promise a multi-page snapshot or perform authorization.
            An Evidence or Run-family page names its set to active read records.
        """
        return self._read(
            query, after=after, limit=limit, latest=False, membership=_page_membership(query),
        )

    def finalize_evidence_page(
        self,
        query: FinalizeEvidenceQuery,
        *,
        after: FinalizeEvidenceCursor | None = None,
        limit: int = 128,
        session_id: str | None = None,
    ) -> tuple[EvidenceRowV1, ...]:
        """Read eligible full Evidence in the frozen support order.

        Args:
            query: Exact scope, Writer tenant and fixed eligibility timestamp.
            after: Exclusive final cursor of the preceding page, or None to start.
            limit: SQL result bound from 1 through 512, inclusive.
            session_id: Return only this session's rows, in the same order the
                unnarrowed scan returns them. A consumer that wants one
                session's members read the whole scope and discarded the rest,
                once per session and once per cohort child, and every discarded
                row still decoded its embedding vector (3.4 GB of scope pages in eighteen minutes).
                The predicate narrows the same statement; it changes no row's
                eligibility, its order or its cursor.

        Returns:
            Fully validated Evidence rows, preserving both pending and indexed
            routes, in descending trust/time and binary ascending identity order.
            All rows in a page validate before any is returned.

        Raises:
            QuoteMemError: Invalid request, malformed/substituted stored row or page,
                or incompatible physical registration/routing.
            RuntimeError: The borrowed workspace or backend is closed.

        Notes:
            SQL LIMIT bounds returned rows, not work examined or index cost.
            Pages are not a stable snapshot. The caller owns authorization,
            content lifetime, held owner/lease proof, fixed freeze_time, complete
            traversal from None, and before/after generation or input tokens.
            Expiry is exclusive: expires_at == freeze_time is ineligible.
        """
        selected, cursor = self._finalize_evidence_bounds(query, after, limit, session_id)
        statement = self._finalize_evidence_statement(
            selected, cursor, limit, session_id, lambda table: (table,),
        )
        try:
            rows = tuple(EvidenceRowV1.model_validate(row) for row in self._select(
                "evidence", statement,
            ))
            _check_finalize_evidence(rows, selected, cursor, limit, session_id)
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid",
                                "Invalid stored finalize Evidence page") from error
        return rows

    def _finalize_evidence_bounds(
        self, query: FinalizeEvidenceQuery, after: FinalizeEvidenceCursor | None, limit: int,
        session_id: str | None,
    ) -> tuple[FinalizeEvidenceQuery, FinalizeEvidenceCursor | None]:
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
            raise QuoteMemError("invalid_request",
                                "Indexed page size is outside its resource bound")
        if session_id is not None and (type(session_id) is not str or not session_id):
            raise QuoteMemError("invalid_request", "Invalid finalize Evidence session")
        try:
            selected = FinalizeEvidenceQuery.model_validate(query.model_dump(mode="python"))
            cursor = None if after is None else FinalizeEvidenceCursor.model_validate(
                after.model_dump(mode="python")
            )
        except (AttributeError, ValidationError) as error:
            raise QuoteMemError("invalid_request", "Invalid finalize Evidence page") from error
        return selected, cursor

    def _finalize_evidence_statement(
        self, selected: FinalizeEvidenceQuery, cursor: FinalizeEvidenceCursor | None,
        limit: int, session_id: str | None, columns: Callable[[Table], Sequence[Any]],
    ) -> Callable[[Table], Select[Any]]:
        def statement(table: Table) -> Select[Any]:
            rank = case(_FINALIZE_TRUST_RANK, value=table.c.trust_tier, else_=4)
            observed = func.coalesce(table.c.source_observed_at, table.c.ingested_at)
            identity = table.c.object_id.collate("BINARY")
            value = select(*columns(table)).where(_finalize_evidence_predicate(
                table, selected, self._column("evidence", table, "tenant_id"),
            ))
            if session_id is not None:
                value = value.where(table.c.session_id == session_id)
            if cursor is not None:
                value = value.where(or_(
                    # Unknown trust must surface even if this scan resumes after
                    # a valid cursor; NULL CASE results would be silently lost.
                    rank == 4,
                    rank < cursor.rank,
                    and_(rank == cursor.rank, observed < cursor.observed_at),
                    and_(rank == cursor.rank, observed == cursor.observed_at,
                         identity > cursor.evidence_id),
                ))
            return value.order_by(rank.desc(), observed.desc(), identity.asc()).limit(limit)

        return statement

    def latest_run(self, run_kind: RunKind, idempotency_key: str) -> dict[str, Any] | None:
        """Read the highest stored generation of exactly one Run family.

        Args:
            run_kind (RunKind): Uses the Run kind values documented on RunRows.
            idempotency_key (str): The canonical idem_ family digest.

        Returns:
            dict[str, Any] | None: The complete HB-decoded latest Run, or None.

        Raises:
            QuoteMemError: If the family identity or physical routing is invalid.
            RuntimeError: If the workspace or its backend is closed.

        Notes:
            The caller must compare admission_digest under its Writer/session
            protocol before deciding whether to reuse or admit another generation.
        """
        try:
            query = RunRows(run_kind=run_kind, idempotency_key=idempotency_key)
        except ValidationError as error:
            raise QuoteMemError("invalid_request", "Invalid indexed Run family") from error
        rows = self._read(query, after=None, limit=1, latest=True, membership=(
            membership_family("run"), membership_key("run", idempotency_key),
        ))
        return rows[0] if rows else None

    def latest_scope_owner(self, scope: ScopeV1) -> dict[str, Any] | None:
        """Locate one actual scope Run for a content-free quarantine receipt.

        This read grants no mutation authority. The Writer must re-read the
        selected Run and any active finalize lease before recording a failure.
        """
        scope = ScopeV1.model_validate(scope.model_dump(mode="json"))

        def statement(table: Table) -> Select[Any]:
            return select(table).where(
                table.c.scope_kind == scope.scope_kind, table.c.scope_id == scope.scope_id,
            ).order_by(table.c.object_id.desc()).limit(1)

        rows = self._select("run", statement)
        return rows[0] if rows else None

    def has_scope_rows(self, scope_kind: ScopeKind, scope_id: str) -> bool:
        """Check whether a direct physical owner already occupies this canonical ID.

        Args:
            scope_kind (ScopeKind): Uses the exact scope kinds documented on EvidenceRows.
            scope_id (str): Canonical sc_ scope identity.

        Returns:
            bool: Whether a Run, Evidence, memory, Candidate, Recall or Artifact
                row has a matching scope ID, including a corrupt different-kind
                owner. Each table uses SELECT object_id with LIMIT 1; all but
                Recall seek their scope index with every declared scope kind.

        Raises:
            QuoteMemError: If the scope identity or physical routing is invalid.
            RuntimeError: If the workspace or its backend is closed.

        Notes:
            Both requested scope fields are validated before SQL. Occupancy is
            an integrity check, so a wrong-kind row cannot masquerade as absence.
            One shared workspace lock covers all probes. Revision and Link
            ownership is reached through the memory. Entity, entity_name and
            Tombstone rows are written only by Runs of their scope, so they are
            not probed. This is not an orphan-integrity scan or an authorization decision.
        """
        query = self._scope_query(scope_kind, scope_id)

        def statement(kind: str, table: Table) -> Select[Any]:
            if kind == "recall" and self._normalized(kind):
                parent = table.alias("recall_parent")
                interaction = table.c.record_class == "interaction"
                invalid = and_(interaction, or_(
                    parent.c.object_id.is_(None), parent.c.record_class == "interaction",
                ))
                scope = case((interaction, parent.c.scope_id), else_=table.c.scope_id)
                return select(
                    table.c.object_id, invalid.label("_aml_authority_invalid"),
                ).select_from(table.outerjoin(
                    parent, table.c.parent_recall_id == parent.c.object_id,
                )).where(or_(scope == query.scope_id, invalid)).limit(1)
            return (
                select(table.c.object_id)
                .where(
                    table.c.scope_kind.in_(ENUMS["scope_kind"]),
                    table.c.scope_id == query.scope_id,
                )
                .limit(1)
            )

        with self._workspace._lock.read():
            self._workspace._need_live()
            return any(
                self._select(kind, lambda table, selected=kind: statement(selected, table),
                             complete=False)
                for kind in _SCOPE_OWNERS
            )

    def bench_job_controls(self, *, after: str | None = None) -> list[dict[str, Any]]:
        """Read the next two job Controls through their indexed key range.

        The consumer continues by the last key; two jobs prove competing ownership.
        The exact bare namespace key is included for strict rejection;
        malformed suffixes are never filtered out. The caller validates each
        complete decoded row and its initialized bench profile.

        Returns:
            At most two complete Control rows, in ascending binary key order.

        Raises:
            QuoteMemError: If the physical owner or response projection is invalid.
            RuntimeError: If the workspace or backend has closed.
        """
        def statement(table: Table) -> Select[Any]:
            key = table.c.key.collate("BINARY")
            selected = select(table).where(or_(
                key == "job", and_(key >= "job:", key < "job;"),
            ))
            if after is not None:
                selected = selected.where(key > after)
            return selected.order_by(key).limit(2)

        return self._select("control", statement)

    def judge_output_controls(
        self, scope_id: str, *, after: str | None = None, limit: int = 128,
    ) -> list[dict[str, Any]]:
        """Page one scope's ``judge_output`` locators through their indexed key range.

        Privacy erase deletes every locator whose call read erased Evidence;
        the key range ``judge_output:{scope}:`` is the only way to find them
        without a table scan.

        Args:
            scope_id (str): Canonical sc_ identity; no raw scope or SQL input.
            after (str | None): Exclusive last key of the preceding page.
            limit (int): SQL page bound from 1 through MAX_PAGE_SIZE (512).

        Returns:
            list[dict[str, Any]]: Complete decoded Control rows in ascending
                binary key order; the caller validates each value.

        Raises:
            QuoteMemError: If the scope, page bound or routing is invalid.
            RuntimeError: If the workspace or backend has closed.
        """
        try:
            canonical = _SCOPE_ID.validate_python(scope_id, strict=True)
        except ValidationError:
            raise QuoteMemError("invalid_request",
                                "Judge locators require a canonical scope") from None
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
            raise QuoteMemError("invalid_request",
                                "Indexed page size is outside its resource bound")
        low, high = f"judge_output:{canonical}:", f"judge_output:{canonical};"
        if after is not None and not (low <= after < high):
            raise QuoteMemError("invalid_request", "Indexed page cursor is invalid")

        def statement(table: Table) -> Select[Any]:
            key = table.c.key.collate("BINARY")
            selected = select(table).where(and_(key >= low, key < high))
            if after is not None:
                selected = selected.where(key > after)
            return selected.order_by(key).limit(limit)

        return self._select("control", statement)

    def scope_controls(
        self, namespace: Literal["scope", "scope_quarantine"], *, after: str | None = None,
        limit: int = 128,
    ) -> list[dict[str, Any]]:
        """Page the scope registrations or the scope quarantines in key order.

        Args:
            namespace: ``scope`` (``scope:{s}`` registrations) or
                ``scope_quarantine`` (``scope_quarantine:{s}`` receipts).
            after: The exclusive last canonical scope id of the previous page.
            limit: The page bound, 1 through MAX_PAGE_SIZE.

        Returns:
            Complete decoded Control rows; a short page exhausts the namespace.
            Concurrent registrations may change later pages; no snapshot is promised.

        Raises:
            QuoteMemError: If the cursor, page bound or physical routing is invalid.
        """
        self._check_page(after, limit, _SCOPE_ID)
        prefix = namespace + ":"

        def statement(table: Table) -> Select[Any]:
            key = table.c.key.collate("BINARY")
            selected = select(table).where(key >= prefix, key < namespace + ";")
            if after is not None:
                selected = selected.where(key > prefix + after)
            return selected.order_by(key).limit(limit)

        return self._select("control", statement)

    def forget_intents(
        self, *, after: str | None = None, limit: int = 128,
    ) -> list[dict[str, Any]]:
        """Page the forget intents (``intent:{s}:{p}:forget:{binding}:{nonce}``) in key order.

        Args:
            after: The exclusive last key of the previous page.
            limit: The page bound, 1 through MAX_PAGE_SIZE.

        Returns:
            Complete decoded Control rows; a short page exhausts the namespace. No snapshot
            is promised.

        Raises:
            QuoteMemError: If the cursor, page bound or physical routing is invalid.
        """
        self._check_page(after, limit, _CONTROL_KEY)

        def statement(table: Table) -> Select[Any]:
            key = table.c.key.collate("BINARY")
            selected = select(table).where(key >= "intent:", key < "intent;")
            if after is not None:
                selected = selected.where(key > after)
            return selected.order_by(key).limit(limit)

        return self._select("control", statement)

    def scope_fences(self, scope_id: str) -> list[dict[str, Any]]:
        """Read at most three fence owners in one exact canonical scope.

        Args:
            scope_id (str): Canonical sc_ identity; no raw scope or SQL input.

        Returns:
            list[dict[str, Any]]: Complete decoded erasure/maintenance Control
                rows, ordered by key after the bounded read. One of each may
                coexist during takeover. Three rows prove duplicate namespace
                ownership; callers must fail closed rather than treat this as
                an exhaustive list. Unknown suffixes are returned for the
                caller's integrity gate, never silently discarded.

        Raises:
            QuoteMemError: If the scope identity or physical routing is invalid.
            RuntimeError: If the workspace or its backend is closed.

        Notes:
            Scope ownership is assigned through these key namespaces.
            The fixed ASCII colon-to-semicolon successor bounds each prefix
            under BINARY collation. SQL LIMIT 3 and the key index bound this
            read; it does not scan or parse every Control value. Existing
            shared-lock, codec, and response-drain rules apply through _select.
        """
        try:
            canonical = _SCOPE_ID.validate_python(scope_id, strict=True)
        except ValidationError:
            raise QuoteMemError("invalid_request",
                                "Fence lookup requires a canonical scope") from None

        def statement(table: Table) -> Select[Any]:
            key = table.c.key.collate("BINARY")
            return (
                select(table)
                .where(
                    or_(
                        *[
                            and_(key >= prefix + ":", key < prefix + ";")
                            for prefix in (
                                f"erasure:{canonical}",
                                f"maintenance:{canonical}",
                            )
                        ]
                    )
                )
                .limit(3)
            )

        return sorted(self._select("control", statement), key=lambda row: row["key"])

    def bench_scope_activity(
        self, tenant_id: str, *, after: str | None = None, limit: int = 128,
    ) -> tuple[BenchScopeActivity, ...]:
        """Page exact-tenant user reception metadata for background discovery.

        All Evidence states contribute chronology. Eligibility must be proven
        independently from full Evidence. SQL bounds returned headers; grouping
        work itself is not constant time.
        """
        self._check_page(after, limit, _SCOPE_ID)
        tenant_adapter: TypeAdapter[str] = TypeAdapter(TenantId)
        tenant = tenant_adapter.validate_python(tenant_id, strict=True)

        def statement(table: Table) -> Select[Any]:
            identity = table.c.scope_id.collate("BINARY")
            selected = select(
                table.c.scope_id,
                func.max(table.c.ingested_at).label("last_ingested_at"),
                func.max(table.c.ingested_commit_seq).label("last_input_seq"),
            ).where(self._column("evidence", table, "tenant_id") == tenant,
                    table.c.scope_kind == "user")
            if after is not None:
                selected = selected.where(identity > after)
            return selected.group_by(identity).order_by(identity).limit(limit)

        try:
            values = tuple(BenchScopeActivity.model_validate(row) for row in self._select(
                "evidence", statement, complete=False,
            ))
            keys = [value.scope_id for value in values]
            if len(values) > limit or keys != sorted(set(keys)) or any(
                after is not None and key <= after for key in keys
            ):
                raise ValueError("Scope activity changed its exact page")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid", "Invalid stored scope activity") from error
        return values

    def has_available_evidence(self, scope: ScopeV1, tenant_id: str, at_time: int) -> bool:
        """Check live Evidence membership without hydrating any source body.

        This scheduling hint grants no frozen eligibility or execution authority.
        Expiry is exclusive, matching the actual finalize Evidence scan.
        """
        query = ScopeV1.model_validate(scope.model_dump(mode="python"))
        tenant_adapter: TypeAdapter[str] = TypeAdapter(TenantId)
        time_adapter: TypeAdapter[int] = TypeAdapter(SystemTimestamp)
        tenant = tenant_adapter.validate_python(tenant_id, strict=True)
        now = time_adapter.validate_python(at_time, strict=True)

        def statement(table: Table) -> Select[Any]:
            return select(*self._columns(
                "evidence", table, _EvidenceReferenceHeader.model_fields,
            )).where(
                self._column("evidence", table, "tenant_id") == tenant,
                table.c.scope_kind == query.scope_kind,
                table.c.scope_id == query.scope_id, table.c.state == "active",
                or_(table.c.expires_at.is_(None), table.c.expires_at > now),
            ).limit(1)

        try:
            values = tuple(_EvidenceReferenceHeader.model_validate(row) for row in self._select(
                "evidence", statement, complete=False,
            ))
            if len(values) > 1 or any(
                (value.tenant_id, value.scope_kind, value.scope_id)
                != (tenant, query.scope_kind, query.scope_id) for value in values
            ):
                raise ValueError("Available Evidence changed its exact membership")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid",
                                "Invalid available Evidence membership") from error
        return bool(values)

    def latest_bench_activity(self, scope: ScopeV1, tenant_id: str) -> int | None:
        """Read latest durable Add/execution activity without hydrating source or input.

        Evidence uses its bound tenant authority; Run has scope columns only. The
        two bounded scalar reads are not a snapshot. The idle scheduler must
        recheck its Writer mutation witness before using this admission hint.
        """
        query = ScopeV1.model_validate(scope.model_dump(mode="python"))
        tenant_adapter: TypeAdapter[str] = TypeAdapter(TenantId)
        tenant = tenant_adapter.validate_python(tenant_id, strict=True)
        if query.scope_kind != "user":
            raise QuoteMemError("invalid_request", "Bench activity requires an exact user")

        def reception(table: Table) -> Select[Any]:
            return select(func.max(table.c.ingested_at).label("last_activity")).where(
                self._column("evidence", table, "tenant_id") == tenant,
                table.c.scope_kind == query.scope_kind,
                table.c.scope_id == query.scope_id,
            ).limit(1)

        def execution(table: Table) -> Select[Any]:
            activity = func.coalesce(
                table.c.updated_at, table.c.finished_at, table.c.started_at, table.c.created_at,
            )
            return select(func.max(activity).label("last_activity")).where(
                table.c.scope_kind == query.scope_kind, table.c.scope_id == query.scope_id,
                table.c.run_kind.in_(("session_ingestion", "finalize")),
            ).limit(1)

        values: list[int] = []
        try:
            for kind, statement in (("evidence", reception), ("run", execution)):
                rows = self._select(kind, statement, complete=False)
                if len(rows) != 1:
                    raise ValueError("Activity aggregate did not return exactly one scalar")
                value = _ActivityTimestamp.model_validate(rows[0]).last_activity
                if value is not None:
                    values.append(value)
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid", "Invalid stored activity timestamp") from error
        return max(values) if values else None

    def bench_session_activity(
        self, scope: ScopeV1, tenant_id: str, *, after: str | None = None, limit: int = 128,
    ) -> tuple[BenchSessionActivity, ...]:
        """Page an exact user's durable session chronology without source bodies."""
        self._check_page(after, limit, _EXTERNAL_ID)
        query = ScopeV1.model_validate(scope.model_dump(mode="python"))
        if query.scope_kind != "user":
            raise QuoteMemError("invalid_request", "Bench activity requires an exact user")
        tenant_adapter: TypeAdapter[str] = TypeAdapter(TenantId)
        tenant = tenant_adapter.validate_python(tenant_id, strict=True)

        def statement(table: Table) -> Select[Any]:
            identity = table.c.session_id.collate("BINARY")
            selected = select(
                table.c.session_id,
                func.min(table.c.ingested_commit_seq).label("first_input_seq"),
                func.max(table.c.ingested_commit_seq).label("last_input_seq"),
                func.max(table.c.ingested_at).label("last_ingested_at"),
            ).where(
                self._column("evidence", table, "tenant_id") == tenant,
                table.c.scope_kind == query.scope_kind,
                table.c.scope_id == query.scope_id,
            )
            if after is not None:
                selected = selected.where(identity > after)
            return selected.group_by(identity).order_by(identity).limit(limit)

        try:
            values = tuple(BenchSessionActivity.model_validate(row) for row in self._select(
                "evidence", statement, complete=False,
            ))
            keys = [value.session_id for value in values]
            if len(values) > limit or keys != sorted(set(keys)) or any(
                after is not None and key <= after for key in keys
            ) or any(value.first_input_seq > value.last_input_seq for value in values):
                raise ValueError("Session activity changed its exact page")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid", "Invalid stored session activity") from error
        return values

    def pending_finalizes(
        self, *, after: str | None = None, limit: int = 128,
    ) -> list[dict[str, Any]]:
        """Page nonterminal finalize roots for strict startup classification.

        The returned complete Run grants no epoch adoption or execution ticket.
        """
        self._check_page(after, limit, _ULID)

        def statement(table: Table) -> Select[Any]:
            selected = select(table).where(
                table.c.run_kind == "finalize", table.c.commit_mode == "orchestrator",
                table.c.parent_run_id.is_(None),
                table.c.status.in_(("queued", "running", "manual_attention")),
            )
            if after is not None:
                selected = selected.where(table.c.object_id > after)
            return selected.order_by(table.c.object_id).limit(limit)

        return self._select("run", statement)

    def active_finalize_leases(
        self, *, after: str | None = None, limit: int = 128,
    ) -> tuple[dict[str, Any], ...]:
        """Page durable preparing/held leases without retaining the active backlog.

        Full strict Controls are validated, including malformed states selected
        for rejection. Callers must recheck each current lease under the Writer
        before using its existing issued ticket; a page grants no authority.
        """
        from quotemem.core.controls import (
            ControlRowV1,
            HeldFinalizeLeaseV1,
            PreparingFinalizeLeaseV1,
        )

        self._check_page(after, limit, _SCOPE_ID)
        prefix = "lease:finalize:"

        def statement(table: Table) -> Select[Any]:
            key = table.c.key.collate("BINARY")
            state = self._control_state(
                table.c.value, ("preparing", "held", "released", "cancelled"),
            )
            selected = select(table).where(
                key >= prefix, key < "lease:finalize;",
                or_(state.not_in(("released", "cancelled")), state.is_(None)),
            )
            if after is not None:
                selected = selected.where(key > prefix + after)
            return selected.order_by(key).limit(limit)

        try:
            values = tuple(self._select("control", statement))
            keys: list[str] = []
            for raw in values:
                row = ControlRowV1.model_validate(raw)
                if not row.key.startswith(prefix) or not isinstance(
                    row.value, PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1,
                ):
                    raise ValueError("Active finalize lease changed its selected namespace")
                keys.append(_SCOPE_ID.validate_python(row.key[len(prefix):], strict=True))
            if len(values) > limit or keys != sorted(set(keys)) or any(
                after is not None and key <= after for key in keys
            ):
                raise ValueError("Active finalize leases changed their exact page")
        except (ValueError, ValidationError) as error:
            raise QuoteMemError("schema_invalid", "Invalid active finalize lease page") from error
        return values

    def _control_state(self, column: Any, states: tuple[str, ...]) -> Any:
        """Project a closed state through the bound HB JSON compiler.

        HB's native JSON column can contain the string produced by Json.encode.
        Its dialect compiler handles that storage envelope and scalar kinds.
        Unknown, null and malformed payloads stay NULL for strict caller rejection.
        The caller invokes this inside the existing workspace/backend read lock.
        """
        dialect = self._json_dialect()
        return case(*(
            (dialect.json_get_expression(column, ("state",), "eq", state), state)
            for state in states
        ), else_=None)

    def _json_dialect(self) -> Any:
        """Resolve JSON semantics only from the already-bound main backend."""
        from heavenbase.backends.sql.capabilities import sql_dialect_spec

        backend = self._workspace.backends.get("main")
        return sql_dialect_spec(backend.type, resolver=backend.module_resolver)

    def finalize_scope_headers(
        self, scope: ScopeV1, kind: Literal["evidence"],
        *, after: str | None = None, limit: int = 128,
    ) -> tuple[dict[str, Any], ...]:
        """Page all-state scope membership without hydrating source content.

        This closed discovery does not establish liveness or source eligibility.
        The initial empty-finalize consumer rejects any returned row, including
        expired/erased Evidence. Evidence retains its actual tenant column.
        """
        if kind != "evidence":
            raise QuoteMemError("invalid_request", "Finalize header kind is not supported")
        self._check_page(after, limit, _ULID)
        query = ScopeV1.model_validate(scope.model_dump(mode="json"))
        header = _EvidenceReferenceHeader

        def statement(table: Table) -> Select[Any]:
            selected = select(*self._columns(kind, table, header.model_fields))
            selected = selected.where(
                table.c.scope_kind == query.scope_kind, table.c.scope_id == query.scope_id,
            )
            if after is not None:
                selected = selected.where(table.c.object_id > after)
            return selected.order_by(table.c.object_id).limit(limit)

        try:
            values = tuple(header.model_validate(row) for row in self._select(
                kind, statement, complete=False,
            ))
            identities = [value.object_id for value in values]
            if len(values) > limit or identities != sorted(set(identities)) or any(
                (value.scope_kind, value.scope_id) != (query.scope_kind, query.scope_id)
                or after is not None and value.object_id <= after for value in values
            ):
                raise ValueError("Finalize membership changed its exact page")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid", "Invalid stored finalize membership") from error
        return tuple(value.model_dump(mode="json") for value in values)

    def latest_finalize_input(self, scope: ScopeV1, *, before: int) -> str | None:
        """Locate the latest earlier owner input for incremental inventory reuse.

        This is only a hint. The consumer validates its exact artifact scope,
        expanded members and semantic binding before borrowing any suffix.
        """
        selected = ScopeV1.model_validate(scope.model_dump(mode="python"))

        def statement(table: Table) -> Select[Any]:
            return select(table.c.input_ref).where(
                table.c.scope_kind == selected.scope_kind,
                table.c.scope_id == selected.scope_id,
                table.c.run_kind == "finalize", table.c.parent_run_id.is_(None),
                table.c.created_at < before, table.c.input_ref.is_not(None),
            ).order_by(table.c.created_at.desc(), table.c.object_id.desc()).limit(1)

        rows = self._select("run", statement, complete=False)
        return None if not rows else str(rows[0]["input_ref"])

    def scoped_request_ledger_headers(
        self, scope: ScopeV1, *, after: str | None = None, limit: int = 128,
    ) -> tuple[dict[str, Any], ...]:
        """Enumerate every exact-scope reception ledger, including retired sessions."""
        from quotemem.core.identity import control_id

        self._check_page(after, limit, _SESSION_HASH)
        selected = ScopeV1.model_validate(scope.model_dump(mode="json"))
        prefix = f"request_ledger:{selected.scope_id}"

        def statement(table: Table) -> Select[Any]:
            key = table.c.key.collate("BINARY")
            query = select(table.c.object_id, table.c.key, table.c.version).where(
                key >= prefix + ":", key < prefix + ";",
            )
            if after is not None:
                query = query.where(key > prefix + ":" + after)
            return query.order_by(key).limit(limit)

        try:
            values = tuple(_RequestLedgerHeader.model_validate(row) for row in self._select(
                "control", statement, complete=False,
            ))
            keys = [value.key for value in values]
            if len(values) > limit or keys != sorted(set(keys)):
                raise ValueError("Reception directory changed its exact page")
            for value in values:
                if not value.key.startswith(prefix + ":") or value.object_id != control_id(
                    value.key,
                ):
                    raise ValueError("Reception directory changed its namespace identity")
                session = _SESSION_HASH.validate_python(value.key[len(prefix) + 1:], strict=True)
                if after is not None and session <= after:
                    raise ValueError("Reception directory precedes its page boundary")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid", "Invalid stored reception directory") from error
        return tuple(value.model_dump(mode="json") for value in values)

    def inputless_adds(self, *, after: str | None = None, limit: int = 128) -> list[dict[str, Any]]:
        """Page the exact inputless ordinary Add startup header.

        Args:
            after (str | None): Exclusive last Run ULID from the preceding page.
            limit (int): SQL page bound from 1 through MAX_PAGE_SIZE (512).

        Returns:
            list[dict[str, Any]]: Complete session_ingestion/orchestrator Run
                rows with queued status, await_input stage, and NULL input_ref,
                in object_id order. Header selection does not authorize recovery,
                adoption, abandonment, or restoration of any orphan receipt.

        Raises:
            QuoteMemError: If the cursor, page size, or physical routing is invalid.
            RuntimeError: If the workspace or backend is closed.

        Notes:
            Each page uses the current stored view. The recovery owner must
            revalidate its final proof under the applicable Writer protocol.
        """
        self._check_page(after, limit, _ULID)

        def statement(table: Table) -> Select[Any]:
            query = select(table).where(
                table.c.run_kind == "session_ingestion",
                table.c.commit_mode == "orchestrator",
                table.c.status == "queued",
                table.c.stage == "await_input",
                table.c.input_ref.is_(None),
            )
            if after is not None:
                query = query.where(table.c.object_id > after)
            return query.order_by(table.c.object_id).limit(limit)

        return self._select("run", statement)

    def pending_privacy(
        self, scope_id: str | None = None, *, after: str | None = None, limit: int = 128
    ) -> list[dict[str, Any]]:
        """Page unresolved affected-set retention owners before ordinary Add recovery.

        Args:
            scope_id: Optional canonical exact scope, never a raw external identity.
            after: Exclusive last Run ULID from the preceding page.
            limit: Actual SQL page bound from 1 through 512.

        Returns:
            Complete queued/running retention/destructive_scan/affected_set headers.
            A caller must prove the strict input or reserve an inputless owner;
            this selector never assumes an orphan Artifact is an attached input.

        Raises:
            QuoteMemError: For invalid scalar bounds or physical routing.

        Notes:
            The Run kind index narrows candidates; status/mode/target predicates
            are residual filters. LIMIT bounds returned rows, not examined rows.
        """
        self._check_page(after, limit, _ULID)
        try:
            canonical = (
                None if scope_id is None else _SCOPE_ID.validate_python(scope_id, strict=True)
            )
        except ValidationError:
            raise QuoteMemError("invalid_request",
                                "Privacy lookup requires a canonical scope") from None

        def statement(table: Table) -> Select[Any]:
            query = select(table).where(
                table.c.run_kind == "retention",
                table.c.commit_mode == "destructive_scan",
                table.c.target_kind == "affected_set",
                table.c.status.in_(("queued", "running")),
            )
            if canonical is not None:
                query = query.where(table.c.scope_id == canonical)
            if after is not None:
                query = query.where(table.c.object_id > after)
            return query.order_by(table.c.object_id).limit(limit)

        return self._select("run", statement)

    def run_children(
        self, run_id: str, *, after: str | None = None, limit: int = 128
    ) -> list[dict[str, Any]]:
        """Read one parent's complete child headers with bound keyset paging.

        Args:
            run_id: Canonical parent Run ULID.
            after: Exclusive last child ULID.
            limit: Actual SQL page bound from 1 through 512.

        Returns:
            Full children ordered by object_id, selected through parent_run_id.

        Raises:
            QuoteMemError: For invalid identities, bounds, or physical routing.
        """
        identifier = self._run_id(run_id)
        self._check_page(after, limit, _ULID)

        def statement(table: Table) -> Select[Any]:
            query = select(table).where(table.c.parent_run_id == identifier)
            if after is not None:
                query = query.where(table.c.object_id > after)
            return query.order_by(table.c.object_id).limit(limit)

        # Every Run upsert retires its parent's child set, so this page is
        # proven by that membership and the rows it returned.
        return self._select("run", statement, membership=(
            membership_family("run-child"), membership_key("run-child", identifier),
        ))

    def run_effects(self, run_id: str, *, include_children: bool = True) -> bool:
        """Probe direct persisted effects with bounded SQL.

        Args:
            run_id (str): Exact producer/parent Run ULID.
            include_children (bool): Include child headers by default. False
                proves direct domain effects only; callers must inspect children.

        Returns:
            bool: Whether a child Run, a candidate it owns, or a revision,
                Evidence row, entity or tombstone it wrote names this Run
                through its physical producer field. All probes share one
                workspace read lock; each fully drains its LIMIT 1 response.
                Artifact/Control exemptions and request-bound Evidence without
                origin_run_id require separate caller proofs.

        Raises:
            QuoteMemError: If the Run identity or physical routing is invalid.
            RuntimeError: If the workspace or backend is closed.

        Notes:
            This is a narrow zero-effect prerequisite, not a complete
            graph traversal or permission to abandon the Run. Each producer
            column has an index, so an absent effect costs six
            index searches, never a table scan.
        """
        identifier = self._run_id(run_id)
        with self._workspace._lock.read():
            self._workspace._need_live()
            for kind, field in _EFFECT_PRODUCERS:
                if kind not in self._entities:
                    continue
                if kind == "run" and not include_children:
                    continue
                if self._select(
                    kind,
                    lambda table, column=field: select(table.c.object_id)
                    .where(table.c[column] == identifier)
                    .limit(1),
                    complete=False,
                ):
                    return True
            return False

    def control_provenance(
        self, run_id: str, *, after: str | None = None, limit: int = 128
    ) -> list[dict[str, Any]]:
        """Page full Controls last updated by one exact Run for exemption checks.

        Args:
            run_id (str): Producer Run ULID matched by updated_by_run_id.
            after (str | None): Exclusive last ControlId from the preceding page.
            limit (int): SQL page bound from 1 through MAX_PAGE_SIZE (512).

        Returns:
            list[dict[str, Any]]: Complete decoded Controls in object_id order,
                read through ix_control_run. No value or namespace
                is suppressed: only the recovery owner proves an exemption, such
                as a spent extraction allowance.

        Raises:
            QuoteMemError: If the Run, cursor, page bound, or routing is invalid.
            RuntimeError: If the workspace or backend is closed.
        """
        identifier = self._run_id(run_id)
        self._check_page(after, limit, _CONTROL_ID)

        def statement(table: Table) -> Select[Any]:
            query = select(table).where(table.c.updated_by_run_id == identifier)
            if after is not None:
                query = query.where(table.c.object_id > after)
            return query.order_by(table.c.object_id).limit(limit)

        return self._select("control", statement)

    @staticmethod
    def _run_id(run_id: str) -> str:
        try:
            return _ULID.validate_python(run_id, strict=True)
        except ValidationError:
            raise QuoteMemError(
                "invalid_request", "Indexed producer must be a canonical Run ID"
            ) from None

    def artifact_header(self, object_id: str) -> dict[str, Any] | None:
        """Read one exact immutable Artifact ownership header without body bytes.

        The locator may name a direct logical Artifact or a manifest. Callers
        must already hold the complete body proof; this metadata projection is
        suitable for final fixed-key checks backed by an unchanged generation.
        It cannot establish logical content integrity on its own.
        """
        try:
            identifier = _ARTIFACT_ID.validate_python(object_id, strict=True)
        except ValidationError:
            raise QuoteMemError("invalid_request",
                                "Artifact header requires a canonical ID") from None

        def statement(table: Table) -> Select[Any]:
            return select(*self._columns("artifact", table, (
                "object_id", "artifact_kind", "schema_version", "tenant_id", "scope_kind",
                "scope_id", "created_by_run_id",
            ))).where(table.c.object_id == identifier).limit(1)

        rows = self._select("artifact", statement, complete=False)
        return rows[0] if rows else None

    def artifact_headers(
        self, creator: str, kind: str, *, limit: int = 2,
    ) -> list[dict[str, Any]]:
        """Read bounded physical ownership headers without selecting Artifact body bytes.

        These headers establish locators and envelope ownership only. Logical
        content/address verification remains the authorized scoped reader's job.
        SQL applies the creator/kind predicates and the bound before decoding.
        """
        self._check_page(None, limit, _ARTIFACT_ID)
        try:
            query = ArtifactRows(created_by_run_id=creator, kinds=(kind,))
        except ValidationError:
            raise QuoteMemError(
                "invalid_request", "Artifact headers require a declared creator/kind",
            ) from None

        def statement(table: Table) -> Select[Any]:
            return select(*self._columns("artifact", table, (
                "object_id", "artifact_kind", "schema_version", "tenant_id", "scope_kind",
                "scope_id", "created_by_run_id",
            ))).where(
                table.c.created_by_run_id == query.created_by_run_id,
                table.c.artifact_kind == kind,
            ).order_by(table.c.object_id).limit(limit)

        return self._select("artifact", statement, complete=False)

    def artifact_reference_headers(
        self, creator: str, kind: str, *, after: str | None = None, limit: int = 128,
    ) -> list[dict[str, Any]]:
        """Page one creator's Artifacts of one kind with their stored references, no body bytes.

        SQL applies the creator/kind predicates, the exclusive object_id cursor
        and the bound before decoding. A header proves neither logical content
        nor its references: a caller that selects rows by ``artifact_refs``
        reads each selected Artifact through the logical reader, which verifies
        both. Like a full creator page, the page names the creator's Artifact
        set to active read records.
        """
        self._check_page(after, limit, _ARTIFACT_ID)
        try:
            query = ArtifactRows(created_by_run_id=creator, kinds=(kind,))
        except ValidationError:
            raise QuoteMemError(
                "invalid_request", "Artifact reference headers require a declared creator/kind",
            ) from None

        def statement(table: Table) -> Select[Any]:
            selected = select(*self._columns("artifact", table, (
                "object_id", "artifact_kind", "tenant_id", "scope_kind", "scope_id",
                "created_by_run_id", "artifact_refs",
            ))).where(
                table.c.created_by_run_id == query.created_by_run_id,
                table.c.artifact_kind == kind,
            )
            if after is not None:
                selected = selected.where(table.c.object_id > after)
            return selected.order_by(table.c.object_id).limit(limit)

        rows = self._select(
            "artifact", statement, complete=False, membership=_query_membership(query),
        )
        identities = [row["object_id"] for row in rows]
        if len(rows) > limit or identities != sorted(set(identities)) or any(
            (row["created_by_run_id"], row["artifact_kind"]) != (query.created_by_run_id, kind)
            or after is not None and row["object_id"] <= after
            or not isinstance(row["artifact_refs"], list)
            for row in rows
        ):
            raise QuoteMemError("schema_invalid",
                                "Artifact reference headers changed their exact page")
        return rows

    def artifact_ids(
        self, creator: str, *, after: str | None = None, limit: int = 128
    ) -> tuple[str, ...]:
        """Read only bounded physical identities for a creator's RowOp budget.

        The actual SQL uses the creator index and a keyset cursor. This method
        does not decode retained content or prove logical Artifact integrity;
        consumers use the normal logical reader when those proofs are needed.
        """
        identifier = self._run_id(creator)
        self._check_page(after, limit, _ARTIFACT_ID)

        def statement(table: Table) -> Select[Any]:
            query = select(table.c.object_id).where(table.c.created_by_run_id == identifier)
            if after is not None:
                query = query.where(table.c.object_id > after)
            return query.order_by(table.c.object_id).limit(limit)

        rows = self._select("artifact", statement, complete=False)
        return tuple(row["object_id"] for row in rows)

    def session_request_ids(
        self, scope: ScopeV1, session_id: str, *, after: str | None = None, limit: int = 128,
    ) -> tuple[str, ...]:
        """Page distinct raw request identities from every Evidence lifecycle.

        SQL selects only request_id for the exact scope/session, in ascending
        request_id order after the exclusive raw-ID cursor. No content, state,
        TTL or route filter is applied. The caller proves receipt completeness
        and a stable multi-page view under its existing generation/lifetime.

        Raises:
            QuoteMemError: For invalid scope/session/page arguments or malformed
                returned identifiers, ordering or page size.
        """
        try:
            selected_session = _EXTERNAL_ID.validate_python(session_id, strict=True)
        except ValidationError as error:
            raise QuoteMemError("invalid_request", "Invalid indexed session identity") from error
        query = self._scope_query(scope.scope_kind, scope.scope_id, selected_session)
        self._check_page(after, limit, _EXTERNAL_ID)

        def statement(table: Table) -> Select[Any]:
            selected = select(table.c.request_id).distinct().where(
                table.c.scope_kind == query.scope_kind, table.c.scope_id == query.scope_id,
                table.c.session_id == query.session_id,
            )
            if after is not None:
                selected = selected.where(table.c.request_id > after)
            return selected.order_by(table.c.request_id).limit(limit)

        rows = self._select("evidence", statement, complete=False)
        try:
            if any(set(row) != {"request_id"} for row in rows):
                raise ValueError("Request identity projection has other columns")
            values = tuple(_EXTERNAL_ID.validate_python(row["request_id"], strict=True)
                           for row in rows)
            if len(values) > limit or list(values) != sorted(set(values)) or (
                after is not None and any(value <= after for value in values)
            ):
                raise ValueError("Request identity projection violates its exact page")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid",
                                "Invalid stored session request identities") from error
        return values

    def session_source_coordinates(
        self, scope: ScopeV1, session_id: str, request_id: str,
        *, after: str | None = None, limit: int = 128,
    ) -> tuple[dict[str, Any], ...]:
        """Page only immutable receipt coordinates, including canonical tombstones.

        The exact scope/session/request predicates and exclusive object_id cursor
        run before decoding. Returned rows contain the nine _SourceCoordinate
        fields only; they grant no Evidence content, eligibility or erasure proof.

        Raises:
            QuoteMemError: For invalid coordinates/page arguments or malformed stored
                fields, scope/identity substitutions, ordering or page size.
        """
        self._check_page(after, limit, _ULID)
        try:
            query = EvidenceRows(
                scope_kind=scope.scope_kind, scope_id=scope.scope_id,
                session_id=_EXTERNAL_ID.validate_python(session_id, strict=True),
                request_id=_EXTERNAL_ID.validate_python(request_id, strict=True),
            )
        except ValidationError as error:
            raise QuoteMemError("invalid_request", "Invalid indexed source coordinates") from error

        def statement(table: Table) -> Select[Any]:
            selected = select(*self._columns(
                "evidence", table, _SourceCoordinate.model_fields,
            )).where(
                table.c.scope_kind == query.scope_kind, table.c.scope_id == query.scope_id,
                table.c.session_id == query.session_id, table.c.request_id == query.request_id,
            )
            if after is not None:
                selected = selected.where(table.c.object_id > after)
            return selected.order_by(table.c.object_id).limit(limit)

        try:
            values = tuple(_SourceCoordinate.model_validate(row) for row in self._select(
                "evidence", statement, complete=False, membership=_query_membership(query),
            ))
            identities = [value.object_id for value in values]
            if len(values) > limit or identities != sorted(set(identities)) or any(
                (value.scope_kind, value.scope_id, value.session_id, value.request_id)
                != (query.scope_kind, query.scope_id, query.session_id, query.request_id)
                or after is not None and value.object_id <= after for value in values
            ):
                raise ValueError("Source coordinate projection violates its exact page")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid",
                                "Invalid stored session source coordinates") from error
        return tuple(value.model_dump(mode="json") for value in values)

    def source_coordinate(self, object_id: str) -> dict[str, Any] | None:
        """Recheck one exact receipt coordinate without hydrating Evidence content.

        This fixed lookup complements a previously complete issued inventory.
        Its caller retains scope, receipt, generation and privacy authority.

        Raises:
            QuoteMemError: For an invalid ULID or malformed/substituted stored metadata.
        """
        identifier = self._run_id(object_id)

        def statement(table: Table) -> Select[Any]:
            return select(*self._columns("evidence", table, _SourceCoordinate.model_fields)).where(
                table.c.object_id == identifier,
            ).limit(1)

        rows = self._select("evidence", statement, complete=False)
        if not rows:
            return None
        try:
            value = _SourceCoordinate.model_validate(rows[0])
            if len(rows) != 1 or value.object_id != identifier:
                raise ValueError("Source coordinate projection changed its exact locator")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid", "Invalid stored source coordinate") from error
        return value.model_dump(mode="json")

    def source_ordering_header(self, object_id: str) -> dict[str, Any] | None:
        """Read exact original ordering metadata without hydrating Evidence content.

        The caller joins the nine coordinates to a complete actual receipt proof
        and seals this separate timestamp projection for fixed final rechecks.
        It grants no content, source eligibility, or whole-session authority.
        """
        identifier = self._run_id(object_id)

        def statement(table: Table) -> Select[Any]:
            return select(*self._columns(
                "evidence", table, _SourceOrderingHeader.model_fields,
            )).where(
                table.c.object_id == identifier,
            ).limit(1)

        rows = self._select("evidence", statement, complete=False)
        if not rows:
            return None
        try:
            value = _SourceOrderingHeader.model_validate(rows[0])
            if len(rows) != 1 or value.object_id != identifier:
                raise ValueError("Ordering header changed its exact locator")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid",
                                "Invalid stored source ordering header") from error
        return value.model_dump(mode="json")

    def evidence_reference_header(self, object_id: str) -> dict[str, Any] | None:
        """Read typed Evidence existence and ownership without content hydration.

        Args:
            object_id: Canonical Evidence locator in this reader's owned table.

        Returns:
            Only object_id, tenant_id, scope_kind and scope_id, or None for an
            absent locator. Erased, expired and inactive rows remain valid
            reference targets; the caller checks tenant and reference scope.
            This metadata grants no content or source eligibility authority.

        Raises:
            QuoteMemError: For an invalid locator or malformed/substituted metadata.
        """
        identifier = self._run_id(object_id)

        def statement(table: Table) -> Select[Any]:
            return select(*self._columns(
                "evidence", table, _EvidenceReferenceHeader.model_fields,
            )).where(
                table.c.object_id == identifier,
            ).limit(1)

        rows = self._select("evidence", statement, complete=False)
        if not rows:
            return None
        try:
            value = _EvidenceReferenceHeader.model_validate(rows[0])
            if len(rows) != 1 or value.object_id != identifier:
                raise ValueError("Evidence reference header changed its exact locator")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid",
                                "Invalid stored Evidence reference header") from error
        return value.model_dump(mode="json")

    def evidence_reference_headers(
        self, object_ids: Sequence[str],
    ) -> dict[str, dict[str, Any] | None]:
        """Read ``evidence_reference_header`` for several locators, 32 per statement.

        Returns:
            Each distinct locator's header, or None for an absent locator,
            validated exactly as the single read validates it.

        Raises:
            QuoteMemError: For any invalid locator or malformed/substituted metadata.
        """
        return self._headers_by_identity(
            object_ids, _EvidenceReferenceHeader, "Invalid stored Evidence reference header",
        )

    def source_ordering_headers(
        self, object_ids: Sequence[str],
    ) -> dict[str, dict[str, Any] | None]:
        """Read ``source_ordering_header`` for several locators, 32 per statement.

        Returns:
            Each distinct locator's header, or None for an absent locator,
            validated exactly as the single read validates it.

        Raises:
            QuoteMemError: For any invalid locator or malformed/substituted metadata.
        """
        return self._headers_by_identity(
            object_ids, _SourceOrderingHeader, "Invalid stored source ordering header",
        )

    def _headers_by_identity(
        self, object_ids: Sequence[str], header: type[StrictModel], message: str,
    ) -> dict[str, dict[str, Any] | None]:
        identifiers = list(dict.fromkeys(self._run_id(value) for value in object_ids))
        result: dict[str, dict[str, Any] | None] = dict.fromkeys(identifiers)
        for offset in range(0, len(identifiers), MAX_IN_VALUES):
            chunk = identifiers[offset:offset + MAX_IN_VALUES]

            def statement(table: Table, chunk: list[str] = chunk) -> Select[Any]:
                return select(*self._columns("evidence", table, header.model_fields)).where(
                    table.c.object_id.in_(chunk),
                )

            rows = self._select("evidence", statement, complete=False)
            try:
                for row in rows:
                    value = header.model_validate(row).model_dump(mode="json")
                    identifier = value["object_id"]
                    if identifier not in chunk or result[identifier] is not None:
                        raise ValueError("Evidence headers changed their exact locators")
                    result[identifier] = value
            except (ValidationError, ValueError) as error:
                raise QuoteMemError("schema_invalid", message) from error
        return result

    def entity_header(
        self, scope: ScopeV1, entity_id: str, *, state: EntityState = "active",
    ) -> dict[str, Any] | None:
        """Read one entity's content-free header when it has ``state`` in exactly ``scope``.

        This is the bounded single-row read reception's admission uses to check
        that an SDK-bound entity is active in the scope:
        the primary key with exact scope and state predicates.

        Args:
            scope: The exact scope the entity must belong to.
            entity_id: The entity ULID; the virtual ``user`` has no row.
            state: The state the entity must have, ``active`` unless stated.

        Returns:
            ``object_id``, ``scope_kind``, ``scope_id``, ``entity_type``, ``state``
            and ``commit_seq``, or None when the entity is absent, belongs to
            another scope or has another state.

        Raises:
            QuoteMemError: For an invalid scope, identity or state, or a malformed row.
        """
        scope = ScopeV1.model_validate(scope.model_dump(mode="json"))
        try:
            identifier = _ULID.validate_python(entity_id, strict=True)
            wanted = _ENTITY_STATE.validate_python(state, strict=True)
        except ValidationError:
            raise QuoteMemError("invalid_request", "Invalid indexed entity or state") from None

        def statement(table: Table) -> Select[Any]:
            return select(*(table.c[name] for name in _EntityHeader.model_fields)).where(
                table.c.object_id == identifier,
                table.c.scope_kind == scope.scope_kind,
                table.c.scope_id == scope.scope_id,
                table.c.state == wanted,
            ).limit(2)

        rows = self._select("entity", statement, complete=False)
        if not rows:
            return None
        try:
            value = _EntityHeader.model_validate(rows[0])
            if len(rows) != 1 or value.object_id != identifier:
                raise ValueError("Entity header changed its exact locator")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid", "Invalid stored entity header") from error
        return value.model_dump(mode="json")

    def reentry_candidates(
        self, scope: ScopeV1, *, owner_run_id: str, limit: int = MAX_REENTRY,
    ) -> list[dict[str, Any]]:
        """Select the pending positions one owner re-enters, in rotation order.

        The rows are the exact scope's ``pending`` candidates whose code is
        re-judged ``yes``, other than the owner's own, ascending by
        ``(reentry_count, created_commit_seq, position_key)`` and at most
        ``limit``, read in one statement through ``ix_candidate_status``.

        Args:
            scope: The owner's exact scope.
            owner_run_id: The selecting Add owner, whose own rows never re-enter.
            limit: At most this many rows, 1 to ``reentry_limit`` (64).

        Returns:
            The complete candidate rows in selection order.

        Raises:
            QuoteMemError: For an invalid scope, owner or limit.
        """
        scope = ScopeV1.model_validate(scope.model_dump(mode="json"))
        owner = self._run_id(owner_run_id)
        if type(limit) is not int or not 1 <= limit <= MAX_REENTRY:
            raise QuoteMemError("invalid_request", "A re-entry selection takes 1 to 64 positions")

        def statement(table: Table) -> Select[Any]:
            return select(table).where(
                table.c.scope_kind == scope.scope_kind,
                table.c.scope_id == scope.scope_id,
                table.c.status == "pending",
                table.c.reason_code.in_(_REENTERED_CODES),
                table.c.owner_run_id != owner,
            ).order_by(
                table.c.reentry_count, table.c.created_commit_seq, table.c.owner_run_id,
                table.c.unit_no, table.c.position,
            ).limit(limit)

        return self._select("candidate", statement)

    def evidence_eligibility_header(self, object_id: str) -> dict[str, Any] | None:
        """Read fixed-key source state and TTL before authorizing any content hydration."""
        identifier = self._run_id(object_id)

        def statement(table: Table) -> Select[Any]:
            return select(*self._columns(
                "evidence", table, _EvidenceEligibilityHeader.model_fields,
            )).where(table.c.object_id == identifier).limit(1)

        rows = self._select("evidence", statement, complete=False)
        if not rows:
            return None
        try:
            value = _EvidenceEligibilityHeader.model_validate(rows[0])
            if len(rows) != 1 or value.object_id != identifier:
                raise ValueError("Evidence eligibility header changed its exact locator")
        except (ValidationError, ValueError) as error:
            raise QuoteMemError(
                "schema_invalid", "Invalid stored Evidence eligibility header",
            ) from error
        return value.model_dump(mode="json")

    def scope_source_headers(
        self, scope: ScopeV1, *, high_watermark: int,
        after: str | None = None, limit: int = 128,
    ) -> tuple[dict[str, Any], ...]:
        """Page original scope receipt coordinates without hydrating any source body.

        Every lifecycle is included so an unavailable immediate receipt remains
        a visible gap. This metadata page never supplies content or permission
        to skip that gap; the caller proves its exact receipt and TTL inventory.
        """
        return tuple(value.model_dump(mode="json") for value in self.scope_headers(
            scope, high_watermark=high_watermark, after=after, limit=limit,
        ))

    def scope_headers(
        self, scope: ScopeV1, *, high_watermark: int | None = None,
        after: str | None = None, limit: int = MAX_PAGE_SIZE,
    ) -> tuple[FinalizeSourceHeaderV1, ...]:
        """Page the fixed source headers of a scope's Evidence of every state.

        Args:
            scope: The exact scope whose Evidence is paged.
            high_watermark: Only rows ingested at or before this commit, or all.
            after: Exclusive last object_id of the previous page.
            limit: Inclusive page bound from 1 through 512.

        Returns:
            The headers in ascending object_id order, without text or vectors.
            The page is provable by the scope's Evidence membership, as an
            ``EvidenceRows`` page is.

        Notes:
            The reception census reads every Evidence row of its exact scope,
            whatever its state or receipt. Discovering each page and then
            reading each member's header with a point query of its own cost one
            statement per row: three census passes of 564 parts issued 1,692 of
            them in one Add of a long replay. One statement returns the same
            rows in the same order. Neither content, liveness nor source
            eligibility is returned or granted.

        Raises:
            QuoteMemError: Invalid request or malformed/substituted stored header.
        """
        from quotemem.core.work import FinalizeSourceHeaderV1

        self._check_page(after, limit, _ULID)
        query = ScopeV1.model_validate(scope.model_dump(mode="json"))

        def statement(table: Table) -> Select[Any]:
            selected = select(*self._columns(
                "evidence", table, FinalizeSourceHeaderV1.model_fields,
            )).where(table.c.scope_kind == query.scope_kind, table.c.scope_id == query.scope_id)
            if high_watermark is not None:
                selected = selected.where(table.c.ingested_commit_seq <= high_watermark)
            if after is not None:
                selected = selected.where(table.c.object_id > after)
            return selected.order_by(table.c.object_id).limit(limit)

        rows = self._select("evidence", statement, complete=False, membership=(
            membership_family("evidence"),
            membership_key("evidence", f"{query.scope_kind}:{query.scope_id}"),
        ))
        try:
            members = tuple(_EvidenceReferenceHeader.model_validate(
                {name: row[name] for name in _EvidenceReferenceHeader.model_fields},
            ) for row in rows)
            identities = [value.object_id for value in members]
            if len(members) > limit or identities != sorted(set(identities)) or any(
                (value.scope_kind, value.scope_id) != (query.scope_kind, query.scope_id)
                or after is not None and value.object_id <= after for value in members
            ):
                raise ValueError("Finalize membership changed its exact page")
        except (KeyError, ValidationError, ValueError) as error:
            raise QuoteMemError("schema_invalid", "Invalid stored finalize membership") from error
        try:
            values = tuple(FinalizeSourceHeaderV1.model_validate(row) for row in rows)
        except ValidationError as error:
            raise QuoteMemError("schema_invalid",
                                "Invalid stored finalize source header") from error
        if high_watermark is not None and any(
            value.ingested_commit_seq > high_watermark for value in values
        ):
            raise QuoteMemError("schema_invalid", "Invalid stored scope source metadata")
        return values

    def source_headers(
        self, object_ids: Sequence[str],
    ) -> dict[str, FinalizeSourceHeaderV1 | None]:
        """Read the fixed source headers of several Evidence rows, 32 per statement.

        Args:
            object_ids: Evidence locators, of any scope and state.

        Returns:
            Each distinct locator's header as ``scope_headers`` returns it, or
            None for an absent row. No scope is implied: the caller compares
            each header's scope with the set it maintains.

        Raises:
            QuoteMemError: For an invalid locator or a malformed stored header.
        """
        from quotemem.core.work import FinalizeSourceHeaderV1

        identifiers = list(dict.fromkeys(self._run_id(value) for value in object_ids))
        result: dict[str, FinalizeSourceHeaderV1 | None] = dict.fromkeys(identifiers)
        for offset in range(0, len(identifiers), MAX_IN_VALUES):
            chunk = identifiers[offset:offset + MAX_IN_VALUES]

            def statement(table: Table, chunk: list[str] = chunk) -> Select[Any]:
                return select(*self._columns(
                    "evidence", table, FinalizeSourceHeaderV1.model_fields,
                )).where(table.c.object_id.in_(chunk))

            try:
                for row in self._select("evidence", statement, complete=False, membership=()):
                    value = FinalizeSourceHeaderV1.model_validate(row)
                    if value.object_id not in chunk or result[value.object_id] is not None:
                        raise ValueError("Evidence headers changed their exact locators")
                    result[value.object_id] = value
            except (ValidationError, ValueError) as error:
                raise QuoteMemError("schema_invalid",
                                    "Invalid stored finalize source header") from error
        return result

    @staticmethod
    def _check_page[T: str](after: str | None, limit: int, adapter: TypeAdapter[T]) -> None:
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
            raise QuoteMemError("invalid_request",
                                "Indexed page size is outside its resource bound")
        try:
            if after is not None:
                adapter.validate_python(after, strict=True)
        except ValidationError:
            raise QuoteMemError("invalid_request", "Indexed page cursor is invalid") from None

    def active_session_tail(
        self,
        scope_kind: ScopeKind,
        scope_id: str,
        session_id: str,
        *,
        now_ms: int,
        byte_budget: int = 512,
    ) -> list[dict[str, Any]]:
        """Read the original rows covering the active session's newest text suffix.

        Args:
            scope_kind (ScopeKind): Uses the exact scope kinds documented on EvidenceRows.
            scope_id (str): Canonical sc_ scope identity.
            session_id (str): Original external session identity, matched exactly.
            now_ms (int): Nonnegative trusted time in epoch milliseconds; expired
                rows (expires_at <= now_ms) and non-active rows are excluded.
            byte_budget (int): UTF-8 suffix budget from 1 through 512 bytes.

        Returns:
            list[dict[str, Any]]: Complete decoded Evidence in chronological
                (ingested_commit_seq, ordinal, part_no, object_id) order. The
                newest rows cover at least byte_budget bytes when available.
                The oldest boundary row remains complete, preserving byte offsets
                for the caller's scanner; fewer than byte_budget bytes may remain.

        Raises:
            QuoteMemError: If scope, session, time, budget, routing, or active text is invalid.
            RuntimeError: If the workspace or its backend is closed.

        Notes:
            At most byte_budget rows are fetched with SQL LIMIT, since each active
            Evidence must have nonempty text. This is a bounded page, not a retained
            snapshot. Callers must recheck identity before writing or using a tail.
            Pending and indexed routes are both included; text is never rewritten.
        """
        query = self._scope_query(scope_kind, scope_id, session_id)
        if type(now_ms) is not int or now_ms < 0:
            raise QuoteMemError("invalid_request",
                                "Session tail time must be nonnegative milliseconds")
        if type(byte_budget) is not int or not 1 <= byte_budget <= 512:
            raise QuoteMemError("invalid_request",
                                "Session tail byte budget must be between 1 and 512")

        def statement(table: Table) -> Select[Any]:
            # The tail feeds prior-context spans (identity, text, byte range);
            # the embedding vector is never read from it, so the projection
            # leaves the vector column in storage instead of decoding it for
            # every Add (375 MB of vectors in six minutes).
            return (
                select(*self._columns("evidence", table, _SESSION_TAIL_COLUMNS))
                .where(
                    table.c.scope_kind == query.scope_kind,
                    table.c.scope_id == query.scope_id,
                    table.c.session_id == query.session_id,
                    table.c.state == "active",
                    or_(table.c.expires_at.is_(None), table.c.expires_at > now_ms),
                )
                .order_by(
                    table.c.ingested_commit_seq.desc(),
                    table.c.ordinal.desc(),
                    table.c.part_no.desc(),
                    table.c.object_id.desc(),
                )
                .limit(byte_budget)
            )

        rows = self._select("evidence", statement, complete=False)
        total = 0
        selected: list[dict[str, Any]] = []
        for row in rows:
            text = row["text"]
            if not isinstance(text, str) or not text:
                raise QuoteMemError("parity_mismatch",
                                    "Active session Evidence has no original text")
            selected.append(row)
            total += len(text.encode("utf-8"))
            if total >= byte_budget:
                break
        return list(reversed(selected))

    @staticmethod
    def _scope_query(
        scope_kind: ScopeKind, scope_id: str, session_id: str | None = None
    ) -> EvidenceRows:
        try:
            return EvidenceRows(scope_kind=scope_kind, scope_id=scope_id, session_id=session_id)
        except ValidationError as error:
            raise QuoteMemError("invalid_request", "Invalid indexed scope or session") from error

    def _normalized(self, kind: str) -> bool:
        """Inspect bound declarations only; registration is checked inside _select."""
        from quotemem.store.normalization import LOGICAL_IMAGE_KINDS, NORMALIZED_SCHEMA_SPECS

        if kind not in LOGICAL_IMAGE_KINDS:
            return False
        entity: Any = self._entities.get(kind)
        if entity is None:
            raise QuoteMemError("schema_mismatch", "Indexed Entity is not registered")
        fields = set(entity.schema().fields)
        if fields == {"object_id", *NORMALIZED_SCHEMA_SPECS[kind].fields}:
            return True
        raise QuoteMemError("schema_mismatch", "Indexed Entity has an unsupported physical layout")

    def _tenant(self) -> str:
        if self._tenant_id is None:
            raise QuoteMemError("authority_unavailable",
                                "Normalized reads require workspace tenant")
        return self._tenant_id

    def _column(self, kind: str, table: Table, name: str) -> Any:
        """Resolve only the removed header scalars from their declared authorities."""
        if name in table.c:
            return table.c[name]
        if not self._normalized(kind):
            raise QuoteMemError("schema_mismatch", "Indexed header column is absent")
        if name == "tenant_id" and kind in {"evidence", "artifact", "recall"}:
            return literal(self._tenant()).label(name)
        if name == "byte_end" and kind == "evidence":
            return (
                table.c.byte_start + func.length(sql_cast(table.c.text, LargeBinary))
            ).label(name)
        raise QuoteMemError("schema_mismatch", "Indexed header has no declared scalar authority")

    def _columns(self, kind: str, table: Table, names: Collection[str]) -> list[Any]:
        columns = [self._column(kind, table, name) for name in names]
        if kind == "evidence" and "byte_end" in names and "byte_end" not in table.c:
            encoding = func.pragma_encoding().table_valued("encoding")
            columns.append(select(encoding.c.encoding).scalar_subquery().label("_aml_encoding"))
        return columns

    def _related_table(self, kind: str) -> Table:
        """Resolve a join target within the existing workspace/backend read owner."""
        entity: Any = self._entities.get(kind)
        if entity is None:
            raise QuoteMemError("schema_mismatch", "Indexed authority Entity is not registered")
        schema = entity.schema()
        plan = self._workspace.storage_plans.get(schema.entity_id)
        if self._workspace.entities.get(schema.entity_id) is not entity or plan is None or any(
            binding.backend != "main" or binding.strategy not in {"inline", "json", "vector"}
            for binding in plan.bindings
        ):
            raise QuoteMemError("schema_mismatch",
                                "Indexed authority requires registered main routing")
        table = cast(Table, self._workspace.backends.get("main").tables[schema.entity_id])
        if set(table.c.keys()) != set(schema.fields):
            raise QuoteMemError(
                "schema_mismatch", "Indexed authority requires complete physical columns",
            )
        return table

    def _recall_statement(
        self, table: Table, query: LabeledRecallRows | RecallChildren,
        after: str | None, limit: int,
    ) -> Select[Any]:
        """Select interaction authority in the same bounded statement as the child."""
        parent = table.alias("recall_parent")
        selected = select(
            *table.c, *(column.label("_parent__" + column.name) for column in parent.c),
        ).select_from(table.outerjoin(parent, table.c.parent_recall_id == parent.c.object_id))
        if isinstance(query, RecallChildren):
            selected = selected.where(table.c.parent_recall_id == query.parent_recall_id)
        else:
            invalid = or_(
                parent.c.object_id.is_(None), parent.c.record_class == "interaction",
                parent.c.scope_kind.is_(None), parent.c.scope_id.is_(None),
                parent.c.environment.is_(None),
            )
            selected = selected.where(
                table.c.record_class == "interaction",
                or_(invalid, and_(
                    parent.c.scope_kind == query.scope_kind,
                    parent.c.scope_id == query.scope_id,
                    parent.c.environment != "benchmark",
                    table.c.interaction.in_(("used", "feedback")),
                )),
            )
        if after is not None:
            selected = selected.where(table.c.object_id > after)
        return selected.order_by(table.c.object_id).limit(limit)

    def _restore_recall(self, row: dict[str, Any]) -> dict[str, Any]:
        from quotemem.store.domain import DOMAIN_ADAPTERS
        from quotemem.store.normalization import NORMALIZED_SCHEMA_SPECS
        from quotemem.store.normalized_row_codec import restore_row_image

        fields = {"object_id", *NORMALIZED_SCHEMA_SPECS["recall"].fields}
        if set(row) != fields | {"_parent__" + field for field in fields}:
            raise QuoteMemError("schema_mismatch", "Recall join has an unexpected physical shape")
        physical = {field: row[field] for field in fields}
        parent = {field: row["_parent__" + field] for field in fields}
        if parent["object_id"] is None:
            raise QuoteMemError("reference_missing", "Recall interaction has no retained delivery")
        entity: Any = self._entities["recall"]
        decoded = self._workspace._row_codec.decode_schema_row(entity.schema(), parent)
        if decoded["record_class"] == "interaction":
            raise QuoteMemError("parity_mismatch",
                                "Recall interaction cannot own another interaction")
        authority = DOMAIN_ADAPTERS["recall"].validate_python(
            restore_row_image("recall", decoded, tenant_id=self._tenant()),
        )
        return restore_row_image(
            "recall", physical, tenant_id=self._tenant(), parent_recall=authority,
        )

    def _read(
        self, query: IndexedQuery, *, after: str | None, limit: int, latest: bool,
        membership: tuple[tuple[str, str], ...] | None = None,
    ) -> list[dict[str, Any]]:
        kind = _QUERY_KINDS.get(type(query))
        if kind is None:
            raise QuoteMemError("unsupported_variant", "Indexed query type is not declared")
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
            raise QuoteMemError("invalid_request",
                                "Indexed page size is outside its resource bound")
        try:
            # Revalidate model_copy/construct inputs at the actual storage boundary.
            query = type(query).model_validate(vars(query))
            if after is not None:
                adapter = _ARTIFACT_ID if kind == "artifact" else (
                    _RECALL_ID if kind == "recall" else _ULID
                )
                adapter.validate_python(after, strict=True)
        except (ValidationError, ValueError) as error:
            raise QuoteMemError("invalid_request",
                                "Invalid indexed query or page cursor") from error

        normalized = isinstance(
            query, LabeledRecallRows | RecallChildren,
        ) and self._normalized(kind)
        if normalized:
            self._tenant()

        def statement(table: Table) -> Select[Any]:
            if normalized and isinstance(query, LabeledRecallRows | RecallChildren):
                return self._recall_statement(table, query, after, limit)
            return self._statement(table, query, after, limit, latest)

        if normalized and isinstance(query, LabeledRecallRows | RecallChildren):
            rows = [
                self._restore_recall(row)
                for row in self._select(kind, statement, complete=False)
            ]
            identities = [row["object_id"] for row in rows]
            for row in rows:
                if isinstance(query, LabeledRecallRows) and (
                    (row["scope_kind"], row["scope_id"]) != (query.scope_kind, query.scope_id)
                ):
                    raise QuoteMemError("schema_invalid", "Normalized page changed its exact scope")
                if isinstance(query, RecallChildren) and (
                    row["parent_recall_id"] != query.parent_recall_id
                ):
                    raise QuoteMemError("schema_invalid", "Normalized page changed its delivery")
                if isinstance(query, LabeledRecallRows) and (
                    row["record_class"] != "interaction"
                    or row["interaction"] not in {"used", "feedback"}
                    or row["environment"] == "benchmark"
                ):
                    raise QuoteMemError("schema_invalid",
                                        "Normalized page changed its label selection")
            if len(rows) > limit or identities != sorted(set(identities)) or any(
                after is not None and identifier <= after for identifier in identities
            ):
                raise QuoteMemError("schema_invalid", "Normalized page changed its exact bounds")
            return rows
        if membership is None:
            membership = _query_membership(query)
        return self._select(kind, statement, membership=membership)

    @attribute_cost("storage.indexed_read")
    def _select(
        self, kind: str, statement: Callable[[Table], Select[Any]], *, complete: bool = True,
        membership: tuple[tuple[str, str], ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Run one statement; ``membership`` names the set it enumerates, if provable.

        Active read records receive that set's membership keys and every row
        returned. Without them a scan cannot be proven by row keys, so it makes
        every active record unprovable.
        """
        if membership is None:
            note_unprovable_read()
        rows = self._selected(kind, statement, complete=complete)
        if membership is not None:
            note_reads((MEMBERSHIP_ANY, *membership,
                        *((kind, str(row["object_id"])) for row in rows)))
        return rows

    def _selected(
        self, kind: str, statement: Callable[[Table], Select[Any]], *, complete: bool = True
    ) -> list[dict[str, Any]]:
        workspace = self._workspace
        with workspace._lock.read():
            workspace._need_live()
            entity: Any = self._entities.get(kind)
            if entity is None:
                raise QuoteMemError("schema_mismatch", "Indexed Entity is not registered")
            schema = entity.schema()
            registered = workspace.entities.get(schema.entity_id)
            plan = workspace.storage_plans.get(schema.entity_id)
            if (
                registered is not entity
                or plan is None
                or any(
                    binding.backend != "main"
                    or binding.strategy not in {"inline", "json", "vector"}
                    for binding in plan.bindings
                )
            ):
                raise QuoteMemError("schema_mismatch",
                                    "Indexed reads require registered main routing")
            backend = workspace.backends.get("main")
            # The backend lock covers only the liveness check and the table and
            # database references this read needs, exactly as HeavenBase's own
            # read methods hold it; the statement executes on its own pooled
            # connection while other readers and the Writer proceed. Reads share
            # the workspace lock for the whole read, per HeavenBase's model.
            with backend._lock:
                backend._need_db()
                table = cast(Table, backend.tables[schema.entity_id])
                if set(table.c.keys()) != set(schema.fields):
                    raise QuoteMemError(
                        "schema_mismatch", "Indexed rows require inline physical columns"
                    )
                database = backend.db
                selected = statement(table)
            # The Evidence restorer reads its admission token now, before the
            # statement reads any row (``Writer._evidence_images``).
            evidence_image = (
                self._evidence_image()
                if complete and kind == "evidence" and self._evidence_image is not None
                else None
            )
            response = database.orm_execute(selected, readonly=True)
            try:
                raw: list[dict[str, Any]] = response.to_dicts()
            finally:
                response.close()
            # close() ends HB's cursor but retains its reusable fetched rows.
            # Our detached dictionaries own the result from here onward.
            del response
            decoded: list[dict[str, Any]] = []
            with cost_span("storage.decode"):
                for row in raw:
                    if "_aml_authority_invalid" in row and row.pop("_aml_authority_invalid"):
                        raise QuoteMemError("reference_missing",
                                            "Normalized header lacks its authority")
                    if "_aml_encoding" in row and row.pop("_aml_encoding") != "UTF-8":
                        raise QuoteMemError("schema_mismatch",
                                            "Byte projection requires UTF-8 storage")
                    value = workspace._row_codec.decode_schema_row(schema, row)
                    if complete:
                        workspace._materializer.assert_query_row_schema(entity, value)
                        if self._normalized(kind):
                            from quotemem.store.normalized_row_codec import restore_row_image

                            tenant = self._tenant()
                            if kind == "artifact" and self._artifact_image is not None:
                                value = self._artifact_image(value)
                            elif evidence_image is not None:
                                value = evidence_image(value)
                            else:
                                value = restore_row_image(kind, value, tenant_id=tenant)
                    decoded.append(value)
                    cost_count("storage.rows_decoded")
            return decoded

    @staticmethod
    def _statement(
        table: Table, query: IndexedQuery, after: str | None, limit: int, latest: bool,
    ) -> Select[Any]:
        statement = select(table)
        for field, value in query.model_dump().items():
            if value is None or field in {"kinds", "high_watermark", "name_norms", *_LOWER_BOUNDS}:
                continue
            statement = statement.where(table.c[field] == value)
        for field, column in _LOWER_BOUNDS.items():
            bound = getattr(query, field, None)
            if bound is not None:
                statement = statement.where(table.c[column] > bound)
        if isinstance(query, EntityNameRows) and query.name_norms:
            statement = statement.where(table.c.name_norm.in_(query.name_norms))
        if isinstance(query, ArtifactRows) and query.kinds:
            statement = statement.where(table.c.artifact_kind.in_(query.kinds))
        if isinstance(query, LabeledRecallRows):
            statement = statement.where(
                table.c.record_class == "interaction",
                table.c.interaction.in_(("used", "feedback")),
                table.c.environment != "benchmark",
            )
        if isinstance(query, RecallDeliveries):
            statement = statement.where(
                table.c.record_class.in_(("replayable_delivery", "minimal_ref_delivery")),
            )
        if isinstance(query, EvidenceRows) and query.high_watermark is not None:
            statement = statement.where(table.c.ingested_commit_seq <= query.high_watermark)
        if after is not None:
            statement = statement.where(table.c.object_id > after)
        ordering = table.c.run_generation.desc() if latest else table.c.object_id.asc()
        return statement.order_by(ordering).limit(limit)

