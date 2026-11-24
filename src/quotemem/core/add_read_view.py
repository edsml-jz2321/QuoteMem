"""The read view every Add-time stage reads through.

Context selection, the catalog, settlement, re-entry and nomination read one
consistent pin of one scope through `AddReadView`. Two implementations serve
it: the RecallSnapshot (`quotemem.core.recall_snapshot.SnapshotAddView`) in the
service and the in-memory lab store in the prompt lab;
a shared conformance suite runs against both.

A view returns readable content only. Memories, versions and names pass the
readability predicates of `quotemem.core.readability` at the view's pin and clock before a
view returns them; Evidence is returned in any state with its withheld ranges,
because providers see it only through ``Readability.render_masked``. Records
are plain frozen values and never touch storage. Tombstones never enter the
RecallSnapshot: the service view reads them from the owner-private
tombstone index through a `TombstoneSource`.

σ orders spans, messages and versions by source time, never by fact
time: ``(observed_at ?? received_at, ingested_commit_seq, ordinal, part_no,
byte_start)``, where ``received_at`` is the Evidence ``ingested_at``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from quotemem.core.entities import (
    ActivationMode,
    EffectiveBasis,
    EntityBindingV1,
    EvidenceKind,
    EvidenceState,
    MemoryKind,
    MemoryLifecycle,
    MemorySnapshotV2,
    NameKind,
    RevisionOp,
    RevisionReason,
    RouteState,
    SourceRole,
    SupersedeRelation,
    WithheldSpanV1,
)
from quotemem.core.semantic import ScopeKind, SecurityLevel, TimeGranularity, TrustTier
from quotemem.core.source_participants import SourceParticipantV1

Sigma = tuple[int, int, int, int, int]
"""σ: ``(source time, ingested_commit_seq, ordinal, part_no, byte_start)``."""

EndedBy = Literal["update", "correct", "retract", "supplement"]
"""How a history entry was ended: the next revision's reason or the link relation."""


class SigmaSource(Protocol):
    """The Evidence header fields σ reads."""

    @property
    def source_observed_at(self) -> int | None: ...

    @property
    def ingested_at(self) -> int: ...

    @property
    def ingested_commit_seq(self) -> int: ...

    @property
    def ordinal(self) -> int: ...

    @property
    def part_no(self) -> int: ...


def span_sigma(evidence: SigmaSource, byte_start: int) -> Sigma:
    """Return σ of the span that starts at ``byte_start`` in an Evidence part.

    Args:
        evidence (SigmaSource): The part the span lies in.
        byte_start (int): The span's first message byte.

    Returns:
        Sigma: The source time (``source_observed_at``, else ``ingested_at``),
            then ``ingested_commit_seq``, ``ordinal``, ``part_no`` and
            ``byte_start``.
    """
    observed = evidence.source_observed_at
    return (
        evidence.ingested_at if observed is None else observed,
        evidence.ingested_commit_seq,
        evidence.ordinal,
        evidence.part_no,
        byte_start,
    )


@dataclass(frozen=True, slots=True)
class ReadPin:
    """The consistent read a view serves (``pin``; stamps).

    Attributes:
        commit_seq (int): The global commit the view reflects.
        erase_epoch (int): The scope's ``erase_epoch`` at that commit.
        consent_epoch (int): The scope's ``consent_epoch`` at that commit.
        tombstone_seq (int): The scope's ``tombstone_seq`` at that commit.
        recall_snapshot_version (int): The RecallSnapshot generation read.
    """

    commit_seq: int
    erase_epoch: int
    consent_epoch: int
    tombstone_seq: int
    recall_snapshot_version: int


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One Evidence part: headers, text, vector and withheld ranges.

    Offsets are message coordinates: ``byte_start`` and ``byte_end`` locate the
    part inside its unsplit message, and withheld ranges and entity bindings
    use the same coordinates. An erased part keeps its placeholders: empty
    text, zero offsets and a zero vector.
    """

    object_id: str
    scope_kind: ScopeKind
    scope_id: str
    session_id: str
    request_id: str
    ordinal: int
    part_no: int
    kind: EvidenceKind
    role: SourceRole | None
    speaker_entity: str | None
    source_participant: SourceParticipantV1 | None
    source_observed_at: int | None
    ingested_at: int
    ingested_commit_seq: int
    byte_start: int
    byte_end: int
    time_granularity: TimeGranularity
    trust_tier: TrustTier
    security_level: SecurityLevel
    state: EvidenceState
    route_state: RouteState
    expires_at: int | None
    text: str
    embedding: tuple[float, ...]
    embedding_space: str
    withheld_spans: tuple[WithheldSpanV1, ...]
    entity_bindings: tuple[EntityBindingV1, ...]


@dataclass(frozen=True, slots=True)
class MessageKey:
    """The identity of one message: its session, request and ordinal in that request."""

    session_id: str
    request_id: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """Every Evidence part of one message, in any state, ordered by ``part_no``."""

    key: MessageKey
    parts: tuple[EvidenceRecord, ...]

    @property
    def sigma(self) -> Sigma:
        """σ of the message: that of its lowest-numbered part at the part's first byte."""
        first = self.parts[0]
        return span_sigma(first, first.byte_start)


@dataclass(frozen=True, slots=True)
class VersionRecord:
    """One readable revision of a memory: it passes ``p_ver`` at the view's pin.

    Attributes:
        memory_id (str): The memory.
        revision_id (str): This revision.
        version (int): Its version number, from 1.
        op (RevisionOp): The write that created it.
        reason_code (RevisionReason): The reason paired with ``op``.
        recorded_at (int): When it was recorded, in epoch milliseconds.
        commit_seq (int): The transaction that wrote it.
        snapshot (MemorySnapshotV2): Its content and support.
        sigma (Sigma): The maximum σ over its quote spans.
        tokens (tuple[str, ...]): ``memory_tokens(statement, keywords)``.
    """

    memory_id: str
    revision_id: str
    version: int
    op: RevisionOp
    reason_code: RevisionReason
    recorded_at: int
    commit_seq: int
    snapshot: MemorySnapshotV2
    sigma: Sigma
    tokens: tuple[str, ...]

    @property
    def anchors(self) -> tuple[str, ...]:
        """The primary anchor followed by the ``about_entity_ids``."""
        return self.snapshot.anchors


@dataclass(frozen=True, slots=True)
class ChainLink:
    """An active ``supersedes`` link from a successor's head to the memory it ended.

    Attributes:
        memory_id (str): The successor memory that carries the link.
        revision_id (str): The successor's head revision, the link's source.
        relation (SupersedeRelation): ``update``, ``correct`` or ``retract``.
        effective_at (int | None): When the update took effect in the world;
            only for ``update``.
        effective_basis (EffectiveBasis | None): What ``effective_at`` came from.
        verdict_ref (str): The judge result that decided the relation.
    """

    memory_id: str
    revision_id: str
    relation: SupersedeRelation
    effective_at: int | None
    effective_basis: EffectiveBasis | None
    verdict_ref: str


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A memory at the view's pin together with its readable current head.

    Attributes:
        memory_id (str): The memory.
        lifecycle_state (MemoryLifecycle): Its lifecycle.
        activation_mode (ActivationMode): Its activation mode.
        commit_seq (int): The last transaction that changed the memory.
        retired_at (int | None): When it stopped being active; None while active.
        head (VersionRecord): Its current revision, which passes ``p_ver``.
        successors (tuple[ChainLink, ...]): The active ``supersedes`` links that
            ended it, sorted by successor ``memory_id``; empty while it is active.
    """

    memory_id: str
    lifecycle_state: MemoryLifecycle
    activation_mode: ActivationMode
    commit_seq: int
    retired_at: int | None
    head: VersionRecord
    successors: tuple[ChainLink, ...]

    @property
    def current_revision_id(self) -> str:
        """The revision id of the current head."""
        return self.head.revision_id


@dataclass(frozen=True, slots=True)
class WithheldHeadRecord:
    """An active memory whose current head fails ``p_ver``: its identity only, never its content.

    A forget's withheld spans can leave a surviving sibling without a usable
    group; it stays unreadable until independent new support arrives.
    Settlement merges an exact restatement's new support into it;
    nothing of the head is returned, so no reader or provider sees it.

    Attributes:
        memory_id (str): The memory.
        current_revision_id (str): Its current, unreadable revision.
        entity_id (str): The head's primary anchor.
        about_entity_ids (tuple[str, ...]): The head's other anchors, sorted.
        support_groups (int): How many support groups the head has.
        quote_spans (int): How many quote spans the head has.
    """

    memory_id: str
    current_revision_id: str
    entity_id: str
    about_entity_ids: tuple[str, ...]
    support_groups: int
    quote_spans: int


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """A readable earlier version shown beside a head.

    Attributes:
        version (VersionRecord): The earlier version; its ``memory_id`` is the
            head's own memory, or a memory the head's chain ended.
        ended_by (EndedBy): For a revision of the same memory, the reason of the
            revision that replaced its statement (supplement → ``supplement``,
            correction → ``correct``, retraction → ``retract``); for the head of
            an ended memory, the relation that ended it.
        reworded (bool): A rewording replaced this version: its
            description was refuted while the memory's fact continues in the head,
            so it stays for the containment test but is never a retired
            version the corrects route or the judge's history context offers.
    """

    version: VersionRecord
    ended_by: EndedBy
    reworded: bool = False


@dataclass(frozen=True, slots=True)
class EntityNameRecord:
    """One ``entity_name`` row that passes ``p_name``."""

    object_id: str
    entity_id: str
    name: str
    name_norm: str
    name_kind: NameKind
    naming_memory_id: str
    naming_revision_id: str
    annotation_index: int
    commit_seq: int


@dataclass(frozen=True, slots=True)
class EntityRecord:
    """An active entity with its readable names.

    The virtual user has ``entity_id`` and ``entity_type`` ``"user"``, no keys
    and ``commit_seq`` 0.

    Attributes:
        entity_id (str): The `EntityRef`.
        entity_type (str): Free-text type; ``"person"`` for participants.
        participant_key (str | None): Set for a participant only.
        mint_key (str | None): Set for an extraction-declared entity only.
        commit_seq (int): The last transaction that changed the entity row.
        names (tuple[EntityNameRecord, ...]): Its ``p_name`` rows in catalog order:
            ``name`` < ``alias`` < ``former``, then newest ``commit_seq``, then
            ``name_norm``, then ``naming_revision_id`` and ``annotation_index``.
    """

    entity_id: str
    entity_type: str
    participant_key: str | None
    mint_key: str | None
    commit_seq: int
    names: tuple[EntityNameRecord, ...]


@dataclass(frozen=True, slots=True)
class TombstoneRecord:
    """One tombstone, read only by Add; its text never reaches any other read."""

    object_id: str
    scope_kind: ScopeKind
    scope_id: str
    entity_id: str
    about_entity_ids: tuple[str, ...]
    statement: str
    statement_digest: str
    kind: MemoryKind
    valid_from: int | None
    valid_to: int | None
    time_granularity: TimeGranularity
    embedding: tuple[float, ...]
    embedding_space: str
    tombstone_seq: int
    created_at: int = 0
    """When the forget that wrote it was admitted (epoch milliseconds): the ``agent`` mode
    learns again what the user states after it (``ingestion/relearn.py``)."""


@runtime_checkable
class TombstoneSource(Protocol):
    """The owner-private tombstone reads of one scope, as the TombstoneIndex serves them.

    The index is built per scope and ``tombstone_seq`` from tombstone rows and
    never enters the read model; an Add view takes it as a parameter and
    reports its sequence in the pin.
    """

    @property
    def scope_kind(self) -> ScopeKind:
        """The kind of the index's exact scope."""
        ...

    @property
    def scope_id(self) -> str:
        """The id of the index's exact scope."""
        ...

    @property
    def tombstone_seq(self) -> int:
        """The scope's ``tombstone_seq`` the index was built at."""
        ...

    def tombstones(self) -> tuple[TombstoneRecord, ...]:
        """Return every tombstone of the scope.

        Args:
            None.

        Returns:
            tuple[TombstoneRecord, ...]: Sorted by ``object_id``.
        """
        ...

    def tombstone(self, entity_id: str, statement_digest: str) -> TombstoneRecord | None:
        """Return the tombstone of one ``(entity_id, statement_digest)``, its unique key.

        Args:
            entity_id (str): The forgotten version's primary anchor.
            statement_digest (str): The forgotten statement's digest.

        Returns:
            TombstoneRecord | None: The tombstone, or None when the pair was never
                forgotten in the scope.
        """
        ...


@runtime_checkable
class AddReadView(Protocol):
    """One consistent read of one scope for the Add pipeline.

    Every method answers for the view's exact scope at its pin; another
    scope's rows behave as absent. Memories, versions and names are returned
    only when readable at the pin and clock; Evidence is returned in any
    state, to be checked with ``Readability.p_span`` and rendered only with
    ``Readability.render_masked``.
    """

    @property
    def scope_kind(self) -> ScopeKind:
        """The kind of the view's exact scope."""
        ...

    @property
    def scope_id(self) -> str:
        """The id of the view's exact scope."""
        ...

    @property
    def pin(self) -> ReadPin:
        """The commit, epochs and snapshot generation the view reflects."""
        ...

    @property
    def now(self) -> int:
        """The trusted clock of this read in epoch milliseconds, fixed for the view's life."""
        ...

    @property
    def embedding_space(self) -> str:
        """The workspace's active embedding space; every vector the view returns uses it."""
        ...

    def heads(self) -> tuple[MemoryRecord, ...]:
        """Return every ``p_mem`` memory of the scope: active with a head that passes ``p_ver``.

        Args:
            None.

        Returns:
            tuple[MemoryRecord, ...]: The memories, sorted by ``memory_id``.
        """
        ...

    def memory(self, memory_id: str) -> MemoryRecord | None:
        """Return one memory in any lifecycle when its current head passes ``p_ver``.

        Args:
            memory_id (str): The memory.

        Returns:
            MemoryRecord | None: The memory, or None when it does not exist in the
                scope or its head fails ``p_ver`` (an erased head always fails).
        """
        ...

    def history(self, memory_id: str) -> tuple[HistoryEntry, ...]:
        """Return the readable history entries of a memory's current head.

        The entries are (1) every earlier revision of the memory that a ``revise``
        replaced, that is every revision whose next version has op ``revise``,
        and (2) the current head of every memory that this memory's head ends
        through an active ``supersedes`` link. Only versions that pass ``p_ver``
        are returned; merges, requotes and erasures change no statement and end
        nothing.

        Args:
            memory_id (str): The memory whose head the entries belong to.

        Returns:
            tuple[HistoryEntry, ...]: Newest first by ``recorded_at``, then by
                ``revision_id`` descending; empty when the memory does not exist
                in the scope.
        """
        ...

    def retired(self) -> tuple[HistoryEntry, ...]:
        """Return the readable retired versions the corrects route scores.

        These are the history entries of every ``p_mem`` head, plus the current head
        of every memory whose lifecycle is ``superseded``, ``corrected`` or
        ``retracted`` (``ended_by`` ``update``, ``correct`` or ``retract``) when
        that head passes ``p_ver``. Each revision appears once.

        Args:
            None.

        Returns:
            tuple[HistoryEntry, ...]: Newest first by ``recorded_at``, then by
                ``revision_id`` descending.
        """
        ...

    def entity(self, entity_id: str) -> EntityRecord | None:
        """Return an active entity with its ``p_name`` names, or the virtual user.

        Args:
            entity_id (str): An `EntityRef`; ``"user"`` returns the virtual user
                with the names of naming memories that anchor it.

        Returns:
            EntityRecord | None: The entity, or None when it does not exist in
                the scope or is erased.
        """
        ...

    def participant(self, participant_key: str) -> EntityRecord | None:
        """Return the active participant with a key, as a ``participant_key`` lookup.

        Args:
            participant_key (str): A source tuple or labelled-speaker key.

        Returns:
            EntityRecord | None: The participant, or None before its first mint.
        """
        ...

    def speakers(self) -> Mapping[str, tuple[str, ...]]:
        """Return the speaker names source participants were shown under.

        A source participant is keyed by its tuple, so its names are the
        metadata ``speaker`` of its Evidence that passes ``p_src`` at the view's
        clock (`statement.speaker_index`). A labelled speaker needs none: its
        label keys it.

        Args:
            None.

        Returns:
            Mapping[str, tuple[str, ...]]: Each speaker name's ``name_norm`` with
                the participant keys whose readable Evidence carries it.
        """
        ...

    def minted(self, mint_key: str) -> EntityRecord | None:
        """Return the active entity minted for one extraction declaration.

        Args:
            mint_key (str): The declaration's ``mint_key``.

        Returns:
            EntityRecord | None: The entity, or None when it was never minted
                or an erasure ended it (`spent`).
        """
        ...

    def spent(self, mint_key: str) -> bool:
        """Whether an erasure ended the entity a declaration minted.

        A spent declaration is never minted again: its erased entity keeps the
        unique key. A slot naming it is open, as one naming an erased entity is.

        Args:
            mint_key (str): The declaration's ``mint_key``.

        Returns:
            bool: True when the declaration's entity was minted and then erased.
        """
        ...

    def entities(self) -> tuple[EntityRecord, ...]:
        """Return every active entity of the scope with its ``p_name`` names.

        Args:
            None.

        Returns:
            tuple[EntityRecord, ...]: Sorted by ``entity_id``; the virtual user
                is not listed.
        """
        ...

    def tombstones(self) -> tuple[TombstoneRecord, ...]:
        """Return every tombstone of the scope at the pin; only Add reads them.

        Args:
            None.

        Returns:
            tuple[TombstoneRecord, ...]: Sorted by ``object_id``.
        """
        ...

    def evidence(self, evidence_id: str) -> EvidenceRecord | None:
        """Return one Evidence part in any state.

        Args:
            evidence_id (str): The part.

        Returns:
            EvidenceRecord | None: The part, or None when it does not exist in
                the scope.
        """
        ...

    def message(self, key: MessageKey) -> MessageRecord | None:
        """Return every part of one message.

        Args:
            key (MessageKey): The message.

        Returns:
            MessageRecord | None: Its parts in any state by ``part_no``, or None
                when the scope has no part of it.
        """
        ...

    def messages_before(self, key: MessageKey, limit: int) -> tuple[MessageRecord, ...]:
        """Return the messages immediately before one message in its session.

        Session order is ascending message σ, then ``request_id`` and
        ``ordinal``; messages of other sessions never appear.

        Args:
            key (MessageKey): The message to look before.
            limit (int): The most messages to return; 0 or more.

        Returns:
            tuple[MessageRecord, ...]: Up to ``limit`` messages, oldest first;
                empty when the message is the session's first or does not exist.
        """
        ...

    def heads_by_digest(self, statement_digest: str) -> tuple[MemoryRecord, ...]:
        """Return the ``p_mem`` memories whose head has one statement digest (keyed read).

        Settlement looks up restatements by digest only; the service
        view answers from the RecallSnapshot's digest index.

        Args:
            statement_digest (str): The ``statement_digest`` to look up.

        Returns:
            tuple[MemoryRecord, ...]: The readable active memories whose current
                head has that digest, sorted by ``memory_id``.
        """
        ...

    def withheld_heads(self, statement_digest: str) -> tuple[WithheldHeadRecord, ...]:
        """Return the active memories whose unreadable head has one statement digest.

        Only settlement asks, for an exact restatement's new support; the
        records carry identities, never content.

        Args:
            statement_digest (str): The ``statement_digest`` to look up.

        Returns:
            tuple[WithheldHeadRecord, ...]: The active memories whose current head
                has that digest and fails ``p_ver``, sorted by ``memory_id``.
        """
        ...

    def entities_by_name(self, name_norm: str) -> tuple[EntityRecord, ...]:
        """Return the active entities with a readable name of one normal form (keyed read).

        Settlement checks a declared new entity's name against the scope's live
        names; the service view answers from the RecallSnapshot's name index.

        Args:
            name_norm (str): The ``name_norm`` to look up.

        Returns:
            tuple[EntityRecord, ...]: Every active entity, the virtual user
                included, with a ``p_name`` name of that ``name_norm``, with all its
                ``p_name`` names, sorted by ``entity_id``.
        """
        ...

    def tombstone(self, entity_id: str, statement_digest: str) -> TombstoneRecord | None:
        """Return the tombstone of one ``(entity_id, statement_digest)`` (keyed read).

        Settlement denies a forgotten statement of the same entity;
        the service view answers from the owner-private tombstone index.

        Args:
            entity_id (str): The resolved primary anchor.
            statement_digest (str): The statement's digest.

        Returns:
            TombstoneRecord | None: The scope's tombstone of that pair, or None.
        """
        ...
