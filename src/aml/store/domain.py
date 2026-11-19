"""Strict rows of the AML tables (A.3): memory, Evidence, entity, run and recall rows.

Each model validates every rule the contract states for one row by itself.
Rules across rows (a head that matches its Item, an active anchor, carried
links, lifecycle transitions) are the single Writer's checks.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from aml.core.entities import (
    REVISION_REASONS,
    VERDICT_REF,
    ActivationMode,
    CandidateStatus,
    EffectiveBasis,
    EntityBindingV1,
    EntityRef,
    EntityState,
    MemoryKind,
    MemoryLifecycle,
    MemorySnapshotV2,
    NameKind,
    ReasonCode,
    RevisionOp,
    RevisionReason,
    SupersedeRelation,
    WithheldSpanV1,
)
from aml.core.errors import AMLError
from aml.core.identity import (
    ULID,
    ArtifactId,
    ContentBindingDigest,
    IntegrityDigest,
    PrincipalId,
    ScopeId,
    integrity_digest,
)
from aml.core.outcome_table import outcome_rule
from aml.core.query import FrozenScopeGrantSetV1, QueryBundleV1, ReturnedItemV1
from aml.core.semantic import (
    EmbeddingVector,
    HistoricalTimestamp,
    LongText,
    ScopeKind,
    SecurityLevel,
    ShortText,
    SystemTimestamp,
    TimeGranularity,
    TrustTier,
)
from aml.core.source_participants import SourceParticipantV1
from aml.core.statement import (
    ERASED_DIGEST,
    MAX_ABOUT_ENTITIES,
    MAX_ENTITY_BINDINGS,
    MAX_ENTITY_TYPE_CHARS,
    MAX_NAME_CHARS,
    MAX_NAME_NORM_CHARS,
    MAX_NAMING,
    MAX_REPLY_FACTS,
    MAX_STATEMENT_CHARS,
    MAX_TOMBSTONE_SOURCES,
    MAX_UNIT_NO,
    MAX_WITHHELD_SPANS,
    HexDigest,
    canonical_entity_type,
    canonical_statement,
    is_name,
    name_norm,
    statement_digest,
)
from aml.store.metadata import DerivedSpec, IntegrityBindingSpec, OpaqueIdSpec, RefSpec, StrictModel

TenantId = Annotated[str, Field(min_length=1, max_length=63)]
Nonnegative = Annotated[int, Field(ge=0)]
ERASED_CONTENT_HASH = IntegrityDigest("0" * 64)


class ByteRangeV1(StrictModel):
    """An external source byte interval."""

    byte_start: Nonnegative
    byte_end: Nonnegative

    @model_validator(mode="after")
    def check_order(self) -> Self:
        """Reject an inverted external interval."""
        if self.byte_end < self.byte_start:
            raise ValueError("Source byte range is inverted")
        return self


class SourceDescriptorV1(StrictModel):
    """Closed external source metadata; these are never AML foreign keys."""

    uri: Annotated[ShortText, OpaqueIdSpec("external")] | None = None
    document_id: Annotated[ShortText, OpaqueIdSpec("external")] | None = None
    file_path_hash: Annotated[IntegrityDigest, OpaqueIdSpec("digest")] | None = None
    mime: ShortText | None = None
    checksum: Annotated[IntegrityDigest, OpaqueIdSpec("digest")] | None = None
    tool_name: ShortText | None = None
    tool_call_id: Annotated[ShortText, OpaqueIdSpec("external")] | None = None
    byte_range: ByteRangeV1 | None = None


def _utf8_boundary(encoded: bytes, offset: int) -> bool:
    """Tell whether ``offset`` is a character boundary of the UTF-8 bytes ``encoded``."""
    return offset == len(encoded) or (
        0 <= offset < len(encoded) and (encoded[offset] & 0xC0) != 0x80
    )


class EvidenceRowV1(StrictModel):
    """Original bytes, exact source roles, and a closed erased tombstone branch.

    Two fields are new in the NL model (A.3.5). ``entity_bindings`` holds the
    SDK's bindings of message bytes to existing entities, null when the message
    had none. ``withheld_spans`` holds the byte ranges forget withheld, sorted
    by ``(byte_start, byte_end, reason, run_id)``. Both use original-message
    coordinates, lie inside this part and fall on character boundaries.
    Withholding never changes ``text``, ``content_hash`` or ``embedding``; an
    erased row has no bindings and no withheld ranges. That each bound entity
    is active in the scope, and that withheld ranges only grow, are Writer
    checks across rows.
    """

    object_id: Annotated[ULID, OpaqueIdSpec("local")]
    tenant_id: Annotated[TenantId, DerivedSpec("workspace.tenant_id"), OpaqueIdSpec("local")]
    scope_kind: ScopeKind
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    kind: Literal[
        "user_message",
        "assistant_message",
        "system_message",
        "tool_result",
        "document",
        "file",
        "external",
        "import",
    ]
    session_id: Annotated[ShortText, OpaqueIdSpec("external")]
    request_id: Annotated[ShortText, OpaqueIdSpec("external")]
    ordinal: Nonnegative
    role: Literal["user", "assistant", "tool", "system"] | None = None
    speaker_entity: ShortText | None = None
    source_participant: SourceParticipantV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    source_descriptor: SourceDescriptorV1 | None = None
    origin_run_id: Annotated[ULID, RefSpec("run")] | None = None
    text: LongText
    byte_start: Nonnegative
    byte_end: Nonnegative
    source_observed_at: HistoricalTimestamp | None = None
    ingested_at: SystemTimestamp
    ingested_commit_seq: Nonnegative
    part_no: Nonnegative
    time_granularity: TimeGranularity
    trust_tier: TrustTier
    security_level: SecurityLevel
    state: Literal["active", "invalidated", "expired", "erased"]
    route_state: Literal["pending", "indexed"]
    expires_at: SystemTimestamp | None = None
    content_hash: Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
    embedding: EmbeddingVector
    embedding_space: Annotated[ShortText, OpaqueIdSpec("local")]
    entity_bindings: (
        Annotated[list[EntityBindingV1], Field(min_length=1, max_length=MAX_ENTITY_BINDINGS)]
        | None
    ) = None
    withheld_spans: Annotated[list[WithheldSpanV1], Field(max_length=MAX_WITHHELD_SPANS)] = (
        Field(default_factory=list[WithheldSpanV1])
    )

    @model_validator(mode="after")
    def check_source_and_tombstone(self) -> Self:
        """Enforce the plan's source role matrix and fixed privacy placeholders."""
        if self.state == "erased":
            if (
                self.text
                or self.role is not None
                or self.source_descriptor is not None
                or self.speaker_entity is not None
                or self.source_participant is not None
                or self.byte_start
                or self.byte_end
                or self.route_state != "pending"
                or self.content_hash != ERASED_CONTENT_HASH
                or any(self.embedding)
                or self.entity_bindings is not None
                or self.withheld_spans
            ):
                raise ValueError("Erased Evidence must contain only fixed content placeholders")
            return self
        roles = {
            "user_message": "user",
            "assistant_message": "assistant",
            "system_message": "system",
            "tool_result": "tool",
        }
        if self.source_participant is not None and not self.speaker_entity:
            raise ValueError("Source participant requires its explicit original speaker")
        if self.role != roles.get(self.kind):
            raise ValueError("Evidence kind and role do not match")
        if (
            self.kind in {"document", "file", "external", "import", "tool_result"}
            and self.source_descriptor is None
        ):
            raise ValueError("This Evidence source requires a source_descriptor")
        if (
            self.kind in {"user_message", "assistant_message", "system_message"}
            and self.origin_run_id is not None
        ):
            raise ValueError("Conversation messages cannot claim a producing Run")
        if self.byte_end - self.byte_start != len(self.text.encode("utf-8")):
            raise ValueError("Evidence offsets must preserve the original UTF-8 bytes")
        if not any(self.embedding):
            raise ValueError("Non-erased Evidence embedding must be nonzero")
        expected = IntegrityDigest(hashlib.sha256(self.text.encode("utf-8")).hexdigest())
        if self.content_hash != expected:
            raise ValueError("Evidence content_hash does not match original bytes")
        self._check_part_ranges()
        return self

    def _check_part_ranges(self) -> None:
        """Keep bindings and withheld ranges inside this part, on character boundaries."""
        encoded = self.text.encode("utf-8")

        def inside(byte_start: int, byte_end: int) -> bool:
            return self.byte_start <= byte_start < byte_end <= self.byte_end and all(
                _utf8_boundary(encoded, offset - self.byte_start)
                for offset in (byte_start, byte_end)
            )

        bindings = self.entity_bindings or []
        if len(set(bindings)) != len(bindings):
            raise ValueError("Evidence entity_bindings are unique")
        if not all(inside(binding.byte_start, binding.byte_end) for binding in bindings):
            raise ValueError("An entity binding lies inside its part on character boundaries")
        keys = [
            (span.byte_start, span.byte_end, span.reason, span.run_id)
            for span in self.withheld_spans
        ]
        if keys != sorted(set(keys)):
            raise ValueError("withheld_spans are unique, sorted by (start, end, reason, run_id)")
        if not all(inside(span.byte_start, span.byte_end) for span in self.withheld_spans):
            raise ValueError("A withheld range lies inside its part on character boundaries")


class ItemRowV2(StrictModel):
    """One memory's identity, lifecycle and indexed head keys (A.3.1, A0.4).

    Content lives in the head revision's snapshot; the row keeps only what the
    indexes and lifecycle need. ``entity_id`` and ``statement_digest`` repeat the
    head's primary anchor and digest, and are empty and ``ERASED_DIGEST``
    exactly when the head is erased. Only an active Item changes lifecycle,
    and an erased one has an erased head. That the row agrees with its head and
    changes lifecycle only per A.3.1 are Writer checks (DM-02).
    """

    object_id: Annotated[ULID, OpaqueIdSpec("local")]
    scope_kind: ScopeKind
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    current_revision_id: Annotated[ULID, RefSpec("revision", scope_rule="same_scope")]
    entity_id: Annotated[
        EntityRef | Literal[""],
        DerivedSpec("current_revision.snapshot.entity_id"),
        OpaqueIdSpec("local"),
    ]
    statement_digest: Annotated[
        HexDigest, DerivedSpec("current_revision.snapshot.statement_digest"),
        OpaqueIdSpec("digest"),
    ]
    lifecycle_state: MemoryLifecycle
    activation_mode: ActivationMode
    commit_seq: Nonnegative
    retired_at: SystemTimestamp | None = None
    created_at: SystemTimestamp
    updated_at: SystemTimestamp

    @model_validator(mode="after")
    def check_head_keys(self) -> Self:
        """Pair the erased head keys and keep ``retired_at`` exactly off active Items."""
        erased_head = self.statement_digest == ERASED_DIGEST
        if (self.entity_id == "") != erased_head:
            raise ValueError("An Item's primary anchor is empty exactly when its head is erased")
        if erased_head and self.lifecycle_state == "active":
            raise ValueError("An active Item has a readable head")
        if self.lifecycle_state == "erased" and not erased_head:
            raise ValueError("An erased Item has an erased head")
        if (self.retired_at is None) != (self.lifecycle_state == "active"):
            raise ValueError("retired_at is null exactly while the Item is active")
        if self.updated_at < self.created_at:
            raise ValueError("An Item is updated no earlier than it was created")
        return self


class RevisionRowV2(StrictModel):
    """One immutable memory version: its op, reason and snapshot (A.3.2).

    The snapshot is the content and support authority; ``content_hash`` is its
    integrity digest and ``evidence_ids`` the sorted unique Evidence its quote
    spans cite. Version 1 creates the Item and only it has no predecessor; only
    a privacy erase writes the erased snapshot, and every revision except a
    requote, an erase or a rewording is written by its committing candidate. A
    rewording (C.9.4a) is a ``revise`` with no candidate that rewords a head a
    correction outdated: the correction that caused it wrote its own revision with
    its candidate, in the same transaction. Rows are deleted only by forget or purge.
    """

    object_id: Annotated[ULID, OpaqueIdSpec("local")]
    memory_id: Annotated[ULID, RefSpec("item", scope_rule="same_scope")]
    version: Annotated[int, Field(ge=1)]
    previous_revision_id: (
        Annotated[ULID, RefSpec("revision", scope_rule="same_scope")] | None
    ) = None
    op: RevisionOp
    reason_code: RevisionReason
    snapshot: MemorySnapshotV2
    content_hash: Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
    evidence_ids: Annotated[
        list[ULID],
        DerivedSpec("snapshot.quote_spans[].evidence_id"),
        RefSpec("evidence", cardinality="many", scope_rule="same_scope"),
    ]
    source_candidate_id: Annotated[
        ULID,
        RefSpec(
            "candidate", ref_mode="snapshot_locator", gc_root=False, scope_rule="same_scope",
            target_forget_action="retain_locator",
        ),
    ] | None = None
    origin_run_id: Annotated[ULID, RefSpec("run", target_forget_action="retain_locator")]
    policy_release_ref: Annotated[
        ArtifactId, RefSpec("artifact", artifact_kinds=("policy_release",)),
    ]
    commit_seq: Nonnegative
    recorded_at: SystemTimestamp

    @model_validator(mode="after")
    def check_version(self) -> Self:
        """Apply A.3.2's op, version, snapshot and support rules to this row."""
        if self.reason_code not in REVISION_REASONS[self.op]:
            raise ValueError("A revision op carries only its own reasons")
        if (self.version == 1) != (self.previous_revision_id is None):
            raise ValueError("previous_revision_id is null exactly for version 1")
        if (self.version == 1) != (self.op == "create"):
            raise ValueError("Exactly version 1 creates its Item")
        if self.snapshot.erased != (self.op == "erase"):
            raise ValueError("Exactly a privacy erase writes the erased snapshot")
        if self.op in {"requote", "erase"} and self.source_candidate_id is not None:
            raise ValueError("A requote or an erase names no candidate")
        if self.source_candidate_id is None and self.op not in {"requote", "erase"} and (
            (self.op, self.reason_code) != ("revise", "rewording")
        ):
            raise ValueError(
                "Every revision but a requote, an erase or a rewording names its candidate")
        if self.reason_code == "rewording" and self.source_candidate_id is not None:
            raise ValueError("A rewording names no candidate")
        if self.content_hash != integrity_digest(self.snapshot):
            raise ValueError("content_hash is the integrity digest of the snapshot")
        if self.evidence_ids != sorted({span.evidence_id for span in self.snapshot.quote_spans}):
            raise ValueError("evidence_ids are the sorted unique Evidence of the quote spans")
        return self


_LINKED_ITEM = RefSpec("item", ref_mode="snapshot_locator", gc_root=False, scope_rule="same_scope")
"""A ``supersedes`` or ``contradicts`` target; forgetting it deletes the link (A.3.3)."""

EXTRACTION_REF = RefSpec(
    "artifact", ref_mode="snapshot_locator", gc_root=True, artifact_kinds=("llm_output",),
    target_forget_action="retain_locator",
)
"""A candidate's unit reply (A.3.4): it exists when the row is written and is retained with
it, and the row never depends on it. A privacy erase deletes a reply whose request showed
erased content (C.13, D.7.5); the row keeps the locator, and a rewrite of a pending row keeps
it unchanged."""

PROPOSAL_REF = RefSpec(
    "artifact", ref_mode="snapshot_locator", gc_root=True,
    artifact_kinds=("proposal_snapshot",), target_forget_action="retain_locator",
)
"""A candidate's ``ProposalV2`` (A.3.4): it exists when the row is written and is retained
with it. A privacy erase deletes proposals that hold erased content (C.13, D.7.5);
correction provenance belongs to the content-free contradicts link (D.3)."""


class _LinkRowV2(StrictModel):
    """Shared link facts: the carrying revision and its Item's scope (A.3.3).

    A link is active while its source revision is the head of a non-erased
    Item; that and the carry rules are Writer checks.
    """

    object_id: Annotated[ULID, OpaqueIdSpec("local")]
    scope_kind: ScopeKind
    scope_id: Annotated[
        ScopeId, DerivedSpec("source_revision.item.scope_id"), OpaqueIdSpec("local"),
    ]
    source_revision_id: Annotated[ULID, RefSpec("revision", scope_rule="same_scope")]
    created_at: SystemTimestamp


class SupersedesLinkRowV2(_LinkRowV2):
    """The writing version ends its target memory by update, correction or retraction.

    Only an update records when the change took effect: ``effective_at`` with
    its basis, both or neither (A.3.3). ``correction_requested`` records that the
    writing proposal stated an explicit correction (``judge_structure.explicit_correction``):
    the target then carries the value the correction refuted, so a forget of the
    writing memory takes it whatever the relation (D.3, FG-2).
    """

    link_type: Literal["supersedes"]
    target_key: Annotated[ULID, _LINKED_ITEM]
    relation: SupersedeRelation
    effective_at: HistoricalTimestamp | None = None
    effective_basis: EffectiveBasis | None = None
    verdict_ref: Annotated[ArtifactId, VERDICT_REF]
    correction_requested: bool = False

    @model_validator(mode="after")
    def check_effective(self) -> Self:
        """Keep the effective time paired and on updates only."""
        if (self.effective_at is None) != (self.effective_basis is None):
            raise ValueError("effective_at and effective_basis are both present or both absent")
        if self.effective_at is not None and self.relation != "update":
            raise ValueError("Only an update records when its change took effect")
        return self


class ContradictsLinkRowV2(_LinkRowV2):
    """The writing version contradicts its target memory, which stays current."""

    link_type: Literal["contradicts"]
    target_key: Annotated[ULID, _LINKED_ITEM]
    relation: None = None
    effective_at: None = None
    effective_basis: None = None
    verdict_ref: Annotated[ArtifactId, VERDICT_REF]
    correction_requested: bool = False
    """Whether the creating proposal requested correction (C.1), retained without its text."""


class CorrectsHistoryLinkRowV2(_LinkRowV2):
    """A ``historical`` Item corrects one retired version, a revision (C.9.5a).

    Readers honour such a row on any revision of the correcting Item, its
    erased head included (the A.3.3 exception). Forgetting the corrected
    version's Item deletes the correcting Item with its rows (D.3 P2).
    """

    link_type: Literal["corrects_history"]
    target_key: Annotated[
        ULID,
        RefSpec(
            "revision", ref_mode="snapshot_locator", gc_root=False, scope_rule="same_scope",
            target_forget_action="cascade_source_item",
        ),
    ]
    relation: None = None
    correction_requested: Literal[False] = False
    effective_at: None = None
    effective_basis: None = None
    verdict_ref: Annotated[ArtifactId, VERDICT_REF]


class AboutEntityLinkRowV2(_LinkRowV2):
    """A projection of one of the snapshot's ``about_entity_ids`` (DM-04)."""

    link_type: Literal["about_entity"]
    target_key: Annotated[EntityRef, OpaqueIdSpec("local")]
    relation: None = None
    effective_at: None = None
    effective_basis: None = None
    verdict_ref: None = None
    correction_requested: Literal[False] = False


LinkRowV2 = Annotated[
    SupersedesLinkRowV2 | ContradictsLinkRowV2 | CorrectsHistoryLinkRowV2
    | AboutEntityLinkRowV2,
    Field(discriminator="link_type"),
]
"""One A.3.3 link; ``target_key`` is a memory, a revision or an ``EntityRef`` by type."""


class CandidateRowV2(StrictModel):
    """One admitted position: its disposition, targets and re-entry state (A.3.4).

    The row carries identity, lifecycle and indexed keys only; content lives
    in ``ProposalV2``. Its status follows from its reason code (C.10), it is
    terminal exactly when not pending, and it names a revision exactly when
    published. ``last_input_digest`` is the member input of its latest review,
    and ``last_wake_digest`` the wake key of its latest re-entry rebuild (C.3);
    both are null until the first such touch.
    """

    object_id: Annotated[ULID, OpaqueIdSpec("local")]
    scope_kind: ScopeKind
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    owner_run_id: Annotated[ULID, RefSpec("run", target_forget_action="retain_locator")]
    unit_no: Annotated[int, Field(ge=0, le=MAX_UNIT_NO)]
    position: Annotated[int, Field(ge=0, lt=MAX_REPLY_FACTS)]
    extraction_ref: Annotated[ArtifactId, EXTRACTION_REF]
    proposal_ref: Annotated[ArtifactId, PROPOSAL_REF]
    statement_digest: Annotated[
        HexDigest, DerivedSpec("proposal.statement_digest"), OpaqueIdSpec("digest"),
    ]
    entity_id: Annotated[EntityRef, OpaqueIdSpec("local")] | None = None
    status: CandidateStatus
    reason_code: ReasonCode
    judge_result_ref: Annotated[ArtifactId, VERDICT_REF] | None = None
    memory_id: Annotated[ULID, _LINKED_ITEM] | None = None
    revision_id: Annotated[
        ULID,
        RefSpec("revision", ref_mode="snapshot_locator", gc_root=False, scope_rule="same_scope"),
    ] | None = None
    target_memory_ids: Annotated[
        list[ULID],
        RefSpec(
            "item", cardinality="many", ref_mode="historical_locator", fulfillment="none",
            scope_rule="same_scope",
        ),
    ]
    evidence_ids: Annotated[
        list[ULID], Field(min_length=1),
        RefSpec("evidence", cardinality="many", scope_rule="same_scope"),
    ]
    hold_evidence: bool
    last_input_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")] | None = None
    last_wake_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")] | None = None
    reentry_count: Nonnegative
    created_commit_seq: Nonnegative
    commit_seq: Nonnegative
    created_at: SystemTimestamp
    terminal_at: SystemTimestamp | None = None

    @property
    def position_key(self) -> tuple[str, int, int]:
        """The position's ``(owner_run_id, unit_no, position)``, its unique key."""
        return (self.owner_run_id, self.unit_no, self.position)

    @model_validator(mode="after")
    def check_disposition(self) -> Self:
        """Apply C.10's status for the code and A.3.4's row-local rules."""
        if self.status not in outcome_rule(self.reason_code).statuses:
            raise ValueError("A candidate's status follows from its reason code (C.10)")
        if (self.terminal_at is None) != (self.status == "pending"):
            raise ValueError("terminal_at is null exactly while the candidate is pending")
        if (self.revision_id is not None) != (self.status == "published"):
            raise ValueError("A candidate names its revision exactly when it is published")
        if self.status == "published" and self.memory_id is None:
            raise ValueError("A published candidate names the memory it wrote")
        if self.hold_evidence and self.status != "pending":
            raise ValueError("Only a pending candidate holds its Evidence (D.6)")
        if self.evidence_ids != sorted(set(self.evidence_ids)) or (
            self.target_memory_ids != sorted(set(self.target_memory_ids))
        ):
            raise ValueError("Candidate Evidence and target identities are sorted and unique")
        if self.commit_seq < self.created_commit_seq or (
            self.terminal_at is not None and self.terminal_at < self.created_at
        ):
            raise ValueError("A candidate changes only after it was created")
        return self


class EntityRowV1(StrictModel):
    """One entity of a scope: type, state and its participant or mint key (A.3.6, UD2).

    It has no names: they derive from naming memories (``entity_name``). A
    participant has a ``participant_key``, an extraction-declared entity a
    ``mint_key`` and an SDK entity neither. An erased entity is terminal: its
    type is empty and a participant's key is cleared, so a returning
    participant gets a new entity.
    """

    object_id: Annotated[ULID, OpaqueIdSpec("local")]
    scope_kind: ScopeKind
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    entity_type: Annotated[str, Field(max_length=MAX_ENTITY_TYPE_CHARS)]
    participant_key: Annotated[HexDigest, OpaqueIdSpec("digest")] | None = None
    mint_key: Annotated[HexDigest, OpaqueIdSpec("digest")] | None = None
    state: EntityState
    created_by_run_id: Annotated[ULID, RefSpec("run", target_forget_action="retain_locator")]
    commit_seq: Nonnegative
    created_at: SystemTimestamp
    updated_at: SystemTimestamp

    @model_validator(mode="after")
    def check_entity(self) -> Self:
        """Apply A.3.6's type, key and erasure rules to this row."""
        erased = self.state == "erased"
        if (self.entity_type == "") != erased:
            raise ValueError("An entity's type is empty exactly when it is erased")
        if not erased and canonical_entity_type(self.entity_type) != self.entity_type:
            raise ValueError("An entity type is stored after NFC and trimming")
        if self.participant_key is not None and self.mint_key is not None:
            raise ValueError("An entity is a participant, a declared entity or an SDK entity")
        if erased and self.participant_key is not None:
            raise ValueError("An erased participant entity keeps no participant key")
        if self.updated_at < self.created_at:
            raise ValueError("An entity is updated no earlier than it was created")
        return self


class EntityNameRowV1(StrictModel):
    """One derived name of an entity: ``naming[annotation_index]`` of a head (A.3.7).

    Rows exist for every annotation of every active Item's non-erased head and
    are re-derived whenever a head or lifecycle changes (DM-05, a Writer check).
    """

    object_id: Annotated[ULID, OpaqueIdSpec("local")]
    scope_kind: ScopeKind
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    entity_id: Annotated[
        EntityRef, DerivedSpec("naming_revision.snapshot.naming[].entity_id"),
        OpaqueIdSpec("local"),
    ]
    name: Annotated[str, Field(min_length=1, max_length=MAX_NAME_CHARS)]
    name_norm: Annotated[str, Field(min_length=1, max_length=MAX_NAME_NORM_CHARS)]
    name_kind: NameKind
    naming_memory_id: Annotated[ULID, RefSpec("item", scope_rule="same_scope")]
    naming_revision_id: Annotated[ULID, RefSpec("revision", scope_rule="same_scope")]
    annotation_index: Annotated[int, Field(ge=0, lt=MAX_NAMING)]
    commit_seq: Nonnegative
    created_at: SystemTimestamp

    @model_validator(mode="after")
    def check_name(self) -> Self:
        """Require a valid name stored with its ``name_norm`` (A0.6)."""
        if not is_name(self.name):
            raise ValueError("An entity name is a valid A0.6 name")
        if self.name_norm != name_norm(self.name):
            raise ValueError("name_norm is the name's normalized form")
        return self


class TombstoneRowV1(StrictModel):
    """A forgotten version's key, unreadable statement and copied vector (A.3.8, UD3).

    One row per forgotten ``(entity_id, statement_digest)``. It names no
    memory, revision, quote or state; only ``TombstoneIndex`` and the Writer
    guard read it, and its statement is never readable. Its validity bounds
    are signed Integer milliseconds (F.7 H09).
    """

    object_id: Annotated[ULID, OpaqueIdSpec("local")]
    scope_kind: ScopeKind
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    entity_id: Annotated[EntityRef, OpaqueIdSpec("local")]
    about_entity_ids: Annotated[
        list[EntityRef], Field(max_length=MAX_ABOUT_ENTITIES), OpaqueIdSpec("local"),
    ]
    statement: Annotated[str, Field(max_length=MAX_STATEMENT_CHARS)]
    statement_digest: Annotated[HexDigest, OpaqueIdSpec("digest")]
    kind: MemoryKind
    valid_from: HistoricalTimestamp | None = None
    valid_to: HistoricalTimestamp | None = None
    time_granularity: TimeGranularity
    embedding: EmbeddingVector
    embedding_space: Annotated[ShortText, OpaqueIdSpec("local")]
    source_evidence_ids: Annotated[
        list[ULID], Field(min_length=1, max_length=MAX_TOMBSTONE_SOURCES),
        RefSpec("evidence", cardinality="many", scope_rule="same_scope"),
    ]
    forget_run_id: Annotated[ULID, RefSpec("run", target_forget_action="retain_locator")]
    tombstone_seq: Nonnegative
    created_at: SystemTimestamp

    @model_validator(mode="after")
    def check_tombstone(self) -> Self:
        """Apply the copied version's statement, anchor, time and vector rules."""
        if canonical_statement(self.statement) != self.statement:
            raise ValueError("A tombstone keeps its version's canonical statement")
        if self.statement_digest != statement_digest(self.statement):
            raise ValueError("statement_digest does not match the statement")
        about = self.about_entity_ids
        if about != sorted(set(about)) or self.entity_id in about:
            raise ValueError("about_entity_ids are sorted, unique and exclude the primary")
        bounds = (self.valid_from, self.valid_to)
        if (bounds == (None, None)) != (self.time_granularity == "unanchored"):
            raise ValueError("A tombstone has bounds exactly when its time is anchored")
        if self.kind == "state" and self.valid_to is not None:
            raise ValueError("Only an event carries valid_to")
        if self.valid_from is not None and self.valid_to is not None and (
            self.valid_from >= self.valid_to
        ):
            raise ValueError("valid_from precedes valid_to")
        if not any(self.embedding):
            raise ValueError("A tombstone copies its version's non-zero vector")
        if self.source_evidence_ids != sorted(set(self.source_evidence_ids)):
            raise ValueError("source_evidence_ids are sorted and unique")
        return self


RunKind = Literal[
    "session_ingestion",
    "finalize",
    "publish",
    "evaluate",
    "retention",
    "forget",
    "erasure_controller",
    "entity_update",
    "policy_release",
    "embedding_switch",
]
FailureCode = Literal[
    "erase_fenced",
    "consent_fenced",
    "stale_evidence",
    "input_abandoned",
    # A head-stale original whose single-item re-review cannot keep its operation.
    "stale_publication",
    "policy_reject",
    "insufficient_guard",
    "batch_plan_conflict",
    "provider_error",
    "schema_invalid",
    "grounding_failed",
    "artifact_digest_collision",
    "parity_mismatch",
    "lease_lost",
    "budget_exceeded",
    "cas_outcome_ambiguous",
]
RUN_STAGES: dict[str, frozenset[str]] = {
    # The reception root only: admission, durable input and ordering (store-Q9).
    "session_ingestion": frozenset({"await_input", "settle"}),
    # The Add owner (C.2a); no reextract or reconcile stage (store-Q10).
    "finalize": frozenset(
        {"await_input", "lease", "freeze", "extract", "settle", "review", "commit", "release"}
    ),
    "publish": frozenset({"await_input", "commit", "done"}),
    "evaluate": frozenset({"await_input", "run_cases", "verdict"}),
    "retention": frozenset({"await_input", "select", "compensate", "delete", "scan"}),
    "forget": frozenset(
        {
            "await_input",
            "resolve",
            "deny",
            "closure",
            "rewrite",
            "delete",
            "purge_derived",
            "scan",
            "receipt",
        }
    ),
    "erasure_controller": frozenset({"await_input", "plan", "dispatch", "verify", "receipt"}),
    "entity_update": frozenset({"await_input", "commit", "reconcile"}),
    "policy_release": frozenset({"await_input", "resolve", "write_release", "cas_current"}),
    "embedding_switch": frozenset(
        {
            "await_input",
            "resolve",
            "persist_intent",
            "reserve_generation",
            "cas_pointer",
            "publish_snapshot",
        }
    ),
}
# The admitted (run kind, target) pairs and their commit modes (A.3.10).
RUN_MODES: dict[tuple[str, str], frozenset[str]] = {
    ("session_ingestion", "session"): frozenset({"orchestrator"}),
    ("finalize", "scope"): frozenset({"orchestrator"}),
    ("publish", "scope"): frozenset({"single_compensatable"}),
    ("evaluate", "scope"): frozenset({"single_compensatable"}),
    ("retention", "candidate"): frozenset({"destructive_scan"}),
    ("retention", "scope"): frozenset({"destructive_scan"}),
    ("retention", "affected_set"): frozenset({"destructive_scan"}),
    ("forget", "memory"): frozenset({"destructive_scan"}),
    ("erasure_controller", "none"): frozenset({"orchestrator"}),
    ("entity_update", "scope"): frozenset({"single_compensatable"}),
    ("policy_release", "none"): frozenset({"cas_only"}),
    ("embedding_switch", "none"): frozenset({"cas_only"}),
}
GLOBAL_RUN_KINDS = frozenset({"policy_release", "embedding_switch", "erasure_controller"})




class _EvolutionRunRowV1(StrictModel):
    """One attempt's lifecycle and authoritative scope; no workflow execution here."""

    object_id: Annotated[ULID, OpaqueIdSpec("local")]
    run_kind: RunKind
    commit_mode: Literal["single_compensatable", "orchestrator", "destructive_scan", "cas_only"]
    status: Literal["queued", "running", "succeeded", "failed", "manual_attention"]
    stage: ShortText
    attempt: Nonnegative
    idempotency_key: Annotated[
        str, Field(pattern=r"^idem_[A-Za-z0-9_-]{43}$"), OpaqueIdSpec("digest")
    ]
    run_generation: Nonnegative
    admission_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
    parent_run_id: (
        Annotated[
            ULID,
            RefSpec(
                "run",
                ref_mode="snapshot_locator",
                fulfillment="pre_state",
                gc_root=False,
                scope_rule="same_scope",
            ),
        ]
        | None
    ) = None
    principal_id: Annotated[PrincipalId, OpaqueIdSpec("local")]
    scope_kind: ScopeKind | None = None
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")] | None = None
    auth_context_ref: (
        Annotated[
            ArtifactId,
            RefSpec("artifact", artifact_kinds=("auth_context",), target_forget_action="clear_ref"),
        ]
        | None
    ) = None
    input_ref: (
        Annotated[
            ArtifactId,
            RefSpec(
                "artifact",
                artifact_kinds=("run_input", "manifest"),
                target_forget_action="clear_ref",
            ),
        ]
        | None
    ) = None
    result_ref: (
        Annotated[
            ArtifactId,
            RefSpec(
                "artifact",
                artifact_kinds=("result_index", "manifest"),
                target_forget_action="clear_ref",
            ),
        ]
        | None
    ) = None
    commit_marker: Annotated[ArtifactId, OpaqueIdSpec("digest")] | None = None
    commit_seq: Annotated[Nonnegative | None, IntegrityBindingSpec("Run.commit_marker")] = None
    writer_epoch: Nonnegative
    admitted_erase_epoch: Nonnegative | None = None
    admitted_consent_epoch: Nonnegative | None = None
    failure_code: FailureCode | None = None
    created_at: SystemTimestamp
    started_at: SystemTimestamp | None = None
    updated_at: SystemTimestamp | None = None
    finished_at: SystemTimestamp | None = None

    @model_validator(mode="after")
    def check_run_state(self) -> Self:
        """Reject impossible marker, scope, lifecycle, and stage combinations."""
        target = str(getattr(self, "target_kind", ""))
        if self.run_kind == "retention" and target not in {"candidate", "affected_set", "scope"}:
            raise AMLError("unsupported_variant", "Retention target has no implemented run_input")
        if self.commit_mode not in RUN_MODES.get((self.run_kind, target), frozenset()):
            raise ValueError("Run kind, target, and commit mode violate the admitted matrix")
        if self.stage not in RUN_STAGES[self.run_kind]:
            raise ValueError("Stage is not declared for this Run kind")
        if (self.scope_kind is None) != (self.scope_id is None):
            raise ValueError("Run scope_kind and scope_id must be present together")
        scoped = self.scope_id is not None
        if scoped != (
            self.admitted_erase_epoch is not None and self.admitted_consent_epoch is not None
        ):
            raise ValueError("Scoped Runs require both admitted epochs")
        if not scoped and (
            self.admitted_erase_epoch is not None or self.admitted_consent_epoch is not None
        ):
            raise ValueError("Global Runs forbid admitted scope epochs")
        if self.run_kind in GLOBAL_RUN_KINDS and scoped:
            raise ValueError("This Run kind is global")
        if self.run_kind not in GLOBAL_RUN_KINDS | {"evaluate"} and not scoped:
            raise ValueError("This data Run kind requires one exact scope")
        if (self.commit_marker is None) != (self.commit_seq is None) and not (
            self.status == "manual_attention" and self.failure_code == "parity_mismatch"
        ):
            raise ValueError("Run commit_marker and commit_seq must be paired")
        if self.commit_mode != "single_compensatable" and (
            self.commit_marker is not None or self.commit_seq is not None
        ):
            raise ValueError("Only single_compensatable Runs can have a business marker")
        if (self.status in {"failed", "manual_attention"}) != (self.failure_code is not None):
            raise ValueError("Only failed/manual_attention Runs carry failure_code")
        if self.status == "queued" and (
            self.stage != "await_input"
            or self.started_at is not None
            or self.finished_at is not None
        ):
            raise ValueError("Queued Runs await input and have no execution timestamps")
        if self.status == "running" and (self.started_at is None or self.finished_at is not None):
            raise ValueError("Running Runs require started_at and forbid finished_at")
        if self.status in {"succeeded", "failed"} and self.finished_at is None:
            raise ValueError("Terminal Runs require finished_at")
        if self.status == "succeeded" and self.started_at is None:
            raise ValueError("Succeeded Runs require started_at")
        if (
            self.status == "failed"
            and self.started_at is None
            and self.failure_code
            not in {
                "input_abandoned",
                "stale_evidence",
                "erase_fenced",
                "consent_fenced",
            }
            and not (
                # Writer requires the dedicated privacy-child completion proof
                # before this unexecuted parent can settle (plan.md §5.6).
                self.failure_code == "policy_reject"
                and self.run_kind == "session_ingestion"
                and self.commit_mode == "orchestrator"
                and getattr(self, "target_kind", None) == "session"
                and self.stage == "settle"
                and self.attempt == 0
                and self.input_ref is None
                and self.result_ref is None
                and self.auth_context_ref is None
                and self.commit_marker is None
                and self.commit_seq is None
            )
        ):
            raise ValueError("Executed failures require started_at")
        if self.status == "manual_attention" and self.finished_at is not None:
            raise ValueError("Manual attention is not terminal")
        if self.parent_run_id == self.object_id:
            raise ValueError("A Run cannot be its own parent")
        if self.run_generation and self.run_kind != "session_ingestion":
            raise ValueError("Only external session families can have a recovery generation")
        return self


class SessionTargetRunRowV1(_EvolutionRunRowV1):
    """An irreversible internal session, scope, or affected-set locator."""

    target_kind: Literal["session", "scope", "affected_set"]
    target_id: Annotated[str, Field(min_length=1, max_length=63), OpaqueIdSpec("local")]

    @model_validator(mode="after")
    def check_affected_set_binding(self) -> Self:
        """The opened cleanup affected set is a keyed binding, never raw source bytes."""
        if self.target_kind == "affected_set":
            ContentBindingDigest(self.target_id)
        return self


class CandidateTargetRunRowV1(_EvolutionRunRowV1):
    """A Run targeting a frozen Candidate identity."""

    target_kind: Literal["candidate"]
    target_id: Annotated[ULID, RefSpec("candidate", ref_mode="snapshot_locator", gc_root=False)]


class MemoryTargetRunRowV1(_EvolutionRunRowV1):
    """A forget Run targeting the selected Item (store-Q11).

    The Run row outlives the Item it forgets, so its target is a content-free
    locator that is never dereferenced after the delete.
    """

    target_kind: Literal["memory"]
    target_id: Annotated[
        ULID,
        RefSpec(
            "item",
            ref_mode="snapshot_locator",
            gc_root=False,
            target_forget_action="retain_locator",
        ),
    ]


class UntargetedRunRowV1(_EvolutionRunRowV1):
    """A global bootstrap or coordinator with no invented target identity."""

    target_kind: Literal["none"]
    target_id: None = None


EvolutionRunRowV1 = Annotated[
    SessionTargetRunRowV1
    | CandidateTargetRunRowV1
    | MemoryTargetRunRowV1
    | UntargetedRunRowV1,
    Field(discriminator="target_kind"),
]


class _RecallEnvelopeV1(StrictModel):
    """Common scoped observation attributes, independent of the record class."""

    tenant_id: Annotated[TenantId, DerivedSpec("workspace.tenant_id"), OpaqueIdSpec("local")]
    scope_kind: ScopeKind
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    environment: Literal["production", "test", "eval", "shadow", "benchmark"]
    principal_id: Annotated[PrincipalId, OpaqueIdSpec("local")]
    security_level: SecurityLevel
    created_at: SystemTimestamp
    latency_ms: Nonnegative | None = None
    injected_tokens: Nonnegative | None = None


class _RecallRowV1(_RecallEnvelopeV1):
    """The common delivery receipt envelope; no search execution is implemented."""

    object_id: Annotated[ULID, OpaqueIdSpec("local")]
    parent_recall_id: None = None
    target_kind: None = None
    target_object_id: None = None
    target_revision_id: None = None
    interaction: None = None
    task_ref: None = None
    outcome: None = None
    score_parts: None = None


class ReplayableRecallRowV1(_RecallRowV1):
    """A replayable delivery retains its released QueryBundle and exact returned identities."""

    record_class: Literal["replayable_delivery"]
    trigger: Literal[
        "core_hook",
        "auto",
        "model_tool",
        "before_plan",
        "before_tool",
        "error",
        "retry",
        "low_confidence",
        "benchmark_adapter",
        "manual",
    ]
    query_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
    query_bundle: QueryBundleV1
    actual_release_ref: None = None
    allowed_scopes: FrozenScopeGrantSetV1
    snapshot_generation: Nonnegative
    returned: list[ReturnedItemV1]
    delivered_kind: Literal["core", "menu", "deep", "rendered_evidence"]


class MinimalRecallRowV1(_RecallRowV1):
    """A strict minimal delivery shape without frozen provider-dependent input."""

    record_class: Literal["minimal_ref_delivery"]
    query_bundle: None = None
    trigger: Literal[
        "core_hook",
        "auto",
        "model_tool",
        "before_plan",
        "before_tool",
        "error",
        "retry",
        "low_confidence",
        "benchmark_adapter",
        "manual",
    ]
    query_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
    actual_release_ref: Annotated[
        ArtifactId,
        RefSpec(
            "artifact",
            ref_mode="snapshot_locator",
            gc_root=True,
            artifact_kinds=("policy_release",),
        ),
    ]
    allowed_scopes: FrozenScopeGrantSetV1
    snapshot_generation: Nonnegative
    returned: list[ReturnedItemV1]
    delivered_kind: Literal["core", "menu", "deep", "rendered_evidence"]


RecallObservationId = Annotated[
    str, Field(pattern=r"^(?:[0-7][0-9A-HJKMNP-TV-Z]{25}|rci_[A-Za-z0-9_-]{43})$")
]


class TaskOutcomeBindingV1(StrictModel):
    """Existing external task locators from §5.8; they do not authorize attribution."""

    trace_id: Annotated[ShortText, OpaqueIdSpec("external")] | None = None
    task_id: Annotated[ShortText, OpaqueIdSpec("external")] | None = None
    tool_call_id: Annotated[ShortText, OpaqueIdSpec("external")] | None = None
    step_id: Annotated[ShortText, OpaqueIdSpec("external")] | None = None


class _InteractionRecallRowV1(_RecallEnvelopeV1):
    """Read an existing observation; this schema does not open Writer admission."""

    object_id: Annotated[RecallObservationId, OpaqueIdSpec("local")]
    record_class: Literal["interaction"]
    parent_recall_id: Annotated[
        ULID,
        RefSpec("recall", fulfillment="pre_state", target_forget_action="delete_owner"),
    ]
    interaction: Literal["expanded", "used", "feedback"]
    delivered_kind: Literal["none"]
    outcome: Literal["helpful", "irrelevant", "wrong", "outdated", "unsafe"] | None = None
    task_ref: TaskOutcomeBindingV1 | None = None
    trigger: None = None
    query_digest: None = None
    query_bundle: None = None
    actual_release_ref: None = None
    allowed_scopes: None = None
    snapshot_generation: None = None
    returned: None = None
    score_parts: None = None

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        """Keep feedback identity/outcome distinct from random expanded/used IDs."""
        feedback = self.interaction == "feedback"
        if feedback != (self.outcome is not None) or feedback != self.object_id.startswith("rci_"):
            raise ValueError("Feedback requires its deterministic identity and explicit outcome")
        if self.object_id == self.parent_recall_id:
            raise ValueError("An interaction cannot be its own delivery parent")
        return self


class MemoryInteractionRecallRowV1(_InteractionRecallRowV1):
    """A memory observation names the exact Revision frozen in its parent delivery."""

    target_kind: Literal["memory"]
    target_object_id: Annotated[ULID, RefSpec("item", target_forget_action="delete_owner")]
    target_revision_id: Annotated[ULID, RefSpec("revision", target_forget_action="delete_owner")]


class EvidenceInteractionRecallRowV1(_InteractionRecallRowV1):
    """An Evidence observation carries no invented Revision identity."""

    target_kind: Literal["evidence"]
    target_object_id: Annotated[ULID, RefSpec("evidence", target_forget_action="delete_owner")]
    target_revision_id: None = None


InteractionRecallRowV1 = Annotated[
    MemoryInteractionRecallRowV1 | EvidenceInteractionRecallRowV1,
    Field(discriminator="target_kind"),
]
RecallRowV1 = Annotated[
    ReplayableRecallRowV1 | MinimalRecallRowV1 | InteractionRecallRowV1,
    Field(discriminator="record_class"),
]


def validate_interaction_parent(
    interaction: MemoryInteractionRecallRowV1 | EvidenceInteractionRecallRowV1,
    parent: ReplayableRecallRowV1 | MinimalRecallRowV1,
) -> ReturnedItemV1:
    """Verify §5.8's exact existing parent/member relationship without granting reads.

    The caller must obtain both strict rows from the actual store and separately
    prove current authorization, privacy fences, retention and source integrity.
    This pure join never resolves a new current head for a delivered memory.
    """
    if interaction.parent_recall_id != parent.object_id or any(
        getattr(interaction, name) != getattr(parent, name)
        for name in ("tenant_id", "principal_id", "scope_kind", "scope_id", "environment")
    ) or parent.created_at > interaction.created_at:
        raise AMLError("parity_mismatch", "Interaction differs from its exact delivery parent")
    matches = [
        member for member in parent.returned
        if member.kind == interaction.target_kind
        and member.object_id == interaction.target_object_id
        and getattr(member, "revision_id", None) == interaction.target_revision_id
    ]
    if len(matches) != 1:
        raise AMLError("parity_mismatch", "Interaction lacks one exact delivered target version")
    return matches[0].model_copy(deep=True)


DOMAIN_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    "item": TypeAdapter(ItemRowV2),
    "revision": TypeAdapter(RevisionRowV2),
    "link": TypeAdapter(LinkRowV2),
    "candidate": TypeAdapter(CandidateRowV2),
    "evidence": TypeAdapter(EvidenceRowV1),
    "entity": TypeAdapter(EntityRowV1),
    "entity_name": TypeAdapter(EntityNameRowV1),
    "tombstone": TypeAdapter(TombstoneRowV1),
    "run": TypeAdapter(EvolutionRunRowV1),
    "recall": TypeAdapter(RecallRowV1),
}
"""The strict row parser of every domain table; ``control`` and ``artifact`` rows have
their own models in ``aml.core.controls`` and ``aml.core.artifacts``."""

