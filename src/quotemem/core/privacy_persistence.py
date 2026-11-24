"""Content-free privacy-erase plans, progress and receipts.

A plan stores locators, keyed row-state digests and closed transforms, never a
post-image or source text. Recovery rebuilds each temporary post-image from the
verified existing rows and the closed transform, then compares the complete
keyed digests before it applies any effect.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Self, get_args

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import (
    ULID,
    ArtifactId,
    ContentBindingDigest,
    IntegrityDigest,
    canonical_json,
    integrity_digest,
)
from quotemem.core.statement import MAX_TOMBSTONE_SOURCES
from quotemem.core.technical_types import NonNegativeInt, PositiveInt, Timestamp
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

if TYPE_CHECKING:
    from quotemem.store.domain import EvidenceRowV1

RowDigest = Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
Integrity = Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
NotCovered = Literal[
    "wal", "backups", "replicas", "provider_copies", "host_session_logs", "existing_exports"
]
"""What a privacy receipt cannot cover: copies outside the workspace's rows."""
ResidualKind = Literal[
    "item", "revision", "link", "candidate", "evidence", "entity", "entity_name", "tombstone",
    "control", "run", "recall", "artifact", "llm_output", "proposal_snapshot",
    "semantic_embedding", "publish_plan", "write_set", "result_index", "finalize_plan",
    "request_ledger", "decision_set", "erasure_plan", "manifest", "auth_context", "receipt",
    "policy_config", "policy_release", "run_input",
]
"""A stored table, or an Artifact's logical kind (``artifact`` for its envelope)."""


class CredentialFindingsV1(StrictModel):
    """Recognized credentials an erase found in its scope's stored content: counts.

    ``records_deleted`` counts the logical records deleted because one held a credential
    the reception detector recognizes: an ended owner's model replies and proposals and
    the judge records naming them, deleted as the erasure manifest deletes them.
    ``retained`` counts, by stored kind, the rows still holding one that no rule of this
    erase removes: a running owner's records, which its recovery reads, and Evidence,
    memory versions, names and every other row, which only their own privacy erase or
    forget removes. No text, locator or digest.
    """

    records_deleted: NonNegativeInt = 0
    retained: dict[ResidualKind, PositiveInt] = Field(default_factory=dict[ResidualKind, int])


def _no_findings(value: CredentialFindingsV1) -> bool:
    return not value.records_deleted and not value.retained


class AffectedEvidenceV1(StrictModel):
    """Freeze an existing target without retaining its text or a reversible before-image."""

    evidence_id: Annotated[
        ULID,
        RefSpec("evidence", ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False),
    ]
    expected_row_digest: RowDigest


class _PrivacyEraseInputV1(StrictModel):
    """One erase's affected Evidence set and the scope state it was selected at.

    The owning Run supplies scope; no source byte is copied, and incoming
    boundary bytes are excluded.
    """

    schema_version: Literal[1]
    affected: Annotated[list[AffectedEvidenceV1], Field(min_length=1, max_length=512)]
    expected_scope_seq: NonNegativeInt
    erase_epoch: NonNegativeInt
    consent_epoch: NonNegativeInt

    @model_validator(mode="after")
    def check_targets(self) -> Self:
        """One bounded selection has a deterministic, duplicate-free target set."""
        keys = [entry.evidence_id for entry in self.affected]
        if keys != sorted(set(keys)):
            raise ValueError("Privacy erase targets must be sorted and unique")
        return self


class _SplitSecretInputV1(_PrivacyEraseInputV1):
    """Split-secret cleanup, privacy erase's one product producer."""

    retention_role: Literal["split_secret"]
    reason_code: Literal["retroactive_split_secret"]


class PreAddSplitSecretInputV1(_SplitSecretInputV1):
    """An independent internal retention root precedes rejected Add admission."""

    trigger: Literal["pre_add"]


class IngestionSplitSecretInputV1(_SplitSecretInputV1):
    """A dedicated child binds its already admitted parent without copying source bytes."""

    trigger: Literal["ingestion_child"]
    parent_binding_digest: RowDigest


class SourceExpiryInputV1(_PrivacyEraseInputV1):
    """Erase sources whose ``expires_at`` has passed (source expiry).

    No public route sets ``expires_at`` in v1; only test fixtures and the
    controlled-clock harness select this erase.
    ``observed_at`` is the trusted time of the selection: every target's
    ``expires_at`` is at or before it, and admission checks it against the
    stored rows and its own clock.
    """

    retention_role: Literal["source_expiry"]
    reason_code: Literal["source_expired"]
    trigger: Literal["clock"]
    observed_at: Timestamp


class ConsentWithdrawalInputV1(_PrivacyEraseInputV1):
    """Erase sources whose consent was withdrawn (consent withdrawal).

    The withdrawal advances ``consent_epoch:{s}``, which fences every operation
    admitted under the old epoch; this erase is admitted at the advanced epoch.
    No public route advances ``consent_epoch`` in v1; only test fixtures do.
    """

    retention_role: Literal["consent_withdrawal"]
    reason_code: Literal["consent_withdrawn"]
    trigger: Literal["consent_epoch"]

    @model_validator(mode="after")
    def check_withdrawn(self) -> Self:
        """A withdrawal has advanced the scope's consent epoch at least once."""
        if self.consent_epoch < 1:
            raise ValueError("Consent withdrawal requires an advanced consent epoch")
        return self


SplitSecretInputV1 = Annotated[
    PreAddSplitSecretInputV1 | IngestionSplitSecretInputV1, Field(discriminator="trigger")
]
SPLIT_SECRET_INPUT: TypeAdapter[SplitSecretInputV1] = TypeAdapter(SplitSecretInputV1)
PrivacyEraseInputV1 = Annotated[
    PreAddSplitSecretInputV1
    | IngestionSplitSecretInputV1
    | SourceExpiryInputV1
    | ConsentWithdrawalInputV1,
    Field(discriminator="trigger"),
]
"""Every privacy erase's admitted selection: split secret, source expiry, consent withdrawal."""
PRIVACY_ERASE_INPUT: TypeAdapter[PrivacyEraseInputV1] = TypeAdapter(PrivacyEraseInputV1)


class PrivacyCountsV1(StrictModel):
    """Logical affected rows by kind; physical RowOp accounting belongs to the Writer.

    ``candidate`` counts the pending candidates an erase expires and
    ``tombstone`` the tombstones it deletes once all their sources are erased;
    the receipt declares both.
    """

    evidence: NonNegativeInt
    item: NonNegativeInt
    revision: NonNegativeInt
    link: NonNegativeInt
    candidate: NonNegativeInt
    tombstone: NonNegativeInt


ZERO_COUNTS = PrivacyCountsV1(evidence=0, item=0, revision=0, link=0, candidate=0, tombstone=0)
"""The counts of a prepared erase, before its first batch applies."""


class PrivacyEraseProgressV1(StrictModel):
    """A privacy erase's maintenance fence: phase, cursor and the completed-batch proof.

    ``erase_epoch_bumps`` counts the ``erase_epoch:{s}`` advances of the owner's
    committed batches: the first batch advances it, and so does
    every batch that deletes a tombstone. The owner's ticket stays at its
    admitted epoch, and the Writer admits it at that epoch plus these bumps.
    """

    state: Literal["active"]
    fence_generation: PositiveInt
    owner_run_id: Annotated[ULID, RefSpec("run", scope_rule="same_scope")]
    evidence_id: Annotated[
        ULID,
        RefSpec("evidence", ref_mode="historical_locator", scope_rule="same_scope", gc_root=False),
    ]
    reason_code: Literal["privacy_erase"]
    phase: Literal["prepared", "applying", "verifying"]
    plan_digest: Integrity
    cursor: NonNegativeInt
    affected_counts: PrivacyCountsV1
    erase_epoch_bumps: NonNegativeInt
    last_completed_digest: Integrity | None = None
    credential_records_deleted: NonNegativeInt = Field(
        default=0, exclude_if=lambda value: value == 0,
    )
    """The records holding a recognized credential that committed transactions deleted
    after the last batch (`CredentialFindingsV1.records_deleted`)."""

    @model_validator(mode="after")
    def check_progress(self) -> Self:
        """The cursor advances only with an explicit completed-batch digest."""
        if (self.cursor == 0) != (self.last_completed_digest is None):
            raise ValueError("Privacy cursor and completed batch proof disagree")
        if (self.cursor == 0) != (self.erase_epoch_bumps == 0) or (
            self.erase_epoch_bumps > self.cursor
        ):
            raise ValueError("The first batch and only completed batches advance erase_epoch")
        if self.phase == "prepared" and (
            self.cursor != 0 or any(self.affected_counts.model_dump().values())
        ):
            raise ValueError("Prepared privacy progress must have no applied effects")
        if (self.phase == "verifying" and self.cursor == 0) or (
            self.cursor > 0 and not any(self.affected_counts.model_dump().values())
        ):
            raise ValueError("Completed privacy batches require explicit affected counts")
        return self


def _source(kind: str = "evidence") -> RefSpec:
    return RefSpec(
        kind,
        ref_mode="historical_locator",
        existence_expectation="must_exist",
        fulfillment="pre_state",
        scope_rule="same_scope",
        gc_root=False,
        target_forget_action="retain_locator",
    )


class EvidenceLocatorV1(StrictModel):
    """Locate a pre-existing Evidence row without retaining any source bytes."""

    entity_kind: Literal["evidence"]
    object_ref: Annotated[ULID, _source()]


class TombstoneSourceMetadataV1(StrictModel):
    """Retain only source coordinates and eligibility fields destroyed by a tombstone.

    The containing entry's keyed before-state digest binds these fields to the
    original row. This descriptor is not an Evidence row or an independent
    authority to restore source content.
    """

    byte_start: NonNegativeInt
    byte_end: NonNegativeInt
    state: Literal["active", "invalidated", "expired"]
    route_state: Literal["pending", "indexed"]

    @model_validator(mode="after")
    def check_interval(self) -> Self:
        """Empty original messages are valid, while reversed byte intervals are not."""
        if self.byte_end < self.byte_start:
            raise ValueError("Original source byte interval must be ordered")
        return self


def freeze_tombstone_source_metadata(source: EvidenceRowV1) -> TombstoneSourceMetadataV1:
    """Project exact metadata from a complete validated, not-yet-erased source row."""
    from quotemem.store.domain import EvidenceRowV1

    try:
        checked = EvidenceRowV1.model_validate(source.model_dump(mode="json"))
        return TombstoneSourceMetadataV1.model_validate(
            {
                "byte_start": checked.byte_start,
                "byte_end": checked.byte_end,
                "state": checked.state,
                "route_state": checked.route_state,
            }
        )
    except ValidationError:
        raise QuoteMemError(
            "schema_invalid", "Tombstone metadata requires a valid original Evidence row"
        ) from None


class IngestionErasureTransformV1(StrictModel):
    """The closed Evidence tombstone transform; no scripts and no post-images."""

    transform_kind: Literal["tombstone_evidence"]
    effect_time: Timestamp
    source_locators: Annotated[list[EvidenceLocatorV1], Field(min_length=1, max_length=4096)]
    source_digests: Annotated[list[RowDigest], Field(min_length=1, max_length=4096)]
    drop_locators: Annotated[list[EvidenceLocatorV1], Field(max_length=4096)]
    closed_patch: TombstoneSourceMetadataV1

    @model_validator(mode="after")
    def check_sources(self) -> Self:
        """Keep every frozen source digest aligned with one unique ordered locator."""
        keys = [source.object_ref for source in self.source_locators]
        drops = [source.object_ref for source in self.drop_locators]
        if len(keys) != len(self.source_digests) or keys != sorted(set(keys)):
            raise ValueError("Privacy source locators and digests must be aligned and unique")
        if drops != sorted(set(drops)):
            raise ValueError("Privacy drop locators must be sorted and unique")
        return self


def verify_tombstone_source_metadata(
    transform: IngestionErasureTransformV1, source: EvidenceRowV1
) -> None:
    """Check the frozen source witness against its actual original row before mutation."""
    if transform.closed_patch != freeze_tombstone_source_metadata(source):
        raise QuoteMemError("parity_mismatch",
                            "Tombstone source metadata differs from its original row")


class IngestionEvidenceTombstoneV1(StrictModel):
    """Bind the existing Evidence identity and its fixed tombstone transform."""

    entity_kind: Literal["evidence"]
    object_ref: Annotated[ULID, _source()]
    action: Literal["tombstone"]
    expected_before_digest: RowDigest
    transform: IngestionErasureTransformV1
    intended_after_digest: RowDigest

    @model_validator(mode="after")
    def check_transform(self) -> Self:
        """The target is the transform's one frozen source and part of the dropped set."""
        if (
            [source.object_ref for source in self.transform.source_locators] != [self.object_ref]
            or self.object_ref not in {source.object_ref for source in self.transform.drop_locators}
        ):
            raise ValueError("Evidence tombstone requires its exact target source and transform")
        if self.transform.source_digests != [self.expected_before_digest]:
            raise ValueError("Privacy target source and expected-before digests disagree")
        if self.expected_before_digest == self.intended_after_digest:
            raise ValueError("An Evidence tombstone must change its row")
        return self


class IngestionCandidateExpiryV1(StrictModel):
    """Expire one pending candidate that cites erased Evidence.

    ``erased_evidence_ids`` are the admitted targets the candidate cites; the
    batch that erases the first of them expires it. Its post-image is the
    frozen candidate with status ``expired``, reason ``evidence_invalidated``,
    no hold, and the committing batch's time and ``commit_seq``, which exist only
    when the batch commits: the entry keeps the keyed pre-state alone, and an
    expired candidate is terminal, which proves the effect afterwards.
    """

    entity_kind: Literal["candidate"]
    object_ref: Annotated[ULID, _source("candidate")]
    action: Literal["expire"]
    expected_before_digest: RowDigest
    erased_evidence_ids: Annotated[
        list[Annotated[ULID, _source()]], Field(min_length=1, max_length=512),
    ]

    @model_validator(mode="after")
    def check_erased(self) -> Self:
        """The erased Evidence that expires the candidate is one sorted, unique set."""
        if self.erased_evidence_ids != sorted(set(self.erased_evidence_ids)):
            raise ValueError("A candidate expiry names sorted, unique erased Evidence")
        return self


class IngestionTombstoneDeleteV1(StrictModel):
    """Delete one tombstone whose sources are all erased once its batch commits.

    ``erased_evidence_ids`` are its sources among the admitted targets; every
    other source was erased before. The batch that erases the last of them
    deletes it and advances ``tombstone_seq`` with ``erase_epoch``, and the
    absent row is the after-state.
    """

    entity_kind: Literal["tombstone"]
    object_ref: Annotated[ULID, _source("tombstone")]
    action: Literal["delete"]
    expected_before_digest: RowDigest
    erased_evidence_ids: Annotated[
        list[Annotated[ULID, _source()]], Field(min_length=1, max_length=MAX_TOMBSTONE_SOURCES),
    ]

    @model_validator(mode="after")
    def check_erased(self) -> Self:
        """The tombstone's erased sources are one sorted, unique set."""
        if self.erased_evidence_ids != sorted(set(self.erased_evidence_ids)):
            raise ValueError("A tombstone deletion names sorted, unique erased sources")
        return self


IngestionErasureEntryV1 = Annotated[
    IngestionEvidenceTombstoneV1 | IngestionCandidateExpiryV1 | IngestionTombstoneDeleteV1,
    Field(discriminator="entity_kind"),
]
"""One planned effect: an Evidence placeholder, or a dependent row the erase changes or deletes."""
ERASURE_ENTRY: TypeAdapter[IngestionErasureEntryV1] = TypeAdapter(IngestionErasureEntryV1)


def batch_bumps(batch_no: int, entries: Iterable[IngestionErasureEntryV1]) -> frozenset[str]:
    """The scope counters one erase batch advances besides ``commit_seq`` and ``scope_seq``.

    The first batch advances ``erase_epoch`` (+1 per privacy erase), so an
    owner pinned before the erase meets a target change at its commit fence; a
    batch that deletes a tombstone advances ``tombstone_seq`` with ``erase_epoch``.
    """
    if any(isinstance(entry, IngestionTombstoneDeleteV1) for entry in entries):
        return frozenset({"tombstone_seq", "erase_epoch"})
    return frozenset({"erase_epoch"}) if batch_no == 0 else frozenset()


class IngestionErasureBatchV1(StrictModel):
    """Declare the exact logical batch size independently of physical blocks."""

    record_kind: Literal["batch"]
    batch_no: NonNegativeInt
    phase: Literal["compensate"]
    entry_count: Annotated[int, Field(ge=1, le=2000)]


class IngestionErasureEntryRecordV1(StrictModel):
    """Place one content-free effect in the complete ordered batch stream."""

    record_kind: Literal["entry"]
    batch_no: NonNegativeInt
    entry_no: NonNegativeInt
    entry: IngestionErasureEntryV1


IngestionErasureRecordV1 = Annotated[
    IngestionErasureBatchV1 | IngestionErasureEntryRecordV1, Field(discriminator="record_kind")
]


class IngestionErasureTargetV1(StrictModel):
    """Bind the admitted Evidence set without granting whole-scope erasure authority."""

    kind: Literal["evidence_set"]
    evidence_digest: RowDigest


class IngestionErasurePlanV1(StrictModel):
    """Persist one bounded chain block; the consumer proves the complete logical stream."""

    controller_digest: RowDigest
    admitted_erase_epoch: NonNegativeInt
    effect_time: Timestamp
    target: IngestionErasureTargetV1
    previous_ref: (
        Annotated[
            ArtifactId,
            RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("erasure_plan",)),
        ]
        | None
    ) = None
    block_no: NonNegativeInt
    records: Annotated[list[IngestionErasureRecordV1], Field(min_length=1, max_length=512)]
    record_count: PositiveInt
    total_record_count: PositiveInt
    cumulative_digest: Integrity
    is_head: bool

    @model_validator(mode="after")
    def check_block(self) -> Self:
        """Check local shape without pretending it proves predecessor or batch completeness."""
        if (self.previous_ref is None) != (self.block_no == 0):
            raise ValueError("Privacy predecessor must match its block position")
        if self.record_count != len(self.records) or self.total_record_count < self.record_count:
            raise ValueError("Privacy record counts do not match this block")
        if self.block_no == 0 and self.total_record_count != self.record_count:
            raise ValueError("Initial privacy record count has no predecessor")
        order = [
            (
                record.batch_no,
                -1 if isinstance(record, IngestionErasureBatchV1) else record.entry_no,
            )
            for record in self.records
        ]
        if order != sorted(set(order)):
            raise ValueError("Privacy records must be strictly ordered")
        for record in self.records:
            if (
                isinstance(record, IngestionErasureEntryRecordV1)
                and isinstance(record.entry, IngestionEvidenceTombstoneV1)
                and record.entry.transform.effect_time != self.effect_time
            ):
                raise ValueError("Privacy transform time differs from its fixed header")
        return self


class RedactionReceiptV1(StrictModel):
    """The verified completion of one privacy erase, without text or object locators.

    ``affected_digests`` are the sorted, unique ``expected_row_digest`` of every
    admitted target. ``completion_commit_seq`` is the visibility
    sequence of the transaction that selected this receipt. A schema-valid or
    prewritten receipt supplies no ordering authority: the selected succeeded
    owner, its complete plan and the final receipt selection must independently
    prove the same completion transaction. ``credential_findings``, absent when
    there are none, declares the recognized credentials found beyond the targets.
    """

    receipt_kind: Literal["redaction"]
    reason_code: Literal["retroactive_split_secret", "source_expired", "consent_withdrawn"]
    status: Literal["verified"]
    affected_digests: Annotated[list[RowDigest], Field(min_length=1, max_length=512)]
    counts: PrivacyCountsV1
    not_covered: list[NotCovered]
    completion_commit_seq: PositiveInt
    credential_findings: CredentialFindingsV1 = Field(
        default_factory=CredentialFindingsV1, exclude_if=_no_findings,
    )

    @model_validator(mode="after")
    def check_canonical_sets(self) -> Self:
        """Reject duplicate proof identities or omission of known external-provider copies."""
        if self.affected_digests != sorted(set(self.affected_digests)):
            raise ValueError("Redaction affected digests must be sorted and unique")
        if (
            self.not_covered != sorted(set(self.not_covered))
            or "provider_copies" not in self.not_covered
        ):
            raise ValueError("Redaction coverage must be canonical and disclose provider copies")
        return self


REDACTION_REASONS = frozenset(get_args(RedactionReceiptV1.model_fields["reason_code"].annotation))
"""The reasons of the implemented redaction producers."""
NOT_COVERED: list[NotCovered] = sorted(get_args(NotCovered))
"""A receipt's complete canonical ``not_covered`` list."""

_RECORD: TypeAdapter[IngestionErasureRecordV1] = TypeAdapter(IngestionErasureRecordV1)
_NONNEGATIVE: TypeAdapter[int] = TypeAdapter(NonNegativeInt)
_TIMESTAMP: TypeAdapter[int] = TypeAdapter(Timestamp)


def entry_postcondition(entry: IngestionErasureEntryV1) -> str:
    """The state one entry leaves its row in: an exact after-digest, or its terminal action.

    A dependent row's post-image carries the committing batch's ``commit_seq``,
    so it is named by the action applied to its exact keyed pre-state.
    """
    if isinstance(entry, IngestionEvidenceTombstoneV1):
        return entry.intended_after_digest
    return f"{entry.action}:{entry.expected_before_digest}"


def ingestion_batch_postcondition_digest(
    batch_no: int, entries: Iterable[IngestionErasureEntryV1]
) -> IntegrityDigest:
    """Bind the complete sorted intended row states to one logical maintenance cursor step."""
    number = _NONNEGATIVE.validate_python(batch_no, strict=True)
    conditions: list[dict[str, str]] = []
    for supplied in entries:
        entry = ERASURE_ENTRY.validate_python(supplied.model_dump(mode="json"))
        conditions.append(
            {
                "entity_kind": entry.entity_kind,
                "object_identity": entry.object_ref,
                "postcondition": entry_postcondition(entry),
            }
        )
    conditions.sort(key=lambda condition: (condition["entity_kind"], condition["object_identity"]))
    identities = [
        (condition["entity_kind"], condition["object_identity"]) for condition in conditions
    ]
    if not conditions or len(conditions) > 2000 or len(set(identities)) != len(identities):
        raise QuoteMemError(
            "parity_mismatch", "Privacy batch postconditions require exact unique coverage"
        )
    return integrity_digest({"batch_no": number, "postconditions": conditions})


class IngestionErasureStream:
    """Validate and hash one canonical logical stream without retaining full plan records."""

    def __init__(
        self,
        *,
        controller_digest: ContentBindingDigest,
        admitted_erase_epoch: int,
        effect_time: int,
        target: IngestionErasureTargetV1,
    ) -> None:
        """Freeze the header while keeping physical block identities out of the plan digest."""
        self._header = {
            "admitted_erase_epoch": _NONNEGATIVE.validate_python(admitted_erase_epoch, strict=True),
            "controller_digest": ContentBindingDigest(controller_digest),
            "effect_time": _TIMESTAMP.validate_python(effect_time, strict=True),
        }
        self._target = IngestionErasureTargetV1.model_validate(target.model_dump(mode="json"))
        self._records = hashlib.sha256(b"[")
        prefix = canonical_json(self._header)[:-1] + ',"ordered_batches":['
        self._plan = hashlib.sha256(prefix.encode("utf-8"))
        self._record_count = 0
        self._batch_count = 0
        self._entry_count = 0
        self._remaining = 0
        self._batch_entries = 0
        self._phase: str | None = None
        self._seen: set[tuple[str, str]] = set()
        self._finished: IntegrityDigest | None = None

    def copy(self) -> IngestionErasureStream:
        """Fork incremental hashes for bounded candidate-block sizing without replaying records."""
        cloned = copy(self)
        cloned._records = self._records.copy()
        cloned._plan = self._plan.copy()
        cloned._seen = self._seen.copy()
        return cloned

    @property
    def record_count(self) -> int:
        """Return the exact record-prefix count for a physical block header."""
        return self._record_count

    @property
    def batch_count(self) -> int:
        """Return the number of encountered logical effect batch headers."""
        return self._batch_count

    @property
    def entry_count(self) -> int:
        """Return the number of complete logical effects in the consumed stream."""
        return self._entry_count

    @property
    def prefix_digest(self) -> IntegrityDigest:
        """Hash the canonical record array prefix without closing the incremental stream."""
        snapshot = self._records.copy()
        snapshot.update(b"]")
        return IntegrityDigest(snapshot.hexdigest())

    def _close_batch(self) -> None:
        if self._phase is not None:
            if self._remaining:
                raise QuoteMemError("parity_mismatch", "Privacy batch is missing declared entries")
            self._plan.update(('],"phase":' + canonical_json(self._phase) + "}").encode())

    def append(self, record: StrictModel) -> None:
        """Consume one validated record, enforcing global contiguous batch/entry coordinates."""
        if self._finished is not None:
            raise QuoteMemError("parity_mismatch", "A completed privacy stream cannot be extended")
        validated = _RECORD.validate_python(record.model_dump(mode="json"))
        if isinstance(validated, IngestionErasureBatchV1):
            self._close_batch()
            if validated.batch_no != self._batch_count:
                raise QuoteMemError("parity_mismatch", "Privacy batch numbers are not contiguous")
            if self._batch_count:
                self._plan.update(b",")
            self._plan.update(('{"batch_no":' + str(validated.batch_no) + ',"entries":[').encode())
            self._batch_count += 1
            self._remaining = validated.entry_count
            self._batch_entries = 0
            self._phase = validated.phase
        else:
            if (
                validated.batch_no != self._batch_count - 1
                or validated.entry_no != self._batch_entries
                or self._remaining < 1
            ):
                raise QuoteMemError(
                    "parity_mismatch",
                    "Privacy entry coverage is missing, duplicated, or outside its batch",
                )
            entry = validated.entry
            if isinstance(entry, IngestionEvidenceTombstoneV1) and (
                entry.transform.effect_time != self._header["effect_time"]
            ):
                raise QuoteMemError("parity_mismatch",
                                    "Privacy entry changed its frozen effect time")
            key = (entry.entity_kind, entry.object_ref)
            if key in self._seen:
                raise QuoteMemError("parity_mismatch", "Privacy stream repeats a physical target")
            self._seen.add(key)
            if self._batch_entries:
                self._plan.update(b",")
            self._plan.update(canonical_json(validated.entry).encode("utf-8"))
            self._remaining -= 1
            self._batch_entries += 1
            self._entry_count += 1
        if self._record_count:
            self._records.update(b",")
        self._records.update(canonical_json(validated).encode("utf-8"))
        self._record_count += 1

    def finish(self) -> IntegrityDigest:
        """Require complete logical coverage and return the physical-layout-independent digest."""
        if self._finished is None:
            if self._batch_count == 0:
                raise QuoteMemError(
                    "parity_mismatch", "Ingestion erasure requires a nonempty effect stream"
                )
            self._close_batch()
            self._plan.update(('],"target":' + canonical_json(self._target) + "}").encode())
            self._finished = IntegrityDigest(self._plan.hexdigest())
        return self._finished


@dataclass(frozen=True)
class IngestionErasureSummary:
    """A validated chain's ephemeral proof; no persisted alternative progress authority."""

    plan_digest: IntegrityDigest
    head_ref: ArtifactId
    record_count: int
    batch_count: int
    entry_count: int


def validate_ingestion_erasure_chain(
    blocks: Iterable[tuple[ArtifactId, IngestionErasurePlanV1]],
) -> IngestionErasureSummary:
    """Verify a streamed chain of already resolved direct/manifest logical blocks.

    Artifact addresses and manifest integrity remain the ArtifactStore's duty;
    this function proves exact predecessor, header, ordering, prefix, and head
    parity without assuming that block boundaries are effect boundaries.
    """
    stream: IngestionErasureStream | None = None
    fixed: tuple[ContentBindingDigest, int, int, IngestionErasureTargetV1] | None = None
    previous: ArtifactId | None = None
    found_head = False
    count = 0
    for reference, supplied in blocks:
        block = IngestionErasurePlanV1.model_validate(supplied.model_dump(mode="json"))
        header = (
            block.controller_digest,
            block.admitted_erase_epoch,
            block.effect_time,
            block.target,
        )
        if found_head or block.block_no != count or block.previous_ref != previous:
            raise QuoteMemError("parity_mismatch",
                                "Privacy chain predecessor or head is inconsistent")
        if fixed is None:
            fixed = header
            stream = IngestionErasureStream(
                controller_digest=header[0],
                admitted_erase_epoch=header[1],
                effect_time=header[2],
                target=header[3],
            )
        elif header != fixed:
            raise QuoteMemError("parity_mismatch", "Privacy chain changed its fixed header")
        assert stream is not None
        for record in block.records:
            stream.append(record)
        if (
            block.total_record_count != stream.record_count
            or block.cumulative_digest != stream.prefix_digest
        ):
            raise QuoteMemError(
                "parity_mismatch", "Privacy chain record prefix or cumulative digest differs"
            )
        previous = ArtifactId(reference)
        found_head = block.is_head
        count += 1
    if stream is None or previous is None or not found_head:
        raise QuoteMemError("parity_mismatch", "Privacy chain has no complete selected head")
    return IngestionErasureSummary(
        stream.finish(), previous, stream.record_count, stream.batch_count, stream.entry_count
    )
