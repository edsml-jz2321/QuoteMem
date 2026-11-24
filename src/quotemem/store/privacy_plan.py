"""Read-only privacy-erase planning and reconstruction.

Persisted records contain locators, keyed digests and closed transformations.
Complete source rows and post-images exist only while discovering or rebuilding
one bounded atomic group. The sole Writer owns admission, fences and mutation.

A plan erases the admitted Evidence and, in the same batches, the effects that
follow from it today: every pending candidate citing an erased source expires
and every tombstone whose sources are all erased is deleted.
Each batch groups Evidence with its dependents: a candidate goes with the
first of its erased sources, a tombstone with the last.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pydantic import TypeAdapter, ValidationError

from quotemem.core.admission import (
    admission_identity,
    privacy_erase_admission,
    split_secret_parent_binding,
)
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import (
    ArtifactId,
    ContentBindingDigest,
    IntegrityDigest,
    canonical_json,
    control_id,
)
from quotemem.core.ingestion_cleanup import CleanupInput, cleanup_target, validate_cleanup_input
from quotemem.core.privacy import (
    EvidenceTombstoneActionV1,
    action_key,
    expire_candidate,
    tombstone_evidence,
)
from quotemem.core.privacy_persistence import (
    ERASURE_ENTRY,
    ZERO_COUNTS,
    AffectedEvidenceV1,
    IngestionCandidateExpiryV1,
    IngestionErasureBatchV1,
    IngestionErasureEntryRecordV1,
    IngestionErasureEntryV1,
    IngestionErasurePlanV1,
    IngestionErasureRecordV1,
    IngestionErasureStream,
    IngestionErasureTargetV1,
    IngestionErasureTransformV1,
    IngestionEvidenceTombstoneV1,
    IngestionSplitSecretInputV1,
    IngestionTombstoneDeleteV1,
    PrivacyCountsV1,
    PrivacyEraseProgressV1,
    batch_bumps,
    freeze_tombstone_source_metadata,
    ingestion_batch_postcondition_digest,
    validate_ingestion_erasure_chain,
    verify_tombstone_source_metadata,
)
from quotemem.core.query import ScopeV1
from quotemem.core.semantic import ScopeKind
from quotemem.store.domain import DOMAIN_ADAPTERS, EvidenceRowV1
from quotemem.store.indexed import MAX_PAGE_SIZE, CandidateRows, IndexedQuery, TombstoneRows
from quotemem.store.writer import MAX_TX_ROWS, RowDelete, RowWrite, Writer

_RECORD: TypeAdapter[IngestionErasureRecordV1] = TypeAdapter(IngestionErasureRecordV1)
_RowKey = tuple[str, str]
_RowOps = Callable[[Sequence[RowWrite], Sequence[RowDelete], frozenset[str]], int]
"""Measure a batch's main RowOps: its row writes, its deletes and its counter bumps."""
_KIND_ORDER = {"evidence": 0, "candidate": 1, "tombstone": 2}
"""A batch lists its Evidence placeholders, then candidate expiries, then tombstone deletions."""


@dataclass(frozen=True)
class RowPrecondition:
    """An immutable keyed full-row proof; None means that the identity must be absent."""

    entity_kind: str
    object_id: str
    digest: ContentBindingDigest | None


@dataclass(frozen=True)
class PrivacyBatch:
    """Freeze canonical entry bytes without exposing mutable typed proof objects."""

    batch_no: int
    entry_json: tuple[str, ...]

    @property
    def entries(self) -> tuple[IngestionErasureEntryV1, ...]:
        """Return independent strict values; mutating one cannot change this batch."""
        return tuple(ERASURE_ENTRY.validate_json(value) for value in self.entry_json)

    @property
    def bumps(self) -> frozenset[str]:
        """The scope counters the batch advances besides ``commit_seq`` and ``scope_seq``."""
        return batch_bumps(self.batch_no, self.entries)


@dataclass(frozen=True)
class PreparedPrivacyEffects:
    """Content-free optimistic proof, requiring the Writer's generation recheck."""

    owner_run_id: str
    scope_kind: ScopeKind
    scope_id: str
    ref_generation: int
    preconditions: tuple[RowPrecondition, ...]
    controller_digest: ContentBindingDigest
    admitted_erase_epoch: int
    effect_time: int
    evidence_digest: ContentBindingDigest
    input_json: str
    batches: tuple[PrivacyBatch, ...]
    plan_digest: IntegrityDigest
    head_ref: ArtifactId | None = None

    def records(self) -> Iterator[IngestionErasureRecordV1]:
        """Yield the same complete logical stream for every physical block size."""
        for batch in self.batches:
            yield IngestionErasureBatchV1(
                record_kind="batch",
                batch_no=batch.batch_no,
                phase="compensate",
                entry_count=len(batch.entry_json),
            )
            for number, entry in enumerate(batch.entries):
                yield IngestionErasureEntryRecordV1.model_validate({
                    "record_kind": "entry", "batch_no": batch.batch_no,
                    "entry_no": number, "entry": entry.model_dump(mode="json"),
                })


@dataclass(frozen=True)
class ReconstructedPrivacyBatch:
    """Temporary full images with exact recheck witnesses, never a persisted plan.

    Attributes:
        batch_no: The batch.
        rows: Its Evidence placeholders' post-images; once applied, the stored
            placeholders.
        expiring: The stored pending candidates it expires; the Writer completes
            each post-image with the batch's commit (``expire_candidate``).
            Empty once applied.
        deleting: The tombstones it deletes. Empty once applied.
        preconditions: Every keyed row state the batch depends on.
        already_applied: Whether every entry is at its after-state.
    """

    batch_no: int
    rows: tuple[RowWrite, ...]
    preconditions: tuple[RowPrecondition, ...]
    already_applied: bool
    expiring: tuple[dict[str, Any], ...] = ()
    deleting: tuple[RowDelete, ...] = ()


def _digest(writer: Writer, row: object) -> ContentBindingDigest:
    return writer.bind_content("row-state", row)


def _key(entry: IngestionErasureEntryV1) -> _RowKey:
    return entry.entity_kind, entry.object_ref


def _order(entry: IngestionErasureEntryV1) -> tuple[int, str]:
    return _KIND_ORDER[entry.entity_kind], entry.object_ref


def _pages(writer: Writer, query: IndexedQuery) -> Iterator[dict[str, Any]]:
    """Every row of one bounded indexed query, page by page."""
    reader = writer.indexed_reader()
    after: str | None = None
    while True:
        page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
        yield from page
        if len(page) < MAX_PAGE_SIZE:
            return
        after = page[-1]["object_id"]


def _get(writer: Writer, kind: str, identity: str) -> dict[str, Any]:
    row = writer.get_row(kind, identity)
    if row is None:
        raise QuoteMemError("reference_missing", "A frozen privacy source is missing")
    if row["object_id"] != identity or row.get("tenant_id", writer.tenant_id) != writer.tenant_id:
        raise QuoteMemError("parity_mismatch", "Privacy source identity or tenant differs")
    return row


def _scope(row: Mapping[str, Any], scope: tuple[str, str]) -> None:
    if (row.get("scope_kind"), row.get("scope_id")) != scope:
        raise QuoteMemError("scope_denied", "Privacy source belongs to another exact scope")


def _remember(
    writer: Writer, proofs: dict[_RowKey, RowPrecondition], kind: str, row: dict[str, Any]
) -> None:
    key = (kind, row["object_id"])
    proof = RowPrecondition(*key, _digest(writer, row))
    if key in proofs and proofs[key] != proof:
        raise QuoteMemError("stale_evidence", "Privacy discovery observed a source changing")
    proofs[key] = proof


def _transform(
    writer: Writer,
    source: EvidenceRowV1,
    drops: frozenset[str],
    effect_time: int,
) -> IngestionErasureTransformV1:
    """Freeze one Evidence row's tombstone transform, its source witness and the drop set."""
    if source.state == "erased":
        raise QuoteMemError("parity_mismatch", "Split-secret target is already erased")
    try:
        return IngestionErasureTransformV1.model_validate(
            {
                "transform_kind": "tombstone_evidence",
                "effect_time": effect_time,
                "source_locators": [{"entity_kind": "evidence", "object_ref": source.object_id}],
                "source_digests": [_digest(writer, source.model_dump(mode="json"))],
                "drop_locators": [
                    {"entity_kind": "evidence", "object_ref": key} for key in sorted(drops)
                ],
                "closed_patch": freeze_tombstone_source_metadata(source).model_dump(mode="json"),
            }
        )
    except ValidationError as error:
        raise QuoteMemError(
            "artifact_too_large", "A minimum privacy transform exceeds its strict source budget"
        ) from error


def _entry(
    action: EvidenceTombstoneActionV1, transform: IngestionErasureTransformV1
) -> IngestionEvidenceTombstoneV1:
    kind, identity = action_key(action)
    return IngestionEvidenceTombstoneV1.model_validate({
        "entity_kind": kind,
        "transform": transform,
        "intended_after_digest": action.intended_after_digest,
        "object_ref": identity,
        "action": "tombstone",
        "expected_before_digest": action.expected_before_digest,
    })


def validate_cleanup_owner(
    writer: Writer,
    owner_run_id: str,
    supplied: CleanupInput,
) -> dict[str, Any]:
    """Verify the existing cleanup Run's complete canonical admission and ancestry.

    Args:
        writer: Sole storage and tenant-keyed binding owner; this call is read-only.
        owner_run_id: Exact durable cleanup identity, without admitting a new Run.
        supplied: Complete selected closed admission, never a partial target list.

    Returns:
        The current owner row after identity, scope, epoch and parent checks.

    Raises:
        QuoteMemError: If any canonical admission or selected ancestor binding differs.
            Validity alone does not authorize reading sources or writing a plan.
    """
    run = _get(writer, "run", owner_run_id)
    if (
        run["run_kind"] != "retention"
        or run["commit_mode"] != "destructive_scan"
        or run["target_kind"] != "affected_set"
        or run["scope_id"] is None
        or run["admitted_erase_epoch"] != supplied.erase_epoch
        or run["admitted_consent_epoch"] != supplied.consent_epoch
    ):
        raise QuoteMemError("binding_mismatch",
                            "Privacy planning requires its exact admitted owner")
    target = cleanup_target(supplied, writer.bind_content)
    admission = privacy_erase_admission(
        ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"]),
        supplied,
        run["principal_id"],
        writer.bind_content,
    )
    identity, digest, _ = admission_identity(admission, writer.bind_content)
    if (run["idempotency_key"], run["admission_digest"], run["target_id"]) != (
        identity,
        digest,
        target,
    ):
        raise QuoteMemError("binding_mismatch", "Privacy owner differs from canonical admission")
    if not isinstance(supplied, IngestionSplitSecretInputV1):
        if run["parent_run_id"] is not None:
            raise QuoteMemError("binding_mismatch", "A privacy-erase root must be independent")
    else:
        parent = _get(writer, "run", run["parent_run_id"])
        _scope(parent, (run["scope_kind"], run["scope_id"]))
        if (
            split_secret_parent_binding(parent, writer.bind_content)
            != supplied.parent_binding_digest
            or parent["run_kind"] != "session_ingestion"
        ):
            raise QuoteMemError("binding_mismatch",
                                "Privacy child lost its ingestion parent binding")
    return run


def _evidence(
    writer: Writer,
    identities: Sequence[str] | set[str],
    scope: tuple[str, str],
    proofs: dict[_RowKey, RowPrecondition],
) -> dict[str, EvidenceRowV1]:
    result: dict[str, EvidenceRowV1] = {}
    for identity in sorted(set(identities)):
        row = _get(writer, "evidence", identity)
        _scope(row, scope)
        _remember(writer, proofs, "evidence", row)
        result[identity] = EvidenceRowV1.model_validate(row)
    return result


def privacy_targets(proof: PreparedPrivacyEffects) -> tuple[AffectedEvidenceV1, ...]:
    """Read content-free original target bindings from the complete selected plan.

    This projection grants no write authority. The plan loader verifies its full
    count and keyed digest against the creating Run's admitted input before use.
    """
    result: dict[str, AffectedEvidenceV1] = {}
    for batch in proof.batches:
        for entry in batch.entries:
            if not isinstance(entry, IngestionEvidenceTombstoneV1):
                continue
            if entry.object_ref in result:
                raise QuoteMemError("parity_mismatch", "Privacy plan repeats an Evidence target")
            result[entry.object_ref] = AffectedEvidenceV1(
                evidence_id=entry.object_ref, expected_row_digest=entry.expected_before_digest
            )
    return tuple(result[key] for key in sorted(result))


@dataclass(frozen=True)
class _Dependent:
    """A row the erase changes because of the admitted targets it depends on."""

    kind: str
    row: dict[str, Any]
    erased: tuple[str, ...]

    @property
    def anchor(self) -> str:
        """The target whose batch carries it: a candidate's first erased source, a
        tombstone's last."""
        return self.erased[0] if self.kind == "candidate" else self.erased[-1]


def _pending_candidates(
    writer: Writer,
    scope: tuple[ScopeKind, str],
    admitted: frozenset[str],
    proofs: dict[_RowKey, RowPrecondition],
) -> list[_Dependent]:
    """Every pending candidate that cites an admitted target."""
    found: list[_Dependent] = []
    query = CandidateRows(scope_kind=scope[0], scope_id=scope[1], status="pending")
    for row in _pages(writer, query):
        _scope(row, scope)
        if row["status"] != "pending":
            raise QuoteMemError("parity_mismatch",
                                "A pending-candidate page returned another status")
        erased = tuple(sorted(set(row["evidence_ids"]) & admitted))
        if erased:
            _remember(writer, proofs, "candidate", row)
            found.append(_Dependent("candidate", row, erased))
    return found


def _exhausted_tombstones(
    writer: Writer,
    scope: tuple[ScopeKind, str],
    admitted: frozenset[str],
    proofs: dict[_RowKey, RowPrecondition],
) -> list[_Dependent]:
    """Every tombstone whose sources are all erased once the admitted targets are.

    A tombstone qualifies when one of its sources is admitted and every other
    source is already erased. A missing source fails closed.
    """
    found: list[_Dependent] = []
    query = TombstoneRows(scope_kind=scope[0], scope_id=scope[1])
    for row in _pages(writer, query):
        _scope(row, scope)
        sources = set(row["source_evidence_ids"])
        erased = tuple(sorted(sources & admitted))
        if not erased:
            continue
        others = [_get(writer, "evidence", identity) for identity in sorted(sources - admitted)]
        if any(other["state"] != "erased" for other in others):
            continue
        for other in others:
            _scope(other, scope)
            _remember(writer, proofs, "evidence", other)
        _remember(writer, proofs, "tombstone", row)
        found.append(_Dependent("tombstone", row, erased))
    return found


def _dependents(
    writer: Writer,
    scope: tuple[ScopeKind, str],
    admitted: frozenset[str],
    proofs: dict[_RowKey, RowPrecondition],
) -> list[_Dependent]:
    """The dependent rows an erase of the admitted targets changes today."""
    return [
        *_pending_candidates(writer, scope, admitted, proofs),
        *_exhausted_tombstones(writer, scope, admitted, proofs),
    ]


def prepare_privacy_effects(
    writer: Writer,
    supplied: CleanupInput,
    *,
    owner_run_id: str,
    effect_time: int,
    rowops: _RowOps,
    max_rowops: int = MAX_TX_ROWS,
) -> PreparedPrivacyEffects:
    """Discover exact current dependencies and pack complete groups using actual HB RowOps.

    The callback measures a batch's rows, deletes and counter bumps with every
    Run/progress/counter row of its overhead. This function never calls
    providers or persistence. The returned generation and all row preconditions
    require a locked recheck before selecting the plan.
    """
    supplied = validate_cleanup_input(supplied.model_dump(mode="json"))
    affected_sources = tuple(supplied.affected)
    if type(effect_time) is not int or not 0 <= effect_time <= 253402300799000:
        raise QuoteMemError("invalid_request", "Privacy effect time is outside the storage range")
    if type(max_rowops) is not int or not 1 <= max_rowops <= MAX_TX_ROWS:
        raise QuoteMemError("invalid_request", "Invalid privacy physical transaction budget")
    generation = writer.ref_generation
    run = validate_cleanup_owner(writer, owner_run_id, supplied)
    if run["input_ref"] is not None or run["status"] != "queued":
        raise QuoteMemError("binding_mismatch",
                            "New privacy discovery requires its empty queued owner")
    scope = (run["scope_kind"], run["scope_id"])
    proofs: dict[_RowKey, RowPrecondition] = {}
    _remember(writer, proofs, "run", run)
    for name, expected in (
        ("scope_seq", supplied.expected_scope_seq),
        ("erase_epoch", supplied.erase_epoch),
        ("consent_epoch", supplied.consent_epoch),
    ):
        control = _get(writer, "control", control_id(f"{name}:{scope[1]}"))
        if control["value"]["seq"] != expected:
            raise QuoteMemError("stale_evidence",
                                "Privacy admission counters changed before discovery")
        _remember(writer, proofs, "control", control)
    drops = frozenset(entry.evidence_id for entry in affected_sources)
    targets = _evidence(writer, set(drops), scope, proofs)
    for affected in affected_sources:
        if (
            _digest(writer, targets[affected.evidence_id].model_dump(mode="json"))
            != affected.expected_row_digest
        ):
            raise QuoteMemError("stale_evidence", "An admitted privacy target changed")
        if targets[affected.evidence_id].state == "erased":
            raise QuoteMemError(
                "parity_mismatch", "A new privacy plan cannot target an existing tombstone"
            )
    batches = _pack_batches(
        writer, targets=targets, dependents=_dependents(writer, scope, drops, proofs),
        effect_time=effect_time, rowops=rowops, max_rowops=max_rowops,
    )
    controller = ContentBindingDigest(run["admission_digest"])
    target_digest = cleanup_target(supplied, writer.bind_content)
    stream = IngestionErasureStream(
        controller_digest=controller,
        admitted_erase_epoch=supplied.erase_epoch,
        effect_time=effect_time,
        target=IngestionErasureTargetV1(kind="evidence_set", evidence_digest=target_digest),
    )
    provisional = PreparedPrivacyEffects(
        owner_run_id,
        scope[0],
        scope[1],
        generation,
        tuple(proofs[key] for key in sorted(proofs)),
        controller,
        supplied.erase_epoch,
        effect_time,
        target_digest,
        canonical_json(supplied),
        tuple(batches),
        IntegrityDigest("0" * 64),
    )
    for record in provisional.records():
        stream.append(record)
    return replace(provisional, plan_digest=stream.finish())


_Effect = tuple[IngestionErasureEntryV1, RowWrite | None, RowDelete | None]
"""One planned entry with the row write or delete that measures it."""


def _dependent_effect(
    writer: Writer, dependent: _Dependent, *, sequence: int, effect_time: int,
) -> _Effect:
    """Freeze one dependent row's entry and the temporary image that measures it."""
    identity = dependent.row["object_id"]
    common = {
        "object_ref": identity,
        "expected_before_digest": _digest(writer, dependent.row),
        "erased_evidence_ids": list(dependent.erased),
    }
    if dependent.kind == "candidate":
        image = expire_candidate(
            dependent.row, commit_seq=sequence,
            now=max(effect_time, dependent.row["created_at"]),
        )
        return (
            IngestionCandidateExpiryV1.model_validate(
                {"entity_kind": "candidate", "action": "expire", **common}
            ),
            RowWrite("candidate", DOMAIN_ADAPTERS["candidate"].validate_python(image)),
            None,
        )
    return (
        IngestionTombstoneDeleteV1.model_validate(
            {"entity_kind": "tombstone", "action": "delete", **common}
        ),
        None,
        RowDelete("tombstone", identity),
    )


def _pack_batches(
    writer: Writer,
    *,
    targets: Mapping[str, EvidenceRowV1],
    dependents: Sequence[_Dependent],
    effect_time: int,
    rowops: _RowOps,
    max_rowops: int,
) -> list[PrivacyBatch]:
    """Pack each admitted Evidence target with its dependents into bounded atomic batches.

    A candidate goes with the first of its erased sources, so the batch that
    first makes one of its citations unusable expires it; a tombstone goes with
    the last, so the batch that erases its last source deletes it.
    Groups are never split; a batch's entries are canonically ordered. This
    function is read-only and its complete source preconditions need a
    Writer-held recheck.
    """
    drops = frozenset(targets)
    attached: dict[str, list[_Dependent]] = {identity: [] for identity in targets}
    for dependent in dependents:
        attached[dependent.anchor].append(dependent)
    sequence = writer.get_counter("commit_seq") + 1
    batches: list[PrivacyBatch] = []
    pending: list[_Effect] = []

    def measure(batch_no: int, effects: Sequence[_Effect]) -> int:
        return rowops(
            [row for _, row, _ in effects if row is not None],
            [delete for _, _, delete in effects if delete is not None],
            batch_bumps(batch_no, [entry for entry, _, _ in effects]),
        )

    def flush() -> None:
        entries = sorted((entry for entry, _, _ in pending), key=_order)
        batches.append(PrivacyBatch(len(batches), tuple(canonical_json(e) for e in entries)))
        pending.clear()

    for identity in sorted(targets):
        row = targets[identity]
        action = tombstone_evidence(row, lambda value: _digest(writer, value))
        group: list[_Effect] = [(
            _entry(action, _transform(writer, row, drops, effect_time)),
            RowWrite(action_key(action)[0], action.post_image),
            None,
        )]
        group += [
            _dependent_effect(writer, dependent, sequence=sequence, effect_time=effect_time)
            for dependent in sorted(attached[identity], key=lambda value: value.row["object_id"])
        ]
        if pending and measure(len(batches), [*pending, *group]) <= max_rowops:
            pending.extend(group)
            continue
        if pending:
            flush()
        if measure(len(batches), group) > max_rowops:
            raise QuoteMemError(
                "invalid_request", "An indivisible privacy group exceeds the physical row budget"
            )
        pending.extend(group)
    flush()
    return batches


def _stream(proof: PreparedPrivacyEffects) -> IngestionErasureStream:
    return IngestionErasureStream(
        controller_digest=proof.controller_digest,
        admitted_erase_epoch=proof.admitted_erase_epoch,
        effect_time=proof.effect_time,
        target=IngestionErasureTargetV1(kind="evidence_set", evidence_digest=proof.evidence_digest),
    )


class PlanBlockBuilder:
    """Construct bounded physical blocks; only the caller's Writer stores them.

    ``fits`` must prove actual Artifact envelope/reference/manifest StoragePlan
    capacity. Every candidate block, including one minimum record, passes that
    callback. Its result is a capacity verdict, never permission to truncate.
    """

    def __init__(
        self,
        proof: PreparedPrivacyEffects,
        *,
        fits: Callable[[IngestionErasurePlanV1], bool],
        max_records: int = 512,
    ) -> None:
        if type(max_records) is not int or not 1 <= max_records <= 512:
            raise QuoteMemError("invalid_request", "Privacy blocks require 1 through 512 records")
        self._proof = proof
        self._records = tuple(canonical_json(record) for record in proof.records())
        self._position = 0
        self._block_no: int = 0
        self._max_records = max_records
        self._fits = fits
        self._stream = _stream(proof)
        validation = _stream(proof)
        for record in proof.records():
            validation.append(record)
        if validation.finish() != proof.plan_digest:
            raise QuoteMemError("parity_mismatch", "Prepared privacy stream digest is inconsistent")

    @property
    def complete(self) -> bool:
        """Report whether all complete logical records have been emitted."""
        return self._position == len(self._records)

    def next_block(self, previous_ref: ArtifactId | None) -> IngestionErasurePlanV1:
        """Consume a nonempty bounded record prefix after the caller persists its predecessor."""
        if self.complete or (previous_ref is None) != (self._block_no == 0):
            raise QuoteMemError("parity_mismatch",
                                "Privacy block predecessor or exhaustion differs")
        available = [
            _RECORD.validate_json(value)
            for value in self._records[self._position : self._position + self._max_records]
        ]

        def candidate(count: int) -> IngestionErasurePlanV1:
            trial = self._stream.copy()
            for record in available[:count]:
                trial.append(record)
            return IngestionErasurePlanV1.model_validate(dict(
                controller_digest=self._proof.controller_digest,
                admitted_erase_epoch=self._proof.admitted_erase_epoch,
                effect_time=self._proof.effect_time,
                target=IngestionErasureTargetV1(
                    kind="evidence_set", evidence_digest=self._proof.evidence_digest
                ),
                previous_ref=previous_ref,
                block_no=self._block_no,
                records=[row.model_dump(mode="json") for row in available[:count]],
                record_count=count,
                total_record_count=trial.record_count,
                cumulative_digest=trial.prefix_digest,
                is_head=self._position + count == len(self._records),
            ))

        accepted = candidate(1)
        if not self._fits(accepted):
            raise QuoteMemError(
                "invalid_request", "A minimum privacy record exceeds physical Artifact capacity"
            )
        lower, upper = 2, len(available)
        # Added records cannot reduce body/ref/RowOp capacity. Binary search
        # bounds actual StoragePlan probes independently of the 512 record cap.
        while lower <= upper:
            count = (lower + upper) // 2
            proposed = candidate(count)
            if self._fits(proposed):
                accepted = proposed
                lower = count + 1
            else:
                upper = count - 1
        for record in accepted.records:
            self._stream.append(record)
        self._position += len(accepted.records)
        self._block_no += 1
        if self.complete and self._stream.finish() != self._proof.plan_digest:
            raise QuoteMemError(
                "parity_mismatch", "Privacy block emission changed the complete logical stream"
            )
        return accepted


def load_privacy_plan(
    writer: Writer,
    head_ref: ArtifactId,
    *,
    owner_run_id: str,
    supplied: CleanupInput,
    expected_plan_digest: IntegrityDigest | None,
    input_ref: ArtifactId | None = None,
    allow_unattached_input: bool = False,
) -> PreparedPrivacyEffects:
    """Prove the selected input, complete chain and logical digest before returning any batch.

    Before initial atomic attachment, the caller may explicitly supply an already
    persisted immutable input. Ordinary recovery requires the exact attached input.
    The proof contains only canonical record bytes and cannot serve as a write ticket.
    """
    supplied = validate_cleanup_input(supplied.model_dump(mode="json"))
    generation = writer.ref_generation
    run = validate_cleanup_owner(writer, owner_run_id, supplied)
    if expected_plan_digest is None and run["status"] != "succeeded":
        raise QuoteMemError("binding_mismatch", "An active cleanup requires its frozen plan digest")
    selected = run["input_ref"]
    if selected is None:
        if not allow_unattached_input or input_ref is None or run["status"] != "queued":
            raise QuoteMemError("input_required",
                                "Privacy recovery requires its attached durable input")
        selected = input_ref
    elif input_ref is not None and selected != input_ref:
        raise QuoteMemError(
            "binding_mismatch", "Privacy input differs from the selected owner reference"
        )
    envelope = _get(writer, "artifact", selected)
    scope = (run["scope_kind"], run["scope_id"])
    _scope(envelope, scope)
    body = writer.read_logical(selected, "run_input").model_dump(mode="json")
    if (
        envelope["created_by_run_id"] != owner_run_id
        or body.get("plan_ref") != head_ref
        or body.get("run_kind") != "retention"
        or body.get("retention_role") != supplied.retention_role
        or body.get("admission_digest") != run["admission_digest"]
        or body.get("target_locator") != run["target_id"]
        or body.get("business_key") != supplied.model_dump(mode="json")
        or body.get("payload_digest")
        != writer.bind_content("run-input-payload", supplied.model_dump(mode="json"))
    ):
        raise QuoteMemError("binding_mismatch",
                            "Privacy input lost its owner, payload or selected head")
    reverse: list[tuple[ArtifactId, IngestionErasurePlanV1]] = []
    seen: set[str] = set()
    current: ArtifactId | None = ArtifactId(head_ref)
    proofs: dict[_RowKey, RowPrecondition] = {}
    _remember(writer, proofs, "run", run)
    _remember(writer, proofs, "artifact", envelope)
    while current is not None:
        if current in seen:
            raise QuoteMemError("reference_cycle", "Privacy chain repeats a physical block")
        seen.add(current)
        physical = _get(writer, "artifact", current)
        _scope(physical, scope)
        if physical["created_by_run_id"] != owner_run_id:
            raise QuoteMemError("binding_mismatch", "Privacy block belongs to another owner")
        block = writer.read_logical(current, "erasure_plan")
        if not isinstance(block, IngestionErasurePlanV1):
            raise QuoteMemError("binding_mismatch", "Privacy chain holds another erasure plan")
        if (
            block.controller_digest != run["admission_digest"]
            or block.admitted_erase_epoch != supplied.erase_epoch
            or block.target.evidence_digest
            != cleanup_target(supplied, writer.bind_content)
        ):
            raise QuoteMemError("binding_mismatch",
                                "Privacy block differs from canonical admission")
        _remember(writer, proofs, "artifact", physical)
        reverse.append((current, block))
        current = block.previous_ref
    blocks = list(reversed(reverse))
    summary = validate_ingestion_erasure_chain(blocks)
    if summary.head_ref != head_ref or (
        expected_plan_digest is not None and summary.plan_digest != expected_plan_digest
    ):
        raise QuoteMemError("parity_mismatch", "Privacy head or complete logical digest differs")
    batches: list[PrivacyBatch] = []
    entries: list[str] = []
    for _, block in blocks:
        for record in block.records:
            if isinstance(record, IngestionErasureBatchV1):
                if entries:
                    batches.append(PrivacyBatch(len(batches), tuple(entries)))
                    entries = []
            else:
                entries.append(canonical_json(record.entry))
    batches.append(PrivacyBatch(len(batches), tuple(entries)))
    first = blocks[0][1]
    proof = PreparedPrivacyEffects(
        owner_run_id,
        scope[0],
        scope[1],
        generation,
        tuple(proofs[key] for key in sorted(proofs)),
        first.controller_digest,
        first.admitted_erase_epoch,
        first.effect_time,
        first.target.evidence_digest,
        canonical_json(supplied),
        tuple(batches),
        summary.plan_digest,
        ArtifactId(head_ref),
    )
    _validate_target_coverage(proof)
    return proof


def _validate_target_coverage(proof: PreparedPrivacyEffects) -> None:
    """Bind every entry to the admitted targets and to its batch before any effect.

    Every Evidence entry drops the complete admitted set, and together they
    cover it exactly. A batch erases Evidence and lists its entries in canonical
    order. A dependent entry names only admitted Evidence and sits in the batch
    that erases its anchor: a candidate's first erased source, a tombstone's last.
    """
    supplied = validate_cleanup_input(json.loads(proof.input_json))
    targets = privacy_targets(proof)
    expected = {value.evidence_id: value.expected_row_digest for value in supplied.affected}
    actual = {value.evidence_id: value.expected_row_digest for value in targets}
    admitted = set(expected)
    erased_in: dict[str, int] = {}
    for batch in proof.batches:
        entries = batch.entries
        if [_order(entry) for entry in entries] != sorted(_order(entry) for entry in entries):
            raise QuoteMemError("parity_mismatch",
                                "Privacy batch entries are not in canonical order")
        evidence = [entry for entry in entries if isinstance(entry, IngestionEvidenceTombstoneV1)]
        if not evidence:
            raise QuoteMemError("parity_mismatch", "A privacy batch erases no Evidence")
        for entry in evidence:
            if {drop.object_ref for drop in entry.transform.drop_locators} != admitted:
                raise QuoteMemError(
                    "binding_mismatch", "Privacy entry changed the exact admitted drop set"
                )
            erased_in[entry.object_ref] = batch.batch_no
    if actual != expected:
        raise QuoteMemError(
            "parity_mismatch", "Privacy plan does not cover every admitted Evidence target"
        )
    for batch in proof.batches:
        for entry in batch.entries:
            if isinstance(entry, IngestionEvidenceTombstoneV1):
                continue
            if not set(entry.erased_evidence_ids) <= admitted:
                raise QuoteMemError(
                    "binding_mismatch", "A dependent privacy entry names unadmitted Evidence"
                )
            anchor = (
                entry.erased_evidence_ids[0]
                if isinstance(entry, IngestionCandidateExpiryV1)
                else entry.erased_evidence_ids[-1]
            )
            if erased_in[anchor] != batch.batch_no:
                raise QuoteMemError(
                    "parity_mismatch", "A dependent privacy entry is outside its anchor's batch"
                )


def reconstruct_privacy_batch(
    writer: Writer,
    proof: PreparedPrivacyEffects,
    batch_no: int,
) -> ReconstructedPrivacyBatch:
    """Materialize one exact all-before batch, or prove that the whole batch is already after.

    The caller must reconcile ``already_applied`` with atomic progress; it never
    authorizes accepting after-state with an unchanged durable cursor. A partial
    batch, changed dependency or changed planned identity fails closed.
    """
    if type(batch_no) is not int or not 0 <= batch_no < len(proof.batches):
        raise QuoteMemError("invalid_request", "Privacy batch index is outside the selected plan")
    _validate_target_coverage(proof)
    return _reconstruct_source_effects(
        writer, entries=proof.batches[batch_no].entries,
        scope=(proof.scope_kind, proof.scope_id), batch_no=batch_no,
        all_after={
            _key(entry): entry.intended_after_digest
            for batch in proof.batches for entry in batch.entries
            if isinstance(entry, IngestionEvidenceTombstoneV1)
        },
    )


def _dependent_state(
    writer: Writer,
    entry: IngestionCandidateExpiryV1 | IngestionTombstoneDeleteV1,
    row: dict[str, Any] | None,
) -> str:
    """Classify a dependent row as its entry's exact before-state or terminal after-state.

    A candidate's after-state is its expiry by erased Evidence, a tombstone's
    is its absence. The maintenance fence admits no other writer of either, so
    any other state is a third state.
    """
    if row is not None and _digest(writer, row) == entry.expected_before_digest:
        cited = row["evidence_ids"] if entry.entity_kind == "candidate" else (
            row["source_evidence_ids"]
        )
        if not set(entry.erased_evidence_ids) <= set(cited):
            raise QuoteMemError("parity_mismatch",
                                "A privacy dependent no longer cites its Evidence")
        return "before"
    if isinstance(entry, IngestionTombstoneDeleteV1):
        if row is None:
            return "after"
    elif row is not None and (
        (row["status"], row["reason_code"], row["hold_evidence"])
        == ("expired", "evidence_invalidated", False)
        and set(entry.erased_evidence_ids) <= set(row["evidence_ids"])
    ):
        return "after"
    raise QuoteMemError("parity_mismatch", "Privacy dependent has a third state")


def _reconstruct_source_effects(
    writer: Writer,
    *,
    entries: Sequence[IngestionErasureEntryV1],
    scope: tuple[str, str],
    batch_no: int,
    all_after: Mapping[_RowKey, str],
) -> ReconstructedPrivacyBatch:
    """Rebuild the same closed transformations for an actual selected owner.

    This helper grants no admission or cursor authority. Its caller proves the
    complete selected source set and owner before entering, and proves original
    row preconditions under the commit mutex before using the temporary rows.
    """
    if not entries or len(entries) > MAX_TX_ROWS:
        raise QuoteMemError("budget_exceeded",
                            "Source reconstruction needs a bounded complete batch")
    preconditions: dict[_RowKey, RowPrecondition] = {}
    current: dict[_RowKey, dict[str, Any] | None] = {}
    states: set[str] = set()
    for entry in entries:
        key = _key(entry)
        row = writer.get_row(*key)
        if row is not None:
            _scope(row, scope)
        digest = _digest(writer, row) if row is not None else None
        if isinstance(entry, IngestionEvidenceTombstoneV1):
            if entry.expected_before_digest == entry.intended_after_digest:
                raise QuoteMemError(
                    "parity_mismatch", "Privacy neutral witness is not an exact tombstone"
                )
            if digest == entry.intended_after_digest:
                states.add("after")
            elif digest == entry.expected_before_digest:
                verify_tombstone_source_metadata(entry.transform, EvidenceRowV1.model_validate(row))
                states.add("before")
            else:
                raise QuoteMemError("parity_mismatch", "Privacy target has a third state")
        else:
            states.add(_dependent_state(writer, entry, row))
        preconditions[key] = RowPrecondition(*key, digest)
        current[key] = row
    if len(states) > 1:
        raise QuoteMemError("parity_mismatch", "Privacy atomic batch is only partially applied")
    if states == {"after"}:
        return ReconstructedPrivacyBatch(
            batch_no,
            tuple(
                RowWrite("evidence", DOMAIN_ADAPTERS["evidence"].validate_python(row))
                for (kind, _), row in sorted(current.items())
                if kind == "evidence" and row is not None
            ),
            tuple(preconditions[key] for key in sorted(preconditions)),
            True,
        )
    sources: dict[_RowKey, dict[str, Any]] = {}
    source_sets: dict[_RowKey, set[_RowKey]] = {}
    evidence = [entry for entry in entries if isinstance(entry, IngestionEvidenceTombstoneV1)]
    for entry in evidence:
        keys: set[_RowKey] = set()
        for locator, expected in zip(
            entry.transform.source_locators, entry.transform.source_digests, strict=True
        ):
            key = (locator.entity_kind, locator.object_ref)
            row = _get(writer, *key)
            _scope(row, scope)
            actual = _digest(writer, row)
            if actual != expected and not (key[0] == "evidence" and actual == all_after.get(key)):
                raise QuoteMemError("parity_mismatch", "Privacy transform source has changed")
            if key in sources and sources[key] != row:
                raise QuoteMemError(
                    "stale_evidence", "Privacy reconstruction observed a changing source"
                )
            sources[key] = row
            preconditions[key] = RowPrecondition(*key, actual)
            keys.add(key)
        source_sets[_key(entry)] = keys
    rows: list[RowWrite] = []
    for entry in evidence:
        key = _key(entry)
        if source_sets[key] != {key}:
            raise QuoteMemError(
                "parity_mismatch", "Tombstone must derive only from its exact Evidence"
            )
        action = tombstone_evidence(
            EvidenceRowV1.model_validate(sources[key]), lambda value: _digest(writer, value)
        )
        if action.intended_after_digest != entry.intended_after_digest:
            raise QuoteMemError(
                "parity_mismatch", "Reconstructed privacy post-image differs from its frozen digest"
            )
        rows.append(RowWrite(entry.entity_kind, action.post_image))
    return ReconstructedPrivacyBatch(
        batch_no,
        tuple(rows),
        tuple(preconditions[key] for key in sorted(preconditions)),
        False,
        expiring=tuple(
            row for (kind, _), row in sorted(current.items())
            if kind == "candidate" and row is not None
        ),
        deleting=tuple(
            RowDelete(kind, identity) for kind, identity in sorted(current) if kind == "tombstone"
        ),
    )


def _count(counts: dict[str, int], entries: Sequence[IngestionErasureEntryV1]) -> None:
    for entry in entries:
        counts[entry.entity_kind] += 1


def verify_privacy_completion(writer: Writer, proof: PreparedPrivacyEffects) -> PrivacyCountsV1:
    """Prove that every planned effect reached its intended state and none is left over.

    Afterwards no pending candidate cites an admitted target and no tombstone
    has only erased sources among them; no memory's head cites
    a target unless a historical correction targets that exact head and a complete
    group survives without the erased sources. That immutable head retains only
    filtered readable quotes and its correction binding. No judge record depends
    on a target or on a deleted tombstone, and no
    ended owner's record of the scope holds a recognized
    credential (`credential_records`). The sole Writer must bind this read-only
    scan to its unchanged reference generation before clearing the fence. Raw secret
    scans and reader quiescence are separate completion obligations owned by the
    cleanup orchestrator.
    """
    counts = ZERO_COUNTS.model_dump()
    for batch in proof.batches:
        rebuilt = reconstruct_privacy_batch(writer, proof, batch.batch_no)
        if not rebuilt.already_applied:
            raise QuoteMemError(
                "closure_incomplete", "Privacy effects have not all reached their intended state"
            )
        _count(counts, batch.entries)
    _add_compensation(writer, proof, counts)
    admitted = frozenset(target.evidence_id for target in privacy_targets(proof))
    if _dependents(writer, (proof.scope_kind, proof.scope_id), admitted, {}):
        raise QuoteMemError(
            "closure_incomplete", "A candidate or tombstone still depends on erased Evidence"
        )
    if _erase_dependents(writer, proof, len(proof.batches)):
        raise QuoteMemError(
            "closure_incomplete",
            "An Item or a judge record still depends on erased Evidence, or a record "
            "holding a credential remains",
        )
    return PrivacyCountsV1.model_validate(counts)


def _erase_dependents(writer: Writer, proof: PreparedPrivacyEffects, cursor: int) -> bool:
    """Whether a memory or judge record still depends on the first ``cursor`` batches, or,
    after the last one, a record the erase deletes for holding a credential remains."""
    from quotemem.store.privacy_compensation import credential_records, erase_dependents

    scope = (proof.scope_kind, proof.scope_id)
    entries = [entry for batch in proof.batches[:cursor] for entry in batch.entries]
    return bool(erase_dependents(
        writer, scope,
        frozenset(entry.object_ref for entry in entries
                  if isinstance(entry, IngestionEvidenceTombstoneV1)),
        frozenset(entry.object_ref for entry in entries
                  if isinstance(entry, IngestionTombstoneDeleteV1)),
        now=proof.effect_time,
    )) or cursor == len(proof.batches) and bool(credential_records(writer, scope).groups)


def _add_compensation(
    writer: Writer, proof: PreparedPrivacyEffects, counts: dict[str, int],
) -> None:
    """Add the memories, revisions and links the owner's privacy compensation wrote."""
    from quotemem.store.extraction_erasure import verified_candidate_expiries
    from quotemem.store.privacy_compensation import compensation_counts

    for kind, value in compensation_counts(
        writer, (proof.scope_kind, proof.scope_id), proof.owner_run_id,
    ).items():
        counts[kind] += value
    derived = verified_candidate_expiries(
        writer, (proof.scope_kind, proof.scope_id), proof.owner_run_id,
    )
    planned = {entry.object_ref for batch in proof.batches for entry in batch.entries
               if entry.entity_kind == "candidate"}
    if derived & planned:
        raise QuoteMemError("parity_mismatch", "A derived expiry repeats a planned candidate")
    counts["candidate"] += len(derived)


def privacy_effect_counts(writer: Writer, proof: PreparedPrivacyEffects) -> PrivacyCountsV1:
    """Count a completed erase's effects: its plan entries and its compensation (the receipt's).

    Planned entries are Evidence, candidates and tombstones; compensation
    is derived at each batch from the stored rows, so its memories,
    revisions and links are counted from the durable rows the owner wrote.
    """
    counts = planned_privacy_counts(proof).model_dump()
    _add_compensation(writer, proof, counts)
    return PrivacyCountsV1.model_validate(counts)


def planned_privacy_counts(proof: PreparedPrivacyEffects) -> PrivacyCountsV1:
    """Count the rows the complete selected plan affects, by kind (the receipt's counts)."""
    counts = ZERO_COUNTS.model_dump()
    for batch in proof.batches:
        _count(counts, batch.entries)
    return PrivacyCountsV1.model_validate(counts)


def planned_erase_epoch_bumps(proof: PreparedPrivacyEffects, cursor: int) -> int:
    """The ``erase_epoch`` advances of the plan's first ``cursor`` batches."""
    return sum("erase_epoch" in batch.bumps for batch in proof.batches[:cursor])


def verify_privacy_prefix(
    writer: Writer,
    proof: PreparedPrivacyEffects,
    progress: PrivacyEraseProgressV1,
) -> None:
    """Prove the exact durable cursor partition before epoch-only startup adoption.

    Every prefix batch must be wholly at its intended after-state and every
    suffix batch wholly before. A lost acknowledgement is resolved only by the
    already-advanced atomic cursor; after-state with an old cursor is corruption.
    Past the last batch the phase is ``applying`` while a memory or judge record
    still depends on the erased Evidence, or a record the erase deletes for holding
    a recognized credential remains (`Writer.apply_privacy_erase`).
    The caller verifies the unchanged generation, owner, input and fence under
    the sole Writer lock before adoption. This helper performs no writes.
    """
    progress = PrivacyEraseProgressV1.model_validate(progress.model_dump(mode="json"))
    targets = privacy_targets(proof)
    if (
        progress.owner_run_id != proof.owner_run_id
        or not targets
        or progress.evidence_id != targets[0].evidence_id
        or progress.plan_digest != proof.plan_digest
        or progress.cursor > len(proof.batches)
    ):
        raise QuoteMemError(
            "parity_mismatch", "Privacy progress differs from its selected owner or plan"
        )
    expected_phase = (
        "prepared"
        if progress.cursor == 0
        else "applying"
        if progress.cursor < len(proof.batches)
        or _erase_dependents(writer, proof, progress.cursor)
        else "verifying"
    )
    if progress.phase != expected_phase:
        raise QuoteMemError("parity_mismatch",
                            "Privacy progress phase differs from its exact cursor")
    counts = ZERO_COUNTS.model_dump()
    for batch in proof.batches:
        expected_after = batch.batch_no < progress.cursor
        rebuilt = reconstruct_privacy_batch(writer, proof, batch.batch_no)
        if rebuilt.already_applied != expected_after:
            raise QuoteMemError(
                "parity_mismatch", "Privacy rows differ from the atomic cursor partition"
            )
        if expected_after:
            _count(counts, batch.entries)
    _add_compensation(writer, proof, counts)
    expected_last = (
        ingestion_batch_postcondition_digest(
            progress.cursor - 1, proof.batches[progress.cursor - 1].entries
        )
        if progress.cursor
        else None
    )
    if (
        progress.affected_counts.model_dump() != counts
        or progress.last_completed_digest != expected_last
        or progress.erase_epoch_bumps != planned_erase_epoch_bumps(proof, progress.cursor)
    ):
        raise QuoteMemError(
            "parity_mismatch", "Privacy progress counts, epoch bumps or last-batch digest differ"
        )
