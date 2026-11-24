"""Bounded, content-free provenance for erased Evidence.

This reader authenticates selected completed cleanup records, never reconstructs
source content and never grants source eligibility. The caller owns authorization
and the scoped read lifetime, and revalidates the proof under its admission lock.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import weakref
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from pydantic import TypeAdapter, ValidationError

from quotemem.core.artifacts import ArtifactRowV1, ManifestV1, ReceiptResultIndexV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import (
    ULID,
    ArtifactId,
    ContentBindingDigest,
    IntegrityDigest,
    canonical_json,
)
from quotemem.core.ingestion_cleanup import CleanupRunInput
from quotemem.core.privacy_persistence import (
    NOT_COVERED,
    IngestionErasurePlanV1,
    IngestionEvidenceTombstoneV1,
    RedactionReceiptV1,
    TombstoneSourceMetadataV1,
    validate_ingestion_erasure_chain,
)
from quotemem.core.query import ScopeV1
from quotemem.core.semantic import ScopeKind
from quotemem.store.domain import DOMAIN_ADAPTERS, EvidenceRowV1, SessionTargetRunRowV1
from quotemem.store.indexed import IndexedReader, ScopedRunRows
from quotemem.store.metadata import StrictModel
from quotemem.store.privacy_plan import (
    PreparedPrivacyEffects,
    RowPrecondition,
    load_privacy_plan,
    planned_privacy_counts,
    privacy_targets,
)
from quotemem.store.proof_reads import ProofReads, recording_proof_reads, require_proof_reads
from quotemem.store.row_cache import membership_family, membership_key, note_reads, suspended_reads
from quotemem.store.writer import Writer

_ULID: TypeAdapter[str] = TypeAdapter(ULID)
_ISSUANCE_KEY = secrets.token_bytes(32)


@dataclass(frozen=True)
class SourceHistoryLimits:
    """Explicit read budgets; exhaustion fails without truncating discovery."""

    max_sources: int = 4096
    max_runs: int = 4096
    max_blocks: int = 4096
    max_records: int = 65536
    max_dependencies: int = 65536
    max_logical_bytes: int = 16 * 1024 * 1024

    def validate(self) -> None:
        """Reject boolean, noninteger and nonpositive budgets."""
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise QuoteMemError("invalid_request",
                                "Source history requires positive integer budgets")


_DEFAULT_LIMITS = SourceHistoryLimits()


@dataclass(frozen=True)
class ErasedSourceOrigin:
    """One selected nonneutral cleanup witness, without original source content.

    ``reason_code`` is the erase's trigger: ``retroactive_split_secret``,
    ``source_expired`` or ``consent_withdrawn``, as its verified receipt states.
    """

    evidence_id: str
    expected_before_digest: ContentBindingDigest
    intended_after_digest: ContentBindingDigest
    metadata_json: str = field(repr=False)
    owner_run_id: str
    input_ref: ArtifactId
    result_ref: ArtifactId
    receipt_ref: ArtifactId
    plan_ref: ArtifactId
    plan_digest: IntegrityDigest
    completion_commit_seq: int
    reason_code: str

    @property
    def metadata(self) -> TombstoneSourceMetadataV1:
        """Return an independent strict value without exposing mutable proof state."""
        return TombstoneSourceMetadataV1.model_validate_json(self.metadata_json)


def cleanup_completion_sequence(source: ErasedSourceOrigin) -> int:
    """Read the positive ordering scalar of an independently authenticated origin.

    Args:
        source: A selected origin retained with its issued history proof.

    Returns:
        The cleanup's completion commit sequence, independent of wall time.

    Raises:
        QuoteMemError: If the sequence is missing, noninteger or nonpositive. This
            accessor alone authenticates neither the origin nor its ordering.
    """
    value = source.completion_commit_seq
    if type(value) is not int or value < 1:
        raise QuoteMemError("binding_mismatch", "Cleanup completion sequence is malformed")
    return value


@dataclass(frozen=True)
class ErasedSourceHistory:
    """Writer-bound read proof; it confers no mutation or source-read authority."""

    scope_kind: ScopeKind
    scope_id: str
    reads: ProofReads = field(repr=False, compare=False)
    sources: tuple[ErasedSourceOrigin, ...]
    dependencies: tuple[RowPrecondition, ...]
    owner: weakref.ReferenceType[Writer] = field(repr=False, compare=False)
    seal: str = field(repr=False, compare=False)


def _seal(proof: ErasedSourceHistory) -> str:
    value = {
        "scope_kind": proof.scope_kind,
        "scope_id": proof.scope_id,
        "reads": [proof.reads.token, id(proof.reads)],
        "sources": [asdict(source) for source in proof.sources],
        "dependencies": [asdict(dependency) for dependency in proof.dependencies],
        "owner_identity": id(proof.owner()),
    }
    return hmac.new(_ISSUANCE_KEY, canonical_json(value).encode(), hashlib.sha256).hexdigest()


def validate_erased_source_history(writer: Writer, proof: ErasedSourceHistory) -> None:
    """Revalidate issuance and exact dependencies under the caller's admission lock.

    Args:
        writer: The same Writer that issued the proof.
        proof: The complete immutable result of erased-source discovery.

    Raises:
        QuoteMemError: If issuance or fixed authorities differ, completion is ahead
            of visibility, or a write overtook the recorded rows or scoped Run
            membership. No retry occurs here; current authorization and privacy
            fences remain the caller's responsibility.
    """
    if proof.owner() is not writer or not hmac.compare_digest(proof.seal, _seal(proof)):
        raise QuoteMemError("binding_mismatch", "Source history lacks its original issuance")
    require_proof_reads(writer, proof.reads, "Source history discovery was overtaken")
    sequences = [cleanup_completion_sequence(source) for source in proof.sources]
    with suspended_reads():
        visible = writer.get_counter("commit_seq")
    if sequences and max(sequences) > visible:
        raise QuoteMemError("binding_mismatch", "Source history completion is ahead of visibility")
    for dependency in proof.dependencies:
        row = writer.get_row(dependency.entity_kind, dependency.object_id)
        if row is None or writer.bind_content("row-state", row) != dependency.digest:
            raise QuoteMemError("binding_mismatch", "Source history fixed dependency changed")


class _Read:
    """Capture bounded authorities, including every selected physical artifact part."""

    def __init__(self, writer: Writer, scope: ScopeV1, limits: SourceHistoryLimits) -> None:
        self.writer, self.scope, self.limits = writer, scope, limits
        self.dependencies: dict[tuple[str, str], RowPrecondition] = {}
        self.blocks = 0
        self.records = 0

    def remember(self, kind: str, row: dict[str, Any]) -> None:
        key = kind, row["object_id"]
        note_reads((key,))
        proof = RowPrecondition(*key, self.writer.bind_content("row-state", row))
        prior = self.dependencies.get(key)
        if prior is not None and prior != proof:
            raise QuoteMemError("stale_evidence", "Source history observed a changing authority")
        if prior is None and len(self.dependencies) >= self.limits.max_dependencies:
            raise QuoteMemError("budget_exceeded",
                                "Source history fixed dependency budget exhausted")
        self.dependencies[key] = proof

    def row(self, kind: str, identity: str) -> dict[str, Any]:
        row = self.writer.get_row(kind, identity)
        if row is None or row.get("object_id") != identity:
            raise QuoteMemError("reference_missing", "Selected source history authority is missing")
        self.remember(kind, row)
        return row

    def exact_scope(self, row: dict[str, Any]) -> None:
        if (row.get("scope_kind"), row.get("scope_id")) != (
            self.scope.scope_kind,
            self.scope.scope_id,
        ):
            raise QuoteMemError("scope_denied", "Source history authority belongs to another scope")

    def artifact(self, reference: str, kind: str, creator: str) -> StrictModel:
        raw = self.row("artifact", reference)
        self.exact_scope(raw)
        row = ArtifactRowV1.model_validate(raw)
        if (
            row.created_by_run_id != creator
            or row.tenant_id != self.writer.tenant_id
            or row.security_level != "restricted"
        ):
            raise QuoteMemError("binding_mismatch", "Source history Artifact envelope differs")
        if row.artifact_kind == "manifest":
            manifest = ManifestV1.model_validate_json(row.body)
            if manifest.logical_kind != kind:
                raise QuoteMemError("binding_mismatch", "Source history manifest kind differs")
            if manifest.total_bytes > self.limits.max_logical_bytes or len(manifest.parts) > (
                self.limits.max_dependencies - len(self.dependencies)
            ):
                raise QuoteMemError(
                    "budget_exceeded", "Source history logical Artifact budget exhausted"
                )
            for part in manifest.parts:
                self.row("artifact", part)
        elif row.artifact_kind != kind:
            raise QuoteMemError("binding_mismatch", "Source history Artifact kind differs")
        elif len(row.body.encode("utf-8")) > self.limits.max_logical_bytes:
            raise QuoteMemError("budget_exceeded",
                                "Source history logical Artifact budget exhausted")
        # The owning reader validates content identities, canonical reconstruction
        # and each part's creator, scope and protection envelope.
        return self.writer.read_logical(reference, kind)


@dataclass(frozen=True)
class _CompletedCleanup:
    input_ref: ArtifactId
    result_ref: ArtifactId
    receipt_ref: ArtifactId
    completion_commit_seq: int
    plan: PreparedPrivacyEffects


def _completed_cleanup(
    read: _Read,
    run: SessionTargetRunRowV1,
    body: CleanupRunInput,
) -> _CompletedCleanup:
    if (
        run.status != "succeeded"
        or run.input_ref is None
        or run.result_ref is None
        or run.stage != "scan"
        or run.started_at is None
        or run.finished_at is None
        or run.commit_marker is not None
        or run.commit_seq is not None
    ):
        raise QuoteMemError("binding_mismatch",
                            "Source history requires completed selected cleanup")
    if run.parent_run_id is not None:
        read.exact_scope(read.row("run", run.parent_run_id))
    result = read.artifact(run.result_ref, "result_index", run.object_id)
    if not isinstance(result, ReceiptResultIndexV1):
        raise QuoteMemError("binding_mismatch", "Cleanup selected result has no verified receipt")
    receipt = read.artifact(result.receipt_ref, "receipt", run.object_id)
    if not isinstance(receipt, RedactionReceiptV1) or (
        receipt.reason_code != body.business_key.reason_code
    ):
        raise QuoteMemError("unsupported_variant",
                            "Cleanup requires its sequenced completion receipt")
    sequence = receipt.completion_commit_seq
    with suspended_reads():
        visible = read.writer.get_counter("commit_seq")
    if type(sequence) is not int or sequence < 1 or sequence > visible:
        raise QuoteMemError("binding_mismatch", "Selected cleanup completion is outside visibility")

    reverse: list[tuple[ArtifactId, IngestionErasurePlanV1]] = []
    seen: set[str] = set()
    reference: ArtifactId | None = body.plan_ref
    while reference is not None:
        if reference in seen:
            raise QuoteMemError("reference_cycle", "Selected cleanup plan repeats a block")
        if read.blocks >= read.limits.max_blocks:
            raise QuoteMemError("budget_exceeded", "Source history plan block budget exhausted")
        seen.add(reference)
        read.blocks += 1
        block = read.artifact(reference, "erasure_plan", run.object_id)
        if not isinstance(block, IngestionErasurePlanV1):
            raise QuoteMemError("unsupported_variant", "Source history has another cleanup plan")
        read.records += block.record_count
        if read.records > read.limits.max_records:
            raise QuoteMemError("budget_exceeded", "Source history plan record budget exhausted")
        reverse.append((reference, block))
        reference = block.previous_ref
    summary = validate_ingestion_erasure_chain(reversed(reverse))
    plan = load_privacy_plan(
        read.writer,
        body.plan_ref,
        owner_run_id=run.object_id,
        supplied=body.business_key,
        expected_plan_digest=summary.plan_digest,
    )
    for dependency in plan.preconditions:
        raw = read.row(dependency.entity_kind, dependency.object_id)
        if read.writer.bind_content("row-state", raw) != dependency.digest:
            raise QuoteMemError("stale_evidence",
                                "Selected cleanup changed between its proof reads")
    targets = privacy_targets(plan)
    if receipt.affected_digests != sorted(target.expected_row_digest for target in targets):
        raise QuoteMemError("binding_mismatch", "Erase receipt changed its selected targets")
    # The planned kinds are bound exactly; the compensation kinds (item,
    # revision, link) are bound by the owner's completion proof, because
    # counting them here would enumerate the scope's memories and make this
    # proof stale on any other scope's write (`proof_reads`).
    planned = planned_privacy_counts(plan).model_dump()
    if (
        any(receipt.counts.model_dump()[kind] != planned[kind]
            for kind in ("evidence", "candidate", "tombstone"))
        or receipt.not_covered != NOT_COVERED
    ):
        raise QuoteMemError("binding_mismatch",
                            "Selected receipt differs from complete cleanup effects")
    return _CompletedCleanup(run.input_ref, run.result_ref, result.receipt_ref, sequence, plan)


def _selected_cleanup(
    read: _Read,
    run: SessionTargetRunRowV1,
    requested: dict[str, EvidenceRowV1],
) -> tuple[ErasedSourceOrigin, ...]:
    if run.input_ref is None:
        raise QuoteMemError("binding_mismatch", "Completed cleanup has no selected input")
    body = read.artifact(run.input_ref, "run_input", run.object_id)
    if not isinstance(body, CleanupRunInput):
        raise QuoteMemError("unsupported_variant",
                            "Source history opens only selected cleanup inputs")
    if not {target.evidence_id for target in body.business_key.affected}.intersection(requested):
        return ()
    completed = _completed_cleanup(read, run, body)
    sources: list[ErasedSourceOrigin] = []
    for batch in completed.plan.batches:
        for entry in batch.entries:
            # Candidate expiries and tombstone deletions are not source transitions.
            if not isinstance(entry, IngestionEvidenceTombstoneV1):
                continue
            if entry.object_ref not in requested:
                continue
            current = requested[entry.object_ref]
            if (
                read.writer.bind_content("row-state", current.model_dump(mode="json"))
                != entry.intended_after_digest
                or entry.expected_before_digest == entry.intended_after_digest
            ):
                raise QuoteMemError(
                    "binding_mismatch", "Erased source differs from its selected transition"
                )
            metadata = entry.transform.closed_patch
            sources.append(
                ErasedSourceOrigin(
                    current.object_id,
                    entry.expected_before_digest,
                    entry.intended_after_digest,
                    canonical_json(metadata),
                    run.object_id,
                    completed.input_ref,
                    completed.result_ref,
                    completed.receipt_ref,
                    body.plan_ref,
                    completed.plan.plan_digest,
                    completed.completion_commit_seq,
                    body.business_key.reason_code,
                )
            )
    return tuple(sources)


def resolve_erased_source_history(
    writer: Writer,
    scope: ScopeV1,
    evidence_ids: Sequence[str],
    *,
    limits: SourceHistoryLimits = _DEFAULT_LIMITS,
) -> ErasedSourceHistory:
    """Resolve erased sources through a complete bounded search of selected cleanup.

    Args:
        writer: Sole storage owner, inside the caller's authorized scoped read.
        scope: Exact scope of every requested source and selected authority.
        evidence_ids: Nonempty distinct erased Evidence IDs.
        limits: Positive budgets; reaching one never returns a partial proof.

    Returns:
        A sealed proof with sources sorted by ID and exact fixed dependencies.
        Mutable compensated rows need not match historical cleanup post-images;
        the requested erased Evidence must still match its selected after-state.

    Raises:
        QuoteMemError: For malformed input, exhausted budgets, missing or ambiguous
            witnesses, changed dependencies or invalid selected records. Only
            currently produced sequenced cleanup records are accepted.
    """
    limits.validate()
    if not evidence_ids or len(evidence_ids) > limits.max_sources:
        raise QuoteMemError("invalid_request",
                            "Source history requires bounded distinct source IDs")
    try:
        scope = ScopeV1.model_validate(scope.model_dump(mode="json"))
        ids = tuple(_ULID.validate_python(value, strict=True) for value in evidence_ids)
    except ValidationError:
        raise QuoteMemError("invalid_request", "Source history coordinates are invalid") from None
    if len(ids) != len(set(ids)):
        raise QuoteMemError("invalid_request",
                            "Source history requires bounded distinct source IDs")
    read = _Read(writer, scope, limits)
    try:
        with recording_proof_reads(writer) as recorded:
            requested: dict[str, EvidenceRowV1] = {}
            for identity in sorted(ids):
                raw = read.row("evidence", identity)
                read.exact_scope(raw)
                row = EvidenceRowV1.model_validate(raw)
                if row.tenant_id != writer.tenant_id or row.state != "erased":
                    raise QuoteMemError(
                        "binding_mismatch", "Source history requires actual erased Evidence"
                    )
                requested[identity] = row
            reader = IndexedReader(
                writer.runtime.workspace, writer.entities, tenant_id=writer.tenant_id
            )
            query = ScopedRunRows(
                scope_kind=scope.scope_kind,
                scope_id=scope.scope_id,
                run_kind="retention",
            )
            after = None
            run_count = 0
            found: dict[str, ErasedSourceOrigin] = {}
            # Paging does not record membership itself. This exact scoped Run set
            # is retired by every Run upsert in the scope, including insertions.
            note_reads(
                (
                    membership_family("run-scope"),
                    membership_key("run-scope", f"{scope.scope_kind}:{scope.scope_id}"),
                )
            )
            while True:
                with suspended_reads():
                    page = reader.page(
                        query, after=after, limit=min(128, limits.max_runs - run_count + 1)
                    )
                if not page:
                    break
                for raw in page:
                    run_count += 1
                    if run_count > limits.max_runs:
                        raise QuoteMemError(
                            "budget_exceeded", "Source history scoped Run budget exhausted"
                        )
                    read.exact_scope(raw)
                    read.remember("run", raw)
                    run = DOMAIN_ADAPTERS["run"].validate_python(raw)
                    if run.status != "succeeded":
                        continue
                    if not isinstance(run, SessionTargetRunRowV1) or (
                        run.commit_mode != "destructive_scan" or run.target_kind != "affected_set"
                    ):
                        raise QuoteMemError(
                            "unsupported_variant", "Source history opens only scoped cleanup"
                        )
                    for origin in _selected_cleanup(read, run, requested):
                        if origin.evidence_id in found:
                            raise QuoteMemError(
                                "binding_mismatch", "Source history has ambiguous completed owners"
                            )
                        found[origin.evidence_id] = origin
                after = page[-1]["object_id"]
        if found.keys() != requested.keys():
            raise QuoteMemError("unsupported_variant",
                                "Erased source has no completed selected witness")
        proof = ErasedSourceHistory(
            scope.scope_kind,
            scope.scope_id,
            recorded(),
            tuple(found[key] for key in sorted(found)),
            tuple(read.dependencies[key] for key in sorted(read.dependencies)),
            weakref.ref(writer),
            "",
        )
        proof = replace(proof, seal=_seal(proof))
        validate_erased_source_history(writer, proof)
        return proof
    except ValidationError:
        raise QuoteMemError("schema_invalid",
                            "Stored source history authority is invalid") from None
