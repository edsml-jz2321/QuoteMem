"""Content-free extraction records that let an erased owner finish without replay sends."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from aml.core.entities import PositionKey
from aml.core.errors import AMLError
from aml.core.extraction_records import (
    ErasedCandidateExpiryV1,
    ErasedExtractionV1,
    ErasedProposalV1,
    ErasedQuoteSpanV1,
    ErasedReplyV1,
)
from aml.core.inference import StoredInferenceOutput
from aml.core.proposal import ProposalV2
from aml.core.semantic import ScopeKind
from aml.store.indexed import MAX_PAGE_SIZE, ScopedRunRows

if TYPE_CHECKING:
    from aml.store.registry import Reference
    from aml.store.writer import RowWrite, Writer


def erased_extraction_record(
    writer: Writer, owner_run_id: str, references: Collection[str],
) -> ErasedExtractionV1:
    """Capture only identities, admission codes and source coordinates before deleting bodies.

    Args:
        writer: The sole Writer, while the erase holds its write scope.
        owner_run_id: The extraction owner whose selected records will leave.
        references: Logical ids selected for deletion, including replies and proposals.

    Returns:
        Metadata to persist under the privacy owner before deleting any body.
    """
    replies: list[ErasedReplyV1] = []
    proposals: list[ErasedProposalV1] = []
    for reference, body in writer.find_logical_candidates(owner_run_id, "llm_output"):
        if reference not in references or not isinstance(body, StoredInferenceOutput) or (
            body.purpose != "extract"
        ):
            continue
        replies.append(ErasedReplyV1(input_digest=body.input_digest, call_no=body.call_no,
                                    extraction_ref=reference, admission=body.admission))
    for reference, body in writer.find_logical_candidates(owner_run_id, "proposal_snapshot"):
        if reference not in references or not isinstance(body, ProposalV2):
            continue
        primary = body.anchors[0]
        entity_id = "user" if primary.kind == "user" else (
            primary.entity_id if primary.kind == "existing" or primary.kind == "participant"
            else None
        )
        proposals.append(ErasedProposalV1(
            position_key=body.position_key, proposal_ref=reference,
            extraction_ref=body.extraction_ref,
            quote_spans=[ErasedQuoteSpanV1.model_validate(span.model_dump())
                         for span in body.quote_spans],
            statement_digest=body.statement_digest, entity_id=entity_id,
        ))
    return ErasedExtractionV1(
        owner_run_id=owner_run_id,
        erased_extraction=sorted(replies, key=lambda reply: (reply.input_digest, reply.call_no)),
        proposals=sorted(proposals, key=lambda proposal: proposal.position_key),
    )


def erased_extraction_records(
    writer: Writer, owner_run_id: str,
) -> tuple[dict[tuple[str, int], ErasedReplyV1], dict[PositionKey, ErasedProposalV1]]:
    """Read an owner's cumulative erasure accounting and reject conflicting overlap.

    Returns:
        Calls by request digest and logical call number, and proposals by original position.

    Raises:
        AMLError: ``binding_mismatch`` if two erasures disagree about one original record.
    """
    replies: dict[tuple[str, int], ErasedReplyV1] = {}
    proposals: dict[PositionKey, ErasedProposalV1] = {}
    owner = writer.get_row("run", owner_run_id)
    if owner is None:
        raise AMLError("reference_missing", "Erasure metadata requires its extraction owner")
    query = ScopedRunRows(scope_kind=cast(ScopeKind, owner["scope_kind"]),
                          scope_id=owner["scope_id"], run_kind="retention")
    reader = writer.indexed_reader()
    after: str | None = None
    while True:
        page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
        for run in page:
            for _reference, body in writer.find_logical_candidates(run["object_id"],
                                                                  "decision_set"):
                if not isinstance(body, ErasedExtractionV1) or body.owner_run_id != owner_run_id:
                    continue
                for reply in body.erased_extraction:
                    key = reply.input_digest, reply.call_no
                    if replies.setdefault(key, reply) != reply:
                        raise AMLError("binding_mismatch", "Erasure metadata changed a stored call")
                for proposal in body.proposals:
                    if proposals.setdefault(proposal.position_key, proposal) != proposal:
                        raise AMLError("binding_mismatch",
                                       "Erasure metadata changed a stored position")
        if len(page) < MAX_PAGE_SIZE:
            break
        after = page[-1]["object_id"]
    return replies, proposals


def erased_candidate_reference(
    writer: Writer, row: Mapping[str, Any], reference: Reference,
) -> bool:
    """Prove a candidate's missing artifact from its original erasure accounting.

    A surviving proposal proves its position when only the unit's reply left.
    A deleted proposal proves only an exact ``evidence_invalidated`` rejection.
    """
    if reference.path not in ("extraction_ref", "proposal_ref"):
        return False
    owner = writer.get_row("run", row["owner_run_id"])
    if owner is None or (owner["scope_kind"], owner["scope_id"]) != (
        row["scope_kind"], row["scope_id"],
    ):
        return False
    replies, proposals = erased_extraction_records(writer, row["owner_run_id"])
    key = row["owner_run_id"], row["unit_no"], row["position"]
    if reference.path == "extraction_ref" and any(
        reply.extraction_ref == reference.target_id for reply in replies.values()
    ) and writer.get_row("artifact", row["proposal_ref"]) is not None:
        surviving = writer.read_logical(row["proposal_ref"], "proposal_snapshot")
        if isinstance(surviving, ProposalV2):
            return (surviving.position_key, surviving.extraction_ref,
                    surviving.statement_digest,
                    sorted({span.evidence_id for span in surviving.quote_spans})) == (
                        key, reference.target_id, row["statement_digest"], row["evidence_ids"],
                    )
    if (row["status"], row["reason_code"]) != ("rejected", "evidence_invalidated"):
        return False
    proposal = proposals.get(key)
    return proposal is not None and (
        proposal.proposal_ref, proposal.extraction_ref, proposal.statement_digest,
        proposal.entity_id, sorted({span.evidence_id for span in proposal.quote_spans}),
    ) == (row["proposal_ref"], row["extraction_ref"], row["statement_digest"],
          row["entity_id"], row["evidence_ids"]) and reference.target_id == row[reference.path]


def candidate_expiry_records(
    writer: Writer, rows: Sequence[RowWrite],
) -> list[ErasedExtractionV1]:
    """Build witnesses that the Writer stores in each derived expiry's own transaction."""
    owners: dict[str, list[ErasedCandidateExpiryV1]] = {}
    for write in rows:
        row = write.row.model_dump(mode="json")
        position = row["owner_run_id"], row["unit_no"], row["position"]
        owners.setdefault(row["owner_run_id"], []).append(ErasedCandidateExpiryV1(
            candidate_id=row["object_id"], position_key=position,
            after_digest=writer.bind_content("row-state", row),
        ))
    return [ErasedExtractionV1(
        owner_run_id=owner, erased_extraction=[], proposals=[],
        expired_candidates=sorted(expiries, key=lambda item: item.candidate_id),
    ) for owner, expiries in sorted(owners.items())]


def verified_candidate_expiries(
    writer: Writer, scope: tuple[str, str], privacy_owner: str,
) -> frozenset[str]:
    """Prove the privacy owner's derived terminal candidate post-images for receipt/recovery."""
    found: set[str] = set()
    for _reference, body in writer.find_logical_candidates(privacy_owner, "decision_set"):
        if not isinstance(body, ErasedExtractionV1):
            continue
        for expiry in body.expired_candidates:
            row = writer.get_row("candidate", expiry.candidate_id)
            if row is None or (row["scope_kind"], row["scope_id"]) != scope or (
                row["owner_run_id"], row["unit_no"], row["position"]
            ) != expiry.position_key or (row["status"], row["reason_code"]) != (
                "expired", "evidence_invalidated",
            ) or writer.bind_content("row-state", row) != expiry.after_digest:
                raise AMLError("parity_mismatch", "A derived candidate expiry lost its post-image")
            if expiry.candidate_id in found:
                raise AMLError("parity_mismatch", "A privacy owner repeated a candidate expiry")
            found.add(expiry.candidate_id)
    return frozenset(found)
