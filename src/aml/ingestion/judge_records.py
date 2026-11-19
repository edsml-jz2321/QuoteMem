"""Build content-free judge records from the actual nominations and verdicts.

`build_call_manifest` writes a call's summary (judge-Q22); `resolve_targets`
writes a verdict's resolved target map from the member's own snapshot
(W3-RT-11).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from aml.core.errors import AMLError
from aml.core.identity import ArtifactId, IntegrityDigest
from aml.core.judge_output import MAX_PACKET_MEMBERS, JudgeV8
from aml.core.judge_records import (
    JudgeCallManifestV1,
    JudgeResolvedRelationV1,
    JudgeResolvedSiblingV1,
    JudgeResolvedSlotV1,
    JudgeResolvedTargetsV1,
)
from aml.core.neighbor_snapshot import NeighborSnapshotV1, PositionV1
from aml.ingestion.nomination import NominationV1


def build_call_manifest(
    nominations: Sequence[NominationV1], *, snapshot_refs: Sequence[ArtifactId],
    binding_digest: IntegrityDigest, request_digest: IntegrityDigest,
    judge_attempt: Literal[1, 2], timeout_ms: int,
    judge_timeout_policy_digest: IntegrityDigest, send_reserved: bool = True,
) -> JudgeCallManifestV1:
    """Summarize every input span and revision without copying member content.

    Args:
        nominations: The actual physical request's nominations, in request order.
        snapshot_refs: Their persisted snapshot references in the same order.
        binding_digest: The complete frozen judge binding digest.
        request_digest: Integrity digest of the physical InferenceRequest.
        judge_attempt: First planned request or second solo request.
        timeout_ms: Effective read wait frozen in that request.
        judge_timeout_policy_digest: Complete frozen first/second timeout policy.
        send_reserved: Whether a physical send was durably reserved for this
            exact request; a crash or later time shortfall can prevent HTTP.

    Returns:
        A manifest ready for the accepted reply's atomic Writer transaction.
        Context tails retain the whole parts' spans, including withheld ranges.

    Raises:
        AMLError: ``binding_mismatch`` if membership is empty, oversized or repeated,
            or if a nomination has no paired snapshot reference.
    """
    if not 1 <= len(nominations) <= MAX_PACKET_MEMBERS or (
        len(nominations) != len(snapshot_refs)
    ) or len({nomination.member.position_key for nomination in nominations}) != len(nominations):
        raise AMLError("binding_mismatch", "Judge manifest differs from its member snapshots")
    spans = {
        (span.evidence_id, span.byte_start, span.byte_end)
        for nomination in nominations for span in nomination.member.quotes
    }
    spans.update(
        (part.object_id, part.byte_start, part.byte_end)
        for nomination in nominations for source in nomination.sources
        for part in source.message.parts
    )
    revisions = {
        revision for nomination in nominations for revision in (
            *nomination.dependencies.revision_ids, *nomination.dependencies.naming_revision_ids,
        )
    }
    return JudgeCallManifestV1.model_validate({
        "binding_digest": binding_digest, "request_digest": request_digest,
        "judge_attempt": judge_attempt, "timeout_ms": timeout_ms,
        "judge_timeout_policy_digest": judge_timeout_policy_digest,
        "send_reserved": send_reserved,
        "member_snapshot_refs": list(snapshot_refs),
        "tombstone_issued": any(nomination.tombstones for nomination in nominations),
        "evidence_spans": [
            {"evidence_id": evidence, "byte_start": start, "byte_end": end}
            for evidence, start, end in sorted(spans)
        ],
        "revision_ids": sorted(revisions),
    })


def resolve_targets(snapshot: NeighborSnapshotV1, verdict: JudgeV8) -> JudgeResolvedTargetsV1:
    """Map one accepted verdict's handles to the ids its snapshot issued (W3-RT-11).

    Args:
        snapshot: The member's ``NeighborSnapshotV1`` in this round; for a
            borrowed verdict, the borrowing member's snapshot (an identical
            member input names the same objects under the same handles).
        verdict: The accepted ``JudgeV8``.

    Returns:
        For each relation, the target's kind and ids (a sibling by position);
        for each slot in order, its verdict (``mechanical`` when not judged) and
        the existing entity it binds, null when none exists yet.

    Raises:
        AMLError: ``binding_mismatch`` when the verdict names a handle or slot
            the snapshot did not issue, or misses a judged slot.
    """
    targets = {target.handle: target for target in snapshot.targets}
    entities = {facts.handle: facts.entity_id for facts in snapshot.entity_facts}
    verdicts = {entry.slot: entry for entry in verdict.entities}
    relations: list[JudgeResolvedRelationV1] = []
    siblings: list[JudgeResolvedSiblingV1] = []
    for item in verdict.relations:
        target = targets.get(item.target)
        if target is None:
            raise AMLError("binding_mismatch", "A verdict relates an unissued target")
        if target.kind == "sibling":
            owner, unit, position = target.object_id.rsplit(":", 2)
            siblings.append(JudgeResolvedSiblingV1(position=PositionV1(
                owner_run_id=owner, unit_no=int(unit), position=int(position),
            ), relation=item.relation))
            continue
        if target.revision_id is None:
            raise AMLError("binding_mismatch", "A committed target lost its revision")
        relations.append(JudgeResolvedRelationV1(
            kind=target.kind, memory_id=target.object_id, revision_id=target.revision_id,
            relation=item.relation,
        ))
    slots: list[JudgeResolvedSlotV1] = []
    for slot in snapshot.entities:
        judged = slot.binding in ("nominated", "open")
        entry = verdicts.pop(slot.slot, None)
        if judged != (entry is not None):
            raise AMLError("binding_mismatch", "A verdict differs from the judged slots")
        if entry is None:
            bound = None if slot.entity_handle is None else entities.get(slot.entity_handle)
            slots.append(JudgeResolvedSlotV1(slot=slot.slot, verdict="mechanical",
                                             entity_id=bound))
            continue
        candidates = {candidate.handle: candidate.entity_id for candidate in slot.candidates}
        chosen = {"confirm": slot.nominated_handle, "repick": entry.choice,
                  "ambiguous": None}[entry.verdict]
        if chosen is not None and chosen not in candidates:
            raise AMLError("binding_mismatch", "A slot verdict chose an unissued candidate")
        slots.append(JudgeResolvedSlotV1(
            slot=slot.slot, verdict=entry.verdict,
            entity_id=None if chosen is None else candidates[chosen],
        ))
    if verdicts:
        raise AMLError("binding_mismatch", "A verdict judged a slot the snapshot lacks")
    return JudgeResolvedTargetsV1(relations=relations, siblings=siblings, slots=slots)
