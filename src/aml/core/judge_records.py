"""Content-free judge record contracts (C.13); storage belongs to the single Writer."""

from __future__ import annotations

import re
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from aml.core.entities import EntityRef, PositionKey
from aml.core.identity import ULID, ArtifactId, IntegrityDigest, integrity_digest
from aml.core.inference import InferenceBindingV1
from aml.core.judge_output import JUDGE_BATCH_V8_SCHEMA_ID, MAX_PACKET_MEMBERS, JudgeFault, Relation
from aml.core.neighbor_snapshot import PositionV1
from aml.core.provider_diagnostics import ProviderSchemaRefusalV1
from aml.core.review_rounds import MAX_ROUND, RoundNo
from aml.core.technical_types import ContentDigest, Integrity, NonNegativeInt
from aml.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

SnapshotRef = Annotated[
    ArtifactId, RefSpec("artifact", scope_rule="same_scope", ref_mode="snapshot_locator",
                        artifact_kinds=("decision_set", "manifest")),
]
OutputRef = Annotated[
    ArtifactId, RefSpec("artifact", scope_rule="same_scope", ref_mode="snapshot_locator",
                        artifact_kinds=("llm_output", "manifest")),
]
ResultRef = Annotated[
    ArtifactId, RefSpec("artifact", scope_rule="same_scope", ref_mode="snapshot_locator",
                        artifact_kinds=("result_index", "manifest")),
]
RunRef = Annotated[
    ULID, RefSpec("run", scope_rule="same_scope", ref_mode="historical_locator",
                 fulfillment="none"),
]

CONFIRMATION_VOTES: Final = 3
"""Verdicts that decide a consequential member: one a verdict denies as a restated
forgotten fact or that ends an issued head (C.8 #9, ``judge_confirmation``).

The member's own verdict is the first; at most two fresh solo votes follow, each from
its own judge child, until two share a consequence."""


class JudgeEvidenceSpanV1(StrictModel):
    """An input range for Why/erase, including an erased part's (0, 0) placeholder.

    The Evidence reference remains even when its erased part has no coordinates;
    P-SPAN refuses that Evidence. Member quote ranges remain nonempty upstream.
    """

    evidence_id: Annotated[
        ULID, RefSpec("evidence", scope_rule="same_scope", ref_mode="snapshot_locator"),
    ]
    byte_start: NonNegativeInt
    byte_end: NonNegativeInt

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        if self.byte_start > self.byte_end or self.byte_start == self.byte_end != 0:
            raise ValueError("Judge input ranges are nonempty or an erased-part placeholder")
        return self


class JudgeCallManifestV1(StrictModel):
    """One call's ordered snapshots and content-free whole-call readability summary.

    No prompt, member text, handle, entity or tombstone identity is stored here.
    The numbered attempt, effective request wait and complete timeout policy
    identify the call. ``send_reserved`` records a durable send reservation,
    not proof of HTTP dispatch: a crash or a later time shortfall may prevent
    the reserved send. The Writer persists this and every accepted member in
    one transaction.
    """

    binding_digest: Integrity
    request_digest: Integrity
    judge_attempt: Literal[1, 2]
    timeout_ms: Annotated[int, Field(ge=1, le=900000)]
    judge_timeout_policy_digest: Integrity
    send_reserved: bool
    member_snapshot_refs: Annotated[
        list[SnapshotRef], Field(min_length=1, max_length=MAX_PACKET_MEMBERS),
    ]
    tombstone_issued: bool
    evidence_spans: Annotated[list[JudgeEvidenceSpanV1], Field(min_length=1)]
    revision_ids: Annotated[
        list[ULID], RefSpec("revision", cardinality="many", scope_rule="same_scope",
                            ref_mode="snapshot_locator"),
    ]

    @model_validator(mode="after")
    def canonical_summary(self) -> Self:
        if len(set(self.member_snapshot_refs)) != len(self.member_snapshot_refs):
            raise ValueError("A judge call cannot carry a snapshot twice")
        spans = [(span.evidence_id, span.byte_start, span.byte_end) for span in self.evidence_spans]
        if spans != sorted(set(spans)) or self.revision_ids != sorted(set(self.revision_ids)):
            raise ValueError("Judge call dependencies must be sorted and unique")
        return self


class JudgeResolvedRelationV1(StrictModel):
    """One relation verdict on an issued committed target, by id (W3-RT-11)."""

    kind: Literal["head", "retired"]
    memory_id: Annotated[
        ULID, RefSpec("item", ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False),
    ]
    revision_id: Annotated[
        ULID, RefSpec("revision", ref_mode="snapshot_locator", scope_rule="same_scope"),
    ]
    relation: Relation


class JudgeResolvedSiblingV1(StrictModel):
    """One relation verdict on an issued sibling of the same Add, by position."""

    position: PositionV1
    relation: Relation


class JudgeResolvedSlotV1(StrictModel):
    """One slot's verdict with the entity it binds; ``mechanical`` for an unjudged slot.

    ``entity_id`` is a local identity (``user`` has no row), as in the
    snapshot it is resolved from. It is null for ``ambiguous``, for a choice of a new declaration
    and for a slot bound to a participant or declaration not yet minted.
    """

    slot: Annotated[int, Field(ge=0, le=7)]
    verdict: Literal["confirm", "repick", "ambiguous", "mechanical"]
    entity_id: Annotated[EntityRef, OpaqueIdSpec("local")] | None

    @model_validator(mode="after")
    def ambiguous_binds_nothing(self) -> Self:
        if self.verdict == "ambiguous" and self.entity_id is not None:
            raise ValueError("An ambiguous slot binds no entity")
        return self


class JudgeResolvedTargetsV1(StrictModel):
    """The content-free resolved target map of one accepted verdict (W3-RT-11).

    It is written with the judgment from the member's own snapshot and never
    re-derived, so Why reads verdicts without loading a snapshot (DM-06). It
    holds ids and closed codes only: no handle, text or tombstone identity.
    """

    relations: Annotated[list[JudgeResolvedRelationV1], Field(max_length=64)]
    siblings: Annotated[list[JudgeResolvedSiblingV1], Field(max_length=64)]
    slots: Annotated[list[JudgeResolvedSlotV1], Field(max_length=8)]

    @model_validator(mode="after")
    def one_entry_each(self) -> Self:
        relations = [(item.memory_id, item.revision_id) for item in self.relations]
        siblings = [(item.position.owner_run_id, item.position.unit_no, item.position.position)
                    for item in self.siblings]
        slots = [item.slot for item in self.slots]
        if len(set(relations)) != len(relations) or len(set(siblings)) != len(siblings) or (
            slots != sorted(set(slots))
        ):
            raise ValueError("A resolved map names each target and slot once, slots in order")
        return self


class JudgeReplyRefusalV1(StrictModel):
    """One judge reply that left a member without an accepted verdict (C.7.2, C.8 #4).

    Content-free. ``fault`` is the first C.7.2 rule the member's result broke in a
    reply the adapter accepted (``missing`` when the reply left the member out,
    ``envelope`` when its member ids were unusable, ``result_schema`` when the result
    is outside the closed vocabularies). Otherwise the adapter refused the whole reply
    (``schema_invalid``) and ``schema_refusal`` is its reason: ``truncated`` for a
    finish reason other than ``stop``, ``content_json``, ``output_schema`` and the
    others of B.5; null only when the adapter gave none.

    Attributes:
        members: How many members the physical request carried (1: sent alone).
        fault: The broken acceptance rule, for a reply the adapter accepted.
        schema_refusal: The adapter's reason for refusing the whole reply.
        output_tokens: The reply's completion tokens, when the provider reported them.
    """

    members: Annotated[int, Field(ge=1, le=MAX_PACKET_MEMBERS)]
    fault: JudgeFault | None
    schema_refusal: ProviderSchemaRefusalV1 | None
    output_tokens: NonNegativeInt | None

    @model_validator(mode="after")
    def one_cause(self) -> Self:
        if self.fault is not None and self.schema_refusal is not None:
            raise ValueError("A refused judge reply is refused by acceptance or the adapter")
        return self


class JudgeMemberUnavailableV1(StrictModel):
    """Why a member still has no accepted verdict after its solo resend (C.8 #4, INV:E20).

    Its unavailable result records it. ``replies`` lists each reply that was not
    accepted, in send order: the first (shared or alone) and the solo resend. A
    member whose shared request timed out has only its solo reply (C.8 #5).
    """

    position: PositionV1
    replies: Annotated[list[JudgeReplyRefusalV1], Field(min_length=1, max_length=2)]


class JudgeResultIndexV8(StrictModel):
    """A verdict locator or an unavailable result; unavailable is never a verdict.

    ``resolved`` is the verdict's content-free target map (W3-RT-11), present
    exactly for a verdict, a borrowed one included (resolved against the
    borrowing member's own snapshot, whose identical input names the same
    objects). An unavailable result is an exhausted read timeout or spent
    request (C.8 #6), or, with ``refusal``, a member whose solo resend was
    still not accepted (C.8 #4): only the former meets the owner's census.
    ``refusal`` is left out when absent, so every other result keeps its bytes.
    """

    kind: Literal["verdict", "unavailable"]
    output_ref: OutputRef | None
    snapshot_ref: SnapshotRef
    call_ref: SnapshotRef
    member_input_digest: ContentDigest
    binding_digest: Integrity
    reason_readable: bool
    borrowed_from: ResultRef | None
    resolved: JudgeResolvedTargetsV1 | None
    refusal: JudgeMemberUnavailableV1 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def verdict_only_fields(self) -> Self:
        if (self.kind == "verdict") != (self.output_ref is not None) or (
            (self.kind == "verdict") != (self.resolved is not None)
        ):
            raise ValueError("Exactly a verdict names an accepted output and its resolved map")
        if self.kind == "unavailable" and (self.reason_readable or self.borrowed_from is not None):
            raise ValueError("An unavailable result has neither reason nor borrowed verdict")
        if self.refusal is not None and self.kind != "unavailable":
            raise ValueError("Only an unavailable result records a refused member")
        return self


class JudgmentAdmissionSeedV1(StrictModel):
    """What admits one judge child: its reviewing owner, position, round, vote and input (C.13).

    One child exists per (reviewing owner, position, round, vote); a re-entered
    position keeps its original owner in ``position_key``. The snapshot is
    attached with the input, after the child exists to create it. ``vote`` is 1
    for the member's review and 2 or 3 for a confirmation vote (C.8 #9); it is
    left out when 1, so a review's admission keeps its bytes.
    """

    owner_run_id: RunRef
    position_key: PositionKey
    round: RoundNo
    member_input_digest: ContentDigest
    binding: InferenceBindingV1
    vote: Annotated[int, Field(ge=1, le=CONFIRMATION_VOTES)] = Field(
        default=1, exclude_if=lambda value: value == 1,
    )

    @model_validator(mode="after")
    def judge_binding(self) -> Self:
        if self.binding.purpose != "judge" or self.binding.schema_id != JUDGE_BATCH_V8_SCHEMA_ID:
            raise ValueError("A judgment child requires the physical judge-v8 binding")
        return self


def judgment_business_key(seed: JudgmentAdmissionSeedV1) -> IntegrityDigest:
    """Name one judge child family: the reviewing owner, the position, the round and vote."""
    return judgment_family_key(seed.owner_run_id, seed.position_key, seed.round, seed.vote)


def judgment_family_key(
    owner_run_id: str, position_key: PositionKey, round_no: RoundNo, vote: int = 1,
) -> IntegrityDigest:
    """`judgment_business_key` without the member input the admission binds (C.13).

    A review (vote 1) keeps the key it had before confirmation votes existed; a
    confirmation vote's key also names the vote (C.8 #9).
    """
    key: dict[str, object] = {
        "kind": "judgment", "version": 1, "owner_run_id": owner_run_id,
        "position_key": list(position_key), "round": round_no,
    }
    if vote != 1:
        key["vote"] = vote
    return integrity_digest(key)


def judgment_target(seed: JudgmentAdmissionSeedV1) -> str:
    """The child's target id ``{owner_run_id}.{unit_no}.{position}.r{round}``, with
    ``v{vote}`` appended for a confirmation vote (C.13, C.8 #9)."""
    owner, unit, position = seed.position_key
    vote = "" if seed.vote == 1 else f"v{seed.vote}"
    return f"{owner}.{unit}.{position}.r{seed.round}{vote}"


def judgment_targets(position_key: PositionKey) -> tuple[str, ...]:
    """Every target a position's judge children can have under one reviewing owner (C.13):
    each round's review, settle rounds included, and its confirmation votes (C.8 #9)."""
    owner, unit, position = position_key
    return tuple(f"{owner}.{unit}.{position}.r{round_no}" + ("" if vote == 1 else f"v{vote}")
                 for round_no in range(1, MAX_ROUND + 1)
                 for vote in range(1, CONFIRMATION_VOTES + 1))


_TARGET: Final = re.compile(
    rf"[^.]+\.\d+\.\d+\.r[1-{MAX_ROUND}](?:v[2-{CONFIRMATION_VOTES}])?")


def is_judgment_target(target: str) -> bool:
    """Whether a Run target is a judge child's (``judgment_target``): a review's
    ``...r{round}`` or a confirmation vote's ``...r{round}v{vote}``."""
    return _TARGET.fullmatch(target) is not None


class JudgmentRunInputV1(JudgmentAdmissionSeedV1):
    """The immutable evaluate:judgment input for one position, round and vote (C.13).

    The snapshot is a locator, never copied input text; the store verifies its
    position, round, owner and member input digest before dispatch.
    """

    admission_digest: ContentDigest
    payload_digest: ContentDigest
    target_locator: ContentDigest | None = None
    run_kind: Literal["evaluate"]
    evaluation_kind: Literal["judgment"]
    business_key: Integrity
    snapshot_ref: SnapshotRef

    @model_validator(mode="after")
    def judgment_family(self) -> Self:
        seed = JudgmentAdmissionSeedV1.model_validate({
            name: getattr(self, name) for name in JudgmentAdmissionSeedV1.model_fields
        })
        if self.business_key != judgment_business_key(seed):
            raise ValueError("Judgment business key changed its owner, position, round or vote")
        return self


class JudgeOutputLocatorV1(StrictModel):
    """The scope's accepted judge output for one ``(binding, member input)`` (C.8 #1).

    Key ``judge_output:{scope}:{bind("judge-output-locator", [binding digest,
    member input digest])}``, written in the reply's transaction. Reuse reads it
    only while the output and result exist unscrubbed; privacy erase deletes it
    with them (C.13).
    """

    child_run_id: RunRef
    output_ref: OutputRef
    result_ref: ResultRef


class CheckedSourceV1(StrictModel):
    """Completion progress for one source output; never publication authorization.

    Pending positions may re-enter a later Add. Counts describe this completed
    source, without retaining retired review-policy fields or input text.
    """

    state: Literal["checked"]
    source_run_id: RunRef
    source_input_ref: Annotated[
        ArtifactId, RefSpec("artifact", ref_mode="historical_locator", fulfillment="none",
                            scope_rule="same_scope", artifact_kinds=("run_input", "manifest")),
    ]
    source_output_ref: OutputRef
    source_output_digest: Integrity
    checked_by_run_id: RunRef
    checked_result_ref: ResultRef
    source_positions: NonNegativeInt
    reviewed_positions: NonNegativeInt
    pending_positions: list[NonNegativeInt]
    selections_digest: ContentDigest
    policy_digest: Integrity

    @model_validator(mode="after")
    def complete_counts(self) -> Self:
        pending = self.pending_positions
        if (
            self.reviewed_positions > self.source_positions or pending != sorted(set(pending))
            or any(position >= self.source_positions for position in pending)
        ):
            raise ValueError("Checked-source counts or pending positions exceed the source")
        return self
