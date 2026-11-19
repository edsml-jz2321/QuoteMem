"""The durable D.3 forget Controls: the bound intent and the erasure fence plan.

``intent:{scope}:{principal}:forget:{binding}:{ulid}`` holds one prepared
selection (P0); the atomic admission (P1) consumes it in the transaction that
installs ``erasure:{scope}:{controller}``. The fence carries the frozen plan
inputs (INV W21): the selected Item, the stating spans S, the tombstones of
D, the post-bump epochs and the phase a restart resumes from. Parsing a value
authorizes nothing; the Writer-side owner (`store/forget_run.py`) rechecks
every row each batch reads.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self, cast

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from aml.core.entities import EntityRef, QuoteSpanV1
from aml.core.forget import MAX_FORGET_SOURCES, MAX_FORGET_TOMBSTONES, ForgetCountsV1
from aml.core.identity import ULID, ArtifactId, PrincipalId, ScopeId
from aml.core.technical_types import ContentDigest, NonNegativeInt, Timestamp
from aml.store.metadata import OpaqueIdSpec, StrictModel


def _array(value: object) -> object:
    """Accept JSON arrays without relaxing strict element validation."""
    return tuple(cast(list[object], value)) if isinstance(value, list) else value

MAX_FORGET_SPANS = 4_096
"""The forgotten span set S a fence may carry; a larger S is ``budget_exceeded`` before writes."""
MAX_FORGET_ENTITIES = 512
"""Entities whose anchoring the forgotten Item may have held (P2 erasure candidates)."""
MAX_FORGET_ARTIFACTS = 16_384
"""Retention-unit Artifacts (with their manifest parts) one forget deletes in P2."""
MAX_FORGET_CONTROLS = 4_096
"""Retention-unit Controls (``judge_output`` locators) one forget deletes in P2."""
MAX_FORGET_RECALLS = 16_384
"""Search deliveries (``recall`` rows) one forget deletes in P2."""
MAX_FORGET_CHAIN = 64
"""Items the selected one ended by ``correct`` or ``retract``, transitively, that one forget
takes with it (FG-2); a longer chain is ``budget_exceeded`` before writes."""
INTENT_TTL_MS = 10 * 60 * 1000
"""A prepared forget intent's lifetime; confirmation after it is ``intent_expired``."""

_Local = Annotated[ULID, OpaqueIdSpec("local")]


class ForgetIntentV1(StrictModel):
    """One prepared, bounded forget selection (D.3 P0).

    ``active`` until the atomic admission consumes it; the consumed value
    names its controller so a repeated confirm returns the original admission.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    state: Literal["active", "consumed"]
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    principal_id: Annotated[PrincipalId, OpaqueIdSpec("local")]
    session_token: ContentDigest
    memory_id: _Local
    expected_revision_id: _Local
    selection_digest: ContentDigest
    erase_epoch: NonNegativeInt
    issued_at: Timestamp
    expires_at: Timestamp
    controller_run_id: _Local | None = None

    @model_validator(mode="after")
    def consumed_names_controller(self) -> Self:
        if (self.state == "consumed") != (self.controller_run_id is not None):
            raise ValueError("Only a consumed intent names its controller")
        if self.expires_at <= self.issued_at:
            raise ValueError("A forget intent must expire after it is issued")
        return self


ErasurePhase = Literal["admitting", "admitted", "deleted", "purged", "scanned", "receipted"]
"""Fence installed (P1 step a); P1 effects committed; P2 Item deletion committed;
P2 retention records deleted; P3 scan bound; receipts written (P4 releases the fence)."""


class ErasureFenceV1(StrictModel):
    """The scope's single erasure fence and its frozen forget plan (D.3 P1-P4).

    Attributes:
        phase: Where a restart resumes (`ForgetOwner.resume`).
        controller_run_id / worker_run_id: The global controller and the
            scoped destructive worker admitted in P1.
        memory_id: The selected Item; FS1 proves it and its retention rows gone.
        chain_ids: The Items the selected one ended by ``correct`` or ``retract``,
            transitively, in id order (FG-2): a correction or retraction that changed
            the primary anchor created a new Item (C.9.4), so these are the selected
            memory's earlier versions. D, S and FS1 cover them like the selected Item.
        reworded_ids: The surviving Items whose head a forgotten correction reworded
            (C.9.4a), in id order: outside the chain, the versions they had before that
            rewording join D, and their head's own groups stay readable (OPR-2).
        intent_key: The consumed intent.
        spans: S, the stating spans: every span of every support group of every
            non-erased version, less the phrase a surviving readable naming memory
            of that version's primary entity quotes as its own support and less a
            bare name a neutral-type-only naming memory of a non-user entity quotes,
            and less the quote of a detail keeper (an owner- or user-anchored naming
            memory of an entity the version states a different detail of),
            unless that would leave a group with nothing withheld. Sensitive naming
            heads keep nothing on any path; a bare name or a detail keeper keeps
            nothing when any forgotten version has a sensitivity class
            (``plan_stating_spans``, O11).
        tombstone_ids: One tombstone per element of D (reused or created).
        created_tombstone_ids: The subset P1 created.
        entity_ids: Entities the Item's versions anchored (P2 erasure candidates).
        erase_epoch / tombstone_seq / scope_seq / commit_seq: Post-P1 values.
        counts: Accumulated effects; P3 recounts what it can prove.
        spans_done: How many spans of S the P1/P2 batches withheld.
        dependents_after: The last Item, in id order, whose dependent
            compensation committed; the next batch starts after it.
        expired_after: The last pending candidate, in id order, whose expiry
            committed; the next expiry batch starts after it.
        cancelled_owner_run_id: The live Add owner whose finalize lease P1
            cancelled (D.3 P1 #7), or None. P2 settles its cohort before the
            Item's own rows leave (D.5).
        trim_dependents: Whether P2 may keep a dependent readable on the readable
            quotes of one of its groups (D.7.3 #4a, ``trimmed_support``): P1 sets it
            when no forgotten version has a sensitivity class.
        artifacts / controls: The deleted positions' retention records,
            frozen with the deletion of their rows.
        recalls: The Search deliveries that returned a deleted Item or an
            Evidence row of S, frozen with the same deletion (D.3 P2); the
            fence bars every delivery of the scope from admission on.
            ``purged`` counts the Artifacts, then the Controls, then these
            deliveries the P2 batches deleted.
        scan_digest / prepare_ref / final_ref / index_ref: Bound by P3 and P4;
            ``index_ref`` is the controller's ``result_ref``.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    state: Literal["active"] = "active"
    phase: ErasurePhase
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    controller_run_id: _Local
    worker_run_id: _Local
    memory_id: _Local
    chain_ids: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_CHAIN), BeforeValidator(_array),
    ] = ()
    reworded_ids: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_CHAIN), BeforeValidator(_array),
    ] = ()
    intent_key: Annotated[str, Field(min_length=1, max_length=512)]
    spans: Annotated[
        tuple[QuoteSpanV1, ...], Field(max_length=MAX_FORGET_SPANS), BeforeValidator(_array),
    ]
    tombstone_ids: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_TOMBSTONES), BeforeValidator(_array),
    ]
    created_tombstone_ids: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_TOMBSTONES), BeforeValidator(_array),
    ]
    entity_ids: Annotated[
        tuple[Annotated[EntityRef, OpaqueIdSpec("local")], ...],
        Field(max_length=MAX_FORGET_ENTITIES), BeforeValidator(_array),
    ]
    erase_epoch: NonNegativeInt
    tombstone_seq: NonNegativeInt
    scope_seq: NonNegativeInt
    commit_seq: NonNegativeInt
    counts: ForgetCountsV1
    spans_done: NonNegativeInt = 0
    dependents_after: _Local | None = None
    expired_after: _Local | None = None
    cancelled_owner_run_id: _Local | None = None
    trim_dependents: bool = False
    artifacts: Annotated[
        tuple[Annotated[ArtifactId, OpaqueIdSpec("local")], ...],
        Field(max_length=MAX_FORGET_ARTIFACTS), BeforeValidator(_array),
    ] = ()
    controls: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=512)], ...],
        Field(max_length=MAX_FORGET_CONTROLS), BeforeValidator(_array),
    ] = ()
    recalls: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_RECALLS), BeforeValidator(_array),
    ] = ()
    purged: NonNegativeInt = 0
    scan_digest: ContentDigest | None = None
    prepare_ref: Annotated[str, OpaqueIdSpec("local")] | None = None
    final_ref: Annotated[str, OpaqueIdSpec("local")] | None = None
    index_ref: Annotated[str, OpaqueIdSpec("local")] | None = None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if not set(self.created_tombstone_ids) <= set(self.tombstone_ids):
            raise ValueError("Created tombstones must belong to the plan's tombstone set")
        if list(self.tombstone_ids) != sorted(set(self.tombstone_ids)):
            raise ValueError("Tombstones must be unique and ordered")
        if list(self.chain_ids) != sorted(set(self.chain_ids) - {self.memory_id}):
            raise ValueError("The chain must be unique, ordered and exclude the selected Item")
        if list(self.reworded_ids) != sorted(
                set(self.reworded_ids) - {self.memory_id, *self.chain_ids}):
            raise ValueError("Reworded Items must be unique, ordered and outside the chain")
        if self.spans_done > len(self.spans):
            raise ValueError("Withholding progress exceeds the frozen span set")
        records = len(self.artifacts) + len(self.controls) + len(self.recalls)
        if self.purged > records or (
            self.phase in {"purged", "scanned", "receipted"} and self.purged != records
        ):
            raise ValueError("Retention-record progress disagrees with the frozen records")
        if any(list(frozen) != sorted(set(frozen))
               for frozen in (self.artifacts, self.controls, self.recalls)):
            raise ValueError("Retention records must be unique and ordered")
        evidence = {span.evidence_id for span in self.spans}
        if len(evidence) > MAX_FORGET_SOURCES:
            raise ValueError("The forgotten span set cites too many Evidence rows")
        if self.phase in {"scanned", "receipted"} and self.scan_digest is None:
            raise ValueError("A scanned fence binds its final scan digest")
        if (self.phase == "receipted") != (self.index_ref is not None) or (
            (self.final_ref is None) != (self.index_ref is None)
        ) or (
            self.final_ref is not None and self.prepare_ref is None
        ):
            raise ValueError("Only a receipted fence names its receipt pair")
        return self

    @property
    def evidence_ids(self) -> frozenset[str]:
        """The Evidence rows S cites."""
        return frozenset(span.evidence_id for span in self.spans)
