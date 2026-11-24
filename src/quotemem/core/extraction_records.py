"""Content-free records of extraction admission and settlement.

``ExtractionAdmissionV2`` lives beside the verbatim reply in its ``llm_output``
and names, in reply coordinates, every change program admission made: masked
declarations, refused facts, dropped groups and anchors, masked fields and
in-reply merges. It is absent when admission changed nothing, and replay
recomputes it exactly (a different record is ``binding_mismatch``).

``DigestSettlementV1`` records how settlement settled one position
before review: rejected for an exact forgotten digest, unchanged because a
current memory already holds it, or a duplicate of an earlier position.

Neither record copies model or source text.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from quotemem.core.entities import EntityRef, PositionKey, QuoteSpanV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID, ArtifactId, ContentBindingDigest
from quotemem.core.statement import MAX_REPLY_FACTS, HexDigest
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

WholeOutputFault = Literal[
    "provider_schema_invalid", "truncated", "output_bound", "no_fresh_source",
]
"""The whole-output faults, the only reasons for 422 ``extraction_refused``."""

DeclarationRule = Literal[
    "declaration_invalid",
    "declaration_repeated",
    "name_unlocated",
    "name_shared",
    "reason_unknown",
    "candidates_dropped",
]
"""``name_unlocated`` keeps the declaration without its name: no source
writes that name, so it may be a transliteration or an invention, never the entity's
name. ``reason_unknown`` reads the reason as ``insufficient_context`` and
``candidates_dropped`` drops unissued or masked candidates and keeps the first 8.
Every other rule masks the declaration."""

GroupRule = Literal[
    "source_unknown", "quote_blank", "quote_bound", "quote_not_found",
    "quote_overlaps_withheld", "duplicate_span", "no_fresh_quote", "span_bound", "group_bound",
    "quote_question", "quote_unstated", "quote_request",
]
FactRule = Literal[
    "no_valid_group", "statement_invalid", "kind_invalid", "fact_bound", "secret",
    "sensitive_inference", "user_unattested", "fact_count_exceeded",
]
"""Fact rules: each refuses its fact. ``user_unattested`` refuses a fact that
refers to the user (the ``user`` anchor, the template's wording for the user or its
closed translations) when the unit renders no authenticated-self source or, for that
wording, no admitted group quotes an authenticated-self source."""
AnchorRule = Literal[
    "self_not_proven", "handle_unknown", "declaration_missing", "declaration_invalid",
    "duplicate", "anchor_bound", "secondary_unresolved", "anchor_conflict",
]
FieldName = Literal[
    "statement", "kind", "time", "naming", "revises", "corrects_text", "keywords", "sensitivity",
]
FieldRule = Literal[
    "time_inconsistent", "time_unlocated",
    "naming_bound", "naming_invalid", "naming_not_anchor", "naming_not_in_statement",
    "naming_unquoted", "naming_former_unknown", "naming_duplicate",
    "revises_unknown", "revises_duplicate", "revises_bound",
    "corrects_text_invalid", "keyword_invalid", "keyword_bound", "sensitivity_invalid",
    "secret", "alias_normalized",
]
"""Field rules. ``secret`` notes a bench fact whose credential forced
``restricted`` instead of refusing it. ``alias_normalized`` records each field changed
by the source-grounded alias naming rule, with no source or model text."""

Position = Annotated[int, Field(ge=0, lt=MAX_REPLY_FACTS)]
_Index = Annotated[int, Field(ge=0)]


class ExtractionRefused(QuoteMemError):
    """A whole-output fault of one unit: the owner answers 422 ``extraction_refused``.

    Attributes:
        fault: The fault.
        unit_no: The unit whose call the fault ended, once the extraction stage
            names it; None where the fault is raised below the stage
            (``extract``, ``admit``), which does not know it.
    """

    def __init__(
        self, fault: WholeOutputFault, message: str, *, unit_no: int | None = None,
    ) -> None:
        """Create the refusal with its fault, its unit when known and a payload-free text."""
        super().__init__("extraction_refused", message)
        self.fault: WholeOutputFault = fault
        self.unit_no: int | None = unit_no


class DeclarationNoteV1(StrictModel):
    """A declaration admission changed: its list, its reply index and the rule."""

    list: Literal["new", "unresolved"]
    index: _Index
    rule: DeclarationRule


class FactNoteV1(StrictModel):
    """A refused or overflowing fact position."""

    position: Position
    rule: FactRule


class GroupNoteV1(StrictModel):
    """A dropped support group: its index in the fact's reply groups."""

    position: Position
    group: _Index
    rule: GroupRule


class AnchorNoteV1(StrictModel):
    """A degraded or dropped anchor: its index in the fact's reply anchors."""

    position: Position
    anchor: _Index
    rule: AnchorRule


class FieldNoteV1(StrictModel):
    """A masked, dropped or changed field value; ``index`` locates a list item."""

    position: Position
    field: FieldName
    index: _Index | None
    rule: FieldRule


class MergeV1(StrictModel):
    """An in-reply duplicate and the earlier fact it merges into."""

    position: Position
    primary: Position

    @model_validator(mode="after")
    def earlier(self) -> Self:
        """Require the representative to come first."""
        if self.primary >= self.position:
            raise ValueError("A duplicate merges into an earlier fact")
        return self


class ExtractionAdmissionV2(StrictModel):
    """How program admission read one extract-v12 reply.

    Every list is in reply order: notes are sorted by position, then by their
    index, then by rule, and each key appears once.
    """

    reply_facts: Annotated[int, Field(ge=0, le=MAX_REPLY_FACTS)]
    declarations: list[DeclarationNoteV1]
    facts: list[FactNoteV1]
    groups: list[GroupNoteV1]
    anchors: list[AnchorNoteV1]
    fields: list[FieldNoteV1]
    merges: list[MergeV1]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Require reply order, unique keys and positions inside the reply."""
        keyed: list[list[tuple[int | str, ...]]] = [
            [(note.list, note.index, note.rule) for note in self.declarations],
            [(note.position,) for note in self.facts],
            [(note.position, note.group) for note in self.groups],
            [(note.position, note.anchor, note.rule) for note in self.anchors],
            [(note.position, note.field, -1 if note.index is None else note.index, note.rule)
             for note in self.fields],
            [(note.position,) for note in self.merges],
        ]
        if any(keys != sorted(set(keys)) for keys in keyed):
            raise ValueError("Admission notes are in reply order and name each item once")
        positions = [
            note.position for notes in (self.facts, self.groups, self.anchors, self.fields)
            for note in notes
        ] + [note.position for note in self.merges] + [note.primary for note in self.merges]
        if any(position >= self.reply_facts for position in positions):
            raise ValueError("An admission note names a position outside the reply")
        return self


SettlementCode = Literal[
    "denied_by_forget", "restatement_satisfied", "merged_support", "duplicate_in_add",
    "support_budget_exceeded",
]


def snapshot_locator(target_kind: str) -> RefSpec:
    """Declare a row of the owner's scope that an extraction record read at its pin.

    The row exists when the record is written, and the record never retains it:
    settlement's head and tombstone, and a ``revises`` version.

    Args:
        target_kind (str): The row's kind.

    Returns:
        RefSpec: A same-scope snapshot locator that is no GC root.
    """
    return RefSpec(target_kind, ref_mode="snapshot_locator", scope_rule="same_scope",
                   gc_root=False)


class DigestSettlementV1(StrictModel):
    """How settlement settled one position before review.

    Attributes:
        position_key: ``(owner_run_id, unit_no, position)``.
        code: ``denied_by_forget``, ``restatement_satisfied``,
            ``merged_support`` (new support for an unreadable head),
            ``duplicate_in_add`` (the final code mirrors the
            representative's at commit) or ``support_budget_exceeded`` (a
            duplicate whose groups do not fit; it does not mirror).
        memory_id, revision_id: The head that holds a restatement or that a
            ``merged_support`` position restores, or the memory a duplicate's settled
            representative holds.
        tombstone_id: The tombstone that rejects a position.
        representative: The position a duplicate joins or mirrors; never itself
            a duplicate.
        read_view_commit_seq, tombstone_seq: The pin settlement read.

    The three row identities are scoped snapshot locators of the rows read at
    the pin: they exist when the record is written and are never retained by it.
    """

    position_key: PositionKey
    code: SettlementCode
    memory_id: Annotated[ULID, snapshot_locator("item")] | None
    revision_id: Annotated[ULID, snapshot_locator("revision")] | None
    tombstone_id: Annotated[ULID, snapshot_locator("tombstone")] | None
    representative: PositionKey | None
    read_view_commit_seq: Annotated[int, Field(ge=0)]
    tombstone_seq: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def complete(self) -> Self:
        """Require the references each code needs, and only those."""
        if (self.code == "denied_by_forget") != (self.tombstone_id is not None):
            raise ValueError("Exactly a forgotten digest names its tombstone")
        if (self.code in ("duplicate_in_add", "support_budget_exceeded")) != (
            self.representative is not None
        ):
            raise ValueError("Exactly a duplicate names its representative")
        if self.code in ("restatement_satisfied", "merged_support") and (
            self.memory_id is None or self.revision_id is None
        ):
            raise ValueError("A restatement names the head that holds it")
        if self.code == "denied_by_forget" and self.memory_id is not None:
            raise ValueError("A forgotten digest names no memory")
        return self


class ErasedReplyV1(StrictModel):
    """An erased extraction call's identity and content-free admission accounting."""

    input_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
    extraction_ref: Annotated[ArtifactId, OpaqueIdSpec("local")]
    call_no: Literal[0, 1]
    admission: ExtractionAdmissionV2 | None


class ErasedQuoteSpanV1(QuoteSpanV1):
    """Original source coordinates that neither retain nor reconstruct source content."""

    evidence_id: Annotated[ULID, snapshot_locator("evidence")]


class ErasedProposalV1(StrictModel):
    """An erased proposal's position and original source coordinates, without its text.

    Artifact ids are opaque locators: the erasure deliberately deletes their bodies.
    The retained coordinates attribute completion to the original request.
    """

    position_key: PositionKey
    proposal_ref: Annotated[ArtifactId, OpaqueIdSpec("local")]
    extraction_ref: Annotated[ArtifactId, OpaqueIdSpec("local")]
    statement_digest: Annotated[HexDigest, OpaqueIdSpec("digest")]
    entity_id: Annotated[EntityRef, OpaqueIdSpec("local")] | None
    quote_spans: list[ErasedQuoteSpanV1]


class ErasedCandidateExpiryV1(StrictModel):
    """One derived expiry's terminal post-image, stored atomically with its candidate."""

    candidate_id: Annotated[ULID, snapshot_locator("candidate")]
    position_key: PositionKey
    after_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]


class ErasedExtractionV1(StrictModel):
    """Content-free extraction accounting persisted before privacy deletes its bodies.

    Each immutable record describes one erasure's selected records. Repeated
    erasures may overlap; readers require overlapping entries to agree.
    """

    owner_run_id: Annotated[ULID, OpaqueIdSpec("local")]
    erased_extraction: list[ErasedReplyV1]
    proposals: list[ErasedProposalV1]
    expired_candidates: list[ErasedCandidateExpiryV1] = Field(
        default_factory=list[ErasedCandidateExpiryV1], exclude_if=lambda value: not value,
    )

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Require ordered, unique call and proposal identities of this owner."""
        calls = [(reply.input_digest, reply.call_no) for reply in self.erased_extraction]
        keys = [proposal.position_key for proposal in self.proposals]
        candidates = [expiry.candidate_id for expiry in self.expired_candidates]
        if calls != sorted(set(calls)) or keys != sorted(set(keys)):
            raise ValueError("Erasure metadata is ordered and names each identity once")
        if candidates != sorted(set(candidates)):
            raise ValueError("Derived expiries are sorted and name each candidate once")
        if any(key[0] != self.owner_run_id for key in (
            *keys, *(expiry.position_key for expiry in self.expired_candidates),
        )):
            raise ValueError("Erasure metadata names only its owner's positions")
        return self
