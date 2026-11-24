"""``ExtractContextV2``: the frozen context of one extraction unit.

The context is persisted (``decision_set:extraction_context``) before the
extraction call and recovery never recomputes it. It holds locators, not text:
the unit's fresh parts, the previous-batch and tail locators, the withheld
ranges and turn labels in force at selection, the catalog, the bindings and the
context memories, with the pin's stamps. Rendering those rows again with the
recorded withheld ranges and labels reproduces the request byte for byte, which
``input_digest`` (the fresh sources) and ``context_digest`` (everything else
the request shows) bind.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from quotemem.core.identity import ULID, ContentBindingDigest, IntegrityDigest
from quotemem.core.statement import (
    MAX_CATALOG_ENTITIES,
    MAX_CONTEXT_MEMORIES,
    MAX_EARLIER_VERSIONS,
    MAX_MESSAGE_BYTES,
    HexDigest,
)
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

_Offset = Annotated[int, Field(ge=0, le=MAX_MESSAGE_BYTES)]
_Count = Annotated[int, Field(ge=0)]
UnitNo = Annotated[int, Field(ge=0, le=65_535)]
_EvidenceLocator = Annotated[
    ULID, RefSpec("evidence", ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False),
]
_ExternalId = Annotated[str, Field(min_length=1), OpaqueIdSpec("external")]


class ByteRangeV1(StrictModel):
    """A non-empty half-open range of original-message bytes."""

    byte_start: _Offset
    byte_end: _Offset

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        """Require a non-empty half-open range."""
        if self.byte_end <= self.byte_start:
            raise ValueError("A byte range is non-empty and half-open")
        return self


class MemberBindingV1(StrictModel):
    """One fresh Evidence part of the unit, in unit order."""

    evidence_id: _EvidenceLocator
    request_id: _ExternalId
    ordinal: _Count
    part_no: _Count


class ContextUnitV1(StrictModel):
    """The unit the context belongs to and its fresh parts."""

    owner_run_id: Annotated[
        ULID, RefSpec(
            "run", ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False,
            target_forget_action="retain_locator",
        ),
    ]
    session_id: _ExternalId
    request_id: _ExternalId
    unit_no: UnitNo
    member_bindings: Annotated[list[MemberBindingV1], Field(min_length=1)]


class PreviousBatchV1(StrictModel):
    """The previous batch the context shows read-only.

    ``first_kept_ordinal`` is the oldest message kept; ``truncated`` tells that
    older whole messages of the batch were dropped to fit the bounds.
    """

    session_id: _ExternalId
    request_id: _ExternalId
    first_kept_ordinal: _Count
    truncated: bool


class ContextSpanV1(ByteRangeV1):
    """A rendered byte range of one Evidence part (a tail source's part)."""

    evidence_id: _EvidenceLocator


class WithheldRangesV1(StrictModel):
    """The withheld ranges of one rendered Evidence part at selection."""

    evidence_id: _EvidenceLocator
    ranges: Annotated[list[ByteRangeV1], Field(min_length=1)]


class LabelDecisionV1(ByteRangeV1):
    """One turn label in force at selection: its range and state."""

    state: Literal["speaker", "ambiguous"]


class MessageLabelsV1(StrictModel):
    """The turn labels of one rendered message."""

    session_id: _ExternalId
    request_id: _ExternalId
    ordinal: _Count
    labels: Annotated[list[LabelDecisionV1], Field(min_length=1)]


class CatalogRefV1(StrictModel):
    """One catalog entry by reference: its handle and the rows it was built from."""

    handle: Annotated[str, Field(pattern=r"^(?:user|E[1-9][0-9]*)$")]
    kind: Literal["user", "participant", "entity"]
    # The virtual user has no Entity row; catalog names carry the row locators.
    entity_id: Annotated[
        str, Field(min_length=1, max_length=64), OpaqueIdSpec("local"),
    ] | None
    participant_key: Annotated[HexDigest, OpaqueIdSpec("digest")] | None
    name_row_ids: Annotated[
        list[str], RefSpec(
            "entity_name", cardinality="many", ref_mode="snapshot_locator",
            scope_rule="same_scope", gc_root=False,
        ),
    ]


class ContextBindingV1(ByteRangeV1):
    """One SDK binding the request shows: its bytes and the handle it binds."""

    evidence_id: _EvidenceLocator
    handle: Annotated[str, Field(pattern=r"^E[1-9][0-9]*$")]


class ContextMemoryRefV1(StrictModel):
    """One context memory or earlier version by reference.

    ``of_handle`` names the context memory an ``earlier`` version belongs to.
    """

    handle: Annotated[str, Field(pattern=r"^M[1-9][0-9]*$")]
    memory_id: Annotated[
        ULID, RefSpec("item", ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False),
    ]
    revision_id: Annotated[
        ULID, RefSpec(
            "revision", ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False,
        ),
    ]
    content_hash: Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
    origin: Literal["relevance", "reserved", "earlier"]
    of_handle: Annotated[str, Field(pattern=r"^M[1-9][0-9]*$")] | None

    @model_validator(mode="after")
    def earlier_of(self) -> Self:
        """Require ``of_handle`` exactly for an earlier version."""
        if (self.origin == "earlier") != (self.of_handle is not None):
            raise ValueError("Only an earlier version names the memory it belongs to")
        return self


class RollupRefV1(StrictModel):
    """The session's newest rollup that the unit's rollup call reads as its prior.

    A reference, not the statement: a later forget or privacy erasure of the rollup
    reaches it, and reconstruction reads the revision again by id.
    """

    memory_id: Annotated[
        ULID, RefSpec("item", ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False),
    ]
    revision_id: Annotated[
        ULID, RefSpec(
            "revision", ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False,
        ),
    ]


class ExtractContextV2(StrictModel):
    """The persisted context of one extraction unit.

    Attributes:
        schema_version: Always 2.
        unit: The unit and its fresh parts in order.
        previous: The previous batch's locator, or None.
        tail: The tail sources' rendered part ranges, in order.
        withheld: Withheld ranges of every rendered part that had any.
        labels: Turn labels of every rendered message that had any.
        catalog: The catalog in handle order; ``catalog_truncated`` tells that
            entities were dropped.
        bindings: The SDK bindings the request shows.
        memories: Context memories (``M1``..``M40``) and their earlier versions
            (``M41`` upwards), in handle order.
        read_view_commit_seq, tombstone_seq, erase_epoch, consent_epoch: The
            owner pin's stamps.
        render_config_digest: The render configuration the sources were cut and
            rendered under.
        input_digest: ``bind("extract-input", ...)`` over the fresh sources.
        context_digest: ``bind("extract-context", ...)`` over every other part
            of the request.
        rollup: The prior rollup the session rollup call reads (``core/session_rollup.py``),
            or None, then left out, so a context without one keeps its bytes.
    """

    schema_version: Literal[2]
    unit: ContextUnitV1
    previous: PreviousBatchV1 | None
    tail: list[ContextSpanV1]
    withheld: list[WithheldRangesV1]
    labels: list[MessageLabelsV1]
    catalog: Annotated[list[CatalogRefV1], Field(max_length=MAX_CATALOG_ENTITIES + 1)]
    catalog_truncated: bool
    bindings: list[ContextBindingV1]
    memories: Annotated[
        list[ContextMemoryRefV1], Field(max_length=MAX_CONTEXT_MEMORIES + MAX_EARLIER_VERSIONS),
    ]
    read_view_commit_seq: _Count
    tombstone_seq: _Count
    erase_epoch: _Count
    consent_epoch: _Count
    render_config_digest: Annotated[IntegrityDigest, OpaqueIdSpec("digest")]
    input_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
    context_digest: Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
    rollup: RollupRefV1 | None = Field(default=None, exclude_if=lambda value: value is None)
