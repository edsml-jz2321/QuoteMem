"""The persisted per-position record of extraction, which extraction hands to review.

An Add owner writes one ``ProposalV2`` per admitted position, create-once,
after the embedding of its unsettled positions and before the settlement
transaction, whose candidate rows reference it. Positions
that settlement settles carry null embedding references; every other position,
an unresolved primary included, points at its statement vector and, when it has
a ``corrects_text``, its cue vector in the owner's ``StatementEmbeddingV1``
batches. Re-entry and review rebuild members from it; no product
read path loads it.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, Self

from pydantic import Field, field_validator, model_validator

from quotemem.core.entities import (
    EntityChoiceV1,
    EntityRef,
    MemoryKind,
    NameKind,
    Origin,
    PositionKey,
    QuoteSpanV1,
    SensitivityBasis,
)
from quotemem.core.extraction_policy import DurabilityV1
from quotemem.core.extraction_records import snapshot_locator
from quotemem.core.identity import ULID, ArtifactId
from quotemem.core.semantic import (
    AnchorV1,
    HistoricalTimestamp,
    SecurityLevel,
    TimeExpressionV1,
    TimeGranularity,
    TrustTier,
    UnitFloat,
    validate_support_groups,
)
from quotemem.core.semantic_embedding import MAX_EMBEDDING_BATCH
from quotemem.core.sensitivity import SensitivityClass
from quotemem.core.statement import (
    MAX_ANCHORS,
    MAX_CORRECTS_TEXT_CHARS,
    MAX_GROUP_SPANS,
    MAX_KEYWORDS,
    MAX_MEMORY_SPANS,
    MAX_NAMING,
    MAX_REFERENT_CANDIDATES,
    MAX_REVISES,
    MAX_SUPPORT_GROUPS,
    HexDigest,
    canonical_statement,
    is_name,
    name_norm,
    statement_digest,
    token_contained,
)
from quotemem.core.technical_types import LocalId
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

ACTIONABILITY: Final = 0.5
"""The actionability input of importance, the same for every memory."""

Concern = Literal[
    "new_name_matches_existing", "new_entity_unnamed", "time_unlocated", "naming_masked",
    "catalog_truncated",
]
"""The concerns a position carries to review: settlement raises the first two, admission
the last three."""


class ImportanceInputsV1(StrictModel):
    """The importance inputs of a position; resolution adds novelty."""

    durability: UnitFloat
    actionability: UnitFloat


def importance_inputs(kind: MemoryKind, *, naming: bool,
                      durability: DurabilityV1) -> ImportanceInputsV1:
    """Return a position's importance inputs under the owner's frozen policy.

    Args:
        kind (MemoryKind): ``state`` or ``event``.
        naming (bool): Whether the position is a naming memory (it kept a naming
            annotation).
        durability (DurabilityV1): The frozen policy's durability values.

    Returns:
        ImportanceInputsV1: A naming memory's durability, else its kind's, and
            actionability 0.5.
    """
    by_kind = {"state": durability.state, "event": durability.event}
    return ImportanceInputsV1(
        durability=durability.naming if naming else by_kind[kind], actionability=ACTIONABILITY,
    )


class ProposalNamingV1(StrictModel):
    """A naming annotation kept by admission: the anchor it names (by slot) and the name."""

    anchor_index: Annotated[int, Field(ge=0, lt=MAX_ANCHORS)]
    name: str
    name_kind: NameKind

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        """Require a valid name."""
        if not is_name(value):
            raise ValueError("A name has 1..128 characters and a bound length of 1..24")
        return value


class RevisedVersionV1(StrictModel):
    """A memory a ``revises`` handle named and the version it showed."""

    memory_id: Annotated[ULID, snapshot_locator("item")]
    revision_id: Annotated[ULID, snapshot_locator("revision")]


class CutSpanV1(QuoteSpanV1):
    """Bytes of a forgotten name settlement cut a position's quotes around: a locator in the
    position's scope that its proposal only withholds, never reads or retains."""

    evidence_id: Annotated[ULID, snapshot_locator("evidence")]


class EmbeddingUseV1(StrictModel):
    """One entry of an owner's ``StatementEmbeddingV1`` batch: ``entries[index]`` of ``ref``."""

    ref: Annotated[
        ArtifactId,
        RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("semantic_embedding",)),
    ]
    index: Annotated[int, Field(ge=0, lt=MAX_EMBEDDING_BATCH)]


class ProposalV2(StrictModel):
    """One admitted position as review consumes it.

    Its fields are admission's after settlement: a duplicate's
    representative carries the joined groups and the united sensitivity of its
    duplicates, and every position the concerns settlement raised. Quote spans
    are in their first appearance over the admitted groups in reply order, and
    the groups are canonical: sorted, unique and covering every span.
    """

    schema_version: Literal[2]
    position_key: PositionKey
    extraction_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("llm_output",)),
    ]
    context_ref: Annotated[
        ArtifactId, RefSpec("artifact", scope_rule="same_scope", artifact_kinds=("decision_set",)),
    ]
    statement: str
    statement_digest: Annotated[HexDigest, OpaqueIdSpec("digest")]
    kind: MemoryKind
    anchors: Annotated[list[EntityChoiceV1], Field(min_length=1, max_length=MAX_ANCHORS)]
    naming: Annotated[list[ProposalNamingV1], Field(max_length=MAX_NAMING)]
    quote_spans: Annotated[list[QuoteSpanV1], Field(min_length=1, max_length=MAX_MEMORY_SPANS)]
    support_groups: Annotated[list[list[int]], Field(min_length=1, max_length=MAX_SUPPORT_GROUPS)]
    time_expr: TimeExpressionV1
    valid_from: HistoricalTimestamp | None
    valid_to: HistoricalTimestamp | None
    time_granularity: TimeGranularity
    anchored_from: AnchorV1
    keywords: Annotated[list[str], Field(max_length=MAX_KEYWORDS)]
    importance_inputs: ImportanceInputsV1
    origin: Origin
    trust_tier: TrustTier
    security_level: SecurityLevel
    sensitivity_classes: list[SensitivityClass]
    sensitivity_basis: list[SensitivityBasis]
    revises: Annotated[list[RevisedVersionV1], Field(max_length=MAX_REVISES)]
    corrects_text: Annotated[str, Field(min_length=1, max_length=MAX_CORRECTS_TEXT_CHARS)] | None
    concerns: list[Concern]
    collision_entity_ids: Annotated[
        list[EntityRef], Field(max_length=MAX_REFERENT_CANDIDATES), OpaqueIdSpec("local"),
    ]
    statement_embedding: EmbeddingUseV1 | None
    cue_embedding: EmbeddingUseV1 | None
    embedding_space: LocalId
    pipeline_version: LocalId
    cut: Annotated[list[CutSpanV1], Field(exclude_if=lambda value: not value)] = []
    """What withholds the forgotten names settlement cut this position's quotes around
    (``statement_settlement.Settlement.cut``), sorted and unique. The owner's withholding
    transaction withholds it before review, whatever the review then decides. Empty, and
    left out of the body, for a position that met no forgotten name."""

    @model_validator(mode="after")
    def check_proposal(self) -> Self:
        """Apply the value rules of the embedding, token and admission contracts that the record
        itself can show."""
        if canonical_statement(self.statement) != self.statement or (
            self.statement_digest != statement_digest(self.statement)
        ):
            raise ValueError("A proposal stores its canonical statement and that one's digest")
        cut = [(span.evidence_id, span.byte_start, span.byte_end) for span in self.cut]
        if cut != sorted(set(cut)):
            raise ValueError("The spans a proposal withholds are sorted and unique")
        if any(choice.kind == "unresolved" for choice in self.anchors[1:]):
            raise ValueError("Only the primary anchor may be unresolved")
        self._check_naming()
        self._check_support()
        self._check_time()
        if self.sensitivity_classes != sorted(set(self.sensitivity_classes)) or (
            self.sensitivity_basis != sorted(set(self.sensitivity_basis))
        ):
            raise ValueError("Sensitivity classes and basis are sorted and unique")
        if bool(self.sensitivity_classes) != bool(self.sensitivity_basis):
            raise ValueError("sensitivity_basis is non-empty exactly when classes are")
        revised = [(item.memory_id, item.revision_id) for item in self.revises]
        if len(set(revised)) != len(revised):
            raise ValueError("revises names each version once")
        if self.concerns != sorted(set(self.concerns)) or (
            self.collision_entity_ids != sorted(set(self.collision_entity_ids))
        ):
            raise ValueError("Concerns and collision entities are sorted and unique")
        if bool(self.collision_entity_ids) != ("new_name_matches_existing" in self.concerns):
            raise ValueError("Collision entities exist exactly with new_name_matches_existing")
        if self.cue_embedding is not None and (
            self.corrects_text is None or self.statement_embedding is None
        ):
            raise ValueError("A cue vector belongs to an embedded position with corrects_text")
        return self

    def _check_naming(self) -> None:
        seen: set[tuple[int, str, str]] = set()
        for naming in self.naming:
            if naming.anchor_index >= len(self.anchors) or (
                self.anchors[naming.anchor_index].kind == "unresolved"
            ):
                raise ValueError("A naming annotation names a resolved anchor")
            if not token_contained(naming.name, self.statement):
                raise ValueError("A naming annotation's name is token-contained in the statement")
            key = (naming.anchor_index, name_norm(naming.name), naming.name_kind)
            if key in seen:
                raise ValueError("Naming annotations are unique by anchor, name_norm and kind")
            seen.add(key)

    def _check_support(self) -> None:
        spans = [(span.evidence_id, span.byte_start, span.byte_end) for span in self.quote_spans]
        if len(set(spans)) != len(spans):
            raise ValueError("Quote spans are unique")
        if any(len(group) > MAX_GROUP_SPANS for group in self.support_groups):
            raise ValueError(f"A support group locates at most {MAX_GROUP_SPANS} spans")
        validate_support_groups(self.support_groups, len(spans))

    def _check_time(self) -> None:
        bounds = (self.valid_from, self.valid_to)
        if self.time_expr.kind == "none" or self.anchored_from.kind == "not_applicable":
            if (self.time_expr.kind, self.anchored_from.kind, self.time_granularity) != (
                "none", "not_applicable", "unanchored",
            ) or bounds != (None, None):
                raise ValueError("No time expression means not_applicable, unanchored, no bounds")
        elif self.anchored_from.kind == "unanchored":
            if bounds != (None, None) or self.time_granularity != "unanchored":
                raise ValueError("An unanchored expression has no bounds and no granularity")
        elif bounds == (None, None) or self.time_granularity == "unanchored":
            raise ValueError("A resolved anchor needs a bound and a granularity")
        if self.kind == "state" and self.valid_to is not None:
            raise ValueError("Only an event carries valid_to")
        if self.valid_from is not None and self.valid_to is not None and (
            self.valid_from >= self.valid_to
        ):
            raise ValueError("valid_from precedes valid_to")
