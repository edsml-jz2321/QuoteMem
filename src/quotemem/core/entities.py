"""Pure types of the natural-language memory model: closed enums, entity choices and stored values.

These are the HeavenBase-free types of the twelve memory entities
that every pure stage, wire and row builds on:

- the closed enums, written as ``Literal`` types;
- ``EntityRef`` and ``EntityChoiceV1``, the entity references of the entity
  table and of ``ProposalV2``;
- the Json values that rows embed: ``QuoteSpanV1``, ``WithheldSpanV1``
  and ``EntityBindingV1``, ``NamingAnnotationV1`` and
  ``MemorySnapshotV2`` with its erased form.

Every model is a frozen ``StrictModel``: exact types, no undeclared fields,
and validation of every value rule the contract states for the value itself.
Rules that need other rows (an anchor that must be an active entity, a span
that must lie inside its Evidence part) belong to the Writer. Identity fields
carry the registry metadata the schema inventory and the Writer's reference
checks read: an Evidence span, an SDK binding and the admitting verdict are
references; an ``EntityRef`` is a local identity, because ``user`` has no row.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final, Literal, Self, cast

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from quotemem.core.identity import ULID, ArtifactId
from quotemem.core.semantic import (
    AnchorV1,
    EmbeddingVector,
    HistoricalTimestamp,
    SecurityLevel,
    SystemTimestamp,
    TimeExpressionV1,
    TimeGranularity,
    TrustTier,
    UnitFloat,
    validate_support_groups,
)
from quotemem.core.sensitivity import SensitivityClass
from quotemem.core.statement import (
    ERASED_DIGEST,
    MAX_ABOUT_ENTITIES,
    MAX_DESCRIPTOR_CHARS,
    MAX_GROUP_SPANS,
    MAX_KEYWORD_CHARS,
    MAX_KEYWORDS,
    MAX_MEMORY_SPANS,
    MAX_MENTION_CHARS,
    MAX_MESSAGE_BYTES,
    MAX_NAME_CHARS,
    MAX_NAME_TOKENS,
    MAX_NAMING,
    MAX_REFERENT_CANDIDATES,
    MAX_REPLY_FACTS,
    MAX_STATEMENT_CHARS,
    MAX_SUPPORT_GROUPS,
    MAX_UNIT_NO,
    HexDigest,
    canonical_entity_type,
    canonical_statement,
    is_name,
    name_norm,
    statement_digest,
    token_contained,
)
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

MemoryLifecycle = Literal["active", "superseded", "corrected", "retracted", "historical", "erased"]
ActivationMode = Literal["contextual", "on_demand"]
MemoryKind = Literal["state", "event"]
Origin = Literal[
    "agent_inference", "tool_output", "external_document", "extraction_supplement",
    "session_rollup", "implied_health",
]
"""Where a memory's statement came from: from the extraction call over its cited
Evidence, whose kinds name the first three, from the extraction supplement call
(``core/extraction_supplement.py``), whose overview and assistant's-work facts Add
context selection leaves out and Search keeps from crowding ordinary evidence and whose
implied health conditions (``implied_health``) Add context leaves out and Search reads as
ordinary memories, or from the session rollup (``core/session_rollup.py``), which Add
context and every Search corpus leave out and only summary and ordering reads deliver,
one per session."""
RevisionOp = Literal["create", "merge", "revise", "requote", "erase"]
RevisionReason = Literal[
    "independent", "contradiction", "external_correction", "update", "correction",
    "retraction", "same_support", "supplement", "historical_correction",
    "privacy_requote", "privacy_erase", "rewording",
]
LinkType = Literal["supersedes", "contradicts", "about_entity", "corrects_history"]
SupersedeRelation = Literal["update", "correct", "retract"]
EffectiveBasis = Literal["valid_from", "source_observed_at"]
CandidateStatus = Literal["pending", "published", "unchanged", "rejected", "expired"]
ReasonCode = Literal[
    "created", "created_with_contradiction", "external_correction",
    "updated", "corrected", "retracted",
    "history_corrected",
    "supplemented", "merged_support",
    "redundant", "restatement_satisfied", "duplicate_in_add", "superseded_in_add",
    "restates_retired_version",
    "denied_by_forget",
    "unsupported_by_source", "not_atomic", "source_fidelity_incomplete",
    "unsupported_sensitive_inference",
    "secret_rejected", "extraction_item_invalid",
    "evidence_invalidated",
    "semantic_evidence_insufficient", "source_fidelity_uncertain",
    "identity_unresolved",
    "relation_unclear", "correction_target_unresolved", "operation_incomplete",
    "target_changed", "review_unavailable",
    "review_unavailable_repeated",
    "target_changed_repeated", "older_than_target", "review_capacity_exceeded",
    "support_budget_exceeded",
    "extraction_capacity_exceeded",
]
"""The 36 disposition codes of the outcome table, in table order."""
Rejudge = Literal["no", "yes", "explicit"]
EntityState = Literal["active", "erased"]
NameKind = Literal["name", "alias", "former"]
WithheldReason = Literal["forgotten", "restated_forgotten"]
EntityChoiceKind = Literal["user", "participant", "existing", "new", "unresolved"]
SensitivityBasis = Literal["explicit_source", "policy_classification"]

# Copied value sets of the Evidence row, named for the pure types.
EvidenceKind = Literal[
    "user_message", "assistant_message", "system_message", "tool_result",
    "document", "file", "external", "import",
]
SourceRole = Literal["user", "assistant", "tool", "system"]
EvidenceState = Literal["active", "invalidated", "expired", "erased"]
RouteState = Literal["pending", "indexed"]

REVISION_REASONS: Final[Mapping[RevisionOp, frozenset[RevisionReason]]] = MappingProxyType({
    "create": frozenset({
        "independent", "contradiction", "external_correction", "update", "correction",
        "retraction", "historical_correction",
    }),
    "merge": frozenset({"same_support"}),
    "revise": frozenset({"supplement", "correction", "retraction", "rewording"}),
    "requote": frozenset({"privacy_requote"}),
    "erase": frozenset({"privacy_erase"}),
})
"""The reason codes each revision op may carry."""

USER: Final = "user"
"""The virtual entity of the authenticated user; it has no entity row."""

EntityRef = Annotated[
    str, StringConstraints(pattern=r"^(?:user|[0-7][0-9A-HJKMNP-TV-Z]{25})$"),
]
"""``"user"`` or an entity ULID; physically ``ShortText(64)``."""

LocalKey = Annotated[str, StringConstraints(pattern=r"^N(?:[1-9]|[12][0-9]|3[0-2])$")]
"""A new-entity declaration handle of one extraction reply, ``N1``..``N32``."""


def _json_tuple(value: object) -> object:
    """Read a JSON array as the tuple it encodes, without coercing its members."""
    return tuple(cast(list[object], value)) if isinstance(value, list) else value


PositionKey = Annotated[
    tuple[
        ULID,
        Annotated[int, Field(ge=0, le=MAX_UNIT_NO)],
        Annotated[int, Field(ge=0, lt=MAX_REPLY_FACTS)],
    ],
    BeforeValidator(_json_tuple),
]
"""One admitted position, ``(owner_run_id, unit_no, position)``.

The owner is a ULID, ``unit_no`` is 0..65535 and ``position`` 0..511. A stored
JSON array reads back as the same tuple; its members are never coerced.
"""

_Offset = Annotated[int, Field(ge=0, le=MAX_MESSAGE_BYTES)]
_NAME_RULE = f"A name has 1..{MAX_NAME_CHARS} characters and a bound length of 1..{MAX_NAME_TOKENS}"


def _check_range(byte_start: int, byte_end: int) -> None:
    if byte_end <= byte_start:
        raise ValueError("A byte range must be non-empty and half-open")


class QuoteSpanV1(StrictModel):
    """Half-open original-message UTF-8 coordinates inside one Evidence part.

    Offsets count bytes of the unsplit message, as Evidence ``byte_start`` and
    ``byte_end`` do. The Writer checks that the span lies inside its part and
    on character boundaries.
    """

    model_config = ConfigDict(frozen=True)
    evidence_id: Annotated[ULID, RefSpec("evidence", scope_rule="same_scope")]
    byte_start: _Offset
    byte_end: _Offset

    @model_validator(mode="after")
    def check_range(self) -> Self:
        """Require ``0 <= byte_start < byte_end <= 1,048,576``."""
        _check_range(self.byte_start, self.byte_end)
        return self


class ForgetRemainderV1(StrictModel):
    """The readable rest of an Evidence part a complete forget withheld part of (``agent``
    mode), as the interactive Search ranks it: the vector of its readable bytes alone
    (``core.forget_remainder.remainder_text``), embedded once by the forget's run.

    The part's stored vector and terms encode every byte, so a part with a withheld range is
    never ranked by them (``p_evr``); this one encodes none of its withheld bytes. It holds
    for exactly the withheld ``ranges`` it was embedded under: a forget withholding more of
    the part drops it in the same transaction, and erasing the part drops it with the part's
    withheld ranges.

    Attributes:
        ranges: The part's withheld ranges, sorted, when its readable bytes were embedded.
        embedding: The vector of those readable bytes.
        embedding_space: The space of ``embedding``, the part's own.
    """

    model_config = ConfigDict(frozen=True)
    ranges: Annotated[list[Annotated[list[_Offset], Field(min_length=2, max_length=2)]],
                      Field(min_length=1)]
    embedding: EmbeddingVector
    embedding_space: Annotated[str, Field(min_length=1, max_length=256)]


class WithheldSpanV1(StrictModel):
    """One withheld byte range of an Evidence part.

    Offsets use the same message coordinates as `QuoteSpanV1`. A span
    intersecting a withheld range is unreadable (``p_span``). The part's last withheld
    range may carry its readable rest's vector (``remainder``, ``ForgetRemainderV1``).
    """

    model_config = ConfigDict(frozen=True)
    byte_start: _Offset
    byte_end: _Offset
    reason: WithheldReason
    run_id: Annotated[ULID, RefSpec("run", ref_mode="historical_locator", fulfillment="none")]
    remainder: ForgetRemainderV1 | None = Field(
        default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def check_range(self) -> Self:
        """Require ``0 <= byte_start < byte_end <= 1,048,576``."""
        _check_range(self.byte_start, self.byte_end)
        return self


class EntityBindingV1(StrictModel):
    """An SDK binding of message bytes to one existing entity."""

    model_config = ConfigDict(frozen=True)
    entity_id: Annotated[
        ULID, RefSpec("entity", scope_rule="same_scope", target_forget_action="retain_locator"),
    ]
    byte_start: _Offset
    byte_end: _Offset

    @model_validator(mode="after")
    def check_range(self) -> Self:
        """Require ``0 <= byte_start < byte_end <= 1,048,576``."""
        _check_range(self.byte_start, self.byte_end)
        return self


class NamingAnnotationV1(StrictModel):
    """One name a naming memory gives to one of its anchors.

    The snapshot that carries it checks that ``entity_id`` is one of its
    anchors and that the name is token-contained in its statement.
    """

    model_config = ConfigDict(frozen=True)
    entity_id: Annotated[EntityRef, OpaqueIdSpec("local")]
    name: str
    name_kind: NameKind

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        """Require a valid name (`is_name`): 1..128 characters, bound length 1..24."""
        if not is_name(value):
            raise ValueError(_NAME_RULE)
        return value


class UserChoiceV1(StrictModel):
    """The authenticated user as a referent (judge binding ``self``)."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["user"]


class ParticipantChoiceV1(StrictModel):
    """A source or labelled participant, minted at its first anchoring commit."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["participant"]
    participant_key: Annotated[HexDigest, OpaqueIdSpec("digest")]
    entity_id: Annotated[ULID, OpaqueIdSpec("local")] | None


class ExistingChoiceV1(StrictModel):
    """An existing entity, bound by an SDK binding or chosen from the catalog."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["existing"]
    entity_id: Annotated[ULID, OpaqueIdSpec("local")]
    via: Literal["sdk", "catalog"]


class NewEntityChoiceV1(StrictModel):
    """An entity a reply declares as new; minted on ``mint_key`` when committed."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["new"]
    local_key: LocalKey
    name: str | None
    descriptor: Annotated[str, Field(min_length=1, max_length=MAX_DESCRIPTOR_CHARS)]
    entity_type: str

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str | None) -> str | None:
        """Require a valid name (`is_name`) when a name is declared."""
        if value is not None and not is_name(value):
            raise ValueError(_NAME_RULE)
        return value

    @field_validator("entity_type")
    @classmethod
    def check_entity_type(cls, value: str) -> str:
        """Require the stored form: NFC, trimmed, 1..64 characters."""
        if canonical_entity_type(value) != value:
            raise ValueError("An entity type is stored after NFC and trimming")
        return value


class LocalKeyRefV1(StrictModel):
    """A candidate that is another declaration of the same reply."""

    model_config = ConfigDict(frozen=True)
    local_key: LocalKey


class UnresolvedChoiceV1(StrictModel):
    """A referent the sources do not establish; it is never guessed."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["unresolved"]
    mention: Annotated[str, Field(min_length=1, max_length=MAX_MENTION_CHARS)] | None
    reason: Literal["ambiguous", "insufficient_context"]
    candidates: Annotated[
        list[EntityRef | LocalKeyRefV1], Field(max_length=MAX_REFERENT_CANDIDATES),
    ]


EntityChoiceV1 = Annotated[
    UserChoiceV1 | ParticipantChoiceV1 | ExistingChoiceV1 | NewEntityChoiceV1
    | UnresolvedChoiceV1,
    Field(discriminator="kind"),
]
"""One anchor slot of ``ProposalV2``, discriminated by ``kind``."""

VERDICT_REF: Final = RefSpec(
    "artifact", ref_mode="snapshot_locator", gc_root=True, artifact_kinds=("result_index",),
    target_forget_action="retain_locator",
)
"""The ``result_index:judge`` that admitted a version's content: it exists when
the version is written and is retained with it, and a version never depends on it."""

_ERASED_VALUES: dict[str, object] = {
    "statement_digest": ERASED_DIGEST,
    "entity_id": "",
    "about_entity_ids": [],
    "naming": [],
    "quote_spans": [],
    "support_groups": [],
    "time_expr": {"kind": "none"},
    "valid_from": None,
    "valid_to": None,
    "time_granularity": "unanchored",
    "anchored_from": {"kind": "not_applicable", "source_time": None},
    "keywords": [],
    "sensitivity_classes": [],
    "sensitivity_basis": [],
    "last_confirmed_at": None,
    "embedding_input_digest": ERASED_DIGEST,
    "verdict_ref": None,
}
"""Every snapshot field read or derived from source text, at its erased value."""


class MemorySnapshotV2(StrictModel):
    """The content and support authority of one memory version.

    A snapshot is either content-bearing or erased. The erased snapshot is the
    deterministic content-free head of privacy erasure (`erase`): its statement
    is empty and only ``kind``, ``importance``, ``origin``, ``trust_tier``,
    ``security_level``, ``embedding_space`` and ``pipeline_version`` survive.
    Every other snapshot carries a canonical statement with its digests, its
    anchors and naming, at least one support group over its quote spans, its
    time, keywords, sensitivity, vector and the verdict that admitted it.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[2]
    statement: Annotated[str, Field(max_length=MAX_STATEMENT_CHARS)]
    statement_digest: Annotated[HexDigest, OpaqueIdSpec("digest")]
    kind: MemoryKind
    entity_id: Annotated[EntityRef | Literal[""], OpaqueIdSpec("local")]
    about_entity_ids: Annotated[
        list[EntityRef], Field(max_length=MAX_ABOUT_ENTITIES), OpaqueIdSpec("local"),
    ]
    naming: Annotated[list[NamingAnnotationV1], Field(max_length=MAX_NAMING)]
    quote_spans: Annotated[list[QuoteSpanV1], Field(max_length=MAX_MEMORY_SPANS)]
    support_groups: Annotated[list[list[int]], Field(max_length=MAX_SUPPORT_GROUPS)]
    time_expr: TimeExpressionV1
    valid_from: HistoricalTimestamp | None
    valid_to: HistoricalTimestamp | None
    time_granularity: TimeGranularity
    anchored_from: AnchorV1
    keywords: Annotated[list[str], Field(max_length=MAX_KEYWORDS)]
    importance: UnitFloat
    origin: Origin
    trust_tier: TrustTier
    security_level: SecurityLevel
    sensitivity_classes: list[SensitivityClass]
    sensitivity_basis: list[SensitivityBasis]
    last_confirmed_at: SystemTimestamp | None
    embedding: EmbeddingVector
    embedding_space: Annotated[str, Field(min_length=1, max_length=1024)]
    embedding_input_digest: Annotated[HexDigest, OpaqueIdSpec("digest")]
    pipeline_version: Annotated[str, Field(min_length=1, max_length=1024)]
    verdict_ref: Annotated[ArtifactId, VERDICT_REF] | None

    @property
    def erased(self) -> bool:
        """Whether this is the erased, content-free snapshot (its statement is empty)."""
        return self.statement == ""

    @property
    def anchors(self) -> tuple[str, ...]:
        """The primary anchor followed by the ``about_entity_ids``; empty when erased."""
        return (self.entity_id, *self.about_entity_ids) if self.entity_id else ()

    def erase(self) -> MemorySnapshotV2:
        """Return the erased snapshot that privacy erasure writes for this version.

        The result depends only on the kept fields, so erasing twice gives the
        same bytes, and erasing an erased snapshot returns an equal one.

        Args:
            None.

        Returns:
            MemorySnapshotV2: The erased snapshot, with a zero vector of the same
                dimension.
        """
        return MemorySnapshotV2.model_validate({
            **self.model_dump(mode="json"),
            **_ERASED_VALUES,
            "statement": "",
            "embedding": [0.0] * len(self.embedding),
        })

    @model_validator(mode="after")
    def check_snapshot(self) -> Self:
        """Apply the version rules: the exact erased shape, or a complete version."""
        parts = self.embedding_space.rsplit("@", 2)
        if len(parts) != 3 or not all(parts) or parts[2] != str(len(self.embedding)):
            raise ValueError("embedding_space is {model}@{canary_digest}@{dim} of this vector")
        if self.erased:
            if (
                self.model_dump(mode="json", include=set(_ERASED_VALUES)) != _ERASED_VALUES
                or any(self.embedding)
            ):
                raise ValueError("An erased snapshot keeps no statement, support or vector")
            return self
        self._check_statement()
        self._check_anchors()
        self._check_support()
        self._check_time()
        self._check_fields()
        return self

    def _check_statement(self) -> None:
        if canonical_statement(self.statement) != self.statement:
            raise ValueError("A statement is stored in its canonical form")
        if self.statement_digest != statement_digest(self.statement):
            raise ValueError("statement_digest does not match the statement")
        expected = hashlib.sha256(self.statement.encode("utf-8")).hexdigest()
        if self.embedding_input_digest != expected:
            raise ValueError("embedding_input_digest is sha256 of the statement")

    def _check_anchors(self) -> None:
        if not self.entity_id:
            raise ValueError("A version that is not erased has a primary anchor")
        about = self.about_entity_ids
        if about != sorted(set(about)) or self.entity_id in about:
            raise ValueError("about_entity_ids are sorted, unique and exclude the primary")
        seen: set[tuple[str, str, str]] = set()
        for annotation in self.naming:
            if annotation.entity_id not in self.anchors:
                raise ValueError("A naming annotation names one of the anchors")
            if not token_contained(annotation.name, self.statement):
                raise ValueError("A naming annotation's name is token-contained in the statement")
            key = (annotation.entity_id, name_norm(annotation.name), annotation.name_kind)
            if key in seen:
                raise ValueError("Naming annotations are unique by entity, name_norm and kind")
            seen.add(key)

    def _check_support(self) -> None:
        if not self.quote_spans or not self.support_groups:
            raise ValueError("A version that is not erased has quote spans and support groups")
        if any(len(group) > MAX_GROUP_SPANS for group in self.support_groups):
            raise ValueError(f"A support group locates at most {MAX_GROUP_SPANS} spans")
        validate_support_groups(self.support_groups, len(self.quote_spans))

    def _check_time(self) -> None:
        bounds = (self.valid_from, self.valid_to)
        if self.anchored_from.kind == "not_applicable" or self.time_expr.kind == "none":
            if (
                self.anchored_from.kind != "not_applicable"
                or self.time_expr.kind != "none"
                or bounds != (None, None)
                or self.time_granularity != "unanchored"
            ):
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

    def _check_fields(self) -> None:
        keys: set[str] = set()
        for keyword in self.keywords:
            if (
                keyword != keyword.strip()
                or not 1 <= len(keyword) <= MAX_KEYWORD_CHARS
                or not any(char.isalnum() for char in keyword)
            ):
                raise ValueError("A keyword is trimmed, 1..128 characters, with a letter or digit")
            key = unicodedata.normalize("NFKC", keyword).casefold()
            if key in keys:
                raise ValueError("Keywords are distinct after NFKC and casefolding")
            keys.add(key)
        if round(self.importance, 6) != self.importance:
            raise ValueError("importance is rounded to 6 decimals")
        if self.sensitivity_basis != sorted(set(self.sensitivity_basis)):
            raise ValueError("sensitivity_basis is sorted and unique")
        if bool(self.sensitivity_classes) != bool(self.sensitivity_basis):
            raise ValueError("sensitivity_basis is non-empty exactly when classes are")
        if self.last_confirmed_at is None:
            raise ValueError("A version that is not erased records last_confirmed_at")
        if not any(self.embedding):
            raise ValueError("A version that is not erased has a non-zero vector")
        if self.verdict_ref is None:
            raise ValueError("A version that is not erased records the verdict that admitted it")
