"""The judge-batch-v8 wire: physical request, strict schema, acceptance.

One physical request reviews 1 to 4 independent members (``MAX_BATCH_MEMBERS``).
Its system message is the ``memory-judge-v8`` prompt and its user message the
canonical JSON of one ``JudgeBatchRequestV8``: a catalog holding each referenced
entry once, and the members keyed by their full member input digest. The advertised
schema's enums are the issued member ids and every handle issued in the request,
read back from that message, so a reply can name nothing the request did not issue.

The reply model checks only the envelope. Each member's result is parsed and
accepted on its own, so one malformed member is resent alone while the others
keep their verdicts. A duplicate or unknown member id makes the whole
envelope unusable; a missing member is that member's fault only. Two slips keep
the verdict: a ``confirm`` that names the slot's own nominee is stored with its
choice null, and a reason over 400 characters is kept unreadable.

A member whose own form rules a member-level verdict value out
(``judge_structure``) gets a result schema without it, and acceptance refuses
a result that still uses one as ``result_schema``. A request without such
a member advertises exactly the schema it always did. An accepted ``correct`` of a
target the member only denies, or ``update`` of a naming head it denies without
naming anything, is stored as ``retract``, an accepted ``inferred``
of a speaker's explicit report about a third party as ``direct``, a
``restates`` of an untimed state tombstone by a bounded earlier episode of that state
as ``other`` as is one of a recurring habit by a bounded past episode of
its activity or by a request that only asks about its topic,
and a ``same`` of a sibling sharing no topic word with the member as ``independent``.
A ``compound`` on one attribute whose subject only has an identifying
relational descriptor is stored as ``atomic``.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal, Self, cast, get_args

from pydantic import Field, StringConstraints, ValidationError, field_validator, model_validator

from quotemem.core.add_read_view import Sigma
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_output import (
    AdvertisedRequest,
    MemoryKind,
    NameKind,
    SourceRole,
    Statement,
    WireDate,
    WireText,
)
from quotemem.core.identity import canonical_json
from quotemem.core.judge_structure import (
    NO_CONSTRAINTS,
    MemberConstraintsV1,
    StructureQuoteV1,
    bounded_episode,
    denies,
    explicit_report,
    habit_episode,
    identified_attribute,
    member_constraints,
    quote_context,
    quote_sentence,
    self_report,
    settle_attribution,
    settle_relation,
    settle_sibling,
    settle_tombstone,
    topic_request,
    unrelated,
)
from quotemem.core.judge_template_v8 import MEMORY_JUDGE_PROMPT_V8
from quotemem.core.semantic import TimeGranularity
from quotemem.core.sensitivity import SensitivityClass
from quotemem.core.statement import token_seq
from quotemem.store.metadata import OpaqueIdSpec, OpaquePayloadSpec, StrictModel

JUDGE_BATCH_V8_SCHEMA_ID = "judge-batch-v8"
JUDGE_V8_SCHEMA_ID = "judge-v8"
JUDGE_BATCH_V8_SCHEMA_REVISION = 12
"""Revision of judge-batch-v8's per-member result schema and acceptance.

The wire keeps the schema name ``judge_batch_v8``, so a judge binding pins this
revision and no verdict is reused across one. A recognized member form
narrows that member's own result definition and acceptance settles
the relation a denial fixes. Acceptance stores a speaker's
explicit report of a third party's sensitive trait, judged ``inferred``, as
``direct``. A consequential verdict, one that denies its
member or ends an issued head, stands only once two of at most three verdicts on the
member's input agree (``judge_confirmation``), and acceptance stores a
``restates`` of an untimed state tombstone by a bounded earlier episode of that state
as ``other``. A speaker's explicit report about a third party is also recognized in
a later clause, a relative clause or appositive, or a sentence or clause opening with an
unambiguous ``he``/``she``, and never through relayed, doubted or alternative wording or
a shortened possessive chain. Acceptance stores a ``same`` of a sibling that
shares no topic word with the member as ``independent``.
Acceptance stores a ``restates`` of a tombstone that states a
recurring present habit with its own schedule as ``other`` when the member places the
habit's activity in a bounded past period and names no schedule. Revision
8: acceptance stores a ``restates`` of a tombstone as ``other`` when the member only asks
the assistant about the tombstone's topic and leaves out its asserted value.
The confirmed verdict reads a sibling it folds (``same``, ``supplement`` or a
change) ``independent`` unless every verdict sharing its consequence folds that sibling
too (``judge_confirmation.agreed_folds``). Acceptance settles a speaker's explicit sensitive
self-report as ``direct``, including a participant identified by its transcript line label and
grammatical restatements of the speaker's own experience.
Self-reports use the full attribution enum, as third-party reports already do.
Message entries carry their source order when known. Dialogue ellipsis
and an object pronoun are settled only with comparable, explicit source chronology.
Acceptance stores ``compound`` as ``atomic`` for one attribute of a subject
whose relational descriptor only identifies it (``identified_attribute``).
"""
MAX_BATCH_MEMBERS = 4
"""The full mode's members a request."""
MAX_PACKET_MEMBERS = 128
"""The wire bound of members a request: the batched packing reviews one Add's members,
at most ``MAX_UNIT_FACTS`` (128) a unit, in one request where it fits."""
"""Members one physical request carries at most."""
REASON_CHARS = 400
COPY_CHECK_TOKENS = 6
HANDLE_PATTERN = r"^[MHSTENCQ][a-z2-7]{10}(?:[a-z2-7]{6})?$"
SAID_AT_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z$"
Handle = Annotated[str, StringConstraints(pattern=HANDLE_PATTERN)]
SaidAt = Annotated[str, StringConstraints(pattern=SAID_AT_PATTERN)]
Slot = Annotated[int, Field(ge=0, le=7)]
EntityType = Annotated[str, Field(min_length=1, max_length=64)]
SupportVerdict = Literal["supported", "contradicted", "insufficient"]
Relation = Literal[
    "independent", "contradicts", "same", "supplement", "update", "correct", "retract",
    "unclear",
]
Atomicity = Literal["atomic", "indivisible", "compound", "uncertain"]
Attribution = Literal["direct", "inferred", "none"]
JudgeFault = Literal[
    "envelope", "missing", "result_schema", "group_support", "entities", "relations",
    "restates_history", "tombstones", "external_only", "change_span",
    "sensitive_attribution",
]
"""Why a member has no accepted verdict: the rule it broke, or its absence."""
SUPPORT_VERDICTS: tuple[str, ...] = get_args(SupportVerdict)
RELATIONS: tuple[str, ...] = get_args(Relation)
ATOMICITY: tuple[str, ...] = get_args(Atomicity)
ATTRIBUTIONS: tuple[str, ...] = get_args(Attribution)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_RETIRED_TARGET_RELATIONS = frozenset({"independent", "same", "correct", "retract", "unclear"})
_CHANGE_RELATIONS = frozenset({"update", "correct", "retract"})


def render_said_at(value: int) -> str:
    """Render an object's maximum σ time as ``said_at``.

    Args:
        value: Epoch milliseconds of a system time.

    Returns:
        str: RFC 3339 UTC with milliseconds, such as ``2024-05-08T13:56:00.000Z``.
    """
    moment = _EPOCH + timedelta(milliseconds=value)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JudgeTimeV8(StrictModel):
    """A fact's time expression and its anchored validity, dated."""

    text: Annotated[str, Field(max_length=1024)] | None
    valid_from: WireDate | None
    valid_to: WireDate | None
    granularity: TimeGranularity | None


class JudgeMemoryEntryV8(StrictModel):
    """A current stored memory (handle M); ``history`` only for a history correction."""

    type: Literal["memory"]
    status: Literal["current"]
    statement: Statement
    kind: MemoryKind
    time: JudgeTimeV8
    said_at: SaidAt
    about: list[Handle]
    naming: bool
    history: Annotated[list[Handle], Field(max_length=2)]


class JudgeHistoryEntryV8(StrictModel):
    """An earlier version (handle H); ``of`` is null for the head of an ended memory."""

    type: Literal["history"]
    status: Literal["earlier_version"]
    of: Handle | None
    statement: Statement
    kind: MemoryKind
    time: JudgeTimeV8
    said_at: SaidAt
    ended_by: Literal["update", "correct", "retract", "supplement"]


class JudgeSiblingEntryV8(StrictModel):
    """An earlier member of the same owner and round (handle S); never evidence.

    ``about`` lists E handles, and N handles for anchors not minted yet.
    """

    type: Literal["sibling"]
    status: Literal["in_this_add"]
    statement: Statement
    kind: MemoryKind
    time: JudgeTimeV8
    said_at: SaidAt
    about: list[Handle]


class JudgeForgottenEntryV8(StrictModel):
    """A nominated tombstone (handle T): text that only near-match review may see."""

    type: Literal["forgotten"]
    status: Literal["forgotten"]
    statement: Statement
    kind: MemoryKind
    time: JudgeTimeV8
    about: list[Handle]


class JudgeEntityNameV8(StrictModel):
    """One readable name of an entity."""

    name: Annotated[str, Field(min_length=1, max_length=128)]
    kind: NameKind


class JudgeEntityEntryV8(StrictModel):
    """An existing entity (handle E), incl. the user and participants.

    The entry is the same for every member of a round: ``binding`` says
    what the entity is, the user (``self``), a participant or a ``third_party``,
    and an SDK binding shows on the anchor that has it; ``facts`` are the
    entity's most recent readable memories, newest first.
    """

    type: Literal["entity"]
    binding: Literal["self", "participant", "third_party"]
    entity_type: EntityType | None
    names: list[JudgeEntityNameV8]
    facts: Annotated[list[Statement], Field(max_length=3)]


class JudgeNewEntityEntryV8(StrictModel):
    """A new-entity nomination (handle N), keyed by its mint key."""

    type: Literal["new_entity"]
    entity_type: EntityType
    proposed_name: Annotated[str, Field(min_length=1, max_length=128)] | None
    descriptor: Annotated[str, Field(min_length=1, max_length=256)]
    mention: Annotated[str, Field(min_length=1, max_length=256)] | None


class JudgeMessageEntryV8(StrictModel):
    """A cited or context message (handle Q), rendered masked.

    ``excerpt`` marks a partly shown text: a context message's tail, or the quoted parts
    of a cited message in a member too large to send whole.
    ``order`` is the source σ when known; grammar that requires a preceding turn
    or an antecedent cannot be settled without it.
    """

    type: Literal["message"]
    role: SourceRole
    speaker: WireText | None
    said_at: SaidAt
    text: WireText
    excerpt: bool
    order: Sigma | None = None

    @field_validator("order", mode="before")
    @classmethod
    def source_order(cls, value: object) -> object:
        """Read the canonical JSON array while retaining strict coordinate validation."""
        return tuple(cast(list[object], value)) if isinstance(value, list) else value


CatalogEntryV8 = Annotated[
    JudgeMemoryEntryV8 | JudgeHistoryEntryV8 | JudgeSiblingEntryV8 | JudgeForgottenEntryV8
    | JudgeEntityEntryV8 | JudgeNewEntityEntryV8 | JudgeMessageEntryV8,
    Field(discriminator="type"),
]
_STATED_ENTRIES = (
    JudgeMemoryEntryV8, JudgeHistoryEntryV8, JudgeSiblingEntryV8, JudgeForgottenEntryV8,
)
_ENTRY_PREFIX = {
    "memory": "M", "history": "H", "sibling": "S", "forgotten": "T", "entity": "E",
    "new_entity": "N", "message": "Q",
}


class JudgeNamingV8(StrictModel):
    """One naming annotation of the member, on one of its anchor slots."""

    slot: Slot
    name: Annotated[str, Field(min_length=1, max_length=128)]
    name_kind: NameKind


class JudgeAnchorV8(StrictModel):
    """One anchor slot; only ``nominated`` and ``open`` slots are judged.

    ``catalog_complete`` is true when every live entity whose type
    or names could match the slot is among its candidates, false otherwise, and
    null for a slot that is not judged.
    """

    slot: Slot
    role: Literal["primary", "secondary"]
    binding: Literal["self", "participant", "sdk", "nominated", "open"]
    entity: Handle | None
    mention: Annotated[str, Field(max_length=256)] | None
    candidates: Annotated[list[Handle], Field(max_length=7)]
    catalog_complete: bool | None


class JudgeQuoteV8(StrictModel):
    """One located quote span of the member (handle C) inside its message (handle Q)."""

    handle: Handle
    source: Handle
    text: WireText


def _bound_anchor(anchor: JudgeAnchorV8) -> bool:
    """Candidates and ``catalog_complete`` exist exactly for a judged slot; a nominated
    entity is one of its candidates."""
    candidates = anchor.candidates
    judged = anchor.binding in ("nominated", "open")
    if len(set(candidates)) != len(candidates) or any(
        handle[0] not in "EN" for handle in candidates
    ) or (anchor.catalog_complete is not None) != judged:
        return False
    if anchor.binding == "open":
        return anchor.entity is None and bool(candidates)
    if anchor.binding == "nominated":
        return anchor.entity is not None and anchor.entity in candidates
    return anchor.entity is not None and anchor.entity[0] == "E" and not candidates


class JudgeMemberV8(StrictModel):
    """One member's logical input: the proposed memory and the handles it must judge."""

    statement: Statement
    kind: MemoryKind
    time: JudgeTimeV8
    said_at: SaidAt
    naming: Annotated[list[JudgeNamingV8], Field(max_length=4)]
    anchors: Annotated[list[JudgeAnchorV8], Field(min_length=1, max_length=8)]
    quotes: Annotated[list[JudgeQuoteV8], Field(min_length=1, max_length=256)]
    support_groups: Annotated[
        list[Annotated[list[Handle], Field(min_length=1, max_length=32)]],
        Field(min_length=1, max_length=8),
    ]
    context: Annotated[list[Handle], Field(max_length=4)]
    sensitivity_classes: Annotated[list[SensitivityClass], Field(max_length=8)]
    correction_requested: bool
    corrects_text: Annotated[str, Field(max_length=512)] | None
    revises: Annotated[list[Handle], Field(max_length=8)]
    entity_partition_complete: bool
    external_only_allowed: bool
    targets: Annotated[list[Handle], Field(max_length=64)]
    history: Annotated[list[Handle], Field(max_length=80)]
    tombstones: Annotated[list[Handle], Field(max_length=8)]

    @model_validator(mode="after")
    def member_handles(self) -> Self:
        """Check slots, bindings and the kind of every handle the member lists."""
        slots = [anchor.slot for anchor in self.anchors]
        if slots != sorted(set(slots)) or slots[0] != 0 or [
            anchor.role for anchor in self.anchors
        ] != ["primary", *["secondary"] * (len(slots) - 1)]:
            raise ValueError("Anchors hold ascending distinct slots with the primary at slot 0")
        if not all(_bound_anchor(anchor) for anchor in self.anchors) or any(
            naming.slot not in slots for naming in self.naming
        ):
            raise ValueError("Anchor bindings, candidates and naming slots disagree")
        quotes = [quote.handle for quote in self.quotes]
        if len(set(quotes)) != len(quotes) or any(
            quote.handle[0] != "C" or quote.source[0] != "Q" for quote in self.quotes
        ) or any(
            len(set(group)) != len(group) or not set(group) <= set(quotes)
            for group in self.support_groups
        ):
            raise ValueError("Groups list distinct C handles of the member's own quotes")
        for handles, prefixes in (
            (self.context, "Q"), (self.targets, "MSH"), (self.history, "H"),
            (self.tombstones, "T"), (self.revises, "MH"),
        ):
            if len(set(handles)) != len(handles) or any(
                handle[0] not in prefixes for handle in handles
            ):
                raise ValueError("Context, targets, history, tombstones and revises are typed")
        if self.external_only_allowed and not (
            self.correction_requested and self.entity_partition_complete
        ):
            raise ValueError("An external-only correction needs a complete primary partition")
        return self


class JudgeBatchItemRequestV8(StrictModel):
    """One member of a physical request, identified by its full member input digest."""

    member_id: Annotated[str, Field(min_length=1, max_length=128)]
    member: JudgeMemberV8


def _references(entry: CatalogEntryV8) -> list[tuple[str, str]]:
    if isinstance(entry, JudgeMemoryEntryV8):
        return [(handle, "E") for handle in entry.about] + [
            (handle, "H") for handle in entry.history
        ]
    if isinstance(entry, JudgeHistoryEntryV8):
        return [] if entry.of is None else [(entry.of, "M")]
    if isinstance(entry, JudgeSiblingEntryV8):
        # A sibling may be anchored to a declaration no commit has minted yet.
        return [(handle, "EN") for handle in entry.about]
    if isinstance(entry, JudgeForgottenEntryV8):
        return [(handle, "E") for handle in entry.about]
    return []


def _member_references(member: JudgeMemberV8) -> list[tuple[str, str]]:
    anchors = [
        (handle, "EN") for anchor in member.anchors
        for handle in ([] if anchor.entity is None else [anchor.entity]) + anchor.candidates
    ]
    return anchors + [(quote.source, "Q") for quote in member.quotes] + [
        (handle, "Q") for handle in member.context
    ] + [(handle, "MSH") for handle in member.targets] + [
        (handle, "H") for handle in member.history
    ] + [(handle, "T") for handle in member.tombstones] + [
        (handle, "MH") for handle in member.revises
    ]


class JudgeBatchRequestV8(StrictModel):
    """The user message of one judge-batch-v8 request.

    The catalog maps each referenced handle to one entry whose type matches the
    handle's prefix and holds only the entries its members reach, directly or
    through other entries; a C handle shared by two members names one span.
    """

    catalog: dict[Handle, CatalogEntryV8]
    items: Annotated[
        list[JudgeBatchItemRequestV8], Field(min_length=1, max_length=MAX_PACKET_MEMBERS),
    ]

    @model_validator(mode="after")
    def closed_catalog(self) -> Self:
        """Resolve every reference to an entry of its kind and refuse any extra entry."""
        identities = [item.member_id for item in self.items]
        if len(set(identities)) != len(identities):
            raise ValueError("Member ids are distinct")
        if any(handle[0] != _ENTRY_PREFIX[entry.type] for handle, entry in self.catalog.items()):
            raise ValueError("Each catalog handle's prefix names its entry type")
        spans: dict[str, tuple[str, str]] = {}
        pending: list[tuple[str, str]] = []
        for item in self.items:
            pending += _member_references(item.member)
            for quote in item.member.quotes:
                if spans.setdefault(quote.handle, (quote.source, quote.text)) != (
                    quote.source, quote.text,
                ):
                    raise ValueError("One quote handle names one span")
        reached: set[str] = set()
        while pending:
            handle, prefixes = pending.pop()
            if handle[0] not in prefixes or handle not in self.catalog:
                raise ValueError("Every reference names a catalog entry of its kind")
            if handle not in reached:
                reached.add(handle)
                pending += _references(self.catalog[handle])
        if reached != set(self.catalog):
            raise ValueError("The catalog holds only the entries its members reach")
        return self

    def issued_handles(self) -> list[str]:
        """Return every handle issued in this request (catalog and quotes), sorted."""
        return sorted({
            *self.catalog, *(quote.handle for item in self.items for quote in item.member.quotes),
        })


def judge_messages(request: JudgeBatchRequestV8) -> tuple[str, str]:
    """Return the system and user message texts of one judge-batch-v8 request.

    Args:
        request: The physical request's members and catalog.

    Returns:
        tuple[str, str]: The ``memory-judge-v8`` prompt exactly, and the canonical
            JSON of ``request``.
    """
    return MEMORY_JUDGE_PROMPT_V8, canonical_json(request)


def judge_member_constraints(
    member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8],
) -> MemberConstraintsV1:
    """Return the verdict values one member's own input rules out.

    Args:
        member: The member as a request lists it.
        catalog: A catalog holding every entry the member reaches.

    Returns:
        MemberConstraintsV1: ``judge_structure.member_constraints`` of the member's
            statement, ``correction_requested`` and the statements of its current
            targets (memories and siblings; a retired version is a history
            correction's own target).
    """
    targets = [entry.statement for handle in member.targets
               if isinstance(entry := catalog[handle], JudgeMemoryEntryV8 | JudgeSiblingEntryV8)]
    return member_constraints(
        member.statement, correction_requested=member.correction_requested,
        targets=targets,
    )


def _user_quotes(
    member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8],
) -> list[StructureQuoteV1]:
    """The member's quotes on ``user`` messages, with each speaker, sentence and the
    text of the sentence's line before it."""
    quotes: list[StructureQuoteV1] = []
    for quote in member.quotes:
        source = catalog[quote.source]
        if isinstance(source, JudgeMessageEntryV8) and source.role == "user":
            quotes.append(StructureQuoteV1(
                quote.text, source.speaker,
                quote_sentence(source.text, quote.text, excerpt=source.excerpt),
                quote_context(source.text, quote.text, excerpt=source.excerpt),
                source.order,
            ))
    return quotes


def _user_messages(
    member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8],
) -> list[StructureQuoteV1]:
    """Shown user messages, used only to rule out intervening referents."""
    return [StructureQuoteV1(source.text, source.speaker,
                             None if source.excerpt else source.text,
                             None if source.excerpt else "", source.order)
            for handle in sorted({*(quote.source for quote in member.quotes), *member.context})
            if isinstance(source := catalog[handle], JudgeMessageEntryV8) and source.role == "user"]


def request_constraints(request: JudgeBatchRequestV8) -> dict[str, MemberConstraintsV1]:
    """Return the constraints of each constrained member of one physical request.

    Args:
        request: The physical request.

    Returns:
        dict[str, MemberConstraintsV1]: By member id, only members with a constraint.
    """
    found: dict[str, MemberConstraintsV1] = {}
    for item in request.items:
        constraints = judge_member_constraints(item.member, request.catalog)
        if constraints:
            found[item.member_id] = constraints
    return found


def judge_batch_schema(
    member_ids: Sequence[str], handles: Sequence[str],
    constraints: Mapping[str, MemberConstraintsV1] | None = None,
) -> dict[str, Any]:
    """Build the advertised strict response schema for one physical request.

    Every ``Handle`` is an enum of the handles issued in the request; acceptance
    still checks each member's own handles. No ``pattern`` keyword is used and
    nullable fields use ``anyOf`` with null. A constrained member gets its
    own result definition ``JudgeV8m<position>`` without its excluded values, chosen
    by its member id; without constraints the schema is the plain one.

    Args:
        member_ids: The request's member ids in request order.
        handles: Every handle issued in the request.
        constraints: Excluded verdict values by member id; members not listed, or
            listed without an exclusion, use ``JudgeV8``.

    Returns:
        dict[str, Any]: A schema requiring exactly one item per member.
    """

    def enum(values: Sequence[str]) -> dict[str, Any]:
        return {"type": "string", "enum": list(values)}

    def nullable(schema: dict[str, Any]) -> dict[str, Any]:
        return {"anyOf": [schema, {"type": "null"}]}

    def closed(properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "object", "additionalProperties": False, "required": list(properties),
            "properties": properties,
        }

    def handle() -> dict[str, Any]:
        return {"$ref": "#/$defs/Handle"}

    def array(items: dict[str, Any], maximum: int) -> dict[str, Any]:
        return {"type": "array", "maxItems": maximum, "items": items}

    def allowed(values: Sequence[str], excluded: frozenset[str]) -> list[str]:
        return [value for value in values if value not in excluded]

    def result(excluded: MemberConstraintsV1) -> dict[str, Any]:
        return closed({
            "support": enum(SUPPORT_VERDICTS),
            "group_support": {
                "type": "array", "minItems": 1, "maxItems": 8, "items": enum(SUPPORT_VERDICTS),
            },
            "fidelity": enum(["complete", "incomplete", "uncertain"]),
            "atomicity": enum(allowed(ATOMICITY, excluded.atomicity)),
            "sensitive_attribution": nullable(enum(allowed(ATTRIBUTIONS, excluded.attribution))),
            "entities": array(closed({
                "slot": {"type": "integer", "minimum": 0, "maximum": 7},
                "verdict": enum(["confirm", "repick", "ambiguous"]),
                "choice": nullable(handle()),
            }), 8),
            "relations": array(closed({
                "target": handle(), "relation": enum(RELATIONS),
            }), 64),
            "restates_history": array(handle(), 80),
            "tombstones": array(closed({
                "tombstone": handle(), "verdict": enum(["restates", "revises", "other"]),
            }), 8),
            "external_only": nullable({"type": "boolean"}),
            "change_span": nullable(handle()),
            "reason": {"type": "string", "minLength": 1, "maxLength": REASON_CHARS},
        })

    members = len(member_ids)
    narrowed = {member_id: found for member_id in member_ids
                if (found := (constraints or {}).get(member_id, NO_CONSTRAINTS))}
    plain = [member_id for member_id in member_ids if member_id not in narrowed]
    definitions: dict[str, Any] = {"Handle": enum(handles)}
    if plain:
        definitions["JudgeV8"] = result(NO_CONSTRAINTS)
    branches: list[dict[str, Any]] = []
    for position, member_id in enumerate(member_ids):
        if member_id in narrowed:
            name = f"JudgeV8m{position}"
            definitions[name] = result(narrowed[member_id])
            branches.append(closed({"member_id": enum([member_id]),
                                    "result": {"$ref": f"#/$defs/{name}"}}))
    if plain:
        branches.append(closed({"member_id": enum(plain), "result": {"$ref": "#/$defs/JudgeV8"}}))
    return {
        "type": "object", "additionalProperties": False, "required": ["items"],
        "properties": {"items": {
            "type": "array", "minItems": members, "maxItems": members,
            "items": branches[0] if len(branches) == 1 else {"anyOf": branches},
        }},
        "$defs": definitions,
    }


def judge_request_schema(request: JudgeBatchRequestV8) -> dict[str, Any]:
    """Build the advertised strict response schema of one physical request.

    Args:
        request: The physical request.

    Returns:
        dict[str, Any]: ``judge_batch_schema`` of its member ids, issued handles
            and ``request_constraints``.
    """
    return judge_batch_schema(
        [item.member_id for item in request.items], request.issued_handles(),
        request_constraints(request),
    )


def advertise_judge_batch_v8(request: AdvertisedRequest) -> dict[str, Any]:
    """Advertise the strict judge-batch-v8 schema of one actual request.

    The output registry calls this for ``("judge", "judge-batch-v8")``; the
    enums and every member's constraints come from the request's own user message.

    Args:
        request: The provider request whose last message is the user message.

    Returns:
        dict[str, Any]: The request's strict schema.

    Raises:
        QuoteMemError: binding_mismatch if the user message is not a ``JudgeBatchRequestV8``.
    """
    try:
        payload = JudgeBatchRequestV8.model_validate_json(request.messages[-1].content)
    except (IndexError, ValidationError) as error:
        raise QuoteMemError("binding_mismatch",
                            "Judge request lost its members and handles") from error
    return judge_request_schema(payload)


def advertise_judge(request: AdvertisedRequest) -> dict[str, Any]:
    """Advertise the stored member contract from a singleton physical input.

    Args:
        request: The frozen request containing exactly one judge member.

    Returns:
        The closed member schema, without the member's excluded values, with
        the request's issued-handle enum.

    Raises:
        QuoteMemError: ``binding_mismatch`` if the input contains several members.
    """
    batch = advertise_judge_batch_v8(request)
    items = batch["properties"]["items"]
    if items["maxItems"] != 1:
        raise QuoteMemError("binding_mismatch", "A member schema requires one judge member")
    definitions = batch["$defs"]
    name = items["items"]["properties"]["result"]["$ref"].removeprefix("#/$defs/")
    return {**definitions[name], "$defs": {"Handle": definitions["Handle"]}}


class JudgeEntityVerdictV8(StrictModel):
    """The verdict on one judged anchor slot."""

    slot: Slot
    verdict: Literal["confirm", "repick", "ambiguous"]
    choice: Annotated[str, OpaqueIdSpec("local")] | None


class JudgeRelationV8(StrictModel):
    """The one relation of the member to one issued target."""

    target: Annotated[str, OpaqueIdSpec("local")]
    relation: Relation


class JudgeTombstoneVerdictV8(StrictModel):
    """The one verdict of the member on one issued tombstone."""

    tombstone: Annotated[str, OpaqueIdSpec("local")]
    verdict: Literal["restates", "revises", "other"]


class JudgeV8(StrictModel):
    """One member's result in its closed vocabularies; acceptance checks its handles.

    ``reason`` is the only free text. Only groups judged ``supported`` become
    stored support, so a bad citation cannot inflate evidence.
    """

    support: SupportVerdict
    group_support: Annotated[list[SupportVerdict], Field(min_length=1, max_length=8)]
    fidelity: Literal["complete", "incomplete", "uncertain"]
    atomicity: Atomicity
    sensitive_attribution: Attribution | None
    entities: Annotated[list[JudgeEntityVerdictV8], Field(max_length=8)]
    relations: Annotated[list[JudgeRelationV8], Field(max_length=64)]
    restates_history: Annotated[
        list[Annotated[str, OpaqueIdSpec("local")]], Field(max_length=80),
    ]
    tombstones: Annotated[list[JudgeTombstoneVerdictV8], Field(max_length=8)]
    external_only: bool | None
    change_span: Annotated[str, OpaqueIdSpec("local")] | None
    reason: Annotated[str, Field(min_length=1)]


class JudgeBatchReplyItemV8(StrictModel):
    """One returned item; its result is accepted per member, never as a whole."""

    member_id: Annotated[str, OpaqueIdSpec("local")]
    result: Annotated[object, OpaquePayloadSpec(max_bytes=65_536)]  # one reply's bound


class JudgeBatchOutputV8(StrictModel):
    """The verbatim judge-batch-v8 reply, checked as an envelope only."""

    items: Annotated[list[JudgeBatchReplyItemV8], Field(max_length=MAX_PACKET_MEMBERS)]


@dataclass(frozen=True)
class JudgeMemberAcceptanceV8:
    """One member's outcome of one reply: an accepted verdict or why there is none.

    Attributes:
        member_id: The member's full input digest.
        verdict: The accepted result, a ``confirm`` that named its nominee stored with
            its choice null, an unasked ``external_only`` false stored
            null and a settled relation stored settled; None when the
            member must be resent alone.
        fault: None when accepted, else the rule broken or ``missing``.
        reason_readable: False when the reason is over 400 characters or failed the
            copy check; the verdict is kept and only its reason is never shown.
    """

    member_id: str
    verdict: JudgeV8 | None
    fault: JudgeFault | None
    reason_readable: bool


def _refused(member_id: str, fault: JudgeFault) -> JudgeMemberAcceptanceV8:
    return JudgeMemberAcceptanceV8(member_id, None, fault, False)


def _entities_fault(member: JudgeMemberV8, entries: Sequence[JudgeEntityVerdictV8]) -> bool:
    judged = {anchor.slot: anchor for anchor in member.anchors if anchor.binding in (
        "nominated", "open",
    )}
    if sorted(entry.slot for entry in entries) != sorted(judged):
        return True
    for entry in entries:
        anchor = judged[entry.slot]
        if entry.verdict == "repick":
            allowed = entry.choice in anchor.candidates and entry.choice != anchor.entity
        elif entry.verdict == "confirm":
            # Naming the nominee itself still confirms it.
            allowed = anchor.binding == "nominated" and entry.choice in (None, anchor.entity)
        else:
            allowed = entry.choice is None
        if not allowed:
            return True
    return False


def _confirmed(verdict: JudgeV8) -> JudgeV8:
    """Store a ``confirm`` that named its nominee with its choice null.

    Resolution reads a confirm only without a choice; any other verdict is returned
    as it is, so its stored bytes do not change.
    """
    if all(entry.verdict != "confirm" or entry.choice is None for entry in verdict.entities):
        return verdict
    return verdict.model_copy(update={"entities": [
        entry.model_copy(update={"choice": None}) if entry.verdict == "confirm" else entry
        for entry in verdict.entities
    ]})


def _external_only_asked(member: JudgeMemberV8, verdict: JudgeV8) -> bool:
    """Whether acceptance asks the member's ``external_only``: allowed and no change relation."""
    return member.external_only_allowed and not any(
        relation.relation in _CHANGE_RELATIONS for relation in verdict.relations
    )


def _unasked_false_dropped(member: JudgeMemberV8, verdict: JudgeV8) -> JudgeV8:
    """Store an ``external_only`` false that acceptance did not ask as null.

    Outside the asked case the field is null. A false there only denies an
    external-only repair, which the change relation (the member repairs a target's own
    fact) or the missing request already says, and resolution reads it as null (reads only true),
    so the paid verdict is accepted and stored as a null would be. A
    true there stays malformed.
    """
    if verdict.external_only is not False or _external_only_asked(member, verdict):
        return verdict
    return verdict.model_copy(update={"external_only": None})


def _excluded(verdict: JudgeV8, constraints: MemberConstraintsV1) -> bool:
    """Whether a result uses a value its member's constraints exclude."""
    return (verdict.atomicity in constraints.atomicity
            or verdict.sensitive_attribution in constraints.attribution)


def _settled(
    member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8], verdict: JudgeV8,
) -> JudgeV8:
    """Store what the member's own form settles (``judge_structure``).

    The judge decides whether a target held and ended or was wrong from the start; a
    member that only denies a target states no replacement value, so its ``correct``
    of that target is stored as ``retract``, and its ``update`` of a naming head as
    ``retract`` when it names nothing. A speaker's explicit report of a third party's
    sensitive trait, or their own explicit sensitive report, judged ``inferred`` is
    stored as ``direct`` (``explicit_report``, ``self_report``).
    A ``restates`` of an untimed state tombstone by a bounded earlier episode of that
    state, or of a tombstone's recurring habit by a bounded past episode of its activity, is
    stored as ``other`` (``bounded_episode``, ``habit_episode``), as is one by a member that
    only asks the assistant about the tombstone's topic (``topic_request``). A ``same`` of a
    sibling that shares no topic word with the member is stored as ``independent``
    (``unrelated``). A ``compound`` for one attribute whose subject only has an
    identifying relational descriptor is stored as ``atomic`` (``identified_attribute``).
    Any other verdict is returned as it is, so its stored bytes do not change.
    """
    relations: list[JudgeRelationV8] = []
    for relation in verdict.relations:
        entry = catalog[relation.target]
        settled = relation.relation
        if isinstance(entry, _STATED_ENTRIES) and denies(member.statement, entry.statement):
            settled = settle_relation(
                relation.relation, denied=True,
                naming_head=isinstance(entry, JudgeMemoryEntryV8) and entry.naming,
                names=bool(member.naming),
            )
        if isinstance(entry, JudgeSiblingEntryV8):
            settled = settle_sibling(settled, unrelated=settled == "same" and unrelated(
                member.statement, entry.statement))
        relations.append(relation if settled == relation.relation
                         else relation.model_copy(update={"relation": settled}))
    changes: dict[str, object] = {}
    if verdict.atomicity == "compound" and identified_attribute(member.statement):
        changes["atomicity"] = "atomic"
    if any(new is not old for new, old in zip(relations, verdict.relations, strict=True)):
        changes["relations"] = relations
    quotes = _user_quotes(member, catalog)
    attribution = settle_attribution(
        verdict.sensitive_attribution,
        reported=verdict.sensitive_attribution == "inferred" and (
            "credentials" not in member.sensitivity_classes
            and self_report(member.statement, primary=member.anchors[0].binding,
                            user_quotes=quotes, user_messages=_user_messages(member, catalog))
            or explicit_report(member.statement, sensitivity_classes=member.sensitivity_classes,
                               user_quotes=quotes)
        ),
    )
    if attribution != verdict.sensitive_attribution:
        changes["sensitive_attribution"] = attribution
    tombstones = [entry if (settled := settle_tombstone(entry.verdict, bounded=_bounded(
        member, catalog[entry.tombstone]) or _habit(member, catalog[entry.tombstone])
        or _topic_asked(member, catalog[entry.tombstone]))) == entry.verdict
        else entry.model_copy(update={"verdict": settled}) for entry in verdict.tombstones]
    if any(new is not old for new, old in zip(tombstones, verdict.tombstones, strict=True)):
        changes["tombstones"] = tombstones
    return verdict.model_copy(update=changes) if changes else verdict


def _bounded(member: JudgeMemberV8, entry: CatalogEntryV8) -> bool:
    """Whether the member is a bounded earlier episode of a tombstone's state (rule 6)."""
    if not isinstance(entry, JudgeForgottenEntryV8):
        return False
    time = entry.time
    return bounded_episode(
        member.statement, member.time.text, entry.statement, tombstone_kind=entry.kind,
        tombstone_timed=time.text is not None or time.valid_from is not None
        or time.valid_to is not None or time.granularity not in (None, "unanchored"),
    )


def _habit(member: JudgeMemberV8, entry: CatalogEntryV8) -> bool:
    """Whether the member is a bounded earlier episode of a tombstone's habit (rule 8)."""
    if not isinstance(entry, JudgeForgottenEntryV8):
        return False
    return habit_episode(
        member.statement, member.time.text, entry.statement,
        tombstone_time_text=entry.time.text, tombstone_ended=entry.time.valid_to is not None,
    )


def _topic_asked(member: JudgeMemberV8, entry: CatalogEntryV8) -> bool:
    """Whether the member only asks the assistant about a tombstone's topic (rule 9)."""
    return isinstance(entry, JudgeForgottenEntryV8) and topic_request(
        member.statement, entry.statement)


def _member_fault(member: JudgeMemberV8, verdict: JudgeV8) -> JudgeFault | None:
    """Return the first rule the verdict breaks, else None."""
    if len(verdict.group_support) != len(member.support_groups) or (
        verdict.support == "supported"
    ) != ("supported" in verdict.group_support):
        return "group_support"
    if _entities_fault(member, verdict.entities):
        return "entities"
    if sorted(relation.target for relation in verdict.relations) != sorted(member.targets) or any(
        relation.target[0] == "H" and relation.relation not in _RETIRED_TARGET_RELATIONS
        for relation in verdict.relations
    ):
        return "relations"
    restated = verdict.restates_history
    if len(set(restated)) != len(restated) or not set(restated) <= set(member.history):
        return "restates_history"
    if sorted(entry.tombstone for entry in verdict.tombstones) != sorted(member.tombstones):
        return "tombstones"
    changed = any(relation.relation in _CHANGE_RELATIONS for relation in verdict.relations)
    if verdict.external_only is None if _external_only_asked(member, verdict) else (
        verdict.external_only is True
    ):
        return "external_only"
    if (verdict.change_span is not None) != (changed or verdict.external_only is True) or (
        verdict.change_span is not None
        and verdict.change_span not in {quote.handle for quote in member.quotes}
    ):
        return "change_span"
    if (verdict.sensitive_attribution is not None) != bool(member.sensitivity_classes):
        return "sensitive_attribution"
    return None


def _grams(text: str) -> set[tuple[str, ...]]:
    tokens = token_seq(text)
    return {
        tuple(tokens[index:index + COPY_CHECK_TOKENS])
        for index in range(len(tokens) - COPY_CHECK_TOKENS + 1)
    }


def _reason_readable(
    member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8], reason: str,
) -> bool:
    """Apply the reason rule to one accepted verdict: its bound and copy check.

    The reason is unreadable when it is over 400 characters after NFC, holds a
    tombstone handle issued to the member, or holds six consecutive tokens that
    occur in a statement issued to the member (a target, history entry or
    tombstone) but in neither its own statement nor its quotes.
    """
    text = unicodedata.normalize("NFC", reason)
    if len(text) > REASON_CHARS or any(handle in text for handle in member.tombstones):
        return False
    own = _grams(member.statement).union(*(_grams(quote.text) for quote in member.quotes))
    issued: set[tuple[str, ...]] = set()
    for handle in (*member.targets, *member.history, *member.tombstones):
        entry = catalog[handle]
        if isinstance(entry, _STATED_ENTRIES):
            issued |= _grams(entry.statement)
    return not (_grams(text) & issued) - own


def accept_judge_batch(
    request: JudgeBatchRequestV8, output: JudgeBatchOutputV8,
) -> tuple[JudgeMemberAcceptanceV8, ...]:
    """Accept each member of one reply on its own.

    Args:
        request: The physical request the reply answers.
        output: The reply's structurally valid envelope.

    Returns:
        tuple[JudgeMemberAcceptanceV8, ...]: One outcome per member, in request
            order. A duplicate or unknown member id makes every outcome
            ``envelope``; an absent member is ``missing``; a result outside the
            closed vocabularies, or using a value its member's constraints exclude,
            is ``result_schema``; otherwise the first broken
            acceptance rule is named. An accepted ``confirm`` that named its
            nominee has its choice null, an unasked ``external_only`` false is
            null, a relation the member's form settles is stored settled;
            a compound whose subject descriptor only identifies it is atomic;
            an accepted verdict whose reason is
            over the bound or fails the copy check is kept with ``reason_readable``
            false.
    """
    issued = [item.member_id for item in request.items]
    received = [item.member_id for item in output.items]
    if len(set(received)) != len(received) or not set(received) <= set(issued):
        return tuple(_refused(member_id, "envelope") for member_id in issued)
    results = {item.member_id: item.result for item in output.items}
    constraints = request_constraints(request)
    outcomes: list[JudgeMemberAcceptanceV8] = []
    for item in request.items:
        if item.member_id not in results:
            outcomes.append(_refused(item.member_id, "missing"))
            continue
        try:
            verdict = JudgeV8.model_validate(results[item.member_id])
        except ValidationError:
            outcomes.append(_refused(item.member_id, "result_schema"))
            continue
        if _excluded(verdict, constraints.get(item.member_id, NO_CONSTRAINTS)):
            outcomes.append(_refused(item.member_id, "result_schema"))
            continue
        fault = _member_fault(item.member, verdict)
        outcomes.append(_refused(item.member_id, fault) if fault is not None else (
            JudgeMemberAcceptanceV8(
                item.member_id,
                _settled(item.member, request.catalog,
                         _confirmed(_unasked_false_dropped(item.member, verdict))),
                None, _reason_readable(item.member, request.catalog, verdict.reason),
            )
        ))
    return tuple(outcomes)
