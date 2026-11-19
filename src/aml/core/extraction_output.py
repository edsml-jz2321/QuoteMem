"""The extract-v12 wire: request user message (B.5), strict schema (B.6.1), reply (B.6.2).

An extraction request sends template v50 as its system message and the canonical
JSON of one ``ExtractRequestV12`` as its user message. That message issues the
source ids and the entity and memory handles, which are exactly the per-request
enums of the advertised strict schema: ``advertise_extract_v12`` reads them back
from the request itself, so the provider can only name what it was shown.

The reply model checks shape alone: exact keys, value types, the three top-level
array bounds, duplicate keys and the canonical size. Enum membership, string
lengths and per-fact array lengths are item rules of program admission (B.7), so
a reply that breaks one costs that fact, never the whole Add (INV:D16).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, NoReturn, Protocol, Self, cast, get_args

from pydantic import Field, StringConstraints, ValidationError, model_validator

from aml.core.errors import AMLError
from aml.core.extraction_bounds import OUTPUT_CANONICAL_CHARS, canonical_limit
from aml.core.extraction_template_v50 import (
    WORKFLOW_EXTRACTION_TEMPLATE_V50,
    WORKFLOW_TEMPLATE_V50_DIGEST,
)
from aml.core.identity import IntegrityDigest, canonical_json
from aml.core.semantic import TimeGranularity
from aml.core.sensitivity import SensitivityClass
from aml.core.statement import (
    MAX_ANCHORS,
    MAX_CATALOG_ENTITIES,
    MAX_CATALOG_NAMES,
    MAX_CONTEXT_MEMORIES,
    MAX_CORRECTS_TEXT_CHARS,
    MAX_DESCRIPTOR_CHARS,
    MAX_EARLIER_PER_MEMORY,
    MAX_ENTITY_TYPE_CHARS,
    MAX_FRESH_SOURCES,
    MAX_GROUP_QUOTES,
    MAX_KEYWORD_CHARS,
    MAX_KEYWORDS,
    MAX_MENTION_CHARS,
    MAX_NAME_CHARS,
    MAX_NAMING,
    MAX_NEW_ENTITIES,
    MAX_QUOTE_CHARS,
    MAX_REFERENT_CANDIDATES,
    MAX_REPLY_FACTS,
    MAX_REVISES,
    MAX_SENSITIVITY_FINDINGS,
    MAX_SOURCES,
    MAX_STATEMENT_CHARS,
    MAX_SUPPORT_GROUPS,
    MAX_TAIL_SOURCES,
    MAX_TIME_TEXT_CHARS,
    MAX_UNRESOLVED_REFERENTS,
)
from aml.store.metadata import OpaqueIdSpec, StrictModel

EXTRACT_V12_SCHEMA_ID = "extract-v12"
NEW_ENTITY_HANDLES = tuple(f"N{number}" for number in range(1, MAX_NEW_ENTITIES + 1))
UNRESOLVED_HANDLES = tuple(f"R{number}" for number in range(1, MAX_UNRESOLVED_REFERENTS + 1))
SENSITIVITY_CATEGORIES: tuple[str, ...] = get_args(SensitivityClass)

SourceRole = Literal["user", "assistant", "tool", "system", "unknown", "participant"]
"""A source's wire role; ``participant`` is a labelled participant's envelope transcript line."""
SourceBatch = Literal["previous", "tail", "current"]
EntityKind = Literal["user", "participant", "entity"]
NameKind = Literal["name", "alias", "former"]
MemoryKind = Literal["state", "event"]
ExtractReplyFault = Literal["provider_schema_invalid", "output_bound"]

WIRE_DATE_PATTERN = r"^(?:[0-9]{4}|[+-][0-9]{4,})(?:-[0-9]{2}(?:-[0-9]{2}(?:T[0-9]{2}Z)?)?)?$"
WireDate = Annotated[str, StringConstraints(pattern=WIRE_DATE_PATTERN)]
SourceId = Annotated[str, StringConstraints(pattern=r"^S[1-9][0-9]*$")]
EntityHandle = Annotated[str, StringConstraints(pattern=r"^(?:user|E[1-9][0-9]*)$")]
MemoryHandle = Annotated[str, StringConstraints(pattern=r"^M[1-9][0-9]*$")]
Statement = Annotated[str, Field(min_length=1, max_length=MAX_STATEMENT_CHARS)]
WireText = Annotated[str, Field(min_length=1)]

_BATCH_ORDER: tuple[SourceBatch, ...] = ("previous", "tail", "current")
_DAY_MS = 86_400_000
_HOUR_MS = 3_600_000


def _civil_date(days: int) -> tuple[int, int, int]:
    """Return the proleptic Gregorian (year, month, day) of a day count from 1970-01-01."""
    shifted = days + 719_468
    era = shifted // 146_097
    day_of_era = shifted - era * 146_097
    year_of_era = (
        day_of_era - day_of_era // 1_460 + day_of_era // 36_524 - day_of_era // 146_096
    ) // 365
    day_of_year = day_of_era - (365 * year_of_era + year_of_era // 4 - year_of_era // 100)
    month_index = (5 * day_of_year + 2) // 153
    day = day_of_year - (153 * month_index + 2) // 5 + 1
    month = month_index + 3 if month_index < 10 else month_index - 9
    return year_of_era + era * 400 + (1 if month <= 2 else 0), month, day


def render_wire_date(value: int | None, granularity: TimeGranularity) -> str | None:
    """Render one stored time at its own granularity, as B.5 and C.6.2 show dates.

    Args:
        value: Epoch milliseconds (any signed 64-bit value), or None when absent.
        granularity: The stored granularity of that value.

    Returns:
        str | None: UTC ``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD`` or ``YYYY-MM-DDTHHZ``
            for YEAR, MONTH, DAY or HOUR; a year outside 0..9999 takes the signed
            ISO 8601 expanded form (``-0753``). None when the value is absent or
            its granularity is ``unanchored``.
    """
    if value is None or granularity == "unanchored":
        return None
    days, remainder = divmod(value, _DAY_MS)
    year, month, day = _civil_date(days)
    text = f"{year:04d}" if 0 <= year <= 9999 else f"{year:+05d}"
    if granularity == "YEAR":
        return text
    text += f"-{month:02d}"
    if granularity == "MONTH":
        return text
    text += f"-{day:02d}"
    if granularity == "DAY":
        return text
    return f"{text}T{remainder // _HOUR_MS:02d}Z"


class ExtractSourceV12(StrictModel):
    """One rendered source: a run of one message's parts with one role and speaker (B.4.1).

    ``speaker_handle`` is ``user`` for an authenticated-self source and the
    participant's E handle for a tuple or labelled speaker, else null. A source
    is fresh exactly when it belongs to the unit's own batch (``current``).
    ``role`` is the message role, except ``participant`` for a labelled
    participant's turn behind an envelope header: an envelope sends every
    speaker's line under a role of its own choosing, which says nothing about
    who speaks it.
    """

    source_id: SourceId
    fresh: bool
    batch: SourceBatch
    role: SourceRole
    speaker: WireText | None
    speaker_handle: EntityHandle | None
    observed_at: WireDate | None
    text: WireText


class ExtractNameV12(StrictModel):
    """One readable name of a catalog entity, from a live naming memory (P-NAME)."""

    name: Annotated[str, Field(min_length=1, max_length=MAX_NAME_CHARS)]
    kind: NameKind


class ExtractEntityV12(StrictModel):
    """One catalog entry (B.4.2); ``speaker_names`` bind speakers and are not entity names."""

    handle: EntityHandle
    kind: EntityKind
    type: Annotated[str, Field(min_length=1, max_length=MAX_ENTITY_TYPE_CHARS)]
    names: Annotated[list[ExtractNameV12], Field(max_length=MAX_CATALOG_NAMES)]
    speaker_names: list[WireText]
    memories: list[MemoryHandle]


class ExtractBindingV12(StrictModel):
    """One SDK entity binding over text of a rendered source."""

    source_id: SourceId
    text: WireText
    handle: EntityHandle


class ExtractEarlierV12(StrictModel):
    """One readable earlier version of a context memory (PD:P8), handle M41 or above."""

    handle: MemoryHandle
    statement: Statement
    valid_from: WireDate | None
    valid_to: WireDate | None


class ExtractMemoryV12(StrictModel):
    """One context memory; an anchor without a catalog handle is null, primary first."""

    handle: MemoryHandle
    statement: Statement
    kind: MemoryKind
    valid_from: WireDate | None
    valid_to: WireDate | None
    anchors: Annotated[list[EntityHandle | None], Field(min_length=1, max_length=MAX_ANCHORS)]
    earlier: Annotated[
        list[ExtractEarlierV12], Field(max_length=MAX_EARLIER_PER_MEMORY),
    ]


class IssuedExtractHandlesV12(StrictModel):
    """The ids and handles one request issues, in issue order: its schema enums."""

    source_ids: Annotated[list[str], Field(min_length=1)]
    entity_handles: list[str]
    memory_handles: list[str]


class ExtractRequestV12(StrictModel):
    """The user message of one extract-v12 request (B.5).

    Sources come in the order previous batch, tail, fresh, numbered ``S1..Sn``;
    the catalog lists ``user`` first when issued, then ``E1..Ek``; context
    memories are ``M1..Mm`` and their earlier versions ``M41`` upwards in memory
    order. ``catalog_truncated`` tells the model that the catalog dropped
    entities (B.4.2), which the template needs to keep a definite description
    from binding to the only listed fit (UD5).
    """

    sources: Annotated[list[ExtractSourceV12], Field(min_length=1, max_length=MAX_SOURCES)]
    entities: Annotated[list[ExtractEntityV12], Field(max_length=MAX_CATALOG_ENTITIES + 1)]
    bindings: list[ExtractBindingV12]
    memories: Annotated[list[ExtractMemoryV12], Field(max_length=MAX_CONTEXT_MEMORIES)]
    catalog_truncated: bool

    @model_validator(mode="after")
    def issued_in_order(self) -> Self:
        """Require B.4's order and numbering and resolve every handle the message uses."""
        sources = [source.source_id for source in self.sources]
        if sources != [f"S{number}" for number in range(1, len(sources) + 1)]:
            raise ValueError("Sources are numbered S1..Sn in order")
        batches = [source.batch for source in self.sources]
        if batches != sorted(batches, key=_BATCH_ORDER.index) or any(
            source.fresh != (source.batch == "current") for source in self.sources
        ):
            raise ValueError("Sources list the previous batch, the tail, then the fresh sources")
        if not 1 <= batches.count("current") <= MAX_FRESH_SOURCES or (
            batches.count("tail") > MAX_TAIL_SOURCES
        ):
            raise ValueError(
                f"A request has 1..{MAX_FRESH_SOURCES} fresh sources and at most "
                f"{MAX_TAIL_SOURCES} tail sources"
            )
        entities = [entity.handle for entity in self.entities]
        self_issued = bool(entities) and entities[0] == "user"
        numbered = entities[1:] if self_issued else entities
        if numbered != [f"E{number}" for number in range(1, len(numbered) + 1)] or any(
            (entity.handle == "user") != (entity.kind == "user") for entity in self.entities
        ) or len(numbered) > MAX_CATALOG_ENTITIES:
            raise ValueError(f"The catalog lists user first, then E1..E{MAX_CATALOG_ENTITIES}")
        context = [memory.handle for memory in self.memories]
        earlier = [version.handle for memory in self.memories for version in memory.earlier]
        # Earlier versions are numbered after the last possible context memory.
        first_earlier = MAX_CONTEXT_MEMORIES + 1
        if context != [f"M{number}" for number in range(1, len(context) + 1)] or earlier != [
            f"M{number}" for number in range(first_earlier, first_earlier + len(earlier))
        ]:
            raise ValueError(
                f"Memories are M1..Mm and their earlier versions M{first_earlier} upwards"
            )
        issued, sourced = {*entities}, {*sources}
        if any(
            source.speaker_handle is not None and source.speaker_handle not in issued
            for source in self.sources
        ) or any(
            binding.source_id not in sourced or binding.handle not in issued
            for binding in self.bindings
        ) or any(
            anchor is not None and anchor not in issued
            for memory in self.memories for anchor in memory.anchors
        ) or any(
            len(set(entity.memories)) != len(entity.memories)
            or not set(entity.memories) <= set(context)
            for entity in self.entities
        ):
            raise ValueError("Every speaker, binding, anchor and memory handle is issued")
        return self

    def issued(self) -> IssuedExtractHandlesV12:
        """Return the source ids and handles this message issues, in issue order."""
        return IssuedExtractHandlesV12(
            source_ids=[source.source_id for source in self.sources],
            entity_handles=[entity.handle for entity in self.entities],
            memory_handles=[memory.handle for memory in self.memories] + [
                version.handle for memory in self.memories for version in memory.earlier
            ],
        )


EXTRACTION_TEMPLATES: Final[Mapping[IntegrityDigest, str]] = MappingProxyType({
    WORKFLOW_TEMPLATE_V50_DIGEST: WORKFLOW_EXTRACTION_TEMPLATE_V50,
})
"""Every extraction template a policy may name, by its pinned digest (B.0)."""


def extraction_messages(
    request: ExtractRequestV12, template: IntegrityDigest = WORKFLOW_TEMPLATE_V50_DIGEST,
) -> tuple[str, str]:
    """Return the system and user message texts of one extract-v12 call (B.5).

    Args:
        request: The frozen user message content of the call.
        template: The owner policy's ``prompt_template_digest``; template v50 by default.

    Returns:
        tuple[str, str]: The template exactly, and the canonical JSON of ``request``.

    Raises:
        KeyError: If ``template`` is not in ``EXTRACTION_TEMPLATES``.
    """
    return EXTRACTION_TEMPLATES[template], canonical_json(request)


def _enum(values: Sequence[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _text(minimum: int, maximum: int) -> dict[str, Any]:
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _array(items: dict[str, Any], maximum: int, minimum: int | None = None) -> dict[str, Any]:
    bounds = {} if minimum is None else {"minItems": minimum}
    return {"type": "array", **bounds, "maxItems": maximum, "items": items}


def _closed(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False, "required": list(properties),
        "properties": properties,
    }


def _ref(name: str) -> dict[str, Any]:
    return {"$ref": f"#/$defs/{name}"}


def extract_v12_schema(issued: IssuedExtractHandlesV12) -> dict[str, Any]:
    """Build the advertised strict JSON Schema of B.6.1 for one request's issued handles.

    Every property is required, nullable fields use ``anyOf`` with null, every
    object is closed, and nothing has a default or a ``pattern``. Properties are
    listed evidence first and declarations before facts, so strict decoding
    copies quotes before it writes a statement and declares handles before use.

    Args:
        issued: The source ids and handles the request's user message issues.

    Returns:
        dict[str, Any]: The schema; ``revises`` admits no item when no memory
            handle is issued.
    """
    entities = [*issued.entity_handles, *NEW_ENTITY_HANDLES]
    revises = (
        _array(_enum(issued.memory_handles), MAX_REVISES) if issued.memory_handles
        else {"type": "array", "maxItems": 0, "items": {"type": "string"}}
    )
    return {
        "type": "object", "additionalProperties": False,
        "required": ["new_entities", "unresolved_referents", "facts"],
        "properties": {
            "new_entities": _array(_ref("NewEntity"), MAX_NEW_ENTITIES),
            "unresolved_referents": _array(_ref("UnresolvedReferent"), MAX_UNRESOLVED_REFERENTS),
            "facts": _array(_ref("Fact"), MAX_REPLY_FACTS),
        },
        "$defs": {
            "NewEntity": _closed({
                "handle": _enum(NEW_ENTITY_HANDLES),
                "name": _nullable(_text(1, MAX_NAME_CHARS)),
                "descriptor": _text(1, MAX_DESCRIPTOR_CHARS),
                "type": _text(1, MAX_ENTITY_TYPE_CHARS),
            }),
            "UnresolvedReferent": _closed({
                "handle": _enum(UNRESOLVED_HANDLES),
                "mention": _text(1, MAX_MENTION_CHARS),
                "reason": _enum(["ambiguous", "insufficient_context"]),
                "candidates": _array(_enum(entities), MAX_REFERENT_CANDIDATES),
            }),
            "Quote": _closed({
                "source_id": _enum(issued.source_ids),
                "quote": _text(1, MAX_QUOTE_CHARS),
            }),
            "SupportGroup": _closed({"quotes": _array(_ref("Quote"), MAX_GROUP_QUOTES, 1)}),
            "Time": _closed({
                "kind": _enum(["none", "absolute", "relative"]),
                "text": _nullable(_text(1, MAX_TIME_TEXT_CHARS)),
            }),
            "Naming": _closed({
                "entity": _enum(entities),
                "name": _text(1, MAX_NAME_CHARS),
                "name_kind": _enum(["name", "alias", "former"]),
            }),
            "SensitivityFinding": _closed({
                "category": _enum(SENSITIVITY_CATEGORIES),
                "quote": _text(1, MAX_QUOTE_CHARS),
            }),
            "Fact": _closed({
                "support_groups": _array(_ref("SupportGroup"), MAX_SUPPORT_GROUPS, 1),
                "anchors": _array(_enum([*entities, *UNRESOLVED_HANDLES]), MAX_ANCHORS, 1),
                "statement": _text(1, MAX_STATEMENT_CHARS),
                "kind": _enum(["state", "event"]),
                "time": _ref("Time"),
                "naming": _array(_ref("Naming"), MAX_NAMING),
                "revises": revises,
                "corrects_text": _nullable(_text(1, MAX_CORRECTS_TEXT_CHARS)),
                "keywords": _array(_text(1, MAX_KEYWORD_CHARS), MAX_KEYWORDS, 1),
                "sensitivity": _array(_ref("SensitivityFinding"), MAX_SENSITIVITY_FINDINGS),
            }),
        },
    }


class _Message(Protocol):
    @property
    def content(self) -> str: ...


class AdvertisedRequest(Protocol):
    """What an output contract reads from a provider request: its messages, user last."""

    @property
    def messages(self) -> Sequence[_Message]: ...


def advertise_extract_v12(request: AdvertisedRequest) -> dict[str, Any]:
    """Advertise the strict extract-v12 schema of one actual request (B.6.1).

    The output registry calls this for ``("extract", "extract-v12")``; the enums
    come from the request's own user message, never from a second copy.

    Args:
        request: The provider request whose last message is the user message.

    Returns:
        dict[str, Any]: The request's strict schema.

    Raises:
        AMLError: binding_mismatch if the user message is not an ``ExtractRequestV12``.
    """
    try:
        payload = ExtractRequestV12.model_validate_json(request.messages[-1].content)
    except (IndexError, ValidationError) as error:
        raise AMLError("binding_mismatch", "Extraction request lost its issued handles") from error
    return extract_v12_schema(payload.issued())


class NewEntityV12(StrictModel):
    """A declared new entity as returned; its bounds are admission rules (B.7.2)."""

    handle: str
    name: str | None
    descriptor: str
    type: str


class UnresolvedReferentV12(StrictModel):
    """A declared unresolved referent as returned (B.7.2)."""

    handle: str
    mention: str
    reason: str
    candidates: list[str]


class QuoteV12(StrictModel):
    """One quote of one issued source, located by the program (B.11)."""

    source_id: Annotated[str, OpaqueIdSpec("local")]
    quote: str


class SupportGroupV12(StrictModel):
    """One alternative set of quotes that together state the whole fact."""

    quotes: list[QuoteV12]


class FactTimeV12(StrictModel):
    """The fact's own source time expression; the text-null rule is admission's."""

    kind: str
    text: str | None


class NamingV12(StrictModel):
    """One naming annotation as returned (B.6.3)."""

    entity: str
    name: str
    name_kind: str


class SensitivityFindingV12(StrictModel):
    """One model sensitivity finding; it raises protection and never lowers a floor."""

    category: str
    quote: str


class FactV12(StrictModel):
    """One fact position of the reply, in the strict schema's field order."""

    support_groups: list[SupportGroupV12]
    anchors: list[str]
    statement: str
    kind: str
    time: FactTimeV12
    naming: list[NamingV12]
    revises: list[str]
    corrects_text: str | None
    keywords: list[str]
    sensitivity: list[SensitivityFindingV12]


class ExtractOutputV12(StrictModel):
    """The verbatim extract-v12 reply, checked for structure only (B.6.2).

    Exact keys and value types, at most 32 declarations of each kind and 512
    facts, and a canonical size of at most 200,000 characters. Everything else is
    an item rule of admission, which keeps this reply verbatim.
    """

    new_entities: Annotated[list[NewEntityV12], Field(max_length=MAX_NEW_ENTITIES)]
    unresolved_referents: Annotated[
        list[UnresolvedReferentV12], Field(max_length=MAX_UNRESOLVED_REFERENTS),
    ]
    facts: Annotated[list[FactV12], Field(max_length=MAX_REPLY_FACTS)]
    supplement_from: Annotated[int, Field(ge=0)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """The position of the first fact that the extraction supplement call returned
    (``ingestion/extract.merged_reply``), so admission marks it and every later fact
    ``extraction_supplement``; None (left out, as in every provider reply) when the
    reply holds the extraction call's facts alone. The strict schema never advertises
    it and the decoder refuses a provider reply that carries it."""
    rollup_from: Annotated[int, Field(ge=0)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """The position of the session rollup's fact (``core/session_rollup.py``), which
    follows the supplement's facts, so admission marks it ``session_rollup``; None when
    the reply holds no rollup. Like ``supplement_from`` it is never advertised and the
    decoder refuses a provider reply that carries it."""

    @model_validator(mode="after")
    def bounded_reply(self) -> Self:
        """Refuse an overlarge whole reply instead of keeping a prefix of it."""
        canonical_limit(self, OUTPUT_CANONICAL_CHARS, "provider output")
        if self.supplement_from is not None and self.supplement_from >= len(self.facts):
            raise ValueError("A supplement boundary must precede at least one fact")
        if self.rollup_from is not None and not (
            self.supplement_from is not None and self.supplement_from < self.rollup_from
            and self.rollup_from == len(self.facts) - 1
        ):
            raise ValueError("A rollup boundary marks the last fact, after the supplement's")
        return self


_TOP_LEVEL_BOUNDS = {
    "new_entities": MAX_NEW_ENTITIES,
    "unresolved_referents": MAX_UNRESOLVED_REFERENTS,
    "facts": MAX_REPLY_FACTS,
}


class ExtractReplyRefused(AMLError):
    """A whole-output fault of one extract-v12 reply (B.7.1).

    Its code is ``schema_invalid``, the provider boundary's output-schema
    refusal; ``fault`` names the B.7.1 fault the refusal records.

    Attributes:
        fault: ``provider_schema_invalid`` (not JSON, a duplicate key or a wrong
            structure) or ``output_bound`` (over 200,000 canonical characters or a
            top-level array beyond its bound).
    """

    def __init__(self, fault: ExtractReplyFault, message: str) -> None:
        """Create the refusal with its fault and a payload-free explanation."""
        super().__init__("schema_invalid", message)
        self.fault: ExtractReplyFault = fault


def _distinct_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError("Duplicate JSON object key")
        decoded[key] = value
    return decoded


def _finite_only(constant: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON number {constant}")


def _beyond_output_bound(raw: object) -> bool:
    if len(canonical_json(raw)) > OUTPUT_CANONICAL_CHARS:
        return True
    fields: dict[str, object] = cast(dict[str, object], raw) if isinstance(raw, dict) else {}
    return any(
        isinstance(value := fields.get(key), list) and len(cast(list[object], value)) > bound
        for key, bound in _TOP_LEVEL_BOUNDS.items()
    )


def decode_extract_reply(content: str) -> ExtractOutputV12:
    """Decode the content of one extract-v12 reply, refusing only whole-output faults.

    ``truncated`` (a finish reason other than stop) is decided before the content
    is read, by the provider boundary.

    Args:
        content: ``choices[0].message.content`` of an HTTP 2xx reply.

    Returns:
        ExtractOutputV12: The structurally valid verbatim reply.

    Raises:
        ExtractReplyRefused: ``provider_schema_invalid`` for content that is not
            standard JSON, repeats an object key or has the wrong structure;
            ``output_bound`` for a top-level array beyond its bound or a canonical
            form over 200,000 characters.
    """
    try:
        raw: object = json.loads(
            content, object_pairs_hook=_distinct_keys, parse_constant=_finite_only,
        )
    except (ValueError, RecursionError) as error:
        raise ExtractReplyRefused(
            "provider_schema_invalid", "Extraction reply is not standard JSON with distinct keys",
        ) from error
    if _beyond_output_bound(raw):
        raise ExtractReplyRefused("output_bound", "Extraction reply exceeds its output bound")
    if isinstance(raw, dict) and not {"supplement_from", "rollup_from"}.isdisjoint(
        cast(dict[str, object], raw),
    ):
        raise ExtractReplyRefused(
            "provider_schema_invalid", "Extraction reply has the wrong structure",
        )
    try:
        return ExtractOutputV12.model_validate(raw)
    except ValidationError as error:
        raise ExtractReplyRefused(
            "provider_schema_invalid", "Extraction reply has the wrong structure",
        ) from error
