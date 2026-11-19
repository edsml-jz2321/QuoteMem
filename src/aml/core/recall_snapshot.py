"""Immutable read-model projections, their root (A.6.1, I4) and the Add read view (I1b).

The Writer supplies durable, validated post-images; a `RecallSnapshot` root holds
them at one generation, the global ``commit_seq``. For every registered scope a
root carries:

- ``ScopeFence``: the scope's delivery fence and counters;
- ``SourceRowV2``: every Evidence part, with its withheld ranges and entity
  bindings;
- ``MemoryRowV2``: every Item in every lifecycle, with its current head, the
  head's active links, the earlier revisions a ``revise`` replaced and, for a
  ``historical`` Item, the content-free correction targets of all its revisions;
- ``EntityRowV1`` and ``EntityNameRowV1``: the active entities and the
  ``entity_name`` rows;
- ``HeldEvidenceV1``: the Evidence cited by candidates with ``hold_evidence``.

A root never holds tombstones, candidate content, runs, recall rows or general
artifacts. `SnapshotAddView` serves the `AddReadView` protocol over one root and
the owner-private tombstone index. These values never read storage or providers;
reader lifetime and privacy scrubbing belong to the consuming read model.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields, is_dataclass
from itertools import pairwise
from operator import countOf
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as validated_dataclass

from aml.core.add_read_view import (
    ChainLink,
    EndedBy,
    EntityNameRecord,
    EntityRecord,
    EvidenceRecord,
    HistoryEntry,
    MemoryRecord,
    MessageKey,
    MessageRecord,
    ReadPin,
    TombstoneRecord,
    TombstoneSource,
    VersionRecord,
    WithheldHeadRecord,
    span_sigma,
)
from aml.core.entities import (
    REVISION_REASONS,
    USER,
    ActivationMode,
    EffectiveBasis,
    EntityBindingV1,
    EntityRef,
    EvidenceKind,
    EvidenceState,
    LinkType,
    MemoryLifecycle,
    MemorySnapshotV2,
    NameKind,
    RevisionOp,
    RevisionReason,
    RouteState,
    SourceRole,
    SupersedeRelation,
    WithheldSpanV1,
)
from aml.core.errors import AMLError
from aml.core.gc_safe import IdentityMemo
from aml.core.identity import (
    ULID,
    ArtifactId,
    ContentBindingDigest,
    IntegrityDigest,
    ScopeId,
    canonical_json,
    integrity_digest,
)
from aml.core.readability import Readability
from aml.core.recall_policy import RecallPolicySnapshot, parse_recall_policy_snapshot
from aml.core.semantic import (
    HistoricalTimestamp,
    ScopeKind,
    SecurityLevel,
    ShortText,
    SystemTimestamp,
    TimeGranularity,
    TrustTier,
)
from aml.core.snapshot_memo import decoded_snapshot, remember_snapshot
from aml.core.source_participants import (
    SourceMetadataVersion,
    SourceParticipantId,
    SourceParticipantV1,
)
from aml.core.statement import (
    ERASED_DIGEST,
    MAX_ENTITY_BINDINGS,
    MAX_WITHHELD_SPANS,
    HexDigest,
    is_name,
    memory_tokens,
    name_norm,
    speaker_index,
)

_STRICT = ConfigDict(strict=True, extra="forbid")
_Nonnegative = Annotated[int, Field(ge=0)]
_Positive = Annotated[int, Field(ge=1)]
_Finite = Annotated[float, Field(allow_inf_nan=False)]
_TargetKey = Annotated[str, Field(min_length=1, max_length=64)]
_EntityType = Annotated[str, Field(min_length=1, max_length=64)]
_TIME: TypeAdapter[int] = TypeAdapter(SystemTimestamp)
_ULID: TypeAdapter[ULID] = TypeAdapter(ULID)
_ENTITY_REF: TypeAdapter[str] = TypeAdapter(EntityRef)
_MAX_LAYERS: Final = 32
"""Overlay layers a persistent map keeps before it folds them into one."""

ReviseEnd = Literal["supplement", "correct", "retract"]
"""How a revision a ``revise`` replaced ended (C.1 history entry)."""

REVISE_ENDS: Final[Mapping[str, ReviseEnd]] = MappingProxyType({
    "supplement": "supplement", "correction": "correct", "retraction": "retract",
    "rewording": "correct",
})
"""How a ``revise`` reason ends the revision it replaces (C.1, C.6.2). A rewording (C.9.4a)
replaced a wording whose description a correction made untrue, so its history entry reads
as corrected."""

_LIFECYCLE_ENDS: Final[Mapping[str, EndedBy]] = MappingProxyType({
    "superseded": "update", "corrected": "correct", "retracted": "retract",
})
"""How an ended Item's lifecycle names the relation that ended it."""

_NAME_KIND_ORDER: Final[Mapping[str, int]] = MappingProxyType(
    {"name": 0, "alias": 1, "former": 2},
)


def _parsed_snapshot(snapshot_json: str, content_hash: str) -> MemorySnapshotV2:
    """Decode a stored snapshot and prove its canonical bytes and digest (A.6.1 parity)."""
    try:
        snapshot = MemorySnapshotV2.model_validate_json(snapshot_json)
    except ValidationError as error:
        raise ValueError("Memory projection holds an invalid snapshot") from error
    if canonical_json(snapshot) != snapshot_json or integrity_digest(snapshot) != content_hash:
        raise ValueError("Memory projection has divergent snapshot bytes")
    # Every view of this head decodes the same text; keep this proven decoding.
    remember_snapshot(snapshot_json, snapshot)
    return snapshot


def _snapshot_json(snapshot: Mapping[str, Any]) -> str:
    """Return the canonical bytes of a stored revision's snapshot."""
    return canonical_json(MemorySnapshotV2.model_validate(snapshot))


def _snapshot_in_space(snapshot_json: str, space: str) -> bool:
    """Whether a canonical snapshot's ``embedding_space`` is ``space``, without decoding it.

    Canonical JSON escapes every quote inside a string value, so the unescaped
    key-value pair can only be the snapshot's own top-level field.
    """
    return f'"embedding_space":{json.dumps(space, ensure_ascii=False)}' in snapshot_json


class CarriedRows:
    """Each projection's canonical identity encoding, carried from one scope view to the next.

    A scope view's root identity digests every row of its scope (``scope_view``),
    and every Add's finalize restricts the root again. A row unchanged since the
    last view is the very same object in the persistent maps, so its encoding,
    ``canonical_json(_identity_value(row))`` in UTF-8, is served under its
    identity instead of being dumped and formatted again. Each entry holds its
    row, so no other object can take over its ``id`` while the entry lives; a
    view keeps exactly the entries it used, so a row that left the scope or
    was replaced is dropped by the next view. The digest stays byte for byte
    the one ``integrity_digest`` takes of the whole value.
    """

    __slots__ = ("_current", "_previous")

    def __init__(self) -> None:
        self._previous: dict[int, tuple[object, bytes]] = {}
        self._current: dict[int, tuple[object, bytes]] = {}

    def begin(self) -> None:
        """Start a view: the last view's encodings are served, then dropped unless used."""
        self._previous, self._current = self._current, {}

    def end(self) -> None:
        """End a view, keeping only the encodings it used."""
        self._previous = {}

    def encoded(self, row: object) -> bytes:
        """Return the row's canonical identity encoding, formatting it only when new."""
        key = id(row)
        entry = self._current.get(key)
        if entry is None:
            entry = self._previous.get(key)
            if entry is None or entry[0] is not row:
                entry = (row, canonical_json(_identity_value(row)).encode("utf-8"))
            self._current[key] = entry
        return entry[1]


_ROWS: ContextVar[CarriedRows | None] = ContextVar("aml_carried_rows", default=None)


@contextmanager
def carried_rows(carried: CarriedRows) -> Iterator[None]:
    """Serve root identity digests built inside the block from ``carried`` encodings.

    Args:
        carried: The encodings the caller carries between its scope views.

    Yields:
        None: The block runs with the carrier active.
    """
    carried.begin()
    token = _ROWS.set(carried)
    try:
        yield
    finally:
        _ROWS.reset(token)
        carried.end()


def _root_identity(
    value: dict[str, object], rows: Mapping[str, Sequence[object]],
) -> IntegrityDigest:
    """Return ``integrity_digest`` of ``value`` plus each of ``rows`` as its identity values.

    Inside ``carried_rows`` each row's encoding comes from the carrier and the
    digest is fed piece by piece in the canonical encoder's order (sorted keys,
    ``,`` and ``:`` separators), so it equals the digest of the whole value.
    """
    carried = _ROWS.get()
    if carried is None:
        return integrity_digest({**value, **{
            key: [_identity_value(row) for row in entries] for key, entries in rows.items()
        }})
    digest = hashlib.sha256()
    separator = b"{"
    for key in sorted((*value, *rows)):
        digest.update(separator)
        digest.update(canonical_json(key).encode("utf-8"))
        digest.update(b":")
        if key in rows:
            digest.update(b"[" + b",".join(carried.encoded(row) for row in rows[key]) + b"]")
        else:
            digest.update(canonical_json(value[key]).encode("utf-8"))
        separator = b","
    digest.update(b"}")
    return IntegrityDigest(digest.hexdigest())


def _identity_value(value: object) -> object:
    """Return the JSON value of a projection for a root identity digest."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _identity_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple | list):
        entries = cast(Sequence[object], value)
        if countOf(map(type, entries), float) == len(entries):
            # A vector: each float is its own value, so it is not visited one by one.
            return list(entries)
        return [_identity_value(entry) for entry in entries]
    return value


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class SourceParticipantProjection:
    """Detached participant metadata with JSON-safe dataclass digest projection."""

    namespace: SourceParticipantId
    conversation_id: SourceParticipantId
    participant_id: SourceParticipantId
    schema_version: SourceMetadataVersion = 1


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class SourceRowV2:
    """One Evidence part as the read model holds it (A.6.1 ``SourceRowV2``).

    It is the Evidence projection shared by source routes, hydration and the
    Add read view: the reception headers, text and vector, plus the part's
    ``withheld_spans`` and ``entity_bindings`` (A.3.5, store-Q15). Offsets are
    message coordinates. An erased part keeps its placeholders, with no
    withheld span and no binding.
    """

    object_id: ULID
    scope_kind: ScopeKind
    scope_id: ScopeId
    kind: EvidenceKind
    session_id: ShortText
    request_id: ShortText
    ordinal: _Nonnegative
    part_no: _Nonnegative
    role: SourceRole | None
    source_observed_at: HistoricalTimestamp | None
    ingested_at: SystemTimestamp
    ingested_commit_seq: _Nonnegative
    byte_start: _Nonnegative
    byte_end: _Nonnegative
    time_granularity: TimeGranularity
    trust_tier: TrustTier
    security_level: SecurityLevel
    state: EvidenceState
    route_state: RouteState
    expires_at: SystemTimestamp | None
    content_hash: IntegrityDigest
    embedding_space: ShortText
    embedding: tuple[_Finite, ...]
    text: str
    withheld_spans: tuple[WithheldSpanV1, ...]
    entity_bindings: tuple[EntityBindingV1, ...]
    speaker_entity: ShortText | None = None
    source_participant: SourceParticipantProjection | None = None

    def __post_init__(self) -> None:
        """Keep the vector dimension, the part range and its spans and bindings coherent."""
        if len(self.embedding) not in {1024, 2048} or self.byte_end < self.byte_start:
            raise ValueError("Source projection has an invalid vector or byte range")
        if self.state == "erased" and (self.withheld_spans or self.entity_bindings):
            raise ValueError("An erased part has no withheld span and no binding")
        if len(self.withheld_spans) > MAX_WITHHELD_SPANS or (
            len(self.entity_bindings) > MAX_ENTITY_BINDINGS
        ):
            raise ValueError("Source projection exceeds its withheld-span or binding bound")
        order = [
            (span.byte_start, span.byte_end, span.reason, span.run_id)
            for span in self.withheld_spans
        ]
        if order != sorted(order):
            raise ValueError("Withheld spans are sorted by start, end, reason and run")
        for value in (*self.withheld_spans, *self.entity_bindings):
            if not self.byte_start <= value.byte_start < value.byte_end <= self.byte_end:
                raise ValueError("A withheld span or binding lies outside its part")

    @classmethod
    def from_evidence(cls, row: Mapping[str, Any]) -> Self:
        """Copy a validated Evidence row without retaining mutable caller buffers.

        Args:
            row: The complete decoded Evidence post-image supplied by the Writer.

        Returns:
            A frozen projection with a tuple vector, its withheld spans and bindings.
        """
        values = {
            field.name: row[field.name]
            for field in fields(cls)
            if field.name not in {"source_participant", "withheld_spans", "entity_bindings"}
        }
        participant = row.get("source_participant")
        if participant is not None:
            validated = SourceParticipantV1.model_validate(
                asdict(participant)
                if isinstance(participant, SourceParticipantProjection) else participant
            )
            values["source_participant"] = SourceParticipantProjection(
                namespace=validated.namespace,
                conversation_id=validated.conversation_id,
                participant_id=validated.participant_id,
                schema_version=validated.schema_version,
            )
        values["embedding"] = tuple(values["embedding"])
        values["withheld_spans"] = tuple(
            WithheldSpanV1.model_validate(span) for span in row["withheld_spans"]
        )
        bindings = row["entity_bindings"]
        values["entity_bindings"] = () if bindings is None else tuple(
            EntityBindingV1.model_validate(binding) for binding in bindings
        )
        return cls(**values)


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class SourceDeletion:
    """A tombstone that masks every older source version with this identity."""

    object_id: ULID


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class MemoryLinkV2:
    """One active link of an Item's current head (A.3.3); the head is its source.

    ``target_key`` is a memory ULID (``supersedes``, ``contradicts``), the
    revision ULID of the corrected version (``corrects_history``) or an
    `EntityRef` (``about_entity``).
    """

    object_id: ULID
    link_type: LinkType
    target_key: _TargetKey
    relation: SupersedeRelation | None
    effective_at: HistoricalTimestamp | None
    effective_basis: EffectiveBasis | None
    verdict_ref: ArtifactId | None

    def __post_init__(self) -> None:
        """Apply the A.3.3 field rules of each link type."""
        if (self.relation is None) != (self.link_type != "supersedes"):
            raise ValueError("A relation is set exactly on a supersedes link")
        if (self.effective_at is None) != (self.effective_basis is None) or (
            self.effective_at is not None and self.relation != "update"
        ):
            raise ValueError("effective_at and its basis come together, on an update only")
        if (self.verdict_ref is None) != (self.link_type == "about_entity"):
            raise ValueError("Every link but about_entity records the verdict that decided it")
        try:
            if self.link_type == "about_entity":
                _ENTITY_REF.validate_python(self.target_key)
            else:
                _ULID.validate_python(self.target_key)
        except ValidationError as error:
            raise ValueError("A link target does not match its link type") from error

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Self:
        """Project one committed link row.

        Args:
            row: The link's strict JSON row.

        Returns:
            The link without its scope, source revision and creation time,
            which its Item's projection already carries.
        """
        return cls(**{field.name: row[field.name] for field in fields(cls)})


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class ReplacedVersionV1:
    """An earlier revision of an Item that a ``revise`` replaced (C.1 history entry).

    ``ended_by`` is the reason of the revision that replaced it: supplement,
    correction or retraction. Revisions are immutable, so the entry never
    changes; readers apply P-VER at their own pin.
    """

    revision_id: ULID
    version: _Positive
    op: RevisionOp
    reason_code: RevisionReason
    recorded_at: SystemTimestamp
    commit_seq: _Nonnegative
    content_hash: IntegrityDigest
    snapshot_json: str
    ended_by: ReviseEnd
    reworded: bool = False

    def __post_init__(self) -> None:
        """Prove the snapshot bytes and the revision's op and reason pair."""
        snapshot = _parsed_snapshot(self.snapshot_json, self.content_hash)
        if self.reason_code not in REVISION_REASONS[self.op] or snapshot.erased or (
            (self.op == "create") != (self.version == 1)
        ):
            raise ValueError("A replaced version is a content revision with a valid reason")

    @classmethod
    def from_rows(cls, earlier: Mapping[str, Any], later: Mapping[str, Any]) -> Self:
        """Project the revision a ``revise`` replaced.

        Args:
            earlier: The replaced revision's strict JSON row.
            later: The ``revise`` revision that replaced it.

        Returns:
            The replaced version, ended by the reason of the revise.
        """
        return cls(
            revision_id=earlier["object_id"],
            version=earlier["version"],
            op=earlier["op"],
            reason_code=earlier["reason_code"],
            recorded_at=earlier["recorded_at"],
            commit_seq=earlier["commit_seq"],
            content_hash=earlier["content_hash"],
            snapshot_json=_snapshot_json(earlier["snapshot"]),
            ended_by=REVISE_ENDS[later["reason_code"]],
            reworded=later["reason_code"] == "rewording",
        )


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class MemoryRowV2:
    """One Item in any lifecycle with its current head (A.6.1 ``MemoryRowV2``).

    The Item fields come first; the head revision follows with its parity-checked
    ``snapshot_json``, its ``reason_code`` and the transaction that wrote it
    (``head_commit_seq``). ``links`` are the head's active links, sorted by id.
    ``replaced`` lists the revisions a ``revise`` replaced, by version, for the
    Add read view's history. A ``historical`` Item also carries
    ``corrected_targets``, the target revision ids of every ``corrects_history``
    row of any of its revisions, erased heads included, and ``created_commit_seq``,
    the ``commit_seq`` of its version 1 (D-14); both are content-free.
    """

    object_id: ULID
    scope_kind: ScopeKind
    scope_id: ScopeId
    current_revision_id: ULID
    entity_id: EntityRef | Literal[""]
    statement_digest: HexDigest
    lifecycle_state: MemoryLifecycle
    activation_mode: ActivationMode
    commit_seq: _Nonnegative
    retired_at: SystemTimestamp | None
    created_at: SystemTimestamp
    updated_at: SystemTimestamp
    version: _Positive
    op: RevisionOp
    reason_code: RevisionReason
    recorded_at: SystemTimestamp
    head_commit_seq: _Nonnegative
    content_hash: IntegrityDigest
    snapshot_json: str
    links: tuple[MemoryLinkV2, ...] = ()
    replaced: tuple[ReplacedVersionV1, ...] = ()
    corrected_targets: tuple[ULID, ...] = ()
    created_commit_seq: _Positive | None = None

    def __post_init__(self) -> None:
        """Prove head parity, the Item-head agreement (DM-02) and the link projection."""
        snapshot = _parsed_snapshot(self.snapshot_json, self.content_hash)
        if self.entity_id != snapshot.entity_id or (
            self.statement_digest != snapshot.statement_digest
        ):
            raise ValueError("Item fields differ from their head snapshot")
        if self.reason_code not in REVISION_REASONS[self.op] or (
            (self.op == "create") != (self.version == 1)
            or (self.op == "erase") != snapshot.erased
        ):
            raise ValueError("Memory head has an invalid revision op")
        if (self.retired_at is None) != (self.lifecycle_state == "active") or (
            self.lifecycle_state == "erased" and not snapshot.erased
        ):
            raise ValueError("Memory lifecycle differs from its head")
        self._check_links(snapshot)
        self._check_history()

    def _check_links(self, snapshot: MemorySnapshotV2) -> None:
        identities = [link.object_id for link in self.links]
        if identities != sorted(set(identities)) or len(
            {(link.link_type, link.target_key) for link in self.links},
        ) != len(self.links):
            raise ValueError("Memory links are not unique and sorted by id")
        if snapshot.erased and self.links:
            raise ValueError("An erased head carries no link")
        about = {link.target_key for link in self.links if link.link_type == "about_entity"}
        if about != set(snapshot.about_entity_ids):
            raise ValueError("about_entity links differ from the snapshot")
        corrected = {
            link.target_key for link in self.links if link.link_type == "corrects_history"
        }
        historical = self.lifecycle_state == "historical"
        if (corrected and not historical) or not corrected <= set(self.corrected_targets):
            raise ValueError("Correction links lie outside their targets")
        if historical != bool(self.corrected_targets) or historical != (
            self.created_commit_seq is not None
        ):
            raise ValueError("Only a historical Item carries correction targets")
        if list(self.corrected_targets) != sorted(set(self.corrected_targets)) or (
            self.created_commit_seq is not None and self.created_commit_seq > self.commit_seq
        ):
            raise ValueError("Correction targets are sorted, unique and dated")

    def _check_history(self) -> None:
        versions = [entry.version for entry in self.replaced]
        if versions != sorted(set(versions)) or any(
            version >= self.version for version in versions
        ) or len({entry.revision_id for entry in self.replaced}) != len(self.replaced) or any(
            entry.revision_id == self.current_revision_id for entry in self.replaced
        ):
            raise ValueError("Replaced versions precede the head, once each")

    @classmethod
    def from_rows(
        cls,
        item: Mapping[str, Any],
        chain: Sequence[Mapping[str, Any]],
        links: Callable[[str], Iterable[Mapping[str, Any]]],
    ) -> Self:
        """Project one committed Item from its rows (A.6.1, D-14).

        The cold rebuild and the commit delta both build an Item's projection
        here, so a delta root and a cold root of the same commit agree.

        Args:
            item: The Item's strict JSON row.
            chain: Its revisions' strict JSON rows, from version 1 to its head.
            links: Every stored link row a revision carries, by revision id.

        Returns:
            The Item with its head and the head's links, the versions a
            ``revise`` replaced and, for a ``historical`` Item, the target
            revisions of the ``corrects_history`` rows of all its revisions,
            erased heads included, and the ``commit_seq`` of its version 1.

        Raises:
            AMLError: ``parity_mismatch`` when the chain is not the Item's
                complete version chain ending at its head.
        """
        if not chain or chain[-1]["object_id"] != item["current_revision_id"] or (
            [row["version"] for row in chain] != list(range(1, len(chain) + 1))
            or any(row["memory_id"] != item["object_id"] for row in chain)
            or any(
                later["previous_revision_id"] != earlier["object_id"]
                for earlier, later in pairwise(chain)
            )
        ):
            raise AMLError("parity_mismatch", "A memory projection needs its whole version chain")
        head = chain[-1]
        historical = item["lifecycle_state"] == "historical"
        corrected = {
            link["target_key"]
            for revision in (chain if historical else ())
            for link in links(revision["object_id"])
            if link["link_type"] == "corrects_history"
        }
        return cls(
            object_id=item["object_id"],
            scope_kind=item["scope_kind"],
            scope_id=item["scope_id"],
            current_revision_id=item["current_revision_id"],
            entity_id=item["entity_id"],
            statement_digest=item["statement_digest"],
            lifecycle_state=item["lifecycle_state"],
            activation_mode=item["activation_mode"],
            commit_seq=item["commit_seq"],
            retired_at=item["retired_at"],
            created_at=item["created_at"],
            updated_at=item["updated_at"],
            version=head["version"],
            op=head["op"],
            reason_code=head["reason_code"],
            recorded_at=head["recorded_at"],
            head_commit_seq=head["commit_seq"],
            content_hash=head["content_hash"],
            snapshot_json=_snapshot_json(head["snapshot"]),
            links=tuple(sorted(
                (MemoryLinkV2.from_row(link) for link in links(head["object_id"])),
                key=lambda link: link.object_id,
            )),
            replaced=tuple(
                ReplacedVersionV1.from_rows(earlier, later)
                for earlier, later in pairwise(chain) if later["op"] == "revise"
            ),
            corrected_targets=tuple(sorted(corrected)),
            created_commit_seq=chain[0]["commit_seq"] if historical else None,
        )


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class MemoryDeletion:
    """Remove an Item's projection: it was forgotten or purged."""

    object_id: ULID


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class EntityRowV1:
    """One active entity (A.6.1 ``EntityRowV1``; A.3.6), with its content-free keys.

    ``mint_key`` is kept beside ``participant_key`` because the Add read view
    looks up a minted entity by it (I1b, C.3). An erased entity leaves the
    projection.
    """

    object_id: ULID
    scope_kind: ScopeKind
    scope_id: ScopeId
    entity_type: _EntityType
    participant_key: HexDigest | None
    mint_key: HexDigest | None
    commit_seq: _Nonnegative

    def __post_init__(self) -> None:
        """A participant has no mint key and an extraction entity no participant key."""
        if self.participant_key is not None and self.mint_key is not None:
            raise ValueError("An entity is a participant or a minted entity, not both")

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Self:
        """Project one committed active entity row.

        Args:
            row: The entity's strict JSON row.

        Returns:
            Its content-free projection.

        Raises:
            AMLError: ``parity_mismatch`` when the entity is erased; an erased
                entity leaves the projection (A.3.6).
        """
        if row["state"] != "active":
            raise AMLError("parity_mismatch", "Only an active entity has a projection")
        return cls(**{field.name: row[field.name] for field in fields(cls)})


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class EntityDeletion:
    """Remove an entity's projection: it was erased or purged."""

    object_id: ULID


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class SpentMintV1:
    """An erased extraction-declared entity: its declaration's ``mint_key`` is spent.

    Erasure is terminal (A.3.6) and the entity row keeps its unique
    ``mint_key``, so the declaration can never be minted again. The root keeps
    only that key, beside the active entities, for the Add read view's
    ``spent`` (C.9.6): a slot that names a spent declaration is open, as one
    naming an erased entity is. As a change it also removes the entity's
    projection, as `EntityDeletion` does. Entity rows are never deleted, so a
    spent key stays spent.
    """

    object_id: ULID
    scope_kind: ScopeKind
    scope_id: ScopeId
    mint_key: HexDigest

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Self:
        """Project one committed erased declared entity row.

        Args:
            row: The entity's strict JSON row.

        Returns:
            Its spent key.

        Raises:
            AMLError: ``parity_mismatch`` for an active entity or one without
                a ``mint_key`` (a participant or an SDK entity).
        """
        if row["state"] != "erased" or row["mint_key"] is None:
            raise AMLError("parity_mismatch", "Only an erased declared entity spends its key")
        return cls(**{field.name: row[field.name] for field in fields(cls)})


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class EntityNameRowV1:
    """One ``entity_name`` row (A.3.7); it counts at read time only under P-NAME."""

    object_id: ULID
    scope_kind: ScopeKind
    scope_id: ScopeId
    entity_id: EntityRef
    name: Annotated[str, Field(min_length=1, max_length=128)]
    name_norm: Annotated[str, Field(min_length=1, max_length=256)]
    name_kind: NameKind
    naming_memory_id: ULID
    naming_revision_id: ULID
    annotation_index: Annotated[int, Field(ge=0, le=3)]
    commit_seq: _Nonnegative
    created_at: SystemTimestamp

    def __post_init__(self) -> None:
        """Require a valid name and its exact normal form."""
        if not is_name(self.name) or name_norm(self.name) != self.name_norm:
            raise ValueError("An entity name row holds a valid name and its name_norm")

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Self:
        """Project one committed ``entity_name`` row.

        Args:
            row: The name's strict JSON row.

        Returns:
            The row as the read model holds it.
        """
        return cls(**{field.name: row[field.name] for field in fields(cls)})


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class EntityNameDeletion:
    """Remove an ``entity_name`` row: its naming head changed or left ``active``."""

    object_id: ULID


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class HeldEvidenceV1:
    """The Evidence one pending candidate with ``hold_evidence`` cites (D.6)."""

    candidate_id: ULID
    scope_kind: ScopeKind
    scope_id: ScopeId
    evidence_ids: tuple[ULID, ...]

    def __post_init__(self) -> None:
        """A hold cites at least one Evidence part, sorted and unique."""
        if not self.evidence_ids or list(self.evidence_ids) != sorted(set(self.evidence_ids)):
            raise ValueError("A hold cites sorted, unique Evidence")

    @classmethod
    def from_candidate(cls, row: Mapping[str, Any]) -> Self:
        """Project the hold of one committed candidate row with ``hold_evidence``.

        Args:
            row: The candidate's strict JSON row.

        Returns:
            The Evidence the candidate holds.

        Raises:
            AMLError: ``parity_mismatch`` when the candidate holds nothing.
        """
        if not row["hold_evidence"]:
            raise AMLError("parity_mismatch", "Only a holding candidate has a hold projection")
        return cls(
            candidate_id=row["object_id"], scope_kind=row["scope_kind"],
            scope_id=row["scope_id"], evidence_ids=tuple(row["evidence_ids"]),
        )


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class HeldEvidenceRelease:
    """Release a candidate's hold: it left ``pending`` or its latest review issued no tombstone."""

    candidate_id: ULID


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class FinalizedMarker:
    """The frozen finalized Control projection, valid only at its scope sequence."""

    run_id: ULID
    lease_generation: _Nonnegative
    at: SystemTimestamp
    input_heads_digest: ContentBindingDigest
    finalized_scope_seq: _Nonnegative


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class FinalizeBarrier:
    """Content-free identity of one process-retained held-cycle baseline."""

    run_id: ULID
    lease_generation: _Positive
    writer_epoch: _Positive
    baseline_snapshot_generation: _Nonnegative
    baseline_root_identity: IntegrityDigest


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class FinalizePhase:
    """Actual process-published cycle identity, independent of its content view."""

    state: Literal["preparing", "held"]
    run_id: ULID
    lease_generation: _Positive
    writer_epoch: _Positive


@validated_dataclass(frozen=True, slots=True, config=_STRICT)
class ScopeFence:
    """One exact scope's complete currently implemented delivery fence."""

    scope_kind: ScopeKind
    scope_id: ScopeId
    scope_seq: _Nonnegative
    erase_epoch: _Nonnegative
    consent_epoch: _Nonnegative
    tomb: bool = False
    visibility_barrier: bool = False
    active_erasure: bool = False
    maintenance: bool = False
    quarantined: bool = False
    finalized: FinalizedMarker | None = None
    finalize_barrier: FinalizeBarrier | None = None
    finalize_phase: FinalizePhase | None = None
    held_evidence: tuple[ULID, ...] = ()
    """Evidence a forget-cancelled cohort's deleted rows held until their requests
    settle (D.5; ``evidence_hold:{s}``), sorted and unique."""

    def __post_init__(self) -> None:
        """Outliving holds are sorted and unique."""
        if list(self.held_evidence) != sorted(set(self.held_evidence)):
            raise ValueError("A scope's outliving holds are sorted and unique")

    @property
    def effective_finalized(self) -> FinalizedMarker | None:
        """Ignore a marker whose recorded source generation no longer matches."""
        if (
            self.finalize_barrier is None
            and self.finalize_phase is None
            and self.finalized is not None
            and self.finalized.finalized_scope_seq == self.scope_seq
        ):
            return self.finalized
        return None


@dataclass(frozen=True, slots=True)
class RecallPolicySelection:
    """An immutable policy selection; decoding always returns a detached model."""

    release_ref: ArtifactId
    config_digest: IntegrityDigest
    policy_snapshot_digest: IntegrityDigest
    snapshot_json: str

    def __post_init__(self) -> None:
        """Validate exact policy bytes and digest before a root can retain them."""
        ArtifactId(self.release_ref)
        IntegrityDigest(self.config_digest)
        snapshot = parse_recall_policy_snapshot(self.snapshot_json)
        if (
            canonical_json(snapshot) != self.snapshot_json
            or integrity_digest(snapshot) != self.policy_snapshot_digest
            or snapshot.config_digest != self.config_digest
        ):
            raise AMLError("parity_mismatch", "Snapshot policy bytes and digest differ")

    @classmethod
    def from_snapshot(
        cls,
        release_ref: ArtifactId,
        config_digest: IntegrityDigest,
        policy_snapshot: RecallPolicySnapshot | Mapping[str, Any],
        policy_snapshot_digest: IntegrityDigest,
    ) -> Self:
        """Freeze a strict normalized policy without retaining its mutable model.

        Args:
            release_ref: The current immutable recall-policy release.
            config_digest: The Writer-verified released config digest.
            policy_snapshot: Its complete normalized policy snapshot.
            policy_snapshot_digest: Expected integrity digest of that snapshot.

        Returns:
            A detached immutable selection suitable for atomic root publication.
        """
        frozen = parse_recall_policy_snapshot(policy_snapshot)
        return cls(release_ref, config_digest, policy_snapshot_digest, canonical_json(frozen))

    def policy_snapshot(self) -> RecallPolicySnapshot:
        """Decode a fresh copy for an explicit QueryBundle or another typed consumer."""
        return parse_recall_policy_snapshot(self.snapshot_json)


def _absent(value: object) -> bool:
    """Whether a persistent-map value records only an absence: a deletion or an empty set."""
    return value is None or (isinstance(value, frozenset) and not value)


@dataclass(frozen=True, slots=True)
class _PersistentMap[T]:
    """Immutable overlay heads; only changed keys allocate a new private layer.

    A map folds its layers into one when they exceed ``_MAX_LAYERS``, so a
    lookup walks a bounded chain. Folding drops absences, which mask nothing
    once no older layer remains.
    """

    entries: Mapping[str, T]
    previous: _PersistentMap[T] | None = None
    depth: int = 1

    @classmethod
    def layer(cls, entries: Mapping[str, T], previous: Self | None = None) -> Self:
        if previous is None:
            return cls(MappingProxyType(dict(entries)))
        if previous.depth >= _MAX_LAYERS:
            merged = {key: value for key, value in previous.items() if not _absent(value)}
            for key, value in entries.items():
                if _absent(value):
                    merged.pop(key, None)
                else:
                    merged[key] = value
            return cls(MappingProxyType(merged))
        return cls(MappingProxyType(dict(entries)), previous, previous.depth + 1)

    def get(self, key: str) -> T | None:
        layer: _PersistentMap[T] | None = self
        while layer is not None:
            if key in layer.entries:
                return layer.entries[key]
            layer = layer.previous
        return None

    def items(self) -> Iterator[tuple[str, T]]:
        seen: set[str] = set()
        layer: _PersistentMap[T] | None = self
        while layer is not None:
            for key, value in layer.entries.items():
                if key not in seen:
                    seen.add(key)
                    yield key, value
            layer = layer.previous

    def values(self) -> Iterator[T]:
        for _, value in self.items():
            yield value


@dataclass(frozen=True, slots=True)
class _SourceVersion:
    object_id: str
    generation: int
    row: SourceRowV2 | None


@dataclass(frozen=True, slots=True, weakref_slot=True)
class SourceSegment:
    """An immutable source delta; both lexical and dense consumers share its heads."""

    generation: int
    entries: tuple[_SourceVersion, ...]
    previous: SourceSegment | None = None


@dataclass(frozen=True, slots=True)
class _ScopeMembers:
    """The identities of every projection of one scope."""

    sources: frozenset[str] = frozenset()
    memories: frozenset[str] = frozenset()
    entities: frozenset[str] = frozenset()
    names: frozenset[str] = frozenset()
    held: frozenset[str] = frozenset()
    spent: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _Delta:
    """The scopes one delta root changed, by the corpus each change invalidates (A.6.1)."""

    base: IntegrityDigest
    memory: frozenset[str]
    entity: frozenset[str]
    evidence: frozenset[str]
    visibility: frozenset[str]
    hold: frozenset[str]


def _digest_key(scope_id: str, statement_digest: str) -> str:
    return f"d\x00{scope_id}\x00{statement_digest}"


def _norm_key(scope_id: str, value: str) -> str:
    return f"n\x00{scope_id}\x00{value}"


def _participant_key(scope_id: str, key: str) -> str:
    return f"p\x00{scope_id}\x00{key}"


def _mint_key(scope_id: str, key: str) -> str:
    return f"m\x00{scope_id}\x00{key}"


def _spent_key(scope_id: str, key: str) -> str:
    return f"s\x00{scope_id}\x00{key}"


class _Changes:
    """The scope memberships and secondary-index keys one delta rewrites."""

    def __init__(
        self, members: _PersistentMap[_ScopeMembers], keys: _PersistentMap[frozenset[str]],
    ) -> None:
        self._base_members = members
        self._base_keys = keys
        self.members: dict[str, dict[str, set[str]]] = {}
        self.keys: dict[str, set[str]] = {}

    def member(self, scope_id: str, kind: str) -> set[str]:
        scope = self.members.get(scope_id)
        if scope is None:
            base = self._base_members.get(scope_id) or _ScopeMembers()
            scope = {item.name: set(getattr(base, item.name)) for item in fields(_ScopeMembers)}
            self.members[scope_id] = scope
        return scope[kind]

    def index(self, key: str) -> set[str]:
        values = self.keys.get(key)
        if values is None:
            values = set(self._base_keys.get(key) or ())
            self.keys[key] = values
        return values

    def maps(self) -> tuple[_PersistentMap[_ScopeMembers], _PersistentMap[frozenset[str]]]:
        members = self._base_members if not self.members else _PersistentMap[_ScopeMembers].layer(
            {
                scope_id: _ScopeMembers(**{kind: frozenset(ids) for kind, ids in scope.items()})
                for scope_id, scope in self.members.items()
            },
            self._base_members,
        )
        keys = self._base_keys if not self.keys else _PersistentMap[frozenset[str]].layer(
            {key: frozenset(ids) for key, ids in self.keys.items()}, self._base_keys,
        )
        return members, keys


def _visibility_changed(old: SourceRowV2 | None, new: SourceRowV2 | None) -> bool:
    """Whether an Evidence change can change P-VER or P-NAME of what cites it (A.6.1, read-Q9)."""
    if old is None:
        return False
    if new is None:
        return True
    return (old.state, old.route_state, old.expires_at, old.withheld_spans) != (
        new.state, new.route_state, new.expires_at, new.withheld_spans,
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class RecallSnapshot:
    """One immutable data generation and its co-published Control projections.

    A delta root allocates only its supplied changes. Source enumeration is a
    deterministic baseline over retained segments; it makes no retrieval-speed
    claim. Reader lifetime, privacy scrubbing, and physical segment reclamation
    remain responsibilities of the consuming runtime.
    """

    generation: int
    root_identity: IntegrityDigest
    embedding_space: str
    embedding_token: IntegrityDigest
    recall_policy: RecallPolicySelection
    _scopes: _PersistentMap[ScopeFence]
    _source_heads: _PersistentMap[_SourceVersion]
    source_segments: SourceSegment | None = None
    _memories: _PersistentMap[MemoryRowV2 | None] = _PersistentMap[MemoryRowV2 | None].layer({})
    _entities: _PersistentMap[EntityRowV1 | None] = _PersistentMap[EntityRowV1 | None].layer({})
    _names: _PersistentMap[EntityNameRowV1 | None] = _PersistentMap[EntityNameRowV1 | None].layer(
        {},
    )
    _held: _PersistentMap[HeldEvidenceV1 | None] = _PersistentMap[HeldEvidenceV1 | None].layer({})
    _spent: _PersistentMap[SpentMintV1 | None] = _PersistentMap[SpentMintV1 | None].layer({})
    _members: _PersistentMap[_ScopeMembers] = _PersistentMap[_ScopeMembers].layer({})
    _keys: _PersistentMap[frozenset[str]] = _PersistentMap[frozenset[str]].layer({})
    _delta: _Delta | None = None

    @classmethod
    def rebuild(
        cls,
        *,
        generation: int,
        scopes: Iterable[ScopeFence],
        embedding_space: str,
        embedding_token: IntegrityDigest,
        recall_policy: RecallPolicySelection,
        sources: Iterable[SourceRowV2],
        memories: Iterable[MemoryRowV2] = (),
        entities: Iterable[EntityRowV1 | SpentMintV1] = (),
        names: Iterable[EntityNameRowV1] = (),
        held: Iterable[HeldEvidenceV1] = (),
        identity: IntegrityDigest | None = None,
    ) -> Self:
        """Construct a complete cold root from caller-verified durable state.

        Args:
            generation: The fully rebuilt durable commit sequence.
            scopes: Every registered scope and its same-root delivery fences.
            embedding_space: The effective embedding space.
            embedding_token: The complete embedding Control state token.
            recall_policy: The released, verified normalized recall policy.
            sources: All Evidence projections, including pending metadata.
            memories: Every Item projection, in every lifecycle.
            entities: Every active entity and every spent declaration's key.
            names: Every ``entity_name`` row.
            held: Every candidate hold.
            identity: The root identity, when the caller binds it instead of the
                digest of every row (``search_view``).

        Returns:
            A new immutable root, with pending rows absent from source segments.

        Raises:
            AMLError: If any projection lacks its scope or coherent generation.
        """
        empty = cls(
            0,
            integrity_digest({"empty_snapshot": 2}),
            embedding_space,
            embedding_token,
            recall_policy,
            _PersistentMap[ScopeFence].layer({}),
            _PersistentMap[_SourceVersion].layer({}),
        )
        return empty._apply(
            generation,
            sources=tuple(sources),
            memories=tuple(memories),
            entities=tuple(entities),
            names=tuple(names),
            held=tuple(held),
            scopes=tuple(scopes),
            embedding_space=embedding_space,
            embedding_token=embedding_token,
            policy=recall_policy,
            rebuilding=True,
            identity=identity,
        )

    def apply(
        self,
        *,
        generation: int,
        source_changes: Iterable[SourceRowV2 | SourceDeletion] = (),
        memory_changes: Iterable[MemoryRowV2 | MemoryDeletion] = (),
        entity_changes: Iterable[EntityRowV1 | EntityDeletion | SpentMintV1] = (),
        name_changes: Iterable[EntityNameRowV1 | EntityNameDeletion] = (),
        held_changes: Iterable[HeldEvidenceV1 | HeldEvidenceRelease] = (),
        scope_changes: Iterable[ScopeFence] = (),
        embedding_space: str | None = None,
        embedding_token: IntegrityDigest | None = None,
        recall_policy: RecallPolicySelection | None = None,
    ) -> Self:
        """Build a private delta root without mutating or copying previous layers.

        Args:
            generation: Committed data sequence; data changes require advancement.
            source_changes: Complete Evidence post-images or deletions, one per key.
            memory_changes: Complete Item post-images or deletions, one per key.
            entity_changes: Active-entity post-images, deletions or spent keys, one per key.
            name_changes: ``entity_name`` post-images or deletions, one per key.
            held_changes: Candidate holds or releases, one per candidate.
            scope_changes: Changed complete scope fences, with monotonic counters.
            embedding_space: Replacement effective space, if changed.
            embedding_token: Replacement complete state token, if changed.
            recall_policy: Replacement released selection, if changed.

        Returns:
            The complete candidate root for the Writer's single pointer swap.

        Raises:
            AMLError: If generation, scope, identity, or complete post-state differs.
        """
        return self._apply(
            generation,
            sources=tuple(source_changes),
            memories=tuple(memory_changes),
            entities=tuple(entity_changes),
            names=tuple(name_changes),
            held=tuple(held_changes),
            scopes=tuple(scope_changes),
            embedding_space=self.embedding_space if embedding_space is None else embedding_space,
            embedding_token=self.embedding_token if embedding_token is None else embedding_token,
            policy=self.recall_policy if recall_policy is None else recall_policy,
        )

    def _apply(
        self,
        generation: int,
        *,
        sources: tuple[SourceRowV2 | SourceDeletion, ...],
        memories: tuple[MemoryRowV2 | MemoryDeletion, ...],
        entities: tuple[EntityRowV1 | EntityDeletion | SpentMintV1, ...],
        names: tuple[EntityNameRowV1 | EntityNameDeletion, ...],
        held: tuple[HeldEvidenceV1 | HeldEvidenceRelease, ...],
        scopes: tuple[ScopeFence, ...],
        embedding_space: str,
        embedding_token: IntegrityDigest,
        policy: RecallPolicySelection,
        rebuilding: bool = False,
        identity: IntegrityDigest | None = None,
    ) -> Self:
        sources = tuple(sorted(sources, key=lambda change: change.object_id))
        memories = tuple(sorted(memories, key=lambda change: change.object_id))
        entities = tuple(sorted(entities, key=lambda change: change.object_id))
        names = tuple(sorted(names, key=lambda change: change.object_id))
        held = tuple(sorted(held, key=lambda change: change.candidate_id))
        scopes = tuple(sorted(scopes, key=lambda scope: scope.scope_id))
        if type(generation) is not int or generation < self.generation:
            raise AMLError("parity_mismatch", "Snapshot generation cannot regress")
        if (
            (sources or memories or entities or names or held)
            and generation == self.generation and not rebuilding
        ):
            raise AMLError("parity_mismatch", "Different data post-states need a new generation")
        if not embedding_space:
            raise AMLError("parity_mismatch", "Snapshot requires its effective embedding space")
        if not rebuilding and embedding_space != self.embedding_space:
            raise AMLError(
                "unsupported_variant", "Embedding-space changes require a complete rebuilt root"
            )
        IntegrityDigest(embedding_token)
        scope_heads = self._scope_heads(scopes)
        changes = _Changes(self._members, self._keys)
        touched: dict[str, set[str]] = {
            "memory": set(), "entity": set(), "evidence": set(), "visibility": set(), "hold": set(),
        }

        def require_scope(scope_kind: str, scope_id: str, what: str) -> None:
            scope = scope_heads.get(scope_id)
            if scope is None or scope.scope_kind != scope_kind:
                raise AMLError("scope_denied", f"{what} projection lacks its exact scope fence")

        def same_scope(old: Any, new: Any, what: str) -> None:
            if old is not None and new is not None and (old.scope_kind, old.scope_id) != (
                new.scope_kind, new.scope_id,
            ):
                raise AMLError("scope_denied", f"{what} identity cannot move between scopes")

        heads, segment = self._source_delta(
            sources, generation, embedding_space, require_scope, same_scope, changes, touched,
        )
        memory_updates: dict[str, MemoryRowV2 | None] = {}
        for change in memories:
            if change.object_id in memory_updates:
                raise AMLError("parity_mismatch", "Memory delta contains duplicate identities")
            old = self._memories.get(change.object_id)
            row = change if isinstance(change, MemoryRowV2) else None
            if row is not None:
                require_scope(row.scope_kind, row.scope_id, "Memory")
                same_scope(old, row, "Memory")
                if max(row.commit_seq, row.head_commit_seq) > generation or not all(
                    _snapshot_in_space(snapshot_json, embedding_space)
                    for snapshot_json in (
                        row.snapshot_json, *(entry.snapshot_json for entry in row.replaced),
                    )
                ):
                    raise AMLError("parity_mismatch", "Memory projection belongs to another state")
                changes.member(row.scope_id, "memories").add(row.object_id)
                touched["memory"].add(row.scope_id)
            if old is not None:
                changes.index(_digest_key(old.scope_id, old.statement_digest)).discard(
                    old.object_id,
                )
                touched["memory"].add(old.scope_id)
                if row is None:
                    changes.member(old.scope_id, "memories").discard(old.object_id)
            if row is not None and row.statement_digest != ERASED_DIGEST:
                changes.index(_digest_key(row.scope_id, row.statement_digest)).add(row.object_id)
            memory_updates[change.object_id] = row
        entity_updates: dict[str, EntityRowV1 | None] = {}
        spent_updates: dict[str, SpentMintV1] = {}
        for change in entities:
            if change.object_id in entity_updates:
                raise AMLError("parity_mismatch", "Entity delta contains duplicate identities")
            old = self._entities.get(change.object_id)
            row = change if isinstance(change, EntityRowV1) else None
            if row is not None:
                require_scope(row.scope_kind, row.scope_id, "Entity")
                same_scope(old, row, "Entity")
                if row.commit_seq > generation:
                    raise AMLError("parity_mismatch", "Entity projection belongs to another state")
                changes.member(row.scope_id, "entities").add(row.object_id)
                touched["entity"].add(row.scope_id)
            if isinstance(change, SpentMintV1):
                require_scope(change.scope_kind, change.scope_id, "Entity")
                same_scope(old, change, "Entity")
                if old is not None and old.mint_key != change.mint_key:
                    raise AMLError("parity_mismatch", "An erased entity keeps its mint key")
                changes.member(change.scope_id, "spent").add(change.object_id)
                changes.index(_spent_key(change.scope_id, change.mint_key)).add(change.object_id)
                touched["entity"].add(change.scope_id)
                spent_updates[change.object_id] = change
            if old is not None:
                for key in (
                    None if old.participant_key is None
                    else _participant_key(old.scope_id, old.participant_key),
                    None if old.mint_key is None else _mint_key(old.scope_id, old.mint_key),
                ):
                    if key is not None:
                        changes.index(key).discard(old.object_id)
                touched["entity"].add(old.scope_id)
                if row is None:
                    changes.member(old.scope_id, "entities").discard(old.object_id)
            if row is not None:
                for key in (
                    None if row.participant_key is None
                    else _participant_key(row.scope_id, row.participant_key),
                    None if row.mint_key is None else _mint_key(row.scope_id, row.mint_key),
                ):
                    if key is not None:
                        changes.index(key).add(row.object_id)
            entity_updates[change.object_id] = row
        name_updates: dict[str, EntityNameRowV1 | None] = {}
        for change in names:
            if change.object_id in name_updates:
                raise AMLError("parity_mismatch", "Name delta contains duplicate identities")
            old = self._names.get(change.object_id)
            row = change if isinstance(change, EntityNameRowV1) else None
            if row is not None:
                require_scope(row.scope_kind, row.scope_id, "Name")
                same_scope(old, row, "Name")
                if row.commit_seq > generation:
                    raise AMLError("parity_mismatch", "Name projection belongs to another state")
                changes.member(row.scope_id, "names").add(row.object_id)
                changes.index(_norm_key(row.scope_id, row.name_norm)).add(row.object_id)
                touched["entity"].add(row.scope_id)
            if old is not None:
                if row is None or row.name_norm != old.name_norm:
                    changes.index(_norm_key(old.scope_id, old.name_norm)).discard(old.object_id)
                touched["entity"].add(old.scope_id)
                if row is None:
                    changes.member(old.scope_id, "names").discard(old.object_id)
            name_updates[change.object_id] = row
        held_updates: dict[str, HeldEvidenceV1 | None] = {}
        for change in held:
            if change.candidate_id in held_updates:
                raise AMLError("parity_mismatch", "Hold delta contains duplicate identities")
            old = self._held.get(change.candidate_id)
            row = change if isinstance(change, HeldEvidenceV1) else None
            if row is not None:
                require_scope(row.scope_kind, row.scope_id, "Hold")
                same_scope(old, row, "Hold")
                changes.member(row.scope_id, "held").add(row.candidate_id)
                touched["hold"].add(row.scope_id)
            if old is not None:
                touched["hold"].add(old.scope_id)
                if row is None:
                    changes.member(old.scope_id, "held").discard(old.candidate_id)
            held_updates[change.candidate_id] = row
        members, keys = changes.maps()
        root_identity = identity if identity is not None else _root_identity({
            "previous_root": self.root_identity,
            "generation": generation,
            "embedding_space": embedding_space,
            "embedding_token": embedding_token,
            "recall_policy": asdict(policy),
        }, {
            "sources": sources, "memories": memories, "entities": entities, "names": names,
            "held": held, "scopes": scopes,
        })
        return type(self)(
            generation,
            root_identity,
            embedding_space,
            embedding_token,
            policy,
            scope_heads,
            heads,
            segment,
            self._memories if not memories else _PersistentMap[MemoryRowV2 | None].layer(
                memory_updates, self._memories,
            ),
            self._entities if not entities else _PersistentMap[EntityRowV1 | None].layer(
                entity_updates, self._entities,
            ),
            self._names if not names else _PersistentMap[EntityNameRowV1 | None].layer(
                name_updates, self._names,
            ),
            self._held if not held else _PersistentMap[HeldEvidenceV1 | None].layer(
                held_updates, self._held,
            ),
            self._spent if not spent_updates else _PersistentMap[SpentMintV1 | None].layer(
                dict[str, SpentMintV1 | None](spent_updates), self._spent,
            ),
            members,
            keys,
            None if rebuilding else _Delta(
                base=self.root_identity,
                **{kind: frozenset(scope_ids) for kind, scope_ids in touched.items()},
            ),
        )

    def _scope_heads(self, scopes: tuple[ScopeFence, ...]) -> _PersistentMap[ScopeFence]:
        new_scopes = {scope.scope_id: scope for scope in scopes}
        if len(new_scopes) != len(scopes):
            raise AMLError("parity_mismatch", "Scope delta contains duplicate identities")
        for scope in scopes:
            old = self._scopes.get(scope.scope_id)
            if old is not None and (
                old.scope_kind != scope.scope_kind
                or any(
                    getattr(scope, field) < getattr(old, field)
                    for field in ("scope_seq", "erase_epoch", "consent_epoch")
                )
            ):
                raise AMLError("parity_mismatch", "Snapshot scope identity or counter regressed")
        return _PersistentMap[ScopeFence].layer(new_scopes, self._scopes) if scopes else (
            self._scopes
        )

    def _source_delta(
        self,
        changes: tuple[SourceRowV2 | SourceDeletion, ...],
        generation: int,
        embedding_space: str,
        require_scope: Callable[[str, str, str], None],
        same_scope: Callable[[Any, Any, str], None],
        delta: _Changes,
        touched: dict[str, set[str]],
    ) -> tuple[_PersistentMap[_SourceVersion], SourceSegment | None]:
        versions: dict[str, _SourceVersion] = {}
        indexed: list[_SourceVersion] = []
        for change in changes:
            if change.object_id in versions:
                raise AMLError("parity_mismatch", "Source delta contains duplicate identities")
            old = self.get_source(change.object_id)
            row = change if isinstance(change, SourceRowV2) else None
            if row is not None:
                require_scope(row.scope_kind, row.scope_id, "Source")
                if row.ingested_commit_seq > generation or row.embedding_space != embedding_space:
                    raise AMLError("parity_mismatch", "Source projection belongs to another state")
                same_scope(old, row, "Source")
                delta.member(row.scope_id, "sources").add(row.object_id)
                touched["evidence"].add(row.scope_id)
            if old is not None:
                touched["evidence"].add(old.scope_id)
                if _visibility_changed(old, row):
                    touched["visibility"].add(old.scope_id)
                if row is None:
                    delta.member(old.scope_id, "sources").discard(old.object_id)
            version = _SourceVersion(change.object_id, generation, row)
            versions[change.object_id] = version
            visible = row is not None and row.state == "active" and row.route_state == "indexed"
            prior_visible = (
                old is not None and old.state == "active" and old.route_state == "indexed"
            )
            if visible or prior_visible:
                indexed.append(
                    version if visible else _SourceVersion(change.object_id, generation, None)
                )
        heads = (
            _PersistentMap[_SourceVersion].layer(versions, self._source_heads)
            if changes
            else self._source_heads
        )
        segment = (
            SourceSegment(generation, tuple(indexed), self.source_segments)
            if indexed
            else self.source_segments
        )
        return heads, segment

    def compact(self) -> Self:
        """Detach the latest complete state from every previous layer and segment.

        Args:
            None.

        Returns:
            A same-generation root containing no retired source or memory post-image.

        Raises:
            AMLError: If rebuilding detects inconsistent frozen projection state.
        """
        return type(self).rebuild(
            generation=self.generation,
            scopes=self._scopes.values(),
            embedding_space=self.embedding_space,
            embedding_token=self.embedding_token,
            recall_policy=self.recall_policy,
            sources=(value.row for value in self._source_heads.values() if value.row is not None),
            memories=(row for row in self._memories.values() if row is not None),
            entities=(*(row for row in self._entities.values() if row is not None),
                      *(row for row in self._spent.values() if row is not None)),
            names=(row for row in self._names.values() if row is not None),
            held=(row for row in self._held.values() if row is not None),
        )

    def scope_view(self, scope_id: str) -> Self:
        """This root restricted to one scope: its fence and every projection of it.

        A process-retained finalize baseline serves one scope's owner and that
        scope's held Searches (C.2a). It keeps nothing of another scope, so no
        other scope's erasure, consent change or drain reaches it (D.6). The view
        keeps the generation, policy and embedding identity; its root identity is
        a pure function of what it holds, so the same root always gives the same
        view.

        Args:
            scope_id: An exact scope of this root.

        Returns:
            A complete compact root holding only that scope.

        Raises:
            AMLError: ``scope_denied`` when this root has no such scope.
        """
        return self._restricted(scope_id, None)

    def search_view(self, scope_id: str) -> Self:
        """This root restricted to one scope for that scope's Searches (D.6; SR-1).

        It holds what ``scope_view`` holds, so another scope's privacy drain never
        needs to revoke a Search reading it: it keeps no byte of any other scope.
        Its identity binds this root and the scope instead of digesting every row
        (vectors included): the read model builds it again whenever the scope
        changes, and no stored record carries it.

        Args:
            scope_id: An exact scope of this root.

        Returns:
            A complete compact root holding only that scope.

        Raises:
            AMLError: ``scope_denied`` when this root has no such scope.
        """
        return self._restricted(scope_id, integrity_digest({
            "search_view": self.root_identity, "scope_id": scope_id,
        }))

    def _restricted(self, scope_id: str, identity: IntegrityDigest | None) -> Self:
        return type(self).rebuild(
            generation=self.generation,
            scopes=(self.scope_fence(scope_id),),
            embedding_space=self.embedding_space,
            embedding_token=self.embedding_token,
            recall_policy=self.recall_policy,
            sources=self.scope_sources(scope_id),
            memories=self.memory_rows(scope_id),
            entities=(*self.entity_rows(scope_id), *self.spent_rows(scope_id)),
            names=self.entity_name_rows(scope_id),
            held=self.held_rows(scope_id),
            identity=identity,
        )

    def holds_scope(self, scope_id: str) -> bool:
        """Whether this root projects the scope at all, in any state."""
        return self._scopes.get(scope_id) is not None

    def retains_source_preimage(self, object_id: str) -> bool:
        """Detect an older source body reachable through this root's overlays.

        Args:
            object_id: An affected Evidence identity whose latest image must be exclusive.

        Returns:
            True when a retained head layer or segment contains a different body.

        Raises:
            None.
        """
        current = self.get_source(object_id)
        layer: _PersistentMap[_SourceVersion] | None = self._source_heads
        while layer is not None:
            version = layer.entries.get(object_id)
            if version is not None and version.row is not None and version.row != current:
                return True
            layer = layer.previous
        segment = self.source_segments
        while segment is not None:
            if any(
                version.object_id == object_id
                and version.row is not None
                and version.row != current
                for version in segment.entries
            ):
                return True
            segment = segment.previous
        return False

    def contains_owned_text(
        self,
        scope_id: str,
        forbidden: tuple[bytes, ...],
        *,
        recognizes: Callable[[bytes], bool] | None = None,
    ) -> bool:
        """Inspect retained semantic copies without exporting content from a blocked root.

        Args:
            scope_id: The exact scope being verified.
            forbidden: Nonempty transient byte spans; this function never records them.
            recognizes: Optional code-owned signature detector, including recovery without spans.

        Returns:
            True if any retained source, memory version or name contains a span.

        Raises:
            AMLError: If neither spans nor a detector were supplied, or a span is empty.
        """
        if (not forbidden and recognizes is None) or any(not value for value in forbidden):
            raise AMLError("parity_mismatch", "Content verification requires nonempty spans")

        def matches(value: object) -> bool:
            if isinstance(value, str):
                data = value.encode("utf-8")
                return any(span in data for span in forbidden) or (
                    recognizes is not None and recognizes(data)
                )
            if isinstance(value, list | tuple):
                return any(matches(entry) for entry in cast(Sequence[object], value))
            if isinstance(value, dict):
                return any(matches(entry) for entry in cast(dict[str, object], value).values())
            return False

        def snapshot_matches(snapshot_json: str) -> bool:
            return matches(cast(object, json.loads(snapshot_json)))

        sources: _PersistentMap[_SourceVersion] | None = self._source_heads
        while sources is not None:
            if any(
                value.row is not None
                and value.row.scope_id == scope_id
                and matches(asdict(value.row))
                for value in sources.entries.values()
            ):
                return True
            sources = sources.previous
        segment = self.source_segments
        while segment is not None:
            if any(
                value.row is not None
                and value.row.scope_id == scope_id
                and matches(asdict(value.row))
                for value in segment.entries
            ):
                return True
            segment = segment.previous
        memories: _PersistentMap[MemoryRowV2 | None] | None = self._memories
        while memories is not None:
            for row in memories.entries.values():
                if row is not None and row.scope_id == scope_id and (
                    snapshot_matches(row.snapshot_json)
                    or any(snapshot_matches(entry.snapshot_json) for entry in row.replaced)
                ):
                    return True
            memories = memories.previous
        names: _PersistentMap[EntityNameRowV1 | None] | None = self._names
        while names is not None:
            if any(
                row is not None and row.scope_id == scope_id and matches([row.name, row.name_norm])
                for row in names.entries.values()
            ):
                return True
            names = names.previous
        return False

    def scope_fence(self, scope_id: str) -> ScopeFence:
        """Resolve only the exact frozen scope; unknown scope state fails closed."""
        scope = self._scopes.get(scope_id)
        if scope is None:
            raise AMLError("scope_denied", "Snapshot has no such exact scope")
        return scope

    def _members_of(self, scope_id: str) -> _ScopeMembers:
        self.scope_fence(scope_id)
        return self._members.get(scope_id) or _ScopeMembers()

    def get_source(self, object_id: str) -> SourceRowV2 | None:
        """Read latest metadata, including pending rows; this is not a route query."""
        version = self._source_heads.get(object_id)
        return None if version is None else version.row

    def get_memory(self, object_id: str) -> MemoryRowV2 | None:
        """Read one Item's latest projection in any lifecycle, or None when absent."""
        return self._memories.get(object_id)

    def get_entity(self, object_id: str) -> EntityRowV1 | None:
        """Read one active entity's projection, or None when absent or erased."""
        return self._entities.get(object_id)

    def get_entity_name(self, object_id: str) -> EntityNameRowV1 | None:
        """Read one ``entity_name`` row, or None when absent."""
        return self._names.get(object_id)

    def get_hold(self, candidate_id: str) -> HeldEvidenceV1 | None:
        """Read one candidate's hold, or None when it holds nothing."""
        return self._held.get(candidate_id)

    def scope_sources(self, scope_id: str) -> tuple[SourceRowV2, ...]:
        """Return every Evidence part of a registered scope in any state, by id."""
        rows = (self.get_source(object_id) for object_id in self._members_of(scope_id).sources)
        return tuple(sorted(
            (row for row in rows if row is not None), key=lambda row: row.object_id,
        ))

    def memory_rows(self, scope_id: str) -> tuple[MemoryRowV2, ...]:
        """Return every Item projection of a registered scope in any lifecycle, by id.

        Eligibility is never inferred from presence here; readers apply the
        D.2 predicates. Visibility fences are the reading guard's to check.
        """
        rows = (self._memories.get(object_id) for object_id in self._members_of(scope_id).memories)
        return tuple(sorted(
            (row for row in rows if row is not None), key=lambda row: row.object_id,
        ))

    def entity_rows(self, scope_id: str) -> tuple[EntityRowV1, ...]:
        """Return every active entity of a registered scope, by id."""
        rows = (self._entities.get(object_id) for object_id in self._members_of(scope_id).entities)
        return tuple(sorted(
            (row for row in rows if row is not None), key=lambda row: row.object_id,
        ))

    def spent_rows(self, scope_id: str) -> tuple[SpentMintV1, ...]:
        """Return every spent declaration of a registered scope, by entity id."""
        rows = (self._spent.get(object_id) for object_id in self._members_of(scope_id).spent)
        return tuple(sorted(
            (row for row in rows if row is not None), key=lambda row: row.object_id,
        ))

    def entity_name_rows(self, scope_id: str) -> tuple[EntityNameRowV1, ...]:
        """Return every ``entity_name`` row of a registered scope, by id."""
        rows = (self._names.get(object_id) for object_id in self._members_of(scope_id).names)
        return tuple(sorted(
            (row for row in rows if row is not None), key=lambda row: row.object_id,
        ))

    def held_rows(self, scope_id: str) -> tuple[HeldEvidenceV1, ...]:
        """Return every candidate hold of a registered scope, by candidate id."""
        rows = (self._held.get(object_id) for object_id in self._members_of(scope_id).held)
        return tuple(sorted(
            (row for row in rows if row is not None), key=lambda row: row.candidate_id,
        ))

    def held_evidence(self, scope_id: str) -> frozenset[str]:
        """Return the Evidence ids a pending ``hold_evidence`` candidate cites (D.6), and
        those a forget-cancelled cohort's deleted rows held until their requests settle
        (D.5, ``ScopeFence.held_evidence``)."""
        return frozenset(
            evidence_id for row in self.held_rows(scope_id) for evidence_id in row.evidence_ids
        ) | frozenset(self.scope_fence(scope_id).held_evidence)

    def memories_with_digest(self, scope_id: str, digest: str) -> tuple[MemoryRowV2, ...]:
        """Return the Items whose current head has a statement digest, from the digest index.

        Args:
            scope_id: A registered scope.
            digest: A ``statement_digest``; the erased digest is not indexed.

        Returns:
            The Items in any lifecycle, sorted by id.
        """
        self.scope_fence(scope_id)
        rows = (self._memories.get(object_id) for object_id in (
            self._keys.get(_digest_key(scope_id, digest)) or ()
        ))
        return tuple(sorted(
            (row for row in rows if row is not None), key=lambda row: row.object_id,
        ))

    def names_with_norm(self, scope_id: str, value: str) -> tuple[EntityNameRowV1, ...]:
        """Return the ``entity_name`` rows with a ``name_norm``, from the name index, by id."""
        self.scope_fence(scope_id)
        rows = (self._names.get(object_id) for object_id in (
            self._keys.get(_norm_key(scope_id, value)) or ()
        ))
        return tuple(sorted(
            (row for row in rows if row is not None), key=lambda row: row.object_id,
        ))

    def participant_entity(self, scope_id: str, key: str) -> EntityRowV1 | None:
        """Return the active participant entity with a ``participant_key`` (A.6.5)."""
        return self._keyed_entity(scope_id, _participant_key(scope_id, key))

    def minted_entity(self, scope_id: str, key: str) -> EntityRowV1 | None:
        """Return the active entity minted for one extraction declaration's ``mint_key``."""
        return self._keyed_entity(scope_id, _mint_key(scope_id, key))

    def spent_mint(self, scope_id: str, key: str) -> bool:
        """Whether an erasure ended the entity minted for a declaration's ``mint_key``."""
        self.scope_fence(scope_id)
        return bool(self._keys.get(_spent_key(scope_id, key)))

    def _keyed_entity(self, scope_id: str, key: str) -> EntityRowV1 | None:
        self.scope_fence(scope_id)
        found = self._keys.get(key) or frozenset()
        if len(found) > 1:
            raise AMLError("parity_mismatch", "An entity key names more than one entity")
        return next((row for object_id in found if (row := self._entities.get(object_id))), None)

    def same_state(self, other: RecallSnapshot) -> bool:
        """Compare every data projection for same-generation pointer publication.

        Scope fences may change at one generation; data post-states may not.
        """
        return (
            self.same_source_state(other)
            and _same_heads(self._memories, other._memories)
            and _same_heads(self._entities, other._entities)
            and _same_heads(self._names, other._names)
            and _same_heads(self._held, other._held)
            and _same_heads(self._spent, other._spent)
        )

    def same_source_state(self, other: RecallSnapshot) -> bool:
        """Check cold-root parity while shared metadata-only roots compare in O(1)."""
        if self._source_heads is other._source_heads:
            return True
        return {
            value.object_id: value.row
            for value in self._source_heads.values()
            if value.row is not None
        } == {
            value.object_id: value.row
            for value in other._source_heads.values()
            if value.row is not None
        }

    def shares_corpus(
        self, previous: RecallSnapshot, scope_id: str, *,
        kind: Literal["memory", "entity", "evidence"],
    ) -> bool:
        """Whether a corpus prepared at ``previous`` still holds at this root (A.6.1).

        Item, revision and link changes invalidate the memory corpus; entity and
        name changes the entity corpus; Evidence and hold changes the Evidence
        corpus; Evidence withheld-span, state and expiry changes the memory and
        entity corpora too. This grants reuse of pure preparation only, never a
        result or permission. A rebuilt root starts cold.

        Args:
            previous: The root the corpus was prepared at.
            scope_id: The corpus's exact scope.
            kind: The corpus.

        Returns:
            True when this root is ``previous``, or is its direct delta and that
            delta changed none of the corpus's inputs in the scope.
        """
        if self is previous:
            return True
        if (self.embedding_space, self.embedding_token) != (
            previous.embedding_space, previous.embedding_token,
        ):
            return False
        before, after = previous._scopes.get(scope_id), self._scopes.get(scope_id)
        if before is None or after is None or (
            (before.erase_epoch, before.consent_epoch) != (after.erase_epoch, after.consent_epoch)
            or after.tomb or after.active_erasure or after.maintenance
            or after.visibility_barrier or after.quarantined
            or kind == "evidence" and before.held_evidence != after.held_evidence
        ):
            return False
        delta = self._delta
        if delta is None or delta.base != previous.root_identity:
            return False
        changed = {
            "memory": delta.memory | delta.visibility,
            "entity": delta.entity | delta.visibility,
            "evidence": delta.evidence | delta.hold,
        }[kind]
        return scope_id not in changed

    def source_rows(self, scope_id: str, query_time: int) -> tuple[SourceRowV2, ...]:
        """Enumerate exact-scope usable sources at a caller-frozen query time.

        This eligibility view shares the lexical corpus's head and fence
        selection, then applies the caller's frozen TTL boundary.
        """
        _TIME.validate_python(query_time)
        return tuple(
            row for row in self.source_corpus(scope_id)
            if row.expires_at is None or row.expires_at > query_time
        )

    def source_corpus(self, scope_id: str) -> tuple[SourceRowV2, ...]:
        """Return frozen latest route documents, independent of query-time TTL.

        BM25 statistics belong to this root. Eligibility changes with the
        query time without changing its document count, lengths or frequencies.
        """
        scope = self.scope_fence(scope_id)
        if scope.tomb:
            raise AMLError("scope_denied", "Snapshot scope is tombstoned")
        if (
            scope.visibility_barrier or scope.active_erasure or scope.maintenance
            or scope.finalize_barrier is not None
        ):
            raise AMLError("index_rebuilding", "Snapshot scope has an active visibility fence")
        result: list[SourceRowV2] = []
        segment = self.source_segments
        while segment is not None:
            for version in segment.entries:
                head = self._source_heads.get(version.object_id)
                row = version.row
                if (
                    head is not version
                    or version.generation > self.generation
                    or row is None
                    or row.scope_id != scope_id
                    or row.scope_kind != scope.scope_kind
                    or row.state != "active"
                    or row.route_state != "indexed"
                ):
                    continue
                result.append(row)
            segment = segment.previous
        return tuple(sorted(result, key=lambda row: row.object_id))

    def add_view(
        self, scope_kind: ScopeKind, scope_id: str, *, now: int, tombstones: TombstoneSource,
        carry: AddViewCarry | None = None,
    ) -> SnapshotAddView:
        """Return the Add read view of one scope at this root (I1b).

        Args:
            scope_kind: The kind of the exact scope.
            scope_id: The exact scope.
            now: The read's trusted clock in epoch milliseconds.
            tombstones: The owner-private tombstone index of the same scope.
            carry: The records the same owner's earlier views of this scope
                built, which the view reuses where nothing they read changed;
                None builds every record afresh. Either way the view serves
                exactly the same reads.

        Returns:
            SnapshotAddView: The view; it holds this root until it is dropped.

        Raises:
            AMLError: ``scope_denied`` when the scope is unknown, of another kind
                or tombstoned, or the tombstone index serves another scope.
        """
        return SnapshotAddView(
            self, scope_kind, scope_id, now=now, tombstones=tombstones, carry=carry,
        )


def _same_heads[T](left: _PersistentMap[T | None], right: _PersistentMap[T | None]) -> bool:
    if left is right:
        return True
    return {key: value for key, value in left.items() if value is not None} == {
        key: value for key, value in right.items() if value is not None
    }


class SnapshotPublisher:
    """Publish one complete pointer under the owning Writer's external mutex."""

    def __init__(self, initial: RecallSnapshot) -> None:
        """Retain the complete boot-rebuilt root; do not create another state mirror."""
        self._current = initial

    @property
    def current(self) -> RecallSnapshot:
        """Read one root pointer; consumers own the required reader lifetime guard."""
        return self._current

    def publish(self, following: RecallSnapshot) -> None:
        """Swap a completed root while preserving one post-state per generation.

        The caller must own the Writer mutex and revoke health if durable state
        advanced but construction/publication failed. This class does not claim
        storage durability, concurrent-writer CAS, or reader cancellation.
        """
        current = self._current
        if following.generation < current.generation:
            raise AMLError("parity_mismatch", "Snapshot publication cannot regress")
        if following.generation == current.generation and not following.same_state(current):
            raise AMLError("parity_mismatch", "A generation already has another data post-state")
        self._current = following


class _EvidenceIndex:
    """One view's Evidence records by id, built lazily from its root and kept for its life.

    It references neither the view nor its predicates, so the view that owns
    it, and the `Readability` that looks Evidence up through it, form no
    reference cycle and are freed as soon as their read drops them.
    """

    __slots__ = ("_records", "_root", "_scope_id", "_scope_kind")

    def __init__(self, root: RecallSnapshot, scope_kind: ScopeKind, scope_id: str) -> None:
        self._root = root
        self._scope_kind = scope_kind
        self._scope_id = scope_id
        self._records: dict[str, EvidenceRecord | None] = {}

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        """Return one Evidence part of the scope in any state, or None."""
        if evidence_id in self._records:
            return self._records[evidence_id]
        row = self._root.get_source(evidence_id)
        record = (
            evidence_record(row)
            if row is not None and (row.scope_kind, row.scope_id) == (
                self._scope_kind, self._scope_id,
            )
            else None
        )
        self._records[evidence_id] = record
        return record


_tokens: IdentityMemo[MemorySnapshotV2, tuple[str, ...]] = IdentityMemo()


def _memory_tokens(snapshot: MemorySnapshotV2) -> tuple[str, ...]:
    """``memory_tokens`` of a snapshot, once per decoded snapshot object.

    Every Add view builds the record of each head of its scope, and a decoded
    head is the same object at every pin while it is unchanged.
    """
    tokens = _tokens.recall(snapshot)
    if tokens is None:
        tokens = tuple(memory_tokens(snapshot.statement, snapshot.keywords))
        _tokens.remember(snapshot, tokens)
    return tokens


@dataclass(frozen=True, slots=True)
class _CarriedVersion:
    """One version's record as a view built it, with every input the build read.

    ``meta`` is the version's ``(memory_id, version, op, reason_code, recorded_at,
    commit_seq)``; ``sources`` is the root's Evidence row (or None) for every
    Evidence id the snapshot's quote spans cite; ``expiries`` the ``expires_at``
    of those rows that have one; ``now`` the clock P-SRC compared them with.
    """

    meta: tuple[str, int, str, str, int, int]
    snapshot_json: str
    snapshot: MemorySnapshotV2
    sources: tuple[tuple[str, SourceRowV2 | None], ...]
    expiries: tuple[int, ...]
    now: int
    record: VersionRecord | None


@dataclass(frozen=True, slots=True)
class _CarriedMemory:
    """One Item's record as a view built it, from its row, head record and successors."""

    row: MemoryRowV2
    head: VersionRecord | None
    successors: tuple[ChainLink, ...]
    record: MemoryRecord | None


@dataclass(frozen=True, slots=True)
class _CarriedLinks:
    """The ``supersedes`` links one Item row contributes, by the Item each ends."""

    row: MemoryRowV2
    links: tuple[tuple[str, ChainLink], ...]


@dataclass(frozen=True, slots=True)
class _CarryState:
    """The carried records of one binding: scope, epochs and embedding space."""

    binding: tuple[str, str, int, int, str]
    versions: dict[str, _CarriedVersion]
    memories: dict[str, _CarriedMemory]
    links: dict[str, _CarriedLinks]


class AddViewCarry:
    """The records one owner's successive Add views of one scope reuse (FENCE-SPEEDUP).

    A finalize owner reads its scope through a fresh `SnapshotAddView` for every
    component it commits (C.11.2 #1), and each view used to rebuild the record of
    every memory of the scope: decoding its head, P-VER over its Evidence, σ and
    tokens. On a scope of 12.9k memories that was about 2.8 s per component.

    A record is a pure function of what its build read, so a later view reuses
    it exactly when those inputs are unchanged, and otherwise builds it afresh:

    - a version record, of its version's metadata and snapshot text, the root's
      row (or absence) for every Evidence id its quote spans cite, and, for each
      of those rows with an ``expires_at``, which side of the view's clock it
      lies on (P-SRC);
    - an Item record, of its row, its head's version record (the very object
      reused, or a rebuilt one) and its successors;
    - the ``supersedes`` links an Item contributes, of its row.

    Rows are compared by identity, then by value, so a reuse never depends on a
    row object surviving between roots. A view of another scope, epoch or
    embedding space starts an empty carry. Entries are immutable and replaced
    whole, and every reuse is checked by the view that reads it, so views that
    share a carry concurrently can at worst build a record twice.
    """

    __slots__ = ("_lock", "_state")

    def __init__(self) -> None:
        """Start empty; the first view binds the carry to its scope."""
        self._lock = threading.Lock()
        self._state: _CarryState | None = None

    def bind(self, binding: tuple[str, str, int, int, str]) -> _CarryState:
        """Return the carried records of one binding, starting empty for a new one.

        Args:
            binding: The view's scope kind, scope id, erase and consent epochs and
                embedding space.

        Returns:
            _CarryState: The records views of exactly this binding share.
        """
        with self._lock:
            state = self._state
            if state is None or state.binding != binding:
                state = self._state = _CarryState(binding, {}, {}, {})
            return state


class SnapshotAddView:
    """The production `AddReadView` of one scope over one `RecallSnapshot` root (I1b).

    The root supplies every projection; the owner-private `TombstoneSource`
    supplies the tombstones, which never enter the root (A.3.8, A.6.6), and the
    pin's ``tombstone_seq``. Memories, versions and names pass the D.2
    predicates of `Readability` at the view's clock before they are returned;
    Evidence is returned in any state. Records are built lazily and kept for the
    view's life: one owner read at one pin. With an `AddViewCarry` the view also
    reuses the records the owner's earlier views built whose inputs are unchanged,
    and keeps the records it builds for the next. The view holds its root, so a
    caller drops it when its guarded read ends.
    """

    def __init__(
        self, root: RecallSnapshot, scope_kind: ScopeKind, scope_id: str, *, now: int,
        tombstones: TombstoneSource, carry: AddViewCarry | None = None,
    ) -> None:
        """Bind the view to an available exact scope of ``root``; see `RecallSnapshot.add_view`."""
        _TIME.validate_python(now)
        fence = root.scope_fence(scope_id)
        if fence.scope_kind != scope_kind or fence.tomb:
            raise AMLError("scope_denied", "An Add view needs an available exact scope")
        if (tombstones.scope_kind, tombstones.scope_id) != (scope_kind, scope_id):
            raise AMLError("scope_denied", "The tombstone index serves another scope")
        sequence = tombstones.tombstone_seq
        if type(sequence) is not int or sequence < 0:
            raise AMLError("parity_mismatch", "The tombstone index has an invalid sequence")
        self._root = root
        self._scope_kind: ScopeKind = scope_kind
        self._scope_id = scope_id
        self._now = now
        self._tombstones = tombstones
        self._pin = ReadPin(
            commit_seq=root.generation,
            erase_epoch=fence.erase_epoch,
            consent_epoch=fence.consent_epoch,
            tombstone_seq=sequence,
            recall_snapshot_version=root.generation,
        )
        # The predicates look Evidence up through an index the view owns rather
        # than through the view itself: a view that referenced itself was a
        # reference cycle, so neither it nor the root, records and vectors it
        # holds were freed when its read ended, only by a later full collection.
        self._evidence = _EvidenceIndex(root, scope_kind, scope_id)
        self._read = Readability(scope_kind, scope_id, now, self._evidence.get)
        self._snapshots: dict[str, MemorySnapshotV2] = {}
        self._versions: dict[str, VersionRecord | None] = {}
        self._records: dict[str, MemoryRecord | None] = {}
        self._successors: dict[str, tuple[ChainLink, ...]] | None = None
        self._names: dict[str, tuple[EntityNameRecord, ...]] | None = None
        self._sessions: dict[str, tuple[MessageRecord, ...]] | None = None
        self._speakers: dict[str, tuple[str, ...]] | None = None
        self._rows: tuple[MemoryRowV2, ...] | None = None
        self._heads: tuple[MemoryRecord, ...] | None = None
        self._sources: dict[str, SourceRowV2 | None] = {}
        self._carry = None if carry is None else carry.bind((
            scope_kind, scope_id, fence.erase_epoch, fence.consent_epoch, root.embedding_space,
        ))

    @property
    def scope_kind(self) -> ScopeKind:
        """The kind of the view's exact scope."""
        return self._scope_kind

    @property
    def scope_id(self) -> str:
        """The id of the view's exact scope."""
        return self._scope_id

    @property
    def pin(self) -> ReadPin:
        """The root generation, the scope's epochs and the tombstone index's sequence."""
        return self._pin

    @property
    def now(self) -> int:
        """The trusted clock of this read in epoch milliseconds."""
        return self._now

    @property
    def embedding_space(self) -> str:
        """The root's effective embedding space."""
        return self._root.embedding_space

    def _in_scope(self, row: Any) -> bool:
        return row is not None and (row.scope_kind, row.scope_id) == (
            self._scope_kind, self._scope_id,
        )

    def _memory_row(self, memory_id: str) -> MemoryRowV2 | None:
        row = self._root.get_memory(memory_id)
        return row if self._in_scope(row) else None

    def _memory_rows(self) -> tuple[MemoryRowV2, ...]:
        """The root's Item rows of the scope, by id, read once per view."""
        if self._rows is None:
            self._rows = self._root.memory_rows(self._scope_id)
        return self._rows

    def _source_row(self, evidence_id: str) -> SourceRowV2 | None:
        """The root's Evidence row of one id in any scope, or None, read once per view."""
        rows = self._sources
        if evidence_id in rows:
            return rows[evidence_id]
        row = rows[evidence_id] = self._root.get_source(evidence_id)
        return row

    def _snapshot(self, revision_id: str, snapshot_json: str) -> MemorySnapshotV2:
        snapshot = self._snapshots.get(revision_id)
        if snapshot is None:
            kept = None if self._carry is None else self._carry.versions.get(revision_id)
            snapshot = (
                kept.snapshot if kept is not None and (
                    kept.snapshot_json is snapshot_json or kept.snapshot_json == snapshot_json
                ) else decoded_snapshot(snapshot_json)
            )
            self._snapshots[revision_id] = snapshot
        return snapshot

    def _unchanged(self, kept: _CarriedVersion) -> bool:
        """Whether every input of a carried version record reads the same in this view."""
        for evidence_id, row in kept.sources:
            current = self._source_row(evidence_id)
            if current is not row and current != row:
                return False
        now = self._now
        return all((at > now) == (at > kept.now) for at in kept.expiries)

    def _version(
        self, memory_id: str, revision_id: str, version: int, op: RevisionOp,
        reason_code: RevisionReason, recorded_at: int, commit_seq: int, snapshot_json: str,
    ) -> VersionRecord | None:
        if revision_id in self._versions:
            return self._versions[revision_id]
        meta = (memory_id, version, op, reason_code, recorded_at, commit_seq)
        carry = self._carry
        if carry is not None:
            kept = carry.versions.get(revision_id)
            if kept is not None and kept.meta == meta and (
                kept.snapshot_json is snapshot_json or kept.snapshot_json == snapshot_json
            ) and self._unchanged(kept):
                self._snapshots.setdefault(revision_id, kept.snapshot)
                self._versions[revision_id] = kept.record
                return kept.record
        content = self._snapshot(revision_id, snapshot_json)
        record: VersionRecord | None = None
        if self._read.p_ver(content):
            sigmas = [
                span_sigma(part, span.byte_start) for span in content.quote_spans
                if (part := self.evidence(span.evidence_id)) is not None
            ]
            record = VersionRecord(
                memory_id=memory_id, revision_id=revision_id, version=version, op=op,
                reason_code=reason_code, recorded_at=recorded_at, commit_seq=commit_seq,
                snapshot=content, sigma=max(sigmas),
                tokens=_memory_tokens(content),
            )
        self._versions[revision_id] = record
        if carry is not None:
            sources = tuple(
                (evidence_id, self._source_row(evidence_id))
                for evidence_id in dict.fromkeys(span.evidence_id for span in content.quote_spans)
            )
            carry.versions[revision_id] = _CarriedVersion(
                meta, snapshot_json, content, sources,
                tuple(row.expires_at for _, row in sources
                      if row is not None and row.expires_at is not None),
                self._now, record,
            )
        return record

    def _head(self, row: MemoryRowV2) -> VersionRecord | None:
        return self._version(
            row.object_id, row.current_revision_id, row.version, row.op, row.reason_code,
            row.recorded_at, row.head_commit_seq, row.snapshot_json,
        )

    def _links(self, row: MemoryRowV2) -> tuple[tuple[str, ChainLink], ...]:
        """The active ``supersedes`` links one Item row carries, with the Item each ends."""
        carry = self._carry
        if carry is not None:
            kept = carry.links.get(row.object_id)
            if kept is not None and (kept.row is row or kept.row == row):
                return kept.links
        links = tuple(
            (link.target_key, ChainLink(
                memory_id=row.object_id, revision_id=row.current_revision_id,
                relation=link.relation, effective_at=link.effective_at,
                effective_basis=link.effective_basis, verdict_ref=link.verdict_ref or "",
            ))
            for link in row.links if link.relation is not None
        )
        if carry is not None:
            carry.links[row.object_id] = _CarriedLinks(row, links)
        return links

    def _incoming(self) -> dict[str, tuple[ChainLink, ...]]:
        """Index every active ``supersedes`` link of the scope by the Item it ends."""
        if self._successors is None:
            found: dict[str, list[ChainLink]] = {}
            for row in self._memory_rows():
                for target, link in self._links(row):
                    found.setdefault(target, []).append(link)
            self._successors = {
                target: tuple(sorted(links, key=lambda link: link.memory_id))
                for target, links in found.items()
            }
        return self._successors

    def _record(self, row: MemoryRowV2) -> MemoryRecord | None:
        if row.object_id in self._records:
            return self._records[row.object_id]
        head = self._head(row)
        record: MemoryRecord | None = None
        if head is not None:
            successors = self._incoming().get(row.object_id, ())
            carry = self._carry
            kept = None if carry is None else carry.memories.get(row.object_id)
            if kept is not None and kept.head is head and (
                kept.row is row or kept.row == row
            ) and (kept.successors is successors or kept.successors == successors):
                record = kept.record
            else:
                record = MemoryRecord(
                    memory_id=row.object_id, lifecycle_state=row.lifecycle_state,
                    activation_mode=row.activation_mode, commit_seq=row.commit_seq,
                    retired_at=row.retired_at, head=head, successors=successors,
                )
                if carry is not None:
                    carry.memories[row.object_id] = _CarriedMemory(row, head, successors, record)
        self._records[row.object_id] = record
        return record

    def heads(self) -> tuple[MemoryRecord, ...]:
        """Return every P-MEM memory of the scope: active with a head that passes P-VER.

        Args:
            None.

        Returns:
            tuple[MemoryRecord, ...]: The memories, sorted by ``memory_id``.
        """
        if self._heads is None:
            self._heads = tuple(
                record for row in self._memory_rows()
                if row.lifecycle_state == "active" and (record := self._record(row)) is not None
            )
        return self._heads

    def memory(self, memory_id: str) -> MemoryRecord | None:
        """Return one Item in any lifecycle when its current head passes P-VER.

        Args:
            memory_id (str): The Item.

        Returns:
            MemoryRecord | None: The Item, or None when it does not exist in the
                scope or its head fails P-VER.
        """
        row = self._memory_row(memory_id)
        return None if row is None else self._record(row)

    def history(self, memory_id: str) -> tuple[HistoryEntry, ...]:
        """Return the readable history entries of an Item's current head (C.1).

        Args:
            memory_id (str): The Item whose head the entries belong to.

        Returns:
            tuple[HistoryEntry, ...]: The revisions a ``revise`` replaced and the
                heads of the Items its head ends, that pass P-VER, newest first
                by ``recorded_at`` then ``revision_id``.
        """
        row = self._memory_row(memory_id)
        return () if row is None else self._history(row)

    def _history(self, row: MemoryRowV2) -> tuple[HistoryEntry, ...]:
        """`history` of an Item row of the scope."""
        entries: list[HistoryEntry] = []
        for entry in row.replaced:
            version = self._version(
                row.object_id, entry.revision_id, entry.version, entry.op, entry.reason_code,
                entry.recorded_at, entry.commit_seq, entry.snapshot_json,
            )
            if version is not None:
                entries.append(HistoryEntry(version, entry.ended_by, entry.reworded))
        for target_key, link in self._links(row):
            target = self._memory_row(target_key)
            if target is not None and (version := self._head(target)) is not None:
                entries.append(HistoryEntry(version, link.relation))
        return _newest_first(entries) if entries else ()

    def retired(self) -> tuple[HistoryEntry, ...]:
        """Return the readable retired versions the corrects route scores (C.4.3 #4).

        A version a rewording replaced is none (C.9.4a): its description was refuted
        and its fact continues in the head, so no correction of history may target it,
        and after a forget of the correction its wording is tombstoned text that
        reaches the judge only as a forgotten entry (C4).

        Args:
            None.

        Returns:
            tuple[HistoryEntry, ...]: Each revision once, newest first by
                ``recorded_at`` then ``revision_id``.
        """
        rows = self._memory_rows()
        entries = [
            entry for row in rows
            if row.lifecycle_state == "active" and self._record(row) is not None
            for entry in self._history(row) if not entry.reworded
        ]
        for row in rows:
            ended = _LIFECYCLE_ENDS.get(row.lifecycle_state)
            if ended is not None and (record := self._record(row)) is not None:
                entries.append(HistoryEntry(record.head, ended))
        unique = {entry.version.revision_id: entry for entry in reversed(entries)}
        return _newest_first(unique.values())

    def _name_index(self) -> dict[str, tuple[EntityNameRecord, ...]]:
        """Group the scope's P-NAME rows by entity, in catalog order."""
        if self._names is None:
            found: dict[str, list[EntityNameRecord]] = {}
            for row in self._root.entity_name_rows(self._scope_id):
                if self._readable_name(row):
                    found.setdefault(row.entity_id, []).append(EntityNameRecord(
                        object_id=row.object_id, entity_id=row.entity_id, name=row.name,
                        name_norm=row.name_norm, name_kind=row.name_kind,
                        naming_memory_id=row.naming_memory_id,
                        naming_revision_id=row.naming_revision_id,
                        annotation_index=row.annotation_index, commit_seq=row.commit_seq,
                    ))
            self._names = {
                entity_id: tuple(sorted(rows, key=lambda row: (
                    _NAME_KIND_ORDER[row.name_kind], -row.commit_seq, row.name_norm,
                    row.naming_revision_id, row.annotation_index,
                )))
                for entity_id, rows in found.items()
            }
        return self._names

    def _readable_name(self, row: EntityNameRowV1) -> bool:
        """P-NAME of one name row; its P-VER is the naming head's version record.

        The naming Item's head has a version record exactly when it passes P-VER,
        so the record, carried or built, decides the predicate's last clause.
        """
        naming = self._memory_row(row.naming_memory_id)
        if naming is None:
            return False
        self._snapshot(naming.current_revision_id, naming.snapshot_json)
        return (
            naming.lifecycle_state == "active"
            and naming.current_revision_id == row.naming_revision_id
            and self._head(naming) is not None
        )

    def _entity(self, row: EntityRowV1) -> EntityRecord:
        return EntityRecord(
            entity_id=row.object_id, entity_type=row.entity_type,
            participant_key=row.participant_key, mint_key=row.mint_key,
            commit_seq=row.commit_seq, names=self._name_index().get(row.object_id, ()),
        )

    def _active(self, row: EntityRowV1 | None) -> EntityRecord | None:
        return self._entity(row) if row is not None and self._in_scope(row) else None

    def entity(self, entity_id: str) -> EntityRecord | None:
        """Return an active entity with its P-NAME names, or the virtual user.

        Args:
            entity_id (str): An `EntityRef`; ``"user"`` returns the virtual user
                with the names of naming memories that anchor it.

        Returns:
            EntityRecord | None: The entity, or None when it does not exist in
                the scope or is erased.
        """
        if entity_id == USER:
            return EntityRecord(USER, "user", None, None, 0, self._name_index().get(USER, ()))
        return self._active(self._root.get_entity(entity_id))

    def participant(self, participant_key: str) -> EntityRecord | None:
        """Return the active participant with a key, by the root's key index.

        Args:
            participant_key (str): A source tuple or labelled-speaker key (A0.6).

        Returns:
            EntityRecord | None: The participant, or None before its first mint.
        """
        return self._active(self._root.participant_entity(self._scope_id, participant_key))

    def speakers(self) -> Mapping[str, tuple[str, ...]]:
        """Return the speaker names source participants were shown under (B.4.2).

        Args:
            None.

        Returns:
            Mapping[str, tuple[str, ...]]: `speaker_index` of the scope's
                Evidence that passes P-SRC at the view's clock.
        """
        if self._speakers is None:
            self._speakers = speaker_index(
                row for row in self._root.scope_sources(self._scope_id) if self._read.p_src(row)
            )
        return self._speakers

    def minted(self, mint_key: str) -> EntityRecord | None:
        """Return the active entity minted for one extraction declaration.

        Args:
            mint_key (str): The declaration's ``mint_key`` (A0.6, C.9.6).

        Returns:
            EntityRecord | None: The entity, or None when it was never minted
                or an erasure ended it (`spent`).
        """
        return self._active(self._root.minted_entity(self._scope_id, mint_key))

    def spent(self, mint_key: str) -> bool:
        """Whether an erasure ended the entity a declaration minted (A.3.6, C.9.6).

        A spent declaration is never minted again: its erased entity keeps the
        unique key. A slot naming it is open, as one naming an erased entity is.

        Args:
            mint_key (str): The declaration's ``mint_key`` (A0.6, C.9.6).

        Returns:
            bool: True when the declaration's entity was minted and then erased.
        """
        return self._root.spent_mint(self._scope_id, mint_key)

    def entities(self) -> tuple[EntityRecord, ...]:
        """Return every active entity of the scope with its P-NAME names.

        Args:
            None.

        Returns:
            tuple[EntityRecord, ...]: Sorted by ``entity_id``; the virtual user
                is not listed.
        """
        return tuple(self._entity(row) for row in self._root.entity_rows(self._scope_id))

    def tombstones(self) -> tuple[TombstoneRecord, ...]:
        """Return every tombstone of the scope at the pin, from the tombstone index.

        Args:
            None.

        Returns:
            tuple[TombstoneRecord, ...]: Sorted by ``object_id``.

        Raises:
            AMLError: ``scope_denied`` when the index returns another scope's row.
        """
        rows = self._tombstones.tombstones()
        if not all(self._in_scope(row) for row in rows):
            raise AMLError("scope_denied", "The tombstone index returned another scope's row")
        return tuple(sorted(rows, key=lambda row: row.object_id))

    def evidence(self, evidence_id: str) -> EvidenceRecord | None:
        """Return one Evidence part in any state.

        Args:
            evidence_id (str): The part.

        Returns:
            EvidenceRecord | None: The part, or None when it does not exist in
                the scope.
        """
        return self._evidence.get(evidence_id)

    def _session_messages(self, session_id: str) -> tuple[MessageRecord, ...]:
        if self._sessions is None:
            parts: dict[MessageKey, list[EvidenceRecord]] = {}
            for row in self._root.scope_sources(self._scope_id):
                record = self.evidence(row.object_id)
                assert record is not None
                key = MessageKey(record.session_id, record.request_id, record.ordinal)
                parts.setdefault(key, []).append(record)
            sessions: dict[str, list[MessageRecord]] = {}
            for key, records in parts.items():
                sessions.setdefault(key.session_id, []).append(MessageRecord(
                    key, tuple(sorted(records, key=lambda record: record.part_no)),
                ))
            self._sessions = {
                session: tuple(sorted(messages, key=lambda message: (
                    message.sigma, message.key.request_id, message.key.ordinal,
                )))
                for session, messages in sessions.items()
            }
        return self._sessions.get(session_id, ())

    def message(self, key: MessageKey) -> MessageRecord | None:
        """Return every part of one message.

        Args:
            key (MessageKey): The message.

        Returns:
            MessageRecord | None: Its parts in any state by ``part_no``, or None
                when the scope has no part of it.
        """
        return next(
            (message for message in self._session_messages(key.session_id) if message.key == key),
            None,
        )

    def messages_before(self, key: MessageKey, limit: int) -> tuple[MessageRecord, ...]:
        """Return the messages immediately before one message in its session (C.4.1).

        Args:
            key (MessageKey): The message to look before.
            limit (int): The most messages to return; 0 or more.

        Returns:
            tuple[MessageRecord, ...]: Up to ``limit`` messages in session order
                (σ, request, ordinal), oldest first.

        Raises:
            ValueError: If ``limit`` is negative.
        """
        if limit < 0:
            raise ValueError("limit is 0 or more")
        messages = self._session_messages(key.session_id)
        index = next((i for i, message in enumerate(messages) if message.key == key), None)
        return () if index is None else messages[max(0, index - limit):index]

    def heads_by_digest(self, statement_digest: str) -> tuple[MemoryRecord, ...]:
        """Return the P-MEM memories whose head has a statement digest (B.8, cross-Q10).

        Args:
            statement_digest (str): The digest to look up, by the root's digest index.

        Returns:
            tuple[MemoryRecord, ...]: The readable active memories, by ``memory_id``.
        """
        return tuple(
            record for row in self._root.memories_with_digest(self._scope_id, statement_digest)
            if row.lifecycle_state == "active" and (record := self._record(row)) is not None
        )

    def withheld_heads(self, statement_digest: str) -> tuple[WithheldHeadRecord, ...]:
        """Return the active Items whose unreadable head has a statement digest (B.8 #2b).

        Args:
            statement_digest (str): The digest to look up, by the root's digest index.

        Returns:
            tuple[WithheldHeadRecord, ...]: Identities only, by ``memory_id``.
        """
        found: list[WithheldHeadRecord] = []
        for row in self._root.memories_with_digest(self._scope_id, statement_digest):
            if row.lifecycle_state != "active" or self._record(row) is not None:
                continue
            snapshot = self._snapshot(row.current_revision_id, row.snapshot_json)
            found.append(WithheldHeadRecord(
                memory_id=row.object_id, current_revision_id=row.current_revision_id,
                entity_id=snapshot.entity_id, about_entity_ids=tuple(snapshot.about_entity_ids),
                support_groups=len(snapshot.support_groups), quote_spans=len(snapshot.quote_spans),
            ))
        return tuple(found)

    def entities_by_name(self, name_norm: str) -> tuple[EntityRecord, ...]:
        """Return the active entities with a readable name of one normal form (B.8, cross-Q10).

        Args:
            name_norm (str): The ``name_norm`` to look up, by the root's name index.

        Returns:
            tuple[EntityRecord, ...]: The entities, the virtual user included, whose
                P-NAME names include one with this ``name_norm``, by ``entity_id``.
        """
        found: set[str] = {
            row.entity_id for row in self._root.names_with_norm(self._scope_id, name_norm)
            if self._readable_name(row)
        }
        records = (self.entity(entity_id) for entity_id in sorted(found))
        return tuple(record for record in records if record is not None)

    def tombstone(self, entity_id: str, statement_digest: str) -> TombstoneRecord | None:
        """Return the tombstone of one ``(entity_id, statement_digest)`` (B.8, cross-Q10).

        Args:
            entity_id (str): The forgotten version's primary anchor.
            statement_digest (str): The forgotten statement's digest.

        Returns:
            TombstoneRecord | None: The tombstone from the owner-private index, or
                None when the pair was never forgotten in the scope.

        Raises:
            AMLError: ``scope_denied`` when the index returns another scope's row.
        """
        row = self._tombstones.tombstone(entity_id, statement_digest)
        if row is not None and not self._in_scope(row):
            raise AMLError("scope_denied", "The tombstone index returned another scope's row")
        return row


def _newest_first(entries: Iterable[HistoryEntry]) -> tuple[HistoryEntry, ...]:
    return tuple(sorted(
        entries, key=lambda entry: (entry.version.recorded_at, entry.version.revision_id),
        reverse=True,
    ))


def evidence_record(row: SourceRowV2) -> EvidenceRecord:
    """Present one root Evidence projection as the Add read view and quote speakers read it."""
    participant = row.source_participant
    return EvidenceRecord(
        object_id=row.object_id, scope_kind=row.scope_kind, scope_id=row.scope_id,
        session_id=row.session_id, request_id=row.request_id, ordinal=row.ordinal,
        part_no=row.part_no, kind=row.kind, role=row.role, speaker_entity=row.speaker_entity,
        source_participant=None if participant is None else SourceParticipantV1(
            schema_version=participant.schema_version, namespace=participant.namespace,
            conversation_id=participant.conversation_id,
            participant_id=participant.participant_id,
        ),
        source_observed_at=row.source_observed_at, ingested_at=row.ingested_at,
        ingested_commit_seq=row.ingested_commit_seq, byte_start=row.byte_start,
        byte_end=row.byte_end, time_granularity=row.time_granularity,
        trust_tier=row.trust_tier, security_level=row.security_level, state=row.state,
        route_state=row.route_state, expires_at=row.expires_at, text=row.text,
        embedding=row.embedding, embedding_space=row.embedding_space,
        withheld_spans=row.withheld_spans, entity_bindings=row.entity_bindings,
    )
