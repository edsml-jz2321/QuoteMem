"""Receipt checkpoint ordering and immutable source descriptors.

Whole-part unit packing and physical capacity proofs live in ingestion.units.
This module preserves the ordering seams used by the owner and source readers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ContentBindingDigest, canonical_json
from quotemem.core.query import ScopeV1
from quotemem.core.technical_types import (
    MessageRangeKey,
    MessageRangeKeyV1,
    MessageRangeKeyV2,
    parse_message_key,
)
from quotemem.core.work import FinalizeSourceHeaderV1
from quotemem.ingestion.request_order import RequestCoordinate, request_partitions
from quotemem.ingestion.sources import word_count as source_word_count
from quotemem.store.domain import EvidenceRowV1

BindContent = Callable[[str, object], ContentBindingDigest]
_CHUNK = re.compile(r"(?:^|:)chunk-([0-9]+)$")


def natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Order ASCII digit runs numerically, with an exact-string final tie-break."""
    return tuple(
        (0, int(part)) if part.isascii() and part.isdecimal() else (1, part)
        for part in re.split(r"([0-9]+)", value)
    ) + ((2, value),)


def message_chunk_index(request_id: str) -> int | None:
    """Read only the exact trailing ASCII ``chunk-N`` checkpoint coordinate.

    This shared parser needs no Evidence body or eligibility decision. Missing
    suffixes retain the existing frozen timestamp/request-ID fallback ordering.
    """
    parsed = _CHUNK.search(request_id)
    return int(parsed[1]) if parsed else None


def _time_key(value: int | None) -> tuple[bool, int]:
    return value is None, value if value is not None else 0


def message_order(key: MessageRangeKey) -> tuple[object, ...]:
    """Order the exact frozen variant without interpreting receipt order as time."""
    chunk: tuple[object, ...] = (
        (0, key.chunk_index, key.request_id)
        if key.chunk_index is not None
        else (1, _time_key(key.chunk_event_time), key.request_id)
    )
    if isinstance(key, MessageRangeKeyV2):
        chunk = (
            (key.receipt_partition_seq, 0, key.received_commit_seq, key.request_id)
            if key.chunk_index is None and not key.source_time_present
            else (key.receipt_partition_seq, 1, chunk)
        )
    return (
        _time_key(key.session_event_time),
        natural_key(key.session_id),
        chunk,
        key.ordinal,
        key.part_no,
        key.evidence_id,
    )


def message_key(
    row: EvidenceRowV1 | FinalizeSourceHeaderV1, *, session_event_time: int | None,
    chunk_event_time: int | None, coordinate_version: Literal[1, 2],
    received_commit_seq: int, source_time_present: bool,
    receipt_partition_seq: int = 0,
) -> MessageRangeKey:
    """Derive a coordinate from a validated source and its complete request anchors."""
    values = dict(
        session_event_time=session_event_time, session_id=row.session_id,
        request_id=row.request_id, chunk_index=message_chunk_index(row.request_id),
        chunk_event_time=chunk_event_time, ordinal=row.ordinal, part_no=row.part_no,
        evidence_id=row.object_id,
    )
    if coordinate_version == 2:
        return MessageRangeKeyV2.model_validate({
            **values, "key_kind": "message_v2", "received_commit_seq": received_commit_seq,
            "source_time_present": source_time_present,
            "receipt_partition_seq": receipt_partition_seq,
        })
    return MessageRangeKeyV1.model_validate({**values, "key_kind": "message"})


def checkpoint_partitions(
    sessions: dict[str, int | None],
    chunks: dict[tuple[str, str], tuple[int, int, str, int | None]],
    receipts: dict[tuple[str, str], tuple[int, bool]],
) -> dict[tuple[str, str], int]:
    """Use the same complete-request block order for every checkpoint consumer."""
    result: dict[tuple[str, str], int] = {}
    for session_id, event_time in sessions.items():
        coordinates: list[RequestCoordinate] = []
        for (session, request_id), anchor in chunks.items():
            if session != session_id:
                continue
            chunk_index = message_chunk_index(request_id)
            key = MessageRangeKeyV1(
                key_kind="message", session_event_time=event_time, session_id=session,
                request_id=request_id, chunk_index=chunk_index, chunk_event_time=anchor[3],
                ordinal=anchor[0], part_no=anchor[1], evidence_id=anchor[2],
            )
            receipt, timed = receipts[session, request_id]
            coordinates.append(RequestCoordinate(
                (session, request_id), message_order(key), receipt,
                chunk_index is not None or timed,
            ))
        result.update(request_partitions(coordinates))
    return result


@dataclass(frozen=True)
class CheckpointMember:
    """A source locator and frozen row proof, without its original plaintext."""

    evidence_id: str
    row_binding_digest: ContentBindingDigest
    key_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.key.evidence_id != self.evidence_id:
            raise QuoteMemError("binding_mismatch", "Checkpoint coordinate names another Evidence")

    @property
    def key(self) -> MessageRangeKey:
        """Return a fresh strict coordinate, never mutable shared state."""
        return parse_message_key(self.key_json)


def checkpoint_order(
    rows: Iterable[EvidenceRowV1],
    *,
    scope: ScopeV1,
    checkpoint_time: int,
    input_high_watermark: int,
    bind_content: BindContent,
    processed: Callable[[EvidenceRowV1], bool],
    owned: Callable[[EvidenceRowV1], bool],
    session_id: str | None = None,
    max_descriptors: int = 100_000,
) -> tuple[CheckpointMember, ...]:
    """Freeze eligible descriptor order using only the checkpoint's own members.

    ``processed`` consumes the verified starting cursor; ``owned`` consumes the
    immutable reception classification. Neither callback is an authorization
    grant. The coordinator must revalidate session ownership before attachment.
    The explicit descriptor cap rejects without truncation. Callers with a stable
    indexed streams may pass whole parts to ``units.plan_units``.
    """
    if min(checkpoint_time, input_high_watermark) < 0 or max_descriptors < 1:
        raise QuoteMemError("invalid_request", "Checkpoint bounds must be nonnegative")
    members: list[CheckpointMember] = []
    sessions: dict[str, int | None] = {}
    chunks: dict[tuple[str, str], tuple[int, int, str, int | None]] = {}
    receipts: dict[tuple[str, str], tuple[int, bool]] = {}
    seen: set[str] = set()
    for supplied in rows:
        row = EvidenceRowV1.model_validate(supplied.model_dump(mode="json"))
        if (row.scope_kind, row.scope_id) != (scope.scope_kind, scope.scope_id):
            raise QuoteMemError("scope_denied", "Checkpoint source is outside its exact scope")
        if session_id is not None and row.session_id != session_id:
            raise QuoteMemError("scope_denied", "Checkpoint source belongs to another session")
        eligible = not (
            row.state != "active"
            or (row.expires_at is not None and row.expires_at <= checkpoint_time)
            or row.ingested_commit_seq > input_high_watermark
            or processed(row)
            or not owned(row)
        )
        # Receipt anchors use complete original request metadata. An erased or
        # consumed part cannot make the same request switch ordering modes.
        # Confirmed requests belong to another extraction owner but remain
        # admitted chronology. Calling owned still verifies their immutable
        # receipt classification; its False result cannot grant membership.
        if not eligible:
            if row.ingested_commit_seq > input_high_watermark:
                continue
            owned(row)
        if row.object_id in seen:
            raise QuoteMemError("parity_mismatch",
                                "Checkpoint contains a duplicate Evidence identity")
        if len(seen) == max_descriptors:
            raise QuoteMemError("budget_exceeded", "Checkpoint descriptor budget is exhausted")
        seen.add(row.object_id)
        previous = sessions.get(row.session_id)
        observed = row.source_observed_at
        if row.session_id not in sessions or (
            observed is not None and (previous is None or observed < previous)
        ):
            sessions[row.session_id] = observed
        chunk = row.session_id, row.request_id
        earliest = row.ordinal, row.part_no, row.object_id, observed
        if chunk not in chunks or earliest[:3] < chunks[chunk][:3]:
            chunks[chunk] = earliest
        prior_receipt = receipts.get(chunk, (row.ingested_commit_seq, False))
        receipts[chunk] = (
            min(prior_receipt[0], row.ingested_commit_seq),
            prior_receipt[1] or observed is not None,
        )
        if not eligible:
            continue
        key = message_key(
            row, session_event_time=None, chunk_event_time=None,
            coordinate_version=2,
            received_commit_seq=row.ingested_commit_seq,
            source_time_present=observed is not None,
        )
        members.append(
            CheckpointMember(
                row.object_id,
                bind_content("row-state", row.model_dump(mode="json")),
                canonical_json(key),
            )
        )
    partitions = checkpoint_partitions(sessions, chunks, receipts)
    frozen: list[CheckpointMember] = []
    for member in members:
        key = member.key
        key.session_event_time = sessions[key.session_id]
        key.chunk_event_time = chunks[key.session_id, key.request_id][3]
        if isinstance(key, MessageRangeKeyV2):
            key.received_commit_seq, key.source_time_present = receipts[
                key.session_id, key.request_id
            ]
            key.receipt_partition_seq = partitions[key.session_id, key.request_id]
        frozen.append(replace(member, key_json=canonical_json(key)))
    return tuple(sorted(frozen, key=lambda member: message_order(member.key)))


def word_count(text: str) -> int:
    """Count source words using the shared NL (natural-language) source renderer's rule."""
    return source_word_count(text)
