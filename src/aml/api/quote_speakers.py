"""The quote-speaker resolution History and Why share with Search delivery (E.2 ``Quote``; I13).

Search names a quote's speaker from S2a's frozen source labels
(``api/bench_search.quote_speakers`` -> ``api/bench_render.frozen_quote_speakers``):
no provider is called and no label is classified. `FrozenLabelSpeakers` answers
the same question for one quoted span at a time, the shape
``publication/history.QuoteSpeakers`` asks for, so the runtime hands one
resolution to the memory readers (S8a F1).

The seam receives the read's guarded rows and the whole span (a recorded design
decision): the quoted message is read through those
rows, so the delivery recheck covers it, and the span's end reaches the same
resolution Search uses, so a span crossing into a later source is null in
History and Why exactly as in Search. It lives beside Search delivery in
``aml.api`` so the publication readers never import the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from aml.api.bench_render import frozen_quote_speakers
from aml.core.add_read_view import EvidenceRecord, MessageKey, MessageRecord
from aml.core.entities import QuoteSpanV1
from aml.core.errors import AMLError
from aml.core.readability import SpanRef
from aml.publication.history import MemoryRows
from aml.store.domain import EvidenceRowV1
from aml.store.indexed import MAX_PAGE_SIZE, EvidenceRows
from aml.store.source_labels import source_label_view

__all__ = ["MAX_MESSAGE_PARTS", "FrozenLabelSpeakers", "evidence_record"]

MAX_MESSAGE_PARTS = 4096
"""The most Evidence rows one request may page while its message is collected."""


def evidence_record(row: EvidenceRowV1) -> EvidenceRecord:
    """Present one stored Evidence part as the label locator and source runs read it.

    Args:
        row (EvidenceRowV1): The stored part, in any state.

    Returns:
        EvidenceRecord: The same part in message coordinates.
    """
    return EvidenceRecord(
        object_id=row.object_id, scope_kind=row.scope_kind, scope_id=row.scope_id,
        session_id=row.session_id, request_id=row.request_id, ordinal=row.ordinal,
        part_no=row.part_no, kind=row.kind, role=row.role, speaker_entity=row.speaker_entity,
        source_participant=row.source_participant,
        source_observed_at=row.source_observed_at, ingested_at=row.ingested_at,
        ingested_commit_seq=row.ingested_commit_seq, byte_start=row.byte_start,
        byte_end=row.byte_end, time_granularity=row.time_granularity,
        trust_tier=row.trust_tier, security_level=row.security_level, state=row.state,
        route_state=row.route_state, expires_at=row.expires_at, text=row.text,
        embedding=tuple(row.embedding), embedding_space=row.embedding_space,
        withheld_spans=tuple(row.withheld_spans),
        entity_bindings=tuple(row.entity_bindings or ()),
    )


@dataclass(frozen=True, slots=True)
class _Quoted:
    """The one span to resolve, in the shape Search delivery's resolution reads."""

    quote_spans: tuple[SpanRef, ...]
    statement: str = ""
    support_groups: tuple[tuple[int, ...], ...] = ()


def _message(rows: MemoryRows, evidence: EvidenceRowV1) -> MessageRecord:
    """Read every stored part of the quoted message through the guarded rows."""
    scope = rows.scope
    query = EvidenceRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id,
                         session_id=evidence.session_id, request_id=evidence.request_id)
    parts: list[EvidenceRecord] = []
    after: str | None = None
    seen = 0
    while True:
        page = rows.page(query, after=after, limit=MAX_PAGE_SIZE)
        seen += len(page)
        if seen > MAX_MESSAGE_PARTS:
            raise AMLError("parity_mismatch", "A quoted request exceeds its bounded parts")
        for found in page:
            if found["ordinal"] == evidence.ordinal:
                row = rows.evidence(found["object_id"])
                if row is not None:
                    parts.append(evidence_record(row))
        if len(page) < MAX_PAGE_SIZE:
            break
        after = page[-1]["object_id"]
    key = MessageKey(evidence.session_id, evidence.request_id, evidence.ordinal)
    return MessageRecord(key, tuple(sorted(parts, key=lambda part: part.part_no)))


class FrozenLabelSpeakers:
    """Who speaks a quoted span, from the frozen labels Search delivery reads (B.3a, B.4.1)."""

    def speaker(self, rows: MemoryRows, span: QuoteSpanV1) -> str | None:
        """Return the speaker of one quoted span, or None.

        Args:
            rows (MemoryRows): The read's guarded rows; the message's parts and
                their request lookup are recorded for the delivery recheck.
            span (QuoteSpanV1): The quoted span, start and end.

        Returns:
            str | None: The attributed speaker; None when the Evidence is gone or
                of another scope, the source has none, an ``ambiguous`` label
                governs it, labels and metadata conflict or the span crosses a
                source boundary (as Search renders it).

        Raises:
            AMLError: The guard's refusal, or ``parity_mismatch`` beyond
                ``MAX_MESSAGE_PARTS`` rows in the quoted request.
        """
        evidence = rows.evidence(span.evidence_id)
        scope = rows.scope
        if evidence is None or (evidence.scope_kind, evidence.scope_id) != (
            scope.scope_kind, scope.scope_id,
        ):
            return None
        message = _message(rows, evidence)
        if not any(part.object_id == evidence.object_id for part in message.parts):
            return None
        labels = source_label_view(rows.writer, rows.readability(rows.clock()), [message])
        speakers = frozen_quote_speakers(
            [_Quoted((span,))], parts=message.parts, labels=labels, scope=scope,
        )
        return speakers.get((span.evidence_id, span.byte_start, span.byte_end))
