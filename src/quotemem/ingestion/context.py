"""Context selection and the extract-v12 request of one unit.

``build_context`` is a pure function of the owner pin's read view, the unit,
the frozen turn labels and the read-only context the owner supplies (the
previous batch and the request's earlier parts). It renders the sources,
builds the catalog, selects the context memories and returns the
`ExtractionContext` that ``build_request`` turns into the request
and admission reads, together with the `ExtractContextV2` the owner persists
before the call. It reads no tombstone and makes no provider call.

Memory selection scores every ``p_mem`` head of the scope:

- dense: the highest cosine (`quotemem.core.vector_score`: float64, rounded
  half-even to 6 decimals; floor 0.2) against at most 8 word-balanced groups of
  the fresh parts and at most 2 of the read-only parts, each group the
  normalized mean of its parts' reception vectors;
- lexical: BM25 (`SparseStatistics.bm25`, k1 1.2, b 0.75) of the fresh text's
  distinct tokens over each head's statement and keywords, with statistics over
  the heads at the pin; every head sharing a token is ranked;
- fusion ``1.0/(60 + r_dense) + 0.8/(60 + r_lex)``, the top 30 by fused score
  then id, plus at most 10 reserved memories: the best of each catalog entity
  that has an eligible memory but none in the top 30, in catalog order.

Each context memory lists its two most recent readable history entries as
``earlier`` versions. A head of origin ``extraction_supplement`` (an
overview or the assistant's work, ``core/extraction_supplement.py``), ``implied_health``
(an implied health condition of the same call) or ``session_rollup``
(``core/session_rollup.py``) is never selected, so the extraction call reads what it
would read without the supplement; the supplement call sends the same user message and
sees none either. Only the rollup call reads the newest rollup of the unit's session
(`session_rollup_of`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Final, Literal

from quotemem.core.add_read_view import (
    AddReadView,
    EvidenceRecord,
    MessageKey,
    MessageRecord,
    VersionRecord,
)
from quotemem.core.entities import USER
from quotemem.core.envelope import envelope_end
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_context import (
    ByteRangeV1,
    CatalogRefV1,
    ContextBindingV1,
    ContextMemoryRefV1,
    ContextSpanV1,
    ContextUnitV1,
    ExtractContextV2,
    LabelDecisionV1,
    MemberBindingV1,
    MessageLabelsV1,
    PreviousBatchV1,
    RollupRefV1,
    WithheldRangesV1,
)
from quotemem.core.extraction_output import ExtractRequestV12, SourceRole, render_wire_date
from quotemem.core.identity import ContentBindingDigest, IntegrityDigest, integrity_digest
from quotemem.core.query import ScopeV1
from quotemem.core.readability import WITHHELD_MARKER, Readability
from quotemem.core.recall_corpora import SparseStatistics, sparse_terms, statement_document
from quotemem.core.request_order import VersionOrder, version_order
from quotemem.core.session_rollup import PriorRollup, newest_per_session, prior_rollup
from quotemem.core.snapshot_memo import snapshot_rows
from quotemem.core.statement import (
    MAX_CONTEXT_MEMORIES,
    MAX_EARLIER_PER_MEMORY,
    MAX_RELEVANCE_MEMORIES,
    MAX_RESERVED_MEMORIES,
    MAX_SOURCES,
    MAX_TAIL_SOURCES,
    token_contained,
)
from quotemem.core.vector_score import UnitVector, unit_vector
from quotemem.core.work import FinalizeSourceHeaderV1
from quotemem.ingestion.catalog import (
    Catalog,
    CatalogEntry,
    CatalogRoute,
    build_catalog,
    speaker_mentions,
)
from quotemem.ingestion.inventory import load_request_inventories
from quotemem.ingestion.request_order import (
    RequestCoordinate,
    load_request_order,
    request_partitions,
)
from quotemem.ingestion.segment import message_chunk_index, message_order
from quotemem.ingestion.segment import message_key as order_key
from quotemem.ingestion.sources import (
    RENDER_VERSION,
    Batch,
    LabelView,
    MessageLabels,
    RenderedSource,
    SourceRun,
    TurnLabel,
    message_key,
    render_source,
    source_runs,
    tail_runs,
    word_count,
)
from quotemem.ingestion.units import Unit
from quotemem.store.indexed import IndexedReader
from quotemem.store.row_cache import note_reads
from quotemem.store.scope_evidence import scope_keys
from quotemem.store.writer import Writer

BindContent = Callable[[str, object], ContentBindingDigest]
"""The Writer's tenant-keyed content binding, ``bind(kind, value)``."""

DENSE_FLOOR = 0.2
FRESH_QUERY_GROUPS = 8
READ_ONLY_QUERY_GROUPS = 2
RRF_K = 60
DENSE_WEIGHT = 1.0
LEXICAL_WEIGHT = 0.8
TAIL_WORDS = 600
PREVIOUS_BATCH_WORDS = 16_000

MemoryOrigin = Literal["relevance", "reserved", "earlier"]
RECOUNTING_ORIGINS: Final = frozenset({
    "extraction_supplement", "implied_health", "session_rollup",
})
"""Memory origins of the supplement and rollup calls, which never enter the extraction
context: overviews and rollups recount whole sessions, and an implied health condition
must not change what the extraction call reads."""


@dataclass(frozen=True, slots=True)
class PreviousBatch:
    """The previous batch of the unit's session or source conversation.

    Attributes:
        messages (tuple[MessageRecord, ...]): Its whole messages, oldest first;
            the receipt-proven selector chose them.
    """

    messages: tuple[MessageRecord, ...]


@dataclass(frozen=True, slots=True)
class ContextMemory:
    """One context memory or earlier version shown to the extractor.

    Attributes:
        handle (str): ``M1``..``M40`` for a context memory, ``M41`` upwards for
            an earlier version.
        origin (MemoryOrigin): ``relevance``, ``reserved`` or ``earlier``.
        version (VersionRecord): The head, or the earlier version.
        of_handle (str | None): For an earlier version, its memory's handle.
    """

    handle: str
    origin: MemoryOrigin
    version: VersionRecord
    of_handle: str | None

    @property
    def memory_id(self) -> str:
        """The memory the version belongs to."""
        return self.version.memory_id


@dataclass(frozen=True, slots=True)
class RenderedBinding:
    """One SDK binding the request shows (``bindings``)."""

    source_id: str
    handle: str
    entity_id: str
    evidence_id: str
    byte_start: int
    byte_end: int
    text: str


@dataclass(frozen=True, slots=True)
class ExtractionContext:
    """Everything one unit's request shows, and its persisted record.

    Attributes:
        scope_id (str): The scope.
        unit (Unit): The unit.
        sources (tuple[RenderedSource, ...]): ``S1``..``Sn``: previous batch,
            tail, then the fresh sources.
        catalog (Catalog): The handle catalog.
        bindings (tuple[RenderedBinding, ...]): The SDK bindings shown.
        memories (tuple[ContextMemory, ...]): The context memories in handle
            order, then their earlier versions in handle order.
        record (ExtractContextV2): The persisted context.
        rollup (PriorRollup | None): The newest rollup of the unit's session at the
            pin (`session_rollup_of`), which only the rollup call reads. The record keeps
            its reference (``ExtractContextV2.rollup``), so a reconstructed context reads
            the same revision by id (`_recorded_rollup`) without selecting again.
        forgotten (tuple[str, ...]): The scope's forgotten statements (its tombstones at
            the pin), which the rollup call's input and its stored rollup leave out
            (``core/session_rollup.restates_forgotten``); not part of the record either.
    """

    scope_id: str
    unit: Unit
    sources: tuple[RenderedSource, ...]
    catalog: Catalog
    bindings: tuple[RenderedBinding, ...]
    memories: tuple[ContextMemory, ...]
    record: ExtractContextV2
    rollup: PriorRollup | None = None
    forgotten: tuple[str, ...] = ()
    _sources: dict[str, RenderedSource] = field(init=False, repr=False, compare=False)
    _memories: dict[str, ContextMemory] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_sources", {source.source_id: source for source in self.sources})
        object.__setattr__(self, "_memories", {memory.handle: memory for memory in self.memories})

    def source(self, source_id: str) -> RenderedSource | None:
        """Return an issued source by id, or None."""
        return self._sources.get(source_id)

    def memory(self, handle: str) -> ContextMemory | None:
        """Return an issued memory or earlier version by handle, or None."""
        return self._memories.get(handle)


def _query_groups(parts: Sequence[EvidenceRecord], limit: int) -> list[list[EvidenceRecord]]:
    """Cut parts into at most ``limit`` contiguous word-balanced groups.

    Group i closes at the first part where the cumulative words reach i·W/k,
    with k = min(limit, parts); the last group takes the rest.
    """
    count = min(limit, len(parts))
    if count == 0:
        return []
    words = [word_count(part.text) for part in parts]
    total = sum(words)
    groups: list[list[EvidenceRecord]] = []
    current: list[EvidenceRecord] = []
    cumulative, closing = 0, 1
    for part, part_words in zip(parts, words, strict=True):
        current.append(part)
        cumulative += part_words
        if closing < count and cumulative * count >= closing * total:
            groups.append(current)
            current, closing = [], closing + 1
    if current:
        groups.append(current)
    return groups


def _queries(
    sources: Sequence[RenderedSource], read: Readability, space: str,
) -> list[UnitVector]:
    """Return the memory-selection query vectors from the parts' reception vectors.

    Each is the unit vector of a group's summed vectors. Only parts that ``p_evr``
    admits contribute: their vectors encode no withheld or unreadable byte.
    """
    queries: list[UnitVector] = []
    for fresh, limit in ((True, FRESH_QUERY_GROUPS), (False, READ_ONLY_QUERY_GROUPS)):
        parts: dict[str, EvidenceRecord] = {}
        for source in sources:
            if source.fresh is fresh:
                for part in source.run.parts:
                    if read.p_evr(part, held=False) and any(part.embedding):
                        if part.embedding_space != space:
                            raise QuoteMemError(
                                "embedding_space_mismatch",
                                "A reception vector lies in another embedding space",
                            )
                        parts.setdefault(part.object_id, part)
        for group in _query_groups(list(parts.values()), limit):
            total = [math.fsum(column) for column in zip(
                *(part.embedding for part in group), strict=True,
            )]
            if any(total):
                queries.append(unit_vector(total))
    return queries


def _dense(heads: Sequence[VersionRecord], queries: Sequence[UnitVector]) -> dict[str, float]:
    """The best cosine of each head over the queries, where it reaches the floor.

    A ``p_mem`` head is never erased, so its vector has a direction.
    """
    scores: dict[str, float] = {}
    if not queries or not heads:
        return scores
    rows = snapshot_rows([head.snapshot for head in heads])
    per_query = [rows.cosines(query) for query in queries]
    for index, head in enumerate(heads):
        best = max(values[index] for values in per_query)
        if best >= DENSE_FLOOR:
            scores[head.memory_id] = best
    return scores


def _lexical(heads: Sequence[VersionRecord], terms: set[str]) -> dict[str, float]:
    """BM25 of the query terms over the heads' statement and keyword tokens.

    Every head sharing a term is scored, even when its score rounds to zero.
    """
    documents = [statement_document(head.snapshot) for head in heads]
    statistics = SparseStatistics.of(documents)
    return {
        head.memory_id: statistics.bm25(terms, document)
        for head, document in zip(heads, documents, strict=True)
        if not terms.isdisjoint(document.counts)
    }


def _ranks(
    scores: dict[str, float], heads: dict[str, VersionRecord],
) -> dict[str, int]:
    ordered = sorted(scores, key=lambda memory_id: (
        -scores[memory_id], version_order(heads[memory_id]),
    ))
    return {memory_id: rank for rank, memory_id in enumerate(ordered, start=1)}


def _newest_rollup(
    view: AddReadView, unit: Unit, withdrawn: frozenset[str],
) -> VersionRecord | None:
    """The head of the newest rollup of the unit's session at the pin (`session_rollup_of`)."""
    session = unit.parts[0].session_id

    def parts(head: VersionRecord) -> list[EvidenceRecord]:
        return [part for span in head.snapshot.quote_spans
                if (part := view.evidence(span.evidence_id)) is not None]

    rollups = [record.head for record in view.heads()
               if record.head.snapshot.origin == "session_rollup"]
    newest = newest_per_session(
        rollups,
        session_of=lambda head: next((part.session_id for part in parts(head)), None),
        recency=lambda head: (
            max((part.ingested_commit_seq for part in parts(head)), default=-1), head.commit_seq,
        ),
    ).get(session)
    return None if newest is None or newest.memory_id in withdrawn else newest


def session_rollup_of(
    view: AddReadView, unit: Unit, withdrawn: frozenset[str] = frozenset(),
) -> PriorRollup | None:
    """The newest rollup of the unit's session at the pin, or None before its first.

    A rollup belongs to the session of the parts it quotes; the newest is the one whose
    quoted parts were ingested last, then the one committed last, the same rule by which
    Search delivers one rollup per session (``core/recall.py``). A newest rollup that an
    in-chat forget withdrew is no prior: an older one may state the forgotten fact too, so
    the session's rollup starts again.

    Args:
        view: The owner pin's read view.
        unit: The unit.
        withdrawn: The scope's withdrawn summary-only memories (``forget_withdrawn:{s}``).

    Returns:
        The rollup's statement and its span's first day, or None.
    """
    newest = _newest_rollup(view, unit, withdrawn)
    return None if newest is None else prior_rollup(newest.snapshot.statement)


def _recorded_rollup(
    view: AddReadView, saved: RollupRefV1 | None, withdrawn: frozenset[str],
) -> PriorRollup | None:
    """The recorded prior rollup, read by id, while it still is what selection chose.

    Reconstruction selects no context, so it reads no heads. A prior that a later forget
    withdrew, erased (its head fails ``p_ver``), ended or revised is no prior any more, the
    same as at selection: the session's rollup starts again.
    """
    if saved is None or saved.memory_id in withdrawn:
        return None
    found = view.memory(saved.memory_id)
    if found is None or found.lifecycle_state != "active" or (
        found.head.revision_id != saved.revision_id
    ):
        return None
    return prior_rollup(found.head.snapshot.statement)


def _forgotten(view: AddReadView) -> tuple[str, ...]:
    """The scope's forgotten statements at the pin, which no rollup may restate."""
    return tuple(tombstone.statement for tombstone in view.tombstones())


Hidden = Callable[[VersionRecord], bool]
"""Whether an accepted forget of the scope withholds a version from reads until its erasure
ends (``quotemem.store.erasure_gate.withheld_versions``)."""


def select_memories(
    view: AddReadView, sources: Sequence[RenderedSource], catalog: Catalog,
    hidden: Hidden | None = None,
) -> tuple[ContextMemory, ...]:
    """Select the context memories of one request and their earlier versions.

    A version an accepted forget withholds (``hidden``) is neither selected nor shown as
    an earlier version, as its erasure will leave it out: the extraction call never selects
    a memory version a waiting forget withholds. The selection is recorded with the
    context, so a resumed unit reads the same memories whatever is withheld then. The
    rendered sources (the unit's earlier parts and the previous batch) are not masked:
    they are rendered again on a resumed unit and checked byte for byte, and a waiting
    forget's withheld set is not stored; a restatement the model makes from them is the
    pre-existing limit that a confirmation takes nothing its prepare did not name.

    Args:
        view (AddReadView): The owner pin's read view.
        sources (Sequence[RenderedSource]): The rendered sources.
        catalog (Catalog): The catalog of routes 1 to 4, whose order
            the reserved slots follow.
        hidden (Hidden | None): What an accepted forget withholds now; None when
            nothing is withheld.

    Returns:
        tuple[ContextMemory, ...]: The relevance memories by rank, the reserved
            ones in catalog order, then the earlier versions; handles ``M1``
            upwards and ``M41`` upwards.

    Raises:
        QuoteMemError: ``embedding_space_mismatch`` if a reception vector lies in another
            embedding space.
    """
    read = Readability(view.scope_kind, view.scope_id, view.now, view.evidence)
    # The supplement call's memories and the session rollups stay out: overviews and
    # rollups recount whole sessions, so they would fill the context and lengthen the
    # call, and no supplement memory may change what the extraction call reads.
    heads = [record.head for record in view.heads()
             if record.head.snapshot.origin not in RECOUNTING_ORIGINS
             and (hidden is None or not hidden(record.head))]
    terms = {
        token for source in sources if source.fresh
        for text in source.readable_texts() for token in sparse_terms(text)
    }
    by_id = {head.memory_id: head for head in heads}
    dense = _ranks(_dense(heads, _queries(sources, read, view.embedding_space)), by_id)
    lexical = _ranks(_lexical(heads, terms), by_id)
    fused = {
        head.memory_id: (
            (DENSE_WEIGHT / (RRF_K + dense[head.memory_id]) if head.memory_id in dense else 0.0)
            + (LEXICAL_WEIGHT / (RRF_K + lexical[head.memory_id])
               if head.memory_id in lexical else 0.0)
        )
        for head in heads
    }
    by_id = {head.memory_id: head for head in heads}
    ranked = sorted(
        (memory_id for memory_id, score in fused.items() if score > 0),
        key=lambda memory_id: (-fused[memory_id], version_order(by_id[memory_id])),
    )[:MAX_RELEVANCE_MEMORIES]
    chosen: list[tuple[MemoryOrigin, VersionRecord]] = [
        ("relevance", by_id[memory_id]) for memory_id in ranked
    ]
    taken = set(ranked)

    def preference(head: VersionRecord) -> tuple[float, int, VersionOrder]:
        # The best anchored memory: the highest fused score, else the newest.
        return (-fused[head.memory_id], -head.recorded_at, version_order(head))

    reserved = 0
    for entry in catalog.entries:
        if reserved == MAX_RESERVED_MEMORIES:
            break
        anchored = [
            head for head in heads
            if entry.entity_id is not None and entry.entity_id in head.anchors
        ]
        if not anchored or any(head.memory_id in taken for head in anchored):
            continue
        best = min(anchored, key=preference)
        chosen.append(("reserved", best))
        taken.add(best.memory_id)
        reserved += 1
    memories = [
        ContextMemory(f"M{number}", origin, version, None)
        for number, (origin, version) in enumerate(chosen, start=1)
    ]
    earlier: list[ContextMemory] = []
    for memory in memories:
        for entry in sorted(
            # A description a rewording replaced is no earlier version to show;
            # after a forget of its correction it is tombstoned text.
            (entry for entry in view.history(memory.memory_id) if not entry.reworded
             and (hidden is None or not hidden(entry.version))),
            key=lambda entry: (entry.version.recorded_at, version_order(entry.version)),
            reverse=True,
        )[:MAX_EARLIER_PER_MEMORY]:
            handle = f"M{MAX_CONTEXT_MEMORIES + 1 + len(earlier)}"
            earlier.append(ContextMemory(handle, "earlier", entry.version, memory.handle))
    return (*memories, *earlier)


def _render_all(
    runs: Sequence[SourceRun], batch: Batch, read: Readability, scope_id: str,
) -> list[RenderedSource]:
    rendered = (
        render_source(run, batch=batch, source_id="S0", read=read, scope_id=scope_id)
        for run in runs
    )
    return [source for source in rendered if source is not None]


def _previous(
    previous: PreviousBatch | None, labels: LabelView, read: Readability, scope_id: str,
    *, budget: int, max_words: int,
) -> tuple[list[RenderedSource], PreviousBatchV1 | None]:
    """Keep the newest whole previous-batch messages within the word and source budgets."""
    if previous is None or not previous.messages:
        return [], None
    kept: list[tuple[MessageRecord, list[RenderedSource]]] = []
    words = 0
    for message in reversed(previous.messages):
        rendered = _render_all(source_runs(message.parts, labels), "previous", read, scope_id)
        size = sum(word_count(part.text) for part in message.parts)
        count = sum(len(sources) for _, sources in kept)
        if words + size > max_words or count + len(rendered) > budget:
            break
        kept.append((message, rendered))
        words += size
    kept.reverse()
    if not kept:
        first = previous.messages[0].key
        return [], PreviousBatchV1(
            session_id=first.session_id, request_id=first.request_id,
            first_kept_ordinal=previous.messages[-1].key.ordinal + 1, truncated=True,
        )
    oldest = kept[0][0].key
    return [source for _, sources in kept for source in sources], PreviousBatchV1(
        session_id=oldest.session_id, request_id=oldest.request_id,
        first_kept_ordinal=oldest.ordinal, truncated=len(kept) < len(previous.messages),
    )


def _bindings(sources: Sequence[RenderedSource], catalog: Catalog) -> list[RenderedBinding]:
    found: list[RenderedBinding] = []
    for source in sources:
        for part in source.run.parts:
            for binding in sorted(part.entity_bindings, key=lambda item: item.byte_start):
                text = source.rendered_bytes(part.object_id, binding.byte_start, binding.byte_end)
                handle = catalog.handle_of(binding.entity_id)
                if text is not None and handle is not None:
                    found.append(RenderedBinding(
                        source.source_id, handle, binding.entity_id, part.object_id,
                        binding.byte_start, binding.byte_end, text,
                    ))
    return found


def wire_role(source: RenderedSource) -> SourceRole:
    """The role a source shows the extractor.

    A labelled participant's turn behind an AMLB envelope header shows
    ``participant``: the envelope's wire role is transport framing, and a
    ``user`` role there would read as the authenticated user's own line. Every
    other source shows its message role.

    Args:
        source (RenderedSource): The source.

    Returns:
        SourceRole: ``participant`` or the message role.
    """
    label = source.run.label
    if label is not None and label.framed and source.attribution.participant_kind == "label":
        return "participant"
    return source.role


def _request(
    sources: Sequence[RenderedSource], catalog: Catalog,
    bindings: Sequence[RenderedBinding], memories: Sequence[ContextMemory],
) -> ExtractRequestV12:
    def speaker_handle(source: RenderedSource) -> str | None:
        attribution = source.attribution
        if attribution.authenticated_self:
            return catalog.handle_of(USER)
        if attribution.participant_key is not None:
            return catalog.handle_of_participant(attribution.participant_key)
        return None

    def date(version: VersionRecord, value: int | None) -> str | None:
        return render_wire_date(value, version.snapshot.time_granularity)

    current = [memory for memory in memories if memory.origin != "earlier"]
    return ExtractRequestV12.model_validate({
        "sources": [
            {
                "source_id": source.source_id, "fresh": source.fresh, "batch": source.batch,
                "role": wire_role(source), "speaker": source.attribution.speaker,
                "speaker_handle": speaker_handle(source),
                "observed_at": render_wire_date(source.observed_at, source.granularity),
                "text": source.text,
            }
            for source in sources
        ],
        "entities": [
            {
                "handle": entry.handle, "kind": entry.kind, "type": entry.entity_type,
                "names": [{"name": row.name, "kind": row.name_kind} for row in entry.names],
                "speaker_names": list(entry.speaker_names), "memories": list(entry.memories),
            }
            for entry in catalog.entries
        ],
        "bindings": [
            {"source_id": binding.source_id, "text": binding.text, "handle": binding.handle}
            for binding in bindings
        ],
        "memories": [
            {
                "handle": memory.handle, "statement": memory.version.snapshot.statement,
                "kind": memory.version.snapshot.kind,
                "valid_from": date(memory.version, memory.version.snapshot.valid_from),
                "valid_to": date(memory.version, memory.version.snapshot.valid_to),
                "anchors": [catalog.handle_of(anchor) for anchor in (
                    *memory.version.anchors[:1], *sorted(
                        memory.version.anchors[1:],
                        key=lambda anchor: next((index for index, entry in
                            enumerate(catalog.entries) if entry.entity_id == anchor),
                            len(catalog.entries)),
                    ),
                )],
                "earlier": [
                    {
                        "handle": version.handle, "statement": version.version.snapshot.statement,
                        "valid_from": date(version.version, version.version.snapshot.valid_from),
                        "valid_to": date(version.version, version.version.snapshot.valid_to),
                    }
                    for version in memories if version.of_handle == memory.handle
                ],
            }
            for memory in current
        ],
        "catalog_truncated": catalog.truncated,
    })


def build_request(context: ExtractionContext) -> ExtractRequestV12:
    """Build the user message of one extract-v12 call from its context.

    Args:
        context (ExtractionContext): The unit's context.

    Returns:
        ExtractRequestV12: Its canonical JSON is the call's user message; the
            system message is the policy's template exactly.
    """
    return _request(context.sources, context.catalog, context.bindings, context.memories)


def _spans(runs: Sequence[RenderedSource]) -> list[ContextSpanV1]:
    return [
        ContextSpanV1(
            evidence_id=part.object_id,
            byte_start=max(source.run.byte_start, part.byte_start),
            byte_end=min(source.run.byte_end, part.byte_end),
        )
        for source in runs for part in source.run.parts
        if part.state != "erased"
        and max(source.run.byte_start, part.byte_start) < min(source.run.byte_end, part.byte_end)
    ]


def _record(
    view: AddReadView, unit: Unit, owner_run_id: str, labels: LabelView,
    sources: Sequence[RenderedSource], previous: PreviousBatchV1 | None, catalog: Catalog,
    bindings: Sequence[RenderedBinding], memories: Sequence[ContextMemory],
    request: ExtractRequestV12, bind: BindContent, render_digest: IntegrityDigest,
    rollup: RollupRefV1 | None,
) -> ExtractContextV2:
    withheld: dict[str, list[tuple[int, int]]] = {}
    messages: dict[tuple[str, str, int], MessageLabelsV1] = {}
    for source in sources:
        for part in source.run.parts:
            if part.withheld_spans and part.object_id not in withheld:
                withheld[part.object_id] = sorted({
                    (span.byte_start, span.byte_end) for span in part.withheld_spans
                })
        key = source.run.message
        turns = labels.turns(key)
        if turns and (key.session_id, key.request_id, key.ordinal) not in messages:
            messages[(key.session_id, key.request_id, key.ordinal)] = MessageLabelsV1(
                session_id=key.session_id, request_id=key.request_id, ordinal=key.ordinal,
                labels=[
                    LabelDecisionV1(
                        byte_start=label.byte_start, byte_end=label.byte_end, state=label.state,
                    )
                    for label in turns
                ],
            )
    # The input digest binds what the fresh sources say; their ids and speaker
    # handles depend on the rest of the context and are bound with it.
    dumped = request.model_dump(mode="json")
    positional = ("source_id", "speaker_handle")
    fresh = [
        {name: value for name, value in source.items() if name not in positional}
        for source in dumped["sources"] if source["fresh"]
    ]
    shown = {**dumped, "sources": [
        source if not source["fresh"] else {name: source[name] for name in positional}
        for source in dumped["sources"]
    ]}
    pin = view.pin
    return ExtractContextV2(
        schema_version=2,
        unit=ContextUnitV1(
            owner_run_id=owner_run_id, session_id=unit.session_id, request_id=unit.request_id,
            unit_no=unit.unit_no,
            member_bindings=[
                MemberBindingV1(
                    evidence_id=part.object_id, request_id=part.request_id,
                    ordinal=part.ordinal, part_no=part.part_no,
                )
                for part in unit.parts
            ],
        ),
        previous=previous,
        tail=_spans([source for source in sources if source.batch == "tail"]),
        withheld=[
            WithheldRangesV1(evidence_id=evidence_id, ranges=[
                ByteRangeV1(byte_start=start, byte_end=end) for start, end in ranges
            ])
            for evidence_id, ranges in sorted(withheld.items())
        ],
        labels=[messages[key] for key in sorted(messages)],
        catalog=[
            CatalogRefV1(
                handle=entry.handle, kind=entry.kind, entity_id=entry.entity_id,
                participant_key=entry.participant_key,
                name_row_ids=[row.object_id for row in entry.names],
            )
            for entry in catalog.entries
        ],
        catalog_truncated=catalog.truncated,
        bindings=[
            ContextBindingV1(
                evidence_id=binding.evidence_id, byte_start=binding.byte_start,
                byte_end=binding.byte_end, handle=binding.handle,
            )
            for binding in bindings
        ],
        memories=[
            ContextMemoryRefV1(
                handle=memory.handle, memory_id=memory.memory_id,
                revision_id=memory.version.revision_id,
                content_hash=integrity_digest(memory.version.snapshot),
                origin=memory.origin, of_handle=memory.of_handle,
            )
            for memory in memories
        ],
        read_view_commit_seq=pin.commit_seq,
        tombstone_seq=pin.tombstone_seq,
        erase_epoch=pin.erase_epoch,
        consent_epoch=pin.consent_epoch,
        render_config_digest=render_digest,
        input_digest=bind("extract-input", fresh),
        context_digest=bind("extract-context", shown),
        rollup=rollup,
    )


def build_context(
    view: AddReadView,
    unit: Unit,
    *,
    owner_run_id: str,
    labels: LabelView,
    bind: BindContent,
    preceding: Sequence[EvidenceRecord] = (),
    previous: PreviousBatch | None = None,
    tail_words: int = TAIL_WORDS,
    previous_words: int = PREVIOUS_BATCH_WORDS,
    withdrawn: frozenset[str] = frozenset(),
    hidden: Hidden | None = None,
) -> ExtractionContext | None:
    """Select the context of one unit at the owner pin.

    Args:
        view (AddReadView): The owner pin's read view.
        unit (Unit): The unit.
        owner_run_id (str): The Add owner run.
        labels (LabelView): The frozen turn labels.
        bind (BindContent): The Writer's content binding, for the digests.
        preceding (Sequence[EvidenceRecord]): The parts of the unit's request
            before the unit, in message order; their newest runs are the tail.
        previous (PreviousBatch | None): The previous batch, if any.
        tail_words (int): The policy's ``tail_words`` (600).
        previous_words (int): The policy's ``previous_batch_words`` (16,000).
        withdrawn (frozenset[str]): The scope's withdrawn summary-only memories
            (`session_rollup_of`).
        hidden (Hidden | None): What an accepted forget withholds now
            (`select_memories`).

    Returns:
        ExtractionContext | None: The context, or None when no fresh byte of the
            unit is readable: such a unit makes no extraction call.

    Raises:
        QuoteMemError: ``embedding_space_mismatch`` if a reception vector lies in another
            embedding space.
    """
    read = Readability(view.scope_kind, view.scope_id, view.now, view.evidence)
    scope_id = view.scope_id
    fresh = _render_all(source_runs(unit.parts, labels), "current", read, scope_id)
    if not fresh:
        return None
    tail = _render_all(
        tail_runs(source_runs(preceding, labels), max_sources=MAX_TAIL_SOURCES,
                  max_words=tail_words),
        "tail", read, scope_id,
    )
    before, previous_record = _previous(
        previous, labels, read, scope_id,
        budget=MAX_SOURCES - len(fresh) - len(tail), max_words=previous_words,
    )
    sources = tuple(
        replace(source, source_id=f"S{number}")
        for number, source in enumerate((*before, *tail, *fresh), start=1)
    )
    memories = select_memories(view, sources, build_catalog(view, sources), hidden)
    catalog = build_catalog(view, sources, [
        (memory.handle, memory.version.anchors) for memory in memories
        if memory.origin != "earlier"
    ])
    bindings = tuple(_bindings(sources, catalog))
    request = _request(sources, catalog, bindings, memories)
    render_digest = _render_config_digest(tail_words, previous_words)
    newest = _newest_rollup(view, unit, withdrawn)
    record = _record(
        view, unit, owner_run_id, labels, sources, previous_record, catalog, bindings,
        memories, request, bind, render_digest,
        None if newest is None else RollupRefV1(
            memory_id=newest.memory_id, revision_id=newest.revision_id,
        ),
    )
    return ExtractionContext(
        scope_id, unit, sources, catalog, bindings, memories, record,
        None if newest is None else prior_rollup(newest.snapshot.statement), _forgotten(view),
    )



def _render_config_digest(tail_words: int, previous_words: int) -> IntegrityDigest:
    return integrity_digest({
        "render_version": RENDER_VERSION, "withheld_marker": WITHHELD_MARKER,
        "tail_sources": MAX_TAIL_SOURCES, "tail_words": tail_words,
        "previous_words": previous_words, "max_sources": MAX_SOURCES,
    })


def _context_mismatch() -> QuoteMemError:
    return QuoteMemError("binding_mismatch", "The recorded extraction context cannot be reproduced")


def _recorded_labels(
    view: AddReadView, record: ExtractContextV2, read: Readability,
) -> MessageLabels:
    messages: dict[MessageKey, tuple[TurnLabel, ...]] = {}
    for saved in record.labels:
        key = MessageKey(saved.session_id, saved.request_id, saved.ordinal)
        message = view.message(key)
        if message is None or key in messages:
            raise _context_mismatch()
        turns: list[TurnLabel] = []
        for label in saved.labels:
            rendered = read.render_masked(
                message.parts, start=label.byte_start, end=label.byte_end,
            )
            visible = sum(segment.msg_byte_end - segment.msg_byte_start
                          for segment in rendered.segments)
            complete = visible == label.byte_end - label.byte_start
            # The recorded range begins at the name, or at the envelope header
            # before it, and ends after its colon. Ambiguous privacy/refusal
            # barriers may have no name or colon.
            name = ""
            head = envelope_end(rendered.text) if complete else 0
            if complete and rendered.text.endswith((":", "：")):
                name = rendered.text[head:-1].strip(" \t\u00a0\u3000")
            if label.state == "speaker" and not name:
                raise _context_mismatch()
            turns.append(TurnLabel(label.byte_start, label.byte_end, name, label.state, head > 0))
        messages[key] = tuple(turns)
    return MessageLabels(messages)


def _recorded_tail(
    view: AddReadView, record: ExtractContextV2, labels: LabelView,
) -> list[SourceRun]:
    ranges: dict[MessageKey, list[ContextSpanV1]] = {}
    for span in record.tail:
        part = view.evidence(span.evidence_id)
        if part is None or (part.session_id, part.request_id) != (
            record.unit.session_id, record.unit.request_id,
        ) or not part.byte_start <= span.byte_start < span.byte_end <= part.byte_end:
            raise _context_mismatch()
        ranges.setdefault(message_key(part), []).append(span)
    runs: list[SourceRun] = []
    for key, spans in ranges.items():
        message = view.message(key)
        if message is None:
            raise _context_mismatch()
        for run in source_runs(message.parts, labels):
            included = [span for span in spans
                        if span.byte_start < run.byte_end and run.byte_start < span.byte_end]
            if included:
                low = max(run.byte_start, min(span.byte_start for span in included))
                high = min(run.byte_end, max(span.byte_end for span in included))
                parts = tuple(part for part in run.parts if part.state == "erased" or
                              part.byte_start < high and low < part.byte_end)
                runs.append(replace(run, parts=parts, byte_start=low, byte_end=high))
    return runs


def _recorded_memories(view: AddReadView, record: ExtractContextV2) -> tuple[ContextMemory, ...]:
    memories: dict[str, ContextMemory] = {}
    for saved in record.memories:
        if saved.handle in memories:
            raise _context_mismatch()
        if saved.origin == "earlier":
            parent = memories.get(saved.of_handle or "")
            if parent is None or parent.origin == "earlier":
                raise _context_mismatch()
            version = next((entry.version for entry in view.history(parent.memory_id)
                            if entry.version.revision_id == saved.revision_id), None)
        else:
            memory = view.memory(saved.memory_id)
            version = None if memory is None or memory.lifecycle_state != "active" else memory.head
        if version is None or (version.memory_id, version.revision_id) != (
            saved.memory_id, saved.revision_id,
        ) or integrity_digest(version.snapshot) != saved.content_hash:
            raise _context_mismatch()
        memories[saved.handle] = ContextMemory(
            saved.handle, saved.origin, version, saved.of_handle,
        )
    return tuple(memories.values())


def _recorded_catalog(
    view: AddReadView, record: ExtractContextV2, sources: Sequence[RenderedSource],
    memories: Sequence[ContextMemory],
) -> Catalog:
    entries: list[CatalogEntry] = []
    mentions = speaker_mentions(view.scope_id, sources, {
        saved.participant_key for saved in record.catalog if saved.participant_key is not None
    }, view.speakers())
    for saved in record.catalog:
        entity = None if saved.entity_id is None else view.entity(saved.entity_id)
        if saved.entity_id is not None and entity is None:
            raise _context_mismatch()
        available = {} if entity is None else {name.object_id: name for name in entity.names}
        if any(row_id not in available for row_id in saved.name_row_ids):
            raise _context_mismatch()
        if entity is not None and (entity.entity_id != saved.entity_id or
                                   entity.participant_key != saved.participant_key):
            raise _context_mismatch()
        spoken = [source for kind in ("tuple", "label") for source in sources
                  if saved.participant_key is not None
                  and source.attribution.participant_key == saved.participant_key
                  and source.attribution.participant_kind == kind]
        sdk = any(binding.entity_id == saved.entity_id and source.rendered_bytes(
            part.object_id, binding.byte_start, binding.byte_end,
        ) is not None for source in sources for part in source.run.parts
                  for binding in part.entity_bindings)
        mentioned = () if spoken or saved.participant_key is None else mentions.names.get(
            saved.participant_key, (),
        )
        route: CatalogRoute
        if saved.kind == "user":
            if saved.handle != USER or saved.entity_id != USER:
                raise _context_mismatch()
            route = "self" if any(source.attribution.authenticated_self
                                  for source in sources) else "memory"
        elif spoken:
            route = "participant" if any(source.attribution.participant_kind == "tuple"
                                         for source in spoken) else "label"
        elif sdk:
            route = "sdk"
        else:
            # The name route precedes the memory route. Read all readable names
            # at the pin, since the matching name need not be one of the eight
            # names shown, and the labels the sources name. This resolves a
            # recorded entry; it nominates nothing.
            named = entity is not None and (bool(mentioned) or any(
                token_contained(name.name, text) for name in entity.names
                for source in sources for text in source.readable_texts()
            ))
            route = "name" if named else "memory"
        if saved.entity_id is None and (saved.kind != "participant" or not spoken):
            raise _context_mismatch()
        entries.append(CatalogEntry(
            handle=saved.handle, kind=saved.kind, route=route, entity_id=saved.entity_id,
            participant_key=saved.participant_key,
            entity_type="person" if saved.kind == "user" or entity is None else entity.entity_type,
            names=tuple(available[row_id] for row_id in saved.name_row_ids),
            speaker_names=tuple(dict.fromkeys(
                name for source in spoken for name in source.attribution.names
            )) if spoken else mentioned if route == "name" else (),
            memories=tuple(memory.handle for memory in memories if memory.origin != "earlier"
                           and saved.entity_id in memory.version.anchors),
        ))
    return Catalog(tuple(entries), record.catalog_truncated)


def reconstruct_context(
    view: AddReadView,
    record: ExtractContextV2,
    *,
    owner_run_id: str,
    bind: BindContent,
    check_valid: Callable[[], None],
    tail_words: int = TAIL_WORDS,
    previous_words: int = PREVIOUS_BATCH_WORDS,
    withdrawn: frozenset[str] = frozenset(),
) -> ExtractionContext:
    """Re-render a persisted context without selecting context or classifying labels.

    The caller holds the original owner's guarded read view throughout this
    call. A historical pin alone is not authorization: ``check_valid`` must
    compose the current owner/lease checks with the read guard's operation
    check. The function invokes it before reading and immediately before
    returning. Only recorded choices are hydrated, through the view's predicate
    filtering and the shared masked renderer; changed bytes fail closed.

    A cold-adopted owner holds the current root, a later commit than its
    records' pin. Before its first commit nothing of its own is in
    that root, so a record re-rendered there whose every render and choice is
    unchanged is the same context, and it is reproduced as recorded.

    Args:
        view (AddReadView): Guarded view at the recorded owner pin, or at a later
            commit with the same scope epochs, with the current trusted clock and
            the complete original requests available.
        record (ExtractContextV2): The persisted, validated context record.
        owner_run_id (str): The owner resuming this unit.
        bind (BindContent): The same tenant-keyed binding used before persistence.
        check_valid (Callable[[], None]): Required validity check. The caller
            supplies the existing owner lease and snapshot operation checks,
            raising their existing errors on revoked ownership or privacy access.
        tail_words (int): The owner's frozen tail word policy.
        previous_words (int): The owner's frozen previous-batch word policy.
        withdrawn (frozenset[str]): The scope's withdrawn summary-only memories
            (``forget_withdrawn:{s}``), which drop the recorded prior rollup
            (`_recorded_rollup`).

    Returns:
        ExtractionContext: A context whose record and request bindings equal the
            persisted record, ready for admission replay or the call path.

    Raises:
        QuoteMemError: ``binding_mismatch`` if the owner, the pin's scope epochs, an
            earlier commit, configuration, locators, readable content or request
            digests differ. Validity-check errors
            propagate unchanged. A lost fresh source fails instead of selecting
            a replacement or returning a partially reconstructed request.
        ValueError: A malformed source range fails the shared renderer or strict
            request validation. No request is returned.
    """
    check_valid()
    pin = view.pin
    render_digest = _render_config_digest(tail_words, previous_words)
    if owner_run_id != record.unit.owner_run_id or (
        pin.tombstone_seq, pin.erase_epoch, pin.consent_epoch,
    ) != (record.tombstone_seq, record.erase_epoch, record.consent_epoch) or (
        pin.commit_seq < record.read_view_commit_seq
    ) or render_digest != record.render_config_digest:
        raise _context_mismatch()
    read = Readability(view.scope_kind, view.scope_id, view.now, view.evidence)
    labels = _recorded_labels(view, record, read)
    parts: list[EvidenceRecord] = []
    for member in record.unit.member_bindings:
        part = view.evidence(member.evidence_id)
        if part is None or (part.scope_kind, part.scope_id, part.session_id, part.request_id,
                            part.ordinal, part.part_no) != (
            view.scope_kind, view.scope_id, record.unit.session_id, record.unit.request_id,
            member.ordinal, member.part_no,
        ) or member.request_id != record.unit.request_id:
            raise _context_mismatch()
        parts.append(part)
    unit = Unit(record.unit.unit_no, tuple(parts))
    fresh = _render_all(source_runs(unit.parts, labels), "current", read, view.scope_id)
    if not fresh:
        raise _context_mismatch()
    tail = _render_all(_recorded_tail(view, record, labels), "tail", read, view.scope_id)
    before: list[RenderedSource] = []
    if record.previous is not None:
        previous = record.previous
        ordinal = previous.first_kept_ordinal
        while (message := view.message(MessageKey(
            previous.session_id, previous.request_id, ordinal,
        ))) is not None:
            before.extend(_render_all(source_runs(message.parts, labels), "previous",
                                      read, view.scope_id))
            ordinal += 1
    sources = tuple(replace(source, source_id=f"S{number}")
                    for number, source in enumerate((*before, *tail, *fresh), start=1))
    memories = _recorded_memories(view, record)
    catalog = _recorded_catalog(view, record, sources, memories)
    bindings = tuple(_bindings(sources, catalog))
    request = _request(sources, catalog, bindings, memories)
    rebuilt = _record(view, unit, owner_run_id, labels, sources, record.previous, catalog,
                      bindings, memories, request, bind, render_digest, record.rollup)
    if rebuilt.model_copy(update={"read_view_commit_seq": record.read_view_commit_seq}) != record:
        raise _context_mismatch()
    check_valid()
    return ExtractionContext(view.scope_id, unit, sources, catalog, bindings, memories, record,
                             _recorded_rollup(view, record.rollup, withdrawn), _forgotten(view))


@dataclass(frozen=True, slots=True)
class ScopeRequests:
    """The scope's Evidence at one owner pin, grouped by request.

    One complete scan serves the previous-batch selection of every request of the
    pinned view: the selection reads nothing else of the scan, and the scan reads
    only the pin.

    Attributes:
        pin (int): The pin's ``commit_seq`` the scan read up to.
        descriptors (int): How many Evidence rows the scan read.
        batches (dict[tuple[str, str], list[FinalizeSourceHeaderV1]]): Each
            request's row headers in scan order, by ``(session_id, request_id)``.
        conversations (dict[tuple[str, str], set[tuple[str, str]]]): Each
            request's source conversations.
    """

    pin: int
    descriptors: int
    batches: dict[tuple[str, str], list[FinalizeSourceHeaderV1]]
    conversations: dict[tuple[str, str], set[tuple[str, str]]]


def scan_scope_requests(
    writer: Writer, view: AddReadView, *,
    check_valid: Callable[[], None], max_descriptors: int = 100_000,
) -> ScopeRequests:
    """Scan the scope's Evidence headers at the owner pin once for every previous-batch selection.

    The selection reads only each row's fixed source header. Reading complete
    rows decoded and validated every earlier row's text and embedding vector
    in every Add: a BEAM-1M user's 6,000 rows, once per Add.
    The headers come from the Writer's per-scope index, which
    re-reads only rows retired since its last read.

    Args:
        writer (Writer): Persistence owner, used only for bounded indexed reads.
        view (AddReadView): The owner's guarded, frozen scope view.
        check_valid (Callable[[], None]): Check the owner lease and held privacy
            operation before each page.
        max_descriptors (int): Complete scan ceiling; exceeding it refuses.

    Returns:
        ScopeRequests: The complete scan.

    Raises:
        QuoteMemError: ``budget_exceeded`` past the bound; ``scope_denied`` for a row
            outside the scope or after the pin.
    """
    index = IndexedReader(writer.runtime.workspace, writer.entities, tenant_id=writer.tenant_id)
    scope = ScopeV1(scope_kind=view.scope_kind, scope_id=view.scope_id)
    batches: dict[tuple[str, str], list[FinalizeSourceHeaderV1]] = {}
    check_valid()
    snapshot = writer.scope_evidence().snapshot(index, scope, check_valid)
    check_valid()
    pin = view.pin.commit_seq
    count = 0
    for row in snapshot.headers.values():
        if row.ingested_commit_seq > pin:
            continue
        if (row.tenant_id, row.scope_kind, row.scope_id) != (
            writer.tenant_id, view.scope_kind, view.scope_id,
        ):
            raise QuoteMemError("scope_denied", "Previous-batch scan crossed the owner pin")
        count += 1
        if count > max_descriptors:
            raise QuoteMemError("budget_exceeded", "Previous-batch scan exceeds its complete bound")
        batches.setdefault((row.session_id, row.request_id), []).append(row)
    note_reads((*scope_keys(scope), *(
        ("evidence", row.object_id) for rows in batches.values() for row in rows
    )))
    conversations = {
        key: {(row.source_participant.namespace, row.source_participant.conversation_id)
              for row in rows if row.source_participant is not None}
        for key, rows in batches.items()
    }
    return ScopeRequests(view.pin.commit_seq, count, batches, conversations)


def select_previous_batch(
    writer: Writer, view: AddReadView, unit: Unit, *,
    check_valid: Callable[[], None], max_descriptors: int = 100_000,
    scan: ScopeRequests | None = None,
) -> PreviousBatch | None:
    """Select the immediate receipt-proven predecessor at the owner pin.

    Args:
        writer (Writer): Persistence owner, used only for bounded indexed reads.
        view (AddReadView): The owner's guarded, frozen scope view.
        unit (Unit): A unit of the current request.
        check_valid (Callable[[], None]): Check the owner lease and held privacy
            operation before reads and before returning.
        max_descriptors (int): Complete scan ceiling; exceeding it refuses.
        scan (ScopeRequests | None): The pinned view's complete scan
            (``scan_scope_requests``), shared by the selections of its requests;
            None scans here.

    Returns:
        PreviousBatch | None: Complete messages of the immediate previous
            request in the same session or source conversation. Masking and
            whole-message trimming happen in build_context. An unavailable
            predecessor is never replaced by an older request.

    Raises:
        QuoteMemError: On incomplete receipts, changed sources, mixed conversations
            or exhausted bounds. Ambiguous erased conversation gaps fail closed
            pending the owner's cleanup proof; ``binding_mismatch`` for a scan of
            another pin.
    """
    check_valid()
    if max_descriptors < 1:
        raise QuoteMemError("invalid_request", "Previous-batch scan requires a positive bound")
    if scan is None:
        scan = scan_scope_requests(writer, view, check_valid=check_valid,
                                   max_descriptors=max_descriptors)
    elif scan.pin != view.pin.commit_seq:
        raise QuoteMemError("binding_mismatch", "A previous-batch scan belongs to another pin")
    elif scan.descriptors > max_descriptors:
        raise QuoteMemError("budget_exceeded", "Previous-batch scan exceeds its complete bound")
    scope = ScopeV1(scope_kind=view.scope_kind, scope_id=view.scope_id)
    batches, conversations = scan.batches, scan.conversations
    current = unit.session_id, unit.request_id
    if current not in batches:
        raise QuoteMemError("binding_mismatch", "Current request is absent at its owner pin")
    if len(conversations[current]) > 1:
        raise QuoteMemError("binding_mismatch", "Current request mixes source conversations")
    conversation = next(iter(conversations[current]), None)
    compatible = {
        key: rows for key, rows in batches.items()
        if (key[0] == current[0] or conversation is not None and (
            conversation in conversations[key]
            or not conversations[key] and any(row.state == "erased" for row in rows)
        )) and not (conversations[key] and conversation is not None
                    and conversation not in conversations[key])
    }
    orders: dict[tuple[str, str], RequestCoordinate] = {}
    for session in sorted({key[0] for key in compatible}):
        anchors, _ = load_request_order(
            writer, scope, session, None,
            at_time=view.now, input_high_watermark=view.pin.commit_seq,
        )
        times = [row.source_observed_at for rows in batches.values() for row in rows
                 if row.session_id == session and row.source_observed_at is not None]
        for request, anchor in anchors.items():
            key = session, request
            if key not in batches:
                raise QuoteMemError("reference_missing", "A receipt lost its complete source set")
            if key not in compatible:
                continue
            first = min(compatible[key], key=lambda row: (row.ordinal, row.part_no, row.object_id))
            if (anchor.first, anchor.chunk_event_time, anchor.received_commit_seq,
                anchor.source_time_present) != (
                (first.ordinal, first.part_no, first.object_id), first.source_observed_at,
                min(row.ingested_commit_seq for row in compatible[key]),
                any(row.source_observed_at is not None for row in compatible[key]),
            ):
                raise QuoteMemError("reception_stale", "Previous-batch order changed its receipt")
            order = message_order(order_key(
                first, session_event_time=min(times) if times else None,
                chunk_event_time=anchor.chunk_event_time, coordinate_version=1,
                received_commit_seq=anchor.received_commit_seq,
                source_time_present=anchor.source_time_present,
            ))
            orders[key] = RequestCoordinate(
                key, order, anchor.received_commit_seq,
                message_chunk_index(request) is not None or anchor.source_time_present,
            )
    if set(orders) != set(compatible):
        raise QuoteMemError("binding_mismatch", "Previous-batch scan lost a receipt")
    partitions = request_partitions(tuple(orders.values()))

    def ordering(key: tuple[str, str]) -> tuple[object, ...]:
        coordinate = orders[key]
        if coordinate.has_source_coordinate:
            return partitions[key], 1, coordinate.source_order
        return partitions[key], 0, coordinate.received_commit_seq

    candidates = [key for key in orders if key != current and ordering(key) < ordering(current)]
    previous = max(candidates, key=ordering) if candidates else None
    if previous is not None and len(conversations[previous]) > 1:
        raise QuoteMemError("binding_mismatch", "Previous request mixes source conversations")
    messages: list[MessageRecord] = []
    for key in (current,) if previous is None else (current, previous):
        proof = load_request_inventories(writer, scope, key[0], (key[1],), at_time=view.now,
                                        input_high_watermark=view.pin.commit_seq)
        inventory = proof.inventories.get(key[1])
        if inventory is None or {part.evidence_id for part in inventory.parts} != {
            row.object_id for row in batches[key]
        }:
            raise QuoteMemError("binding_mismatch",
                                "Previous-batch receipt lost a complete request")
        for ordinal in range(inventory.message_count):
            message = view.message(MessageKey(*key, ordinal))
            expected = [part for part in inventory.parts if part.ordinal == ordinal]
            if message is None or [part.object_id for part in message.parts] != [
                part.evidence_id for part in expected
            ]:
                raise QuoteMemError("binding_mismatch",
                                    "Owner view differs from its complete receipt")
            if key == previous:
                messages.append(message)
    if previous is not None and previous[0] != current[0] and (
        conversations[previous] != conversations[current]
    ):
        raise QuoteMemError("stale_evidence", "Previous conversation requires an erasure witness")
    check_valid()
    return PreviousBatch(tuple(messages)) if previous is not None else None
