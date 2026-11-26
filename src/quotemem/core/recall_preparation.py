"""Frozen ranking-input preparation over the corpora, with Search eligibility.

No storage, providers or delivery writes live here. A caller holds the read pin
and passes the exact frozen query, grants, and optional same-root cache.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue

from quotemem.core.delivery import AutomaticRead, KnowledgeSelection, MemoryMode, memory_eligible
from quotemem.core.entities import USER
from quotemem.core.errors import QuoteMemError
from quotemem.core.query import AsOfV1, FrozenScopeGrantSetV1, ObjectLocatorV1
from quotemem.core.query_routes import EVIDENCE_ROUTES, VECTOR_ROUTES, select_query_routes
from quotemem.core.readability import Readability
from quotemem.core.recall_corpora import (
    CorpusVectors,
    EntityRetrieval,
    EvidenceDocument,
    MemoryCorpus,
    MemoryDocument,
    SnapshotRecallCache,
    SparseStatistics,
    keyword_forms,
    sparse_terms,
)
from quotemem.core.recall_policy import RecallPolicySnapshotV4, Route
from quotemem.core.recall_snapshot import MemoryRowV2, RecallSnapshot
from quotemem.core.statement import named_participants, token_seq
from quotemem.core.vector_score import UnitVector, unit_vector
from quotemem.ingestion.temporal import calendar_windows

FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself"})
"""Folded tokens by which a query speaks about its own user in the first person."""


class RankingQuery(Protocol):
    """Read-only retrieval fields of the frozen QueryBundleV1.

    Policy is an explicit argument to ranking, not read implicitly from
    a currently installed release. Persistence and provider metadata remain
    the bundle owner's responsibility.
    """

    @property
    def query(self) -> str: ...
    @property
    def options(self) -> dict[str, JsonValue] | list[str]: ...
    @property
    def query_time(self) -> int: ...
    @property
    def as_of(self) -> AsOfV1 | None: ...
    @property
    def exclude(self) -> Sequence[ObjectLocatorV1]: ...
    @property
    def session_lru(self) -> Sequence[ObjectLocatorV1]: ...
    @property
    def active_routes(self) -> Sequence[str]: ...
    @property
    def embedding_space(self) -> str: ...
    @property
    def embedding_epoch(self) -> str: ...
    @property
    def query_vector(self) -> Sequence[float] | None: ...
    @property
    def top_k(self) -> int: ...
    @property
    def carried(self) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class PreparedRecall:
    """Same-root eligible documents and privacy-filtered ranking populations.

    ``dated_memories`` and ``dated_evidence`` are the eligible documents that the
    query's explicit calendar dates name (explicit-date vote): an Evidence
    part observed inside a named window, and a memory whose own validity starts
    inside one or that quotes a same-scope part observed inside one. Both are
    empty when the query names no calendar day or month.

    ``memory_typicality`` and ``evidence_typicality`` are each channel corpus's
    `centroid_typicality` by object id, which the dense routes subtract. Both are
    empty unless the query names a participant: a speaker's name, which most of
    the participants' own texts carry, then pulls every text near the store's
    centroid toward the query. Without such a name the centroid is the store's
    own topic, which a query about it must keep.

    ``others_memories`` and ``others_evidence`` are another person's content for a
    choice read the user asks about themself: one in a user scope that carries
    answer choices, speaks in the first person (`FIRST_PERSON`) and names no
    entity or participant but the user. Such a read asks which choice fits the
    user, and an answer treats whatever it is shown as the user's. Its choices
    also join the query vector (``OPTIONS_VECTOR_WEIGHT`` in
    `quotemem.api.bench_query`), and a choice written from another person's preference
    pulls that person's content up. Others' memories are anchored first to another entity.
    Others' Evidence is a part that only such memories quote (a text the user
    shared about someone else, such as a colleague's email), and the assistant
    part that answers such a user part unless the user's own memories quote it.
    Both are empty for any other query, so a query naming that person reads them
    as before.

    ``choice_read`` says the query carries at least one answer choice (an MCQ read,
    whose answer picks among them). Such a read delivers each event once
    (`quotemem.core.recall.recall_pure`).

    ``participant_read`` holds when the query itself, its choices aside, names a
    labelled participant of a conversation other than the user, and carries no
    answer choices: a third-person question about someone in a multi-party
    conversation, such as "which books has Blair read". It also delivers each
    event once. A first-person read and a store without labelled speakers never
    qualify; a read that is neither ranks exactly as before.
    """

    memories: tuple[MemoryDocument, ...]
    statistics_memories: tuple[MemoryDocument, ...]
    evidence: tuple[EvidenceDocument, ...]
    memory_statistics: SparseStatistics
    evidence_statistics: SparseStatistics
    terms: frozenset[str]
    trigger_terms: frozenset[str]
    entity_memories: frozenset[str]
    vector: UnitVector | None
    routes: tuple[Route, ...]
    scope_kind: str
    scope_id: str
    memory_typicality: Mapping[str, float]
    evidence_typicality: Mapping[str, float]
    dated_memories: frozenset[str] = frozenset()
    dated_evidence: frozenset[str] = frozenset()
    others_memories: frozenset[str] = frozenset()
    others_evidence: frozenset[str] = frozenset()
    choice_read: bool = False
    participant_read: bool = False
    rollups: tuple[MemoryDocument, ...] = ()
    """The scope's current session rollups (``core/session_rollup.py``) for a plain or
    current read; they rank in no channel, and only a summary or ordering read delivers
    them (``core/recall.py``)."""
    memory_vectors: CorpusVectors | None = None
    """The memory corpus's stacked unit vectors, read by the dense routes."""
    evidence_vectors: CorpusVectors | None = None
    """The Evidence corpus's stacked unit vectors, read by the dense routes."""
    remainders: tuple[EvidenceDocument, ...] = ()
    """The parts a complete forget withheld bytes of that keep a current remainder, as the
    Evidence corpus holds them (``EvidenceCorpus.remainders``), under ``evidence``'s own
    filters; only the interactive Search of an ``agent`` workspace ranks them
    (``core/recall.py``)."""


def prepare_recall(
    root: RecallSnapshot,
    query: RankingQuery,
    policy: RecallPolicySnapshotV4,
    *,
    allowed_scopes: FrozenScopeGrantSetV1,
    mode: MemoryMode,
    cache: SnapshotRecallCache | None = None,
    knowledge: Mapping[str, MemoryRowV2] | None = None,
    lexical_only: bool = False,
) -> PreparedRecall:
    """Prepare exactly one authorized scope under immutable eligibility.

    Args:
        root: The caller's read-only snapshot.
        query: Frozen retrieval fields of QueryBundleV1.
        policy: Explicit normalized v4 policy, including measurement off values.
        allowed_scopes: Exact singleton grant with matching deletion epochs.
        mode: Already resolved by Search; plain_history=none restricts plain to current.
        cache: Optional RecallSnapshot cache of this exact root; failures propagate.
        knowledge: Knowledge reader's newest-at-t projection per retained memory.
            Required only for knowledge; the reader proves the revision selection.
        lexical_only: The query freezes the policy's routes without the vector routes
            (the interactive Search's keyword fallback); otherwise it must freeze them all.

    Returns:
        The pure ranking population and its frozen statistics.

    Raises:
        QuoteMemError: On authorization, binding, mode or cache failures. Never
            converts a failed read into an empty population.
    """
    requested = allowed_scopes.requested_scope
    if len(allowed_scopes.grants) != 1:
        raise QuoteMemError("scope_denied", "Search requires one exact scope grant")
    grant = allowed_scopes.grants[0]
    if (
        grant.relation != "exact"
        or (grant.scope_kind, grant.scope_id)
        != (
            requested.scope_kind,
            requested.scope_id,
        )
        or not allowed_scopes.issued_at <= query.query_time < allowed_scopes.expires_at
    ):
        raise QuoteMemError("scope_denied", "Search requires a live exact scope grant")
    fence = root.scope_fence(grant.scope_id)
    if fence.quarantined:
        raise QuoteMemError("service_unhealthy", "Search scope is quarantined")
    if (fence.scope_kind, fence.erase_epoch, fence.consent_epoch) != (
        grant.scope_kind,
        grant.erase_epoch,
        grant.consent_epoch,
    ):
        raise QuoteMemError("scope_denied", "Search grant differs from the current scope fence")
    if (root.embedding_space, root.embedding_token) != (
        query.embedding_space,
        query.embedding_epoch,
    ):
        raise QuoteMemError("binding_mismatch", "Query embedding binding differs from the snapshot")
    if not 1 <= query.top_k <= policy.parameters.max_top_k:
        raise QuoteMemError("invalid_request", "top_k exceeds the explicit policy limit")
    released = select_query_routes(policy, mode)
    routes: tuple[Route, ...] = tuple(
        route for route in released if route not in VECTOR_ROUTES
    ) if lexical_only else released
    if set(query.active_routes) != set(routes) or len(query.active_routes) != len(routes):
        raise QuoteMemError("binding_mismatch",
                            "Frozen routes differ from the explicit policy and mode")
    vector = None
    if VECTOR_ROUTES & set(routes):
        if query.query_vector is None:
            raise QuoteMemError("embedding_unavailable",
                                "An active vector route needs a query vector")
        vector = unit_vector(query.query_vector)
    if cache is None:
        cache = SnapshotRecallCache(root)
    elif cache.root is not root:
        raise QuoteMemError("binding_mismatch", "Recall cache belongs to another snapshot")
    corpus = cache.memory(grant.scope_id)
    read = Readability(grant.scope_kind, grant.scope_id, query.query_time, root.get_source)
    at = None if query.as_of is None else query.as_of.at
    if mode in {"world", "knowledge"}:
        if query.as_of is None or query.as_of.mode != mode:
            raise QuoteMemError("binding_mismatch", "As-of mode differs from the frozen query")
    elif query.as_of is not None:
        raise QuoteMemError("binding_mismatch", "Current reads cannot carry an as-of point")
    if mode == "knowledge" and knowledge is None:
        raise QuoteMemError("binding_mismatch", "Knowledge reads require selected revisions")
    effective_mode = (
        "current" if (mode == "plain" and policy.parameters.plain_history == "none") else mode
    )
    memories = _eligible(corpus, read, effective_mode, at, knowledge)
    statistics_memories = corpus.readable_documents
    entity_corpus = cache.entity(grant.scope_id)
    if mode in {"world", "knowledge"}:
        names = EntityRetrieval.of_versions(
            (document.snapshot for document in memories),
            root.entity_rows(grant.scope_id),
            include_user=grant.scope_kind == "user",
        )
    else:
        names = entity_corpus.retrieval(read)
    texts = [query.query]
    choices = query.options if isinstance(query.options, list) else query.options.get("choices", [])
    if isinstance(choices, list):
        texts.extend(choice for choice in choices if isinstance(choice, str))
    expansions = [
        names.expansion(text, max_entities=policy.parameters.entity_expansion_max_entities)
        for text in texts
    ]
    terms = frozenset(
        term
        for text in (*texts, *(name.name for found in expansions for name in found.names))
        for term in sparse_terms(text)
    )
    matched = {entity for found in expansions for entity in found.matched}
    # A participant has no entity name: the query names it by its label or by
    # a speaker name of its readable Evidence, the catalog's rule (route 4).
    participants = entity_corpus.participant_entities
    named: set[str] = set()
    # Who the query itself names (its choices aside): an entity by name, a participant
    # by a label or speaker name.
    asked = set(expansions[0].matched)
    asked_participants: set[str] = set()
    if participants:
        speakers = entity_corpus.speakers(read, texts)
        for index, text in enumerate(texts):
            found = {
                participants[key]
                for key, _ in named_participants(grant.scope_id, text, participants, speakers)
            }
            named.update(found)
            if not index:
                asked.update(found)
                asked_participants.update(found)
        matched.update(named)
    entity_memories = frozenset(
        document.memory_id for document in memories if matched & set(document.snapshot.anchors)
    )
    # Whose content a choice read asks for: only the user's own, when it speaks in
    # the first person and names no one else (`PreparedRecall.others_memories`).
    has_choices = isinstance(choices, list) and any(isinstance(choice, str) for choice in choices)
    about_self = (
        grant.scope_kind == "user"
        and has_choices
        and not FIRST_PERSON.isdisjoint(token_seq(query.query))
        and asked <= {USER}
    )
    evidence: tuple[EvidenceDocument, ...] = ()
    remainders: tuple[EvidenceDocument, ...] = ()
    evidence_statistics = SparseStatistics.of(())
    evidence_typicality: Mapping[str, float] = {}
    evidence_vectors: CorpusVectors | None = None
    if EVIDENCE_ROUTES & set(routes):
        sources = cache.evidence(grant.scope_id)
        evidence_vectors = sources.vectors
        held = root.held_evidence(grant.scope_id)

        def shown(document: EvidenceDocument) -> bool:
            return mode not in {"auto", "core_hook"} or (
                document.row.security_level != "restricted"
                and document.row.trust_tier not in {"inferred", "untrusted"})

        evidence = tuple(
            document
            for document in sources.documents
            if read.p_evr(document.row, held=document.row.object_id in held)
            and shown(document)
        )
        remainders = tuple(document for document in sources.remainders
                           if document.row.object_id not in held and read.p_src(document.row)
                           and shown(document))
        evidence_statistics = sources.statistics
        if named and "evidence_dense" in routes:
            evidence_typicality = sources.typicality
    windows = calendar_windows(query.query)
    others_evidence: frozenset[str] = frozenset()
    if about_self and evidence:
        quoted: dict[str, set[bool]] = {}
        for document in memories:
            if anchor := document.snapshot.entity_id:
                for span in document.snapshot.quote_spans:
                    quoted.setdefault(span.evidence_id, set()).add(anchor == USER)
        others_evidence = frozenset(
            document.row.object_id for document in evidence
            if quoted.get(document.row.object_id) == {False}
        )
        answered = {
            (row.session_id, row.request_id, row.ordinal + 1)
            for document in evidence
            if (row := document.row).object_id in others_evidence and row.role == "user"
        }
        others_evidence = others_evidence | frozenset(
            document.row.object_id for document in evidence
            if document.row.role == "assistant"
            and (document.row.session_id, document.row.request_id, document.row.ordinal)
            in answered
            and True not in quoted.get(document.row.object_id, set())
        )

    def dated(at: int | None) -> bool:
        return at is not None and any(start <= at < end for start, end in windows)

    def quoted_on(document: MemoryDocument) -> bool:
        for span in document.snapshot.quote_spans:
            source = root.get_source(span.evidence_id)
            if (
                source is not None
                and (source.scope_kind, source.scope_id) == (grant.scope_kind, grant.scope_id)
                and dated(source.source_observed_at)
            ):
                return True
        return False

    return PreparedRecall(
        memories,
        statistics_memories,
        evidence,
        corpus.search_statistics,
        evidence_statistics,
        terms,
        frozenset(form for term in terms for form in keyword_forms(term)),
        entity_memories,
        vector,
        routes,
        grant.scope_kind,
        grant.scope_id,
        corpus.typicality if named and "dense" in routes else {},
        evidence_typicality,
        frozenset(
            document.memory_id for document in memories
            if dated(document.snapshot.valid_from) or quoted_on(document)
        ) if windows else frozenset(),
        frozenset(
            document.row.object_id for document in evidence
            if dated(document.row.source_observed_at)
        ) if windows else frozenset(),
        frozenset(
            document.memory_id for document in memories
            if document.snapshot.entity_id not in {"", USER}
        ) if about_self else frozenset(),
        others_evidence,
        has_choices,
        bool(asked_participants - {USER}) and not has_choices,
        _eligible(corpus, read, "current", None, None, corpus.rollups)
        if effective_mode in {"plain", "current"} else (),
        corpus.vectors,
        evidence_vectors,
        remainders,
    )


def listed_memories(
    root: RecallSnapshot, cache: SnapshotRecallCache, *, scope_kind: str, scope_id: str,
    now: int,
) -> tuple[tuple[MemoryDocument, ...], tuple[EvidenceDocument, ...], frozenset[str]]:
    """A plain read's eligible memories and readable Evidence of one scope, without a query.

    The same eligibility as `prepare_recall` in ``plain`` mode (current memories and past
    states with a witnessed end), and also a memory an undated update replaced
    (`Readability.p_replaced`), which has no end to date; the same Evidence readability
    (``p_evr``), for a listing of the scope's memories. The caller authorizes the scope.

    Args:
        root: The read pin's snapshot.
        cache: A recall cache of exactly ``root``.
        scope_kind: The exact scope's kind.
        scope_id: The exact scope's id.
        now: The read's trusted clock.

    Returns:
        The memory documents, the Evidence documents and the ids of the memories only an
        undated update's replacement admits, which no Search reads.

    Raises:
        QuoteMemError: ``binding_mismatch`` for a cache of another root.
    """
    if cache.root is not root:
        raise QuoteMemError("binding_mismatch", "Recall cache belongs to another snapshot")
    read = Readability(scope_kind, scope_id, now, root.get_source)
    corpus = cache.memory(scope_id)
    memories = _eligible(corpus, read, "plain", None, None, replaced=True)
    plain = {document.memory_id for document in _eligible(corpus, read, "plain", None, None)}
    held = root.held_evidence(scope_id)
    evidence = tuple(
        document for document in cache.evidence(scope_id).documents
        if read.p_evr(document.row, held=document.row.object_id in held)
    )
    return memories, evidence, frozenset(
        document.memory_id for document in memories if document.memory_id not in plain)


def _eligible(
    corpus: MemoryCorpus,
    read: Readability,
    mode: MemoryMode,
    at: int | None,
    knowledge: Mapping[str, MemoryRowV2] | None,
    documents: Sequence[MemoryDocument] | None = None,
    *,
    replaced: bool = False,
) -> tuple[MemoryDocument, ...]:
    result: list[MemoryDocument] = []
    for current in corpus.documents if documents is None else documents:
        document = current
        selection = None
        if mode == "knowledge":
            if knowledge is None or (row := knowledge.get(current.memory_id)) is None:
                continue
            if (
                (row.object_id, row.scope_kind, row.scope_id)
                != (
                    current.memory_id,
                    corpus.scope_kind,
                    corpus.scope_id,
                )
                or at is None
                or row.recorded_at > at
            ):
                raise QuoteMemError(
                    "binding_mismatch", "Knowledge projection differs from its selection"
                )
            document = MemoryDocument.of(row)
            selection = KnowledgeSelection(document.snapshot, row.lifecycle_state)
        snapshot = document.snapshot
        if memory_eligible(
            current.world,
            mode=mode,
            read=read,
            index=corpus,
            at=at,
            automatic=AutomaticRead(
                current.row.activation_mode,
                snapshot.security_level,
                snapshot.trust_tier,
                tuple(snapshot.sensitivity_classes),
            ),
            knowledge=selection,
            replaced=replaced,
        ):
            result.append(document)
    return tuple(result)
