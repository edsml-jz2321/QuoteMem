"""Pure recall v4 ranking, independent nomination and exact quote folding (A.6.3).

Scores use Decimal fusion and the shared C.4.2 float64 cosine. The bench profile
normalizes Memory and Evidence scores per channel, then applies a shared pool,
threshold and MMR before top_k. Selected memory quote lines alone can fold
Evidence; product current reads remain memory-only.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

from aml.core.delivery import MemoryMode, QuoteKey, QuoteSource, format_date, rendered_quotes
from aml.core.entities import MemorySnapshotV2, Origin
from aml.core.errors import AMLError
from aml.core.pool_rerank import (
    RERANK_CANDIDATES,
    RERANK_MIN_PARTS,
    Rerank,
    RerankCandidate,
)
from aml.core.query import (
    MAX_CARRIED_PARTS,
    FrozenScopeGrantSetV1,
    ReturnedEvidenceV1,
    ReturnedItemV1,
    ReturnedMemoryV1,
)
from aml.core.query_routes import CONTENT_ROUTES, EVIDENCE_ROUTES, MEMORY_ROUTES
from aml.core.recall_corpora import (
    EvidenceDocument,
    MemoryDocument,
    SnapshotRecallCache,
    SparseDocument,
    SparseStatistics,
    sparse_terms,
)
from aml.core.recall_policy import NormalizedRecallPolicyV4, RecallPolicySnapshotV4, Route
from aml.core.recall_preparation import PreparedRecall, RankingQuery, prepare_recall
from aml.core.recall_snapshot import MemoryRowV2, RecallSnapshot, SourceRowV2
from aml.core.recall_spread import (
    aggregate_query,
    one_per_event,
    order_query,
    said_by_user_first,
    spread_sessions,
)
from aml.core.semantic import TrustTier
from aml.core.session_rollup import (
    MAX_ROLLUPS_FIRST,
    ROLLUP_MIN_PARTS,
    by_session_start,
    newest_per_session,
)
from aml.core.statement import token_seq
from aml.core.time_order import RECALL_POLICY_VERSION, TimeOrderV1, read_time_order
from aml.core.vector_score import UnitRows, UnitVector, cosine

LEXICAL_PIPELINE_VERSION = "memory-statement-keywords-categories-nfkc-casefold-v2"
EXPLICIT_DATE_WEIGHT = Decimal("0.6")
"""RRF weight of the explicit-date vote (A.6.3): a query that writes a calendar day
or month ranks the documents that date names by one more vote, so a question
about "9 November, 2022" reaches what was said or happened that day."""
TYPICALITY_WEIGHT = Decimal("1.0")
"""Share of a document's `centroid_typicality` that the dense routes subtract (A.6.3).

A dense cosine overrates what is typical of the whole store when the query names a
participant: greetings, thanks and statements that say little beyond the speaker's
name lie near every query that names that speaker. Subtracting it once moved more
supporting turns into the top hits than the weights 0.5, 1.5 and 2.0 did. It is gated
on queries that name a participant (`prepare_recall`): ungated, it hurt queries that
name none, such as contradiction-resolution and summary questions."""
OTHERS_FACTOR = Decimal("0.5")
"""Factor on the score of another person's content in a choice read about the user (A.6.3).

`prepare_recall` marks that content (`PreparedRecall.others_memories` and
``others_evidence``) only for a read that carries answer choices. It stays
retrievable, below the user's own facts. `_channel` applies the factor to the
fused score before the quality cut, and `_context_candidates` to a marked
neighbor that an unmarked direct match nominates, so every marked candidate
carries it once. A factor of 0.5 places most marked memories below the 100th
hit, which helped choice questions about the user's own preferences and
sensitive information more than 0.75 did. A read without choices ranks exactly
as before."""
CHOICE_READ_ONCE = True
"""A plain read that carries answer choices delivers each event once (A.6.3).

As for an aggregating read (`aml.core.recall_spread.one_per_event`), a memory whose
quoted parts are all delivered as Evidence is passed over and the freed slots go to
the next candidates; unlike it, the relevance order is kept (no session spread).
This helps choice questions about forget requests and sensitive information. Reads
without choices rank exactly as before: delivering every read once hurt
contradiction-resolution and abstention questions."""
OPTION_RESERVED_SLOTS = 2
"""Hits a plain read with answer choices reserves for each choice's own best matches (A.6.3).

The query vector and the lexical route read the question and all choices at once, so a
choice about a detail the question does not name (another speaker's action, a separate
event) can rank below the 100th hit. For each choice, its label removed, the read scores
every eligible memory and Evidence part by BM25 of the choice's own terms (each channel
normalized by its best score) and makes sure the top matches are delivered: one already
selected covers its slot, and the rest replace the lowest-ranked selections. Only a
multi-select read (`MULTI_ANSWER`) reserves. Two slots worked better than one or three
in internal replays. A read without choices or with a single answer ranks exactly as
before."""
_CHOICE_LABEL = re.compile(r"\A\s*\(?[A-Fa-f][).:]\s+")
MULTI_ANSWER = re.compile(
    r"\b(?:all (?:the )?correct (?:options|answers|choices)"
    r"|all (?:options|answers|choices) that apply|all that apply|select all|choose all"
    r"|every correct (?:option|answer|choice))\b",
    re.IGNORECASE,
)
"""A read whose question asks for every correct choice (multi-select): only it reserves
slots (`OPTION_RESERVED_SLOTS`), since each choice is judged on its own there. On single-
answer reads the reservation displaced the user's own memories."""
NEAR_SAME_JACCARD = 0.7
"""Alphabetic-token Jaccard from which two delivered memories of one anchor and kind
state nearly the same thing, so a plain read shows the newer one first (A.6.3).
"""
ECHO_MIN_CHARS = 200
"""Shortest whitespace-normalized source part that Search treats as already said by a
query containing it verbatim (A.6.3). Conversational questions are far shorter (at
most a few hundred characters in public conversational benchmarks), so only a query
that itself carries a document can contain such a part. When the question holds a
whole reference document, such hits would only repeat the question."""
MAX_SIBLING_PARTS = 16
"""Most other parts of one long assistant message that a read asking what the
assistant said nominates beside a direct match (`_context_candidates`)."""
MESSAGE_COMPLETION_PARTS = 12
"""Most undelivered parts of partly delivered long messages one read adds (A.6.3)."""
MESSAGE_COMPLETION_SCOPE = 200
"""Largest readable scope, in Evidence parts, whose partly delivered messages are completed.

A long message is stored in parts of at most 1,000 bytes, and a table or list split that
way matches a question in only a few of its parts, so the relevance cut can drop rows
of a small table while delivering the rest of their message. Conversational scopes are
usually larger than this bound, so their reads are unchanged."""
RECALLS_ASSISTANT = re.compile(
    r"\byou (?:\w+ )?(?:provided|gave|recommended|suggested|mentioned|told|listed|shared|said"
    r"|wrote|described|explained|came up with|offered)\b"
    r"|\b(?:our|the|that) (?:previous|earlier|last) (?:conversation|chat|discussion)\b",
    re.IGNORECASE,
)
"""Wording that asks back what the assistant said earlier: what "you" provided or said,
or "our previous conversation". A bare "remind me" also asks for the user's own facts
(on BEAM-100K it delivered whole assistant answers to an instruction-following read)."""
SUMMARY_QUERY = re.compile(
    r"\b(?:summar(?:y|ies|ise[ds]?|ize[ds]?|ising|izing)|recap|overview)\b", re.IGNORECASE,
)
"""Wording that asks for a summary of the conversations: summarize, summary, recap or
overview, whole words, no topic vocabulary.

Only such a read may deliver memories of origin ``extraction_supplement`` (an overview or the
assistant's work, ``core/extraction_supplement.py``); every other read ranks as if they were not
stored. Each recounts a whole session, so it lies near every query about that session: delivered
to every read they took the top hits of instruction-following, knowledge-update and
multi-session reads. The supplement's implied health conditions (origin ``implied_health``) are
ordinary memories for every read."""
RERANK_LIFT = Decimal("0.1")
"""Margin by which a candidate the pool re-rank chose (`core/pool_rerank.py`) outscores
every other pool candidate. It exceeds the largest MMR similarity penalty,
``1 - mmr_lambda`` (0.05), so no unchosen candidate is selected before a chosen one."""
SUMMARY_ONLY_ORIGINS: frozenset[Origin] = frozenset({"extraction_supplement", "session_rollup"})
"""Origins whose memories only a summary read delivers (`summary_read`; a session rollup also
an ordering read, `_rollups_first`); an in-chat forget withdraws one that states a forgotten
fact (``forget_withdrawn:{s}``), and no read delivers it again."""
_ZERO = Decimal(0)
_ONE = Decimal(1)
_SIX = Decimal("0.000001")
type _Key = tuple[str, str]


def _number(value: int | float | str) -> Decimal:
    return Decimal(repr(value) if isinstance(value, float) else str(value))


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(_SIX, rounding=ROUND_HALF_EVEN)


def _clip(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    return min(high, max(low, value))


@dataclass(frozen=True, slots=True)
class _Document:
    value: MemoryDocument | EvidenceDocument

    @property
    def key(self) -> _Key:
        return (
            "memory" if isinstance(self.value, MemoryDocument) else "evidence",
            self.value.row.object_id,
        )

    @property
    def importance(self) -> Decimal:
        return (
            _number(self.value.snapshot.importance)
            if isinstance(self.value, MemoryDocument)
            else Decimal("0.5")
        )

    @property
    def trust(self) -> TrustTier:
        return (
            self.value.snapshot.trust_tier
            if isinstance(self.value, MemoryDocument)
            else self.value.row.trust_tier
        )

    @property
    def vector(self) -> UnitVector:
        vector = self.value.vector
        if vector is None:
            raise AMLError("binding_mismatch", "Eligible memory has no vector")
        return vector

    @property
    def tie(self) -> tuple[Decimal, int, int, str, str]:
        return -self.importance, 0, 0, *self.key


@dataclass(frozen=True, slots=True)
class _Scored:
    document: _Document
    final: Decimal


def _lexical_scorer(
    statistics: SparseStatistics, terms: frozenset[str],
) -> Callable[[SparseDocument], Decimal]:
    """BM25 of one query's terms over documents of one corpus (C.4.2).

    Each term's inverse document frequency depends only on the corpus, so it is
    computed once per query, at the first document that holds the term, by the
    same expression in the same decimal context: every score equals the
    per-document computation it replaces.
    """
    if not statistics.count or not statistics.length:
        return lambda _document: _ZERO
    count = Decimal(statistics.count)
    average = Decimal(statistics.length) / count
    ordered = sorted(terms)
    inverses: dict[str, Decimal] = {}

    def score(document: SparseDocument) -> Decimal:
        total = _ZERO
        for term in ordered:
            frequency = document.counts.get(term, 0)
            if not frequency:
                continue
            inverse = inverses.get(term)
            if inverse is None:
                df = Decimal(statistics.frequencies.get(term, 0))
                inverse = inverses[term] = (
                    _ONE + (count - df + Decimal("0.5")) / (df + Decimal("0.5"))
                ).ln()
            denominator = frequency + Decimal("1.2") * (
                Decimal("0.25") + Decimal("0.75") * document.length / average
            )
            total += inverse * frequency * Decimal("2.2") / denominator
        return _quantize(total)

    return score


def _route_scores(prepared: PreparedRecall) -> dict[Route, dict[_Key, Decimal]]:
    documents = {
        "memory": tuple(_Document(document) for document in prepared.memories),
        "evidence": tuple(_Document(document) for document in prepared.evidence),
    }
    scores: dict[Route, dict[_Key, Decimal]] = {}
    for route in prepared.routes:
        values: dict[_Key, Decimal] = {}
        channel = "evidence" if route in EVIDENCE_ROUTES else "memory"
        if route in {"dense", "evidence_dense", "entity"}:
            if prepared.vector is None:
                raise AMLError("embedding_unavailable", "A vector route has no frozen query vector")
            scored = [document for document in documents[channel]
                      if route != "entity" or document.key[1] in prepared.entity_memories]
            vectors = prepared.memory_vectors if channel == "memory" else prepared.evidence_vectors
            cosines = None if vectors is None else vectors.cosines(
                [document.value for document in scored], prepared.vector,
            )
            if cosines is None:
                cosines = UnitRows([document.vector for document in scored]).cosines(
                    prepared.vector,
                )
            typical: Mapping[str, float] = (
                {} if route == "entity"
                else prepared.memory_typicality if channel == "memory"
                else prepared.evidence_typicality
            )
            values = {
                document.key: _quantize(
                    _number(value)
                    - TYPICALITY_WEIGHT * _number(typical.get(document.key[1], 0.0)),
                ) if typical else _number(value)
                for document, value in zip(scored, cosines, strict=True)
            }
        elif route in {"lexical", "evidence_lexical"}:
            statistics = (
                prepared.memory_statistics if channel == "memory" else prepared.evidence_statistics
            )
            lexical = _lexical_scorer(statistics, prepared.terms)
            values = {
                document.key: lexical(document.value.sparse) for document in documents[channel]
            }
        elif route == "trigger":
            population = prepared.statistics_memories
            frequencies: Counter[str] = Counter()
            for document in population:
                frequencies.update(document.keyword_forms & prepared.trigger_terms)
            inverses = {
                term: (
                    _ONE
                    + (len(population) - frequency + Decimal("0.5")) / (frequency + Decimal("0.5"))
                ).ln()
                for term, frequency in frequencies.items()
            }
            values = {
                ("memory", document.memory_id): _quantize(
                    sum(
                        (
                            inverses[term]
                            for term in sorted(document.keyword_forms & prepared.trigger_terms)
                            if term in inverses
                        ),
                        _ZERO,
                    )
                )
                for document in prepared.memories
                if document.keyword_forms & prepared.trigger_terms
            }
        scores[route] = values
    return scores


def _boost(document: _Document, policy: NormalizedRecallPolicyV4) -> Decimal:
    importance = _clip(
        _number(policy.importance.base)
        + _number(policy.importance.coefficient) * document.importance,
        _number(policy.importance.min),
        _number(policy.importance.max),
    )
    return min(
        _number(policy.boost_cap),
        importance
        * _number(
            getattr(policy.trust_weights, document.trust),
        ),
    )


def _channel(
    documents: tuple[_Document, ...],
    routes: frozenset[Route],
    scores: dict[Route, dict[_Key, Decimal]],
    policy: NormalizedRecallPolicyV4,
    query: RankingQuery,
    mode: MemoryMode,
    excluded: set[_Key],
    dated: frozenset[_Key] = frozenset(),
    others: frozenset[_Key] = frozenset(),
) -> list[_Scored]:
    """Fuse one channel's route votes into thresholded, boosted candidates.

    ``others`` holds another person's content for a choice read the user asks
    about themself; its final scores take `OTHERS_FACTOR` before the quality cut.

    ``dated`` holds the documents the query's explicit calendar dates name. They
    cast one more RRF vote, weighted `EXPLICIT_DATE_WEIGHT`: each channel's
    dated documents are ranked by their own dense cosine (by the tie order when
    the policy has no dense route) and capped like any route quota. The vote is
    added on top of the content vote, so the channel's normalizing denominator
    is unchanged and an undated query ranks exactly as before.
    """
    by_key = {document.key: document for document in documents}
    ranks: dict[Route, dict[_Key, int]] = {}
    nominated: set[_Key] = set()
    raw: dict[_Key, Decimal] = {}
    denominators = {
        kind: sum((
            _number(getattr(policy.route_weights, route)) / (policy.rrf_k + 1)
            for route in scores
            if route in content
        ), _ZERO)
        for kind, content in (("memory", CONTENT_ROUTES), ("evidence", EVIDENCE_ROUTES))
    }
    for route in sorted(routes & scores.keys()):
        values = scores[route]
        weight = _number(getattr(policy.route_weights, route))
        if not weight:
            continue
        if route in {"trigger", "entity"}:
            admitted = list(values)
        else:
            floor = _number(getattr(policy, f"{route}_floor"))
            admitted = [
                key
                for key, score in values.items()
                if (score >= floor if route in {"dense", "evidence_dense"} else score > floor)
            ]
        admitted.sort(key=lambda key: (-values[key], *by_key[key].tie))
        quota = admitted[: policy.route_quota_multiplier * query.top_k]
        if route in MEMORY_ROUTES:
            nominated.update(quota)
            if route in CONTENT_ROUTES:
                ranks[route] = {key: rank for rank, key in enumerate(admitted, 1)}
        else:
            for rank, key in enumerate(quota, 1):
                raw[key] = raw.get(key, _ZERO) + weight / (policy.rrf_k + rank)
    for key in sorted(nominated):
        raw[key] = sum(
            (
                _number(getattr(policy.route_weights, route)) / (policy.rrf_k + route_ranks[key])
                for route, route_ranks in ranks.items()
                if key in route_ranks
            ),
            _ZERO,
        )
    if EXPLICIT_DATE_WEIGHT and dated:
        dense: tuple[tuple[str, Route], ...] = (("memory", "dense"), ("evidence", "evidence_dense"))
        for kind, route in dense:
            cosines: dict[_Key, Decimal] = scores.get(route, {})
            members = sorted(
                (key for key in dated if key[0] == kind and key in by_key),
                key=lambda key: (-cosines.get(key, _ZERO), *by_key[key].tie),
            )
            for rank, key in enumerate(members[: policy.route_quota_multiplier * query.top_k], 1):
                raw[key] = raw.get(key, _ZERO) + EXPLICIT_DATE_WEIGHT / (policy.rrf_k + rank)
    fused = {
        key: _quantize(value / denominators[key[0]])
        for key, value in raw.items()
        if value > 0 and denominators[key[0]] > 0
    }
    pool = sorted(fused, key=lambda key: (-fused[key], *by_key[key].tie))[
        : policy.candidate_pool_limit
    ]
    if not pool or (
        mode in {"auto", "core_hook"} and (fused[pool[0]] < _number(policy.inject_threshold))
    ):
        return []
    boosts = {key: _boost(by_key[key], policy) for key in pool}
    if CONTENT_ROUTES <= ranks.keys():
        # Same-trust strict agreement caps importance. The pool is bounded at
        # 200; a dominating row's original boost yields the same lower envelope
        # as B's prefix-minimum sweep, including transitive domination.
        original = boosts.copy()
        for key in pool:
            if key in excluded or any(key not in ranks[route] for route in CONTENT_ROUTES):
                continue
            for other in pool:
                if (
                    other not in excluded
                    and by_key[key].trust == by_key[other].trust
                    and all(
                        other in ranks[route] and scores[route][other] > scores[route][key]
                        for route in CONTENT_ROUTES
                    )
                ):
                    boosts[key] = min(boosts[key], original[other])
    final = {
        key: _quantize(fused[key] * boosts[key] * (OTHERS_FACTOR if key in others else _ONE))
        for key in pool
    }
    threshold = max(
        _number(policy.absolute_threshold),
        _number(policy.relative_threshold) * max(final.values()),
    )
    return [
        _Scored(by_key[key], final[key])
        for key in pool
        if final[key] >= threshold and key not in excluded
    ]


def summary_read(query: str) -> bool:
    """Return whether a read asks for a summary, so it may deliver supplement memories.

    Args:
        query: The frozen query text.

    Returns:
        bool: True when the query carries summary wording (`SUMMARY_QUERY`).
    """
    return SUMMARY_QUERY.search(query) is not None


def _ordinary(member: _Scored) -> bool:
    """Whether a candidate is not a memory of a summary-only origin."""
    value = member.document.value
    return not (isinstance(value, MemoryDocument)
                and value.snapshot.origin in SUMMARY_ONLY_ORIGINS)


def _select(
    remaining: list[_Scored],
    count: int,
    weight: Decimal,
    previous: list[UnitVector],
    fold: Callable[[_Scored], frozenset[str]] | None = None,
) -> list[_Scored]:
    selected: list[_Scored] = []
    similarities: dict[_Key, Decimal] = {}
    initial = previous
    # Every candidate's cosine with one chosen vector comes from one product over
    # all candidates (``UnitRows``, equal to ``cosine`` value for value), kept
    # per vector: MMR compares each remaining candidate with each choice.
    candidates: UnitRows | None = None
    members = list(remaining)
    positions = {id(member): index for index, member in enumerate(members)}
    against: dict[int, tuple[UnitVector, list[float]]] = {}

    def similarity(member: _Scored, vector: UnitVector) -> float:
        nonlocal candidates
        if candidates is None:
            candidates = UnitRows([candidate.document.vector for candidate in members])
        kept = against.get(id(vector))
        if kept is None or kept[0] is not vector:
            kept = against[id(vector)] = (vector, candidates.cosines(vector))
        return kept[1][positions[id(member)]]

    while remaining and len(selected) < count:
        if weight != _ONE:
            for member in remaining:
                if previous:
                    value = max(
                        _number(similarity(member, vector)) for vector in previous
                    )
                    key = member.document.key
                    similarities[key] = max(similarities.get(key, value), value)

        def order(member: _Scored) -> tuple[Decimal, Decimal, int, int, str, str]:
            score = (
                member.final
                if weight == _ONE
                else _quantize(
                    weight * member.final
                    - (_ONE - weight) * similarities.get(member.document.key, _ZERO),
                )
            )
            return -score, *member.document.tie

        chosen = min(remaining, key=order)
        remaining.remove(chosen)
        selected.append(chosen)
        previous = [chosen.document.vector]
        if fold is not None:
            folded = fold(chosen)
            if folded:
                remaining[:] = [member for member in remaining
                                if member.document.key[0] != "evidence"
                                or member.document.key[1] not in folded]
                kept = [member for member in selected if member.document.key[0] != "evidence"
                        or member.document.key[1] not in folded]
                if len(kept) != len(selected):
                    # A removed Evidence must no longer penalize later MMR choices.
                    similarities.clear()
                    previous = [*initial, *(member.document.vector for member in kept)]
                selected = kept
    return selected


def _context_candidates(
    ranked: list[_Scored],
    evidence: tuple[_Document, ...],
    policy: NormalizedRecallPolicyV4,
    excluded: set[_Key],
    whole_messages: bool = False,
    others: frozenset[_Key] = frozenset(),
) -> list[_Scored]:
    """Nominate neighboring turns of eight direct Evidence matches at the same pin.

    Only a readable first part at the exact preceding/following ordinal of the
    same request and session is eligible. This never bridges an unreadable gap,
    another Add or another session. Inherited context receives 0.95 of its
    anchor's score so an unmatched neighbor cannot win the direct match's tie.
    Its own trust boost can lower that score further before the content quality
    cut. Ordinary selected-memory quote folding and final delivery checks apply.

    With ``whole_messages`` (a read that asks what the assistant said,
    `recalls_assistant`) the other parts of a direct match's assistant message
    are nominated the same way, so a long answer is delivered whole: one item
    of a long list rarely matches the question that names it by position.

    A neighbor in ``others`` (`_channel`) nominated by a match outside it takes
    `OTHERS_FACTOR`; one nominated by a marked match inherits the factor already.
    """
    if not ranked:
        return ranked
    threshold = max(
        _number(policy.absolute_threshold),
        _number(policy.relative_threshold) * max(member.final for member in ranked),
    )
    first_parts = {
        (row.session_id, row.request_id, row.ordinal): document
        for document in evidence
        if isinstance(document.value, EvidenceDocument)
        if (row := document.value.row).part_no == 0
        and document.key not in excluded
    }
    parts: dict[tuple[str, str, int], list[_Document]] = {}
    if whole_messages:
        for document in evidence:
            if isinstance(document.value, EvidenceDocument) and document.key not in excluded:
                row = document.value.row
                parts.setdefault((row.session_id, row.request_id, row.ordinal), []).append(document)
        for members in parts.values():
            members.sort(key=_part_no)
    candidates = {member.document.key: member for member in ranked}
    seeds = sorted(
        (member for member in ranked if isinstance(member.document.value, EvidenceDocument)),
        key=lambda member: (-member.final, *member.document.tie),
    )[:8]
    for seed in seeds:
        assert isinstance(seed.document.value, EvidenceDocument)
        source = seed.document.value.row
        anchor_boost = _boost(seed.document, policy)
        if not anchor_boost:
            continue
        siblings = [
            document
            for document in parts.get((source.session_id, source.request_id, source.ordinal), ())
            if document.key != seed.document.key
        ] if source.role == "assistant" else []
        neighbors = [first_parts.get((source.session_id, source.request_id, ordinal))
                     for ordinal in (source.ordinal - 1, source.ordinal + 1)]
        for neighbor in (*neighbors, *siblings[:MAX_SIBLING_PARTS]):
            if neighbor is None:
                continue
            score = _quantize(seed.final * Decimal("0.95") * min(
                _ONE, _boost(neighbor, policy) / anchor_boost,
            ) * (OTHERS_FACTOR if neighbor.key in others
                 and seed.document.key not in others else _ONE))
            if score < threshold:
                continue
            current = candidates.get(neighbor.key)
            if current is None or score > current.final:
                candidates[neighbor.key] = _Scored(neighbor, score)
    return sorted(candidates.values(), key=lambda member: (-member.final, *member.document.tie))[
        :policy.candidate_pool_limit
    ]


def _complete_messages(
    selected: list[_Scored],
    evidence: tuple[_Document, ...],
    skipped: set[_Key],
    scored: Mapping[_Key, Decimal],
) -> list[_Scored]:
    """Deliver the other parts of partly delivered long messages in place of the last hits.

    Messages are taken in the order of their first delivered part. Their readable parts
    that are not delivered, not ``skipped`` (excluded, echoed or folded into a delivered
    memory) are added, at most `MESSAGE_COMPLETION_PARTS` in all, and each completed
    message reads as one run in part order at the place of its first delivered part. The
    same number of the lowest-ranked hits outside completed messages make room, so the
    number of hits never changes.
    """

    def message(document: _Document) -> tuple[str, str, int] | None:
        value = document.value
        if not isinstance(value, EvidenceDocument):
            return None
        return value.row.session_id, value.row.request_id, value.row.ordinal

    parts: dict[tuple[str, str, int], list[_Document]] = {}
    for document in evidence:
        if (key := message(document)) is not None:
            parts.setdefault(key, []).append(document)
    delivered = {member.document.key for member in selected}
    added: dict[tuple[str, str, int], list[_Document]] = {}
    count = 0
    for member in selected:
        key = message(member.document)
        if key is None or key in added or len(parts.get(key, ())) < 2:
            continue
        missing = [document for document in sorted(parts[key], key=_part_no)
                   if document.key not in delivered and document.key not in skipped]
        missing = missing[:MESSAGE_COMPLETION_PARTS - count]
        if missing:
            added[key] = missing
            count += len(missing)
        if count == MESSAGE_COMPLETION_PARTS:
            break
    removable = [index for index in reversed(range(len(selected)))
                 if message(selected[index].document) not in added]
    while count > len(removable):
        last = next(reversed(added))
        count -= 1
        added[last].pop()
        if not added[last]:
            del added[last]
            removable = [index for index in reversed(range(len(selected)))
                         if message(selected[index].document) not in added]
    if not count:
        return selected
    dropped = set(removable[:count])
    result: list[_Scored] = []
    emitted: set[tuple[str, str, int]] = set()
    for index, member in enumerate(selected):
        key = message(member.document)
        if index in dropped or key in emitted:
            continue
        if key is None or key not in added:
            result.append(member)
            continue
        emitted.add(key)
        run = [other for other in selected if message(other.document) == key] + [
            _Scored(document, scored.get(document.key, _ZERO)) for document in added[key]
        ]
        result.extend(sorted(run, key=lambda part: _part_no(part.document)))
    return result


def _rerank_candidate(document: _Document) -> RerankCandidate:
    value = document.value
    if isinstance(value, EvidenceDocument):
        row = value.row
        observed = row.source_observed_at
        return RerankCandidate(
            row.text, row.role or "user",
            None if observed is None else format_date(observed, row.time_granularity),
        )
    snapshot = value.snapshot
    start = snapshot.valid_from
    return RerankCandidate(
        snapshot.statement, "memory",
        None if start is None else format_date(start, snapshot.time_granularity),
    )


def _reranked(pool: list[_Scored], rerank: Rerank, query: str) -> list[_Scored]:
    """Move the pool candidates the re-rank chose ahead of the rest (`core/pool_rerank.py`).

    The model reads the leading `RERANK_CANDIDATES` in fused order. Each chosen
    candidate scores `RERANK_LIFT` above the pool's best, plus one millionth per
    place in the model's order; the rest keep their fused scores. No choice, or a
    failed re-rank, leaves the pool as fused.
    """
    head = pool[:RERANK_CANDIDATES]
    order = rerank(query, [_rerank_candidate(member.document) for member in head]) if head else None
    if not order:
        return pool
    lift = max(member.final for member in pool) + RERANK_LIFT
    count = len(order)
    chosen = {
        id(head[position]): _Scored(
            head[position].document, _quantize(lift + (count - rank) * _SIX),
        )
        for rank, position in enumerate(order)
    }
    return [chosen.get(id(member), member) for member in pool]


def _part_no(document: _Document) -> int:
    value = document.value
    return value.row.part_no if isinstance(value, EvidenceDocument) else 0


def folded_evidence(
    version: MemorySnapshotV2,
    *,
    sources: Mapping[str, QuoteSource],
    speakers: Mapping[QuoteKey, str | None],
    scope_kind: str,
    scope_id: str,
    now: int,
) -> frozenset[str]:
    """Return parts covered whole by one actually rendered quote (AC12b).

    Call only for memories selected for this response. A citation, a union of
    spans, and a fifth or later quote never establish coverage. Offsets are
    original-message coordinates, with UTF-8 edge-whitespace trimming only.
    """
    quotes = rendered_quotes(
        version,
        sources=sources,
        speakers=speakers,
        scope_kind=scope_kind,
        scope_id=scope_id,
        now=now,
    )
    folded: set[str] = set()
    for line in quotes.lines:
        source = sources[line.evidence_id]
        trimmed = source.text.strip()
        if not trimmed:
            continue
        start = source.byte_start + len(
            source.text[: len(source.text) - len(source.text.lstrip())].encode("utf-8")
        )
        end = start + len(trimmed.encode("utf-8"))
        if line.byte_start <= start and line.byte_end >= end:
            folded.add(line.evidence_id)
    return frozenset(folded)


def _newer_first(selected: list[_Scored], sources: Mapping[str, SourceRowV2]) -> list[_Scored]:
    """Move each delivered memory directly above an older one that states nearly the same.

    Two memories state nearly the same thing when they share their anchor and
    kind and their statements' alphabetic tokens reach `NEAR_SAME_JACCARD`. The
    newer is the one with the later (source date, ingest order) of its quoted
    parts. Nothing is removed, and every other item keeps its relative order,
    so an update says what it changed before the stale value is read.
    """

    def fact(document: MemoryDocument) -> tuple[str, str, frozenset[str], tuple[int, int]]:
        quoted = [row for span in document.snapshot.quote_spans
                  if (row := sources.get(span.evidence_id)) is not None]
        recency = (
            max((row.source_observed_at for row in quoted
                 if row.source_observed_at is not None), default=-1),
            max((row.ingested_commit_seq for row in quoted), default=document.row.commit_seq),
        )
        words = frozenset(
            token for token in token_seq(document.snapshot.statement)
            if any(character.isalpha() for character in token)
        )
        return document.snapshot.entity_id, document.snapshot.kind, words, recency

    facts = {
        index: fact(member.document.value) for index, member in enumerate(selected)
        if isinstance(member.document.value, MemoryDocument)
        and member.document.value.snapshot.entity_id
    }

    def near(left: int, right: int) -> bool:
        first, second = facts[left], facts[right]
        union = first[2] | second[2]
        return first[:2] == second[:2] and bool(union) and (
            len(first[2] & second[2]) / len(union) >= NEAR_SAME_JACCARD
        )

    order = list(range(len(selected)))
    for index in sorted(facts):
        position = order.index(index)
        older = next((
            place for place, other in enumerate(order[:position])
            if other in facts and facts[other][3] < facts[index][3] and near(other, index)
        ), None)
        if older is not None:
            order.insert(older, order.pop(position))
    return [selected[index] for index in order]


def _per_event(
    selected: list[_Scored], count: int, sources: Mapping[str, SourceRowV2], *, spread: bool,
    refill: bool,
) -> list[_Scored]:
    """Deliver each event of a read once; spread an aggregating read's first slots over sessions.

    See `aml.core.recall_spread`. A choice read (`CHOICE_READ_ONCE`) or a participant
    read (``spread`` False) keeps its relevance order; a participant read (``refill``)
    gives the slots no other candidate takes to the memories it passed over.
    """

    def part_of(member: _Scored) -> str | None:
        return member.document.key[1] if member.document.key[0] == "evidence" else None

    def quoted_by(member: _Scored) -> frozenset[str]:
        value = member.document.value
        return frozenset(span.evidence_id for span in value.snapshot.quote_spans) if (
            isinstance(value, MemoryDocument)
        ) else frozenset()

    def sessions_of(member: _Scored) -> frozenset[str]:
        value = member.document.value
        if isinstance(value, EvidenceDocument):
            return frozenset({value.row.session_id})
        return frozenset(row.session_id for span in value.snapshot.quote_spans
                         if (row := sources.get(span.evidence_id)) is not None)

    kept = one_per_event(selected, count, part_of=part_of, quoted_by=quoted_by, refill=refill)
    return spread_sessions(kept, sessions_of=sessions_of) if spread else kept


def _user_said_first(
    selected: list[_Scored], sources: Mapping[str, SourceRowV2],
) -> list[_Scored]:
    """Defer an order read's candidates said only by the assistant.

    Evidence is assistant-only when its part has the assistant role, a memory
    when every part it quotes has. See `aml.core.recall_spread.said_by_user_first`.
    """

    def assistant_only(member: _Scored) -> bool:
        value = member.document.value
        rows = [value.row] if isinstance(value, EvidenceDocument) else [
            row for span in value.snapshot.quote_spans
            if (row := sources.get(span.evidence_id)) is not None
        ]
        return bool(rows) and all(row.role == "assistant" for row in rows)

    return said_by_user_first(selected, assistant_only=assistant_only)


def _rollups_first(
    selected: list[_Scored], rollups: Iterable[MemoryDocument], count: int,
    sources: Mapping[str, SourceRowV2], excluded: Collection[_Key], withdrawn: Collection[str],
    vector: UnitVector | None,
) -> list[_Scored]:
    """Open a summary or ordering read with the sessions' rollups, in date order.

    A session's rollup is its newest (`aml.core.session_rollup.newest_per_session`):
    the one whose quoted parts were ingested last, then the one committed last. A session
    whose newest rollup an in-chat forget withdrew has none: an older one may state the
    forgotten fact too. With more than `MAX_ROLLUPS_FIRST` sessions, those whose rollups
    lie nearest the query vector are kept (the latest without one). Sessions follow their
    first source time, then their first ingest order. The ranked hits follow, cut to
    ``count`` in all.
    """

    def rows(document: MemoryDocument) -> list[SourceRowV2]:
        return [row for span in document.snapshot.quote_spans
                if (row := sources.get(span.evidence_id)) is not None]

    newest = newest_per_session(
        (document for document in rollups
         if ("memory", document.memory_id) not in excluded),
        session_of=lambda document: next((row.session_id for row in rows(document)), None),
        recency=lambda document: (
            max((row.ingested_commit_seq for row in rows(document)), default=-1),
            document.row.commit_seq,
        ),
    )
    newest = {session: document for session, document in newest.items()
              if document.memory_id not in withdrawn}
    if not newest:
        return selected
    starts: dict[str, tuple[int, int]] = {}
    for row in sources.values():
        if row.session_id in newest:
            key = (2**63 if row.source_observed_at is None else row.source_observed_at,
                   row.ingested_commit_seq)
            starts[row.session_id] = min(starts.get(row.session_id, key), key)
    ordered = by_session_start(newest, starts)
    if len(ordered) > MAX_ROLLUPS_FIRST:
        def nearness(session: str) -> tuple[float, int]:
            document = newest[session]
            near = (cosine(vector, document.vector)
                    if vector is not None and document.vector is not None else 0.0)
            return near, ordered.index(session)

        kept = set(sorted(ordered, key=nearness, reverse=True)[:MAX_ROLLUPS_FIRST])
        ordered = [session for session in ordered if session in kept]
    score = selected[0].final if selected else _ONE
    first = [_Scored(_Document(newest[session]), score) for session in ordered]
    return [*first, *selected][:count]


def recalls_assistant(query: str) -> bool:
    """Return whether a read asks back what the assistant said (`RECALLS_ASSISTANT`).

    Args:
        query: The frozen query text.

    Returns:
        bool: True when the query carries such wording.
    """
    return RECALLS_ASSISTANT.search(query) is not None


def carried_parts(query: str, bounded: str, sources: Iterable[SourceRowV2]) -> list[str]:
    """Return the parts a bounded query's whole text carries that its bounded text lacks.

    A Search query too long for its bundle keeps a head and tail around
    `QUERY_OMISSION`; a document pasted into it then echoes (`_echoes`) only in
    those ends. A part counts as carried when its whitespace-normalized text has
    at least `ECHO_MIN_CHARS` characters and occurs verbatim in the whole
    normalized query but not in the bounded one. At most `MAX_CARRIED_PARTS`
    ids are returned, in id order; a query that kept its exact text has none.
    """
    if bounded == query:
        return []
    whole, kept = " ".join(query.split()), " ".join(bounded.split())
    return sorted(
        row.object_id for row in sources
        if len(text := " ".join(row.text.split())) >= ECHO_MIN_CHARS
        and text in whole and text not in kept
    )[:MAX_CARRIED_PARTS]


def _echoes(
    query: str,
    documents: tuple[_Document, ...],
    sources: Mapping[str, SourceRowV2],
    carried: Collection[str] = (),
) -> frozenset[_Key]:
    """Return the candidates whose stored text the query itself already carries.

    A source part echoes when its whitespace-normalized text has at least
    `ECHO_MIN_CHARS` characters and occurs verbatim in the whitespace-normalized
    query. An Evidence candidate echoes when its part does; a memory echoes when
    it quotes at least one part and every part it quotes echoes, since it then
    restates only what the reader already has. A shorter query echoes nothing.
    The ``carried`` parts of a bounded query (`carried_parts`) echo as well.
    """
    said = " ".join(query.split())
    if len(said) < ECHO_MIN_CHARS:
        return frozenset()
    echoed = {
        object_id for object_id, row in sources.items()
        if len(text := " ".join(row.text.split())) >= ECHO_MIN_CHARS and text in said
    } | set(carried)
    if not echoed:
        return frozenset()
    keys: set[_Key] = set()
    for document in documents:
        value = document.value
        if isinstance(value, EvidenceDocument):
            if value.row.object_id in echoed:
                keys.add(document.key)
        elif (spans := value.snapshot.quote_spans) and all(
            span.evidence_id in echoed for span in spans
        ):
            keys.add(document.key)
    return frozenset(keys)


def _time_key(document: _Document, order: TimeOrderV1) -> int | None:
    occurred = (
        document.value.snapshot.valid_from
        if isinstance(document.value, MemoryDocument)
        else document.value.row.source_observed_at
    )
    if occurred is None:
        return None
    return -occurred if order.kind == "latest" else abs(occurred - order.at)


def _time_order(selected: list[_Scored], order: TimeOrderV1 | None) -> list[_Scored]:
    if order is None:
        return selected
    dated = iter(
        sorted(
            (member for member in selected if _time_key(member.document, order) is not None),
            key=lambda member: _time_key(member.document, order) or 0,
        )
    )
    return [
        next(dated) if _time_key(member.document, order) is not None else member
        for member in selected
    ]


def _option_reserved(
    selected: list[_Scored],
    choices: Iterable[str],
    documents: Iterable[_Document],
    prepared: PreparedRecall,
    eligible: Callable[[_Document], bool],
    count: int,
) -> list[_Scored]:
    """Make sure each answer choice's own best matches are delivered (`OPTION_RESERVED_SLOTS`).

    Args:
        selected: The read's selection, best first.
        choices: The answer choices the query carries.
        documents: Every readable memory and Evidence document of the read.
        prepared: The prepared read (per-channel sparse statistics).
        eligible: Whether a document may be delivered by this read.
        count: The read's hit budget.

    Returns:
        The kept selections, then the reserved documents, then the selections they
        displaced (refill order for the caller's per-event pass); unchanged when every
        choice's top matches are already selected.
    """
    pool = [document for document in documents if eligible(document)]
    statistics = {"memory": prepared.memory_statistics, "evidence": prepared.evidence_statistics}
    taken = {member.document.key for member in selected}
    floor = min((member.final for member in selected), default=_ZERO)
    reserved: list[_Scored] = []
    for choice in choices:
        terms = frozenset(sparse_terms(_CHOICE_LABEL.sub("", choice, count=1)))
        if not terms:
            continue
        scorers = {kind: _lexical_scorer(stats, terms) for kind, stats in statistics.items()}
        scored = [(scorers[document.key[0]](document.value.sparse), document) for document in pool]
        best = {kind: max((value for value, document in scored if document.key[0] == kind),
                          default=_ZERO) for kind in statistics}
        ranked = sorted(
            ((_quantize(value / best[document.key[0]]), document)
             for value, document in scored if value > 0),
            key=lambda pair: (-pair[0], *pair[1].tie),
        )
        for _, document in ranked[:OPTION_RESERVED_SLOTS]:
            if document.key not in taken:
                taken.add(document.key)
                reserved.append(_Scored(document, floor))
    if not reserved:
        return selected
    reserved = reserved[:count]
    keep = max(0, min(len(selected), count - len(reserved)))
    return selected[:keep] + reserved + selected[keep:]


def recall_pure(
    root: RecallSnapshot,
    query: RankingQuery,
    policy: RecallPolicySnapshotV4,
    *,
    allowed_scopes: FrozenScopeGrantSetV1,
    mode: MemoryMode,
    cache: SnapshotRecallCache | None = None,
    speakers: Mapping[QuoteKey, str | None] | None = None,
    knowledge: Mapping[str, MemoryRowV2] | None = None,
    rerank: Rerank | None = None,
    withdrawn: Collection[str] = (),
) -> tuple[ReturnedItemV1, ...]:
    """Rank a frozen query on a read-only snapshot with an explicit v4 policy.

    Fixed snapshot rows, query fields (including clock and vector), policy and
    auxiliary inputs (including rerank outcomes) produce the same ordered hits,
    independent of row iteration,
    hash seed or corpus-cache warmth. Ranking ties use importance, kind and stored
    object id. Independently rebuilt stores can mint different ids; query text
    alone also leaves the embedding, readability clock and optional rerank unfrozen.

    Candidates the query already carries verbatim (`_echoes`) are never selected,
    so a query that holds a whole document is not answered with copies of it, and
    memories of a summary-only origin reach only a summary read (`summary_read`), never
    one an in-chat forget withdrew (``withdrawn``). Session rollups rank in no channel; a
    plain or current summary or ordering read (`order_query`) of a scope above
    `ROLLUP_MIN_PARTS` readable parts opens with at most `MAX_ROLLUPS_FIRST` sessions'
    rollups, in date order, leaving out a withdrawn one (`_rollups_first`); every read of a
    smaller scope ranks as if no rollup were stored.
    A plain summary read (`summary_read`) of a scope with more than `RERANK_MIN_PARTS`
    readable Evidence parts moves the pool candidates ``rerank`` chooses ahead of the
    rest (`_reranked`); every other read ranks exactly as fused.

    Args:
        root: The read pin's snapshot. No storage fallback is performed.
        query: Frozen QueryBundleV1 retrieval fields.
        policy: Complete v4 snapshot; Search supplies its released policy and
            measurement tooling supplies a registered configuration.
        allowed_scopes: Frozen exact singleton authorization.
        mode: Search's explicit read mode, never inferred from query text.
        cache: Optional same-root bounded corpus cache.
        speakers: Per-span frozen label resolutions used by the renderer.
        knowledge: Selected newest-at-t projections supplied by the knowledge reader.
        rerank: The Search's pool re-rank (``aml.api.bench_rerank``), or None.
        withdrawn: The scope's summary-only memories an in-chat forget withdrew
            (``forget_withdrawn:{s}``); empty for a scope that never asked.

    Returns:
        Bound locators in authoritative delivery order, with six-decimal scores.
        Re-rendering and delivery rechecks, recall.latency_ms and the single
        stale recompute belong to the Search use case.

    Raises:
        AMLError: A failed authorization, corpus, embedding or quote read;
            failures propagate and never masquerade as successful empty Search.
    """
    prepared = prepare_recall(
        root,
        query,
        policy,
        allowed_scopes=allowed_scopes,
        mode=mode,
        cache=cache,
        knowledge=knowledge,
    )
    parameters = policy.parameters
    excluded = {(entry.kind, entry.object_id) for entry in (*query.exclude, *query.session_lru)}
    memories = tuple(_Document(document) for document in prepared.memories)
    evidence = tuple(_Document(document) for document in prepared.evidence)
    with localcontext() as context:
        context.prec = 40
        scores = _route_scores(prepared)
        dated: frozenset[_Key] = frozenset(
            {("memory", memory_id) for memory_id in prepared.dated_memories}
            | {("evidence", evidence_id) for evidence_id in prepared.dated_evidence},
        )
        others = frozenset(
            {("memory", memory_id) for memory_id in prepared.others_memories}
            | {("evidence", evidence_id) for evidence_id in prepared.others_evidence},
        )
        sources = {row.object_id: row for row in root.source_corpus(prepared.scope_id)}

        def fold(member: _Scored) -> frozenset[str]:
            document = member.document.value
            return folded_evidence(
                document.snapshot, sources=sources,
                speakers={} if speakers is None else speakers,
                scope_kind=prepared.scope_kind, scope_id=prepared.scope_id, now=query.query_time,
            ) if (parameters.evidence_folding == "exact_whole_part"
                  and isinstance(document, MemoryDocument)) else frozenset()

        order = read_time_order(query.options, RECALL_POLICY_VERSION)
        echo = _echoes(query.query, (*memories, *evidence), sources, query.carried)
        summary = summary_read(query.query)

        def kept(member: _Scored) -> bool:
            # Never an echo of the query; a supplement memory only for a summary read,
            # and never one a forget withdrew.
            return member.document.key not in echo and (_ordinary(member) or (
                summary and member.document.key[1] not in withdrawn))
        if parameters.evidence_placement in {"fused_interleave", "context_interleave"}:
            candidates = _channel(
                (*memories, *evidence), MEMORY_ROUTES | EVIDENCE_ROUTES,
                scores, parameters, query, mode, excluded, dated, others,
            )
            if parameters.evidence_placement == "context_interleave" and mode == "plain":
                candidates = _context_candidates(
                    candidates, evidence, parameters, excluded, recalls_assistant(query.query),
                    others,
                )
            spread = mode == "plain" and aggregate_query(query.query)
            # A choice read (`CHOICE_READ_ONCE`) or a read about a named participant shows
            # each event once, in relevance order (no session spread).
            once = mode == "plain" and not spread and (
                (CHOICE_READ_ONCE and prepared.choice_read) or prepared.participant_read)
            ordered = spread and order_query(query.query) and not recalls_assistant(query.query)
            pool = [member for member in candidates if kept(member)]
            if (rerank is not None and mode == "plain" and summary
                    and len(evidence) > RERANK_MIN_PARTS):
                pool = _reranked(pool, rerank, query.query)
            selected = _select(
                pool,
                2 * query.top_k if spread or once else query.top_k,
                _number(parameters.mmr_lambda),
                [document.vector for document in (*memories, *evidence)
                 if document.key in excluded], fold,
            )
            if ordered:
                selected = _user_said_first(selected, sources)
            if spread or once:
                selected = _per_event(
                    selected, query.top_k, sources, spread=spread,
                    refill=not spread and prepared.participant_read,
                )
            if (mode == "plain" and prepared.choice_read and OPTION_RESERVED_SLOTS
                    and MULTI_ANSWER.search(query.query)):
                choices = (query.options if isinstance(query.options, list)
                           else query.options.get("choices", []))
                covered = _option_reserved(
                    selected, [choice for choice in choices if isinstance(choice, str)]
                    if isinstance(choices, list) else [],
                    (*memories, *evidence), prepared,
                    lambda document: (document.key not in excluded
                                      and document.key not in echo
                                      and document.key not in others
                                      and (_ordinary(_Scored(document, _ZERO))
                                           or (summary and document.key[1] not in withdrawn))),
                    query.top_k,
                )
                if covered is not selected:
                    # A reserved memory whose parts are already delivered is passed over
                    # and the displaced selections refill its slot (`CHOICE_READ_ONCE`).
                    selected = _per_event(covered, query.top_k, sources, spread=False,
                                          refill=False)
            if mode == "plain":
                selected = _newer_first(selected, sources)
            selected = _time_order(selected, order)
            if mode == "plain" and len(evidence) <= MESSAGE_COMPLETION_SCOPE:
                selected = _complete_messages(
                    selected, evidence,
                    excluded | echo | {("evidence", source) for member in selected
                                       for source in fold(member)},
                    {member.document.key: member.final for member in candidates},
                )
        else:
            selected = _select(
                [member for member in _channel(
                    memories, MEMORY_ROUTES, scores, parameters, query, mode, excluded, dated,
                    others,
                ) if kept(member)],
                query.top_k, _number(parameters.mmr_lambda),
                [document.vector for document in memories if document.key in excluded],
            )
            folded = {source for member in selected for source in fold(member)}
            evidence_pool = _channel(
                evidence, EVIDENCE_ROUTES, scores, parameters, query, mode, excluded, dated,
                others,
            )
            secondary = _select(
                [member for member in evidence_pool
                 if member.document.key[1] not in folded and member.document.key not in echo],
                query.top_k - len(selected), _number(parameters.mmr_lambda),
                [document.vector for document in evidence if document.key in excluded],
            )
            selected = [*_time_order(selected, order), *_time_order(secondary, order)]
        if (mode in {"plain", "current"} and len(evidence) > ROLLUP_MIN_PARTS
                and (summary or order_query(query.query))):
            selected = _rollups_first(selected, prepared.rollups, query.top_k, sources, excluded,
                                      withdrawn, prepared.vector)
        returned: list[ReturnedItemV1] = []
        for rank, member in enumerate(selected, 1):
            value = member.document.value
            if isinstance(value, MemoryDocument):
                returned.append(
                    ReturnedMemoryV1(
                        kind="memory",
                        object_id=value.row.object_id,
                        revision_id=value.row.current_revision_id,
                        content_hash=value.row.content_hash,
                        rank=rank,
                        score=float(member.final),
                    )
                )
            else:
                returned.append(
                    ReturnedEvidenceV1(
                        kind="evidence",
                        object_id=value.row.object_id,
                        content_hash=value.row.content_hash,
                        rank=rank,
                        score=float(member.final),
                    )
                )
        return tuple(returned)
