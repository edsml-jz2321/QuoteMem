"""The guarded Search use case: recall v4 ranking, V1 rendering and one-recompute delivery.

One Search (E.1.3) resolves an exact singleton grant, freezes its retrieval-only
QueryBundleV1 outside every lock, ranks the pinned snapshot with ``recall_pure``
under the released policy, renders exact V1 bytes from S7b-2's delivery
projections, and then, at the head of the application's FIFO delivery queue
and inside one outermost Writer critical section, rechecks authority and every
shown byte (``SnapshotReadGuard.check_search_delivery``), writes the ``recall``
delivery row with its ``latency_ms`` and enqueues the response. A changed hit,
end or quote line makes the attempt ``delivery_stale``: the Search is recomputed
once at a fresh pin, keeping its FIFO position, and a second stale attempt is
429 ``delivery_busy``. Failures are E.3 errors, never an apparently successful
empty answer (INV A12). No provider call runs under the Writer mutex.

An attempt that ranked but could not deliver (it was not yet the FIFO head, or the
Writer was busy) unwinds and tries again from a fresh pin. The next attempt takes the
earlier ranking's locators when an exact digest of every ranking input
(``_ranking_digest``) is unchanged, instead of ranking again; everything after the
ranking, the delivery recheck included, runs as before. A large user's concurrent
Searches each ranked twice (about 71 s per Search for one large user).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, astuple, dataclass
from typing import Literal, cast

from aml.api.bench import SearchDelivery, SearchRequest, SearchResult
from aml.api.bench_conflict_judge import ConflictJudge, conflict_noted
from aml.api.bench_conflicts import ConflictCandidate, lexical_pairs
from aml.api.bench_interval import delivered_interval
from aml.api.bench_premise import PremiseJudge, noted
from aml.api.bench_query import (
    PreparedBenchQuery,
    QueryVectorCache,
    prepare_bench_query,
    search_mode,
)
from aml.api.bench_reader import ReaderNote, reader_noted
from aml.api.bench_render import delivered_said, frozen_quote_speakers, render_projections
from aml.api.bench_rerank import PoolReranker
from aml.api.bench_temporal import temporal_note
from aml.core.add_read_view import EvidenceRecord, MessageKey, MessageRecord
from aml.core.controls import ForgetMarksV1, ForgetWithdrawnV1
from aml.core.delivery import DeliveryProjection, MemoryMode, QuoteKey, delivery_projections
from aml.core.entities import USER, QuoteSpanV1
from aml.core.errors import AMLError
from aml.core.identity import integrity_digest, new_ulid, scope_id
from aml.core.knowledge import KnowledgeReadView
from aml.core.query import FrozenScopeGrantSetV1, QueryBundleV1, ReturnedItemV1, ScopeV1
from aml.core.readability import Readability, VersionContent
from aml.core.recall import carried_parts, recall_pure
from aml.core.recall_corpora import MemoryDocument
from aml.core.recall_snapshot import RecallSnapshot, evidence_record
from aml.providers.auth import AuthProvider, AuthRequest, PrincipalGrant
from aml.providers.contracts import EmbeddingProvider
from aml.service.delivery import DeliveryAttempt, DeliveryQueue, DeliveryTicket
from aml.store.domain import ReplayableRecallRowV1
from aml.store.knowledge import read_knowledge_view
from aml.store.read_model import SearchContext, SnapshotReadGuard, StoreReadModel
from aml.store.source_labels import source_label_view
from aml.store.writer import Writer

MAX_STALE_RECOMPUTES = 1
"""E.1.3: one recompute at a fresh pin; the next stale attempt is ``delivery_busy``."""

type Profile = Literal["product", "bench"]


def _clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _check_delivery(delivery: SearchDelivery) -> None:
    if delivery.cancelled or time.monotonic() >= delivery.deadline:
        raise AMLError("delivery_busy", "Search delivery is no longer available")


def _check_guarded(delivery: SearchDelivery, guard: SnapshotReadGuard) -> None:
    _check_delivery(delivery)
    if guard.cancelled:
        raise AMLError("index_rebuilding", "Search snapshot was revoked")


def _latency_ms(delivery: SearchDelivery, now: float) -> int:
    """Whole milliseconds from the complete request bytes to this delivery."""
    return max(0, round((now - delivery.received_at) * 1000))


@dataclass
class _SearchState:
    """Request-owned continuation; it retains no guard, row or rendered hit.

    ``grant`` and ``query_time`` belong to the current computation: contention
    keeps both, while the single stale recompute resolves a fresh grant and
    time. ``prepared`` keeps only the frozen query and exact provider outcomes;
    ``premise`` keeps the premise judge's note under the digest of its exact
    input, so a recompute reuses it only when it delivers the same hit text;
    ``reader`` keeps the reader note the same way.
    """

    grant: PrincipalGrant
    query_time: int
    prepared: PreparedBenchQuery | None = None
    recomputes: int = 0
    premise: tuple[str, str | None] | None = None
    """The premise judge's input digest and note; the recompute reuses it for the same input."""
    conflict: tuple[str, ConflictCandidate | None] | None = None
    """The conflict check's input digest and confirmed pair; reused for the same input."""
    reader: tuple[str, str | None] | None = None
    """The reader note's input digest and note; the recompute reuses it for the same input."""
    ranked: tuple[str, tuple[ReturnedItemV1, ...]] | None = None
    """The ranking inputs' digest and the locators ``recall_pure`` returned for them.

    Content-free: ids, revisions, content hashes, ranks and scores.
    """


def _ranking_digest(
    root: RecallSnapshot,
    scope: ScopeV1,
    bundle: QueryBundleV1,
    allowed: FrozenScopeGrantSetV1,
    mode: MemoryMode,
    withdrawn: frozenset[str],
    *,
    reranked: bool,
) -> str:
    """The exact digest of every input of one ``recall_pure`` call but its cache.

    The pinned root is named by its generation and root identity (a pure function of
    its rows), with its scope fence, embedding binding and policy; a forget or any
    other change of the scope therefore changes the digest. The bundle is complete
    (query, options, clock, vector, policy, routes, exclusions and carried parts).
    The cache only memoizes the root, and a knowledge read is never reused, so its
    rows are no input here. Floats encode in their shortest round-trip form, so equal
    digests mean equal inputs.
    """
    value = {
        "generation": root.generation,
        "root_identity": root.root_identity,
        "fence": asdict(root.scope_fence(scope.scope_id)),
        "embedding": [root.embedding_space, root.embedding_token],
        "policy": root.recall_policy.policy_snapshot_digest,
        "allowed_scopes": allowed.model_dump(mode="json"),
        "mode": mode,
        "bundle": bundle.model_dump(mode="json"),
        "withdrawn": sorted(withdrawn),
        "rerank": reranked,
    }
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def quote_speakers(
    writer: Writer,
    root: RecallSnapshot,
    scope: ScopeV1,
    versions: Sequence[VersionContent],
    *,
    now: int,
) -> dict[QuoteKey, str | None]:
    """Resolve the shown spans' speakers from S2a's frozen labels at one pin.

    The label view is built from the durable decisions alone
    (``store.source_labels.source_label_view``): no provider is called and no
    label is classified here. Every part of each quoted message is read, so a
    label-bearing neighbour part still attributes its turn.

    Args:
        writer: The sole Writer, read only.
        root: The pinned root whose Evidence the spans name.
        scope: The exact authorized scope.
        versions: The delivered memory versions whose quotes may be shown.
        now: Trusted readability clock of the same pin.

    Returns:
        One speaker or None per quoted span (A missing key renders null).
    """
    wanted: set[MessageKey] = set()
    for version in versions:
        for span in version.quote_spans:
            row = root.get_source(span.evidence_id)
            if row is not None and (row.scope_kind, row.scope_id) == (
                scope.scope_kind, scope.scope_id,
            ):
                wanted.add(MessageKey(row.session_id, row.request_id, row.ordinal))
    if not wanted:
        return {}
    parts: dict[MessageKey, list[EvidenceRecord]] = {}
    for row in root.scope_sources(scope.scope_id):
        key = MessageKey(row.session_id, row.request_id, row.ordinal)
        if key in wanted:
            parts.setdefault(key, []).append(evidence_record(row))
    records = [
        MessageRecord(key, tuple(sorted(found, key=lambda part: part.part_no)))
        for key, found in sorted(
            parts.items(), key=lambda item: (item[0].session_id, item[0].request_id,
                                             item[0].ordinal),
        )
    ]
    by_id = {part.object_id: part for record in records for part in record.parts}
    labels = source_label_view(
        writer, Readability(scope.scope_kind, scope.scope_id, now, by_id.get), records,
    )
    return frozen_quote_speakers(
        versions, parts=[part for record in records for part in record.parts],
        labels=labels, scope=scope,
    )


def _delivered_versions(
    root: RecallSnapshot, returned: Sequence[ReturnedItemV1],
) -> tuple[VersionContent, ...]:
    """The shown memory versions; delivery_projections rechecks their identity."""
    versions: list[VersionContent] = []
    for target in returned:
        if target.kind != "memory":
            continue
        row = root.get_memory(target.object_id)
        if row is None or row.current_revision_id != target.revision_id:
            raise AMLError("delivery_stale", "A ranked memory changed before rendering")
        versions.append(MemoryDocument.of(row).snapshot)
    return tuple(versions)


class SearchUseCase:
    """Compose the registered Search authority, recall v4 ranking and guarded delivery (I13).

    Args:
        writer: The sole initialized Writer.
        publisher: That Writer's registered snapshot publisher.
        auth: The exact-scope authorization boundary, resolved outside AML locks.
        embedding: The bound embedding boundary for the frozen query vector.
        queue: The application-owned FIFO shared by every Search delivery.
        profile: The serving profile; bench Search without ``as_of`` reads plain.
        clock: Trusted wall clock in integer milliseconds.
        vector_cache: Optional service-owned query-vector cache of ``embedding``.
        premise: Optional premise judge (``bench_premise``) of plain Searches.
        rerank: Optional pool re-rank (``bench_rerank``) of plain Searches of very
            large scopes (``core/pool_rerank.py``).
        conflicts: Optional conflict check (``bench_conflict_judge``) of plain Searches.
        reader: Optional reader note (``bench_reader``) of plain Searches.

    Construction starts no worker and establishes no readiness.
    """

    def __init__(
        self,
        writer: Writer,
        publisher: StoreReadModel,
        auth: AuthProvider,
        embedding: EmbeddingProvider,
        queue: DeliveryQueue,
        *,
        profile: Profile,
        clock: Callable[[], int] = _clock_ms,
        vector_cache: QueryVectorCache | None = None,
        premise: PremiseJudge | None = None,
        rerank: PoolReranker | None = None,
        conflicts: ConflictJudge | None = None,
        reader: ReaderNote | None = None,
    ) -> None:
        if StoreReadModel.for_writer(writer) is not publisher:
            raise AMLError("binding_mismatch", "Search requires the Writer's registered publisher")
        if profile not in ("product", "bench"):
            raise AMLError("schema_invalid", "Search profile is not released")
        self._writer = writer
        self._publisher = publisher
        self._auth = auth
        self._embedding = embedding
        self._queue = queue
        self._profile: Profile = profile
        self._clock = clock
        self._vectors = vector_cache
        self._premise = premise
        self._rerank = rerank
        self._conflicts = conflicts
        self._reader = reader

    def __call__(self, request: SearchRequest, delivery: SearchDelivery) -> None:
        """Run ``search``; the controller's Search operation takes this object."""
        self.search(request, delivery)

    def search(self, request: SearchRequest, delivery: SearchDelivery) -> None:
        """Deliver one Search exactly once at the final guarded enqueue.

        Args:
            request: The admitted exact-user request; query and options serve
                retrieval only (INV A16).
            delivery: Transport deadline, cancellation and the atomic enqueue.

        Returns:
            None after the response and its ``recall`` row are committed in one
            Writer critical section.

        Raises:
            AMLError: E.3 failures: authorization, read-model, corpus, cache or
                embedding failures (429), ``delivery_busy`` after the
                single stale recompute or when the delivery budget ends.
        """
        if (
            not isinstance(cast(object, request), SearchRequest)
            or type(request.user_id) is not str or not request.user_id
        ):
            raise AMLError("invalid_request", "Search requires an exact user identity")
        scope = ScopeV1(scope_kind="user", scope_id=scope_id("user", request.user_id))
        mode = search_mode(self._profile, request.as_of)
        _check_delivery(delivery)
        state = _SearchState(self._resolve(scope), self._clock())
        _check_delivery(delivery)
        with self._queue.ticket(delivery.deadline) as ticket:
            with delivery.on_cancel(lambda: self._queue.cancel(ticket)):
                self._queue.wait_for_prefetch(ticket)
                while True:
                    _check_delivery(delivery)
                    try:
                        outcome, generation = self._attempt(
                            request, scope, mode, delivery, ticket, state,
                        )
                    except AMLError as error:
                        if error.code != "delivery_stale":
                            raise
                        # The attempt's guard and content have already unwound.
                        state.recomputes += 1
                        if state.recomputes > MAX_STALE_RECOMPUTES:
                            raise AMLError(
                                "delivery_busy", "Search state changed again before delivery",
                            ) from None
                        state.grant = self._resolve(scope)
                        state.query_time = self._clock()
                        continue
                    if outcome == "delivered":
                        return
                    if outcome == "queued":
                        self._queue.wait_for_turn(ticket)
                    elif not self._writer.mutex.wait_for_progress(
                        generation, deadline=delivery.deadline, cancellation=delivery,
                    ):
                        raise AMLError("delivery_busy", "Search delivery wait ended")

    def _marks(self, scope: ScopeV1) -> tuple[QuoteSpanV1, ...]:
        """The scope's marked sentences (``forget_marks:{s}``).

        Marks only grow (``StatementForget.chat_forget``), so the latest set is the one to
        show; another scope's marks are never read.
        """
        stored = self._writer.get_control(f"forget_marks:{scope.scope_id}")
        return () if stored is None else tuple(ForgetMarksV1.model_validate(stored["value"]).spans)

    def _withdrawn(self, scope: ScopeV1) -> frozenset[str]:
        """The scope's summary-only memories an in-chat forget withdrew (``forget_withdrawn:{s}``).

        The record only grows (``StatementForget.chat_forget``); another scope's is never
        read.
        """
        stored = self._writer.get_control(f"forget_withdrawn:{scope.scope_id}")
        return frozenset() if stored is None else frozenset(
            ForgetWithdrawnV1.model_validate(stored["value"]).memory_ids)

    def _resolve(self, scope: ScopeV1) -> PrincipalGrant:
        return self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))

    def _attempt(
        self,
        request: SearchRequest,
        scope: ScopeV1,
        mode: MemoryMode,
        delivery: SearchDelivery,
        ticket: DeliveryTicket,
        state: _SearchState,
    ) -> tuple[DeliveryAttempt, int]:
        """One guarded computation and at most one head delivery attempt."""
        context = self._publisher.begin_search(
            state.grant, query_time=state.query_time, raw_user_id=request.user_id,
            clock=self._clock,
        )
        frozen = context.allowed_scopes
        # The provider call runs here: after the content-free pin metadata,
        # before any guard, reservation or Writer critical section.
        state.prepared = prepare_bench_query(
            request, policy=context.recall_policy, mode=mode,
            embedding_space=context.embedding_space, embedding_epoch=context.embedding_token,
            query_time=context.query_time, embedding=self._embedding, previous=state.prepared,
            vector_cache=self._vectors,
            vector_isolation=None if self._vectors is None else integrity_digest({
                "principal_id": frozen.principal_id,
                "requested_scope": frozen.requested_scope.model_dump(mode="json"),
                "grants": [grant.model_dump(mode="json") for grant in frozen.grants],
            }),
        )
        prepared = state.prepared
        _check_delivery(delivery)
        # Only the ready FIFO head carries ticket priority for the Writer.
        reservation: AbstractContextManager[None] = (
            self._writer.mutex.delivery_reservation()
            if self._queue.is_head(ticket) else nullcontext()
        )
        outcome: DeliveryAttempt = "queued"
        generation = 0
        with reservation, self._publisher.pin_search(context) as guard:

            def operation(root: RecallSnapshot) -> None:
                nonlocal outcome, generation
                outcome, generation = self._compute_and_deliver(
                    request, scope, mode, delivery, ticket, guard, context, root,
                    guard.search_bundle(context, prepared.bundle), state,
                )

            try:
                guard.consume_search(context, operation)
            except AMLError as error:
                if cast(DeliveryAttempt, outcome) == "delivered":
                    # Revoked after its delivery committed: the response stands.
                    return outcome, generation
                if error.code == "index_rebuilding" and guard.cancelled:
                    # A Search reads its own scope's view, so another scope's drain
                    # never revokes it (SR-1). A revocation left (an embedding
                    # change, a failed publication, a finalize phase change of its
                    # own scope) is its one recompute; its own scope's fence answers
                    # that recompute's pin with 429 ``index_rebuilding``.
                    raise AMLError("delivery_stale", "Search snapshot was revoked") from None
                raise
        # No guard, row or rendered body leaves the attempt.
        return outcome, generation

    def _compute_and_deliver(
        self,
        request: SearchRequest,
        scope: ScopeV1,
        mode: MemoryMode,
        delivery: SearchDelivery,
        ticket: DeliveryTicket,
        guard: SnapshotReadGuard,
        context: SearchContext,
        root: RecallSnapshot,
        bundle: QueryBundleV1,
        state: _SearchState,
    ) -> tuple[DeliveryAttempt, int]:
        """Rank, render and attempt the delivery inside the registered guard callback."""
        writer = self._writer
        as_of = None if request.as_of is None else request.as_of.at
        knowledge: KnowledgeReadView | None = None
        if mode == "knowledge":
            assert as_of is not None
            knowledge = read_knowledge_view(
                writer, root, scope, recorded_as_of=as_of, query_time=context.query_time,
                check_active=lambda: _check_guarded(delivery, guard),
                baseline=guard.reads_baseline,
            )
        shown = root if knowledge is None else knowledge.root
        carried = carried_parts(
            request.query, bundle.query, root.source_corpus(scope.scope_id),
        ) if bundle.query != request.query else []
        if carried:
            # The recorded bundle keeps the parts the omitted query middle carried (A.6.3).
            bundle = QueryBundleV1.model_validate(
                {**bundle.model_dump(mode="json"), "carried": carried},
            )
        allowed = context.allowed_scopes
        withdrawn = self._withdrawn(scope)
        # A knowledge read ranks its selected projections; it is never reused.
        key = None if knowledge is not None else _ranking_digest(
            root, scope, bundle, allowed, mode, withdrawn, reranked=self._rerank is not None,
        )
        if key is not None and state.ranked is not None and state.ranked[0] == key:
            returned = tuple(item.model_copy() for item in state.ranked[1])
        else:
            returned = recall_pure(
                root, bundle, bundle.policy_snapshot, allowed_scopes=allowed,
                mode=mode, cache=self._publisher.recall_corpora(root),
                knowledge=None if knowledge is None else {
                    row.object_id: row for row in knowledge.root.memory_rows(scope.scope_id)
                },
                rerank=self._rerank,
                withdrawn=withdrawn,
            )
        state.ranked = None if key is None else (key, returned)
        _check_guarded(delivery, guard)
        versions = _delivered_versions(shown, returned)
        speakers = quote_speakers(writer, shown, scope, versions, now=context.query_time)
        projections = delivery_projections(
            shown, scope=scope, returned=returned, mode=mode, at_time=context.query_time,
            speakers=speakers, as_of=as_of,
        )
        result = render_projections(projections, scope=scope, query=request.query,
                                    marks=self._marks(scope))
        reader = None
        if self._reader is not None and mode in {"plain", "current"}:
            # Started first so that its call overlaps the conflict and premise checks.
            reader = self._reader_begin(result, request.query, root, scope, delivery, guard,
                                        state)
        if self._conflicts is not None and mode in {"plain", "current"}:
            result = self._conflict_noted(result, projections, state)
        premised = result
        if self._premise is not None and mode in {"plain", "current"} and not request.options:
            # A Search with answer choices asks to pick one, not to recall a premise.
            premised = self._premise_noted(
                result, request.query, projections, root, scope, delivery, guard, state,
                context.query_time,
            )
        if reader is not None and (note := reader()) is not None and premised is result:
            # A premise note says the record lacks what the question asks; the reader
            # note's facts would read as support for it, so a premise note replaces it.
            premised = reader_noted(premised, note)
        result = premised
        _check_guarded(delivery, guard)
        # Fresh authority for the delivery instant, resolved outside AML locks.
        fresh = self._resolve(scope)
        _check_guarded(delivery, guard)
        generation = writer.mutex.progress_generation
        stale = False

        def commit() -> bool:
            nonlocal stale
            if not writer.mutex.acquire(blocking=False):
                return False
            try:
                try:
                    self._deliver_locked(
                        guard, context, fresh, projections, mode=mode, as_of=as_of,
                        knowledge=knowledge, shown=shown, versions=versions, bundle=bundle,
                        returned=returned, result=result, delivery=delivery,
                    )
                except AMLError as error:
                    if error.code != "delivery_stale":
                        raise
                    # A stale attempt keeps this FIFO position for its recompute.
                    stale = True
                    return False
            finally:
                writer.mutex.release()
            return True

        outcome = self._queue.attempt(ticket, commit)
        if stale:
            raise AMLError("delivery_stale", "Search requires its one fresh recompute")
        return outcome, generation

    def _conflict_noted(
        self,
        result: SearchResult,
        projections: tuple[DeliveryProjection, ...],
        state: _SearchState,
    ) -> SearchResult:
        """Add a model-confirmed conflict note; the lexical rule only nominates pairs.

        The check reads only delivered user statements (``bench_conflict_judge``)
        and runs in the guard callback, outside every Writer section, like the
        premise judge; its note opens the negated hit after the fixed notice, and
        the premise note, added later, still opens the first hit. Without a
        confirmed pair, the first lexical nomination gets the note showing both
        statements as possibly conflicting (``possible_conflict_note``), never the
        conflict note.
        """
        assert self._conflicts is not None
        items = [delivered_said(projection) for projection in projections]
        candidates = self._conflicts.candidates(items)
        if not candidates:
            return result
        digest = hashlib.sha256(json.dumps(
            [astuple(candidate) for candidate in candidates], ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        if state.conflict is None or state.conflict[0] != digest:
            state.conflict = (digest, self._conflicts.choose(candidates))
        chosen = state.conflict[1]
        if chosen is not None:
            return conflict_noted(result, items, chosen)
        nominated = lexical_pairs(items)
        return conflict_noted(result, items, nominated[0], confirmed=False) if nominated else result

    def _complete(
        self,
        root: RecallSnapshot,
        scope: ScopeV1,
        delivery: SearchDelivery,
        guard: SnapshotReadGuard,
    ) -> bool:
        """Whether the scope's whole record is readable and indexed at this pin.

        A model note reads only delivered hits, and only over a complete record (no
        withheld, held, erased or unindexed part), so it never rests on or echoes
        forgotten text.
        """
        evidence = self._publisher.recall_corpora(root).evidence(
            scope.scope_id, check=lambda: _check_guarded(delivery, guard),
        )
        return len(evidence.documents) == len(root.scope_sources(scope.scope_id))

    def _reader_begin(
        self,
        result: SearchResult,
        query: str,
        root: RecallSnapshot,
        scope: ScopeV1,
        delivery: SearchDelivery,
        guard: SnapshotReadGuard,
        state: _SearchState,
    ) -> Callable[[], str | None] | None:
        """Start the reader note of a Search over a complete record; None when it gets none.

        The note reads the hits as rendered, before any model note joins them. Its call
        runs in the guard callback, outside every Writer section, beside the conflict
        and premise checks; a forget or revocation meanwhile cancels the guard, and the
        attempt's next check recomputes. A recompute that delivers the same hit text
        reuses the note. Should the attempt end before the returned wait runs, the call
        still finishes on its own read timeout and nothing waits for it.

        Returns:
            The wait for the note body (None on any failure or a late answer), or
            None when the query is not read or the record is not complete.
        """
        assert self._reader is not None
        message = self._reader.message(query, [hit.content for hit in result.hits])
        if message is None or not self._complete(root, scope, delivery, guard):
            return None
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        if state.reader is not None and state.reader[0] == digest:
            kept = state.reader[1]
            return lambda: kept
        wait = self._reader.begin(message, len(result.hits))

        def finish() -> str | None:
            note = wait()
            state.reader = (digest, note)
            return note

        return finish

    def _premise_noted(
        self,
        result: SearchResult,
        query: str,
        projections: tuple[DeliveryProjection, ...],
        root: RecallSnapshot,
        scope: ScopeV1,
        delivery: SearchDelivery,
        guard: SnapshotReadGuard,
        state: _SearchState,
        query_time: int,
    ) -> SearchResult:
        """Add the premise judge's note to a gated recall Search over a complete record.

        The judge reads only delivered hits, and only while the scope's whole
        record is readable and indexed (no withheld, held, erased or unindexed
        part), so its note never rests on or echoes forgotten text. Its call runs
        in the guard callback, outside every Writer section; a forget or revocation
        meanwhile cancels the guard, and the attempt's next check recomputes.

        A read that gets the temporal note (``bench_temporal``) or the interval
        note (``bench_interval``) is a computation over delivered dates, which
        the judge misreads: it is not judged.
        The premise note opens the first hit right after its fixed notice, so it
        precedes that hit's conflict and temporal notes.
        """
        assert self._premise is not None
        corpora = self._publisher.recall_corpora(root)
        # The user's own readable names (P-NAME at the Search clock): a question
        # that names the user is judged like a first-person one.
        user = corpora.entity(
            scope.scope_id, check=lambda: _check_guarded(delivery, guard),
        ).retrieval(
            Readability(scope.scope_kind, scope.scope_id, query_time, root.get_source),
        ).entity(USER)
        message = self._premise.message(
            query, [hit.content for hit in result.hits],
            () if user is None else tuple(name.name for name in user.names),
        )
        if message is None or (
            temporal_note(query, projections) is not None
            or delivered_interval(query, projections) is not None
        ):
            return result
        if not self._complete(root, scope, delivery, guard):
            return result
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        if state.premise is None or state.premise[0] != digest:
            state.premise = (digest, self._premise.note(message))
        note = state.premise[1]
        return result if note is None else noted(result, note)

    def _deliver_locked(
        self,
        guard: SnapshotReadGuard,
        context: SearchContext,
        fresh: PrincipalGrant,
        projections: tuple[DeliveryProjection, ...],
        *,
        mode: MemoryMode,
        as_of: int | None,
        knowledge: KnowledgeReadView | None,
        shown: RecallSnapshot,
        versions: Sequence[VersionContent],
        bundle: QueryBundleV1,
        returned: tuple[ReturnedItemV1, ...],
        result: SearchResult,
        delivery: SearchDelivery,
    ) -> None:
        """Recheck, persist the delivery row and enqueue in one Writer critical section."""
        writer = self._writer
        now = self._clock()
        scope = context.allowed_scopes.requested_scope
        # Labels are re-derived at the delivery pin; check_search_delivery then
        # proves the fences unchanged, so ``shown`` still holds these parts.
        speakers: Mapping[QuoteKey, str | None] = quote_speakers(
            writer, shown, scope, versions, now=now,
        )
        guard.check_search_delivery(
            context, fresh, projections, mode=mode, at_time=now, speakers=speakers,
            as_of=as_of, knowledge=knowledge,
        )
        _check_delivery(delivery)
        frozen = context.allowed_scopes
        bench = self._profile == "bench"
        writer.record_search_delivery(ReplayableRecallRowV1(
            object_id=new_ulid(), tenant_id=writer.tenant_id,
            scope_kind=scope.scope_kind, scope_id=scope.scope_id,
            environment="benchmark" if bench else "production",
            principal_id=frozen.principal_id, security_level="restricted",
            created_at=now, latency_ms=_latency_ms(delivery, time.monotonic()),
            record_class="replayable_delivery",
            trigger="benchmark_adapter" if bench else "manual",
            query_digest=writer.bind_content("recall-query", {
                "query": bundle.query, "options": bundle.options,
                "requested_scope": scope.model_dump(mode="json"),
                "as_of": None if bundle.as_of is None else bundle.as_of.model_dump(mode="json"),
                "top_k": bundle.top_k,
            }),
            query_bundle=bundle, allowed_scopes=frozen,
            snapshot_generation=context.metadata.generation,
            returned=list(returned), delivered_kind="rendered_evidence",
        ))
        delivery.deliver(result)
