"""The guarded Search use case: recall v4 ranking, V1 rendering and one-recompute delivery.

One Search resolves an exact singleton grant, freezes its retrieval-only
QueryBundleV1 outside every lock, ranks the pinned snapshot with ``recall_pure``
under the released policy, renders exact V1 bytes from the delivery
projections, and then, at the head of the application's FIFO delivery queue
and inside one outermost Writer critical section, rechecks authority and every
shown byte (``SnapshotReadGuard.check_search_delivery``), writes the ``recall``
delivery row with its ``latency_ms`` and enqueues the response. A changed hit,
end or quote line makes the attempt ``delivery_stale``: the Search is recomputed
once at a fresh pin, keeping its FIFO position, and a second stale attempt is
429 ``delivery_busy``. Failures are typed errors, never an apparently successful
empty answer. No provider call runs under the Writer mutex.

A plugin delivery (``PluginDelivery``, ``--concurrency interactive``) takes no Writer
mutex at all: it rechecks against the published root and enqueues under the read
model's condition (``SnapshotReadGuard.deliver_search_published``), queues its
``recall`` row for a batched write after the response (``store/search_rows.py``),
waits only in its own user's FIFO (``ScopedDeliveryQueues``) and runs with priority
over background Add work (``service/search_priority.py``). A background Add
therefore never holds up its user's Search. A knowledge read keeps the Writer path.

An attempt that ranked but could not deliver (it was not yet the FIFO head, or the
Writer was busy) unwinds and tries again from a fresh pin. The next attempt takes the
earlier ranking's locators when an exact digest of every ranking input
(``_ranking_digest``) is unchanged, instead of ranking again; everything after the
ranking, the delivery recheck included, runs as before. A large user's concurrent
Searches each ranked twice (about 71 s per Search for one large user).
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import asdict, astuple, dataclass
from typing import Any, Literal, Protocol, cast

from quotemem.api.bench import SearchDelivery, SearchRequest, SearchResult
from quotemem.api.bench_conflict_judge import ConflictJudge, conflict_noted
from quotemem.api.bench_conflicts import ConflictCandidate, lexical_pairs
from quotemem.api.bench_interval import delivered_interval
from quotemem.api.bench_premise import PremiseJudge, noted
from quotemem.api.bench_query import (
    PreparedBenchQuery,
    QueryVectorCache,
    prepare_bench_query,
    search_mode,
)
from quotemem.api.bench_reader import ReaderNote, reader_noted
from quotemem.api.bench_render import delivered_said, frozen_quote_speakers, render_projections
from quotemem.api.bench_rerank import PoolReranker
from quotemem.api.bench_temporal import temporal_note
from quotemem.api.interactive_search import DeadlineDelivery, InteractiveSearch, render_compact
from quotemem.core.add_read_view import EvidenceRecord, MessageKey, MessageRecord
from quotemem.core.controls import (
    HANDLED_PURPOSE,
    ForgetAddV1,
    ForgetKeptV1,
    ForgetMarksV1,
    ForgetShadowV1,
    ForgetWithdrawnV1,
)
from quotemem.core.delivery import DeliveryProjection, MemoryMode, QuoteKey, delivery_projections
from quotemem.core.entities import USER, QuoteSpanV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import integrity_digest, new_ulid, scope_id
from quotemem.core.knowledge import KnowledgeReadView
from quotemem.core.query import FrozenScopeGrantSetV1, QueryBundleV1, ReturnedItemV1, ScopeV1
from quotemem.core.readability import Readability, VersionContent
from quotemem.core.recall import carried_parts, recall_interactive, recall_pure
from quotemem.core.recall_agent import SHADOW_PURPOSE, AgentContext, ForgetShadow, Status
from quotemem.core.recall_corpora import MemoryDocument
from quotemem.core.recall_snapshot import RecallSnapshot, evidence_record
from quotemem.core.run_inputs import InputMessageV1
from quotemem.ingestion.forget_requests import asked_part, forget_requests, request_reach
from quotemem.ingestion.relearn import asserts
from quotemem.providers.auth import AuthProvider, AuthRequest, PrincipalGrant
from quotemem.providers.contracts import EmbeddingProvider
from quotemem.service.delivery import (
    DeliveryAttempt,
    DeliveryQueue,
    DeliveryTicket,
    ScopedDeliveryQueues,
)
from quotemem.service.search_priority import SearchActivity
from quotemem.store.domain import ReplayableRecallRowV1
from quotemem.store.knowledge import read_knowledge_view
from quotemem.store.read_model import SearchContext, SnapshotReadGuard, StoreReadModel
from quotemem.store.search_rows import DeferredRecallRows
from quotemem.store.source_labels import source_label_view
from quotemem.store.writer import Writer

_BENCH_RANKING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "aml_search_bench_ranking", default=False)
"""Set while a forget by statement searches (`bench_ranked`)."""
MAX_SHADOW_DIGESTS = 200_000
"""Term digests the interactive Search keeps for its forget shadow before starting over."""
MAX_STALE_RECOMPUTES = 1
"""One recompute at a fresh pin; the next stale attempt is ``delivery_busy``."""

type Profile = Literal["product", "bench"]


def _clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _check_delivery(delivery: SearchDelivery) -> None:
    if delivery.cancelled or time.monotonic() >= delivery.deadline:
        raise QuoteMemError("delivery_busy", "Search delivery is no longer available")


def _check_guarded(delivery: SearchDelivery, guard: SnapshotReadGuard) -> None:
    _check_delivery(delivery)
    if guard.is_cancelled:
        raise QuoteMemError("index_rebuilding", "Search snapshot was revoked")


def _latency_ms(delivery: SearchDelivery, now: float) -> int:
    """Whole milliseconds from the complete request bytes to this delivery."""
    return max(0, round((now - delivery.received_at) * 1000))


@dataclass(frozen=True)
class PluginDelivery:
    """The delivery of a plugin service process (no Writer mutex on Search).

    Attributes:
        rows: The deferred writer of the delivered Searches' ``recall`` rows.
        activity: The process's running Searches, which background Add work yields to.
        queues: One FIFO per user scope.
    """

    rows: DeferredRecallRows
    activity: SearchActivity
    queues: ScopedDeliveryQueues


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
    ranked: tuple[str, tuple[ReturnedItemV1, ...], Mapping[str, Status]] | None = None
    """The ranking inputs' digest, the locators ranked for them and (interactive) the
    delivered memories' statuses.

    Content-free: ids, revisions, content hashes, ranks, scores and statuses.
    """
    lexical_only: bool = False
    """An interactive Search whose query embedding missed its deadline ranks by keywords."""


def _ranking_digest(
    root: RecallSnapshot,
    scope: ScopeV1,
    bundle: QueryBundleV1,
    allowed: FrozenScopeGrantSetV1,
    mode: MemoryMode,
    withdrawn: frozenset[str],
    *,
    reranked: bool,
    agent: AgentContext | None = None,
) -> str:
    """The exact digest of every input of one ``recall_pure`` call but its cache.

    The pinned root is named by its generation and root identity (a pure function of
    its rows), with its scope fence, embedding binding and policy; a forget or any
    other change of the scope therefore changes the digest. The bundle is complete
    (query, options, clock, vector, policy, routes, exclusions and carried parts).
    The cache only memoizes the root, and a knowledge read is never reused, so its
    rows are no input here. Floats encode in their shortest round-trip form, so equal
    digests mean equal inputs. An interactive ranking (``agent``) adds the scope's marks
    and recorded forget shadow, as keys only.
    """
    value: dict[str, object] = {
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
    if agent is not None:
        recorded = agent.recorded
        shadow: dict[str, object] = {
            "marks": [list(mark) for mark in agent.marks],
            "names": sorted(recorded.names),
            "topics": sorted(sorted(topic) for topic in recorded.topics),
        }
        if agent.unmatched:
            # Adds whose requests' words hide nothing more (`AgentContext.unmatched`).
            shadow["unmatched"] = sorted(agent.unmatched)
        if recorded.name_times or recorded.topic_times:
            # A complete forget's stamps, only where it wrote them.
            shadow["times"] = [
                sorted(recorded.name_times.items()),
                sorted((sorted(topic), when) for topic, when in recorded.topic_times.items()),
            ]
        value["agent"] = shadow
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
    """Resolve the shown spans' speakers from the frozen labels at one pin.

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


class ControlReader(Protocol):
    """Reads one Control by key: the Writer, or a stopped store (``ExistingStore``)."""

    def get_control(self, key: str) -> dict[str, Any] | None: ...


def scope_marks(writer: ControlReader, scope: ScopeV1) -> tuple[QuoteSpanV1, ...]:
    """The scope's marked sentences (``forget_marks:{s}``); another scope's are never read."""
    stored = writer.get_control(f"forget_marks:{scope.scope_id}")
    return () if stored is None else tuple(ForgetMarksV1.model_validate(stored["value"]).spans)


def scope_withdrawn(writer: ControlReader, scope: ScopeV1) -> frozenset[str]:
    """The summary-only memories an in-chat forget withdrew (``forget_withdrawn:{s}``)."""
    stored = writer.get_control(f"forget_withdrawn:{scope.scope_id}")
    return frozenset() if stored is None else frozenset(
        ForgetWithdrawnV1.model_validate(stored["value"]).memory_ids)


def agent_context(
    writer: ControlReader, scope: ScopeV1, digest: Callable[[str], str],
    bind: Callable[[str, Any], str],
) -> AgentContext:
    """An agent read's forget inputs: marks, the recorded shadow, the request finder, what
    the complete forgets kept and the Adds whose in-chat forget matched nothing.

    Each is read from its own Control (another scope's is never read); the shadow holds
    keyed digests only, and ``digest`` keys candidate terms the same way
    (``Writer.bind_content`` under `SHADOW_PURPOSE`); ``bind`` is ``Writer.bind_content``,
    which keys an Add as ``forget_add:{s}`` does (`HANDLED_PURPOSE`).
    """
    stored = writer.get_control(f"forget_shadow:{scope.scope_id}")
    recorded = ForgetShadow(digest=digest)
    if stored is not None:
        value = ForgetShadowV1.model_validate(stored["value"])
        names: list[str] = list(value.names)
        topics: tuple[frozenset[str], ...] = tuple(frozenset(topic) for topic in value.topics)
        # A complete forget (the ``agent`` Add mode) stamps its terms.
        name_times: dict[str, int] = dict(zip(names, value.name_times, strict=True)) if (
            value.name_times) else {}
        topic_times: dict[frozenset[str], int] = dict(zip(
            topics, value.topic_times, strict=True)) if value.topic_times else {}
        # A stamped topic keyword's forget read the scope when it stamped it: a hit stored
        # before is what it left. A name is not: only what its forget judged and kept
        # is (``forget_kept:{s}``).
        recorded = ForgetShadow(
            frozenset(names), topics, digest=digest,
            name_times=name_times, topic_times=topic_times, topic_seen=topic_times)
    return AgentContext(
        tuple((span.evidence_id, span.byte_start, span.byte_end)
              for span in scope_marks(writer, scope)),
        recorded, _asked, asserts, scope_kept(writer, scope), scope_unmatched(writer, scope),
        lambda request_id: bind(HANDLED_PURPOSE, {"scope": scope.scope_id, "add": request_id}),
        _reach)


def scope_unmatched(writer: ControlReader, scope: ScopeV1) -> frozenset[str]:
    """The keyed digests of the scope's Adds whose in-chat forget ran and matched nothing for
    any of their requests (``forget_add:{s}``); a pending or forgotten request leaves its Add
    out."""
    stored = writer.get_control(f"forget_add:{scope.scope_id}")
    if stored is None:
        return frozenset()
    return frozenset(entry.add for entry in ForgetAddV1.model_validate(stored["value"]).entries
                     if entry.requests
                     and all(row.status == "nothing_matched" for row in entry.requests))


def scope_kept(
    writer: ControlReader, scope: ScopeV1,
) -> dict[tuple[str, str], tuple[tuple[int, int], ...]]:
    """What the scope's complete forgets judged and kept (``forget_kept:{s}``): each kept
    memory and Evidence part with the clock windows of the forgets that kept it."""
    stored = writer.get_control(f"forget_kept:{scope.scope_id}")
    if stored is None:
        return {}
    found: dict[tuple[str, str], tuple[tuple[int, int], ...]] = {}
    for entry in ForgetKeptV1.model_validate(stored["value"]).entries:
        window = ((entry.start, entry.end),)
        for key in (*(("memory", memory_id) for memory_id in entry.memories),
                    *(("evidence", part_id) for part_id in entry.parts)):
            found[key] = found.get(key, ()) + window
    return found


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
            raise QuoteMemError("delivery_stale", "A ranked memory changed before rendering")
        versions.append(MemoryDocument.of(row).snapshot)
    return tuple(versions)


class SearchUseCase:
    """Compose the registered Search authority, recall v4 ranking and guarded delivery.

    Args:
        writer: The sole initialized Writer.
        publisher: That Writer's registered snapshot publisher.
        auth: The exact-scope authorization boundary, resolved outside QuoteMem locks.
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
        interactive: The interactive Search profile (``interactive_search``): compact
            hits, a deadline and a keyword fallback for a late query embedding. It
            takes none of the four model notes.
        plugin_delivery: Optional plugin delivery (``--concurrency
            interactive``); None keeps the bench delivery under the Writer mutex. A
            knowledge read always delivers under the Writer through ``queue``.

    Construction starts no worker and establishes no readiness.

    Raises:
        QuoteMemError: ``schema_invalid`` for an unreleased profile, or for an interactive
            Search given a model note.
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
        interactive: InteractiveSearch | None = None,
        plugin_delivery: PluginDelivery | None = None,
    ) -> None:
        if StoreReadModel.get(writer) is not publisher:
            raise QuoteMemError("binding_mismatch",
                                "Search requires the Writer's registered publisher")
        if profile not in ("product", "bench"):
            raise QuoteMemError("schema_invalid", "Search profile is not released")
        if interactive is not None and any(
            note is not None for note in (premise, rerank, conflicts, reader)
        ):
            raise QuoteMemError("schema_invalid", "An interactive Search takes no model note")
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
        self._interactive = interactive
        self._plugin_delivery = plugin_delivery
        # Interactive query embeddings run here, so a late one is left to finish on its
        # own provider timeout (and fill the vector cache) while the Search moves on.
        self._embeddings = None if interactive is None else ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="quotemem-search-embedding",
        )
        self._shadow_keys: dict[str, str] = {}

    def __call__(self, request: SearchRequest, delivery: SearchDelivery) -> None:
        """Run ``search``; the controller's Search operation takes this object."""
        self.search(request, delivery)

    def search(self, request: SearchRequest, delivery: SearchDelivery) -> None:
        """Deliver one Search exactly once at the final guarded enqueue.

        Args:
            request: The admitted exact-user request; query and options serve
                retrieval only.
            delivery: Transport deadline, cancellation and the atomic enqueue.

        Returns:
            None after the response and its ``recall`` row are committed in one
            Writer critical section; a plugin delivery returns once the
            response is enqueued, with its row queued for the batched write.

        Raises:
            QuoteMemError: Search failures: authorization, read-model, corpus, cache or
                embedding failures (429), ``delivery_busy`` after the
                single stale recompute or when the delivery budget ends.
        """
        if (
            not isinstance(cast(object, request), SearchRequest)
            or type(request.user_id) is not str or not request.user_id
        ):
            raise QuoteMemError("invalid_request", "Search requires an exact user identity")
        scope = ScopeV1(scope_kind="user", scope_id=scope_id("user", request.user_id))
        mode = search_mode(self._profile, request.as_of)
        if self._interactive is not None:
            delivery = DeadlineDelivery(
                delivery, delivery.received_at + self._interactive.deadline_seconds,
            )
        _check_delivery(delivery)
        plugin_delivery = self._plugin_delivery
        with ExitStack() as stack:
            queue = self._queue
            if plugin_delivery is not None:
                stack.enter_context(plugin_delivery.activity.searching())
                if mode != "knowledge":
                    queue = stack.enter_context(plugin_delivery.queues.queue(scope.scope_id))
            published = queue is not self._queue
            self._search(request, scope, mode, delivery, queue, published=published)

    def _search(
        self,
        request: SearchRequest,
        scope: ScopeV1,
        mode: MemoryMode,
        delivery: SearchDelivery,
        queue: DeliveryQueue,
        *,
        published: bool,
    ) -> None:
        """Run one Search's attempts in ``queue``; ``published`` delivers without the Writer."""
        state = _SearchState(self._resolve(scope), self._clock())
        _check_delivery(delivery)
        with queue.ticket(delivery.deadline) as ticket:
            with delivery.on_cancel(lambda: queue.cancel(ticket)):
                queue.wait_for_prefetch(ticket)
                while True:
                    _check_delivery(delivery)
                    try:
                        outcome, generation = self._attempt(
                            request, scope, mode, delivery, queue, ticket, state,
                            published=published,
                        )
                    except QuoteMemError as error:
                        if error.code != "delivery_stale":
                            raise
                        # The attempt's guard and content have already unwound.
                        state.recomputes += 1
                        if state.recomputes > MAX_STALE_RECOMPUTES:
                            raise QuoteMemError(
                                "delivery_busy", "Search state changed again before delivery",
                            ) from None
                        state.grant = self._resolve(scope)
                        state.query_time = self._clock()
                        continue
                    if outcome == "delivered":
                        return
                    if outcome == "queued":
                        queue.wait_for_turn(ticket)
                    elif not self._writer.mutex.wait_for_progress(
                        generation, deadline=delivery.deadline, cancellation=delivery,
                    ):
                        raise QuoteMemError("delivery_busy", "Search delivery wait ended")

    def _marks(self, scope: ScopeV1) -> tuple[QuoteSpanV1, ...]:
        """The scope's marked sentences (``forget_marks:{s}``).

        Marks only grow (``StatementForget.chat_forget``), so the latest set is the one to
        show; another scope's marks are never read.
        """
        return scope_marks(self._writer, scope)

    def _withdrawn(self, scope: ScopeV1) -> frozenset[str]:
        """The scope's summary-only memories an in-chat forget withdrew (``forget_withdrawn:{s}``).

        The record only grows (``StatementForget.chat_forget``); another scope's is never
        read.
        """
        return scope_withdrawn(self._writer, scope)

    def _shadow_key(self, term: str) -> str:
        """The keyed digest of a forget shadow term, memoized (``forget_shadow:{s}``)."""
        key = self._shadow_keys.get(term)
        if key is None:
            if len(self._shadow_keys) >= MAX_SHADOW_DIGESTS:
                self._shadow_keys = {}
            key = self._shadow_keys[term] = self._writer.bind_content(SHADOW_PURPOSE, term)
        return key

    def _agent_context(self, scope: ScopeV1) -> AgentContext:
        """The interactive ranking's forget inputs: marks, the recorded shadow, requests.

        Each is read from its own Control (another scope's is never read); the shadow
        holds keyed digests only, and candidate terms are keyed the same way.
        """
        return agent_context(self._writer, scope, self._shadow_key, self._writer.bind_content)

    def _resolve(self, scope: ScopeV1) -> PrincipalGrant:
        return self._auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock()))

    def _attempt(
        self,
        request: SearchRequest,
        scope: ScopeV1,
        mode: MemoryMode,
        delivery: SearchDelivery,
        queue: DeliveryQueue,
        ticket: DeliveryTicket,
        state: _SearchState,
        *,
        published: bool,
    ) -> tuple[DeliveryAttempt, int]:
        """One guarded computation and at most one head delivery attempt."""
        context = self._publisher.begin_search(
            state.grant, query_time=state.query_time, raw_user_id=request.user_id,
            clock=self._clock,
        )
        # The provider call runs here: after the content-free pin metadata,
        # before any guard, reservation or Writer critical section.
        state.prepared = self._prepare(request, mode, context, delivery, state)
        prepared = state.prepared
        _check_delivery(delivery)
        # Only the ready FIFO head carries ticket priority for the Writer; a
        # published delivery never takes the Writer.
        reservation: AbstractContextManager[None] = (
            self._writer.mutex.delivery_reservation()
            if not published and queue.is_head(ticket) else nullcontext()
        )
        outcome: DeliveryAttempt = "queued"
        generation = 0
        with reservation, self._publisher.pin_search(context) as guard:

            def operation(root: RecallSnapshot) -> None:
                nonlocal outcome, generation
                outcome, generation = self._compute_and_deliver(
                    request, scope, mode, delivery, queue, ticket, guard, context, root,
                    guard.search_bundle(context, prepared.bundle), state, published=published,
                )

            try:
                guard.consume_search(context, operation)
            except QuoteMemError as error:
                if cast(DeliveryAttempt, outcome) == "delivered":
                    # Revoked after its delivery committed: the response stands.
                    return outcome, generation
                if error.code == "index_rebuilding" and guard.is_cancelled:
                    # A Search reads its own scope's view, so another scope's drain
                    # never revokes it. A revocation left (an embedding
                    # change, a failed publication, a finalize phase change of its
                    # own scope) is its one recompute; its own scope's fence answers
                    # that recompute's pin with 429 ``index_rebuilding``.
                    raise QuoteMemError("delivery_stale", "Search snapshot was revoked") from None
                raise
        # No guard, row or rendered body leaves the attempt.
        return outcome, generation

    def _prepare(
        self,
        request: SearchRequest,
        mode: MemoryMode,
        context: SearchContext,
        delivery: SearchDelivery,
        state: _SearchState,
    ) -> PreparedBenchQuery:
        """Freeze the query bundle; an interactive Search waits for its embedding only so long.

        An interactive Search whose embedding has not answered `InteractiveSearch.embedding_seconds`
        after the request, or failed with ``provider_unavailable``, freezes the keyword
        routes alone for the rest of the Search (``state.lexical_only``). The late call
        still finishes on its pool thread and fills the query-vector cache.
        """
        frozen = context.allowed_scopes

        def freeze(lexical_only: bool) -> PreparedBenchQuery:
            return prepare_bench_query(
                request, policy=context.recall_policy, mode=mode,
                embedding_space=context.embedding_space, embedding_epoch=context.embedding_token,
                query_time=context.query_time, embedding=self._embedding,
                previous=state.prepared, vector_cache=self._vectors,
                vector_isolation=None if self._vectors is None else integrity_digest({
                    "principal_id": frozen.principal_id,
                    "requested_scope": frozen.requested_scope.model_dump(mode="json"),
                    "grants": [grant.model_dump(mode="json") for grant in frozen.grants],
                }),
                lexical_only=lexical_only,
            )

        if self._interactive is None or self._embeddings is None:
            return freeze(False)
        if not state.lexical_only:
            wait = delivery.received_at + self._interactive.embedding_seconds - time.monotonic()
            future = self._embeddings.submit(contextvars.copy_context().run, freeze, False)
            try:
                return future.result(timeout=max(0.0, wait))
            except FutureTimeout:
                pass
            except QuoteMemError as error:
                if error.code != "provider_unavailable":
                    raise
            state.lexical_only = True
        return freeze(True)

    def _compute_and_deliver(
        self,
        request: SearchRequest,
        scope: ScopeV1,
        mode: MemoryMode,
        delivery: SearchDelivery,
        queue: DeliveryQueue,
        ticket: DeliveryTicket,
        guard: SnapshotReadGuard,
        context: SearchContext,
        root: RecallSnapshot,
        bundle: QueryBundleV1,
        state: _SearchState,
        *,
        published: bool,
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
            # The recorded bundle keeps the parts the omitted query middle carried.
            bundle = QueryBundleV1.model_validate(
                {**bundle.model_dump(mode="json"), "carried": carried},
            )
        allowed = context.allowed_scopes
        withdrawn = self._withdrawn(scope)
        # The interactive profile ranks plain and current reads for an agent plugin.
        agent = self._agent_context(scope) if (
            self._interactive is not None and knowledge is None
            and mode in {"plain", "current"} and not _BENCH_RANKING.get()) else None
        # A knowledge read ranks its selected projections; it is never reused.
        key = None if knowledge is not None else _ranking_digest(
            root, scope, bundle, allowed, mode, withdrawn, reranked=self._rerank is not None,
            agent=agent,
        )
        statuses: Mapping[str, Status] = {}
        if key is not None and state.ranked is not None and state.ranked[0] == key:
            returned = tuple(item.model_copy() for item in state.ranked[1])
            statuses = state.ranked[2]
        elif agent is not None and self._interactive is not None:
            ranking = recall_interactive(
                root, bundle, bundle.policy_snapshot, allowed_scopes=allowed, mode=mode,
                context=agent, depth=self._interactive.max_top_k,
                cache=self._publisher.recall_corpora(root),
                withdrawn=withdrawn, lexical_only=state.lexical_only,
            )
            returned, statuses = ranking.returned, ranking.statuses
        else:
            returned = recall_pure(
                root, bundle, bundle.policy_snapshot, allowed_scopes=allowed,
                mode=mode, cache=self._publisher.recall_corpora(root),
                knowledge=None if knowledge is None else {
                    row.object_id: row for row in knowledge.root.memory_rows(scope.scope_id)
                },
                rerank=self._rerank,
                withdrawn=withdrawn,
                lexical_only=state.lexical_only,
            )
        state.ranked = None if key is None else (key, returned, statuses)
        _check_guarded(delivery, guard)
        versions = _delivered_versions(shown, returned)
        speakers = quote_speakers(writer, shown, scope, versions, now=context.query_time)
        projections = delivery_projections(
            shown, scope=scope, returned=returned, mode=mode, at_time=context.query_time,
            speakers=speakers, as_of=as_of,
            remainders=agent is not None and self._interactive is not None,
        )
        result = render_projections(
            projections, scope=scope, query=request.query, marks=self._marks(scope),
        ) if self._interactive is None else render_compact(
            projections, scope=scope, query=request.query, marks=self._marks(scope),
            lexical_only=state.lexical_only,
            statuses=None if agent is None else statuses,
        )
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
        # Fresh authority for the delivery instant, resolved outside QuoteMem locks.
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
                except QuoteMemError as error:
                    if error.code != "delivery_stale":
                        raise
                    # A stale attempt keeps this FIFO position for its recompute.
                    stale = True
                    return False
            finally:
                writer.mutex.release()
            return True

        def commit_published() -> bool:
            nonlocal stale
            assert self._plugin_delivery is not None
            rows = self._plugin_delivery.rows
            now = self._clock()
            _check_delivery(delivery)
            row = self._recall_row(context, bundle, returned, delivery, now)

            def enqueue() -> None:
                _check_delivery(delivery)
                rows.enqueue(row, erase_epoch=context.fence.erase_epoch,
                             consent_epoch=context.fence.consent_epoch)
                delivery.deliver(result)

            try:
                guard.deliver_search_published(
                    context, fresh, projections, mode=mode, at_time=now, speakers=speakers,
                    as_of=as_of, enqueue=enqueue,
                )
            except QuoteMemError as error:
                if error.code != "delivery_stale":
                    raise
                stale = True
                return False
            return True

        outcome = queue.attempt(ticket, commit_published if published else commit)
        if stale:
            raise QuoteMemError("delivery_stale", "Search requires its one fresh recompute")
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
        # The user's own readable names (``p_name`` at the Search clock): a question
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
        writer.record_search_delivery(self._recall_row(context, bundle, returned, delivery, now))
        delivery.deliver(result)

    def _recall_row(
        self,
        context: SearchContext,
        bundle: QueryBundleV1,
        returned: tuple[ReturnedItemV1, ...],
        delivery: SearchDelivery,
        now: int,
    ) -> ReplayableRecallRowV1:
        """The ``recall`` delivery row of one Search delivered at ``now``."""
        writer = self._writer
        frozen = context.allowed_scopes
        scope = frozen.requested_scope
        bench = self._profile == "bench"
        return ReplayableRecallRowV1(
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
        )


@functools.lru_cache(maxsize=16_384)
def _asked(text: str) -> tuple[str, ...]:
    """What a text, read as one user message, asks to forget (the in-chat forget's finder,
    with the forms the ``agent`` mode's plugins' users write: an interactive Search serves
    those plugins, whose in-chat forget reads them too).

    Memoized: every interactive Search of a scope reads the same parts again.
    """
    return tuple(asked_part(request.statement) for request in forget_requests(
        [InputMessageV1(role="user", content=text)], complete=True))


@functools.lru_cache(maxsize=16_384)
def _reach(sentence: str) -> int | None:
    """Where the request of a request sentence ends (``forget_requests.request_reach``, with
    the ``agent`` mode's forms, as `_asked` reads them). Memoized as `_asked` is."""
    return request_reach(sentence, complete=True)


def bench_ranked(
    search: Callable[[SearchRequest, SearchDelivery], None],
) -> Callable[[SearchRequest, SearchDelivery], None]:
    """A Search operation whose Searches use the bench ranking in every profile.

    A forget by statement searches for what to forget; the interactive ranking withholds
    what forget requests name, which is exactly what the forget must find. Delivery,
    rendering and the deadline stay the profile's.

    Args:
        search: The controller's Search operation (it runs the use case on its caller's
            thread).

    Returns:
        The operation, ranking as bench.
    """
    def ranked(request: SearchRequest, delivery: SearchDelivery) -> None:
        token = _BENCH_RANKING.set(True)
        try:
            search(request, delivery)
        finally:
            _BENCH_RANKING.reset(token)

    return ranked
