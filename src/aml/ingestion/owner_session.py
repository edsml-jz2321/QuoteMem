"""The production owner session (C.2 steps 0-2 and 8; I15): the real Writer behind ComposedOwner.

``WriterOwnerSessions`` is the ``OwnerSessions`` the factory binds. One invocation:

1. binds the issued lease ticket (``Writer.finalize_ticket``; a live one is reused),
   resolves a fresh exact grant, and moves a ``preparing`` cycle to ``held``
   (``close_finalize_drain``, ``hold_finalize``); a ``held`` one is held again,
   which binds a fresh baseline to a cycle that boot adopted (C.13 R13); a held
   owner a privacy erase left behind first moves to the scope's current epochs
   (``finalize_privacy.held_ticket``, C.11.1 #1);
2. pins the owner's input root (``StoreReadModel.pin_finalize_input``) and, for a
   cycle that has not frozen yet, freezes and selects its input (C.2a:
   ``prepare_finalize_input`` + ``persist_finalize_input``);
3. reads the frozen prefix's message keys from the selected input's receipt
   census, and runs the body inside the pinned Add view
   (``SnapshotReadGuard.consume_add_view``, with the owner-private
   ``TombstoneIndex``); the view only exists inside that callback;
4. after the callback, a completed body records the completion
   (``record_add_completion``) and releases the owner (``release_finalize``).
   The completion reads every position the owner settled from its durable
   records (``owner_positions``), so the dispositions settled before the
   judge, the duplicates written with their representative and what a stopped
   invocation committed are counted; admission refusals come from the units'
   stored admission records, and the units a provider refusal ended are listed.
   Its census carries the owner's phases and the rounds' counts, both null for
   an invocation that did not run C.2 from its start (INV H09). An owner whose
   completion an earlier invocation recorded runs no stage again: it attaches
   that record if the invocation died before attaching it (``stored_completion``)
   and continues its release from it (C.13 R13).

The session never grants storage authority: every Writer call re-proves the
lease, the epochs and the grant. A refusal and a provider pause are durable
owner stops (``refuse_budget``; ``owner_pause.pause_owner``). A pause records
the stage that failed: extraction, statement embedding or the judge.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Final

from aml.core.add_outcome import (
    AddCompletionRecordV1,
    OwnerCensusV1,
    OwnerPhaseSecondsV1,
    ReentryCensusV1,
    ReviewedPositionsV1,
    Round2CensusV1,
    UnitRefusalV1,
    attribute_outcomes,
)
from aml.core.add_read_view import AddReadView, EvidenceRecord, MessageKey, MessageRecord
from aml.core.errors import AMLError
from aml.core.extraction_bounds import ExtractionRefusalDetailV1
from aml.core.extraction_policy import ExtractionPolicyConfigV2
from aml.core.identity import ScopeId, vector_encodings
from aml.core.query import ScopeV1
from aml.core.readability import Readability
from aml.core.recall_snapshot import AddViewCarry
from aml.core.run_inputs import FinalizeOwnerRunInputV1
from aml.core.technical_types import MessageRangeKeyV2
from aml.core.timing import FinalizeCompletion
from aml.ingestion.context import (
    ExtractionContext,
    PreviousBatch,
    ScopeRequests,
    scan_scope_requests,
    select_previous_batch,
)
from aml.ingestion.extract import WriterReplies
from aml.ingestion.extraction_stage import (
    ExtractionOutcome,
    ExtractionOwner,
    Phase,
    UnitCallAccounting,
    build_unit_context,
)
from aml.ingestion.finalize_freeze import prepare_finalize_input
from aml.ingestion.finalize_receipts import FinalizeReceipt, inspect_finalize_receipts
from aml.ingestion.owner_pipeline import OwnerSession, RoundsCensus
from aml.ingestion.segment import checkpoint_partitions, message_chunk_index, message_order
from aml.ingestion.sources import LabelView, word_count
from aml.ingestion.tombstone_index import TombstoneIndex
from aml.ingestion.units import StorageBudget, Unit
from aml.providers.adapters import ProviderError
from aml.providers.auth import AuthProvider, AuthRequest, PrincipalGrant
from aml.providers.contracts import EmbeddingProvider, StructuredInferenceProvider
from aml.store.add_outcome import (
    completion_requests,
    owner_positions,
    record_add_completion,
    stored_completion,
)
from aml.store.extraction_budget import OwnerUnitAccounting
from aml.store.finalize_privacy import held_ticket
from aml.store.finalize_ticket import FinalizeLeaseTicket
from aml.store.owner_lease import (
    end_abandoned_reviews,
    persist_finalize_input,
    release_finalize,
)
from aml.store.owner_pause import OwnerStageKind, pause_owner
from aml.store.read_model import SnapshotReadGuard, StoreReadModel
from aml.store.request_completion import read_budget, refuse_budget
from aml.store.source_labels import (
    SourceLabelsKeyRefused,
    SourceLabelsPaused,
    run_source_labels,
)
from aml.store.writer import CommittedRows, Writer, WriteTicket
from aml.store.writer_mutex import ProgressCancellation

type Measure = Callable[[ExtractionOwner, Unit, int], StorageBudget]
"""Measure a unit's worst admission transaction on the Writer (W3-RT-14)."""

_LOG = logging.getLogger(__name__)


def _wall_clock() -> int:
    return time.time_ns() // 1_000_000


def _receipt_message_keys(receipts: Sequence[FinalizeReceipt]) -> tuple[MessageKey, ...]:
    """Order a proved receipt census by the existing frozen source coordinates.

    The census is a directory, whose lexical identity order carries no source
    chronology. Use the same session anchors and unclocked request barriers as
    the frozen source inventory, without reading any source body.
    """
    sessions: dict[str, int | None] = {}
    chunks: dict[tuple[str, str], tuple[int, int, str, int | None]] = {}
    received: dict[tuple[str, str], tuple[int, bool]] = {}
    headers = [(receipt, receipt.headers) for receipt in receipts]
    for receipt, parts in headers:
        observed = [part.source_observed_at for part in parts
                    if part.source_observed_at is not None]
        prior = sessions.get(receipt.session_id)
        sessions[receipt.session_id] = min(
            [*observed, *([] if prior is None else [prior])], default=None,
        )
        first = min(parts, key=lambda part: (part.ordinal, part.part_no, part.object_id))
        key = receipt.session_id, receipt.request_id
        chunks[key] = first.ordinal, first.part_no, first.object_id, first.source_observed_at
        received[key] = first.ingested_commit_seq, bool(observed)
    partitions = checkpoint_partitions(sessions, chunks, received)
    ordered: dict[MessageKey, tuple[object, ...]] = {}
    for receipt, parts in headers:
        key = receipt.session_id, receipt.request_id
        for part in parts:
            coordinate = MessageRangeKeyV2(
                key_kind="message_v2", session_id=receipt.session_id,
                request_id=receipt.request_id, session_event_time=sessions[receipt.session_id],
                chunk_index=message_chunk_index(receipt.request_id),
                chunk_event_time=chunks[key][3], ordinal=part.ordinal,
                part_no=part.part_no, evidence_id=part.object_id,
                received_commit_seq=received[key][0], source_time_present=received[key][1],
                receipt_partition_seq=partitions[key],
            )
            message = MessageKey(receipt.session_id, receipt.request_id, part.ordinal)
            ordered.setdefault(message, message_order(coordinate))
    return tuple(sorted(ordered, key=ordered.__getitem__))


def _newest_within(messages: Sequence[MessageRecord], max_words: int) -> list[MessageRecord]:
    """The newest whole messages within ``max_words`` words, oldest first (B.3).

    Context selection renders the newest of them that its source budget allows (B.4.1).
    """
    kept: list[MessageRecord] = []
    words = 0
    for message in reversed(messages):
        words += sum(word_count(part.text) for part in message.parts)
        if words > max_words:
            break
        kept.append(message)
    return kept[::-1]


def _census(phases: OwnerPhaseSecondsV1, rounds: RoundsCensus | None) -> OwnerCensusV1:
    """The owner census: its phases and its rounds' counts, null when unavailable (H09)."""
    if rounds is None:
        return OwnerCensusV1(
            reentry=ReentryCensusV1.from_actions(None, digest_changed=None),
            pending_selectable=None, round1_members=None,
            round2=Round2CensusV1.from_entries(None, target_changed=None, judge_requests=None),
            owner_phase_seconds=phases, reviewed=ReviewedPositionsV1.from_rounds(None, None),
            judge_refused_members=None,
        )
    return OwnerCensusV1(
        reentry=rounds.reentry, pending_selectable=rounds.pending_selectable,
        round1_members=rounds.round1_members, round2=rounds.round2,
        owner_phase_seconds=phases, reviewed=rounds.reviewed,
        judge_refused_members=rounds.judge_refused_members,
    )


@dataclass
class _Completed:
    """A body that ended in ``complete``: the record to write after the view callback."""

    phases: OwnerPhaseSecondsV1
    outcome: ExtractionOutcome
    rounds: RoundsCensus | None


_PENDING: Final = object()


@dataclass
class _WriterExtractionOwner:
    """The ``ExtractionOwner`` of one invocation (I8), bound to its frozen inputs.

    A unit's context is a function of the owner's pin and frozen inputs, so the one
    its fitting sizing probe built is the one its extraction stores: ``measure`` keeps
    the latest fitting probe's context per unit number and ``unit_context`` serves it
    once to that exact unit, after the owner's validity check.
    """

    session: _WriterSession
    labels: LabelView
    accounting: UnitCallAccounting
    mark: Callable[[Phase], None]
    replies: WriterReplies
    probed: dict[int, tuple[tuple[str, ...], ExtractionContext | None]] = field(
        default_factory=dict[int, tuple[tuple[str, ...], ExtractionContext | None]])
    _built: tuple[int, tuple[str, ...], ExtractionContext | None] | None = None

    @property
    def owner_run_id(self) -> str:
        return self.session.owner_run_id

    @property
    def ticket(self) -> WriteTicket:
        return self.session.write_ticket

    @property
    def writer(self) -> Writer:
        return self.session.sessions.writer

    @property
    def policy(self) -> ExtractionPolicyConfigV2:
        return self.session.policy

    @property
    def prefix(self) -> Sequence[EvidenceRecord]:
        return self.session.prefix

    @property
    def view(self) -> AddReadView:
        return self.session.view

    def settlement_view(self) -> AddReadView:
        # The settlement pin is the owner's input pin: the held fence keeps the
        # scope's Memory state unchanged until release (C.2a; open item: a
        # fresh publication pin needs a non-callback read-model handle).
        return self.session.view

    @property
    def provider(self) -> StructuredInferenceProvider:
        return self.session.sessions.require_provider()

    @property
    def embedder(self) -> EmbeddingProvider:
        return self.session.sessions.require_embedder()

    def measure(self, unit: Unit, facts: int) -> StorageBudget:
        self._built = None
        budget = self.session.sessions.measure(self, unit, facts)
        key = (unit.unit_no, tuple(part.object_id for part in unit.parts))
        if budget.fits and self._built is not None and self._built[:2] == key:
            self.probed[unit.unit_no] = key[1], self._built[2]
        self._built = None
        return budget

    def unit_context(self, unit: Unit) -> ExtractionContext | None:
        parts = tuple(part.object_id for part in unit.parts)
        held = self.probed.get(unit.unit_no)
        if held is not None and held[0] == parts:
            del self.probed[unit.unit_no]
            self.check_valid()
            built = held[1]
        else:
            built = build_unit_context(self, unit)
        self._built = unit.unit_no, parts, built
        return built

    def previous_batch(self, unit: Unit) -> PreviousBatch | None:
        return self.session.previous_batch(unit)

    def authorize(self) -> None:
        self.session.check()

    def check_valid(self) -> None:
        self.session.check()

    def mark_phase(self, phase: Phase) -> None:
        self.session.stage = phase
        self.mark(phase)

    def publish(self, committed: CommittedRows) -> None:
        sessions = self.session.sessions
        sessions.publisher.publish_commit(sessions.writer, committed)

    def clock(self) -> int:
        return self.session.sessions.clock()


@dataclass
class _LabelOwner:
    """The ``LabelOwner`` of one invocation (C.2 step 0)."""

    session: _WriterSession

    @property
    def writer(self) -> Writer:
        return self.session.sessions.writer

    @property
    def run_id(self) -> str:
        return self.session.owner_run_id

    @property
    def attempt(self) -> int:
        return self.session.attempt

    @property
    def deadline_at(self) -> int:
        return self.session.deadline_at

    @property
    def read(self) -> Readability:
        view = self.session.view
        return Readability(view.scope_kind, view.scope_id, view.now, view.evidence)

    @property
    def provider(self) -> StructuredInferenceProvider:
        return self.session.sessions.require_provider()

    def grant(self) -> PrincipalGrant:
        return self.session.grant()

    def check(self) -> None:
        self.session.check()

    def clock(self) -> int:
        return self.session.sessions.clock()


@dataclass
class _WriterSession:
    """One invocation's ``OwnerSession`` over the real Writer."""

    sessions: WriterOwnerSessions
    ticket: FinalizeLeaseTicket
    scope: ScopeV1
    write_ticket: WriteTicket
    selected: FinalizeOwnerRunInputV1
    policy: ExtractionPolicyConfigV2
    attempt: int
    deadline_at: int
    whole_run: bool
    cancellation: ProgressCancellation
    keys: tuple[MessageKey, ...]
    view: AddReadView = field(init=False)
    messages: tuple[MessageRecord, ...] = ()
    prefix: tuple[EvidenceRecord, ...] = ()
    label_view: LabelView | None = None
    completed: _Completed | None = None
    stage: Phase | None = None
    _accounting: OwnerUnitAccounting | None = None
    tombstones: TombstoneIndex | None = None
    _previous: dict[tuple[str, str], PreviousBatch | None] = field(
        default_factory=dict[tuple[str, str], PreviousBatch | None],
    )
    _scan: ScopeRequests | None = None
    _previous_lock: threading.Lock = field(default_factory=threading.Lock)
    view_carry: AddViewCarry = field(default_factory=AddViewCarry)
    """The records this invocation's views of its scope reuse from each other."""

    @property
    def owner_run_id(self) -> str:
        return self.ticket.owner_run_id

    @property
    def scope_id(self) -> ScopeId:
        return self.ticket.scope_id

    @property
    def cycle(self) -> int:
        return self.ticket.lease_generation

    @property
    def accounting(self) -> UnitCallAccounting:
        if self._accounting is None:
            self._accounting = OwnerUnitAccounting(
                self.sessions.writer, self.owner_run_id, clock=self.sessions.clock,
            )
        return self._accounting

    def bind_view(self, view: AddReadView) -> None:
        """Bind the pinned view and read the frozen prefix's messages from it.

        The view starts with no previous-batch selection (``previous_batch``).
        """
        self.view = view
        self._previous = {}
        self._scan = None
        messages: list[MessageRecord] = []
        for key in self.keys:
            message = view.message(key)
            if message is None:
                raise AMLError("binding_mismatch", "A frozen prefix message left the owner pin")
            messages.append(message)
        self.messages = tuple(messages)
        self.prefix = tuple(part for message in messages for part in message.parts)

    def grant(self) -> PrincipalGrant:
        return self.sessions.grant(self.scope)

    def check(self) -> None:
        """Refuse a cancelled invocation; the Writer re-proves the lease at every write."""
        if self.cancellation.cancelled:
            raise AMLError("lease_lost", "The owner invocation was cancelled")

    def labels(self) -> LabelView:
        """Label the frozen prefix and its previous batch at the owner pin (C.2 step 0, B.3a)."""
        self.label_view = run_source_labels(
            _LabelOwner(self), self.messages, self._previous_batch(),
        )
        return self.label_view

    def previous_batch(self, unit: Unit) -> PreviousBatch | None:
        """Select the immediate receipt-proven predecessor of the unit's request (B.3).

        The selection (``select_previous_batch``) reads only the owner pin and the
        unit's request, so it is made once per request of the bound view: the label
        stage, every sizing probe and every unit's context reuse it, and a reuse
        rechecks the invocation first. The selections of the view's requests share
        one complete scan of the scope's Evidence at the pin (``scan_scope_requests``).
        ``bind_view`` starts a view with neither, so a selection never crosses an
        owner pin or a request.
        """
        key = unit.session_id, unit.request_id
        with self._previous_lock:
            if key in self._previous:
                self.check()
                return self._previous[key]
            if self._scan is None:
                self.check()
                self._scan = scan_scope_requests(
                    self.sessions.writer, self.view, check_valid=self.check,
                )
            batch = select_previous_batch(
                self.sessions.writer, self.view, unit, check_valid=self.check, scan=self._scan,
            )
            self._previous[key] = batch
            return batch

    def _previous_batch(self) -> tuple[MessageRecord, ...]:
        """Select the previous-batch messages the label stage covers (B.3, B.3a).

        Each request of the frozen prefix reads its immediate receipt-proven
        predecessor at the owner pin (``previous_batch``, the selection context
        building reuses), cut to its newest whole messages within the frozen
        ``previous_batch_words``: every previous-batch message a unit can render.
        Prefix messages are labelled as prefix. The selection reads only the
        owner pin and the frozen prefix, so every invocation of the owner selects
        the same messages and replays the same label view.
        """
        limit = self.policy.parameters.previous_batch_words
        requests: dict[tuple[str, str], Unit] = {}
        for message in self.messages:
            key = message.key.session_id, message.key.request_id
            requests.setdefault(key, Unit(0, message.parts))
        prefix = {message.key for message in self.messages}
        selected: dict[MessageKey, MessageRecord] = {}
        for unit in requests.values():
            batch = self.previous_batch(unit)
            if batch is None:
                continue
            for message in _newest_within(batch.messages, limit):
                if message.key not in prefix:
                    selected.setdefault(message.key, message)
        return tuple(selected.values())

    # The judge round and its commit (``owner_rounds.RoundSession``) ------------------------

    @property
    def writer(self) -> Writer:
        return self.sessions.writer

    @property
    def run_id(self) -> str:
        return self.ticket.owner_run_id

    @property
    def embedder(self) -> EmbeddingProvider:
        """The workspace's single embedder, which also embeds descriptor rewordings (C.9.4a)."""
        return self.sessions.require_embedder()

    def mark_stage(self, stage: Phase) -> None:
        """Name the stage whose provider failure pauses the owner from now on (C.16)."""
        self.stage = stage

    def clock(self) -> int:
        return self.sessions.clock()

    @property
    def policy_release_ref(self) -> str:
        return self.selected.extraction_policy_release_ref

    def publish(self, committed: CommittedRows) -> None:
        self.sessions.publisher.publish_commit(self.sessions.writer, committed)

    def read_view(self, consumer: Callable[[AddReadView], None]) -> None:
        """Run ``consumer`` on a fresh publication pin: the current root, after
        every commit this owner published (C.11.2 #1, C.2 step 7)."""
        sessions = self.sessions
        if self.tombstones is None:
            raise AMLError("binding_mismatch", "The owner read runs outside its session")
        with sessions.publisher.pin_finalize_publication(self.ticket) as guard:
            guard.consume_add_view(
                self.scope, now=sessions.clock(), tombstones=self.tombstones, consumer=consumer,
                carry=self.view_carry,
            )

    def at_review_pin[T](self, body: Callable[[], T]) -> T:
        """Keep a later frozen request's review and commit in one fresh view callback."""
        result: list[T] = []
        stopped: list[ProviderError | SourceLabelsPaused] = []

        def at_pin(view: AddReadView) -> None:
            previous = self.view
            self.view = view
            try:
                result.append(body())
            except (ProviderError, SourceLabelsPaused) as error:
                # As in round 2, provider failures leave the read callback
                # without its frames; raise the typed pause outside the guard.
                error.__cause__ = error.__context__ = None
                stopped.append(error.with_traceback(None))
            finally:
                self.view = previous

        self.read_view(at_pin)
        if stopped:
            raise stopped[0]
        return result[0]

    def extraction_owner(
        self, labels: LabelView, accounting: UnitCallAccounting,
        mark_phase: Callable[[Phase], None],
    ) -> ExtractionOwner:
        writer = self.sessions.writer
        replies = WriterReplies(
            writer, self.write_ticket, self.policy, authorize=self.check,
            clock=self.sessions.clock,
        )
        return _WriterExtractionOwner(self, labels, accounting, mark_phase, replies)


    def refuse(self, unit_no: int, refusal: ExtractionRefusalDetailV1) -> None:
        refuse_budget(
            self.sessions.writer, self.ticket, unit_no=unit_no, refusal=refusal,
            clock=self.sessions.clock,
        )

    def pause(self, cause: ProviderError | SourceLabelsPaused, *, judge: bool = False) -> None:
        if isinstance(cause, SourceLabelsPaused):
            # The failed label child wrote its own index (S2a); the owner only
            # stops. A refused key leaves the child running, so the owner
            # indexes the refusal for its resume (LABELS-RESUME).
            pause_owner(
                self.sessions.writer, self.ticket,
                cause.error if isinstance(cause, SourceLabelsKeyRefused) else None,
                kind="source_labels", clock=self.sessions.clock,
            )
            _LOG.warning("Add owner paused: owner=%s stage=source_labels", self.run_id)
            return
        kind: OwnerStageKind = (
            "judge" if judge else "embedding" if self.stage == "embedding" else "extract"
        )
        pause_owner(self.sessions.writer, self.ticket, cause, kind=kind, clock=self.sessions.clock)
        _LOG.warning("Add owner paused: owner=%s stage=%s error_class=%s", self.run_id, kind,
                     cause.error_class)

    def complete(self, phases: OwnerPhaseSecondsV1, outcome: ExtractionOutcome,
                 rounds: RoundsCensus | None = None) -> FinalizeCompletion:
        # Recording and release run after the view callback (``WriterOwnerSessions.run``):
        # the release publishes the owner's root, which must not happen under its pin.
        self.completed = _Completed(phases, outcome, rounds)
        return FinalizeCompletion(self.scope_id, self.owner_run_id, self.cycle, 0)


@dataclass
class WriterOwnerSessions:
    """The production ``OwnerSessions``: one Writer, its read model, auth and providers.

    Attributes:
        writer: The sole Writer.
        publisher: The Writer's registered read model.
        auth: Resolves a fresh exact grant per invocation (C.14).
        provider: The structured provider for labels and extraction.
        embedder: The statement embedding provider (B.9).
        measure: W3-RT-14's unit measurement (``owner_measure.measure_unit``).
        clock: Epoch milliseconds.
    """

    writer: Writer
    publisher: StoreReadModel
    auth: AuthProvider
    provider: StructuredInferenceProvider | None
    embedder: EmbeddingProvider | None
    measure: Measure
    clock: Callable[[], int] = _wall_clock

    def require_ready(self) -> None:
        """Refuse to run without the providers the owner calls."""
        if self.provider is None or self.embedder is None:
            raise AMLError("unsupported_variant", "The Add owner has no semantic provider")

    def require_provider(self) -> StructuredInferenceProvider:
        """The structured provider, or ``unsupported_variant``."""
        if self.provider is None:
            raise AMLError("unsupported_variant", "The Add owner has no structured provider")
        return self.provider

    def require_embedder(self) -> EmbeddingProvider:
        """The statement embedder, or ``unsupported_variant``."""
        if self.embedder is None:
            raise AMLError("unsupported_variant", "The Add owner has no statement embedder")
        return self.embedder

    def grant(self, scope: ScopeV1) -> PrincipalGrant:
        """Resolve a fresh exact grant of the owner's scope."""
        return self.auth.resolve(AuthRequest(requested_scope=scope, now_ms=self.clock()))

    def run[T](
        self, owner_run_id: str, body: Callable[[OwnerSession], T], *,
        cancellation: ProgressCancellation, concurrency: int,
    ) -> T:
        """Hold, freeze, pin and run ``body`` once inside the owner's Add view.

        Args:
            owner_run_id: The admitted owner.
            body: The composed owner's stages.
            cancellation: The invocation's cancellation.
            concurrency: The extraction concurrency (unused by the session).

        Returns:
            T: What ``body`` returned; a completion is recorded and released first.

        Raises:
            AMLError: ``lease_lost`` and the Writer's other fences; ``binding_mismatch``.
        """
        del concurrency
        self.require_ready()
        # The owner canonicalizes its scope's Evidence rows in every stage below.
        with vector_encodings():
            return self._run(owner_run_id, body, cancellation)

    def _run[T](
        self, owner_run_id: str, body: Callable[[OwnerSession], T],
        cancellation: ProgressCancellation,
    ) -> T:
        writer, clock = self.writer, self.clock
        ticket = held_ticket(
            writer, self.publisher, owner_run_id, grant=self.grant, cancellation=cancellation,
            clock=clock,
        )
        scope = ScopeV1(scope_kind="user", scope_id=ticket.scope_id)
        if ticket.state == "preparing":
            writer.close_finalize_drain(ticket, clock=clock)
        writer.hold_finalize(ticket, self.publisher, clock=clock)
        run = writer.get_row("run", owner_run_id)
        if run is None:
            raise AMLError("binding_mismatch", "The owner Run is missing")
        if run["result_ref"] is not None:
            # Interrupted between its completion record and its release: the frozen
            # record is released as it stands, never computed again (C.13 R13).
            record = writer.read_logical(run["result_ref"], "result_index")
            if not isinstance(record, AddCompletionRecordV1) or (
                record.owner_run_id != owner_run_id
            ):
                raise AMLError("binding_mismatch", "The owner's completion names another owner")
            return self._release(ticket, scope)  # type: ignore[return-value]
        if run["input_ref"] is not None and (stored := stored_completion(
            writer, owner_run_id, self._recovery_count(scope, owner_run_id),
        )) is not None:
            # Interrupted after storing its completion record, before attaching it:
            # that record is attached as it stands, never computed again (C.13 R13).
            record_add_completion(writer, ticket, stored, grant=self.grant(scope),
                                  clock=clock)
            return self._release(ticket, scope)  # type: ignore[return-value]
        whole_run = run["input_ref"] is None
        result: object = _PENDING
        session: _WriterSession | None = None
        with self.publisher.pin_finalize_input(ticket) as guard:
            if run["input_ref"] is None:
                self._freeze(ticket, guard, scope)
            session = self._session(ticket, scope, whole_run, cancellation)
            with self._tombstones(scope) as tombstones:
                session.tombstones = tombstones

                def consume(view: AddReadView) -> None:
                    nonlocal result
                    assert session is not None
                    session.bind_view(view)
                    result = body(session)

                guard.consume_add_view(
                    scope, now=clock(), tombstones=tombstones, consumer=consume,
                    carry=session.view_carry,
                )
        if result is _PENDING:
            raise AMLError("binding_mismatch", "The owner body did not run")
        if session.completed is not None:
            return self._complete(session, session.completed)  # type: ignore[return-value]
        return result  # type: ignore[return-value]

    def _freeze(
        self, ticket: FinalizeLeaseTicket, guard: SnapshotReadGuard, scope: ScopeV1,
    ) -> None:
        grant = self.grant(scope)

        def freeze() -> None:
            proof = prepare_finalize_input(self.writer, ticket, guard, grant, clock=self.clock)
            persist_finalize_input(self.writer, proof, grant=grant, clock=self.clock)

        guard.consume_operation(scope, freeze)

    @contextmanager
    def _tombstones(self, scope: ScopeV1) -> Iterator[TombstoneIndex]:
        writer = self.writer

        def check() -> None:
            with writer.mutex:
                writer.validate_storage_owner()

        index = TombstoneIndex.load(
            writer.indexed_reader(), scope,
            sequence=partial(writer.counter, f"tombstone_seq:{scope.scope_id}"), check=check,
        )
        try:
            yield index
        finally:
            index.close()

    def _session(
        self, ticket: FinalizeLeaseTicket, scope: ScopeV1, whole_run: bool,
        cancellation: ProgressCancellation,
    ) -> _WriterSession:
        from aml.store.source_labels import (
            _extraction_policy,  # pyright: ignore[reportPrivateUsage]
        )

        writer = self.writer
        run = writer.get_row("run", ticket.owner_run_id)
        if run is None or run["input_ref"] is None:
            raise AMLError("binding_mismatch", "The owner has no selected input")
        selected = writer.read_logical(run["input_ref"], "run_input")
        if not isinstance(selected, FinalizeOwnerRunInputV1):
            raise AMLError("binding_mismatch", "The owner input is not a finalize input")
        budget = read_budget(writer, scope, ticket.owner_run_id)
        if budget is None:
            raise AMLError("binding_mismatch", "The owner has no budget")
        census = inspect_finalize_receipts(
            writer, scope, receipt_high_watermark=selected.receipt_high_watermark,
            check=lambda: None,
        )
        # The prefix is the requests no earlier owner completed (C.2a frozen
        # prefix, INV:E04): a request whose first result froze is not re-extracted.
        split = completion_requests(writer, scope, census.past, check=lambda: None)
        keys = tuple(key for key in _receipt_message_keys(census.past)
                     if split.keys[(key.session_id, key.request_id)] in split.pending)
        _, _, write_ticket = writer._finalize_identity(ticket)  # pyright: ignore[reportPrivateUsage]
        value = budget.value.model_dump(mode="json")
        return _WriterSession(
            self, ticket, scope, write_ticket, selected,
            _extraction_policy(writer, selected.extraction_policy_release_ref),
            attempt=int(run["attempt"]), deadline_at=int(value["deadline_at"]),
            whole_run=whole_run, cancellation=cancellation, keys=tuple(keys),
        )

    def _complete(self, session: _WriterSession, completed: _Completed) -> FinalizeCompletion:
        writer, ticket, scope = self.writer, session.ticket, session.scope
        grant = self.grant(scope)
        census = inspect_finalize_receipts(
            writer, scope, receipt_high_watermark=session.selected.receipt_high_watermark,
            check=lambda: None,
        )
        split = completion_requests(writer, scope, census.past, check=lambda: None)
        outcome = completed.outcome
        positions = owner_positions(
            writer, ticket.owner_run_id, split, admitted=outcome.proposals.keys(),
            refused=[(item.position_key, item.code, item.sources) for item in outcome.refused],
        )
        record = AddCompletionRecordV1(
            completed_requests=attribute_outcomes(
                positions, pending=split.pending, finished=split.finished, cause=split.cause,
            ),
            owner_run_id=ticket.owner_run_id,
            recovery_count=self._recovery_count(scope, ticket.owner_run_id),
            census=_census(completed.phases, completed.rounds),
            refused_units=[UnitRefusalV1(unit_no=unit.unit_no, request_key=split.newest(
                unit.sources)) for unit in outcome.spent],
        )
        record_add_completion(writer, ticket, record, grant=grant, clock=self.clock)
        return self._release(ticket, scope)

    def _release(self, ticket: FinalizeLeaseTicket, scope: ScopeV1) -> FinalizeCompletion:
        grant = self.grant(scope)
        end_abandoned_reviews(self.writer, ticket, grant=grant, clock=self.clock)
        release_finalize(self.writer, ticket, self.publisher, grant=grant, clock=self.clock)
        return FinalizeCompletion(
            ticket.scope_id, ticket.owner_run_id, ticket.lease_generation, time.monotonic_ns(),
        )

    def _recovery_count(self, scope: ScopeV1, owner_run_id: str) -> int:
        budget = read_budget(self.writer, scope, owner_run_id)
        if budget is None:
            raise AMLError("binding_mismatch", "The owner has no budget")
        return int(budget.value.model_dump(mode="json").get("recovery_count", 0))
