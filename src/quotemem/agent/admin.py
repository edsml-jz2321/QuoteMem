"""Operations on an open agent store: details, quarantine repair and user deletion.

``AgentStore`` holds the serving runtime's own objects (its Writer, read model,
authorization, Add controller and request lifetime), captured through the
controller factory the agent composition hands the bench runtime; it adds no
second owner of anything. Every operation runs on a worker thread inside the
runtime's request lifetime, as an HTTP request does, so a shutdown drains it.

- ``details``: the store section of ``/agent/status`` (registered users,
  quarantined users, active Add owners, the embedding canary state); cached for a
  few seconds, from bounded control pages, never content.
- ``repair``: the existing explicit recovery of a quarantined user:
  ``resume`` re-runs the owner its receipt names (``CompletionResumeRequestV3``),
  which helps once the fault's cause is cleared; ``skip`` skips the quarantined
  cycle (``store.quarantine_skip``) when the fault recurs with the same input.
  ``auto_repair`` applies an opt-in policy to every quarantined user.
- ``completed_adds``: which of some Add requests already have their completed
  result, so the gateway can let a client read one back while new Adds are
  deferred, and count an Add sent with ``Prefer: respond-async`` as in flight
  until its extraction finished.
- ``retire``: ends, without a successor, the active memories most of whose
  quoted words are given ranges (``api.memory_retire``); a rebuild's replay of
  the memories its journal names as ``retired``.
- ``forget_withheld``: withholds the withheld markers of a user's Evidence by a forget
  of sentences seeded with a forgotten statement; a rebuild's replay of a forget whose
  words its journal withholds.
- ``delete_user``: erases one user's memories and sources. Their Adds are held
  off and their Add owner drains; consent is withdrawn (``store.consent_withdrawal``),
  so an Add received before the deletion is never stored; every memory is
  forgotten (the memory, its revisions, links, names and candidates are
  deleted) and every source is erased (privacy erase:
  the Evidence text, the judge records, extraction replies, proposals,
  embeddings and reception payloads citing it, and the forget tombstones whose
  sources are all erased). The report counts what remains, which should be
  nothing readable.

SQLite keeps deleted bytes in free space and in the write-ahead log until they
are overwritten. An agent store therefore runs every connection of its database
with ``secure_delete`` on (deleted content is zeroed), and ``delete_user`` ends
with a truncating checkpoint, so the log no longer holds pages from before the
erase. The bench profile does neither.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence, Set
from dataclasses import dataclass
from typing import Any, Literal, cast

from sqlalchemy import event

from quotemem.api.bench_recovery import BenchRecoveryService
from quotemem.api.completion_recovery import (
    CompletionRecoveryRequestV3,
    CompletionRecoveryResponseV3,
    CompletionResumeRequestV3,
    QuarantineSkipRequestV1,
)
from quotemem.api.memory_forget import MemoryForgetPrepareRequestV1
from quotemem.api.memory_list import MemoryListing, MemoryListRequestV1
from quotemem.api.memory_retire import MemoryRetire, QuotedRange, memories_quoting
from quotemem.core.bootstrap import InitializedWorkspace
from quotemem.core.entities import QuoteSpanV1, SupersedeRelation
from quotemem.core.errors import QuoteMemError
from quotemem.core.forget_state import MAX_RESTATING_SPANS
from quotemem.core.identity import scope_id
from quotemem.core.privacy_persistence import ConsentWithdrawalInputV1
from quotemem.core.query import ScopeV1
from quotemem.core.readability import WITHHELD_MARKER
from quotemem.core.statement import MAX_TOMBSTONE_SOURCES
from quotemem.ingestion.authorization import check_add_grant
from quotemem.ingestion.privacy_cleanup import PrivacyCleanup
from quotemem.providers.auth import AuthRequest
from quotemem.service.auth import BenchExactUserAuth
from quotemem.service.lifecycle import ServiceLifetime, ServiceUnavailable
from quotemem.service.runtime import BenchController, ControllerFactory
from quotemem.store.add_outcome import read_admitted_add
from quotemem.store.consent_withdrawal import withdraw_consent
from quotemem.store.indexed import (
    MAX_PAGE_SIZE,
    CandidateRows,
    EvidenceRows,
    IndexedQuery,
    MemoryTableRows,
    TombstoneRows,
)
from quotemem.store.read_model import StoreReadModel

type RepairAction = Literal["inspect", "resume", "skip"]
type RepairPolicy = Literal["off", "resume", "skip"]

DETAILS_TTL_MS = 5_000
MAX_ERASE_SOURCES = 512
"""Sources per privacy erase: ``_PrivacyEraseInputV1``'s bound."""
DRAIN_TIMEOUT_SECONDS = 120.0
"""How long ``delete_user`` waits for the user's Add owner and fences to finish."""
CHECKPOINT_TIMEOUT_SECONDS = 30.0


def _engine(ready: InitializedWorkspace) -> Any:
    """The SQLAlchemy engine of the store's ``memory.db``."""
    workspace: Any = ready.writer.runtime.workspace
    return workspace.backends.get("main").db.engine


def secure_delete(ready: InitializedWorkspace) -> None:
    """Zero deleted content on every connection of the store's database from now on."""
    def checkout(connection: Any, _record: Any, _proxy: Any) -> None:
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA secure_delete = ON")
        finally:
            cursor.close()

    event.listen(_engine(ready), "checkout", checkout)


@dataclass
class _Captured:
    ready: InitializedWorkspace
    publisher: StoreReadModel
    auth: BenchExactUserAuth
    controller: BenchController


def repair_policy(value: str) -> RepairPolicy:
    """Parse ``QUOTEMEM_AUTO_REPAIR``: ``off`` (default), ``resume`` or ``skip``.

    Raises:
        QuoteMemError: ``schema_invalid`` for any other value.
    """
    value = value.strip().lower() or "off"
    if value not in {"off", "resume", "skip"}:
        raise QuoteMemError("schema_invalid", "QUOTEMEM_AUTO_REPAIR is off, resume or skip")
    return cast(RepairPolicy, value)


class AgentStore:
    """The open store's operations for the agent gateway.

    Args:
        clock_ms: The trusted wall clock in epoch milliseconds.
        sleep: Waits between polls (tests replace it).
    """

    def __init__(
        self, *, clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock_ms = clock_ms
        self._sleep = sleep
        self._captured: _Captured | None = None
        self._lifetime: ServiceLifetime | None = None
        self._details: tuple[int, dict[str, object]] | None = None
        self._lock = threading.Lock()
        self._deleting: set[str] = set()
        self._repaired: dict[str, str] = {}

    # Composition -----------------------------------------------------------------
    def capturing(self, factory: ControllerFactory) -> ControllerFactory:
        """Wrap the runtime's controller factory to keep its objects; nothing else changes."""
        def build(
            ready: InitializedWorkspace, publisher: StoreReadModel, auth: BenchExactUserAuth,
            /, **kwargs: Any,
        ) -> BenchController:
            controller = factory(ready, publisher, auth, **kwargs)
            secure_delete(ready)
            self._captured = _Captured(ready, publisher, auth, controller)
            return controller

        return cast(ControllerFactory, build)

    def attach(self, lifetime: ServiceLifetime) -> None:
        """Bind the runtime's request lifetime once it has started."""
        self._lifetime = lifetime

    @property
    def open(self) -> bool:
        """Whether the runtime has started and is bound."""
        return self._captured is not None and self._lifetime is not None

    def _admitted[T](self, operation: Callable[[_Captured], T]) -> T:
        captured, lifetime = self._captured, self._lifetime
        if captured is None or lifetime is None:
            raise QuoteMemError("service_unhealthy", "The store is not open")
        try:
            with lifetime.admission():
                return operation(captured)
        except ServiceUnavailable:
            raise QuoteMemError("service_unhealthy", "The store is stopping") from None

    def _scope(self, captured: _Captured, user_id: str) -> ScopeV1:
        if not user_id:
            raise QuoteMemError("invalid_request", "A user id is required")
        return ScopeV1(scope_kind="user", scope_id=scope_id("user", user_id))

    def _grant(self, captured: _Captured, scope: ScopeV1) -> Any:
        return captured.auth.resolve(AuthRequest(requested_scope=scope, now_ms=self._clock_ms()))

    # Details ---------------------------------------------------------------------
    def deleting_any(self) -> bool:
        """Whether any user is being deleted (the gateway then reads each Add's user)."""
        with self._lock:
            return bool(self._deleting)

    def deleting(self, user_id: str) -> bool:
        """Whether ``user_id`` is being deleted; its Adds are refused meanwhile."""
        with self._lock:
            return user_id in self._deleting

    def details(self) -> dict[str, object]:
        """The store section of ``/agent/status``; content-free and cached briefly."""
        now = self._clock_ms()
        with self._lock:
            cached = self._details
            if cached is not None and now - cached[0] < DETAILS_TTL_MS:
                return cached[1]
        document = self._admitted(self._read_details)
        with self._lock:
            self._details = (now, document)
            document = {**document, "deleting_users": len(self._deleting)}
        return document

    def _controls(self, captured: _Captured, namespace: Literal["scope", "scope_quarantine"],
                  ) -> Iterator[dict[str, Any]]:
        reader = captured.ready.writer.indexed_reader()
        after: str | None = None
        while True:
            page = reader.scope_controls(namespace, after=after, limit=MAX_PAGE_SIZE)
            yield from page
            if len(page) < MAX_PAGE_SIZE:
                return
            after = cast(str, page[-1]["key"]).split(":", 1)[1]

    def _read_details(self, captured: _Captured) -> dict[str, object]:
        writer = captured.ready.writer
        users: dict[str, str] = {}
        scopes = 0
        for row in self._controls(captured, "scope"):
            scopes += 1
            value = cast(dict[str, Any], row["value"])
            if value.get("scope_kind") == "user":
                users[cast(str, row["key"]).split(":", 1)[1]] = cast(str, value["raw_external_id"])
        quarantined: list[dict[str, object]] = []
        for row in self._controls(captured, "scope_quarantine"):
            scope = cast(str, row["key"]).split(":", 1)[1]
            value = cast(dict[str, Any], row["value"])
            quarantined.append({
                "user_id": users.get(scope), "scope_id": scope, "stage": value.get("stage"),
                "error_code": value.get("error_code"), "since_ms": row.get("updated_at"),
            })
        reader = writer.indexed_reader()
        owners = 0
        after: str | None = None
        while True:
            page = reader.active_finalize_leases(after=after, limit=MAX_PAGE_SIZE)
            owners += len(page)
            if len(page) < MAX_PAGE_SIZE:
                break
            after = cast(str, page[-1]["key"]).removeprefix("lease:finalize:")
        ready = captured.ready
        return {
            "registered_scopes": scopes, "users": len(users),
            "active_add_owners": owners, "quarantined_users": quarantined,
            "embedding": {
                "canary_state": ready.drift_state,
                "mean_cosine_deviation": round(ready.mean_cosine_deviation, 6),
            },
            "measured_at_ms": self._clock_ms(),
        }

    # Listing ---------------------------------------------------------------------
    def list_memories(self, request: MemoryListRequestV1) -> bytes:
        """One page of a user's memories as ``/memory/list`` answers it (``api.memory_list``).

        Raises:
            QuoteMemError: ``service_unhealthy`` while the store is not open or stopping;
                the listing's refusals.
        """
        def read(captured: _Captured) -> bytes:
            return MemoryListing(captured.ready.writer, captured.publisher, captured.auth,
                                 clock=self._clock_ms).encoded(request)

        return self._admitted(read)

    # Repair ----------------------------------------------------------------------
    def repair(self, user_id: str, action: RepairAction) -> dict[str, object]:
        """Inspect, resume or skip one user's stopped Add owner.

        ``resume`` resumes the owner the observed budget names: it lifts a
        quarantine once the fault's cause is cleared, and also renews an exhausted
        or refused allowance. ``skip`` skips a quarantined cycle; its requests
        complete as skipped and nothing committed is lost.

        Raises:
            QuoteMemError: ``operation_conflict`` when there is nothing to resume or skip,
                or the owner moved since it was observed; the core's refusals.
        """
        captured = self._captured
        lifetime = self._lifetime
        if captured is None or lifetime is None:
            raise QuoteMemError("service_unhealthy", "The store is not open")
        service = BenchRecoveryService(captured.controller, lifetime)
        observed = cast(CompletionRecoveryResponseV3, service.inspect_completion_budget(
            CompletionRecoveryRequestV3(schema_version=3, user_id=user_id),
        ))
        scope = self._scope(captured, user_id)
        quarantined = captured.ready.writer.is_scope_quarantined(scope.scope_id)
        budget = observed.budget
        report: dict[str, object] = {
            "user_id": user_id, "quarantined": quarantined, "action": action,
            "failure_cause": observed.failure_cause,
            "budget_state": None if budget is None else budget.state,
            "recovery_count": None if budget is None else budget.recovery_count,
        }
        if action == "inspect":
            return report
        if budget is None or observed.budget_version is None:
            raise QuoteMemError("operation_conflict", "The user has no stopped Add owner to repair")
        if action == "skip":
            if not quarantined:
                raise QuoteMemError("operation_conflict", "Only a quarantined user can be skipped")
            skipped = service.skip_quarantined_cycle(QuarantineSkipRequestV1(
                schema_version=1, user_id=user_id, owner_run_id=budget.owner_run_id,
                budget_version=observed.budget_version, recovery_count=budget.recovery_count,
            ))
            report.update(skipped_requests=skipped.skipped_requests,
                          dropped_records=skipped.dropped_records)
        else:
            if not quarantined and budget.state == "active":
                raise QuoteMemError("operation_conflict", "The user's Add owner is not stopped")
            service.resume_completion_budget(CompletionResumeRequestV3(
                schema_version=3, user_id=user_id, owner_run_id=budget.owner_run_id,
                budget_version=observed.budget_version, recovery_count=budget.recovery_count,
            ))
        report["quarantined_after"] = captured.ready.writer.is_scope_quarantined(scope.scope_id)
        with self._lock:
            self._details = None
        return report

    def auto_repair(self, policy: RepairPolicy) -> list[dict[str, object]]:
        """Apply ``policy`` to every quarantined user once per quarantine receipt.

        ``resume`` resumes each newly quarantined owner once. ``skip`` also
        resumes first, and skips a quarantine that came back for the same owner
        after its resume. Each action is returned for the caller's log line.
        """
        if policy == "off" or not self.open:
            return []
        with self._lock:
            self._details = None
        details = self.details()
        actions: list[dict[str, object]] = []
        for entry in cast(list[dict[str, object]], details["quarantined_users"]):
            user_id = entry.get("user_id")
            if not isinstance(user_id, str):
                continue
            receipt = f"{entry['scope_id']}:{entry.get('since_ms')}"
            previous = self._repaired.get(user_id)
            if previous == receipt:
                continue
            action: RepairAction = "skip" if (
                policy == "skip" and previous is not None
            ) else "resume"
            try:
                result = self.repair(user_id, action)
                outcome: dict[str, object] = {"ok": True, **result}
            except QuoteMemError as error:
                outcome = {"ok": False, "user_id": user_id, "action": action,
                           "code": error.code}
            self._repaired[user_id] = receipt
            actions.append(outcome)
        return actions

    # Forget intents ----------------------------------------------------------------
    def expire_forget_intents(self) -> int:
        """Drop the words and vector of every forget intent that expired unconfirmed, keeping
        only its binding (``ForgetOwner.expire_intents``); how many. Nothing while closed."""
        captured = self._captured
        if captured is None or not self.open:
            return 0
        from quotemem.store.forget_run import ForgetOwner

        return ForgetOwner(captured.ready.writer, captured.publisher,
                           clock=self._clock_ms).expire_intents()

    # Adds and retirement -----------------------------------------------------------
    def completed_adds(
        self, requests: Iterable[tuple[str, str, str]],
    ) -> frozenset[tuple[str, str, str]]:
        """The ``(user_id, session_id, request_id)`` of ``requests`` whose result is stored.

        A point read of each request's completed result (``add_outcome`` Control), as
        an exact resend of the Add reads it: a result the owner completed or an
        operator skipped, under the request's own id or, for an Add a forget
        overtook, a re-admission's (``core/readmission.py``). A user that cannot be
        read (unknown, quarantined, being erased) has none.
        """
        def read(captured: _Captured) -> frozenset[tuple[str, str, str]]:
            writer = captured.ready.writer
            done: set[tuple[str, str, str]] = set()
            for user_id, session_id, request_id in sorted(set(requests)):
                if not user_id or self.deleting(user_id):
                    continue
                scope = self._scope(captured, user_id)
                try:
                    grant = self._grant(captured, scope)
                    with writer.mutex:
                        check_add_grant(grant, scope, self._clock_ms())
                        if read_admitted_add(writer, scope, session_id, request_id):
                            done.add((user_id, session_id, request_id))
                except QuoteMemError:
                    continue
            return frozenset(done)

        return self._admitted(read)

    def erase_epoch(self, user_id: str) -> int | None:
        """The user's erase epoch, which each committed forget advances; None while the
        store is closed or the user has no scope (``service/add_followup.py``)."""
        def read(captured: _Captured) -> int | None:
            writer = captured.ready.writer
            scope = self._scope(captured, user_id)
            if writer.get_control(f"scope:{scope.scope_id}") is None:
                return None
            return writer.get_counter(f"erase_epoch:{scope.scope_id}")

        try:
            return self._admitted(read)
        except QuoteMemError:
            return None

    def retire(self, user_id: str, ranges: Sequence[QuotedRange],
               relation: SupersedeRelation) -> int:
        """End, as ``relation``, the user's active memories that quote mostly ``ranges``.

        Each memory is ended by its own idempotent ``entity_update`` run (request id
        ``retire:<memory id>``), so a repeat finds nothing left to end.

        Returns:
            How many memories this call ended.

        Raises:
            QuoteMemError: The core's refusals (``api.memory_retire``).
        """
        def run(captured: _Captured) -> int:
            scope = self._scope(captured, user_id)
            writer = captured.ready.writer
            retire = MemoryRetire(writer, captured.auth, captured.publisher,
                                  clock=self._clock_ms)
            grant = retire.grant(scope)
            with writer.mutex:
                check_add_grant(grant, scope, self._clock_ms())
                found = memories_quoting(writer, scope, ranges)
            return sum(retire.retire(
                scope=scope, request_id=f"retire:{memory.memory_id}",
                memory_id=memory.memory_id, expected_revision_id=memory.revision_id,
                relation=relation,
            ) for memory in found)

        return self._admitted(run)

    def forget_withheld(self, user_id: str, statement: str) -> int:
        """Tombstone ``statement`` over the user's withheld markers; how many were withheld.

        A rebuild's replay of a forget whose words its journal replaced by the withheld
        marker (``quotemem.agent.rebuild``): every marker an active Evidence part of the user
        holds is withheld by a forget of sentences seeded with ``statement``
        (``StatementForget.forget_sentences``), so the new store holds the old one's
        tombstone and forget shadow, and Search reads a part holding a marker as the old
        store read the part whose words it withheld: as no hit. Markers already withheld
        are withheld again, which changes nothing, so every forgotten statement gets its
        tombstone and a resumed rebuild repeats it harmlessly. Nothing is done, and 0
        returned, when the user's Evidence holds no marker.

        Raises:
            QuoteMemError: ``operation_conflict`` when a forget of sentences did not complete;
                the core's refusals (``store.forget_run``).
        """
        from quotemem.service.forget_statement import StatementForget
        from quotemem.store.forget_run import ForgetOwner

        def run(captured: _Captured) -> int:
            scope = self._scope(captured, user_id)
            spans = self._markers(captured, scope)
            if not spans:
                return 0
            writer = captured.ready.writer
            forget = StatementForget(
                writer, ForgetOwner(writer, captured.publisher, clock=self._clock_ms),
                search=None, provider=None, clock=self._clock_ms, complete=True,
                embedder=captured.ready.providers.embedding,
            )
            chunk: list[QuoteSpanV1] = []
            chunks: list[list[QuoteSpanV1]] = [chunk]
            for span in spans:
                if len(chunk) == MAX_RESTATING_SPANS or (
                        span.evidence_id not in {held.evidence_id for held in chunk}
                        and len({held.evidence_id for held in chunk}) == MAX_TOMBSTONE_SOURCES):
                    chunk = []
                    chunks.append(chunk)
                chunk.append(span)
            for part in chunks:
                outcome = forget.forget_sentences(
                    scope, self._grant(captured, scope), session_id="quotemem-rebuild",
                    statement=statement, spans=part)
                if outcome.outcome != "verified":
                    raise QuoteMemError("operation_conflict",
                                        "A replayed forget did not complete; run again")
            return len(spans)

        return self._admitted(run)

    def forget_earlier(self, user_id: str, statement: str,
                       sources: Set[tuple[str, str]]) -> int:
        """Forget ``statement`` again over the Evidence of ``sources`` alone; how many
        memories went.

        A rebuild's safety pass (``quotemem.agent.rebuild``): ``sources`` are the (session id,
        request id) of the user's Adds the old store took at or before the forget, and only
        a current memory drawn from their Evidence alone is forgotten
        (``StatementForget.forget_earlier``), so whatever the rebuild's chat model decided,
        nothing said before a forget stays current, and a later restatement is untouched.

        Raises:
            QuoteMemError: ``provider_unavailable`` when a pick decision got fewer than two
                answered votes; the core's refusals (``store.forget_run``).
        """
        from quotemem.service.forget_statement import StatementForget
        from quotemem.store.forget_run import ForgetOwner

        def run(captured: _Captured) -> int:
            scope = self._scope(captured, user_id)
            writer = captured.ready.writer
            forget = StatementForget(
                writer, ForgetOwner(writer, captured.publisher, clock=self._clock_ms),
                search=None, provider=captured.ready.providers.inference,
                clock=self._clock_ms, complete=True,
                embedder=captured.ready.providers.embedding,
            )
            return forget.forget_earlier(
                scope, lambda: self._grant(captured, scope), session_id="quotemem-rebuild",
                statement=statement, sources=sources)

        return self._admitted(run)

    def _markers(self, captured: _Captured, scope: ScopeV1) -> list[QuoteSpanV1]:
        """Each withheld marker in the active Evidence of ``scope``, in message bytes."""
        marker = WITHHELD_MARKER.encode("utf-8")
        found: list[QuoteSpanV1] = []
        for row in self._pages(captured, EvidenceRows(scope_kind=scope.scope_kind,
                                                      scope_id=scope.scope_id)):
            if row["state"] != "active" or WITHHELD_MARKER not in row["text"]:
                continue
            data = cast(str, row["text"]).encode("utf-8")
            at = data.find(marker)
            while at >= 0:
                start = cast(int, row["byte_start"]) + at
                found.append(QuoteSpanV1(evidence_id=row["object_id"], byte_start=start,
                                         byte_end=start + len(marker)))
                at = data.find(marker, at + len(marker))
        return sorted(found, key=lambda span: (span.evidence_id, span.byte_start))

    # Deletion --------------------------------------------------------------------
    def delete_user(self, user_id: str) -> dict[str, object]:
        """Erase every memory and source of one user (see the module docstring).

        Raises:
            QuoteMemError: ``operation_conflict`` while the user is quarantined (repair
                or skip first), already being deleted, or still busy after the
                drain timeout; the core's refusals of a forget or erase.
        """
        with self._lock:
            if user_id in self._deleting:
                raise QuoteMemError("operation_conflict", "The user is already being deleted")
            self._deleting.add(user_id)
        try:
            return self._admitted(lambda captured: self._delete(captured, user_id))
        finally:
            with self._lock:
                self._deleting.discard(user_id)
                self._details = None

    def _pages(self, captured: _Captured, query: IndexedQuery) -> Iterator[dict[str, Any]]:
        reader = captured.ready.writer.indexed_reader()
        after: str | None = None
        while True:
            page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
            yield from page
            if len(page) < MAX_PAGE_SIZE:
                return
            after = cast(str, page[-1]["object_id"])

    def _idle(self, captured: _Captured, scope: ScopeV1) -> bool:
        writer = captured.ready.writer
        lease = writer.get_control(f"lease:finalize:{scope.scope_id}")
        state = None if lease is None else cast(dict[str, Any], lease["value"]).get("state")
        if state in {"preparing", "held"}:
            return False
        return not writer.indexed_reader().scope_fences(scope.scope_id)

    def _wait_idle(self, captured: _Captured, scope: ScopeV1) -> None:
        deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
        while not self._idle(captured, scope):
            if time.monotonic() >= deadline:
                raise QuoteMemError("operation_conflict",
                                    "The user's Add or forget work did not finish; try again later")
            self._sleep(0.2)

    def _delete(self, captured: _Captured, user_id: str) -> dict[str, object]:
        writer = captured.ready.writer
        scope = self._scope(captured, user_id)
        report: dict[str, object] = {"user_id": user_id, "scope_id": scope.scope_id}
        if writer.get_control(f"scope:{scope.scope_id}") is None:
            report.update(found=False, forgotten_memories=0, erased_sources=0, remaining={})
            return report
        if writer.is_scope_quarantined(scope.scope_id):
            raise QuoteMemError("operation_conflict",
                           "The user is quarantined; run quotemem repair first")
        self._wait_idle(captured, scope)
        # Consent goes first, before any forget advances the erase epoch: an Add received
        # before the deletion is then refused ``consent_fenced`` for good, never re-admitted
        # as one a forget overtook (``service/add_followup.py``), even when the user has no
        # source left to erase.
        withdraw_consent(writer, scope, now=self._clock_ms())
        forgotten, refused = self._forget_all(captured, user_id, scope)
        self._wait_idle(captured, scope)
        erased = self._erase_all(captured, scope)
        report.update(found=True, forgotten_memories=forgotten, forget_refusals=refused,
                      erased_sources=erased, remaining=self._remaining(captured, scope),
                      log_truncated=self._truncate_log(captured))
        return report

    def _truncate_log(self, captured: _Captured) -> bool:
        """Checkpoint and truncate the write-ahead log; False when readers kept it busy."""
        engine = _engine(captured.ready)
        deadline = time.monotonic() + CHECKPOINT_TIMEOUT_SECONDS
        while True:
            with engine.connect() as connection:
                row = connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if row is not None and row[0] == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            self._sleep(0.2)

    def _forget_all(self, captured: _Captured, user_id: str, scope: ScopeV1) -> tuple[int, int]:
        from quotemem.service.forget import ForgetService

        service = ForgetService(captured.ready, captured.publisher, captured.auth,
                                controller=captured.controller, clock=self._clock_ms)
        writer = captured.ready.writer
        memory_ids = [cast(str, row["object_id"]) for row in self._pages(
            captured, MemoryTableRows(scope_kind="user", scope_id=scope.scope_id))]
        forgotten = refused = 0
        for memory_id in memory_ids:
            item = writer.get_row("item", memory_id)
            if item is None:
                continue  # Deleted with an earlier unit (a correcting memory).
            try:
                prepared = service.prepare_memory_forget(MemoryForgetPrepareRequestV1(
                    schema_version=1, user_id=user_id, client_session_id="agent-delete-user",
                    request_id=f"delete-user-{memory_id}", memory_id=memory_id,
                    expected_revision_id=cast(str, item["current_revision_id"]),
                ))
                service.forget_memory(prepared.confirmation)
                forgotten += 1
            except QuoteMemError as error:
                if error.code == "scope_denied" and writer.get_row("item", memory_id) is None:
                    continue
                refused += 1
            self._wait_idle(captured, scope)
        return forgotten, refused

    def _erase_all(self, captured: _Captured, scope: ScopeV1) -> int:
        writer = captured.ready.writer
        sources = [cast(str, row["object_id"]) for row in self._pages(
            captured, EvidenceRows(scope_kind="user", scope_id=scope.scope_id))]
        active = sorted(identity for identity in sources if (
            (row := writer.get_row("evidence", identity)) is not None and row["state"] == "active"
        ))
        if not active:
            return 0
        cleanup = PrivacyCleanup(writer, captured.publisher, clock=self._clock_ms)
        erased = 0
        for start in range(0, len(active), MAX_ERASE_SOURCES):
            rows = [writer.get_row("evidence", identity)
                    for identity in active[start:start + MAX_ERASE_SOURCES]]
            selected = [row for row in rows if row is not None and row["state"] == "active"]
            if not selected:
                continue
            selection = ConsentWithdrawalInputV1.model_validate({
                "schema_version": 1,
                "affected": [{"evidence_id": row["object_id"],
                              "expected_row_digest": writer.bind_content("row-state", row)}
                             for row in selected],
                "expected_scope_seq": writer.get_counter(f"scope_seq:{scope.scope_id}"),
                "erase_epoch": writer.get_counter(f"erase_epoch:{scope.scope_id}"),
                "consent_epoch": writer.get_counter(f"consent_epoch:{scope.scope_id}"),
                "retention_role": "consent_withdrawal", "reason_code": "consent_withdrawn",
                "trigger": "consent_epoch",
            })
            cleanup.execute(scope, selection, self._grant(captured, scope))
            erased += len(selected)
            self._wait_idle(captured, scope)
        return erased

    def _remaining(self, captured: _Captured, scope: ScopeV1) -> dict[str, int]:
        writer = captured.ready.writer
        items = list(self._pages(
            captured, MemoryTableRows(scope_kind="user", scope_id=scope.scope_id)))
        sources = [row for row in self._pages(
            captured, EvidenceRows(scope_kind="user", scope_id=scope.scope_id))
            if (stored := writer.get_row("evidence", cast(str, row["object_id"]))) is not None
            and stored["state"] == "active"]
        tombstones = list(self._pages(captured, TombstoneRows(
            scope_kind="user", scope_id=scope.scope_id, after_tombstone_seq=0)))
        candidates = list(self._pages(captured, CandidateRows(
            scope_kind="user", scope_id=scope.scope_id, status="pending")))
        return {
            "memories": len(items),
            "readable_memories": sum(row.get("lifecycle_state") != "erased" for row in items),
            "active_sources": len(sources), "tombstones": len(tombstones),
            "pending_candidates": len(candidates),
        }
