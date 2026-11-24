"""Ordinary request-bound Add orchestration using QuoteMem's one Writer.

Provider calls and ledger-chain reads happen outside the commit mutex. A
successful result additionally requires publication of the matching generation.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Protocol, cast

from quotemem.api.bench import AddDispatch
from quotemem.core.add_outcome import AddOutcomeV1
from quotemem.core.admission import split_secret_parent_binding
from quotemem.core.artifacts import EmbeddingCanaryConfigV1, RequestLedgerV1
from quotemem.core.controls import (
    CompletedPrefixV2,
    ControlRowV1,
    RequestLedgerValueV1,
    ScopeRegistrationV1,
)
from quotemem.core.embedding import embedding_state_token
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import control_id, new_ulid, scope_id, vector_encodings
from quotemem.core.inference import DEFAULT_PROVIDER_TIMEOUT_MS, default_retry_policy
from quotemem.core.privacy_persistence import SPLIT_SECRET_INPUT, SplitSecretInputV1
from quotemem.core.query import ScopeV1
from quotemem.core.request_budget import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    check_request_budget,
    current_budget,
)
from quotemem.core.run_inputs import SessionDurableRunInputV1
from quotemem.core.semantic import SecurityLevel
from quotemem.core.technical_types import RequestProgressV1
from quotemem.core.timing import measure
from quotemem.ingestion import add_admission
from quotemem.ingestion.authorization import check_add_grant
from quotemem.ingestion.boundary import (
    AddAdmissionBinding,
    AddRequestV1,
    PriorEvidenceSpan,
    Profile,
    SecretPreflightResult,
    build_add_admission,
    parse_add_request,
    secret_preflight,
    split_evidence,
)
from quotemem.ingestion.privacy_cleanup import PrivacyCleanup
from quotemem.ingestion.receipts import LedgerHead, inspect_ledger
from quotemem.ingestion.recovery import verify_durable_replay
from quotemem.ingestion.source_security import classify_source_security
from quotemem.providers.auth import AuthProvider, AuthRequest, PrincipalGrant
from quotemem.providers.contracts import EmbeddingProvider, EmbeddingRequest, EmbeddingResult
from quotemem.store.domain import EvidenceRowV1, SessionTargetRunRowV1
from quotemem.store.indexed import IndexedReader
from quotemem.store.read_model import StoreReadModel
from quotemem.store.request_completion import effective_receipt_deadline
from quotemem.store.writer import RowWrite, Writer


@dataclass(frozen=True)
class _CleanupSelection:
    """Leave both Add locks without retaining old source text or an exception traceback."""

    scope: ScopeV1
    supplied: SplitSecretInputV1
    parent_run_id: str | None


@dataclass(frozen=True)
class AddReceipt:
    """Expose no source text or provider output in the SDK reception result."""

    run_id: str
    visibility_commit_seq: int
    message_count: int
    reused: bool
    work_deadline_at: int | None = None
    outcome: AddOutcomeV1 | None = None


@dataclass(frozen=True)
class _PreparedAdd:
    run_id: str


@dataclass(frozen=True)
class _ReceptionContext:
    queue: add_admission.AddAdmissionQueue
    ticket: add_admission.AddReceptionTicket
    dispatch: _AddDispatch


_RECEPTION: ContextVar[_ReceptionContext | None] = ContextVar("aml_add_reception", default=None)


def current_completion_wait_deadline(scope: str) -> float | None:
    """Read this attempt's fixed completion wait bound across joined worker phases."""
    context = _RECEPTION.get()
    if context is None or context.ticket.scope_id != scope:
        return None
    return context.dispatch.completion_deadline


def reconcile_add_admission(writer: Writer, scope: str) -> None:
    """Retire only process-local scheduling whose durable admission epochs are obsolete.

    The Writer proves the current epochs and deletes rejected raw input under
    its mutex. Ordered receipts remain immutable. An actual preparing or held
    owner is never released here. Definitive cancellation or completed-release
    proof retires only its process-local hint, including a lost acknowledgement.
    """
    queue = writer.add_admission
    pending, committed = queue.reconciliation_runs(scope)
    if not pending and not committed:
        return
    with writer.mutex:
        writer.validate_storage_owner()
        epochs = writer.get_counter(f"erase_epoch:{scope}"), writer.get_counter(
            f"consent_epoch:{scope}")

        def obsolete(run_id: str) -> tuple[dict[str, Any], bool]:
            run = writer.get_row("run", run_id)
            if run is None:
                raise QuoteMemError("binding_mismatch", "Reception scheduling lost its Run proof")
            if (run["scope_kind"], run["scope_id"], run["run_kind"]) != (
                "user", scope, "session_ingestion",
            ):
                raise QuoteMemError("binding_mismatch",
                                    "Reception scheduling changed its Run scope")
            return run, (run["admitted_erase_epoch"], run["admitted_consent_epoch"]) != epochs

        for run_id in pending:
            run, rejected = obsolete(run_id)
            if rejected and run["input_ref"] is None:
                writer.discard_rejected_add_input(run_id)
                # As `adopt_add_run`'s caller names it: a deletion withdraws consent
                # before it forgets, so a consent change is final; a forget alone is
                # an overtaken Add, which the agent service admits again.
                queue.retire_rejected(scope, run_id, "consent_fenced" if (
                    run["admitted_consent_epoch"] != epochs[1]) else "erase_fenced")
        if not committed:
            return
        lease = writer.get_control(f"lease:finalize:{scope}")
        if lease is not None and lease["value"]["state"] in {"preparing", "held"}:
            return
        if all(obsolete(run_id)[1] for run_id in committed):
            queue.retire_owner_reservation(scope, committed)
            return
        if lease is None or lease["value"]["state"] != "released":
            return
        raw_prefix = writer.get_control(f"completed_prefix:{scope}")
        prefix = None if raw_prefix is None else ControlRowV1.model_validate(raw_prefix).value
        if not isinstance(prefix, CompletedPrefixV2) or (
            prefix.lease.model_dump(mode="json") != lease["value"]
            or (prefix.erase_epoch, prefix.consent_epoch) != epochs
        ):
            return
        owner = writer.get_row("run", prefix.lease.owner_run_id)
        if owner is None or owner["status"] != "succeeded":
            return
        for run_id in committed:
            receipt = writer.read_run_input(run_id)
            if not isinstance(receipt, SessionDurableRunInputV1) or (
                receipt.visibility_commit_seq > prefix.accepted_prefix_commit_seq
            ):
                return
        queue.owner_released(scope)


class _AddDispatch:
    """Keep one admitted attempt across short preparation and ordering invocations."""

    def __init__(
        self, service: AddService, request: AddRequestV1, ticket: add_admission.AddReceptionTicket,
    ) -> None:
        self.service = service
        self.request: AddRequestV1 | None = request
        self.ticket = ticket
        self.security: Callable[[], SecurityLevel] | None = functools.cache(
            partial(service._security, request),  # pyright: ignore[reportPrivateUsage]
        )
        budget = current_budget()
        self.admission_deadline = (
            budget.deadline if service.requires_completion and budget is not None
            else time.monotonic() + (
                DEFAULT_REQUEST_TIMEOUT_SECONDS if service.requires_completion
                else DEFAULT_REQUEST_TIMEOUT_SECONDS
            )
        )
        self.completion_deadline: float | None = (
            self.admission_deadline if service.requires_completion else None
        )
        self.receipt: AddReceipt | None = None
        self._run_id: str | None = None
        self._generation = service.writer.add_admission.generation

    def step(self) -> bool:
        """Save first, then order only when this input owns the scope's next turn."""
        try:
            with vector_encodings(), (
                nullcontext() if self._run_id is None
                else self.service._discard_on_rejection(  # pyright: ignore[reportPrivateUsage]
                    self._run_id,
                )
            ):
                return self._step()
        except QuoteMemError as error:
            if self._run_id is not None:
                reconcile_add_admission(self.service.writer, self.ticket.scope_id)
            if error.code != "scope_quarantined":
                raise
            raise QuoteMemError("repair_required",
                                "The Add scope requires explicit repair") from None

    def _step(self) -> bool:
        request = self.request
        if request is None:
            raise RuntimeError("The Add reception context already ended")
        check_request_budget()
        queue = self.service.writer.add_admission
        self._generation = queue.generation
        if self.receipt is not None:
            return True
        if time.monotonic() >= self.admission_deadline:
            raise QuoteMemError("delivery_busy", "The scope's admission queue is busy; retry")
        if self._run_id is None:
            result = self.service._add(  # pyright: ignore[reportPrivateUsage]
                request, prepare_only=True, block_admission=False,
                admission_deadline=self.admission_deadline,
            )
            if isinstance(result, _PreparedAdd):
                self._run_id = result.run_id
                return False
        else:
            try:
                ready = self.service._wait_admission(  # pyright: ignore[reportPrivateUsage]
                    request, self._run_id, self.admission_deadline, wait=False,
                )
            except QuoteMemError as error:
                if error.code != "reception_stale":
                    raise
                # A Writer-proved concurrent rejection removed this source.
                # Re-entry proves authority and receives that same Run again.
                self._run_id = None
                return False
            if not ready:
                return False
            result = self.service._add(  # pyright: ignore[reportPrivateUsage]
                request, block_admission=False, admission_deadline=self.admission_deadline,
            )
        if isinstance(result, AddReceipt):
            self.receipt = result
            return True
        return False

    @property
    def accepted(self) -> bool:
        """Whether this request's source is durably saved: prepared, ordered or completed.

        A saved request no longer needs its receiver: the scheduler orders and completes
        it (``ingestion/add_scheduler.py``), and recovery resumes it after a restart.
        """
        return self.receipt is not None or self._run_id is not None

    def completed(self) -> bool:
        """Point-read whether this ordered request's result is visible, without waiting.

        Returns:
            bool: True once the owner that covers the receipt has succeeded; False while
                the request is only saved or its owner still runs.
        """
        check_request_budget()
        receipt = self.receipt
        if receipt is None:
            return False
        from quotemem.store.add_outcome import read_add_outcome

        scope = ScopeV1(scope_kind="user", scope_id=self.ticket.scope_id)
        grant = self.service._grant(scope)  # pyright: ignore[reportPrivateUsage]
        try:
            return read_add_outcome(
                self.service.writer, scope, receipt, grant=grant, clock=self.service.clock,
            ) is not None
        except QuoteMemError as error:
            if error.code != "scope_quarantined":
                raise
            raise QuoteMemError("repair_required",
                                "The Add scope requires explicit repair") from None

    def completion_ready(self) -> bool:
        """Never occupy a worker while another selected source still needs ordering."""
        try:
            return self._completion_ready()
        except QuoteMemError as error:
            if error.code != "scope_quarantined":
                raise
            raise QuoteMemError("repair_required",
                                "The Add scope requires explicit repair") from None

    def _completion_ready(self) -> bool:
        check_request_budget()
        queue = self.service.writer.add_admission
        self._generation = queue.generation
        if self.receipt is None:
            return False
        lease = self.service.writer.get_control(f"lease:finalize:{self.ticket.scope_id}")
        if lease is not None and lease["value"]["state"] in {"preparing", "held"}:
            return True
        try:
            queue.require_finalize_allowed(self.ticket.scope_id)
        except QuoteMemError as error:
            if error.code != "finalize_pending":
                raise
            # A released owner's result may already cover this receipt while
            # newer traffic prepares its next input. That traffic cannot make
            # an already completed request wait behind itself.
            from quotemem.store.add_outcome import read_add_outcome

            request = self.request
            if request is None:
                raise RuntimeError("The Add reception context already ended") from None
            scope = ScopeV1(scope_kind="user", scope_id=self.ticket.scope_id)
            grant = self.service._grant(scope)  # pyright: ignore[reportPrivateUsage]
            completed = read_add_outcome(
                self.service.writer, scope, self.receipt, grant=grant,
                clock=self.service.clock,
            )
            if completed is not None:
                return True
            if self.completion_deadline is not None and (
                time.monotonic() >= self.completion_deadline
            ):
                raise QuoteMemError(
                    "delivery_busy", "The scope's completion owner is still busy; retry",
                ) from None
            return False
        return True

    async def wait_for_change(self) -> None:
        """Keep the receiver's budget and cancellation while returning its worker slot."""
        deadline = self.admission_deadline if self.receipt is None else (
            self.completion_deadline or self.admission_deadline
        )
        budget = current_budget()
        if budget is not None:
            deadline = min(deadline, budget.deadline)
        queue = self.service.writer.add_admission
        with measure("queue"), (
            nullcontext() if budget is None else budget.on_cancel(queue.wake)
        ):
            await queue.wait_for_notification(
                self._generation, deadline=deadline, check=check_request_budget,
            )


def _clock_ms() -> int:
    return time.time_ns() // 1_000_000


FINALIZE_FENCE_POLL_SECONDS = 0.025
"""Sleep between lock-free checks of a held finalize lease row."""

FINALIZE_FENCE_RECHECK_SECONDS = 1.0
"""Longest wait before an Add re-runs its full admission while a fence is held."""

_REJECTED_RECEPTION = frozenset(
    {"scope_denied", "erase_fenced", "consent_fenced", "secret_rejected"}
)
"""Refusals that end a received request before ordering; its saved source is deleted."""


class RowLedger(Protocol):
    """The Writer's lock-free proof that named rows were not rewritten after a token."""

    def is_borrow_unchanged(self, keys: Iterable[tuple[str, str]], token: int) -> bool: ...


def await_fence_release(
    writer: RowLedger, key: tuple[str, str], token: int, until: float, *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait, without the Writer, until ``key`` is rewritten after ``token`` or ``until``.

    ``is_borrow_unchanged`` proves from the in-process retirement ledger
    that no mutation named the finalize lease row since the admission
    attempt read it; it takes no lock and reads no storage. The
    caller re-runs the complete admission afterwards, which is what decides
    reception; this wait only chooses when to try again.

    Args:
        writer: The sole Writer whose ledger orders row mutations.
        key: The finalize lease Control row of the request's scope.
        token: ``storage_epoch`` captured before the attempt read that row.
        until: Absolute ``time.monotonic`` bound for this wait.
        sleep: Injectable sleep for tests.
    """
    while writer.is_borrow_unchanged((key,), token):
        remaining = until - time.monotonic()
        if remaining <= 0:
            return
        sleep(min(FINALIZE_FENCE_POLL_SECONDS, remaining))


@contextmanager
def _session_admission(lock: threading.RLock | threading.Lock) -> Iterator[None]:
    """Observe actual contended admission without including the protected work."""
    if not lock.acquire(blocking=False):
        with measure("queue"):
            budget = current_budget()
            if budget is None:
                lock.acquire()
            else:
                while not lock.acquire(timeout=min(0.05, budget.remaining())):
                    budget.check()
    try:
        yield
    finally:
        lock.release()


class AddService:
    """Coordinate reception and leave semantic extraction to its durable scheduler.

    Args:
        writer: The initialized exclusive owner's sole persistence authority.
        embedding: Actual provider boundary, frozen in ordinary tests.
        auth: Principal resolver called outside both QuoteMem locks.
        publisher: Complete generation publication authority.
        profile: Trusted deployment profile.
        clock: Trusted wall clock in integer milliseconds.
        checkpoint: Optional observer for crash tests, never an external provider.
    """

    requires_completion = False
    """The serving adapter retains each ordered Add's next-owner turn."""

    def __init__(
        self,
        writer: Writer,
        embedding: EmbeddingProvider,
        auth: AuthProvider,
        publisher: StoreReadModel,
        *,
        profile: Profile,
        clock: Callable[[], int] = _clock_ms,
        checkpoint: Callable[[str], None] | None = None,
    ) -> None:
        self.writer, self.embedding, self.auth, self.publisher = writer, embedding, auth, publisher
        self.profile: Profile = profile
        self.clock, self.checkpoint = clock, checkpoint
        self.reader = IndexedReader(
            writer.runtime.workspace,
            writer.entities,
            tenant_id=writer.tenant_id,
        )
        self.cleanup = PrivacyCleanup(writer, publisher, clock=clock, checkpoint=checkpoint)

    def _grant(self, scope: ScopeV1, principal_id: str | None = None) -> PrincipalGrant:
        check_request_budget()
        now = self.clock()
        grant = self.auth.resolve(AuthRequest(requested_scope=scope, now_ms=now))
        check_add_grant(grant, scope, self.clock(), principal_id)
        check_request_budget()
        return grant

    def _check_finalize_pending(self, request: AddRequestV1) -> None:
        """Let a serving adapter report a settled scope refusal before fence waiting."""
        return None

    def _scope_gate(
        self, scope: ScopeV1, raw_external_id: str, *, allow_pending_privacy: bool = False
    ) -> None:
        writer = self.writer
        if writer.get_control(f"tomb:{scope.scope_id}") is not None:
            raise QuoteMemError("scope_denied", "The requested scope was removed")
        pending, fences = writer.admission_reads_hint(scope.scope_id)
        if not allow_pending_privacy and pending:
            raise QuoteMemError("index_rebuilding",
                                "Pending privacy cleanup blocks ordinary reception")
        for row in fences:
            if row["key"].startswith(f"erasure:{scope.scope_id}:"):
                raise QuoteMemError("erasure_in_progress", "The requested scope is being erased")
            if row["key"].startswith(f"maintenance:{scope.scope_id}:"):
                raise QuoteMemError("index_rebuilding", "The requested scope is under maintenance")
        registration = writer.get_control(f"scope:{scope.scope_id}")
        counters = [
            writer.get_control(f"{key}:{scope.scope_id}")
            for key in ("scope_seq", "erase_epoch", "consent_epoch")
        ]
        if registration is None:
            if any(row is not None for row in counters) or self.reader.has_scope_rows(
                scope.scope_kind, scope.scope_id
            ):
                raise QuoteMemError("service_unhealthy",
                                    "Existing scope state lost its registration")
        elif registration["value"] != {
            "scope_kind": scope.scope_kind,
            "raw_external_id": raw_external_id,
        } or any(row is None for row in counters):
            raise QuoteMemError("service_unhealthy", "Scope registration or counters fail parity")

    def _scope_registered(self, scope: ScopeV1, raw_external_id: str) -> bool:
        """Report whether the scope's Controls and its published fence already exist.

        Every Add of a registered user answers this from a borrowed read of the
        ``scope:`` Control and the publication root, so only a user's first
        Add enters the registering critical section. The
        result authorizes nothing: the later Run admission re-proves the
        durable registration under the Writer mutex before any write.

        Args:
            scope: Canonical exact scope of the request.
            raw_external_id: Original external identity the scope hashes.

        Returns:
            ``True`` when the registration Control carries this exact source
            identity and the current root already publishes the scope's fence.

        Raises:
            QuoteMemError: If the publisher is unhealthy or its published scope kind
                differs, exactly as the registering path would raise.
        """
        self.writer.probe_storage_owner()
        registration = self.writer.get_control(f"scope:{scope.scope_id}")
        if registration is None or registration["value"] != {
            "scope_kind": scope.scope_kind,
            "raw_external_id": raw_external_id,
        }:
            return False
        return self.publisher.is_scope_published(scope)

    def _ledger(
        self, scope: ScopeV1, session_hash: str, binding: AddAdmissionBinding
    ) -> tuple[ControlRowV1 | None, RequestProgressV1 | None, LedgerHead]:
        try:
            raw = self.writer.get_control(f"request_ledger:{scope.scope_id}:{session_hash}")
            control = None if raw is None else ControlRowV1.model_validate(raw)
            value = RequestLedgerValueV1.model_validate(
                {"recent": {}} if control is None else control.value.model_dump(mode="json")
            )
            hit, head = inspect_ledger(
                value,
                binding.request_key,
                lambda ref: cast(RequestLedgerV1, self.writer.read_logical(ref, "request_ledger")),
                self.writer.bind_content,
            )
        except Exception:
            # Corrupt or unavailable persistence is neither a request conflict
            # nor a client-owned missing reference. No partial ledger hit may
            # bypass the complete proof or expose decoded storage diagnostics.
            raise QuoteMemError("service_unhealthy",
                                "Reception ledger proof is unavailable") from None
        return control, hit, head

    def _durable_replay(
        self, binding: AddAdmissionBinding, ledger_hit: RequestProgressV1
    ) -> tuple[str, SessionDurableRunInputV1]:
        proof = verify_durable_replay(self.writer, binding=binding, ledger_binding=ledger_hit)
        return proof.latest_run.object_id, proof.witness

    def _replay_receipt(
        self,
        request: AddRequestV1,
        scope: ScopeV1,
        binding: AddAdmissionBinding,
        hit: RequestProgressV1,
        grant: PrincipalGrant,
    ) -> AddReceipt:
        """Recheck action-time authority after the complete durable witness proof."""
        try:
            run_id, witness = self._durable_replay(binding, hit)
        except QuoteMemError as error:
            if error.code == "operation_conflict":
                check_add_grant(grant, scope, self.clock(), binding.admission.principal_id)
                self._scope_gate(scope, request.user_id)
            raise
        check_add_grant(grant, scope, self.clock(), binding.admission.principal_id)
        self._scope_gate(scope, request.user_id)
        if self.requires_completion:
            self._check_finalize_pending(request)
        self.publisher.check_reception(self.writer, witness)
        self.writer.discard_received_add_input(run_id)
        ticket = self._reception_ticket(request)
        if ticket is not None:
            self.writer.add_admission.replayed(ticket, run_id)
        return AddReceipt(
            run_id, witness.visibility_commit_seq, witness.message_count, True,
            effective_receipt_deadline(witness),
        )

    def _tail(self, scope: ScopeV1, session_id: str) -> list[PriorEvidenceSpan]:
        rows = self.reader.active_session_tail(
            scope.scope_kind, scope.scope_id, session_id, now_ms=self.clock()
        )
        return [
            PriorEvidenceSpan(
                row["object_id"],
                row["scope_id"],
                row["session_id"],
                row["text"],
                row["byte_start"],
                row["byte_end"],
            )
            for row in rows
        ]

    @contextmanager
    def _discard_on_rejection(self, run_id: str) -> Iterator[None]:
        """Delete the saved source of a reception that this call rejects for good.

        Authority, scope-epoch and secret rejections end the request before it is
        ordered, so its source bytes must not outlive them. A deferral (vector
        failure, held fence, changed embedding state) keeps the saved input, and
        the same request's resend or background worker continues from it.
        """
        try:
            yield
        except QuoteMemError as error:
            if error.code in _REJECTED_RECEPTION:
                self.writer.discard_rejected_add_input(run_id)
            raise

    def _cleanup_selection(
        self, scope: ScopeV1, secret: SecretPreflightResult, parent_run_id: str | None = None
    ) -> _CleanupSelection:
        selection = self._select_cleanup(scope, secret, parent_run_id)
        if parent_run_id is not None:
            # The parent is rejected whatever the cleanup's outcome, and its saved
            # source holds part of the recognized secret: delete it before the
            # cleanup child exists, whose rejection proof requires its absence.
            self.writer.discard_rejected_add_input(parent_run_id)
        return selection

    def _select_cleanup(
        self, scope: ScopeV1, secret: SecretPreflightResult, parent_run_id: str | None
    ) -> _CleanupSelection:
        with self.writer.mutex:
            affected: list[dict[str, object]] = []
            for object_id in sorted({span.evidence_id for span in secret.affected_old_spans}):
                row = self.writer.get_row("evidence", object_id)
                if (
                    row is None
                    or row["state"] != "active"
                    or (row["scope_kind"], row["scope_id"]) != (scope.scope_kind, scope.scope_id)
                ):
                    raise QuoteMemError("reception_stale",
                                        "Recognized source changed before selection")
                affected.append(
                    {
                        "evidence_id": object_id,
                        "expected_row_digest": self.writer.bind_content("row-state", row),
                    }
                )
            value: dict[str, object] = {
                "schema_version": 1,
                "retention_role": "split_secret",
                "reason_code": "retroactive_split_secret",
                "trigger": "pre_add" if parent_run_id is None else "ingestion_child",
                "affected": affected,
                "expected_scope_seq": self.writer.get_counter(f"scope_seq:{scope.scope_id}"),
                "erase_epoch": self.writer.get_counter(f"erase_epoch:{scope.scope_id}"),
                "consent_epoch": self.writer.get_counter(f"consent_epoch:{scope.scope_id}"),
            }
            if parent_run_id is not None:
                parent = self.writer.get_row("run", parent_run_id)
                if parent is None:
                    raise QuoteMemError("binding_mismatch", "Admitted reception lost its owner")
                value["parent_binding_digest"] = split_secret_parent_binding(
                    parent, self.writer.bind_content
                )
            return _CleanupSelection(
                scope, SPLIT_SECRET_INPUT.validate_python(value), parent_run_id
            )

    def _security(self, request: AddRequestV1) -> SecurityLevel:
        """The request's source security, as the Writer re-derives it to check the rows.

        Bench stores source credentials by explicit profile exception but still
        classifies recognized sensitive source bytes as restricted.
        """
        return classify_source_security(
            "".join(message.content for message in request.messages), profile=self.profile
        ).security_level

    def _evidence(
        self, request: AddRequestV1, vectors: Sequence[Sequence[float]], embedding_space: str,
        security: SecurityLevel,
    ) -> list[EvidenceRowV1]:
        import hashlib

        parts = split_evidence(request)
        if len(parts) != len(vectors):
            raise QuoteMemError("provider_error", "Embedding output count differs from the request")
        return [
            EvidenceRowV1.model_validate(
                {
                    "object_id": new_ulid(),
                    "tenant_id": self.writer.tenant_id,
                    "scope_kind": "user",
                    "scope_id": scope_id("user", request.user_id),
                    "kind": {
                        "user": "user_message",
                        "assistant": "assistant_message",
                        "system": "system_message",
                    }[part.role],
                    "session_id": request.session_id,
                    "request_id": request.request_id,
                    "ordinal": part.ordinal,
                    "part_no": part.part_no,
                    "role": part.role,
                    "speaker_entity": part.speaker,
                    **({"source_participant": part.source_participant.model_dump(mode="json")}
                       if part.source_participant is not None else {}),
                    # The SDK's anchors inside this part; null when it has none.
                    "entity_bindings": [
                        binding.model_dump(mode="json") for binding in part.entity_bindings
                    ] or None,
                    "text": part.text,
                    "byte_start": part.byte_start,
                    "byte_end": part.byte_end,
                    "source_observed_at": part.source_observed_at,
                    "ingested_at": 1,
                    "ingested_commit_seq": 1,
                    "time_granularity": "unanchored" if part.source_observed_at is None else "HOUR",
                    "trust_tier": {
                        "user": "reported",
                        "assistant": "inferred",
                        "system": "untrusted",
                    }[part.role],
                    "security_level": security,
                    "state": "active",
                    "route_state": "indexed" if self.profile == "bench" else "pending",
                    "content_hash": hashlib.sha256(part.text.encode()).hexdigest(),
                    "embedding": list(vector),
                    "embedding_space": embedding_space,
                }
            )
            for part, vector in zip(parts, vectors, strict=True)
        ]

    def _require_unexpired(self, run_id: str) -> None:
        """Expire only an unsaved root; accepted input survives receiver deadlines."""
        if self.writer.abandon_add_run(run_id, self.clock()):
            raise QuoteMemError(
                "operation_conflict",
                "Accepted Add expired before ordering; a new request identity is required",
            )

    def _source_vectors(
        self, texts: list[str], model: str,
    ) -> dict[str, Sequence[float]]:
        """Embed each unique source text once; any failure leaves the input saved, unordered.

        The request stays received with its original identity, so resending the
        same request continues it. Transient failures are retryable
        ``provider_unavailable``; a deterministic provider input refusal keeps
        its non-retryable ``invalid_request``. Actual provider failures retain
        their typed errors so transport classifications survive.
        """
        embedding_request = EmbeddingRequest(
            dimensions=cast(Any, self.writer.dimension),
            texts=texts,
            timeout_ms=DEFAULT_PROVIDER_TIMEOUT_MS,
            retry_policy=default_retry_policy(),
        )
        try:
            with measure("embedding"):
                check_request_budget()
                result = self.embedding.embed(embedding_request)
                check_request_budget()
            result = EmbeddingResult.model_validate(result.model_dump(mode="json"))
            if len(result.vectors) != len(texts):
                raise QuoteMemError("provider_error", "Embedding result lost a unique source input")
            if result.dimensions != self.writer.dimension:
                raise QuoteMemError("provider_error",
                                    "Embedding dimensions differ from the workspace")
            if result.resolved_model != model:
                raise QuoteMemError(
                    "provider_error", "Embedding model differs from the frozen workspace baseline"
                )
        except QuoteMemError as error:
            if error.code not in {"provider_error", "schema_invalid"}:
                raise
            raise QuoteMemError(
                "provider_unavailable", "Source vectors failed before ordering",
            ) from None
        finally:
            del embedding_request
        return dict(zip(texts, result.vectors, strict=True))

    def add(self, raw: object) -> AddReceipt:
        """Accept one complete boundary, including idempotent durable replay.

        Reception first saves the source and request identity, then embeds and
        orders the source-ordered queue head. While a finalize owner of
        the scope is preparing or held, each saved Add retains its queue place;
        an owner that needs explicit repair refuses the wait at once. A deferred
        request keeps its saved source for the same request's resend; an
        authority, scope-epoch or secret rejection deletes it. Receiver expiry
        leaves accepted work available to background processing and exact retry.

        Args:
            raw: Strict request DTO or decoded object with the four Add fields.

        Returns:
            The durable receipt after its complete root is published.

        Raises:
            QuoteMemError: For invalid input, authorization, secrets, conflicts, or unhealthy
                storage; ``provider_unavailable`` when source vectors fail before
                ordering (the same request resends and continues); ``operation_conflict``
                when an unsaved root expired or the reception failed (a new
                request identity is required); ``repair_required`` for a
                quarantined scope or a fence owner that needs explicit repair;
                ``delivery_busy`` when bounded admission defers saved input.
        """
        try:
            # Reception encodes each new row's vector many times (``vector_encodings``).
            with vector_encodings():
                result = self._add(parse_add_request(raw))
                assert isinstance(result, AddReceipt)
                return result
        except QuoteMemError as error:
            if error.code != "scope_quarantined":
                raise
            raise QuoteMemError("repair_required",
                                "The Add scope requires explicit repair") from None

    def _reception_ticket(self, request: AddRequestV1) -> add_admission.AddReceptionTicket | None:
        context = _RECEPTION.get()
        if context is None or context.queue is not self.writer.add_admission:
            return None
        ticket = context.ticket
        if (ticket.scope_id, ticket.identity) != (
            scope_id("user", request.user_id), (request.session_id, request.request_id),
        ):
            return None
        return ticket

    def resume_received(self, run_id: str) -> AddReceipt | None:
        """Order one saved queue head through ordinary Add authority, without a client.

        Args:
            run_id: Original accepted Add Run selected by the bounded scheduler.

        Returns:
            Its proven receipt, or None if another invocation completed it or
            the current source-order head changed before ordering.

        Raises:
            QuoteMemError: When saved identity, current authority or source binding
                cannot be proved. No replacement request or Run is constructed.
        """
        saved = self.writer.read_received_add_input(run_id)
        if saved is None:
            return None
        # A save acknowledgement can be lost after durable bookkeeping revoked
        # the read root. Rebuild through the existing complete catalog/privacy
        # publication proof before resuming this independently proved input.
        try:
            _ = self.publisher.current
        except QuoteMemError as error:
            if error.code != "service_unhealthy":
                raise
            self.writer.recover_catalog()
            self.publisher.recover_privacy(self.writer)
        run = self.writer.get_row("run", run_id)
        if run is None:
            raise QuoteMemError("binding_mismatch", "Saved Add lost its original Run")
        raw = self.writer.get_control(f"scope:{run['scope_id']}")
        registration = None if raw is None else ControlRowV1.model_validate(raw).value
        if not isinstance(registration, ScopeRegistrationV1) or (
            registration.scope_kind != "user" or saved.request_payload.profile != self.profile
            or scope_id("user", registration.raw_external_id) != run["scope_id"]
        ):
            raise QuoteMemError("binding_mismatch", "Saved Add lost its external scope binding")
        request = parse_add_request({
            "user_id": registration.raw_external_id,
            **saved.request_payload.model_dump(mode="json", exclude={"profile"}),
        })
        with vector_encodings():
            result = self._add(request, block_admission=False)
        if isinstance(result, AddReceipt) and result.run_id != run_id:
            raise QuoteMemError("binding_mismatch", "Saved Add resumed another original request")
        return result if isinstance(result, AddReceipt) else None

    @contextmanager
    def add_reception(self, request: AddRequestV1) -> Iterator[AddDispatch]:
        """Reserve a parsed Add's reception before preparation or worker dispatch.

        The HTTP owner enters this context before submitting its joined worker;
        context propagation gives the worker the same attempt token. SDK calls
        enter it themselves. No source is accepted and no Writer authority is
        issued by this content-free reservation. A saved exact retry retains
        its original Run coordinate; separate attempts withdraw independently.

        Args:
            request: The validated request with its original logical identity.

        Yields:
            Cooperative preparation/ordering control for this same attempt.
        """
        if self._reception_ticket(request) is not None:
            context = _RECEPTION.get()
            assert context is not None
            yield context.dispatch
            return
        times = [message.timestamp for message in request.messages]
        source_time = min((value for value in times if value is not None), default=None)
        queue = self.writer.add_admission
        budget = current_budget()
        received_at = self.clock() if budget is None else budget.received_at(self.clock())
        ticket = queue.reserve(
            scope_id("user", request.user_id), (request.session_id, request.request_id),
            received_at, source_time=source_time if all(value is not None for value in times)
            else None,
        )
        dispatch = _AddDispatch(self, request, ticket)
        token = _RECEPTION.set(_ReceptionContext(queue, ticket, dispatch))
        try:
            yield dispatch
        finally:
            _RECEPTION.reset(token)
            queue.finish(ticket)
            dispatch.request = None
            dispatch.security = None

    def _add(
        self, request: AddRequestV1, *, prepare_only: bool = False,
        block_admission: bool = True, admission_deadline: float | None = None,
    ) -> AddReceipt | _PreparedAdd | None:
        with self.add_reception(request):
            return self._add_received(
                request, prepare_only=prepare_only, block_admission=block_admission,
                admission_deadline=admission_deadline,
            )

    def _add_received(
        self, request: AddRequestV1, *, prepare_only: bool = False,
        block_admission: bool = True, admission_deadline: float | None = None,
    ) -> AddReceipt | _PreparedAdd | None:
        registered: set[str] = set()
        scope = scope_id("user", request.user_id)
        ticket = self._reception_ticket(request)
        attempt = object() if ticket is None else ticket.attempt

        def received(run_id: str) -> None:
            registered.add(run_id)
            self.writer.add_admission.activate(scope, run_id, attempt)

        try:
            result = self._add_wait(
                request, registered, received=received, prepare_only=prepare_only,
                block_admission=block_admission, admission_deadline=admission_deadline,
            )
            if isinstance(result, AddReceipt) and not self.requires_completion:
                self.writer.add_admission.reception_only(scope)
            context = _RECEPTION.get()
            if isinstance(result, AddReceipt) and context is not None:
                original = context.dispatch.receipt
                if original is not None:
                    if (
                        result.run_id, result.visibility_commit_seq, result.message_count,
                        result.work_deadline_at,
                    ) != (
                        original.run_id, original.visibility_commit_seq, original.message_count,
                        original.work_deadline_at,
                    ):
                        raise QuoteMemError("binding_mismatch",
                                            "Add continuation changed its receipt")
                    result = replace(result, reused=original.reused)
            return result
        except QuoteMemError as error:
            if error.code in _REJECTED_RECEPTION:
                for run_id in registered:
                    self.writer.discard_rejected_add_input(run_id)
            raise
        finally:
            if block_admission and not prepare_only:
                for run_id in registered:
                    self.writer.add_admission.withdraw(scope, run_id, attempt=attempt)

    def _wait_admission(
        self, request: AddRequestV1, run_id: str, deadline: float, *, wait: bool = True,
    ) -> bool:
        """Wait for the single ordering turn without a session or Writer lock."""
        scope = scope_id("user", request.user_id)
        queue = self.writer.add_admission
        reconcile_add_admission(self.writer, scope)
        budget = current_budget()

        def fence_open() -> bool:
            lease = self.writer.get_control(f"lease:finalize:{scope}")
            if lease is None or lease["value"]["state"] == "cancelled":
                return True
            if lease["value"]["state"] != "released":
                return False
            owner = self.writer.get_row("run", lease["value"]["owner_run_id"])
            return owner is not None and owner["status"] == "succeeded"

        def check() -> None:
            check_request_budget()
            self._check_finalize_pending(request)

        with measure("queue"), (
            nullcontext() if budget is None else budget.on_cancel(queue.wake)
        ):
            return queue.wait_turn(
                scope, run_id, fence_open=fence_open, check=check, deadline=deadline,
                serving=self.requires_completion, wait=wait,
            )

    def _add_wait(
        self, request: AddRequestV1, registered: set[str], *, received: Callable[[str], None],
        prepare_only: bool = False, block_admission: bool = True,
        admission_deadline: float | None = None,
    ) -> AddReceipt | _PreparedAdd | None:
        fence_key = (
            "control", control_id(f"lease:finalize:{scope_id('user', request.user_id)}"),
        )
        budget = current_budget()
        deadline = time.monotonic() + 20.0 if budget is None else budget.deadline
        admission_deadline = min(
            deadline, admission_deadline or (
                time.monotonic() + DEFAULT_REQUEST_TIMEOUT_SECONDS
            ),
        )
        if not block_admission:
            deadline = min(deadline, admission_deadline)
        while True:
            check_request_budget()
            # Taken before the attempt reads the finalize lease, so a release
            # that lands during or after the attempt is never missed.
            fence_token = self.writer.storage_epoch
            try:
                result = self._add_once(
                    request, received=received, admission_deadline=admission_deadline,
                    prepare_only=prepare_only, block_admission=block_admission,
                )
                if isinstance(result, _CleanupSelection):
                    # Reader/worker drainage must never wait while the session
                    # mutex or global commit mutex remains held by this request.
                    self.cleanup.execute(
                        result.scope,
                        result.supplied,
                        self._grant(result.scope),
                        parent_run_id=result.parent_run_id,
                    )
                    raise QuoteMemError(
                        "secret_rejected", "Credential-bearing source cannot be accepted"
                    )
                return result
            except QuoteMemError as error:
                if error.code not in {
                    "authorization_expired",
                    "reception_stale",
                    "finalize_pending",
                }:
                    raise
                try:
                    _ = self.publisher.current
                except QuoteMemError as publication_error:
                    if publication_error.code != "service_unhealthy":
                        raise
                    self.writer.recover_catalog()
                    self.publisher.recover_privacy(self.writer)
                if error.code == "finalize_pending":
                    self._check_finalize_pending(request)
                    if not block_admission:
                        return None
                    if registered:
                        self._wait_admission(request, next(iter(registered)), admission_deadline)
                        continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not block_admission:
                        raise QuoteMemError(
                            "delivery_busy", "The scope's admission queue is busy; retry",
                        ) from None
                    if error.code == "finalize_pending":
                        raise QuoteMemError(
                            "index_rebuilding", "Finalize has not released the reception fence"
                        ) from None
                    raise QuoteMemError(
                        "service_unhealthy", "Reception could not obtain a stable authority"
                    ) from None
                # Retry outside both short locks. Expiry changes grant lifetime,
                # never the canonical admission or durable request identity.
                with measure("queue"):
                    if error.code == "finalize_pending":
                        # A held finalize keeps the fence for its whole
                        # execution. Re-running admission every 10 ms took the
                        # Writer mutex twice per attempt for nothing; wait
                        # without the Writer until the lease row is rewritten
                        # (release, or a heartbeat), bounded by the same
                        # deadline and a periodic full recheck.
                        await_fence_release(
                            self.writer, fence_key, fence_token,
                            min(deadline, time.monotonic() + FINALIZE_FENCE_RECHECK_SECONDS),
                        )
                    else:
                        time.sleep(min(0.01, remaining))

    def _add_once(
        self, request: AddRequestV1, *, received: Callable[[str], None] | None = None,
        admission_deadline: float | None = None,
        prepare_only: bool = False, block_admission: bool = True,
    ) -> AddReceipt | _CleanupSelection | _PreparedAdd | None:
        if any(message.role == "tool" for message in request.messages):
            raise QuoteMemError("invalid_request",
                                "Tool messages require a trusted source descriptor")
        scope = ScopeV1(scope_kind="user", scope_id=scope_id("user", request.user_id))
        grant = self._grant(scope)
        binding = build_add_admission(
            request,
            principal_id=grant.principal_id,
            profile=self.profile,
            bind_content=self.writer.bind_content,
        )
        session_hash = cast(str, binding.admission.target_locator)
        # The first Add of a scope builds provisional rows for its capacity check and
        # then its real rows; one request is classified once (a 1 MiB text per call).
        context = _RECEPTION.get()
        security = (
            context.dispatch.security if context is not None
            and context.queue is self.writer.add_admission and context.dispatch.service is self
            and context.dispatch.request is request
            and context.dispatch.security is not None
            else functools.cache(partial(self._security, request))
        )
        session_lock = self.writer.session_mutex(scope.scope_id, session_hash)
        with _session_admission(session_lock):
            self._scope_gate(scope, request.user_id, allow_pending_privacy=True)
            existing = self.reader.latest_run("session_ingestion", binding.idempotency_key)
            ledger, hit, head = self._ledger(scope, session_hash, binding)
            if hit is not None:
                return self._replay_receipt(request, scope, binding, hit, grant)
            if existing is not None and existing["admission_digest"] != binding.admission_digest:
                check_add_grant(grant, scope, self.clock(), binding.admission.principal_id)
                raise QuoteMemError("operation_conflict", "Request family binds another input")
            if existing is not None and (
                existing["status"] == "failed" and existing["failure_code"] == "policy_reject"
            ):
                from quotemem.ingestion.privacy_recovery import privacy_rejected_parent

                self._scope_gate(scope, request.user_id)
                children = self.reader.run_children(existing["object_id"], limit=2)
                if len(children) != 1:
                    raise QuoteMemError("service_unhealthy",
                                        "Rejected reception lost its cleanup proof")
                privacy_rejected_parent(
                    self.writer,
                    existing["object_id"],
                    children[0]["object_id"],
                    allow_terminal=True,
                )
                check_add_grant(grant, scope, self.clock(), binding.admission.principal_id)
                self._scope_gate(scope, request.user_id)
                raise QuoteMemError("secret_rejected",
                                    "Credential-bearing source cannot be accepted")
            secret = secret_preflight(
                request, profile=self.profile, prior_evidence=self._tail(scope, request.session_id)
            )
            if secret.disposition == "cleanup_required":
                return self._cleanup_selection(
                    scope, secret, None if existing is None else existing["object_id"]
                )
            if secret.disposition == "reject_current":
                raise QuoteMemError("secret_rejected",
                                    "Credential-bearing source cannot be accepted")
            self._scope_gate(scope, request.user_id)
            if self.requires_completion:
                self._check_finalize_pending(request)
            embedding = self.writer.get_control("policy:embedding_space")
            lease = self.writer.get_control("lease:writer")
            if embedding is None or lease is None:
                raise QuoteMemError("service_unhealthy", "Reception bootstrap is incomplete")
            embedding_token = embedding_state_token(ControlRowV1.model_validate(embedding))
            reception_ticket = self._reception_ticket(request)
            now = self.clock()
            root = SessionTargetRunRowV1.model_validate(
                {
                    "object_id": new_ulid(),
                    "run_kind": "session_ingestion",
                    "commit_mode": "orchestrator",
                    "status": "queued",
                    "stage": "await_input",
                    "attempt": 0,
                    "idempotency_key": binding.idempotency_key,
                    "run_generation": 0,
                    "admission_digest": binding.admission_digest,
                    "principal_id": grant.principal_id,
                    "scope_kind": "user",
                    "scope_id": scope.scope_id,
                    "target_kind": "session",
                    "target_id": session_hash,
                    "writer_epoch": lease["value"]["writer_epoch"],
                    "admitted_erase_epoch": 0,
                    "admitted_consent_epoch": 0,
                    "created_at": now if reception_ticket is None else reception_ticket.received_at,
                }
            )
            provisional = self._evidence(
                request,
                [[1.0] + [0.0] * (self.writer.dimension - 1)] * len(split_evidence(request)),
                embedding["value"]["embedding_space"], security(),
            )
            self.writer.count_reception_rowops(
                [
                    *[RowWrite("evidence", row) for row in provisional],
                    RowWrite("run", root),
                    *[
                        RowWrite(
                            "control", self.writer.materialize_control(key, {"seq": 1}, 0, now)
                        )
                        for key in ("commit_seq", f"scope_seq:{scope.scope_id}")
                    ],
                    RowWrite(
                        "control",
                        self.writer.materialize_control(
                            f"request_ledger:{scope.scope_id}:{session_hash}",
                            {
                                "recent": {
                                    binding.request_key: {
                                        "admission_digest": binding.admission_digest
                                    }
                                }
                            },
                            0,
                            now,
                        ),
                    ),
                ]
            )
            if not self._scope_registered(scope, request.user_id):
                # Only a user's first Add registers; the block re-reads the
                # registration under the mutex and keeps its fail-closed path.
                with self.writer.mutex:
                    registration_generation = self.writer.n_reserved_writes
                    try:
                        self.writer.register_scope("user", scope.scope_id, request.user_id, now)
                        self.publisher.publish_scope_registration(
                            self.writer, scope, request.user_id
                        )
                    except BaseException:
                        if self.writer.n_reserved_writes != registration_generation:
                            self.publisher.fail_closed()
                        raise
            if existing is None:
                root = SessionTargetRunRowV1.model_validate(
                    {
                        **root.model_dump(mode="json"),
                        "admitted_erase_epoch": self.writer.get_counter(
                            f"erase_epoch:{scope.scope_id}"
                        ),
                        "admitted_consent_epoch": self.writer.get_counter(
                            f"consent_epoch:{scope.scope_id}"
                        ),
                    }
                )
                try:
                    ticket = self.writer.admit_add_root(root)
                except QuoteMemError as error:
                    if error.code not in {"erase_fenced", "consent_fenced"}:
                        raise
                    # The epochs were read outside the Writer mutex. A forget or
                    # consent change that committed since then refuses this root
                    # before any Run or input is written, so the request ID is not
                    # spent: the next attempt re-reads the epochs and admits under
                    # the new ones. Only an admitted or adopted Run is terminal.
                    raise QuoteMemError(
                        "reception_stale", "A privacy change committed before admission",
                    ) from None
            else:
                if existing["status"] == "failed":
                    raise QuoteMemError(
                        "operation_conflict",
                        "Accepted Add failed before ordering; a new request identity is required",
                    )
                # Only an unsaved expired root is void; saved source is retained.
                self._require_unexpired(existing["object_id"])
                # A scope epoch that changed since the first reception (a forget)
                # rejects the adopted root here, and its saved source goes too.
                with self._discard_on_rejection(existing["object_id"]):
                    try:
                        ticket = self.writer.adopt_add_run(existing["object_id"], now)
                    except QuoteMemError as error:
                        # The Writer checks the erase epoch first. A user deletion
                        # withdraws consent before it forgets (agent/admin.py), so
                        # name the consent change: an Add a forget alone overtook
                        # may be re-admitted, one received before a deletion never.
                        if error.code == "erase_fenced" and self.writer.get_counter(
                            f"consent_epoch:{scope.scope_id}"
                        ) != existing["admitted_consent_epoch"]:
                            raise QuoteMemError(
                                "consent_fenced", "Consent changed after the Add was received",
                            ) from None
                        raise
            run_id = cast(str, ticket.run_id)
            if reception_ticket is not None:
                self.writer.add_admission.bind(
                    reception_ticket, run_id,
                    root.created_at if existing is None else existing["created_at"],
                )
            with self._discard_on_rejection(run_id):
                with self.writer.mutex:
                    before_save = self.writer.n_reserved_writes
                    try:
                        self.writer.receive_add_input(ticket, binding, grant, clock=self.clock)
                    except BaseException:
                        if reception_ticket is not None and (
                            self.writer.n_reserved_writes != before_save
                        ):
                            self.writer.add_admission.retain(reception_ticket)
                        raise
                if received is not None:
                    received(run_id)
                if self.requires_completion:
                    self.writer.add_admission.request_completion(scope.scope_id, run_id)
        if self.checkpoint is not None:
            self.checkpoint("input_received")
        if prepare_only:
            return _PreparedAdd(run_id)
        finalize = self.writer.get_control(f"lease:finalize:{scope.scope_id}")
        if finalize is not None and finalize["value"]["state"] in {"preparing", "held"}:
            raise QuoteMemError(
                "finalize_pending", "Accepted input awaits the preceding semantic prefix",
            )
        budget = current_budget()
        if admission_deadline is None:
            admission_deadline = time.monotonic() + DEFAULT_REQUEST_TIMEOUT_SECONDS
        if budget is not None:
            admission_deadline = min(admission_deadline, budget.deadline)
        if not self._wait_admission(request, run_id, admission_deadline, wait=block_admission):
            return None
        # A rejection deletes the saved source before the executor is released, so
        # no concurrent attempt of this request orders in between.
        with (
            _session_admission(self.writer.run_executor(run_id)),
            self._discard_on_rejection(run_id),
        ):
            grant = self._grant(scope, binding.admission.principal_id)
            with _session_admission(session_lock):
                self._scope_gate(scope, request.user_id)
                _, hit, _ = self._ledger(scope, session_hash, binding)
                if hit is not None:
                    return self._replay_receipt(request, scope, binding, hit, grant)
                self._require_unexpired(run_id)
                self.writer.issue_ticket(run_id)
                check_add_grant(grant, scope, self.clock(), binding.admission.principal_id)
            source_texts = [part.text for part in split_evidence(request)]
            unique_texts = list(dict.fromkeys(source_texts))
            baseline = cast(EmbeddingCanaryConfigV1, self.writer.read_logical(
                embedding["value"]["canary_baseline_ref"], "policy_config",
            ))
            vectors = self._source_vectors(unique_texts, baseline.parameters.model)
            source_rows = self._evidence(
                request, [vectors[text] for text in source_texts],
                embedding["value"]["embedding_space"], security(),
            )
            grant = self._grant(scope, binding.admission.principal_id)
            with _session_admission(session_lock):
                self._scope_gate(scope, request.user_id)
                ledger, hit, head = self._ledger(scope, session_hash, binding)
                if hit is not None:
                    return self._replay_receipt(request, scope, binding, hit, grant)
                # An unsaved root or scope epoch may change while this attempt
                # embeds outside the session lock; re-prove it before ordering.
                self._require_unexpired(run_id)
                secret = secret_preflight(
                    request, profile=self.profile,
                    prior_evidence=self._tail(scope, request.session_id),
                )
                if secret.disposition == "cleanup_required":
                    return self._cleanup_selection(scope, secret, run_id)
                if secret.disposition == "reject_current":
                    raise QuoteMemError("secret_rejected",
                                        "Session source changed during reception")
                # Writer revokes publication only after a write attempt can have
                # changed persistent state. A pre-write fence rejection leaves the
                # current root intact and must remain an ordinary bounded wait.
                try:
                    reference = self.writer.commit_add(
                        ticket,
                        request,
                        binding,
                        source_rows,
                        grant,
                        profile=self.profile,
                        ledger=ledger,
                        ledger_head=head,
                        embedding_token=embedding_token,
                        clock=self.clock,
                        publish=lambda receipt, evidence: self.publisher.publish_reception(
                            self.writer, receipt, evidence
                        ),
                        checkpoint=self.checkpoint,
                        completion_required=self.requires_completion,
                    )
                except QuoteMemError as error:
                    if error.code == "delivery_busy":
                        # Re-prove whether an unsaved root was independently voided.
                        self._require_unexpired(run_id)
                    raise
                witness = cast(
                    SessionDurableRunInputV1, self.writer.read_logical(reference, "run_input")
                )
                self.publisher.check_reception(self.writer, witness)
                self.writer.discard_received_add_input(run_id)
                return AddReceipt(
                    run_id,
                    witness.visibility_commit_seq,
                    len(request.messages),
                    False,
                    effective_receipt_deadline(witness),
                )
