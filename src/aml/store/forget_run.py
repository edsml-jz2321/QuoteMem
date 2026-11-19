"""The D.3 forget producer on the single Writer: admission, effects, final scan, receipts.

Phases (contract D.3), each a durable Writer transaction whose progress lives in
the scope's ``erasure:{scope}:{controller}`` fence value (``ErasureFenceV1``):

* P0 ``prepare``: a bounded ``intent:…:forget:…`` Control bound to the exact head.
* P1 ``admit``: (a) the global ``erasure_controller`` Run is written first, alone:
  a Control's ``updated_by_run_id`` must name a Run that already exists. One
  transaction then admits the scoped ``forget`` worker, consumes the intent and
  installs the fence with the frozen plan (the stating span set S, tombstone ids,
  entity candidates); both Controls name the controller as their writer. S is
  every span the forgotten versions cite, less the phrase a surviving readable
  naming memory of a version's primary entity quotes as its own support, less a
  bare name a neutral-type-only naming memory quotes, and less the "my" phrase a
  surviving memory quotes as its own subject (``plan_stating_spans``, O11): the
  forget hides what states the forgotten fact, not the name or the referring phrase
  of an entity it was about. Sensitive keeper heads keep nothing; bare names keep
  nothing when any forgotten version is sensitive. The worker has
  no parent Run (a parent shares its child's exact scope, and the controller is
  global); its admission and the fence bind it to its controller. (b) the
  worker's first NL transaction writes the tombstones (reusing an existing
  ``(entity_id, statement_digest)`` row, A.3.8), advances ``tombstone_seq``/
  ``erase_epoch``/``scope_seq``/``commit_seq`` together and withholds the first
  spans of S (up to ``MAX_TX_ROWS``). The fence installed in (a) bars every reader
  and writer of the scope, so (a)+(b) is atomic to every observer; a crash between
  them leaves phase ``admitting`` and resume redoes (b) from the frozen plan.
* P2 ``delete``: bounded batches inside the fence withhold the rest of S, requote
  every dependent failing P-VER after the fence's cursor (D.7.3
  ``choose_compensation(mode="forget_dependent")``, never erase; one it leaves
  unreadable keeps the readable parts of a group whose words carry it, its statement
  showing no word of S or of a forgotten statement and its keywords dropping any,
  ``trimmed_support``, C22, NM18; every other readable survivor drops, in a
  keyword-only revision, each keyword carrying a word of what a forgotten statement
  says about its subject that its statement and readable quotes do not show,
  ``forgotten_keywords``, #4b, KW-1n), delete the
  ``supersedes``/``contradicts`` links other Items carry to the retention unit's
  Items, and expire the pending candidates left without a usable group
  (``evidence_invalidated``) in id order after their own cursor. Then one
  transaction deletes the unit's own rows (the Item with its revisions, their
  links, its names and candidates, and every Item correcting its history),
  erases the extraction entities left without an anchor (a participant only with
  its last source) and freezes into the fence the retention records of the
  deleted positions and the Search deliveries that returned a deleted Item or an
  Evidence row of S. Later batches delete those Artifacts, Controls and
  deliveries. Every batch rereads its rows (INV W21). P1 refuses, before any
  write, a unit whose own rows cannot fit that transaction or whose records
  exceed the fence's bounds: its fence could never finish.
* P3 ``scan``: FS1–FS6 over durable storage and the read view (FS5 reruns the
  dependent and entity choices, which must find nothing left to do); the keyed
  scan digest is bound into the fence, then into ``ForgetReceiptV3``.
* P4 ``release``: receipts written and verified, ``privacy_quiescence``, the fence
  removed and both Runs succeed.

``resume`` continues from the fence phase without redoing a committed write;
boot step 13 calls ``recover_forgets``, which first adopts the fence's Runs under
the new Writer epoch (C.13).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError

from aml.core.artifacts import ManifestV1
from aml.core.controls import CancelledFinalizeLeaseV1, ControlRowV1, EvidenceHoldV1
from aml.core.entities import MemorySnapshotV2, QuoteSpanV1, RevisionReason
from aml.core.errors import AMLError
from aml.core.forget import (
    MAX_FORGET_SOURCES,
    MAX_FORGET_TOMBSTONES,
    ForgetCountsV1,
    ForgetInternalRetentionV1,
    ForgetPrepareReceiptV3,
    ForgetReceiptV3,
    ForgetWithheldSourceV1,
)
from aml.core.forget_protocol import verify_forget_receipt_pair
from aml.core.forget_receipts import ErasureReceiptResultIndexV1
from aml.core.forget_state import (
    INTENT_TTL_MS,
    MAX_FORGET_ARTIFACTS,
    MAX_FORGET_CHAIN,
    MAX_FORGET_CONTROLS,
    MAX_FORGET_ENTITIES,
    MAX_FORGET_RECALLS,
    MAX_FORGET_SPANS,
    ErasureFenceV1,
    ErasurePhase,
    ForgetIntentV1,
)
from aml.core.identity import ContentBindingDigest, new_ulid
from aml.core.judge_records import (
    JudgeCallManifestV1,
    JudgeResultIndexV8,
    JudgmentRunInputV1,
    judgment_targets,
)
from aml.core.privacy import TrimmedSupport, choose_compensation, support_words, trimmed_support
from aml.core.proposal import ProposalV2
from aml.core.query import ScopeV1
from aml.core.readability import Readability
from aml.core.statement import name_norm
from aml.ingestion.authorization import check_add_grant
from aml.providers.auth import PrincipalGrant
from aml.store.add_outcome import request_key
from aml.store.domain import EntityNameRowV1, ItemRowV2, RevisionRowV2, TombstoneRowV1
from aml.store.entity_ops import (
    Row,
    TxnRows,
    erase_unanchored_entities,
    rekey,
    requote_on,
)
from aml.store.forget_admission import (
    CorrectionEdge,
    ForgetVersionSelection,
    ForgottenCitation,
    KeeperGroup,
    bare_name_spans,
    correction_chain,
    forgotten_keywords,
    named_by_type,
    plan_earlier_wordings,
    plan_forget_admission,
    plan_stating_spans,
    possession_entailed,
    predicate_words,
)
from aml.store.forget_final_scan import privacy_quiescence
from aml.store.indexed import (
    MAX_PAGE_SIZE,
    CandidateRows,
    ItemRows,
    LinkTargetRows,
    RecallDeliveries,
    RevisionRows,
    ScopedRunRows,
    TombstoneRows,
)
from aml.store.judge_records import LOCATOR_PURPOSE
from aml.store.memory_commit import (
    GroupsOf,
    expire_unusable,
    proposal_groups,
    readability,
    withhold_spans,
)
from aml.store.privacy_compensation import participant_has_source
from aml.store.read_model import StoreReadModel
from aml.store.writer import MAX_TX_ROWS, RowWrite, Writer, WriteTicket

FORGET_RECEIPT_KINDS = frozenset({
    "receipt:forget_prepare", "receipt:forget_final", "result_index:erasure_receipt",
})
"""The only artifacts the erasure controller writes (E.2.4; A.3.12)."""
_SPAN_BATCH = 256
"""Spans withheld per P1/P2 transaction; each touches at most this many Evidence rows."""
_CASCADE_LIMIT = 64
"""Historical correcting Items one forget may cascade to before refusing."""
_DEPENDENT_BATCH = 64
"""Dependent Items compensated per P2 transaction."""
_LINK_BATCH = 512
"""Links other Items carry to the retention unit's Items, deleted per P2 transaction."""
_EXPIRY_BATCH = 512
"""Pending candidates expired per P2 transaction."""
_UNIT_OVERHEAD = 3
"""RowOps the unit's P2 transaction adds besides its rows: the fence, ``commit_seq`` and
``scope_seq``."""
_PURGE_BATCH = 256
"""Retention-unit Artifacts and Controls deleted per P2 transaction."""
_COHORT_BATCH = 256
"""Children ended, or candidates deleted, per cancelled-cohort P2 transaction (D.5)."""
_ENDED = frozenset({"succeeded", "failed"})
"""Run statuses a cancelled cohort leaves as they are."""


def _ended(run: Mapping[str, Any], now: int) -> dict[str, Any]:
    """A cancelled cohort Run's post-image: failed ``erase_fenced`` (D.5)."""
    return dict(run, status="failed", failure_code="erase_fenced", updated_at=now,
                finished_at=now)


def owns_erasure_fence(run: Mapping[str, Any], control: Mapping[str, Any]) -> bool:
    """Whether ``run`` is the controller or worker the scope's erasure fence names."""
    raw: object = control.get("value")
    if not isinstance(raw, Mapping):
        return False
    fence = cast(Mapping[str, Any], raw)
    if fence.get("schema_version") != 1:
        return False
    return run.get("object_id") in {fence.get("controller_run_id"), fence.get("worker_run_id")}


def fence_erase_epoch(control: Mapping[str, Any]) -> int:
    """The ``erase_epoch`` the fence owner writes at: pre-bump before P1 commits, then +1."""
    fence = ErasureFenceV1.model_validate(control["value"])
    return fence.erase_epoch if fence.phase == "admitting" else fence.erase_epoch + 1


def check_controller_artifact(
    writer: Writer, ticket: WriteTicket, kind: str, security_level: str, source: object,
) -> None:
    """The D.3 receipt write proof: only the fence's controller, only after the scan.

    The controller writes exactly its receipt pair and their selection, at
    ``restricted``, while its ``erasure`` fence is in the ``scanned`` phase; a
    final receipt must carry the fence's bound ``final_scan_digest`` (FS1–FS6).

    Raises:
        AMLError: ``invalid_write_mode`` for another kind, level, phase, scan
            digest, or a controller whose fence is gone.
    """
    receipt = isinstance(
        source, ForgetPrepareReceiptV3 | ForgetReceiptV3 | ErasureReceiptResultIndexV1,
    )
    if kind not in {"receipt", "result_index"} or not receipt or security_level != "restricted":
        raise AMLError("invalid_write_mode", "The erasure controller writes only its receipts")
    control = None if ticket.run_id is None else _fence_of_controller(writer, ticket.run_id)
    if control is None:
        raise AMLError("invalid_write_mode", "Forget receipts require the controller's fence")
    fence = ErasureFenceV1.model_validate(control["value"])
    if fence.controller_run_id != ticket.run_id or fence.phase != "scanned":
        raise AMLError("invalid_write_mode", "Forget receipts follow the bound final scan")
    if isinstance(source, ForgetReceiptV3) and source.final_scan_digest != fence.scan_digest:
        raise AMLError("invalid_write_mode", "The final receipt names another scan")


def _fence_of_controller(writer: Writer, controller_run_id: str) -> Row | None:
    run = writer.get_row("run", controller_run_id)
    if run is None:
        return None
    for row in writer.iter_rows("control"):
        key = cast(str, row["key"])
        if key.startswith("erasure:") and key.endswith(f":{controller_run_id}"):
            return row
    return None


def intent_key(scope_id: str, principal_id: str, binding: str, nonce: str) -> str:
    """``intent:{scope}:{principal}:forget:{binding}:{ulid}`` (A.3.9)."""
    return f"intent:{scope_id}:{principal_id}:forget:{binding}:{nonce}"


def fence_key(scope_id: str, controller_run_id: str) -> str:
    """``erasure:{scope}:{controller}`` (A.3.9)."""
    return f"erasure:{scope_id}:{controller_run_id}"


@dataclass(frozen=True)
class ForgetPrepared:
    """P0's durable intent and the exact head it binds."""

    intent_key: str
    memory_id: str
    expected_revision_id: str
    expected_content_hash: str
    expires_at: int


@dataclass(frozen=True)
class ForgetOutcome:
    """A verified forget: the controller, its receipt pair and the published scope state."""

    controller_run_id: str
    prepare_ref: str
    receipt_ref: str
    receipt: ForgetReceiptV3


@dataclass(frozen=True)
class FinalScan:
    """FS1–FS6 counts over durable storage and the read view, all zero when verified.

    ``records`` counts the retention-unit rows left besides the Item's own: a
    candidate naming the Item, a frozen Artifact, Control or Search delivery,
    or a delivery that returned the Item or an Evidence row of S (FS1).
    ``unwithheld_spans`` counts the spans of S no withheld range of their row
    covers; a privacy-erased row keeps no ranges and needs none (FS3, A.3.5).
    ``pending_unusable`` counts the pending candidates with a quote overlapping
    S and no usable group left: P2's expiry must have ended them (FS4). A
    pending sibling that keeps a usable group, on a disjoint span of the same
    part included, is not residual work (D.2).
    ``dependents`` and ``entities`` count the FS5 effects still to apply: a
    dependent the chooser would still requote or a readable survivor still carrying a
    keyword the forget drops (D.7.3 #4b), an entity it would still erase.
    """

    item: int
    revision: int
    link: int
    records: int
    unwithheld_spans: int
    pending_unusable: int
    tombstones_missing: int
    dependents: int
    entities: int
    read_view: int

    def verified(self) -> bool:
        """Every check found nothing left."""
        return not any((self.item, self.revision, self.link, self.records,
                        self.unwithheld_spans, self.pending_unusable, self.tombstones_missing,
                        self.dependents, self.entities, self.read_view))


def _no_checkpoint(_name: str) -> None:
    """The default crash-point hook: no injected failure."""


class ForgetOwner:
    """The single Writer's forget producer (D.3) for one workspace."""

    def __init__(
        self, writer: Writer, publisher: StoreReadModel, *, clock: Callable[[], int],
        policy_release_ref: str | None = None,
        checkpoint: Callable[[str], None] | None = None,
        groups_of: GroupsOf | None = None,
    ) -> None:
        self._writer, self._publisher, self._clock = writer, publisher, clock
        self._groups_of: GroupsOf = groups_of or proposal_groups(writer)
        self._policy_release_ref = policy_release_ref
        self._checkpoint: Callable[[str], None] = checkpoint or _no_checkpoint

    # P0 -------------------------------------------------------------------------------
    def prepare(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str, memory_id: str,
        expected_revision_id: str,
    ) -> ForgetPrepared:
        """Issue one bounded intent bound to the Item's current head (D.3 P0).

        Raises:
            AMLError: ``scope_denied`` for an absent or other-scope Item,
                ``operation_conflict`` when the head moved, ``erasure_in_progress``
                while a fence holds the scope.
        """
        writer = self._writer
        session = self._session_token(scope, session_id)
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            now = self._clock()
            check_add_grant(grant, scope, now)
            self._refuse_fenced(scope)
            item, head = self._current(scope, memory_id)
            if item["current_revision_id"] != expected_revision_id:
                raise AMLError("operation_conflict", "The memory changed since it was selected")
            erase_epoch = writer.counter(f"erase_epoch:{scope.scope_id}")
            binding = self._selection(scope, grant.principal_id, session, item, head,
                                      erase_epoch)
            key = intent_key(scope.scope_id, grant.principal_id, str(binding), new_ulid())
            value = ForgetIntentV1(
                state="active", scope_id=scope.scope_id, principal_id=grant.principal_id,
                session_token=session, memory_id=memory_id,
                expected_revision_id=expected_revision_id, selection_digest=binding,
                erase_epoch=erase_epoch, issued_at=now, expires_at=now + INTENT_TTL_MS,
            )
            control = writer.materialize_control(key, value.model_dump(mode="json"), 0, now)
            _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
                [RowWrite("control", control)], None,
            )
            writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return ForgetPrepared(key, memory_id, expected_revision_id, head["content_hash"],
                              value.expires_at)

    # P1 -------------------------------------------------------------------------------
    def forget(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str, memory_id: str,
        expected_revision_id: str, expected_content_hash: str, intent: str,
    ) -> ForgetOutcome:
        """Admit, run P1–P4 and return the verified receipt (D.3).

        A repeated confirm of a consumed intent resumes or returns its original
        controller's outcome (D.3 P1 #1).
        """
        controller = self.admit(
            scope, grant, session_id=session_id, memory_id=memory_id,
            expected_revision_id=expected_revision_id,
            expected_content_hash=expected_content_hash, intent=intent,
        )
        return self.resume(scope, controller)

    def admit(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str, memory_id: str,
        expected_revision_id: str, expected_content_hash: str, intent: str,
    ) -> str:
        """P1 (a): bind the intent, admit both Runs and install the fence with the plan.

        A live Add owner of the scope (a ``preparing`` or ``held`` finalize
        lease) is cancelled in the same transaction (D.3 P1 #7, D.5): its lease
        becomes ``cancelled``, its Run fails ``erase_fenced``, and its pool
        invocation is told to stop. P2 settles its cohort; the next owner
        reprocesses its requests.

        Returns:
            The controller Run id (the original one on a repeated confirm).

        Raises:
            AMLError: ``operation_conflict`` for an unknown, expired or differently
                bound intent or a moved head; ``erasure_in_progress`` behind another
                fence; ``budget_exceeded``, before any write, for a plan over its
                bounds (``_preflight``).
        """
        session = self._session_token(scope, session_id)
        writer = self._writer
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            now = self._clock()
            check_add_grant(grant, scope, now)
            stored = writer.get_control(intent)
            if stored is None or not intent.startswith(f"intent:{scope.scope_id}:"):
                raise AMLError("operation_conflict", "The forget confirmation is not current")
            value = ForgetIntentV1.model_validate(stored["value"])
            if (value.principal_id, value.session_token, value.memory_id,
                    value.expected_revision_id) != (grant.principal_id, session, memory_id,
                                                    expected_revision_id):
                raise AMLError("operation_conflict", "The confirmation repeats another selection")
            if value.state == "consumed":
                assert value.controller_run_id is not None
                return value.controller_run_id
            if now >= value.expires_at:
                raise AMLError("intent_expired", "The forget confirmation expired")
            self._refuse_fenced(scope)
            self._refuse_maintenance(scope)
            live = self._live_owner(scope)
            item, head = self._current(scope, memory_id)
            erase_epoch = writer.counter(f"erase_epoch:{scope.scope_id}")
            if (
                item["current_revision_id"] != expected_revision_id
                or head["content_hash"] != expected_content_hash
                or erase_epoch != value.erase_epoch
                or self._selection(scope, grant.principal_id, session, item, head,
                                   erase_epoch) != value.selection_digest
            ):
                raise AMLError("operation_conflict", "The memory changed since it was selected")
            txn = self._txn(scope, None, now)
            chain = self._chain(txn, memory_id)
            forgotten = (memory_id, *chain)
            reworded = self._reworded(scope, txn, forgotten)
            versions, entity_ids, cited = self._chain_plan(forgotten, reworded)
            spans = plan_stating_spans(
                cited, [*self._keeper_groups(scope, txn, set(forgotten), cited),
                        *self._reworded_keepers(txn, reworded)],
                withdrawn=self._withdrawn(txn, set(forgotten), cited),
            )
            evidence = frozenset(span.evidence_id for span in spans)
            if (len(versions) > MAX_FORGET_TOMBSTONES or len(spans) > MAX_FORGET_SPANS
                    or len(entity_ids) > MAX_FORGET_ENTITIES
                    or len(evidence) > MAX_FORGET_SOURCES):
                raise AMLError("budget_exceeded", "The forget plan exceeds its bounds")
            self._preflight(scope, (memory_id, *chain), evidence, len(entity_ids), now)
            tombstones, created = self._tombstone_ids(scope, versions)
            lease = writer.get_control("lease:writer")
            if lease is None:
                raise AMLError("lease_lost", "Forget admission has no active Writer")
            epoch = lease["value"]["writer_epoch"]
            consent = writer.counter(f"consent_epoch:{scope.scope_id}")
            controller_id = self._controller(grant, epoch, now, intent)
            worker_id = new_ulid()
            worker = self._run_row(
                worker_id, "forget", "destructive_scan", grant, scope,
                {"target_kind": "memory", "target_id": memory_id}, epoch, now, intent,
                controller=controller_id, erase_epoch=erase_epoch, consent_epoch=consent,
            )
            fence = ErasureFenceV1(
                phase="admitting", scope_id=scope.scope_id, controller_run_id=controller_id,
                worker_run_id=worker_id, memory_id=memory_id, chain_ids=chain,
                reworded_ids=reworded,
                intent_key=intent,
                spans=tuple(spans), tombstone_ids=tuple(sorted(tombstones.values())),
                created_tombstone_ids=tuple(sorted(created)), entity_ids=tuple(entity_ids),
                erase_epoch=erase_epoch,
                tombstone_seq=writer.counter(f"tombstone_seq:{scope.scope_id}"),
                scope_seq=writer.counter(f"scope_seq:{scope.scope_id}"),
                commit_seq=writer.counter("commit_seq"), counts=ForgetCountsV1(),
                cancelled_owner_run_id=None if live is None else live[1]["object_id"],
                trim_dependents=not any(version.sensitive for version in cited),
            )
            consumed = value.model_copy(update={
                "state": "consumed", "controller_run_id": controller_id,
            })
            rows = [
                RowWrite("run", writer._parse_row("run", worker)),  # pyright: ignore[reportPrivateUsage]
                RowWrite("control", writer.materialize_control(
                    intent, ForgetIntentV1.model_validate(consumed.model_dump()).model_dump(
                        mode="json"), stored["version"] + 1, now, controller_id)),
                RowWrite("control", writer.materialize_control(
                    fence_key(scope.scope_id, controller_id), fence.model_dump(mode="json"), 0,
                    now, controller_id)),
                *([] if live is None else self._cancellation(*live, controller_id, now)),
            ]
            _, plan = writer._prepare_rows(rows, None)  # pyright: ignore[reportPrivateUsage]
            writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
            self._publisher.publish_erasure(writer, scope, None, writer.counter("commit_seq"),
                                            active=True)
            if live is not None:
                self._retire(scope, live[1]["object_id"])
        self._checkpoint("admitted_fence")
        return controller_id

    def _controller(self, grant: PrincipalGrant, epoch: int, now: int, intent: str) -> str:
        """Write the intent's global erasure controller before anything names it (D.3 P1 #7).

        The admission transaction's Controls name the controller as their
        writer, and the Writer accepts only an existing Run there, so the
        controller is its own first write, under the same Writer mutex. It holds
        no fence and changes nothing a reader of the scope sees. Only a crash
        separates it from the admission transaction (a failure after a write
        fails the read model closed); one controller exists per intent (its
        admission family), so that intent is spent and the client prepares again.
        The next boot ends such a controller (`abandon_orphan_controllers`).

        Raises:
            AMLError: ``operation_conflict`` when the intent already has a controller.
        """
        writer = self._writer
        row = self._run_row(new_ulid(), "erasure_controller", "orchestrator", grant, None,
                            {"target_kind": "none", "target_id": None}, epoch, now, intent)
        if writer.indexed_reader().latest_run(
            "erasure_controller", row["idempotency_key"],
        ) is not None:
            raise AMLError("operation_conflict", "The forget confirmation is not current")
        _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
            [RowWrite("run", writer._parse_row("run", row))], None,  # pyright: ignore[reportPrivateUsage]
        )
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return cast(str, row["object_id"])

    # Resume ---------------------------------------------------------------------------
    def resume(self, scope: ScopeV1, controller_id: str) -> ForgetOutcome:
        """Continue a forget from its durable fence phase; committed writes never repeat.

        A forget whose fence is gone completed: its outcome is read back from the
        result index.
        """
        while True:
            control = self._writer.get_control(fence_key(scope.scope_id, controller_id))
            if control is None:
                return self._outcome(controller_id)
            fence = ErasureFenceV1.model_validate(control["value"])
            step: dict[ErasurePhase, Callable[[ScopeV1, ErasureFenceV1, Row], None]] = {
                "admitting": self._effects,
                "admitted": self._delete,
                "deleted": self._purge,
                "purged": self._scan,
                "scanned": self._receipts,
                "receipted": self._release,
            }
            step[fence.phase](scope, fence, control)
            self._checkpoint(fence.phase)

    # P1 (b) ---------------------------------------------------------------------------
    def _effects(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        writer = self._writer
        ticket = writer.ticket(fence.worker_run_id)
        now = self._clock()
        txn = self._txn(scope, fence.worker_run_id, now)
        versions = self._versions(fence)
        written = 0
        for version, tombstone_id in self._creations(scope, fence, versions):
            snapshot = version["snapshot"]
            txn.put("tombstone", TombstoneRowV1.model_validate({
                "object_id": tombstone_id, "scope_kind": scope.scope_kind,
                "scope_id": scope.scope_id, "entity_id": snapshot["entity_id"],
                "about_entity_ids": snapshot["about_entity_ids"],
                "statement": snapshot["statement"],
                "statement_digest": snapshot["statement_digest"],
                "kind": snapshot["kind"], "valid_from": snapshot.get("valid_from"),
                "valid_to": snapshot.get("valid_to"),
                "time_granularity": snapshot["time_granularity"],
                "embedding": snapshot["embedding"],
                "embedding_space": snapshot["embedding_space"],
                "source_evidence_ids": version["evidence_ids"],
                "tombstone_seq": writer.counter(f"tombstone_seq:{scope.scope_id}") + 1,
                "forget_run_id": fence.worker_run_id, "created_at": now,
            }).model_dump(mode="json"))
            written += 1
        first = fence.spans[:_SPAN_BATCH]
        touched = withhold_spans(txn, first, reason="forgotten", run_id=fence.worker_run_id)
        counts = fence.counts.model_copy(update={
            "tombstones_written": written,
            "evidence_spans_withheld": _newly(first, touched),
            "evidence_rows_withheld": len(touched),
        })
        self._commit(scope, ticket, txn, fence, control, "admitted", counts, tombstones=True,
                     spans_done=len(first))

    # P2 -------------------------------------------------------------------------------
    def _delete(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        """One P2 transaction: a bounded batch, or the retention unit's own rows.

        The spans of S are withheld first; then the cohort of an Add owner P1
        cancelled is settled (``_settle_cohort``); then the dependents are compensated
        a batch at a time after the fence's cursor; then the ``supersedes`` and
        ``contradicts`` links other Items carry to the unit's Items leave a batch
        at a time, and the pending candidates S leaves without a usable group
        expire a batch at a time, in id order after their own cursor. Once none
        is left, one transaction deletes the unit's own rows, which P1 proved fit
        one transaction (``_preflight``), and freezes the Artifacts, Controls and
        deliveries the following batches delete (``_purge``).
        """
        writer = self._writer
        ticket = writer.ticket(fence.worker_run_id)
        txn = self._txn(scope, fence.worker_run_id, self._clock())
        done = fence.spans_done
        if done < len(fence.spans):
            batch = fence.spans[done:done + _SPAN_BATCH]
            touched = withhold_spans(txn, batch, reason="forgotten", run_id=fence.worker_run_id)
            counts = fence.counts.model_copy(update={
                "evidence_spans_withheld": fence.counts.evidence_spans_withheld
                + _newly(batch, touched),
                "evidence_rows_withheld": fence.counts.evidence_rows_withheld + len(touched),
            })
            self._advance(scope, ticket, txn, fence, control, counts,
                          spans_done=done + len(batch))
            return
        if fence.cancelled_owner_run_id is not None and self._settle_cohort(
            scope, ticket, txn, fence, control,
        ):
            return
        unit = self._unit(txn, (fence.memory_id, *fence.chain_ids))
        statements = self._tombstone_statements(txn, fence)
        dependents = self._dependents(scope, txn, fence, unit=set(unit), statements=statements,
                                      after=fence.dependents_after, limit=_DEPENDENT_BATCH)
        if dependents:
            read = readability(txn)
            forgotten = self._forgotten_words(txn, fence)
            requoted = sum(1 for memory_id in dependents
                           if self._settle(scope, txn, fence, read, memory_id, forgotten,
                                           statements))
            counts = fence.counts.model_copy(update={
                "dependent_requoted": fence.counts.dependent_requoted + requoted,
            })
            self._advance(scope, ticket, txn, fence, control, counts,
                          dependents_after=dependents[-1])
            return
        incoming = self._incoming(txn, unit, limit=_LINK_BATCH)
        if incoming:
            for link in incoming:
                txn.delete("link", link["object_id"])
            counts = fence.counts.model_copy(update={"link": fence.counts.link + len(incoming)})
            self._advance(scope, ticket, txn, fence, control, counts)
            return
        expired = expire_unusable(txn, set(fence.evidence_ids), groups_of=self._groups_of,
                                  after=fence.expired_after, limit=_EXPIRY_BATCH)
        if expired:
            counts = fence.counts.model_copy(update={
                "candidate_expired": fence.counts.candidate_expired + len(expired),
            })
            self._advance(scope, ticket, txn, fence, control, counts, expired_after=expired[-1])
            return
        removed, artifacts, controls, recalls = self._delete_unit(
            scope, txn, unit, fence.evidence_ids,
        )
        if fence.cancelled_owner_run_id is not None:
            artifacts = tuple(sorted(
                set(artifacts) | self._orphan_proposals(txn, fence.cancelled_owner_run_id),
            ))
            if len(artifacts) > MAX_FORGET_ARTIFACTS:
                raise AMLError("budget_exceeded", "The forget's retention records exceed bounds")
        erased = erase_unanchored_entities(
            txn, fence.entity_ids, participant_has_source=participant_has_source(txn),
        )
        counts = fence.counts.model_copy(update={
            **removed, "link": fence.counts.link + removed["link"], "entity_erased": len(erased),
        })
        self._commit(scope, ticket, txn, fence, control, "deleted", counts,
                     artifacts=artifacts, controls=controls, recalls=recalls)

    def _settle_cohort(
        self, scope: ScopeV1, ticket: WriteTicket, txn: TxnRows, fence: ErasureFenceV1,
        control: Row,
    ) -> bool:
        """One P2 batch settling the cancelled owner's cohort (D.5); False once settled.

        First the cancelled owner's unended child Runs fail ``erase_fenced``, a
        batch at a time; then its candidate rows that are not formal (every
        status but ``published``) leave, a batch at a time, so the next owner
        that reprocesses its requests finds each position once (INV P06).
        Formal rows stay: revisions reference them, and the next owner
        re-derives their positions as unchanged. Re-entered rows of earlier
        owners keep theirs. The cancelled owner's proposals no remaining row
        cites leave with the Item's own rows (``_orphan_proposals``); its judge
        outputs, with the result indexes and snapshots that hold them, stay
        reusable by digest (C.8 #1).
        """
        owner = cast(str, fence.cancelled_owner_run_id)
        if self._end_children(fence, control, owner):
            return True
        stale = self._cohort_candidates(scope, owner)
        if not stale:
            return False
        for candidate in stale:
            txn.delete("candidate", candidate["object_id"])
        self._commit(scope, ticket, txn, fence, control, "admitted", fence.counts,
                     holds=self._outliving_holds(scope, txn, stale, ticket))
        return True

    def _outliving_holds(
        self, scope: ScopeV1, txn: TxnRows, stale: Sequence[Row], ticket: WriteTicket,
    ) -> RowWrite | None:
        """The ``evidence_hold:{s}`` post-image keeping the batch's deleted holds (D.5).

        A pending row with ``hold_evidence`` keeps its Evidence out of Evidence
        routes. Deleting the row must not release that Evidence before the owner
        that reprocesses its request settles it, so the hold moves to the scope's
        ``evidence_hold`` Control in the deleting transaction, under the request
        the row is attributed to: the newest among its sources (INV:D20). The
        reprocessing owner's release removes it (``owner_lease.release_finalize``).

        Raises:
            AMLError: ``reference_missing`` for a held row without its Evidence;
                ``budget_exceeded`` past the Control's bounds.
        """
        held = [row for row in stale if row["hold_evidence"]]
        if not held:
            return None
        writer = self._writer
        key = f"evidence_hold:{scope.scope_id}"
        stored = writer.get_control(key)
        requests: dict[str, set[str]] = {}
        if stored is not None:
            current = ControlRowV1.model_validate(stored).value
            if not isinstance(current, EvidenceHoldV1):
                raise AMLError("binding_mismatch", "A scope's Evidence hold has another variant")
            requests = {request.request_key: set(request.evidence_ids)
                        for request in current.requests}
        for row in held:
            sources = [writer.get_row("evidence", evidence_id)
                       for evidence_id in row["evidence_ids"]]
            present = [source for source in sources if source is not None]
            if len(present) != len(sources) or not present:
                raise AMLError("reference_missing", "A held row lost its Evidence")
            newest = max(present, key=lambda source: (
                source["ingested_commit_seq"], source["ordinal"], source["part_no"]))
            request = request_key(writer, scope, newest["session_id"], newest["request_id"])
            requests.setdefault(str(request), set()).update(row["evidence_ids"])
        try:
            value = EvidenceHoldV1.model_validate({"requests": [
                {"request_key": request, "evidence_ids": sorted(parts)}
                for request, parts in sorted(requests.items())
            ]})
        except ValidationError as error:
            raise AMLError("budget_exceeded", "A scope's Evidence holds exceed bounds") from error
        return RowWrite("control", writer.materialize_control(
            key, value.model_dump(mode="json"), 0 if stored is None else stored["version"] + 1,
            txn.now, ticket.run_id,
        ))

    def _end_children(self, fence: ErasureFenceV1, control: Row, owner: str) -> bool:
        """Fail up to ``_COHORT_BATCH`` unended children of the cancelled owner, read under
        the Writer, with the fence's progress; False when none is left."""
        writer = self._writer
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            if writer.get_control(cast(str, control["key"])) != control:
                raise AMLError("binding_mismatch", "The erasure fence changed under its owner")
            reader = writer.indexed_reader()
            children: list[Row] = []
            after: str | None = None
            while len(children) < _COHORT_BATCH and (
                page := reader.run_children(owner, after=after, limit=MAX_PAGE_SIZE)
            ):
                children += [child for child in page if child["status"] not in _ENDED]
                after = page[-1]["object_id"]
            if not children:
                return False
            now = self._clock()
            rows = [RowWrite("run", writer._parse_row("run", _ended(child, now)))  # pyright: ignore[reportPrivateUsage]
                    for child in children[:_COHORT_BATCH]]
            rows.append(self._fence_row(fence, control, fence.phase, fence.counts, now))
            _, plan = writer._prepare_rows(rows, None)  # pyright: ignore[reportPrivateUsage]
            writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return True

    def _cohort_candidates(self, scope: ScopeV1, owner: str) -> list[Row]:
        """Up to ``_COHORT_BATCH`` of the cancelled owner's candidates that are not formal."""
        found: list[Row] = []
        for status in ("pending", "unchanged", "rejected", "expired"):
            for row in self._pages(CandidateRows(
                    scope_kind=scope.scope_kind, scope_id=scope.scope_id, status=status)):
                if row["owner_run_id"] == owner:
                    found.append(row)
                    if len(found) == _COHORT_BATCH:
                        return found
        return found

    def _orphan_proposals(self, txn: TxnRows, owner: str) -> set[str]:
        """The cancelled owner's proposals no candidate row cites once ``txn`` commits (D.5).

        Its uncommitted positions never had rows and its non-formal rows left
        in the cohort batches, so their ``ProposalV2`` records, which hold
        statement text, leave in the purge batches with the retention records.
        """
        cited = {
            row["proposal_ref"] for row in txn.merged(
                "candidate", self._pages(CandidateRows(
                    scope_kind=txn.scope_kind, scope_id=txn.scope_id, status="published")),
                lambda row: row["owner_run_id"] == owner,
            )
        }
        refs = self._writer.find_logical_candidate_refs(owner, "proposal_snapshot")
        return {str(reference) for reference in refs} - cited

    def _advance(
        self, scope: ScopeV1, ticket: WriteTicket, txn: TxnRows, fence: ErasureFenceV1,
        control: Row, counts: ForgetCountsV1, **update: Any,
    ) -> None:
        """Commit one P2 batch's rows with the fence's progress, or the progress alone."""
        if txn.written or txn.deleted:
            self._commit(scope, ticket, txn, fence, control, "admitted", counts, **update)
        else:
            self._progress(scope, fence, control, "admitted", counts, **update)

    def _dependents(
        self, scope: ScopeV1, txn: TxnRows, fence: ErasureFenceV1, *,
        statements: Sequence[str], unit: Set[str] = frozenset(), after: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Surviving Items P2 settles, in id order after ``after``: those whose head cites S,
        and those whose head carries a keyword the forget drops (D.7.3 #4b).

        P2 settles them ``limit`` at a time and moves the fence's cursor
        past each committed batch, so every dependent is settled once,
        however many there are (FS5); the final scan visits them all. The
        retention unit's Items are deleted, not compensated: only surviving
        memories are dependents (D.3 P2), and their own rows stay as P1 counted.
        """
        evidence = fence.evidence_ids
        reader = self._writer.indexed_reader()
        query = ItemRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id)
        found: list[str] = []
        while True:
            page = reader.page(query, after=after, limit=MAX_PAGE_SIZE)
            for item in page:
                if item["object_id"] == fence.memory_id or item["object_id"] in unit:
                    continue
                head = txn.get("revision", item["current_revision_id"])
                if head is not None and (evidence.intersection(head["evidence_ids"]) or (
                        forgotten_keywords(head["snapshot"]["statement"],
                                           head["snapshot"]["keywords"], statements))):
                    found.append(item["object_id"])
                    if len(found) == limit:
                        return found
            if len(page) < MAX_PAGE_SIZE:
                return found
            after = page[-1]["object_id"]

    def _settle(
        self, scope: ScopeV1, txn: TxnRows, fence: ErasureFenceV1, read: Readability,
        memory_id: str, forgotten: _Forgotten | None, statements: Sequence[str],
    ) -> bool:
        """One surviving Item's P2 effect; True when it wrote a revision.

        A head citing S is compensated (D.7.3 ``forget_dependent``); then a head that
        passes P-VER and still carries a keyword the forget drops gets a keyword-only
        revision (D.7.3 #4b). A compensating requote already drops those keywords, so an
        Item gets at most one revision.
        """
        item = txn.get("item", memory_id)
        head = None if item is None else txn.get("revision", item["current_revision_id"])
        compensated = head is not None and bool(
            fence.evidence_ids.intersection(head["evidence_ids"])) and self._compensate(
            scope, txn, read, memory_id, forgotten, statements)
        return self._rekey(txn, read, memory_id, statements) or compensated

    def _rekey(
        self, txn: TxnRows, read: Readability, memory_id: str, statements: Sequence[str],
    ) -> bool:
        """D.7.3 #4b: a readable head drops every keyword carrying a forgotten word that its
        statement and readable quotes do not show (KW-1n).

        A head a ``corrects_history`` row targets keeps its keywords: the row targets that
        revision, so a new head would no longer be hidden by the correction (A.6.4).
        """
        item = txn.get("item", memory_id)
        head = None if item is None else txn.get("revision", item["current_revision_id"])
        if head is None:
            return False
        snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
        if not read.p_ver(snapshot) or txn.links_to("corrects_history", head["object_id"]):
            return False
        quotes = [text for index in read.p_quote(snapshot).spans
                  if (text := _quoted_text(txn, snapshot.quote_spans[index])) is not None]
        dropped = forgotten_keywords(snapshot.statement, snapshot.keywords, statements, quotes)
        if not dropped:
            return False
        rekey(txn, memory_id, keywords=[word for word in snapshot.keywords if word not in dropped],
              policy_release_ref=self._policy_release_ref or head["policy_release_ref"])
        return True

    def _tombstone_statements(self, txn: TxnRows, fence: ErasureFenceV1) -> tuple[str, ...]:
        """The statements of the forget's tombstones: what D.7.3 #4b's keywords may not
        carry, whatever the forgotten versions' sensitivity."""
        found: list[str] = []
        for tombstone_id in fence.tombstone_ids:
            tombstone = txn.get("tombstone", tombstone_id)
            if tombstone is not None and tombstone["statement"]:
                found.append(tombstone["statement"])
        return tuple(found)

    def _compensate(
        self, scope: ScopeV1, txn: TxnRows, read: Readability, memory_id: str,
        forgotten: _Forgotten | None, statements: Sequence[str],
    ) -> bool:
        """D.7.3 ``forget_dependent`` for one dependent: keep, requote or leave unreadable.

        A dependent steps 1-3 leave unreadable is requoted on the readable parts of its
        head's groups when ``trimmed_support`` proves they carry it (step 4a);
        ``forgotten`` is None when the fence allows no trim. A requote keeps the
        version's keywords less those the forget drops (#4b).

        A head a ``corrects_history`` row targets is never rewritten, nor is a version
        such a row targets requoted: the row targets that revision, so the new head would
        no longer be hidden by the correction (A.6.4, C26E). Nor is a head trimmed that may
        restate a forgotten possession (``possession_entailed``).
        """
        item = txn.get("item", memory_id)
        if item is None or txn.links_to("corrects_history", item["current_revision_id"]):
            return False
        versions = [
            _Version(row) for row in txn.merged(
                "revision", self._pages(RevisionRows(memory_id=memory_id)),
                lambda row: row["memory_id"] == memory_id,
            )
        ]

        def tombstoned(entity_id: str, digest: str) -> bool:
            return any(
                row["entity_id"] == entity_id
                for row in txn.merged("tombstone", self._pages(TombstoneRows(
                    scope_kind=scope.scope_kind, scope_id=scope.scope_id,
                    statement_digest=digest)), lambda row: row["statement_digest"] == digest)
            )

        choice = choose_compensation(
            versions, read, tombstoned=tombstoned, mode="forget_dependent",
        )
        if choice.action == "leave_unreadable" and forgotten is not None:
            head = max(versions, key=lambda value: value.version)
            snapshot = head.snapshot
            if tombstoned(snapshot.entity_id, snapshot.statement_digest) or any(
                    possession_entailed(statement, snapshot.statement)
                    for statement in forgotten.statements):
                return False
            words = forgotten.words.union(*(
                predicate_words(statement, snapshot.statement)
                for statement in forgotten.statements))
            trimmed = trimmed_support(snapshot, read, forgotten=words,
                                      quoted=lambda span: _quoted_text(txn, span),
                                      speakers=lambda span: _speaker_words(txn, span))
            if trimmed is None:
                groups = _bare_name_support(txn, read, snapshot, forgotten.words)
                if not groups:
                    return False
                quotes = [text for group in groups for span in group
                          if (text := _quoted_text(txn, span)) is not None]
                dropped = forgotten_keywords(snapshot.statement, snapshot.keywords,
                                             statements, quotes)
                trimmed = TrimmedSupport(groups, tuple(
                    word for word in snapshot.keywords if word not in dropped))
            requote_on(txn, memory_id, version=head.row, support=trimmed.groups,
                       keywords=trimmed.keywords,
                       policy_release_ref=self._policy_release_ref
                       or head.row["policy_release_ref"])
            return True
        if choice.action != "requote" or choice.revision_id is None or txn.links_to(
                "corrects_history", choice.revision_id):
            return False
        version = next(v.row for v in versions if v.object_id == choice.revision_id)
        snapshot = MemorySnapshotV2.model_validate(version["snapshot"])
        support = [[snapshot.quote_spans[index] for index in snapshot.support_groups[group]]
                   for group in choice.groups]
        quotes = [text for group in support for span in group
                  if (text := _quoted_text(txn, span)) is not None]
        dropped = forgotten_keywords(snapshot.statement, snapshot.keywords, statements, quotes)
        # requote_on also re-derives the Item's names (I11).
        requote_on(txn, memory_id, version=version, support=support,
                   keywords=[word for word in snapshot.keywords if word not in dropped],
                   policy_release_ref=self._policy_release_ref or version["policy_release_ref"])
        return True

    def _forgotten_words(self, txn: TxnRows, fence: ErasureFenceV1) -> _Forgotten | None:
        """What no trimmed dependent may show (D.7.3 #4a): ``support_words`` of every span
        of S, and the statements of the forget's tombstones (``predicate_words`` reads
        each against the dependent's own statement).

        Both are frozen by P1 and retained, so P2 and FS5 read the same words. None
        when the fence allows no trim or a span or tombstone cannot be read: then no
        dependent is trimmed.
        """
        if not fence.trim_dependents:
            return None
        words: set[str] = set()
        for span in fence.spans:
            text = _quoted_text(txn, span)
            if text is None:
                return None
            words |= support_words(text)
        statements: list[str] = []
        for tombstone_id in fence.tombstone_ids:
            tombstone = txn.get("tombstone", tombstone_id)
            if tombstone is None:
                return None
            statements.append(tombstone["statement"])
        return _Forgotten(frozenset(words), tuple(statements))

    def _chain(self, txn: TxnRows, memory_id: str) -> tuple[str, ...]:
        """The Items the corrections of ``memory_id`` refuted, transitively (FG-2).

        A correction or retraction that changes the primary anchor creates a new
        Item (C.9.4), so the Items it ended by ``correct`` or ``retract`` are the
        forgotten memory's earlier versions, as a same-anchor correction's revised
        versions are (UD3). A correction the judge finds only contradicts a head,
        because it makes that head's description of an entity untrue, leaves the
        head current with a ``contradicts`` link (C.9.5 (2b), CT-1): that head
        carries the description the correction refuted, as an ended version does,
        so the chain takes it too. Which of the two relations the judge chose must
        not decide what a forget leaves readable. A head that correction reworded
        (C.9.4a) no longer carries that description; it stays outside the chain as a
        reworded Item (``_reworded``, OPR-2). A ``contradicts`` link that a
        version written by no correction carries is a disagreement, not a
        refutation, and an ``update`` ends a state that was true in its time:
        neither target is in the chain; an updated target keeps its lifecycle
        (D.3 Effects) and loses only the forgotten Item's link (A.3.3). An update
        whose link records an explicit correction (the lean judge relates
        "<new>, not <old>" as ``update``) ended the value it refuted, so the chain
        takes its target as it would a ``correct`` one. Links on
        every revision count, since revise, merge and requote carry them; the
        selected head is CAS-bound and only an active Item gains links, so the
        chain cannot change between prepare and P1.

        Returns:
            The chain's Items other than ``memory_id``, in id order.

        Raises:
            AMLError: ``budget_exceeded`` past ``MAX_FORGET_CHAIN`` Items.
        """
        def links(current: str) -> Iterator[CorrectionEdge]:
            revisions = self._revisions(txn, current)
            verdicts = _verdicts(revisions)
            for revision in revisions:
                for link in txn.links_of(revision["object_id"]):
                    target = link["target_key"]
                    if link["link_type"] == "contradicts":
                        if not link["correction_requested"] or self._reworded_by(
                                txn, target, verdicts):
                            continue
                    elif link["link_type"] != "supersedes" or (
                            link["relation"] not in ("correct", "retract")
                            and not link["correction_requested"]):
                        continue
                    item = txn.get("item", target)
                    if item is None or (item["scope_kind"], item["scope_id"]) != (
                            txn.scope_kind, txn.scope_id):
                        continue
                    # A contradiction reaches here only when its link records that a
                    # correction was requested (O-E keeps that on the link itself).
                    yield CorrectionEdge(target, link["link_type"], link["relation"],
                                         bool(link["correction_requested"]))
        return correction_chain(memory_id, links)

    def _chain_plan(
        self, memory_ids: Sequence[str], reworded: Sequence[str] = (),
    ) -> tuple[list[ForgetVersionSelection], list[str], list[ForgottenCitation]]:
        """D, the anchored entities and what each version cites (D.3 P1).

        Each Item contributes ``plan_forget_admission`` over its complete revision
        chain; a pair two Items share keeps the version of the first (the selected
        Item comes first), so admission and P1 (b) pair the same versions with the
        planned tombstone ids. Each reworded Item adds to D the wordings it had
        before a forgotten correction reworded it (``plan_earlier_wordings``), never
        to the citations: it survives and its head keeps its quotes (OPR-2).
        Admission derives S from the citations with ``plan_stating_spans``.

        Raises:
            AMLError: ``binding_mismatch`` for an Item already gone.
        """
        versions: dict[tuple[str, str], ForgetVersionSelection] = {}
        entities: set[str] = set()
        cited: list[ForgottenCitation] = []
        for memory_id in memory_ids:
            item = self._writer.get_row("item", memory_id)
            if item is None:
                raise AMLError("binding_mismatch", "A forgotten Item went before its tombstones")
            revisions = sorted(self._pages(RevisionRows(memory_id=memory_id)),
                               key=lambda row: row["version"])
            selection = _plan(item, revisions)
            for version in selection.versions:
                snapshot = version.snapshot
                versions.setdefault((snapshot.entity_id, snapshot.statement_digest), version)
            cited.extend(selection.cited)
            entities.update(
                entity for revision in revisions
                for entity in _entities(MemorySnapshotV2.model_validate(revision["snapshot"]))
                if entity != "user"
            )
        verdicts = frozenset(ref for memory_id in memory_ids for ref in _verdicts(
            self._pages(RevisionRows(memory_id=memory_id))))
        for memory_id in reworded:
            revisions = [RevisionRowV2.model_validate(row)
                         for row in self._pages(RevisionRows(memory_id=memory_id))]
            for version in plan_earlier_wordings(revisions, verdicts):
                snapshot = version.snapshot
                versions.setdefault((snapshot.entity_id, snapshot.statement_digest), version)
        return [versions[pair] for pair in sorted(versions)], sorted(entities), cited

    def _reworded_by(self, txn: TxnRows, memory_id: str, verdicts: Set[str]) -> bool:
        """Whether one of an Item's versions is a rewording under one of ``verdicts``."""
        return bool(verdicts) and any(
            revision["reason_code"] == "rewording"
            and revision["snapshot"].get("verdict_ref") in verdicts
            for revision in self._revisions(txn, memory_id))

    def _reworded(self, scope: ScopeV1, txn: TxnRows, forgotten: Sequence[str]) -> tuple[str, ...]:
        """The surviving Items a forgotten correction reworded (C.9.4a; OPR-2).

        A rewording carries the verdict of the correction that caused it. An active Item
        outside the forgotten set is reworded when its head is such a rewording, or when
        a forgotten version's ``contradicts`` link targets it and one of its versions is
        (a head revised again after its rewording is found only through that link).
        Read under the Writer mutex in the admission transaction and frozen in the fence.

        Raises:
            AMLError: ``budget_exceeded`` past ``MAX_FORGET_CHAIN`` Items.
        """
        verdicts = frozenset(ref for memory_id in forgotten
                             for ref in _verdicts(self._revisions(txn, memory_id)))
        if not verdicts:
            return ()
        excluded = set(forgotten)
        found: set[str] = set()
        for memory_id in forgotten:
            for revision in self._revisions(txn, memory_id):
                for link in txn.links_of(revision["object_id"]):
                    target = link["target_key"]
                    if (link["link_type"] == "contradicts" and target not in excluded
                            and self._reworded_by(txn, target, verdicts)):
                        found.add(target)
        for item in self._pages(ItemRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id)):
            if item["object_id"] in excluded:
                continue
            head = txn.get("revision", item["current_revision_id"])
            if head is not None and head["reason_code"] == "rewording" and (
                    head["snapshot"].get("verdict_ref") in verdicts):
                found.add(item["object_id"])
        active = sorted(memory_id for memory_id in found
                        if (item := txn.get("item", memory_id)) is not None
                        and item["lifecycle_state"] == "active")
        if len(active) > MAX_FORGET_CHAIN:
            raise AMLError("budget_exceeded", "The forget's reworded Items exceed their bound")
        return tuple(active)

    def _reworded_keepers(self, txn: TxnRows, reworded: Sequence[str]) -> list[KeeperGroup]:
        """The readable groups of each reworded head, kept as its own support (path f)."""
        read = readability(txn)
        keepers: list[KeeperGroup] = []
        for memory_id in reworded:
            item = txn.get("item", memory_id)
            head = None if item is None else txn.get("revision", item["current_revision_id"])
            if head is None:
                continue
            snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
            if snapshot.erased or not read.p_ver(snapshot):
                continue
            for group in snapshot.support_groups:
                if read.p_group(snapshot.quote_spans, group):
                    keepers.append(KeeperGroup(
                        frozenset(), tuple(snapshot.quote_spans[index] for index in group),
                        sensitive=bool(snapshot.sensitivity_classes), reworded=True))
        return keepers

    def _withdrawn(self, txn: TxnRows, forgotten: Set[str],
                   cited: Sequence[ForgottenCitation]) -> frozenset[tuple[str, str]]:
        """The names a forgotten version gives that no surviving memory gives (D.3 P1 #6)."""
        given = {pair for version in cited for pair in version.names}
        surviving = {(row["entity_id"], row["name_norm"])
                     for entity in {entity for entity, _ in given}
                     for row in txn.names_of(entity)
                     if row["naming_memory_id"] not in forgotten}
        return frozenset(given - surviving)

    def _keeper_groups(
        self, scope: ScopeV1, txn: TxnRows, forgotten: Set[str],
        cited: Sequence[ForgottenCitation],
    ) -> list[KeeperGroup]:
        """The readable groups of the surviving memories a forget may keep (O11).

        A survivor is an ``active`` Item outside the forgotten set whose head
        passes P-VER now and cites an Evidence row a forgotten version cites. A
        survivor a recorded link joins to a forgotten Item, in either direction,
        is none: a memory the forgotten memory contradicts, corrects or supersedes
        carries the very description the forget is about (a corrected kind, C.9.5).
        Each group of a survivor that passes P-GROUP is a keeper group, marked
        ``bare`` when the head anchors one entity alone, names it by type only
        (``named_by_type``); ``plan_stating_spans`` keeps only exact name subspans
        inside its quotes, using sensitivity from each snapshot. Sensitive heads keep
        nothing; bare names keep nothing after any sensitive forgotten version.
        The type must be exactly an allowlisted neutral kind, never a role or clause.
        Bare names need no anchor on the forgotten version: a split
        user-only fact can still quote the name. Each keeper also carries its
        head's statement, anchors and quoted text for the detail keeper
        (``detail_names``) and the referent keeper (``referent_spans``), which a
        survivor naming nothing may also be. A non-naming survivor also carries its
        primary entity's P-NAME-readable names from surviving naming memories, only for
        the exact descriptor-detail fallback. Read under the Writer mutex in the admission
        transaction, so the plan frozen in the fence cannot drift.
        """
        anchored = {anchor for version in cited
                    for anchor in (version.entity_id, *version.about)} - {"", "user"}
        evidence = {span.evidence_id for version in cited
                    for group in version.groups for span in group}
        if not evidence:
            return []
        runs = {entity: {tuple(row["name_norm"].split()) for row in txn.names_of(entity)}
                | {tuple(name.split()) for version in cited for owner, name in version.names
                   if owner == entity}
                for entity in anchored}
        runs["user"] = {("user",)}
        joined = set(forgotten)
        for memory_id in forgotten:
            for revision in self._revisions(txn, memory_id):
                joined.update(link["target_key"] for link in txn.links_of(revision["object_id"]))
            for kind in ("supersedes", "contradicts"):
                for link in txn.links_to(kind, memory_id):
                    source = txn.get("revision", link["source_revision_id"])
                    if source is not None:
                        joined.add(source["memory_id"])
        read = readability(txn)
        reference_cache: dict[str, frozenset[tuple[str, str]]] = {}

        def reference_names(entity: str) -> frozenset[tuple[str, str]]:
            if entity in reference_cache:
                return reference_cache[entity]
            found: set[tuple[str, str]] = set()
            if entity not in ("", "user"):
                for raw in txn.names_of(entity):
                    row = EntityNameRowV1.model_validate(raw)
                    if row.naming_memory_id in joined:
                        continue
                    naming = txn.get("item", row.naming_memory_id)
                    if naming is None:
                        continue
                    naming_item = ItemRowV2.model_validate(naming)
                    head = txn.get("revision", naming_item.current_revision_id)
                    if head is not None and read.p_name(
                            row, naming_item, MemorySnapshotV2.model_validate(head["snapshot"])):
                        found.add((entity, row.name_norm))
            result = frozenset(found)
            reference_cache[entity] = result
            return result

        keepers: list[KeeperGroup] = []
        for item in self._pages(ItemRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id)):
            if item["object_id"] in joined or item["lifecycle_state"] != "active":
                continue
            head = txn.get("revision", item["current_revision_id"])
            if head is None or evidence.isdisjoint(head["evidence_ids"]):
                continue
            snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
            if not read.p_ver(snapshot):
                continue
            names = frozenset((entry.entity_id, name_norm(entry.name))
                              for entry in snapshot.naming if entry.entity_id not in ("", "user"))
            references: frozenset[tuple[str, str]] = (
                reference_names(snapshot.entity_id) if not snapshot.naming else frozenset()
            )
            sole = snapshot.anchors[0] if len(snapshot.anchors) == 1 else None
            given = {name for entity, name in names if entity == sole}
            if given and not named_by_type(snapshot.statement, given, {
                    run for entity, found in runs.items() if entity != sole for run in found}):
                given = set[str]()
            persons = frozenset(entity for entity, _ in names | references
                                if _is_person(txn, entity))
            stated = support_words(snapshot.statement)
            if snapshot.time_expr.kind != "none":
                stated |= support_words(snapshot.time_expr.text)
            for group in snapshot.support_groups:
                if not read.p_group(snapshot.quote_spans, group):
                    continue
                spans = tuple(snapshot.quote_spans[index] for index in group)
                quoted = tuple(_quoted_text(txn, span) for span in spans)
                bare = bool(given)
                keepers.append(KeeperGroup(
                    names, spans, sole if bare else None, bool(snapshot.sensitivity_classes),
                    snapshot.statement, snapshot.anchors,
                    tuple(text for text in quoted if text is not None), persons, stated,
                    tuple(snapshot.embedding), snapshot.embedding_space,
                    speakers=tuple(_speaker_words(txn, span) for span in spans),
                    reference_names=references,
                    entity_type=(entity["entity_type"] if (
                        entity := txn.get("entity", snapshot.entity_id)) is not None else ""),
                ))
        return keepers

    def _unit(self, txn: TxnRows, memory_ids: Sequence[str]) -> list[str]:
        """The retention unit's Items, in discovery order (D.3 P2; A.3.3).

        The forgotten Item, its chain (``_chain``, frozen in the fence) and,
        transitively, every ``historical`` Item whose ``corrects_history`` link
        targets one of their revisions (``cascade_source_item``); an Item already
        gone is skipped.

        Raises:
            AMLError: ``budget_exceeded`` past ``_CASCADE_LIMIT`` historical Items.
        """
        unit: list[str] = []
        pending, seen = list(reversed(memory_ids)), set[str]()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            if len(seen) >= len(memory_ids) + _CASCADE_LIMIT - 1:
                raise AMLError("budget_exceeded", "The forget cascade exceeds its bound")
            seen.add(current)
            if txn.get("item", current) is None:
                continue
            unit.append(current)
            for revision in self._revisions(txn, current):
                for link in txn.links_to("corrects_history", revision["object_id"]):
                    source = txn.get("revision", link["source_revision_id"])
                    if source is not None:
                        pending.append(source["memory_id"])
        return unit

    def _incoming(self, txn: TxnRows, unit: Sequence[str], *, limit: int) -> list[Row]:
        """Up to ``limit`` ``supersedes``/``contradicts`` links targeting the unit's Items.

        Other Items carry them, however many, so they are not the unit's own
        rows: P2 deletes them in batches before the unit's transaction, and
        each batch takes the first ones still stored.
        """
        found: list[Row] = []
        for memory_id in unit:
            for link_type in ("supersedes", "contradicts"):
                for link in self._pages(LinkTargetRows(
                        scope_kind=txn.scope_kind, scope_id=txn.scope_id,
                        link_type=link_type, target_key=memory_id)):
                    found.append(link)
                    if len(found) == limit:
                        return found
        return found

    def _preflight(
        self, scope: ScopeV1, memory_ids: Sequence[str], evidence: Set[str], entities: int,
        now: int,
    ) -> None:
        """Refuse, before any write, a forget whose P2 could never finish (D.3 P1 #3).

        P2 batches S, the dependents, the incoming links and the expiries, but
        the unit's own rows leave in one transaction with at most ``entities``
        entity erasures, and their records and deliveries are frozen into the
        fence. That transaction is built here on rows never committed: past
        ``MAX_TX_ROWS`` or a fence bound, P2 would refuse it on every resume.

        Raises:
            AMLError: ``budget_exceeded`` past the cascade, transaction, record
                or delivery bound.
        """
        txn = self._txn(scope, None, now)
        self._delete_unit(scope, txn, self._unit(txn, memory_ids), evidence)
        if len(txn.deleted) + entities + _UNIT_OVERHEAD > MAX_TX_ROWS:
            raise AMLError("budget_exceeded", "The forgotten Item's own rows exceed a transaction")

    def _delete_unit(
        self, scope: ScopeV1, txn: TxnRows, unit: Sequence[str], evidence: Set[str],
    ) -> tuple[dict[str, int], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Delete the unit's own rows in ``txn``; find the records that leave with them.

        Returns:
            The deletion counts, then the Artifacts, Controls and Search
            deliveries the purge batches delete.

        Raises:
            AMLError: ``budget_exceeded`` past the record or delivery bounds.
        """
        candidates = list(self._pages(CandidateRows(
            scope_kind=scope.scope_kind, scope_id=scope.scope_id)))
        removed, deleted = self._delete_item(txn, unit, candidates)
        artifacts, controls = self._retention_records(scope, deleted, candidates)
        recalls = self._deliveries(scope, set(unit), evidence)
        if len(recalls) > MAX_FORGET_RECALLS:
            raise AMLError("budget_exceeded", "The forget's Search deliveries exceed their bound")
        return removed, artifacts, controls, recalls

    def _delete_item(
        self, txn: TxnRows, unit: Sequence[str], candidates: Sequence[Row],
    ) -> tuple[dict[str, int], list[Row]]:
        """Delete the unit's own rows: each Item with its revisions and their rows (D.3 P2).

        The rows are the Item, its revisions, their outgoing links, the
        ``corrects_history`` links targeting them (the unit's own), its derived
        names and the candidates naming it or one of its revisions. The links
        other Items carry to it left in earlier batches (``_incoming``).

        Returns:
            The deletion counts and the deleted candidate rows.
        """
        counts = {"item": 0, "revision": 0, "link": 0, "entity_name": 0, "candidate": 0}
        deleted: list[Row] = []
        for memory_id in unit:
            revisions = self._revisions(txn, memory_id)
            revision_ids = {revision["object_id"] for revision in revisions}
            for revision in revisions:
                for link in txn.links_of(revision["object_id"]):
                    txn.delete("link", link["object_id"])
                    counts["link"] += 1
                for link in txn.links_to("corrects_history", revision["object_id"]):
                    if txn.get("link", link["object_id"]) is not None:
                        txn.delete("link", link["object_id"])
                        counts["link"] += 1
            for revision in revisions:
                for entity_id in sorted({e["entity_id"] for e in revision["snapshot"]["naming"]}):
                    for row in txn.names_of(entity_id):
                        if row["naming_memory_id"] == memory_id:
                            txn.delete("entity_name", row["object_id"])
                            counts["entity_name"] += 1
            for candidate in candidates:
                if (candidate["memory_id"] == memory_id
                        or candidate["revision_id"] in revision_ids) and txn.get(
                            "candidate", candidate["object_id"]) is not None:
                    txn.delete("candidate", candidate["object_id"])
                    deleted.append(candidate)
                    counts["candidate"] += 1
            for revision in revisions:
                txn.delete("revision", revision["object_id"])
                counts["revision"] += 1
            txn.delete("item", memory_id)
            counts["item"] += 1
        return counts, deleted

    def _retention_records(
        self, scope: ScopeV1, deleted: Sequence[Row], candidates: Sequence[Row],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The Artifacts and Controls that leave with the deleted candidates (D.3 P2).

        Each deleted position takes its ProposalV2 and the records of every
        review of it and of every confirmation vote (C.8 #9), under any reviewing
        owner: the judge result indexes, their neighbour snapshots, the member
        outputs they own and their ``judge_output`` locators. A call manifest
        leaves once no member it lists keeps its snapshot; an owner's extraction
        reply, context record and embedding batches leave once no surviving
        candidate of that owner refers to them;
        a manifest takes its parts. A record a surviving row still refers to
        stays, internal and unreadable (E.2.4 ``provider_records``); content-free
        Runs keep their inputs and result locators.

        Returns:
            The sorted Artifact ids and the sorted Control keys.

        Raises:
            AMLError: ``binding_mismatch`` for a review record of another
                shape; ``budget_exceeded`` past the fence's record bounds.
        """
        writer = self._writer
        artifacts: set[str] = set()
        snapshots: set[str] = set()
        calls: set[str] = set()
        controls: set[str] = set()
        shared: dict[str, set[str]] = {}
        for candidate in deleted:
            owner = cast(str, candidate["owner_run_id"])
            artifacts.add(candidate["proposal_ref"])
            shared.setdefault(owner, set()).update(self._proposal_records(candidate))
            results = {candidate["judge_result_ref"]} - {None}
            for target in judgment_targets((owner, candidate["unit_no"], candidate["position"])):
                for child in self._pages(ScopedRunRows(
                        scope_kind=scope.scope_kind, scope_id=scope.scope_id,
                        run_kind="evaluate", target_id=target)):
                    results.add(child["result_ref"])
                    selected = self._record(child["input_ref"], "run_input", JudgmentRunInputV1)
                    if selected is not None:
                        snapshots.add(selected.snapshot_ref)
            for reference in results - {None}:
                result = self._record(reference, "result_index", JudgeResultIndexV8)
                if result is None:
                    continue
                artifacts.add(reference)
                snapshots.add(result.snapshot_ref)
                calls.add(result.call_ref)
                if result.output_ref is not None and result.borrowed_from is None:
                    artifacts.add(result.output_ref)
                key = "judge_output:{}:{}".format(scope.scope_id, writer.bind_content(
                    LOCATOR_PURPOSE, [result.binding_digest, result.member_input_digest],
                ))
                locator = writer.get_control(key)
                if locator is not None and locator["value"]["result_ref"] == reference:
                    controls.add(key)
        artifacts |= snapshots
        for reference in calls:
            call = self._record(reference, "decision_set", JudgeCallManifestV1)
            if call is not None and all(
                member in snapshots or writer.get_row("artifact", member) is None
                for member in call.member_snapshot_refs
            ):
                artifacts.add(reference)
        gone = {row["object_id"] for row in deleted}
        for owner, records in shared.items():
            kept: set[str] = set()
            for candidate in candidates:
                if candidate["owner_run_id"] == owner and candidate["object_id"] not in gone:
                    kept |= self._proposal_records(candidate)
            artifacts |= records - kept
        for reference in sorted(artifacts):
            row = writer.get_row("artifact", reference)
            if row is not None and row["artifact_kind"] == "manifest":
                artifacts.update(ManifestV1.model_validate_json(row["body"]).parts)
        if len(artifacts) > MAX_FORGET_ARTIFACTS or len(controls) > MAX_FORGET_CONTROLS:
            raise AMLError("budget_exceeded", "The forget retention records exceed their bounds")
        return tuple(sorted(artifacts)), tuple(sorted(controls))

    def _deliveries(
        self, scope: ScopeV1, items: Set[str], evidence: Set[str],
    ) -> tuple[str, ...]:
        """The scope's Search deliveries that returned one of ``items`` or ``evidence``.

        A delivery keeps its query, options and returned identities (E.1.3); one
        that returned a forgotten Item or an Evidence row whose spans the forget
        withholds belongs to the retention unit (D.3 P2). Its interactions cannot
        exist: the Writer admits no ``interaction`` record.

        Returns:
            The delivery ids, sorted.
        """
        return tuple(
            row["object_id"] for row in self._pages(RecallDeliveries(
                scope_kind=scope.scope_kind, scope_id=scope.scope_id))
            if any(hit["object_id"] in (items if hit["kind"] == "memory" else evidence)
                   for hit in row["returned"])
        )

    def _proposal_records(self, candidate: Row) -> set[str]:
        """The owner records a candidate refers to: its reply, context and embedding batches."""
        records = {cast(str, candidate["extraction_ref"])}
        proposal = self._record(candidate["proposal_ref"], "proposal_snapshot", ProposalV2)
        if proposal is not None:
            records |= {proposal.extraction_ref, proposal.context_ref} | {
                use.ref for use in (proposal.statement_embedding, proposal.cue_embedding)
                if use is not None
            }
        return records

    def _record[T](self, reference: str | None, kind: str, model: type[T]) -> T | None:
        """One stored logical record of ``model``, or None when absent (already erased).

        Raises:
            AMLError: ``binding_mismatch`` for a record of another variant.
        """
        if reference is None or self._writer.get_row("artifact", reference) is None:
            return None
        record = self._writer.read_logical(reference, kind)
        if not isinstance(record, model):
            raise AMLError("binding_mismatch", "A retention record has another variant")
        return record

    def _purge(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        """P2: delete the frozen retention records, one bounded batch per transaction.

        The fence's progress is written in the transaction that deletes the
        batch; a record already gone is skipped, so a resumed batch deletes
        nothing twice.
        """
        del scope
        writer = self._writer
        records = [("artifact", reference) for reference in fence.artifacts] + [
            ("control", key) for key in fence.controls
        ] + [("recall", recall) for recall in fence.recalls]
        done = fence.purged + _PURGE_BATCH
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            if writer.get_control(cast(str, control["key"])) != control:
                raise AMLError("binding_mismatch", "The erasure fence changed under its owner")
            removed: list[tuple[str, dict[str, Any]]] = []
            for kind, identity in records[fence.purged:done]:
                row = (writer.get_control(identity) if kind == "control"
                       else writer.get_row(kind, identity))
                if row is not None:
                    removed.append((kind, row))
            counts = fence.counts.model_copy(update={
                kind: getattr(fence.counts, kind) + sum(k == kind for k, _ in removed)
                for kind in ("artifact", "control", "recall")
            })
            done = min(done, len(records))
            progress = self._fence_row(
                fence, control, "purged" if done == len(records) else "deleted", counts,
                self._clock(), purged=done,
            )
            _, plan = writer._prepare_rows([progress], None)  # pyright: ignore[reportPrivateUsage]
            if removed:
                writer._plan_deletes(plan, removed)  # pyright: ignore[reportPrivateUsage]
            writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]

    # P3 -------------------------------------------------------------------------------
    def final_scan(self, scope: ScopeV1, fence: ErasureFenceV1) -> FinalScan:
        """FS1–FS6 over durable storage and the published read view (D.3 P3).

        FS5 reruns P2's choices on a transaction that is never committed: no
        dependent may still need a requote, no readable survivor a keyword-only revision
        (D.7.3 #4b) and no entity an erasure.
        """
        writer = self._writer
        txn = self._txn(scope, fence.worker_run_id, self._clock())
        targets = (fence.memory_id, *fence.chain_ids)
        revisions = [revision for memory_id in targets
                     for revision in self._pages(RevisionRows(memory_id=memory_id))]
        links = sum(
            len(txn.links_of(revision["object_id"])) for revision in revisions
        ) + sum(len(txn.links_to(kind, memory_id))
                for memory_id in targets for kind in ("supersedes", "contradicts"))
        records = sum(1 for candidate in self._pages(CandidateRows(
            scope_kind=scope.scope_kind, scope_id=scope.scope_id))
            if candidate["memory_id"] in targets)
        records += sum(1 for reference in fence.artifacts
                       if writer.get_row("artifact", reference) is not None)
        records += sum(1 for key in fence.controls if writer.get_control(key) is not None)
        records += sum(1 for recall in fence.recalls
                       if writer.get_row("recall", recall) is not None)
        records += len(self._deliveries(scope, set(targets), fence.evidence_ids))
        unwithheld = 0
        for span in fence.spans:
            row = writer.get_row("evidence", span.evidence_id)
            # FS3: covered unless its row is erased (a privacy erase clears the ranges, D.7).
            if row is not None and row["state"] != "erased" and not any(
                held["byte_start"] <= span.byte_start and span.byte_end <= held["byte_end"]
                for held in row["withheld_spans"]
            ):
                unwithheld += 1
        read = readability(txn)
        forgotten: dict[str, list[QuoteSpanV1]] = {}
        for span in fence.spans:
            forgotten.setdefault(span.evidence_id, []).append(span)
        pending_unusable = 0
        for candidate in self._pages(CandidateRows(
                scope_kind=scope.scope_kind, scope_id=scope.scope_id, status="pending")):
            if fence.evidence_ids.isdisjoint(candidate["evidence_ids"]):
                continue
            groups = self._groups_of(candidate["proposal_ref"])
            if _overlaps(groups, forgotten) and not _usable(groups, read):
                pending_unusable += 1
        missing = sum(1 for tombstone in fence.tombstone_ids
                      if writer.get_row("tombstone", tombstone) is None)
        words = self._forgotten_words(txn, fence)
        statements = self._tombstone_statements(txn, fence)
        dependents = sum(
            1 for memory_id in self._dependents(scope, txn, fence, statements=statements)
            if self._settle(scope, txn, fence, read, memory_id, words, statements))
        entities = len(erase_unanchored_entities(
            txn, fence.entity_ids, participant_has_source=participant_has_source(txn),
        ))
        residual = sum(1 for memory_id in targets
                       if self._publisher.retains_memory(scope, memory_id))
        return FinalScan(
            item=sum(1 for memory_id in targets if writer.get_row("item", memory_id) is not None),
            revision=len(revisions), link=links, records=records, unwithheld_spans=unwithheld,
            pending_unusable=pending_unusable, tombstones_missing=missing,
            dependents=dependents, entities=entities, read_view=residual,
        )

    def _scan(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        scan = self.final_scan(scope, fence)
        if not scan.verified():
            raise AMLError("service_unhealthy", "The forget final scan found residual content")
        digest = self._writer.bind_content("forget-final-scan", {
            "memory": fence.memory_id, "chain": list(fence.chain_ids),
            "reworded": list(fence.reworded_ids),
            "tombstones": list(fence.tombstone_ids),
            "spans": [span.model_dump(mode="json") for span in fence.spans],
            "records": [*fence.artifacts, *fence.controls, *fence.recalls],
            "commit_seq": self._writer.counter("commit_seq"),
            "tombstone_seq": self._writer.counter(f"tombstone_seq:{scope.scope_id}"),
        })
        self._progress(scope, fence, control, "scanned", fence.counts, scan_digest=digest)

    # P4 -------------------------------------------------------------------------------
    def _receipts(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        writer = self._writer
        prepare, final = self._receipt_pair(scope, fence)
        ticket = writer.ticket(fence.controller_run_id)
        now = self._clock()
        prepare_row = writer.put_forget_receipt(ticket, "receipt:forget_prepare", prepare,
                                                created_at=now)
        final = final.model_copy(update={"prepare_ref": prepare_row})
        final = ForgetReceiptV3.model_validate(final.model_dump(mode="json"))
        verify_forget_receipt_pair(prepare, final, prepare_ref=prepare_row)
        final_row = writer.put_forget_receipt(ticket, "receipt:forget_final", final,
                                              created_at=now)
        index_row = writer.put_forget_receipt(
            ticket, "result_index:erasure_receipt",
            ErasureReceiptResultIndexV1(erasure_receipt_ref=final_row), created_at=now,
        )
        self._progress(scope, fence, control, "receipted", fence.counts,
                       prepare_ref=prepare_row, final_ref=final_row, index_ref=index_row)

    def _release(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        writer, publisher = self._writer, self._publisher

        def barrier() -> None:
            writer.release_erased_caches()

        privacy_quiescence(writer, publisher, scope, barrier=barrier)
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            now = self._clock()
            current = writer.get_control(cast(str, control["key"]))
            if current != control:
                raise AMLError("binding_mismatch", "The erasure fence changed before release")
            rows: list[RowWrite] = []
            for run_id in (fence.worker_run_id, fence.controller_run_id):
                run = dict(cast(Row, writer.get_row("run", run_id)))
                run.update(status="succeeded", stage="receipt", started_at=run["started_at"] or now,
                           updated_at=now, finished_at=now)
                if run_id == fence.controller_run_id:
                    run["result_ref"] = fence.index_ref
                rows.append(RowWrite("run", writer._parse_row("run", run)))  # pyright: ignore[reportPrivateUsage]
            _, plan = writer._prepare_rows(rows, None)  # pyright: ignore[reportPrivateUsage]
            writer._plan_deletes(plan, [("control", dict(control))])  # pyright: ignore[reportPrivateUsage]
            writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
            publisher.publish_erasure(writer, scope, None, writer.counter("commit_seq"),
                                      active=False)

    # Shared ---------------------------------------------------------------------------
    def _receipt_pair(
        self, scope: ScopeV1, fence: ErasureFenceV1,
    ) -> tuple[ForgetPrepareReceiptV3, ForgetReceiptV3]:
        bind = self._writer.bind_content
        sources: dict[str, list[QuoteSpanV1]] = {}
        for span in fence.spans:
            sources.setdefault(span.evidence_id, []).append(span)
        withheld = sorted(
            (ForgetWithheldSourceV1(
                source_token=bind("forget-source", {"scope": scope.scope_id, "evidence": ev}),
                spans_digest=bind("forget-spans", [s.model_dump(mode="json") for s in spans]),
            ) for ev, spans in sources.items()),
            key=lambda source: str(source.source_token),
        )
        common: dict[str, Any] = {
            "target_digest": bind("forget-target", {"scope": scope.scope_id,
                                                   "memory": fence.memory_id}),
            "worker_token": bind("forget-worker", fence.worker_run_id),
            "maintenance_handoff_digest": bind("forget-handoff", {"maintenance": None}),
            "plan_digest": bind("forget-plan", {
                "chain": list(fence.chain_ids), "reworded": list(fence.reworded_ids),
                "tombstones": list(fence.tombstone_ids), "entities": list(fence.entity_ids),
                "spans": [span.model_dump(mode="json") for span in fence.spans],
            }),
            "erase_epoch": fence.erase_epoch + 1,
            "tombstone_seq": fence.tombstone_seq + 1,
            "intended_generation": fence.commit_seq + 1,
            "intended_scope_seq": fence.scope_seq + 1,
            "withheld_sources": withheld,
        }
        prepare = ForgetPrepareReceiptV3.model_validate(common)
        assert fence.scan_digest is not None
        final = ForgetReceiptV3.model_validate({
            **common, "prepare_ref": "art_" + "0" * 43, "counts": fence.counts,
            "final_scan_digest": fence.scan_digest,
            "internal_retention": ForgetInternalRetentionV1.model_validate({
                "tombstones": len(fence.tombstone_ids),
                "tombstones_created": len(fence.created_tombstone_ids),
                "tombstone_set_digest": bind("forget-tombstones", list(fence.tombstone_ids)),
            }),
        })
        return prepare, final

    def _outcome(self, controller_id: str) -> ForgetOutcome:
        writer = self._writer
        run = writer.get_row("run", controller_id)
        if run is None or run["status"] != "succeeded" or run["result_ref"] is None:
            raise AMLError("service_unhealthy", "The forget controller has no verified receipt")
        index = writer.read_logical(run["result_ref"], "result_index")
        receipt_ref = ErasureReceiptResultIndexV1.model_validate(
            index.model_dump(mode="json"),
        ).erasure_receipt_ref
        body = writer.read_logical(receipt_ref, "receipt")
        final = ForgetReceiptV3.model_validate(body.model_dump(mode="json"))
        prepare = writer.read_logical(final.prepare_ref, "receipt")
        verify_forget_receipt_pair(
            ForgetPrepareReceiptV3.model_validate(prepare.model_dump(mode="json")), final,
            prepare_ref=final.prepare_ref,
        )
        return ForgetOutcome(controller_id, final.prepare_ref, receipt_ref, final)

    def _commit(
        self, scope: ScopeV1, ticket: WriteTicket, txn: TxnRows, fence: ErasureFenceV1,
        control: Row, phase: ErasurePhase, counts: ForgetCountsV1, *, tombstones: bool = False,
        holds: RowWrite | None = None, **update: Any,
    ) -> None:
        writer = self._writer
        progress = self._fence_row(
            fence, control, phase, counts, txn.now, by=ticket.run_id, **update,
        )
        with writer.mutex:
            # Built outside the mutex: another scope may have committed since (TxnRows.stamp).
            txn.stamp()
            writer.commit_rows(
                ticket, txn.row_writes(), deletes=txn.row_deletes(),
                bumps=frozenset({"tombstone_seq", "erase_epoch"}) if tombstones else frozenset(),
                updated_at=txn.now, progress=[progress, *([] if holds is None else [holds])],
                publish=lambda committed: self._publisher.publish_commit(writer, committed),
            )

    def _progress(
        self, scope: ScopeV1, fence: ErasureFenceV1, control: Row, phase: ErasurePhase,
        counts: ForgetCountsV1, **update: Any,
    ) -> None:
        writer = self._writer
        with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
            if writer.get_control(cast(str, control["key"])) != control:
                raise AMLError("binding_mismatch", "The erasure fence changed under its owner")
            row = self._fence_row(fence, control, phase, counts, self._clock(), **update)
            _, plan = writer._prepare_rows([row], None)  # pyright: ignore[reportPrivateUsage]
            writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]

    def _fence_row(
        self, fence: ErasureFenceV1, control: Row, phase: ErasurePhase, counts: ForgetCountsV1,
        now: int, *, by: str | None = None, **update: Any,
    ) -> RowWrite:
        value = ErasureFenceV1.model_validate({
            **fence.model_dump(mode="json"), "phase": phase,
            "counts": counts.model_dump(mode="json"), **update,
        })
        return RowWrite("control", self._writer.materialize_control(
            cast(str, control["key"]), value.model_dump(mode="json"), control["version"] + 1,
            now, by or fence.controller_run_id,
        ))

    def _session_token(self, scope: ScopeV1, client_session_id: str) -> ContentBindingDigest:
        """The keyed session binding; no raw client session enters a Control."""
        return self._writer.bind_content(
            "forget-session", {"scope": scope.scope_id, "session": client_session_id},
        )

    def _refuse_fenced(self, scope: ScopeV1) -> None:
        for row in self._writer.indexed_reader().scope_fences(scope.scope_id):
            if cast(str, row["key"]).startswith("erasure:"):
                raise AMLError("erasure_in_progress", "Another erasure holds the scope")

    def _refuse_maintenance(self, scope: ScopeV1) -> None:
        """Refuse admission behind a maintenance owner.

        D.3 P1 #2 takes maintenance over; that handoff has no producer yet
        (read-Q1), so admission refuses instead of writing a state no reader
        validates.
        """
        for row in self._writer.indexed_reader().scope_fences(scope.scope_id):
            if cast(str, row["key"]).startswith(f"maintenance:{scope.scope_id}:"):
                raise AMLError("index_rebuilding", "Maintenance holds the scope")

    def _live_owner(self, scope: ScopeV1) -> tuple[Row, Row] | None:
        """The scope's live finalize lease and its owner Run, or None (D.3 P1 #7).

        A ``released`` cycle has no owner left to cancel: its history stays
        provable after the forget (``released_history``), so only a
        ``preparing`` or ``held`` cycle is cancelled.
        """
        lease = self._writer.get_control(f"lease:finalize:{scope.scope_id}")
        if lease is None or lease["value"].get("state") not in {"preparing", "held"}:
            return None
        owner = self._writer.get_row("run", lease["value"]["owner_run_id"])
        if owner is None:
            raise AMLError("reference_missing", "A live finalize lease lost its owner")
        return lease, owner

    def _cancellation(self, lease: Row, owner: Row, controller_id: str, now: int) -> list[RowWrite]:
        """The live cycle's cancellation rows in the admission transaction (D.3 P1 #7, D.5).

        The lease becomes ``cancelled`` under the controller, and the owner Run,
        unless already ended, fails ``erase_fenced``: neither can act again, so a
        late provider reply is neither stored nor settled (INV:I08).
        """
        writer = self._writer
        value = lease["value"]
        cancelled = CancelledFinalizeLeaseV1(
            state="cancelled", lease_generation=value["lease_generation"],
            owner_run_id=value["owner_run_id"], writer_epoch=value["writer_epoch"],
            heartbeat_at=value["heartbeat_at"], cancelled_from=value["state"],
            cancelled_at=now, cancel_reason="hard_erase",
        )
        rows = [RowWrite("control", writer.materialize_control(
            cast(str, lease["key"]), cancelled.model_dump(mode="json"), lease["version"] + 1,
            now, controller_id,
        ))]
        if owner["status"] not in _ENDED:
            rows.append(RowWrite("run", writer._parse_row("run", _ended(owner, now))))  # pyright: ignore[reportPrivateUsage]
        return rows

    def _retire(self, scope: ScopeV1, owner_run_id: str) -> None:
        """Tell the cancelled owner's pool invocation to stop at its next check (D.5)."""
        from aml.service.bench_workers import BenchWork

        pool = self._writer._finalize_pool  # pyright: ignore[reportPrivateUsage]
        if pool is not None:
            pool.retire_owner(BenchWork(scope.scope_id, owner_run_id, "finalize"))

    def _current(self, scope: ScopeV1, memory_id: str) -> tuple[Row, Row]:
        item = self._writer.get_row("item", memory_id)
        if item is None or (item["scope_kind"], item["scope_id"]) != (
                scope.scope_kind, scope.scope_id):
            raise AMLError("scope_denied", "The memory is not in this scope")
        head = self._writer.get_row("revision", item["current_revision_id"])
        if head is None:
            raise AMLError("service_unhealthy", "The memory lost its current revision")
        return item, head

    def _selection(
        self, scope: ScopeV1, principal_id: str, session: str, item: Row, head: Row,
        erase_epoch: int,
    ) -> ContentBindingDigest:
        return self._writer.bind_content("forget-selection", {
            "scope": scope.scope_id, "principal": principal_id, "session": session,
            "memory": item["object_id"], "revision": head["object_id"],
            "content_hash": head["content_hash"], "erase_epoch": erase_epoch,
        })

    def _tombstone_ids(
        self, scope: ScopeV1, versions: Iterable[Any],
    ) -> tuple[dict[tuple[str, str], str], set[str]]:
        """One tombstone per distinct pair, reusing a stored one (A.3.8 unique key)."""
        ids: dict[tuple[str, str], str] = {}
        created: list[str] = []
        for version in versions:
            snapshot = version.snapshot
            pair = (snapshot.entity_id, snapshot.statement_digest)
            existing = next((
                row["object_id"] for row in self._pages(TombstoneRows(
                    scope_kind=scope.scope_kind, scope_id=scope.scope_id,
                    statement_digest=snapshot.statement_digest))
                if row["entity_id"] == snapshot.entity_id
            ), None)
            if existing is None:
                existing = new_ulid()
                created.append(existing)
            ids[pair] = existing
        # Created ids are handed out in selection order, sorted (``_creations``).
        ordered = sorted(created)
        rename = dict(zip(created, ordered, strict=True))
        return {pair: rename.get(value, value) for pair, value in ids.items()}, set(ordered)

    def _versions(self, fence: ErasureFenceV1) -> list[Row]:
        """The newest revision of each forgotten pair, from durable storage."""
        versions, _, _ = self._chain_plan((fence.memory_id, *fence.chain_ids),
                                          fence.reworded_ids)
        rows = [self._writer.get_row("revision", version.revision_id) for version in versions]
        if any(row is None for row in rows):
            raise AMLError("binding_mismatch", "A forgotten version went before its tombstone")
        return cast(list[Row], rows)

    def _creations(
        self, scope: ScopeV1, fence: ErasureFenceV1, versions: Sequence[Row],
    ) -> list[tuple[Row, str]]:
        """Pair each version without a stored tombstone with its planned new id.

        Admission assigned the sorted created ids to those versions in selection
        order; before P1 (b) commits none exists, so the pairing is reproducible.
        """
        missing = [
            version for version in versions
            if not any(row["entity_id"] == version["snapshot"]["entity_id"]
                       for row in self._pages(TombstoneRows(
                           scope_kind=scope.scope_kind, scope_id=scope.scope_id,
                           statement_digest=version["snapshot"]["statement_digest"])))
        ]
        if len(missing) != len(fence.created_tombstone_ids):
            raise AMLError("binding_mismatch", "The forget plan lost a tombstone")
        return list(zip(missing, fence.created_tombstone_ids, strict=True))

    def _txn(self, scope: ScopeV1, run_id: str | None, now: int) -> TxnRows:
        return TxnRows(self._writer, scope.scope_kind, scope.scope_id,
                       self._writer.counter("commit_seq") + 1, now, run_id or "")

    def _pages(self, query: Any) -> Iterable[Row]:
        return TxnRows(self._writer, "user", "", 0, 0, "").pages(query)

    def _revisions(self, txn: TxnRows, memory_id: str) -> list[Row]:
        """An Item's revisions as ``txn`` leaves them, in id order."""
        return txn.merged("revision", self._pages(RevisionRows(memory_id=memory_id)),
                          lambda row: row["memory_id"] == memory_id)

    def _run_row(
        self, run_id: str, kind: str, mode: str, grant: PrincipalGrant, scope: ScopeV1 | None,
        target: Mapping[str, Any], writer_epoch: int, now: int, intent: str, *,
        controller: str | None = None, erase_epoch: int | None = None,
        consent_epoch: int | None = None,
    ) -> dict[str, Any]:
        # A.3.10 has no canonical forget payload (the admission is the intent's
        # atomic consumption, not a client request); the keyed admission binds
        # the intent, the Run's role, its scope and target, and the worker's
        # controller. Neither Run has a parent: a parent shares its child's exact
        # scope, and the controller is global.
        admission = _admission(kind, mode, grant.principal_id, scope, target, intent, controller)
        family = _family(self._writer, admission)
        digest = self._writer.bind_content("forget-run-admission", {**admission, "run": run_id})
        return {
            "object_id": run_id, "run_kind": kind, "commit_mode": mode, "status": "running",
            "stage": "delete" if kind == "forget" else "dispatch", "attempt": 0,
            "idempotency_key": family, "run_generation": 0, "admission_digest": digest,
            "parent_run_id": None, "principal_id": grant.principal_id,
            **(scope.model_dump(mode="json") if scope is not None else {}),
            **target, "writer_epoch": writer_epoch, "admitted_erase_epoch": erase_epoch,
            "admitted_consent_epoch": consent_epoch, "created_at": now, "started_at": now,
            "updated_at": now,
        }


def _admission(
    kind: str, mode: str, principal_id: str, scope: ScopeV1 | None, target: Mapping[str, Any],
    intent: str, controller: str | None,
) -> dict[str, Any]:
    """The keyed admission of a forget Run: the intent, the Run's role, scope and target."""
    return {
        "run_kind": kind, "commit_mode": mode, "principal_id": principal_id,
        "scope": scope.model_dump(mode="json") if scope is not None else None,
        "target": dict(target), "intent": intent, "controller": controller,
    }


def _family(writer: Writer, admission: Mapping[str, Any]) -> str:
    """The admission family (``idempotency_key``) of a forget Run."""
    return "idem_" + str(writer.bind_content("forget-run-key", dict(admission)))


@dataclass(frozen=True)
class _Forgotten:
    """What a forget's trimmed dependents may not show (D.7.3 #4a).

    Attributes:
        words: ``support_words`` of every span of S.
        statements: The statements of the forget's tombstones.
    """

    words: frozenset[str]
    statements: tuple[str, ...]


@dataclass(frozen=True)
class _Version:
    """A revision row as D.7.3 reads it."""

    row: Row

    @property
    def object_id(self) -> str:
        return cast(str, self.row["object_id"])

    @property
    def version(self) -> int:
        return cast(int, self.row["version"])

    @property
    def reason_code(self) -> RevisionReason:
        return cast(RevisionReason, self.row["reason_code"])

    @property
    def snapshot(self) -> MemorySnapshotV2:
        return MemorySnapshotV2.model_validate(self.row["snapshot"])


def _plan(item: Row, revisions: Sequence[Row]) -> Any:
    """D.3 P1 selection over the complete chain (``plan_forget_admission``)."""
    return plan_forget_admission(
        ItemRowV2.model_validate(item), [RevisionRowV2.model_validate(row) for row in revisions],
    )


def _is_person(txn: TxnRows, entity_id: str) -> bool:
    """Whether an entity's broad type is "person" (participants and people; B.6.3)."""
    row = txn.get("entity", entity_id)
    return row is not None and str(row["entity_type"]).strip().casefold() == "person"


def _speaker_words(txn: TxnRows, span: QuoteSpanV1) -> frozenset[str]:
    """``support_words`` of the speaker a span's Evidence names (``speaker_entity``, the
    caller's own name for whoever wrote the message), or none."""
    row = txn.get("evidence", span.evidence_id)
    speaker = None if row is None else row.get("speaker_entity")
    return support_words(speaker) if isinstance(speaker, str) else frozenset[str]()


def _bare_name_support(
    txn: TxnRows, read: Readability, snapshot: MemorySnapshotV2, forgotten: frozenset[str],
) -> tuple[tuple[QuoteSpanV1, ...], ...]:
    """Requote a neutral naming head only on its still-readable exact name bytes.

    The neutral kind and naming verb need not occur in the source. Every source of
    the original group must remain readable; a name carried by withheld bytes,
    sensitivity or a time-qualified naming head receives no exception. One readable
    name occurrence supports each group, so repeated names cannot exceed span bounds.
    """
    if (snapshot.erased or snapshot.sensitivity_classes or len(snapshot.anchors) != 1
            or snapshot.time_expr.kind != "none"):
        return ()
    names = frozenset((entry.entity_id, name_norm(entry.name)) for entry in snapshot.naming)
    if any(support_words(name) & forgotten for _, name in names):
        return ()
    kept: set[tuple[QuoteSpanV1, ...]] = set()
    for group in snapshot.support_groups:
        spans = tuple(snapshot.quote_spans[index] for index in group)
        if all(read.p_span(span) for span in spans) or any(
                (evidence := read.evidence(span.evidence_id)) is None or not read.p_src(evidence)
                for span in spans):
            continue
        quoted = tuple(_quoted_text(txn, span) for span in spans)
        keeper = KeeperGroup(
            names, spans, bare=snapshot.anchors[0], statement=snapshot.statement,
            anchors=snapshot.anchors, quoted=tuple(text for text in quoted if text is not None),
        )
        found = next((span for span in bare_name_spans(keeper) if read.p_span(span)), None)
        if found is not None:
            kept.add((found,))
    return tuple(sorted(kept, key=lambda group: [
        (span.evidence_id, span.byte_start, span.byte_end) for span in group]))


def _quoted_text(txn: TxnRows, span: QuoteSpanV1) -> str | None:
    """The text a span quotes, or None when its Evidence cannot show it."""
    row = txn.get("evidence", span.evidence_id)
    if row is None:
        return None
    data = str(row["text"]).encode("utf-8")
    start, end = span.byte_start - row["byte_start"], span.byte_end - row["byte_start"]
    if not 0 <= start < end <= len(data):
        return None
    try:
        return data[start:end].decode("utf-8")
    except UnicodeDecodeError:
        return None


def _verdicts(revisions: Iterable[Row]) -> frozenset[str]:
    """The verdicts that admitted an Item's versions (a rewording carries its cause's)."""
    return frozenset(ref for revision in revisions
                     if (ref := revision["snapshot"].get("verdict_ref")) is not None)


def _entities(snapshot: MemorySnapshotV2) -> set[str]:
    """The entities a version names; a content-free (erased) version names none."""
    if snapshot.erased:
        return set()
    return {snapshot.entity_id, *snapshot.about_entity_ids,
            *(entry.entity_id for entry in snapshot.naming)}


def _newly(spans: Sequence[QuoteSpanV1], touched: set[str]) -> int:
    """The spans of a batch whose Evidence row this transaction withheld anew.

    A span an earlier forget already withheld leaves its row unchanged, so the
    receipt counts only the rows and spans this forget changed (E.2.4).
    """
    return sum(1 for span in spans if span.evidence_id in touched)


def _overlaps(
    groups: Sequence[Sequence[QuoteSpanV1]], spans: Mapping[str, Sequence[QuoteSpanV1]],
) -> bool:
    """Whether a quote of ``groups`` intersects one of ``spans`` (by Evidence id; D.2 P-SPAN)."""
    return any(
        held.byte_start < span.byte_end and span.byte_start < held.byte_end
        for group in groups for span in group for held in spans.get(span.evidence_id, ())
    )


def _usable(groups: Sequence[Sequence[QuoteSpanV1]], read: Readability) -> bool:
    """Whether one group has every span passing P-SPAN (D.2 P-GROUP)."""
    return any(read.p_group(tuple(group), tuple(range(len(group)))) for group in groups)


def _adopt(writer: Writer, scope: ScopeV1, controller_id: str) -> None:
    """Adopt a surviving fence's controller and worker under this boot's Writer epoch.

    Boot step 13 (C.13): a fence that survived a restart names Runs of an
    earlier Writer epoch, whose tickets the Writer refuses (``lease_lost``).
    Only ``writer_epoch`` changes, as in the other startup adoptions; the
    fence, its frozen plan and every timestamp stay as they are.

    Raises:
        AMLError: ``binding_mismatch`` for a Run the fence does not name or
            that already ended, ``lease_lost`` for a Run of a later epoch.
    """
    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        control = writer.get_control(fence_key(scope.scope_id, controller_id))
        lease = writer.get_control("lease:writer")
        if control is None or lease is None:
            return
        epoch = lease["value"]["writer_epoch"]
        fence = ErasureFenceV1.model_validate(control["value"])
        rows: list[RowWrite] = []
        for run_id in (fence.worker_run_id, fence.controller_run_id):
            run = writer.get_row("run", run_id)
            if run is None or not owns_erasure_fence(run, control) or run["status"] not in {
                "queued", "running",
            }:
                raise AMLError("binding_mismatch", "A surviving fence lost its running Runs")
            if run["writer_epoch"] > epoch:
                raise AMLError("lease_lost", "A surviving fence names a later Writer epoch")
            if run["writer_epoch"] != epoch:
                rows.append(RowWrite("run", writer._parse_row(  # pyright: ignore[reportPrivateUsage]
                    "run", dict(run, writer_epoch=epoch))))
        if rows:
            _, plan = writer._prepare_rows(rows, None)  # pyright: ignore[reportPrivateUsage]
            writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]


def recover_forgets(owner: ForgetOwner, scopes: Iterable[ScopeV1]) -> list[str]:
    """Resume every forget whose fence survived a restart (boot step 13, C.13).

    Each fence's Runs are first adopted under the current Writer epoch.

    Returns:
        The controller ids resumed, in scope order.
    """
    resumed: list[str] = []
    writer = owner._writer  # pyright: ignore[reportPrivateUsage]
    for scope in scopes:
        for row in writer.indexed_reader().scope_fences(scope.scope_id):
            key = cast(str, row["key"])
            if key.startswith(f"erasure:{scope.scope_id}:"):
                controller = key.rsplit(":", 1)[1]
                _adopt(writer, scope, controller)
                owner.resume(scope, controller)
                resumed.append(controller)
    return resumed


def abandon_orphan_controllers(writer: Writer, *, now: int) -> list[str]:
    """End every erasure controller a crash left before its admission (boot step 13, R17).

    ``ForgetOwner._controller`` writes the controller in its own transaction, before
    the admission transaction that names it in the intent, the fence, the worker and
    the cancellation. A crash between the two leaves a running controller that holds
    nothing: its intent is spent, so the client prepares again. Its zero-effect proof
    (C.13 R17): no fence, no Control it wrote and no child Run names it. Proved, it
    fails ``lease_lost``; not proved, it is left ``manual_attention``
    ``parity_mismatch`` and never retried. Fenced controllers resume through
    `recover_forgets`, which runs first.

    Returns:
        The controllers ended ``lease_lost``, in id order.

    Raises:
        AMLError: ``lease_lost`` for a controller of a later Writer epoch.
    """
    abandoned: list[str] = []
    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        lease = writer.get_control("lease:writer")
        if lease is None:
            return abandoned
        epoch = lease["value"]["writer_epoch"]
        controls = list(writer.iter_rows("control"))
        named = {row["updated_by_run_id"] for row in controls} | {
            str(row["key"]).rsplit(":", 1)[1] for row in controls
            if str(row["key"]).startswith("erasure:")
        }
        reader = writer.indexed_reader()
        # Only an intent the admission never consumed can have such a controller; its
        # family names it (one controller per intent).
        found = [
            reader.latest_run("erasure_controller", _family(writer, _admission(
                "erasure_controller", "orchestrator", value.principal_id, None,
                {"target_kind": "none", "target_id": None}, str(row["key"]), None,
            )))
            for row in controls if str(row["key"]).startswith("intent:")
            and ":forget:" in str(row["key"])
            and (value := ForgetIntentV1.model_validate(row["value"])).state != "consumed"
        ]
        rows: list[RowWrite] = []
        for run in sorted((run for run in found if run is not None),
                          key=lambda row: row["object_id"]):
            if run["status"] not in {"queued", "running"}:
                continue
            if run["writer_epoch"] > epoch:
                raise AMLError("lease_lost", "An erasure controller names a later Writer epoch")
            if run["object_id"] in named:
                continue
            if reader.run_children(run["object_id"], after=None, limit=1):
                ended = dict(run, writer_epoch=epoch, status="manual_attention",
                             failure_code="parity_mismatch", updated_at=now)
            else:
                ended = dict(run, writer_epoch=epoch, status="failed", failure_code="lease_lost",
                             updated_at=now, finished_at=now)
                abandoned.append(run["object_id"])
            rows.append(RowWrite("run", writer._parse_row("run", ended)))  # pyright: ignore[reportPrivateUsage]
        if rows:
            _, plan = writer._prepare_rows(rows, None)  # pyright: ignore[reportPrivateUsage]
            writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
    return abandoned


__all__ = [
    "FORGET_RECEIPT_KINDS", "FinalScan", "ForgetOutcome", "ForgetOwner", "ForgetPrepared",
    "MAX_TX_ROWS", "abandon_orphan_controllers", "check_controller_artifact", "fence_key",
    "intent_key", "owns_erasure_fence", "recover_forgets",
]
