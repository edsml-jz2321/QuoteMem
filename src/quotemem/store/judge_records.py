"""Durable judge children, sends, replies and pending settlement.

`WriterJudgeRecords` is the Writer-backed `JudgeRecords` of `quotemem.ingestion.judge`.
Every judge child is an ``evaluate:judgment`` Run under the reviewing owner,
one per (reviewing owner, position, round, vote), target
``{owner_run_id}.{unit_no}.{position}.r{round}`` with the position's original
owner, and ``v{vote}`` appended for a confirmation vote. Its input
(`JudgmentRunInputV1`) names its ``NeighborSnapshotV1``, which the child creates
once before any dispatch.

**Confirmation votes.** ``vote(n)`` gives the records of vote ``n``'s
children: the same member input sent alone again, each vote its own child with
its own sends, output and result. A vote is never reused and writes no
``judge_output`` locator, so reuse keeps serving the member's review. A vote that
ends unavailable (an exhausted read timeout, spent sends or a refused solo
resend) settles pending at once: it pauses nothing and counts as no
vote.

**Send reservations.** Each physical send is reserved in every
carried member's child before HTTP. The child freezes its numbered phase, exact
request digest and effective timeout. Transient transport retries share the
phase's three-send cap. Only the explicit first-to-second transition starts
another allowance; a changed timeout never resets it. Recovery advances past a
reserved phase: the first proceeds alone to the second, and a reserved second
settles pending without another send. Each phase leaves the binding's commit
reserve before the owner's current deadline. A proven terminal non-read outage
that spends all three transport sends records pending before propagating the
provider pause; explicit resume reads that result without granting another phase.
A read timeout or an interrupted send does not establish that outage proof.

**Replies.** One Writer transaction stores a reply: the call
manifest, every accepted member's output and result index, the finish of each
of their children and each member's ``judge_output`` locator
(``Writer.commit_artifacts``), so a reply is durable whole or not at all. A
reuse child's result and finish are one transaction too.

**Pending settlement.** Every exhausted or budget-limited
member settles unavailable. Its result and finish are durable; `recorded`
finishes an interrupted settlement without another call. No unavailable member
pauses its owner. A later Add re-enters the pending member.

"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from quotemem.core.admission import CanonicalRunAdmission, admission_identity, judgment_family
from quotemem.core.controls import ControlRowV1
from quotemem.core.entities import PositionKey
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import (
    ArtifactId,
    ContentBindingDigest,
    canonical_json,
    integrity_digest,
    new_ulid,
)
from quotemem.core.inference import InferenceBindingV1, StoredInferenceOutput
from quotemem.core.judge_output import JudgeV8
from quotemem.core.judge_records import (
    JudgeMemberUnavailableV1,
    JudgeOutputLocatorV1,
    JudgeResultIndexV8,
    JudgmentAdmissionSeedV1,
    JudgmentRunInputV1,
    judgment_business_key,
    judgment_target,
)
from quotemem.core.judge_timeout import judge_owner_wait
from quotemem.core.neighbor_snapshot import NeighborSnapshotV1
from quotemem.core.provider_budget import (
    EXTRACTION_REQUEST_CAP,
    OWNER_BUDGET_SPAN_MS,
    ExtractionOwnerBudgetV1,
    JudgeTimeShortfall,
    SendsSpent,
)
from quotemem.core.provider_diagnostics import access_refused, access_refused_sends
from quotemem.core.query import ScopeV1
from quotemem.core.request_budget import RequestBudget, current_budget, request_budget
from quotemem.core.resolution import PickV1
from quotemem.core.review_rounds import RoundNo
from quotemem.ingestion.authorization import check_add_grant
from quotemem.ingestion.judge import JudgmentV1, ReusableOutputV1, ReviewMemberV1
from quotemem.ingestion.judge_batch import JudgeReplyV1, plan_of
from quotemem.ingestion.judge_input import JudgeRequestPlanV1, MemberInputV1
from quotemem.ingestion.judge_records import build_call_manifest, resolve_targets
from quotemem.providers.adapters import ProviderError
from quotemem.providers.attempts import provider_attempts
from quotemem.providers.contracts import InferenceRequest
from quotemem.store.metadata import StrictModel

if TYPE_CHECKING:
    from quotemem.providers.auth import PrincipalGrant
    from quotemem.store.writer import PreparedArtifact, RowWrite, Writer, WriteTicket

LOCATOR_PURPOSE = "judge-output-locator"
"""Binding purpose of a ``judge_output`` key: the binding and member input digests."""

@dataclass(slots=True)
class _Record:
    """One reply Artifact, retaining its private encoding once its address is needed."""

    ticket: WriteTicket
    kind: str
    body: StrictModel
    security_level: str
    prepared: PreparedArtifact | None = None

type _Finished = tuple[str, ArtifactId, JudgeResultIndexV8, str]
"""One child a transaction finishes: the child, its result, the result body, its input digest."""


@dataclass(frozen=True, slots=True)
class EarlierReviewV1:
    """A review the owner already made of a position in one round.

    One judge child exists per (owner, position, round), admitted with its
    member input, so a stopped owner that runs the round again finds it.

    Attributes:
        member_input_digest: The input the child was admitted with: the only
            input the position is reviewed with in that round.
        result_ref: The child's ``result_index:judge`` once it settled.
        picks: The slots its accepted verdict re-picked, each named by the
            candidate its snapshot issued under the chosen handle.
    """

    member_input_digest: str
    result_ref: str | None
    picks: tuple[PickV1, ...]


class JudgeOwner(Protocol):
    """The reviewing Add owner as the judge records need it (the owner context)."""

    @property
    def writer(self) -> Writer:
        """The Writer."""
        ...

    @property
    def run_id(self) -> str:
        """The owner run."""
        ...

    @property
    def attempt(self) -> int:
        """The owner's current attempt."""
        ...

    def grant(self) -> PrincipalGrant:
        """The current principal grant, resolved outside the Writer mutex."""
        ...

    def check(self) -> None:
        """Raise when the owner is cancelled."""
        ...

    def clock(self) -> int:
        """The trusted action clock in milliseconds."""
        ...


def judgment_admission(owner: Mapping[str, Any], seed: JudgmentAdmissionSeedV1,
                       ) -> CanonicalRunAdmission:
    """Project the owner and seed into the ordinary internal evaluate admission."""
    return CanonicalRunAdmission.model_validate({
        "v": 1, "run_kind": "evaluate", "commit_mode": "single_compensatable",
        "principal_id": owner["principal_id"], "scope_kind": owner["scope_kind"],
        "scope_id": owner["scope_id"], "target_kind": "scope",
        "target_locator": owner["scope_id"], "raw_business_key": judgment_business_key(seed),
        "request_payload": seed.model_dump(mode="json"), "options": {"internal": True},
    })


def validate_input(writer: Writer, run: Mapping[str, Any], selected: JudgmentRunInputV1) -> None:
    """Prove a judge child's attached input against its owner, admission and snapshot.

    Raises:
        QuoteMemError: ``binding_mismatch`` when the owner, scope, epochs, admission
            identity, target or snapshot differ.
    """
    owner = writer.get_row("run", selected.owner_run_id)
    seed = _seed(selected)
    if owner is None or owner["run_kind"] != "finalize" or (
        run["parent_run_id"] != owner["object_id"]
        or (run["run_kind"], run["commit_mode"], run["target_kind"], run["target_id"])
        != ("evaluate", "single_compensatable", "scope", judgment_target(seed))
        or any(run[name] != owner[name] for name in (
            "scope_kind", "scope_id", "principal_id", "admitted_erase_epoch",
            "admitted_consent_epoch",
        ))
        or run["commit_marker"] is not None or run["commit_seq"] is not None
        or run["run_generation"] != 0 or run["auth_context_ref"] is not None
    ):
        raise QuoteMemError("binding_mismatch", "A judge child changed its original owner")
    identity = admission_identity(judgment_admission(owner, seed), writer.bind_content)
    if identity != (run["idempotency_key"], run["admission_digest"], selected.payload_digest) or (
        selected.admission_digest != run["admission_digest"]
    ):
        raise QuoteMemError("binding_mismatch", "A judge child changed its complete admission")
    snapshot = writer.read_logical(selected.snapshot_ref, "decision_set")
    from quotemem.core.neighbor_snapshot import NeighborSnapshotV1

    if not isinstance(snapshot, NeighborSnapshotV1) or (
        snapshot.owner_run_id, snapshot.round, snapshot.member_input_digest,
        (snapshot.position.owner_run_id, snapshot.position.unit_no, snapshot.position.position),
    ) != (selected.owner_run_id, selected.round, selected.member_input_digest,
          tuple(selected.position_key)):
        raise QuoteMemError("binding_mismatch", "A judge child's snapshot differs from its input")


def _seed(selected: JudgmentRunInputV1) -> JudgmentAdmissionSeedV1:
    return JudgmentAdmissionSeedV1.model_validate({
        name: getattr(selected, name) for name in JudgmentAdmissionSeedV1.model_fields
    })


class WriterJudgeRecords:
    """The Writer-backed `JudgeRecords` of one owner and binding."""

    def __init__(self, owner: JudgeOwner, *, binding: InferenceBindingV1, vote: int = 1,
                 ) -> None:
        """Bind the owner, the frozen judge binding and the vote; construction does no I/O."""
        self._owner, self._writer, self._binding = owner, owner.writer, binding
        self._binding_digest = integrity_digest(binding)
        self._vote = vote
        self._children: dict[tuple[Any, ...], str] = {}
        self._members: dict[tuple[Any, ...], ReviewMemberV1] = {}

    def vote(self, number: int) -> WriterJudgeRecords:
        """The records of confirmation vote ``number``'s children."""
        return WriterJudgeRecords(self._owner, binding=self._binding, vote=number)

    # Authority ---------------------------------------------------------------------------

    def _check_owner(self, grant: PrincipalGrant) -> dict[str, Any]:
        """Recheck the live owner inside the Writer scope: Run, attempt and grant."""
        writer, owner = self._writer, self._owner
        writer.issue_ticket(owner.run_id)
        run = writer.get_row("run", owner.run_id)
        if run is None or run["run_kind"] != "finalize" or run["status"] != "running" or (
            run["input_ref"] is None or run["commit_marker"] is not None
        ):
            raise QuoteMemError("lease_lost", "The judge requires its live uncommitted owner")
        if run["attempt"] != owner.attempt:
            raise QuoteMemError("lease_lost", "The judge owner changed its attempt")
        check_add_grant(grant, ScopeV1(scope_kind=run["scope_kind"], scope_id=run["scope_id"]),
                        owner.clock(), run["principal_id"])
        return run

    def _seed(self, member: ReviewMemberV1) -> JudgmentAdmissionSeedV1:
        return JudgmentAdmissionSeedV1(
            owner_run_id=self._owner.run_id, position_key=member.position_key,
            round=member.snapshot.round,
            member_input_digest=ContentBindingDigest(member.input.digest),
            binding=self._binding, vote=self._vote,
        )

    def _child(self, owner: Mapping[str, Any], seed: JudgmentAdmissionSeedV1, *,
               identity: tuple[str, str, str] | None = None,
               ) -> dict[str, Any] | None:
        from quotemem.store.indexed import IndexedReader

        writer = self._writer
        family, digest, _ = identity or admission_identity(
            judgment_admission(owner, seed), writer.bind_content,
        )
        reader = IndexedReader(writer.runtime.workspace, writer.entities,
                               tenant_id=writer.tenant_id)
        run = reader.latest_run("evaluate", family)
        if run is not None and (
            run["admission_digest"] != digest or run["parent_run_id"] != owner["object_id"]
        ):
            raise QuoteMemError("binding_mismatch", "A judge child family changed its admission")
        return run

    def _input(self, run: Mapping[str, Any]) -> JudgmentRunInputV1:
        selected = self._writer.read_logical(run["input_ref"], "run_input")
        if not isinstance(selected, JudgmentRunInputV1):
            raise QuoteMemError("binding_mismatch", "A judge child lost its judgment input")
        return selected

    def _running_child(self, member: MemberInputV1) -> dict[str, Any]:
        child_id = self._children.get(member.position_key)
        run = None if child_id is None else self._writer.get_row("run", child_id)
        if run is None or run["status"] != "running" or run["input_ref"] is None:
            raise QuoteMemError("binding_mismatch", "A judged member has no running child")
        if self._input(run).member_input_digest != member.digest:
            raise QuoteMemError("binding_mismatch", "A judge child changed its member input")
        return run

    def authorize(self, members: Sequence[MemberInputV1]) -> None:
        """Recheck the owner and exactly these members' running children."""
        grant = self._owner.grant()
        with self._writer.write_scope():
            self._check_owner(grant)
            for member in members:
                self._writer.issue_ticket(self._running_child(member)["object_id"])

    # Admission ----------------------------------------------------------------------------

    def admit(self, member: ReviewMemberV1) -> None:
        """Admit the member's child with its snapshot and input, create-once."""
        from quotemem.store.writer import RowWrite

        writer, owner = self._writer, self._owner
        seed = self._seed(member)
        self._members[member.position_key] = member
        grant = owner.grant()
        with writer.write_scope():
            owner_run = self._check_owner(grant)
            identity = admission_identity(judgment_admission(owner_run, seed), writer.bind_content)
            existing = self._child(owner_run, seed, identity=identity)
            if existing is not None and existing["input_ref"] is not None:
                if self._input(existing).member_input_digest != member.input.digest:
                    raise QuoteMemError("binding_mismatch",
                                        "A judge child changed its member input")
                self._children[member.position_key] = existing["object_id"]
                return
            family, digest, payload = identity
            now = owner.clock()
            if existing is None:
                row = writer.parse_row("run", {
                    "object_id": new_ulid(), "run_kind": "evaluate",
                    "commit_mode": "single_compensatable", "status": "queued",
                    "stage": "await_input", "attempt": 0, "run_generation": 0,
                    "idempotency_key": family, "admission_digest": digest,
                    "parent_run_id": owner_run["object_id"],
                    "principal_id": owner_run["principal_id"],
                    "scope_kind": owner_run["scope_kind"], "scope_id": owner_run["scope_id"],
                    "target_kind": "scope", "target_id": judgment_target(seed),
                    "writer_epoch": owner_run["writer_epoch"],
                    "admitted_erase_epoch": owner_run["admitted_erase_epoch"],
                    "admitted_consent_epoch": owner_run["admitted_consent_epoch"],
                    "created_at": now,
                })
                ticket = writer._admit_run(row)  # pyright: ignore[reportPrivateUsage]
                existing = writer.get_row("run", ticket.run_id or "")
                if existing is None:
                    raise QuoteMemError("reference_missing", "Judge child admission was lost")
            ticket = writer.issue_ticket(existing["object_id"])
            snapshot_ref = self._create_once(ticket, existing["object_id"], "decision_set",
                                             member.snapshot, now)
            selected = JudgmentRunInputV1.model_validate({
                **seed.model_dump(mode="json"), "admission_digest": digest,
                "payload_digest": payload, "run_kind": "evaluate",
                "evaluation_kind": "judgment", "business_key": judgment_business_key(seed),
                "snapshot_ref": snapshot_ref,
            })
            reference = self._create_once(ticket, existing["object_id"], "run_input",
                                          selected, now)
            attached = dict(existing, input_ref=reference, status="running", stage="run_cases",
                            started_at=now, updated_at=now)
            rows = [RowWrite("run", writer.parse_row("run", attached))]
            _, plan = writer.plan_rows(rows, ticket)
            writer.write_plan(plan)
            self._children[member.position_key] = existing["object_id"]

    def _create_once(self, ticket: WriteTicket, creator: str, kind: str, body: Any,
                     now: int) -> ArtifactId:
        """Put one creator-bound logical artifact once; an equal earlier one is reused."""
        writer = self._writer
        for reference, stored in writer.find_logical_candidates(creator, kind):
            if canonical_json(stored) == canonical_json(body):
                return reference
        return writer.put_logical_artifact(ticket, kind, body, created_at=now)

    # Results ------------------------------------------------------------------------------

    def _judgment(self, result_ref: str, borrowed: bool | None = None) -> JudgmentV1:
        result = self._writer.read_logical(result_ref, "result_index")
        if not isinstance(result, JudgeResultIndexV8):
            raise QuoteMemError("binding_mismatch", "A judge result has another variant")
        if result.kind == "unavailable":
            return JudgmentV1("unavailable", None, False, ArtifactId(result_ref), False,
                              refused=result.refusal is not None)
        assert result.output_ref is not None
        output = self._writer.read_logical(result.output_ref, "llm_output")
        if not isinstance(output, StoredInferenceOutput) or not isinstance(output.output, JudgeV8):
            raise QuoteMemError("binding_mismatch", "A judge result lost its accepted output")
        return JudgmentV1("verdict", output.output, result.reason_readable,
                          ArtifactId(result_ref),
                          result.borrowed_from is not None if borrowed is None else borrowed)

    def earlier(self, position_key: PositionKey, round_no: RoundNo) -> EarlierReviewV1 | None:
        """The owner's earlier review of a position in one round, or None.

        Read without a member input: the child family names the owner, the
        position and the round only. A stopped owner compares its rebuilt input
        with the recorded one before it reviews the position in that round again.

        Raises:
            QuoteMemError: ``lease_lost`` without the owner Run; ``binding_mismatch``
                for a child of another owner or a snapshot of another kind.
        """
        from quotemem.store.indexed import IndexedReader

        writer = self._writer
        owner = writer.get_row("run", self._owner.run_id)
        if owner is None:
            raise QuoteMemError("lease_lost", "The judge owner no longer exists")
        family = judgment_family(
            principal_id=owner["principal_id"],
            scope=ScopeV1(scope_kind=owner["scope_kind"], scope_id=owner["scope_id"]),
            owner_run_id=owner["object_id"], position_key=position_key, round_no=round_no,
        )
        reader = IndexedReader(writer.runtime.workspace, writer.entities,
                               tenant_id=writer.tenant_id)
        run = reader.latest_run("evaluate", family)
        if run is None or run["input_ref"] is None:
            return None
        if run["parent_run_id"] != owner["object_id"]:
            raise QuoteMemError("binding_mismatch", "A judge child family changed its owner")
        selected = self._input(run)
        picks: tuple[PickV1, ...] = ()
        result_ref: str | None = run["result_ref"]
        # A privacy erase deletes the snapshot and the result of a review that
        # depended on erased Evidence: such a review re-picks nothing and
        # names no result.
        if result_ref is not None and writer.get_row("artifact", result_ref) is None:
            result_ref = None
        if run["status"] == "succeeded" and result_ref is not None and (
            writer.get_row("artifact", selected.snapshot_ref) is not None
        ):
            verdict = self._judgment(result_ref).verdict
            snapshot = writer.read_logical(selected.snapshot_ref, "decision_set")
            if not isinstance(snapshot, NeighborSnapshotV1):
                raise QuoteMemError("binding_mismatch", "A judge child's snapshot has another kind")
            issued = {(slot.slot, candidate.handle): candidate.entity_id or candidate.mint_key
                      for slot in snapshot.entities for candidate in slot.candidates}
            picks = tuple(
                PickV1(slot=entry.slot, handle=entry.choice, entity=entity)
                for entry in sorted(() if verdict is None else verdict.entities,
                                    key=lambda item: item.slot)
                if entry.verdict == "repick" and entry.choice is not None
                and (entity := issued.get((entry.slot, entry.choice)))
            )
        return EarlierReviewV1(str(selected.member_input_digest), result_ref, picks)

    def recorded(self, member: ReviewMemberV1) -> JudgmentV1 | None:
        """Return the child's judgment; an undecided exhaustion is decided first.

        A verdict is stored with its child's finish. An interruption may leave
        an unavailable result on a running child; finish it without another call.
        """
        writer, owner = self._writer, self._owner
        grant = owner.grant()
        with writer.write_scope():
            owner_run = self._check_owner(grant)
            run = self._child(owner_run, self._seed(member))
            if run is None or run["input_ref"] is None:
                return None
            if run["status"] == "succeeded":
                return self._judgment(run["result_ref"])
            if run["status"] != "running":
                raise QuoteMemError("binding_mismatch", "A judge child ended without its judgment")
            written = writer.find_logical_candidates(run["object_id"], "result_index")
            if not written:
                return None
            reference, result = written[0]
            if len(written) != 1 or not isinstance(result, JudgeResultIndexV8) or (
                result.kind != "unavailable"
            ):
                raise QuoteMemError("binding_mismatch", "A judge child holds another result")
            finished = (run["object_id"], reference, result, member.input.digest)
            self._publish(owner_run, [], [finished], locate=False)
            return self._judgment(reference)

    def _locator_key(self, scope_id: str, member_input_digest: str) -> str:
        digest = self._writer.bind_content(
            LOCATOR_PURPOSE, [self._binding_digest, member_input_digest],
        )
        return f"judge_output:{scope_id}:{digest}"

    def reusable(self, member: ReviewMemberV1) -> ReusableOutputV1 | None:
        """Return the scope's accepted, unscrubbed output for this input.

        A confirmation vote is a fresh sample: it never reuses an output.
        """
        if self._vote != 1:
            return None
        writer = self._writer
        owner = writer.get_row("run", self._owner.run_id)
        if owner is None:
            raise QuoteMemError("lease_lost", "The judge owner no longer exists")
        raw = writer.get_control(self._locator_key(owner["scope_id"], member.input.digest))
        if raw is None:
            return None
        value = ControlRowV1.model_validate(raw).value
        if not isinstance(value, JudgeOutputLocatorV1):
            raise QuoteMemError("binding_mismatch", "A judge output locator has another value")
        if writer.get_row("artifact", value.result_ref) is None or (
            writer.get_row("artifact", value.output_ref) is None
        ):
            return None
        judgment = self._judgment(value.result_ref)
        if judgment.verdict is None:
            raise QuoteMemError("binding_mismatch", "A judge output locator names no verdict")
        return ReusableOutputV1(judgment.verdict, judgment.reason_readable, value.output_ref,
                                value.result_ref)

    def reuse(self, member: ReviewMemberV1, output: ReusableOutputV1) -> JudgmentV1:
        """Finish the member's child as a reuse child of ``output``, without a call."""
        writer = self._writer
        child_id = self._children[member.position_key]
        grant = self._owner.grant()
        with writer.write_scope():
            owner_run = self._check_owner(grant)
            run = writer.get_row("run", child_id)
            if run is None:
                raise QuoteMemError("reference_missing", "A reuse child was lost")
            selected = self._input(run)
            original = writer.read_logical(output.result_ref, "result_index")
            if not isinstance(original, JudgeResultIndexV8) or original.kind != "verdict" or (
                original.member_input_digest != member.input.digest
                or original.binding_digest != self._binding_digest
            ):
                raise QuoteMemError("binding_mismatch", "A reused output belongs to another input")
            result = JudgeResultIndexV8(
                kind="verdict", output_ref=output.output_ref, snapshot_ref=selected.snapshot_ref,
                call_ref=original.call_ref,
                member_input_digest=ContentBindingDigest(member.input.digest),
                binding_digest=self._binding_digest, reason_readable=output.reason_readable,
                borrowed_from=output.result_ref,
                resolved=resolve_targets(member.snapshot, output.verdict),
            )
            record = _Record(writer.issue_ticket(child_id), "result_index", result, "internal")
            reference = self._reference(record)
            self._publish(owner_run, [record], [(child_id, reference, result, member.input.digest)],
                          locate=False)
        return self._judgment(reference, borrowed=True)

    # Sends --------------------------------------------------------------------------------

    def _deadline(self, owner: Mapping[str, Any], now: int, *,
                  outer: RequestBudget | None) -> int:
        """The owner's deadline less the frozen commit reserve, also bounded by its context."""
        reserve = self._binding.judge_commit_reserve_ms
        if reserve is None:
            raise QuoteMemError("binding_mismatch", "The judge binding lost its commit reserve")
        deadline = now + OWNER_BUDGET_SPAN_MS["judge"]
        raw = self._writer.get_control(f"finalize_budget:{owner['scope_id']}")
        if raw is not None and raw["value"].get("owner_run_id") == owner["object_id"]:
            deadline = min(deadline, raw["value"]["deadline_at"])
        if deadline <= now:
            raise QuoteMemError("delivery_busy", "The judge owner's complete budget has ended")
        if outer is not None:
            outer.check()
            deadline = min(deadline, now + int(outer.remaining() * 1000))
        return deadline - reserve

    def _budget(self, member: MemberInputV1) -> ExtractionOwnerBudgetV1 | None:
        child = self._running_child(member)
        raw = self._writer.get_control(
            f"extraction_budget:{child['scope_id']}:{child['object_id']}",
        )
        if raw is None:
            return None
        value = ControlRowV1.model_validate(raw).value
        if not isinstance(value, ExtractionOwnerBudgetV1) or value.purpose != "judge" or (
            value.judge_attempt is None or value.judge_timeout_ms is None
        ):
            raise QuoteMemError("binding_mismatch", "A judge allowance lost its numbered attempt")
        return value

    def attempt(self, member: MemberInputV1) -> Literal[1, 2, 3]:
        """Advance past any reserved phase after interruption; never resend it on recovery.

        A phase whose every send the provider refused for its key's access or
        quota holds no reservation (``_refund``): it never reached a judge, so
        recovery runs that same phase again.
        """
        with self._writer.mutex:
            value = self._budget(member)
            if value is None:
                return 1
            if value.attempts_reserved == 0 and value.judge_attempt is not None:
                return value.judge_attempt
            return 2 if value.judge_attempt == 1 else 3

    def timeout(self, members: Sequence[MemberInputV1], attempt: Literal[1, 2]) -> int | None:
        """Select the phase's read timeout within the owner's unspent commit reserve."""
        owner = self._writer.get_row("run", self._owner.run_id)
        if owner is None:
            raise QuoteMemError("lease_lost", "The judge owner no longer exists")
        cap = self._binding.timeout_ms if attempt == 1 else self._binding.judge_second_timeout_ms
        if cap is None:
            raise QuoteMemError("binding_mismatch", "The judge binding lost its second timeout")
        if len(members) == 1 and attempt == 2:
            reserved = self._budget(members[0])
            if reserved is not None and reserved.judge_attempt == 2 and (
                reserved.attempts_reserved
            ):
                return reserved.judge_timeout_ms
        now = self._owner.clock()
        remaining = self._deadline(owner, now, outer=current_budget()) - now
        return min(cap, remaining) if remaining > 0 else None

    def _allowance(self, child: Mapping[str, Any], owner: Mapping[str, Any],
                   request: InferenceRequest, now: int, *, spent: int, deadline: int,
                   ) -> ExtractionOwnerBudgetV1:
        """Freeze one phase's exact request identity without resetting it on a changed cap."""
        minimum = self._binding.judge_min_attempt_ms
        if minimum is None or request.judge_attempt is None:
            raise QuoteMemError("binding_mismatch", "A judge reservation lost its phase policy")
        if deadline - now < minimum:
            raise JudgeTimeShortfall
        return ExtractionOwnerBudgetV1(
            purpose="judge", source_run_id=child["object_id"], source_attempt=owner["attempt"],
            policy_digest=integrity_digest(request), started_at=now, deadline_at=deadline,
            attempts_reserved=spent, judge_attempt=request.judge_attempt,
            judge_timeout_ms=request.timeout_ms,
        )

    def _reserve(self, members: Sequence[MemberInputV1], request: InferenceRequest,
                 *, deadline: int, outer: RequestBudget | None) -> int:
        """Reserve a transient send in this phase; only the explicit phase transition resets it."""
        from quotemem.store.writer import RowWrite

        writer, owner = self._writer, self._owner
        grant = owner.grant()
        with writer.write_scope():
            owner_run = self._check_owner(grant)
            now = owner.clock()
            minimum = self._binding.judge_min_attempt_ms
            if minimum is None or request.judge_attempt is None:
                raise QuoteMemError("binding_mismatch", "A judge send lost its phase policy")
            deadline = min(deadline, self._deadline(owner_run, now, outer=outer))
            if deadline - now < minimum:
                raise JudgeTimeShortfall
            rows: list[RowWrite] = []
            deadlines: list[int] = []
            for member in members:
                child = self._running_child(member)
                key = f"extraction_budget:{owner_run['scope_id']}:{child['object_id']}"
                raw = writer.get_control(key)
                previous = None if raw is None else ControlRowV1.model_validate(raw)
                value = self._budget(member)
                if value is None:
                    if request.judge_attempt != 1:
                        raise QuoteMemError("binding_mismatch",
                                            "A judge cannot skip its first phase")
                    value = self._allowance(child, owner_run, request, now, spent=0,
                                            deadline=deadline)
                elif value.judge_attempt == 1 and request.judge_attempt == 2 or (
                    value.attempts_reserved == 0 and value.judge_attempt == request.judge_attempt
                ):
                    # A refunded phase (``_refund``) is re-run under the
                    # owner's current attempt with its own fresh allowance.
                    value = self._allowance(child, owner_run, request, now, spent=0,
                                            deadline=deadline)
                elif value.judge_attempt != request.judge_attempt or (
                    value.policy_digest != integrity_digest(request)
                    or value.judge_timeout_ms != request.timeout_ms
                ):
                    raise QuoteMemError("binding_mismatch",
                                        "A reserved judge phase changed its request")
                if value.source_attempt != owner_run["attempt"]:
                    # Recovery advances the phase before reservation. A live
                    # phase never borrows a later owner attempt's fresh time.
                    raise QuoteMemError("binding_mismatch",
                                        "A judge phase changed its owner attempt")
                if now < value.started_at:
                    raise QuoteMemError("clock_rollback", "The judge send clock moved backwards")
                if value.attempts_reserved >= EXTRACTION_REQUEST_CAP:
                    raise SendsSpent
                if value.deadline_at - now < minimum:
                    raise JudgeTimeShortfall
                following = value.model_copy(update={
                    "attempts_reserved": value.attempts_reserved + 1,
                })
                rows.append(RowWrite("control", writer.materialize_control(
                    key, following, 0 if previous is None else previous.version + 1, now,
                    owner_run["object_id"],
                )))
                deadlines.append(following.deadline_at)
            ticket = writer.issue_ticket(owner_run["object_id"])
            writer.check_ticket(ticket)
            _, plan = writer.plan_rows(rows, ticket)
            writer.write_plan(plan)
            return min(deadlines)

    def _refund(self, members: Sequence[MemberInputV1], sends: int) -> None:
        """Release the sends a provider refused for its key's access or quota.

        The provider answered those sends without evaluating the request, so
        they reached no judge. Releasing them lets recovery re-run the same
        phase (``attempt``) instead of settling its members pending, which
        would complete the Add without its judgment. A send that might have
        been evaluated (a timeout, a lost acknowledgement, a crash) stays spent.
        """
        from quotemem.store.writer import RowWrite

        writer, owner = self._writer, self._owner
        grant = owner.grant()
        with writer.write_scope():
            owner_run = self._check_owner(grant)
            now = owner.clock()
            rows: list[RowWrite] = []
            for member in members:
                child = self._running_child(member)
                key = f"extraction_budget:{owner_run['scope_id']}:{child['object_id']}"
                raw = writer.get_control(key)
                value = self._budget(member)
                if raw is None or value is None or value.attempts_reserved < sends:
                    raise QuoteMemError("binding_mismatch",
                                        "A refused judge send lost its reservation")
                following = value.model_copy(update={
                    "attempts_reserved": value.attempts_reserved - sends,
                })
                rows.append(RowWrite("control", writer.materialize_control(
                    key, following, ControlRowV1.model_validate(raw).version + 1, now,
                    owner_run["object_id"],
                )))
            ticket = writer.issue_ticket(owner_run["object_id"])
            writer.check_ticket(ticket)
            _, plan = writer.plan_rows(rows, ticket)
            writer.write_plan(plan)

    @contextmanager
    def sends(self, members: Sequence[MemberInputV1], request: InferenceRequest,
              ) -> Iterator[None]:
        """Reserve sends durably and stop judge work before it consumes commit time."""
        owner = self._owner
        deadlines: list[int] = []

        outer = current_budget()

        def reserve() -> None:
            owner.check()
            deadlines.append(self._reserve(members, request, deadline=deadline, outer=outer))
            owner.check()

        now = owner.clock()
        owner_run = self._writer.get_row("run", owner.run_id)
        if owner_run is None:
            raise QuoteMemError("lease_lost", "The judge owner no longer exists")
        deadline = self._deadline(owner_run, now, outer=outer)
        remaining = deadline - now
        minimum = self._binding.judge_min_attempt_ms
        if minimum is None:
            raise QuoteMemError("binding_mismatch",
                                "The judge binding lost its useful attempt limit")
        if remaining < minimum:
            raise JudgeTimeShortfall
        limited = RequestBudget(remaining / 1000)
        reserve_ms = self._binding.judge_commit_reserve_ms
        if reserve_ms is None:
            raise QuoteMemError("binding_mismatch", "The judge binding lost its commit reserve")

        def owner_wait_ms() -> int:
            available = deadline - owner.clock()
            if outer is not None:
                available = min(available, int(outer.remaining() * 1000) - reserve_ms)
            return max(0, available)
        try:
            with ExitStack() as stack:
                if outer is not None:
                    stack.enter_context(outer.on_cancel(limited.cancel))
                stack.enter_context(request_budget(limited))
                stack.enter_context(judge_owner_wait(owner_wait_ms))
                stack.enter_context(provider_attempts(reserve))
                yield
                limited.check()
                if deadlines and owner.clock() >= min(deadlines):
                    self._deadline(owner_run, owner.clock(), outer=outer)
                    raise JudgeTimeShortfall
        except ProviderError as error:
            detail = error.diagnostics
            if access_refused(detail):
                # The key was refused, not the request: the owner pauses with
                # this phase unspent and resumes it when the provider works.
                refused = min(access_refused_sends(detail), len(deadlines))
                if refused:
                    try:
                        self._refund(members, refused)
                    except QuoteMemError as refund:
                        from quotemem.store.failures import is_runtime_failure

                        if is_runtime_failure(refund):
                            raise
                        error.add_note("The refused judge sends kept their reservation.")
                raise
            if detail is not None and len(deadlines) == EXTRACTION_REQUEST_CAP and (
                len(detail.attempts) == EXTRACTION_REQUEST_CAP
                and detail.attempts[-1].decision == "exhausted"
                and detail.attempts[-1].timeout_phase != "read"
            ):
                # Retain known transport exhaustion before the owner
                # pauses. A read timeout still gets phase two; a crash has no
                # terminal proof and also keeps its recovery transition.
                self.spent(plan_of(members), request)
            raise
        except QuoteMemError as error:
            if error.code != "delivery_busy" or limited.remaining() > 0:
                raise
            # The limited judge context has ended. Preserve a real parent
            # cancellation or revoked owner; only its own reserve cutoff settles.
            if outer is not None:
                outer.check()
            owner.check()
            raise JudgeTimeShortfall from None

    # Persistence --------------------------------------------------------------------------

    def _reference(self, record: _Record) -> ArtifactId:
        """The address ``Writer.commit_artifacts`` gives one record; nothing is written."""
        return self._prepare(record).reference

    def _prepare(self, record: _Record) -> PreparedArtifact:
        """Detach and encode a record once, keeping it only until this reply commits."""
        if record.prepared is None:
            record.prepared = self._writer.prepare_artifact(
                record.ticket, record.kind, record.body, created_at=self._owner.clock(),
                security_level=record.security_level,
            )
        return record.prepared

    def _publish(self, owner: Mapping[str, Any], records: Sequence[_Record],
                 finished: Sequence[_Finished], *, locate: bool = True) -> None:
        """One transaction: the records, the finish of every child and its locator.

        The caller holds the Writer scope and checked the owner in it.
        """
        from quotemem.store.writer import RowWrite

        writer = self._writer
        now = self._owner.clock()
        rows: list[RowWrite] = []
        for child_id, reference, result, digest in finished:
            run = writer.get_row("run", child_id)
            if run is None or run["parent_run_id"] != owner["object_id"]:
                raise QuoteMemError("binding_mismatch", "A judge reply names another owner's child")
            if run["status"] == "succeeded":
                if run["result_ref"] != reference:
                    raise QuoteMemError("artifact_digest_collision",
                                        "A judge child already holds another result")
                continue
            if run["status"] != "running":
                raise QuoteMemError("binding_mismatch", "A judge child ended before its reply")
            following = dict(run, result_ref=reference, status="succeeded", stage="verdict",
                             updated_at=now, finished_at=now)
            rows.append(RowWrite("run", writer.parse_row("run", following)))
            if locate and result.output_ref is not None:
                key = self._locator_key(owner["scope_id"], digest)
                raw = writer.get_control(key)
                rows.append(RowWrite("control", writer.materialize_control(
                    key, JudgeOutputLocatorV1(child_run_id=child_id,
                                              output_ref=result.output_ref,
                                              result_ref=reference),
                    0 if raw is None else raw["version"] + 1, now, owner["object_id"],
                )))
        writer.commit_artifacts(writer.issue_ticket(owner["object_id"]),
                                [self._prepare(record) for record in records], rows,
                                created_at=now)

    def _snapshots(self, members: Sequence[MemberInputV1]) -> list[ArtifactId]:
        refs: list[ArtifactId] = []
        for member in members:
            refs.append(self._input(self._running_child(member)).snapshot_ref)
        return refs

    def _member_snapshot(self, reference: str) -> Any:
        from quotemem.core.neighbor_snapshot import NeighborSnapshotV1

        snapshot = self._writer.read_logical(reference, "decision_set")
        if not isinstance(snapshot, NeighborSnapshotV1):
            raise QuoteMemError("binding_mismatch", "A judge child lost its snapshot")
        return snapshot

    def persist(self, plan: JudgeRequestPlanV1, request: InferenceRequest,
                reply: JudgeReplyV1) -> None:
        """Store every accepted member of one reply in one Writer transaction."""
        writer = self._writer
        by_position = {member.position_key: member for member in plan.members}
        grant = self._owner.grant()
        with writer.write_scope():
            owner_run = self._check_owner(grant)
            lead = self._running_child(plan.members[0])["object_id"]
            manifest = self._manifest(lead, plan, self._snapshots(plan.members), request)
            call_ref = self._reference(manifest)
            records = [manifest]
            finished: list[_Finished] = []
            for accepted in reply.accepted:
                member = by_position[accepted.position_key]
                child = self._running_child(member)
                child_id, selected = child["object_id"], self._input(child)
                ticket = writer.issue_ticket(child_id)
                self._check_output_identity(child_id, accepted.output)
                output = _Record(ticket, "llm_output", accepted.output, "restricted")
                verdict = accepted.output.output
                if not isinstance(verdict, JudgeV8):
                    raise QuoteMemError("binding_mismatch", "An accepted member lost its verdict")
                result = JudgeResultIndexV8(
                    kind="verdict", output_ref=self._reference(output),
                    snapshot_ref=selected.snapshot_ref, call_ref=call_ref,
                    member_input_digest=ContentBindingDigest(member.digest),
                    binding_digest=self._binding_digest, reason_readable=accepted.reason_readable,
                    borrowed_from=None,
                    resolved=resolve_targets(self._member_snapshot(selected.snapshot_ref),
                                             verdict),
                )
                record = _Record(ticket, "result_index", result, "internal")
                records.extend((output, record))
                finished.append((child_id, self._reference(record), result, member.digest))
            self._publish(owner_run, records, finished, locate=self._vote == 1)

    def _manifest(self, lead: str, plan: JudgeRequestPlanV1, snapshot_refs: Sequence[ArtifactId],
                  request: InferenceRequest) -> _Record:
        """The call's content-free manifest, created by its first member's child."""
        nominations = [self._members[member.position_key].nomination for member in plan.members]
        if request.judge_attempt is None or request.judge_timeout_policy_digest is None:
            raise QuoteMemError("binding_mismatch", "A judge manifest lost its timeout policy")
        reserved = self._budget(plan.members[0])
        digest = integrity_digest(request)
        manifest = build_call_manifest(
            nominations, snapshot_refs=snapshot_refs, binding_digest=self._binding_digest,
            request_digest=digest, judge_attempt=request.judge_attempt,
            timeout_ms=request.timeout_ms,
            judge_timeout_policy_digest=request.judge_timeout_policy_digest,
            send_reserved=reserved is not None and reserved.policy_digest == digest
            and reserved.attempts_reserved > 0,
        )
        return _Record(self._writer.issue_ticket(lead), "decision_set", manifest, "internal")

    def _check_output_identity(self, creator: str, output: StoredInferenceOutput) -> None:
        """Refuse other bytes at a stored member output's identity."""
        for _, stored in self._writer.find_logical_candidates(creator, "llm_output"):
            if isinstance(stored, StoredInferenceOutput) and (
                stored.input_digest, stored.inference_binding_digest,
            ) == (output.input_digest, output.inference_binding_digest) and (
                canonical_json(stored) != canonical_json(output)
            ):
                raise QuoteMemError("artifact_digest_collision",
                                    "Output identity already binds different bytes")

    def spent(self, plan: JudgeRequestPlanV1, request: InferenceRequest) -> None:
        """Settle the members of a request whose sends are all spent, without a call.

        A spent phase or insufficient useful time produces no call. Persist the
        exact request identity and all pending members in canonical order, so
        recovery finishes the same disposition and a later Add re-enters it.
        """
        writer, owner = self._writer, self._owner
        grant = owner.grant()
        with writer.write_scope():
            owner_run = self._check_owner(grant)
            lead = self._running_child(plan.members[0])["object_id"]
            manifest = self._manifest(lead, plan, self._snapshots(plan.members), request)
            call_ref = self._reference(manifest)
            records: list[_Record] = [manifest]
            children: list[tuple[str, JudgeResultIndexV8]] = []
            for member in plan.members:
                child = self._running_child(member)
                child_id, selected = child["object_id"], self._input(child)
                result = JudgeResultIndexV8(
                    kind="unavailable", output_ref=None, snapshot_ref=selected.snapshot_ref,
                    call_ref=call_ref, member_input_digest=ContentBindingDigest(member.digest),
                    binding_digest=self._binding_digest, reason_readable=False,
                    borrowed_from=None, resolved=None,
                )
                records.append(_Record(writer.issue_ticket(child_id), "result_index", result,
                                       "internal"))
                children.append((child_id, result))
            _, *references = writer.commit_artifacts(
                writer.issue_ticket(owner_run["object_id"]),
                [self._prepare(record) for record in records],
                created_at=owner.clock(),
            )
            for member, (child_id, result), reference in zip(
                plan.members, children, references, strict=True,
            ):
                finished = (child_id, reference, result, member.digest)
                self._publish(owner_run, [], [finished], locate=False)

    def refused(self, member: MemberInputV1, request: InferenceRequest,
                refusal: JudgeMemberUnavailableV1) -> None:
        """Settle a member still unaccepted after its solo resend as pending.

        One Writer transaction stores its solo request's manifest and its
        ``unavailable`` result with the refusal, and finishes its child; nothing
        of the refused replies is stored. Later owners re-enter the member.
        """
        writer, owner = self._writer, self._owner
        grant = owner.grant()
        with writer.write_scope():
            owner_run = self._check_owner(grant)
            child = self._running_child(member)
            child_id, selected = child["object_id"], self._input(child)
            manifest = self._manifest(child_id, _solo(member), [selected.snapshot_ref], request)
            result = JudgeResultIndexV8(
                kind="unavailable", output_ref=None, snapshot_ref=selected.snapshot_ref,
                call_ref=self._reference(manifest),
                member_input_digest=ContentBindingDigest(member.digest),
                binding_digest=self._binding_digest, reason_readable=False, borrowed_from=None,
                resolved=None, refusal=refusal,
            )
            record = _Record(writer.issue_ticket(child_id), "result_index", result, "internal")
            self._publish(owner_run, [manifest, record],
                          [(child_id, self._reference(record), result, member.digest)],
                          locate=False)



def _solo(member: MemberInputV1) -> JudgeRequestPlanV1:
    from quotemem.ingestion.judge_batch import plan_of

    return plan_of([member])

