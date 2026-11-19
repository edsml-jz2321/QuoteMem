"""The original extraction's durable deadline and physical-send reservations.

A provider child keys its allowance by its own Run; an Add owner's extraction
call has no child and keys it by unit and logical call (W3-RT-13,
``OwnerUnitAccounting``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from aml.core.controls import ControlRowV1, FinalizeBudgetV2
from aml.core.errors import AMLError
from aml.core.extraction_context import ExtractContextV2
from aml.core.identity import integrity_digest
from aml.core.inference import InferenceBindingV1, StoredInferenceOutput
from aml.core.provider_budget import (
    EXTRACTION_REQUEST_CAP,
    CallAllowanceEnded,
    ExtractionOwnerBudgetV1,
    ProviderCallBudgetV2,
    SendsSpent,
    UnitCallEndV1,
)
from aml.core.provider_diagnostics import access_refused
from aml.core.request_budget import RequestBudget, current_budget, request_budget
from aml.providers.adapters import ProviderError
from aml.providers.attempts import provider_attempts

if TYPE_CHECKING:
    from aml.store.indexed import IndexedReader
    from aml.store.writer import RowWrite, Writer


def _key(run: Mapping[str, Any]) -> str:
    return f"extraction_budget:{run['scope_id']}:{run['object_id']}"


def _read(
    writer: Writer, run: Mapping[str, Any], binding: InferenceBindingV1,
) -> tuple[ControlRowV1, ExtractionOwnerBudgetV1] | None:
    raw = writer.get_control(_key(run))
    if raw is None:
        return None
    control = ControlRowV1.model_validate(raw)
    value = control.value
    if not isinstance(value, ExtractionOwnerBudgetV1) or (
        value.source_run_id != run["object_id"]
        or value.policy_digest != integrity_digest(binding)
        or control.version < value.attempts_reserved
        or not value.started_at <= control.updated_at <= value.deadline_at
    ):
        raise AMLError("binding_mismatch", "Extraction budget changed its original source owner")
    return control, value


def has_foreign_controls(writer: Writer, reader: IndexedReader, run_id: str) -> bool:
    """Whether a Run last updated any Control except the extraction allowance it spends.

    An extraction Run creates its own allowance immediately before inference
    and reserves every physical send against that row. The row survives
    failure by design, so zero-effect proofs exempt only this exact stored
    allowance; every other Control remains a durable effect.
    """
    controls = reader.control_provenance(run_id, limit=2)
    return len(controls) > 1 or any(
        not _is_spent_allowance(writer, run_id, raw) for raw in controls
    )


def has_direct_effects(writer: Writer, reader: IndexedReader, run_id: str) -> bool:
    """Whether a Run has direct domain effects or a Control other than its spent allowance.

    Both probes enumerate rows by their producer, which no row or membership
    key names, so they stay out of every recorded read set: a zero-effect proof
    reruns them before its action whenever another write may have intervened.
    """
    from aml.store.row_cache import suspended_reads

    with suspended_reads():
        return reader.run_effects(run_id, include_children=False) or (
            has_foreign_controls(writer, reader, run_id)
        )


def _is_spent_allowance(writer: Writer, run_id: str, raw: Mapping[str, Any]) -> bool:
    if not str(raw["key"]).startswith("extraction_budget:"):
        return False
    control = ControlRowV1.model_validate(raw)
    value = control.value
    if not isinstance(value, ExtractionOwnerBudgetV1) or control.updated_by_run_id != run_id or (
        run_id != value.source_run_id
    ):
        return False
    owner = writer.get_row("run", value.source_run_id)
    stored = writer.get_control(control.key)
    if owner is None or owner["input_ref"] is None or control.key != _key(owner) or (
        stored is None or ControlRowV1.model_validate(stored) != control
    ):
        return False
    actual = writer.read_logical(owner["input_ref"], "run_input").model_dump(mode="json")
    if actual.get("inference_binding") is None:
        return False
    binding = InferenceBindingV1.model_validate(actual["inference_binding"])
    return _read(writer, owner, binding) is not None


def ensure_budget(
    writer: Writer, run_id: str, binding: InferenceBindingV1, *, clock: Callable[[], int],
) -> ExtractionOwnerBudgetV1:
    """Create once per child attempt immediately before inference, capped by the Add owner.

    Within one attempt the allowance is never reset or extended. The first
    inference of an explicitly retried attempt replaces the spent allowance of
    the earlier attempt with its own; the original owner's deadline still bounds it.
    """
    from aml.store.writer import RowWrite

    if binding.purpose != "extract" or binding.call_budget is None:
        raise AMLError("binding_mismatch", "Extraction requires its frozen logical allowance")
    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        run = writer.get_row("run", run_id)
        if run is None or run["input_ref"] is None or run["commit_marker"] is not None:
            raise AMLError("binding_mismatch", "Extraction budget requires its actual active input")
        selected = writer.read_logical(run["input_ref"], "run_input")
        actual_binding = selected.model_dump(mode="json").get("inference_binding")
        if actual_binding != binding.model_dump(mode="json"):
            raise AMLError("binding_mismatch", "Extraction budget changed its admitted policy")
        writer._admit(writer.ticket(run_id))  # pyright: ignore[reportPrivateUsage]
        previous = None
        if (existing := _read(writer, run, binding)) is not None:
            previous, spent = existing
            if spent.source_attempt == run["attempt"]:
                return spent
            if spent.source_attempt > run["attempt"]:
                raise AMLError("binding_mismatch", "Extraction budget belongs to a later attempt")
        now = clock()
        deadline = now + binding.call_budget.logical_wall_ms
        if (parent_budget := current_budget()) is not None:
            parent_budget.check()
            deadline = min(deadline, now + int(parent_budget.remaining() * 1000))
        parent = run
        seen: set[str] = set()
        while True:
            identity = parent["object_id"]
            if identity in seen:
                raise AMLError("binding_mismatch", "Extraction budget owner contains a cycle")
            seen.add(identity)
            if parent["run_kind"] == "finalize":
                original = writer.get_control(f"finalize_budget:{parent['scope_id']}")
                if original is not None and original["value"]["owner_run_id"] == identity:
                    deadline = min(deadline, original["value"]["deadline_at"])
            if parent["parent_run_id"] is None:
                break
            parent = writer.get_row("run", parent["parent_run_id"])
            if parent is None:
                raise AMLError("reference_missing", "Extraction lost its original Add owner")
        if deadline <= now:
            raise AMLError("delivery_busy", "Original extraction allowance is exhausted")
        value = ExtractionOwnerBudgetV1(
            purpose="extract", source_run_id=run_id, source_attempt=run["attempt"],
            policy_digest=integrity_digest(binding), started_at=now, deadline_at=deadline,
        )
        row = writer.materialize_control(
            _key(run), value, 0 if previous is None else previous.version + 1, now, run_id,
        )
        _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
            [RowWrite("control", row)], writer.ticket(run_id),
        )
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return value


def reserve_attempt(
    writer: Writer, run_id: str, binding: InferenceBindingV1, *, clock: Callable[[], int],
) -> ExtractionOwnerBudgetV1:
    """Reserve before the physical HTTP send; interruption never refunds a send."""
    from aml.store.writer import RowWrite

    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        run = writer.get_row("run", run_id)
        if run is None:
            raise AMLError("reference_missing", "Extraction budget lost its source")
        actual = _read(writer, run, binding)
        if actual is None:
            raise AMLError("binding_mismatch", "Extraction send has no original allowance")
        control, value = actual
        if value.source_attempt != run["attempt"]:
            raise AMLError("binding_mismatch", "Extraction send has another attempt's allowance")
        now = clock()
        if now < value.started_at:
            raise AMLError("binding_mismatch", "Extraction clock preceded its original start")
        if now >= value.deadline_at or value.attempts_reserved >= min(
            EXTRACTION_REQUEST_CAP, binding.retry_policy.max_attempts,
        ):
            raise AMLError("delivery_busy", "Original extraction send allowance is exhausted")
        ticket = writer.ticket(run_id)
        writer._admit(ticket)  # pyright: ignore[reportPrivateUsage]
        writer._execution_fences(run)  # pyright: ignore[reportPrivateUsage]
        updated = value.model_copy(update={"attempts_reserved": value.attempts_reserved + 1})
        row = writer.materialize_control(control.key, updated, control.version + 1, now, run_id)
        _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
            [RowWrite("control", row)], ticket,
        )
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return updated


@contextmanager
def extraction_allowance(
    writer: Writer, run_id: str, binding: InferenceBindingV1, *, clock: Callable[[], int],
) -> Iterator[None]:
    """Bind transport time and retries to the persisted original allowance."""
    value = ensure_budget(writer, run_id, binding, clock=clock)
    now = clock()
    if now >= value.deadline_at or value.attempts_reserved >= min(
        EXTRACTION_REQUEST_CAP, binding.retry_policy.max_attempts,
    ):
        raise AMLError("delivery_busy", "Original extraction allowance is exhausted")
    parent = current_budget()
    seconds = (value.deadline_at - now) / 1000
    if parent is not None:
        parent.check()
        seconds = min(seconds, parent.remaining())
    limited = RequestBudget(seconds)

    def reserve() -> None:
        reserve_attempt(writer, run_id, binding, clock=clock)

    with (nullcontext() if parent is None else parent.on_cancel(limited.cancel)):
        with request_budget(limited), provider_attempts(reserve):
            yield
            limited.check()
            if parent is not None:
                parent.check()


def unit_call_key(scope_id: str, owner_run_id: str, unit_no: int, call: int) -> str:
    """The Control key of one logical extraction call's allowance (W3-RT-13).

    Args:
        scope_id: The owner's scope.
        owner_run_id: The Add owner.
        unit_no: The unit within the owner.
        call: 0 for the unit's call, 1 for its one fresh call.

    Returns:
        ``extraction_budget:{scope}:{owner_run_id}:{unit_no}:{call}``.

    Raises:
        ValueError: For a negative unit or a call other than 0 and 1.
    """
    if unit_no < 0 or call not in (0, 1):
        raise ValueError("An extraction call is a non-negative unit and call 0 or 1")
    return f"extraction_budget:{scope_id}:{owner_run_id}:{unit_no}:{call}"


def _send_cap(binding: InferenceBindingV1) -> int:
    """The physical sends one logical extraction call may reserve (B.5)."""
    return min(EXTRACTION_REQUEST_CAP, binding.retry_policy.max_attempts)


def _owner(writer: Writer, owner_run_id: str) -> dict[str, Any]:
    owner = writer.get_row("run", owner_run_id)
    if owner is None or owner["run_kind"] != "finalize" or owner["parent_run_id"] is not None:
        raise AMLError("binding_mismatch", "A unit call allowance needs its Add owner")
    return owner


def _read_unit_call(
    writer: Writer, owner: Mapping[str, Any], key: str, binding: InferenceBindingV1 | None,
) -> tuple[ControlRowV1, ExtractionOwnerBudgetV1] | None:
    raw = writer.get_control(key)
    if raw is None:
        return None
    control = ControlRowV1.model_validate(raw)
    value = control.value
    if not isinstance(value, ExtractionOwnerBudgetV1) or (
        value.purpose != "extract" or value.source_run_id != owner["object_id"]
        or control.updated_by_run_id != owner["object_id"]
        or binding is not None and value.policy_digest != integrity_digest(binding)
        or control.version < value.attempts_reserved
        or not value.started_at <= control.updated_at <= value.deadline_at
    ):
        raise AMLError("binding_mismatch", "A unit call allowance changed its owner or policy")
    return control, value


def unit_calls_started(writer: Writer, owner_run_id: str, unit_no: int) -> int:
    """Count the unit's logical calls that hold a durable allowance (0 to 2).

    Args:
        writer: The sole Writer.
        owner_run_id: The Add owner.
        unit_no: The unit within the owner.

    Returns:
        0 before the unit's call, 1 once call 0 holds its allowance, 2 once
        its fresh call does.

    Raises:
        AMLError: ``binding_mismatch`` for a fresh call without its first call,
            or an allowance of another owner.
    """
    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        owner = _owner(writer, owner_run_id)
        held = [
            _read_unit_call(
                writer, owner, unit_call_key(owner["scope_id"], owner_run_id, unit_no, call), None,
            ) is not None
            for call in (0, 1)
        ]
    if held[1] and not held[0]:
        raise AMLError("binding_mismatch", "A fresh call holds an allowance without its first call")
    return sum(held)


def ensure_unit_call(
    writer: Writer, owner_run_id: str, unit_no: int, call: int, binding: InferenceBindingV1, *,
    clock: Callable[[], int],
) -> ExtractionOwnerBudgetV1:
    """Create a logical call's allowance once per owner attempt, capped by the owner (B.5, C.5).

    The allowance holds at most ``min(EXTRACTION_REQUEST_CAP, max_attempts)``
    physical sends within ``logical_wall_ms``, never past the owner's current
    deadline (``owner_deadline``). Within one owner attempt it is never reset or
    extended. An explicit resume, a later attempt with its fresh owner
    allowance, renews only the call's time window: the physical sends the same
    ``(owner, unit, call)`` reserved before the resume stay spent, including
    one whose response was lost (C.5, C.13, ingest-Q23, W3-RT-13). Explicit
    recovery grants whole-output refusals a fresh allowance (AR-1), and exhausted
    outage calls one additional send (OR-1), in the resume transaction itself.
    This function preserves those grants. Call 1 (the fresh call) requires call
    0's allowance.

    Args:
        writer: The sole Writer.
        owner_run_id: The held Add owner.
        unit_no: The unit within the owner.
        call: 0 or 1.
        binding: The call's frozen extraction binding.
        clock: Trusted millisecond action clock.

    Returns:
        The call's allowance for the owner's current attempt.

    Raises:
        AMLError: ``binding_mismatch`` for another purpose, an allowance of
            another owner, policy or later attempt, or a fresh call without its
            first; ``delivery_busy`` when the owner's allowance no longer
            permits sends; the Writer's fences for a lost owner.
    """
    from aml.store.request_completion import owner_deadline
    from aml.store.writer import RowWrite

    if binding.purpose != "extract" or binding.call_budget is None:
        raise AMLError("binding_mismatch", "Extraction requires its frozen logical allowance")
    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        owner = _owner(writer, owner_run_id)
        if owner["status"] != "running" or owner["input_ref"] is None:
            raise AMLError("binding_mismatch", "A unit call allowance needs the running owner")
        ticket = writer.ticket(owner_run_id)
        key = unit_call_key(owner["scope_id"], owner_run_id, unit_no, call)
        if call == 1 and _read_unit_call(writer, owner, unit_call_key(
            owner["scope_id"], owner_run_id, unit_no, 0,
        ), binding) is None:
            raise AMLError("binding_mismatch", "A fresh call needs the unit's first call")
        previous, reserved, outage_resumes, ended = None, 0, 0, None
        if (existing := _read_unit_call(writer, owner, key, binding)) is not None:
            previous, spent = existing
            reserved = spent.attempts_reserved
            outage_resumes = spent.outage_resumes
            ended = spent.ended
            if spent.source_attempt == owner["attempt"]:
                return spent
            if spent.source_attempt > owner["attempt"]:
                raise AMLError("binding_mismatch", "A call allowance belongs to a later attempt")
        now = clock()
        deadline = min(now + binding.call_budget.logical_wall_ms, owner_deadline(
            writer, owner_run_id,
        ))
        if (parent_budget := current_budget()) is not None:
            parent_budget.check()
            deadline = min(deadline, now + int(parent_budget.remaining() * 1000))
        if deadline <= now:
            raise AMLError("delivery_busy", "The owner's extraction allowance is exhausted")
        value = ExtractionOwnerBudgetV1(
            purpose="extract", source_run_id=owner_run_id, source_attempt=owner["attempt"],
            policy_digest=integrity_digest(binding), started_at=now, deadline_at=deadline,
            attempts_reserved=reserved,
            outage_resumes=outage_resumes,
            ended=ended,
        )
        row = writer.materialize_control(
            key, value, 0 if previous is None else previous.version + 1, now, owner_run_id,
        )
        _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
            [RowWrite("control", row)], ticket,
        )
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return value


def reserve_unit_send(
    writer: Writer, owner_run_id: str, unit_no: int, call: int, binding: InferenceBindingV1, *,
    clock: Callable[[], int],
) -> ExtractionOwnerBudgetV1:
    """Reserve one physical send of a logical call before HTTP; a crash never refunds it.

    Args:
        writer: The sole Writer.
        owner_run_id: The held Add owner.
        unit_no: The unit within the owner.
        call: 0 or 1.
        binding: The call's frozen extraction binding.
        clock: Trusted millisecond action clock.

    Returns:
        The allowance with this send reserved.

    Raises:
        SendsSpent: Every send of the call is already reserved (W3-RT-20).
        AMLError: ``binding_mismatch`` without the call's allowance for the
            owner's attempt; ``delivery_busy`` past its deadline.
    """
    from aml.store.writer import RowWrite

    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        owner = _owner(writer, owner_run_id)
        key = unit_call_key(owner["scope_id"], owner_run_id, unit_no, call)
        actual = _read_unit_call(writer, owner, key, binding)
        if actual is None or actual[1].source_attempt != owner["attempt"]:
            raise AMLError("binding_mismatch", "An extraction send has no allowance of its owner")
        control, value = actual
        now = clock()
        if now < value.started_at:
            raise AMLError("binding_mismatch", "Extraction clock preceded its allowance")
        if value.attempts_reserved >= _send_cap(binding) + value.outage_resumes:
            raise SendsSpent
        if now >= value.deadline_at:
            raise AMLError("delivery_busy", "The call's extraction send allowance is exhausted")
        ticket = writer.ticket(owner_run_id)
        updated = value.model_copy(update={"attempts_reserved": value.attempts_reserved + 1})
        row = writer.materialize_control(key, updated, control.version + 1, now, owner_run_id)
        _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
            [RowWrite("control", row)], ticket,
        )
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]
        return updated


def resumed_call_allowance(
    writer: Writer, owner: Mapping[str, Any], unit_no: int, *, now: int, deadline_at: int,
    outage: bool = False,
) -> RowWrite | None:
    """Grant the refused unit's call a fresh allowance for the owner's explicit resume (B.7.1).

    An explicit resume of a whole-output refusal re-runs only the refused unit's
    call: the unit's last started logical call, which stored no reply (B.7.1,
    E.5, AR-1). An exhausted provider-outage call instead keeps its reserved
    sends and gains exactly one send (OR-1). Both get the call's logical wall
    from the resume, inside the resumed owner's deadline. The caller commits
    it in the resume's own transaction, so each
    explicit resume grants exactly one allowance, and a crash neither loses nor
    repeats it; every send reserved under it stays spent (ingest-Q23).

    No call is renewed once the owner stored its settlement record (B.8): a unit that
    ended spent then is settled without its facts, every later invocation replays that
    settlement (``extraction_stage.store_settlement``), and a reply the unit got after it
    would make the replay compute another record, which no resume could repair.

    Args:
        writer: The sole Writer, inside the resume transaction.
        owner: The refused owner's Run, before its next attempt.
        unit_no: The refused unit.
        now: The transaction's trusted clock.
        deadline_at: The resumed owner's deadline.
        outage: Grant one additional send only if the call is exhausted.

    Returns:
        The call's allowance for the next attempt, or None when the unit started
        no call, its last call ended for good (``UnitCallEndV1``), an outage
        call still has sends left, or the owner already settled.

    Raises:
        AMLError: ``binding_mismatch`` when the owner lost its frozen extraction
            input, or a call allowance changed its owner, policy or call order.
    """
    from aml.core.run_inputs import FinalizeOwnerRunInputV1
    from aml.providers.bindings import inference_binding_for_policy
    from aml.store.source_labels import (
        _extraction_policy,  # pyright: ignore[reportPrivateUsage]
    )
    from aml.store.writer import RowWrite

    if owner["input_ref"] is None or not isinstance(
        selected := writer.read_logical(owner["input_ref"], "run_input"), FinalizeOwnerRunInputV1,
    ):
        raise AMLError("binding_mismatch", "A refused resume needs the owner's frozen input")
    binding = inference_binding_for_policy(
        _extraction_policy(writer, selected.extraction_policy_release_ref),
    )
    if binding.call_budget is None:
        raise AMLError("binding_mismatch", "Extraction requires its frozen logical allowance")
    keys = [unit_call_key(owner["scope_id"], owner["object_id"], unit_no, call) for call in (0, 1)]
    held = [_read_unit_call(writer, owner, key, binding) for key in keys]
    if held[1] is not None and held[0] is None:
        raise AMLError("binding_mismatch", "A fresh call holds an allowance without its first call")
    call = 1 if held[1] is not None else 0
    found = held[call]
    if found is None:
        return None
    control, spent = found
    if spent.source_attempt > owner["attempt"]:
        raise AMLError("binding_mismatch", "A call allowance belongs to a later attempt")
    if spent.ended is not None:
        # An ended call never sends again (TRUNC-LOOP); its mark stays as written.
        return None
    if settled(writer, owner["object_id"]):
        return None
    if outage and spent.attempts_reserved < _send_cap(binding) + spent.outage_resumes:
        return None
    value = ExtractionOwnerBudgetV1(
        purpose="extract", source_run_id=owner["object_id"], source_attempt=owner["attempt"] + 1,
        policy_digest=spent.policy_digest, started_at=now,
        deadline_at=min(now + binding.call_budget.logical_wall_ms, deadline_at),
        attempts_reserved=spent.attempts_reserved if outage else 0,
        outage_resumes=spent.outage_resumes + 1 if outage else 0,
    )
    return RowWrite("control", writer.materialize_control(
        keys[call], value, control.version + 1, now, owner["object_id"],
    ))


def settled(writer: Writer, owner_run_id: str) -> bool:
    """Whether the Add owner stored its settlement record, after which no unit calls again.

    Args:
        writer: The sole Writer.
        owner_run_id: The Add owner.

    Returns:
        bool: True once its ``decision_set:settlement`` exists (B.8).
    """
    from aml.core.artifacts import DigestSettlementSetV1

    return any(isinstance(body, DigestSettlementSetV1)
               for _, body in writer.find_logical_candidates(owner_run_id, "decision_set"))


def resumed_outage_allowances(
    writer: Writer, owner: Mapping[str, Any], *, now: int, deadline_at: int,
) -> tuple[RowWrite, ...]:
    """Grant each exhausted unanswered call one send in the explicit resume (OR-1).

    Args:
        writer: The sole Writer, inside the provider-resume transaction.
        owner: The paused extraction owner before its next attempt.
        now: The resume transaction's trusted clock.
        deadline_at: The resumed owner's deadline.

    Returns:
        Call controls to commit with the owner and lease. Spent reservations
        remain charged; each control permits exactly one additional send.
        Accepted replies keep their write-once identities and receive no grant,
        and neither does any call of an owner that already settled
        (``resumed_call_allowance``).

    Raises:
        AMLError: If a call's frozen owner, policy or call order changed.
    """
    if settled(writer, owner["object_id"]):
        return ()
    accepted = {
        (body.input_digest, body.inference_binding_digest, body.call_no)
        for _, body in writer.find_logical_candidates(owner["object_id"], "llm_output")
        if isinstance(body, StoredInferenceOutput) and body.purpose == "extract"
    }
    grants: list[RowWrite] = []
    for _, context in writer.find_logical_candidates(owner["object_id"], "decision_set"):
        if not isinstance(context, ExtractContextV2):
            continue
        unit_no = context.unit.unit_no
        # Reuse AR-1's frozen-binding, call-order and deadline validation.
        renewed = resumed_call_allowance(writer, owner, unit_no, now=now,
                                         deadline_at=deadline_at, outage=True)
        if renewed is None:
            continue
        control = ControlRowV1.model_validate(renewed.row.model_dump(mode="json"))
        call = int(control.key.rsplit(":", 1)[1])
        value = control.value
        assert isinstance(value, ExtractionOwnerBudgetV1)
        if (context.input_digest, value.policy_digest, call) not in accepted:
            grants.append(renewed)
    return tuple(grants)


def unit_call_end(
    writer: Writer, owner_run_id: str, unit_no: int, call: int,
) -> UnitCallEndV1 | None:
    """Read how a unit call ended for good, if it did (TRUNC-LOOP).

    Args:
        writer: The sole Writer.
        owner_run_id: The Add owner.
        unit_no: The unit within the owner.
        call: 0 or 1.

    Returns:
        The call's end mark, or None for a call that has no allowance or did not end.

    Raises:
        AMLError: ``binding_mismatch`` for an allowance of another owner.
    """
    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        owner = _owner(writer, owner_run_id)
        found = _read_unit_call(
            writer, owner, unit_call_key(owner["scope_id"], owner_run_id, unit_no, call), None,
        )
    return None if found is None else found[1].ended


def end_unit_call(
    writer: Writer, owner_run_id: str, unit_no: int, call: int, binding: InferenceBindingV1,
    end: UnitCallEndV1, *, clock: Callable[[], int],
) -> None:
    """Mark a unit call ended for good before the stage acts on its end (TRUNC-LOOP).

    The call's allowance is created first when the call never started (a piece the
    owner's deadline no longer fits). Marking is idempotent; a different mark on the
    same call is a replay that took another path.

    Args:
        writer: The sole Writer.
        owner_run_id: The held Add owner.
        unit_no: The unit within the owner.
        call: 0 or 1.
        binding: The call's frozen extraction binding.
        end: How the call ended and what the stage makes of it.
        clock: Trusted millisecond action clock.

    Raises:
        AMLError: ``binding_mismatch`` when the call already ended another way, or
            for the allowance checks of ``ensure_unit_call``; the Writer's fences
            for a lost owner.
    """
    from aml.store.writer import RowWrite

    with writer._write_scope():  # pyright: ignore[reportPrivateUsage]
        owner = _owner(writer, owner_run_id)
        key = unit_call_key(owner["scope_id"], owner_run_id, unit_no, call)
        if _read_unit_call(writer, owner, key, binding) is None:
            ensure_unit_call(writer, owner_run_id, unit_no, call, binding, clock=clock)
        found = _read_unit_call(writer, owner, key, binding)
        assert found is not None
        control, value = found
        if value.ended is not None:
            if value.ended != end:
                raise AMLError("binding_mismatch", "A unit call already ended another way")
            return
        # The mark keeps the allowance's window invariant (started_at <= updated_at <=
        # deadline_at); a reply can arrive in the window's last millisecond.
        now = min(max(clock(), value.started_at), value.deadline_at)
        row = writer.materialize_control(
            key, value.model_copy(update={"ended": end}), control.version + 1, now,
            owner_run_id,
        )
        _, plan = writer._prepare_rows(  # pyright: ignore[reportPrivateUsage]
            [RowWrite("control", row)], writer.ticket(owner_run_id),
        )
        writer._write_plan(plan)  # pyright: ignore[reportPrivateUsage]


@dataclass(frozen=True, slots=True)
class OwnerUnitAccounting:
    """The Add owner's durable per-unit call accounting (S3c's ``UnitCallAccounting``).

    Each logical extraction call of a unit holds its own allowance, keyed
    ``extraction_budget:{scope}:{owner_run_id}:{unit_no}:{call}`` (W3-RT-13):
    initially at most three physical sends inside its 600 s (B.5), within the
    owner's deadline. Each explicit outage resume can add one send (OR-1).
    A granted send that fails ends the unit (W3-RT-20), unless the provider
    refused the key for access or quota (401/403/429): that keeps the owner
    paused, so an Add never completes on a unit its provider never evaluated.
    A call whose own window ends without a reply raises ``CallAllowanceEnded``,
    in that invocation and in every later one of the same attempt: only the
    owner's next counted resume renews the window (RESUMENOW).
    A fresh call starts only while that deadline still covers its
    first send and the reserve for the stages after extraction (D14,
    ``fresh_call_fits``).

    Attributes:
        writer: The sole Writer.
        owner_run_id: The held Add owner.
        clock: Trusted millisecond action clock.
    """

    writer: Writer
    owner_run_id: str
    clock: Callable[[], int]

    def calls_started(self, unit_no: int) -> int:
        """Return how many logical calls of the unit hold a durable allowance (0 to 2)."""
        return unit_calls_started(self.writer, self.owner_run_id, unit_no)

    def allowance(
        self, unit_no: int, call: Literal[0, 1],
    ) -> Callable[[InferenceBindingV1], AbstractContextManager[None]]:
        """Return the allowance one logical call of the unit enters (S3a's ``extract``)."""

        @contextmanager
        def enter(binding: InferenceBindingV1) -> Iterator[None]:
            value = ensure_unit_call(
                self.writer, self.owner_run_id, unit_no, call, binding, clock=self.clock,
            )
            now = self.clock()
            if value.attempts_reserved >= _send_cap(binding) + value.outage_resumes:
                # Sends reserved before an explicit resume or a crash stay spent.
                raise SendsSpent
            if now >= value.deadline_at:
                raise CallAllowanceEnded
            parent = current_budget()
            seconds = (value.deadline_at - now) / 1000
            if parent is not None:
                parent.check()
                seconds = min(seconds, parent.remaining())
            limited = RequestBudget(seconds)

            def reserve() -> None:
                reserve_unit_send(
                    self.writer, self.owner_run_id, unit_no, call, binding, clock=self.clock,
                )

            with (nullcontext() if parent is None else parent.on_cancel(limited.cancel)):
                with request_budget(limited), provider_attempts(reserve):
                    try:
                        yield
                    except ProviderError as error:
                        current = ensure_unit_call(
                            self.writer, self.owner_run_id, unit_no, call, binding,
                            clock=self.clock,
                        )
                        # A key refused for access or quota never spends the unit:
                        # the owner stays paused and each resume grants one send
                        # (OR-1) until the provider works again (PAUSE200).
                        if current.outage_resumes and current.attempts_reserved >= (
                            _send_cap(binding) + current.outage_resumes
                        ) and not access_refused(error.diagnostics):
                            raise SendsSpent from error
                        raise
                    except AMLError as error:
                        # The call's own window, not the owner's, cancelled it:
                        # this attempt cannot send the call again.
                        if error.code == "delivery_busy" and not isinstance(
                            error, SendsSpent | CallAllowanceEnded,
                        ) and (limited.cancelled or self.clock() >= value.deadline_at) and not (
                            parent is not None and parent.cancelled
                        ):
                            raise CallAllowanceEnded from error
                        raise
                    limited.check()
                    if parent is not None:
                        parent.check()

        return enter

    def ended(self, unit_no: int, call: Literal[0, 1]) -> UnitCallEndV1 | None:
        """Return how the unit call ended for good, if it did (``unit_call_end``)."""
        return unit_call_end(self.writer, self.owner_run_id, unit_no, call)

    def end(
        self, unit_no: int, call: Literal[0, 1], binding: InferenceBindingV1, end: UnitCallEndV1,
    ) -> None:
        """Mark the unit call ended for good before acting on it (``end_unit_call``)."""
        end_unit_call(self.writer, self.owner_run_id, unit_no, call, binding, end,
                      clock=self.clock)

    def admits_piece(self, need_ms: int) -> bool:
        """Tell whether the owner's current deadline still fits one piece's call and the reserve.

        A split piece starts only when ``need_ms`` (its cut parent's duration) and the
        reserve fit (``fresh_call_fits``); unlike the fresh call, a resumed owner's
        renewed deadline counts. The stage records a piece that did not fit
        (``UnitCallEndV1`` ``not_started``), so a replay never starts it.
        """
        from aml.store.request_completion import fresh_call_fits, owner_deadline

        return fresh_call_fits(owner_deadline(self.writer, self.owner_run_id), self.clock(),
                               attempt_wall_ms=max(1, need_ms))

    def admits_fresh_call(self, call_budget: ProviderCallBudgetV2) -> bool:
        """Tell whether the admitted deadline covers a fresh call's first send and the reserve.

        Once this answers False for an owner it keeps answering False, durably
        (W3-RT-17): the answer reads only the owner's budget record and the
        trusted clock. Before any explicit resume (``recovery_count == 0``) the
        deadline is the admitted one, so a False answer stays False as time
        advances. An explicit resume replaces the deadline, and a stood first
        reply must not be replaced on resume, so after a resume no fresh call
        starts: the first reply stands. A fresh call that already holds
        its allowance (``calls_started == 2``) is never asked about again.
        """
        from aml.core.query import ScopeV1
        from aml.store.request_completion import fresh_call_fits, owner_deadline, read_budget

        deadline = owner_deadline(self.writer, self.owner_run_id)
        owner = _owner(self.writer, self.owner_run_id)
        row = read_budget(self.writer, ScopeV1(
            scope_kind=owner["scope_kind"], scope_id=owner["scope_id"],
        ), self.owner_run_id)
        if row is None or not isinstance(row.value, FinalizeBudgetV2):
            raise AMLError("binding_mismatch", "The Add owner has no admitted budget")
        if row.value.recovery_count > 0:
            return False
        return fresh_call_fits(deadline, self.clock(), attempt_wall_ms=call_budget.attempt_wall_ms)
