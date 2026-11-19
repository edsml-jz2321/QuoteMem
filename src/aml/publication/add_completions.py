"""The completion records' reader for add-items and Why item targets (E.2.1, E.2.3; I14).

`StoredAddCompletions` reads one request's frozen completion through the memory
read's guarded rows, so the delivery recheck covers every record:

1. the request's ``add_outcome:{scope}:{request_key}`` result (A.3.9), visible
   only once its completing owner succeeded (`store.add_outcome.completed_result`'s
   rule);
2. that owner's completion record (``result_index:add_completion``, the owner's
   ``result_ref``, chunked with ``part`` when large) and the request's frozen
   items (``core.add_outcome.CompletedRequestV1``);
3. per item, its candidate row (ids and codes only; DM-06), found by the
   position's own key through a declared candidate index, for a formal item's
   written revision. Why's verdict is the item's own frozen ``judge_result_ref``,
   never the row's current one (E.2.3).

The record's ids are historical locators: a forget or erasure leaves them in
place, and the readers drop forgotten, absent and unreadable objects (INV:P05).
"""

from __future__ import annotations

from typing import Final

from pydantic import TypeAdapter, ValidationError

from aml.core.add_outcome import (
    AddCompletionRecordV1,
    CompletedAddOutcomeV1,
    CompletedItemV1,
    CompletedRequestV1,
)
from aml.core.controls import ControlRowV1
from aml.core.entities import QuoteSpanV1
from aml.core.errors import AMLError
from aml.core.identity import control_id
from aml.core.outcome_table import outcome_rule
from aml.publication.add_items import FrozenAddItem, FrozenCompletion
from aml.publication.history import MemoryRows
from aml.store.add_outcome import request_key
from aml.store.domain import CandidateRowV2
from aml.store.indexed import MAX_PAGE_SIZE, CandidateRows

__all__ = ["MAX_CANDIDATE_PAGES", "StoredAddCompletions", "frozen_completion"]

MAX_CANDIDATE_PAGES: Final = 64
"""The most index pages one item's candidate lookup reads before it fails closed."""

_CANDIDATE: Final[TypeAdapter[CandidateRowV2]] = TypeAdapter(CandidateRowV2)


def _result(rows: MemoryRows, key: str) -> CompletedAddOutcomeV1 | None:
    raw = rows.row("control", control_id(f"add_outcome:{rows.scope.scope_id}:{key}"))
    if raw is None:
        return None
    value = ControlRowV1.model_validate(raw).value
    if not isinstance(value, CompletedAddOutcomeV1) or value.request_key != key:
        raise AMLError("binding_mismatch", "Add result changed its request binding")
    return value


def _candidate(rows: MemoryRows, item: CompletedItemV1) -> CandidateRowV2 | None:
    """Find the position's candidate row by its key, or None when it has none.

    A resolved position's row is under its primary anchor (``ix_candidate_entity``);
    any other is under one of its code's statuses (``ix_candidate_status``). A row
    re-entered and re-resolved since the completion may sit elsewhere, and a
    position admission refused has none; it is then not found, and the item
    names no candidate and no written revision.
    """
    rule = outcome_rule(item.reason_code)
    scope = rows.scope
    queries = [
        CandidateRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id, entity_id=entity)
        for entity in item.entity_ids[:1]
    ] + [
        CandidateRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id, status=status)
        for status in sorted(rule.statuses)
    ]
    for query in queries:
        found = _find(rows, query, item)
        if found is not None:
            try:
                return _CANDIDATE.validate_python(rows.row("candidate", found))
            except ValidationError:
                raise AMLError("schema_invalid", "A stored candidate is malformed") from None
    return None


def _find(rows: MemoryRows, query: CandidateRows, item: CompletedItemV1) -> str | None:
    """Page one candidate index until the item's position, bounded by ``MAX_CANDIDATE_PAGES``."""
    after: str | None = None
    for _ in range(MAX_CANDIDATE_PAGES):
        found = rows.page(query, after=after, limit=MAX_PAGE_SIZE)
        for raw in found:
            if (raw["owner_run_id"], raw["unit_no"], raw["position"]) == item.position_key:
                return raw["object_id"]
        if len(found) < MAX_PAGE_SIZE:
            return None
        after = found[-1]["object_id"]
    raise AMLError("parity_mismatch", "An item's candidate lookup exceeds its bound")


def frozen_item(item: CompletedItemV1, candidate: CandidateRowV2 | None) -> FrozenAddItem:
    """Project one frozen item and its candidate row onto the readers' shape.

    ``revision_id`` is the candidate's written revision only while the row still
    holds the frozen code: a formal code is terminal, and a later re-entry that
    wrote a revision never makes an older non-formal item formal.
    """
    owner, unit, position = item.position_key
    formal = candidate is not None and candidate.reason_code == item.reason_code
    return FrozenAddItem(
        owner_run_id=owner, unit=unit, position=position, reason_code=item.reason_code,
        candidate_id=None if candidate is None else candidate.object_id,
        revision_id=candidate.revision_id if formal and candidate is not None else None,
        memory_ids=tuple(item.memory_ids), target_memory_ids=tuple(item.target_memory_ids),
        entity_ids=tuple(item.entity_ids),
        quote_spans=tuple(
            QuoteSpanV1(evidence_id=span.evidence_id, byte_start=span.byte_start,
                        byte_end=span.byte_end)
            for span in item.quote_spans
        ),
        judge_result_ref=item.judge_result_ref,
    )


def frozen_completion(
    result: CompletedAddOutcomeV1, request: CompletedRequestV1,
    items: tuple[FrozenAddItem, ...],
) -> FrozenCompletion:
    """Assemble one request's frozen completion, checking the record against its result.

    Raises:
        AMLError: ``binding_mismatch`` when the record's request or outcome is
            not the frozen result's.
    """
    if request.request_key != result.request_key or request.outcome != result.outcome:
        raise AMLError("binding_mismatch", "A completion record differs from its frozen result")
    return FrozenCompletion(
        completed_by_run_id=result.completed_by_run_id, recovery_count=result.recovery_count,
        outcome=result.outcome, items=items,
    )


class StoredAddCompletions:
    """Read frozen completion records from the store (the `AddCompletions` seam)."""

    def completion(
        self, rows: MemoryRows, session_id: str, request_id: str,
    ) -> FrozenCompletion | None:
        """Read one request's completion record in the read's exact scope.

        Args:
            rows (MemoryRows): The read's guarded rows; their scope is the caller's.
            session_id (str): The request's session.
            request_id (str): The request.

        Returns:
            FrozenCompletion | None: The record, or None while the scope has no
                result of this request or its owner has not succeeded.

        Raises:
            AMLError: ``binding_mismatch`` when a result, its owner, its record
                or an item's candidate disagree; the guard's refusal.
        """
        key = request_key(rows.writer, rows.scope, session_id, request_id)
        result = _result(rows, key)
        if result is None:
            return None
        owner = rows.row("run", result.completed_by_run_id)
        if owner is None or owner["run_kind"] != "finalize":
            raise AMLError("binding_mismatch", "Add result lost its completion owner")
        if (owner["status"], owner["stage"]) != ("succeeded", "release"):
            return None
        if owner["result_ref"] is None:
            raise AMLError("binding_mismatch", "A completed owner has no completion record")
        record = rows.writer.read_logical_observed(owner["result_ref"], "result_index", rows.row)
        if not isinstance(record, AddCompletionRecordV1) or (
            record.owner_run_id, record.recovery_count,
        ) != (result.completed_by_run_id, result.recovery_count):
            raise AMLError("binding_mismatch", "A completion record names another owner")
        request = next(
            (request for request in record.completed_requests if request.request_key == key),
            None,
        )
        if request is None:
            raise AMLError("binding_mismatch", "A completion record lacks its request")
        items = tuple(frozen_item(item, _candidate(rows, item)) for item in request.items)
        return frozen_completion(result, request, items)
