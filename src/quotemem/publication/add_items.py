"""Add items: one completed request's per-item results, frozen at first completion.

An Add's ``/add`` receipt counts six categories and never lists its items.
This read returns the items behind those counts from the request's completion
record, in ``(owner_run_id, unit, position)`` order: one per attributed
position, plus re-entered positions whose formal effect counted toward the
request. Their categories sum to the receipt's ``outcome``. Nothing is
recomputed from the mutable candidate rows; the code, category and rejudge of
an item come from its frozen code and the table.

A request without a completed record in the caller's scope (never received,
in flight, failed, another user's) is ``not_completed`` with nothing else, so
none of those cases can be told apart. At read time each id list drops what
is forgotten, absent or unreadable: a memory whose Item is gone or whose head
fails ``p_ver``, and an entity that is gone or erased.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Literal, Protocol, Self

from pydantic import BeforeValidator, Field, TypeAdapter, ValidationError, model_validator

from quotemem.core.add_outcome import AddOutcomeV1
from quotemem.core.entities import USER, EntityRef, QuoteSpanV1, ReasonCode, Rejudge
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID
from quotemem.core.outcome_table import OutcomeCategory, outcome_rule
from quotemem.core.query import ScopeV1
from quotemem.core.request_budget import MAX_OWNER_RESUMES
from quotemem.core.technical_types import ExternalId
from quotemem.publication.history import MemoryReads, MemoryRows, WireValue, cite, json_tuple

MAX_ADD_ITEMS_LIMIT: Final = 1000
"""The most items one add-items page returns (``limit``)."""

ITEM_REF_PATTERN: Final = r"^[0-7][0-9A-HJKMNP-TV-Z]{25}\.[0-9]{1,5}\.[0-9]{1,3}$"
"""``{owner_run_id}.{unit}.{position}``."""

ItemRef = Annotated[str, Field(pattern=ITEM_REF_PATTERN)]
"""One Add item's public reference."""

_ITEM_REF: Final[TypeAdapter[str]] = TypeAdapter(ItemRef)
_EXTERNAL: Final[TypeAdapter[str]] = TypeAdapter(ExternalId)


def item_ref(owner_run_id: str, unit: int, position: int) -> str:
    """Return the public reference of one position, ``{owner_run_id}.{unit}.{position}``."""
    return f"{owner_run_id}.{unit}.{position}"


def item_key(reference: str) -> tuple[str, int, int]:
    """Parse an item reference into the ``(owner_run_id, unit, position)`` it sorts by.

    Raises:
        QuoteMemError: ``invalid_request`` for a malformed reference.
    """
    try:
        owner, unit, position = _ITEM_REF.validate_python(reference, strict=True).split(".")
    except ValidationError:
        raise QuoteMemError("invalid_request", "An item reference is malformed") from None
    return owner, int(unit), int(position)


class AddItemV1(WireValue):
    """One Add item (``AddItem``): content-free ids and the frozen disposition."""

    item_ref: ItemRef
    owner_run_id: ULID
    unit: Annotated[int, Field(ge=0, le=65_535)]
    position: Annotated[int, Field(ge=0, le=511)]
    category: OutcomeCategory
    reason_code: ReasonCode
    rejudge: Rejudge
    memory_ids: Annotated[tuple[ULID, ...], BeforeValidator(json_tuple)]
    target_memory_ids: Annotated[tuple[ULID, ...], BeforeValidator(json_tuple)]
    entity_ids: Annotated[tuple[EntityRef, ...], BeforeValidator(json_tuple)]

    @model_validator(mode="after")
    def check_identity(self) -> Self:
        """Keep ``item_ref`` the item's own position and the disposition the code's."""
        if self.item_ref != item_ref(self.owner_run_id, self.unit, self.position):
            raise ValueError("An item reference names its own position")
        rule = outcome_rule(self.reason_code)
        if (self.category, self.rejudge) != (rule.category, rule.rejudge):
            raise ValueError("An item's category and rejudge are its code's")
        return self


class AddItemsV1(WireValue):
    """One page of a request's Add items (``data``).

    ``not_completed`` carries no ids, no outcome and no items. A completed
    page lists items after ``after_item_ref`` in reference order;
    ``next_item_ref`` names the last item of a full page when more follow.
    """

    request_id: ExternalId
    session_id: ExternalId
    state: Literal["completed", "not_completed"]
    completed_by_run_id: ULID | None
    recovery_count: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)] | None
    outcome: AddOutcomeV1 | None
    items: Annotated[tuple[AddItemV1, ...], BeforeValidator(json_tuple)]
    next_item_ref: ItemRef | None

    @model_validator(mode="after")
    def check_state(self) -> Self:
        """Carry ids and an outcome exactly for a completed request, items in order."""
        completed = self.state == "completed"
        if any(
            (value is None) == completed
            for value in (self.completed_by_run_id, self.recovery_count, self.outcome)
        ):
            raise ValueError("A completed request carries its run, recovery count and outcome")
        if not completed and (self.items or self.next_item_ref is not None):
            raise ValueError("A request that is not completed lists nothing")
        keys = [item_key(item.item_ref) for item in self.items]
        if keys != sorted(set(keys)):
            raise ValueError("Add items are unique and in reference order")
        if self.next_item_ref is not None and (
            not self.items or self.next_item_ref != self.items[-1].item_ref
        ):
            raise ValueError("A continuation names the page's last item")
        return self


@dataclass(frozen=True, slots=True)
class FrozenAddItem:
    """One item of a request's completion record, as frozen at first completion.

    Attributes:
        owner_run_id (str): The Add owner that admitted the position.
        unit (int): Its unit.
        position (int): Its position in the unit's reply.
        reason_code (ReasonCode): Its single final code.
        candidate_id (str | None): The position's candidate row; None for a
            position admission refused, which has no row.
        revision_id (str | None): The revision a formal item wrote; None otherwise.
        memory_ids (tuple[str, ...]): The written memory or the one already
            stating the fact, as frozen.
        target_memory_ids (tuple[str, ...]): Ended or contradicted targets, or
            an unresolved item's judged targets, as frozen.
        entity_ids (tuple[str, ...]): Its anchors, primary first.
        quote_spans (tuple[QuoteSpanV1, ...]): The position's admitted quote
            spans, sorted and unique; never returned by add-items.
        judge_result_ref (str | None): The ``result_index:judge`` frozen with the
            item (its verdict or exhausted-timeout result); never the candidate
            row's current one, which a later re-entry replaces. Never returned
            by add-items.
    """

    owner_run_id: str
    unit: int
    position: int
    reason_code: ReasonCode
    candidate_id: str | None
    revision_id: str | None
    memory_ids: tuple[str, ...]
    target_memory_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    quote_spans: tuple[QuoteSpanV1, ...]
    judge_result_ref: str | None = None

    @property
    def key(self) -> tuple[str, int, int]:
        """The item's sort key, ``(owner_run_id, unit, position)``."""
        return self.owner_run_id, self.unit, self.position

    @property
    def ref(self) -> str:
        """The item's public reference."""
        return item_ref(self.owner_run_id, self.unit, self.position)


@dataclass(frozen=True, slots=True)
class FrozenCompletion:
    """One request's completion record, frozen at its first completion.

    Attributes:
        completed_by_run_id (str): The owner whose success completed it.
        recovery_count (int): The owner's counted automatic/explicit resumes, 0 to 33.
        outcome (AddOutcomeV1): The six counts its items sum to.
        items (tuple[FrozenAddItem, ...]): Its items in reference order.
    """

    completed_by_run_id: str
    recovery_count: int
    outcome: AddOutcomeV1
    items: tuple[FrozenAddItem, ...]


def frozen_item(items: Sequence[FrozenAddItem], reference: str) -> FrozenAddItem | None:
    """Find one frozen item by its public reference, or None."""
    wanted = item_key(reference)
    return next((item for item in items if item.key == wanted), None)


class AddCompletions(Protocol):
    """The frozen completion records of Add requests (the completion writer's reader)."""

    def completion(
        self, rows: MemoryRows, session_id: str, request_id: str,
    ) -> FrozenCompletion | None:
        """Read one request's completion record in the read's exact scope.

        Reads go through ``rows`` so the read's delivery recheck covers them.

        Args:
            rows (MemoryRows): The read's guarded rows; their scope is the caller's.
            session_id (str): The request's session.
            request_id (str): The request.

        Returns:
            FrozenCompletion | None: The record, or None when the scope has no
                completed record of this request.
        """
        ...


def not_completed(session_id: str, request_id: str) -> AddItemsV1:
    """The one answer for a request without a completed record in the caller's scope."""
    return AddItemsV1(
        request_id=request_id, session_id=session_id, state="not_completed",
        completed_by_run_id=None, recovery_count=None, outcome=None, items=(),
        next_item_ref=None,
    )


def readable_memories(rows: MemoryRows, memory_ids: Sequence[str]) -> tuple[str, ...]:
    """Record the memories behind ``memory_ids`` and their heads' Evidence; keep what exists.

    Call `readable_memory` on each kept id after `MemoryRows.finish` to apply ``p_ver``.
    """
    kept: list[str] = []
    for memory_id in memory_ids:
        item = rows.memory(memory_id)
        if item is not None and not rows.withheld(item):
            cite(rows, rows.head(item).snapshot)
            kept.append(memory_id)
    return tuple(kept)


def readable_memory(rows: MemoryRows, memory_id: str, now: int) -> bool:
    """Whether a recorded memory exists and its head passes ``p_ver`` at ``now``."""
    item = rows.memory(memory_id)
    return item is not None and not rows.withheld(item) and rows.readability(now).p_ver(
        rows.head(item).snapshot)


def active_entities(rows: MemoryRows, entity_ids: Sequence[str]) -> tuple[str, ...]:
    """Keep the anchors that are ``user`` or an active entity of the exact scope."""
    kept: list[str] = []
    for entity_id in entity_ids:
        if entity_id == USER:
            kept.append(entity_id)
            continue
        entity = rows.row("entity", entity_id)
        if entity is not None and entity["state"] == "active":
            kept.append(entity_id)
    return tuple(kept)


def read_add_items(
    rows: MemoryRows,
    completions: AddCompletions,
    *,
    session_id: str,
    request_id: str,
    after_item_ref: str | None,
    limit: int,
) -> AddItemsV1:
    """Read one page of a request's Add items inside a pinned scope's callback.

    Args:
        rows (MemoryRows): The read's guarded rows.
        completions (AddCompletions): The completion records.
        session_id (str): The request's session.
        request_id (str): The request.
        after_item_ref (str | None): List only items after this reference.
        limit (int): The most items, 1 through `MAX_ADD_ITEMS_LIMIT`.

    Returns:
        AddItemsV1: The page, or ``not_completed``.

    Raises:
        QuoteMemError: ``reception_stale`` when the store changed during the read.
    """
    rows.authorize()
    record = completions.completion(rows, session_id, request_id)
    if record is None:
        rows.finish()
        return not_completed(session_id, request_id)
    after = None if after_item_ref is None else item_key(after_item_ref)
    remaining = [item for item in record.items if after is None or item.key > after]
    page = remaining[:limit]
    kept = [
        (
            item, readable_memories(rows, item.memory_ids),
            readable_memories(rows, item.target_memory_ids),
            active_entities(rows, item.entity_ids),
        )
        for item in page
    ]
    rows.finish()
    now = rows.clock()
    items = tuple(
        AddItemV1(
            item_ref=item.ref, owner_run_id=item.owner_run_id, unit=item.unit,
            position=item.position, category=outcome_rule(item.reason_code).category,
            reason_code=item.reason_code, rejudge=outcome_rule(item.reason_code).rejudge,
            memory_ids=tuple(found for found in memories if readable_memory(rows, found, now)),
            target_memory_ids=tuple(
                found for found in targets if readable_memory(rows, found, now)
            ),
            entity_ids=entities,
        )
        for item, memories, targets, entities in kept
    )
    return AddItemsV1(
        request_id=request_id, session_id=session_id, state="completed",
        completed_by_run_id=record.completed_by_run_id, recovery_count=record.recovery_count,
        outcome=record.outcome, items=items,
        next_item_ref=items[-1].item_ref if len(remaining) > limit else None,
    )


class AddItemsReader:
    """Answer which items one completed Add request produced.

    Args:
        reads (MemoryReads): The shared guarded read entry.
        completions (AddCompletions): The completion records.
    """

    def __init__(self, reads: MemoryReads, completions: AddCompletions) -> None:
        self._reads, self._completions = reads, completions

    def consume(
        self,
        *,
        scope: ScopeV1,
        session_id: str,
        request_id: str,
        consumer: Callable[[AddItemsV1], None],
        after_item_ref: str | None = None,
        limit: int = MAX_ADD_ITEMS_LIMIT,
    ) -> None:
        """Deliver one page of a request's items while the scope's pin is held.

        Args:
            scope (ScopeV1): The caller's exact scope.
            session_id (str): The request's session.
            request_id (str): The request.
            consumer (Callable[[AddItemsV1], None]): Called once, inside the
                pinned callback, with the page.
            after_item_ref (str | None): List only items after this reference.
            limit (int): The most items, 1 through 1000.

        A scope the caller cannot read, or that has no rows yet, holds no
        completed record of the request: the answer is ``not_completed``, never
        ``scope_denied``. An Add owner holding the scope, running, paused
        or refused, never fences the read.

        Raises:
            QuoteMemError: ``invalid_request`` for an invalid id, reference or limit;
                ``reception_stale`` when the store changed during the read; the
                shared entry's other refusals.
        """
        try:
            session = _EXTERNAL.validate_python(session_id, strict=True)
            request = _EXTERNAL.validate_python(request_id, strict=True)
        except ValidationError:
            raise QuoteMemError("invalid_request",
                                "Add items need a session and a request id") from None
        if after_item_ref is not None:
            item_key(after_item_ref)
        if type(limit) is not int or not 1 <= limit <= MAX_ADD_ITEMS_LIMIT:
            raise QuoteMemError("invalid_request", "Add items limit is outside 1 to 1000")
        try:
            self._reads.consume(
                scope,
                lambda rows: read_add_items(
                    rows, self._completions, session_id=session, request_id=request,
                    after_item_ref=after_item_ref, limit=limit,
                ),
                consumer,
                completions=True,
            )
        except QuoteMemError as error:
            if error.code != "scope_denied":
                raise
            consumer(not_completed(session, request))
