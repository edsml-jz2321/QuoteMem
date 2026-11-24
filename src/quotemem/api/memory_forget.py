"""Explicit memory-level forget through the single Writer's application owner.

Preparation selects current content and issues a bounded intent. Confirmation
must repeat that exact selection. Only the verified final receipt establishes
completion; transport acknowledgement or an admitted worker does not.

Forget by statement prepares a plan instead: the memories that state one
statement, picked by the product's judge, each with its exact head. Its
confirmation repeats that plan and forgets every target through the same
forget, one after another; each target's forget is atomic and the statement as
a whole is resumable by repeating the confirmation.

In the ``agent`` Add mode (``explicit-request-v2``) a plan also names sentences of the
user's Evidence that state the statement (``spans``): the confirmation withholds them with
the first forgotten memory, or, when no memory states the fact, through a forget of the
sentences alone (``span_intent``), whose outcome names no memory. A target
bundling the fact with an unrelated one is named in ``bundled``, with the user's
``statement``: its forget tombstones the user's words, not the bundle.
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quotemem.core.entities import QuoteSpanV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.forget import ForgetReceiptV3
from quotemem.core.forget_match import Exclusion
from quotemem.core.forget_pick import (
    FORGET_PICK_PROMPT_VERSION,
    MAX_FORGET_CANDIDATES,
    MAX_FORGET_PICK_VOTES,
    MAX_FORGET_STATEMENT_CHARS,
    Method,
    Role,
    Stage,
)
from quotemem.core.forget_state import MAX_RESTATING_SPANS
from quotemem.core.identity import ULID, ArtifactId, IntegrityDigest
from quotemem.core.technical_types import (
    ContentDigest,
    ExternalId,
    LocalId,
    NonNegativeInt,
    Timestamp,
)
from quotemem.service.lifecycle import ServiceLifetime, ServiceUnavailable
from quotemem.store.metadata import StrictModel


class MemoryForgetPrepareRequestV1(StrictModel):
    """Select one memory for an explicit operation in the authenticated user scope."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    schema_version: Literal[1]
    user_id: Annotated[ExternalId, Field(repr=False)]
    client_session_id: Annotated[ExternalId, Field(repr=False)]
    request_id: Annotated[ExternalId, Field(repr=False)]
    memory_id: ULID
    expected_revision_id: ULID

    @field_validator("user_id", "client_session_id", "request_id")
    @classmethod
    def valid_identity(cls, value: str) -> str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Operation identity must contain valid Unicode") from None
        return value


class MemoryForgetRequestV1(MemoryForgetPrepareRequestV1):
    """Confirm the exact prepared current head; never infer consent from free text."""

    expected_content_hash: IntegrityDigest
    intent_key: Annotated[LocalId, Field(repr=False)]


class ForgetStatementAffectedV1(StrictModel):
    """A memory outside what the user picked that a forget changes anyway (``agent`` mode).

    Clause-level withholding keeps another memory's own clause readable where it can; a
    memory whose only support still lies in what the forget withholds cannot keep it, and a
    memory the forget takes with its target (a correction's refuted value, or a later memory
    that replaced it and still names it: "... instead of playing badminton") goes with it. A
    prepare lists each such memory beforehand with its ``statement`` (``unreadable``: it may
    stop being readable; ``forgotten``: it is forgotten too), and the statement confirmation
    lists what happened to each memory outside the plan's targets that was readable before
    it and is no longer: ``unreadable`` (it stays, unreadable until told again) or
    ``forgotten``. Nothing of a forgotten memory's words is returned by the confirmation.

    A memory that tells the forgotten detail beside other facts ("Lin Xiaoyu is an architect." after
    "forget my wife's name") is ``reworded`` instead: the prepare lists it with the statement
    it will have ("The user's wife is an architect.", which shows nothing forgotten), and the
    confirmation lists exactly those memories once they state only the rest, still readable
    and found. A memory the forget cannot reword safely is listed ``unreadable`` or
    ``forgotten`` by both."""

    model_config = ConfigDict(frozen=True)
    memory_id: ULID
    effect: Literal["unreadable", "forgotten", "reworded"]
    statement: Annotated[str | None, Field(repr=False,
                                           exclude_if=lambda value: value is None)] = None


class MemoryForgetPreparedV2(StrictModel):
    """The complete confirmation body, its expiry and the retention notice.

    The notice discloses before confirmation that forget keeps an unreadable
    tombstone to block relearning; nothing here claims any deletion.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[2] = 2
    status: Literal["confirmation_required"] = "confirmation_required"
    confirmation: MemoryForgetRequestV1
    expires_at: Timestamp
    retention_notice: Literal["forget_keeps_unreadable_tombstone_v1"] = (
        "forget_keeps_unreadable_tombstone_v1"
    )
    affected: Annotated[list[ForgetStatementAffectedV1],
                        Field(exclude_if=lambda value: not value)] = []
    """The other memories the confirmation forgets or may leave unreadable (``agent`` mode):
    what the forget takes with the memory, and a kept memory whose only support it withholds."""


class MemoryForgetResponseV2(StrictModel):
    """Verified forget with its retained receipt v3 and exact client-operation binding."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[2] = 2
    status: Literal["verified"] = "verified"
    request_digest: IntegrityDigest
    controller_run_id: ULID
    receipt_ref: ArtifactId
    receipt: ForgetReceiptV3
    scope_state: Literal["ready", "maintenance_pending"]


class MemoryForgetStatementPrepareRequestV1(StrictModel):
    """Name one statement to forget in the authenticated user scope."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    schema_version: Literal[1]
    user_id: Annotated[ExternalId, Field(repr=False)]
    client_session_id: Annotated[ExternalId, Field(repr=False)]
    request_id: Annotated[ExternalId, Field(repr=False)]
    statement: Annotated[str, Field(min_length=1, max_length=MAX_FORGET_STATEMENT_CHARS,
                                    repr=False)]

    @field_validator("user_id", "client_session_id", "request_id", "statement")
    @classmethod
    def valid_text(cls, value: str) -> str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Operation text must contain valid Unicode") from None
        if not value.strip():
            raise ValueError("Operation text must not be blank")
        return value


class ForgetStatementTargetV1(StrictModel):
    """One picked memory with the exact head its forget binds."""

    model_config = ConfigDict(frozen=True)
    memory_id: ULID
    expected_revision_id: ULID
    expected_content_hash: IntegrityDigest


class ForgetStatementCandidateRefV1(StrictModel):
    """One candidate the pick read, at the head it read."""

    model_config = ConfigDict(frozen=True)
    memory_id: ULID
    revision_id: ULID


MAX_FORGET_TARGETS = 64
"""Memories one statement plan forgets: the judge's picks and, in the ``agent`` mode, the
other memories it finds stating the statement."""


class MemoryForgetStatementRequestV1(StrictModel):
    """Confirm a prepared statement plan exactly as prepared; never inferred from text.

    ``plan_binding`` is the workspace's keyed binding of the whole plan (principal,
    session, request, statement, targets, candidates, sentences, ``erase_epoch``, expiry),
    so a confirmation names only a plan this workspace prepared. ``spans`` are the
    sentences that state the statement (``agent`` mode); ``span_intent`` is the prepared
    forget of those sentences when no memory is a target.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    schema_version: Literal[1]
    user_id: Annotated[ExternalId, Field(repr=False)]
    client_session_id: Annotated[ExternalId, Field(repr=False)]
    request_id: Annotated[ExternalId, Field(repr=False)]
    targets: Annotated[list[ForgetStatementTargetV1], Field(max_length=MAX_FORGET_TARGETS)]
    candidates: Annotated[list[ForgetStatementCandidateRefV1],
                          Field(max_length=MAX_FORGET_CANDIDATES)]
    erase_epoch: NonNegativeInt
    expires_at: Timestamp
    plan_binding: ContentDigest
    spans: Annotated[list[QuoteSpanV1], Field(max_length=MAX_RESTATING_SPANS,
                                              exclude_if=lambda value: not value)] = []
    span_intent: Annotated[str | None, Field(max_length=512, repr=False,
                                             exclude_if=lambda value: value is None)] = None
    bundled: Annotated[list[ULID], Field(max_length=MAX_FORGET_TARGETS,
                                         exclude_if=lambda value: not value)] = []
    """Targets that bundle the fact with an unrelated one (``agent`` mode): each
    is forgotten whole, and its tombstone states ``statement`` instead of the bundle, so the
    unrelated fact may be told again."""
    statement: Annotated[str | None, Field(min_length=1, max_length=MAX_FORGET_STATEMENT_CHARS,
                                           repr=False,
                                           exclude_if=lambda value: value is None)] = None
    """The user's words of the fact, given exactly when ``bundled`` names a target."""

    @field_validator("user_id", "client_session_id", "request_id")
    @classmethod
    def valid_identity(cls, value: str) -> str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Operation identity must contain valid Unicode") from None
        return value

    @model_validator(mode="after")
    def forgets_something(self) -> Self:
        if not self.targets and not self.spans:
            raise ValueError("A statement plan forgets a memory or a sentence")
        if self.span_intent is not None and (self.targets or not self.spans):
            raise ValueError("Only a plan of sentences alone carries their prepared forget")
        if self.targets and not self.candidates:
            raise ValueError("A plan of memories names the candidates it read")
        if (self.statement is None) != (not self.bundled) or not set(self.bundled) <= {
                target.memory_id for target in self.targets}:
            raise ValueError("Bundled targets are targets, given with the user's statement")
        return self


class ForgetStatementCandidateV1(StrictModel):
    """One candidate the pick read: its head, statement, decided role and whether picked."""

    model_config = ConfigDict(frozen=True)
    memory_id: ULID
    revision_id: ULID
    statement: Annotated[str, Field(repr=False)]
    role: Role
    target: bool
    excluded: Annotated[Exclusion | None, Field(exclude_if=lambda value: value is None)] = None
    """Why a candidate the judge picked is no target (``agent`` mode)."""
    bundled: Annotated[bool, Field(exclude_if=lambda value: not value)] = False
    """A target that also states a fact the judge finds unrelated to the statement (``agent``
    mode): it is forgotten whole, as its words state the fact; a client may tell the user
    what else it said."""


class ForgetStatementSpanV1(StrictModel):
    """One sentence of the user's Evidence the ``agent`` mode's prepare judged:
    its position, text, decided role, whether the plan withholds it, and why not."""

    model_config = ConfigDict(frozen=True)
    evidence_id: ULID
    byte_start: NonNegativeInt
    byte_end: NonNegativeInt
    text: Annotated[str, Field(repr=False)]
    role: Role | None
    """None for a forget request asking this statement, taken without a vote."""
    target: bool
    excluded: Annotated[Exclusion | None, Field(exclude_if=lambda value: value is None)] = None


class ForgetStatementCallsV1(StrictModel):
    """The provider calls one ``agent``-mode prepare made: judge votes and embedding calls."""

    model_config = ConfigDict(frozen=True)
    judge: NonNegativeInt
    embedding: NonNegativeInt


type PickPromptVersion = Literal["memory-forget-pick-v1", "memory-forget-pick-detail-v1"]
"""The prompt a pick's classify votes sent: the pick's, or a detail request's."""


class ForgetStatementVoteV1(StrictModel):
    """One decision's answered votes (content-free)."""

    model_config = ConfigDict(frozen=True)
    stage: Stage
    key: Annotated[str, Field(min_length=1, max_length=256)]
    votes: dict[str, Annotated[int, Field(ge=0, le=MAX_FORGET_PICK_VOTES)]]
    decided: Annotated[str, Field(min_length=1, max_length=32)]
    fallback: bool


class ForgetStatementDecisionV1(StrictModel):
    """How the judge reached the targets: method, calls, fallbacks and every decision's votes."""

    model_config = ConfigDict(frozen=True)
    prompt_version: PickPromptVersion = FORGET_PICK_PROMPT_VERSION
    prompt_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    method: Method | None
    calls: NonNegativeInt
    answered: NonNegativeInt
    fallbacks: NonNegativeInt
    votes: list[ForgetStatementVoteV1]


class ForgetStatementEarlierV1(StrictModel):
    """What the two forget paths had done for the same request of the session (``agent`` mode):
    the forget by statement (its prepare's result or its confirmation) and the
    in-chat forget an Add holding the same "please forget ..." ran, with the memories and
    sentences both forgot for it. A plugin reads it to tell the user the final outcome."""

    model_config = ConfigDict(frozen=True)
    statement: Literal["nothing_matched", "prepared", "forgotten"] | None
    in_chat: Literal["nothing_matched", "forgotten"] | None
    forgotten: NonNegativeInt
    withheld: NonNegativeInt
    at: Timestamp


class MemoryForgetStatementPreparedV1(StrictModel):
    """The statement plan to confirm, or ``nothing_matched`` with what was read.

    ``confirmation`` is the complete body to send back; it is null exactly when
    nothing matched. ``decision`` is null when the Search found no current memory,
    so no judge call was made.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    status: Literal["confirmation_required", "nothing_matched"]
    confirmation: MemoryForgetStatementRequestV1 | None
    expires_at: Timestamp | None
    retention_notice: Literal["forget_keeps_unreadable_tombstone_v1"] = (
        "forget_keeps_unreadable_tombstone_v1"
    )
    candidates: list[ForgetStatementCandidateV1]
    decision: ForgetStatementDecisionV1 | None
    spans: Annotated[list[ForgetStatementSpanV1], Field(exclude_if=lambda value: not value)] = []
    """The sentences judged (``agent`` mode); those with ``target`` are the plan's."""
    calls: Annotated[ForgetStatementCallsV1 | None,
                     Field(exclude_if=lambda value: value is None)] = None
    """Every provider call of the prepare (``agent`` mode)."""
    earlier: Annotated[ForgetStatementEarlierV1 | None,
                       Field(exclude_if=lambda value: value is None)] = None
    """What either forget path did for the same request of the session before this prepare
    (``agent`` mode): an Add's in-chat forget may have forgotten it already."""
    affected: Annotated[list[ForgetStatementAffectedV1],
                        Field(exclude_if=lambda value: not value)] = []
    """The memories outside the targets the confirmation may leave unreadable (``agent``
    mode): a kept memory whose only support the forget withholds."""

    @model_validator(mode="after")
    def plan_fields(self) -> Self:
        if (self.status == "nothing_matched") != (self.confirmation is None) or (
                (self.confirmation is None) != (self.expires_at is None)):
            raise ValueError("Only a plan to confirm carries a confirmation and expiry")
        return self


class ForgetStatementOutcomeV1(StrictModel):
    """One target's result: forgotten now with its receipt, gone before, or changed.

    ``forgotten_earlier``: the memory no longer exists (an earlier confirmation of this
    plan, or another forget or erase, removed it). ``changed``: its head moved after the
    plan was prepared, so it was not forgotten. The forget of a plan's sentences alone
    names no memory: its ``withheld_spans`` counts them, and
    ``forgotten_earlier`` says a forget already withheld them all.
    """

    model_config = ConfigDict(frozen=True)
    memory_id: ULID | None
    withheld_spans: Annotated[NonNegativeInt | None,
                              Field(exclude_if=lambda value: value is None)] = None
    outcome: Literal["verified", "forgotten_earlier", "changed"]
    controller_run_id: ULID | None = None
    receipt_ref: ArtifactId | None = None
    receipt: ForgetReceiptV3 | None = None


class MemoryForgetStatementResponseV1(StrictModel):
    """Every target's outcome; ``verified`` only when no target is left current."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    status: Literal["verified", "incomplete"]
    request_digest: IntegrityDigest
    outcomes: list[ForgetStatementOutcomeV1]
    scope_state: Literal["ready", "maintenance_pending"]
    earlier: Annotated[ForgetStatementEarlierV1 | None,
                       Field(exclude_if=lambda value: value is None)] = None
    """Both forget paths' record of this request after the confirmation (``agent`` mode),
    the in-chat forget's included; a repeated confirmation reads it again."""
    forgotten: Annotated[NonNegativeInt | None, Field(exclude_if=lambda value: value is None)
                         ] = None
    """Memories this confirmation removed (``agent`` mode): its targets, the memories left
    without readable support that it forgot too, and every other memory that stated the fact
    and is no longer readable; never one of ``affected``."""
    withheld_spans: Annotated[NonNegativeInt | None,
                              Field(exclude_if=lambda value: value is None)] = None
    """Sentences this confirmation withheld (``agent`` mode)."""
    affected: Annotated[list[ForgetStatementAffectedV1],
                        Field(exclude_if=lambda value: not value)] = []
    """Memories the plan kept that this confirmation changed all the same (``agent`` mode)."""


class MemoryForgetOperations(Protocol):
    """The initialized single-Writer owner authorizes and executes both operations."""

    def prepare_memory_forget(
        self, request: MemoryForgetPrepareRequestV1,
    ) -> MemoryForgetPreparedV2: ...

    def forget_memory(self, request: MemoryForgetRequestV1) -> MemoryForgetResponseV2: ...


@runtime_checkable
class StatementForgetOperations(Protocol):
    """The owner's forget by statement: prepare a plan, then confirm it."""

    def prepare_statement_forget(
        self, request: MemoryForgetStatementPrepareRequestV1,
    ) -> MemoryForgetStatementPreparedV1: ...

    def forget_statement(
        self, request: MemoryForgetStatementRequestV1,
    ) -> MemoryForgetStatementResponseV1: ...


type ForgetRequest = (MemoryForgetPrepareRequestV1 | MemoryForgetStatementPrepareRequestV1
                      | MemoryForgetStatementRequestV1)


def parse_memory_forget_request[RequestT: ForgetRequest](
    raw: object, model: type[RequestT],
) -> RequestT:
    """Validate wire or SDK input while dropping payload-bearing validation errors."""
    if isinstance(raw, StrictModel):
        payload = dict(raw.__dict__)
        payload.update(raw.__pydantic_extra__ or {})
        raw = payload
    try:
        return model.model_validate(raw)
    except ValidationError:
        raise QuoteMemError("invalid_request", "Memory forget selection is invalid") from None


class MemoryForgetService:
    """Share the running application's admission and shutdown drain without new workers."""

    def __init__(self, operations: MemoryForgetOperations, lifetime: ServiceLifetime) -> None:
        self._operations, self._lifetime = operations, lifetime

    def prepare_memory_forget(
        self, request: MemoryForgetPrepareRequestV1,
    ) -> MemoryForgetPreparedV2:
        try:
            with self._lifetime.admission():
                return self._operations.prepare_memory_forget(
                    parse_memory_forget_request(request, MemoryForgetPrepareRequestV1),
                )
        except ServiceUnavailable:
            raise QuoteMemError("service_unhealthy", "Memory forget is unavailable") from None

    def forget_memory(self, request: MemoryForgetRequestV1) -> MemoryForgetResponseV2:
        try:
            with self._lifetime.admission():
                return self._operations.forget_memory(
                    parse_memory_forget_request(request, MemoryForgetRequestV1),
                )
        except ServiceUnavailable:
            raise QuoteMemError("service_unhealthy", "Memory forget is unavailable") from None


class StatementForgetService:
    """Admit forget by statement under the running application's lifetime, like forget."""

    def __init__(self, operations: StatementForgetOperations, lifetime: ServiceLifetime) -> None:
        self._operations, self._lifetime = operations, lifetime

    def prepare_statement_forget(
        self, request: MemoryForgetStatementPrepareRequestV1,
    ) -> MemoryForgetStatementPreparedV1:
        try:
            with self._lifetime.admission():
                return self._operations.prepare_statement_forget(
                    parse_memory_forget_request(request, MemoryForgetStatementPrepareRequestV1),
                )
        except ServiceUnavailable:
            raise QuoteMemError("service_unhealthy", "Memory forget is unavailable") from None

    def forget_statement(
        self, request: MemoryForgetStatementRequestV1,
    ) -> MemoryForgetStatementResponseV1:
        try:
            with self._lifetime.admission():
                return self._operations.forget_statement(
                    parse_memory_forget_request(request, MemoryForgetStatementRequestV1),
                )
        except ServiceUnavailable:
            raise QuoteMemError("service_unhealthy", "Memory forget is unavailable") from None
