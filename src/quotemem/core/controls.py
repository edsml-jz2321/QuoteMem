"""Closed Control values and namespace dispatch."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Annotated, Final, Literal, Self, cast

from pydantic import (
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from quotemem.core.add_outcome import CompletedAddOutcomeV1
from quotemem.core.entities import QuoteSpanV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_bounds import ExtractionRefusalDetailV1
from quotemem.core.forget_reword import MAX_REWORDINGS
from quotemem.core.forget_state import ErasureFenceV1, ForgetIntentV1
from quotemem.core.identity import (
    ULID,
    ArtifactId,
    ContentBindingDigest,
    ControlId,
    PrincipalId,
    ScopeId,
    control_id,
)
from quotemem.core.inference import InferenceBindingV1
from quotemem.core.judge_records import CheckedSourceV1, JudgeOutputLocatorV1
from quotemem.core.privacy_persistence import PrivacyEraseProgressV1
from quotemem.core.provider_budget import ExtractionOwnerBudgetV1
from quotemem.core.references import ReferenceProjectionV1
from quotemem.core.request_budget import DEFAULT_COMPLETION_WORK_BUDGET_MS, MAX_OWNER_RESUMES
from quotemem.core.source_labels import SourceLabelMessageReservationV1, SourceLabelsBudgetV1
from quotemem.core.statement import (
    MAX_KEYWORD_CHARS,
    MAX_KEYWORDS,
    MAX_STATEMENT_CHARS,
    MAX_SUPPORT_GROUPS,
    HexDigest,
)
from quotemem.core.technical_types import (
    ContentDigest,
    ExternalId,
    Integrity,
    LocalId,
    NonNegativeInt,
    PolicyReleaseRef,
    PositiveInt,
    RequestProgressV1,
    SessionHash,
    Timestamp,
)
from quotemem.store.metadata import DerivedSpec, OpaqueIdSpec, RefSpec, StrictModel

CanaryRef = Annotated[
    ArtifactId,
    RefSpec(
        "artifact",
        scope_rule="global",
        target_forget_action="retain_locator",
        artifact_kinds=("policy_config",),
    ),
]
LiveOwner = Annotated[ULID, RefSpec("run", scope_rule="same_scope")]
HistoricalOwner = Annotated[ULID, RefSpec("run", ref_mode="historical_locator", fulfillment="none")]


class WorkspaceBindingV1(StrictModel):
    """Bind a workspace once to its physical schema and embedding dimension."""

    schema_version: Literal[1]
    db_uuid: Annotated[str, OpaqueIdSpec("local")]
    workspace_id: LocalId
    tenant_digest: Integrity
    embedding_dimension: Literal[1024, 2048]
    schema_fingerprint: Integrity
    created_at: Timestamp


class CounterV1(StrictModel):
    """Store a monotonic nonnegative sequence value."""

    seq: NonNegativeInt


class PolicyPointerV1(StrictModel):
    """Retain the current global recall or extraction release."""

    current_release_ref: PolicyReleaseRef


class ActiveEmbeddingSpaceV1(StrictModel):
    """Retain the active embedding space and its frozen canary vectors."""

    state: Literal["active"]
    embedding_space: Annotated[str, Field(min_length=1, max_length=1024)]
    canary_digest: Integrity
    canary_baseline_ref: CanaryRef
    state_commit_seq: PositiveInt


class WriterLeaseV1(StrictModel):
    """Identify the single process holding the separately enforced file lock."""

    owner_instance_id: LocalId
    writer_epoch: PositiveInt
    acquired_at: Timestamp


class ScopeRegistrationV1(StrictModel):
    """Preserve the external scope identity for collision checks."""

    scope_kind: Literal["user", "project", "agent", "organization"]
    raw_external_id: ExternalId


class ScopeQuarantineV1(StrictModel):
    """Retain a scoped recovery refusal without changing its original authority."""

    owner_run_id: HistoricalOwner
    finalize_cycle: PositiveInt | None
    writer_epoch: PositiveInt
    stage: LocalId
    error_code: LocalId


class RequestLedgerValueV1(StrictModel):
    """Retain accepted request bindings through session retirement."""

    recent: Annotated[dict[ContentDigest, RequestProgressV1], Field(max_length=64)]
    ledger_ref: (
        Annotated[ArtifactId, RefSpec("artifact", artifact_kinds=("request_ledger",))] | None
    ) = None
    ledger_digest: ContentDigest | None = None

    @model_validator(mode="after")
    def validate_ledger_pair(self) -> Self:
        """Require both pieces of the persistent ledger-head identity."""
        if (self.ledger_ref is None) != (self.ledger_digest is None):
            raise ValueError("Request ledger reference and digest must appear together")
        return self


class PreparingFinalizeLeaseV1(StrictModel):
    """Freeze a single bounded drain cycle before installing an input fence."""

    state: Literal["preparing"]
    lease_generation: PositiveInt
    owner_run_id: LiveOwner
    writer_epoch: PositiveInt
    heartbeat_at: Timestamp
    drain_state: Literal["open", "closed"]
    grace_deadline: Timestamp


class HeldFinalizeLeaseV1(StrictModel):
    """Bind the held input fence to one retained baseline snapshot."""

    state: Literal["held"]
    lease_generation: PositiveInt
    owner_run_id: LiveOwner
    writer_epoch: PositiveInt
    heartbeat_at: Timestamp
    baseline_snapshot_generation: NonNegativeInt
    baseline_root_identity: Integrity


class ReleasedFinalizeLeaseV1(StrictModel):
    """Preserve the settled cycle identity as nonretaining history."""

    state: Literal["released"]
    lease_generation: PositiveInt
    owner_run_id: HistoricalOwner
    writer_epoch: PositiveInt
    heartbeat_at: Timestamp
    baseline_snapshot_generation: NonNegativeInt
    baseline_root_identity: Integrity
    released_at: Timestamp


class CancelledFinalizeLeaseV1(StrictModel):
    """A live cycle a forget cancelled: it never releases.

    The forget's erasure controller writes it (``updated_by_run_id``) in its
    admission, with the owner Run failed ``erase_fenced``; the next cycle is
    admitted once that forget completed and the cancelled cohort settled.
    ``repair_skip`` is an operator's skip of a quarantined cycle
    (``store.quarantine_skip``): the owner writes it, failed ``input_abandoned``.
    """

    state: Literal["cancelled"]
    lease_generation: PositiveInt
    owner_run_id: HistoricalOwner
    writer_epoch: PositiveInt
    heartbeat_at: Timestamp
    cancelled_from: Literal["preparing", "held"]
    cancelled_at: Timestamp
    cancel_reason: Literal["hard_erase", "repair_skip"]


FinalizeLeaseV1 = Annotated[
    PreparingFinalizeLeaseV1 | HeldFinalizeLeaseV1 | ReleasedFinalizeLeaseV1
    | CancelledFinalizeLeaseV1,
    Field(discriminator="state"),
]


MAX_HELD_REQUESTS: Final = 1024
"""Requests one scope's outliving Evidence holds may name."""
MAX_HELD_PARTS: Final = 1024
"""Evidence parts one held request may name."""


class HeldRequestV1(StrictModel):
    """The Evidence a cancelled owner's deleted pending rows of one request held.

    ``request_key`` is the request each row is attributed to: the newest request
    among its sources.
    """

    request_key: ContentDigest
    evidence_ids: Annotated[
        list[ULID], Field(min_length=1, max_length=MAX_HELD_PARTS),
        RefSpec("evidence", cardinality="many", scope_rule="same_scope"),
    ]

    @model_validator(mode="after")
    def validate_parts(self) -> Self:
        """Require sorted, unique Evidence ids."""
        if self.evidence_ids != sorted(set(self.evidence_ids)):
            raise ValueError("Held Evidence ids are sorted and unique")
        return self


MAX_FORGET_MARKS: Final = 4096
"""Sentences one scope's forget marks may name."""


class ForgetMarksV1(StrictModel):
    """The scope's sentences that state a fact its user asked to forget: ``forget_marks:{s}``.

    An in-chat forget keeps the scope's Evidence readable; each sentence the judge finds
    stating the forgotten fact is named here by its span, and Search shows it as something
    the user later asked not to use (``api/bench_render.py``). The spans name Evidence
    coordinates only; nothing of what they say is copied.
    """

    spans: Annotated[list[QuoteSpanV1], Field(min_length=1, max_length=MAX_FORGET_MARKS)]

    @model_validator(mode="after")
    def validate_marks(self) -> Self:
        """Require spans sorted by Evidence and position, each once."""
        keys = [(span.evidence_id, span.byte_start, span.byte_end) for span in self.spans]
        if keys != sorted(set(keys)):
            raise ValueError("Marked spans are sorted and unique")
        return self


MAX_FORGET_WITHDRAWN: Final = 4096
"""Summary-only memories one scope's forget record may withdraw."""


class ForgetWithdrawnV1(StrictModel):
    """Summary-only memories an in-chat forget withdrew: ``forget_withdrawn:{s}``.

    A memory of a summary-only origin (``recall.SUMMARY_ONLY_ORIGINS``: an overview or
    the assistant's work) recounts many points of a session, so a forgotten fact that is
    one of them is not forgotten with it, which would withhold every span it quotes;
    the judge's decision that it states the fact withdraws it here, and no Search
    delivers it again (``core/recall.recall_pure``). The ids name memories only.
    """

    memory_ids: Annotated[
        list[ULID], Field(min_length=1, max_length=MAX_FORGET_WITHDRAWN),
        RefSpec("item", cardinality="many", scope_rule="same_scope"),
    ]

    @model_validator(mode="after")
    def validate_withdrawn(self) -> Self:
        """Require sorted, unique memory ids."""
        if self.memory_ids != sorted(set(self.memory_ids)):
            raise ValueError("Withdrawn memory ids are sorted and unique")
        return self


MAX_FORGET_SHADOW_TERMS: Final = 4096
"""Name keys, and topic keywords, one scope's forget shadow holds at most (the oldest go)."""
MAX_SHADOW_TOPIC_TERMS: Final = 8
"""Terms one shadow topic keyword holds at most."""


class ForgetShadowV1(StrictModel):
    """What the scope's forgotten facts were about, as keyed digests: ``forget_shadow:{s}``.

    Every forget (so also each target of an in-chat forget) adds the name terms and
    topic keywords of the versions it tombstones, a forget of sentences those
    of its seed (``recall_agent.shadow_terms``), each term as the workspace's keyed digest
    ``forget-shadow-term``; nothing of what they say is copied. The interactive Search
    drops the hits they mark (``recall_agent.forget_shadowed``);
    the bench Search never reads it.
    """

    names: Annotated[list[ContentDigest], Field(max_length=MAX_FORGET_SHADOW_TERMS)]
    topics: Annotated[
        list[Annotated[list[ContentDigest],
                       Field(min_length=1, max_length=MAX_SHADOW_TOPIC_TERMS)]],
        Field(max_length=MAX_FORGET_SHADOW_TERMS),
    ]
    name_times: Annotated[list[Timestamp], Field(max_length=MAX_FORGET_SHADOW_TERMS,
                                                 exclude_if=lambda value: not value)] = []
    """When a complete forget (``explicit-request-v2``, the ``agent`` Add mode) last added
    each name, in ``names`` order; empty (left out) in every other mode. The interactive
    Search does not shadow with a name what the user stated again in their own words after
    it (``ingestion/relearn.py``)."""
    topic_times: Annotated[list[Timestamp], Field(max_length=MAX_FORGET_SHADOW_TERMS,
                                                  exclude_if=lambda value: not value)] = []
    """The same for each topic keyword, in ``topics`` order."""

    @model_validator(mode="after")
    def validate_shadow(self) -> Self:
        """Require sorted, unique names, topics and topic terms, and some content."""
        if self.names != sorted(set(self.names)):
            raise ValueError("Shadow names are sorted and unique")
        if any(topic != sorted(set(topic)) for topic in self.topics) or (
                self.topics != sorted({tuple(topic): topic for topic in self.topics}.values())):
            raise ValueError("Shadow topics are sorted and unique")
        if not self.names and not self.topics:
            raise ValueError("A forget shadow names something")
        if (self.name_times or self.topic_times) and (
                len(self.name_times) != len(self.names)
                or len(self.topic_times) != len(self.topics)):
            raise ValueError("Shadow times name every name and topic")
        return self


MAX_FORGET_HANDLED: Final = 256
"""Forget requests one scope's handled record keeps; the oldest go first."""
MAX_HANDLED_ASKED: Final = 8
"""Prepared plans that find one handled request: its latest (``asked``) and the newest alias
entries leading to it (`ForgetHandledEntryV1`)."""
type HandledStatement = Literal["nothing_matched", "prepared", "forgotten"]
type HandledInChat = Literal["nothing_matched", "forgotten"]


class ForgetHandledEntryV1(StrictModel):
    """What the two forget paths did for one request of one session.

    Attributes:
        request: The keyed digest of the session and the request's words
            (``forget-handled-request``); nothing of what they say is copied.
        asked: The keyed digest of the session, the ``request_id`` and the plan binding of
            the latest plan the forget by statement prepared for it, which a confirmation
            names; None when only the in-chat forget ran. An earlier plan of the same request
            is found through an alias entry: its ``request`` is a keyed digest of that plan's
            digest and its ``asked`` is the ``request`` of the entry it leads to; the alias
            carries nothing else (`MAX_HANDLED_ASKED`).
        statement_seq: The commit up to which the forget by statement's prepare read the
            scope: once it is confirmed or matched nothing, a later in-chat forget of the same
            request reads only what changed after it (an unconfirmed plan settles nothing).
        chat_seq: The commit up to which the in-chat forget of the same request read the
            scope: when it forgot something, a later prepare of it reads only what changed
            after it.
        at: When the entry last changed; once ``statement`` is ``forgotten``, when the
            confirmation forgot it.
        statement: What the forget by statement did: nothing matched, a plan was prepared,
            or its confirmation forgot it.
        in_chat: What the in-chat forget of the same request did after it: ``forgotten``
            only when it forgot a memory or withheld a sentence other than the request.
        forgotten: Memories the two paths forgot for it.
        withheld: Sentences they withheld for it.
    """

    request: ContentDigest
    asked: ContentDigest | None = None
    statement_seq: NonNegativeInt | None = None
    chat_seq: NonNegativeInt | None = None
    at: Timestamp
    statement: HandledStatement | None = None
    in_chat: HandledInChat | None = None
    forgotten: NonNegativeInt = 0
    withheld: NonNegativeInt = 0


MAX_FORGET_ADDS: Final = 256
"""Adds one scope's in-chat forget record keeps; the oldest go first."""
MAX_FORGET_ADD_REQUESTS: Final = 32
"""Requests one Add's in-chat forget record keeps."""


HANDLED_PURPOSE: Final = "forget-handled-request"
"""The keyed-digest purpose naming a request in ``forget_handled:{s}`` and an Add and its
requests in ``forget_add:{s}``."""


class ForgetAddRequestV1(StrictModel):
    """What one in-chat forget request of an Add came to: see ``ForgetAddEntryV1``.

    Attributes:
        request: The keyed digest of the Add, the request's position and its words
            (``forget-handled-request``); nothing of what they say is copied.
        status: Something was forgotten for it, nothing matched, or the plugin's forget by
            statement holds a plan for it not yet confirmed.
        forgotten: The memories removed for it.
        withheld: The sentences withheld for it.
        bundled: Its forgotten memories that bundled an unrelated fact.
        earlier: The plugin's forget by statement had forgotten it before the Add read it.
    """

    request: ContentDigest
    status: Literal["forgotten", "nothing_matched", "pending"]
    forgotten: NonNegativeInt = 0
    withheld: NonNegativeInt = 0
    bundled: NonNegativeInt = 0
    earlier: bool = False


class ForgetAddEntryV1(StrictModel):
    """One Add's in-chat forget: its requests' outcomes and ``taken``, the memories the
    Add itself made that its forget took (no memory it added)."""

    add: ContentDigest
    at: Timestamp
    taken: NonNegativeInt = 0
    requests: Annotated[list[ForgetAddRequestV1], Field(max_length=MAX_FORGET_ADD_REQUESTS)]


class ForgetAddV1(StrictModel):
    """The scope's recent Adds' in-chat forget outcomes: ``forget_add:{s}`` (``agent`` mode).

    An in-chat forget is never silent: the Add's answer reports each request's outcome,
    read from here on every resend of the same Add.
    """

    entries: Annotated[list[ForgetAddEntryV1], Field(min_length=1, max_length=MAX_FORGET_ADDS)]

    @model_validator(mode="after")
    def validate_adds(self) -> Self:
        """Require each Add once."""
        if len({entry.add for entry in self.entries}) != len(self.entries):
            raise ValueError("Recorded Adds are unique")
        return self


MAX_FORGET_KEPT: Final = 64
"""Complete forgets one scope's judged-and-kept record keeps; the oldest go first."""
MAX_FORGET_KEPT_PLANS: Final = 64
"""Prepared plans awaiting confirmation the record keeps; the oldest go first."""
MAX_FORGET_KEPT_HITS: Final = 512
"""Memories, and Evidence parts, one forget's record keeps."""

_LocalId = Annotated[ULID, OpaqueIdSpec("local")]


class ForgetKeptEntryV1(StrictModel):
    """What one complete forget judged and kept, so its own names do not hide it.

    Attributes:
        start: The forget's clock before it stamped any shadow term.
        end: Its clock after the last; a shadow name stamped in between is its own.
        memories: The memories it judged and kept, every readable quote naming one of its
            names lying in a sentence it judged and kept.
        parts: The Evidence parts each of whose readable sentences naming one of its
            names it judged and kept.
    """

    start: Timestamp
    end: Timestamp
    memories: Annotated[list[_LocalId], Field(max_length=MAX_FORGET_KEPT_HITS)] = []
    parts: Annotated[list[_LocalId], Field(max_length=MAX_FORGET_KEPT_HITS)] = []

    @model_validator(mode="after")
    def validate_entry(self) -> Self:
        """Require an ordered window and sorted, unique ids."""
        if self.end < self.start:
            raise ValueError("A kept record's window ends after it starts")
        if self.memories != sorted(set(self.memories)) or self.parts != sorted(set(self.parts)):
            raise ValueError("Kept ids are sorted and unique")
        return self


class ForgetRewordPlanV1(StrictModel):
    """A memory a complete prepare plans to reword instead of taking: its
    head's ``statement_digest``, the statement it reads once the forgotten detail is gone, the
    support and keywords it keeps and the tie that statement names the forgotten name's
    bearer by, if any (``ForgetRewordV1``, which the confirmation gives the statement's
    vector). The statement shows nothing forgotten."""

    memory_id: _LocalId
    base_digest: Annotated[HexDigest, OpaqueIdSpec("digest")]
    statement: Annotated[str, Field(min_length=1, max_length=MAX_STATEMENT_CHARS)]
    support: Annotated[list[list[QuoteSpanV1]], Field(max_length=MAX_SUPPORT_GROUPS)] = []
    keywords: Annotated[list[Annotated[str, Field(min_length=1, max_length=MAX_KEYWORD_CHARS)]],
                        Field(max_length=MAX_KEYWORDS)] = []
    tie: Annotated[str | None, Field(max_length=256, exclude_if=lambda value: value is None)] = None


class ForgetKeptPlanV1(StrictModel):
    """A complete prepare's judged-and-kept candidates, until its plan is confirmed.

    Attributes:
        plan: The plan's binding (``plan_binding``).
        expires_at: When the plan expires; the record goes with it.
        memories: The memories the prepare judged and did not take.
        spans: The sentences it judged and did not take.
        listed: The memories the prepare named as taken or changed (its targets and
            ``affected``): the confirmation's cascade forgets no other memory.
        past: The targets the prepare read past, not current: only they may be past at the
            confirmation.
        allowed: The memories a word rule nominated and the forget judge confirmed state or
            presuppose the fact: the only ones the confirmation takes beyond its picks and
            the base links.
        nickname: Whether the request asked a nickname, not the name ("forget my son's
            nickname"): its confirmation forgets the nicknames alone. Absent, a confirmation
            forgets every name of the one named, as for a request for the name.
        residue: The bare naming memories the prepare named as forgotten with the fact
            (its residue): the only bare names the confirmation's cascade takes.
        modifier: Whether the user asked to forget only a modifier of a thing of theirs
            (``forget_admission.asks_modifier``): the confirmation forgets as the prepare
            planned it, keeping a survivor about that thing (``ForgetIntentV1.modifier_only``).
        name_or_all: Whether the user's words ask a name (``forget_match.asks_name``) or
            everything about someone (``forget_run.asks_everything``): the
            confirmation's cascade takes the memories still showing a bare name it takes as
            the asked name's residue, as the prepare named them.
        rewordings: The memories the confirmation rewords instead of taking them.
        asked: The keyed digests of the shadow names of the values the user asked to forget
            (``ForgetIntentV1.asked``), which the confirmation's forgets carry.
        name_asked: Whether the user asked to forget a name (``ForgetIntentV1.name_asked``).
    """

    plan: ContentDigest
    expires_at: Timestamp
    memories: Annotated[list[_LocalId], Field(max_length=MAX_FORGET_KEPT_HITS)] = []
    spans: Annotated[list[QuoteSpanV1], Field(max_length=MAX_FORGET_KEPT_HITS)] = []
    listed: Annotated[list[_LocalId], Field(max_length=MAX_FORGET_KEPT_HITS)] = []
    past: Annotated[list[_LocalId], Field(max_length=MAX_FORGET_KEPT_HITS)] = []
    allowed: Annotated[list[_LocalId], Field(max_length=MAX_FORGET_KEPT_HITS,
                                             exclude_if=lambda value: not value)] = []
    nickname: Annotated[bool, Field(exclude_if=lambda value: not value)] = False
    residue: Annotated[list[_LocalId], Field(max_length=MAX_FORGET_KEPT_HITS,
                                             exclude_if=lambda value: not value)] = []
    modifier: Annotated[bool, Field(exclude_if=lambda value: not value)] = False
    name_or_all: Annotated[bool, Field(exclude_if=lambda value: not value)] = False
    rewordings: Annotated[list[ForgetRewordPlanV1], Field(
        max_length=MAX_REWORDINGS, exclude_if=lambda value: not value)] = []
    asked: Annotated[list[ContentDigest], Field(
        max_length=MAX_KEYWORDS, exclude_if=lambda value: not value)] = []
    name_asked: Annotated[bool, Field(exclude_if=lambda value: not value)] = False


class ForgetKeptV1(StrictModel):
    """What the scope's complete forgets judged and kept: ``forget_kept:{s}`` (``agent`` mode).

    A forget's shadow names (``forget_shadow:{s}``) hide every hit that names them from the
    interactive Search, except what that forget itself read and decided states another fact
    (``recall_agent.forget_shadowed``): a kept memory, a correction naming the old city.
    A hit it never judged (an older message naming the forgotten pet) stays hidden.
    """

    entries: Annotated[list[ForgetKeptEntryV1], Field(max_length=MAX_FORGET_KEPT)] = []
    plans: Annotated[list[ForgetKeptPlanV1], Field(max_length=MAX_FORGET_KEPT_PLANS)] = []

    @model_validator(mode="after")
    def validate_kept(self) -> Self:
        """Require each plan once."""
        if len({plan.plan for plan in self.plans}) != len(self.plans):
            raise ValueError("Kept plans are unique")
        return self


class ForgetHandledV1(StrictModel):
    """The scope's recent forget requests and what each path did: ``forget_handled:{s}``.

    The ``agent`` Add mode's plugin sends one "please forget ..." both as a forget by
    statement and inside its Add; both paths record here, so the in-chat forget never
    contradicts the plan the plugin already reported, and either path's response says what
    the other did (``StatementForget``).
    """

    entries: Annotated[list[ForgetHandledEntryV1],
                       Field(min_length=1, max_length=MAX_FORGET_HANDLED)]

    @model_validator(mode="after")
    def validate_handled(self) -> Self:
        """Require each request once."""
        if len({entry.request for entry in self.entries}) != len(self.entries):
            raise ValueError("Handled forget requests are unique")
        return self


class EvidenceHoldV1(StrictModel):
    """Holds that outlive a forget-cancelled cohort's rows: ``evidence_hold:{s}``.

    The cancelled owner's pending ``hold_evidence`` rows leave in the forget's
    deletion phase (``ForgetOwner._settle_cohort``), but the Evidence they held stays out of
    Evidence routes until the owner that reprocesses their requests settles
    them: its release removes the requests it completed
    (``owner_lease.release_finalize``). The read model adds these parts to the
    scope's held Evidence (``ScopeFence.held_evidence``).
    """

    requests: Annotated[
        list[HeldRequestV1], Field(min_length=1, max_length=MAX_HELD_REQUESTS),
    ]

    @model_validator(mode="after")
    def validate_requests(self) -> Self:
        """Require requests sorted and unique by key."""
        keys = [request.request_key for request in self.requests]
        if keys != sorted(set(keys)):
            raise ValueError("Held requests are sorted and unique by request key")
        return self


class FinalizedValueV1(StrictModel):
    """Record the released cycle and its frozen-input digest."""

    run_id: HistoricalOwner
    lease_generation: PositiveInt
    at: Timestamp
    input_heads_digest: ContentDigest
    finalized_scope_seq: NonNegativeInt


class RequestCompletionPolicyV2(StrictModel):
    """Explicitly select completed-prefix Add and one bounded shared owner."""

    contract_version: Literal[2] = 2
    work_budget_ms: Annotated[
        int, Field(ge=1, le=DEFAULT_COMPLETION_WORK_BUDGET_MS),
    ] = DEFAULT_COMPLETION_WORK_BUDGET_MS


class FinalizeBudgetV2(StrictModel):
    """One durable owner allowance, retained across yields, waiters and restart."""

    contract_version: Literal[2] = 2
    owner_run_id: HistoricalOwner
    owner_attempt: NonNegativeInt
    accepted_prefix_commit_seq: NonNegativeInt
    recovery_count: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)] = 0
    state: Literal["active", "exhausted"]
    admitted_at: Timestamp
    deadline_at: Timestamp
    erase_epoch: NonNegativeInt
    consent_epoch: NonNegativeInt

    @model_validator(mode="after")
    def bounded_deadline(self) -> Self:
        if not 0 < self.deadline_at - self.admitted_at <= DEFAULT_COMPLETION_WORK_BUDGET_MS:
            raise ValueError("Finalize owner budget must be positive and bounded")
        return self


class FinalizeBudgetRefusalV4(StrictModel):
    """An owner stopped by a whole-output extraction fault, keeping its original clock.

    The four faults (``provider_schema_invalid``, ``truncated``,
    ``output_bound``, ``no_fresh_source``) are the only refusals: an item fault
    never refuses an output, and a label-call failure pauses the owner.
    The record keeps the ``FinalizeBudgetV2`` fields of the allowance it ended,
    names the refused ``unit_no`` and copies that unit's content-free
    ``ExtractionRefusalDetailV1``. Nothing in the owner's
    prefix commits. A counted V3 completion resume replaces the refusal with
    a fresh allowance. It counts as a recovery, and the resumed owner re-runs
    only the refused unit's call.
    """

    contract_version: Literal[4] = 4
    owner_run_id: HistoricalOwner
    owner_attempt: NonNegativeInt
    accepted_prefix_commit_seq: NonNegativeInt
    recovery_count: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)] = 0
    state: Literal["refused"] = "refused"
    admitted_at: Timestamp
    deadline_at: Timestamp
    erase_epoch: NonNegativeInt
    consent_epoch: NonNegativeInt
    unit_no: NonNegativeInt
    refusal: ExtractionRefusalDetailV1

    @model_validator(mode="after")
    def bounded_deadline(self) -> Self:
        if not 0 < self.deadline_at - self.admitted_at <= DEFAULT_COMPLETION_WORK_BUDGET_MS:
            raise ValueError("Refused finalize allowance must retain its bounded deadline")
        return self


class CompletedPrefixV2(StrictModel):
    """Projection written with actual release/settlement, never inferred by a waiter."""

    contract_version: Literal[2] = 2
    accepted_prefix_commit_seq: NonNegativeInt
    lease: ReleasedFinalizeLeaseV1
    marker: FinalizedValueV1
    owner_digest: Integrity
    input_ref: Annotated[
        ArtifactId, RefSpec(
            "artifact", artifact_kinds=("run_input",),
            ref_mode="historical_locator", fulfillment="none",
        ),
    ]
    result_ref: Annotated[
        ArtifactId, RefSpec(
            "artifact", artifact_kinds=("result_index", "manifest"),
            ref_mode="historical_locator", fulfillment="none",
        ),
    ]
    input_digest: Integrity
    result_digest: Integrity
    erase_epoch: NonNegativeInt
    consent_epoch: NonNegativeInt

    @model_validator(mode="after")
    def one_released_owner(self) -> Self:
        if (self.lease.owner_run_id, self.lease.lease_generation) != (
            self.marker.run_id, self.marker.lease_generation,
        ):
            raise ValueError("Completed prefix must bind one released owner")
        if self.accepted_prefix_commit_seq > self.lease.baseline_snapshot_generation:
            raise ValueError("Completed prefix exceeds its actual frozen read baseline")
        return self


class AddCompletionV1(StrictModel):
    """One owner's completion bookkeeping, written after publication and before release.

    Its request results and checked sources share this row's bounded transaction
    and stay invisible until the owner succeeds. Release validates this binding.
    """

    owner_run_id: HistoricalOwner
    result_ref: Annotated[
        ArtifactId, RefSpec(
            "artifact", artifact_kinds=("result_index", "manifest"),
            ref_mode="historical_locator", fulfillment="none",
        ),
    ]
    recovery_count: Annotated[int, Field(ge=0, le=MAX_OWNER_RESUMES)]
    request_count: NonNegativeInt
    source_count: NonNegativeInt


class EmbeddingProviderBindingV1(StrictModel):
    """Freeze the public embedding choice without storing credentials."""

    provider: Annotated[str, Field(min_length=1, max_length=128)]
    requested_model: Annotated[str, Field(min_length=1, max_length=256)]
    dimension: Literal[1024, 2048]


class JobProviderBindingsV1(StrictModel):
    """Freeze the bench provider choices at its external job boundary."""

    embedding: EmbeddingProviderBindingV1
    extract: InferenceBindingV1
    judge: InferenceBindingV1

    @model_validator(mode="after")
    def validate_purposes(self) -> Self:
        """Reject a binding filed under a different inference purpose."""
        if self.extract.purpose != "extract" or self.judge.purpose != "judge":
            raise ValueError("Job bindings must match their inference purposes")
        return self


class JobValueV1(StrictModel):
    """Track bench provider choices and the monotonic traffic watermark."""

    provider_bindings: JobProviderBindingsV1
    last_platform_traffic_at: Timestamp


ControlValue = (
    WorkspaceBindingV1
    | CounterV1
    | PolicyPointerV1
    | ActiveEmbeddingSpaceV1
    | WriterLeaseV1
    | ScopeRegistrationV1
    | ScopeQuarantineV1
    | RequestLedgerValueV1
    | PreparingFinalizeLeaseV1
    | HeldFinalizeLeaseV1
    | ReleasedFinalizeLeaseV1
    | CancelledFinalizeLeaseV1
    | FinalizedValueV1
    | CompletedAddOutcomeV1
    | RequestCompletionPolicyV2
    | FinalizeBudgetV2
    | FinalizeBudgetRefusalV4
    | SourceLabelsBudgetV1
    | ExtractionOwnerBudgetV1
    | SourceLabelMessageReservationV1
    | CompletedPrefixV2
    | AddCompletionV1
    | JobValueV1
    | CheckedSourceV1
    | JudgeOutputLocatorV1
    | EvidenceHoldV1
    | ForgetMarksV1
    | ForgetWithdrawnV1
    | ForgetShadowV1
    | ForgetHandledV1
    | ForgetAddV1
    | ForgetKeptV1
    | PrivacyEraseProgressV1
    | ForgetIntentV1
    | ErasureFenceV1
)
type PolicyTier = Literal["fast"]
"""A per-user Add tier with its own released extraction policy (``ProviderConfig.add_tiers``);
the default tier's policy has no tier."""

FAST_EXTRACTION_POLICY_KEY: Final = "policy:extraction_fast"
"""The pointer to the fast tier's released extraction policy, beside ``policy:extraction``."""

POLICY_POINTERS: Final[dict[str, tuple[Literal["recall", "extraction"], PolicyTier | None]]] = {
    "policy:recall": ("recall", None),
    "policy:extraction": ("extraction", None),
    FAST_EXTRACTION_POLICY_KEY: ("extraction", "fast"),
}
"""Each policy pointer key's released policy kind and tier."""


def policy_pointer_key(kind: Literal["recall", "extraction"], tier: PolicyTier | None) -> str:
    """The pointer key of a released policy kind and tier (``POLICY_POINTERS``).

    Raises:
        QuoteMemError: unsupported_variant for a tier of the recall policy.
    """
    for key, selected in POLICY_POINTERS.items():
        if selected == (kind, tier):
            return key
    raise QuoteMemError("unsupported_variant", "Only the extraction policy has a fast tier")


CONTROL_MODELS: dict[str, type[StrictModel]] = {
    "binding:workspace": WorkspaceBindingV1,
    "counter": CounterV1,
    "policy:recall": PolicyPointerV1,
    "policy:extraction": PolicyPointerV1,
    FAST_EXTRACTION_POLICY_KEY: PolicyPointerV1,
    "policy:embedding_space:active": ActiveEmbeddingSpaceV1,
    "lease:writer": WriterLeaseV1,
    "scope": ScopeRegistrationV1,
    "scope_quarantine": ScopeQuarantineV1,
    "request_ledger": RequestLedgerValueV1,
    "lease:finalize:preparing": PreparingFinalizeLeaseV1,
    "lease:finalize:held": HeldFinalizeLeaseV1,
    "lease:finalize:released": ReleasedFinalizeLeaseV1,
    "lease:finalize:cancelled": CancelledFinalizeLeaseV1,
    "finalized": FinalizedValueV1,
    "add_outcome": CompletedAddOutcomeV1,
    "request_completion": RequestCompletionPolicyV2,
    "finalize_budget": FinalizeBudgetV2,
    "finalize_budget:v4": FinalizeBudgetRefusalV4,
    "source_labels_budget": SourceLabelsBudgetV1,
    "extraction_budget": ExtractionOwnerBudgetV1,
    "source_labels_message": SourceLabelMessageReservationV1,
    "completed_prefix": CompletedPrefixV2,
    "add_completion": AddCompletionV1,
    "job": JobValueV1,
    "checked_source": CheckedSourceV1,
    "judge_output": JudgeOutputLocatorV1,
    "maintenance:evidence_invalidation:active": PrivacyEraseProgressV1,
    # The forget intent and the erasure fence, with their producer.
    "intent:forget:active": ForgetIntentV1,
    "intent:forget:consumed": ForgetIntentV1,
    "intent:forget:expired": ForgetIntentV1,
    "erasure": ErasureFenceV1,
    "evidence_hold": EvidenceHoldV1,
    "forget_marks": ForgetMarksV1,
    "forget_withdrawn": ForgetWithdrawnV1,
    "forget_shadow": ForgetShadowV1,
    "forget_handled": ForgetHandledV1,
    "forget_add": ForgetAddV1,
    "forget_kept": ForgetKeptV1,
}


class ControlRowV1(StrictModel):
    """Materialize the complete physical Control row before typed CAS."""

    object_id: Annotated[ControlId, DerivedSpec("key"), OpaqueIdSpec("local")]
    key: LocalId
    control_refs: Annotated[list[ReferenceProjectionV1], DerivedSpec("value")]
    value: ControlValue
    version: NonNegativeInt
    updated_at: Timestamp
    updated_by_run_id: (
        Annotated[
            ULID,
            RefSpec(
                "run",
                ref_mode="snapshot_locator",
                scope_rule="protocol",
                fulfillment="pre_state",
                target_forget_action="retain_locator",
            ),
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_row(self) -> Self:
        """Check key materialization and the externally discriminated value."""
        if self.object_id != control_id(self.key):
            raise ValueError("Control object identity does not match its namespace key")
        expected = CONTROL_MODELS.get(control_variant(self.key, self.value))
        if expected is None or type(self.value) is not expected:
            raise ValueError("Control value does not match its namespace")
        if isinstance(self.value, SourceLabelsBudgetV1):
            if self.key.split(":")[2] != self.value.child_run_id:
                raise ValueError("Source-label budget key and child owner differ")
        elif isinstance(self.value, ExtractionOwnerBudgetV1):
            if self.key.split(":")[2] != self.value.source_run_id:
                raise ValueError("Extraction budget key and original owner differ")
        elif isinstance(self.value, CompletedAddOutcomeV1):
            if self.key.split(":")[2] != self.value.request_key or (
                self.updated_by_run_id != self.value.completed_by_run_id
            ):
                raise ValueError("Add outcome key or completion owner differs")
        elif isinstance(self.value, AddCompletionV1):
            if self.updated_by_run_id != self.value.owner_run_id:
                raise ValueError("Add completion bookkeeping changed its owner")
        elif isinstance(self.value, SourceLabelMessageReservationV1):
            if self.updated_by_run_id != self.value.owner.child_run_id or (
                (self.version == 0) != (self.value.owner == self.value.first_origin)
            ):
                raise ValueError("Source-label reservation changed its original or current owner")
        return self


# Every namespace whose value is a monotonic `CounterV1`. `tombstone_seq:{s}`
# advances once per forget admission and per tombstone-deleting transaction.
COUNTER_NAMESPACES = frozenset({
    "commit_seq", "scope_seq", "tombstone_seq", "erase_epoch", "consent_epoch",
})

_SCOPE_KEY_ADAPTER: TypeAdapter[str] = TypeAdapter(ScopeId)
_SESSION_KEY_ADAPTER: TypeAdapter[str] = TypeAdapter(SessionHash)
_EVIDENCE_KEY_ADAPTER: TypeAdapter[str] = TypeAdapter(SessionHash)
_EVIDENCE_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(ULID)
_CANONICAL_UNIT_NO: Final = re.compile(r"0|[1-9][0-9]{0,8}")
"""A unit number in an extraction call key: a canonical non-negative decimal."""
_BINDING_KEY_ADAPTER: TypeAdapter[str] = TypeAdapter(ContentBindingDigest)
_PRINCIPAL_KEY_ADAPTER: TypeAdapter[str] = TypeAdapter(PrincipalId)


def maintenance_fence_key(
    scope: str,
    evidence_id: str,
    bind_content: Callable[[str, object], ContentBindingDigest],
) -> str:
    """Use the tenant-keyed flat Evidence anchor.

    The reserved evidence-invalidation HMAC purpose binds the exact flat
    envelope. Writer checks this key against the typed value's evidence_id;
    schema validation alone cannot access a tenant key or prove that parity.
    """
    scope = _SCOPE_KEY_ADAPTER.validate_python(scope)
    evidence_id = _EVIDENCE_ID_ADAPTER.validate_python(evidence_id)
    digest = bind_content("evidence-invalidation", {"evidence_id": evidence_id})
    return f"maintenance:{scope}:evidence_invalidation:{digest[:22]}"


def _control_key_parts(key: object) -> tuple[str, str | None]:
    """Parse the key grammar once without normalizing opaque protocol bytes."""
    if not isinstance(key, str) or not key or len(key) > 1024:
        raise QuoteMemError("schema_invalid", "Control key length is invalid")
    if key in {
        "binding:workspace",
        "lease:writer",
        "policy:recall",
        "policy:extraction",
        FAST_EXTRACTION_POLICY_KEY,
        "policy:embedding_space",
        "commit_seq",
    }:
        return key, None
    if key == "erasure" or key.startswith("erasure:"):
        parts = key.split(":")
        if len(parts) != 3:
            raise QuoteMemError("schema_invalid", "Erasure key requires one scope and controller")
        try:
            scope = _SCOPE_KEY_ADAPTER.validate_python(parts[1])
            _EVIDENCE_ID_ADAPTER.validate_python(parts[2])
        except ValidationError as error:
            raise QuoteMemError("schema_invalid", "Erasure key has invalid coordinates") from error
        return "erasure", scope
    if key == "intent" or key.startswith("intent:"):
        parts = key.split(":")
        if len(parts) != 6:
            raise QuoteMemError("schema_invalid", "Intent key requires its complete exact binding")
        if parts[3] != "forget":
            raise QuoteMemError("unsupported_variant", "This intent consumer is not implemented")
        try:
            scope = _SCOPE_KEY_ADAPTER.validate_python(parts[1])
            _PRINCIPAL_KEY_ADAPTER.validate_python(parts[2])
            _BINDING_KEY_ADAPTER.validate_python(parts[4])
            _EVIDENCE_ID_ADAPTER.validate_python(parts[5])
        except ValidationError as error:
            raise QuoteMemError("schema_invalid", "Intent key has invalid coordinates") from error
        return "intent:forget", scope
    if key == "maintenance" or key.startswith("maintenance:"):
        parts = key.split(":")
        if len(parts) != 4:
            raise QuoteMemError("schema_invalid", "Maintenance key requires exact scope and anchor")
        if parts[2] != "evidence_invalidation":
            raise QuoteMemError("unsupported_variant", "This maintenance reason is not implemented")
        try:
            scope = _SCOPE_KEY_ADAPTER.validate_python(parts[1])
            _EVIDENCE_KEY_ADAPTER.validate_python(parts[3])
        except ValidationError as error:
            raise QuoteMemError(
                "schema_invalid", "Maintenance scope or anchor is not canonical"
            ) from error
        return "maintenance:evidence_invalidation", scope
    if any(key == prefix or key.startswith(prefix + ":")
           for prefix in ("source_labels_message", "add_outcome", "judge_output")):
        parts = key.split(":")
        if len(parts) != 3:
            raise QuoteMemError("schema_invalid",
                                "Source-label message key lost its scope or digest")
        try:
            scope = _SCOPE_KEY_ADAPTER.validate_python(parts[1])
            _BINDING_KEY_ADAPTER.validate_python(parts[2])
        except ValidationError as error:
            raise QuoteMemError("schema_invalid",
                                "Source-label message key is not canonical") from error
        return parts[0], scope
    if any(key == prefix or key.startswith(prefix + ":") for prefix in (
        "source_labels_budget", "extraction_budget", "checked_source",
    )):
        prefix = key.split(":", 1)[0]
        suffix = key.removeprefix(prefix + ":").split(":")
        # An extraction call has no child Run: its allowance is keyed by the
        # accounting unit, extraction_budget:{scope}:{owner_run_id}:{unit_no}:{call_no};
        # a judge child keeps extraction_budget:{scope}:{child}.
        unit_call = prefix == "extraction_budget" and len(suffix) == 4
        if len(suffix) != 2 and not unit_call:
            raise QuoteMemError("schema_invalid", "Child Control key lost its scope or child")
        try:
            scope = _SCOPE_KEY_ADAPTER.validate_python(suffix[0])
            _EVIDENCE_ID_ADAPTER.validate_python(suffix[1])
        except ValidationError as error:
            raise QuoteMemError("schema_invalid", "Child Control key is noncanonical") from error
        if unit_call and (
            _CANONICAL_UNIT_NO.fullmatch(suffix[2]) is None or suffix[3] not in {"0", "1"}
        ):
            raise QuoteMemError("schema_invalid",
                                "Extraction call key has a noncanonical unit or call")
        return prefix, scope
    for prefix, has_session in (
        ("scope_seq", False),
        ("tombstone_seq", False),
        ("erase_epoch", False),
        ("consent_epoch", False),
        ("lease:finalize", False),
        ("scope_quarantine", False),
        ("scope", False),
        ("request_ledger", True),
        ("finalized", False),
        ("request_completion", False),
        ("finalize_budget", False),
        ("completed_prefix", False),
        ("add_completion", False),
        ("evidence_hold", False),
        ("forget_marks", False),
        ("forget_withdrawn", False),
        ("forget_shadow", False),
        ("forget_handled", False),
        ("forget_add", False),
        ("forget_kept", False),
    ):
        if key != prefix and not key.startswith(prefix + ":"):
            continue
        suffix = key[len(prefix) + 1 :].split(":")
        if len(suffix) != (2 if has_session else 1):
            raise QuoteMemError("schema_invalid", "Control key has invalid namespace coordinates")
        try:
            scope = _SCOPE_KEY_ADAPTER.validate_python(suffix[0])
            if has_session:
                _SESSION_KEY_ADAPTER.validate_python(suffix[1])
        except ValidationError as error:
            raise QuoteMemError(
                "schema_invalid", "Control key has a noncanonical scope or session hash"
            ) from error
        return prefix, scope
    if key == "job" or key.startswith("job:"):
        if not key.removeprefix("job:") or key == "job":
            raise QuoteMemError("schema_invalid", "Job Control key requires an opaque job identity")
        return "job", None
    raise QuoteMemError("unsupported_variant", "Control namespace is not implemented")


def control_key_occupies_scope(key: str, scope_id: str) -> bool:
    """Report whether one Control key carries the exact canonical scope coordinate.

    Every scoped namespace, opened or not, places the canonical
    scope identity as one complete colon-delimited segment after its namespace
    prefix, and no other segment of a scoped key is a canonical scope identity.
    This lexical test therefore needs no namespace grammar: an unopened
    namespace keeps its typed unsupported value while its key still occupies
    the scope, and a neighbor scope's row never raises. An opaque job identity
    never supplies scope authority.
    """
    if not key or len(key) > 1024:
        raise QuoteMemError("schema_invalid", "Control key length is invalid")
    try:
        canonical = _SCOPE_KEY_ADAPTER.validate_python(scope_id)
    except ValidationError as error:
        raise QuoteMemError("schema_invalid",
                            "Scope occupancy requires a canonical scope") from error
    if key == "job" or key.startswith("job:"):
        return False
    return canonical in key.split(":")[1:]


def control_scope_id(key: str) -> str | None:
    """Return the authoritative scope coordinate, or None for a global Control.

    An opaque job identity can contain punctuation or scope-looking text. It
    never supplies scope authority; scoped namespaces use the scoped key grammar.
    """
    return _control_key_parts(key)[1]


def control_variant(key: str, raw: object) -> str:
    """Resolve the validated complete Control namespace and its strict state variant."""
    namespace, _ = _control_key_parts(key)
    state = (
        cast(dict[str, object], raw).get("state")
        if isinstance(raw, dict)
        else getattr(raw, "state", None)
    )
    if namespace in COUNTER_NAMESPACES:
        return "counter"
    if namespace == "finalize_budget":
        version = cast(dict[str, object], raw).get("contract_version") if isinstance(
            raw, dict,
        ) else getattr(
            raw, "contract_version", None,
        )
        return "finalize_budget:v4" if type(version) is int and version == 4 else namespace
    if namespace in {
        "policy:embedding_space",
        "lease:finalize",
        "maintenance:evidence_invalidation",
        "intent:forget",
    }:
        return f"{namespace}:{state}"
    return namespace


def parse_control_value(key: str, raw: object) -> ControlValue:
    """Parse a Control value using its key as the authoritative discriminator.

    Args:
        key: The full namespace key, including any scope/session suffix.
        raw: A decoded JSON value or an already typed value.

    Returns:
        The implemented strict Control value for this namespace and state.

    Raises:
        QuoteMemError: For unsupported variants or schema violations.
    """
    variant = control_variant(key, raw)
    model = CONTROL_MODELS.get(variant)
    if model is None:
        raise QuoteMemError("unsupported_variant", f"Control variant is not implemented: {variant}")
    try:
        value = model.model_validate(raw)
    except ValidationError as error:
        raise QuoteMemError("schema_invalid",
                            "Control value violates its namespace schema") from error
    if not isinstance(value, ControlValue):
        raise QuoteMemError("schema_invalid", "Control dispatch returned an invalid model")
    return value

