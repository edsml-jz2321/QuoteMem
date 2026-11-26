"""The durable forget Controls: the bound intent and the erasure fence plan.

``intent:{scope}:{principal}:forget:{binding}:{ulid}`` holds one prepared
selection; the atomic admission consumes it in the transaction that
installs ``erasure:{scope}:{controller}``. The fence carries the frozen plan
inputs: the selected memory, the stating spans S, the tombstones of
D, the post-bump epochs and the phase a restart resumes from. Parsing a value
authorizes nothing; the Writer-side owner (`store/forget_run.py`) rechecks
every row each batch reads.

A forget of sentences selects no memory: a fact the user stated that no
memory states is forgotten by its sentences alone. Its intent and fence name no memory,
S is exactly the selected sentences, and its one tombstone is the ``ForgetSeedV1`` the
selection froze (the user's words of the fact and their vector), so a later restatement
is checked against it as against any forgotten statement.

A complete forget of a memory that bundles the fact asked with an unrelated one carries a seed too:
the user's words of the fact. Its tombstone stands for the head's statement, so a later Add may
teach the unrelated fact again while a restatement of the forgotten one is still checked against the
user's words; the memory's earlier statements keep their own tombstones.

An active intent that expired (``ForgetOwner.expire_intents``, or the first admission that
finds it expired) keeps no seed: it becomes ``expired``, keeping only its
``selection_digest``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self, cast

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from quotemem.core.entities import EntityRef, QuoteSpanV1
from quotemem.core.forget import MAX_FORGET_SOURCES, MAX_FORGET_TOMBSTONES, ForgetCountsV1
from quotemem.core.forget_reword import MAX_REWORDINGS
from quotemem.core.identity import ULID, ArtifactId, PrincipalId, ScopeId
from quotemem.core.semantic import EmbeddingVector
from quotemem.core.statement import (
    MAX_KEYWORD_CHARS,
    MAX_KEYWORDS,
    MAX_MEMORY_SPANS,
    MAX_STATEMENT_CHARS,
    MAX_SUPPORT_GROUPS,
    HexDigest,
    canonical_statement,
)
from quotemem.core.technical_types import ContentDigest, NonNegativeInt, Timestamp
from quotemem.store.metadata import OpaqueIdSpec, StrictModel


def _optional_array(value: object) -> object:
    """``_array``, None kept."""
    return None if value is None else _array(value)


def _array(value: object) -> object:
    """Accept JSON arrays without relaxing strict element validation."""
    return tuple(cast(list[object], value)) if isinstance(value, list) else value

MAX_FORGET_SPANS = 4_096
"""The forgotten span set S a fence may carry; a larger S is ``budget_exceeded`` before writes."""
MAX_FORGET_ENTITIES = 512
"""Entities whose anchoring the forgotten memory may have held (erasure candidates)."""
MAX_FORGET_ARTIFACTS = 16_384
"""Retention-unit Artifacts (with their manifest parts) one forget deletes."""
MAX_FORGET_CONTROLS = 4_096
"""Retention-unit Controls (``judge_output`` locators) one forget deletes."""
MAX_FORGET_RECALLS = 16_384
"""Search deliveries (``recall`` rows) one forget deletes."""
MAX_FORGET_CHAIN = 64
MAX_FORGET_KEPT_LISTED = 512
"""The most memories a forget's intent records its prepare named (``ForgetIntentV1.listed``)."""
"""Memories the selected one ended by ``correct`` or ``retract``, transitively, that one forget
takes with it; a longer chain is ``budget_exceeded`` before writes."""
MAX_RESTATING_SPANS = 256
"""Restating sentences one intent adds to its S (`ForgetIntentV1.restating_spans`)."""
INTENT_TTL_MS = 10 * 60 * 1000
"""A prepared forget intent's lifetime; confirmation after it is ``intent_expired``."""

_Local = Annotated[ULID, OpaqueIdSpec("local")]


class ForgetSeedV1(StrictModel):
    """The tombstone a forget of sentences writes: what the user asked to
    forget, in the user's words, and its vector in the workspace's embedding space.

    No version states the fact, so the seed stands for one: its tombstone is the user's
    primary entity's, a state without time. Like every tombstone it is never readable.
    """

    model_config = ConfigDict(frozen=True)
    statement: Annotated[str, Field(min_length=1, max_length=MAX_STATEMENT_CHARS)]
    embedding: EmbeddingVector
    embedding_space: Annotated[str, Field(min_length=1, max_length=1024)]
    entity_id: Annotated[
        EntityRef, OpaqueIdSpec("local"), Field(exclude_if=lambda value: value == "user"),
    ] = "user"
    """The tombstone's primary entity: the user's for a forget of sentences, the bundling
    head's for a bundled memory; left out when the user's."""

    @model_validator(mode="after")
    def canonical(self) -> Self:
        if canonical_statement(self.statement) != self.statement:
            raise ValueError("A forget seed keeps a canonical statement")
        return self


_Support = Annotated[
    tuple[Annotated[tuple[QuoteSpanV1, ...], Field(min_length=1, max_length=MAX_MEMORY_SPANS),
                    BeforeValidator(_array)], ...],
    Field(max_length=MAX_SUPPORT_GROUPS), BeforeValidator(_array),
]


class ForgetRewordV1(StrictModel):
    """A complete forget's planned rewording of a memory that tells the forgotten detail beside
    other facts (``core/forget_reword.py``): the statement it reads once the detail is gone,
    that statement's vector in the workspace's embedding space, the support it rests on and
    the keywords it keeps.

    The prepare plans it against everything the confirmation withholds (``ForgetOwner.
    admit_rewordings``): ``support`` is the groups of the memory's quotes, whole or in readable
    parts, that none of the plan's forgets withholds and that carry the new statement;
    ``keywords`` are the memory's keywords showing nothing forgotten. The first forget of the
    plan tombstones the present wording (``base_digest``, the planned head's
    ``statement_digest``) and writes the new statement as the memory's next head on that
    support, once it checks it again (``ForgetOwner._admit_wording``); otherwise the memory
    is left as the plan's withholding leaves it. ``tie`` is the tie the statement names the
    forgotten name's bearer by ("the user's wife"), the only words it may share with what is
    forgotten. The statement shows nothing forgotten, so it may be kept."""

    model_config = ConfigDict(frozen=True)
    memory_id: _Local
    base_digest: Annotated[HexDigest, OpaqueIdSpec("digest")]
    statement: Annotated[str, Field(min_length=1, max_length=MAX_STATEMENT_CHARS)]
    embedding: EmbeddingVector
    embedding_space: Annotated[str, Field(min_length=1, max_length=1024)]
    support: _Support = ()
    keywords: Annotated[tuple[Annotated[str, Field(min_length=1, max_length=MAX_KEYWORD_CHARS)],
                              ...], Field(max_length=MAX_KEYWORDS), BeforeValidator(_array)] = ()
    tie: Annotated[str | None, Field(max_length=256, exclude_if=lambda value: value is None)] = None

    @model_validator(mode="after")
    def canonical(self) -> Self:
        if canonical_statement(self.statement) != self.statement:
            raise ValueError("A rewording keeps a canonical statement")
        return self


_Rewordings = Annotated[
    tuple[ForgetRewordV1, ...],
    Field(max_length=MAX_REWORDINGS, exclude_if=lambda value: not value),
    BeforeValidator(_array),
]


class ForgetIntentV1(StrictModel):
    """One prepared, bounded forget selection.

    ``active`` until the atomic admission consumes it; the consumed value
    names its controller so a repeated confirm returns the original admission.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    state: Literal["active", "consumed", "expired"]
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    principal_id: Annotated[PrincipalId, OpaqueIdSpec("local")]
    session_token: ContentDigest
    memory_id: _Local | None
    """The selected memory; None for a forget of sentences."""
    expected_revision_id: _Local | None
    selection_digest: ContentDigest
    erase_epoch: NonNegativeInt
    issued_at: Timestamp
    expires_at: Timestamp
    controller_run_id: _Local | None = None
    restating_spans: Annotated[
        tuple[QuoteSpanV1, ...],
        Field(max_length=MAX_RESTATING_SPANS, exclude_if=lambda value: not value),
        BeforeValidator(_array),
    ] = ()
    """Sentences that restate what is forgotten without being cited by the memory's versions
    (``ForgetOwner.prepare``'s ``restating``): admission adds them to S, so the forget
    withholds them with the rest. Empty for every forget but a complete forget's, and then
    left out of the stored value. A forget of sentences selects exactly these."""
    seed: Annotated[ForgetSeedV1 | None, Field(exclude_if=lambda value: value is None)] = None
    """A forget of sentences' tombstone, or the user's words of the fact a
    bundled memory is forgotten for; None, and left out, otherwise. A consumed
    intent drops it: the fence carries it on, the intent keeps only the
    ``selection_digest`` that binds it. An expired one drops it too."""

    enforces: Annotated[Timestamp | None, Field(exclude_if=lambda value: value is None)] = None
    """When the forget this one enforces was admitted (the ``agent`` mode): the
    relearn check withholding a restatement of an already forgotten fact (an assistant echo)
    forgets nothing new, so the forget shadow keeps the stamps of the terms it holds and
    stamps a term it adds with this time, not with now; the user's own restatement after
    that forget still lifts them. None, and left out, for every other forget."""
    allowed: Annotated[tuple[_Local, ...] | None, Field(
        max_length=MAX_FORGET_KEPT_LISTED, exclude_if=lambda value: value is None),
        BeforeValidator(_optional_array)] = None
    """What a complete forget may take besides the base links: the memories the forget
    judge confirmed state or presuppose the forgotten fact (``forget_pick.tells``), among
    those a word rule nominated; None, and left out, for a forget that is not complete, whose
    chain is the base one."""
    listed: Annotated[tuple[_Local, ...] | None, Field(
        max_length=MAX_FORGET_KEPT_LISTED, exclude_if=lambda value: value is None),
        BeforeValidator(_optional_array)] = None
    """What the prepare named it takes or changes: an admission whose chain takes a memory
    outside it refuses (``operation_conflict``); None, and left out, when unbounded."""
    support_only: Annotated[bool, Field(exclude_if=lambda value: not value)] = False
    """A forget of a memory only for its support, which other forgets withheld (a complete
    forget's cascade): it takes its base links and no replacer, since its own value is
    not forgotten; left out when False."""
    modifier_only: Annotated[bool, Field(exclude_if=lambda value: not value)] = False
    """A forget the user asked for a modifier alone of a thing of theirs ("forget that my
    cat is orange",
    ``forget_admission.asks_modifier``): a survivor about that thing keeps what does not show
    the modifier (``ForgottenCitation.modifier``), at admission as at the deletion stage; left
    out when False."""
    rewordings: _Rewordings = ()
    """The memories the forget rewords instead of taking (``ForgetRewordV1``): a complete
    forget's; the fence carries them on and a consumed intent drops them. Left out when
    empty."""
    asked: Annotated[tuple[ContentDigest, ...], Field(
        max_length=MAX_KEYWORDS, exclude_if=lambda value: not value),
        BeforeValidator(_array)] = ()
    """The keyed digests (``SHADOW_PURPOSE``) of the shadow names of the values the user asked
    to forget, when the request shows a value of a statement naming more than one: the forget
    shadow takes only these of such a statement's names, never the value the user kept
    ("Hangzhou" of "The user moved from Hangzhou to Shanghai." for "forget that I lived in
    Shanghai"). A consumed intent drops them; left out when empty."""
    name_asked: Annotated[bool, Field(exclude_if=lambda value: not value)] = False
    """Whether the user asked to forget a name (``forget_match.asks_name``): of what the
    forget withholds, a rewording then reads only the name's words as forgotten
    (``ForgetOwner._wording_words``). Left out when False."""

    @model_validator(mode="after")
    def consumed_names_controller(self) -> Self:
        if (self.state == "consumed") != (self.controller_run_id is not None):
            raise ValueError("Only a consumed intent names its controller")
        if (self.memory_id is None) != (self.expected_revision_id is None) or (
                self.memory_id is None and not self.restating_spans) or (
                self.memory_id is None and self.state == "active" and self.seed is None):
            raise ValueError("An intent selects an Item at its head, or sentences with a seed")
        if self.state == "expired" and self.seed is not None:
            raise ValueError("An expired intent keeps no seed")
        if self.expires_at <= self.issued_at:
            raise ValueError("A forget intent must expire after it is issued")
        return self


ErasurePhase = Literal["admitting", "admitted", "deleted", "purged", "scanned", "receipted"]
"""Fence installed (step a); admission effects committed; memory deletion committed;
retention records deleted; final scan bound; receipts written, which releases the fence."""


class ErasureFenceV1(StrictModel):
    """The scope's single erasure fence and its frozen forget plan.

    Attributes:
        phase: Where a restart resumes (`ForgetOwner.resume`).
        controller_run_id / worker_run_id: The global controller and the
            scoped destructive worker admitted.
        memory_id: The selected memory; the final scan proves it and its retention rows gone. None
            for a forget of sentences, whose S is the selected sentences and whose
            one tombstone is ``seed``'s.
        chain_ids: The memories the selected one ended by ``correct`` or ``retract``,
            transitively, in id order: a correction or retraction that changed
            the primary anchor created a new memory, so these are the selected
            memory's earlier versions. D, S and the final scan cover them like the selected memory.
        reworded_ids: The surviving memories whose head a forgotten correction reworded,
            in id order: outside the chain, the versions they had before that
            rewording join D, and their head's own groups stay readable.
        intent_key: The consumed intent.
        spans: S, the stating spans: every span of every support group of every
            non-erased version, less the phrase a surviving readable naming memory
            of that version's primary entity quotes as its own support and less a
            bare name a neutral-type-only naming memory of a non-user entity quotes,
            and less the quote of a detail keeper (an owner- or user-anchored naming
            memory of an entity the version states a different detail of),
            unless that would leave a group with nothing withheld. Sensitive naming
            heads keep nothing on any path; a bare name or a detail keeper keeps
            nothing when any forgotten version has a sensitivity class
            (``plan_stating_spans``).
        tombstone_ids: One tombstone per element of D (reused or created).
        created_tombstone_ids: The subset admission created.
        entity_ids: Entities the memory's versions anchored (erasure candidates).
        erase_epoch / tombstone_seq / scope_seq / commit_seq: The values after admission.
        counts: Accumulated effects; the final scan recounts what it can prove.
        spans_done: How many spans of S the batches withheld.
        dependents_after: The last memory, in id order, whose dependent
            compensation committed; the next batch starts after it.
        expired_after: The last pending candidate, in id order, whose expiry
            committed; the next expiry batch starts after it.
        cancelled_owner_run_id: The live Add owner whose finalize lease admission
            cancelled, or None. Deletion settles its cohort before the
            memory's own rows leave.
        trim_dependents: Whether deletion may keep a dependent readable on the readable
            quotes of one of its groups (``trimmed_support``): admission sets it
            when no forgotten version has a sensitivity class. The deletion step clears it if a
            dependents batch was refused (``ForgetOwner._settle_untrimmed``).
        artifacts / controls: The deleted positions' retention records,
            frozen with the deletion of their rows.
        recalls: The Search deliveries that returned a deleted memory or an
            Evidence row of S, frozen with the same deletion; the
            fence bars every delivery of the scope from admission on.
            ``purged`` counts the Artifacts, then the Controls, then these
            deliveries the batches deleted.
        scan_digest / prepare_ref / final_ref / index_ref: Bound by the scan and release;
            ``index_ref`` is the controller's ``result_ref``.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    state: Literal["active"] = "active"
    phase: ErasurePhase
    scope_id: Annotated[ScopeId, OpaqueIdSpec("local")]
    controller_run_id: _Local
    worker_run_id: _Local
    memory_id: _Local | None
    chain_ids: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_CHAIN), BeforeValidator(_array),
    ] = ()
    reworded_ids: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_CHAIN), BeforeValidator(_array),
    ] = ()
    intent_key: Annotated[str, Field(min_length=1, max_length=512)]
    spans: Annotated[
        tuple[QuoteSpanV1, ...], Field(max_length=MAX_FORGET_SPANS), BeforeValidator(_array),
    ]
    tombstone_ids: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_TOMBSTONES), BeforeValidator(_array),
    ]
    created_tombstone_ids: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_TOMBSTONES), BeforeValidator(_array),
    ]
    entity_ids: Annotated[
        tuple[Annotated[EntityRef, OpaqueIdSpec("local")], ...],
        Field(max_length=MAX_FORGET_ENTITIES), BeforeValidator(_array),
    ]
    erase_epoch: NonNegativeInt
    tombstone_seq: NonNegativeInt
    scope_seq: NonNegativeInt
    commit_seq: NonNegativeInt
    counts: ForgetCountsV1
    spans_done: NonNegativeInt = 0
    dependents_after: _Local | None = None
    expired_after: _Local | None = None
    cancelled_owner_run_id: _Local | None = None
    trim_dependents: bool = False
    artifacts: Annotated[
        tuple[Annotated[ArtifactId, OpaqueIdSpec("local")], ...],
        Field(max_length=MAX_FORGET_ARTIFACTS), BeforeValidator(_array),
    ] = ()
    controls: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=512)], ...],
        Field(max_length=MAX_FORGET_CONTROLS), BeforeValidator(_array),
    ] = ()
    recalls: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_RECALLS), BeforeValidator(_array),
    ] = ()
    purged: NonNegativeInt = 0
    scan_digest: ContentDigest | None = None
    prepare_ref: Annotated[str, OpaqueIdSpec("local")] | None = None
    final_ref: Annotated[str, OpaqueIdSpec("local")] | None = None
    index_ref: Annotated[str, OpaqueIdSpec("local")] | None = None
    seed: Annotated[ForgetSeedV1 | None, Field(exclude_if=lambda value: value is None)] = None
    """A forget of sentences' tombstone (``ForgetIntentV1.seed``), or a bundled memory's:
    its tombstone stands for the head's statement; None, and left out, otherwise."""
    any_script: Annotated[bool, Field(exclude_if=lambda value: not value)] = False
    """Whether the clause keeper cut quotes in any script (``plan_stating_spans``) and the
    trim does (``trimmed_support``): a complete forget's; left out when False."""
    replacer_ids: Annotated[
        tuple[_Local, ...], Field(max_length=MAX_FORGET_CHAIN, exclude_if=lambda value: not value),
        BeforeValidator(_array),
    ] = ()
    """The chain's memories taken only because they replaced a forgotten memory and still name its
    value (a complete forget's ``ForgetOwner._chain_parts``), in id order: D and S cover them,
    but their versions get no tombstone and no forget shadow, since their own value was not
    forgotten (so a dependent's trim is not refused for sharing their words); left out when
    empty."""
    rewordings: _Rewordings = ()
    """The memories the deletion stage rewords instead of taking (``ForgetIntentV1.rewordings``): a
    replacer among them leaves the chain, so its quotes' parts naming the forgotten value are
    withheld and it is settled as a dependent; left out when empty."""
    name_asked: Annotated[bool, Field(exclude_if=lambda value: not value)] = False
    """``ForgetIntentV1.name_asked``, which the rewording reads; left out when False."""
    asked: Annotated[tuple[ContentDigest, ...], Field(
        max_length=MAX_KEYWORDS, exclude_if=lambda value: not value),
        BeforeValidator(_array)] = ()
    """``ForgetIntentV1.asked``, which the rewording reads (``ForgetOwner._kept_names``);
    left out when empty."""

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.memory_id is None and (
                self.seed is None or self.chain_ids or self.reworded_ids or not self.spans
                or not 1 <= len(self.tombstone_ids) <= 1 + len(self.rewordings)):
            # Its tombstones are the seed and the present wordings of what it rewords.
            raise ValueError("A forget of sentences withholds them under its one seed")
        if not set(self.created_tombstone_ids) <= set(self.tombstone_ids):
            raise ValueError("Created tombstones must belong to the plan's tombstone set")
        if list(self.tombstone_ids) != sorted(set(self.tombstone_ids)):
            raise ValueError("Tombstones must be unique and ordered")
        if list(self.chain_ids) != sorted(set(self.chain_ids) - {self.memory_id}):
            raise ValueError("The chain must be unique, ordered and exclude the selected Item")
        if list(self.reworded_ids) != sorted(
                set(self.reworded_ids) - {self.memory_id, *self.chain_ids}):
            raise ValueError("Reworded Items must be unique, ordered and outside the chain")
        if list(self.replacer_ids) != sorted(set(self.replacer_ids) & set(self.chain_ids)):
            raise ValueError("Replacers must be unique, ordered and in the chain")
        if self.spans_done > len(self.spans):
            raise ValueError("Withholding progress exceeds the frozen span set")
        records = len(self.artifacts) + len(self.controls) + len(self.recalls)
        if self.purged > records or (
            self.phase in {"purged", "scanned", "receipted"} and self.purged != records
        ):
            raise ValueError("Retention-record progress disagrees with the frozen records")
        if any(list(frozen) != sorted(set(frozen))
               for frozen in (self.artifacts, self.controls, self.recalls)):
            raise ValueError("Retention records must be unique and ordered")
        evidence = {span.evidence_id for span in self.spans}
        if len(evidence) > MAX_FORGET_SOURCES:
            raise ValueError("The forgotten span set cites too many Evidence rows")
        if self.phase in {"scanned", "receipted"} and self.scan_digest is None:
            raise ValueError("A scanned fence binds its final scan digest")
        if (self.phase == "receipted") != (self.index_ref is not None) or (
            (self.final_ref is None) != (self.index_ref is None)
        ) or (
            self.final_ref is not None and self.prepare_ref is None
        ):
            raise ValueError("Only a receipted fence names its receipt pair")
        return self

    @property
    def evidence_ids(self) -> frozenset[str]:
        """The Evidence rows S cites."""
        return frozenset(span.evidence_id for span in self.spans)
