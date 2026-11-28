"""The forget producer on the single Writer: admission, effects, final scan, receipts.

Phases, each a durable Writer transaction whose progress lives in
the scope's ``erasure:{scope}:{controller}`` fence value (``ErasureFenceV1``):

* ``prepare``: a bounded ``intent:…:forget:…`` Control bound to the exact head.
* ``admit``: (a) the global ``erasure_controller`` Run is written first, alone:
  a Control's ``updated_by_run_id`` must name a Run that already exists. One
  transaction then admits the scoped ``forget`` worker, consumes the intent and
  installs the fence with the frozen plan (the stating span set S, tombstone ids,
  entity candidates); both Controls name the controller as their writer. S is
  every span the forgotten versions cite, less the phrase a surviving readable
  naming memory of a version's primary entity quotes as its own support, less a
  bare name a neutral-type-only naming memory quotes, and less the "my" phrase a
  surviving memory quotes as its own subject (``plan_stating_spans``): the
  forget hides what states the forgotten fact, not the name or the referring phrase
  of an entity it was about. Sensitive keeper heads keep nothing; bare names keep
  nothing when any forgotten version is sensitive. The worker has
  no parent Run (a parent shares its child's exact scope, and the controller is
  global); its admission and the fence bind it to its controller. (b) the
  worker's first NL (natural-language) transaction writes the tombstones (reusing an existing
  ``(entity_id, statement_digest)`` row), advances ``tombstone_seq``/
  ``erase_epoch``/``scope_seq``/``commit_seq`` together and withholds the first
  spans of S (up to ``MAX_TX_ROWS``). The fence installed in (a) bars every reader
  and writer of the scope, so (a)+(b) is atomic to every observer; a crash between
  them leaves phase ``admitting`` and resume redoes (b) from the frozen plan.
* ``delete``: bounded batches inside the fence withhold the rest of S, requote
  every dependent failing ``p_ver`` after the fence's cursor
  (``choose_compensation(mode="forget_dependent")``, never erase; one it leaves unreadable keeps the
  readable parts of a group whose words carry it, its statement showing no word of S or of a
  forgotten statement and its keywords dropping any, ``trimmed_support``; every other readable
  survivor drops, in a keyword-only revision, each keyword carrying a word of what a forgotten
  statement says about its subject that its statement and readable quotes do not show,
  ``forgotten_keywords``), delete the
  ``supersedes``/``contradicts`` links other memories carry to the retention unit's
  memories, and expire the pending candidates left without a usable group
  (``evidence_invalidated``) in id order after their own cursor. Then one
  transaction deletes the unit's own rows (the memory with its revisions, their
  links, its names and candidates, and every memory correcting its history),
  erases the extraction entities left without an anchor (a participant only with
  its last source) and freezes into the fence the retention records of the
  deleted positions and the Search deliveries that returned a deleted memory or an
  Evidence row of S. Later batches delete those Artifacts, Controls and
  deliveries. Every batch rereads its rows. ``admit`` refuses, before any
  write, a unit whose own rows cannot fit that transaction or whose records
  exceed the fence's bounds: its fence could never finish.
* ``scan``: six checks over durable storage and the read view (one of them reruns the
  dependent and entity choices, which must find nothing left to do); the keyed
  scan digest is bound into the fence, then into ``ForgetReceiptV3``.
* ``release``: receipts written and verified, ``privacy_quiescence``, the fence
  removed and both Runs succeed.

A forget of sentences (`ForgetOwner.prepare_spans`) runs the same phases
without a memory: S is the selected sentences less what a surviving memory keeps of them (the
keeper paths of ``plan_stating_spans``, the sentences cited by the seed's statement), D
holds only the present wordings of the memories it rewords (``_wording_versions``), its one
other tombstone is the seed the intent froze (the user's words of the fact and their
vector, the user's state), ``delete`` withholds S, compensates the dependents
citing it, expires the candidates it leaves without a usable
group and deletes the deliveries that returned its Evidence; the scan has no memory to prove gone.

A complete forget (``explicit-request-v2``, the ``agent`` Add mode) differs in two ways:

* its clause keeper cuts a quote in any script at its clauses (``clause_spans``,
  ``any_script``), so a Chinese or mixed sentence stating the forgotten fact and another
  memory's keeps that memory's clause readable; the trim does the same
  (``trimmed_support``), as the fence records (``ErasureFenceV1.any_script``);
* a memory that bundles the fact asked with an unrelated one is forgotten under the user's
  words (``ForgetOwner.prepare``'s ``seed``): that one tombstone stands for
  the head's statement, so a later Add may teach the unrelated fact again; the memory's
  earlier statements and its chain's keep their own tombstones.

An intent that expires unconfirmed keeps only its binding (``expire_intents``, and the
admission that meets it): the seed, the user's words and their vector, is dropped.

``resume`` continues from the fence phase without redoing a committed write;
boot step 13 calls ``recover_forgets``, which first adopts the fence's Runs under
the new Writer epoch.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence, Set
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import Any, Final, Literal, cast

from pydantic import ValidationError

from quotemem.core.artifacts import ManifestV1
from quotemem.core.controls import (
    MAX_FORGET_SHADOW_TERMS,
    MAX_SHADOW_TOPIC_TERMS,
    ActiveEmbeddingSpaceV1,
    CancelledFinalizeLeaseV1,
    ControlRowV1,
    EvidenceHoldV1,
    ForgetShadowV1,
)
from quotemem.core.entities import USER, MemorySnapshotV2, QuoteSpanV1, RevisionReason
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_policy import CHAT_FORGET_COMPLETE
from quotemem.core.failure_reason import failure_reason
from quotemem.core.forget import (
    MAX_FORGET_SOURCES,
    MAX_FORGET_TOMBSTONES,
    ForgetCountsV1,
    ForgetInternalRetentionV1,
    ForgetPrepareReceiptV3,
    ForgetReceiptV3,
    ForgetWithheldSourceV1,
)
from quotemem.core.forget_detail import (
    attribute,
    attribute_reworded,
    detail_question,
    detail_request,
    detail_values,
    detail_words,
    existence_leads,
    quote_ranges,
)
from quotemem.core.forget_match import TIES, asks_name, naming_tie, other_subject
from quotemem.core.forget_protocol import verify_forget_receipt_pair
from quotemem.core.forget_receipts import ErasureReceiptResultIndexV1
from quotemem.core.forget_remainder import FORGET_REASONS, remainder_text, stored_remainder
from quotemem.core.forget_reword import (
    MAX_REWORDINGS,
    code_forms,
    contrasted_values,
    forgotten_values,
    han_forms,
    keeps_predicate,
    kin_word,
    kind_of,
    left_out,
    name_kinds,
    named_part,
    replaced_values,
    reworded_statement,
    shows,
    tie_descriptions,
    value_pattern,
)
from quotemem.core.forget_state import (
    INTENT_TTL_MS,
    MAX_FORGET_ARTIFACTS,
    MAX_FORGET_CHAIN,
    MAX_FORGET_CONTROLS,
    MAX_FORGET_ENTITIES,
    MAX_FORGET_RECALLS,
    MAX_FORGET_SPANS,
    MAX_RESTATING_SPANS,
    ErasureFenceV1,
    ErasurePhase,
    ForgetIntentV1,
    ForgetRewordV1,
    ForgetSeedV1,
)
from quotemem.core.identity import ContentBindingDigest, canonical_json, new_ulid
from quotemem.core.inference import DEFAULT_PROVIDER_TIMEOUT_MS, default_retry_policy
from quotemem.core.judge_records import (
    JudgeCallManifestV1,
    JudgeResultIndexV8,
    JudgmentRunInputV1,
    judgment_targets,
)
from quotemem.core.privacy import (
    TrimmedSupport,
    choose_compensation,
    clause_words,
    english,
    fact_terms,
    kept_beside,
    naming_value,
    shown_terms,
    support_words,
    trimmed_support,
    unshown_names,
)
from quotemem.core.proposal import ProposalV2
from quotemem.core.query import ScopeV1
from quotemem.core.readability import ByteRange, EvidenceHeader, Readability
from quotemem.core.recall_agent import SHADOW_PURPOSE, ForgetShadowTerms, shadow_terms
from quotemem.core.retained_policy import read_released_extraction_config
from quotemem.core.sensitivity_rules import classify
from quotemem.core.statement import (
    MAX_KEYWORDS,
    MAX_SUPPORT_GROUPS,
    MAX_TOMBSTONE_SOURCES,
    canonical_statement,
    name_norm,
    own_token,
    statement_digest,
    token_contained,
    token_seq,
)
from quotemem.ingestion.authorization import check_add_grant
from quotemem.ingestion.forget_requests import is_request
from quotemem.providers.auth import PrincipalGrant
from quotemem.providers.contracts import EmbeddingProvider, EmbeddingRequest
from quotemem.store.add_outcome import request_key
from quotemem.store.domain import EntityNameRowV1, MemoryRow, RevisionRowV2, TombstoneRowV1
from quotemem.store.entity_ops import (
    Row,
    TxnRows,
    erase_unanchored_entities,
    rekey,
    requote_on,
    reworded_view,
)
from quotemem.store.erasure_gate import (
    FORGET_WAIT_SECONDS,
    background,
    check_request_end,
    gate_for,
    note_admitted,
    one_operation,
    patient_deadline,
    user_request,
    wait_deadline,
)
from quotemem.store.failures import is_runtime_failure
from quotemem.store.forget_admission import (
    CorrectionEdge,
    ForgetVersionSelection,
    ForgottenCitation,
    KeeperGroup,
    bare_name_spans,
    correction_chain,
    forgotten_keywords,
    mentions,
    named_by_type,
    names_another,
    names_replaced,
    naming_parts,
    plan_earlier_wordings,
    plan_forget_admission,
    plan_stating,
    possession_entailed,
    predicate_words,
    replaced_parts,
    restated_by_vector,
    said_terms,
    states_value,
)
from quotemem.store.forget_final_scan import privacy_quiescence
from quotemem.store.indexed import (
    MAX_PAGE_SIZE,
    CandidateRows,
    EvidenceRows,
    LinkTargetRows,
    MemoryTableRows,
    RecallDeliveries,
    RevisionRows,
    ScopedRunRows,
    TombstoneRows,
)
from quotemem.store.judge_records import LOCATOR_PURPOSE
from quotemem.store.memory_commit import (
    GroupsOf,
    expire_unusable,
    proposal_groups,
    readability,
    withhold_spans,
)
from quotemem.store.privacy_compensation import participant_has_source
from quotemem.store.read_model import StoreReadModel
from quotemem.store.writer import MAX_TX_ROWS, RowWrite, Writer, WriteTicket

FORGET_RECEIPT_KINDS = frozenset({
    "receipt:forget_prepare", "receipt:forget_final", "result_index:erasure_receipt",
})
"""The only artifacts the erasure controller writes."""
_CLAUSE_MARKS = re.compile(r"[,;，；、。！？!?\n]|\.(?=\s|$)")
"""Where a replacer's quote is cut into clauses (``ForgetOwner._mentioning``)."""
_HAN_RUNS: Final = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
"""A run of Han characters."""
_SPAN_BATCH = 256
MAX_REMAINDERS: Final = 64
"""The most Evidence parts one forget embeds the readable rest of (``ForgetOwner._prefill``);
a later forget embeds those left."""
_EMBED_FAILURES: Final = frozenset({
    "provider_unavailable", "schema_invalid", "embedding_space_mismatch", "rate_limited",
    "deadline_exceeded", "budget_exceeded",
})
"""Embedding failures after which a forget's run keeps no remainder: its parts stay unfound,
as ``p_evr`` keeps them, until a later complete forget embeds them."""
type RemainderEmbedder = Callable[[Sequence[str]], tuple[str, list[tuple[float, ...]]]]
"""Embeds readable texts in the workspace's space: that space and one vector per text."""
"""Spans withheld per ``admit`` or ``delete`` transaction; each touches at most this many
Evidence rows."""
_CASCADE_LIMIT = 64
"""Historical correcting memories one forget may cascade to before refusing."""
_PLAN_ROUNDS: Final = 4
"""Times a preview reads its plans again without the memories they left unreadable
(``ForgetOwner.preview_plan``): each round only adds losses, so a few reach what the forgets,
run one after another, leave."""
_DEPENDENT_BATCH = 64
"""Dependent memories compensated per ``delete`` transaction."""
_LINK_BATCH = 512
"""Links other memories carry to the retention unit's memories, deleted per ``delete``
transaction."""
_EXPIRY_BATCH = 512
"""Pending candidates expired per ``delete`` transaction."""
_UNIT_OVERHEAD = 3
"""RowOps the unit's ``delete`` transaction adds besides its rows: the fence, ``commit_seq`` and
``scope_seq``."""
_PURGE_BATCH = 256
"""Retention-unit Artifacts and Controls deleted per ``delete`` transaction."""
_COHORT_BATCH = 256
"""Children ended, or candidates deleted, per cancelled-cohort ``delete`` transaction."""
_ENDED = frozenset({"succeeded", "failed"})
"""Run statuses a cancelled cohort leaves as they are."""


class _Residual(QuoteMemError):
    """The final scan's refusal: its own scope's content, not the runtime."""


def _forget_alone(error: Exception) -> bool:
    """Whether a failure is one forget's own: the runtime is sound (``is_runtime_failure``)
    and the Writer still owns the store."""
    return isinstance(error, _Residual) or not (is_runtime_failure(error) or (
        isinstance(error, QuoteMemError) and error.code == "lease_lost"))


_TRANSIENT_CODES: Final = frozenset({
    "provider_unavailable", "authorization_unavailable", "authorization_expired",
    "index_rebuilding", "delivery_busy", "erasure_in_progress", "finalize_pending",
    "extraction_busy",
})
"""The refusals a later attempt of a running forget outlives: provider, authorization, index
and queue states. Never a bound (``budget_exceeded``: a batch over a size bound is the same
batch at every resume) or a changed selection (``reception_stale``), which the forget routes
answer as final."""


def _transient(error: Exception) -> bool:
    """Whether a refusal is one a later attempt may outlive (`_TRANSIENT_CODES`): a forget
    never gives up trimming its dependents for it (``ForgetOwner._delete``)."""
    return isinstance(error, QuoteMemError) and error.code in _TRANSIENT_CODES


def _cancelled_post_image(run: Mapping[str, Any], now: int) -> dict[str, Any]:
    """A cancelled cohort Run's post-image: failed ``erase_fenced``."""
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
    """The ``erase_epoch`` the fence owner writes at: pre-bump before ``admit`` commits, then +1."""
    fence = ErasureFenceV1.model_validate(control["value"])
    return fence.erase_epoch if fence.phase == "admitting" else fence.erase_epoch + 1


def check_controller_artifact(
    writer: Writer, ticket: WriteTicket, kind: str, security_level: str, source: object,
) -> None:
    """The receipt write proof: only the fence's controller, only after the scan.

    The controller writes exactly its receipt pair and their selection, at
    ``restricted``, while its ``erasure`` fence is in the ``scanned`` phase; a
    final receipt must carry the fence's bound ``final_scan_digest``.

    Raises:
        QuoteMemError: ``invalid_write_mode`` for another kind, level, phase, scan
            digest, or a controller whose fence is gone.
    """
    receipt = isinstance(
        source, ForgetPrepareReceiptV3 | ForgetReceiptV3 | ErasureReceiptResultIndexV1,
    )
    if kind not in {"receipt", "result_index"} or not receipt or security_level != "restricted":
        raise QuoteMemError("invalid_write_mode", "The erasure controller writes only its receipts")
    control = None if ticket.run_id is None else _fence_of_controller(writer, ticket.run_id)
    if control is None:
        raise QuoteMemError("invalid_write_mode", "Forget receipts require the controller's fence")
    fence = ErasureFenceV1.model_validate(control["value"])
    if fence.controller_run_id != ticket.run_id or fence.phase != "scanned":
        raise QuoteMemError("invalid_write_mode", "Forget receipts follow the bound final scan")
    if isinstance(source, ForgetReceiptV3) and source.final_scan_digest != fence.scan_digest:
        raise QuoteMemError("invalid_write_mode", "The final receipt names another scan")


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
    """``intent:{scope}:{principal}:forget:{binding}:{ulid}``."""
    return f"intent:{scope_id}:{principal_id}:forget:{binding}:{nonce}"


def fence_key(scope_id: str, controller_run_id: str) -> str:
    """``erasure:{scope}:{controller}``."""
    return f"erasure:{scope_id}:{controller_run_id}"


@dataclass(frozen=True)
class ForgetPrepared:
    """The durable intent and the exact head it binds."""

    intent_key: str
    memory_id: str
    expected_revision_id: str
    expected_content_hash: str
    expires_at: int


@dataclass(frozen=True)
class SpansPrepared:
    """A forget of sentences' durable intent (`ForgetOwner.prepare_spans`)."""

    intent_key: str
    expires_at: int


_EXACT_SCAN_LIMIT: Final = 20_000
"""The most tombstones `ForgetOwner.plan_still_exact` reads before it answers not exact."""


@dataclass(frozen=True)
class _AdmissionPlan:
    """What a forget's admission installs (``ForgetOwner._admission_plan``), with the groups
    its keepers keep (``kept``), the memories it rewords (``rewords``) and what it cites
    (``cited``), which a preview reads."""

    chain: tuple[str, ...]
    replacers: tuple[str, ...]
    reworded: tuple[str, ...]
    versions: tuple[ForgetVersionSelection, ...]
    entity_ids: tuple[str, ...]
    sensitive: bool
    spans: tuple[QuoteSpanV1, ...]
    any_script: bool
    kept: frozenset[tuple[str, tuple[QuoteSpanV1, ...]]] = frozenset()
    rewords: tuple[str, ...] = ()
    cited: tuple[ForgottenCitation, ...] = ()


@dataclass(frozen=True)
class ForgetOutcome:
    """A verified forget: the controller, its receipt pair and the published scope state."""

    controller_run_id: str
    prepare_ref: str
    receipt_ref: str
    receipt: ForgetReceiptV3


@dataclass(frozen=True)
class FinalScan:
    """The scan's six counts over durable storage and the read view, all zero when verified.

    ``records`` counts the retention-unit rows left besides the memory's own: a
    candidate naming the memory, a frozen Artifact, Control or Search delivery,
    or a delivery that returned the memory or an Evidence row of S.
    ``unwithheld_spans`` counts the spans of S no withheld range of their row
    covers; a privacy-erased row keeps no ranges and needs none.
    ``pending_unusable`` counts the pending candidates with a quote overlapping
    S and no usable group left: the expiry must have ended them. A
    pending sibling that keeps a usable group, on a disjoint span of the same
    part included, is not residual work. ``dependents`` and
    ``entities`` count the effects still to apply: a
    dependent the chooser would still requote or a readable survivor still carrying a
    keyword the forget drops, an entity it would still erase.
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
    """The single Writer's forget producer for one workspace."""

    def __init__(
        self, writer: Writer, publisher: StoreReadModel, *, clock: Callable[[], int],
        policy_release_ref: str | None = None,
        checkpoint: Callable[[str], None] | None = None,
        groups_of: GroupsOf | None = None, complete: bool | None = None,
        embed_remainders: RemainderEmbedder | None = None,
    ) -> None:
        self._writer, self._publisher, self._clock = writer, publisher, clock
        self._groups_of: GroupsOf = groups_of or proposal_groups(writer)
        self._policy_release_ref = policy_release_ref
        self._checkpoint: Callable[[str], None] = checkpoint or _no_checkpoint
        self._complete_rule = complete
        self._embed_remainders = embed_remainders
        self._prefilled: dict[str, Mapping[str, tuple[str, list[float], str]]] = {}
        """The remainders each running forget embedded before its admission, by controller."""

    def embed_remainders_with(self, embed: RemainderEmbedder | None) -> None:
        """Give the owner the workspace's embedder for the remainders a complete forget keeps
        (``_prefill``): a complete workspace's services give it; without one, none is kept."""
        self._embed_remainders = embed

    def _complete(self) -> bool:
        """Whether the workspace forgets completely (``explicit-request-v2``, the ``agent``
        Add mode): its released extraction policy's in-chat forget rule, read once; a
        workspace without a released policy does not."""
        if self._complete_rule is None:
            pointer = self._writer.get_control("policy:extraction")
            self._complete_rule = pointer is not None and read_released_extraction_config(
                self._writer, pointer["value"]["current_release_ref"],
            ).parameters.chat_forget == CHAT_FORGET_COMPLETE
        return self._complete_rule

    def forgets_completely(self) -> bool:
        """Whether this workspace's forgets are complete (``_complete``): only then does a
        forget take replacers and a prepare list them."""
        return self._complete()

    # Prepare -------------------------------------------------------------------------------
    def prepare(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str, memory_id: str,
        expected_revision_id: str, restating: Sequence[QuoteSpanV1] = (),
        seed: ForgetSeedV1 | None = None, enforces: int | None = None,
        allowed: Sequence[str] | None = None, listed: Sequence[str] | None = None,
        support_only: bool = False, modifier_only: bool = False,
        rewordings: Sequence[ForgetRewordV1] = (),
        asked: Sequence[ContentBindingDigest] = (), name_asked: bool = False,
    ) -> ForgetPrepared:
        """Issue one bounded intent bound to the memory's current head, in the
        scope's erasure turn (`quotemem.store.erasure_gate`): behind another erasure of the scope
        it waits up to ``FORGET_WAIT_SECONDS`` and then checks everything again
        (`_prepare` has the rest)."""
        return self._in_turn(scope, enforces is None, lambda: self._prepare(
            scope, grant, session_id=session_id, memory_id=memory_id,
            expected_revision_id=expected_revision_id, restating=restating, seed=seed,
            enforces=enforces, allowed=allowed, listed=listed, support_only=support_only,
            modifier_only=modifier_only, rewordings=rewordings, asked=asked,
            name_asked=name_asked))

    def _prepare(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str, memory_id: str,
        expected_revision_id: str, restating: Sequence[QuoteSpanV1] = (),
        seed: ForgetSeedV1 | None = None, enforces: int | None = None,
        allowed: Sequence[str] | None = None, listed: Sequence[str] | None = None,
        support_only: bool = False, modifier_only: bool = False,
        rewordings: Sequence[ForgetRewordV1] = (),
        asked: Sequence[ContentBindingDigest] = (), name_asked: bool = False,
    ) -> ForgetPrepared:
        """Issue one bounded intent bound to the memory's current head.

        ``seed`` is the user's words of the fact a complete forget takes a memory for, given
        when the memory bundles that fact with an unrelated one: its tombstone,
        under the head's primary entity, stands for the head's statement, so a later Add
        may teach the unrelated fact again while a restatement of the forgotten one is still
        checked against the user's words; the memory's earlier statements and its chain's keep
        their own tombstones. The intent binds it.

        ``restating`` names sentences of the scope's Evidence that restate what the memory
        states without being cited by its versions (a complete forget's judged sentences,
        ``service/forget_statement.py``): admission plans them as cited by the head's
        statement, so a survivor quoting one keeps what states only its own fact (the keeper
        paths of ``plan_stating_spans``), ``admit`` and ``delete`` withhold the rest, the dependents
        citing them are compensated and the scan proves them withheld, exactly as the memory's own
        quotes.

        ``enforces`` is the admission time of the forget this one enforces (the relearn
        check of a complete forget withholding a restatement of an already forgotten fact):
        the forget shadow keeps its stamps and stamps what it adds with that
        time (`_shadow_rows`).

        A complete forget takes beyond its base links only the memories a word rule
        nominated and the forget judge confirmed (``allowed``, ``nominations``); none when
        not given. With ``listed`` (the memories its prepare named) its admission refuses
        with ``operation_conflict`` to take any other. With ``support_only`` (a memory known
        only from what other forgets withheld) it takes no replacer at all. The intent binds
        all three. With ``modifier_only`` the user asked to forget only a modifier of a thing
        of theirs (``asks_modifier``): its plan and deletion step forbid only the modifier to a
        survivor about that thing (``ForgottenCitation.modifier``); the intent binds it too. A
        complete forget rewords the memories ``rewordings`` names instead of taking them
        (`_reword_planned`). ``asked`` (keyed digests of the asked values' shadow names)
        keeps the value the user kept out of the forget shadow (`_shadow`).

        Raises:
            QuoteMemError: ``scope_denied`` for an absent or other-scope memory,
                ``operation_conflict`` when the head moved, ``erasure_in_progress``
                while a fence holds the scope.
        """
        writer = self._writer
        session = self._session_token(scope, session_id)
        with writer.write_scope():
            now = self._clock()
            check_add_grant(grant, scope, now)
            self._refuse_fenced(scope)
            item, head = self._current(scope, memory_id)
            if item["current_revision_id"] != expected_revision_id:
                raise QuoteMemError("operation_conflict",
                                    "The memory changed since it was selected")
            erase_epoch = writer.get_counter(f"erase_epoch:{scope.scope_id}")
            if seed is not None:
                seed = ForgetSeedV1.model_validate({
                    **seed.model_dump(mode="json"),
                    "entity_id": MemorySnapshotV2.model_validate(head["snapshot"]).entity_id})
            binding = self._bind_selection(scope, grant.principal_id, session, item, head,
                                           erase_epoch, seed=seed)
            key = intent_key(scope.scope_id, grant.principal_id, str(binding), new_ulid())
            value = ForgetIntentV1(
                state="active", scope_id=scope.scope_id, principal_id=grant.principal_id,
                session_token=session, memory_id=memory_id,
                expected_revision_id=expected_revision_id, selection_digest=binding,
                erase_epoch=erase_epoch, issued_at=now, expires_at=now + INTENT_TTL_MS,
                restating_spans=self._restating(scope, restating), seed=seed,
                enforces=enforces,
                # Only a complete forget takes beyond its base links (``_chain_parts``).
                allowed=None if allowed is None or not self._complete()
                else tuple(sorted(set(allowed))),
                listed=None if listed is None or not self._complete()
                else tuple(sorted(set(listed))),
                support_only=support_only and self._complete(),
                modifier_only=modifier_only,
                rewordings=tuple(rewordings)[:MAX_REWORDINGS] if self._complete() else (),
                asked=tuple(sorted(set(asked)))[:MAX_KEYWORDS] if self._complete() else (),
                name_asked=name_asked and self._complete(),
            )
            control = writer.materialize_control(key, value.model_dump(mode="json"), 0, now)
            _, plan = writer.plan_rows(
                [RowWrite("control", control)], None,
            )
            writer.write_plan(plan)
        return ForgetPrepared(key, memory_id, expected_revision_id, head["content_hash"],
                              value.expires_at)

    def prepare_spans(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str,
        spans: Sequence[QuoteSpanV1], seed: ForgetSeedV1, enforces: int | None = None,
        rewordings: Sequence[ForgetRewordV1] = (),
    ) -> SpansPrepared:
        """`_prepare_spans` in the scope's erasure turn, as `prepare` runs `_prepare`."""
        return self._in_turn(scope, enforces is None, lambda: self._prepare_spans(
            scope, grant, session_id=session_id, spans=spans, seed=seed, enforces=enforces,
            rewordings=rewordings))

    def _prepare_spans(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str,
        spans: Sequence[QuoteSpanV1], seed: ForgetSeedV1, enforces: int | None = None,
        rewordings: Sequence[ForgetRewordV1] = (),
    ) -> SpansPrepared:
        """Issue one bounded intent that forgets sentences no memory states.

        ``spans`` are sentences of the scope's active Evidence that state a fact the user
        asked to forget; ``seed`` is that fact in the user's words with its vector, the
        forget's one tombstone. The intent binds the sentences, the seed and the scope's
        ``erase_epoch``; admission refuses it when the epoch moved (``operation_conflict``).
        ``enforces`` and ``rewordings`` are as `prepare`'s.

        Raises:
            QuoteMemError: ``reference_missing`` for a sentence outside the scope's active
                Evidence, ``budget_exceeded`` past ``MAX_RESTATING_SPANS`` sentences or
                ``MAX_TOMBSTONE_SOURCES`` Evidence rows, ``erasure_in_progress`` while a
                fence holds the scope.
        """
        writer = self._writer
        session = self._session_token(scope, session_id)
        with writer.write_scope():
            now = self._clock()
            check_add_grant(grant, scope, now)
            self._refuse_fenced(scope)
            selected = self._restating(scope, spans)
            if not selected or len(selected) > MAX_RESTATING_SPANS or len(
                    {span.evidence_id for span in selected}) > MAX_TOMBSTONE_SOURCES:
                raise QuoteMemError("budget_exceeded",
                                    "The sentences to forget exceed their bounds")
            erase_epoch = writer.get_counter(f"erase_epoch:{scope.scope_id}")
            binding = self._bind_span_selection(scope, grant.principal_id, session, selected, seed,
                                                erase_epoch)
            key = intent_key(scope.scope_id, grant.principal_id, str(binding), new_ulid())
            value = ForgetIntentV1(
                state="active", scope_id=scope.scope_id, principal_id=grant.principal_id,
                session_token=session, memory_id=None, expected_revision_id=None,
                selection_digest=binding, erase_epoch=erase_epoch, issued_at=now,
                expires_at=now + INTENT_TTL_MS, restating_spans=selected, seed=seed,
                enforces=enforces,
                rewordings=tuple(rewordings)[:MAX_REWORDINGS] if self._complete() else (),
            )
            control = writer.materialize_control(key, value.model_dump(mode="json"), 0, now)
            _, plan = writer.plan_rows(
                [RowWrite("control", control)], None,
            )
            writer.write_plan(plan)
        return SpansPrepared(key, value.expires_at)

    def forget_spans(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str, intent: str,
    ) -> ForgetOutcome:
        """Admit a forget of sentences (`prepare_spans`), run its phases, return its receipt.

        Reads withhold the sentences from the moment the confirmation is received with a
        current intent until it ends (`quotemem.store.erasure_gate`), also while it waits."""
        with self._withheld(scope, grant, intent, None), \
                self._turn(scope, user_request()) as deadline:
            prefilled = self._prefill(scope, intent, None)
            controller = self._admit_in_turn(scope, deadline, lambda: self._admit(
                scope, grant, session_id=session_id, memory_id=None, expected_revision_id=None,
                expected_content_hash=None, intent=intent,
            ))
            note_admitted()
            return self._run(scope, controller, prefilled)

    def preview(
        self, scope: ScopeV1, memory_ids: Sequence[str], *,
        restating: Sequence[QuoteSpanV1] = (), seed: ForgetSeedV1 | None = None,
        bundled: Set[str] = frozenset(), words: Sequence[str] = (), cascade: bool = False,
        allowed: Set[str] | None = None, rewordings: Sequence[ForgetRewordV1] = (),
    ) -> list[tuple[str, bool]]:
        """The readable memories outside ``memory_ids``, current or past, a plan may leave
        unreadable: ``preview_plan``'s first part."""
        return self.preview_plan(scope, memory_ids, restating=restating, seed=seed,
                                 bundled=bundled, words=words, cascade=cascade,
                                 allowed=allowed, rewordings=rewordings)[0]

    def preview_plan(
        self, scope: ScopeV1, memory_ids: Sequence[str], *,
        restating: Sequence[QuoteSpanV1] = (), seed: ForgetSeedV1 | None = None,
        bundled: Set[str] = frozenset(), words: Sequence[str] = (), cascade: bool = False,
        allowed: Set[str] | None = None, requests: Sequence[str] = (), modifier: bool = False,
        rewordings: Sequence[ForgetRewordV1] = (),
    ) -> tuple[list[tuple[str, bool]], tuple[QuoteSpanV1, ...]]:
        """The readable memories outside ``memory_ids``, current or past, a plan may leave
        unreadable, and the spans the plans withhold.

        Writes nothing. Each memory's forget is planned as admission plans it
        (``plan_stating``, the keeper paths a-f), the first with ``restating``; with no
        memory, the sentences alone under ``seed``. A memory any of whose
        readable support groups survives every plan, by meeting no withheld byte or as a
        group the plan keeps (whole, or requoted on its kept parts), stays readable; every
        other readable memory citing a withheld byte is returned, in id order, with whether
        a complete forget then forgets it too (``StatementForget._cascade``): every quote of it
        lies in what the plans withhold, or, for a bare naming memory that is the forgotten
        facts' residue (``bare_name_stays``), in that or what the targets quote. Privacy
        compensation may still requote one
        of them on an earlier version, so the list is what may happen, not less: the
        confirmation reports what did (``StatementForget.confirm_report``). A ``bundled``
        memory is planned as forgotten for the user's ``words`` (``_chain_parts``). A group
        two plans withhold bytes of survives only when each of them keeps it. With
        ``cascade`` (a complete forget by statement's confirmation forgets what is left known
        only from the forget, ``StatementForget._cascade``), a readable bare naming memory
        every quote of which lies in what the plans withhold or the forgotten memories quote
        is returned as forgotten too, so the prepare names it, unless its name stays for the
        user's ``requests`` (``bare_name_stays``). With ``modifier`` the forget is of a
        modifier alone (``ForgetOwner.prepare``'s ``modifier_only``). A readable head a
        target's forget refutes (``_chain_parts``) is returned as forgotten too. A memory
        the plan rewords (``rewordings``) stays readable, and is no such memory.
        """
        writer = self._writer
        with writer.write_scope():
            txn = self._open_txn(scope, None, self._clock())
            withheld: dict[str, list[tuple[int, int]]] = {}
            forgotten_all: set[str] = set()
            refuted_all: set[str] = set()
            planned = self._plans(scope, txn, memory_ids, restating=restating, seed=seed,
                                  bundled=bundled, words=words, allowed=allowed,
                                  rewordings=rewordings, modifier=modifier)
            for memory_id, plan in planned:
                if memory_id is not None:
                    forgotten_all.update((memory_id, *plan.chain))
                    refuted_all.update(set(plan.chain) - set(plan.replacers))
            # A memory the plan rewords stays readable under its new wording.
            reworded = {memory for _, plan in planned for memory in plan.rewords}
            wordings = {wording.memory_id: wording for wording in rewordings}
            # Every plan is read with all the targets gone, as the last of the confirmation's
            # forgets runs: a target keeps nothing for another (each one's forget, run while
            # the other is alive, keeps its words for it, and the other's forget then withholds
            # them), so what the plans withhold together is never less than what the forgets
            # one after another withhold. Nor does a memory the plans leave unreadable keep
            # anything: the forgets, and the cascade's after them, each run on what the ones
            # before it left, so the plans are read again without it until none is lost anew.
            facts: list[ForgottenFact] = []
            told: list[str] = []
            told_vectors: list[tuple[tuple[float, ...], str]] = []
            forgotten_names: set[str] = set()
            forgotten_quotes: list[QuoteSpanV1] = []
            for memory_id in forgotten_all:
                item = writer.get_row("item", memory_id)
                head = None if item is None else writer.get_row(
                    "revision", item["current_revision_id"])
                if head is not None:
                    forgotten_head = MemorySnapshotV2.model_validate(head["snapshot"])
                    facts.append((forgotten_head.statement, forgotten_head.entity_id,
                                  forgotten_head.about_entity_ids))
                    told.append(forgotten_head.statement)
                    told_vectors.append((tuple(forgotten_head.embedding),
                                         forgotten_head.embedding_space))
                    forgotten_names |= {entry.name for entry in forgotten_head.naming}
                    forgotten_quotes += forgotten_head.quote_spans
            read = readability(txn)

            def meets(group: tuple[QuoteSpanV1, ...],
                      ranges: Mapping[str, Sequence[tuple[int, int]]]) -> bool:
                return any(start < span.byte_end and span.byte_start < end
                           for span in group for start, end in ranges.get(span.evidence_id, ()))

            excluded: set[str] = set()
            lost: list[tuple[str, MemorySnapshotV2]] = []
            for _ in range(_PLAN_ROUNDS):
                planned = self._plans(scope, txn, memory_ids, restating=restating, seed=seed,
                                      bundled=bundled, words=words, allowed=allowed,
                                      rewordings=rewordings, modifier=modifier,
                                      gone=forgotten_all, excluded=excluded)
                # Each plan's own withheld bytes and kept groups: a group survives only when
                # every plan withholding a byte of it keeps it (one memory's forget may keep it,
                # the cascade's forget of another, quoting the same words, not).
                withheld = {}
                results: list[tuple[dict[str, list[tuple[int, int]]],
                                    set[tuple[str, tuple[QuoteSpanV1, ...]]]]] = []
                for _, plan in planned:
                    mine: dict[str, list[tuple[int, int]]] = {}
                    for span in plan.spans:
                        withheld.setdefault(span.evidence_id, []).append(
                            (span.byte_start, span.byte_end))
                        mine.setdefault(span.evidence_id, []).append(
                            (span.byte_start, span.byte_end))
                    results.append((mine, set(plan.kept)))
                if not withheld:
                    return [], ()
                lost = []
                dependents: list[tuple[str, MemorySnapshotV2]] = []
                for item in self._pages(MemoryTableRows(scope_kind=scope.scope_kind,
                                                        scope_id=scope.scope_id)):
                    # A past memory the list shows is named too: the plan may leave it
                    # unreadable as well as a current one.
                    memory_id = item["object_id"]
                    if memory_id in forgotten_all or memory_id in reworded or (
                            item["lifecycle_state"] == "erased"):
                        continue
                    head = txn.get("revision", item["current_revision_id"])
                    if head is None or not set(withheld).intersection(head["evidence_ids"]):
                        continue
                    snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
                    if not (read.p_mem(item["lifecycle_state"], snapshot)
                            if item["lifecycle_state"] == "active" else read.p_ver(snapshot)):
                        continue
                    groups = [tuple(snapshot.quote_spans[index] for index in group)
                              for group in snapshot.support_groups
                              if read.p_group(snapshot.quote_spans, group)]
                    if groups and not any(all(
                            (memory_id, group) in kept_here or not meets(group, mine)
                            for mine, kept_here in results) for group in groups):
                        lost.append((memory_id, snapshot))
                    elif groups and all(meets(group, withheld) for group in groups):
                        # Privacy compensation keeps it only if it can requote it
                        # on the planned parts.
                        dependents.append((memory_id, snapshot))
                if dependents:
                    trim = self._planned_trim(txn, read, scope, withheld, restating,
                                              [plan.cited for _, plan in planned],
                                              modifier=modifier)
                    lost += [(memory_id, snapshot) for memory_id, snapshot in dependents
                             if not trim(snapshot)]
                again = {memory_id for memory_id, _ in lost} - excluded
                if not again:
                    break
                excluded |= again
            if lost and seed is not None and detail_question(seed.statement) is not None:
                # When forgetting a detail, the deletion step requotes a dependent on the readable
                # rest of its quotes, as `_compensate` reads it, so the plan does not name it.
                spans = [span for _, plan in planned for span in plan.spans]
                texts = [_quoted_text(txn, span) for span in spans]
                withhold_spans(txn, spans, reason="forgotten", run_id=new_ulid())
                told_now = _Forgotten(
                    frozenset(word for text in texts if text for word in support_words(text)),
                    (*told, seed.statement), True,
                    frozenset(term for text in texts if text for term in fact_terms(text)),
                    (*told_vectors, (tuple(seed.embedding), seed.embedding_space)),
                    modifier=modifier)
                after = readability(txn)
                lost = [entry for entry in lost if None in texts or not _detail_requoted(
                    txn, after, entry[1], told_now)]
            covered = {evidence: list(ranges) for evidence, ranges in withheld.items()}
            # The sentences withheld as stating the fact cover a bare name quoted in them, as a
            # forgotten memory's quotes do (`StatementForget._leftovers`), and so do those.
            for span in (*self._restating(scope, restating), *forgotten_quotes):
                covered.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
            # Every memory the forget leaves readable, with the names and ties they give: what
            # a bare name's residue test reads (`bare_name_stays`). A memory the plans leave
            # unreadable tells no entity.
            unreadable = {memory_id for memory_id, _ in lost}
            left: list[tuple[str, MemorySnapshotV2]] = []
            for item in self._pages(MemoryTableRows(scope_kind=scope.scope_kind,
                                                    scope_id=scope.scope_id)):
                memory_id = item["object_id"]
                head = txn.get("revision", item["current_revision_id"])
                if (memory_id in forgotten_all or memory_id in unreadable or head is None
                        or item["lifecycle_state"] != "active"):
                    continue
                snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
                if not read.p_ver(snapshot):
                    continue
                if memory_id in reworded and memory_id in wordings:
                    # It is read as its rewording leaves it, never as its old head told.
                    snapshot = reworded_view(snapshot, wordings[memory_id], lambda entity: [
                        str(row["name"]) for row in txn.names_of(entity)])
                left.append((memory_id, snapshot))
            about = {entity for _, snapshot in left if not naming_only(snapshot)
                     for entity in (snapshot.entity_id, *snapshot.about_entity_ids)}
            names = {entry.name for _, snapshot in [*left, *lost] for entry in snapshot.naming}
            names |= forgotten_names
            tied = names_tied([*((snapshot.statement, snapshot.entity_id,
                                  snapshot.about_entity_ids) for _, snapshot in [*left, *lost]),
                               *facts])

            def residue(snapshot: MemorySnapshotV2) -> bool:
                return naming_only(snapshot) and bool(snapshot.quote_spans) and all(
                    _within(span, covered.get(span.evidence_id, ()))
                    for span in snapshot.quote_spans) and not bare_name_stays(
                    snapshot, requests, names, facts, tied, told=snapshot.entity_id in about)

            # Forgotten as the cascade forgets it (`_leftovers`): a bare name that is the
            # forgotten facts' residue, any other memory only when every quote of it lies in
            # what is withheld. Else it stays unreadable.
            # A bare name the plan leaves without a readable byte, beside what a forgotten
            # memory quotes, goes whatever the forgotten statements show (`_leftovers` takes an
            # unreadable memory so): "A person is named Rowan." under "Rowan's nickname is
            # Ollie." when the name's bytes go too.
            found: list[tuple[str, bool]] = [
                (memory_id, (residue(snapshot) or all(
                    _within(span, withheld.get(span.evidence_id, ()))
                    for span in snapshot.quote_spans) and any(
                    span.evidence_id == other.evidence_id and span.byte_start < other.byte_end
                    and other.byte_start < span.byte_end
                    for span in snapshot.quote_spans for other in forgotten_quotes))
                    if naming_only(snapshot) else all(
                    _within(span, withheld.get(span.evidence_id, ()))
                    for span in snapshot.quote_spans))
                for memory_id, snapshot in lost]
            if cascade:
                found += [(memory_id, True) for memory_id, snapshot in left
                          if memory_id not in reworded and residue(snapshot)]
            # The heads a target's forget refutes go with it, current or past: named so, as
            # the plans read them gone (a replacer keeps what it says of its own).
            along = refuted_all - set(memory_ids)
            for row in self._pages(MemoryTableRows(scope_kind=scope.scope_kind,
                                                   scope_id=scope.scope_id)) if along else ():
                memory_id, state = row["object_id"], row["lifecycle_state"]
                head = txn.get("revision", row["current_revision_id"])
                if memory_id not in along or head is None or state == "erased":
                    continue
                snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
                if read.p_mem(state, snapshot) if state == "active" else read.p_ver(snapshot):
                    found.append((memory_id, True))
        return sorted(found), tuple(
            QuoteSpanV1(evidence_id=evidence_id, byte_start=start, byte_end=end)
            for evidence_id, ranges in sorted(withheld.items()) for start, end in sorted(ranges))

    def _plans(
        self, scope: ScopeV1, txn: TxnRows, memory_ids: Sequence[str], *,
        restating: Sequence[QuoteSpanV1], seed: ForgetSeedV1 | None, bundled: Set[str],
        words: Sequence[str], allowed: Set[str] | None,
        rewordings: Sequence[ForgetRewordV1] = (), modifier: bool = False,
        gone: Set[str] = frozenset(), excluded: Set[str] = frozenset(),
    ) -> list[tuple[str | None, _AdmissionPlan]]:
        """The plan of each memory's forget, as its admission plans it (`_plan_of`): the
        first withholding ``restating``, a ``bundled`` one for the user's ``words``, each
        rewording ``rewordings`` but its own memory, all of a ``modifier`` alone when the user
        asked that; with no memory, the sentences alone under ``seed``. A
        memory not in the scope has none. A preview reads each with the memories ``gone``
        forgotten too and no keeper of the ``excluded`` memories (``preview_plan``)."""
        writer = self._writer
        found: list[tuple[str | None, _AdmissionPlan]] = []
        for memory_id in memory_ids:
            item = writer.get_row("item", memory_id)
            if item is None or (item["scope_kind"], item["scope_id"]) != (
                    scope.scope_kind, scope.scope_id):
                continue
            head = writer.get_row("revision", item["current_revision_id"])
            if head is None:
                continue
            found.append((memory_id, self._plan_of(
                scope, txn, head, restating=() if found else restating, seed=None,
                words=words if memory_id in bundled else (), allowed=allowed,
                rewordings=tuple(row for row in rewordings if row.memory_id != memory_id),
                modifier=modifier, gone=gone, excluded=excluded)))
        if not found and restating and seed is not None:
            found.append((None, self._plan_of(
                scope, txn, None, restating=restating, seed=seed, words=(), allowed=allowed,
                rewordings=rewordings, excluded=excluded)))
        return found

    def admit_rewordings(
        self, scope: ScopeV1, memory_ids: Sequence[str], *,
        restating: Sequence[QuoteSpanV1] = (), seed: ForgetSeedV1 | None = None,
        bundled: Set[str] = frozenset(), words: Sequence[str] = (),
        allowed: Set[str] | None = None, rewordings: Sequence[ForgetRewordV1] = (),
        modifier: bool = False, asked: Set[str] = frozenset(),
    ) -> list[ForgetRewordV1]:
        """The ``rewordings`` a confirmation of forgetting ``memory_ids`` (planned as
        `preview` plans them, the cascade's forgets among them, of a ``modifier`` alone when
        the user asked that) writes, each with the support and keywords it keeps;
        writes nothing.

        Every forget of the plan is read at once: its withholding is applied to a transaction
        never committed, and what the forgets withhold and their tombstones say is what no
        rewording may show. A rewording is admitted when its memory's head is not one the plan
        forgets, the plan holds no sensitive version (its forgets then trim nothing) and
        `_admit_wording` admits it on the support the withholding leaves: each group of the
        memory's quotes no forget touches whose text shows nothing forgotten, and each
        readable part ``trimmed_support`` keeps. The confirmation's first forget writes
        exactly these: what it withholds and says is contained in what this read does, and
        the support lies outside every forget's withholding, so no later one reaches it.
        """
        if not rewordings:
            return []
        writer = self._writer
        with writer.write_scope():
            txn = self._open_txn(scope, None, self._clock())
            planned = self._plans(scope, txn, memory_ids, restating=restating, seed=seed,
                                  bundled=bundled, words=words, allowed=allowed,
                                  rewordings=rewordings, modifier=modifier)
            if not planned or any(plan.sensitive for _, plan in planned):
                return []
            digests = {row.base_digest for row in rewordings}
            taken = {memory for memory_id, plan in planned if memory_id is not None
                     for memory in (memory_id, *plan.chain)}
            spans = [span for _, plan in planned for span in plan.spans]
            withhold_spans(txn, spans, reason="forgotten", run_id=new_ulid())
            read = readability(txn)
            # As the confirmation's first forget reads them (`_forgotten_words`): a memory's
            # forget skips its restating spans, a forget of spans alone reads them all.
            held = self._restating(scope, restating) if restating and memory_ids else ()
            words_seen: set[str] = set()
            terms: set[str] = set()
            said: list[str] = []
            for span in spans:
                if any(span.evidence_id == other.evidence_id
                       and other.byte_start <= span.byte_start
                       and span.byte_end <= other.byte_end for other in held):
                    continue  # as `_forgotten_words` reads the first forget's restating spans
                text = _quoted_text(txn, span)
                if text is None:
                    return []
                words_seen |= support_words(text)
                terms |= fact_terms(text)
                said.append(text)
            versions = [version for _, plan in planned for version in plan.versions
                        if version.snapshot.statement_digest not in digests]
            told = [version.snapshot.statement for version in versions]
            told += [*words] if bundled else []
            told += [seed.statement] if seed is not None else []
            forgotten = _Forgotten(frozenset(words_seen), tuple(told), True, frozenset(terms),
                                   tuple((tuple(version.snapshot.embedding),
                                          version.snapshot.embedding_space)
                                         for version in versions), modifier=modifier,
                                   name_asked=any(asks_name(word) for word in words),
                                   kept=self._kept_names(told, asked), said=tuple(said))
            pairs = {(version.snapshot.entity_id, version.snapshot.statement_digest)
                     for version in versions}
            admitted: list[ForgetRewordV1] = []
            for wording in rewordings:
                item = txn.get("item", wording.memory_id)
                head = None if item is None else txn.get(
                    "revision", item["current_revision_id"])
                if item is None or head is None or wording.memory_id in taken or (
                        item["lifecycle_state"] != "active"):
                    continue
                snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
                if (snapshot.entity_id, snapshot.statement_digest) in pairs or any(
                        row["entity_id"] == snapshot.entity_id
                        for row in self._pages(TombstoneRows(
                            scope_kind=scope.scope_kind, scope_id=scope.scope_id,
                            statement_digest=snapshot.statement_digest))):
                    continue  # its own wording is forgotten
                if wording.base_digest == statement_digest(wording.statement):
                    # A replacer kept as it is stays when no forget touches what it rests on.
                    if read.p_ver(snapshot) and wording.base_digest == snapshot.statement_digest:
                        admitted.append(wording)
                    continue
                support = self._planned_support(txn, read, snapshot, wording, forgotten)
                if not support:
                    continue
                planned_wording = wording.model_copy(update={
                    "support": support, "keywords": tuple(
                        word for word in snapshot.keywords
                        if not any(value_pattern(value).search(word) for value in left_out(
                            snapshot.statement, wording.statement)))})
                keywords = self._admit_wording(txn, read, planned_wording, forgotten, told)
                if keywords is not None:
                    admitted.append(planned_wording.model_copy(update={"keywords": keywords}))
        return admitted

    def _planned_support(
        self, txn: TxnRows, read: Readability, snapshot: MemorySnapshotV2,
        wording: ForgetRewordV1, forgotten: _Forgotten,
    ) -> tuple[tuple[QuoteSpanV1, ...], ...]:
        """The support a rewording rests on under the plan's withholding (``read``): each
        group of the head no forget touches whose text shows nothing forgotten, and each
        group's readable parts ``trimmed_support`` keeps for what the new statement says
        besides its tie, in that order, ``MAX_SUPPORT_GROUPS`` at most."""
        words = self._wording_words(snapshot.statement, wording, forgotten)
        if words is None:
            return ()
        attributed = attribute_reworded(snapshot.statement, wording.statement)
        if attributed is not None:
            # Having the thing rests on what says so ("我的" of "我的金毛叫旺财"), never on the
            # rest of its quote; without it there is no support (fail closed).
            leads = [(span, found) for span in snapshot.quote_spans
                     if (text := _quoted_text(txn, span)) is not None and (
                         found := existence_leads(text, wording.statement, attributed[1],
                                                  attributed[0])) is not None]
            if leads and len(leads) == len(snapshot.quote_spans):
                parts = [span.model_copy(update={
                    "byte_start": span.byte_start + len(text[:start].encode("utf-8")),
                    "byte_end": span.byte_start + len(text[:end].encode("utf-8"))})
                    for span, found in leads for start, end in found
                    if (text := _quoted_text(txn, span)) is not None]
                return tuple((part,) for part in dict.fromkeys(parts)
                             if read.p_span(part))[:MAX_SUPPORT_GROUPS]
        shown = shown_terms if forgotten.any_script else support_words
        removed = left_out(snapshot.statement, wording.statement)
        forms = [value_pattern(form) for form in removed]
        groups: list[tuple[QuoteSpanV1, ...]] = []
        for group in snapshot.support_groups:
            spans = tuple(snapshot.quote_spans[index] for index in group)
            texts = [_quoted_text(txn, span) for span in spans]
            if read.p_group(snapshot.quote_spans, group) and all(
                    text is not None and not shown(text) & words
                    and not any(form.search(text) for form in forms) for text in texts):
                groups.append(spans)
        # A Latin name the old wording already showed over the same quotes is the head's own
        # reading of them ("Hangzhou" of "The user moved from Shanghai to Hangzhou." over
        # "我上个月从上海搬到了杭州"): the kept parts need not show it in Latin.
        said = _probe(wording, snapshot.statement)
        quoted = [text for span in snapshot.quote_spans
                  if (text := _quoted_text(txn, span)) is not None]
        # Only while every Han run the plan withholds in its quotes is matched by a name or
        # value the new wording leaves out ("上海" by "Shanghai", "我弟弟叫李伟" by "李伟"): a
        # Latin form of a forgotten value it still showed ("Li Wei" for a withheld "李伟")
        # would otherwise pass as the head's own reading. A kin word withheld beside it
        # ("儿子") is no name another script could show.
        if forgotten.any_script and english(wording.statement) and not all(
                text.isascii() for text in quoted) and len([
                    run for run in _withheld_runs(txn, read, snapshot.quote_spans)
                    if not kin_word(run)]) <= len(removed):
            said = " ".join(word for word in said.split() if not (
                word[:1].isupper() and word.strip(".,;").isascii()
                and token_contained(word.strip(".,;"), snapshot.statement)
                and word.strip(".,;").casefold() not in {"the", "a", "an"}))
        probe = snapshot.model_copy(update={"statement": said})
        # The word a kept name ends with that the forgotten one ends with too ("幼儿园" of a
        # kept 蓝湾幼儿园 beside a forgotten 启明幼儿园, "Kindergarten" of Lanwan and Qiming
        # Kindergarten) names neither: a rewording that only drops the phrases naming
        # forgotten values as replaced keeps it (`_contrast_values`).
        shared = frozenset(term for suffix in name_kinds(wording.statement)
                           for term in (*shown(suffix), *fact_terms(suffix))) if (
            _contrast_values(snapshot.statement, wording, forgotten)) else frozenset[str]()
        # A Han form is read by its terms, as the withheld spans are (``fact_terms``): "琴"
        # of a forgotten "陈雅琴" is no word of a kept "钢琴老师".
        trimmed = trimmed_support(probe, read, forgotten=words.union(*(
            fact_terms(form) if forgotten.any_script else shown(form) for form in removed))
                                  - shared,
                                  quoted=lambda span: _quoted_text(txn, span),
                                  speakers=lambda span: _speaker_words(txn, span),
                                  any_script=forgotten.any_script, shared=shared)
        groups += [] if trimmed is None else list(trimmed.groups)
        return tuple(dict.fromkeys(groups))[:MAX_SUPPORT_GROUPS]

    # Admit -------------------------------------------------------------------------------
    def forget(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str, memory_id: str,
        expected_revision_id: str, expected_content_hash: str, intent: str,
    ) -> ForgetOutcome:
        """Admit, run the phases and return the verified receipt.

        A repeated confirm of a consumed intent resumes or returns its original
        controller's outcome. Reads withhold the memory from the moment the
        confirmation is received with a current intent until it ends, also while it waits
        for the scope's erasure turn (`quotemem.store.erasure_gate`).
        """
        with self._withheld(scope, grant, intent, memory_id, expected_revision_id,
                            expected_content_hash), \
                self._turn(scope, user_request()) as deadline:
            prefilled = self._prefill(scope, intent, memory_id)
            controller = self._admit_in_turn(scope, deadline, lambda: self._admit(
                scope, grant, session_id=session_id, memory_id=memory_id,
                expected_revision_id=expected_revision_id,
                expected_content_hash=expected_content_hash, intent=intent,
            ))
            note_admitted()
            return self._run(scope, controller, prefilled)

    def _run(self, scope: ScopeV1, controller: str,
             prefilled: Mapping[str, tuple[str, list[float], str]]) -> ForgetOutcome:
        """Run an admitted forget to its receipt, its keeping the ``prefilled`` remainders
        (``_prefill``)."""
        if prefilled:
            self._prefilled[controller] = prefilled
        try:
            # A repeated confirm runs a forget left to repair again (`_attend`).
            _adopt(self._writer, scope, controller)
            return self.resume(scope, controller)
        except Exception as error:
            # A forget refused for good would hold its user until some boot resumes it.
            if _forget_alone(error) and not _transient(error):
                _attend(self._writer, scope, controller, self._clock(), "forget_run", error)
            raise
        finally:
            self._prefilled.pop(controller, None)

    def _intent_exact(self, scope: ScopeV1, value: ForgetIntentV1,
                      planned: _AdmissionPlan) -> bool:
        """Whether an intent issued at an older ``erase_epoch`` is still exactly its plan.

        The caller has already recomputed the selection at the intent's own epoch (same head,
        content, session, seed and sentences). The plan computed now must also concern only
        rows that no change since the intent was issued touched (`plan_still_exact`): every
        memory it forgets, rewords, replaces or lists, every Evidence it withholds or cites,
        and no tombstone or memory of its facts written since. A workspace whose intent did not
        bind the memories its prepare listed binds none, so the plan computed now stands in.
        """
        memories = {*planned.chain, *planned.replacers, *planned.reworded, *planned.rewords,
                    *(value.listed or ()), *(value.allowed or ()),
                    *(reword.memory_id for reword in value.rewordings)}
        if value.memory_id is not None:
            memories.add(value.memory_id)
        spans = [*planned.spans, *value.restating_spans,
                 *(quote for version in planned.versions
                   for quote in version.snapshot.quote_spans)]
        # A source Evidence none of the plan's sentences or quotes points into is read whole.
        quoted = {span.evidence_id for span in spans}
        evidence = {item for version in planned.versions
                    for item in version.source_evidence_ids if item not in quoted}
        digests = {version.snapshot.statement_digest for version in planned.versions}
        entities = set(planned.entity_ids)
        if value.seed is not None and value.seed.entity_id:
            entities.add(value.seed.entity_id)
        return self.plan_still_exact(
            scope, since_ms=value.issued_at, since_epoch=value.erase_epoch, memories=memories,
            evidence=evidence, spans=spans, digests=digests, entities=entities)

    def plan_still_exact(
        self, scope: ScopeV1, *, since_ms: int, since_epoch: int, memories: Collection[str],
        evidence: Collection[str] = (), spans: Collection[QuoteSpanV1] = (),
        digests: Collection[str] = (), entities: Collection[str] = (),
    ) -> bool:
        """Whether a forget plan read at ``since_epoch`` and ``since_ms`` is still exact now,
        although another forget has moved the scope's ``erase_epoch`` since.

        Invariant: a plan whose epoch is behind is confirmed only when nothing it concerns
        changed after it was read. "Changed" is decided from the rows, never from the epoch,
        and on the safe side: anything written at or after the plan's read time, or by a
        forget admitted at or after its epoch, counts as a change. Concretely, it is exact
        when all of these hold:

        - every memory in ``memories`` exists in the scope, its memory was not updated and its
          current revision not recorded since (superseded, corrected, re-taught or forgotten
          memories fail), and the ranges its head quotes pass the next rule;
        - every Evidence the plan reads is active; one in ``evidence`` (read whole) holds no
          withheld range written since (by a Run created since or admitted at or after the
          epoch), and one read only through ``spans`` or the memories' quotes holds no such
          range that overlaps or touches a range the plan reads (a partly erased sentence of
          the plan fails; another sentence of the same message withheld does not change the
          plan's bytes);
        - no tombstone written since (by time or by its forget Run's epoch) shares a
          statement digest with the plan or cites an Evidence the plan reads whole (a
          tombstone of another fact of the same subject or message does not change this
          plan: its bytes are checked by the rule above, and every row the plan reads is
          checked by the other rules);
        - no entity row the plan names was updated since (erased, merged or renamed);
        - no memory created since states one of the plan's facts (same statement digest).

        Changes the epoch did not mark (an Add between the prepare and the other forget) are
        what a plan at an unchanged epoch already accepts, subject to the same head and
        selection checks.

        What does not depend on the clock: every row written by a Run admitted after the
        epoch moved, or by a forget admitted at the plan's epoch or later (the Run's
        ``admitted_erase_epoch``, which no clock sets), is seen whatever time it carries,
        when it is a head revision of a memory the plan reads, a withheld range of an
        Evidence it reads, a tombstone, or the head of a memory created since that states
        one of its facts; so is a memory or Evidence row deleted or no longer active.

        What does depend on it: a memory row whose lifecycle changed without a new head (its
        update carries no Run), an entity row updated, and any row written by a Run admitted
        before the epoch moved, are judged by their timestamps against ``since_ms``. The store
        holds no write sequence for the plan to be compared with without a format change: an
        intent records the epoch and the time it was read, not a commit sequence. A clock
        behind ``since_ms`` now answers not exact; a clock set back and then forward again
        past ``since_ms`` between the prepare and the confirmation can hide such a row's
        change. A scope whose tombstones exceed ``_EXACT_SCAN_LIMIT`` is not scanned and the
        plan is not exact. Writes nothing; runs under the Writer mutex.

        Args:
            scope: The plan's scope.
            since_ms: When the plan was read (the prepare's clock).
            since_epoch: The ``erase_epoch`` the plan was read at.
            memories: Every memory the plan targets, reads, lists or rewords.
            evidence: Every Evidence the plan reads whole (no byte ranges known).
            spans: Every sentence the plan withholds or quotes beyond the memories' own.
            digests: The statement digests of the facts the plan forgets.
            entities: The entities the plan's tombstones and names concern.
        """
        writer = self._writer
        with writer.write_scope():
            # A clock behind the plan's read time cannot order rows against it.
            if writer.get_counter(f"erase_epoch:{scope.scope_id}") < since_epoch or (
                    self._clock() < since_ms):
                return False
            runs: dict[str, bool] = {}

            def recent_run(run_id: str) -> bool:
                if run_id not in runs:
                    run = writer.get_row("run", run_id)
                    # A forget admitted at the plan's epoch or later ran after the plan was
                    # read, or is counted so, and so is any Run admitted after the epoch
                    # moved (a Run admitted before cannot commit after it: erase_fenced).
                    # These two rules need no clock; the time rule adds the rest.
                    admitted = None if run is None else run.get("admitted_erase_epoch")
                    runs[run_id] = run is None or run["created_at"] >= since_ms or (
                        admitted is not None and (admitted > since_epoch or (
                            run["run_kind"] == "forget" and admitted >= since_epoch)))
                return runs[run_id]

            whole = set(evidence)
            facts = set(digests)
            ranges: dict[str, list[tuple[int, int]]] = {}
            for span in spans:
                ranges.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
            for memory_id in sorted(set(memories)):
                item = writer.get_row("item", memory_id)
                if item is None or item["scope_id"] != scope.scope_id or (
                        item["updated_at"] >= since_ms):
                    return False
                head = writer.get_row("revision", item["current_revision_id"])
                if head is None or head["recorded_at"] >= since_ms or recent_run(
                        head["origin_run_id"]):
                    return False
                for quote in head["snapshot"]["quote_spans"]:
                    ranges.setdefault(quote["evidence_id"], []).append(
                        (quote["byte_start"], quote["byte_end"]))
            for evidence_id in sorted(whole | ranges.keys()):
                row = writer.get_row("evidence", evidence_id)
                if row is None or row["scope_id"] != scope.scope_id or row["state"] != "active":
                    return False
                read = ranges.get(evidence_id, [])
                for held in row["withheld_spans"]:
                    if not recent_run(held["run_id"]):
                        continue
                    # Touching counts: a sentence grown into the plan's bytes is a change.
                    if evidence_id in whole or any(
                            held["byte_start"] <= end and held["byte_end"] >= start
                            for start, end in read):
                        return False
            scanned = 0
            for row in self._pages(TombstoneRows(scope_kind=scope.scope_kind,
                                                 scope_id=scope.scope_id,
                                                 after_tombstone_seq=0)):
                scanned += 1
                if scanned > _EXACT_SCAN_LIMIT:
                    return False
                if not (row["created_at"] >= since_ms or recent_run(row["forget_run_id"])):
                    continue
                if (row["statement_digest"] in facts
                        or whole.intersection(row["source_evidence_ids"])):
                    return False
            for entity_id in sorted(set(entities)):
                # A reserved subject ("user") has no row; an entity row the other forget
                # erased, merged or renamed was updated since.
                entity = writer.get_row("entity", entity_id)
                if entity is not None and entity["updated_at"] >= since_ms:
                    return False
            for digest in sorted(facts):
                for row in self._pages(MemoryTableRows(scope_kind=scope.scope_kind,
                                                       scope_id=scope.scope_id,
                                                       statement_digest=digest)):
                    if row["object_id"] in memories:
                        continue
                    if row["created_at"] >= since_ms:
                        return False
                    head = writer.get_row("revision", row["current_revision_id"])
                    if head is None or recent_run(head["origin_run_id"]):
                        return False
        return True

    def _admission_plan(self, scope: ScopeV1, txn: TxnRows, value: ForgetIntentV1,
                        head: Row | None) -> _AdmissionPlan:
        """What admitting ``value`` forgets and withholds at the scope's current rows: the
        chain, D, the entities and S (``plan_stating_spans``), as admission part (a) installs them
        in the fence; with no ``head``, a forget of sentences under the intent's seed. Writes
        nothing: ``_prefill`` reads it ahead of the admission."""
        return self._plan_of(
            scope, txn, head, restating=value.restating_spans, seed=value.seed,
            words=() if value.seed is None else (value.seed.statement,),
            allowed=frozenset(value.allowed or ()), support_only=value.support_only,
            rewordings=value.rewordings, modifier=value.modifier_only)

    def _plan_of(
        self, scope: ScopeV1, txn: TxnRows, head: Row | None, *,
        restating: Sequence[QuoteSpanV1], seed: ForgetSeedV1 | None, words: Sequence[str],
        allowed: Set[str] | None, support_only: bool = False,
        rewordings: Sequence[ForgetRewordV1] = (), modifier: bool = False,
        gone: Set[str] = frozenset(), excluded: Set[str] = frozenset(),
    ) -> _AdmissionPlan:
        """The plan of one forget (`_admission_plan`), which a prepare's preview reads too
        (``preview``, ``admit_rewordings``): of the memory at ``head`` withholding ``restating``,
        bundled for the user's ``words``, taking beyond its base links only ``allowed`` (every
        nomination when None), rewording ``rewordings``, of a ``modifier`` alone when the user
        asked that (``ForgetIntentV1.modifier_only``); with no ``head``, the sentences alone
        under ``seed``. A preview plans it with the memories ``gone`` forgotten too and no
        keeper of the ``excluded`` memories (``preview_plan``)."""
        if head is None:
            assert seed is not None
            restated = _restated(
                USER, seed.statement, seed.embedding, seed.embedding_space,
                self._restating(scope, restating),
                sensitive=bool(classify([seed.statement])))
            keepers = [keeper for keeper in self._keeper_groups(scope, txn, set(), [restated])
                       if keeper.memory_id not in excluded]
            detail = detail_question(seed.statement)
            if detail is not None and not attribute(detail) and not _NAME_ASKED.search(
                    seed.statement):
                # A question for a place: a memory that only names something may be what it
                # asks, so its name is not cut out across languages or inside Han text.
                keepers = [replace(keeper, uncut=len(keeper.anchors) == 1 and {
                    entity for entity, _ in keeper.names} == set(keeper.anchors))
                           for keeper in keepers]
            asking = self._seed_asking(scope, txn, restated)
            spans, live = plan_stating(
                [asking], keepers, any_script=True,
                quoted=lambda span: _quoted_text(txn, span))
            # A memory the forget rewords has its present wording tombstoned here too, as a
            # forget of it whole would (a naming memory's name stays known as forgotten).
            return _AdmissionPlan((), (), (), tuple(self._with_wordings((), rewordings)), (),
                                  False, tuple(spans), True, frozenset(
                (keepers[index].memory_id, keepers[index].spans)
                for index in live if keepers[index].memory_id), cited=(asking,))
        memory_id = cast(str, head["memory_id"])
        # A complete forget takes beyond the base links and its linked replacers only what its
        # judge confirmed (``ForgetIntentV1.allowed``), and nothing its prepare did not name.
        refuted, replacers = self._chain_parts(
            txn, memory_id, self._restating(scope, restating), words, allowed=allowed,
            support_only=support_only)
        # A replacer the forget rewords stays (``ForgetRewordV1``): its quotes' parts naming the
        # forgotten value are withheld, and its first forget rewords it.
        staying = tuple(sorted(set(replacers) & {row.memory_id for row in rewordings}))
        replacers = tuple(memory for memory in replacers if memory not in staying)
        chain = tuple(sorted({*refuted, *replacers}))
        forgotten = (memory_id, *chain)
        reworded = self._reworded(scope, txn, forgotten)
        # A bundle naming the user's words as a value it replaced goes for them: its own
        # quotes are cited as a replacer's are, so a memory sharing its clause keeps the
        # words that name no forgotten value (``_chain_parts``).
        named = self._bundle_values(txn, memory_id, {memory_id, *refuted}, words)
        # One kept under a new wording cites only the words that wording leaves out, in its
        # quotes' own script ("上海" of "我上个月从上海搬到了杭州" for "Shanghai"); one kept as
        # it is cites what a replacer's quotes name of the forgotten value.
        forgotten_quotes = self._quoted_texts(txn, (memory_id, *refuted))
        # The values a contrast tail it drops names are left out too ("4471" of "The user's
        # locker code is 8836 now, not 4471." beside a forgotten "My locker code is 4471."),
        # though no capitalized run shows them (`contrasted_values`).
        told = [*forgotten_quotes, *(said for memory in (memory_id, *refuted)
                                    if (said := self._head_statement(txn, memory)) is not None)]
        left = {memory: self._left_forms(txn, memory, list(dict.fromkeys([
                    *left_out(statement, row.statement),
                    *(value for value in contrasted_values(statement, told)
                      if not shows(row.statement, value))])), forgotten_quotes)
                for row in rewordings if (memory := row.memory_id) in staying
                and (statement := self._head_statement(txn, memory)) not in (None, row.statement)}
        cited_as = (*replacers, *(memory for memory in staying if memory not in left))
        versions, entity_ids, cited = self._chain_plan(
            (*forgotten, *staying), reworded,
            replacers=self._replaced(txn, (memory_id, *refuted), cited_as)
            if named is None else dict.fromkeys((memory_id, *cited_as), named),
            narrowing=None if named is None else dict.fromkeys((memory_id, *cited_as), words),
            exact=left)
        if modifier:
            cited = [replace(version, modifier=True) for version in cited]
        if staying:
            # It keeps its versions and its entities: only its cited parts go.
            versions, entity_ids = self._without(txn, versions, entity_ids, forgotten, staying)
        versions = self._with_wordings(versions, rewordings)
        sensitive = any(version.sensitive for version in cited)
        if restating:
            # The sentences restating the fact are cited by the selected head's own
            # statement, so a survivor quoting one keeps its own clause as
            # it would in the head's quote (paths a-e).
            snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
            cited = [*cited, _restated(
                snapshot.entity_id, snapshot.statement, snapshot.embedding,
                snapshot.embedding_space, self._restating(scope, restating),
                sensitive=sensitive, about=snapshot.about_entity_ids, modifier=modifier)]
        # A complete forget's clause keeper cuts quotes in any script (``clause_spans``); only
        # a complete forget restates sentences or seeds a bundled memory.
        any_script = bool(restating) or seed is not None or bool(words) or self._complete()
        keepers = [keeper for keeper in (
            *self._keeper_groups(scope, txn, {*forgotten, *staying, *gone}, cited),
            *self._reworded_keepers(txn, reworded)) if keeper.memory_id not in excluded]
        spans, live = plan_stating(
            cited, keepers, withdrawn=self._withdrawn(txn, {*forgotten, *staying, *gone}, cited),
            any_script=any_script, quoted=lambda span: _quoted_text(txn, span))
        exposed = {memory for memory in left
                   if not left[memory] or not self._breaks(txn, memory, spans)}
        if exposed:
            # A replacer kept under a new wording that S leaves a group of whole, or whose left
            # out words its quotes do not show, would keep its old wording readable should its
            # rewording fail: it is taken as before.
            return self._plan_of(
                scope, txn, head, restating=restating, seed=seed, words=words, allowed=allowed,
                support_only=support_only,
                rewordings=tuple(row for row in rewordings if row.memory_id not in exposed),
                modifier=modifier, gone=gone, excluded=excluded)
        return _AdmissionPlan(chain, tuple(replacers), tuple(reworded), tuple(versions),
                              tuple(entity_ids), sensitive, tuple(spans), any_script, frozenset(
                                  (keepers[index].memory_id, keepers[index].spans)
                                  for index in live if keepers[index].memory_id),
                              tuple(row.memory_id for row in rewordings), tuple(cited))

    def _left_forms(self, txn: TxnRows, memory_id: str, values: Sequence[str],
                    forgotten: Sequence[str]) -> list[str]:
        """The forms in which a reworded replacer's quotes show the ``values`` its new wording
        leaves out: each value, and for a value no quote shows in its own script, the Han runs
        a Han quote shares with the ``forgotten`` memories' quotes (``han_forms``)."""
        texts = [text for text in (self._span_text(span) for span in self._head_quotes(
            txn, memory_id)) if text]
        forms = list(values)
        for value in values:
            if not any(value_pattern(value).search(text) for text in texts):
                # A code its quotes give by its core is shown so ("3K728" for "浙A3K728").
                coded = [form for text in texts for form in code_forms(value, text)]
                forms += coded or [form for text in texts if not text.isascii()
                                   for form in han_forms(text, forgotten)]
        return list(dict.fromkeys(forms))

    def _head_quotes(self, txn: TxnRows, memory_id: str) -> list[QuoteSpanV1]:
        item = txn.get("item", memory_id)
        head = None if item is None else txn.get("revision", item["current_revision_id"])
        return [] if head is None else MemorySnapshotV2.model_validate(
            head["snapshot"]).quote_spans

    @staticmethod
    def _head_statement(txn: TxnRows, memory_id: str) -> str | None:
        item = txn.get("item", memory_id)
        head = None if item is None else txn.get("revision", item["current_revision_id"])
        return None if head is None else head["snapshot"]["statement"]

    def _breaks(self, txn: TxnRows, memory_id: str, spans: Sequence[QuoteSpanV1]) -> bool:
        """Whether ``spans`` withhold a byte of every support group of the memory's head, so
        that its present wording is read no more."""
        item = txn.get("item", memory_id)
        head = None if item is None else txn.get("revision", item["current_revision_id"])
        if head is None:
            return False
        snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
        return all(any(_overlaps_any(snapshot.quote_spans[index], spans) for index in group)
                   for group in snapshot.support_groups)

    def _without(
        self, txn: TxnRows, versions: Sequence[ForgetVersionSelection], entity_ids: Sequence[str],
        forgotten: Sequence[str], staying: Sequence[str],
    ) -> tuple[list[ForgetVersionSelection], list[str]]:
        """D and the anchored entities of a plan less what only the ``staying`` replacers
        give: their versions get no tombstone and their entities are no erasure candidates
        (``_admission_plan``)."""
        rows = [self._writer.get_row("revision", version.revision_id) for version in versions]
        kept = [version for version, row in zip(versions, rows, strict=True)
                if row is None or row["memory_id"] not in staying]
        anchored = {entity for memory_id in forgotten
                    for revision in self._revisions(txn, memory_id)
                    for entity in _entities(MemorySnapshotV2.model_validate(revision["snapshot"]))}
        return kept, [entity for entity in entity_ids if entity in anchored]

    def _wording_versions(self, rewordings: Sequence[ForgetRewordV1]
                          ) -> list[ForgetVersionSelection]:
        """The present wordings of the memories a forget rewords: each one's newest version
        stating what its rewording was planned from (``ForgetRewordV1.base_digest``). D
        tombstones them, so the old wording, which tells the forgotten detail, is never told
        again; they join no forget shadow and add no forgotten word (`_reword_planned`)."""
        found: list[ForgetVersionSelection] = []
        for wording in rewordings:
            if wording.base_digest == statement_digest(wording.statement):
                continue  # kept as it is: nothing of it is forgotten
            item = self._writer.get_row("item", wording.memory_id)
            if item is None or item["lifecycle_state"] != "active":
                continue
            revisions = sorted(self._pages(RevisionRows(memory_id=wording.memory_id)),
                               key=lambda row: row["version"])
            base = [row["object_id"] for row in revisions
                    if row["snapshot"]["statement_digest"] == wording.base_digest]
            if base:
                found += [version for version in _plan_selection(item, revisions).versions
                          if version.revision_id == base[-1]]
        return found

    def _with_wordings(self, versions: Sequence[ForgetVersionSelection],
                       rewordings: Sequence[ForgetRewordV1]) -> list[ForgetVersionSelection]:
        """D with the present wordings of the memories the forget rewords
        (`_wording_versions`), one per pair, in pair order as `_chain_plan` gives it."""
        if not rewordings:
            return list(versions)
        merged = {(version.snapshot.entity_id, version.snapshot.statement_digest): version
                  for version in self._wording_versions(rewordings)}
        merged |= {(version.snapshot.entity_id, version.snapshot.statement_digest): version
                   for version in versions}
        return [merged[pair] for pair in sorted(merged)]

    def _prefill(self, scope: ScopeV1, intent: str, memory_id: str | None,
                 ) -> dict[str, tuple[str, list[float], str]]:
        """The remainders a complete forget of ``intent`` will keep (``ForgetRemainderV1``),
        embedded before its admission, outside every lock and before any fence holds the
        scope, so neither its Search nor its Adds wait for the call.

        The admission's plan is read ahead (``_admission_plan``): each Evidence part its S
        withholds bytes of, and each earlier one a forget withheld bytes of that still keeps
        no remainder, gets its readable text once S is withheld (``remainder_text``), embedded
        by one call. ``delete`` keeps a vector only on a part whose readable text is still exactly
        that one (``_keep_prefilled``); a crash, a plan that moved or a failed call keeps
        none, and the part stays unfound (``p_evr``) until a later forget's prefill. Empty without
        an embedder or a complete workspace, for an intent no longer active, or when the
        call fails (``_EMBED_FAILURES``)."""
        if self._embed_remainders is None or not self._complete():
            return {}
        writer = self._writer
        with writer.write_scope():
            stored = writer.get_control(intent)
            if stored is None or not intent.startswith(f"intent:{scope.scope_id}:"):
                return {}
            value = ForgetIntentV1.model_validate(stored["value"])
            if value.state != "active" or value.memory_id != memory_id:
                return {}
            txn = self._open_txn(scope, None, self._clock())
            try:
                head = None if memory_id is None else self._current(scope, memory_id)[1]
                planned = self._admission_plan(scope, txn, value, head)
            except QuoteMemError:
                return {}  # the admission refuses it itself
            added: dict[str, list[tuple[int, int]]] = {}
            for span in planned.spans:
                added.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
            texts: dict[str, str] = {}
            for row, text in self._remainder_parts(txn):
                if row["object_id"] not in added:
                    texts[row["object_id"]] = text
            for evidence_id, ranges in added.items():
                row = txn.get("evidence", evidence_id)
                if row is None or row["state"] != "active" or any(
                        span["reason"] not in FORGET_REASONS for span in row["withheld_spans"]):
                    continue
                text = remainder_text(row["text"], row["byte_start"], [
                    *((span["byte_start"], span["byte_end"]) for span in row["withheld_spans"]),
                    *ranges])
                if any(character.isalnum() for character in text):
                    texts[evidence_id] = text
            texts = dict(list(texts.items())[:MAX_REMAINDERS])
        if not texts:
            return {}
        try:
            space, vectors = self._embed_remainders(list(texts.values()))
        except QuoteMemError as error:
            if error.code not in _EMBED_FAILURES:
                raise
            return {}
        return {evidence_id: (text, list(vector), space)
                for (evidence_id, text), vector in zip(texts.items(), vectors, strict=True)}

    def admit(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str, memory_id: str | None,
        expected_revision_id: str | None, expected_content_hash: str | None, intent: str,
    ) -> str:
        """`_admit` in the scope's erasure turn (`quotemem.store.erasure_gate`): the only place an
        erasure fence is installed, so no erasure starts outside a turn."""
        controller = self._in_turn(scope, user_request(), lambda: self._admit(
            scope, grant, session_id=session_id, memory_id=memory_id,
            expected_revision_id=expected_revision_id,
            expected_content_hash=expected_content_hash, intent=intent))
        note_admitted()
        return controller

    def _admit(
        self, scope: ScopeV1, grant: PrincipalGrant, *, session_id: str, memory_id: str | None,
        expected_revision_id: str | None, expected_content_hash: str | None, intent: str,
    ) -> str:
        """Bind the intent, admit both Runs and install the fence with the plan.

        A live Add owner of the scope (a ``preparing`` or ``held`` finalize
        lease) is cancelled in the same transaction: its lease
        becomes ``cancelled``, its Run fails ``erase_fenced``, its pool
        invocation is told to stop, and the in-memory admission order stops
        holding it as running. ``delete`` settles its cohort; the next owner
        reprocesses its requests.

        Returns:
            The controller Run id (the original one on a repeated confirm).

        Raises:
            QuoteMemError: ``operation_conflict`` for an unknown, expired or differently
                bound intent or a moved head; ``erasure_in_progress`` behind another
                fence; ``budget_exceeded``, before any write, for a plan over its
                bounds (``_preflight``).
        """
        session = self._session_token(scope, session_id)
        writer = self._writer
        with writer.write_scope():
            now = self._clock()
            check_add_grant(grant, scope, now)
            stored = writer.get_control(intent)
            if stored is None or not intent.startswith(f"intent:{scope.scope_id}:"):
                raise QuoteMemError("operation_conflict", "The forget confirmation is not current")
            value = ForgetIntentV1.model_validate(stored["value"])
            if (value.principal_id, value.session_token, value.memory_id,
                    value.expected_revision_id) != (grant.principal_id, session, memory_id,
                                                    expected_revision_id):
                raise QuoteMemError("operation_conflict",
                                    "The confirmation repeats another selection")
            if value.state == "consumed":
                assert value.controller_run_id is not None
                return value.controller_run_id
            if value.state == "expired" or now >= value.expires_at:
                self._expire(stored, value, now)
                raise QuoteMemError("intent_expired", "The forget confirmation expired")
            self._refuse_fenced(scope)
            self._refuse_maintenance(scope)
            live = self._live_owner(scope)
            erase_epoch = writer.get_counter(f"erase_epoch:{scope.scope_id}")
            seed = value.seed
            shadowed = seed
            if memory_id is None:
                # A forget of sentences: S is the sentences less what a
                # survivor keeps of them (path e: its own clause), D only what it rewords.
                assert seed is not None
                selected = self._restating(scope, value.restating_spans)
                if all(is_request(self._span_text(span) or "", complete=True)
                       for span in selected):
                    # Only the requests' own sentences, for a fact no memory or other
                    # sentence states: nothing forgotten names the request's words, so they
                    # hide nothing more (a memory the forget kept stays found).
                    shadowed = None
                # The selection is recomputed at the epoch the intent was issued at: the same
                # sentences, the same seed, the same session, or a conflict.
                if self._bind_span_selection(
                        scope, grant.principal_id, session, selected, seed,
                        value.erase_epoch) != value.selection_digest:
                    raise QuoteMemError("operation_conflict",
                                        "The scope changed since the selection")
                planned = self._admission_plan(scope, self._open_txn(scope, None, now), value, None)
                if erase_epoch != value.erase_epoch and not self._intent_exact(
                        scope, value, planned):
                    raise QuoteMemError("operation_conflict",
                                        "The scope changed since the selection")
            else:
                item, head = self._current(scope, memory_id)
                if (
                    item["current_revision_id"] != expected_revision_id
                    or head["content_hash"] != expected_content_hash
                    or self._bind_selection(scope, grant.principal_id, session, item, head,
                                            value.erase_epoch, seed=seed) != value.selection_digest
                ):
                    raise QuoteMemError("operation_conflict",
                                        "The memory changed since it was selected")
                planned = self._admission_plan(scope, self._open_txn(scope, None, now), value, head)
                if erase_epoch != value.erase_epoch and not self._intent_exact(
                        scope, value, planned):
                    raise QuoteMemError("operation_conflict",
                                        "The memory changed since it was selected")
                if value.listed is not None and not set(planned.chain) <= set(value.listed):
                    raise QuoteMemError("operation_conflict",
                                        "The forget would take a memory its prepare did not name")
            chain, replacers, reworded = planned.chain, planned.replacers, planned.reworded
            versions, entity_ids, spans = planned.versions, planned.entity_ids, planned.spans
            sensitive, any_script = planned.sensitive, planned.any_script
            evidence = frozenset(span.evidence_id for span in spans)
            if (len(versions) > MAX_FORGET_TOMBSTONES or len(spans) > MAX_FORGET_SPANS
                    or len(entity_ids) > MAX_FORGET_ENTITIES
                    or len(evidence) > MAX_FORGET_SOURCES):
                raise QuoteMemError("budget_exceeded", "The forget plan exceeds its bounds")
            self._preflight(scope, _items(memory_id, chain), evidence, len(entity_ids), now)
            # A bundled memory's seed stands for its head's statement alone: the
            # Memory's earlier statements, its chain's and the rewordings keep their tombstones.
            tombstoned = (versions if seed is None or memory_id is None
                          else self._unseeded(versions, memory_id))
            # A replacer goes for the forgotten value it names: its own value gets no
            # tombstone and joins no forget shadow (``_chain_parts``).
            tombstoned = self._unreplaced(tombstoned, replacers)
            # A memory the forget rewords keeps its words found: its present wording is
            # tombstoned and joins no forget shadow (``ForgetRewordV1``).
            whole = {version.revision_id
                     for version in self._wording_versions(value.rewordings)}
            tombstones, created = self._tombstone_ids(scope, tombstoned)
            if seed is not None:
                tombstones, created = self._with_seed(scope, seed, tombstones, created)
            lease = writer.get_control("lease:writer")
            if lease is None:
                raise QuoteMemError("lease_lost", "Forget admission has no active Writer")
            epoch = lease["value"]["writer_epoch"]
            consent = writer.get_counter(f"consent_epoch:{scope.scope_id}")
            controller_id = self._write_controller(grant, epoch, now, intent)
            worker_id = new_ulid()
            worker = self._run_row(
                worker_id, "forget", "destructive_scan", grant, scope,
                {"target_kind": "memory", "target_id": memory_id} if memory_id is not None
                else {"target_kind": "scope", "target_id": scope.scope_id},
                epoch, now, intent, controller=controller_id, erase_epoch=erase_epoch,
                consent_epoch=consent,
            )
            fence = ErasureFenceV1(
                phase="admitting", scope_id=scope.scope_id, controller_run_id=controller_id,
                worker_run_id=worker_id, memory_id=memory_id, chain_ids=chain,
                reworded_ids=reworded, replacer_ids=replacers,
                intent_key=intent,
                spans=tuple(spans), tombstone_ids=tuple(sorted(tombstones.values())),
                created_tombstone_ids=tuple(sorted(created)), entity_ids=tuple(entity_ids),
                erase_epoch=erase_epoch,
                tombstone_seq=writer.get_counter(f"tombstone_seq:{scope.scope_id}"),
                scope_seq=writer.get_counter(f"scope_seq:{scope.scope_id}"),
                commit_seq=writer.get_counter("commit_seq"), counts=ForgetCountsV1(),
                cancelled_owner_run_id=None if live is None else live[1]["object_id"],
                trim_dependents=not sensitive, seed=seed, any_script=any_script,
                rewordings=value.rewordings, name_asked=value.name_asked, asked=value.asked,
            )
            # The fence carries the seed now; the consumed intent keeps only its digest
            # (``selection_digest``), so no second copy of the forgotten words remains.
            consumed = value.model_copy(update={
                "state": "consumed", "controller_run_id": controller_id, "seed": None,
                "rewordings": (), "asked": (),
            })
            rows = [
                RowWrite("run", writer.parse_row("run", worker)),
                RowWrite("control", writer.materialize_control(
                    intent, ForgetIntentV1.model_validate(consumed.model_dump()).model_dump(
                        mode="json"), stored["version"] + 1, now, controller_id)),
                RowWrite("control", writer.materialize_control(
                    fence_key(scope.scope_id, controller_id), fence.model_dump(mode="json"), 0,
                    now, controller_id)),
                *([] if live is None else self._cancellation(*live, controller_id, now)),
                *self._shadow(scope, [version for version in tombstoned
                                      if version.revision_id not in whole], now, controller_id,
                              seed=shadowed, left=[
                                  form for version in tombstoned
                                  if version.revision_id in whole
                                  for wording in value.rewordings
                                  if wording.base_digest == version.snapshot.statement_digest
                                  for form in left_out(version.snapshot.statement,
                                                       wording.statement)],
                              enforces=value.enforces, asked=value.asked),
            ]
            _, plan = writer.plan_rows(rows, None)
            writer.write_plan(plan)
            self._publisher.publish_erasure(writer, scope, None, writer.get_counter("commit_seq"),
                                            active=True)
            if live is not None:
                self._retire(scope, live[1]["object_id"])
                # As a release does: the in-memory admission order no longer holds
                # the cancelled owner as running, so a waiting Add meets the
                # erasure fence now and the next cycle admits once it is gone.
                writer.add_admission.owner_released(scope.scope_id)
        self._checkpoint("admitted_fence")
        return controller_id

    def _restating(
        self, scope: ScopeV1, spans: Sequence[QuoteSpanV1],
    ) -> tuple[QuoteSpanV1, ...]:
        """The restating sentences an intent carries: unique, ordered, each inside an active
        Evidence row of the scope (``reference_missing`` otherwise)."""
        found: dict[tuple[str, int, int], QuoteSpanV1] = {}
        for span in spans:
            row = self._writer.get_row("evidence", span.evidence_id)
            if row is None or (row["scope_kind"], row["scope_id"]) != (
                    scope.scope_kind, scope.scope_id) or row["state"] != "active" or not (
                    row["byte_start"] <= span.byte_start < span.byte_end
                    <= row["byte_start"] + len(row["text"].encode("utf-8"))):
                raise QuoteMemError("reference_missing",
                                    "A restating span lies in no Evidence here")
            found[span.evidence_id, span.byte_start, span.byte_end] = span
        return tuple(found[key] for key in sorted(found))

    def _shadow(
        self, scope: ScopeV1, versions: Sequence[ForgetVersionSelection], now: int,
        controller_id: str, *, seed: ForgetSeedV1 | None = None, enforces: int | None = None,
        left: Sequence[str] = (), asked: Sequence[str] = (),
    ) -> list[RowWrite]:
        """The scope's forget shadow grown by the tombstoned versions (``forget_shadow:{s}``).

        Each version's statement and keywords give their name terms and topic keywords
        (``recall_agent.shadow_terms``), stored as keyed digests only. The interactive
        Search withholds what they name; nothing else reads them.

        A forget of sentences shadows no version (it tombstones only the
        wordings it rewords): its ``seed``, the user's words of the fact, joins instead, so the
        interactive Search still withholds a restatement a later Add stores; one withholding
        only forget requests' own sentences passes no ``seed`` here (`admit`). Only a complete
        forget (``explicit-request-v2``, the ``agent`` Add mode) forgets sentences alone, so
        no benchmark workspace gains a shadow this way.

        A memory the forget rewords keeps its words found; only the names and values its new
        wording leaves out (``left``, "Henrik", "林晓雨") join the shadow, a place's by what
        names it (`named_part`: "Qiming" of "Qiming Kindergarten"): the word ending it names
        nothing, and a kept name may end with it ("Lanwan Kindergarten").

        A statement naming the value the user asked to forget beside another (``asked``, the
        keyed digests of the asked values' names) gives only the asked names: "The user moved
        from Hangzhou to Shanghai." forgotten for "forget that I lived in Shanghai" hides no
        memory of Hangzhou, the city the user kept.
        """
        terms = ForgetShadowTerms()
        wanted = set(asked)
        for version in versions:
            found = shadow_terms(version.snapshot.statement, version.snapshot.keywords)
            if wanted:
                own = {name for name in found.names
                       if self._writer.bind_content(SHADOW_PURPOSE, name) in wanted}
                if own and own != set(found.names):
                    found = ForgetShadowTerms(frozenset(own), found.topics)
            terms = terms | found
        if seed is not None:
            terms = terms | shadow_terms(seed.statement)
        for form in left:
            terms = terms | shadow_terms(f"the name {named_part(form)}")
        return self._shadow_rows(scope, terms, now, controller_id, enforces=enforces)

    def record_shadow(self, scope: ScopeV1, texts: Sequence[str]) -> None:
        """Grow the scope's forget shadow by what forgotten texts name (``forget_shadow:{s}``).

        A complete in-chat forget (``explicit-request-v2``) withholds the request sentences
        with the fact, so the interactive Search can no longer read what they asked to
        forget from Evidence; their statements are recorded here instead, as keyed digests.
        """
        terms = ForgetShadowTerms()
        for text in texts:
            terms = terms | shadow_terms(text)
        writer = self._writer
        with writer.write_scope():
            rows = self._shadow_rows(scope, terms, self._clock(), None)
            if rows:
                _, plan = writer.plan_rows(rows, None)
                writer.write_plan(plan)

    def restamp_names(self, scope: ScopeV1, *, since: int, at: int) -> int:
        """Stamp with ``at`` every shadow name a complete forget stamped at ``since`` or later;
        how many were.

        A rebuild replays a forget the old store admitted at ``at`` (``quotemem.agent.rebuild``),
        so the new store stamps its names with the rebuild's clock, while the messages it
        replays keep their dates. A name is lifted only by a restatement dated after its
        stamp (``core/relearn.after``), so the user's restatement after the forget would stay
        hidden. Stamped back with the forget's own time, each name stands where the old store
        put it. Topic keywords keep their stamps: they also say which hits the forget read
        (``ForgetShadow.topic_seen``), on the new store's clock.
        """
        writer = self._writer
        key = f"forget_shadow:{scope.scope_id}"
        with writer.write_scope():
            stored = writer.get_control(key)
            if stored is None:
                return 0
            current = ForgetShadowV1.model_validate(stored["value"])
            changed = [when >= since for when in current.name_times]
            if not any(changed):
                return 0
            value = current.model_copy(update={"name_times": [
                at if moved else when for when, moved in zip(
                    current.name_times, changed, strict=True)]})
            rows = [RowWrite("control", writer.materialize_control(
                key, value.model_dump(mode="json"), stored["version"] + 1, self._clock(),
                None))]
            _, plan = writer.plan_rows(rows, None)
            writer.write_plan(plan)
        return sum(changed)

    def _shadow_rows(
        self, scope: ScopeV1, terms: ForgetShadowTerms, now: int, controller_id: str | None,
        *, enforces: int | None = None,
    ) -> list[RowWrite]:
        """The shadow Control holding its terms and ``terms``, as digests; none when unchanged.

        A full shadow keeps what it holds and takes no more. A complete forget (the
        ``agent`` Add mode) also stamps each term it adds, or adds again, with ``now``
        (``name_times``, ``topic_times``), so the interactive Search can tell a later
        restatement by the user from what the forget left. A forget that
        ``enforces`` an earlier one (the relearn check withholding an assistant echo of a
        forgotten fact) forgets nothing new: a term already held keeps its stamp, and a term
        it adds takes the enforced forget's time, so the user's restatement after that
        forget is still found.
        """
        writer = self._writer

        def bind(term: str) -> str:
            return writer.bind_content(SHADOW_PURPOSE, term)

        key = f"forget_shadow:{scope.scope_id}"
        stored = writer.get_control(key)
        current = None if stored is None else ForgetShadowV1.model_validate(stored["value"])
        old_names: list[str] = [] if current is None else list(current.names)
        old_topics: list[tuple[str, ...]] = [] if current is None else [
            tuple(topic) for topic in current.topics]
        stamped = self._complete()
        name_at: dict[str, int] = {} if current is None or not current.name_times else dict(
            zip(old_names, current.name_times, strict=True))
        topic_at: dict[tuple[str, ...], int] = {} if current is None or not (
            current.topic_times) else dict(zip(old_topics, current.topic_times, strict=True))
        new_names = sorted(map(bind, terms.names))
        new_topics = [tuple(sorted(set(map(bind, topic)))) for topic in terms.topics
                      if len(topic) <= MAX_SHADOW_TOPIC_TERMS]
        names: list[str] = list(dict.fromkeys(
            [*old_names, *new_names]))[:MAX_FORGET_SHADOW_TERMS]
        topics: list[tuple[str, ...]] = list(dict.fromkeys([
            *old_topics, *new_topics]))[:MAX_FORGET_SHADOW_TERMS]
        # A new forget stamps what it adds now; one enforcing an earlier forget only stamps
        # the terms the shadow lacks, with that forget's time.
        at = now if enforces is None else enforces
        if stamped:
            for name in new_names:
                if enforces is None or name not in name_at:
                    name_at[name] = at
            for topic in new_topics:
                if enforces is None or topic not in topic_at:
                    topic_at[topic] = at
        if (not names and not topics) or (
                (set(names), set(topics)) == (set(old_names), set(old_topics))
                and (not stamped or enforces is not None or not (new_names or new_topics))):
            return []
        ordered_names = sorted(names)
        ordered_topics = sorted(list(topic) for topic in topics)
        body: dict[str, object] = {"names": ordered_names, "topics": ordered_topics}
        if stamped:
            body["name_times"] = [name_at.get(name, at) for name in ordered_names]
            body["topic_times"] = [topic_at.get(tuple(topic), at) for topic in ordered_topics]
        value = ForgetShadowV1.model_validate(body)
        return [RowWrite("control", writer.materialize_control(
            key, value.model_dump(mode="json"), 0 if stored is None else stored["version"] + 1,
            now, controller_id))]

    def _write_controller(self, grant: PrincipalGrant, epoch: int, now: int, intent: str) -> str:
        """Write the intent's global erasure controller before anything names it.

        The admission transaction's Controls name the controller as their
        writer, and the Writer accepts only an existing Run there, so the
        controller is its own first write, under the same Writer mutex. It holds
        no fence and changes nothing a reader of the scope sees. Only a crash
        separates it from the admission transaction (a failure after a write
        fails the read model closed); one controller exists per intent (its
        admission family), so that intent is spent and the client prepares again.
        The next boot ends such a controller (`abandon_orphan_controllers`).

        Raises:
            QuoteMemError: ``operation_conflict`` when the intent already has a controller.
        """
        writer = self._writer
        row = self._run_row(new_ulid(), "erasure_controller", "orchestrator", grant, None,
                            {"target_kind": "none", "target_id": None}, epoch, now, intent)
        if writer.indexed_reader().latest_run(
            "erasure_controller", row["idempotency_key"],
        ) is not None:
            raise QuoteMemError("operation_conflict", "The forget confirmation is not current")
        _, plan = writer.plan_rows(
            [RowWrite("run", writer.parse_row("run", row))], None,
        )
        writer.write_plan(plan)
        return cast(str, row["object_id"])

    # Resume ---------------------------------------------------------------------------
    def resume(self, scope: ScopeV1, controller_id: str) -> ForgetOutcome:
        """Continue a forget from its durable fence phase; committed writes never repeat.

        A forget whose fence is gone completed: its outcome is read back from the
        result index. It runs in the scope's erasure turn (`quotemem.store.erasure_gate`).
        """
        with self._turn(scope, user_request()):
            return self._resume(scope, controller_id)

    def _resume(self, scope: ScopeV1, controller_id: str) -> ForgetOutcome:
        while True:
            control = self._writer.get_control(fence_key(scope.scope_id, controller_id))
            if control is None:
                return self._outcome(controller_id)
            fence = ErasureFenceV1.model_validate(control["value"])
            step: dict[ErasurePhase, Callable[[ScopeV1, ErasureFenceV1, Row], None]] = {
                "admitting": self._apply_effects,
                "admitted": self._delete,
                "deleted": self._purge,
                "purged": self._scan,
                "scanned": self._write_receipts,
                "receipted": self._release,
            }
            step[fence.phase](scope, fence, control)
            self._checkpoint(fence.phase)

    # Admit, part (b) ---------------------------------------------------------------------------
    def _apply_effects(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        writer = self._writer
        ticket = writer.issue_ticket(fence.worker_run_id)
        now = self._clock()
        txn = self._open_txn(scope, fence.worker_run_id, now)
        versions = [row for row in self._versions(fence)
                    if row["memory_id"] not in fence.replacer_ids]
        written = 0
        created = list(fence.created_tombstone_ids)
        if fence.seed is not None:
            # The seed stands for the head's statement (a forget of sentences
            # has no version); a new seed tombstone took the last created id (`_with_seed`).
            stands: set[str] = (set() if fence.memory_id is None
                                else _seed_stands_for(versions, fence.memory_id))
            versions = [row for row in versions if row["object_id"] not in stands]
            pairs = {(row["snapshot"]["entity_id"], row["snapshot"]["statement_digest"])
                     for row in versions}
            if (fence.seed.entity_id, statement_digest(fence.seed.statement)) not in pairs and (
                    self._seed_stored(scope, fence.seed) is None):
                txn.put("tombstone", self._seed_row(scope, fence, now, created[-1]))
                created.pop()
                written += 1
        for version, tombstone_id in self._creations(scope, versions, created):
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
                "tombstone_seq": writer.get_counter(f"tombstone_seq:{scope.scope_id}") + 1,
                "forget_run_id": fence.worker_run_id, "created_at": now,
            }).model_dump(mode="json"))
            written += 1
        first = fence.spans[:_SPAN_BATCH]
        touched = withhold_spans(txn, first, reason="forgotten", run_id=fence.worker_run_id)
        counts = fence.counts.model_copy(update={
            "tombstones_written": written,
            "evidence_spans_withheld": _count_newly_withheld(first, touched),
            "evidence_rows_withheld": len(touched),
        })
        self._commit(scope, ticket, txn, fence, control, "admitted", counts, tombstones=True,
                     spans_done=len(first))

    def _seed_row(
        self, scope: ScopeV1, fence: ErasureFenceV1, now: int, tombstone_id: str,
    ) -> dict[str, Any]:
        """The tombstone row of a forget's seed (``ForgetSeedV1``): a forget of sentences', or
        a bundled memory's, under the head's primary entity."""
        seed = fence.seed
        assert seed is not None
        return TombstoneRowV1.model_validate({
            "object_id": tombstone_id, "scope_kind": scope.scope_kind,
            "scope_id": scope.scope_id, "entity_id": seed.entity_id, "about_entity_ids": [],
            "statement": seed.statement, "statement_digest": statement_digest(seed.statement),
            "kind": "state", "valid_from": None, "valid_to": None,
            "time_granularity": "unanchored", "embedding": list(seed.embedding),
            "embedding_space": seed.embedding_space,
            "source_evidence_ids": sorted(fence.evidence_ids),
            "tombstone_seq": self._writer.get_counter(f"tombstone_seq:{scope.scope_id}") + 1,
            "forget_run_id": fence.worker_run_id, "created_at": now,
        }).model_dump(mode="json")

    # Delete -------------------------------------------------------------------------------
    def _delete(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        """One transaction: a bounded batch, or the retention unit's own rows.

        The spans of S are withheld first; then the cohort of an Add owner ``admit``
        cancelled is settled (``_settle_cohort``); then the dependents are compensated
        a batch at a time after the fence's cursor; then the ``supersedes`` and
        ``contradicts`` links other memories carry to the unit's memories leave a batch
        at a time, and the pending candidates S leaves without a usable group
        expire a batch at a time, in id order after their own cursor. Once none
        is left, one transaction deletes the unit's own rows, which ``admit`` proved fit
        one transaction (``_preflight``), and freezes the Artifacts, Controls and
        deliveries the following batches delete (``_purge``). A refused dependents batch
        never ends the forget: it goes on without trims (``_settle_untrimmed``).
        """
        writer = self._writer
        ticket = writer.issue_ticket(fence.worker_run_id)
        txn = self._open_txn(scope, fence.worker_run_id, self._clock())
        done = fence.spans_done
        if done < len(fence.spans):
            batch = fence.spans[done:done + _SPAN_BATCH]
            touched = withhold_spans(txn, batch, reason="forgotten", run_id=fence.worker_run_id)
            counts = fence.counts.model_copy(update={
                "evidence_spans_withheld": fence.counts.evidence_spans_withheld
                + _count_newly_withheld(batch, touched),
                "evidence_rows_withheld": fence.counts.evidence_rows_withheld + len(touched),
            })
            self._advance(scope, ticket, txn, fence, control, counts,
                          spans_done=done + len(batch))
            return
        prefilled = self._prefilled.pop(fence.controller_run_id, None)
        if prefilled and self._keep_prefilled(txn, prefilled):
            self._advance(scope, ticket, txn, fence, control, fence.counts)
            return
        if fence.cancelled_owner_run_id is not None and self._settle_cohort(
            scope, ticket, txn, fence, control,
        ):
            return
        unit = self._unit(txn, _items(fence.memory_id, fence.chain_ids))
        statements = self._tombstone_statements(txn, fence)
        dependents = self._dependents(scope, txn, fence, unit=set(unit), statements=statements,
                                      after=fence.dependents_after, limit=_DEPENDENT_BATCH)
        if dependents:
            try:
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
            except Exception as error:
                # The fence is written with the batch: unchanged, nothing of it was. A
                # transient refusal leaves the batch to a later resume, which may trim it.
                if not fence.trim_dependents or not _forget_alone(error) or _transient(error) or (
                        writer.get_control(cast(str, control["key"])) != control):
                    raise
                self._settle_untrimmed(scope, ticket, fence, control, statements, error)
            return
        if fence.rewordings and self._reword_planned(txn, fence, statements):
            self._advance(scope, ticket, txn, fence, control, fence.counts.model_copy(update={
                "dependent_requoted": fence.counts.dependent_requoted + 1}))
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
                raise QuoteMemError("budget_exceeded",
                                    "The forget's retention records exceed bounds")
        erased = erase_unanchored_entities(
            txn, fence.entity_ids, participant_has_source=participant_has_source(txn),
        )
        counts = fence.counts.model_copy(update={
            **removed, "link": fence.counts.link + removed["link"], "entity_erased": len(erased),
        })
        self._commit(scope, ticket, txn, fence, control, "deleted", counts,
                     artifacts=artifacts, controls=controls, recalls=recalls)

    def _settle_untrimmed(
        self, scope: ScopeV1, ticket: WriteTicket, fence: ErasureFenceV1, control: Row,
        statements: Sequence[str], error: Exception,
    ) -> None:
        """Go on past a dependents batch refused before any write: deterministic (never
        `_transient`), it would be refused again at every resume and the fence would hold the
        scope forever.

        The planned rewordings are written first, one per call, since no read may tell an old
        wording once the fence is gone; then the fence allows no trim any more. The
        unsettled dependents stay unreadable because deletion and the final scan use this fence."""
        logging.getLogger(__name__).warning(
            "A forget's dependents batch was refused: %s",
            canonical_json(failure_reason("forget_dependents", error).to_dict()),
        )
        txn = self._open_txn(scope, fence.worker_run_id, self._clock())
        if fence.rewordings and self._reword_planned(txn, fence, statements):
            self._advance(scope, ticket, txn, fence, control, fence.counts.model_copy(update={
                "dependent_requoted": fence.counts.dependent_requoted + 1}))
            return
        self._write_progress(scope, fence, control, "admitted", fence.counts, trim_dependents=False)

    def _settle_cohort(
        self, scope: ScopeV1, ticket: WriteTicket, txn: TxnRows, fence: ErasureFenceV1,
        control: Row,
    ) -> bool:
        """One batch settling the cancelled owner's cohort; False once settled.

        First the cancelled owner's unended child Runs fail ``erase_fenced``, a
        batch at a time; then its candidate rows that are not formal (every
        status but ``published``) leave, a batch at a time, so the next owner
        that reprocesses its requests finds each position once.
        Formal rows stay: revisions reference them, and the next owner
        re-derives their positions as unchanged. Re-entered rows of earlier
        owners keep theirs. The cancelled owner's proposals no remaining row
        cites leave with the memory's own rows (``_orphan_proposals``); its judge
        outputs, with the result indexes and snapshots that hold them, stay
        reusable by digest.
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
        """The ``evidence_hold:{s}`` post-image keeping the batch's deleted holds.

        A pending row with ``hold_evidence`` keeps its Evidence out of Evidence
        routes. Deleting the row must not release that Evidence before the owner
        that reprocesses its request settles it, so the hold moves to the scope's
        ``evidence_hold`` Control in the deleting transaction, under the request
        the row is attributed to: the newest among its sources. The
        reprocessing owner's release removes it (``owner_lease.release_finalize``).

        Raises:
            QuoteMemError: ``reference_missing`` for a held row without its Evidence;
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
                raise QuoteMemError("binding_mismatch",
                                    "A scope's Evidence hold has another variant")
            requests = {request.request_key: set(request.evidence_ids)
                        for request in current.requests}
        for row in held:
            sources = [writer.get_row("evidence", evidence_id)
                       for evidence_id in row["evidence_ids"]]
            present = [source for source in sources if source is not None]
            if len(present) != len(sources) or not present:
                raise QuoteMemError("reference_missing", "A held row lost its Evidence")
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
            raise QuoteMemError("budget_exceeded",
                                "A scope's Evidence holds exceed bounds") from error
        return RowWrite("control", writer.materialize_control(
            key, value.model_dump(mode="json"), 0 if stored is None else stored["version"] + 1,
            txn.now, ticket.run_id,
        ))

    def _end_children(self, fence: ErasureFenceV1, control: Row, owner: str) -> bool:
        """Fail up to ``_COHORT_BATCH`` unended children of the cancelled owner, read under
        the Writer, with the fence's progress; False when none is left."""
        writer = self._writer
        with writer.write_scope():
            if writer.get_control(cast(str, control["key"])) != control:
                raise QuoteMemError("binding_mismatch", "The erasure fence changed under its owner")
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
            rows = [RowWrite("run", writer.parse_row("run", _cancelled_post_image(child, now)))
                    for child in children[:_COHORT_BATCH]]
            rows.append(self._fence_row(fence, control, fence.phase, fence.counts, now))
            _, plan = writer.plan_rows(rows, None)
            writer.write_plan(plan)
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
        """The cancelled owner's proposals no candidate row cites once ``txn`` commits.

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

    def _keep_prefilled(self, txn: TxnRows,
                        prefilled: Mapping[str, tuple[str, list[float], str]]) -> bool:
        """The step once S is withheld: keep each ``prefilled`` vector (``_prefill``) on its
        part's withheld ranges (``ForgetRemainderV1``) where the part's readable text is still
        exactly the one embedded, no byte of it withheld since and no remainder kept yet; a
        write only, no call. The interactive Search then finds the part's readable bytes by
        that vector and their own terms, never by the stored vector, which encodes every byte
        (``p_evr``): "Where do I live now?" finds "... moved to Xiamen, and I live in Xiamen now."
        once "from Chengdu" is withheld. Whether any was put in ``txn``."""
        kept = False
        for evidence_id, (text, vector, space) in sorted(prefilled.items()):
            row = txn.get("evidence", evidence_id)
            if row is None or row["state"] != "active" or not row["withheld_spans"] or any(
                    span["reason"] not in FORGET_REASONS for span in row["withheld_spans"]) or (
                    stored_remainder(row) is not None) or (
                    space != row["embedding_space"] or len(vector) != len(row["embedding"])):
                continue
            spans = [{key: value for key, value in span.items() if key != "remainder"}
                     for span in row["withheld_spans"]]
            if remainder_text(row["text"], row["byte_start"], [
                    (span["byte_start"], span["byte_end"]) for span in spans]) != text:
                continue
            spans[-1] = {**spans[-1], "remainder": {
                "ranges": [[span["byte_start"], span["byte_end"]] for span in spans],
                "embedding": vector, "embedding_space": space}}
            txn.put("evidence", {**row, "withheld_spans": spans})
            kept = True
        return kept

    def _remainder_parts(self, txn: TxnRows) -> list[tuple[Row, str]]:
        """The scope's active Evidence parts withheld only by forgets that keep no current
        remainder and still have readable words, with their readable text
        (``forget_remainder.remainder_text``), ``MAX_REMAINDERS`` at most, in id order."""
        found: list[tuple[Row, str]] = []
        for row in self._pages(EvidenceRows(scope_kind=txn.scope_kind, scope_id=txn.scope_id)):
            spans = row["withheld_spans"]
            if row["state"] != "active" or not spans or any(
                    span["reason"] not in FORGET_REASONS for span in spans) or (
                    stored_remainder(row) is not None):
                continue
            text = remainder_text(row["text"], row["byte_start"],
                                  [(span["byte_start"], span["byte_end"]) for span in spans])
            if any(character.isalnum() for character in text):
                found.append((row, text))
            if len(found) >= MAX_REMAINDERS:
                break
        return found

    def _advance(
        self, scope: ScopeV1, ticket: WriteTicket, txn: TxnRows, fence: ErasureFenceV1,
        control: Row, counts: ForgetCountsV1, **update: Any,
    ) -> None:
        """Commit one batch's rows with the fence's progress, or the progress alone."""
        if txn.written or txn.deleted:
            self._commit(scope, ticket, txn, fence, control, "admitted", counts, **update)
        else:
            self._write_progress(scope, fence, control, "admitted", counts, **update)

    def _dependents(
        self, scope: ScopeV1, txn: TxnRows, fence: ErasureFenceV1, *,
        statements: Sequence[str], unit: Set[str] = frozenset(), after: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Surviving memories ``delete`` settles, in id order after ``after``: those whose head
        cites S, and those whose head carries a keyword the forget drops.

        ``delete`` settles them ``limit`` at a time and moves the fence's cursor
        past each committed batch, so every dependent is settled once,
        however many there are; the final scan visits them all. The
        retention unit's memories are deleted, not compensated: only surviving
        memories are dependents, and their own rows stay as ``admit`` counted.
        """
        evidence = fence.evidence_ids
        reader = self._writer.indexed_reader()
        query = MemoryTableRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id)
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
        """One surviving memory's ``delete`` effect; True when it wrote a revision.

        A head citing S is compensated (``forget_dependent``); then a head that
        passes ``p_ver`` and still carries a keyword the forget drops gets a keyword-only
        revision. A compensating requote already drops those keywords, so an
        memory gets at most one revision.
        """
        item = txn.get("item", memory_id)
        head = None if item is None else txn.get("revision", item["current_revision_id"])
        compensated = head is not None and bool(
            fence.evidence_ids.intersection(head["evidence_ids"])) and self._compensate(
            scope, txn, read, memory_id, forgotten, statements,
            {wording.memory_id: wording for wording in fence.rewordings})
        return self._rekey(txn, read, memory_id, statements) or compensated

    def _rekey(
        self, txn: TxnRows, read: Readability, memory_id: str, statements: Sequence[str],
    ) -> bool:
        """A readable head drops every keyword carrying a forgotten word that its
        statement and readable quotes do not show.

        A head a ``corrects_history`` row targets keeps its keywords: the row targets that
        revision, so a new head would no longer be hidden by the correction.
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
        """The statements of the forget's tombstones: what the keywords may not
        carry, whatever the forgotten versions' sensitivity. The present wording of a memory
        the forget rewords is tombstoned only so no read shows it again: what it says is not
        forgotten (`_wording_versions`)."""
        found: list[str] = []
        kept = _wording_digests(fence)
        for tombstone_id in fence.tombstone_ids:
            tombstone = txn.get("tombstone", tombstone_id)
            if tombstone is not None and tombstone["statement"] and (
                    tombstone["statement_digest"] not in kept):
                found.append(tombstone["statement"])
        return tuple(found)

    def _compensate(
        self, scope: ScopeV1, txn: TxnRows, read: Readability, memory_id: str,
        forgotten: _Forgotten | None, statements: Sequence[str],
        rewordings: Mapping[str, ForgetRewordV1] | None = None,
    ) -> bool:
        """The ``forget_dependent`` compensation for one dependent: keep, requote or leave
        unreadable.

        A dependent the compensation choice leaves unreadable is requoted on the readable parts of
        its head's groups when ``trimmed_support`` proves they carry it;
        ``forgotten`` is None when the fence allows no trim. A requote keeps the
        version's keywords less those the forget drops. One the forget rewords
        (``rewordings``, a complete forget's) still stating its planned wording is left as it
        is: its rewording follows (`_reword_planned`).

        A head a ``corrects_history`` row targets is never rewritten, nor is a version
        such a row targets requoted: the row targets that revision, so the new head would
        no longer be hidden by the correction. Nor is a head trimmed that may
        restate a forgotten possession (``possession_entailed``), nor one still showing the
        colour or breed a forget of that detail withholds (`_detail_dependent`).
        """
        item = txn.get("item", memory_id)
        if item is None or txn.links_to("corrects_history", item["current_revision_id"]):
            return False
        wording = (rewordings or {}).get(memory_id)
        head_row = txn.get("revision", item["current_revision_id"])
        if wording is not None and head_row is not None and (
                head_row["snapshot"]["statement_digest"] == wording.base_digest
                != statement_digest(wording.statement)):
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
            forgotten = _detail_dependent(forgotten, snapshot.statement)
            if forgotten is None:
                return False
            if tombstoned(snapshot.entity_id, snapshot.statement_digest) or _trim_refused(
                    txn, snapshot, forgotten):
                return False
            trimmed = _trimmed_dependent(txn, read, snapshot, forgotten)
            if trimmed is None:
                groups = _bare_name_support(txn, read, snapshot, forgotten.words)
                if not groups:
                    return False
                trimmed = TrimmedSupport(groups, tuple(snapshot.keywords))
            # As `_rekey` reads them: a memory gets one revision a batch.
            quotes = [text for group in trimmed.groups for span in group
                      if (text := _quoted_text(txn, span)) is not None]
            dropped = forgotten_keywords(snapshot.statement, trimmed.keywords,
                                         statements, quotes)
            requote_on(txn, memory_id, version=head.row, support=trimmed.groups,
                       keywords=[word for word in trimmed.keywords if word not in dropped],
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
        # requote_on also re-derives the memory's names.
        requote_on(txn, memory_id, version=version, support=support,
                   keywords=[word for word in snapshot.keywords if word not in dropped],
                   policy_release_ref=self._policy_release_ref or version["policy_release_ref"])
        return True

    def _reword_planned(self, txn: TxnRows, fence: ErasureFenceV1,
                        statements: Sequence[str]) -> bool:
        """Reword one memory the forget planned to reword (``ErasureFenceV1.rewordings``) that
        still states its present wording; True when one was written (one per batch).

        Admission tombstoned that wording (`_wording_versions`), so it is never told again; the new
        statement becomes the memory's next head on the support the prepare planned, which
        none of the plan's forgets withholds, once `_admit_wording` admits it. Otherwise the
        memory stays as the forget's withholding leaves it."""
        forgotten = self._forgotten_words(txn, fence)
        if forgotten is None:
            return False
        read = readability(txn)
        for wording in fence.rewordings:
            item = txn.get("item", wording.memory_id)
            head = None if item is None else txn.get("revision", item["current_revision_id"])
            if item is None or head is None or item["lifecycle_state"] != "active" or (
                    head["snapshot"]["statement_digest"] != wording.base_digest) or (
                    wording.base_digest == statement_digest(wording.statement)):
                continue
            keywords = self._admit_wording(txn, read, wording, forgotten, statements)
            if keywords is None:
                continue
            requote_on(txn, wording.memory_id, version=head,
                       support=[list(group) for group in wording.support],
                       keywords=list(keywords), wording=wording,
                       policy_release_ref=self._policy_release_ref or head["policy_release_ref"])
            return True
        return False

    def _admit_wording(
        self, txn: TxnRows, read: Readability, wording: ForgetRewordV1,
        forgotten: _Forgotten, statements: Sequence[str],
    ) -> tuple[str, ...] | None:
        """The keywords a memory reworded as ``wording`` keeps when the rewording may be
        written now, else None. The confirmation's forget checks it as the
        prepare did (``admit_rewordings``), against what it withholds and what its tombstones
        say, which the prepare's read of every forget of the plan contains:

        * the memory's head still states what the rewording was planned from, no
          ``corrects_history`` row targets it, it has no sensitivity class and the new
          statement's vector is in its embedding space;
        * every span of the support is readable and lies in a quote of one of its versions;
        * the new statement shows no word of what the forget withholds or of what its
          tombstones say of it (``_forgotten_terms``), but the tie it names the bearer by,
          which no forgotten statement but one tying the name states ("Sofia is the user's
          wife." for "the user's wife"); without a tie it restates no forgotten possession;
        * it keeps the statement's predicate and other arguments (``keeps_predicate``);
        * no text of the support shows a forgotten word or a form of what the new statement
          leaves out (`left_out`, "Luo Xiaodong" beside a forgotten "罗晓东"), and each group
          carries what the new statement says besides the tie (`_carried`).

        Its keywords are the planned ones the head still has, less every one showing a
        forgotten word."""
        item = txn.get("item", wording.memory_id)
        head = None if item is None else txn.get("revision", item["current_revision_id"])
        if item is None or head is None or txn.links_to("corrects_history", head["object_id"]):
            return None
        snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
        if (snapshot.erased or snapshot.sensitivity_classes
                or snapshot.statement_digest != wording.base_digest
                or snapshot.embedding_space != wording.embedding_space
                or len(snapshot.embedding) != len(wording.embedding) or not wording.support):
            return None
        quotes = [span for revision in self._revisions(txn, wording.memory_id)
                  for span in MemorySnapshotV2.model_validate(revision["snapshot"]).quote_spans]
        if not all(read.p_span(span) and any(
                span.evidence_id == quote.evidence_id and quote.byte_start <= span.byte_start
                and span.byte_end <= quote.byte_end for quote in quotes)
                for group in wording.support for span in group):
            return None
        shown = shown_terms if forgotten.any_script else support_words
        words = self._wording_words(snapshot.statement, wording, forgotten)
        if words is None:
            return None
        # Besides what it leaves out by rule, it drops only words the forget withholds (a
        # lowercase name: "bob" of "The user took the user's cat bob to the vet yesterday."),
        # or a phrase naming forgotten values as replaced, whole (`_contrast_values`).
        kept = set(token_seq(wording.statement))
        tail = bool(_contrast_values(snapshot.statement, wording, forgotten))
        removed = [*left_out(snapshot.statement, wording.statement), *(
            token for token in token_seq(snapshot.statement)
            if token not in kept and (tail or shown(token) & words))]
        # A colour or a breed left out of the thing it describes keeps the
        # statement's predicate by the rule that wrote it (`attribute_reworded`).
        attributed = attribute_reworded(snapshot.statement, wording.statement)
        if shown(wording.statement) & words or (wording.tie is None and any(
                possession_entailed(statement, wording.statement, modifier=forgotten.modifier)
                for statement in forgotten.statements)) or (attributed is None and not (
                    keeps_predicate(snapshot.statement, wording.statement, wording.tie,
                                    removed))):
            return None
        probe = _probe(wording, snapshot.statement)
        forms = [value_pattern(form) for form in left_out(snapshot.statement, wording.statement)]
        # A group of the head's own, untouched, carries what the head said; the rewording
        # only stands a tie or a kind for what it leaves out (``keeps_predicate``).
        own = {frozenset((span.evidence_id, span.byte_start, span.byte_end)
                         for span in (snapshot.quote_spans[index] for index in group))
               for group in snapshot.support_groups}
        ever = " ".join(text for span in quotes if (text := _quoted_text(txn, span)) is not None)
        texts: list[str] = []
        for group in wording.support:
            quoted = [_quoted_text(txn, span) for span in group]
            if any(text is None or shown(text) & words or any(
                    form.search(text) for form in forms) or (
                        attributed is not None and quote_ranges(text, attributed[1], attributed[0]))
                    for text in quoted):
                return None
            whole = frozenset((span.evidence_id, span.byte_start, span.byte_end)
                              for span in group) in own
            if not whole and not _carried(
                    probe, [text for text in quoted if text is not None],
                    any_script=forgotten.any_script, given=snapshot.statement, ever=ever):
                return None
            texts += [text for text in quoted if text is not None]
        keywords = [word for word in wording.keywords
                    if word in snapshot.keywords and not shown(word) & words]
        dropped = forgotten_keywords(wording.statement, keywords, statements, texts)
        return tuple(word for word in keywords if word not in dropped)

    @staticmethod
    def _wording_words(statement: str, wording: ForgetRewordV1,
                       forgotten: _Forgotten) -> frozenset[str] | None:
        """What a memory stating ``statement`` reworded as ``wording`` may not show
        (``_forgotten_terms`` of both statements) less its tie's words; None when a forgotten
        statement that ties no name, nor asks to forget one ("forget my wife's name"), states
        the tie ("Sofia is my wife" forgotten as a fact, not as a name), whose words the tie
        would bring back. A request to forget a name forgets the name, which the withheld
        spans and the left-out forms hold, not its own words: "猫" of "请忘掉我的猫的名字" is
        the tie a rewording keeps ("The user has a cat." on "我的猫"), "女儿" of "忘掉我女儿学校
        的名字" what one of the daughter keeps."""
        shown = shown_terms if forgotten.any_script else support_words
        kept = {term for name in forgotten.kept
                for term in (name, *support_words(name), *fact_terms(name))}
        if contrast := _contrast_values(statement, wording, forgotten):
            # A rewording that only drops the phrases naming forgotten values as replaced (a
            # contrast tail, a clause saying one no longer holds, a place left for another)
            # states what holds now: of the forget, only the words of its names, values and
            # numbers are forgotten for it, never a word it shares with the old fact ("live"
            # of "The user used to live at 14 Oak Street." for "The user lives at 27 Pine Lane
            # now."), nor one of the value that holds now ("jonah" of a new
            # "jonah@new.example.com" for an old "jonah@old.example.com", "浙" of a new plate
            # "浙A9M105", "幼儿园" of a new "蓝湾幼儿园"), which shows none of the forgotten
            # values whole (`reworded_statement`). The values are those of the forgotten
            # statements written in English, which give their values whole ("The user works
            # at 青禾科技."): the user's own words of the request ("我原来的车牌") are a kind,
            # never a value ("车牌" of "车牌现在是浙A9M105").
            return _value_terms([*(told for told in forgotten.statements if english(told)),
                                 *contrast], shown) - kept - shown(wording.statement)
        tie: frozenset[str] = frozenset() if wording.tie is None else (
            shown(wording.tie) - {"user"})
        # Only a request for a name lets a tie stand, and then a forgotten statement showing
        # the tie's noun beside the asked name ("adopted a rabbit named Clover") vetoes none.
        if tie and not forgotten.name_asked and any(
                tie & shown(other) for other in forgotten.statements
                if not tie_descriptions([other]) and not asks_name(other)):
            return None
        # The words a name request describes the bearer by are no forgotten words either,
        # though a withheld echo of the name shows them ("猫" of "你的猫叫 Bob").
        asked = {term for other in forgotten.statements if asks_name(other)
                 for term in shown(other) if term in TIES}
        # So are the words a request for a detail names the thing by ("猫" of "我的猫是什么颜色
        # 的", "son" of "which school my son goes to"): it forgets the value.
        asked |= {term for other in forgotten.statements
                  if attribute(detail := detail_question(other))
                  for term in shown(other) if term in detail_words(detail, other)}
        # Nor is a stated colour or breed told by a wording that leaves out exactly it ("The
        # user's cat is named Bob." for "Bob is orange").
        attributed = attribute_reworded(statement, wording.statement)
        dropped = {value.casefold() for value in attributed[1]} if attributed else set[str]()

        def detailed(other: str) -> bool:
            detail = detail_request(other)
            return detail is not None and (not detail.claimed or (
                bool(dropped) and dropped <= set(detail.claimed)))

        if forgotten.name_asked or any(asks_name(other) or detailed(other)
                                       for other in forgotten.statements):
            # A request for a name forgets the name: of what the forget withholds, only the
            # words the old wording shows and the new one leaves out are forgotten ("bob" of
            # the assistant's withheld "好的，记住了，你的猫叫 Bob。", never its "猫叫").
            named = shown(statement) - shown(wording.statement)
            forgotten = replace(forgotten, words=forgotten.words & named,
                                terms=forgotten.terms & named)
            # The tie stands in either script: "奶奶" of a withheld nickname "兰兰奶奶" is the
            # tie "paternal grandmother" its rewording keeps on "我奶奶今年在学太极拳".
            covered = {kind for word in tie for kind in TIES.get(word, ())}
            tie |= {term for word, kinds in TIES.items()
                    if not word.isascii() and kinds & covered for term in shown(word)}
            # So does the kind standing for the place or organisation whose name it asks
            # ("an elementary school" of "Maple Grove Elementary", whose forgotten naming
            # "A school is named Maple Grove Elementary." shows "school" too).
            tie |= {term for value in left_out(statement, wording.statement)
                    if (kind := kind_of(value)) is not None for term in shown(kind)}
            # And the Han names the new wording keeps: "小" of a kept "小王" beside a forgotten
            # "小李" ("小李" itself stays forbidden).
            tie |= {term for term in shown(wording.statement) if not term.isascii()}
        forgotten = replace(forgotten, statements=tuple(
            other for other in forgotten.statements
            if not asks_name(other) and not detailed(other)))
        return (_forgotten_terms(forgotten, statement)
                | _forgotten_terms(forgotten, wording.statement)) - tie - asked - kept

    def _forgotten_words(self, txn: TxnRows, fence: ErasureFenceV1) -> _Forgotten | None:
        """What no trimmed dependent may show: ``support_words`` of every span
        of S, and the statements of the forget's tombstones (``predicate_words`` reads
        each against the dependent's own statement).

        Both are frozen by ``admit`` and retained, so ``delete`` and the scan read the same words.
        None when the fence allows no trim or a span or tombstone cannot be read: then no
        dependent is trimmed.
        """
        if not fence.trim_dependents:
            return None
        restating: Sequence[QuoteSpanV1] = ()
        modifier = False
        if fence.memory_id is not None:
            stored = self._writer.get_control(fence.intent_key)
            if stored is not None:
                intent = ForgetIntentV1.model_validate(stored["value"])
                restating, modifier = intent.restating_spans, intent.modifier_only
        words: set[str] = set()
        terms: set[str] = set()
        said: list[str] = []
        for span in fence.spans:
            if any(span.evidence_id == other.evidence_id and other.byte_start <= span.byte_start
                   and span.byte_end <= other.byte_end for other in restating):
                # A sentence withheld as restating the memory (an assistant's "Got it — Pixel,
                # your four-year-old tabby cat.") also says what a survivor's own clause says
                # ("tabby cat"); what it says of the forgotten fact is what the tombstones'
                # statements say of the survivor (``predicate_words``), forbidden below.
                continue
            text = _quoted_text(txn, span)
            if text is None:
                return None
            words |= support_words(text)
            terms |= fact_terms(text)
            said.append(text)
        statements: list[str] = []
        vectors: list[tuple[tuple[float, ...], str]] = []
        kept = _wording_digests(fence)
        for tombstone_id in fence.tombstone_ids:
            tombstone = txn.get("tombstone", tombstone_id)
            if tombstone is None:
                return None
            if tombstone["statement_digest"] in kept:
                # A wording tombstoned only so no read shows it again (`_wording_versions`).
                continue
            statements.append(tombstone["statement"])
            vectors.append((tuple(tombstone["embedding"]), tombstone["embedding_space"]))
        return _Forgotten(frozenset(words), tuple(statements), fence.any_script,
                          frozenset(terms), tuple(vectors), modifier=modifier,
                          name_asked=fence.name_asked,
                          kept=self._kept_names(statements, fence.asked), said=tuple(said))

    def _kept_names(self, statements: Sequence[str],
                    asked: Collection[str]) -> frozenset[str]:
        """The names of the ``statements`` the user kept: of each naming a value the user
        asked to forget (``asked``, keyed digests) beside another, the other names ("hangzhou"
        of "The user lives in Hangzhou, not Shanghai." for "forget that I lived in
        Shanghai"). No rewording reads them as forgotten."""
        if not asked:
            return frozenset()
        wanted = set(asked)
        kept: set[str] = set()
        for statement in statements:
            names = shadow_terms(statement).names
            own = {name for name in names
                   if self._writer.bind_content(SHADOW_PURPOSE, name) in wanted}
            if own:
                kept |= names - own
        return frozenset(kept)

    def _chain(self, txn: TxnRows, memory_id: str,
               restating: Sequence[QuoteSpanV1] = (), words: Sequence[str] = (),
               *, allowed: Set[str] | None = None) -> tuple[str, ...]:
        """The memories the corrections of ``memory_id`` refuted, transitively.

        A correction or retraction that changes the primary anchor creates a new
        memory, so the memories it ended by ``correct`` or ``retract`` are the
        forgotten memory's earlier versions, as a same-anchor correction's revised
        versions are. A correction the judge finds only contradicts a head,
        because it makes that head's description of an entity untrue, leaves the
        head current with a ``contradicts`` link: that head
        carries the description the correction refuted, as an ended version does,
        so the chain takes it too. Which of the two relations the judge chose must
        not decide what a forget leaves readable. A head that correction reworded
        no longer carries that description; it stays outside the chain as a
        reworded memory (``_reworded``). A ``contradicts`` link that a
        version written by no correction carries is a disagreement, not a
        refutation, and an ``update`` ends a state that was true in its time:
        neither target is in the chain; an updated target keeps its lifecycle
        (Effects) and loses only the forgotten memory's link. An update
        whose link records an explicit correction (the lean judge relates
        "<new>, not <old>" as ``update``) ended the value it refuted, so the chain
        takes its target as it would a ``correct`` one. Links on
        every revision count, since revise, merge and requote carry them; the
        selected head is CAS-bound and only an active memory gains links, so the
        chain cannot change between prepare and admission.

        A complete forget (``explicit-request-v2``) also takes every readable memory, current
        or past, that replaced by ``update`` or ``correct`` a memory the forget takes or
        leaves unreadable and whose statement still names a value of what is forgotten
        (``_replacing``): "The user goes climbing at a gym near West Lake instead of playing
        badminton." tells the badminton a forget of it is about. It goes on through the
        memories it takes, so a past replacer is found behind a current one, but a replacer
        naming only such a memory ("swimming instead of climbing") names nothing forgotten
        and stays. Such a memory can gain a link after the prepare; admission reads the chain again
        in its admission transaction and freezes it in the fence. ``restating`` are the
        sentences the forget also withholds (``prepare``'s); ``words`` those a bundled memory is
        forgotten for (``_chain_parts``).

        Returns:
            The chain's memories other than ``memory_id``, in id order.

        Raises:
            QuoteMemError: ``budget_exceeded`` past ``MAX_FORGET_CHAIN`` memories.
        """
        refuted, replacers = self._chain_parts(txn, memory_id, restating, words,
                                               allowed=allowed)
        return tuple(sorted({*refuted, *replacers}))

    def _links(self, txn: TxnRows) -> Callable[[str], Iterator[CorrectionEdge]]:
        """The correction edges of a memory's revisions that a forget follows (``_chain``)."""
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
                    # correction was requested (keeps that on the link itself).
                    yield CorrectionEdge(target, link["link_type"], link["relation"],
                                         bool(link["correction_requested"]))
        return links

    def _chain_parts(
        self, txn: TxnRows, memory_id: str, restating: Sequence[QuoteSpanV1] = (),
        words: Sequence[str] = (), *, allowed: Set[str] | None = None,
        nominated: set[str] | None = None, support_only: bool = False,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """``_chain`` in two parts: the memories refuted and the replacers (complete
        forgets, ``_replacing``). A replacer is forgotten for the forgotten value it names,
        not for its own one, so its versions get no tombstone (``ErasureFenceV1.replacer_ids``)
        and no forget shadow: a later memory naming only it ("swimming instead of climbing")
        stays found by Search and the memory list, its new value may be told again, and a
        memory sharing its other words ("after moving") may still be trimmed.

        ``words`` are the user's words a bundled memory is forgotten for (its
        seed's). A memory whose head names them as a value it replaced ("climbing ... instead
        of playing badminton", ``names_replaced``) goes for that value, not for its own new
        one: its replacers are read against the words and its parts naming a replaced value
        (``_bundle_values``), so "swimming instead of climbing" stays behind it, as it does
        behind the forget of the badminton memory by its id.

        A replacer an ``update`` or ``correct`` link ties to what is taken goes by the rule; an
        unlinked one is a nomination (``_replacing``): with ``allowed`` (a forget's admission
        and its prepare's final plan) only those the forget judge confirmed are taken; without
        (a prepare reading what to ask the judge) every nomination is, and each is added to
        ``nominated``."""
        links = self._links(txn)
        if not self._complete() or support_only:
            return correction_chain(memory_id, links), ()
        forgotten = {memory_id, *correction_chain(memory_id, links)}
        taken, replacers = self._replacing(
            txn, forgotten, self._restated_by(txn, memory_id, restating), links,
            values=self._bundle_values(txn, memory_id, forgotten, words), allowed=allowed,
            nominated=nominated)
        return (tuple(sorted(taken - replacers - {memory_id})), tuple(sorted(replacers)))

    def _bundle_values(self, txn: TxnRows, memory_id: str, forgotten: Set[str],
                       words: Sequence[str]) -> list[str] | None:
        """The values a bundled memory's forget is about when its head names one of the user's
        ``words`` as replaced (``_chain_parts``): the words, the head's parts naming a replaced
        value (``forget_admission.replaced_parts``) and every other statement of what the
        forget takes (its earlier versions and correction chain keep their own values); None for
        any other memory, whose replacers are read against all its statements."""
        item = txn.get("item", memory_id) if words else None
        head = None if item is None else txn.get("revision", item["current_revision_id"])
        if head is None:
            return None
        statement = head["snapshot"]["statement"]
        if not any(names_replaced(statement, word, unsure=False) for word in words):
            return None
        return [*(text for text in self._statements(txn, forgotten) if text != statement),
                *words, *replaced_parts(statement)]

    def _restated_by(self, txn: TxnRows, memory_id: str | None,
                     restating: Sequence[QuoteSpanV1], seed: ForgetSeedV1 | None = None,
                     ) -> list[ForgottenCitation]:
        """The citation of the sentences a forget withholds as restating what it forgets, as
        admission plans them: cited by the selected head's statement, or with no memory by the
        seed's (``_restated``); none without sentences."""
        if not restating:
            return []
        item = None if memory_id is None else txn.get("item", memory_id)
        head = None if item is None else txn.get("revision", item["current_revision_id"])
        if head is not None:
            snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
            return [_restated(snapshot.entity_id, snapshot.statement, snapshot.embedding,
                              snapshot.embedding_space, restating,
                              sensitive=bool(snapshot.sensitivity_classes),
                              about=snapshot.about_entity_ids)]
        if seed is None:
            return []
        return [_restated(USER, seed.statement, seed.embedding, seed.embedding_space, restating,
                          sensitive=bool(classify([seed.statement])))]

    def _replacing(
        self, txn: TxnRows, forgotten: Set[str], restated: Sequence[ForgottenCitation],
        links: Callable[[str], Iterable[CorrectionEdge]], about: Sequence[str] = (),
        values: Sequence[str] | None = None, allowed: Set[str] | None = None,
        nominated: set[str] | None = None,
    ) -> tuple[set[str], set[str]]:
        """``forgotten`` with every memory that replaced one it takes or leaves unreadable and
        names a value of ``forgotten`` or of those unreadable memories (complete forgets), and
        those replacers alone.

        A memory is left unreadable when the forget's own plan, its clause keepers included,
        leaves no readable support group of it (``_left_unreadable``). Its value counts as
        forgotten only when it states a value of what the forget takes or of the user's
        ``about`` words (``forget_admission.states_value``): "The user works at Tencent.",
        left unreadable by the withheld "I play badminton every Saturday morning, and I work
        at Tencent.", states nothing forgotten, so "The user works at Alibaba now instead of
        Tencent." stays. A replacer taken brings its chain and what it cites, so another
        round reads the plan again; the rounds stop when nothing more is taken. ``values``
        replace the statements of ``forgotten`` as what is forgotten (``_bundle_values``).
        A memory no link ties to what is taken counts as a replacer too when its head, about
        the same primary entity, names such a value as replaced (``_naming_unlinked``). A
        replacer an ``update`` or ``correct`` link ties to what is taken, naming its forgotten
        value, is taken by the rule alone (its own ended or replaced part names the value's
        predicate, ``forget_admission.names_replaced``); one tied only to a memory the plan
        leaves unreadable, and an unlinked one, are nominations: with ``allowed`` only those
        the forget judge confirmed are taken; every one found is added to ``nominated``.

        Raises:
            QuoteMemError: ``budget_exceeded`` past ``MAX_FORGET_CHAIN`` memories.
        """
        taken = set(forgotten)
        replacers: set[str] = set()
        values = self._statements(txn, forgotten) if values is None else list(values)
        statements, quotes = [*values, *about], self._quoted_texts(txn, forgotten)
        read = readability(txn)
        entities = self._entities_of(txn, forgotten)
        while True:
            # A memory that replaced one taken and names its forgotten value goes by the rule,
            # for that value alone (its own change names the value's predicate), and so does
            # one that replaced it in turn naming that value too; left unreadable, its own value
            # ("climbing") is no forgotten one.
            ruled: set[str] = set()
            pending = list(taken)
            while pending:
                for replacer in self._replacers(txn, read, pending.pop(), values):
                    if replacer not in taken and replacer not in ruled:
                        ruled.add(replacer)
                        pending.append(replacer)
            unreadable = {memory_id for memory_id in self._left_unreadable(
                txn, taken, replacers, restated) if memory_id not in ruled and states_value(
                    self._statements(txn, [memory_id]), self._quoted_texts(txn, [memory_id]),
                    statements, quotes)}
            named = [*values, *self._statements(txn, unreadable)]
            # A memory that replaced one the plan only leaves unreadable (an unreadable memory
            # still holds the words it names), and one no link ties to what is taken, are only
            # nominations: the forget judge confirms them (``allowed``).
            asked: set[str] = set()
            pending = list(unreadable)
            while pending:
                for replacer in self._replacers(txn, read, pending.pop(), named):
                    if replacer not in taken and replacer not in ruled and replacer not in asked:
                        asked.add(replacer)
                        pending.append(replacer)
            asked |= set(self._naming_unlinked(txn, read, taken | ruled | asked, named,
                                               entities))
            if nominated is not None:
                nominated.update(asked)
            if allowed is not None:
                asked &= allowed
            found = ruled | asked
            if not found:
                return taken, replacers
            replacers |= found
            for replacer in found:
                taken |= {replacer, *correction_chain(replacer, links)}
            if len(taken) > MAX_FORGET_CHAIN + 1:
                raise QuoteMemError("budget_exceeded", "The forget chain exceeds its bound")

    def _left_unreadable(self, txn: TxnRows, taken: Set[str], replacers: Set[str],
                         restated: Sequence[ForgottenCitation]) -> set[str]:
        """The readable memories outside ``taken``, current or past, that a forget of ``taken``
        (its ``replacers`` citing only what mentions a forgotten value) withholding
        ``restated`` leaves with no readable support group: S and the clause keepers as
        admission plans them (``plan_stating``)."""
        scope = ScopeV1(scope_kind=txn.scope_kind, scope_id=txn.scope_id)
        ordered = sorted(taken)
        reworded = self._reworded(scope, txn, ordered)
        _, _, cited = self._chain_plan(ordered, reworded, replacers=self._replaced(
            txn, sorted(taken - replacers), sorted(replacers)))
        cited = [*cited, *restated]
        keepers = [*self._keeper_groups(scope, txn, set(taken), cited),
                   *self._reworded_keepers(txn, reworded)]
        spans, live = plan_stating(cited, keepers, withdrawn=self._withdrawn(
            txn, set(taken), cited), any_script=True,
            quoted=lambda span: _quoted_text(txn, span))
        kept = {(keepers[index].memory_id, keepers[index].spans) for index in live
                if keepers[index].memory_id}
        withheld: dict[str, list[tuple[int, int]]] = {}
        for span in spans:
            withheld.setdefault(span.evidence_id, []).append((span.byte_start, span.byte_end))
        if not withheld:
            return set()
        read = readability(txn)
        found: set[str] = set()
        for item in self._pages(MemoryTableRows(scope_kind=txn.scope_kind, scope_id=txn.scope_id)):
            memory_id = item["object_id"]
            if memory_id in taken or item["lifecycle_state"] == "erased":
                continue
            head = txn.get("revision", item["current_revision_id"])
            if head is None or not set(withheld).intersection(head["evidence_ids"]):
                continue
            snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
            if not read.p_ver(snapshot):
                continue
            groups = [tuple(snapshot.quote_spans[index] for index in group)
                      for group in snapshot.support_groups
                      if read.p_group(snapshot.quote_spans, group)]
            if groups and not any(
                    (memory_id, group) in kept or not any(
                        start < span.byte_end and span.byte_start < end
                        for span in group for start, end in withheld.get(span.evidence_id, ()))
                    for group in groups):
                found.add(memory_id)
        return found

    def _replacers(self, txn: TxnRows, read: Readability, memory_id: str,
                   values: Sequence[str]) -> list[str]:
        """The readable memories of the scope, current or past, that replaced ``memory_id`` (an
        ``update`` or ``correct`` link on any of their revisions) and whose head statement
        names one of ``values`` (``forget_admission.names_replaced``)."""
        found: list[str] = []
        for link in txn.links_to("supersedes", memory_id):
            if link["relation"] not in ("update", "correct"):
                continue
            source = txn.get("revision", link["source_revision_id"])
            item = None if source is None else txn.get("item", source["memory_id"])
            if item is None or item["lifecycle_state"] == "erased" or (
                    item["scope_kind"], item["scope_id"]) != (txn.scope_kind, txn.scope_id):
                continue
            head = txn.get("revision", item["current_revision_id"])
            if head is None:
                continue
            snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
            if read.p_ver(snapshot) and any(names_replaced(snapshot.statement, value)
                                            for value in values):
                found.append(item["object_id"])
        return found

    def _naming_unlinked(self, txn: TxnRows, read: Readability, taken: Set[str],
                         values: Sequence[str], entities: Set[str]) -> list[str]:
        """The readable memories outside ``taken``, current or past, that no link ties to what a
        complete forget takes but whose head, about one of the same primary ``entities``,
        names one of ``values`` as a value it replaced, ended or left
        (``forget_admission.names_replaced``, never failing closed on a guess): "The user
        moved from Shanghai to Hangzhou last month." stored beside "The user lives in
        Shanghai." with no update link. "The user's sister moved from Shanghai." is about
        another entity and stays."""
        found: list[str] = []
        for item in self._pages(MemoryTableRows(scope_kind=txn.scope_kind, scope_id=txn.scope_id)):
            memory_id = item["object_id"]
            if memory_id in taken or item["lifecycle_state"] == "erased":
                continue
            head = txn.get("revision", item["current_revision_id"])
            if head is None:
                continue
            snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
            if snapshot.entity_id in entities and read.p_ver(snapshot) and any(
                    not other_subject(value, snapshot.statement)
                    and names_replaced(snapshot.statement, value, unsure=False)
                    for value in values):
                found.append(memory_id)
        return found

    def _entities_of(self, txn: TxnRows, memory_ids: Iterable[str]) -> set[str]:
        """The primary entities of the memories' heads."""
        found: set[str] = set()
        for memory_id in memory_ids:
            item = txn.get("item", memory_id)
            head = None if item is None else txn.get("revision", item["current_revision_id"])
            if head is not None:
                found.add(MemorySnapshotV2.model_validate(head["snapshot"]).entity_id)
        return found

    def _statements(self, txn: TxnRows, memory_ids: Iterable[str]) -> list[str]:
        """The statements of every version of the memories, in id order."""
        return [revision["snapshot"]["statement"] for memory_id in sorted(memory_ids)
                for revision in self._revisions(txn, memory_id)]

    def _quoted_texts(self, txn: TxnRows, memory_ids: Iterable[str]) -> list[str]:
        """The quoted words of every version of the memories, readable or not."""
        texts: list[str] = []
        for memory_id in sorted(memory_ids):
            for revision in self._revisions(txn, memory_id):
                for raw in revision["snapshot"].get("quote_spans", ()):
                    text = self._span_text(QuoteSpanV1.model_validate(raw))
                    if text is not None:
                        texts.append(text)
        return texts

    def _span_text(self, span: QuoteSpanV1) -> str | None:
        """The words of one span of an Evidence row, withheld or not; None when the row is
        gone."""
        row = self._writer.get_row("evidence", span.evidence_id)
        if row is None:
            return None
        raw = row["text"].encode("utf-8")
        return raw[span.byte_start - row["byte_start"]:span.byte_end - row["byte_start"]
                   ].decode("utf-8", "replace")

    def promotions(self, scope: ScopeV1, memory_ids: Sequence[str], *,
                   restating: Sequence[QuoteSpanV1] = (), seed: ForgetSeedV1 | None = None,
                   about: Sequence[str] = (), bundled: Set[str] = frozenset(),
                   allowed: Set[str] | None = None) -> list[str]:
        """The memories a complete forget by statement also takes as targets (``replacing``).

        A plan of ``memory_ids`` withholding ``restating`` (with no memory, under ``seed``)
        leaves a memory it does not take unreadable, its clause keepers included
        (``_left_unreadable``: a past "The user plays badminton every Saturday morning at a
        club in Xuhui." whose sentence it withholds); when a readable memory replaced that memory
        and still names its value, the plan takes the memory itself, so that memory's forget
        takes its replacer as its chain does (``_chain``). Only a memory stating a value of
        the targets or of the user's ``about`` words is taken so
        (``forget_admission.states_value``): one the plan only leaves unreadable as part of a
        withheld sentence is not. A ``bundled`` memory is planned as forgotten for the ``about``
        words (``_chain_parts``). Writes nothing.

        Returns:
            Those memories in id order; empty for a forget that is not complete.
        """
        if not self._complete():
            return []
        writer = self._writer
        with writer.write_scope():
            txn = self._open_txn(scope, None, self._clock())
            selected = self._restating(scope, restating)
            taken: set[str] = set()
            replacers: set[str] = set()
            present = [memory_id for memory_id in memory_ids
                       if txn.get("item", memory_id) is not None]
            for position, memory_id in enumerate(present):
                refuted, replaced = self._chain_parts(
                    txn, memory_id, selected if position == 0 else (),
                    about if memory_id in bundled else (), allowed=allowed)
                taken |= {memory_id, *refuted, *replaced}
                replacers |= set(replaced)
            restated = self._restated_by(txn, present[0] if present else None, selected, seed)
            read = readability(txn)
            statements = [*self._statements(txn, taken - replacers), *about]
            quotes = self._quoted_texts(txn, taken - replacers)
            found = [memory_id for memory_id in sorted(
                self._left_unreadable(txn, taken, replacers, restated))
                     if states_value(self._statements(txn, [memory_id]),
                                     self._quoted_texts(txn, [memory_id]), statements, quotes)
                     and self._replacers(txn, read, memory_id,
                                         self._statements(txn, [memory_id]))]
            if len(found) > MAX_FORGET_CHAIN:
                raise QuoteMemError("budget_exceeded", "The forget chain exceeds its bound")
            return found

    def affected(
        self, scope: ScopeV1, memory_ids: Sequence[str], *,
        restating: Sequence[QuoteSpanV1] = (), seed: ForgetSeedV1 | None = None,
        allowed: Set[str] | None = None,
    ) -> list[tuple[str, Literal["unreadable", "forgotten"], str | None]]:
        """What a complete plan of ``memory_ids`` changes besides them, for its prepare to name
        before the user confirms: every memory their forgets take with them (``chains``:
        the memories its corrections refuted and the replacers naming a forgotten value),
        ``forgotten``, then every readable memory the plan may leave unreadable (``preview``),
        ``forgotten`` when all its quotes lie in what the plan withholds, each with its head
        statement. Empty for a forget that is not complete. Writes nothing."""
        if not self._complete():
            return []
        taken = self.chains(scope, memory_ids, restating=restating, allowed=allowed)
        found: dict[str, Literal["unreadable", "forgotten"]] = dict.fromkeys(taken, "forgotten")
        for memory_id, covered in self.preview(scope, memory_ids, restating=restating,
                                               seed=seed, allowed=allowed):
            found.setdefault(memory_id, "forgotten" if covered else "unreadable")
        writer = self._writer
        rows: list[tuple[str, Literal["unreadable", "forgotten"], str | None]] = []
        for memory_id, effect in found.items():
            item = writer.get_row("item", memory_id)
            head = None if item is None else writer.get_row("revision",
                                                            item["current_revision_id"])
            rows.append((memory_id, effect,
                         None if head is None else head["snapshot"]["statement"] or None))
        return rows

    def chains(self, scope: ScopeV1, memory_ids: Sequence[str], *,
               restating: Sequence[QuoteSpanV1] = (), bundled: Set[str] = frozenset(),
               words: Sequence[str] = (), allowed: Set[str] | None = None) -> list[str]:
        """The memories a plan of ``memory_ids`` forgets besides them (their ``_chain``, the first
        with ``restating``, a ``bundled`` one's for the user's ``words``), in id order, so a
        prepare can name each one. Writes nothing."""
        writer = self._writer
        with writer.write_scope():
            txn = self._open_txn(scope, None, self._clock())
            found: set[str] = set()
            for position, memory_id in enumerate(memory_ids):
                if txn.get("item", memory_id) is not None:
                    found.update(self._chain(txn, memory_id,
                                             restating if position == 0 else (),
                                             words if memory_id in bundled else (),
                                             allowed=allowed))
            return sorted(found - set(memory_ids))

    def tombstoned(self, scope: ScopeV1, memory_ids: Sequence[str], *,
                   restating: Sequence[QuoteSpanV1] = (), bundled: Set[str] = frozenset(),
                   words: Sequence[str] = (), allowed: Set[str] | None = None,
                   ) -> list[tuple[str, tuple[str, ...], tuple[float, ...], str]]:
        """The statements, keywords and vectors (with their embedding spaces) a plan of
        ``memory_ids`` tombstones, as ``chains``
        plans them: each memory's and the memories it refutes, every version of each, but no
        replacer's (``_chain_parts``) and, for a ``bundled`` memory, the user's ``words`` in its
        stead: what its forget shadow names (``_shadow``). The memories are
        forgotten in order, so one an earlier one's forget took is forgotten earlier and
        tombstones nothing of its own. Writes nothing."""
        writer = self._writer
        found: list[tuple[str, tuple[str, ...], tuple[float, ...], str]] = []
        taken: set[str] = set()
        with writer.write_scope():
            txn = self._open_txn(scope, None, self._clock())
            for position, memory_id in enumerate(memory_ids):
                if txn.get("item", memory_id) is None or memory_id in taken:
                    continue
                refuted, replacers = self._chain_parts(
                    txn, memory_id, restating if position == 0 else (),
                    words if memory_id in bundled else (), allowed=allowed)
                taken |= {memory_id, *refuted, *replacers}
                if memory_id in bundled:
                    found += [(text, (), (), "") for text in words]
                for stated in ((memory_id,) if memory_id not in bundled else ()) + refuted:
                    found += [(cast(str, revision["snapshot"]["statement"]),
                               tuple(cast(list[str], revision["snapshot"].get("keywords") or [])),
                               tuple(cast(list[float],
                                          revision["snapshot"].get("embedding") or [])),
                               cast(str, revision["snapshot"].get("embedding_space") or ""))
                              for revision in self._revisions(txn, stated)
                              if revision["snapshot"].get("statement")]
        return found

    def nominations(self, scope: ScopeV1, memory_ids: Sequence[str], *,
                    restating: Sequence[QuoteSpanV1] = (), bundled: Set[str] = frozenset(),
                    words: Sequence[str] = ()) -> list[str]:
        """The memories a complete plan of ``memory_ids`` would take beyond its base links
        and its linked replacers were every word rule's nomination confirmed: the
        unlinked memories naming a forgotten value as replaced (``_chain_parts``), in id order.
        The prepare asks the forget judge about each (``forget_pick.tells``); empty for a
        forget that is not complete. Writes nothing."""
        if not self._complete():
            return []
        writer = self._writer
        with writer.write_scope():
            txn = self._open_txn(scope, None, self._clock())
            found: set[str] = set()
            for position, memory_id in enumerate(memory_ids):
                if txn.get("item", memory_id) is not None:
                    self._chain_parts(txn, memory_id, restating if position == 0 else (),
                                      words if memory_id in bundled else (), nominated=found)
            return sorted(found - set(memory_ids))

    def _replaced(self, txn: TxnRows, forgotten: Sequence[str],
                  replacers: Sequence[str]) -> dict[str, list[str]]:
        """Each replacer with the statements and quoted words of what the forget takes for its
        own sake, the values a replacer's quotes are read against (``_chain_plan``)."""
        values = ([*self._statements(txn, forgotten), *self._quoted_texts(txn, forgotten)]
                  if replacers else [])
        return {replacer: values for replacer in replacers}

    def _chain_plan(
        self, memory_ids: Sequence[str], reworded: Sequence[str] = (), *,
        replacers: Mapping[str, Sequence[str]] | None = None,
        narrowing: Mapping[str, Sequence[str]] | None = None,
        exact: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[list[ForgetVersionSelection], list[str], list[ForgottenCitation]]:
        """D, the anchored entities and what each version cites.

        A replacer (``replacers``, with the forgotten values it names, ``_chain_parts``)
        cites only the clauses of its quotes that mention one of them, or that no test can
        read against them (``forget_admission.mentions``, ``_mentioning``): of "搬家后我改去
        西湖附近的攀岩馆攀岩了，不再打羽毛球。" it cites "不再打羽毛球", so another memory quoting
        the first clause keeps it readable. Each such clause is cut to its parts naming a
        replaced value when its rest mentions none of them, or none of its ``narrowing``
        values (a named bundle's: the user's words alone, ``_bundle_values``).

        A replacer the forget rewords (``exact``, with the names and values its new wording
        leaves out) cites only their occurrences in its quotes ("上海" of "我上个月从上海搬到了
        杭州"), so the rest of its sentence stays readable for its new wording.

        Each memory contributes ``plan_forget_admission`` over its complete revision
        chain; a pair two memories share keeps the version of the first (the selected
        memory comes first), so admission and its part (b) pair the same versions with the
        planned tombstone ids. Each reworded memory adds to D the wordings it had
        before a forgotten correction reworded it (``plan_earlier_wordings``), never
        to the citations: it survives and its head keeps its quotes.
        Admission derives S from the citations with ``plan_stating_spans``.

        Raises:
            QuoteMemError: ``binding_mismatch`` for a memory already gone.
        """
        versions: dict[tuple[str, str], ForgetVersionSelection] = {}
        entities: set[str] = set()
        cited: list[ForgottenCitation] = []
        for memory_id in memory_ids:
            item = self._writer.get_row("item", memory_id)
            if item is None:
                raise QuoteMemError("binding_mismatch",
                                    "A forgotten Item went before its tombstones")
            revisions = sorted(self._pages(RevisionRows(memory_id=memory_id)),
                               key=lambda row: row["version"])
            selection = _plan_selection(item, revisions)
            for version in selection.versions:
                snapshot = version.snapshot
                versions.setdefault((snapshot.entity_id, snapshot.statement_digest), version)
            values = (replacers or {}).get(memory_id)
            narrow = (narrowing or {}).get(memory_id)
            left = (exact or {}).get(memory_id)
            if left is not None:
                cited.extend(replace(citation, groups=tuple(group for group in (
                    tuple(part for span in spans for part in self._occurrences(span, left))
                    for spans in citation.groups) if group)) for citation in selection.cited)
                continue
            cited.extend(selection.cited if values is None else [
                replace(citation, groups=tuple(group for group in (
                    tuple(part for span in spans
                          for part in self._mentioning(span, values, narrow))
                    for spans in citation.groups) if group))
                for citation in selection.cited])
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

    def _occurrences(self, span: QuoteSpanV1, values: Sequence[str]) -> list[QuoteSpanV1]:
        """Where a quote shows each of ``values`` (``forget_reword.value_pattern``); the whole
        quote when its words cannot be read."""
        text = self._span_text(span)
        if text is None:
            return [span]
        return [span.model_copy(update={
            "byte_start": span.byte_start + len(text[:match.start()].encode("utf-8")),
            "byte_end": span.byte_start + len(text[:match.end()].encode("utf-8"))})
            for value in values for match in value_pattern(value).finditer(text)]

    def _mentioning(self, span: QuoteSpanV1, values: Sequence[str],
                    narrow: Sequence[str] | None = None) -> list[QuoteSpanV1]:
        """The clauses of a quote, cut at clause marks in any script, that mention one of
        ``values`` or that no test reads against them (``forget_admission.mentions``), each
        cut to its parts naming a value as replaced when the rest mentions none of them, or
        of ``narrow`` when given (``forget_admission.naming_parts``: "instead of playing
        badminton"); the whole quote when its words cannot be read, so it is withheld."""
        text = self._span_text(span)
        if text is None:
            return [span]
        found: list[QuoteSpanV1] = []
        start = 0
        for end in [*(match.start() for match in _CLAUSE_MARKS.finditer(text)), len(text)]:
            clause = text[start:end]
            lead = start + len(clause) - len(clause.lstrip())
            body = clause.strip()
            if body and mentions(body, values):
                for low, high in naming_parts(body, values if narrow is None else narrow):
                    first = span.byte_start + len(text[:lead + low].encode("utf-8"))
                    found.append(span.model_copy(update={
                        "byte_start": first,
                        "byte_end": first + len(body[low:high].encode("utf-8"))}))
            start = end + 1
        return found

    def _reworded_by(self, txn: TxnRows, memory_id: str, verdicts: Set[str]) -> bool:
        """Whether one of a memory's versions is a rewording under one of ``verdicts``."""
        return bool(verdicts) and any(
            revision["reason_code"] == "rewording"
            and revision["snapshot"].get("verdict_ref") in verdicts
            for revision in self._revisions(txn, memory_id))

    def _reworded(self, scope: ScopeV1, txn: TxnRows, forgotten: Sequence[str]) -> tuple[str, ...]:
        """The surviving memories a forgotten correction reworded.

        A rewording carries the verdict of the correction that caused it. An active memory
        outside the forgotten set is reworded when its head is such a rewording, or when
        a forgotten version's ``contradicts`` link targets it and one of its versions is
        (a head revised again after its rewording is found only through that link).
        Read under the Writer mutex in the admission transaction and frozen in the fence.

        Raises:
            QuoteMemError: ``budget_exceeded`` past ``MAX_FORGET_CHAIN`` memories.
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
        for item in self._pages(
            MemoryTableRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id)
        ):
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
            raise QuoteMemError("budget_exceeded", "The forget's reworded Items exceed their bound")
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
                        sensitive=bool(snapshot.sensitivity_classes), reworded=True,
                        memory_id=memory_id))
        return keepers

    def _withdrawn(self, txn: TxnRows, forgotten: Set[str],
                   cited: Sequence[ForgottenCitation]) -> frozenset[tuple[str, str]]:
        """The names a forgotten version gives that no surviving memory gives."""
        given = {pair for version in cited for pair in version.names}
        surviving = {(row["entity_id"], row["name_norm"])
                     for entity in {entity for entity, _ in given}
                     for row in txn.names_of(entity)
                     if row["naming_memory_id"] not in forgotten}
        return frozenset(given - surviving)

    def _planned_trim(
        self, txn: TxnRows, read: Readability, scope: ScopeV1,
        withheld: Mapping[str, Sequence[tuple[int, int]]], restating: Sequence[QuoteSpanV1],
        plans: Sequence[Sequence[ForgottenCitation]], *, modifier: bool = False,
    ) -> Callable[[MemorySnapshotV2], bool]:
        """Whether privacy compensation trims a dependent the ``plans`` (what each cites) keep only
        on their parts, read as it reads one after their forgets (``_compensate``): ``withheld``
        withheld too, the words of S less the sentences restating the fact (``restating``)
        and the plans' cited statements as the tombstones', so the preview never names one
        kept that ends unreadable ("The user has a daughter." quoting "我的三岁女儿叫小美"
        beside a forgotten "The user's daughter 小美 is three years old.": its only readable
        clause holds 小美). Every forget's S is read at once, which forbids no fewer words
        than any one of them."""
        held = self._restating(scope, restating)
        words: set[str] = set()
        terms: set[str] = set()
        for evidence_id, ranges in withheld.items():
            for start, end in ranges:
                span = QuoteSpanV1(evidence_id=evidence_id, byte_start=start, byte_end=end)
                if any(span.evidence_id == other.evidence_id
                       and other.byte_start <= span.byte_start
                       and span.byte_end <= other.byte_end for other in held):
                    continue
                text = _quoted_text(txn, span)
                if text is not None:
                    words |= support_words(text)
                    terms |= fact_terms(text)
        cited = [version for versions in plans for version in versions if version.statement]
        forgotten = _Forgotten(
            frozenset(words), tuple(version.statement for version in cited), True,
            frozenset(terms), tuple((version.vector, version.space) for version in cited),
            modifier=modifier)
        planned = _planned_read(read, withheld)

        def trims(snapshot: MemorySnapshotV2) -> bool:
            if _trim_refused(txn, snapshot, forgotten):
                return False
            return trimmed_support(
                snapshot, planned, forgotten=_forgotten_terms(forgotten, snapshot.statement),
                quoted=lambda span: _quoted_text(txn, span),
                speakers=lambda span: _speaker_words(txn, span), any_script=True,
            ) is not None or bool(_bare_name_support(txn, planned, snapshot, forgotten.words))

        return trims

    def _seed_asking(self, scope: ScopeV1, txn: TxnRows,
                     restated: ForgottenCitation) -> ForgottenCitation:
        """The citation of a forget of sentences under the user's words (``restated``), giving
        the names those words ask (``_NAME_ASKED``), so the plan keeps them nowhere and no
        clause showing one stays ("陈雅琴" of "我女儿的钢琴老师叫陈雅琴" for "忘掉我女儿钢琴老师的
        名字", whose judge picked the sentence alone). Every name a readable memory gives that
        the sentences show is asked, whether a bare keeper gives it or not, unless each memory
        giving it keeps it for those words (``bare_name_stays``, read on its own entity's ties,
        ``names_tied``): a name the words' tie chain does not resolve goes, failing closed.
        Unchanged for words asking no name."""
        statement = restated.statement
        if not _NAME_ASKED.search(statement):
            return restated
        texts = [text for group in restated.groups for span in group
                 if (text := _quoted_text(txn, span))]
        read = readability(txn)
        heads: list[tuple[str, MemorySnapshotV2]] = []
        rows = MemoryTableRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id)
        for row in self._pages(rows):
            head = txn.get("revision", row["current_revision_id"])
            if head is None or row["lifecycle_state"] != "active":
                continue
            snapshot = MemorySnapshotV2.model_validate(head["snapshot"])
            if read.p_ver(snapshot):
                heads.append((row["object_id"], snapshot))
        names = {entry.name for _, snapshot in heads for entry in snapshot.naming}
        tied = names_tied((snapshot.statement, snapshot.entity_id, snapshot.about_entity_ids)
                          for _, snapshot in heads)
        shown: dict[tuple[str, str], list[MemorySnapshotV2]] = {}
        for _, snapshot in heads:
            for entry in snapshot.naming:
                if entry.entity_id != USER and any(
                        token_contained(entry.name, text) for text in texts):
                    shown.setdefault((entry.entity_id, name_norm(entry.name)), []).append(
                        snapshot)
        asked = frozenset(pair for pair, givers in shown.items() if not all(
            bare_name_stays(snapshot, [statement], names, (), tied, told=False)
            for snapshot in givers))
        return replace(restated, names=restated.names | asked) if asked else restated

    def _keeper_groups(
        self, scope: ScopeV1, txn: TxnRows, forgotten: Set[str],
        cited: Sequence[ForgottenCitation],
    ) -> list[KeeperGroup]:
        """The readable groups of the surviving memories a forget may keep.

        A survivor is an ``active`` memory outside the forgotten set whose head
        passes ``p_ver`` now and cites an Evidence row a forgotten version cites. A
        survivor a recorded link joins to a forgotten memory, in either direction,
        is none: a memory the forgotten memory contradicts, corrects or supersedes
        carries the very description the forget is about (a corrected kind).
        Each group of a survivor that passes ``p_group`` is a keeper group, marked
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
        primary entity's names that pass ``p_name`` from surviving naming memories, only for
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
                    naming_memory = MemoryRow.model_validate(naming)
                    head = txn.get("revision", naming_memory.current_revision_id)
                    if head is not None and read.p_name(
                            row, naming_memory, MemorySnapshotV2.model_validate(head["snapshot"])):
                        found.add((entity, row.name_norm))
            result = frozenset(found)
            reference_cache[entity] = result
            return result

        keepers: list[KeeperGroup] = []
        for item in self._pages(
            MemoryTableRows(scope_kind=scope.scope_kind, scope_id=scope.scope_id)
        ):
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
                    memory_id=item["object_id"],
                ))
        return keepers

    def _unit(self, txn: TxnRows, memory_ids: Sequence[str]) -> list[str]:
        """The retention unit's memories, in discovery order.

        The forgotten memory, its chain (``_chain``, frozen in the fence) and,
        transitively, every ``historical`` memory whose ``corrects_history`` link
        targets one of their revisions (``cascade_source_item``); a memory already
        gone is skipped.

        Raises:
            QuoteMemError: ``budget_exceeded`` past ``_CASCADE_LIMIT`` historical memories.
        """
        unit: list[str] = []
        pending, seen = list(reversed(memory_ids)), set[str]()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            if len(seen) >= len(memory_ids) + _CASCADE_LIMIT - 1:
                raise QuoteMemError("budget_exceeded", "The forget cascade exceeds its bound")
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
        """Up to ``limit`` ``supersedes``/``contradicts`` links targeting the unit's memories.

        Other memories carry them, however many, so they are not the unit's own
        rows: ``delete`` deletes them in batches before the unit's transaction, and
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
        """Refuse, before any write, a forget whose ``delete`` could never finish.

        ``delete`` batches S, the dependents, the incoming links and the expiries, but
        the unit's own rows leave in one transaction with at most ``entities``
        entity erasures, and their records and deliveries are frozen into the
        fence. That transaction is built here on rows never committed: past
        ``MAX_TX_ROWS`` or a fence bound, ``delete`` would refuse it on every resume.

        Raises:
            QuoteMemError: ``budget_exceeded`` past the cascade, transaction, record
                or delivery bound.
        """
        txn = self._open_txn(scope, None, now)
        self._delete_unit(scope, txn, self._unit(txn, memory_ids), evidence)
        if len(txn.deleted) + entities + _UNIT_OVERHEAD > MAX_TX_ROWS:
            raise QuoteMemError("budget_exceeded",
                                "The forgotten Item's own rows exceed a transaction")

    def _delete_unit(
        self, scope: ScopeV1, txn: TxnRows, unit: Sequence[str], evidence: Set[str],
    ) -> tuple[dict[str, int], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Delete the unit's own rows in ``txn``; find the records that leave with them.

        Returns:
            The deletion counts, then the Artifacts, Controls and Search
            deliveries the purge batches delete.

        Raises:
            QuoteMemError: ``budget_exceeded`` past the record or delivery bounds.
        """
        candidates = list(self._pages(CandidateRows(
            scope_kind=scope.scope_kind, scope_id=scope.scope_id)))
        removed, deleted = self._delete_memory(txn, unit, candidates)
        artifacts, controls = self._retention_records(scope, deleted, candidates)
        recalls = self._deliveries(scope, set(unit), evidence)
        if len(recalls) > MAX_FORGET_RECALLS:
            raise QuoteMemError("budget_exceeded",
                                "The forget's Search deliveries exceed their bound")
        return removed, artifacts, controls, recalls

    def _delete_memory(
        self, txn: TxnRows, unit: Sequence[str], candidates: Sequence[Row],
    ) -> tuple[dict[str, int], list[Row]]:
        """Delete the unit's own rows: each memory with its revisions and their rows.

        The rows are the memory, its revisions, their outgoing links, the
        ``corrects_history`` links targeting them (the unit's own), its derived
        names and the candidates naming it or one of its revisions. The links
        other memories carry to it left in earlier batches (``_incoming``).

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
        """The Artifacts and Controls that leave with the deleted candidates.

        Each deleted position takes its ProposalV2 and the records of every
        review of it and of every confirmation vote, under any reviewing
        owner: the judge result indexes, their neighbour snapshots, the member
        outputs they own and their ``judge_output`` locators. A call manifest
        leaves once no member it lists keeps its snapshot; an owner's extraction
        reply, context record and embedding batches leave once no surviving
        candidate of that owner refers to them;
        a manifest takes its parts. A record a surviving row still refers to
        stays, internal and unreadable (``provider_records``); content-free
        Runs keep their inputs and result locators.

        Returns:
            The sorted Artifact ids and the sorted Control keys.

        Raises:
            QuoteMemError: ``binding_mismatch`` for a review record of another
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
            raise QuoteMemError("budget_exceeded",
                                "The forget retention records exceed their bounds")
        return tuple(sorted(artifacts)), tuple(sorted(controls))

    def _deliveries(
        self, scope: ScopeV1, items: Set[str], evidence: Set[str],
    ) -> tuple[str, ...]:
        """The scope's Search deliveries that returned one of ``items`` or ``evidence``.

        A delivery keeps its query, options and returned identities; one
        that returned a forgotten memory or an Evidence row whose spans the forget
        withholds belongs to the retention unit. Its interactions cannot
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
            QuoteMemError: ``binding_mismatch`` for a record of another variant.
        """
        if reference is None or self._writer.get_row("artifact", reference) is None:
            return None
        record = self._writer.read_logical(reference, kind)
        if not isinstance(record, model):
            raise QuoteMemError("binding_mismatch", "A retention record has another variant")
        return record

    def _purge(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        """Delete the frozen retention records, one bounded batch per transaction.

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
        with writer.write_scope():
            if writer.get_control(cast(str, control["key"])) != control:
                raise QuoteMemError("binding_mismatch", "The erasure fence changed under its owner")
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
            _, plan = writer.plan_rows([progress], None)
            if removed:
                writer.plan_deletes(plan, removed)
            writer.write_plan(plan)

    # Scan -------------------------------------------------------------------------------
    def final_scan(self, scope: ScopeV1, fence: ErasureFenceV1) -> FinalScan:
        """The six scan checks over durable storage and the published read view.

        The scan reruns the choices on a transaction that is never committed: no
        dependent may still need a requote, no readable survivor a keyword-only revision
        and no entity an erasure.
        """
        writer = self._writer
        txn = self._open_txn(scope, fence.worker_run_id, self._clock())
        targets = _items(fence.memory_id, fence.chain_ids)
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
            # Covered unless its row is erased (a privacy erase clears the ranges).
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
            raise _Residual("service_unhealthy", "The forget final scan found residual content")
        digest = self._writer.bind_content("forget-final-scan", {
            "memory": fence.memory_id, "chain": list(fence.chain_ids),
            "reworded": list(fence.reworded_ids),
            "tombstones": list(fence.tombstone_ids),
            "spans": [span.model_dump(mode="json") for span in fence.spans],
            "records": [*fence.artifacts, *fence.controls, *fence.recalls],
            "commit_seq": self._writer.get_counter("commit_seq"),
            "tombstone_seq": self._writer.get_counter(f"tombstone_seq:{scope.scope_id}"),
        })
        self._write_progress(scope, fence, control, "scanned", fence.counts, scan_digest=digest)

    # Release -------------------------------------------------------------------------------
    def _write_receipts(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        writer = self._writer
        prepare, final = self._receipt_pair(scope, fence)
        ticket = writer.issue_ticket(fence.controller_run_id)
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
        self._write_progress(scope, fence, control, "receipted", fence.counts,
                             prepare_ref=prepare_row, final_ref=final_row, index_ref=index_row)

    def _release(self, scope: ScopeV1, fence: ErasureFenceV1, control: Row) -> None:
        writer, publisher = self._writer, self._publisher

        def barrier() -> None:
            writer.release_erased_caches()

        privacy_quiescence(writer, publisher, scope, barrier=barrier)
        with writer.write_scope():
            now = self._clock()
            current = writer.get_control(cast(str, control["key"]))
            if current != control:
                raise QuoteMemError("binding_mismatch", "The erasure fence changed before release")
            rows: list[RowWrite] = []
            for run_id in (fence.worker_run_id, fence.controller_run_id):
                run = dict(cast(Row, writer.get_row("run", run_id)))
                run.update(status="succeeded", stage="receipt", started_at=run["started_at"] or now,
                           updated_at=now, finished_at=now)
                if run_id == fence.controller_run_id:
                    run["result_ref"] = fence.index_ref
                rows.append(RowWrite("run", writer.parse_row("run", run)))
            _, plan = writer.plan_rows(rows, None)
            writer.plan_deletes(plan, [("control", dict(control))])
            writer.write_plan(plan)
            publisher.publish_erasure(writer, scope, None, writer.get_counter("commit_seq"),
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
            raise QuoteMemError("service_unhealthy",
                                "The forget controller has no verified receipt")
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

    def _write_progress(
        self, scope: ScopeV1, fence: ErasureFenceV1, control: Row, phase: ErasurePhase,
        counts: ForgetCountsV1, **update: Any,
    ) -> None:
        writer = self._writer
        with writer.write_scope():
            if writer.get_control(cast(str, control["key"])) != control:
                raise QuoteMemError("binding_mismatch", "The erasure fence changed under its owner")
            row = self._fence_row(fence, control, phase, counts, self._clock(), **update)
            _, plan = writer.plan_rows([row], None)
            writer.write_plan(plan)

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

    def _withheld(
        self, scope: ScopeV1, grant: PrincipalGrant, intent: str, memory_id: str | None,
        expected_revision_id: str | None = None, expected_content_hash: str | None = None,
    ) -> AbstractContextManager[None]:
        """Withhold from reads, while a confirmation of ``intent`` waits for its turn and
        runs, everything its erasure takes (`_held_by`), planned without writing as its
        admission will plan it (`_admission_plan`).

        Nothing is withheld for a confirmation its admission refuses on what it carries: an
        intent of another scope or principal, one no longer active, expired or of another
        memory, or a memory whose head or content changed since the prepare (a stale
        confirmation never hides a live memory while it waits to be refused)."""
        writer = self._writer
        if not intent.startswith(f"intent:{scope.scope_id}:"):
            return nullcontext()
        with writer.write_scope():
            stored = writer.get_control(intent)
            value = None if stored is None else ForgetIntentV1.model_validate(stored["value"])
            if (value is None or value.state != "active"
                    or value.principal_id != grant.principal_id
                    or self._clock() >= value.expires_at or value.memory_id != memory_id):
                return nullcontext()
            head: Row | None = None
            if memory_id is not None:
                item = writer.get_row("item", memory_id)
                if item is None or (item["scope_kind"], item["scope_id"]) != (
                        scope.scope_kind, scope.scope_id) or (
                        item["current_revision_id"] != expected_revision_id):
                    return nullcontext()
                head = writer.get_row("revision", item["current_revision_id"])
                if head is None or head["content_hash"] != expected_content_hash:
                    return nullcontext()
            txn = self._open_txn(scope, None, self._clock())
            intent_value = value
            # The plan is read exactly as the admission will read it, so only what it takes
            # is withheld: its memories and the bytes it withholds.
            memory_ids, spans = self._held_by(
                txn, memory_id, head,
                lambda: self._admission_plan(scope, txn, intent_value, head), exact=True)
        return gate_for(writer).withhold(
            scope.scope_id, memory_ids, [*spans, *value.restating_spans])

    def withhold(
        self, scope: ScopeV1, memory_ids: Sequence[str], spans: Sequence[QuoteSpanV1] = (),
        *, allowed: Set[str] | None = frozenset(), also: Sequence[str] = (),
    ) -> AbstractContextManager[None]:
        """Withhold from reads of ``scope``, for the body, what forgets of ``memory_ids``
        and of ``also`` (the memories the operation may take or reword beside its forgets:
        by names, ties, listings, records or residue) take, each planned as a forget taking
        beyond its base links only ``allowed`` plans it (`_held_by`: its memories, the quotes
        of every version and the sentences it cites), and the sentences ``spans``, as one set
        (`ErasureGate.withhold`). A memory that is no longer active withholds itself and the
        quotes of its versions."""
        writer = self._writer
        held: list[str] = []
        quoted = list(spans)
        planned = list(dict.fromkeys([*memory_ids, *also]))
        if planned:
            with writer.write_scope():
                txn = self._open_txn(scope, None, self._clock())
                for memory_id in planned:
                    item = writer.get_row("item", memory_id)
                    head = None if item is None or (item["scope_kind"], item["scope_id"]) != (
                        scope.scope_kind, scope.scope_id) else (
                        writer.get_row("revision", item["current_revision_id"]))
                    if item is None or head is None:
                        held.append(memory_id)
                        continue
                    ids, quotes = self._held_by(
                        txn, memory_id, head, self._current_plan(scope, txn, item, head, allowed))
                    held += ids
                    quoted += quotes
        return gate_for(writer).withhold(scope.scope_id, list(dict.fromkeys(held)), quoted)

    def _current_plan(
        self, scope: ScopeV1, txn: TxnRows, item: Row, head: Row, allowed: Set[str] | None,
    ) -> Callable[[], _AdmissionPlan]:
        """The plan of a forget of an active memory at ``head`` (`_plan_of`), to read later; it
        refuses a memory that is no longer active."""
        def plan() -> _AdmissionPlan:
            if item["lifecycle_state"] != "active":
                raise QuoteMemError("operation_conflict", "The memory is not current")
            return self._plan_of(scope, txn, head, restating=(), seed=None, words=(),
                                 allowed=allowed)
        return plan

    def _held_by(
        self, txn: TxnRows, memory_id: str | None, head: Row | None,
        plan: Callable[[], _AdmissionPlan], *, exact: bool = False,
    ) -> tuple[list[str], list[QuoteSpanV1]]:
        """What reads withhold of one forget while it waits for its turn and runs: the memory
        and every memory of the chain it forgets (corrected, retracted, replaced) and the
        sentences it withholds (S).

        With ``exact`` the plan is the one the admission installs, so that is all: a memory
        quoting the forgotten versions' other bytes reads after the erasure as before it, and
        stays readable. Otherwise (a statement plan or an in-chat forget, whose plans the run
        makes again with more inputs) the quotes of every version of each memory and every
        sentence the versions it cites quote are withheld too, a wider hide that cannot miss
        what the run withholds. A plan that cannot be read now (``plan`` refuses, and the
        admission then refuses the forget or plans it again in its turn) gives the memory and
        the quotes of its versions. Writes nothing."""
        memory_ids = [] if memory_id is None else [memory_id]
        spans: list[QuoteSpanV1] = [] if head is None else list(
            MemorySnapshotV2.model_validate(head["snapshot"]).quote_spans)
        try:
            planned: _AdmissionPlan | None = plan()
        except QuoteMemError:
            planned = None
        if planned is not None and exact:
            chain = [memory for memory in planned.chain if memory not in memory_ids]
            unique = {(span.evidence_id, span.byte_start, span.byte_end): span
                      for span in planned.spans}
            return [*memory_ids, *chain], list(unique.values())
        if planned is not None:
            memory_ids += [memory for memory in planned.chain if memory not in memory_ids]
            spans += planned.spans
            spans += [quote for version in planned.versions
                      for quote in version.snapshot.quote_spans]
            spans += [span for cited in planned.cited for group in cited.groups
                      for span in group]
        for memory in memory_ids:
            for revision in self._revisions(txn, memory):
                spans += MemorySnapshotV2.model_validate(revision["snapshot"]).quote_spans
        unique = {(span.evidence_id, span.byte_start, span.byte_end): span for span in spans}
        return memory_ids, list(unique.values())

    def wait_unheld(self, scope: ScopeV1) -> int:
        """Let a prepare wait, up to ``FORGET_WAIT_SECONDS`` within its request, until no
        accepted forget of the scope withholds anything (`ErasureGate.wait_unheld`), so its
        preview reads the scope as that forget leaves it; ``erasure_in_progress`` after.
        Returns the scope's hold generation at the instant nothing was withheld."""
        held = self._writer.mutex._is_owned()  # pyright: ignore[reportPrivateUsage]
        return gate_for(self._writer).wait_unheld(
            scope.scope_id, deadline=wait_deadline(FORGET_WAIT_SECONDS), mutex_held=held)

    @contextmanager
    def erasure_turn(self, scope: ScopeV1, *, patient: bool = False) -> Iterator[None]:
        """Hold the scope's erasure turn for a whole operation of several forgets (a
        confirmation of a statement plan, an in-chat forget): nothing else erases the scope
        between its checks and its forgets, and a forget that arrives meanwhile waits.

        A ``patient`` caller (an in-chat forget, whose client is the Add) waits past
        ``FORGET_WAIT_SECONDS``, keeping what it withholds and what it picked rather than
        failing and running again, up to `PATIENT_WAIT_SECONDS` and a margin before its
        request's end (`patient_deadline`), so the forget runs within its budget."""
        with one_operation(), self._turn(scope, user_request(), patient=patient):
            yield

    @contextmanager
    def _turn(self, scope: ScopeV1, user: bool, *, patient: bool = False) -> Iterator[float]:
        """Hold the scope's erasure turn (`quotemem.store.erasure_gate`, rules 2 and 5); yield the
        end of this request's wait. The turn is never waited for under the Writer mutex."""
        deadline = patient_deadline() if patient else wait_deadline(FORGET_WAIT_SECONDS)
        held = self._writer.mutex._is_owned()  # pyright: ignore[reportPrivateUsage]
        with gate_for(self._writer).turn(scope.scope_id, user=user, deadline=deadline,
                                         mutex_held=held):
            yield deadline

    def _admit_in_turn[T](self, scope: ScopeV1, deadline: float, admit: Callable[[], T]) -> T:
        """Run ``admit`` in the turn; behind a fence no turn holds (one left by a failed run)
        try again until ``deadline``, every check repeated, the Writer mutex released. Past
        the request's own end an operation's first forget is not admitted
        (`check_request_end`): the operation fails as a whole and shows what it withheld
        again; once one was admitted, the operation's others are (`one_operation`)."""
        check_request_end()
        if self._writer.mutex._is_owned():  # pyright: ignore[reportPrivateUsage]
            return admit()
        return gate_for(self._writer).retry_fenced(
            admit, deadline=deadline, fenced=lambda: self._fenced(scope))

    def _in_turn[T](self, scope: ScopeV1, user: bool, operation: Callable[[], T]) -> T:
        """Run ``operation`` in the scope's erasure turn (`_turn`, `_admit_in_turn`)."""
        with self._turn(scope, user and user_request()) as deadline:
            return self._admit_in_turn(scope, deadline, operation)

    def _fenced(self, scope: ScopeV1) -> bool:
        """Whether an erasure fence holds the scope now."""
        with self._writer.write_scope():
            return any(cast(str, row["key"]).startswith("erasure:") for row in
                       self._writer.indexed_reader().scope_fences(scope.scope_id))

    def _refuse_fenced(self, scope: ScopeV1) -> None:
        for row in self._writer.indexed_reader().scope_fences(scope.scope_id):
            if cast(str, row["key"]).startswith("erasure:"):
                raise QuoteMemError("erasure_in_progress", "Another erasure holds the scope")

    def _refuse_maintenance(self, scope: ScopeV1) -> None:
        """Refuse admission behind a maintenance owner.

        admission takes maintenance over; that handoff has no producer yet,
        so admission refuses instead of writing a state no reader validates.
        """
        for row in self._writer.indexed_reader().scope_fences(scope.scope_id):
            if cast(str, row["key"]).startswith(f"maintenance:{scope.scope_id}:"):
                raise QuoteMemError("index_rebuilding", "Maintenance holds the scope")

    def _live_owner(self, scope: ScopeV1) -> tuple[Row, Row] | None:
        """The scope's live finalize lease and its owner Run, or None.

        A ``released`` cycle has no owner left to cancel: its history stays
        provable after the forget (``released_history``), so only a
        ``preparing`` or ``held`` cycle is cancelled.
        """
        lease = self._writer.get_control(f"lease:finalize:{scope.scope_id}")
        if lease is None or lease["value"].get("state") not in {"preparing", "held"}:
            return None
        owner = self._writer.get_row("run", lease["value"]["owner_run_id"])
        if owner is None:
            raise QuoteMemError("reference_missing", "A live finalize lease lost its owner")
        return lease, owner

    def _cancellation(self, lease: Row, owner: Row, controller_id: str, now: int) -> list[RowWrite]:
        """The live cycle's cancellation rows in the admission transaction.

        The lease becomes ``cancelled`` under the controller, and the owner Run,
        unless already ended, fails ``erase_fenced``: neither can act again, so a
        late provider reply is neither stored nor settled.
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
            rows.append(RowWrite("run", writer.parse_row("run", _cancelled_post_image(owner, now))))
        return rows

    def _retire(self, scope: ScopeV1, owner_run_id: str) -> None:
        """Tell the cancelled owner's pool invocation to stop at its next check."""
        from quotemem.service.bench_workers import BenchWork

        pool = self._writer._finalize_pool  # pyright: ignore[reportPrivateUsage]
        if pool is not None:
            pool.retire_owner(BenchWork(scope.scope_id, owner_run_id, "finalize"))

    def _current(self, scope: ScopeV1, memory_id: str) -> tuple[Row, Row]:
        item = self._writer.get_row("item", memory_id)
        if item is None or (item["scope_kind"], item["scope_id"]) != (
                scope.scope_kind, scope.scope_id):
            raise QuoteMemError("scope_denied", "The memory is not in this scope")
        head = self._writer.get_row("revision", item["current_revision_id"])
        if head is None:
            raise QuoteMemError("service_unhealthy", "The memory lost its current revision")
        return item, head

    def _bind_selection(
        self, scope: ScopeV1, principal_id: str, session: str, item: Row, head: Row,
        erase_epoch: int, *, seed: ForgetSeedV1 | None = None,
    ) -> ContentBindingDigest:
        """A memory forget's selection; a bundled memory's seed joins it."""
        selection: dict[str, object] = {
            "scope": scope.scope_id, "principal": principal_id, "session": session,
            "memory": item["object_id"], "revision": head["object_id"],
            "content_hash": head["content_hash"], "erase_epoch": erase_epoch,
        }
        if seed is not None:
            selection["seed"] = seed.model_dump(mode="json")
        return self._writer.bind_content("forget-selection", selection)

    def _bind_span_selection(
        self, scope: ScopeV1, principal_id: str, session: str, spans: Sequence[QuoteSpanV1],
        seed: ForgetSeedV1, erase_epoch: int,
    ) -> ContentBindingDigest:
        """A forget of sentences' selection: its sentences, its seed and the scope's epoch."""
        return self._writer.bind_content("forget-span-selection", {
            "scope": scope.scope_id, "principal": principal_id, "session": session,
            "spans": [span.model_dump(mode="json") for span in spans],
            "seed": seed.model_dump(mode="json"), "erase_epoch": erase_epoch,
        })

    def _seed_tombstone(
        self, scope: ScopeV1, seed: ForgetSeedV1,
    ) -> tuple[dict[tuple[str, str], str], set[str]]:
        """A seed's one tombstone: a stored one of its entity and statement, else a new id
        (unique key, as `_tombstone_ids`)."""
        existing = self._seed_stored(scope, seed)
        pair = (seed.entity_id, statement_digest(seed.statement))
        if existing is not None:
            return {pair: existing}, set()
        created = new_ulid()
        return {pair: created}, {created}

    def _seed_stored(self, scope: ScopeV1, seed: ForgetSeedV1) -> str | None:
        """The stored tombstone of a seed's entity and statement, or None."""
        digest = statement_digest(seed.statement)
        return next((row["object_id"] for row in self._pages(TombstoneRows(
            scope_kind=scope.scope_kind, scope_id=scope.scope_id, statement_digest=digest))
            if row["entity_id"] == seed.entity_id), None)

    def _unreplaced(
        self, versions: Sequence[ForgetVersionSelection], replacers: Sequence[str],
    ) -> list[ForgetVersionSelection]:
        """The versions of D that are no replacer's (``_chain_parts``): those get tombstones."""
        if not replacers:
            return list(versions)
        rows = [self._writer.get_row("revision", version.revision_id) for version in versions]
        return [version for version, row in zip(versions, rows, strict=True)
                if row is None or row["memory_id"] not in replacers]

    def _unseeded(
        self, versions: Sequence[ForgetVersionSelection], memory_id: str,
    ) -> list[ForgetVersionSelection]:
        """The versions of D a bundled memory's seed does not stand for (``_seed_stands_for``):
        all but the head's statement. An earlier statement of the memory ("The user lives in
        Beijing." before a correction), a chain memory's and a reworded head's may state another
        forgotten fact, so each keeps its own tombstone."""
        rows = [self._writer.get_row("revision", version.revision_id) for version in versions]
        stands = _seed_stands_for([row for row in rows if row is not None], memory_id)
        return [version for version, row in zip(versions, rows, strict=True)
                if row is None or row["object_id"] not in stands]

    def _with_seed(
        self, scope: ScopeV1, seed: ForgetSeedV1, tombstones: Mapping[tuple[str, str], str],
        created: set[str],
    ) -> tuple[dict[tuple[str, str], str], set[str]]:
        """The tombstones of D with a bundled memory's seed's. A new seed
        tombstone takes the last of the created ids, the versions the others in their order,
        so admission part (b) pairs them as admission did (`_apply_effects`). A seed of a version's
        own pair is that version's tombstone."""
        pair = (seed.entity_id, statement_digest(seed.statement))
        if pair in tombstones:
            return dict(tombstones), set(created)
        seeded, fresh = self._seed_tombstone(scope, seed)
        if not fresh:
            return {**tombstones, **seeded}, set(created)
        ordered = sorted(created | fresh)
        rename = dict(zip(sorted(created), ordered[:-1], strict=True))
        merged = {key: rename.get(value, value) for key, value in tombstones.items()}
        merged[pair] = ordered[-1]
        return merged, set(ordered)

    def _expire(self, stored: Row, value: ForgetIntentV1, now: int) -> None:
        """Keep only an expired active intent's binding: its seed, the user's words of the fact
        and their vector, is dropped (``expired``). Under the Writer mutex."""
        if value.state != "active" or now < value.expires_at:
            return
        expired = ForgetIntentV1.model_validate(
            {**value.model_dump(mode="json"), "state": "expired", "seed": None})
        writer = self._writer
        control = writer.materialize_control(
            stored["key"], expired.model_dump(mode="json"), stored["version"] + 1, now)
        _, plan = writer.plan_rows(
            [RowWrite("control", control)], None)
        writer.write_plan(plan)

    def expire_intent(self, intent: str) -> None:
        """Expire one prepared intent if it expired unconfirmed (``_expire``); a confirmation
        found expired before its admission calls it."""
        writer = self._writer
        with writer.write_scope():
            stored = writer.get_control(intent)
            if stored is not None and intent.startswith("intent:"):
                self._expire(stored, ForgetIntentV1.model_validate(stored["value"]),
                             self._clock())

    def expire_intents(self, *, limit: int = MAX_PAGE_SIZE) -> int:
        """Sweep the workspace's prepared intents that expired unconfirmed (``_expire``).

        The agent runtime calls it periodically; an admission that meets one expires it as
        well. At most ``limit`` intents are expired per call, each under the Writer mutex.

        Returns:
            How many intents were expired.
        """
        writer = self._writer
        now = self._clock()
        done = 0
        after: str | None = None
        while done < limit:
            page = writer.indexed_reader().forget_intents(after=after, limit=MAX_PAGE_SIZE)
            for row in page:
                value = ForgetIntentV1.model_validate(row["value"])
                if value.state != "active" or now < value.expires_at:
                    continue
                with writer.write_scope():
                    current = writer.get_control(row["key"])
                    if current is None:
                        continue
                    fresh = ForgetIntentV1.model_validate(current["value"])
                    if fresh.state == "active" and self._clock() >= fresh.expires_at:
                        self._expire(current, fresh, self._clock())
                        done += 1
                if done >= limit:
                    break
            if len(page) < MAX_PAGE_SIZE:
                break
            after = cast(str, page[-1]["key"])
        return done

    def _tombstone_ids(
        self, scope: ScopeV1, versions: Iterable[Any],
    ) -> tuple[dict[tuple[str, str], str], set[str]]:
        """One tombstone per distinct pair, reusing a stored one (unique key)."""
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
        """The newest revision of each forgotten pair, from durable storage (for a forget of
        sentences, only the present wordings of the memories it rewords)."""
        versions: Sequence[ForgetVersionSelection] = ()
        if fence.memory_id is not None:
            versions, _, _ = self._chain_plan((fence.memory_id, *fence.chain_ids),
                                              fence.reworded_ids)
        versions = self._with_wordings(versions, fence.rewordings)
        rows = [self._writer.get_row("revision", version.revision_id) for version in versions]
        if any(row is None for row in rows):
            raise QuoteMemError("binding_mismatch", "A forgotten version went before its tombstone")
        return cast(list[Row], rows)

    def _creations(
        self, scope: ScopeV1, versions: Sequence[Row], created: Sequence[str],
    ) -> list[tuple[Row, str]]:
        """Pair each version without a stored tombstone with its planned new id.

        Admission assigned the sorted created ids to those versions in selection
        order (a new seed tombstone took the last, ``_with_seed``); before admission part (b)
        commits none exists, so the pairing is reproducible.
        """
        missing = [
            version for version in versions
            if not any(row["entity_id"] == version["snapshot"]["entity_id"]
                       for row in self._pages(TombstoneRows(
                           scope_kind=scope.scope_kind, scope_id=scope.scope_id,
                           statement_digest=version["snapshot"]["statement_digest"])))
        ]
        if len(missing) != len(created):
            raise QuoteMemError("binding_mismatch", "The forget plan lost a tombstone")
        return list(zip(missing, created, strict=True))

    def _open_txn(self, scope: ScopeV1, run_id: str | None, now: int) -> TxnRows:
        return TxnRows(self._writer, scope.scope_kind, scope.scope_id,
                       self._writer.get_counter("commit_seq") + 1, now, run_id or "")

    def _pages(self, query: Any) -> Iterable[Row]:
        return TxnRows(self._writer, "user", "", 0, 0, "").pages(query)

    def _revisions(self, txn: TxnRows, memory_id: str) -> list[Row]:
        """A memory's revisions as ``txn`` leaves them, in id order."""
        return txn.merged("revision", self._pages(RevisionRows(memory_id=memory_id)),
                          lambda row: row["memory_id"] == memory_id)

    def _run_row(
        self, run_id: str, kind: str, mode: str, grant: PrincipalGrant, scope: ScopeV1 | None,
        target: Mapping[str, Any], writer_epoch: int, now: int, intent: str, *,
        controller: str | None = None, erase_epoch: int | None = None,
        consent_epoch: int | None = None,
    ) -> dict[str, Any]:
        # There is no canonical forget payload (the admission is the intent's
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


def _within(span: QuoteSpanV1, ranges: Sequence[tuple[int, int]]) -> bool:
    """Whether ``ranges`` together cover every byte of ``span``."""
    position = span.byte_start
    for start, end in sorted(ranges):
        if start > position:
            break
        position = max(position, end)
        if position >= span.byte_end:
            return True
    return position >= span.byte_end


def _restated(
    entity_id: str, statement: str, vector: Sequence[float], space: str,
    spans: Sequence[QuoteSpanV1], *, sensitive: bool, about: Sequence[str] = (),
    modifier: bool = False,
) -> ForgottenCitation:
    """The citation of the sentences a forget withholds as restating what it forgets:
    each sentence a group of its own, stating ``statement``.

    ``plan_stating_spans`` plans it with the forgotten versions, so a survivor quoting a
    restating sentence keeps the part that states only its own fact ("my sister lives in
    Berlin" in "I drink green tea every morning and my sister lives in Berlin.") and the
    rest is withheld, exactly as for a sentence the forgotten head quotes.
    """
    return ForgottenCitation(
        entity_id=entity_id, names=frozenset(), groups=tuple((span,) for span in spans),
        about=frozenset(about) - {USER}, sensitive=sensitive, statement=statement,
        vector=tuple(vector), space=space, modifier=modifier)


def _seed_stands_for(rows: Sequence[Row], memory_id: str) -> set[str]:
    """The revisions of D a bundled memory's seed stands for: those of
    ``memory_id`` with its newest one's statement, the one the judge found bundled.

    Admission and its part (b) read the same rows (``_versions``), so both leave out the same
    versions and pair the created tombstone ids alike.
    """
    own = [row for row in rows if row["memory_id"] == memory_id]
    if not own:
        return set()
    head = max(own, key=lambda row: cast(int, row["version"]))["snapshot"]
    pair = (head["entity_id"], head["statement_digest"])
    return {cast(str, row["object_id"]) for row in own
            if (row["snapshot"]["entity_id"], row["snapshot"]["statement_digest"]) == pair}


def _items(memory_id: str | None, chain: Sequence[str]) -> tuple[str, ...]:
    """A forget's memories: the selected one and its chain (none for a forget of sentences)."""
    return () if memory_id is None else (memory_id, *chain)


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


def _wording_digests(fence: ErasureFenceV1) -> frozenset[str]:
    """The statement digests of the present wordings a forget tombstones only because it
    rewords their memories (``ForgetOwner._wording_versions``)."""
    return frozenset(wording.base_digest for wording in fence.rewordings
                     if wording.base_digest != statement_digest(wording.statement))


def _probe(wording: ForgetRewordV1, statement: str = "") -> str:
    """What a rewording of ``statement`` says besides its tie and the kinds standing for the
    values it leaves out (`kind_of`), which its support must carry (`_carried`): its new
    statement less them ("is an architect." of "The user's wife is an architect.", "Ruby goes
    to ." of "Ruby goes to an elementary school.")."""
    rest = wording.statement
    if wording.tie is not None:
        rest = re.sub(re.escape(wording.tie), " ", rest, flags=re.IGNORECASE)
    for value in left_out(statement, wording.statement) if statement else ():
        if (kind := kind_of(value)) is not None:
            rest = re.sub(rf"\b{re.escape(kind)}\b", " ", rest, flags=re.IGNORECASE)
    return " ".join(rest.split())


def _carried(statement: str, quoted: Sequence[str], *, any_script: bool,
             given: str = "", ever: str = "") -> bool:
    """Whether whole readable quotes carry what a statement says (as ``trimmed_support`` reads
    a kept part): an English statement over quotes in another script needs only its own Han
    words and the names it gives in them, but those the quotes' own wording ``given`` already
    gave on them ("Tai Chi" over "太极拳"); any other needs every word in them."""
    if not statement or not quoted:
        return False
    if any_script and english(statement) and not all(text.isascii() for text in quoted):
        needed = {word for word in clause_words(statement) if own_token(word)}
        return needed <= clause_words(" ".join(quoted)) and not unshown_names(
            statement, [*quoted, given])
    read = clause_words if any_script else support_words
    # A word the head said that none of its quotes ever showed as written (a romanised or
    # translated place, "Huangshan" for "黄山") is not asked of the kept parts.
    unshowable = read(given) - read(ever) if given and ever else frozenset[str]()
    return read(statement) - unshowable <= read(" ".join(quoted))


def _overlaps_any(span: QuoteSpanV1, spans: Sequence[QuoteSpanV1]) -> bool:
    return any(other.evidence_id == span.evidence_id and other.byte_start < span.byte_end
               and span.byte_start < other.byte_end for other in spans)


def _detail_requoted(txn: TxnRows, read: Readability, snapshot: MemorySnapshotV2,
                     forgotten: _Forgotten) -> bool:
    """Whether forgetting a detail requotes a dependent during deletion on the readable rest of its
    quotes, as `ForgetOwner._compensate` trims it under ``read``."""
    found = _detail_dependent(forgotten, snapshot.statement)
    if found is None or _trim_refused(txn, snapshot, found):
        return False
    return _trimmed_dependent(txn, read, snapshot, found) is not None


def _restated_over_han(txn: TxnRows, snapshot: MemorySnapshotV2, forgotten: _Forgotten) -> bool:
    """Whether an English dependent over Han quotes, which rests on no shared word, has a
    vector restating a forgotten statement: it keeps nothing (as ``clause_spans``)."""
    return forgotten.any_script and english(snapshot.statement) and any(
        not (_quoted_text(txn, span) or "").isascii()
        for span in snapshot.quote_spans) and restated_by_vector(
        snapshot.embedding, snapshot.embedding_space, [
            vector for statement, vector in zip(
                forgotten.statements, forgotten.vectors, strict=False)
            if not names_another(statement, snapshot.statement)])


def _trimmed_dependent(txn: TxnRows, read: Readability, snapshot: MemorySnapshotV2,
                       forgotten: _Forgotten) -> TrimmedSupport | None:
    """The readable parts of a dependent's groups that carry it
    and show nothing forgotten (``trimmed_support``)."""
    return trimmed_support(snapshot, read, forgotten=_forgotten_terms(
        forgotten, snapshot.statement), quoted=lambda span: _quoted_text(txn, span),
                           speakers=lambda span: _speaker_words(txn, span),
                           any_script=forgotten.any_script)


def _detail_dependent(forgotten: _Forgotten, statement: str) -> _Forgotten | None:
    """What a dependent stating ``statement`` is trimmed against when the forget is of a
    detail: the request asks a question, which states nothing a dependent
    could restate ("忘掉我的猫是什么颜色的" beside "The user's cat is named bob.", "忘掉我女儿在
    哪个学校上学" beside "The user has a daughter."), so it is no forgotten statement, and the
    withheld value's own bytes are what no readable part may show; None when the dependent's
    statement still shows an asked colour or breed ("The user's orange cat is named bob."): it
    is left unreadable. Any other forget is read as it is."""
    details = [detail for other in forgotten.statements
               if (detail := detail_question(other)) is not None]
    if not details:
        return forgotten
    if any(detail_values(statement, detail) for detail in details):
        return None
    kept = [index for index, other in enumerate(forgotten.statements)
            if detail_question(other) is None]
    return replace(forgotten, statements=tuple(forgotten.statements[index] for index in kept),
                   vectors=tuple(forgotten.vectors[index] for index in kept
                                 if index < len(forgotten.vectors)))


_VALUE_TOKEN: Final = re.compile(r"[^\s,;:()]*[\d@][^\s,;:()]*")
"""A token with a digit or an at sign: a number, a code or an address."""


def _contrast_values(statement: str, wording: ForgetRewordV1,
                     forgotten: _Forgotten) -> list[str]:
    """The forgotten values a rewording leaves out with the phrases naming them as replaced,
    when that is all it changes (`replaced_values`, `reworded_statement`): "14 Oak Street" of
    "The user lives at 27 Pine Lane now, not 14 Oak Street." reworded "The user lives at 27
    Pine Lane now.", "启明幼儿园" of "The user's son has transferred from 启明幼儿园 to
    蓝湾幼儿园." reworded "The user's son has transferred to 蓝湾幼儿园."; none for any other
    rewording. A name the wording leaves out with a clause or a departure naming it as
    replaced counts though no tombstone of this forget shows it in its script: the prepare
    planned the wording against every forget of the plan, which the first one may precede
    ("Qiming Kindergarten" of "... no longer attends Qiming Kindergarten and has transferred
    to Lanwan Kindergarten." under a forget tombstoned by "启明幼儿园")."""
    if wording.tie is not None:
        return []
    values = replaced_values(statement, [*forgotten.statements, *forgotten.said]) or [
        value for value in left_out(statement, wording.statement)
        if replaced_values(statement, [value])]
    found = reworded_statement(statement, names={}, values=values) if values else None
    if found is None or found[1] is not None or (
            canonical_statement(found[0]) != canonical_statement(wording.statement)):
        return []
    return values


def _value_terms(texts: Sequence[str], shown: Callable[[str], frozenset[str]],
                 ) -> frozenset[str]:
    """The words of the names, values and numbers ``texts`` show (`forgotten_values` of
    each, and each token with a digit or an at sign), never their other words."""
    return frozenset(term for text in texts
                     for value in (*forgotten_values(text, [text]), *_VALUE_TOKEN.findall(text))
                     for term in shown(value))


def _forgotten_terms(forgotten: _Forgotten, statement: str) -> frozenset[str]:
    """What a dependent stating ``statement`` may not show: the words of S, and
    what each tombstone's statement says of its subject (``predicate_words``); in any script a
    Han text is compared by its terms (``fact_terms``), not its characters, and a statement
    that only names someone forgets the name alone (``naming_value``, as the clause cut reads
    it): "dog" of "The user's dog is named 豆豆." is no forgotten word of "The user has a cat
    and a dog."."""
    if forgotten.any_script:
        return forgotten.terms.union(*(
            (_name_words(name) | fact_terms(name))
            if english(statement) and english(other)
            and (name := naming_value(other)) is not None
            else said_terms(other, statement, modifier=forgotten.modifier)
            for other in forgotten.statements))
    return forgotten.words.union(*(
        predicate_words(other, statement, modifier=forgotten.modifier)
        for other in forgotten.statements))


@dataclass(frozen=True)
class _Forgotten:
    """What a forget's trimmed dependents may not show.

    Attributes:
        words: ``support_words`` of every span of S.
        statements: The statements of the forget's tombstones.
        any_script: Whether the trim cuts a piece in any script (``ErasureFenceV1.any_script``).
        terms: ``fact_terms`` of every span of S, which the trim reads given ``any_script``.
        vectors: The tombstones' statement vectors, each with its embedding space.
        modifier: Whether the forget is of a modifier alone (``ForgetIntentV1.modifier_only``).
    """

    words: frozenset[str]
    statements: tuple[str, ...]
    any_script: bool = False
    terms: frozenset[str] = frozenset()
    vectors: tuple[tuple[tuple[float, ...], str], ...] = ()
    modifier: bool = False
    name_asked: bool = False
    """Whether the user asked to forget a name: a rewording then reads only the name's words
    of what is withheld as forgotten (``ForgetOwner._wording_words``)."""
    kept: frozenset[str] = frozenset()
    """The names the user kept of a forgotten statement naming an asked value beside them
    (``ForgetOwner._kept_names``): no rewording reads them as forgotten."""
    said: tuple[str, ...] = ()
    """The texts of S: a contrast tail naming a value one shows names a forgotten value
    (`_contrast_values`), as one naming a value of a tombstone's statement does."""


def _trim_refused(txn: TxnRows, snapshot: MemorySnapshotV2, forgotten: _Forgotten) -> bool:
    """Whether privacy compensation refuses to trim a dependent whatever its readable parts: it may
    restate a forgotten possession (``possession_entailed``), or, English over a quote in
    another script, which no shared word grounds, its vector restates a forgotten statement
    (`_restated_over_han`)."""
    return any(possession_entailed(statement, snapshot.statement, modifier=forgotten.modifier)
               for statement in forgotten.statements) or _restated_over_han(
        txn, snapshot, forgotten)


@dataclass(frozen=True)
class _PlannedPart:
    """An Evidence header as a plan leaves it, the plan's withheld ranges added
    (``_planned_read``)."""

    object_id: str
    scope_kind: str
    scope_id: str
    state: str
    route_state: str
    expires_at: int | None
    withheld_spans: tuple[ByteRange, ...]


def _planned_read(read: Readability,
                  held: Mapping[str, Sequence[tuple[int, int]]]) -> Readability:
    """The scope's readability once the ``held`` ranges are withheld too: what privacy compensation
    reads after a forget withholding them (a preview's plan)."""

    def evidence(evidence_id: str) -> EvidenceHeader | None:
        row = read.evidence(evidence_id)
        if row is None or not held.get(evidence_id):
            return row
        return _PlannedPart(row.object_id, row.scope_kind, row.scope_id, row.state,
                            row.route_state, row.expires_at, (*row.withheld_spans, *(
                                QuoteSpanV1(evidence_id=evidence_id, byte_start=start,
                                            byte_end=end)
                                for start, end in held[evidence_id])))

    return Readability(read.scope_kind, read.scope_id, read.now, evidence)


@dataclass(frozen=True)
class _Version:
    """A revision row as privacy compensation reads it."""

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


def _plan_selection(item: Row, revisions: Sequence[Row]) -> Any:
    """Forget admission selection over the complete chain (``plan_forget_admission``)."""
    return plan_forget_admission(
        MemoryRow.model_validate(item), [RevisionRowV2.model_validate(row) for row in revisions],
    )


def _is_person(txn: TxnRows, entity_id: str) -> bool:
    """Whether an entity's broad type is "person" (participants and people)."""
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


def _withheld_runs(txn: TxnRows, read: Readability, spans: Sequence[QuoteSpanV1]) -> set[str]:
    """The Han runs of the bytes ``read`` withholds inside ``spans`` ("上海" of "我上个月从上海搬
    到了杭州" when the plan withholds it there)."""
    found: set[str] = set()
    for span in spans:
        header = read.evidence(span.evidence_id)
        row = txn.get("evidence", span.evidence_id)
        if header is None or row is None:
            continue
        data = str(row["text"]).encode("utf-8")
        for part in header.withheld_spans:
            start = max(part.byte_start, span.byte_start) - row["byte_start"]
            end = min(part.byte_end, span.byte_end) - row["byte_start"]
            if 0 <= start < end <= len(data):
                found.update(_HAN_RUNS.findall(data[start:end].decode("utf-8", "ignore")))
    # Overlapping withholdings ("我弟弟叫李伟" and its "李伟") are one run.
    return {run for run in found if not any(run in other and run != other for other in found)}


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
    """The verdicts that admitted a memory's versions (a rewording carries its cause's)."""
    return frozenset(ref for revision in revisions
                     if (ref := revision["snapshot"].get("verdict_ref")) is not None)


def _name_words(name: str) -> frozenset[str]:
    """The words of a name in other scripts than Han, Kana or Hangul: a Han name is compared
    by its terms (``fact_terms``), so "小李" forbids "小李", not the "小" of "小王"."""
    return frozenset(word for word in support_words(name) if not own_token(word))


def naming_only(snapshot: MemorySnapshotV2) -> bool:
    """Whether a memory names its single non-user entity and anchors nothing else ("A person
    is named Alice.")."""
    return snapshot.entity_id != USER and not snapshot.about_entity_ids and bool(
        snapshot.naming) and all(name.entity_id == snapshot.entity_id for name in snapshot.naming)


_NAME_ASKED: Final = re.compile(r"\bnames?\b|名字|姓名|叫什么", re.IGNORECASE)
"""Words by which a request asks for a name ("forget Ben's name", "忘掉小李的名字")."""

_WHOLE: Final = re.compile(
    r"\b(?:everything|all|anything)\s+(?:\w+\s+)?(?:about|regarding|concerning|to do with)\b"
    r"|一切|所有|全部", re.IGNORECASE)
"""A request for everything about someone or something ("forget everything about my ex",
"忘掉关于我前任的一切"): it takes the whole entity, its name with it."""


def asks_everything(text: str) -> bool:
    """Whether the user's words ask to forget everything about someone or something
    (``_WHOLE``): its name goes whole, as a name asked does."""
    return _WHOLE.search(text) is not None


_EN_LINKS = r"[a-z]+(?:\s+[a-z]+){0,2}(?:['\u2019]s\s+[a-z]+(?:\s+[a-z]+){0,2}){0,2}"
_EN_ASKED_CHAINS: Final = (
    re.compile(rf"\bmy\s+(?P<chain>{_EN_LINKS})['\u2019]s\s+(?:(?:first|last|full)\s+)?name\b",
               re.IGNORECASE),
    re.compile(rf"\bthe\s+(?:(?:first|last|full)\s+)?name\s+of\s+my\s+(?P<chain>{_EN_LINKS})",
               re.IGNORECASE),
)
"""A request asking the name of something of the user's by a possessive chain: "my son's
school's name", "the name of my son's school", "my sister's husband's name"."""
_ZH_ASKED_CHAIN: Final = re.compile(
    r"我的?(?P<chain>[\u4e00-\u9fff]{1,12}?)的?(?:名字|全名|叫什么|叫啥)")
_HAN_TIE_WORDS: Final = tuple(sorted((word for word in TIES if not word.isascii()),
                                     key=len, reverse=True))
_FILLER: Final = frozenset({"please", "too", "now", "thanks", "again"})


def asked_links(text: str) -> tuple[frozenset[str], ...] | None:
    """The ties of each link of the possessive chain whose last link's name ``text`` asks,
    outer first: "the name of my son's school" gives ({son}, {}), the school being named by
    no tie; "my sister's husband's name" ({sister ties}, {husband}); "忘掉我女儿钢琴老师的名字"
    ({daughter}, {}); "forget my brother's name" ({brother ties},). None when no chain is
    found, or a link the chain opens with names no tie."""
    for pattern in _EN_ASKED_CHAINS:
        match = pattern.search(text)
        if match is None:
            continue
        links: list[frozenset[str]] = []
        for link in re.split(r"['\u2019]s\s+", match["chain"].strip()):
            words = [word for word in link.casefold().split() if word not in _FILLER]
            links.append(TIES.get(words[-1], frozenset()) if words else frozenset())
        return tuple(links) if links and links[0] else None
    match = _ZH_ASKED_CHAIN.search(text)
    if match is None:
        return None
    links = []
    for piece in match["chain"].split("的"):
        while piece:
            word = next((tie for tie in _HAN_TIE_WORDS if piece.startswith(tie)), None)
            if word is None:
                links.append(frozenset())  # a role or a thing ("钢琴老师", "学校")
                break
            links.append(TIES[word])
            piece = piece[len(word):]
    return tuple(links) if links and links[0] else None


ForgottenFact = tuple[str, str, Sequence[str]]
"""A fact a forget takes: its statement, its primary entity and its other anchored entities."""


def names_tied(facts: Iterable[ForgottenFact]) -> dict[tuple[str, str], frozenset[str]]:
    """The ties each name is given to the user by a statement that only ties it
    (``naming_tie``: "Leo is the user's son." gives "leo" the son's), by the entity the
    statement is about and casefolded name: a cousin and a colleague both called 志强 have ties
    of their own. A statement about no entity but the user ties the name under ``""``."""
    found: dict[tuple[str, str], frozenset[str]] = {}
    for statement, entity, about in facts:
        tie = naming_tie(statement)
        if tie is None:
            continue
        for key in ({entity, *about} - {USER, ""}) or {""}:
            pair = (key, tie[1].casefold())
            found[pair] = found.get(pair, frozenset()) | tie[0]
    return found


_NAME_WORDS: Final = r"(?:(?:first|last|full|real|given|family)\s+)?names?\b"
_EN_POSSESSOR: Final = re.compile(rf"['\u2019]s\s+(?!{_NAME_WORDS})\w", re.IGNORECASE)
"""What makes a Latin name the possessor of a thing the request asks ("Ben's wife"), not the
one whose name it asks ("Ben's name")."""
_EN_BEARER_AFTER: Final = re.compile(rf"['\u2019]s\s+{_NAME_WORDS}", re.IGNORECASE)
_EN_BEARER_BEFORE: Final = re.compile(r"\bnames?\s+of\s+$", re.IGNORECASE)
_ZH_BEARER: Final = re.compile(r"的?(?:名字|全名|姓名|叫什么|叫啥)")
_ZH_POSSESSOR: Final = re.compile(
    r"的(?!名字|全名|姓名|事|一切|所有|全部|信息|资料)[\u3400-\u4dbf\u4e00-\u9fff]")
"""What makes a Han name the possessor of a thing the request asks ("小明的狗"), besides a
tie word right after it ("小明老婆")."""


def _occurrences(name: str, text: str, *, written: bool = False) -> list[tuple[str, str]]:
    """The text before and after each occurrence of ``name`` in ``text`` as whole tokens, in
    any case; with ``written``, a capitalized Latin name only as written ("Will", not the
    "will" of "I will move")."""
    if not name.strip():
        return []
    if own_token(name[0]):
        pattern = re.escape(name)
    else:
        pattern = (r"(?<![0-9A-Za-z])" + r"\s+".join(re.escape(word) for word in name.split())
                   + r"(?![0-9A-Za-z])")
    flags = 0 if written and name != name.casefold() else re.IGNORECASE
    return [(text[:match.start()], text[match.end():])
            for match in re.finditer(pattern, text, flags)]


def _possessor(after: str) -> bool:
    """Whether the text after a name in a request makes it the possessor of what the request
    asks ("Ben's wife", "小明老婆", "小明的狗"), not the one it asks or gives."""
    return (_EN_POSSESSOR.match(after) is not None or _ZH_POSSESSOR.match(after) is not None
            or after.startswith(_HAN_TIE_WORDS))


def _bearer(name: str, text: str) -> bool:
    """Whether a request asks the name of the one called ``name`` ("forget Ben's name", "the
    name of Ben", "忘掉小李的名字"), a capitalized Latin name as written."""
    return any(_EN_BEARER_AFTER.match(after) is not None or _ZH_BEARER.match(after) is not None
               or (_EN_BEARER_BEFORE.search(before) is not None and not _possessor(after))
               for before, after in _occurrences(name, text, written=True))


def _owner_in(name: str, statement: str) -> bool:
    """Whether a statement gives ``name`` as a possessor ("Leo" of "Willowbrook Primary is
    Leo's school.")."""
    return re.search(rf"(?<![\w'\u2019]){re.escape(name)}['\u2019]s\b", statement,
                     re.IGNORECASE) is not None


def bare_name_stays(snapshot: MemorySnapshotV2, requests: Sequence[str], names: Set[str],
                    forgotten: Sequence[ForgottenFact],
                    ties: Mapping[tuple[str, str], frozenset[str]], *,
                    told: bool) -> bool:
    """Whether a bare naming memory every quote of which lies in what a forget of the
    ``forgotten`` facts for ``requests``, the user's words, withholds or forgets keeps its
    name: it is no residue of the forgotten fact (``StatementForget._leftovers``' second
    kind). ``names``: the names the scope's memories give. ``told``: whether another memory
    the forget leaves readable is about its entity.

    * A request giving the name keeps it never; one giving it only as the possessor of what
      it asks ("forget the name of Ben's wife", "忘掉小明老婆的名字", "forget Leo's age")
      keeps it always: its entity is not what is asked.
    * A request for everything about something (``_WHOLE``: "forget everything about my
      ex") keeps it never: the whole entity may be asked.
    * A request asking a name (``_NAME_ASKED``) forgets that name: it keeps a name of the
      scope's other ``names`` when it asks that one ("forget Ben's name" keeps "A person is
      named Alice.", ``_bearer``), and one no forgotten fact shows or anchors, another's
      beside them (with no fact, a forget of sentences alone, the words decide alone);
      asked by a possessive chain (``asked_links``), it asks the
      name of the chain's last link, so the name of the user's person a thing of whom is
      asked ("the name of my son's school", "我女儿钢琴老师的名字") stays when a statement
      ties it to them (``ties``: "Leo is the user's son."), and a request for a person
      ("forget my son's name") keeps only a name tied to another tie, unless a forgotten
      fact is about its entity: then it is the asked one's, whoever else bears it. Any other
      name, or one asked by a chain not read, goes: which entity it means may be one the name
      rules do not find.
    * A request for no name forgets the facts. A name of the thing a forgotten fact gives as
      another entity's goes, whichever entity the fact is about ("Willowbrook Primary" of
      "Leo's school is Willowbrook Primary." and of "Willowbrook Primary is Leo's school.",
      ``_owner_in``), and so does a name of an entity a fact without a possessor anchors
      beside its subject. A name none of them shows is another's beside them and stays
      ("A person is named 小王." beside a forgotten "A person is named 小李.", or the name a
      forgotten nickname was given beside, ``kept_beside``); one they show stays only when
      the entity is ``told`` beyond them and no fact shows it as another entity's but as
      its possessor ("Leo", the subject or the possessor of the forgotten fact, stays).
    """
    own = {entry.name for entry in snapshot.naming}
    mentioned = [_possessor(after) for name in own for text in requests
                 for _, after in _occurrences(name, text)]
    if mentioned:
        return all(mentioned)
    if any(asks_everything(text) for text in requests):
        return False
    asking = [text for text in requests if _NAME_ASKED.search(text)]
    if asking:
        if any(_bearer(name, text) for name in names - own for text in asking) or (
                forgotten and not any(
                    snapshot.entity_id in (entity, *about) or any(
                        token_contained(name, statement) for name in own)
                    for statement, entity, about in forgotten)):
            return True
        if any(entity == snapshot.entity_id for _, entity, _ in forgotten) and all(
                (links := asked_links(text)) is not None and len(links) == 1
                for text in asking):
            # The asked person's own name: a forgotten fact is about its entity and the
            # request asks the name of the user's one ("忘掉我表哥的名字" beside "志强 is the
            # user's older male cousin."), however another of that name is tied ("志强 is the
            # user's colleague.").
            return False
        # The ties of this entity's name (``names_tied``), and those no entity holds.
        mine = frozenset[str]().union(*(
            ties.get((key, name.casefold()), frozenset())
            for name in own for key in (snapshot.entity_id, "")))
        for text in asking:
            links = asked_links(text)
            if not mine or links is None or len(links) > 2 or (
                    len(links) == 1 and not mine.isdisjoint(links[0])) or (
                    len(links) == 2 and mine.isdisjoint(links[0])):
                return False
        return True

    def owns(statement: str) -> bool:
        return any(_owner_in(name, statement) for name in own)

    if any(not owns(statement) and (
            (entity != snapshot.entity_id and snapshot.entity_id in about)
            or (entity == snapshot.entity_id
                and any(_owner_in(name, statement) for name in names - own)))
           for statement, entity, about in forgotten):
        return False

    def shows(statement: str) -> bool:
        # The name a forgotten nickname was given beside ("林可欣" of "林可欣 is also known
        # as 安安.") is no part of it (``kept_beside``).
        return any(token_contained(name, statement) and not kept_beside(name, statement)
                   for name in own)

    if not any(shows(statement) for statement, _, _ in forgotten):
        return True
    return told and not any(entity != snapshot.entity_id and shows(statement)
                            and not owns(statement) for statement, entity, _ in forgotten)


def _entities(snapshot: MemorySnapshotV2) -> set[str]:
    """The entities a version names; a content-free (erased) version names none."""
    if snapshot.erased:
        return set()
    return {snapshot.entity_id, *snapshot.about_entity_ids,
            *(entry.entity_id for entry in snapshot.naming)}


def _count_newly_withheld(spans: Sequence[QuoteSpanV1], touched: set[str]) -> int:
    """The spans of a batch whose Evidence row this transaction withheld anew.

    A span an earlier forget already withheld leaves its row unchanged, so the
    receipt counts only the rows and spans this forget changed.
    """
    return sum(1 for span in spans if span.evidence_id in touched)


def _overlaps(
    groups: Sequence[Sequence[QuoteSpanV1]], spans: Mapping[str, Sequence[QuoteSpanV1]],
) -> bool:
    """Whether a quote of ``groups`` intersects one of ``spans`` (by Evidence id; ``p_span``)."""
    return any(
        held.byte_start < span.byte_end and span.byte_start < held.byte_end
        for group in groups for span in group for held in spans.get(span.evidence_id, ())
    )


def _usable(groups: Sequence[Sequence[QuoteSpanV1]], read: Readability) -> bool:
    """Whether one group has every span passing ``p_span`` (``p_group``)."""
    return any(read.p_group(tuple(group), tuple(range(len(group)))) for group in groups)


def _adopt(writer: Writer, scope: ScopeV1, controller_id: str) -> None:
    """Adopt a surviving fence's controller and worker under this boot's Writer epoch.

    Boot step 13: a fence that survived a restart names Runs of an
    earlier Writer epoch, whose tickets the Writer refuses (``lease_lost``).
    Only ``writer_epoch`` changes, as in the other startup adoptions; the
    fence, its frozen plan and every timestamp stay as they are. A Run a refused
    forget left ``manual_attention`` (`_attend`) runs again: the boot, or the repair path
    (`resume_attended`), is its retry.

    Raises:
        QuoteMemError: ``binding_mismatch`` for a Run the fence does not name or
            that already ended, ``lease_lost`` for a Run of a later epoch.
    """
    with writer.write_scope():
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
                "queued", "running", "manual_attention",
            }:
                raise QuoteMemError("binding_mismatch", "A surviving fence lost its running Runs")
            if run["writer_epoch"] > epoch:
                raise QuoteMemError("lease_lost", "A surviving fence names a later Writer epoch")
            if run["status"] == "manual_attention":
                rows.append(RowWrite("run", writer.parse_row("run", dict(
                    run, writer_epoch=epoch, status="running", failure_code=None))))
            elif run["writer_epoch"] != epoch:
                rows.append(RowWrite("run", writer.parse_row(
                    "run", dict(run, writer_epoch=epoch))))
        if rows:
            _, plan = writer.plan_rows(rows, None)
            writer.write_plan(plan)


def workspace_embedding(
    writer: Writer, embedder: EmbeddingProvider, texts: Sequence[str],
) -> tuple[str, list[tuple[float, ...]]]:
    """The texts' vectors in the workspace's active embedding space, by one call: that space
    and one vector per text. A complete forget's statement vectors use it, and its run's
    remainders (``ForgetOwner.embed_remainders_with``).

    Raises:
        QuoteMemError: ``service_unhealthy`` without a space; ``embedding_space_mismatch`` when the
            space has no dimension or the provider answers in another; the provider's own.
    """
    stored = writer.get_control("policy:embedding_space")
    if stored is None:
        raise QuoteMemError("service_unhealthy", "Forget by statement has no embedding space")
    space = ActiveEmbeddingSpaceV1.model_validate(stored["value"]).embedding_space
    model, _, dimensions = space.rsplit("@", 2)
    if dimensions not in ("1024", "2048"):
        raise QuoteMemError("embedding_space_mismatch", "The embedding space has no dimension")
    result = embedder.embed(EmbeddingRequest(
        dimensions=cast(Literal[1024, 2048], int(dimensions)), texts=list(texts),
        timeout_ms=DEFAULT_PROVIDER_TIMEOUT_MS, retry_policy=default_retry_policy()))
    if result.resolved_model != model or len(result.vectors) != len(texts):
        raise QuoteMemError("embedding_space_mismatch",
                            "The statement's vector is of another space")
    return space, [tuple(vector) for vector in result.vectors]


def recover_forgets(owner: ForgetOwner, scopes: Iterable[ScopeV1]) -> list[str]:
    """Resume every forget whose fence survived a restart (boot step 13).

    Each fence's Runs are first adopted under the current Writer epoch. A forget that
    cannot resume keeps its fence, which holds its own scope alone; only a failure of the
    runtime itself (``is_runtime_failure``, a lost Writer lease) fails the boot.

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
                try:
                    _adopt(writer, scope, controller)
                    with background():
                        owner.resume(scope, controller)
                except Exception as error:
                    if not _forget_alone(error):
                        raise
                    # Its fence keeps this scope closed; every other scope still opens.
                    if _transient(error):
                        logging.getLogger(__name__).warning(
                            "Forget recovery refused for scope %s, controller %s: %s",
                            scope.scope_id, controller,
                            canonical_json(failure_reason("forget_recovery", error).to_dict()),
                        )
                    else:
                        _attend(writer, scope, controller,
                                owner._clock(),  # pyright: ignore[reportPrivateUsage]
                                "forget_recovery", error)
                    _drop_frames(error)
                    continue
                resumed.append(controller)
    return resumed


def _attend(writer: Writer, scope: ScopeV1, controller_id: str, now: int, site: str,
            error: Exception) -> None:
    """Leave a forget refused for good to the repair path: its fence stays, so nothing it
    withholds is read, and its controller and worker go ``manual_attention``
    (``parity_mismatch``) until the next boot or a repair (``quotemem repair``,
    `resume_attended`) runs them again. The warning names the scope and the controller,
    never content."""
    logging.getLogger(__name__).warning(
        "A forget needs repair: scope %s, controller %s: %s", scope.scope_id, controller_id,
        canonical_json(failure_reason(site, error).to_dict()),
    )
    with writer.write_scope():
        control = writer.get_control(fence_key(scope.scope_id, controller_id))
        if control is None:
            return
        fence = ErasureFenceV1.model_validate(control["value"])
        rows: list[RowWrite] = []
        for run_id in (fence.worker_run_id, fence.controller_run_id):
            run = writer.get_row("run", run_id)
            if run is not None and owns_erasure_fence(run, control) and (
                    run["status"] == "running"):
                rows.append(RowWrite("run", writer.parse_row("run", dict(
                    run, status="manual_attention", failure_code="parity_mismatch",
                    updated_at=now))))
        if rows:
            _, plan = writer.plan_rows(rows, None)
            writer.write_plan(plan)


def attended_forgets(writer: Writer, scope: ScopeV1) -> list[str]:
    """The controllers of the scope's fenced forgets left ``manual_attention`` (`_attend`)."""
    found: list[str] = []
    with writer.write_scope():
        for row in writer.indexed_reader().scope_fences(scope.scope_id):
            key = cast(str, row["key"])
            if not key.startswith(f"erasure:{scope.scope_id}:"):
                continue
            fence = ErasureFenceV1.model_validate(row["value"])
            run = writer.get_row("run", fence.controller_run_id)
            if run is not None and run["status"] == "manual_attention":
                found.append(key.rsplit(":", 1)[1])
    return found


def resume_attended(owner: ForgetOwner, scope: ScopeV1) -> list[str]:
    """Run each of the scope's forgets left ``manual_attention`` again (the repair path);
    the controllers that completed. A forget refused again is left to repair again.

    Raises:
        QuoteMemError: the refusal of a forget that did not complete.
    """
    writer = owner._writer  # pyright: ignore[reportPrivateUsage]
    done: list[str] = []
    for controller in attended_forgets(writer, scope):
        _adopt(writer, scope, controller)
        try:
            owner.resume(scope, controller)
        except Exception as error:
            if _forget_alone(error) and not _transient(error):
                _attend(writer, scope, controller,
                        owner._clock(),  # pyright: ignore[reportPrivateUsage]
                        "forget_repair", error)
            raise
        done.append(controller)
    return done


def _drop_frames(error: BaseException) -> None:
    """Drop the traceback of ``error`` and of every exception it chains to, by cause and by
    context: a refused recovery's frames hold the boot's own publisher, which must go
    before step 14."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        held = pending.pop()
        if id(held) in seen:
            continue
        seen.add(id(held))
        held.__traceback__ = None
        pending.extend(link for link in (held.__cause__, held.__context__) if link is not None)


def abandon_orphan_controllers(writer: Writer, *, now: int) -> list[str]:
    """End every erasure controller a crash left before its admission (boot step 13).

    ``ForgetOwner._write_controller`` writes the controller in its own transaction, before
    the admission transaction that names it in the intent, the fence, the worker and
    the cancellation. A crash between the two leaves a running controller that holds
    nothing: its intent is spent, so the client prepares again. Its zero-effect proof:
    no fence, no Control it wrote and no child Run names it. Proved, it
    fails ``lease_lost``; not proved, it is left ``manual_attention``
    ``parity_mismatch`` and never retried. Fenced controllers resume through
    `recover_forgets`, which runs first.

    Returns:
        The controllers ended ``lease_lost``, in id order.

    Raises:
        QuoteMemError: ``lease_lost`` for a controller of a later Writer epoch.
    """
    abandoned: list[str] = []
    with writer.write_scope():
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
                raise QuoteMemError("lease_lost",
                                    "An erasure controller names a later Writer epoch")
            if run["object_id"] in named:
                continue
            if reader.run_children(run["object_id"], after=None, limit=1):
                ended = dict(run, writer_epoch=epoch, status="manual_attention",
                             failure_code="parity_mismatch", updated_at=now)
            else:
                ended = dict(run, writer_epoch=epoch, status="failed", failure_code="lease_lost",
                             updated_at=now, finished_at=now)
                abandoned.append(run["object_id"])
            rows.append(RowWrite("run", writer.parse_row("run", ended)))
        if rows:
            _, plan = writer.plan_rows(rows, None)
            writer.write_plan(plan)
    return abandoned


__all__ = [
    "FORGET_RECEIPT_KINDS", "FinalScan", "ForgetOutcome", "ForgetOwner", "ForgetPrepared",
    "SpansPrepared",
    "MAX_TX_ROWS", "abandon_orphan_controllers", "check_controller_artifact", "fence_key",
    "attended_forgets", "intent_key", "owns_erasure_fence", "recover_forgets",
    "resume_attended",
]
