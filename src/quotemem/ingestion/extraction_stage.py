"""The owner's extraction stage: units to the barrier, settlement, vectors, proposals.

The Add owner hands the stage an `ExtractionOwner`. The stage packs the
owner's frozen prefix into capacity-proven units and admits every unit
up to the barrier, at most ``extraction_concurrency`` at a time
(`admit_units`). A unit is idempotent: a stored reply is replayed with no
call, a stored context is rebuilt and its call resumed, and a new context is
stored before its call.

Each unit gets one extraction call (`extract`). A call whose physical
sends are exhausted with no accepted reply ends its unit with a provider
refusal: the unit yields no fact and its messages stay as Evidence.
A call refused for its output length (``truncated`` or ``output_bound``)
is never sent again: its unit splits into two halves of whole parts, each a
smaller request extracted as its own unit (`split_unit`), down to single parts. A
one-part piece that is cut again ends like a spent unit, so only the parts a provider
keeps cutting short stay unextracted, a planned unit of n parts gets at most 2n - 1
first calls, and the Add completes on what the other pieces extracted.
A piece starts only while the owner's deadline still fits its cut parent's duration
again and the reserve (``admits_piece``): a piece is smaller than its parent, so its
call is expected no longer. Each such end is
written to the call's allowance before the stage acts on it (``UnitCallEndV1``), so a
replay takes the same path without a send.
An explicit resume of an extraction-provider pause first grants each exhausted
unanswered call one additional send; a failed resend follows that same
refusal path. Automatic recovery grants nothing. A contract-valid first
reply gets one fresh call of the identical request, under the owner-deadline reserve,
when it lost facts a fresh sample restores (`needs_fresh_call`). A reply with zero facts
(including a reply whose every statement is blank and refused ``statement_invalid``)
gets it when the unit's fresh sources hold at least `ZERO_PROPOSAL_RETRY_WORDS` words that
no assistant wrote (byte-identical requests have returned a
whole dense unit without a single fact), or a readable fresh non-assistant stretch
contains a declarative named possessive phrase (English, or another language's closed
phrase in `quotemem.ingestion.resend_lexicon`), or a closed personal assertion in that
module's closed verb/content rule, or when it declares a
new entity or an unresolved referent: a declaration exists only to anchor a fact, so such
a reply dropped the facts it prepared, whatever the unit's length. A reply with facts gets
it when the unit renders no authenticated-self source and a fact refers to the user while
quoting a source a participant speaks: the user says nothing in the unit, so the extractor
wrote the user for that participant, and admission refuses the fact (``user_unattested``).
The fresh call's reply stands, even when it is empty or repeats the
fault, and so does a first reply that the owner has no time to repeat. Every logical call
has its own durable allowance in the owner's per-unit call accounting, and
every accepted reply is stored with its admission record under its own call
number before the next call is sent: both calls stay recorded, the
reply of the highest stored call stands, and an interrupted unit resumes the
call it had started.

After the barrier, in this order: settlement at the settlement pin
and its record (`store_settlement`); the vectors of the
unsettled positions (``ingestion/statement_embedding.py``); every position's
``ProposalV2``, create-once (`build_proposals`, `store_proposals`); and the
settlement transaction (`plan_settlement`, the chunked ``commit_settlement``),
whose candidate rows reference them. `ExtractionStage.run` composes these
steps; every one is idempotent, so a resumed owner replays them with no
provider call it already made and no row written twice, also a cold-adopted one
at its later pin before its first commit. An owner whose pin moved
past its settlement, by a privacy erase or a cold adoption after its
commits, replays none of them: its outcome is read from its durable records
(`rebased_outcome`).
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, Protocol

from quotemem.core.add_read_view import AddReadView, EvidenceRecord, Sigma, span_sigma
from quotemem.core.artifacts import DigestSettlementSetV1
from quotemem.core.commit_plan import (
    CommitJournalV1,
    DuplicatePositionV1,
    SettledCode,
    SettledPositionV1,
)
from quotemem.core.controls import ControlRowV1, ForgetWithdrawnV1
from quotemem.core.embedding import embedding_state_token
from quotemem.core.entities import MemoryLifecycle, PositionKey, QuoteSpanV1, ReasonCode
from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_context import ExtractContextV2
from quotemem.core.extraction_output import ExtractOutputV12
from quotemem.core.extraction_policy import (
    CHAT_FORGET_COMPLETE,
    QUESTION_SUPPORT_RULE,
    ExtractionPolicyConfigV2,
    ExtractionPolicyV2,
    agent_release,
    extraction_pipeline,
)
from quotemem.core.extraction_records import (
    DigestSettlementV1,
    ErasedProposalV1,
    ErasedQuoteSpanV1,
    ErasedReplyV1,
    ExtractionAdmissionV2,
    ExtractionRefused,
    WholeOutputFault,
)
from quotemem.core.extraction_supplement import IMPLIED_HEALTH_RULES
from quotemem.core.identity import ArtifactId, integrity_digest
from quotemem.core.inference import InferenceBindingV1, StoredInferenceOutput
from quotemem.core.proposal import (
    EmbeddingUseV1,
    ProposalNamingV1,
    ProposalV2,
    RevisedVersionV1,
    importance_inputs,
)
from quotemem.core.provider_budget import ProviderCallBudgetV2, SendsSpent, UnitCallEndV1
from quotemem.core.quote_location import LocatedQuote, locate_quote
from quotemem.core.semantic import SecurityLevel
from quotemem.core.statement import MAX_UNIT_NO, token_seq
from quotemem.ingestion.admission_measurement import observe_admission, observe_refusal
from quotemem.ingestion.context import (
    ExtractionContext,
    PreviousBatch,
    build_context,
    reconstruct_context,
)
from quotemem.ingestion.extract import ExtractionAllowance, ExtractionResult, extract
from quotemem.ingestion.extraction_admission import (
    REFUSALS,
    Admission,
    AdmittedFact,
    RefusedFact,
    SpanKey,
    refers_to_user,
    replay_admission,
    span_key,
)
from quotemem.ingestion.nomination import resolve_slots
from quotemem.ingestion.resend_lexicon import assertion_spans, asserts_personal_fact, names_someone
from quotemem.ingestion.sources import LabelView, word_count
from quotemem.ingestion.statement_embedding import (
    OwnerEmbedding,
    TextKey,
    embed_texts,
    embedding_texts,
)
from quotemem.ingestion.statement_settlement import (
    Settlement,
    UnitFacts,
    anchor_entity,
    name_placeholders,
    settle,
)
from quotemem.ingestion.units import StorageBudget, Unit, plan_units
from quotemem.providers.bindings import inference_binding_for_policy
from quotemem.providers.contracts import EmbeddingProvider, StructuredInferenceProvider
from quotemem.store.erasure_gate import withheld_versions
from quotemem.store.extraction_erasure import erased_extraction_records
from quotemem.store.memory_commit import (
    DispositionV1,
    PositionV1,
    RejectionEntryV1,
    commit_rejection,
    commit_settlement,
    journaled_positions,
)

if TYPE_CHECKING:
    from quotemem.store.writer import CommittedRows, Writer, WriteTicket

_LOG = logging.getLogger(__name__)

ZERO_PROPOSAL_RETRY_WORDS: Final = 100
"""Fresh words no assistant wrote from which a reply without facts gets the fresh call,
also when it declares nothing.

Words are statement tokens (``token_seq``): word runs, and each Han, Kana or Hangul
character; punctuation and symbols count for nothing, and withheld text is not
readable, so it counts for nothing either.

The recorded failures include fact-rich units of 126 and 138 words; the
former 200-word threshold missed both. Length is only a resend signal, never
evidence that a fact exists: a second empty reply still stands.
"""

UnitCallNumber = Literal[0, 1]
"""A unit's logical extraction calls: 0 is its call, 1 the one fresh call."""

LENGTH_FAULTS: Final[frozenset[WholeOutputFault]] = frozenset({"truncated", "output_bound"})
"""The output-length faults: the reply stopped before ``stop`` or outgrew its bound.

The same request would end the same way again (a replay saw dozens of refused
calls of identical bytes), so the unit splits instead
(`split_unit`)."""

def _length_fault(refused: ExtractionRefused) -> Literal["truncated", "output_bound"] | None:
    """The refusal's fault when it is one of `LENGTH_FAULTS`."""
    fault = refused.fault
    return fault if fault == "truncated" or fault == "output_bound" else None


_UNIT_CALL: contextvars.ContextVar[UnitCallNumber | None] = contextvars.ContextVar(
    "aml_unit_call", default=None,
)


def current_unit_call() -> UnitCallNumber | None:
    """Name the logical extraction call this context is sending: 0, or 1 for the fresh call.

    Content-free attribution for an opted-in HTTP observer, as the recorded-replay
    transport is: a unit's fresh call sends call 0's exact bytes,
    so only this tells the two calls apart.

    Returns:
        UnitCallNumber | None: The sending call's number, or None outside a unit's call.
    """
    return _UNIT_CALL.get()


class UnitCallAccounting(Protocol):
    """The owner's durable per-unit call accounting and deadline reserve.

    Every logical extraction call of a unit has its own durable allowance of
    initially at most three physical sends inside its 600 s, within the
    owner's deadline; explicit outage resumes grant one extra send each.
    Creating call 1's allowance records that the unit's first reply
    needed the fresh call (`needs_fresh_call`) and was replaced; the two
    allowances keep both calls' sends.
    """

    def calls_started(self, unit_no: int) -> int:
        """Return how many logical calls of the unit hold a durable allowance (0 to 2).

        Args:
            unit_no (int): The unit within the owner.

        Returns:
            int: 0 before the unit's first call, 1 once call 0 holds its
                allowance, 2 once the fresh call does.
        """
        ...

    def allowance(self, unit_no: int, call: UnitCallNumber) -> ExtractionAllowance:
        """Return the allowance that one logical call of the unit enters (`extract`).

        Args:
            unit_no (int): The unit within the owner.
            call (UnitCallNumber): 0 for the unit's call, 1 for its fresh call.

        Returns:
            ExtractionAllowance: Creates the call's allowance once, reserves each
                physical send before HTTP and bounds the call by its persisted
                deadline; a spent send stays spent, and a call with no send left
                raises ``SendsSpent``.
        """
        ...

    def admits_fresh_call(self, call_budget: ProviderCallBudgetV2) -> bool:
        """Tell whether the owner's deadline still covers one more call and the work after it.

        The stage asks it once call 0's reply is stored, and asks again when a
        resumed unit finds that reply stored but no fresh call started; once it
        answers False for an owner it must keep answering False, so that a
        first reply that stood is never replaced on resume.

        Args:
            call_budget (ProviderCallBudgetV2): The frozen extraction call budget;
                a fresh call needs at least its ``attempt_wall_ms`` for the first
                send, besides the reserve the owner keeps for the stages after
                extraction.

        Returns:
            bool: True when the fresh call may start now.
        """
        ...

    def admits_piece(self, need_ms: int) -> bool:
        """Tell whether the owner's current deadline fits one piece's call and the reserve.

        Asked once before a split piece's first call; a resumed owner's
        renewed deadline counts.

        Args:
            need_ms (int): The piece's expected call time: its cut parent's duration.

        Returns:
            bool: True when the piece may start now.
        """
        ...

    def ended(self, unit_no: int, call: UnitCallNumber) -> UnitCallEndV1 | None:
        """Return how the unit call ended for good, if it did.

        Args:
            unit_no (int): The unit within the owner.
            call (UnitCallNumber): 0 or 1.

        Returns:
            UnitCallEndV1 | None: The durable mark, or None.
        """
        ...

    def end(self, unit_no: int, call: UnitCallNumber, binding: InferenceBindingV1,
            end: UnitCallEndV1) -> None:
        """Mark the unit call ended for good, durably, before the stage acts on it.

        Args:
            unit_no (int): The unit within the owner.
            call (UnitCallNumber): 0 or 1.
            binding (InferenceBindingV1): The call's frozen extraction binding.
            end (UnitCallEndV1): How it ended and what the stage makes of it.
        """
        ...


@dataclass(frozen=True, slots=True)
class UnitExtraction:
    """One unit's standing reply, stored.

    Attributes:
        reference (ArtifactId): Its ``llm_output``.
        call (UnitCallNumber): The logical call whose reply stands; 1 when the
            fresh call replaced the first reply (`needs_fresh_call`).
        reply (ExtractOutputV12): The verbatim reply.
        admission (Admission): Its admission, made by the call or replayed from
            the stored record.
        spent (bool): Whether the fresh call's sends were spent with no
            accepted reply: call 0's reply stands, and the unit ends with a
            provider refusal.
    """

    reference: ArtifactId
    call: UnitCallNumber
    reply: ExtractOutputV12
    admission: Admission
    spent: bool = False


def fresh_words(context: ExtractionContext) -> int:
    """Count the readable words of a unit's fresh sources that no assistant wrote.

    A message's parts render as one source, so a word that a part boundary
    splits counts once; readable stretches never join across withheld text.

    Args:
        context (ExtractionContext): The unit's frozen context.

    Returns:
        int: The tokens of the fresh, non-assistant sources' readable text.
    """
    return sum(
        len(token_seq(text)) for source in context.sources
        if source.fresh and source.role != "assistant" for text in source.readable_texts()
    )


def writes_user_for_participant(reply: ExtractOutputV12, context: ExtractionContext) -> bool:
    """Tell whether a reply wrote the user for a participant of a unit the user is not in.

    The unit renders no authenticated-self source, so no fact can be the user's
    (``user_unattested``), and a fact that refers to the user anyway
    (`refers_to_user`) quotes a source a participant speaks: the extractor wrote the
    template's user wording, or the ``user`` handle, for that participant.

    Args:
        reply (ExtractOutputV12): A contract-valid reply of the unit.
        context (ExtractionContext): The unit's frozen context.

    Returns:
        bool: True when such a fact exists.
    """
    if any(source.attribution.authenticated_self for source in context.sources):
        return False
    for fact in reply.facts:
        if not refers_to_user(fact):
            continue
        for group in fact.support_groups:
            for quote in group.quotes:
                source = context.source(quote.source_id)
                if source is not None and source.attribution.participant_key is not None:
                    return True
    return False


def leaves_assertion_unquoted(reply: ExtractOutputV12, context: ExtractionContext) -> bool:
    """Tell whether a reply with facts quotes nothing of a fresh short personal assertion.

    A fresh, non-assistant readable sentence with a closed assertion shape
    (`quotemem.ingestion.resend_lexicon.assertion_spans`) that no quote of the reply
    overlaps, and that contains no quote's text either (a repeated sentence is
    quoted once), was dropped while the reply kept the source's other facts ("Our apartment has a
    balcony." beside the kept "We moved to Denver in 2021."). Only a short unit qualifies, one with
    fewer than `ZERO_PROPOSAL_RETRY_WORDS` fresh
    words: in a long dialogue the extractor's selection is expected, and a recorded
    run's two fresh calls on one such sentence in a dialogue of about 500 words left
    it unquoted again while replacing the first reply. Like the
    empty-reply triggers it only buys the unit's one fresh call.

    Args:
        reply (ExtractOutputV12): A contract-valid reply of the unit with facts.
        context (ExtractionContext): The unit's frozen context.

    Returns:
        bool: True when such a sentence exists.
    """
    if fresh_words(context) >= ZERO_PROPOSAL_RETRY_WORDS:
        return False
    covered: dict[str, list[tuple[int, int]]] = {}
    texts: set[str] = set()
    for fact in reply.facts:
        for group in fact.support_groups:
            for quote in group.quotes:
                text = unicodedata.normalize("NFC", quote.quote).strip()
                if text:
                    texts.update({text, text[0].swapcase() + text[1:]})
                source = context.source(quote.source_id)
                if source is None or not text:
                    continue
                located = locate_quote(quote.quote, source.masked)
                if isinstance(located, LocatedQuote):
                    covered.setdefault(quote.source_id, []).append(
                        (located.char_start, located.char_end))
    for source in context.sources:
        if not source.fresh or source.role == "assistant":
            continue
        quoted = covered.get(source.source_id, [])
        for start, end in source.readable_ranges():
            for first, last in assertion_spans(source.text[start:end]):
                left, right = start + first, start + last
                sentence = unicodedata.normalize("NFC", source.text[left:right])
                if not any(a < right and b > left for a, b in quoted) and not any(
                        text in sentence for text in texts):
                    return True
    return False


_NAMED_POSSESSIVE = re.compile(
    r"(?:^|[.!:\n]\s*)(?i:my|our|his|her|their)\s+[a-z]+\s+[A-Z][a-z]+\b"
    r"[^.!?\n]*(?:[.!]|$)",
)
"""A declarative sentence's possessive, lower-case noun and capitalized name.

This spelling shape only buys a fresh sample; it never establishes a fact or an
entity. Questions, assistant text, previous sources and withheld stretches do not
qualify. Readable stretches are tested separately, never joined across redaction.
"""


def needs_fresh_call(reply: ExtractOutputV12, context: ExtractionContext) -> bool:
    """Tell whether a unit's first reply gets the one fresh call.

    Args:
        reply (ExtractOutputV12): The unit's first reply, already contract-valid
            (a whole-output fault never reaches here).
        context (ExtractionContext): The unit's frozen context.

    Returns:
        bool: True when the reply states no nonblank fact and either declares a new entity
            or an unresolved referent, which only a fact can anchor, or the unit has
            at least `ZERO_PROPOSAL_RETRY_WORDS` fresh words that no assistant wrote,
            or a fresh non-assistant readable stretch has a named possessive phrase,
            in English or in a closed phrase of another language
            (`quotemem.ingestion.resend_lexicon`), or a closed first-person multilingual
            personal-assertion shape from that module (subject, listed predicate and
            content; no question, quotation or small-talk complement);
            or when it states facts and one writes the user for a participant of a
            unit the user is not in (`writes_user_for_participant`), or no quote of it
            covers a fresh sentence with that personal-assertion shape
            (`leaves_assertion_unquoted`).
    """
    # Blank statements are malformed positions, never a fact for this decision.
    # A mixed reply is not empty; only a blank-only one shares the empty triggers.
    if any(fact.statement.strip() for fact in reply.facts):
        return (writes_user_for_participant(reply, context)
                or leaves_assertion_unquoted(reply, context))
    if reply.new_entities or reply.unresolved_referents:
        return True
    return fresh_words(context) >= ZERO_PROPOSAL_RETRY_WORDS or any(
        _NAMED_POSSESSIVE.search(text) is not None or names_someone(text)
        or asserts_personal_fact(text)
        for source in context.sources if source.fresh and source.role != "assistant"
        for text in source.readable_texts()
    )


def extract_unit(
    unit_no: int,
    context: ExtractionContext,
    policy: ExtractionPolicyConfigV2,
    provider: StructuredInferenceProvider,
    *,
    authorize: Callable[[], None],
    accounting: UnitCallAccounting,
    replies: ReplyStore,
) -> UnitExtraction | None:
    """Make or replay a unit's extraction call, and its one fresh call when its reply needs it.

    Each accepted reply is stored with its admission record under its call
    number before anything else is sent, so call 0's reply is
    durable before call 1 is dispatched. The reply of the highest stored call
    stands. A unit whose standing reply is stored replays it with no send; one
    whose call 0 is stored and whose fresh call has started resumes that call;
    one that started only call 0 sends it again within that call's allowance.
    No unit ever makes a third call. Each fresh reply, the replaced one
    included, is counted in the admission measurements, and so is a
    whole-output refusal.

    Args:
        unit_no (int): The unit within the owner.
        context (ExtractionContext): Its frozen context, persisted before the call.
        policy (ExtractionPolicyConfigV2): The owner's frozen extraction policy.
        provider (StructuredInferenceProvider): the structured provider.
        authorize (Callable[[], None]): Rechecks grant, lease, cancellation and
            privacy validity around every send (`extract`).
        accounting (UnitCallAccounting): The owner's per-unit call accounting.
        replies (ReplyStore): The owner's create-once reply storage.

    A fresh call whose sends are spent, or whose reply has a length fault
    (`LENGTH_FAULTS`), leaves the first reply standing, marked ``spent``;
    the length fault is marked on the call first, so a replay never resends it.

    Returns:
        UnitExtraction | None: The standing reply, or None when the unit renders
            no readable fresh source, which makes no call and stores nothing.

    Raises:
        ExtractionRefused: A whole-output fault of call 0, or a fault of the fresh
            call outside `LENGTH_FAULTS`.
        SendsSpent: Call 0 has no accepted reply and no send left.
        ProviderError: A provider failure, which pauses the owner.
        QuoteMemError: Revoked authority, an exhausted allowance, ``binding_mismatch``
            when the stored replies or started calls break the call order or a
            replayed record differs, and ``artifact_digest_collision`` when a
            call already stored another reply.
    """
    pipeline = policy.parameters.pipeline_version
    profile = extraction_pipeline(pipeline).profile

    def measured(call: UnitCallNumber) -> UnitExtraction | None:
        token = _UNIT_CALL.set(call)
        try:
            result = extract(context, policy, provider, authorize=authorize,
                             allowance=accounting.allowance(unit_no, call))
        except ExtractionRefused as refused:
            observe_refusal(refused.fault)
            # The refusal names the unit it ended, keeping its cause.
            refused.unit_no = unit_no
            raise
        finally:
            _UNIT_CALL.reset(token)
        if result is None:
            return None
        observe_admission(result.admission)
        if not isinstance(result.result.output, ExtractOutputV12):
            raise QuoteMemError("binding_mismatch", "Extraction returned another output contract")
        return UnitExtraction(replies.store(context, call, result), call, result.result.output,
                              result.admission)

    def replayed(found: StoredReply) -> UnitExtraction:
        return UnitExtraction(found.reference, found.call, found.reply, replay_admission(
            found.reply, context, profile, found.record, pipeline_version=pipeline,
            implied_health_rule=policy.parameters.extraction_supplement in IMPLIED_HEALTH_RULES,
            self_identity=policy.parameters.self_identity is not None,
            question_support_rule=policy.parameters.quote_grounding == QUESTION_SUPPORT_RULE,
            plain_corrections=agent_release(policy),
        ))

    stored = {found.call: found for found in replies.stored(context)}
    if 0 in stored:
        first = replayed(stored[0])
    elif stored or accounting.calls_started(unit_no) > 1:
        raise QuoteMemError("binding_mismatch",
                            "A unit's fresh call came before its stored first reply")
    else:
        made = measured(0)
        if made is None:
            return None
        first = made
    if not needs_fresh_call(first.reply, context):
        if 1 in stored:
            raise QuoteMemError("binding_mismatch",
                                "A unit's first reply needed no fresh call yet was replaced")
        return first
    if 1 in stored:
        return replayed(stored[1])
    if accounting.ended(unit_no, 1) is not None:
        return replace(first, spent=True)
    if accounting.calls_started(unit_no) < 2 and not accounting.admits_fresh_call(
        policy.parameters.call_budget,
    ):
        return first
    try:
        return measured(1)
    except SendsSpent:
        return replace(first, spent=True)
    except ExtractionRefused as refused:
        # A fresh call cut short is a lost resample, not a lost unit: call 0's
        # reply stands, as when the fresh call's sends are spent.
        if (fault := _length_fault(refused)) is None:
            raise
        accounting.end(unit_no, 1, inference_binding_for_policy(policy),
                       UnitCallEndV1(cause=fault, outcome="first_stands"))
        return replace(first, spent=True)


# ------------------------------------------------------------------- the owner's units


@dataclass(frozen=True, slots=True)
class StoredReply:
    """One logical call's stored reply: its ``llm_output`` and the admission record beside it.

    Attributes:
        reference (ArtifactId): The ``llm_output``.
        call (UnitCallNumber): The unit's logical call that made the reply.
        reply (ExtractOutputV12): The verbatim reply.
        record (ExtractionAdmissionV2 | None): Its admission record; None when
            admission changed nothing.
    """

    reference: ArtifactId
    call: UnitCallNumber
    reply: ExtractOutputV12
    record: ExtractionAdmissionV2 | None


class ReplyStore(Protocol):
    """Create-once storage of each logical call's accepted reply with its admission record.

    A call's reply and its ``ExtractionAdmissionV2`` share one ``llm_output``
    (stages 4–6) whose identity carries the call number;
    the write-once check holds per call.
    """

    def stored(self, context: ExtractionContext) -> tuple[StoredReply, ...]:
        """Return the unit's stored replies in call order; empty before its first stored call.

        Args:
            context (ExtractionContext): The unit's frozen context.

        Returns:
            tuple[StoredReply, ...]: At most one reply per logical call.
        """
        ...

    def store(
        self, context: ExtractionContext, call: UnitCallNumber, extraction: ExtractionResult,
    ) -> ArtifactId:
        """Store one logical call's accepted reply and its admission record once.

        Args:
            context (ExtractionContext): The unit's frozen context.
            call (UnitCallNumber): The logical call that made the reply.
            extraction (ExtractionResult): The call's admitted reply.

        Returns:
            ArtifactId: Its ``llm_output``; the stored one when the call
                already stored these bytes.
        """
        ...


Phase = Literal["extraction", "embedding", "commit"]
"""The owner phases the extraction stage marks: extraction up to the barrier, embedding from
settlement through the ``ProposalV2`` writes, and the settlement transaction's commit."""


class ExtractionOwner(Protocol):
    """What the Add owner hands its extraction stage: frozen inputs, checks, services.

    The owner holds its lease and its frozen prefix, policy and pins; every check
    it supplies raises the owner's own errors (``lease_lost``, privacy refusals).
    """

    @property
    def owner_run_id(self) -> str:
        """The Add owner run."""
        ...

    @property
    def ticket(self) -> WriteTicket:
        """The owner's ticket, for its create-once artifacts."""
        ...

    @property
    def writer(self) -> Writer:
        """The single Writer."""
        ...

    @property
    def policy(self) -> ExtractionPolicyConfigV2:
        """The owner's frozen extraction policy."""
        ...

    @property
    def prefix(self) -> Sequence[EvidenceRecord]:
        """The owner's frozen prefix: whole parts in message order."""
        ...

    @property
    def labels(self) -> LabelView:
        """The frozen turn labels of the prefix and its previous batches."""
        ...

    @property
    def view(self) -> AddReadView:
        """The guarded read view at the owner pin, right after the prefix froze."""
        ...

    def settlement_view(self) -> AddReadView:
        """Return the guarded read view right after the barrier, the settlement pin."""
        ...

    @property
    def provider(self) -> StructuredInferenceProvider:
        """The structured provider of extraction calls."""
        ...

    @property
    def embedder(self) -> EmbeddingProvider:
        """The single embedding provider."""
        ...

    @property
    def accounting(self) -> UnitCallAccounting:
        """The owner's per-unit call accounting and deadline reserve."""
        ...

    @property
    def replies(self) -> ReplyStore:
        """Create-once storage of the units' replies."""
        ...

    def measure(self, unit: Unit, facts: int) -> StorageBudget:
        """Measure a unit's worst admission transaction with the Writer's plan."""
        ...

    def unit_context(self, unit: Unit) -> ExtractionContext | None:
        """Return the unit's extraction context at the owner pin (`build_unit_context`).

        Its sizing probe and its extraction need exactly this context; an owner
        may serve the one its probe built, from the same pin and inputs, once.
        """
        ...

    def previous_batch(self, unit: Unit) -> PreviousBatch | None:
        """Return the receipt-proven previous batch of the unit's request at the owner pin.

        One ``select_previous_batch`` per request of the owner's view serves the
        label stage, every sizing probe and every unit's context; a reuse rechecks
        ``check_valid``.
        """
        ...

    def authorize(self) -> None:
        """Recheck grant, lease, cancellation and privacy validity before an action."""
        ...

    def check_valid(self) -> None:
        """Recheck the owner lease and the held privacy operation around a read."""
        ...

    def mark_phase(self, phase: Phase) -> None:
        """Start one owner phase of the census."""
        ...

    def publish(self, committed: CommittedRows) -> None:
        """Publish one committed settlement chunk to the read model."""
        ...

    def clock(self) -> int:
        """Return the trusted time in epoch milliseconds."""
        ...


@dataclass(frozen=True, slots=True)
class UnitAdmission:
    """One unit's admitted positions and the artifacts their proposals cite.

    Attributes:
        unit_no (int): The unit.
        facts (tuple[AdmittedFact, ...]): Its admitted positions.
        extraction_ref (ArtifactId): Its reply's ``llm_output``.
        context_ref (ArtifactId): Its ``decision_set:extraction_context``.
        refused (tuple[RefusedFact, ...]): The positions admission refused.
        sources (tuple[str, ...]): Its fresh Evidence parts, which name its request.
    """

    unit_no: int
    facts: tuple[AdmittedFact, ...]
    extraction_ref: ArtifactId
    context_ref: ArtifactId
    refused: tuple[RefusedFact, ...] = ()
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SpentUnit:
    """A unit whose extraction call has no accepted reply and no send left.

    Its sends were reserved before an explicit resume or a crash and stay spent,
    so the unit ends with a provider refusal: it yields no fact,
    and its messages stay as Evidence.

    Attributes:
        unit_no (int): The unit.
        sources (tuple[str, ...]): Its fresh Evidence parts, which name its request.
    """

    unit_no: int
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AdmittedUnits:
    """Every unit of the owner at the barrier.

    Attributes:
        units (tuple[UnitAdmission, ...]): The admitted units in unit order; a
            unit with no readable fresh source makes no call and is absent.
        spent (tuple[SpentUnit, ...]): The units a provider refusal ended, in
            unit order.
    """

    units: tuple[UnitAdmission, ...]
    spent: tuple[SpentUnit, ...] = ()


def _stored_contexts(owner: ExtractionOwner) -> dict[int, tuple[ArtifactId, ExtractContextV2]]:
    """The owner's stored extraction contexts by unit."""
    found: dict[int, tuple[ArtifactId, ExtractContextV2]] = {}
    for reference, body in owner.writer.find_logical_candidates(owner.owner_run_id,
                                                                "decision_set"):
        if isinstance(body, ExtractContextV2):
            if body.unit.unit_no in found:
                raise QuoteMemError("binding_mismatch", "A unit has two stored contexts")
            found[body.unit.unit_no] = (reference, body)
    return found


def withdrawn_memories(writer: Writer, scope_id: str) -> frozenset[str]:
    """The scope's summary-only memories an in-chat forget withdrew (``forget_withdrawn:{s}``).

    The session rollup's call reads none of them (``ingestion/context.session_rollup_of``).
    """
    stored = writer.get_control(f"forget_withdrawn:{scope_id}")
    return frozenset() if stored is None else frozenset(
        ForgetWithdrawnV1.model_validate(stored["value"]).memory_ids)


def build_unit_context(owner: ExtractionOwner, unit: Unit) -> ExtractionContext | None:
    """Select the context of one unit at the owner pin, as its probe and its call see it.

    Args:
        owner: The Add owner, at its pinned view.
        unit: The unit.

    Returns:
        ExtractionContext | None: ``build_context`` over the unit's preceding parts of
            its request and the owner's previous batch.
    """
    parameters = owner.policy.parameters
    first = unit.parts[0]
    request = (first.session_id, first.request_id)
    preceding = [
        part for part in owner.prefix[:owner.prefix.index(first)]
        if (part.session_id, part.request_id) == request
    ]
    return build_context(
        owner.view, unit, owner_run_id=owner.owner_run_id, labels=owner.labels,
        bind=owner.writer.bind_content, preceding=preceding, previous=owner.previous_batch(unit),
        tail_words=parameters.tail_words, previous_words=parameters.previous_batch_words,
        withdrawn=withdrawn_memories(owner.writer, owner.view.scope_id),
        hidden=withheld_versions(owner.writer, owner.view.scope_id),
    )


def _context(
    owner: ExtractionOwner, unit: Unit, stored: tuple[ArtifactId, ExtractContextV2] | None,
) -> tuple[ExtractionContext, ArtifactId] | None:
    """A unit's context: rebuilt from its stored record, or selected and stored before the call."""
    parameters = owner.policy.parameters
    if stored is not None:
        reference, record = stored
        context = reconstruct_context(
            owner.view, record, owner_run_id=owner.owner_run_id,
            bind=owner.writer.bind_content, check_valid=owner.check_valid,
            tail_words=parameters.tail_words, previous_words=parameters.previous_batch_words,
            withdrawn=withdrawn_memories(owner.writer, owner.view.scope_id),
        )
        if [part.object_id for part in context.unit.parts] != [
            part.object_id for part in unit.parts
        ]:
            raise QuoteMemError("binding_mismatch",
                                "A stored context covers other parts than its unit")
        return context, reference
    built = owner.unit_context(unit)
    if built is None:
        return None
    reference = owner.writer.put_logical_artifact(
        owner.ticket, "decision_set", built.record, created_at=owner.clock(),
    )
    return built, reference


UnsplitReason = Literal["single_part", "unit_bound"]
"""Why a piece cannot split: it is one part, or its subtree's numbers would pass
``MAX_UNIT_NO``."""


def piece_bases(units: Sequence[Unit]) -> dict[int, int]:
    """Each planned unit's base: its piece ``i > 0`` is unit ``base + i`` (`split_unit`).

    A unit of n parts has at most 2n - 1 pieces, itself included, so the pieces of the
    planned units take consecutive number ranges past the planned units, in plan order.

    Args:
        units (Sequence[Unit]): The owner's planned units, numbered 0 to len - 1.

    Returns:
        dict[int, int]: The base of each planned unit's piece numbers.
    """
    bases: dict[int, int] = {}
    next_number = len(units)
    for unit in units:
        bases[unit.unit_no] = next_number - 1
        next_number += 2 * len(unit.parts) - 2
    return bases


def unsplit_reason(unit: Unit, *, index: int, base: int) -> UnsplitReason | None:
    """Why a piece cannot split, or None when it can (`split_unit`).

    Args:
        unit (Unit): The piece.
        index (int): Its place in its planned unit's halving tree (0 is the planned unit).
        base (int): Its planned unit's base (`piece_bases`).

    Returns:
        UnsplitReason | None: The reason, or None for a piece that splits.
    """
    if len(unit.parts) < 2:
        return "single_part"
    if base + index + 2 * len(unit.parts) - 2 > MAX_UNIT_NO:
        return "unit_bound"
    return None


def split_unit(
    unit: Unit, *, index: int, base: int,
) -> tuple[tuple[int, Unit], tuple[int, Unit]] | None:
    """The two halves a piece of a planned unit splits into after a length fault.

    The halves cut between whole parts where their word counts come closest to equal;
    the second keeps the first as preceding context (`build_unit_context`), so it still
    reads what came before it. Pieces are numbered in preorder of the planned unit's
    full halving tree, whose n single-part leaves give it 2n - 1 nodes: piece ``index``
    of k parts has its halves at ``index + 1`` and ``index + 2 * len(first)``, and is
    unit ``base + index`` (the planned unit itself, index 0, keeps its own number).
    The numbers are distinct, past every planned unit, and the same on every resume,
    since ``plan_units`` is a pure function of the frozen prefix.

    Args:
        unit (Unit): The piece.
        index (int): Its place in its planned unit's halving tree.
        base (int): Its planned unit's base (`piece_bases`).

    Returns:
        tuple[tuple[int, Unit], tuple[int, Unit]] | None: Each half's index and unit,
            in message order; None when `unsplit_reason` names why it cannot split.
    """
    if unsplit_reason(unit, index=index, base=base) is not None:
        return None
    words = [word_count(part.text) for part in unit.parts]
    total, running, cut, gap = sum(words), 0, 1, None
    for position in range(1, len(words)):
        running += words[position - 1]
        if gap is None or abs(2 * running - total) < gap:
            cut, gap = position, abs(2 * running - total)
    first, second = index + 1, index + 2 * cut
    return ((first, Unit(base + first, unit.parts[:cut])),
            (second, Unit(base + second, unit.parts[cut:])))


class _SplitStopped(Exception):
    """A split piece did not start because another unit of the owner failed."""


def _admit_unit(
    owner: ExtractionOwner,
    unit: Unit,
    stored: Mapping[int, tuple[ArtifactId, ExtractContextV2]],
    *,
    index: int,
    base: int,
    stop: threading.Event,
    need_ms: int | None = None,
) -> tuple[UnitAdmission | SpentUnit, ...]:
    """One unit or piece: its context, then its reply replayed or its calls made and stored.

    A spent call ends the unit with a provider refusal; when it is the
    fresh call, call 0's admitted reply stands beside the refusal. A call refused
    for its length (`LENGTH_FAULTS`) is not sent again: the unit's halves are
    admitted in its place (`split_unit`), and a unit that cannot split ends like a
    spent one, its messages kept as Evidence. A piece starts only while the owner's
    deadline fits ``need_ms``, its cut parent's duration, and the reserve
    (``admits_piece``); one that does not ends spent unstarted. Each such end is
    marked on the call (``UnitCallEndV1``) before the stage acts on it, and a
    replay reads the mark before anything is sent. No
    further piece starts once another unit of the owner failed (`admit_units`).
    """
    accounting = owner.accounting
    sources = tuple(part.object_id for part in unit.parts)
    halves = split_unit(unit, index=index, base=base)

    def ended(mark: UnitCallEndV1) -> tuple[UnitAdmission | SpentUnit, ...]:
        if mark.outcome != "split":
            return (SpentUnit(unit.unit_no, sources),)
        if halves is None or mark.elapsed_ms is None:
            raise QuoteMemError("binding_mismatch", "A unit marked split cannot split")
        admitted: list[UnitAdmission | SpentUnit] = []
        for half_index, half in halves:
            if stop.is_set():
                raise _SplitStopped
            admitted += _admit_unit(owner, half, stored, index=half_index, base=base,
                                    stop=stop, need_ms=mark.elapsed_ms)
        return tuple(admitted)

    if (mark := accounting.ended(unit.unit_no, 0)) is not None:
        return ended(mark)
    found = _context(owner, unit, stored.get(unit.unit_no))
    if found is None:
        return ()
    context, context_ref = found
    binding = inference_binding_for_policy(owner.policy)
    if need_ms is not None and accounting.calls_started(unit.unit_no) == 0 and (
        not accounting.admits_piece(need_ms)
    ):
        accounting.end(unit.unit_no, 0, binding,
                       UnitCallEndV1(cause="owner_time", outcome="not_started"))
        _LOG.warning("Split piece not started: owner=%s unit=%d reason=owner_time",
                     owner.owner_run_id, unit.unit_no)
        return (SpentUnit(unit.unit_no, sources),)
    started = owner.clock()
    try:
        extraction = extract_unit(
            unit.unit_no, context, owner.policy, owner.provider, authorize=owner.authorize,
            accounting=accounting, replies=owner.replies,
        )
    except SendsSpent:
        # The unit's facts are lost and its Evidence stays searchable; this is the only trace.
        _LOG.warning("Extraction unit ends spent: owner=%s unit=%d reason=sends",
                     owner.owner_run_id, unit.unit_no)
        return (SpentUnit(unit.unit_no, sources),)
    except ExtractionRefused as refused:
        if (fault := _length_fault(refused)) is None:
            raise
        reason = unsplit_reason(unit, index=index, base=base)
        mark = (UnitCallEndV1(cause=fault, outcome=reason) if reason is not None else
                UnitCallEndV1(cause=fault, outcome="split",
                              elapsed_ms=max(0, owner.clock() - started)))
        accounting.end(unit.unit_no, 0, binding, mark)
        if reason is not None:
            _LOG.warning("Length-faulted unit ends spent: owner=%s unit=%d reason=%s",
                         owner.owner_run_id, unit.unit_no, reason)
        return ended(mark)
    if extraction is None:
        return ()
    admitted = UnitAdmission(unit.unit_no, extraction.admission.facts, extraction.reference,
                             context_ref, extraction.admission.refused, sources)
    return (admitted, SpentUnit(unit.unit_no, sources)) if extraction.spent else (admitted,)


def admit_units(owner: ExtractionOwner) -> AdmittedUnits:
    """Extract and admit every unit of the owner up to the barrier.

    Units come from the capacity-proven packing. At most
    ``extraction_concurrency`` units are in flight; each runs in a copy of the
    caller's context, so the owner's phase tag reaches every provider call. A
    unit is idempotent: a stored reply is replayed with no call, a stored
    context is rebuilt and its call resumed, and anything else is selected,
    stored and called. A unit whose call has no accepted reply and no send
    left ends with a provider refusal while the others go on. After
    a failure no further unit or split piece starts; the units in flight finish
    their current call, and the failure of the lowest failed unit is raised.

    Args:
        owner (ExtractionOwner): The Add owner.

    Returns:
        AdmittedUnits: The admitted units and the units a provider refusal
            ended, each in unit order.

    Raises:
        ExtractionRefused: A whole-output fault of a unit outside `LENGTH_FAULTS`;
            a length fault splits the unit instead (`split_unit`).
        ProviderError: A provider failure, which pauses the owner.
        QuoteMemError: ``budget_exceeded`` before any call, or an owner check.
    """
    owner.mark_phase("extraction")
    parameters = owner.policy.parameters
    units = plan_units(owner.prefix, owner.labels, policy=parameters, measure=owner.measure)
    stored = _stored_contexts(owner)
    bases = piece_bases(units)
    limit = max(1, min(parameters.extraction_concurrency, len(units)))
    results: dict[int, tuple[UnitAdmission | SpentUnit, ...]] = {}
    failures: dict[int, Exception] = {}
    stop = threading.Event()
    waiting = list(units)
    with ThreadPoolExecutor(max_workers=limit) as pool:
        running: dict[Future[tuple[UnitAdmission | SpentUnit, ...]], int] = {}
        while waiting or running:
            while waiting and len(running) < limit and not failures:
                unit = waiting.pop(0)
                running[pool.submit(contextvars.copy_context().run, _admit_unit, owner, unit,
                                    stored, index=0, base=bases[unit.unit_no],
                                    stop=stop)] = unit.unit_no
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                unit_no = running.pop(future)
                try:
                    results[unit_no] = future.result()
                except Exception as error:  # the owner's failure, raised after the barrier
                    failures[unit_no] = error
                    stop.set()
    # A unit stopped mid-split only follows another unit's failure, which is raised.
    if real := {unit_no: error for unit_no, error in failures.items()
                if not isinstance(error, _SplitStopped)}:
        raise real[min(real)]
    # A split unit's pieces come back in message order; numbers order the outcome.
    ended = sorted((item for unit in units for item in results[unit.unit_no]),
                   key=lambda item: item.unit_no)
    return AdmittedUnits(
        tuple(item for item in ended if isinstance(item, UnitAdmission)),
        tuple(item for item in ended if isinstance(item, SpentUnit)),
    )


# ------------------------------------------------- settlement record and proposals

_SECURITY: Final[dict[SecurityLevel, int]] = {"internal": 0, "confidential": 1, "restricted": 2}


def store_settlement(owner: ExtractionOwner, settlement: Settlement) -> ArtifactId:
    """Store the owner's settlement record once, or prove the stored one equal.

    Settlement is pure, so recomputing it at the same pin gives the same record.
    A cold-adopted owner with no commit yet settles again at a later commit of
    the same scope epochs (`rebased_outcome`), and its stored record stands when
    only that commit differs. Any other difference means the pin or
    the inputs moved, and it fails.

    Args:
        owner (ExtractionOwner): The Add owner.
        settlement (Settlement): Its settlement.

    Returns:
        ArtifactId: The ``decision_set:settlement``.

    Raises:
        QuoteMemError: ``binding_mismatch`` when another settlement record is stored.
    """
    body = DigestSettlementSetV1(
        settlements=sorted(settlement.records, key=lambda record: record.position_key),
    )
    stored = [
        (reference, found)
        for reference, found in owner.writer.find_logical_candidates(owner.owner_run_id,
                                                                     "decision_set")
        if isinstance(found, DigestSettlementSetV1)
    ]
    if stored:
        reference, found = stored[0]
        if len(stored) > 1 or len(found.settlements) != len(body.settlements) or not all(
            old.read_view_commit_seq <= new.read_view_commit_seq
            and new.model_copy(update={"read_view_commit_seq": old.read_view_commit_seq}) == old
            for old, new in zip(found.settlements, body.settlements, strict=True)
        ):
            raise QuoteMemError("binding_mismatch", "Settlement replay computed another record")
        return reference
    return owner.writer.put_logical_artifact(
        owner.ticket, "decision_set", body, created_at=owner.clock(),
    )


def _support(
    groups: Sequence[Sequence[QuoteSpanV1]],
) -> tuple[list[QuoteSpanV1], list[list[int]]]:
    """Quote spans in first appearance over the groups, and the canonical groups."""
    index: dict[SpanKey, int] = {}
    spans: list[QuoteSpanV1] = []
    encoded: set[tuple[int, ...]] = set()
    for group in groups:
        positions: set[int] = set()
        for span in group:
            key = span_key(span)
            if key not in index:
                index[key] = len(spans)
                spans.append(span)
            positions.add(index[key])
        encoded.add(tuple(sorted(positions)))
    return spans, [list(group) for group in sorted(encoded)]


def _embedding_positions(settlement: Settlement) -> set[PositionKey]:
    """Keep review followers rebuildable if their representative stays pending."""
    records = {record.position_key: record for record in settlement.records}
    return set(settlement.order) - records.keys() | {
        key for key, record in records.items()
        if record.code == "duplicate_in_add" and record.representative not in records
    }


def build_proposals(
    units: Sequence[UnitAdmission],
    settlement: Settlement,
    embeddings: Mapping[TextKey, EmbeddingUseV1],
    *,
    owner_run_id: str,
    policy: ExtractionPolicyV2,
    embedding_space: str,
) -> dict[PositionKey, ProposalV2]:
    """Build every admitted position's ``ProposalV2`` after settlement.

    A position settlement settled carries null embedding references; every other
    position points at its statement entry and, with a ``corrects_text``, its
    cue entry. A representative that duplicates joined carries the joined
    groups, the united classes and the highest security of the joined
    positions; every position carries admission's concerns with settlement's.

    Args:
        units (Sequence[UnitAdmission]): The owner's admitted units.
        settlement (Settlement): Their settlement.
        embeddings (Mapping[TextKey, EmbeddingUseV1]): Where each embedded text
            is stored (`embed_texts`).
        owner_run_id (str): The Add owner.
        policy (ExtractionPolicyV2): The owner's frozen policy, for durability.
        embedding_space (str): The workspace's active space.

    Returns:
        dict[PositionKey, ProposalV2]: The proposals in canonical order.
    """
    found = {
        (owner_run_id, unit.unit_no, fact.position): (unit, fact)
        for unit in units for fact in unit.facts
    }
    records = {record.position_key: record for record in settlement.records}
    embedded_positions = _embedding_positions(settlement)
    security: dict[PositionKey, SecurityLevel] = {
        key: fact.security_level for key, (_, fact) in found.items()
    }
    for record in settlement.records:
        representative = record.representative
        if record.code == "duplicate_in_add" and representative is not None and (
            representative not in records
        ):
            joined = security[record.position_key]
            if _SECURITY[joined] > _SECURITY[security[representative]]:
                security[representative] = joined
    proposals: dict[PositionKey, ProposalV2] = {}
    for key in settlement.order:
        unit, fact = found[key]
        spans, groups = _support(settlement.groups.get(key, fact.groups))
        classes = settlement.sensitivity.get(key, fact.sensitivity_classes)
        embedded = key in embedded_positions
        cue = fact.corrects_text
        proposals[key] = ProposalV2(
            schema_version=2, position_key=key, extraction_ref=unit.extraction_ref,
            context_ref=unit.context_ref, statement=fact.statement,
            statement_digest=fact.statement_digest, kind=fact.kind, anchors=list(fact.anchors),
            naming=[ProposalNamingV1(
                anchor_index=naming.anchor_index, name=naming.name, name_kind=naming.name_kind,
            ) for naming in fact.naming],
            quote_spans=spans, support_groups=groups, time_expr=fact.time.expression,
            valid_from=fact.time.valid_from, valid_to=fact.time.valid_to,
            time_granularity=fact.time.granularity, anchored_from=fact.time.anchored_from,
            keywords=list(fact.keywords),
            importance_inputs=importance_inputs(
                fact.kind, naming=bool(fact.naming), durability=policy.durability,
            ),
            origin=fact.origin, trust_tier=fact.trust_tier, security_level=security[key],
            sensitivity_classes=list(classes),
            sensitivity_basis=["policy_classification"] if classes else [],
            revises=[RevisedVersionV1(memory_id=memory.memory_id, revision_id=memory.revision_id)
                     for memory in fact.revises],
            corrects_text=cue,
            concerns=sorted(fact.concerns | settlement.concerns.get(key, frozenset())),
            collision_entity_ids=list(settlement.collisions.get(key, ())),
            statement_embedding=embeddings[("statement", fact.statement)] if embedded else None,
            cue_embedding=(
                embeddings[("correction_cue", cue)] if embedded and cue is not None else None
            ),
            embedding_space=embedding_space, pipeline_version=policy.pipeline_version,
        )
    return proposals


def store_proposals(
    owner: ExtractionOwner, proposals: Mapping[PositionKey, ProposalV2],
) -> dict[PositionKey, ArtifactId]:
    """Store each position's ``ProposalV2`` once, before the settlement transaction.

    A proposal a resumed owner already stored must equal the rebuilt one, since
    settlement and the stored batches it reads are replayed exactly.

    Args:
        owner (ExtractionOwner): The Add owner.
        proposals (Mapping[PositionKey, ProposalV2]): Every admitted position's.

    Returns:
        dict[PositionKey, ArtifactId]: Each position's ``proposal_snapshot``.

    Raises:
        QuoteMemError: ``binding_mismatch`` when a stored proposal differs.
    """
    stored: dict[PositionKey, tuple[ArtifactId, ProposalV2]] = {}
    for reference, body in owner.writer.find_logical_candidates(owner.owner_run_id,
                                                                "proposal_snapshot"):
        if not isinstance(body, ProposalV2) or body.position_key in stored:
            raise QuoteMemError("binding_mismatch", "An owner stored two proposals of one position")
        stored[body.position_key] = (reference, body)
    located: dict[PositionKey, ArtifactId] = {}
    for key, proposal in proposals.items():
        if key in stored:
            if stored[key][1] != proposal:
                raise QuoteMemError("binding_mismatch", "A stored proposal differs from its replay")
            located[key] = stored[key][0]
            continue
        located[key] = owner.writer.put_logical_artifact(
            owner.ticket, "proposal_snapshot", proposal, security_level="restricted",
            created_at=owner.clock(),
        )
    return located


# ------------------------------------------------- the settlement transaction


@dataclass(frozen=True, slots=True)
class SettlementPlan:
    """The settlement transaction's positions and what goes on to review.

    Attributes:
        settled (tuple[SettledPositionV1, ...]): What the settlement transaction
            writes, in canonical order.
        review (tuple[PositionKey, ...]): The round-1 members: unsettled positions
            whose primary is not unresolved, in canonical order.
        duplicates (tuple[DuplicatePositionV1, ...]): The duplicates of
            members, which each member's transaction writes, in canonical order.
    """

    settled: tuple[SettledPositionV1, ...]
    review: tuple[PositionKey, ...]
    duplicates: tuple[DuplicatePositionV1, ...]


def plan_settlement(
    units: Sequence[UnitAdmission],
    settlement: Settlement,
    proposals: Mapping[PositionKey, ArtifactId],
    *,
    owner_run_id: str,
) -> SettlementPlan:
    """Map settlement onto the settlement transaction.

    ``denied_by_forget`` is written with its own located spans withheld, ``restatement_satisfied``
    with the head the transaction compares, an unfit duplicate as ``support_budget_exceeded``,
    and an unsettled position with an unresolved primary as a pending
    ``identity_unresolved``; every other unsettled position is a round-1
    member. A duplicate mirrors its representative, which is never itself a
    duplicate: a member's duplicate goes to review with it, an unresolved
    primary's waits beside it, a held restatement's is ``duplicate_in_add``
    with that memory, and an unfit duplicate's is ``support_budget_exceeded``.

    Args:
        units (Sequence[UnitAdmission]): The owner's admitted units.
        settlement (Settlement): Their settlement.
        proposals (Mapping[PositionKey, ArtifactId]): Every position's stored
            ``ProposalV2``.
        owner_run_id (str): The Add owner.

    Returns:
        SettlementPlan: The settled positions, the members and their duplicates.
    """
    found = {
        (owner_run_id, unit.unit_no, fact.position): (unit, fact)
        for unit in units for fact in unit.facts
    }

    def position(key: PositionKey) -> dict[str, object]:
        unit, fact = found[key]
        groups = settlement.groups.get(key, fact.groups)
        return {
            "position_key": key, "extraction_ref": unit.extraction_ref,
            "proposal_ref": proposals[key], "statement_digest": fact.statement_digest,
            "entity_id": settlement.primaries[key],
            "evidence_ids": sorted({span.evidence_id for group in groups for span in group}),
        }

    return _plan_positions(
        settlement.order, {record.position_key: record for record in settlement.records},
        frozenset(settlement.unresolved), position=position,
        spans=lambda key: found[key][1].spans, lifecycle=lambda key: settlement.lifecycles[key],
        groups=lambda key: settlement.groups.get(key, found[key][1].groups),
    )


def _plan_positions(
    order: Sequence[PositionKey], records: Mapping[PositionKey, DigestSettlementV1],
    unresolved: frozenset[PositionKey], *, position: Callable[[PositionKey], dict[str, object]],
    spans: Callable[[PositionKey], Sequence[QuoteSpanV1]],
    lifecycle: Callable[[PositionKey], MemoryLifecycle],
    groups: Callable[[PositionKey], Sequence[Sequence[QuoteSpanV1]]],
) -> SettlementPlan:
    """`plan_settlement` over each position's fields, located spans, restated head lifecycle and,
    for a merge, admitted groups.

    `plan_settlement` reads them from the admitted units and the settlement;
    `rebased_outcome` rebuilds them from the owner's stored records.
    """
    settled: list[SettledPositionV1] = []
    review: list[PositionKey] = []
    duplicates: list[DuplicatePositionV1] = []

    def write(position: dict[str, object], code: SettledCode, **fields: object) -> None:
        settled.append(SettledPositionV1.model_validate({**position, "code": code, **fields}))

    for key in order:
        base = position(key)
        record = records.get(key)
        if record is None:
            if key in unresolved:
                write(base, "identity_unresolved")
            else:
                review.append(key)
        elif record.code == "denied_by_forget":
            write(base, "denied_by_forget", withheld=list(spans(key)))
        elif record.code in ("restatement_satisfied", "merged_support"):
            write(base, record.code, memory_id=record.memory_id, head={
                "memory_id": record.memory_id, "current_revision_id": record.revision_id,
                "lifecycle_state": lifecycle(key),
            }, groups=[list(group) for group in groups(key)] if (
                record.code == "merged_support"
            ) else [])
        elif record.code == "support_budget_exceeded":
            write(base, "support_budget_exceeded")
        else:
            representative = record.representative
            assert representative is not None
            lead = records.get(representative)
            if lead is None and representative in unresolved:
                write(base, "identity_unresolved")
            elif lead is None:
                duplicates.append(DuplicatePositionV1.model_validate(
                    {**base, "representative": representative},
                ))
            elif lead.code in ("restatement_satisfied", "merged_support"):
                write(base, "duplicate_in_add", memory_id=record.memory_id)
            elif lead.code == "support_budget_exceeded":
                write(base, "support_budget_exceeded")
            else:
                raise QuoteMemError("binding_mismatch", "A duplicate mirrors no representative")
    return SettlementPlan(tuple(settled), tuple(review), tuple(duplicates))


@dataclass(frozen=True, slots=True)
class RefusedPosition:
    """A reply position program admission refused: it has no candidate row.

    Attributes:
        position_key (PositionKey): ``(owner_run_id, unit_no, reply position)``.
        code (ReasonCode): Its outcome code.
        sources (tuple[str, ...]): Its unit's fresh Evidence parts. It has no
            admitted span, so they name its request.
    """

    position_key: PositionKey
    code: ReasonCode
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """What the extraction stage hands the owner's review and completion.

    Attributes:
        proposals (Mapping[PositionKey, ArtifactId]): Every admitted position's
            original proposal locator. Erased positions keep their identity
            for completion and never enter review.
        settlement_ref (ArtifactId): The owner's ``decision_set:settlement``.
        embeddings (Mapping[TextKey, EmbeddingUseV1]): Where each unsettled
            position's statement and cue vectors are stored.
        review (tuple[PositionKey, ...]): The round-1 members in canonical order.
        duplicates (tuple[DuplicatePositionV1, ...]): Their duplicates, which
            each member's transaction writes.
        refused (tuple[RefusedPosition, ...]): The positions admission refused,
            replayed from each unit's stored admission record.
        spent (tuple[SpentUnit, ...]): The units a provider refusal ended.
        withheld (tuple[QuoteSpanV1, ...]): The spans the settlement transaction withheld
            for a forget (``agent`` mode only): the owner's input pin predates them, so its
            first review reads a pin after them (``owner_pipeline``) and never shows a
            forgotten name to the judge.
        placeholders (tuple[tuple[QuoteSpanV1, str], ...]): Those of them that are a
            forgotten name's occurrences, each with how the judge sees it ("[the user's
            cat]", ``statement_settlement.name_placeholders``); every other withheld range
            renders as the marker.
    """

    proposals: Mapping[PositionKey, ArtifactId]
    settlement_ref: ArtifactId
    embeddings: Mapping[TextKey, EmbeddingUseV1]
    review: tuple[PositionKey, ...]
    duplicates: tuple[DuplicatePositionV1, ...]
    refused: tuple[RefusedPosition, ...] = ()
    spent: tuple[SpentUnit, ...] = ()
    withheld: tuple[QuoteSpanV1, ...] = ()
    placeholders: tuple[tuple[QuoteSpanV1, str], ...] = ()


def rebased_outcome(owner: ExtractionOwner) -> ExtractionOutcome | None:
    """The outcome of an owner whose pin moved past its settlement.

    ``store/finalize_privacy.rebase_finalize_privacy`` moves a held owner to its
    scope's current pin only after its first review: its units were extracted,
    settled and committed at its original input pin, whose root the erase
    discarded, so its stored contexts no longer reconstruct at the new pin
    (`reconstruct_context`). A cold adoption binds an owner to the current root
    (``StoreReadModel.capture_cold_finalize_baseline``), a later commit; once
    the owner has committed, that root holds its commits, and settling or
    reviewing its facts again would read them. Either way nothing of it is
    extracted, settled or embedded again. The outcome comes from its durable
    records instead: every ``ProposalV2``, the settlement record and the
    settlement transaction's journal, which wrote every position settlement
    settles before any review. A member an earlier invocation committed in a
    rejection or component transaction is settled and reviewed no more,
    and its duplicates were written with it. A member's duplicate resolves its
    primary at the owner's current pin. The positions admission refused and
    the units a provider refusal ended come from each unit's standing reply and
    its call accounting (`_rebased_refusals`). An owner stopped between two
    chunks of its settlement transaction finishes it first, at the first
    position without a row, from the same records (`_finish_settlement`).
    Erased replies and proposals leave content-free accounting instead of their
    bodies. Uncommitted erased positions settle ``evidence_invalidated``; journaled
    positions retain their committed classification. Surviving positions continue
    from their original proposals without another extraction or embedding call.

    Args:
        owner (ExtractionOwner): The Add owner.

    Returns:
        ExtractionOutcome | None: The outcome; None for an owner whose stored
            contexts were all rendered at its pin, or a cold-adopted one with no
            commit yet, whose records are rebuilt at its later pin
            (`reconstruct_context`, `store_settlement`).

    Raises:
        QuoteMemError: ``binding_mismatch`` when the records disagree, the
            settlement transaction wrote a position it does not settle or a
            rebuilt position differs from the one its journal holds.
    """
    stored = _stored_contexts(owner)
    if not stored:
        return None
    view = owner.view
    pin = view.pin
    if all((record.read_view_commit_seq, record.tombstone_seq, record.erase_epoch,
            record.consent_epoch) == (pin.commit_seq, pin.tombstone_seq, pin.erase_epoch,
                                      pin.consent_epoch)
           for _reference, record in stored.values()):
        return None
    if any(record.read_view_commit_seq > pin.commit_seq or record.erase_epoch > pin.erase_epoch
           for _reference, record in stored.values()):
        raise QuoteMemError("binding_mismatch", "A stored context is newer than its owner pin")
    writer, owner_run_id = owner.writer, owner.owner_run_id
    journals = [body for _reference, body in writer.find_logical_candidates(owner_run_id,
                                                                            "write_set")
                if isinstance(body, CommitJournalV1) and body.owner_run_id == owner_run_id]
    if not journals and all(record.erase_epoch == pin.erase_epoch
                            for _reference, record in stored.values()):
        return None
    owner.check_valid()
    proposals: dict[PositionKey, tuple[ArtifactId, ProposalV2]] = {}
    for reference, body in writer.find_logical_candidates(owner_run_id, "proposal_snapshot"):
        if not isinstance(body, ProposalV2) or body.position_key in proposals:
            raise QuoteMemError("binding_mismatch", "An owner stored two proposals of one position")
        proposals[body.position_key] = (reference, body)
    _erased_replies, erased = erased_extraction_records(writer, owner_run_id)
    erased = {key: value for key, value in erased.items() if key not in proposals}
    settlements = [(reference, body) for reference, body in
                   writer.find_logical_candidates(owner_run_id, "decision_set")
                   if isinstance(body, DigestSettlementSetV1)]
    if len(settlements) != 1:
        raise QuoteMemError("binding_mismatch", "A rebased owner has one settlement record")
    settlement_ref, record = settlements[0]
    records = {item.position_key: item for item in record.settlements}
    # A duplicate cannot review independently after its representative was erased:
    # it has no independent statement vector or judgment allowance.
    for key, item in records.items():
        if item.code != "duplicate_in_add" or item.representative not in erased or (
            key not in proposals
        ):
            continue
        reference, proposal = proposals.pop(key)
        erased[key] = ErasedProposalV1(
            position_key=key, proposal_ref=reference, extraction_ref=proposal.extraction_ref,
            quote_spans=[ErasedQuoteSpanV1.model_validate(span.model_dump())
                         for span in proposal.quote_spans],
            statement_digest=proposal.statement_digest,
            entity_id=anchor_entity(proposal.anchors[0], key[1], owner_run_id, view),
        )
    if not records.keys() <= proposals.keys() | erased.keys():
        raise QuoteMemError("binding_mismatch", "A rebased owner's position lost its proposal")
    records = {key: item for key, item in records.items() if key not in erased}
    unresolved = {key for key, (_reference, proposal) in proposals.items()
                  if key not in records and proposal.anchors[0].kind == "unresolved"}
    review = {key for key in proposals if key not in records and key not in unresolved}
    joined = {key for key, item in records.items()
              if item.code == "duplicate_in_add" and item.representative in review}
    written = journaled_positions(writer, owner_run_id, owner_run_id=owner_run_id, round_no=1,
                                  txn="settlement")
    written -= erased.keys()
    settles = (records.keys() - joined) | unresolved
    if not records.keys() <= proposals.keys() or not written <= settles:
        raise QuoteMemError("binding_mismatch", "A rebased owner's settlement did not commit")
    committed = {member.position_key for body in journals
                 if body.txn in ("rejection", "component") for member in body.members}
    journaled = {member.position_key for body in journals for member in body.members}

    def canonical(key: PositionKey) -> tuple[Sigma, PositionKey]:
        sigmas: list[Sigma] = []
        for span in proposals[key][1].quote_spans:
            evidence = view.evidence(span.evidence_id)
            if evidence is None:
                raise QuoteMemError("binding_mismatch",
                                    "A position cites Evidence outside its scope")
            sigmas.append(span_sigma(evidence, span.byte_start))
        return max(sigmas), key

    order = sorted(proposals, key=canonical)
    if written != settles:
        if committed:
            raise QuoteMemError("binding_mismatch",
                                "A rebased owner reviewed before its settlement")
        _finish_settlement(owner, order, proposals, records, frozenset(unresolved), settles)
    invalidated: list[RejectionEntryV1] = []
    for key, proposal in erased.items():
        if key in journaled:
            continue
        sigmas: list[Sigma] = []
        for span in proposal.quote_spans:
            evidence = view.evidence(span.evidence_id)
            if evidence is None:
                raise QuoteMemError("binding_mismatch",
                                    "An erased position lost its source identity")
            sigmas.append(span_sigma(evidence, span.byte_start))
        invalidated.append(RejectionEntryV1(
            order=(max(sigmas), key),
            position=PositionV1(
                key, proposal.extraction_ref, proposal.proposal_ref, proposal.statement_digest,
                tuple(sorted({span.evidence_id for span in proposal.quote_spans})),
            ),
            disposition=DispositionV1("evidence_invalidated", proposal.entity_id),
        ))
    if invalidated:
        owner.mark_phase("commit")
        owner.authorize()
        commit_rejection(writer, owner.ticket, owner_run_id=owner_run_id, round_no=1,
                         entries=invalidated, updated_at=owner.clock(), publish=owner.publish,
                         clock=owner.clock)
    embeddings: dict[TextKey, EmbeddingUseV1] = {}
    for key in order:
        proposal = proposals[key][1]
        if key not in review and key not in joined:
            continue
        if proposal.statement_embedding is None:
            raise QuoteMemError("binding_mismatch", "A reviewed position has a statement embedding")
        embeddings[("statement", proposal.statement)] = proposal.statement_embedding
        if proposal.corrects_text is not None and proposal.cue_embedding is not None:
            embeddings[("correction_cue", proposal.corrects_text)] = proposal.cue_embedding
    duplicates: list[DuplicatePositionV1] = []
    for key in order:
        representative = records[key].representative if key in joined else None
        if representative is None:
            continue
        reference, proposal = proposals[key]
        duplicates.append(DuplicatePositionV1(
            position_key=key, representative=representative,
            extraction_ref=proposal.extraction_ref, proposal_ref=reference,
            statement_digest=proposal.statement_digest,
            entity_id=resolve_slots(view, key, proposal.anchors)[0].entity_id,
            evidence_ids=sorted({span.evidence_id for span in proposal.quote_spans}),
        ))
    refused, spent = _rebased_refusals(owner, stored)
    owner.check_valid()
    # The settlement's withholding for a forget, as `ExtractionStage.run` reports it: a
    # resumed owner's first review reads a pin after it too.
    relearn = owner.policy.parameters.chat_forget == CHAT_FORGET_COMPLETE
    denied = [key for key in order if key in records and records[key].code == "denied_by_forget"]
    withheld = tuple(span for key in denied for span in proposals[key][1].quote_spans
                     ) if relearn else ()
    placeholders = name_placeholders(view, (
        (proposals[key][1].statement, proposals[key][1].quote_spans, records[key].tombstone_id)
        for key in denied)) if relearn else ()
    return ExtractionOutcome(
        {**{key: proposals[key][0] for key in order},
         **{key: proposal.proposal_ref for key, proposal in sorted(erased.items())}},
        settlement_ref, embeddings,
        tuple(key for key in order if key in review and key not in committed),
        tuple(item for item in duplicates if item.representative not in committed),
        refused, spent, withheld, placeholders,
    )


def _finish_settlement(
    owner: ExtractionOwner, order: Sequence[PositionKey],
    proposals: Mapping[PositionKey, tuple[ArtifactId, ProposalV2]],
    records: Mapping[PositionKey, DigestSettlementV1], unresolved: frozenset[PositionKey],
    settles: set[PositionKey],
) -> None:
    """Finish a rebased owner's settlement transaction from its stored records.

    A process that died between two chunks of the settlement transaction left
    the positions of the committed chunks journaled. The positions are rebuilt
    as `plan_settlement` built them: settlement's fields from each position's
    ``ProposalV2`` and settlement record, its primary resolved as settlement
    resolved it, and a head's lifecycle read at the owner's pin. Only
    settlement chunks committed, and they mint no entity and change no head,
    so the primaries are those of the settlement pin; each journaled position
    must rebuild to the digest its chunk recorded. `commit_settlement` then
    writes every position no journal covers.

    Raises:
        QuoteMemError: ``binding_mismatch`` when the rebuilt positions are not the
            ones the settlement settles or one differs from its journal;
            ``target_changed`` when a head left the pin; or a refusal of
            `commit_settlement`.
    """
    writer, owner_run_id, view = owner.writer, owner.owner_run_id, owner.view

    def position(key: PositionKey) -> dict[str, object]:
        reference, proposal = proposals[key]
        return {
            "position_key": key, "extraction_ref": proposal.extraction_ref,
            "proposal_ref": reference, "statement_digest": proposal.statement_digest,
            "entity_id": anchor_entity(proposal.anchors[0], key[1], owner_run_id, view),
            "evidence_ids": sorted({span.evidence_id for span in proposal.quote_spans}),
        }

    def lifecycle(key: PositionKey) -> MemoryLifecycle:
        record = records[key]
        if record.code == "merged_support":
            # A head is unreadable by definition; its memory is active.
            return "active"
        held = None if record.memory_id is None else view.memory(record.memory_id)
        if held is None:
            raise QuoteMemError("target_changed", "A restatement's head left the owner pin")
        return held.lifecycle_state

    def groups(key: PositionKey) -> list[list[QuoteSpanV1]]:
        proposal = proposals[key][1]
        return [[proposal.quote_spans[index] for index in group]
                for group in proposal.support_groups]

    plan = _plan_positions(order, records, unresolved, position=position,
                           spans=lambda key: proposals[key][1].quote_spans, lifecycle=lifecycle,
                           groups=groups)
    journaled = {
        member.position_key: member.resolution_digest
        for _reference, body in writer.find_logical_candidates(owner_run_id, "write_set")
        if isinstance(body, CommitJournalV1)
        and (body.owner_run_id, body.round, body.txn) == (owner_run_id, 1, "settlement")
        for member in body.members
    }
    if {item.position_key for item in plan.settled} != settles or any(
        journaled.get(item.position_key, digest) != digest
        for item in plan.settled for digest in (integrity_digest(item),)
    ):
        raise QuoteMemError("binding_mismatch", "A rebuilt settlement differs from its journal")
    owner.mark_phase("commit")
    owner.authorize()
    commit_settlement(owner.writer, owner.ticket, owner_run_id=owner_run_id,
                      positions=plan.settled, updated_at=owner.clock(),
                      publish=owner.publish, clock=owner.clock)


def _rebased_refusals(
    owner: ExtractionOwner, stored: Mapping[int, tuple[ArtifactId, ExtractContextV2]],
) -> tuple[tuple[RefusedPosition, ...], tuple[SpentUnit, ...]]:
    """A rebased owner's admission refusals and spent units, read from its records.

    A rebased owner completed `admit_units` before its first review, so each unit
    with a stored context ended in one of three ways. Its standing reply, the
    stored reply of its highest call (the unit's request input digest is its
    context's), names every fact admission refused in its admission record. A
    unit whose last started call stored no reply ended with that call's sends
    spent, beside call 0's standing reply when it was the fresh call,
    unless its call is marked split (``UnitCallEndV1``): its pieces stand in its place.
    A unit with neither rendered no readable source and made no call.
    An erased reply's content-free accounting supplies the same call number and
    admission notes, so deleting its body cannot turn an accepted call into a
    spent unit or lose an admission refusal.
    """
    writer, owner_run_id = owner.writer, owner.owner_run_id
    standing: dict[str, StoredInferenceOutput | ErasedReplyV1] = {}
    erased, _proposals = erased_extraction_records(writer, owner_run_id)
    for body in erased.values():
        held = standing.get(body.input_digest)
        if held is None or body.call_no > held.call_no:
            standing[body.input_digest] = body
    for _reference, body in writer.find_logical_candidates(owner_run_id, "llm_output"):
        if not isinstance(body, StoredInferenceOutput) or body.purpose != "extract":
            continue
        held = standing.get(body.input_digest)
        if held is None or body.call_no > held.call_no:
            standing[body.input_digest] = body
    refused: list[RefusedPosition] = []
    spent: list[SpentUnit] = []
    split = {unit_no for unit_no in stored
             if (mark := owner.accounting.ended(unit_no, 0)) is not None
             and mark.outcome == "split"}
    for unit_no, (_reference, record) in sorted(stored.items()):
        sources = tuple(binding.evidence_id for binding in record.unit.member_bindings)
        reply = standing.get(record.input_digest)
        if unit_no not in split and (
            -1 if reply is None else reply.call_no
        ) < owner.accounting.calls_started(unit_no) - 1:
            spent.append(SpentUnit(unit_no, sources))
        if reply is None:
            continue
        notes = () if reply.admission is None else reply.admission.facts
        refused += [RefusedPosition((owner_run_id, unit_no, note.position), REFUSALS[note.rule],
                                    sources) for note in notes]
    return tuple(refused), tuple(spent)


class ExtractionStage:
    """An Add owner's extraction up to its settlement transaction."""

    def run(self, owner: ExtractionOwner) -> ExtractionOutcome:
        """Run the owner's extraction through the settlement transaction, idempotently.

        Units are admitted up to the barrier (`admit_units`); then, at the
        settlement pin, settlement and its record, the vectors of the unsettled
        positions outside any Writer lock, every ``ProposalV2``, and the chunked
        settlement transaction. A resumed owner replays each step:
        stored replies, batches and proposals are reused, and journaled chunks
        are not written again.

        Args:
            owner (ExtractionOwner): The Add owner.

        Returns:
            ExtractionOutcome: The proposals, the settlement record, the
                vectors, the round-1 members with their duplicates, the refused
                positions and the units a provider refusal ended.

        Raises:
            ExtractionRefused: A whole-output fault of a unit; nothing
                commits.
            ProviderError: An extraction or embedding provider failure, which
                pauses the owner.
            QuoteMemError: ``budget_exceeded`` before any call, ``lease_lost`` and the
                owner's other checks, or ``target_changed`` when a held head moved
                before the settlement transaction.
        """
        rebased = rebased_outcome(owner)
        if rebased is not None:
            return rebased
        admitted = admit_units(owner)
        units = admitted.units
        owner.mark_phase("embedding")
        owner.authorize()
        view = owner.settlement_view()
        relearn = owner.policy.parameters.chat_forget == CHAT_FORGET_COMPLETE
        settlement = settle([UnitFacts(unit.unit_no, unit.facts) for unit in units], view,
                            owner_run_id=owner.owner_run_id, relearn=relearn)
        settlement_ref = store_settlement(owner, settlement)
        if settlement.facts:
            # Positions settled without a forgotten name are reviewed as settlement left them.
            units = tuple(replace(unit, facts=tuple(
                settlement.facts.get((owner.owner_run_id, unit.unit_no, fact.position), fact)
                for fact in unit.facts)) for unit in units)
        embedded_positions = _embedding_positions(settlement)
        found = {(owner.owner_run_id, unit.unit_no, fact.position): fact
                 for unit in units for fact in unit.facts}
        control = ControlRowV1.model_validate(owner.writer.get_control("policy:embedding_space"))
        embeddings = embed_texts(
            embedding_texts([(key, found[key].statement, found[key].corrects_text)
                             for key in settlement.order if key in embedded_positions]),
            OwnerEmbedding(owner_run_id=owner.owner_run_id, ticket=owner.ticket,
                           space=view.embedding_space, token=embedding_state_token(control),
                           clock=owner.clock, authorize=owner.authorize),
            view=view, provider=owner.embedder, store=owner.writer,
        )
        owner.authorize()
        proposals = store_proposals(owner, build_proposals(
            units, settlement, embeddings, owner_run_id=owner.owner_run_id,
            policy=owner.policy.parameters, embedding_space=view.embedding_space,
        ))
        plan = plan_settlement(units, settlement, proposals, owner_run_id=owner.owner_run_id)
        owner.mark_phase("commit")
        owner.authorize()
        commit_settlement(owner.writer, owner.ticket, owner_run_id=owner.owner_run_id,
                          positions=plan.settled, updated_at=owner.clock(),
                          publish=owner.publish, clock=owner.clock)
        refused = tuple(
            RefusedPosition((owner.owner_run_id, unit.unit_no, fact.position), fact.code,
                            unit.sources)
            for unit in units for fact in unit.refused
        )
        withheld = tuple(span for position in plan.settled for span in position.withheld)
        records = {record.position_key: record for record in settlement.records}
        placeholders = name_placeholders(view, (
            (found[key].statement, found[key].spans, record.tombstone_id)
            for key, record in records.items() if record.code == "denied_by_forget"))
        return ExtractionOutcome(proposals, settlement_ref, embeddings, plan.review,
                                 plan.duplicates, refused, admitted.spent,
                                 withheld if relearn else (), placeholders if relearn else ())
