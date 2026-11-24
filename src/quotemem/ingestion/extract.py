"""One unit's bounded extract-v12 call, its mechanical admission and its stored reply.

The stage persists or reconstructs the frozen context before calling
``extract``. The transport owns the only retry loop: network retries and
schema regenerations share the policy's send count and logical deadline.
The caller supplies the durable allowance and current grant/lease/privacy
check. Admission leaves the reply untouched and attaches its record to the
result, which is the complete ``llm_output`` content.

`WriterReplies` is the stage's create-once reply storage (``ReplyStore``):
each logical call's accepted reply and its
``ExtractionAdmissionV2`` share one ``llm_output`` of the Add owner through the
single Writer (stages 4–6), keyed by the unit's input digest, the
frozen binding and the call number, and a stored reply reads back exactly what was stored.
"""

from __future__ import annotations

import contextvars
import logging
import re
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_output import ExtractOutputV12, FactV12, extraction_messages
from quotemem.core.extraction_policy import (
    QUESTION_SUPPORT_RULE,
    ExtractionPolicyConfigV2,
    agent_release,
    extraction_pipeline,
)
from quotemem.core.extraction_records import ExtractionRefused, WholeOutputFault
from quotemem.core.extraction_supplement import IMPLIED_HEALTH_RULES, SUPPLEMENTS
from quotemem.core.identity import ArtifactId, canonical_json, integrity_digest
from quotemem.core.inference import InferenceBindingV1, ProviderUsageV1, StoredInferenceOutput
from quotemem.core.provider_diagnostics import regenerable_refusal
from quotemem.core.session_rollup import (
    ROLLUP_MAX_TOKENS,
    ROLLUP_PROMPT,
    ROLLUP_PROMPT_VERSION,
    ROLLUP_SCHEMA_ID,
    ROLLUP_TIMEOUT_MS,
    SessionRollupV1,
    rollup_fact,
    rollup_message,
    rollup_quote,
)
from quotemem.core.statement import fit_statement
from quotemem.ingestion.context import ExtractionContext, build_request
from quotemem.ingestion.extraction_admission import Admission, admit
from quotemem.providers.adapters import ProviderError
from quotemem.providers.attempts import (
    current_provider_attempt,
    provider_attempts,
    provider_attempts_from,
)
from quotemem.providers.bindings import inference_binding_for_policy
from quotemem.providers.contracts import (
    InferenceRequest,
    InferenceResult,
    Message,
    OutputIdentity,
    StructuredInferenceProvider,
    validate_binding,
)
from quotemem.providers.reuse import stored_output, stored_result

if TYPE_CHECKING:
    from quotemem.ingestion.extraction_stage import StoredReply, UnitCallNumber
    from quotemem.store.writer import Writer, WriteTicket

_LOG = logging.getLogger(__name__)
_DECLARED = re.compile(r"[NR][0-9]+")
"""A handle that a reply declares itself (new entities, unresolved referents)."""

ExtractionAllowance = Callable[[InferenceBindingV1], AbstractContextManager[None]]
"""Enter the unit call's durable send/deadline allowance for its frozen binding.

The owner binds the unit and logical-call identity into this callable. Its
context must reserve every physical send through ``provider_attempts`` and
bound the call by the persisted deadline through ``request_budget``
(``store.extraction_budget.extraction_allowance``). It must preserve spent sends on interruption. It
is required, including for explicit resume; this module
neither substitutes an in-memory allowance nor resets a persisted allowance.
"""


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """A completed per-unit call, admitted before any output persistence.

    Attributes:
        binding: The frozen provider binding derived from the owner's policy.
        request: The exact request and frozen context's keyed provenance.
        result: The verbatim typed reply and provider accounting, carrying
            ``admission.record`` as its admission record: the complete
            ``llm_output`` content.
        admission: Admitted/refused positions and the content-free record
            stored beside the reply. A valid empty reply returns normally;
            The extraction stage alone decides whether its zero-proposal rule needs a fresh call.
    """

    binding: InferenceBindingV1
    request: InferenceRequest
    result: InferenceResult
    admission: Admission


def _whole_output_fault(error: ProviderError) -> WholeOutputFault | None:
    refusal = error.schema_refusal
    if error.error_class != "schema_invalid" or refusal is None:
        return None
    if refusal.reason == "output_bound":
        return "output_bound"
    if not regenerable_refusal(refusal):
        return None
    return "truncated" if refusal.reason == "truncated" else "provider_schema_invalid"


def extraction_request(
    context: ExtractionContext, policy: ExtractionPolicyConfigV2,
) -> tuple[InferenceBindingV1, InferenceRequest]:
    """Build a unit's frozen binding and its exact extract-v12 request.

    The request is a pure function of the frozen context and the owner's
    policy, so a resumed or replayed unit rebuilds the same bytes.

    Args:
        context: the selected or reconstructed context.
        policy: The owner's frozen extraction config.

    Returns:
        The binding the policy derives and the request it binds.

    Raises:
        QuoteMemError: binding_mismatch if the request leaves its binding.
    """
    pipeline = extraction_pipeline(policy.parameters.pipeline_version)
    binding = inference_binding_for_policy(policy)
    system, user = extraction_messages(
        build_request(context), policy.parameters.prompt_template_digest,
    )
    request = InferenceRequest(
        purpose="extract",
        provider=pipeline.provider,
        requested_model=binding.requested_model,
        schema_id=binding.schema_id,
        prompt_version=binding.prompt_version,
        input_digest=context.record.input_digest,
        context_digest=context.record.context_digest,
        messages=[Message(role="system", content=system), Message(role="user", content=user)],
        timeout_ms=binding.timeout_ms,
        retry_policy=binding.retry_policy,
        call_budget=binding.call_budget,
        retry_output_schema=binding.retry_output_schema,
        reasoning_effort=binding.reasoning_effort,
    )
    validate_binding(request, binding)
    return binding, request


def supplement_request(
    context: ExtractionContext, policy: ExtractionPolicyConfigV2,
) -> InferenceRequest | None:
    """Build a unit's supplement request beside its extraction request, or None.

    The supplement sends the extraction request's user message under its rule's system
    message, without reasoning (``core/extraction_supplement.py``), so it is a pure
    function of the frozen context and policy too.

    Args:
        context: the selected or reconstructed context.
        policy: The owner's frozen extraction config.

    Returns:
        The supplement request, or None when the policy names no supplement rule.
    """
    rule = policy.parameters.extraction_supplement
    if rule is None:
        return None
    system, effort = SUPPLEMENTS[rule]
    _, request = extraction_request(context, policy)
    return request.model_copy(update={
        "messages": [Message(role="system", content=system), request.messages[1]],
        "reasoning_effort": effort, "thinking": "disabled",
    })


def merged_reply(reply: ExtractOutputV12, supplement: ExtractOutputV12) -> ExtractOutputV12:
    """Append the supplement's facts to an extraction reply, before its admission.

    The extraction reply alone declares new entities and unresolved referents, so a
    supplement fact naming a handle that a reply declares (``N`` or ``R``) is left out:
    in the joint reply that handle would mean the extraction reply's entity. A supplement
    statement over the statement bound (an overview recounting a long source, most often)
    is cut at its last clause within it (``core/statement.fit_statement``) rather than
    refused at admission. The joint
    reply keeps the structural bounds of a single reply; when it would break them, the
    extraction reply stands alone.

    Args:
        reply: The extraction call's verbatim reply.
        supplement: The supplement call's verbatim reply.

    Returns:
        The extraction reply with the supplement's facts after its own, from
        ``supplement_from``, so admission marks them ``extraction_supplement`` (an
        implied health statement under its rule ``implied_health``).
    """
    kept = [
        fact for fact in supplement.facts
        if not any(_DECLARED.fullmatch(handle)
                   for handle in (*fact.anchors, *(naming.entity for naming in fact.naming)))
    ]
    if not kept:
        return reply
    body = reply.model_dump(mode="json")
    body["supplement_from"] = len(reply.facts)
    body["facts"] += [
        fact.model_copy(update={"statement": fit_statement(fact.statement)}).model_dump(mode="json")
        for fact in kept
    ]
    try:
        return ExtractOutputV12.model_validate(body)
    except ValidationError:
        return reply


def _fresh_span(context: ExtractionContext) -> tuple[int | None, int | None]:
    """The earliest and latest source time of the unit's fresh sources."""
    times = [source.observed_at for source in context.sources
             if source.fresh and source.observed_at is not None]
    return (min(times), max(times)) if times else (None, None)


def rollup_request(
    context: ExtractionContext, supplement: InferenceRequest, reply: ExtractOutputV12,
) -> InferenceRequest | None:
    """Build the session rollup request over a supplement reply, or None without its facts.

    It reads the session's newest rollup (``context.rollup``) and the supplement's statements
    about the fresh sources, with their span, less any entry or note restating a forgotten
    statement (``context.forgotten``, ``core/session_rollup.py``), and goes to the
    supplement's model without reasoning, under the supplement request's digests.

    Args:
        context: The unit's frozen context.
        supplement: The unit's supplement request.
        reply: The supplement call's reply.

    Returns:
        The rollup request, or None when the reply states no fact.
    """
    if not reply.facts:
        return None
    first, last = _fresh_span(context)
    message = rollup_message(context.rollup, [fact.statement for fact in reply.facts], first, last,
                             context.forgotten)
    return InferenceRequest.model_validate(supplement.model_copy(update={
        "schema_id": ROLLUP_SCHEMA_ID, "prompt_version": ROLLUP_PROMPT_VERSION,
        "messages": [Message(role="system", content=ROLLUP_PROMPT),
                     Message(role="user", content=message)],
        "timeout_ms": ROLLUP_TIMEOUT_MS, "call_budget": None,
        "retry_output_schema": False, "max_output_tokens": ROLLUP_MAX_TOKENS,
    }).model_dump(mode="python", exclude_unset=False))


def session_rollup(
    context: ExtractionContext, reply: ExtractOutputV12, entries: Sequence[str],
) -> FactV12 | None:
    """The rollup fact of a rollup reply: its span runs from the session's first day.

    Args:
        context: The unit's frozen context.
        reply: The supplement call's reply, whose keywords the fact takes.
        entries: The rollup call's timeline, oldest first.

    Returns:
        The fact, quoting the unit's first fresh user source, or None without such a
        source or when nothing of it can stand (`rollup_fact`).
    """
    source = next((source for source in context.sources
                   if source.fresh and source.attribution.authenticated_self
                   and any(text.strip() for text in source.readable_texts())), None)
    if source is None:
        return None
    text = next(text for text in source.readable_texts() if text.strip())
    first, last = _fresh_span(context)
    prior = context.rollup
    start = first if prior is None or prior.start is None else prior.start
    return rollup_fact(entries, source.source_id, rollup_quote(text), reply, start, last,
                       context.forgotten)


def with_rollup(reply: ExtractOutputV12, rollup: FactV12) -> ExtractOutputV12:
    """Append the rollup fact after a merged reply's supplement facts, before admission.

    Args:
        reply: A `merged_reply` result.
        rollup: The session rollup's fact.

    Returns:
        The reply with the rollup as its last fact, from ``rollup_from``, so admission
        marks it ``session_rollup``; the reply unchanged when it holds no supplement fact
        or the joint reply would break a structural bound.
    """
    if reply.supplement_from is None:
        return reply
    body = reply.model_dump(mode="json")
    body["rollup_from"] = len(reply.facts)
    body["facts"].append(rollup.model_dump(mode="json"))
    try:
        return ExtractOutputV12.model_validate(body)
    except ValidationError:
        return reply


def _supplement(
    provider: StructuredInferenceProvider, request: InferenceRequest,
    check: Callable[[], None] | None,
) -> InferenceResult | None:
    """Send the supplement request; its failure leaves the extraction reply standing.

    It runs under exactly the attempt check bound before the extraction call's allowance:
    every send still checks grant, lease, cancellation and privacy, and none spends the
    extraction call's own sends, so a slow or refused supplement never costs extraction a
    retry. The copied context keeps the call's request budget, which bounds its time.
    """
    try:
        with provider_attempts_from(check):
            result = provider.infer(request)
    except Exception as error:  # a supplement never fails its extraction call
        _LOG.warning("Extraction supplement dropped: %s", type(error).__name__)
        return None
    return result if isinstance(result.output, ExtractOutputV12) else None


def _rollup(
    provider: StructuredInferenceProvider, request: InferenceRequest,
    check: Callable[[], None] | None,
) -> InferenceResult | None:
    """Send the rollup request like the supplement's: its failure leaves no rollup."""
    try:
        with provider_attempts_from(check):
            result = provider.infer(request)
    except Exception as error:  # a rollup never fails its extraction call
        _LOG.warning("Session rollup dropped: %s", type(error).__name__)
        return None
    return result if isinstance(result.output, SessionRollupV1) else None


@dataclass(frozen=True, slots=True)
class _Supplemented:
    """The supplement call's result and the rollup call's result after it."""

    supplement: InferenceResult | None
    rollup: InferenceResult | None


def _supplemented(
    provider: StructuredInferenceProvider, request: InferenceRequest,
    check: Callable[[], None] | None, context: ExtractionContext,
) -> _Supplemented:
    """Send the supplement, then the session rollup over its reply, one after the other."""
    supplement = _supplement(provider, request, check)
    if supplement is None or not isinstance(supplement.output, ExtractOutputV12):
        return _Supplemented(supplement, None)
    follow = rollup_request(context, request, supplement.output)
    return _Supplemented(supplement, None if follow is None else _rollup(provider, follow, check))


def _start_supplement(
    provider: StructuredInferenceProvider, request: InferenceRequest | None,
    check: Callable[[], None] | None, context: ExtractionContext,
) -> Future[_Supplemented] | None:
    """Start the supplement and rollup calls in a copy of the current context, at the same
    time as extraction."""
    if request is None:
        return None
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quotemem-supplement")
    try:
        return pool.submit(
            contextvars.copy_context().run, _supplemented, provider, request, check, context,
        )
    finally:
        pool.shutdown(wait=False)


def extract(
    context: ExtractionContext,
    policy: ExtractionPolicyConfigV2,
    provider: StructuredInferenceProvider,
    *,
    authorize: Callable[[], None],
    allowance: ExtractionAllowance,
) -> ExtractionResult | None:
    """Send one logical extraction call for a frozen unit and admit its reply.

    Args:
        context: the selected or reconstructed context, already persisted.
        policy: The owner's frozen extraction config, including its template and definitions.
        provider: the structured provider, which owns retries and regeneration.
        authorize: Recheck grant, lease, cancellation and privacy validity. Called
            before allowance creation, before every physical send (including
            retries), after the reply and after admission. Failures propagate.
        allowance: Bind the durable reservation and deadline for this unit and
            logical call, using the supplied frozen binding. Authorization wraps
            the allowance so a revoked send fails before its reservation.

    Returns:
        The verbatim result and pure admission, or None without allocating an
        allowance or invoking the provider when no readable fresh source exists.

    Raises:
        ExtractionRefused: A content fault after the transport finishes its
            allowed regenerations; output_bound is never regenerated. The
            original ProviderError remains the cause with its diagnostics.
        ProviderError: Network failures and invalid provider metadata/envelopes;
            these pause the owner rather than refusing its extraction output.
        QuoteMemError: Revoked authority, exhausted owner allowance, or binding mismatch.

    This function performs no persistence. The integration must replay any
    stored reply with ``replay_admission`` before invoking it, and atomically
    recheck authority at the Writer boundary before storing reply and admission.

    A policy with a supplement rule sends its supplement request at the same time
    (`supplement_request`), outside the call's send allowance, and waits for both; the
    supplement's facts join the reply before admission (`merged_reply`) and its tokens
    the reply's usage, so the stored reply is the one replay reads. A supplement that
    fails is left out and the extraction reply stands alone. As soon as the supplement
    answers, the session rollup request follows it (`rollup_request`), still beside the
    extraction call; its fact follows the supplement's (`with_rollup`) and its tokens join
    the usage too. A rollup that fails, or whose fact cannot stand, is left out.
    """
    authorize()
    if not any(source.fresh and source.readable_texts() for source in context.sources):
        return None
    pipeline = extraction_pipeline(policy.parameters.pipeline_version)
    binding, request = extraction_request(context, policy)
    extra = supplement_request(context, policy)
    supplemented = _Supplemented(None, None)
    try:
        with provider_attempts(authorize):
            check = current_provider_attempt()
            with allowance(binding):
                pending = _start_supplement(provider, extra, check, context)
                result = provider.infer(request)
                authorize()
                if pending is not None:
                    supplemented = pending.result()
    except ProviderError as error:
        fault = _whole_output_fault(error)
        if fault is None:
            raise
        raise ExtractionRefused(
            fault, "Extraction reply failed its whole-output contract",
        ) from error
    if result.schema_id != binding.schema_id or not isinstance(result.output, ExtractOutputV12):
        raise QuoteMemError("binding_mismatch",
                            "Extraction provider returned another output contract")
    output, usage = result.output, result.usage
    supplement, rollup = supplemented.supplement, supplemented.rollup
    if supplement is not None and isinstance(supplement.output, ExtractOutputV12):
        output = merged_reply(result.output, supplement.output)
        usage = ProviderUsageV1(
            input_tokens=usage.input_tokens + supplement.usage.input_tokens,
            output_tokens=usage.output_tokens + supplement.usage.output_tokens,
        )
        if rollup is not None and isinstance(rollup.output, SessionRollupV1):
            usage = ProviderUsageV1(
                input_tokens=usage.input_tokens + rollup.usage.input_tokens,
                output_tokens=usage.output_tokens + rollup.usage.output_tokens,
            )
            fact = session_rollup(context, supplement.output, rollup.output.entries)
            if fact is not None:
                output = with_rollup(output, fact)
    admission = admit(
        output, context, pipeline.profile,
        pipeline_version=policy.parameters.pipeline_version,
        implied_health_rule=policy.parameters.extraction_supplement in IMPLIED_HEALTH_RULES,
        self_identity=policy.parameters.self_identity is not None,
        question_support_rule=policy.parameters.quote_grounding == QUESTION_SUPPORT_RULE,
        plain_corrections=agent_release(policy),
    )
    if agent_release(policy):
        log_refusals(context.record.input_digest, admission)
    authorize()
    admitted = InferenceResult(
        resolved_model=result.resolved_model, schema_id=result.schema_id, output=output,
        usage=usage, provider_request_id=result.provider_request_id,
        provider_attempts=result.provider_attempts, admission=admission.record,
    )
    return ExtractionResult(binding, request, admitted, admission)


def log_refusals(unit: str, admission: Admission) -> None:
    """Log each fact admission refused, with the rule that refused it.

    One warning a refused position: the unit's input digest, the position, its rule and
    reason code, and the rule of each of its support groups that admission dropped and of
    each anchor it degraded. Content-free: no statement, quote or name, so the service log of
    an ``agent`` workspace says why a correction never reached the judge.
    """
    record = admission.record
    if record is None:
        return
    for refused in admission.refused:
        groups = [f"{note.group}:{note.rule}" for note in record.groups
                  if note.position == refused.position]
        anchors = [f"{note.anchor}:{note.rule}" for note in record.anchors
                   if note.position == refused.position]
        _LOG.warning("Extraction item refused: unit=%s position=%d rule=%s code=%s groups=[%s] "
                     "anchors=[%s]", unit[:16], refused.position, refused.rule, refused.code,
                     ",".join(groups), ",".join(anchors))


@dataclass(frozen=True, slots=True)
class WriterReplies:
    """Create-once storage of each call's accepted reply through the Writer (``ReplyStore``).

    A call's reply is the Add owner's ``llm_output`` whose input digest and
    binding are the unit's and whose ``call_no`` is the call's: the
    verbatim reply, its provider accounting and its admission record
    in one body, chunked with ``part`` when large. A unit's call 0 and
    its fresh call 1 send the identical request, so the call number keeps
    their replies apart and the write-once check holds per call. Lookup
    rebuilds each body from the unit's frozen request, so a stored reply that
    differs from its binding, context or request fails.

    Attributes:
        writer: The single Writer.
        ticket: The Add owner's ticket; the owner creates every reply.
        policy: The owner's frozen extraction config.
        authorize: Rechecks grant, lease, cancellation and privacy validity
            right before a reply is written.
        clock: The trusted time in epoch milliseconds, for a new reply.
    """

    writer: Writer
    ticket: WriteTicket
    policy: ExtractionPolicyConfigV2
    authorize: Callable[[], None]
    clock: Callable[[], int]

    def _identity(
        self, binding: InferenceBindingV1, request: InferenceRequest, call: UnitCallNumber,
    ) -> OutputIdentity:
        if self.ticket.run_id is None:
            raise QuoteMemError("invalid_write_mode", "A unit's reply needs its admitted Add owner")
        return OutputIdentity(
            run_id=self.ticket.run_id, purpose="extract", input_digest=request.input_digest,
            inference_binding_digest=integrity_digest(binding), call_no=call,
        )

    def _lookup(
        self, binding: InferenceBindingV1, request: InferenceRequest,
    ) -> dict[UnitCallNumber, tuple[ArtifactId, StoredInferenceOutput]]:
        identity = self._identity(binding, request, 0)
        found: dict[UnitCallNumber, tuple[ArtifactId, StoredInferenceOutput]] = {}
        for reference, body in self.writer.find_logical_candidates(identity.run_id,
                                                                   "llm_output"):
            if not isinstance(body, StoredInferenceOutput) or (
                body.purpose, body.input_digest, body.inference_binding_digest,
            ) != (identity.purpose, identity.input_digest, identity.inference_binding_digest):
                continue
            if body.call_no in found:
                raise QuoteMemError("artifact_digest_collision",
                                    "A unit's call has more than one stored reply")
            call_identity = self._identity(binding, request, body.call_no)
            if body != stored_output(call_identity, request, stored_result(body)):
                raise QuoteMemError("binding_mismatch",
                                    "A stored reply differs from its frozen request")
            found[body.call_no] = (reference, body)
        return found

    def stored(self, context: ExtractionContext) -> tuple[StoredReply, ...]:
        """Return the unit's stored replies with their admission records, in call order.

        Args:
            context: The unit's frozen context.

        Returns:
            Each stored call's verbatim reply and its record exactly as stored
            (None when admission changed nothing), at most one per call and
            empty before the first is stored; the stage replays admission
            against them.

        Raises:
            QuoteMemError: binding_mismatch if a stored body leaves the unit's
                frozen request; artifact_digest_collision for two replies of
                one call.
        """
        from quotemem.ingestion.extraction_stage import StoredReply

        binding, request = extraction_request(context, self.policy)
        replies: list[StoredReply] = []
        for call, (reference, body) in sorted(self._lookup(binding, request).items()):
            if not isinstance(body.output, ExtractOutputV12):
                raise QuoteMemError("binding_mismatch",
                                    "A unit's stored reply has another output contract")
            replies.append(StoredReply(reference, call, body.output, body.admission))
        return tuple(replies)

    def store(
        self, context: ExtractionContext, call: UnitCallNumber, extraction: ExtractionResult,
    ) -> ArtifactId:
        """Store one logical call's accepted reply with its admission record, once.

        Storing the same reply again for the call returns its ``llm_output``;
        another body at the call's identity is refused.

        Args:
            context: The unit's frozen context.
            call: The logical call that made the reply.
            extraction: The call's admitted reply from `extract`.

        Returns:
            The reply's ``llm_output``.

        Raises:
            QuoteMemError: binding_mismatch if the reply was made for another
                request or lost its admission record;
                artifact_digest_collision if the call stored another reply;
                the owner's authorization and Writer refusals.
        """
        binding, request = extraction_request(context, self.policy)
        if extraction.binding != binding or extraction.request != request:
            raise QuoteMemError("binding_mismatch", "A reply belongs to another unit request")
        if extraction.result.admission != extraction.admission.record:
            raise QuoteMemError("binding_mismatch", "A reply lost its admission record")
        body = stored_output(self._identity(binding, request, call), request, extraction.result)
        found = self._lookup(binding, request).get(call)
        if found is not None:
            reference, existing = found
            if canonical_json(existing) != canonical_json(body):
                raise QuoteMemError(
                    "artifact_digest_collision", "The unit's call already stored another reply",
                )
            return reference
        self.authorize()
        return self.writer.put_logical_artifact(
            self.ticket, "llm_output", body, security_level="restricted",
            created_at=self.clock(),
        )
