"""The lean Add mode's local judge: judge-v8 verdicts without a model call.

The full Add mode reviews every extracted member with the memory judge, a
model call per member or per packet of four. On LoCoMo (C69, arm A) the judge
made 92% of all chat calls and 89% of the chat tokens: about 920 of 1,031 tokens
per source word. Nearly all of that review confirms what a fresh member already
is: 98.75% of member/target pairs were ``independent``, and 74% of members had
no relation, identity, support or form finding at all.

The lean mode (``judge_mode = "local-lean-v1"`` in the frozen extraction
policy) keeps extraction, source labels and every stage around the judge, and
answers each judge-batch-v8 request locally and deterministically:

- every support group is ``supported``, the fidelity ``complete`` and the
  atomicity ``atomic`` (``indivisible`` when the member's form excludes it);
- a ``nominated`` slot is confirmed; an ``open`` slot is ``ambiguous``, as the
  model judged every open slot in C69, so its member settles
  ``identity_unresolved`` exactly as before;
- a sensitive member's attribution is ``direct`` (resolution still requires a
  user-supported quote for it);
- a current memory of the same kind and time whose statement is a near copy
  of the member's (token Jaccard at least ``SAME_JACCARD``) is ``same``;
- when the member states an explicit correction (``judge_structure.explicit_correction``:
  the extraction wrote its ``corrects_text``, or its statement says "<new>, not <old>"), a
  current, non-naming memory its ``revises`` names said strictly before the member's latest
  quote is ``update``, with that quote as ``change_span``: the newer value ends the stale
  memory, which stays in its history, instead of standing beside it as a current fact. Its
  ``supersedes`` link records the explicit correction, so forgetting the correction also
  forgets the value it ended and both are tombstoned (FG-2), as after a ``correct``. A
  bare ``revises`` hint stays ``independent``: updating on it ended memories that still
  hold (CORRECTION-STRAND arm B, BEAM-100K multi-session -8.2), while an explicit
  correction left beside its stale value lost knowledge edits to the old value
  (SMOKE-ONLY: 72% of public MQuAKE-Remastered answers);
- every other target is ``independent``;
- a tombstone whose statement the member nearly restates (Jaccard at least
  ``RESTATES_JACCARD``, or the member holding ``RESTATES_CONTAINMENT`` of its
  tokens) is ``restates``, so a forgotten fact still is not learned again;
  any other tombstone is ``other``.

The verdicts pass the same acceptance (C.7.2), persistence, confirmation votes
and resolution as a model's, so identity, binding, idempotency and the strict
request queue are unchanged. They are recorded with ``LEAN_JUDGE_MODEL`` as both
requested and resolved model and zero usage, so a lean verdict is never mistaken
for, or reused as, a model verdict: the lean judge binding differs from the
model's (``providers/bindings.judge_binding_for_policy``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from aml.core.errors import AMLError
from aml.core.inference import ProviderUsage
from aml.core.judge_output import (
    JUDGE_BATCH_V8_SCHEMA_ID,
    CatalogEntryV8,
    JudgeAnchorV8,
    JudgeBatchOutputV8,
    JudgeBatchReplyItemV8,
    JudgeBatchRequestV8,
    JudgeEntityVerdictV8,
    JudgeForgottenEntryV8,
    JudgeMemberV8,
    JudgeMemoryEntryV8,
    JudgeMessageEntryV8,
    JudgeRelationV8,
    JudgeTombstoneVerdictV8,
    JudgeV8,
    request_constraints,
)
from aml.core.judge_structure import NO_CONSTRAINTS, MemberConstraintsV1, explicit_correction
from aml.core.statement import token_seq
from aml.providers.bindings import LEAN_JUDGE_MODEL
from aml.providers.contracts import InferenceRequest, InferenceResult

SAME_JACCARD: Final = 0.9
"""Token Jaccard from which a current memory of the same kind and time is ``same``.

C69 arm A: at 0.9, 86% of the memory pairs flagged were judged ``same`` by the model.
"""

RESTATES_JACCARD: Final = 0.5
RESTATES_CONTAINMENT: Final = 0.8
"""A tombstone is ``restates`` from this Jaccard, or when the member holds this share of
its tokens; lower than ``SAME_JACCARD`` because relearning a forgotten fact is worse than
a duplicate."""

LEAN_REASON: Final = "Lean local review: no model call; no near-identical target."
LEAN_SAME_REASON: Final = "Lean local review: a current memory states the same fact."
LEAN_UPDATE_REASON: Final = "Lean local review: the member updates the memory its revises names."


def _tokens(text: str) -> frozenset[str]:
    return frozenset(token_seq(text))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _same(member: JudgeMemberV8, words: frozenset[str], entry: CatalogEntryV8) -> bool:
    """Whether a target is a current memory stating the member's fact again."""
    return (
        isinstance(entry, JudgeMemoryEntryV8) and entry.kind == member.kind
        and entry.time == member.time and _jaccard(words, _tokens(entry.statement)) >= SAME_JACCARD
    )


def _restates(words: frozenset[str], entry: CatalogEntryV8) -> bool:
    """Whether the member nearly restates a tombstone's statement."""
    if not isinstance(entry, JudgeForgottenEntryV8):
        return False
    forgotten = _tokens(entry.statement)
    held = len(words & forgotten) / len(forgotten) if forgotten else 0.0
    return _jaccard(words, forgotten) >= RESTATES_JACCARD or held >= RESTATES_CONTAINMENT


def _latest_quote(
    member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8],
) -> tuple[str, str] | None:
    """The member's latest quote handle and its message's ``said_at``; None without a dated one."""
    dated = [(entry.said_at, quote.handle) for quote in member.quotes
             if isinstance(entry := catalog.get(quote.source), JudgeMessageEntryV8)]
    if not dated:
        return None
    said_at, handle = max(dated, key=lambda pair: pair[0])
    return handle, said_at


def _revised(member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8],
             said_at: str) -> frozenset[str]:
    """The issued current memories the member's ``revises`` names, said before ``said_at``.

    A naming memory is left out: it changes only through a naming fact of its entity (G5),
    which the lean judge does not decide.
    """
    return frozenset(handle for handle in member.revises if handle in member.targets
                     and isinstance(entry := catalog[handle], JudgeMemoryEntryV8)
                     and not entry.naming and entry.said_at < said_at)


def _entity(anchor: JudgeAnchorV8) -> JudgeEntityVerdictV8:
    verdict = "confirm" if anchor.binding == "nominated" else "ambiguous"
    return JudgeEntityVerdictV8(slot=anchor.slot, verdict=verdict, choice=None)


def lean_verdict(
    member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8],
    constraints: MemberConstraintsV1 = NO_CONSTRAINTS,
) -> JudgeV8:
    """Return the lean local verdict of one member (see the module docstring).

    Args:
        member: The member as its request lists it.
        catalog: The request's catalog; it holds every handle the member reaches.
        constraints: The values the member's own form excludes (C.6.4).

    Returns:
        JudgeV8: A verdict that judge-v8 acceptance (C.7.2) accepts for the member.
    """
    words = _tokens(member.statement)
    same_handles = {handle for handle in member.targets if _same(member, words, catalog[handle])}
    latest = (_latest_quote(member, catalog)
              if member.correction_requested
              and explicit_correction(member.statement, member.corrects_text) else None)
    updated = (_revised(member, catalog, latest[1]) - same_handles) if latest else frozenset[str]()
    relations = [JudgeRelationV8(
        target=handle, relation="same" if handle in same_handles else (
            "update" if handle in updated else "independent"),
    ) for handle in member.targets]
    tombstones = [JudgeTombstoneVerdictV8(
        tombstone=handle,
        verdict="restates" if _restates(words, catalog[handle]) else "other",
    ) for handle in member.tombstones]
    atomicity = "indivisible" if "atomic" in constraints.atomicity else "atomic"
    attribution = None
    if member.sensitivity_classes:
        attribution = "none" if "direct" in constraints.attribution else "direct"
    same = bool(same_handles)
    reason = LEAN_UPDATE_REASON if updated else LEAN_SAME_REASON if same else LEAN_REASON
    return JudgeV8(
        support="supported", group_support=["supported"] * len(member.support_groups),
        fidelity="complete", atomicity=atomicity, sensitive_attribution=attribution,
        entities=[_entity(anchor) for anchor in member.anchors
                  if anchor.binding in ("nominated", "open")],
        relations=relations, restates_history=[], tombstones=tombstones,
        # Without a change relation the field is asked exactly when it is allowed.
        external_only=False if member.external_only_allowed and not updated else None,
        change_span=latest[0] if updated and latest else None, reason=reason,
    )


def lean_reply(request: JudgeBatchRequestV8) -> JudgeBatchOutputV8:
    """Answer one judge-batch-v8 request with each member's lean verdict, in order."""
    constraints = request_constraints(request)
    return JudgeBatchOutputV8(items=[JudgeBatchReplyItemV8(
        member_id=item.member_id,
        result=lean_verdict(item.member, request.catalog,
                            constraints.get(item.member_id, NO_CONSTRAINTS),
                            ).model_dump(mode="json"),
    ) for item in request.items])


class LeanLocalJudge:
    """A ``StructuredInferenceProvider`` that answers lean judge requests locally.

    It accepts only requests of a lean judge binding (``LEAN_JUDGE_MODEL``) and
    makes no network call; the result has zero usage.
    """

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Return the lean reply to one physical judge-batch-v8 request.

        Raises:
            AMLError: ``binding_mismatch`` for any other request.
        """
        if (request.purpose != "judge" or request.schema_id != JUDGE_BATCH_V8_SCHEMA_ID
                or request.requested_model != LEAN_JUDGE_MODEL):
            raise AMLError("binding_mismatch", "The lean judge answers only lean judge requests")
        parsed = JudgeBatchRequestV8.model_validate_json(request.messages[-1].content)
        return InferenceResult(
            resolved_model=LEAN_JUDGE_MODEL, schema_id=JUDGE_BATCH_V8_SCHEMA_ID,
            output=lean_reply(parsed), usage=ProviderUsage(input_tokens=0, output_tokens=0),
        )

