"""The targeted judge mode's gate: which members the memory judge model reviews.

The targeted mode (``judge_mode`` ``targeted-v1``) sends a member to the batched
memory judge only when its review can change something the lean local verdict
(``ingestion/lean_judge.py``) would not: a near target it may repeat, supplement,
update or correct, a change it announces, a plan it may have carried out, a
sensitive attribution, or a forgotten fact it may restate (a change cue also with a same-Add
sibling under the ``agent`` gate's ``sibling_cues``), or one its user told again (`gated`).
Every other member is reviewed locally, as in the lean mode. The rules (`gate_reasons`)
read only the member input and the target cosines nomination already holds
(`target_cosines`), so the gate makes no call, and they are frozen in the extraction policy's
``JudgeGateV1`` (``core/extraction_policy.py``).

The LoCoMo calibration run (all ten conversations, 15,083 member inputs) fixed the default
bars. Its judge found ``same``, ``supplement``, ``update`` or ``correct`` for 14%
of members, yet most members have a close target: half of them a head or sibling
at cosine 0.85 or more. Cosine alone therefore skips at most a fifth to a third of
members at the recall wanted; the cue, plan, sensitivity and tombstone rules lift
that to 43% at update+correct recall 0.97 and same+supplement recall 0.90.

`TargetedJudgeStage` is the round's judge stage: gated members go to the model
stage under the policy's judge binding and the batched packing, the rest to the
local stage under the lean binding, so a lean verdict is never mistaken for, or
reused as, a model verdict. A member's input fixes its gate, so a
member keeps its binding across replay and recovery.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal

from quotemem.core.entities import PositionKey
from quotemem.core.extraction_policy import JudgeGateV1
from quotemem.core.judge_output import (
    CatalogEntryV8,
    JudgeMemberV8,
    JudgeMemoryEntryV8,
    JudgeSiblingEntryV8,
)
from quotemem.core.review_rounds import RoundNo
from quotemem.core.vector_score import cosine, unit_vector
from quotemem.ingestion.judge import JudgedMemberV1, JudgeStage, ReviewMemberV1

GateReason = Literal[
    "head_similar", "sibling_similar", "correction_flagged", "change_cue", "plan_event",
    "sensitive", "tombstone",
]
"""Why a member goes to the model judge; a member without a reason is reviewed locally."""

CHANGE_CUES: Final = re.compile(
    r"""\b(now|nowadays|these\s+days|currently|any\s?more|no\s+longer|not\s+\w+\s+anymore|
    actually|instead|turns?\s+out|used\s+to|finally|lately|recently|at\s+last|since\s+then|
    moved|moving|relocat\w*|changed?|switch(ed|ing)?|quit|quitting|stopp?(ed|ing)|start(ed|ing)|
    began|begun|became|become|left|joined|retired|graduated|promoted|hired|fired|sold|bought|
    adopted|broke\s+up|divorced|married|engaged|cancel+ed|postponed|resched\w*|finished|completed|
    new\s+(job|house|home|place|apartment|car|role|position)|
    correction|update|i\s+meant|to\s+clarify|mistake|wrong)\b""",
    re.IGNORECASE | re.VERBOSE,
)
"""``change-cues-v1``: closed words of a changed or corrected fact in a member's statement.

The update adverbs and the ``correction``/``update`` lead-ins follow the resend lexicon
(``quotemem.ingestion.resend_lexicon``); the verbs name a state ending or starting (moved, quit,
started, married, ...). A cue counts only with a target memory on a shared entity."""

PLAN_FORM: Final = re.compile(
    r"""\b(plan(s|ned|ning)?\s+to|planning|going\s+to|gonna|will|want(s|ed)?\s+to|
    hop(e|es|ed|ing)\s+to|looking\s+forward|can'?t\s+wait|cannot\s+wait|intend(s|ed)?|
    thinking\s+(about|of)|consider(s|ing)?|next\s+(week|month|year|weekend|summer|spring|fall|winter)|
    soon|upcoming|aim(s|ing)?\s+to|would\s+like\s+to|dream(s|ing)?\s+of|wish(es)?\s+to|eager\s+to|
    excited\s+(to|about|for))\b""",
    re.IGNORECASE | re.VERBOSE,
)
"""``change-cues-v1``: a stored statement in plan or future form (``can't wait for the
tour``), which a later event on the same entity (``started touring``) may carry out."""


def _shared_heads(member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8],
                  ) -> list[JudgeMemoryEntryV8]:
    """The member's target memories that share one of its bound entities."""
    entities = {anchor.entity for anchor in member.anchors
                if anchor.entity is not None and anchor.binding != "open"}
    return [entry for handle in member.targets
            if isinstance(entry := catalog[handle], JudgeMemoryEntryV8)
            and not entities.isdisjoint(entry.about)]


def _shared_siblings(member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8]) -> bool:
    """Whether a same-Add sibling target shares one of the member's bound entities."""
    entities = {anchor.entity for anchor in member.anchors
                if anchor.entity is not None and anchor.binding != "open"}
    return any(isinstance(entry := catalog[handle], JudgeSiblingEntryV8)
               and not entities.isdisjoint(entry.about) for handle in member.targets)


def _near(member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8],
          cosines: Mapping[str, float], entry_type: type[CatalogEntryV8], permille: int) -> bool:
    """Whether a target of this entry type is at cosine ``permille / 1000`` or more.

    Cosines carry nomination's six decimals, so the bar compares exact integers.
    """
    return any(isinstance(catalog[handle], entry_type) and handle in cosines
               and round(cosines[handle] * 1_000_000) >= permille * 1000
               for handle in member.targets)


def gate_reasons(
    member: JudgeMemberV8, catalog: Mapping[str, CatalogEntryV8], cosines: Mapping[str, float],
    gate: JudgeGateV1,
) -> tuple[GateReason, ...]:
    """Return why the member goes to the model judge, in rule order; empty when it does not.

    Args:
        member: The member as its request lists it.
        catalog: Its catalog; it holds every handle the member reaches.
        cosines: The cosine of each target handle to the member (`target_cosines`); a
            target without one is not near.
        gate: The policy's frozen gate.

    Returns:
        tuple[GateReason, ...]: Each rule that holds (see ``JudgeGateV1``).
    """
    found: list[GateReason] = []
    if _near(member, catalog, cosines, JudgeMemoryEntryV8, gate.head_cosine_permille):
        found.append("head_similar")
    if _near(member, catalog, cosines, JudgeSiblingEntryV8, gate.sibling_cosine_permille):
        found.append("sibling_similar")
    if member.correction_requested or member.corrects_text is not None or member.revises:
        found.append("correction_flagged")
    shared = _shared_heads(member, catalog)
    cue = CHANGE_CUES.search(member.statement) is not None
    if cue and (shared or gate.sibling_cues and _shared_siblings(member, catalog)):
        found.append("change_cue")
    if member.kind == "event" and any(PLAN_FORM.search(entry.statement) for entry in shared):
        found.append("plan_event")
    if member.sensitivity_classes:
        found.append("sensitive")
    if member.tombstones:
        found.append("tombstone")
    return tuple(found)


def target_cosines(member: ReviewMemberV1) -> dict[str, float]:
    """Each issued target's cosine to the member, by its handle in the member input.

    A head or retired version is scored by its stored embedding, a sibling by its
    statement vector, all rounded as nomination rounds them.

    Args:
        member: The reviewed member with its nomination and input.

    Returns:
        dict[str, float]: Cosine by target handle.
    """
    handles = {key: handle for handle, key in member.input.identities.items()}
    query = unit_vector(member.nomination.member.vector)
    found: dict[str, float] = {}
    for target in member.nomination.targets:
        handle = handles.get(target.handle_key)
        if handle is None:
            continue
        if target.sibling is not None:
            vector: Sequence[float] = target.sibling.vector
        elif target.version is not None:
            vector = target.version.snapshot.embedding
        else:
            continue
        found[handle] = cosine(query, unit_vector(vector))
    return found


def gated(member: ReviewMemberV1, gate: JudgeGateV1) -> bool:
    """Whether the targeted mode sends the member to the model judge: a `gate_reasons` rule
    holds, or nomination left a tombstone out because the member's user told the fact
    again after its forget (the ``agent`` mode's ``relearned``). The relearn
    check reads words and vectors only, so the judge, not the local verdict that supports
    every group, decides whether the user's later words support what the member says: a
    member stating the age of a forgotten "The user's tabby cat Pixel is four years old."
    from "My cat is called Pixel.", whose vector lies near it, is the judge's to refuse.
    The relearned tombstones follow from the member's quotes and the pin, so the gate stays
    fixed across replay."""
    return bool(member.nomination.relearned) or bool(gate_reasons(
        member.input.member, member.input.catalog, target_cosines(member), gate))


class TargetedJudgeStage:
    """The targeted mode's judge stage: gated members by the model, the others locally.

    Both stages are ordinary `JudgeStage` rounds over one owner's records: the model
    stage under the policy's judge binding and batched packing, the local stage under
    the lean binding with `LeanLocalJudge`. ``requests``, ``votes`` and ``cancelled`` list
    the model stage's first, then the local stage's.
    """

    def __init__(self, model: JudgeStage, lean: JudgeStage,
                 gate: Callable[[ReviewMemberV1], bool]) -> None:
        """Bind the model stage, the local stage and the member gate; no I/O."""
        self._model, self._lean, self._gate = model, lean, gate
        self.requests: tuple[tuple[PositionKey, ...], ...] = ()
        self.votes: tuple[tuple[PositionKey, int], ...] = ()
        self.cancelled: tuple[tuple[int, tuple[PositionKey, ...]], ...] = ()

    def review(self, members: Sequence[ReviewMemberV1], round: RoundNo,
               ) -> tuple[JudgedMemberV1, ...]:
        """Review one round's members, each by the stage its gate selects.

        Returns:
            One outcome per member, in canonical order, as `JudgeStage.review`.
        """
        selected = [self._gate(member) for member in members]
        chosen = [member for member, gated in zip(members, selected, strict=True) if gated]
        rest = [member for member, gated in zip(members, selected, strict=True) if not gated]
        judged: dict[PositionKey, JudgedMemberV1] = {}
        requests: list[tuple[PositionKey, ...]] = []
        votes: list[tuple[PositionKey, int]] = []
        cancelled: list[tuple[int, tuple[PositionKey, ...]]] = []
        for stage, part in ((self._model, chosen), (self._lean, rest)):
            if not part:
                continue
            judged.update((item.position_key, item) for item in stage.review(part, round))
            requests.extend(stage.requests)
            votes.extend(stage.votes)
            cancelled.extend(stage.cancelled)
        self.requests, self.votes = tuple(requests), tuple(votes)
        self.cancelled = tuple(cancelled)
        ordered = sorted(members, key=lambda member: member.input.order)
        return tuple(judged[member.position_key] for member in ordered)
