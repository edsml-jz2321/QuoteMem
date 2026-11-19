"""Which judge verdicts need confirmation, and which verdict stands after the votes (C.8 #9).

One sampled verdict can change memory on its own: a ``restates`` tombstone verdict denies its
member (C.9.1 G1), and an ``update``, ``correct`` or ``retract`` ends a current memory the
member was issued. A verdict that does either is *consequential*. Its member input is judged
again, alone, by fresh votes until two of at most ``CONFIRMATION_VOTES`` verdicts share a
consequence, and the earliest of those verdicts that supports the member stands (else the
earliest): a verdict that only fails to support the member also ends nothing, and must not
reject a member another verdict of that consequence supports. Without a majority memory
stays as it is: a denial stands, and a member whose verdicts disagree on which heads it
ends, or how, is resolved by its first verdict with each head it ends read ``unclear``, so
it settles ``relation_unclear`` (C.9.1 G4) and ends nothing.

The standing verdict's relation to a same-Add sibling is one more sample the votes did not
compare: ``same`` or ``supplement`` folds the member and its sibling into one write, and
a change of the sibling supersedes it (C.9.3). Such a fold stands only when every other
verdict sharing the standing verdict's consequence folds that sibling too; otherwise the
sibling relation is read ``independent`` and both write (a regression case, open-quit-job: the
quit event's first verdict ``supplement``-ed "Avery no longer works at Northstar
Analytics." and its vote related it ``independent``; the fold left the event to carry the
ended employment, a wrong target, and the employment's end was never written).

These functions are pure: the judge stage and the prompt lab apply the same rule.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Final, Literal

from aml.core.judge_output import CatalogEntryV8, JudgeMemoryEntryV8, JudgeSiblingEntryV8, JudgeV8
from aml.core.judge_records import CONFIRMATION_VOTES

ENDING_RELATIONS: Final = frozenset({"update", "correct", "retract"})
"""Relations that end the issued head they name (C.9.4: the head is no longer current)."""
DENIED: Final = "denied"
"""The consequence of a verdict that denies its member (C.9.1 G1)."""
SIBLING_FOLDS: Final = frozenset({"same", "supplement", "update", "correct", "retract"})
"""Relations that fold a same-Add sibling into the member's write or supersede it (C.9.3)."""

type Endings = frozenset[tuple[str, str]]
type Consequence = Literal["denied"] | Endings
"""What a verdict alone would do to memory: deny the member, or end these (head, relation)
pairs; the empty set is a verdict that does neither."""


def issued_heads(targets: Sequence[str], catalog: Mapping[str, CatalogEntryV8]) -> frozenset[str]:
    """The handles of a member's issued current memories: the heads a verdict may end.

    Args:
        targets (Sequence[str]): The member's issued target handles.
        catalog (Mapping[str, CatalogEntryV8]): A catalog holding them.

    Returns:
        frozenset[str]: The targets whose entry is a current ``memory``; siblings and
            earlier versions are not heads.
    """
    return frozenset(handle for handle in targets
                     if isinstance(catalog.get(handle), JudgeMemoryEntryV8))


def denies(verdict: JudgeV8) -> bool:
    """Whether a verdict denies its member as a restated forgotten fact (C.9.1 G1).

    A ``restates`` tombstone verdict with support ``contradicted`` rejects the member
    as unsupported instead, so it denies nothing.
    """
    return verdict.support != "contradicted" and any(
        entry.verdict == "restates" for entry in verdict.tombstones)


def consequence(verdict: JudgeV8, heads: Collection[str]) -> Consequence:
    """Return what a verdict alone would do to memory (C.8 #9).

    Args:
        verdict (JudgeV8): An accepted verdict.
        heads (Collection[str]): The member's ``issued_heads``.

    Returns:
        Consequence: ``DENIED`` when it denies the member; else, for a supported member,
            the (head, relation) pairs it ends; else the empty set, since an unsupported
            member writes nothing.
    """
    if denies(verdict):
        return DENIED
    if verdict.support != "supported":
        return frozenset()
    return frozenset((item.target, item.relation) for item in verdict.relations
                     if item.relation in ENDING_RELATIONS and item.target in heads)


def consequential(found: Consequence) -> bool:
    """Whether a consequence needs confirmation: a denial, or at least one ended head."""
    return found == DENIED or bool(found)


def standing(found: Sequence[Consequence], *, closed: bool,
             supported: Sequence[bool] = ()) -> int | Literal["unclear"] | None:
    """Decide which ballot stands once a consequential first verdict has had its votes.

    Args:
        found (Sequence[Consequence]): The consequences of the ballots so far, the first
            verdict first, then each vote in order.
        closed (bool): Whether no further vote can come (a vote ended unavailable).
        supported (Sequence[bool]): Per ballot, whether its support is ``supported``; a
            ballot that only fails to support the member also ends nothing, so among the
            ballots sharing the majority consequence the earliest supporting one stands.

    Returns:
        int | Literal["unclear"] | None: The index of the ballot that stands: of the ballots
            sharing a consequence with another, the earliest that supports the member, else
            the earliest; None while another vote may still decide; once no majority can
            form, 0 when the first verdict denies (the denial stands) and ``"unclear"``
            otherwise (the member ends nothing, ``unclear_endings``).

    Examples:
        ```python
        standing([DENIED, frozenset()], closed=False)                # None
        standing([DENIED, frozenset(), frozenset()], closed=False)  # 1
        standing([frozenset({("M1", "update")}), frozenset({("M1", "correct")}),
                  frozenset({("M1", "correct")})], closed=False)    # 1
        ```
    """
    for index, value in enumerate(found):
        if value in found[index + 1:]:
            shared = [other for other, item in enumerate(found) if item == value]
            return next((other for other in shared
                         if other < len(supported) and supported[other]), index)
    if not closed and len(found) < CONFIRMATION_VOTES:
        return None
    return 0 if found and found[0] == DENIED else "unclear"


def unclear_endings(verdict: JudgeV8, heads: Collection[str]) -> JudgeV8:
    """The verdict a member is resolved by when no majority confirmed its ended heads.

    Args:
        verdict (JudgeV8): The member's first verdict, which ends some issued head.
        heads (Collection[str]): The member's issued current memories.

    Returns:
        JudgeV8: The verdict with each relation that ends an issued head read ``unclear``
            ("you cannot decide"), so resolution settles ``relation_unclear`` (C.9.1 G4)
            and no head ends; every other field is the verdict's own.
    """
    return verdict.model_copy(update={"relations": [
        item.model_copy(update={"relation": "unclear"})
        if item.relation in ENDING_RELATIONS and item.target in heads else item
        for item in verdict.relations
    ]})


def issued_siblings(targets: Sequence[str],
                    catalog: Mapping[str, CatalogEntryV8]) -> frozenset[str]:
    """The handles of a member's issued same-Add siblings (C.9.3).

    Args:
        targets (Sequence[str]): The member's issued target handles.
        catalog (Mapping[str, CatalogEntryV8]): A catalog holding them.

    Returns:
        frozenset[str]: The targets whose entry is a ``sibling``.
    """
    return frozenset(handle for handle in targets
                     if isinstance(catalog.get(handle), JudgeSiblingEntryV8))


def agreed_folds(ballots: Sequence[JudgeV8], found: Sequence[Consequence], index: int,
                 siblings: Collection[str]) -> JudgeV8:
    """The standing ballot with each sibling fold its agreeing ballots do not share read
    ``independent`` (C.8 #9, C.9.3).

    Args:
        ballots (Sequence[JudgeV8]): The member's verdicts, the first verdict first, then
            each vote in order.
        found (Sequence[Consequence]): Their consequences, in the same order.
        index (int): The ballot that stands (`standing`).
        siblings (Collection[str]): The member's ``issued_siblings``.

    Returns:
        JudgeV8: The standing ballot itself when every other ballot sharing its
            consequence folds each sibling it folds (``SIBLING_FOLDS``); else a copy with
            each such disputed sibling relation read ``independent``.
    """
    standing_ballot = ballots[index]
    fellows = [ballot for other, ballot in enumerate(ballots)
               if other != index and found[other] == found[index]]
    disputed = {
        item.target for item in standing_ballot.relations
        if item.target in siblings and item.relation in SIBLING_FOLDS and any(
            next((relation.relation for relation in fellow.relations
                  if relation.target == item.target), None) not in SIBLING_FOLDS
            for fellow in fellows)
    }
    if not disputed:
        return standing_ballot
    return standing_ballot.model_copy(update={"relations": [
        item.model_copy(update={"relation": "independent"}) if item.target in disputed else item
        for item in standing_ballot.relations
    ]})
