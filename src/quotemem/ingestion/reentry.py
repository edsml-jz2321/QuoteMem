"""Re-entry of pending positions: pinned selection, wake key, actions and touches.

Only unresolved positions re-enter a later owner: a pending candidate whose
reason code is re-judged ``yes``. Settled positions are never re-entered
or re-extracted, and positions left ``explicit`` stay pending until expiry, forget or purge.

After its barrier an owner selects at most 64 such candidates committed by
earlier owners, in ascending ``(reentry_count, created_commit_seq,
position_key)``. Every selected candidate is touched in the owner's round-1
rejection transaction, whatever its action: ``reentry_count`` + 1 and the
wake key of this owner's rebuild as ``last_wake_digest``. ``last_input_digest``
changes only for a member that is reviewed, so it stays the input of its latest
review. With r new re-judged positions per owner (r < 64), every pending
position is selected within ``ceil(pending / (64 - r))`` owners.

Re-entry is read by relevance: a pending position is re-judged when its wake key
changes, not whenever an entry of its judge input changes. `wake_digest` binds
W, what the rebuilt member's verdict turns on: its quotes, support groups and
shown messages as ``render_masked`` segment maps; its anchors and the speakers
of its cited turns; its judged slots' candidates with their names and shown
facts; the names and facts of the entities it is bound to; the tombstones
nominated for it; for the three relation codes, what it was issued; and its
siblings from the owner's own units. W never holds re-entered siblings, the
shown facts of the user or of a participant, or entities named only as anchors
of what the member was issued (the accepted limit). The actions, in order
(`reentry_action`):

1. an ``open`` slot without candidates: ``unreviewed``;
2. ``review_unavailable``: ``review``;
3. ``target_changed``: ``borrow`` its accepted output when its input is
   unchanged and that output exists, else ``review``;
4. ``correction_target_unresolved`` selected ``SETTLE_AFTER`` times or more: as 3, and
   resolution settles it;
5. any other code: ``quiet`` while its wake key is unchanged, else ``review``.

A quiet member makes no call and reuses no verdict. A re-entered member still
pending after round 1 is rebuilt at the round-2 pin when the owner committed in
round 1 and either its code is ``identity_unresolved`` or ``relation_unclear``
or it has an ``open`` or ``nominated`` slot; it gets its one round-2 review only
when its wake key differs between the two rebuilds, and that review rewrites
both digests.

The wake, action and touch functions operate on plain records. ReentryStage
reads candidates under the owner's fixed-pin guard; accepted-output lookup and
touch writes remain with the judge and the Writer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

from pydantic import ValidationError

from quotemem.core.add_read_view import AddReadView, EntityRecord
from quotemem.core.entities import USER, CandidateStatus, PositionKey, QuoteSpanV1, ReasonCode
from quotemem.core.errors import QuoteMemError
from quotemem.core.outcome_table import OUTCOME_TABLE
from quotemem.core.query import ScopeV1
from quotemem.core.readability import MaskedText, Readability
from quotemem.core.statement import MAX_REENTRY
from quotemem.ingestion.nomination import ContentBinder, MemberV1, NominationV1, RevisesRouteV1
from quotemem.ingestion.sources import Attribution, LabelView, attribute, message_key, source_runs
from quotemem.store.domain import CandidateRowV2
from quotemem.store.indexed import MAX_PAGE_SIZE, CandidateRows, IndexedReader

WAKE_PURPOSE: Final = "reentry-wake-v1"
"""The binding purpose of the wake key: ``bind_content("reentry-wake-v1", W)``."""

REBUILD_CODES: Final[frozenset[ReasonCode]] = frozenset({
    "identity_unresolved", "relation_unclear",
})
"""Round-1 codes after which a re-entered member is rebuilt at the round-2 pin
whatever its slots: an ``ambiguous`` re-pick or an open slot without
candidates, and the later verdict gates."""

SETTLE_AFTER: Final = 1
"""Selections after which a ``correction_target_unresolved`` position stops waiting.

Its first re-entry reviews it again, so a target that arrives in the next owner (a
reversed arrival) resolves it as before. From its second it is never quiet: it
reuses its accepted output (else is reviewed) and resolution settles it. A
correction whose earlier assertion was never stored therefore publishes at the second
later owner instead of staying pending for ever."""

IDENTITY_SETTLE_AFTER: Final = 2
"""Selections after which a re-entered ``identity_unresolved`` position expires under the
``user-role-v1`` self-identity rule (the ``agent`` Add mode).

Its first two re-entries are rebuilt as before, so an identity that a later turn supplies
still resolves it; at its third selection it expires (``expired``,
``evidence_invalidated``: the closed disposition table has no identity-specific terminal code).
An open slot without candidates never resolves at a later owner (``unreviewed``), so
without this rule a long-lived memory keeps such a position pending, and re-entering it,
for ever. The benchmark modes keep it pending."""

TARGET_CODES: Final[frozenset[ReasonCode]] = frozenset({
    "relation_unclear", "correction_target_unresolved", "operation_incomplete",
})
"""Codes whose wake key also holds what the member was issued."""

ReentryAction = Literal["unreviewed", "review", "borrow", "quiet"]
"""What a rebuilt re-entered member costs, in the order re-entry decides it.

- ``unreviewed``: an open slot has no candidate; the member stays
  ``identity_unresolved`` without a call.
- ``review``: it is judged.
- ``borrow``: a ``target_changed`` member whose input is unchanged reuses its
  accepted output without a call.
- ``quiet``: its wake key is unchanged; it makes no call, reuses no verdict and
  keeps its disposition.
"""


@dataclass(frozen=True, slots=True)
class PendingCandidateV1:
    """The ``candidate`` fields re-entry reads.

    Attributes:
        position_key (PositionKey): ``(owner_run_id, unit_no, position)``.
        status (CandidateStatus): The row's status.
        reason_code (ReasonCode): Its latest disposition.
        reentry_count (int): Re-entry selections so far.
        created_commit_seq (int): The transaction that created the row.
        commit_seq (int): The last transaction that changed the row.
        last_input_digest (str | None): The member input digest of its latest
            review; None before any review.
        last_wake_digest (str | None): The wake key of its latest rebuild; None
            until its first re-entry, as no owner writes one at creation.
    """

    position_key: PositionKey
    status: CandidateStatus
    reason_code: ReasonCode
    reentry_count: int
    created_commit_seq: int
    commit_seq: int
    last_input_digest: str | None
    last_wake_digest: str | None

    @property
    def settles(self) -> bool:
        """Whether its wait is over: ``correction_target_unresolved`` selected ``SETTLE_AFTER``
        times.

        The row alone decides re-entry's action (rule 4). Resolution settles only a member
        that is itself a correction: a duplicate that inherited the code from an
        unresolved correction resolves on its own verdict
        (``review_round.resolution_member``).
        """
        return (self.reason_code == "correction_target_unresolved"
                and self.reentry_count >= SETTLE_AFTER)

    @property
    def identity_expires(self) -> bool:
        """Whether its identity wait is over: ``identity_unresolved`` selected
        ``IDENTITY_SETTLE_AFTER`` times (``user-role-v1`` only; the caller checks the rule)."""
        return (self.reason_code == "identity_unresolved"
                and self.reentry_count >= IDENTITY_SETTLE_AFTER)

    @property
    def reenters(self) -> bool:
        """Whether a later owner re-enters it: pending and re-judged ``yes``."""
        return self.status == "pending" and OUTCOME_TABLE[self.reason_code].rejudge == "yes"


@dataclass(frozen=True, slots=True)
class ReentryTouchV1:
    """The candidate fields a selection writes in the owner's rejection transaction.

    Attributes:
        position_key (PositionKey): The selected position.
        reentry_count (int): The new count, one more than at selection.
        last_wake_digest (str): The wake key of the owner's latest rebuild of it.
        last_input_digest (str | None): The member input digest of the owner's
            review; None when the owner made no review, which leaves the row's
            value unchanged (quiet, borrowed and unreviewed members).
    """

    position_key: PositionKey
    reentry_count: int
    last_wake_digest: str
    last_input_digest: str | None

    def rewrite(self, *, wake_digest: str, reviewed_input: str) -> ReentryTouchV1:
        """Return the touch after the member's round-2 review, which rewrites both digests.

        Args:
            wake_digest (str): The wake key of its latest rebuild: the
                round-2 rebuild's, else the round-1 rebuild's, as W is never
                computed over a round-2 pick.
            reviewed_input (str): The member input digest the round-2 review judged.

        Returns:
            ReentryTouchV1: The same position and count with both digests replaced,
                written in the round-2 rejection transaction.
        """
        return replace(self, last_wake_digest=wake_digest, last_input_digest=reviewed_input)

    def apply(self, candidate: PendingCandidateV1) -> PendingCandidateV1:
        """Return a candidate row as this touch leaves it.

        Args:
            candidate (PendingCandidateV1): The touched position's row.

        Returns:
            PendingCandidateV1: The row with the touch's count and wake key, and its
                ``last_input_digest`` replaced only when the touch carries one.

        Raises:
            ValueError: If the row is another position's.
        """
        if candidate.position_key != self.position_key:
            raise ValueError("A touch applies to its own position's row")
        reviewed = self.last_input_digest
        return replace(candidate, reentry_count=self.reentry_count,
                       last_wake_digest=self.last_wake_digest,
                       last_input_digest=candidate.last_input_digest if reviewed is None
                       else reviewed)


def select_reentry(
    candidates: Iterable[PendingCandidateV1], *, owner_run_id: str, limit: int = MAX_REENTRY,
) -> tuple[PendingCandidateV1, ...]:
    """Select the pending positions one owner re-enters.

    Args:
        candidates (Iterable[PendingCandidateV1]): The scope's candidate rows at
            the owner's barrier, in any order.
        owner_run_id (str): The selecting owner; its own rows (such as the
            unresolved primaries its settlement wrote) are not re-entered.
        limit (int): At most this many; ``reentry_limit`` is 64.

    Returns:
        tuple[PendingCandidateV1, ...]: Rows committed by earlier owners that are
            pending and re-judged ``yes``, ascending by ``(reentry_count,
            created_commit_seq, position_key)``, at most ``limit``.

    Raises:
        ValueError: If ``limit`` is negative or a position appears twice.
    """
    if limit < 0:
        raise ValueError("A re-entry limit is 0 or more")
    rows = list(candidates)
    if len({row.position_key for row in rows}) != len(rows):
        raise ValueError("A position has one candidate row")
    eligible = [row for row in rows if row.reenters and row.position_key[0] != owner_run_id]
    eligible.sort(key=lambda row: (row.reentry_count, row.created_commit_seq, row.position_key))
    return tuple(eligible[:limit])


# The wake key (W) --------------------------------------------------------------------------


def wake_digest(
    nomination: NominationV1, view: AddReadView, *, bind: ContentBinder, owner_run_id: str,
    reason_code: ReasonCode, attributions: Sequence[Attribution],
) -> str:
    """Return the wake key of a rebuilt re-entered member: ``bind("reentry-wake-v1", W)``.

    W is computed over the member as rebuilt from its ``ProposalV2`` at the pin,
    never over a round-2 pick or a target resolution asked round 2 to issue.
    Whatever the code, W holds:

    - (a) each quote's ``render_masked`` segments, each support group's kept
      spans, and each cited and context message's whole-message segments with
      its rendered length (segment maps only, never rendered text);
    - (b) every slot's re-resolution (binding, entity id, ``mint_key``,
      ``participant_key``) and each quote's turn speaker and ``speaker_conflict``;
    - (c) each ``open`` or ``nominated`` slot's mention, ``catalog_complete`` and
      ordered candidates, each with its naming revision ids and the revision ids
      of its shown facts;
    - (d) each slot bound to an existing entity: its naming revision ids and the
      revision ids of its shown facts;
    - (e) the ids of the tombstones nominated for it;
    - (f) for ``relation_unclear``, ``correction_target_unresolved`` and
      ``operation_incomplete`` only: the issued heads and retired versions as
      ``(kind, memory id, revision id)``, the history entries shown beside them,
      the ``revises`` resolution, ``entity_partition_complete`` and
      ``external_only_allowed``;
    - (g) the sibling targets drawn from this owner's own units.

    W never holds the shown facts of the user or of a participant, a sibling
    re-entered from an earlier owner, or the entry of an entity the input names
    only as an anchor of an issued head, sibling or tombstone. Information that
    reaches the judge only through those never wakes the member (the accepted
    limit). Keyed by the workspace binder, the digest is content-free.

    Args:
        nomination (NominationV1): The member's nomination at the round's pin.
        view (AddReadView): That pin; messages render under its scope and clock.
        bind (ContentBinder): The workspace's keyed content binder.
        owner_run_id (str): The owner rebuilding the member; its siblings from
            this owner's units enter W, re-entered ones do not.
        reason_code (ReasonCode): The candidate's code read at selection; both
            rebuilds of one owner use it (W (f)).
        attributions (Sequence[Attribution]): For each of the member's quotes, in
            quote order, the attribution of the source run it lies in.

    Returns:
        str: The wake key, comparable with ``last_wake_digest``.

    Raises:
        ValueError: If the member is the owner's own position, its nomination
            issued targets that resolution asked round 2 to issue, or
            ``attributions`` does not give one attribution per quote.
        QuoteMemError: ``binding_mismatch`` when a quote lies in no cited message.
    """
    member = nomination.member
    if member.position_key[0] == owner_run_id:
        raise ValueError("Only a position re-entered from an earlier owner has a wake key")
    if any(isinstance(route, RevisesRouteV1) and route.round2_issue
           for route in nomination.routes):
        raise ValueError("A wake key is never computed over targets round 2 was asked to issue")
    if len(attributions) != len(member.quotes):
        raise ValueError("A wake key takes one attribution per member quote")
    return bind(WAKE_PURPOSE, _wake(nomination, view, owner_run_id, reason_code, attributions))


def _wake(nomination: NominationV1, view: AddReadView, owner_run_id: str,
          reason_code: ReasonCode, attributions: Sequence[Attribution]) -> dict[str, object]:
    member = nomination.member
    read = Readability(view.scope_kind, view.scope_id, view.now, view.evidence)
    cited = {part.object_id: source.message.parts for source in nomination.sources
             if source.cited for part in source.message.parts}
    quotes: list[object] = []
    for span in member.quotes:
        parts = cited.get(span.evidence_id)
        if parts is None:
            raise QuoteMemError("binding_mismatch", "A member quote lies in no cited message")
        quotes.append([*_span(span), _segments(read.render_masked(
            parts, start=span.byte_start, end=span.byte_end,
        ))])
    messages: list[object] = []
    for source in nomination.sources:
        masked = read.render_masked(source.message.parts)
        messages.append([source.cited, [part.object_id for part in source.message.parts],
                         _segments(masked), len(masked.text)])

    def names(entity: EntityRecord | None) -> list[str]:
        return [] if entity is None else sorted({row.naming_revision_id for row in entity.names})

    def facts(entity: EntityRecord | None) -> list[str]:
        if entity is None or entity.entity_id == USER or entity.participant_key is not None:
            return []
        return [fact.head.revision_id for fact in nomination.entity_facts[entity.entity_id]]

    return {
        "quotes": quotes,
        "groups": [[_span(member.quotes[index]) for index in group] for group in member.groups],
        "messages": messages,
        "anchors": [[slot.slot, slot.binding, slot.entity_id, slot.mint_key, slot.participant_key]
                    for slot in member.slots],
        "speakers": [[item.speaker, item.speaker_conflict] for item in attributions],
        "judged": [[item.slot.slot, item.slot.mention, item.catalog_complete, [
            ["entity" if candidate.entity is not None else "new_entity", candidate.key,
             names(candidate.entity), facts(candidate.entity)]
            for candidate in item.candidates
        ]] for item in nomination.slots if item.slot.judged],
        "bound": [[item.slot.slot, item.entity.entity_id, names(item.entity), facts(item.entity)]
                  for item in nomination.slots if item.entity is not None],
        "tombstones": [item.record.object_id for item in nomination.tombstones],
        "targets": _targets(nomination) if reason_code in TARGET_CODES else None,
        "siblings": [[target.object_id, target.statement_digest, list(target.sigma),
                      list(target.anchors)]
                     for target in nomination.targets if target.sibling is not None
                     and target.sibling.position_key[0] == owner_run_id],
    }


def _targets(nomination: NominationV1) -> dict[str, object]:
    """W (f): the committed targets, history, ``revises`` resolution and two flags."""
    flags = nomination.flags
    return {
        "issued": [[target.kind, target.object_id, target.revision_id]
                   for target in nomination.targets if target.sibling is None],
        "history": [[item.head_memory_id, item.entry.version.memory_id,
                     item.entry.version.revision_id, item.entry.ended_by]
                    for item in nomination.history],
        "revises": [[reference.memory_id, reference.revision_id, reference.issued_memory_id,
                     reference.issued_revision_id]
                    for route in nomination.routes if isinstance(route, RevisesRouteV1)
                    for reference in route.references],
        "entity_partition_complete": flags.entity_partition_complete,
        "external_only_allowed": flags.external_only_allowed,
    }


def _span(span: QuoteSpanV1) -> list[object]:
    return [span.evidence_id, span.byte_start, span.byte_end]


def _segments(masked: MaskedText) -> list[object]:
    return [[segment.evidence_id, segment.msg_byte_start, segment.msg_byte_end,
             segment.rendered_char_start, segment.rendered_char_end]
            for segment in masked.segments]


# Actions and touches ---------------------------------------------------------------------------


def reentry_action(
    candidate: PendingCandidateV1, *, reviewable: bool, wake_digest: str,
    member_input_digest: str | None, accepted_output: bool,
) -> ReentryAction:
    """Decide what a rebuilt re-entered member costs, in order.

    Args:
        candidate (PendingCandidateV1): The selected row, as read at selection.
        reviewable (bool): False when an open slot has no candidate
            (`NominationV1.reviewable`).
        wake_digest (str): The member's wake key at this owner's rebuild.
        member_input_digest (str | None): Its rebuilt member input's digest;
            None exactly when it is not reviewable.
        accepted_output (bool): Whether an accepted, unscrubbed output exists in
            the scope for ``(binding_digest, member_input_digest)``.

    Returns:
        ReentryAction: ``unreviewed`` without candidates for an open slot;
            ``review`` for ``review_unavailable``; for ``target_changed`` and a
            correction that `settles` ``borrow`` when its input equals
            ``last_input_digest`` and the output exists, else ``review``; for any
            other code ``quiet`` when its wake key equals ``last_wake_digest``,
            else ``review``.

    Raises:
        ValueError: If ``member_input_digest`` is given for an unreviewable
            member or missing for a reviewable one.
    """
    if reviewable == (member_input_digest is None):
        raise ValueError("A reviewable member has a member input digest, an unreviewable one none")
    if not reviewable:
        return "unreviewed"
    if candidate.reason_code == "review_unavailable":
        return "review"
    if candidate.reason_code == "target_changed" or candidate.settles:
        unchanged = member_input_digest == candidate.last_input_digest
        return "borrow" if unchanged and accepted_output else "review"
    return "quiet" if wake_digest == candidate.last_wake_digest else "review"


def touch(
    candidate: PendingCandidateV1, action: ReentryAction, *, wake_digest: str,
    reviewed_input: str | None = None,
) -> ReentryTouchV1:
    """Return the round-1 touch of a selected candidate, whatever its action.

    Args:
        candidate (PendingCandidateV1): The selected row, as read at selection.
        action (ReentryAction): Its action at this owner.
        wake_digest (str): Its wake key at this owner's round-1 rebuild.
        reviewed_input (str | None): For ``review``, the member input digest the
            review judged; None when the review never went out. Every other
            action leaves ``last_input_digest`` unchanged.

    Returns:
        ReentryTouchV1: ``reentry_count`` + 1, ``last_wake_digest`` and, after a
            review, ``last_input_digest``.

    Raises:
        ValueError: If a quiet, borrowed or unreviewed member is given a
            reviewed input.
    """
    if action != "review" and reviewed_input is not None:
        raise ValueError("Only a reviewed member writes last_input_digest")
    return ReentryTouchV1(candidate.position_key, candidate.reentry_count + 1, wake_digest,
                          reviewed_input)


def fence_failure_code(candidate: PendingCandidateV1, *, borrowed: bool) -> ReasonCode:
    """Return the code of a re-entered member whose component failed the fence.

    Args:
        candidate (PendingCandidateV1): The selected row.
        borrowed (bool): Whether this round borrowed its recorded verdict.

    Returns:
        ReasonCode: ``target_changed_repeated`` (explicit) when a borrowed
            ``target_changed`` member fails again, else ``target_changed``.
    """
    if borrowed and candidate.reason_code == "target_changed":
        return "target_changed_repeated"
    return "target_changed"


# Round 2 -----------------------------------------------------------------------------


def needs_round2_rebuild(member: MemberV1, round1_code: ReasonCode, *,
                         owner_committed: bool) -> bool:
    """Tell whether a re-entered member is rebuilt at the round-2 pin.

    Quiet members are rebuilt like reviewed ones; a rebuilt member is reviewed
    again only when its wake key changed (`round2_review`).

    Args:
        member (MemberV1): The member as rebuilt in round 1; its slot bindings
            count.
        round1_code (ReasonCode): Its disposition after round 1.
        owner_committed (bool): Whether this owner committed a component in
            round 1; only its commits can wake the member.

    Returns:
        bool: True when the owner committed and the member is still pending,
            with either ``identity_unresolved`` or ``relation_unclear`` or an ``open`` or
            ``nominated`` slot, which a code can mask.
    """
    if not owner_committed or "pending" not in OUTCOME_TABLE[round1_code].statuses:
        return False
    return round1_code in REBUILD_CODES or any(slot.judged for slot in member.slots)


def round2_review(*, round1_wake: str, round2_wake: str) -> bool:
    """Tell whether a rebuilt re-entered member gets its one round-2 review.

    Args:
        round1_wake (str): Its wake key at the round-1 rebuild.
        round2_wake (str): Its wake key rebuilt at the round-2 pin, with the
            code read at selection.

    Returns:
        bool: True when the wake key changed; otherwise it stays pending.
    """
    return round2_wake != round1_wake


@dataclass(frozen=True, slots=True)
class ReentryCensusV1:
    """Content-free re-entry action counts for the completion census.

    ``pending_selectable`` counts the pre-cap population; the four actions
    count selected, rebuilt members. Expiries are separate dispositions.
    """

    pending_selectable: int
    unreviewed: int = 0
    review: int = 0
    borrow: int = 0
    quiet: int = 0

    @classmethod
    def of(cls, pending_selectable: int, actions: Iterable[ReentryAction]) -> ReentryCensusV1:
        """Count the actions once; refuse an impossible pre-cap population."""
        counts = Counter(actions)
        if pending_selectable < sum(counts.values()):
            raise ValueError("Re-entry actions exceed the selectable population")
        return cls(pending_selectable, counts["unreviewed"], counts["review"],
                   counts["borrow"], counts["quiet"])


@dataclass(frozen=True, slots=True)
class ReentrySelectionV1:
    """Selected durable rows (including Proposal references) and the pre-cap population."""

    candidates: tuple[CandidateRowV2, ...]
    pending_selectable: int


def pending_candidate(row: CandidateRowV2) -> PendingCandidateV1:
    """Read the selection/touch fields from the authoritative candidate row."""
    return PendingCandidateV1(
        (row.owner_run_id, row.unit_no, row.position), row.status, row.reason_code,
        row.reentry_count, row.created_commit_seq, row.commit_seq,
        row.last_input_digest, row.last_wake_digest,
    )


class ReentryStage:
    """Pinned durable selection; rebuilding consumes the Proposal producer.

    Construction does no I/O. ``check`` must validate the owner's pinned read
    across every page, including changes to candidate rows, not just privacy
    epochs. This class never writes or requests a provider.
    """

    def __init__(self, reader: IndexedReader, scope: ScopeV1, *,
                 check: Callable[[], None]) -> None:
        """Bind the owner's exact scope, indexed reader and fixed-pin check."""
        self._reader, self._scope, self._check = reader, scope.model_copy(), check

    def select(
        self, owner_run_id: str, *, rows: Mapping[PositionKey, CandidateRowV2] | None = None,
        exclude: Collection[PositionKey] = (),
    ) -> ReentrySelectionV1:
        """Select at most 64 earlier-owner pending rows in fair rotation order.

        Reads the scope's pending rows under the caller's fixed pin (`rows`)
        unless the caller passes the ones it read. The returned rows retain their
        Proposal references for the rebuild; the count is taken before applying
        the cap. A row in ``exclude`` (one a rebased owner routes past round 1)
        is neither selected nor counted. A read or guard failure aborts selection.
        """
        found = self.rows() if rows is None else rows
        candidates = tuple(pending_candidate(row) for key, row in found.items()
                           if key not in exclude)
        selected = select_reentry(candidates, owner_run_id=owner_run_id)
        return ReentrySelectionV1(
            tuple(found[candidate.position_key] for candidate in selected),
            sum(candidate.reenters and candidate.position_key[0] != owner_run_id
                for candidate in candidates),
        )

    def rows(self) -> dict[PositionKey, CandidateRowV2]:
        """Read every pending row of the scope by position, in bounded pages."""
        query = CandidateRows(scope_kind=self._scope.scope_kind, scope_id=self._scope.scope_id,
                              status="pending")
        rows: dict[PositionKey, CandidateRowV2] = {}
        after: str | None = None
        while True:
            self._check()
            page = self._reader.page(query, after=after, limit=MAX_PAGE_SIZE)
            self._check()
            for raw in page:
                try:
                    row = CandidateRowV2.model_validate(raw)
                except ValidationError:
                    raise QuoteMemError("parity_mismatch",
                                        "Invalid stored pending candidate") from None
                if (row.scope_kind, row.scope_id) != (
                    self._scope.scope_kind, self._scope.scope_id,
                ) or row.status != "pending":
                    raise QuoteMemError("parity_mismatch", "Candidate page differs from its scope")
                candidate = pending_candidate(row)
                if candidate.position_key in rows or (
                    after is not None and row.object_id <= after
                ):
                    raise QuoteMemError("parity_mismatch",
                                        "Candidate page repeats or reorders a row")
                after = row.object_id
                rows[candidate.position_key] = row
            if len(page) < MAX_PAGE_SIZE:
                break
        self._check()
        return rows


def cited_attributions(
    member: MemberV1, view: AddReadView, labels: LabelView,
) -> tuple[Attribution, ...]:
    """Derive cited-turn attribution once for both wake and judge input.

    Returns one attribution per quote in quote order, using frozen labels of
    the whole cited message. A quote that cannot be assigned to exactly one
    source run fails closed; no metadata-only fallback substitutes an identity.
    """
    result: list[Attribution] = []
    for quote in member.quotes:
        part = view.evidence(quote.evidence_id)
        message = None if part is None else view.message(message_key(part))
        if message is None:
            raise QuoteMemError("binding_mismatch", "Cited message is unavailable")
        matching = [run for run in source_runs(message.parts, labels)
                    if run.byte_start <= quote.byte_start and quote.byte_end <= run.byte_end]
        if len(matching) != 1:
            raise QuoteMemError("binding_mismatch", "Cited span crosses source attribution")
        result.append(attribute(matching[0], view.scope_id))
    return tuple(result)
