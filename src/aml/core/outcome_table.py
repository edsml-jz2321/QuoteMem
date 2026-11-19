"""The single disposition table of Add results (C.10; INV H06).

Every position an Add owner admits or refuses ends with exactly one of 36
reason codes. A code fixes the public category the ``/add`` receipt counts,
the candidate status its row may hold (or that a refused position has no row),
whether later owners re-enter it, and the rule that produces it. Completion,
the ``/add`` counts, add-items and re-entry all read this table; none keeps a
private copy of it, and no precedence list ranks one code over another: a
position's last round gives its only disposition.
"""

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Literal, NamedTuple, get_args

from aml.core.entities import CandidateStatus, ReasonCode, Rejudge
from aml.core.errors import AMLError

OutcomeCategory = Literal[
    "added", "corrected", "supplemented", "unchanged", "rejected", "unresolved",
]
"""The six public counts, in the ``/add`` receipt order (E.1.2)."""

PositionKey = tuple[str, int, int]  # S1b-align
"""One admitted position: ``(owner_run_id, unit_no, position)`` (A.3.4)."""

OUTCOME_CODES: tuple[ReasonCode, ...] = get_args(ReasonCode)
"""The 36 reason codes of C.10, in table order: the closed ``candidate.reason_code`` enum (A.2)."""
OUTCOME_CATEGORIES: tuple[OutcomeCategory, ...] = get_args(OutcomeCategory)
FORMAL_CATEGORIES: frozenset[OutcomeCategory] = frozenset({"added", "corrected", "supplemented"})
"""Categories whose position wrote a revision: a formal effect."""


class OutcomeRule(NamedTuple):
    """One C.10 row for one code.

    Attributes:
        category: The public count the position adds to.
        statuses: The candidate statuses its row may hold; empty when the code
            never has a row.
        refused: True when the position may have no candidate row because
            admission refused it (B.7.3, B.7.5).
        rejudge: Whether later owners re-enter the position (C.3).
        producers: The contract rules that produce the code.
    """

    category: OutcomeCategory
    statuses: tuple[CandidateStatus, ...]
    refused: bool
    rejudge: Rejudge
    producers: tuple[str, ...]

    @property
    def formal(self) -> bool:
        """Whether the code records a formal effect: a revision was written."""
        return self.category in FORMAL_CATEGORIES


def _rows(
    codes: tuple[ReasonCode, ...], category: OutcomeCategory,
    statuses: tuple[CandidateStatus, ...], rejudge: Rejudge, *producers: str,
    refused: bool = False,
) -> dict[ReasonCode, OutcomeRule]:
    return {code: OutcomeRule(category, statuses, refused, rejudge, producers) for code in codes}


_PUBLISHED: tuple[CandidateStatus, ...] = ("published",)
_UNCHANGED: tuple[CandidateStatus, ...] = ("unchanged",)
_REJECTED: tuple[CandidateStatus, ...] = ("rejected",)
_PENDING: tuple[CandidateStatus, ...] = ("pending",)

_ROWS: dict[ReasonCode, OutcomeRule] = {
    **_rows(("created", "created_with_contradiction", "external_correction"),
            "added", _PUBLISHED, "no", "C.9.4", "C.9.5"),
    **_rows(("updated", "corrected", "retracted"), "corrected", _PUBLISHED, "no", "C.9.4"),
    **_rows(("history_corrected",), "corrected", _PUBLISHED, "no", "C.9.5a"),
    **_rows(("supplemented", "merged_support"), "supplemented", _PUBLISHED, "no", "C.9.4"),
    **_rows(("redundant",), "unchanged", _UNCHANGED, "no", "C.9.4"),
    **_rows(("restatement_satisfied",), "unchanged", _UNCHANGED, "no", "B.8 #2"),
    **_rows(("duplicate_in_add",), "unchanged", _UNCHANGED, "no", "B.8 #3", "C.9.3"),
    **_rows(("superseded_in_add",), "unchanged", _UNCHANGED, "no", "C.9.3"),
    **_rows(("restates_retired_version",), "unchanged", _UNCHANGED, "no", "C.9.2"),
    **_rows(("denied_by_forget",), "rejected", _REJECTED, "no",
            "B.8 #1", "C.9.1 G1", "C.11.1 #6"),
    **_rows(("unsupported_by_source",), "rejected", _REJECTED, "no", "C.9.1 G1", "C.9.1 G2"),
    **_rows(("not_atomic", "source_fidelity_incomplete"), "rejected", _REJECTED, "no",
            "C.9.1 G2"),
    **_rows(("unsupported_sensitive_inference",), "rejected", _REJECTED, "no",
            "C.9.1 G2s", "B.7.3", refused=True),
    **_rows(("secret_rejected", "extraction_item_invalid"), "rejected", (), "no", "B.7.3",
            refused=True),
    **_rows(("evidence_invalidated",), "rejected", ("rejected", "expired"), "no",
            "C.11.1", "C.3", "D.3", "D.7"),
    **_rows(("semantic_evidence_insufficient", "source_fidelity_uncertain"), "unresolved",
            _PENDING, "yes", "C.9.1 G2"),
    **_rows(("identity_unresolved",), "unresolved", _PENDING, "yes", "B.7.3", "C.9.1 G3"),
    **_rows(("relation_unclear",), "unresolved", _PENDING, "yes", "C.9.1 G4-G6", "C.9.3"),
    **_rows(("correction_target_unresolved",), "unresolved", _PENDING, "yes", "C.9.5"),
    **_rows(("operation_incomplete",), "unresolved", _PENDING, "yes", "C.9.3"),
    **_rows(("target_changed",), "unresolved", _PENDING, "yes", "C.11"),
    **_rows(("review_unavailable",), "unresolved", _PENDING, "yes", "C.8 #6"),
    **_rows(("review_unavailable_repeated",), "unresolved", _PENDING, "explicit",
            "C.3", "C.8 #6"),
    **_rows(("target_changed_repeated",), "unresolved", _PENDING, "explicit", "C.3"),
    **_rows(("older_than_target",), "unresolved", _PENDING, "explicit", "C.9.2"),
    **_rows(("review_capacity_exceeded",), "unresolved", _PENDING, "explicit", "C.6.3"),
    **_rows(("support_budget_exceeded",), "unresolved", _PENDING, "explicit",
            "B.8", "C.9.3", "C.12"),
    **_rows(("extraction_capacity_exceeded",), "unresolved", (), "explicit", "B.7.3",
            refused=True),
}

OUTCOME_TABLE: Mapping[str, OutcomeRule] = MappingProxyType({
    code: _ROWS[code] for code in OUTCOME_CODES
})
"""Every C.10 row, keyed on ``get_args(ReasonCode)`` in its order; a code without a row fails
at import, and the typed rows admit no code outside ``ReasonCode``."""


def outcome_rule(code: str) -> OutcomeRule:
    """Return the C.10 row of one stored or produced reason code.

    Args:
        code (str): A reason code, typically read back from a candidate row or
            a frozen completion record.

    Returns:
        OutcomeRule: The code's category, statuses, refusal, rejudge and producers.

    Raises:
        AMLError: ``binding_mismatch`` when the code has no row in the table.
    """
    if code not in OUTCOME_TABLE:
        raise AMLError("binding_mismatch", "Add disposition has no outcome table row")
    return OUTCOME_TABLE[code]


class OutcomeItem(NamedTuple):
    """One position's final disposition as one receipt counts it.

    Attributes:
        position_key: The position, counted once with all its targets (INV:D19).
        code: Its single final reason code.
        status: Its candidate row's status, or None when it has no row.
        revision_id: The revision it wrote, or None.
    """

    position_key: PositionKey
    code: str
    status: CandidateStatus | None
    revision_id: str | None


def tally(items: Iterable[OutcomeItem]) -> dict[OutcomeCategory, int]:
    """Count one receipt's positions into the six public categories (C.10, E.1.2).

    Each position counts once, in its code's category, and the counts sum to
    the number of positions. Attribution (which request a position counts
    toward) is the caller's; this only checks and counts what it is given.

    Args:
        items (Iterable[OutcomeItem]): The final dispositions attributed to one
            request.

    Returns:
        dict[OutcomeCategory, int]: All six categories, in receipt order
            (``added, corrected, supplemented, unchanged, rejected,
            unresolved``), zero when absent.

    Raises:
        AMLError: ``binding_mismatch`` for two final dispositions of one
            position, an unknown code, a revision present on a non-formal code
            or missing on a formal one, a code that is never refused without a
            row, or a row whose status the code does not allow.
    """
    counts: dict[OutcomeCategory, int] = dict.fromkeys(OUTCOME_CATEGORIES, 0)
    seen: set[PositionKey] = set()
    for item in items:
        if item.position_key in seen:
            raise AMLError("binding_mismatch", "A position has two final dispositions")
        seen.add(item.position_key)
        rule = outcome_rule(item.code)
        if rule.formal != (item.revision_id is not None):
            raise AMLError("binding_mismatch", "A formal code needs exactly its own revision")
        if item.status is None and not rule.refused:
            raise AMLError("binding_mismatch", "A code that is never refused has no row")
        if item.status is not None and item.status not in rule.statuses:
            raise AMLError("binding_mismatch", "A row's status does not match its code")
        counts[rule.category] += 1
    return counts
