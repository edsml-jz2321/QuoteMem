"""The single disposition table of Add results.

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

from quotemem.core.entities import CandidateStatus, ReasonCode, Rejudge
from quotemem.core.errors import QuoteMemError

OutcomeCategory = Literal[
    "added", "corrected", "supplemented", "unchanged", "rejected", "unresolved",
]
"""The six public counts, in the ``/add`` receipt order."""

PositionKey = tuple[str, int, int]
"""One admitted position: ``(owner_run_id, unit_no, position)``."""

OUTCOME_CODES: tuple[ReasonCode, ...] = get_args(ReasonCode)
"""The 36 reason codes, in table order: the closed ``candidate.reason_code`` enum."""
OUTCOME_CATEGORIES: tuple[OutcomeCategory, ...] = get_args(OutcomeCategory)
FORMAL_CATEGORIES: frozenset[OutcomeCategory] = frozenset({"added", "corrected", "supplemented"})
"""Categories whose position wrote a revision: a formal effect."""


class OutcomeRule(NamedTuple):
    """One row for one code.

    Attributes:
        category: The public count the position adds to.
        statuses: The candidate statuses its row may hold; empty when the code
            never has a row.
        refused: True when the position may have no candidate row because
            admission refused it.
        rejudge: Whether later owners re-enter the position.
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
            "added", _PUBLISHED, "no", "write forms", "targetless correction"),
    **_rows(("updated", "corrected", "retracted"), "corrected", _PUBLISHED, "no", "write forms"),
    **_rows(("history_corrected",), "corrected", _PUBLISHED, "no", "history correction"),
    **_rows(("supplemented", "merged_support"), "supplemented", _PUBLISHED, "no", "write forms"),
    **_rows(("redundant",), "unchanged", _UNCHANGED, "no", "write forms"),
    **_rows(("restatement_satisfied",), "unchanged", _UNCHANGED, "no", "pre-review settlement"),
    **_rows(("duplicate_in_add",), "unchanged", _UNCHANGED, "no",
            "pre-review settlement", "in-Add relations"),
    **_rows(("superseded_in_add",), "unchanged", _UNCHANGED, "no", "in-Add relations"),
    **_rows(("restates_retired_version",), "unchanged", _UNCHANGED, "no",
            "direction check"),
    **_rows(("denied_by_forget",), "rejected", _REJECTED, "no",
            "pre-review settlement", "forget gate", "commit fence"),
    **_rows(("unsupported_by_source",), "rejected", _REJECTED, "no",
            "forget gate", "source support gate"),
    **_rows(("not_atomic", "source_fidelity_incomplete"), "rejected", _REJECTED, "no",
            "source support gate"),
    **_rows(("unsupported_sensitive_inference",), "rejected", _REJECTED, "no",
            "sensitive inference gate", "extraction admission", refused=True),
    **_rows(("secret_rejected", "extraction_item_invalid"), "rejected", (), "no",
            "extraction admission", refused=True),
    **_rows(("evidence_invalidated",), "rejected", ("rejected", "expired"), "no",
            "commit fence", "re-entry", "forget", "privacy erase"),
    **_rows(("semantic_evidence_insufficient", "source_fidelity_uncertain"), "unresolved",
            _PENDING, "yes", "source support gate"),
    **_rows(("identity_unresolved",), "unresolved", _PENDING, "yes",
            "extraction admission", "identity gate"),
    **_rows(("relation_unclear",), "unresolved", _PENDING, "yes",
            "relation gates", "in-Add relations"),
    **_rows(("correction_target_unresolved",), "unresolved", _PENDING, "yes",
            "targetless correction"),
    **_rows(("operation_incomplete",), "unresolved", _PENDING, "yes", "in-Add relations"),
    **_rows(("target_changed",), "unresolved", _PENDING, "yes", "commit fence"),
    **_rows(("review_unavailable",), "unresolved", _PENDING, "yes", "judge requests"),
    **_rows(("review_unavailable_repeated",), "unresolved", _PENDING, "explicit",
            "re-entry", "judge requests"),
    **_rows(("target_changed_repeated",), "unresolved", _PENDING, "explicit", "re-entry"),
    **_rows(("older_than_target",), "unresolved", _PENDING, "explicit",
            "direction check"),
    **_rows(("review_capacity_exceeded",), "unresolved", _PENDING, "explicit",
            "review capacity"),
    **_rows(("support_budget_exceeded",), "unresolved", _PENDING, "explicit",
            "pre-review settlement", "in-Add relations", "writer transactions"),
    **_rows(("extraction_capacity_exceeded",), "unresolved", (), "explicit",
            "extraction admission", refused=True),
}

OUTCOME_TABLE: Mapping[str, OutcomeRule] = MappingProxyType({
    code: _ROWS[code] for code in OUTCOME_CODES
})
"""Every row, keyed on ``get_args(ReasonCode)`` in its order; a code without a row fails
at import, and the typed rows admit no code outside ``ReasonCode``."""


def outcome_rule(code: str) -> OutcomeRule:
    """Return the row of one stored or produced reason code.

    Args:
        code (str): A reason code, typically read back from a candidate row or
            a frozen completion record.

    Returns:
        OutcomeRule: The code's category, statuses, refusal, rejudge and producers.

    Raises:
        QuoteMemError: ``binding_mismatch`` when the code has no row in the table.
    """
    if code not in OUTCOME_TABLE:
        raise QuoteMemError("binding_mismatch", "Add disposition has no outcome table row")
    return OUTCOME_TABLE[code]


class OutcomeItem(NamedTuple):
    """One position's final disposition as one receipt counts it.

    Attributes:
        position_key: The position, counted once with all its targets.
        code: Its single final reason code.
        status: Its candidate row's status, or None when it has no row.
        revision_id: The revision it wrote, or None.
    """

    position_key: PositionKey
    code: str
    status: CandidateStatus | None
    revision_id: str | None


def tally(items: Iterable[OutcomeItem]) -> dict[OutcomeCategory, int]:
    """Count one receipt's positions into the six public categories.

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
        QuoteMemError: ``binding_mismatch`` for two final dispositions of one
            position, an unknown code, a revision present on a non-formal code
            or missing on a formal one, a code that is never refused without a
            row, or a row whose status the code does not allow.
    """
    counts: dict[OutcomeCategory, int] = dict.fromkeys(OUTCOME_CATEGORIES, 0)
    seen: set[PositionKey] = set()
    for item in items:
        if item.position_key in seen:
            raise QuoteMemError("binding_mismatch", "A position has two final dispositions")
        seen.add(item.position_key)
        rule = outcome_rule(item.code)
        if rule.formal != (item.revision_id is not None):
            raise QuoteMemError("binding_mismatch", "A formal code needs exactly its own revision")
        if item.status is None and not rule.refused:
            raise QuoteMemError("binding_mismatch", "A code that is never refused has no row")
        if item.status is not None and item.status not in rule.statuses:
            raise QuoteMemError("binding_mismatch", "A row's status does not match its code")
        counts[rule.category] += 1
    return counts
