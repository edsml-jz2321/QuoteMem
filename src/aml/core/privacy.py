"""The privacy-erase Evidence placeholder, candidate expiry and the D.7.3 compensation choice.

``choose_compensation`` is the one implementation of D.7.3 steps 1-4: privacy
erase uses its ``erase`` mode and forget its ``forget_dependent`` mode; S5b's I11
operations apply the choice. ``trimmed_support`` is forget's step 4a for a dependent
left unreadable (C22). ``expire_candidate`` is D.7.3 step 5 for a pending
candidate citing erased Evidence (A.3.4).

The placeholder function constructs an ephemeral typed post-image. It must never
be registered or serialized as an erasure_plan body; persisted plans are the
content-free contracts in privacy_persistence. The sole Writer owns input
admission, exact row digests, fence tickets, physical RowOp budgets, persistence,
reader cleanup, and the final zero-residual proof.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal, Protocol, Self

from pydantic import model_validator

from aml.core.entities import MemorySnapshotV2, QuoteSpanV1, RevisionReason
from aml.core.errors import AMLError
from aml.core.identity import ULID, ContentBindingDigest
from aml.core.readability import ByteRange, Readability
from aml.core.statement import token_seq, token_spans
from aml.store.domain import ERASED_CONTENT_HASH, CandidateRowV2, EvidenceRowV1
from aml.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

RowDigest = Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
_RowBinder = Callable[[object], ContentBindingDigest]


class PrivacyEvidencePostImageV1(EvidenceRowV1):
    """The fixed erased placeholder (D.7, A.3.5); structural identity survives."""

    object_id: Annotated[
        ULID,
        RefSpec("evidence", ref_mode="snapshot_locator", scope_rule="same_scope", gc_root=False),
    ]

    @model_validator(mode="after")
    def check_erased(self) -> Self:
        """The inherited fixed-placeholder invariant applies only to erased rows."""
        if self.state != "erased":
            raise ValueError("Privacy Evidence post-image must be an erased tombstone")
        return self


class EvidenceTombstoneActionV1(StrictModel):
    """Clear one entire Evidence row without retaining original content in the plan.

    Effect replay is permitted only from the exact frozen before or intended
    after state.
    """

    expected_before_digest: RowDigest
    intended_after_digest: RowDigest
    action: Literal["evidence_tombstone"]
    post_image: PrivacyEvidencePostImageV1


def action_key(action: EvidenceTombstoneActionV1) -> tuple[str, str]:
    """Order complete row effects deterministically, independently of discovery order."""
    return "evidence", action.post_image.object_id


def tombstone_evidence(row: EvidenceRowV1, bind_row: _RowBinder) -> EvidenceTombstoneActionV1:
    """Build the sole permitted erased post-image while retaining structural metadata.

    The placeholder keeps no text, speaker, participant or descriptor, and erasure
    clears the row's entity bindings and withheld ranges (A.3.5): privacy erase
    inserts no withheld span (D.7, DM-10).
    """
    row = EvidenceRowV1.model_validate(row.model_dump(mode="json"))
    post = PrivacyEvidencePostImageV1.model_validate(
        {
            **row.model_dump(mode="json"),
            "state": "erased",
            "route_state": "pending",
            "text": "",
            "role": None,
            "source_descriptor": None,
            "speaker_entity": None,
            "source_participant": None,
            "byte_start": 0,
            "byte_end": 0,
            "content_hash": ERASED_CONTENT_HASH,
            "embedding": [0.0] * len(row.embedding),
            "entity_bindings": None,
            "withheld_spans": [],
        }
    )
    return EvidenceTombstoneActionV1(
        action="evidence_tombstone",
        expected_before_digest=bind_row(row.model_dump(mode="json")),
        intended_after_digest=bind_row(post.model_dump(mode="json")),
        post_image=post,
    )


def expire_candidate(row: Mapping[str, Any], *, commit_seq: int, now: int) -> dict[str, Any]:
    """The post-image of a pending candidate whose cited Evidence is erased (D.7.3 #5, A.3.4).

    The candidate becomes ``expired`` with reason ``evidence_invalidated``,
    releases its hold (D.6) and records the expiring transaction's time and
    ``commit_seq``; its position, proposal, citations and re-entry state stay.

    Args:
        row: The stored pending candidate.
        commit_seq: The expiring transaction's ``commit_seq``.
        now: The expiring transaction's time in epoch milliseconds.

    Returns:
        The candidate's strict post-image.

    Raises:
        AMLError: ``immutable_field`` for a candidate that is not pending.
    """
    if row["status"] != "pending":
        raise AMLError("immutable_field", "Only a pending candidate expires (A.3.4)")
    return CandidateRowV2.model_validate({
        **row, "status": "expired", "reason_code": "evidence_invalidated",
        "hold_evidence": False, "terminal_at": now, "commit_seq": commit_seq,
    }).model_dump(mode="json")


CompensationMode = Literal["erase", "forget_dependent"]
"""Privacy erase may end at the erased head; forget's dependent mode never erases (D.7.3 #4)."""
CompensationAction = Literal["keep", "requote", "erase_head", "leave_unreadable"]
"""D.7.3's outcome for one affected Item: nothing, a requote of one version from its
surviving groups, the content-free head, or (``forget_dependent`` only) no change."""
_DISQUALIFYING_REASONS = frozenset({"correction", "retraction", "rewording"})
"""A version older than a correction, retraction or rewording of its Item never becomes
current again."""


class CompensatedVersion(Protocol):
    """One revision of an affected Item, as D.7.3 reads it (A.3.2; ``RevisionRowV2``)."""

    @property
    def object_id(self) -> str: ...

    @property
    def version(self) -> int: ...

    @property
    def reason_code(self) -> RevisionReason: ...

    @property
    def snapshot(self) -> MemorySnapshotV2: ...


@dataclass(frozen=True, slots=True)
class CompensationChoice:
    """What D.7.3 does to one affected Item; content-free.

    Attributes:
        action: ``keep`` (unchanged or corrected head with surviving support),
            ``requote`` (``revision_id``
            with the indexes of its surviving ``groups``), ``erase_head`` or
            ``leave_unreadable``.
        revision_id: The requoted version: the head or an earlier version.
        groups: The requoted version's support groups that still pass P-GROUP,
            as indexes into its ``support_groups``, ascending.
    """

    action: CompensationAction
    revision_id: str | None = None
    groups: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """A requote names its version and at least one surviving group; nothing else does."""
        if (self.action == "requote") != (self.revision_id is not None) or (
            (self.action == "requote") != bool(self.groups)
        ) or list(self.groups) != sorted(set(self.groups)):
            raise ValueError("A requote alone names its version and its surviving groups")


def surviving_groups(snapshot: MemorySnapshotV2, read: Readability) -> tuple[int, ...]:
    """Return the indexes of the support groups that pass P-GROUP; none for an erased version."""
    if snapshot.erased:
        return ()
    return tuple(
        index for index, group in enumerate(snapshot.support_groups)
        if read.p_group(snapshot.quote_spans, group)
    )


def choose_compensation(
    versions: Sequence[CompensatedVersion],
    read: Readability,
    *,
    tombstoned: Callable[[str, str], bool],
    mode: CompensationMode,
    corrected: Callable[[str], bool] | None = None,
) -> CompensationChoice:
    """Choose D.7.3 steps 1-4 for one affected Item; the Writer's I11 operations apply it.

    Steps, in order: (1) the head passes P-VER and either lost no group or is a
    historical correction target: keep; (2) the head passes P-VER but lost groups:
    requote the head from its surviving groups; (3) else the newest earlier
    non-erased version not targeted by a historical correction whose
    ``(entity_id, statement_digest)`` is not tombstoned, which passes P-VER and
    is not older than any ``correction`` or ``retraction`` revision of the Item:
    requote it from its surviving groups; (4) else ``erase_head`` in ``erase``
    mode and ``leave_unreadable`` in ``forget_dependent`` mode. An erased head
    has nothing left to compensate. Readability is the single D.2
    implementation (D.0), evaluated at the erase's state and clock.

    Args:
        versions: Every revision of the Item, the head being the highest version.
        read: The scope's readability after the erase or withholding.
        tombstoned: Whether a ``(entity_id, statement_digest)`` pair has a tombstone.
        mode: ``erase`` (privacy erase) or ``forget_dependent`` (forget, D.3 P2).
        corrected: Whether a historical correction targets a revision. Such a head
            keeps its identity while a complete group survives; an earlier corrected
            version is never requoted. None leaves this check to the caller.

    Returns:
        The content-free choice.

    Raises:
        ValueError: If no version is given or two share a version number.
    """
    ordered = sorted(versions, key=lambda value: value.version)
    if not ordered or len({value.version for value in ordered}) != len(ordered):
        raise ValueError("Compensation needs an Item's distinct versions")
    head = ordered[-1]
    if head.snapshot.erased:
        return CompensationChoice("keep")
    groups = surviving_groups(head.snapshot, read)
    if groups:
        if len(groups) == len(head.snapshot.support_groups) or (
                corrected is not None and corrected(head.object_id)):
            return CompensationChoice("keep")
        return CompensationChoice("requote", head.object_id, groups)
    barrier = max(
        (value.version for value in ordered if value.reason_code in _DISQUALIFYING_REASONS),
        default=0,
    )
    for earlier in reversed(ordered[:-1]):
        if earlier.version < barrier:
            break
        snapshot = earlier.snapshot
        if snapshot.erased or tombstoned(snapshot.entity_id, snapshot.statement_digest) or (
                corrected is not None and corrected(earlier.object_id)):
            continue
        groups = surviving_groups(snapshot, read)
        if groups:
            return CompensationChoice("requote", earlier.object_id, groups)
    return CompensationChoice("erase_head" if mode == "erase" else "leave_unreadable")


_TRIM_LIGHT: Final = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "they", "them", "their", "theirs", "themselves",
    "it", "its", "itself", "who", "whom", "whose", "which", "user", "users",
    "am", "is", "are", "was", "were", "be", "been", "being", "s", "m", "re", "ve", "ll", "d",
    "t", "do", "does", "did", "have", "has", "had", "will", "would", "shall", "should",
    "can", "could", "might", "must",
    "of", "to", "in", "on", "at", "for", "with", "from", "by", "about", "as", "into", "onto",
    "and", "or", "named", "called", "name", "nicknamed",
})
"""Tokens the trim ignores: articles, pronouns, "user", forms of be, do and have, modals
but "may" (a month too), contraction tails, common prepositions, "and", "or" and the naming
verbs. Negation, numbers, dates, time words, names and every other word count."""

_LEAD: Final = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "i", "me", "my", "mine",
    "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "it", "its", "user", "users",
})
"""Light tokens a ``word_runs`` run may begin with before its first word: determiners and
pronouns ("I run", "My brother"), never a conjunction, preposition or verb."""

_CONTRACTED: Final = frozenset({
    "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn", "won",
    "wouldn", "can", "couldn", "shouldn", "mustn", "mightn", "needn", "ain",
})
"""Tokens that a following ``t`` (from ``n't``) makes a negated auxiliary."""


def _stem(token: str) -> str:
    """A crude English stem, so that "loves" and "love" or "planned" and "plan" compare."""
    word = token
    if len(word) >= 5 and word.endswith(("ies", "ied")):
        word = word[:-3] + "y"
    elif len(word) >= 6 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) >= 5 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) >= 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) >= 5 and word.endswith("e"):
        word = word[:-1]
    return word


_FORMS: Final = {"another": "other"}
"""Tokens compared as another word: "another" is "an other" ("another cat" carries the
"other" of "Blair's other cat")."""


def _token_words(tokens: Sequence[str]) -> list[str | None]:
    """Each token's compared word: its crude stem, "not" for both tokens of ``n't``, or None
    for a token of ``_TRIM_LIGHT``."""
    words: list[str | None] = []
    for index, token in enumerate(tokens):
        if (token == "t" and index and tokens[index - 1] in _CONTRACTED) or (
                token in _CONTRACTED and tokens[index + 1:index + 2] == ["t"]):
            words.append("not")
        elif token in _TRIM_LIGHT:
            words.append(None)
        else:
            words.append(_stem(_FORMS.get(token, token)))
    return words


def support_words(text: str) -> frozenset[str]:
    """The words a dependent's trim compares (D.7.3 #4a): stems of ``token_seq``'s tokens.

    ``n't`` reads as "not"; the tokens of ``_TRIM_LIGHT`` are dropped. A word in another
    script is one token, a Han, Kana or Hangul character each, compared exactly as English
    words are.

    Args:
        text: A statement, quote, keyword, time expression or forgotten statement.

    Returns:
        The stems that may carry a fact.
    """
    return frozenset(word for word in _token_words(token_seq(text)) if word is not None)


def word_runs(text: str, words: frozenset[str]) -> tuple[tuple[int, int], ...]:
    """The character ranges of an ASCII ``text`` that carry only ``words`` (D.3 P1 #6 (e)).

    A run is a maximal sequence of tokens each of which is a ``_TRIM_LIGHT`` token or has
    its compared word in ``words``, cut to begin at its first token whose word is in
    ``words``, extended back over the determiners and pronouns just before it, and to
    end at its last such token; a run with no such token is none. In "I volunteer
    at the animal shelter on weekends and I run in the mornings" the words of "The user
    runs in the mornings." give one run, "I run in the mornings".

    Args:
        text: An ASCII quote, so that character offsets are byte offsets.
        words: ``support_words`` of the statement the runs may carry.

    Returns:
        The runs as ``(start, end)`` character offsets, in text order.
    """
    tokens = token_spans(text)
    compared = _token_words([token for token, _, _ in tokens])
    runs: list[tuple[int, int]] = []
    first: int | None = None
    last: int | None = None
    for index, word in enumerate([*compared, ""]):
        if index < len(tokens) and (word is None or word in words):
            if word is not None:
                first = index if first is None else first
                last = index
            continue
        if first is not None and last is not None:
            lead = first
            while lead and compared[lead - 1] is None and tokens[lead - 1][0] in _LEAD:
                lead -= 1
            runs.append((tokens[lead][1], tokens[last][2]))
        first = last = None
    return tuple(runs)


def _pieces(span: QuoteSpanV1, withheld: Sequence[ByteRange]) -> tuple[
        tuple[QuoteSpanV1, ...], tuple[QuoteSpanV1, ...]]:
    """The parts of ``span`` outside and inside its Evidence's withheld ranges."""
    readable = [(span.byte_start, span.byte_end)]
    hidden: list[tuple[int, int]] = []
    for held in withheld:
        start, end = max(span.byte_start, held.byte_start), min(span.byte_end, held.byte_end)
        if start < end:
            hidden.append((start, end))
        readable = [part for low, high in readable for part in (
            ((low, high),) if held.byte_end <= low or high <= held.byte_start
            else tuple(piece for piece in ((low, held.byte_start), (held.byte_end, high))
                       if piece[0] < piece[1])
        )]

    def spans(parts: Sequence[tuple[int, int]]) -> tuple[QuoteSpanV1, ...]:
        return tuple(QuoteSpanV1(evidence_id=span.evidence_id, byte_start=start,
                                 byte_end=end) for start, end in sorted(set(parts)))

    return spans(readable), spans(hidden)


@dataclass(frozen=True, slots=True)
class TrimmedSupport:
    """What D.7.3 #4a requotes a forget's dependent on (C22, FP16, NM18).

    Attributes:
        groups: Each qualifying group's readable parts as spans, sorted and unique.
        keywords: The head's keywords, in order, less every keyword carrying a word of
            the forgotten content or of the kept groups' withheld bytes.
    """

    groups: tuple[tuple[QuoteSpanV1, ...], ...]
    keywords: tuple[str, ...]


def trimmed_support(
    snapshot: MemorySnapshotV2,
    read: Readability,
    *,
    quoted: Callable[[QuoteSpanV1], str | None],
    forgotten: frozenset[str],
    speakers: Callable[[QuoteSpanV1], frozenset[str]] | None = None,
) -> TrimmedSupport | None:
    """The groups a forget's dependent keeps on the readable parts of its quotes (D.7.3 #4a).

    A support group is an AND of quotes, so one withheld byte hides it whole. An
    extracted group can carry a quote its statement does not need: "My birthday is
    March 3" beside "I'm planning a party for my sister" for "The user is planning a
    party for the user's sister." (C22, NM18). One quote can also state two facts, "I
    volunteer at the animal shelter on weekends and I run in the mornings": forgetting
    the volunteering withholds the sentence but the part a clause keeper keeps (D.3 P1
    #6 (e), FP16). When D.7.3 steps 1-3 leave such a dependent unreadable, a group still
    supports it on its readable parts alone when the words prove it:

    * the head is not erased and has no sensitivity class;
    * the group loses a byte to withholding only (its Evidence passes P-SRC) and keeps a
      readable part: a whole readable quote, or a run of the statement's own words and
      light words (``word_runs``) in the readable bytes of a partly withheld ASCII quote.
      Such a run carries no word the statement does not, so no forgotten word, even one
      another keeper left readable beside it ("Acme Corp" in "I work at Acme Corp as an
      engineer" after "The user works at Acme Corp." is forgotten: "an engineer" is kept);
    * every word of the statement and of its time expression is a word of those parts or
      of the name of a part's speaker (its readable provenance, "Blair" for Blair's
      "we got another cat named Juno too"): what the memory says rests on what stays
      readable;
    * no word of the statement or of its time expression is a word of the group's
      withheld bytes or one of ``forgotten``: what the memory says shows no word of the
      forgotten content, even one its readable parts repeat. A keyword is the
      extractor's search term, not what the memory says: one carrying such a word is
      dropped from the requoted head instead ("birthday" on "The user is planning a
      party for the user's sister." after "The user's birthday is March 3." is
      forgotten, NM18), so the requoted memory shows none.

    The test is on words and errs toward withholding: a synonym ("bachelor" for
    "single") is a word the readable parts must hold, and a statement in English over
    quotes in another language keeps nothing. No withheld byte becomes readable; the
    forget's stating spans are unchanged.

    Args:
        snapshot: The dependent's head.
        read: The scope's readability after the forget's withholding.
        quoted: The text a quote span or a piece of it quotes, or None when its Evidence
            cannot show it.
        forgotten: ``support_words`` of the forget's stating spans and of what its
            tombstones' statements say.
        speakers: ``support_words`` of the name of the speaker a quote span is attributed
            to; None when no quote has one.

    Returns:
        The qualifying groups and the keywords the requoted head keeps; None when no
        group qualifies.
    """
    if snapshot.erased or snapshot.sensitivity_classes:
        return None
    stated = support_words(snapshot.statement)
    if snapshot.time_expr.kind != "none":
        stated |= support_words(snapshot.time_expr.text)
    if not stated or stated & forgotten:
        return None

    def words(parts: Sequence[QuoteSpanV1]) -> frozenset[str] | None:
        found: set[str] = set()
        for part in parts:
            text = quoted(part)
            if text is None:
                return None
            found |= support_words(text)
        return frozenset(found)

    trimmed: set[tuple[QuoteSpanV1, ...]] = set()
    withheld_words: set[str] = set()
    for group in snapshot.support_groups:
        kept: list[QuoteSpanV1] = []
        hidden: list[QuoteSpanV1] = []
        usable = True
        for index in group:
            span = snapshot.quote_spans[index]
            if read.p_span(span):
                kept.append(span)
                continue
            evidence = read.evidence(span.evidence_id)
            if evidence is None or not read.p_src(evidence):
                usable = False
                break
            readable, withheld = _pieces(span, evidence.withheld_spans)
            hidden.extend(withheld)
            for piece in readable:
                text = quoted(piece)
                if text is None:
                    usable = False
                    break
                if not text.isascii():
                    continue  # A piece in another script is never cut.
                kept.extend(QuoteSpanV1(evidence_id=piece.evidence_id,
                                        byte_start=piece.byte_start + start,
                                        byte_end=piece.byte_start + end)
                            for start, end in word_runs(text, stated))
        if not usable or not hidden or not kept:
            continue
        kept_words, hidden_words = words(kept), words(hidden)
        if kept_words is not None and speakers is not None:
            kept_words = kept_words.union(*(speakers(part) for part in kept))
        if (kept_words is not None and hidden_words is not None
                and stated <= kept_words and not stated & hidden_words):
            trimmed.add(tuple(sorted(set(kept), key=lambda part: (
                part.evidence_id, part.byte_start, part.byte_end))))
            withheld_words |= hidden_words
    if not trimmed:
        return None
    dropped = forgotten | withheld_words
    return TrimmedSupport(
        groups=tuple(sorted(trimmed, key=lambda parts: [
            (part.evidence_id, part.byte_start, part.byte_end) for part in parts])),
        keywords=tuple(keyword for keyword in snapshot.keywords
                       if not support_words(keyword) & dropped),
    )
