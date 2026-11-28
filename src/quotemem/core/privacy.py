"""The privacy-erase Evidence placeholder, candidate expiry and the compensation choice.

``choose_compensation`` is the one implementation of the compensation choice: privacy
erase uses its ``erase`` mode and forget its ``forget_dependent`` mode; the
operations apply the choice. ``trimmed_support`` is forget's further step for a dependent
left unreadable. ``expire_candidate`` is the expiry step for a pending
candidate citing erased Evidence.

The placeholder function constructs an ephemeral typed post-image. It must never
be registered or serialized as an erasure_plan body; persisted plans are the
content-free contracts in privacy_persistence. The sole Writer owns input
admission, exact row digests, fence tickets, physical RowOp budgets, persistence,
reader cleanup, and the final zero-residual proof.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal, Protocol, Self

from pydantic import model_validator

from quotemem.core.entities import MemorySnapshotV2, QuoteSpanV1, RevisionReason
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ULID, ContentBindingDigest
from quotemem.core.readability import ByteRange, Readability
from quotemem.core.statement import own_token, token_contained, token_seq, token_spans
from quotemem.store.domain import ERASED_CONTENT_HASH, CandidateRowV2, EvidenceRowV1
from quotemem.store.metadata import OpaqueIdSpec, RefSpec, StrictModel

RowDigest = Annotated[ContentBindingDigest, OpaqueIdSpec("digest")]
_RowBinder = Callable[[object], ContentBindingDigest]


class PrivacyEvidencePostImageV1(EvidenceRowV1):
    """The fixed erased placeholder; structural identity survives."""

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
    clears the row's entity bindings and withheld ranges: privacy erase
    inserts no withheld span.
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
    """The post-image of a pending candidate whose cited Evidence is erased.

    The candidate becomes ``expired`` with reason ``evidence_invalidated``,
    releases its hold and records the expiring transaction's time and
    ``commit_seq``; its position, proposal, citations and re-entry state stay.

    Args:
        row: The stored pending candidate.
        commit_seq: The expiring transaction's ``commit_seq``.
        now: The expiring transaction's time in epoch milliseconds.

    Returns:
        The candidate's strict post-image.

    Raises:
        QuoteMemError: ``immutable_field`` for a candidate that is not pending.
    """
    if row["status"] != "pending":
        raise QuoteMemError("immutable_field", "Only a pending candidate expires")
    return CandidateRowV2.model_validate({
        **row, "status": "expired", "reason_code": "evidence_invalidated",
        "hold_evidence": False, "terminal_at": now, "commit_seq": commit_seq,
    }).model_dump(mode="json")


CompensationMode = Literal["erase", "forget_dependent"]
"""Privacy erase may end at the erased head; forget's dependent mode never erases."""
CompensationAction = Literal["keep", "requote", "erase_head", "leave_unreadable"]
"""The outcome for one affected memory: nothing, a requote of one version from its
surviving groups, the content-free head, or (``forget_dependent`` only) no change."""
_DISQUALIFYING_REASONS = frozenset({"correction", "retraction", "rewording"})
"""A version older than a correction, retraction or rewording of its memory never becomes
current again."""


class CompensatedVersion(Protocol):
    """One revision of an affected memory, as privacy compensation reads it (``RevisionRowV2``)."""

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
    """What privacy compensation does to one affected memory; content-free.

    Attributes:
        action: ``keep`` (unchanged or corrected head with surviving support),
            ``requote`` (``revision_id``
            with the indexes of its surviving ``groups``), ``erase_head`` or
            ``leave_unreadable``.
        revision_id: The requoted version: the head or an earlier version.
        groups: The requoted version's support groups that still pass ``p_group``,
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
    """Return the indexes of the support groups that pass ``p_group``; none for an erased
    version."""
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
    """Choose the compensation for one affected memory; the Writer's compensation operations apply
    it.

    Steps, in order: (1) the head passes ``p_ver`` and either lost no group or is a
    historical correction target: keep; (2) the head passes ``p_ver`` but lost groups:
    requote the head from its surviving groups; (3) else the newest earlier
    non-erased version not targeted by a historical correction whose
    ``(entity_id, statement_digest)`` is not tombstoned, which passes ``p_ver`` and
    is not older than any ``correction``, ``retraction`` or rewording revision of the memory
    (a forget's rewording too: a privacy requote stating something else than the version
    before it):
    requote it from its surviving groups; (4) else ``erase_head`` in ``erase``
    mode and ``leave_unreadable`` in ``forget_dependent`` mode. An erased head
    has nothing left to compensate. Readability is the single predicate
    implementation, evaluated at the erase's state and clock.

    Args:
        versions: Every revision of the memory, the head being the highest version.
        read: The scope's readability after the erase or withholding.
        tombstoned: Whether a ``(entity_id, statement_digest)`` pair has a tombstone.
        mode: ``erase`` (privacy erase) or ``forget_dependent`` (forget).
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
    # A forget's rewording (a privacy requote stating something else than the version
    # before it) bars every earlier version: each may tell what the forget took out.
    earlier_of = {value.version: value for value in ordered}
    barrier = max(
        (value.version for value in ordered if value.reason_code in _DISQUALIFYING_REASONS
         or (value.reason_code == "privacy_requote"
             and (previous := earlier_of.get(value.version - 1)) is not None
             and previous.snapshot.statement_digest != value.snapshot.statement_digest)),
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
    """The words a dependent's trim compares: stems of ``token_seq``'s tokens.

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
    """The character ranges of an ASCII ``text`` that carry only ``words``.

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


_HAN_LIGHT: Final = frozenset("我你您他她它们的地得了着过吗呢吧啊呀是在也都就还又很")
"""Han characters the clause cut (``clause_words``) reads as light, as ``_TRIM_LIGHT`` for
English: personal pronouns and 们, the structural and aspect particles, 是 and 在, and the
adverbs 也 都 就 还 又 很."""

_CLAUSE_CUT: Final = re.compile(
    r"\s*(?:[，,；;、。！？!?：:]"
    r"|(?:而且|并且|但是|可是|不过|同时|所以|因为|还有|和|还|也|但)(?=我|你|您|他|她|它|用户)"
    r"|\b(?:and|but|so)\b)\s*",
    re.IGNORECASE,
)
"""Where a quote in any script is cut into clauses (``clause_runs``): clause marks, and a
connective (和, 并且, 而且, 还, 也, 但是 …) only where a subject follows it, so it joins two
clauses, not two nouns; "and", "but" and "so" in a mixed quote."""


def clause_words(text: str) -> frozenset[str]:
    """``support_words`` of a text as the clause cut compares them in any script: "用户" (the
    user) and the Han light characters (``_HAN_LIGHT``) are no words. Of an ASCII text,
    exactly its ``support_words``.

    Args:
        text: A statement, quote or clause.

    Returns:
        The stems and Han characters that may carry a fact.
    """
    return support_words(text.replace("用户", " ")) - _HAN_LIGHT


def _own_runs(text: str) -> list[list[str]]:
    """The runs of adjacent Han, Kana or Hangul characters of a text, "用户" and the Han light
    characters (``_HAN_LIGHT``) cutting them."""
    runs: list[list[str]] = []
    run: list[str] = []
    last = -1
    for token, start, end in token_spans(text.replace("用户", " ")):
        if not own_token(token) or token in _HAN_LIGHT:
            run = _flushed(runs, run)
            continue
        if start != last:
            run = _flushed(runs, run)
        run.append(token)
        last = end
    _flushed(runs, run)
    return runs


def _flushed(runs: list[list[str]], run: list[str]) -> list[str]:
    """Close ``run`` into ``runs`` when it holds a character; a new empty run."""
    if run:
        runs.append(run)
    return []


def fact_terms(text: str) -> frozenset[str]:
    """The terms a forgotten text forbids a kept clause to show, in any script.

    The words of other scripts are its ``clause_words``. A Han, Kana or Hangul run (cut by
    "用户" and the light characters) gives the pairs of its adjacent characters, and a run of
    one character that character: "用户每天早上喝绿茶" forbids 每天, 天早, 早上, 上喝, 喝绿 and
    绿茶, so "我姐姐住在上海" shows none of them though it shares 上; "用户养了猫" forbids 养
    and 猫. Read against ``shown_terms``.

    Args:
        text: A forgotten statement, or a sentence a forget withholds.

    Returns:
        The forbidden terms.
    """
    found = {word for word in clause_words(text) if not own_token(word)}
    for run in _own_runs(text):
        found |= {run[0]} if len(run) == 1 else {a + b for a, b in zip(run, run[1:], strict=False)}
    return frozenset(found)


def shown_terms(text: str) -> frozenset[str]:
    """The terms a text shows, as ``fact_terms`` forbids them: its words of other scripts,
    and of each Han, Kana or Hangul run its adjacent pairs and every character.

    Args:
        text: A statement, keyword, clause or quote kept readable.

    Returns:
        The shown terms.
    """
    found = {word for word in clause_words(text) if not own_token(word)}
    for run in _own_runs(text):
        found |= {*run, *(a + b for a, b in zip(run, run[1:], strict=False))}
    return frozenset(found)


def clause_runs(
    text: str, words: frozenset[str], terms: frozenset[str] | None = None,
) -> tuple[tuple[int, int], ...]:
    """The clauses of a quote in any script that carry only ``words``.

    ``word_runs`` cuts an ASCII quote at words; a Chinese or mixed quote has no spaces
    between words, so it is cut at its clauses instead (``_CLAUSE_CUT``): "我每天早上喝绿茶，
    我姐姐住在柏林" gives "我每天早上喝绿茶" and "我姐姐住在柏林". A clause counts when it
    carries a word (``clause_words``) and every word it carries is one of ``words``; its
    clause mark or connective is never part of it. Given ``terms``, every term of the
    clause (``fact_terms``: its pairs of adjacent Han characters, not only the characters)
    must be one of them too: "我辞职了" carries only characters of "用户在柏林的职位是辞典编辑。"
    but its 辞职 is no pair of it, so it may state another fact and is none.

    Args:
        text: A quote in any script.
        words: ``clause_words`` of the statement the clauses may carry.
        terms: ``shown_terms`` of that statement, or None to compare characters only.

    Returns:
        The clauses as ``(start, end)`` UTF-8 byte offsets, in text order.
    """
    runs: list[tuple[int, int]] = []
    start = 0
    for cut in [*_CLAUSE_CUT.finditer(text), None]:
        end = len(text) if cut is None else cut.start()
        piece = text[start:end]
        found = clause_words(piece)
        if piece.strip() and found and found <= words and (
                terms is None or fact_terms(piece) <= terms):
            lead = len(piece) - len(piece.lstrip())
            begin = len(text[:start + lead].encode("utf-8"))
            runs.append((begin, begin + len(piece.strip().encode("utf-8"))))
        if cut is not None:
            start = cut.end()
    return tuple(runs)


def english(statement: str) -> bool:
    """Whether a statement is written in English: one of its Latin words is an English function
    word (``_TRIM_LIGHT``: "the", "user", "is", "has", "in" …), as in "The user's cat is named
    bob." and "The user lives in 杭州西湖区.". The extraction writes every statement so,
    copying names in their source script; "用户的姐姐住在柏林。" and "用户的猫叫bob。" are not."""
    return any(word.casefold() in _TRIM_LIGHT for word in _LATIN.findall(statement))


def _given_safe(given: str, hidden: Sequence[str]) -> bool:
    """Whether a head's statement (``given``) may hold its capitalized names for a wording
    over a group's readable parts (``trimmed_support``): every other Han character of the
    group's withheld bytes (``hidden``) it shows as written, so the forgotten value is one it
    names in its own script ("李伟" of "The user's younger brother 李伟 attends university in
    Beijing."), and none of its Latin names may be that value in another script ("Li Wei" of
    "The user's younger brother Li Wei ..." over a withheld "我弟弟叫李伟", "Shanghai" over a
    withheld "我在上海住"). No transliteration is read, so it errs toward holding nothing."""
    return all(char in given for text in hidden for char in text
               if own_token(char) and char not in _GIVEN_FREE)


def unshown_names(statement: str, texts: Sequence[str]) -> bool:
    """Whether an English statement shows a capitalized word that none of ``texts`` holds as a
    whole word ("Shiyan" of "The user's daughter goes to Shiyan Primary School." over "我女儿",
    "上三年级"): a name the statement gives in another script than its quotes, perhaps the
    forgotten value transliterated or translated. Function words ("The") are no names.

    Args:
        statement: An English statement.
        texts: The readable text it would rest on.

    Returns:
        True when such a word is found.
    """
    held = {word.casefold() for text in texts for word in _LATIN.findall(text)}
    return any(word[0].isupper() and word.casefold() not in _TRIM_LIGHT
               and word.casefold() not in held for word in _LATIN.findall(statement))


_LATIN: Final = re.compile(r"[0-9A-Za-z]+")
"""An ASCII word of a quote in another script ("bob" of "我的猫叫bob")."""

_FRAME: Final = support_words("attends goes lives studies works")
"""Words of an English predicate that name only its frame, not its value: what a Han
sentence says by a marker or a verb next to the value ("在阿里上班" for "works at 阿里")."""

_NAMING: Final = (
    re.compile(r"(?:An?|The) [A-Za-z'\u2019]+(?: [A-Za-z'\u2019]+){0,3} is (?:also )?"
               r"(?:named|called) (?P<name>.+?)\."),
    re.compile(r"(?:An?|The) [A-Za-z'\u2019]+(?: [A-Za-z'\u2019]+){0,3}['\u2019]s name is "
               r"(?P<name>.+?)\."),
    re.compile(r"The name of (?:an?|the) [A-Za-z'\u2019]+(?: [A-Za-z'\u2019]+){0,3} is "
               r"(?P<name>.+?)\."),
    # Another name of someone named already: the alias alone ("林可欣 is also known as 安安.").
    re.compile(r"(?:(?:Mrs|Mr|Ms|Dr|Prof)\. )?(?:[A-Z][\w'\u2019-]*(?: [A-Z][\w'\u2019-]*){0,2}"
               r"|[\u3400-\u4dbf\u4e00-\u9fff]{1,8}) is (?:also (?:known as|called|named)|known as|"
               r"nicknamed|referred to as|goes by) (?P<name>.+?)\."),
    # A nickname of someone, however the statement names them: the nickname alone ("The
    # person named 林可欣 is also known as 安安.", "Rowan's nickname is Ollie.", "The user's son
    # Rowan is nicknamed Ollie.", "林可欣 has the nickname 安安.").
    re.compile(r".+? (?:is|was) also (?:known as|called|named|referred to as) (?P<name>.+?)\."),
    re.compile(r".+?['\u2019]s (?:nick ?name|pet name) is (?P<name>.+?)\."),
    re.compile(r".+? (?:is|was) (?:also )?nicknamed (?P<name>.+?)\."),
    re.compile(r".+? (?:has|had|goes by) the (?:nick ?name|pet name) (?P<name>.+?)\."),
    re.compile(r".+? (?:goes|went) by (?P<name>.+?)\."),
    # The same written in Chinese ("名叫欧文的人小名叫小欧", "欧文的小名是小欧").
    re.compile(r".+?(?:小名|昵称|外号|乳名|绰号|爱称)(?:叫|是|为)(?P<name>[^，。,.\s]+?)[。.]?"),
)
"""A statement that only names someone or something ("A person is named 小李.", "The user's
cat is named bob.", "林可欣 is also known as 安安.", "Rowan's nickname is Ollie."): its
detail is the name (the alias of a statement giving another name), its subject and "named"
are its frame."""

_HAN_RUN: Final = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def naming_value(statement: str) -> str | None:
    """The name a statement that only names someone or something gives (``_NAMING``): "小李"
    of "A person is named 小李.", "bob" of "The user's cat is named bob."; None for any other.

    Args:
        statement: A statement.

    Returns:
        The name as written, or None.
    """
    for pattern in _NAMING:
        match = pattern.fullmatch(statement.strip())
        if match is not None:
            return match["name"]
    return None


_ANOTHER_NAME: Final = re.compile(
    r"\b(?:also (?:known as|called|named)|nicknamed|known as|goes by|go by|nick ?names?"
    r"|pet names?)\b|小名|昵称|外号|乳名|绰号|爱称", re.IGNORECASE)
"""The words of a statement giving someone another name beside the name."""


def kept_beside(name: str, statement: str) -> bool:
    """Whether ``name`` is the name a statement giving someone another name gives it beside
    ("林可欣" of "林可欣 is also known as 安安."): forgetting that statement forgets the other
    name (``naming_value``), not this one.

    Args:
        name: A name the statement shows.
        statement: A forgotten statement.

    Returns:
        True when the statement gives another name than ``name``.
    """
    if _ANOTHER_NAME.search(statement) is None:
        return False
    other = naming_value(statement)
    return other is not None and other.casefold() != name.casefold()


_LEADS: Final = ("名字叫做", "名字叫", "名字是", "取名为", "取名叫", "取名", "叫做", "名叫",
                 "住在", "叫", "在", "是", "于", "到", "去")
"""The marker just before a value that belongs to it ("叫bob", "在阿里", "住在杭州")."""

_TRAILS: Final = ("上班", "工作", "上学", "读书", "念书", "打工", "实习", "就读", "任职")
"""The frame verb just after a value that belongs to it ("在阿里上班")."""


def foreign_rest(text: str, said: frozenset[str], forgotten: str,
                 elsewhere: Sequence[str] = ()) -> tuple[tuple[int, int], ...]:
    """The clauses of a Chinese or mixed quote left readable when one detail of it is forgotten
    and the statements are in English (across languages).

    An English statement shares no word with a Han quote, so ``clause_runs`` keeps nothing of
    "我的猫叫bob" for "The user has a cat." The forgotten detail is found by its own words
    instead: the names the extraction copies in their source script. Every word of the
    forgotten predicate (``said``) that is no frame word (``_FRAME``) must be an ASCII word of
    the quote or of the forgotten version's other quotes (``elsewhere``: "bob"), and every Han
    run of the forgotten statement (cut by "用户" and the light characters) must occur in one
    of them ("实验小学", "阿里"); otherwise the detail is not located, it may be told by
    translated words anywhere in the quote, and nothing is kept. A statement that only names
    someone or something (``_NAMING``: "A person is named 小李.") states the name alone: its
    subject and "named" are its frame, so "person" need not be a word of "我的两个同事小王、小
    李", and only the name's own words are sought. A quote holding none of
    those words ("我女儿" beside the version's "在实验小学") tells nothing of the detail.
    Each occurrence is withheld with the marker before it (``_LEADS``: "叫", "在", "住在") and
    the frame verb after it (``_TRAILS``: "上班"); every whole word of a name goes, so no part
    of it ("叫b") stays. The rest is cut at its clauses (``_CLAUSE_CUT``) and at the withheld
    words, and each piece carrying a word (``clause_words``) is kept: "我的猫" of "我的猫叫bob",
    "我女儿" and "上三年级" of "我女儿在实验小学上三年级", "我住在杭州西湖区" of "我住在杭州西湖
    区，在阿里上班". The caller still checks each piece against the forgotten words.

    Args:
        text: A quote that is not ASCII.
        said: What the forgotten statement says of the survivor's subject
            (``predicate_words``), its Latin words read here.
        forgotten: The forgotten statement, its Han runs read here.
        elsewhere: The text of the forgotten version's other quotes, where its words may
            be found instead.

    Returns:
        The kept pieces as ``(start, end)`` UTF-8 byte offsets, in text order; empty when the
        detail is not located or nothing readable is left.
    """
    latin = {word for word in said if not own_token(word)} - _FRAME
    values = ["".join(run) for run in _own_runs(forgotten)]
    name = naming_value(forgotten)
    if name is not None:
        # A naming statement's detail is the name alone: "小李" of "A person is named 小李.",
        # whatever the quote calls the person ("我的两个同事小王、小李").
        latin &= support_words(name)
        values = _HAN_RUN.findall(name)
    texts = (text, *elsewhere)
    found = {word for other in texts for match in _LATIN.finditer(other)
             for word in support_words(match.group()) & latin}
    if (not latin and not values) or not latin <= found or not all(
            any(value in other for other in texts) for value in values):
        return ()
    held = [match.span() for match in _LATIN.finditer(text)
            if support_words(match.group()) & latin]
    held += [(match.start(), match.end()) for value in values
             for match in re.finditer(re.escape(value), text)]
    cut: list[tuple[int, int]] = []
    for start, end in held:
        before = text[:start].rstrip()
        lead = next((mark for mark in _LEADS if before.endswith(mark)), "")
        start = len(before) - len(lead) if lead else start
        after = text[end:]
        trail = next((verb for verb in _TRAILS if after.lstrip().startswith(verb)), "")
        end = end + len(after) - len(after.lstrip()) + len(trail) if trail else end
        cut.append((start, end))
    kept: list[tuple[int, int]] = []
    position = 0
    for start, end in [*sorted(cut), (len(text), len(text))]:
        kept += clause_pieces(text, position, start)
        position = max(position, end)
    return tuple(kept)


_OWNED: Final = {
    "女儿": ("daughter",), "儿子": ("son",), "孩子": ("child", "kid"),
    "老婆": ("wife",), "妻子": ("wife",), "老公": ("husband",), "丈夫": ("husband",),
    "爸爸": ("father", "dad"), "父亲": ("father",), "妈妈": ("mother", "mom"),
    "母亲": ("mother",),
    "哥哥": ("brother", "older brother", "elder brother", "big brother"),
    "弟弟": ("brother", "younger brother", "little brother"),
    "姐姐": ("sister", "older sister", "elder sister", "big sister"),
    "妹妹": ("sister", "younger sister", "little sister"),
    "爷爷": ("grandfather", "grandpa"), "外公": ("grandfather", "grandpa"),
    "奶奶": ("grandmother", "grandma"), "外婆": ("grandmother", "grandma"),
    "男朋友": ("boyfriend",), "女朋友": ("girlfriend",),
    "猫": ("cat",), "猫咪": ("cat",), "狗": ("dog",), "狗狗": ("dog",), "仓鼠": ("hamster",),
    "兔子": ("rabbit",), "鹦鹉": ("parrot",), "乌龟": ("turtle", "tortoise"),
}
"""The user's kin and pets a Chinese quote names by a bare noun, each with the English
descriptions an existence memory gives it ("The user has a younger brother." for 弟弟)."""

_KIN: Final = frozenset(noun for noun in _OWNED if noun not in (
    "猫", "猫咪", "狗", "狗狗", "仓鼠", "兔子", "鹦鹉", "乌龟"))
"""The kin of ``_OWNED``, which a speaker names without 的 ("我女儿")."""


_GIVEN_FREE: Final = frozenset("".join(_OWNED)) | frozenset("表堂叫名字称作个位只") | _HAN_LIGHT
"""Han characters a withheld clause may show that are no value a head names: the user's kin
and pets (``_OWNED``), cousin prefixes, naming words, measure words and light characters
(``_given_safe``)."""

_REFERENT: Final = re.compile(
    f"我(的)?({'|'.join(sorted(_OWNED, key=len, reverse=True))})"
    f"(?=名字|叫|住|今年|上|读|[0-9一二三四五六七八九十两几{''.join(sorted(_HAN_LIGHT))}]|$)")
"""A clause opening with the user's own bare noun, "我的猫" of "我的猫叫bob" or "我女儿" of
"我女儿在实验小学…", the noun ending there: a naming, dwelling, schooling or age word, a light
character or the clause's end follows it."""


def referent_prefixes(text: str, description: str) -> tuple[tuple[int, int], ...]:
    """The phrases of a Chinese quote by which the user names a thing of theirs ("我的猫",
    "我女儿") at the head of a clause: what a survivor saying only that the user has that thing
    ("The user has a <description>.") keeps, or one about that thing ("The user's
    <description> is six years old this year."), when the forgotten detail about it cannot
    be found by its own words (``foreign_rest``).

    The phrase is "我" and a kin noun, or "我的" and a kin or pet noun (``_OWNED``), with no word
    between them and nothing joined after the noun, and ``description`` is one the noun is
    given: a modifier before the noun is very often the forgotten detail ("我的三岁女儿",
    "我的橘猫", "我的金毛", "我在杭州的房子"), so any other phrase keeps nothing. The caller still
    checks the phrase against the forgotten words.

    Args:
        text: A quote that is not ASCII.
        description: The survivor's description of the thing ("cat", "younger brother").

    Returns:
        The phrases as ``(start, end)`` UTF-8 byte offsets, in text order.
    """
    found: list[tuple[int, int]] = []
    for start, end in clause_pieces(text):
        piece = text.encode("utf-8")[start:end].decode("utf-8")
        match = _REFERENT.match(piece)
        if match is None or description not in _OWNED[match.group(2)] or (
                match.group(1) is None and match.group(2) not in _KIN):
            continue
        found.append((start, start + len(match.group().encode("utf-8"))))
    return tuple(found)


_MODIFIED_PHRASE: Final = re.compile(
    r"我的(?P<phrase>[\u4e00-\u9fff]{1,6}?)"
    r"(?:(?P<marker>名字叫做|名字叫|名字是|名叫|叫做|叫)(?P<name>[^\s，,。；;、！？!?]+))?$")
"""A clause naming the user's thing by a phrase of its own ("我的三岁女儿叫小美", "我的金毛叫旺
财", "我的橘猫"), and the name it gives it."""


def modifier_rest(text: str, description: str, statement: str) -> tuple[tuple[int, int], ...]:
    """The pieces of a Chinese quote a survivor about the user's own thing keeps when the
    forgotten detail is a modifier of that thing written before its noun, which no word of
    the detail finds (``foreign_rest``): "我的" and the noun when one ``description`` is given
    ends the phrase ("女儿" of "我的三岁女儿叫小美", "猫" of "我的橘猫叫bob"), and the naming
    after the phrase when it gives a name ``statement`` shows ("叫小美"). Every character of
    the phrase before the noun is withheld, whatever part of it the detail is. A phrase with
    no such noun ("我的金毛叫旺财", where the breed is the noun) or a clause of any other shape
    keeps nothing, and the caller checks each piece against the forgotten words.

    Args:
        text: A quote that is not ASCII.
        description: The survivor's description of the thing ("daughter", "cat").
        statement: The survivor's statement.

    Returns:
        The kept pieces as ``(start, end)`` UTF-8 byte offsets, in text order.
    """
    nouns = sorted((noun for noun, described in _OWNED.items() if description in described),
                   key=len, reverse=True)
    found: list[tuple[int, int]] = []
    for start, end in clause_pieces(text):
        piece = text.encode("utf-8")[start:end].decode("utf-8")
        match = _MODIFIED_PHRASE.match(piece)
        if match is None:
            continue
        phrase = match["phrase"]
        noun = next((noun for noun in nouns if phrase.endswith(noun) and phrase != noun), "")
        if not noun:
            continue  # The detail may be the noun itself ("金毛"): nothing is kept.

        def at(offset: int, piece: str = piece, start: int = start) -> int:
            return start + len(piece[:offset].encode("utf-8"))

        found += [(at(0), at(2)),  # "我的"
                  (at(match.end("phrase") - len(noun)), at(match.end("phrase")))]
        if match["name"] and token_contained(match["name"], statement):
            found.append((at(match.start("marker")), at(match.end("name"))))
    return tuple(found)


def own_clauses(text: str, statement: str, description: str | None) -> tuple[tuple[int, int], ...]:
    """The clauses of a Chinese quote that say nothing but a survivor's own Han words: every
    Han character of the clause, light ones aside (``clause_words``), is one of the survivor's
    statement, of the user's bare noun for the thing it is about (``description``) or the
    naming word 叫 ("叫小美" for "小美 is the user's daughter."). Such a clause carries no
    forgotten detail no word of the survivor's states, wherever the forget cites it, so a
    survivor whose own name the forgotten statement quotes in its own span keeps it ("叫小美"
    beside "我的三岁女儿" quoted apart); the caller still checks it against the forgotten
    words and names.

    Args:
        text: A quote that is not ASCII.
        statement: The survivor's statement.
        description: The survivor's description of the thing it is about, if any.

    Returns:
        The kept clauses as ``(start, end)`` UTF-8 byte offsets, in text order.
    """
    allowed = {word for word in clause_words(statement) if own_token(word)} | {"叫"}
    if description is not None:
        allowed |= {char for noun, described in _OWNED.items() if description in described
                    for char in noun}
    encoded = text.encode("utf-8")
    found: list[tuple[int, int]] = []
    for start, end in clause_pieces(text):
        han = {word for word in clause_words(encoded[start:end].decode("utf-8"))
               if own_token(word)}
        if han and han <= allowed:
            found.append((start, end))
    return tuple(found)


def clause_pieces(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """The clauses of ``text[start:end]`` (character offsets) that carry a word
    (``clause_words``), cut at ``_CLAUSE_CUT`` and stripped, as UTF-8 byte offsets of
    ``text``: "它三岁了" of "，它三岁了"; none of "我"."""
    stop = len(text) if end is None else end
    pieces: list[tuple[int, int]] = []
    begin = start
    for mark in [*_CLAUSE_CUT.finditer(text, start, max(start, stop)), None]:
        piece = text[begin:stop if mark is None else mark.start()]
        body = piece.strip()
        if body and clause_words(body):
            head = len(text[:begin + len(piece) - len(piece.lstrip())].encode("utf-8"))
            pieces.append((head, head + len(body.encode("utf-8"))))
        if mark is not None:
            begin = mark.end()
    return pieces


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
    """What privacy compensation requotes a forget's dependent on.

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
    any_script: bool = False,
    given: str = "",
    shared: frozenset[str] = frozenset(),
) -> TrimmedSupport | None:
    """The groups a forget's dependent keeps on the readable parts of its quotes.

    A support group is an AND of quotes, so one withheld byte hides it whole. An
    extracted group can carry a quote its statement does not need: "My birthday is
    March 3" beside "I'm planning a party for my sister" for "The user is planning a
    party for the user's sister.". One quote can also state two facts, "I
    volunteer at the animal shelter on weekends and I run in the mornings": forgetting
    the volunteering withholds the sentence but the part a clause keeper keeps.
    When compensation leaves such a dependent unreadable, a group still
    supports it on its readable parts alone when the words prove it:

    * the head is not erased and has no sensitivity class;
    * the group loses a byte to withholding only (its Evidence passes ``p_src``) and keeps a
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
      forgotten), so the requoted memory shows none.

    The test is on words and errs toward withholding: a synonym ("bachelor" for
    "single") is a word the readable parts must hold. A complete forget's English statement
    over a Han or mixed quote shares no word with it: there each readable clause showing no
    forgotten term is kept, and only the statement's own Han words must be read in the kept
    parts (``foreign_rest`` withheld the detail by its own words). No withheld byte becomes
    readable; the forget's stating spans are unchanged.

    Args:
        snapshot: The dependent's head.
        read: The scope's readability after the forget's withholding.
        quoted: The text a quote span or a piece of it quotes, or None when its Evidence
            cannot show it.
        forgotten: ``support_words`` of the forget's stating spans and of what its
            tombstones' statements say.
        speakers: ``support_words`` of the name of the speaker a quote span is attributed
            to; None when no quote has one.
        any_script: Whether a piece in another script is cut too, at its clauses
            (``clause_runs``), its words read by ``clause_words`` (a complete forget,
            ``explicit-request-v2``); otherwise such a piece keeps nothing. What the
            statement shows is then compared by ``shown_terms`` against the ``fact_terms``
            of the withheld bytes, and ``forgotten`` holds such terms too: a Han
            character the statement shares with the forgotten text ("上" of "上海" and
            "早上") is no shared word, and a clause is kept only when its character pairs
            are the statement's too (``clause_runs``).
        given: The statement the head showed over these quotes, when ``snapshot`` states a
            new wording of it (a rewording that leaves the forgotten value out): a
            capitalized name the head already showed over them ("Beijing" of "...attends
            university in Beijing." over "在北京上大学") is held by them, as ``_carried``
            reads whole quotes; one ``forgotten`` holds is still refused, as every forgotten
            word is. The statement is English when either is. The own requote of a
            head gives none: its statement is the head's. It holds nothing when the head may
            show a withheld value in another script (``_given_safe``: "Li Wei" over a withheld
            李伟, "Shanghai" over a withheld 上海).
        shared: The words the statement may share with the group's withheld bytes: the word a
            new value's name ends with that the forgotten name ends with too ("幼儿园" of a
            kept 蓝湾幼儿园 beside a withheld 启明幼儿园), which names neither.

    Returns:
        The qualifying groups and the keywords the requoted head keeps; None when no
        group qualifies.
    """
    if snapshot.erased or snapshot.sensitivity_classes:
        return None
    read_words = clause_words if any_script else support_words
    shown = shown_terms if any_script else support_words
    stated = read_words(snapshot.statement)
    shows = shown(snapshot.statement)
    if snapshot.time_expr.kind != "none":
        stated |= read_words(snapshot.time_expr.text)
        shows |= shown(snapshot.time_expr.text)
    if not stated or shows & forgotten:
        return None
    # An English statement over a Han quote shares no word with it (``foreign_rest``): the
    # admission withheld the forgotten detail by its own words, and each readable clause left
    # that shows no forgotten term is kept; only the statement's own Han words must be read,
    # and every name it gives must be shown by what it keeps (``unshown_names``). The caller
    # refuses a dependent whose vector restates a forgotten statement (``CLAUSE_COSINE``).
    crossed = any_script and (english(snapshot.statement) or bool(given) and english(given))
    needed = frozenset(word for word in stated if own_token(word)) if crossed else stated

    def words(parts: Sequence[QuoteSpanV1],
              read: Callable[[str], frozenset[str]] = read_words) -> frozenset[str] | None:
        found: set[str] = set()
        for part in parts:
            text = quoted(part)
            if text is None:
                return None
            found |= read(text)
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
                if text.isascii():
                    runs = word_runs(text, stated)
                elif crossed:
                    encoded = text.encode("utf-8")
                    runs = tuple((start, end) for start, end in clause_pieces(text)
                                 if not shown_terms(encoded[start:end].decode("utf-8"))
                                 & forgotten)
                elif any_script:
                    runs = clause_runs(text, stated, shows)
                else:
                    continue  # A piece in another script is never cut.
                kept.extend(QuoteSpanV1(evidence_id=piece.evidence_id,
                                        byte_start=piece.byte_start + start,
                                        byte_end=piece.byte_start + end)
                            for start, end in runs)
        if not usable or not hidden or not kept:
            continue
        kept_words = words(kept)
        hidden_words = words(hidden, fact_terms if any_script else read_words)
        if kept_words is not None and speakers is not None:
            kept_words = kept_words.union(*(speakers(part) for part in kept))
        if crossed and unshown_names(snapshot.statement, [
                *(text for part in kept if (text := quoted(part)) is not None),
                given if given and _given_safe(given, [
                    text for part in hidden if (text := quoted(part)) is not None]) else ""]):
            continue
        if (kept_words is not None and hidden_words is not None
                and needed <= kept_words and not shows & hidden_words - shared):
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
                       if not shown(keyword) & dropped),
    )
