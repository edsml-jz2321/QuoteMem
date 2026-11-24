"""Pure forget admission selection over the selected memory's whole natural-language revision chain.

This preflight grants no mutation authority. Admission must bind its inputs
under the Writer, check authorization/intent/epochs, and commit atomically.
``plan_stating_spans`` then narrows the cited spans to the stating spans S that
the forget withholds (the bare-name rule, the
referent keeper and the clause keeper).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence, Set
from dataclasses import dataclass, field, replace
from typing import Final, Literal

from quotemem.core.entities import LinkType, MemorySnapshotV2, QuoteSpanV1, SupersedeRelation
from quotemem.core.errors import QuoteMemError
from quotemem.core.forget import MAX_FORGET_SOURCES, MAX_FORGET_TOMBSTONES
from quotemem.core.forget_match import TIES
from quotemem.core.forget_state import MAX_FORGET_CHAIN
from quotemem.core.judge_structure import (
    FORMAL_KINSHIP,
    content_words,
    distinctive_words,
    ended_parts,
    replaced_ranges,
)
from quotemem.core.naming_syntax import bare_naming
from quotemem.core.privacy import (
    clause_pieces,
    clause_runs,
    clause_words,
    english,
    fact_terms,
    foreign_rest,
    kept_beside,
    modifier_rest,
    naming_value,
    own_clauses,
    referent_prefixes,
    shown_terms,
    support_words,
    unshown_names,
    word_runs,
)
from quotemem.core.sensitivity_rules import classify
from quotemem.core.statement import (
    MAX_TOMBSTONE_SOURCES,
    name_norm,
    own_token,
    token_contained,
    token_seq,
)
from quotemem.store.domain import MemoryRow, RevisionRowV2

_ENDED_MARKS: Final = (
    "不再", "再也不", "没再", "不打了", "而不是", "而非", "停", "戒", "放弃", "退出", "代替",
    "取代", "替代",
    "以前", "原来", "之前", "曾经", "改", "换", "もう", "代わり", "やめ", "대신", "더 이상",
    "전에", "그만",
)
"""Words of a clause in Chinese, Japanese or Korean that end or replace a value. A bare
negation ("没打过", "不打") states the opposite and presupposes no old value."""
_CLAUSE_CUT: Final = re.compile(r"[,;，；、。！？!?\n]|\.(?:\s|$)")
_PEOPLE: Final = frozenset({
    *TIES, "coach", "coaches", "teacher", "teachers", "trainer", "tutor", "doctor", "manager",
    "classmate", "classmates", "teammate", "teammates", "mentor", "landlord", "student",
    "students", "client", "clients", "pet", "pets", "老师", "教练", "朋友", "同事", "同学",
    "室友", "邻居", "老板", "医生", "宠物", "队友",
})
"""Words that make a possessive or a clause about another person or an animal ("my coach",
"the user's cat", "妹妹"), with the kin and pets of ``forget_match.TIES``."""
_NOT_NAMES: Final = frozenset({
    "the", "a", "an", "since", "after", "before", "until", "when", "once", "last", "this",
    "these", "next", "every", "now", "nowadays", "lately", "recently", "currently", "today",
    "then", "so", "but", "and", "in", "at", "on", "for", "from", "with", "instead", "rather",
    "as", "during", "while", "because", "correction", "also", "still", "no", "not", "never",
    "yesterday", "tonight", "got", "ok", "okay", "sure", "great", "nice", "noted", "yes",
    "yeah", "sorry", "well", "fine", "thanks", "sounds", "welcome", "congrats",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
})
"""Capitalized words that open or date a clause and name nobody (``_party``)."""
_CLAUSE_OPENERS: Final = frozenset({
    "since", "after", "before", "until", "when", "once", "last", "this", "these", "next",
    "every", "now", "nowadays", "lately", "recently", "currently", "today", "then", "in", "at",
    "on", "during", "while", "because", "correction", "ever", "yesterday", "tonight",
})
"""Words that open a clause with when or where it holds ("Since the move I ...", "Last month
I ..."): the subject follows them (``_speaker``)."""
_ADJUNCTS: Final = frozenset({
    "since", "after", "before", "because", "when", "while", "as", "at", "in", "on", "for",
    "with", "to", "from", "last", "every", "this", "that", "which", "who", "and", "but", "so",
    "then", "near", "during", "until", "anymore", "now", "these", "or",
})
"""Words that open what a replaced part says around its value ("watching badminton on TV",
"the blue Toyota and now ride a bike"): a part's own value ends before them (``_beyond``)."""
_NEUTRAL_WORDS: Final = frozenset(content_words(
    "part routine routines life habit habits schedule thing things pet pets sport sports game "
    "games food foods dish dishes drink drinks car cars vehicle vehicles activity activities "
    "hobby hobbies place places"))
"""Words of an ended clause that name no value of their own ("Badminton is no longer part of
the user's routine.")."""
_NEUTRAL_HAN: Final = frozenset("去来在了的过再也还都就要会是而")
"""Han characters a Chinese ended clause may put before the target's words without naming a
value of its own ("不再去俱乐部打羽毛球")."""
_WORD: Final = re.compile(r"[A-Za-z][\w'\u2019-]*")
_ACKNOWLEDGED: Final = frozenset(distinctive_words(
    "got it noted okay ok sure thanks thank alright yes yeah right understood gotcha "
    "remember remembered keep mind will"))
"""Words of an acknowledgement, which say nothing of the fact acknowledged (``echoes``); a
judgement ("great", "nice") is a word of its own ("Shanghai has great food")."""
_ACKNOWLEDGED_HAN: Final = frozenset("好的嗯是对记住收到明白知道")
"""Han characters of an acknowledgement ("好的", "记住了")."""
_HAN_TIME: Final = re.compile(
    r"(?:每|这|上|下|本)?(?:周|星期|礼拜)[一二三四五六日天末]?|(?:每天|今天|昨天|明天|周末|每晚|"
    r"早上|上午|中午|下午|晚上|凌晨|傍晚|平时|现在|以前|最近)")
"""Chinese time words, adjuncts of a statement's value ("每周六早上")."""
_HAN_ACTIONS: Final = (
    "打", "玩", "踢", "练", "学", "做", "吃", "喝", "看", "弹", "唱", "跳", "游", "跑", "骑", "开",
    "养", "种", "买", "读", "写", "画", "参加", "教", "滑", "钓", "下棋",
)
"""Chinese verbs that take an object after a place phrase ("在徐汇的俱乐部打羽毛球")."""
_HAN_PLACE: Final = re.compile(
    r"(?:在|从|于)[^\s，。！？、；,.!?]{1,12}?(?=" + "|".join(_HAN_ACTIONS) + ")")
"""A Chinese place phrase before a verb with an object: an adjunct of the value, not it."""
_HAN_GENERIC: Final = re.compile(r"工作|上班|生活|居住|住")
"""Chinese verbs that name no value of their own, the place or employer is ("在腾讯工作")."""


def names_replaced(statement: str, target: str, *, unsure: bool = True) -> bool:
    """Whether a statement names a value a target states as one it replaced, ended or denied.

    A complete forget of the target reads it (``ForgetOwner``'s replacers): "The user goes
    climbing at a gym near West Lake instead of playing badminton.", "The user no longer
    plays badminton.", "The user moved from Shanghai to Hangzhou." and "The user sold the
    blue Toyota." still tell the value a forget of the target is about, whatever else they
    state. In English a part that names a replaced, left, ended or denied value
    (``judge_structure.ended_parts``) shares a word with the target that names a value of its
    own (``distinctive_words``: no generic verb, category or time word, so "sushi rather than
    any other food" does not name "The user's favourite food is pizza."). In Chinese,
    Japanese or Korean a clause with a word of ``_ENDED_MARKS`` ("不再打羽毛球") shows a term
    of the target (``privacy.fact_terms``). Such a part names the target's value only when it
    is about the same person (``_party``: never "Mei no longer plays badminton.", "My coach
    left the Xuhui club." or "My cat is not allowed at the Xuhui club.") and names no value
    the target does not state (``_beyond``: never "I stopped watching badminton on TV." or "I
    do not like badminton shoes." for a target that plays it). A statement mostly in another
    script than the target's, or a replaced value in another script ("instead of 打羽毛球"
    against an English target), is read as naming it: no test here reads one against the
    other, so the forget fails closed.

    Args:
        statement (str): A statement that replaced the target.
        target (str): One of the target's statements.
        unsure (bool): What a statement no test can read against the target comes to: True
            for a memory linked to the target as its replacer (fail closed), False for one
            the forget only found (``StatementForget``'s named candidates).

    Returns:
        bool: Whether the statement names a value of the target.
    """
    if _own_script(statement) != _own_script(target):
        return unsure
    text = unicodedata.normalize("NFKC", statement)
    held, terms = content_words(target), _own_terms(target)
    value, core = _value_words(target), value_terms(target)
    for part in ended_parts(statement):
        if distinctive_words(part) & value or shown_terms(part) & terms:
            if _party(text, part, target) != "other" and not _beyond(part, held):
                return True
            continue
        if not terms and any(own_token(token) for token in token_seq(part)):
            return unsure  # a value in another script than the target's
    return bool(terms) and any(
        (any(mark in clause for mark in _ENDED_MARKS) and shown_terms(clause) & core
         and not _beyond_han(clause, target, core)
         or any(shown_terms(left["left"]) & core for left in _HAN_LEFT.finditer(clause)))
        and _party(text, clause, target) != "other"
        for clause in _CLAUSE_CUT.split(text))


def value_terms(target: str) -> frozenset[str]:
    """The Han, Kana or Hangul terms naming a target's value (``privacy.fact_terms``): those of
    its predicate, never of the time or place it holds at ("每周六早上", "在徐汇的俱乐部" of
    "我每周六早上在徐汇的俱乐部打羽毛球": its value is 打羽毛球) nor of a verb naming none
    ("上海" of "我住在上海", "腾讯" of "我在腾讯工作"), as ``_value_words`` reads an English one;
    all its terms when nothing else is left."""
    text = unicodedata.normalize("NFKC", target)
    core = _HAN_PLACE.sub(" ", _HAN_TIME.sub(" ", text))
    found = _own_terms(_HAN_GENERIC.sub(" ", core)) or _own_terms(core)
    return found or _own_terms(text)


def _value_words(target: str) -> set[str]:
    """The words naming a target's value (``distinctive_words``): those of its predicate, the
    verb's object or, without one, its first complement ("badminton" of "I play badminton
    every Saturday morning at a club in Xuhui.", "Shanghai" of "I live in Shanghai."), never
    the adjuncts after it (the time, the club, the district): a replaced or ended part names
    the target's value only by these, so "works in Pudong now, not Xuhui" names none."""
    tokens = _tokens_of(target)
    at = 0
    # The subject: "I", "the user", a possessive and its noun, or a name.
    while at < len(tokens) and tokens[at] in ("the", "a", "an", "i", "user", "my", "we"):
        at += 1
    if at < len(tokens) and tokens[at - 1:at] in (["my"], ["user's"]):
        at += 1
    if at < len(tokens) and tokens[at] in ("user's", "my"):
        at += 2
    kept: list[str] = []
    found = False
    for token in tokens[at + 1:]:  # the verb, tokens[at], names no value of its own
        if found and token in _ADJUNCTS:
            break
        kept.append(token)
        found = found or bool(distinctive_words(token))
    return distinctive_words(" ".join(kept)) or distinctive_words(target)


def _tokens_of(text: str) -> list[str]:
    """A text's English words, folded, apostrophes kept ("user's")."""
    return [word.casefold().replace("\u2019", "'")
            for word in _WORD.findall(unicodedata.normalize("NFKC", text))]


def _party(statement: str, part: str, target: str) -> Literal["user", "other"] | None:
    """Whom a statement's replaced ``part`` is about: the person who speaks of a part that
    opens its clause ("Mei no longer plays badminton"), or the subject before it ("My coach Li
    left the Xuhui club"). "other" is another person or an animal: a word of ``_PEOPLE`` after
    "my" or "the user's" or in a Chinese clause, or a capitalized name the target does not
    use; None when the words name nobody."""
    at = statement.find(part)
    before = statement[:at] if at >= 0 else ""
    local = _CLAUSE_CUT.split(before)[-1]
    # A part that opens its clause names its own subject; any other is said of the subject
    # of its clause ("..., the user goes climbing instead of playing badminton"), then of the
    # sentence.
    for text in ((part, before) if not local.strip() else (local, part, before)):
        found = _speaker(text, target)
        if found is not None:
            return found
    return _speaker(statement, target)


def _speaker(text: str, target: str) -> Literal["user", "other"] | None:
    """Who the first word naming a person in ``text`` names (``_party``)."""
    if any(own_token(token) for token in token_seq(text)):
        if "我们" in text or any(join in text for join in ("我和", "我跟", "我与", "和我", "跟我")):
            return "user"  # a joint subject that includes the user tells the user's value
        if any(word in text and word not in target for word in _PEOPLE if not word.isascii()):
            return "other"
        return "user" if "我" in text else None
    own = {word.casefold().replace("\u2019", "'") for word in _WORD.findall(target)}
    tokens = _WORD.findall(text)
    # The subject comes first, after what opens the clause ("Since the move I ...", "Got it
    # — ..."); a name later in the clause ("... at a gym near West Lake") is no subject.
    opening = bool(tokens) and tokens[0].casefold() in _CLAUSE_OPENERS
    for index, token in enumerate(tokens):
        word = token.casefold().replace("\u2019", "'")
        if word in ("i", "i'm", "i've", "i'd", "i'll", "me", "user", "we", "we're", "we've",
                    "us", "our"):
            return "user"
        # A joint subject that includes the user ("My wife and I", "Mei and I") is the
        # user's: it tells the user's value too.
        joint = [part.casefold() for part in tokens[index + 1:index + 5]]
        joined = any(joint[at] == "and" and joint[at + 1] in ("i", "me", "the")
                     for at in range(len(joint) - 1))
        if word in ("my", "user's"):
            following = tokens[index + 1].casefold() if index + 1 < len(tokens) else ""
            return "other" if following in _PEOPLE and following not in own and not joined \
                else "user"
        named = token[:1].isupper() and word not in _NOT_NAMES and word not in own and not (
            index == 0 and word.endswith("ing"))  # never an activity opening it ("Climbing")
        if named:
            return "user" if joined else "other"
        if not (opening or word in _NOT_NAMES or word in own or token[:1].isupper()):
            return None  # the clause opens with what is said, not with who says it
    return None


def _beyond(part: str, held: Set[str]) -> bool:
    """Whether an English replaced part names a value the target does not state: its words up
    to the first word of ``_ADJUNCTS`` after its start carry a topic word (``distinctive_words``)
    the target lacks ("watching badminton" for a target that plays it, "badminton shoes")."""
    tokens = part.split()
    words = [token.casefold().strip(",.;:!?") for token in tokens]
    # A clause says its value after its subject ("Since the move I no longer play
    # badminton", "My wife and I no longer play badminton", "The user and the user's wife
    # no longer play badminton").
    subject = [index for index, word in enumerate(words[:8])
               if word in ("i", "user", "we", "us") or (
                   index and words[index - 1] in ("user's", "my", "our")
                   and "and" in words[:index - 1])]
    first = subject[-1] + 1 if subject else 0
    cut = next((index for index in range(first + 1, len(words)) if words[index] in _ADJUNCTS),
               len(words))
    return bool(content_words(" ".join(tokens[first:cut])) - held - _NEUTRAL_WORDS)


def _beyond_han(clause: str, target: str, core: Set[str]) -> bool:
    """Whether a Chinese, Japanese or Korean ended clause names a value the target does not
    state: past the word ending its value (``_ending_at``: "而不是打羽毛球" of "我现在周六早上去
    攀岩而不是打羽毛球"), one character before the first term naming the target's value
    (``value_terms``: its ``core``), or one between them, that the target lacks and that is
    no light word of ``_NEUTRAL_HAN`` ("看" of "不再看羽毛球比赛" for a target that plays it)."""
    text = clause[next(iter(_ending_at(clause)), 0):]
    for mark in _ENDED_MARKS:
        text = text.replace(mark, " ")
    shared = [term for term in shown_terms(text) if len(term) > 1 and term in core]
    if not shared:
        return False
    starts = [text.find(term) for term in shared]
    ends = [text.find(term) + len(term) for term in shared]
    low, high = max(0, min(starts) - 1), max(ends)
    return any(own_token(character) and character not in target
               and character not in _NEUTRAL_HAN for character in text[low:high])


def echoes(clause: str, values: Sequence[str]) -> bool:
    """Whether a clause of the assistant's reply states a forgotten value: it names a value of
    one of ``values`` (their statements and quoted words: ``_value_words``, ``value_terms``)
    and either says nothing that one does not (every English word naming something, a
    category included, ``content_words``, acknowledgements aside; every Han, Kana or Hangul
    character, light ones aside), or, naming no other person, states it with its predicate
    (the value's verb) or two more of its words (its time, place or club). "Shanghai, noted.",
    "You live in Shanghai" and "Saturday morning badminton at your club in Xuhui is a standing
    date" state a forgotten value; "Shanghai has great food", "Have a nice Saturday",
    "Tencent is a great company" and "Climbing for you, badminton for Mei" state none, nor
    does "数据平台开发很有意思" a forgotten interview at 青川数据."""
    words = {word for word in content_words(clause) if word.isascii()} - _ACKNOWLEDGED
    marks = {character for character in clause
             if own_token(character) and character not in _NEUTRAL_HAN
             and character not in _ACKNOWLEDGED_HAN}
    terms = shown_terms(clause)
    for value in values:
        if marks:
            core = value_terms(value)
            if terms & core and (
                    (marks <= set(value) and words <= content_words(value)) or (
                        len(terms & (_own_terms(value) - core)) >= 2
                        and not any(word in clause for word in _PEOPLE
                                    if not word.isascii()))):
                return True
        elif named := words & _value_words(value):
            if words <= content_words(value):
                return True
            told = (words - named) & content_words(value)
            verb = _verb_of(value)
            if not _names_other(clause, value) and (len(told) >= 2 or verb is not None and (
                    verb in {_stem(token) for token in _tokens_of(clause)})):
                return True
    return False


def _verb_of(target: str) -> str | None:
    """The stem of a target's verb, after its subject ("play" of "I play badminton ...")."""
    tokens = _tokens_of(target)
    at = 0
    while at < len(tokens) and tokens[at] in ("the", "a", "an", "i", "user", "my", "we"):
        at += 1
    if at < len(tokens) and tokens[at] in ("user's", "my"):
        at += 2
    return _stem(tokens[at]) if at < len(tokens) else None


def _names_other(clause: str, value: str) -> bool:
    """Whether a clause names another person or an animal than the value's subject: a word of
    ``_PEOPLE`` or a capitalized name the value does not use ("badminton for Mei")."""
    own = {word.casefold() for word in _WORD.findall(value)}
    tokens = _WORD.findall(clause)
    return any(
        token.casefold() in _PEOPLE or (
            index and token[:1].isupper() and token.casefold() not in own
            and token.casefold() not in _NOT_NAMES)
        for index, token in enumerate(tokens))


def replaced_parts(statement: str) -> list[str]:
    """The parts of a statement that name a value as replaced, ended or denied, as
    ``names_replaced`` reads them: the English ones (``judge_structure.ended_parts``) and each
    Chinese, Japanese or Korean clause with a word of ``_ENDED_MARKS`` ("不再打羽毛球")."""
    return [*ended_parts(statement), *(
        clause for clause in _CLAUSE_CUT.split(unicodedata.normalize("NFKC", statement))
        if any(mark in clause for mark in _ENDED_MARKS))]


def mentions(text: str, values: Sequence[str]) -> bool:
    """Whether a clause of a replacer's quote mentions one of the forgotten ``values`` (their
    statements and quoted words), so a forget withholds it.

    The clause mentions a value when it shows an English word of it that names a value of its
    own (``distinctive_words``) or a term of another script (``privacy.fact_terms``). When no
    value is mostly in the clause's script, no test reads the clause, so it is withheld all
    the same (the forget fails closed): "搬家后我改去西湖附近的攀岩馆攀岩了" is read against the
    Chinese sentence "我每周六早上在徐汇的俱乐部打羽毛球。" and mentions none of it.
    """
    words = distinctive_words(text)
    terms = shown_terms(text)
    if any(words & distinctive_words(value) or terms & _own_terms(value) for value in values):
        return True
    script = _own_script(text)
    return not any(_own_script(value) == script for value in values)


def naming_parts(clause: str, values: Sequence[str], *, strict: bool = False,
                 ) -> list[tuple[int, int]]:
    """What a forget withholds of a clause that mentions one of the forgotten ``values``
    (``mentions``): only its parts naming a value as replaced, ended or left
    (``judge_structure.replaced_ranges``) that mention one, when the rest of the clause
    mentions none, so another memory quoting the clause can be requoted on the rest;
    otherwise the whole clause.

    "Since the move I go climbing at a gym near West Lake instead of playing badminton" keeps
    "Since the move I go climbing at a gym near West Lake" readable and "Last month I moved
    from Shanghai to Hangzhou" keeps "Last month I moved" and "to Hangzhou": neither rest names
    a forgotten value. A clause in another script than every value is read by no test and is
    withheld whole, as ``mentions`` fails closed.

    ``strict`` reads a clause no judge decided ("I go climbing on Saturday mornings instead of
    playing badminton"): only the words naming a value of ``values`` count (``names_value``),
    never a shared time or place, and a Chinese, Japanese or Korean clause is cut at the word
    ending its value ("而不是打羽毛球" of "我现在周六早上去攀岩而不是打羽毛球").

    Args:
        clause (str): One clause, with no clause mark inside.
        values (Sequence[str]): The forgotten statements and quoted words.
        strict (bool): Withhold only what names a value (a word rule's own withholding).

    Returns:
        list[tuple[int, int]]: Character ranges into ``clause``, in order.
    """
    def named(text: str) -> bool:
        return names_value(text, values) if strict else mentions(text, values)

    ranges = [(start, end) for start, end in replaced_ranges(clause) if named(clause[start:end])]
    if strict and not ranges:
        # A Chinese move keeps its arrival: only "从上海" of "从上海搬到了杭州" is withheld.
        ranges = [match.span("left") for match in _HAN_LEFT.finditer(clause)
                  if named(match["left"])]
    if strict and not ranges:
        ranges = [(start, len(clause)) for start in _ending_at(clause)][:1]
        if ranges and not named(clause[ranges[0][0]:]):
            ranges = []
    rest = clause
    for start, end in ranges:
        rest = rest[:start] + " " * (end - start) + rest[end:]
    if not ranges or named(rest):
        return [(0, len(clause))]
    return ranges


_HAN_LEFT: Final = re.compile(r"从(?P<left>[^\s，。！？、；,.!?]{1,12}?)(?:搬|迁|移居)")
"""A Chinese move naming the place it left ("从上海搬到了杭州"): that place is a value it ended."""
_HAN_ENDING: Final = ("不再", "再也不", "没再", "而不是", "而非", "代替", "取代", "替代", "放弃",
                      "退出", "停", "戒", "もう", "代わり", "やめ", "대신", "더 이상", "그만")
"""Words of a Chinese, Japanese or Korean clause after which it names the value it ended or
replaced ("不再打羽毛球"); "以前" or "改" mark the whole clause."""


def _ending_at(clause: str) -> list[int]:
    """Where a Chinese, Japanese or Korean clause starts naming the value it ended (the first
    word of ``_HAN_ENDING``), when it has one."""
    found = [at for mark in _HAN_ENDING if (at := clause.find(mark)) >= 0]
    return [min(found)] if found else []


def names_value(text: str, values: Sequence[str]) -> bool:
    """Whether a text names a value of one of ``values``: an English word of its predicate
    (``_value_words``) or a term of it (``value_terms``); a time, a place or another adjunct
    it shares names none ("in Xuhui" for a badminton club in Xuhui)."""
    words, terms = distinctive_words(text), shown_terms(text)
    return any(words & _value_words(value) or terms & value_terms(value) for value in values)


def states_value(texts: Sequence[str], quoted: Sequence[str], statements: Sequence[str],
                 quotes: Sequence[str]) -> bool:
    """Whether a memory a forget leaves unreadable states a value of what it forgets, so its
    own value is forgotten too and a memory that replaced it and names it goes with it.

    A memory only made unreadable because it quotes part of a withheld sentence ("The user
    works at Tencent." in "I play badminton every Saturday morning, and I work at Tencent.")
    states nothing forgotten: its statements (``texts``) share no word that names a value of
    its own (``distinctive_words``) or term (``privacy.fact_terms``) with the forgotten
    ``statements`` in their script. With no forgotten statement in a statement's script, its
    ``quoted`` words are read against the forgotten statements and ``quotes`` in theirs
    ("我每周六早上在徐汇的俱乐部打羽毛球" against the request "我每周六早上打羽毛球").
    """
    for text in texts:
        same = [value for value in statements if _own_script(value) == _own_script(text)]
        if same:
            if any(_shares(text, value) for value in same):
                return True
            continue
        pool = [*statements, *quotes]
        if any(_shares(quote, value) for quote in quoted for value in pool
               if _own_script(value) == _own_script(quote)):
            return True
    return False


def _shares(text: str, value: str) -> bool:
    """Whether two texts share a word that names a value of its own or a term of another
    script."""
    return bool(distinctive_words(text) & distinctive_words(value)
                or _own_terms(text) & _own_terms(value))


def _own_script(text: str) -> bool:
    """Whether a text is mostly Han, Kana or Hangul: more such characters than Latin words."""
    tokens = token_seq(text)
    own = sum(1 for token in tokens if own_token(token))
    latin = sum(1 for token in tokens if not own_token(token) and token.isascii()
                and any(character.isalpha() for character in token))
    return own > latin


def _own_terms(text: str) -> frozenset[str]:
    """The Han, Kana or Hangul terms a text forbids (``privacy.fact_terms``)."""
    return frozenset(term for term in fact_terms(text) if not term.isascii())


@dataclass(frozen=True)
class CorrectionEdge:
    """A same-scope extant memory link with the source proposal's correction provenance.

    ``correction_requested`` is the link's own flag: on a ``contradicts`` link the
    proposal requested a correction, on a ``supersedes`` link it stated an
    explicit correction.
    """

    target: str
    kind: LinkType
    relation: SupersedeRelation | None
    correction_requested: bool


def correction_chain(memory_id: str, links: Callable[[str], Iterable[CorrectionEdge]],
                     ) -> tuple[str, ...]:
    """Return every memory the selected memory transitively refuted.

    Args:
        memory_id: The selected memory, excluded from the result even in a cycle.
        links: All revisions' links to extant memories of the same scope. The caller
            proves scope and original proposal provenance before yielding each edge.

    Returns:
        Sorted correction/retraction predecessors, including contradiction targets
        only when their source proposal requested a correction and update targets
        only when it stated an explicit correction. Other updates and ordinary
        disagreements remain outside the retention chain.

    Raises:
        QuoteMemError: The chain exceeds the production admission bound.
    """
    found: set[str] = set()
    pending = [memory_id]
    while pending:
        for edge in links(pending.pop()):
            if not (edge.kind == "contradicts" and edge.correction_requested or
                    edge.kind == "supersedes" and (edge.relation in ("correct", "retract")
                                                   or edge.correction_requested)):
                continue
            if edge.target == memory_id or edge.target in found:
                continue
            if len(found) >= MAX_FORGET_CHAIN:
                raise QuoteMemError("budget_exceeded", "The forget chain exceeds its bound")
            found.add(edge.target)
            pending.append(edge.target)
    return tuple(sorted(found))


@dataclass(frozen=True)
class ForgetVersionSelection:
    """Newest version of one distinct pair, with all of that pair's sources.

    The snapshot carries the original vector; Forget never embeds it again.
    This is private plan input, never a receipt or a read projection.
    """

    revision_id: str
    snapshot: MemorySnapshotV2 = field(repr=False)
    source_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ForgottenCitation:
    """What one non-erased forgotten version cites, for the stating-span plan.

    Attributes:
        entity_id: The version's primary entity.
        names: The ``(entity_id, name_norm)`` pairs its naming annotations give.
        groups: Its support groups, each as the spans it quotes.
        about: Its other anchors (``about_entity_ids``).
        sensitive: Whether this version has any sensitivity class.
        statement: Its statement, read only by the detail-keeper test
            (``detail_names``); private plan input, never rendered.
        vector: Its statement vector and ``space`` its embedding space, read only by
            the clause keeper (``clause_spans``); private plan input.
        modifier: Whether the user asked to forget only the modifiers of a thing the
            version says the user has (``asks_modifier``): then a survivor about that thing
            keeps what does not show them (``possession_entailed``, ``predicate_words``).
    """

    entity_id: str
    names: frozenset[tuple[str, str]] = field(repr=False)
    groups: tuple[tuple[QuoteSpanV1, ...], ...]
    about: frozenset[str] = field(default=frozenset[str](), repr=False)
    sensitive: bool = False
    statement: str = field(default="", repr=False)
    vector: tuple[float, ...] = field(default=(), repr=False)
    space: str = field(default="", repr=False)
    modifier: bool = False


@dataclass(frozen=True)
class KeeperGroup:
    """One readable support group of a surviving memory that may keep quotes.

    A naming memory's group may keep its quotes on paths (a)-(c); any memory's group
    may keep its referent quote on path (d) (``referent_spans``); a head a forgotten
    correction reworded keeps its own groups on path (f) (``reworded``).

    Attributes:
        names: The ``(entity_id, name_norm)`` pairs the survivor's head gives; empty
            for a memory that names nothing.
        spans: The group's quote spans; each passes ``p_span`` at admission.
        bare: The one entity the survivor's head anchors when the head names it
            by type alone ("A person is named Rowan."). Its exact name spans,
            including names within broader quotes, are the only bytes this path keeps;
            otherwise None.
        sensitive: Whether the survivor's head has any sensitivity class; a
            sensitive keeper keeps nothing on any path.
        statement: The survivor head's statement (``detail_names``, ``referent_spans``).
        anchors: The survivor head's anchors, primary first.
        quoted: The text each of ``spans`` quotes, in the same order.
        persons: The named entities whose type is "person" (``detail_names``).
        stated: ``support_words`` of the head's statement and time expression
            (``clause_spans``).
        vector: The head's statement vector and ``space`` its embedding space
            (``clause_spans``).
        reworded: The survivor's head is a rewording by a forgotten correction:
            the group is its own support, kept for every version that withdraws no
            name (path f, ``plan_stating_spans``).
        speakers: ``support_words`` of the speaker each of ``spans`` is attributed to
            (``Evidence.speaker_entity``), in the same order; empty when unknown. A quote's
            speaker is readable provenance, so it grounds the speaker's name in the
            survivor's statement (``clause_spans``).
        reference_names: Readable names of the primary entity given by surviving naming
            memories. Only the descriptor-detail fallback reads these; they grant no
            naming annotation, bare-name keeper or identity binding.
        uncut: A naming-only memory under a question for a place: no clause of it is cut
            across languages and no name of it is located inside Han text
            (``ForgetOwner._plan_of``).
        memory_id: The surviving memory, read only by a forget's preview
            (``ForgetOwner.preview``).
        entity_type: The stored broad type of the primary entity, used only to
            reject a referent that restates possession of that same kind.
    """

    names: frozenset[tuple[str, str]] = field(repr=False)
    spans: tuple[QuoteSpanV1, ...]
    bare: str | None = None
    sensitive: bool = False
    statement: str = field(default="", repr=False)
    anchors: tuple[str, ...] = field(default=(), repr=False)
    quoted: tuple[str, ...] = field(default=(), repr=False)
    persons: frozenset[str] = field(default=frozenset[str](), repr=False)
    stated: frozenset[str] = field(default=frozenset[str](), repr=False)
    vector: tuple[float, ...] = field(default=(), repr=False)
    space: str = field(default="", repr=False)
    reworded: bool = False
    speakers: tuple[frozenset[str], ...] = field(default=(), repr=False)
    reference_names: frozenset[tuple[str, str]] = field(default=frozenset(), repr=False)
    entity_type: str = field(default="", repr=False)
    memory_id: str = field(default="", repr=False)
    uncut: bool = False


@dataclass(frozen=True)
class ForgetAdmissionSelection:
    """Complete distinct tombstone versions and exact cited spans of one memory.

    ``spans`` is every cited span; ``cited`` keeps each non-erased version's own
    groups so that ``plan_stating_spans`` can tell what states the forgotten fact.
    """

    versions: tuple[ForgetVersionSelection, ...] = field(repr=False)
    spans: tuple[QuoteSpanV1, ...]
    cited: tuple[ForgottenCitation, ...] = field(default=(), repr=False)


def plan_forget_admission(
    item: MemoryRow,
    revisions: Sequence[RevisionRowV2],
) -> ForgetAdmissionSelection:
    """Select D and S before any write, independently of heartbeat traffic.

    Revisions must be the complete chain of exactly the selected memory. All
    lifecycle states are accepted. Erased versions add neither tombstones nor
    spans; an erased head still retains the non-erased versions before it.
    The caller must prove inventory completeness against the durable store.

    Raises:
        QuoteMemError: For an incomplete/misbound chain or the pre-write bounds.
    """
    item = MemoryRow.model_validate(item.model_dump(mode="python"))
    ordered = sorted(
        (RevisionRowV2.model_validate(row.model_dump(mode="python")) for row in revisions),
        key=lambda row: row.version,
    )
    if not ordered:
        raise QuoteMemError("closure_incomplete",
                            "Forget needs the complete selected revision chain")
    previous: str | None = None
    seen: set[str] = set()
    newest: dict[tuple[str, str], RevisionRowV2] = {}
    sources: dict[tuple[str, str], set[str]] = {}
    spans: dict[tuple[str, int, int], QuoteSpanV1] = {}
    cited: list[ForgottenCitation] = []
    for version, row in enumerate(ordered, 1):
        if row.memory_id != item.object_id:
            raise QuoteMemError("binding_mismatch", "Forget revision belongs to another Item")
        if row.version != version or row.previous_revision_id != previous or row.object_id in seen:
            raise QuoteMemError("closure_incomplete",
                                "Forget revision chain is incomplete or ambiguous")
        seen.add(row.object_id)
        previous = row.object_id
        snapshot = row.snapshot
        if snapshot.erased:
            continue
        cited.append(ForgottenCitation(
            snapshot.entity_id,
            # A version giving someone another name forgets that name, not the name it gives
            # it beside ("Rowan" of "Rowan's nickname is Ollie.", ``kept_beside``).
            frozenset((entry.entity_id, name_norm(entry.name)) for entry in snapshot.naming
                      if not kept_beside(entry.name, snapshot.statement)),
            tuple(tuple(snapshot.quote_spans[index] for index in group)
                  for group in snapshot.support_groups),
            frozenset(snapshot.about_entity_ids),
            bool(snapshot.sensitivity_classes),
            snapshot.statement,
            tuple(snapshot.embedding),
            snapshot.embedding_space,
        ))
        pair = snapshot.entity_id, snapshot.statement_digest
        newest[pair] = row
        pair_sources = sources.setdefault(pair, set())
        for group in snapshot.support_groups:
            for index in group:
                span = snapshot.quote_spans[index]
                pair_sources.add(span.evidence_id)
                spans[(span.evidence_id, span.byte_start, span.byte_end)] = span
        if len(newest) > MAX_FORGET_TOMBSTONES or len(pair_sources) > MAX_TOMBSTONE_SOURCES:
            raise QuoteMemError("budget_exceeded", "Forget exceeds its tombstone admission bound")
    head = ordered[-1]
    if head.object_id != item.current_revision_id or (
        head.snapshot.entity_id, head.snapshot.statement_digest
    ) != (item.entity_id, item.statement_digest):
        raise QuoteMemError("binding_mismatch", "Forget head differs from the selected Item")
    if len({key[0] for key in spans}) > MAX_FORGET_SOURCES:
        raise QuoteMemError("budget_exceeded", "Forget exceeds its withheld source admission bound")
    return ForgetAdmissionSelection(
        versions=tuple(
            ForgetVersionSelection(newest[pair].object_id, newest[pair].snapshot,
                                   tuple(sorted(sources[pair])))
            for pair in sorted(newest)
        ),
        spans=tuple(spans[key] for key in sorted(spans)),
        cited=tuple(cited),
    )


def plan_earlier_wordings(
    revisions: Sequence[RevisionRowV2], verdicts: Set[str],
) -> tuple[ForgetVersionSelection, ...]:
    """The wordings a surviving memory had before a forgotten correction reworded it.

    The memory's versions older than its newest ``rewording`` whose verdict is one of
    ``verdicts`` (the forgotten correction's) carry the description that correction
    replaced, so they join D like the versions of a forgotten memory and a restatement of
    them is refused; the memory itself survives and its head keeps its own quotes.

    Args:
        revisions: Every revision of one surviving memory.
        verdicts: The verdicts of the forgotten memories' versions.

    Returns:
        The newest version of each distinct ``(entity_id, statement_digest)`` pair among
        those earlier versions, with all of that pair's sources; empty when no
        rewording carries one of ``verdicts``.

    Raises:
        QuoteMemError: ``budget_exceeded`` past the tombstone admission bounds.
    """
    ordered = sorted(revisions, key=lambda row: row.version)
    cut = max((row.version for row in ordered if row.reason_code == "rewording"
               and row.snapshot.verdict_ref in verdicts), default=None)
    newest: dict[tuple[str, str], RevisionRowV2] = {}
    sources: dict[tuple[str, str], set[str]] = {}
    for row in ordered:
        if cut is None or row.version >= cut or row.snapshot.erased:
            continue
        pair = row.snapshot.entity_id, row.snapshot.statement_digest
        newest[pair] = row
        sources.setdefault(pair, set()).update(row.evidence_ids)
        if len(newest) > MAX_FORGET_TOMBSTONES or len(sources[pair]) > MAX_TOMBSTONE_SOURCES:
            raise QuoteMemError("budget_exceeded", "Forget exceeds its tombstone admission bound")
    return tuple(ForgetVersionSelection(newest[pair].object_id, newest[pair].snapshot,
                                        tuple(sorted(sources[pair]))) for pair in sorted(newest))


def named_by_type(statement: str, given: Set[str], others: Set[tuple[str, ...]]) -> bool:
    """Whether a naming statement names its entity by type alone, as one of ``given``.

    A bare-name keeper's head must: its type is exactly one allowlisted
    neutral kind, never a modifier, relation, role, occupation, animal or clause.
    It never says "user" nor names
    another anchor of the forgotten versions (a token run of ``others``), so "A person who
    is Avery's spouse is named Rowan." or "A friend of the user is named Rowan." does
    not keep, since it restates "Avery is married to Rowan." or a fact of the user.

    Args:
        statement: The naming head's statement.
        given: The ``name_norm`` of each name the head gives its one anchor.
        others: The ``name_norm`` token runs of "user" and of the other anchors' names.

    Returns:
        True for the bare-name syntax (`bare_naming`: an English neutral kind or an exact
        Chinese classifier kind) with a name of ``given`` and no forbidden kind token run.
    """
    found = bare_naming(statement)
    if found is None or name_norm(found[1]) not in given:
        return False
    kind = token_seq(found[0])
    return not any(
        kind[start:start + len(run)] == list(run)
        for run in others if run for start in range(len(kind) - len(run) + 1)
    )


def bare_name_spans(keeper: KeeperGroup) -> tuple[QuoteSpanV1, ...]:
    """Locate the literal name of a neutral naming head inside its quoted support.

    A broad quote such as "My friend Nora" supports a neutral naming head without
    granting the relationship words readability. Match the complete written name,
    with word boundaries (a Han name after a naming word, "叫小美", or with no other Han
    character beside it but 的), and convert original character offsets to UTF-8 bytes;
    normalized tokens are never used as offsets into source text.

    Args:
        keeper: A group marked bare only after the neutral-kind and sole-anchor
            checks. The head must give exactly the name its naming syntax states.

    Returns:
        Sorted unique name subspans, or none for an unmatched or non-neutral head.
    """
    found = bare_naming(keeper.statement)
    if (keeper.bare in (None, "", "user") or found is None
            or keeper.anchors != (keeper.bare,)
            or keeper.names != frozenset({(keeper.bare, name_norm(found[1]))})
            or len(keeper.quoted) != len(keeper.spans)):
        return ()
    han = not keeper.uncut and own_token(found[1][0]) and own_token(found[1][-1])
    # A Han name stands among Han characters: it is located after a naming word ("叫小美"),
    # at the clause's start, or before 的 or the clause's end, never inside another word.
    pattern = re.compile(
        rf"(?:(?<=叫)|(?<=名为)|(?<=是)|(?<![\u3400-\u9fff])){re.escape(found[1])}"
        r"(?![\u3400-\u4dbf\u4e00-\u9fff](?<!的))" if han
        else r"(?<!\w)" + re.escape(found[1]) + r"(?!\w)")
    located: set[QuoteSpanV1] = set()
    for span, text in zip(keeper.spans, keeper.quoted, strict=True):
        if name_norm(text) == name_norm(found[1]):
            located.add(span)
            continue
        for match in pattern.finditer(text):
            if (match.start() and unicodedata.category(text[match.start() - 1]).startswith("M")
                    or match.end() < len(text)
                    and unicodedata.category(text[match.end()]).startswith("M")):
                continue
            start = span.byte_start + len(text[:match.start()].encode("utf-8"))
            located.add(QuoteSpanV1(
                evidence_id=span.evidence_id, byte_start=start,
                byte_end=start + len(match.group().encode("utf-8")),
            ))
    return tuple(sorted(located, key=lambda span: (
        span.evidence_id, span.byte_start, span.byte_end)))


def _stem(token: str) -> str:
    """A crude English stem, so that "bathes", "bathed" and "bathe" compare equal."""
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


def _stems(words: str) -> frozenset[str]:
    return frozenset(_stem(word) for word in words.split())


_LIGHT = _stems("""
    a an the this that these those some any each every all both either neither such same
    i me my mine myself we us our ours you your yours he him his she her hers they them their
    theirs it its itself who whom whose which what one ones s t re ve ll d m
    is are was were be been being am do does did doing done will would shall should can could
    may might must named called name names nicknamed known goes go by as
    of to in on at for with from into onto about over under after before during until till
    since than then and or but nor so if while because also too just only even still again
    already now currently recently there here not no never longer anymore none
    today yesterday tomorrow tonight morning mornings evening evenings night nights afternoon
    afternoons day days daily week weeks weekly weekend weekends month months monthly year
    years yearly ago last next ever always often sometimes usually soon later early earlier
    monday tuesday wednesday thursday friday saturday sunday january february march april
    june july august september october november december
    new other another second third first older oldest younger youngest elder eldest
    two three four five six seven eight nine ten eleven twelve twenty thirty hundred
""")
"""Words that never carry a fact on their own here: function words, naming verbs, negation,
time and number words, and the set-apart words a naming fact keeps."""

_RELATION = _stems("""
    have has had having own owns owned owning owner owners possess possesses possessed
    adopt adopts adopted adopting adoption buy buys bought buying purchase purchased
    rescue rescued acquire acquired receive received gift gifted get gets got gotten getting
    keep keeps kept foster fostered inherit inherited raise raised breed bred
    marry marries married marrying marriage wed weds wedded wedding spouse husband wife
    partner partners engaged engagement divorce divorced dating date dated girlfriend
    boyfriend fiance fiancee sibling siblings sister sisters brother brothers mother father
    mom dad mum parent parents son sons daughter daughters child children kid kids baby
    babies aunt uncle cousin niece nephew grandmother grandfather grandma grandpa grandparent
    grandparents grandchild grandson granddaughter twin relative relatives family friend
    friends roommate roommates neighbor neighbors neighbour neighbours colleague colleagues
    coworker coworkers boss manager employer employee employees mentor student teacher
    classmate teammate landlord tenant client patient doctor therapist dentist work works
    worked working employ employs employed employment hire hired job jobs intern interned
    internship volunteer volunteers volunteered volunteering use uses used using member
    members belong belongs join joins joined attend attends attended study studies studied
    enroll enrolled
""")
"""Possession, acquisition, kinship, partnership, social, employment, use and membership
words: any of them among what a forgotten version adds over a naming memory makes the
naming memory a possible restatement of it, so the detail keeper keeps nothing."""


_QUOTE_LIGHT = _stems("""
    a an the my mine our ours your his her hers their its i we you he she they it me us him
    them s is are was were named called name nicknamed
""")
"""Words a detail keeper's quote may carry beyond its own statement's: determiners, pronouns,
"is" and naming verbs ("Our dog is called Biscuit", "a tortoise named Pip")."""


def _content(text: str) -> frozenset[str]:
    """The stems of ``text``'s tokens (``token_seq``) that may carry a fact."""
    return frozenset(_stem(token) for token in token_seq(text)
                     if not token.isdigit()) - _LIGHT


_POSSESSORS = re.compile(r"(?<![\w'\u2019])(the user|[A-Z][A-Za-z-]*(?: [A-Z][A-Za-z-]*){0,2})"
                         r"['\u2019]s\b", re.IGNORECASE)
"""Every "<possessor>'s" of a statement: "the user's", or a capitalized name's."""


def _owners(keeper: KeeperGroup) -> list[list[str]]:
    """The token runs naming the owner a naming memory describes its entity by: "user"
    when its head anchors the user, and each possessor of "<possessor>'s" in its
    statement ("Avery" in "Avery's hamster is named Mochi.")."""
    owners = [["user"]] if "user" in keeper.anchors else []
    for match in _POSSESSORS.finditer(keeper.statement):
        words = match.group(1)
        if words.casefold() == "the user":
            owners.append(["user"])
        elif words.isascii() and all(word[:1].isupper() for word in words.split()):
            owners.append(token_seq(words))
    return owners


def _owner_takes_part(statement: str, owners: Sequence[Sequence[str]]) -> bool:
    """Whether ``statement`` names one of ``owners`` other than as "<owner>'s"."""
    tokens = token_seq(statement)
    for owner in owners:
        size = len(owner)
        for start in range(len(tokens) - size + 1):
            after = tokens[start + size:start + size + 1]
            if tokens[start:start + size] == list(owner) and after != ["s"]:
                return True
    return False


def detail_names(version: ForgottenCitation, keeper: KeeperGroup) -> set[tuple[str, str]]:
    """The names a detail keeper keeps for ``version``.

    A naming memory that also anchors its entity's owner or the user ("The user's
    tortoise is named Pip.") keeps the quote that establishes the entity when the
    forgotten version states a different detail of that entity: "The user bathes the
    user's tortoise Pip every morning." It keeps nothing when the version could be a
    restatement of it: "The user has a tortoise." or "Avery has a hamster named
    Mochi." The test is on words and errs toward withholding:

    * the keeper names exactly one entity e, never the user; e and every anchor of the
      keeper's head are anchors of the version;
    * every word of the keeper's statement other than e's name, function, naming, time,
      number or set-apart words, is a word of the version's statement (it describes e
      exactly as the version does and says nothing the version does not);
    * what the version says beyond the keeper is not empty and has no possession,
      acquisition, kinship, partnership, social, employment, use or membership word;
      those word lists are English, so a word beyond the keeper in another script or
      with a diacritic is one the test cannot judge, and the keeper keeps nothing;
    * when e is a person, the version names the keeper's owner (the user, or the
      possessor of "<owner>'s") only as that possessor: a version in which the owner
      takes part ("The user cohabits with the user's roommate Rowan.", "The user rents
      to the user's tenant Sam.") states a relation between the owner and e, which the
      keeper's description of e ("roommate", "tenant") can restate in other words that
      no closed word list catches, so the keeper keeps nothing. A version about
      e itself ("The user's roommate Rowan works as an engineer.") still keeps;
    * every word of every quote of the keeper is a word of the keeper's own statement,
      a determiner, a pronoun, "is" or a naming verb: the bytes left readable state
      nothing the naming memory does not, so no byte of the forgotten detail, and no
      preposition or verb hinting at it ("at St. Mary's Hospital"), stays readable.

    Args:
        version: One non-erased forgotten version, with its statement.
        keeper: One readable group of a surviving naming memory, with its head's
            statement, anchors and quoted text.

    Returns:
        The keeper's names when it keeps its quotes for ``version``; else empty.
    """
    entities = {entity for entity, _ in keeper.names}
    anchors = {version.entity_id, *version.about}
    if len(entities) != 1 or not version.statement or not keeper.statement:
        return set()
    (entity,) = entities
    if (entity in ("", "user") or entity not in anchors or not keeper.anchors
            or not set(keeper.anchors) <= anchors or len(keeper.quoted) != len(keeper.spans)):
        return set()
    named = frozenset(_stem(token) for _, name in keeper.names for token in name.split())
    said = _content(keeper.statement)
    stated = _content(version.statement)
    if not said - named <= stated:
        return set()
    extra = stated - said
    if not extra or extra & _RELATION or not all(word.isascii() for word in extra):
        return set()
    if entity in keeper.persons and _owner_takes_part(version.statement, _owners(keeper)):
        return set()
    allowed = frozenset(_stem(token) for token in token_seq(keeper.statement)) | _QUOTE_LIGHT
    if any(not frozenset(_stem(token) for token in token_seq(text)) <= allowed
           for text in keeper.quoted):
        return set()
    return set(keeper.names)


_POSSESSOR = re.compile(r"(the user|[A-Z][A-Za-z-]*(?: [A-Z][A-Za-z-]*){0,2})['\u2019]s\s+",
                        re.IGNORECASE)
"""A statement's opening possessor: "The user's", or a capitalized English name of at most
three words ("Avery's"), as a statement restates a speaker's "my"."""

_REFERENT_WORDS = 3
"""Description words a referent keeper's quote may hold after "my" (rule 5's bound)."""

_SUBJECT = re.compile(r"([A-Z][A-Za-z-]*(?: [A-Z][A-Za-z-]*){0,2})(?=,? )")
"""A statement's opening subject: one to three capitalized English words before a space
or an appositive comma ("Avery applied ...", "Blair got ...")."""

_OPENERS = frozenset({
    "yesterday", "today", "tomorrow", "tonight", "last", "next", "every", "each", "once",
    "after", "before", "when", "while", "during", "since", "until", "recently", "currently",
    "previously", "now", "then", "later", "earlier", "sometimes", "often", "always", "never",
    "usually", "both", "all", "most", "many", "several", "no", "none", "not", "one", "two",
    "three", "there", "here", "how", "what", "why", "where", "if", "although", "because",
})
"""Capitalized words that open a sentence without naming its subject."""

_HAVE = frozenset({"has", "have", "had", "own", "owns", "owned"})
"""Verbs by which a statement says its subject possesses something ("Blair has pets.")."""

_GAIN = _HAVE | frozenset({
    "got", "gets", "keeps", "kept", "bought", "buys", "adopted", "adopts", "acquired",
    "received", "found",
})
"""Verbs by which a statement says its subject possesses or gains something."""


def _opening(statement: str) -> tuple[str, list[str]] | None:
    """The possessor and the tokens after it when ``statement`` opens with "<possessor>'s"."""
    match = _POSSESSOR.match(statement)
    if match is None or not match.group(1).isascii() or (
        match.group(1).casefold() != "the user"
        and not all(word[:1].isupper() for word in match.group(1).split())
    ):
        return None
    return match.group(1).casefold(), token_seq(statement[match.end():])


def _formal(words: Sequence[str]) -> list[str]:
    return [FORMAL_KINSHIP.get(word, word) for word in words]


def _describes(rest: Sequence[str], description: Sequence[str]) -> bool:
    """Whether ``rest`` opens with ``description`` as a whole noun phrase: the same words
    (an informal kinship word may be formal on either side), then a word other than "s"."""
    size = len(description)
    return (len(rest) > size and rest[size] != "s"
            and _formal(rest[:size]) == _formal(description))


def _said(words: Sequence[str]) -> frozenset[str]:
    """The content stems of ``words``, each word also in its formal kinship spelling."""
    return frozenset(_stem(form) for word in words if not word.isdigit()
                     for form in (word, FORMAL_KINSHIP.get(word, word))) - _LIGHT


def referent_spans(
    version: ForgottenCitation, keeper: KeeperGroup, forgotten: Set[tuple[str, str]],
) -> tuple[QuoteSpanV1, ...]:
    """The quotes a referent keeper keeps readable for ``version``.

    A forgotten sentence can carry the phrase by which another memory names its own
    subject: "My brother" in "My brother is allergic to peanuts. He loves chess." supports
    both the forgotten "The user's brother is allergic to peanuts." and the surviving
    "The user's brother loves chess." (group "My brother", "He loves chess."). The
    survivor keeps that phrase readable, so it stays readable itself, while every word of
    the forgotten predicate is withheld. A quote counts only when all of these hold, and
    the test errs toward withholding:

    * the keeper's head has no sensitivity class and its primary entity is not the user;
    * the quote is "my" and one to three English description words, none of them "s";
    * the head's statement opens with "<possessor>'s" ("The user's" or a capitalized name)
      and exactly that description as a whole noun phrase (an informal kinship word may be
      formal on either side: "My mom" for "The user's mother loves chess.");
    * the quote has no sensitivity class (``classify``) and contains no name a forgotten
      version gives, so "my diabetic brother" or "my brother Tom" (Tom forgotten) keeps
      nothing when the classifier or the names show it;
    * no content stem of the description is a content stem of what the version states
      about it: its words after the same opening when it opens with one, else all of its
      words. "My brother" keeps for "The user's brother is allergic to peanuts." (the
      predicate is "is allergic to peanuts") and nothing for "The user has a brother.",
      "The user's brother's wife is ill." or "The user's brother is a monk's brother.".

    A possession of the same subject also disqualifies the phrase when its entire
    object matches the primary entity's stored broad type: a pet's "My cat" restates
    "The user has pets.". Negation, quantity, time and additional descriptors are not
    discarded by this kind check. It introduces no taxonomy or identity inference.

    A sensitive forgotten version does not by itself disqualify this path, as for path
    (a): only the survivor's own referring phrase stays readable.

    Args:
        version: One non-erased forgotten version, with its statement.
        keeper: One readable group of a surviving memory, with its head's statement,
            anchors and quoted text.
        forgotten: Every ``(entity_id, name_norm)`` pair a forgotten version gives.

    Returns:
        The keeper's quotes that are its referent for ``version``; empty when none is.
    """
    head = _opening(keeper.statement)
    if (head is None or keeper.sensitive or not version.statement or not keeper.anchors
            or keeper.anchors[0] in ("", "user") or len(keeper.quoted) != len(keeper.spans)):
        return ()
    possessor, rest = head
    subject = _subject(version.statement)
    if subject is not None and possession_entailed(version.statement, keeper.statement,
                                                   modifier=version.modifier):
        tokens = token_seq(version.statement)
        possessed = support_words(" ".join(tokens[len(token_seq(subject)) + 1:]))
        if possessed and possessed == support_words(keeper.entity_type):
            return ()
    stated = _opening(version.statement)
    kept: list[QuoteSpanV1] = []
    for span, text in zip(keeper.spans, keeper.quoted, strict=True):
        words = token_seq(text)
        description = words[1:]
        if (not text.isascii() or words[:1] != ["my"] or "s" in description
                or not 1 <= len(description) <= _REFERENT_WORDS
                or not _describes(rest, description) or classify([text])
                or any(token_contained(name, text) for _, name in forgotten)):
            continue
        if stated is not None and stated[0] == possessor and _describes(stated[1], description):
            predicate = stated[1][len(description):]
        else:
            predicate = token_seq(version.statement)
        if _said(description).isdisjoint(_said(predicate)):
            kept.append(span)
    return tuple(kept)


def _ancestor_naming_words(forgotten: str, survivor: str) -> frozenset[str] | None:
    """The descendant and name alone for an exact ancestor-existence keeper.

    ``The user has a sister.`` states neither the dog nor its name in ``The user's
    sister's dog is named Pip.``, nor the husband or his name in ``Gregor is the user's
    sister's husband.`` or ``Gregor is the husband of the user's sister.`` The ancestor
    phrase can therefore keep its own
    source runs, while the descendant and the name remain forbidden. Match both entire
    canonical statements: ordinary possession/existence forgets, ancestor naming and
    predicates outside the matched nominal phrases receive no exemption. Each
    description has one to three ASCII content words and no negation; the terminal
    name has only capitalized ASCII name words.
    """
    head = re.fullmatch(r"The user has (?:a|an) ([a-z]+(?: [a-z]+){0,2})\.", survivor)
    if (head is None or not all(support_words(word) for word in head[1].split())
            or not {"no", "not", "never"}.isdisjoint(head[1].split())):
        return None
    ancestor = re.escape(head[1])
    descendant = r"(?P<descendant>[a-z]+(?: [a-z]+){0,2})"
    name = r"(?P<name>(?:(?:Mrs|Mr|Ms|Dr|Prof)\. )?[A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*)"
    named = next((match for pattern in (
        rf"The user's {ancestor}'s {descendant} is named {name}\.",
        # The same tie said the other way round ("Gregor is the user's sister's husband.").
        rf"{name} is the user's {ancestor}'s {descendant}\.",
        rf"{name} is the {descendant} of the user's {ancestor}\.",
    ) if (match := re.fullmatch(pattern, forgotten)) is not None), None)
    if named is None:
        # The user's words asking that name ("the name of my daughter's teacher"), which a
        # forget of its sentences alone is made under: the descendant and "name" only.
        named = next((match for pattern in (
            rf"(?:please )?(?:forget )?(?:the (?:first |last |full )?name of )my {ancestor}'s "
            rf"{descendant}",
            rf"(?:please )?(?:forget )?my {ancestor}'s {descendant}'s (?:first |last |full )?name",
        ) if (match := re.fullmatch(pattern, forgotten.strip().rstrip(".!?"),
                                    re.IGNORECASE)) is not None), None)
        if named is None or not all(support_words(word)
                                    for word in named["descendant"].split()):
            return None
        return support_words(f"{named['descendant']} name")
    if (not all(support_words(word) for word in named["descendant"].split())
            or not {"no", "not", "never"}.isdisjoint(named["descendant"].split())):
        return None
    return support_words(f"{named['descendant']} {named['name']}")


def _existence_words(forgotten: str, survivor: str, *,
                     modifier: bool = False) -> frozenset[str] | None:
    """What a forgotten statement says of a thing a survivor says the user has (forget5).

    ``The user has a tabby cat.`` states only that the cat exists; ``The user's tabby cat is
    named Pixel.`` says that and its name. Forgetting the name leaves the existence a fact of
    its own (the forget's judge keeps it as the request's ``identity``), so the forbidden
    words are the forgotten predicate after the shared description, "named" and "Pixel",
    and the survivor's clause "I have a tabby cat" stays readable. Both statements must
    match entirely: the survivor "The user has a/an <description>." (one to three ASCII
    content words, no negation), the forgotten one opening "The user's <description>" with
    the same description as a whole noun phrase (``_describes``), or as the possessor of a
    thing of its own ("The user's son's birthday is May 20." for "The user has a son."), and
    saying something after it; or the forgotten one ties a name to a thing of the user's
    thing so described (``_thing_of_owner``: "Mrs. Halvorsen is the user's daughter's
    teacher."), or, for a forget of the ``modifier`` alone (``asks_modifier``), says the user
    has that thing with modifiers ("The user has an orange cat.": only "orange" is
    forbidden). A survivor may say the user has two things ("The user has a cat and also a
    dog."), each read so. Any other pair gets no exemption.
    """
    for description in _existences(survivor):
        found = _existence_of(forgotten, description, modifier=modifier)
        if found is not None:
            return found
    return None


_THING = r"[a-z]+(?: [a-z]+){0,2}"
_EXISTENCES: Final = re.compile(
    rf"The user has (?:a|an) (?P<first>{_THING})"
    rf"(?: and (?:also )?(?:a|an) (?P<second>{_THING}))?(?: too| as well)?\.")
"""An existence survivor, of one thing or two ("The user has a cat and also a dog.")."""


def _existences(survivor: str) -> list[list[str]]:
    """The descriptions of the things an existence survivor says the user has: one to three
    ASCII content words each, no negation; none for any other statement."""
    head = _EXISTENCES.fullmatch(survivor)
    if head is None:
        return []
    found = [token_seq(thing) for thing in (head["first"], head["second"]) if thing]
    if any(not all(support_words(word) for word in words)
           or not {"no", "not", "never"}.isdisjoint(words) for words in found):
        return []
    return found


_MODIFIED: Final = re.compile(rf"The user has (?:a|an) (?P<modifiers>{_THING}) (?P<noun>[a-z]+)\.")
"""A forgotten existence with modifiers ("The user has an orange cat.")."""


_MODIFIER_ASKED: Final = re.compile(
    r"\b(?:colou?rs?|breeds?|kinds?|sorts?|ages?|size)\b|\bhow\s+old\b|\byears?\s+old\b"
    r"|\bmy\s+(?:[a-z]+\s+){0,2}?[a-z]+\s+(?:is|was)\b(?!\s+(?:called|named)\b)"
    r"|颜色|品种|年龄|岁数|几岁|多大|我的?[\u4e00-\u9fff]{1,6}?是(?!叫)", re.IGNORECASE)
"""How a request asks how the user's thing is ("forget what colour my cat is", "forget that my
cat is orange", "忘掉我的猫是什么颜色的", "忘掉我的狗是金毛")."""
_POSSESSION_ASKED: Final = re.compile(
    r"\b(?:have|has|had|own|owns|owned|got)\s+(?:a|an|one|two|some|any|the)\b"
    r"|\b(?:named|called|names?)\b|(?:养|有)(?!的)|叫|名字", re.IGNORECASE)
"""How a request asks that the user has a thing ("forget that I have a cat", "忘掉我养猫这件
事"), or its name: no request for a modifier alone."""


def asks_modifier(requests: Sequence[str]) -> bool:
    """Whether the user's words ask to forget only how a thing of theirs is, a modifier of it
    (``_MODIFIER_ASKED``), never that they have it or its name (``_POSSESSION_ASKED``): only
    then is a forgotten "The user has an orange cat." no possession a survivor about the cat
    restates (``possession_entailed``). Errs toward the possession: unread words ask it."""
    return any(_MODIFIER_ASKED.search(text) is not None
               and _POSSESSION_ASKED.search(text) is None for text in requests)


def _modifiers(forgotten: str, description: Sequence[str]) -> frozenset[str] | None:
    """The modifier words of a forgotten existence of the described thing ("orange" of "The
    user has an orange cat." for "cat"), when they are all it says beyond the thing."""
    match = _MODIFIED.fullmatch(forgotten)
    if match is None or [match["noun"]] != list(description[-1:]) or len(description) != 1:
        return None
    return support_words(match["modifiers"]) or None


def _existence_of(forgotten: str, description: list[str], *,
                  modifier: bool = False) -> frozenset[str] | None:
    """`_existence_words` for one described thing of the survivor's."""
    modified = _modifiers(forgotten, description) if modifier else None
    if modified is not None:
        return modified
    tied = _thing_of_owner(forgotten)
    if tied is not None:
        # "Mrs. Halvorsen is the user's daughter's teacher.": a thing of the user's daughter.
        return tied[1] if _formal(tied[0]) == _formal(description) else None
    opening = _opening(forgotten)
    if opening is None or opening[0] != "the user":
        return None
    size = len(description)
    if opening[1][:size + 1] == [*description, "s"]:
        size += 1  # "The user's son's birthday is May 20.": a thing of the user's son
    elif not _describes(opening[1], description):
        return None
    rest = support_words(" ".join(opening[1][size:]))
    return rest or None


_THINGS_OF_OWNERS: Final = (
    re.compile(r"(?P<name>.+?) (?:is|was) the user's (?P<owner>[a-z]+(?: [a-z]+){0,2})'s "
               r"(?P<thing>[a-z]+(?: [a-z]+){0,2})\."),
    re.compile(r"(?P<name>.+?) (?:is|was) the (?P<thing>[a-z]+(?: [a-z]+){0,2}) of the user's "
               r"(?P<owner>[a-z]+(?: [a-z]+){0,2})\."),
)
"""A statement tying a name to a thing of something of the user's: "Mrs. Halvorsen is the
user's daughter's teacher.", "Gregor is the husband of the user's sister."."""


_TIED_OWNER: Final = re.compile(r"[^']+? (?:is|was) the user's (?P<owner>[a-z]+(?: [a-z]+){0,2})\.")
"""A statement tying a name to a thing of the user's ("Ivy is the user's daughter.")."""


def _thing_of_owner(statement: str) -> tuple[list[str], frozenset[str]] | None:
    """The owner's description and the words of the name and the thing of a statement tying a
    name to a thing of something of the user's (``_THINGS_OF_OWNERS``): (["daughter"],
    {"mrs", "halvorsen", "teacher"}) of "Mrs. Halvorsen is the user's daughter's teacher.";
    None for any other statement."""
    for pattern in _THINGS_OF_OWNERS:
        match = pattern.fullmatch(statement.strip())
        if match is not None and "'s" not in match["name"] and all(
                support_words(word) for word in match["owner"].split()):
            return token_seq(match["owner"]), support_words(f"{match['name']} {match['thing']}")
    return None


_OWNED_THING: Final = re.compile(r"The user has (?:a|an) ([a-z]+(?: [a-z]+){0,2})\.")
"""An existence survivor and its description ("The user has a younger brother.")."""


def _lead_name(statement: str) -> str | None:
    """The named subject a statement opens with, casefolded: its possessor ("Blair's pet
    is named Pip.") or its subject ("Avery applied to adoption agencies."); None for
    "The user", a sentence-opening function or time word, and any other opening."""
    opening = _opening(statement)
    if opening is not None:
        return None if opening[0] == "the user" else opening[0]
    match = _SUBJECT.match(statement)
    if (match is None or not match.group(1).isascii() or not support_words(match.group(1))
            or match.group(1).split()[0].casefold() in _OPENERS):
        return None
    return match.group(1).casefold()


def _verb_after(statement: str, lead: str) -> str | None:
    """The token after a statement's lead name ``lead``."""
    tokens, size = token_seq(statement), len(token_seq(lead))
    return tokens[size] if len(tokens) > size else None


_NAME_TOKEN: Final = re.compile(r"[A-Za-z][\w'\u2019-]*|[^\W\dA-Za-z_]+")
"""A word of a statement as ``_other_subject`` compares them: an English word, or a run of
other letters ("小王")."""


def _other_subject(forgotten: str, survivor: str) -> frozenset[str] | None:
    """What only the forgotten name has when both statements say exactly the same of two
    different names: "Ben is the user's friend." and "Alice is the user's friend.", "A person
    is named 小李." and "A person is named 小王.". Their words (``_NAME_TOKEN``) differ only in
    one run of at most three names on each side, capitalized English words or words in
    another script, and the words they share carry an English word of their own. The
    survivor's fact is its own; only whom the forgotten one names is forgotten, less the
    words both names share: "小李" beside "小王" forbids "李" alone, as both have "小" (the
    clause cut still compares whole terms, ``fact_terms``, so no kept clause shows "小李").
    None for any other pair.
    """
    mine, theirs = _NAME_TOKEN.findall(forgotten), _NAME_TOKEN.findall(survivor)
    head = 0
    while head < min(len(mine), len(theirs)) and mine[head].casefold() == theirs[head].casefold():
        head += 1
    tail = 0
    while (tail < min(len(mine), len(theirs)) - head
           and mine[-1 - tail].casefold() == theirs[-1 - tail].casefold()):
        tail += 1
    names = (mine[head:len(mine) - tail], theirs[head:len(theirs) - tail])
    shared = [word for word in (*mine[:head], *mine[len(mine) - tail:]) if word.isascii()]
    if not all(1 <= len(run) <= _REFERENT_WORDS
               and all(word[0].isupper() or not word.isascii() for word in run)
               for run in names) or not support_words(" ".join(shared)):
        return None
    return support_words(" ".join(names[0])) - support_words(" ".join(names[1])) or None


_APPOSITIVE: Final = re.compile(
    r"The user's (?:[a-z]+ ){1,3}(?P<name>[A-Z][A-Za-z-]*(?: [A-Z][A-Za-z-]*){0,2}) [a-z]")
"""A statement about the user's someone named in apposition ("The user's son Rowan loves
dinosaurs.")."""


_HAN_LEAD: Final = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff]{1,8}) (?=[a-z])")
"""A statement opening with a Han name as its subject ("小美 is the user's daughter.")."""


def _shared_lead(forgotten: str, survivor: str, *, modifier: bool = False) -> str | None:
    """The named subject both statements are about (``_lead_name``, or the name the survivor
    gives the user's someone in apposition, ``_APPOSITIVE``: "Rowan" of "The user's son
    Rowan loves dinosaurs." beside "Rowan's nickname is Ollie."), or the survivor's named
    subject that the forgotten statement gives as a possessor ("Leo" of "Leo is in fourth
    grade." in "Willowbrook Primary is Leo's school."), unless the survivor may restate a
    forgotten possession (``possession_entailed``)."""
    lead = _lead_name(survivor)
    if lead is None and (named := _APPOSITIVE.match(survivor)) is not None:
        lead = named["name"].casefold()
    han = _HAN_LEAD.match(survivor) if lead is None else None
    if han is not None and not possession_entailed(forgotten, survivor, modifier=modifier) and (
            (other := _HAN_LEAD.match(forgotten)) is not None and other[1] == han[1]
            or re.search(rf"(?<![\w'\u2019]){re.escape(han[1])}['\u2019]s\b", forgotten)):
        # A Han name the survivor opens with, which the forgotten statement opens with too or
        # gives as a possessor ("小美" of "小美 is the user's daughter." beside "小美 is three
        # years old."): ``_lead_name`` reads Latin names only.
        return han[1]
    if lead is None or possession_entailed(forgotten, survivor, modifier=modifier) or (
            lead != _lead_name(forgotten) and re.search(
                rf"(?<![\w'\u2019]){re.escape(lead)}['\u2019]s\b", forgotten,
                re.IGNORECASE) is None):
        return None
    return lead


def _subject(statement: str) -> str | None:
    """The subject a statement opens with: "the user" for "The user ..." (not "The
    user's"), else its named lead (``_lead_name``)."""
    if _opening(statement) is None and [
            token.casefold() for token in token_seq(statement)[:2]] == ["the", "user"]:
        return "the user"
    return _lead_name(statement)


def possession_entailed(forgotten: str, survivor: str, *, modifier: bool = False) -> bool:
    """Whether a survivor may restate a forgotten possession by one of its things.

    The forgotten statement says its subject has something: a ``_HAVE`` verb after "The
    user" or a named subject ("The user has pets.", "Blair has pets."). The survivor
    states a possession of the same subject: it opens with "<subject>'s" or goes on with a
    ``_GAIN`` verb ("The user got another cat.", "Blair's cat is named Juno."). Its
    thing may be of the forgotten kind, which no word test sees, so such a survivor keeps
    nothing on the clause path (e) or the dependent trim, whatever its words,
    as a named subject's survivor already did through ``_shared_lead``. The referent
    keeper (d) applies this guard only with an exact match between the possessed
    object and its primary entity's stored broad type.

    A forget of the ``modifier`` alone (``asks_modifier``: "忘掉我的猫是橘猫") is no forget
    of the possession: a survivor about the thing whose modifiers are all the forgotten
    statement says beyond it ("The user's cat is named bob." beside "The user has an orange
    cat.") restates none of it, and only the modifier is forbidden (``predicate_words``).

    Args:
        forgotten: A forgotten version's statement.
        survivor: A surviving memory's statement.
        modifier: Whether the forget is of the forgotten statement's modifiers alone.

    Returns:
        True when the survivor may restate the forgotten possession.
    """
    subject = _subject(forgotten)
    if subject is None or _opening(forgotten) is not None or (
            _verb_after(forgotten, subject) not in _HAVE):
        return False
    opening = _opening(survivor)
    if modifier and any(_modifiers(forgotten, description) is not None for description in (
            _existences(survivor) or ([_lead_description(opening[1])]
                                      if opening is not None and opening[0] == "the user"
                                      else []))):
        # The forgotten possession says only how the survivor's own thing is ("The user has
        # an orange cat." beside "The user's cat is named bob."): its modifier is forbidden.
        return False
    if opening is not None:
        return opening[0] == subject
    return _subject(survivor) == subject and _verb_after(survivor, subject) in _GAIN


_WORK = frozenset({"work", "works", "worked", "working"})
"""The forms of "work" whose frame says which aspect of a job a statement gives."""


def _work_aspect(statement: str, subject: str) -> str | None:
    """"role" for "<subject> works as ...", "place" for "<subject> works at ..." or "works
    for ...", else None."""
    if _verb_after(statement, subject) not in _WORK:
        return None
    tokens, size = token_seq(statement), len(token_seq(subject))
    frame = tokens[size + 1].casefold() if len(tokens) > size + 1 else None
    return "role" if frame == "as" else "place" if frame in ("at", "for") else None


_SHARED_TIME: Final = re.compile(
    r"\b(?:(?:last|this|next)\s+(?:week|weekend|month|year|night|summer|winter|spring|fall"
    r"|autumn)|yesterday|today|tonight|\w+\s+(?:days|weeks|months|years)\s+ago)\b",
    re.IGNORECASE)
"""A relative time a statement closes with ("last week", "two years ago")."""


def predicate_words(forgotten: str, survivor: str, *, existence: bool = False,
                    modifier: bool = False) -> frozenset[str]:
    """`_predicate_words`, less a time both statements give the same subject's facts ("last
    week" of "The user left Northwind last week." beside "The user joined Fabrikam last
    week."): when each closes with it, it dates the survivor's own fact, as the shared subject
    names its own subject, and is no forgotten word of it."""
    words = _predicate_words(forgotten, survivor, existence=existence, modifier=modifier)
    mine = _SHARED_TIME.search(survivor)
    theirs = _SHARED_TIME.search(forgotten)
    if (mine is None or theirs is None or mine.group().casefold() != theirs.group().casefold()
            or survivor[mine.end():].strip(" .") or forgotten[theirs.end():].strip(" .")
            or _subject(forgotten) is None or _subject(forgotten) != _subject(survivor)):
        return words
    return words - support_words(mine.group())


def _predicate_words(forgotten: str, survivor: str, *, existence: bool = False,
                     modifier: bool = False) -> frozenset[str]:
    """``support_words`` of what a forgotten statement says about a survivor's subject.

    When both statements open with the same "<possessor>'s <description>" ("The user's
    brother is allergic to peanuts." and "The user's brother loves chess."), the shared
    description is the survivor's own subject, as a referent keeper's "my" phrase is
    (path d): only the words after it, here "allergic" and "peanut", are the forgotten
    predicate. The description is the longest shared run of at most three words before a
    light word, and must describe the forgotten opening as a whole noun phrase. Otherwise
    every word of the forgotten statement counts. An exact ancestor-existence keeper
    also excludes only its own ancestor phrase from an exact descendant naming:
    ``The user has a sister.`` against ``The user's sister's dog is named Pip.``
    leaves ``dog`` and ``Pip`` forbidden (``_ancestor_naming_words``). This
    exemption changes neither ordinary possession nor the ancestor's own naming.

    When both statements are about the same named subject, as possessor or subject
    (``_lead_name``: "Blair's other cat is named Juno." and "Blair's pet is named
    Pip."; "Avery applied to adoption agencies." and "Avery has not adopted a
    child yet."), that name is the survivor's own subject, as "the user" is a light
    word for the user's own facts: it is not a forgotten word, and every other word of
    the forgotten statement is still a forgotten word. The same exception covers
    the survivor's named subject when the forgotten statement gives
    it as a possessor ("Leo is in fourth grade." beside a
    forgotten "Willowbrook Primary is Leo's school."). Not when the forgotten statement is
    a possession and the survivor states one of the same subject (``_shared_lead``).

    When both statements say the same of two names ("Ben is the user's friend." and "Alice
    is the user's friend.", "小李 is the user's colleague." and "小王 is the user's
    colleague.", ``_other_subject``), only the forgotten name's words are forgotten: two
    people named in one sentence ("My friends Alice and Ben") keep the other one's facts. A
    bare naming ("A person is named 小李.", ``bare_naming``) says nothing but the name. When
    the forgotten statement is about a thing of the survivor's subject ("The user's son's
    birthday is May 20." for "The user's son is six years old."), that subject is the
    survivor's own, as for a shared description: only the words after "<description>'s"
    are forgotten; so when it ties a name to such a thing ("Mrs. Halvorsen is the user's
    daughter's teacher.", ``_thing_of_owner``): only the name and the thing are.

    When the forgotten statement names the user's someone in apposition and the survivor
    shows that name ("The user's daughter 小美 is three years old." and "小美 is the user's
    daughter.", ``_named_thing``), only what it says after the name is forgotten.

    When the same subject works in both and one statement gives the job's role ("works
    as an engineer") and the other its place ("works at Acme Corp", "works for Acme
    Corp"), the shared "work" is the survivor's own verb: a role and a place
    entail nothing of each other that their words do not show. Any other frame ("works
    with children", "works in France") keeps "work" forbidden, because a role or a place
    can entail it ("a pediatrician", "the mayor of Paris").

    Given ``existence`` (a complete forget, ``explicit-request-v2``), a survivor saying only
    that the user has the thing the forgotten statement is about keeps its own clause: only
    the forgotten predicate after their shared description is forbidden
    (``_existence_words``). The judge of such a forget decided the survivor states no
    part of the request, so the forget changes only its targets.

    Args:
        forgotten: A forgotten version's statement.
        survivor: A surviving memory's statement.
        existence: Whether the existence exemption applies (complete forgets only).

    Returns:
        The words a survivor of ``forgotten`` must not show or keep readable.
    """
    ancestor = _ancestor_naming_words(forgotten, survivor)
    if ancestor is not None:
        return ancestor
    if existence and (kept := _existence_words(forgotten, survivor,
                                               modifier=modifier)) is not None:
        return kept
    other = _other_subject(forgotten, survivor)
    if other is not None:
        return other
    named = bare_naming(forgotten)
    if named is not None:
        # "A person is named 小李." says of nothing else but the name.
        return support_words(named[1])
    apposed = _named_thing(forgotten, survivor)
    if apposed is not None:
        # The survivor shows the name the forgotten statement gives the user's someone in
        # apposition, so that someone is its own subject ("小美" of "The user's daughter 小美 is
        # three years old." for "小美 is the user's daughter."): only what is said after the
        # name is forgotten.
        return support_words(apposed["rest"])
    lead = _shared_lead(forgotten, survivor, modifier=modifier)
    words = support_words(forgotten) - (support_words(lead) if lead is not None else frozenset())
    subject = _subject(forgotten)
    if subject is not None and subject == _subject(survivor) and {
            _work_aspect(forgotten, subject), _work_aspect(survivor, subject)} == {"role", "place"}:
        words -= support_words("work")
    head, stated = _opening(survivor), _opening(forgotten)
    tied = _thing_of_owner(forgotten)
    if tied is not None and ((head is not None and head[0] == "the user" and (
            _describes(head[1], tied[0]) or head[1][:len(tied[0]) + 1] == [*tied[0], "s"]))
            or (owner := _TIED_OWNER.fullmatch(survivor)) is not None
            and _formal(token_seq(owner["owner"])) == _formal(tied[0])):
        # The survivor is about the user's thing whose thing the forgotten statement names
        # ("The user's daughter is named Ivy." or "Ivy is the user's daughter." beside a
        # forgotten "Mrs. Halvorsen is the user's daughter's teacher."): the owner is the
        # survivor's own subject.
        return tied[1]
    if modifier and head is not None and head[0] == "the user" and (
            modified := _modifiers(forgotten, _lead_description(head[1]))) is not None:
        # "The user's cat is named bob." beside a forgotten "The user has an orange cat.", when
        # the forget is of the modifier alone (``asks_modifier``).
        return modified
    if head is None or stated is None or head[0] != stated[0]:
        return words
    description: list[str] = []
    for mine, theirs in zip(head[1], stated[1], strict=False):
        if (mine != theirs or not support_words(mine)
                or len(description) == _REFERENT_WORDS):
            break
        description.append(mine)
    if description and stated[1][len(description):len(description) + 1] == ["s"]:
        # A possessive chain: the forgotten statement is about a thing of the survivor's
        # subject ("The user's son's birthday is May 20." for "The user's son is six years
        # old."), which is the survivor's own subject still.
        return support_words(" ".join(stated[1][len(description) + 1:]))
    if not description or not _describes(stated[1], description):
        return words
    return support_words(" ".join(stated[1][len(description):]))


_NAMED_THING: Final = re.compile(
    r"The user's (?P<description>[a-z]+(?: [a-z]+){0,2}) "
    r"(?P<name>[A-Z][\w'-]*(?: [A-Z][\w'-]*){0,2}|[\u4e00-\u9fff]+) (?P<rest>.+)")
"""A statement about the user's someone named in apposition ("The user's daughter 小美 is
three years old.")."""


def _named_thing(forgotten: str, survivor: str) -> re.Match[str] | None:
    """The parts of a forgotten statement about the user's someone named in apposition when
    the survivor shows that name too (``_NAMED_THING``), else None."""
    match = _NAMED_THING.fullmatch(forgotten)
    if match is None or not all(support_words(word) for word in match["description"].split()):
        return None
    return match if token_contained(match["name"], survivor) else None


def _lead_description(rest: Sequence[str]) -> list[str]:
    """The one to three content words a statement's tokens after "The user's" open with,
    before a light word ("son" of "son is six years old"); none when there are none."""
    for size in range(1, _REFERENT_WORDS + 1):
        if len(rest) > size and not support_words(rest[size]) and all(
                support_words(word) for word in rest[:size]):
            return list(rest[:size])
    return []


def forgotten_keywords(
    statement: str, keywords: Sequence[str], forgotten: Sequence[str],
    quotes: Sequence[str] = (),
) -> tuple[str, ...]:
    """The keywords a version drops for forgotten statements.

    A keyword is Search's and nomination's term for the memory, never shown to a reader,
    but a query for the forgotten word would still find the memory through it. A keyword
    is dropped when it carries a word of what a forgotten statement says about the
    version's subject (``predicate_words``) that nothing returned for the version shows:
    neither its statement nor one of its readable ``quotes``. A keyword whose word the
    version's own content already shows reveals nothing new and stays. "Maya" leaves the
    keywords of "The user has a sister." after "A person is named Maya." is forgotten
    and "birthday" those of the party plan after "The user's birthday is March 3." is.

    Args:
        statement: The version's statement.
        keywords: Its keywords, in stored order.
        forgotten: Forgotten statements: the forget's tombstones, the scope's at Add.
        quotes: The text of the version's readable quotes.

    Returns:
        The dropped keywords, in stored order; empty when it keeps them all.
    """
    words = frozenset[str]().union(*(predicate_words(text, statement) for text in forgotten))
    hidden = words - support_words(statement).union(*(support_words(text) for text in quotes))
    return tuple(keyword for keyword in keywords if support_words(keyword) & hidden)


CLAUSE_COSINE = 0.85
"""A survivor whose statement vector is this close to a forgotten version's keeps nothing
on path (e): a restatement in other words, which no word test sees. A 59-text
measurement put every restatement at cosine 0.878 or above; the survivors path (e) keeps
in the saved gates score 0.22-0.48 against their forgotten statements, the chess fact
0.64."""


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """The cosine of two vectors; 1.0, the refusing value, when one is missing or zero."""
    if not left or len(left) != len(right):
        return 1.0
    norm = math.sqrt(math.fsum(value * value for value in left)
                     * math.fsum(value * value for value in right))
    return 1.0 if norm == 0 else math.fsum(a * b for a, b in zip(left, right, strict=True)) / norm


def names_another(forgotten: str, survivor: str) -> bool:
    """Whether a forgotten statement only names someone (``naming_value``) whom a survivor's
    statement does not name: "A person is named 小李." and "A person is named 小王." lie close
    by vector, yet the second restates nothing of the first.

    Args:
        forgotten: A forgotten statement.
        survivor: A surviving memory's statement.

    Returns:
        True for such a pair.
    """
    name = naming_value(forgotten)
    return name is not None and not token_contained(name, survivor)


def said_terms(forgotten: str, survivor: str, *, modifier: bool = False) -> frozenset[str]:
    """What a forgotten statement says of a survivor's subject as the clause cut compares it
    in any script: the words of other scripts of ``predicate_words`` (complete forget), and
    the forgotten statement's Han terms (``fact_terms``) all of whose characters it says,
    so a name the survivor shows as its own subject is none ("小美" of "The user's daughter
    小美 is three years old." for "小美 is the user's daughter.")."""
    said = predicate_words(forgotten, survivor, existence=True, modifier=modifier)
    han = {word for word in said if own_token(word)}
    return frozenset({word for word in said if not own_token(word)} | {
        term for term in fact_terms(forgotten) if own_token(term[0]) and set(term) <= han})


def restated_by_vector(vector: Sequence[float], space: str,
                       stones: Sequence[tuple[Sequence[float], str]]) -> bool:
    """Whether a statement vector lies within ``CLAUSE_COSINE`` of a forgotten statement's in
    the same embedding space (``stones``: vector and space each): a restatement in other words
    or another language, which no word test sees."""
    return any(other_space == space and _cosine(vector, other) >= CLAUSE_COSINE
               for other, other_space in stones)


def _owned_description(forgotten: str, survivor: str, *, modifier: bool = False) -> str | None:
    """The survivor's description of the user's own thing it is about, by which a Chinese
    quote names it with a bare noun (``referent_prefixes``): the thing an existence survivor
    says the user has ("cat" of "The user has a cat.", when only the forgotten predicate after
    their shared description is forbidden, ``_existence_words``), or the one to three words
    after "The user's" before a light word ("son" of "The user's son is six years old this
    year."). None for any other survivor."""
    apposed = _named_thing(forgotten, survivor)
    if apposed is not None:
        # The survivor names the someone the forgotten statement describes ("小美 is the
        # user's daughter.", "A person is named 小美."): the description is of that someone.
        return apposed["description"]
    owned = _OWNED_THING.fullmatch(survivor)
    if owned is not None:
        return owned[1] if _existence_words(forgotten, survivor,
                                            modifier=modifier) is not None else None
    tied = _TIED_OWNER.fullmatch(survivor)
    if tied is not None and _HAN_LEAD.match(survivor) is not None and (
            _shared_lead(forgotten, survivor, modifier=modifier) is not None):
        # "小美 is the user's daughter." beside a forgotten "小美 is three years old.": the
        # tie describes the subject both are about.
        return tied["owner"]
    opening = _opening(survivor)
    if opening is None or opening[0] != "the user":
        return None
    return " ".join(_lead_description(opening[1])) or None


def _meets(span: QuoteSpanV1, other: QuoteSpanV1) -> bool:
    return (span.evidence_id == other.evidence_id and span.byte_start < other.byte_end
            and other.byte_start < span.byte_end)


def clause_spans(
    version: ForgottenCitation, keeper: KeeperGroup, forgotten: Set[tuple[str, str]], *,
    any_script: bool = False, quoted: Callable[[QuoteSpanV1], str | None] | None = None,
) -> tuple[QuoteSpanV1, ...]:
    """The parts of a survivor's quotes a clause keeper keeps readable.

    One quote can state two facts: "I volunteer at the animal shelter on weekends and I
    run in the mornings" can support both "The user volunteers at the animal shelter on
    weekends." and "The user runs in the mornings.". Forgetting the first then
    withholds the whole sentence and the second loses its only support. The survivor
    keeps readable the parts of its quotes that carry only its own words, "I run in the
    mornings", and the rest of the forgotten quote is withheld. The test is on words and
    errs toward withholding; a part counts only when all of these hold:

    * the survivor's head has no sensitivity class (the caller also requires that no
      forgotten version has one), and its statement vector, in the version's embedding
      space, is below cosine ``CLAUSE_COSINE`` to the version's: a restatement in other
      words ("never married" for "single") keeps nothing; but over a quote in another
      script, a version that only names someone the survivor does not name restates nothing
      of it (``names_another``: "A person is named 小王." beside a forgotten "A person is
      named 小李.");
    * the quote meets a span the version cites, and is ASCII: a quote in another script
      is never cut, unless ``any_script``;
    * the part is a run of the quote's tokens each of which is a word of the survivor's
      statement or time expression or a light word (``word_runs``): it says nothing the
      survivor does not. Given ``any_script`` (a complete forget, ``explicit-request-v2``),
      a quote in another script or mixed is cut at its clauses instead, and a part is a
      clause all of whose words are the survivor's (``clause_runs``): "我姐姐住在柏林" in
      "我每天早上喝绿茶，我姐姐住在柏林" for "用户的姐姐住在柏林。". Words are then read by
      ``clause_words`` throughout ("用户" and the Han light characters are none), which
      reads an ASCII text as ``support_words`` does. When the survivor's and the version's
      statements are both English (``english``), no Han clause shares their words, so the
      quote keeps what is left once the forgotten detail is withheld by its own words
      (``foreign_rest``), and only the survivor's Han words must be held by the parts. When
      the detail is not found so, the quote keeps only the user's bare noun for the
      survivor's subject (``referent_prefixes``, ``_owned_description``: "我儿子" of
      "我儿子的生日是五月二十号" for "The user's son is six years old this year.") and its
      clauses outside every span the version cites;
    * no word of the part, and no word of the survivor's statement or time expression,
      is a word of what the version states about the survivor's subject
      (``predicate_words``), so no forgotten word stays readable; a keyword carrying
      one is dropped when the survivor is requoted, so none is shown.
      Given ``any_script``, Han, Kana and Hangul text is compared by its terms, not its
      characters: the part and the survivor's statement show their ``shown_terms``, the
      version's statement forbids its ``fact_terms`` (pairs of adjacent characters), so
      "我姐姐住在上海" keeps readable though "我每天早上喝绿茶" shares its 上. The time
      expression's Han characters, known as words only, are still compared one by one.
      A clause's character pairs must also all be the survivor's (``clause_runs``):
      "我辞职了" is made of characters of "用户在柏林的职位是辞典编辑。" but says 辞职, which
      the survivor does not, so it is withheld even against "The user quit their job.",
      whose words no Han text shares;
    * the part has no sensitivity class (``classify``) and contains no name a forgotten
      version gives;
    * the parts and the survivor's quotes outside the version's spans, with the names of
      their speakers (a quote's readable provenance), hold every word of its statement and
      time expression, so the survivor can be requoted on them; otherwise it keeps nothing.

    A part never carries a word of the forgotten statement. A survivor that restated the
    forgotten fact in other words below the cosine bound would keep its own words, as settlement
    keeps a duplicate whose support lies in readable bytes: it is a separate memory the
    judge did not merge.

    Args:
        version: One non-erased forgotten version, with its groups and statement.
        keeper: One readable group of a surviving memory, with its words.
        forgotten: Every ``(entity_id, name_norm)`` pair a forgotten version gives.
        any_script: Whether a quote in another script, or mixed, is cut at its clauses.
        quoted: The text a span of the version quotes, or None when it cannot be read; a
            Han quote under English statements finds the forgotten detail in them too
            (``foreign_rest``'s ``elsewhere``).

    Returns:
        The kept parts as spans inside the keeper's quotes; empty when it keeps nothing.
    """
    if (keeper.sensitive or not version.statement or not keeper.statement
            or not keeper.stated or len(keeper.quoted) != len(keeper.spans)
            or not keeper.space or keeper.space != version.space
            or (_cosine(keeper.vector, version.vector) >= CLAUSE_COSINE
                and not (any_script and not all(text.isascii() for text in keeper.quoted)
                         and names_another(version.statement, keeper.statement)))):
        return ()
    cited = [span for group in version.groups for span in group]
    said = predicate_words(version.statement, keeper.statement, existence=any_script,
                           modifier=version.modifier)
    stated = keeper.stated
    shows = stated
    read_words = shown = support_words
    if any_script:
        # "用户" and the Han light characters are no words in any script (``clause_words``);
        # the time expression's words stay as they are. A Han text is compared with the
        # forgotten statement by its terms (``fact_terms``, ``shown_terms``), not by its
        # characters, so a shared 上 of 早上 and 上海 is no shared word.
        read_words, shown = clause_words, shown_terms
        said = said_terms(version.statement, keeper.statement, modifier=version.modifier)
        time = stated - support_words(keeper.statement)
        stated = (stated & clause_words(keeper.statement)) | time
        shows = shown_terms(keeper.statement) | time
        # The time expression is known by its words alone: its Han characters are still
        # compared with the forgotten statement's one by one.
        if {word for word in time if own_token(word)} & clause_words(version.statement):
            return ()
    # English statements over a Han or mixed quote share no word with it: the forgotten
    # detail is found by the names both copy (``foreign_rest``), and only the survivor's own
    # Han words must stay readable.
    # A version that only names someone forgets the name alone in any language ("名叫欧文的人
    # 小名叫小欧" is read as "A person named 欧文 is nicknamed 小欧.").
    crossed = any_script and not keeper.uncut and english(keeper.statement) and (
        english(version.statement) or naming_value(version.statement) is not None)
    name = naming_value(version.statement) if crossed else None
    if name is not None:
        # A statement that only names someone forgets the name alone ("A person is named 小
        # 李."): its frame words ("person") are no forgotten word of a survivor naming another
        # ("A person is named 小王.").
        said = frozenset(word for word in said if word in support_words(name)
                         or word in fact_terms(name))
    if shows & said or possession_entailed(version.statement, keeper.statement,
                                           modifier=version.modifier):
        return ()
    referent = _owned_description(version.statement, keeper.statement,
                                  modifier=version.modifier) if crossed else None
    elsewhere = [text for span in cited if quoted is not None
                 and (text := quoted(span)) is not None] if crossed else []
    needed = frozenset(word for word in stated if own_token(word)) if crossed else stated
    kept: list[QuoteSpanV1] = []
    grounded: set[str] = set()
    shown_texts: list[str] = []
    speakers = (keeper.speakers if len(keeper.speakers) == len(keeper.spans)
                else (frozenset[str](),) * len(keeper.spans))
    for span, text, speaker in zip(keeper.spans, keeper.quoted, speakers, strict=True):
        if not any(_meets(span, other) for other in cited):
            grounded |= read_words(text) | speaker
            shown_texts.append(text)
            continue
        if text.isascii():
            runs = list(word_runs(text, stated))
        elif any_script:
            encoded = text.encode("utf-8")
            found = (foreign_rest(text, said, version.statement, elsewhere)
                     if crossed else clause_runs(text, stated, shows))
            if not found and crossed:
                # The detail is not found by its own words: the survivor keeps the user's bare
                # noun for the thing it is about ("我儿子" of "我儿子的生日是五月二十号"), and the
                # clauses of its quote the version does not cite, which no forget withholds.
                # When the detail is a modifier of that thing written before its noun ("三岁"
                # of "我的三岁女儿叫小美"), the noun and the naming after it stay too.
                found = [*(referent_prefixes(text, referent) if referent is not None else ()),
                         *(modifier_rest(text, referent, keeper.statement)
                           if referent is not None and not support_words(referent) & said
                           else ()),
                         # A clause saying only the survivor's own words, though cited ("叫小美"
                         # quoted apart from "我的三岁女儿").
                         *own_clauses(text, keeper.statement, referent),
                         *((start, end) for start, end in clause_pieces(text) if not any(
                             _meets(QuoteSpanV1(evidence_id=span.evidence_id,
                                                byte_start=span.byte_start + start,
                                                byte_end=span.byte_start + end), other)
                             for other in cited))]
            runs = [(len(encoded[:start].decode("utf-8")), len(encoded[:end].decode("utf-8")))
                    for start, end in sorted(set(found))]
        else:
            return ()
        for start, end in runs:
            part = text[start:end]
            if (shown(part) & said or classify([part])
                    or any(token_contained(name, part) for _, name in forgotten)):
                continue
            kept.append(QuoteSpanV1(evidence_id=span.evidence_id,
                                    byte_start=span.byte_start + len(text[:start].encode()),
                                    byte_end=span.byte_start + len(text[:end].encode())))
            grounded |= read_words(part) | speaker
            shown_texts.append(part)
    if not kept or not needed <= grounded or (
            crossed and unshown_names(keeper.statement, shown_texts)):
        # Across scripts a name the survivor gives that its readable text does not show may
        # be the forgotten value transliterated or translated ("Shiyan Primary School").
        return ()
    return tuple(kept)


_Range = tuple[int, int]


def _subtract(span: _Range, kept: Sequence[_Range]) -> list[_Range]:
    """The non-empty pieces of ``span`` outside every range of ``kept``."""
    pieces = [span]
    for start, end in kept:
        pieces = [piece for low, high in pieces for piece in (
            ((low, high),) if end <= low or high <= start
            else tuple(part for part in ((low, start), (end, high)) if part[0] < part[1])
        )]
    return pieces


_DESCRIPTOR_STATEMENT = re.compile(
    r"(?P<name>[^,.;:]+?) (?:is|was) (?P<descriptor>[^,;:]+)\.",
)
"""A non-naming head that describes its named primary entity (detail path c)."""


def _reference_detail_names(
    version: ForgottenCitation, keeper: KeeperGroup,
) -> set[tuple[str, str]]:
    """Apply the detail guard to an exact description of a readably named primary.

    After naming/restatement decomposition, ``Pip is the user's sister.`` retains
    ``My sister Pip`` for a different detail just as the former naming descriptor did.
    Its known name is used only by the detail guard, never by the own-name path.
    """
    if keeper.names or not keeper.anchors or keeper.anchors[0] in ("", "user"):
        return set()
    match = _DESCRIPTOR_STATEMENT.fullmatch(keeper.statement)
    if match is None:
        return set()
    reference = (keeper.anchors[0], name_norm(match["name"]))
    if reference not in keeper.reference_names:
        return set()
    return detail_names(version, replace(keeper, names=frozenset({reference})))


def _own_names(version: ForgottenCitation, keeper: KeeperGroup,
               sensitive_forget: bool) -> set[tuple[str, str]]:
    """The names under which ``keeper`` keeps its quotes for ``version``.

    A non-sensitive naming memory of the version's primary entity (never the user)
    keeps any quote, even when a forgotten version is sensitive (path a).
    A naming memory of another non-user entity e keeps
    only as a bare-name keeper: its head anchors e alone and names it by type
    alone with an allowlisted neutral kind. Only exact name subspans of the group's
    quotes may survive. This path (b) keeps nothing when any forgotten version
    is sensitive. A sensitive keeper keeps nothing on either path.
    """
    if keeper.sensitive:
        return set()
    if version.entity_id not in ("", "user"):
        own = {pair for pair in keeper.names if pair[0] == version.entity_id}
        if own:
            return own
    if sensitive_forget:
        return set()
    if keeper.bare not in (None, "", "user"):
        return {pair for pair in keeper.names if pair[0] == keeper.bare}
    return detail_names(version, keeper) or _reference_detail_names(version, keeper)


def plan_stating_spans(
    cited: Sequence[ForgottenCitation], keepers: Sequence[KeeperGroup], *,
    withdrawn: Set[tuple[str, str]] = frozenset(), any_script: bool = False,
    quoted: Callable[[QuoteSpanV1], str | None] | None = None,
) -> tuple[QuoteSpanV1, ...]:
    """The stating spans S a forget withholds (`plan_stating`'s spans)."""
    return plan_stating(cited, keepers, withdrawn=withdrawn, any_script=any_script,
                        quoted=quoted)[0]


_LISTED_BEFORE: Final = re.compile(r"(?:[、，,]\s*|\s+and\s+|[和与及])?(?:叫|名叫|名为)?$")
"""What joins a listed name to the name before it ("、" of "小王、小李", " and " of "Tom and
Ben"), with a naming word right before it."""


def _named_alone(version: ForgottenCitation, others: Set[str],
                 quoted: Callable[[QuoteSpanV1], str | None] | None) -> ForgottenCitation:
    """A forgotten naming (``bare_naming``: "A person is named 小李.") cut to its own name in
    each quote that lists another name too (``others``, the ``name_norm`` of the survivors'
    names): "、小李" of "同事小王、小李", its list mark and naming word with it, never the other
    listed name or their shared description ("同事"). A quote naming no other name, or in
    which the name is not found, is cited whole, as is every quote of a sensitive version."""
    found = bare_naming(version.statement)
    if found is None or quoted is None or version.sensitive or not version.names:
        return version
    name = found[1]
    own = name_norm(name)
    pattern = re.compile(re.escape(name) if own_token(name[0]) else
                         r"(?<![0-9A-Za-z])" + re.escape(name) + r"(?![0-9A-Za-z])")
    groups: list[tuple[QuoteSpanV1, ...]] = []
    for group in version.groups:
        cut: list[QuoteSpanV1] = []
        for span in group:
            text = quoted(span)
            matches = [] if text is None or not any(
                other != own and token_contained(other, text) for other in others
            ) else list(pattern.finditer(text))
            if not matches:
                cut.append(span)
                continue
            assert text is not None
            for match in matches:
                before = _LISTED_BEFORE.search(text[:match.start()])
                start = match.start() - (len(before.group()) if before is not None else 0)
                cut.append(QuoteSpanV1(
                    evidence_id=span.evidence_id,
                    byte_start=span.byte_start + len(text[:start].encode("utf-8")),
                    byte_end=span.byte_start + len(text[:match.end()].encode("utf-8"))))
        groups.append(tuple(cut))
    return replace(version, groups=tuple(groups))


def plan_stating(
    cited: Sequence[ForgottenCitation], keepers: Sequence[KeeperGroup], *,
    withdrawn: Set[tuple[str, str]] = frozenset(), any_script: bool = False,
    quoted: Callable[[QuoteSpanV1], str | None] | None = None,
) -> tuple[tuple[QuoteSpanV1, ...], frozenset[int]]:
    """The stating spans S a forget withholds.

    A forgotten version's quote can carry the phrase that introduces and names
    an entity the version is about (the establishing phrase of a
    ``new`` referent): "My partner Alex" in "My partner Alex works as a nurse".
    When a surviving naming memory of that entity quotes the phrase as its own
    readable support, the forget withholds the rest and leaves the phrase
    readable for that memory, so the entity keeps its name and the relearn
    guard keeps finding it. Everything else a forgotten version cites is
    withheld, as before:

    * a keeper counts for a version whose primary entity (never the user) it
      names, or, as a bare-name keeper, when its head anchors one non-user entity e
      alone and names it by exactly one allowlisted neutral kind. The bare keeper
      retains only the name itself within its quotes (``bare_name_spans``): "Rowan" in
      "I'm married to Rowan" stays readable for "A person is named Rowan.",
      while a keeper that also anchors or names the user or another anchor
      ("A person who is Avery's spouse is named Rowan.") keeps nothing. A broad
      quote ("married to Rowan") can keep only "Rowan". The bare-name entity need
      not be anchored by the forgotten version: a user-only profession fact
      can quote an employer's name without claiming anything about that entity;
    * a naming memory that also anchors its entity's owner or the user keeps its
      quote as a detail keeper (path c, ``detail_names``) when the version states
      a different detail of an entity it anchors ("The user bathes Pip, the user's
      tortoise, every morning." keeps "a tortoise named Pip" for "The user's
      tortoise is named Pip."), never when the version could restate it ("The
      user has a tortoise."); a non-naming head of the exact form "<known primary
      name> is|was <descriptor>." uses only that same detail guard, with reference
      names given by surviving naming memories that pass ``p_name``;
    * any other surviving memory keeps only its referent quote (path d,
      ``referent_spans``): "My brother" in "My brother is allergic to peanuts." stays
      readable for "The user's brother loves chess." (group "My brother", "He loves
      chess."), never a quote that carries a word of the forgotten predicate;
    * a surviving memory that keeps nothing on the paths above keeps, as a clause keeper
      (path e, ``clause_spans``), the runs of its quotes that carry only its own words
      and none of the forgotten predicate: "I run in the mornings" in "I volunteer at
      the animal shelter on weekends and I run in the mornings" for "The user runs in
      the mornings.";
    * a sensitive keeper keeps nothing on any path; paths (b), (c), (e) and (f) keep nothing
      when any forgotten version has a sensitivity class. Paths (a) and (d) still
      permit a non-sensitive keeper of a sensitive version;
    * never with a name some forgotten version gives that entity (it would
      restate the forgotten naming);
    * a head a forgotten correction reworded (path f, ``reworded``) keeps its
      own groups for every version that withdraws no name (``withdrawn``: a name a
      forgotten version gives that no surviving memory gives), unless the head or any
      forgotten version is sensitive: its wording states neither value and its quotes
      are its own support, so forgetting the correction leaves it readable,
      while a forgotten name and the correction's other words stay withheld;
    * a group whose every byte the version's keepers quote is withheld whole,
      so every forgotten group keeps a withheld remainder and none stays usable;
    * a keeper any of whose quotes still meets a withheld range is dropped (it
      stays unreadable), a clause keeper when one of its runs or of its quotes
      outside every cited span does, and the plan is recomputed until none is left; a keeper
      of whole quotes keeps its own clauses instead first (``clause_spans``), as it may have
      withheld a group whole.

    Bytes left readable are bytes a surviving, readable naming memory of the
    version's primary entity quotes as its own judged support, a name a
    bare-name keeper quotes, a detail keeper's quote, which carries no word
    of the forgotten detail, a referent keeper's "my" phrase or a clause keeper's
    run, which carry no word of the forgotten predicate, or a reworded head's own
    quotes, which carry no withdrawn name; every other cited byte is withheld, and
    every forgotten group keeps a withheld byte.

    Args:
        cited: Every non-erased version of the forgotten memories, with its groups.
        keepers: The readable groups of the surviving memories whose heads cite an
            Evidence row the forgotten versions cite, and of the heads a forgotten
            correction reworded.
        withdrawn: The ``(entity_id, name_norm)`` pairs the forget withdraws.
        any_script: Whether the clause keeper also cuts a quote in another script, or
            mixed, at its clauses (``clause_spans``; a complete forget).
        quoted: The text a cited span quotes (``clause_spans``), or None.

    Returns:
        Unique spans in (evidence, start, end) order; a span no keeper overlaps
        is unchanged, an overlapped one is cut into the pieces outside the kept
        quotes, and a fully kept one is absent. With them, the indices of the keeper
        groups the plan keeps: none of the bytes each needs is withheld, so its memory
        stays readable on that group, whole or requoted on its kept parts (a forget's
        preview reads them, ``ForgetOwner.preview``).
    """
    # A naming forgotten over a quote listing several names withholds its own name alone.
    others = {name for keeper in keepers for _, name in keeper.names}
    cited = [_named_alone(version, others, quoted) for version in cited]
    forgotten_names = {pair for version in cited for pair in version.names}
    sensitive_forget = any(version.sensitive for version in cited)
    cited_spans = [span for version in cited for group in version.groups for span in group]
    live = set(range(len(keepers)))
    demoted: set[int] = set()
    while True:
        withheld: dict[str, set[_Range]] = {}
        parts: dict[int, list[QuoteSpanV1]] = {}
        whole_keepers: set[int] = set()
        for version in cited:
            kept: dict[str, list[_Range]] = {}
            for index in sorted(live):
                keeper = keepers[index]
                if index in demoted:
                    # A keeper of whole quotes that withheld a group whole keeps its own
                    # clauses only (`clause_spans`).
                    spans = () if sensitive_forget or keeper.uncut else clause_spans(
                        version, keeper, forgotten_names, any_script=any_script, quoted=quoted)
                    parts.setdefault(index, []).extend(spans)
                elif keeper.reworded:
                    spans: Sequence[QuoteSpanV1] = keeper.spans if (
                        not keeper.sensitive and not sensitive_forget
                        and version.names.isdisjoint(withdrawn)) else ()
                    if spans:
                        whole_keepers.add(index)
                else:
                    own = _own_names(version, keeper, sensitive_forget)
                    if own and own.isdisjoint(forgotten_names) and keeper.bare is not None:
                        spans = bare_name_spans(keeper)
                        parts.setdefault(index, []).extend(spans)
                    else:
                        spans = (keeper.spans if own and own.isdisjoint(forgotten_names)
                                 else referent_spans(version, keeper, forgotten_names))
                        if spans:
                            whole_keepers.add(index)
                        elif not sensitive_forget:
                            spans = clause_spans(version, keeper, forgotten_names,
                                                 any_script=any_script, quoted=quoted)
                            parts.setdefault(index, []).extend(spans)
                for span in spans:
                    kept.setdefault(span.evidence_id, []).append(
                        (span.byte_start, span.byte_end))
            for group in version.groups:
                whole = all(not _subtract((span.byte_start, span.byte_end),
                                          kept.get(span.evidence_id, ())) for span in group)
                for span in group:
                    withheld.setdefault(span.evidence_id, set()).update(_subtract(
                        (span.byte_start, span.byte_end),
                        () if whole else kept.get(span.evidence_id, ()),
                    ))

        # A clause keeper needs its parts and its quotes outside every cited span; any
        # other keeper needs every quote of its group.
        needed = {index: keepers[index].spans if index in whole_keepers or not parts.get(index)
                  else (*parts[index], *(span for span in keepers[index].spans
                                         if not any(_meets(span, other) for other in cited_spans)))
                  for index in live}
        dropped = {
            index for index in live
            if any(start < span.byte_end and span.byte_start < end
                   for span in needed[index]
                   for start, end in withheld.get(span.evidence_id, ()))
        }
        if not dropped:
            return tuple(
                QuoteSpanV1(evidence_id=evidence_id, byte_start=start, byte_end=end)
                for evidence_id in sorted(withheld)
                for start, end in sorted(withheld[evidence_id])
            ), frozenset(live)
        # A keeper of whole quotes may have withheld a group whole: it keeps its own clauses
        # instead, and the others are planned again with it so.
        if dropped & whole_keepers - demoted:
            demoted |= dropped & whole_keepers
        else:
            live -= dropped
