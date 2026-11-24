"""What a complete forget by statement must not take, whatever the judge says.

The judge (``memory-forget-pick-v1``) decides which memories and sentences state what the user
asked to forget; it is the only one that adds a target. These mechanical checks only take a
target away, in the ``agent`` Add mode's forgets (``service/forget_statement.py``: the prepare,
the in-chat forget and the relearn check of every Add), when its text plainly states a
different fact than the user's words:

- **Another subject** (`other_subject`): the user's words name a person or an animal by their
  tie to the user, the clause of the text that carries the claim names one too, and no tie of
  one can be a tie of the other ("my sister" and "the user's brother", "my dog" and "the
  user's cat", 我姐姐 and 用户的妹妹). English and Chinese words read alike; an English word
  that covers two Chinese ones ("sister": 姐姐 or 妹妹) is compatible with both. A name the
  words give their person ("my sister Mia") found in the text vetoes nothing.
- **The opposite claim** (`negated`): the clause of the text that carries the claim is
  negated and the words are not, or the other way round ("I am allergic to peanuts" and "The
  user is not allergic to peanuts."). A negation elsewhere in the text ("I never drink
  coffee, only green tea.") vetoes nothing, and nor do the words asking not to keep the
  fact ("please don't remember that ...", "别记我 ...", `asked_fact`): they negate the
  request, not the fact.

The clause carrying the claim (`_claim`) is the clause of the text sharing the most terms with
the words, at least half of theirs; in a text in another script than the words, which shares
too few, every clause carries it: no clause of "My brother is learning Japanese, and he lives
in Osaka." names a tie "我妹妹" can be, so the text is not about the user's sister. Each check
errs towards keeping a target the judge picked, as forgetting too much is the safer error: a
text naming no tie, a clause sharing too little, or a word these lists do not know never
vetoes.

`clauses` splits a statement that may bundle several facts; the prepare reports a target
bundling a clause the judge finds unrelated (it is forgotten whole all the same).

One rule adds targets instead: a request to forget the name of a person or an
animal of the user's (`name_request`: "forget my brother's name", "忘掉我弟弟的名字") takes every
memory that ties a name to them (`naming_tie`: "Tom is the user's younger brother.", "The
user's cat is named Bob.") and the naming memories of that entity, whatever the judge picked;
no other fact about them, and nothing for a request about anything else.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from quotemem.core.privacy import support_words
from quotemem.core.recall_agent import content_terms

Exclusion = Literal["other_subject", "negated"]
"""Why a complete forget kept a target the judge picked out of the plan."""

_SISTERS = frozenset({"older-sister", "younger-sister"})
_BROTHERS = frozenset({"older-brother", "younger-brother"})
_PARENTS = frozenset({"mother", "father"})
_CHILDREN = frozenset({"son", "daughter"})
_SPOUSES = frozenset({"wife", "husband"})
_PARTNERS = _SPOUSES | {"girlfriend", "boyfriend"}
_GRANDPARENTS = frozenset({"grandmother", "grandfather"})
_EXES = frozenset({"ex-husband", "ex-wife", "ex-boyfriend", "ex-girlfriend"})
_PETS = frozenset({"dog", "cat", "bird", "hamster", "rabbit", "horse", "parrot", "turtle"})

TIES: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    # English: a word may cover several ties of the Chinese lists.
    "mother": frozenset({"mother"}), "mom": frozenset({"mother"}), "mum": frozenset({"mother"}),
    "father": frozenset({"father"}), "dad": frozenset({"father"}),
    "parent": _PARENTS, "parents": _PARENTS,
    "wife": frozenset({"wife"}), "husband": frozenset({"husband"}), "spouse": _SPOUSES,
    "partner": _PARTNERS, "girlfriend": frozenset({"girlfriend"}),
    "boyfriend": frozenset({"boyfriend"}),
    "ex-husband": frozenset({"ex-husband"}), "ex-wife": frozenset({"ex-wife"}),
    "ex-boyfriend": frozenset({"ex-boyfriend"}), "ex-girlfriend": frozenset({"ex-girlfriend"}),
    "ex-spouse": frozenset({"ex-husband", "ex-wife"}), "ex-partner": _EXES, "ex": _EXES,
    "son": frozenset({"son"}), "daughter": frozenset({"daughter"}),
    "child": _CHILDREN, "children": _CHILDREN, "kid": _CHILDREN, "kids": _CHILDREN,
    "sister": _SISTERS, "sisters": _SISTERS, "brother": _BROTHERS, "brothers": _BROTHERS,
    "sibling": _SISTERS | _BROTHERS, "siblings": _SISTERS | _BROTHERS,
    "grandmother": frozenset({"grandmother"}), "grandma": frozenset({"grandmother"}),
    "grandfather": frozenset({"grandfather"}), "grandpa": frozenset({"grandfather"}),
    "grandparent": _GRANDPARENTS, "grandparents": _GRANDPARENTS,
    "aunt": frozenset({"aunt"}), "uncle": frozenset({"uncle"}),
    "cousin": frozenset({"cousin"}), "niece": frozenset({"niece"}),
    "nephew": frozenset({"nephew"}), "friend": frozenset({"friend"}),
    "roommate": frozenset({"roommate"}), "neighbor": frozenset({"neighbor"}),
    "neighbour": frozenset({"neighbor"}), "boss": frozenset({"boss"}),
    "manager": frozenset({"boss"}), "supervisor": frozenset({"boss"}),
    "colleague": frozenset({"colleague"}), "coworker": frozenset({"colleague"}),
    "classmate": frozenset({"classmate"}), "teacher": frozenset({"teacher"}),
    "tutor": frozenset({"teacher"}), "mentor": frozenset({"mentor"}),
    "coach": frozenset({"coach"}), "doctor": frozenset({"doctor"}),
    "landlord": frozenset({"landlord"}), "landlady": frozenset({"landlord"}),
    "dog": frozenset({"dog"}), "puppy": frozenset({"dog"}), "cat": frozenset({"cat"}),
    "kitten": frozenset({"cat"}), "bird": frozenset({"bird"}), "parrot": frozenset({"parrot"}),
    "hamster": frozenset({"hamster"}), "rabbit": frozenset({"rabbit"}),
    "horse": frozenset({"horse"}), "turtle": frozenset({"turtle"}), "pet": _PETS,
    # Chinese, longest first where one contains another.
    "妈妈": frozenset({"mother"}), "母亲": frozenset({"mother"}),
    "爸爸": frozenset({"father"}), "父亲": frozenset({"father"}), "父母": _PARENTS,
    "老婆": frozenset({"wife"}), "妻子": frozenset({"wife"}),
    "老公": frozenset({"husband"}), "丈夫": frozenset({"husband"}),
    "前夫": frozenset({"ex-husband"}), "前妻": frozenset({"ex-wife"}),
    "前男友": frozenset({"ex-boyfriend"}), "前女友": frozenset({"ex-girlfriend"}),
    "女朋友": frozenset({"girlfriend"}), "男朋友": frozenset({"boyfriend"}), "对象": _PARTNERS,
    "儿子": frozenset({"son"}), "女儿": frozenset({"daughter"}), "孩子": _CHILDREN,
    "姐姐": frozenset({"older-sister"}), "妹妹": frozenset({"younger-sister"}),
    "姐妹": _SISTERS, "哥哥": frozenset({"older-brother"}),
    "弟弟": frozenset({"younger-brother"}), "兄弟": _BROTHERS,
    "奶奶": frozenset({"grandmother"}), "外婆": frozenset({"grandmother"}),
    "姥姥": frozenset({"grandmother"}), "爷爷": frozenset({"grandfather"}),
    "外公": frozenset({"grandfather"}), "姥爷": frozenset({"grandfather"}),
    "阿姨": frozenset({"aunt"}), "姑姑": frozenset({"aunt"}), "叔叔": frozenset({"uncle"}),
    "舅舅": frozenset({"uncle"}), "表哥": frozenset({"cousin"}), "表姐": frozenset({"cousin"}),
    "表弟": frozenset({"cousin"}), "表妹": frozenset({"cousin"}),
    "堂哥": frozenset({"cousin"}), "堂姐": frozenset({"cousin"}), "朋友": frozenset({"friend"}),
    "室友": frozenset({"roommate"}), "邻居": frozenset({"neighbor"}),
    "老板": frozenset({"boss"}), "经理": frozenset({"boss"}), "领导": frozenset({"boss"}),
    "同事": frozenset({"colleague"}), "同学": frozenset({"classmate"}),
    "老师": frozenset({"teacher"}), "导师": frozenset({"mentor", "teacher"}),
    "教练": frozenset({"coach"}), "医生": frozenset({"doctor"}), "大夫": frozenset({"doctor"}),
    "房东": frozenset({"landlord"}),
    "宠物": _PETS, "狗": frozenset({"dog"}), "猫": frozenset({"cat"}),
    "鹦鹉": frozenset({"parrot"}), "仓鼠": frozenset({"hamster"}), "兔子": frozenset({"rabbit"}),
    "乌龟": frozenset({"turtle"}),
})
"""Words that name a person or an animal by their tie to the user, and the ties each covers."""

_LATIN_WORD = re.compile(r"[a-z]+")
_HAN_TIES = sorted((word for word in TIES if not word.isascii()), key=len, reverse=True)
_NEGATION_EN = re.compile(r"\b(?:not|never|cannot|nor|no longer)\b|n't\b", re.IGNORECASE)
_NEGATION_ZH = re.compile(
    r"(?:不|没|无|别|未)(?![错同少久断但仅论管如然止安满足聊意限疑计人的来]|过(?!敏))")
_CLAUSE = re.compile(
    r"\s*(?:[;；。]|[,，、]\s*(?:and|but|while|并且|而且|同时|还)?|\s(?:and|but|while)\s|"
    r"并且|而且|同时|以及)\s*", re.IGNORECASE)
MIN_CLAUSE_TERMS: Final = 2
"""The `content_terms` a clause carries to count as a fact of its own (`clauses`)."""


def word_ties(text: str) -> Iterator[frozenset[str]]:
    """The ties to the user each person or animal word of a text covers, word by word."""
    for word in _LATIN_WORD.findall(text.lower().replace("'s", " ")):
        if word in TIES:
            yield TIES[word]
    rest = text
    for word in _HAN_TIES:
        if word in rest:
            yield TIES[word]
            rest = rest.replace(word, " ")


def ties(text: str) -> frozenset[str]:
    """The ties to the user every person or animal word of a text covers, joined."""
    return frozenset[str]().union(*word_ties(text))


def named_ties(text: str) -> frozenset[str]:
    """The ties of the person or animal words of a text that each name one ("猫" names
    "cat"); a word covering several ("pet", "父母") names none."""
    return frozenset(tie for found in word_ties(text) if len(found) == 1 for tie in found)


_PIECE = re.compile(
    r"\s*(?:[,，;；。:：、]|\s(?:and|but|while|so|because|since|although|though|whereas)\s|"
    r"并且|而且|同时|以及|但是|可是|所以|因为|虽然)\s*", re.IGNORECASE)
_APPOSITIVE = re.compile(r"\b([a-z]+)\s+([A-Z][a-z]+)\b")
_NOT_TERMS = frozenset({"not", "no", "never", "nor", "longer", "user"})
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
CLAIM_SHARE: Final = 0.5
"""The share of the words' terms a clause holds to carry their claim (`_claim`)."""


def _terms(text: str) -> frozenset[str]:
    """Stems of a text's words (``support_words``) and its CJK bigrams, negations aside."""
    return (support_words(text) | content_terms(text)) - _NOT_TERMS


def _pieces(text: str) -> list[str]:
    """A text's clauses: split at clause marks and conjunctions, those carrying a term."""
    return [piece for piece in (part.strip() for part in _PIECE.split(text))
            if piece and _terms(piece)] or [text]


def _claim(words: str, text: str) -> list[str]:
    """The clauses of ``text`` carrying the claim of ``words``: those sharing the most terms
    with them, at least ``CLAIM_SHARE`` of the words' terms; every clause of a text in
    another script, which no term test can place; none otherwise (no veto)."""
    asked = _terms(words)
    pieces = _pieces(text)
    if not asked:
        return []
    scored = [(len(asked & _terms(piece)), piece) for piece in pieces]
    top = max(score for score, _ in scored)
    if top and top >= CLAIM_SHARE * len(asked):
        return [piece for score, piece in scored if score == top]
    other_script = bool(_CJK.search(words)) != bool(_CJK.search(text))
    return pieces if other_script else []


def _named(words: str, text: str) -> bool:
    """Whether a name the words give a person or animal by its tie ("sister Mia") is in the
    text."""
    return any(tie.lower() in TIES and re.search(rf"\b{re.escape(name)}\b", text)
               for tie, name in _APPOSITIVE.findall(words))


def other_subject(words: str, text: str) -> bool:
    """Whether the words and the clause of the text carrying their claim name people or
    animals by a tie and no tie of one can be the other's: a statement about the user's
    brother is not one about the user's sister."""
    asked = ties(words)
    if not asked or _named(words, text):
        return False
    stated = frozenset[str]().union(*(ties(piece) for piece in _claim(words, text)))
    return bool(stated) and not asked & stated


_ASKING_EN = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|also|and|so|now|please|kindly|(?:can|could|would) you)[\s,]+)*"
    r"(?:(?:do not|don't|dont|never|no longer)\s+(?:remember|keep|store|save|record|retain|"
    r"memori[sz]e)|stop\s+(?:remembering|keeping|storing|saving|recording))\b",
    re.IGNORECASE)
_ASKING_ZH = re.compile(
    r"^\W*(?:请|麻烦|能不能|可以)?你?(?:帮我)?(?:以后|今后|之后|从现在起|从今以后)?"
    r"(?:不要|别|不用|不必|无需|甭)再?(?:记住|记得|记着|记录|保存|保留|存|记)")


def asked_fact(words: str) -> str:
    """The user's words of a fact without the words opening them that ask not to keep it:
    "please don't remember that I'm afraid of spiders" leaves "that I'm afraid of spiders",
    "以后不要记得我恐高了" leaves "我恐高了". Those words negate the request, never the fact,
    so only what they leave is read for the fact's own negation (`opposite`): "don't
    remember that I don't eat meat" still states a negated fact."""
    return _ASKING_ZH.sub("", _ASKING_EN.sub("", words, count=1), count=1)


def negated(text: str) -> bool:
    """Whether a text negates its claim (an English or Chinese negation word)."""
    return bool(_NEGATION_EN.search(text) or _NEGATION_ZH.search(text))


def opposite(words: str, text: str) -> bool:
    """Whether the clauses of the text carrying the words' claim all negate it and the
    words' fact does not, or the other way round. The words asking not to keep the fact
    ("don't remember", "别记", `asked_fact`) are no negation of it."""
    fact = asked_fact(words)
    claim = _claim(fact, text)
    return bool(claim) and all(negated(piece) != negated(fact) for piece in claim)


def exclusion(words: str, text: str) -> Exclusion | None:
    """Why a target the judge picked plainly states another fact than ``words``, or None."""
    if other_subject(words, text):
        return "other_subject"
    if opposite(words, text):
        return "negated"
    return None


def words_exclusion(words: str, text: str) -> Exclusion | None:
    """Why a memory found through a target's statement plainly states another fact than the
    user's ``words``: another subject, or a claim the text negates while the
    words do not. The words' own negation is not read: a request's words often carry its
    "以后不要再记得了" or "don't remember", which is no claim of the fact."""
    if other_subject(words, text):
        return "other_subject"
    if not negated(words) and opposite(words, text):
        return "negated"
    return None


_ZH_FILLER = re.compile(
    r"(?:这件事情?|这个事情?|这些事情?|那件事情?|这事儿?|的事情?|这一点|这些|这部分)"
    r"(?:吧|了|啊|呀)?[\s。.!！]*$")
_ZH_ABOUT = re.compile(r"^(?:关于|有关)\s*")
_EN_FILLER = re.compile(
    r"^(?:(?:the|any|all)\s+(?:fact|facts|detail|details|information|info|part|bit|thing|"
    r"things)\s+(?:that|about|of|regarding)\s+|that\s+(?=(?:i|i'm|i've|i'd|my|me)\b))",
    re.IGNORECASE)


def probe(words: str) -> str:
    """The user's words of a fact without the words that only point at it, for finding the
    memories that state it: a trailing 这件事, 的事 or 这一点, a leading 关于 or 有关, and a
    leading "the fact that", "the detail about" or "that" before the first person.

    "我在学日语这件事" is read as "我在学日语" and "the fact that I only drink oolong tea" as
    "I only drink oolong tea": the pointing words carry no claim, and they move the words'
    vector away from a statement of the fact in another language (measured with the agent
    mode's embedding model: 0.599 to "The user takes two Japanese lessons a week." with
    这件事, 0.690 without). The judge still reads the user's own words; words that are all
    pointer are kept as they are.
    """
    text = words.strip()
    for pattern in (_ZH_FILLER, _ZH_ABOUT, _EN_FILLER):
        cut = pattern.sub("", text).strip()
        if cut and len(content_terms(cut)) >= min(1, len(content_terms(text))):
            text = cut
    return text


def clauses(text: str) -> tuple[str, ...]:
    """A statement's clauses that carry their own content (``MIN_CLAUSE_TERMS`` terms each),
    split at clause marks and conjunctions; one clause or none when it bundles nothing."""
    return tuple(piece for piece in (part.strip() for part in _CLAUSE.split(text))
                 if len(content_terms(piece)) >= MIN_CLAUSE_TERMS)


_ORDER: Final = {"younger": "younger", "little": "younger", "baby": "younger",
                 "older": "older", "elder": "older", "big": "older"}
"""Words that say which sibling a description means ("my younger brother" is the
younger-brother tie of "brother")."""

_GIVEN = r"(?P<given>[A-Z][\w'’-]*(?: [A-Z][\w'’-]*)?)"
_END = (r"(?:,?\s*(?:please|too|as well|for me|now|thanks|thank you))?"
        r"\s*[.!?。！？]*\s*$")
_LINK = r"(?:[a-z]+(?:-[a-z]+)?\s+){0,2}?(?!name\b)[a-z]+(?:-[a-z]+)?"
_CHAIN = rf"(?P<chain>{_LINK}(?:['’]s\s+{_LINK}){{0,2}})"
"""A description of a person or an animal of the user's, one to three possessive links ("my
sister", "my sister's husband"), each up to two modifiers and a head word."""


def _en_requests(noun: str) -> tuple[re.Pattern[str], ...]:
    """The two English forms of a request ending with ``noun`` of someone of the user's."""
    return (re.compile(rf"\b[Mm]y\s+{_CHAIN}(?:\s+{_GIVEN})?['’]s\s+{noun}{_END}"),
            re.compile(rf"\b[Tt]he\s+{noun}\s+of\s+my\s+{_CHAIN}"
                       rf"(?!['’]s)(?:\s+{_GIVEN})?{_END}"))


_EN_NAME_REQUESTS: Final = _en_requests(r"(?:(?:first|last|full)\s+)?name")
"""A request to forget the name of a person or an animal of the user's, which ends the
request: "forget my brother's name", "forget my friend Alice's name", "the name of my cat",
"my sister's husband's name", "the name of my brother's girlfriend". "that my brother's name
is on the lease" and "my cat's nickname" are not one."""
_EN_NICKNAME_REQUESTS: Final = _en_requests(r"(?:nick ?name|pet name)")
"""The same of a nickname: "forget my cat's nickname", "the nickname of my daughter"."""
_ASKS_NAME: Final = (
    re.compile(rf"['’]s\s+(?:(?:first|last|full)\s+)?name{_END}"),
    re.compile(rf"\b[Tt]he\s+(?:(?:first|last|full)\s+)?name\s+of\s+\S.*?{_END}"),
    re.compile("(?:名字|全名|叫什么|叫啥)(?:吧|了|啊|呀|哦)?[\\s。.!！?？]*$"),
)
"""Any request ending with a name asked for, whoever bears it (`asks_name`)."""
_HAN_TIE = "|".join(_HAN_TIES)


def _zh_request(nouns: str) -> re.Pattern[str]:
    """A Chinese request ending with one of ``nouns`` of someone of the user's."""
    return re.compile(
        f"我的?(?P<chain>(?:{_HAN_TIE})(?:的?(?:{_HAN_TIE})){{0,2}})"
        f"(?P<given>[^\\s的，。、和与跟及它他她]{{0,4}}?)的?"
        f"(?:{nouns})(?:吧|了|啊|呀|哦)?[\\s。.!！?？]*$")


_ZH_NAME_REQUEST = _zh_request("名字|全名|叫什么|叫啥")
_ZH_NICKNAME_REQUEST = _zh_request("小名|昵称|外号|乳名|绰号|爱称")
_ZH_LINK = re.compile(f"(的)?({_HAN_TIE})")
_ZH_ROLES: Final = ("老师", "教练", "医生", "护士", "同学", "校长", "班主任", "老板", "上司",
                    "领导", "经理", "主管", "客户", "房东", "司机", "保姆", "学校", "公司", "店")
"""Role and place nouns that are no given name: "我女儿钢琴老师的名字" asks for the teacher's
name, not the daughter's."""
LINKED: Final = ">"
"""Joins the ties of a possessive chain, outer first: "older-sister>husband" is the husband
of the user's older sister (`chain_ties`)."""
_LINK_WORDS = rf"(?:[a-z]+(?:-[a-z]+)?(?: [a-z]+(?:-[a-z]+)?){{0,2}}|{_HAN_TIE})"
"""One link of a description: up to three Latin words, or a Han tie word an English statement
kept ("The user's 表哥 is named 志强.")."""
_LINKS = rf"{_LINK_WORDS}(?:'s {_LINK_WORDS}){{0,2}}"
_DESCRIPTION = rf"(?P<description>{_LINKS})"
_OUTER = r"(?P<outer>[a-z]+(?:-[a-z]+)?(?: [a-z]+(?:-[a-z]+)?){0,2})"
_GLOSS = r"(?:\s?[（(][^（）()]{1,60}[）)])?"
"""A gloss in parentheses an English statement puts after a Han tie word ("表哥 (cousin)")."""
_FORM = (r"[（(](?:[A-Z][A-Za-z'\u2019.-]*(?: [A-Z][A-Za-z'\u2019.-]*){0,3}"
         r"|[\u3400-\u4dbf\u4e00-\u9fff]{1,8})[）)]")
"""Another form of a name in parentheses beside it: a romanization of capitalized words
("（Lin Xiaoyu）") or a Han form ("(李伟)"); never "(my cat)"."""
PARENTHESIZED_FORM: Final = re.compile(rf"\s?{_FORM}")
"""`_FORM` right after a name in a text, with at most one space before it."""
_NAME = (r"(?P<name>(?:(?:(?:Mrs|Mr|Ms|Dr|Prof)\. )?(?:[A-Z][\w'\u2019-]*|[a-z]+)"
         r"(?: [A-Z][\w'\u2019-]*){0,2}|[\u3400-\u4dbf\u4e00-\u9fff]{1,8})"
         rf"(?:\s?{_FORM})?)")
"""A name: one to three words, each capitalized but a lone lowercase one ("bob"), after an
optional honorific ("Mrs. Halvorsen"), or a Han run; with another form of it in parentheses
("林晓雨（Lin Xiaoyu）", "Li Wei (李伟)")."""
_FORMS: Final = re.compile(rf"(?P<main>.+?)\s?(?P<form>{_FORM})")
_HAN_CHAR: Final = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def other_script(name: str, form: str) -> bool:
    """Whether ``form`` writes ``name`` in the other script, as a romanization or a Han
    translation does: "Lin Xiaoyu" of "林晓雨", "李伟" of "Li Wei"; not `UK` of "Tom"
    nor "我弟弟" of "李伟"."""
    return (_HAN_CHAR.search(name) is None) != (_HAN_CHAR.search(form) is None)


def name_forms(name: str) -> tuple[str, ...]:
    """A name and the names a parenthesis in it gives the same person: "林晓雨（Lin Xiaoyu）" is
    also "林晓雨" and "Lin Xiaoyu", "Li Wei (李伟)" also "Li Wei" and "李伟". A parenthesis
    in the same script is no form of the name (``Tom (UK)`` yields ``Tom``); any other
    name is itself alone."""
    match = _FORMS.fullmatch(name.strip())
    if match is None:
        return (name,)
    main, form = match["main"].strip(), match["form"][1:-1].strip()
    return tuple(dict.fromkeys((name, main, *((form,) if other_script(main, form) else ()))))


_NAMING_TIES: Final = (
    re.compile(rf"The user's {_DESCRIPTION}{_GLOSS} is (?:named|called) {_NAME}\."),
    re.compile(rf"The user's {_DESCRIPTION}'s name is {_NAME}\."),
    re.compile(rf"The name of the user's {_DESCRIPTION}{_GLOSS} is {_NAME}\."),
    re.compile(rf"The user has (?:a|an) {_DESCRIPTION}{_GLOSS} (?:named|called) {_NAME}\."),
    re.compile(rf"{_NAME} is the user's {_DESCRIPTION}{_GLOSS}\."),
    re.compile(rf"{_NAME} is the {_OUTER} of the user's {_DESCRIPTION}{_GLOSS}\."),
    re.compile(rf"The {_OUTER} of the user's {_DESCRIPTION}{_GLOSS} is (?:named|called) "
               rf"{_NAME}\."),
)
"""The statements that only tie a name to a person or animal of the user's: "The user's
younger brother is named Tom.", "Tom is the user's younger brother.", "Gregor is the user's
sister's husband.", "Chloe is the girlfriend of the user's brother.". The name is one to
three words, each capitalized but a lone lowercase one ("bob"), or a Han run: "Max and is
grey" or "The tall man" is no name."""
_NOT_NAMES: Final = frozenset({"the", "a", "an", "this", "that", "my", "his", "her", "their",
                               "our", "its", "someone", "somebody", "nobody"})


@dataclass(frozen=True)
class NameRequest:
    """A request to forget a name (`name_request`).

    Attributes:
        ties: The ties of the person or animal whose name it is; a possessive chain's are
            joined by ``LINKED`` (`chain_ties`).
        given: The name the request gives them ("Alice" of "my friend Alice's name"), or None.
    """

    ties: frozenset[str]
    given: str | None = None


def _described(words: str, head: str) -> frozenset[str]:
    """The ties a description means: its head word's (``TIES``), narrowed to one sibling by
    an order word before it ("younger brother"); none for a head word that names no tie."""
    found = TIES.get(head.lower(), frozenset[str]())
    order = {_ORDER[word] for word in words.lower().split() if word in _ORDER}
    narrowed = frozenset(tie for tie in found if any(tie.startswith(f"{side}-")
                                                     for side in order))
    return narrowed if order and narrowed else found


def _joined(links: list[frozenset[str]]) -> frozenset[str]:
    """The ties of a possessive chain of ``links``' ties, outer first; none when a link names
    no tie."""
    if not links or not all(links):
        return frozenset()
    found = links[0]
    for ties_ in links[1:]:
        found = frozenset(f"{outer}{LINKED}{inner}" for outer in found for inner in ties_)
    return found


_ROLES: Final = frozenset({
    "manager", "supervisor", "classmate", "teacher", "tutor", "mentor", "coach", "doctor",
    "landlord", "landlady", "经理", "领导", "同学", "老师", "导师", "教练", "医生", "大夫", "房东"})
"""Tie words that name a role, not a relation: the user's own ("my teacher's name") is a tie,
another person's ("my son's teacher", "我女儿的老师") is no tie of the user's, so a chain
ending in one names nobody the user is tied to (`chain_ties`, `_zh_chain`)."""


def chain_ties(description: str) -> frozenset[str]:
    """The ties an English description means, one link of a possessive chain after another
    ("sister's husband": {older-sister>husband, younger-sister>husband}); none when a link's
    head word names no tie, or a later link a role ("son's teacher", ``_ROLES``)."""
    links: list[frozenset[str]] = []
    for index, link in enumerate(re.split(r"['’]s\s+", description.strip())):
        words = link.split()
        if not words or (index and words[-1].lower() in _ROLES):
            return frozenset()
        links.append(_described(" ".join(words[:-1]), words[-1]))
    return _joined(links)


def _zh_chain(chain: str) -> frozenset[str]:
    """The ties of a Chinese possessive chain ("姐姐的丈夫", "姐姐老公", "哥哥的狗"). A word
    repeated ("狗狗") is one link; an animal word joined without 的 to the next word is a
    compound naming that next one ("宠物狗": a dog)."""
    links: list[tuple[bool, str]] = [(bool(match[1]), match[2])
                                     for match in _ZH_LINK.finditer(chain)]
    ties_: list[frozenset[str]] = []
    previous = ""
    for marked, word in links:
        if word == previous and not marked:
            continue
        if ties_ and word in _ROLES:
            return frozenset()  # "我女儿的老师": another person's role, no tie of the user's.
        if ties_ and not marked and ties_[-1] <= _PETS:
            ties_[-1] = TIES[word]
        else:
            ties_.append(TIES[word])
        previous = word
    return _joined(ties_)


def name_request(words: str, *, nickname: bool = False) -> tuple[NameRequest, ...]:
    """The requests to forget the name of a person or an animal of the user's that ``words``
    end with ("forget my brother's name", "forget my friend Alice's name", "the name of my
    cat", "forget my sister's husband's name", "忘掉我弟弟的名字", "忘掉我朋友小雪的名字",
    "我的猫叫什么", "忘掉我哥哥的狗的名字"); none for a request about anything else,
    nor for the name of someone or something a link names by no tie ("the
    name of my son's teacher", "我女儿钢琴老师的名字"). "brother" covers both brothers, "my
    younger brother" and 弟弟 the younger one; a possessive chain's ties are joined
    (`chain_ties`), so "my sister's husband" is not "my husband". With ``nickname``, the
    requests for a nickname instead ("forget my daughter's nickname", "忘掉我女儿的小名")."""
    found: list[NameRequest] = []
    for pattern in _EN_NICKNAME_REQUESTS if nickname else _EN_NAME_REQUESTS:
        for match in pattern.finditer(words):
            ties_ = chain_ties(match["chain"])
            if ties_:
                found.append(NameRequest(ties_, match["given"]))
    zh = _ZH_NICKNAME_REQUEST if nickname else _ZH_NAME_REQUEST
    for match in zh.finditer(words.strip()):
        given = match["given"] or None
        if given is not None and any(word in given for word in (*_HAN_TIES, *_ZH_ROLES)):
            continue  # Another person's or thing's name ("我女儿钢琴老师的名字").
        ties_ = _zh_chain(match["chain"])
        if ties_:
            found.append(NameRequest(ties_, given))
    return tuple(found)


_ASKING: Final = re.compile(
    r"^\s*(?:(?:please|pls|kindly|can you|could you|would you|i want you to)\s+)*"
    r"(?:forget|erase|remove|delete|drop|stop remembering)\s+(?:about\s+)?", re.IGNORECASE)
_ZH_ASKING: Final = re.compile(
    r"^\s*(?:请|麻烦|帮我|你|把)*(?:忘掉|忘记|忘了|删掉|删除|去掉|别记得?)")
_GIVEN_ASKED: Final = (
    re.compile(rf"{_GIVEN}['\u2019]s\s+(?:(?:first|last|full)\s+)?name{_END}"),
    re.compile(rf"[Tt]he\s+(?:(?:first|last|full)\s+)?name\s+of\s+{_GIVEN}{_END}"),
    re.compile(r"(?P<given>[A-Za-z][A-Za-z'\u2019-]*(?: [A-Za-z][A-Za-z'\u2019-]*)?"
               r"|(?:(?![的我你他她它们])[\u4e00-\u9fff]){2,4})\s*的?(?:名字|全名)"
               r"(?:吧|了|啊|呀|哦)?[\s。.!！?？]*"),
)
"""A request naming the one whose name it asks by that name alone (`given_request`)."""
_BARE_GIVEN = r"[A-Z][\w'\u2019-]*(?: [A-Z][\w'\u2019-]*)?"
_NAME_ONLY: Final = (
    re.compile(rf"(?:[Mm]y\s+{_CHAIN}(?:\s+{_GIVEN})?|{_BARE_GIVEN})['\u2019]s\s+"
               rf"(?:(?:first|last|full)\s+)?name{_END}"),
    re.compile(rf"[Tt]he\s+(?:(?:first|last|full)\s+)?name\s+of\s+(?:my\s+)?{_CHAIN}"
               rf"(?!['\u2019]s)(?:\s+{_GIVEN})?{_END}"),
    re.compile(r"我?的?[^\s，。、和与跟及还并]{1,12}?的?(?:名字|全名|叫什么|叫啥)(?:吧|了|啊|呀|哦)?"
               r"[\s。.!！?？]*"),
)
"""A request that is a name phrase and nothing else, once its asking words are cut."""


_NICKNAME: Final = re.compile(
    r"\b(?:nick ?names?|pet names?)\b|小名|昵称|外号|乳名|绰号|爱称", re.IGNORECASE)


def nickname_request(words: str) -> bool:
    """Whether ``words`` ask for a nickname rather than the name: "forget my daughter's
    nickname", "忘掉我女儿的小名"."""
    return _NICKNAME.search(words) is not None


def given_request(words: str) -> NameRequest | None:
    """A request to forget the name of someone it names by that name alone, with no tie:
    "forget Ben's name", "忘掉Emma的名字", "Emma的名字", "忘掉志强的名字"; None
    for any other, a request naming a tie included (`name_request`)."""
    text = _ZH_ASKING.sub("", _ASKING.sub("", words.strip(), count=1), count=1).strip()
    for pattern in _GIVEN_ASKED:
        match = pattern.fullmatch(text)
        if match is not None and match["given"].split()[0].casefold() not in _NOT_NAMES:
            return NameRequest(frozenset(), match["given"])
    return None


def name_only(words: str) -> bool:
    """Whether ``words`` ask for a name and nothing else: "forget my dog's name", "忘掉我的猫
    叫什么"; not "忘掉我的狗和它的名字" or "forget that I have a sister and my sister's name",
    which also ask for the dog or the sister."""
    text = _ZH_ASKING.sub("", _ASKING.sub("", words.strip())).strip()
    return any(pattern.fullmatch(text) is not None for pattern in _NAME_ONLY)


def asks_name(words: str) -> bool:
    """Whether ``words`` end by asking for a name, whoever bears it: "forget my manager's
    name", "forget Tom's name", "the name of my son's teacher", "忘掉Emma的名字", "我的猫叫什
    么"; not "my cat's nickname" or "我的猫的小名"."""
    text = words.strip()
    return any(pattern.search(text) for pattern in _ASKS_NAME)


_ASKED: Final = (
    re.compile(rf"\b[Mm]y\s+{_CHAIN}(?:\s+{_GIVEN})?['\u2019]s\s+"
               rf"(?:(?:first|last|full)\s+)?name{_END}"),
    re.compile(rf"\b[Tt]he\s+(?:(?:first|last|full)\s+)?name\s+of\s+my\s+{_CHAIN}"
               rf"(?!['\u2019]s)(?:\s+{_GIVEN})?{_END}"),
    re.compile(rf"(?<![\w'\u2019]){_GIVEN}['\u2019]s\s+(?:(?:first|last|full)\s+)?name{_END}"),
    re.compile(r"(?P<given>[A-Za-z][\w'-]*(?: [A-Za-z][\w'-]*)?)\s*的?"
               r"(?:名字|全名)(?:吧|了|啊|呀|哦)?[\s。.!！?？]*$"),
)
"""A request asking a name by its bearer's description or by the name itself (`asked_by`)."""


def asked_by(words: str, description: str, name: str) -> bool:
    """Whether ``words`` ask the name a statement ties to someone by ``description`` (`tied_name`):
    the request's description ends with the same word or one covering the same tie ("my
    manager's name" and "manager", "my sister's husband's name" and "sister's husband"), or
    the request gives that name itself ("forget Priya's name", "忘掉Priya的名字")."""
    text = words.strip()
    last = description.split()[-1].casefold() if description.split() else ""
    for pattern in _ASKED:
        for match in pattern.finditer(text):
            given = match.groupdict().get("given")
            if given is not None and given.casefold() == name.casefold():
                return True
            chain = match.groupdict().get("chain")
            head = chain.split()[-1].casefold() if chain else ""
            if head and (head == last or TIES.get(head, frozenset()) & TIES.get(
                    last, frozenset())):
                return True
    return False


def tied_name(statement: str) -> tuple[str, str] | None:
    """The description and the name of a statement that only ties a name to someone or
    something of the user's (``_NAMING_TIES``), by any description, a tie word or a role:
    ("manager", "Priya") of "Priya is the user's manager.", ("sister's husband", "Gregor") of
    "Gregor is the user's sister's husband." (a possessive chain's links outer first), a
    gloss after it aside (``_GLOSS``: "older male cousin" of "... cousin (表哥)."); None for any
    other statement."""
    text = statement.strip()
    for pattern in _NAMING_TIES:
        match = pattern.fullmatch(text)
        if match is None:
            continue
        name = match["name"].strip()
        if name.split()[0].casefold() in _NOT_NAMES:
            continue
        outer = match.groupdict().get("outer")
        description = match["description"] if outer is None \
            else f"{match['description']}'s {outer}"
        return description, name
    return None


def naming_tie(statement: str) -> tuple[frozenset[str], str] | None:
    """The ties and the name of a statement that only ties a name to a person or animal of
    the user's (``_NAMING_TIES``): "The user's younger brother is named 李伟." gives
    ({younger-brother}, "李伟"), "Tom is the user's brother." ({older-brother,
    younger-brother}, "Tom"), "Gregor is the user's sister's husband."
    ({older-sister>husband, younger-sister>husband}, "Gregor"); None for any other statement
    or a description a link of which names no tie."""
    parts = tied_name(statement)
    if parts is None:
        return None
    ties_ = chain_ties(parts[0])
    return (ties_, parts[1]) if ties_ else None
