"""A complete forget's rewording of a memory that tells the forgotten detail beside other facts.

"Forget my wife's name" takes the memories that give the name, and with them, before this
step, every other memory whose statement still showed it: "Lin Xiaoyu is an architect." went
with the name although the user asked to forget only the name. A correction lost its new
value the same way ("The user moved from Shanghai to Hangzhou last month." after "forget
that I lived in Shanghai"), and so did a statement bundling the forgotten detail with
another fact ("The user plays badminton at 星河体育馆 on weekends." after "forget which
sports hall I play at").

Such a memory is reworded instead: the forgotten name gives way to the tie the user has to
its bearer, as a stored naming memory states it ("The user's wife is an architect."), and
a forgotten detail leaves with the phrase that carries it ("The user moved to Hangzhou last
month.", "The user plays badminton on weekends."). This module is the pure text step:
`tie_descriptions` reads the ties, `forgotten_values` the details, `reworded_statement`
writes the new statement or refuses. The forget then admits the new statement only after
checking it again (``ForgetOwner._admit_wording``): it shows no forgotten word, it keeps the
statement's predicate, the memory's readable quotes still support what it says, and the
forget judge finds it does not state the forgotten fact. Anything else is taken as before
(fail closed). A tie stands for a name only when the user asked to forget the name.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from quotemem.core.statement import canonical_statement, own_token, token_contained, token_seq

MAX_REWORDINGS: Final = 16
"""The most memories one forget rewords; the others are taken as before."""

_HAN_KIN: Final = (
    "男朋友", "女朋友", "表哥", "表姐", "表弟", "表妹", "堂哥", "堂姐", "堂弟", "堂妹", "哥哥",
    "姐姐", "弟弟", "妹妹", "爸爸", "妈妈", "父亲", "母亲", "老婆", "妻子", "老公", "丈夫",
    "儿子", "女儿", "奶奶", "爷爷", "外婆", "外公", "姥姥", "姥爷", "叔叔", "阿姨", "舅舅",
    "姑姑", "朋友", "同事", "室友", "邻居", "老板", "宠物", "猫", "狗",
)
"""The Han kin words an English statement may keep as its tie ("The user's 表哥 is named
志强."), longest first."""
_HAN_TIE: Final = "|".join(_HAN_KIN)
_SIDE: Final = r"(?: on (?:his|her|their|the user['’]s) (?:mother|father)['’]s side)"
"""The side of the family a tie may be told with ("older male cousin on his mother's side",
as a statement glosses "表哥")."""
_DESCRIPTION: Final = (r"(?P<description>[a-z][a-z-]*(?:(?:['’]s)? [a-z][a-z-]*){0,3}"
                       rf"{_SIDE}?|(?:{_HAN_TIE}))")
"""A tie as it reads after "the user's": "wife", "best friend", one of another person's
("brother's girlfriend", "daughter's piano teacher"), or a Han kin word ("表哥")."""
_NAME: Final = r"(?P<name>[^,.;:]{1,64}?)"
_TIES: Final = (
    re.compile(rf"The user's {_DESCRIPTION} is (?:named|called) {_NAME}\."),
    re.compile(rf"The user's {_DESCRIPTION}'s name is {_NAME}\."),
    re.compile(rf"The name of the user's {_DESCRIPTION} is {_NAME}\."),
    re.compile(rf"The user has (?:a|an) {_DESCRIPTION} (?:named|called) {_NAME}\."),
    re.compile(rf"{_NAME} is the user's {_DESCRIPTION}\."),
    re.compile(rf"{_NAME} is the (?P<role>[a-z][a-z-]*) of the user's {_DESCRIPTION}\."),
)
"""The statements that tie a name to a person or animal of the user's: "林晓雨 is the
user's wife.", "The user's dog is named Biscuit.", "Gregor is the husband of the user's
sister." (the user's sister's husband)."""
_APPOSITION: Final = re.compile(rf"The user's {_DESCRIPTION} (?!(?:his|her|their|its|the|a|an)\b)"
                                rf"{_NAME}(?= (?:is|was|has|had|"
                                r"works|lives|likes|loves|goes|plays|runs|studies|attends)\b)")
"""A memory naming the bearer after the tie it describes ("The user's boss Henrik works
from the Berlin office."): the tie is the memory's own."""
_NOT_TIES: Final = frozenset({"name", "names", "nickname", "real name", "full name"})
_NAMED_NOUN: Final = (r"(\b(?:a|an|the)\s+[a-z][a-z-]*)\s+(?:named|called)\s+{value}"
                      r"(?=[\s.,;]|$)")
"""A noun phrase the value names ("a company named 青柠科技"): the noun stays, the name goes."""
_NAMING_VERB: Final = re.compile(
    r"\b(?:named|called|nicknamed|renamed|known\s+as|name\s+is)\s*$", re.IGNORECASE)
"""A name right after a naming verb states the name itself: such a memory is no rewording's."""
_POSSESSED: Final = re.compile(
    r"(?:\bthe user['’]s|\b(?:his|her|their|its))"
    rf"(?P<words>(?: (?:[a-z][a-z'’-]*|{_HAN_TIE})){{1,4}}{_SIDE}?)(?P<gap>,? )$", re.IGNORECASE)
"""A possessive description right before a name ("the user's dog ", "The user's paternal
grandmother "): when it describes the name's bearer by its tie, the name only identifies it
and leaves alone; any other description ("the user's son walks ") is not the bearer's."""
_PARENTHESIS: Final = re.compile(r"^\s*[(（][^)）]{1,64}[)）]")
"""A parenthesis right after a name or value: another form of it ("李伟 (Li Wei)")."""
_NEGATION: Final = re.compile(
    r"(?:\b(?:not|no|never|nor|without|no longer|neither)\b|n['’]t\b)", re.IGNORECASE)
"""A negated statement: no phrase of it leaves alone ("is not friends with Sofia")."""
_FROM_TO: Final = r"\s+from\s+(?:the\s+)?{value}(?=\s+to\b)"
_LEFT_AND: Final = r"\b(?:left|quit)\s+(?:the\s+)?{value}\s+and\s+"
_FIRST_OF_TWO: Final = (r"(\b(?:with|and|at|in|to|for|from|near|visited|visit|visits)\s+)"
                        r"(?:the\s+)?{value}\s+and\s+"
                        r"(?=(?:the user['’]s |the |a |an |my |his |her |their )\w"
                        r"|(?-i:[A-Z\u3400-\u9fff]))")
"""The forgotten value first of two ("with Tom and Lucas"): only it and its "and" leave, when
the "and" joins it to another noun phrase (never "works at 青柠科技 and lives in Seattle")."""
_LAST_OF_TWO: Final = r",?\s+and\s+(?:the\s+)?{value}(?=[\s.,;]|$)"
"""The forgotten value last of two ("with Lucas and Tom"): only "and" and it leave."""
_MEMBER: Final = r"(?-i:[Tt]he user|[A-Z][\w'’-]*|[\u3400-\u9fff]+)"
"""One member of a list of names: the user, a capitalized word or a Han run."""
_SERIAL: Final = re.compile(
    rf"(?:^\s*|\b(?:[Ww]ith|and|by|of|for|to|visited|met|saw)\s+)"
    rf"{_MEMBER}(?:,\s+{_MEMBER})*(,)\s+{_MEMBER}$")
"""Members of a list of three or more before its last ("The user, 小王", "with Tom, Lucas"):
the comma before the last one left becomes its "and"."""
_COUNTED: Final = re.compile(
    rf"\b(?:two|three|four|five|both|2|3|4|5)\s+(?P<noun>[a-z]+)s,?\s+{_MEMBER}$",
    re.IGNORECASE)
"""A counted plural tie before the one left ("two colleagues, 小王")."""
_PLURAL_SUBJECT: Final = re.compile(
    rf"(?P<owner>\s*the user['’]s\s+)(?P<noun>[a-z]+)s(?P<gap>\s+)(?P<verb>are|were)\s+"
    rf"{_MEMBER}$", re.IGNORECASE)
"""A plural tie the two are the complement of ("The user's colleagues are 小王")."""
_PLURAL_BEFORE: Final = re.compile(rf"\b(?P<plural>(?P<noun>[a-z]+)s)\s+{_MEMBER}$")
"""A plural noun right before the one left ("colleagues 小王")."""
_WHAT_THE_USER_IS: Final = re.compile(r"The user (?:is|was) (?:a|an) [a-z][a-z-]*\.",
                                      re.IGNORECASE)
"""What the user is, one noun: it says something of its own ("The user is a designer." of
"The user is a designer at Bluefin Labs.")."""
_SINGULAR: Final = {"are": "is", "were": "was", "have": "has"}
"""A verb agreeing with two, and its form agreeing with one."""
_AGREE_ANY: Final = frozenset({
    "came", "went", "did", "had", "saw", "made", "took", "got", "met", "said", "ran", "ate",
    "drove", "flew", "gave", "left", "sat", "spent", "won", "will", "would", "can", "could",
    "may", "might", "should", "must",
})
"""Verbs that agree with one as with two."""
_TOGETHER: Final = re.compile(
    r"\b(?:both|together|each other|one another|they|them|their|theirs)\b", re.IGNORECASE)
"""Words that still speak of two: a coordination left with one member cannot keep them."""
_PHRASE: Final = r"\s+(?:at|in|near|from|with|for)\s+(?:the\s+)?{value}(?=[\s.,;]|$)"
_BARE: Final = re.compile(
    r"\b(?:live|lives|lived|living|stay|stays|stayed|work|works|worked|working|is|was|are|"
    r"were|be|based|located|born|move|moves|moved|grew up|go|goes|went|study|studies|studied|"
    r"attend|attends|attended|friends|married|engaged|dating|together)$", re.IGNORECASE)
"""Words a place or company phrase completes ("lives in Shanghai", "works at 青柠科技",
"friends with Sofia"): without it they say nothing, so their phrase never leaves alone."""
_DANGLING: Final = re.compile(
    r"(?:\b(?:at|in|near|from|to|with|for|of|and|the|a|an)|['’]s|,)\s*[.!]?$", re.IGNORECASE)
_LIGHT: Final = frozenset({
    "the", "a", "an", "user", "s", "is", "are", "was", "were", "has", "have", "had", "and",
    "of", "to", "in", "at", "on", "for", "with", "from", "by", "this", "that", "be", "been",
    "also", "now", "i", "the user",
})
_REMOVABLE: Final = frozenset({"from", "at", "in", "near", "with", "for", "and", "the", "a",
                               "an", "left", "quit", "named", "called"})
"""The words a value's phrase takes with it ("from Shanghai", "left Northwind and", "named
Clover")."""
_CONTRASTING: Final = frozenset({"not", "rather", "than", "instead", "of", "no", "longer",
                                 "anymore", "any", "more"})
"""The words a contrast tail takes with the value it names (`_CONTRAST`): only then."""
_KINDS: Final = (
    ("幼儿园", "kindergarten"), ("小学", "primary school"), ("初中", "middle school"),
    ("中学", "middle school"), ("高中", "high school"), ("大学", "university"),
    ("学院", "college"), ("学校", "school"), ("医院", "hospital"), ("诊所", "clinic"),
    ("体育馆", "sports hall"), ("健身房", "gym"), ("银行", "bank"), ("餐厅", "restaurant"),
    ("餐馆", "restaurant"), ("酒店", "hotel"), ("公司", "company"), ("科技", "company"),
    ("集团", "company"), ("Elementary School", "elementary school"),
    ("Primary School", "primary school"), ("Middle School", "middle school"),
    ("High School", "high school"), ("School", "school"), ("Primary", "primary school"),
    ("Elementary", "elementary school"), ("Academy", "school"), ("Preschool", "preschool"),
    ("Kindergarten", "kindergarten"), ("University", "university"),
    ("College", "college"), ("Hospital", "hospital"), ("Clinic", "clinic"), ("Bank", "bank"),
    ("Restaurant", "restaurant"), ("Hotel", "hotel"), ("Inc", "company"),
    ("Corporation", "company"), ("Corp", "company"), ("Company", "company"), ("Ltd", "company"),
    ("面包店", "bakery"), ("咖啡馆", "cafe"), ("咖啡店", "cafe"), ("工作室", "studio"),
    ("理发店", "salon"), ("商店", "shop"), ("Bakery", "bakery"), ("Café", "cafe"),
    ("Cafe", "cafe"), ("Shop", "shop"), ("Store", "store"), ("Studio", "studio"),
    ("Salon", "salon"),
)
"""The word a place's or an organisation's name ends with, and the kind it names."""
_LATIN_RUN: Final = re.compile(r"(?<![\w'’-])[A-Z][\w-]*(?: [A-Z][\w-]*)*")
_HAN_RUN: Final = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]{2,}")
_STARTS: Final = frozenset({"the", "a", "an", "this", "that", "my"})
"""Words a statement starts with only by its sentence case: no name or value."""


_ASKED_KINDS: Final = (
    (("school", "university", "college", "kindergarten"),
     ("学校", "小学", "中学", "大学", "学院", "幼儿园", "school", "university", "college",
      "kindergarten")),
    (("company",), ("公司", "单位", "company", "employer", "firm", "workplace")),
    (("hospital", "clinic"), ("医院", "诊所", "hospital", "clinic")),
    (("sports hall", "gym"), ("体育馆", "健身房", "球馆", "gym", "sports hall")),
    (("restaurant",), ("餐厅", "餐馆", "饭店", "restaurant")),
    (("bank",), ("银行", "bank")),
    (("hotel",), ("酒店", "宾馆", "hotel")),
)
"""The kinds a request may ask a name of, and the words it asks them by ("学校" for 晨光小学)."""


def asked_kinds(request: str) -> frozenset[str]:
    """The kinds a request asks a place of by their own word ("company" of "忘掉我在哪家公司
    上班", "sports hall" of "我在哪个体育馆打球"): a place standing as one of them tells no more
    than the request does."""
    text = request.casefold()
    return frozenset(name for names, words in _ASKED_KINDS for name in names
                     if any(word in text for word in words))


def asked_sources(request: str, statements: Sequence[str]) -> list[str]:
    """What is forgotten of each of ``statements`` (the targets): the values of it the
    request shows, when it shows one ("上海" of "The user moved from 杭州 to 上海." for "忘掉我在
    上海住过": "杭州" is the value the user kept), else the names of the kind whose name it
    asks (`asked_values`: "Maple Grove Elementary" of "Ruby goes to Maple Grove Elementary."
    for "forget the name of my daughter's school", never "Ruby"), else the whole statement;
    and the request. Once one shows the kind asked, one showing none of it gives nothing:
    "Ruby is the user's daughter." only tells whose school it is."""
    found = [request]
    kinds = {statement: asked_values(statement, request) for statement in statements
             } if _NAME_ASKED.search(request) is not None else dict[str, list[str]]()
    shown = any(kinds.values())
    for statement in statements:
        asked = [run for run in _runs(statement) if token_contained(run, request)]
        found += asked or kinds.get(statement, []) or ([] if shown else [statement])
    return found


_NAME_ASKED: Final = re.compile(r"\bnames?\b|名字|名称", re.IGNORECASE)
"""A request asking a name ("the name of my daughter's school", "学校的名字"), not which one
("which school my daughter goes to", "哪个学校": the kind goes too)."""


_DETAIL: Final = re.compile(r"\b(?:where|which|what)\b|哪|什么", re.IGNORECASE)
"""A request asking a detail of a fact ("forget where I work", "忘掉我在哪家公司上班")."""


def detail_values(statement: str, request: str) -> list[str]:
    """The values a request asking a detail ("where I work", "哪家公司") asks of a statement:
    its names and Han runs but the one it is about ("Leo" of "Leo attends Willowbrook
    Primary." for "which school my son goes to"), when the request asks such a detail; none
    otherwise."""
    if not _DETAIL.search(request):
        return []
    return [run for run in _runs(statement) if not re.match(
        rf"\s*(?:the user['’]s\s+(?:(?-i:[a-z-]+)\s+){{1,3}})?{re.escape(run)}(?![\w-])",
        statement, re.IGNORECASE)]


def asked_values(statement: str, request: str) -> list[str]:
    """The names of places or organisations a statement shows whose kind (`kind_of`) the
    request asks a name of: "晨光小学" of "The user's daughter attends 晨光小学." for "忘掉我女儿
    学校的名字", whatever annotations the extraction gave."""
    text = request.casefold()
    found: list[str] = []
    for run in _runs(statement):
        kind = kind_of(run)
        if kind is not None and any(
                kind.endswith(name) and any(word in text for word in words)
                for names, words in _ASKED_KINDS for name in names):
            found.append(run)
    return found


def _runs(statement: str) -> list[str]:
    """The capitalized Latin runs of a statement, the sentence case of a leading article
    left out, and its runs of two or more Han characters."""
    runs: list[str] = []
    for match in _LATIN_RUN.finditer(statement):
        run = match.group()
        if match.start() == 0 and run.split()[0].casefold() in _STARTS:
            run = run.partition(" ")[2]
        if len(run) > 1 and run.casefold() not in _LIGHT:
            runs.append(run)
    return [*runs, *(match.group() for match in _HAN_RUN.finditer(statement))]


def tie_descriptions(statements: Sequence[str]) -> dict[str, str]:
    """The tie each name the ``statements`` give is borne under, as it reads after "the
    user's" ("wife", "best friend", "sister's husband"): from the statements that only tie a
    name to the user's person or animal, and from a memory's own leading description ("The
    user's boss Henrik works ..."). A name two statements tie differently gives none."""
    found: dict[str, set[str]] = {}
    for statement in statements:
        # A gloss in parentheses after the tie is no part of it ("志强 is the user's older male
        # cousin (表哥).").
        text = re.sub(r"\s*[(（][^)）]{1,32}[)）](?=\.$)", "", statement.strip())
        for pattern in _TIES:
            match = pattern.fullmatch(text)
            if match is None:
                continue
            description = match["description"]
            if "role" in match.groupdict() and match["role"]:
                description = f"{description}'s {match['role']}"
            if description not in _NOT_TIES:
                found.setdefault(match["name"].strip(), set()).add(description)
            break
        else:
            match = _APPOSITION.match(text)
            # No kin word is a name: "The user's paternal grandmother is also called 兰兰奶奶."
            if (match is not None and match["description"] not in _NOT_TIES
                    and not _kin_word(match["name"].strip())):
                found.setdefault(match["name"].strip(), set()).add(match["description"])
    return {name: next(iter(ties)) for name, ties in found.items() if len(ties) == 1}


def forgotten_values(statement: str, sources: Sequence[str]) -> list[str]:
    """The values a statement shares with what is forgotten (``sources``: the user's request,
    the forgotten memories' statements): each run of capitalized Latin words ("Shanghai",
    "Northwind") and of two or more Han characters ("星河体育馆", "上海"), whole, that a
    source shows (``token_contained``)."""
    return list(dict.fromkeys(run for run in _runs(statement)
                              if any(token_contained(run, source) for source in sources)))


_NOT_TAIL: Final = re.compile(
    r"(?:,\s+(?:not|no\s+longer)|,?\s+(?:instead\s+of|rather\s+than))\s+(?:(?:the|a|an)\s+)?"
    r"(?P<value>[^\s,;，；][^,;，；]{0,63}?)(?:\s+(?:any\s?more|any\s+longer))?"
    r"\s*[.!?。]?\s*(?=[,;，；]|$)", re.IGNORECASE)
"""A tail that names what is not, or no longer, the case, up to the next comma or the end:
"..., not 14 Oak Street.", "..., not Dr. Okafor.", "..., no longer 4471.", "..., not 3K728
anymore." (the value without "anymore"), "... rather than Arkwright Media."."""
_CODE: Final = re.compile(
    r"[0-9A-Za-z]*[0-9][0-9A-Za-z]*(?:[ -][0-9A-Za-z]*[0-9][0-9A-Za-z]*)*")
"""A number or a code: runs of Latin letters and digits, each with a digit, joined by single
spaces or hyphens ("3K728", "07700 900 123", "555-0142")."""


def _code_key(code: str) -> str:
    return re.sub(r"[ -]", "", code).casefold()


_PLATE_PROVINCES: Final = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
"""The characters a Chinese licence plate starts with, one per province ("浙" of "浙A3K728")."""
_PROVINCE: Final = re.compile(rf"[{_PLATE_PROVINCES}][\s·•]?$")
"""A plate's province character right before its letter ("浙" of "浙A3K728", "浙 A3K728")."""
_DATE: Final = re.compile(
    r"(?<!\d)(?:(?P<year>\d{4})\s*[-/.年]\s*(?P<month>\d{1,2})\s*[-/.月]\s*(?P<day>\d{1,2})日?"
    r"|(?P<first>\d{1,2})\s*[-/.]\s*(?P<second>\d{1,2})(?:\s*[-/.]\s*\d{2,4})?)(?!\d)")
"""A date ("2019-01-05", "05/01/2019", "01-05", "2019年1月5日"): its numbers are a date's."""
_LETTERS: Final = re.compile(r"(?<![0-9A-Za-z])[A-Z]{1,3} $")
"""One to three capital letters and a space right before a number ("BA " of "BA 2490"): part
of its code, as "BA2490" writes it."""
_CODE_CONTRAST: Final = re.compile(
    r"(?:,\s*(?:not|no\s+longer)|\b(?:instead\s+of|rather\s+than))\s[^,;.，；。!?！？]*"
    r"|(?:不|没|非|以前|原来|之前|曾经|代替|取代|替代)[^,;，；、。！？!?\n]*", re.IGNORECASE)
"""A part of a text that names what is not, or no longer, the case (", not 6284", "不是6284
那个了", "以前那个尾号6284的号码"): where a code may be given by its last characters."""


@dataclass(frozen=True)
class _Code:
    """One number or code of a text (`_CODE`), read without its spaces and hyphens."""

    key: str
    """With the capital letters before it when its own characters are fewer than four ("ba249"
    of "BA 249", `_LETTERS`)."""
    groups: tuple[str, ...]
    core: str | None
    """A plate's part after its province and letter ("3k728" of "浙A3K728")."""
    dated: bool
    calendar: bool
    """Whether its date could be one, a month and a day in range (`_calendar`: "12-07"; never
    "40-47-84")."""
    lettered: str | None
    """Its key with the capital letters before it ("ba2490" of "BA 2490", `_LETTERS`)."""
    start: int
    """Where it starts in its text: its capital letters left out, but where its key holds
    them."""
    end: int


def _calendar(match: re.Match[str]) -> bool:
    """Whether a date (`_DATE`) could be one: a month of 1 to 12 and a day of 1 to 31, in
    either order when no year leads ("12-07", "2019-01-05"; never "40-47-84")."""
    if match["year"] is not None:
        return 1 <= int(match["month"]) <= 12 and 1 <= int(match["day"]) <= 31
    first, second = int(match["first"]), int(match["second"])
    return any(1 <= month <= 12 and 1 <= day <= 31
               for month, day in ((first, second), (second, first)))


def _dated(text: str, start: int, end: int) -> tuple[bool, bool]:
    """Whether ``text[start:end]`` lies in a date (`_DATE`), and whether that date could be
    one (`_calendar`)."""
    dates = [match for match in _DATE.finditer(text)
             if match.start() <= start and end <= match.end()]
    return bool(dates), any(_calendar(match) for match in dates)


def _plate_at(text: str, start: int) -> bool:
    """Whether a code starting at ``start`` follows a plate's province character."""
    return _PROVINCE.search(text[:start]) is not None


def _letters(text: str, start: int) -> int | None:
    """Where the capital letters that open the number at ``start`` begin (`_LETTERS`: "BA"
    of "BA 2490"); None when it opens with none."""
    if not text[start:start + 1].isdigit():
        return None
    found = _LETTERS.search(text, max(0, start - 5), start)
    return None if found is None else found.start()


def _codes(text: str) -> list[_Code]:
    """The numbers and codes of a text of four characters or more (`_CODE`), with the capital
    letters a number opens with (`_LETTERS`: "BA 249")."""
    found: list[_Code] = []
    for match in _CODE.finditer(text):
        key = _code_key(match.group())
        plate = (match.group()[:1].isalpha() and len(key) - 1 in (5, 6)
                 and _plate_at(text, match.start()))
        letters = _letters(text, match.start())
        lettered = None if letters is None else _code_key(text[letters:match.start()]) + key
        start = match.start()
        if len(key) < 4:
            # A short number is a code only with its letters ("BA 249"), as "BA249" is.
            if letters is None or lettered is None or len(lettered) < 4:
                continue
            key, lettered, start = lettered, None, letters
        dated, calendar = _dated(text, match.start(), match.end())
        found.append(_Code(
            key, tuple(group.casefold() for group in re.split(r"[ -]", text[start:match.end()])
                       if group),
            key[1:] if plate else None, dated, calendar, lettered, start, match.end()))
    return found


def _one_number(one: str, other: str) -> bool:
    """Whether two numbers of seven digits or more are one number, the longer written with a
    country code or the shorter with a trunk "0": the shorter ends the longer, or does
    without its "0" ("447700900123" of "+44 7700 900123" and "07700900123", "8613800001234"
    and "13800001234"). A number of the same length that shares only its last digits with
    it is another."""
    if not (one.isdigit() and other.isdigit()) or len(one) == len(other):
        return False
    shorter, longer = sorted((one, other), key=len)
    return len(shorter) >= 7 and (longer.endswith(shorter) or (
        shorter.startswith("0") and len(shorter) >= 8 and longer.endswith(shorter[1:])))


def _code_matches(value: str, text: str, *, ending: bool = False,
                  contrast: bool = False) -> list[tuple[_Code, int, int]]:
    """Each number or code of ``text`` that is one of ``value``'s (`code_forms`), with that one
    of the value and where ``text`` shows it."""
    wanted = _codes(value)
    if not wanted:
        return []
    given = _codes(text)
    tails = [] if contrast else [match.span() for match in _CODE_CONTRAST.finditer(text)]
    shown = {key for code in given for key in (code.key, code.lettered) if key is not None}
    found: list[tuple[_Code, int, int]] = []
    for code_of_text in given:
        key, groups, dated = code_of_text.key, code_of_text.groups, code_of_text.dated
        for code in wanted:
            # Its capital letters before it are part of it: "BA 2490" is "BA2490", and "LH
            # 2490" is another code.
            lettered = (code_of_text.lettered is not None
                        and code_of_text.lettered in (code.key, code.lettered)) or (
                key == code.lettered)
            other_letters = (code_of_text.lettered is not None and code.lettered is not None
                             and code_of_text.lettered != code.lettered)
            if (key == code.key and not other_letters) or lettered:
                # A date's numbers are a date's alone, but a number of six digits or more whose
                # date could be none is the same written either way ("40-47-84" of "404784").
                same = dated == code.dated or (
                    key.isdigit() and len(key) >= 6
                    and not (code_of_text.calendar or code.calendar))
            elif dated or code.dated:
                same = False  # A date's numbers are no other code's, nor another's a date's.
            elif _one_number(key, code.key):
                same = True
            elif code.key.endswith(key):
                # A shorter one is its ending only as two or more of its last groups, as a
                # plate's core, or where a contrast names it so and does not show it whole: one
                # last group alone is a number of its own ("0471" of "020 7946 0471").
                same = key == code.core or any(
                    key == "".join(code.groups[index:])
                    for index in range(1, len(code.groups) - 1)
                ) or ((contrast or any(
                    low <= code_of_text.start and code_of_text.end <= high
                    for low, high in tails)) and code.key not in shown
                    and code.lettered not in shown)
            elif key.endswith(code.key):
                # A longer one is it only as a plate whose core it is, or as a number whose
                # last groups are its groups; a tail may name a value by its last characters.
                same = ending or (
                    key[1:] == code.key and key[:1].isalpha() and len(code.key) in (5, 6)
                    and _plate_at(text, code_of_text.start)) or (
                    len(code.groups) > 1 and groups[-len(code.groups):] == code.groups)
            else:
                same = False
            if same:
                found.append((code, code_of_text.start, code_of_text.end))
                break
    return found


def code_forms(value: str, text: str, *, ending: bool = False,
               contrast: bool = False) -> list[str]:
    """The numbers and codes ``text`` shows that are one of ``value``'s, each as ``text``
    writes it. A number or code (`_CODE`, read without its spaces and hyphens, with the
    capital letters before it: "BA 2490" is "BA2490") of four characters or more is one when
    it is the same ("07700900123" of "07700 900 123"); when it is shorter, as two or more of
    the value's last groups ("900 123" of "07700 900 123"), as a plate's part after its
    province and letter ("3K728" of "浙A3K728"), or as its last characters where a contrast
    names it so ("1234" of "以前那个尾号1234的号码" for "13800001234") and shows no copy of it
    whole; when it is longer, as a plate whose part after its letter it is ("浙A3K728" for
    "3K728") or a number whose last groups are its groups ("07700 900 123" for "900 123"). Two
    numbers of seven digits or more and of different lengths are one when the longer ends with
    the shorter, or with the shorter of eight digits or more without its trunk "0"
    ("07700 900123" of "+44 7700 900123"). Any other code is another: "10471", "B0471" or "020
    7946 0471" for "0471", "0471" for "020 7946 0471", "70215" for "770215", "LH 2490" for "BA
    2490". A date's numbers are a date's alone: "2019-01-05" is no form of "0105", nor "12-07"
    of "1207"; a number of six digits or more whose date could be none is the same written
    either way ("40-47-84" of "404784"). With ``ending`` (a tail naming a value by its last
    characters, `contrasted_values`), any longer code ending with one of the value's is one
    too ("13800001234" for "1234"); with ``contrast`` (``text`` is a contrast), any shorter
    one."""
    return list(dict.fromkeys(text[start:end] for _, start, end in _code_matches(
        value, text, ending=ending, contrast=contrast)))


def code_spans(value: str, text: str, *, contrast: bool = False) -> list[tuple[int, int]]:
    """Where ``text`` shows one of ``value``'s numbers or codes (`code_forms`): the character
    range of each."""
    return [(start, end) for _, start, end in _code_matches(value, text, contrast=contrast)]


_OPENING_LETTERS: Final = re.compile(r"[A-Z]{1,3} (?=[0-9])")
"""The capital letters a code opens with and their space ("BA " of "BA 249", `_LETTERS`)."""


def is_code(value: str) -> bool:
    """Whether a value is one number or code (`_CODE`) of four characters or more, a shorter
    number with the capital letters before it ("BA 249", `_LETTERS`), or a plate
    ("浙A3K728"), which a text shows only as a whole number or code or one of its forms
    (`code_forms`): "0471" is no part of "020 7946 0471"."""
    text = value.strip()
    if text[:1] in _PLATE_PROVINCES:
        text = text[1:].lstrip(" ·•")
    letters = _OPENING_LETTERS.match(text)
    if letters is not None:
        number = text[letters.end():]
        return (_CODE.fullmatch(number) is not None and len(_code_key(number)) < 4
                and len(_code_key(text)) >= 4)
    return _CODE.fullmatch(text) is not None and len(_code_key(text)) >= 4


def codes_of(text: str) -> list[str]:
    """The codes of a text that mix letters and digits, four characters or more without their
    spaces and hyphens (`_CODE`): "B7Q2K9" of "粤B7Q2K9", "A7731"."""
    return list(dict.fromkeys(
        code for code in _CODE.findall(text) if len(_code_key(code)) >= 4
        and any(char.isalpha() for char in code)))


def contrasted_values(statement: str, sources: Sequence[str]) -> list[str]:
    """The values a statement's contrast tails (`_NOT_TAIL`) name, whole, that are forgotten
    too: a ``sources`` text shows the tail's value, or a code of it by its core
    (`code_forms`: "3K728" beside a forgotten "浙A3K728"), or the value shows a name or value
    of the statement a source shows (`forgotten_values`). "14 Oak Street" of "The user lives
    at 27 Pine Lane now, not 14 Oak Street." beside a forgotten "The user used to live at 14
    Oak Street.": a rewording takes the tail with its value, number and all, and the statement
    keeps what it says of its own ("The user lives at 27 Pine Lane now.")."""
    shared = forgotten_values(statement, sources)
    return list(dict.fromkeys(
        value for match in _NOT_TAIL.finditer(statement)
        if (value := match["value"].strip())
        and (any(token_contained(value, source) or code_forms(value, source, ending=True)
                 for source in sources)
             or any(token_contained(run, value) for run in shared))))


def replaced_values(statement: str, sources: Sequence[str], *,
                    departures: bool = True) -> list[str]:
    """The forgotten values ``statement`` names as replaced, which a rewording drops with the
    phrase naming them: those its contrast tails name (`contrasted_values`), and each name or
    value a source shows (`forgotten_values`) that it names as no longer the case before what
    holds now ("Qiming Kindergarten" of "The user's son no longer attends Qiming Kindergarten
    and has transferred to Lanwan Kindergarten.") or, with ``departures``, as the place left
    for another ("Shanghai" of "The user moved from Shanghai to Hangzhou last month.")."""
    found = list(contrasted_values(statement, sources))
    for value in forgotten_values(statement, sources):
        escaped = _escaped(value)
        if re.search(_ENDED_AND.format(value=escaped), statement, re.IGNORECASE) or (
                departures and re.search(_FROM_TO.format(value=escaped), statement,
                                         re.IGNORECASE)):
            found.append(value)
    return list(dict.fromkeys(found))


def kind_word(text: str) -> bool:
    """Whether ``text`` is only the word a place's or an organisation's name ends with ("幼儿园"
    of "启明幼儿园" and of "蓝湾幼儿园", `_KINDS`): two names of one kind share it, and it
    names neither."""
    return any(text == suffix for suffix, _ in _KINDS)


def named_part(value: str) -> str:
    """A place's or an organisation's name without the word that ends it ("Qiming" of "Qiming
    Kindergarten", "启明" of "启明幼儿园", `_KINDS`): what names it. The value itself when no
    such word ends it, or when what is left names nothing ("The Shop", "A Cafe")."""
    text = value.strip()
    for suffix, _ in _KINDS:
        if text.endswith(suffix) and len(text) > len(suffix):
            rest = text[:-len(suffix)].strip()
            return text if len(rest) < 2 or rest.casefold() in _STARTS | _LIGHT else rest
    return text


def name_kinds(text: str) -> list[str]:
    """The words the names of places or organisations ``text`` shows end with ("Kindergarten"
    of "Lanwan Kindergarten", "幼儿园" of "蓝湾幼儿园", `_KINDS`)."""
    return list(dict.fromkeys(
        next(suffix for suffix, _ in _KINDS if run.endswith(suffix) and len(run) > len(suffix))
        for run in (run.strip() for run in _runs(text)) if kind_of(run) is not None))


_HAN_LIGHT: Final = re.compile(
    r"上个月|下个月|这个月|上周|下周|今天|昨天|明天|去年|今年|明年|现在|以前|之前|后来|然后"
    r"|我们|你们|他们|她们|它们|[我你他她它的了在是和与从到把被就也都很这那]")
"""Han words that name no place or thing: no run of a quote's that a value aligns with."""


def aligned_values(statement: str, quotes: Sequence[str], sources: Sequence[str]) -> list[str]:
    """The Latin values of an English statement that its Han quote tells as the Han words the
    ``sources`` share ("Shanghai" of "The user moved from Shanghai to Hangzhou last month."
    quoting "我上个月从上海搬到了杭州", for "我在上海住过"): the statement's capitalized runs and
    the quote's Han runs taken in order, when there are as many of each; none otherwise."""
    latin = [run for run in _runs(statement) if run.isascii()]
    found: list[str] = []
    for quote in quotes:
        han = [run for run in (_HAN_LIGHT.sub(" ", part) for part in _HAN_RUN.findall(quote))
               for run in run.split() if len(run) >= 2]
        if not latin or len(han) != len(latin):
            continue
        shared = set(han_forms(quote, list(sources)))
        found += [value for value, run in zip(latin, han, strict=True)
                  if any(form in run or run in form for form in shared)]
    return list(dict.fromkeys(found))


def left_out(statement: str, reworded: str) -> list[str]:
    """The names and values of ``statement`` its rewording no longer shows: each run of
    capitalized Latin words and of two or more Han characters that ``reworded`` does not
    show ("Shanghai" of "The user moved from Shanghai to Hangzhou last month." against "The
    user moved to Hangzhou last month.", "Li Wei" of a parenthesis left out)."""
    return list(dict.fromkeys(run for run in _runs(statement) if not shows(reworded, run)))


def han_forms(text: str, forgotten: Sequence[str]) -> list[str]:
    """The runs of two or more Han characters a quote shares with what is forgotten: the
    longest common runs of ``text`` and each forgotten text ("上海" of "我上个月从上海搬到了杭州"
    and "我在上海住"), by which a Latin value of a statement ("Shanghai") is located in its
    Han quote. Light characters alone are no run."""
    found: set[str] = set()
    for other in forgotten:
        for start in range(len(text)):
            end = start + 2
            while end <= len(text) and _HAN_RUN.fullmatch(text[start:end]) and (
                    text[start:end] in other):
                end += 1
            run = text[start:end - 1]
            if len(run) >= 2 and _HAN_RUN.fullmatch(run) and not any(
                    run in longer and run != longer for longer in found):
                found = {kept for kept in found if kept not in run} | {run}
    return sorted(found, key=text.find)


def keeps_predicate(statement: str, reworded: str, tie: str | None = None,
                    removed: Collection[str] | None = None) -> bool:
    """Whether a rewording keeps what a statement says but the ``removed`` names and values
    (all it leaves out, `left_out`, when not given): every other word of it stays, but the
    words a value's phrase takes with it ("from", "at", "left ... and"); it adds no word but
    the tie standing for a name ("the user's wife"), which it shows whole. "The user's son
    walks the user's dog every morning." keeps "The user's son walks Biscuit every
    morning."; "The user's son walks every morning." does not, nor does "The user went
    hiking on Sunday." keep "The user went hiking with Tom and Lucas on Sunday." when only
    Tom is removed."""
    if reworded == existence_of(statement):
        return True
    gone = left_out(statement, reworded) if removed is None else removed
    removed = {token for run in gone for token in token_seq(run)}
    # One member left of two agrees with one ("is the user's friend" of "are the user's
    # friends"): the words are read with no number.
    old, new = _number_free(token_seq(statement)), _number_free(token_seq(reworded))
    tied: set[str] = set() if tie is None else _number_free(token_seq(tie))
    # A removed value may stand as its kind ("a primary school" of "晨光小学").
    tied |= {token for run in gone if (kind := kind_of(run)) is not None
             for token in token_seq(kind)}
    if tie is not None and not token_contained(tie, reworded):
        return False
    contrast = _CONTRASTING if any(
        re.search(_CONTRAST.format(value=_escaped(run)), statement, re.IGNORECASE)
        for run in gone) else frozenset[str]()
    # A clause naming a removed value as no longer the case leaves whole, its verb with it.
    contrast |= {token for run in gone for match in re.finditer(
        _ENDED_AND.format(value=_escaped(run)), statement, re.IGNORECASE)
        for token in token_seq(match.group())}
    # The one left out of a counted plural stands as "another" ("two colleagues, 小王 and
    # another").
    counted = {"another"} if any(_COUNTED.search(reworded[:match.start()]) for match in
                                 re.finditer(r"\s+and another\b", reworded)) else set[str]()
    return (old - removed - _REMOVABLE - contrast - {"s"} <= new
            and new - old <= tied | counted | {"the", "user", "s"})


def shows(text: str, value: str) -> bool:
    """Whether ``text`` shows ``value``: its tokens in order, a Latin word whole and in any
    case, a Han text by its characters (``token_contained``)."""
    return token_contained(value, text)


def value_pattern(value: str) -> re.Pattern[str]:
    """Where a text shows ``value``: a Han value anywhere, a Latin one as whole words in any
    case, even right beside Han characters ("叫Biscuit，")."""
    escaped = r"\s+".join(re.escape(part) for part in value.split())
    if all(own_token(token) for token in token_seq(value)):
        return re.compile(escaped)
    return re.compile(rf"(?<![0-9A-Za-z_-]){escaped}(?![0-9A-Za-z_-])", re.IGNORECASE)


def _bearer(words: str, description: str) -> bool:
    """Whether a possessive description ("paternal grandmother", "dog") describes the bearer
    of the tie ``description``: the same head noun, and nothing but its modifiers before it."""
    found, tie = words.casefold().split(), description.casefold().split()
    return bool(found) and found[-1] == tie[-1] and len(found) <= len(tie) + 2 and not (
        set(found) & {"and", "or", "with", "to", "of", "is", "was", "has", "had"})


_KIN: Final = frozenset({
    "mother", "mom", "mum", "father", "dad", "parent", "wife", "husband", "spouse", "partner",
    "fiance", "fiancee", "girlfriend", "boyfriend", "son", "daughter", "child", "kid",
    "sister", "brother", "sibling", "grandmother", "grandma", "grandfather", "grandpa",
    "grandparent", "grandson", "granddaughter", "aunt", "uncle", "cousin", "niece", "nephew",
    "friend", "roommate", "neighbor", "neighbour", "boss", "colleague", "coworker",
    "classmate", "dog", "puppy", "cat", "kitten", "bird", "parrot", "hamster", "rabbit",
    "horse", "turtle", "pet",
})
"""The ties to someone of the user's, and the kinds of animal, a rewording reads: its own
list, so a role word another rule reads as a tie ("doctor", "teacher") is none here."""


def _kin_word(word: str) -> bool:
    """Whether a word names a tie to someone or a kind of animal ("friend", "sons", "dog",
    `_KIN`; "children", and a Han kin word, too)."""
    word = word.casefold().removesuffix("'s").removesuffix("’s")
    return (word in _KIN or word == "children" or word in _HAN_KIN
            or (word.endswith("s") and word[:-1] in _KIN))


def kin_word(word: str) -> bool:
    """Whether a word names a tie to someone or a kind of animal ("儿子", "friend"): never a
    name (`_kin_word`)."""
    return _kin_word(word)


_HAD: Final = frozenset({
    "teacher", "coach", "tutor", "trainer", "doctor", "dentist", "therapist", "mentor",
    "manager", "landlord", "plant",
})
"""Who or what else the user, or one of the user's, has by a name besides a kin or a pet: a
"piano teacher", a "plant"."""


_EXISTENCE: Final = (_TIES[0], _TIES[3], _TIES[4])
"""The ties that also say the user has the one they name: "The user's cat is named Bob.",
"The user has a cat named Bob.", "Bob is the user's cat."."""


def existence_of(statement: str) -> str | None:
    """What a statement tying a name to a person, animal or plant of the user's says besides
    the name ("The user has a cat." of "The user's cat is named Bob.", "The user's daughter
    has a piano teacher." of "陈雅琴 is the user's daughter's piano teacher."): the existence a
    forget of the name keeps when no other memory says it; None for any other statement, or a
    tie naming no one the user has (`_KIN`, `_HAD`)."""
    for pattern in _EXISTENCE:
        match = pattern.fullmatch(statement.strip())
        if match is None:
            continue
        description = re.sub(r"[’]", "'", match["description"])
        head = description.split()[-1]
        if not (_kin_word(head) or head in _HAD) or description in _NOT_TIES:
            return None
        owner, _, had = description.rpartition("'s ")
        article = _article(had)
        if owner:
            return f"The user's {owner} has {article} {had}."
        return f"The user has {article} {had}."
    return None


def _parenthesis_of(statement: str, value: str, form: str) -> bool:
    """Whether ``form`` is the parenthesis right after ``value`` in ``statement``."""
    return any(statement[match.end():].lstrip().startswith(("(" + form, "（" + form))
               for match in value_pattern(value).finditer(statement))


def _without_name(text: str, name: str, description: str) -> tuple[str, set[str]] | None:
    """``text`` with every occurrence of a forgotten name taken out, with a parenthesis of
    another form of it right after it ("李伟 (Li Wei)"): dropped after a possessive
    description of its bearer ("The user's dog Biscuit is a corgi." gives "The user's dog is
    a corgi."), else replaced by the bearer's tie wherever the name stands ("林晓雨 is an
    architect." gives "The user's wife is an architect.", "walks Biscuit every morning"
    "walks the user's dog every morning"); None after a naming verb or a possessive
    description of someone else. Also the descriptions that now stand for the bearer ("the
    user's dog", "the user's wife"), lowercased. A name one of two that the tie names
    together in the plural leaves with its "and", what agreed with the two agreeing with the
    one left (`_one_of_two`): "Alice is the user's friend." of "Alice and Ben are the user's
    friends.", where the tie would say "the user's friend" twice."""
    escaped = _escaped(name)
    if re.search(rf"\b{re.escape(description)}s\b", text, re.IGNORECASE):
        found: str | None = None
        tied = _first_after_tie(text, escaped)
        if tied is None:
            return None
        if tied != text:
            found = tied
        elif re.search(_FIRST_OF_TWO.format(value=escaped), text, re.IGNORECASE):
            found = re.sub(_FIRST_OF_TWO.format(value=escaped), r"\1", text,
                           flags=re.IGNORECASE)
        elif re.search(_LAST_OF_TWO.format(value=escaped), text, re.IGNORECASE):
            found = _one_of_two(text, escaped)
            if found is None:
                return None
        if found is not None:
            # The tie it still names the one left by is what it keeps of the two.
            phrase = f"the user's {description}"
            return found, {phrase} if token_contained(phrase, found) else set()
    if _NEGATION.search(text) and any(text[:match.start()].strip()
                                      for match in value_pattern(name).finditer(text)):
        return None  # "not friends with Sofia anymore": no tie stands where it is denied
    ties: set[str] = set()
    for match in reversed(list(value_pattern(name).finditer(text))):
        before, after = text[:match.start()], text[match.end():]
        if (postmodifier := re.search(r"\b(?P<noun>[a-z]+)\s+(?:named|called)\s+$", before,
                                      re.IGNORECASE)) is not None and _bearer(
                postmodifier["noun"], description):
            # "adopted a rabbit named Clover": the bearer's own noun stays, its name leaves.
            text = before[:postmodifier.end("noun")] + after
            continue
        if _NAMING_VERB.search(before):
            return None
        if (parenthesis := _PARENTHESIS.match(after)) is not None:
            after = after[parenthesis.end():]
        possessed = _POSSESSED.search(before)
        if possessed is not None and not _bearer(possessed["words"], description):
            if _kin_word(possessed["words"].split()[-1]):
                return None  # someone else's description: "her friend Sofia" for a wife
            possessed = None  # the name is an object: "the user's son walks Biscuit"
        if possessed is not None:
            gap = possessed["gap"]
            if "," in gap:
                if not after.startswith(","):
                    return None
                after = after[1:]
            ties.add(possessed.group()[:-len(gap)].strip().lower())
            text = before[:len(before) - len(gap)] + after
            continue
        phrase = f"the user's {description}"
        owner = description.rpartition("'s ")[0]
        if owner and (kind := kind_of(name)) is not None and re.match(
                rf"\s*the user['’]s {re.escape(owner)}\b", before, re.IGNORECASE):
            # Its owner is the subject already: "The user's daughter attends a primary
            # school.", not "... attends the user's daughter's school.".
            text = before + kind + after
            continue
        ties.add(phrase)
        if not before.strip():
            phrase = phrase[:1].upper() + phrase[1:]
        text = before + phrase + after
    return text, ties


def _article(words: str) -> str:
    """The indefinite article before ``words``: "an uncle", "a university"."""
    if words.startswith(("uni", "use", "eu", "one")):
        return "a"
    return "an" if words[:1] in "aeiou" else "a"


def kind_of(value: str) -> str | None:
    """The kind a named place or organisation is of, by the word its name ends with ("a
    primary school" of "晨光小学", "a university" of "Fudan University", `_KINDS`); None when
    its name tells none."""
    text = value.strip()
    for suffix, kind in _KINDS:
        if text.endswith(suffix) and len(text) > len(suffix):
            return f"{_article(kind)} {kind}"
    return None


def _escaped(value: str) -> str:
    """A pattern of ``value`` inside a phrase rule, with a parenthesis of another form of it."""
    return value_pattern(value).pattern.removeprefix(r"(?<![0-9A-Za-z_-])").removesuffix(
        r"(?![0-9A-Za-z_-])") + r"(?:\s*[(（][^)）]{1,64}[)）])?"


_TIED_FIRST: Final = (r"(\b(?:the user['’]s|my|his|her|their)\s+(?P<noun>[a-z]+))s\s+"
                      r"(?:the\s+)?{value}\s+and\s+(?=(?P<other>{member})(?P<next>[\s.,;]|$))")
"""The forgotten value first of two a plural tie names ("the user's friends Tom and Lucas"):
the tie, made one, stays before the one left ("the user's friend Lucas")."""


def _first_after_tie(text: str, escaped: str) -> str | None:
    """``text`` without the forgotten value first of two after their plural tie
    (``_TIED_FIRST``), with a verb the two agreed with agreeing with the one left ("The
    user's friend Lucas is a hiker." of "The user's friends Tom and Lucas are hikers." is not
    written: its complement is plural); ``text`` itself when no such phrase names it; None
    when a word still speaks of two (``_TOGETHER``) or the tie is no kin word."""
    pattern = re.compile(_TIED_FIRST.format(value=escaped, member=_MEMBER), re.IGNORECASE)
    match = pattern.search(text)
    if match is None:
        return text
    if not _kin_word(match["noun"]) or _TOGETHER.search(text):
        return None
    found = text[:match.start()] + match.group(1) + " " + text[match.end():]
    after = found[match.start() + len(match.group(1)) + 1 + len(match["other"]):]
    verb = re.match(r"\s+(are|were|have)\b", after, re.IGNORECASE)
    if verb is not None:
        return None  # what the two are or have is no fact of one alone
    return found


def _one_of_two(text: str, escaped: str) -> str | None:
    """``text`` without the forgotten value last of two (``_LAST_OF_TWO``, ``escaped``), with
    what agreed with the two agreeing with the one left: "Alice is the user's friend." of
    "Alice and Ben are the user's friends.", "The user's friend Alice came ..." of "The user's
    friends Alice and Ben came ...". None when the rules cannot make it agree: a word still
    speaking of two ("both", "together", "they"), or a plural noun of no tie ("are doctors").
    """
    match = re.search(_LAST_OF_TWO.format(value=escaped), text, re.IGNORECASE)
    if match is None:
        return text
    before, after = text[:match.start()], text[match.end():]
    serial = _SERIAL.search(before)
    if serial is not None:
        # One of three or more: the two left go on together ("The user and 小王 went
        # together" of "The user, 小王 and 小李 went together").
        return before[:serial.start(1)] + " and" + before[serial.end(1):] + after
    counted = _COUNTED.search(before)
    if counted is not None:
        # A counted plural keeps its count, the one left out standing as "another" ("two
        # colleagues, 小王 and another" of "two colleagues, 小王 and 小李").
        return before + " and another" + after if _kin_word(counted["noun"]) else None
    plural = _PLURAL_SUBJECT.match(before)
    if plural is not None:
        # The plural tie is the subject and the two its complement: "The user's colleague is
        # 小王." of "The user's colleagues are 小王 and 小李.".
        if not _kin_word(plural["noun"]) or _TOGETHER.search(text):
            return None
        return (plural["owner"] + plural["noun"] + plural["gap"]
                + _SINGULAR[plural["verb"].lower()] + before[plural.end("verb"):] + after)
    owner = re.search(r"(\bthe user['’]s\s+)([a-z]+)s(\s+[^\s,.;]+)$", before, re.IGNORECASE)
    if not re.fullmatch(r"\s*(?:the user|(?:the user['’]s\s+[a-z]+\s+)?[^\s,.;]+)", before,
                        re.IGNORECASE):
        # The two are no subject ("with Alice and Ben"); a plural tie right before the one
        # left agrees with it ("with the user's colleague 小王").
        if owner is not None:
            if not _kin_word(owner.group(2)):
                return None
            before = before[:owner.start()] + owner.group(1) + owner.group(2) + owner.group(3)
        elif (bare := _PLURAL_BEFORE.search(before)) is not None and _kin_word(bare["noun"]):
            # So does a bare one ("The user and colleague 小王" of "... colleagues 小王 and 小李").
            before = before[:bare.start("plural")] + bare["noun"] + before[bare.end("plural"):]
        return before + after
    if _TOGETHER.search(text):
        return None
    verb = re.match(r"(\s+)(are|were|have)\b", after, re.IGNORECASE)
    first = re.match(r"\s+([a-z]+)", after, re.IGNORECASE)
    if verb is None and (first is None or not (
            first.group(1).lower() in _AGREE_ANY or first.group(1).lower().endswith("ed"))):
        return None  # a verb that agrees with two: "The user and Sofia go swimming"
    if verb is not None:
        after = verb.group(1) + _SINGULAR[verb.group(2).lower()] + after[verb.end():]
        complement = re.match(r"(\s+\w+\s+(?:the user['’]s\s+)?)([a-z]+)s\b", after)
        if complement is not None:
            if not _kin_word(complement.group(2)):
                return None
            after = complement.group(1) + complement.group(2) + after[complement.end():]
            # Each plural tie it goes on with agrees too: "colleague and best friend".
            clause = re.match(r"[^.,;]*", after)
            if clause is not None:
                words = clause.group()
                plural = [match for match in re.finditer(r"\b([a-z]+)s\b", words)
                          if _kin_word(match.group(1))]
                for match in reversed(plural):
                    words = words[:match.start()] + match.group(1) + words[match.end():]
                after = words + after[clause.end():]
    if owner is not None:
        if not _kin_word(owner.group(2)):
            return None
        before = before[:owner.start()] + owner.group(1) + owner.group(2) + owner.group(3)
    return before + after


def _number_free(tokens: Collection[str]) -> set[str]:
    """``tokens`` read with no number: "are" as "is", "friends" as "friend"."""
    return {_SINGULAR.get(token, token[:-1] if token.endswith("s") and _kin_word(token[:-1])
                          else token) for token in tokens}


_NO_VERB: Final = "|".join(sorted({
    "his", "hers", "its", "this", "theirs", "ours", "yours", "us", "others", "these", "those",
    "children", *(f"{word}s" for word in _KIN)}, key=len, reverse=True))
"""Words ending in "s" that are no verb: a determiner, a pronoun, a plural tie ("parents")."""
_ENDED_AND: Final = (r"\bno\s+longer\s+(?:[\w'’-]+\s+){{0,3}}?(?:the\s+)?{value}"
                     r"(?:\s+any\s?more|\s+any\s+longer)?,?\s+and\s+(?=(?-i:(?:now|also|"
                     rf"instead|currently)\s+)?(?!(?-i:{_NO_VERB})\b)"
                     r"(?-i:has|have|had|is|are|was|were|[a-z]+(?:ed|s))\b)")
"""A clause naming a value as no longer the case before the one saying what holds now, with
its "and" ("no longer attends Qiming Kindergarten and " of "The user's son no longer attends
Qiming Kindergarten and has transferred to Lanwan Kindergarten."): it leaves whole. The "and"
must open a clause, its verb right after it ("and has transferred", "and now lives"), never
join two names or a name and what follows ("with Marco and his parents")."""
_SUBJECT_GAP: Final = re.compile(
    rf"The user(?:['’]s [a-z-]+)?\s+(?:{_NO_VERB}|her|their|our|your|my|that|them|him|it|the|"
    r"a|an)\b")
"""A subject followed by no verb but a determiner, a pronoun or a plural tie ("The user his
parents."): what a rewording that cut the wrong words leaves."""
_CONTRAST: Final = (r"(?:,?\s+(?:not|rather than|instead of)|,\s+no longer)\s+"
                    r"(?:(?:the|a|an)\s+)?{value}(?:\s+(?:any\s?more|any\s+longer))?"
                    r"(?=[\s.,;]|$)")
"""A contrast tail naming the forgotten value as what is not, or no longer, the case ("lives in
Hangzhou, not Shanghai", "is a blue 卡罗拉, not a 思域", "is 8836 now, no longer 4471", "is
浙A9M105, not 3K728 anymore"): it leaves whole, and the statement says what it says
besides."""


def _without_value(text: str, value: str, *, keep_kind: bool = False,
                   apposed: bool = False) -> str | None:
    """``text`` without the phrase carrying a forgotten value, with a parenthesis of another
    form of it: a departure before an arrival ("moved to Hangzhou"), a former employer before a new
    one ("joined Fabrikam"), one of two ("with Tom and Lucas" gives "with Lucas"), or a place phrase
    after what it does not complete ("plays badminton at 星河体育馆"). A value the verb needs
    stands as its kind when its name tells one ("attends a primary school" of "attends
    晨光小学", `kind_of`). None for a negated statement, when the value stands anywhere else
    or its phrase completes a bare verb ("lives in Shanghai", "friends with Sofia"). A contrast
    tail naming it leaves whole ("lives in Hangzhou, not Shanghai"). With ``keep_kind`` (the
    request asks the name of its kind) it stands as its kind in its phrase too: "attends
    second grade at a primary school"."""
    escaped = _escaped(value)
    text = re.sub(_CONTRAST.format(value=escaped), "", text, flags=re.IGNORECASE)
    # A clause naming it as no longer the case leaves before the one saying what holds now:
    # "The user's son has transferred to Lanwan Kindergarten." of "... no longer attends
    # Qiming Kindergarten and has transferred to Lanwan Kindergarten.".
    text = re.sub(_ENDED_AND.format(value=escaped), "", text, flags=re.IGNORECASE)
    # A noun the value names leaves its name only: "works at a company named 青柠科技" says
    # "works at a company".
    text = re.sub(_NAMED_NOUN.format(value=escaped), r"\1", text, flags=re.IGNORECASE)
    if _NEGATION.search(text) or any(_NAMING_VERB.search(text[:match.start()])
                                     for match in value_pattern(value).finditer(text)):
        return None  # "is not friends with Sofia"; "A company is named 青柠科技."
    found = _first_after_tie(text, escaped)
    if found is None:
        return None
    text = found
    # With ``apposed`` (the request names the bearer by this name: "forget Amy's name"), a name
    # right after a possessive tie only identifies its bearer: it leaves, the tie stays ("The
    # user's coworker sits next to the user." of "The user's coworker Amy sits next to the
    # user."), as `_without_name` takes it out when the tie is known. A request by a tie
    # never does: the name may be a namesake's ("my cousin's name" beside "my colleague 志强").
    for match in reversed(list(value_pattern(value).finditer(text)) if apposed else []):
        possessed = _POSSESSED.search(text[:match.start()])
        if possessed is not None and _kin_word(possessed["words"].split()[-1]) and not (
                _NAMING_VERB.search(text[:match.start()])):
            gap, after = possessed["gap"], text[match.end():]
            if "," in gap:
                if not after.startswith(","):
                    return None
                after = after[1:]
            text = text[:match.start() - len(gap)] + after
    for form, keep in ((_FROM_TO, ""), (_LEFT_AND, ""), (_FIRST_OF_TWO, r"\1")):
        text = re.sub(form.format(value=escaped), keep, text, flags=re.IGNORECASE)
    found = _one_of_two(text, escaped)
    if found is None:
        return None
    text = found
    kind = kind_of(value)
    phrases: list[re.Match[str]] = [] if keep_kind and kind is not None else list(
        re.finditer(_PHRASE.format(value=escaped), text, re.IGNORECASE))
    for match in reversed(phrases):
        if _BARE.search(text[:match.start()]) and not re.match(r"\s+as\b",
                                                               text[match.end():]):
            # "works at 青柠科技 as a product manager" says what the user works as without it.
            if kind is None:
                return None
            continue  # it stands as its kind below
        text = text[:match.start()] + " " + text[match.end():]
    if kind is not None:
        found = value_pattern(value).pattern + r"(?:\s*[(（][^)）]{1,64}[)）])?"
        # An appositive kind goes with it: "the company 青柠科技" is "a company".
        head = re.escape(kind.split()[-1])
        text = re.sub(rf"(?:\b(?:the|a|an)\s+)?(?:{head}\s+)?" + found, kind, text,
                      flags=re.IGNORECASE)
    return None if value_pattern(value).search(text) else text


def reworded_statement(statement: str, *, names: Mapping[str, str],
                       values: Collection[str], kinded: Collection[str] = (),
                       apposed: bool = False,
                       ) -> tuple[str, str | None] | None:
    """Reword a statement so it shows none of the forgotten ``names`` (each with the tie its
    bearer is borne under) and ``values``, keeping everything else it says.

    Each name goes as `_without_name` takes it out, then each value as `_without_value`
    does, longest first. The result is canonical, differs from the statement, shows none of
    the names and values, ends on no dangling word ("moved from."), still says something of
    its own (two words beyond "the user" and the copula) and keeps the statement's predicate
    and its other arguments (`keeps_predicate`). A statement that only gives a name ("A
    person is named 林晓雨.", "林晓雨 is the user's wife.") is never reworded.

    Returns:
        The reworded statement with the one description that stands for the forgotten names'
        bearer in it ("the user's wife"), or None when it names none; None when no safe
        rewording exists or two descriptions would stand for names.
    """
    if any(pattern.fullmatch(statement.strip()) for pattern in _TIES):
        return None
    text = statement
    ties: set[str] = set()
    for name in sorted(names, key=len, reverse=True):
        found = _without_name(text, name, names[name])
        if found is None:
            return None
        text, used = found
        ties |= used
    if len(ties) > 1:
        return None
    for value in sorted(values, key=len, reverse=True):
        found = _without_value(text, value, keep_kind=value in kinded, apposed=apposed)
        if found is None:
            return None
        text = found
    text = re.sub(r"\s+([.,;!?])", r"\1", " ".join(text.split()))
    text = re.sub(r",\s*\.", ".", text)
    try:
        result = canonical_statement(text)
    except ValueError:
        return None
    if result == statement or _DANGLING.search(result) or _SUBJECT_GAP.match(result) or any(
            shows(result, value) for value in (*names, *values)):
        return None
    if len([token for token in token_seq(result) if token not in _LIGHT]) < 2 and not (
            _WHAT_THE_USER_IS.fullmatch(result)):
        return None
    result = result[:1].upper() + result[1:]
    tie = next(iter(ties), None)
    forms = [form for form in left_out(statement, result)
             if any(shows(form, value) or shows(value, form) or _parenthesis_of(
                 statement, value, form) for value in (*names, *values))]
    if not keeps_predicate(statement, result, tie, removed=[*names, *values, *forms]):
        return None
    return result, tie
