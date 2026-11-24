"""A request to forget a detail of something the user told, not the thing.

"忘掉我的猫是什么颜色的" over "我的橘猫叫bob" asks to forget the cat's colour: the cat and its
name stay. So do the dog and 旺财 for "忘掉我的狗是什么品种" over "我的金毛叫旺财", and the
user's job for "forget the name of the place I work" over "I work at Pinecrest Bakery as a
pastry chef.". Such a request asks a question rather than stating a claim, and the memories
show the asked value only as a modifier ("orange cat"), as the thing's own noun ("golden
retriever") or inside a phrase ("at Pinecrest Bakery"), so neither the forget judge nor the
value rules of ``core/forget_reword.py`` found it.

This module is the pure text step of such a forget:

- `detail_request` reads which detail a request asks: a colour or a breed of a thing of the
  user's, the place the user works, or which place of a kind ("哪个体育馆", "which school");
- `detail_values` finds the values of a colour or a breed a statement shows of that thing,
  from a lexicon of each value's English and Chinese forms (`COLOURS`, `BREEDS`), and
  `workplace_values` the organisation a statement says the user works at;
- `detail_reworded` writes the statement without the value: a colour leaves as a modifier
  ("The user's orange cat is named bob." gives "The user's cat is named bob."), a breed
  gives way to the kind of thing the request names ("The user's golden retriever is named
  旺财." gives "The user's dog is named 旺财."); `attribute_reworded` tells whether a new
  wording is exactly such a rewording, which the store's admission checks again;
- `quote_forms` locates the value's forms in a quote in any script ("橘" of "我的橘猫叫bob",
  "金毛" of "我的金毛叫旺财"), so a forget withholds those bytes alone.

Anything the rules do not read is left to the existing forget (fail closed): a value no
lexicon lists is not located, so its memory is taken whole, as before.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from quotemem.core.statement import canonical_statement, token_seq

Aspect = Literal["colour", "breed", "workplace", "place"]
"""What a detail request asks: the colour or the breed of a thing of the user's, the place
the user works, or which place of a kind."""


@dataclass(frozen=True)
class DetailRequest:
    """The detail a request asks to forget.

    Attributes:
        aspect: What it asks (`Aspect`).
        thing: The English noun of the thing whose colour or breed it asks ("cat", "dog");
            None for a place, or for a thing the request names by its name alone until
            `with_thing` reads it from the memories.
        name: The name the request calls the thing by ("Bob" of "forget that Bob is
            orange"), or None.
        claimed: The value a request stating it gives ("orange" of "forget that Bob is
            orange"): only that value is forgotten; empty for a question, which asks any.
    """

    aspect: Aspect
    thing: str | None = None
    name: str | None = None
    claimed: tuple[str, ...] = ()


_THINGS: Final = (
    (("dog", "dogs", "puppy"), ("狗狗", "小狗", "狗"), "dog"),
    (("cat", "cats", "kitten"), ("猫咪", "小猫", "猫"), "cat"),
    (("bird",), ("鸟",), "bird"), (("parrot",), ("鹦鹉",), "parrot"),
    (("hamster",), ("仓鼠",), "hamster"), (("rabbit", "bunny"), ("兔子",), "rabbit"),
    (("horse",), ("马",), "horse"), (("turtle",), ("乌龟",), "turtle"),
    (("car",), ("汽车", "车"), "car"), (("bike", "bicycle"), ("自行车",), "bike"),
)
"""The things of the user's whose colour or breed a request may ask: their English and Chinese
words, and the English noun a rewording names them by."""

COLOURS: Final = (
    (("orange", "ginger"), ("橘", "橙")), (("yellow",), ("黄",)), (("black",), ("黑",)),
    (("white",), ("白",)), (("grey", "gray"), ("灰",)), (("brown",), ("棕", "褐", "咖啡")),
    (("red",), ("红",)), (("blue",), ("蓝",)), (("green",), ("绿",)), (("pink",), ("粉",)),
    (("purple",), ("紫",)), (("silver",), ("银",)), (("golden", "gold"), ("金",)),
    (("cream",), ("奶油",)), (("tabby",), ("狸花",)), (("calico",), ("三花",)),
    (("tortoiseshell",), ("玳瑁",)), (("black and white", "black-and-white", "tuxedo"),
                                     ("黑白", "奶牛")),
)
"""Colours (and coat patterns) by their English and Chinese forms. A Chinese form counts in a
quote only right before 色 or the thing's own word ("橘猫", "橘色"), never inside another word
("橘子")."""

BREEDS: Final = (
    ("dog", ("golden retriever", "golden"), ("金毛",)),
    ("dog", ("labrador retriever", "labrador", "lab"), ("拉布拉多",)),
    ("dog", ("siberian husky", "husky"), ("哈士奇", "二哈")),
    ("dog", ("corgi",), ("柯基",)), ("dog", ("toy poodle", "poodle"), ("贵宾", "泰迪")),
    ("dog", ("teddy",), ("泰迪",)), ("dog", ("shiba inu", "shiba"), ("柴犬",)),
    ("dog", ("samoyed",), ("萨摩耶",)), ("dog", ("border collie",), ("边境牧羊犬", "边牧")),
    ("dog", ("german shepherd",), ("德国牧羊犬", "德牧")),
    ("dog", ("french bulldog",), ("法国斗牛犬", "法斗")), ("dog", ("bulldog",), ("斗牛犬",)),
    ("dog", ("chihuahua",), ("吉娃娃",)), ("dog", ("pomeranian",), ("博美",)),
    ("dog", ("schnauzer",), ("雪纳瑞",)), ("dog", ("beagle",), ("比格",)),
    ("dog", ("dachshund",), ("腊肠",)), ("dog", ("alaskan malamute", "malamute"), ("阿拉斯加",)),
    ("dog", ("bichon frise", "bichon"), ("比熊",)), ("dog", ("pug",), ("巴哥",)),
    ("dog", ("akita",), ("秋田",)),
    ("dog", ("chinese rural dog", "rural dog"), ("中华田园犬", "田园犬", "土狗")),
    ("cat", ("british shorthair",), ("英国短毛猫", "英短")),
    ("cat", ("american shorthair",), ("美国短毛猫", "美短")),
    ("cat", ("ragdoll",), ("布偶",)), ("cat", ("siamese",), ("暹罗",)),
    ("cat", ("persian",), ("波斯猫",)), ("cat", ("maine coon",), ("缅因",)),
    ("cat", ("scottish fold",), ("折耳",)), ("cat", ("sphynx",), ("无毛猫",)),
    ("cat", ("exotic shorthair",), ("加菲猫",)),
)
"""Breeds of each kind of animal by their English and Chinese forms, the longest form of each
first."""

_HAN_THING: Final = "|".join(sorted((word for _, words, _ in _THINGS for word in words),
                                    key=len, reverse=True))
_EN_THING: Final = "|".join(sorted((word for words, _, _ in _THINGS for word in words),
                                   key=len, reverse=True))
_ZH_COLOUR: Final = re.compile(
    rf"(?P<thing>{_HAN_THING})(?:是|的)?(?:什么|啥|哪种)(?:颜色|色)|(?P<thing2>{_HAN_THING})的颜色")
_EN_COLOUR: Final = re.compile(
    rf"\bwhat\s+colou?r\s+(?:my|our)\s+(?P<thing>{_EN_THING})\s+(?:is|was|are)\b"
    rf"|\b(?:my|our)\s+(?P<thing2>{_EN_THING})['’]s\s+colou?r\b"
    rf"|\bthe\s+colou?r\s+of\s+(?:my|our)\s+(?P<thing3>{_EN_THING})\b", re.IGNORECASE)
_ZH_BREED: Final = re.compile(
    rf"(?P<thing>{_HAN_THING})(?:是|的)?(?:什么|啥|哪个)品种|(?P<thing2>{_HAN_THING})的品种"
    rf"|(?:什么|啥)品种的(?P<thing3>{_HAN_THING})")
_EN_BREED: Final = re.compile(
    rf"\bwhat\s+breed\s+(?:my|our)\s+(?P<thing>{_EN_THING})\s+(?:is|was|are)\b"
    rf"|\b(?:my|our)\s+(?P<thing2>{_EN_THING})['’]s\s+breed\b"
    rf"|\bthe\s+breed\s+of\s+(?:my|our)\s+(?P<thing3>{_EN_THING})\b"
    rf"|\bwhat\s+(?:breed|kind)\s+of\s+(?P<thing4>dog|cat)\s+(?:I|we)\s+(?:have|had|own)\b",
    re.IGNORECASE)
_WORKPLACE: Final = re.compile(
    r"\b(?:the\s+)?(?:name\s+of\s+(?:the\s+)?)?(?:place|company|employer|firm)\s+"
    r"(?:where\s+|that\s+)?(?:I|we)\s+work(?:ed)?\b"
    r"|\bwhere\s+(?:I|we)\s+work(?:ed)?\b|\bwho\s+(?:I|we)\s+work(?:ed)?\s+for\b"
    r"|\bwhich\s+(?:company|place|employer|firm)\s+(?:I|we)\s+work(?:ed)?\b"
    r"|\bmy\s+(?:employer|workplace)\b"
    r"|在哪(?:里|儿)?(?:上班|工作)|(?:上班|工作)的(?:地方|单位|公司)"
    r"|(?:哪家|哪个)(?:公司|单位)(?:上班|工作)?", re.IGNORECASE)
"""A request for the place the user works: "the name of the place I work", "where I work",
"忘掉我在哪家公司上班"."""
_WHICH_PLACE: Final = re.compile(
    r"哪(?:个|家|所|间)(?:学校|小学|中学|大学|学院|幼儿园|医院|诊所|体育馆|健身房|球馆|餐厅|餐馆|饭店|银行|酒店|宾馆)"
    r"|\bwhich\s+(?:school|university|college|kindergarten|hospital|clinic|gym|sports\s+hall"
    r"|restaurant|bank|hotel)\b", re.IGNORECASE)
"""A request for which place of a kind: "忘掉我在哪个体育馆打球", "forget which school my son goes
to". Only a place's own word asks one: "which car I drive" does not."""
_FIRST_PERSON: Final = re.compile(r"我|\b(?:I|my|we|our)\b", re.IGNORECASE)
"""Who asks for a place speaks of themself; a stored memory speaks of "the user" or by a name
("Leo chose which school to attend.")."""


def _thing(words: str) -> str | None:
    """The English noun of a thing word of a request ("猫" and "kitten" give "cat")."""
    folded = words.casefold()
    for english, chinese, noun in _THINGS:
        if folded in english or words in chinese:
            return noun
    return None


def _first_group(match: re.Match[str]) -> str:
    return next(value for value in match.groupdict().values() if value)


_EN_VALUE: Final = r"(?:an?\s+)?(?P<value>[a-z][a-z -]{1,40}?)(?:\s+(?:one|colou?red))?"
_EN_CLAIMS: Final = (
    re.compile(rf"\b(?:my|our)\s+(?P<thing>{_EN_THING})\s+(?:is|was)\s+{_EN_VALUE}"
               r"\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(rf"(?<![\w'’])(?P<name>[A-Z][\w'’-]*)\s+(?:is|was)\s+{_EN_VALUE}\s*[.!]?\s*$"),
)
"""A request stating the value it forgets: "forget that my cat is orange", "forget that Bob is
orange", "forget that Max is a golden retriever"."""
_ZH_CLAIMS: Final = (
    re.compile(rf"我的?(?P<thing>{_HAN_THING})是(?:一只|只)?(?P<value>[一-鿿]{{1,6}}?)"
               r"(?:色的|色|的)?[\s。.!！]*$"),
    re.compile(r"(?:^|(?<=掉)|(?<=记)|(?<=了)|(?<=除)|(?<=\s))"
               r"(?P<name>[A-Za-z][\w'-]*|[一-鿿]{2,3})是(?:一只|只)?"
               r"(?P<value>[一-鿿]{1,6}?)(?:色的|色|的)?[\s。.!！]*$"),
)
"""The same in Chinese: "忘掉我的狗是金毛", "忘掉我的猫是橘色的", "忘掉团子是橘色的"."""


def _entry(value: str) -> tuple[Aspect, str | None, tuple[str, ...]] | None:
    """The lexicon entry a stated value is: its aspect, the kind of animal a breed is of,
    and every form of it; a colour's Chinese form may be followed by the thing ("橘猫")."""
    folded = " ".join(value.casefold().replace("-", " ").split())
    for kind, english, chinese in BREEDS:
        if folded in english or value in chinese:
            return "breed", kind, (*english, *chinese)
    for english, chinese in COLOURS:
        if folded in (form.replace("-", " ") for form in english) or any(
                value == form or (value.startswith(form) and _thing(value[len(form):]))
                for form in chinese):
            return "colour", None, (*english, *chinese)
    return None


def _claim(text: str) -> DetailRequest | None:
    """The colour or breed a request states of a thing of the user's (`_EN_CLAIMS`,
    `_ZH_CLAIMS`), by the thing's word or by its name."""
    for pattern in (*_EN_CLAIMS, *_ZH_CLAIMS):
        match = pattern.search(text)
        if match is None:
            continue
        entry = _entry(match["value"])
        if entry is None:
            continue
        aspect, kind, forms = entry
        groups = match.groupdict()
        thing = _thing(groups["thing"]) if groups.get("thing") else kind
        if groups.get("thing") and thing is None:
            continue
        if kind is not None and thing is not None and kind != thing:
            continue
        return DetailRequest(aspect, thing, groups.get("name"), forms)
    return None


_MEMORY: Final = re.compile(r"(?:the user|用户)\b", re.IGNORECASE)
"""How a stored memory's statement opens: it is never a request."""


def detail_request(words: str) -> DetailRequest | None:
    """The detail ``words`` ask to forget, or None for any other request.

    "忘掉我的猫是什么颜色的", "忘掉我的猫的颜色" and "forget what colour my cat is" ask the
    cat's colour, "忘掉我的狗是什么品种" and "forget my dog's breed" the dog's breed, "forget
    the name of the place I work" and "忘掉我在哪家公司上班" the place the user works,
    "忘掉我在哪个体育馆打球" which sports hall. A request stating the value asks that value of
    the thing it names, by its word or its name ("forget that my cat is orange", "forget that
    Bob is orange", "忘掉我的狗是金毛"). "forget my cat" and "forget my cat's name" ask
    none. A stored memory's statement ("The user's dog Biscuit is a corgi.", which the store
    reads beside the request among what a forget tells) asks nothing."""
    text = words.strip()
    if _MEMORY.match(text):
        return None
    for aspect, patterns in (("colour", (_ZH_COLOUR, _EN_COLOUR)),
                             ("breed", (_ZH_BREED, _EN_BREED))):
        for pattern in patterns:
            match = pattern.search(text)
            if match is not None and (thing := _thing(_first_group(match))) is not None:
                return DetailRequest("colour" if aspect == "colour" else "breed", thing)
    if (claim := _claim(text)) is not None:
        return claim
    if _FIRST_PERSON.search(text) and _WORKPLACE.search(text):
        return DetailRequest("workplace")
    if _FIRST_PERSON.search(text) and _WHICH_PLACE.search(text):
        return DetailRequest("place")
    return None


def detail_question(words: str) -> DetailRequest | None:
    """`detail_request` of a question alone. A stated colour or breed ("Bob is orange") is a
    claim like any other: what a forget of it tells is forgotten, and a memory stating it is no
    request, though its words are the same."""
    detail = detail_request(words)
    return None if detail is None or detail.claimed else detail


def attribute(detail: DetailRequest | None) -> bool:
    """Whether a detail is a colour or a breed: a value of the thing itself, whose memories
    keep the thing and its name when reworded."""
    return detail is not None and detail.aspect in ("colour", "breed")


def _english(form: str) -> re.Pattern[str]:
    """Where a text shows an English form, as whole words in any case."""
    escaped = r"[\s-]+".join(re.escape(part) for part in form.split())
    return re.compile(rf"(?<![A-Za-z0-9_'’-]){escaped}(?![A-Za-z0-9_-])", re.IGNORECASE)


def _breeds(thing: str | None) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    return [(english, chinese) for kind, english, chinese in BREEDS if kind == thing]


def _lexicon(detail: DetailRequest) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """The values a colour or breed request may find: each one's English and Chinese forms;
    of a request stating a value, that value alone."""
    found = list(COLOURS) if detail.aspect == "colour" else _breeds(detail.thing)
    if detail.claimed:
        found = [(english, chinese) for english, chinese in found
                 if set(detail.claimed) & {*english, *chinese}]
    return found


_NAMED: Final = (
    re.compile(r"The user's (?P<description>[a-z][a-z -]*?) (?:is )?(?:named|called) "
               r"(?P<name>[^,.;:]{1,64}?)\."),
    re.compile(r"The user has (?:a|an) (?P<description>[a-z][a-z -]*?) (?:named|called) "
               r"(?P<name>[^,.;:]{1,64}?)\."),
    re.compile(r"(?P<name>[^,.;:]{1,64}?) is the user's (?P<description>[a-z][a-z -]*?)\."),
)
"""The statements that name a thing of the user's: "The user's orange cat is named Bob."."""


def with_thing(detail: DetailRequest | None, statements: Sequence[str]) -> DetailRequest | None:
    """``detail`` with the thing a request names by its name alone ("Bob" of "forget that Bob
    is orange"), as the ``statements`` that name a thing of the user's so tell it ("The
    user's orange cat is named Bob."); None when none, or more than one kind, is told."""
    if detail is None or detail.thing is not None or detail.name is None:
        return detail
    kinds: set[str] = set()
    for statement in statements:
        for pattern in _NAMED:
            match = pattern.fullmatch(statement.strip())
            if match is not None and match["name"].strip().casefold() == detail.name.casefold():
                head = match["description"].split()[-1]
                kind = _thing(head) or next(
                    (kind for kind, english, _ in BREEDS
                     if any(match["description"].casefold().endswith(form)
                            for form in english)), None)
                if kind is not None:
                    kinds.add(kind)
    if len(kinds) != 1:
        return None
    return DetailRequest(detail.aspect, next(iter(kinds)), detail.name, detail.claimed)


def _thing_words(thing: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    for english, chinese, noun in _THINGS:
        if noun == thing:
            return english, chinese
    return (), ()


def _modifies(statement: str, start: int, end: int, detail: DetailRequest) -> bool:
    """Whether the value at ``start:end`` of a statement is the asked thing's: a colour right
    before the thing's noun or a breed of it ("orange cat", "orange tabby cat", "black
    corgi"), or after "is"/"are" of a statement opening with the thing ("The user's cat is
    orange."); a breed is the thing's anywhere."""
    if detail.aspect == "breed":
        return True
    english, chinese = _thing_words(detail.thing)
    nouns = [*english, *(form for forms, _ in _breeds(detail.thing) for form in forms),
             *chinese, *(form for _, forms in _breeds(detail.thing) for form in forms),
             *([detail.name] if detail.name else [])]
    after = statement[end:]
    words = "|".join(re.escape(noun) for noun in sorted(nouns, key=len, reverse=True))
    if re.match(rf"(?:[\s-]+colou?red)?\s*(?:[a-z-]+\s+){{0,2}}(?:{words})\b", after,
                re.IGNORECASE) or (words and re.match(rf"\s*(?:色的?)?(?:{words})", after)):
        return True
    before = statement[:start]
    return bool(re.search(rf"\b(?:{words})\s+(?:is|are|was|were)\s+(?:a\s+|an\s+)?$", before,
                          re.IGNORECASE))


def detail_values(statement: str, detail: DetailRequest | None) -> list[str]:
    """The forms of a colour or a breed of the asked thing a statement shows, as it writes
    them ("orange" of "The user's orange cat is named bob.", "golden retriever" of "The
    user's golden retriever is named 旺财.", "金毛" of "The user's 金毛 is named 旺财."),
    longest first; none for any other detail, or a value of another thing ("The user's cat
    sleeps on an orange blanket.")."""
    if detail is None or not attribute(detail):
        return []
    found: list[tuple[int, int, str]] = []
    for english, chinese in _lexicon(detail):
        for form in sorted((*english, *chinese), key=len, reverse=True):
            pattern = _english(form) if form.isascii() else re.compile(re.escape(form))
            for match in pattern.finditer(statement):
                if any(start < match.end() and match.start() < end for start, end, _ in found):
                    continue
                if _modifies(statement, match.start(), match.end(), detail):
                    found.append((match.start(), match.end(), match.group()))
    return [value for _, _, value in sorted(found)]


def _forms(value: str, detail: DetailRequest) -> tuple[str, ...]:
    """Every form, in any script, of the lexicon entry a value is one of."""
    folded = value.casefold()
    for english, chinese in _lexicon(detail):
        if folded in english or value in chinese or any(
                folded == form.replace("-", " ") for form in english):
            return (*english, *chinese)
    return (value,)


def quote_ranges(text: str, values: Sequence[str],
                 detail: DetailRequest | None) -> list[tuple[int, int]]:
    """Where a quote shows a form of ``values``, in any script, as character ranges: "橘" of
    "我的橘猫叫bob" for "orange", "金毛" of "我的金毛叫旺财" for "golden retriever". A Chinese
    colour counts only right before 色 or the thing's own word ("橘猫", "橘色"), never inside
    another word ("橘子"); a breed anywhere."""
    if detail is None or not attribute(detail):
        return []
    _, chinese_things = _thing_words(detail.thing)
    breed_words = [form for _, forms in _breeds(detail.thing) for form in forms]
    after = "|".join(re.escape(word) for word in sorted(
        (*chinese_things, *breed_words, "色", "毛"), key=len, reverse=True))
    found: list[tuple[int, int]] = []
    for value in values:
        for form in sorted(_forms(value, detail), key=len, reverse=True):
            if form.isascii():
                pattern = _english(form)
            elif detail.aspect == "colour":
                pattern = re.compile(rf"{re.escape(form)}(?=(?:{after}))")
            else:
                pattern = re.compile(re.escape(form))
            found += [(match.start(), match.end()) for match in pattern.finditer(text)
                      if not any(start <= match.start() and match.end() <= end
                                 for start, end in found)]
    return sorted(set(found))


def quote_forms(text: str, values: Sequence[str], detail: DetailRequest | None) -> list[str]:
    """The forms of ``values`` a quote shows (`quote_ranges`), as it writes them."""
    return list(dict.fromkeys(text[start:end]
                              for start, end in quote_ranges(text, values, detail)))


_LEAD: Final = re.compile(
    r"我们?家?(?:的|(?:有|养了?)(?:一?[只条个头匹])?)$")
"""What says the thing is the user's right before its breed ("我的" of "我的金毛叫旺财", "我养了一
只")."""


def existence_leads(text: str, reworded: str, values: Sequence[str],
                    detail: DetailRequest) -> list[tuple[int, int]] | None:
    """Where a Chinese quote says the user has the thing a rewording says only that of ("The
    user has a dog."), when the withheld breed was the thing's only word in it: "我的" of "我的
    金毛叫旺财", never "叫旺财", which tells nothing of having one. None for any other rewording,
    or a quote still showing the thing, which the ordinary support reads."""
    if detail.aspect != "breed" or detail.thing is None or text.isascii() or not re.fullmatch(
            rf"The user has (?:a|an) {re.escape(detail.thing)}\.", reworded):
        return None
    ranges = quote_ranges(text, values, detail)
    rest = "".join(char for index, char in enumerate(text)
                   if not any(start <= index < end for start, end in ranges))
    english, chinese = _thing_words(detail.thing)
    if any(word in rest for word in (*chinese, *english)):
        return None
    return [(match.start(), match.end()) for start, _ in ranges
            if (match := _LEAD.search(text[:start])) is not None]


_OWNERS: Final = (r"the user['’]s\s+", r"the user\s+(?:has|had|owns|owned|adopted|got|keeps|kept)"
                  r"\s+(?:a|an|one|two|some)\s+")


def owned(statement: str, detail: DetailRequest | None) -> bool:
    """Whether a statement speaks of the user's own thing that a colour or breed request asks
    ("The user's orange cat is named bob.", "The user has a golden retriever.", or a statement
    of the thing's name, "Bob is an orange cat."): never another's ("The user's sister has an
    orange cat.")."""
    if detail is None or not attribute(detail):
        return False
    english, chinese = _thing_words(detail.thing)
    nouns = [*english, *chinese, *(form for forms, others in _breeds(detail.thing)
                                   for form in (*forms, *others))]
    words = "|".join(re.escape(noun) for noun in sorted(nouns, key=len, reverse=True))
    if re.match(rf"\s*(?:{'|'.join(_OWNERS)})(?:[\w-]+\s+){{0,3}}?(?:{words})(?![\w'’])",
                statement, re.IGNORECASE):
        return True
    return detail.name is not None and re.match(
        rf"\s*{re.escape(detail.name)}\s+(?:is|was)\b", statement, re.IGNORECASE) is not None


def _article(words: str) -> str:
    return "an" if words[:1].casefold() in "aeiou" else "a"


def detail_reworded(statement: str, detail: DetailRequest | None,
                    values: Sequence[str]) -> str | None:
    """``statement`` without the colour or breed ``values`` of the asked thing, or None.

    A colour leaves as the modifier it is ("The user has an orange cat." gives "The user has
    a cat."); a breed gives way to the thing's noun ("The user's golden retriever is named
    旺财." gives "The user's dog is named 旺财.", "Bob is an orange cat." gives "Bob is a
    cat."). None when a value is what the statement says the thing is, ending its clause ("The
    user's cat is orange.": the memory is the detail itself), or when a value still shows."""
    if detail is None or not attribute(detail) or not values or detail.thing is None:
        return None
    text = statement
    for value in sorted(values, key=len, reverse=True):
        pattern = _english(value) if value.isascii() else re.compile(re.escape(value))
        found = list(pattern.finditer(text))
        if not found:
            return None
        for match in reversed(found):
            before, after = text[:match.start()], text[match.end():]
            if re.match(r"\s*(?:[.,;!?]|$)", after) and re.search(
                    r"\b(?:is|are|was|were)\s+(?:a\s+|an\s+)?$", before, re.IGNORECASE):
                return None  # "The user's cat is orange.": the value is the claim
            if detail.aspect == "colour":
                after = re.sub(r"^[\s-]+colou?red\b", "", after, flags=re.IGNORECASE)
                text = before + after.lstrip(" -")
                continue
            # A breed gives way to the thing's noun, unless the noun follows already.
            noun = detail.thing
            if re.match(rf"\s+{re.escape(noun)}s?\b", after, re.IGNORECASE):
                text = before + after.lstrip()
            else:
                text = before + noun + after
    # The article agrees with the word now after it ("an orange cat" gives "a cat").
    text = re.sub(r"\b(a|an|A|An)\s+([A-Za-z])", lambda match: (
        (_article(match.group(2)).capitalize() if match.group(1)[0] == "A"
         else _article(match.group(2))) + " " + match.group(2)), text)
    text = " ".join(text.split())
    try:
        result = canonical_statement(text)
    except ValueError:
        return None
    if result == statement or any(
            (_english(form) if form.isascii() else re.compile(re.escape(form))).search(result)
            for value in values for form in _forms(value, detail)):
        return None
    return result


def attribute_reworded(statement: str,
                       reworded: str) -> tuple[DetailRequest, list[str]] | None:
    """The colour or breed request, and the values of it, a rewording leaves out of a
    statement when it is exactly `detail_reworded` of them for some thing; None for any other
    rewording. What the store's admission reads in place of ``keeps_predicate`` for such a
    wording."""
    for _, _, thing in _THINGS:
        for aspect in ("colour", "breed"):
            detail = DetailRequest(aspect, thing)
            values = detail_values(statement, detail)
            if values and detail_reworded(statement, detail, values) == reworded:
                return detail, values
    return None


_AT: Final = re.compile(
    r"\b(?:works?|worked|working|employed|job|is\s+(?:an?\s+)?[a-z]+(?:\s+[a-z]+){0,2})\s+"
    r"(?:at|for|with)\s+(?:(?:the|a|an)\s+)?(?:[a-z]+\s+){0,2}?(?:(?:called|named)\s+)?"
    r"(?P<run>[A-Z][\w&'’.-]*(?:\s+[A-Z][\w&'’.-]*)*"
    r"|[㐀-鿿]{2,})")
"""The organisation a statement says the user works at: "The user works at Pinecrest
Bakery.", "The user is a pastry chef at Pinecrest Bakery.", "The user works at the company
青柠科技."."""


def workplace_values(statement: str) -> list[str]:
    """The organisations a statement of the user's says they work at, or for (`_AT`): what a
    request for the place they work asks, whatever its name tells of its kind."""
    if not re.match(r"\s*the user\b(?!['’]s)", statement, re.IGNORECASE):
        return []
    return list(dict.fromkeys(match["run"].rstrip(".") for match in _AT.finditer(statement)))


def detail_words(detail: DetailRequest | None, words: str) -> frozenset[str]:
    """The tokens of a request that name the asked thing, not its value ("猫" and "cat" of
    "我的猫是什么颜色的"): a rewording keeps them."""
    if detail is None or detail.thing is None:
        return frozenset()
    english, chinese = _thing_words(detail.thing)
    return frozenset({*english, detail.thing, *(token for word in chinese
                                                  for token in token_seq(word))}) & frozenset(
        {*token_seq(words), *english, detail.thing})
