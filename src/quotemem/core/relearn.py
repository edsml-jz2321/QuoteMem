"""When the user told a forgotten fact again as their own statement (``agent``).

In the ``agent`` Add mode (the complete in-chat forget rule, ``explicit-request-v2``) a fact
the user asked to forget is learned again when the user states it again, in their own
words, after the forget: the store follows what the user says now. Nothing else brings it
back:

- the assistant's text (an assistant message recounting the fact is still a restatement,
  denied by the tombstone and withheld);
- the user's forget request itself and the words it quotes ("please forget that I drink
  green tea"), and any sentence naming forgetting, erasing or the memory
  (``ingestion/relearn.asserts``);
- a question ("Do I drink green tea?", "我喝绿茶吗？");
- a user message without a date, or received or dated before the forget: a late batch
  from another tool;
- a later user sentence about something else: it lifts no forget of another fact, even
  when a memory written from it also states the forgotten one.

A memory the user's later words state in full also tells again the part of a forgotten fact
it shares (`retells`), even when the forgotten statement holds more than the user repeated:
"For the record, my cat is called Pixel." after the forget of "The user has a tabby cat."
learns "The user has a cat." again, and the breed stays forgotten. Such a memory claims no
word the user did not say again, so nothing only the forgotten statement holds is read back,
and the words are the user's own, of themself, true now (`states`): not a quotation or a
report of someone else, not another person's fact, not a fact that ended or a wish. Words
around the user's own (a discourse word first, a pronoun or another clause after) change
nothing. Every member relearned this way still goes to the model judge
(``ingestion/judge_gate.py``).

`told` reads one quote of one Evidence part: the part is the user's, it is dated
(``source_observed_at``), and every sentence the quote meets asserts something (``asserts``,
supplied by the caller). `restates` then decides, for one forgotten fact, whether that text
is the user stating it again: it was received after the forget (``ingested_at`` against the
forget's admission time, both the server's clock), it is dated after the forget too, and the
user's own words, speaking of whom the fact is about with its polarity, state the forgotten
fact (its content terms, or the part's vector near the forgotten statement's), or they state
in full a memory that tells part of it (`retells`).
Review (nomination's tombstone route, settlement's exact tombstone and the commit fence's
tombstone guard), the in-chat forget's relearn check and the interactive Search's forget
shadow all read it. The forgotten old Evidence stays withheld.
Bench modes never call it.

Clocks: the forget's time and ``ingested_at`` are the server's, so "after the forget" never
depends on a client clock. The part's date is the client's: requiring it after the forget
keeps a late batch of messages told before the forget out, and a client clock running
behind the server's only delays relearning by the skew (it never learns too early). A batch
from a client whose clock runs ahead of the server's, received after the forget but told
before it, is not told apart from a restatement.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from quotemem.core.forget_match import TIES, exclusion, named_ties, negated, ties
from quotemem.core.recall_agent import CALENDAR, GENERIC, STOP, content_terms, sentences
from quotemem.core.session_rollup import RESTATED_SHARE
from quotemem.core.vector_score import cosine, unit_vector

RESTATE_TERMS: Final = 2
"""Content terms of the forgotten statement the user's words must hold (all of a shorter
statement's), and at least ``RESTATED_SHARE`` of them (``session_rollup``'s test)."""
RESTATE_COSINE: Final = 0.6
"""The cosine of the user's Evidence part to the forgotten statement at which it restates
it in other words or another language (the complete forget's sentence floor)."""
_SINGLE: Final = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9](?![A-Za-z0-9])")
_CONTRACTED: Final = re.compile(r"(?:['’](?:s|t|m|d|ll|ve|re)|n['’]t)\b", re.IGNORECASE)
_HAN: Final = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD: Final = re.compile(r"[A-Za-z0-9]+")
_BREAK: Final = re.compile(r"\.{2,}|…|—|–|[,;:，；：、!?！？。]|\.")
"""Where a clause ends: punctuation, an ellipsis or a dash; a period only at a sentence's
end (`_clauses`)."""
_SENTENCE_BREAK: Final = frozenset("!?！？。")
_ABBREVIATIONS: Final = frozenset("mr mrs ms dr st jr sr prof vs etc mt no".split())
_SENTENCE_END: Final = frozenset(".!?。！？")
_FIRST_PERSON: Final = frozenset({"i", "me", "my", "mine", "myself"})
_OTHER_PERSON: Final = frozenset(
    "he she they him her hers his their theirs them himself herself themselves we us our "
    "ours ourselves".split())
_PRONOUN: Final = frozenset("he she it they him her its them".split())
_TIME: Final = frozenset(
    "recently yesterday today finally sadly luckily lately now then last this morning evening "
    "night week month year".split())
"""Time and attitude words that may come before the user; none points forward ("tomorrow",
"next", "tonight" are `_NOT_NOW`)."""
_HAN_PRONOUN: Final = re.compile(r"(?<![吉其])[他她它]|我们|咱们")
_DISCOURSE: Final = frozenset(
    "actually btw by way so well oh ok okay anyway anyways also just know fyi record honestly "
    "honest really still now yes yeah hey hi um uh too plus again got fact let truth needless "
    "like".split())
_HAN_DISCOURSE: Final = re.compile(
    r"顺便说一下|顺便说|顺便|说一下|我跟你说过|我跟你说|我跟你讲|跟你说过|跟你说|跟你讲|告诉你|说真的|说实话|老实说|老实讲|坦白说|"
    r"坦白讲|不瞒你说|换句话说|怎么说呢|说起来|就像我说的|正如我说的|再说|其实|对了|另外|还有|而且|然后|"
    r"所以|不过|但是|"
    r"补充一下|嗯|哦|啊|那个")
_LABELS: Final = frozenset(
    "correction update note ps p.s fyi edit reminder addendum info context background "
    "important btw also".split())
_HAN_LABELS: Final = frozenset("更正 更新 补充 备注 注 附注 提醒".split())
_SPEAKER: Final = re.compile(
    r"^\s*([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)?|[\u3400-\u9fff]{1,4})\s*[:：]")
_QUOTE: Final = re.compile(r"[\"“](.+?)[\"”]|「(.+?)」|『(.+?)』|«(.+?)»")
_ASIDE: Final = re.compile(
    r"\b(?:as|like)\s+i(?:\s+have|['’]ve)?\s+(?:said|told\s+you|mentioned)(?:\s+before)?\b|"
    r"\bi(?:\s+have|['’]ve)?\s+(?:said\s+it|told\s+you)\s+before\b|"
    r"\bif\s+you\s+(?:want|wanna|care)\s+to\s+know\b|"
    r"\bi(?:\s*['’]d|\s+would)\s+like\s+you\s+to\s+know(?:\s+that)?\b|"
    r"\bi\s+want(?:ed)?\s+you\s+to\s+know(?:\s+that)?\b|"
    r"\bjust\s+so\s+you\s+know\b|\bfor\s+your\s+information\b|"
    r"\bi(?:\s+am|['’]m)\s+telling\s+you(?:\s+again)?\b|\blet\s+me\s+tell\s+you\b|"
    r"\btruth\s+be\s+told\b|\bto\s+tell\s+(?:you\s+)?the\s+truth\b|\bneedless\s+to\s+say\b|"
    r"\bi\s+can\s+(?:confirm|tell\s+you|say)(?:\s+that)?\b",
    re.IGNORECASE)
"""Words the user says to the listener, which state no fact: "as I told you", "like I said",
"if you want to know", "I want you to know that", "let me tell you", "truth be told". Only
the user's own: "Alice wanted you to know that…" is Alice's report."""
_REPORT_VERB: Final = re.compile(
    r"\b(?:told|tells|tell|said|says|say|mentioned|mentions|claims|claimed|thinks|thought|"
    r"believes|reckons|heard|explained|wrote|texted|confirmed)\b|告诉|提到|提过|"
    r"(?<![小学传演解听据])(?:说|讲)(?![明服话一])", re.IGNORECASE)
"""A word reporting what someone said or thought (`_report_tag`)."""
_FILLER: Final = frozenset(
    "me us you him her them it so that this what yesterday today earlier before again too "
    "once already last week time then recently now just also".split())
"""Words a report tag may hold besides its reporter and its verb ("…, Alice told me
yesterday", "…, my vet said so")."""
_REPORTER: Final = frozenset("he she they him her them you".split())
"""Pronouns that may be a report tag's reporter, besides person nouns and names: anyone
but the user ("you" too: "As you said, …" is the listener's account)."""
_SELF_SUBJECT: Final = frozenset({"i", "we"})
"""The user as the subject of a report verb: "I said", "I told Alice", "like I said"."""
_HAN_FILLER: Final = re.compile(
    r"我们|我|你|的|过|了|呢|啊|吧|是|这|那|也|都|就|还|又|昨天|今天|之前|刚才")
_ACCORDING: Final = re.compile(r"\baccording\s+to\b(?!\s+me\b)|据说|听说", re.IGNORECASE)
_HAN_THING: Final = re.compile(r"[这那][家个本只件部条座][\u4e00-\u9fff]{0,4}")
"""A Han demonstrative phrase (这家店, 那本书): a thing, not a reporter."""
_HAN_REPORT: Final = re.compile(
    r"(?<![我小学传演解听据])(?:说|讲)(?![明服话一])|(?<!我)(?:告诉|提到|提过)|听说|据说")
"""A Chinese report in the clause carrying the claim (小王说我的猫, 张老师告诉我); 说 inside a
word (小说, 说明, 学说, 传说, 演说, 说服, 说话) and the user's own 我说 are none."""
_NOT_NOW: Final = re.compile(
    r"\b(?:said|says|say|saying|told|tells|think|thinks|thought|believe|believes|heard|hear|"
    r"according|apparently|reportedly|claims?|claimed|mentioned|wish|wished|might|would|"
    r"could|if|maybe|perhaps|hope|hoping|want|wants|wanted|planning|plan\s+to|going\s+to|"
    r"gonna|someday|will|shall|should|can|must|tomorrow|tonight|next(?!\s+to\b)|later|soon|"
    r"wonder|wondered|wondering|whether|dream|dreams|dreamed|dreamt|dreaming|imagine|"
    r"imagined|pretend|suppose|miss|missed|missing)\b|\b[A-Za-z]+['’](?:d|ll)\b|"
    r"觉得|认为|以为|如果|要是|也许|可能|希望|打算|明天|下周|下个月|明年|以后|将来|想|梦",
    re.IGNORECASE)
"""In the words carrying the claim (`_span`: its subject up to its last claimed word): a
report or opinion ("I think", "told"), or a fact that is not real yet or at all ("wish",
"might", "if", "will", "should", "can", "tomorrow", "next", "wonder", "dream", "miss",
如果, 明天, 想). After the claim it tells something else ("My cat is called Pixel and she
can open doors."); a word the memory claims itself ("Miss Pixel") and "next to" are none."""
_ENDED: Final = re.compile(
    r"\b(?:used\s+to|no\s+longer|any\s?more|stopped|stopping|quit|gave\s+(?:away|up)|died|"
    r"dead|passed\s+away|lost|former|formerly|sold|ran\s+away)\b|\bex-|"
    r"\b(?:gave|give|gives|given|giving|lent|handed)\b(?:\s+[\w'’]+){0,3}?\s+(?:to|away)\b|"
    r"以前|曾经|不再|去世|死了|没了|送人|送给",
    re.IGNORECASE)
"""In the words carrying the claim: a fact that ended or went away ("used to", "I sold my
cat", "I gave my cat to Alice", "I stopped seeing my therapist", 以前)."""
_GONE: Final = re.compile(
    r"\b(?:died|dies|dead|passed\s+away|ran\s+away|is\s+gone|went\s+missing|no\s+longer|"
    r"any\s?more)\b|去世|死了|没了|走丢", re.IGNORECASE)
"""After the words carrying the claim: the claimed one itself ended ("she died", "it ran
away", 它去世了), told only when it is the subject there (`_ended`)."""
_GIVEN: Final = re.compile(
    r"\b(?:lost|sold|sell|sells|gave|give|gives|giving|lent|handed|rehomed)\b(?:\s+(?:up|away))?"
    r"\s+(?:(?:her|him|it|them)\b(?!\s+[a-z]+(?:\s|$))|(?:my|our|the)\s+(\w+))",
    re.IGNORECASE)
"""After the words carrying the claim: the claimed one given up as an object ("I sold
her", "I lost my cat"); "she lost a tooth" and "I sold my car" give up something else."""
_HEDGE: Final = re.compile(
    r"(?:i\s+(?:think|guess|believe|suppose|reckon)|i['’]m\s+not\s+sure|not\s+sure|maybe|"
    r"probably|perhaps|apparently|reportedly|if\s+i\s+remember(?:\s+(?:correctly|right))?|"
    r"if\s+i['’]m\s+not\s+mistaken|if\s+memory\s+serves|or\s+so\s+i\s+think|"
    r"我觉得|好像|大概|可能吧|应该吧|如果我没记错的话?)", re.IGNORECASE)
"""A clause, or the words after a claim, that only hedge it ("…, I think.", "…Pixel I
think", "…, if I remember correctly", "I have a cat apparently"): the user is not sure, so
nothing is told again."""
_UNSURE: Final = re.compile(r"\b(?:probably|guess|apparently|presumably)\b|好像|大概",
                            re.IGNORECASE)
"""A hedge inside the words carrying the claim ("My cat is probably called Pixel.", "I
guess my cat…", 我的猫好像叫Pixel)."""
_FORWARD: Final = re.compile(
    r"\b(?:tomorrow|tonight|next\s+(?:week|month|year|time)|later|soon|someday)\b|"
    r"明天|下周|下个月|明年|以后|将来", re.IGNORECASE)
"""A forward time after the claim in its own part of the clause ("I get a cat tomorrow")."""
_NAME_LINK: Final = re.compile(
    r"\b(?:called|call|calls|named|name|names|known\s+as|goes\s+by)\b|叫|名字|取名|起名",
    re.IGNORECASE)
"""A word linking a noun and its name in the user's words (`_states_in`)."""
_NAMING_MEMORY: Final = re.compile(
    r"\b(?:called|named)\s+(?-i:[A-Z])|\bname\s+is\b", re.IGNORECASE)
"""A memory naming someone or something ("…is called Pixel", "…is named Pixel")."""
_COPULA: Final = re.compile(r"\s+is\s+|\s*是\s*", re.IGNORECASE)
"""An "is" between a noun and its name ("My cat is Pixel.", 我的猫是Pixel)."""
_MINE: Final = re.compile(r"\s+is\s+my\s+|\s*是我的", re.IGNORECASE)
"""An "is my" from a name to the user's noun ("Pixel is my cat.", Pixel是我的猫)."""
_NAME_FIRST: Final = re.compile(r"\b((?-i:[A-Z])[A-Za-z0-9]*)(?:\s+is\s+(?=my\s)|\s*是(?=我的))",
                                re.IGNORECASE)
"""A name a clause opens on, telling the user's noun ("Pixel is my cat.", Pixel是我的猫)."""
_OF: Final = re.compile(r"(?:['’]s|['’](?!s))(?!\s+name\b)|的(?!名字)", re.IGNORECASE)
"""A possessive turning a noun or a name to something else ("Pixel's friend", 猫的兽医); one
before a name word owns the name ("my cat's name", 猫的名字)."""
_NO: Final = re.compile(r"\b(?:no|without)\s+(?:[a-z]+\s+){0,2}?([a-z]+)", re.IGNORECASE)
"""A bare "no" before a word the memory claims negates the claim ("I have no cat."), which
``forget_match.negated`` does not read; before another noun it does not ("no tail")."""
_NAMING: Final = re.compile(r"\bcalled\s+(?-i:[A-Z])")
"""A statement naming someone or something ("is called Pixel")."""
_PARENTHETICAL: Final = frozenset("unlike like except including besides even".split())
"""Words opening an aside between commas that claims nothing ("My cat, unlike Alice's, is
called Pixel.")."""
_CONJUNCTION: Final = re.compile(r"\s(?:and|but|while|whereas)\s|和|而且|并且|但是", re.IGNORECASE)
_JOINED: Final = re.compile(r"(?:and|but|while|whereas|so|then)\s+|而且|并且|但是|所以|然后",
                            re.IGNORECASE)
_POSSESSIVE: Final = re.compile(r"\b[A-Za-z]+['’]s\s+([A-Za-z]+)")
_HAN_TIE: Final = re.compile("|".join(sorted((word for word in TIES if not word.isascii()),
                                             key=len, reverse=True)))
_BEINGS: Final = frozenset(
    "stray kitty kitten pup puppy animal creature vet doctor dentist nurse teacher professor "
    "manager landlord landlady neighbour neighbor roommate classmate mate buddy guy man men "
    "woman women lady girl boy baby toddler client customer owner stranger someone somebody "
    "person people family coach fiance fiancee mouse rat fish snake cow pig sheep goat fox "
    "squirrel deer lion tiger bear monkey duck chicken frog lizard".split())
"""Person and animal nouns besides `TIES` (`_beings`)."""
_HAN_BEINGS: Final = re.compile(r"兽医|医生|老师|房东|同学|经理|领导|宝宝|小孩|别人|有人|动物")
_PLACE: Final = frozenset("in at on from near around since during after before over".split())
"""Prepositions opening a place or time phrase, which may come before the user ("In Hangzhou
I adopted a cat", "After work I…")."""
_AT_PLACE: Final = frozenset({"in", "at", "to", "near"})
"""The only words after which a capitalized word is read as a place ("in Hangzhou"); after
any other ("from Alice") it is a name (`_names`)."""
_HAN_OPENING: Final = re.compile(
    r"今天|昨天|前天|最近|去年|今年|前年|上周|上个月|这周|刚才|刚刚|现在|后来|那时候?|上次|当时|"
    r"在[\u4e00-\u9fff]{1,6}$")
"""Han time words and place phrases (在杭州) that may come before 我."""
_HAN_LINKING: Final = re.compile(r"叫做|名叫|名字叫|叫|是|也|都|就|还|又|现在|一直")
"""Han words a clause without a subject of its own may hold before its claim (叫Pixel)."""
_NOT_NAMES: Final = (STOP | _DISCOURSE | _LABELS | _OTHER_PERSON | _PRONOUN | _FIRST_PERSON
                     | _TIME | {word.casefold() for word in CALENDAR})
"""Capitalized words that name no person (`_names`)."""
_HAN_HAVING: Final = r"(?:家)?(?:有|养了|养着|养的|领养了|领养的|收养了|收养的|买了)?"
_HAN_COUNT: Final = r"(?:[一二两三四五六七八九十几\d]*[只个条位])?"
_OWNER: Final = re.compile(rf"(?:我(?:的|家的|家养的|{_HAN_HAVING}{_HAN_COUNT})|\bmy\s*)$",
                           re.IGNORECASE)
"""What may come right before a Han person or animal word of the user's (`_owned`): 我的,
我家, 我有, 我养了, 我养着, 我领养了, 我家有 and the like, with a count, or "my"."""
_OWNED_BY: Final = re.compile(
    r"(?:\bmy|\b(?:have|has|own|owns|got|adopted|rescued|raise|raises|raised|keep|keeps|"
    r"bought|(?:have|has|['’]ve)\s+had|live\s+with|lives\s+with)\s+"
    r"(?:(?:a|an|one|two|three|four|five|six|\d+|some|another)\s+)?)\s*(?:[a-z]+\s+){0,2}$|"
    rf"我(?:的|家的|家养的|{_HAN_HAVING}{_HAN_COUNT})\s*$",
    re.IGNORECASE)
"""What must come before the first mention of a person or animal the memory says is the
user's, in the user's words: "my", or a verb of having ("have", "own", "got", "adopted",
"rescued", "raise", "keep", "bought", "have had", "live with") with a count and up to two
adjectives, or 我的, 我有, 我养了, 我家有 and the like (`_owns`). "I saw a cat", "I need a
cat" own none."""
_GIVING: Final = re.compile(
    r"\b(?:gave|give|gives|giving|given|sold|sell|lent|handed|sent|passed|bought|got)\b",
    re.IGNORECASE)
"""Verbs of giving or getting for someone, before a claimed noun (`_for_someone`)."""
_FOR_SOMEONE: Final = re.compile(r"\b(?:for|to)\s+(?:(?:my|the|a|an|his|her|their)\s+)?(\w+)",
                                 re.IGNORECASE)
"""A person the claimed noun is for or goes to ("I bought a cat for Alice")."""
_HAN_NAME: Final = ("名字", "名")
_SOMEONES: Final = re.compile(
    r"\b(?!(?:it|she|he|that|there|what|here|who|where|let)['’])[A-Za-z]+['’]s\b", re.IGNORECASE)
"""A possessive in a subject: someone's ("Alice's", "my landlord's"), not "it's" or "she's"."""


class ToldPart(Protocol):
    """The Evidence fields `told` reads (an ``EvidenceRecord``, a ``SourceRowV2``)."""

    @property
    def role(self) -> str | None: ...

    @property
    def source_observed_at(self) -> int | None: ...

    @property
    def ingested_at(self) -> int: ...

    @property
    def byte_start(self) -> int: ...

    @property
    def text(self) -> str: ...

    @property
    def embedding(self) -> Sequence[float]: ...

    @property
    def embedding_space(self) -> str: ...


type Asserts = Callable[[str], bool]
"""Whether one sentence of the user's states something (``ingestion/relearn.asserts``)."""


@dataclass(frozen=True, slots=True)
class Told:
    """The user's own statement a quote, or every quote of a support group, makes.

    Attributes:
        text: The user sentences the quotes meet, in order.
        received: When the server received the earliest of their parts.
        dated: The earliest of their parts' own dates (the client's clock).
        vectors: Each part's vector and its embedding space.
    """

    text: str
    received: int
    dated: int
    vectors: tuple[tuple[tuple[float, ...], str], ...] = ()


@dataclass(frozen=True, slots=True)
class Forgotten:
    """One forgotten fact as `restates` compares a later statement with it.

    Attributes:
        statement: The forgotten statement (a tombstone's, or the user's forget request).
        at: When the forget was admitted (the server's clock, epoch milliseconds).
        vector: The forgotten statement's vector, or empty.
        space: That vector's embedding space.
    """

    statement: str
    at: int
    vector: tuple[float, ...] = ()
    space: str = ""


def told(part: ToldPart, byte_start: int, byte_end: int, asserts: Asserts) -> Told | None:
    """The user's statement the quote ``[byte_start, byte_end)`` of a part makes: when the
    part is the user's, dated, and every sentence the quote meets asserts something; None
    otherwise (the assistant's, an undated part, a forget request, a question)."""
    if part.role != "user" or part.source_observed_at is None:
        return None
    text = part.text
    met: list[str] = []
    for start, end in sentences(text):
        first = part.byte_start + len(text[:start].encode("utf-8"))
        last = part.byte_start + len(text[:end].encode("utf-8"))
        if first < byte_end and byte_start < last:
            met.append(text[start:end])
    if not met or not all(asserts(sentence) for sentence in met):
        return None
    vectors = ((tuple(part.embedding), part.embedding_space),) if any(part.embedding) else ()
    return Told(" ".join(met), part.ingested_at, part.source_observed_at, vectors)


def part_told(part: ToldPart, asserts: Asserts) -> Told | None:
    """`told` of a whole part."""
    return told(part, part.byte_start, part.byte_start + len(part.text.encode("utf-8")),
                asserts)


def group_told(quotes: Iterable[tuple[ToldPart | None, int, int]],
               asserts: Asserts) -> Told | None:
    """The user's statement a support group makes: every quote is `told` (its part known),
    and the group is received and dated when its earliest quote is; else None."""
    found: list[Told] = []
    for part, start, end in quotes:
        quote = None if part is None else told(part, start, end, asserts)
        if quote is None:
            return None
        found.append(quote)
    if not found:
        return None
    return Told(" ".join(quote.text for quote in found),
                min(quote.received for quote in found), min(quote.dated for quote in found),
                tuple(dict.fromkeys(vector for quote in found for vector in quote.vectors)))


def after(statement: Told, at: int) -> bool:
    """Whether a statement came after a forget admitted at ``at``: received after it (the
    server's clock) and dated after it."""
    return statement.received > at and statement.dated > at


def _terms(text: str) -> frozenset[str]:
    """A text's `content_terms`, a plural or third-person ``s`` dropped from a Latin word
    ("drinks", "lives"), so the user's "I drink" meets the statement's "drinks"."""
    return frozenset(term[:-1] if len(term) > 3 and term.isascii() and term.endswith("s")
                     and not term.endswith("ss") else term for term in content_terms(text))


def says(statement: Told, fact: str, vector: Sequence[float] = (), space: str = "") -> bool:
    """Whether the user's words state ``fact``: they hold ``RESTATE_TERMS`` of its content
    terms (all of a shorter one's) and ``RESTATED_SHARE`` of them, or a part of theirs lies
    at ``RESTATE_COSINE`` or more to ``vector`` in the same space."""
    wanted = _terms(fact)
    shared = _terms(statement.text) & wanted
    if shared and len(shared) >= min(RESTATE_TERMS, len(wanted)) and (
            len(shared) >= RESTATED_SHARE * len(wanted)):
        return True
    if not any(vector):
        return False
    unit = unit_vector(vector)
    return any(own == space and any(raw) and cosine(unit, unit_vector(raw)) >= RESTATE_COSINE
               for raw, own in statement.vectors)


def _singles(text: str) -> frozenset[str]:
    """A text's one-character details, which `content_terms` leaves out: each letter or digit
    standing alone ("type O", "vitamin D", "plan A", "2 cats"), once the endings of
    contractions and possessives ("'s", "n't", "'d") are cut, but the pronoun "I", a
    lowercase "a" or "i" and a sentence's opening "A" (an article), and each Han character
    (whose pairs `content_terms` holds)."""
    cut = _CONTRACTED.sub(" ", text)
    found: set[str] = set()
    for single in _SINGLE.finditer(cut):
        letter = single.group()
        before = cut[:single.start()].rstrip()
        if letter in ("I", "a", "i") or letter == "A" and (
                not before or before[-1] in _SENTENCE_END):
            continue
        found.add(letter.casefold())
    return frozenset(found) | frozenset(_HAN.findall(text))


def _claims(text: str) -> frozenset[str]:
    """The terms a statement claims: its `_terms` less the `GENERIC` ones ("user", "named"),
    which any statement about the user carries, and less "called" before a name (it names,
    as "named" does: 我的猫叫Pixel tells "The user's cat is called Pixel."), and its
    one-character details (`_singles`)."""
    naming = frozenset({"called"}) if _NAMING.search(text) else frozenset[str]()
    return (_terms(text) - GENERIC - naming) | _singles(text)


def _said(text: str) -> frozenset[str]:
    """The terms the user's words hold: their `_terms`, their one-character details, and each
    tie a person or animal word of theirs names alone (`named_ties`: "猫" names "cat";
    "pet" names none)."""
    return _terms(text) | _singles(text) | named_ties(text)


def _sentence_clauses(text: str) -> list[list[str]]:
    """A text's clauses, sentence by sentence: split at punctuation, an ellipsis or a dash
    ("…", "...", "—", "–"); a period ends a sentence unless it ends an abbreviation ("Mr.",
    "Dr.") or a word goes on right after it ("U.S."). A subject such as "My brother and his
    wife" stays whole."""
    found: list[list[str]] = [[]]
    start = 0
    for mark in _BREAK.finditer(text):
        if mark.group() == ".":
            words = _LATIN_WORD.findall(text[:mark.start()])
            following = text[mark.end():mark.end() + 1]
            if (words and words[-1].casefold() in _ABBREVIATIONS) or (
                    following and not following.isspace()):
                continue
        piece = text[start:mark.start()].strip()
        if piece:
            found[-1].append(piece)
        start = mark.end()
        if mark.group() == "." or mark.group() in _SENTENCE_BREAK:
            found.append([])
    piece = text[start:].strip()
    if piece:
        found[-1].append(piece)
    return [clauses for clauses in found if clauses]


def _claimed_at(clause: str, claimed: frozenset[str]) -> list[int]:
    """Where the words of a clause holding a term the statement claims start."""
    starts = [word.start() for word in _LATIN_WORD.finditer(clause)
              if _said(word.group()) & claimed]
    starts += [found.start() for found in _HAN_TIE.finditer(clause)
               if named_ties(found.group()) & claimed]
    starts += [index for term in claimed if _HAN.search(term)
               and (index := clause.find(term)) >= 0]
    return starts


def _head(clause: str, claimed: frozenset[str]) -> str:
    """The words of a clause before the first one that holds a term the statement claims:
    where its subject is."""
    return clause[:min(_claimed_at(clause, claimed), default=len(clause))]


def _owned(text: str) -> bool:
    """Whether each Han person or animal word of a text is the user's: right after 我, 我的,
    我家, 我养的, 我有 (一只), 我养了 (一只) or "my" ("我同学的猫", "一只猫" are no cat of the
    user's)."""
    return all(_OWNER.search(text[:found.start()]) is not None
               for found in _HAN_TIE.finditer(text))


def _beings(text: str) -> list[tuple[int, int, frozenset[str]]]:
    """Each person or animal noun of a text, with where it starts and ends and the ties to the
    user it may be: a word of `TIES` ("sister", 猫), or another person or animal noun
    (`_BEINGS`: "stray", "landlord", 兽医), which may be no tie of the user's (none)."""
    found: list[tuple[int, int, frozenset[str]]] = []
    for word in _LATIN_WORD.finditer(text):
        lowered = word.group().casefold()
        if lowered.endswith("s") and (lowered[:-1] in TIES or lowered[:-1] in _BEINGS):
            lowered = lowered[:-1]
        if lowered in TIES:
            found.append((word.start(), word.end(), TIES[lowered]))
        elif lowered in _BEINGS:
            found.append((word.start(), word.end(), frozenset[str]()))
    found += [(tie.start(), tie.end(), TIES[tie.group()]) for tie in _HAN_TIE.finditer(text)]
    found += [(being.start(), being.end(), frozenset[str]())
              for being in _HAN_BEINGS.finditer(text)]
    return sorted(found, key=lambda entry: entry[0])


def _names(text: str, claimed: frozenset[str]) -> list[int]:
    """Where the names of people in a text start: capitalized words that are no function,
    discourse, label, pronoun, first-person, time or calendar word and no word the statement
    claims. Only right after "in", "at", "to" or "near" is one a place ("in Hangzhou", "to
    Berlin"); after any other word, "from" included, it is a name ("a cat from Alice"):
    when unsure, a name."""
    words = list(_LATIN_WORD.finditer(_CONTRACTED.sub(" ", text)))
    return [word.start() for index, word in enumerate(words)
            if word.group()[0].isupper() and len(word.group()) > 1
            and word.group().casefold() not in _NOT_NAMES and not _terms(word.group()) & claimed
            and not (index and words[index - 1].group().casefold() in _AT_PLACE)]


def _report_tag(own: str, clause: str) -> bool:
    """Whether a clause only tells whose report the sentence is, whatever its length: it
    claims nothing of the statement ``own``, holds a report verb (`_REPORT_VERB`) with a
    reporter before it, or "according to" (据说, 听说) with one after it, who is someone other
    than the user (a person or animal noun, a name, a pronoun of `_REPORTER`, a Chinese name),
    and nothing but filler beside ("my vet says", "that's what the vet told me", "Alice told
    me yesterday", "according to Alice", 这是小王告诉我的, 小王提到过, 据说). It is never a tag
    when the user is the reporter ("I told Alice", "like I said", 我跟小王说过), words that
    name no one report nothing ("Let me tell you", 怎么说呢), and a clause holding more is a
    report of something else ("my vet says she's healthy", 我听说过这家店)."""
    claimed = _claims(own)
    if _said(clause) & claimed:
        return False
    according = _ACCORDING.search(clause)
    found = according or _REPORT_VERB.search(clause)
    if found is None:
        return False
    if according is not None:
        reporter, rest = clause[found.end():], clause[:found.start()]
    else:
        reporter, rest = clause[:found.start()], clause[found.end():]
    words = {word.casefold() for word in _LATIN_WORD.findall(_CONTRACTED.sub(" ", reporter))}
    if words & _SELF_SUBJECT or "我" in reporter:
        return False
    han = _HAN_FILLER.sub("", _HAN_THING.sub("", reporter))
    someone = bool(_beings(reporter) or _names(reporter, claimed) or words & _REPORTER
                   or _HAN.search(han))
    if according is not None and not reporter.strip():
        someone = True
    rest_words = {word.casefold() for word in _LATIN_WORD.findall(_CONTRACTED.sub(" ", rest))}
    return someone and rest_words <= STOP | _FILLER | _PRONOUN and not _HAN.search(
        _HAN_FILLER.sub("", rest))


def _another_voice(own: str, text: str) -> bool:
    """Whether the user's words carry someone else's, or the user is not sure of them: a
    speaker's line ("Alice: …"; a label such as "Correction:", "PS:" or 更正： is none); a
    quotation holding the first person, a report or a fact of the past ('"I have a cat,"
    Alice said.'; 'called "Pixel"' and 「Pixel」 are fine); a clause that only tells whose
    report the sentence is (`_report_tag`); or a clause that only hedges it (`_HEDGE`: "…, I
    think."). A report of something else in another clause is none ("My cat is called
    Pixel, my vet says she's healthy.", 我听说过这家店，我的猫叫Pixel)."""
    speaker = _SPEAKER.match(text)
    if speaker is not None and speaker.group(1).casefold().rstrip(".") not in _LABELS and (
            speaker.group(1) not in _HAN_LABELS):
        return True
    for quoted in _QUOTE.finditer(text):
        inside = next(group for group in quoted.groups() if group)
        words = {word.casefold() for word in _LATIN_WORD.findall(inside)}
        if words & _FIRST_PERSON or "我" in inside or _NOT_NOW.search(inside) or (
                _ENDED.search(inside)):
            return True
    plain = _ASIDE.sub(" ", _HAN_DISCOURSE.sub(" ", text))
    return any(_report_tag(own, clause) or _HEDGE.fullmatch(clause.strip()) is not None
               for clauses in _sentence_clauses(plain) for clause in clauses)


def _possessed(own: str, clause: str) -> bool:
    """Whether a possessive of the clause turns to something ``own`` does not claim: an "'s"
    whose next word ``own`` does not claim, with a word it claims after it in the same part
    of the clause ("My cat's vet is called Pixel."), or a claimed Han person or animal word
    followed by 的 (我的猫的兽医叫Pixel). An "'s" before a claimed word is "is" ("My cat's
    called Pixel."), and one before a name word owns the name ("My cat's name is Pixel.",
    我的猫的名字叫Pixel)."""
    claimed = _claims(own)
    for part in _CONJUNCTION.split(clause):
        for found in _POSSESSIVE.finditer(part):
            owned = found.group(1)
            if _said(owned) & claimed or owned.casefold() in GENERIC | STOP | {"called"}:
                continue
            if any(_said(word) & claimed for word in _LATIN_WORD.findall(part[found.end():])):
                return True
        if any(named_ties(tie.group()) & claimed and part.startswith("的", tie.end())
               and not part.startswith(_HAN_NAME, tie.end() + 1)
               for tie in _HAN_TIE.finditer(part)):
            return True
    return False


def _opening(before: str) -> bool:
    """Whether the words before the user in a clause's subject set no one else first: stop,
    discourse, label, time and calendar words, a place or time phrase (`_PLACE`: "After
    work", "In Hangzhou", a capitalized word only right after "in", "at", "to" or "near"),
    and Han time words or a 在 phrase (`_HAN_OPENING`); not a name or a verb ("Alice and I",
    "Alice thinks my", 小王说我)."""
    phrase = False
    words = _LATIN_WORD.findall(_CONTRACTED.sub(" ", before))
    for index, word in enumerate(words):
        lowered = word.casefold()
        if lowered in _PLACE:
            phrase = True
        elif word[0].isupper() and lowered not in _NOT_NAMES:
            if not (index and words[index - 1].casefold() in _AT_PLACE):
                return False
        elif not phrase and lowered not in _NOT_NAMES:
            return False
    return not _HAN.search(_HAN_OPENING.sub("", before))


def _owns(own: str, clause: str) -> bool:
    """Whether the user's words own the person or animal the memory ``own`` says is theirs:
    its first mention in the clause comes after "my", a verb of having or 我的/我有/我养了
    (`_OWNED_BY`). A memory naming no person or animal owns nothing."""
    named = ties(own)
    if not named:
        return True
    mention = next((start for start, _, found in _beings(clause) if found & named), None)
    return mention is not None and _OWNED_BY.search(clause[:mention]) is not None


def _users(own: str, clause: str) -> bool:
    """Whether the clause has the user as the subject of the statement ``own``, from the words
    before the first one it claims (`_head`), discourse words and a name the clause opens on
    before "is my" or 是我的 set aside (`_NAME_FIRST`: "Pixel is my cat." reads "my cat"):

    - no one else is named there: no other person's pronoun ("she", "we", 他, 我们, never the
      他 of 吉他), no person or animal noun but of a tie ``own`` names (`_beings`: "My sister
      has a cat.", "my landlord's cat"), no possessive of someone ("Alice's");
    - the user is: "I", "me" or "my" (我), set first by nothing but discourse, time and place
      words (`_opening`: "Recently my sister", "In Hangzhou I adopted a cat");
    - after the user, no name and no act in progress ("I and Alice", "I'm buying a cat"),
      any other verb being part of what the user says ("I adopted a cat");
    - each Han person or animal noun is the user's (`_owned`).

    A pronoun or another person after the claimed word is part of what the clause says ("My
    sister lives in Paris with her husband.")."""
    claimed, named = _claims(own), ties(own)
    first_name = _NAME_FIRST.search(clause)
    if first_name is not None and first_name.group(1).casefold() not in _NOT_NAMES and (
            first_name.start() <= min(_claimed_at(clause, claimed), default=len(clause))):
        clause = clause[:first_name.start()] + clause[first_name.end():]
    head = _HAN_DISCOURSE.sub(" ", _head(clause, claimed))
    cut = _CONTRACTED.sub(" ", head)
    words = list(_LATIN_WORD.finditer(cut))
    if _OTHER_PERSON.intersection(word.group().casefold() for word in words) or (
            _HAN_PRONOUN.search(head) or _SOMEONES.search(head)):
        return False
    if any(not found & named for _, _, found in _beings(head)):
        return False
    first = next((word for word in words if word.group().casefold() in _FIRST_PERSON), None)
    at = first.start() if first is not None else cut.find("我")
    if at < 0 or not _opening(cut[:at]):
        return False
    after = cut[at:]
    if _names(after, claimed) or any(
            word.casefold().endswith("ing") and len(word) > 4 and word.casefold() not in _TIME
            for word in _LATIN_WORD.findall(after)):
        return False
    return _owned(clause)


def _span(clause: str, claimed: frozenset[str]) -> str:
    """The words of a clause carrying the claim: from its start to the end of the last word
    holding a term the statement claims."""
    ends = [word.end() for word in _LATIN_WORD.finditer(clause) if _said(word.group()) & claimed]
    ends += [found.end() for found in _HAN_TIE.finditer(clause)
             if named_ties(found.group()) & claimed]
    ends += [index + len(term) for term in claimed if _HAN.search(term)
             and (index := clause.rfind(term)) >= 0]
    return clause[:max(ends, default=len(clause))]


def _unreal(span: str, claimed: frozenset[str]) -> bool:
    """Whether the words carrying a claim report it or make it not real (`_NOT_NOW`); a word
    the statement claims itself is none ("Miss Pixel")."""
    return any(not _said(found.group()) & claimed for found in _NOT_NOW.finditer(span))


def _negates(text: str) -> bool:
    """Whether a statement negates its claim (``forget_match.negated``, a bare "no")."""
    return negated(text) or _NO.search(text) is not None


def _denies(span: str, claimed: frozenset[str]) -> bool:
    """Whether the words carrying a claim negate it: ``forget_match.negated``, or "no" before
    a claimed word ("I have no cat."; "My cat has no tail" carries no "no")."""
    return negated(span) or any(_said(found.group(1)) & claimed for found in _NO.finditer(span))


def _for_someone(own: str, clause: str) -> bool:
    """Whether what the memory says is the user's goes to or is for someone else: a "for"
    or "to" a person right after its mention ("I bought a cat for Alice", "a cat last week
    for my son"), or after it in the same part of the clause when a verb of giving comes
    before the mention ("I gave my cat to Alice"). A "to" of another verb is none ("she is
    very attached to my sister", "she listens to Alice")."""
    named, claimed = ties(own), _claims(own)
    found = next(((start, end) for start, end, kind in _beings(clause) if kind & named), None)
    if found is None:
        return False
    start, end = found
    part = _CONJUNCTION.split(clause[end:], maxsplit=1)[0]
    giving = _GIVING.search(clause[:start]) is not None
    for to in _FOR_SOMEONE.finditer(part):
        between = _LATIN_WORD.findall(part[:to.start()])
        near = all(word.casefold() in _TIME | _FILLER | STOP for word in between)
        word = to.group(1)
        person = word.casefold() in _REPORTER or bool(_beings(word)) or bool(
            _names(word, claimed))
        if person and (near or giving):
            return True
    return False


def _ended(own: str, clause: str, span: str, context: str, later: Sequence[str] = ()) -> bool:
    """Whether the claimed one is told ended or given up: an ended fact in the words carrying
    the claim (`_ENDED`: "I used to have a cat", "I sold my cat"); or after them, in a part
    whose subject is the claimed one (the rest of their clause, the rest of the clause
    before a join, or a later part opening with a pronoun or the noun: "My cat died last
    year", "My cat died; she was called Pixel.", "…and she died", 它去世了), that it ended
    (`_GONE`), or, in any later part, that it was given up as an object (`_GIVEN`: "…and I
    sold her"); the ``later`` clauses of the sentence are read as later parts ("我的猫叫
    Pixel，它去世了。"). "…and she lost a tooth", "…and I sold my car", "…and she stopped
    eating" and "a cat that lost her collar" end something else; "…and my grandma died"
    ends someone else."""
    if _ENDED.search(span):
        return True
    named, claimed = ties(own), _claims(own)
    rest = _CONJUNCTION.split(clause[len(span):])
    parts = [(rest[0], True), *((part, False) for part in rest[1:]), (context, True),
             *((part, False) for clause_after in later
               for part in _CONJUNCTION.split(clause_after))]
    for part, about_it in parts:
        part = part.strip()
        if (join := _JOINED.match(part)) is not None:
            part = part[join.end():]
        if not part:
            continue
        if not about_it:
            lead = [word.casefold() for word in _LATIN_WORD.findall(part)[:2]]
            about_it = (bool(lead) and lead[0] in _PRONOUN) or part.startswith(
                ("它", "她", "他")) or any(_said(word) & claimed for word in lead) or (
                _GONE.match(part) is not None)
        if about_it and _GONE.search(part):
            return True
        for given in _GIVEN.finditer(part):
            noun = given.group(1)
            if noun is None or _said(noun) & claimed or any(
                    kind & named for _, _, kind in _beings(noun)):
                return True
    return False


def _names_it(own: str, clause: str, span: str, context: str) -> bool:
    """Whether the user's words link the noun and the name a naming memory states: a naming
    word in the words carrying the claim (``span`` of ``clause``), or in the rest of the
    clause before a join (`_NAME_LINK`: "called", "named", "name", 叫, 名字; "My cat is
    called... Pixel."); the name right after the noun ("my cat Pixel", 我的猫Pixel); an
    "is" between them (`_COPULA`: "My cat is Pixel.", 我的猫是Pixel), the name not turned
    to something else (`_OF`: "My cat is Pixel's friend."); or the name first, then "is my"
    and the noun, itself not turned to something else (`_MINE`: "Pixel is my cat.", Pixel是
    我的猫; not "Pixel is my cat's vet."). "My cat likes Pixel." and "My cat and Pixel are
    friends." link nothing."""
    if _NAME_LINK.search(span) or _NAME_LINK.search(context):
        return True
    named, claimed = ties(own), _claims(own)

    def name_at(at: int) -> re.Match[str] | None:
        word = _LATIN_WORD.match(clause, at)
        return word if word is not None and word.group()[0].isupper() and (
            _said(word.group()) & claimed) else None

    beings = [(start, end) for start, end, kind in _beings(clause) if kind & named]
    for _, end in beings:
        if name_at(end + 1 if clause[end:end + 1] == " " else end) is not None:
            return True
        link = _COPULA.match(clause, end)
        name = name_at(link.end()) if link is not None else None
        if name is not None and _OF.match(clause, name.end()) is None:
            return True
    for word in _LATIN_WORD.finditer(clause):
        link = _MINE.match(clause, word.end()) if name_at(word.start()) is not None else None
        if link is not None and any(start == link.end() and _OF.match(clause, end) is None
                                    for start, end in beings):
            return True
    return False


def _hedged_after(own: str, clause: str, span: str) -> bool:
    """Whether the words after the claim, in its own part of the clause, point forward ("I
    get a cat tomorrow"), or end in words that only hedge it or tag a report ("…Pixel I
    think", "…Pixel my vet says", "…is called Pixel if I remember correctly", "I have a cat
    apparently", "…according to my vet")."""
    tail = _CONJUNCTION.split(clause[len(span):], maxsplit=1)[0].strip(" ,;:.!?")
    if not tail:
        return False
    if _FORWARD.search(tail) is not None:
        return True
    words = list(_LATIN_WORD.finditer(tail))
    for index in range(len(words)):
        end = tail[words[index].start():].strip()
        if _HEDGE.fullmatch(end) is not None or _report_tag(own, end) or _ACCORDING.match(end):
            return True
    return any(_HEDGE.fullmatch(tail[at:]) is not None
               for at in range(len(tail)) if _HAN.match(tail, at))


def _states_in(own: str, clause: str, context: str = "", later: Sequence[str] = ()) -> bool:
    """Whether one clause states the memory ``own`` in full, of the user, as true now (words
    to the listener such as "as I told you" set aside). It holds every term ``own`` claims;
    the words carrying the claim (`_span`) negate it exactly when ``own`` does, both ways
    (`_denies`), report it, make it unreal or hedge it in no way (`_unreal`, `_HAN_REPORT`,
    `_UNSURE`), and, for a naming memory, link the noun and the name (`_names_it`: "called",
    叫, "my cat Pixel", "My cat is Pixel.", "Pixel is my cat."; "My cat likes Pixel." names
    nothing); the words after them do not only hedge
    it, tag a report or point forward (`_hedged_after`); the claimed one is not told ended or
    given up (`_ended`, with the ``context`` of the clause before it when joined to it and the
    ``later`` clauses of its sentence) nor
    given to someone else (`_for_someone`); ``forget_match.exclusion`` finds no other subject
    or opposite claim; no possessive turns to something else (`_possessed`); the user is its
    subject (`_users`) and owns the person or animal it says is the user's (`_owns`)."""
    clause = _ASIDE.sub(" ", clause).strip()
    claimed = _claims(own)
    if not claimed <= _said(clause):
        return False
    span = _span(clause, claimed)
    if _NAMING_MEMORY.search(own) and not _names_it(own, clause, span, context):
        return False
    return (_negates(own) == _denies(span, claimed) and not _unreal(span, claimed)
            and _UNSURE.search(span) is None
            and _HAN_REPORT.search(_HAN_DISCOURSE.sub(" ", span)) is None
            and not _hedged_after(own, clause, span)
            and not _ended(own, clause, span, context, later)
            and not _for_someone(own, clause)
            and exclusion(own, clause) is None and not _possessed(own, clause)
            and _users(own, clause) and _owns(own, clause))


def _subjectless(clause: str, claimed: frozenset[str]) -> bool:
    """Whether a clause names no subject of its own before what it claims: only a pronoun,
    stop, discourse, time or naming words, or Han linking words ("she lives in Paris",
    "she's called Pixel", "is called Pixel", "Pixel", 叫Pixel, 它叫Pixel)."""
    head = _HAN_DISCOURSE.sub(" ", _head(clause, claimed))
    words = [word.casefold() for word in _LATIN_WORD.findall(_CONTRACTED.sub(" ", head))]
    return all(word in _PRONOUN or word in STOP or word in _DISCOURSE or word in _TIME
               or word in GENERIC or word == "called" for word in words) and not _HAN.search(
        _HAN_LINKING.sub("", _HAN_PRONOUN.sub("", head)))


def _antecedent(own: str, before: str) -> int | None:
    """Where the mention of the subject of the statement ``own`` ends in the clause
    ``before``, when a clause after it naming no subject of its own can only point back to
    it: that clause names it (a noun of a tie ``own`` names; for a statement about the user,
    it names no one at all, and the whole clause is the mention), and no other person or
    animal after it, of any kind, nor a name: in "My cat met a dog yesterday — it is called
    Pixel.", "My cat met a stray; it…" or "My cat had a kitten; it…" the "it" may be the
    other animal. None otherwise."""
    named, claimed = ties(own), _claims(own)
    beings = _beings(before)
    if not named:
        return None if beings or _names(before, claimed) else len(before)
    mention = next(((start, end) for start, end, found in beings if found & named), None)
    if mention is None or any(start >= mention[1] for start, _, _ in beings) or _names(
            before[mention[1]:], claimed):
        return None
    return mention[1]


def _parenthetical(clause: str, claimed: frozenset[str]) -> bool:
    """Whether a clause is an aside between commas that claims nothing: opened by a word of
    `_PARENTHETICAL` ("unlike Alice's", "like Alice's dog") or a relative pronoun ("which is
    ten"), or a short apposition with no verb of report, opinion, change or ending ("Alice's
    favourite")."""
    words = _LATIN_WORD.findall(clause)
    if not words or _said(clause) & claimed:
        return False
    first = words[0].casefold()
    if first in _PARENTHETICAL or first in {"which", "who", "whose"}:
        return True
    return len(words) <= 3 and not (_REPORT_VERB.search(clause) or _NOT_NOW.search(clause)
                                    or _ENDED.search(clause))


def states(statement: Told, own: str) -> bool:
    """Whether the user's words state the memory ``own`` in full, of themself, as true now:
    they carry no one else's voice (`_another_voice`), and one clause of theirs states it
    (`_states_in`), or a clause naming no subject of its own joined to the one before it in
    its sentence ("I have a sister; she lives in Paris.", "My cat — she's called Pixel."),
    past an aside between commas ("My cat, unlike Alice's, is called Pixel."). The clause
    before must name the statement's subject and no other person or animal after it
    (`_antecedent`): in "My cat met a dog yesterday — it is called Pixel." the "it" may be
    the dog. Only the words up to that mention join; the rest of that clause is read for an
    ended fact alone ("My cat must be the cutest; she is called Pixel."). A memory adding a word the
    user did not say, the opposite of what they said, a fact about someone else, of the past
    or of a wish is not stated by them. `says` reads none of this, so a word-for-word
    restatement keeps its base reading ("I used to live in Beijing." restates "The user
    lives in Beijing.")."""
    text = statement.text
    claimed = _claims(own)
    if not claimed or _another_voice(own, text):
        return False
    for clauses in _sentence_clauses(text):
        for index, clause in enumerate(clauses):
            later = clauses[index + 1:]
            if _states_in(own, clause, later=later):
                return True
            if not index or not _subjectless(clause, claimed):
                continue
            back = index - 1
            while back > 0 and _parenthetical(clauses[back], claimed):
                back -= 1
            before = clauses[back]
            end = _antecedent(own, before)
            if end is not None and _states_in(own, f"{before[:end]} {clause}", before[end:],
                                              later):
                return True
    return False


def retells(statement: Told, own: str, fact: str) -> bool:
    """Whether the memory ``own`` tells part of the forgotten ``fact`` again in the user's
    words: their words state it in full, of themself (`states`), and it claims a term of the
    fact. It keeps none of the fact's words the user did not say again: "my cat is called
    Pixel" retells "The user has a cat." of a forgotten "The user has a tabby cat.", never
    "tabby"; "My sister has a cat." and "I don't have a cat any more." retell nothing."""
    return states(statement, own) and bool(_claims(own) & _claims(fact))


def restates(statement: Told | None, forgotten: Forgotten, own: str | None = None) -> bool:
    """Whether the user stated ``forgotten`` again: `after` its forget, and `says` it, or
    `retells` the memory ``own`` (when given) of it."""
    return statement is not None and after(statement, forgotten.at) and (
        says(statement, forgotten.statement, forgotten.vector, forgotten.space)
        or (own is not None and retells(statement, own, forgotten.statement)))


def relearned(groups: Iterable[Told | None], forgotten: Forgotten,
              own: str | None = None) -> bool:
    """Whether one of the support groups (`group_told`) of the memory ``own`` (when given)
    `restates` ``forgotten``."""
    return any(restates(group, forgotten, own) for group in groups)
