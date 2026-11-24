"""The interactive Search profile's ranking rules (``agent-v2``): relevance, forgets, order.

An agent plugin searches its user's memory silently, mid-conversation, often in another
language than the stored statements, and shows every hit to a model that takes it at its
word. The interactive profile (``quotemem bench serve --search-profile interactive``) therefore
ranks by its own rules (`quotemem.core.recall.recall_interactive`); the bench profile ranks byte for
byte as before.

1. **A score is a relevance.** Each candidate's score is `relevance`, in [0, 1): its
   cosine with the query above `DENSE_FLOOR` (scaled by `DENSE_SCALE`), joined with the
   share of the query's rarer words it holds (`coverage`, weighted `LEXICAL_WEIGHT`, taken
   over at least the weight of a word the scope never holds, so a query of common words
   alone moves little). It rises with both and depends on nothing but the query, the text
   and the scope's word frequencies: not on the other candidates or ``top_k``, so the same
   score means the same match in every Search of a scope. A memory's cosine is the better
   of its statement's and that of a user part it mostly quotes (`QUOTE_SHARE` of it,
   `QUOTE_COVER`), so a Chinese question finds the English statement of a fact the user
   told in Chinese, and the reverse. Nothing below `RELEVANCE_FLOOR`, or below
   `RELATIVE_SHARE` of the best candidate's relevance, is delivered (`delivery_floor`), so
   a question the store knows nothing about answers no hit; a near topic can still pass
   ("What's my dog's name?" may find the cat). A user part counts `EVIDENCE_FACTOR` of its
   relevance, an assistant part `ASSISTANT_FACTOR`, a ``past`` memory `PAST_FACTOR`, and
   every score is times
   `freshness` (at most `RECENCY_WEIGHT` less for the scope's oldest support).
2. **A question is no fact.** A user part that only asks or instructs (`fact_free`: no
   sentence states a fact, as "Where do I live? One sentence, no tools.") is not
   delivered, unless the query asks what the user asked (`asks_own`). "Now I live in
   Suzhou." and "No, my sister is Lin Xue." state facts.
3. **A part brings its memories** (`draw_quoting`). When no memory quoting a delivered user
   part is ``current``, they come at its score with their statuses (and what replaces
   them), and a part whose memories are all ``past`` is not delivered; otherwise a memory
   quoting it comes when it is relevant itself (`RELEVANCE_FLOOR`).
4. **Nothing forgotten comes back.** A hit is dropped when it quotes or holds a sentence the
   user asked to forget in chat (``forget_marks:{s}``), when it records or makes a forget
   request (`Requests`), or when the forget shadow matches it (`forget_shadowed`): a name
   of a forgotten fact, two of its topic keywords, or a memory lying next to it. The
   shadow comes from three places: the forgotten versions' names and keywords, recorded at
   the forget as keyed digests (``forget_shadow:{s}``, `ForgetShadowTerms`); the marked
   sentences; and what the scope's forget requests asked to forget.
5. **Statements first.** Every delivered memory precedes every Evidence part
   (`recency_order`); an assistant part is dropped unless the query asks what the
   assistant said.
6. **A status from the store.** Each delivered memory's status (`Status`) is read from the
   whole scope by `standing`, never from the other hits of the Search; a memory that
   replaces or may replace another goes directly above it when both are delivered.
7. **No near-duplicates** (`collapse`).

The ranking is computed to a fixed depth, the server's largest ``top_k``, and then cut, so
the hits of a smaller ``top_k`` are the first hits of a larger one. Delivered scores never
increase down the list (`non_increasing`).
"""

from __future__ import annotations

import bisect
import functools
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_EVEN, Decimal
from types import MappingProxyType
from typing import Final, Literal

from quotemem.core.recall_corpora import sparse_terms
from quotemem.core.statement import own_token, token_seq
from quotemem.core.vector_score import UnitVector, cosine

AGENT_VARIANT: Final = "agent-v2"
"""The ranking variant name the interactive profile's ranking digest carries."""

SHADOW_PURPOSE: Final = "forget-shadow-term"
"""The keyed-digest purpose of a forget shadow term (``forget_shadow:{s}``)."""

type Status = Literal["current", "past", "possibly_outdated"]
"""Whether a delivered memory still holds, read from the whole scope (`standing`).

- ``current``: nothing in the store ends or questions it. A past event that stays true
  ("The user moved from Shanghai to Hangzhou last week") is current: it still happened.
- ``past``: the store says it no longer holds. The Add judge replaced, corrected or
  retracted it (its row is no longer active, or a current memory ``supersedes`` it), or
  it is a state whose validity ended before the query. What replaces it goes directly
  above it: the memories that supersede it and, when the judge linked only an event of the
  change, the newer state memories of its anchor close to it.
- ``possibly_outdated``: a later memory or user message conflicts with it, but nothing
  replaced it: a newer live memory ``contradicts`` it; a newer live state memory of the
  same anchor restates the same thing with other content (`SAME_STATE_JACCARD`), or
  carries a change cue (`CHANGE_CUES`) and lies within `CORRECTION_COSINE` of it; or a
  newer user part with a change cue does. A live memory is current and not past: a
  retracted or replaced memory questions nothing. A change the judge settled (a memory
  that ``supersedes`` another, the user words it quotes, and the words a past memory
  quotes) questions nothing else. Only a state memory can be outdated this way; an event
  cannot.

The status depends on the store at the read pin and the query time only, never on the
other hits of a Search or on ``top_k``.
"""

EVIDENCE_FACTOR: Final = Decimal("0.8")
"""Share of its relevance a user Evidence part keeps, so a statement of the same fact
ranks above the raw words it was drawn from."""
ASSISTANT_FACTOR: Final = Decimal("0.7")
"""Share an assistant Evidence part keeps when the query asks what the assistant said
(`asks_back`); any other query drops assistant parts. Below `EVIDENCE_FACTOR`, so the
user's own words still come first."""
ASKS_BACK: Final = re.compile(
    r"\b(?:did|have) you (?:\w+ )?(?:provide|give|recommend|suggest|mention|tell|list|share"
    r"|say|write|describe|explain|offer)\b"
    r"|你(?:之前|上次|刚才|以前)?(?:推荐|建议|说过|提到|告诉|给过|列过)", re.IGNORECASE)
"""Present-tense and Chinese wording that asks back what the assistant said ("What book did
you recommend?", "你之前推荐过什么书？"), beside the bench's `recalls_assistant`."""
RECENCY_WEIGHT: Final = Decimal("0.05")
"""Freshness prior (`freshness`): a score times ``1 - RECENCY_WEIGHT * (1 - r)``, r the rank
of the hit's newest support among the scope's distinct support commits, scaled to [0, 1]."""
DENSE_FLOOR: Final = Decimal("0.30")
"""Query cosine at or below which a text means nothing to the query (`relevance`). On the
real-tool replay (``qwen3.7-text-embedding``, 1024 dimensions) no unrelated statement of
a question the store cannot answer ("What is my blood type?") reached 0.33."""
DENSE_SCALE: Final = Decimal("0.25")
"""Cosine span over which a text's meaning closes 63% of its distance to a full match
(`relevance`): the replay's true hits lay at cosines 0.43 to 0.75."""
QUOTE_SHARE: Final = Decimal("0.9")
"""Share of a quoted user part's query cosine a memory may take as its own: a part may
state more than the memory, so it counts a little less than the statement."""
QUOTE_COVER: Final = 0.5
"""Share of a user part's bytes a memory must quote to take the part's cosine: a memory
drawn from one clause of a long message ("我叫周以宁，在上海做数据工程师，住在上海静安区")
does not inherit what the other clauses match."""
PAST_FACTOR: Final = Decimal("0.8")
"""Share of its score a ``past`` memory keeps, so what holds now ranks above what held
before."""
LEXICAL_WEIGHT: Final = Decimal("0.5")
"""Weight of the query words a text holds (`coverage`) beside its meaning (`relevance`);
above `RELEVANCE_FLOOR`, so a text holding every query word is delivered on that alone."""
RELEVANCE_FLOOR: Final = Decimal("0.25")
"""Smallest `relevance` a hit is delivered with: a cosine of about 0.37 alone. On the
real-tool replay (115 questions, ``qwen3.7-text-embedding``) every unrelated hit of a
question the store cannot answer ("What is my blood type?", "What is the capital of
Australia?", "我有几个孩子？") lay below 0.17, while true answers to loosely worded
questions ("What's my job?", "我是做什么的？", "What's my name?") lay from 0.25 to 0.38. A
near topic still passes ("What's my dog's name?" finds the cat at 0.42), so an empty result
means nothing related, not that every result answers."""
RELATIVE_SHARE: Final = Decimal("0.6")
"""Share of the best candidate's `relevance` a hit also needs (`delivery_floor`): a question
with a strong answer (0.8) does not trail weak neighbours (0.3) after it, and one whose
best match is weak keeps what lies near that match."""
SAME_STATE_JACCARD: Final = 0.5
"""Content-term Jaccard from which a newer state memory of the same anchor restates an
older one with other content (up to `NEAR_SAME_JACCARD`, a mere rewording), so the older
is ``possibly_outdated``."""
CORRECTION_COSINE: Final = 0.55
"""Cosine from which a newer text with a change cue (`CHANGE_CUES`) may correct a state
memory, so the memory is ``possibly_outdated``."""
SHADOW_COSINE: Final = 0.86
"""Cosine to a forgotten statement's vector from which a memory restates it."""
NEAR_SAME_JACCARD: Final = 0.85
"""Content-term Jaccard from which two memories of one anchor and kind are the same; the
older one is not delivered."""
COVERED_SHARE: Final = 0.7
"""Share of an Evidence part's bytes delivered memories quote from which it is not delivered."""
ECHO_JACCARD: Final = 0.8
"""Sparse-term Jaccard from which an Evidence part repeats a higher-ranked one."""
NAME_KEEP_MEMORIES: Final = 2
"""Surviving memories that keep a forgotten name readable: a name two or more surviving
memories still state (a sister's name, a city) is not a shadow name; one that only one
leftover restatement holds stays one."""
MIN_ECHO_CHARS: Final = 20
"""Shortest quoted text an assistant part's ``> `` lines must reproduce to echo a user part."""

STOP: Final = frozenset(
    "a an the and or but of to in on at for with by from about as into over is are was were be "
    "been being am do does did have has had i me my mine myself you your yours we our it its "
    "that this these those please forget remember not no don't dont never any all some".split())
"""Function words that carry no content: one list for this ranking and the in-chat forget
(``ingestion/forget_requests``, ``service/forget_statement``)."""
SHORT_NOT_NAMES: Final = STOP | frozenset(
    "mr mrs ms dr st mt jr sr ok oh hi hey yes yet mom mum dad sir god his her him she they "
    "who why how one two new old big per via etc".split())
"""Capitalized words of two or three letters that are no given name (`_SHORT_NAMED`)."""
GENERIC: Final = frozenset({"user", "person", "someone", "people", "named", "name", "thing"})
"""Content terms that say nothing a more specific statement would not also say."""
ORG_SUFFIX: Final = frozenset({
    "labs", "inc", "ltd", "corp", "company", "group", "tech", "technologies", "ai", "data",
    "数据", "科技", "公司", "集团", "网络", "智能",
})
"""Words that end organization names; alone they name nothing."""
CALENDAR: Final = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday january february march april "
    "may june july august september october november december today tomorrow yesterday".split())
"""Capitalized words that are dates, not names."""
HAN_STOP: Final = frozenset(
    "我在 我们 你们 他们 她们 这件 件事 这个 那个 一个 一家 什么 怎么 可以 因为 所以 "
    "但是 已经 还是 就是 没有 不是 现在 最近 今天 明天 昨天 时候 以后 之前 准备 打算 "
    "喜欢 觉得 知道 我的 你的 的事 事情 东西 我最 了我 我也 我还 也是 一下 一点 不要 "
    "记得 忘掉 忘记 帮我 请帮".split())
"""Han bigrams too common to name a topic."""
ASKING: Final = frozenset(
    "what where when which who whom whose why how remember usually tell know recall "
    "什么 哪里 哪儿 哪个 哪家 哪些 怎么 怎样 为什 多少 记得 还记 平时 在哪 是什".split())
"""Question words and asking phrases: a query's `coverage` leaves them out, since the facts
it asks for never hold them."""
_QUESTION_END: Final = frozenset("?？")
_HAN_ASKING: Final = re.compile(
    r"什么|哪|怎么|怎样|为什么|为何|谁|几[个点岁天号月年次]|多少"
    r"|是否|有没有|是不是|能不能|可不可以")
"""Han words that make a clause a question."""
_HAN_TAIL: Final = re.compile(r"吗\s*[!！]?\s*$|(?:呢|么)\s*[!！]?\s*$")
"""A Han question particle that ends a sentence with no full stop after it ("你记得我住哪吗",
"我住在哪里呢"); 呢 and 么 make a question only beside a question word (`_HAN_ASKING`), since
"我在学日语呢。" states a fact."""
_LATIN_ASKING: Final = re.compile(
    r"^\W*(?:(?:and|or|but|also|so|then|ok(?:ay)?)\W+)*(?:what|where|when|which|who|whom|whose|why"
    r"|how|do|does|did|is|are|was|were|am|can|could|would|should|will|shall|may|have|has"
    r"|any)\b", re.IGNORECASE)
"""A Latin clause that opens with a question word or an auxiliary verb."""
_LATIN_REQUEST: Final = (
    r"(?:(?:please|pls|kindly|just|now|ok(?:ay)?|so|and|then)\W+)*"
    r"(?:please|pls|give|write|show|tell|list|explain|help|make|create|compute|count|calculate"
    r"|translate|summari[sz]e|use|find|search|let'?s|answer|reply|respond|keep|say|describe"
    r"|remind|recommend|suggest|draft)\b"
    r"|(?:in\s+)?(?:one|two|three|1|2|3)(?:\s*(?:or|to|-)\s*(?:two|three|four|2|3|4))?\s+"
    r"(?:short\s+|brief\s+)?(?:sentences?|lines?|words?|paragraphs?|bullets?)\b"
    r"|(?:in\s+)?a\s+(?:short|single|brief|quick|one-line)\s+(?:sentence|answer|reply|line)\b"
    r"|no\s+(?:tools?|need|markdown|explanations?|details?|lists?|bullets?|code|search(?:ing)?"
    r"|preamble|extra)\b"
    r"|(?:don'?t|do\s+not|never)\s+(?:use|call|search|look|explain|add|list|ask)\b")
"""Latin request shapes: an imperative verb (after an optional "please", "just" or "now"), an
answer length ("One sentence", "One or two sentences", "in a short sentence") or a "no
tools" / "don't search" instruction.
Each word ends at a word boundary, so "Now I live ...", "Justin ...", "Running ...",
"Users ..." and "No, my sister is ..." state facts."""
_HAN_REQUEST: Final = (
    r"请(?:你|帮|给|告诉|用|写|列|解释|计算|翻译|总结|回答|回复|只|不要|简单|直接|说|问|推荐|查)"
    r"|帮(?:我|忙)|给我(?:写|列|讲|说|推荐|找|算|查|解释|翻译|总结|来|一)|告诉我"
    r"|写(?:一|个|段|首|篇|份)|列(?:出|一|个|举)|解释(?:一下|下)|算(?:一下|下)"
    r"|翻译(?:一下|下|成|为|这)|总结(?:一下|下|这)|一句话|一两句|简短|简单(?:说|回答)"
    r"|不要(?:调用|用|使用|工具|搜索|解释|列|查)|不用(?:调用|用|工具|搜索|解释|查)"
    r"|别(?:调用|用工具|搜索|解释|查)|只(?:用|回答|说|给)|(?:直接|简单)?回答(?:我|一下|下)"
    r"|回复(?:我|一下|下)|用(?:中文|英文|一句话)(?:回答|回复|说)?")
"""Han request shapes: "请帮我…", "给我写…", "一句话", "不要调用工具", not a word that merely opens
with the same character ("别人都叫我小周", "写代码是我的工作", "计算机", "不用担心")."""
_REQUEST: Final = re.compile(rf"^\s*(?:{_LATIN_REQUEST}|{_HAN_REQUEST})", re.IGNORECASE)
"""A sentence or clause that asks for something instead of stating a fact."""
_CLAUSE: Final = re.compile(r"[,，、;；:：]")
"""Where a question sentence splits into clauses: "我上周搬到苏州了，附近有什么好吃的？" states a
move before it asks."""
FACT_TERMS: Final = 2
"""Content terms a clause of a question needs to state a fact (`fact_free`): "对了" and "By
the way" do not."""
_OWN_QUESTION: Final = re.compile(
    r"\b(?:did|have) I (?:\w+ )?(?:ask|asked|question)|\bwhat (?:questions? )?(?:did|have) I"
    r" (?:\w+ )?ask|\bmy (?:earlier|previous|last) questions?\b"
    r"|我(?:之前|上次|刚才|以前|都)?问(?:过|了)", re.IGNORECASE)
_CLAUSE_BREAK: Final = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；])|\n")
"""Where a sentence ends: Latin end punctuation before white space (not the dot of
"data.csv"), Chinese end punctuation, or a line break."""
_HAN_CUES: Final = ("更正", "搬到", "现在", "已经", "不再", "改成")
_LATIN_CUES: Final = re.compile(r"\b(?:now|moved|no longer|switched|actually|instead)\b",
                                re.IGNORECASE)
CHANGE_CUES: Final = (*_HAN_CUES, "now", "moved", "no longer", "switched", "actually", "instead")
"""Words that say a statement replaces an earlier one (correction cues)."""
SENTENCE: Final = re.compile(r"[^.!?;\n。！？；]+[.!?;。！？；]*")
"""A sentence: a run up to and including its closing punctuation, in English or Chinese."""
_CAPITALIZED: Final = re.compile(r"(?<![\w'’])[A-Z][A-Za-z'’-]{3,}")
_SHORT = r"(?P<name>[A-Z][a-z]{1,2})"
_SHORT_NAMED: Final = (
    re.compile(rf"\b(?:named|called|nicknamed)\s+{_SHORT}\b"),
    re.compile(rf"(?:^|[.!?]\s+){_SHORT}(?:\s+[A-Z][a-z]+)?\s+is\s+(?:the\s+user's|my)\b"),
    re.compile(rf"\b(?:[Tt]he user's|[Mm]y)\s+(?:[a-z]+\s+){{0,2}}?{_SHORT}\b(?!['’])"),
    re.compile(rf"\b{_SHORT}['’]s\s+(?:(?:first|last|full|nick)\s*)?name\b"),
)
"""Where a text gives a person or an animal a given name of two or three letters (`_names`):
named so ("A person is named Tom.", "The user's dog is called Bo."), as the subject of a tie
("Amy is the user's coworker."), after one ("The user's coworker Amy", "My friends Tom and
Lucas") and as the owner of a name ("Ben's name"). A short capitalized word anywhere else ("Go",
"Up", ``LA``) is no name: it would hide every hit holding that word."""
_SENTENCE_END: Final = frozenset(".!?:;\n。！？：；")
_LATIN_LETTER: Final = re.compile(r"[A-Za-z]")
_QUOTED_LINE: Final = re.compile(r"^>\s?(.*)$", re.MULTILINE)
_SIX: Final = Decimal("0.000001")
_ONE: Final = Decimal(1)


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(_SIX, rounding=ROUND_HALF_EVEN)


def _identity(term: str) -> str:
    return term


def content_words(text: str) -> frozenset[str]:
    """The tokens of a text that carry its content (`token_seq` less `STOP` and single
    characters), so no Han, Kana or Hangul character: each is a token of its own."""
    return frozenset(token for token in token_seq(text) if token not in STOP and len(token) > 1)


def content_terms(text: str) -> frozenset[str]:
    """The terms of a text that carry its content, in every script.

    The `sparse_terms` of the text (its tokens, then the bigrams of adjacent Han, Kana and
    Hangul characters) less `STOP` and less single-character terms: a Chinese text keeps
    its two-character words, which `content_words` loses, since every Han character is a
    token of its own. An English text keeps exactly its `content_words`. The interactive
    ranking and the complete in-chat forget (``service/forget_statement``) match texts by
    these terms, so a Chinese request ("我在面试星河科技") shares "面试", "星河" and "科技"
    with every sentence restating it, whichever language the rest is in.

    Args:
        text: Any statement, quote or sentence.

    Returns:
        The content terms.

    Examples:
        ```python
        assert {"面试", "星河", "科技", "company"} <= content_terms("面试 星河科技 company")
        ```
    """
    return frozenset(term for term in sparse_terms(text) if term not in STOP and len(term) > 1)


def word_stem(term: str) -> str:
    """A Latin term without a plural or third-person ``s`` ("orders" -> "order", "lives" ->
    "live"), so a query and a statement in another inflection still share it; any other
    term unchanged."""
    if len(term) > 3 and term.isascii() and term.isalpha() and term.endswith("s") and (
            not term.endswith("ss")):
        return term[:-1]
    return term


@functools.lru_cache(maxsize=1 << 12)
def word_forms(stem: str) -> tuple[str, ...]:
    """The terms whose `word_stem` is ``stem`` (``stem`` itself and its ``s`` form)."""
    return (stem, stem + "s") if len(stem) >= 3 and stem.isascii() and stem.isalpha() else (
        stem,)


def asked_terms(query: str) -> frozenset[str]:
    """The term stems (`word_stem`) of a query that a matching fact holds.

    The `content_terms` of its questions and statements, less `ASKING`, `HAN_STOP` and
    `GENERIC`; a request sentence ("One sentence, no tools.") is left out unless the query
    is nothing else, since a fact never holds how its answer should look.
    """
    pieces = [piece.strip() for piece in _CLAUSE_BREAK.split(query) if piece.strip()]
    asked = [piece for piece in pieces if _question(piece) or not _request(piece)] or pieces
    return frozenset(word_stem(term) for term in content_terms(" ".join(asked))
                     if term not in ASKING and term not in HAN_STOP and term not in GENERIC)


def coverage(asked: Mapping[str, Decimal], held: Collection[str],
             least: Decimal = Decimal(0)) -> Decimal:
    """The weighted share of a query's term stems that a text holds.

    Args:
        asked: Each query term stem (`asked_terms`) with its weight, an inverse document
            frequency, so a rare name counts more than a common word.
        held: The text's terms; a stem is held when one of its `word_forms` is.
        least: The smallest total weight the share is taken over: a query whose only words
            are common ("Where do I work?" asks "work") then covers little in any text, so
            one shared common word cannot pass `RELEVANCE_FLOOR` or outrank the meaning. The
            interactive ranking passes the weight of a word the scope never holds.

    Returns:
        The held stems' weight over the larger of all the weight and ``least``, in [0, 1];
        0 when nothing is asked.
    """
    total = max(sum(asked.values(), Decimal(0)), least)
    if not total:
        return Decimal(0)
    return _quantize(sum((weight for term, weight in asked.items()
                          if any(form in held for form in word_forms(term))),
                         Decimal(0)) / total)


def relevance(cosine_value: Decimal | None, covered: Decimal) -> Decimal:
    """How well a text answers a query, in [0, 1): the score an interactive hit carries.

    The text's meaning is ``1 - exp(-(cosine - DENSE_FLOOR) / DENSE_SCALE)`` above
    `DENSE_FLOOR` and 0 below it, so it rises with the cosine without ever saturating; the
    held query words (`coverage`) join it as an independent chance of a match, squared so
    that a text must hold most of them to count: ``1 - (1 - meaning) * (1 -
    LEXICAL_WEIGHT * covered ** 2)``. It rises with the cosine and with the coverage and
    depends on nothing else. Without a query vector (the keyword fallback) the held words
    are the only evidence, and the relevance is the coverage itself.

    Args:
        cosine_value: The text's cosine with the query vector, or None without one.
        covered: The text's `coverage` of the query.

    Returns:
        The relevance, quantized to six places.
    """
    if cosine_value is None:
        return _quantize(covered)
    meaning = Decimal(0)
    if cosine_value > DENSE_FLOOR:
        meaning = _ONE - (-(cosine_value - DENSE_FLOOR) / DENSE_SCALE).exp()
    return _quantize(_ONE - (_ONE - meaning) * (_ONE - LEXICAL_WEIGHT * covered * covered))


def delivery_floor(best: Decimal) -> Decimal:
    """The smallest relevance a hit of a Search is delivered with, given the best candidate's
    relevance: `RELEVANCE_FLOOR`, or `RELATIVE_SHARE` of the best when that is higher. It
    depends on the query and the store only, never on ``top_k``."""
    return max(RELEVANCE_FLOOR, _quantize(RELATIVE_SHARE * best))


def freshness(rank: Decimal) -> Decimal:
    """The factor of a score whose support has the given rank among the scope's support
    commits, scaled to [0, 1] (1 the newest): ``1 - RECENCY_WEIGHT * (1 - rank)``."""
    return _ONE - RECENCY_WEIGHT * (_ONE - rank)


def _question(sentence: str) -> bool:
    """Whether a sentence asks: it ends with a question mark, with 吗, or with 呢 or 么 beside a
    Han question word, and no full stop after the particle."""
    stripped = sentence.rstrip()
    if not stripped:
        return False
    if stripped[-1] in _QUESTION_END:
        return True
    tail = _HAN_TAIL.search(stripped)
    return tail is not None and (tail.group().lstrip().startswith("吗")
                                 or _HAN_ASKING.search(stripped) is not None)


def _request(piece: str) -> bool:
    """Whether a sentence or clause opens with a request shape (`_REQUEST`)."""
    return _REQUEST.search(piece) is not None


def _asking_clause(clause: str) -> bool:
    """Whether a clause of a question sentence asks: it opens with a Latin question word or
    auxiliary, or holds a Han question word or particle."""
    return (_LATIN_ASKING.search(clause) is not None or _HAN_ASKING.search(clause) is not None
            or _HAN_TAIL.search(clause.rstrip("?？ ")) is not None)


def states_fact(sentence: str) -> bool:
    """Whether a sentence states something of the user, not only asks or instructs.

    A request (`_REQUEST`: "Please ...", "One sentence, no tools.", "请帮我...") states nothing.
    A sentence that does not ask (`_question`) states a fact. A question states one when a
    clause of it (split at commas, semicolons and colons) neither asks nor requests and holds
    `FACT_TERMS` content terms or more: "我上周搬到苏州了，附近有什么好吃的？" does, "Where do I
    live, and what do I drink?" does not.

    Args:
        sentence: One sentence.

    Returns:
        True when it states a fact.
    """
    if _request(sentence):
        return False
    if not _question(sentence):
        return True
    for clause in _CLAUSE.split(sentence):
        clause = clause.strip()
        if not clause or _asking_clause(clause) or _request(clause):
            continue
        if len(content_terms(clause) - ASKING - HAN_STOP - GENERIC) >= FACT_TERMS:
            return True
    return False


@functools.lru_cache(maxsize=1 << 16)
def fact_free(text: str) -> bool:
    """Whether a user part only asks or instructs, so it states no fact of the user.

    No sentence states a fact (`states_fact`): each is a question ("Where do I live now?",
    "你还记得我住在哪里吗") or a request ("One sentence, no tools.", "请...", "一句话..."). A plain
    statement opening with the letters of a request word ("Now I live in Suzhou.", "No, my
    sister is 林雪.", "Justin is my brother.", "写作是我的爱好。") states a fact, and so does a
    question with a stating clause ("我上周搬到苏州了，附近有什么好吃的？").

    Args:
        text: A user part's text.

    Returns:
        True for a part with at least one sentence, none of them stating a fact.
    """
    said = [piece.strip() for piece in _CLAUSE_BREAK.split(text) if piece.strip()]
    return bool(said) and not any(states_fact(piece) for piece in said)


def asks_own(query: str) -> bool:
    """Whether a query asks what the user asked ("What did I ask you yesterday?"), so a
    question part may answer it (`fact_free`)."""
    return _OWN_QUESTION.search(query) is not None


@functools.lru_cache(maxsize=1 << 16)
def change_cue(text: str) -> bool:
    """Whether a text carries a correction cue (`CHANGE_CUES`), memoized."""
    return any(cue in text for cue in _HAN_CUES) or _LATIN_CUES.search(text) is not None


def sentences(text: str) -> list[tuple[int, int]]:
    """The character ranges of a text's sentences, without surrounding white space."""
    ranges: list[tuple[int, int]] = []
    for sentence in SENTENCE.finditer(text):
        piece = sentence.group()
        start = sentence.start() + len(piece) - len(piece.lstrip())
        end = sentence.end() - (len(piece) - len(piece.rstrip()))
        if start < end:
            ranges.append((start, end))
    return ranges


_TITLES: Final = frozenset(
    "mr mrs ms mx dr prof sr jr st mt rev hon fr gen col capt lt sgt gov sen rep".split())
_ABBREVIATIONS: Final = frozenset(
    "etc vs e.g i.e eg ie a.m p.m u.s u.k u.n no nos inc ltd co corp approx dept est fig vol "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split())
_BOUNDARY: Final = re.compile(r"[.!?;]+(?=\s|$)|[。！？；]+|\n+")


def request_sentences(text: str) -> list[tuple[int, int]]:
    """The character ranges of a user message's sentences as the plugins split it to find
    forget requests (``integrations/core/gate.mjs``'s ``sentences``), without surrounding white
    space: a Latin mark ends a sentence only before a space or the end ("v2.3",
    "a.b@example.com"), a period after a title or a single initial never ("Dr. Sato",
    ``J. Smith``), and after an abbreviation only before a capital ("etc.", ``U.S.``)."""
    pieces: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        if match.group() == ".":
            words = text[start:match.start()].split()
            word = words[-1] if words else ""
            bare = word.lstrip("(\"'“‘").lower()
            following = text[end:].lstrip()[:1]
            if bare in _TITLES or (len(word) == 1 and word.isupper()):
                continue
            if bare in _ABBREVIATIONS and following and not following.isupper():
                continue
        if end > start:
            pieces.append((start, end))
        start = end
    if start < len(text):
        pieces.append((start, len(text)))
    ranges: list[tuple[int, int]] = []
    for first, last in pieces:
        piece = text[first:last]
        lead = len(piece) - len(piece.lstrip())
        trail = len(piece) - len(piece.rstrip())
        if first + lead < last - trail:
            ranges.append((first + lead, last - trail))
    return ranges


def _han_bigrams(text: str) -> list[str]:
    """The adjacent Han, Kana and Hangul character pairs of a text, in order."""
    terms = sparse_terms(text)
    return [term for term in terms if len(term) == 2 and all(own_token(c) for c in term)]


def _names(text: str, *, anywhere: bool) -> set[str]:
    """Capitalized Latin words of four letters or more that are names, as their terms, and
    the shorter given names a text gives a person or an animal (`_SHORT_NAMED`: "A person is
    named Tom.", "Amy is the user's coworker.", "Ben's name").

    A word that opens a sentence is a name only when ``anywhere`` (a Han-dominant text,
    where a Latin word is a name wherever it stands); a date word never is.
    """
    found: set[str] = set()
    for match in _CAPITALIZED.finditer(text):
        before = text[:match.start()].rstrip(" \t\"'“‘(「『-—–")
        if not anywhere and (not before or before[-1] in _SENTENCE_END):
            continue
        for term in token_seq(match.group()):
            if len(term) >= 4 and term not in CALENDAR and term not in ORG_SUFFIX:
                found.add(term)
    for pattern in _SHORT_NAMED:
        for match in pattern.finditer(text):
            if match["name"].lower() not in SHORT_NOT_NAMES:
                found.add(match["name"].lower())
    return found


@dataclass(frozen=True, slots=True)
class ForgetShadowTerms:
    """The plain shadow terms of forgotten texts, before they are digested.

    Attributes:
        names: Name terms: a hit holding one is about the forgotten fact.
        topics: Topic keywords, each the set of its terms: a hit holding every term of two
            of them is about the forgotten fact.
    """

    names: frozenset[str] = frozenset()
    topics: tuple[frozenset[str], ...] = ()

    def __or__(self, other: ForgetShadowTerms) -> ForgetShadowTerms:
        """Both texts' terms, topics in first-seen order."""
        return ForgetShadowTerms(self.names | other.names,
                                 tuple(dict.fromkeys((*self.topics, *other.topics))))


def shadow_terms(text: str, keywords: Sequence[str] = ()) -> ForgetShadowTerms:
    """The shadow terms of one forgotten statement or sentence and its keywords.

    Names are the capitalized Latin words of four letters or more, and the given names of two
    or three letters a text gives someone (`_names`), that do not open a sentence (any of them
    in a mostly Han text),
    less `CALENDAR` and `ORG_SUFFIX`; in a mostly Latin text, a Han run is a name (an
    employer, a person) and gives its bigrams, less `ORG_SUFFIX`. A mostly Han text gives a
    topic keyword per Han bigram, less `ORG_SUFFIX` and `HAN_STOP`. Each keyword with a name
    word gives its names; any other keyword is one topic keyword (its terms less `STOP`),
    unless it says only `GENERIC` words.

    Args:
        text: A forgotten statement or a sentence stating a forgotten fact.
        keywords: The statement's keywords, if any.

    Returns:
        The text's shadow terms.
    """
    han = sum(len(term) for term in token_seq(text) if own_token(term))
    latin = len(_LATIN_LETTER.findall(text))
    mostly_han = han > latin
    names = _names(text, anywhere=mostly_han)
    topics: list[frozenset[str]] = []
    bigrams = [term for term in _han_bigrams(text) if term not in ORG_SUFFIX]
    if mostly_han:
        topics.extend(frozenset(sparse_terms(term)) for term in bigrams if term not in HAN_STOP)
    else:
        names.update(bigrams)
    for keyword in keywords:
        named = _names(keyword, anywhere=True)
        if named:
            names |= named
            continue
        terms = frozenset(term for term in sparse_terms(keyword) if term not in STOP)
        if terms and not terms <= GENERIC:
            topics.append(terms)
    return ForgetShadowTerms(frozenset(names), tuple(dict.fromkeys(topics)))


_NO_TIMES: Final[Mapping[str, int]] = MappingProxyType({})
_NO_TOPIC_TIMES: Final[Mapping[frozenset[str], int]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ForgetShadow:
    """What a Search withholds as being about a forgotten fact (`forget_shadowed`).

    Attributes:
        names: Name term keys (see ``digest``).
        topics: Topic keywords, each the keys of its terms.
        vectors: Unit vectors of forgotten statements in the Search's embedding space.
        digest: The key of a term: the workspace's keyed digest for recorded shadows, so
            the shadow names no forgotten word; the identity for plain terms.
        name_times: When a complete forget (the ``agent`` Add mode) last added a name; a
            name without one is never lifted.
        topic_times: The same for topic keywords.
        topic_seen: When the store had received what the complete forget that last added
            a topic keyword read, on the Writer's clock (a recorded keyword: its
            ``topic_times``). A hit already stored then is not hidden by that keyword: a
            kept memory sharing "drink" and "preference" with the forgotten tea. A keyword
            without one (a marked sentence's) hides every hit; a request's is seen when the
            request came. A name has no such time: it hides an older hit too, unless the
            forget that stamped it judged
            and kept that hit (`forget_shadowed`'s ``kept``).
    """

    names: frozenset[str] = frozenset()
    topics: tuple[frozenset[str], ...] = ()
    vectors: tuple[UnitVector, ...] = ()
    digest: Callable[[str], str] = field(default=_identity, compare=False)
    name_times: Mapping[str, int] = field(default=_NO_TIMES, compare=False)
    topic_times: Mapping[frozenset[str], int] = field(default=_NO_TOPIC_TIMES, compare=False)
    topic_seen: Mapping[frozenset[str], int] = field(default=_NO_TOPIC_TIMES, compare=False)

    @classmethod
    def of(cls, terms: ForgetShadowTerms, digest: Callable[[str], str] = _identity,
           vectors: Sequence[UnitVector] = (), *, at: int | None = None) -> ForgetShadow:
        """Key plain shadow terms with ``digest``; ``at`` stamps every term."""
        names = frozenset(map(digest, terms.names))
        topics = tuple(frozenset(map(digest, topic)) for topic in terms.topics)
        if at is None:
            return cls(names, topics, tuple(vectors), digest)
        return cls(names, topics, tuple(vectors), digest, dict.fromkeys(names, at),
                   dict.fromkeys(topics, at))

    def __bool__(self) -> bool:
        return bool(self.names or self.topics or self.vectors)

    def joined(self, other: ForgetShadow) -> ForgetShadow:
        """Both shadows under this one's ``digest`` (``other`` must use the same); a term
        both stamp keeps the later time, and one only one of them names keeps that one's."""
        def merged[K](mine: Mapping[K, int], theirs: Mapping[K, int]) -> Mapping[K, int]:
            if not theirs:
                return mine
            joined = dict(mine)
            for key, when in theirs.items():
                joined[key] = max(when, joined.get(key, when))
            return joined

        # A keyword either side names without a seen time hides every hit.
        unseen = {topic for side in (self, other) for topic in side.topics
                  if topic not in side.topic_seen}
        seen = {topic: when for topic, when in merged(self.topic_seen, other.topic_seen).items()
                if topic not in unseen}
        return ForgetShadow(self.names | other.names,
                            tuple(dict.fromkeys((*self.topics, *other.topics))),
                            (*self.vectors, *other.vectors), self.digest,
                            merged(self.name_times, other.name_times),
                            merged(self.topic_times, other.topic_times), seen)

    def without_names(self, kept: Iterable[str]) -> ForgetShadow:
        """This shadow less the name keys in ``kept``."""
        return replace(self, names=self.names - frozenset(kept))


def forget_shadowed(
    terms: frozenset[str], vector: UnitVector | None, is_memory: bool, shadow: ForgetShadow,
    lifted: Callable[[frozenset[str], int], bool] | None = None, stored: int | None = None,
    kept: Callable[[str], bool] | None = None,
) -> bool:
    """Whether a candidate hit is about a forgotten fact, so the interactive Search drops it.

    A candidate is shadowed when (a) one of its terms is a shadow name, (b) it holds every
    term of two or more topic keywords, or (c) it is a memory whose vector lies within
    `SHADOW_COSINE` of a forgotten statement's. A name or topic keyword a complete forget
    stamped that ``lifted`` lifts (the user stated it again, in the candidate's own words,
    after that forget: ``core/relearn.py``) does not count: what the user
    says now is not what the forget left. A candidate whose words lift nothing is still
    compared by vector.

    What a complete forget left is not hidden by it either, but only as far as that forget
    read it. A name does not count when ``kept`` says the forget that stamped it judged the
    candidate and kept it, every readable sentence of it naming that forget's names
    included (a correction naming the old city); a name hides every other candidate,
    older or not (an older message naming the forgotten pet that no pick read). A topic
    keyword does not count for a candidate ``stored`` before its forget read the scope
    (``topic_seen``): a memory kept beside the forgotten one ("The user never drinks
    coffee." sharing "drink" and "preference" with the forgotten tea). A request's own
    terms have neither, so "Please forget X" hides X before its forget has run.

    Args:
        terms: The candidate's `sparse_terms` (a memory's statement, keywords and quotes,
            an Evidence part's text).
        vector: The candidate's unit vector, if any.
        is_memory: Whether the candidate is a memory (only memories are compared by vector).
        shadow: The scope's forget shadow.
        lifted: Given a term's keys and its stamp, whether the user told it again in the
            candidate's words after that time; None lifts nothing.
        stored: When the store received the candidate (a memory's creation, a part's
            receipt), on the Writer's clock, or None.
        kept: Given a name key, whether the complete forget that stamped it judged and kept
            the candidate; None keeps none.

    Returns:
        True when the candidate must not be delivered.
    """
    if not shadow:
        return False
    keys = frozenset(map(shadow.digest, terms))
    lifts = 0

    def holds(key: frozenset[str], at: int | None) -> bool:
        """Whether a shadow term still shadows the candidate: not lifted for it."""
        nonlocal lifts
        if lifted is None or at is None or not lifted(key, at):
            return True
        lifts += 1
        return False

    def left(topic: frozenset[str]) -> bool:
        """Whether a topic keyword's forget read the scope after the candidate was stored."""
        seen = shadow.topic_seen.get(topic)
        return stored is not None and seen is not None and stored < seen

    if any([holds(frozenset({name}), shadow.name_times.get(name))
            for name in keys & shadow.names if kept is None or not kept(name)]):
        return True
    if sum(1 for topic in shadow.topics
           if topic <= keys and not left(topic)
           and holds(topic, shadow.topic_times.get(topic))) >= 2:
        return True
    return is_memory and vector is not None and not lifts and any(
        cosine(vector, other) >= SHADOW_COSINE for other in shadow.vectors)


@dataclass(frozen=True, slots=True)
class AgentHit:
    """One ranked candidate, or one memory or user part of the scope, as the rules read it.

    Attributes:
        key: The memory or Evidence id.
        kind: ``memory`` or ``evidence``.
        final: Its score.
        text: The memory's statement or the part's text.
        role: The part's role (Evidence only).
        anchor: The memory's primary entity (memories only).
        state: Whether the memory states a state (kind ``state``).
        current: Whether the store still holds the memory: its row is active and, for a
            state, its validity has not ended before the query time. One that is not is
            ``past``.
        support: The newest commit supporting it: a memory's newest quoted part's ingest
            commit, a part's own.
        quotes: The memory's quote spans as (evidence id, byte start, byte end).
        span: The part's own (evidence id, byte start, byte end).
        cue: Whether a user part it quotes (or the part itself) carries a change cue.
        supersedes: The memories this memory's head ends through its ``supersedes``
            links (an update, correction or retraction the Add judge decided).
        contradicts: The memories this memory's head ``contradicts`` (a conflict the Add
            judge saw but did not resolve).
    """

    key: str
    kind: Literal["memory", "evidence"]
    final: Decimal
    text: str = ""
    role: str | None = None
    anchor: str = ""
    state: bool = False
    current: bool = True
    support: int = 0
    quotes: tuple[tuple[str, int, int], ...] = ()
    span: tuple[str, int, int] | None = None
    cue: bool = False
    supersedes: frozenset[str] = frozenset()
    contradicts: frozenset[str] = frozenset()


type Cosines = Callable[[AgentHit, Sequence[AgentHit]], Sequence[float]]
"""The cosine of one hit with each of several others, in order."""


def _jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


@functools.lru_cache(maxsize=1 << 16)
def _statement_terms(text: str) -> frozenset[str]:
    """`content_terms` of a statement, memoized: every Search reads the scope's statements."""
    return content_terms(text)


def asks_back(query: str) -> bool:
    """Whether a query asks what the assistant said, in the wording `ASKS_BACK` adds."""
    return ASKS_BACK.search(query) is not None


def ended(memories: Iterable[AgentHit]) -> frozenset[str]:
    """The memories of a scope that are ``past``: the store no longer holds them
    (`AgentHit.current`), or a current memory of the scope ``supersedes`` them."""
    found: set[str] = set()
    for memory in memories:
        if not memory.current:
            found.add(memory.key)
        else:
            found.update(memory.supersedes)
    return frozenset(found)


@dataclass(frozen=True, slots=True)
class Standing:
    """Each memory's `Status` in the store, and what replaces or may replace it.

    Attributes:
        status: The status of every memory asked about.
        replacers: For a ``past`` or ``possibly_outdated`` memory, the memories and user
            parts that replace or may replace it, all newer than it.
    """

    status: Mapping[str, Status]
    replacers: Mapping[str, tuple[str, ...]]


def standing(
    asked: Sequence[AgentHit], memories: Sequence[AgentHit], parts: Sequence[AgentHit],
    cosines: Cosines,
) -> Standing:
    """Read each asked memory's `Status` from the whole scope.

    A memory is ``past`` when the store no longer holds it (`AgentHit.current`) or when a
    current memory of the scope ``supersedes`` it; what replaces it is those memories, and,
    when none of them is a state (the judge linked only the event of the change, "moved from
    Hangzhou to Suzhou"), every newer live state memory of its anchor within
    `CORRECTION_COSINE` of it: what holds now. A state memory that is not past
    is ``possibly_outdated`` when a newer live memory ``contradicts`` it, when a newer live
    state memory of its anchor restates it with other content (content-term Jaccard from
    `SAME_STATE_JACCARD` up to, not including, `NEAR_SAME_JACCARD`) or carries a change cue
    within `CORRECTION_COSINE` of it, or when a newer user part with a change cue that it
    does not quote lies within `CORRECTION_COSINE` of it. A live memory is current and not
    past (`ended`): a retracted or replaced memory questions nothing. A change the Add judge
    already settled is not read again: a memory that ``supersedes`` another, a user part such
    a memory quotes, and a user part a past memory quotes question no third memory ("Since
    the move I climb instead of playing badminton" ends the badminton memory, not the new
    address). Every other memory is ``current``. The other hits of a Search play no part, so
    a memory has one status at one pin.

    Args:
        asked: The memories whose status is wanted.
        memories: Every memory of the scope the Search may read.
        parts: Every user part of the scope the Search may read that states a fact
            (not `fact_free`).
        cosines: The cosine of a memory with each of several memories or parts.

    Returns:
        The statuses and replacers.
    """
    return StandingReader(memories, parts, cosines).read(asked)


def _same_state_prefix(terms: frozenset[str], rank: Mapping[str, tuple[int, str]]) -> list[str]:
    """The prefix of ``terms`` (rarest first by ``rank``) two term sets reaching
    `SAME_STATE_JACCARD` always share a term of.

    Jaccard t means an overlap of at least ``t * max(|A|, |B|)`` terms. Then the rarest
    common term lies within the first ``|X| - floor(t * |X|) + 1`` terms of either set,
    since every other common term follows it; ``floor`` only lengthens the prefix.
    """
    ordered = sorted(terms, key=rank.__getitem__)
    return ordered[:len(ordered) - int(SAME_STATE_JACCARD * len(ordered)) + 1]


def _by_evidence(
    quotes: Iterable[tuple[str, int, int]],
) -> dict[str, list[tuple[int, int]]]:
    found: dict[str, list[tuple[int, int]]] = {}
    for evidence_id, low, high in quotes:
        found.setdefault(evidence_id, []).append((low, high))
    return found


class StandingReader:
    """The scope's links and change cues, read once per Search, for `standing`.

    Building it reads every memory and user part of the scope once; `read` then costs only
    the memories asked about, so a Search that asks twice (for what it selected, then for
    what joined) pays the scope once.

    A listing asks about every memory of the scope. It passes ``reach``: for an asked
    memory, the keys whose cosine with it may reach `CORRECTION_COSINE` (every other is
    below). Then a memory is compared only with those keys and with the memories of its
    anchor whose rarest-first term prefix meets its own (`_same_state_prefix`; a pair
    sharing none of those terms stays below `SAME_STATE_JACCARD`), in the scope's own
    order, so every status and replacer is the one read without ``reach``.
    """

    def __init__(self, memories: Sequence[AgentHit], parts: Sequence[AgentHit],
                 cosines: Cosines,
                 reach: Callable[[AgentHit], Collection[str]] | None = None) -> None:
        """Index the scope; see `standing` and the class docstring for the arguments."""
        gone = ended(memories)
        self._cosines = cosines
        self._reach = reach
        self._places: dict[int, dict[str, int]] = {}
        self._prefixes: dict[str, tuple[dict[str, tuple[int, str]], dict[str, set[str]]]] = {}
        self._states_keys = {memory.key for memory in memories if memory.state}
        self._support = {memory.key: memory.support for memory in memories}
        self._endings: dict[str, list[str]] = {}
        self._conflicts: dict[str, list[str]] = {}
        self._by_anchor: dict[str, list[AgentHit]] = {}
        self._states: dict[str, list[AgentHit]] = {}
        settled: list[tuple[str, int, int]] = []
        for memory in memories:
            if memory.current:
                for target in sorted(memory.supersedes):
                    self._endings.setdefault(target, []).append(memory.key)
            if not memory.current or memory.key in gone:
                settled.extend(memory.quotes)
                continue
            if memory.supersedes:
                settled.extend(memory.quotes)
            if memory.state and memory.anchor:
                self._states.setdefault(memory.anchor, []).append(memory)
                if not memory.supersedes:
                    self._by_anchor.setdefault(memory.anchor, []).append(memory)
            for target in sorted(memory.contradicts):
                self._conflicts.setdefault(target, []).append(memory.key)
        self._terms: dict[str, dict[str, set[str]]] = {}
        self._sorted: dict[int, tuple[list[int], list[int]]] = {}
        index = _by_evidence(settled)
        self._cued = [part for part in parts if part.cue and (
            part.span is None or not any(
                low < part.span[2] and part.span[1] < high
                for low, high in index.get(part.span[0], ())))]

    def read(self, asked: Sequence[AgentHit]) -> Standing:
        """The status and replacers of each asked memory (`standing`)."""
        status: dict[str, Status] = {}
        replacers: dict[str, tuple[str, ...]] = {}
        for memory in asked:
            if memory.kind != "memory" or memory.key in status:
                continue
            found = list(self._endings.get(memory.key, ()))
            if found or not memory.current:
                status[memory.key] = "past"
                if memory.state and memory.anchor and not any(
                        key in self._states_keys for key in found):
                    states = self._states.get(memory.anchor, ())
                    if self._reach is not None:
                        states = self._within(states, self._reach(memory))
                    newer = [other for other in states
                             if other.key != memory.key and other.support > memory.support
                             and other.key not in found]
                    if newer:
                        found.extend(other.key for other, value in zip(
                            newer, self._cosines(memory, newer), strict=True)
                            if value >= CORRECTION_COSINE)
                if found:
                    replacers[memory.key] = tuple(found)
                continue
            if memory.state:
                found = self._outdating(memory)
            if found:
                status[memory.key] = "possibly_outdated"
                replacers[memory.key] = tuple(dict.fromkeys(found))
            else:
                status[memory.key] = "current"
        return Standing(status, replacers)

    def _newer(self, members: Sequence[AgentHit], support: int,
               cache: dict[int, tuple[list[int], list[int]]]) -> list[AgentHit]:
        """The members newer than ``support``, in their own order."""
        order = cache.get(id(members))
        if order is None:
            places = sorted(range(len(members)), key=lambda place: members[place].support)
            order = cache[id(members)] = (
                places, [members[place].support for place in places])
        places, supports = order
        return [members[place]
                for place in sorted(places[bisect.bisect_right(supports, support):])]

    def _within(self, members: Sequence[AgentHit], keys: Iterable[str]) -> list[AgentHit]:
        """The members of ``keys``, in their own order."""
        places = self._places.get(id(members))
        if places is None:
            places = self._places[id(members)] = {
                member.key: place for place, member in enumerate(members)}
        found = sorted({places[key] for key in keys if key in places})
        return [members[place] for place in found]

    def _prefix_sharing(self, anchor: str, words: frozenset[str]) -> set[str]:
        """The memories of ``anchor`` whose rarest-first term prefix meets that of ``words``."""
        found = self._prefixes.get(anchor)
        if found is None:
            members = self._by_anchor.get(anchor, ())
            counts: dict[str, int] = {}
            for member in members:
                for term in _statement_terms(member.text):
                    counts[term] = counts.get(term, 0) + 1
            rank = {term: (count, term) for term, count in counts.items()}
            index: dict[str, set[str]] = {}
            for member in members:
                for term in _same_state_prefix(_statement_terms(member.text), rank):
                    index.setdefault(term, set()).add(member.key)
            found = self._prefixes[anchor] = (rank, index)
        rank, index = found
        ranked = {term: rank.get(term, (0, term)) for term in words}
        return {key for term in _same_state_prefix(words, ranked)
                for key in index.get(term, ())}

    def _outdating(self, memory: AgentHit) -> list[str]:
        """What questions a live state memory: newer conflicts, restatements and cues."""
        found = [key for key in self._conflicts.get(memory.key, ())
                 if self._support.get(key, 0) > memory.support]
        words = _statement_terms(memory.text)
        anchored: Sequence[AgentHit] = self._by_anchor.get(memory.anchor, ())
        index = self._terms.get(memory.anchor)
        if index is None:
            index = self._terms[memory.anchor] = {}
            for other in anchored:
                for term in _statement_terms(other.text) - GENERIC:
                    index.setdefault(term, set()).add(other.key)
        # A memory sharing no specific term with this one shares at most its generic ones
        # ("user"); when those are under half of its terms, the Jaccard stays below
        # `SAME_STATE_JACCARD` and need not be computed.
        generic = len(words & GENERIC)
        every = bool(words) and generic >= SAME_STATE_JACCARD * len(words)
        sharing = {key for term in words - GENERIC for key in index.get(term, ())}
        reach = None if self._reach is None else self._reach(memory)
        # Without ``reach`` every newer memory of the anchor is read. With it, one below
        # `SAME_STATE_JACCARD` can only join ``near``, and only within ``reach``.
        if reach is None:
            newer = self._newer(anchored, memory.support, self._sorted)
            cued = self._newer(self._cued, memory.support, self._sorted)
        else:
            newer = [other for other in self._within(
                anchored, self._prefix_sharing(memory.anchor, words) | set(reach))
                if other.support > memory.support]
            cued = [part for part in self._within(self._cued, reach)
                    if part.support > memory.support]
        near: list[AgentHit] = []
        for other in newer:
            if other.key == memory.key:
                continue
            overlap = (_jaccard(words, _statement_terms(other.text))
                       if every or other.key in sharing else 0.0)
            if overlap >= NEAR_SAME_JACCARD:
                continue
            if overlap >= SAME_STATE_JACCARD:
                found.append(other.key)
            elif other.cue:
                near.append(other)
        own = _by_evidence(memory.quotes)
        near.extend(part for part in cued if (
            part.span is None or part.span[0] not in own or not any(
                low < part.span[2] and part.span[1] < high for low, high in own[part.span[0]])))
        if near:
            found.extend(other.key for other, value in zip(
                near, self._cosines(memory, near), strict=True) if value >= CORRECTION_COSINE)
        return found


@dataclass(frozen=True, slots=True)
class Drawn:
    """The memories delivered with the user parts that quote-source them (`draw_quoting`).

    Attributes:
        memories: Each memory to deliver, with the score it takes: the part's, or None for
            its own.
        dropped: The parts not delivered: every open memory quoting them is ``past``.
    """

    memories: tuple[tuple[str, Decimal | None], ...]
    dropped: frozenset[str]


def draw_quoting(
    parts: Sequence[tuple[str, Decimal]], quoting: Mapping[str, Sequence[str]],
    status: Mapping[str, Status], relevances: Mapping[str, Decimal], chosen: Collection[str],
) -> Drawn:
    """The memories a delivered user part brings with it.

    The part matched the query, and the memories quoting it state what it said. When every
    one of them no longer holds or is questioned (not ``current``), the part is only about
    them: they are delivered at the part's score with their statuses, so its words never
    stand without them ("Hi! 我住在杭州" beside the past Hangzhou memory and, above it, what
    replaced it), and a part whose memories are all ``past`` is not delivered. When the part
    also states what still holds, the query may have matched another of its clauses ("我叫
    周以宁，在上海做数据工程师，住在上海静安区" for "我做什么工作？"): a memory quoting it is
    delivered only when its own relevance reaches `RELEVANCE_FLOOR` (though not the share
    of the best match, `RELATIVE_SHARE`), with its own score, and the part stays.

    Args:
        parts: The delivered user parts with their scores, in rank order.
        quoting: The open memories quoting each part.
        status: The status of every memory in ``quoting`` (`standing`).
        relevances: Each memory's own `relevance`.
        chosen: The memories already delivered.

    Returns:
        The memories to add and the parts to drop.
    """
    taken = set(chosen)
    memories: list[tuple[str, Decimal | None]] = []
    dropped: set[str] = set()
    for part, score in parts:
        keys = quoting.get(part, ())
        if not keys:
            continue
        sole = all(status[key] != "current" for key in keys)
        if all(status[key] == "past" for key in keys):
            dropped.add(part)
        for key in keys:
            if key in taken:
                continue
            if sole:
                memories.append((key, score))
            elif relevances[key] >= RELEVANCE_FLOOR:
                memories.append((key, None))
            else:
                continue
            taken.add(key)
    return Drawn(tuple(memories), frozenset(dropped))


def recency_order(
    hits: Sequence[AgentHit], replacers: Mapping[str, tuple[str, ...]], *,
    statements_first: bool = True,
) -> list[AgentHit]:
    """Order delivered candidates by score, statements first, replacements above.

    1. By score; every memory, then every Evidence part, each in score order (unless
       ``statements_first`` is False: a query asking what the assistant said).
    2. A memory, then a user part, that replaces or may replace a hit ranked above it
       (`Standing.replacers`) goes directly above the highest such hit, taking its score
       when that is higher; a user part does so only for a memory that no delivered
       memory replaces.

    Args:
        hits: The candidates, any order; assistant parts must already be dropped.
        replacers: What replaces or may replace each memory (`standing`).
        statements_first: Put every memory before every Evidence part.

    Returns:
        The ordered hits.
    """
    ranked = sorted(hits, key=lambda hit: -hit.final)
    order = [hit for hit in ranked if hit.kind == "memory"] + [
        hit for hit in ranked if hit.kind == "evidence"] if statements_first else ranked
    replaced: dict[str, set[str]] = {}
    for older, newer in replacers.items():
        for key in newer:
            replaced.setdefault(key, set()).add(older)
    movers = [hit for hit in order if hit.kind == "memory" and hit.key in replaced] + [
        hit for hit in order if hit.kind == "evidence" and hit.key in replaced]
    shown = {hit.key for hit in order if hit.kind == "memory"}
    for hit in movers:
        index = next(place for place, other in enumerate(order) if other.key == hit.key)
        above = [place for place, other in enumerate(order[:index])
                 if other.key in replaced[hit.key] and (hit.kind == "memory" or not any(
                     key in shown for key in replacers.get(other.key, ())))]
        if above:
            moved = order.pop(index)
            target = order[above[0]]
            order.insert(above[0], replace(moved, final=max(moved.final, target.final)))
    return order


def _covered(span: tuple[str, int, int], quotes: Iterable[tuple[str, int, int]]) -> int:
    """Bytes of ``span`` that the quotes cover (overlaps merged)."""
    evidence, start, end = span
    ranges = sorted((max(start, low), min(end, high)) for other, low, high in quotes
                    if other == evidence and low < end and start < high)
    covered, reach = 0, start
    for low, high in ranges:
        low = max(low, reach)
        if high > low:
            covered += high - low
            reach = high
    return covered


def quoted_share(span: tuple[str, int, int], quotes: Iterable[tuple[str, int, int]]) -> float:
    """The share of a part's bytes (``span``) that the quotes cover, overlaps merged."""
    size = span[2] - span[1]
    return _covered(span, quotes) / size if size > 0 else 0.0


def overlaps(span: tuple[str, int, int], quotes: Iterable[tuple[str, int, int]]) -> bool:
    evidence, start, end = span
    return any(other == evidence and low < end and start < high for other, low, high in quotes)


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _echoes(hit: AgentHit, kept: Sequence[AgentHit]) -> bool:
    """An Evidence part repeating a kept part: most of its terms, or a quoted user part."""
    terms = frozenset(sparse_terms(hit.text))
    for other in kept:
        if other.kind != "evidence":
            continue
        if _jaccard(terms, frozenset(sparse_terms(other.text))) >= ECHO_JACCARD:
            return True
        if hit.role == "assistant" and other.role == "user":
            quoted = _normalized(" ".join(_QUOTED_LINE.findall(hit.text)))
            said = _normalized(other.text)
            if len(quoted) >= MIN_ECHO_CHARS and (quoted in said or said in quoted):
                return True
    return False


def collapse(ranked: Sequence[AgentHit], count: int) -> list[AgentHit]:
    """Deliver at most ``count`` hits in rank order without near-duplicates.

    - A memory is passed over when a higher-ranked kept memory overlaps every one of
      its quote spans and states every content term it states (less `GENERIC`).
    - Of memories with the same anchor and kind whose content terms reach
      `NEAR_SAME_JACCARD`, only the one with the newest support is delivered.
    - An Evidence part is passed over when the quotes of the memories delivered before
      it cover `COVERED_SHARE` of its bytes (a memory delivered after it, or not at all,
      hides nothing).
    - An Evidence part is passed over when its sparse terms reach `ECHO_JACCARD` with a
      kept part, or when it is an assistant part whose ``> `` lines reproduce a kept user
      part.

    A passed-over hit's slot goes to the next candidate.

    Args:
        ranked: The candidates in delivery order.
        count: The most hits delivered.

    Returns:
        The delivered hits, in order.
    """
    words = {hit.key: content_terms(hit.text) - GENERIC for hit in ranked}
    memories = [hit for hit in ranked if hit.kind == "memory"]
    newest: set[str] = set()
    for hit in memories:
        if not any(other.key != hit.key and other.anchor == hit.anchor
                   and other.state == hit.state and other.support > hit.support
                   and _jaccard(words[other.key], words[hit.key]) >= NEAR_SAME_JACCARD
                   for other in memories):
            newest.add(hit.key)
    statements: list[AgentHit] = []
    for hit in memories:
        if hit.key not in newest:
            continue
        if any(all(overlaps(quote, other.quotes) for quote in hit.quotes) and hit.quotes
               and words[hit.key] <= words[other.key] for other in statements):
            continue
        statements.append(hit)
    stated = {hit.key for hit in statements}
    quoted: list[tuple[str, int, int]] = []
    kept: list[AgentHit] = []
    for hit in ranked:
        if len(kept) == count:
            break
        if hit.kind == "memory":
            if hit.key in stated:
                kept.append(hit)
                quoted.extend(hit.quotes)
            continue
        span = hit.span
        if span is not None and span[2] > span[1] and (
                _covered(span, quoted) >= COVERED_SHARE * (span[2] - span[1])):
            continue
        if _echoes(hit, kept):
            continue
        kept.append(hit)
    return kept


def non_increasing[T](items: Sequence[T], score: Callable[[T], Decimal],
                      rescored: Callable[[T, Decimal], T]) -> list[T]:
    """The items with each score lowered to at most the one before it."""
    result: list[T] = []
    ceiling: Decimal | None = None
    for item in items:
        value = score(item)
        if ceiling is not None and value > ceiling:
            item = rescored(item, ceiling)
            value = ceiling
        result.append(item)
        ceiling = value
    return result


def surviving_names(
    shadow: ForgetShadow, memories: Iterable[frozenset[str]],
) -> frozenset[str]:
    """The shadow name keys `NAME_KEEP_MEMORIES` or more surviving memories still hold.

    Args:
        shadow: The scope's shadow.
        memories: Each surviving memory's term keys.

    Returns:
        The name keys to leave out of the shadow.
    """
    counts: dict[str, int] = dict.fromkeys(shadow.names, 0)
    for keys in memories:
        for name in keys & shadow.names:
            counts[name] += 1
    return frozenset(name for name, count in counts.items() if count >= NAME_KEEP_MEMORIES)


def request_ranges(
    text: str, requests: Callable[[str], Sequence[str]],
    reach: Callable[[str], int | None] = len,
) -> tuple[list[tuple[int, int]], list[str]]:
    """The byte ranges of a user part's forget requests and what they ask to forget.

    Args:
        text: The part's text.
        requests: The statements a text, read as one user message, asks to forget.
        reach: Where the request of a request sentence ends in it (characters); the whole
            sentence by default. A clear replacement said after the request in the same
            sentence ("Forget my old address, I live at 27 Birch Lane now.") stays readable.

    Returns:
        The requests' UTF-8 byte ranges within ``text`` and their statements.
    """
    if not requests(text):
        return [], []
    ranges: list[tuple[int, int]] = []
    asked: list[str] = []
    for start, end in request_sentences(text):
        found = requests(text[start:end])
        if found:
            stop = start + (reach(text[start:end]) or end - start)
            ranges.append((len(text[:start].encode("utf-8")), len(text[:stop].encode("utf-8"))))
            asked.extend(found)
    if not ranges:
        # A request that refers to an earlier sentence of its message: the whole part.
        ranges.append((0, len(text.encode("utf-8"))))
        asked.extend(requests(text))
    return ranges, asked


def replacement_ranges(
    text: str, requests: Callable[[str], Sequence[str]], reach: Callable[[str], int | None],
) -> list[tuple[int, int]]:
    """The UTF-8 byte ranges of a user part told after a request in the request's own
    sentence: the clear replacement ("..., I live at 27 Birch Lane now.") `request_ranges`
    leaves readable, a fact the user told with the request, not a restatement of it."""
    if not requests(text):
        return []
    found: list[tuple[int, int]] = []
    for start, end in request_sentences(text):
        sentence = text[start:end]
        stop = reach(sentence) if requests(sentence) else None
        if stop is not None and stop < len(sentence.rstrip()):
            found.append((len(text[:start + stop].encode("utf-8")),
                          len(text[:end].encode("utf-8"))))
    return found


def kind_factor(role: str | None, recalls_assistant: bool) -> Decimal | None:
    """The share of its fused score an Evidence part keeps; None when it is dropped."""
    if role == "assistant":
        return ASSISTANT_FACTOR if recalls_assistant else None
    return EVIDENCE_FACTOR


def _no_requests(_text: str) -> tuple[str, ...]:
    return ()


@dataclass(frozen=True, slots=True)
class AgentContext:
    """What the interactive ranking reads besides the snapshot: the scope's forgets.

    Attributes:
        marks: The scope's marked sentences (``forget_marks:{s}``) as (evidence id, byte
            start, byte end).
        recorded: The recorded shadow (``forget_shadow:{s}``) with the workspace's
            keyed digest of a term, which also keys every other shadow term.
        requests: What a text, read as one user message, asks to forget (the in-chat
            forget's own request finder).
        asserts: The ``agent`` mode's statement test, or None.
        kept: What the scope's complete forgets judged and kept (``forget_kept:{s}``): for
            a ``("memory", id)`` or ``("evidence", id)`` hit, the clock windows of those
            forgets; a name one of them stamped does not hide the hit (`forget_shadowed`).
        unmatched: The keyed digests (``add_key``) of the Adds whose in-chat forget ran and
            matched nothing for any of their requests (``forget_add:{s}``): those requests'
            own words hide nothing more, as nothing was forgotten for them, so a memory the
            forget judged and kept stays found ("forget that I lived in Shanghai" beside
            "The user moved from Shanghai to Hangzhou."). Their sentences stay withheld.
        add_key: The keyed digest of an Add's ``request_id`` in ``unmatched``.
        reach: Where the request of a request sentence ends in it, in characters
            (`request_ranges`); the whole sentence by default.
    """

    marks: tuple[tuple[str, int, int], ...] = ()
    recorded: ForgetShadow = ForgetShadow()
    requests: Callable[[str], Sequence[str]] = _no_requests
    asserts: Callable[[str], bool] | None = None
    """Whether one user sentence states something (``ingestion/relearn.asserts``); given, and
    with a recorded shadow a complete forget stamped, a hit the user told again after a
    shadow term's forget is not shadowed by it (``core/relearn.py``)."""
    kept: Mapping[tuple[str, str], tuple[tuple[int, int], ...]] = field(
        default_factory=lambda: MappingProxyType({}))
    unmatched: frozenset[str] = frozenset()
    add_key: Callable[[str], str] = _identity
    reach: Callable[[str], int | None] = len


_FORGET_HINT: Final = re.compile(r"forget|remember|memor|record|notes|history|忘|记|清除",
                                 re.IGNORECASE)
"""Every request form names forgetting or remembering, or the memory, records, notes or
history it drops a fact from (``ingestion/forget_requests``)."""


def may_request(text: str) -> bool:
    """A cheap test that a text could hold a forget request (`AgentContext.requests`)."""
    return _FORGET_HINT.search(text) is not None
