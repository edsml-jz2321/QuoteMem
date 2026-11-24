"""Contradiction candidates for Search's conflict check; pure, no I/O.

A user sometimes says something never happened ("I've never met Kelly") and,
elsewhere, that it did ("I met Kelly at the book club"). A reader answering
from the delivered items then tends to pick one side. When a highly ranked
item carries such a negated user statement and another delivered item carries
the same speaker's statement that may affirm it, the pair is a candidate for
the model check (``bench_conflict_judge``); only a pair the check confirms
gets a note naming both statements and saying the record does not settle
which is correct. Shared words never assert a contradiction on their own:
"I never got a library card" and "how do I get the library app to renew
loans?" repeat a word, not a claim. A lexical nomination the check does not
confirm shows both statements as possibly conflicting, conditional on their
being about the same thing (`possible_conflict_note`), so the reader sees both
versions without a claim that they contradict. Nothing stored or ranked changes.

The lexical rule nominates first (`lexical_pairs`). A pair must be the same
claim, not merely share words:

- Both statements come from the same speaker. A conversation can store two
  people with the user role, and one person's "never" does not contradict
  another's "did". An unnamed statement counts as the one named speaker's
  only when the delivered user statements name at most one person.
- The negated clause makes a claim about what happened: a predicate (the
  first word the negation governs) with an object after it or a subject
  before it. A negated state of mind ("didn't know", "never thought") and a
  hypothetical, future or questioning negation ("would never", "if I had
  never", "I'll never", "why haven't I") make no such claim and take no part,
  nor does a negation limited in time ("haven't played in a while", "yet").
- The affirming statement repeats that predicate, as the same word, a close
  synonym ("taken a course" and "enrolled in a course") or one that implies
  it ("completed the course"), not denied, made hypothetical or only intended
  earlier in its clause ("considering attending", "help me build"); it
  neither asks a question nor makes a negated claim. It repeats the noun
  phrase the predicate takes and at least ``MIN_COVERAGE`` (or ``MIN_SHARED``)
  of the clause's objects, else of its subject.

A negated statement is paired with the highest-ranked items that affirm it,
memories and raw turns alike; within one item, with the sentence repeating
most of its nouns. Bracketed transcript metadata and a speaker label opening
a turn are not words of the statement. Pairs the lexical rule leaves unpaired
that merely share a word follow (`conflict_candidates`).
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self

TOP = 10
"""Only a negated statement among the first ``TOP`` delivered items is a candidate."""
MIN_COVERAGE = 0.5
"""Share of the negated clause's objects (or subject) the affirming statement must repeat."""
MIN_SHARED = 2
"""Shared objects that suffice below ``MIN_COVERAGE``: a long clause's two named objects."""
MAX_QUOTE = 400
CANDIDATES = 8
"""Most candidate pairs a response offers the model check (`conflict_candidates`)."""
PER_NEGATION = 2
"""Most candidate partners of one negated sentence, lexical and relaxed each."""

_NEGATION = re.compile(r"\b(?:never|haven't|hasn't|have not|has not|didn't|did not)\b")
_DENIAL = re.compile(r"(?:never|not|no|cannot|[a-z]+n't)")
"""A word that denies what follows it."""
_SCOPE_END = re.compile(r"[,;:.!?]|\b(?:but|so|which|because|though|although)\b")
"""Where the words a negation governs end: the clause's punctuation or connective."""
_CLAUSE_START = re.compile(
    r"[,;:.!?]|\b(?:and|but|so|which|that|who|because|though|although|when|while)\b"
)
"""Where a clause starts: after punctuation, a conjunction or a relative pronoun."""
_SPAN_START = re.compile(
    r"[,;:.!?]|\b(?:but|so|which|that|who|because|though|although|when|while)\b"
)
"""Where the words governing an affirming predicate end: a clause start other than
``and``, which joins verbs under one "not sure how to" or "help me"."""
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_ABBREVIATION = re.compile(r"(?:^|\s)(?:Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|Mt|vs|etc|e\.g|i\.e)\.$")
"""A title or abbreviation whose period ends no sentence: "Dr. Kaya"."""
_WORD = re.compile(r"[a-z0-9][a-z0-9']*")
_CODE = re.compile(r"\s*(?://|#|/\*|--|```|[<{}$])")
"""A sentence that opens like code or markup rather than prose."""
_TAGS = re.compile(r"\s*(?:\[[^\]\n]*\]\s*)+")
"""Bracketed transcript metadata opening a turn: ``[corpus][session_1][D1:3][text]``."""
_LABEL = re.compile(r"([A-Z][\w'.-]*(?: [A-Z][\w'.-]*){0,3}): ")
"""A transcript's speaker label at the start of a message: ``Blair: ...``."""
_STOP = frozenset(
    """a an the and or but if of to in on at for from by with about as into than then so that
    this these those it its i i'm i've i'd me my mine we our us you your he she they them their
    his her is are was were be been being am do does did done have has had having can could would
    should will shall may might must not no any some all each every just also too very really
    kinda sort up out over more most much many what which who whom whose when where why how there
    here such own same other another again ever still yet even because while since until though
    although don doesn isn aren wasn weren won wouldn couldn shouldn can't don't doesn't isn't
    i'll you're it's that's user user's assistant help want need like know think make get got
    going trying try wondering wonder sure one ones anything something everything anyone someone
    everyone""".split()
)
_LEAD = frozenset(
    """a an the any ever actually really even yet once truly personally previously before
    fully properly successfully formally officially seriously directly been""".split()
)
"""Words between a negation and its predicate: articles, adverbs and ``been``."""
_PLACE = frozenset("to in at on inside".split())
"""After ``been``, a preposition makes the predicate a visit: "never been to Lisbon"."""
_HYPOTHETICAL = frozenset(
    """if unless would could should might may will can must wish whether why how i'll you'll
    we'll he'll she'll they'll won't""".split()
)
"""A word before a negation in its clause that makes it hypothetical, future or a question."""
_INTENT = frozenset(
    """considering consider trying planning plan plans goal hoping hope want wants going intend
    intending aim aiming help need keen eager thinking""".split()
)
"""A word before an affirming predicate in its clause that makes it an intention or a request."""
_ASKING = frozenset(
    """what which where when who whom whose why how do does did is are was were have has had can
    could would should will shall may might""".split()
)
"""A question opening with one of these asks; it affirms nothing."""
_BOUNDED = re.compile(
    r"\b(?:yet|lately|recently|anymore|in (?:a while|ages|years|months|weeks|a long time)"
    r"|for a while|this (?:week|month|year|morning|weekend)|today)\b"
)
"""A time limit on a negation ("not ... in a while", "yet", "this week"): an affirmation
outside that window does not contradict it, and the record rarely says which window it is."""
_PRONOUN = frozenset("i you we he she they it".split())
"""A pronoun right after a negation inverts a question: "didn't you say ...?"."""
_PREPOSITION = frozenset(
    """in on at for from with about to into of by after before during than over under through via
    near like since until within without against between""".split()
)
"""Where the noun phrase a predicate takes as its object ends."""
_IRREGULAR = {
    form: base
    for base, forms in {
        "meet": "met", "take": "took taken", "go": "went gone goes going", "do": "did done",
        "use": "used uses using", "get": "got gotten", "make": "made", "buy": "bought",
        "see": "saw seen",
        "write": "wrote written", "give": "gave given", "run": "ran", "begin": "began begun",
        "feel": "felt", "hear": "heard", "hold": "held", "keep": "kept", "leave": "left",
        "lose": "lost", "pay": "paid", "send": "sent", "spend": "spent", "sell": "sold",
        "tell": "told", "think": "thought", "win": "won", "build": "built", "bring": "brought",
        "catch": "caught", "choose": "chose chosen", "drive": "drove driven", "eat": "ate eaten",
        "fly": "flew flown", "find": "found", "teach": "taught", "read": "read", "set": "set",
        "put": "put", "forget": "forgot forgotten", "understand": "understood",
        "have": "had has", "be": "been was were",
    }.items()
    for form in forms.split()
}
"""Irregular English verb forms folded to their base, so "met" repeats "meet"."""
_STATE = frozenset(
    """know sure remember recall think understand realize realise notice expect decide believe
    imagine mind care wonder mean forget consider figure guess suppose doubt""".split()
)
"""Predicates of a state of mind; their negation says nothing about what happened."""
_SYNONYMS = (
    "take enroll attend join register",
    "complete finish",
    "write implement build create code develop add",
    "obtain get acquire receive request generate",
    "use try employ utilize adopt",
    "visit go travel attend",
    "buy purchase order",
    "learn study",
    "install deploy configure",
    "start begin launch",
    "fail crash break",
    "eat taste",
    "tell share mention",
)
"""Close synonyms a predicate may be repeated by."""
_IMPLIED = (("take enroll attend join register", "complete finish"),)
"""Predicates whose affirmation implies another's: completing a course means having taken it."""
_BARE = frozenset("have be do make own".split())
"""Verbs too common to count as an affirming statement's predicate on their own."""


def _fold(word: str) -> str:
    """Fold an English word to a stem shared by its inflections ("obtained" and "obtain")."""
    word = _IRREGULAR.get(word, word)
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if word.endswith("i"):
                word = word[:-1] + "y"
            break
    if len(word) >= 4 and word[-1] == word[-2] and word[-1] not in "aeiousl":
        word = word[:-1]
    return word[:-1] if len(word) >= 4 and word.endswith("e") else word


_CLASSES = tuple(frozenset(_fold(word) for word in line.split()) for line in _SYNONYMS)
_IMPLIERS = tuple(
    (frozenset(map(_fold, implied.split())), frozenset(map(_fold, by.split())))
    for implied, by in _IMPLIED
)
_FOLDED_STATE = frozenset(_fold(word) for word in _STATE)
_FOLDED_BARE = frozenset(_fold(word) for word in _BARE)
_VAGUE = frozenset(_fold(word) for word in "be do make".split())
"""Predicates too vague to name a claim: "never did that"."""


@dataclass(frozen=True, slots=True)
class Said:
    """One delivered item's text, whether the user said it, and who said it.

    ``speakers`` holds the speaker names the item is attributed to; it is empty
    when the source names none, as in a single-user transcript.
    """

    text: str
    user: bool
    speakers: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _Claim:
    """What a negated clause denies.

    ``predicate`` holds the accepted forms of its predicate, ``head`` the noun
    phrase right after it, ``objects`` every noun after it and ``subject`` the
    nouns before the negation.
    """

    predicate: frozenset[str]
    head: frozenset[str]
    objects: frozenset[str]
    subject: frozenset[str]


def _spoken(text: str) -> str:
    """A turn's text without its opening bracketed metadata."""
    match = _TAGS.match(text)
    return text if match is None else text[match.end() :]


def labelled_speaker(text: str) -> frozenset[str]:
    """The speaker named by a transcript label (``Name: ...``) opening a message, if any."""
    match = _LABEL.match(_spoken(text))
    return frozenset() if match is None else frozenset((match.group(1),))


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("\u2019", "'"))


def _content(tokens: Sequence[str]) -> frozenset[str]:
    return frozenset(
        _fold(word)
        for word in tokens
        if word not in _STOP and word not in _LEAD and len(word) > 2 and not _DENIAL.fullmatch(word)
    )


def _claims(sentence: str, *, relaxed: bool = False) -> list[_Claim]:
    """What each negation in a sentence denies, skipping states of mind and hypotheticals.

    ``relaxed`` (for `conflict_candidates`) also keeps a vague predicate ("never done
    any 5K runs") and a claim without nouns ("never practiced this before").
    """
    lowered = sentence.lower().replace("\u2019", "'")
    claims: list[_Claim] = []
    for match in _NEGATION.finditer(lowered):
        before = lowered[: match.start()]
        starts = [found.end() for found in _CLAUSE_START.finditer(before)]
        lead_in = _tokens(before[starts[-1] if starts else 0 :])
        if _HYPOTHETICAL.intersection(lead_in):
            continue
        rest = lowered[match.end() :]
        end = _SCOPE_END.search(rest)
        clause = rest if end is None else rest[: end.start()]
        words = _tokens(clause)
        if words and words[0] in _PRONOUN or _BOUNDED.search(clause):
            continue
        position = 0
        while position < len(words) and (
            words[position] in _LEAD or words[position : position + 2] == ["able", "to"]
        ):
            position += 2 if words[position] == "able" else 1
        if position == len(words):
            continue
        predicate = _fold(words[position])
        if position and words[position - 1] == "been" and words[position] in _PLACE:
            predicate = _fold("visit")
        if predicate in _FOLDED_STATE or (predicate in _VAGUE and not relaxed):
            continue
        forms = frozenset({predicate}).union(*(group for group in _CLASSES if predicate in group))
        forms = forms.union(*(by for implied, by in _IMPLIERS if predicate in implied))
        after = words[position + 1 :]
        stop = next(
            (at for at, word in enumerate(after) if at and word in _PREPOSITION), len(after)
        )
        head = _content(after[:stop])
        objects = _content(after)
        subject = _content(lead_in)
        if objects or subject or relaxed:
            claims.append(_Claim(forms - _FOLDED_BARE, head, objects, subject))
    return claims


@dataclass(frozen=True, slots=True)
class _Sentence:
    """One sentence of a delivered user statement, tokenized once, clause by clause."""

    text: str
    clauses: tuple[tuple[str, ...], ...]
    content: frozenset[str]
    asks: bool

    @classmethod
    def of(cls, text: str) -> Self:
        lowered = text.lower().replace("\u2019", "'")
        clauses = tuple(
            tuple(_tokens(part)) for part in _SPAN_START.split(lowered) if _tokens(part)
        )
        tokens = _tokens(lowered)
        asks = text.rstrip().endswith("?") and bool(tokens) and tokens[0] in _ASKING
        return cls(text, clauses, _content(tokens), asks)


def _repeats(claim: _Claim, sentence: _Sentence) -> int | None:
    """How many of a claim's nouns an affirming sentence repeats; None if it does not affirm it."""
    if sentence.asks or any(_NEGATION.search(" ".join(clause)) for clause in sentence.clauses):
        return None
    if not _states(claim.predicate, sentence):
        return None
    if claim.head and not claim.head & sentence.content:
        return None
    nouns = claim.objects or claim.subject
    shared = len(nouns & sentence.content)
    enough = shared / len(nouns) >= MIN_COVERAGE or shared >= MIN_SHARED
    return shared if shared and enough else None


def _states(predicate: frozenset[str], sentence: _Sentence) -> bool:
    """Whether a sentence uses one of the predicate forms as a fact: not denied, hypothetical
    or only intended earlier in its clause."""
    return any(
        _fold(word) in predicate
        and not any(
            _DENIAL.fullmatch(prior) or prior in _HYPOTHETICAL or prior in _INTENT
            for prior in clause[:index]
        )
        for clause in sentence.clauses
        for index, word in enumerate(clause)
    )


def _sentences(text: str) -> list[_Sentence]:
    body = _spoken(text)
    if (label := _LABEL.match(body)) is not None:
        body = body[label.end() :]
    parts: list[str] = []
    for part in _SENTENCE.split(body):
        if parts and _ABBREVIATION.search(parts[-1]):
            parts[-1] += " " + part
        else:
            parts.append(part)
    return [
        _Sentence.of(" ".join(part.split()))
        for part in parts
        if part.strip() and not _CODE.match(part)
    ]


def _quoted(sentence: str) -> str:
    return sentence if len(sentence) <= MAX_QUOTE else sentence[: MAX_QUOTE - 1] + "…"


def _same_speaker(items: Sequence[Said]) -> Callable[[Said, Said], bool]:
    named = {name for item in items if item.user for name in item.speakers}

    def same(one: Said, other: Said) -> bool:
        if one.speakers == other.speakers:
            return True
        return len(named) <= 1 and not (one.speakers and other.speakers)

    return same


@dataclass(frozen=True, slots=True)
class ConflictCandidate:
    """A negated user sentence and a same-speaker sentence that may contradict it.

    ``negated`` and ``affirmed`` are delivered item indexes, ``negated_text``
    and ``affirmed_text`` the two sentences.
    """

    negated: int
    negated_text: str
    affirmed: int
    affirmed_text: str


def conflict_candidates(items: Sequence[Said]) -> tuple[ConflictCandidate, ...]:
    """The pairs a response offers the model check of whether they conflict.

    The lexical pairs (`lexical_pairs`) come first, in their order. Then each
    negated user sentence of the first ``TOP`` items that makes a claim
    (`_claims`, relaxed) is offered with up to ``PER_NEGATION`` sentences of
    other first ``TOP`` items by the same speaker that negate nothing, ask
    nothing and share at least one content word or predicate form with it, most
    shared words first, then by rank, and, when none of those states its
    predicate as a fact (`_states`), the first such sentence of any delivered
    item that does; a pair already offered is skipped; at most ``CANDIDATES``
    pairs in all, negated sentences in rank order.

    Args:
        items: Delivered items in response order.

    Returns:
        The candidate pairs, possibly empty.
    """
    pairs = list(lexical_pairs(items))
    if len(pairs) >= CANDIDATES:
        return tuple(pairs[:CANDIDATES])
    sentences = [_sentences(item.text) if item.user else [] for item in items]
    same = _same_speaker(items)
    for index in range(min(TOP, len(items))):
        for negated in sentences[index]:
            claims = _claims(negated.text, relaxed=True)
            if not claims:
                continue
            words = negated.content.union(*(claim.predicate for claim in claims))
            predicate = frozenset[str]().union(*(claim.predicate for claim in claims))
            found: list[tuple[int, int, str, bool]] = []
            for other, candidates in enumerate(sentences):
                if other == index or not same(items[index], items[other]):
                    continue
                for affirmed in candidates:
                    if affirmed.asks or any(
                        _NEGATION.search(" ".join(clause)) for clause in affirmed.clauses
                    ):
                        continue
                    if shared := len(words & _overlap_words(affirmed)):
                        found.append((-shared, other, affirmed.text, _states(predicate, affirmed)))
            found.sort()
            chosen = [entry for entry in found if entry[1] < TOP][:PER_NEGATION]
            if not any(states for *_, states in chosen):
                # The first partner, at any rank, stating the denied predicate as a fact.
                chosen += [entry for entry in found if entry[3] and entry not in chosen][:1]
            for _, other, text, _ in chosen:
                pair = ConflictCandidate(index, negated.text, other, text)
                if pair in pairs:
                    continue
                pairs.append(pair)
                if len(pairs) == CANDIDATES:
                    return tuple(pairs)
    return tuple(pairs)


def _overlap_words(sentence: _Sentence) -> frozenset[str]:
    """A sentence's folded content words and every folded word form, for the relaxed overlap."""
    forms = frozenset(_fold(word) for clause in sentence.clauses for word in clause)
    return sentence.content | forms


def _speaker(items: Sequence[Said], candidate: ConflictCandidate) -> str:
    speakers = items[candidate.negated].speakers or items[candidate.affirmed].speakers
    return next(iter(speakers)) if len(speakers) == 1 else "the user"


def candidate_note(items: Sequence[Said], candidate: ConflictCandidate) -> str:
    """The note for a candidate pair the model check confirmed.

    Args:
        items: The delivered items the candidate came from.
        candidate: One `conflict_candidates` pair.

    Returns:
        The note text, unescaped.
    """
    return (
        f"Conflicting user statements: {_speaker(items, candidate)} said both "
        f'"{_quoted(candidate.negated_text)}" and "{_quoted(candidate.affirmed_text)}". '
        "These cannot both be true, and the record does not say which is correct."
    )


def possible_conflict_note(items: Sequence[Said], candidate: ConflictCandidate) -> str:
    """The note on a lexical nomination the model check did not confirm.

    It shows both statements and says they conflict only if both are about the
    same thing: shared words make a pair worth showing, never a contradiction.
    Showing the partner, not the denial alone, keeps both versions in front of
    the reader; the condition leaves the reader to decide whether they are the
    same thing.

    Args:
        items: The delivered items the candidate came from.
        candidate: The first `lexical_pairs` pair.

    Returns:
        The note text, unescaped.
    """
    return (
        f"Possibly conflicting user statements: {_speaker(items, candidate)} said "
        f'"{_quoted(candidate.negated_text)}" and also "{_quoted(candidate.affirmed_text)}". '
        "If both are about the same thing, they cannot both be true, and the record does not "
        "say which is correct."
    )


def lexical_pairs(items: Sequence[Said]) -> tuple[ConflictCandidate, ...]:
    """The pairs the lexical rule nominates, negated statements in rank order.

    Args:
        items: Delivered items in response order.

    Returns:
        For each negated user sentence of the first ``TOP`` items whose claim
        (`_claims`) other items' statements by the same speaker repeat
        (`_repeats`), that sentence paired with the ``PER_NEGATION``
        highest-ranked such items (within one item, the sentence repeating most
        of its nouns); at most ``CANDIDATES`` pairs.
    """
    sentences = [_sentences(item.text) if item.user else [] for item in items]
    same_speaker = _same_speaker(items)
    pairs: list[ConflictCandidate] = []
    for index in range(min(TOP, len(items))):
        for negated in sentences[index]:
            if not (claims := _claims(negated.text)):
                continue
            partners: list[tuple[int, str]] = []
            for other, candidates in enumerate(sentences):
                if len(partners) == PER_NEGATION:
                    break
                if other == index or not same_speaker(items[index], items[other]):
                    continue
                best: tuple[int, str] | None = None
                for affirmed in candidates:
                    found = [c for claim in claims if (c := _repeats(claim, affirmed)) is not None]
                    if found and (best is None or max(found) > best[0]):
                        best = (max(found), affirmed.text)
                if best is not None:
                    partners.append((other, best[1]))
            for other, text in partners:
                pairs.append(ConflictCandidate(index, negated.text, other, text))
                if len(pairs) == CANDIDATES:
                    return tuple(pairs)
    return tuple(pairs)
