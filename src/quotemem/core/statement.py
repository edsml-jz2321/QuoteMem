"""Statement, token, name and key helpers, and the shared bounds.

This module is the single owner of the value rules that extraction, admission,
the judge, the Writer and every read path apply identically: the canonical
statement and its digest, lexical tokens and token containment, name
normalization, the participant and mint keys, and the bounds that the
extract-v12 and judge-v8 wire schemas use. Every helper is a pure function of
its arguments.

Tokens follow one rule, extended for scripts beyond English and CJK:
``token_seq`` is the word runs of ``NFKC(text).casefold()``, where

- every Han, Kana and Hangul character is a token of its own;
- every letter or digit of a script written without spaces between words and
  without ideographs (Thai, Lao, Myanmar, Khmer and the Tai scripts) is, with
  the combining marks that follow it, a token of its own: one grapheme cluster;
- every other run of word characters is one token, and a combining mark (Unicode
  category M) continues the run it follows instead of ending it, so a Hindi,
  Bengali, Tamil, vocalized Arabic or Hebrew word stays one token;
- a casefolded dotted capital I (``i`` followed by U+0307) folds to ``i``, so
  ``İstanbul``, ``ISTANBUL`` and ``Istanbul`` share their tokens.

Latin, Cyrillic, Greek, digit and CJK text without combining marks tokenizes
exactly as ``re.findall(r"\\w+", ...)`` with the CJK rule. Without the second
and third rules a name, quote or time expression written inside unspaced Thai
text could never be token-contained, and a Devanagari word would split at
every vowel sign, so that different names such as राम and रोम shared one
``name_norm``.
"""

from __future__ import annotations

import functools
import hashlib
import re
import unicodedata
from collections.abc import Container, Iterable, Iterator, Mapping, Sequence
from typing import Annotated, Final, Protocol

from pydantic import StringConstraints

from quotemem.core.identity import canonical_json

# Statement bounds. The extract-v12 and judge-v8 wire schemas, admission, the Writer
# and the read paths use these values; each name is the inclusive maximum.
MAX_STATEMENT_CHARS: Final = 1024
MAX_KEYWORDS: Final = 32
MAX_KEYWORD_CHARS: Final = 128
MAX_ANCHORS: Final = 8
MAX_ABOUT_ENTITIES: Final = 7
MAX_NAMING: Final = 4
MAX_NAME_CHARS: Final = 128
# 24 rather than the 8: under the token rule every CJK character
# is a token, so the bound must admit CJK names such as organisation names. Like
# LABEL_MAX_TOKENS it bounds a name's `bound_length`.
MAX_NAME_TOKENS: Final = 24
MAX_NAME_NORM_CHARS: Final = 256
LABEL_MAX_TOKENS: Final = 8
"""The `bound_length` of a speaker label's name; it bounds names written without spaces."""
SEA_UNITS_PER_TOKEN: Final = 3
"""Letter-plus-marks units of an unspaced Southeast Asian script that the name and label
bounds count as one token.

`token_seq` makes each such unit a token, as it makes each Han character one, but a
Thai, Lao, Myanmar or Khmer syllable spans two to three units where a Han character
is one syllable. Counted one by one, a Thai full name such as ``ประยุทธ์ จันทร์โอชา``
(14 units) exceeded `LABEL_MAX_TOKENS`; counted by threes it is 5, as its two words
would be in English."""
MAX_CORRECTS_TEXT_CHARS: Final = 512
MAX_REVISES: Final = 8
MAX_SUPPORT_GROUPS: Final = 8
MAX_GROUP_QUOTES: Final = 16
MAX_GROUP_SPANS: Final = 32
MAX_MEMORY_SPANS: Final = 256
MAX_QUOTE_CHARS: Final = 4096
# Wire bounds that the statement bounds do not list: a fact's time expression and its
# sensitivity findings.
MAX_TIME_TEXT_CHARS: Final = 1024
MAX_SENSITIVITY_FINDINGS: Final = 32
MAX_REPLY_FACTS: Final = 512
MAX_UNIT_FACTS: Final = 128
MAX_UNIT_NO: Final = 65_535  # an owner's units, so a position_key's unit_no
MAX_NEW_ENTITIES: Final = 32
MAX_UNRESOLVED_REFERENTS: Final = 32
MAX_DESCRIPTOR_CHARS: Final = 256
MAX_MENTION_CHARS: Final = 256
MAX_ENTITY_TYPE_CHARS: Final = 64
MAX_REFERENT_CANDIDATES: Final = 8
MAX_CATALOG_ENTITIES: Final = 64
MAX_CATALOG_NAMES: Final = 8  # readable names per extraction catalog entry
MAX_RELEVANCE_MEMORIES: Final = 30
MAX_RESERVED_MEMORIES: Final = 10
MAX_CONTEXT_MEMORIES: Final = 40
MAX_EARLIER_PER_MEMORY: Final = 2
MAX_EARLIER_VERSIONS: Final = 80
MAX_SOURCES: Final = 600
MAX_FRESH_SOURCES: Final = 256
MAX_TAIL_SOURCES: Final = 6
MAX_JUDGE_TARGETS: Final = 64
MAX_JUDGE_HISTORY: Final = 80
MAX_JUDGE_TOMBSTONES: Final = 8
MAX_ENTITY_CANDIDATES: Final = 6
MAX_WITHHELD_SPANS: Final = 256
MAX_TOMBSTONE_SOURCES: Final = 256  # Evidence a tombstone cites
MAX_ENTITY_BINDINGS: Final = 128
MAX_REENTRY: Final = 64
MAX_ARTIFACT_BODY_CHARS: Final = 262_144
MAX_MESSAGE_BYTES: Final = 1_048_576

ERASED_DIGEST: Final = "0" * 64
"""The statement and embedding-input digest of an erased head."""

LABEL_SPEAKER_NAMESPACE: Final = "aml-label-speaker-v1"
"""The participant namespace of a speaker proven only by a line label."""

HexDigest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
"""A lowercase hex SHA-256: statement digests, participant keys and mint keys."""

_CJK = (
    "ᄀ-ᇿ"  # Hangul Jamo
    "々〇〡-〩〸-〻"  # Han iteration marks and numerals
    "぀-ヿ"  # Hiragana, Katakana
    "㄰-㆏"  # Hangul Compatibility Jamo
    "ㇰ-ㇿ"  # Katakana Phonetic Extensions
    "㐀-䶿"  # CJK Unified Ideographs Extension A
    "一-鿿"  # CJK Unified Ideographs
    "ꥠ-꥿"  # Hangul Jamo Extended-A
    "가-퟿"  # Hangul Syllables, Hangul Jamo Extended-B
    "豈-﫿"  # CJK Compatibility Ideographs
    "\U00016fe3"  # Old Chinese iteration mark
    "\U0001aff0-\U0001b16f"  # Kana Extended-B, Kana Supplement, Kana Extended-A, Small Kana
    "\U00020000-\U0003ffff"  # Supplementary and Tertiary Ideographic Planes
)
UNSPACED_SCRIPTS: Final = (
    "฀-๿"  # Thai
    "຀-໿"  # Lao
    "က-႟"  # Myanmar
    "ក-៿"  # Khmer
    "ᥐ-᥿"  # Tai Le
    "ᦀ-᧟"  # New Tai Lue
    "᧠-᧿"  # Khmer Symbols
    "ᨠ-᪯"  # Tai Tham
    "ꧠ-꧿"  # Myanmar Extended-B
    "ꩠ-ꩿ"  # Myanmar Extended-A
    "ꪀ-꫟"  # Tai Viet
    "\U00011700-\U0001174f"  # Ahom
)
"""Regex class body of the scripts written without spaces between words that have no
ideographs (the Line_Break SA scripts): Thai, Lao, Myanmar, Khmer and the Tai scripts."""
_SEA = UNSPACED_SCRIPTS


def _mark_class() -> str:
    """Return every combining mark outside the CJK ranges as a regex class body.

    A mark is a character of Unicode category Mn, Mc or Me in the running
    Python's Unicode database, the same database that defines ``\\w``. Marks
    live only in planes 0, 1 and 14. Marks inside the CJK ranges (the kana
    voicing marks) keep the rule and stay tokens of their own.
    """
    cjk = re.compile(f"[{_CJK}]")
    points = [
        point for point in (*range(0x20000), *range(0xE0000, 0xE0200))
        if unicodedata.category(chr(point))[0] == "M" and cjk.match(chr(point)) is None
    ]
    ranges: list[list[int]] = []
    for point in points:
        if ranges and ranges[-1][1] == point - 1:
            ranges[-1][1] = point
        else:
            ranges.append([point, point])
    return "".join(
        re.escape(chr(low)) if low == high else f"{re.escape(chr(low))}-{re.escape(chr(high))}"
        for low, high in ranges
    )


COMBINING_MARKS: Final = _mark_class()
"""Regex class body of every combining mark outside the CJK ranges."""
_MARKS = COMBINING_MARKS
_WORD = f"[^\\W{_CJK}{_SEA}]"
# No mark is ASCII: the lookahead skips the large mark class after an ASCII character,
# which keeps English text within a fifth of the tokenizing time.
_TOKEN = re.compile(
    f"[{_CJK}]"
    f"|[{_SEA}](?<=\\w)[{_MARKS}]*"
    f"|{_WORD}+(?:(?=[^\\x00-\\x7f])[{_MARKS}]+{_WORD}*)*"
)
_OWN = re.compile(f"[{_CJK}{_SEA}]")
_OWN_SEA = re.compile(f"[{_SEA}]")
_DIGEST_EDGES = " .,;:!?…、。！，：；？"
_BIDI_CONTROLS = frozenset(map(chr, (*range(0x202A, 0x202F), *range(0x2066, 0x206A))))
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cs", "Co"})


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_statement(text: str) -> str:
    """Return the canonical form of a statement, refusing an invalid one.

    The canonical form is ``" ".join(NFC(text).split())``: NFC, with every
    whitespace run collapsed to one space and none at either end, so a renderer
    can print it as one raw line.

    Args:
        text (str): A statement as written by the extractor or stored in a
            snapshot.

    Returns:
        str: The canonical statement.

    Raises:
        ValueError: If the canonical form is outside 1..1024 characters, has no
            letter or digit, or contains a Cc, Cs or Co character or a bidi
            control (U+202A..U+202E, U+2066..U+2069).
    """
    canonical = " ".join(unicodedata.normalize("NFC", text).split())
    if not 1 <= len(canonical) <= MAX_STATEMENT_CHARS:
        raise ValueError(
            f"A statement has 1..{MAX_STATEMENT_CHARS} characters, not {len(canonical)}"
        )
    if not any(char.isalnum() for char in canonical):
        raise ValueError("A statement needs at least one letter or digit")
    if any(
        char in _BIDI_CONTROLS or unicodedata.category(char) in _FORBIDDEN_CATEGORIES
        for char in canonical
    ):
        raise ValueError(
            "A statement cannot contain a control, surrogate, private-use or bidi character"
        )
    return canonical


def fit_statement(text: str) -> str:
    """Cut a statement longer than the bound at its last clause within it.

    The canonical whitespace form of a statement over ``MAX_STATEMENT_CHARS`` ends at
    the last ``"; "`` or ``". "`` in its second half within the bound, closed with a
    period, else at its last word with an ellipsis. A statement within the bound comes
    back unchanged; every other check of ``canonical_statement`` still applies.

    Args:
        text (str): A statement as written by the extractor.

    Returns:
        str: The statement, or its fitted canonical form when it is too long.
    """
    canonical = " ".join(unicodedata.normalize("NFC", text).split())
    if len(canonical) <= MAX_STATEMENT_CHARS:
        return text
    head = canonical[:MAX_STATEMENT_CHARS]
    clause = max(head.rfind("; "), head.rfind(". "))
    if clause >= MAX_STATEMENT_CHARS // 2:
        return f"{head[:clause].rstrip(' ,:;.')}."
    return f"{head[:MAX_STATEMENT_CHARS - 1].rsplit(' ', 1)[0].rstrip(' ,:;.')}…"


def statement_digest(text: str) -> str:
    """Return the unkeyed digest that identifies a statement's wording.

    The digest is ``sha256(canonical_json(["aml-statement-v1", norm(text)]))``
    in lowercase hex, where ``norm`` applies NFKC, casefolds, collapses every
    whitespace run to one space and strips the edge characters
    `` .,;:!?…、。！，：；？``. Case, width, spacing and closing punctuation
    therefore never change the digest. It is internal only and never enters a
    public payload.

    Args:
        text (str): A statement; a canonical statement and its raw source give
            the same digest.

    Returns:
        str: 64 lowercase hex characters.
    """
    norm = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return _sha256(["aml-statement-v1", norm.strip(_DIGEST_EDGES)])


def folded(text: str) -> str:
    """Return the folded text that `token_seq` tokenizes.

    The fold is ``NFKC(text).casefold()`` with every ``i`` + U+0307 (the
    casefold of the dotted capital I) folded to ``i``.

    Args:
        text (str): Any text.

    Returns:
        str: The folded text.
    """
    return unicodedata.normalize("NFKC", text).casefold().replace("i̇", "i")


def token_seq(text: str) -> list[str]:
    """Return the lexical tokens of a text, the one tokenization every natural-language rule uses.

    Tokens are the word runs of `folded` text. Every Han, Kana and Hangul
    character is a token of its own. Every letter or digit of an
    unspaced Southeast Asian script (Thai, Lao, Myanmar, Khmer, the Tai scripts)
    is a token of its own together with the combining marks after it, one
    grapheme cluster. Every other run of word characters (Latin, digits,
    underscore and other scripts) is one token, as ``re.findall(r"\\w+", ...)``
    finds it, except that a combining mark continues the run it follows,
    so ``हिन्दी`` is one token. Punctuation, spaces and symbols separate
    tokens and are never part of one.

    Args:
        text (str): Any statement, quote, name, keyword, label or reason.

    Returns:
        list[str]: The tokens in text order; empty when the text has no word
            character.

    Examples:
        ```python
        token_seq("My sister 小雅 lives in Lisbon")
        # ['my', 'sister', '小', '雅', 'lives', 'in', 'lisbon']
        token_seq("मेरी बहन प्रिया")  # ['मेरी', 'बहन', 'प्रिया']
        token_seq("ชื่อสมศักดิ์")  # ['ชื่', 'อ', 'ส', 'ม', 'ศั', 'ก', 'ดิ์']
        token_seq("İstanbul")  # ['istanbul']
        ```
    """
    return _TOKEN.findall(folded(text))


def token_spans(text: str) -> list[tuple[str, int, int]]:
    """Return each `token_seq` token with its character span in the `folded` text.

    Args:
        text (str): Any text.

    Returns:
        list[tuple[str, int, int]]: ``(token, start, end)`` in token order; the
            tokens are exactly ``token_seq(text)``.
    """
    return [(match.group(), *match.span()) for match in _TOKEN.finditer(folded(text))]


def own_token(token: str) -> bool:
    """Tell whether a token is a token of its own.

    A Han, Kana or Hangul character, or a grapheme cluster of an unspaced
    Southeast Asian script, never joins the token next to it; every other token
    is a whole word run.

    Args:
        token (str): A `token_seq` token.

    Returns:
        bool: True for a CJK character or an unspaced-script cluster.
    """
    return _OWN.match(token) is not None


def memory_tokens(statement: str, keywords: Sequence[str]) -> list[str]:
    """Return the unigram sequence of a memory: its statement, then its keywords.

    This identity-level sequence does not include CJK bigrams. BM25 callers
    use ``recall_corpora.sparse_terms`` on the same statement/keyword composition.

    Args:
        statement (str): The memory's statement.
        keywords (Sequence[str]): The memory's keywords, in stored order.

    Returns:
        list[str]: ``token_seq`` of the statement followed by ``token_seq`` of
            each keyword.
    """
    return token_seq(" ".join((statement, *keywords)))


def name_norm(text: str) -> str:
    """Return the normalized form under which names are stored and compared.

    Args:
        text (str): A name, label or mention.

    Returns:
        str: The tokens of ``token_seq(text)`` joined by single spaces; for a
            Chinese name every character is a token, so ``王小明`` becomes
            ``王 小 明``.
    """
    return " ".join(token_seq(text))


def is_name(text: str) -> bool:
    """Tell whether a text is a valid entity name.

    A name has 1..128 characters as written and a `bound_length` of 1..24. Its
    ``name_norm`` must also fit the 256 characters of ``entity_name.name_norm``,
    which only a name that NFKC expands far beyond its written length
    can exceed. Because every Han, Kana or Hangul character is a token, a
    name written in those scripts has at most 24 characters; a Thai name has at
    most 72 letter-plus-marks units (`SEA_UNITS_PER_TOKEN`).

    Args:
        text (str): A candidate name exactly as written.

    Returns:
        bool: True when the name is valid.
    """
    if not 1 <= len(text) <= MAX_NAME_CHARS:
        return False
    tokens = token_seq(text)
    return (
        1 <= bound_length(tokens) <= MAX_NAME_TOKENS
        and len(" ".join(tokens)) <= MAX_NAME_NORM_CHARS
    )


def bound_length(tokens: Sequence[str]) -> int:
    """Return the length of a token run under the name and label bounds.

    Every token counts one, except the letter-plus-marks units of an unspaced
    Southeast Asian script, which count one per `SEA_UNITS_PER_TOKEN`, rounded up:
    ``สมชาย ชินวัตร`` has 10 such tokens and a length of 4. Text without those
    scripts has a length of its token count.

    Args:
        tokens (Sequence[str]): `token_seq` tokens.

    Returns:
        int: The length that `MAX_NAME_TOKENS` and `LABEL_MAX_TOKENS` bound.
    """
    units = sum(1 for token in tokens if _OWN_SEA.match(token) is not None)
    return len(tokens) - units + -(-units // SEA_UNITS_PER_TOKEN)


def token_contained(needle: str, haystack: str) -> bool:
    """Tell whether one text is token-contained in another.

    ``needle`` is contained in ``haystack`` when ``token_seq(needle)`` is a
    non-empty contiguous subsequence of ``token_seq(haystack)``. A partial
    token never matches, so ``Bai`` is not contained in ``Juno``; a Chinese
    name is contained in Chinese text written around it, because each Han
    character is a token. A needle without tokens is never contained, so a
    token-less name or time expression stays unlocated.

    Args:
        needle (str): The text to find, such as a name or a time expression.
        haystack (str): The text to search, such as a statement or a source.

    Returns:
        bool: True when ``needle`` has tokens and they occur contiguously in
            ``haystack``.
    """
    small = token_seq(needle)
    if not small:
        return False
    # No token contains NUL, so delimiting every token with it turns a
    # contiguous token match into a plain substring match.
    return "\0" + "\0".join(small) + "\0" in "\0" + "\0".join(token_seq(haystack)) + "\0"


def canonical_entity_type(text: str) -> str:
    """Return the stored form of a free-text entity type.

    Args:
        text (str): A type as declared by the extractor or the Entity SDK.

    Returns:
        str: The type after NFC and trimming of edge whitespace.

    Raises:
        ValueError: If the stored form is outside 1..64 characters.
    """
    stored = unicodedata.normalize("NFC", text).strip()
    if not 1 <= len(stored) <= MAX_ENTITY_TYPE_CHARS:
        raise ValueError(
            f"An entity type has 1..{MAX_ENTITY_TYPE_CHARS} characters after NFC and trimming"
        )
    return stored


@functools.lru_cache(maxsize=1 << 16)
def participant_key(namespace: str, conversation_id: str, participant_id: str) -> str:
    """Return the key that identifies one participant within a scope.

    Labelled speakers are found by keying every short token run of a request's
    text (`labelled_mentions`), and an Add sizes and builds its context several
    times, so the same keys are derived again and again: each is kept.

    Args:
        namespace (str): The source adapter's participant namespace, or
            `LABEL_SPEAKER_NAMESPACE` for a labelled speaker.
        conversation_id (str): The source conversation, or the scope id for a
            labelled speaker.
        participant_id (str): The source participant, or the `name_norm` of the
            label for a labelled speaker.

    Returns:
        str: ``sha256(canonical_json(["aml-participant-v1", namespace,
            conversation_id, participant_id]))`` in lowercase hex.
    """
    return _sha256(["aml-participant-v1", namespace, conversation_id, participant_id])


def label_speaker_key(scope_id: str, label: str) -> str:
    """Return the participant key of a speaker proven by a line label.

    Within one scope a label names one speaker, so labels with the same
    `name_norm` share one participant.

    Args:
        scope_id (str): The scope the labelled message belongs to.
        label (str): The speaker label as written, without its colon.

    Returns:
        str: `participant_key` over `LABEL_SPEAKER_NAMESPACE`, the scope id and
            ``name_norm(label)``.

    Raises:
        ValueError: If the label has no tokens.
    """
    participant = name_norm(label)
    if not participant:
        raise ValueError("A speaker label needs at least one token")
    return participant_key(LABEL_SPEAKER_NAMESPACE, scope_id, participant)


def labelled_mentions(
    scope_id: str, text: str, keys: Container[str], *, max_tokens: int,
) -> list[tuple[str, str]]:
    """Return the labelled speakers a text names, each with its mention.

    A labelled speaker is keyed by its scope and its label's `name_norm`
    (`label_speaker_key`), so a text names that speaker exactly where the label
    is token-contained in it. Every run of tokens with a `bound_length` of 1 to
    ``max_tokens`` is keyed and looked up in ``keys``.

    Args:
        scope_id (str): The scope the speakers were labelled in.
        text (str): The text, such as one readable stretch of a source.
        keys (Container[str]): The participant keys to look for.
        max_tokens (int): The longest `bound_length` of a label name.

    Returns:
        list[tuple[str, str]]: ``(participant_key, mention)`` pairs in text
            order. A mention is written as in ``NFKC(text)``, or as its
            `name_norm` where casefolding would tokenize the text differently.
    """
    return [
        (key, mention) for name, mention in _token_runs(text, max_tokens)
        if (key := participant_key(LABEL_SPEAKER_NAMESPACE, scope_id, name)) in keys
    ]


class SpokenPart(Protocol):
    """The headers of an Evidence part that say which source participant spoke it."""

    @property
    def speaker_entity(self) -> str | None: ...

    @property
    def source_participant(self) -> SourceTuple | None: ...


class SourceTuple(Protocol):
    """A source participant tuple."""

    @property
    def namespace(self) -> str: ...

    @property
    def conversation_id(self) -> str: ...

    @property
    def participant_id(self) -> str: ...


def speaker_index(parts: Iterable[SpokenPart]) -> dict[str, tuple[str, ...]]:
    """Index the metadata speaker names that source participants' Evidence carries.

    A source participant is keyed by its tuple, not by a name, so the
    names it was shown under are only the metadata ``speaker`` of its Evidence.
    The caller passes the parts that pass ``p_src``, so a name shows only while its
    Evidence does.

    Args:
        parts (Iterable[SpokenPart]): Evidence parts of one scope.

    Returns:
        dict[str, tuple[str, ...]]: Each speaker name's `name_norm` with the
            participant keys whose Evidence carries it, sorted.
    """
    found: dict[str, set[str]] = {}
    for part in parts:
        source, speaker = part.source_participant, part.speaker_entity
        if source is None or speaker is None or not (norm := name_norm(speaker)):
            continue
        found.setdefault(norm, set()).add(participant_key(
            source.namespace, source.conversation_id, source.participant_id,
        ))
    return {norm: tuple(sorted(keys)) for norm, keys in found.items()}


def named_participants(
    scope_id: str, text: str, keys: Container[str], speakers: Mapping[str, Sequence[str]],
) -> list[tuple[str, str]]:
    """Return the participants a text names, each with its mention (route 4).

    A participant has no entity name. A text names a labelled speaker
    where its label is token-contained in it (`labelled_mentions`) and a source
    participant where a speaker name of its readable Evidence is
    (`spoken_mentions`). The catalog, nomination's entity candidates and
    Search's participant lookup read participants by this one rule.

    Args:
        scope_id (str): The scope the speakers were labelled in.
        text (str): The text, such as one readable stretch of a source.
        keys (Container[str]): The participant keys to look for.
        speakers (Mapping[str, Sequence[str]]): `speaker_index` of the scope.

    Returns:
        list[tuple[str, str]]: ``(participant_key, mention)`` pairs, labelled
            speakers first, each in text order.
    """
    return [
        *labelled_mentions(scope_id, text, keys, max_tokens=LABEL_MAX_TOKENS),
        *spoken_mentions(text, speakers, keys),
    ]


def spoken_mentions(
    text: str, speakers: Mapping[str, Sequence[str]], keys: Container[str],
) -> list[tuple[str, str]]:
    """Return the source participants a text names by a speaker name of theirs.

    The text names a participant exactly where one of its speaker names
    (`speaker_index`) is token-contained in it, as a label names a labelled
    speaker (`labelled_mentions`).

    Args:
        text (str): The text, such as one readable stretch of a source.
        speakers (Mapping[str, Sequence[str]]): `speaker_index` of the scope.
        keys (Container[str]): The participant keys to look for.

    Returns:
        list[tuple[str, str]]: ``(participant_key, mention)`` pairs in text
            order, the mention written as in `labelled_mentions`.
    """
    longest = max((bound_length(norm.split(" ")) for norm in speakers), default=0)
    return [
        (key, mention) for name, mention in _token_runs(text, longest)
        for key in speakers.get(name, ()) if key in keys
    ]


def _token_runs(text: str, max_tokens: int) -> Iterator[tuple[str, str]]:
    """Yield every token run of a text whose `bound_length` is 1 to ``max_tokens``.

    Each run comes as its `name_norm` and its mention: the run as written in
    ``NFKC(text)``, or its `name_norm` where folding would tokenize the text
    differently.
    """
    normal = unicodedata.normalize("NFKC", text)
    tokens = token_seq(normal)
    runs = [match.span() for match in _TOKEN.finditer(normal)]
    written = [normal[start:end].casefold() for start, end in runs] == tokens
    unspaced = [_OWN_SEA.match(token) is not None for token in tokens]
    for start in range(len(tokens)):
        units = others = 0
        for end in range(start + 1, len(tokens) + 1):
            if unspaced[end - 1]:
                units += 1
            else:
                others += 1
            if others + -(-units // SEA_UNITS_PER_TOKEN) > max_tokens:
                break
            name = " ".join(tokens[start:end])
            yield name, normal[runs[start][0]:runs[end - 1][1]] if written else name


def mint_key(owner_run_id: str, unit_no: int, local_key: str) -> str:
    """Return the content-free key of an entity declared by one extraction unit.

    Args:
        owner_run_id (str): The Add owner run that extracted the declaration.
        unit_no (int): The unit within that owner, 0..65535.
        local_key (str): The declaration's handle, ``N1``..``N32``.

    Returns:
        str: ``sha256(canonical_json(["aml-mint-v1", owner_run_id, unit_no,
            local_key]))`` in lowercase hex.
    """
    return _sha256(["aml-mint-v1", owner_run_id, unit_no, local_key])
