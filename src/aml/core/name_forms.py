"""Closed inflected forms under which a name is located in quotes and sources (ML-1).

A Russian source writes a name in its grammatical case (``за Роуэном``, ``хомяку
Оскару``, ``в больнице Святой Марии``) and an Arabic source writes it with an attached
conjunction or preposition (``لمحمد``, ``ومحمد``), while a declaration and a naming
annotation write the name itself. Token containment (A0.6) then misses the name: B.7.2
masks the declaration (``name_unlocated``), B.7.3 the annotation (``naming_unquoted``),
and the facts about that person lose their entity. `name_located` also accepts, token
by token, the closed forms of `forms`:

- a Cyrillic token of four or more letters, with the singular case endings of its
  final letter: a consonant (``роуэн``: ``роуэна``, ``роуэну``, ``роуэном``, ``роуэне``),
  ``-а`` (``маша``: ``маши``, ``маше``, ``машу``, ``машей``), ``-я``, ``-ия``, ``-й`` or
  ``-ь``, and ``-ым`` after ``-ов``, ``-ев``, ``-ин`` or ``-ын`` (``ивановым``);
- an Arabic token of four or more letters after one or two of the proclitics ``و``,
  ``ف`` and then ``ب``, ``ل`` or ``ك`` (``لمحمد``, ``وبمحمد``), with ``لل`` for ``ل``
  before the article (``للحسن``);
- a Greek token of four or more letters: ``-α``, ``-η`` or ``-ω`` gains ``-ς``;
  ``-ας`` or ``-ης`` loses ``-ς`` (also with an accent on that vowel); ``-ος``
  becomes ``-ου`` or ``-ο``.
  Accents stay unchanged (``Μαρία``: ``Μαρίας``; ``Νίκος``: ``Νίκου``).

For a Latin name, only its final token can gain German genitive ``-s``, and only
before a noun in `_GERMAN_POSSESSIONS`, with at most one adjective in
`_GERMAN_MODIFIERS` (``Gretas kleiner Hund``). This context belongs to
`name_pattern`, not `forms`: a bare ``Gretas`` or English plural ``the Rowans``
cannot locate a singular name. Names ending in a sibilant stay literal; their
apostrophe already separates a token (``Felix' Schwester``).

`name_located` additionally accepts the first token of seven listed Russian names
in `_RUSSIAN_CONTEXT_FORMS`, including short names and fleeting-vowel stems
(``Лев``: ``со Львом``; ``Пётр``: ``с Петром``). An exact case-specific preposition
must immediately precede the capitalized source form, separated only by spaces or
tabs. This rule does not expand `forms` or `name_pattern`: bare ``Яна`` cannot locate
``Ян``, nor can the lowercase animal noun in ``о льве`` locate ``Лев``.

A form is a whole token, never a prefix or a similarity: ``машина`` is no form of
``маша``. Outside the contextual Russian exceptions, shorter tokens and unlisted
forms match only themselves, since a short stem's endings make other words
(``лев``: ``левом``; ``علي``: ``فعلي``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Final

from aml.core.statement import token_seq

MIN_INFLECTED_LETTERS: Final = 4
"""Letters a token needs before its closed forms count (module docstring)."""

_CYRILLIC = re.compile("[\u0400-\u04ff]+")
_ARABIC = re.compile("[\u0620-\u064a\u066e-\u06d3\u06d5\u06fa-\u06ff]+")
_ARABIC_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_GREEK = re.compile("[α-ωάέήίόύώϊϋΐΰ]+")
_LATIN = re.compile("[a-z\u00e0-\u00f6\u00f8-\u024f]+")

_CONSONANT_ENDINGS: Final = ("а", "у", "ом", "ем", "е")
_ENDINGS: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    # (final letters, letters the endings replace, endings), first match wins.
    ("ия", "я", ("и", "ю", "ей", "ею")),
    ("а", "а", ("ы", "и", "е", "у", "ой", "ей", "ою", "ею")),
    ("я", "я", ("и", "е", "ю", "ей", "ею")),
    ("ий", "й", ("я", "ю", "ем", "е", "и")),
    ("й", "й", ("я", "ю", "ем", "е")),
    ("ь", "ь", ("я", "ю", "ем", "е", "и", "ью")),
)
_SURNAMES: Final = ("ов", "ев", "ин", "ын")
_CONJUNCTIONS: Final = ("", "و", "ف")
_PREPOSITIONS: Final = ("", "ب", "ل", "ك")
_GREEK_ENDINGS: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    ("οσ", "οσ", ("ου", "ο")),
    ("ασ", "σ", ("",)),
    ("άσ", "σ", ("",)),
    ("ησ", "σ", ("",)),
    ("ήσ", "σ", ("",)),
    ("α", "", ("σ",)),
    ("ά", "", ("σ",)),
    ("η", "", ("σ",)),
    ("ή", "", ("σ",)),
    ("ω", "", ("σ",)),
    ("ώ", "", ("σ",)),
)
_RUSSIAN_CASE_PREPOSITIONS: Final = (
    ("от", "у", "для", "без"),
    ("к", "ко"),
    ("с", "со", "за", "перед", "над"),
    ("о", "об", "при"),
)
_RUSSIAN_CONTEXT_FORMS: Final = {
    # Genitive, dative, instrumental, prepositional; source spelling is capitalized.
    "лев": (("Льва",), ("Льву",), ("Львом",), ("Льве",)),
    "павел": (("Павла",), ("Павлу",), ("Павлом",), ()),
    "пётр": (("Петра",), ("Петру",), ("Петром",), ()),
    "ян": (("Яна",), ("Яну",), ("Яном",), ()),
    "ева": (("Евы",), ("Еве",), ("Евой", "Евою"), ("Еве",)),
    "зоя": (("Зои",), ("Зое",), ("Зоей", "Зоею"), ("Зое",)),
    "лия": (("Лии",), ("Лии",), ("Лией", "Лиею"), ("Лии",)),
}
# Restrict the ambiguous Latin -s to these adjacent German possession nouns.
# English words are deliberately absent (including the German noun Mutter).
_GERMAN_POSSESSIONS: Final = (
    "hund", "katze", "vater", "schwester", "bruder", "tochter", "sohn",
    "ehemann", "ehefrau", "freund", "freundin", "nachbar", "nachbarin", "kollege",
    "kollegin", "wohnung", "haus", "buch", "eltern", "grossmutter", "grossvater",
    "geburtstag", "arbeitsplatz", "arbeitgeber", "lieblingsfarbe", "adresse", "beruf",
    "fahrrad", "schlüssel",
)
_GERMAN_MODIFIERS: Final = tuple(
    stem + ending for stem in ("klein", "neu", "alt", "älter", "jünger", "best")
    for ending in ("e", "er", "es", "en", "em")
)


def _cyrillic_forms(token: str) -> set[str]:
    for final, replaced, endings in _ENDINGS:
        if token.endswith(final):
            stem = token[: len(token) - len(replaced)]
            return {stem + ending for ending in endings}
    if token[-1] in "аеёиоуыэюя":
        return set()
    found = {token + ending for ending in _CONSONANT_ENDINGS}
    if token.endswith(_SURNAMES):
        found.add(token + "ым")
    return found


def _arabic_forms(token: str) -> set[str]:
    found = {
        conjunction + preposition + token
        for conjunction in _CONJUNCTIONS for preposition in _PREPOSITIONS
        if conjunction or preposition
    }
    if token.startswith("ال"):
        found.update(conjunction + "لل" + token[2:] for conjunction in _CONJUNCTIONS)
    return found


def _greek_forms(token: str) -> set[str]:
    for final, replaced, endings in _GREEK_ENDINGS:
        if token.endswith(final):
            stem = token[: len(token) - len(replaced)]
            return {stem + ending for ending in endings}
    return set()


@lru_cache(maxsize=4096)
def forms(token: str) -> frozenset[str]:
    """Return the forms under which a `token_seq` token of a name is located.

    Args:
        token (str): A token of a name, folded as `token_seq` folds it.

    Returns:
        frozenset[str]: The token itself and, for a Cyrillic, Arabic or Greek token of
            `MIN_INFLECTED_LETTERS` or more letters, its closed forms (module docstring).
    """
    found = {token}
    if _CYRILLIC.fullmatch(token) and len(token) >= MIN_INFLECTED_LETTERS:
        found |= _cyrillic_forms(token)
    elif _ARABIC.fullmatch(_ARABIC_MARKS.sub("", token)) and (
        len(_ARABIC_MARKS.sub("", token)) >= MIN_INFLECTED_LETTERS
    ):
        found |= _arabic_forms(token)
    elif _GREEK.fullmatch(token) and len(token) >= MIN_INFLECTED_LETTERS:
        found |= _greek_forms(token)
    return frozenset(found)


def name_pattern(name: str) -> re.Pattern[str] | None:
    """Return the pattern of a name's forms in ``"\\0"``-joined tokens, or None without tokens.

    Args:
        name (str): A name exactly as declared or annotated.

    Returns:
        re.Pattern[str] | None: A pattern matching the name's tokens, each in one of
            its `forms`, contiguously and whole in ``"\\0" + "\\0".join(tokens) + "\\0"``.
            The last token also accepts the contextual German genitive (module docstring).
            Its following noun is a lookahead and is not part of the name.
    """
    tokens = token_seq(name)
    if not tokens:
        return None
    token_patterns = [
        "(?:" + "|".join(re.escape(form) for form in sorted(forms(token), key=len, reverse=True))
        + ")" for token in tokens
    ]
    last = tokens[-1]
    if (_LATIN.fullmatch(last) and len(last) >= MIN_INFLECTED_LETTERS
            and not last.endswith(("s", "ß", "x", "z", "c"))):
        context = ("\0(?:(?:" + "|".join(_GERMAN_MODIFIERS) + ")\0)?(?:"
                   + "|".join(_GERMAN_POSSESSIONS) + ")\0")
        genitive = re.escape(last + "s") + "(?=" + context + ")"
        token_patterns[-1] = "(?:" + token_patterns[-1] + "|" + genitive + ")"
    return re.compile("\0" + "\0".join(token_patterns) + "\0")


@lru_cache(maxsize=4096)
def _russian_context_patterns(first: str) -> tuple[re.Pattern[str], ...]:
    paradigm = _RUSSIAN_CONTEXT_FORMS.get(first)
    if paradigm is None:
        return ()
    return tuple(
        re.compile(
            r"(?<!\w)((?i:" + "|".join(prepositions) + r")[ \t]+)(?:"
            + "|".join(case_forms) + r")(?!\w)",
        ) for prepositions, case_forms in zip(_RUSSIAN_CASE_PREPOSITIONS, paradigm, strict=True)
        if case_forms
    )


def name_located(name: str, texts: Sequence[str]) -> bool:
    """Tell whether a name occurs in some text, token by token in one of its `forms`.

    Token containment (A0.6, `aml.core.statement.token_contained`) with each token of
    the name also matched by its closed inflected forms. The source-capitalized
    Russian exceptions additionally need an exact case-specific preposition.

    Args:
        name (str): A name exactly as declared or annotated.
        texts (Sequence[str]): Unfolded texts to search, such as quotes.

    Returns:
        bool: True when the name has tokens and some text holds them in order.
    """
    pattern = name_pattern(name)
    if pattern is None:
        return False
    first = token_seq(name)[0]
    contextual_patterns = _russian_context_patterns(first)
    for text in texts:
        if pattern.search("\0" + "\0".join(token_seq(text)) + "\0") is not None:
            return True
        for contextual in contextual_patterns:
            restored = contextual.sub(lambda match: match[1] + first, text)
            if restored != text and pattern.search(
                "\0" + "\0".join(token_seq(restored)) + "\0",
            ) is not None:
                return True
    return False
