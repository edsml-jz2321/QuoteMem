"""Descriptor rewording: a corrected value reworded in the heads it outdated (C.9.4a).

A correction keeps the value it replaces (B.13 #8: "Mochi, Avery's pet, is a hamster,
not a guinea pig."). When a stored head's own fact is something else and the corrected
value only describes one of its entities, the judge finds that the correction makes the
head's description untrue and judges the pair ``contradicts`` (C.15 rule 6, CT-1): the
name in "Avery's guinea pig is named Mochi.", the ownership in "Avery has a guinea
pig." and the activity in "Avery's guinea pig Mochi has been great." stay true, so
retiring the head would lose them, and before the rewording each such head stayed current
with the obsolete descriptor beside the correction (O-D: 6 of 6 live naming heads).

The head's fact is unchanged, so the new version rewords it rather than corrects it
(C.9.4a): only the identifying description changes, and it changes to wording that
states neither the replaced nor the new value, so the head's own quotes still support it
and it never depends on the correction (OPR-2). This module is the pure text step.
`descriptor_change` reads the replaced and the new value from the correction's own
statement, only in a closed repairing form (B.13 #8): one copular assertion that ends
"NEW, not OLD" (or "rather than" / "instead of"), "is named NEW, not OLD" for a name, or
the possessive "has a NEW, not a OLD" (``has``, ``had`` or ``owns``; OPR-3). When the
extraction writes a correction as a plain kind assertion without the replaced value
("Avery's pet Mochi is a hamster."), `kind_assertion` reads its new kind and
`named_kinds` the kind words by which the heads the judge related ``contradicts`` name
the same pet ("Avery's guinea pig is named Mochi."); `kind_change` takes the replaced
kind from them only when they agree on exactly one value under the same value rules
(OPR-4 para-pet).
`repaired_statement` rewords the head's one whole-word occurrence of the replaced value:
a kind takes the entity's generic descriptor (`generic_descriptor`: "Avery's pet is
named Mochi."), and a name is dropped after the possessive descriptor that precedes it
("The user's cat sleeps on the windowsill all day.") or else becomes "the" and that
descriptor ("The cat's bowl is blue."). Everything else is no rewording: a report or a
quotation, a value with a number, a negation, a conjunction or a preposition (a phrase
like "on Monday" is no parallel of "Tuesday"), a value that occurs twice or not at all,
a kind without a usable descriptor, and any edit that would touch a name the head gives
(a naming head keeps every name byte for byte, UD2).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final

from aml.core.statement import (
    MAX_KEYWORD_CHARS,
    canonical_statement,
    is_name,
    name_norm,
    token_contained,
    token_seq,
)

MAX_VALUE_TOKENS: Final = 6
"""The most tokens a replaced or new value may have: a descriptor phrase, never a clause."""

_COPULAR: Final = re.compile(
    r"^(?P<subject>.+?)\s+(?:is|are|was|were)\s+(?:(?P<naming>named|called)\s+)?"
    r"(?P<new>[^,;:]+?),?\s+(?:not|rather than|instead of)\s+(?P<old>[^,;:]+?)\s*[.!]?$",
    re.IGNORECASE,
)
"""B.13 #8's repairing form: ``<subject> is [named] NEW, not OLD``, ending the statement."""
_POSSESSIVE: Final = re.compile(
    r"^(?P<subject>.+?)\s+(?:has|had|owns)\s+an?\s+(?P<new>[^,;:]+?),?\s+"
    r"(?:not|rather than|instead of)\s+an?\s+(?P<old>[^,;:]+?)\s*[.!]?$",
    re.IGNORECASE,
)
"""The possessive repairing form ``<owner> has|had|owns a NEW, not a OLD`` (OPR-3): a kind
change of the thing the owner has, never a name."""
_KIND_ASSERTION: Final = re.compile(
    r"^(?P<subject>.+?)\s+(?:is|are|was|were)\s+an?\s+(?P<new>[^,;:]+?)\s*[.!]?$",
    re.IGNORECASE,
)
"""A plain kind assertion ``<subject> is a|an NEW`` ending the statement: the form of a
correction whose extraction left out the replaced value ("Turns out Mochi is a hamster")."""
_NEGATION: Final = re.compile(
    r"(?:\b(?:not|no|never|nor|rather than|instead of|no longer)\b|n['’]t\b)", re.IGNORECASE)
"""A negated or contrasted assertion is no plain kind assertion (``descriptor_change``)."""
_KIND_WORDS: Final = r"(?P<kind>[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})"
_POSSESSOR: Final = r"(?:['’]s|\b(?:his|her|their|its|my|your|our))"
_NAMED_KIND_FORMS: Final = (
    _POSSESSOR + r"\s+" + _KIND_WORDS + r"\s+{name}(?![\w-])",
    _POSSESSOR + r"\s+" + _KIND_WORDS + r"\s+(?:is|was)\s+(?:named|called)\s+{name}(?![\w-])",
    r"\b(?:has|had|owns|owned|adopted|got)\s+an?\s+" + _KIND_WORDS
    + r"\s+(?:named|called)\s+{name}(?![\w-])",
    r"(?<![\w-]){name}\s+(?:is|was)\s+(?:an?\s+|\S+['’]s\s+)" + _KIND_WORDS + r"\s*[.!]?$",
    r"(?<![\w-]){name},\s+(?:an?\s+|\S+['’]s\s+)" + _KIND_WORDS + r",",
)
"""How a head names a pet with a kind word: right before the name after a possessive
("Avery's guinea pig Mochi"), as the possessed noun its name is given to ("Avery's
guinea pig is named Mochi.", "Avery has a guinea pig named Mochi."), or as what the
named pet is ("Mochi is Avery's guinea pig.", "Mochi, Avery's guinea pig, ...")."""

_REPORT: Final = re.compile(
    r"\b(?:says?|said|tells?|told|reports?|reported|relays?|relayed|claims?|claimed|thinks?|"
    r"thought|believes?|believed|asks?|asked|wonders?|wondered|quotes?|quoted|writes?|wrote|"
    r"hears?|heard|according|mentions?|mentioned|suggests?|suggested|guess(?:es|ed)?|"
    r"insists?|insisted|that|if|whether)\b",
    re.IGNORECASE,
)
"""Words that make the subject a report or a clause, which asserts nothing about the head."""

_QUOTES: Final = frozenset("\"“”«»「」")
_LEADING: Final = re.compile(
    r"^(?:(?:now|actually|really|in fact|indeed|still|just|truly|a|an|the)\s+)+", re.IGNORECASE)
_TRAILING: Final = re.compile(r"(?:\s+(?:now|actually|really|instead|after all|too))+$",
                              re.IGNORECASE)
_EXCLUDED: Final = frozenset({
    "not", "no", "never", "none", "nor", "and", "or", "but", "nt",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion", "dozen", "half",
    "first", "second", "third", "once", "twice",
    "about", "across", "after", "at", "before", "between", "by", "during", "for", "from",
    "in", "into", "near", "of", "on", "onto", "over", "since", "through", "to", "under",
    "until", "via", "with", "within", "without", "been", "being", "named", "called",
})
"""Tokens no repaired value may hold: negations, conjunctions and numbers, which are
correctable values of a fact, never an identifying descriptor; prepositions, whose phrase
is no parallel of the other value; and verb parts a mis-read clause leaves behind."""
_ARTICLE: Final = re.compile(r"(?:^|(?<=\s))(?P<article>an?|An?)\s+$")
_NAMING_VERB: Final = re.compile(
    r"\b(?:named|called|nicknamed|renamed|known\s+as)\s+$", re.IGNORECASE)
"""A replaced name right after a naming verb states the name itself ("used to be called
Juno"): that is the judge's to correct, never a rewording's."""
_VOWELS: Final = frozenset("aeiouAEIOU")
_MODIFIERS: Final = (
    "other|new|old|older|younger|little|big|first|second|third|pet|baby|best|dear|late|"
    "former|adopted|rescue|family"
)
_DESCRIBED: Final = re.compile(
    r"(?:['’]s|\b(?:his|her|their|its))\s+(?:(?:" + _MODIFIERS + r")\s+)?"
    r"[a-z][a-z-]*(?P<gap>,?\s+)$",
)
"""A possessive descriptor right before a name ("the user's cat ", "Blair's other cat, "):
one word, or one of a closed set of modifiers and one word, so a verb ("Blair's dog
chases ") never counts. The name only identifies what the descriptor describes."""
_LEADING_APPOSITION: Final = re.compile(
    r"^(?P<appositive>(?:the|a|an)\s[^,]+|[^,]*['’]s\s[^,]+),\s+(?P<rest>\S.*)$",
    re.IGNORECASE,
)
"""After a leading name, the appositive that can stand for it ("Juno, the user's cat,
sleeps" gives "The user's cat sleeps")."""
_STANDS_ALONE: Final = re.compile(
    r"(?:^|\b(?:with|to|for|of|on|at|by|from|about|near|beside|behind|toward|towards|"
    r"around|like)\s+)$",
    re.IGNORECASE,
)
"""Where "the" and a descriptor can stand for a name: the start of the statement or
right after a preposition, never after a verb or another word of a noun phrase."""


@dataclass(frozen=True, slots=True)
class HeadRepairV1:
    """One planned rewording of a head a committing correction outdated (C.9.4a).

    The commit writes it as the head's next version in the correction's own transaction:
    op ``revise``, reason ``rewording``, no candidate, the correction's verdict. Every
    field the rewording does not list is the head's own, its support included: the new
    wording states less than the old one, so the head's own quotes still support it.

    Attributes:
        memory_id: The head's Item.
        statement_digest: The digest of the head statement the repair was planned from;
            the commit repairs only a head that still states it, so a support-only move
            of the head (C.11.1 #2 re-binding) keeps the repair and any other move drops it.
        statement: The repaired statement.
        keywords: The repaired keywords.
        embedding: The repaired statement's vector, from the owner's embedding journal (B.9).
        embedding_space: The vector's space.
        embedding_input_digest: ``sha256`` of the repaired statement's UTF-8 bytes.
    """

    memory_id: str
    statement_digest: str
    statement: str
    keywords: tuple[str, ...]
    embedding: tuple[float, ...]
    embedding_space: str
    embedding_input_digest: str


@dataclass(frozen=True, slots=True)
class DescriptorChangeV1:
    """The replaced and the new value a correction's statement names (B.13 #8).

    Attributes:
        old: The replaced value as the statement writes it, without an article.
        new: Its replacement, likewise.
        naming: The statement corrects a name (``is named NEW, not OLD``); a naming
            head that gives the old name is never edited, since the correction
            corrects that head itself. The possessive form is never a name change.
    """

    old: str
    new: str
    naming: bool


def _value(text: str) -> str | None:
    """A replaced or new value without its leading article or adverbs, if it is one."""
    value = _TRAILING.sub("", _LEADING.sub("", text.strip())).strip()
    tokens = token_seq(value)
    if not tokens or len(tokens) > MAX_VALUE_TOKENS or any(
        token in _EXCLUDED or any(char.isdigit() for char in token) for token in tokens
    ):
        return None
    return value


def descriptor_change(statement: str) -> DescriptorChangeV1 | None:
    """Read the value a correction's statement replaces and its replacement (C.9.4a).

    Only one assertion in a closed repairing form is read, ending the statement:
    a subject, a copula (``is``, ``are``, ``was`` or ``were``), optionally ``named`` or
    ``called``, the new value, then ``not``, ``rather than`` or ``instead of`` and the
    replaced value (B.13 #8); or an owner, ``has``, ``had`` or ``owns``, ``a``/``an`` and
    the new value, then ``not``, ``rather than`` or ``instead of``, ``a``/``an`` and the
    replaced value (OPR-3: "Avery has a hamster, not a guinea pig."), always a kind.
    A subject that reports or quotes (``Blair said that ...``), a statement with a
    quotation mark, and a value that is not a short descriptor or name give no change.

    Args:
        statement (str): The accepted correction's statement.

    Returns:
        DescriptorChangeV1 | None: The replaced and the new value, or None when the
            statement does not name both in one of those forms.
    """
    if any(char in _QUOTES for char in statement):
        return None
    text = statement.strip()
    match = _COPULAR.match(text) or _POSSESSIVE.match(text)
    if match is None or _REPORT.search(match["subject"]):
        return None
    old, new = _value(match["old"]), _value(match["new"])
    if old is None or new is None or name_norm(old) == name_norm(new):
        return None
    naming = match.re is _COPULAR and match["naming"] is not None
    if naming and not (is_name(old) and is_name(new)):
        return None
    return DescriptorChangeV1(old=old, new=new, naming=naming)


def kind_assertion(statement: str) -> tuple[str, str] | None:
    """The subject and the new kind of a plain kind assertion (C.9.4a, OPR-4 para-pet).

    The statement is one assertion ``<subject> is|are|was|were a|an NEW``, ending it, with
    no negation or contrast, no quotation mark and no reporting subject; the new kind is
    a value under the same rules as `descriptor_change`'s ("Avery's pet Mochi is a
    hamster." gives ``("Avery's pet Mochi", "hamster")``).

    Args:
        statement (str): The accepted correction's statement.

    Returns:
        tuple[str, str] | None: The subject and the new kind, or None.
    """
    if any(char in _QUOTES for char in statement) or _NEGATION.search(statement):
        return None
    match = _KIND_ASSERTION.match(statement.strip())
    if match is None or _REPORT.search(match["subject"]):
        return None
    new = _value(match["new"])
    return None if new is None else (match["subject"], new)


def named_kinds(statement: str, names: Collection[str]) -> tuple[str, ...]:
    """The kind words by which a head names a pet with one of ``names`` (C.9.4a).

    The kind is one to three lowercase words in one of `_NAMED_KIND_FORMS` and a value
    under `descriptor_change`'s rules; a head that only mentions a kind without naming
    the pet ("Avery has a guinea pig.") names none.

    Args:
        statement (str): The head's statement.
        names (Collection[str]): The pet's names.

    Returns:
        tuple[str, ...]: The distinct kind words, in the order found.
    """
    found: dict[str, str] = {}
    for name in names:
        if not name.strip():
            continue
        for form in _NAMED_KIND_FORMS:
            pattern = form.replace("{name}", re.escape(name))
            for match in re.finditer(pattern, statement):
                value = _value(match["kind"])
                if value is not None and not token_contained(name, value):
                    found.setdefault(name_norm(value), value)
    return tuple(found.values())


def kind_change(statement: str, kinds: Collection[str]) -> DescriptorChangeV1 | None:
    """The kind change of a plain kind assertion, its replaced kind read from the heads.

    ``kinds`` are the kind words by which the heads the judge related ``contradicts``
    name the same pet (`named_kinds`), without the new kind and the pet's own broad
    type. They must agree on exactly one value, under ``name_norm``, which the
    correction's statement does not hold; that value is the replaced kind (OPR-4
    para-pet: "Avery's pet Mochi is a hamster." beside "Avery's guinea pig is named
    Mochi." replaces "guinea pig").

    Args:
        statement (str): The accepted correction's statement.
        kinds (Collection[str]): The heads' kind words for the pet.

    Returns:
        DescriptorChangeV1 | None: The kind change, or None.
    """
    assertion = kind_assertion(statement)
    if assertion is None:
        return None
    new = assertion[1]
    values = {name_norm(kind): kind for kind in kinds if name_norm(kind) != name_norm(new)}
    if len(values) != 1:
        return None
    (old,) = values.values()
    if _value(old) != old or _occurrences(statement, old):
        return None
    return DescriptorChangeV1(old=old, new=new, naming=False)


def _occurrences(text: str, value: str) -> list[re.Match[str]]:
    """Whole-word, case-insensitive occurrences of a value; a possessive may follow."""
    return list(re.finditer(r"(?<![\w-])" + re.escape(value) + r"(?![\w-])", text,
                            re.IGNORECASE))


def _described(before: str, after: str, descriptor: str) -> str:
    """A kind replaced by its generic descriptor; an ``a``/``an`` before it follows suit."""
    new = descriptor
    if not before.strip():
        new = new[:1].upper() + new[1:]
    article = _ARTICLE.search(before)
    if article is not None:
        chosen = "an" if new[:1] in _VOWELS else "a"
        if article["article"][:1].isupper():
            chosen = chosen.capitalize()
        before = before[:article.start("article")] + chosen + " "
    return before + new + after


def _nameless(before: str, after: str, descriptor: str | None) -> str | None:
    """A name taken out of a head: dropped after a possessive descriptor or before a
    leading appositive, else "the" and the descriptor where a noun phrase can stand."""
    described = _DESCRIBED.search(before)
    if described is not None:
        gap = described["gap"]
        if "," in gap:
            if not after.startswith(","):
                return None
            after = after[1:]
        return before[:len(before) - len(gap)] + after
    if not before.strip() and after.startswith(","):
        apposition = _LEADING_APPOSITION.match(after[1:].strip())
        if apposition is None:
            return None
        appositive = apposition["appositive"]
        return f"{appositive[:1].upper()}{appositive[1:]} {apposition['rest']}"
    if descriptor is None or not _STANDS_ALONE.search(before):
        return None
    new = f"the {descriptor}"
    if not before.strip():
        new = new[:1].upper() + new[1:]
    return before + new + after


def repaired_statement(
    statement: str, change: DescriptorChangeV1, *, names: Collection[str],
    descriptor: str | None,
) -> str | None:
    """Reword a head's one occurrence of the replaced value without either value (C.9.4a).

    The head must hold the replaced value exactly once as whole words, outside every
    name it gives; under a name change a naming head that gives the replaced name is
    never edited, nor a replaced name right after a naming verb, which states the name
    itself. A kind takes ``descriptor`` ("Avery's pet is named Mochi."), with an
    article ``a``/``an`` before it following its first letter. A name is dropped when
    a possessive descriptor precedes it ("The user's cat sleeps on the windowsill all
    day.", also "the user's cat, Juno, sleeps") or a leading appositive can stand for
    it ("Juno, the user's cat, sleeps" gives "The user's cat sleeps"); otherwise it
    becomes "the" and ``descriptor`` at the start of the statement or right after a
    preposition ("The pet's bowl is blue.", "plays with the pet"), and anywhere else
    the head is not reworded. A value starting the statement keeps its capital. The
    result is canonical, differs from the head, still contains every name the head
    gives and holds neither value.

    Args:
        statement (str): The head's statement.
        change (DescriptorChangeV1): The correction's replaced and new value.
        names (Collection[str]): The names the head's naming annotations give.
        descriptor (str | None): The corrected entity's generic descriptor
            (`generic_descriptor`), or None when it has none.

    Returns:
        str | None: The reworded statement, or None when the head cannot be reworded
            by this substitution alone.
    """
    if descriptor is not None and generic_descriptor(descriptor, change) != descriptor:
        return None
    if change.naming and any(name_norm(name) == name_norm(change.old) for name in names):
        return None
    found = _occurrences(statement, change.old)
    if len(found) != 1:
        return None
    start, end = found[0].span()
    before, after = statement[:start], statement[end:]
    if change.naming and _NAMING_VERB.search(before):
        return None
    for name in names:
        if any(match.start() < end and start < match.end()
               for match in _occurrences(statement, name)):
            return None
    text = _nameless(before, after, descriptor) if change.naming else (
        None if descriptor is None else _described(before, after, descriptor))
    if text is None:
        return None
    try:
        result = canonical_statement(text)
    except ValueError:
        return None
    if (result == statement or not all(token_contained(name, result) for name in names)
            or _occurrences(result, change.old) or _occurrences(result, change.new)):
        return None
    return result


def repaired_keywords(keywords: Sequence[str], change: DescriptorChangeV1, *,
                      descriptor: str | None) -> tuple[str, ...]:
    """Reword the replaced value in a head's keywords, keeping their order (B.6.3 keywords).

    A keyword that holds the replaced value as whole words takes the head's new wording
    instead: ``descriptor`` for a kind, nothing for a name. A keyword left empty is
    dropped, one that would grow past its bound or still hold the value is dropped, and
    a repeat of an earlier one is dropped, so the head is never found by either value.

    Args:
        keywords (Sequence[str]): The head's keywords.
        change (DescriptorChangeV1): The correction's replaced and new value.
        descriptor (str | None): The kind's generic descriptor; unused for a name.

    Returns:
        tuple[str, ...]: The reworded keywords, unique under ``name_norm``.
    """
    new = "" if change.naming else (descriptor or "")
    repaired: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        value = keyword
        for match in reversed(_occurrences(keyword, change.old)):
            value = value[:match.start()] + new + value[match.end():]
        value = " ".join(value.split())
        key = name_norm(value)
        if (not key or key in seen or len(value) > MAX_KEYWORD_CHARS
                or _occurrences(value, change.old) or _occurrences(value, change.new)):
            continue
        seen.add(key)
        repaired.append(value)
    return tuple(repaired)


_GENERIC: Final = re.compile(r"^[a-z]+(?: [a-z]+){0,2}$")
_NOT_GENERIC: Final = frozenset({"person", "user", "participant", "people", "thing", "entity"})
"""Entity types that read wrongly as a possessive descriptor ("Avery's person")."""


def generic_descriptor(entity_type: str, change: DescriptorChangeV1) -> str | None:
    """The descriptor that stands for a corrected kind or a renamed entity's name.

    A head reworded with the new kind restates the correction itself ("Avery's hamster
    is named Mochi." beside "Avery's pet Mochi is a hamster, not a guinea pig."), keeps
    a kind the next correction must change again and needs the correction's quotes, so it
    takes the entity's own broad type instead ("Avery's pet is named Mochi."), which
    the head's own quotes support ("my guinea pig Mochi"); a renamed entity's name becomes
    "the" and this descriptor where no descriptor precedes it ("The pet's bowl is blue.").
    It is the free-text type the extraction declared for the entity (UD2), used only when
    it is one to three lowercase words, not a person, and shares no word with the
    replaced or the new value.

    Args:
        entity_type (str): The named entity's free-text type.
        change (DescriptorChangeV1): The correction's replaced and new value.

    Returns:
        str | None: The descriptor, or None when the type is no usable descriptor.
    """
    value = entity_type.strip()
    if not _GENERIC.fullmatch(value) or value in _NOT_GENERIC:
        return None
    words = set(token_seq(value))
    if words & (set(token_seq(change.old)) | set(token_seq(change.new))):
        return None
    return value


__all__ = [
    "MAX_VALUE_TOKENS",
    "DescriptorChangeV1",
    "HeadRepairV1",
    "descriptor_change",
    "generic_descriptor",
    "kind_assertion",
    "kind_change",
    "named_kinds",
    "repaired_keywords",
    "repaired_statement",
]
