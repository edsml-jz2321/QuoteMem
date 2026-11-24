"""Ground affirmative alias naming in located source clauses.

The provider reply stays verbatim. Admission uses this closed normalization only
for a naming fact and a quoted alias marker, never for an activity that happens
to mention somebody's nickname or for a source that also states a rename.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from quotemem.core.extraction_output import FactV12, NamingV12
from quotemem.core.readability import MaskedText
from quotemem.core.statement import is_name, name_norm, token_contained, token_seq

_MARKER: Final = (
    r"(?:(?:now|also|still|currently)\s+)?(?:go(?:es)?\s+by|prefer(?:s)?\s+to\s+be\s+called"
    r"|is\s+(?:also\s+)?known\s+as|is\s+nicknamed|has\s+(?:the\s+)?nickname)"
)
_CLAUSE: Final = re.compile(
    r"(?P<subject>.+?)(?:\s+(?P<marker>" + _MARKER + r")"
    r"|(?P<possessive>['’]s\s+nickname\s+is)|\s+(?P<nickname>nickname\s+is))\s+"
    r"(?P<alias>.+?)(?P<tail>\s+(?:now|these\s+days|currently))?[.!]?",
    re.IGNORECASE,
)
_RENAME: Final = re.compile(
    r"\brenam(?:e[ds]?|ing)\b|\bchang(?:e[ds]?|ing)\s+(?:(?:his|her|their|my|its|our|the)\s+)?"
    r"(?:legal\s+)?name\b|\b(?:new|legal|former|previous|old|birth)\s+name\b"
    r"|\b(?:used\s+to\s+be|formerly)\s+(?:named|called|known)\b",
    re.IGNORECASE,
)
_NON_ASSERTION: Final = re.compile(
    r"\b(?:not|never|no\s+longer|used\s+to|if|unless|whether|would|could|might"
    r"|denies|denied|deny|false|untrue|wonder|wonders|wondering|ask|asks|asked"
    r"|says|said|reports|reported|claims|claimed|believes|believed|thinks|thought)\b|n['’]t\b",
    re.IGNORECASE,
)
_NAMING_VERB: Final = frozenset({
    "name", "named", "names", "nickname", "nicknamed", "called", "known", "go", "goes",
    "prefer", "prefers", "changed", "change", "renamed", "rename", "formerly",
})
_NAMING_TAIL: Final = _NAMING_VERB | frozenset({
    "is", "was", "are", "were", "has", "have", "had", "been", "now", "currently", "also", "still",
    "as", "by", "to", "be", "from", "their", "his", "her", "its", "my", "our", "the",
    "a", "an", "new", "old", "former", "previously", "not", "instead", "of", "these", "days",
    "legal", "first", "last", "full",
})
_RELATIONS: Final = frozenset({
    "friend", "cousin", "sister", "brother", "mother", "father", "parent", "child",
    "daughter", "son", "wife", "husband", "partner", "aunt", "uncle", "niece", "nephew",
    "grandmother", "grandfather", "grandparent", "grandchild", "colleague", "coworker",
    "boss", "neighbor", "neighbour", "roommate", "teacher", "student",
})


def _sentences(source: str, names: Sequence[str]) -> list[tuple[int, int, str]]:
    """Keep sentence offsets and punctuation inside admitted names intact."""
    protected = source
    for name in names:
        if "." in name:
            protected = re.sub(re.escape(name), lambda match: match[0].replace(".", "\0"),
                               protected, flags=re.IGNORECASE)
    sentences: list[tuple[int, int, str]] = []
    start = 0
    for boundary in re.finditer(r"(?<=[.!?;])\s+|\n|$", protected):
        part = source[start:boundary.start()]
        sentence = part.strip()
        if sentence:
            offset = start + len(part) - len(part.lstrip())
            sentences.append((offset, offset + len(sentence), sentence))
        start = boundary.end()
    return sentences


def _naming_only(statement: str, names: Sequence[str]) -> bool:
    """A naming predicate's tail has only naming words, names and the closed time words."""
    tokens = token_seq(statement)
    first = next((index for index, token in enumerate(tokens) if token in _NAMING_VERB), None)
    if first is None:
        return False
    if tokens[first] == "nickname" and first >= 2 and tokens[first - 2:first] == ["has", "the"]:
        first -= 2  # The fixed article belongs to this naming predicate, not its subject.
    name_tokens = {token for name in names for token in token_seq(name)}
    prefix = tuple(token for token in tokens[:first] if token not in name_tokens | {
        "s", "is", "was", "are", "were", "has", "have", "had", "been", "now", "also", "currently",
        "still",
    })
    nominal = not prefix or prefix == ("the", "user") or (
        len(prefix) == 2 and prefix[0] in {"a", "an", "the"}
    ) or (len(prefix) == 3 and prefix[:2] == ("the", "user"))
    if not nominal:
        return False
    return all(token in _NAMING_TAIL or token in name_tokens for token in tokens[first:])


def normalize_alias_fact(
    fact: FactV12, *, quoted: Sequence[tuple[MaskedText, int, int]],
    names: Mapping[str, Sequence[str]],
) -> FactV12 | None:
    """Return a source-grounded alias fact, or None outside the closed alias rule.

    Args:
        fact: A fact whose groups, anchors and naming have been admitted. Its raw
            provider representation remains stored separately.
        quoted: Masked sources and their located quote character bounds. The
            quote must contain the alias marker and target in the same readable
            clause; the source also prevents a clipped quote from hiding a rename.
        names: Admitted anchor handles and their catalog/declaration names. The
            authenticated user additionally has ``I`` and ``my`` as source forms.

    Returns:
        FactV12 | None: One state naming only the supported alias, with a grounded
        statement and no invented former name or correction; None when a source
        is mixed, negated, conditional, ambiguous or the fact has another claim.
    """
    if not set(token_seq(fact.statement)) & _NAMING_VERB:
        return None
    all_known = [*(name for values in names.values() for name in values),
                 *(item.name for item in fact.naming)]
    candidates: dict[tuple[str, str], tuple[str, str, str]] = {}
    for source, quote_start, quote_end in quoted:
        for start, end, sentence in _sentences(source.text, all_known):
            overlap_start, overlap_end = max(start, quote_start), min(end, quote_end)
            if overlap_start >= overlap_end or source.spans(start, end) is None:
                continue
            if sentence.lstrip().startswith(('"', "'", "“", "‘")):
                continue
            clause = _CLAUSE.fullmatch(sentence)
            if clause is None or _NON_ASSERTION.search(sentence):
                continue
            alias = clause["alias"].strip(' \t"“”‘’')
            matching = [item.name for item in fact.naming
                        if name_norm(item.name) == name_norm(alias)]
            alias = next((name for name in matching
                          if sentence[clause.start("alias"):].startswith(name)), alias)
            if not is_name(alias) or any(mark in alias for mark in ",;:?!"):
                continue
            marker = clause["marker"] or clause["possessive"] or clause["nickname"]
            # The subject may be outside a minimal naming quote, but its marker
            # and alias cannot be borrowed from elsewhere in the rendered source.
            quoted_marker = re.sub(
                r"^(?:now|also|still|currently)\s+", "", marker, flags=re.IGNORECASE,
            )
            if not token_contained(
                quoted_marker + " " + alias, source.text[overlap_start:overlap_end],
            ):
                continue
            subject = clause["subject"].strip()
            matches = [
                (handle, name) for handle, values in names.items() for name in values
                if token_contained(name, subject)
            ]
            handles = {handle for handle, _ in matches}
            if len(handles) != 1:
                continue
            handle = next(iter(handles))
            if any(item.entity != handle for item in fact.naming):
                continue
            known = [name for item_handle, name in matches if item_handle == handle]
            # Only a closed possessive relationship may precede a known name;
            # a name inside a relational subject does not identify that subject.
            subject_tokens = token_seq(subject)
            subject_name = max(known, key=len)
            name_tokens = token_seq(subject_name)
            prefix = subject_tokens[:-len(name_tokens)]
            if subject_tokens[-len(name_tokens):] != name_tokens or (
                prefix and not (
                    prefix[-1] in _RELATIONS
                    and tuple(prefix[:-1]) in {("my",), ("our",), ("the", "user", "s")}
                )
            ):
                continue
            all_names = [alias, *names[handle], *(item.name for item in fact.naming)]
            if any(
                _RENAME.search(other) and any(token_contained(name, other) for name in all_names)
                for cited_source, _, _ in quoted
                for _, _, other in _sentences(cited_source.text, all_known)
            ):
                continue
            if not _naming_only(fact.statement, all_names):
                continue
            annotated = any(name_norm(item.name) == name_norm(alias) for item in fact.naming)
            if not annotated and not all(
                word[:1].isupper() for word in alias.split() if word not in {"de", "van", "von"}
            ):
                continue
            if handle == "user" and name_norm(subject_name) in {"i", "my"}:
                subject_name = "The user"
            tail = clause["tail"] or ""
            marker_words = token_seq(marker)
            marker_tokens = set(marker_words)
            modifier = (
                marker_words[0]
                if marker_words[0] in {"now", "also", "still", "currently"}
                else None
            )
            if marker_tokens & {"prefer", "prefers"}:
                predicate = "prefers to be called"
            elif "known" in marker_tokens:
                qualifier = "also " if "also" in marker_tokens else ""
                if modifier and modifier != "also":
                    qualifier = f"{modifier} {qualifier}"
                predicate = f"is {qualifier}known as"
            elif marker_tokens & {"nickname", "nicknamed"}:
                predicate = "has the nickname"
            else:
                predicate = "goes by"
            if modifier and "known" not in marker_tokens:
                predicate = f"{modifier} {predicate}"
            statement = f"{subject_name} {predicate} {alias}{tail}"
            if not statement.endswith("."):
                statement += "."
            if (fact.kind == "state"
                    and not _RENAME.search(fact.statement)
                    and not _NON_ASSERTION.search(fact.statement) and fact.naming) and all(
                item.name_kind == "alias" and name_norm(item.name) == name_norm(alias)
                for item in fact.naming
            ) and alias in fact.statement and token_contained(
                predicate + " " + alias + tail, fact.statement,
            ):
                statement = fact.statement
            candidates[(handle, name_norm(alias))] = (handle, alias, statement)
    if len(candidates) != 1:
        return None
    handle, alias, statement = next(iter(candidates.values()))
    return fact.model_copy(update={
        "statement": statement,
        "kind": "state",
        "naming": [NamingV12(entity=handle, name=alias, name_kind="alias")],
        "corrects_text": None,
    })
