"""Regex alternations of the closed word lists of the ML-1 lexicons."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


def alternation(words: Iterable[str], space: str) -> str:
    """Return one group matching any of the words, in their composed and decomposed forms.

    Args:
        words (Iterable[str]): Closed words or phrases; a space inside a phrase
            stands for ``space``.
        space (str): The regex a space reads as, such as ``\\s+``.

    Returns:
        str: A non-capturing group of the escaped forms, longest first.
    """
    forms = {unicodedata.normalize(form, word) for word in words for form in ("NFC", "NFD")}
    return "(?:" + "|".join(
        space.join(re.escape(part) for part in form.split(" "))
        for form in sorted(forms, key=lambda form: (-len(form), form))
    ) + ")"
