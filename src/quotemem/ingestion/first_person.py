"""Singular first-person mentions: the speaker of their turn and nobody else.

The ``user-role-v1`` self-identity rule binds such a mention to the authenticated user when
every quote of its fact lies in the user's own turns (``extraction_admission``). This
module only recognizes the mention; it reads no source.
"""

from __future__ import annotations

import unicodedata
from typing import Final

FIRST_PERSON_FORMS: Final = frozenset({
    # Chinese.
    "我", "我的", "我自己", "本人", "俺", "俺的",
    # English.
    "i", "me", "my", "mine", "myself", "i'm", "im", "i've", "i'd", "i'll",
    # Japanese.
    "私", "私の", "わたし", "僕", "僕の", "ぼく", "俺の", "おれ",
    # Korean.
    "나", "내", "저", "제",
})
"""Mentions, after `mention_form`, that name the speaker of their turn and nobody else.

Plural and inclusive forms ("我们", "we", "us", "咱们") are left out: they name a group the
user belongs to, not the user."""

_EDGE_MARKS: Final = "\"'`“”‘’「」『』《》()（）[]【】.,;:!?。，；：！？、 \t\u3000"


def mention_form(mention: str) -> str:
    """Normalize a mention for `FIRST_PERSON_FORMS`: NFKC, case-folded, edge marks stripped.

    Args:
        mention (str): The mention as the extractor wrote it.

    Returns:
        str: Its comparison form; the curly apostrophe reads as ``'``.
    """
    text = unicodedata.normalize("NFKC", mention).replace("\u2019", "'").casefold()
    return text.strip(_EDGE_MARKS)


def first_person(mention: str | None) -> bool:
    """Whether a mention is a singular first-person form (`FIRST_PERSON_FORMS`)."""
    return mention is not None and mention_form(mention) in FIRST_PERSON_FORMS
