"""The bracketed envelope header that can open an Add message's text.

Some clients prepend a bracketed transport header, such as
``[<dataset>][session_<n>][<turn>][<kind>] <speaker>: <text>``. The header is
transport framing, not speech: the
date trailer reader (``adapters/http/models.py``) requires it, and the speaker
label locator (``ingestion/speaker_labels.py``) reads a label through it, so an
envelope ``Name:`` line is the same turn label as a plain ``Name:`` line.
"""

from __future__ import annotations

import re
from typing import Final

ENVELOPE_HEADER: Final = re.compile(
    r"\[[A-Za-z0-9_.-]{1,64}\]\[session_[^\]\n]{1,32}\]\[[^\]\n]{1,32}\]\[[A-Za-z_]{1,32}\] "
)
"""One envelope header, bracket fields and the single space after them, matched at a position."""


def envelope_end(text: str, start: int = 0) -> int:
    """Return the offset after an envelope header at ``start``, or ``start`` when none opens there.

    Args:
        text (str): The text.
        start (int): Where the header would begin, such as a line start.

    Returns:
        int: The first offset after the header, or ``start`` itself.
    """
    header = ENVELOPE_HEADER.match(text, start)
    return start if header is None else header.end()
