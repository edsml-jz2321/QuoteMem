"""Optional scrubbing of email addresses and phone numbers before provider calls.

With ``QUOTEMEM_SCRUB_PII=1`` every chat message and embedding input leaving the
process has its email addresses and phone numbers replaced by placeholders such as
``[pii-email-1]``. The store keeps the verbatim evidence locally: the core built
the request from it and never sees the placeholders, because a chat reply has
each placeholder the request issued restored to its original text before the core
reads it. A quote that copies a placeholder therefore still matches its source;
one that mangles it fails the core's quote check like any other wrong quote.

Embedding inputs are scrubbed without restoration: stored vectors and query
vectors both represent the scrubbed text, so they stay comparable.

The patterns are deliberately narrow, to leave identifiers, digests and
timestamps in structured requests alone: an email address, an international
number (``+`` and at least eight digits), a separated local number (groups joined
by spaces, dots or hyphens, or an area code in parentheses), or an 11-digit
mainland China mobile number. A bare run of digits is never treated as a phone
number otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
                    r"(?![\w-])")
_PHONE = re.compile(
    r"(?<![\w+/.:-])(?:"
    r"\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){2,5}"  # international
    r"|\(\d{2,4}\)[ .-]?\d{3,4}[ .-]?\d{3,4}"  # (area) local
    r"|\d{3,4}[ .-]\d{3,4}[ .-]\d{3,4}"  # separated groups
    r"|1[3-9]\d{9}"  # mainland China mobile
    r")(?![\w/:-])"
)
_PLACEHOLDER = re.compile(r"\[pii-(?:email|phone)-\d+\]")


def scrub_enabled(environment: Mapping[str, str]) -> bool:
    """Whether ``QUOTEMEM_SCRUB_PII`` asks for scrubbing."""
    return environment.get("QUOTEMEM_SCRUB_PII", "").strip().lower() in {"1", "true", "yes"}


def _digits(value: str) -> int:
    return sum(character.isdigit() for character in value)


@dataclass
class Scrubbing:
    """The placeholders one request issued, for restoring its reply.

    Attributes:
        originals: Each placeholder and the text it replaced.
        counts: How many emails and phone numbers were replaced.
    """

    originals: dict[str, str] = field(default_factory=dict[str, str])
    counts: dict[str, int] = field(default_factory=lambda: {"email": 0, "phone": 0})
    _assigned: dict[tuple[str, str], str] = field(default_factory=dict[tuple[str, str], str])
    _taken: set[str] = field(default_factory=set[str])

    def reserve(self, text: str) -> None:
        """Never issue a placeholder that already occurs in the outgoing text."""
        self._taken.update(_PLACEHOLDER.findall(text))

    def _placeholder(self, kind: str, original: str) -> str:
        found = self._assigned.get((kind, original))
        if found is not None:
            return found
        index = self.counts[kind] + 1
        token = f"[pii-{kind}-{index}]"
        while token in self._taken:
            index += 1
            token = f"[pii-{kind}-{index}]"
        self._taken.add(token)
        self._assigned[(kind, original)] = token
        self.originals[token] = original
        self.counts[kind] += 1
        return token

    def scrub(self, text: str) -> str:
        """Replace email addresses, then phone numbers, in ``text``."""
        text = _EMAIL.sub(lambda match: self._placeholder("email", match.group(0)), text)

        def phone(match: re.Match[str]) -> str:
            value = match.group(0)
            if _digits(value) < 8:
                return value
            return self._placeholder("phone", value)

        return _PHONE.sub(phone, text)

    def restore(self, text: str) -> str:
        """Put back every placeholder this request issued."""
        if not self.originals:
            return text
        return _PLACEHOLDER.sub(
            lambda match: self.originals.get(match.group(0), match.group(0)), text,
        )

    @property
    def replaced(self) -> int:
        """The number of replacements."""
        return sum(self.counts.values())
