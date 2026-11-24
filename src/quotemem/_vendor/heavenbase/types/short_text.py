"""ShortText logical type."""

from __future__ import annotations

__all__ = [
    "ShortText",
]

from .limits import _BoundedText


class ShortText(_BoundedText):
    """Short text logical type with descriptor-owned length semantics.

    Canonical form is a ``str``. ``None`` normalises to ``""``. Length is
    capped by the immutable `max_chars` descriptor value, which defaults to
    255 and is included in serialized schemas.

    Args:
        max_chars (int | None): Positive maximum character count. `None` uses
            the stable built-in default of 255.

    Warnings:
        Values longer than the configured limit raise ``ValueError`` during
        validation, not truncation.

    Examples:
        ``hb.ShortText().validate("hello")`` returns ``"hello"``.
        ``hb.ShortText().validate(None)`` returns ``""``.
    """

    identifier = "short-text"
    default_max_chars = 255
