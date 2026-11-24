"""LongText logical type."""

from __future__ import annotations

__all__ = [
    "LongText",
]

from .limits import _BoundedText


class LongText(_BoundedText):
    """Long text logical type with descriptor-owned length semantics.

    Canonical form is a ``str``. ``None`` normalises to ``""``. Length is
    capped by the immutable `max_chars` descriptor value, which defaults to
    65535 and is included in serialized schemas.

    Args:
        max_chars (int | None): Positive maximum character count. `None` uses
            the stable built-in default of 65535.

    Warnings:
        Values longer than the configured limit raise ``ValueError`` during
        validation, not truncation.

    Examples:
        ``hb.LongText().validate("body")`` returns ``"body"``.
        ``hb.LongText().validate(None)`` returns ``""``.
    """

    identifier = "long-text"
    default_max_chars = 65535
