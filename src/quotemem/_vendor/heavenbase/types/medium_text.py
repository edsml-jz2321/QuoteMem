"""MediumText logical type."""

from __future__ import annotations

__all__ = [
    "MediumText",
]

from .limits import _BoundedText


class MediumText(_BoundedText):
    """Medium text logical type with descriptor-owned length semantics.

    Canonical form is a ``str``. ``None`` normalises to ``""``. Length is
    capped by the immutable `max_chars` descriptor value, which defaults to
    4095 and is included in serialized schemas.

    Args:
        max_chars (int | None): Positive maximum character count. `None` uses
            the stable built-in default of 4095.

    Warnings:
        Values longer than the configured limit raise ``ValueError`` during
        validation, not truncation.

    Examples:
        ``hb.MediumText().validate("summary")`` returns ``"summary"``.
        ``hb.MediumText().validate(None)`` returns ``""``.
    """

    identifier = "medium-text"
    default_max_chars = 4095
