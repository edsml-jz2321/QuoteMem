"""Artifact (BLOB) logical type."""

from __future__ import annotations

__all__ = [
    "Artifact",
]

from heavenbase.utils import Any
from heavenbase.utils.serialize import loads_b64

from .base import LogicalType


class Artifact(LogicalType):
    """Binary large object logical type.

    Canonical form is ``bytes``. ``None`` normalises to empty ``bytes``.
    Strings are encoded as UTF-8 on input. Objects exposing ``tobytes()``
    (for example NumPy arrays) are accepted during validation.

    :meth:`encode` returns raw ``bytes``. Backends that cannot store binary
    (VARCHAR metadata, JSON file rows, SQL dialects without BLOB) apply
    standard base64 text encoding at their own boundary. :meth:`decode`
    accepts native ``bytes`` and base64 ``str`` values read from storage.

    Warnings:
        User-facing string inputs are interpreted as UTF-8 text, not base64.
        Pass ``bytes`` when storing opaque binary payloads.

    Examples:
        ``hb.Artifact().validate(b"\x00\x01")`` returns ``b"\x00\x01"``.
        ``hb.Artifact().validate("hello")`` returns ``b"hello"``.
        ``hb.Artifact().encode(b"\xab")`` returns ``b"\xab"``.
    """

    identifier = "artifact"

    def validate(self, value: Any) -> bytes:
        """Normalise artifact values to ``bytes``.

        Args:
            value: ``bytes``, ``bytearray``, UTF-8 ``str``, ``None``, or an
                object with ``tobytes()``.

        Returns:
            Normalised ``bytes`` payload.

        Raises:
            ValueError: If ``value`` cannot be converted to bytes.
        """
        if value is None:
            return b""
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, str):
            return value.encode("utf-8")
        if hasattr(value, "tobytes"):
            return value.tobytes()
        raise ValueError(f"Artifact expects bytes, str, or None, got {type(value).__name__}")

    def encode(self, value: Any) -> bytes:
        """Return canonical bytes for backend storage.

        Args:
            value: Canonical or raw input accepted by :meth:`validate`.

        Returns:
            Normalised ``bytes``. Text-only backends base64-encode at their
            own adapter boundary; SQL BLOB columns store these bytes directly.
        """
        return self.validate(value)

    def decode(self, value: Any) -> bytes:
        """Restore canonical bytes from storage or backend values.

        Args:
            value: Native BLOB ``bytes``, base64 ``str`` from text-only
                storage, or other input accepted by :meth:`validate`.

        Returns:
            Canonical ``bytes`` payload.
        """
        if isinstance(value, str):
            try:
                return loads_b64(value)
            except ValueError:
                return self.validate(value)
        return self.validate(value)
