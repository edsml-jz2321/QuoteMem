"""Redact secret-like substrings from user-visible text."""

from __future__ import annotations

__all__ = ["redact_data", "redact_secrets"]

import re
from collections.abc import Mapping

from .typing import Any

_SECRET_NAME = r"api[_-]?key|access[_-]?key|authorization|auth[_-]?token|client[_-]?secret|credential|password|private[_-]?key|refresh[_-]?token|secret|token"
_SECRET_VALUE_RE = re.compile(rf"(?i)\b({_SECRET_NAME})\b\s*([=:])\s*(.+?)" rf"(?=\s+\b(?:{_SECRET_NAME})\b\s*[=:]|\s+[a-z][a-z0-9+.-]*://|[\n\r,;]|$)")
_URL_PASSWORD_RE = re.compile(r"([a-z][a-z0-9+.-]*://[^\s/@:]+:)([^\s/@]+)(@)", re.IGNORECASE)
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)^(?:.*[_-])?(api[_-]?key|access[_-]?key(?:[_-]?id)?|authorization|auth[_-]?token|client[_-]?secret|credential|dsn|password|private[_-]?key|refresh[_-]?token|secret(?:[_-]?access[_-]?key)?|token)$"
)


def redact_secrets(text: str, replacement: str = "<redacted>") -> str:
    """Return ``text`` with secret-like values and URL passwords replaced.

    Args:
        text (str): Text that may contain credentials.
        replacement (str): Marker substituted for each credential. Defaults
            to ``"<redacted>"``.

    Returns:
        str: Redacted text.
    """
    redacted = _SECRET_VALUE_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}{replacement}", str(text))
    return _URL_PASSWORD_RE.sub(lambda match: f"{match.group(1)}{replacement}{match.group(3)}", redacted)


def redact_data(value: Any, replacement: str = "<redacted>") -> Any:
    """Return a JSON-like value with sensitive keys and strings redacted.

    Args:
        value (Any): JSON-like value to copy and redact.
        replacement (str): Marker substituted for credentials. Defaults to
            ``"<redacted>"``.

    Returns:
        Any: Recursively redacted copy of ``value``.
    """
    if isinstance(value, str):
        return redact_secrets(value, replacement=replacement)
    if isinstance(value, Mapping):
        field = value.get("key")
        redact_value = isinstance(field, str) and bool(_SENSITIVE_KEY_RE.match(field))
        out = {}
        for key, item in value.items():
            key_str = str(key)
            sensitive = bool(_SENSITIVE_KEY_RE.match(key_str)) or (redact_value and key_str == "value")
            out[key_str] = replacement if sensitive else redact_data(item, replacement=replacement)
        return out
    if isinstance(value, (list, tuple)):
        return [redact_data(item, replacement=replacement) for item in value]
    return value
