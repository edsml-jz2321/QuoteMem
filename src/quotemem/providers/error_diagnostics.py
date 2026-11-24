"""Bounded provider code/message diagnostics without request credentials."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import cast

import httpx

_CREDENTIAL_NAME = re.compile(r"auth|token|key|secret|cookie|password", re.IGNORECASE)
_CREDENTIAL_TEXT = re.compile(
    r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+|\bsk-[A-Za-z0-9_-]{8,}\b|"
    r"\b(?:api[_ -]?key|access[_ -]?token|password|secret|authorization)"
    r"\s*[:=]\s*[\"']?[^\s\"',;]+", re.IGNORECASE,
)
_MAX_BYTES = 2048


def provider_error_body(
    response: httpx.Response, request: httpx.Request | None = None,
) -> dict[str, str]:
    """Return a provider error's sanitized code/message within 2 KiB of JSON.

    Only code and message fields are retained from JSON objects. Plain-text error
    bodies become the message. Credentials carried by the request and common
    credential spellings are redacted before the diagnostic is truncated.

    Args:
        response (httpx.Response): An already-read provider response.
        request (httpx.Request | None): Its request, when the response has none.

    Returns:
        dict[str, str]: Bounded diagnostic fields suitable for persisted evidence.
    """
    if request is None:
        try:
            request = response.request
        except RuntimeError:
            pass
    secrets: set[str] = set()
    if request is not None:
        for name, value in (*request.headers.multi_items(), *request.url.params.multi_items()):
            if value and _CREDENTIAL_NAME.search(name):
                secrets.add(value)
                if name.lower() in ("authorization", "proxy-authorization"):
                    secrets.add(value.rsplit(" ", 1)[-1])
        secrets.update(value for value in (request.url.username, request.url.password) if value)

    def sanitize(value: object) -> str:
        text = str(value)
        for secret in sorted(secrets, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        return " ".join(_CREDENTIAL_TEXT.sub("[REDACTED]", text).split())

    try:
        raw: object = response.json()
    except (ValueError, UnicodeError):
        raw = response.text
    found: dict[str, str] = {}
    if isinstance(raw, Mapping):
        supplied = cast(Mapping[str, object], raw)
        nested = supplied.get("error")
        details = cast(Mapping[str, object], nested) if isinstance(nested, Mapping) else supplied
        for field, names in (("code", ("code", "Code", "error_code")),
                             ("message", ("message", "Message", "error_message",
                                          "error_description"))):
            value = next((details[name] for name in names if name in details), None)
            if isinstance(value, str | int | float):
                found[field] = sanitize(value)
        if not found and isinstance(nested, str):
            found["message"] = sanitize(nested)
    elif isinstance(raw, str):
        found["message"] = sanitize(raw)
    if not found:
        found["message"] = "[provider error body has no code or message]"

    # Preserve the short provider code when an unusually long message needs clipping.
    for field in ("code", "message"):
        if field not in found:
            continue
        limit = 256 if field == "code" else _MAX_BYTES - len(json.dumps({
            key: value for key, value in found.items() if key != "message"
        }, ensure_ascii=True).encode()) - 16
        text = found[field]
        while len(json.dumps(text, ensure_ascii=True).encode()) > limit:
            if not text:
                break
            text = text[:max(0, len(text) - max(1, len(text) // 8))]
            found[field] = text
    return found
