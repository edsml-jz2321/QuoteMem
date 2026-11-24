"""Normalize ordinary agent chat input before the strict bench wire contract reads it.

The bench contract was written for a benchmark platform that always sends complete,
well-formed requests. A plugin capturing live chat sends what the chat has. Every
rule here is a pure function of the request body, so a retried capture produces the
identical normalized request and keeps Add idempotent:

- ``user_id`` defaults to the service's default user and ``session_id`` to
  ``default``;
- a missing ``request_id`` is derived from the user, session and normalized
  messages (``agent-`` and 40 hex digits);
- ``tool`` and ``function`` messages are dropped: tool output is not the user's
  conversation;
- text-part arrays are joined, and credential spans that the code-owned detector
  recognizes are replaced by ``[redacted credential]`` instead of storing them (the
  bench profile stores them verbatim, the product profile rejects the whole Add);
- timestamps in seconds become milliseconds, and ISO 8601 strings become Unix
  milliseconds (UTC when they carry no offset). Missing timestamps stay missing:
  stamping the receive time would make a retry differ from its first attempt, so
  the client stamps each message when it captures it.

Search gets the default user and ``top_k`` when they are absent. Malformed bodies
are forwarded untouched, so the strict contract reports them as before.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, cast

from quotemem.ingestion.boundary import recognized_credential_spans

REDACTION: Final = "[redacted credential]"
DROPPED_ROLES: Final = frozenset({"tool", "function"})
DEFAULT_SESSION_ID: Final = "default"
_SECONDS_LIMIT: Final = 100_000_000_000
"""Positive timestamps below this are seconds: as milliseconds it is March 1973."""


@dataclass
class IngestNotes:
    """What normalization changed in one request; counts only, no content."""

    redacted_spans: int = 0
    dropped_messages: int = 0
    scaled_timestamps: int = 0
    parsed_timestamps: int = 0
    derived_request_id: bool = False
    defaults: list[str] = field(default_factory=lambda: [])


def redact_credentials(text: str) -> tuple[str, int]:
    """Replace each recognized credential span; overlapping spans merge.

    Returns:
        The text and the number of merged spans replaced.
    """
    data = text.encode("utf-8", errors="surrogatepass")
    spans = sorted((start, end) for _, start, end in recognized_credential_spans(data))
    if not spans:
        return text, 0
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    pieces: list[bytes] = []
    cursor = 0
    for start, end in merged:
        pieces.extend((data[cursor:start], REDACTION.encode()))
        cursor = end
    pieces.append(data[cursor:])
    return b"".join(pieces).decode("utf-8", errors="surrogatepass"), len(merged)


def _timestamp(value: object, notes: IngestNotes) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and 0 < value < _SECONDS_LIMIT:
        notes.scaled_timestamps += 1
        return round(value * 1000)
    if isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return value
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        notes.parsed_timestamps += 1
        return round(moment.timestamp() * 1000)
    return value


def _text(content: object) -> object:
    if not isinstance(content, list):
        return content
    parts = cast(list[Any], content)
    if all(
        isinstance(part, str) or (
            isinstance(part, dict) and isinstance(cast(dict[str, Any], part).get("text"), str)
        ) for part in parts
    ):
        return "".join(
            part if isinstance(part, str) else cast(dict[str, Any], part)["text"]
            for part in parts
        )
    return parts


def derive_request_id(user_id: object, session_id: object, messages: object) -> str:
    """A deterministic request id for one capture."""
    digest = hashlib.sha256(json.dumps(
        {"user_id": user_id, "session_id": session_id, "messages": messages},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8", errors="surrogatepass")).hexdigest()
    return "agent-" + digest[:40]


def normalize_add(body: bytes, *, default_user_id: str) -> tuple[bytes, IngestNotes]:
    """Normalize one Add body.

    Args:
        body: The HTTP body as received.
        default_user_id: The user of a request that names none.

    Returns:
        The body to forward and what changed; the original bytes when it is not a
        JSON object with a ``messages`` array.
    """
    notes = IngestNotes()
    try:
        raw: object = json.loads(body)
    except (ValueError, RecursionError):
        return body, notes
    if not isinstance(raw, dict) or not isinstance(cast(dict[str, Any], raw).get("messages"),
                                                     list):
        return body, notes
    request = cast(dict[str, Any], raw)
    if request.get("user_id") in (None, ""):
        request["user_id"] = default_user_id
        notes.defaults.append("user_id")
    if request.get("session_id") in (None, ""):
        request["session_id"] = DEFAULT_SESSION_ID
        notes.defaults.append("session_id")
    messages: list[Any] = []
    for message in cast(list[Any], request["messages"]):
        if not isinstance(message, dict):
            messages.append(message)
            continue
        fields = dict(cast(dict[str, Any], message))
        if fields.get("role") in DROPPED_ROLES:
            notes.dropped_messages += 1
            continue
        if "content" in fields:
            content = _text(fields["content"])
            if isinstance(content, str):
                content, count = redact_credentials(content)
                notes.redacted_spans += count
            fields["content"] = content
        if "timestamp" in fields:
            fields["timestamp"] = _timestamp(fields["timestamp"], notes)
        messages.append(fields)
    request["messages"] = messages
    if request.get("request_id") in (None, ""):
        request["request_id"] = derive_request_id(
            request["user_id"], request["session_id"], messages,
        )
        notes.derived_request_id = True
    return json.dumps(request).encode("ascii"), notes


def normalize_search(body: bytes, *, default_user_id: str, default_top_k: int) -> bytes:
    """Fill a Search body's missing ``user_id`` and ``top_k``; other bodies are unchanged."""
    try:
        raw: object = json.loads(body)
    except (ValueError, RecursionError):
        return body
    if not isinstance(raw, dict):
        return body
    request = cast(dict[str, Any], raw)
    if "user_id" in request and "top_k" in request:
        return body
    request.setdefault("user_id", default_user_id)
    request.setdefault("top_k", default_top_k)
    return json.dumps(request).encode("ascii")
