"""Value-free shape records of inbound HTTP requests, for checking what a caller sends.

A record names every JSON key path the body uses, with each path's value types,
occurrence count and sizes, plus the request's header names. It never holds a
value: no content, query, identifier, credential or header value. The only
derived facts are string lengths, array lengths, a count of strings that look
like a calendar date or time, and benchmark-name tokens: the dataset segment
of the documented ``eval:<run>:<dataset>:...`` identifier convention, and the
leading ``[<dataset>][session_...]`` tag of envelope-format message contents
(the platform hashes identifiers in practice, so the tag is the usable one).

The router writes one record per user-route POST when ``quotemem bench route`` runs with
``--shape-log``; it is off by default. A summary step can turn a
log into a table of fields sent against fields the HTTP models read.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Final, cast

MAX_PATHS: Final = 256
"""Distinct key paths kept per record; further paths are counted in ``paths_dropped``."""
MAX_KEY_CHARS: Final = 64
"""Longer object keys are recorded as ``<long-key>``, so a key cannot smuggle a value."""

_DATE_LIKE: Final = re.compile(
    r"\b(?:19|20)\d\d[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d\d\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+(?:19|20)\d\d\b"
    r"|\b(?:[01]?\d|2[0-3]):[0-5]\d\b",
    re.IGNORECASE,
)
_DATASET: Final = re.compile(r"eval:[^:]{1,128}:([A-Za-z0-9_.-]{1,48})(?::|\Z)")
_KEY: Final = re.compile(r"[A-Za-z0-9_.$@-]{1,64}")
_ENVELOPE: Final = re.compile(r"\[([A-Za-z0-9_.-]{1,48})\]\[session_")


def _type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _key(name: str) -> str:
    if len(name) > MAX_KEY_CHARS or not _KEY.fullmatch(name):
        return "<long-key>" if len(name) > MAX_KEY_CHARS else "<other-key>"
    return name


class _Shape:
    def __init__(self) -> None:
        self.paths: dict[str, dict[str, object]] = {}
        self.dropped = 0

    def add(self, path: str, value: object) -> None:
        stats = self.paths.get(path)
        if stats is None:
            if len(self.paths) >= MAX_PATHS:
                self.dropped += 1
                return
            stats = self.paths[path] = {"n": 0, "types": {}}
        stats["n"] = cast(int, stats["n"]) + 1
        types = cast(dict[str, int], stats["types"])
        kind = _type(value)
        types[kind] = types.get(kind, 0) + 1
        size: int | None = None
        if isinstance(value, str):
            size = len(value)
            if _DATE_LIKE.search(value):
                stats["date_like"] = cast(int, stats.get("date_like", 0)) + 1
        elif isinstance(value, list):
            size = len(cast(list[object], value))
        if size is not None:
            low, high, total = cast(list[int], stats.get("len", [size, size, 0]))
            stats["len"] = [min(low, size), max(high, size), total + size]

    def walk(self, value: object, path: str) -> None:
        self.add(path, value)
        if isinstance(value, dict):
            for name, item in cast(dict[str, object], value).items():
                self.walk(item, f"{path}.{_key(name)}" if path else _key(name))
        elif isinstance(value, list):
            for item in cast(list[object], value):
                self.walk(item, f"{path}[]")


def dataset_token(body: object) -> str | None:
    """The benchmark name in a platform ``eval:<run>:<dataset>:...`` identifier, if any.

    Args:
        body: The decoded request body.

    Returns:
        The dataset segment of ``request_id`` (else ``user_id``) when that
        identifier follows the documented convention, otherwise None.
    """
    if not isinstance(body, dict):
        return None
    fields = cast(dict[str, object], body)
    for name in ("request_id", "user_id"):
        value = fields.get(name)
        if isinstance(value, str):
            match = _DATASET.match(value)
            if match:
                return match.group(1)
    return None


def envelope_tags(body: object) -> dict[str, int]:
    """Count the ``[<dataset>][session_`` tags that open Add message contents.

    Args:
        body: The decoded request body.

    Returns:
        Tag to message count; empty for any other body or content format.
    """
    tags: dict[str, int] = {}
    messages = cast(dict[str, object], body).get("messages") if isinstance(body, dict) else None
    if isinstance(messages, list):
        for message in cast(list[object], messages):
            content = cast(dict[str, object], message).get("content") if isinstance(
                message, dict) else None
            if isinstance(content, str):
                match = _ENVELOPE.match(content)
                if match:
                    tags[match.group(1)] = tags.get(match.group(1), 0) + 1
    return tags


def request_shape(
    route: str, body: bytes, header_names: Iterable[str], status: int,
) -> dict[str, object]:
    """Describe one request without any of its values.

    Args:
        route: The request path, such as ``/add``.
        body: The raw request bytes.
        header_names: The request's header names; their values are never read.
        status: The status code answered to the caller.

    Returns:
        A JSON-ready record: route, status, body size, sorted lowercase header
        names, dataset token, envelope tags (Add only, when present), and
        per-path ``n``/``types``/``len``/``date_like``.
        A body that is not JSON records only ``"json": false``.
    """
    record: dict[str, object] = {
        "route": route, "status": status, "bytes": len(body),
        "headers": sorted({name.lower() for name in header_names}),
    }
    try:
        decoded: object = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        record["json"] = False
        return record
    shape = _Shape()
    try:
        shape.walk(decoded, "")
    except RecursionError:
        record["json"] = "too_deep"
        return record
    shape.paths.pop("", None)
    record["json"] = True
    record["top"] = _type(decoded)
    record["dataset"] = dataset_token(decoded)
    tags = envelope_tags(decoded)
    if tags:
        record["envelope"] = tags
    record["paths"] = shape.paths
    if shape.dropped:
        record["paths_dropped"] = shape.dropped
    return record
