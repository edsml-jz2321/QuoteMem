"""Best-effort parsers for structured text and LLM responses."""

from __future__ import annotations

__all__ = ["parse_keys", "parse_md"]

import re
from collections.abc import Sequence
from typing import Literal, overload

from .values import raise_mismatch

_MODE_VALUES = ("list", "dict")
_AUTO_KEY_PATTERN = re.compile(r"^[ \t]*(?P<key>[\w.-]+)[ \t]*:", re.MULTILINE)
_OPEN_TAG_PATTERN = re.compile(r"<(?P<tag>[^\W\d_][\w.:-]*)>")


@overload
def parse_keys(response: str, keys: Sequence[str] | None = None, *, mode: Literal["dict"] = "dict") -> dict[str, str | None]:
    """Parse key/value blocks in dictionary mode.

    Args:
        response (str): Text containing line-oriented key/value blocks.
        keys (Sequence[str] | None): Optional exact keys to recognize.
        mode (Literal["dict"]): Dictionary output mode.

    Returns:
        dict[str, str | None]: Parsed key/value mapping.
    """
    ...


@overload
def parse_keys(response: str, keys: Sequence[str] | None = None, *, mode: Literal["list"]) -> list[dict[str, str]]:
    """Parse key/value blocks in list mode.

    Args:
        response (str): Text containing line-oriented key/value blocks.
        keys (Sequence[str] | None): Optional exact keys to recognize.
        mode (Literal["list"]): List output mode preserving duplicate blocks.

    Returns:
        list[dict[str, str]]: Ordered key/value block records.
    """
    ...


@overload
def parse_keys(response: str, keys: Sequence[str] | None, mode: Literal["dict"]) -> dict[str, str | None]:
    """Parse key/value blocks in dictionary mode.

    Args:
        response (str): Text containing line-oriented key/value blocks.
        keys (Sequence[str] | None): Optional exact keys to recognize.
        mode (Literal["dict"]): Dictionary output mode.

    Returns:
        dict[str, str | None]: Parsed key/value mapping.
    """
    ...


@overload
def parse_keys(response: str, keys: Sequence[str] | None, mode: Literal["list"]) -> list[dict[str, str]]:
    """Parse key/value blocks in list mode.

    Args:
        response (str): Text containing line-oriented key/value blocks.
        keys (Sequence[str] | None): Optional exact keys to recognize.
        mode (Literal["list"]): List output mode preserving duplicate blocks.

    Returns:
        list[dict[str, str]]: Ordered key/value block records.
    """
    ...


def parse_keys(
    response: str,
    keys: Sequence[str] | None = None,
    mode: Literal["list", "dict"] = "dict",
) -> list[dict[str, str]] | dict[str, str | None]:
    """Parse line-oriented ``key: value`` blocks from a text response.

    Values continue across lines until the next recognized key. When ``keys``
    is omitted, identifiers containing letters, digits, underscores, dots, or
    hyphens are discovered at the start of lines. In dictionary mode, requested
    but missing keys map to ``None`` and later duplicate keys overwrite earlier
    values.

    Args:
        response: Text containing line-oriented key/value blocks.
        keys: Optional exact keys to recognize. Their order is preserved in
            dictionary mode.
        mode: Output shape. ``"dict"`` returns a mapping and ``"list"`` keeps
            ordered duplicate blocks.

    Returns:
        Parsed key/value blocks in the requested output shape.

    Raises:
        ValueError: If ``mode`` is not ``"dict"`` or ``"list"``.

    Examples:
        >>> parse_keys("name: Ada\\nnotes: first\\n  second")
        {'name': 'Ada', 'notes': 'first\\n  second'}
        >>> parse_keys("name: Ada", keys=["name", "age"])
        {'name': 'Ada', 'age': None}
    """
    if mode not in _MODE_VALUES:
        raise_mismatch(_MODE_VALUES, mode, name="mode")

    requested = None if keys is None else list(dict.fromkeys(str(key) for key in keys))
    occurrences: list[tuple[int, int, str]] = []
    if requested is None:
        occurrences.extend((match.start(), match.end(), match.group("key")) for match in _AUTO_KEY_PATTERN.finditer(response))
    else:
        for key in requested:
            pattern = re.compile(rf"^[ \t]*{re.escape(key)}[ \t]*:", re.MULTILINE)
            occurrences.extend((match.start(), match.end(), key) for match in pattern.finditer(response))
    occurrences.sort(key=lambda occurrence: occurrence[0])

    blocks: list[dict[str, str]] = []
    for index, (_, value_start, key) in enumerate(occurrences):
        value_end = occurrences[index + 1][0] if index + 1 < len(occurrences) else len(response)
        blocks.append({"key": key, "value": response[value_start:value_end].strip()})

    if mode == "list":
        return blocks

    parsed = dict.fromkeys(requested or (block["key"] for block in blocks))
    for block in blocks:
        parsed[block["key"]] = block["value"]
    return parsed


@overload
def parse_md(response: str, recurse: bool = False, *, mode: Literal["dict"] = "dict") -> dict[str, str]:
    """Parse Markdown-like blocks in dictionary mode.

    Args:
        response (str): Markdown-like text to parse.
        recurse (bool): Whether to recursively parse nested tag content.
        mode (Literal["dict"]): Flattened dictionary output mode.

    Returns:
        dict[str, str]: Parsed block mapping.
    """
    ...


@overload
def parse_md(response: str, recurse: bool = False, *, mode: Literal["list"]) -> list[dict[str, object]]:
    """Parse Markdown-like blocks in list mode.

    Args:
        response (str): Markdown-like text to parse.
        recurse (bool): Whether to recursively parse nested tag content.
        mode (Literal["list"]): Ordered block list output mode.

    Returns:
        list[dict[str, object]]: Parsed block records in source order.
    """
    ...


@overload
def parse_md(response: str, recurse: bool, mode: Literal["dict"]) -> dict[str, str]:
    """Parse Markdown-like blocks in dictionary mode.

    Args:
        response (str): Markdown-like text to parse.
        recurse (bool): Whether to recursively parse nested tag content.
        mode (Literal["dict"]): Flattened dictionary output mode.

    Returns:
        dict[str, str]: Parsed block mapping.
    """
    ...


@overload
def parse_md(response: str, recurse: bool, mode: Literal["list"]) -> list[dict[str, object]]:
    """Parse Markdown-like blocks in list mode.

    Args:
        response (str): Markdown-like text to parse.
        recurse (bool): Whether to recursively parse nested tag content.
        mode (Literal["list"]): Ordered block list output mode.

    Returns:
        list[dict[str, object]]: Parsed block records in source order.
    """
    ...


def parse_md(
    response: str,
    recurse: bool = False,
    mode: Literal["list", "dict"] = "dict",
) -> list[dict[str, object]] | dict[str, str]:
    """Parse tagged, fenced, and plain-text blocks from Markdown-like text.

    XML-style tags without attributes become named blocks, fenced code blocks
    use their language as the key, and remaining content uses ``"text"``.
    Variable-length backtick fences and incomplete streaming output are
    supported. Recursive dictionary mode flattens nested blocks with dotted
    keys; later duplicate keys overwrite earlier values.

    Args:
        response: Markdown-like text to parse.
        recurse: Whether to recursively parse content inside XML-style tags.
        mode: Output shape. ``"dict"`` returns a flattened mapping and
            ``"list"`` preserves block order and duplicates.

    Returns:
        Parsed blocks in the requested output shape.

    Raises:
        ValueError: If ``mode`` is not ``"dict"`` or ``"list"``.

    Examples:
        >>> parse_md("<score>10</score>")
        {'score': '10'}
        >>> parse_md("text\\n```json\\n{\\"ok\\": true}\\n```")
        {'text': 'text', 'json': '{"ok": true}'}
    """
    if mode not in _MODE_VALUES:
        raise_mismatch(_MODE_VALUES, mode, name="mode")

    blocks = _parse_md_blocks(response, recurse=recurse)
    if mode == "list":
        return blocks

    parsed: dict[str, str] = {}

    def flatten(items: list[dict[str, object]], prefix: tuple[str, ...] = ()) -> None:
        for block in items:
            key = str(block["key"])
            value = block["value"]
            path = prefix + (key,)
            if isinstance(value, list):
                flatten(value, path)
            else:
                parsed[".".join(path)] = str(value)

    flatten(blocks)
    return parsed


def _parse_md_blocks(response: str, *, recurse: bool) -> list[dict[str, object]]:
    """Return ordered Markdown-like blocks while accepting incomplete input."""
    blocks: list[dict[str, object]] = []
    text: list[str] = []
    position = 0

    def flush_text() -> None:
        content = "".join(text).strip()
        if content:
            blocks.append({"key": "text", "value": content})
        text.clear()

    while position < len(response):
        tag_match = _OPEN_TAG_PATTERN.match(response, position)
        if tag_match is not None:
            tag = tag_match.group("tag")
            content_start = tag_match.end()
            close_start = _find_matching_close_tag(response, tag, content_start)
            flush_text()
            if close_start < 0:
                content = response[content_start:].strip()
                position = len(response)
            else:
                content = response[content_start:close_start].strip()
                position = close_start + len(tag) + 3
            value: object = _parse_md_blocks(content, recurse=True) if recurse and content else content
            blocks.append({"key": tag, "value": value})
            continue

        if response[position] == "`":
            fence_end = position
            while fence_end < len(response) and response[fence_end] == "`":
                fence_end += 1
            fence = response[position:fence_end]
            if len(fence) >= 3:
                line_end = response.find("\n", fence_end)
                if line_end < 0:
                    line_end = len(response)
                language = response[fence_end:line_end].strip() or "markdown"
                content_start = min(line_end + 1, len(response))
                close_start = _find_code_fence_close(response, fence, content_start)
                flush_text()
                if close_start < 0:
                    content = response[content_start:].strip()
                    position = len(response)
                else:
                    content = response[content_start:close_start].strip()
                    position = close_start + len(fence)
                    while position < len(response) and response[position] in " \t":
                        position += 1
                    if position < len(response) and response[position] == "\n":
                        position += 1
                blocks.append({"key": language, "value": content})
                continue

        text.append(response[position])
        position += 1

    flush_text()
    return blocks


def _find_matching_close_tag(response: str, tag: str, start: int) -> int:
    """Return the matching close position, including nested equal-name tags."""
    open_pattern = re.compile(rf"<{re.escape(tag)}>")
    close_pattern = re.compile(rf"</{re.escape(tag)}>")
    depth = 1
    position = start

    while position < len(response):
        open_match = open_pattern.search(response, position)
        close_match = close_pattern.search(response, position)
        if close_match is None:
            return -1
        if open_match is None or close_match.start() < open_match.start():
            depth -= 1
            if depth == 0:
                return close_match.start()
            position = close_match.end()
        else:
            depth += 1
            position = open_match.end()
    return -1


def _find_code_fence_close(response: str, fence: str, start: int) -> int:
    """Return the next exact fence that starts a line."""
    position = start
    while position < len(response):
        match_start = response.find(fence, position)
        if match_start < 0:
            return -1
        match_end = match_start + len(fence)
        starts_line = match_start == 0 or response[match_start - 1] == "\n"
        exact_length = match_end == len(response) or response[match_end] != "`"
        if starts_line and exact_length:
            return match_start
        position = match_start + 1
    return -1
