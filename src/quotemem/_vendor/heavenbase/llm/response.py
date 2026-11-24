"""Response containers with include-based projection."""

from __future__ import annotations

__all__ = ["LLMResponse"]

from time import time

from heavenbase.utils import raise_mismatch
from heavenbase.utils.typing import Any, Dict, Iterable, List, Optional, Union, dataclass, field

_Include = Optional[Union[str, Iterable[str]]]


def _message_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _endswith(items: List[Dict[str, Any]], suffix: List[Dict[str, Any]]) -> bool:
    return bool(suffix) and len(items) >= len(suffix) and items[-len(suffix) :] == suffix


def _delta_tool_calls(delta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for message in delta:
        calls.extend(dict(call) for call in message.get("tool_calls") or [] if isinstance(call, dict))
    return calls


def _include_list(include: _Include, default: Iterable[str]) -> List[str]:
    if include is None:
        return list(default)
    if isinstance(include, str):
        return [include]
    return list(include)


def _project(data: Dict[str, Any], include: _Include = None, default: Iterable[str] = ("text",), reduce: bool = True) -> Any:
    fields = _include_list(include, default)
    for name in fields:
        if name not in data:
            raise_mismatch(data.keys(), name, name="include")
    out = {name: data.get(name) for name in fields}
    return out[fields[0]] if reduce and len(fields) == 1 else out


@dataclass
class LLMResponse:
    """Represent one normalized chat response and its projection fields.

    Args:
        text (str): Assistant answer text without reasoning markup.
        think (str): Normalized reasoning text.
        tool_calls (List[Dict[str, Any]]): OpenAI-compatible tool calls.
        message (Dict[str, Any]): Assistant message for conversation history.
        usage (Dict[str, Any]): Normalized provider usage values.
        raw (Any): Original provider response payload.
        elapsed (float): Request duration in seconds.
        created_at (float): Response creation time as Unix seconds.
        messages (List[Dict[str, Any]]): Full normalized conversation history.
        delta (List[Dict[str, Any]]): Messages added by this inference call.
        structured (Any): Parsed structured response value when available.
        tool_turns (int): Executable-tool rounds this response actually ran.
        max_tool_turns (int | None): Round budget the executable-tool loop was given.
            `None` means no executable-tool loop ran, so no budget applies.
        truncated (bool): Whether the executable-tool loop stopped on its budget while
            the model was still calling tools, leaving this answer mid-investigation.
            A budget that was merely reached is not truncation, so this flag cannot be
            derived from `tool_turns` and `max_tool_turns`; it is recorded at the exit.
        content_override (str | None): Explicit combined content loaded from storage.
    """

    text: str = ""
    think: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    message: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    elapsed: float = 0.0
    created_at: float = field(default_factory=time)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    delta: List[Dict[str, Any]] = field(default_factory=list)
    structured: Any = None
    tool_turns: int = 0
    max_tool_turns: Optional[int] = None
    truncated: bool = False
    content_override: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMResponse":
        """Build a response from normalized fields.

        Args:
            data (Dict[str, Any]): Normalized response fields. A missing `delta`
                is derived from `message`, and duplicate history suffixes are avoided.

        Returns:
            LLMResponse: Hydrated response value.
        """
        message = dict(data.get("message") or {})
        raw_delta = data.get("delta")
        delta = _message_list(raw_delta) if isinstance(raw_delta, list) else ([message] if message else [])
        messages = list(data.get("messages") or [])
        if delta and not _endswith(messages, delta):
            messages = [*messages, *delta]
        response = cls(
            text=data.get("text") or "",
            think=data.get("think") or "",
            tool_calls=list(data.get("tool_calls") or []),
            message=message,
            usage=dict(data.get("usage") or {}),
            raw=data.get("raw"),
            elapsed=float(data.get("elapsed") or 0.0),
            created_at=float(data.get("created_at") or time()),
            messages=messages,
            delta=delta,
            structured=data.get("structured"),
            tool_turns=int(data.get("tool_turns") or 0),
            max_tool_turns=None if data.get("max_tool_turns") is None else int(data["max_tool_turns"]),
            truncated=bool(data.get("truncated")),
        )
        content = data.get("content")
        if content is not None and content != response.content:
            response.content_override = content
        return response

    @property
    def content(self) -> str:
        """Return text with reasoning wrapped when reasoning content exists."""
        if self.content_override is not None:
            return self.content_override
        return f"<think>\n{self.think}\n</think>\n{self.text}" if self.think else self.text

    def to_dict(self) -> Dict[str, Any]:
        """Return all normalized response fields.

        Args:
            None.

        Returns:
            Dict[str, Any]: Independent top-level response dictionary.
        """
        message = self.message or {"role": "assistant", "content": self.text}
        delta = list(self.delta)
        messages = list(self.messages)
        if delta and not _endswith(messages, delta):
            messages = [*messages, *delta]
        tool_calls = self.tool_calls or _delta_tool_calls(delta)
        return {
            "text": self.text,
            "think": self.think,
            "content": self.content,
            "tool_calls": tool_calls,
            "message": message,
            "usage": self.usage,
            "raw": self.raw,
            "elapsed": self.elapsed,
            "created_at": self.created_at,
            "messages": messages,
            "delta": delta,
            "structured": self.structured,
            "tool_turns": self.tool_turns,
            "max_tool_turns": self.max_tool_turns,
            "truncated": self.truncated,
        }

    def get(self, name: str, default: Any = None) -> Any:
        """Read one top-level response field.

        Args:
            name (str): Field name returned by `to_dict`.
            default (Any): Value returned when the field is absent.

        Returns:
            Any: Response field value or `default`.
        """
        return self.to_dict().get(name, default)

    def include(self, include: _Include = None, reduce: bool = True) -> Any:
        """Project the response to requested fields.

        Use this method to keep large provider payloads out of call sites while still allowing full
        inspection when needed. Supported include fields are the keys returned by `to_dict`:
        `text`, `think`, `content`, `tool_calls`, `message`, `usage`, `raw`, `elapsed`, `created_at`,
        `delta`, `messages`, `structured`, `tool_turns`, `max_tool_turns`, and `truncated`. `message`
        is the assistant response dictionary in OpenAI chat format. `delta` is the list of new messages
        produced by this inference call, starting with `message` and followed by tool-result messages
        when tool execution is available. `messages` is the full conversation history: normalized input
        messages plus `delta`. `truncated` says the executable-tool loop ran out of rounds mid-answer,
        with `tool_turns` and `max_tool_turns` as its evidence. Passing any other name raises through
        `raise_mismatch`.

        Args:
            include (str | Iterable[str] | None): Optional field or field list to return.
            reduce (bool): Whether a one-field include returns the field value directly.

        Returns:
            Any: The selected value when reduced, otherwise a dictionary keyed by included field names.
        """
        return _project(self.to_dict(), include=include, default=("text",), reduce=reduce)
