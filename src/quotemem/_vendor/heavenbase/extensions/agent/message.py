"""Persistent agent message entity."""

from __future__ import annotations

__all__ = ["Message", "message_content_text", "message_tool_call_count"]

from heavenbase.entity import Entity, field
from heavenbase.types import Identifier, Integer, Json, LongText, MediumText, ShortText
from heavenbase.utils import Any, Dict, Iterable, List, Mapping, deepcopy, hash_id

from ._workspace import resolve_agent_workspace


def message_content_text(payload: Mapping[str, Any]) -> str:
    """Return a compact text projection from an OpenAI-format message.

    Args:
        payload (Mapping[str, Any]): OpenAI-format message dictionary.

    Returns:
        str: Text content extracted from ``content`` blocks or strings.
    """
    content = payload.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def message_tool_call_count(payload: Mapping[str, Any]) -> int:
    """Return the number of assistant tool calls embedded in a message.

    Args:
        payload (Mapping[str, Any]): OpenAI-format message dictionary.

    Returns:
        int: Number of entries in ``tool_calls``, or ``0`` when absent.
    """
    calls = payload.get("tool_calls")
    return len(calls) if isinstance(calls, list) else 0


class Message(Entity):
    """OpenAI-format message payload with small query projections."""

    identifier = "agent-message"
    __derive__ = [
        Entity.derive("catalog.desc").compute(
            lambda content_text: None if content_text is None else content_text[: MediumText.default_max_chars],
            inputs=["content_text"],
        ),
        Entity.derive("catalog.type").compute(lambda: "message"),
        Entity.derive("catalog.statistics").compute(
            lambda tool_call_count: {"tool_call_count": tool_call_count},
            inputs=["tool_call_count"],
        ),
    ]

    session_id = field(Identifier).default("").desc("Owning agent-session id")
    idx = field(Integer).default(0).desc("Ordered message index in the session")
    role = field(ShortText).default("user").desc("OpenAI message role")
    content_text = field(LongText).default("").desc("Text projection of message content")
    tool_call_count = field(Integer).default(0).desc("Number of tool calls in payload")
    payload = field(Json).default({}).desc("Full OpenAI-format message payload")

    def __init__(self, **values: Any) -> None:
        """Construct one persistent agent message entity row.

        Args:
            **values (Any): Message field values accepted by ``Entity``.

        Returns:
            None: This initializer does not return a value.
        """
        super().__init__(**values)
        for key, value in self._data.items():
            setattr(self, key, value)

    @classmethod
    def wrap(cls, payload: Mapping[str, Any], session_id: str = "", idx: int = 0) -> "Message":
        """Build a message row from one OpenAI-format payload.

        Args:
            payload: OpenAI-format message dictionary.
            session_id: Owning ``agent-session`` id. Defaults to ``""``.
            idx: Ordered position within the session. Defaults to ``0``.

        Returns:
            Message: Message entity with the full payload and bounded text
            and Catalog projections.
        """
        data = deepcopy(dict(payload))
        row_id = str(data.get("object_id") or hash_id("message", session_id or "", int(idx), data))
        return cls(
            object_id=row_id,
            session_id=session_id or "",
            idx=int(idx),
            role=str(data.get("role") or "user"),
            content_text=message_content_text(data)[: LongText.default_max_chars],
            tool_call_count=message_tool_call_count(data),
            payload=data,
        )

    @staticmethod
    def payloads(messages: Iterable["Message | Mapping[str, Any]"]) -> List[Dict[str, Any]]:
        """Return OpenAI-format payloads from message rows or mappings.

        Args:
            messages: Message objects or row dictionaries.

        Returns:
            List[Dict[str, Any]]: Payload dictionaries in input order.
        """
        out = []
        for message in messages:
            data = message.to_dict() if isinstance(message, Message) else dict(message)
            out.append(deepcopy(dict(data.get("payload") or data)))
        return out

    @property
    def payload_dict(self) -> Dict[str, Any]:
        """Return a defensive copy of the OpenAI-format payload."""
        return deepcopy(dict(self._data.get("payload") or {}))

    @classmethod
    def load(cls, object_id: str, ws: Any = None) -> "Message":
        """Load one persistent message row by ``object_id``.

        Args:
            object_id: Message row id.
            ws: Workspace used for storage. Defaults to the active workspace.

        Returns:
            Message: Loaded message entity.

        Raises:
            KeyError: If the message does not exist.
        """
        workspace = resolve_agent_workspace(ws)
        row = workspace.get(object_id, entity=cls)
        if row is None:
            raise KeyError(f"No agent message found for {object_id!r}")
        return cls.from_dict(row)
