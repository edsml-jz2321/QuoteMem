"""Persistent agent session entity."""

from __future__ import annotations

__all__ = ["Session", "log_chat_session", "open_mcp_call_log"]

from typing import TYPE_CHECKING
from uuid import uuid4

from heavenbase.entity import Entity, field
from heavenbase.types import Integer, Json, LongText, ShortText
from heavenbase.utils import Any, Dict, Iterable, List, Mapping, Optional, deepcopy, hash_id

from ._workspace import resolve_agent_workspace
from .message import Message

if TYPE_CHECKING:
    from heavenbase.llm.session import LLMSession


def _merge_usage(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> Dict[str, Any]:
    out = dict(left or {})
    for key, value in dict(right or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            current = out.get(key)
            # A provider may report a count as null; a later count replaces it.
            out[key] = current + value if isinstance(current, (int, float)) and not isinstance(current, bool) else value
        elif key not in out or out.get(key) in (None, ""):
            out[key] = value
    return out


def _turn_count(payloads: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for payload in payloads if payload.get("role") == "user")


def _tool_call_count(payloads: Iterable[Mapping[str, Any]]) -> int:
    return sum(Message.wrap(payload).tool_call_count for payload in payloads)


def _session_id(values: Mapping[str, Any]) -> str:
    """Return the given ``object_id``, or a fresh one: a random token keeps sessions created in one clock tick apart."""
    return str(values.get("object_id") or hash_id("session", values.get("title") or "", values.get("agent_ref") or "", uuid4().hex))


class Session(Entity):
    """Durable conversation state backed by ordered ``Message`` rows."""

    identifier = "agent-session"
    __derive__ = [
        Entity.derive("catalog.name").compute(lambda title: title, inputs=["title"]),
        Entity.derive("catalog.desc").compute(lambda summary: summary, inputs=["summary"]),
        Entity.derive("catalog.type").compute(lambda: "session"),
        Entity.derive("catalog.statistics").compute(
            lambda turn_count, tool_call_count, usage: {"turn_count": turn_count, "tool_call_count": tool_call_count, "usage": usage},
            inputs=["turn_count", "tool_call_count", "usage"],
        ),
    ]

    title = field(ShortText).default("").desc("Session title")
    summary = field(LongText).default("").desc("Short session summary")
    status = field(ShortText).default("active").desc("Session status")
    agent_ref = field(ShortText).default("").desc("Owning or source agent reference")
    message_ids = field(Json).default([]).desc("Ordered agent-message ids")
    turn_count = field(Integer).default(0).desc("Number of user turns")
    tool_call_count = field(Integer).default(0).desc("Number of assistant tool calls")
    state = field(Json).default({}).desc("Runtime and source metadata")
    usage = field(Json).default({}).desc("Aggregated LLM usage")

    def __init__(self, **values: Any) -> None:
        """Construct one persistent agent session entity row.

        Args:
            **values (Any): Session field values. ``ws`` selects the bound
                workspace.

        Returns:
            None: This initializer does not return a value.
        """
        self.ws = values.pop("ws", None)
        data = {
            "title": "",
            "summary": "",
            "status": "active",
            "agent_ref": "",
            "message_ids": [],
            "turn_count": 0,
            "tool_call_count": 0,
            "state": {},
            "usage": {},
            **values,
        }
        data["object_id"] = _session_id(data)
        data["message_ids"] = list(data.get("message_ids") or [])
        data["state"] = dict(data.get("state") or {})
        data["usage"] = dict(data.get("usage") or {})
        super().__init__(**data)
        self._sync_attrs()

    def _sync_attrs(self) -> None:
        for key, value in self._data.items():
            setattr(self, key, value)

    def _set(self, **values: Any) -> None:
        self._data.update(values)
        self._sync_attrs()

    def append(self, message: Mapping[str, Any], **meta: Any) -> Message:
        """Append and persist one OpenAI-format message.

        Args:
            message: OpenAI-format message payload.
            **meta: Optional metadata forwarded to ``append_messages``.

        Returns:
            Message: Written message row.
        """
        return self.append_messages([message], **meta)[0]

    def append_messages(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        source: str = "",
        usage: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        ws: Any = None,
    ) -> List[Message]:
        """Append OpenAI-format messages and update session aggregates.

        Args:
            messages: OpenAI-format payloads in execution order.
            source: Optional source label stored in session state.
            usage: Optional LLM usage dictionary to merge. Defaults to ``None``.
            state: Optional runtime metadata merged into ``state``. Defaults to ``None``.
            ws: Workspace used for storage. Defaults to this session's workspace or active workspace.

        Returns:
            List[Message]: Written message rows.
        """
        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        payloads = [deepcopy(dict(message)) for message in messages]
        start = len(self.message_ids)
        rows = [Message.wrap(payload, session_id=self.object_id, idx=start + offset) for offset, payload in enumerate(payloads)]
        if rows:
            workspace.upsert_many(Message, rows)
        next_state = dict(self.state or {})
        if source:
            next_state["source"] = source
        next_state.update(dict(state or {}))
        message_ids = [*list(self.message_ids or []), *[row.object_id for row in rows]]
        self._set(
            message_ids=message_ids,
            turn_count=int(self.turn_count or 0) + _turn_count(payloads),
            tool_call_count=int(self.tool_call_count or 0) + _tool_call_count(payloads),
            usage=_merge_usage(self.usage, usage),
            state=next_state,
        )
        workspace.upsert(type(self), self.to_dict())
        self.ws = workspace
        return rows

    def set_messages(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        source: str = "",
        usage: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        ws: Any = None,
    ) -> List[Message]:
        """Replace persisted message history and update session aggregates.

        Args:
            messages: OpenAI-format payloads in the retained session order.
            source: Optional source label stored in session state.
            usage: Optional complete usage dictionary for the retained history. Defaults to an empty aggregate.
            state: Optional runtime metadata merged into ``state``. Defaults to ``None``.
            ws: Workspace used for storage. Defaults to this session's workspace or active workspace.

        Returns:
            List[Message]: Written message rows matching ``messages``.
        """
        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        payloads = [deepcopy(dict(message)) for message in messages]
        rows = [Message.wrap(payload, session_id=self.object_id, idx=offset) for offset, payload in enumerate(payloads)]
        next_ids = [row.object_id for row in rows]
        next_id_set = set(next_ids)
        removed_ids = [message_id for message_id in list(self.message_ids or []) if message_id not in next_id_set]
        if removed_ids:
            workspace.delete_many([(Message, message_id) for message_id in removed_ids])
        if rows:
            workspace.upsert_many(Message, rows)
        next_state = dict(self.state or {})
        if source:
            next_state["source"] = source
        next_state.update(dict(state or {}))
        self._set(
            message_ids=next_ids,
            turn_count=_turn_count(payloads),
            tool_call_count=_tool_call_count(payloads),
            usage=deepcopy(dict(usage or {})),
            state=next_state,
        )
        workspace.upsert(type(self), self.to_dict())
        self.ws = workspace
        return rows

    def messages(self, ws: Any = None) -> List[Dict[str, Any]]:
        """Return ordered OpenAI-format message payloads.

        Args:
            ws: Workspace used for storage. Defaults to this session's workspace or active workspace.

        Returns:
            List[Dict[str, Any]]: OpenAI-format messages in session order.
        """
        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        rows = [row for row in workspace.get_many(self.message_ids, entity=Message) if row is not None]
        rows = sorted(rows, key=lambda row: int(row.get("idx") or 0))
        return Message.payloads(rows)

    def to_llm_session(self, llm: Any = None, ws: Any = None) -> LLMSession:
        """Hydrate this persistent session into the runtime ``LLMSession``.

        Args:
            llm: Optional LLM runtime. Defaults to a new ``LLM()`` through ``LLMSession``.
            ws: Workspace used to read messages. Defaults to this session's workspace.

        Returns:
            LLMSession: Runtime session with restored message history.
        """
        from heavenbase.llm import LLM
        from heavenbase.llm.session import LLMSession

        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        runtime = llm if llm is not None else LLM(context=workspace.context)
        return LLMSession(llm=runtime, messages=self.messages(ws=workspace))

    @classmethod
    def from_llm_session(cls, session: LLMSession, title: str = "", agent_ref: str = "", ws: Any = None) -> "Session":
        """Capture a runtime ``LLMSession`` into persistent rows.

        Args:
            session: Runtime session to capture.
            title: Optional persistent title. Defaults to ``""``.
            agent_ref: Optional source agent reference. Defaults to ``""``.
            ws: Workspace used for storage. Defaults to the active workspace.

        Returns:
            Session: Persistent session with copied messages.
        """
        out = cls(title=title, agent_ref=agent_ref, ws=ws)
        _upsert_session(out, ws=ws)
        out.append_messages(session.messages, source="LLMSession", ws=ws)
        return out

    @classmethod
    def load(cls, ref: str, ws: Any = None) -> "Session":
        """Load one persistent session by ``object_id``.

        Args:
            ref: Session ``object_id``.
            ws: Workspace used for storage. Defaults to the active workspace.

        Returns:
            Session: Loaded session.

        Raises:
            KeyError: If no session row matches.
        """
        workspace = resolve_agent_workspace(ws)
        row = workspace.get(ref, entity=cls)
        if row is None:
            raise KeyError(f"No agent session found for {ref!r}")
        return cls.from_dict(row, ws=workspace)


def open_mcp_call_log(ws: Any, *, toolkit_name: str, profile: str = "") -> "Session":
    """Open one Agent Session call log for an MCP Toolkit exposure.

    This Agent Session groups persisted call messages. It is not an MCP
    protocol session and does not carry request-local protocol state.

    Args:
        ws: Workspace with the ``agent`` extension enabled.
        toolkit_name: MCP Toolkit name stored on the call-log row.
        profile: Active MCP profile name stored in session state.

    Returns:
        Session: Persisted Agent Session used for subsequent call logging.
    """
    session = Session(title=toolkit_name, agent_ref=toolkit_name, ws=ws)
    session._set(
        state={
            "source": "mcp",
            "scope": "toolkit-exposure",
            "toolkit": toolkit_name,
            "profile": profile,
        }
    )
    _upsert_session(session, ws=ws)
    return session


def log_chat_session(
    messages: Iterable[Mapping[str, Any]],
    *,
    title: str = "",
    source: str = "",
    usage: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    ws: Any = None,
) -> Session:
    """Persist one completed chat/session trajectory.

    Args:
        messages: OpenAI-format messages to persist.
        title: Optional session title. Defaults to ``""``.
        source: Optional source label stored in state. Defaults to ``""``.
        usage: Optional usage dictionary. Defaults to ``None``.
        state: Optional metadata dictionary. Defaults to ``None``.
        ws: Workspace used for storage. Defaults to the active workspace.

    Returns:
        Session: Persistent session row.
    """
    payloads = [dict(message) for message in messages]
    session = Session(title=title, summary=_summary(payloads), ws=ws)
    _upsert_session(session, ws=ws)
    session.append_messages(payloads, source=source, usage=usage, state=state, ws=ws)
    return session


def _summary(messages: Iterable[Mapping[str, Any]]) -> str:
    for message in messages:
        text = Message.wrap(message).content_text.strip()
        if text:
            return text[:512]
    return ""


def _upsert_session(session: Session, ws: Any = None, overwrite: bool = True) -> str:
    """Persist one Session row through the owning workspace CRUD API."""
    workspace = resolve_agent_workspace(ws if ws is not None else session.ws)
    row_id = workspace.upsert(Session, session.to_dict(), overwrite=overwrite)
    session.ws = workspace
    return row_id
