"""Assistant session-log trace-event entity."""

from __future__ import annotations

__all__ = ["AgentTrace"]

from heavenbase.entity import Entity, field
from heavenbase.types import Identifier, Integer, Json, ShortText
from heavenbase.utils import Any, Mapping, hash_id


class AgentTrace(Entity):
    """One immutable provider-call event of a logged assistant conversation.

    Trace rows are the authoritative record of what a model actually saw:
    ``request`` rows persist the canonical provider-bound request before
    provider I/O, ``response``/``error`` rows close them, and one ``turn``
    row summarizes each completed assistant turn. Rows have deterministic ids
    and are written once, so a ``request`` row without its terminal sibling is
    itself the record of an interrupted call. After a row is written, the
    recorder changes stored rows in exactly two ways: a retry's successful ``turn`` row overwrites the
    provisional row of the errored or interrupted attempt it supersedes, and
    the retention pass run when admitting a new conversation deletes every
    row of each conversation it evicts. Retention never deletes the trace of a
    conversation whose session row is missing. ``Session``/``Message`` rows
    are a readable projection derived from these events, never the other way
    around.

    The entity belongs to the optional ``agent-log`` extension, so a workspace
    only stores these rows after ``ws.enable_extension("agent-log")``.
    """

    identifier = "agent-trace"
    __derive__ = [
        Entity.derive("catalog.type").compute(lambda: "trace"),
        Entity.derive("catalog.desc").compute(
            lambda kind, call, seq: f"{kind} {call} #{seq}".replace("  ", " "),
            inputs=["kind", "call", "seq"],
        ),
    ]

    conversation_id = field(Identifier).default("").desc("Logged conversation this event belongs to")
    turn_id = field(ShortText).default("").desc("Client-supplied turn identity; retries reuse it, making writes idempotent")
    seq = field(Integer).default(0).desc("Provider-call order within an attempt, or seal order within a conversation for turn rows")
    kind = field(ShortText).default("").desc("Event kind: request | response | error | turn")
    call = field(ShortText).default("").desc("Call origin: round | turn_compaction | replay_compaction; empty for turn rows")
    ws_id = field(ShortText).default("").desc("Workspace the assistant turn served")
    payload = field(Json).default({}).desc("Canonical request, terminal data, or turn summary; absent keys were never captured")

    @classmethod
    def event_id(cls, conversation_id: str, turn_id: str, seq: int, kind: str) -> str:
        """Return the deterministic row id of one trace event.

        Args:
            conversation_id (str): Logged conversation identity.
            turn_id (str): Turn identity within the conversation.
            seq (int): Event sequence number. For ``turn`` rows the id uses a
                fixed sequence so it stays stable across retries.
            kind (str): Event kind (``request``/``response``/``error``/``turn``).

        Returns:
            str: Stable ``object_id``: the same event coordinates always map
            to the same row, so a retried write targets the existing row
            instead of adding a duplicate. ``upsert(..., overwrite=False)``
            of an existing event raises ``ValueError``; the recorder treats
            that error as an idempotent replay.
        """
        return hash_id("agent-trace", conversation_id, turn_id, 0 if kind == "turn" else int(seq), kind)

    @classmethod
    def event(
        cls, conversation_id: str, turn_id: str, seq: int, kind: str, *, call: str = "", ws_id: str = "", payload: Mapping[str, Any] | None = None
    ) -> dict:
        """Return one complete trace-event row ready for ``upsert``.

        Args:
            conversation_id (str): Logged conversation identity.
            turn_id (str): Turn identity within the conversation.
            seq (int): Event sequence number, or conversation seal order for
                ``turn`` rows.
            kind (str): Event kind (``request``/``response``/``error``/``turn``).
            call (str): Call origin tag. Defaults to ``""``.
            ws_id (str): Workspace the turn served. Defaults to ``""``.
            payload (Mapping[str, Any] | None): Event payload. Defaults to ``{}``.

        Returns:
            dict: Row dictionary with its deterministic ``object_id``.
        """
        return {
            "object_id": cls.event_id(conversation_id, turn_id, seq, kind),
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "seq": int(seq),
            "kind": kind,
            "call": call,
            "ws_id": ws_id,
            "payload": dict(payload or {}),
        }
