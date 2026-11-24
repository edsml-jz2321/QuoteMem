"""Optional persistent agent extension."""

from __future__ import annotations

from heavenbase._lazy import _attach_registry_exports
from heavenbase.extensions.ids import AGENT_EXTENSION_ID

__all__ = [
    "AGENT_EXTENSION_ID",
    "Agent",
    "AgentRuntime",
    "Message",
    "Session",
    "Skill",
    "agent_entities",
    "agent_entity_ids",
    "log_chat_session",
]

_REGISTRY_ENTITY_EXPORTS = {
    "Agent": "agent-agent",
    "Message": "agent-message",
    "Session": "agent-session",
    "Skill": "agent-skill",
}

_REGISTRY_SIBLING_EXPORTS = {
    "AgentRuntime": ("agent-agent", "agent", "AgentRuntime"),
    "agent_entities": ("agent-agent", "extension", "agent_entities"),
    "agent_entity_ids": ("agent-agent", "extension", "agent_entity_ids"),
    "log_chat_session": ("agent-agent", "session", "log_chat_session"),
}

__getattr__, __dir__ = _attach_registry_exports(
    globals(),
    __all__,
    entities=_REGISTRY_ENTITY_EXPORTS,
    siblings=_REGISTRY_SIBLING_EXPORTS,
)
