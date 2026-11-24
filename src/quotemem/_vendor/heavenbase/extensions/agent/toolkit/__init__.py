"""Agent extension MCP toolkit family."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["AGENT_SKILL_TOOLS", "agent_tool_functions"]

_LAZY_EXPORTS = {
    "AGENT_SKILL_TOOLS": (".profile", "AGENT_SKILL_TOOLS"),
    "agent_tool_functions": (".factory", "agent_tool_functions"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
