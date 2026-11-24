"""Built-in HeavenBase CLI command groups."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["config_group", "llm_group", "mcp_group", "prompt_group", "root_commands", "ws_group"]

_LAZY_EXPORTS = {
    "config_group": (".config", "config_group"),
    "llm_group": (".llm", "llm_group"),
    "mcp_group": (".mcp", "mcp_group"),
    "prompt_group": (".prompt", "prompt_group"),
    "root_commands": (".root", "root_commands"),
    "ws_group": (".ws", "ws_group"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
