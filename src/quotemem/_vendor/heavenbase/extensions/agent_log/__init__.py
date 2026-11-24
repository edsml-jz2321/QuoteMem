"""Optional assistant session-log extension.

The ``agent-log`` extension owns the ``agent-trace`` event entity
(:class:`AgentTrace`) that the dashboard assistant session log writes. It is
not a default extension: a workspace gains the ``agent-trace`` table only when
something enables ``agent-log`` explicitly, which the session log does for its
own log workspace. It requires the ``agent`` extension, whose ``Session`` and
``Message`` rows hold the log's readable projection.
"""

from __future__ import annotations

from heavenbase._lazy import _attach_registry_exports
from heavenbase.extensions.ids import AGENT_LOG_EXTENSION_ID

__all__ = [
    "AGENT_LOG_EXTENSION_ID",
    "AgentTrace",
]

_REGISTRY_ENTITY_EXPORTS = {
    "AgentTrace": "agent-trace",
}

__getattr__, __dir__ = _attach_registry_exports(
    globals(),
    __all__,
    entities=_REGISTRY_ENTITY_EXPORTS,
)
