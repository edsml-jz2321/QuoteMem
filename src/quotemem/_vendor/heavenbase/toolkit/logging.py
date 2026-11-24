"""Opt-in MCP tool-call logging into Agent Session rows."""

from __future__ import annotations

__all__ = [
    "attach_mcp_call_logging",
    "mcp_call_logging_enabled",
    "open_mcp_call_log",
]

from heavenbase.utils import Any, Mapping, dumps_jstr
from heavenbase.utils.log import get_logger
from heavenbase.utils.redact import redact_data, redact_secrets

from heavenbase.extensions.contributions import eligible_contributions
from heavenbase.modules import ModuleResolver, module_resolver
from .toolkit import Toolkit

_logger = get_logger(__name__)


def _args_summary(kwargs: Mapping[str, Any]) -> str:
    return redact_secrets(dumps_jstr(redact_data(dict(kwargs)), compact=True))


def _result_size(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, str):
        return len(result)
    try:
        return len(dumps_jstr(result, compact=True))
    except Exception:
        return len(str(result))


def mcp_call_logging_enabled(*, config: Any | None = None) -> bool:
    """Return whether MCP Toolkit builds should persist call-log rows.

    Args:
        config: Context-owned config authority. When omitted, the process
            default is selected.

    Returns:
        bool: Whether MCP call logging is enabled.
    """
    from heavenbase.context.authority import omitted_config

    manager = omitted_config() if config is None else config
    return bool(manager.get("heavenbase.mcp.log_calls", default=False))


def open_mcp_call_log(
    ws: Any,
    *,
    toolkit_name: str,
    profile: str = "",
    resolver: ModuleResolver | None = None,
) -> Any:
    """Open one optional Toolkit-exposure call log from its registered provider.

    The returned object may be an Agent Session because that domain owns
    persisted message rows. It is not an MCP protocol session.

    Args:
        ws: Workspace that owns the session rows.
        toolkit_name: Runtime toolkit name.
        profile: Resolved MCP profile name.
        resolver: Context-owned module resolver. When omitted, the workspace's
            resolver is selected.

    Returns:
        Any: Provider call-log owner, or `None` when no provider is registered.

    Raises:
        TypeError: If the provider record does not resolve to a callable.
    """
    chosen = module_resolver(resolver if resolver is not None else ws.module_resolver)
    rows = eligible_contributions(chosen.inspect("mcp_call_logger", "agent"), ws)
    if not rows:
        return None
    opener = chosen.resolve("mcp_call_logger", "agent")
    if not callable(opener):
        raise TypeError("MCP call logger 'agent' must resolve to a callable")
    return opener(ws, toolkit_name=toolkit_name, profile=profile)


def _log_tool_call(
    session: Any,
    ws: Any,
    *,
    tool_name: str,
    kwargs: Mapping[str, Any],
    result: Any,
    duration_s: float,
    error: bool,
    cached: bool,
) -> None:
    if session is None or not hasattr(session, "append"):
        return
    payload = {
        "role": "tool",
        "name": tool_name,
        "content": dumps_jstr(
            {
                "args": _args_summary(kwargs),
                "result_size": _result_size(result),
                "duration_ms": round(duration_s * 1000.0, 3),
                "error": bool(error),
                "cached": bool(cached),
            },
            compact=True,
        ),
    }
    session.append(payload, ws=ws)


def _safe_log_tool_call(session: Any, ws: Any, **fields: Any) -> None:
    try:
        _log_tool_call(session, ws, **fields)
    except Exception as exc:
        _logger.warning("MCP tool-call logging failed for %r: %s", fields.get("tool_name"), exc)


def attach_mcp_call_logging(toolkit: Toolkit, ws: Any, log: Any) -> Toolkit:
    """Attach MCP call logging to Toolkit execution without replacing methods.

    Logging failures are isolated and never change tool-call outcomes. The
    hook is idempotent per toolkit instance.

    Args:
        toolkit: Workspace MCP toolkit to instrument.
        ws: Workspace used for agent row persistence.
        log: Open call-log owner from the registered optional provider.

    Returns:
        Toolkit: The same toolkit instance with wrapped ``run``/``arun`` methods.
    """
    if getattr(toolkit, "_mcp_call_logging_attached", False):
        return toolkit
    if log is None:
        return toolkit

    def _logger_hook(**fields: Any) -> None:
        _safe_log_tool_call(log, ws, **fields)

    toolkit._add_call_hook(_logger_hook)
    toolkit._mcp_call_log_id = log.object_id  # type: ignore[attr-defined]
    toolkit._mcp_call_logging_attached = True  # type: ignore[attr-defined]
    toolkit._mcp_call_logging_hook = _logger_hook  # type: ignore[attr-defined]
    return toolkit
