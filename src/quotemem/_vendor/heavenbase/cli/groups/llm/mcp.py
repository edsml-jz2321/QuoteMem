"""MCP tool loading helpers for the LLM CLI."""

from __future__ import annotations

from heavenbase.utils import Any

from .runtime import _runtime_context


def _is_mcp_url(source: str) -> bool:
    return "://" in str(source or "")


def _load_mcp_ref(context: Any, ref: str, workspace_id: str | None = None) -> Any:
    from heavenbase.entity.registry import load_entity_artifact_export
    from heavenbase.workspace import HeavenBase

    owner = _runtime_context(context)
    workspace = HeavenBase.load(workspace_id, context=owner)
    load_toolkit_ref = load_entity_artifact_export("sys-toolkit", "refs", "load_toolkit_ref", resolver=owner.modules())
    return load_toolkit_ref(
        ref,
        ws=workspace,
    )


def _load_mcp_source(context: Any, source: str) -> Any:
    if not _is_mcp_url(source):
        return _load_mcp_ref(context, source)
    from heavenbase.entity.registry import load_entity_type

    owner = _runtime_context(context)
    Toolkit = load_entity_type("sys-toolkit", resolver=owner.modules())
    return Toolkit.from_fastmcp(source, resolver=owner.modules(), config=owner.config)


def _load_mcp_tools(context: Any, sources: list[str] | None = None) -> list[Any]:
    return [_load_mcp_source(context, source) for source in sources or []]


def _session_add_toolkit(context: Any, session: Any, toolkit: Any) -> None:
    session.add_tool(toolkit)
    context.out.muted(f"added toolkit {toolkit.name} ({len(toolkit.tools)} tools)")


def _session_load_mcp(context: Any, session: Any, source: str) -> None:
    _session_add_toolkit(context, session, _load_mcp_source(context, source))


def _session_emit_tools(context: Any, session: Any) -> None:
    rows = []
    for item in session.tools:
        if hasattr(item, "tools"):
            rows.extend([item.name, name, tool.description] for name, tool in item.tools.items())
        else:
            rows.append([type(item).__name__, getattr(item, "__name__", "<schema>"), getattr(item, "__doc__", "") or ""])
    context.out.table(["toolkit", "tool", "description"], rows) if rows else context.out.muted("no active tools")
