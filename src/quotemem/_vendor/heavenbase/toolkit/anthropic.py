"""Anthropic Programmatic Tool Calling export helpers."""

from __future__ import annotations

__all__ = ["arun_for_anthropic", "run_for_anthropic", "to_anthropic_tools"]

from heavenbase.utils import Any, Dict, Mapping

from .errors import AsyncToolRequired, ToolExportError


def run_for_anthropic(tool: Any, arguments: Mapping[str, Any]) -> str:
    """Run a tool and return a string result for code execution.

    Args:
        tool (Any): Tool object with ``run_to_str``.
        arguments (Mapping[str, Any]): Tool-call arguments.

    Returns:
        str: Serialized tool result.

    Raises:
        TypeError: If the tool is async and needs ``arun_for_anthropic``.
    """
    try:
        return tool.run_to_str(**dict(arguments))
    except AsyncToolRequired as exc:
        raise TypeError("async tool execution needs arun_for_anthropic") from exc


async def arun_for_anthropic(tool: Any, arguments: Mapping[str, Any]) -> str:
    """Run a tool asynchronously and return a string result for code execution.

    Args:
        tool (Any): Tool object with ``arun_to_str``.
        arguments (Mapping[str, Any]): Tool-call arguments.

    Returns:
        str: Serialized tool result.
    """
    return await tool.arun_to_str(**dict(arguments))


def to_anthropic_tools(toolkit: Any, *, allowed_callers: list[str] | None = None, code_tool_name: str = "code_execution") -> list[Dict[str, Any]]:
    """Return Anthropic-compatible tool definitions.

    Args:
        toolkit (Any): Toolkit whose tools should be exported.
        allowed_callers (list[str] | None): Caller tool names permitted to invoke each exported tool. Defaults to ``[code_tool_name]``.
        code_tool_name (str): Default caller name when ``allowed_callers`` is omitted. Defaults to ``"code_execution"``.

    Returns:
        list[Dict[str, Any]]: Anthropic Programmatic Tool Calling definition dictionaries.

    Raises:
        ToolExportError: If a tool cannot be exported for programmatic use.
    """
    callers = list(allowed_callers or [code_tool_name])
    tools = []
    for tool in toolkit.tools.values():
        cap = tool.capsule.manifest.capabilities.get("anthropic_programmatic", {})
        if not cap.get("ok", True):
            raise ToolExportError(f"Tool {tool.name!r} cannot be exported for Anthropic programmatic use: {cap.get('reasons')}")
        tools.append({"name": tool.name, "description": tool.description, "input_schema": tool.input_schema, "allowed_callers": callers})
    return tools
