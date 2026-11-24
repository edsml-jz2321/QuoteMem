"""Factory for python-owned workspace MCP tools."""

from __future__ import annotations

__all__ = ["python_tool_functions"]

from heavenbase.utils import Any, Dict

from ..namespace import python_namespace
from .tools.python_guide import build_tool as build_python_guide_tool
from .tools.run_python import build_tool as build_run_python_tool


def python_tool_functions(workspace: Any) -> Dict[str, Any]:
    """Return MCP-callable python tool functions bound to one workspace.

    Args:
        workspace (Any): Live workspace bound to python MCP tools.

    Returns:
        Dict[str, Any]: Tool name to callable mapping for python MCP tools.
    """
    namespace = python_namespace(workspace)
    return {
        "run_python": build_run_python_tool(workspace, namespace=namespace),
        "python_guide": build_python_guide_tool(workspace, bindings=namespace.copy()),
    }
