"""MCP tool returning the Python surface an agent writes against."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict

from ...guide import _render_workspace_guide, render_python_guide


def build_tool(workspace: Any, *, bindings: Dict[str, Any] | None = None) -> Any:
    """Build the ``python_guide`` tool for one workspace.

    Args:
        workspace (Any): Live workspace whose entities ground the examples.
        bindings (Dict[str, Any] | None): Initial namespace of the paired
            runner. A standalone guide describes the current workspace.

    Returns:
        Any: Callable MCP tool function for ``python_guide``.
    """

    def python_guide() -> str:
        """Return how to write code for ``run_python`` against this workspace.

        Read this before the first ``run_python`` call: it names the query
        surface, the reducers, the Json path and dict-fold forms, and the result
        fields that say whether a count is whole. Every example in it is
        executed by the test suite, so it cannot drift from the API.
        """
        if bindings is None:
            return render_python_guide(workspace)
        return _render_workspace_guide(workspace, bindings)

    return python_guide
