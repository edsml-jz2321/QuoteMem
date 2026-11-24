"""MCP tool for running caller-authored Python against one workspace."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict

from ...namespace import python_namespace
from ...runner import run_python_code


def build_tool(workspace: Any, *, namespace: Dict[str, Any] | None = None) -> Any:
    """Build the ``run_python`` tool for one workspace.

    Args:
        workspace (Any): Live workspace the executed program queries.
        namespace (Dict[str, Any] | None): Persistent globals prepared for
            the workspace. Defaults to a fresh workspace namespace.

    Returns:
        Any: Callable MCP tool function for ``run_python``.
    """
    if namespace is None:
        namespace = python_namespace(workspace)

    def run_python(code: str) -> Dict[str, Any]:
        """Run Python against the open workspace and return what it produced.

        ``hb``, ``ws``, and entity classes are preloaded when the tools are
        built. Use ``ws.entities`` to look up entities registered later.
        Statements run in order and a trailing expression becomes the result,
        the same way a notebook cell behaves. Names assigned by one call stay
        bound for the next.

        Use this when the answer needs something the ``query`` spec cannot
        express — correlating two entities, a statistic with no registered
        reducer, ranking within groups, or deriving a value per row. Prefer
        ``query`` when the spec can say it: the engine validates a spec and
        names what is legal when it cannot, while a wrong program returns a
        plausible number with no such signal.

        Args:
            code: Python source. Multiple statements are fine; the last
                expression, if any, is returned as ``result``.

        Returns:
            ``{"ok", "stdout", "result", "elapsed"}``, plus ``error`` and
            ``traceback`` when the program raised. A ``ResultFrame`` result is
            rendered as the standard query payload, so it carries ``total`` and
            ``truncated`` when pagination withheld rows — report ``total``, not
            the rows you received. ``stdout_truncated``/``result_truncated``
            with a ``*_chars`` length appear when output exceeded its cap.
        """
        return run_python_code(code, namespace)

    return run_python
