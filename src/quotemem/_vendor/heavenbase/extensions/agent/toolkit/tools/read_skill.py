"""MCP tool for reading one workspace skill file."""

from __future__ import annotations

__all__ = ["build_tool"]

from heavenbase.utils import Any, Dict

from ..errors import mcp_raise
from ..skills import read_skill_text


def build_tool(context: Any) -> Any:
    """Build the ``read_skill`` tool for one workspace context.

    Args:
        context (Any): Workspace MCP tool context supplying workspace and skill
            scope.

    Returns:
        Any: Callable MCP tool function for reading one skill file.
    """
    rows: Dict[str, Any] = {}

    def read_skill(skill_name: str, path: str = "SKILL.md") -> Dict[str, Any]:
        """Read one skill file, including the generated workspace guide.

        Args:
            skill_name: Persisted skill name or ``workspace-guide``.
            path: Relative archive path. Defaults to ``SKILL.md``.

        Returns:
            Skill name, path, and formatted file text.

        Raises:
            RuntimeError: If the skill cannot be read; the message is redacted for MCP use.
        """
        try:
            text = read_skill_text(
                skill_name,
                path=path,
                ws=context.workspace,
                scope=context.skill_scope,
                rows=rows,
                workspace_guide=context.workspace_guide(),
            )
            return {"skill_name": skill_name, "path": path, "text": text}
        except Exception as exc:
            mcp_raise(exc)

    return read_skill
