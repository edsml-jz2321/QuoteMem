"""Shared descriptor for database-owned SQL templates."""

from __future__ import annotations

__all__ = ["SqlTemplate"]

from heavenbase.utils import Any, dataclass


@dataclass(frozen=True)
class SqlTemplate:
    """Static SQL template bundled as a Python resource.

    Args:
        name (str): Stable dotted resource name.
        text (str): SQL text with ``str.format`` placeholders.

    Returns:
        SqlTemplate: Immutable SQL template descriptor.
    """

    name: str
    text: str

    def render(self, **values: Any) -> str:
        """Render the SQL template with caller-supplied quoted fragments.

        Args:
            **values (Any): Pre-validated template values. Identifier values must already be quoted by the caller.

        Returns:
            str: Rendered SQL text.
        """
        return self.text.format(**values)
