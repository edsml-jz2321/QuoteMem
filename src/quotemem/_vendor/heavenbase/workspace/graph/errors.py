"""Graph mutation errors."""

from __future__ import annotations

__all__ = ["GraphDeleteRestrictedError"]


class GraphDeleteRestrictedError(ValueError):
    """Report surviving GraphEdge references that block a target delete.

    Args:
        references (list[dict[str, str]]): Deterministic reference records with
            `entity`, `field`, `source`, `target_entity`, and `target` keys.
    """

    def __init__(self, references: list[dict[str, str]]) -> None:
        """Initialize one GraphEdge delete restriction error.

        Returns:
            None: This initializer does not return a value.
        """
        self.references = [dict(item) for item in references]
        preview = ", ".join(f"{item['entity']}.{item['field']}:{item['source']}->{item['target']}" for item in self.references[:8])
        suffix = "" if len(self.references) <= 8 else f", +{len(self.references) - 8} more"
        super().__init__(
            f"Delete restricted by {len(self.references)} GraphEdge reference(s): {preview}{suffix}. "
            "Pass edges='detach' to remove the relationships explicitly."
        )
