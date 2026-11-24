"""In-memory scalar-index handler contribution."""

from __future__ import annotations

from heavenbase.execution.handlers.indexed import register_indexed_handlers


def register_handlers(registry, ctx) -> None:
    """Register InMem native equality/index compilers.

    Args:
        registry: Workspace handler registry receiving descriptors.
        ctx: Shared Registry-derived handler vocabulary.

    Returns:
        None: The supplied workspace registry is updated in place.
    """
    register_indexed_handlers(registry, ctx, "inmem")
