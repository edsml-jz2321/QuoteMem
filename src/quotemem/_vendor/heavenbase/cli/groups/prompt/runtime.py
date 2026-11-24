"""Workspace authority shared by prompt CLI handlers."""

from __future__ import annotations

from heavenbase.utils import Any

__all__: list[str] = []


def prompt_workspace(context: Any) -> Any:
    """Return the workspace owned by one CLI invocation's machine Context.

    Args:
        context (Any): ``PackageCLIContext`` carrying a callable
            ``runtime_context`` authority.

    Returns:
        Any: Loaded ``HeavenBase`` workspace facade for prompt commands.

    Raises:
        TypeError: If ``context`` is not a package CLI context with runtime authority.
    """
    from heavenbase.workspace import HeavenBase

    runtime = getattr(context, "runtime_context", None)
    if not callable(runtime):
        raise TypeError("Prompt CLI commands require a PackageCLIContext runtime authority")
    return HeavenBase.load(context=runtime())


def prompt_type(context: Any) -> type:
    """Return the Prompt class from the CLI workspace's artifact generation.

    Args:
        context (Any): ``PackageCLIContext`` used to resolve the active workspace.

    Returns:
        type: ``sys-prompt`` entity class from the workspace module resolver.
    """
    from heavenbase.entity.registry import load_entity_type

    workspace = prompt_workspace(context)
    return load_entity_type("sys-prompt", resolver=workspace.module_resolver)
