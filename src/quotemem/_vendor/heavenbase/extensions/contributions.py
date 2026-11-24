"""Generic workspace-activation constraints for Registry contributions."""

from __future__ import annotations

__all__ = [
    "contribution_enabled",
    "eligible_contributions",
    "resolve_contribution",
    "workspace_requirements",
]

from heavenbase.utils import Any, Iterable, Mapping
from heavenbase.utils.registry import named_key

from ..modules import ModuleResolver, module_resolver


def workspace_requirements(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return exact extension identifiers required by one Registry record.

    The requirement is activation policy, not a module-installation dependency.
    It therefore lives beside the record definition and is evaluated only when
    a workspace consumes that contribution.

    Args:
        row (Mapping[str, Any]): Installed Registry contribution row.

    Returns:
        tuple[str, ...]: Required extension identifiers, or an empty tuple when
        none are declared.

    Raises:
        TypeError: If ``workspace_requires`` is not a list of extension ids.
        ValueError: If ``workspace_requires`` contains duplicates.
    """
    record = row.get("record")
    meta = record.get("meta") if isinstance(record, Mapping) else None
    raw = meta.get("workspace_requires") if isinstance(meta, Mapping) else None
    if raw in (None, ()):
        return ()
    if not isinstance(raw, (list, tuple)):
        identifier = str(row.get("identifier") or "")
        raise TypeError(f"Registry contribution {identifier!r} workspace_requires must be a list of extension identifiers")
    requirements = tuple(named_key(value, "workspace extension requirement") for value in raw)
    if len(requirements) != len(set(requirements)):
        identifier = str(row.get("identifier") or "")
        raise ValueError(f"Registry contribution {identifier!r} workspace_requires contains duplicates")
    return requirements


def contribution_enabled(row: Mapping[str, Any], workspace: Any | None) -> bool:
    """Return whether one contribution may be consumed by ``workspace``.

    Passing no workspace preserves installation-level inspection: callers may
    examine every installed contribution without claiming it is active.

    Args:
        row (Mapping[str, Any]): Installed Registry contribution row.
        workspace (Any | None): Live workspace whose enabled extensions should
            be checked. Defaults to ``None``.

    Returns:
        bool: ``True`` when every declared requirement is active, or when no
        workspace or requirements are supplied.
    """
    requirements = workspace_requirements(row)
    if workspace is None or not requirements:
        return True
    enabled = getattr(workspace, "_enabled_extensions", None)
    if not isinstance(enabled, Mapping):
        report = getattr(workspace, "extensions", None)
        enabled = report() if callable(report) else {}
    return all(identifier in enabled for identifier in requirements)


def eligible_contributions(
    rows: Iterable[Mapping[str, Any]],
    workspace: Any | None,
) -> tuple[Mapping[str, Any], ...]:
    """Filter installed Registry rows by exact workspace activation state.

    Args:
        rows (Iterable[Mapping[str, Any]]): Installed Registry contribution
            rows.
        workspace (Any | None): Live workspace whose enabled extensions should
            be checked. Defaults to ``None``.

    Returns:
        tuple[Mapping[str, Any], ...]: Rows eligible for the supplied
        workspace activation state.
    """
    return tuple(row for row in rows if contribution_enabled(row, workspace))


def resolve_contribution(
    kind: str,
    identifier: str,
    *,
    resolver: ModuleResolver | None = None,
    workspace: Any | None = None,
    callable_only: bool = False,
) -> Any:
    """Resolve one installed contribution after workspace eligibility checks.

    Args:
        kind (str): Registry contribution kind such as ``toolkit_function``.
        identifier (str): Canonical contribution identifier.
        resolver (ModuleResolver | None): Module resolver used for lookup.
            Defaults to ``None``.
        workspace (Any | None): Live workspace used for eligibility checks.
            Defaults to ``None``.
        callable_only (bool): Whether the resolved value must be callable.
            Defaults to ``False``.

    Returns:
        Any: Resolved contribution value.

    Raises:
        RuntimeError: If the contribution is unavailable for this workspace.
        TypeError: If ``callable_only`` is true and the resolved value is not
            callable.
    """
    chosen = module_resolver(resolver if resolver is not None else getattr(workspace, "module_resolver", None))
    rows = eligible_contributions(chosen.inspect(kind, identifier), workspace)
    if not rows:
        raise RuntimeError(f"Registry contribution {kind}:{identifier} is unavailable for this workspace")
    value = chosen.resolve(kind, identifier)
    if callable_only and not callable(value):
        raise TypeError(f"Registry contribution {kind}:{identifier} must resolve to a callable")
    return value
