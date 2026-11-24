"""Workspace MCP toolkit family value contract."""

from __future__ import annotations

__all__ = ["WorkspaceToolkitFamilySpec"]

from types import MappingProxyType

from heavenbase.utils import Any, Callable, Mapping, dataclass, field


@dataclass(frozen=True)
class WorkspaceToolkitFamilySpec:
    """Declare one workspace MCP tool family contribution.

    A toolkit family is the smallest extension unit for workspace-bound MCP
    tools: it names the family, lists the concrete tools it contributes, and
    points to the builder that creates those tool callables for one workspace.

    Attributes:
        name: Stable family identifier mirrored into `toolkit_family` metadata rows.
        tool_names: Concrete MCP tool names contributed by this family.
        builder: Callable that returns a mapping of tool name to tool callable.
        desc: Human-readable family description for toolkits and system metadata.
        meta: Additional inspectable metadata. Set `builder_context` to
            `"workspace_tool_context"` when the builder needs profile/tool scope
            instead of the bare workspace object.

    Raises:
        ValueError: If `name` or any tool name is empty.
        TypeError: If `builder` is not callable.
    """

    name: str
    tool_names: tuple[str, ...]
    builder: Callable[[Any], Mapping[str, Any]]
    desc: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("workspace toolkit family name must be non-empty")
        tool_names = tuple(str(tool_name).strip() for tool_name in self.tool_names)
        if not tool_names or any(not tool_name for tool_name in tool_names):
            raise ValueError("workspace toolkit family needs non-empty tool names")
        if not callable(self.builder):
            raise TypeError(f"workspace toolkit family {name!r} builder must be callable")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tool_names", tool_names)
        object.__setattr__(self, "desc", str(self.desc or ""))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta or {})))
