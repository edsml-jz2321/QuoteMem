"""MCP profile value contract for workspace tool exposure."""

from __future__ import annotations

__all__ = ["EntitiesScope", "ProfileSpec", "SkillsScope"]

from heavenbase.utils import Any, Literal, Mapping, Tuple, dataclass, field

EntitiesScope = Tuple[str, ...] | Literal["*"]
SkillsScope = Tuple[str, ...] | Literal["*"]


@dataclass(frozen=True)
class ProfileSpec:
    """Declarative MCP exposure profile.

    Attributes:
        name: Profile identifier used by `workspace.to_mcp(...)`.
        extends: Parent profile names flattened before tool resolution.
        tools: Tool names or tool groups selected by this profile.
        entities: Entity ids visible to workspace tools, or `"*"`.
        skills: Skill ids visible to agent skill tools, or `"*"`.
        serializer: Registered serializer name for tool outputs.
        desc: Human-readable profile description for system metadata.
        meta: Additional inspectable metadata mirrored into system rows.
    """

    name: str
    extends: Tuple[str, ...] = ()
    tools: Tuple[str, ...] = ()
    entities: EntitiesScope = "*"
    skills: SkillsScope = ()
    serializer: str = "jstr"
    desc: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)
