"""Runtime Toolkit for reading persisted Skills."""

from __future__ import annotations

__all__ = [
    "WORKSPACE_GUIDE_SKILL",
    "list_skill_entries",
    "read_skill_text",
    "skill_allowed",
    "skill_toolkit",
]

from heavenbase.utils import Any, Dict, Iterable, List, Mapping, Optional

from ..skill import Skill

SkillsScope = tuple[str, ...] | str
WORKSPACE_GUIDE_SKILL = "workspace-guide"


def skill_allowed(skill_name: str, scope: SkillsScope) -> bool:
    """Return whether one skill name is visible under a profile skills scope.

    Args:
        skill_name (str): Persisted skill name or ``workspace-guide``.
        scope (SkillsScope): Profile skills scope, either ``"*"`` or explicit
            skill names.

    Returns:
        bool: ``True`` when the skill is visible under ``scope``.
    """
    if skill_name == WORKSPACE_GUIDE_SKILL:
        return True
    if scope == "*":
        return True
    return skill_name in scope


def _active_skill_rows(ws: Any) -> List[Mapping[str, Any]]:
    if ws is None or "agent-skill" not in getattr(ws, "entities", {}):
        return []
    return [row for row in ws.rows(Skill) if str(row.get("status") or "active") == "active"]


def list_skill_entries(ws: Any, scope: SkillsScope) -> List[Dict[str, str]]:
    """List skill index entries for one workspace and profile skills scope.

    Args:
        ws: Workspace that may hold persisted ``agent-skill`` rows.
        scope: Profile skills scope (``"*"`` or explicit skill names).

    Returns:
        Sorted skill entries with ``name`` and ``description`` keys.
    """
    entries: Dict[str, Dict[str, str]] = {}
    if scope:
        entries[WORKSPACE_GUIDE_SKILL] = {
            "name": WORKSPACE_GUIDE_SKILL,
            "description": "Generated live workspace guide (entities, profile, query shape, object_id semantics).",
        }
    for row in _active_skill_rows(ws):
        name = str(row.get("name") or "")
        if not name or not skill_allowed(name, scope):
            continue
        entries[name] = {"name": name, "description": str(row.get("description") or "").strip()}
    return [entries[name] for name in sorted(entries)]


def read_skill_text(
    skill_name: str,
    *,
    path: str = "SKILL.md",
    ws: Any = None,
    scope: SkillsScope = "*",
    rows: Optional[Dict[str, Skill]] = None,
    workspace_guide: Optional[str] = None,
) -> str:
    """Read one skill file, including the built-in workspace guide when requested.

    Args:
        skill_name: Persisted skill name or ``workspace-guide``.
        path: Relative archive path. Defaults to ``SKILL.md``.
        ws: Workspace used to lazy-load missing skills.
        scope: Profile skills scope used to enforce visibility.
        rows: Optional in-memory skill cache keyed by name.
        workspace_guide: Pre-rendered workspace guide text for ``workspace-guide``.

    Returns:
        str: Formatted skill file content.

    Raises:
        KeyError: If the skill is unknown or outside the profile scope.
        ValueError: If the archive path is unsafe.
    """
    name = str(skill_name or "").strip()
    if not skill_allowed(name, scope):
        raise KeyError(f"Skill {name!r} is outside the active MCP profile skills scope")
    if name == WORKSPACE_GUIDE_SKILL:
        if workspace_guide is None:
            raise KeyError(f"No workspace guide is available for skill {name!r}")
        return f"Loading Skill: {WORKSPACE_GUIDE_SKILL}\nPath: {path}\n\n{workspace_guide}"
    cache = rows if rows is not None else {}
    skill = cache.get(name)
    if skill is None:
        skill = Skill.load(name, ws=ws)
        cache[skill.name] = skill
    return skill.read(path)


def skill_toolkit(skills: Iterable[Skill], ws: Any = None) -> Any:
    """Return a runtime Toolkit exposing ``read_skill``.

    Args:
        skills: Skills available for reads.
        ws: Optional workspace used to lazy-load missing skill names.

    Returns:
        Toolkit: Runtime-only toolkit.
    """
    rows: Dict[str, Skill] = {skill.name: skill for skill in skills}

    def read_skill(skill_name: str, path: str = "SKILL.md") -> str:
        """Read a file from a persisted Skill archive."""
        return read_skill_text(skill_name, path=path, ws=ws, scope="*", rows=rows)

    if ws is not None and "toolkit" not in ws.extensions():
        raise RuntimeError("Agent Skill Toolkit support requires the workspace 'toolkit' extension")
    from heavenbase.entity.registry import load_entity_type

    resolver = getattr(ws, "module_resolver", None)
    Toolkit = load_entity_type("sys-toolkit", resolver=resolver)
    return Toolkit(
        "agent-skills",
        {"read_skill": read_skill},
        namespace="agent",
        resolver=getattr(ws, "module_resolver", None),
        config=getattr(getattr(ws, "context", None), "config", None),
    )
