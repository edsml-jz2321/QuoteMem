"""Persistent Anthropic-style Skill entity."""

from __future__ import annotations

__all__ = ["Skill"]

from heavenbase.entity import Entity, field
from heavenbase.types import Artifact, Json, LongText, ShortText
from heavenbase.utils import Any, Dict, Iterable, List, Mapping, Optional, deepcopy, get_file_basename, hash_id, loads_yaml, sha256hash

from ._workspace import resolve_agent_workspace
from .skill_archive import pack_skill_dir, read_archive_file


def _frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            header = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            data = loads_yaml(header) or {}
            return dict(data) if isinstance(data, Mapping) else {}, body
    return {}, text


def _toolkit_refs(frontmatter: Mapping[str, Any]) -> List[str]:
    value = frontmatter.get("toolkit", frontmatter.get("toolkits", []))
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)]


def _skill_id(name: str, version: str) -> str:
    return hash_id("skill", name, version)


class Skill(Entity):
    """Workspace-persisted Skill folder with archive-backed files."""

    identifier = "agent-skill"
    __derive__ = [
        Entity.derive("catalog.desc").compute(lambda description: description, inputs=["description"]),
        Entity.derive("catalog.summary").compute(lambda skill_body: skill_body, inputs=["skill_body"]),
        Entity.derive("catalog.type").compute(lambda: "skill"),
    ]

    name = field(ShortText).desc("Skill name")
    version = field(ShortText).default("1").desc("Skill version")
    status = field(ShortText).default("active").desc("Skill status")
    description = field(LongText).default("").desc("Skill description")
    tags = field(Json).default([]).desc("Skill tags")
    frontmatter = field(Json).default({}).desc("Parsed SKILL.md frontmatter")
    toolkit_refs = field(Json).default([]).desc("Toolkit refs declared by this skill")
    skill_body = field(LongText).default("").desc("SKILL.md body")
    file_index = field(Json).default({}).desc("Archived file metadata")
    archive = field(Artifact).desc("Deterministic zipped skill folder")
    fingerprints = field(Json).default({}).desc("Archive and frontmatter fingerprints")

    def __init__(self, **values: Any) -> None:
        """Construct one persistent Skill entity row.

        Args:
            **values (Any): Skill field values. ``ws`` selects the bound
                workspace.

        Returns:
            None: This initializer does not return a value.
        """
        self.ws = values.pop("ws", None)
        data = {
            "name": "",
            "version": "1",
            "status": "active",
            "description": "",
            "tags": [],
            "frontmatter": {},
            "toolkit_refs": [],
            "skill_body": "",
            "file_index": {},
            "archive": b"",
            "fingerprints": {},
            **values,
        }
        name = str(data.get("name") or "skill")
        version = str(data.get("version") or "1")
        data["object_id"] = str(data.get("object_id") or _skill_id(name, version))
        data["tags"] = list(data.get("tags") or [])
        data["frontmatter"] = dict(data.get("frontmatter") or {})
        data["toolkit_refs"] = list(data.get("toolkit_refs") or [])
        data["file_index"] = dict(data.get("file_index") or {})
        data["fingerprints"] = dict(data.get("fingerprints") or {})
        super().__init__(**data)
        self._sync_attrs()

    def _sync_attrs(self) -> None:
        for key, value in self._data.items():
            setattr(self, key, value)

    @classmethod
    def from_path(cls, path: str, *, version: str = "1", ws: Any = None) -> "Skill":
        """Build a Skill row from a folder containing ``SKILL.md``.

        Args:
            path: Skill folder path.
            version: Skill version. Defaults to ``"1"``.
            ws: Optional workspace used to resolve the generation-local
                ``agent-skill`` class before construction. Defaults to ``None``,
                which keeps the calling class.

        Returns:
            Skill: Skill entity with archive bytes and metadata.
        """
        target_cls: type[Skill] = cls
        workspace = None
        if ws is not None:
            workspace = resolve_agent_workspace(ws)
            registered = workspace.entities.get(cls.identifier)
            if registered is not None:
                target_cls = registered
        archive, file_index = pack_skill_dir(path)
        skill_md = read_archive_file(archive, "SKILL.md").decode("utf-8", errors="replace")
        frontmatter, body = _frontmatter(skill_md)
        name = str(frontmatter.get("name") or get_file_basename(path))
        description = str(frontmatter.get("description") or "").strip()
        tags = frontmatter.get("tags") or []
        tags = [tags] if isinstance(tags, str) else list(tags or [])
        skill = target_cls(
            object_id=_skill_id(name, version),
            name=name,
            version=version,
            description=description,
            tags=tags,
            frontmatter=frontmatter,
            toolkit_refs=_toolkit_refs(frontmatter),
            skill_body=body.strip(),
            file_index=file_index,
            archive=archive,
            fingerprints={"archive": sha256hash(archive), "frontmatter": sha256hash(frontmatter)},
            ws=workspace,
        )
        return skill

    def register(self, ws: Any = None, overwrite: bool = True) -> str:
        """Register this Skill row in a workspace.

        Args:
            ws: Workspace used for storage. Defaults to the active workspace.
            overwrite: Whether to replace an existing row. Defaults to ``True``.

        Returns:
            str: Stored Skill ``object_id``.
        """
        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        entity_cls = workspace.entities.get(self.identifier, type(self))
        row_id = workspace.upsert_many(entity_cls, [self.to_dict()], overwrite=overwrite)[0]
        self.ws = workspace
        return row_id

    @classmethod
    def load(cls, ref: str, ws: Any = None, version: Optional[str] = None) -> "Skill":
        """Load one active Skill by object id or name.

        Args:
            ref: Skill object id or name.
            ws: Workspace used for storage. Defaults to the active workspace.
            version: Optional version when loading by name. Defaults to ``None``.

        Returns:
            Skill: Loaded Skill entity.

        Raises:
            KeyError: If no active Skill matches.
        """
        workspace = resolve_agent_workspace(ws)
        row = workspace.get(ref, entity=cls)
        if row is None:
            rows = [item for item in workspace.query(cls).where({"name": ref}).execute().rows() if item.get("status") == "active"]
            if version is not None:
                rows = [item for item in rows if str(item.get("version") or "1") == str(version)]
            row = sorted(rows, key=lambda item: str(item.get("version") or "1"))[-1] if rows else None
        if row is None:
            raise KeyError(f"No active agent skill found for {ref!r}")
        return cls.from_dict(row, ws=workspace)

    def read(self, path: str = "SKILL.md") -> str:
        """Read one archived Skill file as text.

        Args:
            path: Relative path inside the skill archive. Defaults to ``"SKILL.md"``.

        Returns:
            str: Formatted skill file content.

        Raises:
            KeyError: If the file is absent.
            ValueError: If the path is unsafe.
        """
        data = read_archive_file(self.archive, path)
        text = data.decode("utf-8", errors="replace")
        return f"Loading Skill: {self.name}\nPath: {path}\n\n{text}"

    def to_prompt(self) -> str:
        """Return a short model-facing Skill preview.

        Args:
            None.

        Returns:
            str: Skill name, description, toolkit refs, and body preview.
        """
        refs = ", ".join(self.toolkit_refs) if self.toolkit_refs else "none"
        return f'<skill name="{self.name}">\nDescription: {self.description}\nToolkits: {refs}\n{self.skill_body}\n</skill>'

    @classmethod
    def toolkit(cls, skills: Iterable["Skill"], ws: Any = None) -> Any:
        """Return a Toolkit exposing ``read_skill`` over the supplied Skills.

        Args:
            skills: Skills available to the agent.
            ws: Workspace used to load referenced Skill rows. Defaults to ``None``.

        Returns:
            Toolkit: Runtime toolkit with one ``read_skill`` tool.
        """
        from .toolkit.skills import skill_toolkit

        return skill_toolkit(list(skills), ws=ws)
