"""Persistent Agent recipe entity."""

from __future__ import annotations

__all__ = ["Agent", "AgentRuntime"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

from heavenbase.entity import Entity, field
from heavenbase.types import Json, LongText, ShortText
from heavenbase.utils import Any, Dict, Iterable, List, Optional, hash_id

from ._workspace import resolve_agent_workspace
from .session import Session
from .skill import Skill

if TYPE_CHECKING:
    from heavenbase.llm import LLM, LLMSession


@dataclass
class AgentRuntime:
    """Resolved runtime view of a persistent Agent recipe."""

    agent: "Agent"
    llm: LLM
    session: LLMSession
    skills: List[Skill]
    toolkits: List[Any]


def _agent_id(name: str, version: str) -> str:
    return hash_id("agent", name, version)


class Agent(Entity):
    """Composable Agent recipe assembled from existing HeavenBase runtimes."""

    identifier = "agent-agent"
    __derive__ = [
        Entity.derive("catalog.desc").compute(lambda description: description, inputs=["description"]),
        Entity.derive("catalog.type").compute(lambda: "agent"),
    ]

    name = field(ShortText).desc("Agent name")
    version = field(ShortText).default("1").desc("Agent version")
    status = field(ShortText).default("active").desc("Agent status")
    description = field(LongText).default("").desc("Agent description")
    tags = field(Json).default([]).desc("Agent tags")
    llm_preset = field(ShortText).default("chat").desc("LLM preset")
    llm_args = field(Json).default({}).desc("LLM runtime args")
    prompt_ref = field(ShortText).default("").desc("Prompt row ref or literal system prompt")
    prompt_args = field(Json).default({}).desc("Prompt render arguments")
    skill_refs = field(Json).default([]).desc("Skill refs loaded from the workspace")
    toolkit_refs = field(Json).default([]).desc("Registered Toolkit refs")
    args = field(Json).default({"max_steps": 20}).desc("Agent runtime arguments")
    manifest = field(Json).default({}).desc("Agent manifest")

    def __init__(self, **values: Any) -> None:
        """Construct one persistent Agent recipe entity row.

        Args:
            **values (Any): Agent field values. ``ws`` selects the bound
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
            "llm_preset": "chat",
            "llm_args": {},
            "prompt_ref": "",
            "prompt_args": {},
            "skill_refs": [],
            "toolkit_refs": [],
            "args": {"max_steps": 20},
            "manifest": {},
            **values,
        }
        name = str(data.get("name") or "agent")
        version = str(data.get("version") or "1")
        data["object_id"] = str(data.get("object_id") or _agent_id(name, version))
        for key in ("tags", "skill_refs", "toolkit_refs"):
            data[key] = list(data.get(key) or [])
        for key in ("llm_args", "prompt_args", "args", "manifest"):
            data[key] = dict(data.get(key) or {})
        super().__init__(**data)
        self._sync_attrs()

    def _sync_attrs(self) -> None:
        for key, value in self._data.items():
            setattr(self, key, value)

    def register(self, ws: Any = None, overwrite: bool = True) -> str:
        """Register this Agent recipe in a workspace.

        Args:
            ws: Workspace used for storage. Defaults to this Agent's workspace or active workspace.
            overwrite: Whether to replace an existing row. Defaults to ``True``.

        Returns:
            str: Stored Agent ``object_id``.
        """
        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        row_id = workspace.upsert_many(type(self), [self.to_dict()], overwrite=overwrite)[0]
        self.ws = workspace
        return row_id

    @classmethod
    def load(cls, ref: str, ws: Any = None, version: Optional[str] = None) -> "Agent":
        """Load one active Agent by object id or name.

        Args:
            ref: Agent object id or name.
            ws: Workspace used for storage. Defaults to the active workspace.
            version: Optional version when loading by name. Defaults to ``None``.

        Returns:
            Agent: Loaded Agent recipe.

        Raises:
            KeyError: If no active Agent matches.
        """
        workspace = resolve_agent_workspace(ws)
        row = workspace.get(ref, entity=cls)
        if row is None:
            rows = [item for item in workspace.query(cls).where({"name": ref}).execute().rows() if item.get("status") == "active"]
            if version is not None:
                rows = [item for item in rows if str(item.get("version") or "1") == str(version)]
            row = sorted(rows, key=lambda item: str(item.get("version") or "1"))[-1] if rows else None
        if row is None:
            raise KeyError(f"No active agent found for {ref!r}")
        return cls.from_dict(row, ws=workspace)

    def resolve_prompt(self, ws: Any = None) -> List[Dict[str, Any]]:
        """Resolve this Agent's prompt into initial runtime messages.

        Args:
            ws: Workspace used to load prompt rows. Defaults to this Agent's workspace.

        Returns:
            List[Dict[str, Any]]: Initial OpenAI-format messages.
        """
        if not self.prompt_ref:
            return []
        from heavenbase.entity.registry import load_entity_type

        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        content = self.prompt_ref
        if "prompt" in workspace.extensions():
            Prompt = load_entity_type("sys-prompt", resolver=workspace.module_resolver)
            try:
                content = Prompt.load(self.prompt_ref, ws=workspace)(**self.prompt_args)
            except KeyError:
                pass
        return [{"role": "system", "content": str(content)}]

    def resolve_skills(self, ws: Any = None) -> List[Skill]:
        """Resolve Skill refs from the workspace.

        Args:
            ws: Workspace used to load Skill rows. Defaults to this Agent's workspace.

        Returns:
            List[Skill]: Loaded Skill rows.
        """
        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        return [Skill.load(str(ref), ws=workspace) for ref in self.skill_refs]

    def resolve_toolkits(self, skills: Iterable[Skill] = (), ws: Any = None) -> List[Any]:
        """Resolve registered Toolkit refs declared by the Agent and Skills.

        Args:
            skills: Already loaded Skill rows. Defaults to ``()``.
            ws: Workspace used for registry-aware resolution. Defaults to this Agent's workspace.

        Returns:
            List[Toolkit]: Loaded Toolkits in attachment order.
        """
        refs: List[str] = []
        for skill in skills:
            refs.extend(str(ref) for ref in skill.toolkit_refs)
        refs.extend(str(ref) for ref in self.toolkit_refs)
        if not refs:
            return []
        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        if "toolkit" not in workspace.extensions():
            raise RuntimeError("Agent Toolkit references require the workspace 'toolkit' extension")
        from heavenbase.entity.registry import load_entity_artifact_export

        load_toolkit_ref = load_entity_artifact_export(
            "sys-toolkit",
            "refs",
            "load_toolkit_ref",
            resolver=workspace.module_resolver,
        )
        return [
            load_toolkit_ref(
                ref,
                ws=workspace,
            )
            for ref in refs
        ]

    def build(self, ws: Any = None) -> AgentRuntime:
        """Resolve this recipe into an ``LLMSession`` with attached tools.

        Args:
            ws: Workspace used to load prompts and Skills. Defaults to this Agent's workspace.

        Returns:
            AgentRuntime: Runtime bundle of Agent, LLM, session, Skills, and Toolkits.
        """
        from heavenbase.llm import LLM, LLMSession

        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        llm = LLM(context=workspace.context, preset=self.llm_preset or None, **self.llm_args)
        session = LLMSession(llm=llm, messages=self.resolve_prompt(ws=workspace))
        skills = self.resolve_skills(ws=workspace)
        toolkits = self.resolve_toolkits(skills, ws=workspace)
        if skills:
            session.add_tool(Skill.toolkit(skills, ws=workspace))
        for toolkit in toolkits:
            session.add_tool(toolkit)
        return AgentRuntime(agent=self, llm=llm, session=session, skills=skills, toolkits=toolkits)

    def new_session(self, title: str = "", messages: Optional[Iterable[Dict[str, Any]]] = None, ws: Any = None) -> Session:
        """Create a persistent Session for this Agent.

        Args:
            title: Optional session title. Defaults to the Agent name.
            messages: Optional OpenAI-format initial messages. Defaults to ``None``.
            ws: Workspace used for storage. Defaults to this Agent's workspace.

        Returns:
            Session: Persisted Session row.
        """
        workspace = resolve_agent_workspace(ws if ws is not None else self.ws)
        session = Session(title=title or self.name, agent_ref=self.name, ws=workspace)
        workspace.upsert(Session, session.to_dict())
        if messages:
            session.append_messages(messages, source="agent", ws=workspace)
        return session
