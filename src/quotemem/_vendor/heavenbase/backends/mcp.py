"""Declarative backend MCP tool and profile contracts."""

from __future__ import annotations

__all__ = ["BackendMCP", "BackendMCPProfile", "BackendMCPTool"]

from heavenbase.utils import Any, Callable, Iterable, Mapping, Sequence, dataclass, field, raise_mismatch


@dataclass
class BackendMCPTool:
    """One backend-owned MCP tool contract.

    Args:
        name: Stable MCP tool name exposed to agents and local ``Toolkit.run(...)`` callers.
        fn: Callable that implements the tool against the live backend instance.
        desc: Tool prompt/description shown to MCP clients. Defaults to ``name`` when omitted.
        read_only: Whether the tool is intended to inspect or query rather than mutate backend state.

    Examples:
        Define a small backend inspection tool::

            spec.tool("backend_ping", lambda: {"ok": True}, desc="Return whether this backend is reachable.")
    """

    name: str
    fn: Callable[..., Any]
    desc: str = ""
    read_only: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize the tool contract after dataclass construction.

        Raises:
            ValueError: If ``name`` is empty.
            TypeError: If ``fn`` is not callable.
        """
        name = str(self.name).strip()
        if not name:
            raise ValueError("backend MCP tool name must be non-empty")
        if not callable(self.fn):
            raise TypeError(f"backend MCP tool {name!r} must be callable")
        self.name = name
        self.desc = str(self.desc or name)
        self.read_only = bool(self.read_only)


@dataclass
class BackendMCPProfile:
    """Named backend MCP tool selection.

    Args:
        name: Profile identifier. Built-in backends should usually provide ``"inspect"`` and may provide
            ``"full"`` when mutating tools exist.
        tools: Ordered MCP tool names included by this profile.
        desc: Profile prompt/description shown in backend metadata and useful for agent routing.

    Examples:
        Create a custom read-only profile and pass it directly to a backend::

            profile = BackendMCPProfile("safe", ("describe_backend", "backend_tags"))
            toolkit = backend.to_mcp(profile=profile)
    """

    name: str
    tools: Sequence[str] = ()
    desc: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize the profile contract after dataclass construction.

        Raises:
            ValueError: If ``name`` or any selected tool name is empty.
        """
        name = str(self.name).strip()
        if not name:
            raise ValueError("backend MCP profile name must be non-empty")
        raw_tools = (self.tools,) if isinstance(self.tools, str) else self.tools
        tools = tuple(str(tool).strip() for tool in raw_tools)
        if any(not tool for tool in tools):
            raise ValueError(f"backend MCP profile {name!r} contains an empty tool name")
        self.name = name
        self.tools = tools
        self.desc = str(self.desc or "")

    @classmethod
    def load(cls, raw: Any, *, default_name: str = "custom") -> "BackendMCPProfile":
        """Load a profile from a profile object, mapping, or iterable of tool names.

        Args:
            raw: Existing ``BackendMCPProfile``, mapping with ``name`` / ``tools`` and
                ``desc`` / ``description`` / ``prompt`` keys, or an iterable of tool names.
            default_name: Profile name used when ``raw`` does not carry one.

        Returns:
            BackendMCPProfile: Normalized profile object.

        Raises:
            TypeError: If ``raw`` cannot be interpreted as a backend MCP profile.

        Examples:
            ``BackendMCPProfile.load({"name": "ops", "tools": ["describe_backend"]})``.
        """
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, Mapping):
            return cls(
                str(raw.get("name") or default_name),
                raw.get("tools") or (),
                desc=str(raw.get("desc") or raw.get("description") or raw.get("prompt") or ""),
            )
        if isinstance(raw, str):
            raise TypeError("backend MCP profile strings name registered profiles; use tools=[...] for ad-hoc tool lists")
        try:
            return cls(default_name, tuple(raw or ()))
        except TypeError as exc:
            raise TypeError("backend MCP profile must be a BackendMCPProfile, mapping, or iterable of tool names") from exc


@dataclass
class BackendMCP:
    """Complete backend MCP surface specification.

    A backend returns one ``BackendMCP`` object from :meth:`Backend.mcp`. The spec keeps tool callables,
    prompts/descriptions, profiles, default profile, and serializer together so provider authors do not
    have to maintain parallel hook methods.

    Args:
        desc: Toolkit-level prompt/description for this backend MCP surface.
        default_profile: Profile selected when ``Backend.to_mcp(profile=...)`` is omitted.
        serializer: Registered toolkit serializer name used for backend tool outputs.
        tools: Optional initial tool map keyed by tool name.
        profiles: Optional initial profile map keyed by profile name.

    Examples:
        Extend the default backend MCP surface from a provider::

            spec = super().mcp()
            spec.tool("provider_probe", self.provider_probe, desc="Inspect provider-specific state.")
            spec.profile("inspect", ("describe_backend", "backend_tags", "provider_probe"))
            return spec
    """

    desc: str = ""
    default_profile: str = "inspect"
    serializer: str = "jstr"
    tools: dict[str, BackendMCPTool] = field(default_factory=dict)
    profiles: dict[str, BackendMCPProfile] = field(default_factory=dict)

    def tool(self, name: str, fn: Callable[..., Any], *, desc: str = "", read_only: bool = True) -> "BackendMCP":
        """Add or replace one backend MCP tool.

        Args:
            name: Stable MCP tool name.
            fn: Callable that implements the tool.
            desc: Tool prompt/description. Defaults to ``name``.
            read_only: Whether the tool is intended to avoid backend mutations.

        Returns:
            BackendMCP: This spec for chained setup.

        Raises:
            ValueError: If ``name`` is empty.
            TypeError: If ``fn`` is not callable.
        """
        item = BackendMCPTool(name, fn, desc=desc, read_only=read_only)
        self.tools[item.name] = item
        return self

    def profile(self, name: str, tools: Iterable[str], *, desc: str = "") -> "BackendMCP":
        """Add or replace one backend MCP profile.

        Args:
            name: Profile identifier.
            tools: Ordered MCP tool names selected by this profile.
            desc: Profile prompt/description.

        Returns:
            BackendMCP: This spec for chained setup.

        Raises:
            ValueError: If profile or tool names are empty.
        """
        item = BackendMCPProfile(name, tools, desc=desc)
        self.profiles[item.name] = item
        return self

    def profile_summary(self) -> dict[str, dict[str, Any]]:
        """Return profile metadata for backend and agent inspection.

        The reserved ``full`` profile is always summarized from the current tool map so provider
        subclasses can add tools without manually keeping the inherited full profile synchronized.

        Args:
            None.

        Returns:
            dict[str, dict[str, Any]]: Profile names mapped to their ordered tools and descriptions.
        """
        rows = {name: {"tools": list(profile.tools), "desc": profile.desc} for name, profile in self.profiles.items()}
        full = self._load_profile("full")
        rows[full.name] = {"tools": list(full.tools), "desc": full.desc}
        return rows

    def select(self, *, profile: Any = None, tools: Iterable[str] | None = None) -> tuple[BackendMCPProfile, list[BackendMCPTool]]:
        """Resolve the requested profile and return selected tool contracts.

        Args:
            profile: ``None`` for ``default_profile``, a registered profile name, a ``BackendMCPProfile``,
                or a mapping accepted by ``BackendMCPProfile.load``.
            tools: Explicit tool allowlist. When provided, this overrides ``profile`` and creates an
                ad-hoc ``custom`` profile.

        Returns:
            tuple[BackendMCPProfile, list[BackendMCPTool]]: Resolved profile plus ordered tool contracts.

        Raises:
            ValueError: If the selected profile or tool list references an unknown tool.
            TypeError: If ``profile`` has an unsupported shape.
        """
        selected = BackendMCPProfile("custom", tools) if tools is not None else self._load_profile(profile)
        out = []
        for tool_name in selected.tools:
            item = self.tools.get(tool_name)
            if item is None:
                raise_mismatch(set(self.tools), tool_name, name="backend MCP tool")
            out.append(item)
        return selected, out

    def _load_profile(self, profile: Any = None) -> BackendMCPProfile:
        """Load a profile selection into a concrete profile object.

        Args:
            profile: ``None``, registered profile name, ``BackendMCPProfile``, or mapping/iterable
                accepted by ``BackendMCPProfile.load``.

        Returns:
            BackendMCPProfile: Concrete profile selection.

        Raises:
            ValueError: If a named profile is unknown.
            TypeError: If ``profile`` has an unsupported shape.
        """
        if profile is None:
            profile = self.default_profile
        if isinstance(profile, BackendMCPProfile):
            return profile
        if isinstance(profile, Mapping):
            return BackendMCPProfile.load(profile)
        if isinstance(profile, str):
            if profile == "full":
                found = self.profiles.get("full")
                return BackendMCPProfile("full", tuple(self.tools), desc=found.desc if found else "All backend MCP tools.")
            found = self.profiles.get(profile)
            if found is None:
                raise_mismatch(set(self.profiles) | {"full"}, profile, name="backend MCP profile")
            return found
        return BackendMCPProfile.load(profile)
