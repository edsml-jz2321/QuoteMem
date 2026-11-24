"""LLM chat session state API."""

from __future__ import annotations

__all__ = ["LLMSession"]

from heavenbase.utils.typing import Any, Mapping, dataclass, field
from ..utils.serialize import load_json, save_json

from .base import LLM, format_messages


@dataclass
class LLMSession:
    """Own one mutable LLM conversation and its runtime tools.

    `LLMSession` keeps OpenAI-format history around one `LLM` runtime. Runtime
    tools participate in every `send` call but are deliberately excluded from
    serialization.

    Args:
        llm (LLM): Runtime used for each turn. Defaults to a new `LLM`.
        messages (list[dict[str, Any]]): Initial OpenAI-format history.
        tools (list[Any]): Initial runtime tools used by every `send` call.
    """

    llm: LLM = field(default_factory=LLM)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = format_messages(self.messages)
        tools, self.tools = list(self.tools), []
        for tool in tools:
            self.add_tool(tool)

    @staticmethod
    def _tool_names(tool: Any) -> list[str]:
        if isinstance(tool, Mapping):
            function = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
            name = function.get("name") or tool.get("name")
            return [str(name)] if name else []
        toolkit_tools = getattr(tool, "tools", None)
        if isinstance(toolkit_tools, Mapping):
            return [str(name) for name in toolkit_tools]
        if callable(getattr(tool, "to_openai_tool", None)):
            return LLMSession._tool_names(tool.to_openai_tool())
        name = getattr(tool, "__name__", "") if callable(tool) else ""
        return [str(name)] if name else []

    def list_tools(self) -> list[str]:
        """Return active runtime tool names in send order.

        Args:
            None.

        Returns:
            list[str]: Names resolved from attached callables, schemas, Tools,
            Toolkits, and imported MCP Toolkits.
        """
        return [name for tool in self.tools for name in self._tool_names(tool)]

    def add_tool(self, tool: Any) -> Any:
        """Attach one tool-like object to this session.

        Args:
            tool (Any): Callable, HeavenBase ``Tool``, HeavenBase ``Toolkit``, or OpenAI-compatible schema tool.

        Returns:
            Any: The same tool object, allowing decorator-style use.

        Raises:
            ValueError: If an attached tool already has the same name.
        """
        active = set(self.list_tools())
        names = self._tool_names(tool)
        duplicates = sorted(name for name in names if name in active)
        if duplicates:
            raise ValueError(f"Duplicate session tool name(s): {', '.join(duplicates)}")
        self.tools.append(tool)
        return tool

    def add_mcp(self, server: Any, *, name: str | None = None, namespace: str = "mcp-client") -> Any:
        """Import an MCP server as a runtime Toolkit and attach it.

        Args:
            server (Any): FastMCP server, URL, transport, or any value accepted by ``fastmcp.Client``.
            name (str | None): Local Toolkit name. Defaults to the remote server
                name or `"mcp-client"`.
            namespace (str): Namespace for runtime proxy Capsules. Defaults to
                `"mcp-client"`.

        Returns:
            Any: Runtime Toolkit attached to this session.

        Warnings:
            Imported MCP Toolkits close over a live remote server or transport and are runtime-only; they are not safe to
            persist in the Capsule registry.

        Raises:
            ValueError: If the imported Toolkit duplicates an attached tool name.
        """
        from heavenbase.entity.registry import load_entity_type

        spec = getattr(self.llm, "spec", None)
        engine = getattr(self.llm, "engine", None)
        resolver = getattr(spec, "resolver", None)
        config = getattr(engine, "manager", None)
        if (resolver is None) != (config is None):
            raise RuntimeError("LLMSession LLM exposes an incomplete runtime authority")
        Toolkit = load_entity_type("sys-toolkit", resolver=resolver)
        return self.add_tool(
            Toolkit.from_fastmcp(
                server,
                name=name,
                namespace=namespace,
                resolver=resolver,
                config=config,
            )
        )

    def send(self, prompt: str, system: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Send one user prompt and append the new assistant turn.

        Attached tools run in a loop bounded by `max_tool_turns`, which defaults to 8.
        A turn that runs out of rounds still returns its mid-conversation message, so the
        returned content carries an explicit `[incomplete: ...]` note; read it rather than
        assuming every returned message is a finished answer. Call `LLM.chat` directly with
        `include="truncated"` when the verdict is needed as a flag instead of prose.

        Args:
            prompt (str): User message content for this turn.
            system (str | None): System message inserted at the beginning when
                history does not already contain one. Defaults to `None`.
            **kwargs (Any): Extra `LLM.chat` options such as `temperature`,
                `max_tool_turns`, or `cache`. Attach tools with `add_tool` or
                `add_mcp`; the response projection is always the new delta.

        Returns:
            dict[str, Any]: Last assistant message from the generated delta, or
                an empty dictionary when the provider returns none. A message cut
                short by the tool-round budget states that in its content.

        Raises:
            TypeError: If `include` or `tools` attempts to replace session-owned
                call options.
        """
        reserved = sorted({"include", "tools"} & kwargs.keys())
        if reserved:
            raise TypeError(f"LLMSession.send owns option(s): {', '.join(reserved)}")
        if system and not any(message.get("role") == "system" for message in self.messages):
            self.messages.insert(0, {"role": "system", "content": system})
        self.messages.extend(format_messages(prompt))
        delta = self.llm.chat(self.messages, include="delta", tools=self.tools, **kwargs)
        self.messages.extend(delta)
        return next((message for message in reversed(delta) if message.get("role") == "assistant"), {})

    def undo(self) -> bool:
        """Remove the latest user turn and everything after it.

        Args:
            None.

        Returns:
            bool: `True` when a user turn was removed, otherwise `False`.
        """
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index].get("role") == "user":
                del self.messages[index:]
                return True
        return False

    def clear(self) -> None:
        """Clear all session messages.

        Args:
            None.

        Returns:
            None: The session message list is mutated in place.
        """
        self.messages.clear()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe session data.

        Args:
            None.

        Returns:
            dict[str, Any]: Message-only payload. Runtime tools are omitted.
        """
        return {"messages": [dict(message) for message in self.messages]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], llm: LLM | None = None) -> "LLMSession":
        """Build a session from JSON-safe data.

        Args:
            data (Mapping[str, Any]): Message-only payload from `to_dict`.
            llm (LLM | None): Runtime to attach. Defaults to a new `LLM`.

        Returns:
            LLMSession: Session with restored messages and no runtime tools.

        Raises:
            TypeError: If `data` is not a mapping or `messages` is not a list.
            ValueError: If a message cannot be normalized.
        """
        if not isinstance(data, Mapping):
            raise TypeError(f"LLM session data must be a mapping, got {type(data).__name__}")
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            raise TypeError(f"LLM session messages must be a list, got {type(messages).__name__}")
        return cls(llm=llm or LLM(), messages=messages)

    def save(self, path: str) -> None:
        """Save session messages to JSON.

        Args:
            path (str): Destination JSON path.

        Returns:
            None: File contents are written through HeavenBase's JSON utility layer.
        """
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str, llm: LLM | None = None) -> "LLMSession":
        """Load session messages from JSON.

        Args:
            path (str): Source JSON path created by ``save``.
            llm (LLM | None): Runtime to attach. Defaults to a new `LLM`.

        Returns:
            LLMSession: Restored session with message history and no runtime tools.

        Raises:
            FileNotFoundError: If `path` does not exist.
            TypeError: If the decoded payload has an invalid session shape.
            ValueError: If a decoded message cannot be normalized.
        """
        return cls.from_dict(load_json(path, strict=True), llm=llm)
