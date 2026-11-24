"""Lifespan-scoped native-result caching for MCP tool calls."""

from __future__ import annotations

__all__ = ["MCPToolCache", "MCPToolCacheConfig"]

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from time import monotonic

from heavenbase.utils import Any, Mapping, deepcopy, sha256hash


@dataclass(frozen=True)
class MCPToolCacheConfig:
    """Resolved MCP tool-call cache policy.

    Args:
        enabled (bool): Whether the cache may serve explicitly selected tools.
        tools (frozenset[str]): Exact tool names eligible for caching. The
            explicit value `"*"` selects every tool.
        ttl (float | None): Entry lifetime in seconds. `None` keeps entries
            until eviction or server shutdown.
        max_entries (int): Maximum entries retained by one MCP server.
    """

    enabled: bool
    tools: frozenset[str]
    ttl: float | None
    max_entries: int

    @classmethod
    def load(cls, config: Any) -> "MCPToolCacheConfig":
        """Resolve one cache policy from a HeavenBase configuration authority.

        Args:
            config (Any): Config authority that owns `heavenbase.mcp.cache`.

        Returns:
            MCPToolCacheConfig: Validated immutable cache policy.

        Raises:
            TypeError: If the cache configuration or tool allowlist has an
                unsupported shape.
            ValueError: If `ttl` or `max_entries` is outside its valid range.
        """
        raw = config.get("heavenbase.mcp.cache", default={}) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("heavenbase.mcp.cache must be a mapping")

        selected = raw.get("tools", ())
        if isinstance(selected, str):
            selected = (selected,)
        elif isinstance(selected, Mapping):
            raise TypeError("heavenbase.mcp.cache.tools must be a string or iterable of tool names")
        try:
            tools = frozenset(str(name).strip() for name in selected or ())
        except TypeError as exc:
            raise TypeError("heavenbase.mcp.cache.tools must be a string or iterable of tool names") from exc
        if "" in tools:
            raise ValueError("heavenbase.mcp.cache.tools cannot contain an empty tool name")

        raw_ttl = raw.get("ttl", 300)
        ttl = None if raw_ttl is None else float(raw_ttl)
        if ttl is not None and ttl <= 0:
            raise ValueError("heavenbase.mcp.cache.ttl must be greater than 0 or null")

        max_entries = int(raw.get("max_entries", 256))
        if max_entries <= 0:
            raise ValueError("heavenbase.mcp.cache.max_entries must be greater than 0")

        return cls(
            enabled=bool(raw.get("enabled", False)),
            tools=tools,
            ttl=ttl,
            max_entries=max_entries,
        )

    def allows(self, tool_name: str) -> bool:
        """Return whether one tool is explicitly eligible for caching.

        Args:
            tool_name (str): Exact Toolkit tool name.

        Returns:
            bool: Whether caching is enabled and the tool was selected.
        """
        return self.enabled and ("*" in self.tools or tool_name in self.tools)


class MCPToolCache:
    """Bounded native-result cache owned by one FastMCP server lifespan."""

    def __init__(self, config: MCPToolCacheConfig) -> None:
        """Create a closed cache from one resolved policy.

        Args:
            config (MCPToolCacheConfig): Validated cache policy.

        Returns:
            None: This initializer stores an empty closed cache.
        """
        self.config = config
        self._entries: "OrderedDict[str, tuple[float | None, Any]]" = OrderedDict()
        self._lock = RLock()
        self._open = False

    def __len__(self) -> int:
        """Return the number of currently retained entries."""
        with self._lock:
            return len(self._entries)

    def open(self) -> None:
        """Activate this cache for its owning server lifespan.

        Args:
            None.

        Returns:
            None: This method activates cache reads and writes.
        """
        with self._lock:
            self._open = True

    def close(self) -> None:
        """Clear and deactivate this cache at server shutdown.

        Args:
            None.

        Returns:
            None: This method releases all cached native results.
        """
        with self._lock:
            self._entries.clear()
            self._open = False

    @staticmethod
    def _key(tool_name: str, arguments: Mapping[str, Any]) -> str:
        return sha256hash({"tool": tool_name, "arguments": dict(arguments)})

    def get(self, tool_name: str, arguments: Mapping[str, Any]) -> tuple[bool, Any]:
        """Return a detached cached native value when present and fresh.

        Args:
            tool_name (str): Exact Toolkit tool name.
            arguments (Mapping[str, Any]): Validated MCP call arguments.

        Returns:
            tuple[bool, Any]: Hit flag and detached native value. Misses return
            `(False, None)`.
        """
        if not self.config.allows(tool_name):
            return False, None
        key = self._key(tool_name, arguments)
        with self._lock:
            if not self._open:
                return False, None
            found = self._entries.get(key)
            if found is None:
                return False, None
            expires_at, value = found
            if expires_at is not None and monotonic() >= expires_at:
                self._entries.pop(key, None)
                return False, None
            self._entries.move_to_end(key)
            return True, deepcopy(value)

    def set(self, tool_name: str, arguments: Mapping[str, Any], value: Any) -> bool:
        """Cache a detached native value for an explicitly selected tool.

        Args:
            tool_name (str): Exact Toolkit tool name.
            arguments (Mapping[str, Any]): Validated MCP call arguments.
            value (Any): Native tool result.

        Returns:
            bool: Whether the value was cached. Values that cannot be copied
            are deliberately skipped.
        """
        if not self.config.allows(tool_name):
            return False
        try:
            stored = deepcopy(value)
        except Exception:
            return False
        key = self._key(tool_name, arguments)
        with self._lock:
            if not self._open:
                return False
            expires_at = None if self.config.ttl is None else monotonic() + self.config.ttl
            self._entries[key] = (expires_at, stored)
            self._entries.move_to_end(key)
            while len(self._entries) > self.config.max_entries:
                self._entries.popitem(last=False)
        return True
