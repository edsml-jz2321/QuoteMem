"""Optional tool-call argument repair for LLM responses."""

from __future__ import annotations

__all__ = ["LLMToolCallRepair"]

import threading
from typing import TYPE_CHECKING
import weakref

from heavenbase.utils import Any, Dict, Iterable, List, Mapping, Optional, dmerge, dumps_json, loads_json, resolve_interpolations
from heavenbase.utils.typing import deepcopy

if TYPE_CHECKING:
    from heavenbase.utils.config import ConfigManager

TRUE_VALUES = {"1", "true", "yes", "on", "repair"}
FALSE_VALUES = {"0", "false", "no", "off", "none"}


def _repair_root(manager: "ConfigManager") -> Dict[str, Any]:
    default = ((manager.load_default().get("heavenbase", {}) or {}).get("llm", {}) or {}).get("tool_call_repair", {}) or {}
    runtime = manager.get("heavenbase.llm.tool_call_repair", default={}) or {}
    return resolve_interpolations(dmerge([default, runtime]))


def _enabled(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return bool(value.get("enabled", default))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_VALUES:
            return True
        if lowered in FALSE_VALUES:
            return False
    return bool(value)


def _tool_schema_map(tools: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for tool in list(tools or []):
        if not isinstance(tool, Mapping):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
        name = fn.get("name")
        if name:
            out[str(name)] = dict(fn.get("parameters") or {})
    return out


def _strip_fence(text: str) -> str:
    value = text.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return value.strip("`").strip()


def _balanced_json(text: str) -> str:
    stack: List[str] = []
    out: List[str] = []
    in_str = False
    escaped = False
    for char in text:
        if in_str:
            if escaped:
                out.append(char)
                escaped = False
            elif char == "\\":
                out.append(char)
                escaped = True
            elif char == '"':
                out.append(char)
                in_str = False
            elif char == "\n":
                out.append("\\n")
            elif char == "\r":
                out.append("\\r")
            elif char == "\t":
                out.append("\\t")
            else:
                out.append(char)
            continue
        if char == '"':
            out.append(char)
            in_str = True
        elif char == "{":
            out.append(char)
            stack.append("}")
        elif char == "[":
            out.append(char)
            stack.append("]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
                out.append(char)
        else:
            out.append(char)
    if escaped:
        out.append("\\")
    if in_str:
        out.append('"')
    out.extend(reversed(stack))
    return "".join(out)


def _schema_default(schema: Mapping[str, Any]) -> Any:
    if "default" in schema:
        return deepcopy(schema.get("default"))
    typ = schema.get("type")
    typ = typ[0] if isinstance(typ, list) and typ else typ
    return {"string": "", "number": 0, "integer": 0, "boolean": False, "array": [], "object": {}}.get(str(typ), None)


def _fill_required(data: Any, schema: Mapping[str, Any]) -> Any:
    if not isinstance(data, dict) or not isinstance(schema, Mapping):
        return data
    props = schema.get("properties") or {}
    for name in schema.get("required") or []:
        if name not in data:
            data[name] = _schema_default(props.get(name, {}) if isinstance(props, Mapping) else {})
    return data


class LLMToolCallRepair:
    """Own configuration and repair of OpenAI-compatible tool-call arguments.

    Args:
        config (Mapping[str, Any] | None): Repair configuration merged over
            `heavenbase.llm.tool_call_repair` defaults.
        config_manager (ConfigManager | None): Configuration authority. Omission selects
            the default Context's config.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None, *, config_manager: "ConfigManager | None" = None) -> None:
        """Initialize this repair service from configuration.

        Returns:
            None: This function does not return a value.
        """
        if config_manager is None:
            from heavenbase.context import default_context

            config_manager = default_context().config
        self._config_manager_ref = weakref.ref(config_manager)
        self.config = dmerge([_repair_root(config_manager), dict(config or {})])

    @property
    def config_manager(self) -> "ConfigManager":
        """Return the live configuration authority for this service."""
        manager = self._config_manager_ref()
        if manager is None:
            raise RuntimeError("LLM tool-call repair authority is no longer available")
        return manager

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]] = None, *, config_manager: "ConfigManager | None" = None, **values: Any) -> "LLMToolCallRepair":
        """Build a repair service from configuration values.

        Args:
            data (Mapping[str, Any] | None): Base repair configuration.
            config_manager (ConfigManager | None): Configuration authority. Omission selects
                the default Context's config.
            **values (Any): Configuration values that override `data`.

        Returns:
            LLMToolCallRepair: Configured repair service.
        """
        cfg = dict(data or {})
        cfg.update(values)
        return cls(cfg, config_manager=config_manager)

    def to_dict(self) -> Dict[str, Any]:
        """Return this repair service configuration.

        Args:
            None.

        Returns:
            Dict[str, Any]: Independent configuration dictionary.
        """
        return dict(self.config)

    @classmethod
    def current(cls, *, config_manager: "ConfigManager | None" = None) -> "LLMToolCallRepair":
        """Return the repair service owned by one configuration authority.

        Args:
            config_manager (ConfigManager | None): Configuration authority. Omission selects
                the default Context's config.

        Returns:
            LLMToolCallRepair: Lazily constructed authority-local service.
        """
        if config_manager is None:
            from heavenbase.context import default_context

            config_manager = default_context().config
        with _LLM_TOOL_CALL_REPAIR_LOCK:
            found = _LLM_TOOL_CALL_REPAIRS.get(config_manager)
            if found is None:
                found = cls(config_manager=config_manager)
                _LLM_TOOL_CALL_REPAIRS[config_manager] = found
            return found

    @classmethod
    def reset(cls) -> None:
        """Reset the process repair service.

        Args:
            None.

        Returns:
            None: This method does not return a value.
        """
        with _LLM_TOOL_CALL_REPAIR_LOCK:
            _LLM_TOOL_CALL_REPAIRS.clear()

    def enabled(self, option: Any = None) -> bool:
        """Return whether repair should run for one call.

        Args:
            option (Any): Optional per-call override. Booleans, recognized
                true/false strings, and mappings with `enabled` are supported.

        Returns:
            bool: Whether `repair` should modify malformed arguments.
        """
        return _enabled(option, bool(self.config.get("enabled", False)))

    def repair(self, tool_calls: Iterable[Mapping[str, Any]], tools: Any = None, option: Any = None) -> List[Dict[str, Any]]:
        """Copy tool calls and repair their argument strings when enabled.

        Args:
            tool_calls (Iterable[Mapping[str, Any]]): OpenAI-compatible tool calls.
            tools (Any): Optional tool definitions used to fill required arguments.
            option (Any): Optional per-call enablement override.

        Returns:
            List[Dict[str, Any]]: Independent tool-call dictionaries in input order.

        Raises:
            ValueError: If strict repair is enabled and an argument string cannot be parsed.
        """
        calls = [deepcopy(dict(call)) for call in tool_calls]
        if not self.enabled(option):
            return calls
        schemas = _tool_schema_map(tools)
        return [self._repair_call(call, schemas.get(((call.get("function") or {}).get("name") or ""))) for call in calls]

    def _repair_call(self, call: Mapping[str, Any], schema: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Return one repaired tool call."""
        out = deepcopy(dict(call))
        fn = out.get("function") if isinstance(out.get("function"), dict) else {}
        args = fn.get("arguments", "")
        fn["arguments"] = self._repair_args(args, schema=schema or {})
        out["function"] = fn
        return out

    def _repair_args(self, arguments: Any, schema: Optional[Mapping[str, Any]] = None) -> str:
        """Return a JSON argument string repaired from common provider mistakes."""
        original = "" if arguments is None else str(arguments)
        candidates = []
        for candidate in (original, _strip_fence(original)):
            if candidate not in candidates:
                candidates.append(candidate)
        for candidate in list(candidates):
            fixed = _balanced_json(candidate)
            if fixed not in candidates:
                candidates.append(fixed)
        for candidate in candidates:
            try:
                data = loads_json(candidate)
                data = _fill_required(data, schema or {})
                return dumps_json(data, compact=True, separators=(",", ":"))
            except Exception:
                continue
        if self.config.get("strict"):
            raise ValueError(f"Unable to repair LLM tool-call arguments: {original}")
        return original


_LLM_TOOL_CALL_REPAIRS: weakref.WeakKeyDictionary["ConfigManager", LLMToolCallRepair] = weakref.WeakKeyDictionary()
_LLM_TOOL_CALL_REPAIR_LOCK = threading.RLock()
