"""Transport-neutral application service for the HeavenBase dashboard."""

from __future__ import annotations

__all__ = ["DashboardAPI"]

import difflib
from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import date, datetime, time
from decimal import Decimal
import math
import os
import re
import threading
import warnings

from heavenbase.utils.config import ConfigManager, dflat, dget, dmerge
from heavenbase.utils.config.api import ConfigOps
from heavenbase.utils.values import raise_mismatch
from heavenbase.utils.redact import redact_data
from heavenbase.utils.serialize import dumps_yaml

# One conservative fallback accounting rate is shared with the LLM loop. Real
# provider usage remains authoritative whenever it is reported.
from heavenbase.llm.base import _CHARS_PER_TOKEN, _merge_usage
from heavenbase.utils.hash import sha256hash
from heavenbase.utils.serialize import dumps_jstr
from heavenbase.utils.typing import Any, Dict, Generator, List, Optional, Tuple

from heavenbase.query.summary import stat_kind

# Options that ``LLM(...)`` reads only as keywords: the route selectors that the LLM config resolves the route
# from, its authority, cache and repair options, and its `args` mapping.
_LLM_KEYWORD_OPTIONS = frozenset(
    {
        "preset",
        "model",
        "provider",
        "gateway",
        "endpoint",
        "desc",
        "args",
        "context",
        "resolver",
        "config",
        "engine",
        "cache",
        "cache_config",
        "tool_call_repair",
    }
)

MAX_PAGE_SIZE = 1000
_ASSIST_STREAM_STOP: ContextVar[Optional[threading.Event]] = ContextVar("assist_stream_stop", default=None)
# Conservative dashboard defaults preserve the existing bounded interaction;
# operators can opt into larger evidence and context budgets in config.
_MAX_ASSIST_FRAMES = 5
_DEFAULT_MAX_ASSIST_RESULT_CHARS = 6000
_MAX_ASSIST_TOOL_TURNS = 8
# Prior turns replayed to the model. The client posts the whole transcript, so
# anything older is dropped — and the model is told, rather than left to assume
# it saw the whole conversation. Turn count stopped being the limiter when
# replay is also bounded by the client's matching storage ceiling.
_ASSIST_HISTORY_TURNS = 40
# Cross-turn evidence carry. Replayed turns are prose only, so the tool results
# a turn was grounded in are gone by the next question; while ``evidence_total``
# grants a budget, each turn's executed calls are cached per service instance
# keyed by the answer text the client replays, and re-attached to replayed
# turns newest-first — the full rendering while the budget allows, else the
# brief one, else nothing.
_ASSIST_EVIDENCE_TOTAL = 0
# Single tool results are not clipped at capture by default; the per-turn tail
# cap is the memory backstop when cross-turn evidence carry is enabled.
_ASSIST_EVIDENCE_TURN_CHARS = 200000
_ASSIST_EVIDENCE_BRIEF_CHARS = 1500
# Measured replay: while the assembled request — system
# prompt, windowed history with FULL evidence, question — measures under
# ``history_chars``, everything ships full with zero degradation. Over it, the
# oldest turns are compacted in one model call into at most ``compact_chars``
# of conversation memory folded into the system notice, and the remaining
# suffix still ships full. The per-turn ladder above survives only as the
# fallback (threshold 0, injected runtimes, or a failed compaction).
# Share of the model window the assembled history may hold before the oldest
# turns are compacted: 55% for history leaves the current turn ~37% of the
# window before the llm loop's own in-turn compactor engages at 92%. An
# explicit ``history_chars`` config value overrides in absolute characters
# (0 keeps the legacy per-turn ladder).
_ASSIST_HISTORY_WINDOW_SHARE = 0.55
# An explicit model ``context_window`` wins; the fallback protects routes whose
# registry metadata does not declare one.
_ASSIST_CONTEXT_WINDOW_FALLBACK = 128000
_ASSIST_COMPACT_CHARS = 4000
# Generic LLM behavior remains unchanged unless the dashboard opts in.
_ASSIST_RESULT_CAPTURE_CHARS = 0
_ASSIST_COMPACT_RATIO = 0.92
# Boilerplate reserve for the notice text around the summary block, so the
# post-compaction assembly still measures under the threshold.
_ASSIST_COMPACT_OVERHEAD = 400
_ASSIST_COMPACT_CACHE_ENTRIES = 8
# Attach-everything budget for the measured path: the threshold, not a
# per-turn ladder budget, governs total size there.
_ASSIST_ATTACH_ALL = 10**12
_ASSIST_EVIDENCE_BRIEF_ARGS = 200
_ASSIST_EVIDENCE_BRIEF_RESULT = 400
_ASSIST_EVIDENCE_ENTRIES = 40
_ASSIST_EVIDENCE_HEADER = "[tool evidence from this turn]"
_DASHBOARD_PREFERENCES_PATH = "heavenbase.dashboard.preferences"
_DASHBOARD_ASSISTANT_PROFILE_META = "dashboard_assistant"
_DASHBOARD_PREFERENCE_DEFAULTS = {
    "theme": "theme-light",
    "language": "en",
    "font": "default",
    "font_scale": "default",
}
_DASHBOARD_PREFERENCE_VALUES = {
    "theme": {"theme-light", "theme-dark", "theme-anthropic"},
    "language": {"en", "zh"},
    "font": {"small", "default", "large"},
    "font_scale": {"small", "default", "large", "xlarge"},
}
_DEFAULT_ASSIST_QUERY_LIMIT = 50
# Row reads stay tightly bounded because they stream source rows into the
# model. Aggregate results are the answer itself — aggregation and HAVING run
# before limit (ADR 0009), so the same cap would silently drop result groups.
_MAX_ASSIST_QUERY_ROWS = 100
_MAX_ASSIST_GROUP_ROWS = 1000
_REMOVE_MARKER = "__HB_REMOVE__"
_OMIT = object()
_ENV_REFERENCE = re.compile(r"^\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,.*)?\}$", re.DOTALL)


def _query_uses_code_reduce(spec: Any, resolver: Any) -> bool:
    """Return whether a query spec names ``code_reduce`` in any spelling the planner runs.

    The planner reads every operation name through ``normalize_op``, which
    ignores case and surrounding space and resolves registered aliases such as
    ``$code_reduce``. This check resolves every mapping key the same way, at any
    depth, and the value of every ``op`` key, where a Json ``get`` filter names
    its comparison. JSON text is decoded and walked too, whether it is the whole
    spec, which the ``query`` tool decodes, or a nested value, which the planner
    rejects today. Other values are data.

    Args:
        spec (Any): Query spec as the caller wrote it: a mapping, a sequence,
            JSON text, or ``None``.
        resolver (Any): Module resolver of the workspace that would execute
            ``spec``.

    Returns:
        bool: Whether an operation name in ``spec`` resolves to ``code_reduce``.
    """
    from heavenbase.filters.ops import normalize_op
    from heavenbase.utils.serialize import loads_json

    def names_code_reduce(name: Any) -> bool:
        try:
            return normalize_op(str(name), resolver=resolver) == "code_reduce"
        except (LookupError, ValueError):  # not an operation name
            return False

    pending: List[Any] = [spec]
    while pending:
        node = pending.pop()
        if isinstance(node, str):
            text = node.strip()
            if text[:1] in ("{", "[", '"'):
                try:
                    # Plain JSON holds every key that restoring HeavenBase's tagged values would keep.
                    pending.append(loads_json(text, restore=False))
                except (ValueError, RecursionError):  # not JSON, so no query tool decodes it into a spec
                    pass
        elif isinstance(node, Mapping):
            for key, value in node.items():
                if names_code_reduce(key) or (key == "op" and names_code_reduce(value)):
                    return True
                pending.append(value)
        elif isinstance(node, (list, tuple, set, frozenset)):
            pending.extend(node)
    return False


def _drop_config_tombstones(value: Any) -> Any:
    """Return a display copy of a config layer without internal delete markers."""
    if isinstance(value, dict):
        out = {}
        removed = False
        for key, child in value.items():
            if key == _REMOVE_MARKER and child is True:
                removed = True
                continue
            cleaned = _drop_config_tombstones(child)
            if cleaned is _OMIT:
                removed = True
                continue
            out[key] = cleaned
        if value.get(_REMOVE_MARKER) is True and not out:
            return _OMIT
        if removed and not out:
            return _OMIT
        return out
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _drop_config_tombstones(item)) is not _OMIT]
    return value


def _display_config_layer(cm: ConfigManager, scope: str, version: Optional[int]) -> Dict[str, Any]:
    """Return one config layer as it should appear in dashboard diffs."""
    raw = cm.layer(scope=scope, version=version) if version is not None else {}
    cleaned = _drop_config_tombstones(raw)
    return {} if cleaned is _OMIT else cleaned


def _resolve_config_scope(cm: ConfigManager, scope: Optional[str] = None) -> Optional[str]:
    """Resolve a short dashboard scope token to a complete config scope."""
    if not scope:
        return None
    value = scope.strip().lower()
    if value == ".":
        return cm.base_scope
    return value if value == cm.base_scope or value.startswith(f"{cm.base_scope}.") else f"{cm.base_scope}.{value}"


def _resolve_config_version(cm: ConfigManager, scope: str, version: Optional[int]) -> Optional[int]:
    """Resolve a possibly negative retained-version index."""
    if version is None or version >= 0:
        return version
    versions = sorted(int(row["ver"]) for row in cm.history(scope=scope, limit=1000))
    if not versions:
        return None
    try:
        return versions[version]
    except IndexError:
        return versions[0]


def _with_history_notice(system: str, history: Any, turns: int = _ASSIST_HISTORY_TURNS) -> str:
    """Append a disclosure when older conversation turns were not replayed.

    The client posts the whole transcript and the server keeps only the most
    recent window, so without this the model receives a conversation that looks
    complete and summarises it as if it were.

    Args:
        system (str): Assembled system prompt.
        history (Any): Prior turns supplied by the client.
        turns (int): Replay window actually applied, so the notice states the
            count the slice used. Defaults to ``_ASSIST_HISTORY_TURNS``.

    Returns:
        str: The prompt, with a truncation notice when turns were dropped.
    """
    dropped = max(0, len(history or []) - turns)
    if not dropped:
        return system
    return (
        f"{system}\nOnly the last {turns} of {len(history)} prior turns are in your context; "
        f"{dropped} earlier turns exist but were not sent. Say so rather than implying you have seen the whole conversation."
    )


def _counting_usage(trace: Any, usages: List[Mapping[str, Any]]) -> Any:
    """Return ``trace`` that also keeps the usage of every provider response it sees in ``usages``.

    Every provider call a logged assistant turn makes through the built-in runtime (each tool-loop round, the
    in-turn compaction summary and the history compaction summary) reports its usage in a ``response`` event.
    Collecting those gives a successful attempt's usage, whichever path made the call. A response counts when it
    arrives, before ``trace`` writes its row, so it counts even when that write fails.
    """

    def _count(event: Any) -> None:
        if event.kind == "response":
            usages.append(event.payload.get("usage") or {})
        trace(event)

    return _count


def _bounded_assist_result(value: Any, max_chars: int = _DEFAULT_MAX_ASSIST_RESULT_CHARS) -> str:
    """Bound one tool result without emitting malformed truncated JSON."""
    from heavenbase.utils.serialize import dumps_json, loads_json

    text = "" if value is None else str(value)
    limit = max(256, int(max_chars))
    if len(text) <= limit:
        return text
    try:
        loads_json(text)
    except (TypeError, ValueError):
        suffix = f"\n… [result truncated; original length {len(text)} characters]"
        return f"{text[: max(0, limit - len(suffix))]}{suffix}"
    preview = text[: max(32, limit // 2)]
    payload = dumps_json({"truncated": True, "characters": len(text), "preview": preview}, compact=True)
    while len(payload) > limit and len(preview) > 32:
        preview = preview[: max(32, len(preview) // 2)]
        payload = dumps_json({"truncated": True, "characters": len(text), "preview": preview}, compact=True)
    return payload


def _assist_int_setting(cm: ConfigManager, key: str, default: int, low: int, high: int) -> int:
    """Read one bounded integer assistant setting, failing loudly on bad values.

    Args:
        cm (ConfigManager): Config authority the dashboard reads.
        key (str): Dotted config key, e.g. ``"heavenbase.dashboard.assistant.max_frames"``.
        default (int): Value used when the key is unset.
        low (int): Smallest accepted value.
        high (int): Largest accepted value.

    Returns:
        int: Validated setting value.

    Raises:
        TypeError: If the configured value is a bool or not convertible to ``int``.
        ValueError: If the configured value falls outside ``[low, high]``.
    """
    value = cm.get(key, default=default)
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{key} must be an integer") from exc
    if limit < low or limit > high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return limit


def _bounded_query_result(result: Dict[str, Any], limit: int, cap: int, aggregate: bool) -> Dict[str, Any]:
    """State any truncation in the payload, using the engine's exact total.

    Query execution paginates last, on a complete result, so a truncated frame
    reports how many rows or groups actually matched. The tool passes that
    total through verbatim: the model is told the exact size of what it did not
    receive, never a floor it might report as a total.

    Args:
        result (Dict[str, Any]): Frame payload from the workspace ``query`` tool.
        limit (int): Effective row or group limit applied to this call.
        cap (int): Configured maximum the caller could have asked for.
        aggregate (bool): Whether the spec aggregates, making result groups the
            unit that gets truncated.

    Returns:
        Dict[str, Any]: The payload with ``limit`` and ``truncated`` always set,
        plus a ``note`` naming the withheld count when one happened.
    """
    truncated = bool(result.get("truncated"))
    if truncated:
        unit = "groups" if aggregate else "rows"
        total = result.get("total")
        hint = f"raise limit up to {cap}" if limit < cap else f"narrow the query with filter/having — the dashboard cap is {cap} {unit}"
        returned = int(result.get("n") or 0)
        result["note"] = f"TRUNCATED: {total} {unit} matched and only {returned} are returned; report {total} as the count, not {returned} ({hint})."
    result["truncated"] = truncated
    result["limit"] = limit
    return result


def _assist_query_args(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize serialized query-call arguments for capture matching."""
    from heavenbase.utils.serialize import loads_json

    try:
        payload = loads_json(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or "entity" not in payload:
        return None
    normalized = dict(payload)
    if normalized.get("spec") is None:
        normalized.pop("spec", None)
    return normalized


def _assist_tool_args(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize serialized tool arguments for evidence-frame matching."""
    from heavenbase.utils.serialize import loads_json

    try:
        payload = loads_json(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _assist_query_frame(
    arguments: Any,
    captures: List[Dict[str, Any]],
    used: set[int],
) -> Optional[Dict[str, Any]]:
    """Return the frame captured by the matching executed query call."""
    call_args = _assist_query_args(arguments)
    if call_args is None:
        return None
    for index, capture in enumerate(captures):
        if index not in used and capture.get("args") == call_args:
            used.add(index)
            frame = capture.get("frame")
            return frame if isinstance(frame, dict) else None
    return None


def _assist_evidence_frame(
    name: str,
    arguments: Any,
    captures: List[Dict[str, Any]],
    used: set[int],
) -> Optional[Dict[str, Any]]:
    """Return the structured evidence captured by one matching tool call."""
    call_args = _assist_tool_args(arguments)
    if call_args is None:
        return None
    for index, capture in enumerate(captures):
        if index not in used and capture.get("name") == name and capture.get("args") == call_args:
            used.add(index)
            frame = capture.get("frame")
            return frame if isinstance(frame, dict) else None
    return None


def _assist_json_value(value: Any) -> Any:
    """Detach one tool value into a lossless JSON-safe dashboard value."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value if -(1 << 53) < value < (1 << 53) else str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _assist_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_assist_json_value(item) for item in value]
    return str(value)


def _assist_sql_frame(arguments: Mapping[str, Any], result: Any, spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Build one lossless SQL execution frame from profile-declared arguments."""
    if not isinstance(result, Mapping):
        return None
    statement_arg = str(spec.get("statement_arg") or "").strip()
    source_arg = str(spec.get("source_arg") or "").strip()
    params_arg = str(spec.get("params_arg") or "").strip()
    statement = arguments.get(statement_arg)
    rows = result.get("rows")
    columns = result.get("columns")
    if not statement_arg or not source_arg or not isinstance(statement, str) or not statement.strip():
        return None
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return None
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        return None
    params = arguments.get(params_arg) if params_arg else None
    return {
        "database_id": str(arguments.get(source_arg) or result.get("database_id") or ""),
        "sql": statement.strip(),
        "params": _assist_json_value(dict(params)) if isinstance(params, Mapping) else {},
        "columns": list(columns),
        "rows": [_assist_json_value(dict(row)) for row in rows],
        "n": int(result.get("n") or len(rows)),
        "total": int(result.get("total")) if result.get("total") is not None else None,
        "limit": int(result.get("limit")) if result.get("limit") is not None else None,
        "truncated": bool(result.get("truncated")),
        "readonly": bool(result.get("readonly", True)),
    }


def _evidenced(tool: Any, captures: Optional[List[Dict[str, Any]]], spec: Mapping[str, Any]) -> Any:
    """Wrap one profile tool so its declared structured evidence is retained."""
    if captures is None:
        return tool
    import functools
    import inspect

    kind = str(spec.get("kind") or "").strip()
    if kind != "sql":
        raise ValueError(f"Unsupported dashboard assistant evidence frame kind {kind!r}")
    signature = inspect.signature(tool)

    @functools.wraps(tool)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        call_args = dict(bound.arguments)
        capture = {"name": str(getattr(tool, "name", getattr(tool, "__name__", ""))), "args": call_args, "frame": None}
        captures.append(capture)
        result = tool(*args, **kwargs)
        capture["frame"] = _assist_sql_frame(call_args, result, spec)
        return result

    return wrapped


def _assist_events(
    delta: Optional[List[Dict[str, Any]]],
    frames: List[Dict[str, Any]],
    max_result_chars: int = _DEFAULT_MAX_ASSIST_RESULT_CHARS,
    evidence_frames: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build the interleaved think/tool timeline from one reply's delta messages.

    Args:
        delta (List[Dict[str, Any]] | None): Ordered OpenAI-format messages
            produced by the call (assistant rounds with ``reasoning_content``
            and ``tool_calls``, followed by ``tool`` result messages).
        frames (List[Dict[str, Any]]): Executed-query capture records containing
            normalized call arguments and an optional lossless frame.

    Returns:
        List[Dict[str, Any]]: Events in execution order — ``{"type": "think",
        "text"}`` or ``{"type": "tool", "name", "args"}`` with an attached
        ``frame`` (query calls) or truncated ``result`` text (other tools).
    """
    events: List[Dict[str, Any]] = []
    matched_frames: set[int] = set()
    matched_evidence: set[int] = set()
    by_call_id: Dict[str, Dict[str, Any]] = {}
    for message in delta or []:
        role = message.get("role")
        if role == "assistant":
            think = message.get("reasoning_content")
            if think:
                events.append({"type": "think", "text": str(think)})
            content = message.get("content")
            if content and message.get("tool_calls"):
                # a spoken preamble before calling tools is narrative, not the answer
                events.append({"type": "think", "text": str(content)})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                event: Dict[str, Any] = {"type": "tool", "name": str(function.get("name") or ""), "args": str(function.get("arguments") or "")[:400]}
                if event["name"] == "query":
                    frame = _assist_query_frame(function.get("arguments"), frames, matched_frames)
                    if frame is not None:
                        event["frame"] = frame
                else:
                    evidence = _assist_evidence_frame(event["name"], function.get("arguments"), evidence_frames or [], matched_evidence)
                    if evidence is not None:
                        event["sql_frame"] = evidence
                events.append(event)
                if call.get("id"):
                    by_call_id[str(call["id"])] = event
        elif role == "tool":
            event = by_call_id.get(str(message.get("tool_call_id") or ""))
            if event is not None:
                result = message.get("content") or ""
                if "frame" not in event and "sql_frame" not in event:
                    event["result"] = _bounded_assist_result(result, max_result_chars)
    return events


def _narrated(tool: Any, progress: Optional[List[Dict[str, Any]]]) -> Any:
    """Wrap one assist tool so each invocation appends a live activity line.

    Args:
        tool (Any): Tool callable exposed to the assistant.
        progress (List[Dict[str, Any]] | None): Live feed sink; ``None`` disables wrapping.

    Returns:
        Any: The original tool, or a signature-preserving wrapper that records
        ``name(args)`` lines before delegating.
    """
    if progress is None:
        return tool
    import functools

    @functools.wraps(tool)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        parts = [str(value) for value in args] + [f"{key}={value}" for key, value in kwargs.items()]
        progress.append({"type": "tool", "text": f"{tool.__name__}({', '.join(parts)})"[:160]})
        return tool(*args, **kwargs)

    return wrapped


def _recorded(tool: Any, sink: Optional[List[Tuple[str, str, str]]]) -> Any:
    """Wrap one assist tool so each completed call lands in the evidence sink.

    Args:
        tool (Any): Tool callable exposed to the assistant (possibly narrated).
        sink (List[Tuple[str, str, str]] | None): Per-turn ``(name, args,
            result)`` capture list; ``None`` disables wrapping.

    Returns:
        Any: The original tool, or a signature-preserving wrapper that records
        the rendered call after delegating. Capture is best-effort: a call
        whose arguments or result cannot be rendered is skipped, never
        surfaced as a tool error.
    """
    if sink is None:
        return tool
    import functools

    @functools.wraps(tool)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = tool(*args, **kwargs)
        try:
            # ``session_id`` is MCP plumbing, not evidence the model should reread.
            parts = [str(value) for value in args] + [f"{key}={value}" for key, value in kwargs.items() if key != "session_id"]
            sink.append((str(getattr(tool, "name", getattr(tool, "__name__", ""))), ", ".join(parts), str(result)))
        except Exception:
            pass  # an unrenderable argument or result skips the capture, never the tool
        return result

    return wrapped


def _assist_tool_read_only(tool: Any) -> bool:
    """Return whether a tool explicitly declares read-only behavior."""
    annotations = getattr(tool, "annotations", None)
    annotation_dump = getattr(annotations, "model_dump", None)
    if callable(annotation_dump):
        annotations = annotation_dump(by_alias=True, exclude_none=True)
    metadata = getattr(tool, "metadata", None)
    return (
        getattr(tool, "read_only", False) is True
        or (isinstance(annotations, Mapping) and annotations.get("readOnlyHint") is True)
        or (isinstance(metadata, Mapping) and metadata.get("read_only") is True)
    )


def _assist_mcp_source(spec: Mapping[str, Any]) -> Any:
    """Build a FastMCP client source from one dashboard server specification."""
    command = spec.get("command")
    url = spec.get("url")
    if bool(command) == bool(url):
        raise ValueError("requires exactly one of command or url")
    if url:
        if not isinstance(url, str):
            raise TypeError("url must be a string")
        return url
    if isinstance(command, str):
        executable = command.strip()
        args = spec.get("args", [])
    elif isinstance(command, (list, tuple)) and command:
        if "args" in spec:
            raise ValueError("args must not be supplied when command is an argv list")
        executable = str(command[0]).strip()
        args = list(command[1:])
    else:
        raise TypeError("command must be a string or non-empty argv list")
    if not executable or not isinstance(args, (list, tuple)):
        raise TypeError("command requires an executable and an argument list")
    environment = spec.get("env")
    if environment is not None and not isinstance(environment, Mapping):
        raise TypeError("env must be a mapping")
    cwd = spec.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise TypeError("cwd must be a string")
    import fastmcp
    from fastmcp.client.transports import StdioTransport

    return StdioTransport(
        executable,
        [str(value) for value in args],
        env={str(key): str(value) for key, value in environment.items()} if environment is not None else None,
        cwd=cwd,
        keep_alive=False,
    )


def _assist_mcp_tool(tool: Any, server: str, read_only: bool) -> Any:
    """Namespace one runtime MCP callable for the dashboard assistant."""
    import functools

    @functools.wraps(tool)
    def call_mcp_tool(**arguments: Any) -> Any:
        return tool(**arguments)

    native_name = str(getattr(tool, "__name__", "") or "")
    call_mcp_tool.__name__ = re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp__{server}__{native_name}")
    call_mcp_tool.__tool_schema__ = dict(getattr(tool, "__tool_schema__", None) or {})
    call_mcp_tool.read_only = read_only
    return call_mcp_tool


def _assist_config_fingerprint(value: Any) -> str:
    """Return a stable warning key even for malformed non-JSON config."""
    try:
        serialized = dumps_jstr(value, compact=True)
    except Exception:
        try:
            serialized = repr(value)
        except Exception:
            serialized = f"<{type(value).__module__}.{type(value).__qualname__}>"
    return sha256hash(serialized)[:16]


def _capture_clipped_frame(frame: Dict[str, Any], result: Any, cap_chars: int) -> Dict[str, Any]:
    """Align one query frame with the model's capture-clipped view.

    When the serialized tool result overflowed the model-side capture clip,
    the model saw a prefix, not the whole payload — and the window presents
    the same view in both directions. Rows are kept while their serialized forms still fit inside
    the clip: a conservative reading of the prefix the model received, stated
    through the ``truncated``/``note`` fields the GUI already renders.

    Args:
        frame (Dict[str, Any]): Captured query frame, mutated in place.
        result (Any): The tool result the model-side message serialized.
        cap_chars (int): ``result_capture_chars`` for this turn; ``0``
            leaves the frame untouched.

    Returns:
        Dict[str, Any]: The frame, clipped and annotated when the model was.
    """
    if not cap_chars:
        return frame
    serialized = result if isinstance(result, str) else dumps_jstr(result, compact=True)
    if len(serialized) <= cap_chars:
        return frame
    rows = list(frame.get("rows") or [])
    used = len(serialized) - sum(len(dumps_jstr(row, compact=True)) + 2 for row in rows)  # envelope + separators
    kept: List[Any] = []
    for row in rows:
        cost = len(dumps_jstr(row, compact=True)) + 2
        if used + cost > cap_chars:
            break
        kept.append(row)
        used += cost
    frame["rows"] = kept
    frame["truncated"] = True
    frame["note"] = (
        f"CAPTURE-CLIPPED: the serialized result ran {len(serialized)} characters against the {cap_chars}-character "
        f"model-side clip; the model saw only a prefix, so only the first {len(kept)} of {len(rows)} fetched rows are shown."
    )
    return frame


def _assist_evidence_key(answer: str, question: str = "", scope: str = "") -> str:
    """Return the evidence-store key for one scoped conversation turn."""
    return sha256hash(dumps_jstr([scope, question.strip(), answer.strip()], compact=True))[:32]


def _assist_evidence_tail(lines: List[str], cap: int) -> str:
    """Join the newest lines that fit under ``cap`` characters.

    Lines are kept from the tail because the freshest evidence grounds the
    next question best; a single over-budget line keeps its head so one huge
    result cannot blank the whole rendering.
    """
    kept: List[str] = []
    used = 0
    for line in reversed(lines):
        cost = len(line) + (1 if kept else 0)
        if used + cost > cap:
            if not kept:
                kept.append(line[:cap])
            break
        kept.append(line)
        used += cost
    kept.reverse()
    return "\n".join(kept)


class DashboardAPI:
    """Dashboard service methods over existing config and workspace surfaces.

    One instance backs one dashboard server process. Config methods wrap
    the Context-owned ``ConfigManager``/``ConfigOps`` and edit scoped layers
    only. Opening a workspace persists that selection through the same Context
    so every local dashboard host resumes it.

    Examples:
        >>> import heavenbase as hb
        >>> from heavenbase.api import DashboardAPI
        >>> api = DashboardAPI()
        >>> api.workspaces()  # doctest: +SKIP
        [{'id': 'heavenbase', ...}]
    """

    def __init__(self, context: Any = None) -> None:
        """Initialize the dashboard service.

        Args:
            context: Context owning config, modules, and workspaces. Defaults
                to the process Context.

        Returns:
            None: This function does not return a value.
        """
        from ..context import context_or_default

        self.context = context_or_default(context)
        self.cm = self.context.config
        self.config_ops = ConfigOps(self.cm)
        self.ws: Any = None
        self.ws_id: Optional[str] = None
        self._workspace_lock = threading.RLock()
        self._assist_progress: Dict[str, List[Dict[str, Any]]] = {}
        # Cross-turn evidence carry: per-instance so parallel dashboards never
        # share turns, bounded at _ASSIST_EVIDENCE_ENTRIES by oldest eviction.
        self._assist_evidence: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()
        self._assist_evidence_lock = threading.Lock()
        # Compacted-prefix summaries, keyed by the prefix turns' content keys:
        # a stable prefix pays its one compaction call exactly once.
        self._assist_compact_cache: "OrderedDict[Tuple[str, ...], str]" = OrderedDict()
        self._assist_compact_lock = threading.Lock()
        # Invalid external MCP definitions warn once per instance and exact
        # config fingerprint; a corrected definition gets a fresh warning key.
        self._assist_mcp_warned: set[str] = set()
        self._assist_mcp_lock = threading.Lock()
        # Durable session log; constructed on first enabled turn so a
        # disabled config costs nothing.
        self._assist_session_log: Any = None
        self._assist_session_log_lock = threading.Lock()

    # ------------------------------------------------------------------- meta

    def meta(self) -> Dict[str, Any]:
        """Return dashboard application metadata for the About dialog.

        Args:
            None.

        Returns:
            Dict[str, Any]: Package name and running HeavenBase version.
        """
        from heavenbase.version import __version__

        return {"name": "HeavenBase", "version": __version__}

    # ------------------------------------------------------------------ config

    def config_meta(self) -> Dict[str, Any]:
        """Return config store metadata for the workbench shell.

        Args:
            None.

        Returns:
            Dict[str, Any]: Scopes, active scope chain, per-scope versions,
            and the config resource root.
        """
        chain = list(self.cm.scope_chain)
        return {
            "scopes": self.cm.scopes(),
            "scope": self.cm.scope,
            "scope_chain": chain,
            "versions": dict(zip(chain, self.cm.version_chain)),
            "root": self.cm.root,
        }

    def config_tree(self, scope: Optional[str] = None, version: Optional[int] = None) -> Dict[str, Any]:
        """Return the merged raw config tree with per-leaf origins.

        Layers are merged without interpolation so ``${env:...}`` references
        stay as written and secrets are never resolved for display.

        Args:
            scope (str | None): Explicit scope whose raw layer should be
                merged on top of the defaults. ``None`` inspects the active
                scope when a ``version`` is requested, otherwise merges the
                full active scope chain. Defaults to ``None``.
            version (int | None): Retained version of the target scope to
                inspect; negative values index back from the latest. When set,
                a single scope is inspected (the active scope if ``scope`` is
                ``None``) rather than the merged chain. Defaults to ``None``
                (latest).

        Returns:
            Dict[str, Any]: ``{"tree", "origins", "scope", "version"}`` where
            ``origins`` maps flattened (dot-escaped) leaf keys to ``builtin``
            (matches the default layer), ``override`` (default key changed by a
            user layer), or ``custom`` (absent from the default layer). ``scope``
            reports the inspected scope and ``version`` the resolved version.
        """
        default = self.cm.load_default()
        resolved = _resolve_config_scope(self.cm, scope)
        if version is not None:
            target = resolved or self.cm.scope
            resolved_version = _resolve_config_version(self.cm, target, version)
            layers = [{} if resolved_version is None else self.cm.layer(scope=target, version=resolved_version)]
        elif resolved:
            target = resolved
            resolved_version = None
            layers = [self.cm.layer(scope=resolved)]
        else:
            target = self.cm.scope
            resolved_version = None
            layers = [self.cm.layer(scope=chain_scope) for chain_scope in self.cm.scope_chain]
        tree = dmerge(layers, start=default)
        default_flat = dict(dflat(default))
        origins = {key: "custom" if key not in default_flat else ("builtin" if value == default_flat[key] else "override") for key, value in dflat(tree)}
        return {"tree": tree, "origins": origins, "scope": target, "version": resolved_version}

    def config_value_status(self, key: str, reveal: bool = False) -> Dict[str, Any]:
        """Report the resolved value and whether its environment override won.

        Args:
            key (str): Flattened, dot-escaped configuration key.
            reveal (bool): Include the resolved string value for a local field
                editor. Defaults to ``False``.

        Returns:
            Dict[str, Any]: Effective source, environment detection state, and
            a non-sensitive tail; plus ``value`` when explicitly requested.
        """
        raw_value = dget(self.config_tree()["tree"], key, default=None)
        value = self.cm.get(key, default="")
        value_text = "" if value is None else str(value)
        env_match = _ENV_REFERENCE.match(raw_value) if isinstance(raw_value, str) else None
        env_var = env_match.group(1) if env_match else None
        env_set = env_var in os.environ if env_var else None
        status = {
            "key": key,
            "set": bool(value_text),
            "tail": value_text[-4:] if len(value_text) >= 8 else "",
            "source": "environment" if env_set else "manual",
            "env_var": env_var,
            "env_set": env_set,
        }
        if reveal:
            status["value"] = value_text
        return status

    def config_set(self, key: str, value: Any, scope: Optional[str] = None, parse: str = "auto") -> Dict[str, Any]:
        """Set one config key in a scoped layer.

        Args:
            key (str): Dotted config key, for example ``heavenbase.llm.default_preset``.
            value (Any): New value; strings are parsed per ``parse``.
            scope (str | None): Target scope. Defaults to the active scope.
            parse (str): Value parsing mode: ``auto``, ``json``, or ``raw``.
                Defaults to ``auto``.

        Returns:
            Dict[str, Any]: ``{"ok": True, "key": key}`` on success.

        Raises:
            ValueError: If ``key`` is empty or ``parse`` is unsupported.
        """
        if not key:
            raise ValueError("config_set requires a non-empty key")
        raise_mismatch(["auto", "json", "raw"], parse, name="parse")
        self.config_ops.set(key, value, scope=scope, parse=parse)
        return {"ok": True, "key": key}

    def config_unset(self, key: str, scope: Optional[str] = None) -> Dict[str, Any]:
        """Remove one custom config key from a scoped layer.

        Built-in keys (present in the default layer) cannot be removed; reset
        them with :meth:`config_reset` instead so defaults keep showing
        through.

        Args:
            key (str): Dotted config key to remove.
            scope (str | None): Target scope. Defaults to the active scope.

        Returns:
            Dict[str, Any]: ``{"ok": True, "key": key}`` on success.

        Raises:
            ValueError: If ``key`` is empty or names a built-in default key.
        """
        if not key:
            raise ValueError("config_unset requires a non-empty key")
        if self._in_default(key):
            raise ValueError(f"{key!r} is a built-in default key; use config_reset to restore its default value")
        self.config_ops.unset(key, scope=scope)
        return {"ok": True, "key": key}

    def config_reset(self, key: str, scope: Optional[str] = None) -> Dict[str, Any]:
        """Reset one key to its built-in default value.

        Built-in keys are re-copied from the default layer so history is
        retained; custom keys are removed from the scoped layer.

        Args:
            key (str): Dotted config key to reset.
            scope (str | None): Target scope. Defaults to the active scope.

        Returns:
            Dict[str, Any]: ``{"ok": True, "key": key, "origin": origin}``
            where ``origin`` reports whether the key was ``builtin`` or
            ``custom`` before the reset.

        Raises:
            ValueError: If ``key`` is empty.
        """
        if not key:
            raise ValueError("config_reset requires a non-empty key")
        if self._in_default(key):
            self.config_ops.copy(key=key, source="default", target_scope=scope)
            return {"ok": True, "key": key, "origin": "builtin"}
        self.config_ops.unset(key, scope=scope)
        return {"ok": True, "key": key, "origin": "custom"}

    def config_history(self, scope: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Return retained version history rows for one scope.

        Args:
            scope (str | None): Target scope. Defaults to the active scope.
            limit (int): Maximum number of history rows. Defaults to ``20``.

        Returns:
            List[Dict[str, Any]]: History rows with ``scope``, ``ver``,
            ``gen``, ``pkg_ver``, and ``created_at``.
        """
        return self.cm.history(scope=_resolve_config_scope(self.cm, scope), limit=limit)

    def config_diff(
        self,
        scope_a: Optional[str] = None,
        version_a: Optional[int] = None,
        scope_b: Optional[str] = None,
        version_b: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a unified diff between two config layers.

        Args:
            scope_a (str | None): From scope. Defaults to the active scope.
            version_a (int | None): From version. Defaults to latest.
            scope_b (str | None): To scope. Defaults to the active scope.
            version_b (int | None): To version. Defaults to latest.

        Returns:
            Dict[str, Any]: ``{"diff": text}`` with an empty string when the
            layers match.
        """
        resolved_a = _resolve_config_scope(self.cm, scope_a) or self.cm.scope
        resolved_b = _resolve_config_scope(self.cm, scope_b) or self.cm.scope
        resolved_version_b = _resolve_config_version(self.cm, resolved_b, -1 if version_b is None else version_b)
        resolved_version_a = _resolve_config_version(self.cm, resolved_a, (-2 if resolved_a == resolved_b else -1) if version_a is None else version_a)
        data_a = _display_config_layer(self.cm, resolved_a, resolved_version_a)
        data_b = _display_config_layer(self.cm, resolved_b, resolved_version_b)
        label_a = f"{resolved_a} v{resolved_version_a}"
        label_b = f"{resolved_b} v{resolved_version_b}"
        diff = "".join(difflib.unified_diff(dumps_yaml(data_a).splitlines(True), dumps_yaml(data_b).splitlines(True), fromfile=label_a, tofile=label_b))
        return {"diff": diff}

    def _in_default(self, key: str) -> bool:
        def contains(parent: str, child: str) -> bool:
            return child == parent or child.startswith(f"{parent}.") or child.startswith(f"{parent}[")

        # dflat preserves the ConfigManager's escaped-dot path grammar. Treat
        # both ancestors and descendants of a built-in leaf as default-owned:
        # the former resets a subtree, while the latter must not tombstone a
        # scalar ancestor. Plain ``str.split('.')`` cannot distinguish dots in
        # model names such as ``gpt-5.6-luna``.
        return any(contains(key, default_key) or contains(default_key, key) for default_key, _ in dflat(self.cm.load_default()))

    # ------------------------------------------------------------- preferences

    def preferences(self) -> Dict[str, Any]:
        """Return shared dashboard appearance preferences.

        Args:
            None.

        Returns:
            Dict[str, Any]: Theme, language, font, and font-scale selections,
            plus whether a user layer has configured any of them.
        """
        raw = self.cm.snapshot(refresh=True).get(_DASHBOARD_PREFERENCES_PATH, default={}) or {}
        values = raw if isinstance(raw, Mapping) else {}
        result: Dict[str, Any] = {"configured": values.get("configured") is True}
        for key, default in _DASHBOARD_PREFERENCE_DEFAULTS.items():
            value = str(values.get(key, default))
            result[key] = value if value in _DASHBOARD_PREFERENCE_VALUES[key] else default
        return result

    def update_preferences(
        self,
        theme: Optional[str] = None,
        language: Optional[str] = None,
        font: Optional[str] = None,
        font_scale: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist supplied dashboard appearance preferences.

        Args:
            theme (str | None): Light, dark, or warm theme token.
            language (str | None): Interface language token.
            font (str | None): Whole-interface zoom preset.
            font_scale (str | None): Text-size preset.

        Returns:
            Dict[str, Any]: Complete resolved preference set after the update.
        """
        updates = {
            "theme": theme,
            "language": language,
            "font": font,
            "font_scale": font_scale,
        }
        supplied = {key: value for key, value in updates.items() if value is not None}
        if not supplied:
            return self.preferences()
        for key, value in supplied.items():
            if value not in _DASHBOARD_PREFERENCE_VALUES[key]:
                raise ValueError(f"Unsupported dashboard preference {key}={value!r}")
        current = self.preferences()
        persisted = {key: supplied.get(key, current[key]) for key in _DASHBOARD_PREFERENCE_DEFAULTS}
        persisted["configured"] = True
        with self.cm.scoped(self.cm.base_scope):
            self.cm.set(_DASHBOARD_PREFERENCES_PATH, persisted, scope=self.cm.base_scope)
        return self.preferences()

    # --------------------------------------------------------------- workspace

    def workspaces(self) -> List[Dict[str, Any]]:
        """Return globally registered workspace specs.

        Args:
            None.

        Returns:
            List[Dict[str, Any]]: ``WorkspaceSpec.to_dict()`` payloads without
            the schema manifest bodies, plus ``open`` marking the workspace
            currently held by this dashboard session.
        """
        from heavenbase.workspace import WorkspaceSpec

        with self._workspace_lock:
            open_ws_id = self.ws_id
        specs = []
        for spec in WorkspaceSpec.list(context=self.context):
            data = spec.to_dict()
            data.pop("manifest", None)
            # WorkspaceSpec persistence omits this projection. The dashboard
            # treats the configured default as selected when no explicit
            # active workspace exists, so startup never falls through to an
            # unrelated alphabetically first registration.
            data["active"] = spec.id == self.context.effective_workspace_id
            data["open"] = spec.id == open_ws_id
            specs.append(data)
        return specs

    def open(self, ws_id: str) -> Dict[str, Any]:
        """Open one registered workspace for browsing.

        Args:
            ws_id (str): Registered workspace id.

        Returns:
            Dict[str, Any]: Health payload for the opened workspace.

        Raises:
            KeyError: If no registered workspace matches ``ws_id``.
        """
        from heavenbase.workspace import WorkspaceSpec

        with self._workspace_lock:
            # Serialize the complete transition so every caller observes one
            # linear workspace session boundary, even when HTTP requests run in
            # separate worker threads.
            try:
                spec = WorkspaceSpec.load(ws_id, context=self.context)
            except KeyError:
                raise
            try:
                workspace = spec.open(context=self.context)
                health = workspace.health()
                self.context.activate_workspace(workspace.id)
            except Exception as error:
                raise RuntimeError(f"could not open workspace {ws_id!r}: {error}") from error
            self.ws = workspace
            self.ws_id = workspace.id
        return {"ws_id": workspace.id, **health}

    def remove_workspace(self, ws_id: str) -> Dict[str, Any]:
        """Unregister one workspace without deleting its backing data.

        If this dashboard currently holds the workspace, its live facade is
        released from process lookup and the dashboard returns to its empty
        state. Backend databases remain untouched.

        Args:
            ws_id (str): Registered workspace id to remove.

        Returns:
            Dict[str, Any]: Workspace id, whether a registration was removed,
            and whether it was open in this dashboard session.

        Raises:
            ValueError: If ``ws_id`` is not a valid workspace identifier.
            RuntimeError: If the open workspace cannot be released.
        """
        from heavenbase.workspace import WorkspaceSpec
        from heavenbase.utils import check_workspace_id

        clean_id = check_workspace_id(ws_id)
        with self._workspace_lock:
            was_open = self.ws_id == clean_id
            workspace = self.ws if was_open else None
            removed = WorkspaceSpec.unregister(clean_id, context=self.context)
            if workspace is not None:
                workspace.release()
            if was_open:
                self.ws = None
                self.ws_id = None
        return {"ws_id": clean_id, "removed": removed, "was_open": was_open}

    def health(self) -> Dict[str, Any]:
        """Return runtime health for the open workspace.

        Args:
            None.

        Returns:
            Dict[str, Any]: ``ws.health()`` payload with the workspace id.

        Raises:
            ValueError: If no workspace is open.
        """
        workspace = self._open_ws()
        return {"ws_id": workspace.id, **workspace.health()}

    def entities(self, include_system: bool = False, include_counts: bool = True) -> List[Dict[str, Any]]:
        """Return entity cards for the browser panel.

        Args:
            include_system (bool): Whether hidden system/extension entities
                are included. Defaults to ``False``.
            include_counts (bool): Whether row counts are computed per
                entity. Defaults to ``True``.

        Returns:
            List[Dict[str, Any]]: Cards with ``entity_id``, ``name``,
            ``desc``, ``tags``, ``system``, ``extension``, and ``count``.

        Raises:
            ValueError: If no workspace is open.
        """
        ws = self._open_ws()
        visible = set(ws.list_entities())
        owners = {entity_id: ext_id for ext_id, ext in ws.extensions().items() for entity_id in ext.get("entities", [])}
        cards = []
        for entity_id in sorted(ws.entities):
            system = entity_id not in visible
            if system and not include_system:
                continue
            schema = ws.entities[entity_id].schema()
            cards.append(
                {
                    "entity_id": entity_id,
                    "name": schema.name,
                    "desc": schema.desc,
                    "tags": list(schema.tags),
                    "system": system,
                    "extension": owners.get(entity_id),
                    "count": ws.count(entity_id) if include_counts else None,
                }
            )
        return cards

    def entity_schema(self, entity_id: str) -> Dict[str, Any]:
        """Return schema, storage placement, and capability metadata for one entity.

        Args:
            entity_id (str): Registered entity identifier.

        Returns:
            Dict[str, Any]: ``{"schema", "storage", "capabilities", "backends"}``
            where ``schema`` is the entity schema dict, ``storage`` maps field
            names to storage binding rows from MetaSchema, ``capabilities``
            maps field names to their query-capability metadata, and
            ``backends`` lists backend metadata rows referenced by the
            bindings.

        Raises:
            KeyError: If ``entity_id`` is not registered.
            ValueError: If no workspace is open.
        """
        ws = self._open_ws()
        if entity_id not in ws.entities:
            raise KeyError(f"No registered entity {entity_id!r}")
        from heavenbase import MetaSchema

        meta_rows = ws.query_json(MetaSchema, {"filter": {"subject_id": entity_id}}).execute().rows()
        storage: Dict[str, List[Dict[str, Any]]] = {}
        capabilities: Dict[str, Any] = {}
        for row in meta_rows:
            if not row.get("field"):
                continue
            if row.get("kind") == "storage":
                storage.setdefault(row["field"], []).append(
                    {
                        "backend": row.get("backend"),
                        "type": row.get("type"),
                        "strategy": row.get("strategy"),
                        "source": row.get("source"),
                        "meta": row.get("meta"),
                    }
                )
            elif row.get("kind") == "field" and isinstance(row.get("meta"), dict) and row["meta"].get("query_capabilities"):
                capabilities[row["field"]] = row["meta"]["query_capabilities"]
        backend_names = sorted({binding["backend"] for bindings in storage.values() for binding in bindings if binding.get("backend")})
        backends = ws.query_json(MetaSchema, {"filter": {"kind": "backend", "subject_id": {"$in": backend_names}}}).execute().rows() if backend_names else []
        return {"schema": ws.entities[entity_id].schema().to_dict(), "storage": storage, "capabilities": capabilities, "backends": backends}

    def entity_rows(
        self,
        entity_id: str,
        filter: Optional[Dict[str, Any]] = None,
        order: Optional[List[str]] = None,
        select: Optional[List[str]] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Return one page of entity rows for the data table.

        Args:
            entity_id (str): Registered entity identifier.
            filter (Dict[str, Any] | None): Mongo-style filter dict passed to
                ``ws.query_json``. Defaults to ``None``.
            order (List[str] | None): Order fields; a ``-`` prefix sorts
                descending. Defaults to ``["object_id"]`` for stable paging.
            select (List[str] | None): Projected fields; ``object_id`` is
                always included. Defaults to all fields.
            offset (int): Page offset. Defaults to ``0``.
            limit (int): Page size, capped at ``MAX_PAGE_SIZE``. Defaults to ``50``.

        Returns:
            Dict[str, Any]: ``{"rows", "columns", "total", "has_more",
            "offset", "limit"}`` where ``total`` counts the rows matching
            ``filter`` when one is given, else the whole entity.

        Raises:
            KeyError: If ``entity_id`` is not registered.
            ValueError: If no workspace is open or paging values are negative.
        """
        ws = self._open_ws()
        if entity_id not in ws.entities:
            raise KeyError(f"No registered entity {entity_id!r}")
        if offset < 0 or limit <= 0:
            raise ValueError(f"entity_rows requires offset >= 0 and limit > 0, got offset={offset} limit={limit}")
        limit = min(limit, MAX_PAGE_SIZE)
        spec: Dict[str, Any] = {"order": order or ["object_id"], "offset": offset, "limit": limit + 1}
        if filter:
            spec["filter"] = filter
        if select:
            spec["select"] = select
        frame = ws.query_json(entity_id, spec).execute()
        rows = frame.rows()
        has_more = len(rows) > limit
        if filter:
            # engine-side count under the same filter, so filtered paging can jump exactly
            counted = ws.query_json(entity_id, {"filter": filter, "aggregate": {"n": {"count": {"field": "object_id"}}}}).execute().rows()
            total = int(counted[0]["n"]) if counted else 0
        else:
            total = ws.count(entity_id)
        return {
            "rows": rows[:limit],
            "columns": frame.col_names(),
            "total": total,
            "has_more": has_more,
            "offset": offset,
            "limit": limit,
        }

    def entity_row(self, entity_id: str, object_id: str) -> Dict[str, Any]:
        """Return one row for the row inspector.

        Args:
            entity_id (str): Registered entity identifier.
            object_id (str): Row ``object_id``.

        Returns:
            Dict[str, Any]: The merged logical row.

        Raises:
            KeyError: If the entity or row does not exist.
            ValueError: If no workspace is open.
        """
        row = self._open_ws().get(object_id, entity_id)
        if row is None:
            raise KeyError(f"No row {object_id!r} in entity {entity_id!r}")
        return row

    def column_stats(self, entity_id: str, field: str) -> Dict[str, Any]:
        """Return kind-aware statistics for one entity column, over every row.

        The field dtype selects what is computed: numeric columns get quartiles
        and a histogram, text columns a length distribution plus either top
        values or examples, time columns a parsed range, vectors dimension and
        L2-norm stats, and Json columns their key and element distributions.

        Every statistic is folded inside the aggregate stage, so it covers the
        whole column rather than a scanned prefix, and the rows never enter this
        process. ``scanned`` and ``total`` stay two separate measurements — how
        many rows the fold was handed, and how many the entity holds — so
        ``sampled`` reports their agreement instead of asserting it. They agree
        on every path that exists today; one that ever hands the fold less than
        the whole entity would be reported rather than silently believed.

        Args:
            entity_id (str): Registered entity identifier.
            field (str): Declared field name on that entity.

        Returns:
            Dict[str, Any]: Common keys ``{"field", "dtype", "kind", "total",
            "scanned", "sampled", "nulls", "null_pct"}`` plus kind-specific keys
            such as ``percentiles``/``histogram`` (numeric), ``length``/
            ``length_histogram``/``examples`` (text), ``format``/``earliest``/
            ``latest``/``span_days``/``unparsed`` (time), ``keys``/``keys_exact``/
            ``distinct_keys``/``key_count``/``size``/``scalars``/``scalar_values`` (json), ``dims``/``norm``
            (vector), and ``items``/``elements``/``elements_exact`` (list).
            ``top`` and ``distinct`` appear where they are meaningful. ``keys``
            and ``elements`` carry whole-column counts; a published list cut at
            its display bound is marked by ``<key>_truncated``. Every published value cell — ``top``,
            ``elements``, and ``examples`` — carries its own
            ``value_truncated``/``value_len`` when the text was cut for display,
            so a prefix is never quoted back as if it were the whole value.

        Raises:
            KeyError: If the entity or field is not registered.
            ValueError: If no workspace is open.
        """
        ws = self._open_ws()
        if entity_id not in ws.entities:
            raise KeyError(f"No registered entity {entity_id!r}")
        schema = ws.entities[entity_id].schema()
        if field not in schema.fields:
            raise KeyError(f"No field {field!r} in entity {entity_id!r}")
        dtype_id = type(schema.fields[field].dtype).identifier
        kind = stat_kind(dtype_id)

        # One aggregate carries every metric, so the engine materializes the
        # column once. column_summary is a registered reducer rather than an
        # expression, so these statistics do not depend on the switch that
        # refuses caller-authored code. The fold never sees nulls, so the null
        # count is the row count minus the values that reached it — and both
        # counts come from this one query, because subtracting a count taken
        # here from one taken elsewhere can go negative when two routed backends
        # disagree, and clamping that away would hide the disagreement.
        spec = {
            "aggregate": {
                "rows": {"count": {"field": "object_id"}},
                "present": {"count": {"field": field}},
                "summary": {"column_summary": {"field": field, "kind": kind}},
            }
        }
        rows = ws.query_json(entity_id, spec).execute().rows()
        row = rows[0] if rows else {}
        scanned = int(row.get("rows") or 0)
        present = int(row.get("present") or 0)
        summary = row.get("summary") or {}
        # Two independent measurements: the entity's own row count, and how many
        # rows the fold was actually handed. They agree today, and `sampled`
        # states that rather than asserting it — a path that ever hands the fold
        # less than the whole entity is then reported, not silently believed.
        total = max(ws.count(entity_id), scanned)
        nulls = scanned - present
        stats: Dict[str, Any] = {
            "field": field,
            "dtype": dtype_id,
            "kind": kind,
            "total": total,
            "scanned": scanned,
            "sampled": scanned < total,
            "nulls": nulls,
            "null_pct": round(nulls / scanned * 100, 1) if scanned else 0.0,
            **summary,
        }
        # These distributions are exact when the fold was handed the whole
        # entity, which is the same reading `sampled` reports — derive them from
        # it rather than asserting, so all four flags move together.
        exact = not stats["sampled"]
        if "keys" in stats:
            stats["keys_exact"] = exact
        if "elements" in stats:
            stats["elements_exact"] = exact
        return stats

    def search(self, text: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Search workspace cards by free text.

        Args:
            text (str): Search text matched over Catalog and MetaSchema cards.
            limit (int | None): Maximum result rows. Defaults to the
                workspace search default.

        Returns:
            Dict[str, Any]: ``{"rows", "n", "total", "truncated"}``. Ranking
            scores every matching card before the cap applies, so ``total`` is
            the exact match count and ``truncated`` says whether the cap
            withheld cards. ``n`` counts the rows returned.

        Raises:
            ValueError: If no workspace is open or ``limit`` exceeds the
                configured maximum.
        """
        frame = self._open_ws().search(text, limit=limit)
        return {"rows": frame.rows(), "n": len(frame), "total": frame.total if frame.total is not None else len(frame), "truncated": frame.truncated}

    def assistant_key_status(self, provider: str, reveal: bool = False, field: str = "api_key") -> Dict[str, Any]:
        """Report how one provider field resolves, without exposing secrets.

        Args:
            provider (str): Provider name under ``heavenbase.llm.providers``.
            reveal (bool): Include the resolved value as ``value`` for the
                local workbench field editor. Defaults to ``False``.
            field (str): ``"api_key"`` or ``"base_url"``. Defaults to
                ``"api_key"``.

        Returns:
            Dict[str, Any]: Field status with the resolved-value mask,
            interpolation source, environment-variable name and detection
            state; plus ``value`` when ``reveal`` is set.

        Raises:
            KeyError: If the provider is not configured.
            ValueError: If ``field`` is not an allowed provider field.
        """
        if field not in ("api_key", "base_url"):
            raise ValueError(f"Unsupported provider field {field!r}")
        providers = self.cm.get("heavenbase.llm.providers", default=None) or {}
        if provider not in providers:
            raise KeyError(f"No provider {provider!r}")
        key = f"heavenbase.llm.providers.{provider}.{field}"
        generic = self.config_value_status(key, reveal=reveal)
        source = "environment" if generic["source"] == "environment" else "fallback" if generic["env_var"] else "literal"
        status = {
            "provider": provider,
            "field": field,
            "set": generic["set"],
            "tail": generic["tail"],
            "source": source,
            "env_var": generic["env_var"],
            "env_set": generic["env_set"],
        }
        if reveal:  # the local workbench may show the resolved value on request, like config secret reveal
            status["value"] = generic.get("value", "")
        return status

    def _open_ws(self) -> Any:
        with self._workspace_lock:
            workspace = self.ws
        if workspace is None:
            raise ValueError("No workspace is open; call open(ws_id) first")
        return workspace

    # --------------------------------------------------------------- assistant

    def _assist_turn_log(
        self, question: str, conversation_id: Optional[str], turn_id: Optional[str], ws_id: str, *, cancelled: Optional[threading.Event] = None
    ) -> Any:
        """Open the durable turn recorder, or ``None`` when logging is off."""
        from .assist_log import AssistSessionLog

        if not AssistSessionLog.enabled(self.cm):
            return None
        with self._assist_session_log_lock:
            if self._assist_session_log is None:
                self._assist_session_log = AssistSessionLog(self.cm, context=self.context)
            recorder = self._assist_session_log
        return recorder.begin_turn(question, conversation_id=conversation_id, turn_id=turn_id, ws_id=ws_id, cancelled=cancelled)

    def assist(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        llm: Any = None,
        progress_id: Optional[str] = None,
        preset: Optional[str] = None,
        read_only: Optional[bool] = None,
        conversation_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer one assistant question with the configured chat LLM.

        The assistant is stateless on the server: the browser sends the prior
        turns back as ``history`` on every call. Screen context (route, open
        workspace, active entity, schema summary) is folded into the system
        prompt so answers can reference what the user is looking at.

        Args:
            question (str): User question for this turn.
            context (Dict[str, Any] | None): Screen context from the client,
                for example ``{"route", "entity", "schema"}``. Defaults to ``None``.
            history (List[Dict[str, str]] | None): Prior ``{"role", "content"}``
                turns (``user``/``assistant`` only). Defaults to ``None``.
            llm (Any): Injected chat runtime for tests. Defaults to an ``LLM``
                built from ``heavenbase.dashboard.assistant`` (preset/model/
                provider/gateway), falling back to ``heavenbase.llm.default_preset``.
            progress_id (str | None): Client token registering a live activity
                feed; while this call runs, :meth:`assist_progress` returns the
                tool invocations recorded so far. Defaults to ``None``.
            conversation_id (str | None): Client conversation identity for the
                durable session log. If omitted, a supplied ``turn_id``
                identifies its own conversation. Unused while ``session_log`` is off.
                Defaults to ``None``.
            turn_id (str | None): Client turn identity. Reuse it to replay a
                logged completed turn without provider I/O. Defaults to ``None``.

        Returns:
            Dict[str, Any]: ``{"text", "think", "served_model", "frames",
            "events", "elapsed"}`` where ``think`` is empty when the model exposes no
            reasoning chain, ``frames`` carries up to ``max_frames`` executed
            query result frames, and ``events`` is the interleaved timeline of
            ``{"type": "think"}`` and ``{"type": "tool"}`` steps in execution
            order for agent-style rendering.

        Raises:
            ValueError: If ``question`` is empty, the retry changes the
                question or workspace of a recorded turn, or the LLM call
                fails (for example when no provider key is configured).
        """
        if not question or not question.strip():
            raise ValueError("assist requires a non-empty question")
        # Capture the prompt identity and tool workspace in one critical
        # section. ``open()`` uses the same lock, so a concurrent switch occurs
        # wholly before or after this request's assistant snapshot.
        effective_read_only = self.cm.get("heavenbase.dashboard.assistant.read_only", default=True) is True if read_only is None else bool(read_only)
        injected = llm is not None  # injected runtimes never trigger background model construction
        frames: List[Dict[str, Any]] = []
        evidence_frames: List[Dict[str, Any]] = []
        progress: List[Dict[str, Any]] = []
        evidence_budget = self._assist_evidence_budget()
        evidence_caps = self._assist_evidence_caps() if evidence_budget else None
        evidence_log: Optional[List[Tuple[str, str, str]]] = [] if evidence_budget else None
        with self._workspace_lock:
            evidence_scope = str(self.ws_id or "")
            system = self._assist_system(context or {}, preset=preset, read_only=effective_read_only)
            tools = self._assist_tools(
                frames=frames, evidence_frames=evidence_frames, progress=progress, read_only=effective_read_only, evidence_log=evidence_log
            )
        turn_log = self._assist_turn_log(question, conversation_id, turn_id, evidence_scope)
        replay = turn_log.replay() if turn_log is not None else None
        if replay is not None:
            return replay
        # From here on every exit closes the opened turn: replay assembly, model
        # construction and the progress-feed checks can all raise before
        # ``answer()`` runs, and an unclosed turn would keep its conversation
        # pinned against retention for the life of the process. ``close`` is
        # idempotent, so the paths inside ``answer()`` that already closed it
        # make this a no-op.
        # The usage of every provider call this turn makes, from its traced response events.
        traced_usages: List[Mapping[str, Any]] = []
        try:
            system, replayed = self._assist_replay_history(
                system,
                history or [],
                question,
                turns=self._assist_history_turns(),
                evidence_budget=evidence_budget,
                preset=preset,
                injected=injected,
                scope=evidence_scope,
                trace=_counting_usage(turn_log.llm_trace("replay_compaction"), traced_usages) if turn_log is not None else None,
            )
            messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
            for turn in replayed:
                if turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str):
                    messages.append({"role": turn["role"], "content": turn["content"]})
            messages.append({"role": "user", "content": question})
            result_limit = self._assist_result_limit()
            # ``truncated`` says the tool-call budget ran out, so an answer cut short
            # mid-investigation is never presented to the user as a finished one.
            include = ["text", "think", "raw", "delta", "elapsed", "truncated"]
            if turn_log is not None:
                include.append("usage")  # merged provider usage rides the turn record
            chat_kwargs: Dict[str, Any] = {"include": include, "tools": tools}
            if llm is None:
                from heavenbase.llm import LLM

                # cache=False: assistant turns are context-specific and short-lived, and it avoids
                # spinning up the response-cache workspace for a one-off chat.
                llm = LLM(context=self.context, cache=False, **self._assist_llm_options(preset=preset))
                # Per-call, like on_round, so injected test runtimes keep their
                # plain signatures; the default budget only backstops a runaway
                # loop — the context window is the real limiter.
                chat_kwargs["max_tool_turns"] = self._assist_tool_turns()
                chat_kwargs["tool_result_capture_chars"] = self._assist_result_capture()
                chat_kwargs["compact_ratio"] = self._assist_compact_ratio()
                if turn_log is not None:
                    # Write-ahead: each provider-bound request lands in the log
                    # before provider I/O. Injected test runtimes keep their
                    # plain signatures, so their provider calls go untraced while
                    # the turn row still closes below.
                    chat_kwargs["trace"] = _counting_usage(turn_log.llm_trace(), traced_usages)

                def _on_round(response: Any) -> None:
                    # each completed tool round pushes its reasoning to the live feed
                    if getattr(response, "think", ""):
                        progress.append({"type": "think", "text": str(response.think)[:2000]})

                chat_kwargs["on_round"] = _on_round

            def answer() -> Dict[str, Any]:
                """Return the prepared assistant reply with dashboard metadata."""
                try:
                    # Read-only tools let the model fetch real schema/rows/config/stats instead of guessing;
                    # ``think`` carries the model's reasoning chain for the collapsible UI block.
                    result = llm.chat(messages, **chat_kwargs)
                    raw = result.get("raw")
                    while isinstance(raw, list) and raw:  # tool rounds nest lists; descend to the final response
                        raw = raw[-1]
                    # The provider's own response names the model that actually served the
                    # call — wire-level evidence, unlike config intent or model self-reports.
                    served = getattr(raw, "model", None) or (raw.get("model") if hasattr(raw, "get") else None)
                    events = _assist_events(result.get("delta"), frames, result_limit, evidence_frames)
                    reply = {
                        "text": result.get("text") or "",
                        "think": result.get("think") or "",
                        "served_model": str(served or ""),
                        "frames": [capture["frame"] for capture in frames if isinstance(capture.get("frame"), dict)],
                        "sql_frames": [capture["frame"] for capture in evidence_frames if isinstance(capture.get("frame"), dict)],
                        "events": events,
                        "elapsed": round(float(result.get("elapsed") or 0), 1),
                        "truncated": bool(result.get("truncated")),
                    }
                    if evidence_log is not None:
                        # keyed by the final answer text — exactly what the client replays
                        self._assist_store_evidence(reply["text"], evidence_log, evidence_caps, question=question, scope=evidence_scope)
                    if turn_log is not None:
                        turn_log.close(
                            text=reply["text"],
                            served_model=reply["served_model"],
                            elapsed=reply["elapsed"],
                            truncated=reply["truncated"],
                            # Every provider call the turn made, compaction summaries included; an injected
                            # runtime is untraced and reports its own merged usage.
                            usage=(result.get("usage") if hasattr(result, "get") else None) if injected else _merge_usage(traced_usages),
                        )
                        if turn_log.capture != "complete":
                            reply["capture_warning"] = turn_log.capture  # degraded recording is user-visible, not just log-visible
                    return reply
                except Exception as error:  # surface provider/config failures as one clean client message
                    if turn_log is not None:
                        turn_log.close(error=f"{type(error).__name__}: {error}")
                    raise ValueError(f"assist failed: {error}") from error

            progress_key = str(progress_id) if progress_id else None
            if progress_key:
                with self._workspace_lock:
                    if progress_key in self._assist_progress:
                        raise ValueError(f"assist progress id {progress_key!r} is already active")
                    if len(self._assist_progress) >= 32:
                        raise RuntimeError("dashboard assistant already has 32 active progress feeds")
                    self._assist_progress[progress_key] = progress
            try:
                return answer()
            finally:
                if progress_key:
                    with self._workspace_lock:
                        if self._assist_progress.get(progress_key) is progress:
                            self._assist_progress.pop(progress_key, None)
        except BaseException as error:
            if turn_log is not None:
                turn_log.close(error=f"{type(error).__name__}: {error}")
            raise

    def assist_stream(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        llm: Any = None,
        preset: Optional[str] = None,
        read_only: Optional[bool] = None,
        conversation_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream one assistant answer as ordered event dictionaries.

        Yields the model's own stream as it arrives: reasoning and text deltas
        chunk by chunk, a ``tool``/``tool_result`` pair per executed call (query
        results ride as lossless frames), and a final ``done`` event whose
        payload matches :meth:`assist` so clients can finalize the message
        identically to the blocking path.

        Args:
            question (str): User question for this turn.
            context (Dict[str, Any] | None): Screen context from the client.
                Defaults to ``None``.
            history (List[Dict[str, str]] | None): Prior turns. Defaults to ``None``.
            llm (Any): Injected runtime for tests; must expose ``stream``.
                Defaults to an ``LLM`` from the assistant config.
            conversation_id (str | None): Client conversation identity for the
                durable session log. If omitted, a supplied ``turn_id``
                identifies its own conversation. Unused while ``session_log`` is off.
                Defaults to ``None``.
            turn_id (str | None): Client turn identity. Reuse it to replay a
                logged completed turn without provider I/O. Defaults to ``None``.

        Yields:
            Dict[str, Any]: ``{"kind": "think"|"text"|"tool"|"tool_result"|
            "done"|"error", ...}`` events in stream order. A retry with a
            different question or workspace for a recorded turn emits an
            error event.

        Raises:
            ValueError: If ``question`` is empty.
        """
        if not question or not question.strip():
            raise ValueError("assist requires a non-empty question")
        effective_read_only = self.cm.get("heavenbase.dashboard.assistant.read_only", default=True) is True if read_only is None else bool(read_only)
        injected = llm is not None  # injected runtimes never trigger background model construction
        frames: List[Dict[str, Any]] = []
        evidence_frames: List[Dict[str, Any]] = []
        turn_log: Any = None
        # The usage of every provider call this turn makes, from its traced response events.
        traced_usages: List[Mapping[str, Any]] = []
        cancelled = _ASSIST_STREAM_STOP.get()
        try:
            evidence_budget = self._assist_evidence_budget()
            evidence_caps = self._assist_evidence_caps() if evidence_budget else None
            evidence_log: Optional[List[Tuple[str, str, str]]] = [] if evidence_budget else None
            with self._workspace_lock:
                evidence_scope = str(self.ws_id or "")
                system = self._assist_system(context or {}, preset=preset, read_only=effective_read_only)
                tools = self._assist_tools(frames=frames, evidence_frames=evidence_frames, read_only=effective_read_only, evidence_log=evidence_log)
            turn_log = self._assist_turn_log(question, conversation_id, turn_id, evidence_scope, cancelled=cancelled)
            if cancelled is not None and cancelled.is_set():
                if turn_log is not None:
                    turn_log.close(error="interrupted: stream closed before completion")
                return
            replay = turn_log.replay() if turn_log is not None else None
            if replay is not None:
                yield {"kind": "done", **replay}
                return
            system, replayed = self._assist_replay_history(
                system,
                history or [],
                question,
                turns=self._assist_history_turns(),
                evidence_budget=evidence_budget,
                preset=preset,
                injected=injected,
                scope=evidence_scope,
                trace=_counting_usage(turn_log.llm_trace("replay_compaction"), traced_usages) if turn_log is not None else None,
            )
            messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
            for turn in replayed:
                if turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str):
                    messages.append({"role": turn["role"], "content": turn["content"]})
            messages.append({"role": "user", "content": question})
            loop_kwargs: Optional[Dict[str, Any]] = None
            if llm is None:
                from heavenbase.llm import LLM

                llm = LLM(context=self.context, cache=False, **self._assist_llm_options(preset=preset))
                # Per-call, mirroring the blocking path, so injected test
                # runtimes keep their plain ``stream`` signatures.
                loop_kwargs = {
                    "max_tool_turns": self._assist_tool_turns(),
                    "tool_result_capture_chars": self._assist_result_capture(),
                    "compact_ratio": self._assist_compact_ratio(),
                }
                if turn_log is not None:
                    # write-ahead: requests land in the log before provider I/O
                    loop_kwargs["trace"] = _counting_usage(turn_log.llm_trace(), traced_usages)
        except Exception as error:
            if turn_log is not None:
                turn_log.close(error=f"{type(error).__name__}: {error}")
            yield {"kind": "error", "text": f"assist failed: {error}"}
            return
        except BaseException as error:  # an interrupt before the stream starts still releases the turn's conversation
            if turn_log is not None:
                turn_log.close(error=f"{type(error).__name__}: {error}")
            raise

        def events() -> Generator[Dict[str, Any], None, None]:
            """Yield native events for the prepared assistant stream."""
            raws: List[Any] = []
            thinks: List[str] = []
            round_texts: List[str] = []
            delta_messages: List[Dict[str, Any]] = []
            emitted_calls: set = set()
            elapsed = 0.0
            truncated = False
            chunk_rounds: List[List[Mapping[str, Any]]] = [[]]  # chunk usages per round, for an untraced injected runtime
            matched_frames: set[int] = set()
            matched_evidence: set[int] = set()
            include = ["text", "think", "tool_calls", "message", "delta", "raw", "elapsed", "truncated", "usage"]
            try:
                result_limit = self._assist_result_limit()
                for chunk in llm.stream(messages, include=include, reduce=False, tools=tools, **(loop_kwargs or {})):
                    raws.append(chunk.get("raw"))
                    elapsed = max(elapsed, float(chunk.get("elapsed") or 0))
                    truncated = truncated or bool(chunk.get("truncated"))  # only the closing aggregate knows the verdict
                    if chunk.get("usage"):
                        chunk_rounds[-1].append(chunk["usage"])
                    if chunk.get("think"):
                        thinks.append(chunk["think"])
                        yield {"kind": "think", "text": chunk["think"]}
                    if chunk.get("text"):
                        round_texts.append(chunk["text"])
                        yield {"kind": "text", "text": chunk["text"]}
                    # the longest delta wins: round boundaries carry their round, the
                    # final aggregate carries the whole conversation added by the call
                    if len(chunk.get("delta") or []) > len(delta_messages):
                        delta_messages = list(chunk["delta"])
                    # Only true boundary chunks (their delta carries the executed
                    # tool results) emit tool events; providers stream partial
                    # tool_calls with ids mid-round, which must not fire early.
                    has_results = any(message.get("role") == "tool" for message in chunk.get("delta") or [])
                    if has_results and chunk_rounds[-1]:
                        chunk_rounds.append([])  # the boundary chunk closes the provider round
                    boundary_calls = (chunk.get("message") or {}).get("tool_calls") or [] if has_results else []
                    if boundary_calls:
                        round_texts = []  # a preamble before tool calls is narrative, not the final answer
                    for call in boundary_calls:
                        call_id = str(call.get("id") or "")
                        if not call_id or call_id in emitted_calls:
                            continue
                        emitted_calls.add(call_id)
                        function = call.get("function") or {}
                        event: Dict[str, Any] = {
                            "kind": "tool",
                            "name": str(function.get("name") or ""),
                            "args": str(function.get("arguments") or "")[:400],
                        }
                        if event["name"] == "query":
                            frame = _assist_query_frame(function.get("arguments"), frames, matched_frames)
                            if frame is not None:
                                event["frame"] = frame
                        else:
                            evidence = _assist_evidence_frame(event["name"], function.get("arguments"), evidence_frames, matched_evidence)
                            if evidence is not None:
                                event["sql_frame"] = evidence
                        result = next(
                            (
                                message
                                for message in chunk.get("delta") or []
                                if message.get("role") == "tool" and str(message.get("tool_call_id") or "") == call_id
                            ),
                            None,
                        )
                        yield event
                        if result is not None:
                            yield {"kind": "tool_result", "text": _bounded_assist_result(result.get("content"), result_limit)}
            except Exception as error:  # surface provider/config failures as one clean stream event
                if turn_log is not None:
                    turn_log.close(error=f"{type(error).__name__}: {error}")
                yield {"kind": "error", "text": f"assist failed: {error}"}
                return
            served = None
            for raw in reversed(raws):
                while isinstance(raw, list) and raw:
                    raw = raw[-1]
                served = getattr(raw, "model", None) or (raw.get("model") if hasattr(raw, "get") else None)
                if served:
                    break
            done_text = "".join(round_texts)
            if evidence_log is not None:
                # keyed by the final answer text — exactly what the client replays
                self._assist_store_evidence(done_text, evidence_log, evidence_caps, question=question, scope=evidence_scope)
            done_event = {
                "kind": "done",
                "text": done_text,
                "think": "\n\n".join(part for part in thinks if part),
                "served_model": str(served or ""),
                "frames": [capture["frame"] for capture in frames if isinstance(capture.get("frame"), dict)],
                "sql_frames": [capture["frame"] for capture in evidence_frames if isinstance(capture.get("frame"), dict)],
                "events": _assist_events(delta_messages, frames, result_limit, evidence_frames),
                "elapsed": round(elapsed, 1),
                "truncated": truncated,
            }
            if turn_log is not None:
                # Every provider call the turn made, compaction summaries included. An injected runtime is
                # untraced: its usage is LLM.chat's aggregate, each round's chunks merged, then the rounds.
                usage = _merge_usage([_merge_usage(chunks) for chunks in chunk_rounds if chunks]) if injected else _merge_usage(traced_usages)
                turn_log.close(text=done_text, served_model=str(served or ""), elapsed=round(elapsed, 1), truncated=truncated, usage=usage or None)
                if turn_log.capture != "complete":
                    done_event["capture_warning"] = turn_log.capture  # degraded recording is user-visible, not just log-visible
            yield done_event

        try:
            if cancelled is None or not cancelled.is_set():
                yield from events()
        finally:
            if turn_log is not None:
                # A client disconnect tears this generator down mid-stream;
                # the write-ahead request rows are already durable, and this
                # close (a no-op after a normal finish) marks the turn.
                turn_log.close(error="interrupted: stream closed before completion")

    def assist_progress(self, progress_id: str) -> Dict[str, Any]:
        """Return the live tool-activity feed for one running assist call.

        Args:
            progress_id (str): Token the client sent with :meth:`assist`.

        Returns:
            Dict[str, Any]: ``{"lines": [...]}`` — ordered feed entries
            (``{"type": "tool"|"think", "text"}``) recorded so far; empty once
            the call finishes or for unknown tokens.
        """
        with self._workspace_lock:
            return {"lines": list(self._assist_progress.get(str(progress_id), []))}

    def _assist_tools(
        self,
        frames: Optional[List[Dict[str, Any]]] = None,
        evidence_frames: Optional[List[Dict[str, Any]]] = None,
        progress: Optional[List[Dict[str, Any]]] = None,
        read_only: bool = False,
        evidence_log: Optional[List[Tuple[str, str, str]]] = None,
    ) -> List[Any]:
        """Return the tool callables exposed to the assistant.

        Core tools are thin wrappers over existing read-only ``DashboardAPI``
        methods so the model can inspect the real workspace and config rather
        than answering from a static context snapshot. Active extensions may
        add tools through MCP profile metadata (interactive-mode allowlists are
        owned by those profiles rather than this service) and through the open
        ``assistant_tools`` contribution kind (see
        :meth:`_assist_contributed_tools`); both ride the same narration,
        evidence capture, and name-collision handling as the built-ins.

        ``heavenbase.dashboard.assistant.run_python`` adds one exception, off by
        default: ``run_python`` executes model-authored Python in this process
        against the open workspace. That code is not a read-only wrapper and is
        bounded by nothing the dashboard controls, so the read-only property
        above holds only while the setting is disabled.

        Args:
            frames (List[Dict[str, Any]] | None): When given, every executed
                ``query`` appends its normalized arguments and optional frame.
                Defaults to ``None``.
            progress (List[Dict[str, Any]] | None): When given, every tool
                invocation appends an activity entry for the live busy feed.
                Defaults to ``None``.
            evidence_log (List[Tuple[str, str, str]] | None): When given, every
                completed tool call appends its rendered ``(name, args,
                result)`` for the cross-turn evidence carry. Defaults to
                ``None``.

        Returns:
            List[Any]: Tool callables with docstrings the LLM converts to schemas.
        """
        with self._workspace_lock:
            workspace = self.ws
            workspace_id = self.ws_id
        tool_api = self
        if workspace is not None:
            # Bind every workspace tool to one immutable session snapshot. A
            # concurrent UI switch can update this server's current workspace
            # without mixing two workspaces inside one model tool loop.
            tool_api = DashboardAPI(context=self.context)
            tool_api.ws = workspace
            tool_api.ws_id = workspace_id

        def list_entities() -> List[Dict[str, Any]]:
            """List entities in the open workspace with their row counts.

            Returns:
                List of ``{"entity_id", "name", "count", "system"}`` cards. Call
                this first to discover what entities exist.
            """
            return [{k: card[k] for k in ("entity_id", "name", "count", "system")} for card in tool_api.entities(include_system=True)]

        def entity_schema(entity_id: str) -> Dict[str, Any]:
            """Return the real field schema and storage placement for one entity.

            Args:
                entity_id: Registered entity id, e.g. ``"paper"``.

            Returns:
                ``{"schema", "storage", "capabilities"}`` with true field names,
                dtypes, pk/required/default flags, and per-field storage bindings.
            """
            info = tool_api.entity_schema(entity_id)
            fields = {
                name: {
                    "dtype": (field.get("dtype") or {}).get("type") if isinstance(field.get("dtype"), dict) else field.get("dtype"),
                    "required": field.get("required"),
                    "pk": field.get("pk"),
                    "desc": field.get("desc"),
                }
                for name, field in info["schema"].get("fields", {}).items()
            }
            return {"entity": entity_id, "desc": info["schema"].get("desc"), "fields": fields, "storage": info["storage"], "capabilities": info["capabilities"]}

        def sample_rows(entity_id: str, limit: int = 5) -> Dict[str, Any]:
            """Return a few real rows from an entity to ground answers in actual data.

            Args:
                entity_id: Registered entity id.
                limit: Number of rows to sample (max 20).

            Returns:
                ``{"rows", "columns", "total"}`` for the entity.
            """
            page = tool_api.entity_rows(entity_id, limit=max(1, min(int(limit), 20)))
            return {"rows": page["rows"], "columns": page["columns"], "total": page["total"]}

        def column_stats(entity_id: str, field: str) -> Dict[str, Any]:
            """Return value statistics for one column (nulls, distinct, top values, numeric min/max/mean).

            Every statistic is folded by the engine over every row, so
            ``scanned`` equals ``total`` and ``sampled`` is always false — these
            numbers are not a sample and need no hedging. What can still be
            elided is display, and it says so: a most-common list cut at its
            limit sets ``<key>_truncated``, and a value cut for length carries
            ``value_truncated``/``value_len``. For a Json column, ``scalars``
            and ``scalar_values`` cover the rows that were neither object nor
            list, which the key and element numbers do not describe.
            """
            return tool_api.column_stats(entity_id, field)

        def search_workspace(text: str) -> Dict[str, Any]:
            """Search the workspace Catalog/MetaSchema cards by free text.

            Returns the top-ranked cards with ``rows``, ``n``, ``total``, and
            ``truncated``. Ranking sees every matching card before the cap, so
            when ``truncated`` is true ``total`` is the exact number that
            matched — report ``total``, not the number of rows you received.
            """
            cap = 15
            result = dict(tool_api.search(text, limit=cap))
            if result.get("truncated"):
                result["note"] = (
                    f"TRUNCATED: {result['total']} cards matched and only the top {cap} are returned; "
                    f"report {result['total']} as the count, not {result['n']}."
                )
            return result

        # Fail closed: only a real boolean ``true`` grants credential access;
        # malformed truthy strings such as ``"false"`` must stay masked.
        share_credentials = not read_only and self.cm.get("heavenbase.dashboard.assistant.share_credentials", default=False) is True
        if share_credentials:
            warnings.warn(
                "SECURITY WARNING: dashboard assistant credential access is enabled; get_config may send API keys, tokens, passwords, and other credentials to the external model provider",
                UserWarning,
                stacklevel=2,
            )

        def get_config(key: str) -> Any:
            """Read one HeavenBase config value by dotted key, e.g. ``"heavenbase.llm.default_preset"``.

            Credentials are recursively masked as ``***`` by default. Raw
            credentials are returned only when the user explicitly enables
            dashboard assistant credential access.
            """
            value = self.cm.get(key, default=None)
            if share_credentials:
                return value
            leaf = str(key).rsplit(".", 1)[-1]
            return redact_data({leaf: value}, replacement="***")[leaf]

        tools = [entity_schema, sample_rows, column_stats, get_config]
        if workspace is not None:
            query_tools = [self._assist_query_tool(workspace, frames=frames, read_only=read_only), self._assist_explain_tool(workspace)]
            tools = [list_entities, *tools, *query_tools, search_workspace]
            active_names = {str(getattr(tool, "name", getattr(tool, "__name__", ""))) for tool in tools}
            for tool in self._assist_profile_tools(workspace, read_only=read_only, evidence_frames=evidence_frames):
                name = str(getattr(tool, "name", getattr(tool, "__name__", "")))
                if name and name not in active_names:
                    tools.append(tool)
                    active_names.add(name)
            for identifier, tool in self._assist_contributed_tools(workspace):
                name = str(getattr(tool, "__name__", "") or "")
                if read_only and not _assist_tool_read_only(tool):
                    continue  # contributed tools opt into read-only mode explicitly
                if not name or name in active_names:
                    warnings.warn(
                        f"assistant_tools contribution {identifier!r} skipped tool {name or '<unnamed>'!r}: an earlier tool already serves that name",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                tools.append(tool)
                active_names.add(name)
            if not read_only and self.cm.get("heavenbase.dashboard.assistant.run_python", default=False) is True:
                tools.extend(self._assist_python_tools(workspace))
        served_names = {str(getattr(tool, "name", getattr(tool, "__name__", ""))) for tool in tools}
        for tool in self._assist_mcp_tools(read_only=read_only):
            if tool.__name__ in served_names:
                warnings.warn(
                    f"assistant MCP tool {tool.__name__!r} skipped: an earlier tool already serves that name",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            tools.append(tool)
            served_names.add(tool.__name__)
        return [_recorded(_narrated(tool, progress), evidence_log) for tool in tools]

    def _assist_mcp_tools(self, read_only: bool) -> List[Any]:
        """Import configured MCP servers through the shared FastMCP adapter.

        Each ``heavenbase.dashboard.assistant.mcp_servers`` entry declares a
        name and exactly one FastMCP transport source: an HTTP ``url`` or a
        stdio ``command`` (string plus ``args``, or an argv list). An optional
        ``allowlist`` narrows the import. When
        ``read_only`` is absent, official per-tool ``readOnlyHint`` annotations
        decide read-only visibility; a server-level boolean is an explicit
        operator override. Malformed or unavailable servers warn once per
        instance and exact config fingerprint without affecting other tools.

        Args:
            read_only (bool): The dashboard mode being served.

        Returns:
            List[Any]: ``mcp__<server>__<tool>`` callables ready for the
            shared narration/evidence/clip pipeline.
        """
        servers = self.cm.get("heavenbase.dashboard.assistant.mcp_servers", default=None) or []
        tools: List[Any] = []
        if not isinstance(servers, (list, tuple)):
            self._assist_mcp_warn_once(
                f"config:{_assist_config_fingerprint(servers)}",
                "heavenbase.dashboard.assistant.mcp_servers must be a list; external MCP tools were skipped",
            )
            return tools
        for raw in servers:
            name = str(raw.get("name") or "") if isinstance(raw, Mapping) else ""
            fingerprint = _assist_config_fingerprint(raw)
            try:
                if not isinstance(raw, Mapping) or not name.strip():
                    raise TypeError("each server must be a mapping with a non-empty name")
                spec = dict(raw)
                allow = spec.get("allowlist")
                if allow is not None and (not isinstance(allow, (list, tuple)) or not all(isinstance(item, str) for item in allow)):
                    raise TypeError("allowlist must be a list of tool names")
                allowed = set(allow) if allow is not None else None
                stance = spec.get("read_only")
                if stance is not None and not isinstance(stance, bool):
                    raise TypeError("read_only must be a boolean")
                from heavenbase.toolkit.mcp import callables_from_fastmcp

                server_name = re.sub(r"[^A-Za-z0-9_-]", "_", name.strip())
                for tool in callables_from_fastmcp(_assist_mcp_source(spec)):
                    native_name = str(getattr(tool, "__name__", "") or "")
                    if not native_name or (allowed is not None and native_name not in allowed):
                        continue
                    tool_read_only = stance if stance is not None else _assist_tool_read_only(tool)
                    if not read_only or tool_read_only:
                        tools.append(_assist_mcp_tool(tool, server_name, tool_read_only))
            except Exception as error:
                self._assist_mcp_warn_once(
                    f"server:{name}:{fingerprint}",
                    f"assistant MCP server {name or '<unnamed>'!r} is unavailable and serves no tools: {error}",
                )
        return tools

    def _assist_mcp_warn_once(self, key: str, message: str) -> None:
        """Warn once for one exact external MCP configuration failure."""
        with self._assist_mcp_lock:
            first = key not in self._assist_mcp_warned
            self._assist_mcp_warned.add(key)
        if first:
            warnings.warn(message, UserWarning, stacklevel=3)

    def _assist_contributed_tools(self, workspace: Any) -> List[Tuple[str, Any]]:
        """Return ``(identifier, tool)`` pairs from ``assistant_tools`` contributions.

        The open contribution kind for the assistant: an installed module
        declares a Registry row of kind ``assistant_tools`` whose target
        resolves to a provider callable ``provider(workspace) -> list`` of
        plain tool callables (``__name__``/``__doc__`` and ``__signature__``
        or ``__tool_schema__`` carry the schema; an optional ``read_only=True``
        attribute or standard ``readOnlyHint`` annotation opts a tool into
        read-only mode). Contributions ride the same
        ``meta.workspace_requires`` eligibility policy as every other
        contribution kind, and the caller routes them through the same
        narration, evidence capture, and collision handling as built-ins.
        Providers own ordinary callable composition: selection, partial
        binding, documentation, and result provenance stay in provider code
        instead of creating a second tool-policy DSL. A failing or malformed
        provider is skipped with a warning so an extension bug cannot take the
        assistant down.

        Args:
            workspace: Immutable workspace snapshot for one assistant turn.

        Returns:
            List[Tuple[str, Any]]: Contribution identifier and tool callable
            pairs, in registry order.
        """
        from heavenbase.extensions.contributions import eligible_contributions

        resolver = workspace.module_resolver
        pairs: List[Tuple[str, Any]] = []
        for row in eligible_contributions(resolver.inspect("assistant_tools"), workspace):
            identifier = str(row.get("identifier") or "")
            try:
                provider = resolver.resolve("assistant_tools", identifier)
                served = list(provider(workspace) or ())
                if not all(callable(tool) for tool in served):
                    raise TypeError("provider must return callable tools")
            except Exception as error:
                warnings.warn(f"assistant_tools contribution {identifier!r} failed and was skipped: {error}", UserWarning, stacklevel=2)
                continue
            pairs.extend((identifier, tool) for tool in served)
        return pairs

    def _assist_profiles(self, workspace: Any) -> List[tuple[Any, Dict[str, Any]]]:
        """Return active extension profiles that opt into dashboard chat.

        Args:
            workspace: Immutable workspace snapshot for one assistant turn.

        Returns:
            Profile specs paired with validated dashboard-assistant metadata.

        Raises:
            TypeError: If an opted-in profile has malformed metadata.
        """
        from heavenbase.toolkit.profiles import available_profiles

        profiles = available_profiles(
            resolver=workspace.module_resolver,
            config=workspace.context.config,
            workspace=workspace,
        )
        selected: List[tuple[Any, Dict[str, Any]]] = []
        for spec in profiles.values():
            raw = spec.meta.get(_DASHBOARD_ASSISTANT_PROFILE_META)
            if raw is None:
                continue
            if not isinstance(raw, Mapping):
                raise TypeError(f"MCP profile {spec.name!r} {_DASHBOARD_ASSISTANT_PROFILE_META} metadata must be a mapping")
            settings = dict(raw)
            if settings.get("enabled", True) is True:
                selected.append((spec, settings))
        return selected

    def _assist_profile_tools(
        self,
        workspace: Any,
        *,
        read_only: bool,
        evidence_frames: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Any]:
        """Build extension-owned tools allowed for one assistant mode.

        Args:
            workspace: Immutable workspace snapshot for one assistant turn.
            read_only: Whether mutation-capable extension tools must stay hidden.

        Returns:
            Runtime Tool objects selected by active MCP profile metadata.

        Raises:
            TypeError: If a profile tool allowlist is not a string list.
            ValueError: If a profile names a tool outside its resolved Toolkit.
        """
        field = "read_only_tools" if read_only else "interactive_tools"
        tools: List[Any] = []
        for spec, settings in self._assist_profiles(workspace):
            names = settings.get(field, [])
            if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) and name.strip() for name in names):
                raise TypeError(f"MCP profile {spec.name!r} {field} must be a string list")
            toolkit = workspace.to_mcp(profile=spec.name)
            frame_specs = settings.get("evidence_frames", {}) or {}
            if not isinstance(frame_specs, Mapping) or not all(isinstance(name, str) and isinstance(value, Mapping) for name, value in frame_specs.items()):
                raise TypeError(f"MCP profile {spec.name!r} evidence_frames must map tool names to frame specifications")
            for name in names:
                tool = toolkit.get(name)
                if tool is None:
                    raise ValueError(f"MCP profile {spec.name!r} dashboard assistant tool {name!r} is unavailable")
                tools.append(_evidenced(tool, evidence_frames, frame_specs[name]) if name in frame_specs else tool)
        return tools

    def _assist_query_tool(
        self,
        workspace: Any = None,
        frames: Optional[List[Dict[str, Any]]] = None,
        read_only: bool = False,
    ) -> Any:
        """Build the MCP agent-profile ``query`` tool bound to the open workspace.

        Reuses the registered workspace toolkit implementation (entity name
        resolution, frame output, redacted errors) and clamps ``limit`` so one
        assistant call cannot pull unbounded rows. The full JSON spec is
        accepted, including ``group_by``/``aggregate`` engine-side aggregation.

        Aggregation runs before ordering, offset, and limit (ADR 0009), so an
        aggregate ``limit`` truncates result groups rather than source rows.
        Aggregate calls therefore get their own, larger group cap, keep an
        explicit caller limit, and default to the cap instead of the row
        default. Result frames carry the exact pre-pagination total, so every
        cut is proved and reported instead of silently applied.

        Args:
            workspace (Any): Workspace snapshot the tool binds to. Defaults to
                the currently open workspace.
            frames (List[Dict[str, Any]] | None): Executed-call capture records,
                with successful frames retained up to the response limit.
                Defaults to ``None``.

        Returns:
            Any: ``query(entity, spec)`` callable for the assist tool list.
        """
        from heavenbase.extensions.contributions import resolve_contribution

        workspace = self._open_ws() if workspace is None else workspace
        build_query_tool = resolve_contribution(
            "toolkit_function",
            "workspace_query_tool",
            resolver=workspace.module_resolver,
            workspace=workspace,
            callable_only=True,
        )
        agent_query = build_query_tool(workspace, profile="agent")

        def query(entity: str, spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            capture = None
            if frames is not None:
                call_args: Dict[str, Any] = {"entity": entity}
                if spec is not None:
                    call_args["spec"] = dict(spec) if isinstance(spec, dict) else spec
                capture = {"args": call_args, "frame": None}
                frames.append(capture)
            if isinstance(spec, str):  # some models double-encode the nested spec argument
                from heavenbase.utils.serialize import loads_json

                spec = loads_json(spec)
            spec = dict(spec or {})
            if read_only and _query_uses_code_reduce(spec, workspace.module_resolver):
                raise ValueError("assistant read-only mode does not allow code_reduce")
            aggregate = bool(spec.get("aggregate"))
            cap = self._assist_query_cap(aggregate)
            # An aggregate caller asked for result groups, so an unset limit
            # means "every group up to the cap", not the small row default.
            requested = spec.get("limit") if "limit" in spec else (cap if aggregate else _DEFAULT_ASSIST_QUERY_LIMIT)
            if isinstance(requested, bool) or not isinstance(requested, int):
                raise TypeError("assistant query limit must be an integer")
            if requested < 1:
                raise ValueError("assistant query limit must be positive")
            limit = min(requested, cap)
            spec["limit"] = limit
            result = agent_query(entity, spec)
            if isinstance(result, dict):
                result = _bounded_query_result(result, limit, cap, aggregate)
            if capture is not None and isinstance(result, dict) and sum(isinstance(item.get("frame"), dict) for item in frames) < self._assist_frame_limit():
                frame = {
                    "entity": entity,
                    "spec": spec,
                    "columns": list(result.get("columns") or {}),
                    "rows": list(result.get("rows") or []),  # already bounded by the applied limit
                    "n": result.get("n", 0),
                    "total": result.get("total"),
                    "truncated": bool(result.get("truncated")),
                }
                # window == model: a capture-clipped result must not render rows
                # the model never saw.
                capture["frame"] = _capture_clipped_frame(frame, result, self._assist_result_capture())
            return result

        # One source of truth: expose the MCP tool's own spec documentation to
        # the assistant, plus the dashboard-specific guardrail and rendering notes.
        query.__doc__ = (
            f"{agent_query.__doc__}\n\nDashboard notes: plain reads return at most {self._assist_query_cap(False)} rows and "
            f"aggregate reads at most {self._assist_query_cap(True)} groups (aggregation runs before limit, so an aggregate "
            "``limit`` truncates result groups, not source rows; an unset aggregate ``limit`` uses the group cap). Every result "
            "carries the applied ``limit`` and a ``truncated`` flag; when ``truncated`` is true, ``total`` is the exact number "
            "that matched — report that, not the number of rows you received. Results are also rendered to the user as tables "
            "automatically."
        )
        return query

    def _assist_python_tools(self, workspace: Any = None) -> List[Any]:
        """Build the opt-in Python execution tools for the assistant.

        The tools come from the ``python`` toolkit family, so the dashboard and
        an MCP client run the same code path. Enabling them lets the model
        answer questions the query spec cannot express, at the cost of the
        read-only property the other tools hold: the executed program runs in
        this process with the workspace bound.

        Args:
            workspace (Any): Workspace snapshot the tools bind to. Defaults to
                the currently open workspace.

        Returns:
            List[Any]: ``run_python`` and ``python_guide`` callables.

        Raises:
            RuntimeError: If the ``python`` extension cannot be resolved.
        """
        from heavenbase.extensions.python.toolkit import python_tool_functions

        workspace = self._open_ws() if workspace is None else workspace
        functions = python_tool_functions(workspace)
        return [functions["python_guide"], functions["run_python"]]

    def _assist_explain_tool(self, workspace: Any = None) -> Any:
        """Build the MCP agent-profile ``explain`` tool bound to the open workspace.

        Explain validates a spec and returns the router's plan without running
        it, giving the assistant a zero-row check of filters, dotted json
        paths, and ``group_by``/``aggregate`` shapes before it spends a real
        read. No rows leave the engine, so the query row and group caps do not
        apply here.

        Args:
            workspace (Any): Workspace snapshot the tool binds to. Defaults to
                the currently open workspace.

        Returns:
            Any: ``explain(entity, spec)`` callable for the assist tool list.
        """
        from heavenbase.extensions.contributions import resolve_contribution

        workspace = self._open_ws() if workspace is None else workspace
        build_explain_tool = resolve_contribution(
            "toolkit_function",
            "workspace_explain_tool",
            resolver=workspace.module_resolver,
            workspace=workspace,
            callable_only=True,
        )
        agent_explain = build_explain_tool(workspace, profile="agent")

        def explain(entity: str, spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            if isinstance(spec, str):  # some models double-encode the nested spec argument
                from heavenbase.utils.serialize import loads_json

                spec = loads_json(spec)
            return agent_explain(entity, dict(spec or {}))

        explain.__doc__ = (
            f"{agent_explain.__doc__}\n\nDashboard notes: explain runs nothing and returns no rows, so no row or group cap "
            "applies; use it to validate a spec and read its plan before calling query, and never report plan numbers as data."
        )
        return explain

    def _assist_frame_limit(self) -> int:
        """Return the configured per-turn query frame cap.

        The high bound catches typos rather than policing display: an operator
        may raise the cap to 100000 frames for effectively full evidence.
        """
        return _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.max_frames", _MAX_ASSIST_FRAMES, 1, 100000)

    def _assist_tool_turns(self) -> int:
        """Return the tool-round budget one assistant call hands the LLM loop."""
        return _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.max_tool_turns", _MAX_ASSIST_TOOL_TURNS, 1, 10000)

    def _assist_history_turns(self) -> int:
        """Return how many prior turns one assistant call replays."""
        return _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.history_turns", _ASSIST_HISTORY_TURNS, 1, 1000)

    def _assist_evidence_budget(self) -> int:
        """Return the total evidence-replay character budget; ``0`` disables the carry."""
        return _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.evidence_total", _ASSIST_EVIDENCE_TOTAL, 0, 10000000)

    def _assist_evidence_caps(self) -> Tuple[int, int]:
        """Return the per-turn ``(full, brief)`` evidence rendering caps."""
        return (
            _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.evidence_turn_chars", _ASSIST_EVIDENCE_TURN_CHARS, 256, 10000000),
            _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.evidence_brief_chars", _ASSIST_EVIDENCE_BRIEF_CHARS, 64, 100000),
        )

    def _assist_store_evidence(
        self,
        text: str,
        calls: List[Tuple[str, str, str]],
        caps: Tuple[int, int],
        *,
        question: str = "",
        scope: Optional[str] = None,
    ) -> None:
        """Render and store one turn's tool evidence under its scoped identity.

        Args:
            text (str): Final answer text. Empty answers store nothing.
            calls (List[Tuple[str, str, str]]): Recorded ``(name, args,
                result)`` invocations in execution order.
            caps (Tuple[int, int]): ``(full, brief)`` per-turn rendering caps.
            question (str): User question paired with the answer. Defaults to
                an empty string for direct callers.
            scope (str | None): Workspace or conversation namespace. Defaults
                to the currently open workspace id.
        """
        if not calls or not text.strip():
            return
        full_cap, brief_cap = caps
        full = _assist_evidence_tail([f"{name}({args}) -> {result}" for name, args, result in calls], full_cap)
        brief = _assist_evidence_tail(
            [f"{name}({args[:_ASSIST_EVIDENCE_BRIEF_ARGS]}) -> {result[:_ASSIST_EVIDENCE_BRIEF_RESULT]}" for name, args, result in calls],
            brief_cap,
        )
        key = _assist_evidence_key(text, question, str(self.ws_id or "") if scope is None else scope)
        with self._assist_evidence_lock:
            self._assist_evidence[key] = (full, brief)
            self._assist_evidence.move_to_end(key)
            while len(self._assist_evidence) > _ASSIST_EVIDENCE_ENTRIES:
                self._assist_evidence.popitem(last=False)

    def _assist_carry_history(self, history: List[Any], budget: int, scope: Optional[str] = None) -> List[Any]:
        """Return replayed turns with stored tool evidence re-attached.

        Newest turns first, so the freshest evidence survives a tight budget:
        each assistant turn gets its full rendering while ``budget`` allows,
        else the brief one, else nothing. The attached header counts against
        the budget, turns already carrying the header marker are skipped, and
        malformed entries pass through untouched for the caller's own history
        validation to handle. A ``budget`` of ``0`` returns the turns as they
        came — the stock prose-only replay.
        """
        turns = list(history or [])
        if not budget:
            return turns
        with self._assist_evidence_lock:
            store = dict(self._assist_evidence)
        out = list(turns)
        evidence_scope = str(self.ws_id or "") if scope is None else scope
        for index in range(len(out) - 1, -1, -1):
            turn = out[index]
            role = turn.get("role") if hasattr(turn, "get") else None
            content = turn.get("content") if hasattr(turn, "get") else None
            previous = out[index - 1] if index > 0 else None
            question = previous.get("content") if hasattr(previous, "get") and previous.get("role") == "user" else ""
            entry = (
                store.get(_assist_evidence_key(content, question if isinstance(question, str) else "", evidence_scope))
                if role == "assistant" and isinstance(content, str) and _ASSIST_EVIDENCE_HEADER not in content
                else None
            )
            for rendering in entry or ():  # the full rendering first, then the brief fallback
                block = f"\n\n{_ASSIST_EVIDENCE_HEADER}\n{rendering}"
                if rendering and len(block) <= budget:
                    budget -= len(block)
                    turn = {**turn, "content": f"{content}{block}"}
                    break
            out[index] = turn
        return out

    def _assist_route_model(self, preset: Optional[str] = None) -> Tuple[Optional[str], Any]:
        """Resolve the turn's route to its canonical model registry entry.

        Args:
            preset (str | None): The turn's request preset. Defaults to ``None``.

        Returns:
            Tuple[Optional[str], Any]: ``(canonical name, registry entry)``, or
            ``(None, None)`` when the route names no registered model.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the turn's own resolution warned already
            options = self._assist_llm_options(preset=preset)
        model = options.get("model")
        if not model and options.get("preset"):
            presets = self.cm.get("heavenbase.llm.presets", default=None) or {}
            entry = presets.get(options["preset"])
            model = entry.get("model") if hasattr(entry, "get") else None
        if not model:
            return None, None
        registry = self.cm.get("heavenbase.llm.models", default=None) or {}
        if model in registry:
            return model, registry[model]
        for name, item in registry.items():
            aliases = item.get("aliases") if hasattr(item, "get") else None
            if aliases and model in aliases:
                return name, item
        return None, None

    def _assist_context_window(self, preset: Optional[str] = None) -> int:
        """Resolve the turn's model context window in tokens.

        Explicit ``context_window`` config beats the conservative 128000-token
        fallback. Window discovery belongs to provider/model metadata, not an
        unsolicited request from the dashboard.
        """
        _, record = self._assist_route_model(preset)
        if hasattr(record, "get"):
            window = record.get("context_window")
            if isinstance(window, int) and not isinstance(window, bool) and window > 0:
                return window
        return _ASSIST_CONTEXT_WINDOW_FALLBACK

    def _assist_history_chars(self, preset: Optional[str] = None) -> int:
        """Return the assembled-history character threshold.

        An explicit ``history_chars`` config value is an absolute override
        (``0`` keeps the legacy per-turn ladder). Unset, the threshold tracks
        the model: 55% of the resolved context window, converted at the
        conservative 2.5 chars/token.
        """
        if self.cm.get("heavenbase.dashboard.assistant.history_chars", default=None) is not None:
            return _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.history_chars", 0, 0, 10000000)
        return int(self._assist_context_window(preset) * _ASSIST_HISTORY_WINDOW_SHARE * _CHARS_PER_TOKEN)

    def _assist_compact_chars(self) -> int:
        """Return the character budget for one compacted-prefix summary."""
        return _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.compact_chars", _ASSIST_COMPACT_CHARS, 256, 100000)

    def _assist_compact_ratio(self) -> float:
        """Return the dashboard's optional in-turn compaction threshold."""
        key = "heavenbase.dashboard.assistant.compact_ratio"
        value = self.cm.get(key, default=_ASSIST_COMPACT_RATIO)
        if isinstance(value, bool):
            raise TypeError(f"{key} must be a number")
        try:
            ratio = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{key} must be a number") from error
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
        return ratio

    def _assist_result_capture(self) -> int:
        """Return the character clip for one tool result; ``0`` sends it whole."""
        return _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.result_capture_chars", _ASSIST_RESULT_CAPTURE_CHARS, 0, 10000000)

    def _assist_replay_history(
        self,
        system: str,
        history: List[Any],
        question: str,
        *,
        turns: int,
        evidence_budget: int,
        preset: Optional[str],
        injected: bool,
        scope: str,
        trace: Any = None,
    ) -> Tuple[str, List[Any]]:
        """Return the noticed system prompt and the replayed history turns.

        The measured path governs while ``history_chars`` grants a threshold:
        the actual assembly — system prompt, windowed history with FULL
        evidence attached, and the question — is measured in characters, and
        under the threshold everything ships full with zero degradation. Over
        it, :meth:`_assist_compact_replay` folds the oldest turns into one
        summary. The legacy per-turn ladder remains the fallback: a threshold
        of ``0``, an injected runtime (which never triggers inline model
        construction), and a failed compaction all land there.

        Args:
            system (str): Assembled system prompt, measured as sent.
            history (List[Any]): Full prior-turn list from the client.
            question (str): Current user question, measured as sent.
            turns (int): Replay window in turns (``history_turns``).
            evidence_budget (int): Ladder budget; non-zero also switches
                evidence attachment on.
            preset (str | None): The turn's request preset, handed to the
                compaction route.
            injected (bool): Whether the turn runs an injected runtime.
            scope (str): Workspace/conversation namespace captured with the
                assistant tool snapshot.
            trace (Any): Optional LLM trace callback for the compaction model
                call, so the session log records it. Defaults to ``None``.

        Returns:
            Tuple[str, List[Any]]: ``(system_with_notice, replayed_turns)``.
        """
        window = list(history or [])[-turns:]
        threshold = self._assist_history_chars(preset)
        if threshold:
            attached = self._assist_carry_history(window, _ASSIST_ATTACH_ALL if evidence_budget else 0, scope)
            sizes = [len(turn.get("content")) if hasattr(turn, "get") and isinstance(turn.get("content"), str) else 0 for turn in attached]
            base = len(system) + len(question)
            if base + sum(sizes) <= threshold:
                # Under the threshold everything ships full: zero degradation.
                return _with_history_notice(system, history, turns), attached
            if not injected and window:
                try:
                    return self._assist_compact_replay(system, history, window, attached, sizes, base, threshold, preset=preset, scope=scope, trace=trace)
                except Exception:
                    # Never fail the question because compaction failed: the
                    # ladder below still bounds this request.
                    pass
        return _with_history_notice(system, history, turns), self._assist_carry_history(window, evidence_budget, scope)

    def _assist_compact_replay(
        self,
        system: str,
        history: List[Any],
        window: List[Any],
        attached: List[Any],
        sizes: List[int],
        base: int,
        threshold: int,
        *,
        preset: Optional[str],
        scope: str,
        trace: Any = None,
    ) -> Tuple[str, List[Any]]:
        """Compact the oldest turns into one summary and replay the rest full.

        The minimal prefix is cut so that the remaining suffix — with full
        evidence — fits the threshold minus the summary's own budget and the
        notice boilerplate, then extended so the verbatim suffix opens on a
        user turn (role alternation stays intact behind the system notice).

        Args:
            system (str): Assembled system prompt.
            history (List[Any]): Full prior-turn list, for the notice counts.
            window (List[Any]): Raw windowed turns, aligned with ``attached``;
                their content keys identify the prefix in the summary cache.
            attached (List[Any]): Windowed turns with full evidence attached.
            sizes (List[int]): Measured content size per attached turn.
            base (int): Measured system + question size.
            threshold (int): ``history_chars`` for this request.
            preset (str | None): The turn's request preset for the summary call.
            scope (str): Workspace/conversation namespace for the cache key.

        Returns:
            Tuple[str, List[Any]]: ``(system_with_summary_notice, suffix_turns)``.

        Raises:
            Exception: Whatever the one compaction model call raises; the
                caller falls back to the per-turn ladder.
        """
        room = threshold - base - self._assist_compact_chars() - _ASSIST_COMPACT_OVERHEAD
        running = sum(sizes)
        cut = 0
        while cut < len(attached) and running > room:
            running -= sizes[cut]
            cut += 1
        # The verbatim suffix opens on a user turn so alternation stays intact.
        while cut < len(attached) and (attached[cut].get("role") if hasattr(attached[cut], "get") else None) != "user":
            running -= sizes[cut]
            cut += 1
        prefix, suffix = attached[:cut], attached[cut:]
        route, _ = self._assist_route_model(preset)
        keys = (
            sha256hash(f"{scope}\0{route or preset or ''}")[:16],
            *(sha256hash(dumps_jstr(turn, compact=True))[:16] if isinstance(turn, Mapping) else sha256hash(str(turn))[:16] for turn in attached[:cut]),
        )
        summary = self._assist_compact_prefix(prefix, keys, preset, trace=trace)
        dropped = max(0, len(history or []) - len(window))
        noticed = (
            f"{system}\nSummary of the earlier conversation (compacted): {summary}\n"
            f"Only the last {len(suffix)} of {len(history or [])} prior turns are in your context verbatim; "
            f"{len(prefix)} older turns were compacted into the summary above"
            + (f", and {dropped} yet older turns were not sent" if dropped else "")
            + ". Say so rather than implying you have seen the whole conversation."
        )
        return noticed, suffix

    def _assist_compact_prefix(self, prefix: List[Any], keys: Tuple[str, ...], preset: Optional[str], trace: Any = None) -> str:
        """Summarize one turn prefix into conversation memory, cached per prefix.

        A stable prefix pays its compaction call once: an exact cache hit
        returns without a model call, and when the prefix has grown past a
        cached shorter one, only the old summary plus the newly overflowed
        turns are summarized rather than starting from scratch.

        Args:
            prefix (List[Any]): Oldest windowed turns, with full evidence
                attached, in conversation order.
            keys (Tuple[str, ...]): Content keys identifying the prefix.
            preset (str | None): The turn's request preset, re-resolved through
                :meth:`_assist_llm_options` with warnings suppressed.

        Returns:
            str: Non-empty summary, hard-clipped to ``compact_chars``.

        Raises:
            ValueError: If the model returns an empty summary.
            Exception: Whatever the model call itself raises.
        """
        compact_chars = self._assist_compact_chars()
        seed: Optional[Tuple[Tuple[str, ...], str]] = None
        with self._assist_compact_lock:
            cached = self._assist_compact_cache.get(keys)
            if cached is not None:
                self._assist_compact_cache.move_to_end(keys)
                return cached
            for known, known_summary in self._assist_compact_cache.items():
                if len(known) < len(keys) and keys[: len(known)] == known and (seed is None or len(known) > len(seed[0])):
                    seed = (known, known_summary)
        lines: List[str] = []
        if seed is not None:
            lines.append(f"Summary of the conversation so far:\n{seed[1]}")
            lines.append("Newly elapsed turns to fold in:")
        # The first cache-key item is the route/scope namespace rather than a
        # conversation turn, so exclude it when locating newly elapsed turns.
        elapsed = len(seed[0]) - 1 if seed is not None else 0
        for turn in prefix[elapsed:]:
            if hasattr(turn, "get") and isinstance(turn.get("content"), str):
                lines.append(f"{turn.get('role')}: {turn.get('content')}")
        from heavenbase.llm import LLM

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the turn's own resolution warned already
            options = self._assist_llm_options(preset=preset)
        reply = LLM(context=self.context, cache=False, **options).chat(
            "Compress the earlier turns of an ongoing assistant conversation into conversation memory, not prose. "
            "Preserve every concrete value, decision, identifier, and open question. "
            f"At most {compact_chars} characters.\n\n" + "\n\n".join(lines),
            trace=trace,
        )
        # ``chat`` returns a plain string by default and a mapping under includes.
        summary = str((reply.get("text") if hasattr(reply, "get") else reply) or "").strip()[:compact_chars]
        if not summary:
            raise ValueError("assist compaction returned an empty summary")
        with self._assist_compact_lock:
            self._assist_compact_cache[keys] = summary
            self._assist_compact_cache.move_to_end(keys)
            while len(self._assist_compact_cache) > _ASSIST_COMPACT_CACHE_ENTRIES:
                self._assist_compact_cache.popitem(last=False)
        return summary

    def _assist_result_limit(self) -> int:
        """Return the configured non-query tool-result transport limit.

        The high bound catches typos rather than policing display: an operator
        may raise the limit to 100000000 characters for effectively untruncated
        tool results.
        """
        return _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.max_tool_result_chars", _DEFAULT_MAX_ASSIST_RESULT_CHARS, 256, 100000000)

    def _assist_query_cap(self, aggregate: bool = False) -> int:
        """Return the configured result cap for one assistant query.

        Args:
            aggregate (bool): Whether the spec aggregates, so the cap governs
                result groups instead of source rows. Defaults to ``False``.

        Returns:
            int: Largest row or group count one assistant query may return.
        """
        if aggregate:
            return _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.max_group_rows", _MAX_ASSIST_GROUP_ROWS, 1, 100000)
        return _assist_int_setting(self.cm, "heavenbase.dashboard.assistant.max_query_rows", _MAX_ASSIST_QUERY_ROWS, 1, 10000)

    def _assist_llm_options(self, preset: Optional[str] = None) -> Dict[str, Any]:
        """Return ``LLM`` construction kwargs for the assistant.

        Reads ``heavenbase.dashboard.assistant`` (``preset``/``model``/
        ``provider``/``gateway``) and falls back to the configured LLM default
        preset when nothing is set. ``default_args`` ride along as the LLM's
        ``default_args`` except where the selected preset already defines the key, and retries
        stay off unless the assistant route asks for them.

        Returns:
            Dict[str, Any]: Keyword arguments for :class:`heavenbase.llm.LLM`.
        """
        config = self.cm.get("heavenbase.dashboard.assistant", default=None) or {}
        requested_preset = str(preset).strip() if preset is not None else ""
        if requested_preset:
            presets = self.cm.get("heavenbase.llm.presets", default=None) or {}
            if requested_preset not in presets:
                raise ValueError(f"Assistant preset {requested_preset!r} is not registered under heavenbase.llm.presets")
            options = {"preset": requested_preset}
        elif config.get("preset"):  # a preset selects the model bundle, while default_args remain request-level overrides
            options = {"preset": config["preset"]}
        else:
            options = {key: config[key] for key in ("model", "provider", "gateway") if config.get(key)}
        if options.get("preset"):
            # A preset — requested or configured — already carries the whole
            # route, so a model set beside it never reaches the provider. Say
            # so once instead of serving a different model than the config
            # plainly reads as.
            shadowed = [key for key in ("model", "provider", "gateway") if config.get(key)]
            if shadowed:
                warnings.warn(
                    f"heavenbase.dashboard.assistant {', '.join(shadowed)} ignored: preset {options['preset']!r} selects the assistant route; "
                    "clear the preset to route by model/provider/gateway",
                    UserWarning,
                    stacklevel=2,
                )
        model = options.get("model")
        if model:
            # Scope the failure to the assistant's own model list instead of
            # letting the LLM layer dump the full global model registry.
            registry = self.cm.get("heavenbase.llm.models", default=None) or {}
            known = set(registry)
            for item in registry.values():
                aliases = item.get("aliases") if hasattr(item, "get") else None
                known.update(aliases or [])
            if model not in known:
                raise ValueError(
                    f"Assistant model {model!r} is not registered under heavenbase.llm.models; pick a registered model or register it there first."
                )
        if not options:
            options = {"preset": str(self.cm.get("heavenbase.llm.default_preset", "system"))}
        # A preset's own default_args are part of the bundle it selects, not a
        # weaker default under it: letting machine-level config win there is how
        # ``think: true`` quietly turned the non-thinking 'chat' preset thinking.
        selected = options.get("preset")
        entry = (self.cm.get("heavenbase.llm.presets", default=None) or {}).get(selected) if selected else None
        owned = entry.get("default_args") if hasattr(entry, "get") else None
        # The mapping guard mirrors the ``extra`` one below: a preset whose
        # default_args is a scalar must not take down every assist call.
        reserved = set(options) | set(owned if hasattr(owned, "items") else ())
        extra = config.get("default_args")
        if hasattr(extra, "items"):  # preset-style request defaults ride along as LLM kwargs
            dropped = sorted(key for key in dict(extra) if key in reserved)
            if dropped:
                warnings.warn(
                    f"heavenbase.dashboard.assistant.default_args {', '.join(dropped)} ignored: the selected assistant route already owns them",
                    UserWarning,
                    stacklevel=2,
                )
            # Request arguments go in ``default_args``, where they stay configured defaults: a Claude
            # thinking route then leaves out a configured temperature instead of refusing it as the
            # caller's. What LLM(...) reads only as a keyword (the route selectors, its authority and
            # cache options, and callables) stays a keyword, as before.
            for key, value in dict(extra).items():
                if key in reserved:
                    continue
                if key in _LLM_KEYWORD_OPTIONS or callable(value):
                    options[key] = value
                else:
                    options.setdefault("default_args", {})[key] = value
        if "max_retries" not in reserved and "max_retries" not in options.get("default_args", {}):
            # The dashboard is an interactive panel with no retry feedback, so a
            # stalled call resent three times is ~480s of blank UI. Fail fast
            # unless the route explicitly asks for retries.
            options.setdefault("max_retries", 0)
        return options

    def _assist_system(self, context: Dict[str, Any], preset: Optional[str] = None, read_only: bool = False) -> str:
        from heavenbase.utils.serialize import dumps_json

        lines = [
            "You are the HeavenBase dashboard assistant, a local workbench over one HeavenBase workspace.",
            "Core tools are list_entities, entity_schema, sample_rows, column_stats, query, explain, search_workspace, and get_config. Loaded extension profiles may add tools and instructions below. Core tools read workspace data; query can also run a code_reduce expression, which executes in the server process, so keep those expressions to pure reductions over the values you are handed.",
            "Use query for filtered or sorted reads (JSON spec keys: filter/select/order/offset/limit; Mongo-style operators like $gte/$in inside filter).",
            'Nested values inside json fields filter engine-side with dotted paths, e.g. {"filter": {"meta.venue": "NeurIPS"}} — combine freely with other filters and group_by/aggregate; never fetch rows to count client-side when a filter can do it.',
            'group_by/order/select accept declared fields and json dotted paths directly, e.g. {"group_by": ["meta.venue"], "aggregate": {"n": {"count": {"field": "object_id"}}}} — the output column is named by the path, missing/null paths land in one None group (sorting last), and dotted clauses run engine-side on the portable Python route (fine at dashboard scale). Never fan out per-value filtered queries or fetch rows to group client-side when one dotted group_by can do it.',
            'For counts and per-group statistics use engine-side aggregation in query: group_by (list of fields) + aggregate mapping each output name to one {op: {"field": field, ...options}}, e.g. {"group_by": ["author"], "aggregate": {"papers": {"count": {"field": "object_id"}}}}; select cannot be combined with aggregate.',
            'Registered ops: count (with "distinct": true for a distinct count), min, max, sum, avg, median, percentile (needs "q"), key_sums, key_counts, value_counts. One aggregate carries many outputs at once, so ask for everything you need in a single call, e.g. {"aggregate": {"lo": {"min": {"field": "price"}}, "hi": {"max": {"field": "price"}}}} — then do arithmetic on those exact numbers rather than over rows.',
            "Use explain(entity, spec) to validate a spec and read its plan without returning rows. Every query result carries the applied limit and a truncated flag: when truncated is true, total is the exact number that matched, so report total rather than the number of rows you received.",
            'For a Json column, pick the aggregate by what you are counting: {"key_sums": {"field": f}} totals the VALUES under each key of an '
            'object payload, {"key_counts": {"field": f}} counts the ROWS carrying each key, and {"value_counts": {"field": f}} counts occurrences '
            "of each element in a list payload. All three are engine-computed over every row and exact, and using the wrong one raises rather than "
            "returning an empty result. column_stats(entity_id, field) folds the whole column too — key and element distributions, quartiles, "
            "histograms, length distributions, time ranges — so reach for it before computing any of those yourself. Never page rows to tally by hand.",
            "When no registered op expresses the reduction, code_reduce carries the expression itself: "
            '{"aggregate": {"spread": {"code_reduce": {"field": "amount", "code": "lambda v: max(v) - min(v)", "dtype": "float"}}}}. '
            "The engine hands your expression every non-null value of the column, so its coverage matches any other aggregate, and dtype is optional. "
            "Prefer a registered op whenever one fits: those inspect the payload and raise when it does not match, while your expression folds whatever "
            "it is handed — a Boolean summed as 1, or a list where you assumed an object, yields a plausible wrong number and no error. So use it for a "
            "genuine one-off, not to re-implement an op or something column_stats already reports.",
            "State explicitly when a number you report was not computed by an engine call (eyeballed from sampled rows, added up by hand, or read off a truncated result); never present such a number as an exact count.",
            "Query result tables live inside the collapsed thinking timeline, so the user does not see them unless they expand it. Your reply must stand alone: when the user asks for a table, ranking, or list, present the relevant rows as a concise markdown table in the reply itself \u2014 never refer to a table the user cannot see. Only for large results (roughly 20+ rows) summarize the key rows instead and mention the full table is in the thinking timeline.",
            "ALWAYS call the tools to get real data before answering about entities, fields, schema, rows, stats, or config —",
            "never invent field names or values. Use the screen context below to know what the user is looking at.",
            "Answer concisely in the user's language. Config edits happen in the Config workbench.",
        ]
        if read_only:
            lines = [line for line in lines if "code_reduce" not in line]
            lines.insert(
                1,
                "Read-only mode is active. Use only the exposed inspection tools; credentials stay masked and executable Python reductions are unavailable.",
            )
        else:
            lines.append("Interactive mode is active; configured local execution tools may be available.")
        options = self._assist_llm_options(preset=preset)
        if options.get("preset"):
            identity = f"preset {options['preset']!r}"
        else:
            identity = f"model {options.get('model')!r} via provider {(options.get('provider') or 'default')!r}"
        # The model cannot know its own serving stack; without this it misreads
        # the workspace-wide llm defaults (a different layer) as its identity.
        lines.append(
            f"You yourself are served by {identity} from heavenbase.dashboard.assistant; answer identity questions from this line, not from heavenbase.llm defaults."
        )
        with self._workspace_lock:
            workspace = self.ws
            workspace_id = self.ws_id
        if workspace_id:
            lines.append(f"Open workspace: {workspace_id}")
            entities = ", ".join(sorted(workspace.entities)) if workspace is not None else ""
            if entities:
                lines.append(f"Registered entities: {entities}")
        if workspace is not None:
            field = "read_only_tools" if read_only else "interactive_tools"
            for spec, settings in self._assist_profiles(workspace):
                names = settings.get(field, [])
                if names:
                    lines.append(f"Active extension assistant profile {spec.name!r} adds tools: {', '.join(names)}")
                instructions = settings.get("instructions", [])
                if not isinstance(instructions, (list, tuple)) or not all(isinstance(item, str) and item.strip() for item in instructions):
                    raise TypeError(f"MCP profile {spec.name!r} dashboard assistant instructions must be a string list")
                lines.extend(str(item).strip() for item in instructions)
        for key in ("route", "entity", "schema"):
            if context.get(key):
                value = context[key] if isinstance(context[key], str) else dumps_json(context[key])
                lines.append(f"Current {key}: {value[:4000]}")
        return "\n".join(lines)
