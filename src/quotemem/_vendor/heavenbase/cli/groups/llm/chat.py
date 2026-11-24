"""LLM chat CLI command handler."""

from __future__ import annotations

from heavenbase.utils import Any
from heavenbase.llm.base import format_messages

from .mcp import _load_mcp_tools
from .render import _chat_clip_text, _emit_chat_data, _emit_stream, _message_text, _result_meta
from .runtime import _DEFAULT_MAX_STEPS, _active_workspace, _copy_output, _emit_json, _llm, _llm_state, _max_steps, _max_tool_turns, _prompt, _verbose


def _session_entity(workspace: Any) -> type:
    """Return the Registry-owned Session class already active in a workspace."""
    try:
        return workspace.entities["agent-session"]
    except KeyError as error:
        raise KeyError("LLM session persistence requires the 'agent' extension to be enabled") from error


def _result_delta(data: dict[str, Any]) -> list[dict[str, Any]]:
    delta = data.get("delta")
    if isinstance(delta, list):
        return [dict(message) for message in delta if isinstance(message, dict)]
    message = data.get("message")
    if isinstance(message, dict):
        return [dict(message)]
    return [{"role": "assistant", "content": data.get("text") or ""}]


def _log_chat_messages(context: Any, messages: list[dict[str, Any]], title: str, source: str, usage: Any = None, state: Any = None) -> Any:
    workspace = _active_workspace(context)
    Session = _session_entity(workspace)
    session = Session(title=title[:255], summary=_chat_summary(messages), ws=workspace)
    workspace.upsert(Session, session.to_dict())
    session.append_messages(
        messages, source=source, usage=usage if isinstance(usage, dict) else None, state=state if isinstance(state, dict) else None, ws=workspace
    )
    return session


def _run_state(meta: dict[str, Any]) -> dict[str, Any]:
    """Return the run facts a stored session needs to be read honestly.

    A persisted `usage` of `{}` cannot say whether the run consumed nothing or was
    never measured, and a run cut off by its tool-round budget looks exactly like a
    completed one. Both are recorded here so the row states its own coverage.

    Args:
        meta (dict[str, Any]): Run facts from `_run_meta` or `_result_meta`.

    Returns:
        dict[str, Any]: State fields to merge into the persisted session state.
    """
    state: dict[str, Any] = {"usage_recorded": meta.get("usage") is not None, "truncated": bool(meta.get("truncated"))}
    if meta.get("elapsed") is not None:
        state["elapsed"] = meta["elapsed"]
    return state


def _chat_summary(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        value = _message_text(message).strip()
        if value:
            return value[:512]
    return ""


def _chat_base_messages(message: str, system: str | None = None, images: list[str] | None = None) -> list[dict[str, Any]]:
    return format_messages(message, system=system, images=images)


def _chat_result(llm: Any, messages: Any, system: str | None = None, stream: bool = True, **kwargs: Any) -> Any:
    include = ["text", "think", "message", "delta", "usage", "elapsed", "truncated"]
    return (
        llm.stream(messages, system=system, include=include, reduce=False, **kwargs)
        if stream
        else llm.chat(messages, system=system, include=include, reduce=False, **kwargs)
    )


def _chat(
    context: Any,
    prompt: str | None = None,
    system: str | None = None,
    input_files: list[str] | None = None,
    image_files: list[str] | None = None,
    stream: bool = True,
    preset: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    gateway: str | None = None,
    cache: bool = True,
    as_json: bool = False,
    copy: bool = False,
    verbose: bool = False,
    mcps: list[str] | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> None:
    llm = _llm(context=context, preset=preset or "chat", model=model, provider=provider, gateway=gateway, cache=cache)
    if verbose:
        _verbose(context, llm, cache=cache)
    message = _prompt(prompt, input_files)
    base_messages = _chat_base_messages(message, system=system, images=image_files)
    state = {
        "command": "hb llm chat",
        "llm": _llm_state(llm),
        "stream": bool(stream),
        "mcps": list(mcps or []),
        "max_steps": _max_steps(max_steps),
        "cache": bool(cache),
    }
    tools = _load_mcp_tools(context, mcps)
    if as_json:
        include = ["text", "think", "usage", "elapsed"] if not tools else ["text", "think", "message", "delta", "usage", "elapsed", "truncated"]
        data = llm.chat(
            message,
            system=system,
            images=image_files,
            include=include,
            reduce=False,
            stream=False,
            tools=tools or None,
            max_tool_turns=_max_tool_turns(max_steps),
        )
        output = _emit_json(context, data)
        result_data = data if isinstance(data, dict) else {"text": str(data)}
        meta = _result_meta(result_data)
        _log_chat_messages(
            context,
            [*base_messages, *_result_delta(result_data)],
            title=message or "LLM chat",
            source="hb llm chat",
            usage=meta["usage"],
            state=state | _run_state(meta),
        )
        if copy:
            _copy_output(context, output)
        return
    result = _chat_result(
        llm,
        message,
        system=system,
        stream=stream,
        images=image_files,
        tools=tools or None,
        max_tool_turns=_max_tool_turns(max_steps),
    )
    delta, meta = _emit_stream(context, result, max_steps=max_steps) if stream else _emit_chat_data(context, result, max_steps=max_steps)
    _log_chat_messages(
        context,
        [*base_messages, *delta],
        title=message or "LLM chat",
        source="hb llm chat",
        usage=meta["usage"],
        state=state | _run_state(meta),
    )
    if copy:
        _copy_output(context, _chat_clip_text(delta))
