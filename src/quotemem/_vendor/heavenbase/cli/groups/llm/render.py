"""LLM CLI output rendering helpers."""

from __future__ import annotations

from heavenbase.utils import Any, color_error, color_info, color_success, dumps_json, loads_json

# one merge rule for streamed and non-streamed usage; a second copy would drift
from heavenbase.llm.base import _merge_usage

from .runtime import _DEFAULT_MAX_STEPS, _max_steps

_TOOL_DISPLAY_LIMIT = 1200


def _run_meta(usages: list[dict[str, Any]], elapsed: Any = None, truncated: bool = False) -> dict[str, Any]:
    """Describe what one rendered run actually measured.

    `usage` is `None` when the provider reported no counters at all, which is not the
    same fact as a run that reported zero. Callers persist the difference instead of
    storing an empty aggregate that reads like a measured zero.

    Args:
        usages (list[dict[str, Any]]): Non-empty usage payloads seen during the run.
        elapsed (Any): Authoritative elapsed seconds, or `None` when unmeasured.
        truncated (bool): Whether the run's executable-tool loop ran out of rounds.

    Returns:
        dict[str, Any]: Run facts with keys `usage`, `elapsed`, and `truncated`.
    """
    return {"usage": _merge_usage(usages) if usages else None, "elapsed": elapsed, "truncated": bool(truncated)}


def _result_meta(data: dict[str, Any]) -> dict[str, Any]:
    """Read run facts from one complete non-streamed chat result.

    Args:
        data (dict[str, Any]): Include-projected chat result.

    Returns:
        dict[str, Any]: Run facts with keys `usage`, `elapsed`, and `truncated`.
    """
    usage = data.get("usage")
    return _run_meta([usage] if isinstance(usage, dict) and usage else [], data.get("elapsed"), bool(data.get("truncated")))


def _emit_think_block(context: Any, think: str) -> None:
    value = str(think or "").rstrip()
    if not value:
        return
    context.out.muted("<think>")
    context.out.muted(value)
    context.out.muted("</think>")


def _emit_text(context: Any, text: str = "", think: str = "") -> None:
    _emit_think_block(context, think)
    if text:
        context.out.echo(text)


def _assistant_from_delta(delta: list[dict[str, Any]]) -> dict[str, Any]:
    return next((message for message in reversed(delta) if message.get("role") == "assistant"), {})


def _clip_tool_text(text: Any, limit: int = _TOOL_DISPLAY_LIMIT) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return f"{value[:limit]}... <truncated {len(value) - limit} chars>"


def _format_tool_call(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    name = function.get("name") or tool_call.get("name") or "<unknown_tool>"
    arguments = function.get("arguments", tool_call.get("arguments", ""))
    try:
        parsed = loads_json(arguments) if isinstance(arguments, str) and arguments.strip() else arguments
    except Exception:
        parsed = arguments
    if isinstance(parsed, dict):
        args = ", ".join(f"{key}={dumps_json(value, indent=None)}" for key, value in parsed.items())
        return _clip_tool_text(f"{name}({args})")
    if isinstance(parsed, list):
        return _clip_tool_text(f"{name}({dumps_json(parsed, indent=None)})")
    arg_text = str(parsed or "").strip()
    return _clip_tool_text(f"{name}({arg_text})" if arg_text and arg_text != "{}" else f"{name}()")


def _message_text(message: dict[str, Any], key: str = "content") -> str:
    value = message.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _tool_content_is_error(content: Any) -> bool:
    try:
        data = loads_json(content) if isinstance(content, str) and content.strip() else content
    except Exception:
        return False
    return isinstance(data, dict) and (data.get("ok") is False or bool(data.get("error")))


def _delta_has_tools(delta: Any) -> bool:
    return isinstance(delta, list) and any(message.get("role") == "tool" or message.get("tool_calls") for message in delta if isinstance(message, dict))


def _emit_tool_calls(context: Any, message: dict[str, Any]) -> None:
    calls = message.get("tool_calls") or []
    for call in calls:
        context.out.echo(color_info(f"[Tool: {_format_tool_call(call)}]"))


def _emit_tool_result(context: Any, message: dict[str, Any], calls: dict[str, str]) -> None:
    call_id = str(message.get("tool_call_id") or "")
    signature = calls.get(call_id) or f"{message.get('name') or call_id or '<unknown_tool>'}()"
    content = _clip_tool_text(message.get("content") or "")
    if _tool_content_is_error(message.get("content")):
        context.out.echo(color_error(f"[Response: {signature}]"))
        context.out.echo(color_error(content))
    else:
        context.out.echo(color_success(f"[Response: {signature}]"))
        context.out.echo(color_success(content))


def _emit_step_header(context: Any, state: dict[str, Any]) -> None:
    state["step"] = int(state.get("step") or 0) + 1
    context.out.echo(f"STEPS: {state['step']:03d} / {_max_steps(state.get('max_steps')):03d}")


def _emit_delta_messages(context: Any, delta: list[dict[str, Any]], *, state: dict[str, Any] | None = None) -> None:
    state = state or {"step": 0, "max_steps": _DEFAULT_MAX_STEPS, "calls": {}}
    calls = state.setdefault("calls", {})
    for message in delta:
        if message.get("role") == "assistant":
            _emit_step_header(context, state)
            think = message.get("reasoning_content") or message.get("think") or ""
            _emit_think_block(context, think)
            content = _message_text(message)
            if content:
                context.out.echo(content.rstrip())
            if message.get("tool_calls"):
                for call in message.get("tool_calls") or []:
                    signature = _format_tool_call(call)
                    calls[str(call.get("id") or signature)] = signature
                _emit_tool_calls(context, message)
        elif message.get("role") == "tool":
            _emit_tool_result(context, message, calls)


def _emit_chat_data(context: Any, data: dict[str, Any], max_steps: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render one complete chat result and report what the run measured.

    Args:
        context (Any): CLI context owning the output channel.
        data (Any): Include-projected chat result.
        max_steps (int | None): Step budget shown in tool-round headers.

    Returns:
        tuple[list[dict[str, Any]], dict[str, Any]]: New messages, and the run facts
        from `_run_meta`.
    """
    delta = data.get("delta") or [data.get("message") or {"role": "assistant", "content": data.get("text") or ""}]
    if _delta_has_tools(delta):
        _emit_delta_messages(context, delta, state={"step": 0, "max_steps": _max_steps(max_steps), "calls": {}})
    else:
        _emit_text(context, text=data.get("text") or "", think=data.get("think") or "")
    return delta, _result_meta(data)


def _chat_clip_text(delta: list[dict[str, Any]]) -> str:
    return _message_text(_assistant_from_delta(delta)).rstrip()


def _emit_stream(context: Any, chunks: Any, max_steps: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render a streamed chat and report what the run measured.

    Usage arrives spread across chunks, so it is accumulated here rather than read from
    the closing chunk, which deliberately reports no usage to avoid double counting.
    Elapsed is taken from the last chunk that states it, because that chunk is the
    aggregate holding the whole run's duration rather than one round's.

    Args:
        context (Any): CLI context owning the output channel.
        chunks (Any): Iterable of include-projected stream chunks.
        max_steps (int | None): Step budget shown in tool-round headers.

    Returns:
        tuple[list[dict[str, Any]], dict[str, Any]]: New messages, and the run facts
        from `_run_meta`.
    """
    text = ""
    think = ""
    in_think = False
    message: dict[str, Any] = {}
    final_delta: list[dict[str, Any]] = []
    tool_delta = False
    rendered_delta: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    elapsed: Any = None
    truncated = False
    render_state: dict[str, Any] = {"step": 0, "max_steps": _max_steps(max_steps), "calls": {}}
    for delta in chunks:
        item = delta if isinstance(delta, dict) else {"text": str(delta)}
        think_chunk = item.get("think") or ""
        text_chunk = item.get("text") or ""
        msg = item.get("message") or {}
        msg_delta = item.get("delta") or []
        if isinstance(item.get("usage"), dict) and item["usage"]:
            usages.append(item["usage"])
        if item.get("elapsed") is not None:
            elapsed = item["elapsed"]
        truncated = truncated or bool(item.get("truncated"))
        if isinstance(msg_delta, list) and msg_delta:
            final_delta = [dict(part) for part in msg_delta]
            tool_delta = tool_delta or _delta_has_tools(final_delta)
            if _delta_has_tools(final_delta):
                if len(final_delta) >= len(rendered_delta) and final_delta[: len(rendered_delta)] == rendered_delta:
                    new_messages = final_delta[len(rendered_delta) :]
                    rendered_delta = final_delta
                else:
                    new_messages = final_delta
                    rendered_delta = [*rendered_delta, *new_messages]
                if new_messages:
                    _emit_delta_messages(context, new_messages, state=render_state)
        if isinstance(msg, dict) and msg:
            message = msg
        if tool_delta:
            continue
        if think_chunk:
            think += think_chunk
            if not in_think:
                context.out.muted("<think>", end="\n")
            in_think = True
            context.out.muted(think_chunk, end="")
        if text_chunk:
            text += text_chunk
            if in_think:
                context.out.muted("", end="\n")
                context.out.muted("</think>", end="\n")
                in_think = False
            context.out.echo(text_chunk, end="")
    if in_think:
        context.out.muted("", end="\n")
        context.out.muted("</think>", end="\n")
    final_delta = final_delta or ([message] if message else [{"role": "assistant", "content": text}])
    if not tool_delta:
        context.out.echo()
    return final_delta, _run_meta(usages, elapsed, truncated)
