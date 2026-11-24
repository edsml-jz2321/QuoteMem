"""Interactive LLM session CLI command handler."""

from __future__ import annotations

from heavenbase.utils import Any, color_grey, md5int
from heavenbase.llm.base import format_messages
from heavenbase.llm.session import LLMSession

from .chat import _chat_result, _run_state, _session_entity
from .mcp import _session_emit_tools, _session_load_mcp
from .render import _emit_chat_data, _emit_stream
from .runtime import _DEFAULT_MAX_STEPS, _active_workspace, _llm, _llm_state, _max_tool_turns, _prompt, _verbose

_SESSION_EXIT = ("/exit", "/quit", "/bye", "/e", "/q")
_SESSION_HELP = ("/help", "/h", "/?", "/commands")
_SESSION_SAVE = ("/save", "/s")
_SESSION_LOAD = ("/load", "/l")
_SESSION_CLEAR = ("/clear", "/c")
_SESSION_BACK = ("/back", "/b")
_SESSION_REGEN = ("/regen", "/r")
_SESSION_TOOLS = ("/tools",)
_SESSION_MCP = ("/mcp",)
_SESSION_LOAD_MCP = (*_SESSION_MCP, "/load-mcp")
_SESSION_COMMANDS = [
    *_SESSION_EXIT,
    *_SESSION_HELP,
    *_SESSION_SAVE,
    *_SESSION_LOAD,
    *_SESSION_CLEAR,
    *_SESSION_REGEN,
    *_SESSION_BACK,
    *_SESSION_TOOLS,
    *_SESSION_MCP,
]


def _is_persistent_session(value: Any) -> bool:
    workspace = getattr(value, "ws", None)
    return workspace is not None and type(value) is workspace.entities.get("agent-session")


def _session_help(context: Any) -> None:
    context.out.echo(
        "\n".join(
            [
                color_grey("Available commands:"),
                "  /exit, /quit, /bye, /e, /q       Exit the session",
                "  /help, /h, /?, /commands         Show this help message",
                "  /save [path], /s [path]          Save session JSON (default: session.json)",
                "  /load [path], /l [path]          Load session JSON",
                "  /clear, /c                       Clear session context",
                "  /regen [seed], /r [seed]          Regenerate the latest response",
                "  /back, /b                        Remove the latest user/assistant turn",
                "  /tools                            List active tools",
                "  /mcp SOURCE                       Add MCP tools from URL or Toolkit ref",
                "  Ctrl+C or Ctrl+D                 Exit the session",
            ]
        )
    )


def _sync_persistent_session(session: LLMSession, *, action: str) -> None:
    persistent = getattr(session, "persistent", None)
    if _is_persistent_session(persistent):
        persistent.set_messages(
            session.messages,
            source="hb llm session",
            state={"command": "hb llm session", "action": action, "llm": _llm_state(session.llm)},
        )


def _handle_session_command(
    context: Any,
    session: LLMSession,
    raw: str,
) -> tuple[bool, bool, bool, dict[str, Any]]:
    cmd = raw.strip()
    lower = cmd.lower()
    head = lower.split()[0] if lower else ""
    if lower in _SESSION_EXIT:
        return True, True, False, {}
    if lower in _SESSION_HELP:
        _session_help(context)
        return True, False, False, {}
    if head in _SESSION_SAVE:
        parts = cmd.split(maxsplit=1)
        path = parts[1] if len(parts) > 1 else "session.json"
        session.save(path)
        context.out.muted(f"saved {path}")
        return True, False, False, {}
    if head in _SESSION_LOAD:
        parts = cmd.split(maxsplit=1)
        path = parts[1] if len(parts) > 1 else "session.json"
        session.messages[:] = LLMSession.load(path, llm=session.llm).messages
        _sync_persistent_session(session, action="load")
        context.out.muted(f"loaded {path}")
        return True, False, False, {}
    if lower in _SESSION_CLEAR:
        session.clear()
        _sync_persistent_session(session, action="clear")
        context.out.muted("session context cleared")
        return True, False, False, {}
    if lower in _SESSION_TOOLS:
        _session_emit_tools(context, session)
        return True, False, False, {}
    if head in _SESSION_LOAD_MCP:
        parts = cmd.split(maxsplit=1)
        if len(parts) < 2:
            context.out.warning("usage: /mcp URL_OR_TOOLKIT_REF")
            return True, False, False, {}
        _session_load_mcp(context, session, parts[1])
        return True, False, False, {}
    if lower in _SESSION_BACK:
        backed = session.undo()
        if backed:
            _sync_persistent_session(session, action="back")
        context.out.muted("back one step" if backed else "cannot go back further")
        return True, False, False, {}
    if head in _SESSION_REGEN:
        if len(session.messages) < 2 or session.messages[-1].get("role") != "assistant":
            context.out.warning("nothing to regenerate")
            return True, False, False, {}
        last = session.messages.pop()
        _sync_persistent_session(session, action="regen")
        parts = cmd.split(maxsplit=1)
        seed = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else md5int(last.get("content", "")) % 1000000
        context.out.muted(f"regenerating with seed {seed}")
        return True, False, True, {"seed": seed}
    if cmd.startswith("/"):
        context.out.warning(f"unrecognized command: {cmd}; type /help for commands")
        return True, False, False, {}
    return False, False, False, {}


def _session_prompt() -> Any:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    bindings = KeyBindings()

    @bindings.add(Keys.ControlC)
    def _(event: Any) -> None:
        event.app.exit(result="/quit")

    @bindings.add(Keys.ControlD)
    def _(event: Any) -> None:
        event.app.exit(result="/quit")

    return PromptSession(
        history=InMemoryHistory(),
        key_bindings=bindings,
        completer=WordCompleter(_SESSION_COMMANDS, ignore_case=True),
        complete_while_typing=True,
        enable_history_search=True,
        mouse_support=True,
        multiline=False,
        wrap_lines=True,
    )


def _session_turn(context: Any, session: LLMSession, stream: bool = True, kwargs: dict[str, Any] | None = None) -> None:
    start_idx = int(getattr(session, "_log_start_idx", len(session.messages)))
    call_kwargs = dict(kwargs or {})
    call_kwargs.setdefault("tools", session.tools)
    call_kwargs.setdefault("max_tool_turns", _max_tool_turns(_DEFAULT_MAX_STEPS))
    result = _chat_result(session.llm, session.messages, stream=stream, **call_kwargs)
    if stream:
        delta, meta = _emit_stream(context, result)
    else:
        data = result if isinstance(result, dict) else {"text": str(result)}
        delta, meta = _emit_chat_data(context, data)
    session.messages.extend(delta)
    persistent = getattr(session, "persistent", None)
    if _is_persistent_session(persistent):
        persistent.append_messages(
            session.messages[start_idx:],
            source="hb llm session",
            usage=meta["usage"],
            state={
                "command": "hb llm session",
                "llm": _llm_state(session.llm),
                "stream": bool(stream),
                **_run_state(meta),
            },
        )
    if hasattr(session, "_log_start_idx"):
        delattr(session, "_log_start_idx")


def _session(
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
    verbose: bool = False,
    mcps: list[str] | None = None,
) -> None:
    llm = _llm(context=context, preset=preset or "chat", model=model, provider=provider, gateway=gateway, cache=cache)
    if verbose:
        _verbose(context, llm, cache=cache)
    session = LLMSession(llm)
    workspace = _active_workspace(context)
    Session = _session_entity(workspace)
    persistent = Session(
        title=(prompt or "LLM session")[:255],
        summary=_prompt(prompt, input_files)[:512] if prompt or input_files else "",
        state={"command": "hb llm session", "llm": _llm_state(llm), "mcps": list(mcps or []), "stream": bool(stream), "cache": bool(cache)},
        ws=workspace,
    )
    persistent.ws.upsert(Session, persistent.to_dict())
    session.persistent = persistent
    for source in mcps or []:
        _session_load_mcp(context, session, source)
    if system:
        session.messages.append({"role": "system", "content": system})
        persistent.append_messages([{"role": "system", "content": system}], source="hb llm session")
    console = _session_prompt()
    current = _prompt(prompt, input_files) or None
    pending_images = image_files
    regen = False
    gen_kwargs: dict[str, Any] = {}
    context.out.muted("Session started. Type /help for commands, /bye or /exit to quit.")
    while True:
        if current is None and not regen:
            current = console.prompt(">>> ").strip()
            handled, should_exit, regen, gen_kwargs = _handle_session_command(context, session, current)
            if should_exit:
                return
            if handled:
                current = None
                continue
        if not regen and current:
            session._log_start_idx = len(session.messages)
            session.messages.extend(format_messages(current, images=pending_images))
            pending_images = None
        elif regen or current:
            session._log_start_idx = len(session.messages)
        if regen or current:
            _session_turn(context, session, stream=stream, kwargs=gen_kwargs)
        current = None
        regen = False
        gen_kwargs = {}
