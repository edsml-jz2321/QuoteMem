"""User-facing LLM client with include-based response formatting."""

from __future__ import annotations

__all__ = ["LLM", "format_messages"]

import inspect
from time import perf_counter, time
from typing import TYPE_CHECKING

from .gateways._protocols import (
    CLIENT_ARG_KEYS,
    THINK_BLOCK_TYPES,
    THINK_KEYS,
    _apply_thinking,
    _claude_replay_format,
    _claude_thinking_requested,
    _join_reasoning_details,
    _join_thinking_blocks,
    _portkey_thinking_blocks,
    _without_thinking_content_blocks,
)
from .transport import (
    achat_complete,
    aembed_complete,
    aimagen_complete,
    chat_complete,
    embed_complete,
    imagen_complete,
    response_dict,
)
from .cache import LLMCache
from .config import LLMEngine, LLMSpec
from .gateway import LLMGateway
from .image import LLMImage
from .repair import LLMToolCallRepair
from .response import LLMResponse, _project
from .trace import LLMTraceEvent, TraceCallback, canonical_request, emit_event, forward_trace, trace_state
from heavenbase.utils import NetworkProxy, abatch, dumps_json, dumps_jstr, loads_json, pmap
from heavenbase.utils.config.data import to_mutable_data
from heavenbase.utils.typing import Any, AsyncGenerator, Callable, Dict, Generator, Iterable, List, Mapping, Optional, Union, deepcopy

if TYPE_CHECKING:
    from heavenbase.context import Context
    from heavenbase.modules import ModuleResolver
    from heavenbase.utils.config import ConfigManager

Message = Union[str, Dict[str, Any]]
Messages = Union[Message, Iterable[Message]]
ImageInputs = Optional[Union[Any, Iterable[Any]]]
EMBEDDING_CONTROL_KEYS = ("embedding_batch_size", "embedding_max_workers")
_DROP = object()
_TOOL_BUDGET_NOTE = (
    "[incomplete: the tool-call budget of {turns} round(s) ran out while the model was still calling tools; this answer is mid-investigation, not final]"
)
# Provider-reported usage owns context fullness when available. The fallback
# below deliberately overestimates structured data until the LLM layer gains a
# route-owned tokenizer; it is never exposed as an exact token count.
_CHARS_PER_TOKEN = 2.5
# Generic LLM calls preserve their historical behavior. Applications opt into
# capture and compaction explicitly per call.
_TOOL_RESULT_CAPTURE_CHARS = 0
_COMPACT_RATIO = 0.0
# Protects the smallest common provider window when a model registry entry
# declares no ``context_window`` of its own.
_CONTEXT_WINDOW_FALLBACK = 128000


def _cache_from(option: Any = None, config: Optional[Mapping[str, Any]] = None, *, manager: "ConfigManager | None" = None) -> tuple[LLMCache, Any]:
    if isinstance(option, LLMCache):
        if manager is not None and option.config_manager is not manager:
            raise ValueError("LLM cache and engine belong to different Contexts")
        return option, True
    if isinstance(option, Mapping):
        return LLMCache.from_dict(option, config_manager=manager), True
    if config is not None:
        return LLMCache.from_dict(config, config_manager=manager), option
    return LLMCache.current(config_manager=manager), option


def _repair_from(option: Any = None, config: Optional[Mapping[str, Any]] = None, *, manager: "ConfigManager | None" = None) -> tuple[LLMToolCallRepair, Any]:
    if isinstance(option, LLMToolCallRepair):
        if manager is not None and option.config_manager is not manager:
            raise ValueError("LLM tool-call repair and engine belong to different Contexts")
        return option, True
    if isinstance(option, Mapping):
        return LLMToolCallRepair.from_dict(option, config_manager=manager), True
    if isinstance(config, LLMToolCallRepair):
        if manager is not None and config.config_manager is not manager:
            raise ValueError("LLM tool-call repair and engine belong to different Contexts")
        return config, option
    if isinstance(config, Mapping):
        return LLMToolCallRepair.from_dict(config, config_manager=manager), option
    return LLMToolCallRepair.current(config_manager=manager), option


async def _maybe_await(value: Any) -> Any:
    return await value if hasattr(value, "__await__") else value


async def _aiter(items: Any) -> AsyncGenerator[Any, None]:
    if hasattr(items, "__aiter__"):
        async for item in items:
            yield item
        return
    for item in items:
        yield item


def _usage_count(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _with_total(usage: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(usage or {})
    if not _usage_count(out.get("total_tokens")) and _usage_count(out.get("prompt_tokens")) and _usage_count(out.get("completion_tokens")):
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def _merge_usage(usages: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge usage counters by summing numbers and keeping the first other value per key.

    ``total_tokens`` is the sum of the totals the parts report, plus prompt plus completion
    summed over the parts that report no total. So the chunks of one stream, which may report
    the prompt and completion counts in different chunks and no total, get the total of their
    summed counts, as the same response gets unstreamed; and the rounds of a tool loop, each
    already carrying its total, sum those totals. One usage alone gets the rule through
    ``_usage``. The total is derived only when that remainder holds both counts as numbers.
    """
    out: Dict[str, Any] = {}
    remainder: Dict[str, Any] = {}
    for usage in usages:
        part = dict(usage or {})
        for key, value in part.items():
            if _usage_count(value):
                # A provider may report a count as null; a later count replaces it.
                out[key] = out[key] + value if _usage_count(out.get(key)) else value
            elif key not in out:
                out[key] = value
        if not _usage_count(part.get("total_tokens")):
            for key in ("prompt_tokens", "completion_tokens"):
                if _usage_count(part.get(key)):
                    remainder[key] = remainder[key] + part[key] if key in remainder else part[key]
    derived = _with_total(remainder).get("total_tokens")
    if derived is not None:
        out["total_tokens"] = out["total_tokens"] + derived if _usage_count(out.get("total_tokens")) else derived
    return out


def _batches(items: List[Any], size: int) -> List[List[Any]]:
    return [items[index : index + size] for index in range(0, len(items), max(1, size))]


def _positive_int(value: Any, default: int = 1) -> int:
    return max(1, int(default if value is None else value))


def _nonnegative_int(value: Any, default: int = 0) -> int:
    return max(0, int(default if value is None else value))


def _pop_keys(cfg: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: cfg.pop(key) for key in keys if key in cfg}


def _drop_embedding_controls(cfg: Dict[str, Any]) -> None:
    _pop_keys(cfg, EMBEDDING_CONTROL_KEYS)


def _as_tools(tools: Any = None) -> List[Any]:
    if tools is None:
        return []
    if isinstance(tools, (list, tuple)):
        return list(tools)
    return [tools]


def _object_schema(schema: Any = None) -> Dict[str, Any]:
    schema = deepcopy(dict(schema or {})) if isinstance(schema, Mapping) else {}
    if schema.get("type") != "object" or "properties" not in schema:
        schema = {"type": "object", "properties": schema or {}, "required": []}
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    return schema


def _schema_tool_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(spec.get("function"), Mapping):
        function = deepcopy(dict(spec["function"]))
        function["parameters"] = _object_schema(function.get("parameters") or function.get("input_schema"))
        return {"type": spec.get("type") or "function", "function": function}
    if spec.get("name"):
        return {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec.get("description") or spec.get("desc") or "",
                "parameters": _object_schema(spec.get("parameters") or spec.get("input_schema") or spec.get("schema")),
            },
        }
    return deepcopy(dict(spec))


class _AsyncToolRequired(TypeError):
    """Raised when sync LLM tool execution receives an awaitable result."""


class _CallableToolRunner:
    """Minimal Tool-like runner for plain callables passed to LLM requests."""

    def __init__(self, func: Any) -> None:
        self.func = func
        self.name = str(getattr(func, "__name__", "") or "tool")
        self.description = str(getattr(func, "__doc__", "") or "").strip()
        # An explicit ``__tool_schema__`` wins over signature derivation, so
        # adapters bridging schema-first protocols (e.g. MCP inputSchema) can
        # pass the authoritative schema through a ``**kwargs`` callable.
        # ``functools.wraps`` copies the attribute, so wrapped tools keep it.
        schema = getattr(func, "__tool_schema__", None)
        self.input_schema = dict(schema) if isinstance(schema, Mapping) else _callable_input_schema(func)

    def to_openai_tool(self) -> Dict[str, Any]:
        """Return an OpenAI-compatible function tool schema."""
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.input_schema}}

    def run_to_str(self, **kwargs: Any) -> str:
        """Run the callable synchronously and serialize the result."""
        value = self.func(**kwargs)
        if inspect.isawaitable(value):
            raise _AsyncToolRequired("async tool execution needs arun_to_str")
        return _tool_result_content(value)

    async def arun_to_str(self, **kwargs: Any) -> str:
        """Run the callable asynchronously and serialize the result."""
        value = self.func(**kwargs)
        if inspect.isawaitable(value):
            value = await value
        return _tool_result_content(value)


def _callable_input_schema(func: Any) -> Dict[str, Any]:
    """Return a small JSON schema for a plain callable."""
    properties: Dict[str, Any] = {}
    required: List[str] = []
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {"type": "object", "properties": properties, "required": required}
    for name, param in parameters.items():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        properties[name] = _annotation_schema(param.annotation)
        if param.default is inspect._empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def _annotation_schema(annotation: Any) -> Dict[str, Any]:
    if annotation is inspect._empty:
        return {}
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    if annotation is str or name == "str":
        return {"type": "string"}
    if annotation is int or name == "int":
        return {"type": "integer"}
    if annotation is float or name == "float":
        return {"type": "number"}
    if annotation is bool or name == "bool":
        return {"type": "boolean"}
    if annotation in {list, tuple, set} or name in {"list", "tuple", "set"}:
        return {"type": "array"}
    if annotation is dict or name == "dict":
        return {"type": "object"}
    return {}


def _ensure_tool_runner(item: Any) -> Any:
    """Return a Tool-like runner whose serializer owns LLM tool-result text."""
    if callable(getattr(item, "run_to_str", None)):
        return item
    if callable(item):
        return _CallableToolRunner(item)
    if callable(getattr(item, "run", None)):
        run = item.run

        def _proxy(**kwargs: Any) -> Any:
            return run(**kwargs)

        _proxy.__name__ = str(getattr(item, "name", getattr(run, "__name__", "tool")))
        _proxy.__doc__ = str(getattr(item, "description", getattr(run, "__doc__", "")) or "")
        return _CallableToolRunner(_proxy)
    raise TypeError(f"Executable tool {type(item).__name__} must expose run_to_str or a callable run method")


def _tools_runtime(tools: Any = None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    runners: Dict[str, Any] = {}

    def add_spec(spec: Mapping[str, Any], runner: Any = None) -> None:
        item = _schema_tool_spec(spec)
        name = _tool_call_name(item)
        if runner is not None and not name:
            raise ValueError("Executable tool schema function.name is required")
        if name and any(_tool_call_name(existing) == name for existing in specs):
            raise ValueError(f"Duplicate tool name {name!r} across active tools")
        specs.append(item)
        if runner is not None and name:
            runners[name] = runner

    for item in _as_tools(tools):
        if isinstance(item, Mapping):
            add_spec(item)
        elif hasattr(item, "tools"):
            for name, tool in item.tools.items():
                add_spec(tool.to_openai_tool(), runner=tool)
        elif callable(getattr(item, "to_openai_tool", None)):
            add_spec(item.to_openai_tool(), runner=_ensure_tool_runner(item) if callable(getattr(item, "run", None)) else None)
        elif callable(item):
            runner = _CallableToolRunner(item)
            add_spec(runner.to_openai_tool(), runner=runner)
        else:
            raise TypeError(f"Unsupported tool type: {type(item).__name__}")
    return specs, runners


def _tool_call_name(tool_call: Mapping[str, Any]) -> str:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    return str(function.get("name") or tool_call.get("name") or "")


def _tool_call_arguments(tool_call: Mapping[str, Any]) -> Dict[str, Any]:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    arguments = function.get("arguments", tool_call.get("arguments", {}))
    if isinstance(arguments, str):
        arguments = loads_json(arguments) if arguments.strip() else {}
    if arguments is None:
        return {}
    if not isinstance(arguments, dict):
        raise TypeError("Tool arguments must be a JSON object")
    return dict(arguments)


def _tool_result_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return dumps_jstr(value, compact=True)
    except Exception:
        return str(value)


def _tool_error_content(kind: str, message: str) -> str:
    return _tool_result_content({"ok": False, "error": {"type": kind, "message": message}})


def _clip_tool_result(content: str, cap_chars: int) -> str:
    """Clip one captured tool result at an explicit character boundary.

    The clipped message is what the model rereads AND what a transcript
    presents in both directions. The
    recorded ``delta`` carries the same clipped content, and richer display
    channels can derive their own truncation from the same clip rather than
    rendering payload the model never saw.

    Args:
        content (str): Serialized tool result bound for the tool message.
        cap_chars (int): Character budget; ``0`` disables clipping.

    Returns:
        str: The content, clipped with an actionable truncation note when over.
    """
    if not cap_chars or len(content) <= cap_chars:
        return content
    kept = content[:cap_chars]
    return f"{kept}\n… [tool result truncated: kept {len(kept)} of {len(content)} characters; " "make a narrower call for the rest]"


def _prompt_tokens(usage: Any) -> Optional[int]:
    """Return the provider-reported prompt/input token count, if any."""
    if not hasattr(usage, "get"):
        return None
    for key in ("prompt_tokens", "input_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _conversation_chars(messages: Iterable[Mapping[str, Any]]) -> int:
    """Measure one assembled conversation in characters: contents plus tool-call arguments."""
    total = 0
    for message in messages:
        content = message.get("content")
        if content is not None:
            total += len(content) if isinstance(content, str) else len(str(content))
        for call in message.get("tool_calls") or ():
            function = call.get("function") if isinstance(call, Mapping) else None
            if isinstance(function, Mapping):
                total += len(str(function.get("arguments") or ""))
    return total


def _is_async_tool_required(exc: Exception) -> bool:
    return isinstance(exc, _AsyncToolRequired) or type(exc).__name__ == "AsyncToolRequired"


def _embedding_controls(cfg: Dict[str, Any], n_items: int) -> tuple[int, int]:
    values = _pop_keys(cfg, EMBEDDING_CONTROL_KEYS)
    n_batch = values.get("embedding_batch_size")
    n_workers = values.get("embedding_max_workers")
    return _positive_int(n_batch if n_batch is not None else (n_items or 1)), _positive_int(n_workers)


def format_messages(
    messages: Messages,
    system: Optional[str] = None,
    images: ImageInputs = None,
    *,
    config_manager: "ConfigManager | None" = None,
) -> List[Dict[str, Any]]:
    """
    Normalize strings, dictionaries, and message lists to OpenAI chat messages.

    Use this helper when caller-facing APIs accept either a single prompt string, one message
    dictionary, or an iterable of messages. Tool calls are normalized to the OpenAI ``tool_calls``
    shape and dictionary/list function arguments are serialized to JSON strings.

    Args:
        messages (Messages): Prompt string, message dictionary, or iterable of message items.
        system (Optional[str]): Optional system message prepended before user messages. Defaults to ``None``.
        images (ImageInputs): Optional image input or image inputs attached to the last user message. Defaults to ``None``.
        config_manager (ConfigManager | None): Optional configuration authority used to bind
            lazy image URL timeouts. Truly standalone calls default only when
            URL bytes are later requested.

    Returns:
        List[Dict[str, Any]]: OpenAI-compatible chat messages.
    """
    items = [messages] if isinstance(messages, (str, dict)) or _message_like(messages) or _image_source(messages) else list(messages)
    out = [{"role": "system", "content": system}] if system else []
    for item in items:
        message = _format_message_item(item, config_manager=config_manager)
        if message.get("tool_calls"):
            message = deepcopy(message)
            message["tool_calls"] = [_normalize_tool_call(tool_call) for tool_call in message["tool_calls"]]
        out.append(message)
    return _attach_images(out, images, config_manager=config_manager)


def _message_like(value: Any) -> bool:
    return any(callable(getattr(value, name, None)) for name in ("model_dump", "to_dict", "dict", "json")) or (
        hasattr(value, "role") and hasattr(value, "content")
    )


def _format_message_item(item: Any, *, config_manager: Any = None) -> Dict[str, Any]:
    if isinstance(item, str):
        return {"role": "user", "content": item}
    if isinstance(item, dict):
        message = deepcopy(item)
    elif callable(getattr(item, "model_dump", None)):
        message = item.model_dump()
    elif callable(getattr(item, "to_dict", None)):
        message = item.to_dict()
    elif callable(getattr(item, "dict", None)):
        message = item.dict()
    elif callable(getattr(item, "json", None)):
        message = loads_json(item.json())
    elif hasattr(item, "role") and hasattr(item, "content"):
        message = {"role": getattr(item, "role"), "content": getattr(item, "content")}
    else:
        try:
            message = dict(item)
        except Exception as exc:
            raise TypeError(f"Unsupported message type: {type(item).__name__}") from exc
    if not isinstance(message, dict):
        raise TypeError(f"Message object must resolve to dict, got {type(message).__name__}")
    if "role" not in message:
        raise ValueError("Message dict must contain 'role' field")
    if "content" in message:
        message["content"] = _format_content(message["content"], config_manager=config_manager)
    return deepcopy(message)


def _image_source(value: Any, strings: bool = False) -> bool:
    if isinstance(value, LLMImage):
        return True
    if isinstance(value, dict):
        return any(key in value for key in ("image", "data", "b64_json", "url", "image_url"))
    if isinstance(value, (bytes, bytearray, memoryview)) or hasattr(value, "read"):
        return True
    if strings and isinstance(value, str):
        return True
    if hasattr(value, "save") and hasattr(value, "mode") and hasattr(value, "size"):
        return True
    return hasattr(value, "shape") and (hasattr(value, "__array__") or hasattr(value, "__array_interface__"))


def _image_inputs(images: ImageInputs, *, config_manager: Any = None) -> List[LLMImage]:
    if images is None:
        return []
    if _image_source(images, strings=True):
        values = [LLMImage.from_any(images)]
    else:
        values = [LLMImage.from_any(item) for item in images]
    return [image._bound(config_manager) for image in values] if config_manager is not None else values


def _format_content(content: Any, *, config_manager: Any = None) -> Any:
    if isinstance(content, list):
        return [_format_content_part(item, config_manager=config_manager) for item in content]
    if _image_source(content):
        return [_format_content_part(content, config_manager=config_manager)]
    return content


def _format_content_part(item: Any, *, config_manager: Any = None) -> Dict[str, Any]:
    if isinstance(item, dict):
        part = deepcopy(item)
        kind = part.get("type")
        if kind == "image_url":
            if isinstance(part.get("image_url"), str):
                part["image_url"] = {"url": part["image_url"]}
            return part
        if kind == "text":
            return part
        if kind in {"image", "input_image"} or any(key in part for key in ("image", "data", "b64_json", "url")):
            image = LLMImage.from_any(part)
            return (image._bound(config_manager) if config_manager is not None else image).to_dict()
        return part
    if _image_source(item):
        image = LLMImage.from_any(item)
        return (image._bound(config_manager) if config_manager is not None else image).to_dict()
    return {"type": "text", "text": str(item)}


def _content_parts(content: Any, *, config_manager: Any = None) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        return [_format_content_part(item, config_manager=config_manager) for item in content]
    if content is None:
        return []
    if _image_source(content):
        return [_format_content_part(content, config_manager=config_manager)]
    return [{"type": "text", "text": str(content)}]


def _attach_images(messages: List[Dict[str, Any]], images: ImageInputs = None, *, config_manager: Any = None) -> List[Dict[str, Any]]:
    refs = _image_inputs(images, config_manager=config_manager)
    if not refs:
        return messages
    out = deepcopy(messages)
    parts = [image.to_dict() for image in refs]
    for index in range(len(out) - 1, -1, -1):
        if out[index].get("role") == "user":
            out[index]["content"] = [*_content_parts(out[index].get("content"), config_manager=config_manager), *parts]
            return out
    out.append({"role": "user", "content": parts})
    return out


def _choice(resp: Dict[str, Any]) -> Dict[str, Any]:
    choices = resp.get("choices") or []
    return choices[0] if choices else {}


def _message(resp: Dict[str, Any]) -> Dict[str, Any]:
    return _choice(resp).get("message") or _choice(resp).get("delta") or {}


def _usage(resp: Dict[str, Any]) -> Dict[str, Any]:
    # one rule for a non-streamed response and for gathered stream chunks
    return _merge_usage([resp.get("usage") or {}])


def _images(resp: Dict[str, Any], *, config_manager: Any = None) -> List[LLMImage]:
    images = [LLMImage.from_provider_item(item) for item in resp.get("data", [])]
    return [image._bound(config_manager) for image in images] if config_manager is not None else images


def _shows_output(response: LLMResponse) -> bool:
    """Return whether a stream chunk has anything a stream caller reads."""
    return bool(response.text or response.think or response.tool_calls or response.usage)


def _shows_stream_output(response: LLMResponse, include: Optional[Union[str, Iterable[str]]]) -> bool:
    # The Messages API can report starting counts before there is any output.
    # A text-only caller should still receive the model's first text first.
    if isinstance(response.raw, dict) and response.raw.get("type") == "message_start":
        fields = (include,) if isinstance(include, str) else include or ()
        if not {"usage", "raw"}.intersection(fields):
            return False
    return _shows_output(response)


def _sent_sampling(top: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``temperature``/``top_k`` a request built from ``top`` sends: ``extra_body`` wins over top-level, as the SDK merges it."""
    extra = top.get("extra_body") if isinstance(top.get("extra_body"), Mapping) else {}
    found: Dict[str, Any] = {}
    for key in ("temperature", "top_k"):
        value = extra.get(key) if extra.get(key) is not None else top.get(key)
        if value is not None:
            found[key] = value
    return found


def _caller_sampling(spec: LLMSpec, kwargs: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``temperature``/``top_k`` the caller set that the request would actually send.

    The caller's options merge as the request does: per-call values over those given to ``LLM(...)``
    (as keywords or in ``args``), a per-call ``extra_body`` replacing the constructor's, and
    ``extra_body`` winning over top-level fields. A caller value counts only when it is the value
    sent (``cfg``). Values from configuration (the packaged or user config, a model entry, a preset)
    are not the caller's, and neither are ``default_args`` given to ``LLM(...)``: a component that
    builds an LLM from its own configured defaults passes them there.
    """
    raw = spec.raw if isinstance(spec.raw, Mapping) else {}
    caller: Dict[str, Any] = {}
    for source in (raw.get("args"), raw, kwargs):
        if isinstance(source, Mapping):
            caller.update({key: value for key, value in source.items() if value is not None})
    sent = _sent_sampling(cfg)
    return {key: value for key, value in _sent_sampling(caller).items() if key in sent and sent[key] == value}


def _refuse_caller_sampling_with_claude_thinking(spec: LLMSpec, kwargs: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    """Refuse a caller's sampling setting that Claude does not accept with thinking on.

    With thinking on, Claude accepts no ``top_k`` and no ``temperature`` other than its default of 1.
    The request builders leave out a configured default (such as the packaged ``temperature=0``);
    a value the caller passed is refused here instead of being dropped without notice.
    """
    if not _claude_thinking_requested(spec.model_id, cfg):
        return
    refused = {key: value for key, value in _caller_sampling(spec, kwargs, cfg).items() if not (key == "temperature" and value == 1)}
    if refused:
        names = " and ".join(f"{key}={value!r}" for key, value in sorted(refused.items()))
        raise ValueError(
            f"Claude does not accept {names} with thinking on (temperature only at its default of 1, and no top_k). "
            "Leave it out, or turn thinking off with think=False."
        )


def _think_suppressed(cfg: Mapping[str, Any]) -> bool:
    return cfg.get("think") is False


def _think_block(value: Mapping[str, Any]) -> bool:
    return str(value.get("type") or value.get("kind") or "").lower() in THINK_BLOCK_TYPES


def _think_key(key: Any) -> bool:
    text = str(key).lower()
    return text in THINK_KEYS or text.startswith("reasoning_") or text.startswith("thinking_")


def _strip_think_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        if _think_block(value):
            return _DROP
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if _think_key(key):
                continue
            clean = _strip_think_data(item)
            if clean is not _DROP:
                out[key] = clean
        return out
    if isinstance(value, list):
        return [clean for item in value if (clean := _strip_think_data(item)) is not _DROP]
    return value


def _strip_response_think(response: LLMResponse) -> LLMResponse:
    data = response.to_dict()
    data["think"] = ""
    data["content"] = data.get("text") or ""
    message = _strip_think_data(data.get("message") or {}) or {}
    if message and message.get("content") is None:
        message["content"] = data["text"]
    data["message"] = message
    data["delta"] = _strip_think_data(data.get("delta") or []) or []
    data["messages"] = _strip_think_data(data.get("messages") or []) or []
    data["raw"] = _strip_think_data(data.get("raw"))
    return LLMResponse.from_dict(data)


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_content_text(item.get("text") or item.get("content") if isinstance(item, dict) else item) for item in value)
    return str(value)


def _normalize_tool_call(tool_call: Dict[str, Any], keep_index: bool = False) -> Dict[str, Any]:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    arguments = function.get("arguments", tool_call.get("arguments", ""))
    if isinstance(arguments, (dict, list)):
        arguments = dumps_json(arguments, indent=None)
    out = {
        "id": tool_call.get("id") or "",
        "type": tool_call.get("type") or "function",
        "function": {"name": function.get("name") or tool_call.get("name") or "", "arguments": arguments or ""},
    }
    if keep_index and "index" in tool_call:
        out["index"] = tool_call["index"]
    return out


def _merge_tool_calls(current: List[Dict[str, Any]], deltas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for position, delta in enumerate(deltas):
        index = delta.get("index")
        if index is None:
            match = next((idx for idx, item in enumerate(current) if delta.get("id") and item.get("id") == delta.get("id")), None)
            index = match if match is not None else position if position < len(current) else len(current)
        while index >= len(current):
            current.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        if delta.get("id"):
            current[index]["id"] = delta["id"]
        if delta.get("type"):
            current[index]["type"] = delta["type"]
        function = delta.get("function") or {}
        if function.get("name"):
            current[index]["function"]["name"] = function["name"]
        if function.get("arguments"):
            current[index]["function"]["arguments"] = (current[index]["function"].get("arguments") or "") + function["arguments"]
    return current


def _structured(text: str, message: Dict[str, Any]) -> Any:
    if message.get("parsed") is not None:
        return message["parsed"]
    if not text:
        return None
    try:
        return loads_json(text)
    except Exception:
        return None


def _assistant_message(text: str, tool_calls: List[Dict[str, Any]], message: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = deepcopy(message or {})
    out["role"] = out.get("role") or "assistant"
    if out.get("content") is None:
        out["content"] = text
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _message_delta(message: Dict[str, Any], tool_messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    return ([deepcopy(message)] if message else []) + [deepcopy(item) for item in tool_messages or []]


def _needs_final_delta(include: Optional[Union[str, Iterable[str]]]) -> bool:
    if include is None:
        return False
    fields = [include] if isinstance(include, str) else list(include)
    # Only the closing aggregate knows the whole delta and whether the tool loop
    # finished, so asking for any of these obliges the stream to emit it.
    return bool({"delta", "messages", "tool_turns", "max_tool_turns", "truncated"} & set(fields))


def _includes_text(include: Optional[Union[str, Iterable[str]]]) -> bool:
    """Return whether an include projection exposes streamed answer text."""
    if include is None:
        return True
    fields = [include] if isinstance(include, str) else list(include)
    return "text" in fields


def _stream_final_response(response: LLMResponse, with_usage: bool = True) -> LLMResponse:
    return LLMResponse(
        tool_calls=response.tool_calls,
        message=response.message,
        usage=response.usage if with_usage else {},
        raw=response.raw,
        elapsed=response.elapsed,
        created_at=response.created_at,
        messages=response.messages,
        delta=response.delta,
        structured=response.structured,
        tool_turns=response.tool_turns,
        max_tool_turns=response.max_tool_turns,
        truncated=response.truncated,
    )


def _append_note(content: Any, note: str) -> str:
    text = _content_text(content).rstrip()
    return f"{text}\n\n{note}" if text else note


def _mark_tool_budget(response: LLMResponse) -> LLMResponse:
    """Write the tool-budget cutoff into the answer a default caller actually reads.

    The ``truncated`` field alone cannot close this hole: ``include`` defaults to
    ``text``, and ``LLMSession.send`` returns the assistant message, so an exhausted
    loop would still hand back a mid-investigation value that reads exactly like a
    finished answer. The note therefore travels with the answer itself, in both the
    aggregated ``text`` and the final assistant message that ``delta`` and ``messages``
    carry. The raw provider payload is left untouched as the faithful record.

    Args:
        response (LLMResponse): Aggregate whose executable-tool loop ran out of rounds.

    Returns:
        LLMResponse: The same response, with its answer text and final assistant
        message annotated as incomplete.
    """
    note = _TOOL_BUDGET_NOTE.format(turns=response.max_tool_turns)
    response.text = _append_note(response.text, note)
    last = next((message for message in reversed(response.delta) if message.get("role") == "assistant"), None)
    # `delta` holds a copy of the final assistant message, so both need the note
    for message in {id(item): item for item in (response.message, last) if item}.values():
        message["content"] = _append_note(message.get("content"), note)
    return response


class LLM:
    """
    Small LLM utility over OpenAI-compatible, Anthropic, gateway, or mock transports.

    Use this class for model resolution, gateway materialization, chat, streaming, embeddings, and
    image generation from the shared ``heavenbase.llm`` config. The public methods all support the
    ``include`` projection pattern so callers can request the exact response fields
    they want without losing access to full provider payloads.
    """

    def __init__(
        self,
        preset: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        gateway: Optional[str] = None,
        endpoint: Optional[str] = None,
        *,
        context: "Context | None" = None,
        resolver: "ModuleResolver | None" = None,
        config: "ConfigManager | None" = None,
        engine: Optional[LLMEngine] = None,
        **kwargs: Any,
    ) -> None:
        """
        Resolve an LLM runtime from config plus explicit overrides.

        Use this initializer with a preset, canonical model key, provider, gateway, or arbitrary
        provider arguments. Values are resolved through ``LLMEngine`` and then applied at
        request time. Endpoint grammar and gateway transport remain explicit;
        unavailable routes fail without rewriting normalized state.

        Args:
            preset (Optional[str]): Named preset under ``heavenbase.llm.presets``. Defaults to ``None``.
            model (Optional[str]): Canonical model key, alias, or provider identifier. Defaults to ``None``.
            provider (Optional[str]): Provider config key such as ``openrouter`` or ``deepseek``. Defaults to ``None``.
            gateway (Optional[str]): Gateway key such as ``openai``, ``anthropic``, ``portkey``, ``litellm``, ``bifrost``, or ``mock``. Defaults to ``None``. Use ``gateway="anthropic"`` for the official Anthropic SDK and Messages payloads; use ``gateway="openai"`` for Anthropic's OpenAI-compatible endpoint.
            endpoint (Optional[str]): Request/response grammar such as
                ``openai`` or ``anthropic``. Defaults from the selected route.
            context (Context | None): Context owning both module resolution and
                configuration. It is mutually exclusive with ``resolver``,
                ``config``, and ``engine``.
            resolver (ModuleResolver | None): Explicit Context-owned resolver. Omission
                selects ``DEFAULT_CONTEXT.modules()``.
            config (ConfigManager | None): Explicit manager paired with ``resolver``.
                Omission selects ``DEFAULT_CONTEXT.config``.
            engine (LLMEngine | None): Preconstructed authority-bound engine.
                It is mutually exclusive with ``context``, ``resolver``, and
                ``config``.
            **kwargs (Any): Extra request defaults stored on the resolved spec.

        Returns:
            None: This function returns ``None`` after initializing the object.
        """
        if engine is not None and any(value is not None for value in (context, resolver, config)):
            raise TypeError("LLM engine cannot be combined with context, resolver, or config authority")
        if engine is None:
            engine_options = {"context": context, "resolver": resolver}
            if config is not None:
                engine_options["manager"] = config
            engine = LLMEngine(**engine_options)
        self.engine = engine
        cache_option = kwargs.pop("cache", None)
        cache_config = kwargs.pop("cache_config", None)
        repair_config = kwargs.pop("tool_call_repair", None)
        self._cache, self._cache_option = _cache_from(cache_option, config=cache_config, manager=self.engine.manager)
        self._repair, self._repair_option = _repair_from(config=repair_config, manager=self.engine.manager)
        if repair_config is not None and "repair_tool_calls" not in kwargs and self._repair.enabled():
            kwargs["repair_tool_calls"] = True
        options = {
            k: v
            for k, v in {"preset": preset, "model": model, "provider": provider, "gateway": gateway, "endpoint": endpoint, **kwargs}.items()
            if v is not None
        }
        runtime_options = {k: v for k, v in options.items() if callable(v)}
        config_options = {k: v for k, v in options.items() if k not in runtime_options}
        self.spec: LLMSpec = self.engine.resolve(config_options)
        if runtime_options:
            self.spec = self.spec.with_args(runtime_options)
        self._dim = self.spec.dim

    def _request_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in kwargs.items() if value is not None or key in {"seed"}}

    def _cache_pair(self, option: Any = None) -> tuple[LLMCache, Any]:
        return (self._cache, self._cache_option) if option is None else _cache_from(option, manager=self.engine.manager)

    def _repair_pair(self, option: Any = None, config: Any = None) -> tuple[LLMToolCallRepair, Any]:
        if config is None and not isinstance(option, (LLMToolCallRepair, Mapping)):
            return self._repair, self._repair_option if option is None else option
        return _repair_from(option, config=config, manager=self.engine.manager)

    @property
    def dim(self) -> int:
        """Return embedding dimension from config, or infer it with one test embedding call."""
        if self._dim is not None:
            return self._dim
        if self.spec.dim is not None:
            self._dim = self.spec.dim
            return self._dim
        vector = self.embed("<TEST>")
        if vector and isinstance(vector, list):
            self._dim = len(vector)
            return self._dim
        raise ValueError(f"Unexpected embedding format. This LLM may not support embeddings: {vector}")

    def _repair_tool_calls(
        self, tool_calls: List[Dict[str, Any]], repair: Optional[LLMToolCallRepair] = None, repair_option: Any = None, tools: Any = None
    ) -> List[Dict[str, Any]]:
        return (repair or self._repair).repair(tool_calls, tools=tools, option=repair_option)

    def _chat_response(
        self,
        raw: Any,
        elapsed: float,
        messages: List[Dict[str, Any]],
        delta: Optional[Dict[str, Any]] = None,
        repair: Optional[LLMToolCallRepair] = None,
        repair_option: Any = None,
        tools: Any = None,
        suppress_think: bool = False,
    ) -> LLMResponse:
        resp = response_dict(raw)
        if suppress_think:
            resp = _strip_think_data(resp)
        msg = _message(resp)
        portkey_think = ""
        if not suppress_think and "content_blocks" in msg and _claude_replay_format(self.spec) == "content_blocks":
            blocks = _portkey_thinking_blocks(msg["content_blocks"], stream=delta is not None)
            if blocks:
                msg = {**msg, "thinking_blocks": blocks}
                portkey_think = "".join(block.get("thinking") or "" for block in blocks)
        text = _content_text(msg.get("content") or msg.get("text"))
        think = "" if suppress_think else _content_text(msg.get("reasoning_content") or msg.get("reasoning") or msg.get("thinking")) or portkey_think
        tool_calls = [_normalize_tool_call(tool_call, keep_index=delta is not None) for tool_call in msg.get("tool_calls") or []]
        tool_calls = self._repair_tool_calls(tool_calls, repair=repair, repair_option=repair_option, tools=tools) if tool_calls else tool_calls
        msg = _assistant_message(text, tool_calls, msg)
        message_delta = [] if delta is not None else _message_delta(msg)
        return LLMResponse(
            text=text,
            think=think,
            tool_calls=tool_calls,
            message=msg,
            # A stream chunk keeps the counts it reports: its response's total comes once, from all
            # its chunks' counts merged (_gather_responses), as in 0.1.2.2.
            usage=_usage(resp) if delta is None else dict(resp.get("usage") or {}),
            raw=resp,
            elapsed=elapsed,
            messages=[*messages, *message_delta],
            delta=message_delta,
            structured=_structured(text, msg),
        )

    def _call_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        cfg = to_mutable_data(self.spec.args)
        cfg.update(self._request_kwargs(kwargs))
        return cfg

    def _streamable(self, cfg: Dict[str, Any]) -> bool:
        stream_chat = bool(cfg.pop("stream_chat", True))
        structured_stream = bool(cfg.pop("structured_stream", True))
        enforce_non_stream_structured = bool(cfg.pop("enforce_non_stream_structured", False))
        if enforce_non_stream_structured and cfg.get("response_format"):
            return False
        return stream_chat and (structured_stream or not cfg.get("response_format"))

    def _chat_context(self, messages: Messages, system: Optional[str], images: ImageInputs, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        prepared = format_messages(messages, system=system, images=images, config_manager=self.engine.manager)
        cfg = self._call_kwargs(kwargs)
        _refuse_caller_sampling_with_claude_thinking(self.spec, kwargs, cfg)
        _drop_embedding_controls(cfg)
        tool_specs, tool_runners = _tools_runtime(cfg.pop("tools", None))
        if tool_specs:
            cfg["tools"] = tool_specs
        cache_option = cfg.pop("cache", None)
        cache_policy = cfg.pop("cache_policy", None)
        max_tool_turns = _nonnegative_int(cfg.pop("max_tool_turns", None), default=8)
        tool_result_capture_chars = _nonnegative_int(cfg.pop("tool_result_capture_chars", None), default=_TOOL_RESULT_CAPTURE_CHARS)
        compact_ratio = cfg.pop("compact_ratio", None)
        compact_ratio = _COMPACT_RATIO if compact_ratio is None else float(compact_ratio)
        if compact_ratio < 0.0 or compact_ratio > 1.0:
            raise ValueError("compact_ratio must be between 0 and 1")
        streamable = self._streamable(cfg)
        suppress_think = _think_suppressed(cfg)
        cache, enabled_option = self._cache_pair(cache_option)
        policy = cache.policy("text", cfg, cache_policy)
        active = cache.enabled("text", enabled_option) and policy is not None and not tool_runners
        key = cache.text_key(self.spec, prepared, cfg)
        repair_option = cfg.pop("repair_tool_calls", None)
        repair_config = cfg.pop("tool_call_repair", None)
        repair, repair_option = self._repair_pair(repair_option, config=repair_config)
        return {
            "prepared": prepared,
            "cfg": cfg,
            "streamable": streamable,
            "suppress_think": suppress_think,
            "cache": cache,
            "active": active,
            "key": key,
            "policy": policy,
            "repair": repair,
            "repair_option": repair_option,
            "tools": deepcopy(cfg.get("tools") or []),
            "tool_runners": tool_runners,
            "max_tool_turns": max_tool_turns,
            "tool_result_capture_chars": tool_result_capture_chars,
            "compact_ratio": compact_ratio,
        }

    def _chat_step(self, prepared: List[Dict[str, Any]], ctx: Dict[str, Any], chunks: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        return (
            self._gather_chat(
                prepared,
                deepcopy(ctx["cfg"]),
                chunks=chunks,
                repair=ctx["repair"],
                repair_option=ctx["repair_option"],
                tools=ctx["tools"],
            )
            if ctx["streamable"]
            else self._chat_once(prepared, deepcopy(ctx["cfg"]), repair=ctx["repair"], repair_option=ctx["repair_option"], tools=ctx["tools"])
        )

    async def _achat_step(self, prepared: List[Dict[str, Any]], ctx: Dict[str, Any], chunks: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        return (
            await self._agather_chat(
                prepared,
                deepcopy(ctx["cfg"]),
                chunks=chunks,
                repair=ctx["repair"],
                repair_option=ctx["repair_option"],
                tools=ctx["tools"],
            )
            if ctx["streamable"]
            else await self._achat_once(prepared, deepcopy(ctx["cfg"]), repair=ctx["repair"], repair_option=ctx["repair_option"], tools=ctx["tools"])
        )

    def _run_tool_calls(self, tool_calls: List[Dict[str, Any]], runners: Dict[str, Any], capture_chars: int = 0) -> List[Dict[str, Any]]:
        messages = []
        for tool_call in tool_calls:
            name = _tool_call_name(tool_call)
            tool = runners.get(name)
            if tool is None:
                content = _tool_error_content("UnknownTool", f"Unknown tool: {name}")
            else:
                try:
                    content = tool.run_to_str(**_tool_call_arguments(tool_call))
                except Exception as exc:
                    if _is_async_tool_required(exc):
                        raise TypeError("async tool execution needs achat") from exc
                    content = _tool_error_content(type(exc).__name__, str(exc))
            messages.append({"role": "tool", "tool_call_id": tool_call.get("id") or name, "name": name, "content": _clip_tool_result(content, capture_chars)})
        return messages

    async def _arun_tool_calls(self, tool_calls: List[Dict[str, Any]], runners: Dict[str, Any], capture_chars: int = 0) -> List[Dict[str, Any]]:
        messages = []
        for tool_call in tool_calls:
            name = _tool_call_name(tool_call)
            tool = runners.get(name)
            if tool is None:
                content = _tool_error_content("UnknownTool", f"Unknown tool: {name}")
            else:
                try:
                    content = await tool.arun_to_str(**_tool_call_arguments(tool_call))
                except Exception as exc:
                    content = _tool_error_content(type(exc).__name__, str(exc))
            messages.append({"role": "tool", "tool_call_id": tool_call.get("id") or name, "name": name, "content": _clip_tool_result(content, capture_chars)})
        return messages

    def _trace_request(self, ctx: Dict[str, Any], messages: List[Dict[str, Any]], call: str = "round") -> Optional[int]:
        """Emit one ``request`` trace event for an imminent provider call.

        Args:
            ctx (Dict[str, Any]): Chat context; a missing ``trace`` state
                makes this a no-op.
            messages (List[Dict[str, Any]]): Messages exactly as the provider
                call will receive them.
            call (str): Event ``call`` tag. Defaults to ``"round"``.

        Returns:
            Optional[int]: Sequence number pairing the terminal event, or
            ``None`` when tracing is off.
        """
        state = ctx.get("trace")
        if not state:
            return None
        seq = state["seq"]
        state["seq"] = seq + 1
        payload = canonical_request(messages, ctx["cfg"], self.spec, streamable=ctx.get("streamable"))
        emit_event(state, LLMTraceEvent(kind="request", call=call, seq=seq, payload=payload))
        return seq

    @staticmethod
    def _trace_response(ctx: Dict[str, Any], seq: Optional[int], response: LLMResponse, call: str = "round") -> None:
        """Emit the ``response`` trace event paired with ``seq``."""
        state = ctx.get("trace")
        if not state or seq is None:
            return
        raw = response.raw
        while isinstance(raw, list) and raw:
            raw = raw[-1]
        served = getattr(raw, "model", None) or (raw.get("model") if hasattr(raw, "get") else None)
        payload = {
            "message": deepcopy(response.message),
            "usage": dict(response.usage or {}),
            "served_model": str(served or ""),
            "elapsed": response.elapsed,
        }
        emit_event(state, LLMTraceEvent(kind="response", call=call, seq=seq, payload=payload))

    @staticmethod
    def _trace_error(ctx: Dict[str, Any], seq: Optional[int], error: Exception, call: str = "round") -> None:
        """Emit the ``error`` trace event paired with ``seq``."""
        state = ctx.get("trace")
        if not state or seq is None:
            return
        emit_event(state, LLMTraceEvent(kind="error", call=call, seq=seq, payload={"error": f"{type(error).__name__}: {error}"}))

    def _context_window(self) -> int:
        """Return this route's model context window in tokens.

        The model registry entry's explicit ``context_window`` wins; otherwise
        the conservative 128000-token fallback protects routes whose metadata
        does not declare a window.
        """
        model_cfg = self.spec._model_cfg()
        window = model_cfg.get("context_window")
        if isinstance(window, int) and not isinstance(window, bool) and window > 0:
            return window
        return _CONTEXT_WINDOW_FALLBACK

    def _turn_compaction_prompt(
        self,
        prepared: List[Dict[str, Any]],
        rounds: List[List[Dict[str, Any]]],
        cap_tokens: float,
        usage_base: "Optional[tuple[int, int]]" = None,
    ) -> Optional[str]:
        """Return the compaction prompt when the current tool rounds exceed their token cap.

        Args:
            prepared (List[Dict[str, Any]]): Immutable head messages.
            rounds (List[List[Dict[str, Any]]]): This turn's per-round message
                chunks, oldest first.
            cap_tokens (float): Token ceiling for the assembled conversation
                (normally ``compact_ratio`` × the model window).
            usage_base (tuple[int, int] | None): ``(prompt_tokens, chars)`` of
                the newest provider response that reported usage: the real
                token count and the measured size of the prompt it covered.
                ``None`` on cold start or usage-less providers.

        Returns:
            str | None: Provider prompt for eligible rounds, else ``None``.
        """
        chars = _conversation_chars([*prepared, *(message for chunk in rounds for message in chunk)])
        # Fullness accounting prefers the provider's reported prompt usage as
        # the base, uses the estimator only
        # for the delta appended since that response was measured, and the pure
        # estimate on cold start or when the provider reports no usage.
        if usage_base is not None:
            reported, measured_chars = usage_base
            fullness = reported + max(0, chars - measured_chars) / _CHARS_PER_TOKEN
        else:
            fullness = chars / _CHARS_PER_TOKEN
        if len(rounds) < 3 or fullness < cap_tokens:
            return None
        lines: List[str] = []
        for message in (message for chunk in rounds[:-2] for message in chunk):
            role = str(message.get("role") or "")
            for call in message.get("tool_calls") or ():
                function = call.get("function") if isinstance(call, Mapping) else None
                if isinstance(function, Mapping):
                    lines.append(f"{role} called {function.get('name')}({str(function.get('arguments') or '')})")
            content = message.get("content")
            if content:
                lines.append(f"{role}: {content}")
        return (
            "Compress this turn's earlier tool rounds into working memory, not prose. "
            "Preserve every concrete value, measurement, and conclusion from those rounds.\n\n" + "\n".join(lines)
        )

    def _compacted_turn_rounds(self, rounds: List[List[Dict[str, Any]]], reply: Any) -> tuple[List[List[Dict[str, Any]]], bool]:
        """Replace old tool rounds with a validated compacted-memory message.

        The memo normally comes before the two kept rounds. In a Claude thinking turn it
        comes after them instead: with thinking on, the Messages
        API requires the request's final assistant turn, the assistant messages after the last
        user text, to start with a thinking block, and Claude writes that block in the turn's
        first round, which compaction drops. Ending the request on the memo starts a fresh
        assistant turn, and the kept rounds become an earlier turn, where thinking may be omitted.
        They are sent without their thinking blocks: removing the rounds before them changed what
        those blocks followed, which Claude models with Anthropic's history-editing check refuse
        for a replayed block, and an earlier turn needs none.
        """
        summary = str((reply.get("text") if hasattr(reply, "get") else reply) or "").strip()
        if not summary:
            raise ValueError("in-turn compaction returned an empty summary")
        replay_format = _claude_replay_format(self.spec)
        replay_fields = {"thinking_blocks", "reasoning_details"} if replay_format == "reasoning_details" else {"thinking_blocks"}
        if any(message.get(key) for chunk in rounds for message in chunk for key in replay_fields):
            memo = {"role": "user", "content": f"[compacted working memory of this turn's tool rounds before the two above; continue the task]\n{summary}"}
            kept = [[{key: value for key, value in message.items() if key not in replay_fields} for message in chunk] for chunk in rounds[-2:]]
            if replay_format == "content_blocks":
                for chunk in kept:
                    for message in chunk:
                        if isinstance(message.get("content_blocks"), list):
                            message["content_blocks"] = _without_thinking_content_blocks(message["content_blocks"])
            return [*kept, [memo]], False
        memo = {"role": "user", "content": f"[compacted working memory of this turn's earlier tool rounds]\n{summary}"}
        return [[memo], *rounds[-2:]], False

    @staticmethod
    def _compaction_fallback(rounds: List[List[Dict[str, Any]]]) -> tuple[List[List[Dict[str, Any]]], bool]:
        """Keep gathered evidence and request a tool-free answer after compaction fails."""
        note = {"role": "user", "content": "[context is nearly full — answer now from the evidence already gathered]"}
        return [*rounds, [note]], True

    def _compact_turn_rounds(
        self,
        prepared: List[Dict[str, Any]],
        rounds: List[List[Dict[str, Any]]],
        cap_tokens: float,
        usage_base: "Optional[tuple[int, int]]" = None,
        trace_state: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[List[Dict[str, Any]]], bool]:
        """Synchronously compact old tool rounds when the conversation exceeds its token cap."""
        prompt = self._turn_compaction_prompt(prepared, rounds, cap_tokens, usage_base)
        if prompt is None:
            return rounds, False
        # The summary call is a provider call of this turn: its events forward
        # into the outer trace under the shared sequence.
        trace_kwargs = {"trace": forward_trace(trace_state, "turn_compaction")} if trace_state else {}
        try:
            return self._compacted_turn_rounds(rounds, self.chat(prompt, cache=False, **trace_kwargs))
        except Exception:
            return self._compaction_fallback(rounds)

    async def _acompact_turn_rounds(
        self,
        prepared: List[Dict[str, Any]],
        rounds: List[List[Dict[str, Any]]],
        cap_tokens: float,
        usage_base: "Optional[tuple[int, int]]" = None,
        trace_state: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[List[Dict[str, Any]]], bool]:
        """Asynchronously compact old tool rounds when the conversation exceeds its token cap."""
        prompt = self._turn_compaction_prompt(prepared, rounds, cap_tokens, usage_base)
        if prompt is None:
            return rounds, False
        trace_kwargs = {"trace": forward_trace(trace_state, "turn_compaction")} if trace_state else {}
        try:
            return self._compacted_turn_rounds(rounds, await self.achat(prompt, cache=False, **trace_kwargs))
        except Exception:
            return self._compaction_fallback(rounds)

    @staticmethod
    def _without_tools(ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return a chat context that offers the provider no tools (wrap-up mode)."""
        return {**ctx, "cfg": {key: value for key, value in ctx["cfg"].items() if key != "tools"}, "tools": []}

    def _with_tool_roundtrip(self, prepared: List[Dict[str, Any]], response: LLMResponse, ctx: Dict[str, Any]) -> LLMResponse:
        runners = ctx.get("tool_runners") or {}
        if not runners:
            return response
        budget = ctx["max_tool_turns"]
        if not response.tool_calls:
            response.max_tool_turns = budget  # a loop was available and needed no rounds
            return response
        on_round = ctx.get("on_round")
        if on_round:
            on_round(response)
        delta = list(response.delta)
        raw = [response.raw]
        usages: List[Mapping[str, Any]] = [response.usage]
        thinks: List[str] = [response.think]
        elapsed = response.elapsed
        # Per-round chunks: what the provider re-reads is rebuilt from these,
        # so mid-turn compaction can shrink it while ``delta`` keeps the full
        # evidence record.
        turn_rounds: List[List[Dict[str, Any]]] = [list(response.delta)]
        final = response
        tool_turns = 0
        wrap_up = False
        # Real usage in charge whenever a response reports it: (prompt_tokens,
        # measured chars of the prompt it covered) anchors the fullness check.
        usage_base = None
        reported = _prompt_tokens(response.usage)
        if reported:
            usage_base = (reported, _conversation_chars(prepared))
        for _ in range(budget):
            if not final.tool_calls:
                break
            tool_messages = self._run_tool_calls(final.tool_calls, runners, ctx["tool_result_capture_chars"])
            tool_turns += 1
            delta.extend(tool_messages)
            turn_rounds[-1].extend(tool_messages)
            if ctx["compact_ratio"] and not wrap_up:
                trace_kwargs = {"trace_state": ctx["trace"]} if "trace" in ctx else {}
                compacted, wrap_up = self._compact_turn_rounds(prepared, turn_rounds, ctx["compact_ratio"] * self._context_window(), usage_base, **trace_kwargs)
                if compacted is not turn_rounds:
                    usage_base = None  # the provider re-reports on the shrunken prompt next round
                turn_rounds = compacted
                if wrap_up:
                    ctx = self._without_tools(ctx)
            current = [*prepared, *(message for chunk in turn_rounds for message in chunk)]
            seq = self._trace_request(ctx, current)
            try:
                final = self._chat_step(current, ctx)
            except Exception as error:
                self._trace_error(ctx, seq, error)
                raise
            self._trace_response(ctx, seq, final)
            reported = _prompt_tokens(final.usage)
            if reported:
                usage_base = (reported, _conversation_chars(current))
            if on_round:
                on_round(final)
            delta.extend(final.delta)
            turn_rounds.append(list(final.delta))
            raw.append(final.raw)
            usages.append(final.usage)
            thinks.append(final.think)
            elapsed += final.elapsed
        aggregate = LLMResponse(
            text=final.text,
            # every round reasons (why these tools, then how to read their results);
            # the aggregate keeps the whole chain, not just the last round
            think="\n\n".join(part for part in thinks if part),
            tool_calls=final.tool_calls,
            message=final.message,
            usage=_merge_usage(usages),
            raw=raw,
            elapsed=round(elapsed, 4),
            messages=[*prepared, *delta],
            delta=delta,
            structured=final.structured,
            tool_turns=tool_turns,
            max_tool_turns=budget,
            # the loop only leaves tool calls standing when it ran out of rounds;
            # every clean finish breaks above with none pending
            truncated=bool(final.tool_calls),
        )
        return _mark_tool_budget(aggregate) if aggregate.truncated else aggregate

    async def _awith_tool_roundtrip(self, prepared: List[Dict[str, Any]], response: LLMResponse, ctx: Dict[str, Any]) -> LLMResponse:
        runners = ctx.get("tool_runners") or {}
        if not runners:
            return response
        budget = ctx["max_tool_turns"]
        if not response.tool_calls:
            response.max_tool_turns = budget  # a loop was available and needed no rounds
            return response
        delta = list(response.delta)
        raw = [response.raw]
        thinks: List[str] = [response.think]
        usages: List[Mapping[str, Any]] = [response.usage]
        elapsed = response.elapsed
        turn_rounds: List[List[Dict[str, Any]]] = [list(response.delta)]
        final = response
        tool_turns = 0
        wrap_up = False
        usage_base = None
        reported = _prompt_tokens(response.usage)
        if reported:
            usage_base = (reported, _conversation_chars(prepared))
        for _ in range(budget):
            if not final.tool_calls:
                break
            tool_messages = await self._arun_tool_calls(final.tool_calls, runners, ctx["tool_result_capture_chars"])
            tool_turns += 1
            delta.extend(tool_messages)
            turn_rounds[-1].extend(tool_messages)
            if ctx["compact_ratio"] and not wrap_up:
                trace_kwargs = {"trace_state": ctx["trace"]} if "trace" in ctx else {}
                compacted, wrap_up = await self._acompact_turn_rounds(
                    prepared,
                    turn_rounds,
                    ctx["compact_ratio"] * self._context_window(),
                    usage_base,
                    **trace_kwargs,
                )
                if compacted is not turn_rounds:
                    usage_base = None
                turn_rounds = compacted
                if wrap_up:
                    ctx = self._without_tools(ctx)
            current = [*prepared, *(message for chunk in turn_rounds for message in chunk)]
            seq = self._trace_request(ctx, current)
            try:
                final = await self._achat_step(current, ctx)
            except Exception as error:
                self._trace_error(ctx, seq, error)
                raise
            self._trace_response(ctx, seq, final)
            reported = _prompt_tokens(final.usage)
            if reported:
                usage_base = (reported, _conversation_chars(current))
            delta.extend(final.delta)
            turn_rounds.append(list(final.delta))
            raw.append(final.raw)
            usages.append(final.usage)
            thinks.append(final.think)
            elapsed += final.elapsed
        aggregate = LLMResponse(
            text=final.text,
            think="\n\n".join(part for part in thinks if part),
            tool_calls=final.tool_calls,
            message=final.message,
            usage=_merge_usage(usages),
            raw=raw,
            elapsed=round(elapsed, 4),
            messages=[*prepared, *delta],
            delta=delta,
            structured=final.structured,
            tool_turns=tool_turns,
            max_tool_turns=budget,
            truncated=bool(final.tool_calls),
        )
        return _mark_tool_budget(aggregate) if aggregate.truncated else aggregate

    def _chat_once(
        self,
        prepared: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        repair: Optional[LLMToolCallRepair] = None,
        repair_option: Any = None,
        tools: Any = None,
    ) -> LLMResponse:
        suppress_think = _think_suppressed(cfg)
        with NetworkProxy(http_proxy=cfg.pop("http_proxy", None), https_proxy=cfg.pop("https_proxy", None), no_proxy=cfg.pop("no_proxy", None)):
            t0 = perf_counter()
            raw = chat_complete(self.spec, prepared, stream=False, **cfg)
            return self._chat_response(
                raw,
                elapsed=round(perf_counter() - t0, 4),
                messages=prepared,
                repair=repair,
                repair_option=repair_option,
                tools=tools,
                suppress_think=suppress_think,
            )

    async def _achat_once(
        self,
        prepared: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        repair: Optional[LLMToolCallRepair] = None,
        repair_option: Any = None,
        tools: Any = None,
    ) -> LLMResponse:
        suppress_think = _think_suppressed(cfg)
        with NetworkProxy(http_proxy=cfg.pop("http_proxy", None), https_proxy=cfg.pop("https_proxy", None), no_proxy=cfg.pop("no_proxy", None)):
            t0 = perf_counter()
            raw = await _maybe_await(achat_complete(self.spec, prepared, stream=False, **cfg))
            return self._chat_response(
                raw,
                elapsed=round(perf_counter() - t0, 4),
                messages=prepared,
                repair=repair,
                repair_option=repair_option,
                tools=tools,
                suppress_think=suppress_think,
            )

    def _chat_deltas(self, prepared: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Generator[LLMResponse, None, None]:
        suppress_think = _think_suppressed(cfg)
        with NetworkProxy(http_proxy=cfg.pop("http_proxy", None), https_proxy=cfg.pop("https_proxy", None), no_proxy=cfg.pop("no_proxy", None)):
            t0 = perf_counter()
            thinking: Optional[bool] = None
            for chunk in chat_complete(self.spec, prepared, stream=True, **cfg):
                elapsed = round(perf_counter() - t0, 4)
                response = self._chat_response(
                    chunk, elapsed=elapsed, messages=prepared, delta=response_dict(chunk), repair_option=False, suppress_think=suppress_think
                )
                # A chunk that carries only a thinking-block fragment (such as its signature) is kept for
                # the gathered message; callers of a stream never see it (see _shows_output).
                if not (_shows_output(response) or response.message.get("thinking_blocks") or response.message.get("reasoning_details")):
                    continue
                content = ""
                if response.think:
                    if thinking is None:
                        thinking = True
                        content += "<think>\n"
                    content += response.think
                if response.text:
                    if thinking is True:
                        thinking = False
                        content += "\n</think>\n"
                    content += response.text
                response.content_override = content or None
                yield response

    async def _achat_deltas(self, prepared: List[Dict[str, Any]], cfg: Dict[str, Any]) -> AsyncGenerator[LLMResponse, None]:
        suppress_think = _think_suppressed(cfg)
        with NetworkProxy(http_proxy=cfg.pop("http_proxy", None), https_proxy=cfg.pop("https_proxy", None), no_proxy=cfg.pop("no_proxy", None)):
            t0 = perf_counter()
            thinking: Optional[bool] = None
            stream = await _maybe_await(achat_complete(self.spec, prepared, stream=True, **cfg))
            async for chunk in _aiter(stream):
                elapsed = round(perf_counter() - t0, 4)
                response = self._chat_response(
                    chunk, elapsed=elapsed, messages=prepared, delta=response_dict(chunk), repair_option=False, suppress_think=suppress_think
                )
                # A chunk that carries only a thinking-block fragment (such as its signature) is kept for
                # the gathered message; callers of a stream never see it (see _shows_output).
                if not (_shows_output(response) or response.message.get("thinking_blocks") or response.message.get("reasoning_details")):
                    continue
                content = ""
                if response.think:
                    if thinking is None:
                        thinking = True
                        content += "<think>\n"
                    content += response.think
                if response.text:
                    if thinking is True:
                        thinking = False
                        content += "\n</think>\n"
                    content += response.text
                response.content_override = content or None
                yield response

    def _gather_responses(
        self,
        prepared: List[Dict[str, Any]],
        responses: Iterable[LLMResponse],
        repair: Optional[LLMToolCallRepair] = None,
        repair_option: Any = None,
        tools: Any = None,
    ) -> LLMResponse:
        text = ""
        think = ""
        raw = []
        usages: List[Mapping[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        fragments: List[Any] = []
        reasoning_fragments: List[Dict[str, Any]] = []
        elapsed = 0.0
        for delta in responses:
            text += delta.text
            think += delta.think
            raw.append(delta.raw)
            elapsed = delta.elapsed
            if delta.usage:
                usages.append(delta.usage)
            _merge_tool_calls(tool_calls, delta.tool_calls)
            blocks = (delta.message or {}).get("thinking_blocks")
            fragments.extend(blocks if isinstance(blocks, list) else [])
            details = (delta.message or {}).get("reasoning_details")
            if isinstance(details, list):
                reasoning_fragments.extend(part for part in details if isinstance(part, dict))
        message = {"role": "assistant", "content": text}
        if think:
            message["reasoning_content"] = think
        # Only complete, signed blocks, put back together byte for byte: a provider needs them unmodified.
        if thinking_blocks := _join_thinking_blocks(fragments):
            message["thinking_blocks"] = thinking_blocks
        if reasoning_fragments and _claude_replay_format(self.spec) == "reasoning_details":
            message["reasoning_details"] = _join_reasoning_details(reasoning_fragments)
        if tool_calls:
            tool_calls = self._repair_tool_calls(tool_calls, repair=repair, repair_option=repair_option, tools=tools)
            message["tool_calls"] = tool_calls
        message_delta = _message_delta(message)
        return LLMResponse(
            text=text,
            think=think,
            tool_calls=tool_calls,
            message=message,
            usage=_merge_usage(usages),
            raw=raw,
            elapsed=elapsed,
            messages=[*prepared, *message_delta],
            delta=message_delta,
            structured=_structured(text, message),
        )

    def _gather_chat(
        self,
        prepared: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        chunks: Optional[List[Dict[str, Any]]] = None,
        repair: Optional[LLMToolCallRepair] = None,
        repair_option: Any = None,
        tools: Any = None,
    ) -> LLMResponse:
        responses = []
        for delta in self._chat_deltas(prepared, cfg):
            if chunks is not None:
                chunks.append(delta.to_dict())
            responses.append(delta)
        return self._gather_responses(prepared, responses, repair=repair, repair_option=repair_option, tools=tools)

    async def _agather_chat(
        self,
        prepared: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        chunks: Optional[List[Dict[str, Any]]] = None,
        repair: Optional[LLMToolCallRepair] = None,
        repair_option: Any = None,
        tools: Any = None,
    ) -> LLMResponse:
        responses = []
        async for delta in self._achat_deltas(prepared, cfg):
            if chunks is not None:
                chunks.append(delta.to_dict())
            responses.append(delta)
        return self._gather_responses(prepared, responses, repair=repair, repair_option=repair_option, tools=tools)

    def chat(
        self,
        messages: Messages,
        include: Optional[Union[str, Iterable[str]]] = None,
        reduce: bool = True,
        stream: bool = False,
        system: Optional[str] = None,
        images: ImageInputs = None,
        tools: Any = None,
        on_round: Optional[Callable[[LLMResponse], None]] = None,
        trace: Optional[TraceCallback] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run a chat completion and return only the requested include fields.

        Use this method for regular text, image, and tool-call requests. By default it performs a streaming
        provider request and gathers chunks into one normalized response; pass ``stream=True`` to get
        per-chunk projections instead. The ``include`` parameter is the main response-shaping tool:
        pass ``None`` for the default ``text`` value, a string such as ``"message"`` for one field,
        or a list such as ``["text", "tool_calls", "usage", "elapsed", "raw"]`` for multiple fields.
        Supported chat include fields are:

                - ``text``: visible assistant text, excluding separate reasoning chunks.
                - ``think``: provider reasoning/thinking text when exposed separately.
                - ``content``: ``think`` wrapped in ``<think>`` tags followed by ``text``.
                - ``tool_calls``: normalized OpenAI ``tool_calls`` list from the assistant response, or from the full new ``delta`` when executable tools are run automatically.
                - ``message``: the OpenAI-format assistant message dictionary with ``role``, ``content``,
                    and optional ``tool_calls``. This is the assistant response itself, not the whole history.
                    A Claude thinking turn on an Anthropic Messages route or on LiteLLM also carries
                    ``thinking_blocks``, which are sent back with its tool calls.
                - ``delta``: new OpenAI-format messages produced by this inference call. It starts with
                    ``message`` and will append tool-result messages after it once tool execution is wired in.
                - ``messages``: full conversation history, i.e. normalized input messages plus ``delta``.
                - ``usage``: provider usage counters for this call. Current providers commonly expose
                    ``prompt_tokens``, ``completion_tokens``, and ``total_tokens``; streaming calls merge usage
                    chunks by summing numeric counters and keeping the first non-numeric value per key.
                    A response that reports prompt and completion counts but no total gets
                    ``total_tokens`` as their sum, streamed or not; a tool loop sums every round's total.
                - ``raw``: raw provider response object or gathered raw stream chunks.
                - ``elapsed``: request elapsed seconds measured by HeavenBase.
                - ``created_at``: local response creation timestamp from ``time.time()``.
                - ``structured``: parsed JSON/object output when available, otherwise ``None``.
                - ``truncated``: whether the executable-tool loop stopped on its round budget
                    while the model was still calling tools, so the answer is mid-investigation.
                - ``tool_turns``: executable-tool rounds actually run.
                - ``max_tool_turns``: the round budget the loop was given, or ``None`` when no
                    executable-tool loop applied.

        Unknown field names raise through ``raise_mismatch``.

        Executable tools run in a loop bounded by ``max_tool_turns``, which defaults to 8. When the
        budget runs out with tool calls still pending, this call returns the mid-conversation answer
        rather than raising: ``truncated`` is ``True``, and both ``text`` and the final assistant
        message carry an explicit ``[incomplete: ...]`` note so the default ``text`` projection cannot
        be mistaken for a finished answer. ``tool_result_capture_chars`` (default 0) optionally clips
        each executed tool result's message at an explicit character boundary with an actionable
        truncation note; the recorded ``delta`` carries the same clipped message. ``compact_ratio``
        (default 0) optionally auto-compacts the current turn mid-loop: when the assembled conversation overflows
        the threshold, the oldest tool rounds fold into one summary message written by this same LLM
        while the head and the newest two rounds stay verbatim (``delta`` keeps the full record); a
        failed summary wraps the turn up gracefully without tools instead of failing.

        Args:
            messages (Messages): Prompt string, message dictionary, or iterable of message items.
            include (Optional[Union[str, Iterable[str]]]): Field or fields to return. Defaults to ``None``.
            reduce (bool): Whether a one-field include returns the value directly. Defaults to ``True``.
            stream (bool): Whether to return a generator of chunk projections. Defaults to ``False``.
            system (Optional[str]): Optional system message prepended to the chat. Defaults to ``None``.
            images (ImageInputs): Optional image input or image inputs attached to the last user message. Defaults to ``None``.
            tools (Any): Optional tool or list of tools. Items can be OpenAI-compatible schema dictionaries, Python callables, HeavenBase ``Tool`` objects, or HeavenBase ``Toolkit`` objects. Executable items are run automatically and their tool-result messages are included in ``delta``.
            on_round (Optional[Callable[[LLMResponse], None]]): Executable-tool
                loops call this with each completed round's response so callers
                can surface per-round reasoning live. Defaults to ``None``.
            trace (Optional[TraceCallback]): Provider-call observer. When set,
                every provider-bound request of this call — first round, tool
                rounds, and in-turn compaction summaries — emits a ``request``
                trace event before provider I/O and a ``response``/``error``
                event after it; a cache hit emits nothing. See
                :mod:`heavenbase.llm.trace`. Defaults to ``None``.
            **kwargs (Any): Provider request options such as ``temperature``, ``tool_choice``, or ``response_format``.

        Returns:
            Any: Include-projected chat result or a generator when ``stream=True``.
        """
        if stream:
            if trace is not None:
                kwargs["trace"] = trace
            return self.stream(messages, include=include, reduce=reduce, system=system, images=images, tools=tools, on_round=on_round, **kwargs)
        if tools is not None:
            kwargs["tools"] = tools
        ctx = self._chat_context(messages, system, images, kwargs)
        ctx["on_round"] = on_round
        if trace is not None:
            ctx["trace"] = trace_state(trace)
        if ctx["active"]:
            row = ctx["cache"].get_text(ctx["key"])
            if row:
                response = LLMResponse.from_dict(row.get("response") or {})
                response = _strip_response_think(response) if ctx["suppress_think"] else response
                return response.include(include, reduce=reduce)
        chunks: List[Dict[str, Any]] = []
        seq = self._trace_request(ctx, ctx["prepared"])
        try:
            response = self._chat_step(ctx["prepared"], ctx, chunks=chunks)
        except Exception as error:
            self._trace_error(ctx, seq, error)
            raise
        self._trace_response(ctx, seq, response)
        response = self._with_tool_roundtrip(ctx["prepared"], response, ctx)
        result = response.include(include, reduce=reduce)
        if ctx["active"]:
            ctx["cache"].set_text(ctx["key"], ctx["prepared"], response.to_dict(), chunks, policy=ctx["policy"] or "deterministic")
        return result

    async def achat(
        self,
        messages: Messages,
        include: Optional[Union[str, Iterable[str]]] = None,
        reduce: bool = True,
        stream: bool = False,
        system: Optional[str] = None,
        images: ImageInputs = None,
        tools: Any = None,
        trace: Optional[TraceCallback] = None,
        **kwargs: Any,
    ) -> Any:
        """Run an async chat completion.

        This is the async counterpart to ``chat``. It uses the same include projection, cache policy, tool-call repair, and
        executable-tool loop, including the ``max_tool_turns`` budget of 8 rounds and its ``truncated`` /
        ``[incomplete: ...]`` marking when that budget runs out mid-answer. Pass ``stream=True`` to receive the async
        stream returned by ``astream``.

        Args:
            messages (Messages): Prompt string, message dictionary, or iterable of message items.
            include (Optional[Union[str, Iterable[str]]]): Field or fields to return. Defaults to ``None``.
            reduce (bool): Whether a one-field include returns the value directly. Defaults to ``True``.
            stream (bool): Whether to return an async generator of chunk projections. Defaults to ``False``.
            system (Optional[str]): Optional system message prepended to the chat. Defaults to ``None``.
            images (ImageInputs): Optional image input or image inputs attached to the last user message. Defaults to ``None``.
            tools (Any): Optional schema or executable tools using the same normalization as ``chat``. Defaults to ``None``.
            trace (Optional[TraceCallback]): Provider-call observer with the
                same contract as ``chat``. Defaults to ``None``.
            **kwargs (Any): Provider request options such as ``temperature``, ``tool_choice``, or ``response_format``.

        Returns:
            Any: Include-projected chat result, or an async generator when ``stream=True``.
        """
        if stream:
            if trace is not None:
                kwargs["trace"] = trace
            return self.astream(messages, include=include, reduce=reduce, system=system, images=images, tools=tools, **kwargs)
        if tools is not None:
            kwargs["tools"] = tools
        ctx = self._chat_context(messages, system, images, kwargs)
        if trace is not None:
            ctx["trace"] = trace_state(trace)
        if ctx["active"]:
            row = ctx["cache"].get_text(ctx["key"])
            if row:
                response = LLMResponse.from_dict(row.get("response") or {})
                response = _strip_response_think(response) if ctx["suppress_think"] else response
                return response.include(include, reduce=reduce)
        chunks: List[Dict[str, Any]] = []
        seq = self._trace_request(ctx, ctx["prepared"])
        try:
            response = await self._achat_step(ctx["prepared"], ctx, chunks=chunks)
        except Exception as error:
            self._trace_error(ctx, seq, error)
            raise
        self._trace_response(ctx, seq, response)
        response = await self._awith_tool_roundtrip(ctx["prepared"], response, ctx)
        result = response.include(include, reduce=reduce)
        if ctx["active"]:
            ctx["cache"].set_text(ctx["key"], ctx["prepared"], response.to_dict(), chunks, policy=ctx["policy"] or "deterministic")
        return result

    def _stream_tool_loop(self, ctx: Dict[str, Any], include: Optional[Union[str, Iterable[str]]], reduce: bool) -> Generator[Any, None, None]:
        """Stream every round of a tool conversation, executing tools between rounds.

        Yield order per round: the provider's live chunks (reasoning/text/
        tool-call deltas), then one synthetic boundary chunk whose ``delta``
        carries the round's assistant message plus the executed tool-result
        messages. After the last round a final aggregate chunk follows when the
        ``include`` set requests final-delta fields, mirroring ``chat`` +
        roundtrip semantics (joined ``think``, merged usage, full ``delta``,
        and the ``truncated`` verdict on the round budget).

        Args:
            ctx (Dict[str, Any]): Prepared chat context from ``_chat_context``.
            include (Optional[Union[str, Iterable[str]]]): Chunk projection fields.
            reduce (bool): Whether a one-field include returns the value directly.

        Yields:
            Any: Include-projected stream chunk.
        """
        if include is not None and not isinstance(include, str):
            include = tuple(include)
        runners = ctx["tool_runners"]
        on_round = ctx.get("on_round")
        current = list(ctx["prepared"])
        delta: List[Dict[str, Any]] = []
        raws: List[Any] = []
        usages: List[Mapping[str, Any]] = []
        thinks: List[str] = []
        elapsed = 0.0
        final: Optional[LLMResponse] = None
        tool_turns = 0
        truncated = False
        # Per-round chunks: what the provider re-reads is rebuilt from these,
        # so mid-turn compaction can shrink it while ``delta`` keeps the full
        # evidence record.
        turn_rounds: List[List[Dict[str, Any]]] = []
        wrap_up = False
        # Real usage in charge whenever a response reports it: (prompt_tokens,
        # measured chars of the prompt it covered) anchors the fullness check.
        usage_base = None
        for round_index in range(ctx["max_tool_turns"] + 1):
            seq = self._trace_request(ctx, current)
            try:
                if ctx["streamable"]:
                    responses: List[LLMResponse] = []
                    for resp in self._chat_deltas(current, deepcopy(ctx["cfg"])):
                        responses.append(resp)
                        if _shows_stream_output(resp, include):
                            yield resp.include(include, reduce=reduce)
                    round_response = self._gather_responses(current, responses, repair=ctx["repair"], repair_option=ctx["repair_option"], tools=ctx["tools"])
                    self._trace_response(ctx, seq, round_response)
                else:
                    round_response = self._chat_once(
                        current, deepcopy(ctx["cfg"]), repair=ctx["repair"], repair_option=ctx["repair_option"], tools=ctx["tools"]
                    )
                    # Terminal event before the yield: the provider already
                    # completed, so a consumer disconnecting here must not
                    # leave an interrupted-looking request row.
                    self._trace_response(ctx, seq, round_response)
                    yield round_response.include(include, reduce=reduce)
            except Exception as error:
                self._trace_error(ctx, seq, error)
                raise
            if on_round:
                on_round(round_response)
            raws.append(round_response.raw)
            usages.append(round_response.usage)
            thinks.append(round_response.think)
            elapsed += round_response.elapsed
            reported = _prompt_tokens(round_response.usage)
            if reported:
                usage_base = (reported, _conversation_chars(current))
            delta.extend(round_response.delta)
            turn_rounds.append(list(round_response.delta))
            final = round_response
            if not round_response.tool_calls or round_index == ctx["max_tool_turns"]:
                # calls still standing at the budget mean the answer is a prefix
                truncated = bool(round_response.tool_calls)
                break
            tool_messages = self._run_tool_calls(round_response.tool_calls, runners, ctx["tool_result_capture_chars"])
            tool_turns += 1
            delta.extend(tool_messages)
            turn_rounds[-1].extend(tool_messages)
            if ctx["compact_ratio"] and not wrap_up:
                trace_kwargs = {"trace_state": ctx["trace"]} if "trace" in ctx else {}
                compacted, wrap_up = self._compact_turn_rounds(
                    ctx["prepared"], turn_rounds, ctx["compact_ratio"] * self._context_window(), usage_base, **trace_kwargs
                )
                if compacted is not turn_rounds:
                    usage_base = None  # the provider re-reports on the shrunken prompt next round
                turn_rounds = compacted
                if wrap_up:
                    ctx = self._without_tools(ctx)
            current = [*ctx["prepared"], *(message for chunk in turn_rounds for message in chunk)]
            boundary = LLMResponse(message=round_response.message, tool_calls=round_response.tool_calls, delta=[round_response.message, *tool_messages])
            yield boundary.include(include, reduce=reduce)
        aggregate = LLMResponse(
            text=final.text,
            think="\n\n".join(part for part in thinks if part),
            tool_calls=final.tool_calls,
            message=final.message,
            usage=_merge_usage(usages),
            raw=raws,
            elapsed=round(elapsed, 4),
            messages=[*ctx["prepared"], *delta],
            delta=delta,
            structured=final.structured,
            tool_turns=tool_turns,
            max_tool_turns=ctx["max_tool_turns"],
            truncated=truncated,
        )
        if truncated:
            _mark_tool_budget(aggregate)
        emit_note = truncated and _includes_text(include)
        if _needs_final_delta(include) or emit_note:
            closing = _stream_final_response(aggregate, with_usage=False)
            if emit_note:
                closing.text = _TOOL_BUDGET_NOTE.format(turns=aggregate.max_tool_turns)
            yield closing.include(include, reduce=reduce)
        if ctx["active"]:
            ctx["cache"].set_text(ctx["key"], ctx["prepared"], aggregate.to_dict(), [], policy=ctx["policy"] or "deterministic")

    def stream(
        self,
        messages: Messages,
        include: Optional[Union[str, Iterable[str]]] = None,
        reduce: bool = True,
        system: Optional[str] = None,
        images: ImageInputs = None,
        tools: Any = None,
        on_round: Optional[Callable[[LLMResponse], None]] = None,
        trace: Optional[TraceCallback] = None,
        **kwargs: Any,
    ) -> Generator[Any, None, None]:
        """
        Stream chat completion chunks with include-based projection.

        Use this method when callers need progressive text, reasoning, tool-call deltas, usage
        updates, or raw provider chunks. ``include`` follows the same rules as ``chat`` and validates
        requested fields with ``raise_mismatch``.

        Executable tools stream every round under the same ``max_tool_turns`` budget of 8. Including
        ``delta``, ``messages``, ``truncated``, ``tool_turns``, or ``max_tool_turns`` appends a closing
        aggregate chunk, because only that chunk knows the whole conversation and whether the loop ran
        out of rounds. A loop cut short also annotates its final assistant message with an explicit
        ``[incomplete: ...]`` note.

        Args:
            messages (Messages): Prompt string, message dictionary, or iterable of message items.
            include (Optional[Union[str, Iterable[str]]]): Field or fields to return for each chunk. Defaults to ``None``.
            reduce (bool): Whether a one-field include returns the value directly. Defaults to ``True``.
            system (Optional[str]): Optional system message prepended to the chat. Defaults to ``None``.
            images (ImageInputs): Optional image input or image inputs attached to the last user message. Defaults to ``None``.
            tools (Any): Optional tool or list of tools using the same normalization as ``chat``.
            on_round (Optional[Callable[[LLMResponse], None]]): Executable-tool
                loops call this with each completed round's response. Defaults
                to ``None``.
            trace (Optional[TraceCallback]): Provider-call observer with the
                same contract as ``chat``. Defaults to ``None``.
            **kwargs (Any): Provider request options such as ``temperature`` or ``tool_choice``.

        Yields:
            Any: Include-projected stream chunk.
        """
        if include is not None and not isinstance(include, str):
            include = tuple(include)
        if tools is not None:
            kwargs["tools"] = tools
        ctx = self._chat_context(messages, system, images, kwargs)
        ctx["on_round"] = on_round
        if trace is not None:
            ctx["trace"] = trace_state(trace)
        if ctx["tool_runners"]:
            yield from self._stream_tool_loop(ctx, include, reduce)
            return
        if ctx["active"]:
            row = ctx["cache"].get_text(ctx["key"])
            if row:
                chunks = row.get("chunks") or []
                response = LLMResponse.from_dict(row.get("response") or {})
                response = _strip_response_think(response) if ctx["suppress_think"] else response
                if chunks:
                    for chunk in chunks:
                        chunk_response = LLMResponse.from_dict(chunk)
                        chunk_response = _strip_response_think(chunk_response) if ctx["suppress_think"] else chunk_response
                        if _shows_stream_output(chunk_response, include):
                            yield chunk_response.include(include, reduce=reduce)
                    if _needs_final_delta(include):
                        yield _stream_final_response(response, with_usage=False).include(include, reduce=reduce)
                else:
                    yield response.include(include, reduce=reduce)
                return
        seq = self._trace_request(ctx, ctx["prepared"])
        if ctx["streamable"]:
            chunks: List[Dict[str, Any]] = []
            responses: List[LLMResponse] = []
            try:
                for resp in self._chat_deltas(ctx["prepared"], deepcopy(ctx["cfg"])):
                    chunks.append(resp.to_dict())
                    responses.append(resp)
                    if _shows_stream_output(resp, include):
                        yield resp.include(include, reduce=reduce)
                response = self._gather_responses(ctx["prepared"], responses, repair=ctx["repair"], repair_option=ctx["repair_option"], tools=ctx["tools"])
            except Exception as error:
                self._trace_error(ctx, seq, error)
                raise
            self._trace_response(ctx, seq, response)
            if _needs_final_delta(include):
                yield _stream_final_response(response, with_usage=False).include(include, reduce=reduce)
            if ctx["active"]:
                ctx["cache"].set_text(ctx["key"], ctx["prepared"], response.to_dict(), chunks, policy=ctx["policy"] or "deterministic")
        else:
            try:
                response = self._chat_once(ctx["prepared"], deepcopy(ctx["cfg"]), repair=ctx["repair"], repair_option=ctx["repair_option"], tools=ctx["tools"])
            except Exception as error:
                self._trace_error(ctx, seq, error)
                raise
            self._trace_response(ctx, seq, response)
            yield response.include(include, reduce=reduce)
            if ctx["active"]:
                ctx["cache"].set_text(ctx["key"], ctx["prepared"], response.to_dict(), [], policy=ctx["policy"] or "deterministic")

    async def astream(
        self,
        messages: Messages,
        include: Optional[Union[str, Iterable[str]]] = None,
        reduce: bool = True,
        system: Optional[str] = None,
        images: ImageInputs = None,
        tools: Any = None,
        trace: Optional[TraceCallback] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Stream async chat completion chunks with include-based projection.

        Args:
            messages (Messages): Prompt string, message dictionary, or iterable of message items.
            include (Optional[Union[str, Iterable[str]]]): Field or fields to return for each chunk. Defaults to ``None``.
            reduce (bool): Whether a one-field include returns the value directly. Defaults to ``True``.
            system (Optional[str]): Optional system message prepended to the chat. Defaults to ``None``.
            images (ImageInputs): Optional image input or image inputs attached to the last user message. Defaults to ``None``.
            tools (Any): Optional schema or executable tools using the same normalization as ``chat``. Defaults to ``None``.
            trace (Optional[TraceCallback]): Provider-call observer with the
                same contract as ``chat``. Defaults to ``None``.
            **kwargs (Any): Provider request options such as ``temperature`` or ``tool_choice``.

        Yields:
            Any: Include-projected async stream chunk.
        """
        if include is not None and not isinstance(include, str):
            include = tuple(include)
        if tools is not None:
            kwargs["tools"] = tools
        ctx = self._chat_context(messages, system, images, kwargs)
        if trace is not None:
            ctx["trace"] = trace_state(trace)
        if ctx["tool_runners"]:
            if trace is not None:
                kwargs["trace"] = trace
            yield await self.achat(messages, include=include, reduce=reduce, stream=False, system=system, images=images, **kwargs)
            return
        if ctx["active"]:
            row = ctx["cache"].get_text(ctx["key"])
            if row:
                chunks = row.get("chunks") or []
                response = LLMResponse.from_dict(row.get("response") or {})
                response = _strip_response_think(response) if ctx["suppress_think"] else response
                if chunks:
                    for chunk in chunks:
                        chunk_response = LLMResponse.from_dict(chunk)
                        chunk_response = _strip_response_think(chunk_response) if ctx["suppress_think"] else chunk_response
                        if _shows_stream_output(chunk_response, include):
                            yield chunk_response.include(include, reduce=reduce)
                    if _needs_final_delta(include):
                        yield _stream_final_response(response, with_usage=False).include(include, reduce=reduce)
                else:
                    yield response.include(include, reduce=reduce)
                return
        seq = self._trace_request(ctx, ctx["prepared"])
        if ctx["streamable"]:
            chunks: List[Dict[str, Any]] = []
            responses: List[LLMResponse] = []
            try:
                async for resp in self._achat_deltas(ctx["prepared"], deepcopy(ctx["cfg"])):
                    chunks.append(resp.to_dict())
                    responses.append(resp)
                    if _shows_stream_output(resp, include):
                        yield resp.include(include, reduce=reduce)
                response = self._gather_responses(ctx["prepared"], responses, repair=ctx["repair"], repair_option=ctx["repair_option"], tools=ctx["tools"])
            except Exception as error:
                self._trace_error(ctx, seq, error)
                raise
            self._trace_response(ctx, seq, response)
            if _needs_final_delta(include):
                yield _stream_final_response(response, with_usage=False).include(include, reduce=reduce)
            if ctx["active"]:
                ctx["cache"].set_text(ctx["key"], ctx["prepared"], response.to_dict(), chunks, policy=ctx["policy"] or "deterministic")
        else:
            try:
                response = await self._achat_once(
                    ctx["prepared"], deepcopy(ctx["cfg"]), repair=ctx["repair"], repair_option=ctx["repair_option"], tools=ctx["tools"]
                )
            except Exception as error:
                self._trace_error(ctx, seq, error)
                raise
            self._trace_response(ctx, seq, response)
            yield response.include(include, reduce=reduce)
            if ctx["active"]:
                ctx["cache"].set_text(ctx["key"], ctx["prepared"], response.to_dict(), [], policy=ctx["policy"] or "deterministic")

    def embed(
        self,
        inputs: Union[str, Iterable[str]],
        include: Optional[Union[str, Iterable[str]]] = None,
        reduce: bool = True,
        embedding_batch_size: Optional[int] = None,
        embedding_max_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create embeddings and return include-projected fields.

        Use this method for single-string or batch embedding requests. ``include`` accepts ``None``
        for the default ``embeddings`` value, one field name, or multiple fields. Supported embedding
        include fields are ``embeddings``, ``usage``, ``raw``, ``elapsed``, ``created_at``, and
        ``dim``. Unknown names raise through ``raise_mismatch``.

        Args:
            inputs (Union[str, Iterable[str]]): One input string or a batch of input strings.
            include (Optional[Union[str, Iterable[str]]]): Field or fields to return. Defaults to ``None``.
            reduce (bool): Whether a one-field include returns the value directly. Defaults to ``True``.
            embedding_batch_size (Optional[int]): Miss batch size before provider embedding calls. Defaults to ``None``.
            embedding_max_workers (Optional[int]): Maximum concurrent miss batches. Defaults to ``None``.
            **kwargs (Any): Additional provider request options. Embedding execution controls are kept as config metadata and not sent to the provider.

        Returns:
            Any: Include-projected embedding result.
        """
        single = isinstance(inputs, str)
        batch = [inputs] if single else list(inputs)
        cfg = to_mutable_data(self.spec.args)
        cfg.update(self._request_kwargs(kwargs))
        if embedding_batch_size is not None:
            cfg["embedding_batch_size"] = embedding_batch_size
        if embedding_max_workers is not None:
            cfg["embedding_max_workers"] = embedding_max_workers
        cache_option = cfg.pop("cache", None)
        cache_policy = cfg.pop("cache_policy", None)
        n_batch, n_workers = _embedding_controls(cfg, len(batch))
        cfg.pop("dim", None)
        cache, enabled_option = self._cache_pair(cache_option)
        policy = cache.policy("embedding", cfg, cache_policy)
        active = cache.enabled("embedding", enabled_option) and policy is not None
        unique = list(dict.fromkeys(batch))
        keys = {item: cache.embedding_key(self.spec, item, cfg) for item in unique}
        rows = cache.get_embeddings(keys) if active else {}
        vectors_by_input = {item: row.get("embedding") for item, row in rows.items() if row}
        misses = [item for item in unique if item not in vectors_by_input]
        provider_batches = _batches(misses, n_batch)

        def run_provider_batch(items: List[str]) -> Dict[str, Any]:
            provider_cfg = deepcopy(cfg)
            with NetworkProxy(
                http_proxy=provider_cfg.pop("http_proxy", None), https_proxy=provider_cfg.pop("https_proxy", None), no_proxy=provider_cfg.pop("no_proxy", None)
            ):
                t0 = perf_counter()
                raw = embed_complete(self.spec, items, **provider_cfg)
                elapsed = round(perf_counter() - t0, 4)
            resp = response_dict(raw)
            data = sorted(resp.get("data", []), key=lambda item: item.get("index", 0))
            if len(data) != len(items):
                raise ValueError(f"Embedding provider returned {len(data)} vectors for {len(items)} inputs")
            return {"inputs": items, "data": data, "usage": _usage(resp), "raw": resp, "elapsed": elapsed}

        t0 = perf_counter()
        results = pmap(run_provider_batch, [{"items": items} for items in provider_batches], max_workers=n_workers) if provider_batches else []
        elapsed = round(perf_counter() - t0, 4) if provider_batches else 0.0
        writes = []
        for result in results:
            for input_text, item in zip(result["inputs"], result["data"]):
                vector = item.get("embedding")
                vectors_by_input[input_text] = vector
                if active:
                    writes.append(
                        {
                            "key": keys[input_text],
                            "input_text": input_text,
                            "embedding": vector,
                            "dim": len(vector or []),
                            "usage": result["usage"],
                            "raw": item,
                        }
                    )
        if writes:
            cache.set_embeddings(writes)
        vectors = [vectors_by_input[item] for item in batch]
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        usage = _merge_usage(result["usage"] for result in results)
        raw = {
            "cached": len(unique) - len(misses),
            "misses": len(misses),
            "embedding_batch_size": n_batch,
            "embedding_max_workers": n_workers,
            "batches": [result["raw"] for result in results],
        }
        data = {
            "embeddings": vectors[0] if single and vectors else vectors,
            "usage": usage,
            "raw": raw,
            "elapsed": elapsed,
            "created_at": time(),
            "dim": (len(vectors[0]) if vectors else 0) or self._dim,
        }
        return _project(data, include=include, default=("embeddings",), reduce=reduce)

    async def aembed(
        self,
        inputs: Union[str, Iterable[str]],
        include: Optional[Union[str, Iterable[str]]] = None,
        reduce: bool = True,
        embedding_batch_size: Optional[int] = None,
        embedding_max_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Create embeddings asynchronously and return include-projected fields.

        This is the async counterpart to ``embed``. It uses the same cache policy,
        batching, deduplication, and include projection.

        Args:
            inputs (Union[str, Iterable[str]]): One input string or a batch of input strings.
            include (Optional[Union[str, Iterable[str]]]): Field or fields to return. Defaults to ``None``.
            reduce (bool): Whether a one-field include returns the value directly. Defaults to ``True``.
            embedding_batch_size (Optional[int]): Miss batch size before provider embedding calls. Defaults to ``None``.
            embedding_max_workers (Optional[int]): Maximum concurrent miss batches. Defaults to ``None``.
            **kwargs (Any): Additional provider request options. Embedding execution controls are kept as config metadata and not sent to the provider.

        Returns:
            Any: Include-projected embedding result.
        """
        single = isinstance(inputs, str)
        batch = [inputs] if single else list(inputs)
        cfg = to_mutable_data(self.spec.args)
        cfg.update(self._request_kwargs(kwargs))
        if embedding_batch_size is not None:
            cfg["embedding_batch_size"] = embedding_batch_size
        if embedding_max_workers is not None:
            cfg["embedding_max_workers"] = embedding_max_workers
        cache_option = cfg.pop("cache", None)
        cache_policy = cfg.pop("cache_policy", None)
        n_batch, n_workers = _embedding_controls(cfg, len(batch))
        cfg.pop("dim", None)
        cache, enabled_option = self._cache_pair(cache_option)
        policy = cache.policy("embedding", cfg, cache_policy)
        active = cache.enabled("embedding", enabled_option) and policy is not None
        unique = list(dict.fromkeys(batch))
        keys = {item: cache.embedding_key(self.spec, item, cfg) for item in unique}
        rows = cache.get_embeddings(keys) if active else {}
        vectors_by_input = {item: row.get("embedding") for item, row in rows.items() if row}
        misses = [item for item in unique if item not in vectors_by_input]
        provider_batches = _batches(misses, n_batch)

        async def run_provider_batch(items: List[str]) -> Dict[str, Any]:
            provider_cfg = deepcopy(cfg)
            with NetworkProxy(
                http_proxy=provider_cfg.pop("http_proxy", None), https_proxy=provider_cfg.pop("https_proxy", None), no_proxy=provider_cfg.pop("no_proxy", None)
            ):
                t0 = perf_counter()
                raw = await _maybe_await(aembed_complete(self.spec, items, **provider_cfg))
                elapsed = round(perf_counter() - t0, 4)
            resp = response_dict(raw)
            data = sorted(resp.get("data", []), key=lambda item: item.get("index", 0))
            if len(data) != len(items):
                raise ValueError(f"Embedding provider returned {len(data)} vectors for {len(items)} inputs")
            return {"inputs": items, "data": data, "usage": _usage(resp), "raw": resp, "elapsed": elapsed}

        t0 = perf_counter()
        task_results = (
            await abatch(run_provider_batch, [{"items": items} for items in provider_batches], max_workers=n_workers, on_error="raise")
            if provider_batches
            else []
        )
        results = [result.value for result in task_results]
        elapsed = round(perf_counter() - t0, 4) if provider_batches else 0.0
        writes = []
        for result in results:
            for input_text, item in zip(result["inputs"], result["data"]):
                vector = item.get("embedding")
                vectors_by_input[input_text] = vector
                if active:
                    writes.append(
                        {
                            "key": keys[input_text],
                            "input_text": input_text,
                            "embedding": vector,
                            "dim": len(vector or []),
                            "usage": result["usage"],
                            "raw": item,
                        }
                    )
        if writes:
            cache.set_embeddings(writes)
        vectors = [vectors_by_input[item] for item in batch]
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        usage = _merge_usage(result["usage"] for result in results)
        raw = {
            "cached": len(unique) - len(misses),
            "misses": len(misses),
            "embedding_batch_size": n_batch,
            "embedding_max_workers": n_workers,
            "batches": [result["raw"] for result in results],
        }
        data = {
            "embeddings": vectors[0] if single and vectors else vectors,
            "usage": usage,
            "raw": raw,
            "elapsed": elapsed,
            "created_at": time(),
            "dim": (len(vectors[0]) if vectors else 0) or self._dim,
        }
        return _project(data, include=include, default=("embeddings",), reduce=reduce)

    def imagen(
        self,
        prompt: str,
        include: Optional[Union[str, Iterable[str]]] = None,
        reduce: bool = True,
        images: ImageInputs = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate images and return include-projected fields.

        Use this method for OpenAI-compatible image generation, OpenRouter chat-completion image
        output, or mock image tests. ``include`` accepts ``None``
        for the default ``images`` value, one field name, or multiple fields. Supported image include
        fields are ``images``, ``usage``, ``raw``, ``elapsed``, and ``created_at``. Unknown names raise
        through ``raise_mismatch``.

        Args:
            prompt (str): Text prompt for image generation.
            include (Optional[Union[str, Iterable[str]]]): Field or fields to return. Defaults to ``None``.
            reduce (bool): Whether a one-field include returns the value directly. Defaults to ``True``.
            images (ImageInputs): Optional reference image input or image inputs. Defaults to ``None``.
            **kwargs (Any): Provider-specific image options such as ``size``, ``n``, or ``modalities``.

        Returns:
            Any: Include-projected image generation result.
        """
        refs = _image_inputs(images, config_manager=self.engine.manager)
        cfg = to_mutable_data(self.spec.args)
        cfg.update(self._request_kwargs(kwargs))
        _drop_embedding_controls(cfg)
        cache_option = cfg.pop("cache", None)
        cache_policy = cfg.pop("cache_policy", None)
        cache, enabled_option = self._cache_pair(cache_option)
        policy = cache.policy("image", cfg, cache_policy)
        active = cache.enabled("image", enabled_option) and policy is not None
        key = cache.image_key(self.spec, prompt, refs, cfg)
        if active:
            row = cache.get_image(key)
            if row:
                cached = cache.load_images(row.get("images") or [])
                data = {
                    "images": cached[0] if len(cached) == 1 else cached,
                    "usage": row.get("usage") or {},
                    "raw": row.get("raw") or {},
                    "elapsed": 0.0,
                    "created_at": row.get("created_at") or time(),
                }
                return _project(data, include=include, default=("images",), reduce=reduce)
        with NetworkProxy(http_proxy=cfg.pop("http_proxy", None), https_proxy=cfg.pop("https_proxy", None), no_proxy=cfg.pop("no_proxy", None)):
            t0 = perf_counter()
            raw = imagen_complete(self.spec, prompt, images=refs, **cfg)
            elapsed = round(perf_counter() - t0, 4)
        resp = response_dict(raw)
        outputs = _images(resp, config_manager=self.engine.manager)
        data = {"images": outputs[0] if len(outputs) == 1 else outputs, "usage": _usage(resp), "raw": resp, "elapsed": elapsed, "created_at": time()}
        result = _project(data, include=include, default=("images",), reduce=reduce)
        if active and outputs:
            cache.set_image(key, prompt, cache.dump_images(outputs), _usage(resp), resp)
        return result

    async def aimagen(
        self,
        prompt: str,
        include: Optional[Union[str, Iterable[str]]] = None,
        reduce: bool = True,
        images: ImageInputs = None,
        **kwargs: Any,
    ) -> Any:
        """Generate images asynchronously and return include-projected fields.

        This is the async counterpart to ``imagen``. It uses the same cache policy
        and include projection.

        Args:
            prompt (str): Text prompt for image generation.
            include (Optional[Union[str, Iterable[str]]]): Field or fields to return. Defaults to ``None``.
            reduce (bool): Whether a one-field include returns the value directly. Defaults to ``True``.
            images (ImageInputs): Optional reference image input or image inputs. Defaults to ``None``.
            **kwargs (Any): Provider-specific image options such as ``size``, ``n``, or ``modalities``.

        Returns:
            Any: Include-projected image generation result.
        """
        refs = _image_inputs(images, config_manager=self.engine.manager)
        cfg = to_mutable_data(self.spec.args)
        cfg.update(self._request_kwargs(kwargs))
        _drop_embedding_controls(cfg)
        cache_option = cfg.pop("cache", None)
        cache_policy = cfg.pop("cache_policy", None)
        cache, enabled_option = self._cache_pair(cache_option)
        policy = cache.policy("image", cfg, cache_policy)
        active = cache.enabled("image", enabled_option) and policy is not None
        key = cache.image_key(self.spec, prompt, refs, cfg)
        if active:
            row = cache.get_image(key)
            if row:
                cached = cache.load_images(row.get("images") or [])
                data = {
                    "images": cached[0] if len(cached) == 1 else cached,
                    "usage": row.get("usage") or {},
                    "raw": row.get("raw") or {},
                    "elapsed": 0.0,
                    "created_at": row.get("created_at") or time(),
                }
                return _project(data, include=include, default=("images",), reduce=reduce)
        with NetworkProxy(http_proxy=cfg.pop("http_proxy", None), https_proxy=cfg.pop("https_proxy", None), no_proxy=cfg.pop("no_proxy", None)):
            t0 = perf_counter()
            raw = await _maybe_await(aimagen_complete(self.spec, prompt, images=refs, **cfg))
            elapsed = round(perf_counter() - t0, 4)
        resp = response_dict(raw)
        outputs = _images(resp, config_manager=self.engine.manager)
        data = {"images": outputs[0] if len(outputs) == 1 else outputs, "usage": _usage(resp), "raw": resp, "elapsed": elapsed, "created_at": time()}
        result = _project(data, include=include, default=("images",), reduce=reduce)
        if active and outputs:
            cache.set_image(key, prompt, cache.dump_images(outputs), _usage(resp), resp)
        return result

    __call__ = chat

    def to_client(self) -> Any:
        """Return a sync ``OpenAI`` client for this resolved spec.

        Only OpenAI-compatible gateways (``openai``, ``portkey``, ``bifrost``) are supported.

        Args:
            None.

        Returns:
            Any: An ``openai.OpenAI`` client configured for the resolved gateway.

        Raises:
            ValueError: If the resolved gateway cannot expose an OpenAI-compatible client.
        """
        try:
            client, _runtime = LLMGateway.load(self.spec.gateway, resolver=self.spec.resolver).client(self.spec, "openai")
        except ValueError as error:
            raise ValueError(f"Gateway '{self.spec.gateway}' cannot be exported for OpenAI-compatible clients") from error
        return client

    def to_aclient(self) -> Any:
        """Return an async ``AsyncOpenAI`` client for this resolved spec.

        Only OpenAI-compatible gateways (``openai``, ``portkey``, ``bifrost``) are supported.
        Inside a running event loop this is the client cached for that loop, closed
        when the loop shuts down its async generators (as ``asyncio.run`` does);
        outside a running loop it is a new client the caller owns and closes.

        Args:
            None.

        Returns:
            Any: An ``openai.AsyncOpenAI`` client configured for the resolved gateway.

        Raises:
            ValueError: If the resolved gateway cannot expose an OpenAI-compatible client.
        """
        try:
            client, _runtime = LLMGateway.load(self.spec.gateway, resolver=self.spec.resolver).client(self.spec, "openai", asynchronous=True)
        except ValueError as error:
            raise ValueError(f"Gateway '{self.spec.gateway}' cannot be exported for OpenAI-compatible clients") from error
        return client

    def to_anthropic_client(self) -> Any:
        """Return a sync ``Anthropic`` client for this resolved spec.

        Use this when the resolved gateway is ``anthropic``, ``portkey``, or
        ``bifrost`` and the caller wants Anthropic's native Messages API rather
        than an OpenAI-compatible client.

        Args:
            None.

        Returns:
            Any: An ``anthropic.Anthropic`` client configured for the resolved gateway endpoint.

        Raises:
            ValueError: If the resolved gateway cannot expose an Anthropic-compatible client.
        """
        try:
            client, _runtime = LLMGateway.load(self.spec.gateway, resolver=self.spec.resolver).client(self.spec, "anthropic")
        except ValueError as error:
            raise ValueError("Anthropic clients require gateway='anthropic', 'portkey', or 'bifrost'.") from error
        return client

    def to_anthropic_aclient(self) -> Any:
        """Return an async ``AsyncAnthropic`` client for this resolved spec.

        Use this when the resolved gateway is ``anthropic``, ``portkey``, or
        ``bifrost`` and the caller wants Anthropic's native Messages API rather
        than an OpenAI-compatible client. Inside a running event loop this is the
        client cached for that loop, closed when the loop shuts down its async
        generators (as ``asyncio.run`` does); outside a running loop it is a new
        client the caller owns and closes.

        Args:
            None.

        Returns:
            Any: An ``anthropic.AsyncAnthropic`` client configured for the resolved gateway endpoint.

        Raises:
            ValueError: If the resolved gateway cannot expose an Anthropic-compatible client.
        """
        try:
            client, _runtime = LLMGateway.load(self.spec.gateway, resolver=self.spec.resolver).client(self.spec, "anthropic", asynchronous=True)
        except ValueError as error:
            raise ValueError("Anthropic clients require gateway='anthropic', 'portkey', or 'bifrost'.") from error
        return client

    def to_args(self, **kwargs: Any) -> Dict[str, Any]:
        """Return provider call args (model, temperature, etc.) as a plain dict.

        This is useful for constructing third-party SDK model settings. The dict
        includes the resolved ``model`` name and all provider-facing args, with
        HeavenBase-internal controls (cache, proxy, mock, thinking toggles, etc.)
        stripped. Caller ``**kwargs`` override spec defaults just like ``LLM.chat``.

        Args:
            **kwargs (Any): Request overrides merged over the resolved spec defaults.

        Returns:
            Dict[str, Any]: Provider-facing call arguments keyed by request field.
        """
        cfg = self._call_kwargs(kwargs)
        _drop_embedding_controls(cfg)
        for key in CLIENT_ARG_KEYS:
            cfg.pop(key, None)
        runtime = self.engine.apply(self.spec)
        cfg = _apply_thinking(runtime, self.spec, cfg)
        cfg.pop("model", None)
        return {"model": runtime["model"], **{k: v for k, v in cfg.items() if v is not None or k in {"seed"}}}
