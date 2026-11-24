"""Callable-backed mock provider helpers for LLM tests and offline workflows."""

from __future__ import annotations

__all__ = ["mock_chat_complete", "mock_embed_complete", "mock_imagen_complete"]

import base64
from time import sleep

from heavenbase.utils import StableRNG, dumps_json
from heavenbase.utils.typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

MockCallable = Callable[..., Any]


def _mock_fn(kwargs: Dict[str, Any], name: str) -> Optional[MockCallable]:
    fn = kwargs.get(name) or kwargs.get("mock")
    return fn if callable(fn) else None


def _mock_value(kwargs: Dict[str, Any], name: str) -> Any:
    return kwargs.get(name) if name in kwargs else kwargs.get("mock")


def _mock_script(kwargs: Dict[str, Any], name: str = "mock_chat") -> Dict[str, Any]:
    value = _mock_value(kwargs, name)
    if value is None or callable(value):
        value = {}
    if isinstance(value, str):
        return {"text": value}
    if isinstance(value, dict):
        return dict(value)
    if _is_iterable(value):
        return {"chunks": list(value)}
    return {"text": str(value)}


def _is_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (dict, str, bytes, bytearray))


def _pause_iter(chunks: Iterable[Dict[str, Any]], pause: Any = None, seed: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    bounds = pause if isinstance(pause, (tuple, list)) and len(pause) == 2 else (0.001, 0.005)
    low = max(0.0, float(bounds[0]))
    high = max(low, float(bounds[1]))
    rng = StableRNG(seed=seed, fallback="time")
    for chunk in chunks:
        if high:
            sleep(low + float(rng.rnd()) * (high - low))
        yield chunk


def _chunks(text: str, size: Any) -> List[str]:
    step = max(1, int(size or 16))
    return [text[i : i + step] for i in range(0, len(text), step)] or [""]


def _usage(messages: List[Dict[str, Any]], text: str, usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if usage:
        return dict(usage)
    prompt_tokens = sum(max(1, len(str(message.get("content", "")).split())) for message in messages)
    completion_tokens = max(1, len(text.split()))
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}


def _default_chat(messages: List[Dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    script = _mock_script(kwargs, "mock_chat")
    if script.get("chunks"):
        return script["chunks"]
    text = kwargs.get("mock_text") or script.get("text") or "mock response"
    if kwargs.get("response_format") and script.get("structured") is not None:
        text = dumps_json(script["structured"], indent=None)
    think = script.get("think") or ""
    tool_calls = script.get("tool_calls") or []
    usage = _usage(messages, text, usage=script.get("usage"))
    if not stream:
        message = {"role": "assistant", "content": text}
        if think:
            message["reasoning_content"] = think
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {"choices": [{"message": message}], "usage": usage}
    chunk_size = script.get("chunk_chars", kwargs.get("mock_chunk_chars", 16))
    chunks = [{"choices": [{"delta": {"role": "assistant"}}]}]
    chunks.extend({"choices": [{"delta": {"reasoning_content": part}}]} for part in _chunks(think, chunk_size) if part)
    chunks.extend({"choices": [{"delta": {"content": part}}]} for part in _chunks(text, chunk_size) if part)
    if tool_calls:
        chunks.append({"choices": [{"delta": {"tool_calls": tool_calls}}]})
    chunks.append({"choices": [{"delta": {}}], "usage": usage})
    return chunks


def _chunk_message(chunk: Dict[str, Any]) -> Dict[str, Any]:
    choices = chunk.get("choices") or []
    choice = choices[0] if choices else {}
    return choice.get("message") or choice.get("delta") or {}


def _merge_chunks(chunks: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    text = ""
    think = ""
    usage: Dict[str, Any] = {}
    tool_calls: List[Dict[str, Any]] = []
    raw = []
    for chunk in chunks:
        raw.append(chunk)
        message = _chunk_message(chunk)
        if message.get("content"):
            text += str(message.get("content"))
        if message.get("reasoning_content") or message.get("reasoning") or message.get("thinking"):
            think += str(message.get("reasoning_content") or message.get("reasoning") or message.get("thinking"))
        if message.get("tool_calls"):
            tool_calls.extend(message.get("tool_calls") or [])
        if chunk.get("usage"):
            usage = chunk["usage"]
    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if think:
        message["reasoning_content"] = think
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}], "usage": usage, "raw_chunks": raw}


def mock_chat_complete(spec: Any, messages: List[Dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """
    Run a mock chat completion from a Python callable.

    Use this helper for tests and demos that need OpenAI-format chunks without a remote provider.
    Pass ``mock_chat`` or ``mock`` as a callable accepting keyword arguments such as ``spec``,
    ``messages``, ``stream``, and ``kind``. The callable may return one OpenAI-style response dict
    or an iterable of OpenAI-style chunks. Streaming responses are yielded with a tiny randomized
    pause between chunks; set ``mock_pause=(0, 0)`` for no delay.

    Args:
        spec (Any): Resolved LLM spec.
        messages (List[Dict[str, Any]]): Prepared chat messages.
        stream (bool): Whether to return streaming chunks. Defaults to ``False``.
        **kwargs (Any): Mock callable controls and provider-like request options.

    Returns:
        Any: OpenAI-format response dictionary or iterator of OpenAI-format chunks.
    """
    fn = _mock_fn(kwargs, "mock_chat")
    result = fn(spec=spec, messages=messages, stream=stream, kind="chat", **kwargs) if fn else _default_chat(messages, stream=stream, **kwargs)
    if stream:
        chunks = result if _is_iterable(result) else [result]
        script = _mock_script(kwargs, "mock_chat")
        return _pause_iter(chunks, pause=kwargs.get("mock_pause", script.get("pause")), seed=kwargs.get("mock_seed", script.get("seed")))
    return _merge_chunks(result) if _is_iterable(result) else result


def mock_embed_complete(spec: Any, inputs: List[str], **kwargs: Any) -> Dict[str, Any]:
    """
    Run a mock embedding request from a Python callable.

    Use this helper when embedding tests should cover both single and batch calls without network
    access. Pass ``mock_embed`` or ``mock`` as a callable accepting keyword arguments such as
    ``spec``, ``inputs``, and ``kind``. The callable should return an OpenAI embedding response.

    Args:
        spec (Any): Resolved LLM spec.
        inputs (List[str]): Batch of embedding inputs.
        **kwargs (Any): Mock callable controls and provider-like request options.

    Returns:
        Dict[str, Any]: OpenAI-format embedding response.
    """
    fn = _mock_fn(kwargs, "mock_embed")
    if fn:
        return fn(spec=spec, inputs=inputs, kind="embed", **kwargs)
    script = _mock_script(kwargs, "mock_embed")
    vectors = script.get("embeddings")
    if vectors is not None:
        if inputs and vectors and isinstance(vectors[0], (int, float)):
            vectors = [vectors]
        return {"data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)], "usage": script.get("usage", {})}
    dim = max(1, int(script.get("embed_dim", 3)))
    data = []
    for index, text in enumerate(inputs):
        size = max(1, len(str(text)))
        vector = [float(size), float(index), 1.0]
        while len(vector) < dim:
            vector.append(0.0)
        data.append({"index": index, "embedding": vector[:dim]})
    return {"data": data, "usage": {"prompt_tokens": sum(len(str(item)) for item in inputs), "total_tokens": sum(len(str(item)) for item in inputs)}}


def mock_imagen_complete(spec: Any, prompt: str, images: Iterable[Any] = (), **kwargs: Any) -> Dict[str, Any]:
    """
    Run a mock image generation request from a Python callable.

    Use this helper for image tests that need a valid OpenAI-style ``data`` list. Pass
    ``mock_imagen`` or ``mock`` as a callable accepting keyword arguments such as ``spec``,
    ``prompt``, ``images``, and ``kind``. The callable should return an OpenAI image response.

    Args:
        spec (Any): Resolved LLM spec.
        prompt (str): Image prompt.
        images (Iterable[Any]): Optional reference images. Defaults to ``()``.
        **kwargs (Any): Mock callable controls and provider-like request options.

    Returns:
        Dict[str, Any]: OpenAI-format image response.
    """
    fn = _mock_fn(kwargs, "mock_imagen")
    if fn:
        return fn(spec=spec, prompt=prompt, images=list(images), kind="imagen", **kwargs)
    script = _mock_script(kwargs, "mock_imagen")
    image_text = script.get("image_text") or f"mock image: {prompt}"
    payload = script.get("b64_json") or base64.b64encode(str(image_text).encode("utf-8")).decode("utf-8")
    return {"data": [{"b64_json": payload, "revised_prompt": script.get("revised_prompt", prompt)}], "usage": script.get("usage", {"image_count": 1})}
