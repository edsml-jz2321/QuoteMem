"""Compiler and transports for the LiteLLM gateway."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ...gateway import _cfg_value, _parts, _prefixed
from ...image import LLMImage
from ...mock import mock_chat_complete, mock_embed_complete, mock_imagen_complete
from .._protocols import (
    _add_litellm_client_args,
    _async_stream,
    _embedding_payload,
    _has_mock_request,
    _litellm,
    _litellm_claude_thinking,
    _openai_payload,
    _openrouter_aimagen_litellm,
    _openrouter_imagen_litellm,
    _run_sync,
)


def compile(spec: Any, gateway: str, override: Mapping[str, Any]) -> dict[str, Any]:
    """Compile LiteLLM provider-prefixed arguments.

    Args:
        spec (Any): Resolved LLM specification.
        gateway (str): Gateway identifier under compilation.
        override (Mapping[str, Any]): Per-request runtime argument overrides.

    Returns:
        dict[str, Any]: LiteLLM runtime arguments including provider-prefixed model name.
    """
    part = _parts(spec, gateway, override)
    out = {
        "gateway": gateway,
        "model": _prefixed(part["backend"], spec.model_id),
        "args": part["args"],
        "headers": part["headers"],
        "thinking_control": part["thinking_control"],
    }
    key = part["arg_api_key"] or part["provider_key"]
    base = part["arg_base_url"] or _cfg_value(part["provider_cfg"], "base_url", gateway=gateway)
    if key:
        out["api_key"] = key
    if base:
        out["api_base"] = base
    return out


def available(spec: Any) -> bool:
    """Return whether LiteLLM is importable for this spec.

    Args:
        spec (Any): Resolved LLM specification used to locate LiteLLM.

    Returns:
        bool: ``True`` when LiteLLM can be imported for the resolved route.
    """
    try:
        _litellm(spec)
        return True
    except Exception:
        return False


def _mock(spec: Any, kwargs: Mapping[str, Any]) -> bool:
    return spec.provider_backend() == "mock" or _has_mock_request(dict(kwargs))


def _payload(spec: Any, **kwargs: Any) -> dict[str, Any]:
    runtime = spec.engine.apply(spec)
    payload = _openai_payload(runtime, spec=spec, **kwargs)
    if runtime.get("thinking_control") == "native_thinking":
        payload = _litellm_claude_thinking(payload)
    if runtime.get("api_key"):
        payload["api_key"] = runtime["api_key"]
    if runtime.get("api_base"):
        payload["api_base"] = runtime["api_base"]
    return _add_litellm_client_args(payload, runtime)


def chat(spec: Any, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run chat completion through the LiteLLM transport.

    Args:
        spec (Any): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages.
        stream (bool): Whether to request a streaming response. Defaults to ``False``.
        **kwargs (Any): Provider request options forwarded to LiteLLM.

    Returns:
        Any: LiteLLM chat completion object or stream iterator.
    """
    if _mock(spec, kwargs):
        return mock_chat_complete(spec, messages, stream=stream, **kwargs)
    return _litellm(spec).completion(**_payload(spec, messages=messages, stream=stream, **kwargs))


async def achat(spec: Any, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run async chat completion through the LiteLLM transport.

    Args:
        spec (Any): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages.
        stream (bool): Whether to request a streaming response. Defaults to ``False``.
        **kwargs (Any): Provider request options forwarded to LiteLLM.

    Returns:
        Any: LiteLLM chat completion object or async stream iterator.
    """
    if _mock(spec, kwargs):
        result = await _run_sync(mock_chat_complete, spec, messages, stream=stream, **kwargs)
        return _async_stream(result) if stream and not hasattr(result, "__aiter__") else result
    module = _litellm(spec)
    payload = _payload(spec, messages=messages, stream=stream, **kwargs)
    fn = getattr(module, "acompletion", None)
    if fn:
        return await fn(**payload)
    result = await _run_sync(module.completion, **payload)
    return _async_stream(result) if stream and not hasattr(result, "__aiter__") else result


def embed(spec: Any, inputs: list[str], **kwargs: Any) -> Any:
    """Run embedding creation through the LiteLLM transport.

    Args:
        spec (Any): Resolved LLM specification.
        inputs (list[str]): Embedding input strings.
        **kwargs (Any): Provider request options forwarded to LiteLLM.

    Returns:
        Any: LiteLLM embedding response object.
    """
    if _mock(spec, kwargs):
        return mock_embed_complete(spec, inputs, **kwargs)
    runtime = spec.engine.apply(spec)
    payload = _embedding_payload(runtime, input=inputs, **kwargs)
    if runtime.get("api_key"):
        payload["api_key"] = runtime["api_key"]
    if runtime.get("api_base"):
        payload["api_base"] = runtime["api_base"]
    return _litellm(spec).embedding(**_add_litellm_client_args(payload, runtime))


async def aembed(spec: Any, inputs: list[str], **kwargs: Any) -> Any:
    """Run async embedding creation through the LiteLLM transport.

    Args:
        spec (Any): Resolved LLM specification.
        inputs (list[str]): Embedding input strings.
        **kwargs (Any): Provider request options forwarded to LiteLLM.

    Returns:
        Any: LiteLLM embedding response object.
    """
    if _mock(spec, kwargs):
        return await _run_sync(mock_embed_complete, spec, inputs, **kwargs)
    runtime = spec.engine.apply(spec)
    payload = _embedding_payload(runtime, input=inputs, **kwargs)
    if runtime.get("api_key"):
        payload["api_key"] = runtime["api_key"]
    if runtime.get("api_base"):
        payload["api_base"] = runtime["api_base"]
    payload = _add_litellm_client_args(payload, runtime)
    module = _litellm(spec)
    fn = getattr(module, "aembedding", None)
    return await fn(**payload) if fn else await _run_sync(module.embedding, **payload)


def imagen(spec: Any, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Any:
    """Run image generation through the LiteLLM transport.

    Args:
        spec (Any): Resolved LLM specification.
        prompt (str): Text prompt for image generation.
        images (Iterable[LLMImage]): Optional reference image inputs. Defaults to ``()``.
        **kwargs (Any): Provider request options forwarded to LiteLLM.

    Returns:
        Any: LiteLLM image generation response object.
    """
    if _mock(spec, kwargs):
        return mock_imagen_complete(spec, prompt, images=images, **kwargs)
    if spec.provider_backend() == "openrouter":
        return _openrouter_imagen_litellm(spec, prompt, images=images, **kwargs)
    return _litellm(spec).image_generation(**_payload(spec, prompt=prompt, **kwargs))


async def aimagen(spec: Any, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Any:
    """Run async image generation through the LiteLLM transport.

    Args:
        spec (Any): Resolved LLM specification.
        prompt (str): Text prompt for image generation.
        images (Iterable[LLMImage]): Optional reference image inputs. Defaults to ``()``.
        **kwargs (Any): Provider request options forwarded to LiteLLM.

    Returns:
        Any: LiteLLM image generation response object.
    """
    if _mock(spec, kwargs):
        return await _run_sync(mock_imagen_complete, spec, prompt, images=images, **kwargs)
    if spec.provider_backend() == "openrouter":
        return await _openrouter_aimagen_litellm(spec, prompt, images=images, **kwargs)
    module = _litellm(spec)
    payload = _payload(spec, prompt=prompt, **kwargs)
    fn = getattr(module, "aimage_generation", None)
    return await fn(**payload) if fn else await _run_sync(module.image_generation, **payload)
