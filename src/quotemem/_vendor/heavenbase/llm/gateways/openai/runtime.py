"""Compiler and transports for OpenAI-compatible endpoints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ...gateway import _cfg_value, _parts
from ...image import LLMImage
from ...mock import mock_chat_complete, mock_embed_complete, mock_imagen_complete
from .._protocols import (
    _embedding_payload,
    _has_mock_request,
    _openai_async_client,
    _openai_client,
    _openai_payload,
    _openrouter_aimagen,
    _openrouter_imagen,
    _run_sync,
)


def compile(spec: Any, gateway: str, override: Mapping[str, Any]) -> dict[str, Any]:
    """Compile OpenAI SDK client arguments.

    Args:
        spec (Any): Resolved LLM specification.
        gateway (str): Gateway identifier under compilation.
        override (Mapping[str, Any]): Per-request runtime argument overrides.

    Returns:
        dict[str, Any]: OpenAI SDK client arguments including model, credentials, and headers.
    """
    part = _parts(spec, gateway, override)
    base = part["arg_base_url"] or _cfg_value(part["provider_cfg"], "base_url", gateway=gateway) or _cfg_value(part["gateway_cfg"], "base_url", gateway=gateway)
    return {
        "gateway": gateway,
        "model": spec.model_id,
        "api_key": part["arg_api_key"] or part["provider_key"] or part["gateway_key"] or "openai",
        "base_url": base,
        "headers": part["headers"],
        "thinking_control": part["thinking_control"],
        "args": part["args"],
    }


def available(spec: Any) -> bool:
    """Return whether the OpenAI SDK is importable.

    Args:
        spec (Any): Resolved LLM specification (unused; kept for gateway parity).

    Returns:
        bool: ``True`` when ``openai.OpenAI`` can be imported.
    """
    try:
        from openai import OpenAI  # noqa: F401

        return True
    except Exception:
        return False


def _mock(spec: Any, kwargs: Mapping[str, Any]) -> bool:
    return spec.provider_backend() == "mock" or _has_mock_request(dict(kwargs))


def chat(spec: Any, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run chat completion through the OpenAI SDK transport.

    Args:
        spec (Any): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages.
        stream (bool): Whether to request a streaming response. Defaults to ``False``.
        **kwargs (Any): Provider request options forwarded to the SDK.

    Returns:
        Any: Chat completion object or stream iterator.
    """
    if _mock(spec, kwargs):
        return mock_chat_complete(spec, messages, stream=stream, **kwargs)
    client, runtime = _openai_client(spec, spec.gateway)
    return client.chat.completions.create(**_openai_payload(runtime, spec=spec, messages=messages, stream=stream, **kwargs))


async def achat(spec: Any, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run async chat completion through the OpenAI SDK transport.

    Args:
        spec (Any): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages.
        stream (bool): Whether to request a streaming response. Defaults to ``False``.
        **kwargs (Any): Provider request options forwarded to the SDK.

    Returns:
        Any: Chat completion object or async stream iterator.
    """
    if _mock(spec, kwargs):
        return await _run_sync(mock_chat_complete, spec, messages, stream=stream, **kwargs)
    client, runtime = _openai_async_client(spec, spec.gateway)
    return await client.chat.completions.create(**_openai_payload(runtime, spec=spec, messages=messages, stream=stream, **kwargs))


def embed(spec: Any, inputs: list[str], **kwargs: Any) -> Any:
    """Run embedding creation through the OpenAI SDK transport.

    Args:
        spec (Any): Resolved LLM specification.
        inputs (list[str]): Embedding input strings.
        **kwargs (Any): Provider request options forwarded to the SDK.

    Returns:
        Any: Embedding response object from the SDK.
    """
    if _mock(spec, kwargs):
        return mock_embed_complete(spec, inputs, **kwargs)
    client, runtime = _openai_client(spec, spec.gateway)
    return client.embeddings.create(**_embedding_payload(runtime, input=inputs, **kwargs))


async def aembed(spec: Any, inputs: list[str], **kwargs: Any) -> Any:
    """Run async embedding creation through the OpenAI SDK transport.

    Args:
        spec (Any): Resolved LLM specification.
        inputs (list[str]): Embedding input strings.
        **kwargs (Any): Provider request options forwarded to the SDK.

    Returns:
        Any: Embedding response object from the async SDK.
    """
    if _mock(spec, kwargs):
        return await _run_sync(mock_embed_complete, spec, inputs, **kwargs)
    client, runtime = _openai_async_client(spec, spec.gateway)
    return await client.embeddings.create(**_embedding_payload(runtime, input=inputs, **kwargs))


def imagen(spec: Any, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Any:
    """Run image generation through the OpenAI or OpenRouter transport.

    Args:
        spec (Any): Resolved LLM specification.
        prompt (str): Text prompt for image generation.
        images (Iterable[LLMImage]): Optional reference image inputs. Defaults to ``()``.
        **kwargs (Any): Provider request options forwarded to the SDK.

    Returns:
        Any: Image generation response object.
    """
    if _mock(spec, kwargs):
        return mock_imagen_complete(spec, prompt, images=images, **kwargs)
    if spec.provider_backend() == "openrouter":
        return _openrouter_imagen(spec, prompt, images=images, **kwargs)
    client, runtime = _openai_client(spec, spec.gateway)
    payload = _openai_payload(runtime, spec=spec, prompt=prompt, **kwargs)
    if images:
        payload["extra_body"] = {"images": [item.to_dict() for item in images]}
    return client.images.generate(**payload)


async def aimagen(spec: Any, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Any:
    """Run async image generation through the OpenAI or OpenRouter transport.

    Args:
        spec (Any): Resolved LLM specification.
        prompt (str): Text prompt for image generation.
        images (Iterable[LLMImage]): Optional reference image inputs. Defaults to ``()``.
        **kwargs (Any): Provider request options forwarded to the SDK.

    Returns:
        Any: Image generation response object.
    """
    if _mock(spec, kwargs):
        return await _run_sync(mock_imagen_complete, spec, prompt, images=images, **kwargs)
    if spec.provider_backend() == "openrouter":
        return await _openrouter_aimagen(spec, prompt, images=images, **kwargs)
    client, runtime = _openai_async_client(spec, spec.gateway)
    payload = _openai_payload(runtime, spec=spec, prompt=prompt, **kwargs)
    if images:
        payload["extra_body"] = {"images": [item.to_dict() for item in images]}
    return await client.images.generate(**payload)


def client(spec: Any, protocol: str, *, asynchronous: bool = False) -> tuple[Any, dict[str, Any]]:
    """Return an OpenAI SDK client for one resolved spec.

    Args:
        spec (Any): Resolved LLM specification.
        protocol (str): Client protocol identifier. Must be ``openai``.
        asynchronous (bool): Whether to return an async client. Defaults to ``False``.

    Returns:
        tuple[Any, dict[str, Any]]: SDK client and compiled runtime arguments.

    Raises:
        ValueError: If ``protocol`` is not ``openai``.
    """
    if protocol != "openai":
        raise ValueError("OpenAI gateway exposes only the 'openai' client protocol")
    return (_openai_async_client if asynchronous else _openai_client)(spec, spec.gateway)
