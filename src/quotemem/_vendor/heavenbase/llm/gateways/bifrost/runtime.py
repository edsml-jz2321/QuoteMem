"""Compiler and transports for Bifrost's dual endpoint gateway."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ...gateway import _cfg_value, _parts, _prefixed
from ...image import LLMImage
from ...mock import mock_chat_complete, mock_embed_complete, mock_imagen_complete
from .._protocols import (
    BIFROST_IMAGE_ERROR,
    _anthropic_astream,
    _anthropic_async_client,
    _anthropic_client,
    _anthropic_payload,
    _anthropic_response,
    _anthropic_stream,
    _embedding_payload,
    _has_mock_request,
    _openai_async_client,
    _openai_client,
    _openai_payload,
    _run_sync,
)


def compile(spec: Any, gateway: str, override: Mapping[str, Any]) -> dict[str, Any]:
    """Compile Bifrost router arguments.

    Args:
        spec (Any): Resolved LLM specification.
        gateway (str): Gateway identifier under compilation.
        override (Mapping[str, Any]): Per-request runtime argument overrides.

    Returns:
        dict[str, Any]: Bifrost router client arguments including model and credentials.
    """
    part = _parts(spec, gateway, override)
    return {
        "gateway": gateway,
        "model": _prefixed(part["backend"], spec.model_id),
        "api_key": part["arg_api_key"] or part["gateway_key"] or part["provider_key"] or gateway,
        "base_url": part["arg_base_url"] or _cfg_value(part["gateway_cfg"], "base_url"),
        "headers": part["headers"],
        "thinking_control": part["thinking_control"],
        "args": part["args"],
    }


def available(spec: Any) -> bool:
    """Return whether the endpoint SDK dependencies are importable.

    Args:
        spec (Any): Resolved LLM specification whose endpoint selects Anthropic or OpenAI.

    Returns:
        bool: ``True`` when the endpoint SDK can be imported.
    """
    try:
        if spec.endpoint == "anthropic":
            from anthropic import Anthropic  # noqa: F401
        else:
            from openai import OpenAI  # noqa: F401
        return True
    except Exception:
        return False


def _mock(spec: Any, kwargs: Mapping[str, Any]) -> bool:
    return spec.provider_backend() == "mock" or _has_mock_request(dict(kwargs))


def chat(spec: Any, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run chat completion through the Bifrost dual-endpoint transport.

    Args:
        spec (Any): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages.
        stream (bool): Whether to request a streaming response. Defaults to ``False``.
        **kwargs (Any): Provider request options forwarded to the selected SDK.

    Returns:
        Any: Chat completion object or stream iterator from the selected endpoint SDK.
    """
    if _mock(spec, kwargs):
        return mock_chat_complete(spec, messages, stream=stream, **kwargs)
    if spec.endpoint == "anthropic":
        client, runtime = _anthropic_client(spec, spec.gateway)
        result = client.messages.create(**_anthropic_payload(runtime, spec=spec, messages=messages, stream=stream, **kwargs))
        return _anthropic_stream(result) if stream else _anthropic_response(result)
    client, runtime = _openai_client(spec, spec.gateway)
    return client.chat.completions.create(**_openai_payload(runtime, spec=spec, messages=messages, stream=stream, **kwargs))


async def achat(spec: Any, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run async chat completion through the Bifrost dual-endpoint transport.

    Args:
        spec (Any): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages.
        stream (bool): Whether to request a streaming response. Defaults to ``False``.
        **kwargs (Any): Provider request options forwarded to the selected SDK.

    Returns:
        Any: Chat completion object or async stream iterator from the selected endpoint SDK.
    """
    if _mock(spec, kwargs):
        return await _run_sync(mock_chat_complete, spec, messages, stream=stream, **kwargs)
    if spec.endpoint == "anthropic":
        client, runtime = _anthropic_async_client(spec, spec.gateway)
        result = await client.messages.create(**_anthropic_payload(runtime, spec=spec, messages=messages, stream=stream, **kwargs))
        return _anthropic_astream(result) if stream else _anthropic_response(result)
    client, runtime = _openai_async_client(spec, spec.gateway)
    return await client.chat.completions.create(**_openai_payload(runtime, spec=spec, messages=messages, stream=stream, **kwargs))


def embed(spec: Any, inputs: list[str], **kwargs: Any) -> Any:
    """Run embedding creation through the Bifrost OpenAI-compatible transport.

    Args:
        spec (Any): Resolved LLM specification.
        inputs (list[str]): Embedding input strings.
        **kwargs (Any): Provider request options forwarded to the OpenAI SDK.

    Returns:
        Any: Embedding response object from the OpenAI SDK.

    Raises:
        NotImplementedError: If ``spec.endpoint`` is ``anthropic``.
    """
    if _mock(spec, kwargs):
        return mock_embed_complete(spec, inputs, **kwargs)
    if spec.endpoint == "anthropic":
        raise NotImplementedError("Anthropic-native endpoints do not support embeddings")
    client, runtime = _openai_client(spec, spec.gateway)
    return client.embeddings.create(**_embedding_payload(runtime, input=inputs, **kwargs))


async def aembed(spec: Any, inputs: list[str], **kwargs: Any) -> Any:
    """Run async embedding creation through the Bifrost OpenAI-compatible transport.

    Args:
        spec (Any): Resolved LLM specification.
        inputs (list[str]): Embedding input strings.
        **kwargs (Any): Provider request options forwarded to the OpenAI SDK.

    Returns:
        Any: Embedding response object from the async OpenAI SDK.

    Raises:
        NotImplementedError: If ``spec.endpoint`` is ``anthropic``.
    """
    if _mock(spec, kwargs):
        return await _run_sync(mock_embed_complete, spec, inputs, **kwargs)
    if spec.endpoint == "anthropic":
        raise NotImplementedError("Anthropic-native endpoints do not support embeddings")
    client, runtime = _openai_async_client(spec, spec.gateway)
    return await client.embeddings.create(**_embedding_payload(runtime, input=inputs, **kwargs))


def imagen(spec: Any, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Any:
    """Run image generation through the Bifrost transport.

    Args:
        spec (Any): Resolved LLM specification.
        prompt (str): Text prompt for image generation.
        images (Iterable[LLMImage]): Optional reference image inputs. Defaults to ``()``.
        **kwargs (Any): Provider request options forwarded to the transport.

    Returns:
        Any: Image generation response object when supported.

    Raises:
        NotImplementedError: Bifrost does not support image generation outside mock mode.
    """
    if _mock(spec, kwargs):
        return mock_imagen_complete(spec, prompt, images=images, **kwargs)
    raise NotImplementedError(BIFROST_IMAGE_ERROR)


async def aimagen(spec: Any, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Any:
    """Run async image generation through the Bifrost transport.

    Args:
        spec (Any): Resolved LLM specification.
        prompt (str): Text prompt for image generation.
        images (Iterable[LLMImage]): Optional reference image inputs. Defaults to ``()``.
        **kwargs (Any): Provider request options forwarded to the transport.

    Returns:
        Any: Image generation response object when supported.

    Raises:
        NotImplementedError: Bifrost does not support image generation outside mock mode.
    """
    if _mock(spec, kwargs):
        return await _run_sync(mock_imagen_complete, spec, prompt, images=images, **kwargs)
    raise NotImplementedError(BIFROST_IMAGE_ERROR)


def client(spec: Any, protocol: str, *, asynchronous: bool = False) -> tuple[Any, dict[str, Any]]:
    """Return an OpenAI or Anthropic SDK client for one resolved spec.

    Args:
        spec (Any): Resolved LLM specification.
        protocol (str): Client protocol identifier such as ``openai`` or ``anthropic``.
        asynchronous (bool): Whether to return an async client. Defaults to ``False``.

    Returns:
        tuple[Any, dict[str, Any]]: SDK client and compiled runtime arguments.

    Raises:
        ValueError: If ``protocol`` is unsupported by Bifrost.
    """
    if protocol == "openai":
        return (_openai_async_client if asynchronous else _openai_client)(spec, spec.gateway)
    if protocol == "anthropic":
        return (_anthropic_async_client if asynchronous else _anthropic_client)(spec, spec.gateway)
    raise ValueError(f"Bifrost gateway does not expose client protocol {protocol!r}")
