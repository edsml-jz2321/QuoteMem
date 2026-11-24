"""Compiler and transports for the deterministic mock gateway."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ...gateway import _parts
from ...image import LLMImage
from ...mock import mock_chat_complete, mock_embed_complete, mock_imagen_complete
from .._protocols import _async_stream, _run_sync


def compile(spec: Any, gateway: str, override: Mapping[str, Any]) -> dict[str, Any]:
    """Compile a deterministic mock runtime.

    Args:
        spec (Any): Resolved LLM specification.
        gateway (str): Gateway identifier under compilation.
        override (Mapping[str, Any]): Per-request runtime argument overrides.

    Returns:
        dict[str, Any]: Mock runtime arguments including model and semantic args.
    """
    part = _parts(spec, gateway, override)
    return {
        "gateway": gateway,
        "model": spec.model_id,
        "args": part["args"],
        "headers": part["headers"],
        "thinking_control": part["thinking_control"],
    }


def available(spec: Any) -> bool:
    """Return whether the mock gateway is ready.

    Args:
        spec (Any): Resolved LLM specification (unused; kept for gateway parity).

    Returns:
        bool: Always ``True`` because the mock gateway has no external dependencies.
    """
    return True


def chat(spec: Any, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run deterministic mock chat completion.

    Args:
        spec (Any): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages.
        stream (bool): Whether to return a streaming mock response. Defaults to ``False``.
        **kwargs (Any): Mock request options such as ``mock_seed`` or ``mock_pause``.

    Returns:
        Any: Mock chat response object or stream iterator.
    """
    return mock_chat_complete(spec, messages, stream=stream, **kwargs)


async def achat(spec: Any, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run deterministic async mock chat completion.

    Args:
        spec (Any): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages.
        stream (bool): Whether to return a streaming mock response. Defaults to ``False``.
        **kwargs (Any): Mock request options such as ``mock_seed`` or ``mock_pause``.

    Returns:
        Any: Mock chat response object or async stream iterator.
    """
    result = await _run_sync(mock_chat_complete, spec, messages, stream=stream, **kwargs)
    return _async_stream(result) if stream and not hasattr(result, "__aiter__") else result


def embed(spec: Any, inputs: list[str], **kwargs: Any) -> Any:
    """Run deterministic mock embedding creation.

    Args:
        spec (Any): Resolved LLM specification.
        inputs (list[str]): Embedding input strings.
        **kwargs (Any): Mock request options forwarded to the mock embed helper.

    Returns:
        Any: Mock embedding response object.
    """
    return mock_embed_complete(spec, inputs, **kwargs)


async def aembed(spec: Any, inputs: list[str], **kwargs: Any) -> Any:
    """Run deterministic async mock embedding creation.

    Args:
        spec (Any): Resolved LLM specification.
        inputs (list[str]): Embedding input strings.
        **kwargs (Any): Mock request options forwarded to the mock embed helper.

    Returns:
        Any: Mock embedding response object.
    """
    return await _run_sync(mock_embed_complete, spec, inputs, **kwargs)


def imagen(spec: Any, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Any:
    """Run deterministic mock image generation.

    Args:
        spec (Any): Resolved LLM specification.
        prompt (str): Text prompt for image generation.
        images (Iterable[LLMImage]): Optional reference image inputs. Defaults to ``()``.
        **kwargs (Any): Mock request options forwarded to the mock image helper.

    Returns:
        Any: Mock image generation response object.
    """
    return mock_imagen_complete(spec, prompt, images=images, **kwargs)


async def aimagen(spec: Any, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Any:
    """Run deterministic async mock image generation.

    Args:
        spec (Any): Resolved LLM specification.
        prompt (str): Text prompt for image generation.
        images (Iterable[LLMImage]): Optional reference image inputs. Defaults to ``()``.
        **kwargs (Any): Mock request options forwarded to the mock image helper.

    Returns:
        Any: Mock image generation response object.
    """
    return await _run_sync(mock_imagen_complete, spec, prompt, images=images, **kwargs)
