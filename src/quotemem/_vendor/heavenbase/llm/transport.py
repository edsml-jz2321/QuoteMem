"""Descriptor-driven dispatch for resolved LLM gateway transports."""

from __future__ import annotations

__all__ = [
    "achat_complete",
    "aembed_complete",
    "aimagen_complete",
    "chat_complete",
    "embed_complete",
    "gateway_available",
    "imagen_complete",
    "response_dict",
]

from collections.abc import Iterable
from typing import Any

from .config import LLMSpec
from .gateway import LLMGateway
from .image import LLMImage


def _gateway(spec: LLMSpec) -> LLMGateway:
    """Resolve this spec's gateway through its own Context authority."""
    return LLMGateway.load(spec.gateway, resolver=spec.resolver)


def gateway_available(spec: LLMSpec) -> bool:
    """Return whether the selected descriptor's dependencies are ready.

    Args:
        spec (LLMSpec): Resolved LLM specification whose gateway is probed.

    Returns:
        bool: ``True`` when the selected gateway transport dependencies are ready.
    """
    return _gateway(spec).available(spec)


def chat_complete(spec: LLMSpec, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run chat through the selected descriptor without provider dispatch.

    Args:
        spec (LLMSpec): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages.
        stream (bool): Whether to request a streaming response. Defaults to ``False``.
        **kwargs (Any): Provider request options forwarded to the gateway transport.

    Returns:
        Any: Provider chat response object or stream iterator.
    """
    return _gateway(spec).chat(spec, messages, stream=stream, **kwargs)


async def achat_complete(spec: LLMSpec, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run async chat through the selected descriptor.

    Args:
        spec (LLMSpec): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages.
        stream (bool): Whether to request a streaming response. Defaults to ``False``.
        **kwargs (Any): Provider request options forwarded to the gateway transport.

    Returns:
        Any: Provider chat response object or async stream iterator.
    """
    return await _gateway(spec).achat(spec, messages, stream=stream, **kwargs)


def embed_complete(spec: LLMSpec, inputs: list[str], **kwargs: Any) -> Any:
    """Run embeddings through the selected descriptor.

    Args:
        spec (LLMSpec): Resolved LLM specification.
        inputs (list[str]): Embedding input strings.
        **kwargs (Any): Provider request options forwarded to the gateway transport.

    Returns:
        Any: Provider embedding response object.
    """
    return _gateway(spec).embed(spec, inputs, **kwargs)


async def aembed_complete(spec: LLMSpec, inputs: list[str], **kwargs: Any) -> Any:
    """Run async embeddings through the selected descriptor.

    Args:
        spec (LLMSpec): Resolved LLM specification.
        inputs (list[str]): Embedding input strings.
        **kwargs (Any): Provider request options forwarded to the gateway transport.

    Returns:
        Any: Provider embedding response object.
    """
    return await _gateway(spec).aembed(spec, inputs, **kwargs)


def imagen_complete(spec: LLMSpec, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Any:
    """Run image generation through the selected descriptor.

    Args:
        spec (LLMSpec): Resolved LLM specification.
        prompt (str): Text prompt for image generation.
        images (Iterable[LLMImage]): Optional reference image inputs. Defaults to ``()``.
        **kwargs (Any): Provider request options forwarded to the gateway transport.

    Returns:
        Any: Provider image-generation response object.
    """
    return _gateway(spec).imagen(spec, prompt, images=images, **kwargs)


async def aimagen_complete(spec: LLMSpec, prompt: str, images: Iterable[LLMImage] = (), **kwargs: Any) -> Any:
    """Run async image generation through the selected descriptor.

    Args:
        spec (LLMSpec): Resolved LLM specification.
        prompt (str): Text prompt for image generation.
        images (Iterable[LLMImage]): Optional reference image inputs. Defaults to ``()``.
        **kwargs (Any): Provider request options forwarded to the gateway transport.

    Returns:
        Any: Provider image-generation response object.
    """
    return await _gateway(spec).aimagen(spec, prompt, images=images, **kwargs)


def response_dict(obj: Any) -> dict[str, Any]:
    """Return a provider response as an independent dictionary.

    Args:
        obj (Any): Provider response object or mapping.

    Returns:
        dict[str, Any]: Detached dictionary representation of ``obj``.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return dict(obj)
