"""Compiler and transports for Anthropic-native messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...gateway import _anthropic_url, _cfg_value, _parts
from ...mock import mock_chat_complete
from .._protocols import (
    _anthropic_astream,
    _anthropic_async_client,
    _anthropic_client,
    _anthropic_payload,
    _anthropic_response,
    _anthropic_stream,
    _has_mock_request,
    _run_sync,
)


def compile(spec: Any, gateway: str, override: Mapping[str, Any]) -> dict[str, Any]:
    """Compile Anthropic SDK client arguments.

    Args:
        spec (Any): Resolved LLM specification.
        gateway (str): Gateway identifier under compilation.
        override (Mapping[str, Any]): Per-request runtime argument overrides.

    Returns:
        dict[str, Any]: Anthropic SDK client arguments including model, credentials, and headers.
    """
    part = _parts(spec, gateway, override)
    base = part["arg_base_url"] or _cfg_value(part["provider_cfg"], "base_url", gateway=gateway) or _cfg_value(part["gateway_cfg"], "base_url", gateway=gateway)
    credential = part["arg_auth_token"] or part["arg_api_key"] or part["provider_token"] or part["provider_key"] or part["gateway_token"] or part["gateway_key"]
    out = {
        "gateway": gateway,
        "model": spec.model_id,
        "base_url": _anthropic_url(base),
        "headers": part["headers"],
        "thinking_control": part["thinking_control"],
        "args": part["args"],
    }
    auth = _cfg_value(part["provider_cfg"], "auth_mode", gateway=gateway) or _cfg_value(part["gateway_cfg"], "auth_mode", gateway=gateway)
    if str(auth or "api_key").replace("-", "_").lower() in {"bearer", "auth_token", "token"}:
        if credential:
            out["auth_token"] = credential
    else:
        out["api_key"] = credential or "anthropic"
    return out


def available(spec: Any) -> bool:
    """Return whether the Anthropic SDK is importable.

    Args:
        spec (Any): Resolved LLM specification (unused; kept for gateway parity).

    Returns:
        bool: ``True`` when ``anthropic.Anthropic`` can be imported.
    """
    try:
        from anthropic import Anthropic  # noqa: F401

        return True
    except Exception:
        return False


def chat(spec: Any, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run chat completion through the Anthropic Messages transport.

    Args:
        spec (Any): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages converted for Anthropic.
        stream (bool): Whether to request a streaming response. Defaults to ``False``.
        **kwargs (Any): Provider request options forwarded to the SDK.

    Returns:
        Any: Normalized chat response object or stream iterator.
    """
    if spec.provider_backend() == "mock" or _has_mock_request(kwargs):
        return mock_chat_complete(spec, messages, stream=stream, **kwargs)
    client, runtime = _anthropic_client(spec, spec.gateway)
    result = client.messages.create(**_anthropic_payload(runtime, spec=spec, messages=messages, stream=stream, **kwargs))
    return _anthropic_stream(result) if stream else _anthropic_response(result)


async def achat(spec: Any, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
    """Run async chat completion through the Anthropic Messages transport.

    Args:
        spec (Any): Resolved LLM specification.
        messages (list[dict[str, Any]]): OpenAI-format chat messages converted for Anthropic.
        stream (bool): Whether to request a streaming response. Defaults to ``False``.
        **kwargs (Any): Provider request options forwarded to the SDK.

    Returns:
        Any: Normalized chat response object or async stream iterator.
    """
    if spec.provider_backend() == "mock" or _has_mock_request(kwargs):
        return await _run_sync(mock_chat_complete, spec, messages, stream=stream, **kwargs)
    client, runtime = _anthropic_async_client(spec, spec.gateway)
    result = await client.messages.create(**_anthropic_payload(runtime, spec=spec, messages=messages, stream=stream, **kwargs))
    return _anthropic_astream(result) if stream else _anthropic_response(result)


def client(spec: Any, protocol: str, *, asynchronous: bool = False) -> tuple[Any, dict[str, Any]]:
    """Return an Anthropic SDK client for one resolved spec.

    Args:
        spec (Any): Resolved LLM specification.
        protocol (str): Client protocol identifier. Must be ``anthropic``.
        asynchronous (bool): Whether to return an async client. Defaults to ``False``.

    Returns:
        tuple[Any, dict[str, Any]]: SDK client and compiled runtime arguments.

    Raises:
        ValueError: If ``protocol`` is not ``anthropic``.
    """
    if protocol != "anthropic":
        raise ValueError("Anthropic gateway exposes only the 'anthropic' client protocol")
    return (_anthropic_async_client if asynchronous else _anthropic_client)(spec, spec.gateway)
