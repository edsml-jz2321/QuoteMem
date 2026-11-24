"""User-facing LLM runtime, sessions, and HeavenBase-backed caching."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

_LAZY_EXPORTS = {
    "LLM": (".base", "LLM"),
    "LLMCache": (".cache", "LLMCache"),
    "LLMEmbeddingEntry": (".cache", "LLMEmbeddingEntry"),
    "LLMEngine": (".config", "LLMEngine"),
    "LLMGateway": (".gateway", "LLMGateway"),
    "LLMImage": (".image", "LLMImage"),
    "LLMImageEntry": (".cache", "LLMImageEntry"),
    "LLMResponse": (".response", "LLMResponse"),
    "LLMSession": (".session", "LLMSession"),
    "LLMSpec": (".config", "LLMSpec"),
    "LLMTextEntry": (".cache", "LLMTextEntry"),
    "LLMToolCallRepair": (".repair", "LLMToolCallRepair"),
    "LLMTraceEvent": (".trace", "LLMTraceEvent"),
    "TraceCallback": (".trace", "TraceCallback"),
    "format_messages": (".base", "format_messages"),
    "mock_chat_complete": (".mock", "mock_chat_complete"),
    "mock_embed_complete": (".mock", "mock_embed_complete"),
    "mock_imagen_complete": (".mock", "mock_imagen_complete"),
}

__all__ = [*_LAZY_EXPORTS]


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
