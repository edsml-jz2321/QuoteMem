"""One small call to each configured provider, with a fix-it hint on failure.

The service probes before it opens the store, because opening runs the embedding
canary and fails without a reachable provider. Embedding goes first: without it
neither Add nor Search works. The chat probe asks for a one-field JSON object in
the configured structured-output mode.

A reply whose model the routing did not admit fails the probe with
``model_mismatch``, naming the reported model, even when that name happens to be
one the core would accept: a gateway that reports a dated snapshot such as
``deepseek-v4-1-flash-261017`` for a configured ``deepseek-v4.1-flash`` fails the
probe, where a check that only counted the mismatch would still say ok.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

import httpx

from quotemem.agent.routing import ProviderRoutes, Role
from quotemem.agent.settings import CANONICAL_CHAT_BASE_URL, CANONICAL_EMBEDDING_BASE_URL
from quotemem.core.model_identity import DEEPSEEK_FLASH_MODEL, current_inference_model
from quotemem.providers.config import EMBEDDING_MODEL

PROBE_TIMEOUT_SECONDS = 30.0
PROBE_CHAT_READ_SECONDS = 60.0
"""The chat probe's read wait: under ``QUOTEMEM_CHAT_REASONING=high`` it leaves a reasoning
attempt 30 s and its resend without thinking 30 s (``quotemem.agent.routing``), so the probe
exercises the thinking parameters as every call sends them."""
_PROBE_SCHEMA = {
    "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ProbeResult:
    """One probe's outcome; ``code`` and ``hint`` are content-free."""

    ok: bool
    role: Role | None = None
    code: str | None = None
    hint: str | None = None


def _status_hint(role: Role, status: int, structured: str) -> tuple[str, str]:
    prefix = "QUOTEMEM_CHAT" if role == "chat" else "QUOTEMEM_EMBED"
    if status in {401, 403}:
        return f"http_{status}", f"The {role} provider refused the key; check {prefix}_API_KEY"
    if status == 404:
        return "http_404", f"Check {prefix}_BASE_URL (it ends before /chat/completions or " \
            f"/embeddings) and {prefix}_MODEL"
    if status == 429:
        return "http_429", f"The {role} provider is rate limiting or out of quota"
    if status == 400 and role == "chat" and structured == "json_schema":
        return "http_400", "The chat provider rejected the request; if it lacks strict JSON " \
            "schemas set QUOTEMEM_CHAT_STRUCTURED_OUTPUT=json_object"
    if status == 400:
        return "http_400", f"The {role} provider rejected the request; check {prefix}_MODEL"
    return f"http_{status}", f"The {role} provider is unavailable"


_MODEL_SHOWN = 80
"""Characters of a reported model name a hint shows."""


def _mismatch(role: Role, routes: ProviderRoutes, reported: object) -> ProbeResult:
    prefix = "QUOTEMEM_CHAT" if role == "chat" else "QUOTEMEM_EMBED"
    configured = routes.chat.model if role == "chat" else routes.embedding.model
    shown = reported[:_MODEL_SHOWN] if isinstance(reported, str) and reported else "no model"
    return ProbeResult(
        False, role, "model_mismatch",
        f"The {role} reply names {shown!r}, not the configured {configured!r}; set "
        f"{prefix}_MODEL to the served model, or {prefix}_MODEL_PATTERN to a pattern "
        "that matches both",
    )


def probe_embedding(client: httpx.Client, routes: ProviderRoutes) -> ProbeResult:
    """Embed one text and check the vector fits the store."""
    mismatches = routes.health.count("embedding", "model_mismatch")
    try:
        response = client.post(CANONICAL_EMBEDDING_BASE_URL + "/embeddings", json={
            "model": EMBEDDING_MODEL, "input": ["QuoteMem provider probe"],
            "dimensions": routes.dimensions, "encoding_format": "float",
        }, timeout=PROBE_TIMEOUT_SECONDS)
    except httpx.HTTPError as error:
        return ProbeResult(False, "embedding", type(error).__name__,
                           "The embedding provider is unreachable; check QUOTEMEM_EMBED_BASE_URL")
    if not response.is_success:
        code, hint = _status_hint("embedding", response.status_code, "")
        return ProbeResult(False, "embedding", code, hint)
    try:
        raw = cast(dict[str, Any], response.json())
        vector = cast(list[Any], cast(list[Any], raw["data"])[0]["embedding"])
    except (ValueError, KeyError, IndexError, TypeError):
        return ProbeResult(False, "embedding", "reply_shape",
                           "The embedding reply is not an OpenAI-compatible embeddings object")
    if routes.health.count("embedding", "model_mismatch") > mismatches or (
        raw.get("model") != EMBEDDING_MODEL
    ):
        return _mismatch("embedding", routes, raw.get("model"))
    if len(vector) != routes.dimensions:
        return ProbeResult(False, "embedding", "dimension_mismatch",
                           f"The model returns more than {routes.dimensions} dimensions; set "
                           "QUOTEMEM_EMBED_SEND_DIMENSIONS=true if it accepts a dimensions "
                           "parameter, or choose a smaller model")
    return ProbeResult(True, "embedding")


def probe_chat(client: httpx.Client, routes: ProviderRoutes) -> ProbeResult:
    """Ask for ``{"ok": true}`` in the configured structured-output mode."""
    structured = routes.chat.structured_output
    mismatches = routes.health.count("chat", "model_mismatch")
    try:
        response = client.post(CANONICAL_CHAT_BASE_URL + "/chat/completions", json={
            "model": DEEPSEEK_FLASH_MODEL, "stream": False, "max_tokens": 256,
            "messages": [
                {"role": "system", "content": 'Reply with the JSON object {"ok": true}.'},
                {"role": "user", "content": "ping"},
            ],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "probe", "strict": True, "schema": _PROBE_SCHEMA,
            }},
        }, timeout=httpx.Timeout(PROBE_TIMEOUT_SECONDS, read=PROBE_CHAT_READ_SECONDS))
    except httpx.HTTPError as error:
        return ProbeResult(False, "chat", type(error).__name__,
                           "The chat provider is unreachable; check QUOTEMEM_CHAT_BASE_URL")
    if not response.is_success:
        code, hint = _status_hint("chat", response.status_code, structured)
        return ProbeResult(False, "chat", code, hint)
    try:
        raw = cast(dict[str, Any], response.json())
        content = cast(str, cast(list[Any], raw["choices"])[0]["message"]["content"])
        json.loads(content)
    except (ValueError, KeyError, IndexError, TypeError):
        return ProbeResult(False, "chat", "reply_shape",
                           "The chat reply is not one JSON object; try "
                           "QUOTEMEM_CHAT_STRUCTURED_OUTPUT=json_object")
    if routes.health.count("chat", "model_mismatch") > mismatches or (
        raw.get("model") != current_inference_model("openai", DEEPSEEK_FLASH_MODEL)
    ):
        # Routing reports an admitted reply under the core's identity and keeps a
        # refused reply's own name, which can equal that identity by coincidence.
        return _mismatch("chat", routes, raw.get("model"))
    return ProbeResult(True, "chat")


def probe_providers(client: httpx.Client, routes: ProviderRoutes) -> ProbeResult:
    """Probe embedding, then chat; the first failure is returned."""
    embedded = probe_embedding(client, routes)
    if not embedded.ok:
        return embedded
    chatted = probe_chat(client, routes)
    return chatted if not chatted.ok else ProbeResult(True)
