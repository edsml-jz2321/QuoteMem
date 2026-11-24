"""Explicit composition of the small supported provider set; no provider registry."""

from dataclasses import dataclass

import httpx

from quotemem.providers.adapters import (
    ChatCompletionsInference,
    JsonTransport,
    ModelStudioEmbedding,
    extract_request_effort,
)
from quotemem.providers.config import ProviderConfig, ProviderCredentials
from quotemem.providers.contracts import EmbeddingProvider, StructuredInferenceProvider
from quotemem.providers.deadline_transport import AsyncClientFactory


@dataclass(frozen=True)
class Providers:
    """Provider instances sharing a caller-owned HTTP client lifecycle."""

    embedding: EmbeddingProvider
    inference: StructuredInferenceProvider


def create_providers(
    config: ProviderConfig, credentials: ProviderCredentials, client: httpx.Client,
    *, async_client_factory: AsyncClientFactory | None = None,
) -> Providers:
    """Compose selected adapters after explicit credential validation."""
    config.validate_inference_endpoint()
    studio = JsonTransport(
        client, config.model_studio_base_url, credentials.model_studio,
        async_client_factory=async_client_factory,
    )
    gateway = JsonTransport(
        client, config.openai_base_url, credentials.openai or "",
        async_client_factory=async_client_factory,
    )
    return Providers(
        embedding=ModelStudioEmbedding(studio),
        inference=ChatCompletionsInference(
            gateway, config.inference_provider, extract_effort=extract_request_effort(),
        ),
    )
