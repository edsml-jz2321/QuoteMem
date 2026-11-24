"""Vector backend providers."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

_PROVIDER_EXPORTS = {
    "ChromaBackend": (".chroma", "ChromaBackend"),
    "LanceBackend": (".lance", "LanceBackend"),
    "MilvusBackend": (".milvus", "MilvusBackend"),
    "MilvusLiteBackend": (".milvus", "MilvusLiteBackend"),
    "PgVectorBackend": (".pgvector", "PgVectorBackend"),
    "PineconeBackend": (".pinecone", "PineconeBackend"),
    "QdrantBackend": (".qdrant", "QdrantBackend"),
    "RediSearchBackend": (".redisearch", "RediSearchBackend"),
    "WeaviateBackend": (".weaviate", "WeaviateBackend"),
}

__all__ = [*_PROVIDER_EXPORTS]


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _PROVIDER_EXPORTS, __all__)
