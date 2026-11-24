"""Neo4j graph backend provider (entity/row store + native vector)."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

_PROVIDER_EXPORTS = {
    "Neo4jBackend": (".backend", "Neo4jBackend"),
}

__all__ = [*_PROVIDER_EXPORTS]


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _PROVIDER_EXPORTS, __all__)
