"""SurrealDB multi-model backend package (driver imported lazily by the backend)."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

_PROVIDER_EXPORTS = {
    "SurrealBackend": (".backend", "SurrealBackend"),
}

__all__ = [*_PROVIDER_EXPORTS]


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _PROVIDER_EXPORTS, __all__)
