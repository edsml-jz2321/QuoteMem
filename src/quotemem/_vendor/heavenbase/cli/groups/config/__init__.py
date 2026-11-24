"""Config CLI command group."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["config_group"]

_LAZY_EXPORTS = {
    "config_group": (".group", "config_group"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
