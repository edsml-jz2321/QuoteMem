"""Machine bootstrap and process-local service composition."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

# DEFAULT_CONTEXT stays an attribute but is not listed in __all__: a star import must not build (or, in explicit Context
# mode, refuse to build) the process default Context.
__all__ = [
    "BootstrapSpec",
    "Context",
    "configure_explicit_context",
    "context_or_default",
    "default_context",
    "load_bootstrap",
    "set_default_context",
]

_LAZY_EXPORTS = {
    "BootstrapSpec": (".bootstrap", "BootstrapSpec"),
    "Context": (".runtime", "Context"),
    "configure_explicit_context": ("heavenbase._initialization", "configure_explicit_context"),
    "DEFAULT_CONTEXT": (".runtime", "DEFAULT_CONTEXT"),
    "context_or_default": (".runtime", "context_or_default"),
    "default_context": (".runtime", "default_context"),
    "load_bootstrap": (".bootstrap", "load_bootstrap"),
    "set_default_context": (".runtime", "set_default_context"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
