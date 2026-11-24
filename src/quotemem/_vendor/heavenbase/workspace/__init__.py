"""Public workspace composition facade."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["EntityOptions", "GraphDeleteRestrictedError", "HeavenBase", "WorkspaceManifest", "WorkspaceSkill", "WorkspaceSpec"]

_LAZY_EXPORTS = {
    "EntityOptions": (".schema.options", "EntityOptions"),
    "GraphDeleteRestrictedError": (".graph.errors", "GraphDeleteRestrictedError"),
    "HeavenBase": (".runtime", "HeavenBase"),
    "WorkspaceManifest": (".manifest", "WorkspaceManifest"),
    "WorkspaceSkill": (".mcp", "WorkspaceSkill"),
    "WorkspaceSpec": (".registry", "WorkspaceSpec"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
