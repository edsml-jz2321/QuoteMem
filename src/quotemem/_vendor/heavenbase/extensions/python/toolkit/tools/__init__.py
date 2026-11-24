"""One module per python MCP tool."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__: list[str] = []

# ``tools.run_python`` imports the tool module on first access, as it was bound
# when the toolkit package imported its factory on import.
__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), {}, __all__)
