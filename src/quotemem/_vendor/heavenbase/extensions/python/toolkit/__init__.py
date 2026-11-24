"""Toolkit surface for the python extension."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = ["python_tool_functions"]

_LAZY_EXPORTS = {
    "python_tool_functions": (".factory", "python_tool_functions"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
