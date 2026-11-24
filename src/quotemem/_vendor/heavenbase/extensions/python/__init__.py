"""Optional in-process Python execution extension."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports
from heavenbase.extensions.ids import PYTHON_EXTENSION_ID

__all__ = [
    "PYTHON_EXTENSION_ID",
    "PYTHON_RESULT_CHARS",
    "python_namespace",
    "render_python_guide",
    "run_python_code",
]

_LAZY_EXPORTS = {
    "PYTHON_RESULT_CHARS": (".runner", "PYTHON_RESULT_CHARS"),
    "python_namespace": (".namespace", "python_namespace"),
    "render_python_guide": (".guide", "render_python_guide"),
    "run_python_code": (".runner", "run_python_code"),
}


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
