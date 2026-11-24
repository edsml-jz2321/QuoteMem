"""Copy-on-write Registry with injectable durable stores."""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.util import find_spec
from pkgutil import iter_modules
from typing import Any

__all__ = [
    "Registry",
    "RegistryConflictError",
    "RegistryIntegrityError",
    "RegistryStore",
    "RegistryValuePolicy",
    "RegistryView",
    "SQLiteRegistryStore",
    "class_identifier",
    "identifier_key",
    "named_key",
    "python_target",
    "restore_python_target",
]

_LAZY_EXPORTS = {
    "RegistryConflictError": (".errors", "RegistryConflictError"),
    "RegistryIntegrityError": (".errors", "RegistryIntegrityError"),
    "class_identifier": (".identity", "class_identifier"),
    "identifier_key": (".identity", "identifier_key"),
    "named_key": (".identity", "named_key"),
    "Registry": (".registry", "Registry"),
    "RegistryValuePolicy": (".registry", "RegistryValuePolicy"),
    "RegistryView": (".registry", "RegistryView"),
    "RegistryStore": (".store", "RegistryStore"),
    "SQLiteRegistryStore": (".store", "SQLiteRegistryStore"),
    "python_target": (".targets", "python_target"),
    "restore_python_target": (".targets", "restore_python_target"),
}


# The Registry kernel imports only the standard library, so it can run as a
# standalone package; this facade cannot use the shared ``heavenbase._lazy`` hook.
# It mirrors that hook: declared exports first, then submodules on attribute access.
def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        return _load_submodule(name)
    module_name, attr_name = target
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def _load_submodule(name: str) -> Any:
    missing = AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name.startswith("__") or not name.isidentifier():
        raise missing
    qualified = f"{__name__}.{name}"
    try:
        if qualified not in sys.modules and find_spec(qualified) is None:
            raise missing
        return import_module(qualified)
    except ImportError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r} ({exc})") from exc


def __dir__() -> list[str]:
    submodules = {info.name for info in iter_modules(__path__) if not info.name.startswith("__")}
    return sorted(set(globals()).union(__all__, submodules))
