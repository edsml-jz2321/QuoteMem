"""Module-folder installation, inspection, and resolution lifecycle."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "EXECUTABLE_REF_KIND",
    "EXECUTABLE_REF_VERSION",
    "ExecutableRef",
    "InstallReceipt",
    "ModuleCatalog",
    "ModuleResolver",
    "ModuleService",
    "ResolvedTagValue",
    "TagDefinition",
    "TagView",
    "capsule_executable_ref",
    "module_resolver",
    "python_executable_ref",
    "tag",
    "tag_definition",
    "tags",
]

_LAZY_EXPORTS = {
    "EXECUTABLE_REF_KIND": (".executable", "EXECUTABLE_REF_KIND"),
    "EXECUTABLE_REF_VERSION": (".executable", "EXECUTABLE_REF_VERSION"),
    "ExecutableRef": (".executable", "ExecutableRef"),
    "InstallReceipt": (".manifest", "InstallReceipt"),
    "ModuleCatalog": (".service", "ModuleCatalog"),
    "ModuleResolver": (".base", "ModuleResolver"),
    "ModuleService": (".service", "ModuleService"),
    "ResolvedTagValue": (".tags", "ResolvedTagValue"),
    "TagDefinition": (".tags", "TagDefinition"),
    "TagView": (".tags", "TagView"),
    "capsule_executable_ref": (".executable", "capsule_executable_ref"),
    "module_resolver": (".base", "module_resolver"),
    "python_executable_ref": (".executable", "python_executable_ref"),
    "tag": (".tags", "tag"),
    "tag_definition": (".tags", "tag_definition"),
    "tags": (".tags", "tags"),
}


class _ModulesPackage(ModuleType):
    """Keep ``tags`` bound to the public ``tags()`` function.

    Python sets a submodule as an attribute of its package when the submodule
    is first imported. ``heavenbase.modules.tags`` names both the ``tags()``
    export and the submodule that defines it, so that binding must not replace
    the export.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "tags" and isinstance(value, ModuleType):
            return
        super().__setattr__(name, value)


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
sys.modules[__name__].__class__ = _ModulesPackage
