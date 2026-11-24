"""Shared lazy export loading helpers for package facades."""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.util import find_spec
from pkgutil import iter_modules
from typing import Any, Callable, Collection, Mapping

LazyExportTable = Mapping[str, tuple[str, str | None]]
LazyExportGetter = Callable[[str], Any]
LazyExportDir = Callable[[], list[str]]
RegistryEntityExportTable = Mapping[str, str]
RegistryExecutableExportTable = Mapping[str, tuple[str, str]]
RegistrySiblingExportTable = Mapping[str, tuple[str, str, str]]


def load_lazy_export(
    name: str,
    exports: LazyExportTable,
    package: str | None,
    namespace: dict[str, Any],
    *,
    owner: str | None = None,
) -> Any:
    """Load one lazy export and cache it on the caller namespace.

    Args:
        name: Requested attribute name.
        exports: Mapping from public attribute name to ``(module, attr)``.
        package: Package anchor for relative imports, or ``None`` for absolute imports.
        namespace: Caller globals where the resolved value should be cached.
        owner: Module name to report in ``AttributeError``. Defaults to ``package``.

    Returns:
        The resolved module or module attribute.

    Raises:
        AttributeError: If ``name`` is neither in ``exports`` nor a submodule of
            the package that owns ``namespace``.
    """
    target = exports.get(name)
    module_owner = owner or package or ""
    if target is None:
        return load_submodule(name, namespace, owner=module_owner)
    module_name, attr_name = target
    module = import_module(module_name, package) if package else import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    namespace[name] = value
    return value


def load_submodule(name: str, namespace: dict[str, Any], *, owner: str | None = None) -> Any:
    """Import a package's submodule on attribute access.

    A package that imported its submodules in ``__init__`` bound them as
    attributes on import. A lazy facade imports them on first access instead,
    so ``package.submodule`` keeps working without an explicit import. The
    import runs on the accessing thread, outside any package's import.

    Args:
        name: Requested attribute name.
        namespace: Globals of the package that owns the attribute.
        owner: Module name to report in ``AttributeError``. Defaults to the package name.

    Returns:
        The imported submodule.

    Raises:
        AttributeError: If ``name`` is not a submodule of the package, or the
            submodule cannot be imported (the import error is the cause).
    """
    package = namespace.get("__name__")
    module_owner = owner or package or ""
    missing = AttributeError(f"module {module_owner!r} has no attribute {name!r}")
    if not isinstance(package, str) or "__path__" not in namespace or name.startswith("__") or not name.isidentifier():
        raise missing
    qualified = f"{package}.{name}"
    try:
        if qualified not in sys.modules and find_spec(qualified) is None:
            raise missing
        return import_module(qualified)
    except ImportError as exc:
        raise AttributeError(f"module {module_owner!r} has no attribute {name!r} ({exc})") from exc


def submodule_names(namespace: dict[str, Any]) -> set[str]:
    """Return the importable submodule names of the package that owns ``namespace``.

    Args:
        namespace: Package globals; a plain module has no submodules.

    Returns:
        Submodule names, without dunder names such as ``__main__``.
    """
    path = namespace.get("__path__")
    if path is None:
        return set()
    return {info.name for info in iter_modules(path) if not info.name.startswith("__")}


def _attach_lazy_exports(
    package: str,
    namespace: dict[str, Any],
    exports: LazyExportTable,
    public: Collection[str],
    *,
    owner: str | None = None,
) -> tuple[LazyExportGetter, LazyExportDir]:
    """Create standard lazy attribute and directory hooks for one module.

    Args:
        package: Package anchor used for relative import targets.
        namespace: Module globals where resolved values are cached.
        exports: Public names mapped to ``(module, attr)`` targets.
        public: Complete public surface, normally ``__all__``.
        owner: Module name used in missing-attribute errors. Defaults to ``package``.

    Returns:
        Module ``__getattr__`` and ``__dir__`` hooks.
    """

    def __getattr__(name: str) -> Any:
        return load_lazy_export(name, exports, package, namespace, owner=owner)

    def __dir__() -> list[str]:
        return sorted(set(namespace).union(exports, public, submodule_names(namespace)))

    return __getattr__, __dir__


def _attach_registry_exports(
    namespace: dict[str, Any],
    public: Collection[str],
    *,
    entities: RegistryEntityExportTable | None = None,
    executables: RegistryExecutableExportTable | None = None,
    siblings: RegistrySiblingExportTable | None = None,
    owner: str | None = None,
) -> tuple[LazyExportGetter, LazyExportDir]:
    """Create dynamic facade hooks backed by the active Registry generation.

    Entity exports resolve their own Registry records. Sibling exports use an
    Entity record as the artifact-generation anchor, so helpers and companion
    classes come from the same verified source folder. Executable exports
    resolve their own Registry records.

    Args:
        namespace: Module globals used for directory reporting.
        public: Complete public surface, normally ``__all__``.
        entities: Public names mapped to Entity registration identifiers.
        executables: Public names mapped to ``(kind, identifier)`` records.
        siblings: Public names mapped to ``(entity, module, attribute)``.
        owner: Module name used in missing-attribute errors.

    Returns:
        Module ``__getattr__`` and ``__dir__`` hooks.

    Raises:
        RuntimeError: If a public name has multiple Registry routes or a route
            is absent from ``public``.
    """
    entity_routes = dict(entities or {})
    executable_routes = dict(executables or {})
    sibling_routes = dict(siblings or {})
    routed = [*entity_routes, *executable_routes, *sibling_routes]
    if len(routed) != len(set(routed)):
        raise RuntimeError("Registry facade export names must have exactly one route")
    undeclared = set(routed).difference(public)
    if undeclared:
        raise RuntimeError(f"Registry facade routes are absent from __all__: {sorted(undeclared)!r}")
    module_owner = owner or str(namespace.get("__name__", ""))

    def __getattr__(name: str) -> Any:
        entity_id = entity_routes.get(name)
        if entity_id is not None:
            from .entity.registry import load_entity_type

            return load_entity_type(entity_id)
        executable = executable_routes.get(name)
        if executable is not None:
            from .modules import module_resolver

            return module_resolver().resolve(*executable)
        sibling = sibling_routes.get(name)
        if sibling is not None:
            from .entity.registry import load_entity_artifact_export

            entity_id, module_name, attr_name = sibling
            return load_entity_artifact_export(entity_id, module_name, attr_name)
        return load_submodule(name, namespace, owner=module_owner)

    def __dir__() -> list[str]:
        return sorted(set(namespace).union(public, submodule_names(namespace)))

    return __getattr__, __dir__


_LATE_BINDINGS: dict[str, tuple[str, str]] = {}


def bind_late(name: str, module_name: str, attr_name: str) -> None:
    """Bind a named hook to an attribute of a module imported on first use.

    A lower package that may not import a higher one calls the hook by name;
    the package root, which runs before any submodule, binds it to its owner.
    The owner module is imported only when the hook is first resolved.

    Args:
        name: Hook name shared by the binder and the caller.
        module_name: Absolute module that owns the hook.
        attr_name: Attribute of ``module_name`` that implements the hook.

    Returns:
        None: The binding is recorded for :func:`resolve_late`.
    """
    _LATE_BINDINGS[name] = (module_name, attr_name)


def resolve_late(name: str) -> Any:
    """Return the attribute bound to a named hook, importing its owner.

    Args:
        name: Hook name passed to :func:`bind_late`.

    Returns:
        The bound attribute.

    Raises:
        KeyError: If nothing bound ``name``.
    """
    module_name, attr_name = _LATE_BINDINGS[name]
    return getattr(import_module(module_name), attr_name)
