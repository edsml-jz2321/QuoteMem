"""Registry-backed Backend combinations for benchmark runs."""

from __future__ import annotations

__all__ = ["BackendCombination", "CombinationContext"]

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..context import Context
from ..modules import ModuleResolver, module_resolver
from ..modules.base import ResolverCache
from ..utils.ids import check_id

_COMBINATION_KIND = "benchmark_combination"
_COMBINATION_CACHE = ResolverCache["BackendCombination"]()


def _expand(value: Any, replacements: Mapping[str, str]) -> Any:
    """Clone declarative recipe data and replace benchmark path tokens."""
    if isinstance(value, str):
        for name, replacement in replacements.items():
            value = value.replace("{" + name + "}", replacement)
        return value
    if isinstance(value, Mapping):
        return {key: _expand(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand(item, replacements) for item in value)
    return deepcopy(value)


def _freeze(value: Any) -> Any:
    """Return an immutable snapshot of declarative recipe data."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True)
class CombinationContext:
    """Name and filesystem state for one Context-owned benchmark run.

    Args:
        context (Context): Composition authority used to resolve and construct
            every Backend in the run.
        label (str): Registered benchmark-combination identifier.
        scale (int): Positive workload scale factor.
        repeat (int): One-based repeat number.
        seed (int): Stable random seed for the complete benchmark.
        tmp (str): Absolute temporary directory owned by this run.
        ws_id (str): Temporary workspace identifier owned by this run.
    """

    context: Context
    label: str
    scale: int
    repeat: int
    seed: int
    tmp: str
    ws_id: str


@dataclass(frozen=True)
class BackendCombination:
    """Declarative Backend layout resolved from the module catalog.

    Args:
        label (str): Canonical combination identifier.
        combination_type (str): Human-readable architecture category.
        backends (Mapping[str, Mapping[str, Any]]): Backend construction specs.
            Each leaf uses the same Registry-backed contract accepted by a
            `HeavenBase` workspace.
        scalar_backend (str): Backend instance receiving scalar fields.
        vector_backend (str): Backend instance receiving vector fields.
        resource_paths (Sequence[str]): Filesystem paths sampled for disk use.
            Values may contain `{tmp}`, `{ws_id}`, and `{label}` tokens.
        desc (str): Human-readable recipe purpose.
        suites (Sequence[str]): Benchmark-suite tag values attached to the recipe.
    """

    label: str
    combination_type: str
    backends: Mapping[str, Mapping[str, Any]]
    scalar_backend: str = "main"
    vector_backend: str = "main"
    resource_paths: Sequence[str] = ()
    desc: str = ""
    suites: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate and isolate the declarative recipe.

        Args:
            None.

        Returns:
            None: The recipe is normalized in place as an immutable snapshot.

        Raises:
            TypeError: If Backend specs or resource paths have invalid shapes.
            ValueError: If an identifier, route, or combination type is invalid.
        """
        label = check_id(self.label, "benchmark combination identifier")
        combination_type = str(self.combination_type or "").strip()
        if not combination_type:
            raise ValueError("benchmark combination_type must be non-empty")
        if not isinstance(self.backends, Mapping) or not self.backends:
            raise TypeError("benchmark combination backends must be a non-empty mapping")
        backends: dict[str, dict[str, Any]] = {}
        for alias, raw in self.backends.items():
            name = check_id(str(alias), "benchmark Backend name")
            if not isinstance(raw, Mapping):
                raise TypeError(f"benchmark Backend {name!r} must be a mapping")
            spec = _expand(raw, {})
            backend_type = spec.get("type")
            if not isinstance(backend_type, str) or not backend_type.strip():
                raise ValueError(f"benchmark Backend {name!r} requires a non-empty type")
            backends[name] = spec
        scalar = check_id(self.scalar_backend, "benchmark scalar Backend")
        vector = check_id(self.vector_backend, "benchmark vector Backend")
        for route in (scalar, vector):
            if route not in backends:
                raise ValueError(f"benchmark route {route!r} is absent from backends {sorted(backends)!r}")
        if isinstance(self.resource_paths, (str, bytes)) or not isinstance(self.resource_paths, Sequence):
            raise TypeError("benchmark resource_paths must be a sequence of strings")
        if any(not isinstance(path, str) for path in self.resource_paths):
            raise TypeError("benchmark resource_paths must contain only strings")
        paths = tuple(self.resource_paths)
        if any(not path for path in paths):
            raise ValueError("benchmark resource paths must be non-empty strings")
        if isinstance(self.suites, (str, bytes)) or not isinstance(self.suites, Sequence):
            raise TypeError("benchmark suites must be a sequence of identifiers")
        if any(not isinstance(suite, str) for suite in self.suites):
            raise TypeError("benchmark suites must contain only strings")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "combination_type", combination_type)
        object.__setattr__(self, "backends", _freeze(backends))
        object.__setattr__(self, "scalar_backend", scalar)
        object.__setattr__(self, "vector_backend", vector)
        object.__setattr__(self, "resource_paths", paths)
        object.__setattr__(self, "desc", str(self.desc or "").strip())
        object.__setattr__(self, "suites", tuple(dict.fromkeys(check_id(suite, "benchmark suite") for suite in self.suites)))

    @classmethod
    def load(cls, label: str, *, resolver: ModuleResolver) -> "BackendCombination":
        """Load one active benchmark recipe through a module resolver.

        Args:
            label (str): Canonical identifier or registered alias.
            resolver (ModuleResolver): Context-owned module authority.

        Returns:
            BackendCombination: Validated declarative recipe.

        Raises:
            KeyError: If the recipe is absent or inactive.
            TypeError: If the resolved inline definition is not a mapping.
        """
        if resolver is None:
            raise TypeError("BackendCombination.load requires an explicit ModuleResolver")
        selected = module_resolver(resolver)
        rows = selected.inspect(_COMBINATION_KIND, label)
        if not rows:
            raise KeyError(f"Unknown benchmark combination {label!r}")
        return _combination_from_row(rows[0], selected)

    @classmethod
    def list(
        cls,
        *,
        resolver: ModuleResolver,
        suite: str | None = "default",
    ) -> tuple["BackendCombination", ...]:
        """List registered benchmark recipes in catalog order.

        Args:
            resolver (ModuleResolver): Context-owned module authority.
            suite (str | None): Optional benchmark-suite claim. `"default"`
                returns the dependency-light standard run; `None` returns every
                active recipe.

        Returns:
            tuple[BackendCombination, ...]: Validated registered recipes.
        """
        if resolver is None:
            raise TypeError("BackendCombination.list requires an explicit ModuleResolver")
        selected = module_resolver(resolver)
        clean_suite = check_id(suite, "benchmark suite") if suite is not None else None
        rows = selected.inspect(_COMBINATION_KIND)
        if clean_suite is not None:
            rows = tuple(row for row in rows if clean_suite in _benchmark_suites(row))
        return tuple(_combination_from_row(row, selected) for row in rows)

    def build(self, ctx: CombinationContext) -> dict[str, dict[str, Any]]:
        """Return independent Backend specs for one run.

        Args:
            ctx (CombinationContext): Context-owned run identity and paths.

        Returns:
            dict[str, dict[str, Any]]: Expanded Backend configuration consumed
            through the workspace's Registry-backed Backend construction path.
        """
        replacements = {"tmp": ctx.tmp, "ws_id": ctx.ws_id, "label": ctx.label}
        return _expand(self.backends, replacements)

    def paths(self, ctx: CombinationContext) -> tuple[str, ...]:
        """Return expanded filesystem paths sampled for one run.

        Args:
            ctx (CombinationContext): Context-owned run identity and paths.

        Returns:
            tuple[str, ...]: Expanded resource paths.
        """
        replacements = {"tmp": ctx.tmp, "ws_id": ctx.ws_id, "label": ctx.label}
        return tuple(_expand(path, replacements) for path in self.resource_paths)


def _combination_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> BackendCombination:
    """Materialize one validated recipe at an exact catalog publication."""

    def build(current: Mapping[str, Any]) -> BackendCombination:
        definition = resolver.resolve(_COMBINATION_KIND, str(current["identifier"]))
        if not isinstance(definition, Mapping):
            raise TypeError(f"Benchmark combination {current['identifier']!r} must resolve to a mapping")
        data = dict(definition)
        definition_desc = data.pop("desc", "")
        values = {
            "label": str(current["identifier"]),
            "combination_type": data.pop("combination_type", ""),
            "backends": data.pop("backends", {}),
            "scalar_backend": data.pop("scalar_backend", "main"),
            "vector_backend": data.pop("vector_backend", "main"),
            "resource_paths": data.pop("resource_paths", ()),
            "desc": str(current["record"].get("desc") or definition_desc),
            "suites": _benchmark_suites(current),
        }
        if data:
            raise ValueError(f"Benchmark combination {current['identifier']!r} has unsupported fields {sorted(data)!r}")
        return BackendCombination(**values)

    return _COMBINATION_CACHE.get(resolver, row, build)


def _benchmark_suites(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Interpret one combination's raw benchmark-suite claim locally."""
    record = row.get("record")
    meta = record.get("meta") if isinstance(record, Mapping) else None
    tags = meta.get("tags") if isinstance(meta, Mapping) else None
    value = tags.get("benchmark-suites") if isinstance(tags, Mapping) else None
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"Benchmark combination {row['identifier']!r} tag 'benchmark-suites' must be a list of identifiers")
    if any(not isinstance(suite, str) for suite in value):
        raise TypeError(f"Benchmark combination {row['identifier']!r} tag 'benchmark-suites' must contain only identifiers")
    return tuple(dict.fromkeys(check_id(suite, "benchmark suite") for suite in value))
