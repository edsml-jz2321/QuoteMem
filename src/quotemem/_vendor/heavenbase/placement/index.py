"""Authored scalar-index intent and resolved placement-plan bindings."""

from __future__ import annotations

__all__ = [
    "FieldIndexRule",
    "FieldUniqueRule",
    "IndexBinding",
    "IndexResolver",
    "IndexRule",
    "UniqueBinding",
    "UniqueRule",
    "index",
    "unique",
]

from heavenbase.utils import Any, Iterable, List, Optional, Sequence, dataclass

from .scalar import ScalarIndexSpec
from ..strategies import InlineColumn
from ..utils.ids import check_field_name, check_id


def _fields(values: Iterable[Any]) -> tuple[str, ...]:
    """Return a validated, ordered field tuple."""
    out = tuple(check_field_name(str(value)) for value in values)
    if not out:
        raise ValueError("Scalar index declarations need at least one field")
    if len(set(out)) != len(out):
        raise ValueError(f"Scalar index fields must be unique and ordered; got {list(out)!r}")
    if "object_id" in out:
        raise ValueError("object_id indexing is fixed by the backend and cannot be customized")
    return out


def _backend(value: Optional[str]) -> Optional[str]:
    """Return a validated optional backend instance name."""
    return check_id(value, "backend instance name") if value is not None else None


def _scalar_index(backend: Any) -> ScalarIndexSpec:
    """Materialize one Backend's scalar-index tag as a local typed facade."""
    result = backend.tag("scalar-index")
    try:
        return ScalarIndexSpec.load(result.value if result is not None else {})
    except (TypeError, ValueError):
        return ScalarIndexSpec()


@dataclass(frozen=True)
class FieldIndexRule:
    """Declare scalar index intent before the owning field name is known.

    Args:
        enabled: Whether provider-default indexing should be enabled or disabled.
        backend: Optional workspace backend instance targeted by the declaration.
    """

    enabled: bool = True
    backend: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the authored field-level declaration.

        Raises:
            TypeError: If ``enabled`` is not a boolean.
        """
        if not isinstance(self.enabled, bool):
            raise TypeError("field.index enabled must be a boolean")
        object.__setattr__(self, "backend", _backend(self.backend))


@dataclass(frozen=True)
class FieldUniqueRule:
    """Declare a uniqueness constraint before the owning field name is known.

    Args:
        backend: Optional workspace backend instance that owns enforcement.
    """

    backend: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the optional backend selector."""
        object.__setattr__(self, "backend", _backend(self.backend))


@dataclass(frozen=True)
class IndexRule:
    """Describe authored scalar lookup acceleration independently of uniqueness.

    Args:
        fields: Ordered logical fields covered by the index.
        backend: Optional workspace backend instance targeted by the declaration.
        enabled: Whether the index is requested. ``False`` is field-level opt-out.
        source: Authored declaration source used in diagnostics and metadata.
    """

    fields: tuple[str, ...]
    backend: Optional[str] = None
    enabled: bool = True
    source: str = "entity"

    def __post_init__(self) -> None:
        """Normalize and validate authored index intent."""
        object.__setattr__(self, "fields", _fields(self.fields))
        object.__setattr__(self, "backend", _backend(self.backend))
        if not isinstance(self.enabled, bool):
            raise TypeError("IndexRule enabled must be a boolean")
        object.__setattr__(self, "source", str(self.source))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible declaration.

        Args:
            None.

        Returns:
            dict[str, Any]: Ordered fields plus optional target and opt-out flag.
        """
        out: dict[str, Any] = {"fields": list(self.fields)}
        if self.backend is not None:
            out["to"] = self.backend
        if not self.enabled:
            out["enabled"] = False
        return out


@dataclass(frozen=True)
class UniqueRule:
    """Describe an authored logical uniqueness constraint.

    Args:
        fields: Ordered logical fields whose non-null tuple must be unique.
        backend: Optional workspace backend instance that owns enforcement.
        source: Authored declaration source used in diagnostics and metadata.
    """

    fields: tuple[str, ...]
    backend: Optional[str] = None
    source: str = "entity"

    def __post_init__(self) -> None:
        """Normalize and validate authored uniqueness intent."""
        object.__setattr__(self, "fields", _fields(self.fields))
        object.__setattr__(self, "backend", _backend(self.backend))
        object.__setattr__(self, "source", str(self.source))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible declaration.

        Args:
            None.

        Returns:
            dict[str, Any]: Ordered fields plus the optional enforcement target.
        """
        out: dict[str, Any] = {"fields": list(self.fields)}
        if self.backend is not None:
            out["to"] = self.backend
        return out


def index(*fields: str, to: Optional[str] = None) -> IndexRule:
    """Request a portable scalar lookup index.

    Args:
        *fields: One field, or ordered fields for a composite index.
        to: Optional workspace backend instance. Without it, every compatible
            backend storing all fields receives the performance index.

    Returns:
        IndexRule: Authored index intent for an entity ``__indexes__`` list.

    Examples:
        ``__indexes__ = [hb.index("tenant", "created_at")]``
    """
    return IndexRule(tuple(fields), backend=to)


def unique(*fields: str, to: Optional[str] = None) -> UniqueRule:
    """Require non-null field tuples to be unique.

    Args:
        *fields: One field, or ordered fields for a composite constraint.
        to: Optional workspace backend instance that owns enforcement. This is
            required when more than one compatible stored copy could enforce it.

    Returns:
        UniqueRule: Authored constraint for an entity ``__constraints__`` list.

    Examples:
        ``__constraints__ = [hb.unique("tenant", "slug", to="main")]``
    """
    return UniqueRule(tuple(fields), backend=to)


@dataclass(frozen=True)
class IndexBinding:
    """Describe one resolved scalar lookup index on one backend.

    Args:
        entity_id: Owning entity identifier.
        fields: Ordered logical fields covered by the index.
        backend: Workspace backend instance that owns the physical index.
        source: ``default``, ``field``, or ``entity``.
    """

    entity_id: str
    fields: tuple[str, ...]
    backend: str
    source: str = "default"

    def __post_init__(self) -> None:
        """Normalize the resolved binding."""
        object.__setattr__(self, "entity_id", str(self.entity_id))
        object.__setattr__(self, "fields", _fields(self.fields))
        object.__setattr__(self, "backend", check_id(self.backend, "backend instance name"))
        object.__setattr__(self, "source", str(self.source))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible resolved index metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Entity, ordered fields, backend, and source.
        """
        return {"entity_id": self.entity_id, "fields": list(self.fields), "backend": self.backend, "source": self.source}


@dataclass(frozen=True)
class UniqueBinding:
    """Describe one resolved uniqueness owner on one backend.

    Args:
        entity_id: Owning entity identifier.
        fields: Ordered logical fields whose non-null tuple must be unique.
        backend: Workspace backend instance that enforces the constraint.
        source: ``field`` or ``entity``.
    """

    entity_id: str
    fields: tuple[str, ...]
    backend: str
    source: str = "entity"

    def __post_init__(self) -> None:
        """Normalize the resolved constraint binding."""
        object.__setattr__(self, "entity_id", str(self.entity_id))
        object.__setattr__(self, "fields", _fields(self.fields))
        object.__setattr__(self, "backend", check_id(self.backend, "backend instance name"))
        object.__setattr__(self, "source", str(self.source))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible resolved constraint metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Entity, ordered fields, backend, and source.
        """
        return {"entity_id": self.entity_id, "fields": list(self.fields), "backend": self.backend, "source": self.source}


class IndexResolver:
    """Resolve authored index and uniqueness intent against storage routes."""

    def resolve(
        self,
        schema: Any,
        storage_plan: Any,
        index_rules: Sequence[IndexRule],
        unique_rules: Sequence[UniqueRule],
        backends: Any,
    ) -> Any:
        """Attach validated scalar index bindings to a storage plan.

        Args:
            schema: Compiled logical entity schema.
            storage_plan: Resolved field placement plan to enrich.
            index_rules: Field- and entity-authored performance intent.
            unique_rules: Field- and entity-authored correctness constraints.
            backends: Workspace backend registry.

        Returns:
            StoragePlan: The supplied plan with resolved index and constraint lists.

        Raises:
            LookupError: If a targeted backend does not store every declared field.
            NotImplementedError: If a backend cannot honor explicit intent.
            ValueError: If declarations conflict or uniqueness ownership is ambiguous.
        """
        indexes = self._defaults(schema, storage_plan, backends)
        constraints: dict[tuple[str, tuple[str, ...]], UniqueBinding] = {}
        decisions: dict[tuple[Optional[str], tuple[str, ...]], bool] = {}
        for rule in index_rules:
            key = (rule.backend, rule.fields)
            if key in decisions and decisions[key] != rule.enabled:
                raise ValueError(f"Conflicting index declarations for {schema.entity_id}.{'.'.join(rule.fields)} on {rule.backend or 'all stored backends'}")
            decisions[key] = rule.enabled
            targets = self._targets(schema, storage_plan, rule.fields, rule.backend)
            if rule.enabled:
                selected = [name for name in targets if backends.get(name).supports_scalar_index(schema, rule.fields)]
                if rule.backend is not None and rule.backend not in selected:
                    self._require_support(backends.get(rule.backend), schema, rule.fields, unique=False)
                if not selected:
                    details = ", ".join(f"{name}:{backends.get(name).type}" for name in targets)
                    raise NotImplementedError(f"No backend storing {schema.entity_id}.{'.'.join(rule.fields)} can provide a scalar index ({details})")
            else:
                if rule.backend is not None and not _scalar_index(backends.get(rule.backend)).value:
                    backend = backends.get(rule.backend)
                    raise NotImplementedError(
                        f"Backend {backend.name!r} ({backend.type}) does not expose controllable scalar indexes; "
                        "provider-required search indexes cannot be disabled through field.index"
                    )
                selected = [
                    name for name in targets if _scalar_index(backends.get(name)).value or any((name, (field_name,)) in indexes for field_name in rule.fields)
                ]
            for backend_name in selected:
                backend = backends.get(backend_name)
                binding_key = (backend_name, rule.fields)
                if rule.enabled:
                    indexes[binding_key] = IndexBinding(schema.entity_id, rule.fields, backend_name, rule.source)
                else:
                    if not _scalar_index(backend).opt_out:
                        raise NotImplementedError(f"Backend {backend_name!r} ({backend.type}) cannot disable provider-default scalar indexes")
                    indexes.pop(binding_key, None)

        for rule in unique_rules:
            targets = self._targets(schema, storage_plan, rule.fields, rule.backend)
            supported = [name for name in targets if backends.get(name).supports_scalar_index(schema, rule.fields, unique=True)]
            if rule.backend is not None and rule.backend not in supported:
                self._require_support(backends.get(rule.backend), schema, rule.fields, unique=True)
            if not supported:
                details = ", ".join(f"{name}:{backends.get(name).type}" for name in targets)
                raise NotImplementedError(f"No backend storing {schema.entity_id}.{'.'.join(rule.fields)} can enforce uniqueness ({details})")
            if len(supported) > 1 and rule.backend is None:
                raise ValueError(
                    f"Unique constraint for {schema.entity_id}.{'.'.join(rule.fields)} has multiple possible owners {supported}; pass to=<backend>"
                )
            backend_name = supported[0]
            binding_key = (backend_name, rule.fields)
            if decisions.get((backend_name, rule.fields)) is False or decisions.get((None, rule.fields)) is False:
                raise ValueError(f"Unique constraint for {schema.entity_id}.{'.'.join(rule.fields)} conflicts with index(enabled=False)")
            constraints[binding_key] = UniqueBinding(schema.entity_id, rule.fields, backend_name, rule.source)
            indexes.pop(binding_key, None)  # a uniqueness primitive already provides the lookup path

        owners = sorted({binding.backend for binding in constraints.values()})
        if len(owners) > 1:
            raise ValueError(f"All unique constraints for replicated entity {schema.entity_id!r} must share one enforcement backend; got {owners}")

        storage_plan.indexes = list(indexes.values())
        storage_plan.constraints = list(constraints.values())
        return storage_plan

    def _defaults(self, schema: Any, storage_plan: Any, backends: Any) -> dict[tuple[str, tuple[str, ...]], IndexBinding]:
        """Return provider-default scalar index bindings."""
        out: dict[tuple[str, tuple[str, ...]], IndexBinding] = {}
        for binding in storage_plan.bindings:
            if binding.field_name == schema.pk_name or binding.strategy != InlineColumn.identifier:
                continue
            field_schema = schema.get(binding.field_name)
            backend = backends.get(binding.backend)
            if backend.defaults_scalar_index(field_schema.dtype):
                fields = (binding.field_name,)
                out[(binding.backend, fields)] = IndexBinding(schema.entity_id, fields, binding.backend)
        return out

    def _targets(self, schema: Any, storage_plan: Any, fields: tuple[str, ...], backend: Optional[str]) -> List[str]:
        """Return backends storing every field inline, respecting an explicit target."""
        for field_name in fields:
            schema.get(field_name)
        stored = sorted(
            name
            for name in {binding.backend for binding in storage_plan.bindings}
            if all(storage_plan.has_binding(field_name, backend=name) for field_name in fields)
        )
        candidates = sorted(
            name
            for name in {binding.backend for binding in storage_plan.bindings}
            if all(storage_plan.has_binding(field_name, backend=name, strategy=InlineColumn) for field_name in fields)
        )
        if backend is not None:
            if backend not in candidates:
                raise LookupError(f"Backend {backend!r} does not store every indexed field in {schema.entity_id}.{'.'.join(fields)} as inline scalar data")
            return [backend]
        if not candidates:
            if stored:
                raise NotImplementedError(
                    f"Backends {stored} cannot provide a scalar index for {schema.entity_id}.{'.'.join(fields)} because the fields are not stored with InlineColumn"
                )
            raise LookupError(f"No backend stores every indexed field in {schema.entity_id}.{'.'.join(fields)} as inline scalar data")
        return candidates

    def _require_support(self, backend: Any, schema: Any, fields: tuple[str, ...], *, unique: bool) -> None:
        """Raise when a backend cannot honor one explicit declaration."""
        if backend.supports_scalar_index(schema, fields, unique=unique):
            return
        kind = "unique constraint" if unique else "scalar index"
        dtypes = [type(schema.get(name).dtype).identifier for name in fields]
        raise NotImplementedError(
            f"Backend {backend.name!r} ({backend.type}) cannot provide {kind} for {schema.entity_id}.{'.'.join(fields)} with dtypes {dtypes}"
        )
