"""Resolved field placement routes and entity plans."""

from __future__ import annotations

__all__ = ["StorageBinding", "StoragePlan"]

from dataclasses import InitVar, KW_ONLY

from heavenbase.utils import Any, Dict, List, dataclass, deepcopy, field

from ..modules import ModuleResolver
from ..strategies import StorageStrategy
from .index import IndexBinding, UniqueBinding


@dataclass
class StorageBinding:
    """Describe one resolved field copy on one workspace backend.

    Args:
        entity_id (str): Entity identifier that owns the field.
        field_name (str): Logical field name.
        backend (str): Workspace backend instance name.
        strategy (str): Canonical storage strategy identifier.
        source (str): Rule or profile source that produced this route.
        strategy_options (Dict[str, Any]): Strategy-owned route options.
        params (Dict[str, Any]): Durable placement parameters.
        resolver (ModuleResolver | None): Context-owned module resolver used to
            normalize and compare the strategy. The default Context is selected
            when omitted.
    """

    entity_id: str
    field_name: str
    backend: str
    strategy: str
    source: str = "default"
    strategy_options: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    _: KW_ONLY
    resolver: InitVar[ModuleResolver | None] = None

    def __post_init__(self, resolver: ModuleResolver | None) -> None:
        """Normalize route identity and isolate mutable metadata.

        Args:
            None.

        Returns:
            None: This initializer stores canonical copied values.
        """
        self.entity_id = str(self.entity_id)
        self.field_name = str(self.field_name)
        self.backend = str(self.backend)
        self._module_resolver = resolver
        self.strategy = StorageStrategy.identifier_of(self.strategy, resolver=self._module_resolver)
        self.source = str(self.source)
        self.strategy_options = deepcopy(dict(self.strategy_options))
        self.params = deepcopy(dict(self.params))

    def matches(self, *, backend: str | None = None, strategy: Any = None) -> bool:
        """Return whether this route satisfies every supplied constraint.

        Args:
            backend (str | None): Optional workspace backend name.
            strategy (Any): Optional storage strategy selector.

        Returns:
            bool: Whether all supplied constraints match.
        """
        if backend is not None and self.backend != backend:
            return False
        if strategy is not None and self.strategy != StorageStrategy.identifier_of(strategy, resolver=self._module_resolver):
            return False
        return True

    def strategy_spec(self) -> str | dict[str, Any]:
        """Return the public serialized strategy shape.

        Args:
            None.

        Returns:
            str | dict[str, Any]: Identifier or strategy specification mapping.
        """
        return StorageStrategy.spec_of(self.strategy, self.strategy_options, resolver=self._module_resolver)

    def route_notes(self) -> dict[str, Any]:
        """Return isolated strategy metadata for query fragments.

        Args:
            None.

        Returns:
            dict[str, Any]: Strategy identifier and optional strategy options.
        """
        notes: dict[str, Any] = {"strategy": self.strategy}
        if self.strategy_options:
            notes["strategy_options"] = deepcopy(self.strategy_options)
        return notes

    def to_dict(self) -> dict[str, Any]:
        """Serialize this resolved route.

        Args:
            None.

        Returns:
            dict[str, Any]: JSON-compatible binding metadata.
        """
        data = {
            "entity_id": self.entity_id,
            "field_name": self.field_name,
            "backend": self.backend,
            "strategy": self.strategy,
            "source": self.source,
        }
        if self.strategy_options:
            data["strategy_options"] = deepcopy(self.strategy_options)
        if self.params:
            data["params"] = deepcopy(self.params)
        return data


@dataclass
class StoragePlan:
    """Collect every resolved storage route for one entity.

    Args:
        entity_id (str): Entity identifier.
        bindings (List[StorageBinding]): Effective field routes in resolver order.
        indexes (List[IndexBinding]): Resolved scalar lookup indexes.
        constraints (List[UniqueBinding]): Resolved uniqueness owners.
    """

    entity_id: str
    bindings: List[StorageBinding] = field(default_factory=list)
    indexes: List[IndexBinding] = field(default_factory=list)
    constraints: List[UniqueBinding] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize plan identity and validate resolved route values.

        Args:
            None.

        Returns:
            None: This initializer copies validated bindings in place.

        Raises:
            TypeError: If a binding is not a ``StorageBinding``.
        """
        self.entity_id = str(self.entity_id)
        self.bindings = list(self.bindings)
        self.indexes = list(self.indexes)
        self.constraints = list(self.constraints)
        if any(not isinstance(binding, StorageBinding) for binding in self.bindings):
            raise TypeError("StoragePlan bindings must be StorageBinding values")
        if any(not isinstance(binding, IndexBinding) for binding in self.indexes):
            raise TypeError("StoragePlan indexes must be IndexBinding values")
        if any(not isinstance(binding, UniqueBinding) for binding in self.constraints):
            raise TypeError("StoragePlan constraints must be UniqueBinding values")

    def field_bindings(self, field_name: str) -> List[StorageBinding]:
        """Return all resolved routes for one field.

        Args:
            field_name (str): Logical field name.

        Returns:
            List[StorageBinding]: Matching routes in resolver order.
        """
        return [binding for binding in self.bindings if binding.field_name == field_name]

    def first(self, field_name: str) -> StorageBinding:
        """Return the first resolved route for one field.

        Args:
            field_name (str): Logical field name.

        Returns:
            StorageBinding: First matching route.

        Raises:
            LookupError: If the field has no resolved route.
        """
        bindings = self.field_bindings(field_name)
        if not bindings:
            hint = ""
            if "." in str(field_name):
                hint = (
                    "; dotted paths are not standalone fields — they read nested values of a declared Json field "
                    '(such as {"meta.venue": ...} filters or group_by/order/select ["meta.venue"])'
                )
            raise LookupError(f"No storage binding for {self.entity_id}.{field_name}{hint}")
        return bindings[0]

    def has_binding(self, field_name: str, *, backend: str | None = None, strategy: Any = None) -> bool:
        """Return whether a field has a route matching constraints.

        Args:
            field_name (str): Logical field name.
            backend (str | None): Optional workspace backend name.
            strategy (Any): Optional storage strategy selector.

        Returns:
            bool: Whether a matching route exists.
        """
        return any(binding.matches(backend=backend, strategy=strategy) for binding in self.field_bindings(field_name))

    def backend_indexes(self, backend: str) -> List[IndexBinding]:
        """Return scalar lookup indexes resolved to one backend.

        Args:
            backend: Workspace backend instance name.

        Returns:
            List[IndexBinding]: Matching indexes in resolver order.
        """
        return [binding for binding in self.indexes if binding.backend == backend]

    def backend_constraints(self, backend: str) -> List[UniqueBinding]:
        """Return uniqueness constraints owned by one backend.

        Args:
            backend: Workspace backend instance name.

        Returns:
            List[UniqueBinding]: Matching constraints in resolver order.
        """
        return [binding for binding in self.constraints if binding.backend == backend]

    def has_index(self, field_name: str, *, backend: str | None = None) -> bool:
        """Return whether a single-field index or uniqueness constraint covers a route.

        Args:
            field_name: Logical field name.
            backend: Optional workspace backend instance name.

        Returns:
            bool: Whether the field has a resolved scalar lookup path.
        """
        bindings = [*self.indexes, *self.constraints]
        return any(binding.fields == (field_name,) and (backend is None or binding.backend == backend) for binding in bindings)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this entity storage plan.

        Args:
            None.

        Returns:
            dict[str, Any]: Entity identifier and serialized bindings.
        """
        out = {"entity_id": self.entity_id, "bindings": [binding.to_dict() for binding in self.bindings]}
        if self.indexes:
            out["indexes"] = [binding.to_dict() for binding in self.indexes]
        if self.constraints:
            out["constraints"] = [binding.to_dict() for binding in self.constraints]
        return out
