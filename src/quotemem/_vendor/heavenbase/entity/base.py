"""Declarative entity base class."""

from __future__ import annotations

__all__ = [
    "Entity",
    "EntityMeta",
    "pending_entities",
]

import threading

from heavenbase.utils import Any, Dict, List, Optional

from ..modules import ModuleResolver, module_resolver
from ..schema.model import EntitySchema
from ..utils.ids import check_field_name, check_workspace_id
from .assembly import assemble_entity
from .fields import DeriveSpec, FieldSpec

_PENDING: List[type] = []
_DEFAULT_WS: Any = None
_WORKSPACE_BINDER: Any = None
_ENTITY_LOCK = threading.RLock()


class EntityMeta(type):
    """Attach assembled entity metadata and registration lifecycle state."""

    def __new__(mcs, name: str, bases: tuple, ns: dict, *, _pending: bool = True):
        cls = super().__new__(mcs, name, bases, ns)
        if name == "Entity":
            return cls
        assembly = assemble_entity(name, ns)
        cls.identifier = assembly.schema.entity_id
        for attr, ref in assembly.refs.items():
            setattr(cls, attr, ref)
        cls._hb_schema = assembly.schema
        cls._hb_specs = assembly.specs
        cls._hb_indexes = assembly.indexes
        cls._hb_constraints = assembly.constraints
        cls._hb_derives = _collect_derives(ns, assembly.specs)
        cls._hb_workspace_bindings = set()
        # Bindings declared through Entity.register survive a workspace drop; bindings made by registering in a live workspace do not.
        cls._hb_declared_bindings = set()
        cls._hb_workspace_lock = threading.RLock()
        if not _pending:
            return cls
        with _ENTITY_LOCK:
            _PENDING.append(cls)
            default_ws = _DEFAULT_WS
        with cls._hb_workspace_lock:
            bound_to_default = default_ws is not None and (default_ws.context.identity_token, default_ws.id) in cls._hb_workspace_bindings
        if default_ws is not None and getattr(default_ws, "_register_pending_entities", False) and bound_to_default:
            default_ws.register(cls)
        return cls


class Entity(metaclass=EntityMeta):
    """Base class for logical workspace entities.

    Subclass ``Entity`` and declare fields with ``hb.field(...)``. The
    metaclass compiles those declarations into an ``EntitySchema`` and makes
    field references available for query expressions.

    Args:
        **values: Initial row values stored on this entity instance.

    Examples:
        >>> class Note(hb.Entity):
        ...     title = hb.field(hb.ShortText)
        >>> Note.from_dict({"object_id": "n1", "title": "Hello"}).to_dict()["title"]
        'Hello'
    """

    def __init__(self, **values: Any):
        """Create one entity row value.

        Args:
            **values (Any): Logical field values keyed by field name.
        """
        self._data: Dict[str, Any] = dict(values)

    @classmethod
    def derive(cls, target: str) -> DeriveSpec:
        """Declare a deterministic native-field or Catalog derivation.

        Args:
            target: Native field name such as ``"summary"`` or Catalog
                projection target such as ``"catalog.name"``.

        Returns:
            A ``DeriveSpec`` ready for ``.compute(...)`` chaining.
        """
        return DeriveSpec(str(target))

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, **values: Any) -> "Entity":
        """Build an entity row from mapping and keyword values.

        Args:
            data (Dict[str, Any] | None): Initial logical field values.
            **values (Any): Values that override matching mapping keys.

        Returns:
            Entity: New row value for this entity class.
        """
        merged = dict(data or {})
        merged.update(values)
        return cls(**merged)

    @classmethod
    def from_schema(
        cls,
        spec: Dict[str, Any],
        *,
        computes: Optional[Dict[str, Any]] = None,
        query_computes: Optional[Dict[str, Any]] = None,
        resolver: ModuleResolver | None = None,
    ) -> type:
        """Build an Entity subclass from a JSON-compatible schema definition.

        Args:
            spec (Dict[str, Any]): Entity schema definition with ``entity_id`` and ``fields``.
            computes (Dict[str, Any] | None): Optional compute functions keyed by schema reference. Defaults to ``None``.
            query_computes (Dict[str, Any] | None): Optional query-time compute functions keyed by schema reference. Defaults to ``None``.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            type: Generated ``Entity`` subclass.
        """
        from .compiler import _entity_from_schema

        return _entity_from_schema(
            spec,
            computes=computes,
            query_computes=query_computes,
            resolver=module_resolver(resolver),
        )

    @staticmethod
    def from_pydantic(model: Any, **kwargs: Any) -> Any:
        """Build an InteropSet from a Pydantic model class and optional rows.

        Args:
            model (Any): Pydantic model class defining source fields.
            **kwargs (Any): Forwarded to
                :func:`heavenbase.interop.from_pydantic`.

        Returns:
            InteropSet: Staged entity, normalized rows, and conversion report.
        """
        from ..interop import from_pydantic

        return from_pydantic(model, **kwargs)

    @staticmethod
    def from_pandas(df: Any, **kwargs: Any) -> Any:
        """Build an InteropSet from a Pandas DataFrame.

        Args:
            df (Any): ``pandas.DataFrame`` source table.
            **kwargs (Any): Forwarded to :func:`heavenbase.interop.from_pandas`.

        Returns:
            InteropSet: Staged entity, normalized rows, and conversion report.
        """
        from ..interop import from_pandas

        return from_pandas(df, **kwargs)

    @staticmethod
    def from_pyarrow(table: Any, **kwargs: Any) -> Any:
        """Build an InteropSet from a PyArrow Table.

        Args:
            table (Any): ``pyarrow.Table`` source table.
            **kwargs (Any): Forwarded to
                :func:`heavenbase.interop.from_pyarrow`.

        Returns:
            InteropSet: Staged entity, normalized rows, and conversion report.
        """
        from ..interop import from_pyarrow

        return from_pyarrow(table, **kwargs)

    @staticmethod
    def from_numpy(array: Any, **kwargs: Any) -> Any:
        """Build an InteropSet from a NumPy array.

        Args:
            array (Any): Structured or table-shaped NumPy array.
            **kwargs (Any): Forwarded to :func:`heavenbase.interop.from_numpy`.

        Returns:
            InteropSet: Staged entity, normalized rows, and conversion report.
        """
        from ..interop import from_numpy

        return from_numpy(array, **kwargs)

    @staticmethod
    def from_sql_result(result: Any, **kwargs: Any) -> Any:
        """Build an InteropSet from SQL-like rows or results.

        Args:
            result (Any): SQLAlchemy result, ``SQLResponse``, mapping, or
                table-shaped row sequence.
            **kwargs (Any): Forwarded to
                :func:`heavenbase.interop.from_sql_result`.

        Returns:
            InteropSet: Staged entity, normalized rows, and conversion report.
        """
        from ..interop import from_sql_result

        return from_sql_result(result, **kwargs)

    @staticmethod
    def from_daft(df: Any, **kwargs: Any) -> Any:
        """Build an InteropSet from a Daft DataFrame.

        Args:
            df (Any): ``daft.DataFrame`` source table.
            **kwargs (Any): Forwarded to :func:`heavenbase.interop.from_daft`.

        Returns:
            InteropSet: Staged entity, normalized rows, and conversion report.
        """
        from ..interop import from_daft

        return from_daft(df, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Return an isolated mapping of logical field values.

        Args:
            None.

        Returns:
            Dict[str, Any]: Shallow copy of this row's values.
        """
        return dict(self._data)

    @classmethod
    def schema(cls) -> EntitySchema:
        """Return this entity class's compiled schema IR.

        Args:
            None.

        Returns:
            EntitySchema: Passive schema assembled at class creation.
        """
        return cls._hb_schema

    @classmethod
    def workspace_ids(cls) -> List[str]:
        """Return workspace identifiers bound to this entity class.

        Args:
            None.

        Returns:
            List[str]: Sorted workspace identifiers.
        """
        with getattr(cls, "_hb_workspace_lock", _ENTITY_LOCK):
            return sorted({workspace_id for _, workspace_id in getattr(cls, "_hb_workspace_bindings", set())})

    @classmethod
    def register(cls, ws_id: str, *, context: Any = None) -> type:
        """Bind this entity class to a workspace identifier.

        Args:
            ws_id (str): Existing or future workspace identifier.
            context: Context that owns the workspace binding. Omission selects
                the process default Context.

        Returns:
            type: This entity class for decorator-style use.

        Notes:
            The declaration outlives a drop of that workspace, so a workspace
            created again with the same id registers this class again.
        """
        from ..context import context_or_default

        owner = context_or_default(context)
        ws_id = check_workspace_id(ws_id)
        with getattr(cls, "_hb_workspace_lock", _ENTITY_LOCK):
            cls._hb_workspace_bindings.add((owner.identity_token, ws_id))
            cls._hb_declared_bindings.add((owner.identity_token, ws_id))
        binder = _WORKSPACE_BINDER
        if binder is not None:
            binder(cls, ws_id, owner)
        return cls


def pending_entities() -> List[type]:
    """Return entity classes declared before workspace registration.

    Args:
        None.

    Returns:
        List[type]: Entity subclasses awaiting workspace binding.
    """
    with _ENTITY_LOCK:
        return list(_PENDING)


def _set_default_workspace(workspace: Any) -> None:
    """Set the process default workspace used for auto-registration."""
    global _DEFAULT_WS
    with _ENTITY_LOCK:
        _DEFAULT_WS = workspace


def _set_workspace_binder(binder: Any) -> None:
    """Set the workspace-layer callback for entity-side registration."""
    global _WORKSPACE_BINDER
    with _ENTITY_LOCK:
        _WORKSPACE_BINDER = binder


def _collect_derives(ns: dict, specs: Dict[str, FieldSpec]) -> Dict[str, DeriveSpec]:
    """Return class-level derives with later targets overwriting earlier ones."""
    derives: Dict[str, DeriveSpec] = {}
    for raw in ns.get("__derive__", []) or []:
        if not isinstance(raw, DeriveSpec):
            raise TypeError("__derive__ entries must be created with Entity.derive(...).compute(...)")
        target = _check_derive_target(raw.target, specs)
        if not raw.computes:
            raise ValueError(f"Derive target {target!r} needs compute")
        raw.target = target
        if target in derives:
            derives.pop(target)
        derives[target] = raw
    return derives


def _check_derive_target(target: str, specs: Dict[str, FieldSpec]) -> str:
    """Validate one derive target and return its canonical text."""
    value = str(target).strip()
    if value.startswith("catalog."):
        field_name = value.split(".", 1)[1]
        if not field_name:
            raise ValueError("Catalog derive targets must use catalog.<field>")
        check_field_name(field_name)
        return f"catalog.{field_name}"
    if "." in value:
        raise ValueError("Native derive targets use bare field names; Catalog targets use catalog.<field>")
    field_name = check_field_name(value)
    if field_name not in specs:
        raise ValueError(f"Native derive target {field_name!r} is not an entity field")
    return field_name
