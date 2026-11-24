"""HyperG logical type."""

from __future__ import annotations

__all__ = [
    "HyperG",
]

from heavenbase.utils import Any, Mapping, dataclass

from ..modules import ModuleResolver
from ..utils.ids import check_field_name
from ..filters.ops import normalize_op
from .base import LogicalType
from .descriptor import coerce_type, logical_type_from_any
from .filtering import normalize_scalar_filter_value
from .registry import get_logical_type
from .short_text import ShortText

_DATA_TYPES = frozenset({"json", "artifact", "vector", "array", "hyper-g"})


@dataclass
class _HyperGCol:
    name: str
    dtype: LogicalType
    required: bool = True
    filterable: bool = True

    def validate(self, value: Any) -> Any:
        if value is None:
            if self.required:
                raise ValueError(f"HyperG field {self.name!r} is required")
            return None
        return self.dtype.validate(value)

    def filter_value(self, op: str, value: Any) -> Any:
        if not self.filterable:
            raise ValueError(f"HyperG field {self.name!r} is not filterable")
        if type(self.dtype).filter_value is not LogicalType.filter_value:
            return self.dtype.filter_value(op, value)
        return normalize_scalar_filter_value(self.dtype, op, value)

    def to_dict(self) -> dict:
        return {"name": self.name, "dtype": self.dtype.to_dict(), "required": self.required, "filterable": self.filterable}

    def clone(self) -> "_HyperGCol":
        return _HyperGCol(self.name, self.dtype.clone(), self.required, self.filterable)


class HyperG(LogicalType):
    """Structured hypergraph-like list payload logical type."""

    identifier = "hyper-g"

    @classmethod
    def from_dict(cls, spec: dict, *, resolver: ModuleResolver | None = None) -> "HyperG":
        """Build a HyperG logical type from a serialized descriptor.

        Args:
            spec (dict): Serialized descriptor with ``fields`` or legacy
                ``value`` key.
            resolver (ModuleResolver | None): Context-owned module resolver
                used when resolving nested logical types.

        Returns:
            HyperG: Configured HyperG logical type.
        """
        fields = spec.get("fields", spec.get("value"))
        if isinstance(fields, str):
            fields = [part.strip() for part in fields.split(",") if part.strip()]
        return cls(fields, resolver=resolver)

    def __init__(self, fields: Any, *, resolver: ModuleResolver | None = None):
        """Configure HyperG column descriptors from a field specification.

        Args:
            fields (Any): Mapping, sequence, or slice-based field descriptors.
            resolver (ModuleResolver | None): Context-owned module resolver
                used when resolving nested logical types.

        Returns:
            None: This constructor does not return a value.
        """
        if fields is None:
            raise TypeError("HyperG requires explicit fields; use HyperG['name': ShortText, ...]")
        self._module_resolver = resolver
        self.fields = tuple(self._field_list(fields, self._module_resolver))
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"HyperG fields must be unique, got {names!r}")

    def __class_getitem__(cls, item: Any) -> "HyperG":
        return cls(item)

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return field names in stored HyperG column order."""
        return tuple(field.name for field in self.fields)

    @property
    def filterable_fields(self) -> tuple[_HyperGCol, ...]:
        """Return HyperG columns that can participate in query filters."""
        return tuple(field for field in self.fields if field.filterable)

    def get(self, name: str) -> _HyperGCol:
        """Return one HyperG column by name.

        Args:
            name (str): Field name to resolve.

        Returns:
            _HyperGCol: Matching column descriptor.

        Raises:
            KeyError: If no field with ``name`` is configured.
            ValueError: If ``name`` is not a valid field identifier.
        """
        clean_name = check_field_name(name)
        for field in self.fields:
            if field.name == clean_name:
                return field
        raise KeyError(f"HyperG has no field {name!r}; known={self.field_names!r}")

    def validate(self, value: Any) -> list:
        """Validate a HyperG list payload against the configured fields.

        Args:
            value (Any): List of mapping or tuple mini-rows, or ``None``.

        Returns:
            list: Normalized list of field-validated mini-row mappings.

        Raises:
            ValueError: If ``value`` is not list-like or a mini-row fails field
                validation.
        """
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise ValueError("HyperG expects a list")
        return [self._item(item) for item in value]

    def filter_value(self, op: str, value: Any) -> Any:
        """Normalize HyperG query predicates for supported filter operations.

        Args:
            op (str): Filter operation identifier such as ``exists``, ``any``,
                or ``all``.
            value (Any): Mapping of field predicates for existential or
                universal HyperG filters.

        Returns:
            Any: Normalized predicate mapping, or the unchanged operand for
            unsupported operations.

        Raises:
            TypeError: If a predicate payload is not a mapping.
            ValueError: If a predicate is empty or a field filter fails.
        """
        if op == "all":
            return self._all_filter_value(value)
        if op not in ("exists", "any"):
            return value
        return self._predicate_filter_value(op, value)

    def _predicate_filter_value(self, op: str, value: Any, allow_empty: bool = False) -> dict:
        if not isinstance(value, Mapping):
            raise TypeError(f"HyperG {op} filter expects a mapping")
        out = {}
        for name, condition in value.items():
            field = self.get(str(name))
            predicates = condition if isinstance(condition, Mapping) else {"eq": condition}
            out[field.name] = {
                normalize_op(predicate_op, resolver=self._module_resolver): field.filter_value(
                    normalize_op(predicate_op, resolver=self._module_resolver),
                    predicate_value,
                )
                for predicate_op, predicate_value in predicates.items()
            }
        if not out and not allow_empty:
            raise ValueError(f"HyperG {op} filter needs at least one field predicate")
        return out

    def _all_filter_value(self, value: Any) -> dict:
        if not isinstance(value, Mapping):
            raise TypeError("HyperG all filter expects a mapping")
        if "where" in value or "then" in value or "predicates" in value:
            where = value.get("where") or {}
            then = value.get("then", value.get("predicates", {}))
        else:
            where = {}
            then = value
        if not isinstance(where, Mapping):
            raise TypeError("HyperG all filter where expects a mapping")
        if not isinstance(then, Mapping):
            raise TypeError("HyperG all filter then expects a mapping")
        return {"where": self._predicate_filter_value("all.where", where, allow_empty=True), "then": self._predicate_filter_value("all", then)}

    def to_dict(self) -> dict:
        """Return a JSON-compatible HyperG descriptor.

        Args:
            None.

        Returns:
            dict: Descriptor containing ``type`` and serialized ``fields``.
        """
        return {"type": type(self).identifier, "fields": [field.to_dict() for field in self.fields]}

    def clone(self) -> "HyperG":
        """Return a new HyperG with cloned field descriptors.

        Args:
            None.

        Returns:
            HyperG: Independent HyperG logical type with cloned columns.
        """
        return HyperG([field.clone() for field in self.fields], resolver=self._module_resolver)

    def _item(self, item: Any) -> dict[str, Any]:
        if isinstance(item, Mapping):
            data = dict(item)
        elif isinstance(item, (list, tuple)):
            if len(item) != len(self.fields):
                raise ValueError(f"HyperG tuple items expect {len(self.fields)} values")
            data = dict(zip(self.field_names, item))
        else:
            raise ValueError("HyperG items expect mappings or tuples")
        unknown = sorted(set(data) - set(self.field_names))
        if unknown:
            raise ValueError(f"HyperG item has unknown fields {unknown!r}")
        return {field.name: field.validate(data.get(field.name)) for field in self.fields if field.name in data or field.required}

    def _field_list(self, fields: Any, resolver: ModuleResolver | None) -> list[_HyperGCol]:
        if isinstance(fields, Mapping):
            return [self._field_from_any((name, spec), resolver) for name, spec in fields.items()]
        items = fields if isinstance(fields, (list, tuple)) else (fields,)
        return [self._field_from_any(item, resolver) for item in items]

    def _field_from_any(self, item: Any, resolver: ModuleResolver | None) -> _HyperGCol:
        if isinstance(item, _HyperGCol):
            return item.clone()
        if isinstance(item, str):
            return _HyperGCol(check_field_name(item), ShortText())
        if isinstance(item, slice):
            return self._field_from_parts(item.start, item.stop or ShortText(), item.step, resolver)
        if isinstance(item, Mapping):
            name = item.get("name")
            if name is None and len(item) == 1:
                name, item = next(iter(item.items()))
                return self._field_from_parts(name, item, resolver=resolver)
            return self._field_from_parts(name, item.get("dtype", item.get("type", ShortText())), item, resolver)
        if isinstance(item, tuple) and len(item) in (2, 3):
            return self._field_from_parts(item[0], item[1], item[2] if len(item) == 3 else None, resolver)
        raise TypeError(f"Invalid HyperG field descriptor {item!r}")

    def _field_from_parts(
        self,
        name: Any,
        dtype: Any = None,
        opts: Any = None,
        resolver: ModuleResolver | None = None,
    ) -> _HyperGCol:
        if name is None:
            raise ValueError("HyperG field descriptor needs a name")
        options = opts if isinstance(opts, Mapping) else dtype if isinstance(dtype, Mapping) else {}
        dtype_spec = dtype
        if isinstance(dtype, Mapping):
            dtype_spec = {key: value for key, value in dtype.items() if key not in ("name", "required", "filterable")}
        dtype = self._coerce_dtype(dtype_spec if dtype_spec is not None else options.get("dtype", options.get("type", ShortText())), resolver)
        filterable = bool(options.get("filterable", type(dtype).identifier not in _DATA_TYPES))
        return _HyperGCol(check_field_name(str(name)), dtype, bool(options.get("required", True)), filterable)

    def _coerce_dtype(self, dtype: Any, resolver: ModuleResolver | None) -> LogicalType:
        if isinstance(dtype, Mapping):
            return logical_type_from_any(dtype, resolver=resolver)
        if isinstance(dtype, str):
            return get_logical_type(dtype, resolver=resolver)()
        return coerce_type(dtype)
