"""Entity field declaration helpers."""

from __future__ import annotations

__all__ = [
    "ComputeHook",
    "DeriveSpec",
    "FieldSpec",
    "QueryComputeHook",
    "field",
]

from heavenbase.utils import Any, Callable, Dict, List, Optional, dataclass, deepcopy, field as dcfield

from ..modules import ModuleResolver
from ..schema.model import FieldSchema
from ..placement import StorageRule
from ..placement.index import FieldIndexRule, FieldUniqueRule
from ..strategies import InlineColumn
from ..types import Categorical, LogicalType, coerce_type
from ..utils.ids import check_field_name
from ..filters.ops import normalize_op

_SUPPORTED_QUERY_COMPUTE_OPS = {"near"}


@dataclass
class ComputeHook:
    """Computed-field callback plus the input field names it reads."""

    fn: Callable[..., Any]
    inputs: List[str] = dcfield(default_factory=list)


@dataclass
class QueryComputeHook:
    """Query-time callback plus the query operations it can normalize."""

    fn: Callable[[Any], Any]
    ops: List[str] = dcfield(default_factory=lambda: ["near"])


@dataclass
class DeriveSpec:
    """Entity or Catalog derivation declaration.

    Args:
        target: Native field name or ``catalog.<field>`` projection target.
    """

    target: str
    computes: List[ComputeHook] = dcfield(default_factory=list)

    def compute(self, fn: Callable[..., Any], *, inputs: Optional[List[str]] = None) -> "DeriveSpec":
        """Attach the deterministic compute hook for this derivation.

        Args:
            fn: Callable used to derive the target value.
            inputs: Optional source field names. Without inputs the callable
                receives the whole row, except zero-argument callables are
                treated as constants.

        Returns:
            This ``DeriveSpec`` for fluent chaining.
        """
        self.computes = [ComputeHook(fn, list(inputs or []))]
        return self


@dataclass
class FieldSpec:
    """Declare one logical entity field before schema compilation."""

    dtype: LogicalType
    is_pk: bool = False
    required_flag: Optional[bool] = None
    default_value: Any = None
    has_default: bool = False
    desc_text: str = ""
    tag_values: List[str] = dcfield(default_factory=list)
    examples: List[Any] = dcfield(default_factory=list)
    stores: List[StorageRule] = dcfield(default_factory=list)
    computes: List[ComputeHook] = dcfield(default_factory=list)
    query_computes: List[QueryComputeHook] = dcfield(default_factory=list)
    index_rules: List[FieldIndexRule] = dcfield(default_factory=list)
    unique_rules: List[FieldUniqueRule] = dcfield(default_factory=list)

    def pk(self) -> "FieldSpec":
        """Mark this field as the primary key.

        Args:
            None.

        Returns:
            FieldSpec: This declaration for fluent chaining.
        """
        self.is_pk = True
        self.required_flag = True
        return self

    def required(self) -> "FieldSpec":
        """Mark this field as required.

        Args:
            None.

        Returns:
            FieldSpec: This declaration for fluent chaining.
        """
        self.required_flag = True
        return self

    def optional(self) -> "FieldSpec":
        """Mark this field as optional.

        Args:
            None.

        Returns:
            FieldSpec: This declaration for fluent chaining.
        """
        self.required_flag = False
        return self

    def default(self, value: Any) -> "FieldSpec":
        """Set the value used when writes omit this field.

        Args:
            value (Any): Default field value.

        Returns:
            FieldSpec: This declaration for fluent chaining.
        """
        self.default_value = value
        self.has_default = True
        return self

    def describe(self, desc: str = "", *, tags: Optional[List[str]] = None, examples: Optional[List[Any]] = None) -> "FieldSpec":
        """Replace human-facing description metadata.

        Args:
            desc (str): Short field description.
            tags (List[str] | None): Replacement discovery tags.
            examples (List[Any] | None): Replacement example values.

        Returns:
            FieldSpec: This declaration for fluent chaining.
        """
        self.desc_text = desc
        self.tag_values = list(tags or [])
        self.examples = list(examples or [])
        return self

    def desc(self, text: str) -> "FieldSpec":
        """Set only the short description text.

        Args:
            text (str): Short field description.

        Returns:
            FieldSpec: This declaration for fluent chaining.
        """
        return self.describe(text, tags=self.tag_values, examples=self.examples)

    def tags(self, *tags: str) -> "FieldSpec":
        """Append search and discovery tags.

        Args:
            *tags (str): Tag values in insertion order.

        Returns:
            FieldSpec: This declaration for fluent chaining.
        """
        self.tag_values.extend(tags)
        return self

    def example(self, *examples: Any) -> "FieldSpec":
        """Append agent and documentation examples.

        Args:
            *examples (Any): Example field values.

        Returns:
            FieldSpec: This declaration for fluent chaining.
        """
        self.examples.extend(examples)
        return self

    def store(
        self,
        *,
        to: Optional[str] = None,
        strategy: Any = InlineColumn,
        optional: bool = False,
        reason: str = "field",
        params: Optional[Dict[str, Any]] = None,
        resolver: ModuleResolver | None = None,
    ) -> "FieldSpec":
        """Add one physical storage placement for this field.

        Args:
            to: Backend instance name. ``None`` lets the workspace choose the default backend.
            strategy: Storage strategy class, instance, identifier, or alias.
            optional: Whether to skip this placement when the backend is absent.
            reason: Placement source stored in system metadata.
            params: Explicit per-field placement params (e.g. ``{"target": "cited"}`` for a GraphEdge
                endpoint). Normally carried implicitly on a parametrised ``strategy`` instance; this
                channel restores params when the manifest/dict path supplies the strategy as a bare
                identifier string.
            resolver: Context-owned module resolver used to normalize the
                strategy. The default Context is selected when omitted.

        Returns:
            FieldSpec: This declaration for fluent chaining.

        Examples:
            ``hb.field(hb.Vector[3]).store(to="vec", strategy=hb.VectorIndex)``
        """
        # StorageRule.__post_init__ overlays these explicit params on a strategy-instance's own params,
        # then normalizes the strategy to its identifier (so the normal instance path and this dict path agree).
        self.stores.append(
            StorageRule(
                backend=to,
                strategy=strategy,
                optional=optional,
                reason=reason,
                params=params or {},
                resolver=resolver,
            )
        )
        return self

    def index(self, enabled: bool = True, *, to: Optional[str] = None) -> "FieldSpec":
        """Control scalar lookup indexing without mixing in uniqueness.

        Args:
            enabled: ``True`` requests an index; ``False`` disables a provider
                default index for this field. Defaults to ``True``.
            to: Optional workspace backend instance targeted by the declaration.

        Returns:
            FieldSpec: This declaration for fluent chaining.

        Examples:
            ``hb.field(hb.ShortText).index()``
            ``hb.field(hb.ShortText).index(enabled=False)``
        """
        self.index_rules.append(FieldIndexRule(enabled, to))
        return self

    def unique(self, *, to: Optional[str] = None) -> "FieldSpec":
        """Require non-null values of this field to be unique.

        Args:
            to: Optional workspace backend instance that owns enforcement.

        Returns:
            FieldSpec: This declaration for fluent chaining.

        Examples:
            ``hb.field(hb.ShortText).unique()``
        """
        self.unique_rules.append(FieldUniqueRule(to))
        return self

    def compute(self, fn: Callable[..., Any], *, inputs: Optional[List[str]] = None) -> "FieldSpec":
        """Attach a materialization-time compute hook.

        Args:
            fn (Callable[..., Any]): Deterministic field computation.
            inputs (List[str] | None): Optional source field names.

        Returns:
            FieldSpec: This declaration for fluent chaining.
        """
        self.computes.append(ComputeHook(fn, list(inputs or [])))
        return self

    def query_compute(
        self,
        fn: Callable[[Any], Any],
        *,
        ops: Optional[List[str]] = None,
        resolver: ModuleResolver | None = None,
    ) -> "FieldSpec":
        """Attach a query-time compute hook for supported query operations.

        Args:
            fn: Callable that receives the raw query operation value and returns
                a value valid for this field's logical type.
            ops: Query operations this hook supports. Defaults to ``["near"]``.
            resolver: Context-owned module resolver used to normalize operation
                names. The default Context is selected when omitted.

        Returns:
            This ``FieldSpec`` for fluent chaining.

        Raises:
            ValueError: If an operation is not supported by query-time compute.

        Examples:
            ``hb.field(hb.Vector[3]).query_compute(embed_query)``
        """
        normalized = [normalize_op(op, resolver=resolver) for op in (ops or ["near"])]
        if not normalized:
            raise ValueError("query_compute requires at least one query operation")
        unsupported = sorted({op for op in normalized if op not in _SUPPORTED_QUERY_COMPUTE_OPS})
        if unsupported:
            allowed = ", ".join(sorted(_SUPPORTED_QUERY_COMPUTE_OPS))
            got = ", ".join(unsupported)
            raise ValueError(f"query_compute currently supports only: {allowed}; got {got}")
        self.query_computes.append(QueryComputeHook(fn, normalized))
        return self

    def choices(self, *values: Any, base: Optional[str] = None) -> "FieldSpec":
        """Constrain this field to categorical choices.

        Args:
            *values: Allowed values, or one list/tuple of allowed values.
            base: Optional categorical base. Defaults to ``"integer"`` for
                ``Integer`` fields and ``"string"`` otherwise.

        Returns:
            This ``FieldSpec`` with its dtype replaced by ``Categorical``.

        Raises:
            ValueError: If values are empty, duplicated, or incompatible with
                the selected base.

        Examples:
            ``hb.field(hb.ShortText).choices("todo", "done")``
            ``hb.field(hb.Integer).choices(1, 2, 3)``
        """
        items = values[0] if len(values) == 1 and isinstance(values[0], (list, tuple)) else values
        identifier = self.dtype.identifier if isinstance(self.dtype, type) else type(self.dtype).identifier
        self.dtype = Categorical(items, base or ("integer" if identifier == "integer" else "string"))
        return self

    @property
    def required_value(self) -> bool:
        """Return whether materialization must produce this field.

        Returns:
            bool: Whether the declaration is required or primary-key owned.
        """
        return bool(self.required_flag) or self.is_pk

    def to_schema(self, name: str) -> FieldSchema:
        """Compile this declaration into an isolated field schema snapshot.

        Args:
            name (str): Entity field name assigned to this declaration.

        Returns:
            FieldSchema: Passive compiled field metadata.

        Raises:
            ValueError: If ``name`` is not a valid field name.
        """
        return FieldSchema(
            check_field_name(name),
            self.dtype.clone(),
            self.is_pk,
            self.required_value,
            deepcopy(self.default_value),
            self.has_default,
            self.desc_text,
            list(self.tag_values),
            deepcopy(self.examples),
        )

    def clone(self) -> "FieldSpec":
        """Return an independent copy of this authored declaration.

        Args:
            None.

        Returns:
            FieldSpec: Copy with isolated type, metadata, rules, and hook lists.
        """
        out = FieldSpec(
            self.dtype.clone(),
            self.is_pk,
            self.required_flag,
            deepcopy(self.default_value),
            self.has_default,
            self.desc_text,
            list(self.tag_values),
            deepcopy(self.examples),
        )
        out.stores = [
            StorageRule(
                rule.backend,
                rule.strategy,
                deepcopy(rule.strategy_options),
                rule.optional,
                rule.reason,
                deepcopy(rule.params),
                resolver=rule._module_resolver,
            )
            for rule in self.stores
        ]
        out.computes = [ComputeHook(hook.fn, list(hook.inputs)) for hook in self.computes]
        out.query_computes = [QueryComputeHook(hook.fn, list(hook.ops)) for hook in self.query_computes]
        out.index_rules = [FieldIndexRule(rule.enabled, rule.backend) for rule in self.index_rules]
        out.unique_rules = [FieldUniqueRule(rule.backend) for rule in self.unique_rules]
        return out


def field(dtype: Any, *, desc: str = "", tags: Optional[List[str]] = None, examples: Optional[List[Any]] = None) -> FieldSpec:
    """Create a field declaration for an ``Entity`` class.

    Args:
        dtype: Logical type class, instance, or shorthand such as
            ``hb.Vector[768]``.
        desc: Human-facing field description.
        tags: Optional discovery/search tags.
        examples: Optional example values for docs and agents.

    Returns:
        A ``FieldSpec`` ready for chaining with ``required()``, ``default()``,
        ``store()``, ``index()``, ``unique()``, or ``compute()``.
    """
    return FieldSpec(coerce_type(dtype)).describe(desc, tags=tags, examples=examples)
