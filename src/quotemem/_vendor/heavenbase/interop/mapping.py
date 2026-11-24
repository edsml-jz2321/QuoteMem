"""Shared inference and Entity construction for interop converters."""

from __future__ import annotations

from inspect import Parameter, signature
import re

__all__ = [
    "InteropTypeInferer",
    "TypeInferer",
    "build_set",
    "registered_interop_type_inferers",
    "register_interop_type_inferer",
]

from heavenbase.utils import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, dataclass
from heavenbase.utils.registry import identifier_key

from ..entity import Entity, FieldSpec, field as make_field
from ..modules import ModuleResolver, ModuleService, module_resolver, python_executable_ref
from ..modules.base import ResolverCache
from ..types import (
    Array,
    Boolean,
    Float,
    HyperG,
    Identifier,
    Integer,
    Json,
    LogicalType,
    LongText,
    MediumText,
    ShortText,
    Timestamp,
    Vector,
    coerce_type,
    get_logical_type,
    logical_type_from_any,
)
from ..utils.ids import check_entity_id, check_field_name
from ..utils.values import class_name
from .base import InteropIssue, InteropReport, InteropSet
from .rows import is_missing, source_row

TypeInferer = Callable[..., Optional[LogicalType]]
_INFERER_CACHE = ResolverCache["InteropTypeInferer"]()


@dataclass
class InteropTypeInferer:
    """Sample-based logical-type inference rule for interop imports."""

    fn: TypeInferer
    identifier: str = ""
    priority: Optional[int] = None

    def __post_init__(self) -> None:
        """Normalize the inferer identifier and validate the callable.

        Args:
            None.

        Returns:
            None: This method mutates the inferer metadata in place.

        Raises:
            TypeError: If ``fn`` is not callable.
            ValueError: If the identifier is blank.
        """
        if not callable(self.fn):
            raise TypeError("Interop type inferer must be callable")
        self.identifier = _inferer_key(self.identifier or _inferer_identifier(self.fn))

    @classmethod
    def load(cls, raw: TypeInferer | "InteropTypeInferer") -> "InteropTypeInferer":
        """Return an inferer spec from a callable or existing spec.

        Args:
            raw: Existing ``InteropTypeInferer`` or callable inferer.

        Returns:
            InteropTypeInferer: Normalized inference rule.

        Raises:
            TypeError: If ``raw`` is not callable and not an inferer spec.
        """
        if isinstance(raw, cls):
            return raw
        if callable(raw):
            return cls(raw)
        raise TypeError("Interop type inferer must be callable or an InteropTypeInferer")

    def infer(
        self,
        source: str,
        values: List[Any],
        hint: Any,
        report: InteropReport,
        *,
        resolver: ModuleResolver | None = None,
    ) -> Optional[LogicalType]:
        """Infer one logical type from sampled source values.

        Args:
            source: Source column name being inferred.
            values: Non-missing sample values for that source column.
            hint: Optional source dtype or type hint supplied by the importer.
            report: Active interop report for warnings or errors.
            resolver: Context-owned module resolver used by nested inference.
                When omitted, the default Context is selected.

        Returns:
            LogicalType | None: Inferred logical type, or ``None`` to let the
            next inferer decide.
        """
        dtype = _invoke_inferer(self.fn, source, values, hint, report, module_resolver(resolver))
        return None if dtype is None else coerce_type(dtype)

    def __call__(
        self,
        source: str,
        values: List[Any],
        hint: Any,
        report: InteropReport,
        *,
        resolver: ModuleResolver | None = None,
    ) -> Optional[LogicalType]:
        """Infer one logical type using callable syntax.

        Args:
            source: Source column name being inferred.
            values: Non-missing sample values for that source column.
            hint: Optional source dtype or type hint supplied by the importer.
            report: Active interop report for warnings or errors.
            resolver: Context-owned module resolver used by nested inference.
                When omitted, the default Context is selected.

        Returns:
            LogicalType | None: Inferred logical type, or ``None`` to let the
            next inferer decide.
        """
        return self.infer(source, values, hint, report, resolver=resolver)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible inferer metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Identifier, callable reference, and priority.
        """
        return {"identifier": self.identifier, "fn": _callable_ref(self.fn), "priority": self.priority}


def register_interop_type_inferer(
    inferer: TypeInferer | InteropTypeInferer,
    *,
    first: bool = False,
    replace: bool = False,
    resolver: ModuleResolver | None = None,
) -> InteropTypeInferer:
    """Publish a sample-based interop logical-type inferer record.

    Args:
        inferer: Callable or ``InteropTypeInferer`` that can infer a
            ``LogicalType`` from source samples.
        first: Whether this inferer should run before already registered
            inferers.
        replace: Whether an existing inferer with the same identifier may be
            replaced.
        resolver: Writable Context-owned module resolver. When omitted, the
            default Context is selected.

    Returns:
        InteropTypeInferer: Registered inferer spec.

    Raises:
        TypeError: If ``inferer`` is not callable and not an inferer spec.
        ValueError: If the identifier is blank or already registered.
    """
    chosen = module_resolver(resolver)
    if not isinstance(chosen, ModuleService):
        raise TypeError("Interop inferer registration requires a writable ModuleService")
    spec = InteropTypeInferer.load(inferer)
    if spec.priority is None:
        spec = InteropTypeInferer(spec.fn, spec.identifier, _next_inferer_priority(first, chosen))
    service = chosen
    found = service.inspect("interop_inferer", spec.identifier)
    if found:
        existing = _inferer_from_row(found[0], service)
        if existing == spec and replace:
            return existing
        if not replace:
            raise ValueError(f"Interop type inferer {spec.identifier!r} is already registered")
        coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if coordinate != f"programmatic/interop_inferer/{spec.identifier}":
            raise ValueError(f"Interop type inferer {spec.identifier!r} belongs to {coordinate!r} and cannot be replaced")
    service.register(
        "interop_inferer",
        spec.identifier,
        {
            "name": spec.identifier,
            "desc": _doc_summary(spec.fn),
            "source": "inline",
            "target": "definition",
            "active": True,
            "meta": {
                "schema_version": 1,
                "definition": {
                    "fn": python_executable_ref(spec.fn, label=f"interop type inferer {spec.identifier!r}"),
                    "priority": int(spec.priority),
                },
            },
        },
    )
    return _load_inferer(spec.identifier, service)


def registered_interop_type_inferers(*, resolver: ModuleResolver | None = None) -> Dict[str, InteropTypeInferer]:
    """Return registered interop inferers keyed by identifier.

    Args:
        resolver: Context-owned module resolver. When omitted, the default
            Context is selected.

    Returns:
        dict[str, InteropTypeInferer]: Snapshot of the inferer registry.
    """
    return {inferer.identifier: inferer for inferer in _type_inferers(module_resolver(resolver))}


def _type_inferers(resolver: ModuleResolver) -> tuple[InteropTypeInferer, ...]:
    """Return a snapshot of registered interop inferers."""
    inferers = tuple(_inferer_from_row(row, resolver) for row in resolver.inspect("interop_inferer"))
    return tuple(sorted(inferers, key=lambda item: (item.priority if item.priority is not None else 0, item.identifier)))


def _next_inferer_priority(first: bool, resolver: ModuleResolver) -> int:
    """Return the next inference priority value."""
    priorities = [int(inferer.priority or 0) for inferer in _type_inferers(resolver)]
    if not priorities:
        return -1 if first else 0
    return min(priorities) - 1 if first else max(priorities) + 1


def _load_inferer(identifier: str, resolver: ModuleResolver) -> InteropTypeInferer:
    """Materialize one interop inferer from its authoritative record."""
    rows = resolver.inspect("interop_inferer", _inferer_key(identifier))
    if not rows:
        raise KeyError(identifier)
    return _inferer_from_row(rows[0], resolver)


def _inferer_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> InteropTypeInferer:
    """Materialize and cache one interop inferer inspection row."""

    def build(current_row: Mapping[str, Any]) -> InteropTypeInferer:
        identifier = str(current_row["identifier"])
        raw = resolver.resolve("interop_inferer", identifier)
        if not isinstance(raw, Mapping):
            raise TypeError(f"Interop type inferer {identifier!r} definition must be a mapping")
        reference = raw.get("fn")
        if not isinstance(reference, Mapping):
            raise TypeError(f"Interop type inferer {identifier!r} fn must be an executable reference")
        return InteropTypeInferer(
            resolver.resolve_executable(reference, owner=current_row),
            identifier,
            int(raw.get("priority", 0)),
        )

    return _INFERER_CACHE.get(resolver, row, build)


def _invoke_inferer(
    inferer: TypeInferer,
    source: str,
    values: List[Any],
    hint: Any,
    report: InteropReport,
    resolver: ModuleResolver,
) -> Optional[LogicalType]:
    """Call an inferer, injecting the resolver when its contract accepts it."""
    try:
        parameters = signature(inferer).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    accepts_resolver = any(parameter.name == "resolver" or parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters)
    if accepts_resolver:
        return inferer(source, values, hint, report, resolver=resolver)
    return inferer(source, values, hint, report)


def _doc_summary(fn: Any) -> str:
    """Return the first documentation line for one inferer callable."""
    lines = str(getattr(fn, "__doc__", "") or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def _inferer_key(value: str) -> str:
    """Return the registry key for one interop inferer identifier."""
    return identifier_key(value, "interop type inferer identifier")


def _callable_ref(fn: Any) -> str:
    """Return a stable module-qualified callable reference."""
    return f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', 'callable'))}"


def _inferer_identifier(fn: Any) -> str:
    """Return the default registry identifier for one inferer callable."""
    identifier = re.sub(r"[^a-z0-9]+", "-", _callable_ref(fn).lower()).strip("-")
    if not identifier or not identifier[0].isalpha():
        identifier = f"inferer-{identifier}"
    return identifier


def build_set(
    source: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    entity_id: str,
    id: Optional[str] = None,
    hints: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    rename: Optional[Mapping[str, str]] = None,
    desc: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    field_meta: Optional[Mapping[str, Mapping[str, Any]]] = None,
    required: Optional[Mapping[str, bool]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    issues: Optional[Sequence[InteropIssue]] = None,
    strict: bool = False,
    resolver: ModuleResolver | None = None,
) -> InteropSet:
    """Build an InteropSet from source rows and inferred field specs.

    Args:
        source (str): External source kind recorded in diagnostics.
        rows (Iterable[Mapping[str, Any]]): Source row mappings.
        entity_id (str): Target entity identifier for the generated class.
        id (str | None): Source column mapped to ``object_id``. Defaults to
            ``"id"`` when present.
        hints (Mapping[str, Any] | None): Per-column dtype or type hints from
            the importer.
        overrides (Mapping[str, Any] | None): Per-field dtype or ``FieldSpec``
            overrides keyed by source or target name.
        rename (Mapping[str, str] | None): Source-to-target field renames.
        desc (str | None): Entity description for the generated class.
        tags (Sequence[str] | None): Entity tags for the generated class.
        field_meta (Mapping[str, Mapping[str, Any]] | None): Per-field
            description, examples, and inference hints.
        required (Mapping[str, bool] | None): Per-column required flags keyed
            by source or target name.
        defaults (Mapping[str, Any] | None): Per-column default values keyed by
            source or target name.
        issues (Sequence[InteropIssue] | None): Pre-existing conversion
            diagnostics to attach before inference.
        strict (bool): Promote warnings to errors. Defaults to ``False``.
        resolver (ModuleResolver | None): Context-owned module resolver. When
            omitted, the default Context is selected.

    Returns:
        InteropSet: Staged entity, normalized rows, and conversion report.
    """
    chosen = module_resolver(resolver)
    entity_id = check_entity_id(entity_id)
    report = InteropReport(source, strict=strict)
    rename = {str(k): str(v) for k, v in (rename or {}).items()}
    hints = dict(hints or {})
    overrides = dict(overrides or {})
    field_meta = dict(field_meta or {})
    required = dict(required or {})
    defaults = dict(defaults or {})
    for issue in issues or []:
        report.add(issue.level, issue.code, issue.path, issue.message, issue.hint)
    source_rows = [source_row(row) for row in rows]
    cols = _columns(source_rows, hints, overrides, rename)
    for src in rename:
        if src not in cols:
            report.add("error", "unknown_rename", src, f"Rename source column {src!r} is not present")
    id_source = _id_source(id, cols, rename, report)
    if id_source is None:
        report.add("warning", "missing_id", "object_id", "No object_id source column provided; writes need object_id or a catalog.name derive")
    specs: Dict[str, FieldSpec] = {}
    source_to_target: Dict[str, str] = {}
    for col in cols:
        target = "object_id" if col == id_source else rename.get(col, col)
        try:
            target = check_field_name(target)
        except ValueError as exc:
            report.add("error", "invalid_field_name", col, str(exc), "Use rename={source: target} to choose a Python-safe field name")
            continue
        if target in specs:
            report.add("error", "duplicate_field", target, f"Source column {col!r} maps to duplicate field {target!r}", "Adjust rename or id mapping")
            continue
        source_to_target[col] = target
        values = [row.get(col) for row in source_rows if col in row and not is_missing(row.get(col))]
        spec = _spec_for(
            col,
            target,
            values,
            hints.get(col),
            overrides.get(col, overrides.get(target)),
            _meta(field_meta, col, target),
            report,
            strict,
            chosen,
        )
        if target == "object_id":
            spec = _clone_spec(spec)
            spec.dtype = Identifier()
            spec.pk()
        elif required.get(col, required.get(target)):
            spec.required()
        elif spec.required_flag is None:
            spec.optional()
        default_key = col if col in defaults else target
        if default_key in defaults:
            spec.default(defaults[default_key])
        specs[target] = spec
        report.fields[target] = {"source": col, "dtype": spec.dtype.to_dict(), "required": spec.required_value, "desc": spec.desc_text}
    normalized = []
    for row in source_rows:
        out = {target: row[col] for col, target in source_to_target.items() if col in row and not is_missing(row.get(col))}
        normalized.append(out)
    report.n_rows = len(normalized)
    entity = _entity(entity_id, specs, desc=desc or "", tags=tags or [])
    return InteropSet(entity, normalized, report, source)


def _columns(rows: List[Mapping[str, Any]], hints: Mapping[str, Any], overrides: Mapping[str, Any], rename: Mapping[str, str]) -> List[str]:
    cols = []
    for name in hints:
        if name not in cols:
            cols.append(str(name))
    for row in rows:
        for name in row:
            if name not in cols:
                cols.append(str(name))
    source_names = set(cols)
    rename_targets = set(rename.values())
    for name in overrides:
        name = str(name)
        if name in rename_targets and name not in source_names:
            continue
        if name not in cols:
            cols.append(name)
    return cols


def _id_source(id: Optional[str], cols: Sequence[str], rename: Mapping[str, str], report: InteropReport) -> Optional[str]:
    if id is None:
        return "id" if "id" in cols else None
    value = str(id)
    if value in cols:
        return value
    matches = [src for src, target in rename.items() if target == value and src in cols]
    if len(matches) == 1:
        return matches[0]
    report.add("error", "unknown_id", value, f"ID source column {value!r} is not present", "Pass the source column name or a renamed target")
    return value


def _meta(field_meta: Mapping[str, Mapping[str, Any]], source: str, target: str) -> Mapping[str, Any]:
    return field_meta.get(source) or field_meta.get(target) or {}


def _entity(entity_id: str, specs: Mapping[str, FieldSpec], *, desc: str = "", tags: Sequence[str] = ()) -> type:
    namespace = {"identifier": entity_id, "__doc__": desc, "__tags__": list(tags), "__module__": "heavenbase.interop.dynamic"}
    namespace.update(specs)
    return type(class_name(entity_id), (Entity,), namespace)


def _spec_for(
    source: str,
    target: str,
    values: List[Any],
    hint: Any,
    override: Any,
    meta: Mapping[str, Any],
    report: InteropReport,
    strict: bool,
    resolver: ModuleResolver,
) -> FieldSpec:
    spec = (
        _clone_spec(_override_spec(override, resolver))
        if override is not None
        else make_field(_meta_dtype(hint, meta) or _infer_dtype(source, values, hint, report, strict, resolver))
    )
    if meta:
        spec.describe(
            str(meta.get("desc") or meta.get("description") or spec.desc_text),
            tags=list(meta.get("tags") or spec.tag_values),
            examples=list(meta.get("examples") or spec.examples),
        )
    return spec


def _meta_dtype(hint: Any, meta: Mapping[str, Any]) -> Optional[LogicalType]:
    max_length = meta.get("max_length")
    if max_length is None or not _looks_text_hint(hint):
        return None
    max_length = int(max_length)
    if max_length <= Identifier().max_chars:
        return ShortText()
    if max_length <= ShortText().max_chars:
        return ShortText()
    if max_length <= MediumText().max_chars:
        return MediumText()
    return LongText()


def _looks_text_hint(hint: Any) -> bool:
    return hint is str or any(token in str(hint).lower() for token in ("string", "str", "utf8", "large_string"))


def _override_spec(value: Any, resolver: ModuleResolver) -> FieldSpec:
    if isinstance(value, FieldSpec):
        return value
    return make_field(_dtype_from_any(value, resolver))


def _clone_spec(spec: FieldSpec) -> FieldSpec:
    out = FieldSpec(
        spec.dtype.clone(), spec.is_pk, spec.required_flag, spec.default_value, spec.has_default, spec.desc_text, list(spec.tag_values), list(spec.examples)
    )
    out.stores = list(spec.stores)
    out.computes = list(spec.computes)
    return out


def _dtype_from_any(value: Any, resolver: ModuleResolver) -> LogicalType:
    if isinstance(value, LogicalType):
        return value.clone()
    if isinstance(value, type) and issubclass(value, LogicalType):
        return value()
    if isinstance(value, str):
        return get_logical_type(value, resolver=resolver)()
    if isinstance(value, Mapping):
        return logical_type_from_any(value, resolver=resolver)
    return coerce_type(value)


def _infer_dtype(
    source: str,
    values: List[Any],
    hint: Any,
    report: InteropReport,
    strict: bool,
    resolver: ModuleResolver,
) -> LogicalType:
    hinted = _hint_dtype(hint)
    if hinted is not None:
        if type(hinted).identifier == "long-text":
            report.add("warning", "assumed_long_text", source, "String length is unspecified; using LongText")
        return hinted
    sample = [value for value in values if not is_missing(value)]
    if not sample:
        report.add("warning", "empty_column", source, "No non-missing values available; using LongText")
        return LongText()
    for inferer in _type_inferers(resolver):
        dtype = inferer.infer(source, sample, hint, report, resolver=resolver)
        if dtype is not None:
            return dtype
    report.add("warning", "mixed_object", source, "Mixed or unsupported values import as Json")
    return Json()


def _infer_array_item(
    source: str,
    values: List[Any],
    report: InteropReport,
    strict: bool,
    resolver: ModuleResolver,
) -> LogicalType:
    items = [item for value in values for item in value if not is_missing(item)]
    if not items:
        report.add("warning", "empty_array", source, "Array values are empty; using Array[LongText]")
        return LongText()
    return _infer_dtype(f"{source}[]", items, None, report, strict, resolver)


def _hint_dtype(hint: Any) -> Optional[LogicalType]:
    if hint is None:
        return None
    origin = getattr(hint, "__origin__", None)
    args = list(getattr(hint, "__args__", []) or [])
    if args and "union" in str(origin).lower():
        non_null = [arg for arg in args if arg is not type(None)]
        return _hint_dtype(non_null[0]) if non_null else None
    if origin in (list, List) or str(origin) == "list":
        return Array(_hint_dtype(args[0]) or LongText())
    if origin in (dict, Dict) or str(origin) == "dict":
        return Json()
    if hint in (bool, "bool") or "bool" in str(hint).lower():
        return Boolean()
    text = str(hint).lower()
    if hasattr(hint, "list_size") and hint.list_size is not None and "float" in text:
        return Vector(int(hint.list_size))
    if hint is int or ("int" in text and "point" not in text):
        return Integer()
    if hint is float or any(token in text for token in ("float", "double")):
        return Float()
    if hint is str or any(token in text for token in ("string", "str", "utf8", "large_string")):
        return LongText()
    return None


def _bool_inferer(source: str, values: List[Any], hint: Any, report: InteropReport) -> Optional[LogicalType]:
    return Boolean() if all(isinstance(value, bool) for value in values) else None


def _int_inferer(source: str, values: List[Any], hint: Any, report: InteropReport) -> Optional[LogicalType]:
    return Integer() if all(isinstance(value, int) and not isinstance(value, bool) for value in values) else None


def _float_inferer(source: str, values: List[Any], hint: Any, report: InteropReport) -> Optional[LogicalType]:
    return Float() if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values) else None


def _text_inferer(source: str, values: List[Any], hint: Any, report: InteropReport) -> Optional[LogicalType]:
    if all(isinstance(value, str) for value in values):
        report.add("warning", "assumed_long_text", source, "String length is unspecified; using LongText")
        return LongText()
    return None


def _array_inferer(
    source: str,
    values: List[Any],
    hint: Any,
    report: InteropReport,
    *,
    resolver: ModuleResolver,
) -> Optional[LogicalType]:
    return Array(_infer_array_item(source, values, report, False, resolver)) if all(isinstance(value, (list, tuple)) for value in values) else None


def _json_inferer(source: str, values: List[Any], hint: Any, report: InteropReport) -> Optional[LogicalType]:
    if all(isinstance(value, Mapping) for value in values):
        report.add("warning", "assumed_json", source, "Nested mappings import as Json")
        return Json()
    return None
