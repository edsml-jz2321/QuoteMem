"""SQL dialect capability value contract."""

from __future__ import annotations

__all__ = [
    "SQL_DEFAULT_NATIVE_AGGREGATE_OPS",
    "SQL_DEFAULT_NATIVE_OPS",
    "SQL_JSON_AGG_STYLE_ARRAYAGG",
    "SQL_JSON_AGG_STYLE_GROUP_ARRAY",
    "SQL_JSON_AGG_STYLE_POSTGRES_ORDERED",
    "SQL_JSON_OBJECT_STYLE_BUILD_OBJECT",
    "SQL_JSON_OBJECT_STYLE_OBJECT",
    "SQL_JSON_PATH_STYLE_JSON1",
    "SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH",
    "SQL_REGEX_STYLE_POSTGRES_TILDE",
    "SQL_REGEX_STYLE_REGEXP_LIKE",
    "SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE",
    "SQL_WILDCARD_STYLE_GLOB",
    "SQL_WILDCARD_STYLE_LIKE",
    "SQLDialectSpec",
]

from types import MappingProxyType

from heavenbase.utils import Any, Callable, Dict, Mapping, dataclass, field

from ...utils.ids import check_id

SQL_DEFAULT_NATIVE_OPS = (
    "eq",
    "ne",
    "lt",
    "lte",
    "gt",
    "gte",
    "match",
    "like",
    "ilike",
    "contains",
    "starts_with",
    "ends_with",
    "in",
    "regex",
    "wildcard",
)
SQL_DEFAULT_NATIVE_AGGREGATE_OPS = ("count", "min", "max")
SQL_REGEX_STYLE_REGEXP_LIKE = "regexp_like"
SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE = "regexp_like_case_sensitive"
SQL_REGEX_STYLE_POSTGRES_TILDE = "postgres_tilde"
SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH = "duckdb_tilde_search"
SQL_WILDCARD_STYLE_LIKE = "like"
SQL_WILDCARD_STYLE_GLOB = "glob"
SQL_JSON_OBJECT_STYLE_OBJECT = "json_object"
SQL_JSON_OBJECT_STYLE_BUILD_OBJECT = "json_build_object"
SQL_JSON_PATH_STYLE_JSON1 = "json1"
SQL_JSON_AGG_STYLE_GROUP_ARRAY = "json_group_array"
SQL_JSON_AGG_STYLE_ARRAYAGG = "json_arrayagg"
SQL_JSON_AGG_STYLE_POSTGRES_ORDERED = "postgres_json_agg_ordered"
SQL_REGEX_STYLES = (
    SQL_REGEX_STYLE_REGEXP_LIKE,
    SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE,
    SQL_REGEX_STYLE_POSTGRES_TILDE,
    SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH,
)
SQL_WILDCARD_STYLES = (
    SQL_WILDCARD_STYLE_LIKE,
    SQL_WILDCARD_STYLE_GLOB,
)
SQL_JSON_OBJECT_STYLES = (
    SQL_JSON_OBJECT_STYLE_OBJECT,
    SQL_JSON_OBJECT_STYLE_BUILD_OBJECT,
)
SQL_JSON_AGG_STYLES = (
    SQL_JSON_AGG_STYLE_GROUP_ARRAY,
    SQL_JSON_AGG_STYLE_ARRAYAGG,
    SQL_JSON_AGG_STYLE_POSTGRES_ORDERED,
)
SQL_JSON_PATH_STYLES = (SQL_JSON_PATH_STYLE_JSON1,)
SQL_TEXT_KEY_OPS = ("group", "distinct", "order")


@dataclass(frozen=True)
class SQLDialectSpec:
    """Declare SQL handler and projection metadata for one backend identifier.

    Args:
        identifier: Canonical backend type identifier, such as `"sqlite"`.
        native_ops: Filter ops the SQL handler may consider for native pushdown.
        native_aggregate_ops: Reducers the dialect may lower through the parity-safe SQL aggregate compiler.
        case_insensitive_like: Whether LIKE-style comparisons need lower() or scan guards to match portable semantics.
        supports_regex: Whether the dialect supports the SQL regex compiler.
        supports_wildcard: Whether the dialect supports translated wildcard pushdown.
        regex_style: SQLAlchemy expression style for regex pushdown when supported.
        wildcard_style: SQLAlchemy expression style for translated wildcard pushdown when supported.
        side_table_json_object_style: SQLAlchemy function style for JSON objects projected from SideTable HyperG rows.
        side_table_json_agg_style: SQLAlchemy aggregate style for JSON arrays projected from SideTable rows.
        json_path_style: JSON path expression style for native `get` filter pushdown,
            or `None` when Json path filters must use Python scan.
        unique_nulls_distinct: Whether native unique indexes already ignore tuples
            containing null, matching HeavenBase uniqueness semantics.
        fallback_like: Whether plain `like` must use Python scan to preserve case-sensitive scan semantics.
        fallback_reasons: Backend-specific fallback reason text keyed by operation.
        native_text_ops: Candidate text-stored filter ops whose portable
            semantics this provider can prove with concrete-field evidence.
        native_text_key_ops: Candidate aggregate text-key purposes whose
            portable semantics can be proved with concrete-field evidence.
        text_evidence: Internal provider resolver returning scalar operations
            proved by the engine and enforced schema contract for one concrete
            physical field. Mutable row-content inspection is not proof. Its
            result is intersected with `native_text_ops`.
        text_filter_factory: Internal provider compiler for a proved scalar
            text predicate. It receives the column, operation, and operand.
        text_key_evidence: Internal provider resolver returning aggregate
            text-key purposes proved by the engine and enforced schema contract
            for one concrete field. Mutable row-content inspection is not proof.
        text_key_factory: Internal provider compiler for a proved aggregate
            comparison key. It receives the column and key purpose and must
            preserve the column's canonical text value and SQL result type.
    """

    identifier: str
    native_ops: tuple[str, ...] = SQL_DEFAULT_NATIVE_OPS
    native_aggregate_ops: tuple[str, ...] = ()
    case_insensitive_like: bool = False
    supports_regex: bool = False
    supports_wildcard: bool = False
    regex_style: str = SQL_REGEX_STYLE_REGEXP_LIKE
    wildcard_style: str = SQL_WILDCARD_STYLE_LIKE
    side_table_json_object_style: str = SQL_JSON_OBJECT_STYLE_OBJECT
    side_table_json_agg_style: str = SQL_JSON_AGG_STYLE_GROUP_ARRAY
    json_path_style: str | None = None
    unique_nulls_distinct: bool = True
    fallback_like: bool = False
    fallback_reasons: Mapping[str, str] = field(default_factory=dict)
    native_text_ops: tuple[str, ...] = ()
    native_text_key_ops: tuple[str, ...] = ()
    text_evidence: Callable[[Any, str | None, Any, str], tuple[str, ...]] | None = field(default=None, repr=False, compare=False)
    text_filter_factory: Callable[[Any, str, Any], Any] | None = field(default=None, repr=False, compare=False)
    text_key_evidence: Callable[[Any, str | None, Any, str], tuple[str, ...]] | None = field(default=None, repr=False, compare=False)
    text_key_factory: Callable[[Any, str], Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        identifier = check_id(str(self.identifier).strip(), "SQL dialect identifier")
        native_ops = tuple(str(op).strip() for op in self.native_ops)
        if not native_ops or any(not op for op in native_ops):
            raise ValueError("SQL dialect native_ops must contain non-empty operation names")
        native_text_ops = tuple(str(op).strip() for op in self.native_text_ops)
        if any(not op for op in native_text_ops):
            raise ValueError("SQL dialect native_text_ops must contain non-empty operation names")
        unknown_text_ops = tuple(op for op in native_text_ops if op not in native_ops)
        if unknown_text_ops:
            raise ValueError(f"SQL dialect native_text_ops must be a subset of native_ops; unknown={unknown_text_ops!r}")
        native_text_key_ops = tuple(str(op).strip() for op in self.native_text_key_ops)
        if any(not op for op in native_text_key_ops):
            raise ValueError("SQL dialect native_text_key_ops must contain non-empty purpose names")
        unknown_text_key_ops = tuple(op for op in native_text_key_ops if op not in SQL_TEXT_KEY_OPS)
        if unknown_text_key_ops:
            raise ValueError(f"Unknown SQL dialect native_text_key_ops {unknown_text_key_ops!r}; expected a subset of {SQL_TEXT_KEY_OPS!r}")
        native_aggregate_ops = tuple(str(op).strip() for op in self.native_aggregate_ops)
        if any(not op for op in native_aggregate_ops):
            raise ValueError("SQL dialect native_aggregate_ops must contain non-empty operation names")
        regex_style = str(self.regex_style).strip()
        if regex_style not in SQL_REGEX_STYLES:
            raise ValueError(f"Unknown SQL regex style {regex_style!r}; expected one of {SQL_REGEX_STYLES}")
        wildcard_style = str(self.wildcard_style).strip()
        if wildcard_style not in SQL_WILDCARD_STYLES:
            raise ValueError(f"Unknown SQL wildcard style {wildcard_style!r}; expected one of {SQL_WILDCARD_STYLES}")
        json_object_style = str(self.side_table_json_object_style).strip()
        if json_object_style not in SQL_JSON_OBJECT_STYLES:
            raise ValueError(f"Unknown SQL SideTable JSON object style {json_object_style!r}; expected one of {SQL_JSON_OBJECT_STYLES}")
        json_agg_style = str(self.side_table_json_agg_style).strip()
        if json_agg_style not in SQL_JSON_AGG_STYLES:
            raise ValueError(f"Unknown SQL SideTable JSON aggregate style {json_agg_style!r}; expected one of {SQL_JSON_AGG_STYLES}")
        json_path_style = str(self.json_path_style).strip() if self.json_path_style is not None else None
        if json_path_style is not None and json_path_style not in SQL_JSON_PATH_STYLES:
            raise ValueError(f"Unknown SQL JSON path style {json_path_style!r}; expected one of {SQL_JSON_PATH_STYLES}")
        reasons = {str(op).strip(): str(reason) for op, reason in dict(self.fallback_reasons or {}).items()}
        if any(not op for op in reasons):
            raise ValueError("SQL dialect fallback reason operation names must be non-empty")
        if self.text_evidence is not None and not callable(self.text_evidence):
            raise TypeError("SQL dialect text_evidence must be callable")
        if self.text_filter_factory is not None and not callable(self.text_filter_factory):
            raise TypeError("SQL dialect text_filter_factory must be callable")
        if self.text_key_evidence is not None and not callable(self.text_key_evidence):
            raise TypeError("SQL dialect text_key_evidence must be callable")
        if self.text_key_factory is not None and not callable(self.text_key_factory):
            raise TypeError("SQL dialect text_key_factory must be callable")
        if native_text_key_ops and self.text_key_factory is None:
            raise ValueError("SQL dialect native_text_key_ops require a value-preserving text_key_factory")
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "native_ops", native_ops)
        object.__setattr__(self, "native_text_ops", native_text_ops)
        object.__setattr__(self, "native_text_key_ops", native_text_key_ops)
        object.__setattr__(self, "native_aggregate_ops", native_aggregate_ops)
        object.__setattr__(self, "regex_style", regex_style)
        object.__setattr__(self, "wildcard_style", wildcard_style)
        object.__setattr__(self, "side_table_json_object_style", json_object_style)
        object.__setattr__(self, "side_table_json_agg_style", json_agg_style)
        object.__setattr__(self, "json_path_style", json_path_style)
        object.__setattr__(self, "fallback_reasons", MappingProxyType(reasons))

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-compatible SQL dialect capability metadata.

        Args:
            None.

        Returns:
            Dict[str, object]: SQL filter, regex, wildcard, SideTable projection, and
                fallback capability metadata for this dialect.
        """
        return {
            "identifier": self.identifier,
            "native_ops": list(self.native_ops),
            "native_text_ops": list(self.native_text_ops),
            "native_text_key_ops": list(self.native_text_key_ops),
            "native_aggregate_ops": list(self.native_aggregate_ops),
            "case_insensitive_like": self.case_insensitive_like,
            "supports_regex": self.supports_regex,
            "supports_wildcard": self.supports_wildcard,
            "regex_style": self.regex_style,
            "wildcard_style": self.wildcard_style,
            "side_table_json_object_style": self.side_table_json_object_style,
            "side_table_json_agg_style": self.side_table_json_agg_style,
            "json_path_style": self.json_path_style,
            "unique_nulls_distinct": self.unique_nulls_distinct,
            "fallback_like": self.fallback_like,
            "fallback_reasons": dict(self.fallback_reasons),
        }

    def supports_native_filter(self, op: str) -> bool:
        """Return whether the generic SQL handler may safely push a scalar filter down.

        Args:
            op: Filter operation identifier.

        Returns:
            True when the dialect has a compiler and its static portable guards
            permit the operation before concrete-field proof and value-specific
            guards are applied.
        """
        op_key = str(op).strip()
        if not self.has_native_filter_compiler(op_key):
            return False
        if op_key == "like" and self.fallback_like:
            return False
        return not (op_key in ("starts_with", "ends_with") and self.case_insensitive_like)

    def has_native_filter_compiler(self, op: str) -> bool:
        """Return whether the dialect has a raw compiler for one scalar filter.

        Args:
            op: Filter operation identifier.

        Returns:
            bool: True when compilation exists, independently of portable
            collation/case guards that a query-scoped preference may relax.
        """
        op_key = str(op).strip()
        if op_key not in self.native_ops:
            return False
        if op_key == "get":
            return self.json_path_style is not None
        if op_key == "regex":
            return self.supports_regex
        if op_key == "wildcard":
            return self.supports_wildcard
        return True

    def text_ops(self, backend: Any, entity_id: str | None, field_schema: Any, strategy: str) -> tuple[str, ...]:
        """Return text operations proved for one concrete physical field.

        Args:
            backend: Live SQL backend instance that owns the field.
            entity_id: Logical entity identifier bound to the physical table.
            field_schema: Logical field schema being compiled.
            strategy: Storage strategy identifier for the field binding.

        Returns:
            tuple[str, ...]: Provider-declared operations that also have live
            enforced-contract evidence, in `native_text_ops` order. Missing or
            failed evidence is an empty tuple. The resolver must not treat a
            scan of mutable current rows as durable proof.
        """
        if backend is None or entity_id is None or self.text_evidence is None:
            return ()
        try:
            evidence = {str(op).strip() for op in self.text_evidence(backend, entity_id, field_schema, strategy)}
        except Exception:
            # Provider inspection is a proof boundary: failure withholds
            # permission for native execution instead of changing query behavior.
            return ()
        return tuple(op for op in self.native_text_ops if op in evidence and (self.supports_native_filter(op) or self.supports_native_text_filter(op)))

    def supports_native_text_filter(self, op: str) -> bool:
        """Return whether the provider text factory implements one candidate.

        Args:
            op: Filter operation identifier.

        Returns:
            bool: True when the operation is a declared text candidate and the
            provider owns its expression compiler. This is compiler availability,
            not concrete-field semantic proof.
        """
        return self.text_filter_factory is not None and str(op).strip() in self.native_text_ops

    def text_filter_expression(self, col: Any, op: str, value: Any) -> Any | None:
        """Compile a provider-owned predicate for one proved text operation.

        Args:
            col: SQLAlchemy column expression for the concrete physical field.
            op: Portable filter operation identifier.
            value: Canonical query operand.

        Returns:
            Any | None: Provider predicate, or `None` when the generic compiler
            should lower this proved operation.
        """
        return self.text_filter_factory(col, op, value) if self.text_filter_factory is not None else None

    def text_key_ops(self, backend: Any, entity_id: str | None, field_schema: Any, strategy: str) -> tuple[str, ...]:
        """Return aggregate text-key purposes proved for one physical field.

        Args:
            backend: Live SQL backend instance that owns the field.
            entity_id: Logical entity identifier bound to the physical table.
            field_schema: Logical field schema being compiled.
            strategy: Storage strategy identifier for the field binding.

        Returns:
            tuple[str, ...]: Candidate purposes with enforced concrete-field
                evidence. Missing or failed evidence returns an empty tuple.
                Mutable current-row inspection is not durable proof.
        """
        if backend is None or entity_id is None or self.text_key_evidence is None:
            return ()
        try:
            evidence = {str(op).strip() for op in self.text_key_evidence(backend, entity_id, field_schema, strategy)}
        except Exception:
            # Aggregate proof is independent from scalar predicates and fails closed.
            return ()
        return tuple(op for op in self.native_text_key_ops if op in evidence)

    def text_key_expression(self, col: Any, purpose: str) -> Any:
        """Compile a provider-owned key for proved text aggregation semantics.

        Args:
            col: SQLAlchemy column expression for the concrete physical field.
            purpose: Aggregate key purpose: `group`, `distinct`, or `order`.

        Returns:
            Any: Value-preserving provider comparison-key expression.

        Raises:
            RuntimeError: If no provider key compiler is registered.
        """
        if self.text_key_factory is None:
            raise RuntimeError(f"SQL dialect {self.identifier!r} has no text key compiler")
        return self.text_key_factory(col, str(purpose))

    def unique_index_options(self, columns: tuple[Any, ...]) -> Dict[str, Any]:
        """Return dialect-owned DDL options for portable uniqueness semantics.

        Args:
            columns: SQLAlchemy columns participating in the unique constraint.

        Returns:
            Dict[str, Any]: Empty options when the dialect already permits multiple
                null tuples, otherwise a provider-owned filtered-index predicate.
        """
        if self.unique_nulls_distinct:
            return {}
        from sqlalchemy import and_

        return {f"{self.identifier}_where": and_(*[column.isnot(None) for column in columns])}

    def regex_expression(self, col: Any, value: Any) -> Any:
        """Build a SQLAlchemy regex predicate for this dialect.

        Args:
            col: SQLAlchemy column expression.
            value: User regex pattern after logical type normalization.

        Returns:
            SQLAlchemy predicate expression.

        Raises:
            ValueError: If regex pushdown is not declared for this dialect.
        """
        if not self.supports_regex:
            raise ValueError(f"Regex not supported on {self.identifier}")
        from sqlalchemy import func

        if self.regex_style == SQL_REGEX_STYLE_POSTGRES_TILDE:
            return col.op("~")(value)
        if self.regex_style == SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH:
            # DuckDB ~ matches the full string; wrap to get substring semantics like Python re.search.
            pat = str(value)
            if not pat.startswith("^"):
                pat = ".*" + pat
            if not pat.endswith("$"):
                pat = pat + ".*"
            return col.op("~")(pat)
        if self.regex_style == SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE:
            return func.REGEXP_LIKE(col, value, "c")
        return func.REGEXP_LIKE(col, value)

    def wildcard_expression(self, col: Any, value: Any, *, escape: str = "/") -> Any:
        """Build a SQLAlchemy wildcard predicate for this dialect.

        Args:
            col: SQLAlchemy column expression.
            value: Fnmatch-style wildcard pattern.
            escape: SQL LIKE escape character for `like`-style dialects.

        Returns:
            SQLAlchemy predicate expression.

        Raises:
            ValueError: If wildcard pushdown is not declared for this dialect.
        """
        if not self.supports_wildcard:
            raise ValueError(f"Wildcard not supported on {self.identifier}")
        pattern = str(value)
        if self.wildcard_style == SQL_WILDCARD_STYLE_GLOB:
            return col.op("GLOB")(pattern)
        escaped = pattern.replace(escape, escape + escape).replace("%", escape + "%").replace("_", escape + "_")
        like_pat = escaped.replace("*", "%").replace("?", "_")
        return col.like(like_pat, escape=escape)

    def json_get_expression(self, col: Any, path: tuple[str, ...], op: str, value: Any) -> Any:
        """Build a parity-guarded JSON path predicate for this dialect.

        The expression mirrors the portable Python scan semantics of the `get`
        operation: a missing or null path matches nothing, and heterogeneous
        comparisons match nothing instead of coercing across storage classes.
        Storage produced by ``Json.encode`` plus a JSON string column serializer
        is double-encoded (a JSON string whose content is JSON); the expression
        unwraps that shape per row before extracting the path, and rows whose
        payload is not decodable JSON match nothing.

        Args:
            col: SQLAlchemy column expression holding the Json payload.
            path: Nested object keys. Keys must not contain double quotes,
                backslashes, or control characters; the handler falls back to
                Python scan for such keys before calling this method.
            op: Comparison operation: `eq`, `ne`, `in`, `lt`, `lte`, `gt`, or `gte`.
            value: Scalar operand (`str`, `int`, `float`, `bool`, or `None`),
                or a list of such scalars for `in`.

        Returns:
            Any: SQLAlchemy predicate expression.

        Raises:
            ValueError: If the dialect declares no JSON path style or the
                operation cannot be compiled natively.
        """
        if self.json_path_style != SQL_JSON_PATH_STYLE_JSON1:
            raise ValueError(f"JSON path filters are not supported natively on {self.identifier}")
        from sqlalchemy import and_, case, false, func, null, or_

        inner = func.json_extract(col, "$")
        payload = case(
            (
                func.json_valid(col) == 1,
                case(
                    (func.json_type(col, "$") == "text", case((func.json_valid(inner) == 1, inner), else_=null())),
                    else_=col,
                ),
            ),
            else_=null(),
        )
        path_expr = "$" + "".join(f'."{part}"' for part in path)
        found = func.json_extract(payload, path_expr)
        kind = func.json_type(payload, path_expr)

        def kind_guard(item: Any) -> Any:
            # strict JSON kind compatibility, mirroring the portable _json_leaf_match:
            # booleans only match true/false, numbers only integer/real, strings only text
            if isinstance(item, bool):
                return kind.in_(("true", "false"))
            if isinstance(item, (int, float)):
                return kind.in_(("integer", "real"))
            if isinstance(item, str):
                return kind == "text"
            return None

        def eq_expr(item: Any) -> Any:
            guard = None if item is None else kind_guard(item)
            if guard is None:
                return false()  # null operands and non-scalar kinds never match after the portable pre-guards
            operand = int(item) if isinstance(item, bool) else item  # json true/false extract as 1/0
            return and_(guard, found == operand)

        if op == "eq":
            return eq_expr(value)
        if op == "ne":
            guard = None if value is None else kind_guard(value)
            if guard is None:
                return false()  # heterogeneous kinds match nothing, even for ne
            operand = int(value) if isinstance(value, bool) else value
            return and_(guard, found != operand)
        if op == "in":
            clauses = [eq_expr(item) for item in value]
            return or_(*clauses) if clauses else false()
        if op in ("lt", "lte", "gt", "gte"):
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
                return false()  # ordering is defined only for same-kind numbers and strings
            guard = kind == "text" if isinstance(value, str) else kind.in_(("integer", "real"))
            compare = {"lt": found < value, "lte": found <= value, "gt": found > value, "gte": found >= value}[op]
            return and_(guard, compare)
        raise ValueError(f"JSON path filters cannot compile {op!r} natively")

    def side_table_json_object_expression(self, fields: tuple[tuple[str, Any], ...]) -> Any:
        """Build a SQLAlchemy JSON-object expression for one SideTable item.

        Args:
            fields: Ordered `(name, expression)` pairs for one HyperG item.

        Returns:
            SQLAlchemy JSON object expression.
        """
        from sqlalchemy import func

        args = [item for pair in fields for item in pair]
        if self.side_table_json_object_style == SQL_JSON_OBJECT_STYLE_BUILD_OBJECT:
            return func.json_build_object(*args)
        return func.json_object(*args)

    def side_table_json_agg_expression(self, payload: Any, order_by: Any) -> Any:
        """Build a SQLAlchemy JSON-array aggregate for SideTable row projection.

        The PostgreSQL style orders the aggregate by ``order_by``. The other
        styles cannot order their input (MySQL's ``JSON_ARRAYAGG`` takes no
        ``ORDER BY``), so each element is a two-item array holding the
        ``order_by`` value and the payload. Pass the decoded aggregate to
        ``side_table_json_agg_items`` for the payloads in order.

        Args:
            payload: JSON-compatible item expression.
            order_by: Stable side-table ordering expression.

        Returns:
            SQLAlchemy aggregate expression.
        """
        from sqlalchemy import func

        if self.side_table_json_agg_style == SQL_JSON_AGG_STYLE_POSTGRES_ORDERED:
            from sqlalchemy.dialects.postgresql import aggregate_order_by

            return func.json_agg(aggregate_order_by(payload, order_by))
        positioned = func.json_array(order_by, payload)
        if self.side_table_json_agg_style == SQL_JSON_AGG_STYLE_ARRAYAGG:
            return func.json_arrayagg(positioned)
        return func.json_group_array(positioned)

    def side_table_json_agg_items(self, items: list) -> list:
        """Return the payloads of a decoded ``side_table_json_agg_expression`` aggregate in ``order_by`` order.

        Args:
            items: Decoded JSON array returned by the aggregate.

        Returns:
            list: Item payloads ordered by their ``order_by`` value.
        """
        if self.side_table_json_agg_style == SQL_JSON_AGG_STYLE_POSTGRES_ORDERED:
            return list(items)
        return [payload for _, payload in sorted(items, key=lambda item: item[0])]
