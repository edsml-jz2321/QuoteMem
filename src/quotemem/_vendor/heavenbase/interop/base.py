"""Shared interop staging contracts and diagnostics."""

from __future__ import annotations

__all__ = [
    "InteropIssue",
    "InteropReport",
    "InteropSet",
    "apply_entity_meta",
    "need",
]

from heavenbase.utils import Any, Dict, Iterable, List, Mapping, Optional, Sequence, dataclass, field as dcfield


@dataclass
class InteropIssue:
    """Describe one import/export assumption, warning, or error.

    Args:
        level (str): Diagnostic severity such as `warning` or `error`.
        code (str): Stable machine-readable diagnostic identifier.
        path (str): Source field or conversion location.
        message (str): Human-readable problem description.
        hint (str): Optional remediation guidance.
    """

    level: str
    code: str
    path: str
    message: str
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return this issue as a JSON-compatible mapping.

        Args:
            None.

        Returns:
            dict[str, Any]: Serialized diagnostic fields.
        """
        return {"level": self.level, "code": self.code, "path": self.path, "message": self.message, "hint": self.hint}


@dataclass
class InteropReport:
    """Collect serializable diagnostics for one interop conversion.

    Args:
        source (str): External source kind or user-facing source label.
        fields (Dict[str, Dict[str, Any]]): Per-field inference metadata.
        issues (List[InteropIssue]): Conversion diagnostics in discovery order.
        n_rows (int): Number of source rows inspected.
        strict (bool): Promote warnings to errors when adding diagnostics.
    """

    source: str
    fields: Dict[str, Dict[str, Any]] = dcfield(default_factory=dict)
    issues: List[InteropIssue] = dcfield(default_factory=list)
    n_rows: int = 0
    strict: bool = False

    @property
    def ok(self) -> bool:
        """Return whether the report contains no errors."""
        return not any(issue.level == "error" for issue in self.issues)

    def add(self, level: str, code: str, path: str, message: str, hint: str = "") -> None:
        """Append a diagnostic issue, promoting warnings in strict mode.

        Args:
            level (str): Diagnostic severity such as `warning` or `error`.
            code (str): Stable machine-readable diagnostic identifier.
            path (str): Source field or conversion location.
            message (str): Human-readable problem description.
            hint (str): Optional remediation guidance.

        Returns:
            None: The report is updated in place.
        """
        level = "error" if self.strict and level == "warning" else level
        self.issues.append(InteropIssue(level, code, path, message, hint))

    def raise_if_error(self) -> None:
        """Raise a compact error when conversion errors were reported.

        Args:
            None.

        Returns:
            None: This method returns when the report has no errors.

        Raises:
            ValueError: If one or more error diagnostics are present.
        """
        errors = [issue for issue in self.issues if issue.level == "error"]
        if errors:
            detail = "; ".join(f"{issue.path}: {issue.message}" for issue in errors[:5])
            suffix = "" if len(errors) <= 5 else f"; +{len(errors) - 5} more"
            raise ValueError(f"Interop conversion failed: {detail}{suffix}")

    def to_dict(self) -> dict[str, Any]:
        """Return this report as a JSON-compatible mapping.

        Args:
            None.

        Returns:
            dict[str, Any]: Source, field metadata, diagnostics, and row count.
        """
        return {"source": self.source, "fields": dict(self.fields), "issues": [issue.to_dict() for issue in self.issues], "n_rows": self.n_rows}

    def to_str(self) -> str:
        """Return a compact human-readable summary.

        Args:
            None.

        Returns:
            str: Summary followed by diagnostics when present.
        """
        head = f"InteropReport<{self.source}> rows={self.n_rows} fields={len(self.fields)} ok={self.ok}"
        if not self.issues:
            return head
        return "\n".join([head, *[f"{issue.level.upper()} {issue.code} {issue.path}: {issue.message}" for issue in self.issues]])


@dataclass
class InteropSet:
    """Stage one entity class, normalized rows, and conversion diagnostics.

    Args:
        entity (type): Entity class produced or consumed by the adapter.
        rows (List[Dict[str, Any]]): Normalized logical row mappings.
        report (InteropReport): Conversion diagnostics and inference metadata.
        source (str): External source kind or user-facing source label.
    """

    entity: type
    rows: List[Dict[str, Any]] = dcfield(default_factory=list)
    report: InteropReport = dcfield(default_factory=lambda: InteropReport("entity"))
    source: str = ""

    @classmethod
    def from_entity(cls, entity: Any, rows: Optional[Iterable[Any]] = None, *, source: str = "entity") -> "InteropSet":
        """Build a staging set from an existing entity and optional rows.

        Args:
            entity (Any): Entity class or instance defining the target schema.
            rows (Iterable[Any] | None): Optional record-shaped source values.
            source (str): Source label stored in diagnostics.

        Returns:
            InteropSet: Staging set with normalized row mappings.

        Raises:
            TypeError: If `entity` is not an Entity class or instance, or a row
                is not record-shaped.
        """
        from ..entity import Entity
        from .rows import row_from_any

        entity_cls = type(entity) if isinstance(entity, Entity) else entity
        if not isinstance(entity_cls, type) or not issubclass(entity_cls, Entity):
            raise TypeError("InteropSet.from_entity expects an Entity class or instance")
        out = [row_from_any(row) for row in rows or []]
        return cls(entity_cls, out, InteropReport(source, n_rows=len(out)), source)

    def describe(
        self, *, desc: Optional[str] = None, tags: Optional[Sequence[str]] = None, fields: Optional[Mapping[str, Mapping[str, Any]]] = None
    ) -> "InteropSet":
        """Annotate entity and field metadata before workspace registration.

        Args:
            desc (str | None): Optional entity description.
            tags (Sequence[str] | None): Optional entity tags.
            fields (Mapping[str, Mapping[str, Any]] | None): Field metadata keyed
                by logical field name.

        Returns:
            InteropSet: This staging set for fluent composition.

        Raises:
            KeyError: If `fields` names an unknown logical field.
        """
        apply_entity_meta(self.entity, desc=desc, tags=tags, fields=fields)
        return self

    def instances(self) -> List[Any]:
        """Return staged rows as entity instances.

        Args:
            None.

        Returns:
            List[Any]: Entity instances in source row order.
        """
        return [self.entity.from_dict(row) for row in self.rows]

    def one(self) -> Any:
        """Return the only staged entity instance.

        Args:
            None.

        Returns:
            Any: The sole staged entity instance.

        Raises:
            ValueError: If the staging set does not contain exactly one row.
        """
        if len(self.rows) != 1:
            raise ValueError(f"Expected exactly one row, got {len(self.rows)}")
        return self.entity.from_dict(self.rows[0])

    def register(self, ws: Any) -> type:
        """Register this set's entity class in a workspace.

        Args:
            ws (Any): Workspace receiving the inferred schema.

        Returns:
            type: Registered Entity class.

        Raises:
            ValueError: If the conversion report contains errors.
        """
        self.report.raise_if_error()
        return ws.register(self.entity)

    def upsert(self, ws: Any, *, overwrite: bool = True) -> List[str]:
        """Upsert staged rows for an Entity class already present in a workspace.

        Args:
            ws (Any): Workspace receiving the rows.
            overwrite (bool): Replace existing rows with matching object ids.

        Returns:
            List[str]: Materialized object ids in source row order.

        Raises:
            KeyError: If the Entity class is not present in the workspace.
            ValueError: If the conversion report contains errors or the
                workspace owns a different structure for the same entity id.
        """
        self.report.raise_if_error()
        return ws.upsert_many(self.entity, self.rows, overwrite=overwrite)

    def to_dict(self) -> dict[str, Any]:
        """Return this staging set as a JSON-compatible mapping.

        Args:
            None.

        Returns:
            dict[str, Any]: Schema, normalized rows, diagnostics, and source.
        """
        return {"entity": self.entity.schema().to_dict(), "rows": list(self.rows), "report": self.report.to_dict(), "source": self.source}

    def to_str(self) -> str:
        """Return a compact human-readable summary.

        Args:
            None.

        Returns:
            str: Entity, row count, source, and diagnostic summary.
        """
        return f"InteropSet<{self.entity.schema().entity_id}> rows={len(self.rows)} source={self.source}\n{self.report.to_str()}"


def need(pkg: str, feature: str) -> Any:
    """Import an optional dependency or raise an actionable interop error.

    Args:
        pkg (str): Importable package name.
        feature (str): User-facing feature requiring the package.

    Returns:
        Any: Imported top-level package module.

    Raises:
        ImportError: If the optional package is unavailable.
    """
    try:
        return __import__(pkg)
    except ImportError as exc:
        raise ImportError(f"{pkg} is required for {feature}; install heavenbase[interop] or {pkg}.") from exc


def apply_entity_meta(
    entity: type, *, desc: Optional[str] = None, tags: Optional[Sequence[str]] = None, fields: Optional[Mapping[str, Mapping[str, Any]]] = None
) -> None:
    """Apply metadata-only changes to an Entity class.

    Args:
        entity (type): Entity class whose compiled schema and authored field
            specs receive metadata.
        desc (str | None): Optional entity description.
        tags (Sequence[str] | None): Optional entity tags.
        fields (Mapping[str, Mapping[str, Any]] | None): Field descriptions,
            tags, and examples keyed by logical field name.

    Returns:
        None: Entity schema metadata is updated in place.

    Raises:
        KeyError: If `fields` names an unknown logical field.
    """
    schema = entity.schema()
    if desc is not None:
        schema.desc = str(desc)
    if tags is not None:
        schema.tags = list(tags)
    for name, meta in (fields or {}).items():
        field_schema = schema.get(str(name))
        spec = entity._hb_specs.get(str(name))
        if "desc" in meta:
            field_schema.desc = str(meta.get("desc") or "")
            if spec is not None:
                spec.desc_text = field_schema.desc
        if "tags" in meta:
            field_schema.tags = list(meta.get("tags") or [])
            if spec is not None:
                spec.tag_values = list(field_schema.tags)
        if "examples" in meta:
            field_schema.examples = list(meta.get("examples") or [])
            if spec is not None:
                spec.examples = list(field_schema.examples)
