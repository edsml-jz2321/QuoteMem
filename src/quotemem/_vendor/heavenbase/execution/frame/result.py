"""Public table-shaped execution result value."""

from __future__ import annotations

__all__ = ["ResultFrame"]

from heavenbase.utils import Any, Iterable, Mapping, Sequence, coerce_nonnegint, deepcopy, raise_mismatch


class ResultFrame:
    """Hold one validated table-shaped result without execution policy.

    Backends return identity-bearing frames, workspace services hydrate and transform
    them, and users consume the final table. Frame transformations return new values;
    public column, identity, and metadata accessors return copies so callers cannot
    alter later routing or merge behavior accidentally.

    Args:
        columns (Mapping[str, Iterable[Any]] | None): Column-oriented values. All
            columns must have equal length. Identity-bearing frames require one
            unique, non-null value in ``identity_column`` for every row.
        metadata_columns (Iterable[str] | None): Result metadata columns whose list
            values concatenate during identity-based frame merges.
        identity_column (str | None): Identity column used by ``ids`` and frame set
            algebra. Use ``"object_id"`` for entity results or ``None`` for an
            identityless table.
        total (int | None): Complete pre-pagination row count when this frame is
            a slice. Defaults to ``None`` for an unsliced frame.

    Raises:
        TypeError: If columns, column values, metadata, or identities have invalid
            container shapes.
        ValueError: If names are empty, column lengths differ, identity values are
            missing or duplicated, or a populated identity frame lacks its identity
            column.
    """

    __slots__ = ("_columns", "_metadata_columns", "identity_column", "_total")

    def __init__(
        self,
        columns: Mapping[str, Iterable[Any]] | None = None,
        metadata_columns: Iterable[str] | None = None,
        identity_column: str | None = "object_id",
        total: int | None = None,
    ) -> None:
        """Create a validated result frame.

        Args:
            columns (Mapping[str, Iterable[Any]] | None): Column-oriented values.
            metadata_columns (Iterable[str] | None): Columns with mergeable result
                metadata.
            identity_column (str | None): Identity column, or ``None`` for a table.
            total (int | None): Row count this frame was sliced from, when it is a
                slice of a larger complete result. ``None`` means the frame is not
                known to be a slice.

        Returns:
            None: This function does not return a value.

        Raises:
            TypeError: If input containers or identity values are invalid.
            ValueError: If names, lengths, identity invariants, or the total are invalid.
        """
        if columns is not None and not isinstance(columns, Mapping):
            raise TypeError("ResultFrame columns must be a mapping")
        normalized: dict[str, tuple[Any, ...]] = {}
        for key, values in (columns or {}).items():
            name = str(key).strip()
            if not name:
                raise ValueError("ResultFrame column names must be non-empty")
            if name in normalized:
                raise ValueError(f"ResultFrame column name collision after normalization: {name!r}")
            if isinstance(values, (str, bytes)):
                raise TypeError(f"ResultFrame column {name!r} must be an iterable of values, not text")
            try:
                items = list(values)
            except TypeError as exc:
                raise TypeError(f"ResultFrame column {name!r} must be iterable") from exc
            normalized[name] = tuple(deepcopy(items))

        lengths = {len(values) for values in normalized.values()}
        if len(lengths) > 1:
            raise ValueError("ResultFrame columns must all have the same length")
        row_count = next(iter(lengths), 0)

        if identity_column is not None:
            identity_column = str(identity_column).strip()
            if not identity_column:
                raise ValueError("ResultFrame identity_column must be non-empty or None")
            if identity_column not in normalized:
                if row_count:
                    raise ValueError(f"ResultFrame identity column {identity_column!r} is missing")
                normalized = {identity_column: (), **normalized}
            else:
                normalized = {identity_column: normalized[identity_column], **{key: values for key, values in normalized.items() if key != identity_column}}
            self._validate_ids(normalized[identity_column], identity_column)

        self._columns = normalized
        self._metadata_columns = frozenset(self._normalize_names(metadata_columns or (), "metadata column"))
        self.identity_column = identity_column
        self._total = None if total is None else coerce_nonnegint(total, "ResultFrame total")
        if self._total is not None and self._total < row_count:
            raise ValueError(f"ResultFrame total cannot be smaller than its {row_count} rows")

    @staticmethod
    def _normalize_names(values: Iterable[Any], label: str) -> list[str]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"ResultFrame {label}s must be an iterable of names, not text")
        names = []
        for value in values:
            name = "" if value is None else str(value).strip()
            if not name:
                raise ValueError(f"ResultFrame {label} names must be non-empty")
            if name in names:
                raise ValueError(f"Duplicate ResultFrame {label}: {name!r}")
            names.append(name)
        return names

    @staticmethod
    def _validate_ids(values: tuple[Any, ...], name: str) -> None:
        if any(value is None or (isinstance(value, str) and not value.strip()) for value in values):
            raise ValueError(f"ResultFrame identity column {name!r} contains an empty value")
        try:
            unique = set(values)
        except TypeError as exc:
            raise TypeError(f"ResultFrame identity column {name!r} values must be hashable") from exc
        if len(unique) != len(values):
            raise ValueError(f"ResultFrame identity column {name!r} contains duplicate values")

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        cols: Sequence[str] | None = None,
        metadata_columns: Iterable[str] | None = None,
        identity_column: str | None = "object_id",
    ) -> "ResultFrame":
        """Build a frame from row mappings.

        Args:
            rows (Iterable[Mapping[str, Any]]): Row mappings in result order.
            cols (Sequence[str] | None): Explicit output column order. When omitted,
                columns are discovered and sorted.
            metadata_columns (Iterable[str] | None): Mergeable result metadata
                columns.
            identity_column (str | None): Identity column, or ``None`` for a table.

        Returns:
            ResultFrame: Validated column-oriented frame.

        Raises:
            TypeError: If a row is not a mapping.
            ValueError: If columns or identity values violate frame invariants.
        """
        normalized_rows = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError(f"ResultFrame rows must be mappings, got {type(row).__name__}")
            normalized_row = {}
            for key, value in row.items():
                name = "" if key is None else str(key).strip()
                if not name:
                    raise ValueError("ResultFrame row column names must be non-empty")
                if name in normalized_row:
                    raise ValueError(f"ResultFrame row column collision after normalization: {name!r}")
                normalized_row[name] = value
            normalized_rows.append(normalized_row)

        names = cls._normalize_names(cols, "column") if cols is not None else sorted({key for row in normalized_rows for key in row})
        if identity_column is not None:
            identity_column = str(identity_column).strip()
            if not identity_column:
                raise ValueError("ResultFrame identity_column must be non-empty or None")
            if any(identity_column not in row for row in normalized_rows):
                raise ValueError(f"ResultFrame rows must contain identity column {identity_column!r}")
            names = [identity_column, *[name for name in names if name != identity_column]]
        if normalized_rows and not names:
            raise ValueError("ResultFrame table rows must contain at least one column")
        return cls(
            {name: [row.get(name) for row in normalized_rows] for name in names},
            metadata_columns,
            identity_column,
        )

    @classmethod
    def from_table(
        cls,
        rows: Iterable[Mapping[str, Any]],
        cols: Sequence[str] | None = None,
        metadata_columns: Iterable[str] | None = None,
    ) -> "ResultFrame":
        """Build an identityless frame from table-shaped rows.

        Args:
            rows (Iterable[Mapping[str, Any]]): Table rows in result order.
            cols (Sequence[str] | None): Explicit output column order.
            metadata_columns (Iterable[str] | None): Mergeable result metadata
                columns.

        Returns:
            ResultFrame: Identityless table frame.

        Raises:
            TypeError: If a row is not a mapping.
            ValueError: If columns violate frame invariants.
        """
        return cls.from_rows(rows, cols=cols, metadata_columns=metadata_columns, identity_column=None)

    @classmethod
    def empty(
        cls,
        cols: Sequence[str] = ("object_id",),
        metadata_columns: Iterable[str] | None = None,
        identity_column: str | None = "object_id",
    ) -> "ResultFrame":
        """Create an empty frame with an explicit schema.

        Args:
            cols (Sequence[str]): Column names retained by the empty frame.
            metadata_columns (Iterable[str] | None): Mergeable result metadata
                columns.
            identity_column (str | None): Identity column, or ``None`` for a table.

        Returns:
            ResultFrame: Empty frame with the requested schema.

        Raises:
            TypeError: If a name collection is text instead of an iterable.
            ValueError: If a name is empty or duplicated.
        """
        names = cls._normalize_names(cols, "column")
        return cls({name: [] for name in names}, metadata_columns, identity_column)

    @property
    def columns(self) -> dict[str, list[Any]]:
        """Return a copy of the column-oriented values."""
        return {name: deepcopy(list(values)) for name, values in self._columns.items()}

    @property
    def metadata_columns(self) -> set[str]:
        """Return a copy of the mergeable result metadata column names."""
        return set(self._metadata_columns)

    @property
    def has_identity(self) -> bool:
        """Return whether identity-based APIs are available."""
        return self.identity_column is not None

    def require_identity(self) -> str:
        """Return the identity column required by frame set algebra.

        Args:
            None.

        Returns:
            str: Identity column name.

        Raises:
            ValueError: If this frame is identityless.
        """
        if self.identity_column is None:
            raise ValueError("ResultFrame has no identity column")
        return self.identity_column

    @property
    def ids(self) -> list[Any]:
        """Return a copy of identity values in frame order."""
        return deepcopy(list(self._columns[self.require_identity()]))

    def __eq__(self, other: object) -> bool:
        """Return whether another frame has the same schema and values.

        Args:
            other (object): Value to compare.

        Returns:
            bool: ``True`` when columns, metadata, identity, and total declarations match.
        """
        if not isinstance(other, ResultFrame):
            return NotImplemented
        return (
            self._columns == other._columns
            and self._metadata_columns == other._metadata_columns
            and self.identity_column == other.identity_column
            and self._total == other._total
        )

    def __len__(self) -> int:
        """Return the number of rows.

        Args:
            None.

        Returns:
            int: Row count.
        """
        if not self._columns:
            return 0
        return len(next(iter(self._columns.values())))

    @property
    def total(self) -> int | None:
        """Return the row count this frame was sliced from.

        Query execution applies pagination last, so this counts everything the
        preceding stages produced. That is the whole match set for an ordinary
        read or aggregate. It is not, when an earlier stage was itself bounded:
        a `near` clause feeds its pipeline only the top-k nearest rows, so a
        total taken downstream describes that window, not the entity. Callers
        that publish this number to a user must say which case they are in.

        `None` means the frame was never sliced and :func:`len` is already the
        whole of what it holds.

        Returns:
            int | None: Pre-slice row count, or `None` when unsliced.
        """
        return self._total

    @property
    def truncated(self) -> bool:
        """Return whether pagination withheld rows from this frame.

        Returns:
            bool: `True` when the frame is a proper slice of a larger result.
        """
        return self._total is not None and self._total > len(self)

    def col_names(self) -> list[str]:
        """Return column names in display order.

        Args:
            None.

        Returns:
            list[str]: Ordered column names.
        """
        return list(self._columns)

    def column(self, name: str) -> list[Any]:
        """Return a copy of one column.

        Args:
            name (str): Column name to read.

        Returns:
            list[Any]: Column values in frame order.

        Raises:
            ValueError: If ``name`` is not present.
        """
        name = str(name).strip()
        if name not in self._columns:
            raise_mismatch(self._columns, name, name="result column")
        return deepcopy(list(self._columns[name]))

    def rows(self) -> list[dict[str, Any]]:
        """Return frame data as row dictionaries.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: Row copies in frame order.
        """
        return [deepcopy(row) for row in self._iter_rows()]

    def _iter_rows(self) -> Iterable[dict[str, Any]]:
        """Borrow cells for trusted read-only execution; never expose or mutate them.

        Public exports still deep-copy. Internal validation and portable scoring
        need only read cells and must not repeatedly clone large vector values.
        """
        names = self.col_names()
        for index in range(len(self)):
            yield {name: self._columns[name][index] for name in names}

    def _with_leading_columns(self, names: Sequence[str]) -> "ResultFrame":
        """Return this frame led by `names` in that order, each one it lacks as an all-null column.

        Columns not in `names` follow in their current order. Existing cells are
        shared, not copied, so completing a large result's column set costs one
        null column per missing name.

        Args:
            names (Sequence[str]): Column names the result must hold first, in order.

        Returns:
            ResultFrame: This frame when its columns already have that order, else a derived frame.
        """
        leading = list(dict.fromkeys(names))
        wanted = set(leading)
        order = [*leading, *(name for name in self._columns if name not in wanted)]
        if order == list(self._columns):
            return self
        nulls = (None,) * len(self)
        return self._derive({name: self._columns.get(name, nulls) for name in order}, self._metadata_columns, self._total)

    def _derive(self, columns: dict[str, tuple[Any, ...]], metadata: frozenset[str], total: int | None) -> "ResultFrame":
        """Share privately owned cells for a validated projection or row selection.

        Only existing frame cells may enter this path, never caller/provider
        values. Each frame owns its column mapping and metadata; cells are
        read-only internally and every public value accessor returns copies.
        """
        derived = object.__new__(ResultFrame)
        derived._columns = columns
        derived._metadata_columns = metadata
        derived.identity_column = self.identity_column
        derived._total = total
        return derived

    def scalar(self) -> Any:
        """Return the sole cell of a one-row, one-column result table.

        Args:
            None.

        Returns:
            Any: A copy of the only result value.

        Raises:
            ValueError: If the frame does not contain exactly one row and one column.
        """
        if len(self) != 1 or len(self._columns) != 1:
            raise ValueError(f"ResultFrame.scalar needs exactly one row and one column; got {len(self)} rows and {len(self._columns)} columns")
        return deepcopy(next(iter(self._columns.values()))[0])

    def project(self, cols: Sequence[str]) -> "ResultFrame":
        """Return a frame restricted to requested columns.

        Every requested column must be one the frame holds, as for :meth:`sort`,
        whether or not the frame has rows. An empty frame holds the columns it
        was built with; build it with its schema through :meth:`empty`.

        Args:
            cols (Sequence[str]): Columns to keep. Identity is preserved
                automatically for identity-bearing frames.

        Returns:
            ResultFrame: Projected frame in requested column order.

        Raises:
            ValueError: If a requested column is unknown or duplicated, or an
                identityless projection has no columns.
        """
        names = self._normalize_names(cols, "column")
        if self.identity_column is not None:
            names = [self.identity_column, *[name for name in names if name != self.identity_column]]
        if not names:
            raise ValueError("ResultFrame projection must keep at least one column")
        for name in names:
            if name not in self._columns:
                raise_mismatch(self._columns, name, name="result column")
        return self._derive(
            {name: self._columns[name] for name in names},
            self._metadata_columns & set(names),
            self._total,  # projecting columns keeps every row
        )

    def filter_ids(self, object_ids: Iterable[Any]) -> "ResultFrame":
        """Return rows whose identity appears in a requested collection.

        Args:
            object_ids (Iterable[Any]): Identity values to keep.

        Returns:
            ResultFrame: Filtered frame preserving original row order.

        Raises:
            ValueError: If this frame is identityless.
        """
        self.require_identity()
        keep = set(object_ids)
        return self._take([index for index, object_id in enumerate(self.ids) if object_id in keep])

    def sort(self, field: str, desc: bool = False) -> "ResultFrame":
        """Return rows sorted by one existing column with nulls last.

        Args:
            field (str): Column name used for ordering.
            desc (bool): Sort concrete values in descending order when ``True``.

        Returns:
            ResultFrame: Stably sorted frame. ``None`` values remain last in either
            direction.

        Raises:
            TypeError: If concrete values cannot be compared.
            ValueError: If ``field`` is unknown.
        """
        values = self.column(field)
        concrete = [index for index, value in enumerate(values) if value is not None]
        nulls = [index for index, value in enumerate(values) if value is None]
        try:
            concrete.sort(key=values.__getitem__, reverse=bool(desc))
        except TypeError as exc:
            raise TypeError(f"ResultFrame column {field!r} contains incomparable values") from exc
        return self._take([*concrete, *nulls], total=self._total)  # a reorder keeps every row

    def limit(self, n: int | None = None, offset: int = 0) -> "ResultFrame":
        """Return a validated offset and limit slice.

        The slice records the complete row count as :attr:`total`, so a caller
        can report exactly how much was withheld instead of inferring it from
        the rows it received.

        Args:
            n (int | None): Maximum rows after ``offset``. ``None`` keeps the
                remaining rows.
            offset (int): Number of leading rows to skip.

        Returns:
            ResultFrame: Sliced frame carrying the pre-slice total.

        Raises:
            ValueError: If ``n`` or ``offset`` is negative.
        """
        offset = coerce_nonnegint(offset, "ResultFrame offset")
        stop = None if n is None else offset + coerce_nonnegint(n, "ResultFrame limit")
        complete_total = len(self) if self._total is None else self._total
        return self._take(list(range(len(self)))[offset:stop], total=complete_total)

    def _take(self, indices: Sequence[int], *, total: int | None = None) -> "ResultFrame":
        """Reorder or slice rows; callers state the resulting total explicitly.

        A row-preserving reorder passes the current total forward; a slice
        passes its own pre-slice count; anything that drops rows passes nothing,
        because the old total no longer describes this frame.
        """
        return self._derive(
            {name: tuple(values[index] for index in indices) for name, values in self._columns.items()},
            self._metadata_columns,
            total,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible frame representation.

        Args:
            None.

        Returns:
            dict[str, Any]: Copied columns plus non-default identity and metadata
            declarations.
        """
        out: dict[str, Any] = {"columns": self.columns}
        if self._metadata_columns:
            out["metadata_columns"] = sorted(self._metadata_columns)
        if self.identity_column != "object_id":
            out["identity_column"] = self.identity_column
        if self._total is not None:
            out["total"] = self._total
            out["truncated"] = self.truncated
        return out

    def to_pandas(self, *, columns: Sequence[str] | None = None) -> Any:
        """Export this frame to a pandas DataFrame.

        Args:
            columns (Sequence[str] | None): Optional columns to export.

        Returns:
            Any: ``pandas.DataFrame`` with selected columns.

        Raises:
            ImportError: If pandas is unavailable.
        """
        from ...interop.export import to_pandas

        return to_pandas(self, columns=columns)

    def to_pyarrow(self, *, columns: Sequence[str] | None = None, schema: Any = None) -> Any:
        """Export this frame to a PyArrow table.

        Args:
            columns (Sequence[str] | None): Optional columns to export.
            schema (Any): Optional PyArrow schema.

        Returns:
            Any: ``pyarrow.Table`` with selected columns.

        Raises:
            ImportError: If PyArrow is unavailable.
        """
        from ...interop.export import to_pyarrow

        return to_pyarrow(self, columns=columns, schema=schema)

    def to_numpy(self, *columns: str, dtype: Any = None) -> Any:
        """Export this frame to a NumPy array.

        Args:
            *columns (str): Optional columns to export. Omit to export all columns.
            dtype (Any): Optional NumPy dtype.

        Returns:
            Any: ``numpy.ndarray`` with selected columns.

        Raises:
            ImportError: If NumPy is unavailable.
        """
        from ...interop.export import to_numpy

        return to_numpy(self, *columns, dtype=dtype)

    def to_numpy_dict(self, *, columns: Sequence[str] | None = None) -> dict[str, Any]:
        """Export this frame to named NumPy arrays.

        Args:
            columns (Sequence[str] | None): Optional columns to export.

        Returns:
            dict[str, Any]: Column names mapped to ``numpy.ndarray`` values.

        Raises:
            ImportError: If NumPy is unavailable.
        """
        from ...interop.export import to_numpy_dict

        return to_numpy_dict(self, columns=columns)

    def to_pydantic(self, model: Any, *, rename: Mapping[str, str] | None = None, strict: bool = True) -> list[Any]:
        """Export rows to Pydantic model instances.

        Args:
            model (Any): Pydantic model class used to validate each row.
            rename (Mapping[str, str] | None): Optional source-to-model field map.
            strict (bool): Enable strict validation on Pydantic v2 models.

        Returns:
            list[Any]: Validated model instances in frame order.

        Raises:
            ImportError: If Pydantic is unavailable.
            TypeError: If ``model`` cannot construct row objects.
            ValueError: If row validation fails.
        """
        from ...interop.export import to_pydantic

        return to_pydantic(self, model, rename=rename, strict=strict)

    def to_daft(self, *, columns: Sequence[str] | None = None) -> Any:
        """Export this frame to a Daft DataFrame.

        Args:
            columns (Sequence[str] | None): Optional columns to export.

        Returns:
            Any: ``daft.DataFrame`` with selected columns.

        Raises:
            ImportError: If Daft or PyArrow is unavailable.
        """
        from ...interop.export import to_daft

        return to_daft(self, columns=columns)

    def to_str(self) -> str:
        """Return a compact plain-text preview.

        Args:
            None.

        Returns:
            str: Header and at most 20 rendered rows.
        """
        if len(self) == 0:
            return "ResultFrame<empty>"
        names = self.col_names()
        head = " | ".join(names)
        lines = [head, "-" * len(head)]
        lines.extend(" | ".join(repr(row.get(name)) for name in names) for row in self.rows()[:20])
        if len(self) > 20:
            lines.append(f"... ({len(self) - 20} more)")
        return "\n".join(lines)
