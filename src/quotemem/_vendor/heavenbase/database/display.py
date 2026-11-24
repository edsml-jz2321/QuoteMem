"""Small table display helper for SQL responses and row dictionaries."""

from __future__ import annotations

__all__ = ["table_display"]

from ..utils.typing import Any, Iterable, List, Optional, Union


def _clip(value: Any, max_width: int) -> str:
    text = "" if value is None else str(value).replace("\n", "\\n")
    if len(text) <= max_width:
        return text
    suffix = "." * min(3, max_width)
    return text[: max(0, max_width - len(suffix))] + suffix


def _resolve_names(columns: Optional[List[Union[str, int]]], available: List[str], check: bool) -> List[str]:
    if columns is None:
        return list(available)
    names: List[str] = []
    for col in columns:
        if isinstance(col, int):
            try:
                names.append(available[col])
            except IndexError as exc:
                raise ValueError(f"Column index {col} out of range for {len(available)} columns") from exc
        else:
            name = str(col)
            if check and name not in available:
                raise ValueError(f"Unknown column {name!r}. Available columns: {available}")
            names.append(name)
    return names


def _rows(data: Any, columns: Optional[List[Union[str, int]]], check: bool) -> tuple[List[str], List[dict[str, Any]]]:
    if hasattr(data, "to_dicts") and hasattr(data, "columns"):
        available = list(data.columns)
        names = _resolve_names(columns, available, check=check)
        try:
            return names, data.to_dicts(columns=columns, check=check)
        except TypeError:
            return names, data.to_dicts(columns=columns)
    values = [dict(row) for row in data]
    available = list(values[0].keys()) if values else []
    names = _resolve_names(columns, available, check=check)
    if check:
        missing = sorted({name for row in values for name in names if name not in row})
        if missing:
            raise ValueError(f"Rows are missing columns: {missing}")
    return names, [{name: row.get(name) for name in names} for row in values]


def _ellipsis_row(names: List[str]) -> dict[str, str]:
    return {name: "..." for name in names}


def table_display(
    data: Any,
    columns: Optional[List[Union[str, int]]] = None,
    *,
    max_rows: int = 64,
    max_width: int = 64,
    check: bool = True,
    footer: Union[bool, str] = True,
) -> str:
    """Render SQL rows as a compact ASCII table.

    Args:
        data: ``SQLResponse`` or any iterable of row dictionaries.
        columns: Optional column names or indexes to display. Unknown column names raise when ``check=True``.
        max_rows: Maximum displayed row count. Long tables show a compact sample.
        max_width: Maximum rendered width for each cell before clipping.
        check: Validate requested columns and row shape. Set ``False`` to display missing values as empty cells.
        footer: ``True`` to show the row-count line; ``False`` to suppress all footer text; a ``str`` to append custom text after the row count.

    Returns:
        str: ASCII table optionally followed by a total-row count and hint.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1")
    if max_width < 1:
        raise ValueError("max_width must be >= 1")
    show_footer = footer is not False
    footer_hint = footer if isinstance(footer, str) else ""
    names, rows = _rows(data, columns, check=check)
    total = len(rows)
    if total > max_rows:
        if max_rows == 1:
            rows = [_ellipsis_row(names)]
        elif max_rows == 2:
            rows = [rows[0], rows[-1]]
        else:
            rows = rows[: max_rows - 2] + [_ellipsis_row(names), rows[-1]]
    shown = [[_clip(row.get(name), max_width) for name in names] for row in rows]
    widths = [len(_clip(name, max_width)) for name in names]
    for row in shown:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    if not names:
        parts = ["0 columns"]
        if show_footer:
            parts.append(f"{total} rows")
            if footer_hint:
                parts.append(footer_hint)
        return "\n".join(parts)
    sep = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def fmt(values: Iterable[str]) -> str:
        return "|" + "|".join(f" {value:<{width}} " for value, width in zip(values, widths)) + "|"

    lines = [sep, fmt([_clip(name, max_width) for name in names]), sep]
    lines.extend(fmt(row) for row in shown)
    lines.append(sep)
    if show_footer:
        lines.append(f"{total} rows")
        if footer_hint:
            lines.append(footer_hint)
    return "\n".join(lines)
