"""Dotted Json path syntax shared by query validation and routing.

Dotted names such as ``meta.venue`` address nested payload values of a declared
``Json`` field. Filters desugar them to the registered ``get`` operation; this
module resolves ``group_by``, ``order``, and ``select`` names into neutral path
descriptors. Frame derivation lives under ``execution.frame``. Queries whose
clauses use dotted paths always take the portable Python route instead of native
pushdown.
"""

from __future__ import annotations

__all__ = [
    "json_path_columns",
    "split_json_path",
]

from heavenbase.utils import Any, Sequence


def split_json_path(schema: Any, name: Any) -> tuple[str, list[str]] | None:
    """Split one dotted Json path column name into its base field and key path.

    Args:
        schema (Any): Queried entity schema.
        name (Any): Candidate clause name, such as ``"meta.venue"``.

    Returns:
        tuple[str, list[str]] | None: ``(base_field, path)`` when ``name`` is a
        dotted path on a declared Json field, else ``None`` so unknown-name
        diagnostics stay owned by their existing call sites.

    Raises:
        LookupError: If the dotted base names a declared non-Json field.
    """
    name = str(name)
    if "." not in name:
        return None
    base, _, rest = name.partition(".")
    if not rest:
        return None
    try:
        field_schema = schema.get(base)
    except KeyError:
        return None
    dtype_id = type(field_schema.dtype).identifier
    if dtype_id != "json":
        raise LookupError(
            f"No Json path for {schema.entity_id}.{name}; dotted paths are not standalone fields — "
            f"they read nested values of a declared Json field, but {base} is {dtype_id}"
        )
    return base, rest.split(".")


def json_path_columns(schema: Any, names: Sequence[Any]) -> dict[str, tuple[str, list[str]]]:
    """Resolve every dotted Json path column referenced by query clause names.

    Args:
        schema (Any): Queried entity schema.
        names (Sequence[Any]): Clause names from ``group_by``, ``order``, or
            ``select``; plain declared names pass through unresolved.

    Returns:
        dict[str, tuple[str, list[str]]]: Dotted column name mapped to its
        ``(base_field, path)`` split, in first-seen clause order.

    Raises:
        LookupError: If a dotted base names a declared non-Json field.
    """
    out: dict[str, tuple[str, list[str]]] = {}
    for name in names:
        split = split_json_path(schema, name)
        if split is not None:
            out.setdefault(str(name), split)
    return out
