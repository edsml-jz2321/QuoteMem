"""Stateless value normalization and validation helpers."""

from __future__ import annotations

__all__ = [
    "class_name",
    "clean_doc",
    "coerce_bool",
    "coerce_dict",
    "coerce_list",
    "coerce_nonnegint",
    "coerce_posint",
    "coerce_tuple",
    "field_name",
    "kebab",
    "lflat",
    "raise_mismatch",
    "value_match",
]

from difflib import SequenceMatcher

from .typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence


def kebab(name: str) -> str:
    """Convert a PascalCase or snake_case name to kebab-case.

    Args:
        name (str): Python or public name to normalize.

    Returns:
        str: Kebab-case name.
    """
    out = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            out.append("-")
        out.append(char.lower())
    return "".join(out).replace("_", "-")


def class_name(entity_id: str) -> str:
    """Convert a kebab-case entity identifier to a PascalCase class name.

    Args:
        entity_id (str): Public entity identifier.

    Returns:
        str: PascalCase Python class name.
    """
    return "".join(part.capitalize() for part in entity_id.replace("_", "-").split("-") if part)


def clean_doc(doc: str) -> str:
    """Normalize whitespace in documentation text to a single line.

    Args:
        doc (str): Documentation text to normalize.

    Returns:
        str: Single-line text with normalized whitespace.
    """
    return " ".join((doc or "").strip().split())


def field_name(value: Any) -> str:
    """Return the name of a field reference or the string form of a value.

    Args:
        value (Any): Field reference or plain value.

    Returns:
        str: Field name or string value.
    """
    return value.name if hasattr(value, "name") else str(value)


def coerce_bool(value: Any, label: str = "value") -> bool:
    """Return a strict Boolean boundary value.

    Args:
        value (Any): Value expected to be an actual `bool`.
        label (str): Human-readable name used in validation errors.

    Returns:
        bool: Validated Boolean value.

    Raises:
        TypeError: If `value` is not a `bool`.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be bool")
    return value


def coerce_dict(value: Any, label: str = "value") -> dict[Any, Any]:
    """Copy a mapping into a plain dictionary.

    Args:
        value (Any): Mapping value to normalize.
        label (str): Human-readable name used in validation errors.

    Returns:
        dict[Any, Any]: Shallow plain-dictionary copy.

    Raises:
        TypeError: If `value` is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def coerce_list(value: Any, label: str) -> list[Any]:
    """Return a list value while treating ``None`` as an empty list.

    Args:
        value (Any): Value to validate.
        label (str): Human-readable name used in validation errors.

    Returns:
        list[Any]: A shallow copy of the list, or an empty list for ``None``.

    Raises:
        TypeError: If `value` is neither a list nor ``None``.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    raise TypeError(f"{label} must be a list")


def coerce_posint(value: Any, label: str) -> int:
    """Coerce a value to a positive integer.

    Args:
        value (Any): Integer-compatible value to validate.
        label (str): Human-readable name used in validation errors.

    Returns:
        int: Positive integer value.

    Raises:
        TypeError: If `value` cannot be converted to an integer.
        ValueError: If `value` is not positive.
    """
    out = int(value)
    if out <= 0:
        raise ValueError(f"{label} must be positive")
    return out


def coerce_nonnegint(value: Any, label: str) -> int:
    """Coerce a value to a non-negative integer.

    Args:
        value (Any): Integer-compatible value to validate.
        label (str): Human-readable name used in validation errors.

    Returns:
        int: Non-negative integer value.

    Raises:
        TypeError: If `value` cannot be converted to an integer.
        ValueError: If `value` is negative.
    """
    out = int(value)
    if out < 0:
        raise ValueError(f"{label} must be non-negative")
    return out


def coerce_tuple(value: Any, label: str = "value") -> tuple[Any, ...]:
    """Copy a non-string sequence into a tuple.

    Args:
        value (Any): Sequence value to normalize.
        label (str): Human-readable name used in validation errors.

    Returns:
        tuple[Any, ...]: Shallow tuple copy.

    Raises:
        TypeError: If `value` is a string, bytes, or not a sequence.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def lflat(iterable: Iterable[Iterable[Any]]) -> list[Any]:
    """Flatten a two-level iterable into a list.

    Args:
        iterable (Iterable[Iterable[Any]]): Nested iterable to flatten once.

    Returns:
        list[Any]: Flattened values in iteration order.
    """
    return [item for sublist in iterable for item in sublist]


def value_match(
    supported: Sequence[Any],
    got: Any,
    thres: float = 0.3,
    case_sensitive: bool = True,
    normalizer: Optional[Callable[[Any], Any]] = None,
) -> list[tuple[Any, float]]:
    """Return supported values similar to a provided value.

    Args:
        supported (Sequence[Any]): Supported choices considered for suggestions.
        got (Any): Value to compare with the supported choices.
        thres (float): Inclusive minimum similarity score from 0 to 1.
        case_sensitive (bool): Whether string comparison preserves case.
        normalizer (Optional[Callable[[Any], Any]]): Optional value normalizer.

    Returns:
        list[tuple[Any, float]]: Matching values and scores, ordered by score.
    """
    target = _match_value(got, case_sensitive=case_sensitive, normalizer=normalizer)
    scored = [(value, SequenceMatcher(None, target, _match_value(value, case_sensitive=case_sensitive, normalizer=normalizer)).ratio()) for value in supported]
    return sorted((item for item in scored if item[1] >= thres), key=lambda item: (-item[1], type(item[0]).__name__, repr(item[0])))


def raise_mismatch(
    supported: Sequence[Any],
    got: Any,
    name: str = "value",
    mode: Literal["ignore", "match", "warn", "exit", "raise"] = "raise",
    comment: Optional[str] = None,
    thres: float = 0.3,
    case_sensitive: bool = True,
    normalizer: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Handle an unsupported value with a stable suggestion and failure mode.

    Args:
        supported (Sequence[Any]): Supported choices used for validation.
        got (Any): Value provided by the caller.
        name (str): Human-readable name used in the mismatch message.
        mode (Literal["ignore", "match", "warn", "exit", "raise"]): Mismatch behavior.
            Supported values:
            - `ignore`: Return the original value.
            - `match`: Return the closest match, or the original value.
            - `warn`: Log a warning and return the closest available value.
            - `exit`: Log an error and raise ``SystemExit``.
            - `raise`: Raise ``ValueError``.
        comment (Optional[str]): Optional guidance appended to the message.
        thres (float): Inclusive minimum suggestion score from 0 to 1.
        case_sensitive (bool): Whether string comparison preserves case.
        normalizer (Optional[Callable[[Any], Any]]): Optional value normalizer.

    Returns:
        Any: Original or matched value when the selected mode returns.

    Raises:
        SystemExit: If `mode` is `exit` and the value is unsupported.
        ValueError: If `mode` is `raise` or is itself unsupported.
    """
    values = list(supported)
    if got in values or mode == "ignore":
        return got
    matches = value_match(values, got, thres=thres, case_sensitive=case_sensitive, normalizer=normalizer)
    suggestion = matches[0][0] if matches else None
    if mode == "match":
        return suggestion if suggestion is not None else got
    message = _mismatch_message(values, got, name=name, suggestion=suggestion, comment=comment)
    if mode == "warn":
        from .log import get_logger

        get_logger(__name__).warning(message)
        return suggestion if suggestion is not None else got
    if mode == "exit":
        from .log import get_logger

        get_logger(__name__).error(message)
        raise SystemExit(1)
    if mode == "raise":
        raise ValueError(message)
    raise ValueError(_mismatch_message(["ignore", "match", "warn", "exit", "raise"], mode, name="mode"))


def _type_label(value: Any) -> str:
    """Name the type of ``value`` for an error message, module-qualified outside the builtins.

    NumPy 2 names its boolean scalar type ``bool``, so a bare ``__name__`` would
    report ``np.True_`` as the very type a "must be a bool" message asks for.

    Args:
        value: Value whose type to name.

    Returns:
        str: ``int`` or ``NoneType`` for a builtin type, ``numpy.bool`` or
        ``decimal.Decimal`` for any other.
    """
    kind = type(value)
    return kind.__qualname__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"


def _match_value(value: Any, case_sensitive: bool, normalizer: Optional[Callable[[Any], Any]]) -> str:
    value = normalizer(value) if normalizer else value
    text = str(value)
    return text if case_sensitive else text.lower()


def _mismatch_message(values: Sequence[Any], got: Any, name: str, suggestion: Optional[Any] = None, comment: Optional[str] = None) -> str:
    options = ", ".join(repr(item) for item in values) if values else "none"
    lines = [f"Unsupported {name} {got!r}."]
    if suggestion is not None:
        lines.append(f"Did you mean {suggestion!r}?")
    lines.append(f"Available options: {options}.")
    if comment:
        lines.append(str(comment))
    return "\n".join(lines)
