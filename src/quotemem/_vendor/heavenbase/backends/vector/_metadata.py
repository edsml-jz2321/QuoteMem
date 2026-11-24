"""Field values in scalar-only provider metadata (Chroma, Pinecone).

Chroma and Pinecone keep a row's fields as provider metadata, which holds only
scalars and flat lists of one scalar type. A value the metadata cannot hold as it
is (an object, an empty or mixed list, a list holding ``None``) is stored as
tagged compact JSON under its encoded key ``_hb_<field>`` and restored on read.
So is an integer outside the range the provider's numbers hold exactly (signed
64-bit on Chroma, +/-2**53 on Pinecone, whose numbers are 64-bit floats).
Field names never start with ``_``, so an encoded key, and the ``_hb``
placeholder that 0.1.2.2 wrote as the whole metadata of a row without a field
value, never meet a field. Reads leave the placeholder out.
"""

from __future__ import annotations

__all__ = [
    "ENCODED_PREFIX",
    "FLOAT64_EXACT_INTS",
    "FLOAT64_EXACT_OPERANDS",
    "INT64_EXACT_OPERANDS",
    "INT64_INTS",
    "LEGACY_PLACEHOLDER",
    "SCALAR_TYPES",
    "exact_operand",
    "metadata_entry",
    "metadata_fields",
]

from heavenbase.utils import Any, Callable, Mapping

from heavenbase.utils.serialize import dumps_json, loads_json

ENCODED_PREFIX = "_hb_"
# 0.1.2.2 metadata of a row without a field value; never a field, so reads leave it out.
LEGACY_PLACEHOLDER = "_hb"
SCALAR_TYPES = (str, int, float, bool)
# Inclusive integer ranges a provider's metadata numbers hold exactly.
INT64_INTS = (-(2**63), 2**63 - 1)
FLOAT64_EXACT_INTS = (-(2**53), 2**53)
# Exclusive magnitude below which a provider compares a numeric filter operand exactly.
# An integer outside the exact range is stored as its nearest float, of magnitude at
# least 2**63 on Chroma and at least 2**53 on Pinecone. Chroma compares an integer
# operand with such a float as a float, and an integer within 2**9 of 2**63 rounds to
# 2**63, so its bound stays 2**10 below 2**63. Pinecone compares 64-bit floats.
INT64_EXACT_OPERANDS = 2**63 - 2**10
FLOAT64_EXACT_OPERANDS = 2**53


def _exact(value: Any, ints: tuple[int, int]) -> bool:
    """Return whether a scalar is not an integer outside ``ints``."""
    return not isinstance(value, int) or isinstance(value, bool) or ints[0] <= value <= ints[1]


def exact_operand(value: Any, bound: int) -> bool:
    """Return whether the provider compares a filter operand as exactly as the portable scan.

    A stored integer outside the provider's exact range keeps its nearest float, and the
    provider compares numbers near that float as rounded or clamped numbers. So a filter
    whose operand is or holds a number of magnitude ``bound`` or more runs in the portable
    scan, which compares the exact values.

    Args:
        value (Any): Filter operand: a scalar, or the list, tuple or set of an ``in`` filter.
        bound (int): Exclusive magnitude below which the provider compares a number exactly.

    Returns:
        bool: ``False`` if the operand is or holds an integer or float of magnitude
        ``bound`` or more.
    """
    items = value if isinstance(value, (list, tuple, set)) else (value,)
    return not any(isinstance(item, (int, float)) and not isinstance(item, bool) and abs(item) >= bound for item in items)


def _rounded(value: int) -> float | None:
    """Return the float nearest an integer, or ``None`` when no float holds it."""
    try:
        return float(value)
    except OverflowError:
        return None


def metadata_entry(value: Any, native_list: Callable[[Any], bool], ints: tuple[int, int], *, preserve_null: bool = False) -> tuple[Any, str | None]:
    """Return the metadata value and the encoded value that together store one field value.

    Args:
        value (Any): Backend-ready field value.
        native_list (Callable[[Any], bool]): Whether the provider holds a list as it is.
        ints (tuple[int, int]): Inclusive integer range the provider's numbers hold exactly.
        preserve_null (bool): Store an explicit null under the encoded key so a
            defaulted field can distinguish a cleared value from an absent field.

    Returns:
        tuple[Any, str | None]: ``(native, encoded)``. A scalar or a list the provider
        holds is ``(value, None)``. A null is absent unless ``preserve_null`` is true. Any other value is
        encoded as tagged compact JSON. An integer outside ``ints`` keeps its nearest
        float as ``native``, so native comparisons see the number the provider would
        have stored. A list keeps the non-``None`` items the provider holds exactly as
        ``native`` when the provider holds them as a list, so a native membership filter
        still sees them.

    Raises:
        TypeError: If the value cannot be serialized as JSON.
    """
    if value is None:
        return None, "null" if preserve_null else None
    if isinstance(value, SCALAR_TYPES):
        if _exact(value, ints):
            return value, None
        return _rounded(value), dumps_json(value, sort_keys=True, compact=True)
    if native_list(value) and all(_exact(item, ints) for item in value):
        return value, None
    encoded = dumps_json(value, sort_keys=True, compact=True)
    if isinstance(value, list):
        present = [item for item in value if item is not None and _exact(item, ints)]
        if native_list(present):
            return present, encoded
    return None, encoded


def metadata_fields(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the field values that provider metadata stores, encoded values decoded.

    Args:
        metadata (Mapping[str, Any] | None): Provider metadata of one row.

    Returns:
        dict[str, Any]: Field values by name. An encoded value replaces its field's
        metadata value, and the ``_hb`` placeholder is left out.

    Raises:
        ValueError: If an encoded value is not valid tagged JSON.
    """
    fields: dict[str, Any] = {}
    encoded: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if key == LEGACY_PLACEHOLDER:
            continue
        if key.startswith(ENCODED_PREFIX):
            if value is not None:
                encoded[key[len(ENCODED_PREFIX) :]] = loads_json(value)
            continue
        fields[key] = value
    fields.update(encoded)
    return fields
