"""The fixed-point policy envelope shared by every policy snapshot.

Policies hold their numbers as six-decimal fixed-point strings, so snapshot
bytes never depend on float formatting. The recall policy lives in
``core/recall_policy.py`` and the extraction policy in
``core/extraction_policy.py``; this module keeps only the shared bounded
number types and the fixed-point encoding both use.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from typing import Annotated, Any, cast

from pydantic import Field, StringConstraints

from quotemem.store.metadata import StrictModel

Number01 = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Number02 = Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)]
Number04 = Annotated[float, Field(ge=0, le=4, allow_inf_nan=False)]
Number13 = Annotated[float, Field(ge=0, le=1.3, allow_inf_nan=False)]
Cosine = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
LexicalFloor = Annotated[float, Field(ge=0, le=1000000, allow_inf_nan=False)]
FixedNumber = Annotated[str, StringConstraints(pattern=r"^-?(0|[1-9][0-9]*)\.[0-9]{6}$")]


def to_fixed_point(value: Any) -> Any:
    """Encode a validated policy's float leaves at six decimal places.

    A policy snapshot holds its numbers this way, so its bytes never depend on
    float formatting. Integers, Booleans and strings stay as they are.

    Args:
        value (Any): A validated policy's JSON value.

    Returns:
        Any: The value with each float as a half-even ``FixedNumber`` string and
            each object's keys in sorted order.
    """
    if isinstance(value, float):
        number = Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
        return format(abs(number) if number == 0 else number, ".6f")
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {key: to_fixed_point(mapping[key]) for key in sorted(mapping)}
    if isinstance(value, list):
        return [to_fixed_point(item) for item in cast(list[Any], value)]
    return value


def from_fixed_point(value: dict[str, Any], schema: type[StrictModel]) -> dict[str, Any]:
    """Decode the float fields that a policy model declares from their fixed-point strings.

    Only declared float fields change, so a numeric-looking identifier stays a
    string.

    Args:
        value (dict[str, Any]): A snapshot's fixed-point parameters.
        schema (type[StrictModel]): The complete policy model that they normalize.

    Returns:
        dict[str, Any]: Parameters for ``schema`` to validate.
    """
    restored = value.copy()
    for key, field in schema.model_fields.items():
        annotation = field.annotation
        if annotation is float:
            restored[key] = float(Decimal(value[key]))
        elif isinstance(annotation, type) and issubclass(annotation, StrictModel):
            restored[key] = from_fixed_point(value[key], annotation)
    return restored
