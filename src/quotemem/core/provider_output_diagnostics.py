"""Project real validation errors onto bounded categories and trusted schema structure.

The output models come from the provider output registry
(`quotemem.providers.contracts`), which imports `quotemem.core.inference`; this module
reads the registry when a function runs. A diagnostic keeps only a category and
the schema-owned path of each error, never error text, input or context.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import UnionType
from typing import Annotated, TypeAliasType, Union, cast, get_args, get_origin

from pydantic import BaseModel, ValidationError

from quotemem.core.identity import canonical_json
from quotemem.core.provider_diagnostics import (
    OUTPUT_DIAGNOSTIC_DEPTH,
    OUTPUT_DIAGNOSTIC_ISSUES,
    OUTPUT_DIAGNOSTIC_SEGMENT_CHARS,
    ProviderOutputErrorCategory,
    ProviderOutputIssueV1,
    ProviderOutputSchema,
    ProviderOutputValidationV1,
)

_TYPE_ERRORS = frozenset({
    "arguments_type", "bool_type", "bytes_type", "date_type", "datetime_type", "decimal_type",
    "dict_type", "float_type", "int_type", "iterable_type", "list_type", "mapping_type",
    "model_attributes_type", "model_type", "none_required", "set_type", "string_type",
    "time_type", "tuple_type", "uuid_type",
})
_TOO_SHORT_ERRORS = frozenset({"bytes_too_short", "string_too_short", "too_short"})
_TOO_LONG_ERRORS = frozenset({"bytes_too_long", "string_too_long", "too_long"})
_VALUE_ERRORS = frozenset({
    "bool_parsing", "date_parsing", "datetime_parsing", "decimal_parsing", "finite_number",
    "float_parsing", "greater_than", "greater_than_equal", "int_from_float", "int_parsing",
    "less_than", "less_than_equal", "multiple_of", "string_pattern_mismatch", "time_parsing",
    "uuid_parsing",
})


def _category(error_type: str) -> ProviderOutputErrorCategory:
    if error_type == "missing":
        return "missing"
    if error_type == "extra_forbidden":
        return "extra_field"
    if error_type in _TYPE_ERRORS:
        return "type_mismatch"
    if error_type in _TOO_SHORT_ERRORS:
        return "too_short"
    if error_type in _TOO_LONG_ERRORS:
        return "too_long"
    if error_type == "string_unicode":
        return "invalid_unicode"
    if error_type in _VALUE_ERRORS:
        return "value_constraint"
    if error_type in {"literal_error", "enum"}:
        return "literal_mismatch"
    if error_type in {"union_tag_invalid", "union_tag_not_found"}:
        return "union_tag"
    if error_type in {"value_error", "assertion_error"}:
        return "custom_validation"
    return "schema_invalid"


def _variants(annotation: object) -> Iterator[object]:
    origin = get_origin(annotation)
    if isinstance(annotation, TypeAliasType):
        yield from _variants(annotation.__value__)
    elif origin is Annotated:
        yield from _variants(get_args(annotation)[0])
    elif origin in (Union, UnionType):
        for option in get_args(annotation):
            yield from _variants(option)
    else:
        yield annotation


def _children(nodes: Sequence[object]) -> dict[str, list[object]]:
    """Read only declared model fields/list items; never aliases, tags or dictionary keys."""
    children: dict[str, list[object]] = {}
    for node in nodes:
        for variant in _variants(node):
            if isinstance(variant, type) and issubclass(variant, BaseModel):
                for name, field in variant.model_fields.items():
                    if len(name) <= OUTPUT_DIAGNOSTIC_SEGMENT_CHARS:
                        children.setdefault(name, []).append(field.annotation)
            elif get_origin(variant) is list:
                children.setdefault("[]", []).append(get_args(variant)[0])
    return children


def _model(schema_id: ProviderOutputSchema) -> type[BaseModel]:
    from quotemem.providers.contracts import registered_outputs

    model = next((contract.model for contract in registered_outputs()
                  if contract.schema_id == schema_id), None)
    if model is None:
        raise ValueError("Output diagnostics require a registered inference contract")
    return model


def output_path_is_owned(schema_id: ProviderOutputSchema, path: Sequence[str]) -> bool:
    """Validate restored diagnostic segments against their exact output contract."""
    nodes: Sequence[object] = (_model(schema_id),)
    for segment in path:
        children = _children(nodes)
        if segment not in children:
            return False
        nodes = children[segment]
    return True


def describe_output_validation(
    purpose: str, schema_id: str, error: BaseException | None,
) -> ProviderOutputValidationV1 | None:
    """Describe an actual Pydantic refusal without retaining its text, input or context.

    Unknown locations stop at their last trusted ancestor. In particular, union
    tags/class labels and unknown dictionary keys cannot be promoted to field
    names even when a later segment happens to name a real schema field.

    Args:
        purpose: The request's inference purpose.
        schema_id: The request's registered output schema.
        error: The refusal; only a pydantic ``ValidationError`` is described.

    Returns:
        The bounded description, or None for any other error.

    Raises:
        QuoteMemError: unsupported_variant if the contract is not registered.
    """
    from quotemem.providers.contracts import output_contract

    if not isinstance(error, ValidationError):
        return None
    model = output_contract(purpose, schema_id).model
    issues: list[ProviderOutputIssueV1] = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False)[
        :OUTPUT_DIAGNOSTIC_ISSUES
    ]:
        path: list[str] = []
        nodes: Sequence[object] = (model,)
        location = detail["loc"]
        for segment in location[:OUTPUT_DIAGNOSTIC_DEPTH]:
            token = "[]" if type(segment) is int and segment >= 0 else segment
            children = _children(nodes)
            if not isinstance(token, str) or token not in children:
                break
            path.append(token)
            nodes = children[token]
        issues.append(ProviderOutputIssueV1(
            category=_category(detail["type"]), field_path=path,
            path_limited=len(path) != len(location),
        ))
    return ProviderOutputValidationV1(
        schema_id=cast(ProviderOutputSchema, schema_id), issues=issues,
        truncated=error.error_count() > OUTPUT_DIAGNOSTIC_ISSUES,
    )


def maximum_output_validation(
    schema_id: ProviderOutputSchema | None = None,
) -> ProviderOutputValidationV1:
    """Size the largest allowed projection, never a persisted observation or schema decision."""
    def paths(nodes: Sequence[object], prefix: list[str]) -> Iterator[list[str]]:
        yield prefix
        if len(prefix) < OUTPUT_DIAGNOSTIC_DEPTH:
            for token, child in _children(nodes).items():
                yield from paths(child, [*prefix, token])

    categories = get_args(ProviderOutputErrorCategory.__value__)
    category = cast(ProviderOutputErrorCategory, max(categories, key=len))
    frames: list[ProviderOutputValidationV1] = []
    for candidate in get_args(ProviderOutputSchema.__value__):
        selected = cast(ProviderOutputSchema, candidate)
        longest = max(paths((_model(selected),), []), key=lambda path: len(canonical_json(path)))
        frames.append(ProviderOutputValidationV1(
            schema_id=selected, issues=[ProviderOutputIssueV1(
                category=category, field_path=longest, path_limited=False,
            ) for _ in range(OUTPUT_DIAGNOSTIC_ISSUES)], truncated=False,
        ))
    return max(
        (frame for frame in frames if schema_id is None or frame.schema_id == schema_id),
        key=lambda frame: len(canonical_json(frame)),
    )
