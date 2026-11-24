"""Resolve open-world module tags from one inert Registry snapshot."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType
from typing import Any, Literal, cast

from heavenbase.utils.config.data import copy_data
from heavenbase.utils.registry import identifier_key, named_key

from .base import ModuleResolver, module_resolver
from .manifest import _record_fingerprint

__all__ = [
    "ResolvedTagValue",
    "TagDefinition",
    "TagView",
    "tag",
    "tag_definition",
    "tags",
]

TagStatus = Literal["resolved", "unresolved", "invalid", "ambiguous"]
TagDefinitionStatus = Literal["resolved", "invalid"]
TagSource = Literal["record", "default", "construction", "live"]
TagSubjectPolicy = Literal["exact", "category-open"]

_SCHEMA_TYPES = frozenset(("boolean", "string", "integer", "number", "identifier", "enum", "list", "set", "mapping", "object", "reference", "json"))
_LAYERS = frozenset(("static", "construction", "live"))
_MISSING = object()


def _freeze(value: Any, *, sort_keys: bool = False) -> Any:
    """Return one recursively immutable data value."""
    if isinstance(value, Mapping):
        keys = sorted(value) if sort_keys else value
        return MappingProxyType({str(key): _freeze(value[key], sort_keys=sort_keys) for key in keys})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, sort_keys=sort_keys) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item, sort_keys=sort_keys) for item in value)
    return copy_data(value)


def _wire(value: Any) -> Any:
    """Return the canonical JSON-compatible form of an immutable value."""
    if isinstance(value, Mapping):
        return {str(key): _wire(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_wire(item) for item in value]
    return value


def _token(value: Any) -> str:
    """Return deterministic type-sensitive JSON text for equality and order."""
    return json.dumps(_wire(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Require one string-keyed mapping at a typed Tag boundary."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} must contain only string keys")
    return cast(Mapping[str, Any], value)


def _finite_number(value: Any, path: str) -> int | float:
    """Require one finite non-Boolean number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    return value


def _bounded_integer(value: Any, path: str) -> int:
    """Require one non-negative non-Boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    if value < 0:
        raise ValueError(f"{path} must be non-negative")
    return value


def _pattern(value: Any, path: str) -> str:
    """Compile and return one regular-expression pattern."""
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    try:
        re.compile(value)
    except re.error as error:
        raise ValueError(f"{path} is not a valid regular expression: {error}") from error
    return value


def _normalize_schema(value: Any, path: str = "definition.schema") -> dict[str, Any]:
    """Validate and normalize the accepted declarative Tag schema grammar."""
    raw = _mapping(value, path)
    schema_type = raw.get("type")
    if not isinstance(schema_type, str) or schema_type not in _SCHEMA_TYPES:
        raise ValueError(f"{path}.type must be one of {sorted(_SCHEMA_TYPES)!r}")
    allowed: dict[str, frozenset[str]] = {
        "boolean": frozenset(("type",)),
        "string": frozenset(("type", "pattern", "min_length", "max_length")),
        "integer": frozenset(("type", "minimum", "maximum")),
        "number": frozenset(("type", "minimum", "maximum")),
        "identifier": frozenset(("type", "pattern")),
        "enum": frozenset(("type", "values")),
        "list": frozenset(("type", "items", "unique_keys")),
        "set": frozenset(("type", "items")),
        "mapping": frozenset(("type", "values", "key_pattern")),
        "object": frozenset(("type", "properties", "required", "additional_properties", "constraints")),
        "reference": frozenset(("type", "kind", "state")),
        "json": frozenset(("type",)),
    }
    unknown = set(raw) - allowed[schema_type]
    if unknown:
        raise ValueError(f"{path} has unsupported fields {sorted(unknown)!r}")
    normalized: dict[str, Any] = {"type": schema_type}

    if schema_type in {"string", "identifier"} and "pattern" in raw:
        normalized["pattern"] = _pattern(raw["pattern"], f"{path}.pattern")
    if schema_type == "string":
        for name in ("min_length", "max_length"):
            if name in raw:
                normalized[name] = _bounded_integer(raw[name], f"{path}.{name}")
        if normalized.get("min_length", 0) > normalized.get("max_length", math.inf):
            raise ValueError(f"{path}.min_length must not exceed max_length")
    if schema_type in {"integer", "number"}:
        for name in ("minimum", "maximum"):
            if name in raw:
                boundary = _finite_number(raw[name], f"{path}.{name}")
                if schema_type == "integer" and not isinstance(boundary, int):
                    raise TypeError(f"{path}.{name} must be an integer")
                normalized[name] = boundary
        if normalized.get("minimum", -math.inf) > normalized.get("maximum", math.inf):
            raise ValueError(f"{path}.minimum must not exceed maximum")
    if schema_type == "enum":
        values = raw.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path}.values must be a non-empty list")
        normalized_values: list[Any] = []
        seen: set[str] = set()
        for index, item in enumerate(values):
            item_path = f"{path}.values[{index}]"
            if item is None or isinstance(item, (list, Mapping)):
                raise TypeError(f"{item_path} must be a non-null scalar")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"{item_path} must be finite")
            token = _token(item)
            if token in seen:
                raise ValueError(f"{path}.values repeats {item!r}")
            seen.add(token)
            normalized_values.append(copy.deepcopy(item))
        normalized["values"] = normalized_values
    if schema_type in {"list", "set"}:
        if "items" not in raw:
            raise ValueError(f"{path}.items is required")
        normalized["items"] = _normalize_schema(raw["items"], f"{path}.items")
        if schema_type == "list" and "unique_keys" in raw:
            unique_keys = raw["unique_keys"]
            if not isinstance(unique_keys, list):
                raise TypeError(f"{path}.unique_keys must be a list")
            normalized_keys: list[list[str]] = []
            for index, fields in enumerate(unique_keys):
                key_path = f"{path}.unique_keys[{index}]"
                if not isinstance(fields, list) or not fields:
                    raise ValueError(f"{key_path} must be a non-empty list")
                names = [identifier_key(item, f"{key_path} field") for item in fields]
                if len(set(names)) != len(names):
                    raise ValueError(f"{key_path} repeats a field")
                normalized_keys.append(names)
            if normalized_keys and normalized["items"]["type"] != "object":
                raise ValueError(f"{path}.unique_keys requires object items")
            normalized["unique_keys"] = normalized_keys
    if schema_type == "mapping":
        if "values" not in raw:
            raise ValueError(f"{path}.values is required")
        normalized["values"] = _normalize_schema(raw["values"], f"{path}.values")
        if "key_pattern" in raw:
            normalized["key_pattern"] = _pattern(raw["key_pattern"], f"{path}.key_pattern")
    if schema_type == "object":
        properties = _mapping(raw.get("properties"), f"{path}.properties")
        normalized["properties"] = {str(name): _normalize_schema(schema, f"{path}.properties.{name}") for name, schema in properties.items()}
        required = raw.get("required", [])
        if not isinstance(required, list):
            raise TypeError(f"{path}.required must be a list")
        normalized_required = [identifier_key(name, f"{path}.required field") for name in required]
        if len(set(normalized_required)) != len(normalized_required):
            raise ValueError(f"{path}.required repeats a field")
        unknown_required = set(normalized_required) - set(normalized["properties"])
        if unknown_required:
            raise ValueError(f"{path}.required names unknown properties {sorted(unknown_required)!r}")
        normalized["required"] = normalized_required
        additional = raw.get("additional_properties", False)
        if not isinstance(additional, bool):
            raise TypeError(f"{path}.additional_properties must be Boolean")
        normalized["additional_properties"] = additional
        constraints = raw.get("constraints", [])
        if not isinstance(constraints, list):
            raise TypeError(f"{path}.constraints must be a list")
        normalized_constraints: list[dict[str, str]] = []
        for index, constraint in enumerate(constraints):
            constraint_path = f"{path}.constraints[{index}]"
            value = _mapping(constraint, constraint_path)
            if set(value) != {"kind", "field", "of"} or value.get("kind") != "subset":
                raise ValueError(f"{constraint_path} must be a subset constraint with kind, field, and of")
            field_name = identifier_key(value["field"], f"{constraint_path}.field")
            parent_name = identifier_key(value["of"], f"{constraint_path}.of")
            try:
                field_schema = normalized["properties"][field_name]
                parent_schema = normalized["properties"][parent_name]
            except KeyError as error:
                raise ValueError(f"{constraint_path} names an unknown property {error.args[0]!r}") from error
            if field_schema["type"] not in {"list", "set"} or parent_schema["type"] not in {"list", "set"}:
                raise ValueError(f"{constraint_path} requires list or set properties")
            if _token(field_schema["items"]) != _token(parent_schema["items"]):
                raise ValueError(f"{constraint_path} requires compatible item schemas")
            normalized_constraints.append({"kind": "subset", "field": field_name, "of": parent_name})
        normalized["constraints"] = normalized_constraints
    if schema_type == "reference":
        if "kind" not in raw:
            raise ValueError(f"{path}.kind is required")
        normalized["kind"] = identifier_key(raw["kind"], f"{path}.kind")
        state = raw.get("state", "any")
        if state not in {"any", "active"}:
            raise ValueError(f"{path}.state must be 'any' or 'active'")
        normalized["state"] = state
    return normalized


@dataclass(frozen=True)
class _Reference:
    """Describe one validated soft Registry reference."""

    kind: str
    identifier: str
    state: Literal["any", "active"]
    path: str


def _validate_value(schema: Mapping[str, Any], value: Any, path: str = "value") -> tuple[Any, tuple[_Reference, ...]]:
    """Validate one Tag value and return its immutable canonical form."""
    schema_type = schema["type"]
    if value is None:
        raise TypeError(f"{path} must not be null")
    if schema_type == "boolean":
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be Boolean")
        return value, ()
    if schema_type in {"string", "identifier"}:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string")
        if schema_type == "identifier":
            canonical = identifier_key(value, path)
            if canonical != value:
                raise ValueError(f"{path} must use canonical identifier spelling")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
            raise ValueError(f"{path} does not match pattern {pattern!r}")
        if schema_type == "string":
            if len(value) < schema.get("min_length", 0):
                raise ValueError(f"{path} is shorter than min_length")
            if len(value) > schema.get("max_length", math.inf):
                raise ValueError(f"{path} is longer than max_length")
        return value, ()
    if schema_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path} must be an integer")
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{path} is outside the accepted range")
        return value, ()
    if schema_type == "number":
        number = _finite_number(value, path)
        if number < schema.get("minimum", -math.inf) or number > schema.get("maximum", math.inf):
            raise ValueError(f"{path} is outside the accepted range")
        return number, ()
    if schema_type == "enum":
        choices = {_token(item): item for item in schema["values"]}
        token = _token(value)
        if token not in choices:
            raise ValueError(f"{path} must be one of {list(schema['values'])!r}")
        return copy.deepcopy(choices[token]), ()
    if schema_type in {"list", "set"}:
        if not isinstance(value, list):
            raise TypeError(f"{path} must be a JSON array")
        items: list[Any] = []
        references: list[_Reference] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            canonical, nested = _validate_value(schema["items"], item, f"{path}[{index}]")
            token = _token(canonical)
            if schema_type == "set" and token in seen:
                raise ValueError(f"{path} contains duplicate set item at index {index}")
            seen.add(token)
            items.append(canonical)
            references.extend(nested)
        for unique_key in schema.get("unique_keys", ()):
            unique: set[tuple[str, ...]] = set()
            for index, item in enumerate(items):
                if not isinstance(item, Mapping):
                    raise TypeError(f"{path}[{index}] must be an object for unique_keys")
                missing = [name for name in unique_key if name not in item]
                if missing:
                    raise ValueError(f"{path}[{index}] lacks unique key fields {missing!r}")
                key = tuple(_token(item[name]) for name in unique_key)
                if key in unique:
                    raise ValueError(f"{path} repeats unique key {list(unique_key)!r}")
                unique.add(key)
        if schema_type == "set":
            items.sort(key=_token)
        return tuple(items), tuple(references)
    if schema_type == "mapping":
        raw = _mapping(value, path)
        out: dict[str, Any] = {}
        references: list[_Reference] = []
        pattern = schema.get("key_pattern")
        for key in sorted(raw):
            if isinstance(pattern, str) and re.fullmatch(pattern, key) is None:
                raise ValueError(f"{path} key {key!r} does not match pattern {pattern!r}")
            out[key], nested = _validate_value(schema["values"], raw[key], f"{path}.{key}")
            references.extend(nested)
        return MappingProxyType(out), tuple(references)
    if schema_type == "object":
        raw = _mapping(value, path)
        properties = schema["properties"]
        missing = [name for name in schema["required"] if name not in raw]
        if missing:
            raise ValueError(f"{path} lacks required properties {missing!r}")
        unknown = set(raw) - set(properties)
        if unknown and not schema["additional_properties"]:
            raise ValueError(f"{path} has unsupported properties {sorted(unknown)!r}")
        out: dict[str, Any] = {}
        references: list[_Reference] = []
        for key in sorted(raw):
            if key in properties:
                out[key], nested = _validate_value(properties[key], raw[key], f"{path}.{key}")
            else:
                out[key] = _canonical_json(raw[key], f"{path}.{key}")
                nested = ()
            references.extend(nested)
        for constraint in schema["constraints"]:
            field_values = out.get(constraint["field"], ())
            parent_values = out.get(constraint["of"], ())
            field_tokens = {_token(item) for item in field_values}
            parent_tokens = {_token(item) for item in parent_values}
            if not field_tokens.issubset(parent_tokens):
                raise ValueError(f"{path}.{constraint['field']} must be a subset of {constraint['of']}")
        return MappingProxyType(out), tuple(references)
    if schema_type == "reference":
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string reference")
        clean = named_key(value, path)
        if clean != value:
            raise ValueError(f"{path} must preserve exact trimmed reference spelling")
        reference = _Reference(schema["kind"], value, schema["state"], path)
        return value, (reference,)
    if schema_type == "json":
        return _canonical_json(value, path), ()
    raise ValueError(f"{path} uses unsupported schema type {schema_type!r}")


def _canonical_json(value: Any, path: str) -> Any:
    """Validate and freeze one non-null canonical JSON value."""
    if value is None:
        raise TypeError(f"{path} must not be null")
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, list):
        return tuple(_canonical_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} must contain only string keys")
        return MappingProxyType({key: _canonical_json(value[key], f"{path}.{key}") for key in sorted(value)})
    raise TypeError(f"{path} contains non-JSON value {value!r}")


def _validate_raw_json(value: Any, path: str) -> None:
    """Require one JSON-compatible raw claim without rejecting null."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_raw_json(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} must contain only string keys")
        for key, item in value.items():
            _validate_raw_json(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains non-JSON value {value!r}")


@dataclass(frozen=True)
class TagDefinition:
    """Describe one inert, optionally valid `kind: tag` definition.

    Args:
        identifier (str): Canonical Tag identifier.
        aliases (tuple[str, ...]): Registry aliases for the definition.
        schema (Mapping[str, Any]): Immutable declarative value schema.
        default (Any): Immutable typed default, raw invalid default, or `None`
            when no default was declared.
        has_default (bool): Whether the definition declares a default.
        subjects (tuple[str, ...]): Intended built-in module-record kinds.
        subject_policy (Literal["exact", "category-open"]): Advisory subject
            policy. `category-open` permits explicit claims on future kinds.
        layers (tuple[str, ...]): Intended static, construction, or live layers.
        name (str): Human-readable definition name.
        desc (str): Human-readable definition description.
        status (Literal["resolved", "invalid"]): Definition parse status.
        diagnostics (tuple[str, ...]): Record-local validation messages.
        publication (int): Process-local Registry publication.
        revision (int): Durable Registry revision.
        snapshot (str): Stable token for the captured Registry state.
        fingerprint (str): Definition-record fingerprint.

    Raises:
        TypeError: If collection or status fields have invalid types.
        ValueError: If identifiers, policies, layers, or snapshot evidence are invalid.
    """

    identifier: str
    aliases: tuple[str, ...]
    schema: Mapping[str, Any]
    default: Any
    has_default: bool
    subjects: tuple[str, ...]
    subject_policy: TagSubjectPolicy
    layers: tuple[str, ...]
    name: str
    desc: str
    status: TagDefinitionStatus
    diagnostics: tuple[str, ...]
    publication: int
    revision: int
    snapshot: str
    fingerprint: str

    def __post_init__(self) -> None:
        """Validate and freeze one public definition facade.

        Args:
            None.

        Returns:
            None: The dataclass fields are normalized in place.

        Raises:
            TypeError: If Boolean or Registry evidence fields use invalid types.
            ValueError: If identity, status, policy, layer, or evidence values are invalid.
        """
        object.__setattr__(self, "identifier", identifier_key(self.identifier, "tag identifier"))
        object.__setattr__(self, "aliases", tuple(named_key(alias, "tag alias") for alias in self.aliases))
        object.__setattr__(self, "schema", _freeze(self.schema, sort_keys=True))
        object.__setattr__(self, "default", _freeze(self.default, sort_keys=True))
        object.__setattr__(self, "subjects", tuple(identifier_key(subject, "tag subject") for subject in self.subjects))
        object.__setattr__(self, "layers", tuple(identifier_key(layer, "tag layer") for layer in self.layers))
        object.__setattr__(self, "diagnostics", tuple(str(message) for message in self.diagnostics))
        if not isinstance(self.has_default, bool):
            raise TypeError("TagDefinition has_default must be Boolean")
        if self.subject_policy not in {"exact", "category-open"}:
            raise ValueError("TagDefinition subject_policy must be 'exact' or 'category-open'")
        if set(self.layers) - _LAYERS:
            raise ValueError(f"TagDefinition layers must use {sorted(_LAYERS)!r}")
        if self.status not in {"resolved", "invalid"}:
            raise ValueError("TagDefinition status must be 'resolved' or 'invalid'")
        for name, value in (("publication", self.publication), ("revision", self.revision)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"TagDefinition {name} must be a non-negative integer")
        if not self.snapshot or not self.fingerprint:
            raise ValueError("TagDefinition snapshot and fingerprint must be non-empty")

    @property
    def valid(self) -> bool:
        """Return whether this definition has a usable declarative contract.

        Args:
            None.

        Returns:
            bool: `True` only for a successfully parsed definition.
        """
        return self.status == "resolved"


@dataclass(frozen=True)
class ResolvedTagValue:
    """Expose one raw claim or definition default with typed resolution status.

    Args:
        tag (str): Canonical identifier when defined, otherwise the raw key.
        value (Any): Immutable canonical value, raw unresolved value, or `None`
            for invalid and ambiguous results.
        raw (Any): Immutable authored value or definition default.
        raw_keys (tuple[str, ...]): Authored keys contributing to the result.
        status (Literal["resolved", "unresolved", "invalid", "ambiguous"]):
            Record-local interpretation status.
        source (Literal["record", "default", "construction", "live"]): Value
            layer selected by precedence.
        subject (tuple[str, str]): Canonical subject kind and identifier.
        definition (TagDefinition | None): Loaded definition when present.
        diagnostics (tuple[str, ...]): Local validation or reference messages.
        publication (int): Process-local Registry publication.
        revision (int): Durable Registry revision.
        snapshot (str): Stable token for the captured Registry state.
        instance (Any | None): Optional concrete live-object identity.
    """

    tag: str
    value: Any
    raw: Any
    raw_keys: tuple[str, ...]
    status: TagStatus
    source: TagSource
    subject: tuple[str, str]
    definition: TagDefinition | None
    diagnostics: tuple[str, ...]
    publication: int
    revision: int
    snapshot: str
    instance: Any | None = None

    def __post_init__(self) -> None:
        """Validate and freeze one public resolved-value facade.

        Args:
            None.

        Returns:
            None: The dataclass fields are normalized in place.

        Raises:
            TypeError: If Registry evidence or subject fields have invalid types.
            ValueError: If identity, status, source, or snapshot fields are invalid.
        """
        if not isinstance(self.tag, str):
            raise TypeError("ResolvedTagValue tag must be a string")
        if self.definition is not None:
            object.__setattr__(self, "tag", identifier_key(self.tag, "tag identifier"))
        object.__setattr__(self, "value", _freeze(self.value, sort_keys=True))
        object.__setattr__(self, "raw", _freeze(self.raw))
        if not all(isinstance(key, str) for key in self.raw_keys):
            raise TypeError("ResolvedTagValue raw_keys must contain only strings")
        object.__setattr__(self, "raw_keys", tuple(self.raw_keys))
        object.__setattr__(self, "diagnostics", tuple(str(message) for message in self.diagnostics))
        if self.status not in {"resolved", "unresolved", "invalid", "ambiguous"}:
            raise ValueError("ResolvedTagValue status is unsupported")
        if self.source not in {"record", "default", "construction", "live"}:
            raise ValueError("ResolvedTagValue source is unsupported")
        if not isinstance(self.subject, tuple) or len(self.subject) != 2:
            raise TypeError("ResolvedTagValue subject must be a (kind, identifier) tuple")
        object.__setattr__(
            self,
            "subject",
            (
                identifier_key(self.subject[0], "tag subject kind"),
                identifier_key(self.subject[1], "tag subject identifier"),
            ),
        )
        for name, value in (("publication", self.publication), ("revision", self.revision)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"ResolvedTagValue {name} must be a non-negative integer")
        if not self.snapshot:
            raise ValueError("ResolvedTagValue snapshot must be non-empty")


@dataclass(frozen=True)
class TagView(Mapping[str, ResolvedTagValue]):
    """Provide one immutable static Tag view over a single Registry snapshot.

    Iteration exposes canonical identifiers for known definitions and exact raw
    identifiers for unknown claims. Lookup accepts either canonical identifiers
    or active definition aliases.

    Args:
        subject (tuple[str, str]): Canonical subject kind and identifier.
        raw (Mapping[str, Any]): Exact immutable authored `meta.tags` mapping.
        publication (int): Process-local Registry publication.
        revision (int): Durable Registry revision.
        snapshot (str): Stable token for the captured Registry state.
    """

    subject: tuple[str, str]
    raw: Mapping[str, Any]
    publication: int
    revision: int
    snapshot: str
    _claims: Mapping[str, ResolvedTagValue] = field(repr=False)
    _aliases: Mapping[str, str] = field(repr=False)
    _definitions: Mapping[str, TagDefinition] = field(repr=False)
    _record_states: Mapping[tuple[str, str], bool] = field(repr=False)

    def __post_init__(self) -> None:
        """Validate and freeze one public mapping facade.

        Args:
            None.

        Returns:
            None: The dataclass fields are normalized in place.

        Raises:
            TypeError: If subject, raw data, claims, or evidence have invalid types.
            ValueError: If identities or snapshot evidence are invalid.
        """
        if not isinstance(self.subject, tuple) or len(self.subject) != 2:
            raise TypeError("TagView subject must be a (kind, identifier) tuple")
        object.__setattr__(
            self,
            "subject",
            (
                identifier_key(self.subject[0], "tag subject kind"),
                identifier_key(self.subject[1], "tag subject identifier"),
            ),
        )
        if not isinstance(self.raw, Mapping):
            raise TypeError("TagView raw claims must be a mapping")
        object.__setattr__(self, "raw", _freeze(self.raw))
        object.__setattr__(self, "_claims", MappingProxyType(dict(self._claims)))
        object.__setattr__(self, "_aliases", MappingProxyType(dict(self._aliases)))
        object.__setattr__(self, "_definitions", MappingProxyType(dict(self._definitions)))
        object.__setattr__(self, "_record_states", MappingProxyType(dict(self._record_states)))
        for name, value in (("publication", self.publication), ("revision", self.revision)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"TagView {name} must be a non-negative integer")
        if not self.snapshot:
            raise ValueError("TagView snapshot must be non-empty")

    def __getitem__(self, identifier: str) -> ResolvedTagValue:
        """Return one result by canonical identifier, alias, or unknown raw key.

        Args:
            identifier (str): Tag identifier or active definition alias.

        Returns:
            ResolvedTagValue: Snapshot-bound claim or default result.

        Raises:
            KeyError: If no explicit claim or applicable default exists.
        """
        if not isinstance(identifier, str):
            raise TypeError("TagView lookup identifier must be a string")
        key = identifier
        return self._claims[self._aliases.get(key, key)]

    def __iter__(self) -> Iterator[str]:
        """Iterate canonical known identifiers and exact unknown raw keys.

        Args:
            None.

        Returns:
            Iterator[str]: Stable snapshot-order identifier iterator.
        """
        return iter(self._claims)

    def __len__(self) -> int:
        """Return the number of explicit claims and applicable defaults.

        Args:
            None.

        Returns:
            int: Number of values in this immutable view.
        """
        return len(self._claims)

    def overlay(
        self,
        values: Mapping[str, Any],
        *,
        source: Literal["construction", "live"],
        instance: Any,
    ) -> TagView:
        """Overlay construction or live values with whole-value replacement.

        Chaining a construction overlay and then a live overlay implements the
        accepted precedence while preserving this view's static raw claims and
        Registry snapshot. Known values use the same declarative validator as
        static claims. Unknown identifiers remain unresolved raw metadata.

        Args:
            values (Mapping[str, Any]): Tag identifiers or aliases mapped to
                explicit overlay values.
            source (Literal["construction", "live"]): Effective-value layer.
                Supported values:
                - `construction`: Value derived while constructing the concrete object.
                - `live`: Best-effort current value from the concrete object.
            instance (Any): Concrete object provenance retained on every
                overlaid result.

        Returns:
            TagView: New immutable effective view over the same Registry snapshot.

        Raises:
            TypeError: If `values` is not a string-keyed mapping.
            ValueError: If `source` is unsupported or an identifier is invalid.
        """
        if not isinstance(values, Mapping) or not all(isinstance(key, str) for key in values):
            raise TypeError("TagView overlay values must be a string-keyed mapping")
        if source not in {"construction", "live"}:
            raise ValueError("TagView overlay source must be 'construction' or 'live'")
        for raw_key, raw_value in values.items():
            _validate_raw_json(raw_value, f"{source}.{raw_key}")
        grouped: dict[str, list[tuple[str, Any]]] = {}
        for raw_key, raw_value in values.items():
            key = raw_key
            canonical = self._aliases.get(key, key)
            grouped.setdefault(canonical, []).append((key, raw_value))
        claims = dict(self._claims)
        for canonical, candidates in grouped.items():
            raw_keys = tuple(key for key, _ in candidates)
            raw: Any = candidates[0][1] if len(candidates) == 1 else {key: value for key, value in candidates}
            definition = self._definitions.get(canonical)
            diagnostics: list[str] = []
            if len(candidates) > 1:
                claims[canonical] = ResolvedTagValue(
                    canonical,
                    None,
                    raw,
                    raw_keys,
                    "ambiguous",
                    source,
                    self.subject,
                    definition,
                    (f"Overlay keys {list(raw_keys)!r} resolve to the same Tag {canonical!r}",),
                    self.publication,
                    self.revision,
                    self.snapshot,
                    instance,
                )
                continue
            if definition is None:
                claims[canonical] = ResolvedTagValue(
                    canonical,
                    raw,
                    raw,
                    raw_keys,
                    "unresolved",
                    source,
                    self.subject,
                    None,
                    (f"Tag definition {canonical!r} is absent",),
                    self.publication,
                    self.revision,
                    self.snapshot,
                    instance,
                )
                continue
            if not definition.valid:
                claims[canonical] = ResolvedTagValue(
                    canonical,
                    None,
                    raw,
                    raw_keys,
                    "invalid",
                    source,
                    self.subject,
                    definition,
                    definition.diagnostics,
                    self.publication,
                    self.revision,
                    self.snapshot,
                    instance,
                )
                continue
            if source not in definition.layers:
                diagnostics.append(f"Tag {canonical!r} does not declare the {source} layer")
            if definition.subject_policy == "exact" and self.subject[0] not in definition.subjects:
                diagnostics.append(f"Tag {canonical!r} does not declare subject kind {self.subject[0]!r}")
            references: tuple[_Reference, ...] = ()
            try:
                value, references = _validate_value(definition.schema, raw, f"{source}.{canonical}")
            except (TypeError, ValueError) as error:
                diagnostics.append(str(error))
                value = None
            reference_diagnostics = _reference_diagnostics(references, self._record_states)
            claims[canonical] = ResolvedTagValue(
                canonical,
                value,
                raw,
                raw_keys,
                "invalid" if diagnostics else ("unresolved" if reference_diagnostics else "resolved"),
                source,
                self.subject,
                definition,
                (*diagnostics, *reference_diagnostics),
                self.publication,
                self.revision,
                self.snapshot,
                instance,
            )
        return TagView(
            self.subject,
            self.raw,
            self.publication,
            self.revision,
            self.snapshot,
            claims,
            self._aliases,
            self._definitions,
            self._record_states,
        )


@dataclass(frozen=True)
class _CapturedRows:
    """Hold one detached all-record snapshot for Tag interpretation.

    ``derived`` memoizes the immutable definitions and subject views built from
    this snapshot; a new publication yields a new capture, so nothing here is
    ever invalidated in place.
    """

    rows: tuple[dict[str, Any], ...]
    publication: int
    revision: int
    snapshot: str
    record_states: Mapping[tuple[str, str], bool]
    derived: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def _captured_rows(resolver: ModuleResolver) -> _CapturedRows:
    """Capture every record once without resolving or importing any target.

    A service catalog keeps one capture per publication on its inspection rows,
    so repeated Tag reads between publications share the definitions and views
    built from it. A generic resolver is captured on every call.
    """
    selected = module_resolver(resolver)
    catalog = getattr(selected, "catalog", None)
    capture_rows = getattr(catalog, "_capture_rows", None)
    memo: dict[str, Any] | None = None
    if callable(capture_rows):
        captured = capture_rows()
        memo = getattr(captured, "derived", None)
        if isinstance(memo, dict) and "tags" in memo:
            return memo["tags"]
        view = captured.view
        rows = list(captured.all())
        publication = int(view.publication)
        revision = int(view.revision)
        durable_fingerprint = view.fingerprint
    else:
        for _attempt in range(3):
            before_publication = int(getattr(catalog, "publication", 0))
            before_revision = getattr(catalog, "revision", None)
            rows = [copy.deepcopy(row) for row in selected.inspect(include_inactive=True)]
            after_publication = int(getattr(catalog, "publication", 0))
            after_revision = getattr(catalog, "revision", None)
            publications = {int(value) for row in rows if (value := row.get("publication")) is not None}
            revisions = {int(value) for row in rows if (value := row.get("revision")) is not None}
            if len(publications) > 1 or len(revisions) > 1:
                raise RuntimeError("ModuleResolver.inspect returned records from multiple Registry snapshots")
            row_publication = next(iter(publications), after_publication)
            if before_publication != after_publication or row_publication != after_publication:
                continue
            if before_revision is not None and after_revision is not None and before_revision != after_revision:
                continue
            publication = after_publication
            revision = int(next(iter(revisions), after_revision if after_revision is not None else 0))
            if revisions and after_revision is not None and revision != int(after_revision):
                continue
            durable_fingerprint = None
            break
        else:
            raise RuntimeError("ModuleResolver.inspect could not provide one stable Registry snapshot")
    evidence = [
        [
            str(row["kind"]),
            str(row["identifier"]),
            list(row.get("aliases") or ()),
            str(row.get("fingerprint") or ""),
            bool(row["record"].get("active", True)),
        ]
        for row in rows
    ]
    derived = sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    fingerprint = durable_fingerprint if isinstance(durable_fingerprint, str) and durable_fingerprint else derived
    record_states: dict[tuple[str, str], bool] = {}
    for row in rows:
        active = bool(row["record"].get("active", True))
        kind = str(row["kind"])
        record_states[(kind, str(row["identifier"]))] = active
        for alias in row.get("aliases") or ():
            record_states[(kind, str(alias))] = active
    result = _CapturedRows(
        tuple(rows),
        publication,
        revision,
        f"{publication}:{revision}:{fingerprint}",
        MappingProxyType(record_states),
    )
    if isinstance(memo, dict):
        memo["tags"] = result
    return result


def _definition_from_row(row: Mapping[str, Any], captured: _CapturedRows) -> TagDefinition:
    """Build one valid or locally invalid definition facade."""
    identifier = str(row["identifier"])
    aliases = tuple(str(alias) for alias in row.get("aliases") or ())
    record = _mapping(row["record"], f"tag {identifier!r} record")
    meta = record.get("meta")
    target = record.get("target")
    raw_definition: Any = None
    diagnostics: list[str] = []
    if record.get("source") != "inline" or not isinstance(meta, Mapping) or not isinstance(target, str):
        diagnostics.append(f"Tag definition {identifier!r} must use an inert inline target")
    else:
        raw_definition = meta.get(target)
        if not isinstance(raw_definition, Mapping):
            diagnostics.append(f"Tag definition {identifier!r} target {target!r} must be a mapping")
    definition = dict(raw_definition) if isinstance(raw_definition, Mapping) else {}
    unknown = set(definition) - {"schema", "default", "subjects", "subject_policy", "layers"}
    if unknown:
        diagnostics.append(f"Tag definition {identifier!r} has unsupported fields {sorted(unknown)!r}")

    subjects: tuple[str, ...] = ()
    raw_subjects = definition.get("subjects", [])
    if not isinstance(raw_subjects, list):
        diagnostics.append(f"Tag definition {identifier!r} subjects must be a list")
    else:
        try:
            subjects = tuple(identifier_key(subject, f"Tag definition {identifier!r} subject") for subject in raw_subjects)
            if len(set(subjects)) != len(subjects):
                diagnostics.append(f"Tag definition {identifier!r} repeats a subject")
        except (TypeError, ValueError) as error:
            diagnostics.append(str(error))

    subject_policy: TagSubjectPolicy = "exact"
    raw_policy = definition.get("subject_policy", "exact")
    if raw_policy not in {"exact", "category-open"}:
        diagnostics.append(f"Tag definition {identifier!r} subject_policy must be 'exact' or 'category-open'")
    else:
        subject_policy = cast(TagSubjectPolicy, raw_policy)

    layers: tuple[str, ...] = ("static",)
    raw_layers = definition.get("layers", ["static"])
    if not isinstance(raw_layers, list) or not raw_layers:
        diagnostics.append(f"Tag definition {identifier!r} layers must be a non-empty list")
    else:
        try:
            layers = tuple(identifier_key(layer, f"Tag definition {identifier!r} layer") for layer in raw_layers)
            unknown_layers = set(layers) - _LAYERS
            if unknown_layers:
                diagnostics.append(f"Tag definition {identifier!r} has unsupported layers {sorted(unknown_layers)!r}")
                layers = ("static",)
            if len(set(layers)) != len(layers):
                diagnostics.append(f"Tag definition {identifier!r} repeats a layer")
        except (TypeError, ValueError) as error:
            diagnostics.append(str(error))

    schema: Mapping[str, Any] = {}
    try:
        schema = _normalize_schema(definition.get("schema"))
    except (TypeError, ValueError) as error:
        diagnostics.append(str(error))
        if isinstance(definition.get("schema"), Mapping):
            schema = cast(Mapping[str, Any], definition["schema"])

    has_default = "default" in definition
    default = definition.get("default")
    if has_default and "type" in schema:
        try:
            default, _ = _validate_value(schema, default, f"Tag definition {identifier!r} default")
        except (TypeError, ValueError) as error:
            diagnostics.append(str(error))

    return TagDefinition(
        identifier=identifier,
        aliases=aliases,
        schema=schema,
        default=default,
        has_default=has_default,
        subjects=subjects,
        subject_policy=subject_policy,
        layers=layers,
        name=str(record.get("name") or identifier),
        desc=str(record.get("desc") or ""),
        status="invalid" if diagnostics else "resolved",
        diagnostics=tuple(diagnostics),
        publication=captured.publication,
        revision=captured.revision,
        snapshot=captured.snapshot,
        fingerprint=str(row.get("fingerprint") or _record_fingerprint(record)),
    )


def _definitions(captured: _CapturedRows) -> tuple[tuple[TagDefinition, ...], dict[str, str]]:
    """Return active definitions and their canonical/alias lookup index, built once per capture."""
    cached = captured.derived.get("definitions")
    if cached is not None:
        return cached
    definitions: list[TagDefinition] = []
    lookup: dict[str, str] = {}
    for row in captured.rows:
        if row["kind"] != "tag" or not row["record"].get("active", True):
            continue
        definition = _definition_from_row(row, captured)
        definitions.append(definition)
        for key in (definition.identifier, *definition.aliases):
            existing = lookup.get(key)
            if existing is not None and existing != definition.identifier:
                raise ValueError(f"Tag definition alias {key!r} resolves to both {existing!r} and {definition.identifier!r}")
            lookup[key] = definition.identifier
    result = tuple(definitions), lookup
    captured.derived["definitions"] = result
    return result


def _subject_row(captured: _CapturedRows, kind: str, identifier: str) -> dict[str, Any]:
    """Return one active canonical subject row by identifier or alias."""
    clean_kind = identifier_key(kind, "tag subject kind")
    key = named_key(identifier, "tag subject identifier or alias")
    for row in captured.rows:
        if row["kind"] != clean_kind or not row["record"].get("active", True):
            continue
        if row["identifier"] == key or key in (row.get("aliases") or ()):
            return row
    raise KeyError(f"Unknown active module subject {(clean_kind, key)!r}")


def _reference_diagnostics(
    references: Sequence[_Reference],
    record_states: Mapping[tuple[str, str], bool],
) -> tuple[str, ...]:
    """Return local diagnostics for unresolved soft record references."""
    diagnostics: list[str] = []
    for reference in references:
        active = record_states.get((reference.kind, reference.identifier))
        if active is None:
            diagnostics.append(f"{reference.path} references missing {reference.kind}:{reference.identifier}")
        elif reference.state == "active" and not active:
            diagnostics.append(f"{reference.path} references inactive {reference.kind}:{reference.identifier}")
    return tuple(diagnostics)


def _resolved_value(
    *,
    definition: TagDefinition | None,
    raw: Any,
    raw_keys: tuple[str, ...],
    source: Literal["record", "default"],
    subject: tuple[str, str],
    captured: _CapturedRows,
) -> ResolvedTagValue:
    """Resolve one explicit claim or applicable default locally."""
    tag_identifier = definition.identifier if definition is not None else raw_keys[0]
    if len(raw_keys) > 1:
        return ResolvedTagValue(
            tag_identifier,
            None,
            MappingProxyType({key: _freeze(value) for key, value in raw.items()}),
            raw_keys,
            "ambiguous",
            source,
            subject,
            definition,
            (f"Raw keys {list(raw_keys)!r} resolve to the same Tag {tag_identifier!r}",),
            captured.publication,
            captured.revision,
            captured.snapshot,
        )
    if definition is None:
        return ResolvedTagValue(
            tag_identifier,
            raw,
            raw,
            raw_keys,
            "unresolved",
            source,
            subject,
            None,
            (f"Tag definition {tag_identifier!r} is absent",),
            captured.publication,
            captured.revision,
            captured.snapshot,
        )
    if not definition.valid:
        return ResolvedTagValue(
            tag_identifier,
            None,
            raw,
            raw_keys,
            "invalid",
            source,
            subject,
            definition,
            definition.diagnostics,
            captured.publication,
            captured.revision,
            captured.snapshot,
        )
    diagnostics: list[str] = []
    if source == "record" and "static" not in definition.layers:
        diagnostics.append(f"Tag {tag_identifier!r} does not declare the static layer")
    if definition.subject_policy == "exact" and subject[0] not in definition.subjects:
        diagnostics.append(f"Tag {tag_identifier!r} does not declare subject kind {subject[0]!r}")
    try:
        candidate = _wire(raw) if source == "default" else raw
        value, references = _validate_value(definition.schema, candidate, f"{subject[0]}:{subject[1]}.meta.tags.{raw_keys[0]}")
    except (TypeError, ValueError) as error:
        diagnostics.append(str(error))
        return ResolvedTagValue(
            tag_identifier,
            None,
            raw,
            raw_keys,
            "invalid",
            source,
            subject,
            definition,
            tuple(diagnostics),
            captured.publication,
            captured.revision,
            captured.snapshot,
        )
    reference_diagnostics = _reference_diagnostics(references, captured.record_states)
    return ResolvedTagValue(
        tag_identifier,
        value,
        raw,
        raw_keys,
        "invalid" if diagnostics else ("unresolved" if reference_diagnostics else "resolved"),
        source,
        subject,
        definition,
        (*diagnostics, *reference_diagnostics),
        captured.publication,
        captured.revision,
        captured.snapshot,
    )


def _tag_definition(captured: _CapturedRows, identifier: str) -> TagDefinition | None:
    """Return one active definition from an already captured snapshot."""
    key = named_key(identifier, "tag identifier or alias")
    definitions, lookup = _definitions(captured)
    canonical = lookup.get(key)
    if canonical is None:
        return None
    return next(definition for definition in definitions if definition.identifier == canonical)


def _tags(captured: _CapturedRows, kind: str, identifier: str) -> TagView:
    """Return one static subject view from an already captured snapshot, built once per subject."""
    views: dict[tuple[str, str], TagView] = captured.derived.setdefault("views", {})
    cached = views.get((kind, identifier))
    if cached is not None:
        return cached
    view = _subject_view(captured, kind, identifier)
    views[(kind, identifier)] = view
    return view


def _subject_view(captured: _CapturedRows, kind: str, identifier: str) -> TagView:
    """Build one static subject view from an already captured snapshot."""
    row = _subject_row(captured, kind, identifier)
    subject = str(row["kind"]), str(row["identifier"])
    meta = row["record"].get("meta")
    raw = meta.get("tags", {}) if isinstance(meta, Mapping) else {}
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise TypeError(f"Module subject {subject!r} meta.tags must be a string-keyed mapping")
    definitions, alias_index = _definitions(captured)
    by_identifier = {definition.identifier: definition for definition in definitions}
    grouped: dict[str, list[tuple[str, Any]]] = {}
    for raw_key, value in raw.items():
        canonical = alias_index.get(raw_key, raw_key)
        grouped.setdefault(canonical, []).append((raw_key, value))

    claims: dict[str, ResolvedTagValue] = {}
    for canonical, values in grouped.items():
        definition = by_identifier.get(canonical)
        raw_keys = tuple(key for key, _ in values)
        raw_value: Any = values[0][1] if len(values) == 1 else {key: value for key, value in values}
        claims[canonical] = _resolved_value(
            definition=definition,
            raw=raw_value,
            raw_keys=raw_keys,
            source="record",
            subject=subject,
            captured=captured,
        )

    for definition in definitions:
        if definition.identifier in claims or not definition.valid or not definition.has_default or subject[0] not in definition.subjects:
            continue
        claims[definition.identifier] = _resolved_value(
            definition=definition,
            raw=definition.default,
            raw_keys=(definition.identifier,),
            source="default",
            subject=subject,
            captured=captured,
        )
    view_aliases = {alias: canonical for alias, canonical in alias_index.items() if alias != canonical}
    return TagView(
        subject=subject,
        raw=cast(Mapping[str, Any], raw),
        publication=captured.publication,
        revision=captured.revision,
        snapshot=captured.snapshot,
        _claims=claims,
        _aliases=view_aliases,
        _definitions=by_identifier,
        _record_states=captured.record_states,
    )


def tag_definition(resolver: ModuleResolver, identifier: str) -> TagDefinition | None:
    """Inspect one optional Tag definition without importing its target.

    Args:
        resolver (ModuleResolver): Context-owned module read authority.
        identifier (str): Canonical Tag identifier or Registry alias.

    Returns:
        TagDefinition | None: Snapshot-bound valid or locally invalid
        definition, or `None` when no active definition exists.

    Raises:
        TypeError: If the resolver or stored definition record is malformed.
        ValueError: If the requested identifier or Registry indexes are invalid.
        RuntimeError: If a generic resolver returns a torn multi-snapshot read.
    """
    return _tag_definition(_captured_rows(resolver), identifier)


def tags(resolver: ModuleResolver, kind: str, identifier: str) -> TagView:
    """Inspect raw claims and applicable typed defaults for one module record.

    Args:
        resolver (ModuleResolver): Context-owned module read authority.
        kind (str): Subject module-record kind.
        identifier (str): Canonical subject identifier or Registry alias.

    Returns:
        TagView: Immutable raw and typed view from exactly one Registry snapshot.

    Raises:
        KeyError: If the active subject is absent.
        TypeError: If the resolver or subject's raw Tag mapping is malformed.
        ValueError: If identifiers, definitions, or Registry indexes are invalid.
        RuntimeError: If a generic resolver returns a torn multi-snapshot read.
    """
    return _tags(_captured_rows(resolver), kind, identifier)


def tag(
    resolver: ModuleResolver,
    kind: str,
    identifier: str,
    tag_id: str,
) -> ResolvedTagValue | None:
    """Inspect one static claim or applicable default for a module record.

    Args:
        resolver (ModuleResolver): Context-owned module read authority.
        kind (str): Subject module-record kind.
        identifier (str): Canonical subject identifier or Registry alias.
        tag_id (str): Canonical Tag identifier, definition alias, or raw
            unknown claim identifier.

    Returns:
        ResolvedTagValue | None: Snapshot-bound result, or `None` when the
        subject has neither an explicit claim nor an applicable default.

    Raises:
        KeyError: If the active subject is absent.
        TypeError: If the resolver or subject's raw Tag mapping is malformed.
        ValueError: If identifiers, definitions, or Registry indexes are invalid.
        RuntimeError: If a generic resolver returns a torn multi-snapshot read.
    """
    return _tags(_captured_rows(resolver), kind, identifier).get(tag_id)
