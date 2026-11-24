"""Schema IR used by user entities and built-in system entities."""

from __future__ import annotations

__all__ = [
    "EntitySchema",
    "FieldSchema",
]

from heavenbase.utils import Any, Dict, List, Mapping
from heavenbase.utils import dataclass, field

from ..modules import ModuleResolver, module_resolver
from ..types import Identifier, LogicalType, logical_type_from_any, logical_type_from_mapping
from ..utils.values import raise_mismatch
from ..utils.ids import check_entity_id, check_field_name, check_py_name


@dataclass
class FieldSchema:
    """Describe one compiled logical field without runtime hooks.

    Args:
        name (str): Canonical row and query field name.
        dtype (LogicalType): Isolated logical type instance.
        pk (bool): Whether this is the fixed ``object_id`` primary key.
        required (bool): Whether materialization must produce a value.
        default (Any): Default value when ``has_default`` is true.
        has_default (bool): Whether ``default`` is meaningful, including
            explicit ``None``.
        desc (str): Human-facing description.
        tags (List[str]): Search and discovery tags.
        examples (List[Any]): Agent and documentation examples.
    """

    name: str
    dtype: LogicalType
    pk: bool = False
    required: bool = False
    default: Any = None
    has_default: bool = False
    desc: str = ""
    tags: List[str] = field(default_factory=list)
    examples: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
    ) -> "FieldSchema":
        """Build a field schema from a JSON-compatible mapping.

        Args:
            data (Mapping[str, Any]): Serialized field metadata.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            FieldSchema: Normalized compiled field schema.

        Raises:
            TypeError: If ``data`` or its type descriptor is unsupported.
            ValueError: If the field name or type is missing or invalid.
        """
        if not isinstance(data, Mapping):
            raise TypeError("FieldSchema.from_dict expects a mapping")
        name = check_field_name(str(data.get("name") or ""))
        dtype_raw = data.get("dtype")
        if dtype_raw is None:
            raise ValueError(f"Field {name!r} needs dtype")
        if isinstance(dtype_raw, Mapping):
            dtype = logical_type_from_mapping(dict(dtype_raw), resolver=resolver)
        else:
            dtype = logical_type_from_any(dtype_raw, resolver=resolver)
        has_default = bool(data.get("has_default", False))
        return cls(
            name=name,
            dtype=dtype,
            pk=bool(data.get("pk", False)),
            required=bool(data.get("required", False)),
            default=data.get("default") if has_default else None,
            has_default=has_default,
            desc=str(data.get("desc") or data.get("description") or ""),
            tags=list(data.get("tags") or []),
            examples=list(data.get("examples") or []),
        )

    def to_dict(self) -> dict:
        """Serialize this field schema.

        Args:
            None.

        Returns:
            dict: JSON-compatible field metadata.
        """
        return {
            "name": self.name,
            "dtype": self.dtype.to_dict(),
            "pk": self.pk,
            "required": self.required,
            "default": self.default if self.has_default else None,
            "has_default": self.has_default,
            "desc": self.desc,
            "tags": list(self.tags),
            "examples": list(self.examples),
        }

    def to_str(self, format: str = "llm") -> str:
        """Return this field schema as LLM-friendly text.

        Args:
            format: Text format name. ``"llm"`` is the built-in default.

        Returns:
            Human-readable field schema text.

        Raises:
            ValueError: If ``format`` is not supported for standalone fields.
        """
        if format != "llm":
            raise_mismatch(("llm",), format, name="field schema text format")
        from .text import llm_field_lines

        return "\n".join(llm_field_lines(self))

    def __str__(self) -> str:
        """Return the default LLM-friendly field schema text."""
        return self.to_str()


@dataclass
class EntitySchema:
    """Describe one compiled logical entity without runtime policy.

    Args:
        entity_id (str): Stable entity identifier.
        name (str): Python and display class name.
        desc (str): Human-facing entity description.
        tags (List[str]): Search and discovery tags.
        fields (Dict[str, FieldSchema]): Fields keyed by canonical name.
    """

    entity_id: str
    name: str
    desc: str = ""
    tags: List[str] = field(default_factory=list)
    fields: Dict[str, FieldSchema] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
    ) -> "EntitySchema":
        """Build an entity schema from a JSON-compatible mapping.

        Args:
            data (Mapping[str, Any]): Serialized entity and field metadata.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            EntitySchema: Normalized compiled schema.

        Raises:
            TypeError: If entity or field data has an invalid shape.
            ValueError: If identifiers, names, or fields are invalid.
        """
        selected = module_resolver(resolver)
        if not isinstance(data, Mapping):
            raise TypeError("EntitySchema.from_dict expects a mapping")
        entity_id = check_entity_id(str(data.get("entity_id") or ""))
        name = check_py_name(str(data.get("name") or entity_id), "entity name")
        fields_raw = data.get("fields")
        if not isinstance(fields_raw, Mapping):
            raise ValueError(f"Entity {entity_id!r} needs a fields mapping")
        fields: Dict[str, FieldSchema] = {}
        for key, raw in fields_raw.items():
            field_name = check_field_name(str(key))
            if not isinstance(raw, Mapping):
                raise TypeError(f"Field {field_name!r} definition must be a mapping")
            field_data = dict(raw)
            if "name" not in field_data:
                field_data["name"] = field_name
            field_schema = FieldSchema.from_dict(field_data, resolver=selected)
            if field_schema.name != field_name:
                raise ValueError(f"Field key {field_name!r} does not match field name {field_schema.name!r}")
            fields[field_name] = field_schema
        return cls(
            entity_id=entity_id,
            name=name,
            desc=str(data.get("desc") or data.get("description") or ""),
            tags=list(data.get("tags") or []),
            fields=fields,
        )

    def set(self, field_schema: FieldSchema) -> None:
        """Add or replace one compiled field.

        Args:
            field_schema (FieldSchema): Field metadata to install.

        Returns:
            None: This method updates ``fields`` in place.

        Raises:
            TypeError: If ``field_schema`` is not a ``FieldSchema``.
        """
        if not isinstance(field_schema, FieldSchema):
            raise TypeError("EntitySchema.set expects a FieldSchema")
        self.fields[field_schema.name] = field_schema

    def get(self, name: str) -> FieldSchema:
        """Return one compiled field.

        Args:
            name (str): Canonical field name.

        Returns:
            FieldSchema: Matching field metadata.

        Raises:
            KeyError: If no field has ``name``.
        """
        try:
            return self.fields[name]
        except KeyError:
            raise KeyError(f"{self.entity_id} has no field {name!r}") from None

    @property
    def pk_name(self) -> str:
        """Return the primary-key field name.

        Args:
            None.

        Returns:
            str: Marked field name, or canonical ``object_id`` fallback.
        """
        for field_schema in self.fields.values():
            if field_schema.pk:
                return field_schema.name
        return "object_id"

    def validate(self) -> None:
        """Validate entity and primary-key structural invariants.

        Args:
            None.

        Returns:
            None: Valid schemas return without mutation.

        Raises:
            TypeError: If ``object_id`` is not an ``Identifier`` field.
            ValueError: If names, fields, or primary-key structure are invalid.
        """
        check_entity_id(self.entity_id)
        check_py_name(self.name, "entity name")
        if not self.fields:
            raise ValueError(f"Entity {self.entity_id!r} has no fields")
        pk_fields: List[str] = []
        for key, field_schema in self.fields.items():
            if key != field_schema.name:
                raise ValueError(f"Field key {key!r} does not match field name {field_schema.name!r} in entity {self.entity_id!r}")
            check_field_name(field_schema.name)
            if field_schema.pk:
                pk_fields.append(field_schema.name)
        if len(pk_fields) > 1:
            raise ValueError(f"Entity {self.entity_id!r} has multiple primary keys: {pk_fields}")
        if pk_fields and pk_fields[0] != "object_id":
            raise ValueError(f"Only object_id may be primary key; got {pk_fields[0]!r}")
        if "object_id" not in self.fields:
            raise ValueError(f"Entity {self.entity_id!r} must define object_id")
        object_id = self.fields["object_id"]
        if type(object_id.dtype).identifier != Identifier.identifier:
            raise TypeError("Entity object_id field must use logical type Identifier")
        if not object_id.pk:
            raise ValueError("Entity object_id field must be marked as primary key")

    def to_dict(self) -> dict:
        """Serialize this entity schema.

        Args:
            None.

        Returns:
            dict: JSON-compatible entity and field metadata.
        """
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "desc": self.desc,
            "tags": list(self.tags),
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
        }

    def to_str(
        self,
        format: str = "llm",
        *,
        resolver: ModuleResolver | None = None,
    ) -> str:
        """Return this entity schema as text for LLM and agent boundaries.

        Args:
            format: Registered schema text format. ``"llm"`` is the built-in
                default for structured extraction prompts.
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.

        Returns:
            Text representation of the entity schema.

        Raises:
            ValueError: If ``format`` is not registered.
        """
        from .text import resolve_schema_text

        return resolve_schema_text(format, resolver=resolver)(self)

    def __str__(self) -> str:
        """Return the default LLM-friendly entity schema text."""
        return self.to_str()
