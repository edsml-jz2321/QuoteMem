"""Per-entity workspace registration options."""

from __future__ import annotations

__all__ = [
    "CATALOG_MODES",
    "EntityOptions",
]

from heavenbase.utils import Any, Dict, Literal, Mapping, Optional, dataclass
from heavenbase.utils.values import _type_label

CatalogMode = Literal["full", "identity", "none"]
CATALOG_MODES: tuple[str, ...] = ("full", "identity", "none")
_OPTION_KEYS = ("catalog", "gram")
# MetaSchema entity-row ``meta`` key recording the routes that joined after the entity's first routes
_ROUTE_GENERATIONS_KEY = "route_generations"


def _validate_gram(value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"gram must be a boolean, got {_type_label(value)}")


@dataclass(frozen=True)
class EntityOptions:
    """Describe how one registered entity participates in Catalog and keyword maintenance.

    The value is chosen at ``HeavenBase.register`` time, travels with the
    entity definition in workspace manifests and MetaSchema entity rows, and
    is compared on re-registration like the schema structure: a different
    value needs ``allow_replace=True``.

    Args:
        catalog (str): Catalog projection mode.
            Supported values:
            - ``"full"``: every row write publishes a projected Catalog row
              (signature, same-name projections, derives) and keyword index
              rows for the projected text. This is the default.
            - ``"identity"``: every row write publishes a Catalog row that
              carries only ``target_id``, ``target_entity``, ``ws``, lifecycle
              fields, and field defaults. The existing Catalog row is read by
              id so its ``created_at``, ``active``, and ``visible`` carry over
              (``updated_at`` is the write time); nothing else is merged from
              it, nothing is projected or derived, and no keyword index rows
              are maintained for it.
            - ``"none"``: no Catalog row is written or deleted for the entity.
        gram (bool): Whether the entity's own ``Array[ShortText]`` fields keep
            a sparse-GRAM keyword index. ``False`` stores those fields inline
            without index maintenance and refuses ``contained_in`` queries on
            them instead of scanning. Defaults to ``True``.
    """

    catalog: CatalogMode = "full"
    gram: bool = True

    def __post_init__(self) -> None:
        """Validate the option values.

        Returns:
            None: This initializer validates in place.

        Raises:
            ValueError: If ``catalog`` is not a supported mode.
            TypeError: If ``gram`` is not a boolean.
        """
        if self.catalog not in CATALOG_MODES:
            raise ValueError(f"catalog must be one of {', '.join(repr(mode) for mode in CATALOG_MODES)}, got {self.catalog!r}")
        _validate_gram(self.gram)

    @classmethod
    def resolve(
        cls,
        existing: Optional["EntityOptions"],
        declared: Optional[Mapping[str, Any]],
        *,
        catalog: Optional[str],
        gram: Optional[bool],
    ) -> "EntityOptions":
        """Combine explicit arguments, definition keys, and the existing registration.

        Each option is taken from the first source that states it: the explicit
        argument, then the ``catalog``/``gram`` keys of a JSON entity
        definition, then the entity's existing registration, then the default.

        Args:
            existing (EntityOptions | None): Options of the existing registration
                with the same entity identifier, if any: the live one, or the one
                the store's MetaSchema entity row recorded when none is live.
            declared (Mapping[str, Any] | None): Entity definition mapping whose
                ``catalog`` and ``gram`` keys may state options.
            catalog (str | None): Explicit Catalog mode, or ``None`` to inherit.
            gram (bool | None): Explicit keyword-index flag, or ``None`` to inherit.

        Returns:
            EntityOptions: Effective options for the registration.

        Raises:
            ValueError: If an explicit argument disagrees with the definition
                key for the same option, or a value is unsupported.
            TypeError: If ``gram`` is not a boolean.
        """
        if gram is not None:
            _validate_gram(gram)
        declared_values = {key: declared[key] for key in _OPTION_KEYS if declared is not None and key in declared}
        explicit = {"catalog": catalog, "gram": gram}
        resolved: Dict[str, Any] = {}
        for key in _OPTION_KEYS:
            value = explicit[key]
            if key in declared_values:
                if value is not None and value != declared_values[key]:
                    raise ValueError(f"register {key}={value!r} conflicts with the entity definition's {key}={declared_values[key]!r}")
                value = declared_values[key]
            if value is None:
                value = getattr(existing, key) if existing is not None else getattr(cls, key)
            resolved[key] = value
        return cls(**resolved)

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-compatible option mapping.

        Returns:
            Dict[str, Any]: ``catalog`` and ``gram`` values.
        """
        return {"catalog": self.catalog, "gram": self.gram}
