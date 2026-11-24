"""Catalog discoverability policy for every query input surface."""

from __future__ import annotations

__all__ = ["CatalogAvailable"]

from time import time

from heavenbase.utils import Any, Mapping, dataclass, raise_mismatch

_CATALOG_ENTITY_ID = "sys-catalog"
_OPTION_KEYS = ("invisible", "now", "stale")


@dataclass(frozen=True)
class CatalogAvailable:
    """Normalize and render the Catalog discoverability policy.

    Args:
        now (Any): Evaluation instant as epoch milliseconds, `int`, `float`,
            `datetime`, or ISO-8601 string. `None` snapshots the current clock.
        stale (bool): Whether stale rows remain discoverable.
        invisible (bool): Whether hidden rows remain discoverable.

    Raises:
        TypeError: If `stale` or `invisible` is not a boolean.
        ValueError: If `now` cannot be normalized to epoch milliseconds.
    """

    now: Any = None
    stale: bool = False
    invisible: bool = False

    def __post_init__(self) -> None:
        for name in ("stale", "invisible"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Catalog available {name} must be a boolean")
        object.__setattr__(self, "now", int(time() * 1000) if self.now is None else _coerce_now(self.now))

    @classmethod
    def load(cls, value: Any) -> "CatalogAvailable | None":
        """Load an `available` or `$available` input value.

        Args:
            value (Any): `True` for the default policy, `False` or `None` to
                disable it, or a mapping with `now`, `stale`, and `invisible`.

        Returns:
            CatalogAvailable | None: Normalized policy when enabled, otherwise
            `None`.

        Raises:
            TypeError: If the value or a boolean option has an invalid shape.
            ValueError: If an option is unsupported or `now` is invalid.
        """
        if value is False or value is None:
            return None
        if value is True:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError(f"Catalog available expects true, false, or an object, got {type(value).__name__}")
        for key in value:
            if key not in _OPTION_KEYS:
                raise_mismatch(_OPTION_KEYS, key, name="Catalog available key")
        return cls(now=value.get("now"), stale=value.get("stale", False), invisible=value.get("invisible", False))

    def check(self, entity_id: str) -> "CatalogAvailable":
        """Require this policy to target the system Catalog entity.

        Args:
            entity_id (str): Entity identifier receiving the availability
                policy.

        Returns:
            CatalogAvailable: This validated policy for fluent composition.

        Raises:
            ValueError: If `entity_id` is not `sys-catalog`.
        """
        if entity_id != _CATALOG_ENTITY_ID:
            raise ValueError(f"available applies only to Catalog ({_CATALOG_ENTITY_ID!r}), got {entity_id!r}")
        return self

    def to_expr(self) -> "Expr":
        """Return the policy as the canonical typed query expression.

        Args:
            None.

        Returns:
            Expr: Predicate tree for active, visible, unexpired, and fresh rows
            according to this policy.
        """
        from ..filters.spec import And, Expr, Filter

        clauses: list[Expr] = [Filter(field, op, value) for field, op, value in self._rules()]
        return clauses[0] if len(clauses) == 1 else And(clauses)

    def to_dict(self) -> dict[str, Any]:
        """Return the policy as a Mongo-style query filter.

        Args:
            None.

        Returns:
            dict[str, Any]: `$and` filter accepted by JSON and mapping query
            surfaces.
        """
        predicates = [{field: value if op == "eq" else {"$gt": value}} for field, op, value in self._rules()]
        return {"$and": predicates}

    def _rules(self) -> list[tuple[str, str, Any]]:
        rules: list[tuple[str, str, Any]] = [("active", "eq", True), ("expires_compute", "gt", self.now)]
        if not self.invisible:
            rules.insert(1, ("visible", "eq", True))
        if not self.stale:
            rules.append(("stale_compute", "gt", self.now))
        return rules


def _coerce_now(value: Any) -> int:
    from ..types.timestamp import Timestamp

    try:
        return Timestamp().validate(value)
    except ValueError as exc:
        raise ValueError(f"Catalog available now expects epoch ms, int, float, datetime, or ISO timestamp, got {value!r}") from exc
