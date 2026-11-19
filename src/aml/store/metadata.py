"""Strict model and reference metadata authority for plan.md §5, §6, and §10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, model_validator

_LITERAL_FIELDS: dict[type, tuple[tuple[str, frozenset[tuple[type[Any], Any]]], ...]] = {}
"""Each strict model's Literal fields and their exact ``(type, option)`` pairs, resolved once.

``get_origin``/``get_args`` walk the annotation on every call, and the check
below runs before every validation of every model: under plan section 9
normal load that introspection was 3.3% of the process CPU (capacity profile,
2026-09-12). The fields of a class never change after it is defined. A
supplied value is exact when its ``(type, value)`` pair is one of them: set
membership compares the types by identity and the values by equality, as a
scan of the options would, without one generator per checked field (the 976
positions of an a7 Add run it millions of times).
"""


def _literal_fields(model: type) -> tuple[tuple[str, frozenset[tuple[type[Any], Any]]], ...]:
    """Return ``(name, exact pairs)`` for each Literal field this model declares."""
    resolved = _LITERAL_FIELDS.get(model)
    if resolved is None:
        fields = cast(Any, model).model_fields
        resolved = tuple(
            (name, _exact_pairs(get_args(field.annotation)))
            for name, field in fields.items()
            if get_origin(field.annotation) is Literal
        )
        _LITERAL_FIELDS[model] = resolved
    return resolved


def _exact_pairs(options: tuple[object, ...]) -> frozenset[tuple[type[Any], Any]]:
    """Return the exact ``(type, option)`` pair of each Literal option."""
    return frozenset((type(option), option) for option in options)


class StrictModel(BaseModel):
    """Require exact field types and reject undeclared application JSON fields."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    @model_validator(mode="before")
    @classmethod
    def exact_literal_types(cls, value: Any) -> Any:
        """Reject bool/int equality coercions that Pydantic permits for Literal."""
        if isinstance(value, dict):
            raw = cast("dict[str, Any]", value)
            resolved = _LITERAL_FIELDS.get(cls)
            if resolved is None:
                resolved = _literal_fields(cls)
            for name, exact in resolved:
                if name in raw:
                    supplied: Any = raw[name]
                    try:
                        matched = (type(supplied), supplied) in exact
                    except TypeError:
                        # Unhashable, so unequal to every str, int, bool, None or Enum option.
                        matched = False
                    if not matched:
                        raise ValueError(f"Field {name} requires an exact literal type and value")
            return raw
        return value


@dataclass(frozen=True)
class RefSpec:
    """Declare one authoritative relationship and its independent lifecycle axes.

    Attributes:
        target_kind: Registered domain kind or Artifact/Control target.
        cardinality: Whether the annotated leaf carries one or many locators.
        ref_mode: Existence and fulfillment timing from plan.md §0. No reserved
            or prospective mode: an NL transaction names its rows' ids and
            compares heads explicitly (``CommitPlanV1``, S5b).
        existence_expectation: ``must_exist`` makes a historical locator's
            target required at the write.
        fulfillment: When the target must exist: at the creating write, before it
            (``pre_state``), or never (``none``, a content-free locator).
        scope_rule: Effective scope compatibility rule.
        gc_root: Whether this edge retains its target, independently of existence.
        owner_retention: Optional reverse lifetime dependency on a live target.
        owner_delete_action: Deleting an owner never implicitly deletes its target.
        target_forget_action: Action when the protected target is forgotten.
        artifact_kinds: Closed direct/logical Artifact kind allowlist.
    """

    target_kind: str
    cardinality: Literal["one", "many"] = "one"
    ref_mode: Literal["live", "snapshot_locator", "historical_locator"] = "live"
    existence_expectation: Literal["must_exist"] | None = None
    fulfillment: Literal["creation", "pre_state", "none"] = "creation"
    scope_rule: Literal[
        "same_scope_or_global", "same_scope", "global", "allowed_scopes", "protocol"
    ] = "same_scope_or_global"
    gc_root: bool | Literal["while_owner_nonterminal"] | None = None
    owner_retention: Literal["independent", "while_target_live"] = "independent"
    owner_delete_action: Literal["retain_target"] = "retain_target"
    target_forget_action: (
        Literal["delete_owner", "retain_locator", "clear_ref", "cascade_source_item"] | None
    ) = None
    artifact_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Resolve mode defaults without conflating retention with existence."""
        if self.gc_root is None:
            object.__setattr__(self, "gc_root", self.ref_mode == "live")
        if self.target_forget_action is None:
            action = "retain_locator" if self.ref_mode == "historical_locator" else "delete_owner"
            object.__setattr__(self, "target_forget_action", action)


@dataclass(frozen=True)
class OpaqueIdSpec:
    """Mark an identity that cannot be resolved as an AML object reference."""

    kind: Literal["local", "external", "digest"]


@dataclass(frozen=True)
class DerivedSpec:
    """Name the sole authority used to materialize a read-only projection."""

    source: str


@dataclass(frozen=True)
class ScopeMembershipSpec:
    """Index an authorized scope for privacy without creating a foreign key."""

    owner_forget_action: Literal["delete_owner", "clear_ref"] = "delete_owner"


@dataclass(frozen=True)
class OpaquePayloadSpec:
    """Bound an explicitly external payload; never wrap AML-generated structures."""

    kind: Literal["external"] = "external"
    max_bytes: int = 16384



@dataclass(frozen=True)
class IntegrityBindingSpec:
    """Classify a marker or sequence that proves consistency without owning a relation."""

    source: str
