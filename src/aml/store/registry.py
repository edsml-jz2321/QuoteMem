"""Generate schema leaves and reference extractors from strict model metadata.

The generated inventory is the sole exhaustive authority described by plan.md
§5, §6, and §10. Runtime extraction walks those same annotated model fields.
"""

from __future__ import annotations

import re
import types
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal, TypeAliasType, Union, cast, get_args, get_origin

from pydantic import BaseModel

from aml.core.errors import AMLError
from aml.store.metadata import (
    DerivedSpec,
    IntegrityBindingSpec,
    OpaqueIdSpec,
    OpaquePayloadSpec,
    RefSpec,
    ScopeMembershipSpec,
    StrictModel,
)

Classification = Literal["own_attribute", "authoritative_ref", "derived_index", "integrity_binding"]
_IDENTITY = re.compile(r"(?:^id$|_ids?$|_refs?$|_digest$|_hash$)")
_METADATA_TYPES = (
    RefSpec,
    OpaqueIdSpec,
    DerivedSpec,
    ScopeMembershipSpec,
    OpaquePayloadSpec,
    IntegrityBindingSpec,
)


@dataclass(frozen=True)
class SchemaVariant:
    """Register one implemented discriminant branch of a physical Entity."""

    entity: str
    branch: str
    model: type[StrictModel]


@dataclass(frozen=True)
class SchemaLeaf:
    """Describe one materialized leaf and all declared lifecycle metadata."""

    entity: str
    branch: str
    path: str
    classification: Classification
    required: bool
    nullable: bool
    ref: RefSpec | None
    opaque: OpaqueIdSpec | None
    derived: DerivedSpec | None
    membership: ScopeMembershipSpec | None


@dataclass(frozen=True)
class Reference:
    """Bind one actual locator to the declaration that governs it."""

    path: str
    target_id: str
    spec: RefSpec


@dataclass(frozen=True)
class ScopeMembership:
    """Expose an authorized scope locator for privacy reverse lookup."""

    path: str
    scope_id: str
    spec: ScopeMembershipSpec


@dataclass(frozen=True)
class TextLeaf:
    """One string a content scan inspects, located by its content-free schema path."""

    path: str
    value: str


@dataclass(frozen=True)
class Registry:
    """Hold a validated, deterministic inventory generated from model declarations."""

    leaves: tuple[SchemaLeaf, ...]

    def report(self) -> list[dict[str, Any]]:
        """Return a JSON-compatible report suitable for recorded conformance evidence."""
        return [asdict(leaf) for leaf in self.leaves]


def _unwrap(annotation: Any) -> tuple[Any, list[Any]]:
    metadata: list[Any] = []
    while get_origin(annotation) is Annotated:
        annotation, *extra = get_args(annotation)
        metadata.extend(extra)
    return annotation, metadata


def _one(metadata: list[Any], kind: type[Any]) -> Any:
    found = [item for item in metadata if isinstance(item, kind)]
    if len(found) > 1:
        raise AMLError("registry_invalid", f"Duplicate {kind.__name__} metadata")
    return found[0] if found else None


def _describe(
    metadata: list[Any], path: str, *, null_only: bool = False
) -> tuple[
    Classification,
    RefSpec | None,
    OpaqueIdSpec | None,
    DerivedSpec | None,
    ScopeMembershipSpec | None,
]:
    ref: RefSpec | None = _one(metadata, RefSpec)
    opaque: OpaqueIdSpec | None = _one(metadata, OpaqueIdSpec)
    derived: DerivedSpec | None = _one(metadata, DerivedSpec)
    member: ScopeMembershipSpec | None = _one(metadata, ScopeMembershipSpec)
    payload: OpaquePayloadSpec | None = _one(metadata, OpaquePayloadSpec)
    integrity: IntegrityBindingSpec | None = _one(metadata, IntegrityBindingSpec)
    if ref is not None and opaque is not None:
        raise AMLError("registry_invalid", f"Reference is also declared opaque: {path}")
    if member is not None and (opaque is None or opaque.kind != "local" or ref is not None):
        raise AMLError("registry_invalid", f"Scope membership requires a local opaque id: {path}")
    name = re.sub(r"\[.*?\]", "", path.rsplit(".", 1)[-1])
    if (
        _IDENTITY.search(name)
        and ref is None
        and opaque is None
        and payload is None
        and not null_only
    ):
        raise AMLError("registry_invalid", f"Undeclared identity path: {path}")
    if derived is not None:
        return "derived_index", ref, opaque, derived, member
    if ref is not None:
        if integrity is not None:
            raise AMLError("registry_invalid", f"Integrity binding duplicates a reference: {path}")
        if ref.ref_mode == "live" and ref.existence_expectation is not None:
            raise AMLError("registry_invalid", f"Live reference has future expectation: {path}")
        if ref.existence_expectation is not None and ref.ref_mode != "historical_locator":
            raise AMLError(
                "registry_invalid", f"Only a historical locator has an expectation: {path}"
            )
        return "authoritative_ref", ref, opaque, derived, member
    return (
        "integrity_binding"
        if integrity is not None or opaque is not None and opaque.kind == "digest"
        else "own_attribute",
        ref,
        opaque,
        derived,
        member,
    )


def compile_inventory(variants: Iterable[SchemaVariant]) -> Registry:
    """Validate every implemented strict branch and classify every nested leaf.

    Args:
        variants: Actual implemented variants, never unsupported placeholder models.

    Returns:
        A stable registry with one classification for each leaf and branch.

    Raises:
        AMLError: If identity metadata, branch uniqueness, or strictness is invalid.
    """
    leaves: list[SchemaLeaf] = []
    registered: set[tuple[str, str]] = set()

    def visit(
        annotation: Any,
        metadata: list[Any],
        entity: str,
        branch: str,
        path: str,
        required: bool,
        nullable: bool,
        ancestors: tuple[type[BaseModel], ...],
    ) -> None:
        annotation, extra = _unwrap(annotation)
        metadata = [*metadata, *extra]
        opaque_payload: OpaquePayloadSpec | None = _one(metadata, OpaquePayloadSpec)
        origin = get_origin(annotation)
        args = get_args(annotation)
        if opaque_payload is None and origin in (Union, types.UnionType):
            present = [arg for arg in args if arg is not type(None)]
            is_nullable = len(present) != len(args)
            for child in present:
                child_type, _ = _unwrap(child)
                suffix = (
                    f"/{path}={child_type.__name__}"
                    if len(present) > 1
                    and isinstance(child_type, type)
                    and issubclass(child_type, BaseModel)
                    else ""
                )
                visit(
                    child,
                    metadata,
                    entity,
                    branch + suffix,
                    path,
                    required,
                    nullable or is_nullable,
                    ancestors,
                )
            return
        if opaque_payload is None and origin in (list, tuple, set, frozenset, dict):
            if origin is dict:
                # Dynamic map keys are identities only when explicitly annotated.
                key_type, key_metadata = _unwrap(args[0])
                if any(isinstance(item, _METADATA_TYPES) for item in key_metadata):
                    visit(
                        key_type,
                        key_metadata,
                        entity,
                        branch,
                        path + "{key}",
                        required,
                        False,
                        ancestors,
                    )
                child = args[1]
                suffix = "{value}"
            else:
                child = args[0]
                suffix = "[]"
            visit(child, metadata, entity, branch, path + suffix, required, nullable, ancestors)
            return
        if (
            opaque_payload is None
            and isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
        ):
            if annotation in ancestors:
                raise AMLError("registry_invalid", f"Recursive application schema at {path}")
            if annotation.model_config.get("extra") != "forbid":
                raise AMLError("registry_invalid", f"Non-strict nested model at {path}")
            inherited = [item for item in metadata if isinstance(item, DerivedSpec)]
            for name, field in annotation.model_fields.items():
                visit(
                    field.annotation,
                    [*inherited, *field.metadata],
                    entity,
                    branch,
                    f"{path}.{name}" if path else name,
                    field.is_required(),
                    False,
                    (*ancestors, annotation),
                )
            if annotation.model_fields:
                return
        if annotation in (Any, object) and opaque_payload is None:
            raise AMLError("registry_invalid", f"Unrestricted application payload at {path}")
        category, ref, opaque, derived, member = _describe(
            metadata, path, null_only=annotation is type(None)
        )
        if ref is not None and ref.target_forget_action == "clear_ref" and not nullable:
            raise AMLError("registry_invalid", f"clear_ref requires a nullable leaf: {path}")
        if (
            ref is not None
            and ref.target_forget_action == "cascade_source_item"
            and not (
                path.rsplit(".", 1)[-1] in {"target_memory_id", "target_key"}
                and any(
                    "link_type" in parent.model_fields
                    and get_args(parent.model_fields["link_type"].annotation)
                    == ("corrects_history",)
                    for parent in ancestors
                )
            )
        ):
            raise AMLError("registry_invalid", f"Invalid source cascade declaration: {path}")
        leaves.append(
            SchemaLeaf(
                entity, branch, path, category, required, nullable, ref, opaque, derived, member
            )
        )

    for variant in variants:
        key = (variant.entity, variant.branch)
        if key in registered:
            raise AMLError("registry_invalid", "Duplicate implemented schema branch")
        registered.add(key)
        visit(variant.model, [], variant.entity, variant.branch, "", True, False, ())
    by_key: dict[tuple[str, str, str], SchemaLeaf] = {}
    for leaf in leaves:
        key = (leaf.entity, leaf.branch, leaf.path)
        prior = by_key.get(key)
        if prior is not None and prior != leaf:
            raise AMLError("registry_invalid", f"Ambiguous leaf classification: {leaf.path}")
        by_key[key] = leaf
    return Registry(tuple(by_key[key] for key in sorted(by_key)))


@dataclass
class _ValueDeclaration:
    """Retain one annotation's preparation only for the current extraction walk."""

    # Keep the original objects alive: identity keys cannot alias reclaimed values.
    original: tuple[Any, tuple[Any, ...]]
    annotation: Any
    metadata: list[Any]
    ignored: bool
    args: tuple[Any, ...]
    origin: Any
    specs: tuple[RefSpec | None, ScopeMembershipSpec | None] | None = None
    prune_plan: _ReferenceFreePlan | None = None
    prune_prepared: bool = False


_REFERENCE_FREE_LEAVES = frozenset({float, int, bool, bytes, type(None)})


@dataclass(frozen=True)
class _ReferenceFreePlan:
    """Schema-only shape whose exact runtime values cannot contain references."""

    kind: Literal["scalar", "model", "list", "tuple", "dict"]
    expected: type[Any] | None = None
    fields: tuple[tuple[str, _ReferenceFreePlan], ...] = ()
    children: tuple[_ReferenceFreePlan, ...] = ()


def _reference_free_plan(
    annotation: Any, metadata: list[Any],
    ancestors: frozenset[type[BaseModel]] = frozenset(),
) -> _ReferenceFreePlan | None:
    """Compile only branches with no lifecycle metadata or dynamic type arms.

    The plan lives for one extraction call. A later call sees changed field
    metadata, and the runtime shape check below protects model_copy/construct.
    """
    annotation, extra = _unwrap(annotation)
    if any(isinstance(item, _METADATA_TYPES) for item in (*metadata, *extra)):
        return None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        return None
    if isinstance(annotation, type) and issubclass(annotation, str | int | float | bool | bytes):
        return _ReferenceFreePlan("scalar", expected=annotation)
    if annotation is type(None):
        return _ReferenceFreePlan("scalar", expected=type(None))
    if origin in (list, tuple) and args:
        if origin is tuple and (len(args) != 2 or args[1] is not Ellipsis):
            return None
        child = _reference_free_plan(args[0], [], ancestors)
        if child is not None:
            return _ReferenceFreePlan(
                "list" if origin is list else "tuple", children=(child,),
            )
    if origin is dict and len(args) == 2:
        key = _reference_free_plan(args[0], [], ancestors)
        value = _reference_free_plan(args[1], [], ancestors)
        if key is not None and value is not None:
            return _ReferenceFreePlan("dict", children=(key, value))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in ancestors:
            return None
        fields: list[tuple[str, _ReferenceFreePlan]] = []
        for name, field in annotation.model_fields.items():
            child = _reference_free_plan(
                field.annotation, list(field.metadata), ancestors | {annotation},
            )
            if child is None:
                return None
            fields.append((name, child))
        return _ReferenceFreePlan("model", expected=annotation, fields=tuple(fields))
    return None


def _matches_reference_free(value: object, plan: _ReferenceFreePlan) -> bool:
    """Refuse a shortcut whenever actual values depart from the strict shape."""
    if value is None:
        return True  # The ordinary walk also ignores null before metadata.
    if plan.kind == "scalar":
        return type(value) is plan.expected
    if plan.kind == "model":
        if type(value) is not plan.expected:
            return False
        return all(
            _matches_reference_free(getattr(value, name), child)
            for name, child in plan.fields
        )
    if plan.kind in {"list", "tuple"}:
        if type(value) is not (list if plan.kind == "list" else tuple):
            return False
        items = cast(list[object] | tuple[object, ...], value)
        child, = plan.children
        if child.kind == "scalar":
            return all(item is None or type(item) is child.expected for item in items)
        return all(_matches_reference_free(item, child) for item in items)
    if type(value) is not dict:
        return False
    items_by_key = cast(dict[object, object], value)
    key_plan, value_plan = plan.children
    return all(
        _matches_reference_free(key, key_plan)
        and _matches_reference_free(item, value_plan)
        for key, item in items_by_key.items()
    )


def _union_arm(value: object, args: tuple[Any, ...]) -> Any | None:
    """Select the first non-null union arm whose container or type holds ``value``."""
    for candidate in args:
        if candidate is type(None):
            continue
        child, _ = _unwrap(candidate)
        if (
            get_origin(child) in (list, tuple, dict)
            and isinstance(value, get_origin(child))
            or isinstance(child, type)
            and isinstance(value, child)
        ):
            return candidate
    return None


def _walk_values(
    value: Any,
    annotation: Any,
    metadata: list[Any],
    path: str,
    references: list[Reference],
    memberships: list[ScopeMembership],
    declarations: dict[tuple[int, tuple[int, ...]], _ValueDeclaration],
) -> None:
    if value is None:
        return
    key = (id(annotation), tuple(id(item) for item in metadata))
    declaration = declarations.get(key)
    if declaration is None:
        original = (annotation, tuple(metadata))
        annotation, extra = _unwrap(annotation)
        metadata = [*metadata, *extra]
        ignored = (_one(metadata, DerivedSpec) is not None
                   or _one(metadata, OpaquePayloadSpec) is not None)
        declaration = _ValueDeclaration(
            original, annotation, metadata, ignored, get_args(annotation), get_origin(annotation),
        )
        declarations[key] = declaration
    if declaration.ignored:
        return
    annotation, metadata = declaration.annotation, declaration.metadata
    args, origin = declaration.args, declaration.origin
    if origin in (Union, types.UnionType):
        candidate = _union_arm(value, args)
        if candidate is not None:
            _walk_values(value, candidate, metadata, path, references, memberships, declarations)
            return
    if not declaration.prune_prepared:
        declaration.prune_plan = _reference_free_plan(annotation, metadata)
        declaration.prune_prepared = True
    if (declaration.prune_plan is not None
            and _matches_reference_free(value, declaration.prune_plan)):
        return
    # Union-arm, derived and opaque rules must resolve first. Eagerly validating
    # these specs would reject metadata on branches the original walk ignores.
    if declaration.specs is None:
        ref: RefSpec | None = _one(metadata, RefSpec)
        member: ScopeMembershipSpec | None = _one(metadata, ScopeMembershipSpec)
        declaration.specs = (ref, member)
    ref, member = declaration.specs
    if isinstance(value, str):
        if ref is not None:
            references.append(Reference(path, value, ref))
        if member is not None:
            memberships.append(ScopeMembership(path, value, member))
        return
    if isinstance(value, BaseModel):
        for name, field in type(value).model_fields.items():
            _walk_values(
                getattr(value, name),
                field.annotation,
                list(field.metadata),
                f"{path}.{name}" if path else name,
                references,
                memberships,
                declarations,
            )
        return
    if isinstance(value, list | tuple):
        child = args[0] if args else Any
        walked_leaves: set[type[Any]] = set()
        for index, item in enumerate(cast(list[Any] | tuple[Any, ...], value)):
            leaf: type[Any] = type(cast(object, item))
            if leaf in _REFERENCE_FREE_LEAVES:
                # A number, byte string or None carries no identifier. The first
                # value of each leaf type still walks so every declared arm's
                # metadata is classified and rejected exactly as before; later
                # values of that type are repeats. Non-leaf values still walk.
                if leaf in walked_leaves:
                    continue
                walked_leaves.add(leaf)
            _walk_values(
                item, child, metadata, f"{path}[{index}]", references, memberships, declarations,
            )
    elif isinstance(value, dict):
        key_type, child = args if len(args) == 2 else (Any, Any)
        for key, item in cast(dict[Any, Any], value).items():
            _walk_values(key, key_type, [], f"{path}{{key}}", references, memberships, declarations)
            _walk_values(
                item, child, metadata, f"{path}{{{key}}}", references, memberships, declarations,
            )


def extract_references(model: StrictModel) -> tuple[Reference, ...]:
    """Extract authoritative references without reading a handwritten path list."""
    references: list[Reference] = []
    _walk_values(model, type(model), [], "", references, [], {})
    return tuple(references)


def extract_scope_memberships(model: StrictModel) -> tuple[ScopeMembership, ...]:
    """Extract only explicitly declared authorization scope paths."""
    memberships: list[ScopeMembership] = []
    _walk_values(model, type(model), [], "", [], memberships, {})
    return tuple(memberships)


def extract_text_leaves(value: object) -> Iterator[TextLeaf]:
    """Yield every string and map key a content scan inspects, in declaration order.

    A model leaf declared as an AML reference (``RefSpec``), a local identity or
    a keyed digest (``OpaqueIdSpec`` ``local`` or ``digest``) holds only a
    product-generated identifier and is skipped, as are its container elements.
    Free text, external identifiers, undeclared strings, external payloads and
    every untyped mapping or sequence, keys included, are yielded. Union arms
    resolve as reference extraction resolves them.

    Args:
        value: A strict model, or a decoded JSON-compatible value walked untyped.

    Yields:
        TextLeaf: Each inspected string. Its path uses the inventory leaf form
        (``.field``, ``[]``, ``{key}``, ``{value}``) and never embeds a map key
        or any other value.
    """
    pending: list[tuple[object, Any, list[Any], str]] = [(value, type(value), [], "")]
    while pending:
        current, annotation, metadata, path = pending.pop()
        if current is None:
            continue
        while isinstance(annotation, TypeAliasType) or get_origin(annotation) is Annotated:
            if isinstance(annotation, TypeAliasType):
                annotation = annotation.__value__
            else:
                annotation, *extra = get_args(annotation)
                metadata = [*metadata, *extra]
        if _one(metadata, OpaquePayloadSpec) is not None:
            annotation, metadata = Any, []
        origin, args = get_origin(annotation), get_args(annotation)
        if origin in (Union, types.UnionType):
            arm = _union_arm(current, args)
            if arm is not None:
                pending.append((current, arm, metadata, path))
                continue
        if isinstance(current, str):
            opaque: OpaqueIdSpec | None = _one(metadata, OpaqueIdSpec)
            if _one(metadata, RefSpec) is None and (opaque is None or opaque.kind == "external"):
                yield TextLeaf(path, current)
        elif isinstance(current, BaseModel):
            extra_fields = current.__pydantic_extra__
            if extra_fields:
                pending.append((extra_fields, Any, [], f"{path}{{extra}}"))
            pending.extend(
                (getattr(current, name), field.annotation, list(field.metadata),
                 f"{path}.{name}" if path else name)
                for name, field in reversed(type(current).model_fields.items())
            )
        elif isinstance(current, Mapping):
            key_type, child = args if origin is dict and len(args) == 2 else (Any, Any)
            for key, item in reversed(list(cast(Mapping[object, object], current).items())):
                pending.append((item, child, metadata, f"{path}{{value}}"))
                pending.append((key, key_type, [], f"{path}{{key}}"))
        elif isinstance(current, list | tuple | set | frozenset):
            items = list(cast(Iterable[object], current))
            if origin is tuple and args and args[-1] is not Ellipsis and len(args) == len(items):
                children = args
            else:
                element = args[0] if origin in (list, tuple, set, frozenset) and args else Any
                children = (element,) * len(items)
            pending.extend(
                (item, child, metadata, f"{path}[]")
                for item, child in reversed(list(zip(items, children, strict=True)))
            )
