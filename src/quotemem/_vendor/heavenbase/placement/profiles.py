"""Registry-backed logical-type placement profiles."""

from __future__ import annotations

__all__ = ["registered_storage_profiles", "storage_profile_for"]

from collections.abc import Mapping
import copy
import re
from types import MappingProxyType

from heavenbase.utils import Any, Dict

from ..modules import ModuleResolver, ModuleService, module_resolver
from ..modules.base import ResolverCache
from ..types.descriptor import logical_type_from_any
from .profile import StorageProfile

_PROFILE_CACHE = ResolverCache[StorageProfile]()
_REGISTERED_KEY = "storage-profiles"


def _registration_identifier(dtype: str) -> str:
    """Return the stable Registry identifier for one profile lookup key."""
    if dtype == "*":
        return "default"
    identifier = re.sub(r"[^a-z0-9]+", "-", dtype.lower()).strip("-")
    if not identifier:
        raise ValueError(f"Storage profile dtype {dtype!r} has no Registry-safe identifier")
    return identifier


def _dependencies(profile: StorageProfile, resolver: ModuleResolver) -> list[str]:
    """Return hard dependencies owned by one storage profile definition."""
    dependencies: list[str] = []
    if profile.dtype != "*":
        dtype = logical_type_from_any(profile.dtype, resolver=resolver)
        dependencies.append(f"logical_type:{type(dtype).identifier}")
        item_type = getattr(dtype, "item_type", None)
        if item_type is not None:
            dependencies.append(f"logical_type:{type(item_type).identifier}")
    strategies = (
        profile.default_strategy,
        *(rule.strategy for rule in profile.strategy_rules),
    )
    dependencies.extend(f"strategy:{strategy}" for strategy in strategies)
    return list(dict.fromkeys(dependencies))


def _record(profile: StorageProfile, resolver: ModuleResolver) -> dict[str, Any]:
    """Return the plain registration record for one normalized profile."""
    return {
        "name": f"{profile.dtype} storage profile",
        "desc": profile.reason,
        "source": "inline",
        "target": "profile",
        "active": True,
        "dtype": profile.dtype,
        "strategy": profile.default_strategy,
        "meta": {
            "schema_version": 1,
            "dependencies": _dependencies(profile, resolver),
            "profile": {
                "backend_preferences": [predicate.to_dict() for predicate in profile.backend_preferences],
                "default_strategy": profile.default_strategy,
                "reason": profile.reason,
                "strategy_rules": [rule.to_dict() for rule in profile.strategy_rules],
            },
        },
    }


def _profile_from_row(row: Mapping[str, Any], resolver: ModuleResolver, *, fresh: bool = False) -> StorageProfile:
    """Materialize one validated Registry storage profile.

    The result is the resolver's cached object, which a caller of `register`
    may hold and change, unless `fresh` asks for one built from the record.
    """

    def build(current_row: Mapping[str, Any]) -> StorageProfile:
        record = current_row["record"]
        dtype = record.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise TypeError(f"Storage profile {current_row['identifier']!r} must declare record dtype")
        definition = resolver.resolve("storage_profile", str(current_row["identifier"]))
        if not isinstance(definition, Mapping):
            raise TypeError(f"Storage profile {current_row['identifier']!r} inline definition must be a mapping")
        profile = StorageProfile.from_mapping(dtype, definition, resolver=resolver)
        if record.get("strategy") != profile.default_strategy:
            raise ValueError(f"Storage profile {current_row['identifier']!r} record strategy disagrees with its default strategy")
        if _registration_identifier(profile.dtype) != current_row["identifier"]:
            raise ValueError(f"Storage profile record {current_row['identifier']!r} does not match dtype {profile.dtype!r}")
        return profile

    return _PROFILE_CACHE.get(resolver, row, build, should_cache=lambda current_row: not fresh)


def _module_service(resolver: ModuleResolver) -> ModuleService:
    """Require lifecycle authority for one storage-profile mutation."""
    if not isinstance(resolver, ModuleService):
        raise TypeError("Storage profile registration requires a mutable ModuleService resolver")
    return resolver


def _register_profile(
    profile: StorageProfile,
    *,
    replace: bool = False,
    resolver: ModuleResolver,
) -> StorageProfile:
    """Register one durable storage profile through the central module service.

    Args:
        profile: Normalized profile to publish.
        replace: Whether to reconcile a prior profile owned by the same author.
        resolver: Explicit Context-owned module resolver. Registration requires
            a mutable `ModuleService` implementation.

    Returns:
        StorageProfile: Registry-materialized profile.

    Raises:
        TypeError: If `profile` is not a `StorageProfile`.
        ValueError: If the profile exists, belongs to another owner, or has a
            Registry-identifier collision.
    """
    if not isinstance(profile, StorageProfile):
        raise TypeError("StorageProfile.register expects a StorageProfile")
    identifier = _registration_identifier(profile.identifier)
    coordinate = f"programmatic/storage_profile/{identifier}"
    service = _module_service(resolver)
    found = service.inspect("storage_profile", identifier)
    if found:
        existing = _profile_from_row(found[0], service)
        if existing.identifier != profile.identifier:
            raise ValueError(f"Storage profile Registry identifier {identifier!r} already represents {existing.identifier!r}")
        if not replace:
            raise ValueError(f"Storage profile for logical type {profile.identifier!r} is already registered")
        existing_coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
        if existing_coordinate != coordinate:
            raise ValueError(f"Storage profile {profile.identifier!r} belongs to {existing_coordinate!r} and cannot be replaced")
    service.register(
        "storage_profile",
        identifier,
        _record(profile, service),
        coordinate=coordinate,
    )
    return _profile_from_row(service.inspect("storage_profile", identifier)[0], service)


def storage_profile_for(dtype: Any, *, resolver: ModuleResolver | None = None) -> StorageProfile:
    """Return the most specific registered profile for one logical type.

    Args:
        dtype: Logical type class, instance, identifier, or descriptor.
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        StorageProfile: An isolated copy of the specific registered profile
        or the registered default; changing it changes no other caller's view.

    Raises:
        RuntimeError: If the Registry contains no default storage profile.
    """
    return copy.deepcopy(_storage_profile_for(dtype, resolver=resolver))


def _storage_profile_for(dtype: Any, *, resolver: ModuleResolver | None = None) -> StorageProfile:
    """Return the shared profile `storage_profile_for` copies; callers must not change it."""
    selected = module_resolver(resolver)
    profiles = _registered_profiles(selected)
    for key in StorageProfile.keys_for(dtype, resolver=selected):
        if key in profiles:
            return profiles[key]
    try:
        return profiles["*"]
    except KeyError:
        raise RuntimeError("system-modules has no default storage profile") from None


def registered_storage_profiles(*, resolver: ModuleResolver | None = None) -> Dict[str, StorageProfile]:
    """Return registered profiles keyed by logical-type lookup key.

    Args:
        resolver: Context-owned module resolver. The default Context is
            selected when omitted.

    Returns:
        Dict[str, StorageProfile]: Isolated Registry-derived profile mapping:
        each profile is a copy, so changing it changes no other caller's view.
        For a ``ModuleService`` the catalog is read again only after it
        publishes; any other resolver is read at every call.
    """
    return {key: copy.deepcopy(profile) for key, profile in _registered_profiles(module_resolver(resolver)).items()}


def _registered_profiles(selected: ModuleResolver) -> Mapping[str, StorageProfile]:
    """Return one resolver's registered profiles; callers must not change the mapping or its profiles.

    A `ModuleService` keeps its mapping until its catalog publishes. Any other
    resolver can give the same catalog's records another reading, so its
    profiles are read afresh and never kept for, or taken from, the service.
    """
    derived_of = getattr(selected.catalog, "_derived", None) if isinstance(selected, ModuleService) else None
    if derived_of is None:
        return _read_profiles(selected, kept=False)
    publication, derived = derived_of()
    remembered = derived.get(_REGISTERED_KEY)
    if remembered is not None and remembered[0] is selected:
        return remembered[1]
    profiles = MappingProxyType(_read_profiles(selected, kept=True))
    if selected.catalog.publication == publication:
        derived[_REGISTERED_KEY] = (selected, profiles)
    return profiles


def _read_profiles(selected: ModuleResolver, *, kept: bool) -> dict[str, StorageProfile]:
    """Read every registered profile, as objects of their own when the mapping is to be kept."""
    profiles: dict[str, StorageProfile] = {}
    for row in selected.inspect("storage_profile"):
        profile = _profile_from_row(row, selected, fresh=kept)
        if profile.identifier in profiles:
            raise ValueError(f"Multiple storage profile records own logical type {profile.identifier!r}")
        profiles[profile.identifier] = profile
    return profiles
