"""Placement rule and profile resolution."""

from __future__ import annotations

__all__ = ["StorageResolver"]

from heavenbase.utils import Any, Dict, List, Mapping, Optional

from ..modules import ModuleResolver, module_resolver
from ..schema.model import EntitySchema
from ..strategies import InlineColumn, StorageStrategy
from ..types import LogicalType
from .binding import StorageBinding, StoragePlan
from .profile import StorageProfile
from .profiles import _storage_profile_for
from .rule import StorageRule


class StorageResolver:
    """Resolve authored rules and default profiles into concrete routes."""

    def __init__(
        self,
        default_backend: str = "main",
        *,
        resolver: ModuleResolver | None = None,
    ) -> None:
        """Create a resolver with one preferred fallback backend.

        Args:
            default_backend (str): Preferred workspace backend name.
            resolver (ModuleResolver | None): Context-owned module resolver.
                The default Context is selected when omitted.

        Returns:
            None: This initializer stores the fallback backend name.
        """
        self.default_backend = default_backend
        self.module_resolver = module_resolver(resolver)

    def resolve(
        self,
        schema: EntitySchema,
        declared_backends: Any,
        overrides: Dict[str, List[StorageRule]],
        *,
        storage_profiles: Optional[Mapping[str, StorageProfile]] = None,
    ) -> StoragePlan:
        """Resolve one entity schema into concrete storage routes.

        Args:
            schema (EntitySchema): Entity schema being placed.
            declared_backends (Any): Workspace backend registry or backend names.
            overrides (Dict[str, List[StorageRule]]): Field-authored rules.
            storage_profiles (Mapping[str, StorageProfile] | None): Effective
                workspace profile overrides.

        Returns:
            StoragePlan: Resolved routes for every field and required id copy.

        Raises:
            ValueError: If primary-key storage is customized or no backends exist.
            RuntimeError: If a required target backend is absent or no required
                rule can produce a route.
        """
        names = declared_backends.names() if hasattr(declared_backends, "names") else list(declared_backends)
        if not names:
            raise ValueError("Storage resolution requires at least one backend")
        if overrides.get(schema.pk_name):
            raise ValueError("object_id field storage is fixed and cannot be customized")
        out: List[StorageBinding] = []
        for field_schema in schema.fields.values():
            if field_schema.name == schema.pk_name:
                continue
            rules = overrides.get(field_schema.name) or [self.default_rule(field_schema.dtype, declared_backends, storage_profiles=storage_profiles)]
            fired = 0
            for rule in rules:
                backend = rule.backend or self.default_backend
                if backend not in names:
                    if rule.optional:
                        continue
                    raise RuntimeError(f"Storage for {schema.entity_id}.{field_schema.name} targets unknown backend {backend!r}")
                strategy = StorageStrategy.load(rule.strategy, resolver=self.module_resolver)
                out.append(
                    StorageBinding(
                        schema.entity_id,
                        field_schema.name,
                        backend,
                        strategy,
                        rule.reason,
                        strategy.normalize_options(rule.strategy_options),
                        rule.params,
                        resolver=self.module_resolver,
                    )
                )
                fired += 1
            if fired == 0 and not all(rule.optional for rule in rules):
                raise RuntimeError(f"No effective storage binding for {schema.entity_id}.{field_schema.name}")
        used_backends = {binding.backend for binding in out}
        id_backends = [name for name in names if name in used_backends] or [self.default_backend if self.default_backend in names else names[0]]
        for backend in id_backends:
            out.append(
                StorageBinding(
                    schema.entity_id,
                    schema.pk_name,
                    backend,
                    InlineColumn.identifier,
                    "builtin-id",
                    resolver=self.module_resolver,
                )
            )
        return StoragePlan(schema.entity_id, out)

    def default_rule(
        self,
        dtype: LogicalType,
        declared_backends: Any = None,
        *,
        storage_profiles: Optional[Mapping[str, StorageProfile]] = None,
    ) -> StorageRule:
        """Choose the default authored-equivalent rule for one logical type.

        Args:
            dtype (LogicalType): Logical field type.
            declared_backends (Any): Workspace backend registry or backend names.
            storage_profiles (Mapping[str, StorageProfile] | None): Effective
                workspace profile overrides.

        Returns:
            StorageRule: Rule selected from the profile's ordered tag policy.
        """
        profile = None
        if storage_profiles:
            for key in StorageProfile.keys_for(dtype, resolver=self.module_resolver):
                profile = storage_profiles.get(key)
                if profile is not None:
                    break
        if profile is None:
            profile = _storage_profile_for(dtype, resolver=self.module_resolver)

        backend, claims = self._profile_backend(profile, declared_backends)

        return StorageRule(
            backend=backend,
            strategy=profile.strategy_for(claims) if backend is not None else profile.default_strategy,
            reason=profile.reason,
            resolver=self.module_resolver,
        )

    def _profile_backend(
        self,
        profile: StorageProfile,
        declared_backends: Any,
    ) -> tuple[str | None, Mapping[str, Any]]:
        """Apply ordered predicates to one coherent static Backend snapshot."""
        items_method = getattr(declared_backends, "items", None)
        get_method = getattr(declared_backends, "get", None)
        if not callable(items_method) or not callable(get_method):
            return None, {}
        instances = tuple(items_method())
        for _attempt in range(3):
            rows = self.module_resolver.inspect("backend_type")
            publication = self.module_resolver.catalog.publication
            if rows and any(row["publication"] != publication for row in rows):
                continue
            aliases: dict[str, str] = {}
            claims: dict[str, Mapping[str, Any]] = {}
            ordered_types: list[str] = []
            for row in rows:
                identifier = str(row["identifier"])
                ordered_types.append(identifier)
                aliases[identifier] = identifier
                aliases.update({str(alias): identifier for alias in row.get("aliases", ())})
                claims[identifier] = self._raw_tags(row)
            typed_instances: list[tuple[str, str]] = []
            for name, backend in instances:
                identifier = aliases.get(str(getattr(backend, "type", "")))
                if identifier is not None:
                    typed_instances.append((str(name), identifier))
            preferred = next(((name, identifier) for name, identifier in typed_instances if name == self.default_backend), None)
            selected_result: tuple[str | None, Mapping[str, Any]] = (None, {})
            for predicate in profile.backend_preferences:
                if preferred is not None and predicate.matches(claims.get(preferred[1], {})):
                    selected_result = preferred[0], claims[preferred[1]]
                    break
                selected = None
                for identifier in ordered_types:
                    if not predicate.matches(claims.get(identifier, {})):
                        continue
                    selected = next((name for name, backend_identifier in typed_instances if backend_identifier == identifier), None)
                    if selected is not None:
                        selected_result = selected, claims[identifier]
                        break
                if selected is not None:
                    break
            if self.module_resolver.catalog.publication == publication:
                return selected_result
        raise RuntimeError("Backend tag snapshot changed during placement; reopen the workspace and retry")

    @staticmethod
    def _raw_tags(row: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return raw tag claims from one Backend type record."""
        record = row.get("record")
        meta = record.get("meta") if isinstance(record, Mapping) else None
        tags = meta.get("tags") if isinstance(meta, Mapping) else None
        return dict(tags) if isinstance(tags, Mapping) else {}
