"""Shared context for workspace MCP tool callables."""

from __future__ import annotations

__all__ = ["WorkspaceToolContext"]

from heavenbase.entity import Entity
from heavenbase.utils import Any, List, Mapping, Optional, Sequence, dataclass
from heavenbase.utils.values import kebab

from ...profile import ProfileSpec
from ...profiles import entity_allowed


@dataclass(frozen=True)
class WorkspaceToolContext:
    """Workspace/profile state shared by one workspace MCP toolkit."""

    workspace: Any
    profile: ProfileSpec
    tool_names: tuple[str, ...]

    @property
    def entity_scope(self) -> Any:
        """Return the active profile entity scope."""
        return self.profile.entities

    @property
    def skill_scope(self) -> Any:
        """Return the active profile skill scope."""
        return self.profile.skills

    def workspace_guide(self) -> str:
        """Render the guide supplied by this Toolkit artifact generation.

        Args:
            None.

        Returns:
            str: Generated workspace-guide skill text for MCP clients.
        """
        from .guide import render_workspace_guide

        return render_workspace_guide(self)

    @property
    def allowed_entities(self) -> Optional[frozenset[str]]:
        """Return the closed write scope, or ``None`` for an unrestricted profile."""
        return None if self.entity_scope == "*" else frozenset(self.entity_scope)

    def require_entity(self, entity: Optional[str]) -> Optional[str]:
        """Resolve and validate one entity id against the profile scope.

        Args:
            entity (Optional[str]): Entity id, class name, or display name to resolve.

        Returns:
            Optional[str]: Resolved entity id, or ``None`` when ``entity`` is omitted under an unrestricted scope.

        Raises:
            ValueError: If the resolved entity is outside the profile entity scope.
        """
        if self.entity_scope == "*":
            return self.resolve_entity_id(entity)
        if entity is None:
            return None
        allowed = frozenset(self.entity_scope)
        resolved = str(entity) if str(entity) in allowed else self.resolve_entity_id(entity, entity_ids=self.scoped_entity_ids())
        if not entity_allowed(resolved, allowed):
            raise ValueError(f"Entity is outside MCP profile {self.profile.name!r} entity scope")
        return resolved

    def filter_entities(self, entity_ids: List[str]) -> List[str]:
        """Filter entity ids through the active profile scope.

        Args:
            entity_ids (List[str]): Candidate entity identifiers.

        Returns:
            List[str]: ``entity_ids`` unchanged when the scope is ``"*"``, otherwise the subset allowed by the profile.
        """
        if self.entity_scope == "*":
            return entity_ids
        allowed_ids = frozenset(self.entity_scope)
        return [entity_id for entity_id in entity_ids if entity_id in allowed_ids]

    def scoped_entity_ids(self) -> List[str]:
        """Return registered entity ids visible to this profile.

        Args:
            None.

        Returns:
            List[str]: All registered entity ids when the scope is ``"*"``, otherwise profile-scoped ids that exist in the workspace.
        """
        if self.entity_scope == "*":
            return sorted(self.workspace.entities)
        return [entity_id for entity_id in self.entity_scope if entity_id in self.workspace.entities]

    def scoped_existing_entity(self, object_id: Any, *, for_mutation: bool = False) -> Optional[str]:
        """Resolve an object id to exactly one visible entity when scoped.

        A visible entity whose ``object_id`` domain rejects the id cannot hold
        it and is skipped.

        Args:
            object_id (Any): Row object id to locate within the profile scope.
            for_mutation (bool): Require reliable negative reads before selecting a mutation target.

        Returns:
            Optional[str]: Matching entity id, or ``None`` when no scoped entity contains the row.

        Raises:
            ValueError: If the object id exists in more than one visible entity,
                no visible entity's domain accepts it, or, for a mutation, a
                visible entity that returned no row cannot show that the row is absent.
            TypeError: If no visible entity accepts the id and the widest domain
                rejected it by type.
        """
        if self.entity_scope == "*":
            return None
        ws = self.workspace
        lookup = ws._row_lookup
        candidates = {
            entity_cls.schema().entity_id: normalized_ids[0]
            for entity_cls, normalized_ids in lookup.normalized_ids_by_entity([object_id], entity_ids=self.scoped_entity_ids())
            if normalized_ids[0] is not None
        }
        matches = [entity_id for entity_id, normalized_id in candidates.items() if ws.exists(normalized_id, entity=entity_id)]
        if len(matches) > 1:
            names = ", ".join(matches[:8])
            suffix = "" if len(matches) <= 8 else f", +{len(matches) - 8} more"
            raise ValueError(f"object_id {object_id!r} exists in multiple MCP profile entities: {names}{suffix}; pass entity=... to disambiguate")
        if for_mutation:
            lookup.validate_entity_inference((entity_id, normalized_id) for entity_id, normalized_id in candidates.items() if entity_id not in matches)
        return matches[0] if matches else None

    def entity_for_read(self, object_id: Any, entity: Optional[str]) -> Optional[str]:
        """Resolve the entity argument for read/existence tools.

        Args:
            object_id (Any): Row object id being read or checked.
            entity (Optional[str]): Optional explicit entity id, class name, or display name.

        Returns:
            Optional[str]: Resolved entity id for the workspace call, or ``None`` when disambiguation is unnecessary.
        """
        resolved = self.require_entity(entity)
        if resolved is not None or self.entity_scope == "*":
            return resolved
        return self.scoped_existing_entity(object_id)

    def entity_for_patch(self, row: Mapping[str, Any], entity: Optional[str]) -> Optional[str]:
        """Resolve the entity argument for single-row patch tools.

        Args:
            row (Mapping[str, Any]): Patch payload that may include ``object_id``.
            entity (Optional[str]): Optional explicit entity id, class name, or display name.

        Returns:
            Optional[str]: Resolved entity id for the workspace patch call.

        Raises:
            ValueError: If a scoped profile cannot infer the entity from ``object_id``.
        """
        resolved = self.require_entity(entity)
        if resolved is not None:
            return resolved
        raw_id = row.get("object_id")
        if self.entity_scope == "*":
            return resolved
        if raw_id in (None, ""):
            raise ValueError("set needs object_id in the object or patch")
        scoped = self.scoped_existing_entity(raw_id, for_mutation=True)
        if scoped is None:
            raise ValueError(
                f"set with omitted entity could not resolve object_id {raw_id!r} inside MCP profile {self.profile.name!r} entity scope; pass entity=..."
            )
        return scoped

    def entity_for_many_patch(self, row: Any, entity: Optional[str]) -> Optional[str]:
        """Resolve the entity argument for bulk patch tools.

        Args:
            row (Any): One row mapping from a bulk patch request.
            entity (Optional[str]): Optional explicit entity id shared across the batch.

        Returns:
            Optional[str]: Resolved entity id for this row.

        Raises:
            TypeError: If a scoped profile receives a non-mapping row without an explicit entity.
            ValueError: If the entity cannot be inferred for a scoped profile.
        """
        resolved = self.require_entity(entity)
        if resolved is not None or self.entity_scope == "*":
            return resolved
        if isinstance(row, Mapping) and row.get("entity") is not None:
            return self.require_entity(str(row["entity"]))
        if not isinstance(row, Mapping):
            raise TypeError("set_many under a scoped MCP profile expects row mappings or an explicit entity")
        return self.entity_for_patch(row, None)

    @staticmethod
    def strip_row_entity(row: Any) -> Any:
        """Drop an inline entity discriminator before writing a row.

        Args:
            row (Any): Row payload that may include an ``entity`` helper field.

        Returns:
            Any: ``row`` unchanged when it is not a mapping with ``entity``, otherwise a shallow copy without ``entity``.
        """
        if not isinstance(row, Mapping) or "entity" not in row:
            return row
        out = dict(row)
        out.pop("entity", None)
        return out

    def scoped_delete_target(self, target: Any) -> Optional[Any]:
        """Resolve a delete target through the active profile scope.

        Args:
            target (Any): Entity instance, mapping, ``(entity, object_id)`` pair, or bare object id.

        Returns:
            Optional[Any]: Workspace delete target, or ``None`` when the target is outside the scoped profile.

        Raises:
            ValueError: If a mapping or entity target lacks ``object_id``.
        """
        if self.entity_scope == "*":
            return target
        if isinstance(target, Entity):
            data = target.to_dict()
            raw_id = data.get("object_id")
            if raw_id in (None, ""):
                raise ValueError("delete target needs object_id")
            return (self.require_entity(type(target).schema().entity_id), raw_id)
        if isinstance(target, Mapping):
            raw_id = target.get("object_id")
            if raw_id in (None, ""):
                raise ValueError("delete target needs object_id")
            if target.get("entity") is not None:
                return {"entity": self.require_entity(str(target["entity"])), "object_id": raw_id}
            entity = self.scoped_existing_entity(raw_id, for_mutation=True)
            return {"entity": entity, "object_id": raw_id} if entity is not None else None
        if isinstance(target, (tuple, list)) and len(target) == 2:
            return (self.require_entity(str(target[0])), target[1])
        entity = self.scoped_existing_entity(target, for_mutation=True)
        return {"entity": entity, "object_id": target} if entity is not None else None

    def delete_targets(self, targets: Sequence[Any], *, edges: str) -> List[bool]:
        """Delete scoped targets while enforcing the profile's full mutation boundary.

        Args:
            targets (Sequence[Any]): Delete targets accepted by ``scoped_delete_target``.
            edges (str): Graph-edge delete mode forwarded to the workspace CRUD layer.

        Returns:
            List[bool]: Deletion flags aligned with ``targets``; ``False`` when a target was out of scope.
        """
        ws = self.workspace
        with ws._lock:
            mode = ws._graph_mutation.normalize_delete_mode(edges)
            scoped_targets = [self.scoped_delete_target(target) for target in targets]
            active = [target for target in scoped_targets if target is not None]
            allowed_entities = None if self.entity_scope == "*" else frozenset(self.entity_scope)
            active_deleted = iter(ws._crud.delete_many(active, edges=mode, allowed_entities=allowed_entities))
            return [False if target is None else next(active_deleted) for target in scoped_targets]

    def entity_dict(self, entity_cls: type) -> dict[str, Any]:
        """Return the MCP description payload for one registered entity class.

        Args:
            entity_cls (type): Registered entity class to describe.

        Returns:
            dict[str, Any]: Schema, storage plan, query capabilities, and per-field filter operations.
        """
        from heavenbase.filters.ops import registered_ops
        from heavenbase.workspace.query.filter import option_refusal

        schema = entity_cls.schema()
        options = self.workspace.entity_options.get(schema.entity_id)
        resolver = self.workspace.module_resolver
        storage_plan = self.workspace.storage_plans[schema.entity_id]
        near_ops = {
            str(row["identifier"]) for row in resolver.inspect("operation") if ((row["record"].get("meta") or {}).get("tags") or {}).get("near") is True
        }
        capabilities = {
            (row["dtype"], row["op"], row["backend_identifier"], row["strategy"])
            for row in self.workspace.handlers.capability_rows()
            if row["op"] not in near_ops
        }
        field_filter_ops = {}
        for name, field_schema in schema.fields.items():
            dtype = type(field_schema.dtype).identifier
            routes = {(self.workspace.backends.get(binding.backend).type, binding.strategy) for binding in storage_plan.field_bindings(name)}
            # An operation a route handles is still refused when the entity's registration options rule it out
            # (``gram=False`` and ``contained_in`` on ``Array[ShortText]``), by the query compiler's own rule.
            field_filter_ops[name] = [
                op
                for op in registered_ops(resolver=resolver)
                if any((dtype, op, backend_identifier, strategy) in capabilities for backend_identifier, strategy in routes)
                and option_refusal(schema.entity_id, options, name, op, field_schema) is None
            ]
        return {
            "schema": schema.to_dict(),
            "derives": self.workspace._schema_metadata.entity_derives(entity_cls),
            "storage_plan": storage_plan.to_dict(),
            "query_capabilities": self.workspace._schema_metadata.query_capabilities(entity_cls),
            "filter_ops": field_filter_ops,
        }

    def registered_entity(self, entity_id: str) -> type:
        """Return one registered entity class by id.

        Args:
            entity_id (str): Registered entity identifier.

        Returns:
            type: Entity class registered on the workspace.

        Raises:
            KeyError: If ``entity_id`` is not registered on the workspace.
        """
        if entity_id not in self.workspace.entities:
            raise KeyError(f"Workspace {self.workspace.id!r} has no registered entity {entity_id!r}")
        return self.workspace.entities[entity_id]

    def resolve_entity_id(self, entity: Optional[str], *, entity_ids: Optional[Sequence[str]] = None) -> Optional[str]:
        """Resolve an entity id, class name, or display name.

        Args:
            entity (Optional[str]): Entity id, class name, display name, or ``None``.
            entity_ids (Optional[Sequence[str]]): Optional candidate entity ids to search. Defaults to all workspace entities.

        Returns:
            Optional[str]: Canonical entity id, ``None`` when ``entity`` is ``None``, or the original value when unmatched.

        Raises:
            ValueError: If ``entity`` matches more than one registered entity.
        """
        candidates = self.workspace.entities
        if entity_ids is not None:
            candidates = {entity_id: self.workspace.entities[entity_id] for entity_id in entity_ids if entity_id in self.workspace.entities}
        if entity is None or entity in candidates:
            return entity
        value = str(entity).strip()
        if value in candidates:
            return value
        candidate = kebab(value)
        if candidate in candidates:
            return candidate
        matches = []
        for entity_id, entity_cls in candidates.items():
            schema = entity_cls.schema()
            aliases = {entity_id, entity_cls.__name__, schema.name}
            if value.lower() in {alias.lower() for alias in aliases if alias}:
                matches.append(entity_id)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous entity name {entity!r}; matches {matches}")
        return entity
