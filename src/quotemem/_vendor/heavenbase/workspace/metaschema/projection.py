"""Private MetaSchema projection for Registry and workspace runtime metadata."""

from __future__ import annotations

from collections.abc import Mapping

from heavenbase.utils import Any, Dict, List, deepcopy, sha256hash

from ...backends.base import Backend
from ...backends.registry import BackendType
from ...modules.executable import EXECUTABLE_REF_KIND
from .entity import MetaSchema

_PROJECTED_KINDS = {
    "dialect": "sql_dialect",
    "reducer": "aggregate_reducer",
    "serializer": "mcp_serializer",
    "strategy": "storage_strategy",
}
_PRIVATE_META_FIELDS = frozenset(("installation", "fingerprints", "restore_layers", "tags"))
_STRUCTURAL_GROUPS = frozenset(("backend", "bundle", "dtype", "provenance", "scope", "sqlglot"))
_STRUCTURAL_GROUP_VALUES = frozenset((("tag", "builtin"),))


class _MetaSchemaRegistryRows:
    """Build safe MetaSchema rows from authoritative records and live state."""

    def __init__(self, service: Any):
        self.service = service
        self.workspace = service.workspace
        self._registry_memo: tuple[Any, ...] | None = None

    def workspace_rows(self) -> List[dict]:
        """Return Registry projections plus workspace-owned runtime rows."""
        return [deepcopy(row) for row in self.workspace_rows_and_digest()[0]]

    def workspace_rows_and_digest(self) -> tuple[List[dict], str]:
        """Return the workspace rows, shared with a memo and not to be changed, and their digest.

        The Registry and handler rows are thousands of rows that change only
        when the module catalog publishes, a handler or aggregate compiler is
        registered, or the configured backends change. This projection keeps
        them, with their digest, for as long as those inputs are the same.
        """
        ws = self.workspace
        configured_backend_identifiers = {BackendType.identifier_of(backend, resolver=ws.module_resolver) for _, backend in ws.backends.items()}
        head = [self.service.build_system_row("workspace", ws.id, name=ws.id, desc="Workspace root object")]
        for backend_name, backend in ws.backends.items():
            backend_identifier = BackendType.identifier_of(backend, resolver=ws.module_resolver)
            head.append(
                self.service.build_system_row(
                    "backend",
                    backend_name,
                    name=backend_name,
                    backend=backend_name,
                    type=backend_identifier,
                    meta=self.backend_payload(backend),
                )
            )
        registry_rows, registry_digest = self._registry_rows(configured_backend_identifiers)
        tail = []
        for dtype, profile in sorted(ws._storage_profiles.workspace_overrides().items()):
            meta = profile.to_dict()
            meta["source"] = "workspace"
            tail.append(
                self.service.build_system_row(
                    "storage_profile",
                    dtype,
                    name=dtype,
                    desc=f"Workspace storage profile override for logical type {dtype}.",
                    tags=["storage", "profile", "registry", "workspace"],
                    dtype=dtype,
                    strategy=profile.default_strategy,
                    source="workspace",
                    meta=meta,
                )
            )
        digest = sha256hash([sha256hash(head), registry_digest, sha256hash(tail)])
        return [*head, *registry_rows, *tail], digest

    def _registry_rows(self, configured_backend_identifiers: set[str]) -> tuple[tuple[dict, ...], str]:
        """Return the module, handler and aggregate rows and their digest, reusing unchanged ones."""
        ws = self.workspace
        handlers = ws.handlers
        # The states are immutable published snapshots: the same object means the same registrations.
        states = (handlers._handlers._state, handlers._aggregate_compilers._state)
        key = (
            MetaSchema.schema().entity_id in ws.entities,
            ws.module_resolver.catalog.publication,
            frozenset(configured_backend_identifiers),
            ws.context.config.get("heavenbase.entity.id.hash_length", default=42),
        )
        memo = self._registry_memo
        if memo is not None and memo[0] == key and all(old is new for old, new in zip(memo[1], states)):
            return memo[2], memo[3]
        registrations = ws.module_resolver.inspect()
        rows = [self._module_row(row) for row in registrations]
        aliases = {(str(row["kind"]), str(row["identifier"])): tuple(str(alias) for alias in row.get("aliases") or ()) for row in registrations}
        for row in handlers.capability_rows():
            if row["backend_identifier"] not in configured_backend_identifiers:
                continue
            cid = f"{row['backend_identifier']}:{row['dtype']}:{row['op']}:{row['strategy']}"
            rows.append(
                self.service.build_system_row(
                    "capability",
                    cid,
                    name=row["op"],
                    dtype=row["dtype"],
                    type=row["backend_identifier"],
                    strategy=row["strategy"],
                    source="handler",
                    meta=self.capability_payload(row, aliases),
                )
            )
        for row in handlers.aggregate_capability_rows():
            backend_identifier = row["backend_identifier"]
            if backend_identifier not in configured_backend_identifiers:
                continue
            rows.append(
                self.service.build_system_row(
                    "aggregate_compiler",
                    backend_identifier,
                    name=backend_identifier,
                    type=backend_identifier,
                    strategy=row["strategy"],
                    tags=["registry", "aggregate", "compiler", "native"],
                    source="handler",
                    meta=row,
                )
            )
        frozen = tuple(rows)
        digest = sha256hash(rows)
        if (ws.module_resolver.catalog.publication, handlers._handlers._state, handlers._aggregate_compilers._state) == (key[1], *states):
            self._registry_memo = (key, states, frozen, digest)
        return frozen, digest

    def module_rows(self) -> List[dict]:
        """Project inert `system-modules` records without resolving them."""
        return [self._module_row(row) for row in self.workspace.module_resolver.inspect()]

    def registration_meta(self, kind: str, identifier: str) -> Dict[str, Any]:
        """Return safe registration metadata for a domain-row merge."""
        rows = self.workspace.module_resolver.inspect(kind, identifier)
        return self._safe_record_meta(rows[0]) if rows else {}

    def _module_row(self, row: Mapping[str, Any]) -> dict:
        """Build one safe MetaSchema row from an inspection mapping."""
        source_kind = str(row["kind"])
        category = _PROJECTED_KINDS.get(source_kind, source_kind)
        identifier = str(row["identifier"])
        record = row["record"]
        meta = self._safe_record_meta(row)
        definition = self._definition(record)
        target = self._safe_target(record, definition)
        values: Dict[str, Any] = {
            "name": str(record.get("name") or identifier),
            "desc": str(record.get("desc") or ""),
            "tags": self._module_tags(source_kind, row.get("groups")),
            "source": str(record.get("source") or ""),
            "target": target,
            "active": bool(record.get("active", True)),
            "meta": meta,
        }
        for field in ("dtype", "backend", "type", "strategy", "field"):
            if record.get(field) not in (None, ""):
                values[field] = str(record[field])
        if source_kind == "logical_type":
            values.setdefault("dtype", identifier)
        elif source_kind == "strategy":
            values.setdefault("strategy", identifier)
        elif source_kind in {"backend_type", "database_dialect", "dialect", "sql_dialect"}:
            values.setdefault("type", identifier)
        elif source_kind == "mcp_profile" and isinstance(definition, Mapping):
            serializer = str(definition.get("serializer") or "")
            values.setdefault("type", serializer)
            values["target"] = serializer
        row_data = self.service.build_system_row(category, identifier, **values)
        if source_kind in {"entity", "extension"} and row_data:
            row_data["object_id"] = self.service.system_row_id("registration", f"{source_kind}:{identifier}")
        if row_data:
            row_data["active"] = bool(record.get("active", True))
        return row_data

    def _safe_record_meta(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """Return redacted record metadata plus Registry provenance."""
        record = row["record"]
        raw_meta = record.get("meta")
        meta = (
            {key: self._safe_meta_value(value) for key, value in raw_meta.items() if key not in _PRIVATE_META_FIELDS} if isinstance(raw_meta, Mapping) else {}
        )
        definition = meta.pop(str(record.get("target")), None) if record.get("source") == "inline" else None
        if isinstance(definition, Mapping):
            meta.update(dict(definition))
        elif definition is not None:
            meta["definition"] = definition
        meta.update(
            {
                "identifier": str(row["identifier"]),
                "record_kind": str(row["kind"]),
                "aliases": list(row.get("aliases") or ()),
                "groups": {str(key): list(values) for key, values in (row.get("groups") or {}).items()},
                "registry_revision": int(row["revision"]),
                "registry_fingerprint": str(row["fingerprint"]),
            }
        )
        if record.get("source") == "python-import" and isinstance(record.get("target"), str):
            meta["class"] = str(record["target"])
        return meta

    @staticmethod
    def _definition(record: Mapping[str, Any]) -> Any:
        meta = record.get("meta")
        if record.get("source") != "inline" or not isinstance(meta, Mapping):
            return None
        return meta.get(str(record.get("target")))

    @staticmethod
    def _safe_target(record: Mapping[str, Any], definition: Any) -> str:
        source = str(record.get("source") or "")
        target = record.get("target")
        if source == "artifact" and isinstance(target, Mapping):
            module = str(target.get("module") or "")
            qualname = str(target.get("qualname") or "")
            return f"{module}:{qualname}" if module and qualname else ""
        if source == "inline" and isinstance(definition, Mapping):
            for field in ("serialize", "builder", "fn", "factory", "serializer"):
                value = definition.get(field)
                executable = _MetaSchemaRegistryRows._safe_executable_target(value)
                if executable:
                    return executable
                if isinstance(value, (str, int, float, bool)):
                    return str(value)
        return str(target) if isinstance(target, (str, int, float, bool)) else ""

    @staticmethod
    def _safe_executable_target(value: Any) -> str:
        """Project one canonical executable reference without restoring it."""
        if not isinstance(value, Mapping) or value.get("kind") != EXECUTABLE_REF_KIND:
            return ""
        source = str(value.get("source") or "")
        target = value.get("target")
        if source == "artifact" and isinstance(target, Mapping):
            module = str(target.get("module") or "")
            qualname = str(target.get("qualname") or "")
            return f"{module}:{qualname}" if module and qualname else qualname
        if source in {"python-import", "capsule"} and isinstance(target, str):
            return target
        return ""

    @classmethod
    def _safe_meta_value(cls, value: Any) -> Any:
        """Project nested executable references as inert readable targets."""
        executable = cls._safe_executable_target(value)
        if executable:
            return executable
        if isinstance(value, Mapping):
            return {str(key): cls._safe_meta_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._safe_meta_value(item) for item in value]
        return value

    @staticmethod
    def _module_tags(kind: str, groups: Any) -> List[str]:
        tags = ["registry", kind]
        if isinstance(groups, Mapping):
            for group, values in groups.items():
                for value in values:
                    text = str(value)
                    if (str(group) in _STRUCTURAL_GROUPS or (str(group), text) in _STRUCTURAL_GROUP_VALUES) and text not in tags:
                        tags.append(text)
        return tags

    def capability_payload(
        self,
        row: Mapping[str, Any],
        aliases: Mapping[tuple[str, str], tuple[str, ...]],
    ) -> Dict[str, Any]:
        """Return inspectable vocabulary metadata for one handler key."""
        dtype = str(row["dtype"])
        op = str(row["op"])
        backend_identifier = str(row["backend_identifier"])
        strategy = str(row["strategy"])
        payload = dict(row)
        payload["handler_key"] = [dtype, op, backend_identifier, strategy]
        payload["aliases"] = {
            "dtype": list(aliases.get(("logical_type", dtype), ())),
            "op": list(aliases.get(("operation", op), ())),
            "backend": list(aliases.get(("backend_type", backend_identifier), ())),
            "strategy": list(aliases.get(("strategy", strategy), ())),
        }
        return payload

    def backend_payload(self, backend: Backend) -> dict:
        """Return backend metadata for workspace health and system rows."""
        return backend.to_dict()
