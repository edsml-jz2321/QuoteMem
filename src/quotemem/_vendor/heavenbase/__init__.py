"""HeavenBase public API.

The root package exposes user-facing workspace, entity, query, prompt, LLM,
Capsule, Toolkit, interop, grouped backend, and utility entry points. Backend
registry, handler, storage, and low-level extension contracts live under
``heavenbase.ext``.
"""

from importlib import import_module as _import_module
from typing import Any

from ._lazy import _attach_lazy_exports, bind_late as _bind_late


def _import_asyncio() -> None:
    """Import ``asyncio`` with the package, so no later first use of it can race another thread.

    Python takes a submodule's import lock before its package's, and
    ``typing_extensions.deprecated`` (used by ``pydantic``, ``mcp`` and
    ``openai``) imports ``asyncio.coroutines`` on first use, so a thread doing
    that while another thread first imports ``asyncio`` (which the LLM gateways
    and the API import) can fail with ``_DeadlockError``. Once ``asyncio`` is in
    ``sys.modules`` both imports return at once. This import can itself meet
    such a thread; when it is the one that fails, ``asyncio`` is left out of
    ``sys.modules``, the other thread imports it, and the import here runs
    again. The other thread can still be the one that fails, so a process that
    imports ``heavenbase`` (or ``asyncio``) before it starts threads avoids the
    cycle altogether.
    """
    for _ in range(100):
        try:
            _import_module("asyncio")
        except RuntimeError as error:
            if type(error).__name__ != "_DeadlockError":
                raise
        else:
            return
    _import_module("asyncio")


_import_asyncio()
del _import_asyncio, _import_module

__all__ = [
    "__version__",
    "configure_explicit_context",
    "Array",
    "Artifact",
    "Aggregate",
    "Catalog",
    "capabilities",
    "backends",
    "BackendType",
    "Boolean",
    "Categorical",
    "Date",
    "Duration",
    "Entity",
    "ExternalRef",
    "Float",
    "GraphEdge",
    "GraphDeleteRestrictedError",
    "HeavenBase",
    "HyperG",
    "Identifier",
    "InlineColumn",
    "Integer",
    "index",
    "InvertedIndex",
    "Json",
    "JsonField",
    "LogicalType",
    "LongText",
    "MediumText",
    "MetaSchema",
    "MemoryNote",
    "Message",
    "Prompt",
    "Query",
    "QuerySpec",
    "Registry",
    "RegistryConflictError",
    "RegistryIntegrityError",
    "ResultFrame",
    "ShortText",
    "SideTable",
    "SparseGramIndex",
    "Session",
    "Skill",
    "StorageStrategy",
    "Timestamp",
    "Translation",
    "Vector",
    "VectorIndex",
    "unique",
    "WorkspaceManifest",
    "WorkspaceSkill",
    "field",
    "filters",
    "CM_HVNB",
    "ConfigEngine",
    "ConfigManager",
    "ConfigSnapshot",
    "ConfigSpec",
    "Context",
    "InterpolationPolicy",
    "resolve_interpolations",
    "Database",
    "DBEngine",
    "DBSpec",
    "DB_ENGINE",
    "DatabaseEngineRegistry",
    "SQLResponse",
    "StableRNG",
    "dumps_json",
    "loads_json",
    "md_section",
    "LLM",
    "LLMCache",
    "LLMEmbeddingEntry",
    "LLMEngine",
    "LLMGateway",
    "LLMImage",
    "LLMImageEntry",
    "LLMResponse",
    "LLMSession",
    "LLMSpec",
    "LLMTextEntry",
    "LLMToolCallRepair",
    "Agent",
    "Capsule",
    "capsule_registry",
    "Tool",
    "Toolkit",
    "InteropIssue",
    "InteropReport",
    "InteropSet",
    "InteropTypeInferer",
    "from_daft",
    "from_numpy",
    "from_pandas",
    "from_pyarrow",
    "from_pydantic",
    "from_sql_result",
    "registered_interop_type_inferers",
    "register_interop_type_inferer",
    "ext",
]

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "__version__": (".version", "__version__"),
    "configure_explicit_context": ("._initialization", "configure_explicit_context"),
    "Aggregate": (".query", "Aggregate"),
    "Array": (".types", "Array"),
    "Artifact": (".types", "Artifact"),
    "BackendType": ("heavenbase.backends", "BackendType"),
    "backends": ("heavenbase.backends", None),
    "Boolean": (".types", "Boolean"),
    "Capsule": ("heavenbase.capsule", "Capsule"),
    "Catalog": ("heavenbase.workspace.catalog", "Catalog"),
    "capabilities": ("heavenbase.capabilities", None),
    "Categorical": (".types", "Categorical"),
    "CM_HVNB": (".utils.config.manager", "CM_HVNB"),
    "ConfigEngine": (".utils.config.engine", "ConfigEngine"),
    "ConfigManager": (".utils.config.manager", "ConfigManager"),
    "ConfigSnapshot": (".utils.config.snapshot", "ConfigSnapshot"),
    "ConfigSpec": (".utils.config.spec", "ConfigSpec"),
    "Context": (".context", "Context"),
    "Database": (".database", "Database"),
    "DatabaseEngineRegistry": (".database", "DatabaseEngineRegistry"),
    "Date": (".types", "Date"),
    "DEFAULT_CONTEXT": (".context", "DEFAULT_CONTEXT"),
    "DBEngine": (".database", "DBEngine"),
    "DBSpec": (".database", "DBSpec"),
    "DB_ENGINE": (".database", "DB_ENGINE"),
    "dumps_json": (".utils.serialize", "dumps_json"),
    "Duration": (".types", "Duration"),
    "Entity": (".entity", "Entity"),
    "ExternalRef": (".strategies", "ExternalRef"),
    "field": (".entity", "field"),
    "filters": (".filters", None),
    "Float": (".types", "Float"),
    "GraphDeleteRestrictedError": (".workspace.graph.errors", "GraphDeleteRestrictedError"),
    "GraphEdge": (".strategies", "GraphEdge"),
    "HeavenBase": (".workspace", "HeavenBase"),
    "HyperG": (".types", "HyperG"),
    "Identifier": (".types", "Identifier"),
    "index": (".placement", "index"),
    "InlineColumn": (".strategies", "InlineColumn"),
    "Integer": (".types", "Integer"),
    "InterpolationPolicy": (".utils.config.interpolation", "InterpolationPolicy"),
    "InvertedIndex": (".strategies", "InvertedIndex"),
    "Json": (".types", "Json"),
    "JsonField": (".strategies", "JsonField"),
    "LLM": (".llm", "LLM"),
    "LLMCache": (".llm", "LLMCache"),
    "LLMEmbeddingEntry": (".llm", "LLMEmbeddingEntry"),
    "LLMEngine": (".llm", "LLMEngine"),
    "LLMGateway": (".llm", "LLMGateway"),
    "LLMImage": (".llm", "LLMImage"),
    "LLMImageEntry": (".llm", "LLMImageEntry"),
    "LLMResponse": (".llm", "LLMResponse"),
    "LLMSession": (".llm", "LLMSession"),
    "LLMSpec": (".llm", "LLMSpec"),
    "LLMTextEntry": (".llm", "LLMTextEntry"),
    "LLMToolCallRepair": (".llm", "LLMToolCallRepair"),
    "loads_json": (".utils.serialize", "loads_json"),
    "LogicalType": (".types", "LogicalType"),
    "LongText": (".types", "LongText"),
    "md_section": (".utils.strings", "md_section"),
    "MediumText": (".types", "MediumText"),
    "MetaSchema": ("heavenbase.workspace.metaschema", "MetaSchema"),
    "Query": (".query", "Query"),
    "QuerySpec": (".query", "QuerySpec"),
    "Registry": (".utils.registry", "Registry"),
    "RegistryConflictError": (".utils.registry", "RegistryConflictError"),
    "RegistryIntegrityError": (".utils.registry", "RegistryIntegrityError"),
    "resolve_interpolations": (".utils.config.spec", "resolve_interpolations"),
    "ResultFrame": (".execution.frame", "ResultFrame"),
    "ShortText": (".types", "ShortText"),
    "SideTable": (".strategies", "SideTable"),
    "SQLResponse": (".database", "SQLResponse"),
    "SparseGramIndex": (".strategies", "SparseGramIndex"),
    "StableRNG": (".utils.rng", "StableRNG"),
    "StorageStrategy": (".strategies", "StorageStrategy"),
    "Timestamp": (".types", "Timestamp"),
    "unique": (".placement", "unique"),
    "utils": (".utils", None),
    "Vector": (".types", "Vector"),
    "VectorIndex": (".strategies", "VectorIndex"),
    "capsule_registry": ("heavenbase.capsule", "capsule_registry"),
    "ext": ("heavenbase.ext", None),
    "InteropIssue": ("heavenbase.interop", "InteropIssue"),
    "InteropReport": ("heavenbase.interop", "InteropReport"),
    "InteropSet": ("heavenbase.interop", "InteropSet"),
    "InteropTypeInferer": ("heavenbase.interop", "InteropTypeInferer"),
    "WorkspaceManifest": ("heavenbase.workspace", "WorkspaceManifest"),
    "WorkspaceSkill": ("heavenbase.workspace", "WorkspaceSkill"),
    "from_daft": ("heavenbase.interop", "from_daft"),
    "from_numpy": ("heavenbase.interop", "from_numpy"),
    "from_pandas": ("heavenbase.interop", "from_pandas"),
    "from_pydantic": ("heavenbase.interop", "from_pydantic"),
    "from_pyarrow": ("heavenbase.interop", "from_pyarrow"),
    "from_sql_result": ("heavenbase.interop", "from_sql_result"),
    "registered_interop_type_inferers": ("heavenbase.interop", "registered_interop_type_inferers"),
    "register_interop_type_inferer": ("heavenbase.interop", "register_interop_type_inferer"),
}

_REGISTRY_ENTITY_EXPORTS = {
    "Agent": "agent-agent",
    "MemoryNote": "memory-note",
    "Message": "agent-message",
    "Prompt": "sys-prompt",
    "Session": "agent-session",
    "Skill": "agent-skill",
    "Toolkit": "sys-toolkit",
    "Translation": "sys-translation",
}

_REGISTRY_SIBLING_EXPORTS = {
    "Tool": ("sys-toolkit", "tool", "Tool"),
}

_DYNAMIC_REGISTRY_EXPORTS = set(_REGISTRY_ENTITY_EXPORTS).union(_REGISTRY_SIBLING_EXPORTS)

# Attribute-only exports resolve on access but stay out of ``__all__``: ``utils`` is a namespace, and ``DEFAULT_CONTEXT``
# is left out so a star import does not bind it. A star import still resolves the Registry-backed entity exports in
# ``__all__`` through the default Context, so it builds that Context, and in explicit Context mode it raises.
_ATTRIBUTE_ONLY_EXPORTS = {"utils", "DEFAULT_CONTEXT"}

if set(_LAZY_EXPORTS).union(_DYNAMIC_REGISTRY_EXPORTS) != set(__all__).union(_ATTRIBUTE_ONLY_EXPORTS):
    raise RuntimeError("HeavenBase root facade exports and lazy routes have drifted")


_lazy_getattr, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)

# ``CM_HVNB`` lives in ``utils``, which may not import the Context that persists it. Every import path that can reach
# ``CM_HVNB`` runs this root first, so its first read opens the scopes the process default Context persists.
_bind_late("heavenbase.config.package_scopes", "heavenbase.context.runtime", "_default_context_config_registry")


def __getattr__(name: str) -> Any:
    """Load one public facade export from its authoritative runtime source."""
    entity_id = _REGISTRY_ENTITY_EXPORTS.get(name)
    if entity_id is not None:
        from .entity.registry import load_entity_type

        return load_entity_type(entity_id)
    sibling = _REGISTRY_SIBLING_EXPORTS.get(name)
    if sibling is not None:
        from .entity.registry import load_entity_artifact_export

        entity_id, module_name, attr_name = sibling
        return load_entity_artifact_export(entity_id, module_name, attr_name)
    return _lazy_getattr(name)
