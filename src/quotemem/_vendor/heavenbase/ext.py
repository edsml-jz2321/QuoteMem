"""Extension-author API for backends, handlers, strategies, and registries."""

from __future__ import annotations

from .utils import Any
from ._lazy import load_lazy_export

_BACKEND_PRIMITIVE_EXPORTS = {
    "Backend": (".backends.base", "Backend"),
    "BackendFactory": (".backends._construction", "BackendFactory"),
    "BackendRequest": (".backends._construction", "BackendRequest"),
    "ScalarIndexSpec": (".backends.capabilities", "ScalarIndexSpec"),
    "BackendMCP": (".backends.mcp", "BackendMCP"),
    "BackendMCPProfile": (".backends.mcp", "BackendMCPProfile"),
    "BackendMCPTool": (".backends.mcp", "BackendMCPTool"),
    "BackendRegistry": (".backends.registry", "BackendRegistry"),
    "BackendType": (".backends.registry", "BackendType"),
    "QueryFragment": (".execution.contracts", "QueryFragment"),
    "HandlerContext": (".execution.contracts", "HandlerContext"),
    "RowOp": (".execution.contracts", "RowOp"),
    "StrategyAdapter": (".backends.strategy", "StrategyAdapter"),
}

_TAG_EXPORTS = {
    "ResolvedTagValue": (".modules.tags", "ResolvedTagValue"),
    "TagDefinition": (".modules.tags", "TagDefinition"),
    "TagView": (".modules.tags", "TagView"),
    "tag": (".modules.tags", "tag"),
    "tag_definition": (".modules.tags", "tag_definition"),
    "tags": (".modules.tags", "tags"),
}

_HANDLER_EXPORTS = {
    "AggregateCompileResult": (".execution.handlers.aggregate", "AggregateCompileResult"),
    "AggregateCompiler": (".execution.handlers.aggregate", "AggregateCompiler"),
    "Handler": (".execution.handlers.base", "Handler"),
    "HandlerCompiler": (".execution.handlers.base", "HandlerCompiler"),
    "HandlerKey": (".execution.handlers.base", "HandlerKey"),
    "HandlerPlugin": (".execution.handlers.plugins", "HandlerPlugin"),
    "HandlerRegistry": (".execution.handlers.registry", "HandlerRegistry"),
    "HandlerSeedContext": (".execution.handlers.plugins", "HandlerSeedContext"),
    "Predicate": (".execution.handlers.predicates", "Predicate"),
    "handler_plugins": (".execution.handlers.plugins", "handler_plugins"),
}

_BACKEND_DISCOVERY_EXPORTS = {
    "backend_aliases": (".backends.registry", "backend_aliases"),
    "register_sql_dialect": (".backends.sql.capabilities", "register_sql_dialect"),
    "registered_sql_dialects": (".backends.sql.capabilities", "registered_sql_dialects"),
    "SQL_JSON_AGG_STYLE_ARRAYAGG": (".backends.sql.dialect", "SQL_JSON_AGG_STYLE_ARRAYAGG"),
    "SQL_JSON_AGG_STYLE_GROUP_ARRAY": (".backends.sql.dialect", "SQL_JSON_AGG_STYLE_GROUP_ARRAY"),
    "SQL_JSON_AGG_STYLE_POSTGRES_ORDERED": (".backends.sql.dialect", "SQL_JSON_AGG_STYLE_POSTGRES_ORDERED"),
    "SQL_JSON_OBJECT_STYLE_BUILD_OBJECT": (".backends.sql.dialect", "SQL_JSON_OBJECT_STYLE_BUILD_OBJECT"),
    "SQL_JSON_OBJECT_STYLE_OBJECT": (".backends.sql.dialect", "SQL_JSON_OBJECT_STYLE_OBJECT"),
    "SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH": (".backends.sql.dialect", "SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH"),
    "SQL_REGEX_STYLE_POSTGRES_TILDE": (".backends.sql.dialect", "SQL_REGEX_STYLE_POSTGRES_TILDE"),
    "SQL_REGEX_STYLE_REGEXP_LIKE": (".backends.sql.dialect", "SQL_REGEX_STYLE_REGEXP_LIKE"),
    "SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE": (".backends.sql.dialect", "SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE"),
    "SQL_WILDCARD_STYLE_GLOB": (".backends.sql.dialect", "SQL_WILDCARD_STYLE_GLOB"),
    "SQL_WILDCARD_STYLE_LIKE": (".backends.sql.dialect", "SQL_WILDCARD_STYLE_LIKE"),
    "SQLDialectSpec": (".backends.sql.dialect", "SQLDialectSpec"),
    "sql_dialect_spec": (".backends.sql.capabilities", "sql_dialect_spec"),
}

_CAPSULE_AUTHOR_EXPORTS = {
    "CapsuleLayer": (".capsule.layer", "CapsuleLayer"),
    "CapsuleLayerBuildContext": (".capsule.layer", "CapsuleLayerBuildContext"),
    "register_capsule_layer": (".capsule.layers", "register_capsule_layer"),
    "registered_capsule_layers": (".capsule.layers", "registered_capsule_layers"),
}

_REGISTRY_TOOLKIT_EXPORTS = {
    "ProfileSpec": ("profile", "ProfileSpec"),
    "profile_spec_from_mapping": ("profiles", "profile_spec_from_mapping"),
    "register_profile": ("profiles", "register_profile"),
    "registered_profiles": ("profiles", "registered_profiles"),
    "registered_profile_tool_names": ("profile_tools", "registered_profile_tool_names"),
    "WorkspaceToolkitFamilySpec": ("families.family", "WorkspaceToolkitFamilySpec"),
    "register_workspace_toolkit_family": ("families.registry", "register_workspace_toolkit_family"),
    "registered_workspace_toolkit_families": ("families.registry", "registered_workspace_toolkit_families"),
    "registered_workspace_tool_names": ("workspace_tools", "registered_workspace_tool_names"),
    "ToolSerializer": ("serializers", "ToolSerializer"),
}

_REGISTRY_DATABASE_EXPORTS = {
    "ingest_database": ("ingest", "ingest_database"),
}

_DATABASE_DIALECT_EXPORTS = {
    name: (".database.dialect", name)
    for name in (
        "DatabaseDialectSpec",
        "MYSQL_TEXT_MIN_LEN",
        "ORACLE_VARCHAR_MAX",
        "SQL_TYPE_CLOB",
        "SQL_TYPE_FLOAT64_BLOB",
        "SQL_TYPE_FLOAT_ARRAY",
        "SQL_TYPE_JSON",
        "SQL_TYPE_JSON_TEXT",
        "SQL_TYPE_MYSQL_TEXT",
        "SQL_TYPE_POSTGRESQL_JSONB",
        "SQL_TYPE_STRING",
        "SQL_TYPE_TEXT",
        "database_dialects",
    )
}

_GRAM_EXPORTS = {
    "register_gram_normalizer": (".backends.sql.strategies.gram", "register_gram_normalizer"),
    "register_gram_tokenizer": (".backends.sql.strategies.gram", "register_gram_tokenizer"),
    "resolve_gram_normalizer": (".backends.sql.strategies.gram", "resolve_gram_normalizer"),
    "resolve_gram_tokenizer": (".backends.sql.strategies.gram", "resolve_gram_tokenizer"),
}

_COMPUTE_EXPORTS = {
    "register_compute": (".entity.computes", "register_compute"),
    "register_query_compute": (".entity.computes", "register_query_compute"),
    "registered_computes": (".entity.computes", "registered_computes"),
    "registered_query_computes": (".entity.computes", "registered_query_computes"),
}

_PLACEMENT_EXPORTS = {
    "registered_storage_profiles": (".placement", "registered_storage_profiles"),
    "StorageBinding": (".placement", "StorageBinding"),
    "IndexBinding": (".placement", "IndexBinding"),
    "IndexRule": (".placement", "IndexRule"),
    "StoragePlan": (".placement", "StoragePlan"),
    "StorageProfile": (".placement", "StorageProfile"),
    "StorageResolver": (".placement", "StorageResolver"),
    "StorageRule": (".placement", "StorageRule"),
    "UniqueBinding": (".placement", "UniqueBinding"),
    "UniqueRule": (".placement", "UniqueRule"),
}

_LAZY_EXPORTS = {}
for _exports in (
    _BACKEND_PRIMITIVE_EXPORTS,
    _TAG_EXPORTS,
    _HANDLER_EXPORTS,
    _BACKEND_DISCOVERY_EXPORTS,
    _CAPSULE_AUTHOR_EXPORTS,
    _DATABASE_DIALECT_EXPORTS,
    _GRAM_EXPORTS,
    _COMPUTE_EXPORTS,
    _PLACEMENT_EXPORTS,
):
    _LAZY_EXPORTS.update(_exports)

__all__ = [
    *_BACKEND_PRIMITIVE_EXPORTS,
    *_TAG_EXPORTS,
    *_HANDLER_EXPORTS,
    *_BACKEND_DISCOVERY_EXPORTS,
    *_DATABASE_DIALECT_EXPORTS,
    "llm_schema_text",
    "register_schema_text",
    "registered_schema_texts",
    "resolve_schema_text",
    *_GRAM_EXPORTS,
    *_COMPUTE_EXPORTS,
    "get_logical_type",
    "logical_type_aliases",
    "register_logical_type",
    "AggregateReducer",
    "aggregate_reducers",
    "register_aggregate_reducer",
    "ExternalRef",
    "GraphEdge",
    "SparseGramIndex",
    "register_strategy",
    "StorageStrategy",
    "strategy_aliases",
    "normalize_op",
    "op_aliases",
    "register_op",
    "registered_ops",
    *_PLACEMENT_EXPORTS,
    "AGENT_EXTENSION_ID",
    "CAPSULE_EXTENSION_ID",
    "DATABASE_EXTENSION_ID",
    "Extension",
    "MEMORY_EXTENSION_ID",
    "PROMPT_EXTENSION_ID",
    "TOOLKIT_EXTENSION_ID",
    *_CAPSULE_AUTHOR_EXPORTS,
    *_REGISTRY_TOOLKIT_EXPORTS,
    *_REGISTRY_DATABASE_EXPORTS,
]

from .extensions import (
    AGENT_EXTENSION_ID,
    CAPSULE_EXTENSION_ID,
    DATABASE_EXTENSION_ID,
    MEMORY_EXTENSION_ID,
    PROMPT_EXTENSION_ID,
    TOOLKIT_EXTENSION_ID,
    Extension,
)
from .schema.text import llm_schema_text, register_schema_text, registered_schema_texts, resolve_schema_text
from .strategies import (
    ExternalRef,
    GraphEdge,
    SparseGramIndex,
    register_strategy,
    StorageStrategy,
    strategy_aliases,
)
from .types import (
    get_logical_type,
    logical_type_aliases,
    register_logical_type,
)
from .query.aggregate import AggregateReducer, aggregate_reducers, register_aggregate_reducer
from .filters.ops import normalize_op, op_aliases, register_op, registered_ops


def __getattr__(name: str) -> Any:
    """Load one lazy extension-author export.

    Args:
        name (str): Public symbol requested from `heavenbase.ext`.

    Returns:
        Any: Resolved public symbol cached on this module.

    Raises:
        AttributeError: If `name` is not part of the extension-author surface.
    """
    registry_target = _REGISTRY_TOOLKIT_EXPORTS.get(name)
    if registry_target is not None:
        from .entity.registry import load_entity_artifact_export

        module_name, attr_name = registry_target
        return load_entity_artifact_export("sys-toolkit", module_name, attr_name)
    registry_target = _REGISTRY_DATABASE_EXPORTS.get(name)
    if registry_target is not None:
        from .entity.registry import load_entity_artifact_export

        module_name, attr_name = registry_target
        return load_entity_artifact_export("db-database", module_name, attr_name)
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name, *_ = target
    return load_lazy_export(name, {name: (module_name, attr_name)}, __package__, globals(), owner=__name__)


def __dir__() -> list[str]:
    """List runtime and declared extension-author attributes."""
    return sorted(set(globals()).union(__all__))
