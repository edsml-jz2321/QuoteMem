"""Built-in backend implementations."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

_CONTRACT_EXPORTS = {
    "Backend": (".base", "Backend"),
    "BackendFactory": ("._construction", "BackendFactory"),
    "BackendRequest": ("._construction", "BackendRequest"),
    "BackendMCP": (".mcp", "BackendMCP"),
    "BackendMCPProfile": (".mcp", "BackendMCPProfile"),
    "BackendMCPTool": (".mcp", "BackendMCPTool"),
    "ScalarIndexSpec": (".capabilities", "ScalarIndexSpec"),
    "StrategyAdapter": (".strategy", "StrategyAdapter"),
}
_REGISTRY_EXPORTS = {
    "BackendRegistry": (".registry", "BackendRegistry"),
    "BackendType": (".registry", "BackendType"),
    "backend_aliases": (".registry", "backend_aliases"),
}
_PROVIDER_EXPORTS = {
    "ChromaBackend": (".vector.chroma", "ChromaBackend"),
    "ClickHouseBackend": (".sql.clickhouse", "ClickHouseBackend"),
    "DoltBackend": (".sql.dolt", "DoltBackend"),
    "DuckDBBackend": (".sql.duckdb", "DuckDBBackend"),
    "ElasticsearchBackend": (".elasticsearch.backend", "ElasticsearchBackend"),
    "FileBackend": (".file.base", "FileBackend"),
    "InMemBackend": (".inmem.backend", "InMemBackend"),
    "JsonFileBackend": (".file.json", "JsonFileBackend"),
    "LanceBackend": (".vector.lance", "LanceBackend"),
    "MilvusBackend": (".vector.milvus", "MilvusBackend"),
    "MilvusLiteBackend": (".vector.milvus", "MilvusLiteBackend"),
    "MongoDBBackend": (".mongodb.backend", "MongoDBBackend"),
    "MsSQLBackend": (".sql.mssql", "MsSQLBackend"),
    "MySQLBackend": (".sql.mysql", "MySQLBackend"),
    "Neo4jBackend": (".neo4j.backend", "Neo4jBackend"),
    "OceanBaseBackend": (".sql.oceanbase", "OceanBaseBackend"),
    "OpenSearchBackend": (".opensearch.backend", "OpenSearchBackend"),
    "OracleBackend": (".sql.oracle", "OracleBackend"),
    "PgVectorBackend": (".vector.pgvector", "PgVectorBackend"),
    "PickleFileBackend": (".file.pickle", "PickleFileBackend"),
    "PineconeBackend": (".vector.pinecone", "PineconeBackend"),
    "PostgresBackend": (".sql.postgres", "PostgresBackend"),
    "QdrantBackend": (".vector.qdrant", "QdrantBackend"),
    "RedisBackend": (".redis.backend", "RedisBackend"),
    "RediSearchBackend": (".vector.redisearch", "RediSearchBackend"),
    "SeekDBBackend": (".sql.seekdb", "SeekDBBackend"),
    "SQLBackend": (".sql.base", "SQLBackend"),
    "SQLiteBackend": (".sql.sqlite", "SQLiteBackend"),
    "StarRocksBackend": (".sql.starrocks", "StarRocksBackend"),
    "SupabaseBackend": (".sql.supabase", "SupabaseBackend"),
    "SurrealBackend": (".surreal.backend", "SurrealBackend"),
    "TinyDBBackend": (".tinydb.backend", "TinyDBBackend"),
    "TrinoBackend": (".sql.trino", "TrinoBackend"),
    "WeaviateBackend": (".vector.weaviate", "WeaviateBackend"),
}
_SQL_DIALECT_EXPORTS = {
    "SQL_JSON_AGG_STYLE_ARRAYAGG": (".sql.dialect", "SQL_JSON_AGG_STYLE_ARRAYAGG"),
    "SQL_JSON_AGG_STYLE_GROUP_ARRAY": (".sql.dialect", "SQL_JSON_AGG_STYLE_GROUP_ARRAY"),
    "SQL_JSON_AGG_STYLE_POSTGRES_ORDERED": (".sql.dialect", "SQL_JSON_AGG_STYLE_POSTGRES_ORDERED"),
    "SQL_JSON_OBJECT_STYLE_BUILD_OBJECT": (".sql.dialect", "SQL_JSON_OBJECT_STYLE_BUILD_OBJECT"),
    "SQL_JSON_OBJECT_STYLE_OBJECT": (".sql.dialect", "SQL_JSON_OBJECT_STYLE_OBJECT"),
    "SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH": (".sql.dialect", "SQL_REGEX_STYLE_DUCKDB_TILDE_SEARCH"),
    "SQL_REGEX_STYLE_POSTGRES_TILDE": (".sql.dialect", "SQL_REGEX_STYLE_POSTGRES_TILDE"),
    "SQL_REGEX_STYLE_REGEXP_LIKE": (".sql.dialect", "SQL_REGEX_STYLE_REGEXP_LIKE"),
    "SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE": (".sql.dialect", "SQL_REGEX_STYLE_REGEXP_LIKE_CASE_SENSITIVE"),
    "SQL_WILDCARD_STYLE_GLOB": (".sql.dialect", "SQL_WILDCARD_STYLE_GLOB"),
    "SQL_WILDCARD_STYLE_LIKE": (".sql.dialect", "SQL_WILDCARD_STYLE_LIKE"),
    "SQLDialectSpec": (".sql.dialect", "SQLDialectSpec"),
    "register_sql_dialect": (".sql.capabilities", "register_sql_dialect"),
    "registered_sql_dialects": (".sql.capabilities", "registered_sql_dialects"),
    "sql_dialect_spec": (".sql.capabilities", "sql_dialect_spec"),
}
_LAZY_EXPORTS = {**_CONTRACT_EXPORTS, **_REGISTRY_EXPORTS, **_PROVIDER_EXPORTS, **_SQL_DIALECT_EXPORTS}

__all__ = [*_LAZY_EXPORTS]


__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
