"""Database-dialect SQL templates."""

from __future__ import annotations

__all__ = [
    "DB_ADD_COLUMN",
    "DB_DDL_AFTER_CREATE",
    "DB_DDL_BEFORE_DROP_FORCE",
    "DB_DDL_CREATE",
    "DB_DDL_DROP",
    "DB_DDL_EXISTS",
    "DB_CURRENT_DATABASE",
    "DB_LIST_DATABASES",
    "DB_REFLECT_COLUMNS",
    "DB_REFLECT_PRIMARY_KEYS",
    "DB_RENAME_TABLE",
]

from .base import SqlTemplate

DB_ADD_COLUMN = {
    "generic": SqlTemplate("heavenbase.db.add_column.generic", "ALTER TABLE {table} ADD COLUMN {column}"),
    "mssql": SqlTemplate("heavenbase.db.add_column.mssql", "ALTER TABLE {table} ADD {column}"),
    "oracle": SqlTemplate("heavenbase.db.add_column.oracle", "ALTER TABLE {table} ADD ({column})"),
}

DB_DDL_EXISTS = {
    "postgresql": SqlTemplate("heavenbase.db.exists.postgresql", "SELECT 1 FROM pg_database WHERE datname = :name"),
    "mysql": SqlTemplate("heavenbase.db.exists.mysql", "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = :name"),
    "starrocks": SqlTemplate("heavenbase.db.exists.starrocks", "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = :name"),
    "mssql": SqlTemplate("heavenbase.db.exists.mssql", "SELECT 1 FROM sys.databases WHERE name = :name"),
    "oracle": SqlTemplate("heavenbase.db.exists.oracle", "SELECT COUNT(*) FROM all_users WHERE username = UPPER(:name)"),
}

DB_DDL_CREATE = {
    "postgresql": SqlTemplate("heavenbase.db.create.postgresql", "CREATE DATABASE {name}"),
    "mysql": SqlTemplate("heavenbase.db.create.mysql", "CREATE DATABASE {name}"),
    "starrocks": SqlTemplate("heavenbase.db.create.starrocks", "CREATE DATABASE {name}"),
    "mssql": SqlTemplate("heavenbase.db.create.mssql", "CREATE DATABASE {name}"),
    "oracle": SqlTemplate("heavenbase.db.create.oracle", 'CREATE USER {name} IDENTIFIED BY "{password}"'),
    "hive": SqlTemplate("heavenbase.db.create.hive", "CREATE DATABASE IF NOT EXISTS {name}"),
}

DB_DDL_DROP = {
    "postgresql": SqlTemplate("heavenbase.db.drop.postgresql", "DROP DATABASE {name}"),
    "mysql": SqlTemplate("heavenbase.db.drop.mysql", "DROP DATABASE {name}"),
    "starrocks": SqlTemplate("heavenbase.db.drop.starrocks", "DROP DATABASE {name}"),
    "mssql": SqlTemplate("heavenbase.db.drop.mssql", "DROP DATABASE {name}"),
    "oracle": SqlTemplate("heavenbase.db.drop.oracle", "DROP USER {name} CASCADE"),
    "hive": SqlTemplate("heavenbase.db.drop.hive", "DROP DATABASE IF EXISTS {name} CASCADE"),
}

DB_DDL_BEFORE_DROP_FORCE = {
    "postgresql": SqlTemplate(
        "heavenbase.db.before_drop_force.postgresql", "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = :name AND pid <> pg_backend_pid()"
    ),
    "mssql": SqlTemplate("heavenbase.db.before_drop_force.mssql", "ALTER DATABASE {database} SET SINGLE_USER WITH ROLLBACK IMMEDIATE"),
}

DB_DDL_AFTER_CREATE = {
    "oracle": (
        SqlTemplate("heavenbase.db.after_create.oracle.create_session", "GRANT CREATE SESSION TO {name}"),
        SqlTemplate("heavenbase.db.after_create.oracle.create_table", "GRANT CREATE TABLE TO {name}"),
        SqlTemplate("heavenbase.db.after_create.oracle.create_sequence", "GRANT CREATE SEQUENCE TO {name}"),
        SqlTemplate("heavenbase.db.after_create.oracle.create_view", "GRANT CREATE VIEW TO {name}"),
        SqlTemplate("heavenbase.db.after_create.oracle.unlimited_tablespace", "GRANT UNLIMITED TABLESPACE TO {name}"),
    ),
    "postgresql_owner": (
        SqlTemplate("heavenbase.db.after_create.postgresql.owner", "ALTER DATABASE {database} OWNER TO {username}"),
        SqlTemplate("heavenbase.db.after_create.postgresql.grant", "GRANT ALL PRIVILEGES ON DATABASE {database} TO {username}"),
    ),
}

DB_CURRENT_DATABASE = {
    "sqlite": SqlTemplate("heavenbase.db.current.sqlite", "PRAGMA database_list"),
    "duckdb": SqlTemplate("heavenbase.db.current.duckdb", "SELECT current_database() AS name"),
    "postgresql": SqlTemplate("heavenbase.db.current.postgresql", "SELECT current_database() AS name"),
    "mysql": SqlTemplate("heavenbase.db.current.mysql", "SELECT DATABASE() AS name"),
    "mssql": SqlTemplate("heavenbase.db.current.mssql", "SELECT DB_NAME() AS name"),
    "starrocks": SqlTemplate("heavenbase.db.current.starrocks", "SELECT DATABASE() AS name"),
    "oceanbase": SqlTemplate("heavenbase.db.current.oceanbase", "SELECT DATABASE() AS name"),
}

DB_LIST_DATABASES = {
    "postgresql": SqlTemplate("heavenbase.db.list.postgresql", "SELECT datname AS name FROM pg_database"),
    "mysql": SqlTemplate("heavenbase.db.list.mysql", "SELECT SCHEMA_NAME AS name FROM INFORMATION_SCHEMA.SCHEMATA"),
}

DB_RENAME_TABLE = {
    "generic": SqlTemplate("heavenbase.db.rename_table.generic", "ALTER TABLE {old_name} RENAME TO {new_name}"),
    "mssql": SqlTemplate("heavenbase.db.rename_table.mssql", "EXEC sp_rename :old_name, :new_name"),
}

DB_REFLECT_COLUMNS = {
    "duckdb": SqlTemplate("heavenbase.db.reflect_columns.duckdb", "DESCRIBE {table_name}"),
}

DB_REFLECT_PRIMARY_KEYS = {
    "duckdb": SqlTemplate("heavenbase.db.reflect_primary_keys.duckdb", "PRAGMA table_info({table_name_literal})"),
}
