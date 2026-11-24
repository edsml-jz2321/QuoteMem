"""StarRocks-specific SQL templates consumed by the SQL Backend family."""

from __future__ import annotations

__all__ = ["SR_ADD_COLUMN", "SR_CREATE_DATABASE", "SR_CREATE_OLAP_TABLE", "SR_DROP_TABLE"]

from .base import SqlTemplate

SR_CREATE_DATABASE = SqlTemplate(
    name="heavenbase.starrocks.create_database",
    text="CREATE DATABASE IF NOT EXISTS {database}",
)

SR_CREATE_OLAP_TABLE = SqlTemplate(
    name="heavenbase.starrocks.create_olap_table",
    text=(
        "CREATE TABLE IF NOT EXISTS {table} ({columns}) "
        "ENGINE=OLAP PRIMARY KEY({object_id}) DISTRIBUTED BY HASH({object_id}) BUCKETS {buckets} "
        'PROPERTIES ("replication_num" = "{replication}")'
    ),
)

SR_ADD_COLUMN = SqlTemplate(
    name="heavenbase.starrocks.add_column",
    text="ALTER TABLE {table} ADD COLUMN {column}",
)

SR_DROP_TABLE = SqlTemplate(
    name="heavenbase.starrocks.drop_table",
    text="DROP TABLE IF EXISTS {table}",
)
