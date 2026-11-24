"""MySQL physical text evidence and exact indexed comparison expressions."""

from __future__ import annotations

from heavenbase.utils import Any

MYSQL_TEXT_OPS = ("eq", "ne", "in")
_MYSQL_TEXT_TYPES = {"varchar", "tinytext", "text", "mediumtext", "longtext"}
_MYSQL_VERSION_COMMENTS = ("mysql community server", "mysql enterprise server", "mysql cluster")


def text_evidence(backend: Any, entity_id: str | None, field_schema: Any, strategy: str) -> tuple[str, ...]:
    """Return exact MySQL operations proved for one UTF-8 physical column.

    Args:
        backend: Live SQL backend instance, if available.
        entity_id (str | None): Logical entity identifier for the field.
        field_schema: Logical field schema being compiled.
        strategy (str): Storage strategy identifier.

    Returns:
        tuple[str, ...]: Filter operations with concrete MySQL parity proof.
    """
    if backend is None or entity_id is None or strategy != "inline" or backend.db is None:
        return ()
    table = backend.tables.get(entity_id)
    if table is None or field_schema.name not in table.c:
        return ()
    row = backend.db.execute(
        """
        SELECT VERSION() AS server_version,
               @@version_comment AS version_comment,
               @@character_set_client AS client_charset,
               @@character_set_connection AS connection_charset,
               @@character_set_results AS results_charset,
               c.DATA_TYPE AS data_type,
               c.CHARACTER_SET_NAME AS column_charset,
               c.CHARACTER_MAXIMUM_LENGTH AS max_chars,
               c.CHARACTER_OCTET_LENGTH AS max_bytes,
               c.GENERATION_EXPRESSION AS generation_expression,
               t.TABLE_TYPE AS table_type,
               proof_collation.CHARACTER_SET_NAME AS proof_collation_charset,
               proof_collation.PAD_ATTRIBUTE AS proof_collation_pad
        FROM information_schema.COLUMNS AS c
        JOIN information_schema.TABLES AS t
          ON t.TABLE_SCHEMA = c.TABLE_SCHEMA
         AND t.TABLE_NAME = c.TABLE_NAME
        JOIN information_schema.COLLATIONS AS proof_collation
          ON proof_collation.COLLATION_NAME = 'utf8mb4_bin'
        WHERE c.TABLE_SCHEMA = COALESCE(:table_schema, DATABASE())
          AND c.TABLE_NAME = :table_name
          AND c.COLUMN_NAME = :column_name
        """,
        {"table_schema": table.schema, "table_name": table.name, "column_name": field_schema.name},
        readonly=True,
    ).first()
    if not row:
        return ()
    identity = f"{row.get('server_version', '')} {row.get('version_comment', '')}".lower()
    comment = str(row.get("version_comment") or "").lower()
    if not any(marker in comment for marker in _MYSQL_VERSION_COMMENTS):
        return ()
    if any(marker in identity for marker in ("mariadb", "oceanbase", "tidb", "starrocks", "singlestore", "percona", "aurora")):
        return ()
    charsets = {str(row.get(name) or "").lower() for name in ("client_charset", "connection_charset", "results_charset", "column_charset")}
    max_chars = row.get("max_chars")
    max_bytes = row.get("max_bytes")
    if charsets != {"utf8mb4"} or str(row.get("data_type") or "").lower() not in _MYSQL_TEXT_TYPES:
        return ()
    if str(row.get("table_type") or "").upper() != "BASE TABLE" or str(row.get("generation_expression") or "").strip():
        return ()
    if str(row.get("proof_collation_charset") or "").lower() != "utf8mb4" or str(row.get("proof_collation_pad") or "").upper() != "PAD SPACE":
        return ()
    required_chars = int(field_schema.dtype.max_chars)
    if max_chars is None or max_bytes is None or int(max_chars) < required_chars or int(max_bytes) < required_chars * 4:
        return ()
    return MYSQL_TEXT_OPS


def text_filter(col: Any, op: str, value: Any) -> Any:
    """Compile exact MySQL equality using binary collation plus byte length.

    Args:
        col: SQLAlchemy column expression.
        op (str): Portable filter operation identifier.
        value (Any): Normalized filter operand.

    Returns:
        Any: SQLAlchemy boolean expression with exact MySQL text semantics.
    """
    from sqlalchemy import and_, collate, func, or_

    key = collate(col, "utf8mb4_bin")

    def equal(item: Any) -> Any:
        return and_(key == item, func.octet_length(col) == len(str(item).encode("utf-8")))

    if op == "eq":
        return equal(value)
    if op == "ne":
        return or_(key != value, func.octet_length(col) != len(str(value).encode("utf-8")))
    if op == "in":
        return or_(*[equal(item) for item in value])
    return None
