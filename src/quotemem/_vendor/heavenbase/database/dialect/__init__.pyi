"""Database-owned dialect metadata type surface."""

from .registry import BootstrapDialectCatalog as BootstrapDialectCatalog
from .registry import DatabaseDialectCatalog as DatabaseDialectCatalog
from .registry import ResolverDialectCatalog as ResolverDialectCatalog
from .registry import database_dialects as database_dialects
from .spec import DatabaseDialectSpec as DatabaseDialectSpec
from .spec import MYSQL_TEXT_MIN_LEN as MYSQL_TEXT_MIN_LEN
from .spec import ORACLE_VARCHAR_MAX as ORACLE_VARCHAR_MAX
from .spec import SQL_COMMON_RESERVED_WORDS as SQL_COMMON_RESERVED_WORDS
from .spec import SQL_TYPE_CLOB as SQL_TYPE_CLOB
from .spec import SQL_TYPE_FLOAT64_BLOB as SQL_TYPE_FLOAT64_BLOB
from .spec import SQL_TYPE_FLOAT_ARRAY as SQL_TYPE_FLOAT_ARRAY
from .spec import SQL_TYPE_JSON as SQL_TYPE_JSON
from .spec import SQL_TYPE_JSON_TEXT as SQL_TYPE_JSON_TEXT
from .spec import SQL_TYPE_MYSQL_TEXT as SQL_TYPE_MYSQL_TEXT
from .spec import SQL_TYPE_POSTGRESQL_JSONB as SQL_TYPE_POSTGRESQL_JSONB
from .spec import SQL_TYPE_STRING as SQL_TYPE_STRING
from .spec import SQL_TYPE_TEXT as SQL_TYPE_TEXT

__all__: list[str]
