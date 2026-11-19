"""Which failures stay inside one scope, and which ones fail the runtime.

A shard holds many user scopes behind one Writer, so a failure is fatal only
when it can compromise every scope: the workspace itself, its storage owner or
the shared read model. Everything else is isolated to the scope that raised it.

Fatal (``is_runtime_failure``): resource exhaustion and process termination
(``MemoryError``, a non-``Exception``), the filesystem (``OSError``: disk full,
I/O), every raw HeavenBase or SQLAlchemy storage failure (a locked, read-only,
corrupt or unreachable database, a failed commit whose outcome is unknown), a
lost Writer lease or binding and an unavailable shared read model (the
``AMLError`` codes below).

Scope-local: every other ``AMLError`` (its scope is quarantined, or the owner
retries within its own deadline), any other exception (``internal_error``, a
defect isolated by quarantine) and ``storage_conflict``: a duplicate key that
refused one prepared transaction which HeavenBase rolled back whole
(``constraint_violation``), so nothing of it was written. The Writer raises
``storage_conflict`` only where it proved that rollback (``Writer._write_plan``);
a raw constraint failure anywhere else stays fatal.
"""

import sqlite3

from heavenbase.database.execution import DatabaseError
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from aml.core.errors import AMLError

_CHAIN_LIMIT = 8
"""Causes followed from a wrapper to the database driver's own error."""

_DUPLICATE_KEY_CODES = frozenset({2067, 1555})
"""SQLite's extended result codes SQLITE_CONSTRAINT_UNIQUE and _PRIMARYKEY."""


def is_runtime_failure(error: BaseException) -> bool:
    """Classify failures that cannot be isolated by a scoped durable receipt.

    Storage ownership, uncertain persistence, an unavailable shared read model,
    resource exhaustion and process termination affect the runtime itself.
    HeavenBase execution and SQLAlchemy transaction failures remain fatal even
    when a later write could succeed. Other errors require the caller to
    quarantine its proved scope; this predicate grants no write authority.
    """
    return not isinstance(error, Exception) or isinstance(
        error, MemoryError | OSError | DatabaseError | SQLAlchemyError,
    ) or (
        isinstance(error, AMLError) and error.code in {
            "workspace_unsafe", "workspace_locked", "workspace_binding_mismatch",
            "cas_outcome_ambiguous", "service_unhealthy", "recovery_required",
        }
    )


def constraint_violation(error: BaseException) -> bool:
    """Whether the database driver refused a statement only on a duplicate key.

    Follows the explicit and implicit causes from a HeavenBase or SQLAlchemy
    wrapper to the first driver-level error and accepts only SQLite's UNIQUE or
    PRIMARY KEY constraint (a plan that reuses an existing key, as a reminted
    ``mint_key`` did). Every other driver error stays fatal: NOT NULL, CHECK,
    FOREIGN KEY and a trigger's ``RAISE`` included, as do disk, lock, I/O,
    corruption and programming errors and a chain without a driver error.

    Args:
        error: The raised exception.

    Returns:
        True only for a driver-reported duplicate-key violation.
    """
    current: BaseException | None = error
    for _ in range(_CHAIN_LIMIT):
        if current is None:
            return False
        if isinstance(current, DBAPIError):
            current = current.orig
            continue
        if isinstance(current, sqlite3.Error):
            return isinstance(current, sqlite3.IntegrityError) and getattr(
                current, "sqlite_errorcode", None,
            ) in _DUPLICATE_KEY_CODES
        current = current.__cause__ or current.__context__
    return False
