"""Portable text-semantics proof helpers shared by SQL compilers."""

from __future__ import annotations

__all__ = [
    "TEXT_SEMANTICS_NOT_PROVEN",
    "sql_text_key_expression",
    "sql_text_key_ops",
    "sql_text_key_semantics_proven",
    "sql_text_ops",
    "sql_text_semantics_proven",
]

from heavenbase.utils import Any

from heavenbase.execution.contracts import HandlerContext
from heavenbase.backends.sql.capabilities import sql_dialect_spec
from heavenbase.types.storage import stores_as_text

TEXT_SEMANTICS_NOT_PROVEN = "text_semantics_not_proven"


def sql_text_ops(field_schema: Any, ctx: HandlerContext) -> tuple[str, ...]:
    """Return native text operations proved for a live SQL field.

    Args:
        field_schema: Logical field schema being compiled.
        ctx: Handler context carrying the live backend and entity identity.

    Returns:
        tuple[str, ...]: Intersection of provider candidates and concrete-field
        evidence. Non-text fields and contexts without live evidence return an
        empty tuple.
    """
    if not stores_as_text(field_schema.dtype):
        return ()
    dialect = sql_dialect_spec(ctx.backend_identifier, resolver=ctx.resolver)
    return dialect.text_ops(ctx.backend_instance, ctx.entity_id, field_schema, ctx.strategy)


def sql_text_semantics_proven(field_schema: Any, op: str, ctx: HandlerContext) -> bool:
    """Return whether one SQL text operation has concrete-field parity proof.

    Args:
        field_schema: Logical field schema being compiled.
        op: Portable filter operation identifier.
        ctx: Handler context carrying the live backend and entity identity.

    Returns:
        bool: True for non-text storage or when the provider proves `op` for
        this concrete field.
    """
    return not stores_as_text(field_schema.dtype) or str(op) in sql_text_ops(field_schema, ctx)


def sql_text_key_ops(field_schema: Any, ctx: HandlerContext) -> tuple[str, ...]:
    """Return aggregate text-key purposes proved for a live SQL field.

    Args:
        field_schema: Logical field schema being compiled.
        ctx: Handler context carrying the live backend and entity identity.

    Returns:
        tuple[str, ...]: Provider key purposes with independent physical-field
        evidence. Non-text fields return an empty tuple.
    """
    if not stores_as_text(field_schema.dtype):
        return ()
    dialect = sql_dialect_spec(ctx.backend_identifier, resolver=ctx.resolver)
    return dialect.text_key_ops(ctx.backend_instance, ctx.entity_id, field_schema, ctx.strategy)


def sql_text_key_semantics_proven(field_schema: Any, purpose: str, ctx: HandlerContext) -> bool:
    """Return whether one aggregate text-key purpose has parity proof.

    Args:
        field_schema: Logical field schema being compiled.
        purpose: Aggregate key purpose: `group`, `distinct`, or `order`.
        ctx: Handler context carrying the live backend and entity identity.

    Returns:
        bool: True for non-text storage or independently proved text keys.
    """
    return not stores_as_text(field_schema.dtype) or str(purpose) in sql_text_key_ops(field_schema, ctx)


def sql_text_key_expression(col: Any, field_schema: Any, purpose: str, ctx: HandlerContext) -> Any:
    """Return the provider-owned aggregate key for a proved text field.

    Args:
        col: SQLAlchemy column expression.
        field_schema: Logical field schema being compiled.
        purpose: Aggregate key purpose: `group`, `distinct`, or `order`.
        ctx: Handler context identifying the SQL provider.

    Returns:
        Any: The original column for non-text storage, otherwise the provider's
        value-preserving text comparison expression.
    """
    if not stores_as_text(field_schema.dtype):
        return col
    dialect = sql_dialect_spec(ctx.backend_identifier, resolver=ctx.resolver)
    if ctx.unsafe and dialect.text_key_factory is None:
        return col
    return dialect.text_key_expression(col, purpose)
