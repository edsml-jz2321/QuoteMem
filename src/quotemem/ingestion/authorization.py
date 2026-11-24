"""Exact-scope reception authorization, separate from durable execution tickets."""

from quotemem.core.errors import QuoteMemError
from quotemem.core.query import ScopeV1
from quotemem.providers.auth import PrincipalGrant


def check_add_grant(
    grant: PrincipalGrant, scope: ScopeV1, now_ms: int, principal_id: str | None = None
) -> None:
    """Validate the current singleton grant without invoking its provider.

    Args:
        grant: Fresh resolved authority, obtained outside QuoteMem locks.
        scope: Exact canonical scope derived from the external request.
        now_ms: Trusted action time, rechecked inside the final commit mutex.
        principal_id: Previously admitted principal when reauthorizing a retry.

    Returns:
        None.

    Raises:
        QuoteMemError: If authority is expired, changes principal, or grants another scope.
    """
    if not grant.issued_at <= now_ms < grant.expires_at:
        raise QuoteMemError("authorization_expired", "Reception authorization must be renewed")
    if principal_id is not None and grant.principal_id != principal_id:
        raise QuoteMemError("scope_denied", "Reception principal changed during the operation")
    if (
        grant.requested_scope != scope
        or len(grant.scope_grants) != 1
        or grant.scope_grants[0].relation != "exact"
        or grant.scope_grants[0].scope_kind != scope.scope_kind
        or grant.scope_grants[0].scope_id != scope.scope_id
    ):
        raise QuoteMemError("scope_denied", "Reception requires one exact canonical scope grant")
