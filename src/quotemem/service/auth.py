"""Exact user-scope grants for an already authenticated benchmark job.

HTTP owns credential admission. The use case derives the requested scope from
the exact external user ID; this provider owns neither scope registration nor
epoch enrichment. It grants no product or ancestor-scope access.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import PrincipalId
from quotemem.core.query import ScopeV1
from quotemem.providers.auth import (
    AuthRequest,
    AuthResolutionRefusal,
    PrincipalGrant,
    RawScopeGrant,
)

_PRINCIPAL: TypeAdapter[str] = TypeAdapter(PrincipalId)


def _request_snapshot(request: object) -> AuthRequest:
    if not isinstance(request, AuthRequest):
        raise QuoteMemError("schema_invalid", "Benchmark authorization request has the wrong type")
    try:
        # Preserve copied unknown fields that a declared-field serializer omits.
        payload = dict(request.__dict__)
        payload.update(request.__pydantic_extra__ or {})
        scope = payload.get("requested_scope")
        if isinstance(scope, ScopeV1):
            scope_payload = dict(scope.__dict__)
            scope_payload.update(scope.__pydantic_extra__ or {})
            payload["requested_scope"] = scope_payload
        return AuthRequest.model_validate(payload, strict=True)
    except (ValidationError, TypeError, ValueError):
        raise QuoteMemError("schema_invalid",
                            "Benchmark authorization request is malformed") from None


@dataclass(frozen=True, slots=True)
class BenchExactUserAuth:
    """Bind one immutable internal principal to fresh singleton user grants.

    Args:
        principal_id: Existing validated bootstrap principal, never an HTTP secret
            or an identity inferred from user-controlled content.
        ttl_ms: Exact positive authorization duration, at most one hour.

    Configuration is immutable. Returned protocol DTOs are fresh detached models,
    not shared mutable authority. A grant does not prove registration, current
    consent, erasure state, or that its scope was derived from a raw user ID.
    """

    principal_id: PrincipalId
    ttl_ms: int = 60_000

    def __post_init__(self) -> None:
        """Validate configuration without loading keys or opening storage."""
        try:
            principal = _PRINCIPAL.validate_python(self.principal_id, strict=True)
        except ValidationError:
            raise QuoteMemError(
                "schema_invalid", "Benchmark principal configuration is invalid"
            ) from None
        if type(self.ttl_ms) is not int or not 1 <= self.ttl_ms <= 3_600_000:
            raise QuoteMemError("schema_invalid", "Benchmark grant duration is invalid")
        object.__setattr__(self, "principal_id", principal)

    def resolve(self, request: AuthRequest) -> PrincipalGrant:
        """Grant only the exact requested user scope at the trusted request time.

        Args:
            request: Strict request from the authenticated benchmark use case.

        Returns:
            A fresh epoch-free grant with exactly one exact user relationship.

        Raises:
            QuoteMemError: schema_invalid for malformed caller input.
            AuthResolutionRefusal: scope_denied for a valid non-user request.
        """
        bound = _request_snapshot(request)
        scope = bound.requested_scope
        if scope.scope_kind != "user":
            raise AuthResolutionRefusal("scope_denied", bound)
        return PrincipalGrant(
            principal_id=self.principal_id,
            requested_scope=scope,
            scope_grants=[
                RawScopeGrant(scope_kind="user", scope_id=scope.scope_id, relation="exact")
            ],
            issued_at=bound.now_ms,
            expires_at=bound.now_ms + self.ttl_ms,
        )
