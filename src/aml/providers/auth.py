"""Epoch-free principal resolution; the authoritative Writer enriches grants."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from aml.core.errors import AMLError
from aml.core.identity import PrincipalId, canonical_json
from aml.core.query import ScopeV1 as RequestedScope
from aml.providers.keyring import Keyring
from aml.store.metadata import OpaqueIdSpec, StrictModel


class AuthRequest(StrictModel):
    """Transport-neutral request; credentials do not enter persistent DTOs."""

    requested_scope: RequestedScope
    now_ms: Annotated[int, Field(ge=0)]


class RawScopeGrant(RequestedScope):
    """A principal relationship without AML's current erase/consent epochs."""

    relation: Literal["exact", "ancestor"]


class PrincipalGrant(StrictModel):
    """Authority returned by AuthProvider before Writer epoch enrichment."""

    principal_id: Annotated[PrincipalId, OpaqueIdSpec(kind="local")]
    requested_scope: RequestedScope
    scope_grants: Annotated[list[RawScopeGrant], Field(min_length=1)]
    issued_at: Annotated[int, Field(ge=0)]
    expires_at: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def valid_interval(self) -> PrincipalGrant:
        """Require a positive authorization interval."""
        if self.expires_at <= self.issued_at:
            raise ValueError("Authorization expiry must follow issuance")
        return self


class AuthResolutionRefusal(AMLError):
    """Explicit negative outcome from an actual AuthProvider resolution call.

    External implementations raise this only for a definitive missing grant or
    an inability to establish authorization for the exact request. Other bugs,
    malformed responses and downstream application errors must not be wrapped
    as this outcome. It carries no credentials and grants no mutation authority.
    Public adapters retain the ordinary scope_denied/authorization_unavailable
    error code; internal consumers must independently bind the actual call.
    """

    def __init__(
        self,
        code: Literal["scope_denied", "authorization_unavailable"],
        request: AuthRequest,
    ) -> None:
        if code not in {"scope_denied", "authorization_unavailable"}:
            raise AMLError("schema_invalid", "Unsupported authorization refusal outcome")
        self._request_json = canonical_json(AuthRequest.model_validate(request.model_dump()))
        super().__init__(code, "The authorization provider did not grant the exact request")

    @property
    def request(self) -> AuthRequest:
        """Return a strict clone of the exact refused request."""
        return AuthRequest.model_validate_json(self._request_json)


class AuthProvider(Protocol):
    """Resolve identity and relationships, without reading or writing AML epochs."""

    def resolve(self, request: AuthRequest) -> PrincipalGrant:
        """Return exact authority or fail closed.

        Explicit refusal uses AuthResolutionRefusal bound to this request.
        Generic exceptions remain failures, not proof of an authorization
        decision for an internal state-reduction consumer.
        """
        ...


class SingleUserAuth:
    """Grant only explicitly installed exact scopes to one local principal."""

    def __init__(self, keyring: Keyring, scopes: list[RequestedScope], ttl_ms: int = 60_000):
        if not 1 <= ttl_ms <= 3_600_000:
            raise AMLError("schema_invalid", "Invalid local authorization configuration")
        self._principal_id = keyring.principal_id("AMLsolo", "single_user")
        self._scopes = {(scope.scope_kind, scope.scope_id) for scope in scopes}
        self._ttl_ms = ttl_ms

    @property
    def principal_id(self) -> str:
        """Expose the minimized local principal for authorized global bootstrap Runs."""
        return self._principal_id

    def resolve(self, request: AuthRequest) -> PrincipalGrant:
        """Resolve configured exact scope access at the caller's trusted clock time."""
        scope = request.requested_scope
        if (scope.scope_kind, scope.scope_id) not in self._scopes:
            raise AuthResolutionRefusal("scope_denied", request)
        return PrincipalGrant(
            principal_id=self._principal_id,
            requested_scope=scope,
            scope_grants=[RawScopeGrant(**scope.model_dump(), relation="exact")],
            issued_at=request.now_ms,
            expires_at=request.now_ms + self._ttl_ms,
        )
