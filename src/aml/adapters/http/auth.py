"""Deployment credentials live only at the HTTP boundary from plan.md §9."""

from __future__ import annotations

import hmac
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from aml.core.errors import AMLError


@dataclass(frozen=True)
class HTTPCredentials:
    """One explicitly selected authentication scheme and an unlogged secret.

    This verifies transport admission only. Application AuthProvider still owns
    principal resolution and exact scope authority. No unauthenticated deployment
    mode is enabled without the separately required platform authorization.

    Args:
        mode: The deployment's token, bearer, or x-api-key scheme.
        secret: Nonempty visible ASCII credential, excluded from representations.
    """

    mode: Literal["token", "bearer", "x-api-key"]
    secret: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ("token", "bearer", "x-api-key") or not self.secret:
            raise AMLError("schema_invalid", "HTTP authentication configuration is invalid")
        if any(not 0x21 <= ord(character) <= 0x7E for character in self.secret):
            raise AMLError("schema_invalid", "HTTP authentication configuration is invalid")

    def verify(self, headers: Sequence[tuple[bytes, bytes]]) -> None:
        """Admit exactly one matching credential header with constant-time comparison.

        Args:
            headers: Raw HTTP headers; duplicate or competing credential headers
                are rejected. Header names and HTTP authorization schemes are
                case-insensitive; credential bytes are compared exactly.

        Returns:
            None after admission. Credentials are never returned to the caller.

        Raises:
            AMLError: ``authentication_failed`` with the same safe explanation
                for missing, invalid, duplicated, or incorrectly selected modes.
        """
        supplied = [
            (name.lower(), value)
            for name, value in headers
            if name.lower() in (b"authorization", b"x-api-key")
        ]
        if len(supplied) == 1:
            name, value = supplied[0]
            if self.mode == "x-api-key":
                candidate = value if name == b"x-api-key" else b""
            else:
                scheme, separator, credential = value.partition(b" ")
                candidate = (
                    credential
                    if name == b"authorization"
                    and separator
                    and scheme.lower() == self.mode.encode("ascii")
                    else b""
                )
            if hmac.compare_digest(candidate, self.secret.encode("ascii")):
                return
        raise AMLError("authentication_failed", "HTTP authentication failed")
