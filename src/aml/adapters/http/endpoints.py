"""Validate public submission URLs separately from local replay targets (§9)."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit, urlunsplit

from aml.core.errors import AMLError


def _resolve(hostname: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(address[4][0])
            for address in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        )
    )


def _public_url(url: str, resolve: Callable[[str], Sequence[str]]) -> SplitResult:
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.hostname is None
            or parsed.fragment
            or any(ord(character) < 0x21 or ord(character) == 0x7F for character in url)
        ):
            raise ValueError("Invalid public endpoint")
        # Accessing port also rejects malformed and out-of-range port syntax.
        if parsed.port == 0:
            raise ValueError("Invalid public endpoint port")
        hostname = parsed.hostname
        # Require the ASCII authority actually used by HTTP clients. Python's
        # builtin IDNA codec and a client's IDNA implementation can resolve the
        # same Unicode spelling to different hosts. Explicit A-labels avoid
        # validating one authority while returning another to the consumer.
        if not hostname.isascii():
            raise ValueError("Public endpoint hostnames must use ASCII or explicit IDNA A-labels")
        try:
            addresses = (str(ipaddress.ip_address(hostname)),)
        except ValueError:
            addresses = resolve(hostname)
        parsed_addresses = tuple(ipaddress.ip_address(address) for address in addresses)
        if not parsed_addresses or any(
            not address.is_global or address.is_multicast for address in parsed_addresses
        ):
            raise ValueError("The endpoint does not resolve exclusively to public addresses")
        return parsed
    except (ValueError, UnicodeError, OSError):
        raise AMLError("invalid_request", "A public benchmark endpoint is invalid") from None


@dataclass(frozen=True)
class PublicEndpoints:
    """Validated submission descriptor; validation does not prove service readiness.

    Obtain this descriptor with from_urls. Local replay endpoints intentionally
    use their own configuration and cannot substitute for this public check.
    DNS observations apply only at validation time, not as a reachability claim.
    """

    add_url: str
    search_url: str
    health_url: str

    @classmethod
    def from_urls(
        cls,
        *,
        add_url: str,
        search_url: str,
        health_url: str | None = None,
        resolve: Callable[[str], Sequence[str]] = _resolve,
    ) -> PublicEndpoints:
        """Validate HTTPS public addresses and derive health only from Add's origin.

        Args:
            add_url: Public HTTPS Add endpoint without embedded credentials,
                using an ASCII hostname or explicit IDNA A-labels.
            search_url: Public HTTPS Search endpoint with the same authority rules.
            health_url: Explicit public health endpoint. If absent, only Add's
                scheme and authority supply the fallback /health URL.
            resolve: DNS resolution boundary; defaults to the system resolver.
                Component tests bind deterministic addresses without network calls.

        Returns:
            The exact configured URLs and the explicit or derived health URL.
            This validates addresses, not HTTPS reachability, unauthenticated
            health, public ownership, platform acceptance, or deployment duration.

        Raises:
            AMLError: ``invalid_request`` for malformed URLs, credentials,
                non-ASCII hostnames, non-HTTPS transports, unavailable DNS, or any
                nonpublic address.
        """
        add = _public_url(add_url, resolve)
        _public_url(search_url, resolve)
        health = (
            health_url
            if health_url is not None
            else urlunsplit((add.scheme, add.netloc, "/health", "", ""))
        )
        _public_url(health, resolve)
        return cls(add_url, search_url, health)
