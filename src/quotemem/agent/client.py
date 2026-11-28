"""Where ``quotemem memories``, ``repair`` and ``delete-user`` find the service.

They use the plugins' own settings (``integrations/core/config.mjs`` and ``token.mjs``),
so the commands reach the service the agents use:

- the URL: ``--url``, else ``QUOTEMEM_URL``, else ``http://127.0.0.1:8765``: a plain
  ``http://`` or ``https://`` URL (below) without credentials;
- the token: ``QUOTEMEM_TOKEN``, then ``QUOTEMEM_HTTP_AUTH_SECRET``, then the file
  ``QUOTEMEM_TOKEN_FILE`` names (a regular file of at most 4096 bytes; ``~`` and a
  relative path are taken from the home folder), then, for a service on this computer,
  its ``<data-dir>/token``. The data directory of a service elsewhere is not this
  computer's, so its token is never sent there.

A URL is accepted exactly when the plugins accept it (``config.mjs``): the WHATWG parser
of ``new URL`` reads it as an ``http``/``https`` URL without credentials, and its text,
trimmed as JavaScript trims it, matches `PLAIN_HTTP_URL` (``autostart.mjs``): after
``http://`` or ``https://`` a host of letters, digits and ``._~-`` or a bracketed IPv6
address, an optional ``:port`` of one to five digits, then ``/``, ``?``, ``#`` or the end.
So a backslash, a space, a tab, ``%`` or ``@`` in the authority, ``http:/host``,
``http:\\\\host`` and a port with letters are refused: the WHATWG parser and
``urlsplit`` (and httpx) would read another host (``http://127.0.0.1\\.evil.example``
is 127.0.0.1 to the first and ``127.0.0.1\\.evil.example`` to the others). A
``QUOTEMEM_URL`` or ``--url`` that is set but blank is refused too.

A host with an ``xn--`` label that httpx cannot encode (IDNA 2008, stricter than the
WHATWG parser's UTS 46: an emoji domain) is refused too, so the commands never fail later
in the HTTP client. A URL is also refused when the WHATWG parser would rewrite its host to
another text
(case aside; `canonical_host`): ``127.0.0.1.`` and ``127.1.`` (a final dot), ``127.1``,
``0x7f.0.0.1``, ``2130706433``, a leading zero in the last IPv4 octet, ``0``,
``[0:0:0:0:0:0:0:1]``,
``[::ffff:127.0.0.1]``. The plugins decide on the rewritten host (``127.0.0.1.`` is
127.0.0.1 to them, so they pass this computer's token with it), while urlsplit and httpx
connect to the text as written, and ``127.0.0.1.`` is a DNS name: the token would go to
whatever that name resolves to. So both sides read every accepted URL's host as the same
text, and httpx connects to that text.

A service is on this computer when the host of an accepted URL is ``localhost``,
``127.0.0.1`` or ``[::1]`` (any letter case), exactly as the plugins (``isLoopbackUrl``,
``token.mjs``) use that same set. It excludes other IPv4 loopback addresses, ``0.0.0.0``, ``[::]``
or ``localhost.``, which the WHATWG parser leaves as written.

Before the token goes to a URL that reaches this computer (`on_this_computer`: the hosts
above, any other ``127.x.y.z``, ``0.0.0.0`` and ``[::]``), `verify_listener` checks that
the program on its port proves it holds that token, as the plugins do
(``integrations/core/listener.mjs``): another user of the computer may have taken the port
first.

``npx quotemem <command>`` runs these commands with the settings of
``~/.quotemem/service.env`` (such as its data directory) and the plugins'
``QUOTEMEM_URL``, ``QUOTEMEM_USER_ID`` and ``QUOTEMEM_TOKEN_FILE`` (the environment, then
``~/.quotemem/env``). It passes the token a plugin would send to that URL: service.env's
token or its ``<data-dir>/token`` only for a service on this computer.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import re
import secrets
import ssl
import stat
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, cast
from urllib.parse import SplitResult, urlsplit

import httpcore
import httpx

from quotemem.agent.gateway import listener_proof, plain_address
from quotemem.agent.serve import DEFAULT_PORT
from quotemem.agent.settings import read_token
from quotemem.core.errors import QuoteMemError

DEFAULT_SERVICE_URL: Final = f"http://127.0.0.1:{DEFAULT_PORT}"
"""Where the plugins and ``quotemem serve`` meet unless told otherwise."""

LOOPBACK_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})
"""The host of a service on this computer as ``urlsplit`` writes it (lower case, no
brackets): the plugins' set. Each is its own `canonical_host`."""

MAX_TOKEN_FILE_BYTES: Final = 4096

PLAIN_HTTP_URL: Final = re.compile(
    r"https?://(?:\[[0-9a-f:.]+\]|[a-z0-9._~-]+)(?::\d{1,5})?(?:[/?#]|\Z)",
    re.IGNORECASE | re.ASCII,
)
"""The plugins' ``PLAIN_HTTP_URL`` (``integrations/core/autostart.mjs``), matched at the start.

``re.ASCII``: as in JavaScript, ``[a-z]`` with ``IGNORECASE`` and ``\\d`` match ASCII only
(Unicode matching would take the long s and the Kelvin sign for letters); ``\\Z`` is
JavaScript's ``$`` without the multiline flag."""

_AUTHORITY: Final = re.compile(r"[a-z][a-z0-9+.-]*:[/\\]*([^/\\?#]*)", re.IGNORECASE | re.ASCII)
"""The authority as the WHATWG parser reads it for http(s): after any slashes and
backslashes, up to a slash, a backslash, ``?`` or ``#``."""

_JS_WHITESPACE: Final = (
    "\t\n\x0b\x0c\r \xa0\u1680" + "".join(chr(code) for code in range(0x2000, 0x200B))
    + "\u2028\u2029\u202f\u205f\u3000\ufeff"
)
"""What JavaScript's ``String.prototype.trim`` removes, as the plugins trim QUOTEMEM_URL."""

_UNRESERVED: Final = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-._~")
"""`PLAIN_HTTP_URL`'s host characters (RFC 3986's unreserved ones), lower case, as
``urlsplit`` writes the host."""


def _ipv4_number(text: str) -> int | None:
    """One part of a WHATWG IPv4 address: decimal, ``0x`` hex or ``0`` octal; else None."""
    if not text:
        return None
    radix = 10
    if text[:2] in {"0x", "0X"}:
        text, radix = text[2:], 16
    elif len(text) > 1 and text[0] == "0":
        text, radix = text[1:], 8
    if not text:
        return 0
    if any(digit not in "0123456789abcdef"[:radix] for digit in text.lower()):
        return None
    return int(text, radix)


def _ipv4(host: str) -> str:
    """The WHATWG IPv4 parser: one to four parts, the last one filling what is left."""
    labels = host.split(".")
    if labels[-1] == "" and len(labels) > 1:
        labels.pop()
    numbers = [_ipv4_number(label) for label in labels]
    if len(numbers) > 4 or None in numbers:
        raise ValueError("not an IPv4 address")
    values = [number for number in numbers if number is not None]
    *leading, last = values
    if any(number > 255 for number in leading) or last >= 256 ** (5 - len(values)):
        raise ValueError("not an IPv4 address")
    return str(ipaddress.IPv4Address(
        last + sum(number * 256 ** (3 - index) for index, number in enumerate(leading))))


def _ipv6_text(address: ipaddress.IPv6Address) -> str:
    """The WHATWG IPv6 serializer: eight lower-case hex pieces, the first longest run of
    two or more zero pieces as ``::``, never a dotted IPv4 tail (``ipaddress`` may write
    ``::ffff:127.0.0.1`` where this writes ``::ffff:7f00:1``)."""
    pieces = [(int(address) >> (112 - 16 * index)) & 0xFFFF for index in range(8)]
    start, length, index = -1, 1, 0
    while index < 8:
        end = index
        while end < 8 and pieces[end] == 0:
            end += 1
        if end - index > length:
            start, length = index, end - index
        index = max(end, index + 1)
    if start < 0:
        return ":".join(f"{piece:x}" for piece in pieces)
    head = ":".join(f"{piece:x}" for piece in pieces[:start])
    tail = ":".join(f"{piece:x}" for piece in pieces[start + length:])
    return f"{head}::{tail}"


def canonical_host(parts: SplitResult) -> str:
    """The host as the WHATWG URL parser writes it, an IPv6 address without brackets.

    Raises:
        ValueError: For a host that parser refuses, and for a host character outside
            `PLAIN_HTTP_URL`'s, which it would decode or map.
    """
    host = parts.hostname or ""
    if parts.netloc.rpartition("@")[2].startswith("["):
        address = ipaddress.IPv6Address(host)  # an IPvFuture host fails here
        if address.scope_id is not None:
            raise ValueError("an IPv6 zone")
        return _ipv6_text(address)
    if not host or not all(character in _UNRESERVED for character in host):
        raise ValueError("a host character outside RFC 3986's unreserved ones")
    labels = host.split(".")
    last = labels[-2] if labels[-1] == "" and len(labels) > 1 else labels[-1]
    if last and (last.isdigit() or _ipv4_number(last) is not None):
        return _ipv4(host)
    return host


def _plain(url: str) -> SplitResult | None:
    """The parts of a URL whose text matches `PLAIN_HTTP_URL` and that the WHATWG parser
    would also accept (host and port), else None; its host may still be rewritten."""
    if PLAIN_HTTP_URL.match(url) is None:
        return None
    try:
        # urlsplit itself refuses an invalid bracketed host ("http://[1:2]").
        parts = urlsplit(url)
        _ = parts.port, canonical_host(parts)
    except ValueError:
        return None
    return parts


def service_url(given: str | None, environment: Mapping[str, str]) -> str:
    """``--url``, else ``QUOTEMEM_URL``, else `DEFAULT_SERVICE_URL`, without a final slash.

    Raises:
        QuoteMemError: ``schema_invalid`` for a URL the plugins refuse (module docstring); the
            message names the setting, never its value.
    """
    flag = given is not None
    name = "--url" if flag else "QUOTEMEM_URL"
    raw = given if given is not None else environment.get("QUOTEMEM_URL", "")
    url = raw.strip(_JS_WHITESPACE) if flag or raw else DEFAULT_SERVICE_URL
    authority = _AUTHORITY.match(url)
    if authority is not None and "@" in authority.group(1):
        raise QuoteMemError("schema_invalid", f"{name} must not contain credentials; put the "
                            "token in the file QUOTEMEM_TOKEN_FILE names instead")
    parts = _plain(url)
    if parts is None:
        raise QuoteMemError("schema_invalid", f"{name} is not a valid URL: after http:// or "
                            "https:// it needs a plain host name or address and an optional "
                            ":port, with no backslash, space or other special character")
    if canonical_host(parts) != parts.hostname:
        raise QuoteMemError("schema_invalid", f"{name} must write its host in full: an IPv4 "
                            "address as four decimal numbers without a final dot, an IPv6 "
                            "address in its shortest form")
    try:
        # httpx decodes an xn-- label with IDNA 2008 when it reads the host (each request
        # does), which refuses some names the WHATWG parser's UTS 46 accepts (an emoji
        # domain such as xn--ls8h.la).
        _ = httpx.URL(url).host
    except (httpx.InvalidURL, ValueError):  # the idna errors are UnicodeError, a ValueError
        raise QuoteMemError("schema_invalid", f"{name} has a host name the HTTP client cannot "
                            "use: an xn-- label must be a valid IDNA 2008 name") from None
    return url.rstrip("/")


def is_local(url: str) -> bool:
    """Whether an accepted service URL is written for this computer (module docstring);
    False for a URL `service_url` refuses."""
    parts = _plain(url)
    # Each loopback host is its own canonical_host, so a rewritten host is never one.
    return parts is not None and parts.hostname in LOOPBACK_HOSTS


def service_answers(
    url: str, *, timeout_seconds: float = 5.0, transport: httpx.BaseTransport | None = None,
) -> bool:
    """Whether anything answers HTTP at the service URL (``GET /health``, no token)."""
    try:
        with httpx.Client(base_url=url, timeout=timeout_seconds, trust_env=False,
                          transport=transport) as client:
            client.get("/health")
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return False
    except httpx.HTTPError:
        return True  # It took the connection, then stalled or broke off.
    return True


def service_serving(
    url: str, *, timeout_seconds: float = 5.0, transport: httpx.BaseTransport | None = None,
) -> bool:
    """Whether the service at the URL says it serves (``GET /agent/ready``, no token).

    False while it starts (it probes its providers or opens its store), and for anything
    else that answers there.
    """
    try:
        with httpx.Client(base_url=url, timeout=timeout_seconds, trust_env=False,
                          transport=transport) as client:
            response = client.get("/agent/ready")
        return response.status_code == 200 and response.json().get("ready") is True
    except (httpx.HTTPError, ValueError, AttributeError):
        return False


LISTENER_ROUTE: Final = "/agent/listener"
# The most of an answer to the listener check that is read (the plugins keep 4 KiB too).
LISTENER_ANSWER_BYTES: Final = 4096
# ``localhost``, ``localhost.`` and any name under it (``app.localhost``): names that always
# mean this computer (core/listener.mjs LOCALHOST_NAME).
_LOCALHOST_NAME: Final = re.compile(r"(?:[a-z0-9_-]+\.)*localhost\.?")


class _OneConnection(httpcore.NetworkBackend):
    """The network of a `ListenerClient`: one TCP connection, never a second.

    The connection the listener check proved is the only one its requests may use: once it
    closed, a request fails with ``service_unhealthy`` and nothing is sent, so a program that
    took the port meanwhile gets nothing. While ``deadline`` (a `time.monotonic` value) is
    set, every connect, read and write keeps to it, so a listener that trickles its status
    line or headers is cut off too, not just one that trickles its body.
    """

    def __init__(self) -> None:
        self._network = httpcore.SyncBackend()
        self._opened = False
        self.deadline: float | None = None

    def remaining(self, timeout: float | None) -> float | None:
        """``timeout`` cut to what is left before the deadline.

        Raises:
            httpcore.ReadTimeout: When the deadline has passed.
        """
        if self.deadline is None:
            return timeout
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise httpcore.ReadTimeout("the listener check took too long")
        return left if timeout is None else min(timeout, left)

    def connect_tcp(
        self, host: str, port: int, timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        if self._opened:
            raise QuoteMemError("service_unhealthy", "The connection the listener check proved "
                                "has closed; nothing was sent on another. Run the command again")
        self._opened = True
        stream = self._network.connect_tcp(host, port, self.remaining(timeout), local_address,
                                           socket_options)
        return _DeadlineStream(stream, self)

    def connect_unix_socket(
        self, path: str, timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        raise httpcore.ConnectError("a unix socket is not used")

    def sleep(self, seconds: float) -> None:
        self._network.sleep(seconds)


class _DeadlineStream(httpcore.NetworkStream):
    """A connection of `_OneConnection` whose every step keeps to its deadline."""

    def __init__(self, stream: httpcore.NetworkStream, network: _OneConnection) -> None:
        self._stream = stream
        self._network = network

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._stream.read(max_bytes, self._network.remaining(timeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._stream.write(buffer, self._network.remaining(timeout))

    def close(self) -> None:
        self._stream.close()

    def start_tls(
        self, ssl_context: ssl.SSLContext, server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        stream = self._stream.start_tls(ssl_context, server_hostname,
                                        self._network.remaining(timeout))
        return _DeadlineStream(stream, self._network)

    def get_extra_info(self, info: str) -> object:
        return self._stream.get_extra_info(info)


class _OneConnectionTransport(httpx.HTTPTransport):
    """httpx's transport over a pool of the one connection `_OneConnection` allows."""

    def __init__(self, network: _OneConnection) -> None:
        super().__init__(trust_env=False)
        self._pool.close()
        self._pool = httpcore.ConnectionPool(
            ssl_context=httpx.create_ssl_context(trust_env=False), max_connections=1,
            max_keepalive_connections=1, network_backend=network)


class ListenerClient(httpx.Client):
    """A client whose requests go only on the connection `verify_listener` proved.

    Pass it to `verify_listener` (``client=``), then send the request with it, to the URL
    the check returned: a program that takes the port between the check and the request
    gets nothing. With ``transport`` (tests) requests go there instead.
    """

    def __init__(self, *, timeout_seconds: float,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.connection = _OneConnection() if transport is None else None
        super().__init__(timeout=timeout_seconds, trust_env=False,
                         transport=transport if self.connection is None
                         else _OneConnectionTransport(self.connection))


def on_this_computer(url: str) -> bool:
    """Whether an http(s) URL's host reaches this computer: ``localhost`` (also ``localhost.``
    and a name under it, such as ``app.localhost``), an IPv4 loopback address (127.x.y.z,
    also IPv4-mapped), ``::1``, or the unspecified ``0.0.0.0`` and ``::``, the hosts the plugins
    check too."""
    parts = urlsplit(url)
    host = plain_address(parts.hostname or "")
    if parts.scheme not in ("http", "https") or not host:
        return False
    if _LOCALHOST_NAME.fullmatch(host):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def verify_listener(
    url: str, token: str, *, timeout_seconds: float = 5.0,
    transport: httpx.BaseTransport | None = None, client: ListenerClient | None = None,
) -> str:
    """Check, before the token goes to a service on this computer, that it holds the token.

    The service answers ``GET /agent/listener`` with a fresh challenge by `listener_proof`
    for the address and port it listens on. ``localhost`` and ``0.0.0.0`` mean 127.0.0.1,
    ``[::]`` means ::1, and the returned URL names that address, so the check and the
    request reach one address. Only a listener that proves the token passes: a service from
    before that route (QuoteMem 0.3.2 or older) is refused too. A URL for another computer
    (`on_this_computer`) is returned unchanged.

    The whole check, from connecting to the last byte of the answer, keeps to
    ``timeout_seconds``. With ``client`` (a `ListenerClient`) it goes on that client's one
    connection, and the request it guards must be sent with the same client; else on a
    connection of its own (``transport``: a test's).

    Returns:
        The URL to send the request to.

    Raises:
        httpx.ConnectError: When nothing listens (no request may be sent).
        QuoteMemError: ``authentication_failed`` when the program on the port is not shown
            to hold the token; ``service_unhealthy`` when it stalls or breaks off. The token
            was not sent.
    """
    if not on_this_computer(url):
        return url
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    address = plain_address(parts.hostname or "")
    if _LOCALHOST_NAME.fullmatch(address) or address in ("0.0.0.0", "::"):
        address = "::1" if address == "::" else "127.0.0.1"
    if address != (parts.hostname or "").lower():
        host = f"[{address}]" if ":" in address else address
        url = parts._replace(netloc=f"{host}:{port}" if parts.port else host).geturl()

    def refusal(why: str) -> QuoteMemError:
        return QuoteMemError("authentication_failed", "No token and no memory data were sent "
                             f"to {url}: the program on its port is not shown to be this "
                             f"user's QuoteMem service: {why}. `npx quotemem doctor` tells more")

    def stalled() -> QuoteMemError:
        return QuoteMemError("service_unhealthy", f"The program at {url} did not finish "
                             "answering the listener check; the token was not sent")

    challenge = secrets.token_hex(16)
    # The whole answer keeps to timeout_seconds and LISTENER_ANSWER_BYTES: a program that
    # trickles its status line, headers or a long body is cut off, as by the plugins' check.
    deadline = time.monotonic() + timeout_seconds
    body = b""
    own = None if client is not None else ListenerClient(timeout_seconds=timeout_seconds,
                                                         transport=transport)
    checker = cast(ListenerClient, client if client is not None else own)
    if checker.connection is not None:
        checker.connection.deadline = deadline
    try:
        with checker.stream(
                "GET", url + LISTENER_ROUTE, timeout=timeout_seconds,
                headers={"x-quotemem-challenge": challenge,
                         "accept-encoding": "identity"}) as response:
            status = response.status_code
            # iter_bytes, not iter_raw: an answer read already (as from a mock transport)
            # is replayed; `accept-encoding: identity` keeps it plain.
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > LISTENER_ANSWER_BYTES or time.monotonic() > deadline:
                    raise stalled()
    except httpx.ConnectError:
        raise  # nothing listens: no request is sent; the caller reports it as unreachable
    except httpx.HTTPError:
        # It took the connection, then stalled or broke off: as for any request.
        raise stalled() from None
    finally:
        if checker.connection is not None:
            checker.connection.deadline = None
        if own is not None:
            own.close()
    if time.monotonic() > deadline:
        raise stalled()
    if status == 200:
        try:
            parsed: object = json.loads(body.decode("utf-8"))
            proof = (cast("dict[str, object]", parsed).get("proof")
                     if isinstance(parsed, dict) else None)
        except (ValueError, UnicodeDecodeError):
            proof = None
        want = listener_proof(token, address, port, challenge)
        if isinstance(proof, str) and hmac.compare_digest(proof.encode(), want.encode()):
            return url
    if status in (200, 400):
        raise refusal("it answered the check without this user's token: another user's "
                      "QuoteMem or program, or a service with another token, holds the port")
    raise refusal(f"it does not answer the check ({status}): a QuoteMem service "
                  "from before the check (0.3.2 or older) answers this way. If it is your own, "
                  "restart it once with the current release: stop the `quotemem serve` you "
                  "started and start the current one (`uvx quotemem@latest serve`), or run "
                  "`npx quotemem stop` if an agent plugin started it")


def unreachable(url: str) -> QuoteMemError:
    """``service_unreachable`` for the service URL; for one on this computer, how to start it."""
    parts = urlsplit(url)
    hint = ""
    if is_local(url):
        port = parts.port or (443 if parts.scheme == "https" else 80)
        hint = (f"; start it with `quotemem serve"
                f"{'' if port == DEFAULT_PORT else f' --port {port}'}`")
    return QuoteMemError("service_unreachable", f"No service answers at {url} (--url, else "
                         f"QUOTEMEM_URL, else {DEFAULT_SERVICE_URL}){hint}")


def _visible_ascii(value: str) -> bool:
    return bool(value) and all(0x21 <= ord(character) <= 0x7E for character in value)


def _token_file(named: str) -> str:
    # Like the plugins' readTokenFile: only a regular file of at most MAX_TOKEN_FILE_BYTES,
    # checked before it is opened, so a FIFO or a device never blocks the command.
    path = Path.home() / Path(named).expanduser()
    content = b""
    try:
        metadata = os.stat(path)
        if stat.S_ISREG(metadata.st_mode) and metadata.st_size <= MAX_TOKEN_FILE_BYTES:
            with path.open("rb") as stream:
                content = stream.read(MAX_TOKEN_FILE_BYTES + 1)
    except OSError:
        pass
    token = content.decode("ascii", errors="replace").strip()
    if len(content) > MAX_TOKEN_FILE_BYTES or not _visible_ascii(token):
        raise QuoteMemError("authentication_failed", "QUOTEMEM_TOKEN_FILE cannot be read; it must "
                            "name a file that holds only the service's token")
    return token


def service_token(url: str, data_dir: Path, environment: Mapping[str, str]) -> str:
    """The token for the service at ``url``, in the plugins' order (module docstring).

    Raises:
        QuoteMemError: ``authentication_failed`` when there is none or the named token file
            cannot be read; ``schema_invalid`` for a token that is not visible ASCII.
    """
    if any(environment.get(name, "").strip()
           for name in ("QUOTEMEM_TOKEN", "QUOTEMEM_HTTP_AUTH_SECRET")):
        return read_token(data_dir, environment)
    named = environment.get("QUOTEMEM_TOKEN_FILE", "").strip()
    if named:
        return _token_file(named)
    if not is_local(url):
        raise QuoteMemError("authentication_failed", "No token for a service on another computer: "
                            "set QUOTEMEM_TOKEN_FILE to a private file holding its token")
    return read_token(data_dir, environment)
