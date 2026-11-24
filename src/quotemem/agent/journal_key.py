"""The key that seals a source journal's forgotten statements.

A journal's forget records hold the statements of forgotten memories, which a
rebuild needs to forget them again. They never appear in clear: each forget
record is sealed with AES-256-GCM (``cryptography``) under its own random 96-bit
nonce, with the record's user, time and statement count as associated data, so a
sealed record cannot be edited, or moved to another user, without failing
authentication.

The key is one of:

- a data directory key: 32 random bytes in ``<data dir>/journal.key`` (one
  base64 line, mode 0600), made by the first export and reused by later ones;
- a passphrase key: scrypt over a passphrase read from an environment variable,
  with a random salt and the cost parameters in the journal header, so a backup
  can be restored without the original data directory.

The journal names its key only by ``key_id``, a SHA-256 fingerprint of the key
under a fixed label, which a reader checks before it opens a sealed record. A
journal never holds the key, the passphrase or anything the key can be derived
from without the passphrase.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Final, Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from pydantic import ConfigDict, Field

from quotemem.core.errors import QuoteMemError
from quotemem.core.platform_files import open_private_file
from quotemem.store.metadata import StrictModel

KEY_FILE_NAME: Final = "journal.key"
"""The data directory key's file name inside a data directory."""
SCHEME: Final = "aes-256-gcm"
KEY_BYTES: Final = 32
NONCE_BYTES: Final = 12
SALT_BYTES: Final = 16
SCRYPT_N: Final = 2**15
SCRYPT_R: Final = 8
SCRYPT_P: Final = 1
MIN_PASSPHRASE_CHARS: Final = 8
_KEY_ID_LABEL: Final = b"quotemem-journal-key-v1\x00"


class _Model(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class JournalKdfV1(_Model):
    """How a passphrase key is derived: scrypt with these parameters."""

    name: Literal["scrypt"] = "scrypt"
    salt: Annotated[str, Field(min_length=1)]
    n: Annotated[int, Field(ge=2**14, le=2**20)] = SCRYPT_N
    r: Annotated[int, Field(ge=1, le=16)] = SCRYPT_R
    p: Annotated[int, Field(ge=1, le=4)] = SCRYPT_P


class JournalEncryptionV1(_Model):
    """The header's description of the sealing key; never the key itself."""

    scheme: Literal["aes-256-gcm"] = SCHEME
    key_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    kdf: JournalKdfV1 | None = None


class JournalSealedV1(_Model):
    """One sealed value: its nonce and the ciphertext with its tag, in base64."""

    nonce: Annotated[str, Field(min_length=1)]
    ciphertext: Annotated[str, Field(min_length=1)]


def _decode(value: str, what: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, binascii.Error):
        raise QuoteMemError("schema_invalid", f"The journal's {what} is not base64") from None


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


@dataclass(frozen=True)
class JournalKey:
    """A 256-bit sealing key and, for a passphrase key, how it was derived."""

    secret: bytes = field(repr=False)
    kdf: JournalKdfV1 | None = None

    def __post_init__(self) -> None:
        if len(self.secret) != KEY_BYTES:
            raise QuoteMemError("schema_invalid", "A journal key must be 32 bytes")

    @classmethod
    def generate(cls) -> JournalKey:
        """A new random key."""
        return cls(secrets.token_bytes(KEY_BYTES))

    @classmethod
    def from_passphrase(cls, passphrase: str, kdf: JournalKdfV1 | None = None) -> JournalKey:
        """The key ``passphrase`` derives under ``kdf`` (None: a new random salt).

        Raises:
            QuoteMemError: ``schema_invalid`` for a passphrase shorter than eight characters.
        """
        if len(passphrase) < MIN_PASSPHRASE_CHARS:
            raise QuoteMemError("schema_invalid",
                           f"A journal passphrase needs at least {MIN_PASSPHRASE_CHARS} characters")
        params = kdf or JournalKdfV1(salt=_encode(secrets.token_bytes(SALT_BYTES)))
        secret = Scrypt(
            salt=_decode(params.salt, "key salt"), length=KEY_BYTES, n=params.n, r=params.r,
            p=params.p,
        ).derive(passphrase.encode("utf-8"))
        return cls(secret, params)

    @property
    def key_id(self) -> str:
        """The key's public fingerprint."""
        return hashlib.sha256(_KEY_ID_LABEL + self.secret).hexdigest()[:32]

    @property
    def encryption(self) -> JournalEncryptionV1:
        """What a journal sealed with this key records about it."""
        return JournalEncryptionV1(key_id=self.key_id, kdf=self.kdf)

    def seal(self, plaintext: bytes, associated: bytes) -> JournalSealedV1:
        """``plaintext`` sealed under a fresh nonce, bound to ``associated``."""
        nonce = secrets.token_bytes(NONCE_BYTES)
        sealed = AESGCM(self.secret).encrypt(nonce, plaintext, associated)
        return JournalSealedV1(nonce=_encode(nonce), ciphertext=_encode(sealed))

    def open(self, sealed: JournalSealedV1, associated: bytes) -> bytes:
        """The plaintext of ``sealed``.

        Raises:
            QuoteMemError: ``parity_mismatch`` when it fails authentication (edited,
                moved to another record, or sealed under another key).
        """
        nonce = _decode(sealed.nonce, "nonce")
        if len(nonce) != NONCE_BYTES:
            raise QuoteMemError("schema_invalid", "A sealed journal value has a malformed nonce")
        try:
            return AESGCM(self.secret).decrypt(
                nonce, _decode(sealed.ciphertext, "ciphertext"), associated,
            )
        except InvalidTag:
            raise QuoteMemError(
                "parity_mismatch", "A sealed journal record failed authentication",
            ) from None


def _read_key_file(path: Path) -> JournalKey:
    descriptor = open_private_file(path, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read(4096)
    try:
        secret = base64.b64decode(raw.strip(), validate=True)
    except (ValueError, binascii.Error):
        secret = b""
    if len(secret) != KEY_BYTES:
        raise QuoteMemError("schema_invalid", "The journal key file does not hold a journal key")
    return JournalKey(secret)


def data_dir_key(data_dir: Path) -> tuple[JournalKey, Path]:
    """``data_dir``'s journal key and its file, creating it (mode 0600) when absent.

    Raises:
        QuoteMemError: ``schema_invalid`` when the file holds no key.
        OSError: When the file cannot be created or is not private.
    """
    path = data_dir / KEY_FILE_NAME
    if not (path.exists() or path.is_symlink()):
        write_key_file(path, JournalKey.generate())
    return _read_key_file(path), path


def write_key_file(path: Path, key: JournalKey) -> None:
    """Write ``key`` to the new file ``path`` (mode 0600); an existing file is kept.

    The file is staged and linked into place, so a reader never sees a partial key
    and a concurrent writer's key is never replaced.
    """
    staged = path.with_name(f".{path.name}.{os.getpid()}.partial")
    staged.unlink(missing_ok=True)
    descriptor = open_private_file(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_encode(key.secret).encode("ascii") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, path)
        except FileExistsError:
            pass
    finally:
        staged.unlink(missing_ok=True)


@dataclass(frozen=True)
class KeySource:
    """Where a reader finds a journal's key.

    Attributes:
        key_file: A key file, or a data directory holding ``journal.key``.
        passphrase: The passphrase of a passphrase key (secret).
        passphrase_variable: The environment variable it came from, for messages.
    """

    key_file: Path | None = None
    passphrase: str | None = field(default=None, repr=False)
    passphrase_variable: str | None = None


def passphrase_from(environment: Mapping[str, str], variable: str) -> str:
    """The passphrase in ``variable``; its value is never echoed.

    Raises:
        QuoteMemError: ``input_required`` when the variable is unset or empty.
    """
    value = environment.get(variable, "")
    if not value:
        raise QuoteMemError("input_required",
                            f"--passphrase-env names {variable}, which is unset or empty")
    return value


def key_for(encryption: JournalEncryptionV1, source: KeySource) -> JournalKey:
    """The key a journal sealed under ``encryption`` needs, from ``source``.

    Raises:
        QuoteMemError: ``input_required`` when ``source`` lacks the kind of key the
            journal needs; ``secret_rejected`` when the key or passphrase is not
            the journal's.
    """
    if encryption.kdf is not None:
        if source.passphrase is None:
            raise QuoteMemError(
                "input_required",
                "The journal's forgotten statements are sealed with a passphrase; pass "
                "--passphrase-env VAR, naming the variable that holds it",
            )
        key = JournalKey.from_passphrase(source.passphrase, encryption.kdf)
        if key.key_id != encryption.key_id:
            raise QuoteMemError(
                "secret_rejected",
                f"The passphrase in {source.passphrase_variable or 'the variable'} is not "
                "this journal's passphrase",
            )
        return key
    if source.key_file is None:
        raise QuoteMemError(
            "input_required",
            f"The journal's forgotten statements are sealed with key {encryption.key_id}; "
            "pass --key-file <the exported data dir>/journal.key (or that data directory)",
        )
    path = source.key_file.expanduser()
    key = _read_key_file(path / KEY_FILE_NAME if path.is_dir() else path)
    if key.key_id != encryption.key_id:
        raise QuoteMemError(
            "secret_rejected",
            f"The key file holds key {key.key_id}, not this journal's key "
            f"{encryption.key_id}",
        )
    return key
