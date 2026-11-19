"""Tenant-local HMAC keys with purpose separation required by plan.md §11."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError

from aml.core.errors import AMLError
from aml.core.identity import ULID, ContentBindingDigest, PrincipalId, base64url, canonical_json
from aml.core.platform_files import open_private_file, sync_directory
from aml.core.query import AsOfV1, ScopeV1
from aml.store.metadata import OpaquePayloadSpec, StrictModel

_EVIDENCE_ID: TypeAdapter[str] = TypeAdapter(ULID)


class _RecallQueryBinding(StrictModel):
    """The five validated Search semantic fields in §5.8's flat HMAC envelope."""

    query: Annotated[str, Field(min_length=1)]
    options: Annotated[dict[str, JsonValue] | list[str], OpaquePayloadSpec(max_bytes=16384)]
    requested_scope: ScopeV1
    as_of: AsOfV1 | None
    top_k: Annotated[int, Field(ge=1, le=1000)]


class _IntentRunFamilyBinding(StrictModel):
    """The exact flat consumed-intent family projection from plan section 2."""

    run_kind: Literal["session_ingestion", "erasure_controller"]
    idempotency_key: Annotated[str, Field(pattern=r"^idem_[A-Za-z0-9_-]{43}$")]
    admission_digest: ContentBindingDigest
    principal_id: PrincipalId
    scope: ScopeV1 | None


class Keyring(Protocol):
    """Bind canonical content without exposing the tenant secret."""

    def digest(self, purpose: str, payload: object) -> ContentBindingDigest:
        """Return a purpose-separated HMAC-SHA256 content binding."""
        ...

    def principal_id(self, issuer: str, subject: str) -> PrincipalId:
        """Minimize an authentication identity using the exact tuple in plan.md §0."""
        ...


class ContentArrayBinding(Protocol):
    """Own one append-only canonical array; callers serialize access to its state."""

    @property
    def count(self) -> int:
        """Return the number of successfully appended records."""
        ...

    def append(self, record: object) -> None:
        """Append one canonical JSON record without retaining its content."""
        ...

    def append_canonical(self, encoded: str) -> None:
        """Append one record the caller already canonicalized, byte for byte."""
        ...

    def digest(self) -> ContentBindingDigest:
        """Return the current complete-array HMAC without consuming this prefix."""
        ...


class _LocalContentArrayBinding:
    """Retain only opaque HMAC state, record count and the fixed envelope suffix."""

    __slots__ = ("_state", "_count", "_suffix")

    def __init__(self, state: hmac.HMAC, suffix: bytes) -> None:
        self._state = state
        self._count = 0
        self._suffix = suffix

    @property
    def count(self) -> int:
        return self._count

    def append(self, record: object) -> None:
        try:
            encoded = canonical_json(record).encode("utf-8")
        except (TypeError, ValueError):
            raise AMLError("schema_invalid", "Canonical array record is invalid") from None
        self._absorb(encoded)

    def append_canonical(self, encoded: str) -> None:
        """Append a record whose canonical bytes the caller already holds.

        Callers that memoize a row's canonical JSON (a scope's frozen Evidence
        is streamed by several digests per finalize cycle, and every Add opens
        a new cycle) hand those exact bytes here instead of serializing the row
        again; the array is identical either way.
        """
        self._absorb(encoded.encode("utf-8"))

    def _absorb(self, encoded: bytes) -> None:
        following = self._state.copy()
        if self._count:
            following.update(b",")
        following.update(encoded)
        self._state = following
        self._count += 1

    def digest(self) -> ContentBindingDigest:
        complete = self._state.copy()
        complete.update(self._suffix)
        return ContentBindingDigest(base64url(complete.digest()))


class LocalKeyring:
    """Own a loaded 32-byte tenant key; key bytes never appear in repr or errors."""

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise AMLError("keyring_invalid", "Tenant HMAC key must contain 32 bytes")
        self._key = key

    @classmethod
    def create(cls, path: Path) -> LocalKeyring:
        """Create a new private key once and durably flush it.

        The caller owns creation of the private tenant directory. An existing
        file is never replaced, repaired, or silently regenerated.
        """
        key = os.urandom(32)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = open_private_file(path, flags)
        except OSError as error:
            raise AMLError("keyring_invalid", "Cannot create tenant HMAC key") from error
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(key)
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(path.parent)
        return cls.load(path)

    @classmethod
    def load(cls, path: Path) -> LocalKeyring:
        """Load a private regular key file, enforcing POSIX mode or Windows ACL."""
        try:
            descriptor = open_private_file(path, os.O_RDONLY)
        except OSError as error:
            raise AMLError("keyring_unavailable", "Tenant HMAC key is unavailable") from error
        except AMLError as error:
            if path.is_symlink():
                raise AMLError("keyring_unavailable", "Tenant HMAC key is unavailable") from error
            raise AMLError(
                "keyring_invalid", "Tenant HMAC key must be a private regular 0600/owner-ACL file"
            ) from error
        with os.fdopen(descriptor, "rb") as stream:
            key = stream.read(33)
        return cls(key)

    def array_binding(self, purpose: str) -> ContentArrayBinding:
        """Open a content-free incremental binding for an admitted array grammar.

        Args:
            purpose: A closed Finalize row, plan, original inventory, TTL
                disposition or current-header array role. Other protocols
                remain unchanged.

        Returns:
            A single-owner append stream whose every prefix equals digest of
            the same purpose and materialized JSON array. Models use the existing
            canonical model_dump representation, including explicit null fields.
            Neither the tenant key nor appended records are exposed or retained
            by the returned public interface.

        Raises:
            AMLError: If the purpose is malformed or is not an opened array role.

        The canonical bytes are {"payload":[<comma-separated records>],
        "purpose":<canonical purpose>,"v":1}. Snapshot digests copy the HMAC
        state, so they neither close the array nor consume the current prefix.
        """
        if type(purpose) is not str or not purpose or purpose != purpose.strip():
            raise AMLError("schema_invalid", "HMAC purpose must be a nonempty canonical string")
        if purpose not in {
            "finalize-plan-records", "finalize-source-inventory", "finalize-unit-inventory",
            "finalize-source-ttl", "finalize-current-source-headers",
        }:
            raise AMLError("unsupported_variant", "This purpose has no canonical array binding")
        suffix = b'],"purpose":' + canonical_json(purpose).encode("utf-8") + b',"v":1}'
        return _LocalContentArrayBinding(
            hmac.new(self._key, b'{"payload":[', hashlib.sha256), suffix,
        )

    def digest(self, purpose: str, payload: object) -> ContentBindingDigest:
        """Bind a nonempty purpose and canonical payload to this tenant key."""
        if not purpose or purpose != purpose.strip():
            raise AMLError("schema_invalid", "HMAC purpose must be a nonempty canonical string")
        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json", exclude_none=False)
        if purpose == "erasure-recovery-target":
            values = cast(list[object], payload) if isinstance(payload, list) else []
            if len(values) != 2 or type(values[0]) is not str or (
                values[0] not in {"memory", "scope"}
            ):
                raise AMLError("schema_invalid", "Erasure recovery requires one typed target tuple")
            try:
                identity = ContentBindingDigest(values[1])
            except (TypeError, ValueError):
                raise AMLError("schema_invalid", "Erasure recovery identity is invalid") from None
            encoded = canonical_json([values[0], identity]).encode("utf-8")
            return ContentBindingDigest(base64url(hmac.digest(self._key, encoded, "sha256")))
        if purpose == "erasure-worker-token":
            try:
                worker = _EVIDENCE_ID.validate_python(payload)
            except ValidationError:
                raise AMLError(
                    "schema_invalid", "Erasure worker token requires a Run identity",
                ) from None
            return ContentBindingDigest(base64url(hmac.digest(
                self._key, worker.encode("utf-8"), "sha256",
            )))
        projection: dict[str, object]
        if purpose == "recall-query":
            try:
                selected = _RecallQueryBinding.model_validate(payload)
                if len(canonical_json(selected.options).encode("utf-8")) > 16384:
                    raise ValueError("Recall options exceed their protocol bound")
            except (ValidationError, ValueError, TypeError):
                raise AMLError("schema_invalid", "Recall query binding is invalid") from None
            projection = {"v": 1, "purpose": purpose, **selected.model_dump(mode="json")}
        elif purpose == "intent-run-family":
            try:
                family = _IntentRunFamilyBinding.model_validate(payload)
            except ValidationError:
                raise AMLError(
                    "schema_invalid", "Consumed intent family binding is invalid",
                ) from None
            if (family.run_kind == "erasure_controller") != (family.scope is None):
                raise AMLError("schema_invalid", "Consumed intent family changed its scope role")
            projection = {"v": 1, "purpose": purpose, **family.model_dump(mode="json")}
        elif purpose == "evidence-invalidation":
            # This existing protocol purpose has a flat envelope in plan.md §5.9.
            fields = cast(dict[str, object], payload) if isinstance(payload, dict) else {}
            if set(fields) != {"evidence_id"}:
                raise AMLError(
                    "schema_invalid", "Evidence invalidation requires one typed identity"
                )
            try:
                evidence = _EVIDENCE_ID.validate_python(fields["evidence_id"])
            except ValidationError as error:
                raise AMLError(
                    "schema_invalid", "Evidence invalidation identity is invalid"
                ) from error
            projection = {"v": 1, "purpose": purpose, "evidence_id": evidence}
        else:
            projection = {"v": 1, "purpose": purpose, "payload": payload}
        value = canonical_json(projection).encode()
        digest = hmac.new(self._key, value, hashlib.sha256).digest()
        return ContentBindingDigest(base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii"))

    def principal_id(self, issuer: str, subject: str) -> PrincipalId:
        """Bind the issuer/subject tuple without the content-digest envelope."""
        if not issuer or not subject:
            raise AMLError("schema_invalid", "Authentication issuer and subject must be nonempty")
        payload = canonical_json([issuer, subject]).encode("utf-8")
        digest = hmac.new(self._key, payload, hashlib.sha256).digest()
        return "pr_" + base64url(digest)[:40]
