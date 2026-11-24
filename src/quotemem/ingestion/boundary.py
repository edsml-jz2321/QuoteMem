"""Pure Add validation, byte coordinates, and secret preflight.

These functions neither authorize callers nor prove storage visibility. The
single Writer supplies the current authorized active session tail and rechecks
its identity before committing. A cleanup result describes work; it does not
claim that earlier Evidence or dependent content has been erased.
"""

from __future__ import annotations

import hashlib
import re
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal, Self, cast

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quotemem.core.admission import CanonicalRunAdmission, EmptyOptionsV1, admission_identity
from quotemem.core.entities import EntityBindingV1
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import ContentBindingDigest, PrincipalId, base64url, scope_id
from quotemem.core.ingestion_limits import (
    MAX_ADD_MESSAGES,
    MAX_ADD_UTF8_BYTES,
    MAX_EVIDENCE_PART_BYTES,
)
from quotemem.core.run_inputs import InputMessageV1, MessagesRequestPayloadV1, SessionBusinessKeyV1
from quotemem.core.source_participants import SourceMetadataVersion, SourceParticipantV1
from quotemem.core.technical_types import ExternalId
from quotemem.ingestion import credential_lexicon
from quotemem.store.metadata import StrictModel

MAX_EVIDENCE_BYTES = MAX_EVIDENCE_PART_BYTES
MAX_PRIOR_TAIL_BYTES = 512
Profile = Literal["product", "bench"]
SecretCategory = Literal["password", "api_key", "private_key", "session_token", "verification_code"]


def _utf8(value: str) -> bytes:
    """Reject unpaired surrogates without exposing input in the exception."""
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Text must contain valid Unicode scalar values") from None


class AddMessageV1(InputMessageV1):
    """An immutable SDK message; HTTP separately restricts roles and fields."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    content: Annotated[str, Field(min_length=1, repr=False)]

    @field_validator("content")
    @classmethod
    def valid_utf8(cls, value: str) -> str:
        """Preserve the original value after rejecting unencodable text."""
        _utf8(value)
        return value

    @field_validator("speaker")
    @classmethod
    def valid_speaker(cls, value: str | None) -> str | None:
        """Keep a supplied name exactly as written after rejecting unencodable text."""
        if value is not None:
            _utf8(value)
        return value


class AddRequestV1(StrictModel):
    """The exact Add payload; profile and authentication are trusted external inputs."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    user_id: ExternalId
    session_id: ExternalId
    request_id: ExternalId
    source_metadata_version: SourceMetadataVersion | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    messages: Annotated[
        tuple[AddMessageV1, ...], Field(min_length=1, max_length=MAX_ADD_MESSAGES, repr=False)
    ]

    @field_validator("user_id", "session_id", "request_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        """Validate exact UTF-8 IDs without trimming or normalization."""
        _utf8(value)
        return value

    @field_validator("messages", mode="before")
    @classmethod
    def immutable_messages(cls, value: object) -> object:
        """Materialize JSON arrays as immutable sequences; reject other iterables."""
        if isinstance(value, list):
            return tuple(cast(list[object], value))
        return value

    @model_validator(mode="after")
    def byte_budget(self) -> Self:
        """Check the decoded content byte budget, excluding transport JSON syntax."""
        if sum(len(_utf8(message.content)) for message in self.messages) > MAX_ADD_UTF8_BYTES:
            raise ValueError("Add message content exceeds its UTF-8 byte budget")
        if self.source_metadata_version is None and any(
            message.source_participant is not None for message in self.messages
        ):
            raise ValueError("Source participant metadata requires its explicit payload version")
        return self


def parse_add_request(raw: object) -> AddRequestV1:
    """Validate the complete Add DTO without persistence or payload-bearing errors.

    Args:
        raw: Decoded SDK request with only user_id, session_id, request_id, messages.

    Returns:
        An immutable validated request preserving every input string byte.

    Raises:
        QuoteMemError: ``invalid_request`` for unknown fields, types, bounds, or surrogates.
    """
    try:
        return AddRequestV1.model_validate(raw)
    except ValidationError:
        raise QuoteMemError("invalid_request",
                            "Add request fields or byte limits are invalid") from None


@dataclass(frozen=True)
class AddAdmissionBinding:
    """Bindings derived through the existing canonical admission authority."""

    admission: CanonicalRunAdmission = field(repr=False)
    idempotency_key: str
    admission_digest: ContentBindingDigest
    payload_digest: ContentBindingDigest
    request_key: ContentBindingDigest


def build_add_admission(
    request: AddRequestV1,
    *,
    principal_id: PrincipalId,
    profile: Profile,
    bind_content: Callable[[str, object], ContentBindingDigest],
) -> AddAdmissionBinding:
    """Bind exact Add bytes and the authenticated principal.

    Args:
        request: Validated Add request; no text normalization is performed.
        principal_id: Already authenticated principal, never inferred from user_id.
        profile: Trusted deployment profile, absent from the untrusted wire DTO.
        bind_content: Tenant Keyring binding function, using the existing purposes.

    Returns:
        Canonical admission and family, admission, payload, and ledger bindings.
        Transient grant issuance and expiry do not enter any input here.
    """
    target_scope = scope_id("user", request.user_id)
    admission = CanonicalRunAdmission(
        v=1,
        run_kind="session_ingestion",
        commit_mode="orchestrator",
        principal_id=principal_id,
        scope_kind="user",
        scope_id=target_scope,
        target_kind="session",
        target_locator=base64url(hashlib.sha256(_utf8(request.session_id)).digest())[:22],
        raw_business_key=SessionBusinessKeyV1(
            session_id=request.session_id, request_id=request.request_id
        ),
        request_payload=MessagesRequestPayloadV1(
            profile=profile,
            session_id=request.session_id,
            request_id=request.request_id,
            messages=[InputMessageV1(**message.model_dump()) for message in request.messages],
            source_metadata_version=request.source_metadata_version,
        ),
        options=EmptyOptionsV1(),
    )
    identity, admission_digest, payload_digest = admission_identity(admission, bind_content)
    request_key = bind_content(
        "add-request-key",
        {
            "scope_id": target_scope,
            "session_id": request.session_id,
            "request_id": request.request_id,
        },
    )
    return AddAdmissionBinding(admission, identity, admission_digest, payload_digest, request_key)


@dataclass(frozen=True)
class EvidencePart:
    """A proposed physical part with byte coordinates in its original message.

    ``entity_bindings`` are the SDK's bindings that lie inside this part, in
    original-message coordinates like ``byte_start`` and ``byte_end``.
    """

    ordinal: int
    part_no: int
    role: Literal["system", "user", "assistant", "tool"]
    source_observed_at: int | None
    speaker: str | None
    text: str = field(repr=False)
    byte_start: int
    byte_end: int
    entity_bindings: tuple[EntityBindingV1, ...] = ()
    source_participant: SourceParticipantV1 | None = None


def split_evidence(request: AddRequestV1) -> tuple[EvidencePart, ...]:
    """Split each message at codepoint boundaries within the fixed physical budget.

    Args:
        request: Validated request with original decoded message text.

    Returns:
        Parts ordered by request ordinal then part_no, each carrying its whole
        message's role, observation time and supplied speaker, and the entity
        bindings inside it. Offsets remain relative to the unsplit message, and
        rejoining a message's parts restores its bytes.

    Raises:
        QuoteMemError: ``invalid_request`` when an entity binding crosses a part
            boundary: each binding must lie inside one Evidence part.
    """
    parts: list[EvidencePart] = []
    for ordinal, message in enumerate(request.messages):
        content = _utf8(message.content)
        bindings = tuple(message.entity_bindings or ())
        start, part_no = 0, 0
        while start < len(content):
            end = min(start + MAX_EVIDENCE_BYTES, len(content))
            while end < len(content) and content[end] & 0xC0 == 0x80:
                end -= 1
            if any(binding.byte_start < end < binding.byte_end for binding in bindings):
                raise QuoteMemError(
                    "invalid_request", "Entity binding crosses a physical source part boundary"
                )
            parts.append(
                EvidencePart(
                    ordinal,
                    part_no,
                    message.role,
                    message.timestamp,
                    message.speaker,
                    content[start:end].decode("utf-8"),
                    start,
                    end,
                    entity_bindings=tuple(
                        binding for binding in bindings
                        if start <= binding.byte_start and binding.byte_end <= end
                    ),
                    source_participant=message.source_participant,
                )
            )
            start, part_no = end, part_no + 1
    return tuple(parts)


@dataclass(frozen=True)
class PriorEvidenceSpan:
    """An authorized active Evidence part in the Writer's stable session order.

    The Writer owns state, expiry, ordering, and durable-tail revalidation.
    byte_start/byte_end use the original message coordinates, as in Evidence.
    """

    evidence_id: str
    scope_id: str
    session_id: str
    text: str = field(repr=False)
    byte_start: int
    byte_end: int


@dataclass(frozen=True, order=True)
class MessageByteSpan:
    """A half-open byte range in one current request message."""

    ordinal: int
    byte_start: int
    byte_end: int


@dataclass(frozen=True, order=True)
class EvidenceByteSpan:
    """A matched old Evidence interval, without the matched source text."""

    evidence_id: str
    byte_start: int
    byte_end: int


@dataclass(frozen=True)
class SecretMatch:
    """A detector category and original-source coordinates, never secret bytes."""

    category: SecretCategory
    current_spans: tuple[MessageByteSpan, ...]
    prior_spans: tuple[EvidenceByteSpan, ...]


@dataclass(frozen=True)
class SecretPreflightResult:
    """An observation requiring Writer action, not an erasure completion receipt."""

    disposition: Literal["allowed", "reject_current", "cleanup_required"]
    matches: tuple[SecretMatch, ...] = ()
    scanned_prior_bytes: int = 0

    @property
    def affected_old_spans(self) -> tuple[EvidenceByteSpan, ...]:
        """Return unique sorted old ranges for authoritative cleanup planning."""
        return tuple(sorted({span for match in self.matches for span in match.prior_spans}))


@dataclass(frozen=True)
class _SourceRegion:
    start: int
    end: int
    source_start: int
    ordinal: int | None = None
    evidence_id: str | None = None


_LABEL_VALUE = (
    r"""["']?\s*(?:[:=\uff1a\uff1d]|\bis\b)\s*"""
    r"""(?:["'](?P<quoted>[^"'\r\n]+)["']|(?P<bare>[^\s"',;<>]+))"""
)
_LABELED: tuple[tuple[SecretCategory, re.Pattern[str]], ...] = (
    (
        "password",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:password|passwd|pwd|" + "\u5bc6\u7801" + r")" + _LABEL_VALUE, re.I
        ),
    ),
    (
        "api_key",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:api[ _-]?key|access[ _-]?key(?:[ _-]?secret)?|"
            r"client[ _-]?secret|secret[ _-]?key|" + "\u5bc6\u94a5" + r")" + _LABEL_VALUE,
            re.I,
        ),
    ),
    (
        "session_token",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:session[ _-]?token|auth[ _-]?token|"
            r"refresh[ _-]?token|access[ _-]?token)" + _LABEL_VALUE,
            re.I,
        ),
    ),
)
_PATTERNS: tuple[tuple[SecretCategory, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:(?:RSA|EC|DSA|OPENSSH|ENCRYPTED) )?PRIVATE KEY-----"
            r"[\s\S]*?(?:-----END (?:(?:RSA|EC|DSA|OPENSSH|ENCRYPTED) )?PRIVATE KEY-----|\Z)",
            re.I,
        ),
    ),
    (
        "api_key",
        re.compile(
            r"(?<![A-Za-z0-9_-])(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|"
            r"sk_(?:live|test)_[A-Za-z0-9]{16,}|(?:AKIA|ASIA)[A-Z0-9]{16}|"
            r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
            r"xox[baprs]-[A-Za-z0-9-]{10,})(?![A-Za-z0-9_-])"
        ),
    ),
    ("session_token", re.compile(r"(?<![A-Za-z0-9_])Bearer\s+[A-Za-z0-9._~+/-]{8,}=*", re.I)),
    (
        "session_token",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\."
            r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
        ),
    ),
    (
        "verification_code",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:otp|verification[ _-]?code|"
            r"one[ _-]?time[ _-]?(?:password|code)|"
            + "\u9a8c\u8bc1\u7801"
            + r")\s*(?:[:=\uff1a\uff1d]|\bis\b)?\s*[\"']?[0-9]{4,10}(?![0-9])",
            re.I,
        ),
    ),
)
_PLACEHOLDERS = frozenset({"[redacted]", "[removed]"})


def _matches(content: bytes) -> list[tuple[SecretCategory, int, int]]:
    # A 512-byte tail can start inside one UTF-8 character. Surrogate escape
    # preserves those boundary bytes; no replacement or normalization is used.
    text = content.decode("utf-8", errors="surrogateescape")
    offsets = array("I", [0])
    for character in text:
        offsets.append(offsets[-1] + len(character.encode("utf-8", errors="surrogateescape")))
    found: set[tuple[SecretCategory, int, int]] = set()
    for category, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            found.add((category, offsets[match.start()], offsets[match.end()]))
    labeled = _LABELED
    if credential_lexicon.has_label(text):
        labeled += credential_lexicon.FORMS
    for category, pattern in labeled:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            value = groups.get("quoted") or groups.get("bare") or ""
            if value.casefold() in _PLACEHOLDERS or re.fullmatch(r"\$\{[A-Z_][A-Z0-9_]*\}", value):
                continue
            found.add((category, offsets[match.start()], offsets[match.end()]))
    return sorted(found, key=lambda value: (value[1], value[2], value[0]))


def contains_recognized_credential(content: bytes) -> bool:
    """Apply the same code-owned detector during the final privacy residual scan.

    Args:
        content: Transient UTF-8 source or serialized content; it is never logged.

    Returns:
        Whether a supported credential signature matches. This does not certify
        arbitrary unrecognized text as safe or authorize skipping overlap checks.
    """
    return bool(_matches(content))


def recognized_credential_spans(content: bytes) -> list[tuple[SecretCategory, int, int]]:
    """The byte spans of ``content`` that the same code-owned detector recognizes.

    Args:
        content: Transient UTF-8 text; it is never logged.

    Returns:
        Each match as (category, start byte, end byte), sorted by position. Spans
        may overlap. As with ``contains_recognized_credential``, an empty result does
        not certify the text as free of secrets.
    """
    return _matches(content)


def _source_match(
    category: SecretCategory, start: int, end: int, regions: Sequence[_SourceRegion]
) -> SecretMatch:
    current: list[MessageByteSpan] = []
    prior: list[EvidenceByteSpan] = []
    for region in regions:
        low, high = max(start, region.start), min(end, region.end)
        if low >= high:
            continue
        first = region.source_start + low - region.start
        last = region.source_start + high - region.start
        if region.evidence_id is not None:
            prior.append(EvidenceByteSpan(region.evidence_id, first, last))
        elif region.ordinal is not None:
            current.append(MessageByteSpan(region.ordinal, first, last))
    return SecretMatch(category, tuple(current), tuple(prior))


def secret_preflight(
    request: AddRequestV1,
    *,
    profile: Profile,
    prior_evidence: Sequence[PriorEvidenceSpan] = (),
) -> SecretPreflightResult:
    """Inspect the complete boundary and bounded old tail without side effects.

    Args:
        request: Validated Add input. Content is concatenated in message order
            without inserting or deleting source bytes, including across messages.
        profile: Trusted product/bench selection; only bench bypasses rejection.
        prior_evidence: Already authorized active same-session parts in stable
            source order. Only the last 512 bytes participate; Writer revalidates
            that source selection before any provider or visibility operation.

    Returns:
        A closed disposition and exact original byte overlaps. Any old overlap
        requires cleanup even if a current-only match also exists. An allowed
        result means these deterministic signatures did not match; it does not
        certify arbitrary text as secret-free or prove a cleanup completed.

    Raises:
        QuoteMemError: ``schema_invalid`` for invalid profile or prior byte coordinates,
            or ``scope_denied`` for a selected tail from another scope/session.
    """
    if profile == "bench":
        return SecretPreflightResult("allowed")
    if profile != "product":
        raise QuoteMemError("schema_invalid", "Secret preflight profile is invalid")
    old_parts: list[tuple[PriorEvidenceSpan, bytes, int]] = []
    remaining = MAX_PRIOR_TAIL_BYTES
    seen: set[str] = set()
    expected_scope = scope_id("user", request.user_id)
    for source in reversed(prior_evidence):
        if remaining == 0:
            break
        if source.scope_id != expected_scope or source.session_id != request.session_id:
            raise QuoteMemError(
                "scope_denied", "Secret preflight tail is outside the authorized session"
            )
        try:
            content = _utf8(source.text)
        except ValueError:
            raise QuoteMemError("schema_invalid", "Prior Evidence contains invalid UTF-8") from None
        if (
            not content
            or source.evidence_id in seen
            or source.byte_start < 0
            or source.byte_end - source.byte_start != len(content)
        ):
            raise QuoteMemError("schema_invalid", "Prior Evidence byte identity is invalid")
        seen.add(source.evidence_id)
        used = min(remaining, len(content))
        old_parts.append((source, content[-used:], len(content) - used))
        remaining -= used
    old_parts.reverse()
    old_bytes = b"".join(content for _, content, _ in old_parts)
    old_regions: list[_SourceRegion] = []
    offset = 0
    for source, content, skipped in old_parts:
        old_regions.append(
            _SourceRegion(
                offset,
                offset + len(content),
                source.byte_start + skipped,
                evidence_id=source.evidence_id,
            )
        )
        offset += len(content)
    current_bytes = bytearray()
    current_regions: list[_SourceRegion] = []
    for ordinal, message in enumerate(request.messages):
        content = _utf8(message.content)
        start = len(current_bytes)
        current_bytes.extend(content)
        current_regions.append(_SourceRegion(start, len(current_bytes), 0, ordinal=ordinal))
    found = [
        _source_match(category, start, end, current_regions)
        for category, start, end in _matches(bytes(current_bytes))
    ]
    if old_bytes:
        regions = old_regions + [
            _SourceRegion(
                region.start + len(old_bytes),
                region.end + len(old_bytes),
                0,
                ordinal=region.ordinal,
            )
            for region in current_regions
        ]
        found.extend(
            match
            for category, start, end in _matches(old_bytes + current_bytes)
            if (match := _source_match(category, start, end, regions)).prior_spans
        )
    unique = tuple(dict.fromkeys(found))
    disposition = (
        "cleanup_required"
        if any(match.prior_spans for match in unique)
        else "reject_current"
        if unique
        else "allowed"
    )
    return SecretPreflightResult(disposition, unique, len(old_bytes))
