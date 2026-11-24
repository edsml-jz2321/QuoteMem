"""BenchHttpContractV1 wire validation, without application effects.

The Add parser delegates size, identifier, and timestamp bounds to the pure
request authority after normalizing source roles, message fields and JSON representations.
Passing these contracts does not prove RowOp admission, storage, or visibility.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from decimal import Decimal, DecimalException
from typing import Annotated, Literal, Self, cast

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quotemem.core.add_outcome import AddOutcomeV1
from quotemem.core.envelope import ENVELOPE_HEADER
from quotemem.core.errors import QuoteMemError
from quotemem.core.ingestion_limits import MAX_ADD_MESSAGES
from quotemem.core.query import AsOfV1
from quotemem.core.technical_types import ExternalId
from quotemem.core.time_order import TimeOrderV1
from quotemem.ingestion.boundary import AddMessageV1, AddRequestV1, parse_add_request
from quotemem.store.metadata import StrictModel


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(_: str) -> object:
    raise ValueError("Nonfinite JSON number")


def _object_body(body: bytes, *, exact_numbers: bool = True) -> dict[str, object]:
    try:
        value: object = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
            parse_float=Decimal if exact_numbers else float,
        )
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return cast(dict[str, object], value)
    except (ValueError, RecursionError, DecimalException):
        raise QuoteMemError("invalid_request", "The HTTP body is not a valid JSON object") from None


def _integer(value: object) -> object:
    # Decimal retains the JSON number's exact value. Binary floating-point can
    # turn a nonintegral input into an integer or round a valid int64 out of range.
    if (
        isinstance(value, Decimal)
        and -(2**63) <= value <= 2**63 - 1
        and value == value.to_integral_value()
    ):
        return int(value)
    return value


def _source_text(value: object, *, nullable: bool = False) -> object:
    """Decode textual content parts without interpreting their metadata as authority.

    Null Add content is the empty-message representation. Text parts concatenate
    in source order, with no invented separators. Unsupported media and malformed
    parts remain invalid rather than silently losing source evidence. Only lone
    surrogate codepoints are replaced; valid text, including controls, is exact.
    """
    if value is None and nullable:
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for part in cast(list[object], value):
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                fields = cast(dict[str, object], part)
                text = fields.get("text")
                if fields.get("type") not in ("text", "input_text") or not isinstance(text, str):
                    raise QuoteMemError("invalid_request", "Content parts must contain text")
                parts.append(text)
            else:
                raise QuoteMemError("invalid_request", "Content parts must contain text")
        value = "".join(parts)
    if isinstance(value, str):
        return re.sub(r"[\ud800-\udfff]", "\ufffd", value)
    return value


def _unicode(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Text must contain valid Unicode scalar values") from None
    return value


# Envelope messages (``ENVELOPE_HEADER``) arrive on a synthetic clock (2024-01-01 plus one
# day per session and one minute per message), and the message's real date survives only as
# its last line, "YYYY-MM-DD" or "YYYY-MM-DD HH:MM".
_TRAILER_DATE = re.compile(r"\n([0-9]{4})-([0-9]{2})-([0-9]{2})(?: ([0-9]{2}):([0-9]{2}))?\Z")
_DAY_MS = 86_400_000
_MINUTE_MS = 60_000


def envelope_source_time(content: str, timestamp: object = None) -> int | None:
    """The source time an AMLB envelope message states in its date trailer.

    The trailer is read as UTC, the frame of every other source timestamp: a
    date alone is that day's midnight, ``HH:MM`` that minute. Messages sharing a
    trailer keep the order the synthetic clock gave them: its minute within the
    day is added as milliseconds, which never moves the stated minute or day.

    Args:
        content: The message's normalized content.
        timestamp: The message's own (synthetic) timestamp, if any.

    Returns:
        Unix milliseconds, or None unless the envelope header opens the content,
        the trailer alone forms its last line, and the trailer is a real
        calendar time.
    """
    if not ENVELOPE_HEADER.match(content):
        return None
    trailer = _TRAILER_DATE.search(content)
    if trailer is None:
        return None
    year, month, day, hour, minute = trailer.groups()
    try:
        moment = datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0),
                          tzinfo=UTC)
    except ValueError:
        return None
    position = 0
    if type(timestamp) is int and timestamp >= 0:
        position = min(timestamp % _DAY_MS // _MINUTE_MS, _MINUTE_MS - 1)
    return int(moment.timestamp()) * 1000 + position


class _EmptyAddRequestV1(AddRequestV1):
    """HTTP-only no-op; retain strict identifiers without entering source admission."""

    messages: Annotated[tuple[AddMessageV1, ...], Field(max_length=0, repr=False)]


def parse_add_body(body: bytes) -> AddRequestV1:
    """Normalize one HTTP Add payload before any application side effect.

    Messages retain only role, content and timestamp, plus explicitly versioned
    speaker metadata for user/assistant messages. All other message fields
    (including ``name``, unversioned ``speaker`` and SDK-only entity bindings)
    are ignored before admission: they neither affect digests nor grant speaker
    attribution. Other nonempty string roles are normalized to ``system`` source
    context; system messages never carry human-speaker metadata. Empty and
    whitespace-only content is dropped; retained content is never trimmed.
    An all-empty request is acknowledged as a no-op by the HTTP endpoint.
    An AMLB envelope message whose content ends in a strict date trailer takes
    that date as its timestamp (``envelope_source_time``); its content is kept
    exactly, and messages sharing a date keep their synthetic-clock order.
    Top-level fields remain strict. Duplicate JSON keys, even in discarded
    metadata, are rejected.

    Args:
        body: Original UTF-8 JSON bytes. Duplicate fields and non-JSON numbers
            are rejected. User, assistant and system roles retain their spelling.
            Explicit ``source_metadata_version=1`` additionally admits the
            caller's speaker and source participant metadata for user/assistant
            messages. This selector is independent of the HTTP request budget.

    Returns:
        The immutable Add request, retaining normalized content and message order.
        Null timestamps mean absence; exactly integral floats become integers.
        Text-part arrays concatenate in order, null content becomes empty, and
        lone surrogates become U+FFFD. Other source text stays exact.

    Raises:
        QuoteMemError: ``invalid_request`` for malformed JSON, unknown top-level fields,
            invalid types, empty/non-string roles, or the byte and field limits.
    """
    raw = _object_body(body)
    source_version = raw.get("source_metadata_version")
    if "source_metadata_version" in raw and (
        type(source_version) is not int or source_version != 1
    ):
        raise QuoteMemError("invalid_request", "The source metadata payload version is invalid")
    messages = raw.get("messages")
    if not isinstance(messages, list):
        raise QuoteMemError("invalid_request", "Add messages must be a JSON array")
    if len(cast(list[object], messages)) > MAX_ADD_MESSAGES:
        raise QuoteMemError("invalid_request", "Add message count exceeds its limit")
    normalized: list[dict[str, object]] = []
    for message in cast(list[object], messages):
        if not isinstance(message, dict):
            raise QuoteMemError("invalid_request", "Add messages must be JSON objects")
        supplied = cast(dict[str, object], message)
        if "content" in supplied:
            supplied["content"] = _source_text(supplied["content"], nullable=True)
        role = supplied.get("role")
        if not isinstance(role, str) or not role:
            raise QuoteMemError("invalid_request", "The HTTP message role is invalid")
        content = supplied.get("content")
        if isinstance(content, str) and not content.strip():
            continue
        if role not in ("user", "assistant", "system"):
            role = "system"
        allowed = {"role", "content", "timestamp"}
        if source_version == 1 and role in ("user", "assistant"):
            allowed |= {"speaker", "source_participant"}
        fields = {key: value for key, value in supplied.items() if key in allowed}
        fields["role"] = role
        if "timestamp" in fields:
            fields["timestamp"] = _integer(fields["timestamp"])
        if isinstance(content, str):
            stated = envelope_source_time(content, fields.get("timestamp"))
            if stated is not None:
                fields["timestamp"] = stated
        normalized.append(fields)
    raw["messages"] = normalized
    if not normalized:
        try:
            return _EmptyAddRequestV1.model_validate(raw)
        except ValidationError:
            raise QuoteMemError("invalid_request", "Add request fields are invalid") from None
    return parse_add_request(raw)


class _WireModel(StrictModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


class SearchRequestV1(_WireModel):
    """Exact Search wire input, with no session, principal, or scope selector."""

    query: Annotated[str, Field(min_length=1, repr=False)]
    user_id: Annotated[ExternalId, Field(repr=False)]
    top_k: Annotated[int, Field(ge=1, le=1000)]
    options: Annotated[tuple[str, ...] | None, Field(repr=False)] = None
    time_order: Annotated[TimeOrderV1 | None, Field(repr=False)] = None
    as_of: Annotated[AsOfV1 | None, Field(repr=False)] = None

    @field_validator("query", "user_id")
    @classmethod
    def valid_unicode(cls, value: str) -> str:
        """Reject unencodable text while retaining its exact representation."""
        return _unicode(value)

    @field_validator("options", mode="before")
    @classmethod
    def immutable_options(cls, value: object) -> object:
        """Admit only JSON arrays of strings or the optional null representation."""
        if isinstance(value, list):
            return tuple(cast(list[object], value))
        return value

    @field_validator("options")
    @classmethod
    def valid_options(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Check every option for valid UTF-8 without normalizing or dropping it."""
        if value is not None:
            for option in value:
                _unicode(option)
        return value

    @classmethod
    def from_body(
        cls, body: bytes, *, max_top_k: int, default_top_k: int | None = None,
        cap_top_k: int | None = None,
    ) -> Self:
        """Parse strict Search JSON using the application's active recall limit.

        Args:
            body: UTF-8 JSON request bytes; no credentials or request metadata.
            max_top_k: Trusted RecallPolicy.max_top_k, in the range 1 through 1000.
            default_top_k: The ``top_k`` of a request that omits it (the interactive
                Search profile); None requires the field.
            cap_top_k: Reduce a larger valid ``top_k`` to this bound instead of
                rejecting it (the interactive profile); None keeps ``max_top_k`` strict.

        Returns:
            Validated input with normalized textual query/options and exact user
            identifier bytes. Text-part queries concatenate in order; lone
            surrogates in query/options become U+FFFD.

        Raises:
            QuoteMemError: ``invalid_request`` for invalid client input, or
                ``service_unhealthy`` for an invalid trusted policy limit.
        """
        if type(max_top_k) is not int or not 1 <= max_top_k <= 1000:
            raise QuoteMemError("service_unhealthy", "The active recall limit is invalid")
        raw = _object_body(body)
        if "query" in raw:
            raw["query"] = _source_text(raw["query"])
        if isinstance(raw.get("options"), list):
            raw["options"] = [
                _source_text(option) if isinstance(option, str) else option
                for option in cast(list[object], raw["options"])
            ]
        if default_top_k is not None and "top_k" not in raw:
            raw["top_k"] = default_top_k
        raw["top_k"] = _integer(raw.get("top_k"))
        try:
            request = cls.model_validate(raw)
            if cap_top_k is not None and request.top_k > cap_top_k:
                request = request.model_copy(update={"top_k": cap_top_k})
            if request.top_k > max_top_k:
                raise ValueError("Requested result count exceeds recall policy")
            return request
        except (ValidationError, ValueError):
            raise QuoteMemError(
                "invalid_request", "Search request fields or limits are invalid"
            ) from None


class AddPendingResponseV1(_WireModel):
    """The 202 acknowledgement of an Add sent with ``Prefer: respond-async``.

    The request's source is durably saved, and the service completes it without the
    client: a disconnect, a restart or a crash after this acknowledgement loses nothing,
    since recovery resumes every saved request. The client reads the result by sending
    the exact same request again: a completed request answers 200 with its
    ``AddResponseV1`` replay, and a pending one answers this 202 again, never a second
    extraction.
    """

    success: Literal[True]
    status: Literal["pending"]
    request_id: ExternalId
    user_id: ExternalId
    session_id: ExternalId

    @classmethod
    def from_request(cls, request: AddRequestV1) -> Self:
        """Echo the accepted request's three identities.

        Args:
            request: The admitted request whose source is durably saved.

        Returns:
            The pending acknowledgement.
        """
        return cls(
            success=True, status="pending", request_id=request.request_id,
            user_id=request.user_id, session_id=request.session_id,
        )


class AddResponseV1(_WireModel):
    """Exact request echo with its six completed counts, built only after success.

    A request completed through explicit repair keeps this same shape; the V3
    completion inspection that names the request reports its ``recovery_count``.
    """

    success: Literal[True]
    request_id: ExternalId
    user_id: ExternalId
    session_id: ExternalId

    outcome: AddOutcomeV1

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def valid_unicode(cls, value: str) -> str:
        """Require exact encodable identifier strings in remote acknowledgements."""
        return _unicode(value)

    def acknowledges(self, request: AddRequestV1) -> bool:
        """Report whether this success echoes exactly the request's three identities.

        Args:
            request: The Add request this acknowledgement is claimed to answer.

        Returns:
            True only for equal user, session and request IDs. The six counts
            are the server's result for that request and are not compared here.
        """
        return (self.user_id, self.session_id, self.request_id) == (
            request.user_id, request.session_id, request.request_id,
        )

    @classmethod
    def from_request(cls, request: AddRequestV1, outcome: AddOutcomeV1) -> Self:
        """Build the exact echo after the caller has verified Add completion.

        Args:
            request: The admitted request whose application call succeeded.
            outcome: Actual immutable completion counts returned by the application.

        Returns:
            The exact request echo and its completed semantic item counts.
            Construction itself is not evidence of durable application success.
        """
        return cls(
            success=True,
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            outcome=outcome,
        )

    @classmethod
    def from_body(cls, body: bytes) -> Self:
        """Validate a remote Add acknowledgement without accepting duplicate fields.

        Args:
            body: Original response JSON bytes. The replay consumer separately
                compares the admitted echo with the original request identifiers.

        Returns:
            The strict success acknowledgement, with no extra response fields.

        Raises:
            QuoteMemError: ``service_unhealthy`` for malformed or invalid responses.
        """
        try:
            return cls.model_validate(_object_body(body, exact_numbers=False))
        except (QuoteMemError, ValidationError):
            raise QuoteMemError("service_unhealthy",
                                "The Add response contract is invalid") from None


_RFC3339 = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])\Z"
)


class SearchHitV1(_WireModel):
    """Only the public stable locator, rendered source, and optional annotations."""

    id: Annotated[str, Field(min_length=1)]
    content: Annotated[str, Field(min_length=1, repr=False)]
    score: Annotated[float | None, Field(allow_inf_nan=False)] = None
    created_at: str | None = None

    @field_validator("id", "content")
    @classmethod
    def valid_unicode(cls, value: str) -> str:
        """Retain valid UTF-8 values; response content is never normalized."""
        return _unicode(value)

    @field_validator("created_at")
    @classmethod
    def valid_created_at(cls, value: str | None) -> str | None:
        """Require a real RFC 3339 calendar time with an explicit offset."""
        if value is not None:
            if not _RFC3339.fullmatch(value):
                raise ValueError("Result time must use RFC 3339")
            datetime.fromisoformat(value.upper().replace("Z", "+00:00"))
        return value


class SearchResponseV1(_WireModel):
    """The only Search response: ``{data: [...]}`` in final rank order.

    An empty result serializes as ``{"data":[]}``. Hit identities are unique, so
    a consumer can bind every hit to exactly one memory or Evidence locator.
    """

    data: tuple[SearchHitV1, ...]

    @field_validator("data", mode="before")
    @classmethod
    def immutable_data(cls, value: object) -> object:
        """Accept decoded JSON arrays and retain their supplied final rank order."""
        if isinstance(value, list):
            return tuple(cast(list[object], value))
        return value

    @model_validator(mode="after")
    def unique_identities(self) -> Self:
        """Reject a response that names one result identity twice."""
        if len({hit.id for hit in self.data}) != len(self.data):
            raise ValueError("Search result identities must be unique")
        return self

    @classmethod
    def from_body(cls, body: bytes) -> Self:
        """Validate a complete remote Search response before delivering any hits.

        Args:
            body: Original response JSON bytes; duplicate fields and nonfinite
                scores fail. Optional annotations may be omitted, not null.

        Returns:
            The strict response preserving its received rank order. The caller
            must additionally check its size against the request's top_k.

        Raises:
            QuoteMemError: ``service_unhealthy`` for malformed or invalid responses.
        """
        try:
            return cls.model_validate(_search_response_body(body))
        except (QuoteMemError, ValueError):
            raise QuoteMemError("service_unhealthy",
                                "The Search response contract is invalid") from None


def _search_response_body(body: bytes) -> dict[str, object]:
    raw = _object_body(body, exact_numbers=False)
    data = raw.get("data")
    if isinstance(data, list):
        for hit in cast(list[object], data):
            if isinstance(hit, dict):
                fields = cast(dict[str, object], hit)
                if any(
                    name in fields and fields[name] is None for name in ("score", "created_at")
                ):
                    raise ValueError("Optional response annotations must not be null")
    return raw


def parse_search_response(body: bytes) -> SearchResponseV1:
    """Read one Search response; only the data-only V1 shape exists.

    Args:
        body: Original UTF-8 HTTP response bytes.

    Returns:
        The validated response with its ordered hits.

    Raises:
        QuoteMemError: ``service_unhealthy`` for malformed JSON, any field besides
            ``data`` (a ``schema_version`` included), repeated identities or
            invalid hit annotations.
    """
    return SearchResponseV1.from_body(body)
