"""Provider-call tracing seam for LLM chat loops.

``LLM.chat``, ``achat``, ``stream``, and ``astream`` accept a ``trace``
callback. When present, every provider-bound request made on behalf of that
call — the first round, each tool round, and each in-turn compaction summary
call — emits one ``request`` event immediately BEFORE provider I/O and one
terminal ``response`` or ``error`` event after it. A cache hit emits nothing:
no provider saw a request.

A ``request`` payload is the documented canonical serialization of the
provider-bound request in HeavenBase's normalized form. It captures the
messages, tool schemas, request options, and resolved route before
gateway-specific protocol transforms. :func:`request_fingerprint` hashes
this trace payload.

Trace callbacks are observers: an exception inside one is warned once per
exception type and never fails the traced call.
"""

from __future__ import annotations

__all__ = ["LLMTraceEvent", "TraceCallback", "canonical_request", "emit_event", "forward_trace", "request_fingerprint", "trace_state"]

import json
import numbers
import re
import warnings
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC, Set as SetABC
from dataclasses import dataclass, field, replace
from hashlib import sha256
from urllib.parse import parse_qsl, urlsplit

from heavenbase.utils import Any, Callable, Dict, List, Mapping, Optional, deepcopy
from heavenbase.utils.config.engine import redact_url

TraceCallback = Callable[["LLMTraceEvent"], None]

_TRACE_WARNED: set = set()
# These CLIENT_ARG_KEYS are semantic inputs that the gateway compiler consumes
# into a provider payload. They are safe scalar controls, not client wiring, so
# the canonical request must retain them for reconstruction and fingerprinting.
_PROVIDER_SEMANTIC_OPTION_KEYS = {"reasoning_exclude", "reasoning_max_tokens", "think"}
# Client construction/wiring and credential channels never reach a provider
# payload and must never reach a persisted trace. Header channels extend the
# gateway layer's own strip list.
_NON_REQUEST_OPTION_KEYS_EXTRA = {"default_headers", "extra_headers", "headers", "no_proxy"}
# Any other value stored under a secret-looking name, at any nesting depth
# (``extra_query``/``extra_body`` included, where some gateways and proxies
# accept keys), is replaced by this marker unless it cannot carry a
# credential: ``None``, a boolean, a number, or a list, tuple or set made only
# of those. So ``max_tokens``, ``budget_tokens`` and ``stop_token_ids`` survive
# and two requests that differ in them stay distinguishable, while text,
# bytes, byte arrays, mappings (whose keys are text) and mixed containers are
# replaced whole. An opaque non-JSON value whose text mentions a
# secret-looking name is replaced too, because its ``str()`` form may embed
# the credential it holds.
_REDACTED = "***"
_SECRET_NAME_PARTS = ("key", "token", "secret", "password", "passwd", "credential")
# Documented provider request parameters whose names look like credentials but
# whose values are request-shaping text sent as is: OpenAI's prompt-cache
# routing key.
_NON_SECRET_OPTION_NAMES = frozenset({"prompt_cache_key"})
# A URL in free text starts at a ``scheme://`` (the scheme being the run of
# scheme characters before ``://``, from its first ASCII letter) and ends at
# whitespace, a quote, ``<``, ``>``, a backtick, or where the next URL's
# scheme begins, so URLs joined by punctuation or nested in another URL's
# path or query are each redacted on their own.
_URL_SCHEME_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+.-")
_URL_STOP = re.compile(r"[\s'\"<>`]")
# Top-level options that carry model-visible content and bypass redaction.
_MODEL_VISIBLE_OPTION_KEYS = frozenset({"function_call", "functions", "response_format", "tool_choice"})
# JSON-schema keywords whose value maps property names to sub-schemas, and
# those whose value maps property names to lists of property names
# (``dependencies`` takes either form). Wherever a schema travels (Anthropic
# ``output_config``, vLLM ``guided_json``, a schema inside ``extra_body``),
# the keys of such a mapping held by a schema are property names that the
# model sees, not credential names, so ``api_key`` or ``token_count`` as a
# property keeps its sub-schema.
_SCHEMA_NAME_MAP_KEYWORDS = frozenset({"$defs", "definitions", "dependencies", "dependentSchemas", "patternProperties", "properties"})
_SCHEMA_NAME_LIST_KEYWORDS = frozenset({"dependencies", "dependentRequired"})
# A mapping is a schema when it is empty or every key is one of these
# keywords (JSON Schema draft-07 to 2020-12, plus the OpenAPI ``discriminator``,
# ``example``, ``externalDocs``, ``nullable`` and ``xml`` and the Gemini
# ``propertyOrdering``). A credential container uses some other key, so it
# stays subject to the credential-name rule even under a ``properties`` key.
_JSON_SCHEMA_KEYWORDS = frozenset(
    (
        "$anchor $comment $defs $dynamicAnchor $dynamicRef $id $recursiveAnchor $recursiveRef $ref $schema $vocabulary"
        " additionalItems additionalProperties allOf anyOf const contains contentEncoding contentMediaType contentSchema"
        " default definitions dependencies dependentRequired dependentSchemas deprecated description discriminator else"
        " enum example examples exclusiveMaximum exclusiveMinimum externalDocs format if items maxContains maxItems"
        " maxLength maxProperties maximum minContains minItems minLength minProperties minimum multipleOf not nullable"
        " oneOf pattern patternProperties prefixItems properties propertyNames propertyOrdering readOnly required then"
        " title type unevaluatedItems unevaluatedProperties uniqueItems writeOnly xml"
    ).split()
)


def _non_request_option_keys() -> set:
    """Return option keys excluded from canonical request payloads."""
    from .gateways._protocols import CLIENT_ARG_KEYS

    return (set(CLIENT_ARG_KEYS) - _PROVIDER_SEMANTIC_OPTION_KEYS) | _NON_REQUEST_OPTION_KEYS_EXTRA


@dataclass(frozen=True)
class LLMTraceEvent:
    """One provider-call trace event.

    Attributes:
        kind (str): ``"request"`` before provider I/O, then ``"response"`` on
            success or ``"error"`` on failure. A request whose terminal event
            never arrives was interrupted mid-flight.
        call (str): What made the call: ``"round"`` for the main chat loop,
            ``"turn_compaction"`` for an in-turn compaction summary call, or a
            caller-chosen tag when events are forwarded from a sub-call via
            :func:`forward_trace`.
        seq (int): Monotonic sequence number across all events of one traced
            top-level call, shared with forwarded sub-calls, pairing each
            terminal event with its request.
        payload (Dict[str, Any]): For ``request``, the canonical request from
            :func:`canonical_request`; for ``response``, ``{"message",
            "usage", "served_model", "elapsed"}``; for ``error``,
            ``{"error"}``.
    """

    kind: str
    call: str
    seq: int
    payload: Dict[str, Any] = field(default_factory=dict)


def trace_state(emit: TraceCallback) -> Dict[str, Any]:
    """Return the mutable per-call trace state threaded through chat contexts.

    Args:
        emit (TraceCallback): Caller-supplied event observer.

    Returns:
        Dict[str, Any]: ``{"emit", "seq"}`` state shared by every provider
        call (and forwarded sub-call) of one traced top-level call.
    """
    return {"emit": emit, "seq": 0}


def emit_event(state: Mapping[str, Any], event: LLMTraceEvent) -> None:
    """Deliver one event to the trace callback without failing the call.

    A raising callback is an observability defect, not a chat failure: the
    exception is converted to a ``UserWarning`` once per exception type and
    otherwise swallowed.

    Args:
        state (Mapping[str, Any]): State from :func:`trace_state`.
        event (LLMTraceEvent): Event to deliver.
    """
    try:
        state["emit"](event)
    except Exception as error:
        marker = type(error).__name__
        if marker not in _TRACE_WARNED:
            _TRACE_WARNED.add(marker)
            try:
                warnings.warn(f"LLM trace callback failed ({marker}: {error}); tracing continues best-effort", UserWarning, stacklevel=2)
            except Exception:
                pass  # warnings-as-errors must not turn an observer defect into a chat failure


def forward_trace(state: Dict[str, Any], call: Optional[str] = None) -> TraceCallback:
    """Return a callback that forwards a sub-call's events into ``state``.

    A compaction summary (or any caller-made helper call) runs as its own
    ``chat`` invocation with its own inner sequence numbers. The forwarder
    renumbers each event on the outer shared counter — and re-tags it with
    ``call`` when one is given — so one traced conversation keeps a single
    monotonic event order.

    Args:
        state (Dict[str, Any]): Outer state from :func:`trace_state`.
        call (str | None): Tag applied to every forwarded event; ``None``
            keeps each event's own tag. Defaults to ``None``.

    Returns:
        TraceCallback: Callback to pass as the sub-call's ``trace``.
    """
    remap: Dict[int, int] = {}

    def _forward(event: LLMTraceEvent) -> None:
        if event.kind == "request":
            remap[event.seq] = state["seq"]
            state["seq"] += 1
        emit_event(state, replace(event, call=event.call if call is None else call, seq=remap.get(event.seq, event.seq)))

    return _forward


def _secret_name(name: Any) -> bool:
    """Return whether an option name looks like it carries a credential.

    Matches names containing ``key``, ``token``, ``secret``, ``password``,
    ``passwd`` or ``credential``, and names with a word starting with
    ``auth`` (``authorization``, ``auth_token``, ``x-auth``), case-insensitively.
    """
    lowered = str(name).lower()
    if any(part in lowered for part in _SECRET_NAME_PARTS):
        return True
    return any(word.startswith("auth") for word in re.split(r"[^a-z0-9]+", lowered))


def _redact_one_url(url: str) -> str:
    """Redact user info and secret-looking query values in one URL token.

    User info normally ends at the last ``@`` of the authority, which ends at
    the first ``/``, ``?`` or ``#``. A password holding one of those unescaped
    (``postgres://user:pa/ss@host/db``) would end the authority early and leave
    the password in the path, so when the authority holds a ``:`` but no
    ``@`` and an ``@`` follows later in the token, everything up to the last
    ``@`` of the token is redacted as user info; a password that also holds an
    ``@`` after such a mark (``postgres://user:a/b@c@host/db``) is then found
    whole. A ``host:port`` authority followed by an ``@`` anywhere later, in
    the path, query or fragment (``https://host:8080/p?email=a@b.com``), or a
    password like the first followed by an ``@`` in the path, is redacted up to
    that last ``@`` too (``https://***:***@b.com``): these cannot be told
    apart, and the secret is the case kept safe. User info that holds a ``/``
    before its ``:`` (``postgres://us/er:pass@host/db``) also ends at the last
    ``@`` before the query and fragment, whenever a ``:`` comes before that
    ``@``; so does a path holding a ``:`` and then an ``@`` (``https://h/a:b@c``),
    which cannot be told apart. When the authority already holds user info, it
    is redacted and its host kept, unless a ``:`` and then an ``@`` follow in the
    path (``postgres://us@er/x:pass@host/db``), so an ``@`` in an ordinary path
    keeps the host (``https://user:token@registry/@scope/pkg``). An ``@`` with no
    ``:`` before it is a path or query value (``https://h/@name``). Two cases are
    not fully found: a user name with no password that holds an unescaped ``/``,
    ``?`` or ``#``, and a password that holds an ``@`` and then a ``/`` with no
    ``:`` after it (``postgres://user:p@ss/word@host/db`` keeps ``ss/word``), which
    cannot be told from a path ``@`` after well-formed user info.
    """
    try:
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            return url
        authority = len(parts.scheme) + 3  # the token starts with ``scheme://``
        if not parts.netloc.startswith("["):
            # An "@" after the parsed authority, before the query and fragment, ends user info that
            # held a "/" (``us/er:pass@``) when a ":" comes before it. When the authority already
            # holds user info, its host is kept unless a ":" also comes after it (``us@er/x:pass@``),
            # so an "@" in an ordinary path (``user:token@registry/@scope/pkg``) keeps the host.
            end = authority + len(parts.netloc)
            marks = [index for index in (url.find("?", authority), url.find("#", authority)) if index != -1]
            at = url.rfind("@", end, min(marks, default=len(url)))
            has_user_info = parts.username is not None or parts.password is not None
            if at != -1 and ":" in url[end if has_user_info else authority : at]:
                url = f"{url[:authority]}{_REDACTED}:{_REDACTED}{url[at:]}"
                parts = urlsplit(url)
        if parts.username is None and ":" in parts.netloc and not parts.netloc.startswith("["):
            at = url.rfind("@", authority + len(parts.netloc))
            if at != -1:
                url = f"{url[:authority]}{_REDACTED}:{_REDACTED}{url[at:]}"
                parts = urlsplit(url)
        names = [name for name, _ in parse_qsl(parts.query, keep_blank_values=True) if _secret_name(name)]
        if not names and parts.username is None and parts.password is None:
            return url
        return redact_url(url, replacement=_REDACTED, query_names=names)
    except ValueError:
        return _REDACTED  # a URL-shaped token that will not parse cleanly is dropped rather than kept with its secrets


def _url_starts(value: str) -> List[int]:
    """Return the start offset of every ``scheme://`` URL in ``value``.

    Each ``://`` is found once and its scheme by one backward scan that stops
    at the previous non-scheme character (``/`` at the latest), so the scans
    never overlap and the whole search is linear in ``len(value)``.
    """
    starts: List[int] = []
    sep = value.find("://")
    while sep != -1:
        begin = sep
        while begin > 0 and value[begin - 1] in _URL_SCHEME_CHARS:
            begin -= 1
        while begin < sep and not (value[begin].isascii() and value[begin].isalpha()):
            begin += 1  # a scheme starts with a letter
        if begin < sep:
            starts.append(begin)
        sep = value.find("://", sep + 3)
    return starts


def _redact_url_text(value: str) -> str:
    """Redact every URL embedded in ``value``, whatever text surrounds it.

    A URL is a ``scheme://`` run up to whitespace, a quote, ``<``, ``>``, a
    backtick or the next URL's scheme, so a URL inside a ``repr`` such as
    ``Endpoint(url='...')`` or ``b'...'``, one joined to another by ``,``,
    ``;`` or ``|``, and one nested in another URL's path or query are each
    found. Each URL of any scheme loses its user info (``user:password@``
    becomes ``***:***@``) and the values of secret-looking query parameters;
    ``_redact_one_url`` says which user info it finds.
    """
    if "://" not in value:
        return value
    starts = _url_starts(value)
    if not starts:
        return value
    pieces = [value[: starts[0]]]
    for index, start in enumerate(starts):
        limit = starts[index + 1] if index + 1 < len(starts) else len(value)
        stop = _URL_STOP.search(value, start, limit)
        end = stop.start() if stop else limit
        pieces.append(_redact_one_url(value[start:end]))
        pieces.append(value[end:limit])
    return "".join(pieces)


def _subschema(value: Any) -> bool:
    """Return whether ``value`` has the shape of a JSON schema.

    A schema is a boolean, or a mapping that is empty or whose every key is a
    JSON-schema keyword (``type``, ``$ref``, ``anyOf``, ``description``, ...).
    """
    if isinstance(value, bool):
        return True
    return isinstance(value, MappingABC) and all(isinstance(key, str) and key in _JSON_SCHEMA_KEYWORDS for key in value)


def _property_name(keyword: Any, value: Any) -> bool:
    """Return whether a key of the ``keyword`` mapping held by a schema, valued ``value``, is a property name.

    It is under a keyword that maps property names to schemas when ``value``
    is a schema, and under one that maps them to lists of property names when
    ``value`` is a list or tuple of strings.
    """
    if keyword in _SCHEMA_NAME_MAP_KEYWORDS and _subschema(value):
        return True
    return keyword in _SCHEMA_NAME_LIST_KEYWORDS and isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)


def _credential_safe(value: Any) -> bool:
    """Return whether a value under a credential-looking name may be kept.

    Kept are the values that cannot carry a credential: ``None``, booleans,
    numbers, and lists, tuples or sets made only of those (``stop_token_ids``).
    Text, bytes, byte arrays (``bytearray``, ``memoryview``, ``array.array``),
    mappings (whose keys are text), mixed containers and opaque objects are
    not.
    """
    if value is None or isinstance(value, numbers.Number):
        return True
    if isinstance(value, (list, tuple, SetABC)):
        return all(item is None or isinstance(item, numbers.Number) for item in value)
    return False


def _sorted_json(value: Any) -> str:
    """Serialize with object keys sorted after the JSON encoder converts them to strings."""

    def encode(item: Any) -> Any:
        return dict(item) if isinstance(item, MappingABC) else str(item)

    normalized = json.loads(json.dumps(value, default=encode))
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _canonical_option(name: Any, value: Any, *, redact: bool, _path: frozenset = frozenset(), _in_schema: bool = False) -> Any:
    """Return one option value in JSON form, redacted at every depth when asked.

    Redaction runs on the original value before any conversion, so it
    reaches values inside every container type and the ``str()`` text of
    non-JSON leaves.
    Mappings (any ``Mapping``) become dicts with JSON-safe keys; lists,
    tuples and other non-text sequences become lists; sets become lists in a
    stable order; a container that contains itself is cut at the repeat with
    ``"<cycle>"``. Every other leaf is kept when JSON-serializable, else
    replaced by its ``str()`` form. When ``redact`` is true:

    - a value under a credential-looking name (other than the documented
      non-secret request parameters) is kept only when :func:`_credential_safe`
      accepts it, and is ``"***"`` otherwise;
    - a key of a JSON-schema ``properties``-like mapping held by a schema is a
      property name, never a credential name, when :func:`_property_name`
      accepts its value;
    - every other string, and the ``str()`` text of every other opaque leaf,
      loses the user info and secret-looking query values of each URL it
      holds, and an opaque leaf whose text mentions a credential-looking
      name is ``"***"``.
    """
    if redact and _secret_name(name) and name not in _NON_SECRET_OPTION_NAMES and not _credential_safe(value):
        return _REDACTED
    if isinstance(value, str):
        return _redact_url_text(value) if redact else value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    container = isinstance(value, (MappingABC, SetABC)) or (isinstance(value, SequenceABC) and not isinstance(value, (bytes, bytearray, memoryview)))
    if container:
        if id(value) in _path:
            return "<cycle>"
        path = _path | {id(value)}
        if isinstance(value, MappingABC):
            names_properties = redact and _in_schema and isinstance(name, str)
            is_schema = redact and _subschema(value)
            out: Dict[Any, Any] = {}
            for key, item in value.items():
                json_key = key if key is None or isinstance(key, (str, int, float, bool)) else str(key)
                child_name = "" if names_properties and _property_name(name, item) else json_key  # a schema property name is not a credential name
                out[json_key] = _canonical_option(child_name, item, redact=redact, _path=path, _in_schema=is_schema)
            return out
        items = [_canonical_option("", item, redact=redact, _path=path) for item in value]
        if isinstance(value, SetABC):
            items.sort(key=_sorted_json)
        return items
    try:
        json.dumps(value)
        return deepcopy(value)
    except (TypeError, ValueError):
        pass
    try:
        text = str(value)
    except Exception:
        text = f"<{type(value).__name__}>"  # an unprintable value must not fail the traced call
    if not redact:
        return text
    if _secret_name(text):
        return _REDACTED  # an opaque object's text may embed the credential it holds
    return _redact_url_text(text)


def canonical_request(messages: List[Dict[str, Any]], cfg: Mapping[str, Any], spec: Any, streamable: Optional[bool] = None) -> Dict[str, Any]:
    """Return the canonical serialization of one provider-bound request.

    This is HeavenBase's normalized request form, captured before
    gateway-native compilation. ``options`` carries request-shaping parameters.
    Callable values and client wiring such as API keys, auth tokens, base
    URLs, proxies, mocks, and headers are excluded. The top-level
    ``response_format``, ``tool_choice``, ``functions`` and
    ``function_call`` options are model-visible content, like messages and
    tool schemas, and bypass trace redaction. In every other option, at any
    nesting depth including inside ``extra_query`` and ``extra_body``, a value
    under a name that looks like a credential (``*key*``, ``*token*``,
    ``*secret*``, ``*password*``, ``*passwd*``, ``*credential*``, ``auth*``) is replaced by
    ``"***"`` unless it is null, a boolean, a number, or a list, tuple or set
    made only of those, so ``max_tokens`` and ``stop_token_ids`` survive while
    text, bytes, byte arrays, mappings and mixed containers are replaced
    whole. OpenAI's ``prompt_cache_key`` is a request parameter, not a
    credential, and is kept. A schema here is a boolean, or a mapping that is
    empty or whose every key is a JSON-schema keyword (``type``, ``$ref``,
    ``anyOf``, ``description``, OpenAPI's ``example``, ...). In a schema, a
    key of a ``properties``, ``patternProperties``, ``$defs``,
    ``definitions``, ``dependentSchemas`` or ``dependencies`` mapping whose
    value is a schema, or of a ``dependentRequired`` or ``dependencies``
    mapping whose value is a list of strings, is a property name, not a
    credential name, so a schema sent in ``output_config``, ``guided_json`` or
    ``extra_body`` keeps properties such as ``api_key`` or ``token_count``. In those options,
    every URL inside a string, or inside the ``str()`` text of a non-JSON
    value, loses its user info and secret-looking query values, and an
    opaque non-JSON value whose text mentions a credential-looking name is
    stored as ``"***"``. Redaction runs before any JSON conversion. Safe semantic
    controls such as ``think`` and ``reasoning_max_tokens`` remain because
    gateway compilation turns them into provider payload fields.
    Messages, tool schemas and the top-level model-visible options are
    intentionally captured unredacted and may still be sensitive; so may
    provider error text in ``error`` events.

    Args:
        messages (List[Dict[str, Any]]): Normalized messages after formatting
            and any compaction, before conversion to the provider's protocol.
        cfg (Mapping[str, Any]): Request config at call time; ``tools`` rides
            separately. Containers are converted to JSON form (mappings to
            objects, tuples and other sequences to arrays, sets to sorted
            arrays), and each remaining leaf is kept when JSON-serializable,
            else replaced by its ``str()`` so the payload stays persistable.
        spec (Any): Resolved ``LLMSpec`` naming the route.
        streamable (bool | None): Effective transport mode of this call
            (streaming vs complete), recorded as ``options.stream`` when
            known. Defaults to ``None``.

    Returns:
        Dict[str, Any]: ``{"messages", "tools", "options", "route"}``.
    """
    excluded = _non_request_option_keys()
    options = {
        key: _canonical_option(key, value, redact=key not in _MODEL_VISIBLE_OPTION_KEYS)
        for key, value in cfg.items()
        if key != "tools" and key not in excluded and not callable(value)
    }
    if streamable is not None:
        options["stream"] = bool(streamable)
    route = {
        key: value
        for key, value in {
            "preset": getattr(spec, "preset", None),
            "model": spec.get("model") if hasattr(spec, "get") else None,
            "provider": getattr(spec, "provider", None),
            "gateway": getattr(spec, "gateway", None),
            "endpoint": getattr(spec, "endpoint", None),
        }.items()
        if value
    }
    return {"messages": deepcopy(list(messages)), "tools": deepcopy(list(cfg.get("tools") or [])), "options": options, "route": route}


def request_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return the canonical fingerprint of one request payload.

    The digest uses sorted-key JSON encoded as UTF-8, with lone surrogates
    escaped. Read-only mappings serialize as JSON objects at any depth.
    Object keys take their JSON string form before sorting. If
    multiple keys encode to the same name, the last value wins, as in a JSON
    round trip. Stray non-JSON values use their ``str()`` text.

    Args:
        payload (Mapping[str, Any]): Request payload from
            :func:`canonical_request`.

    Returns:
        str: Hex digest of the canonical byte form.
    """
    return sha256(_sorted_json(payload).encode("utf-8", errors="backslashreplace")).hexdigest()
