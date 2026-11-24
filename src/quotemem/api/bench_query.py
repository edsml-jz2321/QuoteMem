"""Freeze one Search's retrieval-only query bundle under the released recall v4 policy.

The caller supplies the policy, embedding identity and trusted time from one
registered snapshot, and the shared read mode resolved from explicit ``as_of``.
Route sets and policy values come from the modules; none is defined here.
This module reads no storage and grants no access. The frozen bundle belongs
to its request. An optional service-owned cache shares only
validated successful vectors under exact provider and scope bindings.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Literal, cast

from pydantic import ValidationError

from quotemem.api.bench import SearchRequest
from quotemem.core.delivery import MemoryMode
from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import (
    IntegrityDigest,
    ScopeId,
    canonical_json,
    integrity_digest,
    scope_id,
)
from quotemem.core.inference import DEFAULT_PROVIDER_TIMEOUT_MS, default_retry_policy
from quotemem.core.query import (
    CARRIED_RESERVE_BYTES,
    MAX_QUERY_BUNDLE_BYTES,
    QUERY_OMISSION,
    AsOfV1,
    QueryBundleV1,
)
from quotemem.core.query_routes import VECTOR_ROUTES, select_query_routes
from quotemem.core.recall_policy import Route
from quotemem.core.recall_snapshot import RecallPolicySelection
from quotemem.core.time_order import freeze_query_options
from quotemem.providers.adapters import ProviderError
from quotemem.providers.contracts import EmbeddingProvider, EmbeddingRequest, EmbeddingResult

_VectorOutcome = tuple[float, ...]
_EmbeddingOutcome = tuple[tuple[str, ...], _VectorOutcome]
# One initial computation and the single stale recompute delivery allows; an
# unchanged binding reuses its exact outcome instead of calling again.
_MAX_EMBEDDING_OUTCOMES = 2

OPTIONS_VECTOR_WEIGHT = 1.0
"""Share of the answer choices' vector in the frozen query vector.

A Search that carries answer choices embeds them, joined by newlines, in the
same provider request as the query; the vector routes then rank by the unit sum
of the query's unit vector and this share of the choices' unit vector. The
lexical and entity routes already read the choices. Adding the choices' vector
places more supporting messages near the top for questions about the user's own
preferences. Shares 0.5 and 1.0 behaved about the same; the mean of separately
embedded choices did worse. A Search without choices keeps the query's own vector."""


def _bounded_query(query: str, budget: int) -> str:
    """Fit JSON string bytes, keeping 1/4 head and 3/4 tail codepoints.

    The tail usually holds the question; the head retains framing. Count JSON
    escapes as well as UTF-8, and never split a codepoint. Queries already fitting
    the budget retain their exact bytes. The marker makes information loss explicit.
    """
    if len(canonical_json(query).encode("utf-8")) <= budget:
        return query

    def reduced(keep: int) -> str:
        head = keep // 4
        tail = keep - head
        return query[:head] + QUERY_OMISSION + (query[-tail:] if tail else "")

    low, high = 0, len(query)
    while low < high:
        keep = (low + high + 1) // 2
        if len(canonical_json(reduced(keep)).encode("utf-8")) <= budget:
            low = keep
        else:
            high = keep - 1
    return reduced(low)


def search_mode(profile: Literal["product", "bench"], as_of: AsOfV1 | None) -> MemoryMode:
    """Resolve the explicit read mode of one Search; never inferred from text.

    Args:
        profile: The serving profile of the released policy.
        as_of: The request's explicit world or knowledge instant, if any.

    Returns:
        ``world`` or ``knowledge`` for an explicit instant; otherwise ``plain``
        for either serving profile.

    Raises:
        QuoteMemError: invalid_request for another profile or a malformed instant.
    """
    if profile not in ("product", "bench"):
        raise QuoteMemError("invalid_request", "Search profile is not released")
    if as_of is None:
        return "plain"
    if not isinstance(cast(object, as_of), AsOfV1):
        raise QuoteMemError("invalid_request", "Search as_of is malformed")
    return as_of.mode


class QueryVectorCache:
    """Bound successful query vectors to one provider instance and service lifetime.

    Keys are digests, never query text, grants or source content. Entries expire
    after five minutes and at most ``capacity`` vectors survive. This cache owns
    no authorization: its caller must establish a current grant before lookup.
    Concurrent misses may both call the provider; no provider runs under the lock.

    Args:
        provider: Exact provider object owned by the surrounding Search service.
        capacity: Positive maximum number of retained vectors.
        clock: Monotonic seconds for process-local expiry, not query time.
    """

    def __init__(
        self, provider: EmbeddingProvider, *, capacity: int = 128,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 128:
            raise ValueError("Query vector capacity must be between 1 and 128")
        self._provider = provider
        self._capacity = capacity
        self._clock = clock
        self._lock = Lock()
        self._bindings: OrderedDict[ScopeId, tuple[IntegrityDigest, object]] = OrderedDict()
        self._vectors: OrderedDict[
            tuple[ScopeId, IntegrityDigest], tuple[float, tuple[float, ...]],
        ] = OrderedDict()

    def embed(
        self, provider: EmbeddingProvider, request: EmbeddingRequest,
        identity: tuple[str, ...], scope: ScopeId, binding: IntegrityDigest,
    ) -> _VectorOutcome:
        """Reuse a validated success, or call and validate the exact boundary.

        Args:
            provider: Must be the original provider object.
            request: Complete request already checked by query preflight.
            identity: Complete request, release and embedding-space binding.
            scope: Exact scope's opaque identity, used for bounded invalidation.
            binding: Current authority, privacy epochs and embedding-state digest.

        Returns:
            An immutable validated vector.

        Raises:
            QuoteMemError: If the provider binding changed, the provider is
                unavailable or drifted, or its output is malformed.
        """
        if provider is not self._provider:
            raise QuoteMemError("binding_mismatch", "Query vector cache changed provider")
        key = scope, integrity_digest({"identity": list(identity), "binding": binding})
        with self._lock:
            generation = self._bindings.get(scope)
            if generation is None or generation[0] != binding:
                self._discard_scope(scope)
                generation = binding, object()
                self._bindings[scope] = generation
            self._bindings.move_to_end(scope)
            while len(self._bindings) > self._capacity:
                expired_scope, _ = self._bindings.popitem(last=False)
                self._discard_scope(expired_scope)
            cached = self._vectors.pop(key, None)
            if cached is not None and cached[0] > self._clock():
                self._vectors[key] = cached
                return cached[1]
        vector = _embedding(provider, request)
        with self._lock:
            # The admission generation also distinguishes A -> B -> A or
            # eviction/re-entry while this call ran outside the lock.
            if self._bindings.get(scope) is not generation:
                return vector
            self._vectors[key] = (self._clock() + 300.0, vector)
            self._vectors.move_to_end(key)
            while len(self._vectors) > self._capacity:
                self._vectors.popitem(last=False)
        return vector

    def _discard_scope(self, scope: ScopeId) -> None:
        """Evict the scope's old vectors while the caller holds the cache lock."""
        for key in tuple(self._vectors):
            if key[0] == scope:
                del self._vectors[key]


@dataclass(frozen=True, slots=True)
class PreparedBenchQuery:
    """One request's frozen bundle and reusable, content-free provider outcome.

    Only the owning Search may reuse this value after releasing a source guard.
    It contains the external query but no hydrated memory/source bytes. The
    single stale recompute may change query_time while preserving a matching
    provider outcome. Contention alone must keep the original frozen query_time.
    """

    _bundle_json: str = field(repr=False)
    provider_outcomes: tuple[_EmbeddingOutcome, ...] = field(repr=False)

    @property
    def bundle(self) -> QueryBundleV1:
        """Decode a detached bundle so downstream mutation cannot alter reuse."""
        return QueryBundleV1.model_validate_json(self._bundle_json)

    def embedding_for(self, identity: tuple[str, ...]) -> _VectorOutcome | None:
        """Return the immutable provider outcome only for its exact request binding."""
        for binding, outcome in self.provider_outcomes:
            if identity == binding:
                return outcome
        return None


def _strict_result(supplied: object) -> EmbeddingResult:
    if not isinstance(supplied, EmbeddingResult):
        raise QuoteMemError("schema_invalid", "Search embedding response has the wrong type")
    try:
        return EmbeddingResult.model_validate(
            supplied.model_dump(mode="python", warnings=False), strict=True
        )
    except (ValidationError, TypeError, ValueError):
        raise QuoteMemError("schema_invalid", "Search embedding response is malformed") from None


def _embedding(
    provider: EmbeddingProvider, request: EmbeddingRequest
) -> _VectorOutcome:
    """Call the bound embedding boundary; a failure is a Search error."""
    try:
        supplied = provider.embed(request)
    except ProviderError as error:
        if error.error_class == "schema_invalid":
            raise
        raise QuoteMemError("provider_unavailable",
                            "Search query embedding is unavailable") from None
    result = _strict_result(supplied)
    if len(result.vectors) != len(request.texts):
        raise QuoteMemError("schema_invalid", "Search embedding response changed cardinality")
    if (
        result.resolved_model != request.requested_model
        or result.dimensions != request.dimensions
    ):
        raise QuoteMemError("index_rebuilding", "Search embedding space drifted")
    if len(result.vectors) == 1:
        return tuple(result.vectors[0])
    query, choices = (_unit(vector) for vector in result.vectors)
    return _unit([
        own + OPTIONS_VECTOR_WEIGHT * other for own, other in zip(query, choices, strict=True)
    ])


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    """Scale a provider vector to unit length; a zero vector has no direction."""
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not norm:
        raise QuoteMemError("schema_invalid", "Search query vector is unusable")
    return tuple(value / norm for value in vector)


def _strict_request(request: object) -> SearchRequest:
    if (
        not isinstance(request, SearchRequest)
        or type(request.query) is not str or not request.query
        or type(request.user_id) is not str or not request.user_id
        or type(request.top_k) is not int
        or not 1 <= request.top_k <= 1000
        or request.as_of is not None and not isinstance(cast(object, request.as_of), AsOfV1)
        or (
            request.options is not None
            and (
                type(request.options) is not tuple
                or any(type(option) is not str for option in request.options)
            )
        )
    ):
        raise QuoteMemError("invalid_request", "Search application request is malformed")
    try:
        for text in (request.query, request.user_id, *(request.options or ())):
            text.encode("utf-8", errors="strict")
    except UnicodeError:
        raise QuoteMemError("invalid_request", "Search request is not valid UTF-8") from None
    return request


def prepare_bench_query(
    request: SearchRequest,
    *,
    policy: RecallPolicySelection,
    mode: MemoryMode,
    embedding_space: str,
    embedding_epoch: IntegrityDigest,
    query_time: int,
    embedding: EmbeddingProvider,
    previous: PreparedBenchQuery | None = None,
    vector_cache: QueryVectorCache | None = None,
    vector_isolation: IntegrityDigest | None = None,
    lexical_only: bool = False,
) -> PreparedBenchQuery:
    """Freeze one Search's recall v4 query with exactly bound vector reuse.

    Args:
        request: Strict application request, including retrieval-only options.
        policy: Validated released selection from the caller's pinned root.
        mode: The read mode resolved by ``search_mode`` for this request.
        embedding_space: That same root's effective model/canary/dimension identity.
        embedding_epoch: Its complete embedding-state token.
        query_time: Trusted initial/recompute pin time, never external input.
            A retry caused only by contention must preserve the original time.
        embedding: Actual provider boundary with a fully bound request.
        previous: This Search's earlier frozen result after a delivery retry,
            retaining at most two exact provider outcomes (one stale recompute).
        vector_cache: Optional bounded cache owned by this service and provider.
            Only successful vectors cross request boundaries, never bundles.
        vector_isolation: Required with the cache: current principal, exact scope
            and erase/consent bindings, excluding grant issuance and expiry times.
        lexical_only: Freeze the policy's routes without the vector routes and call no
            provider. Only the interactive Search profile asks for it, after its query
            embedding missed the deadline (``interactive_search``); ``recall_pure``
            accepts such a bundle only when told the same.

    Returns:
        An immutable bundle with the routes ``select_query_routes`` permits for
        the policy and mode, plus its request-owned provider outcome. Oversized
        queries retain a quarter head and three quarters tail around an omission
        marker, within the serialized bundle budget including vector space.

    Raises:
        QuoteMemError: invalid_request for input outside the released contract,
            including options that leave the actual vector no room;
            provider_unavailable (429) for an embedding outage, index_rebuilding
            (429) for embedding drift, never a silently reduced route set;
            schema_invalid for malformed or unusable provider output;
            delivery_busy after the recompute budget. No provider runs before
            preflight.
    """
    request = _strict_request(request)
    if (vector_cache is None) != (vector_isolation is None):
        raise QuoteMemError("schema_invalid", "Query vector reuse requires exact isolation")
    selected = RecallPolicySelection(
        policy.release_ref, policy.config_digest, policy.policy_snapshot_digest,
        policy.snapshot_json,
    )
    snapshot = selected.policy_snapshot()
    expected = None if request.as_of is None else request.as_of.mode
    if mode not in ("current", "plain", "world", "knowledge") or (
        mode in ("world", "knowledge")
    ) != (expected is not None) or (expected is not None and expected != mode):
        raise QuoteMemError("invalid_request", "Search mode differs from its explicit as_of")
    if request.as_of is not None and request.as_of.mode == "knowledge" and not (
        0 <= request.as_of.at <= query_time
    ):
        # The knowledge reader selects recorded revisions; a future or
        # pre-epoch recorded time has no reading (store/knowledge.py).
        raise QuoteMemError("invalid_request", "Knowledge time must be nonfuture")
    released = select_query_routes(snapshot, mode)
    active_routes: tuple[Route, ...] = tuple(
        route for route in released if route not in VECTOR_ROUTES
    ) if lexical_only else released
    if lexical_only:
        if not active_routes:
            raise QuoteMemError("provider_unavailable", "Search has no route without its embedding")
    try:
        model, canary, dimension_text = embedding_space.rsplit("@", 2)
        dimension = int(dimension_text)
    except (TypeError, ValueError):
        raise QuoteMemError("embedding_space_mismatch",
                            "Search embedding space is malformed") from None
    if model != "qwen3.7-text-embedding" or not canary or dimension not in {1024, 2048}:
        raise QuoteMemError("embedding_space_mismatch", "Search embedding space is unsupported")
    vector_required = bool(set(active_routes) & VECTOR_ROUTES)
    # Validate non-provider fields with vector routes absent; the final bundle
    # restores those routes only alongside the actual bound provider vector.
    try:
        preflight = QueryBundleV1(
            schema_version=1, query="",
            options=freeze_query_options(
                request.options, request.time_order, snapshot.schema_version,
            ),
            query_time=query_time, exclude=[], session_lru=[],
            as_of=request.as_of,
            active_routes=[route for route in active_routes if route not in VECTOR_ROUTES],
            policy_release_ref=selected.release_ref, policy_snapshot=snapshot,
            embedding_space=embedding_space, embedding_epoch=embedding_epoch,
            query_vector=None, top_k=request.top_k,
        )
    except (ValidationError, ValueError, TypeError):
        raise QuoteMemError("invalid_request",
                            "Search query exceeds its released contract") from None
    sizing = preflight.model_dump(mode="json")
    sizing["active_routes"] = list(active_routes)
    # A finite binary64 JSON number takes at most 24 bytes, plus a comma.
    # Reserve the complete vector before embedding so vector and lexical routes
    # use identical reduced text, including on retries. Options are never trimmed.
    vector_bytes = dimension * 25 + 1 - len("null") if vector_required else 0
    budget = MAX_QUERY_BUNDLE_BYTES - len(canonical_json(sizing).encode("utf-8")) + 2
    # If non-query fields alone consume the worst-case reservation, retain a
    # minimal head/tail. The actual vector may still fit; final validation remains
    # authoritative for bundles whose options + vector alone exceed 64 KiB.
    budget = max(len(canonical_json(QUERY_OMISSION).encode("utf-8")) + 24,
                 budget - vector_bytes)
    query = _bounded_query(request.query, budget)
    if query != request.query:
        # A bounded query's bundle also lists the parts its whole text carries.
        query = _bounded_query(request.query, max(
            len(canonical_json(QUERY_OMISSION).encode("utf-8")) + 24,
            budget - CARRIED_RESERVE_BYTES,
        ))
    preflight = preflight.model_copy(update={"query": query})
    # The adapter batches but does not truncate. Each UTF-8 input, the query and the
    # joined answer choices (`OPTIONS_VECTOR_WEIGHT`), is below 64 KiB as the bundle
    # holds both (also below Qwen3.7's 128,000-token input limit at one token per byte).
    choices = "\n".join(request.options or ())
    provider_request = EmbeddingRequest(
        dimensions=cast(Literal[1024, 2048], dimension),
        texts=[query, choices] if choices.strip() else [query],
        timeout_ms=DEFAULT_PROVIDER_TIMEOUT_MS, retry_policy=default_retry_policy(),
    )
    identity = (
        canonical_json(provider_request), str(selected.release_ref),
        str(selected.policy_snapshot_digest), embedding_space, str(embedding_epoch),
        request.user_id, str(vector_isolation),
    )
    vector: tuple[float, ...] | None = None
    outcomes: tuple[_EmbeddingOutcome, ...] = () if previous is None else previous.provider_outcomes
    if vector_required:
        vector = None if previous is None else previous.embedding_for(identity)
        if vector is None:
            if len(outcomes) >= _MAX_EMBEDDING_OUTCOMES:
                raise QuoteMemError("delivery_busy",
                                    "Search exhausted its provider recompute budget")
            vector = (
                vector_cache.embed(
                    embedding, provider_request, identity, scope_id("user", request.user_id),
                    integrity_digest({
                        "authority": vector_isolation,
                        "embedding_space": embedding_space, "embedding_epoch": embedding_epoch,
                    }),
                )
                if vector_cache is not None and vector_isolation is not None
                else _embedding(embedding, provider_request)
            )
            outcomes = (*outcomes, (identity, vector))
    raw = preflight.model_dump(mode="json")
    raw["active_routes"] = list(active_routes)
    raw["query_vector"] = None if vector is None else list(vector)
    try:
        frozen = QueryBundleV1.model_validate(raw)
    except ValidationError:
        # Both causes are deterministic for this request, so neither may answer
        # the retryable index_rebuilding: a retry repeats them exactly, and the
        # vector cache makes every repeat instant until the client gives up.
        if len(canonical_json(raw).encode("utf-8")) > MAX_QUERY_BUNDLE_BYTES:
            # Only the options and the actual vector are left above the minimal
            # query reserved before embedding: the request exceeds its contract.
            raise QuoteMemError(
                "invalid_request", "Search query exceeds its released contract",
            ) from None
        raise QuoteMemError("schema_invalid", "Search query vector is unusable") from None
    return PreparedBenchQuery(canonical_json(frozen), outcomes)
