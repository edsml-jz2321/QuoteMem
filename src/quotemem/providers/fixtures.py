"""Strict frozen inference fixtures keyed by full request and binding identity."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter, model_validator

from quotemem.core.errors import QuoteMemError
from quotemem.core.identity import canonical_json
from quotemem.core.inference import InferenceBindingV1, validate_output
from quotemem.providers.contracts import (
    EmbeddingRequest,
    EmbeddingResult,
    InferenceRequest,
    InferenceResult,
    OutputIdentity,
    validate_binding,
)
from quotemem.store.metadata import StrictModel


class FrozenInferenceFixture(StrictModel):
    """A complete request/identity/binding and one immutable response."""

    identity: OutputIdentity | None
    binding: InferenceBindingV1 | None
    request: InferenceRequest
    response: InferenceResult

    @model_validator(mode="after")
    def complete_identity(self) -> FrozenInferenceFixture:
        """Reject incomplete run identities, field substitution, and wrong output schemas."""
        if self.identity is None or self.binding is None:
            raise ValueError("Extract/judge fixtures require a complete Run identity and binding")
        digest = validate_binding(self.request, self.binding)
        if (
            self.identity.purpose != self.request.purpose
            or self.identity.input_digest != self.request.input_digest
            or self.identity.inference_binding_digest != digest
        ):
            raise ValueError("Fixture lookup identity does not match its frozen request")
        if self.response.schema_id != self.request.schema_id:
            raise ValueError("Fixture response changed its requested schema")
        validate_output(
            self.request.purpose,
            self.request.schema_id,
            self.response.output.model_dump(mode="json"),
        )
        return self

    def key(self) -> str:
        """Serialize all request fields, the creating Run, and the complete binding."""
        return canonical_json(self.model_dump(mode="json", exclude={"response"}))


class FrozenInferenceFixtures:
    """An explicit fixture collection with no network or input-only fallback."""

    def __init__(self, fixtures: list[FrozenInferenceFixture]) -> None:
        self._fixtures = {fixture.key(): fixture for fixture in fixtures}
        if len(self._fixtures) != len(fixtures):
            raise QuoteMemError("schema_invalid", "Duplicate frozen provider fixture identity")

    @classmethod
    def load(cls, path: Path) -> FrozenInferenceFixtures:
        """Load every fixture through its strict nested schema."""
        fixtures = TypeAdapter(list[FrozenInferenceFixture]).validate_json(path.read_bytes())
        return cls(fixtures)

    def lookup(
        self,
        request: InferenceRequest,
        *,
        identity: OutputIdentity | None,
        binding: InferenceBindingV1 | None,
    ) -> InferenceResult:
        """Return a deep copy only for exactly matching identity and request bytes."""
        key = canonical_json(
            {
                "request": request.model_dump(mode="json"),
                "identity": identity.model_dump(mode="json") if identity else None,
                "binding": binding.model_dump(mode="json") if binding else None,
            }
        )
        fixture = self._fixtures.get(key)
        if fixture is None:
            raise QuoteMemError("fixture_missing",
                                "No fixture matches the complete inference identity")
        return fixture.response.model_copy(deep=True)


class FrozenEmbeddingFixture(StrictModel):
    """A complete embedding request and explicitly synthetic frozen test response."""

    request: EmbeddingRequest
    response: EmbeddingResult

    @model_validator(mode="after")
    def matching_dimensions(self) -> FrozenEmbeddingFixture:
        """Check model, dimensions, and cardinality before any fixture is usable."""
        if self.response.dimensions != self.request.dimensions or len(self.response.vectors) != len(
            self.request.texts
        ):
            raise ValueError("Embedding fixture differs from its complete request")
        return self

    def lookup(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Reject any changed input, model, dimension, timeout, or retry policy."""
        if canonical_json(request) != canonical_json(self.request):
            raise QuoteMemError("fixture_missing",
                                "Embedding fixture requires the complete request")
        return self.response.model_copy(deep=True)
