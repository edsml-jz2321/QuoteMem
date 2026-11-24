"""Observe one semantic provider outage at the exact call boundary that raised it.

A provider outage inside one session slice must settle that slice, never the
runtime. The coordinator settles a durable ``provider_error`` child only when
this observation names the same error object that reached it, so a local
rendering, comparison, persistence or authorization failure can never be
relabelled as a provider outage. Observing grants no mutation permission: the
sole Writer still reauthorizes every settlement this receipt leads to.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from quotemem.core.errors import QuoteMemError
from quotemem.core.extraction_bounds import ExtractionRefusalDetailV1
from quotemem.core.provider_diagnostics import ProviderFailureDetailV1
from quotemem.providers.adapters import ProviderError

type SemanticProviderBoundary = Literal["extract", "embedding", "source_labels"]


@dataclass
class SemanticProviderFailure:
    """Retain the one outage observed inside a single admitted child's execution.

    Attributes:
        run_id: The admitted semantic child whose execution opened this scope.
        boundary: The provider call that refused, once one actually has.
        error: The exact raised object; identity is what the caller matches.
    """

    run_id: str
    boundary: SemanticProviderBoundary | None = None
    error: QuoteMemError | None = None
    extraction_refusal: ExtractionRefusalDetailV1 | None = None
    provider_diagnostics: ProviderFailureDetailV1 | None = None
    diagnostic_error: QuoteMemError | None = None
    diagnostic_boundary: SemanticProviderBoundary | None = None

    def diagnostics_for(self, run_id: str, error: BaseException) -> ProviderFailureDetailV1 | None:
        """Return only this child's actual adapter details, including schema refusals."""
        if self.run_id == run_id and self.diagnostic_error is not None and (
            error is self.diagnostic_error
            or isinstance(error, QuoteMemError) and error.code == self.diagnostic_error.code
        ):
            return self.provider_diagnostics
        return None

    def refusal_for(self, run_id: str, error: BaseException) -> ExtractionRefusalDetailV1 | None:
        """Retain only sanitized diagnostics from this child's own extraction refusal."""
        if run_id == self.run_id and isinstance(error,
                                                QuoteMemError) and error.code == "schema_invalid":
            return self.extraction_refusal
        return None

    def call_kind_for(self, run_id: str, error: BaseException) -> SemanticProviderBoundary | None:
        """Identify only the actual call whose retained diagnostics match this failure."""
        if self.diagnostics_for(run_id, error) is not None:
            return self.diagnostic_boundary
        return None

    def settles(self, run_id: str, error: BaseException) -> SemanticProviderBoundary | None:
        """Return the boundary only for this child's own observed provider outage.

        A registered read callback replaces its failure with a content-free
        ``QuoteMemError`` that keeps the original code and nothing else
        (``StoreReadModel._consume``), so a caller outside that callback cannot
        receive the observed object. The code is then what identifies it: this
        scope covers one admitted child's own execution, and
        ``provider_unavailable`` is raised there only by the adapters.
        """
        if self.boundary is None or self.run_id != run_id or self.error is None:
            return None
        if error is self.error or (
            isinstance(error, QuoteMemError) and error.code == self.error.code
        ):
            return self.boundary
        return None


_SCOPE: ContextVar[SemanticProviderFailure | None] = ContextVar(
    "aml_semantic_provider_failure",
    default=None,
)


@contextmanager
def semantic_provider_failure_scope(run_id: str) -> Iterator[SemanticProviderFailure]:
    """Observe provider outages of one admitted child; the context ends at return."""
    observation = SemanticProviderFailure(run_id)
    token = _SCOPE.set(observation)
    try:
        yield observation
    finally:
        _SCOPE.reset(token)


def observe_semantic_provider_failure(
    boundary: SemanticProviderBoundary, error: BaseException,
) -> None:
    """Record only an actual transport outage raised by this provider boundary.

    ``ProviderError`` is constructed solely by the HTTP adapters, and the
    ``provider_unavailable`` class is raised before any response is parsed or
    persisted, so a recorded observation proves the call produced no output.
    A schema refusal retains diagnostics but keeps its separate settlement code.
    """
    observed = _SCOPE.get()
    if (
        observed is not None
        and isinstance(error, ProviderError)
    ):
        if error.code == "provider_unavailable":
            observed.boundary, observed.error = boundary, error
        observed.diagnostic_error = error
        observed.diagnostic_boundary = boundary
        observed.provider_diagnostics = (
            None if error.diagnostics is None
            else ProviderFailureDetailV1.model_validate(error.diagnostics.model_dump(mode="json"))
        )


def observe_extraction_refusal(detail: ExtractionRefusalDetailV1) -> None:
    """Capture content-free diagnostics before the read guard removes error context."""
    observed = _SCOPE.get()
    if observed is not None:
        observed.extraction_refusal = ExtractionRefusalDetailV1.model_validate(
            detail.model_dump(mode="json"),
        )
